"""
KBO Scouting AI OS — Ver 14.0 (The Pivot)
===========================================
설계: 양승수
변경점 vs Ver 13.0:
  + analyze 명령어 — raw 숫자(K/9, BB/9, IP 등) → 자동 4지표 변환 → 분석
  + Range 출력 — ERA/WHIP/K9 단일값 대신 체감 변동폭(±) 제공
  + Success Pivot — 구속 약점을 무브먼트/유인구로 상쇄하는 케이스 인식
  + AAA 디스카운트 — MiLB Level별 가중치 보정

사용:
  python3 kbo_oracle.py backtest
  python3 kbo_oracle.py scout "Eric Fedde"
  python3 kbo_oracle.py compare "Eric Fedde" "Robert Stock"
  python3 kbo_oracle.py kill "Robert Stock"
  python3 kbo_oracle.py value
  python3 kbo_oracle.py grid
  python3 kbo_oracle.py analyze --input candidate.json     # NEW
  python3 kbo_oracle.py analyze --json '{"name":"Jake Eder",...}'
"""
from __future__ import annotations
import argparse, json, math, sys, logging, os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

# Ver 15.1: 검증된 라이브러리 사용 (바퀴 재발명 제거)
import numpy as np
from sklearn.metrics import (
    confusion_matrix as sk_confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score,
)


# ============================================================
# Logging Configuration
# ============================================================
def setup_logger(name: str = "kbo_oracle", level: Optional[str] = None) -> logging.Logger:
    """프로젝트 표준 로거. 환경변수 KBO_LOG_LEVEL로 레벨 조정 가능.

    Args:
        name: 로거 이름
        level: 로그 레벨 (DEBUG/INFO/WARNING/ERROR). None이면 환경변수 또는 INFO.

    Returns:
        설정된 logging.Logger 인스턴스

    Example:
        >>> log = setup_logger()
        >>> log.info("model loaded")
    """
    log = logging.getLogger(name)
    if log.handlers:  # 이미 설정됐으면 그대로 반환
        return log

    log_level = level or os.environ.get("KBO_LOG_LEVEL", "INFO")
    log.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # 콘솔 핸들러
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    ))
    log.addHandler(handler)
    log.propagate = False
    return log


# 글로벌 로거 (모듈 전체에서 사용)
logger = setup_logger()


class Constitution:
    # Failure Score 가중치
    W_COMMAND, W_STUFF_DECAY, W_ADAPTATION = 25.0, 25.0, 20.0
    W_DURABILITY, W_MENTAL, PROVEN_BONUS = 15.0, 15.0, 20.0
    # 백테스트 가중치
    BT_STUFF, BT_COMMAND, BT_CONTEXT, BT_DURABILITY = 0.35, 0.30, 0.20, 0.15
    # 가드레일
    ZONE_FLOOR_PCT, VELO_DECLINE_MPH = 50.0, 2.0
    IL_DAYS_THRESHOLD, MIN_PITCH_MIX = 60, 3
    PROVEN_VELO_FLOOR, KBO_VELO_DROP = 94.0, 1.5
    # KBO 환경 (추정치)
    KBO_IVB_BONUS, KBO_ERA_BASELINE, MLB_ERA_BASELINE = 1.12, 4.50, 4.30
    # 라벨링
    ACE_BET_MAX, VALUE_SIGN_MAX, FILLER_MAX = 30.0, 50.0, 70.0
    # NEW: Success Pivot
    PIVOT_IVB_FLOOR, PIVOT_WHIFF_FLOOR, PIVOT_CHASE_FLOOR = 17.0, 27.0, 35.0
    PIVOT_BONUS_FS = 12.0
    # NEW: Range 출력
    RANGE_LOW_FACTOR, RANGE_HIGH_FACTOR, RANGE_FS_SCALE = 0.30, 0.55, 25.0
    # Level 디스카운트 (리그 수준 + 환경 보정)
    # MLB > NPB > AAA-IL > AAA(평균) > AAA-PCL > KBO > AA > A+ > A
    # PCL은 타고투저 끝판왕(고지대 + 작은 구장 多) → 더 큰 디스카운트
    # IL은 KBO와 환경 유사 → AAA 평균보다 후하게
    # Ver 14.1: NPB-2 추가 (FanGraphs에서 NPB 1군/2군 혼동 빈번 → 사용자 명시용)
    # NPB 2군은 KBO와 유사 수준 → 0.85
    LEVEL_DISCOUNT = {"MLB":1.00,"NPB":0.92,"NPB-2":0.85,
                      "AAA-IL":0.90,"AAA":0.88,"AAA-PCL":0.82,
                      "KBO":0.85,"AA":0.78,"A+":0.62,"A":0.50,"MiLB":0.78}

    # Ver 14.1: 불펜→선발 전환 페널티 임계값
    BULLPEN_RATIO_THRESHOLD = 0.3   # gs_ratio < 0.3 = 전문 불펜
    SWINGMAN_RATIO_THRESHOLD = 0.7  # 0.3 <= gs_ratio < 0.7 = 스윙맨
    BULLPEN_K9_PENALTY = 1.5
    BULLPEN_VELO_PENALTY = 2.0
    BULLPEN_DURABILITY_CAP = 60
    SWINGMAN_K9_PENALTY = 0.5
    SWINGMAN_VELO_PENALTY = 1.0
    SWINGMAN_DURABILITY_CAP = 75

    # NEW: PCL 환경 보정 — ERA·BABIP·FIP가 부풀려진다 (HR 多)
    # 같은 ERA 4.50이어도 PCL이면 실력은 IL의 ~4.00 수준
    PCL_ERA_INFLATION   = 0.65    # PCL ERA에서 빼야 진짜 실력
    PCL_BABIP_INFLATION = 0.020   # PCL BABIP가 .020 정도 부풀려짐
    PCL_FIP_INFLATION   = 0.40    # PCL FIP도 HR 多 → 0.40 정도 부풀려짐

    # NEW Ver 14.1: Aging Curve
    # KBO 외인은 보통 28~33세 영입. 35세부터 급격히 떨어짐.
    AGING_CURVE = {
        25: +5,  26: +4,  27: +3,  28: +2,  29: +1,
        30:  0,  31:  0,  32: -1,  33: -3,  34: -5,
        35: -8,  36: -12, 37: -16, 38: -20
    }

    # NEW Ver 14.1: 좌투 프리미엄 (KBO 환경)
    # KBO 좌타자 비율 낮음 + 좌투 절대 수 적음 → 좌투 가치 ↑
    LEFTY_BONUS_STUFF = 3.0

    # NEW Ver 14.1: Confidence Score
    # 핵심 필드 (있어야 신뢰도 베이스 확보)
    CONFIDENCE_REQUIRED_FIELDS = ["k9", "bb9", "fip", "ip", "avg_velo", "prev_league"]
    # 보너스 필드 (있으면 신뢰도 ↑)
    CONFIDENCE_BONUS_FIELDS = ["ivb_inch", "babip", "hr9", "whiff_pct"]

    # NEW Ver 14.2: 표본 크기 페널티 (베이지안 수축 개념)
    # 적은 IP에서 만든 K/9·BB/9는 극단값일 가능성 ↑ → 리그 평균 쪽으로 끌어당김
    IP_FULL_TRUST = 100.0      # 100IP+ → 풀 가중치 (shrink = 0)
    IP_PARTIAL_TRUST = 50.0    # 50~100IP → 부분 신뢰
    IP_REJECT_THRESHOLD = 30.0 # 30IP 미만 → 분석 거부 권고
    # 리그 평균 (수축 타겟)
    LEAGUE_AVG_K9 = 8.5
    LEAGUE_AVG_BB9 = 3.0
    LEAGUE_AVG_HR9 = 1.1
    LEAGUE_AVG_FIP = 4.20


@dataclass
class Pitcher:
    name: str
    stuff: float = 0.0
    command: float = 0.0
    context: float = 0.0
    durability: float = 0.0
    actual_war: Optional[float] = None
    type_label: Optional[str] = None
    age: Optional[int] = None
    throws: Optional[str] = None
    last_league: str = "MLB"
    avg_velo: float = 0.0
    peak_velo: float = 0.0
    velo_trend: float = 0.0
    # Ver 15.2.1: 휴리스틱 시절 하드코딩 기본값 제거.
    # 입력 안 한 값을 가짜로 채우면 ML 엔진이 거짓 데이터로 평가함 (기본값 오염).
    # 휴리스틱 함수들은 자체 fallback(else 22.0 등)을 가지므로 None이어도 안전.
    zone_pct: Optional[float] = None
    whiff_pct: Optional[float] = None
    chase_pct: Optional[float] = None
    ivb_inch: Optional[float] = None
    pitch_types: list[str] = field(default_factory=list)
    primary_pitch_pct: Optional[float] = None
    il_days_2y: int = 0
    kbo_verified: bool = False
    kbo_avg_velo: float = 0.0
    mental_concern: float = 0.0
    contract_usd: Optional[int] = None

    # NEW: 외부 검증 데이터용 필드 (raw 스탯 기반)
    babip: Optional[float] = None      # 운/불운 지표 (.300 ± 0.030)
    hr9: Optional[float] = None        # HR/9
    fip: Optional[float] = None        # 수비독립평균자책
    prev_ip: Optional[float] = None    # 직전 리그 IP
    prev_league: Optional[str] = None  # 직전 리그 (MLB/AAA-PCL/AAA-IL/NPB/NPB-2)
    debut_year: Optional[int] = None   # KBO 데뷔년도
    data_source: str = "synthetic"     # synthetic / verified / gemini_unverified

    # Ver 14.1: G/GS (불펜→선발 전환 페널티)
    g: Optional[int] = None            # 직전 시즌 총 등판 경기 수
    gs: Optional[int] = None           # 직전 시즌 선발 등판 수
    role_profile: Optional[str] = None # "STARTER" / "SWINGMAN" / "RELIEVER" (자동 판정)
    raw_k9: Optional[float] = None     # 페널티 전 원본 K/9 (감사용)
    raw_bb9: Optional[float] = None    # 페널티 전 원본 BB/9 (Ver 15.2)
    raw_velo: Optional[float] = None   # 페널티 전 원본 평속 (감사용)

    # === Ver 15.2: 결측 Indicator (Missing as Risk Signal) ===
    # 결측 자체를 ML 모델 입력 신호로 사용. 평균 fallback 대신
    # "이 정보가 없다는 사실"을 명시적으로 모델에 전달.
    is_k9_missing: bool = False        # K/9 원본 결측 여부
    is_bb9_missing: bool = False       # BB/9 원본 결측 여부
    pitch_mix_missing: bool = False    # 구종 정보 결측 여부

    # === Ver 14.4: 신규 변수 5종 (r 0.50+ 목표) ===
    # FanGraphs Plate Discipline 탭에서 수집
    csw_pct: Optional[float] = None    # Called Strike + Whiff % (구위 진짜 신호)
    swstr_pct: Optional[float] = None  # Swinging Strike % (FanGraphs)
    gb_pct: Optional[float] = None     # Ground Ball % (KBO 환경 유리 신호)
    xfip: Optional[float] = None       # Expected FIP

    # FanGraphs 2년 누적 (단년 노이즈 제거)
    prev_2y_ip: Optional[float] = None  # 직전 2년 누적 이닝
    prev_2y_era: Optional[float] = None # 직전 2년 ERA
    prev_2y_k9: Optional[float] = None  # 직전 2년 K/9

    # === 재계약 외인 KBO 1년차 데이터 (RecontractModel용) ===
    # Statiz에서 수집 (평속/IL 제외 — 수집 어려움)
    kbo_y1_ip: Optional[float] = None      # 1년차 IP
    kbo_y1_era: Optional[float] = None     # 1년차 ERA
    kbo_y1_fip: Optional[float] = None     # 1년차 FIP
    kbo_y1_k9: Optional[float] = None      # 1년차 K/9
    kbo_y1_bb9: Optional[float] = None     # 1년차 BB/9
    kbo_y1_war: Optional[float] = None     # 1년차 WAR
    is_recontract: bool = False            # 재계약 외인 플래그


def load_dataset(path: Path) -> list[Pitcher]:
    """JSON 데이터셋 파일에서 Pitcher 객체 리스트 로드.

    Args:
        path: 데이터셋 파일 경로 (.json)

    Returns:
        Pitcher 객체 리스트

    Raises:
        FileNotFoundError: 파일이 존재하지 않을 때
        json.JSONDecodeError: 잘못된 JSON 형식

    Example:
        >>> from pathlib import Path
        >>> dataset = load_dataset(Path("dataset_external.json"))
        >>> len(dataset) > 0
        True
    """
    logger.debug(f"Loading dataset from {path}")
    if not path.exists():
        logger.error(f"Dataset not found: {path}")
        raise FileNotFoundError(f"Dataset file not found: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    valid_fields = set(Pitcher.__dataclass_fields__.keys())
    cleaned = []
    for p in raw:
        c = {k: v for k, v in p.items() if k in valid_fields}
        # raw 스탯만 있으면 4지표 자동 변환
        if c.get("stuff", 0) == 0 and "k9" in p:
            pp = build_pitcher_from_raw(p)
            for fld in ["actual_war", "type_label", "data_source"]:
                if fld in p: setattr(pp, fld, p[fld])
            cleaned.append(pp)
        else:
            cleaned.append(Pitcher(**c))
    logger.info(f"Loaded {len(cleaned)} pitchers from {path.name}")
    return cleaned


def load_combined_dataset(paths: list[Path]) -> list[Pitcher]:
    """여러 데이터셋 합치기 (기존 + 외부 검증)"""
    combined = []
    for p in paths:
        if p.exists():
            combined.extend(load_dataset(p))
    return combined


class StatTranslator:
    """raw 통계 → 4지표 자동 변환 (Ver 14.0+)
       PCL/IL 구분 보정 + BABIP 운/불운 보정 적용"""

    @staticmethod
    def _normalize_level(level: str) -> str:
        """리그명 정규화 — 'AAA-PCL', 'NPB-2' 등 변형 처리"""
        if not level: return "MiLB"
        L = level.upper().replace(" ", "").replace("_", "-")
        # NPB 2군 명시 케이스 (NPB-2, NPB2, NPB-MINOR 등)
        if "NPB" in L and ("2" in L or "MINOR" in L or "FARM" in L):
            return "NPB-2"
        if "NPB" in L: return "NPB"
        if "PCL" in L: return "AAA-PCL"
        if "IL" in L and "MIL" not in L: return "AAA-IL"
        return level if level in Constitution.LEVEL_DISCOUNT else "MiLB"

    @staticmethod
    def adjust_era_for_league(era: float, level: str) -> float:
        """리그 환경 보정 — PCL ERA는 부풀려져 있음"""
        norm = StatTranslator._normalize_level(level)
        if norm == "AAA-PCL":
            return max(0.0, era - Constitution.PCL_ERA_INFLATION)
        return era

    @staticmethod
    def adjust_babip_for_league(babip: float, level: str) -> float:
        """리그 환경 보정 — PCL BABIP도 부풀려짐"""
        norm = StatTranslator._normalize_level(level)
        if norm == "AAA-PCL":
            return max(0.0, babip - Constitution.PCL_BABIP_INFLATION)
        return babip

    @staticmethod
    def luck_factor_from_babip(babip: Optional[float]) -> float:
        """BABIP 운/불운 보정 계수.
           정상 BABIP는 .300. .280 이하 = 행운, .320 이상 = 불운.
           반환값:
             - 1.0 = 중립
             - <1.0 = 행운(낮은 BABIP) → Stuff 곱셈 깎임 (실력 부풀려진 것)
             - >1.0 = 불운(높은 BABIP) → Stuff 곱셈 가산 (실력은 더 좋았음)
           Stuff 산출 시 곱셈으로 적용됨."""
        if babip is None: return 1.0
        deviation = babip - 0.300
        if abs(deviation) <= 0.020: return 1.0
        # 부호: '+' 가 정답
        # BABIP 0.250 (행운) → dev -0.050 → 0.925 (깎임) ✅
        # BABIP 0.350 (불운) → dev +0.050 → 1.075 (가산) ✅
        return 1.0 + (deviation * 1.5)

    @staticmethod
    def k9_to_stuff(k9: float, level: str = "AAA",
                    babip: Optional[float] = None,
                    ivb_inch: Optional[float] = None,
                    throws: Optional[str] = None,
                    age: Optional[int] = None,
                    bb9: Optional[float] = None) -> float:
        """K/9 → Stuff. Level 디스카운트 + BABIP + IVB + 좌투 + aging.
           bb9는 시그니처 호환성 유지용 (현재 미사용)"""
        norm = StatTranslator._normalize_level(level)
        base = 50 + (k9 - 7.0) * 5
        discount = Constitution.LEVEL_DISCOUNT.get(norm, 0.78)
        luck = StatTranslator.luck_factor_from_babip(babip)
        stuff = base * discount * luck + 10

        # IVB 보너스 (있을 때만)
        if ivb_inch is not None and ivb_inch > 0:
            if ivb_inch >= 18.0:   stuff += 6
            elif ivb_inch >= 16.0: stuff += 3
            elif ivb_inch < 14.0:  stuff -= 3

        # 좌투 프리미엄
        stuff += StatTranslator.lefty_bonus(throws)

        # Aging adjustment (절반만 반영)
        stuff += StatTranslator.aging_adjustment(age) * 0.5

        # NOTE Ver 14.3: K-BB% 시너지는 실증 검증 실패로 제거됨 (n=26에서 r 하락)

        return max(40, min(95, stuff))

    @staticmethod
    def bb9_to_command(bb9: float, level: str = "AAA") -> float:
        """BB/9 → Command. 제구는 리그 영향 적음 (절대 능력)"""
        return max(45, min(95, 100 - (bb9 - 1.5) * 8))

    @staticmethod
    def derive_context(level: str, fip: Optional[float] = None,
                       era: Optional[float] = None, babip: Optional[float] = None,
                       hr9: Optional[float] = None) -> float:
        """리그 수준 + FIP/ERA + BABIP 운 보정.
           hr9는 시그니처 호환성 유지용 (현재 미사용, 실증 검증 실패)"""
        norm = StatTranslator._normalize_level(level)
        base = {"MLB":90,"NPB":82,"NPB-2":75,"AAA-IL":80,"AAA":78,"AAA-PCL":74,
                "KBO":75,"AA":70,"A+":58,"A":48,"MiLB":68}.get(norm, 65)
        # FIP는 BABIP/리그 환경 영향 적은 지표 → 신뢰 가중치 ↑
        if fip is not None:
            adj_fip = fip
            if norm == "AAA-PCL":
                adj_fip = max(0.0, fip - Constitution.PCL_FIP_INFLATION)
            if adj_fip < 3.5: base += 7
            elif adj_fip < 4.0: base += 3
            elif adj_fip > 5.5: base -= 7
            elif adj_fip > 5.0: base -= 3
        # ERA는 리그 보정 후 평가
        if era is not None:
            adj_era = StatTranslator.adjust_era_for_league(era, level)
            if adj_era < 3.0: base += 3
            elif adj_era > 5.5: base -= 5
        # BABIP 운 보정
        if babip is not None:
            adj_babip = StatTranslator.adjust_babip_for_league(babip, level)
            if adj_babip < 0.270: base -= 3
            elif adj_babip > 0.330: base += 3
        # NOTE Ver 14.3: HR/9 페널티는 실증 검증 실패로 제거됨 (n=26에서 r 하락)
        return max(40, min(95, base))

    @staticmethod
    def derive_durability(ip: float, il_days: int = 0, age: Optional[int] = None) -> float:
        """이닝 + IL + 나이(aging curve) 기반"""
        s = 60
        if ip >= 150: s += 15
        elif ip >= 120: s += 10
        elif ip >= 100: s += 5
        elif ip < 60: s -= 15
        s -= il_days / 5
        # Aging Curve 적용 (Durability에 직접 반영)
        s += StatTranslator.aging_adjustment(age)
        return max(30, min(95, s))

    @staticmethod
    def apply_role_penalty(g: Optional[int], gs: Optional[int],
                            k9: Optional[float], avg_velo: Optional[float]):
        """Ver 14.1: 불펜→선발 전환 페널티.
           직전 시즌 G/GS 비율로 role 판정 후 K/9·평속 차감.
           반환: (adjusted_k9, adjusted_velo, role_profile, durability_cap)"""
        c = Constitution
        if not g or g <= 0 or gs is None:
            return k9, avg_velo, "UNKNOWN", None

        gs_ratio = gs / g
        adj_k9 = k9
        adj_velo = avg_velo
        cap = None
        profile = "STARTER"

        if gs_ratio < c.BULLPEN_RATIO_THRESHOLD:
            # 전문 불펜 → 선발 전환 시 K/9·구속 다 떨어짐
            profile = "RELIEVER"
            cap = c.BULLPEN_DURABILITY_CAP
            if k9 is not None: adj_k9 = k9 - c.BULLPEN_K9_PENALTY
            if avg_velo is not None: adj_velo = avg_velo - c.BULLPEN_VELO_PENALTY
        elif gs_ratio < c.SWINGMAN_RATIO_THRESHOLD:
            # 스윙맨 → 페널티 일부
            profile = "SWINGMAN"
            cap = c.SWINGMAN_DURABILITY_CAP
            if k9 is not None: adj_k9 = k9 - c.SWINGMAN_K9_PENALTY
            if avg_velo is not None: adj_velo = avg_velo - c.SWINGMAN_VELO_PENALTY
        # else: STARTER → 페널티 없음

        return adj_k9, adj_velo, profile, cap

    @staticmethod
    def sample_size_shrink(value: float, league_avg: float, ip: Optional[float]) -> float:
        """Ver 14.2: 표본 크기 기반 베이지안 수축.
           적은 IP에서 만든 raw 스탯을 리그 평균 쪽으로 끌어당김."""
        if ip is None or ip <= 0: return value
        c = Constitution
        if ip >= c.IP_FULL_TRUST:
            return value
        elif ip >= c.IP_PARTIAL_TRUST:
            weight = 0.5 + 0.5 * (ip - c.IP_PARTIAL_TRUST) / (c.IP_FULL_TRUST - c.IP_PARTIAL_TRUST)
        elif ip >= c.IP_REJECT_THRESHOLD:
            weight = 0.3 + 0.2 * (ip - c.IP_REJECT_THRESHOLD) / (c.IP_PARTIAL_TRUST - c.IP_REJECT_THRESHOLD)
        else:
            weight = 0.2
        return weight * value + (1 - weight) * league_avg

    @staticmethod
    def is_sample_too_small(ip: Optional[float]) -> bool:
        """IP가 분석 거부 임계 미만인가?"""
        return ip is not None and ip < Constitution.IP_REJECT_THRESHOLD

    @staticmethod
    def aging_adjustment(age: Optional[int]) -> float:
        """나이에 따른 점수 조정 (KBO 외인 aging curve)."""
        if age is None: return 0.0
        curve = Constitution.AGING_CURVE
        if age in curve: return curve[age]
        if age < min(curve): return curve[min(curve)]
        return curve[max(curve)]

    @staticmethod
    def lefty_bonus(throws: Optional[str]) -> float:
        """좌투 프리미엄 (KBO 환경)"""
        if throws and throws.upper() == "L":
            return Constitution.LEFTY_BONUS_STUFF
        return 0.0

    @staticmethod
    def confidence_score(data: dict) -> dict:
        """입력 데이터의 신뢰도 산출.
           필수 필드 만점 60 + 보너스 필드 만점 40 = 100점"""
        c = Constitution
        req_present = sum(1 for f in c.CONFIDENCE_REQUIRED_FIELDS
                          if data.get(f) not in (None, "", 0))
        bonus_present = sum(1 for f in c.CONFIDENCE_BONUS_FIELDS
                            if data.get(f) not in (None, "", 0))
        req_score = (req_present / len(c.CONFIDENCE_REQUIRED_FIELDS)) * 60
        bonus_score = (bonus_present / len(c.CONFIDENCE_BONUS_FIELDS)) * 40
        total = req_score + bonus_score
        missing_required = [f for f in c.CONFIDENCE_REQUIRED_FIELDS
                            if data.get(f) in (None, "", 0)]
        return {
            "total": round(total, 1),
            "required_present": req_present,
            "required_total": len(c.CONFIDENCE_REQUIRED_FIELDS),
            "bonus_present": bonus_present,
            "bonus_total": len(c.CONFIDENCE_BONUS_FIELDS),
            "missing_required": missing_required,
            "grade": ("HIGH" if total >= 80 else
                      "MEDIUM" if total >= 60 else
                      "LOW" if total >= 40 else "VERY LOW"),
        }


class BacktestEngine:
    def __init__(self, weights: Optional[dict] = None):
        c = Constitution; w = weights or {}
        self.w_s = w.get("stuff", c.BT_STUFF)
        self.w_c = w.get("command", c.BT_COMMAND)
        self.w_x = w.get("context", c.BT_CONTEXT)
        self.w_d = w.get("durability", c.BT_DURABILITY)

    def score(self, p):
        return p.stuff*self.w_s + p.command*self.w_c + p.context*self.w_x + p.durability*self.w_d

    @staticmethod
    def pearson(xs, ys):
        """Pearson 상관계수. numpy.corrcoef 사용.

        직접 구현 대신 검증된 라이브러리 사용 (Ver 15.1 리팩토링).
        """
        if len(xs) < 2: return 0.0
        try:
            import numpy as np
            arr = np.corrcoef(xs, ys)
            r = arr[0, 1]
            return 0.0 if np.isnan(r) else float(r)
        except Exception:
            return 0.0

    def spearman(self, xs, ys):
        """Spearman 순위 상관계수. scipy.stats.spearmanr 사용.

        직접 구현 대신 검증된 라이브러리 사용 (Ver 15.1 리팩토링).
        """
        if len(xs) < 2: return 0.0
        try:
            from scipy.stats import spearmanr
            r, _ = spearmanr(xs, ys)
            import numpy as np
            return 0.0 if np.isnan(r) else float(r)
        except Exception:
            return 0.0

    def run(self, dataset):
        """Ver 15.2: actual_war이 None인 신규 투수 자동 제외.

        Args:
            dataset: Pitcher 리스트 (신규 투수 actual_war=None 가능)

        Returns:
            dict: 회귀 검증 결과
        """
        # Ver 15.2 항목 7: None 필터링 — 신규 영입 후보(actual_war=None) 자동 제외
        valid_dataset = [p for p in dataset if p.actual_war is not None]
        if len(valid_dataset) == 0:
            logger.warning("BacktestEngine.run: actual_war 보유 선수 없음")
            return {"n": 0, "pearson_r": 0.0, "spearman_r": 0.0,
                    "r_squared": 0.0, "slope": 0.0, "intercept": 0.0,
                    "group_means": {}, "separation": 0.0, "rows": [],
                    "outliers": [],
                    "weights_used":{"stuff":self.w_s,"command":self.w_c,
                                    "context":self.w_x,"durability":self.w_d}}

        if len(valid_dataset) < len(dataset):
            n_skipped = len(dataset) - len(valid_dataset)
            logger.info(f"BacktestEngine.run: 신규 투수 {n_skipped}명 제외 "
                       f"(actual_war=None), 유효 데이터 {len(valid_dataset)}명")

        scores = [self.score(p) for p in valid_dataset]
        wars = [p.actual_war for p in valid_dataset]
        r = self.pearson(scores, wars); rs = self.spearman(scores, wars)
        n = len(scores); mx, my = sum(scores)/n, sum(wars)/n
        sxx = sum((s-mx)**2 for s in scores)
        sxy = sum((scores[i]-mx)*(wars[i]-my) for i in range(n))
        slope = sxy/sxx if sxx > 0 else 0
        intercept = my - slope*mx
        rows = []
        for i, p in enumerate(valid_dataset):
            pred = slope*scores[i] + intercept
            rows.append({"name":p.name,"type":p.type_label,"score":round(scores[i],2),
                         "actual_war":p.actual_war,"predicted_war":round(pred,2),
                         "residual":round(p.actual_war - pred,2)})
        groups = {}
        for p, s in zip(valid_dataset, scores):
            groups.setdefault(p.type_label or "Unknown", []).append(s)
        gm = {k: round(sum(v)/len(v),2) for k,v in groups.items()}
        sep = gm.get("Success",0) - gm.get("Failure",0)
        outliers = sorted(rows, key=lambda r: abs(r["residual"]), reverse=True)[:5]
        return {"n":n,"pearson_r":round(r,4),"spearman_r":round(rs,4),
                "r_squared":round(r*r,4),"slope":round(slope,4),
                "intercept":round(intercept,4),"group_means":gm,
                "separation":round(sep,2),"rows":rows,"outliers":outliers,
                "weights_used":{"stuff":self.w_s,"command":self.w_c,
                                "context":self.w_x,"durability":self.w_d}}


# =============================================================
# RecontractModel — 재계약 외인 전용 예측 모델 (Ver 14.4 NEW)
# =============================================================
class RecontractModel:
    """재계약 외인 KBO 2년차 WAR 예측 모델.

    영입 모델과 분리된 별도 클래스. 입력은 KBO 1년차 데이터 (5개 지표) +
    raw 스탯 일부 (보조). KBO 1년차 실제 성적이 가장 강력한 신호이므로
    재계약 결정 시 더 정확한 예측 가능.

    가중치 근거:
      - kbo_y1_war: 가장 직접적 신호 (50% 가중)
      - kbo_y1_fip: 운과 무관한 능력 (20%)
      - kbo_y1_k9 / bb9: 구위와 제구 안정성 (20%)
      - kbo_y1_ip: 풀시즌 소화력 (10%)

    Example:
        >>> p = Pitcher(name="Kelly", kbo_y1_war=4.2, kbo_y1_ip=161,
        ...             kbo_y1_era=4.13, kbo_y1_fip=3.65,
        ...             kbo_y1_k9=7.2, kbo_y1_bb9=2.8)
        >>> RecontractModel().predict(p)
        4.5  # 예측 WAR (2년차)
    """

    # 1년차 지표 가중치 (합 1.0)
    W_Y1_WAR = 0.50
    W_Y1_FIP = 0.20
    W_Y1_K9  = 0.12
    W_Y1_BB9 = 0.08
    W_Y1_IP  = 0.10

    # 회귀 상수 (2년차 일반적 감소 패턴 반영)
    # KBO 외인 2년차는 평균 WAR이 1년차 대비 ~85% 수준 (regression to mean)
    Y2_DECAY_FACTOR = 0.85
    Y2_BASE_OFFSET = 0.3

    @staticmethod
    def has_required_data(p) -> bool:
        """재계약 모델 적용 가능한지 검사 (KBO 1년차 5개 핵심 지표)"""
        required = [p.kbo_y1_war, p.kbo_y1_ip, p.kbo_y1_fip,
                    p.kbo_y1_k9, p.kbo_y1_bb9]
        return all(v is not None for v in required)

    def predict(self, p) -> Optional[float]:
        """재계약 외인의 KBO 2년차 WAR 예측.

        Args:
            p: Pitcher — 1년차 KBO 데이터 필수

        Returns:
            float: 예측 WAR (2년차). 데이터 부족 시 None.
        """
        if not self.has_required_data(p):
            logger.warning(f"{p.name}: KBO 1년차 데이터 부족 — 재계약 모델 적용 불가")
            return None

        # 1) WAR 직접 가중 (회귀 평균 적용)
        war_score = p.kbo_y1_war * self.Y2_DECAY_FACTOR

        # 2) FIP 보정 (4.0 기준)
        # FIP 낮을수록 보너스, 높을수록 페널티
        fip_adj = (4.0 - p.kbo_y1_fip) * 0.5  # FIP 3.0 → +0.5 WAR

        # 3) K/9 - BB/9 시너지 (5.0 기준)
        k_minus_bb = p.kbo_y1_k9 - p.kbo_y1_bb9
        synergy_adj = (k_minus_bb - 5.0) * 0.15  # K-BB% 7.0이면 +0.3

        # 4) IP 보정 (150 기준 — 풀시즌)
        ip_adj = ((p.kbo_y1_ip or 0) - 150) / 50  # 200IP면 +1.0

        # 5) Raw 스탯 보조 (있을 때만, 가중치 작게)
        raw_bonus = 0
        if p.csw_pct and p.csw_pct >= 28: raw_bonus += 0.2
        if p.gb_pct and p.gb_pct >= 45: raw_bonus += 0.1

        # 6) 종합
        predicted = (
            war_score * self.W_Y1_WAR / self.W_Y1_WAR  # 1년차 WAR (정규화)
            + fip_adj * self.W_Y1_FIP * 2
            + synergy_adj * (self.W_Y1_K9 + self.W_Y1_BB9) * 2
            + ip_adj * self.W_Y1_IP * 1.5
            + raw_bonus
        )
        return round(predicted, 2)

    def evaluate(self, dataset: list) -> dict:
        """재계약 외인 데이터셋 평가.

        is_recontract=True인 선수만 필터링해서 예측 vs 실제 비교.

        Returns:
            dict: pearson_r, mae, predictions
        """
        recontracts = [p for p in dataset if p.is_recontract]
        if len(recontracts) < 3:
            logger.warning(f"재계약 외인 데이터 부족 (n={len(recontracts)}). 최소 3명 필요.")
            return {"error": "insufficient_data", "n": len(recontracts)}

        predictions = []
        actuals = []
        rows = []
        for p in recontracts:
            pred = self.predict(p)
            if pred is None: continue
            if p.actual_war is None: continue
            predictions.append(pred)
            actuals.append(p.actual_war)
            rows.append({
                "name": p.name,
                "kbo_y1_war": p.kbo_y1_war,
                "predicted_y2_war": pred,
                "actual_y2_war": p.actual_war,
                "residual": round(p.actual_war - pred, 2),
            })

        if len(predictions) < 2:
            return {"error": "insufficient_valid", "n": len(predictions)}

        r = BacktestEngine.pearson(predictions, actuals)
        mae = sum(abs(p-a) for p,a in zip(predictions, actuals)) / len(predictions)
        rmse = math.sqrt(sum((p-a)**2 for p,a in zip(predictions, actuals)) / len(predictions))
        return {
            "n": len(predictions),
            "pearson_r": round(r, 4),
            "mae": round(mae, 2),
            "rmse": round(rmse, 2),
            "rows": rows,
        }


class ScoutEngine:

    def guardrails(self, p):
        """Ver 15.2: None/0 안전 처리"""
        c = Constitution
        # Fallback: None/0이면 평균값 사용
        velo_trend = p.velo_trend if p.velo_trend is not None else 0.0
        zone_pct = p.zone_pct if (p.zone_pct and p.zone_pct > 0) else 50.0
        kbo_velo = p.kbo_avg_velo if p.kbo_avg_velo else 0.0
        il_days = p.il_days_2y if p.il_days_2y is not None else 0
        pitch_count = len(p.pitch_types) if p.pitch_types else 0

        velo_ok = velo_trend >= -c.VELO_DECLINE_MPH
        if p.kbo_verified and kbo_velo >= c.PROVEN_VELO_FLOOR:
            velo_ok = True
        return {"command": zone_pct >= c.ZONE_FLOOR_PCT,
                "velocity": velo_ok,
                "durability": il_days < c.IL_DAYS_THRESHOLD,
                "pitch_mix": pitch_count >= c.MIN_PITCH_MIX}

    def check_success_pivot(self, p):
        """Ver 14.0: 구속 약점을 무브먼트/유인구로 상쇄 가능?
        Ver 15.2: None/0 안전 처리"""
        c = Constitution; triggers = []
        # Fallback
        ivb = p.ivb_inch if (p.ivb_inch and p.ivb_inch > 0) else 15.0
        whiff = p.whiff_pct if (p.whiff_pct and p.whiff_pct > 0) else 22.0
        chase = p.chase_pct if (p.chase_pct and p.chase_pct > 0) else 30.0
        velo_trend = p.velo_trend if p.velo_trend is not None else 0.0
        avg_velo = p.avg_velo if (p.avg_velo and p.avg_velo > 0) else 92.0

        if ivb >= c.PIVOT_IVB_FLOOR:
            triggers.append(f"IVB {ivb:.1f}\" ≥ {c.PIVOT_IVB_FLOOR}")
        if whiff >= c.PIVOT_WHIFF_FLOOR:
            triggers.append(f"Whiff% {whiff:.1f} ≥ {c.PIVOT_WHIFF_FLOOR}")
        if chase >= c.PIVOT_CHASE_FLOOR:
            triggers.append(f"Chase% {chase:.1f} ≥ {c.PIVOT_CHASE_FLOOR}")
        weakness = (velo_trend < -1.0) or (avg_velo < 92.0)
        activated = len(triggers) >= 2 and weakness
        return {"activated": activated, "triggers": triggers,
                "weakness_detected": weakness,
                "bonus_applied": c.PIVOT_BONUS_FS if activated else 0.0}

    def failure_score(self, p):
        """Ver 15.2: None/0 안전 처리"""
        c = Constitution
        # Fallback: 0/None은 평균값
        zone_pct = p.zone_pct if (p.zone_pct and p.zone_pct > 0) else 50.0
        velo_trend = p.velo_trend if p.velo_trend is not None else 0.0
        whiff = p.whiff_pct if (p.whiff_pct and p.whiff_pct > 0) else 22.0
        pitch_count = len(p.pitch_types) if p.pitch_types else 0
        primary = p.primary_pitch_pct if (p.primary_pitch_pct and p.primary_pitch_pct > 0) else 50.0
        il_days = p.il_days_2y if p.il_days_2y is not None else 0
        mental = p.mental_concern if p.mental_concern is not None else 0

        cmd_r = max(0, min(25, (55 - zone_pct) * 2.5))
        velo_r = max(0, -velo_trend * 5)
        whiff_r = max(0, (25 - whiff) * 0.5)
        stuff_r = min(25, velo_r + whiff_r)
        mix_r = max(0, (3 - pitch_count) * 5)
        dep_r = max(0, (primary - 40) * 0.25)
        adapt_r = min(20, mix_r + dep_r)
        dur_r = min(15, il_days / 10)
        ment_r = min(15, mental)
        bonus = c.PROVEN_BONUS if (p.kbo_verified and p.kbo_avg_velo >= c.PROVEN_VELO_FLOOR) else 0
        pivot = self.check_success_pivot(p)
        pivot_bonus = pivot["bonus_applied"]
        total = max(0, min(100, cmd_r + stuff_r + adapt_r + dur_r + ment_r - bonus - pivot_bonus))
        return {"command_risk":round(cmd_r,1),"stuff_decay_risk":round(stuff_r,1),
                "adaptation_risk":round(adapt_r,1),"durability_risk":round(dur_r,1),
                "mental_risk":round(ment_r,1),"proven_bonus":bonus,
                "pivot_bonus":pivot_bonus,"pivot_info":pivot,
                "total":round(total,1)}

    def time_decay(self, p):
        """Ver 15.2: None/0 안전 처리 추가"""
        s = 50
        # 결측 fallback: 구종은 평균 3개, IVB는 평균 15, primary_pitch_pct는 50
        pitch_count = len(p.pitch_types) if p.pitch_types else 3
        ivb = p.ivb_inch if (p.ivb_inch and p.ivb_inch > 0) else 15.0
        primary = p.primary_pitch_pct if (p.primary_pitch_pct and p.primary_pitch_pct > 0) else 50.0

        if pitch_count >= 4: s -= 15
        elif pitch_count <= 2: s += 25
        if ivb >= 18: s -= 10
        elif ivb < 14: s += 10
        if p.kbo_verified: s -= 15
        if primary > 55: s += 10
        return max(0, min(100, s))

    def season_sim(self, p, decay, fs_total, residuals=None):
        """Ver 14.0: Range 출력. residuals 전달 시 부트스트랩 기반 진짜 분포 사용.
        Ver 15.2: None/0 안전 처리 추가 (avg_velo, whiff_pct, ivb_inch fallback)
        """
        c = Constitution
        # 결측 fallback: None 또는 0이면 평균값 사용
        avg_velo = p.avg_velo if (p.avg_velo and p.avg_velo > 0) else 92.0
        whiff_pct = p.whiff_pct if (p.whiff_pct and p.whiff_pct > 0) else 22.0
        ivb_inch = p.ivb_inch if (p.ivb_inch and p.ivb_inch > 0) else 15.0

        stuff_q = (avg_velo - 90) * 0.3
        stuff_q += (whiff_pct - 22) * 0.05
        stuff_q += (ivb_inch - 15) * 0.05
        base = max(2.50, 4.20 - stuff_q)
        d = decay / 100
        def whip(e): return round(e/4.5*1.30, 2)
        def k9(e): return round(max(5.0, 7.5 + (whiff_pct-20)*0.15 - (e-base)*0.5), 1)
        pa = base - 0.50
        pb = base + 0.20 + d*0.60
        pc = base + 0.40 + d*1.20
        avg = (pa+pb+pc)/3

        # Range 산출: residuals 있으면 부트스트랩, 없으면 휴리스틱
        if residuals and len(residuals) >= 10:
            # 회귀 잔차 분포에서 5–95 백분위 → 비대칭 그대로 유지
            sr = sorted(residuals)
            low_off = abs(sr[int(len(sr)*0.05)])     # 5%분위 (best 쪽)
            high_off = abs(sr[int(len(sr)*0.95)])    # 95%분위 (worst 쪽)
            # WAR 잔차 → ERA로 환산 (WAR 1.0 ≈ ERA 0.5 가정)
            low_off *= 0.5
            high_off *= 0.5
            range_source = "bootstrap (회귀 잔차 5–95%)"
        else:
            scale = math.sqrt(max(fs_total, 5) / c.RANGE_FS_SCALE)
            low_off = c.RANGE_LOW_FACTOR * scale
            high_off = c.RANGE_HIGH_FACTOR * scale
            range_source = "heuristic (FS √분산)"

        def rng(e): return {"best":round(e-low_off,2),"expected":round(e,2),
                            "worst":round(e+high_off,2)}
        return {"phase_a":{"label":"4~5월 Honeymoon","era":rng(pa),"whip":whip(pa),"k9":k9(pa)},
                "phase_b":{"label":"6~7월 Adaptation","era":rng(pb),"whip":whip(pb),"k9":k9(pb)},
                "phase_c":{"label":"8~10월 Endgame","era":rng(pc),"whip":whip(pc),"k9":k9(pc)},
                "season":{"era":rng(avg),"whip":whip(avg),
                          "ip_low":round(160-d*40-15,0),"ip_expected":round(160-d*40,0),
                          "ip_high":round(160-d*40+10,0)},
                "range_source": range_source}


def assign_label(fs_total, gr, decay, pivot_activated=False):
    c = Constitution
    all_pass = all(gr.values())
    if not all_pass and not pivot_activated:
        return "AVOID"
    if not all_pass and pivot_activated:
        return "VALUE SIGN (Pivot)" if fs_total <= c.VALUE_SIGN_MAX else "AVOID"
    if fs_total <= c.ACE_BET_MAX and decay <= 40: return "ACE BET"
    if fs_total <= c.VALUE_SIGN_MAX: return "VALUE SIGN"
    if fs_total <= c.FILLER_MAX: return "ROTATION FILLER"
    return "AVOID"


def contract_for(label):
    base = label.replace(" (Pivot)", "")
    return {"ACE BET":1_000_000,"VALUE SIGN":700_000,
            "ROTATION FILLER":400_000,"AVOID":0}[base]


def render_scout(p, gr, fs, decay, sim, label, contract):
    """Ver 15.1: renderers 모듈로 위임 (SoC 분리)"""
    from renderers import render_scout_report
    return render_scout_report(p, gr, fs, decay, sim, label, contract)


def render_backtest(result):
    """Ver 15.1: renderers 모듈로 위임 (SoC 분리)"""
    from renderers import render_backtest_report
    return render_backtest_report(result)


def build_pitcher_from_raw(data: dict) -> Pitcher:
    """Ver 14.4: raw 통계 자동 변환.

    처리 순서:
       1) 메타 필드 복사 (기존 + 신규 5종 + 재계약 7종)
       2) G/GS 기반 role 페널티 (K/9·평속 사전 조정)
       3) 조정된 raw → 4지표 변환

    Args:
        data: dict — raw 스탯 입력 (k9, bb9, fip 등)

    Returns:
        Pitcher: 4지표 자동 계산된 인스턴스
    """
    p = Pitcher(name=data.get("name", "Unknown"))
    # Ver 15.2 항목 1: JSON null 덮어쓰기 방어 (None이 아닌 값만 setattr)
    for k in ["age","throws","last_league","contract_usd","actual_war","type_label",
              "prev_league","prev_ip","debut_year","fip","hr9","babip","data_source",
              "g","gs",
              # Ver 14.4 / 15.0 신규 변수
              "csw_pct", "swstr_pct", "gb_pct", "xfip",
              "prev_2y_ip", "prev_2y_era", "prev_2y_k9",
              # 재계약 외인 KBO 1년차
              "kbo_y1_ip", "kbo_y1_era", "kbo_y1_fip",
              "kbo_y1_k9", "kbo_y1_bb9", "kbo_y1_war", "is_recontract"]:
        if k in data and data[k] is not None:
            setattr(p, k, data[k])
    for k in ["avg_velo","peak_velo","velo_trend","zone_pct","whiff_pct",
              "chase_pct","ivb_inch","primary_pitch_pct","il_days_2y",
              "kbo_verified","kbo_avg_velo","mental_concern"]:
        if k in data and data[k] is not None:
            setattr(p, k, data[k])

    # Ver 15.2 항목 3-변형: pitch_types 결측을 indicator로 표시 (자동 채움 X)
    raw_pitch_types = data.get("pitch_types")
    if raw_pitch_types and len(raw_pitch_types) > 0:
        p.pitch_types = raw_pitch_types
        p.pitch_mix_missing = False
    else:
        # 결측을 명시 — 데이터 조작하지 않음
        p.pitch_types = []
        p.pitch_mix_missing = True

    level = data.get("prev_league") or data.get("last_league", "AAA")
    babip = data.get("babip")
    era = data.get("era")
    fip = data.get("fip")
    ip = data.get("ip") or data.get("prev_ip")
    if ip is not None and p.prev_ip is None:
        p.prev_ip = ip

    # Ver 15.2 항목 2-변형: K/9, BB/9 결측을 indicator로 표시 (평균 채움 X)
    raw_k9 = data.get("k9")
    p.is_k9_missing = (raw_k9 is None)
    raw_velo = data.get("avg_velo")

    # === Ver 14.1: G/GS 페널티 (k9_to_stuff 호출 전에 처리) ===
    g = data.get("g")
    gs = data.get("gs")
    adj_k9, adj_velo, profile, dur_cap = StatTranslator.apply_role_penalty(
        g, gs, raw_k9, raw_velo
    )
    p.role_profile = profile
    p.raw_k9 = raw_k9       # 원본 보존 (감사용)
    p.raw_velo = raw_velo
    if adj_velo is not None:
        p.avg_velo = adj_velo    # 페널티 적용된 값을 Pitcher에 저장

    # === Ver 14.2: 표본 크기 베이지안 수축 (Role 페널티 다음) ===
    # 적은 IP에서 만든 raw 스탯은 리그 평균 쪽으로 끌어당김
    c = Constitution
    raw_bb9 = data.get("bb9")
    p.is_bb9_missing = (raw_bb9 is None)   # Ver 15.2: BB/9 결측 indicator
    p.raw_bb9 = raw_bb9   # Ver 15.2: 원본 BB/9 보존 (Risk Engine 정확도용)
    shrunk_k9 = StatTranslator.sample_size_shrink(adj_k9, c.LEAGUE_AVG_K9, ip) if adj_k9 is not None else None
    shrunk_bb9 = StatTranslator.sample_size_shrink(raw_bb9, c.LEAGUE_AVG_BB9, ip) if raw_bb9 is not None else None

    tr = StatTranslator
    if "stuff" in data:
        p.stuff = data["stuff"]
    elif shrunk_k9 is not None:
        # Ver 14.3: 수축된 K/9 + BB/9 시너지
        p.stuff = tr.k9_to_stuff(
            shrunk_k9, level, babip,
            data.get("ivb_inch"),
            data.get("throws"),
            data.get("age"),
            shrunk_bb9,  # K-BB% 시너지용
        )

    if "command" in data: p.command = data["command"]
    elif shrunk_bb9 is not None: p.command = tr.bb9_to_command(shrunk_bb9, level)

    if "context" in data: p.context = data["context"]
    else: p.context = tr.derive_context(level, fip, era, babip, data.get("hr9"))

    if "durability" in data:
        p.durability = data["durability"]
    elif ip is not None:
        p.durability = tr.derive_durability(ip, data.get("il_days_2y", 0), data.get("age"))

    # === Ver 14.1: Durability Cap 적용 (불펜/스윙맨) ===
    if dur_cap is not None:
        p.durability = min(p.durability, dur_cap)

    return p


def parse_v2_text(text: str) -> dict:
    """Ver 14.1 V2 양식 파서.
    포맷: '선수명 / 데뷔년도 / 직전리그 / G / GS / IP / K9 / BB9 / HR9 / FIP / 평속 / IVB / BABIP → WAR'

    빈칸·하이픈·'-' 은 None으로 처리.
    리그 코드: MLB / NPB / NPB-2 / AAA-PCL / AAA-IL / AAA / AA / A+ / A

    예시:
      폰세 / 2025 / NPB / 67 / 11 / 67.0 / 7.52 / 2.15 / 0.81 / 3.13 / 94.0 / - / - → 8.38

    반환: build_pitcher_from_raw에 바로 넣을 수 있는 dict.
    """
    if "→" in text:
        main, war_part = text.split("→", 1)
        actual_war = _parse_float(war_part.strip())
    else:
        main, actual_war = text, None

    parts = [s.strip() for s in main.split("/")]
    if len(parts) < 7:
        raise ValueError(f"V2 양식 필드 부족 (최소 7개 필요, 받음 {len(parts)}): {text}")

    # 정해진 순서대로 매핑 (없는 필드는 None)
    # Ver 15.2: 16변수 입력 규격 — gb_pct, xfip, csw_pct 추가
    keys = ["name", "debut_year", "prev_league", "g", "gs", "ip",
            "k9", "bb9", "hr9", "fip", "avg_velo", "ivb_inch", "babip",
            "gb_pct", "xfip", "csw_pct"]
    data = {}
    for i, key in enumerate(keys):
        val = parts[i].strip() if i < len(parts) else None
        if val in (None, "", "-", "—", "N/A", "n/a", "null"):
            continue
        if key == "name":
            data[key] = val
        elif key == "prev_league":
            data[key] = val.upper()
        elif key in ("debut_year", "g", "gs"):
            try: data[key] = int(val)
            except ValueError: pass
        else:
            f = _parse_float(val)
            if f is not None: data[key] = f

    if actual_war is not None:
        data["actual_war"] = actual_war
    data["data_source"] = "verified_user"
    return data


def _parse_float(s):
    """안전 float 파싱 — 빈 문자열, '-', None 등 처리"""
    if s is None: return None
    s = str(s).strip().replace(",", ".")  # 쉼표→점 자동 보정
    if s in ("", "-", "—", "N/A", "n/a", "null"): return None
    try: return float(s)
    except ValueError: return None


# =============================================================
# Render — Presentation Layer (Ver 15.1)
# =============================================================
class Render:
    """UI 문자열 / 마크다운 / 콘솔 출력 전담 클래스.

    Ver 15.1 SoC 리팩토링:
    - 백엔드 로직(데이터 처리, ML)과 프레젠테이션 로직(렌더링) 분리
    - 모든 render_* 함수를 이 클래스의 staticmethod로 위임
    - 기존 함수는 deprecated이지만 호환성을 위해 유지

    Usage:
        md = Render.scout_report(p, gr, fs, decay, sim, label, contract)
        scatter = Render.ascii_scatter(rows)
        msg = Render.one_liner(label, fs_total, decay, pivot)
    """

    @staticmethod
    def scout_report(p, gr, fs, decay, sim, label, contract) -> str:
        """Scout 마크다운 리포트 (기존 render_scout 위임)"""
        return render_scout(p, gr, fs, decay, sim, label, contract)

    @staticmethod
    def backtest_report(result: dict) -> str:
        """Backtest 마크다운 리포트"""
        return render_backtest(result)

    @staticmethod
    def classify_md(rows, cm, loo, threshold, war_cutoff, n) -> str:
        """Classify 마크다운 리포트"""
        return _render_classify_md(rows, cm, loo, threshold, war_cutoff, n)

    @staticmethod
    def validation_md(base, loo, boot, tts, honest_r) -> str:
        """Validation 마크다운 리포트.

        Ver 15.1: pivot ablation 인자 제거.
        """
        return _render_validation_md(base, loo, boot, tts, honest_r)

    @staticmethod
    def ascii_scatter(rows, w: int = 60, h: int = 18) -> str:
        """ASCII 산점도"""
        return _ascii_scatter(rows, w, h)

    @staticmethod
    def one_liner(label: str, fs_total: float, decay: float, pivot: bool) -> str:
        """라벨별 한 줄 멘트"""
        return _one_liner(label, fs_total, decay, pivot)


def cmd_analyze(args, dataset, out_dir):
    if getattr(args, 'text', None):
        data = parse_v2_text(args.text)
    elif args.json:
        data = json.loads(args.json)
    elif args.input:
        data = json.loads(Path(args.input).read_text(encoding='utf-8'))
    else:
        print("❌ --text, --json, --input 중 하나는 필요"); return

    # Confidence Score 사전 계산
    conf = StatTranslator.confidence_score(data)

    p = build_pitcher_from_raw(data)
    print(f"\n📊 입력 데이터 신뢰도: {conf['total']:.0f}/100 ({conf['grade']})")
    print(f"  · 필수 필드: {conf['required_present']}/{conf['required_total']}")
    print(f"  · 보너스 필드: {conf['bonus_present']}/{conf['bonus_total']}")
    if conf['missing_required']:
        print(f"  ⚠️ 결측 필수: {', '.join(conf['missing_required'])}")

    # Ver 14.1: Role 페널티 결과 표시
    if p.role_profile and p.role_profile not in ("UNKNOWN", "STARTER"):
        print(f"  ⚠️ Role: {p.role_profile} (G={p.g}, GS={p.gs}) — 페널티 적용")
        if p.raw_k9 is not None:
            print(f"     원본 K/9 {p.raw_k9:.2f} → 페널티 후 Stuff 계산")
        if p.raw_velo is not None and abs((p.raw_velo or 0) - (p.avg_velo or 0)) > 0.01:
            print(f"     원본 평속 {p.raw_velo:.1f}mph → 적용 {p.avg_velo:.1f}mph")
    elif p.role_profile == "STARTER":
        print(f"  ✅ Role: STARTER (G={p.g}, GS={p.gs})")

    print(f"\n🔬 자동 변환: Stuff {p.stuff:.1f} / Command {p.command:.1f} / "
          f"Context {p.context:.1f} / Durability {p.durability:.1f}\n")
    eng = ScoutEngine()
    gr = eng.guardrails(p); fs = eng.failure_score(p)
    decay = eng.time_decay(p)
    # NEW: 잔차 부트스트랩
    bt_result = BacktestEngine().run(dataset)
    residuals = [r['residual'] for r in bt_result['rows']]
    sim = eng.season_sim(p, decay, fs['total'], residuals=residuals)
    pivot = fs['pivot_info']['activated']
    label = assign_label(fs['total'], gr, decay, pivot)
    contract = contract_for(label)
    md = Render.scout_report(p, gr, fs, decay, sim, label, contract)
    print(md)
    path = out_dir / f"analyze_{p.name.replace(' ','_')}.md"
    path.write_text(md, encoding='utf-8')
    print(f"\n💾 저장: {path}")


def _find(dataset, name):
    target = next((p for p in dataset if p.name.lower() == name.lower()), None)
    if not target:
        print(f"❌ '{name}' not found"); sys.exit(1)
    return target


def cmd_scout(args, dataset, out_dir):
    p = _find(dataset, args.name)
    eng = ScoutEngine()
    gr = eng.guardrails(p); fs = eng.failure_score(p)
    decay = eng.time_decay(p)
    # NEW: 데이터셋에서 회귀 잔차 추출 → Range 부트스트랩
    bt_result = BacktestEngine().run(dataset)
    residuals = [r['residual'] for r in bt_result['rows']]
    sim = eng.season_sim(p, decay, fs['total'], residuals=residuals)
    pivot = fs['pivot_info']['activated']
    label = assign_label(fs['total'], gr, decay, pivot)
    contract = contract_for(label)
    md = Render.scout_report(p, gr, fs, decay, sim, label, contract)
    print(md)
    (out_dir / f"scout_{p.name.replace(' ','_')}.md").write_text(md, encoding='utf-8')


def cmd_backtest(args, dataset, out_dir):
    result = BacktestEngine().run(dataset)
    md = Render.backtest_report(result)
    print(md)
    (out_dir / f"backtest_n{result['n']}.md").write_text(md, encoding='utf-8')


def cmd_value(args, dataset, out_dir):
    eng = ScoutEngine(); cands = []
    for p in dataset:
        gr = eng.guardrails(p); fs = eng.failure_score(p)
        decay = eng.time_decay(p); pivot = fs['pivot_info']['activated']
        label = assign_label(fs['total'], gr, decay, pivot)
        if "VALUE SIGN" in label:
            cands.append((p, fs['total'], decay, label))
    cands.sort(key=lambda x: x[1])
    print(f"\n💎 VALUE SIGN 후보 ({len(cands)}명)\n")
    print(f"{'Player':<22} {'FS':>5} {'Decay':>5} {'Label':>22}")
    print("-"*60)
    for p, fs, d, lb in cands:
        print(f"{p.name:<22} {fs:>5.1f} {d:>5} {lb:>22}")


def cmd_compare(args, dataset, out_dir):
    a = _find(dataset, args.a); b = _find(dataset, args.b)
    eng = ScoutEngine()
    def stats(p):
        gr = eng.guardrails(p); fs = eng.failure_score(p)
        decay = eng.time_decay(p); sim = eng.season_sim(p, decay, fs['total'])
        pivot = fs['pivot_info']['activated']
        return gr, fs, decay, sim, assign_label(fs['total'], gr, decay, pivot)
    ga,fa,da,sa,la = stats(a); gb,fb,db,sb,lb = stats(b)
    print(f"\n⚔️  {a.name}  vs  {b.name}\n")
    print(f"{'Item':<20} {a.name:>18} {b.name:>18}")
    print("-"*58)
    print(f"{'Label':<20} {la:>18} {lb:>18}")
    print(f"{'Failure Score':<20} {fa['total']:>18.1f} {fb['total']:>18.1f}")
    print(f"{'Time Decay':<20} {da:>18} {db:>18}")
    print(f"{'Season ERA':<20} {sa['season']['era']['expected']:>18.2f} {sb['season']['era']['expected']:>18.2f}")
    pa = '⚡' if fa['pivot_info']['activated'] else '-'
    pb = '⚡' if fb['pivot_info']['activated'] else '-'
    print(f"{'Pivot':<20} {pa:>18} {pb:>18}")





# =============================================================
# VALIDATE — 한계 극복 모듈 (Ver 14.0+)
# =============================================================
import random

def loocv_r(dataset: list[Pitcher]) -> dict:
    """Leave-One-Out CV: 각 선수를 hold-out하고 나머지로 예측 → 진짜 r"""
    n = len(dataset)
    eng = BacktestEngine()
    preds = []; actuals = []
    abs_errors = []
    for hold_idx in range(n):
        train = [p for i, p in enumerate(dataset) if i != hold_idx]
        test = dataset[hold_idx]
        # 훈련셋으로 회귀 학습
        scores = [eng.score(p) for p in train]
        wars = [p.actual_war for p in train]
        mx, my = sum(scores)/len(scores), sum(wars)/len(wars)
        sxx = sum((s-mx)**2 for s in scores)
        sxy = sum((scores[i]-mx)*(wars[i]-my) for i in range(len(train)))
        slope = sxy/sxx if sxx > 0 else 0
        intercept = my - slope*mx
        # 홀드아웃 예측
        test_score = eng.score(test)
        pred = slope*test_score + intercept
        preds.append(pred)
        actuals.append(test.actual_war)
        abs_errors.append(abs(pred - test.actual_war))
    r_loocv = BacktestEngine.pearson(preds, actuals)
    mae = sum(abs_errors) / n
    rmse = math.sqrt(sum(e**2 for e in abs_errors) / n)
    return {"r_loocv": r_loocv, "mae": mae, "rmse": rmse, "n": n}


def bootstrap_r_ci(dataset: list[Pitcher], iters: int = 2000) -> dict:
    """Bootstrap: r값의 95% 신뢰구간 → 진짜 불확실성 표시"""
    eng = BacktestEngine()
    scores = [eng.score(p) for p in dataset]
    wars = [p.actual_war for p in dataset]
    n = len(scores)
    rs = []
    rng = random.Random(42)  # 재현성
    for _ in range(iters):
        idx = [rng.randrange(n) for _ in range(n)]
        xs = [scores[i] for i in idx]
        ys = [wars[i] for i in idx]
        if len(set(xs)) < 2: continue
        rs.append(BacktestEngine.pearson(xs, ys))
    rs.sort()
    if not rs: return {"error": "bootstrap failed"}
    lo = rs[int(len(rs)*0.025)]
    hi = rs[int(len(rs)*0.975)]
    med = rs[len(rs)//2]
    return {"r_median": med, "ci_lo": lo, "ci_hi": hi, "iters": len(rs)}


def train_test_split_r(dataset: list[Pitcher], n_splits: int = 50,
                        test_frac: float = 0.3) -> dict:
    """50회 랜덤 분할 → 평균 test r → 일반화 성능"""
    rng = random.Random(42)
    n = len(dataset); n_test = max(3, int(n * test_frac))
    eng = BacktestEngine()
    test_rs = []
    for _ in range(n_splits):
        idx = list(range(n)); rng.shuffle(idx)
        test_idx = set(idx[:n_test])
        train = [dataset[i] for i in range(n) if i not in test_idx]
        test = [dataset[i] for i in test_idx]
        # 훈련셋 회귀
        ts = [eng.score(p) for p in train]
        tw = [p.actual_war for p in train]
        mx, my = sum(ts)/len(ts), sum(tw)/len(tw)
        sxx = sum((s-mx)**2 for s in ts)
        sxy = sum((ts[i]-mx)*(tw[i]-my) for i in range(len(train)))
        slope = sxy/sxx if sxx > 0 else 0
        intercept = my - slope*mx
        # 테스트셋 예측
        preds = [slope*eng.score(p) + intercept for p in test]
        actuals = [p.actual_war for p in test]
        if len(set(preds)) < 2: continue
        test_rs.append(BacktestEngine.pearson(preds, actuals))
    if not test_rs: return {"error": "split failed"}
    test_rs.sort()
    return {
        "mean_test_r": sum(test_rs)/len(test_rs),
        "median_test_r": test_rs[len(test_rs)//2],
        "p5": test_rs[max(0, int(len(test_rs)*0.05))],
        "p95": test_rs[min(len(test_rs)-1, int(len(test_rs)*0.95))],
        "n_splits": len(test_rs),
    }


# [REMOVED in Ver 15.1] pivot_ablation:
#   Pivot Bonus 휴리스틱이 Ver 15.0 Risk Engine 도입과 함께 제거됨.
#   합성 데이터 기반 ablation은 ML 모델 검증에 부적합.


def cmd_validate(args, dataset, out_dir):
    """4가지 정직성 검증을 한 번에"""
    print("\n" + "="*60)
    print("🧪 VALIDATION SUITE — 한계 극복 검증")
    print("="*60)

    # 0. Baseline (in-sample r)
    base = BacktestEngine().run(dataset)
    print(f"\n[0] Baseline (in-sample, 부풀려진 값)")
    print(f"    r = {base['pearson_r']:+.4f}  (참고용)")

    # 1. LOOCV — 한계 #1 (r 부풀려짐) 극복
    print(f"\n[1] LOOCV — 진짜 일반화 r (한계 #1)")
    loo = loocv_r(dataset)
    delta = loo['r_loocv'] - base['pearson_r']
    print(f"    r_LOOCV = {loo['r_loocv']:+.4f}  (Δ vs baseline: {delta:+.4f})")
    print(f"    MAE = {loo['mae']:.2f} WAR  /  RMSE = {loo['rmse']:.2f} WAR")
    print(f"    해석: in-sample 대비 {abs(delta):.4f} 손실 → "
          f"{'❌ 과적합 의심' if delta < -0.10 else '✅ 견고함' if abs(delta) < 0.03 else '⚠️ 약한 손실'}")

    # 2. Bootstrap CI — 한계 #5 (n=30) 극복
    print(f"\n[2] Bootstrap — r의 95% 신뢰구간 (한계 #5)")
    boot = bootstrap_r_ci(dataset, iters=args.bootstrap_iters)
    print(f"    r 중앙값 = {boot['r_median']:+.4f}")
    print(f"    95% CI = [{boot['ci_lo']:+.4f}, {boot['ci_hi']:+.4f}]")
    width = boot['ci_hi'] - boot['ci_lo']
    print(f"    CI 폭 = {width:.4f}")
    print(f"    해석: {'✅ 안정 (폭<0.1)' if width<0.1 else '⚠️ 중간 (0.1~0.2)' if width<0.2 else '❌ 불안정 (>0.2) — n 부족'}")

    # 3. Train/Test Split — 한계 #1+#5 보강
    print(f"\n[3] Train/Test Split — 50회 랜덤 분할 (한계 #1+#5)")
    tts = train_test_split_r(dataset)
    print(f"    평균 test r = {tts['mean_test_r']:+.4f}")
    print(f"    중앙값      = {tts['median_test_r']:+.4f}")
    print(f"    p5–p95     = [{tts['p5']:+.4f}, {tts['p95']:+.4f}]")
    drop = base['pearson_r'] - tts['mean_test_r']
    print(f"    in-sample 대비 손실: −{drop:.4f}")

    # [4] Pivot Ablation — Ver 15.1에서 제거 (Pivot 휴리스틱이 ML로 대체됨)
    # Risk Engine 검증은 'risk-validate' 명령어로 별도 수행
    print(f"\n[4] Pivot Ablation — [REMOVED] Ver 15.0 Risk Engine으로 대체")
    print(f"    → 'python3 kbo_oracle.py risk-validate' 사용")

    # KBO 트래킹 미보유 한계 (한계 #2) — 코드로 극복 불가
    print(f"\n[5] KBO 트래킹 비공개 (한계 #2)")
    print(f"    ❌ 코드로 극복 불가능. 다음과 같이 정직하게 표기:")
    print(f"    · IVB +12% 보정은 일반론적 추정")
    print(f"    · KBO 타자 컨택률·존 크기 보정 미반영")
    print(f"    · Range·시즌 시뮬은 MLB 기반 추정 외삽")

    # Range 휴리스틱 한계 (한계 #3) — 부분 극복 가능 메모
    print(f"\n[6] Range 휴리스틱 (한계 #3)")
    print(f"    ⚠️ 부분 극복 가능 — Bootstrap WAR sampling으로 진짜 분포")
    print(f"    현재 구현: FS √분산 비대칭 (휴리스틱)")
    print(f"    향후 구현: 회귀 잔차 부트스트랩 → 진짜 신뢰구간")

    # 종합 정직성 점수
    print(f"\n" + "="*60)
    print(f"📊 종합 모델 카드 (Model Card)")
    print(f"="*60)
    honest_r = (loo['r_loocv'] + tts['mean_test_r']) / 2
    print(f"  · 보고 가능 r (정직):  {honest_r:+.4f}")
    print(f"  · 95% CI:              [{boot['ci_lo']:+.4f}, {boot['ci_hi']:+.4f}]")
    print(f"  · 평균 절대 오차:      {loo['mae']:.2f} WAR")
    print(f"  · 미해결 한계:         KBO 트래킹 비공개, Range 부트스트랩 미적용")

    # 마크다운 저장
    md = Render.validation_md(base, loo, boot, tts, honest_r)
    path = out_dir / "validation_report.md"
    path.write_text(md, encoding='utf-8')
    print(f"\n💾 저장: {path}")


def _render_validation_md(base, loo, boot, tts, honest_r) -> str:
    """검증 리포트 마크다운.

    Ver 15.1: pivot ablation 섹션 제거 (Pivot 휴리스틱이 Ver 15.0 Risk Engine으로 대체됨).
    """
    return f"""# 🧪 Validation Report — KBO Scouting AI OS

> 생성: {datetime.now().strftime('%Y-%m-%d %H:%M')}
> 통계적 정직성 검증 결과 (회귀 모델 — 휴리스틱 기반)
> **Note**: ML 기반 검증은 `risk-validate` 명령어 별도 수행

## 📊 모델 카드 (Model Card)

| 지표 | 값 | 의미 |
|---|---|---|
| **r (정직, LOOCV+Split 평균)** | **{honest_r:+.4f}** | 새 선수 예측 시 기대 상관 |
| r (in-sample, 참고용) | {base['pearson_r']:+.4f} | 같은 데이터 회귀 — 부풀려진 값 |
| 95% CI (Bootstrap) | [{boot['ci_lo']:+.4f}, {boot['ci_hi']:+.4f}] | r의 통계적 불확실성 |
| MAE | {loo['mae']:.2f} WAR | 평균 절대 오차 |
| RMSE | {loo['rmse']:.2f} WAR | 큰 오차에 페널티 |

## 🔬 검증 1: LOOCV (한계 #1 — r 부풀려짐)
각 선수를 hold-out하고 나머지로 회귀 학습 → 빠진 1명 예측.

- **r_LOOCV = {loo['r_loocv']:+.4f}**
- Baseline 대비 Δ = {loo['r_loocv'] - base['pearson_r']:+.4f}
- {'✅ 견고함 (손실 <0.03)' if abs(loo['r_loocv'] - base['pearson_r']) < 0.03 else '⚠️ 약한 과적합' if loo['r_loocv'] < base['pearson_r'] else '✅ 손실 없음'}

## 🔬 검증 2: Bootstrap CI (한계 #5 — n=30 작음)
{boot['iters']}회 리샘플링 → r의 95% 신뢰구간.

- **95% CI = [{boot['ci_lo']:+.4f}, {boot['ci_hi']:+.4f}]**
- CI 폭 = {boot['ci_hi'] - boot['ci_lo']:.4f}
- {'✅ 안정' if (boot['ci_hi']-boot['ci_lo'])<0.1 else '⚠️ 중간' if (boot['ci_hi']-boot['ci_lo'])<0.2 else '❌ 불안정'}

## 🔬 검증 3: Train/Test Split (한계 #1+#5)
50회 랜덤 70:30 분할 → 평균 test r.

- 평균 test r = **{tts['mean_test_r']:+.4f}**
- p5–p95 = [{tts['p5']:+.4f}, {tts['p95']:+.4f}]
- in-sample 대비 손실: −{base['pearson_r'] - tts['mean_test_r']:.4f}

## ❌ 미해결 한계

| 한계 | 상태 | 메모 |
|---|---|---|
| KBO 트래킹 데이터 비공개 | 극복 불가 | IVB +12% 등은 추정치. 라이선스 필요 |
| Range 부트스트랩 미적용 | 부분 극복 가능 | 회귀 잔차 부트스트랩으로 향후 강화 가능 |
| 회귀 모델 자체 한계 | 부분 극복 | Ver 15.0 Risk Engine (ML)로 분류 평가 권장 |

## 🎤 발표 시 정직 멘트

> "in-sample r은 0.97로 매우 높게 나오지만, LOOCV와 Train/Test Split으로
> 검증한 정직한 r은 약 **{honest_r:.2f}**이며 95% CI는 **[{boot['ci_lo']:.2f}, {boot['ci_hi']:.2f}]** 입니다.
> 회귀로는 약하지만 **Ver 15.0 Risk Engine** (ML 분류)로 평가하면
> Brier Score 기반 calibrated probability 출력이 가능합니다."
"""


# =============================================================
# CLASSIFY — 분류 모델 평가 (Ver 14.3 NEW)
# =============================================================
def classify_pitcher(p, bt_eng: BacktestEngine, threshold: float = 72.0) -> str:
    """현재 모델의 4지표 가중 합산 점수로 Success/Failure 이진 분류.
       BacktestEngine.score()를 직접 사용 (Stuff*0.35 + Cmd*0.30 + Ctx*0.20 + Dur*0.15).
       임계값 72.0은 26명 데이터 그룹 평균 분석에서 도출:
         Success 평균 73.1 vs Failure 평균 71.8 → 중간 72.5 부근.
       단 임계값은 데이터 보고 정한 거라 LOOCV에서 검증 필요."""
    score = bt_eng.score(p)
    return "Success" if score >= threshold else "Failure"


def war_to_class(war: float, cutoff: float = 3.0) -> str:
    """실제 WAR을 분류 라벨로 변환.
       WAR >= 3.0 → Success (KBO 외인 보통 기대치)"""
    return "Success" if war >= cutoff else "Failure"


def confusion_matrix(actuals: list, predicteds: list) -> dict:
    """이진 분류 혼동행렬 + 평가지표.

    Ver 15.1: sklearn.metrics 사용 (바퀴 재발명 제거).
    actuals, predicteds는 ["Success", "Failure"] 문자열 라벨 리스트.
    """
    n = len(actuals)
    if n == 0:
        return {"tp": 0, "tn": 0, "fp": 0, "fn": 0, "n": 0,
                "accuracy": 0, "precision": 0, "recall": 0, "specificity": 0,
                "f1": 0, "baseline_acc": 0, "improvement_over_baseline": 0}

    # labels 순서 명시: [negative, positive] → [Failure, Success]
    # ravel: [[tn, fp], [fn, tp]] → tn, fp, fn, tp 순서
    cm = sk_confusion_matrix(actuals, predicteds, labels=["Failure", "Success"])
    tn, fp, fn, tp = cm.ravel()

    accuracy = accuracy_score(actuals, predicteds)
    precision = precision_score(actuals, predicteds,
                                pos_label="Success", zero_division=0)
    recall = recall_score(actuals, predicteds,
                          pos_label="Success", zero_division=0)
    f1 = f1_score(actuals, predicteds, pos_label="Success", zero_division=0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    # 베이스라인: 다수 클래스로 다 찍었을 때 정확도
    success_count = sum(1 for a in actuals if a == "Success")
    failure_count = n - success_count
    baseline_acc = max(success_count, failure_count) / n

    return {
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn), "n": n,
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1": float(f1),
        "baseline_acc": float(baseline_acc),
        "improvement_over_baseline": float(accuracy - baseline_acc),
    }


def loocv_classify(dataset: list, war_cutoff: float = 3.0) -> dict:
    """LOOCV 분류 정확도 — 각 선수 hold-out, 나머지에서 최적 임계값 추정 후 예측.
       이 방식이 진짜 분류 성능 (임계값 데이터 누수 차단)."""
    bt_eng = BacktestEngine()
    actuals = []
    predicteds = []
    for hold_idx in range(len(dataset)):
        test = dataset[hold_idx]
        if test.actual_war is None: continue
        train = [p for i, p in enumerate(dataset) if i != hold_idx]
        # train에서 Success/Failure 평균 점수의 중간값을 임계값으로
        s_scores = [bt_eng.score(p) for p in train if p.actual_war >= war_cutoff]
        f_scores = [bt_eng.score(p) for p in train if p.actual_war < war_cutoff]
        if not s_scores or not f_scores:
            thr = 72.0  # fallback
        else:
            thr = (sum(s_scores)/len(s_scores) + sum(f_scores)/len(f_scores)) / 2
        # hold-out 예측
        test_score = bt_eng.score(test)
        pred = "Success" if test_score >= thr else "Failure"
        actual = "Success" if test.actual_war >= war_cutoff else "Failure"
        actuals.append(actual)
        predicteds.append(pred)
    return confusion_matrix(actuals, predicteds)


def cmd_classify(args, dataset, out_dir):
    """분류 모델 평가 — Success/Failure 이진 분류 정확도 측정.

    임계값 자동 결정 시 데이터 누수 방지를 위해 LOOCV에서는 hold-out
    제외한 나머지로 임계값을 다시 학습한다.

    Args:
        args: argparse 결과 (threshold, war_cutoff, quick)
        dataset: Pitcher 객체 리스트
        out_dir: 리포트 저장 경로
    """
    logger.info(f"Starting classification evaluation (n={len(dataset)})")
    bt_eng = BacktestEngine()
    war_cutoff = args.war_cutoff if hasattr(args, 'war_cutoff') and args.war_cutoff else 3.0

    # 임계값 자동 결정: Success/Failure 그룹 평균의 중간값
    valid = [p for p in dataset if p.actual_war is not None]
    s_scores = [bt_eng.score(p) for p in valid if p.actual_war >= war_cutoff]
    f_scores = [bt_eng.score(p) for p in valid if p.actual_war < war_cutoff]
    if hasattr(args, 'threshold') and args.threshold:
        threshold = args.threshold
        threshold_auto = False
    elif s_scores and f_scores:
        threshold = (sum(s_scores)/len(s_scores) + sum(f_scores)/len(f_scores)) / 2
        threshold_auto = True
    else:
        threshold = 72.0
        threshold_auto = False

    # 분류 실행
    rows = []
    actuals = []
    predicteds = []
    for p in valid:
        score = bt_eng.score(p)
        pred = "Success" if score >= threshold else "Failure"
        actual = "Success" if p.actual_war >= war_cutoff else "Failure"
        correct = pred == actual
        rows.append({"name": p.name, "score": score, "pred": pred,
                     "war": p.actual_war, "actual": actual, "correct": correct})
        actuals.append(actual); predicteds.append(pred)

    cm = confusion_matrix(actuals, predicteds)
    loo = loocv_classify(dataset, war_cutoff)

    # ============ QUICK MODE ============
    if getattr(args, 'quick', False):
        delta_base = (loo['accuracy'] - cm['baseline_acc']) * 100
        grade = "🏆" if loo['accuracy']>=0.75 else "✅" if loo['accuracy']>=0.65 else "⚠️" if loo['accuracy']>=0.55 else "❌"
        print(f"\n[Classify Quick] n={len(valid)} · threshold={threshold:.1f}{' (auto)' if threshold_auto else ''}")
        print(f"  Accuracy {loo['accuracy']*100:>5.1f}% (LOOCV)  vs baseline {cm['baseline_acc']*100:.1f}%  {delta_base:+.1f}%p  {grade}")
        print(f"  Precision {loo['precision']*100:.1f}% · Recall {loo['recall']*100:.1f}% · F1 {loo['f1']:.3f}")
        print(f"  TP {cm['tp']} · TN {cm['tn']} · FP {cm['fp']} · FN {cm['fn']}")
        return

    # ============ FULL REPORT ============
    print("\n" + "="*60)
    print("🎯 CLASSIFICATION REPORT — KBO 외인 Success/Failure 예측")
    print("="*60)

    print(f"\n[설정]")
    print(f"  · 모델 점수: Stuff×0.35 + Cmd×0.30 + Ctx×0.20 + Dur×0.15")
    print(f"  · 분류 임계값: 점수 ≥ {threshold:.2f} → 'Success' 예측")
    if threshold_auto:
        print(f"    (Success 그룹 평균 {sum(s_scores)/len(s_scores):.2f}, Failure 그룹 평균 {sum(f_scores)/len(f_scores):.2f}의 중간)")
    print(f"  · 정답 기준: 실제 KBO WAR ≥ {war_cutoff} → 'Success'")
    print(f"  · 데이터셋: n={len(valid)} ({Path(args.data).name})")

    # 결과 출력
    print(f"\n[혼동행렬]")
    print(f"  실제\\예측    Success   Failure")
    print(f"  Success      {cm['tp']:>5}   {cm['fn']:>5}")
    print(f"  Failure      {cm['fp']:>5}   {cm['tn']:>5}")

    print(f"\n[성능 지표 — In-sample]")
    print(f"  Accuracy:      {cm['accuracy']*100:>5.1f}%   ({cm['tp']+cm['tn']}/{cm['n']} 정답)")
    print(f"  Precision:     {cm['precision']*100:>5.1f}%   (Success 예측 중 진짜 Success)")
    print(f"  Recall:        {cm['recall']*100:>5.1f}%   (진짜 Success 중 잡아낸 비율)")
    print(f"  Specificity:   {cm['specificity']*100:>5.1f}%   (진짜 Failure 중 거른 비율)")
    print(f"  F1 Score:      {cm['f1']:.3f}")

    print(f"\n[베이스라인 비교]")
    print(f"  무작위(다수 클래스) Accuracy: {cm['baseline_acc']*100:.1f}%")
    print(f"  모델 Accuracy:                {cm['accuracy']*100:.1f}%")
    delta = cm['improvement_over_baseline']*100
    print(f"  베이스라인 대비:               {delta:+.1f}%p {'✅' if delta>=10 else '⚠️' if delta>=3 else '❌'}")

    print(f"\n[LOOCV 검증 — 진짜 일반화 성능]")
    print(f"  LOOCV Accuracy: {loo['accuracy']*100:.1f}%")
    print(f"  LOOCV F1:       {loo['f1']:.3f}")
    print(f"  in-sample 대비: {(loo['accuracy']-cm['accuracy'])*100:+.1f}%p")
    if loo['accuracy'] - cm['accuracy'] < -0.10:
        print(f"  → ⚠️ 임계값 과적합 의심 (in-sample 거품)")
    else:
        print(f"  → ✅ 견고함")

    # 틀린 케이스
    wrong_count = sum(1 for r in rows if not r['correct'])
    if wrong_count > 0:
        print(f"\n[틀린 케이스 ({wrong_count}개)]")
        for r in sorted(rows, key=lambda x: -abs(x['score']-threshold) if not x['correct'] else 999):
            if not r['correct']:
                kind = "False Positive" if r['pred']=="Success" else "False Negative"
                print(f"  ❌ {r['name']:<25} Score={r['score']:>5.1f} 예측={r['pred']:<8} WAR={r['war']:+.2f} ({kind})")

    # 모델 카드
    print(f"\n" + "="*60)
    print(f"📊 분류 모델 카드")
    print(f"="*60)
    honest_acc = loo['accuracy']
    grade = ("🏆 강함" if honest_acc>=0.75 else
             "✅ 합리적" if honest_acc>=0.65 else
             "⚠️ 약함" if honest_acc>=0.55 else "❌ 베이스라인 수준")
    print(f"  · 정직한 정확도 (LOOCV):  {honest_acc*100:.1f}% ({grade})")
    print(f"  · F1 (LOOCV):              {loo['f1']:.3f}")
    print(f"  · 베이스라인 대비:         {(honest_acc-cm['baseline_acc'])*100:+.1f}%p")
    print(f"  · 권장 사용: 영입 의사결정 1차 필터")

    # 마크다운 저장
    md = Render.classify_md(rows, cm, loo, threshold, war_cutoff, len(valid))
    path = out_dir / "classification_report.md"
    path.write_text(md, encoding='utf-8')
    print(f"\n💾 저장: {path}")


def _render_classify_md(rows, cm, loo, threshold, war_cutoff, n) -> str:
    wrong_rows = "\n".join(
        f"| {r['name']} | {r['score']:.1f} | {r['pred']} | {r['war']:+.2f} | {'FP' if r['pred']=='Success' else 'FN'} |"
        for r in rows if not r['correct']
    )
    return f"""# 🎯 Classification Report — KBO Scouting AI OS

> 생성: {datetime.now().strftime('%Y-%m-%d %H:%M')}
> n={n}, 임계값: Score ≥ {threshold:.2f} → Success, 정답: WAR ≥ {war_cutoff} → Success

## 📊 성능 지표 (LOOCV — 진짜 일반화)

| Metric | In-sample | LOOCV |
|---|---|---|
| **Accuracy** | {cm['accuracy']*100:.1f}% | **{loo['accuracy']*100:.1f}%** |
| Precision | {cm['precision']*100:.1f}% | {loo['precision']*100:.1f}% |
| Recall | {cm['recall']*100:.1f}% | {loo['recall']*100:.1f}% |
| F1 Score | {cm['f1']:.3f} | {loo['f1']:.3f} |

베이스라인(다수 클래스로 다 찍기): {cm['baseline_acc']*100:.1f}%
LOOCV 베이스라인 대비: **{(loo['accuracy']-cm['baseline_acc'])*100:+.1f}%p**

## 🔲 혼동행렬 (In-sample)

| 실제 \\ 예측 | Success | Failure |
|---|---|---|
| **Success** | {cm['tp']} (TP) | {cm['fn']} (FN) |
| **Failure** | {cm['fp']} (FP) | {cm['tn']} (TN) |

## ❌ 틀린 케이스

| Player | Model Score | 예측 | 실제 WAR | 오류 유형 |
|---|---|---|---|---|
{wrong_rows}

## 🎤 발표 시 정직 멘트

> "이진 분류로 평가하면 모델의 LOOCV 정확도는 **{loo['accuracy']*100:.0f}%** 이며,
> 베이스라인(다수 클래스로 다 찍기) 대비 **{(loo['accuracy']-cm['baseline_acc'])*100:+.0f}%p** 우월합니다.
> Precision {loo['precision']*100:.0f}%, Recall {loo['recall']*100:.0f}%로,
> 영입 의사결정의 1차 필터로 사용하기에는 유효합니다."
"""


def cmd_recontract(args, dataset, out_dir):
    """재계약 외인 2년차 WAR 예측 모델 평가.

    is_recontract=True 필터링한 선수들로 별도 모델 검증.
    """
    logger.info("Starting recontract model evaluation")
    print("\n" + "="*60)
    print("🔁 RECONTRACT MODEL — 재계약 외인 2년차 WAR 예측")
    print("="*60)

    model = RecontractModel()
    result = model.evaluate(dataset)

    if "error" in result:
        print(f"\n❌ 평가 불가: {result['error']} (n={result.get('n', 0)})")
        print(f"   필요: is_recontract=True + KBO 1년차 5개 지표 보유 선수")
        print(f"   최소 3명 데이터 필요")
        return

    print(f"\n[설정]")
    print(f"  · 입력: KBO 1년차 IP/ERA/FIP/K9/BB9 (+선택적 raw 스탯)")
    print(f"  · 출력: KBO 2년차 WAR 예측")
    print(f"  · 가중치: WAR 50% / FIP 20% / K9-BB9 20% / IP 10%")
    print(f"  · 데이터: n={result['n']} (재계약 외인)")

    print(f"\n[성능 지표]")
    print(f"  Pearson r: {result['pearson_r']:+.4f}")
    print(f"  MAE:       {result['mae']:.2f} WAR")
    print(f"  RMSE:      {result['rmse']:.2f} WAR")

    print(f"\n[예측 vs 실제]")
    print(f"  {'선수':<25} {'1년차 WAR':>10} {'예측':>8} {'실제':>8} {'잔차':>8}")
    print(f"  " + "-"*65)
    for r in result['rows']:
        marker = "✅" if abs(r['residual']) < 1.0 else "⚠️"
        print(f"  {r['name']:<25} {r['kbo_y1_war']:>10.2f} "
              f"{r['predicted_y2_war']:>8.2f} {r['actual_y2_war']:>8.2f} "
              f"{r['residual']:>+8.2f} {marker}")

    # 평가 메시지
    print(f"\n" + "="*60)
    r = result['pearson_r']
    if r >= 0.6:
        print(f"🏆 강함 (r={r:+.4f}) — 재계약 결정에 실전 활용 가능")
    elif r >= 0.4:
        print(f"✅ 합리적 (r={r:+.4f}) — 의사결정 보조 도구로 유효")
    elif r >= 0.2:
        print(f"⚠️ 약함 (r={r:+.4f}) — 데이터 더 필요")
    else:
        print(f"❌ 부족 (r={r:+.4f}) — 모델 재설계 또는 데이터 부족")


def cmd_risk(args, dataset, out_dir):
    """[Risk Engine] 개별 선수 생존 확률 예측"""
    import warnings
    warnings.filterwarnings("ignore")
    from risk_engine import RiskEngine
    from scout_report import format_prediction_text

    logger.info(f"Risk Engine prediction: {args.name}")

    target = next((p for p in dataset if p.name.lower() == args.name.lower()), None)
    if target is None:
        print(f"❌ '{args.name}' not found in dataset")
        return

    engine = RiskEngine()
    engine.fit(dataset)
    prediction = engine.predict(target)

    text = format_prediction_text(prediction, quick=args.quick,
                                   engine=engine, pitcher=target)
    print(text)

    if not args.quick:
        path = out_dir / f"risk_{target.name.replace(' ','_')}.md"
        path.write_text(text, encoding="utf-8")
        print(f"\n💾 저장: {path}")


def cmd_league(args, dataset, out_dir):
    """[Risk Engine] 리그 전체 risk 평가"""
    import warnings
    warnings.filterwarnings("ignore")
    from risk_engine import RiskEngine
    from scout_report import render_league_table

    logger.info("Risk Engine: League-wide assessment")

    engine = RiskEngine()
    engine.fit(dataset)
    text = render_league_table(engine, dataset, sort_by=args.sort)
    print(text)

    path = out_dir / "risk_league.txt"
    path.write_text(text, encoding="utf-8")
    print(f"\n💾 저장: {path}")


def cmd_risk_validate(args, dataset, out_dir):
    """[Risk Engine] Brier/BSS/Calibration 통계적 검증"""
    import warnings
    warnings.filterwarnings("ignore")
    from risk_engine import validate_engine, loocv_validate

    logger.info("Risk Engine validation")
    print("\n" + "="*68)
    print("🧪 RISK ENGINE VALIDATION")
    print("="*68)

    val = validate_engine(dataset)
    print(f"\n[Stratified K-Fold] n={val.n}")
    print(f"  Brier Score:        {val.brier_score:.4f} (lower = better)")
    print(f"    baseline (prior): {val.brier_baseline_prior:.4f}")
    print(f"    baseline (xfip):  {val.brier_baseline_single:.4f}")
    print(f"  Brier Skill Score:  {val.brier_skill_score:+.4f}")
    print(f"    {'✅ 개선' if val.brier_skill_score > 0 else '⚠️ 미달' if val.brier_skill_score > -0.05 else '❌ 베이스라인 미달'}")
    print(f"  ROC-AUC:            {val.roc_auc:.4f}")
    print(f"  Precision / Recall: {val.precision:.3f} / {val.recall:.3f}")
    print(f"  Catastrophic Miss:  {val.catastrophic_miss_rate*100:.1f}%")
    print(f"    ({'✅ 양호' if val.catastrophic_miss_rate < 0.30 else '⚠️ 주의' if val.catastrophic_miss_rate < 0.50 else '❌ 위험'})")

    print(f"\n[계수 안정성] (sign_changes ≥ 2 = ⚠️)")
    for fname, stats in sorted(val.coefficient_stability.items(),
                                key=lambda x: -abs(x[1]['mean'])):
        flag = '⚠️' if stats['warning'] else '✅'
        print(f"  {flag} {fname:<30} mean={stats['mean']:+.3f}  "
              f"std={stats['std']:.3f}  sign_changes={stats['sign_changes']}")

    print(f"\n[Calibration Bins] (예측 확률 vs 실제 비율)")
    print(f"  {'Predicted':>12} {'Actual':>10} {'N':>5}")
    for b in val.calibration_bins:
        bar = "█" * int(b['actual_avg'] * 20)
        print(f"  {b['predicted_avg']*100:>11.1f}% {b['actual_avg']*100:>9.1f}% {b['n']:>5}  {bar}")

    # LOOCV (보조)
    loo = loocv_validate(dataset)
    if "error" not in loo:
        print(f"\n[LOOCV] n={loo['n']}")
        print(f"  Brier Score:           {loo['brier_score']:.4f}")
        print(f"  ROC-AUC:               {loo['roc_auc']:.4f}")
        print(f"  Success 평균 확률:     {loo['mean_prob_success']*100:.1f}%")
        print(f"  Failure 평균 확률:     {loo['mean_prob_failure']*100:.1f}%")

    print("\n" + "="*68)


def main():
    """CLI 진입점. 9개 서브커맨드 처리 + 예외 핸들링.

    Returns:
        int: 종료 코드 (0=성공, 1=실패, 2=파일 없음)
    """
    parser = argparse.ArgumentParser(
        description="KBO Scouting AI OS Ver 14.3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
환경변수:
  KBO_LOG_LEVEL  로그 레벨 (DEBUG/INFO/WARNING/ERROR, 기본 INFO)

예시:
  python3 kbo_oracle.py --data dataset_external.json classify --quick
  python3 kbo_oracle.py --data dataset_external.json validate
  KBO_LOG_LEVEL=DEBUG python3 kbo_oracle.py backtest
        """
    )
    parser.add_argument('--data', default='dataset.json',
                        help='데이터셋 파일 경로 (기본: dataset.json)')
    parser.add_argument('--out',  default='outputs',
                        help='리포트 저장 경로 (기본: outputs/)')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='DEBUG 레벨 로그 출력')
    parser.add_argument('-q', '--quiet', action='store_true',
                        help='ERROR 레벨만 출력')

    sub = parser.add_subparsers(dest='cmd', required=True)
    p1 = sub.add_parser('scout'); p1.add_argument('name')
    sub.add_parser('backtest')
    p3 = sub.add_parser('compare'); p3.add_argument('a'); p3.add_argument('b')
    sub.add_parser('value')
    pa = sub.add_parser('analyze')
    pa.add_argument('--input', help='JSON 파일')
    pa.add_argument('--json', help='인라인 JSON')
    pa.add_argument('--text', help='V2 한 줄 양식 (선수명/년도/리그/G/GS/IP/K9/BB9/HR9/FIP/평속/IVB/BABIP → WAR)')

    pv = sub.add_parser('validate', help='LOOCV/Bootstrap/Ablation 정직성 검증')
    pv.add_argument('--bootstrap-iters', type=int, default=2000)

    pc = sub.add_parser('classify', help='Success/Failure 이진 분류 평가')
    pc.add_argument('--threshold', type=float, default=None,
                    help='분류 임계값 (없으면 자동: Success/Failure 그룹 평균 중간)')
    pc.add_argument('--war-cutoff', type=float, default=3.0,
                    help='Success 정답 WAR 기준 (기본 3.0)')
    pc.add_argument('--quick', action='store_true',
                    help='핵심 지표만 4줄로 출력')

    # NEW Ver 14.4: 재계약 외인 모델 평가
    pr = sub.add_parser('recontract', help='재계약 외인 2년차 WAR 예측 모델 평가')

    # NEW Ver 15.0: Risk Governance Engine (ML 기반)
    prk = sub.add_parser('risk', help='[Risk Engine] 개별 선수 생존 확률 예측')
    prk.add_argument('name', help='선수 이름')
    prk.add_argument('--quick', action='store_true', help='4줄 요약 출력')

    prk2 = sub.add_parser('league', help='[Risk Engine] 리그 전체 risk 평가')
    prk2.add_argument('--sort', default='survival_prob',
                      choices=['survival_prob', 'name'])

    prk3 = sub.add_parser('risk-validate', help='[Risk Engine] Brier/BSS/Calibration 검증')

    args = parser.parse_args()

    # 로그 레벨 조정
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    elif args.quiet:
        logger.setLevel(logging.ERROR)

    try:
        dataset = load_dataset(Path(args.data))
    except FileNotFoundError as e:
        logger.error(f"Dataset file not found: {args.data}")
        return 2
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in {args.data}: {e}")
        return 1

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cmds = {'scout':cmd_scout,'backtest':cmd_backtest,'compare':cmd_compare,
            'value':cmd_value,'analyze':cmd_analyze,
            'validate':cmd_validate,'classify':cmd_classify,
            'recontract':cmd_recontract,
            # Risk Engine commands (Ver 15.0)
            'risk':cmd_risk, 'league':cmd_league, 'risk-validate':cmd_risk_validate}
    try:
        cmds[args.cmd](args, dataset, out_dir)
        return 0
    except Exception as e:
        logger.exception(f"Command '{args.cmd}' failed: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
