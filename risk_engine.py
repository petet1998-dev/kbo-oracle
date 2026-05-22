"""
KBO Foreign Pitcher Risk-Control Engine
========================================
This system does not attempt to perfectly predict pitcher success.
It attempts to reduce avoidable investment failures under uncertainty
through calibrated probability modeling, statistical validation, and
explainable risk decomposition.

The final decision authority remains with human evaluators;
this engine supports, not replaces, scouting judgment.

핵심 철학:
1. Prediction Engine이 아닌 Investment Risk Governance Engine
2. Lore-Free: Failure Score / Bonus / Pivot 등 휴리스틱 전부 제거
3. Statistical Honesty: Calibration > Accuracy
4. Small-N Robustness: 단일 변수 지배 금지, multiple weak signals 합의

기술 스택:
- sklearn LogisticRegression (elasticnet) — 계수 해석 가능
- CalibratedClassifierCV (sigmoid) — 확률 정직성
- Stratified K-Fold (n=50 환경에 맞춤, temporal validation 대신)
- LOOCV 보조 검증

설계 결정 (원본 프롬프트 대비 조정):
- Temporal Rolling Validation → Stratified K-Fold + LOOCV
  이유: 우리 데이터는 시즌별 분포 불균형 (대부분 2019~2025)
- Sample Weighting Experiment 제외
  이유: n=50에서는 통계적으로 의미 없음
- Data versioning 단순화: 로그만 남기고 archive 디렉토리 생략
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    brier_score_loss, roc_auc_score, precision_score, recall_score,
    confusion_matrix,
)
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


logger = logging.getLogger("kbo_oracle.risk_engine")


# ============================================================
# Feature Definitions
# ============================================================
# 모델이 사용하는 raw 입력 변수만 (lore 변수 없음)
# 결측 시 None — 모델 내부에서 imputation
FEATURE_COLUMNS = [
    # Skill
    "shrunk_k_minus_bb",  # K/9 - BB/9 (shrinkage 적용 가능)
    "swstr_pct",          # Swinging Strike % (= SwStr% from FanGraphs)
    "csw_pct",            # Called Strike + Whiff % (Ver 15.2)
    "gb_pct",             # Ground Ball %
    "xfip",               # Expected FIP (FIP의 단점 보완)
    # Physical
    "age_at_debut",       # 데뷔 시 나이
    "avg_velo",           # 평속 (mph)
    "velo_trend",         # 직전 - 전성기 (음수 = 하락)
    # Role & Usage
    "gs_ratio",           # GS / G (선발 비율)
    "ip_per_g",           # IP / G (등판당 이닝)
    "pitch_mix_count",    # 구종 개수
    # Stability
    "il_days_past_2y",    # 최근 2년 IL 일수
    # Environment (dummies)
    "is_pcl",             # 직전 리그가 AAA-PCL인가
    "is_il",              # 직전 리그가 AAA-IL인가
    "is_asian_league",    # NPB / KBO 출신인가
    # Ver 15.2: 결측 indicator (Missing as Risk Signal)
    "is_k9_missing",
    "is_bb9_missing",
    "is_pitch_mix_missing",
]


# ============================================================
# Survival Target Definition
# ============================================================
def compute_survival_target(actual_war: Optional[float],
                             prev_ip: Optional[float] = None) -> Optional[int]:
    """is_survived 이진 분류 타겟 (Ver 15.2 도메인 반영).

    1 = Survival: KBO 외인으로 진짜 살아남음 (재계약 확정 수준)
       기준: actual_war >= 3.5
    0 = Failure: 재계약 실패 (90% 이상 2년차 퇴출)

    도메인 근거:
        KBO 외인 시장에서 풀타임 WAR < 3.5 선수의 90% 이상이 2년차에 재계약
        실패. 즉 WAR 3.5가 진짜 "재계약 임계점". 우리 모델은 이 비즈니스
        의사결정을 직접 예측.

    이전 정의 (Ver 15.0~15.1)는 WAR ≥ 2.0이었으나, 이는 "풀시즌 채움"
    수준이라 너무 후한 기준이었음. Ver 15.2에서 도메인 전문가 의견 반영.
    """
    if actual_war is None:
        return None
    return 1 if actual_war >= 3.5 else 0


# ============================================================
# Feature Extraction
# ============================================================
def extract_features(pitcher) -> dict:
    """Pitcher 객체에서 ML 입력 변수 추출.

    raw 값이 없는 경우 None 반환 (모델 imputation 처리).
    derived features는 가능하면 계산.

    Args:
        pitcher: Pitcher 인스턴스

    Returns:
        dict: feature 이름 → 값 (또는 None)
    """
    p = pitcher
    features = {}

    # ===== Skill =====
    # K-BB = K/9 - BB/9 (가장 안정적 능력 지표)
    # Ver 15.2: raw 값 직접 사용 (이전 역산 로직 버그 수정)
    raw_k9 = getattr(p, 'raw_k9', None)
    raw_bb9 = getattr(p, 'raw_bb9', None)
    if raw_k9 is not None and raw_bb9 is not None:
        features["shrunk_k_minus_bb"] = raw_k9 - raw_bb9
    else:
        features["shrunk_k_minus_bb"] = None

    # SwStr% — 직접 컬럼 있으면 우선, 없으면 whiff_pct를 proxy
    # Ver 15.2.1: None-safe (기본값 제거로 whiff_pct가 None일 수 있음)
    _swstr = getattr(p, 'swstr_pct', None)
    _whiff = getattr(p, 'whiff_pct', None)
    if _swstr is not None:
        features["swstr_pct"] = _swstr
    elif _whiff is not None and _whiff > 0:
        features["swstr_pct"] = _whiff
    else:
        features["swstr_pct"] = None

    # CSW% — Ver 15.2: 정식 feature 추가 (Called Strike + Whiff)
    features["csw_pct"] = getattr(p, 'csw_pct', None)

    # GB% — 신규 변수
    features["gb_pct"] = getattr(p, 'gb_pct', None)

    # xFIP — 직접 있으면 사용, 없으면 FIP를 proxy
    features["xfip"] = getattr(p, 'xfip', None) or getattr(p, 'fip', None)

    # ===== Physical =====
    features["age_at_debut"] = getattr(p, 'age', None)
    # Ver 15.2: raw_velo 우선 사용 (role 페널티 적용 전 원본)
    # 이유: gs_ratio가 이미 별도 feature로 들어가므로 role 정보 중복 회피
    raw_velo = getattr(p, 'raw_velo', None)
    if raw_velo is not None and raw_velo > 0:
        features["avg_velo"] = raw_velo
    elif p.avg_velo and p.avg_velo > 0:
        features["avg_velo"] = p.avg_velo
    else:
        features["avg_velo"] = None
    features["velo_trend"] = p.velo_trend if p.velo_trend != 0 else None

    # ===== Role =====
    # gs_ratio
    g = getattr(p, 'g', None)
    gs = getattr(p, 'gs', None)
    if g and gs is not None and g > 0:
        features["gs_ratio"] = gs / g
        features["ip_per_g"] = (p.prev_ip or 0) / g if p.prev_ip else None
    else:
        features["gs_ratio"] = None
        features["ip_per_g"] = None

    features["pitch_mix_count"] = len(p.pitch_types) if p.pitch_types else None

    # ===== Stability =====
    features["il_days_past_2y"] = getattr(p, 'il_days_2y', None)

    # ===== Environment Dummies =====
    league = (p.prev_league or p.last_league or "").upper()
    features["is_pcl"] = 1 if "PCL" in league else 0
    features["is_il"] = 1 if "IL" in league and "PCL" not in league else 0
    features["is_asian_league"] = 1 if any(x in league for x in ["NPB", "KBO"]) else 0

    # ===== Ver 15.2: 결측 Indicator (Missing as Risk Signal) =====
    # Pitcher에 명시적으로 저장된 indicator 사용 (build_pitcher_from_raw에서 설정)
    features["is_k9_missing"] = 1 if getattr(p, 'is_k9_missing', False) else 0
    features["is_bb9_missing"] = 1 if getattr(p, 'is_bb9_missing', False) else 0
    features["is_pitch_mix_missing"] = 1 if getattr(p, 'pitch_mix_missing', False) else 0

    return features


# ============================================================
# Pipeline Factory
# ============================================================
def build_pipeline(random_state: int = 42) -> Pipeline:
    """Leakage-free pipeline (Ver 15.2 강화).

    순서: Imputation → Scaling → LogisticRegression

    Ver 15.2 변경:
    - keep_empty_features=True: 특정 fold에서 컬럼 전체가 결측이어도 파이프라인
      터지지 않음. 결측 컬럼은 0으로 채워지지만 indicator로 정보 보존.

    Args:
        random_state: 재현성 시드

    Returns:
        sklearn Pipeline
    """
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            l1_ratio=0.5,
            class_weight="balanced",
            max_iter=5000,
            random_state=random_state,
        )),
    ])


def build_calibrated_model(random_state: int = 42, cv: int = 5) -> CalibratedClassifierCV:
    """확률 보정 적용 모델.

    sigmoid (Platt scaling) 사용 — Small-N에서 isotonic은 과적합.
    cv='prefit' 절대 금지 (leakage).

    Args:
        random_state: 시드
        cv: K-fold

    Returns:
        CalibratedClassifierCV
    """
    return CalibratedClassifierCV(
        estimator=build_pipeline(random_state=random_state),
        method="sigmoid",  # isotonic 금지 (Small-N)
        cv=cv,
    )


# ============================================================
# Risk Tier Logic
# ============================================================
def assign_risk_tier(survival_prob: float) -> dict:
    """확률 → Risk Tier 분류.

    Returns:
        dict: tier, label, color, recommendation
    """
    if survival_prob >= 0.70:
        return {
            "tier": "GREEN",
            "label": "Contractable Target",
            "color": "#10B981",
            "recommendation": "계약 가능 — 정상 영입 절차 진행",
        }
    elif survival_prob >= 0.45:
        return {
            "tier": "YELLOW",
            "label": "Human Scout Review Required",
            "color": "#F59E0B",
            "recommendation": "현장 스카우트 추가 검증 필요 — 영상/메디컬 확인",
        }
    else:
        return {
            "tier": "RED",
            "label": "Reject / High Risk",
            "color": "#EF4444",
            "recommendation": "영입 비추천 — 투자 회피 권고",
        }


# ============================================================
# Result Containers
# ============================================================
@dataclass
class ValidationResult:
    """검증 결과 저장 컨테이너"""
    n: int = 0
    brier_score: float = 0.0
    brier_baseline_prior: float = 0.0
    brier_baseline_single: float = 0.0
    brier_skill_score: float = 0.0
    roc_auc: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    catastrophic_miss_rate: float = 0.0  # Low Risk 판정한 후 실패 비율
    coefficient_stability: dict = field(default_factory=dict)
    fold_results: list = field(default_factory=list)
    calibration_bins: list = field(default_factory=list)


@dataclass
class ScoutPrediction:
    """개별 선수 예측 결과"""
    name: str
    survival_prob: float
    risk_tier: dict
    feature_contributions: list = field(default_factory=list)
    confidence_warning: Optional[str] = None
    used_features: dict = field(default_factory=dict)


# ============================================================
# Risk Engine — 메인 클래스
# ============================================================
class RiskEngine:
    """KBO Foreign Pitcher Investment Risk Governance Engine.

    이 엔진은 raw 야구 데이터로부터:
      1. Survival Probability 추정 (calibrated)
      2. Risk Tier 분류 (GREEN/YELLOW/RED)
      3. 계수 기반 설명 (SHAP 대체)
      4. 통계적 정직성 검증

    을 수행합니다.

    Example:
        >>> engine = RiskEngine()
        >>> engine.fit(pitchers)
        >>> result = engine.predict(new_pitcher)
        >>> result.survival_prob
        0.65
        >>> result.risk_tier['tier']
        'YELLOW'
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.model: Optional[CalibratedClassifierCV] = None
        self.feature_names: list = []
        self.imputer_feature_names: list = []  # add_indicator 추가된 이름
        self.coefficients: Optional[np.ndarray] = None  # 평균 계수 (해석용)
        self.is_fitted = False
        self.training_n = 0

    def _extract_xy(self, pitchers: list, add_missing_indicators: bool = True) -> tuple:
        """Pitcher list → X matrix, y vector.

        Ver 15.2 데이터 누수 차단:
        - 사전 imputation 제거 (전체 데이터 median이 fold로 누수되는 문제)
        - NaN 그대로 pipeline에 전달 → fold별 imputer가 fold 안에서만 median 계산
        - 결측 indicator는 사전에 추가 (이건 데이터 누수 아님, 결측 여부 자체가 정보)
        - pipeline의 SimpleImputer는 keep_empty_features=True로 안전성 보장

        Args:
            pitchers: Pitcher 리스트
            add_missing_indicators: 결측 indicator 컬럼 추가 여부

        Returns:
            (X, y): numpy arrays. X에 NaN 포함될 수 있음 (pipeline이 처리).
        """
        X_dicts = []
        y_list = []
        for p in pitchers:
            target = compute_survival_target(p.actual_war)
            if target is None:
                continue
            features = extract_features(p)
            X_dicts.append(features)
            y_list.append(target)

        if not X_dicts:
            return None, None

        base_feature_names = list(FEATURE_COLUMNS)
        X_base = np.array([[d.get(f) for f in base_feature_names] for d in X_dicts],
                          dtype=np.float64)

        # 결측 indicator (데이터 누수 아님 - 단순 boolean 추출)
        indicator_cols = []
        indicator_names = []
        if add_missing_indicators:
            for i, fname in enumerate(base_feature_names):
                col = X_base[:, i]
                nan_count = np.isnan(col).sum()
                if 0 < nan_count < len(col):
                    indicator = np.isnan(col).astype(np.float64)
                    indicator_cols.append(indicator)
                    indicator_names.append(f"{fname}__missing")

        # ⚠️ Ver 15.2: 사전 imputation 루프 삭제 (데이터 누수 차단)
        # NaN 그대로 pipeline의 SimpleImputer가 fold별로 처리

        if indicator_cols:
            indicator_matrix = np.column_stack(indicator_cols)
            X = np.column_stack([X_base, indicator_matrix])
            self.feature_names = base_feature_names + indicator_names
        else:
            X = X_base
            self.feature_names = base_feature_names

        y = np.array(y_list)
        return X, y

    def fit(self, pitchers: list) -> "RiskEngine":
        """모델 학습.

        Args:
            pitchers: Pitcher 리스트 (actual_war 필수)

        Returns:
            self
        """
        X, y = self._extract_xy(pitchers)
        if X is None or len(X) < 10:
            raise ValueError(f"학습 데이터 부족 (n={len(X) if X is not None else 0}). 최소 10명 필요.")

        n_pos = int((y == 1).sum())
        n_neg = int((y == 0).sum())
        if n_pos < 2 or n_neg < 2:
            raise ValueError(f"클래스 불균형 심각 (pos={n_pos}, neg={n_neg}). 최소 각 2명 필요.")

        # 전부 0인 컬럼 제거 (전부 결측이었던 컬럼 — variance 0 → scaler 문제)
        col_variance = X.std(axis=0)
        keep_cols = col_variance > 0
        self._kept_indices = np.where(keep_cols)[0].tolist()
        self._kept_feature_names = [self.feature_names[i] for i in self._kept_indices]
        X = X[:, keep_cols]

        if X.shape[1] == 0:
            raise ValueError("모든 변수가 상수. 학습 불가.")

        cv_folds = min(5, n_pos, n_neg)
        self.model = build_calibrated_model(random_state=self.random_state, cv=cv_folds)
        self.model.fit(X, y)

        # 계수 추출
        coefs = []
        for calibrated_clf in self.model.calibrated_classifiers_:
            lr = calibrated_clf.estimator.named_steps["model"]
            coefs.append(lr.coef_[0])
        self.coefficients = np.mean(np.array(coefs), axis=0)

        self.is_fitted = True
        self.training_n = len(X)

        # 유사 사례 분석용: 학습 선수 + 그들의 feature 보관
        # (predict 시 가장 가까운 실제 선수를 찾기 위함)
        self._train_pitchers = [
            p for p in pitchers if compute_survival_target(p.actual_war) is not None
        ]
        # 각 학습 선수의 비교용 feature (raw, 결측은 NaN)
        self._train_features = np.array([
            [extract_features(p).get(f) for f in self._kept_feature_names]
            for p in self._train_pitchers
        ], dtype=np.float64)
        # 표준화용 통계 (거리 계산 시 스케일 정규화)
        self._train_feat_mean = np.nanmean(self._train_features, axis=0)
        self._train_feat_std = np.nanstd(self._train_features, axis=0)
        self._train_feat_std[self._train_feat_std == 0] = 1.0  # 0 division 방지

        logger.info(f"RiskEngine fitted on n={len(X)} (pos={n_pos}, neg={n_neg}, "
                   f"cv={cv_folds}, features={X.shape[1]})")
        return self

    def find_similar_cases(self, pitcher, top_k: int = 3) -> list:
        """학습 데이터에서 가장 유사한 실제 선수 top_k명을 찾는다.

        14개 변수 전체를 표준화 거리(정규화 유클리드)로 계산.
        결측 변수는 거리 계산에서 제외 (가용 변수만으로 비교).

        Args:
            pitcher: 평가 대상 Pitcher
            top_k: 반환할 유사 선수 수

        Returns:
            [{name, war, label, distance, prev_league, shared_features}, ...]
            거리 오름차순 (가장 닮은 선수가 먼저)
        """
        if not self.is_fitted or not hasattr(self, "_train_features"):
            return []

        # 대상 선수의 feature
        target = np.array([
            extract_features(pitcher).get(f) for f in self._kept_feature_names
        ], dtype=np.float64)

        # 표준화
        target_z = (target - self._train_feat_mean) / self._train_feat_std
        train_z = (self._train_features - self._train_feat_mean) / self._train_feat_std

        results = []
        for i, tp in enumerate(self._train_pitchers):
            # 자기 자신은 제외 (같은 이름)
            if tp.name == pitcher.name:
                continue
            # 둘 다 값이 있는 변수만으로 거리 계산
            diff = target_z - train_z[i]
            valid = ~np.isnan(diff)
            if valid.sum() == 0:
                continue
            # 정규화 유클리드 거리 (가용 변수 수로 나눠 공정 비교)
            dist = float(np.sqrt(np.nansum(diff[valid] ** 2) / valid.sum()))
            # 어떤 변수가 비슷한지 (가장 가까운 3개)
            results.append({
                "name": tp.name,
                "war": tp.actual_war,
                "label": "Success" if tp.actual_war >= 3.5 else "Failure",
                "distance": dist,
                "prev_league": getattr(tp, "prev_league", None),
                "n_compared": int(valid.sum()),
            })

        results.sort(key=lambda r: r["distance"])
        return results[:top_k]

    def predict(self, pitcher) -> ScoutPrediction:
        """단일 선수 예측.

        Returns:
            ScoutPrediction: 확률 + Risk Tier + 설명
        """
        if not self.is_fitted:
            raise RuntimeError("모델이 학습되지 않음. fit() 먼저 호출.")

        features = extract_features(pitcher)
        X_base = np.array([[features.get(f) for f in FEATURE_COLUMNS]], dtype=np.float64)

        # 학습 시 사용한 indicator 컬럼 동일하게 생성
        indicator_names_in_train = [n for n in self.feature_names if n.endswith("__missing")]
        if indicator_names_in_train:
            indicator_cols = []
            for ind_name in indicator_names_in_train:
                base = ind_name.replace("__missing", "")
                if base in FEATURE_COLUMNS:
                    val = features.get(base)
                    indicator_cols.append(1.0 if val is None else 0.0)
            indicator_arr = np.array([indicator_cols], dtype=np.float64)
            X_full = np.column_stack([X_base, indicator_arr])
        else:
            X_full = X_base

        # 결측 imputation: 학습 시점에 우리가 사용한 median을 모르므로
        # pipeline의 SimpleImputer가 처리하도록 NaN → 0으로 일단 대체
        # (pipeline.imputer가 학습 시 보았던 median으로 다시 채움)
        # 단, NaN인 상태로 두면 pipeline imputer가 정상 작동함
        # 여기서는 NaN 그대로 전달
        X = X_full[:, self._kept_indices]

        prob = float(self.model.predict_proba(X)[0, 1])
        tier = assign_risk_tier(prob)
        contributions = self._explain(features, X)
        warning = self._check_confidence(features, X)

        return ScoutPrediction(
            name=pitcher.name,
            survival_prob=prob,
            risk_tier=tier,
            feature_contributions=contributions,
            confidence_warning=warning,
            used_features={k: v for k, v in features.items() if v is not None},
        )

    def _explain(self, features: dict, X: np.ndarray) -> list:
        """선형 모델 계수 × scaled feature → local contribution.

        SHAP 대체. 선형 모델은 정확한 contribution 계산 가능.
        """
        if self.coefficients is None:
            return []

        first_clf = self.model.calibrated_classifiers_[0].estimator
        imputer = first_clf.named_steps["imputer"]
        scaler = first_clf.named_steps["scaler"]

        X_imputed = imputer.transform(X)
        X_scaled = scaler.transform(X_imputed)

        contributions_raw = X_scaled[0] * self.coefficients

        # 학습 시 유지된 feature 이름과 매칭
        items = list(zip(self._kept_feature_names, contributions_raw, X[0]))

        # 상위 3 positive + 상위 3 negative
        sorted_items = sorted(items, key=lambda x: x[1], reverse=True)
        positives = [self._format_contrib(n, c, v) for n, c, v in sorted_items[:3] if c > 0]
        negatives = [self._format_contrib(n, c, v) for n, c, v in sorted_items[-3:] if c < 0]
        negatives.reverse()

        return positives + negatives

    def _format_contrib(self, name: str, contrib: float, value) -> dict:
        """야구 언어로 번역"""
        translations = {
            "shrunk_k_minus_bb": "탈삼진-볼넷 균형 (K-BB)",
            "swstr_pct": "헛스윙 유도율",
            "csw_pct": "CSW% (루킹 스트라이크 + 헛스윙 비율)",
            "gb_pct": "땅볼 유발 능력",
            "xfip": "수비독립 평균자책 (xFIP)",
            "age_at_debut": "데뷔 나이",
            "avg_velo": "평균 구속",
            "velo_trend": "구속 하락 추세",
            "gs_ratio": "선발 등판 비율",
            "ip_per_g": "등판당 이닝 (체력)",
            "pitch_mix_count": "구종 다양성",
            "il_days_past_2y": "최근 2년 부상 일수",
            "is_pcl": "PCL 환경 효과",
            "is_il": "AAA-IL 환경 효과",
            "is_asian_league": "아시아 리그 경험",
            "is_k9_missing": "K/9 데이터 결측 (리스크 신호)",
            "is_bb9_missing": "BB/9 데이터 결측 (리스크 신호)",
            "is_pitch_mix_missing": "구종 정보 결측 (리스크 신호)",
        }
        base_name = name.replace("__missing", "")
        translated = translations.get(base_name, base_name)
        if "__missing" in name:
            translated += " (데이터 결측 자체가 리스크 신호)"
        return {
            "feature": translated,
            "raw_name": name,
            "contribution": round(float(contrib), 4),
            "direction": "긍정" if contrib > 0 else "부정",
            "value": float(value) if value is not None and not np.isnan(value) else None,
        }

    def _check_confidence(self, features: dict, X: np.ndarray) -> Optional[str]:
        """신뢰도 경고 발생 조건 체크 (Ver 15.1 강화).

        체크 항목:
          1. 결측률 25% 이상
          2. 학습 데이터 부족 (n<30)
          3. Feature value가 학습 분포 2σ 이상 벗어남 (외삽 위험)
          4. 표본 크기 부족 (IP < 30이닝 = 평가 불가)
          5. Role mismatch (학습 데이터 대부분 선발인데 입력은 불펜)
        """
        warnings = []

        # 1. 결측률 (25%로 강화)
        n_missing = sum(1 for v in features.values() if v is None)
        total = len(features)
        missing_rate = n_missing / total if total > 0 else 0
        if missing_rate > 0.25:
            warnings.append(f"입력 {missing_rate*100:.0f}% 결측 — 신뢰도 낮음")

        # 2. 학습 데이터 부족
        if self.training_n < 30:
            warnings.append(f"학습 n={self.training_n} 부족 — 광범위한 신뢰구간")

        # 3. 외삽 (2σ로 강화, 더 민감)
        first_clf = self.model.calibrated_classifiers_[0].estimator
        X_imp = first_clf.named_steps["imputer"].transform(X)
        X_scaled = first_clf.named_steps["scaler"].transform(X_imp)
        if np.any(np.abs(X_scaled) > 2):
            extreme_count = int(np.sum(np.abs(X_scaled) > 2))
            warnings.append(f"입력 {extreme_count}개 변수가 학습 분포 2σ 밖 — 외삽 위험 ⚠️")

        # 4. 표본 부족 (이닝 기반)
        ip_per_g = features.get("ip_per_g")
        if ip_per_g is not None and ip_per_g < 3.0:
            # 등판당 이닝이 3 미만 = 불펜 단기 등판
            warnings.append("등판당 이닝 < 3 — 단기 표본, 평가 신뢰도 매우 낮음")

        # 5. Role mismatch (선발↔불펜)
        gs_ratio = features.get("gs_ratio")
        if gs_ratio is not None:
            if gs_ratio < 0.2:
                warnings.append("불펜 출신 — 학습 데이터 대부분 선발, role mismatch")
            elif 0.2 <= gs_ratio < 0.6:
                warnings.append("Swingman role — 선발/불펜 혼용")

        return " / ".join(warnings) if warnings else None


# ============================================================
# Validation: Stratified K-Fold + LOOCV
# ============================================================
def validate_engine(pitchers: list, random_state: int = 42) -> ValidationResult:
    """엔진 통계적 검증.

    원본 프롬프트의 Temporal Rolling Validation은 우리 데이터의 연도 분포
    한계로 Stratified K-Fold + LOOCV로 대체.

    측정:
      - Brier Score (primary): 확률 정직성
      - Brier Skill Score (vs prior + single-variable baseline)
      - ROC-AUC, Precision, Recall (secondary)
      - Catastrophic Miss Rate (Low Risk 판정 후 실패)
      - Coefficient stability across folds

    Returns:
        ValidationResult
    """
    temp = RiskEngine(random_state=random_state)
    X, y = temp._extract_xy(pitchers)
    if X is None or len(X) < 10:
        raise ValueError("검증 데이터 부족 (최소 10명)")

    # variance 0 컬럼 제거
    col_variance = X.std(axis=0)
    keep_cols = col_variance > 0
    kept_feature_names = [temp.feature_names[i] for i in np.where(keep_cols)[0]]
    X = X[:, keep_cols]

    result = ValidationResult(n=len(X))
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())

    n_folds = min(5, n_pos, n_neg)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    all_probs = []
    all_actuals = []
    fold_coefs = []
    fold_metrics = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        inner_cv = min(3, int((y_train == 1).sum()), int((y_train == 0).sum()))
        if inner_cv < 2:
            inner_cv = 2

        model = build_calibrated_model(random_state=random_state, cv=inner_cv)
        try:
            model.fit(X_train, y_train)
        except Exception as e:
            logger.warning(f"Fold {fold_idx} 학습 실패: {e}")
            continue

        probs = model.predict_proba(X_test)[:, 1]
        all_probs.extend(probs.tolist())
        all_actuals.extend(y_test.tolist())

        coefs = []
        for cal_clf in model.calibrated_classifiers_:
            lr = cal_clf.estimator.named_steps["model"]
            coefs.append(lr.coef_[0])
        fold_coefs.append(np.mean(np.array(coefs), axis=0))

        fold_metrics.append({
            "fold": fold_idx + 1,
            "n_test": len(y_test),
            "brier": float(brier_score_loss(y_test, probs)),
        })

    all_probs = np.array(all_probs)
    all_actuals = np.array(all_actuals)

    # ===== Primary: Brier Score =====
    result.brier_score = float(brier_score_loss(all_actuals, all_probs))

    # ===== Brier Baselines =====
    prior = float(y.mean())
    prior_probs = np.full_like(all_actuals, prior, dtype=np.float64)
    result.brier_baseline_prior = float(brier_score_loss(all_actuals, prior_probs))

    # Single-variable baseline (xFIP — kept_feature_names에 있을 때)
    if "xfip" in kept_feature_names:
        xfip_col = kept_feature_names.index("xfip")
        X_single = X[:, [xfip_col]].copy()
        try:
            single_probs_list = []
            single_actuals_list = []
            for train_idx, test_idx in skf.split(X_single, y):
                single_pipe = Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("model", LogisticRegression(class_weight="balanced", max_iter=5000)),
                ])
                single_pipe.fit(X_single[train_idx], y[train_idx])
                single_probs_list.extend(
                    single_pipe.predict_proba(X_single[test_idx])[:, 1].tolist()
                )
                single_actuals_list.extend(y[test_idx].tolist())
            if single_probs_list:
                result.brier_baseline_single = float(
                    brier_score_loss(np.array(single_actuals_list), np.array(single_probs_list))
                )
            else:
                result.brier_baseline_single = result.brier_baseline_prior
        except Exception as e:
            logger.warning(f"Single-variable baseline 실패: {e}")
            result.brier_baseline_single = result.brier_baseline_prior
    else:
        result.brier_baseline_single = result.brier_baseline_prior

    if result.brier_baseline_prior > 0:
        result.brier_skill_score = 1 - (result.brier_score / result.brier_baseline_prior)

    try:
        result.roc_auc = float(roc_auc_score(all_actuals, all_probs))
    except ValueError:
        result.roc_auc = 0.5

    predicted_labels = (all_probs >= 0.5).astype(int)
    if predicted_labels.sum() > 0:
        result.precision = float(precision_score(all_actuals, predicted_labels, zero_division=0))
    if all_actuals.sum() > 0:
        result.recall = float(recall_score(all_actuals, predicted_labels, zero_division=0))

    # Catastrophic Miss: GREEN/YELLOW 판정인데 실패
    high_conf_mask = all_probs >= 0.45
    if high_conf_mask.sum() > 0:
        failures_in_high_conf = (all_actuals[high_conf_mask] == 0).sum()
        result.catastrophic_miss_rate = float(failures_in_high_conf / high_conf_mask.sum())

    # 계수 안정성 (모든 폴드 같은 shape 보장됨)
    if fold_coefs:
        fold_coefs_arr = np.array(fold_coefs)
        for i, fname in enumerate(kept_feature_names):
            if i >= fold_coefs_arr.shape[1]:
                continue
            coefs_for_feature = fold_coefs_arr[:, i]
            sign_changes = sum(1 for j in range(1, len(coefs_for_feature))
                              if np.sign(coefs_for_feature[j]) != np.sign(coefs_for_feature[j-1]))
            result.coefficient_stability[fname] = {
                "mean": float(np.mean(coefs_for_feature)),
                "std": float(np.std(coefs_for_feature)),
                "sign_changes": sign_changes,
                "warning": sign_changes >= 2,
            }

    bins = np.linspace(0, 1, 11)
    for i in range(len(bins) - 1):
        mask = (all_probs >= bins[i]) & (all_probs < bins[i+1])
        if mask.sum() == 0:
            continue
        result.calibration_bins.append({
            "bin_low": float(bins[i]),
            "bin_high": float(bins[i+1]),
            "n": int(mask.sum()),
            "predicted_avg": float(all_probs[mask].mean()),
            "actual_avg": float(all_actuals[mask].mean()),
        })

    result.fold_results = fold_metrics
    return result


def loocv_validate(pitchers: list, random_state: int = 42) -> dict:
    """LOOCV 검증 (보조).

    각 선수를 hold-out하고 나머지로 학습 → 확률 예측.

    Returns:
        dict: brier, auc, predictions
    """
    temp = RiskEngine(random_state=random_state)
    X, y = temp._extract_xy(pitchers)
    if X is None or len(X) < 10:
        return {"error": "insufficient_data", "n": len(X) if X is not None else 0}

    # variance 0 컬럼 제거
    col_variance = X.std(axis=0)
    X = X[:, col_variance > 0]

    loo = LeaveOneOut()
    probs = []
    actuals = []

    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train = y[train_idx]

        n_pos_train = int((y_train == 1).sum())
        n_neg_train = int((y_train == 0).sum())
        inner_cv = min(3, n_pos_train, n_neg_train)
        if inner_cv < 2:
            continue

        try:
            model = build_calibrated_model(random_state=random_state, cv=inner_cv)
            model.fit(X_train, y_train)
            p = model.predict_proba(X_test)[0, 1]
            probs.append(float(p))
            actuals.append(int(y[test_idx][0]))
        except Exception:
            continue

    if not probs:
        return {"error": "all_folds_failed"}

    probs_arr = np.array(probs)
    actuals_arr = np.array(actuals)

    try:
        auc = float(roc_auc_score(actuals_arr, probs_arr))
    except ValueError:
        auc = 0.5

    return {
        "n": len(probs),
        "brier_score": float(brier_score_loss(actuals_arr, probs_arr)),
        "roc_auc": auc,
        "mean_prob_success": float(probs_arr[actuals_arr == 1].mean()) if (actuals_arr == 1).any() else 0,
        "mean_prob_failure": float(probs_arr[actuals_arr == 0].mean()) if (actuals_arr == 0).any() else 0,
    }
