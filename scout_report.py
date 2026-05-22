"""
Scout Report Generator (Risk Governance Style)

기존 휴리스틱 리포트(Failure Score)를 대체하는 확률 기반 리포트.

핵심 출력:
  - Survival Probability (%)
  - Risk Tier (GREEN / YELLOW / RED)
  - Confidence Warning
  - Local Feature Contributions (야구 언어로 번역)

This system does not attempt to perfectly predict pitcher success.
It attempts to reduce avoidable investment failures under uncertainty.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from risk_engine import RiskEngine, ScoutPrediction


def format_similar_cases(engine, pitcher, top_k: int = 3) -> str:
    """유사 사례 분석을 마크다운으로 생성.

    학습 데이터에서 가장 닮은 실제 외인을 찾아 그들의 KBO 결과를 보여준다.
    "스탯은 좋아 보여도 닮은 선수가 망했다" 같은 통찰을 제공.

    Args:
        engine: fit된 RiskEngine
        pitcher: 평가 대상 Pitcher
        top_k: 표시할 유사 선수 수

    Returns:
        마크다운 문자열 (유사 사례 없으면 빈 문자열)
    """
    cases = engine.find_similar_cases(pitcher, top_k=top_k)
    if not cases:
        return ""

    rows = []
    n_fail = 0
    for c in cases:
        emoji = "🟢" if c["label"] == "Success" else "🔴"
        if c["label"] == "Failure":
            n_fail += 1
        result = f"성공 (재계약)" if c["label"] == "Success" else "실패 (방출)"
        rows.append(
            f"| {emoji} {c['name']} | {c.get('prev_league') or '—'} | "
            f"{c['war']:+.2f} | {result} | {c['distance']:.2f} |"
        )
    table = "\n".join(rows)

    # 종합 판정
    fail_ratio = n_fail / len(cases)
    if fail_ratio >= 0.67:
        verdict = (f"⚠️ **경고**: 가장 닮은 {len(cases)}명 중 {n_fail}명이 KBO에서 "
                   f"실패했습니다. 스탯이 좋아 보여도 유사 프로파일의 실패율이 높습니다.")
    elif fail_ratio <= 0.33:
        verdict = (f"✅ 가장 닮은 {len(cases)}명 중 {len(cases)-n_fail}명이 성공했습니다. "
                   f"긍정적 신호입니다.")
    else:
        verdict = (f"🟡 유사 사례가 성공/실패로 갈립니다. 추가 변수로 판단이 필요합니다.")

    return f"""
## 🔍 유사 사례 분석 (Nearest Neighbors)

학습 데이터에서 14개 변수 기준 가장 닮은 실제 외인:

| 닮은 선수 | 직전리그 | KBO WAR | 결과 | 거리 |
|---|---|---|---|---|
{table}

{verdict}
"""


def format_prediction_text(prediction: ScoutPrediction, quick: bool = False,
                            engine=None, pitcher=None) -> str:
    """예측 결과를 텍스트로 포매팅 (CLI 출력용).

    Args:
        prediction: ScoutPrediction 인스턴스
        quick: True면 4줄 요약, False면 전체 리포트
        engine: (선택) fit된 RiskEngine — 있으면 유사 사례 분석 추가
        pitcher: (선택) 평가 대상 Pitcher — engine과 함께 필요

    Returns:
        str: 출력 가능한 텍스트
    """
    if quick:
        return _format_quick(prediction)
    report = _format_full(prediction)
    # 자동 통찰 추가
    if engine is not None and pitcher is not None:
        insights = generate_insights(prediction, pitcher)
        if insights:
            lines = ["\n## 💡 핵심 통찰\n"]
            for ins in insights:
                lines.append(f"- {ins['text']}")
            insight_md = "\n".join(lines) + "\n"
            report = report.replace("## 📊 Local Risk Decomposition",
                                     insight_md + "\n## 📊 Local Risk Decomposition")
        # 유사 사례 분석 추가
        similar = format_similar_cases(engine, pitcher)
        if similar:
            report = report.replace("## 📋 입력 데이터",
                                     similar + "\n## 📋 입력 데이터")
    return report


def _format_quick(p: ScoutPrediction) -> str:
    """Quick 모드 — 면접 시연용 4줄"""
    tier = p.risk_tier
    prob_str = f"{p.survival_prob*100:.1f}%"
    lines = [
        f"\n[Risk Assessment] {p.name}",
        f"  생존 확률: {prob_str} | Tier: {tier['tier']} ({tier['label']})",
        f"  추천: {tier['recommendation']}",
    ]
    if p.confidence_warning:
        lines.append(f"  ⚠️ {p.confidence_warning}")
    return "\n".join(lines)


def _format_value_clean(raw_name: str, value) -> str:
    """Ver 15.2: contribution 테이블의 '현재 값' 컬럼 포맷.

    - 백분율 변수: 0.83333 → 83.3% 또는 raw value가 이미 % 단위면 그대로
    - 이진 플래그 (1.0/0.0): 정상/위험 언어로 변환
    - 결측: '—' 표시
    - 일반 실수: 소수점 정리

    Args:
        raw_name: feature 원본 이름 (예: 'gb_pct', 'is_pcl', 'avg_velo')
        value: feature 값 (NaN/None 가능)

    Returns:
        사람이 읽기 좋은 문자열
    """
    if value is None:
        return "—"

    try:
        v = float(value)
        if v != v:  # NaN
            return "—"
    except (TypeError, ValueError):
        return str(value)

    # 결측 indicator 처리
    if raw_name.endswith("__missing") or raw_name.startswith("is_"):
        if raw_name in ("is_pcl", "is_il", "is_asian_league"):
            return "해당 ✓" if v >= 0.5 else "비해당"
        if "missing" in raw_name:
            return "결측 ⚠️" if v >= 0.5 else "정상"
        return "Y" if v >= 0.5 else "N"

    # 백분율 변수 (이름에 _pct 또는 _ratio 포함)
    if "_pct" in raw_name:
        # raw 값이 이미 % 단위 (예: gb_pct=50.6) 또는 0~1 (예: gs_ratio=0.7)
        if v <= 1.0:
            return f"{v*100:.1f}%"
        return f"{v:.1f}%"
    if "_ratio" in raw_name:
        return f"{v*100:.1f}%"  # 0~1 가정

    # 이닝, 일수 등 정수처럼 보이는 변수
    if raw_name in ("il_days_past_2y", "pitch_mix_count", "age_at_debut"):
        return f"{int(v)}"

    # 일반 실수
    return f"{v:.2f}"


def _format_full(p: ScoutPrediction) -> str:
    """Full Markdown 리포트 (Ver 15.2: 현재 값 컬럼 추가)"""
    tier = p.risk_tier
    emoji = {"GREEN": "🟢", "YELLOW": "🟡", "RED": "🔴"}.get(tier["tier"], "⚪")

    positives = [c for c in p.feature_contributions if c["direction"] == "긍정"]
    negatives = [c for c in p.feature_contributions if c["direction"] == "부정"]

    # Ver 15.2: 3컬럼 (Feature | 현재 값 | Contribution)
    pos_table = "\n".join(
        f"| {c['feature']} | {_format_value_clean(c.get('raw_name', ''), c.get('value'))} | {c['contribution']:+.3f} |"
        for c in positives
    ) or "| (없음) | — | — |"
    neg_table = "\n".join(
        f"| {c['feature']} | {_format_value_clean(c.get('raw_name', ''), c.get('value'))} | {c['contribution']:+.3f} |"
        for c in negatives
    ) or "| (없음) | — | — |"

    warning_section = ""
    if p.confidence_warning:
        warning_section = f"""
## ⚠️ Confidence Warning
{p.confidence_warning}
"""

    used_features_table = "\n".join(
        f"| {k} | {v} |"
        for k, v in p.used_features.items()
    )

    report = f"""# 🎯 RISK ASSESSMENT REPORT — {p.name}

> 생성: {datetime.now().strftime('%Y-%m-%d %H:%M')}
> KBO Foreign Pitcher Risk-Control Engine (Ver 15.2)

## {emoji} Top-Level Decision

| 항목 | 값 |
|---|---|
| **Survival Probability** | **{p.survival_prob*100:.1f}%** |
| **Risk Tier** | **{tier['tier']}** ({tier['label']}) |
| **Recommendation** | {tier['recommendation']} |
{warning_section}
## 📊 Local Risk Decomposition

### 긍정적 기여 (Survival ↑)
| Feature | 현재 값 | Contribution |
|---|---|---|
{pos_table}

### 부정적 기여 (Risk ↑)
| Feature | 현재 값 | Contribution |
|---|---|---|
{neg_table}

## 📋 입력 데이터
| Feature | Value |
|---|---|
{used_features_table}

---
> ⚠️ This assessment supports, not replaces, scouting judgment.
> Final decision authority remains with human evaluators.
"""
    return report


def render_league_table(engine: RiskEngine, pitchers: list,
                        sort_by: str = "survival_prob") -> str:
    """리그 전체 선수 테이블 (CLI 출력용).

    Args:
        engine: 학습된 RiskEngine
        pitchers: Pitcher 리스트
        sort_by: 정렬 기준 ('survival_prob' / 'name')

    Returns:
        str: 텍스트 테이블
    """
    predictions = []
    for p in pitchers:
        try:
            pred = engine.predict(p)
            predictions.append((p, pred))
        except Exception as e:
            continue

    if sort_by == "survival_prob":
        predictions.sort(key=lambda x: -x[1].survival_prob)
    else:
        predictions.sort(key=lambda x: x[0].name)

    lines = [
        "\n" + "="*78,
        "🏟  LEAGUE-WIDE RISK ASSESSMENT",
        "="*78,
        f"{'Pitcher':<25} {'Prob':>7} {'Tier':<8} {'Actual WAR':>11} {'Hit?':<5}",
        "-"*78,
    ]
    for pitcher, pred in predictions:
        actual = pitcher.actual_war
        is_success = actual >= 2.0 if actual is not None else None
        predicted_success = pred.survival_prob >= 0.5
        hit = "✅" if is_success == predicted_success else "❌" if is_success is not None else "—"
        actual_str = f"{actual:+.2f}" if actual is not None else "  —  "
        lines.append(
            f"{pitcher.name:<25} {pred.survival_prob*100:>6.1f}% "
            f"{pred.risk_tier['tier']:<8} {actual_str:>11} {hit:<5}"
        )
    lines.append("="*78)
    return "\n".join(lines)


def generate_insights(prediction, pitcher) -> list:
    """예측 결과에서 '야구 언어' 통찰을 자동 생성.

    단순 숫자가 아니라 "구속이 빠른데 왜 마이너스인지" 같은
    스카우트가 이해하는 해석을 생성한다.

    Args:
        prediction: ScoutPrediction
        pitcher: 평가 대상 Pitcher

    Returns:
        [{"type": "positive"/"negative"/"warning"/"insight", "text": str}, ...]
    """
    insights = []
    contribs = {c["raw_name"]: c for c in prediction.feature_contributions
                if "raw_name" in c}

    velo = getattr(pitcher, "raw_velo", None) or getattr(pitcher, "avg_velo", None)
    k9 = getattr(pitcher, "raw_k9", None)
    bb9 = getattr(pitcher, "raw_bb9", None)
    prev_league = getattr(pitcher, "prev_league", "")
    gs = getattr(pitcher, "gs", None)
    g = getattr(pitcher, "g", None)

    # 1. 구속 역설 (빠른데 마이너스인 경우)
    velo_contrib = contribs.get("avg_velo")
    if velo and velo >= 94 and velo_contrib and velo_contrib["contribution"] < 0:
        insights.append({
            "type": "insight",
            "text": (f"⚡ 구속 {velo:.1f}mph는 빠르지만 모델은 이를 **위험 신호**로 봅니다. "
                     f"학습 데이터에서 '구속만 빠르고 KBO에서 실패한' 외인이 많았기 때문입니다. "
                     f"구속이 곧 성공은 아니라는 패턴입니다.")
        })

    # 2. K-BB 진단 (Ver 15.2.1: 모델 실제 계수와 일치)
    # 주의: n=26 데이터에선 K-BB가 생존을 거의 예측 못 함 (상관 -0.02).
    # 야구 일반론("K-BB 높으면 좋다")과 달리, 우리 데이터에선 리그 환경에
    # 가려져 신호가 약함. 그래서 K-BB는 "참고용"으로만 제시.
    if k9 is not None and bb9 is not None:
        kbb = k9 - bb9
        if kbb >= 5.5:
            insights.append({"type": "insight",
                "text": (f"📊 K-BB {kbb:.1f}: 일반적으로 우수한 수치지만, 우리 데이터(n=26)에선 "
                         f"리그 환경에 가려져 K-BB 단독 예측력이 약합니다. 직전 리그를 함께 봐야 합니다.")})
        elif kbb < 3.0:
            insights.append({"type": "negative",
                "text": f"🔴 K-BB {kbb:.1f}: 탈삼진 대비 볼넷이 많습니다. 절대 수치로도 낮은 편입니다."})

    # 3. 리그 환경
    if prev_league == "AAA-IL":
        insights.append({"type": "warning",
            "text": "⚠️ AAA-IL은 투수친화 리그라 스탯이 부풀려집니다. 같은 수치라도 PCL/MLB보다 할인해서 봐야 합니다."})
    elif prev_league == "AAA-PCL":
        insights.append({"type": "insight",
            "text": "📊 AAA-PCL은 타자친화 리그라, 이 정도 스탯이면 실제 구위는 더 좋을 수 있습니다."})
    elif prev_league in ("NPB", "NPB-2"):
        insights.append({"type": "positive",
            "text": "✅ NPB 출신은 변화구·제구가 KBO에 잘 적응하는 편입니다. 환경 프리미엄이 있습니다."})

    # 4. 스윙맨 / 풀타임 선발 검증
    if gs is not None and g is not None and g > 0:
        gs_ratio = gs / g
        if gs_ratio < 0.7:
            insights.append({"type": "warning",
                "text": (f"⚠️ 직전 시즌 {g}경기 중 {gs}선발(선발 비율 {gs_ratio*100:.0f}%). "
                         f"스윙맨 역할이라 KBO 풀타임 선발(130이닝+) 검증이 안 됐습니다.")})

    # 5. IP 표본
    ip = getattr(pitcher, "prev_ip", None) or getattr(pitcher, "ip", None)
    if ip is not None and ip < 80:
        insights.append({"type": "warning",
            "text": f"🔴 직전 시즌 {ip:.0f}이닝: 풀시즌(130+) 대비 표본이 부족해 평가 신뢰도가 낮습니다."})

    return insights
