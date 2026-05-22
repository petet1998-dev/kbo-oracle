"""
KBO Foreign Pitcher Risk-Control Governance Dashboard
======================================================
4-Tab Streamlit Dashboard:
  1. Scout Report — 개별 선수 위험 평가
  2. League Analysis — 전체 선수 비교
  3. Validation — 통계적 검증 (Brier/AUC/Calibration)
  4. Governance — 모델 관리자용 (계수 안정성, 위험 알림)

This system does not attempt to perfectly predict pitcher success.
It attempts to reduce avoidable investment failures under uncertainty.
"""
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from kbo_oracle import (
    Pitcher, build_pitcher_from_raw, parse_v2_text,
    load_dataset,
)
from risk_engine import (
    RiskEngine, validate_engine, loocv_validate,
    assign_risk_tier, FEATURE_COLUMNS,
)
from scout_report import format_prediction_text


# ============================================================
# Page Config
# ============================================================
st.set_page_config(
    page_title="KBO Risk Engine",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# Data Loading (cached)
# ============================================================
@st.cache_data(show_spinner=False)
def get_dataset():
    """데이터셋 로드 (캐시 활용).

    Streamlit Cloud 대응: 스크립트 파일 위치 기준으로 경로를 잡아
    작업 디렉토리가 달라도 dataset을 찾도록 함.

    참고: dataset_external.json(26명)은 raw 스탯이 충실해 학습 품질이
    가장 좋음. 86명 통합본은 60명이 raw 결측이라 AUC가 오히려 떨어짐
    (0.426 → 0.400). 따라서 raw 충실한 26명을 기본으로 사용.
    """
    base = Path(__file__).parent
    path = base / "dataset_external.json"
    if path.exists():
        return load_dataset(path)
    # fallback: 현재 디렉토리
    path = Path("dataset_external.json")
    if path.exists():
        return load_dataset(path)
    return []


@st.cache_resource(show_spinner="모델 학습 중...")
def get_trained_engine():
    """모델 학습 (캐시 활용, 한 번만 학습)"""
    dataset = get_dataset()
    if not dataset:
        return None
    try:
        engine = RiskEngine()
        engine.fit(dataset)
        return engine
    except Exception as e:
        st.error(f"모델 학습 실패: {e}")
        return None


@st.cache_data(show_spinner="검증 실행 중...")
def get_validation_result():
    """검증 결과 (캐시)"""
    dataset = get_dataset()
    if not dataset:
        return None
    try:
        return validate_engine(dataset)
    except Exception as e:
        return None


@st.cache_data(show_spinner="LOOCV 실행 중...")
def get_loocv_result():
    """LOOCV 결과 (캐시)"""
    dataset = get_dataset()
    if not dataset:
        return None
    try:
        return loocv_validate(dataset)
    except Exception:
        return None


# ============================================================
# Header / Sidebar
# ============================================================
st.title("⚾ KBO Foreign Pitcher Risk-Control Engine")
st.caption(
    "**This system does not predict success. It reduces avoidable investment failure under uncertainty.**"
    " · Final decision authority remains with human evaluators."
)

with st.sidebar:
    st.markdown("### 🏟 시스템 정보")
    dataset = get_dataset()
    st.metric("학습 데이터", f"{len(dataset)}명")

    engine = get_trained_engine()
    if engine:
        st.metric("Active Features", f"{len(engine._kept_feature_names)}")
        st.success("✅ 엔진 정상")
    else:
        st.error("❌ 엔진 로드 실패")

    st.markdown("---")
    page = st.radio(
        "**Navigation**",
        ["🎯 Scout Report", "🏟 League Analysis",
         "🧪 Validation", "⚙️ Governance"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.caption("Ver 15.2 · Data Governance Mode")


# ============================================================
# Tab 1: Scout Report
# ============================================================
def page_scout():
    st.header("🎯 개별 선수 Risk Assessment")
    st.caption("scout / 운영팀 / 단장 대상. 영입 후보 1명의 생존 확률 평가.")

    if not engine:
        st.error("엔진이 학습되지 않았습니다.")
        return

    # 입력 모드 선택
    input_mode = st.radio(
        "입력 방식",
        ["📋 기존 선수 선택", "✍️ 직접 입력 (V2 양식)"],
        horizontal=True,
    )

    target_pitcher = None

    if input_mode == "📋 기존 선수 선택":
        names = sorted([p.name for p in dataset])
        selected = st.selectbox("선수 선택", names)
        target_pitcher = next((p for p in dataset if p.name == selected), None)
    else:
        st.markdown(
            "**V2 양식 (Ver 15.2 - 16변수)**:\n\n"
            "`이름 / 데뷔년도 / 직전리그 / G / GS / IP / K9 / BB9 / HR9 / FIP / 평속 / IVB / BABIP / GB% / xFIP / CSW% → WAR`\n\n"
            "데이터 없는 필드는 `-` 입력. WAR 모르면 `-`."
        )
        v2_text = st.text_input(
            "한 줄 입력",
            value="신규 / 2026 / NPB / 25 / 22 / 130 / 8.5 / 2.5 / 0.8 / 3.5 / 93.0 / - / 0.295 / 45.2 / 3.40 / 28.5 → -",
            help="16변수 규격. 모르는 값은 '-' 입력.",
        )
        if st.button("분석", type="primary"):
            try:
                data = parse_v2_text(v2_text)
                target_pitcher = build_pitcher_from_raw(data)
            except Exception as e:
                st.error(f"파싱 실패: {e}")
                return

    if target_pitcher is None:
        return

    # === 예측 실행 ===
    try:
        pred = engine.predict(target_pitcher)
    except Exception as e:
        st.error(f"예측 실패: {e}")
        return

    st.markdown("---")

    # === Top-level Decision ===
    tier = pred.risk_tier
    tier_emoji = {"GREEN": "🟢", "YELLOW": "🟡", "RED": "🔴"}.get(tier["tier"], "⚪")

    c1, c2, c3 = st.columns(3)
    c1.metric("Survival Probability", f"{pred.survival_prob*100:.1f}%")
    c2.metric("Risk Tier", f"{tier_emoji} {tier['tier']}")
    c3.metric("Recommendation", tier["label"])

    if tier["tier"] == "GREEN":
        st.success(f"✅ **{tier['recommendation']}**")
    elif tier["tier"] == "YELLOW":
        st.warning(f"⚠️ **{tier['recommendation']}**")
    else:
        st.error(f"🚫 **{tier['recommendation']}**")

    if pred.confidence_warning:
        st.info(f"📢 **Confidence Warning**: {pred.confidence_warning}")

    # === 자동 통찰 (야구 언어 해석) ===
    from scout_report import generate_insights
    insights = generate_insights(pred, target_pitcher)
    if insights:
        st.markdown("### 💡 핵심 통찰")
        for ins in insights:
            t = ins["type"]
            if t == "positive":
                st.success(ins["text"])
            elif t == "negative":
                st.error(ins["text"])
            elif t == "warning":
                st.warning(ins["text"])
            else:  # insight
                st.info(ins["text"])

    # === Feature Contributions ===
    st.markdown("---")
    st.markdown("### 📊 Local Risk Decomposition")
    st.caption("선형 모델 계수 × scaled feature → 이 선수만의 위험 분해")

    pos_col, neg_col = st.columns(2)

    with pos_col:
        st.markdown("**🟢 긍정적 기여 (Survival ↑)**")
        positives = [c for c in pred.feature_contributions if c["direction"] == "긍정"]
        if positives:
            from scout_report import _format_value_clean
            df_pos = pd.DataFrame([
                {
                    "Feature": c["feature"],
                    "현재 값": _format_value_clean(c.get("raw_name", ""), c.get("value")),
                    "Contribution": round(c["contribution"], 3),
                }
                for c in positives
            ])
            st.dataframe(df_pos, use_container_width=True, hide_index=True)
        else:
            st.caption("(긍정적 기여 변수 없음)")

    with neg_col:
        st.markdown("**🔴 부정적 기여 (Risk ↑)**")
        negatives = [c for c in pred.feature_contributions if c["direction"] == "부정"]
        if negatives:
            from scout_report import _format_value_clean
            df_neg = pd.DataFrame([
                {
                    "Feature": c["feature"],
                    "현재 값": _format_value_clean(c.get("raw_name", ""), c.get("value")),
                    "Contribution": round(c["contribution"], 3),
                }
                for c in negatives
            ])
            st.dataframe(df_neg, use_container_width=True, hide_index=True)
        else:
            st.caption("(부정적 기여 변수 없음)")

    # === 유사 사례 분석 (Nearest Neighbors) ===
    st.markdown("---")
    st.subheader("🔍 유사 사례 분석")
    st.caption("학습 데이터에서 14개 변수 기준 가장 닮은 실제 외인을 찾아 그들의 KBO 결과를 보여줍니다.")
    similar = engine.find_similar_cases(target_pitcher, top_k=3)
    if similar:
        n_fail = sum(1 for c in similar if c["label"] == "Failure")
        sim_rows = []
        for c in similar:
            emoji = "🟢" if c["label"] == "Success" else "🔴"
            sim_rows.append({
                "닮은 선수": f"{emoji} {c['name']}",
                "직전리그": c.get("prev_league") or "—",
                "KBO WAR": f"{c['war']:+.2f}",
                "결과": "성공 (재계약)" if c["label"] == "Success" else "실패 (방출)",
                "거리": round(c["distance"], 2),
            })
        st.dataframe(pd.DataFrame(sim_rows), use_container_width=True, hide_index=True)

        # 종합 판정
        fail_ratio = n_fail / len(similar)
        if fail_ratio >= 0.67:
            st.warning(f"⚠️ **경고**: 가장 닮은 {len(similar)}명 중 {n_fail}명이 KBO에서 "
                       f"실패했습니다. 스탯이 좋아 보여도 유사 프로파일의 실패율이 높습니다.")
        elif fail_ratio <= 0.33:
            st.success(f"✅ 가장 닮은 {len(similar)}명 중 {len(similar)-n_fail}명이 "
                       f"성공했습니다. 긍정적 신호입니다.")
        else:
            st.info("🟡 유사 사례가 성공/실패로 갈립니다. 추가 변수로 판단이 필요합니다.")
    else:
        st.caption("유사 사례를 찾을 수 없습니다.")

    # === 입력 데이터 ===
    with st.expander("📋 사용된 입력 데이터"):
        if pred.used_features:
            df_used = pd.DataFrame([
                {"Feature": k, "Value": round(v, 3) if isinstance(v, float) else v}
                for k, v in pred.used_features.items()
            ])
            st.dataframe(df_used, use_container_width=True, hide_index=True)
        else:
            st.caption("입력 데이터 없음")


# ============================================================
# Tab 2: League Analysis
# ============================================================
def page_league():
    st.header("🏟 League-Wide Risk Assessment")
    st.caption("운영팀 / 단장 대상. 데이터셋 전체 선수 risk 비교.")

    if not engine or not dataset:
        st.error("데이터/엔진 없음")
        return

    # 모든 선수 예측
    rows = []
    for p in dataset:
        try:
            pred = engine.predict(p)
            actual = p.actual_war
            is_success = actual >= 2.0 if actual is not None else None
            predicted_success = pred.survival_prob >= 0.5
            hit = "✅" if is_success == predicted_success else "❌" if is_success is not None else "—"
            rows.append({
                "Pitcher": p.name,
                "League": p.prev_league or p.last_league,
                "Prob(%)": round(pred.survival_prob * 100, 1),
                "Tier": pred.risk_tier["tier"],
                "Actual WAR": round(actual, 2) if actual is not None else None,
                "Hit?": hit,
            })
        except Exception:
            continue

    df = pd.DataFrame(rows)

    # 필터
    c1, c2, c3 = st.columns(3)
    with c1:
        tier_filter = st.multiselect("Tier 필터",
            ["GREEN", "YELLOW", "RED"],
            default=["GREEN", "YELLOW", "RED"])
    with c2:
        league_options = sorted(df["League"].dropna().unique().tolist())
        league_filter = st.multiselect("리그 필터", league_options,
            default=league_options)
    with c3:
        sort_by = st.selectbox("정렬",
            ["Prob(%)", "Actual WAR", "Pitcher"])

    # 적용
    filtered = df[df["Tier"].isin(tier_filter) & df["League"].isin(league_filter)]
    filtered = filtered.sort_values(by=sort_by, ascending=False)

    # 메트릭
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total", len(filtered))
    m2.metric("🟢 GREEN", (filtered["Tier"] == "GREEN").sum())
    m3.metric("🟡 YELLOW", (filtered["Tier"] == "YELLOW").sum())
    m4.metric("🔴 RED", (filtered["Tier"] == "RED").sum())

    st.dataframe(filtered, use_container_width=True, hide_index=True, height=600)

    # === 정확도 요약 ===
    st.markdown("---")
    st.markdown("### 📈 데이터셋 내 모델 hit rate (참고용)")
    n_with_war = filtered["Actual WAR"].notna().sum()
    n_hit = (filtered["Hit?"] == "✅").sum()
    if n_with_war > 0:
        st.metric("Hit Rate", f"{n_hit / n_with_war * 100:.1f}%",
                 help="in-sample hit rate. 실제 일반화 성능은 Validation 탭 참조.")


# ============================================================
# Tab 3: Validation
# ============================================================
def page_validation():
    st.header("🧪 Statistical Validation")
    st.caption("정직성 검증. Brier Score / Calibration / Catastrophic Miss.")

    val = get_validation_result()
    if val is None:
        st.error("검증 실행 실패")
        return

    # === 핵심 지표 ===
    st.markdown("### 📊 Core Metrics (Stratified K-Fold)")
    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Brier Score", f"{val.brier_score:.4f}",
              delta=f"{val.brier_score - val.brier_baseline_prior:+.4f} vs prior",
              delta_color="inverse",
              help="확률 정직성. 낮을수록 좋음.")
    c2.metric("Brier Skill Score",
              f"{val.brier_skill_score:+.4f}",
              help="prior baseline 대비 개선도. >0이면 좋음.")
    c3.metric("ROC-AUC", f"{val.roc_auc:.4f}",
              help="0.5 = random, 1.0 = perfect")
    c4.metric("Catastrophic Miss",
              f"{val.catastrophic_miss_rate*100:.1f}%",
              help="Low Risk 판정 후 실제 실패 비율. 낮을수록 좋음.")

    c5, c6 = st.columns(2)
    c5.metric("Precision", f"{val.precision:.3f}")
    c6.metric("Recall", f"{val.recall:.3f}")

    # === 평가 메시지 ===
    if val.brier_skill_score > 0.05:
        st.success(f"✅ **모델이 prior baseline보다 우월** (BSS = {val.brier_skill_score:+.4f})")
    elif val.brier_skill_score > -0.05:
        st.warning(f"⚠️ **모델과 prior baseline이 비슷** (BSS = {val.brier_skill_score:+.4f}). 데이터 더 필요.")
    else:
        st.error(f"❌ **모델이 prior baseline보다 나쁨** (BSS = {val.brier_skill_score:+.4f}). 재설계 필요.")

    # === Calibration Bins ===
    st.markdown("---")
    st.markdown("### 🎯 Calibration Reliability")
    st.caption("70% 예측 확률이면 실제로 70% 생존해야 정직.")

    if val.calibration_bins:
        calib_df = pd.DataFrame(val.calibration_bins)
        calib_df["error"] = abs(calib_df["predicted_avg"] - calib_df["actual_avg"])
        st.dataframe(calib_df[["bin_low", "bin_high", "n",
                                "predicted_avg", "actual_avg", "error"]].round(4),
                     use_container_width=True, hide_index=True)
    else:
        st.caption("Calibration 데이터 없음")

    # === LOOCV ===
    st.markdown("---")
    st.markdown("### 🔁 LOOCV (Leave-One-Out Cross Validation)")
    loo = get_loocv_result()
    if loo and "error" not in loo:
        l1, l2, l3, l4 = st.columns(4)
        l1.metric("LOOCV n", loo["n"])
        l2.metric("Brier", f"{loo['brier_score']:.4f}")
        l3.metric("ROC-AUC", f"{loo['roc_auc']:.4f}")
        gap = loo["mean_prob_success"] - loo["mean_prob_failure"]
        l4.metric("Separation",
                  f"{gap*100:+.1f}%p",
                  help="Success - Failure 평균 확률 차이")
    else:
        st.caption("LOOCV 결과 없음")


# ============================================================
# Tab 4: Governance (Private — Model Admin)
# ============================================================
def page_governance():
    st.header("⚙️ Governance Dashboard")
    st.caption("**🔒 PRIVATE** — 모델 관리자용. 계수 안정성, 데이터 품질, 경고.")

    val = get_validation_result()
    if val is None:
        st.error("검증 실행 실패")
        return

    # === Coefficient Stability ===
    st.markdown("### 🧮 Coefficient Stability (Drift Tracking)")
    st.caption("Fold마다 계수가 안정적인가? sign changes ≥ 2 = 경고")

    if val.coefficient_stability:
        coef_rows = []
        for fname, stats in val.coefficient_stability.items():
            coef_rows.append({
                "Feature": fname,
                "Mean": round(stats["mean"], 4),
                "Std": round(stats["std"], 4),
                "Sign Changes": stats["sign_changes"],
                "Warning": "⚠️" if stats["warning"] else "✅",
            })
        coef_df = pd.DataFrame(coef_rows)
        coef_df = coef_df.reindex(coef_df["Mean"].abs().sort_values(ascending=False).index)
        st.dataframe(coef_df, use_container_width=True, hide_index=True)

        n_warning = sum(1 for r in coef_rows if r["Warning"] == "⚠️")
        if n_warning > 0:
            st.warning(f"⚠️ **{n_warning}개 feature 계수 부호가 불안정합니다.** 데이터 더 필요 또는 변수 재검토 필요.")
        else:
            st.success("✅ 모든 feature 계수가 안정적입니다.")

    # === Fold-level Brier ===
    st.markdown("---")
    st.markdown("### 📋 Fold-Level Performance")
    if val.fold_results:
        fold_df = pd.DataFrame(val.fold_results)
        st.dataframe(fold_df, use_container_width=True, hide_index=True)

    # === Data Quality ===
    st.markdown("---")
    st.markdown("### 📊 Data Quality")
    if engine:
        all_features = FEATURE_COLUMNS
        used_features = engine._kept_feature_names
        unused = [f for f in all_features if f not in used_features]
        missing_indicators = [f for f in used_features if "__missing" in f]

        c1, c2, c3 = st.columns(3)
        c1.metric("총 변수 정의", len(all_features))
        c2.metric("실제 사용", len([f for f in used_features if "__missing" not in f]))
        c3.metric("결측 indicator", len(missing_indicators))

        if unused:
            st.warning(f"⚠️ **사용 안 된 변수** (전부 결측): {', '.join(unused)}")

    # === System Settings ===
    st.markdown("---")
    st.markdown("### 🔧 System Settings")
    settings_df = pd.DataFrame([
        {"Setting": "Model", "Value": "LogisticRegression (elasticnet)"},
        {"Setting": "Calibration", "Value": "sigmoid (Platt scaling)"},
        {"Setting": "Validation", "Value": "Stratified 5-Fold + LOOCV"},
        {"Setting": "Class Weight", "Value": "balanced"},
        {"Setting": "Imputation", "Value": "median + missing indicators"},
        {"Setting": "Risk Tiers", "Value": "GREEN ≥0.70 / YELLOW ≥0.45 / RED <0.45"},
        {"Setting": "Survival Target", "Value": "actual_war ≥ 2.0"},
    ])
    st.dataframe(settings_df, use_container_width=True, hide_index=True)

    # === Honest Limitations ===
    st.markdown("---")
    st.markdown("### ⚠️ Honest Limitations")
    st.info(
        "1. **Small-N (~26명)**: 통계적 신뢰구간 넓음\n"
        "2. **KBO 트래킹 비공개**: IVB, KBO 환경 보정은 추정\n"
        "3. **Temporal Validation 불가**: 연도 분포 한계로 Stratified K-Fold로 대체\n"
        "4. **Catastrophic Miss > 30%**: GREEN/YELLOW 판정 일부가 실패\n"
        "5. **신뢰구간 외삽 위험**: 학습 분포 외 케이스는 신뢰도 낮음"
    )


# ============================================================
# Router
# ============================================================
if page.startswith("🎯"):
    page_scout()
elif page.startswith("🏟"):
    page_league()
elif page.startswith("🧪"):
    page_validation()
elif page.startswith("⚙️"):
    page_governance()
