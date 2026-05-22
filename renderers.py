"""
KBO Oracle Renderer Module (Ver 15.1)
======================================

UI 렌더링을 백엔드 로직과 분리. Separation of Concerns 적용.

이 모듈은 백엔드 계산 결과(dict, dataclass)를 받아서:
  - Markdown 문자열
  - CLI 텍스트
  - 단순 ASCII 차트

로 변환합니다. 백엔드 로직에는 출력 형식 의존성이 없어야 함.

설계 원칙:
- 모든 함수는 pure: 입력 → 출력 (side effect 없음)
- 백엔드 데이터 구조만 의존, Streamlit이나 file IO 없음
- 테스트 가능성을 위해 단위 함수로 분리
"""
from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Any


# ============================================================
# Ver 15.2: None-safe formatting helpers
# ============================================================
def fmt_pct(value, dash: str = "—") -> str:
    """실수형 → '12.3%' / None → '—'"""
    if value is None:
        return dash
    try:
        return f"{float(value):.1f}%"
    except (TypeError, ValueError):
        return dash


def fmt_float(value, prec: int = 1, suffix: str = "", dash: str = "—") -> str:
    """실수형 → '12.3' / None → '—'"""
    if value is None:
        return dash
    try:
        return f"{float(value):.{prec}f}{suffix}"
    except (TypeError, ValueError):
        return dash


def fmt_signed(value, prec: int = 1, dash: str = "—") -> str:
    """실수형 → '+1.5' / None → '—'"""
    if value is None:
        return dash
    try:
        return f"{float(value):+.{prec}f}"
    except (TypeError, ValueError):
        return dash


def fmt_int(value, dash: str = "—") -> str:
    """정수형 → '15' / None → '—'"""
    if value is None:
        return dash
    try:
        return str(int(value))
    except (TypeError, ValueError):
        return dash


# ============================================================
# Scout Report (Ver 14.x 휴리스틱 — 회고용)
# ============================================================
def render_scout_report(pitcher, guardrails: dict, failure_score: dict,
                         decay: int, simulation: dict, label: str,
                         contract: int) -> str:
    """Ver 14.x Scouting Report (Markdown).

    Args:
        pitcher: Pitcher 인스턴스
        guardrails: ScoutEngine.guardrails() 결과
        failure_score: ScoutEngine.failure_score() 결과
        decay: 0~100 time decay score
        simulation: ScoutEngine.season_simulation() 결과
        label: 최종 라벨
        contract: 권장 계약가

    Returns:
        Markdown 문자열
    """
    p = pitcher
    gr = guardrails
    fs = failure_score
    sim = simulation

    ck = lambda b: "✅" if b else "❌"
    decay_info = _decay_status(decay)
    failed = [k for k, v in gr.items() if not v]

    pivot_section = _render_pivot_section(fs)
    sim_table = _render_simulation_table(sim)
    physical = _render_physical_section(p)

    return f"""# 🎯 SCOUTING REPORT — {p.name}

> 생성: {datetime.now().strftime('%Y-%m-%d %H:%M')} · KBO Scouting AI OS

## 🆔 IDENTITY
- **나이 / 투타**: {p.age or '—'} / {p.throws or '—'}
- **최근 소속**: {p.last_league}
- **추정 계약가**: {f"${p.contract_usd:,}" if p.contract_usd else '—'}

## 🛡 GUARDRAIL CHECK
| 가드레일 | 결과 | 기준 |
|---|---|---|
| Command Floor | {ck(gr['command'])} | Zone% ≥ 50 (현재 {fmt_pct(p.zone_pct)}) |
| Velocity Decline | {ck(gr['velocity'])} | 전성기 대비 −2mph 이내 (현재 {fmt_signed(p.velo_trend)}) |
| Durability | {ck(gr['durability'])} | 최근 2년 IL <60일 (현재 {fmt_int(p.il_days_2y)}일) |
| Pitch Mix | {ck(gr['pitch_mix'])} | 결정구 3개+ (현재 {len(p.pitch_types) if p.pitch_types else 0}개) |
{f"### ⚠️ 가드레일 실패: **{', '.join(failed)}**" if failed else "### ✅ 모든 가드레일 통과"}
{pivot_section}
## 📊 FAILURE SCORE: **{fs['total']:.1f} / 100**

| 항목 | 점수 |
|---|---|
| Command Risk | {fs['command_risk']:.1f} / 25 |
| Stuff Decay Risk | {fs['stuff_decay_risk']:.1f} / 25 |
| Adaptation Risk | {fs['adaptation_risk']:.1f} / 20 |
| Durability Risk | {fs['durability_risk']:.1f} / 15 |
| Mental/Off-field | {fs['mental_risk']:.1f} / 15 |
| Proven Asset Bonus | −{fs['proven_bonus']:.1f} |
| **Success Pivot Bonus** | **−{fs['pivot_bonus']:.1f}** |

## ⏳ TIME DECAY RISK: {decay_info['emoji']} **{decay} / 100 ({decay_info['level']})**
- 예상 패턴: {decay_info['message']}

{sim_table}

{physical}

## 🔥 FINAL LABEL: **[{label}]**
- **권장 계약금**: ${contract:,}
- **한 줄 평**: {_one_liner_for_label(label, fs['total'], decay, fs['pivot_info']['activated'])}

---
> ⚠️ 데이터 한계: KBO 트래킹 비공개 → MLB Statcast 기반 추정.
> 💡 Ver 15.0부터는 ML 기반 Risk Engine 권장 (`risk` 명령어)
"""


def _decay_status(decay: int) -> dict:
    """Time decay → 시각적 표시"""
    if decay <= 30:
        return {"emoji": "🟢", "level": "LOW", "message": "시즌 완주 가능"}
    elif decay <= 60:
        return {"emoji": "🟡", "level": "MID", "message": "7~8월 ERA 급등 우려"}
    else:
        return {"emoji": "🔴", "level": "HIGH", "message": "5~6월부터 붕괴 시작"}


def _render_pivot_section(fs: dict) -> str:
    """Success Pivot 발동 시 출력 (안 발동이면 빈 문자열)"""
    if not fs['pivot_info']['activated']:
        return ""
    return f"""
## 🎯 SUCCESS PIVOT 발동 ⚡
약점이 있으나 보완 무기로 상쇄 가능한 케이스로 판정.
- **트리거**: {' / '.join(fs['pivot_info']['triggers'])}
- **약점 감지**: {'구속 하락 또는 92mph 미만' if fs['pivot_info']['weakness_detected'] else 'N/A'}
- **FS 보너스**: −{fs['pivot_bonus']:.1f}
"""


def _render_simulation_table(sim: dict) -> str:
    """Season simulation 테이블"""
    pa, pb, pc, ss = sim['phase_a'], sim['phase_b'], sim['phase_c'], sim['season']
    range_source = sim.get('range_source', 'heuristic')
    interp = ('회귀 잔차 5–95% 백분위 기반'
              if 'bootstrap' in range_source else 'FS 분산 기반 추정')

    return f"""## 🏟 SEASON SIMULATION — Range 출력 (체감 변동폭)

| 구간 | Best ERA | 기대 ERA | Worst ERA | WHIP | K/9 |
|---|---|---|---|---|---|
| 🌸 {pa['label']} | {pa['era']['best']} | **{pa['era']['expected']}** | {pa['era']['worst']} | {pa['whip']:.2f} | {pa['k9']:.1f} |
| ☀️ {pb['label']} | {pb['era']['best']} | **{pb['era']['expected']}** | {pb['era']['worst']} | {pb['whip']:.2f} | {pb['k9']:.1f} |
| 🍁 {pc['label']} | {pc['era']['best']} | **{pc['era']['expected']}** | {pc['era']['worst']} | {pc['whip']:.2f} | {pc['k9']:.1f} |
| **시즌 평균** | **{ss['era']['best']}** | **{ss['era']['expected']}** | **{ss['era']['worst']}** | {ss['whip']:.2f} | IP {ss['ip_low']:.0f}–{ss['ip_high']:.0f} |

→ 변동폭 출처: **{range_source}**. {interp}."""


def _render_physical_section(p) -> str:
    """Physical & Technical 섹션 (Ver 15.2: None 안전 + 신규 변수)"""
    # KBO_IVB_BONUS = 1.08 (도메인 추정값, 변경 근거 없으므로 유지)
    ivb_adj_str = "—"
    if p.ivb_inch is not None:
        try:
            ivb_adj_str = f"{p.ivb_inch * 1.08:.1f}"
        except (TypeError, ValueError):
            pass

    pitch_count = len(p.pitch_types) if p.pitch_types else 0
    pitch_str = ", ".join(p.pitch_types) if p.pitch_types else "—"

    # Ver 15.2 NEW: 고급 세이버 지표 (있으면 출력)
    advanced_lines = []
    if getattr(p, 'csw_pct', None) is not None:
        advanced_lines.append(f"- **CSW%**: {fmt_pct(p.csw_pct)} (루킹 + 헛스윙)")
    if getattr(p, 'gb_pct', None) is not None:
        advanced_lines.append(f"- **GB%**: {fmt_pct(p.gb_pct)} (땅볼 유발)")
    if getattr(p, 'xfip', None) is not None:
        advanced_lines.append(f"- **xFIP**: {fmt_float(p.xfip, prec=2)} (정규화 FIP)")
    advanced_section = "\n".join(advanced_lines)
    if advanced_section:
        advanced_section = "\n\n### 🆕 고급 세이버 지표 (Ver 15.2)\n" + advanced_section

    return f"""## 📡 PHYSICAL & TECHNICAL
- **평속 / 최고**: {fmt_float(p.avg_velo)} / {fmt_float(p.peak_velo)} mph
- **구속 추세**: {fmt_signed(p.velo_trend)} mph (전성기 대비)
- **Zone% / Whiff% / Chase%**: {fmt_pct(p.zone_pct)} / {fmt_pct(p.whiff_pct)} / {fmt_pct(p.chase_pct)}
- **IVB**: {fmt_float(p.ivb_inch)}" → KBO 환경 보정 {ivb_adj_str}"
- **구종 ({pitch_count}개)**: {pitch_str}
- **주력구 의존도**: {fmt_float(p.primary_pitch_pct, prec=0, suffix='%')}{advanced_section}"""


def _one_liner_for_label(label: str, fs_total: float, decay: int,
                          pivot: bool) -> str:
    """라벨에 따른 한 줄 평"""
    if label == "ACE BET":
        return f"가드레일 통과 + FS {fs_total:.0f} + Decay {decay}. 1선발 베팅 권고."
    if "Pivot" in label:
        return f"가드레일 일부 fail이지만 보완 무기로 생존 가능. 한국형 적응 기대."
    if label == "VALUE SIGN":
        return f"FS {fs_total:.0f}. 리스크 대비 가성비 우수."
    if label == "ROTATION FILLER":
        return f"FS {fs_total:.0f}. 안정성은 있으나 폭발력 부족."
    return f"가드레일 또는 Failure Score(={fs_total:.0f}) 미달. 영입 비추천."


# ============================================================
# Backtest Report
# ============================================================
def render_backtest_report(result: dict) -> str:
    """회귀 백테스트 결과 → Markdown.

    Args:
        result: BacktestEngine.run() 결과

    Returns:
        Markdown 문자열
    """
    r = result['pearson_r']
    grade, interp = _grade_pearson_r(r)
    gm = result['group_means']

    g_rows = "\n".join(
        f"| {k} | {v} |" for k, v in sorted(gm.items(), key=lambda x: -x[1])
    )
    out_rows = "\n".join(
        f"| {o['name']} | {o['type']} | {o['score']:.1f} | "
        f"{o['actual_war']:+.1f} | {o['predicted_war']:+.2f} | {o['residual']:+.2f} |"
        for o in result['outliers']
    )
    scatter = ascii_scatter(result['rows'])
    w = result['weights_used']
    wstr = f"S {w['stuff']:.2f}·C {w['command']:.2f}·X {w['context']:.2f}·D {w['durability']:.2f}"

    return f"""# 🧪 BACKTEST REPORT (n={result['n']})

> {datetime.now().strftime('%Y-%m-%d %H:%M')} · KBO Scouting AI OS
> 가중치: {wstr}

## 📈 핵심 지표
| Metric | Value | 판정 |
|---|---|---|
| **Pearson r** | **{result['pearson_r']:+.4f}** | {grade} |
| Spearman ρ | {result['spearman_r']:+.4f} | — |
| R² | {result['r_squared']:.4f} | {result['r_squared']*100:.1f}% 설명 |
| 회귀식 | WAR ≈ {result['slope']:.4f}×Score + ({result['intercept']:.2f}) | — |
| Success vs Failure 분리력 | {result['separation']:.2f}점 | {'✅' if result['separation']>=15 else '⚠️ 부족'} |

→ {interp}

## 🎯 그룹별 평균
| 그룹 | 평균 Score |
|---|---|
{g_rows}

## 🔍 가장 빗나간 선수 TOP 5
| Player | Type | Score | Actual | Predicted | Residual |
|---|---|---|---|---|---|
{out_rows}

## 📊 Scatter
```
{scatter}
```
"""


def _grade_pearson_r(r: float) -> tuple:
    """Pearson r 등급 + 해석"""
    if r >= 0.7:
        return "🏆 S급", "**가중치 밸런스 우수.**"
    if r >= 0.5:
        return "✅ 합리적", "**합리적.**"
    if r >= 0.3:
        return "⚠️ 약함", "**약한 신호.**"
    return "❌ 위험", "**위험. 재설계 필요.**"


# ============================================================
# ASCII Scatter (재사용 가능)
# ============================================================
def ascii_scatter(rows: List[dict], width: int = 60, height: int = 18) -> str:
    """간단한 ASCII scatter plot.

    Args:
        rows: [{'score', 'actual_war', 'type'} 키 포함] dict 리스트
        width: 가로 너비
        height: 세로 높이

    Returns:
        scatter plot 문자열
    """
    if not rows:
        return "(no data)"
    xs = [r['score'] for r in rows]
    ys = [r['actual_war'] for r in rows]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)

    g = [[' '] * width for _ in range(height)]
    for r in rows:
        px = int((r['score'] - xmin) / (xmax - xmin + 1e-9) * (width - 1))
        py = height - 1 - int(
            (r['actual_war'] - ymin) / (ymax - ymin + 1e-9) * (height - 1)
        )
        ch = {'Success': 'S', 'Failure': 'F', 'Marginal': 'M'}.get(r['type'], '?')
        g[py][px] = ch

    return '\n'.join(
        [f"WAR↑ max={ymax:+.1f}"]
        + [''.join(r) for r in g]
        + [f"     min={ymin:+.1f}",
           f"Score: {xmin:.1f} ────→ {xmax:.1f}",
           "S=Success M=Marginal F=Failure"]
    )


# ============================================================
# Classification Report (sklearn 결과 → 텍스트)
# ============================================================
def render_classification_summary(cm: dict, quick: bool = False) -> str:
    """분류 평가 결과 → 텍스트.

    Args:
        cm: confusion_matrix() 결과 dict
        quick: True면 4줄 요약

    Returns:
        텍스트
    """
    if quick:
        return _render_classify_quick(cm)
    return _render_classify_full(cm)


def _render_classify_quick(cm: dict) -> str:
    """4줄 요약"""
    n = cm['n']
    acc_pct = cm['accuracy'] * 100
    baseline_pct = cm['baseline_acc'] * 100
    improvement = (cm['accuracy'] - cm['baseline_acc']) * 100
    indicator = "🏆" if improvement >= 20 else "✅" if improvement >= 10 else "⚠️"

    return (
        f"\n[Classify Quick] n={n}\n"
        f"  Accuracy  {acc_pct:.1f}% (LOOCV)  vs baseline {baseline_pct:.1f}%  "
        f"{improvement:+.1f}%p  {indicator}\n"
        f"  Precision {cm['precision']*100:.1f}% · Recall {cm['recall']*100:.1f}% · "
        f"F1 {cm['f1']:.3f}\n"
        f"  TP {cm['tp']} · TN {cm['tn']} · FP {cm['fp']} · FN {cm['fn']}\n"
    )


def _render_classify_full(cm: dict) -> str:
    """전체 분류 리포트"""
    return f"""
{'='*60}
🎯 BINARY CLASSIFICATION REPORT (n={cm['n']})
{'='*60}

📊 Confusion Matrix
                Predicted
                 Success    Failure
Actual Success      {cm['tp']:>5}   {cm['fn']:>5}
       Failure      {cm['fp']:>5}   {cm['tn']:>5}

📈 지표
  Accuracy:      {cm['accuracy']*100:>5.1f}%   ({cm['tp']+cm['tn']}/{cm['n']} 정답)
  Baseline:      {cm['baseline_acc']*100:>5.1f}%
  Improvement:   {cm['improvement_over_baseline']*100:+5.1f}%p
  Precision:     {cm['precision']*100:>5.1f}%
  Recall:        {cm['recall']*100:>5.1f}%
  Specificity:   {cm['specificity']*100:>5.1f}%
  F1 Score:      {cm['f1']:.3f}
"""
