# ⚾ KBO Foreign Pitcher Risk-Control Engine

> **This system does not attempt to perfectly predict pitcher success.**
> It attempts to reduce avoidable investment failures under uncertainty through
> calibrated probability modeling, statistical validation, and explainable risk decomposition.
> **The final decision authority remains with human evaluators; this engine supports, not replaces, scouting judgment.**

[![CI](https://github.com/[YOUR-USERNAME]/kbo-oracle/actions/workflows/ci.yml/badge.svg)](https://github.com/[YOUR-USERNAME]/kbo-oracle/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Tests](https://img.shields.io/badge/tests-133%20passed-brightgreen)
![Model](https://img.shields.io/badge/model-LogisticRegression%20%2B%20Calibration-orange)

**🌐 Live Demo:** https://kbo-oracle.streamlit.app *(배포 후 URL 교체)*

---

## 📌 한 줄 소개

raw 야구 스탯만으로 KBO 외국인 투수 **생존 확률**(Survival Probability)을
**calibrated probability**로 출력하는 ML 기반 Risk Governance Engine.

## 🎯 핵심 철학

### Risk Governance First
- "누가 에이스인가?"가 아니라 "**누가 100만 달러 투자 후 붕괴하나?**"
- Prediction Engine ❌ → Investment Risk Governance Engine ✅

### Statistical Honesty
- 정확도보다 Calibration 우선
- "70%"라고 말했으면 실제로도 70%여야 함
- Brier Score 기반 평가

### Lore-Free System
- Failure Score / Stuff Bonus / Pivot Bonus 등 휴리스틱 → ML 기반 확률 모델로 교체
- raw baseball signals만 사용

## 🏗 아키텍처 (Ver 15.1 리팩토링)

```
┌────────────────────────────────────────────────┐
│ kbo_oracle.py                                  │
│   ├─ Pitcher (dataclass)                       │
│   ├─ BacktestEngine (회귀, 회고용)             │
│   ├─ ScoutEngine (휴리스틱, 회고용)            │
│   └─ CLI (11개 명령어)                         │
├────────────────────────────────────────────────┤
│ risk_engine.py ⭐ (Ver 15.0 핵심)             │
│   ├─ RiskEngine (sklearn 기반 ML)              │
│   ├─ validate_engine (Brier/Calibration)       │
│   └─ loocv_validate                            │
├────────────────────────────────────────────────┤
│ renderers.py ⭐ (Ver 15.1 SoC 분리)           │
│   ├─ render_scout_report (Markdown)            │
│   ├─ render_backtest_report                    │
│   └─ ascii_scatter                             │
├────────────────────────────────────────────────┤
│ scout_report.py (Risk Engine 리포트)          │
│ csv_to_json.py (데이터 검증 파이프라인)        │
│ app.py (Streamlit 4-tab Dashboard)             │
└────────────────────────────────────────────────┘
```

## 🛠 기술 스택

| 영역 | 도구 |
|---|---|
| ML 모델 | sklearn LogisticRegression (elasticnet) |
| 확률 보정 | CalibratedClassifierCV (sigmoid) |
| 통계 | numpy, scipy (Pearson/Spearman) |
| 검증 | Stratified K-Fold + LOOCV |
| 백엔드 | Python 3.10+ |
| 웹 UI | Streamlit (4-tab Governance Dashboard) |
| 데이터 입력 | 구글 시트 → CSV → JSON 파이프라인 |
| 테스트 | pytest (133개 테스트) |
| CI/CD | GitHub Actions |
| 컨테이너 | Docker |

## 🚀 빠른 시작

```bash
git clone https://github.com/[YOUR-USERNAME]/kbo-oracle.git
cd kbo-oracle
pip install -r requirements.txt
streamlit run app.py
```

브라우저에서 `http://localhost:8501` 자동 열림.

## 🖥 CLI 사용법 (11개 명령어)

### Risk Engine (Ver 15.0 — 핵심)

```bash
# 개별 선수 risk 평가
python3 kbo_oracle.py --data dataset_external.json risk "Cody Ponce" --quick

# 리그 전체 평가
python3 kbo_oracle.py --data dataset_external.json league

# Brier/Calibration 검증
python3 kbo_oracle.py --data dataset_external.json risk-validate
```

### 데이터 입력 워크플로우

```bash
# CSV → JSON 변환 (구글 시트에서 다운받은 후)
python3 csv_to_json.py pitchers.csv

# 검증만
python3 csv_to_json.py pitchers.csv --validate-only
```

### 기존 명령어 (회고용, Ver 14.x 휴리스틱)

```bash
python3 kbo_oracle.py --data dataset_external.json classify --quick
python3 kbo_oracle.py --data dataset_external.json backtest
python3 kbo_oracle.py --data dataset_external.json validate
python3 kbo_oracle.py --data dataset_external.json scout "Cody Ponce"
python3 kbo_oracle.py --data dataset_external.json compare "Cody Ponce" "Dan Straily"
python3 kbo_oracle.py --data dataset_external.json value
```

## 📊 4-Tab Streamlit Dashboard

1. **🎯 Scout Report** — 개별 선수 risk 평가 (단장/스카우트 대상)
2. **🏟 League Analysis** — 전체 선수 ranking (운영팀)
3. **🧪 Validation** — Brier/AUC/Calibration (DS팀)
4. **⚙️ Governance** — 계수 안정성, 데이터 품질 (모델 관리자)

## 🔬 모델 구조

### Feature Set (14개 raw 변수)

**Skill**
- `shrunk_k_minus_bb`: K/9 - BB/9
- `swstr_pct`: Swinging Strike%
- `gb_pct`: Ground Ball%
- `xfip`: Expected FIP

**Physical**
- `age_at_debut`, `avg_velo`, `velo_trend`

**Role**
- `gs_ratio`, `ip_per_g`, `pitch_mix_count`

**Stability**
- `il_days_past_2y`

**Environment**
- `is_pcl`, `is_il`, `is_asian_league`

### Survival Target
- `is_survived` = 1: `actual_war >= 2.0`
- `is_survived` = 0: 그 외

### Risk Tiers
- 🟢 **GREEN** (P ≥ 0.70): Contractable Target
- 🟡 **YELLOW** (0.45 ≤ P < 0.70): Human Scout Review Required
- 🔴 **RED** (P < 0.45): Reject / High Risk

## 🧪 검증

```bash
pytest
```

**현재 133개 테스트 통과** (2초 이내)

| 테스트 | 개수 |
|---|---|
| test_risk_engine.py | 22 |
| test_renderers.py | 25 (NEW) |
| test_classification.py | 13 |
| test_parser.py | 14 |
| test_stat_translator.py | 31 |
| test_recontract.py | 12 |
| test_csv_converter.py | 16 |

## ⚠️ Honest Limitations (정직한 한계)

1. **Small-N (n=26)**: 변수당 데이터 1.86개 — ML 통계 기준 미달
2. **KBO 트래킹 비공개**: IVB·환경 보정은 일반론적 추정
3. **Temporal Validation 불가**: Stratified K-Fold로 대체
4. **Catastrophic Miss > 30%**: GREEN/YELLOW 일부가 실패
5. **계수 안정성 일부 경고**: 9개 변수 중 5개 sign 뒤집힘
6. **AAA-IL 함정**: 모델이 못 잡는 사각지대
7. **K-BB 무력화 (n=26)**: 야구 최고 지표인 K-BB가 우리 데이터에선 생존
   예측력이 거의 없음(상관 -0.02). 성공/실패 그룹의 K-BB 평균이 4.49 vs 4.54로
   사실상 동일. 리그 보정해도 안 살아남. 데이터를 억지로 맞추지 않고
   "K-BB는 리그 환경과 함께 봐야 한다"로 정직하게 기록. n 증가 시 재검토 필요.

## 🔧 Ver 15.2.1 자체 감사 수정 (Self-Audit Fixes)

시스템 전체 감사에서 발견·수정한 결함:

1. **기본값 오염 제거 (심각)**: `Pitcher` 데이터클래스의 `whiff_pct=22.0`,
   `zone_pct=50.0` 등 휴리스틱 시절 하드코딩 기본값이, 입력하지 않은 변수를
   가짜 값으로 채워 ML 엔진이 거짓 데이터로 평가하던 문제. 모두 `None`으로 변경.
   (예: 페디 평가 시 입력 안 한 SwStr%가 22.0으로 들어가 있었음)
2. **통찰-모델 모순 제거**: `generate_insights`가 "K-BB 높으면 우수"라고
   단언했으나 모델 실제 계수는 음수. 통찰을 모델 실제 학습 방향과 일치시킴.
3. **None-safety 강화**: 기본값 제거로 생긴 None 비교 오류를 extract_features에서 방어.

### 배포 직전 2차 감사 (Ver 15.2.2)
4. **데이터 일관성 통일**: 웹사이트가 86명(raw 결측 60명), 엔진이 26명으로
   불일치하던 문제. 실측상 26명(raw 충실)이 86명보다 AUC가 높아(0.426 vs 0.400),
   웹·엔진·배포 패키지를 모두 26명으로 통일.
5. **극단 입력 엣지 케이스 검증**: 전부 0, 음수, 극단값, 이름만 입력 등
   5개 비정상 입력에서 크래시 없음 확인. 데이터 부족 시 confidence_warning이
   "61% 결측 — 신뢰도 낮음"으로 사용자에게 경고함을 검증.

## 🚫 이 모델이 구조적으로 못 잡는 것 (인간 스카우트 영역)

이 시스템은 **스탯 기반**이라, 스탯 밖 정보로 결정되는 케이스를 못 잡습니다.
이건 튜닝으로 해결되지 않는 근본 한계이며, 그래서 시스템 이름이
"예측 엔진"이 아니라 "**리스크 거버넌스 엔진**"입니다.

### 실측 사례: Erick Fedde (2023 NC, WAR 7.63 대박)
- **모델 평가**: 48.4% YELLOW (거의 동전 던지기)
- **이유**: 직전 2022 MLB 시즌이 객관적으로 나빴음 (FIP 5.0, K-BB 3.1).
  가장 닮은 선수 Irvin·Sampson도 실패. 스탯만 보면 의심이 합리적.
- **모델이 못 본 것**: 드래프트 1라운드(18번) 엘리트 출신, 30세 반등기,
  의도적 KBO행(도피 아닌 재정비), 메커닉 개조 — **성공 요인의 99%가 스탯 밖**.

### 모델이 못 잡는 유형
| 유형 | 예시 | 왜 못 잡나 |
|---|---|---|
| **재정비형** | 페디 — 엘리트가 한 단계 내려와 반등 | "왜 KBO 왔는가" 변수 없음 |
| **유망주 등급** | 드래프트 상위 출신 잠재력 | draft_round 변수 없음 |
| **커리어 궤적** | 30세 반등 vs 35세 하락 | 나이 추세 미반영 |
| **심리·적응** | 부상 회복, 한국 적응력 | 스탯에 안 나타남 |

### 설계 철학
> 모델은 **'겉보기 좋은데 숨은 위험'**(Sampson형)을 거르고,
> 인간은 **'겉보기 나쁜데 숨은 보석'**(페디형)을 찾는다.
> 명백한 성공/실패는 인간이, 애매한 회색지대는 모델이 보조한다.

## 📝 Ver 15.1 리팩토링 노트

### 개선 사항
1. **numpy.corrcoef 사용** — Pearson 직접 구현 제거
2. **scipy.stats.spearmanr 사용** — Spearman 직접 구현 제거
3. **sklearn.metrics.confusion_matrix 사용** — 분류 평가 표준화
4. **renderers.py 분리** — UI 로직을 백엔드에서 분리 (SoC)
5. **Dead code 제거** — `kill` 명령어 제거 (risk가 대체)

### 면접 핵심 멘트
> "리팩토링 과정에서 직접 구현했던 Pearson, confusion matrix를 numpy/sklearn으로 교체했습니다.
> 검증된 라이브러리 사용이 바퀴 재발명 회피 + 코드 정확성 보장의 정석입니다.
> 또한 백엔드 로직 안에 있던 600줄 마크다운 템플릿을 renderers.py로 분리해서 SoC를 적용했습니다."

## 📁 파일 구조

```
kbo-oracle/
├── kbo_oracle.py            # 백엔드 + CLI (11개 명령어)
├── risk_engine.py           # Risk Governance ML Engine
├── scout_report.py          # Risk Engine 리포트
├── renderers.py             # ⭐ UI 렌더링 분리 (NEW Ver 15.1)
├── csv_to_json.py           # CSV → JSON 변환
├── app.py                   # Streamlit 4-tab Dashboard
│
├── pitchers_template.csv    # 구글 시트 임포트용
├── recontract_template.csv
├── dataset.json             # 기존 30명
├── dataset_external.json    # 외부 검증 26명
│
├── tests/                   # pytest 133개
│   ├── test_risk_engine.py    (22)
│   ├── test_renderers.py      (25 NEW)
│   ├── test_csv_converter.py  (16)
│   ├── test_classification.py (13)
│   ├── test_parser.py         (14)
│   ├── test_stat_translator.py (31)
│   └── test_recontract.py     (12)
│
├── .github/workflows/ci.yml
├── Dockerfile, docker-compose.yml
├── requirements.txt         # streamlit, openpyxl, scikit-learn, numpy, scipy
├── requirements-dev.txt, pytest.ini
└── README.md
```

## 🎤 면접 정직 멘트

> "처음에 회귀 모델로 시작해서 r=0.97 → 0.17로 추락했습니다. 데이터 누수 발견하고
> 26명 raw 스탯 직접 수집해 재검증한 결과입니다. 평가 방식을 분류로 바꿔서
> Accuracy 61.5%로 베이스라인 +11.5%p 우월함을 확인했습니다.
>
> 마지막으로 **WAR prediction toy가 아닌 Investment Risk Governance Engine**으로
> 재설계했습니다. sklearn LogisticRegression + Platt scaling으로 calibrated
> probability를 출력하고, Brier Score, calibration reliability, catastrophic
> miss rate, coefficient stability 등을 추적합니다.
>
> 리팩토링 과정에서 직접 구현했던 Pearson을 numpy.corrcoef로 교체하고,
> 600줄 마크다운 렌더링 코드를 renderers.py로 분리해서 SoC를 적용했습니다.
>
> N=26 한계와 계수 불안정성을 솔직하게 인정하고 governance dashboard로
> 추적합니다. **최종 의사결정 권한은 인간에게 있으며, 이 시스템은 확률·리스크·
> 불확실성을 정직하게 전달**합니다."

## 📜 라이선스

MIT License

## 👤 만든 이

**양승수** — 데이터 사이언스 포트폴리오 / KBO 야구 데이터 분석
