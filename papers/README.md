# Papers — 4-paper portfolio (tool-use / small-model + deterministic offload)

> 단일 진입점. 연구 기록·도구·결과는 `../scripts/distill/`(권위)·`../reports/facet_rft_2026/sim_results/`(결과). 이 디렉터리 = 논문화. branch `facet-rft-2026`.

## 포트폴리오 골격

| # | 디렉터리 | 성격 | 한 줄 | 정본 설계문서 |
|---|---|---|---|---|
| **1** | `paper1_capability_scale_lever/` | **이론·지도** (ICLR 타깃) | 기능×스케일(7B–235B)×레버 지도: scale이 *무엇을 사고* / LLM이 *무엇을 진짜 못하고*(compliance=scale-invariant) / scaffold+A2가 *무엇을 싸게 메우나* / 잔여. 벤치=측정도구 | `EXPERIMENT_DESIGN §0`·`LOAD_THEORY_DESIGN`·`MAKEORBREAK_VERDICT` |
| **2** | `paper2_a2_generation/` | 학습법 (proposal) | NL 정책 → `GATE_SPEC` 컴파일러 학습. #1/#4가 *given*으로 쓰는 A2를 *만드는* 방법(공통 적 A2 비용 절감) | `A2_FRONTEND_DISTILL_DESIGN` |
| **3** | `paper3_path_selection/` | 학습법·직교축 (proposal) | 도구폭발 하 path-search를 *전이가능 학습 휴리스틱*으로(탐색=offload·LLM=②적용op인식·③value). #1(provenance축)과 직교 | `PATH_SELECTION_AXIS_DESIGN` |
| **4** | `paper4_system_cost/` | 시스템 (proposal) | {모델학습·A2생성·scaffold} 세 레버를 CapEx/OpEx로 최적 배합·knee·field 횡단 배포·TCO(~23×) | `CAPABILITY_LEVER_ALLOCATION_DESIGN`·`TCO_TABLE_DESIGN` |

## 구조 논리
- **#1(이론) ↔ #4(시스템)** = 같은 것의 *이론 vs 엔지니어링*. #1이 *지도*를 그리고(무엇을 어느 레버로) #4가 그 위에서 *비용 최적 배합*을 구현.
- **#2·#3 = 지도의 "학습이 답"인 칸들의 방법** — #2(A2 생성)·#3(path 휴리스틱). 둘 다 #1 프레임 인용.
- **#1·#3 직교**: #1 tau2 = provenance/grounding 축 / #3 = path-search 축. *섞지 않음.* 이번 세션 load 발견(L_branch=scaffold-저항)이 #1→#3 handoff.

## 상태 (2026-06-26)
- **#1**: 초안 작성중. 실증 spine = make-or-break(operand=scale-포화·SFT NO-GO) + load 분해(차원 은퇴) + **F3/F4 compliance scale-invariant**(논문 백본). [ESTIMATE]=235B·multi-bench 전이.
- **#2**: proposal. S0 창발·S1 기각·dose-response 설계까지 실측 있음.
- **#3**: proposal(학술버전만·**CDP/특허 specifics는 GitHub 금지**·[[32]]·confidential은 로컬 `_cdp_private_local`).
- **#4**: proposal. TCO ~23× 실측·전 매트릭스 [ESTIMATE].

## 규율
- 헤드라인 = 결정론 scaffold + 작은 base + A2 + (학습은 도메인-일반·전이=ABox-swap). **"소형>대형"은 rival(ToolOrchestra 2511.21689)이 선점 → 우리 = method(zero-retrain 전이·by-construction compliance·on-prem) + "scale이 못 푸는 guarantee".**
- 결과 인용 = `sim_results/`(전부 gpt-4.1 0). 추정은 [ESTIMATE] 명시.
