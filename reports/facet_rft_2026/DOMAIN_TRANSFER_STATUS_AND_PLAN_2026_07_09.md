# 도메인 전이 — 검증 status 표 + 실험계획서 (2026-07-09)

> ⚠️ **이름 폐기 안내**: 이 문서의 G/BC 코드는 폐기됐다. **통일 정본 = `UNIFIED_TAXONOMY_2026_07_09.md`**(서술형 이름). 이 문서는 status·실험계획(Phase 0-5)의 상세로만 참조.

> 목적: (1) retail서 **실험 검증된 것** vs 3도메인(airline·telecom·banking)의 **G·BC·해결책 예상**을 한 표로, **측정완료/증명필요**를 전부 표시. (2) 그 표를 채우는 **실험계획서**(가정·순서 명확).
> 근거: `TAU2_FRONTIER_..._MASTER §3.2b-d`(전수 sweep·C51/C52) · 등대 §1.4-1.6·§3 원장 · BC 재조합(세션 종합).
> ★규율: **진단(G/BC)=측정, 처방(레버)=증명 대상.** 등급 [S]settled·[M]measured·[P]promise·[D]design·[?]미실행. [D]/[?]를 [M]처럼 쓰지 말 것([[08]]).

---

## 0. 핵심 구분 (표를 읽는 법)
- **진단축(G→BC)**: 궤적서 실패 *기능*이 무엇인가. **4도메인 전수 측정됨**([M]·오늘). 단 G→BC *split*(G6·G4·G5)의 per-case는 **retail만**(C51).
- **처방축(레버)**: 그 실패를 *우리 레버*가 닫는가. **retail만 검증**(gate·calc·provenance·thinking). **3도메인은 예측([D])·증명 필요.**
- ⇒ **"어디서 실패하나"는 안다. "우리 처방이 그 도메인서 듣나"가 미증명.** = 전이(ABox-swap) make-or-break.

## 1. STATUS 표 — 도메인 × 기능 × BC × 해결책 × 상태

### 1.1 retail (진단+처방 모두 우리 실험 있음)
| G (측정%) | → BC | 해결책(레버) | **retail 상태** | 등급 |
|---|---|---|---|---|
| G1 COVERAGE 52% | BC1 | 완결 게이트(read-only) | ▲ 게이트 준수-보장 실증·완결게이트 자체는 부분(E1 소표본) | [S]게이트/[M]완결 |
| G6 OPERAND 30% | BC3+BC4(split✅C51) | 계산→calc·변형매칭→thinking | ✅ calc 실증·thinking 격리천장 .864·Qwen-think F2 0.4% | [M]/[P] |
| G2 REACH 18% | BC2 | 결정론 controller(gather) | ▲ fetch-first 33→9·plan_probe 부하실증·controller 미완측 | [M] |
| G4 PERSIST 10% | BC6 | persistence 게이트(read) | △ 짝맞춤 Δ=0(E1 소표본·미확인) | [M]소 |
| G7 REFERENCE⋈ 9% | BC4 | 경계/map·ASK | ✅ 경계 실증(~.44 flat·scale/CoT/RL 무효) | [S]부분 |
| G5 SCOPE 5% | BC1+BC4 | precondition(env집행)/경계 | ✅ E10 NO-GO(env 이미집행)·순수scope=경계 | [M] |
| 날조(operand) | (BC4 상류) | provenance 게이트 | ✅ **e2e GO**(67→0·다중턴 +3.3pp·C53) | [M] GO |

### 1.2 airline (진단 측정·처방 미검)
| G (측정%) | → BC | 해결책 예상 | **상태** | 등급 |
|---|---|---|---|---|
| G1 COVERAGE 52% | BC1 | 완결 게이트 | 🔬 진단[M]·처방 **증명필요** | G:[M]/처방:[?] |
| G6 OPERAND 28% | BC3+BC4 | calc·thinking | 🔬 split per-case 미검·처방 미검 | [M]/[?] |
| G7 REFERENCE⋈ 23% | BC4 | 경계/map | 🔬 retail보다 큼·경계 재현 미검 | [M]/[?] |
| G4 PERSIST 28% | BC6 | persistence 게이트 | 🔬 airline 최대급·미검 | [M]/[?] |
| G5 SCOPE 21% | BC1+BC4 | precondition/경계 | 🔬 retail보다 큼·미검 | [M]/[?] |
| G2 REACH 2% | BC2 | controller | 🔬 소·미검 | [M]/[?] |

### 1.3 telecom (dual-control·env-assertion)
| G (측정%) | → BC | 해결책 예상 | **상태** | 등급 |
|---|---|---|---|---|
| G1 COVERAGE 128%(다중결함) | BC1 | 완결 게이트(전 결함 수정) | 🔬 진단[M]·처방 미검 | [M]/[?] |
| G4 PERSIST 59%(조기 사람연결) | BC6 | persistence 게이트 | 🔬 telecom 최대·미검 | [M]/[?] |
| G2 REACH 6%(진단 미시도) | BC2 | 진단 controller | 🔬 미검 | [M]/[?] |
| G9 GUIDANCE(사용자 유도) | BC7 | 지시 scaffold | 🔬 dual-control 신규축·미설계 | [?]/[?] |

### 1.4 banking (KB검색+절차·장기)
| G (측정%) | → BC | 해결책 예상 | **상태** | 등급 |
|---|---|---|---|---|
| G2 REACH 38%(도구 discover/unlock) | BC2 | 결정론 controller(도구발견/조립) | 🔬 banking 최대·처방 미검 | [M]/[?] |
| G1 COVERAGE 22%(절차 완수) | BC1 | 완결 게이트 | 🔬 미검 | [M]/[?] |
| G4 PERSIST 12% | BC6 | persistence 게이트 | 🔬 미검 | [M]/[?] |
| G3 VERIFY 5%(log_verification) | BC1/BC3 | verify 게이트 | 🔬 명시 gold·미검 | [M]/[?] |
| HORIZON(절차 9단계) | BC5 | 분해+상태 controller | 🔬 장기·미검 | [M]/[?] |

## 2. 요약 — 무엇이 나왔고 무엇을 증명해야 하나
| 항목 | 상태 |
|---|---|
| **4도메인 G-분포(어디서 실패)** | ✅ **측정완료**([M]·전수 sweep 8/8/8/17 모델·C52) |
| **G→BC split(원인 정밀화)** | ✅ retail(C51 per-case) / 🔬 airline·telecom·banking **per-case 미검** |
| **retail 처방(우리 레버)** | ✅ gate·calc·provenance(e2e GO)·thinking·경계 — **검증** |
| **airline·telecom·banking 처방(전이)** | 🔬 **전부 증명필요** — 우리 scaffold를 그 도메인서 안 돌림 |
| **레버 트레이드오프 계수(커플링)** | 🔬 몇 쌍만 [M]·대부분 [D] |
| **learn(T) 축** | 🔬 미확립(C38·데이터 실패) |
| **G9 GUIDANCE(dual-control)** | 🔬 미설계(telecom 신규축) |

## 3. 실험계획서 — 표를 채우는 방안

### 3.1 가정 (명시)
- **A1 (진단 프록시)**: frontier 잔여 기능분포 ≈ 그 도메인의 *어려운 기능*. ★단 frontier≠우리 소형(우리는 *더* 실패하나 *같은 종류*·C21) → 처방 대상은 우리 소형 실패로 재확인 필요.
- **A2 (전이 가설·make-or-break)**: 우리 scaffold = 도메인-일반(TBox/엔진 고정)·ABox(gate_spec)만 교체. C52(COVERAGE/PERSIST/REACH 4도메인 재현)가 이를 *지지*하나 **처방 전이는 미실증**.
- **A3 (기준)**: db_match(DB도메인)·env_assertion reward(telecom). arm-공통·per-case.
- **A4 (레버 독립성 아님)**: 부작용 커플링 실재(§1.3) → 합성은 측정된 상쇄로만. read-only·Δspurious≤0 필수.
- **A5 (비용)**: 무료(기존궤적·로컬)→저비용→유료 순([[09]]). 유료 full-run은 승인.

### 3.2 순서 (cheapest·free 先 → paid 後)

**Phase 0 — G→BC split per-case (무료·로컬·즉시)**
- 목적: 1.2-1.4의 splitter(G6·G4·G5) 및 REACH를 per-case 정독해 BC 배정 확정(C51 방식을 airline/telecom/banking에).
- 산출: BC-split 셀 [?]→[M]. 특히 airline G6(변형=BC4? 계산=BC3?)·telecom G4(진짜 조기포기? env노이즈?)·banking G2(도구발견=부하? 능력?).
- 게이트: 배정 확정 전 처방 설계 금지.

**Phase 1 — 우리 소형 baseline 재확인 (저비용)**
- 목적: A1 교정 — frontier 진단이 *우리 소형*서도 같은 BC 지배인지(우리 32B를 airline/telecom/banking에). 처방 대상 질량 확정.
- 산출: "우리 소형의 도메인별 실패 BC" 표(현재 retail만 있음).

**Phase 2 — ABox(gate_spec) 저작 + 엔진 수용 (무료·핵심 전이 준비)**
- 목적: airline/telecom/banking의 gate_spec(완결·verify·precondition·persistence)을 도메인-일반 엔진에 인스턴스화. 엔진 리터럴 0 유지([[05]]).
- 산출: A2 전이의 A-레버. 엔진 unit test(도메인 리터럴 0·gate 발화).
- 게이트: telecom은 env-assertion·dual-control이라 게이트 의미 재정의 필요(BC7 GUIDANCE 신규 설계).

**Phase 3 — ★scaffold 전이 실측 (유료·승인·make-or-break)**
- 목적: 우리 scaffold(32B+게이트+calc+provenance) × 3도메인 ABox-swap → **처방 전이 pass** 측정. = A2 검정·특허 도메인-일반 청구 실증.
- GO: 도메인마다 게이트 위반 0 ∧ over-block 0 ∧ Δspurious≤0 ∧ pass↑(vs 소형 floor). retail C1/C2 재현.
- 순서: banking(REACH 지배·controller) → airline(COVERAGE+OPERAND·retail 유사) → telecom(GUIDANCE 신규·최후).

**Phase 4 — 레버 트레이드오프 계수 (커플링 측정·중간비용)**
- 목적: [D] 교차항 측정(thinking↔coverage·gate↔over-action·present 폐기 재확인). §1.3 합성 매트릭스 채움.
- 산출: 최적화 목적함수의 계수 → 도메인별 레버플랜 자동산출 가능.

**Phase 5 — learn(T)·GUIDANCE (게이트 통과 후·최후)**
- learn: C38 데이터 타당성 게이트(D7 근접오답+음성사례+on-policy) 통과 전 착수 금지.
- GUIDANCE: telecom dual-control 지시 scaffold 설계(BC7 신규).

### 3.3 완료 정의 (표가 다 차는 조건)
- 1.2-1.4의 **처방 셀 [?]→[M/S]**: Phase 0(split) + Phase 3(전이 실측)로.
- **레버 커플링 [D]→[M]**: Phase 4.
- **learn/GUIDANCE [?]**: Phase 5(조건부).
- ★**우선순위**: Phase 0(무료·즉시) → Phase 2(무료) → Phase 3(유료·make-or-break·특허 핵심) 이 최단 경로. Phase 1은 A1 불확실 크면 삽입.

## 3.4 ★Phase 0 결과 (2026-07-09 실행·per-case BC 확정)
스크립트 `phase0_bc_split.py`(로컬·무료). 각 도메인 지배 splitter를 per-case 정독:
| 도메인 splitter | per-case 신호 | **확정 BC** | 구제 |
|---|---|---|---|
| **airline G6 OPERAND** | baggage **수치 8/11**(계산)·날조 3 | **BC3 compute** | **calc(무료·decidable)** |
| **telecom G4 PERSIST 59%** | 결함남고 escalate·**fix 전혀 미시도 17** + 진단후 포기 9 (26/30) | **BC6 조기포기** | persistence 게이트(read-only) |
| **banking G2 REACH 38%** | **KB검색 함 119/125** but 절차 unlock 누락 = 발견가능·조립실패 | **BC2 부하(능력 아님)** | 결정론 controller(절차조립) |
| retail G6 (기존 C51) | new_item_ids 문맥실재·오선택 | **BC4 select** | thinking/경계 |

**★핵심 발견 2**: **같은 G가 도메인마다 다른 BC** — G6 OPERAND가 retail=BC4(변형매칭·의미) vs airline=BC3(baggage 계산·decidable). ⇒ **G→BC는 도메인별 per-case 필수**(G 단위 처방 금지). **핵심 발견 2b**: 3도메인 지배 splitter가 **전부 도메인-일반 결정론 구제(calc·persistence게이트·controller)로 라우팅** → A2 전이가설을 *진단 층에서* 지지(처방 실측은 Phase 3).
- 상태 갱신: 1.2-1.4 splitter BC 셀 **[?]→[M]**(진단). 처방 셀은 여전히 Phase 3.

## 3.5 ★Phase 2+3 banking 착수 기록 (2026-07-10 · 사용자 승인 "banking 우선·여유시 airline")
- **Phase 2 완료 (banking A2)**: `a2/banking_knowledge.gate.json` — auth-kind 게이트 1개(**GB1: log_verification 성공 전 고객데이터 도구 deny**·정책 "verify 2-of-4 then call the verification logging tool" 인스턴스). **gold 전수 검증**: 97태스크 중 81=gold가 log_verification 포함·순서위반 0 / 예외 4(015·032·033·035)=transfer/incident 내부도구(고객데이터 미접근·정책 정합) → 엔진에 도메인-일반 `applies_when {arg, in/not_in}` 멤버십 가드 추가(`gate_interpreter._gate_applies`·[[05]] 3질문 no·값=A2)로 면제 = **over-block 0 by construction**. confirm/notice/preconditions는 banking 정책에 결정가능 사실 부재 or env-집행(C12)이라 미인스턴스. 단위테스트 `test_banking_gate.py` 23건 PASS(retail 회귀 포함). `t2_compliance`도 `_gate_applies` 재사용으로 G1 census가 applies_when 반영(`82015b16`).
- **★발견 1 — 기존 32B banking floor는 방법-결함**: `ours_n32int8_floor_bank_t3`(2026-06-24) infra 31/291 = 전부 16384-serve ContextWindowExceeded(banking 장기). → **floor 32768 재런 필수**(bankxfer_floor_bank_t4).
- **★발견 2 — 러너의 banking 변종 덮어쓰기는 죽은 코드였다**: 갱신된 tau2-bench의 `BaseRunConfig`가 banking일 때 `retrieval_config`를 **'alltools'로 자동 디폴트**→env_kwargs로 `retrieval_variant` 명시 전달→registry partial의 키워드를 **조용히 덮어씀**. "no_knowledge로 돌았다"는 종전 출력은 거짓·실제 alltools. 스모크 궤적의 KB_search 성공으로 발각([[08]] 궤적 정독). 수리=`--retrieval_config` 인자+config 경로 전달(`9baf7b27`)·**Phase 3 변종=openai_embeddings 고정**(dense KB·전 도구 작동. alltools는 sandbox binaries 부재로 shell 도구가 '광고되나 고장' 상태). ⚠️구 floor(6/24)의 변종은 당시 코드 기준 불명 — 어차피 16k 결함으로 대체됨. frontier 대조는 변종 상이 caveat 명기.
- **스모크 2회**(8140 공유·타 세션 무간섭): ①alltools-우발(8태스크·infra 0·user_stop 8/8·면제 3태스크 무차단·gate 발화 0=**위반 상황 부재의 정상 침묵**, task_004/035는 자발 검증-선행) ②변종 고정 후(3태스크·KB_search 단일 도구 확인·크래시 0). openai_embeddings 도구셋 15개와 A2 applies_to 정합 확인.
- **Phase 3 발사**(2026-07-10 16:0x): `bankxfer_full_run.sh` — GPU1 전용 32B serve(8142·32k) / floor(`bankxfer_floor_bank_t4`·--gate 0) → arm(`bankxfer_gate_bank_t4`·T2_GATE_REGEN=1·K=1·KINDS=auth) 순차·nt=4·97태스크·gpt-4.1 user-sim·런별 즉시 gzip persist+push. GPU1의 타 세션 잡(ep2_nestful+QwQ serve)은 **사용자 지시로 킬**. GPU0(타 세션 retail 런)=불가침.
- **판정 기준(예고)**: GO = arm 게이트위반 0(compliance G1) ∧ 면제 태스크 false-block 0 ∧ Δspurious≤0(floor-pass가 게이트 귀속으로 fail 전환 0) ∧ tme 폭증 없음. bench/compliant pass^1..4 = 덱 결과⑩ banking 칸.
- **★프로토콜 변경(2026-07-10 밤·사용자 결정 = 옵션 b-변형)**: nt=4 full이 예상보다 10× 느림(floor trial1만 ~6.8h·sim당 4.2분 — 32k 장문화+KB dense 임베딩 왕복 추정) + floor 부분값 축퇴(71 sims mean 0.0597·pass 4/71) → **nt=1 델타 우선**으로 전환. trial 1 완료 시점에 nt4 런 중단·trial1을 `bankxfer_floor_bank_t1`로 salvage(재비용 0)·**gate arm nt=1** 연쇄 발사(`bankxfer_switch.log`). ★arm = **retail 현재 canonical 전-레버 unified**(GB1게이트+prov+DISAMB·nested/calc — 사용자 지시 2026-07-10 밤·게이트-단독에서 격상·직전 3태스크 미니스모크+폴백 내장). 사이클 병합 유효성 위해 후속 사이클 arm도 동일 구성 고정. 내일 델타 확인 후 **추가 trial을 distinct tag로 돌려 태스크-단위 병합**(nt 증설·pass^k 복원). 덱 표기: banking은 우선 p^1(정직 병기). 부분 floor infra 4/71 = 유형 포렌식 대기.

## 4. 리스크·중단조건
- **A2 전이 실패**(scaffold가 타 도메인서 위반↑/over-block): 경계지도로 후퇴·"도메인-일반" 청구 축소(§15.7 정직).
- **telecom dual-control**: 우리 프레임(단일-agent write)과 다름 → 별도 취급·과확대 청구 금지.
- **Phase 3 유료**: 승인·최소 scope·smoke 先([[09]]·[[30]] 즉시영속).
