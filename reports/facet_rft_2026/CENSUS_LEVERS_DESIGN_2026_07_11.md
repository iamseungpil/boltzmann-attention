# CENSUS 레버 설계 — C64 전수 실패의 클래스별 해결책 명세 (2026-07-11)

> 파생: `RETAIL_FULL_FAIL_CENSUS_2026_07_11`(C64) §3 라우팅의 설계-구체화. [[05]]/[[10]]/제1원리(반대편 계측) 전 레버 적용.
> **범위**: 무료 엔진-증분 4종(§1~§4) + B-잔여 스케치(§5) + 단계 B 편입(§6). **A클래스(coverage/discovery)는 별도 정본** = `E_PLAN_LIVE_WIRING_DESIGN`(v1.2 확정).
> 상태: **[D] 설계서.** 각 레버 V0(무료 census) → 단위 → 26-task nt=1 사이클 순. 유료는 승인 후.

---

## 0. 클래스 → 레버 지도 (C64 §2 대응·전량)

| C64 클래스 | 해결 설계 | 위치 |
|---|---|---|
| A coverage/discovery (≈8 task) | E-PLAN L1/L2 + CP5 재-plan walk | `E_PLAN_..._DESIGN` v1.2 |
| B 대화-조건 over-action (≈5) | 결정가능분: §3(배타-ask·notice) / 잔여: §5 [D] | 본 doc |
| C compound-criterion (≈6) | §2a calc 기준-주석 확장 | 본 doc |
| D GET-chain lookup (≈5) | §4 DISAMB-주소 확장 + (미조회 원천은) E-PLAN L2 | 본 doc + 교차참조 |
| E DISAMB 잔여 (63 sims) | T5-C COMP+D-v2 (기구현) | `T5C_SILENT_REPAIR_DESIGN` |
| F 값충실도 (t17·t39) | §1 GROUND-VERBATIM | 본 doc |
| G NL_ONLY (13 sims) | §2b — ★compute-gap과 relay-gap 분리(신규 실측) | 본 doc |
| H 게이트-deny 상관 | 레버 아님(C63 반증·진단 마커만) | census §4 |
| I 노이즈/경계 | P3 정직 보고·robust-레버 간접 | — |

---

## 1. GROUND-VERBATIM — 값충실도 제자리 정규화 (F클래스·t17/t39형)

**표적 실측**: t17 = 사용자가 "Suite 641만 변경" 요청 → 에이전트가 address1을 `"123 Elm St"`로 **축약 복사**(문맥 원문 `"123 Elm Street"`·4/4 동일·v25e 잔존). t39 = address1/2를 **빈 문자열**로 write.

**엔진 op (도메인일반)**: write-인자 값 v에 대해, 문맥(에이전트-기조회 tool 출력)의 해당 필드 후보 집합 C를 구성.
- **fuzzy-|C|=1 치환**: C에서 v와 *유일하게* fuzzy-match(축약·공백·대소문자·구두점 정규화 후 일치 or 접두 토큰 포함)하는 원문 c가 있고 exact-match가 아니면 → **v를 c 원문으로 silent 치환**(T5-C P-A |C|=1 채널 재사용·대화 불변·replay-clean).
- **empty-값 처리(t39)**: v가 빈 값 ∧ |C|=1이면 치환 **후보**로만 — 엔진이 값을 '선택'하는 경계 사례라 **V0서 fix/break 실측 후 채택 결정**(break>0이면 empty는 제외하고 fuzzy만).
- 발화 조건 밖(exact-match·|C|≥2·C=∅)은 무개입. |C|≥2는 DISAMB 관할(§4·서로소 관할 원칙).

**A2 (ABox)**: `ground_verbatim_fields: ["address1","address2","city","state","zip","country"]` — 필드 목록만. 엔진 리터럴 0.
**[[05]]**: 값 원천=에이전트-기조회 문맥(DB 안 읽음·C34 클린). 치환=이미 GO된 P-A 채널.
**검증**: V0 — COMP 167 실패 전수에 오프라인 적용 → fix/break census(t5c_v0_whitelist.py 확장). GO = fix≥5 ∧ break=0. 단위 = 정규화 함수(축약/공백/빈값/다중후보 케이스).
**반대편 계측**: 치환-유발 flip(원래 맞던 값을 바꿈) = V0 break 카운트 + 라이브 Δspurious.

## 2. CALC 확장 — compound-criterion과 NL의 분리 (C·G클래스)

### 2a. 기준-주석 op 확장 (C클래스: 쓰기-선택 기준·decidable·§1.5 Q1)
기존 calc_specs(`count_where`·`sum`·nested/calc 채널=규칙0 클린) 위에 **일반연산 3종 추가**:
| op | 스펙 예(A2) | 닫는 표적 |
|---|---|---|
| `argmax_where` / `argmin_where` | get_product_details → "가장 비싼/싼 available 변형: <item_id·price>" | t20(최고가 업그레이드)·t36/t37(최저가 다운그레이드 후보) |
| `most_recent` | 주문 목록/타임스탬프 → "가장 최근 주문: <order_id·date>" | t71(argmax 날짜·C56 체계핵) |
| `pairwise_diff_sum` | 제안된 exchange (old,new) 쌍들의 가격차 합 → "오늘 지불 총액: $X" | t95-NL(총액)·t99류 |
- 전부 **에이전트-기조회 데이터 위 계산**(트리거=해당 tool 출력·주석으로 부착) = nested/calc 선례. 엔진=op 3개(도메인일반)·A2=trigger/field 선언.
- t37의 "조합이 예산 이하" 전역 제약은 argmin 주석+sum으로 재료 제공까지(조합 탐색은 에이전트 몫 — 조합최적화를 엔진이 풀면 도메인행동 경계·1차 스코프 밖).
- t79("다른 병과 같은 색")류 **cross-record attr-match는 여기 아님** → §4 DISAMB 열거(옵션 필드 표시)가 관할.

### 2b. ★G클래스 신규 실측 — compute-gap이 아니라 relay-gap이 절반 (t3 4-trial 프로브·2026-07-11)
**t3 실측**: calc `count_where`가 **4/4 정확 발화**("...variants: 10" 주입). 실패는 ① tr2/3: 에이전트가 그 수를 **사용자에게 끝내 안 말함**(NL met:false·"agent never...") ② tr1: "modify 하겠다" 선언 후 **write 미실행**(별개=write-loss). ⇒ **"calc 확장 ≈25–30 sims" (census §3 row4)는 과대추정 — 정정**: G클래스 = compute-gap(op 부재·t95총액)과 **relay-gap(계산돼 있는데 전달 누락·t3형)**의 혼합.
- compute-gap → §2a op 확장이 닫음.
- **relay-gap → calc로 못 닫음.** 후보 = CP5 walk의 **communicate-의무 확장**(재-plan이 "답해야 할 질문"도 추출 → 미답이면 리마인더) — E-PLAN CP5와 좌석 공유·**[D]·격리 프로브 先**(재-plan이 질문-의무를 안정 추출하는지 + NL diff의 결정론성 한계 정직 평가). 1차 스코프 밖·E-PLAN §5(d)에 프로브 항목 추가.

**검증**: 단위(op 3종) + 오프라인 재계산(COMP 궤적서 t20/t71/t95 트리거 시점에 주석이 정답을 내는지 = 무료 census) + 라이브 발화율([[30]] 스모크).
**반대편 계측**: 주석-과잉(무관 주석이 창 오염=C43 재료) → 주석은 트리거 tool 출력에만·개수 상한·Δfab 계측.

## 3. FEASIBILITY·NOTICE — B클래스 결정가능분 (t27·t57 일부)

### 3a. 배타-op ask 게이트 (t27형·`exclusivity_specs`)
- **실측 기전**(t27): 동일 주문에 반품+교환 동시 요청·"하나만 되면 교환 선호" → 에이전트가 반품 먼저 실행 → status 변경 → 교환 env-불가 → gold(교환만) 실패 3/4.
- **스펙**: `exclusivity_specs: [{classes:["item_return","item_change_delivered"], scope:"order_id"}]` (A2). 엔진: **E-PLAN ledger의 planned에 같은 scope의 배타-class write가 2+개**일 때 첫 write 시도를 **deny-once + "정책상 동시 불가·사용자에게 어느 쪽인지 확인하라" 피드백**(replay-safe regen 채널) → 에이전트가 ask → 사용자 선호(교환)만 실행.
- 결정성: 배타성=도구/정책 사실(decidable·Q1)·발화 조건=ledger 상태(결정론). **write 강제 0**(ask 유도만). C50과 구분: DB-state 아니라 도구스키마 사실 + 자기-plan 참조.
- 의존: E-PLAN ledger (planned). E-PLAN 미배선 시 폴백 = 대화 내 두 intent 모두 write-시도로 나타난 뒤 2번째에 deny(약한 판)·1차는 ledger판.

### 3b. 환불-목적지 notice (t57형·`notice_specs` 확장)
- **실측 기전**(t57): 조건체인 끝="gift card 환불 안 되면 취소도 하지 말라" → 에이전트가 취소 실행 + **"gift card로 환불했다" 허위 발화**.
- **스펙**: cancel-class write의 confirm 시점에 정책-사실 주석: "환불은 **원결제수단**(<pm>)으로 감" — 기존 notice kind에 A2 1줄. 정책 사실=decidable(policy.md)·DB 값은 에이전트-기조회 pm 사용.
- 기대 경로: 에이전트가 정확한 목적지를 사용자에 전달 → user-sim(체인 보유)이 "그럼 취소하지 마라" → no-op = gold. **에이전트 판단 경유·강제 0.**
- 한계 정직: user-sim이 재확인 안 하면 못 닫음 — t57 커버는 확률적. Δ로 실측.

## 4. DISAMB-주소 확장 (D클래스 중 문맥-실재분·크기는 V0가 확정)
- **실측 기전**: 오답 주소는 전부 문맥-실재 레코드의 오복사(t109 구주소 3/4·t86 Dallas 프로필주소 2/4·t102-tr2 Seattle). **단 ⚠️정답 주소의 문맥-실재 여부는 trial별로 다르고 미전수**: t109-tr0는 같은 trial서 gold 주소 exact-write도 있음(=실재 증거) vs **t86-tr2 발화 "unable to locate the DC address in your previous orders" = 정답 미조회 시사**. ⇒ **D클래스 = 문맥-실재(DISAMB 관할) + 미조회-원천(E-PLAN L2 관할)의 혼합 — 분해는 V0 오프라인 census가 확정**(write 시점 문맥에 gold 주소 존재 여부 전수).
- **스펙**: T5-C P-B(DISAMB 격리 서브콜·silent 치환) 커버리지를 **주소-필드 write**로 확장 — 후보=문맥의 완전한 주소 레코드들(`_candidate_records` 재사용)·서브콜에 사용자 단서(도시/주 언급)가 자연 포함된 transcript 제공 → 서브콜이 선택 → 인자만 치환(대화 불변). |C|=1이면 §1 GROUND-VERBATIM 관할(서로소).
- **미조회-원천 분**은 여기 못 닫음 → **E-PLAN L2**("in one of my orders" = 미검토 sibling 신호) 교차참조 — E-PLAN 설계 §0 표적에 D클래스 명기.
- 검증: **V0 오프라인 = 분해+fix/break 동시**(주소-오복사 실패 전수에 ①gold-실재 판정 ②실재분에 후보 열거+서브콜 격리 재현) → GO시 단계 B 편입. C43 주의: 후보는 이미 창에 있는 레코드만(신규 노출 0).

## 5. B-잔여 — 대화-조건 semantic (t34·t36·t99·t57 잔여) [D]·정직
- 공통형: 사용자 조건체인("X 안 되면 Y, Y도 안 되면 하지 마라"·"내가 직접 하겠다")의 **오해석-실행**. C50 확정대로 DB/도구-사실이 아니라 **대화 semantic** → 게이트 불가.
- 후보 스케치(ENDGAME R1·이 doc 스코프 밖): confirm-게이트의 CP2 내용-검증(동의가 *무엇에 대한* yes인지) + 조건체인 formalize 서브콜(LLM 형식화→결정론 walk·§1.5 Q3-compound 경로). **격리 프로브(무료)로 형식화 정확도 先측정** 후 설계.
- 그 전까지 이 4-task는 **P3 경계 후보**로 정직 계상(§1.3 죽은레버 아님·미개척).

## 6. 단계 B 편입 (T5-C 사이클·`§0b` 프로토콜)
- **config 추가**: `T2_GROUND_VERBATIM=1`(§1) `T2_CALC_EXT=1`(§2a) `T2_EXCLUSIVITY=1`(§3a·ledger-有시) `T2_NOTICE_REFUND=1`(§3b) `T2_DISAMB_ADDR=1`(§4) — 각각 독립 toggle·stderr 마커(`[T2_GVERB]`·`[T2_CALCX]`·`[T2_EXCL]`·`[T2_NOTICE_R]`·`[T2_DISAMB addr]`) = per-case 귀속.
- **순서**: ① 각 레버 V0/오프라인 census+단위(무료·병렬 가능) → ② GO 레버만 스택에 → ③ 표적 26 task(기지 13+신규 SYSTEMIC 13) × nt=1 사이클 → ④ per-case → 사이클 반복. E-PLAN은 **별도 arm 유지**(합산 금지) — 단 §3a ledger 의존분은 E-PLAN 배선 후.
- **공통 GO**: per-case 복구 ∧ Δspurious≤0 ∧ over-block=0 ∧ Δtme≤0 ∧ 위반0 유지. 실패 레버는 개별 제거(§1.3).
- **기대 커버(정직·상한)**: F≈7·C≈15·G-compute≈6·B-decidable≈5 sims 상당. **D는 V0 분해 전 크기 미정**(문맥-실재분만·§4). 중첩·확률분(§3b user-sim 의존) 있어 전부 상한 — **실제 커버는 V0 census가 레버별로 확정**. FLAKY 52-task 질량은 개별 아니라 robust-레버(P2·DISAMB·pin)의 분산 축소로.

## 7. [[05]] 3질문 감사 (레버별)
| 레버 | Q1 엔진=도메인일반? | Q2 A2만 추가? | Q3 도메인행동 대행? |
|---|---|---|---|
| §1 GROUND-VERBATIM | fuzzy-정규화 op | 필드 목록 | ✗ (기존 값 정규화·생성 아님. empty-케이스는 V0 게이트) |
| §2a CALC-EXT | argmax/recent/diff op | trigger/field 선언 | ✗ (주석만·선택은 에이전트) |
| §3a EXCLUSIVITY | ledger-배타 검사 | class 쌍 목록 | ✗ (deny-once+ask·강제 0) |
| §3b NOTICE-REFUND | notice kind 기존 | 문구 1줄 | ✗ (정책 사실 고지) |
| §4 DISAMB-ADDR | P-B 기존 채널 | 필드 커버리지 | ✗ (문맥-실재 후보 중 선택) |
