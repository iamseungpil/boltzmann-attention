# CENSUS 레버 설계 — C64 전수 실패의 클래스별 해결책 명세 (2026-07-11)

> 파생: `RETAIL_FULL_FAIL_CENSUS_2026_07_11`(C64) §3 라우팅의 설계-구체화. [[05]]/[[10]]/제1원리(반대편 계측) 전 레버 적용.
> **범위**: 무료 엔진-증분 4종(§1~§4) + B-잔여 스케치(§5) + 단계 B 편입(§6). **A클래스(coverage/discovery)는 별도 정본** = `E_PLAN_LIVE_WIRING_DESIGN`(v1.2 확정).
> 상태: **[D] 설계서 v1.1 — 리뷰(2026-07-11) 반영.** 각 레버 V0(무료 census) → 단위 → 26-task nt=1 사이클 순. 유료는 승인 후.
> v1.1 변경: ①§3a 폴백 무효 판정·삭제(2번째-write deny는 첫 write 해악 후라 t27 못 닫음) ②ledger 관측-전용 분리로 §3a↔E-PLAN arm 충돌 해소 ③§1 fuzzy 정의 강화(숫자-토큰 불일치 금지·역치환 break-모드 명시·검출규칙 신설임을 명확화) ④§2a pairwise_diff_sum을 confirm-시점 notice 채널로 이동.

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

## 1. GROUND-VERBATIM — ❌ **V0 NO-GO 확정·레버 폐기 → 진짜 레버 = prov rescue per-arg 수정 (2026-07-11 실측)**

### V0 판정 (`census_v0_gverb_addr.py`·COMP 167 + v25e 13 fail·lenient 파싱)
- **fuzzy 치환 표적 = 양 arm 0건**(fix 0·break 0). **empty-치환 = break 실재**(t59: gold address2가 정당한 빈 값인데 'Suite 165' 채움) → 설계의 empty-게이트가 기각 확정. ⇒ **GROUND-VERBATIM 폐기(죽은 레버·§1.3).**

### ★t17 재진단 — "축약-복사"는 오진·실체는 **미조회 날조 + prov rescue 입도 구멍** ([M]·전수)
1. **v25e 4/4 + v25d 전수**: `modify_pending_order_address` write 시점 **read 0회**·"Elm Street"(정답 원문)가 사용자 발화·tool 출력 어디에도 **부재**. "123 Elm St"는 문맥-복사가 아니라 **자유생성 날조**(실주소 근사 = 훈련-데이터 오염 의심). GROUND는 C=∅ → 무개입이 정상 동작.
2. **prov가 못 잡은 진짜 기전**(v25e 로그 157-158행 + 코드 정독으로 확정): prov는 이 write서 fab을 감지했다 — 단 **첫 fab 인자 = order_id `#W8665881`**(사용자는 `W8665881`로 발화·msg[7] 실측 → `#` 접두 불일치 = **거짓양성 fab**) → rescue 분기(`t2_gate_patch.py:881-888`)가 env-검증형 id 조건(`_key_tokens∩env_args ∧ _sig∈{hashid,numid}`) 충족 → **`break`로 regen while-루프 전체 탈출** → 둘째 fab 인자 **address1(자유텍스트)은 검사 자체가 안 됨**. `_first_fab_call`(:405)이 per-call 첫 인자만 반환 + rescue가 per-call `break`인 이중 구조.
3. (v1.1 리뷰의 "substring 통과" 가설은 **기각** — 문맥에 매칭 대상 자체가 없었음. 단 substring 관대함은 별개 구조 구멍으로 유효할 수 있으나 t17의 작동 원인 아님.)

### ⇒ 대체 레버: **PROV-RESCUE-PERARG** (엔진 수정·무료·decidable)
- **수정**(2점·`t2_gate_patch.py`): ① rescue `break`(:888)를 **해당 fab의 스킵-마킹 + 다음 fab 인자 계속 스캔**으로(`_first_fab_call`에 exclusion set 인자) — env-검증형 id fab만 개별 pass-through·**자유텍스트 fab이 남으면 그 인자로 regen 발화**. ② **id `#`-접두 정규화**: ctx 매칭 전 `s.lstrip('#')`류 비교로 거짓양성 fab 자체를 제거(t17의 1차 방아쇠).
- 기대: t17 4/4 → regen 피드백("getter로 실값 조회") → 에이전트 read → 정확 write. C24(free-text 날조는 환경이 못 잡음)의 정확한 봉합·C29 gather 정합.
- [[05]]: 기존 prov 엔진의 입도 수정(도메인 리터럴 0)·A2 불변. 반대편 계측: regen 과발화(Δtme)·over-block(정당 자유텍스트 인자 — 사용자-발화 값은 ctx에 있어 fab 아님 = 구조적 안전).
- 검증: 단위(다중-fab 인자 순회·`#`정규화) → v25e t17 4-trial 오프라인 재현(fab 검출이 address1에 닿는지) → 표적 nt=1.

**(기각된 원 설계는 기록으로 하단 보존·t39 빈값도 미조회 계열(V0 addr: gold not_found 3/3)로 같은 레버 관할.)**

**엔진 op (도메인일반)**: write-인자 값 v에 대해, 문맥(에이전트-기조회 tool 출력)의 해당 필드 후보 집합 C를 구성.
- **fuzzy-|C|=1 치환**: C에서 v와 *유일하게* fuzzy-match하는 원문 c가 있고 exact-match가 아니면 → **v를 c 원문으로 silent 치환**(T5-C P-A |C|=1 *치환 메커니즘* 재사용·대화 불변·replay-clean).
- **fuzzy 정의 (v1.1 확정·단위테스트로 고정)**: 정규화(공백·대소문자·구두점) 후 일치 **or** 토큰 다중집합이 축약-등가(Street↔St·Avenue↔Ave류 일반 축약표=엔진 상수·도메인일반). **★숫자 토큰은 완전 일치 필수** — 숫자 하나라도 다르면 매치 금지(Suite 641 vs Suite 640 = 비매치). 모호했던 "접두 토큰 포함"은 **삭제** — 역치환 break-모드(사용자가 *의도적으로* 구값과 유사한 새 값 요청 → 엔진이 구값으로 되돌림)의 주 통로라서. V0 break census가 이 모드를 1차 감시.
- **검출규칙 신설임 명시 (v1.1·코드 확정)**: t17이 v25e에 *잔존*한 이유 = `_provenance_deny`(t2_gate_patch.py:127)는 address를 검사하나(`DEFAULT_ARG_HINTS`:24에 "address" 有) 판정이 **소문자-substring 포함**(:139 `s.lower() not in ctx`)이라 접두-축약 "123 elm st" ⊂ "123 elm street" → **grounded 통과** → P-A/GROUND 채널 자체가 미발화. (일반화: 모든 접두-축약 오복사는 현 fab-검출의 구조적 구멍.) ∴ GROUND-VERBATIM = 치환 메커니즘은 P-A 재사용이나 **검출 규칙(substring-통과·비-exact-필드-일치)은 신설**·`_provenance_deny` 통과 뒤에 위치 → 독립 toggle·자기 마커([T2_GVERB])·자기 Δ계측 필수(P-A 실적에 합산 금지).
- **empty-값 처리(t39)**: v가 빈 값 ∧ |C|=1이면 치환 **후보**로만 — 엔진이 값을 '선택'하는 경계 사례라 **V0서 fix/break 실측 후 채택 결정**(break>0이면 empty는 제외하고 fuzzy만).
- 발화 조건 밖(exact-match·|C|≥2·C=∅)은 무개입. |C|≥2는 DISAMB 관할(§4·서로소 관할 원칙).

**A2 (ABox)**: `ground_verbatim_fields: ["address1","address2","city","state","zip","country"]` — 필드 목록만. 엔진 리터럴 0.
**[[05]]**: 값 원천=에이전트-기조회 문맥(DB 안 읽음·C34 클린). 치환=이미 GO된 P-A 채널.
**스코프 한계 (v1.1 명시)**: C 원천 = tool 출력만(`_grounded_candidates` 구조 동일). **사용자 발화-원천 값의 오복사**(사용자가 대화로 불러준 새 주소를 축약 write)는 이 레버 밖 — V0 분해에 "원문 원천(tool/dialogue)" 컬럼 추가해 크기 실측.
**검증**: V0 — COMP 167 실패 전수에 오프라인 적용 → fix/break census(t5c_v0_whitelist.py 확장·원천 컬럼 포함). GO = fix≥5 ∧ break=0. 단위 = 정규화 함수(축약/공백/빈값/다중후보/**숫자-토큰 불일치**/역치환 케이스).
**반대편 계측**: 치환-유발 flip(원래 맞던 값을 바꿈) = V0 break 카운트 + 라이브 Δspurious.

## 2. CALC 확장 — compound-criterion과 NL의 분리 (C·G클래스)

### 2a. 기준-주석 op 확장 (C클래스: 쓰기-선택 기준·decidable·§1.5 Q1)
기존 calc_specs(`count_where`·`sum`·nested/calc 채널=규칙0 클린) 위에 **일반연산 3종 추가**:
| op | 스펙 예(A2) | 닫는 표적 |
|---|---|---|
| `argmax_where` / `argmin_where` | get_product_details → "가장 비싼/싼 available 변형: <item_id·price>" | t20(최고가 업그레이드)·t36/t37(최저가 다운그레이드 후보) |
| `most_recent` | 주문 목록/타임스탬프 → "가장 최근 주문: <order_id·date>" | t71(argmax 날짜·C56 체계핵) |
| `pairwise_diff_sum` | 제안된 exchange (old,new) 쌍들의 가격차 합 → "오늘 지불 총액: $X" | t95-NL(총액)·t99류 |
- argmax/argmin·most_recent는 **에이전트-기조회 데이터 위 계산**(트리거=해당 tool 출력·주석으로 부착) = nested/calc 선례(calc_specs=trigger_tool-키·:210). 엔진=op(도메인일반)·A2=trigger/field 선언.
- **★pairwise_diff_sum 채널 (v1.1 정정)**: (old,new) 쌍은 tool *출력*이 아니라 **write 인자 + 기조회 가격**에서 옴 → calc-주석(trigger_tool) 채널에 안 맞음. **confirm-시점 notice 채널(§3b와 동일)**로 이동: exchange-class write의 confirm 직전, 인자의 item 쌍 × 문맥 가격으로 총액 주석. A2 = write-class + 가격 필드 선언.
- t37의 "조합이 예산 이하" 전역 제약은 argmin 주석+sum으로 재료 제공까지(조합 탐색은 에이전트 몫 — 조합최적화를 엔진이 풀면 도메인행동 경계·1차 스코프 밖).
- t79("다른 병과 같은 색")류 **cross-record attr-match는 여기 아님** → §4 DISAMB 열거(옵션 필드 표시)가 관할.

### 2b. ★G클래스 신규 실측 — compute-gap이 아니라 relay-gap이 절반 (t3 4-trial 프로브·2026-07-11)
**t3 실측**: calc `count_where`가 **4/4 정확 발화**("...variants: 10" 주입). 실패는 ① tr2/3: 에이전트가 그 수를 **사용자에게 끝내 안 말함**(NL met:false·"agent never...") ② tr1: "modify 하겠다" 선언 후 **write 미실행**(별개=write-loss). ⇒ **"calc 확장 ≈25–30 sims" (census §3 row4)는 과대추정 — 정정**: G클래스 = compute-gap(op 부재·t95총액)과 **relay-gap(계산돼 있는데 전달 누락·t3형)**의 혼합.
- compute-gap → §2a op 확장이 닫음.
- **relay-gap → calc로 못 닫음.** 후보 = CP5 walk의 **communicate-의무 확장**(재-plan이 "답해야 할 질문"도 추출 → 미답이면 리마인더) — E-PLAN CP5와 좌석 공유·**[D]·격리 프로브 先**(재-plan이 질문-의무를 안정 추출하는지 + NL diff의 결정론성 한계 정직 평가). 1차 스코프 밖·E-PLAN §5(d)에 프로브 항목 추가.

**검증**: 단위(op 3종) + 오프라인 재계산(COMP 궤적서 t20/t71/t95 트리거 시점에 주석이 정답을 내는지 = 무료 census) + 라이브 발화율([[30]] 스모크).
**★V0 결과 (2026-07-11·구현+census 완료·`calcext_offline_census.py`)**: ① 단위 28 PASS·A2 argmax/argmin 2스펙 부착. ② **t20 census: 4 product 중 3 MATCH**(Water Bottle·Keyboard·Makeup Kit) / Running Shoes = **구조적 MISS**(gold=주문과 같은 size 9 중 최고가 — 제약값이 주문-문맥 의존이라 정적 주석 불가·same-size 추론은 에이전트 몫 = 설계 의도 그대로). ③ **most_recent = retail 사망**: 전 456 sim·전 tool 출력에 날짜형 필드 **0건** → 스펙 미부착. **★t71 재라우팅**: "최근 주문"은 도구로 결정 불가(producer 부재) ⇒ **C48 위계상 ASK가 정답** — calc 아니라 DISAMB/ASK 채널로 이관(t71은 user-sim 오확인 고착 이력 있어 재확인형 한계도 계상·C56).
**반대편 계측**: 주석-과잉(무관 주석이 창 오염=C43 재료) → 주석은 트리거 tool 출력에만·개수 상한·Δfab 계측.

## 3. FEASIBILITY·NOTICE — B클래스 결정가능분 (t27·t57 일부)

### 3a. 배타-op ask 게이트 (t27형·`exclusivity_specs`)
- **실측 기전**(t27): 동일 주문에 반품+교환 동시 요청·"하나만 되면 교환 선호" → 에이전트가 반품 먼저 실행 → status 변경 → 교환 env-불가 → gold(교환만) 실패 3/4.
- **스펙**: `exclusivity_specs: [{classes:["item_return","item_change_delivered"], scope:"order_id"}]` (A2). 엔진: **E-PLAN ledger의 planned에 같은 scope의 배타-class write가 2+개**일 때 첫 write 시도를 **deny-once + "정책상 동시 불가·사용자에게 어느 쪽인지 확인하라" 피드백**(replay-safe regen 채널) → 에이전트가 ask → 사용자 선호(교환)만 실행.
- 결정성: 배타성=도구/정책 사실(decidable·Q1)·발화 조건=ledger 상태(결정론). **write 강제 0**(ask 유도만). C50과 구분: DB-state 아니라 도구스키마 사실 + 자기-plan 참조.
- **의존 (v1.1 정정)**: E-PLAN ledger(planned) **필수**. v1.0의 폴백("2번째 write 시도에 deny")은 **무효 판정·삭제** — t27의 해악은 *첫* write(반품 실행→status 변경→교환 env-불가)이고, 2번째 시도 시점엔 이미 회복 불가. 첫 write *전에* 배타-쌍을 알려면 plan(ledger)이 유일한 결정론 원천(대화서 두 intent를 첫 write 전에 읽는 건 semantic 파싱=C50 경계라 불가).
- **arm 충돌 해소 (v1.1)**: ledger를 **관측-전용 부품**으로 분리 — CP0 plan-추출+기록만·에이전트 창 불변·개입 0(주입·deny·리마인더 없음). 관측-전용 ledger는 stage B 스택에 동거 가능(교란 0·추가 비용=plan 서브콜 1회/sim뿐). E-PLAN *arm* 소속은 개입 레버(discovery-enforce·CP5 walk)만. → `E_PLAN_..._DESIGN`에도 이 분리 명문화 필요(v1.3 항목).

### 3b. 환불-목적지 notice (t57형) — **⛔superseded by `NEXT_LEVER_GEN_DESIGN §1.3` G8_REFUND_NOTICE (2026-07-11)**
> 같은 표적(t57)을 **정적-문구 notice 게이트(G8)**가 대체 — 동적 `<pm>` 주석안은 notice_text 동적-값 금지 규약(NEXT_LEVER §1.4)과 충돌해 폐기. 이중 구현 금지. 아래는 기록 보존.
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
- **순서**: ① 각 레버 V0/오프라인 census+단위(무료·병렬 가능) → ② GO 레버만 스택에 → ③ 표적 26 task(기지 13+신규 SYSTEMIC 13) × nt=1 사이클 → ④ per-case → 사이클 반복. E-PLAN *개입 레버*(discovery·walk)는 **별도 arm 유지**(합산 금지). §3a는 **관측-전용 ledger**(§3a v1.1·창 불변) 배선 후 stage B 동거 가능 — 단 첫 사이클은 §3a 제외 권장(ledger 부품 자체의 단위·격리 검증 선행).
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
