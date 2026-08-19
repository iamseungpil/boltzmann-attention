# calc 레버 pass 출처 추적 — "올린 적 있는가 / 그것이 gold 였는가"

작성 2026-08-19 · 근거 = `sim_results/` 전수 재검산 + A2/엔진 축자 + git 이력
관련 규율: [[23]](A2 는 gold 를 보지 않는다) · [[03b]](구현 cheating 금지) · [[62]](레버 전에 결손을 재라) · [[54]](분모 다른 런 비교 금지) · [[08]](집계에서 결론 직행 금지)
선행: `OPERAND_LEVER_AUDIT_2026_08_19.md` · `BANK_COMPUTE_OP_KEYSTONE_DESIGN_2026_07_13.md` · 제거 커밋 `b220745d`

---

## §0. 판정 요약

| 질문 | 답 |
|---|---|
| **Q1. calc 계열이 pass 를 올린 적이 있는가** | **딱 한 번 있다.** `bank_redesign4_20260719` 0/1 → `bank_redesign5_20260719` 1/1 (task_026·seed 동일·1변수). 그 외 짝비교는 전부 **0 또는 음(−)**. |
| **Q2. 그 pass 는 gold 참조였는가** | **전량 gold 참조다.** Δ 를 만든 것은 상수 `display_round_up_frac=0.9` 하나이고, 도입 커밋 축자가 채택 근거를 *"gold census 10/10 fit"* 로 적었다. 정책 축자(전면 truncate)를 **어기고** gold 에 맞춘 값이다. ⇒ **실험 무효·인용 금지**(§4). |
| Q3. 정당한 calc 는 있는가 | 있다(§3 표 8종). 다만 **정당함 ≠ 효과 실증됨** — 그 8종 중 짝비교로 pass 이득이 측정된 것은 **0종**이다. |

**한 문장:** 사용자가 기억하는 "pass 를 올렸었다"는 실재하지만, 그 실체는 **정책을 어기고 gold 재현율로 고른 상수 한 개가 태스크 하나를 뒤집은 n=1 사건**이고, 우리 A2 가 이미 스스로 그 런을 *"tainted 검증런"* 이라 표시해 두었다.

---

## §1. Q1 — 짝비교 전수

### 1-1. 성립한 짝

| # | 레버 | OFF arm | ON arm | Δ | 짝 성립 근거 |
|---|---|---|---|---|---|
| ① | **`display_round_up_frac=0.9`** (scaffold_get_tools[0] 표기 상수) | `bank_redesign4_20260719` **0/1** | `bank_redesign5_20260719` **1/1** | **+1 (n=1)** | task_026 trial 0 · **seed 626729 동일** · commit `5ebebbe827b4` 동일 |
| ② | **T2_RESOLVE** (reference-filter silent-repair 포함) | `bankar_rec_g5` **3/5** | `bankar_rec_gr5` **0/5** | **−3** | seed 300·commit `5ebebbe827b4` 동일·태스크 집합 동일(001/003/007/016/023) |
| ③ | **T2_MATCH_COUNT** | `bank_ax33n_gpu{0,1}_20260803g` **24/64** | `bank_b4_gpu{0,1}_20260803h` **24/64** | **0** (5↑5↓ 상쇄) | 동일 32 태스크·nt=2. ⚠단일변수 아님(이관 통행료 정정과 묶임)·동시 실행 아님 |
| ④ | **T2_CALC** (retail) | `prov_e2e_retail_t4` **263/456** | `comp_retail_t4` **289/456** | +26 | 태스크 집합 동일. ⚠**묶음 arm**(게이트6종+prov+nested+calc) — calc 귀속 불가 |
| ⑤ | (부정통제) T2_HAVE_VALUE | `bank_hve2e9_base_20260723` **1/8** | `bank_hve2e9_hv_20260723` **0/8** | −1, **발화 0회** | 단일변수. **잡음 바닥 = ±1** 계측치 |

**②의 성격**: 양 팔 16 sim 전부 `reward_basis=['DB']`. 이 Δ 는 인자 정오가 아니라 흐름 탈선이다 — ON 팔이 KB_search 를 늘리고 016·023 을 transfer 로 몰아 write 자체에 도달하지 못했다. reference-filter 는 `gold_touch=false`(A2 `_note_reference_filter` 축자 *"출처 = env 레지스트리 + KB 문서"*)이므로, **gold 를 안 쓴 calc-치환도 손해를 낼 수 있다**는 대조군이다.

**⑤의 용도**: 레버가 **한 번도 발화하지 않은** 8 태스크 짝이 1 만큼 흔들렸다. 따라서 |Δ|≤1 은 신호가 아니다. 위 표에서 이 바닥을 넘는 것은 **②(−3) 하나뿐**이고, ①은 n=1 이라 바닥 판정 대상이 아니라 **동일 seed 1변수 대조**로만 성립한다.

### 1-2. 성립하지 않은 짝 — 그리고 왜

| 레버 | 왜 짝이 없나 |
|---|---|
| **T2_COMPUTE** (사용자 질문의 표적) | **OFF 팔이 역사상 0건.** `grep -rn 'T2_COMPUTE=0\|unset T2_COMPUTE'` 레포 전체 = **0건**. 이 이름을 쓰는 셸 스크립트 **42개가 전부 `=1`**, `go_stack.sh:63` 축자 `export T2_COMPUTE=1 T2_RESOLVE=1 ...` 로 상시 ON. ⇒ [[60]] 전-레버-상시-ON 정책의 귀결이며, **이 레버의 격리 A/B 는 존재한 적이 없다.** |
| **T2_SCAFFOLD_GET** (우산) | ON/OFF 는 배너로 직접 읽힌다(로그 373 중 343 이 `[t2_run] SCAFFOLD-GET ON`). 그러나 배너 없는 30건이 **전부 단일-태스크 격리 프로브**라 로스터가 안 맞는다. |
| **개별 scaffold_get op 10종** | A2 스펙 단위 ON/OFF 를 기록한 런이 없다. |

### 1-3. 표적 태스크 통과 이력 — 전 코퍼스 전수 (본 문서 재검산)

`sim_results/*.json{,.gz}` 중 `simulations` 를 가진 **460 파일** 전수. reward 는 이 코퍼스에서 이진이므로 pass ≡ `reward == 1.0`.

| 태스크 | 겨눈 기구 | 통과 |
|---|---|---|
| task_085 | `compute_ops.customer_max_liability_amount` | **0 / 20** |
| task_093 | `compute_ops.amount_difference` | **0 / 20** |
| task_094 | `compute_ops.amount_difference` | **0 / 17** |
| task_040 | reference-filter silent-repair | **0 / 43** |
| task_026 | `display_round_up_frac` | **1 / 40** ← 그 1건이 §2 |

**gold 로 맞춘 상수를 실었던 두 compute_op 의 표적은 도입 前(2026-07-11 floor)부터 제거 當日(2026-08-19)까지 단 한 번도 통과한 적이 없다.** `bank_t7326_halfB_20260819q` task_085 궤적에는 엔진이 써 넣은 `customer_max_liability_amount: 50` 이 실물로 박혀 있는데도 같은 호출의 `transaction_id` 가 날조 `tx123456`·`card_id` `unknown` 이라 reward 0.0 이다 — **채점 인자를 대신 채워 줘도 pass 로 바뀌지 않았다.**

### 1-4. 반전 사례 — calc 출력이 **동일한데** pass 가 갈린다

`bank_t7326_halfA_20260819q` · task_017 · 같은 런 · 같은 엔진 반환:

> *"...update that transaction's rewards to EXACTLY the correct value shown: txn_913d14a20dc5 (recorded 15 points, correct 156 points); txn_cfabb609133d (recorded 47 points, correct 87 points)"*

| trial | 엔진 반환 | `update_transaction_rewards_3847` **실행** | db_match | reward |
|---|---|---|---|---|
| 0 | 위 문면 수신 | **2회** | false | **0.0** |
| 1 | 위 문면 수신 | **0회** | true | **1.0** |

task_017 의 gold action 은 `['log_verification','give_discoverable_user_tool','call_discoverable_user_tool','call_discoverable_user_tool']` — **update 액션이 아예 없다.** 즉 calc 는 값을 정확히 계산해 배달했고, 정답은 **그 write 를 하지 않는 것**이었다.

같은 구조가 task_024 에도 있다(t0/t1 이 `check_card_application_fit {business:true, min_credit_limit:40000}` 를 **인자까지 동일하게** 호출했는데 0.0 ↔ 1.0).

⚠정확히 읽을 것: task_017 은 전 코퍼스 **28/54** 로 통과율이 낮지 않다. 이 표가 말하는 것은 *"calc 가 017 을 죽인다"* 가 아니라 **"calc 출력이 동일해도 pass 가 갈린다 ⇒ calc 는 pass 를 설명하지 못하고, 어떤 자리에서는 과-행동 방향으로 민다"** 이다.

### 1-5. 왜 인자를 맞춰도 안 사는가 — `reward_basis` 가 답이다

표적 태스크군의 basis 는 **전부 DB**다(085 DB 19 · 093 DB 18 · 094 DB 17 · 040 DB 40 · 026 DB 37 · 017 DB 54 · **ACTION basis 0건**). DB basis 에서 점수는 *"엔진이 인자를 gold 로 맞췄는가"* 가 아니라 **"최종 DB 가 gold DB 와 같은가"** 다. 그러려면 값뿐 아니라 **어느 write 를 하고 하지 않을지**까지 맞아야 한다. 그래서

- 인자 하나를 gold 로 치환해도 나머지가 틀리면 0 (085: `0->50` 을 8~18회, 통과 0),
- 값이 맞아도 하지 말아야 할 write 를 하면 0 (017 t0).

---

## §2. Q2 — 그 한 번의 pass 를 축자로 해부

### 2-1. 짝의 동일성 (검증 완료)

| 항목 | redesign4 | redesign5 |
|---|---|---|
| git_commit | `5ebebbe827b455b3ed04fcb9294235c6ef4e5fd6` | 동일 |
| task_026 trial | 0 | 0 |
| **sim seed** | **626729** | **626729** |
| reward_basis | `['DB']` | `['DB']` |
| **reward** | **0.0** | **1.0** |

### 2-2. Δ 의 전량 = 한 토큰

두 런의 엔진 도구 반환은 축자로 비교했을 때 **마지막 한 값만** 다르다.

- redesign4: `... txn_a8f1c2d3e411 (recorded 600 points, correct **1499** points)`
- redesign5: `... txn_a8f1c2d3e411 (recorded 600 points, correct **1500** points)`

앞의 세 값(6300 / 1020 / 3800)은 완전 동일. 모델은 양쪽 모두 그 값을 **그대로 복사**해 4건을 write 했고, `db_match` 가 False → True 로 뒤집혔다.

### 2-3. 그 1499→1500 을 만든 것

도입 커밋 축자 (`418428c3`):

> `display rounding: A2-declared display_round_up_frac=0.9 for expected_disp (**gold census 10/10 fit**, user-approved; closes task_026 one-point residual)`

철회 후 A2 축자 (`a2/banking_knowledge.gate.json:658` `_note_rounding`):

> *"2026-07-19 표기=floor 확정. KB doc_007 'Rewards Points Rounding Policy'가 전면 truncate 명시(249.975→249·무예외). task_026 gold 1500(=1499.9)은 벤치 자체 정책 위반 = gold 저작 버그(전수 census 9/10 floor·유일 예외). **display_round_up_frac=0.9 안은 gold-fitting이라 사용자 지시로 철회(redesign5=tainted 검증런).**"*

### 2-4. 판정

**정책 축자는 truncate(floor) = 1499 다. gold 만 1500 이다.** 상수 `0.9` 는 정책에서 유도되지 않았고 **gold census 적합률로 선택**됐다 ⇒ [[23]] 위반 ⇒ 그 pass 는 **실험 무효**. 우리 A2 가 스스로 `redesign5=tainted 검증런` 이라고 표기해 두었으므로 이는 **소급 판정이 아니라 기존 판정의 재확인**이다.

철회 후 task_026 = **0/38**, 전 코퍼스 통산 **1/40**.

현재 A2 4파일에 `display_round_up_frac` 이 **스펙 키로 선언된 곳은 0건**이다(3건의 grep 히트는 전부 `_note_rounding` 산문 안의 회고 언급). 다만 **기구는 잔존한다** — `t2_compute.py:900-902` 축자 `thr = spec.get("display_round_up_frac")`.

### 2-5. 나머지 gold 경유 두 op — 무엇을 사고 무엇을 팔았나

`compute_ops` 두 개의 제거 근거는 커밋 `b220745d` 본문에 축자로 있다:

> *"one of their constants was chosen by how well it reproduced gold: bank_rule_fit.py sweeps thresholds against reward_info.action_checks, and the A2 note records \"T1=2 (policy literal) 73.6% / T1=30 (proxy) 89.4%\". The policy text says 'within 2 business days of statement'; the shipped value was 30."*

`bank_rule_fit.py` 헤더 축자(:3-6):

> *"It fits thresholds against reward_info.action_checks -- that is gold. Choosing a constant because it reproduces gold better is exactly the violation [[23]] names, and the engine then fills a scored argument, which erases the deficit [[62]] measures."*

**지불한 것**: [[23]] 위반 + [[62]] 결손 소거(엔진이 채점 인자를 채우면 측정 대상이 사라진다).
**산 것**: §1-3 대로 **0 pass**.

---

## §3. Q3 — 정당한 calc 와 부당한 calc 의 경계

### 3-1. 경계 기준 (판정 규칙)

| | 정당 | 부당 |
|---|---|---|
| 값·상수의 출처 | 정책 산문 축자 / KB 문서 / env 레코드 / 도구 스키마에서 **기계적으로 유도** | `reward_info` · gold 액션 · gold 재현율 |
| 선택 기준 | 정책이 그렇게 말하므로 | **gold 를 더 잘 재현하므로** |
| 정책과 충돌할 때 | 정책을 따르고 불일치를 **결함으로 기록** | gold 에 맞춰 정책을 어김 (= §2 사례) |
| 배달 방식 | ⓐ 도구-결과로 **돌려줌**(모델이 옮겨 적어야 점수에 닿음) · ⓑ 값을 **빼는**(drop/deny) 것 | 채점되는 인자에 **직접 써 넣음**(in-place silent-repair) |

> ⚠ⓐ/ⓑ 축과 gold 축은 **독립**이다. ②(reference-filter)는 gold 를 안 쓰고도 −3 이었다. 그래서 "정당하다"는 [[23]] 무결을 뜻할 뿐, **효과가 있다는 뜻이 아니다.**

### 3-2. 정당한 calc — 현존 목록 (gold 무접촉이 축자로 확인된 것만)

| 항목 | 값의 출처 (축자) | 배달 | 짝비교 측정 |
|---|---|---|---|
| `scaffold_get_tools[1]` **verify_identity** | *"출처=정책 축자(…) + 도구 동작(env: 조회가 레코드를 못 찾음 …). **gold 경유 0**"* | ⓐ 판정 문면 | 없음 |
| `scaffold_get_tools[2]` **check_rebate_qualification** | *"출처=정책 산문(KB doc_007/010: 매월 충족 iff·한 달 미달=실격·프로레이션 없음) + 도구 스키마 … **gold 무참조([[23]])**"* | ⓐ | 없음 |
| `scaffold_get_tools[5]` **check_card_closure_eligibility** | 판정식(잔액≤0)은 정책 소관. `_note` 의 gold 언급은 규칙 출처가 아니라 실패 서술 | ⓐ | 없음 |
| `scaffold_get_tools[6]` **check_cli_eligibility** | *"문구=양방향(approve\|deny)·default None=미지 카드타입 abstain. **gold 무관·전거는 KB**"* | ⓐ | 없음 |
| `scaffold_get_tools[7]` **check_card_application_fit** | *"category_rates/base_cashback=**KB 원문 축자 재구조화**(11행·2026-07-27 전수 대조)"* · *"엔진=조회·주석만(선택·순위·값생성 0)"* | ⓐ | 없음 |
| `scaffold_get_tools[9]` **get_checking_atm_fee_totals** | *"요율 출처 전부 정책 문서 축자(ATM_FEE_SCHEDULE_VERBATIM_2026_08_13.md + …)·**gold 미접촉**"* | ⓐ | 없음 |
| **T2_LEDGER** / `t2_ledger.facts_text` | LLM 이 구조화한 행 + A2 키 이름·창 상수. **엔진 파서 전부 제거**(축자 *"그래서 파서를 전부 제거했다"*) | ⓐ | 없음 |
| **T2_MATCH_COUNT** | KB 코퍼스 파일 + 질의 내용어 | ⓐ | **있음 → Δ 0** (표 ③) |
| **T2_SG_GROUND** / **T2_PROD_BIND** | 도구 출력 원장 substring 실재 검사 → **미검증 값을 뺀다**(op→None→abstain) | **ⓑ 빼기형** | 없음 |
| `t2_compute.apply_op` · `t2_offload.run` | 라이브러리(도메인 리터럴 0). 헤더 축자 *"★**operand는 엔진이 만들지 않는다**"* | — | 해당 없음 |

**ⓑ(빼기형)가 [[63]] 과 정합한다** — 모델이 못 하는 것은 배제이고, `T2_SG_GROUND`/`T2_PROD_BIND` 는 값을 만들지 않고 근거 없는 값을 **제거**만 한다. 이 축은 아직 짝비교로 재본 적이 없다.

### 3-3. 부당하거나 오염된 calc — 현존/이력 전수

| 항목 | 상태 | gold 경유 축자 |
|---|---|---|
| `compute_ops.customer_max_liability_amount` | **제거**(b220745d) | *"T1=2(정책literal) 73.6% / T1=30(proxy) **89.4%**"* · 정책은 *"within 2 business days"* 인데 실린 값 30 |
| `compute_ops.amount_difference` | **제거**(b220745d) | *"2026-07-16 확장(**gold-fit 확증**·rule_fit.py)"* · *"the engine filled a scored argument outright"* |
| `display_round_up_frac=0.9` | **철회**(스펙 선언 0건 · 기구 `t2_compute.py:900-902` 잔존) | 도입 커밋 *"**gold census 10/10 fit**"* |
| `scaffold_get_tools[0]` **get_reward_discrepancies** | 가동 중 | `_note_details` 축자 *"floor 표기=**gold 실증**(95=floor(95.66) 등 6/6)"* — 표기 규칙이 gold 대조로 확정 |
| `scaffold_get_tools[3]` **get_correct_savings_apy** | 가동 중 | `_note` 축자 *"… **gold 095 재현 6.85**"* — 스택 규칙 채택 확인이 gold 재현 |
| `scaffold_get_tools[4]` **get_interest_correction** | 가동 중 | `_note` 축자 *"구판 days/365 일할은 **gold(98.00=96000x1.225%/12)와 불일치**=엔진-의미론 결함 2호"* — **식(일할↔월할) 선택이 gold 불일치로 뒤집힘** |
| `scaffold_get_tools[8]` 의 **환급-차감 확장** | **보류·비활성** | `_note_rebate_field` 축자 *"근거(x323 n=24·072 **원장 검산 14.00=gold**)"* (본문 요율 자체는 정책 축자·무참조) |
| `bank_rule_fit.py` | STOP 헤더 부착·산출물 A2 반입 금지 | 도구 자체가 gold 코퍼스 스윕 |
| `bank_eplan_controller.py:332-374` | **제거 범위 밖 — 인라인 사본 잔존** | `:352-356` 축자 `# (4) dispute 판정: agent가 liability만 틀림(**입력 gold**)` · `gold = {... "customer_max_liability_amount": "50"}` · `:375` `assert v["amount_difference"] == 33.0   # **gold**: 0.275%×12000=33.0` |

**경계 사례 — `ref_verify`**: 술어의 **범위**(cross-merchant 한정)가 gold 무오차단 대조로 확정됐다(`_note` 축자 *"merchant-absence deny=슬립 8/8 검출·**gold 25/25 통과(false-block 0**·efiso_detmatch_proof.py)"*). **값을 산출하지 않으므로 [[23]] 위반은 아니다** — 다만 그 25/25 는 gold 대조 수치이므로 인용 시 반드시 그 성격을 명시해야 한다(§4-B).

---

## §4. 인용 금지 목록

### 4-A. [[23]] 위반 — 실험 무효, 성적으로 인용 금지

| 수치 | 출처 | 왜 무효 |
|---|---|---|
| **+348** (`customer_max_liability_amount` 순 교정) | `BANK_COMPUTE_OP_KEYSTONE_DESIGN_2026_07_13.md`:205 | 단위가 **pass 가 아니라 필드**. gold 인자 일치 수 |
| **오치환 27/431 → 교정 375/414** | 동 §8-3 | 동상 |
| **+366 · 755회 발화 · gold 일치 90.9%** (오프라인 replay) | 동 §8-5 (*"오프라인 replay(755·90.9%·+366)가 정량 정본"*) | frontier 궤적 위 gold 재현율. 라이브 pass 아님 |
| **89.4% / 73.6% / 94.7% / 86.8%** (T1 재현율) | A2 `_note_compute_ops` | **임계 자체를 이 수치로 골랐다** — 순환 |
| **"gold census 10/10 fit"** | 커밋 `418428c3` | `display_round_up_frac=0.9` 채택 근거 |
| **`bank_redesign5_20260719` task_026 1/1** 및 그로부터 파생된 모든 task_026 통과 주장 | §2 | A2 자기 표기 *"redesign5=tainted 검증런"* |
| **"gold 095 재현 6.85"** · **"gold 095 재현 98.0"** | A2 `scaffold_get_tools[3]/[4]` `_note` | 스펙 채택 확인이 gold 재현 |
| **"floor 표기=gold 실증 6/6"** | A2 `scaffold_get_tools[0]` `_note_details` | 표기 규칙이 gold 대조로 확정 |
| **"072 원장 검산 14.00=gold"** | A2 `scaffold_get_tools[8]` `_note_rebate_field` | 확장 근거가 gold 대조(현재 보류) |

### 4-B. 조건부 — 인용 가능하나 성격 명시 필수

- `ref_verify` 의 **"gold 25/25 통과(false-block 0)"** — 값 산출이 아니라 **거부 술어의 오차단 부재 검정**이다. "성능"이 아니라 "부작용 없음 확인"으로만 인용하고, gold 대조임을 반드시 병기하라.

### 4-C. gold 문제는 아니지만 **비교 불가** — 레버 효과로 읽지 말 것 ([[54]])

- `bankxfer_floor_bank_t4` ↔ `bank_t7326_*_20260819q` 의 **+11.5pp** (2/44 → 7/40): **user-sim 이 gpt-4.1 ↔ gpt-5.2 로 다르고** 003/004/016/017 은 시행수가 3↔2 다. 게다가 그 7 pass 중 calc 가 관여한 3건(017t1·024t1·050t1)은 §1-4 대로 pass 를 설명하지 못하며, 050 의 채점 write `{requested_increase_amount: 2500}` 는 **사용자가 말한 수치**이지 calc 산출물이 아니다.
- `comp_retail_t4` ↔ `prov_e2e_retail_t4` 의 **+26**: 묶음 arm(게이트6+prov+nested+calc)이라 calc 귀속 불가. 게다가 gpt-4.1 user-sim 이고, 같은 레포 §6d 가 gpt-4.1 floor 를 아티팩트로 판정했다.
- 표 ③의 **Δ 0** 은 "무변화"가 아니라 **5↑5↓ 상쇄**다. 잡음 바닥(±1)을 감안하면 null 이지 "영향 없음의 증명"이 아니다.

---

## §5. 계기 한계 — 이 추적이 보지 못한 것

1. **`bank_t7326_*_20260819q` 의 `.log.gz` 가 영속되지 않았다.** results 만 남아 있다. 커밋 본문의 *"compute silent-repair … **8회**"* 와 *"reference-filter silent-repair transaction_id->txn_… **3회**"* 는 **커밋 인용이지 재검산 수치가 아니다.** (동형 발화는 `bank_t7295_a_20260815n.log.gz` 16회·`bank_n97_gpu1_main_20260805.log.gz` 18회에서 확인되며 해당 런의 085 는 각각 0/4·0/2.) 치환 **결과물**은 궤적 인자에 남아 독립 확인됐다.
2. **`bankar_rec_g{,r}5` 의 `.log.gz` 도 없다.** ②의 −3 은 궤적 본문(도구 호출 계수)으로만 기전을 봤고, silent-repair 발화 횟수는 확인 불가다.
3. **`T2_COMPUTE` 의 ON/OFF 를 명시 기록한 런 로그가 하나도 없다.** [[60]] 상시-ON 정책의 귀결이다. 그래서 이 레버의 격리 A/B 는 **존재한 적이 없고**, §1 의 비교는 전부 *도입 前後* 또는 *동일-출력 trial 대조*로 대체한 것이다.
4. **`t7328` 결과가 0건이다.** `find -iname '*t7328*'` = 런처 `.sh` 하나뿐. 즉 "제거가 pass 를 깎았는가"에 답할 짝은 **설계돼 있으나 아직 비어 있다.**
5. **`bank_eplan_controller.py:332-374` 의 인라인 사본**이 라이브에 배선돼 있는지 확인하지 못했다. `b220745d` 의 제거 범위 밖이라 남았고, `:352-356`/`:375` 가 축자로 gold 를 대조한다.
6. **reward 가 이진이고 basis 가 DB 지배**라, 인자 단위 정오와 점수 사이에 항상 최소 한 다리가 있다. 코퍼스 전체 basis 분포에서 ACTION 은 소수이고 **우리 표적 태스크군에는 ACTION basis 가 0건**이다.
7. **잡음 바닥은 ±1(8 태스크 기준)** 로만 계측됐다(⑤ 한 짝). 더 큰 로스터에서의 바닥은 미측정이다.

---

## §6. 무엇을 재야 하는가

> [[60]] 에 따라 아래는 전부 **측정 항목**이다. 레버 끄기 권고가 아니다.

| # | 재야 할 것 | 어떻게 | 왜 지금 |
|---|---|---|---|
| M1 | **`compute_ops` 제거가 pass 를 깎았는가** | `run_t7328_rebaseline_20260819.sh` 실행. 로스터·PIN·NT·max_steps 가 t7326 과 동일하게 이미 고정. 사전등록 종점도 런처에 축자로 기재: *"endpoint = per-task pass vs t7326 7/40"* | 이 자리가 비어 있는 한 §1 의 T2_COMPUTE 행은 영원히 "짝 없음"이다 |
| M2 | **gold 경유 3종의 정책-재저작판 vs 현행판** (`scaffold_get_tools[0]`·[3]·[4]) | 정책/KB 축자만으로 스펙을 다시 쓰고, 같은 로스터·같은 시드로 현행판과 짝비교 | 지금 이 3종의 수치는 전부 §4-A 라서 **인용할 수 있는 값이 없다**. 재저작판이 그 자리를 채운다 |
| M3 | **`bank_eplan_controller.py` 인라인 사본의 라이브 배선 여부** | `bank_perstep_decomp.py:84` 등 소비측 추적 + 라이브 궤적에서 발화 검색 | 제거가 반쪽이면 §2-5 의 지불이 계속된다 |
| M4 | **빼기형(ⓑ) calc 의 효과** — `T2_SG_GROUND`·`T2_PROD_BIND` | 단일변수 짝. [[63]] 예측(모델은 배제를 못 한다)의 직접 검정 | ⓑ 는 [[23]] 무결이면서 아직 한 번도 안 재봤다 |
| M5 | **잡음 바닥 재계측** | 발화 0 인 부정통제 arm 을 t7326 급 로스터(20태스크×2)에서 1회 | 현재 바닥 추정치가 n=8 짝 하나에서 나왔다 |
| M6 | **로그 영속 복구** | 런처가 `.log.gz` 를 results 와 함께 남기도록 | §5-1·5-2 가 재발하면 발화 계수는 영구히 커밋 인용에 의존한다 |
| M7 | **Δspurious** — 각 calc 레버의 over-action 역효과 | §1-4 의 017/024 패턴(값은 맞고 write 가 틀림)을 지표화 | RESEARCH_MASTER §1 제1원리: 부작용 없는 레버는 없다 |

---

## §7. [[62]] 에 대해 이 결과가 말하는 것

결손을 격리로 재기 **전에** 지은 결정론기라서 실패한 것이 아니다. **결손을 gold 로 재고, 그 gold 재현율을 성적표로 삼았기 때문에** 실패했다. 그러면 *"레버가 산 것"* 과 *"우리가 답을 넣어 준 것"* 이 같은 수가 되어 구분이 **원리상** 불가능해진다 — §4-A 의 수치들이 전부 그 상태다.

현재 calc 축에 대해 말할 수 있는 실측 문장은 이것뿐이다:

> **표준 짝비교에서 calc/resolve 계열이 pass 를 산 사례는 1건(n=1)이고 그것은 gold 참조였다. 판 사례는 1건(−3) 있다. 나머지는 전부 0 이거나 짝이 없다.**
