# T7336 실패 마스터 종합 — 27 실패 sim 전수 · 우리-층 결손 명부 · 처방/프로브 큐 (2026-08-22)

> 이 문서는 `t7336_tasks/T7336_TASK_*.md` **15편**(27 실패 sim per-step) + 반쪽 종합 2편
> (`T7336_FORENSIC_HALFA_2026_08_22.md`·`T7336_FORENSIC_HALFB_2026_08_22.md`) + 별건 감사
> `CLAIM_DEMAND_ISO_VS_LIVE_AUDIT_2026_08_22.md`(C593) 를 전수 정독해 만든 **종합**이다.
> 새 수치를 만들지 않았다 — 모든 수치는 위 18편의 축자 인용이고, 없는 수치는 "미측정"으로 남겼다.
> gold(`reward_info`)는 원 보고서들이 진단용으로만 읽었고, 이 문서는 그 판정을 옮길 뿐 A2 내용을
> 저작하지 않는다([[23]]). **수리·코드 수정 0** — 처방은 큐로만 남긴다.

---

## §1. 성적표 · 태스크별 부호표

### 1.1 런 성적 (전부 인용 수치)

| 런 | halfA | halfB | 합 | 등급 |
|---|---|---|---|---|
| `bank_t7328_*_20260819r/r2` (기준선) | **4/20** (073#0·004#1·100×2) | **2/20** (098×2) | **6/40** | 원장 C590 |
| `bank_t7336_half{A,B}_20260821b` (수리 스택 1단계) | **9/20** | **4/20** | **13/40** | gz 직독 검산·sha 상이라 **[M]** |

부수 계측(원 보고서 축자):

| 축 | halfA | halfB |
|---|---|---|
| `compliance.json` | bench pass^1 **0.45** · g3 위반 **7 sim** | bench pass^1 **0.20** / pass^2 **0.10** / full pass^1 **0.05** |
| 종료 사유 | STOP 14 · TRANSFER 4 · `too_many_errors` 1(072#0) · `###OUT-OF-SCOPE###` 1(017#1) | `user_stop` 18 · **`context_window_exceeded` 2**(074#0 · 079#1) |
| 채점표 결손 | 072#0 `reward_basis=null` | 074#0 · 079#1 `reward_info` 부재 |

⚠ **채점 축은 태스크마다 다르다**(C583ⓖ). 20 태스크 중 **033 만 `reward_basis=["ACTION"]`** 이고
나머지는 전부 `["DB"]` 다. `action_checks` 는 진단 보조일 뿐 성적이 아니다([[69]]) — 실물 반례가
두 건 있다: 017#0 은 `action_checks` 2건 불일치인데 reward 1.0 · 050#1 은 `050_10` 불일치인데 1.0.

### 1.2 태스크별 부호표 (t7328 → t7336 · 분모 2)

| 부호 | 태스크 | t7328 | t7336 (t0/t1) | 판정 근거 |
|---|---|---|---|---|
| **+2** | **003** | 0/2 | 2/2 (1·1) | 표적 레버 신설 **없음** → 표본 2의 분산 가능성 유보(halfA §1) |
| **+2** | **024** | 0/2 | 2/2 (1·1) | 실패 sim 아님 → 포렌식 대상 밖(기전 미조사) |
| **+1** | **004** | 1/2 | 2/2 (1·1) | ACTION 축 태스크 |
| **+1** | **017** | 0/2 | 1/2 (1·0) | t1 만 실패 — 검증 단계에서 'name' 을 끝내 안 물음 |
| **+1** | **033** | 0/2 | 1/2 (0·1) | 분기 = 검색 채널(`shell grep` ↔ `KB_search_dense`) |
| **+1** | **050** | 0/2 | 1/2 (0·1) | 동일 분기. DUP 은 소멸·t0 는 발견 실패 후 MISSING |
| **−1** | **073** | 1/2 | 0/2 (0·0) | **§4 전용 절** — P5 문면 개정이 NET 집계 지시를 팔았다 |
| 유지 2/2 | **098** · **100** | 2/2 | 2/2 | 불변 의무 태스크 |
| **0/2 잔존 11종** | 016 · 040 · 055 · 057 · 063 · 072 · 074 · 079 · 085 · 093 · 094 | 0/2 | 0/2 | 세 런(t7328·t7335·t7336) 연속 0 |

**합 검산**: 003(2)+024(2)+004(2)+017(1)+033(1)+050(1)+098(2)+100(2) = **13** ✓ ·
t7328 = 004(1)+073(1)+098(2)+100(2) = **6** ✓

### 1.3 ★증가분 +7 의 귀속 — 수리 7건 중 어느 것도 귀속되지 않았다

15편 어디에도 *"수리 N이 이 pass 를 샀다"* 는 인과 진술이 없다.

- **033 t1 · 050 t1** (분석된 유일한 신규 pass 2건): 두 보고서 모두 분기를 **모델의 검색 도구 선택**
  (`shell grep` 라인 파편 ↔ `KB_search_*` 문서 본문)으로 확정했다. 코퍼스 대조 4/4 일치
  (033 §4.2: 라인 회수 0/3 ↔ 본문 회수 2/2). 수리 7건은 이 분기에 개입하지 않았다.
- **003 · 004 · 017 t0 · 024**: 실패 sim 이 아니라 per-step 포렌식 대상 밖이다 — 기전 **미조사**.
- ⇒ 정직한 진술: **13/40 은 관측이고, 그 안의 +7 은 수리에 귀속된 바 없다.** t7328↔t7336 은 sha 가
  다르므로 애초에 엄밀 A/B 도 아니다([M]).

---

## §2. 원인 축별 군집표 (축은 27 sim 의 결정 지점에서 나왔다)

각 실패 sim 에 **reward 를 죽인 결정 지점 하나**를 기준으로 1차 축을 배정했다(2차 축은 병기).
27 sim = 15 태스크 × 실패 trial.

### 2.1 축 정의와 sim 수

| 축 | 이름 | sim 수 | 태스크(#trial) | 귀속 주체 |
|---|---|---|---|---|
| **축2** | **우리 층 산출물이 결정점을 오염** — 엔진/격리 서브가 만든 값·이름·판정이 권위 문장으로 실려 인자·결론이 됨 | **7** | 016#0 · 016#1 · 063#1 · 085#1 · 093#0 · 093#1 · 094#1 | our_layer |
| **축5** | **결정점에 있어야 할 것이 없다** — 재료·지목·브레이크의 전달/타이밍 결손(축-소진·cap·지문 억제·진입 술어 사각) | **5** | 033#0 · 055#0 · 055#1 · 063#0 · 074#1(혼합) | our_layer |
| **축1** | **우리 층 자기차단** — 우리가 댄(또는 모델이 낸) 정답 이름·호출을 같은 층의 다른 가드가 막음 | **4** | 040#1 · 050#0 · 085#0 · 094#0 | our_layer |
| **축7** | **모델: 발견/탐색 재개 실패** — 값이 문맥에 **없고** 검색 채널을 재개하지 않음 | **3** | 057#0 · 073#1 · 079#0 | model |
| **축4** | **우리 층 문면의 fix-naming 실패([[64]])** — 거부는 했는데 해소책이 없거나 틀림/이행 불가 | **3** | 072#0 · 072#1 · 073#0 | our_layer |
| **축6** | **모델: knowing-doing** — 값·지시가 문맥에 **축자 실재**하는데 행동 안 함 | **2** | 017#1 · 040#0 | model |
| **축8** | **모델: 청취 전 / 잉여 / 비가역 write** | **2** | 057#1 · 079#1 | model |
| **축10** | **컨텍스트 사망(CWE)** — 우리 층이 유발한 낭비가 판정 직전에 죽임 | **1** | 074#0 | our_layer |
| | **합** | **27** | | |

**귀속 집계**: `our_layer` 1차 = **19 sim**(+074#1 혼합 1) · `model` 1차 = **7 sim** ·
`env` 1차 **0** · `user_sim` 1차 **0**.
(원 보고서들의 `cause_primary` 진술을 그대로 옮긴 것이다 — 016×2·033#0·040#1·050#0·055×2·063×2·
072×2·073#0·074#0·085×2·093×2·094×2 = our_layer / 017#1·040#0·057×2·073#1·079×2 = model.)

⚠ **[[21]] 준수 확인**: 27 sim 어디에도 user-sim 이 1차 원인으로 남지 않았다. 057·085·033 의
허위 주장·불일치 신원은 전부 *시나리오 설계된 압박*으로 판정되고 agent 흡수 실패로 환원됐다.
`env` 도 1차 0 — 유일하게 논쟁적인 `Error: Missing required parameters.` 무지목은 "설계 의도 범위"로
판정됐고, 그것을 사고로 만든 것은 우리 `T2_STALE_STRIP`(축3)이다.

### 2.2 축별 대표 축자 근거

| 축 | 대표 sim | 축자(요지) | 계기 |
|---|---|---|---|
| 축2 | **093#0/#1** | 모델이 `components=[{base 4.0, source:"Balances above the tier 2 threshold earn 4.0% APY."}]` 를 넘겼고 우리 도구가 **`-> 2.75`** 를 반환. `2.75 = 2.5+0.25` 이므로 4.0 은 집합에서 **사라졌다**(산술 확정) | `[T2_SG_ISOLATE] fetch-formalize operand 주입 keys=['components']` → `[T2_SCAFFOLD_GET] … -> 2.75` |
| 축2 | **016#0/#1** | `[T2_DIAG] raw='Platinum Rewards Card — an error has occurred throughout the process.' → Platinum Rewards Card` 가 `is_answer=True` 로 결정 재료 2067자에 실림. 손님이 물은 것은 **Silver·IN_PROGRESS·11/13/2025** | `[T2_ACTION_SUB] 발화를 격리에서 지음 (손님 발화 3건 · 값 2067자)` → 커밋 [24] `get_referrals_by_user(user_id="oadeyemi@gmail.com")` |
| 축2 | **085#1** | `[T2_DOCDECIDE] → 'General'` 242자 배달 → 모델 [35] *"Based on the error message, it appears that the dispute should be categorized as \"General.\""* → **11회 전 제출이 `dispute_category:"General"`** → env 11 BLOCKED | `decided_by_docs_text.format(choice='General')` = 242자 바이트 검산 일치 |
| 축5 | **055#0/#1** | `[T2_SEARCH_AGENT] group=savings_accounts … turn=6` — 손님의 savings 요구는 t0 msg **47** · t1 msg **33** 에서 처음 나온다. 축이 요구보다 **41/27 메시지 먼저 소진**됐다 | `t2_gate_patch.py:2954` `_g = next((g for g in _gs if g not in _done), None)` |
| 축5 | **033#0** | msg[008] 이 `notice_text` 를 산문으로 발화 → `[T2_PREKB] deny waived fam=transfer_to_human_agents (notice already announced)` → 두 턴 뒤 일반 이관 통과 → 에피소드 종료. t1 은 notice 가 늦어 브레이크가 살아 **pass** | `t2_prekb_patch.py:291 _notice_done` + `:544-547` |
| 축1 | **050#0** | `[T2_CLAIMPROV] regen` 이 **정확한 gold 이름** `approve_credit_limit_increase_5847` 를 냈는데 `[T2_UNLOCK_PROV] deny unprovenanced name (followup-regen)` → `regen tool_calls=['shell']` → [60] 산문 날조 | halfB UNLOCK_PROV deny 4건 중 **3건이 실재 이름**(75%) |
| 축1 | **085#0** | 같은 턴에 `[T2_DISCOVERY_STEP2] deny name=get_all_user_accounts_by_user_id_3847 (레지스트리 폴백)` → 모델 순종 → `[T2_RESOLVE] deny … reason=operator-fab`(*"was not discovered from any prior search"*) | `_t2_our_names` 의 유일 기록자가 `_read_routine_pin` 뿐 |
| 축1 | **094#0** | `[T2_RESOLVE] operator-scope … (apply_savings_account_credit_6831, submit_interest_discrepancy_report_7294)` deny **×3** → `stop=resolve_cap(정체 3회)` → [30] *"Given the persistent issue with unlocking the tool, I will manually inform the backend team"* | 정책 축자는 **둘 다** 요구(*"first apply a credit … Then submit the discrepancy report"*) |
| 축4 | **072#0** | `render_recovery` 의 `(2) ask the user for: name, user_id, address, email, phone_number, date_of_birth, time_verified` → 모델이 그 **필드 순서를 그대로** 7턴 복창 → `too_many_errors`. `user_id` 는 손님이 [13]에서 없다고 말했고 `time_verified` 는 같은 A2 가 *"do not ask the customer for the time"* 로 금지한 값 | `gate_interpreter.py::render_recovery` `asks = " OR ".join(…satisfiers…)` |
| 축4 | **072#1** | P3 deny 가 *"the checking account's id copied from **the accounts listing**"* — listing 을 **만드는 도구명**은 미명명 → [60] `account_id="Sky Blue"` 날조 | `requires_reads` 형제 선언(`get_interest_correction`)은 계좌목록을 적고 있다 |
| 축7 | **079#0** | 전 대화 `KB_search_bm25` **1회**(msg 4). `account_id="cr89a2b3c4"`(user_id) ×5 → env 5회 거부에도 재검색 0 → MISSING **11 전량** | `get_all_user_accounts_by_user_id_3847` 이 t0 문맥에 **0회** 등장 |
| 축7 | **073#1** | [50] *"there is no specific tool available to directly apply the credits"* — [42] 이후 KB 검색 **0회**·unlock 시도 0회 | t0 은 같은 자리에서 사전지식으로 `apply_checking_account_credit_5829` 를 냈다 |
| 축6 | **040#0** | [22] 에 015 전문(5기준·NOT Eligible 4사유)이 **축자 실재**. 27 메시지 뒤 [49]~[67] 에서 그 금지 목록에 정확히 해당하는 4건을 전부 `eligible_for_provisional_credit: true` | 같은 문서의 **enum 목록은 [58] 반려 후 즉시 정정**([59]) — "정보 부재" 가설 배제 |
| 축6 | **017#1** | deny 가 `get_user_information_by_name/by_email/by_id` 를 이름으로 댔고 주입된 도구 설명이 *"ASK them for their name, email, or user ID first"* 를 전 턴 상주시켰는데, 4턴 연속 **email/user ID 만** 요구 | 동일 문면 5회 중 **4회 회복**(t7328 t0/t1·t7335A·t7336 t0 전부 "full name") |
| 축8 | **079#1** | msg 93 스키마가 배송·디자인·수수료 3개 열린 선택지를 드러냈는데 [94][96][98] 전부 기본값 커밋 → env *"already a pending debit card order"* ×6 → gold `RUSH·35·PREMIUM` **도달 불가** | 정확한 gold 인자로 4회 재시도했으나 전부 blocked |
| 축10 | **074#0** | `[T2_SG_BYREF]` ×4 가 *"supply 'transactions' yourself with those fields filled in"* → 손-전사 5회 ×6.7~7.8KB → **격리 서브가 전부 덮어씀**(`fetch-formalize operand 주입`) = 순수 낭비 **33KB** → CWE | `_byref_require_fields` 가 `_iso_owns` 우회 `try` **밖** |

### 2.3 축 간 반복 서명 (여러 축에 걸친 공통 기전)

1. **`shell grep` 라인 파편 ↔ `KB_search_*` 문서 본문** — 성패와 **4/4 일치**(033) · 같은 분기가 050·085·079·057·072 에 재현. 우리 층은 어느 검색 도구를 쓰라고 **말한 적이 없다**.
2. **`T2_DOCDECIDE` 가 군마다 같은 상수를 답한다** — 런 전체 분포 `Blue Account`×9 · `Bronze Rewards Card`×8 · `Gold Account`×7 · `Sky Blue`×4 · `Business Bronze…`×3 · `General`×2. x343(n=24) 이 이미 잰 *"요구 없으면 `Gold Account` 24/24 오답"* 지문과 축자 일치.
3. **우리 deny 문면의 손님-대면 복창** — 085#0[37]·085#1[63][67]·057#1[48]·016#1[52]. P2(경고문 에코 차단)가 **못 잡는 형태**로 3회 실측.
4. **`T2_SEARCH_AGENT` 축-소진 후 영구 침묵** — 016·040·050·055·057#1(14회)·073·074#1(8회)·079·085 전부. 재무장 `T2_SEARCH_REARM` 은 코드에 있고 **본런 OFF**.
5. **진짜-이름·무출처 부류** — `UNLOCK_PROV`(050)·`DISPATCH_ROLE`(057)·`operator-fab`(085·040)·`GIVE_QUOTE`(040·055). 전부 "이름이 실재하는데 우리가 환각으로 오판"이고, env 레지스트리(`tau2_domain_toolnames.json`·`_agent_discoverable`·`_user_discoverable`)를 **엔진이 이미 읽는데** 그 출처를 안 쓴다.

---

## §3. 수리 7건 실측 성적표 ([[70]] ± 병기 · 15편 ↔ 반쪽 2편 교차 검증)

판정 칸 정의 — **발화**: 이 런에서 트리거가 성립해 문면/게이트가 실제로 나갔나 ·
**발화하고도 못 삼**: 나갔는데 결정을 못 바꿨나 · **기회 0**: 트리거 조건 자체가 안 왔나
([[55]] 死배선과 무효과를 가른다).

| 수리 | 발화 sim | 발화하고도 못 삼 | 기회 자체가 0 | 매출(의도 효과·실측) | 매입(부작용·실측) |
|---|---|---|---|---|---|
| **P1** `CLAIMPROV` kind-폴백 (`5189b510`) | **40/40** (전 태스크 로그에 `kind-index rescued` 라인) | 072#0 · 063×2 · 094#0 · 074#1 | — | **DUP 계열 소멸**: t7328/t7335 에서 050 DUP **3/3** · 073 DUP ×3 · 085 DUP → t7336 해당 DUP **0**. 전 window `unbacked=0` 유지 | ⑴072#0: 강등 후 `regen tool_calls=[]` 로 회복 행동 소거 → 데드락 ⑵063×2: `unbacked=0` 이 **값-날조 3건**(0.20% APY·paper $2.50·완료보고) 통과 ⑶094#0: env 가 거부한 호출을 *"원장에 있다"* 로 구제(`_evs` 결과 무시) → 날조 완결 주장 3턴 무저지 ⑷074#1: `pending` 축이 **유령 도구명** `apply_credits_to_account_1234`(궤적 0회 등장) 생성 → *"도구가 없다"* → escalation |
| **P2** WRITE_SUB 차단 노트 / 경고문 에코 제거 (`a0fcf07e`) | **0** (전용 마커 0건) | — | **40/40** — 094 는 *드롭될 값이 처음부터 user corpus 로 통과*해 에코 경로 **미재현**, 093 은 경고가 1회씩만 나가고 재호출 없음 | **인과 미확인**(시험 미도달). 에코 재발 0 은 사실이나 P2 덕분이라 말할 수 없다 | 인접 구멍 2종 실측: ⑴**요율 인자의 user-주장 그라운딩**(094 `actual_apy=5.0` 통과 — `corpus_roles` 주석이 *"정책 주장=문서만"* 을 이미 선언했는데 SG_GROUND 요율 축에 미적용) ⑵**deny 문면의 손님-대면 복창 3회**(085#0[37]·085#1[63][67]) — P2 가 못 잡는 형태 |
| **P3** comparator `requires_reads`+`grounded_params` (READ-FIRST) | **6** — 074#0 ×4 · 074#1 ×4 · 085#1 ×1 · 050#0 ×1 · 050#1 ×1 · 072#1 ×1 | 072#1(deny 가 listing 도구 미명명) · 055#1(comparator 반환문의 READ-FIRST 문장 발화·무시) | 016·017·033·040·057·063·073·079·093·094 (comparator 미호출) | **이번 수리 최대 양성.** 074: read **0회 → 5~7회**, t7335 의 2단 날조(`@last:` 참조 날조 → 거래행 통짜 날조 → 우리 comparator 가 날조를 판정해 고객 보고까지 세탁)가 **완전 소멸**. 발화한 6 sim **전부 지목된 read 로 전환**. 050: `consecutive_on_time_payments` 를 지어낸 `"24"` → 실측 `"6"` 복사 | 072#1: 회복 유도 실패(listing 이름 부재) → id 날조로 이탈. 074: read 를 사고 **write 를 못 삼**(credit ×4 여전히 MISSING) |
| **P4** FAB fix-naming (`63443a09`) | **2** — 079#1(`FAB_STRIP dropped 3/1 ungrounded write call(s)`) · 085#1(`dropped 1`) | 2/2 — 노트가 **커밋 메시지에 한 건도 안 남았다**(regen 으로 대체) | halfA **전 sim**(`FAB_STRIP` 런 전체 0회) · halfB 016·033·040·050·057·063·074 | **효과 관측 0** | 미관측. ⚠**소비 경로 부재가 진짜 문제**: `_fab_fix_note`(`t2_gate_patch.py:1834`)의 유일 호출처가 `T2_FAB_STRIP` 블록 하나인데, 072 는 `arg_source_reads.account_id = [get_all_user_accounts_by_user_id_3847]` 라는 **정답이 선언에 있는데** FAB_STRIP 이 0회라 도달 0 |
| **P5** 완결-인상 문구 제거 (`63443a09`) | **4** — 073#0(반환문 [58]·[73]·[83] 3회 노출) · 073#1([43]~[45]) · 074#0 · 074#1(msg 38/44/50) | 074×2(rebate 축 검사 0회 — 발화·무시) | 072×2 (comparator **성공 호출 0회** — 사려던 태스크에 문면이 도달조차 못 함) | **매출 0** — 072 의 누락-rebate 보완검사는 이 런에서 **미실현**(§4) | **매입 1건 실측**: **073#0 넷팅 붕괴**(t7335 net 9.50/9.00/1.50 정확 → t7336 라인별 3/5/1.5·3/3/3 분할 6건). 부수: 자매 도구 `get_checking_atm_fee_totals` 의 완결 문면은 **미수정 잔존**(057); `T2_STALE_STRIP` 이 *"이미 완료한 조회/작업은 반복하지 않았습니다"* 를 8회 발화해 **완결 인상을 다른 경로로 생산**(085#1) |
| **F8** `T2_ARG_PRODUCERS` 에러-형상 게이트 | **0/40** | — | 대부분. 단 **040#1 [84]/[86] 은 트리거 술어가 전부 성립**했는데 `_seen_tools`(이름-등장) 억제로 침묵 | **오발화 제거 성공**: t7335 085 의 KB-본문 오발화 ×2 → **0**. 017 에서 우리 deny 가 `NOT_VERIFIED —` 접두라 설계대로 안 걸림(무해 확인) | **정당 발화도 전멸**: t7328 halfB 7회 · t7335 halfB 5회(그중 040 에서 각 4·2) → **t7336 전 런 0회**. 040#1 [87] 오도구 전환(`get_debit_cards_by_account_id_7823`)은 F8 이 막으라고 만든 바로 그 행동. 같은 표적(`get_card_last_4_digits`)의 피해가 **`T2_VALUE_ACQUIRE` 로 우회 재입장**(079#0 3회·085#1 6회·072#1 6회) — 수리가 한 레버에만 들어갔다 |
| **C587** `requires_reads` 선언 | **9** — 050×2 · 074×2 · 085#1 · 063#1 · 055×2(post-check) · 072#1 | 055×2 — `require_before` 가 replay 정합 때문에 **post-check** 로 약화(`t2_prekb_patch.py:477-497`) → 개설을 **막지 못하고 사후 통지만** | 016·017·033·040·057·073·079·093·094 | 발화한 자리는 **순종률 100%**. 063#1 [036] `get_all_user_accounts_by_user_id_3847` 실행을 실제로 만들었다 | **선언 커버리지 결손 3종**: ⑴`file_credit_card_transaction_dispute` 항목 **0건**(debit 형제만 있음 — 같은 커밋의 누락) → 040 死 ⑵`get_correct_savings_apy.requires_reads = None` → 094 의 결정점을 정확히 겨눈 유일 수리가 **선언 부재로 死** ⑶`get_atm_fee_discrepancies`/`apply_checking_account_credit` 의 `requires_reads` 에 **계좌목록 read 누락** → 072#1 의 deny 가 이름 없는 산문이 됨 |

### 3.1 한 줄 요약

> **7건 중 reward 를 산 것 0건.** 확인된 매출은 두 가지뿐 — **read 축의 GIGO 차단**(P3/C587, 6 sim
> 전부 지목 read 로 전환·074 의 2단 날조 소멸)과 **DUP 계열 소멸**(P1). 확인된 매입은
> **073 넷팅 붕괴**(P5)와 **F8 정당 발화 전멸**(040#1 오억제 1건 실측) 둘. P2·P4 는 **시험 미도달**
> (P2 는 경로 미재현, P4 는 유일 소비 경로 FAB_STRIP 이 표적 sim 에서 0회).

### 3.2 반쪽 보고서의 부분 실측 ↔ 15편 교차 검증 결과

| 반쪽 보고서 주장 | 15편 교차 | 판정 |
|---|---|---|
| halfB §0 *"P3 READ-FIRST 양성 **5 sim**(050×2·074×2·085 t1)"* | halfA 072#1 에서 **1 sim 추가**(런 전체 유일 발화라 기록됨) | **6 sim 으로 정정** |
| halfB §0 *"P1 DUP 재발 **0/20**"* | 057#1 `log_verification` DUP 1 · 079#0 `log_verification` DUP 1 · 033#0 `transfer_to_human_agents` 중복 1(ACTION 축 무해) | **정정**: P1 이 겨눈 **승인·credit 계열 DUP** 은 0. 잔존 DUP 3건은 *모델의 재검증 반복*이라는 **다른 기전** |
| halfB §0 *"F8 오발화 **0/20**"* | halfA 도 0 (017·055·072·073·093·094 전부 0회) · **040#1 오억제 1건** 신규 | **확정 + 매입 1건 추가** |
| halfA §4 *"P5 매출 **미실현**(072 audit 미도달) · 매입 **073#0 넷팅 붕괴 실측**"* | 073 전용 보고서가 3세대 문면 diff·t7335 대조로 **재확인** · 074 가 매출 미실현을 2차 확인(문면 배달됨·rebate 검사 0) | **확정** |
| halfA §4 *"P2 인과 **미확인**"* | 085 이 *"P2 × (실패)"* 로 강화 — 모델이 우리 deny 문면을 손님에게 **3회 복창** | **확정 + 못 잡는 형태 실측** |
| halfA §4 *"P1 … 072#0 pending 루프 잔존"* | 063×2·094#0·074#1 에서 **세 가지 다른 매입**을 추가 관측 | **확대** |

---

## §4. 073 회귀 전용 절 — 무엇을 팔았나 ([[70]] 의무)

### 4.1 판정: **P5 가 073 의 NET 집계 지시를 팔았다 — 확정**

15편의 근거는 halfA 반쪽 보고서의 판정을 **반박하지 않고 보강**한다.

**3세대 문면 diff (같은 도구 `get_atm_fee_discrepancies.return_template` 꼬리·축자)**

| 런 | 문면 꼬리 | 073 결과 | 변이 |
|---|---|---|---|
| t7328 | *"…ONE fee_refund credit for the net correction **across all identified fee discrepancies** of THIS account **= $9.50** (do not credit the same lines twice)."* | **1/2** (t0 = 1.0) | clean |
| t7335 | *"…net correction **across all identified fee discrepancies** of THIS account (do not credit…)"* | 0/1 | **DUP ×3** — 1차 발행 `9.5/9.0/1.5` = **gold 정확** |
| **t7336** | *"**SCOPE OF THIS CHECK - fee-line amounts only:** … This tool did NOT check whether any rebate is missing … ONE fee_refund credit for the net correction **of THIS account** (do not credit…)"* | **0/2** | **WRONGARG 6**(라인별 분할) / MISSING 3 |

**두 번의 매도를 분해한다.**

- **1차 매도(t7328→t7335): `= ${delta_total:.2f}` 제거 = 매도가 아니라 무효 통과의 정정.**
  `_note_delta_total_removed_2026_08_19` 축자: *"delta_total 은 **채점되는 인자 그 자체**다
  (task_073 gold amount 9.50/9.00/1.50 = 계좌별 net). 엔진이 채점되는 값을 만들어 건네면 …
  그 위조판을 재게 된다([[62]]·[[03b]])."* t7328 t0 의 `reward 1.0` 은 **엔진이 gold 인자를 직접
  공급한 통과**였다. ⇒ "1/2 → 0/2 회귀"는 **절반이 수치 정정**이다.
- **2차 매도(t7335→t7336): P5 가 정책 축자 수식어를 함께 지웠다 — 이것이 이번 실패의 주원인.**
  수리 스크립트 `_upd_fee_scope_wording.py` 는 목적을 *"완결 인상 제거(\"across all identified …\" 삭제)"*
  라고 적었으나, 같은 계열 스크립트 `_upd_fee_tool_delta.py` 는 그 구절의 출처를 이렇게 인용한다:
  `doc_bank_accounts_bank_accounts_(general)_017 §2` 축자
  *"apply a credit for the net correction **across all identified fee discrepancies**"*.
  ⇒ 삭제된 것은 "완결 인상"이기 **이전에 `net correction` 의 집계 범위를 지정하는 유일한 수식어**였다.

**실측 대조 (같은 모델·같은 태스크·같은 discrepancy 판정)**

- t7335(수식어 **있음**): `{_1, 9.5}`·`{_2, 9}`·`{_3, 1.5}` — 계좌당 1건 NET 을 **정확히 발행**
- t7336(수식어 **없음**): [68] 한 턴 3발(3/5/1.5) · [76][78][80] 3턴 3발(3/3/3) · [84] 1발(1.5)
  — **라인당 1건으로 쪼갬**. `account_id`·`credit_type` 는 6건 **전부 gold 일치**, 어긋난 필드는
  **`amount` 하나**이고 합계는 정확하다(3+5+1.5 = 9.50 · 3+3+3 = 9.00).

### 4.2 15편 근거로 확정 · 반박 시도 결과

| 반박 가설 | 15편의 증거 | 판정 |
|---|---|---|
| "모델이 산술을 못 했다" | [88] 축자: *"Blue … $3.00 + $5.00 + $1.50 = **$9.50**"* — **gold 3값을 정확히 산출** | **반박 실패**(산술 결손 아님) |
| "탐색이 모자랐다" | 발견·검증·3계좌 audit 전부 정상. 엔진 판정이 gold 합계와 일치 | **반박 실패** |
| "user-sim 이 라인별을 요구했다" | [60] 의 건별 프레이밍 출처는 **[59] 에이전트 자신의 3줄 나열**이다. 손님 페르소나는 *"Honestly, I'm not totally sure what's wrong"* — gold 도 라인 id 도 모른다 | **반박 실패**([[21]] 대로 agent 측 환원) |
| "문면과 무관한 분산" | t7335 는 같은 모델·같은 도구·같은 판정에서 net 3/3 정확. 실패 축이 문면 개정의 **유일한 의미 손실**과 정확히 겹친다 | **[M]**(통제 A/B 아님·t7335 는 1 sim ↔ t7336 2 sim) |
| "P5 가 072 를 샀으니 상쇄" | 072 양 trial 이 comparator **성공 호출 0회** — 문면이 도달조차 못 했다. 074×2 는 문면을 받고도 rebate 축 검사 0 | **매출 미실현 확정**(2 태스크 4 sim 에서 0) |

**부수 효과 2건도 같은 방향**: ⑴ NET 지시가 새로 삽입된 2문장 SCOPE 경고문 **뒤 꼬리**로 밀렸다
⑵ 073 `_3`(Light Green) 이 matched 인 것은 **두 오류의 우연한 상쇄**다 — 도구가 3항목 중 (2)만
리포트(`[coverage] 3 of 6 rows were checked`)해 (1) 미검출과 (3) 미차감이 상쇄됐다.
**NET 계산이 성공한 증거가 아니다.**

### 4.3 [[70]] 판정 3종

1. **전체 reward 짝 A/B**: t7335 073 = 0/1 · t7336 073 = 0/2. 합으로는 둘 다 실패지만 **실패
   지점이 상류로 앞당겨졌다**(재적용 단계 → 최초 발행 단계).
2. **태스크별 부호표**: P5 — **072 = 0**(사려던 이득 미실현·comparator 미도달) ·
   **073 = −1**(확인된 손실) · **074 = 0**(발화·무시) · **057 = 0**(자매 도구 문면 미수정).
3. **무엇을 팔았나**: 072 의 누락-rebate 보완검사를 사려고, 073 이 의존하던 **정책 축자 집계-범위
   수식어**를 팔았다. 두 축은 **분해 가능**하다(SCOPE 경고문 유지 + 수식어 복원 = 둘 다 취함)
   ⇒ [[70]] ③ 분해로 절충 가능했던 **불필요한 매도**. `= ${delta_total}` 은 복원하지 않는다([[23]]).

---

## §5. 우리-층 결손 명부

**칸 정의**
- **반증 판정** = *"이 결손이 실재하는가"* — `CONFIRMED`(코드 경로·로그·축자로 확정) /
  `UNPROVEN`(주장은 있으나 미확정·**미반증**) / `REFUTED`(반증됨).
  **원 보고서에 반증 판정이 없으면 `UNPROVEN`(미반증)으로 표시**했다.
- **귀속** = 손실 인과 등급([S] 결정론 재현 / [M] 1대1 대조 / [?] 미확정).
- **기대 상한** = *이 결손 하나만 고치고 나머지가 그대로일 때 reward 가 뒤집힐 수 있는 sim 수의 상한*.
  같은 sim 에 다른 잔여 차단막이 문서화돼 있으면 **0**, 앞 항목에 이미 계상됐으면 **0(중복)**.

### 5.1 G1 — 우리 층 자기차단 (우리/모델이 낸 정답을 같은 층이 막음)

| id | 결손 | 태스크·sim | 코드 경로 / 선언 키 | 반증 판정 | 귀속 | 기대 상한 |
|---|---|---|---|---|---|---|
| **OL-01** | `T2_UNLOCK_PROV` 출처 집합이 env 레지스트리를 안 본다 — regen 이 낸 **gold 이름** `approve_credit_limit_increase_5847` 를 *"unprovenanced"* deny → `shell` 후퇴 → 산문 날조 | **050#0** (·085#1 1회) | `t2_gate_patch.py:10416-10437`(`_ours2`/`_ctx2`) · `_t2_our_names` 의 유일 기록자 `:2691-2693` | **CONFIRMED** (halfB UNLOCK_PROV deny 4건 중 **3건이 실재 이름**·t1 에서 동일 이름 unlock 성공) | [S] | **1** |
| **OL-02** | `T2_RESOLVE operator-fab` 이 **같은 층 `DISCOVERY_STEP2` 가 방금 지목한 이름**을 "발명"으로 차단 (STEP2 는 `_t2_our_names` 에 등재하지 않고 사이드카로만 나가 `stated_names` 에도 없음) | **085#0** · 085#1 | `t2_resolve.py:175`(후보 `:157-172`) ↔ `:496-503` | **CONFIRMED** (한 턴 안 축자 2회) | [S] | 0 (명목 1 · t1 대조가 잔여 차단막 노출) |
| **OL-03** | `operator-scope` 가 **둘 다 필요한 절차**를 택일로 취급 → gold `apply_savings_account_credit_6831` unlock 을 **3회** 차단 → `resolve_cap(정체 3회)` | **094#0** | `t2_resolve.py:220-226` · 회피로 `:185 declared_required`(두 도구 A2 미선언으로 무력) | **CONFIRMED** (인쇄 순서 `(chosen, want)` 확정 · 정책 축자는 둘 다 요구) | [S] | 0 (잔여: expected 5.5/actual 5.0/amount 40) |
| **OL-04** | `T2_GIVE_QUOTE` 술어(*에이전트 본문*이 손님 발화와 4토큰 축자 공유)가 불성립해 **gold give 를 철회** | **040#1**(gold 040_3) · 055#0(retract 1·결과 손실 0) | `t2_gate_patch.py:11626-11655` / `:11428-11457` | **CONFIRMED** (사전등록 지표: t7336 8발화 중 4철회 · `get_card_last_4_digits` 3발화 중 **2철회 둘 다 gold-필수**) | [S] | 0 (잔여: 040#0 이 보인 `eligible` ×6) |
| **OL-05** | `feedback_user_tool_is_agents` 가 **거짓 부존재 단언**(*"there is no customer-side tool by that name on file"*) — `_tok_overlap` 이 토큰 1개(`deposit`)만 겹쳐도 최대-겹침 항목 반환. 엔진은 **손님-측 레지스트리를 한 번도 조회하지 않는다** | **055#0**(지연) · **055#1**(치명·give 재시도 0) | `t2_gate_patch.py:127`(`_tok_overlap`) · `:9130`(`_reg8=_agent_discoverable`) · `:9160-9165` · 미사용 `_user_discoverable` `:4114` | **CONFIRMED** ([[25]] 위반 — `deposit_check_3847` 실재) | [S] | 0 (잔여: savings 클래스 오선택) |
| **OL-06** | 072#0 **상호 데드락**: `WRITE_ARG_GROUND deny tool=log_verification inner=`(원장 실재 상태) + `CLAIMPROV regen tool_calls=[]` + `GB1` 이 log_verification 을 전제로 계좌 접근 차단 | **072#0** | `T2_WRITE_ARG_GROUND` / `T2_CLAIMPROV` / `GB1_VERIFY_BEFORE_ACCOUNT_ACCESS` | **CONFIRMED**(관측) · **deny 사유는 UNPROVEN**(`inner=` 공란 = 계기 결함) | [M] | 0 (동 sim 은 OL-22 가 1차) |

### 5.2 G2 — 우리 층 산출물이 결정점을 오염 (엔진/서브 산출이 인자·결론이 됨)

| id | 결손 | 태스크·sim | 코드 경로 / 선언 키 | 반증 판정 | 귀속 | 기대 상한 |
|---|---|---|---|---|---|---|
| **OL-07** | 서브 산출이 모델의 **grounded operand 를 무조건 덮어쓴다**(병합·대조·경고 0) — base 4.0 파기 → `-> 2.75` | **093#0 · 093#1** | `t2_scaffold_get.py:2014-2016` `_ctx.update(_sub)` | **CONFIRMED** (2.75 = 2.5+0.25 산술 확정 · `reducers.base=sum` 이면 6.75 여야 함) | [S] | **2** |
| **OL-08** | 관문1 인용 대조가 **전체 문자열 substring** — 인용의 한 단어 차이로 **맞는 성분을 통째로 반려**(`relationship=0.025` "source not found") | 093#0 · 093#1 | `t2_scaffold_get.py:394-402` (`src_ok`·`_norm_ground`) | **CONFIRMED** (x456 격리: 서브가 0.025 를 찾아냄·값·문장 모두 맞음) | [S] 격리 / [M] 라이브 | 0 (중복·OL-07) |
| **OL-09** | `ref_params` 에 **tier 재료(잔액)** 가 없어 서브가 tier 를 고를 근거가 없다. 선언 자신이 tier 를 요구(*"tenure/tier"*) | 093#0/#1 · 094#0/#1 | `t2_scaffold_get.py:699` `ref_params` | **CONFIRMED**(093·산술로 확정) / 094 는 설계 축 **UNPROVEN** | [M] | 0 (중복) |
| **OL-10** | 반환문이 **반영되지 않은 성분을 반영했다고 선언**(*"base + highest checking boost + highest card bonus + all relationship/tier bonuses: 5.5%"* — 실제 `sub=1 rows`) + `required_groups=["base"]` 라 성분 결손에 abstain·플래그 0 | **094#0 · 094#1** (·093) | `a2 scaffold_get_tools[3].return_template` · `.op.required_groups` | **CONFIRMED** ([[25]] 위반 — 모델 [40] *"5.0 + 1.0 = 5.5"* 산술 불가 사후 정당화가 그 귀결) | [S] | **2** |
| **OL-11** | `get_interest_correction` **부호 게이트 부재** — `expected<actual` 이면 −150 을 내고 반환문이 *"Use this as the credit amount"* 로 지시. env 도구는 *"must be greater than 0"* ⇒ **집행 불가능한 지시** | 093#1 | `t2_scaffold_get.py`·`t2_compute.py` (부호 검사 grep **0건**) | **CONFIRMED** | [S] | 0 (중복·OL-07) |
| **OL-12** | `T2_DIAG` **유일-답 강제**가 오진 답문을 `is_answer=True` 로 결정 재료에 심는다 — 선언 축자 *"**One of these records did not pay out.** Reply with that record's account type exactly as written above"* 인데 016 의 15행에는 미지급이 **다수**(REJECTED 3·ERROR 1·IN_PROGRESS 1) | **016#0 · 016#1** | `t2_gate_patch.py:3879-3896` + `a2/banking_knowledge.settings.json` `diagnose_prompt`/`diagnosed_text` | **CONFIRMED** (**5 sim 재현**: t7328×2·t7335×1·t7336×2 로그에 동일 문자열) | [S] | **2** |
| **OL-13** | `T2_ACTION_SUB` 가 손님-발화 턴을 **tool 이력 0 · 직전 추론 0** 문맥에서 다시 짓고 커밋 | 016#0 · 016#1 | `t2_gate_patch.py:9857-9862`(분기) · `:6199-6250`(문맥 조립) · `:8210-8212`(트리거) | **CONFIRMED** (물증 2: [24] 이메일-as-user_id · [30] `<your_user_id>`/`<referral_details>` 자리표시자) | [S] | 0 (중복·OL-12) |
| **OL-14** | `decide_from_docs` 가 **손님 요구 없이** 후보 이름만 보고 결정(`T2_SUB_REQUIREMENT` OFF → `_reqs` 항상 빈 리스트) → 군마다 같은 상수 오답 → *"A separate check was run … It answers: X."* 권위 문장으로 write 결정점에 재제시 | **055#0 · 055#1** · 063×2 · 085#1 · 057 · 016 · 033 | `t2_gate_patch.py:3155-3172` · `:3266` `decided_by_docs_text` | **CONFIRMED** (x343 n=24 블록편차 0: 문서+후보줄만 → `Gold Account` **24/24 오답** / 요구 축자 주면 `Silver Plus` **24/24 정답** / 무관 요구 0/24 부정통제 통과. 라이브 `DOCDECIDE` 분포가 그 지문과 일치) | [S] 격리 / [M] 라이브 | **2** |
| **OL-15** | **퇴화 축**(`doc_index['bank_accounts_bank_accounts']` 클래스가 `_general_` 하나뿐)의 **비-답**을 *"It answers: General."* 로 write 결정점에 주입 → **11/11 제출이 그 값으로 env 거부** | **085#1** | `a2 policy_ontology.doc_index` → `t2_search._disp_name` → `t2_gate_patch.py:3419` → `:8633-8656`(DECISION_CARRY) → `T2_DECIDE_BEFORE_WRITE` | **CONFIRMED** (242자 **바이트 검산** 일치 · KB 031 enum 9종에 'General' 없음) | [S] | **1** |
| **OL-16** | `check_card_application_fit` 이 **필수 필터 operand 드롭 후 무필터 `eligible` 목록** 반환 — `note` 의 *"not applied (no input given): … min_score"* 가 표제어 `eligible` 에 졌다 | 063#1 | `t2_scaffold_get.py:456-462` + `a2 intent_fields[credit_score]` | **CONFIRMED** (t0 는 점수 선확보로 Silver 정답 — 1대1 대조) | [M] | 0 (잔여: savings Platinum 개설) |

### 5.3 G3 — 우리 층 거짓 발화 ([[25]] "우리 도구는 100% 정답 의무")

| id | 결손 | 태스크·sim | 코드 경로 / 선언 키 | 반증 판정 | 귀속 | 기대 상한 |
|---|---|---|---|---|---|---|
| **OL-17** | `T2_STALE_STRIP` 이 **성공을 `m.error` 로 판정** → env 가 에러를 플래그 없이 content 로만 주므로 **실패한 write 가 `done_w` 에 편입** → 동일-인자 재시도 **8회 제거** | **085#1**(×8) · 079#0(×1) · 073#0(×3) | `t2_gate_patch.py:1691-1692`(`ok_ids`) · `_stale_call_ids` `:1687` | **CONFIRMED** (해당 sim 의 `Error` 접두 tool 메시지 **15건 중 14건이 `error=False`** 실측 · F8 수리와 **동형인데 미적용**) | [S] | **1** |
| **OL-18** | 그 노트가 **한국어 + 거짓**으로 손님에게 나간다 — *"[중복 호출 제거: **이미 완료한** 조회/작업은 반복하지 않았습니다.]"* → user-sim 이 [110] *"we've already handled the first two"* 로 **미완료를 완료로 닫음** | 085#1 · 079#0[51] · 073#0 | `t2_gate_patch.py:9981` (바로 위 `:9959` FAB_STRIP 자리에 *"C125: 유저-대면 문자열은 영어"* 규칙이 축자로 있음) | **CONFIRMED** | [S] | 0 (중복·OL-17) |
| **OL-19** | `T2_UNAVAIL_PROMISE` 가 **unlock 된 도구**와 **유령 이름**을 *"does not exist among the tools available to you"* 로 통보 — `"A with B"` 구절을 못 갈라 통째 대조 / `apply_credits_to_account_1234` 는 궤적 **0회 등장**(우리 서브 산출) | **074#0 · 074#1** (2/2 재현) | `t2_gate_patch.py:3688` `_unavailable_promises` + `a2/base/shared.json feedback_unavailable` | **CONFIRMED**(거짓 문장) · **인과 UNPROVEN**(모델이 댄 이유는 다름) | [M] | 0 (잔여: credit ×4 MISSING) |
| **OL-20** | `T2_EPLAN` L2 가 **바로 앞 출력에 전량 실린** 5레코드를 *"have not read their details yet"* 로 판정 + `detail_reader` 가 **credit 도구**인데 대상은 체킹 `btxn_*` | 085#1(4턴 소모·deny cap 도달) | `t2_eplan_patch.py:905-916` · `:249-260` + `a2 gate.json:91 eplan.detail_reader` | **CONFIRMED** | [S] | 0 |
| **OL-21** | `_evs` 원장 집합이 `tool_calls` **이름만** 모으고 결과를 안 본다 → **env 가 거부한 호출**을 *"원장에 있다"* 로 구제 → `unbacked=0` → 날조 완결 주장 3턴 무저지 | 094#0 | `t2_gate_patch.py:11196-11199` + `:3583` | **CONFIRMED** ([[69]] *"상태를 안 바꿔서 해시에 안 남는 것"* 과 정면 충돌) | [S] | 0 |

### 5.4 G4 — 우리 층 문면의 fix-naming 실패 ([[64]])

| id | 결손 | 태스크·sim | 코드 경로 / 선언 키 | 반증 판정 | 귀속 | 기대 상한 |
|---|---|---|---|---|---|---|
| **OL-22** | `render_recovery` 가 auth 게이트 satisfier **인자 7개 전량**을 *"(2) ask the user for: …"* 로 렌더 — `user_id`(레코드 파생·손님이 [13]에서 없다고 선언) · `time_verified`(도구 파생·같은 A2 가 *"do not ask the customer for the time"* 으로 금지) 포함 | **072#0** | `gate_interpreter.py::render_recovery` `asks = " OR ".join(", ".join(v) for v in gate["satisfiers"].values())` + `GB1_VERIFY_BEFORE_ACCOUNT_ACCESS` | **CONFIRMED** (재요구 7턴이 **필드 순서를 그대로** 복창 · 같은 게이트 `note` 는 *"against the record"* 로 정반대) | [S] | 0 (t1 이 계좌목록 벽을 별도 노출) |
| **OL-23** | READ-FIRST deny 가 *"the checking account's id copied from **the accounts listing**"* — listing 을 **만드는 도구명 미명명** | **072#1** · 085#1 | `a2 relations requires_reads` + P3 문면(`{missing}` 치환) | **CONFIRMED** (형제 선언 `get_interest_correction.requires_reads` 는 계좌목록을 적고 있다) | [S] | **1** |
| **OL-24** | `T2_UNLOCK_PROV` deny 문면이 지시하는 검색어 `{name_words}` = `"approve credit limit increase"` = 모델이 **이미 4회 던져 4회 `No matches found`** 를 받은 문자열. 정작 필요한 해소 read(`cat doc_…_007.md`)는 미지목 | 050#0 | `t2_gate_patch.py:10432-10439` (`_fb2`) | **CONFIRMED** (5번째 grep 을 우리 `[DUPLICATE-READ]` 가 다시 거부 = 닫힌 고리) | [S] | 0 (중복·OL-01) |
| **OL-25** | `operator-fab` deny 가 *"was not discovered from any prior search … Search/list the available tools first"* 만 — 이름은 **이미 KB 출력에 축자 실재**. 정확한 문면(`dispatcher_role_check.agent_runs_user_feedback`)이 선언돼 있는데 역할 판정 재료(give 이벤트) 부재로 못 나감 | 040#1(38 메시지 = 95의 40% 소진) · 085 | `t2_resolve.py:176` `[OPERATOR-PROVENANCE]` | **CONFIRMED** (모델이 그 문구를 [33]/[59]/[91] 에서 그대로 반사) | [S] | 0 (중복·OL-04) |
| **OL-26** | `T2_PRESCRIPTION` 신호 코퍼스에 **`role="tool"` 포함** → KB 편재어(`unauthorized`/`fraud`/`fraudulent`)로 오발화 → 정답 credit 시도를 deny 하고 `file_credit_card_transaction_dispute` 로 오유도. deny 문면 자체도 자기모순(*"only for … **fee reversals** - never for disputing a charge"*) | 073#1 | `t2_gate_patch.py:8764-8765`(`_conv`) + `a2 prescription_redirect[0]` | **CONFIRMED** (양 trial signal 히트 **20건 전부 `role="tool"`** · user 발화 0건. 선언 `_note` 가 같은 함정으로 `dispute` 를 이미 뺀 전례) | [S] | 0 (잔여: 모델의 도구 미발견) |
| **OL-27** | **P5 개정이 정책 축자 합산-범위 수식어**(`across all identified fee discrepancies`)를 삭제 → net 지시의 집계 범위가 문면에서 사라짐 + NET 절이 SCOPE 경고문 **뒤 꼬리**로 밀림 | **073#0** | `a2 scaffold_get_tools[8].return_template` (사본 `gate.json:3347` · `split/core.json:1472`) · 수리 `_upd_fee_scope_wording.py` | **CONFIRMED**(문면 diff) · **인과 [M]**(t7335 1 sim ↔ t7336 2 sim) | [M] | **1** |
| **OL-28** | `intent_fields` 드롭 문면이 ⑴범주를 틀리게 말하고(`credit_score` 는 **손님 속성**인데 *"do not add limits they did not state"*) ⑵**다음 행동을 안 댄다**(물어보라는 말 없음) | 063#1 | `a2 scaffold_get_tools.check_card_application_fit.ground.intent_fields[credit_score]` | **CONFIRMED** ([[64]] 위반 형상 · t0/t1 유일 차이가 "물었는가") | [M] | 0 (중복·OL-16) |

### 5.5 G5 — 전달·게이트 타이밍 (결정점에 있어야 할 것이 없다)

| id | 결손 | 태스크·sim | 코드 경로 / 선언 키 | 반증 판정 | 귀속 | 기대 상한 |
|---|---|---|---|---|---|---|
| **OL-29** | `T2_SEARCH_AGENT` **축-소진 영구 잠금** — 군마다 1회 배달 후 *"요청 축 … 모두 처리됨 — 침묵"*. 재무장 `_rearm_subjects` 는 **구현돼 있는데 `T2_SEARCH_REARM` OFF** | 016×2 · 040×2 · 050#0 · 055×2 · 057#1(14) · 073×2 · 074#1(8) · 079×2 · 085×2 | `t2_gate_patch.py:3125-3128`·`:3163`·`:3378/3413` / 수리 `:3009-3057`·호출부 `:3152-3162` | **CONFIRMED** + **격리 A/B 완료**(`x464`: A `said_750` **0/9** ↔ B **6/9** · 60day 6/9 · spend_min 4/9) | [S] 격리 | 0 (016#0 은 OL-12 에 계상 · **016#1 은 술어 불성립**: 모델 발화에 `Silver` 0회) |
| **OL-30** | 축 소진이 **손님 요구 발화보다 먼저** 일어난다 — 술어에 *"그 군의 요구가 발화됐는가"* 가 없다 | 055×2(`turn=6` ↔ 요구 msg 47/33) · 063×2(savings 결정문이 개설 **5턴 후** 도착) | `t2_gate_patch.py:2954` `_g = next((g for g in _gs if g not in _done), None)` | **CONFIRMED** | [S] | 0 (중복·OL-14) |
| **OL-31** | `T2_PREKB` **notice-면제**가 작업 없이 브레이크를 해제 — 모델이 **도구를 부르기도 전에** `notice_text` 를 산문으로 뱉으면 CHECK-FIRST deny 가 **영구 면제**. 같은 A2 의 `ask`(:53)가 그 발화를 유도한다 | **033#0** | `t2_prekb_patch.py:291 _notice_done` + `:544-547` · `a2 gate.json:50 notice_text`/`:53 ask` | **CONFIRMED** (t1 대조: notice 가 늦어 브레이크 유지 → **pass**. 런 계수 **면제 3 : 집행 1**) | [M] 1대1 | **1** |
| **OL-32** | `T2_ACTION_INDEX`(두 gold 도구를 **축자로 담은** A3 43행)가 `status=="deny"` 분기 ∧ `not _m3` 에 갇혀 미발화 — **검색 재료가 하나라도 있으면 통째로 억제** | 033#0 · 085#0 | `t2_gate_patch.py:8496` · `:8603` | **CONFIRMED** ([M]: **같은 seed 626729** 가 t7335 에선 `1회 표면화 4536자` → reward **1.0**. t7336 도달률 ~27%) | [M] | 0 (중복·OL-31) |
| **OL-33** | 검증 게이트 진입 술어가 *"verify_identity 를 **시도**했는가"* — **검증을 아예 시작하지 않은 궤적**에 영구 침묵 | **063#0**(`log_verification` MISSING 의 단독 원인) | `t2_phase.py:60-73 phase_of` + `t2_gate_patch.py:8302 _off_phase` | **CONFIRMED** (같은 gold 의 t1 은 `T2_PHASE_PRECEDE` **20회** 발화) | [S] | 0 (잔여: `open_bank_account` 시도 0) |
| **OL-34** | `T2_WRITE_ARG_GROUND` 가 `not do_gate and not do_prov …` 에 걸려 **조건부 死배선** — 무해한 *순수-조언* 게이트 하나가 **날조-차단 게이트를 통째로 껐다** | **074#1**(`time_verified='2023-11-14 15:30:00 EST'` 날조) | `t2_gate_patch.py:7409-7414` (소비 `_write_arg_ground_deny` `:1370`) | **CONFIRMED** (로그 703줄에 `WEV`/`WRITE_ARG`/`WRITE-GROUND` **0건** · 같은 턴 `stop=other_lever(gate)`) | [S] | 0 (잔여: credit ×4 MISSING) |
| **OL-35** | `t2_stack.admit` **지문-중복 억제**가 최종 턴의 유일 방어선을 제거 — `tag=claimprov`(seen=11) 억제 → **정확히 탐지된**(`unb_p=1 ['record_update']`) 날조 완결 선언이 그대로 손님에게 / `tag=resolve_write dropped`(seen=18) → unlock 직후 *"이제 호출하라"* 드롭 | 050#0 · 074#1 · 093#0 | `t2_gate_patch.py:10335-10339` + `t2_stack.py:758` | **CONFIRMED** | [S] | 0 |
| **OL-36** | `T2_CP2` **단일 슬롯 덮어쓰기 + 도달 계기 부재** — 촉구 발화 109 중 부착 19~22 · 다른 재료로 **덮임 64** · 사이드카에 문자열 **0건**. `_t2_cp2_said` 공유 가드로 결정점 침묵 122회 · x459 형 결정점 도달 **0/8** | (라이브 A/B) t7297 treat 20 sim | `t2_gate_patch.py:8432-8458`·`:8488-8490`·`:8635-8650`·`:9785-9800` | **CONFIRMED** (C593 §3 행 2·3) | [S] | 미측정 |

### 5.6 G6 — 선언 커버리지 결손 ([[72]] 1회 저작 대상)

| id | 결손 | 태스크·sim | 선언 키 | 반증 판정 | 귀속 | 기대 상한 |
|---|---|---|---|---|---|---|
| **OL-37** | `get_atm_fee_discrepancies` / `apply_checking_account_credit` 의 `requires_reads` 에 **계좌목록 read 누락**(`["get_bank_account_transactions"]` 뿐) | 072#1 | `a2 gate.json relations.declarations` | **CONFIRMED** (형제 `get_interest_correction.requires_reads` 는 **둘 다** 적고 있다) | [S] | 0 (중복·OL-23) |
| **OL-38** | `relations.declarations` 11건 중 dispute 는 `file_debit_card_transaction_dispute` **하나뿐** — `file_credit_card_transaction_dispute` 항목 **0건**(C586 같은 커밋의 누락) | 040#0 · 040#1 | `a2 gate.json relations` (grep 0건) | **CONFIRMED** | [S] | 0 (015 의 5기준 중 4는 read 로 안 닫힘 — x467 측정 대상) |
| **OL-39** | `get_correct_savings_apy.requires_reads = None` — 10개 scaffold_get 중 선언된 것은 `check_cli_eligibility`·`get_atm_fee_discrepancies` **둘뿐** | 094#0 · 094#1 (·093) | `a2 specific.json scaffold_get_tools[3]` | **CONFIRMED** (게이트 코드 `t2_scaffold_get.py:1853` 는 **살아 있는데 선언만 없다** · 정책 축자 출처 존재: `doc_…_043`) | [S] | 0 (중복·OL-10) |
| **OL-40** | `arg_producers` 선언이 **`card_last_4_digits` 1건뿐** → 에이전트-측 생산자·`check_amount` 형상이 구조적으로 새어나감 | 055#0([68] `Missing required parameter: check_amount` 형상 부합) · 057×2 · 072#1 | `a2 specific.json:3953 arg_producers` | **CONFIRMED** | [S] | 0 |
| **OL-41** | `_uacts` 가 A2 **도메인 전역 정적 목록**(wrapper 8개)이라 **런타임 discoverable 손님 도구**를 영원히 배제 → 오지목(016 `submit_referral` 38회 · 057 `submit_transaction` 4회). 보완 배선 `T2_PENDING_DISCOVERED` 는 런 PIN 에 **0** | 016×2 · 057×2 | `t2_gate_patch.py:7756`·`:7766` + `a2 gate.json:183 action_tools` + `run_t7336_repaired_stage1_20260821.sh` | **CONFIRMED** (코드 주석 `:7758-7761` 이 이 실패를 **축자로 예언**) | [S] | 0 |
| **OL-42** | 074 **금액 축 미구축** — comparator 가 rebate 축을 명시 기권(`_note_rebate_field` 보류) + Light Blue `oon/forx = null`(명시적 판정 보류) ⇒ gold 27.00/14.50/4.75/3.70 **도달 경로 없음** | 074#0 · 074#1 | `a2 op.steps.{oon,forx}.cases` · `_note_rebate_field` | **CONFIRMED** (4/4 계좌 gold 불일치 실측: 6.00/2.50/6.50/5.50) | [S] | **2** ⚠ 처방은 **[[23]] 위험** — 정책 문서 grep 선행 |
| **OL-43** | savings/checking **class-fit 결정론 표 부재** — `scaffold_get_tools` 11종 중 `check_card_application_fit` 은 카드 전용, `get_checking_atm_fee_totals` 는 ATM 비용 **한 축**뿐(FX·wallet·인출횟수·compounding·rebate 축이 표 밖) | 055×2 · 057×2 · 063×2 | `a2 scaffold_get_tools` | **CONFIRMED**(부재) | [M] | 미측정 (ⓑ) |

### 5.7 G7 — 계기·비용·미확정

| id | 결손 | 태스크·sim | 코드 경로 | 반증 판정 | 귀속 | 기대 상한 |
|---|---|---|---|---|---|---|
| **OL-44** | `T2_GROUND` 가 `agent_tool_name` 인자를 **고객 이름 `CARLOS RODRIGUEZ`** 로 치환(`'name' in 'cardholder_name'`) → `UNLOCK_NAME deny` 연쇄 | 079#1 **14/14** | `t2_gate_patch.py:2253-2274`·`:6650-6656`·`:127` | **CONFIRMED** ([S] 오프라인 결정론 재현: `_grounded_candidates(...)` → `['CARLOS RODRIGUEZ']`) | [S] | 0 (reward 는 msg 98 에 이미 상실 — CWE 기여) |
| **OL-45** | `T2_VALUE_ACQUIRE` 가 **태스크-무관 producer** 를 민다 — 발화 조건에 *"그 write 가 이 대화의 표적인가"* 술어가 없다(주석 `★C4 철회(2026-08-05)` 가 명시). F8 이 얻은 에러-형상 게이트를 **이 레버는 못 얻었다** | 079#0(3회·cap) · 085#1(6회) · 072#1(6회) | `t2_gate_patch.py:7544-7576`·`:1627-1697` + `a2 value_acquisition[0]` | **CONFIRMED** (079: 신호 문자열이 msg 29 **이전 assistant 본문에 0회** → 이름을 처음 꺼낸 건 모델, 우리는 **증폭**) | [M] | 0 |
| **OL-46** | F8 억제 술어가 **이름-등장**(`_seen_tools`)이라 정당 발화까지 전멸 — 의도는 *"이미 값을 얻음"* 인데 구현은 *"이름이 등장함"* | 040#1([84]/[86] 침묵) | `t2_prekb_patch.py:619` | **CONFIRMED** (t7328 7·t7335 5 → t7336 **0**. 정답 술어 `give_exec_idle`(`t2_gate_patch.py:5182`)이 **같은 코드베이스에 이미 있다**) | [S] | 0 |
| **OL-47** | WEV 의 false-block 회피 분기 `if not present: continue` 가 **키 부재를 skip** → 불완전 write 에 처방 피드백 0 | 040#1(t0 7회 ↔ t1 **0회**) | `t2_gate_patch.py:1209` | **CONFIRMED** | [S] | 0 (중복·OL-46) |
| **OL-48** | `_byref_require_fields` 가 `_iso_owns` 우회 `try` **밖**에 있어 `@last:` 참조가 원리상 성립 못 하는 도구에서 **손-전사를 강요** — 그리고 그 행을 격리 서브가 **전부 덮어쓴다**(순수 낭비 33KB) | 074#0(CWE) · 074#1 | `t2_scaffold_get.py:1525-1536` | **CONFIRMED** | [S] | 0 (중복·OL-42) |
| **OL-49** | `t2_forensic.mutation_diff` 가 **ACTION-basis + discoverable-래퍼** 태스크에서 전 항목 빈칸 — `unlock` 은 `GRANTS`, `call` 의 내부 이름은 `mutates=True` 집합 밖 | 033(양 trial) · 072#0·074#0·079#1(`reward_info` 부재) | `t2_forensic.py:516-521`·`:553-576` | **CONFIRMED**(계기 부채) | [S] | 0 |
| **OL-50** | `T2_DISPATCH_ROLE` pre-exec deny 가 **gold 시도를 궤적에서 지운다** — 같은 궤적의 대조군(같은 오류가 env 까지 가서 복구를 낳음)과 부호 반대 | 057#1 | `t2_gate_patch.py:8825-8846` | **UNPROVEN** (PLAUSIBLE · user 압박 유무가 미통제) | [?] | 미측정 |
| **OL-51** | `T2_ENVELOPE_GUARD` 가 **188,848자 봉투 붕괴 턴에서 미발화** — 술어 3항이 전부 성립하는데 그 턴의 `[T2_ENVGUARD]` 라인이 없다 | 079#0 | `t2_gate_patch.py:10522-10527` | **CONFIRMED**(미발화 사실) · **기전 UNPROVEN** | [?] | 미측정 |
| **OL-52** | `T2_GIVE_EXEC` 오발화 의심 — 정본 순수술어 `give_exec_idle` 을 **저장 궤적의 모든 접두**에 오프라인으로 돌리면 전부 `[]` 인데 라이브는 발화 ⇒ **술어가 아니라 입력 채널 결함** 가설 | 055#0 | `t2_gate_patch.py:10560-10578` · `:5182` | **UNPROVEN** (PLAUSIBLE · 리플레이 없이는 확정 불가) | [?] | 미측정 |
| **OL-53** | `no_record_template_v2` 가 v1 의 조회-키 절(*"Ask the customer for their name, email, or user ID"*)을 **버리고 도구 이름만** 남김. 그 자리를 정책 축자 *"Knowing full name or userID is not enough to verify"* 가 **반대 방향**으로 채운다 | 017#1 | `a2 gate.json:903`(사본 `specific.json:709`·`split/core.json:603`) · 발화 `t2_compute.py:728` | **CONFIRMED**(문면 diff) · **분기 요인 UNPROVEN** (동일 문면 **5회 중 4회 회복**) | [?] | 미측정 |
| **OL-54** | `T2_TRANSFER_LEAVES_STEPS` C16 좁힘(*"이관 자체가 선언된 단계이면 침묵"*)이 **모델이 스스로 잘못 활성화한 절차**에도 참이 되어 남은 단계를 말할 자리를 닫는다 | 093#1 (8회 전부 `silent`) | `t2_gate_patch.py:7303-7312` | **CONFIRMED**(발화 상태) · **인과 UNPROVEN**(t0 는 이 태그 0줄) | [?] | 미측정 |
| **OL-55** | `_BLOCK_NOTE` 가 모델 생성분이 빈 문자열일 때 **본문 전체**가 되어 손님에게 나간다 + 사유 문자열이 `has been c` 로 **중간 절단** | 016#1[52] · 074#1[57] | `t2_gate_patch.py:5155` | **CONFIRMED** (016#1 [53] user-sim 역할 혼동을 유발) | [S] | 0 |
| **OL-56** | 우리 deny 문면이 **손님-대면 산문으로 복창**된다 — P2 가 못 잡는 형태 | 085#0[37] · 085#1[63][67] · 057#1[48] | (P2 경로 `t2_scaffold_get.py:192,229`) | **CONFIRMED**(3회 실측) | [M] | 0 |

### 5.8 명부 집계

| 지표 | 값 |
|---|---|
| 총 항목 | **56** |
| **반증 판정 `CONFIRMED`(결손 실재 확정)** | **54** |
| `UNPROVEN`(미반증·결손 실재 자체가 미확정) | **2** (OL-50 · OL-52) |
| `REFUTED` | **0** |
| 존재는 CONFIRMED 이나 **손실 귀속이 UNPROVEN** | 6 (OL-06 · OL-09(094 축) · OL-19 · OL-51 · OL-53 · OL-54) |
| **기대 상한 합** | **15 sim** = 실패 27 의 **56%** |
| 기대 상한이 걸린 **distinct sim** | 15 — 050#0 · 072#1 · 073#0 · 016#0 · 016#1 · 033#0 · 055#0 · 055#1 · 074#0 · 074#1 · 085#1 · 093#0 · 093#1 · 094#0 · 094#1 |
| 기대 상한 0 인 항목의 이유 | ⑴같은 sim 이 앞 항목에 계상(중복) ⑵같은 sim 에 **다른 잔여 차단막이 문서화**돼 있어 단독 수리로는 안 뒤집힘 |

⚠ **읽는 법**: 15 는 *상한의 합*이지 **기대 이득이 아니다**. 각 항목은 "이것만 고치면 뒤집힐 수
있다"의 상한이고, 같은 sim 에 두 개 이상 걸린 곳(093·094·016·055·074)은 **둘 다 고쳐야** 상한에
닿는다. 그리고 [[70]] 대로 어느 수리도 파는 것이 있으므로 **순증이 15 가 아니다.**

⚠ 27 실패 sim 중 **12 sim 은 기대 상한 0** 이다 — 017#1 · 040#0 · 040#1 · 057#0 · 057#1 · 063#0 ·
063#1 · 072#0 · 073#1 · 079#0 · 079#1 · 085#0. 이 중 model 1차가 7, our_layer 1차가 5(전부 잔여
차단막 동반)다. **여기가 "우리 층을 다 고쳐도 안 사는 자리"** 이고, ⓑ/미측정으로 남는다.

---

## §6. 처방 큐 3분할

**분류 규칙(엄격)**
- **ⓐ 무료 수리 즉시** = 결손이 `CONFIRMED` 이고, **술어가 닫혀 있고**([[22]]), *"고친 쪽이 옳다"* 가
  **격리 없이 결정된다**(거짓 발화 제거 · 死배선 복구 · 선언 누락 보정 · 형상 판정 버그). 그래도
  [[70]] 계측 의무(전체 reward 짝 A/B · 태스크별 부호표 · 무엇을 팔았나)는 **면제되지 않는다**.
- **ⓑ 격리 프로브 선행** = 문면/게이트의 **강도·조건·존재 여부가 무엇을 파는지 미측정**이거나,
  [[62]] ① 순서상 결손을 격리로 먼저 재야 하는 것.
- **ⓒ 레버 없음(경계)** = ⛔**A_minimal 정보-맞춘 격리에서 실패를 확인한 것만**([[18]]·[[62]] §1.4).
  라이브 null 은 경계 증거가 **아니다**(C593 교훈). 미측정은 **미측정**으로 남긴다.

### 6.1 ⓐ 무료 수리 즉시 (16건)

| # | 수리 | 대상 결손 | 근거 — 왜 격리가 필요 없나 | [[70]] 계측 의무 |
|---|---|---|---|---|
| **A1** | `_stale_call_ids.ok_ids` 에 에러-형상 게이트(`not error and not content.lstrip().startswith("Error")`) + **노트 문구를 영어로·"이미 완료" 주장 삭제** | OL-17 · OL-18 | env 가 에러를 플래그 없이 content 로 준다는 것이 **실측**(15건 중 14건 `error=False`). F8 이 **같은 술어를 이미 쓴다** — 사본 0. 거짓 발화 제거는 [[25]] 상 무조건 옳다 | 085#1 재시도 8회가 살아나는지 · DUP 증가 |
| **A2** | `_evs` 를 **성공한 호출**로 좁힘(F8 술어 재사용) | OL-21 | [[69]] 축자와 정면 충돌하는 버그. env 거부 호출이 "했다"의 근거가 되는 것은 어느 조건에서도 틀렸다 | CLAIMPROV `unbacked` 계수 · 오탐 복귀 여부 |
| **A3** | `T2_UNAVAIL_PROMISE` 에 **원장-실재 전제**(substring 검산·C45 동형) + `"A with B"`·`"A(B)"` 구절 분할. 실재하지 않으면 **침묵** | OL-19 | 두 건 다 **우리 서브가 만든 문자열**이었고 둘 다 거짓. "모르면 말하지 않는다" | UNAVAIL 발화 수 · 074 [51] 재현 |
| **A4** | `T2_DISCOVERY_STEP2` 가 지목한 이름을 `agent._t2_our_names` 에 **등재**(`t2_resolve.py:496` 직전) | OL-02 | 읽기 루틴은 이미 그렇게 한다(`:2691-2693`). `resolve_operator` 와 `UNLOCK_PROV` 는 **그 집합을 이미 본다 — 등재만 빠져 있다**. 레지스트리 교집합이라 날조 통과는 구조적으로 불가 | operator-fab deny 수 · 환각 통과 |
| **A5** | `T2_UNLOCK_PROV` 출처 집합에 **env 레지스트리**(`_agent_discoverable(env)`·`:2520`) 추가 — 또는 deny 유지·문면만 사실화 | OL-01 | 엔진이 그 레지스트리를 **이미 읽는다**(같은 파일 `:9325` 선례). 오차단율 **3/4** 실측 | ⚠**판다**: 레지스트리 실재하나 엉뚱한 이름의 unlock 통과. `T2_PROV_OURS=1↔0` × `UNLOCK_PROV=1↔0` 4칸 + over-action |
| **A6** | `requires_reads` 선언 3종 보정 — ⑴`get_atm_fee_discrepancies`·`apply_checking_account_credit` 에 `get_all_user_accounts_by_user_id` ⑵`file_credit_card_transaction_dispute` 항목 신설 ⑶`get_correct_savings_apy.requires_reads` | OL-37 · OL-38 · OL-39 · OL-23 | **형제 선언이 이미 그렇게 적혀 있다**(`get_interest_correction`). 게이트 코드는 살아 있고 선언만 없다. 정책 축자 출처 확보(`doc_…_014/_015/_016`·`doc_…_043`) — [[23]] 통과 | ⚠read 강제는 턴을 먹는다 — 전체 reward 짝 + **태스크별 부호표 필수** |
| **A7** | `T2_WRITE_ARG_GROUND`/`rv_specs` 를 `do_gate`·`do_prov` 축과 **분리** | OL-34 | `do_gate` 는 *조언*(순수-advice)이고 WAG 는 *실행 차단* — 성질이 다르다. 조언이 차단을 끄는 것은 **계측되지 않은 상쇄** | Δspurious(과차단) · 074#1 오프라인 재생으로 발화 지점이 turn21 하나인지 |
| **A8** | `get_interest_correction` **부호 게이트** — 결과 < 0 이면 abstain + 이름 있는 지목 | OL-11 | env 도구가 *"must be greater than 0"* 이므로 현행은 **집행 불가능한 지시**. 닫힌 술어(결과<0) | 093#1 재현 · 정당한 환수 케이스 유무 |
| **A9** | F8 억제 술어를 **이름-등장 → 값-가용**으로(`give_exec_idle` 재사용·사본 0) | OL-46 | 주석의 의도가 *"이미 값을 얻음"* 인데 구현이 *"이름이 등장함"* — 의도-구현 불일치 | t7328 7·t7335 5 발화가 회복되는지 **및** t7335 085 KB-본문 오발화가 재발하지 않는지 **둘 다** |
| **A10** | `_byref_require_fields`(+`_byref_map_fields`)를 `_iso_owns` 우회 `try` **안**으로 | OL-48 | `isolate.fetch_formalize` 를 선언한 도구에서 `@last:` 는 **원리상 성립 못 한다** — 우회의 원 의도(C526ⓔ→C531)가 이 자리다. 손-전사 33KB 가 **결과에 1비트도 기여 안 함** 실측 | 본문 크기 · CWE 재발 · 서브 실패 시 over-str 검사 유지 |
| **A11** | `_grounded_candidates` 를 **operator 인자에서 분리** — `agent_tool_name`·`discoverable_tool_name` 은 후보를 레지스트리로 닫는다 | OL-44 | [S] 오프라인 결정론 재현으로 기전 확정. 도구명 인자에 사람 이름이 들어가는 것은 어느 조건에서도 틀렸다 | 치환이 정답이었던 사례 **0건 확인**(부정통제) |
| **A12** | `T2_EPLAN` L2 에 *"같은 목록 출력에 상세가 실려 있으면 제외"* + `detail_reader` 를 **목록**으로(지목 대신 범위 표면화) | OL-20 | `list_from_reads:true` 도메인에서 목록 도구가 전 필드를 반환하면 *"listed but not read"* 는 **항상 거짓** | EPLAN L2 발화 수 · 085#1 4턴 회복 |
| **A13** | `feedback_user_tool_is_agents` 의 **부정 존재 단언 삭제** + `_user_discoverable(env)` 선조회(fail-open) + `feedback_registry_listing` **병기**(선점 금지) | OL-05 | 우리가 확인하지 않은 사실을 단언하고 있다([[25]]). `_user_discoverable` 는 **같은 파일에 실재**(`:4114`)하고 주석이 그 구분을 이미 적어 뒀다 | give 성사율 · 오-give 증가 |
| **A14** | **퇴화 축**(`doc_index[group]` 클래스 집합이 `{"_general_"}`)에서 `decide_from_docs` 결정문 **미배달** | OL-15 | *"고를 것이 없다"* 는 **선언에서 기계 도출**되는 사실이고 도메인 판단 0. *"It answers: X."* 는 고를 것이 실재할 때만 참 | 085#1 `dispute_category` 값 · DOCDECIDE 배달 수 |
| **A15** | `_BLOCK_NOTE` 를 **본문 전체로 커밋 금지**(모델 생성분이 빈 문자열이면 재생성) + 사유 절단 수정 | OL-55 | 기계 노트가 손님 발화가 되는 것은 어느 조건에서도 틀렸다. 016#1 [53] 역할 혼동이 그 귀결 | 빈-본문 턴 수 |
| **A16** | `t2_forensic` 정본에 **`action_diff`** 추가(=`action_checks` 기반 MISSING/MATCH) | OL-49 | **계기 부채**. 지금은 ACTION-basis 태스크마다 손으로 표를 만들게 돼 있다([[67]]: 사본 금지·정본에 넣는다) | 없음(측정 도구) |

### 6.2 ⓑ 격리 프로브 선행 필요 (20건)

| # | 항목 | 대상 결손 | 무엇이 미측정인가 | 덮는 프로브 |
|---|---|---|---|---|
| **B1** | `diagnose_prompt` 유일-답 전제 제거 + `T2_ACTION_SUB` 를 대체→합류 | OL-12 · OL-13 | **판다**: 미지급이 유일했던 4 사례(x213 G_ONTO 100%)의 적중 일부 · x228 이 잰 소유권 6/6. 격리 없이 끄면 그 이득이 얼마인지 모른다 | **없음 → x471** |
| **B2** | `T2_SUB_REQUIREMENT` ON + 축-소진 술어에 *"그 군의 요구가 발화됐는가"* | OL-14 · OL-30 · OL-43(부분) | x343 은 **서브 단독** 격리(n=24)다. **라이브 조립**(축 소비 순서·`_reqs` 주입·write 자리 재제시)에서 같은 부호가 나는지 미측정 | **없음 → x472** |
| **B3** | savings-APY 성분 완전성 3종(덮어쓰기 병합·관문1 완화·tier 재료 전달) | OL-07 · OL-08 · OL-09 | 관문1 완화는 **날조 통과율을 올린다** — (진짜 성분 생존율↑) ↔ (가짜 성분 생존율Δ) 를 **짝으로** 재야 한다 | x468(부분·actual 축만) → **x473** |
| **B4** | `get_correct_savings_apy` 반환문이 **실제 반영 성분만** 말하게 + `required_groups` 결손 플래그 | OL-10 | 성분 내역을 돌려주면 모델이 검산하는가 — 미측정 | x468(부분) → **x473** |
| **B5** | ID-해결: 오투입 자리에서 **해소-read 를 이름으로 지목** | OL-23 · OL-37(부분) | *"지목하면 부르는가"* — x466 이 이 질문 | **x466 ✔** |
| **B6** | dispute/referral write 에서 **정의 문서 전문 배달**이 불리언/범주 판정을 바꾸는가 | OL-38 · 040 eligible 축 · 085 `General` 축 · 016 $750 축 | x465 는 **이관 사슬**에서만 실증했다. write 축은 미측정 | **x467 ✔**(016 diagnose 형은 부분) |
| **B7** | `actual_apy` **파생 검산**(formalize↔calc 분담) | 093/094 actual 축 | 모델 단독 유도보다 나은가 — 미측정 | **x468 ✔** |
| **B8** | **P5 문면 3세대 A/B/C** — credit 호출 개수(계좌당 1 ↔ 라인당 1)만 재기 | OL-27 | 073 보고서 자신이 *"P-A 는 격리 프로브 먼저다"* 로 순서를 못 박았다([[62]] ①). 복원 자체는 정책 축자 되돌리기라 저작 0 | **없음 → x474** |
| **B9** | 074 금액 축(rebate 상계 · Light Blue oon/forx) | OL-42 | ⛔**순서 고정**: ①정책 문서에 조항이 있는지 **grep 먼저**(없으면 기권 유지가 옳다) ②`rebate_field` **추출률부터** 격리 측정. gold 보고 A2 를 채우면 [[23]] 위반 | **없음**(프로브 아님 — 문서 grep 선행) |
| **B10** | savings/checking **class-fit 표** | OL-43 | [[62]] ③ — 위 B2·B11 을 먼저 재고 **그 뒤에도 남는 잔여에만** 신설 | **x472 부분 → x472** |
| **B11** | comparator 가 **필수 필터 결손 시 기권**(eligible 목록 대신) | OL-16 · OL-28 | 기권이 사는지 · 표제어 완화만으로 되는지 미측정 | **없음 → x472** |
| **B12** | `T2_PREKB` notice-면제 절충 + `T2_ACTION_INDEX` `not _m3` 억제 해제 | OL-31 · OL-32 | 면제의 원 취지(C210·004 *"동의-터미널 직후 deny 가 마지막 행동 턴을 소각"*)를 **파는지** 미측정. ACTION_INDEX 발화 자리가 늘면 Δspurious | x465 인접(문서 전달) → **x476** |
| **B13** | 검증 게이트 진입 술어 확장(조건부) | OL-33 | `t2_phase.py` 주석이 *"무조건 확장은 통과 런을 죽인다"* 를 축자로 경고 | **없음 → x476** |
| **B14** | 촉구·완결-사칭 채널의 **전달 수리 T1~T5** | OL-36 · OL-35(부분) · OL-01(T5) | C593 §4: *"결정점에서 D_name 을 실물 도구로 주면 방출하는가"* 는 **측정된 적이 없다** | **x470 ✔** |
| **B15** | `_uacts` 를 env 손님-도구 레지스트리와 교집합 + `T2_PENDING_DISCOVERED` ON | OL-41 | 후보가 늘면 `user-action instruct` 오발화도 는다 — 057#0 의 `submit_transaction` ×4 가 현재 기준선 | x466 부분 → **x475**(보조) |
| **B16** | `T2_VALUE_ACQUIRE` 에 F8 과 같은 게이트(①∨②) 또는 cap 3→1 | OL-45 | 2026-08-05 에 ①을 걸었다가 **053 이 죽어 철회**된 이력 — 끄면 무엇을 파는지 재야 한다 | **없음 → x478** |
| **B17** | 잉여·비가역 write 억제(원장 대조 표면화) | 057#1 · 079#1(축8) | [[62]] 경계선 — *"이미 연 계좌가 있다만 알려주면 모델이 스스로 멈추는가"* 를 **먼저 격리로** | **없음 → x477** |
| **B18** | `T2_GIVE_QUOTE` 조건화(`give 대상 ∈ arg_producers.values()` 면 표면화만) | OL-04 | 원 표적(010 의 **여분** give)을 파는지 — 철회율 분모/분자를 producer/비-producer 로 갈라 부호표 | **없음 → x477** |
| **B19** | `no_record_template_v2` 에 v1 조회-키 절 **병합** | OL-53 | 동일 문면 **5회 중 4회 회복** — 문면 축인지 모델 분산인지 미확정. `x379` 에 팔 하나 추가로 끝난다 | **없음 → x477** |
| **B20** | `T2_PRESCRIPTION` `_conv` 에서 `role="tool"` 제외 | OL-26 | ⛔[[57]] 부정통제 필수 — **사기 태스크에서 이 deny 가 실제로 사는지** Δ 계측 없이는 수리 금지 | **없음 → x478** |

### 6.3 ⓒ 레버 없음 = 경계 — **0건**

⛔ **엄격 규칙 적용 결과 이 칸은 비어 있다.** 27 실패 sim 중 *"A_minimal 정보-맞춘 격리에서도
실패한다"* 가 확인된 항목은 **하나도 없다**. 지금까지 돌린 격리는 전부 반대 방향을 가리켰다:

| 격리 | 결과 | 함의 |
|---|---|---|
| `x464`(축-소진 재무장) | A `said_750` **0/9** ↔ B **6/9** | 재료가 닿으면 모델이 쓴다 = **전달 축** |
| `x343`(요구-실린 축 결정) | 문서+후보줄만 → `Gold Account` **24/24 오답** / 요구 축자 → `Silver Plus` **24/24 정답** / 무관 요구 **0/24** | **전달 축**(부정통제 통과) |
| `x456`(093 관문1) | 서브가 `relationship=0.025` 를 **찾아냈다**(값·문장 모두 맞음) — 우리 관문이 반려 | **우리 층 결손** |
| `x465`(이관 문서 전달) | A 7/7 일반 ↔ B 6/7 사슬 | **전달 축** |
| `x459 ⒝`(완료-사칭) | D_name 15/15 — 단 **도구 없는 naming 계기**·실효 n=5 | **미확정**(C593) |

**경계 후보 3종은 전부 "미측정"으로 남긴다** — 라이브 null 을 경계 증거로 쓰지 않는다(C593 교훈):

1. **085 인자 형식 결손**(`pin_compromised` 불리언 ↔ `"no"` · `card_action=null` · duplicate 쌍에서
   "먼저 것" 규칙) — 085 §8-7(b)(c) 가 *"모델 결손이고 레버 없음"* 으로 박제했으나 **격리 미실행**.
   ⇒ **미측정**.
2. **073#1 credit 도구 미발견**(탐색 0회 "없다" 단정) — t0 은 같은 자리에서 사전지식으로 이름을
   냈다. 격리 미실행 ⇒ **미측정**.
3. **079#1 비가역 주문의 선호 미청취** — R5(열린-열거 인자 게이트)가 제안됐을 뿐 격리 미실행
   ⇒ **미측정**(B17 로 큐잉).

⚠ 추가로 **[[68]] 벤치 결함**과 혼동하지 말 것 — 074 의 금액 축(OL-42)은 *벤치 결함*이 아니라
**우리 A2 의 명시적 기권**(`_note_rebate_field`·`cases=null`)이다. 기권이 옳은지 여부는
정책 문서 grep 으로만 결정된다(B9).

---

## §7. 프로브 명부 — 현재 4종의 ⓑ 커버리지

### 7.1 준비된 프로브 4종이 무엇을 재는가 (docstring 축자 요약)

| 프로브 | 축 | 팔 | 결정점 출처 | 채점 |
|---|---|---|---|---|
| **x466** `id_resolution_iso` | ID-해결 read 생략 · **원장-종류 대조 지목** | `A_asis` / `B_pointed`(도구-결과 채널 지목 피드백) / `S_sham`(지목 read 이름만 규칙 치환) / `N_neg` | t7336 079#0[25]·085#0[31]·074#0[4] 궤적 축자 복원 | `pointed_read` · `pointed_read_direct` · `sham_read` · `same_misinput`(닫힌 술어) |
| **x467** `policy_boolean_doc_iso` | 정책 미독 불리언/범주 — **정의 문서 전달**이 판정을 바꾸나 | `A_asis` / `B_docfull`(D1∪D2 전문) / `B_d1`(1홉) / `N_neg` | write 형 = 040#0[49]·085#1[63] · diagnose 형 = 016#0[22] | 닫힌 술어(⛔불리언 정책값 **기계 도출은 열린 판단**이라 gold 채점 금지) |
| **x468** `actual_apy_formula_grid` | 닫힌 산술 **미유도**(`actual_apy`) | `A_asis` / `B_formalize`(서브 형식화 + 엔진 계산) / `C_hint`(A2 param 문장만) / `N_neg` | 093#0[41]·093#1[37]·094#0[20]·094#1[20] (4 결정점) | `ref_match`(원장 계산 참조 ±0.01) · `off_ref`(+`eq_live`) · `no_field` · `ref_unavail` |
| **x470** `claim_demand_live_iso` | 완료-사칭/행동-촉구 **격리-동형 라이브 재생** | `A_asis` / `N_neg` / `L_live` / `D_name` / `D_sys` / `E_early` | `mutation_diff.missing≠∅ ∧ 마지막 text-only assistant 발화 직전` — 로컬 110개 중 050/072/073/074 계열 **44** ≥ 24 | **emission** 채점(`MISS`/`DUP`/`MUT_OTHER`/`GRANT_MISS`/`READ`/`TEXT`) — naming 채점 안 함 |

### 7.2 ⓑ 20항목 × 4프로브 커버리지 표

`●` = 완전히 덮음 · `◐` = 부분(축의 일부만) · `○` = 미덮임.

| ⓑ 항목 | x466 | x467 | x468 | x470 | 커버 | 잔여(무엇이 안 재지나) |
|---|---|---|---|---|---|---|
| B1 DIAG 유일-답 + ACTION_SUB | ○ | ◐ | ○ | ○ | **0.3** | x467 diagnose 형은 **문서 전달**만 잰다. 오진 답문의 `is_answer` 주입·격리 문맥 박탈은 미측정 |
| B2 SUB_REQUIREMENT + 축-소진 술어 | ○ | ○ | ○ | ○ | **0** | x343 은 서브 단독 — 라이브 조립 미측정 |
| B3 savings-APY 성분 3종 | ○ | ○ | ◐ | ○ | **0.3** | x468 은 `actual` 축. `expected`(덮어쓰기·관문1·tier) 축 미측정 |
| B4 반환문 성분 내역 | ○ | ○ | ◐ | ○ | **0.2** | 성분 내역 노출의 검산 효과 미측정 |
| **B5 ID-해결 지목** | **●** | ○ | ○ | ○ | **1.0** | — |
| **B6 정의 문서 전달(불리언/범주)** | ○ | **●** | ○ | ○ | **1.0** | — |
| **B7 actual_apy 검산** | ○ | ○ | **●** | ○ | **1.0** | — |
| B8 P5 문면 3세대 | ○ | ○ | ○ | ○ | **0** | — |
| B9 074 금액 축 | ○ | ○ | ○ | ○ | **0** | 프로브 이전에 **정책 문서 grep** 단계 |
| B10 class-fit 표 | ○ | ○ | ○ | ○ | **0** | — |
| B11 comparator 필터 결손 기권 | ○ | ○ | ○ | ○ | **0** | — |
| B12 PREKB waive + ACTION_INDEX | ○ | ○ | ○ | ○ | **0** | x465 는 문서 전달 축(인접)이나 waive/억제 축 0 |
| B13 검증 게이트 진입 술어 | ○ | ○ | ○ | ○ | **0** | — |
| **B14 촉구·완결-사칭 전달** | ○ | ○ | ○ | **●** | **1.0** | — |
| B15 `_uacts` 교집합 | ◐ | ○ | ○ | ○ | **0.2** | x466 은 **에이전트-측** 도구 지목 채널. 손님-측 후보 집합 미측정 |
| B16 VALUE_ACQUIRE 게이트 | ○ | ○ | ○ | ○ | **0** | — |
| B17 잉여·비가역 write 억제 | ○ | ○ | ○ | ○ | **0** | — |
| B18 GIVE_QUOTE 조건화 | ○ | ○ | ○ | ○ | **0** | — |
| B19 `no_record_template_v2` 병합 | ○ | ○ | ○ | ○ | **0** | `x379` 골격만 존재(팔 4종) — 팔 추가 미실행 |
| B20 PRESCRIPTION role=tool | ○ | ○ | ○ | ○ | **0** | — |
| **합** | | | | | **5.2 / 20** | |

**커버리지 판정**
- **완전 커버 = 4/20 = 20%**
- **부분 가중 합 = 5.2/20 ≈ 26%**
- 기대 상한이 걸린 **15 sim 기준**으로 다시 세면: x466 → 072#1(1) · x467 → (040 은 상한 0) ·
  x468 → 093·094 의 `actual` 축(4 sim 의 **일부**) · x470 → 050#0·072·073·074 계열 결정점.
  ⇒ **상한 15 sim 중 프로브가 결정점을 실제로 시험하는 것은 072#1 하나(완전) + 093/094 4 sim(부분)**
  = **1~5 sim**. 나머지 **10 sim**(016×2·033#0·055×2·073#0·074×2·085#1·050#0 의 문면 축)은
  **프로브가 없다**.

---

## §8. 덮이지 않은 ⓑ 항목의 새 프로브 사양 (설계서 — **구현 금지·사양만**)

**공통 규약**(전부 [[62]] §1.4 정보-맞춘 격리 · [[71]] 계약 · x459 ⒜/x465/x466 관용구 재사용)
- **재생 인터페이스** = **라이브 system 메시지 재구성**(`LLMAgent(tools, domain_policy=sim["policy"])`
  init state — x459/x465 가 빠뜨린 것·x470 이 교정) + **실물 도구 스키마**(env `alltools` 17종 +
  A3 `scaffold_get_tools` → `_build_tool` + `_augment_byref_params`) + **실제 메시지 객체**(절단 0) +
  `la.generate`. ⛔렌더-텍스트 인터페이스 금지(C584: 라이브를 재현 못 한다).
- **팔 규칙**: 한 번에 **한 변수만**. 팔 문면은 마지막 user 메시지 한 줄 또는 마지막 도구-결과
  꼬리(라이브 deny 채널). 도구 **지목 0**·값 열거 0·gold 0.
- **부정통제 의무**([[57]]): 모든 프로브에 `N_neg`(무내용 재촉 한 줄·`x465.NUDGE` import) 필수.
  `S_sham`(내용만 규칙 치환)은 *"내용이 원인"* 을 주장할 때 필수.
- **A_asis 유효성 게이트**([[55]]): `A_asis` 가 **라이브의 다음 행동을 재현**해야 결과를 쓴다.
  재현 실패 = 계기 결함 → 결과 사용 금지. 코드로 검사해 JSON `valid` 에 박는다(산문 아님).
- **n**: 결정점 ≥ 8, 팔당 표집 = det(t=0) 1 + t=0.7 ×2. 사전 고정 판정에서 **차이 <5 는 잡음**(C483 ±4).
- **채점**: 닫힌 술어 한 함수. `t2_forensic.nameof/argsof/inner_name/flat_args/mut_key` 정본만.
  ⛔`reward_info`/gold 로 채점 금지([[23]]·[[69]]).
- ⛔ **[[62]] 자기점검 4문**을 각 사양 머리에 답한다: ①격리로 결손을 재나 ②격리에서 되면
  레버는 **전달뿐**인가 ③결정론 추가 0인가 ④떠먹이기 0인가.

---

### x471 — `diag_actionsub_iso` : 오진 주입과 격리-문맥 박탈 (B1)

**[[62]] 4문**: ①`T2_DIAG` 답문이 **없을 때** 모델이 스스로 최신 건을 짚는가를 잰다 ②짚으면 레버는
**제거(빼기)** 뿐 ③결정론 추가 0(답문을 **안 내는** 것이 처방) ④떠먹이기 0(어느 레코드인지 우리가 안 고른다).

- **결정점 출처 sim**: 016#0[22] · 016#1[24]·[30]·[36] (`T2_DIAG` 가 돈 턴 직전) — 로그
  `[T2_DIAG] raw=…` 가 있는 턴을 닫힌 술어로 추출. 같은 서명이 t7328×2·t7335×1 에도 있어
  **결정점 5개 이상 확보 가능**(같은 태스크 계열이므로 x467 diagnose 형 컷과 공유).
- **팔**
  | 팔 | 변수 |
  |---|---|
  | `A_asis` | 현행 그대로 — `diagnosed_text`(2067자·`is_answer=True`) **포함**, 생성은 `_gen_action_sub` 문맥(손님 발화 + 값 + 소유권 표) |
  | `B_nodiag` | 같은 자리에서 **`diagnosed_text` 만 제거**(나머지 재료 동일) |
  | `C_full` | `diagnosed_text` 제거 **+ 본-문맥 생성**(tool 이력 포함 — `ACTION_SUB` 를 합류로) |
  | `D_enum` | `diagnosed_text` 를 **답 대신 상태별 열거**로 치환(*"records with a non-paid status: {status: [types]}"* · 선언 `policy_ontology.rows` 기계 도출·순위 0) |
  | `N_neg` | `diagnosed_text` 자리에 무내용 한 줄 |
- **닫힌 채점 술어**
  - `named_target` — 다음 발화/호출에 **손님이 물은 계열의 표시명**이 축자 등장(표시명 집합은 A3
    `doc_index` 슬러그 기계 전개 · 어느 것이 정답인지는 **채점기가 모른다** → 대신 아래 2종으로 가른다)
  - `named_wrong` — `T2_DIAG` 가 낸 답문의 주어가 등장 · `named_multi` — 둘 이상 등장 · `named_none`
  - `asked_friend_id` / `read_friend_txn` — 후속 행동 분류(닫힌: tool 이름·인자 키 존재)
  - `placeholder` — 생성물에 `<…>` 자리표시자 또는 **원장에 없는 인자값**(닫힌 substring 검산)
- **사전 판정 규칙**
  - `A_asis` 의 `named_wrong` < 5/8 → **계기 결함**(라이브 재현 실패) → 결과 사용 금지
  - `B_nodiag.named_wrong` − `A_asis.named_wrong` ≤ −5 → **답문 제거가 오진을 막는다**(A14 형 처방 자격)
  - `C_full.placeholder` = 0 ∧ `A_asis.placeholder` ≥ 5 → **문맥 박탈이 물증의 원인**(합류 처방 자격)
  - `D_enum ≈ B_nodiag` → 열거로 대체해도 손해 없음(= x213 4 사례의 적중을 덜 판다)
  - 전 팔 차이 <5 → 문면 축 무효 → 잔여는 [[13]]
- **무엇을 파는가**: `diagnosed_text` 제거는 **미지급이 유일했던 4 사례의 100% 적중**(x213 G_ONTO
  24셀) 일부를 판다. ⇒ **x213 4 사례를 같은 팔로 재측정하는 블록을 반드시 함께 돌린다**(부호표).

---

### x472 — `requirement_axis_iso` : 요구가 실린 축 결정 (B2 · B10 · B11)

**[[62]] 4문**: ①손님 요구를 축자로 실으면 클래스 선택이 맞는가를 **라이브 조립**에서 잰다
②맞으면 레버는 **전달(요구를 서브에 싣기)** 뿐 ③결정론 추가 0 ④클래스를 우리가 안 고른다.

- **결정점 출처 sim**: 055#0[44]·055#1[30]/[46] · 063#0[013]/[035] · 063#1[042] · 057#0[8] ·
  057#1[38]/[70] — 닫힌 술어 = **`open_bank_account` 계열 write 직전** ∧ 그 sim 에 `[T2_DOCDECIDE]`
  라인이 있는 턴. 8 결정점 확보.
- **팔**
  | 팔 | 변수 |
  |---|---|
  | `A_asis` | 현행 — `_reqs` 빈 리스트(`T2_SUB_REQUIREMENT=0`) 상태의 `decide_from_docs` 결정문 배달 |
  | `B_req` | **손님 요구 메시지를 축자로** 서브 입력에 실음(x343 B 조건의 라이브 조립판) |
  | `C_late` | `B_req` + **축 소진 술어에 요구-발화 조건**(요구 인용 0인 군은 소진하지 않고 다음 결정점으로 미룸) |
  | `D_fit` | `B_req` + **class-fit 표**(클래스 × 문서화 스펙 축) 배달 — B10 의 잔여 측정 |
  | `E_abstain` | comparator 가 **필수 필터 결손 시 기권**(eligible 목록 대신 *"cannot determine … ask the customer"*) — B11 |
  | `N_neg` | 같은 자리 무내용 한 줄 |
- **닫힌 채점 술어** (다음 write 호출의 인자)
  - `class_in_delivered` — `account_class` 값이 **배달된 문서 집합의 클래스**에 속하는가
  - `class_changed_vs_live` — 라이브가 쓴 값과 다른가
  - `asked_before_write` — write 대신 **질문**을 냈는가(tool_calls==[] ∧ 물음표 종결 — 닫힌 형상)
  - `write_emitted` / `no_tool`
  - ⛔ *"정답 클래스를 맞혔는가"* 는 **채점하지 않는다**(gold 경유 금지). 대신 `class_changed_vs_live`
    와 `asked_before_write` 로 방향만 본다. **정답 일치는 본런 reward 에서만**([[69]]).
- **사전 판정 규칙**
  - `A_asis.class_changed_vs_live` ≥ 5/8 → 계기 결함(라이브 재현 실패)
  - `B_req.class_changed_vs_live` − `N_neg` ≥ 5 → **요구 전달이 산다** ⇒ B2 승격
  - `C_late − B_req` ≥ 5 → 타이밍 축 추가 필요 ⇒ OL-30 처방 자격
  - `D_fit − B_req` ≥ 5 → **class-fit 표 신설 정당**([[62]] ③ — 그 전에는 신설 금지)
  - `E_abstain.asked_before_write` − `A_asis` ≥ 5 → B11 승격
- **무엇을 파는가**: `C_late` 는 축 배달 지연 → 다른 태스크의 조기 재료 배달을 판다.
  `D_fit` 은 배달 문자수 ↑ → **컨텍스트(OL-48/CWE 축)와 상쇄 계측 필수**.

---

### x473 — `savings_apy_component_iso` : 성분 완전성 (B3 · B4)

**[[62]] 4문**: ①모델의 grounded 성분이 살아남으면 `expected` 가 맞는가 ②맞으면 레버는
**병합·표면화**(빼기) 뿐 ③엔진이 성분을 **고르지 않는다**(충돌 표면화만) ④떠먹이기 0.

- **결정점 출처 sim**: 093#0[39]·093#1[35]·094#0[14]·094#1[14] (=`get_correct_savings_apy` 호출 턴)
  + 그 **반환 직후 턴**(반환문을 받은 다음 발화) = 결정점 8개.
- **팔**
  | 팔 | 변수 |
  |---|---|
  | `A_asis` | 현행 — 서브 산출이 모델 operand 를 **덮어씀**, 반환문은 네 성분 반영 선언 |
  | `B_merge` | 덮어쓰기 대신 **병합 + 충돌 표면화**(같은 `kind` 가 다른 값이면 둘 다 이름으로 인쇄 · 고르지 않음) |
  | `C_gate1` | 관문1 대조를 **전체 인용 substring → 값을 포함한 최소 span**(값 주변 N자 윈도)으로 완화 |
  | `D_manifest` | 반환문에 **반영된 kind 목록 + 결손 kind 이름**(집합 차 인쇄 · 도메인 판단 0) |
  | `E_ref` | `ref_params` 에 **잔액(tier 재료)** 추가 |
  | `N_neg` | 무내용 한 줄 |
  | **`S_fake`** | ⚠**필수 부정통제** — `C_gate1` 팔에 **날조 성분 1개**(값이 문서에 없는 인용)를 섞어 통과율을 잰다 |
- **닫힌 채점 술어**
  - `n_components_kept` — 서브/모델 성분 합집합 중 관문1 통과 수
  - `expected_delta` — 반환 `result` 가 `A_asis` 대비 **변했는가**(값 자체는 gold 대조 안 함)
  - `recompute` — 반환 직후 턴에서 **같은 도구를 재호출**했는가(D_manifest 의 검산 효과)
  - `fake_survived` — `S_fake` 의 날조 성분이 통과했는가 ← **매입 계측**
  - `negative_result` — 결과 < 0 인가(부호 게이트 A8 의 사전 계수)
- **사전 판정 규칙**
  - `A_asis` 가 093 에서 2.75, 094 에서 5.5 를 재현하지 않으면 → 계기 결함
  - `B_merge.expected_delta` ≥ 5/8 ∧ `N_neg` ≈ 0 → **덮어쓰기가 원인** ⇒ OL-07 처방 자격
  - `C_gate1.n_components_kept` − `A_asis` ≥ 5 **∧** `S_fake.fake_survived` − `A_asis` < 3
    → 관문1 완화 자격. **두 조건 중 하나라도 깨지면 완화 금지**([[70]] 짝 부호)
  - `D_manifest.recompute` − `N_neg` ≥ 5 → 성분 내역 노출이 검산을 만든다 ⇒ B4 승격
  - `E_ref` 는 `n_components_kept` 가 아니라 **base 값의 변화**로만 읽는다
- **무엇을 파는가**: `C_gate1` 은 **날조 통과율**을 판다(그래서 `S_fake` 가 필수). `D_manifest` 는
  반환문 길이 ↑. `B_merge` 는 충돌 표면화가 결정 지연을 만들 수 있다(턴 계수 동반).

---

### x474 — `fee_net_wording_grid` : ATM fee NET 집계 문면 3세대 (B8)

**[[62]] 4문**: ①문면만으로 credit 호출 **개수**가 갈리는지 잰다 ②갈리면 레버는 **문면(전달)** 뿐
③결정론 추가 0 ④`delta_total` 은 **복원하지 않는다**([[23]]·[[62]] — 채점 인자 공급 금지).

- **결정점 출처 sim**: 073#0[58]→[59]/[68] · 073#0[73]→[76] · 073#0[83]→[84] · 073#1[43]~[45]→[50]
  + 074#0[38]/[44]/[50] · 074#1[38]/[44] = **결정점 8~9개**(comparator 반환 직후 턴).
- **팔** (문면만 다르다 — 나머지 바이트 동일)
  | 팔 | 문면 꼬리 |
  |---|---|
  | `A_t7336` | 현행(SCOPE 2문장 + *"ONE fee_refund credit for the net correction **of THIS account**"*) |
  | `B_t7335` | *"…net correction **across all identified fee discrepancies** of THIS account"* (SCOPE 문장 **없음**) |
  | `C_split` | **분해안** — SCOPE 문장 유지 + 수식어 복원, **NET 절을 SCOPE 뒤가 아니라 앞**에 |
  | `D_scope_tail` | SCOPE 유지 + 수식어 복원, NET 절은 **꼬리 그대로**(위치 축 1변수 분리) |
  | `N_neg` | `A_t7336` + 무내용 한 줄 |
  | ⛔ `= ${delta_total}` 팔은 **만들지 않는다** | t7328 의 무효 통과를 재현하는 팔은 실험을 무효화한다 |
- **닫힌 채점 술어** (다음 credit 호출들)
  - `n_credits_per_account` — 계좌당 credit 호출 **개수**(1 ↔ N) ← **1차 수치**
  - `sum_matches_lines` — 호출 금액 합이 comparator 가 인쇄한 라인 차액 합과 일치(닫힌 산술·gold 0)
  - `rebate_scan` — 반환 직후 턴에 **rebate 관련 read/발화**가 있는가 ← P5 의 **매출** 계측
  - `no_credit` / `wrong_family`(credit 아닌 도구)
- **사전 판정 규칙**
  - `A_t7336` 의 `n_credits_per_account` 중앙값 > 1 이어야 라이브 재현(아니면 계기 결함)
  - `B_t7335 − A_t7336` (per-account = 1 인 비율) ≥ 5 → **수식어가 NET 을 산다** ⇒ OL-27 확정
  - `C_split ≈ B_t7335` ∧ `C_split.rebate_scan ≈ A_t7336` → **분해 성공**(둘 다 취함) ⇒ A17 로 승격
  - `D_scope_tail < C_split` (차 ≥5) → **위치 축**이 실재(문단 꼬리 매몰)
  - 전 팔 `rebate_scan` ≈ 0 → **P5 의 매출은 문면으로 살 수 없다** ⇒ 072 축은 별도 처방
- **무엇을 파는가**: `B_t7335` 로 되돌리면 P5 가 사려던 rebate 보완검사를 판다 — 그래서
  `C_split`(분해)이 본 팔이다. `C_split` 은 반환문 길이 ↑(컨텍스트).

---

### x475 — `redundant_write_iso` : 잉여·비가역 write 억제 (B17 · B15 보조)

**[[62]] 4문**: ①*"이미 한 write 가 있다"* 만 알려주면 모델이 스스로 멈추는지 잰다 ②멈추면 레버는
**표면화(전달)** 뿐 — 차단 신설 불필요 ③엔진은 원장 조회만 ④무엇을 쓸지는 모델이 정한다.

- **결정점 출처 sim**
  - 잉여형: 057#1[70](세 번째 checking 개설 직전) · 055#0[56](동일 인자 재개설) · 055#1[46](Gold 추가) · 063#1[042]
  - 비가역형: 079#1[94]/[96]/[98](선호 청취 전 주문 3건) · 079#1[102](chk_2 재주문)
  - 닫힌 술어 = *"직전 손님 발화 이후 같은 도구·같은 `account_type`(또는 같은 `account_id`)으로
    **성공한 write 가 이미 있는데** 또 쓰려는 턴"* — 태스크 id·도메인 어휘 0.
  - 결정점 8개.
- **팔**
  | 팔 | 변수 |
  |---|---|
  | `A_asis` | 현행(억제 없음) |
  | `B_ledger` | 도구-결과 꼬리에 **원장 사실만**: *"this conversation already has a successful `{tool}` for `{key}`: {값 축자}"* — **하지 말라는 말 없음**·대안 지목 0 |
  | `C_enum_ask` | `B_ledger` + **열린-열거 인자**(스키마 `enum` 이면서 값이 대화에 축자 부재)를 **이름으로 열거**하고 손님 선택을 1회 요구 — 079 형 |
  | `S_sham` | `B_ledger` 와 같은 문면에서 **key 만 규칙 치환**(다른 계좌·같은 노출 계급) — 내용이 원인인지 |
  | `N_neg` | 무내용 한 줄 |
- **닫힌 채점 술어**
  - `write_suppressed` — 다음 턴에 그 write 를 **안 냈다**(tool_calls 에 부재)
  - `asked_user` — 대신 질문(tool_calls==[] ∧ 물음표 종결)
  - `write_same` / `write_changed_args`(인자가 바뀌어 나감 — C_enum_ask 의 목표)
  - `enum_from_context` — 열거된 enum 값 중 **손님 발화에 축자 있는** 것을 골랐는가
- **사전 판정 규칙**
  - `A_asis.write_same` ≥ 5/8 (라이브 재현) 아니면 계기 결함
  - `B_ledger.write_suppressed` − `N_neg` ≥ 5 ∧ `S_sham ≈ N_neg` → **원장 표면화만으로 닫힌다**
    ⇒ [[62]] ②대로 레버는 전달뿐 · 차단 게이트 **신설 금지**
  - `B_ledger ≈ N_neg` → 표면화로 안 닫힘 ⇒ **그 단계에만** 결정론 검토([[62]] ③)
  - `C_enum_ask.asked_user` − `B_ledger` ≥ 5 → 079 형 비가역 축은 별도 레버 자격
- **무엇을 파는가**: 표면화가 늘면 **정당한 2회 write**(같은 계좌에 두 번 credit 이 옳은 케이스)를
  지연시킨다 — `write_changed_args` 와 over-action(`ONLY-PRED`)을 짝으로 센다.
- **B15 보조 블록**: 같은 결정점 풀에서 `_uacts` 를 **env 손님-도구 레지스트리와 교집합**한 팔을
  하나 더 두고 `[T2_RESOLVE] user-action instruct` 의 **표적 정확도**(오지목 `submit_transaction` 수)를
  센다 — 057#0 의 4회가 현재 기준선.

---

### x476 — `gate_timing_iso` : 브레이크·진입 술어의 타이밍 (B12 · B13)

**[[62]] 4문**: ①브레이크가 결정점에 **있을 때** 모델이 프로토콜로 가는지 잰다 ②가면 레버는
**타이밍(전달)** 뿐 ③새 게이트 0(기존 게이트의 면제/억제 조건만 건드린다) ④어느 프로토콜인지
우리가 안 고른다(⛔[[05]]/[[66]]: 증상 문구·태스크 id 로 조건 금지).

- **결정점 출처 sim**
  - `waive` 축: 033#0[008]→[012](notice 산문 발화 후 일반 이관) · 033#1[030]/[044](대조·pass) ·
    런 전체 `deny waived (notice already announced)` **3회** ↔ 실집행 1회 — 그 3자리 전부
  - `index` 축: 033#0 turn 4(=`_m3` 가 상품 축 247자였던 자리) · 085#0 turn 37 · 063#0 turn 17
  - `phase` 축: 063#0(전 궤적 — `verify_identity` 0회) · 016#0/#1(검증은 했으나 phase 사각 아님·대조)
  - 결정점 8~9개.
- **팔**
  | 팔 | 변수 |
  |---|---|
  | `A_asis` | 현행 |
  | `B_waive_doc` | `T2_PREKB` 면제 조건에 **닫힌 술어 추가**: *"notice 발화 시점에 그 도구를 정의한 문서가 이 대화에서 회수된 적이 있는가"*(`_docs_naming` 재사용) — 없으면 면제 대신 **1회 CHECK-FIRST 통과** |
  | `C_index_on` | `T2_ACTION_INDEX` 의 `not _m3` 억제 해제 — **닫힌 형태**(*"이 sim 에서 아직 1회도 안 나갔으면 `_m3` 유무와 무관하게 나간다"*) + `status=="deny"` 전속 해제 |
  | `D_phase` | 검증 게이트 진입 술어 확장(조건부·[[70]] ③ 분해): `phase != verify` 여도 **`formalized_target` 이 auth 게이트의 `applies_to` 에 속하는 턴에 한해** 미충족 조상을 1회 표면화 |
  | `N_neg` | 무내용 한 줄 |
- **닫힌 채점 술어**
  - `protocol_tool_named` — 다음 발화/호출에 **A3 `action_index` 가 담은 도구명**이 등장(닫힌 substring)
  - `general_transfer_emitted` — 일반 `transfer_to_human_agents` 를 냈는가 ← **팔아야 할 것**
  - `doc_read_emitted` — 문서 본문 read(`cat`/`KB_search_*`)를 냈는가
  - `verify_emitted` — `verify_identity`/`log_verification` 을 냈는가(D_phase 전용)
  - `spurious_hold` — 게이트가 **정당한 행동을 지연**시킨 턴 수(Δspurious)
- **사전 판정 규칙**
  - `A_asis.general_transfer_emitted` ≥ 5/8 (033 계열 재현) 아니면 계기 결함
  - `B_waive_doc.general_transfer_emitted` − `A_asis` ≤ −5 → 면제 절충 자격
  - `C_index_on.protocol_tool_named` − `N_neg` ≥ 5 → 억제 해제 자격.
    ⚠**동시에** `spurious_hold` 와 over-action 을 재고, ≥5 증가면 **조건부 발화로 후퇴**
  - `D_phase.verify_emitted` − `A_asis` ≥ 5 **∧** `spurious_hold` 증가 <5 → 진입 술어 확장 자격
    (`t2_phase.py` 주석의 경고 — *"무조건 확장은 통과 런을 죽인다"* — 를 이 조건이 집행한다)
- **무엇을 파는가**: `B_waive_doc` 은 면제의 원 취지(C210·004 *"동의-터미널 직후 deny 가 마지막
  행동 턴을 소각"*)를 판다 → **004 를 결정점 풀에 반드시 포함**한다. `C_index_on` 은 배달 예산
  4536자 × 발화 자리 증가. `D_phase` 는 통과 런의 행동-유도 소거 위험.

---

### x477 — `real_name_deny_iso` : 진짜-이름·무출처 부류의 deny 문면 (B18 · B19 + OL-01/05 문면 축)

**[[62]] 4문**: ①deny 문면이 *사실화*되면 모델이 회복하는지 잰다 ②회복하면 레버는 **문면(전달)** 뿐
③deny 판정 자체는 그대로(허용으로 바꾸지 않는다) ④정답 이름을 우리가 대신 부르지 않는다.

- **결정점 출처 sim**
  - `UNLOCK_PROV`: 050#0[58] 직전(regen 이 `…_5847` 를 낸 자리)
  - `DISPATCH_ROLE`: 057#1[116](`deposit_check` bare) · 055#1 turn 50(`deposit_check_mobile_app`)
  - `GIVE_QUOTE`: 040#1 turn 23(retract) · 055#0 msg 72
  - `no_record_template`: 017#1[13]→[14] · t7328A t0/t1 · t7335A(대조 4건)
  - 결정점 8~9개. 닫힌 술어 = *"우리 층 deny 가 나간 직후 턴"*(로그 태그 기준).
- **팔**
  | 팔 | 변수 |
  |---|---|
  | `A_asis` | 현행 deny 문면 |
  | `B_factual` | **deny 유지·문면만 사실화** — 이름이 env 레지스트리(`_agent_discoverable`/`_user_discoverable`)에 실재하면 *"이 이름은 실재하나 이 대화에서 회수되지 않았다 — 정의 문서를 열어 확인하라"*. 부존재 단언 삭제 |
  | `C_allow` | 레지스트리 실재 시 **deny 를 통과**(A5 의 강한 변형 — ⚠별도 팔로만) |
  | `D_askkey` | `no_record_template_v2` + **v1 조회-키 절 병합**(*"Ask the customer for their name, email, or user ID — phone and DOB cannot be used to look up"*) — `x379` 팔 추가로 대체 가능 |
  | `E_prodguard` | `T2_GIVE_QUOTE` 조건화 — `give 대상 ∈ arg_producers.values()` 면 **재질의 대신 표면화만** |
  | `S_sham` | `B_factual` 문면에서 **이름만** 규칙 치환(레지스트리 밖 이름) — 내용이 원인인지 |
  | `N_neg` | 무내용 한 줄 |
- **닫힌 채점 술어**
  - `recovered_call` — 다음 턴에 **그 이름으로** unlock/give/call 을 냈는가
  - `doc_read` — 정의 문서 read 를 냈는가(B_factual 이 노린 경로)
  - `asked_name` — (D_askkey 전용) 요청 문장에 **`name`/`full name`** 이 축자 등장
  - `folded` — tool_calls==[] 로 접힘 · `hallucinated_suffix` — 레지스트리 밖 접미사 방출 ← **매입**
- **사전 판정 규칙**
  - `A_asis.folded` ≥ 5/8 (라이브 재현: 057#1·050#0 의 접힘) 아니면 계기 결함
  - `B_factual.recovered_call + doc_read` − `N_neg` ≥ 5 ∧ `S_sham ≈ N_neg` → **문면 사실화가 산다**
    ⇒ A5/A13 을 "문면만" 형태로 승격(허용 변경 불요)
  - `C_allow − B_factual` ≥ 5 **∧** `hallucinated_suffix` 증가 <3 → 그때만 허용 변형 검토
  - `D_askkey.asked_name` − `A_asis` ≥ 5 → B19 승격. **<5 이면 017#1 은 모델 분산으로 확정**하고
    레버 신설 없음([[66]] 과폭 회피)
  - `E_prodguard` 는 **철회율**(producer/비-producer 분모 분리)로 부호표를 낸다
- **무엇을 파는가**: `B_factual` 은 환각 차단의 **심리적 강도**를 판다(문면이 부드러워진다) —
  `hallucinated_suffix` 가 그 계측. `E_prodguard` 는 010 형 **여분 give** 차단을 판다 →
  010 계열 결정점을 풀에 포함해 부호를 같이 본다.

---

### x478 — `overfire_negative_control` : 오발화 레버를 좁힐 때 무엇을 파는가 (B16 · B20)

**[[62]] 4문**: ①이 레버들이 **정당하게 사는 자리**가 실제로 있는지 잰다 ②없으면 좁힘은 순이득
③새 결정론 0(게이트 조건만) ④떠먹이기 0. — 이 프로브는 **매출이 아니라 매입을 재는** 전용 팔이다.

- **결정점 출처 sim** — 두 블록으로 나눈다.
  - **블록 P(피해 자리)**: `T2_VALUE_ACQUIRE` 가 태스크-무관 producer 를 민 자리 —
    079#0 turn 29/31/… (3회 cap) · 085#1[41]/[43] · 072#1[34]
  - **블록 Q(정당 자리·부정통제)**: 같은 레버가 **사는** 태스크 —
    040#0[33]~[43](give 성사 → user 가 last4 실행 = gold 040_3/040_4/040_5) ·
    2026-08-05 철회 이력의 **053 계열** 결정점(원장 C4 기록에서 좌표 회수)
  - `T2_PRESCRIPTION` 도 동형 2블록: P = 073#1 turn(오유도) · Q = **사기 dispute 태스크**의
    `apply_statement_credit` 시도 자리(신호가 `role="user"` 로 실재하는 sim)
  - 결정점 P 6 + Q 6 = 12.
- **팔**
  | 팔 | 변수 |
  |---|---|
  | `A_asis` | 현행(게이트 없음) |
  | `B_gate` | `T2_VALUE_ACQUIRE` 발화 전제 = ①그 `write` 도구가 이 대화에서 unlock/시도된 적이 있다 **∨** ②직전 tool 출력이 그 `arg` 결핍 에러 형상 (둘 다 닫힌 술어) |
  | `C_cap1` | 게이트 없이 `T2_VALUE_ACQUIRE_CAP` 만 3→1 (강도 조정·[[70]] ①) |
  | `D_userconv` | `T2_PRESCRIPTION` `_conv` 에서 **`role="tool"` 제외**(신호 화자를 손님으로 한정) |
  | `N_neg` | 무내용 한 줄 |
- **닫힌 채점 술어**
  - `lever_fired` — 그 자리에 레버 문면이 나갔는가
  - `wrong_lane` — 다음 호출이 **다른 계열 도구**인가(079 의 credit-카드 흡착 형상: 닫힌 = 도구의
    A3 `doc_index` 군이 대화 주제 군과 불일치)
  - `gold_family_call` — 블록 Q 에서 **그 레버가 노린 계열**의 호출이 나갔는가 ← **매출 계측**
  - `redirect_taken` — (D_userconv) deny 가 지목한 도구로 갈아탔는가
- **사전 판정 규칙**
  - 블록 Q 에서 `A_asis.gold_family_call` ≥ 5/6 이어야 *"정당 자리"* 로 인정(아니면 그 좌표 폐기)
  - `B_gate`: P 에서 `lever_fired` ≤ 2/6 **∧** Q 에서 `gold_family_call` 감소 <3 → **좁힘 자격**
  - `B_gate` 가 Q 에서 3 이상 잃으면 → **`C_cap1` 로 후퇴**(끄기 아니라 강도 조정·[[60]]/[[70]])
  - `D_userconv`: P 에서 `redirect_taken` ≤ 2/6 **∧** Q 에서 감소 <3 → B20 승격.
    Q 블록에서 **신호가 `role="user"` 로 실재하는 sim 을 하나도 못 찾으면** → 그 사실 자체가
    *"이 레버의 정당 자리가 코퍼스에 없다"* 는 관측이고, 그때는 선언 자체를 재상정한다
- **무엇을 파는가**: 이 프로브의 존재 이유가 *"판 것을 재는 것"* 이다. `B_gate` 는 2026-08-05 에
  ①만 걸었다가 **053 이 죽어 철회**된 이력이 있으므로 **①∨② 완화형**으로만 잰다.
  `D_userconv` 는 사기 태스크의 정당 deny 를 판다.

---

### 8.9 새 프로브 요약

| 프로브 | 덮는 ⓑ 항목 | 결정점 수 | 팔 수 | 핵심 사전 판정 |
|---|---|---|---|---|
| **x471** `diag_actionsub_iso` | B1 | 5~8 | 5 | `B_nodiag − A_asis ≤ −5`(오진 제거) · x213 4 사례 재측정 동반 |
| **x472** `requirement_axis_iso` | B2 · B10 · B11 | 8 | 6 | `B_req − N_neg ≥ 5`(요구 전달) · `D_fit − B_req ≥ 5` 여야 표 신설 |
| **x473** `savings_apy_component_iso` | B3 · B4 | 8 | 7 | `C_gate1` 은 `S_fake` 와 **짝으로만** 승격 |
| **x474** `fee_net_wording_grid` | B8 | 8~9 | 5 | `C_split ≈ B_t7335` ∧ `rebate_scan` 유지 = 분해 성공 |
| **x475** `redundant_write_iso` | B17 (+B15 보조) | 8 | 5 | `B_ledger − N_neg ≥ 5` ∧ `S_sham ≈ N_neg` |
| **x476** `gate_timing_iso` | B12 · B13 | 8~9 | 5 | `spurious_hold` 증가 <5 를 **모든 팔에 동반 조건**으로 |
| **x477** `real_name_deny_iso` | B18 · B19 (+OL-01/05 문면) | 8~9 | 7 | `B_factual` 승격 시 `hallucinated_suffix` 동반 계측 |
| **x478** `overfire_negative_control` | B16 · B20 | 12(P6+Q6) | 5 | 블록 Q 가 **매출**을 못 보이면 좌표 폐기 |

**총 8종 신규.** 이들과 기존 4종(x466·x467·x468·x470)을 합치면 ⓑ 20항목 중 **19항목**이 덮인다 —
남는 하나는 **B9(074 금액 축)** 이고, 그것은 프로브가 아니라 **정책 문서 grep** 이 선행 단계다
(문서에 조항이 없으면 기권 유지가 옳고, 있으면 [[72]] 1회 저작 · gold 경유 금지 [[23]]).

---

## §9. 정직 절 — 못 사는 것 · 표본 한계 · 미해결 계기

### 9.1 이 종합이 **사지 못하는 것**

1. **13/40 의 증가분 +7 은 수리에 귀속되지 않았다.** 15편 어디에도 인과가 없고, 분석된 신규 pass
   2건(033 t1·050 t1)은 **모델의 검색 도구 선택**으로 갈렸다. 003·004·017 t0·024 는 **기전 미조사**
   (실패 sim 이 아니라 per-step 포렌식 대상 밖). ⇒ *"수리 스택이 6/40 → 13/40 을 만들었다"* 고
   말할 근거가 **없다**.
2. **우리 층을 다 고쳐도 12 sim 은 안 산다**(§5.8). 그중 7 은 model 1차이고 5 는 our_layer 1차이나
   전부 **같은 sim 에 다른 잔여 차단막**이 문서화돼 있다.
3. **074 는 write 를 고쳐도 pass 가 아니다.** 금액 축(rebate 상계 + Light Blue `null`)이 미구축이라
   MISSING 4 → **WRONGARG 4** 로 바뀔 뿐이다(4/4 계좌 gold 불일치 실측).
4. **040 은 read 를 강제해도 안 닫힐 수 있다.** 015 의 5기준 중 4(이력 상한)만 read 에 달렸고,
   **6건 중 4건은 read 없이도 결정 가능**했다 — 정책 술어 적용 실패는 read 강제로 안 사는 부분이 있다.
5. **`eligible_for_provisional_credit` 의 정책값을 엔진이 기계 도출하는 것은 열린 판단**이다
   (x467 docstring 축자). 엔진이 그 값을 채우면 [[62]] 위반이고 실험이 무효가 된다.
6. **경계는 하나도 확정하지 못했다**(§6.3). ⓒ = 0건이고, 후보 3종은 전부 **미측정**이다.

### 9.2 표본·계기 한계

| 한계 | 내용 | 영향 |
|---|---|---|
| **sha 상이** | t7328 ↔ t7336 은 sha 가 다르다 ⇒ **엄밀 A/B 아님**. 전 성적 비교는 **[M]** | 6/40→13/40 을 인과로 읽을 수 없다 |
| **`info.git_commit` 신뢰 불가** | 055 §8 실측: t7336/t7335/t7328 **네 결과 파일의 `info.git_commit` 이 모두 `fc0055dc` 로 동일**. 057 보고서는 sha 를 `fc0055dc`, 016/033/050/085 는 `c273d93f` 로 적었다 | **런 식별을 이 필드로 하면 안 된다** — 파일명·경로로만 |
| **nt=2** | 태스크당 2 sim. 부호 ±1 은 **분산과 구별 불가**. 003 보고서 자신이 *"표본 2의 분산 가능성 유보"* 를 적었다 | +2/−1 부호를 인과로 읽지 말 것 |
| **t7335 대조가 nt=1** | 073 의 P5 판정 근거는 **t7335 1 sim ↔ t7336 2 sim**. 문면 diff 는 CONFIRMED 지만 인과는 **[M]** | x474 로 격리 확정 필요 |
| **채점표 결손 3 sim** | 072#0(`reward_basis=null`) · 074#0 · 079#1(`reward_info` 부재·CWE) ⇒ **변이 판정 불가**. 079#1 의 `missing=0·extra=13` 은 **아티팩트**이지 판정이 아니다 | 27 중 3 sim 은 변이 축이 아니라 **도달 깊이**로만 읽었다 |
| **ACTION-basis 사각** | `mutation_diff` 가 033 양 trial에서 **전 항목 빈칸**(OL-49). 이 문서의 033 표는 `action_checks` 축자다 | 계기 부채 — A16 로 큐잉 |
| **체인 밖 뷰 채널** | 055 §8: 모델의 *"It seems there was an error…"* 중 일부는 **우리 deny 에 대한 정확한 반응**인데 저장 궤적엔 안 보인다 | **포렌식이 이것을 날조로 오독할 위험 상존**([[55]] 계기 경고) |
| **촉구 draft 미영속** | C593 §6 한계: 촉구 부착 턴의 **draft 본문**이 regen 으로 대체돼 안 남는다 | *"사칭 발화 직후였나"* 를 구조로만 판정 |
| **P1 재판정의 근사** | C593: claim 의 `tool` 지목값이 로그에 없어 **kind 패턴**으로만 재판정 | 88% 오발화율은 비율은 같으나 **지목 미스 ↔ event_map 공백**을 못 가른다 |

### 9.3 미해결 계기 (측정 도구 자체의 부채)

1. **`T2_WRITE_ARG_GROUND` 의 `inner=` 공란 인쇄**(072#0) — 어느 인자가 걸렸는지 로그에 안 남는다.
   그래서 OL-06 의 deny 사유가 **UNPROVEN** 으로 남았다.
2. **`T2_SG_ISOLATE` 의 ground-피드백 항목명 미기록**(093) — 라이브 로그가 *"ground-피드백 1건"*
   만 남겨 **무엇이 반려됐는지** 못 짚었다. x456 프로브로 우회했고 그래서 [M] 로 강등됐다.
   ⇒ 다음 런에 `T2_SG_ISOLATE_TRACE` ON 권고(무료).
3. **`T2_CP2_CLOBBER` 가 안 잡는 덮어쓰기 경로**(C593 행 2) — `VIEW_FB`(`:8635`)는 `_cp2_assign` 을
   거치지 않아 **계기에 안 잡힌다**. 촉구 64회 덮임이 로그에 흔적 없음.
4. **사이드카에 도달 기록 없음** — `fb_*.jsonl`·`trace_*.jsonl` 에 `"Carry out the next step"` **0건**.
   [[55]] *"로그 마크 ≠ 전달"* 의 실물. `arrived` 플래그 신설이 A/B 의 **선행 조건**이다.
5. **`T2_ENVELOPE_GUARD` 배선 생존 미확인**(OL-51) — 술어 3항 성립인데 미발화. 오프라인 테스트
   (`tool_calls=None` + `<tool_call>` + 길이 150k)로 [[67]] 0단계부터.
6. **`FAB_STRIP`·`T2_ARG_PRODUCERS` 가 halfA 런 전체 0회** — P4/F8 은 **배선 생존부터** 재검할 것
   ([[67]] `t2_liveness` 0단계). 072 보고서가 명시적으로 이 플래그를 세웠다.

### 9.4 이 문서가 지킨 규율 · 어긴 것

- [[69]] 채점 단위 = `reward`. `action_checks` 는 진단 보조로만 썼고 반례 2건을 명시했다.
- [[23]] gold 무경유. 모든 A2/선언 처방에 **정책·env 출처**를 병기했고, 출처를 못 대는 것
  (074 금액 축·085 dispute_category)은 **처방하지 않고 grep 선행으로 큐잉**했다.
- [[70]] 모든 처방에 **±를 공개**했고 끄기 대신 절충/분해를 적었다. ⓐ 16건 전부에 계측 의무를 달았다.
- [[62]] ⓒ(경계)를 **0건**으로 남겼다 — 라이브 null 을 경계 증거로 쓰지 않았다(C593 교훈).
- ⚠**어긴 것 / 위험**: §5 의 "기대 상한"은 원 보고서들의 잔여-차단막 진술에 기대 **추정**한 것이고
  측정값이 아니다. 15 sim 이라는 수를 **성과 예측으로 인용하면 안 된다**.
- ⚠ 축 배정(§2)은 *"reward 를 죽인 결정 지점 하나"* 라는 저자 판단이 들어간다. 원 보고서의
  `cause_primary` 를 그대로 옮겼으나 074#1 은 **혼합**(WRONGARG=our_layer / MISSING=model)이라
  단일 축 배정이 불가능했다.

---

### provenance

`reports/facet_rft_2026/t7336_tasks/T7336_TASK_{016,017,033,040,050,055,057,063,072,073,074,079,085,093,094}.md`
(15편 전수) · `T7336_FORENSIC_HALFA_2026_08_22.md` · `T7336_FORENSIC_HALFB_2026_08_22.md` ·
`CLAIM_DEMAND_ISO_VS_LIVE_AUDIT_2026_08_22.md`(C593) · `T7336_FORENSIC_016_2026_08_21.md` ·
`T7336_FORENSIC_033_2026_08_22.md` · 프로브 docstring
`scripts/distill/tau2/{x466_id_resolution_iso,x467_policy_boolean_doc_iso,x468_actual_apy_formula_grid,x470_claim_demand_live_iso}.py`.
**전부 로컬 읽기 전용** — SSH 0 · 결과 gz 무접촉 · 프로브 스크립트 무수정 · 코드 수정 0 · 수리 0.

