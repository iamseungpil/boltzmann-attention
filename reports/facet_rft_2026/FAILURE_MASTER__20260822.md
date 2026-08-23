# bank_t7346 실패 마스터 종합 — 13 태스크 · 23 실패 sim · 반증자 75 판정 반영 (2026-08-22 런 / 작성 2026-08-23)

> 자료 = **태스크당 1 에이전트의 per-step 궤적 포렌식 13편**(016·017·040·050·055·057·063·072·073·
> 074·079·085·094) + 같은 항목에 대한 **반증자(반박 팔) 판정 75건**. 이 문서는 그 둘을 종합할 뿐
> **새 수치를 만들지 않았다**. 없는 수치는 "미제공/미상"으로 남긴다.
>
> ---
> ⛔**2026-08-23 정정 — 이 문서의 DUP/MATCHED 칸은 계기 결함 위에 있었다.**
> `t2_forensic.deny_kind` 가 env 의 `Failed to …` 거절을 **성공으로 세고** 있었다(A-7⑵·커밋
> `73efa6f7`). 전수 반경 = 365 sim 중 **26 sim 분류 변경 · DUP 17 sim 소멸 · MATCHED→BLOCKED 31건**.
> **reward 는 안 바뀐다**(벤치가 매긴다) — 바뀌는 것은 DUP/MATCHED 서술과 **그 위에 세운 수리 공적**이다.
> ⇒ 특히 §3.1 의 *"085: t7328 DUP 5 → 0 유지"* 는 **무효**다(t7328 085 = 성공 변이 1 · 거절 15).
> **정본 = `MASTER_DUP_CORRECTION_2026_08_23.md`. 이 문서의 DUP/MATCHED 를 인용하기 전에 그것부터 읽어라.**
> ---
>
> **승격 규율(이 문서의 뼈대)**: 반증자 **CONFIRMED 만** 우리-층 결손으로 승격한다.
> **UNPROVEN·REFUTED 는 등급을 그대로 달고 표에 남긴다** — 지우지 않는다(다음 런이 같은 자리를
> 다시 파지 않도록). CONFIRMED 안에서도 **기전 확정**과 **reward 인과 확정**을 갈라 적는다
> ([[08]] 집계→결론 직행 금지 · [[56]] 근거 확보한 쪽이 우세 · [[62]] 결손을 격리로 재라).
>
> **수리·코드 수정 0.** 처방은 큐로만 남긴다. gold(`reward_info`)는 원 보고서들이 진단용으로
> 읽은 판정을 옮길 뿐이고 A2 내용을 저작하지 않는다([[23]]).

---

## §1. 성적표 (입력에 있는 수치만)

### 1.1 런 총점 — **미제공**

입력에 런 성적 문장이 없다. 부수 언급 두 곳이 **서로 어긋난다**:

- `TASK_057` 반증자: *"이 런의 0.0 기저율이 28/40(70%)"* ⇒ 통과 12/40
- `TASK_079` 반증자: *"12 sim 중 1통과/11실패 vs **전체 13/40**"* ⇒ 통과 13/40

두 진술은 같은 런을 말하면서 1건 차이가 난다. **총점을 이 문서에서 확정하지 않는다.**
아래 태스크별 수치만 인용 가능한 사실이다. (포렌식 대상 13 태스크 = 26 sim 중 실패 **23 sim**.
런이 40 sim 이면 나머지 14 sim 은 이 포렌식 밖이다 — 그 성적은 미제공.)

### 1.2 태스크별 성적 · 변이(정본 `t2_forensic.mutation_diff`)

| 태스크 | t0 | t1 | 점수 | 변이 요약 | 채점축 |
|---|---|---|---|---|---|
| **016** | 0.0 | 0.0 | 0/2 | 양 trial MISSING 1 (`submit_transaction{friend_user_5839, Silver Rewards Card, Best Buy, 750, Shopping}`) · t1 DUP 2(`log_verification`·DB 무변) | DB |
| **017** | 0.0 | **1.0** | 1/2 | t0 MISSING 2 (`submit_cash_back_dispute_0589`×2) + EXTRA 2 (`file_credit_card_transaction_dispute_4829`×2) | DB |
| **040** | 0.0 | 0.0 | 0/2 | t0 gold 8건 **전량 MISSING**(dispute 0건 실행) · t1 8건 실행했으나 `issue_noticed_date` 8/8·`address` 7/8·`eligible_for_provisional_credit` 6/8 오값 | DB |
| **050** | 0.0 | **1.0** | 1/2 | t0 MISSING 1 (`submit_credit_limit_increase_request_7392`) · t1 clean(matched 3) | DB |
| **055** | 0.0 | 0.0 | 0/2 | t0 MISSING 3 · WRONGARG 4 · t1 MISSING 2 · WRONGARG 2 (전부 `account_class`/그 downstream `account_id`) | DB |
| **057** | 0.0 | 0.0 | 0/2 | 양 trial MISSING 3 (`open_bank_account_4821{checking,"Blue Account"}` · `give`/`call deposit_check_3847`) · WRONGARG 1 (`account_class`) · t1 종료 `max_steps` | DB |
| **063** | 0.0 | 0.0 | 0/2 | t0 MISSING 2 · WRONGARG 3 (`card_type` Platinum↔Silver · `account_class` Gold↔Silver Plus) · t1 MISSING 2 · WRONGARG 1 | DB |
| **072** | 0.0 | 0.0 | 0/2 | t0 MISSING 1(Bluest credit) · WRONGARG 1(3.00↔3.50) · t1 MISSING 1 · WRONGARG 1(**12.00↔14.00** = 누락 fee_rebate $2.00) | DB |
| **073** | **1.0** | 0.0 | 1/2 | t1 MISSING 2 · WRONGARG 6 — `account_id`·`credit_type` 전부 일치, **어긋난 것은 `amount` 뿐이고 합계는 정확**(9.50/9.00) | DB |
| **074** | 0.0 | 0.0 | 0/2 | 양 trial MISSING 4 (credit ×4 계좌) · t0 WRONGARG 7(라인별 분할·과행동) · t1 WRONGARG 0(크레딧 **0건 실행**) | DB |
| **079** | 0.0 | 0.0 | 0/2 | t0 matched 11 · MISSING 2 · WRONGARG 3 · BLOCKED 27 · DUP 1(**계기 위양성**) · t1 matched 1 · **MISSING 11** · BLOCKED **0** | DB |
| **085** | 0.0 | 0.0 | 0/2 | t0 matched 1 · MISSING 3 · WRONGARG 4 · BLOCKED 18 · t1 matched 1 · MISSING 3 · BLOCKED 6 (BLOCKED 24건 **전부 env**) | DB |
| **094** | 0.0 | 0.0 | 0/2 | 양 trial matched 1 · MISSING 2 · WRONGARG 2 — 식별자/범주 6/6 일치, 어긋난 것은 **수치 3필드**뿐이고 셋 다 `get_correct_savings_apy` 출력의 downstream | DB |

**합**: 13 태스크 = **3/26**(017 t1 · 050 t1 · 073 t0). 실패 sim **23건**.

⚠ **채점축은 전부 `reward_basis=["DB"]` 다. `action_checks` 로 읽으면 거짓 결론이 난다**([[69]]).
이 런의 실물 반례 **3건**:

| 반례 | 관측 | 함의 |
|---|---|---|
| **073#t0** | `action_checks` 의 credit 3행 **전부 `action_match=false`** 인데 **reward 1.0** | 액션표로 "실패"라 읽으면 통과를 실패로 센다 |
| **017#t1** | `action_match` 2/4 실패인데 **reward 1.0** | 동상 |
| **085** | `action_checks` **10/13 match** 인데 양 trial **0.0** | 액션표로 "거의 다 됐다"고 읽으면 결손이 사라진다 |

---

## §2. 원인 축별 군집표

축은 **23 실패 sim 의 결정 지점**에서 나왔다. 각 sim 에 reward 를 죽인 결정 지점 하나를
기준으로 1차 축을 배정하고, 반증자 판정을 반영해 귀속을 재정한다.

### 2.1 ★1차 귀속 — 반증자 통과 전 / 후

| 1차 귀속 | 원 보고서 자기 진술 | **반증자 통과 후** | 차이 |
|---|---|---|---|
| `our_layer` | **17 sim** | **5 sim** | **−12** |
| `model` | 6 sim | **12 sim** | +6 |
| **미상**(기전은 우리 층 CONFIRMED · reward 인과 미확정) | 0 | **6 sim** | +6 |
| `env` / `user_sim` | 0 / 0 | **0 / 0** | — |

- 원 진술 `our_layer` 17 = 016×2·017#0·040#0·050#0·055×2·057#1·063×2·072×2·074×2·079#1·094×2
- **반증 후 `our_layer` 1차 5** = **017#t0 · 050#t0 · 057#t1 · 074#t0 · 079#t1**
- **반증 후 `model` 1차 12** = 040×2 · 055×2 · 057#0 · 063×2 · 072#1 · 073#1 · 079#0 · 085×2
- **미상 6** = 016×2 · 072#0 · 074#1 · 094×2

이것이 이 종합의 첫째 산출이다. **per-task 에이전트는 자기 층을 과잉 귀속한다** — 17→5.
[[21]] 준수는 유지된다(user_sim 1차 0건: 040#1·073#0·074#1 의 손님 압박은 전부 agent 흡수로 환원).
[[25]] 관점에서도 env 1차 0건(085 의 `Missing required parameters.` 무명 거부는 루프를 만들었으나
채점 무관이고, 같은 궤적에서 이름을 댄 거부는 전부 1턴에 교정됐다 = [[64]] 대조군).

### 2.2 우리-층 결손 축 (CONFIRMED 항목만 배치)

| 축 | 이름 | CONFIRMED 항목 수 | 태스크(#trial) | reward 인과까지 선 것 |
|---|---|---|---|---|
| **E** | **게이트 우회 채널** — 재생성(regen)·봉투 가드가 만든 호출이 게이트 체인 밖에서 커밋 | 3 | **050#0**(`_ap_regen`→`T2_PROCEDURE` 미재평가) · **057#0**(`T2_ENVELOPE_GUARD` regen 이 while-루프 밖) · 040#1(final-word deny 재생성 실패로 위반 호출 커밋) | **050#0 (강)** |
| **C** | **선언 공백** — 정본 A2 에 항목이 아예 없어 술어가 도달조차 못 함 | 7 | **085**(debit 분쟁 필링이 인자-거버넌스 6종에서 통째 누락) · **079#0**(`order_debit_card_5739` 앞 게이트 0) · 073(`STALE_STRIP` wtools 커버리지 · `SG_DOCS` isolate.docs 부재) · 094(`required_groups` 부재-검사 없음) · 063(`missing_hint` 미선언) · 074(LB 스케줄 `null`) | 없음(전부 부재 진술) |
| **D** | **선언 간 모순** — 같은 A2 의 두 선언이 서로를 부정 | 3 | **079#1**(`declared_required` ↔ `arg_source_reads.card_id`) · 040(`tool_signatures` ↔ `write_arg_grounding` + chain 순서) · 050(READ-FIRST ↔ 절차 `requires` 순서) | **079#1 (준강)** |
| **B** | **우리 층 오지목** — 비존재/오도 도구·행을 권위 문구로 지목 | 6 | **017#0**(`T2_OWNERSHIP_FIX` 겹침=0 가지) · 016(`T2_DIAG` Platinum) · 057(`T2_DISCOVERY_STEP2` `apply_savings_account_credit_6831`) · 063(동상) · 074#1(`user-action instruct target=submit_transaction`·비존재) · 079(`T2_GROUND -> CARLOS RODRIGUEZ` 15/15) | **017#0 (준강)** |
| **A** | **근거창 파괴** — 결정점 전에 우리가 사실을 지운다 | 5 | **057#1**(`CP2_CLOBBER` gold 결정문 247자 폐기) · 016(`T2_VIEW_MSG_CAP=8000` 다이제스트) · 073(`recent_tool_text` tail-cut) · 094(`ref_from_outputs` 래퍼 미뚫음) · 055(축 소진) | **057#1 (중)** |
| **F** | **순서·예산·캡** — 요구 발화 전 소진, 결정점 앞 재-read 유발 | 4 | 055(축 소진 술어에 요구 조건 없음) · 072#0·073#1(`T2_COVERAGE_FU` 가 결정점 앞 6257자 재-덤프 유발) · 016(검색 예산 3) | 073#1 재-덤프 **CONFIRMED(강)** — 단 그 자체가 실패를 만든 인과는 미입증 |
| **G** | **부호 반전 grounding** — 정답을 막고 오답을 통과시킴 | 2 | **094**(gold `actual_apy=5.1` **5회 반려** ↔ 손님의 틀린 `6.0` 통과 ↔ **자기 write 가 원장에 남은 뒤 허용** = C203 자기-그라운딩) · 074(`STALE_STRIP` 이 정당 write 삭제) | **094 OL-B (강)** |
| **H** | **계기(instrumentation) 결함** — 성적 무관·다음 포렌식을 오도 | 6 | 017(`prohibited=` 미인쇄 → 자기 보고서 주장⑤를 반대로 오도) · 079(`deny_kind` 가 `Failed to …` 를 성공으로 접음 → DUP 위양성) · 016(`_t2_vc_logged` sim당 1회) · 073(로그의 "근거 N자" 가 서브 코퍼스와 다름) · 055(247자 동형 페이로드) · 094(operand-size 에 kind 미인쇄) | 해당 없음(성적 무관) |

### 2.3 축별 대표 축자 근거

| 축 | 대표 sim | 축자 근거 | 결정성 |
|---|---|---|---|
| **E** | **050#0** | `_ap_regen`(:10762)은 `unified()` **while-루프 밖**의 직선 코드 → 재생성 호출이 `T2_PROCEDURE`(:7343-7401) 재평가를 못 받는다. 로그 168라인에 `[T2_PROCEDURE]` **0줄**(cap 도 선점도 아님 — 둘이면 `would-fire but suppressed by=` 가 찍힌다). **정본 워커로 오프라인 재생** → `verdict=deny missing=['submit_request','disputes','pending_replacement']` = **짝 trial 1 이 라이브로 받은 deny 문자열과 축자 동일** → t1 reward 1.0 | ★런 전체 40 sim 커밋 호출 전수 재판정에서 deny 대상은 **이 한 건뿐** |
| **E** | **057#0** | 체킹 write(msg26) 턴 로그가 `[T2_ENVGUARD] tool-call envelope unparsed (len=296) — required-channel regen` → `regen tool_calls=['call_discoverable_agent_tool']` 뿐. `T2_ENVELOPE_GUARD` 블록(:11084)은 게이트 while-루프(:7053) **바깥·뒤**, 8칸 들여쓰기 확인 ⇒ `T2_WRITE_ARG_ENUM` 포함 **체인 전체 우회** | ★**반증자가 새로 찾은 미보고 결함**. 40 sim 중 `[T2_ENVGUARD]` 는 057#s626729 **유일** |
| **D** | **079#1** | `declared_required`(t2_resolve.py:1118-1131) 18개에 `get_all_user_accounts_by_user_id_3847` 은 **있고**(→ [060] 통과) `get_debit_cards_by_account_id_7823` 은 **없다**(→ 그 다음 4회 거부). 같은 A2 의 `arg_source_reads.card_id` 는 그 이름을 **필수 원천 read 로 선언**하고 `_note_card_id` 에 *"079 실패 경로가 정확히 이 두 read 부재였다"* 축자 | ★정본 로더로 병합 A2 적재 후 재계산 → **원소 18개가 한 글자도 다르지 않음** |
| **B** | **017#0** | `[T2_OWNERSHIP_FIX] fired give-name=file_reward_discrepancy → agent tool(s) ['file_credit_card_transaction_dispute_4829', …]` → 같은 턴 모델 unlock. **`file_credit_card_transaction_dispute_4829` 는 msg 35 이전 궤적 어디에도 없다**(최초 출현이 unlock 인자 자신). 엔진 자신의 `UNLOCK_PROV allow` 술어도 '대화에 없던 이름'으로 판정 | 런 전체 OWNERSHIP_FIX fired 5발/4 sim **전부 reward 0.0** |
| **A** | **057#1** | `[T2_DOCDECIDE] → 'Blue Account'`(gold) → `재료 247자 배달` → **`[T2_CP2_CLOBBER] VIEW_FB 가 미소비 배달물 247자를 버리고 7413자로 덮어씀`**. `decided_by_docs_text.format('Blue Account')` = **정확히 247자**(길이 검산). t1 전체에 `DECISION_CARRY … (247자)` **0회** = 소비 전 폐기 | 구제 분기는 `len(_prev) >= 10000` 하나뿐 → 247자는 구조적으로 폐기 |
| **G** | **094#t1** | 모델이 `408×12/96000 = 5.1%` 로 **gold 값을 정확히 유도**했는데 `T2_SG_GROUND` 가 **5회 드롭**(msgs 23·25·39·45·47). t0 에서 **손님의 틀린 6.0 은 통과**. [49] 자기 write 출력 `Actual APY: 5.1%` 가 원장에 남은 뒤에야 허용 | ★엔진 술어(`_strip_own_feedback`+`_nums_in`)를 그대로 재현해 **cut 별 True/False 재계산** — 유일 출처가 자기 write |
| **F** | **073#1** | `[T2_COVERAGE_FU] fired tool=get_atm_fee_discrepancies`(turn 39) → msg[39] 본문 0자 + `_3` 재-read → msg[40] **6257자 덤프** → 근거창 4000자에서 comparator 결과 3건(2666자) 축출. 문면이 *"Read the missing value(s) … and call it again"* 이므로 재-read 는 **우리가 지시한 행동**. 재생성 전 초안은 산문 사임 ⇒ 이 6257자는 **우리 층이 없었으면 state.messages 에 못 들어왔다** | 같은 레버가 t0 에서는 write **이후**(msg 49) 발화해 비용 0 ⇒ **부호가 위치에 의존**([[70]]) |
| **C** | **085** | `write_arg_grounding` 7항이 전부 credit 계열, `file_debit_card_transaction_dispute` **0항**. `ref_verify`·`ref_iso`·`write_evidence_specs`·`have_value_reask`·`value_acquisition` 동일. 그 결과 debit 분쟁 22건이 무검문 통과 | 반증자: 같은 런 **040 은 credit 쌍둥이가 전부 살아 있는데도 0.0/0.0** ⇒ 선언 유무가 성적을 안 가름 |

### 2.4 축을 가로지르는 반복 서명 5종

1. **우리 deny 가 env 에러로 오인된다.** 079#t1 은 궤적 전체에 **env 발 `Error:` 0건**인데 모델이
   "unlock 에러"에 3회 반응하고 이관으로 끝났다([040]/[050]/[064]). 074#t1 도 동형.
   ⇒ 우리 문면이 손님-대면 서사가 된다([[25]] 우리 도구 100% 정답 의무 위반의 하류).
2. **같은 레버가 태스크마다 부호가 갈린다.** `T2_SEARCH_REARM`: 016·055·057 음(−) ↔ 050#1·073#0
   양(+·통과 sim). `T2_COVERAGE_FU`: 073#1 음 ↔ 073#0 무해. `user-action instruct`: 074#1 음 ↔
   **033·100·003·024 통과 sim 에서도 동일 발화**. ⇒ [[70]] 끄기 금지·태스크별 부호표 의무.
3. **정답이 문맥에 있는데 안 쓴다(전달로 안 닫히는 잔여).** 055#t0(정답 스펙 문서 + 공식명 9개
   동시 제시, assistant 의 'Silver Plus' 언급 **0건**) · 072(도구 지시·문서 23편·거래 레코드 3중
   근거인데 rebate 검사 0회) · 073#1(정책 축자 3회 도달, 합계는 정확히 계산하고 호출만 분할) ·
   085(직전 tool 출력 `-14.99` 를 `49.99` 로 덮어씀). ⇒ [[62]] ①의 답이 "격리에서도 실패"인 부류.
4. **격리 서브가 재료를 못 받거나 잘못 받는다.** 016(`T2_SUB_REQUIREMENT` 경로 없음) ·
   094(`ref_from_outputs` 가 래퍼 이름을 훑어 정답 레코드 미도달) · 073(tail-cut) ·
   055/063(요구 미전달). ⇒ 그러나 **055/063 축의 "요구를 실으면 산다"는 이미 라이브 A/B 에서
   기각됐다**(원장 C508 — §5.3).
5. **줄번호 드리프트.** 073 반증자가 *"보고서의 `t2_gate_patch.py` 인용 5건 **전부** 오프셋"* 을
   확인했고, 040(9287→9342)·057(−17)·016(8169→8210)·050(7355/10883)·094(1495/1517)·074(8169→8152)
   에서 같은 현상. **런 sha(`ee18d797`) 프리즈본이 아닌 워킹트리를 읽은 결과**다.
   ⇒ 수리 에이전트가 claim 의 줄번호로 파일을 찾으면 실패한다(§8).

---

## §3. 직전 런(t7336) 이후 들어간 수리·레버 — 실측 성적표

판정 칸: **발화** = 트리거가 성립해 문면/게이트가 실제로 나갔나 · **발화하고도 못 삼** = 나갔는데
결정을 못 바꿨나 · **기회 0** = 술어/선언상 이 태스크에 도달 불가(= 死선언) · **미점화** = 배선은
있는데 플래그가 0(= 死배선 아님). [[55]] 死배선과 무효과를 가른다.

### 3.1 발화하고 **샀다**(행동 변화 + gold 행 획득이 관측된 것)

| 수리 | 발화 sim | 산 것(축자) | 남은 부채 |
|---|---|---|---|
| **A6①/OL-37** `requires_reads += get_all_user_accounts_by_user_id` (READ-FIRST 가 **이름을 댄다**) | **072#1 · 074×2 · 085×2 · 094×2**(072#0 은 PROV 선점으로 0회) | 이 런 **최대 양성**. 074: `[READ-FIRST] … Their exact callable forms are: unlock_discoverable_agent_tool(agent_tool_name="get_all_user_accounts_by_user_id_3847")` → [12] 두 도구 unlock ⇒ t7335 의 **"@last 날조 → 거래행 통짜 날조 → 우리 도구가 날조를 세탁"** 사슬 **소멸**. 094: gold 094_1/094_2 `action_match=true`. 085: t7335 를 죽인 계좌-식별 도달 결손 **닫힘**. 072#1: 계좌-id 획득 통과·gold 072_1~072_5 신규 매치 | **read 를 사고 write 를 못 산다** — 074/085/094 전부 다음 단계(인자 값)에서 잃었다. 072#0 은 **우리 층 내부 충돌**로 발화조차 못 함(`T2_PROV` 가 comparator 3건을 선점) |
| **A5/OL-01** `T2_UNLOCK_PROV` 출처에 env 레지스트리 추가 | 050×2 | `[T2_UNLOCK_PROV] registry-provenanced (allow) … val=approve_credit_limit_increase_5847` — t7336 의 **실재 이름 자기차단 재발 0** | 그것이 연 문을 **절차 게이트가 못 지켰다**(축 E) |
| **GB1 Recovery 문면 축소** | 072×2 | t7336 t0 을 `too_many_errors` 로 죽인 **7-필드 되묻기 루프 재발 0**, `log_verification` [16] 성공 | — |
| **`t2_resolve.py:209` 중복-write 대칭 가드** | 057#0 ×5 | `operator-find 침묵: chosen=open_bank_account_4821 는 이미 성공 실행` ⇒ **EXTRA 0 · DUP 0**(t7336 t1 의 잉여 write 2건 소멸) | — |
| **`T2_WRITE_EVIDENCE`** update_transaction_rewards 차단(017 선행 수리) | 017#0 | `deny … inner=update_transaction_rewards_3847` — 선행 대체 경로 **실제 차단**, 커밋 인자 날조 소멸 | 모델이 **새 대체 도구를 찾았고 그것을 손에 쥐어 준 것이 우리 문구**(축 B) |
| **`CLAIMPROV` kind-폴백 / 에러-형상 게이트 수리** | 085(51·15회) · 017#1(승리 요인) | 085: t7328 DUP 5 → **0 유지**. 017#1: `unbacked kind='give'` 로 허위 "I have enabled the tool" 3회를 잡아 give 강제 재생성 → **reward 1.0 을 실제로 샀다** | 017#1 의 승리는 **모델 날조 문구의 우연**(`dispute` 토큰)에 선행 의존 — 안정 능력 아님 |
| **`T2_ARG_PRODUCERS`(F8) 에러-형상 게이트** | 085 **0회 오발화** | t7335 의 F8 오발화 → **0** | 정당 발화도 0(t7336 마스터가 이미 기록한 상쇄) |
| **A15 `T2_BLOCK_NOTE` regen** | 074#1 | `[T2_BLOCK_NOTE] regen ok (356 chars)` — 노트가 본문 전체를 잡아먹지 않음 | 재생성된 **본문 자체가 허위**([23] "I have now successfully verified … and logged it") |
| **A1/OL-17** `_stale_call_ids` ok_ids 를 `t2_resolve._result_ok` 로 교체 | (073 docstring 축자로 확인) | 수리 자체는 착지 | 남은 부채 = **wtools 커버리지**(축 C) |

### 3.2 발화했는데 **못 샀다**(무효과)

| 레버 | 발화 | 실측 결과 | 판정 |
|---|---|---|---|
| **`T2_SEARCH_REARM`**(t7336 OL-29 수리로 ON) | 016(turn 32·11,240자) · 055#1(turn 21·13,998자) · 057×2(7,413자) · 072#0(2회) · 073#0 · 050#1 | 016: 신규 대상이 **platinum**(gold 는 Silver) → 0 매수 · 055#1: **이미 정답인 checking 축**에 마지막 예산 · 057: **모델 자신의 오답 발화를 재수요로 읽어** 오답 계열 배달, t1 에서는 그 배달이 CP2 를 통해 **gold 결정문을 파괴** | **발화 O · 부호 혼재**. 050#1·073#0(통과 sim)에서는 양(+) ⇒ 끄기 금지·조건 조정([[70]]) |
| **P5 완결-인상 문면 승격(2026-08-21)** | 072#1 [54]/[55] 축자 도달 | *"This tool did NOT check whether any rebate is missing … check the account's rebate policy … yourself"* 를 받고도 **rebate 검사 0회** → $12 vs gold $14 | **발화·무시** — 문면 승격만으로는 안 닫힌다 |
| **`T2_OWNERSHIP_FIX` 손님-측 레지스트리 조회(t7336 처방4)** | 055#1 `suppressed(user-side)` · 017#1 동상 | 055: gold 055_6 양 trial 회수(**DB 축이라 reward 0 기여**). 반증자: 055_6 은 t7328(F/T)·t7336(T/F)·t7346(T/T) 로 **매 런 뒤집힘** = n=1 | **거동 변화 CONFIRMED · 매수 미확정** |
| **A13/OL-05** OWNERSHIP_FIX 겹침>0 가지 닫기 | 017×2 | **겹침=0 가지는 그대로** → t0 이 새어 나가 EXTRA 2건 + reward 0 | **부분 수리 · 실물 손실** |
| **`T2_DISCOVERY_STEP2` 레지스트리 폴백** | 079#0(양성: gold read 지목) · 063#0·057#0(음성: 오지목) | 079 에서는 t7336 의 첫 벽을 돌파, 063/057 에서는 존재하는 오도구를 지목해 2턴 소모 | **부호 혼재** |
| **R3 `ref_from_outputs`(신규)** | 094#0 오발화(1024자 카드 덤프) · 094#1 **침묵**(fail-open) | 정답 레코드 생산자가 `call_discoverable_agent_tool` 이라 `producer_contains=["accounts"]` 가 못 뚫음. 반증자: **093 은 REFRAW 없이도 회복**(REFERENCE 에 레벨 이름만 있으면 됨) | **표적 미도달 · 기대 상한 낮음** |
| **`T2_MATERIAL_GATE` resolve_cap** | 016(turn 38~46) · 055 · 072#1 · 085#1(5회) | 016 반증자 ★: Silver 확정 turn 40·손님 질문 직후 42·치명 답변 44 가 **전부 `stop=resolve_cap(정체 3회)` 로 먼저 막혔다** — 보고서가 지목한 검색 예산 가드는 **도달조차 못 했다** | 016 처방3(예산 칸 분리)은 **결정점을 못 연다** |

### 3.3 발화 **기회 자체가 없었다**(死선언 — 술어/선언상 도달 불가)

| 레버 | 태스크 | 도달 불가 사유(선언 축자) |
|---|---|---|
| `T2_SG_DOCS`(본런 ON) | **063·073** 0줄 | `scaffold_get_tools[...].isolate` 에 `docs` 키 미선언 → `t2_scaffold_get.py:737 isinstance(iso.get("docs"), dict)` 실패로 함수 자체가 미호출 (094·093 에서는 정상 발화 = 死레버 아님) |
| `T2_REQUIRE_DOC_DELIVER` | **055·079** 0줄 | `require_doc_before.tools` = 이관 4종뿐 → `open_bank_account_4821`·`order_debit_card_5739` 관할 밖 |
| `T2_WRITE_ARG_ENUM` | **079** | `applies_when.prefix = "open_bank_account"` 하나뿐 |
| `T2_ARG_EMPTY` | **040#1** | discoverable 도구가 `agent.tools` 에 **아예 없다**(디스패처만 등록) → `_schema_required` 가 항상 None. **정본 함수로 4/4 케이스 재현** |
| `T2_ARG_DOC_SUB`(신규 ON) | **063** 5회 | 전부 `spend_category` — 축 무관 |
| `T2_VALUE_FORMULA=full`(신규 ON) | **063** 0회 | 이 태스크 결정 축에 미도달 |
| `write_arg_grounding` 외 5종 | **085** | debit 계열 항목 0건(축 C) |
| `T2_STALE_STRIP` | **073** 0회 | `_wtools` 에 credit 계열 write 미등재 → 규칙② 도달 불가 |
| `T2_ACTIONREQ` 후보 집합 | **074#1** | 손님-측 액션 4종이 A2 정적 목록이라 이 태스크(user_tools=`[]`)에서 영원히 비지 않음 |

### 3.4 **미점화**(배선 있음 · 플래그 0)

| 레버 | 상태 | 판정 |
|---|---|---|
| `T2_PENDING_DISCOVERED=0` | **네 런 연속 미반영**(057·016) — 그 결과 `deposit_check_3847` 이 후보에 영원히 없고 **오지목 `submit_transaction` ×4** | **미이행 처방**. 단 057 반증자: 그 두 턴은 `tools=None` 이라 어차피 호출 불가였다 ⇒ 인과 순위는 낮춰야 함 |
| `T2_RETRY_CONTROLLER` | 057 반증자 ★: **배선은 실재**(`RETRY_LOOP` deny + `T2_WRITE_CAP`)하는데 현행 스택 미등재·로그 0회 | 보고서의 "술어 부재" 는 **거짓**. 새 레버 짓지 말고 이것을 격리로 재라([[67]]) |
| `T2_MATERIAL_BYPASS` | go_stack 미등재 | 085 반증자 ★: **의도적 철회**(C498ⓕ *"앞선 수리는 표적을 빗나갔다"*·`LEVER_ROSTER_CANONICAL` 에 [S] 등재). **[[60]] 위반 아님 — 재론 금지** |
| `T2_SUB_REQUIREMENT=0` / `T2_VERDICT_CARRY=0` | 055·063 이 지목 | 063 반증자 ★: **C508 라이브 A/B 완료**(0/8↔0/8·기전 반증·지연 2.4×·**승격 금지**) + C506 소급 정정 + C510 전달 축 폐쇄. **재론 금지** |
| `T2_VERDICT_GATE=0` | 063 | 호출-트리거형은 미측정이나 push 형(`VERDICT_CARRY`)은 **C543 에서 음수 실측**(073 1.0→0.0) ⇒ ± 없이 켜기 금지 |
| `T2_CP2_QUEUE=0` | 런 이후 커밋(9d217b39·844fa7a2) | 이 런에 미적용. 계기 실물은 098#s626729 |

### 3.5 **미이행 처방**(코드/선언에 들어가지도 않은 것)

| 선행 처방 | 출처 | 이번 런 재현 |
|---|---|---|
| **R1** `_grounded_candidates` operator 인자 분리 | T7336_TASK_079 | `T2_GROUND -> CARLOS RODRIGUEZ` **15/15** 재현(079#0) |
| **R5** 비가역 write 열린-enum 게이트 | T7336_TASK_079 | 079#0 이 정확히 그 자리에서 손실(주문 3건 자기 기본값 커밋) |
| **OL-27** comparator 반환문에 정책 축자 `across all identified fee discrepancies` 복원 + **격리 x474** | T7336_TASK_073 | 문면 **바이트 동일**·x474 스크립트/결과 **전무** |
| **P-B**(PROV 폴백 → `arg_source_reads`) · **P-D**(STALE_STRIP `_wtools`) · **P4**(FAB_STRIP 해소-read) | T7336_TASK_072 | P-B 미착수가 **072#0 의 직접 사인 후보**(단 UNPROVEN) |
| **P5** `actual_apy` 파생 검산 | T7336_TASK_094 | 미이행 + **역효과 관측** — 파생된 gold 값 5.1 이 5회 반려됨 |
| **P1** 손님-측 목록 병기(OWNERSHIP_FIX) | T7336_TASK_055 | 017 에서 겹침=0 가지로 재발 |

---

## §4. 회귀 전용 절 — 내려간 태스크마다 "무엇을 팔았나"([[70]] 의무)

| 태스크 | 계보 | 무엇을 팔았나 | 등급 |
|---|---|---|---|
| **063**(카드 축) | t7328 **2/2** → t7336 1/2 → t7346 **0/2** | **미상.** 보고서가 지목한 우리-층 1차 2건은 반증자가 **둘 다 REFUTED**(주장1 `T2_SUB_REQUIREMENT`=C508 라이브 기각 · 주장2 열거 동봉=런 내 반대 사례 3건). 이번 런 신규 ON 3종(`T2_ARG_DOC_SUB`·`T2_VALUE_FORMULA=full`·`T2_SG_DOCS`)은 **두 결정 축에 하나도 닿지 않았다**(§3.3). 관측된 유일한 변화 = `get_correct_savings_apy` 가 t7336 에서는 6.5 를 **계산**했는데 t7346 은 3회 전부 `(could not compute)` → 무정보 abstain 이 표류를 낳았다 ⇒ **팔린 것 후보 = grounding 강화로 인한 계산 불능(미확정)** | **미상** |
| **072#0** | t7328 t0 은 두 계좌 감사·Light Green **$3.50 정답** → t7346 t0 은 Bluest 감사 **0회** + $3.00 | 후보 = `T2_PROV` 무명 폴백이 comparator 3건을 turn 36 에 선점해 `T2_SG_REQREADS` 0회. **반증자 UNPROVEN** — 형제 trial 1 은 REQREADS 가 발화해 Bluest 를 실제 감사했는데도 072_7 오답이었고, t0 는 선점 뒤에도 스스로 계좌목록에 도달했다(072_1/2/4 `action_match=true`) | **미상**(우리-층 후보 1건 UNPROVEN) |
| **079#1** | t7336 에서는 **같은 seed 가 그 read 를 통과**해 카드 9변이를 전부 정확히 냈다 → t7346 은 ID-해결 도중 붕괴(MISSING 11) | ★**특정됨**: `operator-scope` 가 `get_debit_cards_by_account_id_7823` 을 **4회 거부**(축 D). 그리고 **같은 unlock 이 t0 turn 32 에서는 무해 통과** ⇒ 통과/거부가 후보 집합 크기·직전 손님 발화에 좌우되는 **우리 층이 만든 비결정성** | **CONFIRMED** — 이번 런에서 회귀 원인이 특정된 **유일 건** |
| **040#1**(필드 축) | t7328 t1 은 gold dispute 2건 정확 + 나머지 5건 `eligible` **한 칸만** 오답 → t7346 t1 은 같은 5건이 `address`+`issue_noticed_date` 까지 추가 오답 | **미상.** seed 가 같아도 user-sim 발화가 다르다(t7328 손님은 *"I'd rather not share my full address here"* 로 그쳤고 에이전트가 원장 값을 유지) ⇒ 결정론적 회귀로 단정 불가. 우리-층 후보 `T2_ARG_EMPTY` 死는 CONFIRMED 이나 **필요조건이지 충분조건 아님**(같은 7행이 `issue_noticed_date` 로도 독립 오답 ⇒ 단독 수리의 reward 매수 **0**) | **미상** |
| **094**(수치 축) | t7328 6.5(−0.35pp) → t7335/t7336 5.5 → t7346 **t0 6.1(−0.75pp)** / t1 5.5(−1.35pp) | `T2_SG_DOCS` 계열 **내부**에서는 개선(5.5→6.1)이나 **t7328 기준선에 미달** ⇒ 이 태스크의 부호는 여전히 **음(−)**. sha 상이라 비엄밀 | **비엄밀 관측** |
| **074#1** | t7336 t1 도 *"tools to apply credits directly are not available to me"* 로 크레딧 0 — **동형** | **팔린 것 없음.** 보고서가 지목한 `user-action instruct` 는 반증자가 **부정통제 3갈래로 인과 REFUTED**(⒜t7336 동 seed 는 이 레버 **0회 발화**인데 같은 붕괴 ⒝t7346 t0 은 **동일 발화 후 크레딧 7건 정상 실행** ⒞task_033 은 동일 발화·**reward 1.0**) | **회귀 아님**(동형 유지) |

**개선 방향 기록**(회귀 아님·인과 미귀속):
- **073**: 계보를 **t7328 유효 0/2 → t7346 1/2** 로 정정해 읽어야 한다 — t7328 의 통과는 반환문이
  `= $9.50` 로 **채점 인자를 직접 공급**한 무효 통과([[23]]/[[62]]). t7346 t0 이 **첫 유효 통과**다.
- **057**: 발견 결손 해소 + 잉여 write 0(점수 0/2 불변). **050**: 기전이 DUP→MISSING(approve)→
  MISSING(submit) 로 두 세대 이동 ⇒ 메모리 [[69]] 의 *"050 은 승인 중복으로 실패"* 는 **낡았다**.
- **085**: 원인이 "도달 실패"에서 "인자 전사/검증"으로 한 단계 하류 이동(점수 불변).

---

## §5. 반증자 판정 반영 — 승격 명부

### 5.1 집계

| 판정 | 건수 | 비율 |
|---|---|---|
| **CONFIRMED** | **48** | 64% |
| **UNPROVEN** | **10** | 13% |
| **REFUTED** | **17** | 23% |
| 합 | **75** | |

태스크별 (C / U / R): 016 3/2/1 · 017 5/0/1 · 040 2/0/1 · 050 3/0/0 · 055 6/2/2 · 057 4/1/1 ·
063 2/2/3 · 072 1/2/1 · 073 7/0/1 · 074 4/0/2 · 079 **6/0/0** · 085 1/0/3 · 094 4/1/1.

### 5.2 ★우리-층 결손으로 **승격**(CONFIRMED 중 reward 인과까지 선 것 — 수리 큐 머리)

| # | 항목 | 태스크 | 코드/선언 | 인과 근거의 강도 |
|---|---|---|---|---|
| **1** | `_ap_regen` 산출물이 `T2_PROCEDURE` 재평가를 우회 | **050#0** | `t2_gate_patch.py` while-루프 밖 regen · 재검사 목록에 `procedures` 없음 | ★최강 — 정본 워커 재생 deny 가 **짝 trial 이 라이브로 받은 문자열과 축자 동일**, 그 deny 가 t1 을 1.0 으로 만들었다. 40 sim 전수 재판정에서 유일 |
| **2** | `declared_required` ↔ `arg_source_reads` 모순으로 gold 원천 read 4회 거부 | **079#1** | `t2_resolve.py:1118-1131` · `a2 arg_source_reads.card_id` | ★강 — 병합 A2 재계산 18원소 정확 일치 · env `Error:` 0건 · 우리 deny 반향 축자 3회 |
| **3** | `T2_SG_GROUND` 부호 반전 + 자기-그라운딩 | **094#1** | `t2_scaffold_get.py:438-443` + `_corpus_texts(...,"ledger")` | ★강 — 엔진 술어 재현으로 cut 별 True/False 확정, 5.1 의 유일 출처가 **자기 write** |
| **4** | `T2_OWNERSHIP_FIX` 겹침=0 가지가 손님-측 부재를 확인 없이 단언 | **017#0** | `t2_gate_patch.py:9780-9802`(A13/OL-05 미완) · `feedback_user_tool_is_agents` | 준강 — 지목된 이름이 궤적 최초 출현 · fired 5발 전부 0.0. ⚠단 같은 문면에 레지스트리 목록이 **병기**되므로 "문장 vs 목록" 분리 불가 |
| **5** | `T2_ENVELOPE_GUARD` regen 이 게이트 체인 전체를 우회 | **057#0** | `t2_gate_patch.py:11084`(while-루프 밖) | ★반증자 신규 발견 — 40 sim 유일. **처방 함의: 057 의 "집합-內 continue" 처방을 넣어도 t0 에는 발화하지 않는다** |
| **6** | `T2_COVERAGE_FU` 가 결정점 앞 재-덤프를 유발(부호가 위치 의존) | **073#1 · 072** | `t2_gate_patch.py:11172-11194` | 강(유발까지) · 실패 인과는 미입증 |
| **7** | `CP2_CLOBBER` 가 gold 결정문 247자를 소리 없이 폐기 | **057#1** | `_cp2_assign` 구제 조건 `len(_prev)>=10000` | 중 — 폐기 확정 · 배달됐다면 바뀌었을지는 부정통제 0 |
| **8** | `T2_PRESCRIPTION` 의 `_conv` 가 `role="tool"` 을 신호 코퍼스에 포함 | 073#0(무해) | `t2_gate_patch.py:9159-9160` | ★인과 직접 검증 — user 히트 0 / tool 히트 `['fraud']` 1 ⇒ 오발화 원인 확정 |

### 5.3 ★**등급 그대로 남기는 것**(승격 금지 — 다음 런이 다시 파지 않도록)

| 원 주장 | 판정 | 남기는 이유(반증 축자) |
|---|---|---|
| 016 "재무장이 마지막 검색 예산을 훔쳐 Silver 를 못 샀다" | **REFUTED** | turn 40·42·44 는 전부 `stop=resolve_cap(정체 3회)` 로 **먼저** 막혔고 예산 가드에 도달조차 못 했다. t1 은 msg[32:] 산문에 'Silver' **0회**라 예산이 남아도 못 샀다 |
| 016 "레버 발화표 수치가 로그와 일치" | **REFUTED** | 전수 재계수에서 절반 어긋남(SEARCH_AGENT 11/11→**6/5** 등). **이 표로 부호표·처방 우선순위 세우지 말 것** |
| 017 "절차 금지가 미전달됐다" | **REFUTED** | turn 63 에 **실제 전달**됐고 다음 턴 msg[65] 가 `collect` 어휘까지 되받았다. 보고서 자신의 ⑥항과 자기모순 |
| 040 "우리 층 모순 지시가 give 를 0회로 만들었다" | **REFUTED** | t7328 **같은 seed 가 give 5회 실행(전부 준수형)** · t7346 017#s373753 은 SIGNATURE deny 5회 후 give 2회 실행 + **reward 1.0** ⇒ deny 는 회복 가능 |
| 055 "CP2 클로버로 savings 결정문이 한 번도 소비되지 않았다" | **REFUTED** | 같은 247자가 `WRITE_ARG_ENUM` 경로로 t0 에서 **3회** 모델에 전달됐다(주장 3 이 스스로 증거) |
| 055 "055 결정점에 도달한 KB 재료는 247자 오답뿐" | **REFUTED** | t0 msg36 4위가 **`doc_savings_accounts_silver_plus_account_002`**(gold 스펙) — 'Silver Plus' 궤적 20건. **정답 문서가 있는데 assistant 가 한 번도 쓰지 않았다** ⇒ 전달 결손 아님 |
| 057 "실패한 동일 인자 write 재시도 술어가 없다" | **REFUTED** | `T2_RETRY_CONTROLLER`(RETRY_LOOP) + `T2_WRITE_CAP` **실재**. 정확한 진술은 "꺼져 있다" |
| 063 "`T2_SUB_REQUIREMENT` 死배선이 오답을 만들었다" | **REFUTED** | 원장 **C508 라이브 A/B**: 0/8↔0/8 · 기전 반증(treat 가 오히려 나빠짐) · 지연 2.4× · **승격 금지**. C506 소급 정정(x343 은 정보-맞춤에서 거짓) · C510 전달 축 폐쇄 |
| 063 "열거 게이트 동봉이 오답을 확정시켰다" | **REFUTED** | 같은 동봉(247자 Gold)을 받은 057×2·055#1 이 **자기 값의 최근접 정본 정규화**를 했고 Gold 를 무시. 궤적에 `It answers` **0건** |
| 063 "`T2_VALUE_ACQUIRE` 오발화가 하위목표를 주입" | **REFUTED** | 발화 조건 ③이 **모델 자신의 초안**을 요구 — 우리는 발기자가 아니라 수신자. 덧붙는 문장은 **억제문** |
| 072 "`_upending` 이 window 를 못 닫아 중복 write" | **REFUTED** | force 가드에 `_upending`/`_uacts` **부재**. 창은 `RESIGN` 으로 열린다 ⇒ 처방 P-1 무효과 |
| 073 "grounded_calls 가 원리상 라인별만 배달 가능" | **REFUTED** | `9.0`·`1.5` 는 **접지 통과** — 3계좌 중 2계좌 gold NET 은 배달 가능. 보고서 표가 스스로 모순 |
| 074 "`user-action instruct` 지목이 크레딧 0 을 만들었다" | **REFUTED** | 부정통제 3갈래(§4) |
| 074 "LB null 선언이 gold $14.50 중 $12.00 을 유실" | **REFUTED** | 정책 축자 요율로 채워 재실행해도 net **$4.00** ⇒ 회수는 **$1.50**, 나머지 $10.50 은 '월 2회 무료' **서수 술어** 요구 — 현행 `case`/`lookup_table` 어휘로 표현 불가. **8배 과대계상** |
| 085 "`T2_MATERIAL_BYPASS` 미등재가 결함" | **REFUTED** | 정본에 이미 [S] 로 박제된 **의도적 철회**(C498ⓕ). 게다가 사망 루프 구간에 `MATERIAL_GATE` 줄이 **한 줄도 없다** |
| 085 "SEARCH_REARM 이 write 를 놓쳤다" | **REFUTED** | turn 단위를 잘못 읽었다 — 재무장은 **첫 필링 시도 그 자리**(msg[69]) |
| 085 "`arg_producers` 가 debit 문의에 credit 도구를 민다" | **REFUTED** | 발화한 레버가 읽는 선언은 `value_acquisition` 이고 `arg_producers` 는 **미참조**. 지목 파일:키가 틀림 |
| 094 "`transactions_raw` 도 래퍼 결함" | **REFUTED** | 걸린 생산자는 **자기 실이름** ⇒ `_eff_tool_name` 처방으로 안 바뀜. **별개의 병(느슨한 needle)** |
| 016 T2_DIAG 인과 · 057 ⓕ 인자 접지 · 063 VERDICT_GATE · 072 PROV 결론 · 085 부속 · 094 OL-E 등 | **UNPROVEN 10건** | 격리·부정통제 없이 인과를 세우지 말 것 |

---

## §6. 처방 큐 — 3분할

### 6.1 (가) 무료 수리 가능 — 격리 불필요(닫힌 술어 · 계기 · 선언 동기화 · 모순 제거)

| # | 처방 | 표적 | 기대 상한(입력 근거) | [[70]] 판 것 |
|---|---|---|---|---|
| **A-1** | **regen 산출물을 절차 게이트에 재입력** — 루프 밖 `_ap_regen` 결과가 `T2_PROCEDURE` 를 우회하는 구멍을 닫는다 | **050#0** | **1 sim**(짝 trial 이 같은 deny 로 통과했다는 실증) | 재검사 비용·regen 지연. 40 sim 전수에서 deny 대상 1건뿐이라 폭발 반경은 작다 |
| **A-2** | **`T2_ENVELOPE_GUARD` regen 도 같은 체인에 재입력**(A-1 과 같은 가족·축 E) | **057#0** | 미상(1 sim 도달) | 동상 |
| **A-3** | **`declared_required` 재료에 `arg_source_reads` 합류** — 한 A2 안의 두 선언 모순 제거 | **079#1** | ID-해결 재개 → 카드 9변이 경로 복구. **reward 반사실은 미증명**(t0 는 같은 read 통과 후에도 0.0) | `operator-scope` deny 감소 ↔ 오도구 선택 증가. **태스크별 부호표 필수** |
| **A-4** | **`_evidence_ctx` 생산자 키 `_eff_tool_name` 정규화**(같은 파일이 READ-FIRST 에서 이미 쓰는 술어의 일관 적용·도메인 리터럴 0) | 094 · 072 | **낮음** — 093 반증 사례상 REFERENCE 에 레벨 이름이 없으면 회복 안 됨 | REFERENCE 바이트 1024→1629(+59%)·클래스 과포함 |
| **A-5** | **`get_checking_atm_fee_totals` 에 `ground`/`grounded_params` 선언**(기존 술어 재사용·새 배선 0 — 반증자가 처방을 정정한 항목) | 057 · 055 | 인자 날조 차단 | abstain 증가 |
| **A-6** | **`missing_hint` 1줄 선언**(출처 = `doc_index['savings_accounts']` 9키 기계 전개·[[72]] 1회 저작) | 063 | reward 매수 **0**(반증자: t0 는 이 도구 미호출로도 실패) — **[[64]] 문면 결손 해소만** | abstain 문면 길이 |
| **A-7** | **계기 6종**(성적 무관·[[25]] 우리 도구 100% 정답 의무) | 전 태스크 | 다음 포렌식 오도 제거 | 로그 부피 |

**A-7 세부**: ⑴`[T2_PROCEDURE] deny` 에 `prohibited=<pname>` 병기(017 — 이 결함이 자기 보고서
주장⑤를 **반대 방향으로 오도**했다) ⑵`t2_forensic.deny_kind` 에 `Failed to …` 형상 추가
(079 DUP 위양성 · 반경은 DUP 뿐 아니라 MATCHED/EXTRA 오분류까지) ⑶`_t2_vc_logged` 를 sim당 1회
→ 다이제스트된 메시지마다(016 — 실제 5개/4개인데 로그 1줄) ⑷`T2_WRITE_SUB` 로그의 "근거 N자"
가 **트리거 코퍼스**라 서브가 실제로 본 창(scope='all')과 다름(073) ⑸축 결정문에 **축 이름 병기**
(055 — `Blue Account` 와 `Gold Account` 가 **둘 다 247자**라 길이로 식별 불가) ⑹`T2_SG_ISOLATE`
operand-size 에 kind 목록 병기(094 — 이번 분석도 산술 역산에 의존).

### 6.2 (나) 격리 프로브 선행 필요 — [[62]] 순서 ①격리로 결손 측정 → ②되면 레버는 전달뿐 → ③그 단계만 결정론

| # | 프로브 질문(엔드포인트) | 표적 | 선행 이유 |
|---|---|---|---|
| **B-1** | `T2_MATERIAL_GATE` 의 `resolve_cap(정체 3회)` 를 **write 성공 이력이 있는 sim** 에서 완화하면? (엔드포인트 = 결정점 턴에 채널이 열리는가 · 부작용 = over-action) | **016**(turn 40·42·44 가 전부 이것에 막혔다) · 072 · 085 | 016 의 원 처방 3(예산 칸 분리)이 **반증으로 무효화**됐고 실제 차단자가 여기다. **공유 상류 노드 = 폭발 반경 큼**([[66]]) |
| **B-2** | `recent_tool_text` 를 tail-cut → **메시지 경계 역순 충전**으로 바꾸면 제안 품질이 오르는가? (부정통제 = 제안 N건 → 통과 0건 증가) | **073** · 050/072/074 | 기계 결함은 CONFIRMED 이나 **해악 미입증**(t1 은 pre-draft 를 실행조차 안 했다) |
| **B-3** | debit `write_arg_grounding` + `corpus_roles{disputed_amount:[tool]}` 를 켜면 **정답도 false-block 하는가**? (gold `discovery_date 11/14/2025` 가 시계 출력 형식과 불일치) | **085** | ⚠**reward 매수 0**(085 gold 3행 부재는 그대로) — **날조 차단만**. `arg_corpus_marker` 선행 |
| **B-4** | 비가역 write 의 **열린-enum 게이트**(닫힌 3항: ①스키마가 enum 선언 ②enum 값이 손님 발화에 축자 부재 ③재실행 거부 부류) | **079#0** · 040 | 선행 R5 미구현이 두 런 연속 같은 자리에서 손실 |
| **B-5** | LB 계열 **서수 술어**(월 N회 무료)를 ⓐ정책 축자 동봉만으로 닫는가 ⓑ보류 사유 고지로 닫는가 ⓒ무관 문서 통제 | **074** | ⓐ가 되면 레버는 **전달뿐**이고 op 확장은 불필요([[62]]②). 회수 상한은 요율 채움만으로 **$1.50** |
| **B-6** | `T2_RETRY_CONTROLLER`(이미 있는 배선) 를 켜면 동일-인자 반복이 멈추는가 · write 에 deny 스텁을 남길 때 **replay 정합 비용** | **057#1**(23회 반복→max_steps) · 085#1(5회) | [[67]] 새 레버 짓지 말 것 — **있는 것을 재라** |
| **B-7** | `actual_apy` 를 **파생 검산**(월 이자×12/원금)으로 바꾸거나 `on_fail: drop→flag` | **094** | 자기-그라운딩(C203) 제거. 판 것 = 날조 차단력 일부 |
| **B-8** | `require_complete_groups` 선언(그룹 **부재** abstain) | 094 · 073 | ⚠**순환** — 부재 판정의 원천이 곧 OL-A 가 망가뜨린 REFERENCE. B-9 와 짝으로만 |
| **B-9** | comparator 반환문 OL-27 문면 복원(**x474 선행**·엔드포인트 = 계좌당 credit 호출 **개수**) | 073 | [[62]]① — 문면 되살리기는 채점 인자 위조와 종이 한 장 차이([[23]]) |
| **B-10** | `T2_SEARCH_REARM` 재수요 술어의 **화자 축 분리**(user ↔ assistant)와 **부정문 배제** | 057 · 055 · 016 | 실물: 손님의 **부정문**(*"I haven't mentioned a Platinum Rewards Card"*)도 수요로 읽힌다. **끄지 말 것**([[60]]) — 050#1·073#0 에서 양(+) |

### 6.3 (다) 레버 없음(경계) — [[62]]①의 답이 "격리에서도 실패"인 부류

| 태스크 | 모델 결손(전달로 안 닫힘을 보인 근거) |
|---|---|
| **073#1** | 합계 9.50/9.00/1.50 을 **스스로 정확히 산출**(msg[61])하고도 호출만 라인별 분할. 정책 축자(`ONCE per checking account … combine them into a single credit`)가 KB 1회 + comparator 3회 = **4회 도달** |
| **085** | 직전 tool `amount: -14.99` 를 다음 메시지에서 `-$49.99` 로 덮어씀 · `discovery_date` 4건 전부 *"(assuming you discovered it the next day)"* 로 **묻지 않고 생성**(같은 메시지에서 손님은 *"I just noticed it today"*) · enum 오매핑 · 동일 인자 4~5회 재제출. **t7328 t1 과 자리까지 일치 = 2런·상이 sha·상이 seed 에서 3/3 재현** |
| **055#0** | 정답 스펙 문서(`silver_plus_account_002`, Score 21.34) + 우리가 준 공식명 9개가 **동시에** 앞에 있는데 assistant 의 'Silver Plus' 언급 **0건** |
| **072** | 도구가 축자로 *"check the account's rebate policy … yourself"*, 정책 문서 23편 배달, 거래 원장에 11/14 rebate 부재가 가시 — **rebate 검사 0회** |
| **063** | 두 trial 모두 **손님 신용점수를 묻지 않음**. t1 은 손님이 msg[32]에서 700 을 말한 뒤에도 `check_card_application_fit` 재호출 **0회** |
| **040#0** | 클린한 SIGNATURE 문구를 **3회** 받고도 5회 모두 `arguments` 를 실은 채 재발행([[42]] prior-override) |
| **079#0** | 배송·디자인을 **묻지 않고** 등급 정책 **미독** 상태로 비가역 주문 3건 커밋(*"We will use default options … to simplify the process"*) |
| **074#1** | ARGS-FORMAT deny(*"Re-issue this exact call"*) ×4 에 **4계좌 중 2계좌만** 재송신 — Purple·LightBlue 자발적 절단 |
| **016** | msg[40]에서 Silver·IN_PROGRESS 를 **스스로 확정**하고 msg[44]에서 헤지 · KB 재검색 3회 질의에 'Silver' **0회** |
| **057#1** | 동일 인자 write **23회** 반복 → `max_steps` |
| **094** | t0 이 고객 주장 6.0 을 복제(408×12/96000=5.1 은 **닫힌 산술**이고 재료가 msg[1]에 실재) |

---

## §7. 이 종합이 **못 사는 것**(정직 절)

1. **런 총점을 모른다.** 입력에 성적 문장이 없고 부수 언급 두 곳이 상충한다(28/40 zero vs 13/40 pass).
   이 문서의 13 태스크 3/26 은 **부분 집합**이고, 나머지 sim 의 성적·기전은 **미조사**다.
2. **승격 8건 중 reward 반사실이 증명된 것은 사실상 0건이다.** 050#0 조차 "짝 trial 이 같은 deny
   로 통과했다"는 **n=1 대조**이고 A/B 가 아니다. 079#1·094#1 은 기전만 확정됐다. **"고치면 +N"
   은 이 자료로 말할 수 없다.**
3. **격리 프로브 0건.** 이 문서의 어떤 처방도 [[62]]①(격리로 결손 측정)을 거치지 않았다.
   (나) 큐 10건은 **전부 미측정**이고, (가) 큐도 A-3/A-4 는 부호표 없이 켜면 안 된다.
4. **반증자도 한계를 자인한 자리들.** ⒜비커밋 채널(재무장 델타·REQUIRE_DOC 배달·DIAG 블록·
   pre-draft 본문)은 `state.messages` 에 안 남아 **재현 불가** — "생성-뷰 전체에 750 0회"는 커밋분
   한정이다(016). ⒝`t2_fbsidecar`(fb_*.jsonl)가 **리모트 전용**이라 무엇을 실제로 보냈는지
   대조하지 못했다(050). ⒞094 의 fail-open 은 **로그가 없는 침묵 기반 추론**이다.
5. **"모델 결손 12건"도 최종 판정이 아니다.** [[62]] 순서상 그것들도 **격리 프로브로 재야** 경계가
   확정된다. 지금은 "전달을 더 사도 안 산다"는 **관측**일 뿐이고, 특히 085·055·072 는 격리에서
   같은 결손이 나오는지 재지 않았다.
6. **처방의 기대 상한 대부분이 "미상"이다.** 표에 숫자를 적은 것은 A-1(1 sim) 하나뿐이고,
   B-3 은 **매수 0** 임을 미리 적었다. 나머지는 근거 없는 수를 만들지 않으려고 비워 뒀다.
7. **회귀 6건 중 원인이 특정된 것은 079#1 하나다.** 063(카드 축 2/2→0/2)이 가장 큰 회귀인데
   **무엇을 팔았는지 끝내 못 댔다** — 지목된 우리-층 후보 2건이 둘 다 REFUTED 됐고, 신규 ON 3종은
   결정 축에 닿지도 않았다. 이 상태로 t7346 스택을 래칫에 올리면 **대가를 모른 채 올리는 것**이다.

---

## §8. 다음 포렌식이 반복하면 안 되는 함정 (이번에 실물로 걸린 것만)

| # | 함정 | 실물 |
|---|---|---|
| **1** | **줄번호를 워킹트리에서 읽는다** | 073 반증자: 보고서의 `t2_gate_patch.py` 인용 **5건 전부** 오프셋. 040(9287→9342)·057(−17)·016(8169→8210)·074(8169→8152). **런 sha 프리즈본으로만 인용할 것** |
| **2** | **길이로 페이로드를 식별한다** | 055/057: `decided_by_docs_text` 는 `Blue Account` 와 `Gold Account` 둘 다 **정확히 247자** — 연대순으로만 구분된다. 057 보고서가 "247자 폐기 2회"라 적었으나 두 번째는 **673자로 덮은 savings 건** |
| **3** | **로그 `turn=` 을 assistant 서수로 읽는다** | 085: turn 65 를 "필링 이후"로 오독(실제는 `len(state.messages)` 라 첫 필링 **그 자리**). 073: turn=65 를 63턴 궤적에 대입 = 내부적으로 불가능 |
| **4** | **로그 마크를 전달로 읽는다** | 016 `_t2_vc_logged` 1줄 ↔ 실제 다이제스트 5회 · 073 "근거 N자" ↔ 서브 실제 창. [[55]] 0단계 |
| **5** | **집계 표를 손으로 센다** | 016 부수 주장 REFUTED — SEARCH_AGENT 11/11→6/5 등 **절반이 어긋남**. C583ⓐ 손 비교기 동형 |
| **6** | **부정통제 없이 인과를 세운다** | 040·074·063·085 의 REFUTED 4건이 전부 이 형태. **같은 레버가 통과 sim 에서도 발화한다**는 것을 먼저 확인할 것([[57]]) |
| **7** | **`action_checks` 로 성적을 읽는다** | 073#0(3행 전부 false·reward 1.0) · 085(10/13 match·0.0). [[69]] |

---

### 부록 A. 승격 8건 ↔ 처방 큐 대응

| 승격 # | 처방 | 큐 |
|---|---|---|
| 1 `_ap_regen` 우회 | A-1 | (가) |
| 5 `ENVELOPE_GUARD` 우회 | A-2 | (가) |
| 2 `declared_required` 모순 | A-3 | (가·부호표 필수) |
| 3 `SG_GROUND` 부호 반전 | B-7 | (나) |
| 4 `OWNERSHIP_FIX` 겹침=0 | (처방 미확정 — 목록 병기가 이 실패를 막는다는 보장 **미측정**) | (나) |
| 6 `COVERAGE_FU` 위치 | B-1 과 짝(공유 상류) | (나) |
| 7 `CP2_CLOBBER` | `T2_CP2_QUEUE` 는 런 이후 커밋·기본 OFF ⇒ 격리로 먼저 | (나) |
| 8 `PRESCRIPTION` `role="tool"` | A-7 계열(무해·코퍼스 축 좁히기) | (가) |
