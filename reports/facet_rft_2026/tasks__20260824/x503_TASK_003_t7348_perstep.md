# TASK_003 — t7348 halfA per-step 포렌식  (v2 · 2026-08-24 재검증 갱신)

> ⚠**파일명 주석**: 이 문서의 요청 경로는 `tasks__20260824/TASK_003.md` 였다.
> `C:\workspace\.claude\hooks\scaffold_guard.py:129`(§74-b·[[31]] 규칙 ①)가 `reports/` 아래 **신규 .md**
> 를 프로브형(`xNNN_*`) 외에는 exit 2 로 막는다(v2 갱신 시에도 `TASK_003.md` 로 쓰려다 같은 차단을
> 실측했다). 훅을 우회하지 않고 훅이 명시한 규칙(**정본을 갱신하라**)을 따라 이 파일을 갱신한다.
> 정규 명명이 필요하면 **사용자 승인 후** `TASK_003.md` 로 옮기면 된다(내용 동일).

> **v2 변경 요약**: v1 의 두 CONFIRMED(Ⓐ `T2_SOURCE` 오탐 · Ⓑ 재생성 산출 무검문 커밋)를 **전수
> 재검증했고 전부 재현됐다**. 새로 더한 것 = ①**스텝 표**(이 문서의 본체 · trial 별) ②turn-4
> claimprov 피드백이 두 trial 에서 **sha 바이트 동일**(`e594a653fdaa`) — 문구가 아니라 산출이
> 갈렸음의 증명 ③turn-6 초안은 **3장을 아직 유지**했는데 우리 재생성 산출이 **Platinum 1장으로 더
> 좁혔다** ④**직전 런 t7346(2/2) 짝 대조** — 같은 seed 에서 `shell` KB 읽기 유무가 `unsourced` 0↔5 를
> 가른다 ⑤**x500 §E-2 의 재개 지표가 이 결함을 구조적으로 못 센다**는 지적 ⑥줄번호를 **런 sha 기준**
> 으로 재고정.

- 런: `bank_t7348_halfA_20260824` · agent = `Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8` · user-sim = `openai/gpt-5.2` · nt=2
- 성적: **trial0 = 1.0 (seed 626729) · trial1 = 0.0 (seed 373753)** ⇒ **1/2**
- 대조(직전 런·같은 계열) `bank_t7346_halfA_20260822`(sha `ee18d797`): **2/2** ⇒ 003 은 t7348 에서 **−1**
- 대조(기준선·sha 상이) `bank_t7328_halfA_20260819r`: **0/2** ⇒ 더 먼 기준선 대비로는 **+1**
- 코드 인용 기준: 런 파생 커밋 `aed30e20`. `git diff aed30e20 HEAD -- t2_gate_patch.py t2_source.py go_stack.sh` = **공집합**
  ⇒ 아래 줄번호는 **런 그 자체의 줄번호**다([[31]] 줄번호 드리프트 대응).
- 사이드카 `fb_bank_t7348_halfA_20260824.jsonl.gz` **존재**(task_003 행 17건) ⇒ 재생성 **이전 초안**을
  축자로 회수할 수 있다.

---

## §1 채점축 — 먼저 확인 (C583ⓖ)

`sim['reward_info']` 직독 (양 trial 동일 구조):

```
reward_basis      = ['DB']
reward_breakdown  = {'DB': 1.0}  /  {'DB': 0.0}
db_check          = {'db_match': True, 'db_reward': 1.0} / {'db_match': False, 'db_reward': 0.0}
nl_assertions     = None · communicate_checks 없음
action_checks     = n=1 (진단 보조일 뿐 성적 아님 · [[69]])
termination       = user_stop (양 trial)
```

**DB 해시 축**이다. gold 변이 1건(`action_checks[0].action`):

```
apply_for_credit_card{card_type:"Silver Rewards Card", customer_name:"Catherine Wells",
                      annual_income:180000, rho_bank_subscription:true}      requestor=user
```

---

## §2 변이표 — `t2_forensic.mutation_diff` 정본만 (손 비교기 0 · C583ⓐ)

```python
sys.path.insert(0,'.'); import t2_forensic as F
mut = F.mutating_tools(); m = F.mutation_diff(sim, mut)
```

### trial 0 (seed 626729) — reward 1.0
| 칸 | n | 내용 |
|---|---|---|
| MATCHED | 1 | `apply_for_credit_card{card_type:"Silver Rewards Card", customer_name:"Catherine Wells", annual_income:"180000", rho_bank_subscription:true}` @ `msg_i=5` |
| MISSING / WRONGARG / EXTRA / DUP / BLOCKED | 0 | — |

### trial 1 (seed 373753) — reward 0.0
| 칸 | n | 내용 |
|---|---|---|
| MISSING | 1 | `apply_for_credit_card{card_type:**"Silver Rewards Card"**, …}` (aid=`003_0`) |
| WRONGARG | 1 | `apply_for_credit_card{card_type:**"Platinum Rewards Card"**, …}` @ `msg_i=7` · `ok=True` · `deny=""` |
| EXTRA / DUP / BLOCKED / MATCHED | 0 | — |

**WRONGARG 필드별 대조** (4 필드):

| 필드 | 보낸 값 | gold | 일치 |
|---|---|---|---|
| `card_type` | `"Platinum Rewards Card"` | `"Silver Rewards Card"` | ❌ |
| `customer_name` | `"Catherine Wells"` | `"Catherine Wells"` | ✅ |
| `annual_income` | `"180000"` | `180000` | ✅ |
| `rho_bank_subscription` | `true` | `true` | ✅ |

⇒ 실패 단위는 **`card_type` 한 칸**이다. MISSING 과 WRONGARG 는 **같은 한 번의 실행**이다.
`BLOCKED = 0` ⇒ 우리 층이 이 궤적에서 도구 호출을 **반려한 적은 없다**. 우리 층이 한 것은
반려가 아니라 **재생성으로 발화를 갈아 끼운 것**이다(§3).

---

## §3 ★스텝 표 — 결정이 일어난 자리마다 주체를 적는다 (이 문서의 본체)

> `step` = 영속 궤적 `messages` 인덱스. 소수점(예: 4.1)은 **같은 턴 안의 비커밋 개입**
> (재생성 버퍼 · 사이드카에만 남고 `state.messages` 에는 없다).
> `evidence` 는 전부 축자(궤적 / 로그 / 사이드카). `code_path` 는 `our_layer` 스텝에만 적는다.

### §3.1 trial 1 (`task_003#s373753` · 10 msgs · reward 0.0) — 실패

| step | role | what | actor | kind | code_path | evidence (축자) | consequence | reward_causal |
|---|---|---|---|---|---|---|---|---|
| 1 | user | 요구 4종 선공개 + 목록 질문. **연소득을 말하지 않는다**(trial0 은 말한다) | user_sim | 시나리오 분산 | — | `"- No foreign transaction fees / - Purchase protection / - A credit limit that could potentially be at least $100,000 / - My biggest spending category is travel … Also, I have a Rho-Bank+ subscription. What options do you have that fit those requirements?"` | 이후 에이전트가 `annual_income` 을 **날조**(100000)하는 선행 조건. gold 필드는 손님이 직접 180000 을 보내 복구됨 | 약 |
| 2 | assistant | `check_card_application_fit` 를 **정확한 인자로** 호출 | model | formalize 성공 | — | `{"max_fx_fee":"0","needs_purchase_protection":"true","min_credit_limit":"100000","spend_category":"travel","premium_subscriber":"true"}` | 정답 재료를 부르는 데 성공 | 무관(양) |
| 3 | tool | **우리 스캐폴드가 정답 재료를 전부 전달** | our_layer | 재료 전달 성공 | `t2_scaffold_get.py` → 로그 `[T2_SCAFFOLD_GET] check_card_application_fit -> …` | `'Silver Rewards Card', 'facts': {'annual_fee':0.0, 'fx_fee':2.75, 'fx_fee_with_premium':0.0, 'limit_max':100000, 'purchase_protection':True, 'category_rates':{'software':4.0,'travel':4.0}, "rate_for('travel')":'4.0% (other categories: 1.0%)'}` · `'excluded':[{'card':'Bronze Rewards Card','reason':'fx_fee=2.75 violates max_fx_fee=0.0'}, …]` | **정보 결손 0.** Silver = 요구 3종 충족 + 연회비 최저(Gold 와 동률) + travel 4.0%(Gold 2.5% 초과) ⇒ gold 가 유일해 | 무관(양) |
| 4.0 | assistant(초안#1·비커밋) | 카드 **3장을 붕괴 없이 전부 나열**. Silver 의 프리미엄 fx 감면까지 명시 | model | 옳은 발화 | — | 사이드카 `turn=4 channel=unified_regen kind=reminder-assistant len=1050`: `"3. **Silver Rewards Card** - Annual Fee: $0 - Foreign Transaction Fee: 2.75% (reduced to 0% with Rho-Bank+ subscription) - Cashback: 1% on all purchases, 4% on travel and software categories - Maximum Credit Limit: $100,000 - Purchase Protection: Yes … Would you like more information on any of these cards, or would you like to proceed with applying for one of them?"` | **이 초안이 서 있었으면 trial0 과 같은 종류의 발화**다. 여기까지 잃은 것은 없다 | 무관(양) |
| 4.1 | our_layer | `T2_SOURCE` 가 초안#1 의 수치 **5건을 전부 무근거로 판정 — 5/5 오탐** | our_layer | 계기 오판 → 재생성 유발 | `t2_source.py:265 _anchors()` · `:271 unsourced_claims()` · `:82/:93 build_corpus()` | 로그 `[T2_SOURCE] claims=5 unsourced=5` → 사이드카 `turn=4 kind=reminder-user`: `"Error: [SOURCE] you stated 5 thing(s) as fact that the policy documents decide, without having the document: \"Maximum Credit Limit: $150,000\"; \"Minimum Payment Percentage: 2.5%\"; \"Maximum Credit Limit: $112,500\"; \"Minimum Payment Percentage: 2%\"; \"Maximum Credit Limit: $100,000\""` | 다섯 수치는 **step 3 의 우리 자신의 도구 출력에 축자로 있다**(`limit_max:150000`·`min_payment_pct:2.5`·`limit_max:112500`·`min_payment_pct:2.0`·`limit_max:100000`). 오프라인 재현으로 확정(§6-Ⓐ). **통합 재생성을 연다** | 중 |
| 4.2 | our_layer | `[ACTION]` 소유권문 동시 부착 | our_layer | 표면화 | `t2_gate_patch.py:10579-10846`(`fb=[am]` → `record_many(fb, …, channel="unified_regen")`) | 같은 사이드카 행(1254자): `"Error: [ACTION] 'apply_for_credit_card' is run by the CUSTOMER, not by you. …"` | trial0 과 **동일 발화** ⇒ 갈림의 원인 아님 | 약 |
| 4.3 | our_layer | 검색 팔이 결정 지점에 **우리 자신이 제외한 카드**를 답으로 배달 | our_layer | 문구 모순([[55]]) | `t2_search.py` → `[T2_DOCDECIDE]` · `[T2_SEARCH_ON_PROCEED]` | 로그 `[T2_SEARCH_AGENT] group=business_credit_cards · 문서 82 … turn=4` → `[T2_DOCDECIDE] → 'Business Bronze Rewards Card'` → `[T2_SEARCH_ON_PROCEED] deny 아님 · 재료 263자 배달` · 사이드카 `turn=3 kind=cp2`: `"A separate check was run on the policy documents on record … It answers: Business Bronze Rewards Card."` | step 3 이 `Bronze Rewards Card` 를 *`fx_fee=2.75 violates max_fx_fee=0.0`* 로 **제외**해 놓고 결정점엔 Bronze 를 민다. **trial0 도 축자 동일 배달을 받고 1.0** ⇒ 인과 아님·별건 기록 | 무관 |
| 4.4 | our_layer | `T2_CLAIMPROV` 발화 → `T2_GATE_REGEN(claimprov)` **재생성 개시** | our_layer | 재생성 개시 | `t2_gate_patch.py:12600` `_new2 = _ap_regen("\n".join(_parts), "claimprov")` | 로그 `[T2_CLAIMPROV] window hit(resign) claims=0 unbacked=0 pending=1 unb_p=1 ['give']` → `[T2_CLAIMPROV] owner split: agent=1 user=0 unknown=0` → `[T2_LEVER] T2_GATE_REGEN … claimprov` · 사이드카 `turn=4 channel=claimprov len=275 **sha=e594a653fdaa**`: `"Note: [CLAIM-PROVENANCE] tool ownership — the following are in YOUR OWN tool list, not the customer's: provide apply_for_credit_card tool (tool: give_discoverable_user_tool). …"` | **trial0 의 같은 자리 발화와 sha 바이트 동일** ⇒ 문구는 갈림의 원인이 아니고, **연 것**이 우리 층이다 | 강 |
| 4.5 | model | 재생성이 **정지에 실패**한다 — `give_discoverable_user_tool` 봉투 111개 반복, 길이 상한 소진 | model | 정지 실패(degeneration) | — | `raw_data.choices[0].finish_reason="length"` · `completion_tokens=8192` · `prompt_tokens=12114` · `content` 42,962자 · `<tool_call>` 여는 태그 **111**(닫힘 110) · `Platinum 37 / Gold 37 / Silver 36` · tail `…"rho_bank_subscription": "true"}"}}</tool_call>\n<tool_call>   \n{"name": "give_discoverable_user_tool", "arguments`(미종결) | vLLM hermes 파서가 all-or-nothing 이라 유효 봉투 110개가 통째로 폐기 → `tool_calls=null`, 43k 가 `content` 로 | 강 |
| 4.6 | our_layer | **폭주 산출을 검사 없이 채택·커밋** — `ENVGUARD`/`TRUNCGUARD` 가 이 산출을 **보지 않는다** | our_layer | 배치 사각(guard bypass) | 가드 `t2_gate_patch.py:11762`(ENVGUARD)·`:11783`(TRUNCGUARD) / 배치 주석 `:11738` *"모든 의미-게이트보다 **먼저** 돈다"* / 채택 `:12601-12603` `if _new2 is not None: am = _new2` / `_ap_regen` 채택검사 `:11409-11694` 의 `_denied_calls(:11511)`·`_proc_first_deny(:11554)`·`T2_UNLOCK_*(:11616)` 는 **전부 `tool_calls` 순회** / 플래그 `go_stack.sh:145` | 로그 `[T2_CLAIMPROV] regen tool_calls=[]` (= `_new2 is not None` 분기 **안에서만** 인쇄 ⇒ 채택됨) · `[T2_ENVGUARD]`·`[T2_TRUNCGUARD]` 는 halfA **11,593 줄 전체에서 0건** | `tool_calls == []` 이면 세 검사가 전부 **공(空)통과**한다. 옳았던 초안#1(4.0)이 **43k 쓰레기로 대체돼 영속**됐다. duration 137s → **1077s**(7.8×) | 강 |
| 5 | user | **페르소나 이탈** — 손님이 상담원 답변을 대신 쓰고 KB 와 **반대되는** 단정을 한다 | user_sim | 허위 주장([[21]]·종결 카테고리 아님) | — | `"Thanks, Catherine—based on your hard requirements … ### 3) Silver Rewards Card - Credit limits: **Lowest tier**, **not** a fit if you need the *possibility* of $100,000+ … If the $100,000+ potential credit line is truly a hard requirement, **the Platinum Rewards Card is the only option that fits**."` | KB 반증: step 3 이 `Silver … limit_max: 100000`, 요구는 *"at least $100,000"* ⇒ **충족한다**. 카드 선택을 Platinum 으로 고정 | 강 |
| 6.0 | assistant(초안#1·비커밋) | 모델이 **손님의 허위 단정에 정박** — Platinum 추천으로 붕괴(3장 나열은 아직 유지) + `annual_income` 100000 **날조** | model | 자기 도구 출력 미참조 | — | 사이드카 `turn=6 channel=unified_regen kind=reminder-assistant len=1382`: `"Based on your requirements, the **Platinum Rewards Card** seems to be the best fit since it's the only card that offers the possibility of a $100,000+ credit line. … - **Credit Limits:** Lowest tier; not suitable for a $100,000+ limit"`(Silver) + `{"card_type": "Platinum Rewards Card", … "annual_income": "100000"}` | **`card_type` 가 여기서 뒤집힌다.** step 3 은 여전히 문맥에 있다(`prompt_tokens=20770`) — 다시 보지 않았다 | 강 |
| 6.1 | our_layer | `T2_SOURCE` 가 이번엔 **옳게** 잡는다(3/3 진성) — 그러나 처방이 엉뚱한 곳을 가리킨다 | our_layer | 처방 오지정([[64]] 계열) | `t2_source.py:271` + A2 `arbitration.unsourced_claim_feedback` | 로그 `[T2_SOURCE] claims=3 unsourced=3` · 사이드카 `turn=6 kind=reminder-user`: `"Error: [SOURCE] … \"Credit Limits: Highest tier; possibility of a $100,000+ limit (subject to underwriting)\"; \"Credit Limits: Typically below $100,000\"; \"Credit Limits: Lowest tier; not suitable for a $100,000+ limit\" … **Search the knowledge base** for the document that states it, quote the figure it gives …"` | 그 세 문장은 **step 3 의 우리 도구 출력이 이미 반증**한다(`limit_max` 150000/112500/100000). 문면은 *"KB 를 검색하라"* 라고만 해 **손 안의 반증을 못 가리켰다** ⇒ 구조 회수 실패 | 중 |
| 6.2 | our_layer | 검색 팔이 다시 제외 카드를 답으로 배달 | our_layer | 문구 모순 | `t2_search.py` → `[T2_DOCDECIDE]` | 로그 `[T2_SEARCH_AGENT] group=credit_cards · 문서 110 … turn=6` → `[T2_DOCDECIDE] → 'Bronze Rewards Card'` → `재료 254자 배달` · 사이드카 `turn=5 kind=cp2`: `"It answers: Bronze Rewards Card."` | 모델은 Bronze 를 고르지 않았다 ⇒ 인과 아님 | 약 |
| 6.3 | our_layer | claimprov 재생성이 **후보를 3장 → Platinum 1장으로 더 좁힌다** | our_layer | 후보 붕괴(재생성 산출) | `t2_gate_patch.py:12600-12603` | 반려문(사이드카 `turn=6 channel=claimprov len=709`): `"Error: [CLAIM-PROVENANCE] your reply asserts you already did the following, but the conversation ledger shows NO such event: give: provided apply_for_credit_card tool. …"` → 로그 `[T2_CLAIMPROV] regen tool_calls=[]` | 초안(6.0)은 **1382자·3장**이었는데 영속된 msg[6]은 **698자·Platinum 만**이다 ⇒ 손님이 마지막으로 본 화면에서 **Silver 가 사라졌다** | 중 |
| 6 | assistant | 영속본: 사과 + Platinum 단일 추천 + 신청 JSON 떠먹임 | model+our_layer | — | — | 궤적 msg[6] `tool_calls=null` len=698: `"I apologize for the oversight. Based on your requirements, the **Platinum Rewards Card** is the best fit since it offers the possibility of a higher credit limit … \"card_type\": \"Platinum Rewards Card\""` | 손님의 실행 인자를 확정 | 강 |
| 7 | user | `apply_for_credit_card{card_type:"Platinum Rewards Card", …, annual_income:180000}` 실행 | user_sim | 실행 | — | 궤적 msg[7] tool_call | **WRONGARG 성립 · gold Silver 행 MISSING** | 강 |
| 8 | tool | `Credit card application submitted: …` | env | — | — | 궤적 msg[8] | DB 상태가 gold 와 갈림 | 강 |
| 9 | user | `###STOP###` | user_sim | 종료 | — | `termination_reason=user_stop` | `db_match=False` → reward 0.0 | 강 |

### §3.2 trial 0 (`task_003#s626729` · 8 msgs · reward 1.0) — 성공

| step | role | what | actor | kind | code_path | evidence (축자) | consequence | reward_causal |
|---|---|---|---|---|---|---|---|---|
| 1 | user | 같은 요구 + **연소득 $180,000 명시** + 연회비/캐시백까지 질문 | user_sim | 시나리오 분산 | — | `"Context: I make about $180,000 annually and I do have a Rho-Bank+ subscription. … What card options do you have that meet those requirements, and what are the annual fees/cash back on them?"` | 이후 템플릿의 `annual_income` 이 **180000** 으로 정확히 채워진다 | 약(양) |
| 2.0 | assistant(초안·비커밋) | 도구 호출에 `max_min_payment_pct=None` 을 넣음 | model | 날조 인자 | — | 사이드카 `turn=2 channel=unified_regen kind=tool-deny len=435` | 우리 PROV 게이트가 잡는다 | 무관 |
| 2.1 | our_layer | `T2_MATERIAL_GATE` → `T2_PROV` 재생성 | our_layer | 인자 접지 | `t2_gate_patch.py` PROV 경로(`apply_provenance_regen`) | 로그 `[T2_MATERIAL_GATE] stop=other_lever(prov) turn=2 calls=check_card_application_fit pending=0 axes=9 prose=False` · `[T2_PROV] regen fired tool=check_card_application_fit arg=max_min_payment_pct val=None` · 사이드카 `"Error: [PROVENANCE] argument 'max_min_payment_pct'='None' was not provided by the user nor returned by any tool — it looks invented …"` | 인자가 12개로 채워졌다. **step 3 의 `eligible` 표는 두 trial 이 바이트 동일** ⇒ 인과 아님 | 무관 |
| 3 | tool | 동일한 `eligible` 표(3071자·바이트 동일) | our_layer | 재료 전달 성공 | `t2_scaffold_get.py` | (trial1 step 3 과 축자 동일) | 정보 결손 0 | 무관(양) |
| 4.0 | assistant(초안#1·비커밋) | 3장 중립 나열 | model | 옳은 발화 | — | 사이드카 `turn=4 channel=unified_regen kind=reminder-assistant len=786`: `"3. **Silver Rewards Card** - **Annual Fee:** $0 (with Rho-Bank+ subscription) - **Cash Back:** 4% on travel and software purchases - **Purchase Protection:** Yes - **Credit Limit:** Up to $100,000 … Would you like more information on any of these cards, or would you like to proceed with applying for one?"` | trial1 step 4.0 과 **같은 종류** | 무관(양) |
| 4.1 | our_layer | `T2_SOURCE` **3/3 오탐**(같은 기전) → 통합 재생성 | our_layer | 계기 오판 | `t2_source.py:265/:271/:82` | 로그 `[T2_SOURCE] claims=3 unsourced=3` · 사이드카: `"…without having the document: \"Credit Limit: Up to $150,000\"; \"Credit Limit: Up to $112,500\"; \"Credit Limit: Up to $100,000\""` | trial1 과 **동형 개입인데도 통과**했다 ⇒ 이 오탐 단독으로는 실패를 못 만든다 | 중 |
| 4.2 | our_layer | 같은 `[ACTION]` 소유권문(1178자) · 같은 cp2 `Business Bronze Rewards Card`(263자) 배달 | our_layer | 표면화 / 문구 모순 | `t2_gate_patch.py:10579-10846` · `t2_search.py` | 사이드카 `turn=4 kind=reminder-user` · `turn=3 kind=cp2` | (trial1 4.2·4.3 과 동일) 갈림 아님 | 무관 |
| 4.3 | our_layer→model | `T2_CLAIMPROV` → `GATE_REGEN(claimprov)` · **재생성이 정상 호출 1건을 낸다** | our_layer(개시)·model(산출) | ★**분기점** | `t2_gate_patch.py:12600-12603` | 사이드카 `turn=4 channel=claimprov len=275 **sha=e594a653fdaa**`(trial1 과 바이트 동일) · 로그 `[T2_CLAIMPROV] regen tool_calls=**['give_discoverable_user_tool']**` | **같은 sha·같은 A2·같은 도구 출력·같은 우리 문구**에서 한쪽은 호출 1건, 다른 쪽은 봉투 111개 후 절단. 여기가 갈린 자리다 | 강 |
| 4.4 | our_layer | 2차 게이트 `T2_USER_TOOL_NOTE` → `GATE_REGEN(usertoolnote)` | our_layer | 표면화 | `t2_gate_patch.py` usertoolnote 경로 | 로그 `[T2_USER_TOOL_NOTE] pre-give note: apply_for_credit_card` → `[T2_LEVER] T2_GATE_REGEN … usertoolnote` · 사이드카 `turn=4 channel=usertoolnote len=338`: `"Note: \`apply_for_credit_card\` runs **right here in this conversation** … tell them the arguments to pass."` | 영속 msg[4] 는 `tool_calls=null` 이 됐지만(give 호출은 사라졌다) 본문이 *"you can do so right here in this conversation by calling the \`apply_for_credit_card\` tool"* 로 바뀌어 손님이 즉시 실행 가능 | 중(양) |
| 4 | assistant | 영속본: **3장 나열 + 선택 미지정 템플릿** | model+our_layer | 옳은 최종 발화 | — | 궤적 msg[4] len=1411: `{"card_type": "<Platinum Rewards Card/Gold Rewards Card/Silver Rewards Card>", "customer_name": "Catherine Wells", "annual_income": "180000", "rho_bank_subscription": "true"}` + `"Please replace \`<…>\` with the card type you prefer."` | 카드 선택을 **손님에게 남겼다** ⇒ 결정권 이양 성공 | 강(양) |
| 5 | user | `apply_for_credit_card{card_type:"Silver Rewards Card", …}` | user_sim | 실행 | — | 궤적 msg[5] tool_call | **MATCHED** | 강(양) |
| 6-7 | tool / user | 제출 성공 → `###STOP###` | env / user_sim | — | — | — | `db_match=True` → 1.0 | 강(양) |

### §3.3 분기점 요약

| | trial 0 (1.0) | trial 1 (0.0) |
|---|---|---|
| msg[1] 손님 발화 | 연소득 180,000 **명시** · 연회비/캐시백 질문 | 연소득 **미언급** |
| step 3 도구 출력 | 3071자 | **바이트 동일** |
| turn-4 초안#1 | 3장 중립 나열 | 3장 중립 나열 (**동등**) |
| `[T2_SOURCE]` | claims=3 unsourced=3 (오탐) | claims=5 unsourced=5 (오탐) |
| claimprov 피드백 sha | `e594a653fdaa` | **`e594a653fdaa` (동일)** |
| **`[T2_CLAIMPROV] regen tool_calls=`** | **`['give_discoverable_user_tool']`** | **`[]`** ← 폭주·절단 |
| 2차 게이트 | `T2_USER_TOOL_NOTE` → regen | (없음 — 쓰레기라 후속 게이트가 물 것이 없다) |
| 영속 msg[4] | 3장 + 선택 미지정 템플릿 (1411자·`finish_reason` 정상) | **43k `<tool_call>` 폭주** (42,962자·`finish_reason=length`) |
| 다음 손님 수 | `apply{Silver}` → **1.0** | 페르소나 이탈 산문 → 사과 → `apply{Platinum}` → **0.0** |
| duration | 137s | **1077s** |

**분기점 = turn 4 의 claimprov 재생성 산출 한 줄.**

---

## §4 레버 발화표 (이 두 sim 의 로그 줄만 · halfA 전체 대조)

| 레버 | trial1 | trial0 | halfA 전체 | 판정 |
|---|---|---|---|---|
| `T2_SG_DOCS` | 0 | 0 | 4 | **미발화**(이 태스크에 apy 계열 도구 없음 — 정상) |
| `T2_PIN_READ` | 0 | 0 | 4 | **미발화** |
| `T2_DEMANDED_STEP` | 0 | 0 | 18 | **미발화** |
| `T2_CLAIMPROV` | **7** | 3 | 555 | **발화 → 재생성 유발 → trial1 에서 역효과**(§3.1 4.4·4.6·6.3) |
| `T2_FOLLOWUP` | 0 | 0 | 5 | **미발화** |
| `T2_SEARCH_AGENT` | **4** | 2 | 135 | **발화·침묵 아님 — 그러나 오답 배달**(우리가 제외한 Bronze) |
| `FAB_STRIP` | 0 | 0 | **0** | 런 전체 미발화 |
| `T2_ARG_PRODUCERS`(F8) | 0 | 0 | **0** | 런 전체 미발화 — 런 머리말이 표적으로 건 축인데 halfA **0** |
| READ-FIRST | 0 | 0 | **0** | 런 전체 미발화 |
| `T2_REQUIRE_DOC_DELIVER` | 0 | 0 | 32 | **미발화** |
| `T2_SEARCH_REARM` | 0 | 0 | 18 | **미발화** |
| **`T2_ENVGUARD`** | **0** | 0 | **0** | ⛔**부작위 오발화** — 플래그 `go_stack.sh:145` **ON** 인데 자기 표적을 통과시켰다(§6-Ⓑ) |
| **`T2_TRUNCGUARD`** | **0** | 0 | **0** | ⛔**부작위 오발화** — 동상 |
| `T2_PROCEDURE` / `regen-*`(A-1) | 0 | 0 | **0** | 이 궤적 **미개입** |
| `T2_ARG_EMPTY`(R5) | 0 | 0 | **0** | 이 궤적 **미개입** |
| `T2_WRITE_ARG_ENUM`(R4) | 0 | 0 | 6 | 이 궤적 **미개입** |
| `T2_STALE_NOTE` | 0 | 0 | 0 | 미발화 |

그 밖에 이 궤적에서 실제로 발화한 우리 층: `T2_GUIDED` · `T2_A2_VARIANT`(ratefix·ledger) ·
`T2_SG_TRACE` · `T2_ARG_DOC_SUB`(`spend_category 'travel'→'travel'`) · `T2_SCAFFOLD_GET`(11) ·
`T2_WRITE_SUB`(제안 1 → **근거검산 통과 0**) · `T2_WINDOW` · `T2_ACTIONREQ` · `T2_SOURCE` ·
`T2_ARBITRATE` · `T2_RESOLVE`(`user-action instruct`) · `T2_STACK` · `T2_DOCGROUP` · `T2_GROUPORDER` ·
`T2_NOW_SELFCALL` · `T2_SEARCH_ON_PROCEED` · `T2_WRITEPROV` · `T2_SELFDECL` ·
`T2_DECISION_CARRY`(263자·trial1만) · `T2_EPLAN_WALK`(`walk gap surfaced only (no hold)`·trial1만) ·
`T2_MATERIAL_GATE`/`T2_PROV`/`T2_USER_TOOL_NOTE`/`T2_LEDGER`(trial0만).

### 직전 런 이후 들어간 수리 묶음은 이 궤적에 개입했는가 — **0 건**

t7346(`ee18d797`) → t7348(`aed30e20`) 사이의 9 커밋(R1–R8 · proc-regen recheck)이 노린 레버
(`T2_PROCEDURE`/`regen-*` · `T2_ARG_EMPTY` · `T2_WRITE_ARG_ENUM` · ATM 비교기)는 **이 두 sim 의 로그에
한 줄도 없다**. 003 의 −1 은 수리 묶음의 효과가 아니라 **기존 스택 + 시행 분산**이다.

**t7346 ↔ t7348 로그 대조(같은 seed 쌍)에서 확인한 것**:

- t7346 은 **양 trial 모두** 마지막 결정 턴에서 `[T2_CLAIMPROV] regen tool_calls=[]` 를 찍고도
  **1.0/1.0** 이다 ⇒ *"claimprov 재생성 = 곧 실패"* 가 아니다. 갈리는 것은 **그 재생성의 산출**이다.
- t7346 `s373753` 은 msg[4] 에서 **`shell` 로 KB 를 직접 읽었고**
  (`[T2_RESOLVE_CAP] 리셋(실행): 새 실행 ['shell']`) 그 직후 `[T2_SOURCE] claims=3 **unsourced=0**` 이 됐다.
  t7348 `s373753` 에는 `shell` 호출이 **없다** ⇒ §6-Ⓐ 의 오탐은 *"모델이 KB 를 읽으면"* 소멸하는
  종류이고, **읽지 않은 시행에서만 재생성을 연다**.
- `unb_p` 산정 코드는 **바뀌지 않았다**(`git diff ee18d797 aed30e20 -- t2_gate_patch.py` 에
  `unb_p`/`owner split`/`kind-index`/`tool-miss` 관련 변경 **0줄**). t7346 turn-4 의 `unb_p=0` ↔
  t7348 turn-4 의 `unb_p=1 ['give']` 차이는 **초안 문면이 달라서** 생긴 데이터 차이다.

---

## §5 선행 판정과의 대조 — 같은 원인인가, 달라졌는가

| 문서 | 003 에 대한 선행 판정 | t7348 과의 관계 |
|---|---|---|
| **본 문서 v1** (동일 파일) | Ⓐ `T2_SOURCE` 오탐 · Ⓑ 재생성 산출 무검문 커밋 을 CONFIRMED | **같은 원인이다.** 변이표·오프라인 `unsourced_claims` 5/5·가드 0건·플래그 ON 을 **전수 재현**했다. v2 가 더한 것은 v2 머리말 ①~⑥ |
| `N97_TASK_ROOT_CAUSE_2026_08_06.md` §142 | *"결정권은 손님에게 있는데 **적격 목록을 손님에게 주지 않았다**"* · 후보 D(결정 근거 집합 전달) | **해소됐다.** t7348 양 trial 모두 turn-4 **원본 초안**이 3장을 전부 나열(사이드카 축자). **전달 결손 아님** |
| `FAILURE_AXIS_AND_FIX_ORDER_2026_08_15.md` §13 | "후보 붕괴" 축 철회 · 003 = **원인 미상** | 더는 미상이 아니다(§6) |
| `T7335_NT1_FORENSIC_HALFA_2026_08_21.md` §22-45 | **모델.** *"[6] 실패 지점: 3장 목록 대신 단일 추천 + 신청 인자 JSON 떠먹임"* · *"해당 결정 지점 레버 부재"* | **원인이 바뀌었다.** 관측(WRONGARG `card_type` Platinum↔Silver)은 같지만 기전이 다르다 — t7348 trial1 에서 **에이전트는 turn-4 에 붕괴시키지 않았다**(초안이 3장 나열). 붕괴는 **user-sim 이 step 5 에서** 했고, 그 자리를 연 것은 **우리 재생성의 폭주 커밋**이다 |
| `CP2_STAGE0_PATCH_PLAN_2026_08_23.md` §2-2·§6-1 | *"★가장 위험 — **통과 중인 6 sim 의 마지막 발화**"* · 표에 `task_003 s626729 / s373753 … 1.0 / 1.0` · 등급 **[?]** | **위험이 그 두 sim 중 하나에서 실현됐다.** 다만 채널이 다르다 — 예고한 cp2 부착이 아니라 **claimprov 재생성**이 갈아 끼웠다 ⇒ 예측의 *부호*는 맞고 *채널*은 틀렸다. **[M]** |
| `FAILURE_MASTER__20260822.md` §2.4-2 | *"`user-action instruct`: 074#1 음 ↔ **033·100·003·024 통과 sim 에서도 동일 발화**"* | **유지된다.** t7348 003 에서도 `[T2_RESOLVE] user-action instruct target=apply_for_credit_card` 가 **통과·실패 양 trial 에** 나온다(4회/2회) ⇒ 이 레버는 003 실패의 인과가 아니다 |
| `x500_axis_e_target_2026_08_24.json` §E-2 | ENVGUARD/TRUNCGUARD 축을 **닫았다**: *"t7346 40 sim 중 envguard 1·truncguard 1 … 표적 = 호출 1건이고 그 1건은 체인의 표적이 아니다. **짓지 않는다**"* · 재개 조건 = *"ENVGUARD/TRUNCGUARD 발화가 한 런에서 5건 이상"* | ⚠**결함의 방향이 다르다.** x500 이 닫은 것은 *"**가드의** 재생성 산출이 뒤 체인을 우회한다"* 이고, 여기서 나온 것은 그 **역** — *"**뒤 재생성(claimprov)의** 산출이 가드를 우회한다"*. 그리고 x500 의 **재개 지표가 이 현상을 구조적으로 못 센다**: t7348 halfA 는 `[T2_ENVGUARD]` **0건**인데 폭주 응답이 **커밋됐다**(가드가 볼 수 없는 자리에서 났으므로 발화 카운터에 안 잡힌다). x500 의 결론은 **그 축**에 대해선 유효하나, 이 결함은 **그 축의 미측정 사각**이다 |
| `x501_x431_arms_2026_08_24.json` | `x431_spec_selects.json` 코퍼스에 **003×2** 가 있었으나 스냅샷 차이로 n=5 표본에서 빠졌다 | 이 실패는 **스펙 선택(메뉴 축) 문제가 아니다** — step 2 formalize 는 정확했고 step 3 이 gold 를 유일해로 만드는 표를 냈다. x431 축과 무관 |
| `x502_conditional_cell_audit_2026_08_24.json` · `STATE_OF_PLAY_2026_08_23.md` · `tasks__20260822/` · `t7336_tasks/` | 003 언급 **0건**(003 은 두 디렉터리의 실패 목록에 없다 — 그 런들에서 통과했다) | 대조할 판정 없음 |
| 기준선 `bank_t7328_halfA_20260819r` **0/2** | trial0 = 3장 다 나열했는데 user-sim 이 **Gold** 선택 · trial1 = **에이전트가 단일 추천으로 붕괴** | *"3장만 나열하면 통과"* 는 **반증됐다.** 003 은 **에이전트 측 붕괴 · user-sim 측 선호 오적용** 두 실패원이 번갈아 나오는 태스크다 |

---

## §6 원인 확정

### 인과 사슬 (trial 1)

```
step 3   our_layer  도구가 정답 재료 전달(Silver limit_max=100000·fee 0·travel 4%)        결손 0
step 4.0 model      3장 중립 나열 = 옳은 발화                                              결손 0
  ↓ our_layer  4.1 T2_SOURCE 5/5 오탐 + 4.4 CLAIMPROV → GATE_REGEN(claimprov)   [문구는 trial0 과 sha 동일]
  ↓ model      4.5 재생성이 give_discoverable_user_tool 봉투 111개 반복 → finish_reason=length
  ↓ our_layer  4.6 ENVGUARD·TRUNCGUARD 가 **재생성 산출을 보지 않는다** → 43k 쓰레기 커밋
step 5   user_sim   페르소나 이탈 + "Silver not a fit / Platinum is the only option"        KB 반증
  ↓ model      6.0 자기 도구 출력 대신 손님 단정에 정박 → card_type 뒤집힘
  ↓ our_layer  6.1 SOURCE 는 옳게 잡았으나 처방이 "KB 를 검색하라" — 손 안의 반증을 못 가리킴
  ↓ our_layer  6.3 claimprov 재생성이 후보를 3장 → Platinum 1장으로 더 좁힘
step 7   user_sim   apply{Platinum} → WRONGARG · gold Silver MISSING → db_match=False
```

### 귀속

- **cause_primary = `our_layer`.** 반사실이 궤적 **안에** 있다: turn 4 에 우리가 개입하지 않았다면
  사이드카에 보존된 **초안#1**(4.0)이 그대로 섰고, 그것은 trial0 이 1.0 을 받은 발화와 같은 종류다.
  우리 개입이 **통과 중이던 발화를 소각**했고, 그 소각을 막으라고 만든 두 가드가 **자기 표적을
  통과시켰다**. 코드 경로 = Ⓐ·Ⓑ.
- **cause_secondary = `model` + `user_sim`.**
  · `model` — 정지 실패(봉투 111개)는 우리가 만든 것이 아니다. **같은 sha 의 같은 문구**를 받은
    trial0 은 호출 1건을 정상 산출했다. 6.0 에서 문맥에 살아 있는 자기 도구 출력을 재확인하지 않고
    손님 단정을 받아쓴 것도 모델이다.
  · `user_sim` — 자기 지시문의 *"Never respond as a customer service representative/assistant"* 를
    어기고 상담원 답변을 대신 썼으며 내용이 KB 와 반대였다. **종결 카테고리로 쓰지 않는다**([[21]]).

### Ⓐ CONFIRMED our_layer — `T2_SOURCE` 가 **우리 자신의 도구 출력**을 무근거로 판정한다

**코드 경로**: `scripts/distill/tau2/t2_source.py:265 _anchors()` · `:271 unsourced_claims()` ·
`:82/:93 build_corpus()`.

**오프라인 재현**(이 sim 의 msg[0..3] 을 그대로 넣고 정본 함수 호출 · v2 에서 재실행):

```
ledger chars=3071  lines=1  docs=set()          ← 문서 채널이 비어 있다
unsourced 5 of 5                                 ← 라이브 [T2_SOURCE] claims=5 unsourced=5 와 일치
_anchors("Maximum Credit Limit: $150,000") = {'limit','maximum','credit'}
ledger anchors 표본 = ['annual_fee','base_cashback','card','cashback_scope','category_rates',
                       'limit_max','min_payment_pct','purchase_protection', …]
intersection = set()      need = 2               ← 여기서 떨어진다
'150000' in ledger figures = True                ← 수치는 원장에 있다
```

**기전**: 다섯 수치는 **전부 원장에 있다**. `_anchors` 의 토큰 정규식 `[A-Za-z][A-Za-z_\-]{3,}` 이
`_` 를 토큰 **안에** 남기므로 `limit_max` 는 `limit` 과 절대 만나지 않는다. 우리 스캐폴드 도구는
산문이 아니라 **snake_case 키를 가진 구조체**를 뱉는데, 근거 판정기는 **산문 단어 공기(共起)** 로
검사한다 ⇒ **우리 도구 출력을 자연어로 옮긴 주장은 원리상 통과할 수 없다.**
문서 채널도 막혀 있다: `build_corpus:93` 이 `\bdoc_[A-Za-z0-9_()\-]+` 로 doc id 를 걷는데 우리 도구가
실어 보낸 출처 문자열은 `'source': 'doc platinum_rewards_card_*'`(공백 + glob)라 **0건**이다
(`docs=set()` 실측).

[[25]] 위반형이다 — 우리 도구가 유일한 근거원인데 우리 검증기가 그 출력을 날조로 표시하고, 그
표시가 재생성을 연다.

⚠**단, 이 결함 단독으로는 실패를 못 만든다.** trial0 도 같은 오탐(3/3)을 받고 1.0 이다.
Ⓐ 는 *재생성을 여는 조건*이지 *실패의 충분조건*이 아니다.
⚠halfA 전체 오탐률은 **인용 금지** — 003 의 5+3 건만 검산했다.

### Ⓑ CONFIRMED our_layer — **재생성 산출이 폭주·절단 가드를 통과하지 않는다**

**코드 경로**

| 자리 | 줄 | 내용 |
|---|---|---|
| 플래그 | `go_stack.sh:145` | `export T2_ENVELOPE_GUARD=1 T2_ENVELOPE_CAP=2 T2_TRUNC_GUARD=1 T2_UNAVAIL_PROMISE=1` ⇒ **켜져 있다** |
| ENVGUARD | `t2_gate_patch.py:11762` | `not am.tool_calls and _envtag in am.content` |
| TRUNCGUARD | `t2_gate_patch.py:11783` | `raw_data.choices[0].finish_reason == "length"` |
| 배치 주석 | `t2_gate_patch.py:11738` | 축자 *"**모든 의미-게이트보다 먼저** 돈다"* ⇒ 두 가드는 **게이트 이전의 `am` 만** 본다 |
| claimprov 채택 | `t2_gate_patch.py:12600-12603` | `_new2 = _ap_regen(…, "claimprov")` → `if _new2 is not None: am = _new2` — **가드 뒤** |
| `_ap_regen` 채택검사 | `t2_gate_patch.py:11409-11694` | `_denied_calls(:11511)` · `_proc_first_deny(:11554)` · `T2_UNLOCK_NAME/PROV(:11616)` 이 **전부 `tool_calls` 를 순회**한다. `tool_calls == []` 이면 셋 다 **공통과**하고 `_am2` 가 그대로 반환·커밋된다. `finish_reason` 도 `<tool_call>` 봉투도 이 함수는 **한 번도 보지 않는다** |

두 가드와 claimprov 채택은 **같은 함수 `unified()`(`t2_gate_patch.py:7115`) 본문의 직선 코드**이고
순서가 `11762 → 11783 → 12600` 이다. 즉 가드는 claimprov 산출을 **원리적으로 못 본다**.

**자백이 같은 함수 안에 있다** — `t2_gate_patch.py:11495` 주석 축자:
> *"근거: **폭주 응답(33k자·8k토큰)** 을 그대로 실으면 regen 호출 자체가 창 초과로 죽고…"*

이 함수는 폭주를 **재생성의 입력**으로만 다루고(`_trunc_for_prompt` · `am_override`),
**재생성의 출력**으로는 다루지 않는다.

**실측**: trial1 msg[4] = `finish_reason="length"` · 8192 토큰 · 42,962자 · `<tool_call>` 여는
태그 111개(닫힘 110) · `tool_calls=null`. `[T2_ENVGUARD]`·`[T2_TRUNCGUARD]` 로그 라인은 halfA
**11,593 줄에서 0건** · 캡(`_t2_envguard`/`_t2_truncguard`)은 **소모조차 안 됐다**.
死배선이 아니라 **배치 사각**이다.

**폭발 반경(이 런 한정)**: halfA 20 sim 에서 커밋된 폭주/절단 어시스턴트 턴 = **1건**
(`task_003` trial1 msg[4]). **n=1** — 이 한 건으로 레버 효과를 주장하지 않는다.
비용은 성적 밖에도 있다: 같은 sim 의 duration 이 137s → **1077s**(7.8×).

### Ⓒ UNPROVEN — 우리 층으로 **주장하지 않는다**

- **검색 팔의 오답 배달.** `T2_SEARCH_AGENT`/cp2 가 **우리 자신이 제외한** `Bronze Rewards Card` /
  `Business Bronze Rewards Card` 를 결정 지점에 답으로 밀었다(로그·사이드카 축자 확인). 실재하는
  우리 층 모순이지만 **이 실패의 인과는 아니다** — trial0 도 축자 동일 배달을 받고 1.0 이고 모델은
  Bronze 를 고르지 않았다. **별건 기록.**
- **CLAIMPROV 의 소유권 지적이 폭주를 유발했다**는 인과는 못 세운다. 피드백 sha 가 두 trial 에서
  **바이트 동일**한데 산출만 갈렸다. 세울 수 있는 것은 *"재생성을 열었다"*(4.4)와 *"산출을 검사
  없이 커밋했다"*(4.6) 뿐이다.
- **turn-6 재생성의 후보 축소**(3장 → Platinum 1장 · 6.3)는 실재하지만, 그 시점엔 user-sim 이 이미
  Platinum 으로 결심한 뒤라 **반사실이 안 선다**. 기록만 남긴다.
- **`T2_SOURCE` 처방 문면**(6.1)이 *"KB 를 검색하라"* 라고만 해 손 안의 반증을 못 가리킨 것은
  [[64]] 계열 결함 후보이나, **A2 문면**(`arbitration.unsourced_claim_feedback`)이라 코드 경로 단독
  지목이 아니고 반사실도 미측정 ⇒ **UNPROVEN**.

---

## §7 처방 후보 (⛔실행·코드 수정 금지 — 후보만 · [[70]] 공개 의무)

| # | 처방 | 대상 | 근거 | 무엇을 파나 |
|---|---|---|---|---|
| **P1** | `_ap_regen` **반환 직전**에 ENVGUARD/TRUNCGUARD 술어를 재평가하고, 걸리면 `return None`(= 원본 유지 · 이 함수의 기존 실패 규약과 동형) | `t2_gate_patch.py:11409-11694` | §6-Ⓑ · 술어는 이미 `:11762`·`:11783` 에 있다(사본 0·[[67]]) · 닫힌 술어(`finish_reason` · 봉투 태그 존재 · `tool_calls` 공집합) · 도메인 리터럴 0([[22]]) | 폭주했지만 **산문으로는 쓸모 있던** 재생성도 버려진다 ⇒ 그 턴이 빈손. 부정통제 = 재생성 채택률 · over-action |
| P2 | `build_corpus` 에 `ours_tool_facts` 채널 추가 — 주장의 수치가 **우리 도구가 낸 값 집합**에 있으면 통과 | `t2_source.py:82 build_corpus` · `:271` | §6-Ⓐ 의 근본은 "구조체 ↔ 산문"이지 단어 매칭 세기가 아니다 · [[25]] 우리 도구 = 권위 | `T2_SOURCE` 가 잡는 범위가 좁아진다. 우리 도구가 **틀린 값을 준 경우** 자기 근거가 된다 |
| P3 | `_anchors` 에서 **원장 토큰만** `[_\-]` 로 분해해 부분토큰을 후보에 추가(주장 쪽은 불변) | `t2_source.py:265` | 재현에서 `limit_max` ↔ `limit` 이 만나지 못한다 | 003 은 분해만으로 `need=2` 를 못 넘는다(∩={'limit'}) ⇒ **단독으로는 부족**. 앵커가 헐거워져 다른 항목 줄이 거짓 통과 |
| P4 | A2 의 `"source": "doc <slug>_*"` 표기를 `build_corpus:93` 의 doc-id 정규식이 걷을 수 있는 형식으로 정렬 | `a2/banking_knowledge.specific.json`(+ [[24]] 양방향 `gate.json` 동기화) | `docs=set()` 실측 — 우리가 출처를 쓰고 우리가 못 읽는다 | 없음(표기 정렬). 단 [[24]] 동기화 의무 |
| P5 | *(측정 필요)* user-sim 이 **KB 와 반대되는 단정**을 했을 때 에이전트가 자기 도구 출력으로 되돌아가게 하는 자리. [[21]] 상 agent-측 흡수로 환원해야 한다 | — | 5→6.0 · `prompt_tokens=20770` 로 step 3 은 여전히 문맥에 있었다 | 미측정 **[?]**. 반박 발화는 over-action 위험 |

**순서: P1 이 먼저다.** §6-Ⓑ 는 도메인 무관·닫힌 술어·기존 코드 재사용이고, 003 뿐 아니라
`_ap_regen` 을 지나는 **26개 호출부 전부**의 하방을 막는다. Ⓐ 계열(P2–P4)은 발화 빈도가 훨씬
높으므로 **격리 프로브 후에만** 손댄다([[62]]·[[18]]).

⚠**x500 §E-2 의 재개 지표를 고쳐야 한다**: *"ENVGUARD/TRUNCGUARD 발화 5건 이상"* 은 이 결함을
**구조적으로 셀 수 없다**(가드가 못 보는 자리에서 나므로 발화 0). 대체 지표 후보 =
**커밋된 어시스턴트 턴 중 `finish_reason=length` ∨ (`tool_calls` 공집합 ∧ 봉투 태그 존재)** 의 수
— t7348 halfA 에서 이 값은 **1**, t7346 halfA task_003 에서는 **0**.

---

## §8 남는 불확실

1. **n=1.** 폭주 커밋은 halfA 20 sim 중 1건이다. §6-Ⓑ 의 **결손 자체는 코드로 확정**이지만, 막았을
   때의 **성적 부호는 미측정**이다([[70]]).
2. **turn-4 초안#1 이 재생성 없이도 통과했을지**는 반사실이다. trial0 도 그 초안을 그대로 세운 것이
   아니라 **두 번 더 재생성한 뒤**(claimprov → usertoolnote) 통과했다 ⇒ 성립하는 명제는
   *"개입하지 않았으면 통과"* 가 아니라 *"개입의 산출이 멀쩡하면 통과"* 다.
3. **`T2_SOURCE` 오탐률**은 003 의 5+3 건만 검산했다. 런 전체 비율로 인용하면 안 된다.
4. **halfB 미완**(`bank_t7348_halfBpartial14_20260824`) — 런 전체 판정은 이 문서 밖이다.
5. **user-sim 페르소나 이탈**(step 5)의 재현율 미측정. t7346 두 시행에는 없었고 t7348 trial1 에만
   있다 ⇒ 폭주 턴이 유발했는지(우리 하류) 독립 분산인지 **미분리**.
