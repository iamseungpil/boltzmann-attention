# x811 — C 부류 단순화 설계: **해소 방향을 지정하지 않는다** (2026-09-07)

> ## ⛔ §0 개정 (2026-09-07 · 리뷰 반영 · **이 절이 §4·§6·§7 을 대체한다**)
>
> ### 0-1. §4 의 처방을 **철회한다** — C350 이 이미 죽인 가설이다
>
> `RESEARCH_MASTER.md:247` **C350** 은 **바로 이 문자열**로 격리를 돌린 기록이고
> 하네스가 `x166_our_text_narrowing.py` 로 실재한다. 초판은 이것을 인용하지 않았다([[74]] 위반).
> 축자:
>
> - **②명령** — *"Do the promised work NOW…"* 만 담은 `A_tail` 이 **가장 안전**(0.9822 · 오답 0/10)
>   ⇒ §4 가 빼려던 (c) 가 **그 절**이다.
> - **①길이** — 같은 213자에서 `A_head` 0.5399 ↔ `A_tail` 0.9822 ⇒ *"길이 −38%"* 는 이득 근거가 못 된다.
> - **④절 삭제** — *"but the ledger shows it was never executed"* 한 절 제거 → 0.7132 → **0.1024**
>   (오답 **10/10**). **절을 빼는 방향이 반대로 갔다.**
> - ⒟ 축자: *"내가 댄 이유(명령 제거)는 **틀렸다**"* — **같은 실수를 이미 한 번 했고 기록해 뒀다.**
>
> ⚠**C350 ⒞ 의 «약속 프레임 정정 정도» 서열(E 0.10 < A_head 0.54 < A_imper 0.59 < F 0.71 < B_owner 0.94)은
> 본인이 «사후·사전등록 전까지 인용 금지»라고 못박았다.** 그러므로 그 서열로 새 처방을 세우면
> C350 이 금지한 **네 번째 사후 가설**이 된다. 이 문서는 그 서열을 **arm 설계의 동기**로만 쓰고
> **처방의 근거로는 쓰지 않는다**.
>
> ⚠C350 은 **32B · 단일 결정턴 · 단일 태스크**라 [[79]] 상 레거시다. 반증이 아니라 **재측정 의무**다.
>
> ### 0-2. 착수 순서 교체 — **step 0 = x166 재측정**
>
> | # | 할 일 | 비용 |
> |---|---|---|
> | **0** | `x166` 에 팔 추가 — `E_directive`(§4 의 After) · `F_nodirective_only`. 기준 `A_imper`·`B_owner`·`C_none`. **Q3.8 로**([[79]]) | 유료 0 · **GPU 필요**(⚠리뷰의 "GPU 0" 은 틀림 — `T2_PROBE_URL` 로 엔진에 붙는다) |
> | 1 | 0 의 결과가 처방을 정한다. 서열이 유지되면 §4 는 폐기하고 **«정정 성분을 늘리는»** 쪽으로 다시 쓴다 | 문면 |
> | 2 | `x18_a2_three_layer.py --sync-mono` → `--verify` → `test_claim_*.py` 3종 | 절차 |
> | 3 | `ACTION_REQUIRED_FB`(`t2_resolve.py:336`)를 **같은 팔에** 넣는다 — 004 를 표적으로 걸려면 필수 | **코드** |
> | 4 | §6 을 **D13 인용**으로 다시 쓴다 | 문서 |
> | 5 | A/B (⛔[[86]] 승인 후) — 표적 = «② 발화 태스크 전체 reward 짝» + 부호 반대편 033 계열 | 유료 |
>
> ### 0-3. 배선 정정 (전부 재현 확인)
>
> | 초판 | 정정 | 확인 |
> |---|---|---|
> | 저작 키 `$.claim_prov.*` | **`$.claim_audit.*` @ `a2/base/shared.json`**. `claim_prov` 는 로더가 합성하는 런타임 키 | ✅ base/shared.json 최상위에 `claim_audit` 실재 |
> | §6 *"ACTION(F3)도 문면만·코드 0"* | **거짓** — `ACTION_REQUIRED_FB` 는 **엔진 리터럴**(`t2_resolve.py:336`) | ✅ 축자 확인 |
> | §6 *"F3 를 흡수한다"* | **교체**다. x808 의 F3 는 *조건화*, 이 문서는 *삭제* — [[66]] 상 조건화는 «정책이 이관을 요구하는가»라는 **열린 술어**를 요구하므로 삭제가 옳다. 그러나 «흡수»가 아니다 | — |
> | §5 D12 인용 | ✅ `t2_gate_patch.py:10657` 에 D12ⓐ 수리 실재(2026-09-05) · A2 오버라이드 없음 | ✅ |
> | §2 *"7,226 발화"* | **변종 인스턴스** 수다. `_parts` 가 ownership 과 pending 을 **한 발화에 이어 붙인다**(`:15413`·`:15423`) ⇒ x810 의 7,089 «발화»와 차 137 은 합본 때문 | ⚠ |
> | §2 *"①은 문제 없으니 그대로"* | ①만 나갈 때만 성립. **①+② 합본 비율은 미측정** | ⚠ |
> | §7 *"스위치는 코드 0"* | **거짓** — A2 에 신·구 두 문면을 두고 엔진이 고르는 분기가 필요. **실험 전용 처치 스위치**임을 명시할 것([[60]] 은 레버 얘기지 처치 얘기가 아니다). [[81]] 발화 확인 게이트 필수(사이드카 본문은 `T2_FB_SIDECAR_TEXT=1` 일 때만 저장) | ⚠ |
> | §6 047 진단 | **repo 에 문서 출처 없음**([[77]]④). 이 세션의 인라인 보고뿐 — 인용하려면 먼저 적어야 한다 | ⚠ |
>
> ### 0-4. §4 의 After 문면은 **C-ASSERT 자기위반**이다
>
> 발화창은 `_resign ∨ transfer` 이고 `_resign` 의 정의는 `t2_gate_patch.py:14897` 축자:
> ```python
> _resign = (not getattr(am, "tool_calls", None)
>            and isinstance(getattr(am, "content", None), str) and am.content.strip())
> ```
> = **「이번 응답에 도구 호출이 없고 텍스트가 있다」**. 손님에게 질문만 한 턴도 들어간다. **종료가 아니다.**
> 구판의 *"You are about to end your involvement"* 가 이미 과잉단언인데, §4 의 After 는 그것을
> 문두로 올려 *"your reply **ends** your involvement"* 로 **더 세게** 만든다 — x810 §3 이 잡으려던 형태 그대로다.
>
> **닫힌 대체**(엔진이 이미 두 갈래를 갈라 로그를 찍으므로 새 술어 0):
> - `_resign` 갈래 → *"your reply makes no tool call"*
> - `transfer` 갈래 → *"your reply hands the conversation off"*
>
> ### 0-5. 004 를 표적으로 걸지 마라 (걸려면 화자 셋을 함께)
>
> x808 §7-3 축자: 004 의 원인은 **병렬 둘**이고 *"둘 다 고쳐야 한다"* — ①이관 금지 문면
> ②위조 VERIFIED(`t2_gate_patch.py:593`). 게다가 ①의 **화자도 둘**이다:
> `[CLAIM-PROVENANCE] feedback_pending`(A2) · `[ACTION-REQUIRED]`(**엔진 리터럴**).
> ⇒ step 1 만으로는 **셋 중 하나**만 제거한다. 004 가 0 으로 남아도 계약이 틀렸다는 뜻이 아닌데
> 그렇게 읽힐 자리에 표적을 걸어 두었다. **표적을 «② 발화 태스크 전체의 reward 짝»으로 바꾼다.**
>
> ### 0-6. §6 의 일반화를 좁힌다 — **D13 이 이미 이 계약의 인스턴스다**
>
> 초판 §6 은 판별식을 *"방향강제 절이 있나"* 로 썼는데, §3 의 실격 사유는 **«정책 근거 없음»**이다.
> 두 술어가 다르다. 실측: `PROCEDURE`·`ORDER` 문면은 **정책 출처를 싣고 있다** —
> `.procedures[*].feedback.unmet` 축자 *"Do that before '{tool}' (**source: {source}**)"* ·
> `.procedures[1].feedback.prohibited` *"The policy forbids '{tool}' — verbatim: \"{quote}\""*.
> 이들은 관측이 아니라 **선언된 정책의 재진술**이므로 `C-DIRECT` 대로면 **남는다**.
>
> 진짜 표적은 «정책 근거 없는 선행조건»이고 **그건 이미 이름이 있다 — D13**.
> 근거는 A2 자신의 자백(`banking_knowledge.gate.json:4751` `_note_require_tool_before` 축자):
> > *"나머지 체인(`apply_for_credit_card ← check_card_application_fit` 등)의 선행 도구는
> > **우리가 만든 scaffold GET 도구**이고, **그 체인을 요구하는 정책 문장은 없다** …
> > **[[23]] 소급 대상**: 정책·KB 근거를 찾거나, 못 찾으면 제거해야 한다."*
>
> ⇒ §6 은 *"개별 패치 4건이 접힌다"* 가 아니라 **«D13 이 이 계약의 PROCEDURE/ORDER 인스턴스다»**
> 로 고쳐 쓴다. 더 정확하고 더 강하다.
>
> ### 0-7. 등가게이트 — base 만 고치면 붉어진다 ([[24]])
>
> 같은 문자열이 **세 벌**이다: `a2/base/shared.json`(정본) · `banking_knowledge.gate.json`(레거시 생성물) ·
> `split/banking_knowledge.core.json`. `--verify` 는 `want = base + gate.json` 으로 비교하므로
> **base 만 고치면 `claim_prov` 로 ❌** 가 된다. 현재 baseline은 **3도메인 ✅**(직접 실행 확인 —
> 파일 주석의 *"banking 상시 ❌"* 는 낡았다).
>
> ---

**대상**: `CLAIM-PROVENANCE` — 코퍼스 최대 화자(**7,226 발화** · 총 9.2MB · 중앙 385자 · p90 1,418자).
**계기**: x810 이 C 부류(검산 어려운 궤적 주장)를 단순화 표적으로 지목. 그중 최대.
**선행**: x808 §7-3 — `task_004`(gold = `transfer_to_human_agents`)를 죽인 문면이 이것이다.

---

## 1. ⛔ x810 의 진단을 여기서 **정정한다**

x810 §3 은 C 부류를 *"단언절을 삭제"* 하라고 적었다. **틀렸다.** 문면을 전수로 뜯어보니
단언절은 **참**이고, 병은 그 뒤에 있다.

`{claims}` 의 출처는 우리 해석이 아니라 **모델 자신의 자기감사 서브콜**이다
(`a2 $.claim_prov.question` 축자):

> *"Before your reply is sent, audit yourself. (1) In YOUR REPLY ONLY: list every assertion that
> YOU … have ALREADY performed … (2) Across the WHOLE conversation: list every action YOU said
> you WILL do … that you have NOT completed yet."*

⇒ [[66]] 준수(의도 분류를 엔진이 안 한다). 그리고 원장 대조는 닫힌 술어다.
**`task_004` 실측**: 주장 = `record_update: Change account email to new address` — 모델이 실제로
그렇게 말했고 원장에 실행이 없었다. **단언은 참이었다.**

---

## 2. 세 변종의 절 해부 (전수 · 변종은 3개뿐)

| 변종 | 발화 | 비중 |
|---|---|---|
| ① `feedback_ownership` | 4,313 | 60% |
| ② `feedback_pending` | 1,462 | 20% |
| ③ `feedback` | 1,451 | 20% |

### ① `feedback_ownership` — **문제 없음. 그대로 둔다.**
> `Note: [CLAIM-PROVENANCE] tool ownership — the following are in YOUR OWN tool list, not the customer's: {claims}. The customer cannot run them on your behalf, and describing the steps to them does not execute them.`

단언 = «이 도구는 에이전트 소유다» — **도구 목록에서 직접 나오는 닫힌 사실**. 지시절 없음.
`Note:` 라 거부도 아니다. 60% 가 여기다.

### ② `feedback_pending` — **004 를 죽인 것. 두 절을 뺀다.**
```
(a) 단언   earlier in this conversation you told the customer you WOULD do the following,
           but the conversation ledger shows it was never actually executed: {claims}.
(b) 가치판단 You are about to end your involvement (resign or transfer) with these promises
           unfulfilled — **that abandons the customer's request**.
(c) 방향강제 **Do the promised work NOW** by calling the real tools, and **only transfer or end
           the conversation after the actions are actually executed**
(d) 예외    (or explicitly tell the customer you could not do them and why).
(e) 금지    Do not end your involvement by describing the work as done or under way —
           either call the tool now, or state explicitly that it has NOT been performed.
```
**(a) 참 · (e) 정당**([[25]] 안 한 일을 했다고 말하지 마라).
**(b) 는 가치판단**이고 정책 근거가 없다. **(c) 는 해소 방향을 한쪽으로 강제**한다.
`task_004` 의 gold 는 **이관**이었다 — (c) 가 정확히 그것을 금지했다.

### ③ `feedback` — **이미 양자택일. (b) 만 뺀다.**
> `… your reply asserts you already did the following, but the ledger shows NO such event: {claims}. Do NOT claim actions you have not actually performed — **that misleads the customer**. Either actually do it now …, or correct your statement to say you have not done it yet. …`

*"Either … or …"* 로 **이미 방향을 안 정한다**. 004 형 사고가 구조적으로 덜하다.
남은 잉여는 가치판단 *"that misleads the customer"* 뿐.

---

## 3. 설계 원리 — **우리가 아는 것과 모르는 것을 가른다**

| 우리 층이 **아는 것** | 우리 층이 **모르는 것** |
|---|---|
| 원장에 무엇이 실행됐나 (닫힘) | 그 약속을 지금 지켜야 하는가 (**정책**) |
| 어떤 도구가 에이전트 소유인가 (닫힘) | 떠나는 것이 옳은가 (**정책**) |
| 모델의 자기보고 ↔ 원장의 **불일치** (관측) | 불일치를 **어느 쪽으로** 해소해야 하는가 |

> ### 계약 `C-DIRECT`
> 우리 층은 **관측된 불일치를 진술**할 수 있다.
> **그 불일치의 해소 방향은 지정하지 않는다** — 닫힌 선택지를 제시하고, 고르는 것은 모델이다.
> 가치판단(*"abandons"* · *"misleads"*)은 발화하지 않는다 — 그것은 관측이 아니다.

x810 의 `C-ASSERT`(관측하지 않은 것을 단언하지 마라 · **B 부류** · F1 이 그 인스턴스)와
**짝을 이루는 두 번째 계약**이다. 둘이 C 부류·B 부류를 나눠 덮는다.

근거 정합: [[62]](결정론기는 최소한) · [[52]](엔진=이론 / LLM=해석) ·
[[64]](거부는 «무엇이 틀렸나 + 무엇을 하면 풀리나» — **선택지**이지 명령이 아니다) ·
[[63]](모델은 금지·단정에 닫히므로 **틀린 단정이 비싸다**).

---

## 4. 제안 문면 (before → after)

### ② `feedback_pending`
**Before** (5절 · 이관 금지 포함):
> Error: [CLAIM-PROVENANCE] earlier in this conversation you told the customer you WOULD do the following, but the conversation ledger shows it was never actually executed: {claims}. You are about to end your involvement (resign or transfer) with these promises unfulfilled — that abandons the customer's request. Do the promised work NOW by calling the real tools, and only transfer or end the conversation after the actions are actually executed (or explicitly tell the customer you could not do them and why). Do not end your involvement by describing the work as done or under way — either call the tool now, or state explicitly that it has NOT been performed.

**After** (불일치 + 닫힌 선택지 2 + 금지 1):
> Error: [CLAIM-PROVENANCE] your reply ends your involvement, and the conversation ledger has no execution record for what you said you would do: {claims}. Before ending, make your reply and the ledger agree — either execute what is still required, or state plainly to the customer that it has NOT been done and what remains. Do not end by describing unexecuted work as done or under way.

삭제: (b) *"that abandons the customer's request"* · (c) *"Do the promised work NOW … only transfer or end … after the actions are actually executed"*
유지: 불일치 진술 · 양자택일 · 허위 종료 금지. **길이 −38%**.

### ③ `feedback`
*"— that misleads the customer"* **한 절만 삭제**. 나머지는 그대로.

### ① `feedback_ownership`
**변경 없음.**

---

## 5. ⚠ [[70]] 무엇을 파는가

| 잃는 것 | 실물 |
|---|---|
| *"이관으로 떠넘기지 마라"* 의 강제력 | **D12 가 겨눈 것이 그것이다.** 설계서 D12: 구판 무조건절이 한 태스크에서 gold 이관을 죽였고, 그때 `user_action_feedback` 쪽만 `{tool}` 로 좁혔다 — **`claim_prov` 쪽은 안 좁혔다.** 이번이 그 나머지 반쪽이다 |
| *"지금 하라"* 의 추동력 | 남는 것은 «둘 중 하나를 골라라». [[63]] 상 모델은 **금지에 닫히고 지시에 둔하므로**, 잃는 추동력은 원래 작다 |

**되사는 것**: gold 가 «떠남」인 태스크(004 계열)에서 우리 층이 더 이상 반대를 명령하지 않는다.

⛔**이 절충은 A/B 로 판정한다.** 표본은 충분하다 — ②는 **1,462 발화**.
- 표적 회복: `task_004` (gold = 이관 · 현재 0/1)
- 부호 반대편: D12 가 겨눈 태스크(033 계열 · 이관 떠넘기기)
- 계기: `[T2_CLAIMPROV] … pending` **발화 수는 불변**(문면만 바뀜) — 발화율로 처치를 재인쇄하지 않는다([[57]] C502 함정)

---

## 6. 적용 범위 — 이 설계가 다른 곳에도 붙는가

x810 §1 의 C 부류 나머지에 같은 술어를 대면:

| 마커 | 발화 | 가치판단·방향강제 절이 있나 |
|---|---|---|
| `PROCEDURE` | 912 | **있다** — *"Do that step before continuing"* (047 의 close 를 지목한 그것) |
| `WRITE-EVIDENCE` | 998 | 있다 — *"First read … then call …"* |
| `ORDER` | 2,195 | 미확인 |
| `ACTION` | 2,459 | **있다** — *"do NOT just explain …, advise self-service, or transfer"*(F3) |

⇒ `C-DIRECT` 는 `CLAIM-PROVENANCE` 전용 수리가 아니라 **C 부류 공통 계약**이다.
`F3`(x808)이 이 계약의 `ACTION` 쪽 인스턴스이고, `047` 의 `PROCEDURE` 도 같은 형태다.
**개별 패치 4건이 계약 1개로 접힌다** — 이것이 요청받은 «단순화·일반화»다.

## 7. 착수 순서
1. ② `feedback_pending` 문면 교체 (A2 3층) + ③ 가치판단 1절 삭제 → **문면만·코드 0**
2. 같은 술어를 `ACTION`(F3)·`PROCEDURE`(047)에 적용 → 개별 수리 2건 흡수
3. A/B: `T2_CLAIMPROV_DIRECTIVE=0|1` 한 칸으로 신·구 문면 전환 (⛔[[86]] 승인 후)
