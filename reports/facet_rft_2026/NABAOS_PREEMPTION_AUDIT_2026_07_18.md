# NabaOS 선점 감사 — **우리가 만든 것은 NabaOS인가?** (2026-07-18 · **v3**)

## ★감사 대상 식별자 (인용키·v3 신규)
| | |
|---|---|
| **arXiv** | **`2603.10060` v1** · `[cs.CR] 9 Mar 2026` |
| **제목** | *Tool Receipts, Not Zero-Knowledge Proofs: Practical Hallucination Detection for AI Agents* |
| **저자** | **Abhinaba Basu** (단독저자) |
| **취득/정독** | 2026-07-18 · pdftotext -layout(poppler 4.00) · **891줄** · PDF SHA-256 `9891928a…de3a8a` |
| **줄번호 재현** | ★**`SOURCE_PROVENANCE.md` §1** — 원문은 저작권상 repo에 안 싣는다. **sha256+추출 명령으로 30초 재현.** |
> ⚠️**v1/v2는 이 블록이 없어서 인용 불가 문서였다**(사용자 지적·2026-07-18): 줄번호 ~30개가 **내 scratchpad에만**
> 있는 파일을 가리켰다 = **아무도 반증할 수 없는 권위**. **v1이 죽은 원인이 정확히 그것**이었는데 v2는 인스턴스
> 5건만 고치고 원인을 안 고쳤다. ⇒ `SOURCE_PROVENANCE.md` 개설·**미등재 줄번호 = 인용 무효** 규율.

> 계기: C111 정독 직후 사용자 질문 3개 — ①"프롬프트로 검증가능 영역 전환도 선점인가" ②"프롬프트 안 되면
> 학습도 선점인가" ③**"그럼 이때까지 만든 게 거의 NabaOS 그대로인가?"**
> 정본 관계: `RELWORK §4c-7`·`COMPLETION_EVIDENCE_LEARN_DESIGN §0a-1`의 **상위 감사 문서**.
> 등급: 인용 = **[M-lit] 원문 축자**(pdftotext 891줄 전문·줄번호 표기) · 부재 주장 = **[S] grep** · 우리 실측 = 각 줄 표기.
>
> ## ⚠️v1 → v2: **적대적 리뷰가 내 주장 5건을 죽였다** (2026-07-18·리뷰어 지적 → 내가 원문 6곳 직접 재확인·전건 확인)
> | v1에서 내가 쓴 것 | 실제(줄번호) | 판정 |
> |---|---|---|
> | "학습 언급 **전수 2회** [S]grep" | **≥4회**. `489` SVIP-style = *"A lightweight classifier **trained** on response features"* = **그들이 실제로 돌린 학습 baseline** · `743` *"adversarially **trained** model"* | ❌**거짓**. 내 grep이 `trained`(과거분사)를 놓침 = **33.40과 같은 오류 클래스**(grep 권위로 틀린 수를 말함) |
> | "결과를 **아무도 안 쟀다** · 평가=탐지율뿐" | `598-607` **Table 6 "Actual Correctness"** = 98.7/94.2/82.5/23.4/71.2% · `500` clean control FPR | ❌**거짓**. 그들은 **주장-수준 정확도·캘리브레이션**을 잰다. 살아남는 건 **task-level e2e pass(라이브 루프)**뿐 |
> | §3 "PROV 24/24 = 그들 **가정의 반례**" (§4서 **1순위**) | `743-745` *"fabricated tool call detection and count mismatch detection **remain effective because they do not rely on self-tags**"* · cooperative 가정은 **inference-as-fact에만 스코프** | ❌**빗나감**. 우리 24/24 = **도구명 날조** = 그들이 *self-tag 없이 잡는다고 명시한 바로 그 클래스*. **"가정 반증" 철회**(§3) |
> | §2.1 "`Compromised tools`가 **인자 날조**를 자인" | `739` 전문 = *"**If a tool itself returns incorrect data**, the receipt will be valid…"* = **도구 백엔드 오류** | ❌**오귀속**. 생략부호가 주어를 지웠다. **단 우리 §5.2(env가 거짓 반환)엔 여전히 정확**(§2.4) |
> | §6-3 "그들 **85%가 우리 주장을 약화**시킨다" | `349` *"Non-compliant responses--**those that do not include the verification metadata**"* = **포맷 발화율**이지 태깅 정확도가 아님 | ❌**기우**. 오히려 **오픈웨이트 태깅 정확도는 아무도 안 쟀다**=우리 자리(§2.5) |
> ★**리뷰의 최대 기여**: v1의 절단선 *"주장 vs 인자"*는 **합리화**다 — **보고된 인자는 곧 주장**이라 그들 `pratyakṣa`
> 검사에 걸린다. 진짜 절단선은 **read-path vs write-path**(§2.1). v1은 **자산은 맞게 갖고 이름을 틀리게** 붙였다.

---

## 0. 한 줄 답 (**v5**)
**"거의 NabaOS 그대로"는 아니다**(11칸 중 겹침 3·부분 3·NabaOS 범위 밖 4·미선점 1) — **그러나 겹친 3개가 §4
학습 트랙의 코어였고, 오늘 세운 대체 프레임 3개(§3 게이트-역효과 → write-path → (b)버킷)가 *전부* 죽었다.**
**남은 코어 후보 = 행동 축뿐이고, 그마저 데이터가 있는 건 ASK 하나이며 레인 감사가 0이다**(§6-❺).
⇒ **지금 정직한 상태 = "코어 후보 1개(ASK), 미감사."** 새 프레임을 랭크하기 전에 **§8 하드룰**을 먼저 통과시킬 것.

## ~~0-v2~~ 한 줄 답 (기록 보존)

**아니다 — 스택 10칸 중 겹침 3·범위 밖 5·미선점 2.** 그러나 **하필 겹친 3개가 §4 학습 트랙의 코어**였다.
**스택은 안전하고, 논문 한 편의 프레임이 다쳤다.**
★그리고 v1이 "가장 강한 카드"라 불렀던 §3(게이트가 날조를 제조)은 **적대적 리뷰로 1순위에서 3순위로 강등**됐다.
지금 **가장 방어 가능한 자리 = write-path(보고되지 않는 인자 날조)**다(§2.1).

---

## 1. 선점된 것 (축자·양보 확정)

### 1.1 "프롬프트로 undecidable 표면 → decidable 표면" = **완전 선점** (사용자 질문 ①)
§3.4 Self-Tagging Prompt(`336-344`·pdftotext 줄바꿈만 정리):
```
---VERIFICATION---
For each factual claim in your response , classify:
- claim: "the statement"
- source_type: tool_output | inference | external_source | absence | opinion
- evidence: which tool receipt ID supports this
- checkable: true/false
---END VERIFICATION---
```
- **필드명이 `evidence`이고 `checkable: true/false`까지 있다.** "비슷하다"가 아니라 **같다**.
- Stage 4(`309-311`): *"structured metadata tagging each factual claim with its pramāṇa category and the
  receipt ID it claims as evidence"* · 영수증=HMAC(`17` 초록: *"receipts that the LLM cannot forge"* ·
  `268` §3.2: *"Any claim referencing a non-existent receipt ID is immediately detectable"*).
- ⇒ **§3.1 "논문 코어 재정의(모든 주장 evidence)"를 신규 설계로 제시하면 즉시 리젝.** 외부 독립 증거로만 인용.

### 1.2 개입(block) 존재 · **엔진고정+도메인스펙 패턴도 선점** ⚠️v2 신규
`686-688`: *"a **configurable constitution specified in TOML format per agent**, defining thresholds for each
trust level, **the action to take when a threshold is violated (block, warn, or pass)**, and **domain-specific
verification rules**"* + `289`: *"each **tool adapter defines** what fields constitute 'facts'"*.
⇒ ~~"탐지-only"~~ 철회 **+ ★v1이 놓친 것**: 이 구조 = **엔진 고정 · 도메인 거동은 per-domain 스펙으로 공급** =
**GateInterpreter(gate_spec)+ABox-swap([[05]]/[[16]] 키스톤)과 같은 형태**. **패턴 자체가 선행** ⇒ **인용 필수**.
(우리 delta = 그 스펙이 **행동 레버**(A2 도구·ASK 분기)를 싣는다 vs 그들은 **검증 정책**만 싣는다. `689`:
*"applies it to verification policy rather than behavioral alignment"* — **그들 입으로 행동 축과 선을 그음**.)

### 1.3 self-tag 없이도 구조 대조 — `T2_FOLLOWUP` 부분 선점 **+ §3 강등의 근거**
`354-357`: *"Importantly, self-tagging is **not the sole verification mechanism**. **Even without self-tags**,
the verification engine can detect **fabricated tool calls (receipt ID does not exist)**, count mismatches …"*
`743-745`: *"…could bypass the **inference-as-fact detector**. **However, fabricated tool call detection and
count mismatch detection remain effective because they do not rely on self-tags.**"*

### 1.4 **결정론 offload 검증·스키마 검증도 선점** ⚠️v2 신규(v1이 "범위 밖"이라 오판)
- `374-375`: *"**Computation replay.** For deterministic computations (arithmetic, code execution, data
  transformations), **independently re-execute the computation and compare results**."* ⇒ **[[10]] 결정론
  검증기·concrete-offload와 동형** — Synth COMPUTE 트랙에 닿는다. **인용 필수**.
- `367-368`: *"**Schema validation** … Structural anomalies … indicate fabrication"* ⇒ A2 도구스키마 레버의
  **검사 측면**과 겹침.
- `144-146`: *"**None** address semantic integrity … NabaOS is complementary."* = **의미-그라운딩 레인 전체에
  깃발**을 꽂았다. §4 학습 트랙이 그 레인 안에 있다 ⇒ **포지셔닝 위험**.

---

## 2. 선점되지 **않은** 것 (v2 재절단)

### 2.1 ❌**v5 — write-path 프레임 폐기**. 우리 정본이 기함을 부정한다 (2026-07-18·사용자 ❸/❶·무료 확인)
> **사용자 지시**: *"결과가 'denylist 적용됨'으로 나오면 기함을 놓아주셔야 합니다 — 또 교체하지 마시고."*
> **결과 = 적용됨(두 겹). 놓아준다. 교체하지 않는다. write-path 절단선은 폐기다.**

**① 포크 판정 — 우리 정본이 `log_verification`을 write로 안 본다** (v4가 tau2 env `ToolType.WRITE`만 조회한 결과):
- **설계서 정본** `BANK_EPLAN_ALLACTION_IMPL_DESIGN §2`: **PROCEDURAL(denylist)에 `log_verification`을 이름으로 등재.**
  §3.1 정의: `_is_write(name) = name and not _READ_PREFIX.match(name) and not _PROCEDURAL.search(name)`
  ⇒ 실행 확인: **`_is_write('log_verification') = False`** (`^log_`·`_verification$` 양쪽 매칭).
- **라이브 엔진** `t2_gate_patch.py:2398-2406` (`T2_FAB_STRIP` 경로)이 **자체 `_PRC`**에 `^log_|_verification$`(및
  `get_current_time`)를 넣고 축자 **`return False  # read/procedural = 무해`** ⇒ **버그가 아니라 의도된 설계**로
  건드리지 않는다.
- ⇒ ★**"정본이 둘인데 유리한 쪽만 조회했다"**(사용자 ❶). **tau2 env 타입 = DB 변이 / 우리 엔진 = 강제 대상 여부.**
  **write-path 논증("피해가 세상에 남는다")을 지배하는 건 후자다** — 그리고 후자가 **아니라고 말한다.**

**② ★반증 시험 — 프레임이 예측하는 곳에 증거가 0이다** (사용자 제시·내 데이터로 실행):
| 분류 | 도구 | 적중 |
|---|---|---|
| ★**결과적 쓰기**(`_is_write`=True) | `submit_transaction` · `change_user_email` · `apply_for_credit_card` · `submit_referral` | **0 / 4** |
| 부기 로그(**우리 정본 = 무해**) | `log_verification` | **2** ← v4 기함 **전부** |
| 디스패처(`_is_write`=False) | `call_discoverable_*` | 9 → **전부 오탐**(JSON 직렬화·v4 ④) |
⇒ **write-path 프레임이 옳다면 `submit_transaction(amount/date)` 날조가 있어야 한다. 0이다.**
**증거는 프레임이 예측하는 곳에 없고, 프레임이 신경 쓰지 않는 곳에만 2건 있다 = 확증이 아니라 반증 신호.**

**③ ⚠️v4 ②의 수사가 이 기함에선 안 선다**(사용자 ❹): *"피해는 텍스트가 아니라 세상에 남는다"* — 실물은
**"신원확인 시각을 15:30으로 잘못 기록"**이다. 부서진 것 = **감사추적(audit trail)**. 이건 진짜 논증이지만
**다른 논증**이다 — write-path 위협이 아니라 **compliance/audit-integrity(F1)**이고, **그 레인은 또 다른 선행 밭**이다.
⇒ 프레임을 그리로 옮기려면 **그 레인 감사를 먼저** 하고 옮긴다(§6-❺). **지금은 옮기지 않는다.**

**④ 그래서 v4에서 살아남는 것 / 죽는 것**:
- ❌**폐기**: "write-path = 우리 화이트스페이스"·"(b) 버킷 실재"라는 **헤드라인**. 정확한 서술 =
  **"(b)는 부기 로그 1종에서만 n=2 · 결과적 쓰기 4종 전부 0."**
- ✅**유효(프레임과 독립)**: `facts`=출력-only(`286-287`) · (a)는 잡히고 (b)는 개념상 열린다는 **분석** ·
  ~~"record 46% = write-path"~~ 거짓(v3 ①) · **`_hint_hit` 다중토큰 버그**(§7-1) · **`record` 커버리지 공백**(§7-2).
- ⇒ **NabaOS 사각은 개념적으로 존재하나, 우리 데이터가 그것을 점유하지 않는다.** 논문 코어 후보에서 **제외**.

### ~~2.1-v4~~ (아래는 정본 충돌 확인 前 · **①~④가 헤드라인을 대체** · ①facts·③hint버그·④오탐 판정은 유효)
### 2.1-v4 — v3의 "0"은 내 hint 설정 버그였다 (2026-07-18·사용자 ❷·범위확장 재측정)
> **사용자 지적(❷)**: *"HMAC=무결성 ≠ 출처. 두 축은 직교한다. 검토자가 과잉교정했고 당신이 너무 빨리 받았다."*
> **판정: 옳다 — 다만 방향은 예측과 반대였고, 진짜 성과는 (b) 버킷이다.**

**① `facts` 구성 = 결정됨** (`286-287` 축자): *"the runtime extracts key-value pairs from **the structured tool
output**"* ⇒ **facts = 출력-only. 입력 인자는 `input_hash`로 봉인될 뿐 주장 대조에 안 쓰인다.**
- ⇒ **(a) 값-보고는 잡힌다**(사용자 예측과 **반대**): 날조값이 facts에 없으므로 pratyakṣa 대조가 **실패**→**flag**.
  NabaOS에서 flag는 실패가 아니라 **제품**(trust level). **검토자의 킬은 (a)에 한해 성립.**
- ⇒ ★**(b) 행위만-보고는 열린다**(사용자 옳음·facts 구성과 **무관**): *"접수/확인했습니다"*는 **참인 주장**이고
  (영수증 존재·write 실제 발생) **값을 낭독하지 않으므로 대조할 표면이 없다.**

**② ★(b)의 존재증명 — 실측 2건** (`--all-args`·WRITE 8종·전수·per-case 정독):
| sim | `get_current_time` | `log_verification(time_verified=…)` | 판정 |
|---|---|---|---|
| `c2b17acb`(koff) | ✓ 호출 | `2025-11-14 03:40:00 EST` | grounded |
| **`c4607d53`(kon)** | **없음** | **`2023-11-14 15:30:00 EST`** | **날조** |
| **`495e2f46`(koff)** | **없음** | **`2023-11-14 15:20:00 EST`** | **날조** |
- ★**날조 2건 다 `2023`년 — env 시계는 `2025`년**. **모델이 자기 prior서 연도를 꺼냈다** = 재포맷이 아니라 **환각 확증**.
- `log_verification` = **진짜 WRITE**(tau2 `ToolType.WRITE` 8종 중 하나) · 값은 **미낭독** · 행위만 서술 = **(b)**.
- ⇒ **NabaOS 사각의 실물**: 영수증은 `time_verified`를 **충실히** 기록하고, *"신원을 확인했습니다"*는 **참**이며,
  facts=출력-only라 **입력 타임스탬프는 영영 대조되지 않는다.**
- ⚠️**kon(게이트 ON)서도 살아남았다** — 별도 조사 필요(왜 PROV가 안 잡았나).

**③ ⚠️v3의 "WRITE 날조 0"은 *내 버그*였다** (설계 선택이 아니라): `time_verified`는 **A2 `identifying_arg_types`에
있는데** 내가 census에 **`DEFAULT_ARG_HINTS`만** 넘겨 **A2 enrichment를 빠뜨렸다**. ⇒ 그 "0"은 **내 hint 설정의
아티팩트**. **사용자가 "확인 없이 단정"을 경고한 바로 그 클래스를 한 층 위에서 또 했다.**

**④ 오탐 9건 = JSON 직렬화** (per-case 정독으로 배제·[[08]]): `call_discoverable_agent_tool(arguments='{"transaction_id":
"txn_f093f96e2001", …}')` — `arguments`가 **JSON 문자열 파라미터**라 통째로 잎이 되어 ctx에 축자로 없다.
**내부 값은 전부 `grounded=True`**(실측). ⇒ 11건 중 **9 오탐 · 2 진짜**.

**⑤ 사전등록 반증조건**(결과 보기 前 스크립트에 인쇄): *"WRITE ∧ (b+c) == 0 이면 인자-축 절단선 폐기"* →
**미발동**(2건). ⇒ **인자-축(=(b) 버킷) 절단선 생존. 단 n=2 = 존재증명이지 rate가 아니다**([[08]]: 이걸로
모집단 몫을 주장하지 말 것).

**⑥ 그래서 §2.1-v3의 두 결론 중 하나만 살아남는다**:
- ❌~~"write-path는 우리 데이터가 지지하지 않는다"~~ **철회**(위 ②).
- ✅**"`record` 46% = write-path"는 여전히 거짓**(`verify_identity` = A2 compute 도구·WRITE 아님·v3 ① 유효).
  ⇒ **기함 사례가 바뀐다: `record`가 아니라 `log_verification(time_verified)`.**
- ✅**v3 ⑤(우리 검증기가 LLM 공급 증거를 먹는다)도 유효** — 별개 축이고 처방은 NabaOS 채택(receipt 바인딩).

### ~~2.1-v3~~ (아래는 범위확장 前 · **①~⑥이 대체** · v3 ①④⑤는 유효)
> **v2가 여기서 *"우리 `record` 46%가 정확히 write-path에 산다"*고 **주장**했다. 재보니 **거짓**이다.**
> **적대적 리뷰가 준 프레임(read/write)이 v1보다 나아 보였지만 — 우리 데이터가 그 칸을 채우지 못한다.**

**① 기함 사례가 애초에 WRITE가 아니다** (판정 근거·코드 직독):
- `record` 날조의 대상 도구 = **`verify_identity`** (`bank_fab_probes.cls_record`).
- `verify_identity` = **`a2/banking_knowledge.gate.json` → `scaffold_get_tools[1]`** = `get_reward_discrepancies`와
  **같은 A2 compute 도구**. tau2 banking env에 **없다**([S] 리모트 grep 0).
- tau2 `@is_tool(ToolType.WRITE)` 8종 = `apply_for_credit_card·call_discoverable_agent_tool·
  call_discoverable_user_tool·change_user_email·log_verification·request_human_agent_transfer·
  submit_referral·submit_transaction` — **`verify_identity` 없음**.
⇒ ***"record 46% = write-path"는 틀렸다.*** verify_identity는 세상에 쓰지 않는다. **verdict를 계산할 뿐이다.**

**② 전수 census: WRITE 날조 = 0** (`bank_readpath_writepath.py` · 영속 banking sim **8종 전건**):
| | 보고 | 미보고 | |
|---|---|---|---|
| **WRITE** | **0** | **0** | ★**NabaOS 사각 = 0/5 = 0%** |
| READ | 3 | 2 | **5건 전부 동일 아티팩트**(아래 ③) |
⇒ **live 궤적에 "지어낸 인자로 WRITE 실행" 사례가 (탐지 범위 내에) 없다.**

**③ 그 5건마저 오탐이다**: 전부 `unlock_discoverable_agent_tool(agent_tool_name='get_reward_discrepancies')`.
`_hint_hit('agent_tool_name')=True`(hint `name`에 걸림) — **식별값 날조가 아니라 도구 이름**이다. (=(a1)이 고치는 그 호출.)

**④ ⚠️이 census의 사각 — 0%를 과대해석 금지**([[08]]): 엔진 탐지기는 **hint 매칭 인자만** 본다.
`_hint_hit`: `record`=**False** · `provided`=**False** · `amount`=**False** · `date`=**False** /
`name·email·user_id·address·phone_number·transaction_id`=True.
⇒ **이 census는 `record` 46%를 구조적으로 볼 수 없다**(그래서 n이 5로 작다). **"WRITE 인자 날조 0"은
"식별값 인자에 한해 0"까지만** 말한다. `submit_transaction(amount/date)` 날조는 **미검**.

**⑤ ★그래서 진짜로 남는 것은 write-path가 아니라 이것이다 — 그리고 우리에게 불리하다**:
`record` 날조는 **세상에 쓰는 것도, 사용자에게 말하는 것도 아니다. *우리 검증기에 먹이는 증거*를 지어내는 것이다.**
`verify_identity(provided={...}, record={...})` — **LLM이 주장과 증거를 둘 다 공급한다.**
우리 `cls_record`가 이미 그 공허함을 인코딩해 뒀다: `if set(vals) <= provvals: return ("★record=provided복사(순환)", "FAB")`.
⇒ ★★**NabaOS의 HMAC 영수증은 정확히 이 구멍을 막으려고 존재한다** — `230-232` *"tool calls are executed by
the agent **runtime--not by the LLM**"* · `17` *"receipts that the LLM cannot forge"*.
**즉 이 축에서 NabaOS는 우리보다 낫다.** 그들은 선행일 뿐 아니라 **우리 설계 결함을 진단한다.**
⇒ **처방(신규 작업·[[05]] 안전)**: `verify_identity`가 **LLM이 복사한 record를 받지 말고, 앞선 lookup 호출의
참조(=우리 원장의 receipt)를 받게** 한다. 엔진은 이미 `_called_tools(state.messages)` 원장을 갖고 있다
(`T2_SELF_DECLARATION`). **HMAC은 불요**(엔진이 원장 소유자). ⇒ record 날조 표면 **자체가 소멸**한다.
**이게 §4의 새 1순위 후보다** — 단 이건 **그들 설계의 채택**이지 우리 신규성이 아니다.

### ~~2.1-v2~~ (아래는 계측 前 서술 · 기록 보존용 · **①~⑤가 대체**)
**v1의 오류**: *"그들=주장 / 우리=인자"* 는 **합리화**다. 에이전트가 `CASE-123456`을 지어내 넘기고 **사용자에게
보고하면**, 그건 `pratyakṣa` 주장이라 `facts`(`315-316`)·`result_count`(abhāva) 대조에 **걸린다**. 게다가
`230-232` *"tool calls are executed by the agent runtime--**not by the LLM**"* + `242` `input_hash` +
`256` HMAC ⇒ **인자 경로는 이미 서명 체인에 묶여 있다**. "우리는 인자, 그들은 주장"은 **같은 것의 개명**이라
리뷰어가 안 받는다.

**진짜 빈자리(v2)**: **NabaOS는 read-path 전용이다.** threat model 전체(`724-751`)가 *"사용자에게 거짓을
말했는가"*에서 끝난다. Stage 6도 *"Trust-Annotated Output"*(`328-330`) = **보고 측** 주석이다.
⇒ **지어낸 인자로 WRITE가 실행됐는데 그 값을 사용자에게 보고하지 않는 경우** — 영수증은 **충실히** 기록되고,
**아무도 거짓을 듣지 않았으며**, 피해는 **텍스트가 아니라 세상에** 남는다. **그들에게 이 표면은 없다.**
- ★**우리 실측이 정확히 거기 산다**: `record` 날조 **46%** · grounded 정답 **0/24** · A2 설명레버로도 **0/24**
  (`FAB_PROBES §2`) — `record`는 **WRITE 도구**다.
- ★**`T2_SELF_DECLARATION`이 이미 이 축**(`t2_gate_patch.py:2813`): LLM이 **operand 출처 선언** → 엔진이
  **`_called_tools(state.messages)`(실제 호출 이력)와 대조**. (HMAC 불요 = 엔진이 원장 소유자.)
- **필요 작업**: 우리 실패를 **보고된 것 / 보고되지 않은 것**으로 분리 계측해야 한다(**미측정**). 이 분리가
  없으면 §2.1은 주장일 뿐이다.

### 2.2 **task-level 결과** — v1의 "아무도 안 쟀다"를 **좁힘**
⚠️**v1 거짓**: 그들은 **잰다** — `598-607` Table 6 *"Actual Correctness is the fraction of claims at each trust
level that are **factually correct according to ground truth**"*(98.7/94.2/82.5/23.4/71.2%) · `120-122`
캘리브레이션 · `500` clean FPR.
✅**살아남는 좁은 주장**: `task success`/`pass rate`/`pass^k` **본문 0 hits**([S]) · 벤치가 **라이브 루프 아님** —
시나리오 = *(user request, tool outputs, llm response, ground truth)* **튜플 1,800건**(`394-397`) ·
Lim 6(`717-719`): *"uses **systematically injected hallucinations**. Real-world hallucination patterns may differ"*.
⇒ **"주장 정확도"는 쟀고 "과제 성공"은 안 쟀다.** 그들 94.2% vs 우리 54% **병치 금지**([[08]] 분모).

### 2.3 **ASK가 없다** ([S] 검증됨·리뷰어 독립 재확인)
`clarif|ask the user|query the user|prompt the user|human-in-the-loop|escalat|defer` = **관련 hit 0**.
pramāṇa 6종은 **전부 주장의 라벨**(Table 1 열 이름 = *"Verification Method"*)이고 `abhāva`는 *"없었다고 **주장**"*.
ungrounded = *"Cannot verify — flag"*. `225`: *"**The user can then apply their own judgment.**"*
⇒ **undecidable 표면을 없애지 않고 사람에게 이양**한다. 우리 TOOLGATE→**ASK 24/24 종결**은 다른 설계.

### 2.4 **INFER-calibration = 그들이 "안 한다"고 명시한 사각** ([[16]] 유일 잔여와 일치)
- `anumāna` 검증 = *"verify that the **premises cited exist** in the receipt facts"*(`318-319`) — **전제 존재만**.
- `749-751`: *"**Reasoning errors.** NabaOS verifies that claims are grounded in evidence, **not that the
  agent's reasoning from that evidence is logically valid**."*
- ✅**`Compromised tools`(`739-741`)의 정확한 용처** = **우리 §5.2**: *"**If a tool itself returns incorrect
  data**, the receipt will be valid but the underlying data will be wrong."* — env가 *"This tool is not
  available"*(거짓)을 반환 = **도구가 틀린 데이터를 반환** ⇒ **여기엔 정확히 맞는다.** (v1이 이걸 **인자 날조**에
  갖다 붙인 게 오귀속.)

### 2.5 학습 (사용자 질문 ②) — **NabaOS는 에이전트를 학습 안 시킨다. 그러나 노벨티가 아니다.**
⚠️**v1 [S] 수치 거짓**(전수 재검): 학습 언급 **≥4회** — `152` VerifierQ(*"trains Q-learning-based verifiers"*) ·
`153` SVIP(*"trains a learned inspector model"*) · **`489` SVIP-style = 그들이 실제로 돌린 학습 baseline**
(Table 3서 68%) · `712` Lim 5 · `743` *"adversarially trained model"*.
✅**그럼에도 결론은 생존**: **학습 대상이 전부 "검증기" 아니면 "적대자"다. 검증받는 에이전트를 학습시키는 자리는
비어 있다.**
⚠️★**그러나 비어 있다고 노벨티가 아니다**: *"프롬프트가 천장이면 학습으로 설치"*는 **방법론적 상식**이고 [[42]]에
선행이 쌓여 있다. **"우리는 프롬프트 대신 학습했다"는 기여가 아니라 방법이다**(리뷰어 = *obvious next step*).

---

## 3. ⚠️§3 강등 — v1이 "가장 강한 카드"라 한 것은 **빗나갔다**

**v1 주장**: PROV 게이트 前 **0/24** → 後 **24/24**([M]·`prov_reloc`·독립 2런·sim1 [16] 재현)
⇒ *"NabaOS의 cooperative 가정(`352-353`)에 대한 반례"*.

**왜 죽는가(축자 3겹)**:
1. **스코프 불일치**: cooperative 가정은 **inference-as-fact 검출에만** 걸려 있다(`743-745`·Lim 4 `708-710`).
   우리 24/24 = **도구명 날조** = 그들이 *"remain effective **because they do not rely on self-tags**"*라고
   **명시한 바로 그 클래스**(94.2%). **가정 옆에 떨어졌지 가정을 맞히지 못했다.**
2. **strawman**: *"adversarially trying to circumvent"*(`353`) = 모델이 **회피**하는 것. **비적대 모델이 프롬프트에
   의해 날조하도록 유도되는 것**은 다른 현상이고, 그들은 *"프롬프트가 오류를 유발하지 않는다"*고 주장한 적이 없다
   (Lim 1이 이미 프롬프트의 granularity 손실을 자인).
3. **더 절약적인 대안 설명을 우리가 안 닫았다**: 우리 **게이트끼리 모순**(축자: PROV *"do NOT ask the user … call
   a getter"* ↔ TOOLGATE *"Do not invent tools … ASK"*)이면, 24/24의 가장 단순한 원인은 **우리 프롬프트 버그**다.
   리뷰어가 즉시 든다.
4. **자기모순**: §3서 모집단 몫 **1/13**(`bank_regen_attribution.py`·kon 3중 1·koff 10중 0)을 자인해놓고 §4서
   **1순위**로 랭크했다 = *"본문서 양보하고 요약에서 챙기기"*.

✅**살아남는 좁은 주장(3순위)**: **self-tagging류 프롬프트는 *생성 측* 효과를 갖는데, NyayaVerifyBench는 구조적으로
그걸 볼 수 없다** — 시나리오가 **사전 작성 튜플**(`394-397`)이라 *"게이트가 날조를 만드는가"*를 물을 자리가 없다.
⇒ 이건 **그들 *평가*의 공백**이지 **그들 *가정*의 반증이 아니다.** 이 선을 넘으면 리젝.

---

## 4. 재정렬된 "남는 자리" (v2·강→약)

1. ❌~~**인자-축 (b)버킷 = write-path**~~ — **v5서 폐기**(§2.1-v5): 우리 정본이 기함(`log_verification`)을
   **procedural=무해**로 판정하고(설계서 §2 denylist·`_is_write`=False·엔진 `_PRC`가 축자 *"read/procedural = 무해"*),
   **결과적 쓰기 4종 전부 0**. **프레임이 예측하는 곳에 증거 0 = 반증 신호.** ⇒ **코어 후보서 제외. 기함 재교체 금지**(§8).
2. ★★**ASK 종결**(§2.3) + **INFER-calibration**(§2.4·[[16]] 유일 잔여 = 그들 자인 사각). **행동 축 = 그들이
   `689`서 스스로 선을 그은 곳**(*"verification policy rather than behavioral alignment"*).
3. ★**task-level 결과 계측**(§2.2·좁힘) + **생성-측 효과를 그들 벤치가 못 본다**(§3 잔여).
4. **레버 상쇄 계측**(Δspurious·모트 제1원리) — 그들은 *"conservative fallback"*의 비용을 **자인만 하고 안 잰다**
   (단 Table 6서 Ungrounded 정확도 71.2%는 **쟀다** — v1이 이것도 놓쳤다).
5. **오픈웨이트 태깅 *정확도*** — §2.5 참조. **아무도 안 쟀다**(§6-1).
6. ~~학습으로 설치~~ = **방법**(1~5를 가능케 하는 수단으로만 서술).

---

## 5. 스택 전수 대조 — **"거의 NabaOS 그대로인가?"** (v2)

| 우리 컴포넌트 | 축 | NabaOS 대응 | 판정 |
|---|---|---|---|
| **`T2_WRITE_PROV`**(출력측 출처선언) | 주장 | **self-tagging prompt**(`336`) | ★**겹침** |
| **`T2_FOLLOWUP`**(완료주장 구조이벤트) | 주장 | **receipt lookup**(`354-357`·self-tag 불요) | **부분 겹침**·잔여=**텍스트 파싱 0** |
| **evidence 학습**(계획) | 주장 | **프롬프트로 함** | **설계 겹침**·수단만 다름(§2.5) |
| **`T2_SELF_DECLARATION`**(operand 출처→호출이력) | **인자/write** | **없음**(read-path 전용·§2.1) | ✅**미선점**(단 §2.1 전제) |
| **`T2_TOOLGATE`**(미지 도구 → **ASK**) | 행동 | **없음**(ASK 부재·§2.3) | ✅**범위 밖** |
| **A2 레버**(스키마·인자누적 명시) | 행동 | `367` schema validation과 **일부** 겹침 | ⚠️**부분**(v1 "없음"=오판) |
| **`T2_DISCOVERY_REQUIRED`** | 행동 | 없음 | ✅**범위 밖** |
| **(a1) `T2_SG_TRUTH`**(인터페이스 사실 정정) | 환경 | 없음(단 `739` Compromised tools가 **현상은** 자인) | ✅**범위 밖** |
| **`T2_EPLAN`** | 행동 | 없음 | ✅**범위 밖** |
| **엔진고정+gate_spec**([[05]]/[[16]] 키스톤) | 아키텍처 | **`686` TOML constitution + `289` tool adapter** | ⚠️**패턴 선행**(v1 누락·인용 필수) |
| **결정론 offload 검증**([[10]]) | 검증 | **`374` Computation replay** | ⚠️**겹침**(v1 누락·인용 필수) |
| **실측 발견**(깔때기·KB prior·env 거짓말·인자누적 60→93.3) | — | 없음(라이브 루프 없음) | ✅**범위 밖** |

⇒ **11칸**(v4 교정: **실측 발견 칸 = 컴포넌트가 아니라 발견이므로 분모서 제외** — 사용자 ❹): **겹침 3 ·
부분/패턴 겹침 3 · NabaOS 범위 밖 4 · 미선점 1**.

⚠️★**라벨 교정 (사용자 ❹ — v1/v2의 ✅는 거짓 위안을 날랐다)**:
- ~~"범위 밖 ✅ = 안전"~~ → **"NabaOS 범위 밖 = *타 선행 미감사*"**. **이 표는 논문 1편만 본다.** "그들 범위 밖"은
  **다른 선행이 덮는지에 대해 아무 말도 안 한다.** ★**v1이 "범위 밖"이라던 3칸이 실제로 겹쳤던 그 오류 구조가
  라벨에 그대로 남아 있었다.**
- ~~"미선점 1"~~ → **실은 "0 확정 + 1 조건부"**였고, **v3 계측이 그 조건을 죽였다가 v4가 되살렸다**
  (`T2_SELF_DECLARATION` = (b)버킷 축·존재증명 2건). **여전히 조건부** — n=2는 rate가 아니다.
- ⇒ **§0의 "스택은 안전"은 과잉 결론이다.** 정확히는 **"NabaOS 한 편에 대해서만 대체로 안전"**(§6-❺ 레인 감사 필요).

---

## 7. ★엔지니어링 발견 — **프레임과 독립·논문과 무관하게 수리 대상** (v5)
> 이 둘은 write-path 프레임이 죽어도 **살아남는다**. census가 우연히 꺼냈다(사용자 ❹의 성과).

### 7-1. ★★`_hint_hit`가 **밑줄 든 힌트를 원리적으로 매칭 못 한다** ⇒ A2 `identifying_arg_types` 불활성
```python
def _hint_hit(k, hints):                    # t2_gate_patch
    toks = [t for t in re.split(r"[^a-z0-9]+", str(k).lower()) if t]
    return any(t.startswith(h) for t in toks for h in hints)
```
- 힌트 `time_verified` → 인자명 `time_verified`의 토큰 = `['time','verified']` → **어느 토큰도 `time_verified`로
  시작할 수 없다**(토큰이 더 짧다) ⇒ **False**. 실행 확인 완료.
- ⇒ **banking A2의 `identifying_arg_types` = `['time_verified']` 단일 원소 = 100% 불활성.**
  `a2["_hints"]`에 **들어가 있는데 발화가 0**이다. `user_id`가 통과하는 건 힌트 **`id`**에 우연히 걸려서지
  `user_id` 때문이 아니다.
- **원인**: **2026-07-16 오탐 픽스**(`"id" in "provided"` 차단 → 토큰 startswith)가 **다중토큰 힌트에 거짓음성**을
  만들었다. 픽스가 만든 버그.
- **등급**: **[M] 코드 직독+실행**. **영향 = 선언은 있고 발화는 없음** = [[30]] `_f` 사고와 **같은 형태**
  (2026-07-16: 모델이 우리 도구를 본 적이 없었다). ⚠️**다음 도메인이 다중토큰 식별자를 선언하면 조용히 무시된다.**
- ⚠️**단, 이 버그의 *중요도*는 v4가 생각한 것보다 작다**(§2.1-v5 ①): `log_verification`은 우리 정본상 **procedural=무해**라
  못 봐도 설계 의도와 어긋나지 않는다. **버그는 메커니즘에 있지 그 사례에 있지 않다.**
- **픽스 설계(미적용·arm 수거 후)**: 단일토큰 힌트는 **현행 유지**(오탐 픽스 보존), 다중토큰 힌트만 **키 토큰열의
  연속 부분열 매칭**으로. ⚠️**게이트 거동이 바뀐다 = 모든 arm 비교 기준선이 바뀐다** ⇒ **진행 중 nt=20 수거 後
  적용**·픽스 전후 런 **혼용 금지**.

### 7-2. `record` = PROV 커버리지 공백 (프레임과 독립)
`_hint_hit('record'|'provided'|'amount'|'date') = **False**` ⇒ **라이브 PROV는 `record` 날조를 볼 수 없다.**
우리가 프로브서 **46%**로 잰 그 표면이다(`verify_identity`는 `_is_write`=**True**라 procedural 면제도 못 받는다).
⇒ **"우리가 46%로 측정한 실패를 우리 게이트는 런타임에 탐지할 수 없다"**(사용자 ❹) = **참**. 단 처방은 §2.1-v3 ⑤
(**LLM 복사 record → 원장 receipt 참조 결합**)가 더 근본적이다 — 표면 자체를 없앤다. hint 확장은 대증요법.

## 8. ⚠️★규율 — **인스턴스-엄격 · 프레임-동기추론** (2026-07-18 사용자 명명·오늘 4번째 변종)
> **패턴**: 인스턴스 수준에선 엄격했다(오탐 9건 per-case 배제·자기 버그 박제·반증조건 사전등록).
> **그런데 프레임 수준에선 증거를 교체해 프레임을 구조했다**:
> `record 46%` 죽음 → **기함을 `log_verification`으로 교체** → **프레임(write-path) 생존**.
> 그리고 **새 기함이 옛 기함과 똑같은 미검증 분류 결함**을 가졌다(v3: WRITE 목록 미확인 / v5: 우리 정본 미확인).
> ⇒ **프레임을 시험한 게 아니라 프레임에 맞는 증거를 찾았다.**
- **오늘의 4변종**: ①자기편향(v1) → ②과잉교정(검토자에 논증 없이 항복) → ③과잉기각("계측이 죽였다") →
  ④**프레임-동기추론**(기함 교체). **셋째까지는 방향만 다른 같은 병**이고, 넷째가 가장 위험하다 —
  **인스턴스 규율이 프레임 편향의 알리바이가 되기 때문이다.**
- **하드룰(이후 강제)**:
  1. **프레임을 랭크하기 전에 그 프레임의 *반증 예측*을 먼저 적는다** — *"이 프레임이 옳다면 X에 증거가 있어야
     한다"*. X에 없고 Y에만 있으면 **프레임을 버린다. 기함을 바꾸지 않는다.**
  2. **분류 주장은 정본을 *전수* 조회한다**(유리한 하나가 아니라). 이번 = tau2 env 타입 ✓ / 우리 `_is_write` ✗.
  3. **기함이 죽으면 프레임의 사망 가능성을 먼저 검토**하고, 교체는 **반증 예측을 통과한 뒤에만**.

## 6. 열린 질문 (사용자 리뷰 대상·v2)

1. ★**§0 게이트보다 먼저 할 게 생겼나?** v1은 *"우리 준수율을 재야 한다"*고 했는데 **그 근거(85% 걱정)가
   기우였다**(`349` = **포맷 발화율**). ⇒ 진짜 미측정 = **오픈웨이트 태깅 *정확도***(그들도 안 쟀다·`559-561`이
   *"if the model labels an inference as tool output"* = **준수했는데 틀리는** 경우를 자인). **재정렬 근거 없음 ⇒
   §0 게이트 우선순위 유지.**
2. **§4 학습 트랙**: 설계 신규성은 죽었다. 근거는 이제 **write-path(§2.1)+행동 축(§2.2/2.3)**뿐 ⇒ **게이트가
   통과해도 서사를 다시 써야** 한다. **write-path 분리 계측이 학습보다 먼저 아닌가?**(무료·기존 데이터 재분류)
3. **코어를 어디로?** — v1의 §3(게이트 역효과)은 **3순위로 강등**됐다. 후보 = **write-path**(§2.1) or
   **행동 축**([[16]] ASK/INFER-calibration = 그들이 `689`서 스스로 비운 곳).
4. **특허**: evidence 선언 계열 청구항이 있으면 **NabaOS가 선행**(`336` 프롬프트+`242` input_hash+`686` block).
   **`PATENT_ALIGNMENT` 재확인 필요** — 미실시.
