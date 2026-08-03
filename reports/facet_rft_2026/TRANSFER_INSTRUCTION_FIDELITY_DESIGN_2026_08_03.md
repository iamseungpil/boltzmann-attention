# 이관 지시 충실도 복원 설계 (2026-08-03 · rev3)

> ## ★rev3 머리말 — 진단이 바뀌었다 (코드 확인분)
>
> `gate_interpreter.py:37 notice_sent_in()` 은 **정확 일치가 아니라 정규화 후 앞 48자 부분문자열**이다:
> ```python
> key = notice_norm(notice_text)[:prefix]        # prefix=48
> return any(key in notice_norm(t) for t in texts if isinstance(t, str))
> ```
> docstring 축자: *"032=", Sofia" 1토큰 개인화가 전문-일치를 영구 불충족"* — 과거에 이미 한 번 물렸고
> prefix-48로 완화한 이력이다.
>
> **이 사실로 032 t0을 다시 읽으면 게이트는 설계대로 작동했다:**
>
> | 스텝 | 발화 | prefix 일치 | 게이트 |
> |---|---|---|---|
> | [14] | *"…Would you like to be transferred to a human agent now?"* | ✗ | deny |
> | [16] | *"…Would you like to be transferred…? Once you confirm…"* | ✗ | deny |
> | [18] | *"**TRANSFER NOTICE: I have checked the knowledge base**…"* | **✓** | allow |
> | [20] | `transfer_to_human_agents` 호출 | — | **성공** |
>
> ⇒ [14][16]은 *"지시를 무시한 재질문"이 아니라 **탐지자 prefix를 못 맞춘 실패 시도**"* 다.
> 034 t1은 [18]에서 마침내 맞췄으나 그 턴에 에피소드가 종료됐다.
>
> **따라서 rev1·rev2의 핵심 진단 — "우리 절대문이 행동을 질문으로 바꾼다" — 는 부분 철회한다.**
> 실제 비용은 **축자 탐지자를 맞추는 데 드는 턴**이고, 034형(사용자 요구 5회·호출 0)은 그 비용이
> 에피소드 예산을 초과한 결과다.
>
> **삭제 범위도 바뀐다**: `ask`의 *"send the user exactly this message first: …"* 는 **지울 수 없다**.
> 지우면 모델이 paraphrase → prefix 불일치 → `sent=False` 영속 → **이관 영구 차단**(034형 악화).
> 탐지자와 강결합이다.
>
> 아래 본문은 rev2 그대로 두되, **§4·§6·§11이 rev3으로 대체**된다.

# (rev2 본문 — 이력 보존)

> 근거: `AX33G_SPLIT_FORENSIC_2026_08_03.md` §7.5 · 런 `bank_ax33n_gpu{0,1}_20260803g`(64 sim 완주·front32×nt2)
> 원자료 `sim_results/bank_ax33n_gpu*_20260803g.results.json.gz`(커밋 `583af743`)
> **rev2 = 사용자 리뷰(A~G) 반영.** 반영 내역·수용/불수용 판정은 §11.

## 0. 한 문단

우리 A2 게이트 `GB2_NOTICE_BEFORE_TRANSFER`가 도메인 정책 §5·§6을 **절대문 하나로 평탄화**해
이관-차단 시 복구 메시지로 되돌려주고 있다. 정책은 두 개의 **탈출 조건절**(시나리오 override · 4회 임계)을
명시하는데 우리 재진술은 그중 하나를 **더 느슨한 다른 술어로 바꿔 산문에 묻었고**, 정책에 없는
**고정 템플릿**(검증 불가능한 완결성 주장)을 추가했다. 처방은 레버 추가가 아니라 **우리 저작분의 축소**다.
단 게이트 자체는 **측정된 조기-이관 21.2%를 겨냥해 만든 것**이므로 통째로 끄지 않는다.

## 1. 관측

**⚠교락 경고 (착수 판단의 근거로 쓰지 말 것)**: `ASKED` 상관(있음 n=29 pass .310 / 없음 n=35 pass .429,
총 43회)은 **처치 배정이 무작위가 아니다**. `ask`는 게이트 deny 시에만 발화하고 deny는 에이전트가 이관을
시도했을 때만 난다 ⇒ **이미 곤경에 빠진 sim에서 더 자주 발화**하므로 인과 없이도 상관이 나온다([[08]]).
이 수치는 **동기 부여**일 뿐이며, 인과는 §6의 A vs B″ 짝비교만이 준다.

**설계의 실제 근거는 아래 궤적·코드·정책 대조(§2)이고, 통계와 독립적으로 성립한다.**

| 관측 | 근거 |
|---|---|
| 문구가 8개 이상 sim에서 축자 동일 | 014t0·032t0·032t1·033t0·033t1·034t0·034t1·035t1 |
| 032 t0: 사용자가 [9][11][13] 이관 요구 → 에이전트가 [14][16][18] **3회 더 재질문** | 궤적 정독 |
| 034 t1: 사용자 요구 5회 · **도구 호출 0회**로 종료 | 궤적 정독 |
| 035 t0: 엔진이 *"Proceed immediately with transfer_to_human_agents"* → KB 검색 2회 더 | 궤적 정독 |

## 2. 원인 — 게이트 발화 지점과 정책 대비 차이 (코드 확인분)

### 2.1 발화 지점 = 호출 차단 시 복구 메시지 (상시 주입 아님)

`gate_interpreter.py:137 render_recovery()` — `ask`의 **유일한 런타임 소비처**(:153 `elif gate.get("ask")`).
나머지 참조는 faithfulness/roundtrip 검사·spec_sampler 기본값이라 런타임 주입이 아니다.

```
blocked by policy gate: {predicate} not established.
Recovery: (1) do NOT retry this tool now; (2) {ask}; (3) once this is done, retry the original action
```

`applies_to = ["transfer_to_human_agents"]` · `satisfiers = {}` (비어 있어 `ask` 분기로 떨어짐).

⇒ 이 문구가 뜨는 순간은 **에이전트가 이미 이관을 결심하고 호출한 시점**이다. 거기서 우리가
*"never without asking the user first"* + 고정 템플릿을 돌려주니 **이미 사용자가 요구한 이관을 다시 묻게 만드는
구조**가 된다. 032 t0가 정확히 그 형태다. (rev1의 "system prompt 대비 최신성 우위" 설명은 **철회** — 근거가
추론에서 코드로 승격됐다.)

### 2.2 정책 대비 차이

**도메인 정책** `prompts/components/policy_header.md` — 에이전트 system prompt에 포함됨(results.json `info`에서 확인):

> **§5** *"…**If they do, invoke the appropriate transfer_to_human_agents tool.** … Do not transfer without asking
> the user first. **This guidance may be overridden by specific scenario-based transfer guidance in the knowledge base.**"*
> **§6** *"**If the user asks for a human agent 4 times, then you may invoke the transfer_to_human_agents tool.**
> This guidance may be overridden by specific scenario-based transfer guidance in the knowledge base."*

| 정책 요소 | 우리 A2 `gates[1]` | 판정 |
|---|---|---|
| "Do not transfer without asking the user first" | `ask`에 재진술 | [[23]] 통과 — 날조 아님 |
| **"may be overridden by … knowledge base"** (§5·§6 **2회**) | **없음** | **손실** |
| "asks for a human agent **4 times** → may invoke" | **대응물 있으나 다른 술어**: *"If the customer has already agreed to the transfer (or already asked for it) … immediately CALL"* = **≥1회** (정책은 ≥4회) · **산문** · **미집행** | **부재 아님 — 배치·집행의 문제** |
| `notice_text` 고정 템플릿(*"…there are no further actions I can take"*) | 있음 | **우리 저작 · 정책에 없음 · 검증 불가능한 완결성 주장** |

세 가지가 겹쳐 실패한다: ① 절대문 + 산문 예외절(모델은 절대문을 붙잡음·[[42]]) ② 완결성 주장이 탈출을 정당화
(032/033에서 **거짓** — 문서화된 `initial_transfer_to_human_agent_0218`이 존재) ③ 차단 시점에 재질문을 지시.

**7월에 이미 탈출로로 실측** — `t2_gate_patch.py:6154` 축자: *"038 실측: …→**TRANSFER NOTICE로 탈출**"*. 미조치.

### 2.3 ★게이트를 통째로 끄면 안 되는 이유

`gates[1]._note_gb2` 축자: *"**EARLY_TRANSFER 36 sims(21.2%·gold-밖 transfer) 표적**"* ·
정책-사실 근거 = policy §5 원문 인용. ⇒ **이 게이트는 측정된 조기-이관을 겨냥해 만들어졌다.**
비활성화는 그 21.2%를 되살린다.

## 3. ★[[05]] 결정-시점 3질문 ([[17]] 상설)

**채택안 B″-1′(§4.1)에 대해:**

| # | 질문 | 답 |
|---|---|---|
| 1 | scaffold **또는 A2**의 도메인-특화를 *순증*시키나? | **NO — 감소.** 우리 저작 산문(절대문 재진술·절차 지시)을 **삭제**하고, 추가하는 것은 정책 §5·§6이 **명시한 탈출 조건**뿐이다. 정책 축자이므로 새 도메인 판단 0. 총 A2 문자수는 줄어든다. |
| 2 | 모델이 할 수 있는 *유동적 판단*을 결정론에 *동결*하나? | **NO — 해동한다.** 우리가 동결해 둔 "언제 이관하는가"에 정책 자신의 탈출 조건을 되돌려준다. |
| 3 | scaffold가 모델 대신 *도메인 행동을 수행*하나? | **NO.** 게이트는 거부만 하고 도구를 부르지 않는다. 엔진 코드 변경 0. |

⇒ 셋 다 NO. **기본=GO.** ([[05]] 메타 경고 준수 — "A2로 옮김=제거"의 shell game이 아니라 실제 순감.)

**2차(§4.2 가드 조립)**: Q1 **부분 YES**(조각·가드가 A2에 추가) ⇒ **기본=NO. 1차 측정이 손실을 보일 때만.**

## 4. 설계

### 4.1 1차 = B″-1′ (채택)

| 안 | 내용 | 판정 |
|---|---|---|
| B″-1′ | `notice_text` **유지**(아래 이유) · `ask`에서 절대문 재진술·절차 지시·산문 예외절 **삭제** · 정책의 탈출 조건을 **게이트 적용 조건**으로 이설 | **채택** |
| B″-2 | `gates[1]` 전체 비활성 | **불채택** — §2.3의 21.2% 조기-이관을 되살림 |
| B″-3 | `ask`만 비우고 deny 유지 | **불채택** — 복구 경로 없는 차단 |

**★`notice_text`를 지우지 않는 이유(리뷰 F 확인분에서 도출)**: 이 문자열은 **지시이자 동시에 탐지자**다.
`t2_gate_patch.py:105`가 `a2["_notice_text"]`를 파생하고, `_transfer_msg_sent`(:602)·`_regen_transfer_sent`(:3077)가
`if not notice_text: return None`으로 **"판정 불가"** 를 낸다. 즉 지우면 "고지가 전달됐는가"가 **닫힌 술어에서
열린 술어로 바뀌고**, 게이트는 뒷문으로 무력화된다. **고정 문구는 자의적 장식이 아니라 이 술어를 닫는 장치다**([[22]]).

**삭제 대상(정확히)**
- `ask` 중: *"transfer only if you absolutely have to and you are SURE …"*(절대문 재진술) ·
  *"send the user exactly this message first"*(절차 지시) · *"send the notice at most ONCE / do NOT ask again"*(산문 예외절)
- 남길 것: 복구 (2)단계가 비지 않도록 **한 문장**만 — 도메인 어휘 0·지시 아닌 서술.

**게이트 적용 조건으로 이설(정책 축자 유래)**
```
applies_when:  ¬scenario_override_retrieved        # 정책 §5·§6 "may be overridden by …"
```
즉 시나리오 문서가 회수됐으면 **이 게이트는 적용되지 않는다**. 정책이 그렇게 말한다.

**`term_grant_reminder_extra` = 삭제 대상 아님**(리뷰 E 판정): 문구가 *"Do not ask for more identifiers —
identity verification is not required for a transfer"* 로 **과잉 질문을 억제하는 방향**이고, `_note_reminder_extra`에
[[23]] provenance(도메인 verify note)가 기재돼 있다. **유지**하되 provenance 재확인은 별건.

### 4.2 2차 = 가드된 조각 조립 (조건부·가드 1개로 시작)

1차가 조기-이관을 유발하면(§6 실패 기준) 그때만. **가드는 G2 하나로 시작**한다 — G1(4회 카운터)은 §2.2에서
보듯 이미 산문 대응물이 있어 1순위가 아니다(리뷰 G 수용).

```
guard G2 := scenario_override_retrieved     # 문서 id·도구명 대조 = 닫힌 술어
```

엔진은 **저작된 조각을 고를 뿐 문장을 만들지 않는다** ⇒ [[10]] 선택기 역할 유지.

## 5. 가드의 형식 제약 (2607.22868 정리 6·7)

정리 6: 감소 카운터 2개면 명세 비자명성 **결정 불가** — 가드 상호참조 시 "이 조각이 켜지기는 하는가"를
정적으로 판정할 수 없다. 정리 7 축자: *"Each guard tests one counter … against a binary-encoded constant,
**never another counter**"* + 분리성·키-지역성·컴포넌트 전면성.

**하드 규칙**: ① 단일 플래그/카운터 vs 상수 ② **가드 간 상호참조 금지** ③ 전면성 ④ **조각·가드 수는
측정된 것만 늘린다**(2^N 미검증 구성 방지).

**비-가법성**: [arXiv:2604.14862](https://arxiv.org/pdf/2604.14862) 축자 *"the two channels interact
**non-additively**"* ⇒ 조각별 검증이 조립을 보증하지 않는다. **N=1로 시작**.

## 6. 측정

| arm | 내용 |
|---|---|
| **A** | 현행 |
| **B″-1′** | §4.1 |

- 조건: **front32 × nt2**(현 런과 동일·비교 가능). ASKED 발화 29 sim은 **전부 front32 안에 있다**
  (이 런의 64 sim이 모두 front32이므로 — §10-3 종결).
- 1급 지표: `ASKED` 발화 수 · `transfer_to_human_agents` 호출률 · pass.
- **Δspurious(필수)**: **gold-밖 이관** 증가량. `_note_gb2`의 21.2%가 기준선이다. 증가하면 §4.2로.
- 판정 전 **궤적 전수 포렌식**([[08]]).

## 7. 이 설계가 고치지 **않는** 것

- **liveness**: "결국 도구를 호출해야 한다"는 pre-call 게이트로 집행 불가(2607.22868 정리 1·명제 1).
  사용자가 `###STOP###`으로 끝내는 sim(003 t0·034 t1)은 걸 자리가 없다.
- **회수 공백**: 032/033은 `initial_transfer_to_human_agent_0218`이 궤적에 한 번도 없다. 삭제해도 이름을
  모르면 못 부른다. **레지스트리×문서 조인 병행 필요**([[19]]).
- 018/020/021의 인계 와이어 포맷·전송 손실(별건).

## 8. 위험·선행 부정 증거

- **조기 이관 재발** — §2.3의 21.2%. Δspurious가 잰다.
- **[[07]]** 삭제는 soft 조작이라 보장이 아니다. 잔여는 측정으로만.
- **[[42]]** 이 설계는 *준수를 얻으려는 것이 아니라 우리가 만든 간섭을 걷는 것*이다. 구분을 흐리지 말 것.
- **서브에이전트 대안 불채택**: `RATE_SUBAGENT_DESIGN_2026_07_18` §2d 짝비교 반증(서브 요청 **172,731 토큰**
  초과 → 폴백 → 47/47 판정불가가 discrepant 0으로 **위장**). C6 fleet 저-ROI.

## 9. 구현 표면 ([[24]]) · 회귀 대상

- 수정 = **정본 층** `a2/banking_knowledge.settings.json` + `split/banking_knowledge.core.json`.
  `gate.json`은 생성물 — 직접 수정 금지(2026-08-02 P9 死코드 재발 방지).
- 편집 후 **`load_domain_a2()` 병합 확인** + `test_a2_three_layer.py`.
- 엔진 코드 변경 **0**(1차). `applies_when`은 기존 도메인-일반 멤버십 가드를 재사용한다.
- **회귀 대상 3개소**(`notice_text` 의존): `t2_gate_patch.py:105` 파생 · `_transfer_msg_sent`(:602) ·
  `_regen_transfer_sent`(:3077) · 및 6109~6117. **1차에서 `notice_text`를 유지하므로 이 경로는 불변**이지만,
  회귀 테스트에 포함해 `None`(판정 불가) 반환이 발생하지 않음을 확인한다.
- `scaffold_guard.py` 감시 대상에 A2 조각 저장소 추가 = **§4.2 착수 시 선행 조건**.

## 10. 열린 항목 — 상태

1. **닫힘** — 주입 지점 = `render_recovery` 복구 메시지(코드 확인). §2.1 반영.
2. **닫힘 · 리뷰와 다른 결과** — 동형 문구는 **재현되지 않는다**:
   `retail.settings.json` `notice_text`=1이나 `'TRANSFER NOTICE'`=0·`'without asking'`=0 ·
   `airline.settings.json` `notice_text`=**0**. ⇒ **이 손실은 banking 단독**이고 3도메인 동시 수정은 불필요.
   (구조는 공유하나 내용이 복제되지 않았다.)
3. **닫힘** — 이 런의 64 sim이 전부 front32이므로 ASKED 29 sim ⊂ front32.

## 11. 리뷰 반영 원장

| 항목 | 판정 | 반영 |
|---|---|---|
| **A** 주입 지점 = deny 복구 메시지 | **수용**(코드 확인) | §2.1 재작성·rev1의 "최신성 우위" 철회 |
| **A′** 삭제 형태 사전 선택 · 권고 B″-2 | **부분 수용** — 형태 선택은 수용, **B″-2는 불채택** | §4.1 표. 근거 = `_note_gb2` **EARLY_TRANSFER 21.2%** · `notice_text`의 탐지자 이중역할 |
| **B** 3도메인 반복 | **불수용(반증)** | §10-2 — retail/airline에 해당 문구 없음 |
| **C** ASKED 통계 교락 | **수용** | §1 상단 경고·§0에서 수치 제거 |
| **D** "4회 손실" 과대 | **수용·정련** | §2.2 표 — "부재"가 아니라 **다른 술어(≥1 vs ≥4)·산문·미집행** |
| **E** `term_grant_reminder_extra` 판정 누락 | **수용** | §4.1 — **삭제 대상 아님**(과잉질문 억제 방향·provenance 기재) |
| **F** `notice_text` 소비 3개소 | **수용·결론 반전** | §4.1 — 이 사실 때문에 `notice_text`를 **유지**(닫힌 술어 유지)·§9 회귀 대상 |
| **G** 가드 1개로 시작 | **수용** | §4.2 — G2 단독 |

---

# rev3 — §4·§6·§11 대체본

## 4′. 설계 (rev3)

### 4′.1 재리뷰 신규 지적 수용 — `applies_when` 스키마 불일치

`gate_interpreter.py:263 _gate_applies` 확인:
```python
aw = g.get("applies_when")
if aw:
    v = str((args or {}).get(aw.get("arg")) or "")     # ← 호출 인자만 본다
```
docstring 축자: *"applies_when: {"arg": <인자명>, "in": [...]} 또는 {"arg": ..., "not_in": [...]}"*.
⇒ `scenario_override_retrieved`(대화 이력 상태)는 **현행 스키마로 표현 불가**. rev2의 B″-1′는 엔진 스키마
확장(Q1/Q3 재감사)을 요구하므로 **1차로 분류하면서 2차 비용을 내는 것**이 맞다. **지적 수용 — 철회.**

### 4′.2 채택 = B″-1″ (순수 삭제형·`applies_when` 미변경)

| 대상 | 조치 | 근거 |
|---|---|---|
| `notice_text` | **유지** | 탐지자 겸용(`sent=None`→게이트 무발화·`:295 sent is False`만 deny) · `_note_gb2` EARLY_TRANSFER 21.2% 표적 |
| `term_grant_reminder_extra` | **유지** | 과잉질문 억제 방향 · `_note_reminder_extra`에 provenance |
| `ask` ─ *"transfer only if you absolutely have to and you are SURE …"* | **삭제** | 정책 §5의 중복 재진술 — system prompt에 이미 있음 |
| `ask` ─ *"Search the knowledge base and attempt every applicable procedure … when (and only when) nothing remains"* | **삭제** | 절차 지시([[05]] Q2) · 검증 불가능한 완결성 주장의 근원 |
| `ask` ─ *"send the user exactly this message first: <notice_text>"* | **★유지(rev3 반전)** | 탐지자 prefix-48 강결합. 지우면 이관 영구 차단 |
| `ask` ─ *"IMPORTANT: send the notice at most ONCE … immediately CALL transfer_to_human_agents"* | **유지** | 유일하게 *호출*을 향해 미는 문장. 미집행이나 제거 시 순손실 |
| `applies_when` | **미변경** | §4′.1 |

⇒ `ask`는 **"이 문장을 먼저 보내고, 보냈으면 즉시 호출하라"** 만 남는다. 절대문·절차 지시·완결성 주장 제거.
[[05]] 3질문: Q1 **NO(순감)** · Q2 **NO(해동)** · Q3 **NO(엔진 변경 0)**.

### 4′.3 신규 후보 레버 — 탐지자 비용 (별건·본 arm에 포함하지 않음)

rev3 진단이 가리키는 진짜 비용은 **prefix-48을 맞추는 데 드는 턴**이다. 후보:
`notice_sent_in`의 `prefix` 축소(48 → 더 짧게) 또는 키-토큰 집합 매칭.
- 도메인-일반 엔진 파라미터 · A2 도메인 리터럴 0 ⇒ [[05]] 깨끗
- ☠**반대 방향 비용**: 느슨해질수록 "고지 없이 열림"이 늘어 **EARLY_TRANSFER 21.2%로 회귀**
⇒ **별도 arm으로만.** 본 측정에 섞으면 교락.

## 6′. 측정 (rev3) — 단독·사전등록

재리뷰 권고 수용: **단독 arm**. [[19]] 합성-우선은 *실증된* 레버를 함께 켜라는 규율이지 미완성 설계를
끼워 넣으라는 것이 아니며, 이번 런의 존재 이유가 "ASKED 상관이 교락됐으니 짝비교로 인과를 얻는다"이므로
새 교락을 넣을 수 없다.

| arm | 내용 |
|---|---|
| **A** | 현행 |
| **B″-1″** | §4′.2 |

**조건**: front32 × nt2(현 런과 동일).

**사전등록 예측(박제)**
1. **032·033은 이 수정만으로 뒤집히지 않는다** — `initial_transfer_to_human_agent_0218`이 궤적에 한 번도
   나타난 적 없다(회수 공백·§7). 예측이 맞으면 회수-공백 설계의 근거가 되고, 틀리면 그 우선순위를 내린다.
   **어느 쪽이든 정보를 얻는다.**
2. 판정 표적 = **014·034·035** + `ASKED` 발화 수 + **부당 이관 Δspurious**(기준선 = `_note_gb2` 21.2%).
3. rev3 진단이 옳다면 `ASKED` 감소분의 상당 부분은 *실패한 prefix 시도*의 감소로 나타나야 한다 —
   **prefix 일치/불일치를 분리 계측**한다(불일치 발화 수 = 탐지자 비용의 직접 측정).

판정 전 **궤적 전수 포렌식**([[08]]) — 집계 직행 금지.

## 11′. 재리뷰 반영 원장 (rev3)

| 항목 | 판정 | 반영 |
|---|---|---|
| B(3도메인) 철회 | 확인 | retail `notice_text`=*"YOU ARE BEING TRANSFERRED…"*(무관)·airline notice 게이트 부재 |
| A′ 다리 1·2 수용 | 확인 | §2.3·§4′.2 유지 |
| **신규: `applies_when` 스키마 불일치** | **수용** | §4′.1 — B″-1′ 철회, **B″-1″** 채택 |
| 합성 vs 단독 | **수용** | §6′ 단독 + 032/033 미전환 사전등록 |
| D 정련 유지 권고 | 수용 | §2.2 표 그대로 |
| **★설계자 신규(rev3)**: `notice_sent_in`=prefix-48 | — | 머리말·§4′.2 — *"send exactly this message"* **삭제 불가**로 반전 · §4′.3 신규 후보 레버 |

⚠**한 가지 남는 불편**: rev3 진단이 옳다면 이 수정의 기대 효과는 rev1/rev2가 예상한 것보다 **작다**.
절대문·절차 지시를 지워도 탐지자 비용은 그대로다. 그래도 진행하는 이유는 (a) 우리 저작분 축소는
그 자체로 [[05]] 정합이고 (b) 이 arm이 **탐지자 비용을 분리 계측**해 §4′.3의 필요성을 판정해 주기 때문이다.
효과 크기에 대한 기대를 낮춰 사전등록한다.
