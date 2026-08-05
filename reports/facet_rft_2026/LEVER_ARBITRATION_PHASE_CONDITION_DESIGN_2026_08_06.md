# 레버 조정 = 단계·조건별 발화 설계 (2026-08-06 · rev2)

> 사용자 지적(2026-08-06): *"조정이라는 게 단계별 조건별로 지시하는 거 아니었나?"*
>
> 맞다. 그리고 그건 **이미 있는 물건**이다 — C17 `t2_phase`(2026-08-05)가 그 설계다.
> 이 문서는 새 중재 규칙을 발명하지 않는다. **C17이 선언만 하고 배선하지 않은 칸을 채우는 설계**다.
>
> **rev2(리뷰 반영)**: 초판의 처방부 결함 6+8건을 반영했다. 가장 큰 변경 셋 —
> ① **E3-③(면허 조건)은 철회**한다(구조적 no-op이었다·§5.2) ② **§4.3 선언표는 전부 실측으로 대체**했고,
> 결과는 초판 가설 대부분이 **그 레버를 끄는 것**임을 보인다(§4.3) ③ 모든 수치는 재실행 가능한
> 스크립트 [`x104_lever_arbitration_census.py`](../../scripts/distill/tau2/x104_lever_arbitration_census.py)
> 로 산출·영속했다(등대 §갱신 프로토콜).
>
> 데이터 = `bank_smk_gpu{0,1}_20260806a`(런 a·12 태스크·**nt=1**) + 사이드카 `fb_20260806a.jsonl.gz`.
> 전판 = `HANDOFF_2026_08_06.md` §5.

---

## 0. 문제 진술

같은 세션에서 세 번, **우리 문구가 우리 문구를 밀어냈다**:

| 사례 | 밀어낸 쪽 | 밀린 쪽 | 결과 |
|---|---|---|---|
| 012 (n 통과 → a 실패) | C8 레지스트리 문구 | `usertoolnote` | 통과를 만들던 문구가 영영 미발화 |
| 035 (p 통과 → a 실패) | C17 단계 소유권(유력·미확정) | FOLLOW-UP ×5 | 이관 완결을 밀던 압력 소실 |
| **022 (n 통과 → a 실패·이번 확정)** | §2의 3건 | | 턴 소진 → 이관 → DB 미충족 |

문구를 하나씩 고치는 대응은 이미 실패했다 — **충돌하는 쌍이 런마다 바뀌기 때문**이다(C17 docstring).
필요한 것은 문구 수정이 아니라 **누가 언제 말할 자격이 있는가**의 선언이다.

## 1. 실측 ① — C17은 설계의 1/N만 배선돼 있다

| 설계된 것 | 실제 배선 | 근거 |
|---|---|---|
| 단계 5종 `verify·discover·procedure·decide·open` | **`discover` 반환 분기 없음**. `procedure`/`decide`는 `procedures_state` 인자가 필요한데 유일 호출부가 미전달 ⇒ **실효 단계 = {verify, open}** | `t2_phase.py:48-75` · `t2_gate_patch.py:5441` |
| `owns(phase, lever_phases)` | **모듈 밖 호출 0건 = 死 API** | grep |
| 단계 소유권 | 레버 **1개**(action-push) × 조건 **1개**(`phase=="verify"`) | `T2_PHASE_OWNER` |

## 2. 실측 ② — 022가 받은 지시 (축자)

**(a) turn 2 — 무관한 표적을 3회 밀었다.** `[ACTION] 'submit_transaction' is run by the CUSTOMER …
STOP searching.` 022는 리워드 불일치 감사 태스크이고 `submit_transaction`은 이 대화에 등장한 적이 없다.
같은 turn에 `GB1` deny가 *"먼저 신원을 검증하라"*를 보내고 있었다. 표적은 우리 formalize가 골랐다.

**(b) turn 30 — 정반대 지시가 3사이클 반복.**
```
[VALUE-ACQUIRE] … give get_card_last_4_digits to the customer NOW …
[PROCEDURE]     The policy forbids 'get_card_last_4_digits' in this procedure - verbatim:
                "Do not collect sensitive card details; the tool uses the identifiers provided by the user."
[deny]          resolve the flagged call(s) first; do not call this tool yet.
```
**엔진은 두 사실을 이미 쥐고 있었다**: `cash_back_dispute` 절차가 `get_reward_discrepancies` 실행으로
활성이었고, 그 절차의 `prohibits[get_card_last_4_digits]`가 정책 축자와 함께 선언돼 있다. 금지는
**호출 시점에만** 평가되고(`t2_procedure.decide`), 도구 이름을 **말하는 문구**는 아무도 검사하지 않는다.

**(c) turn 32 — 우리 문구가 가르친 것을 우리 문구가 벌했다.** FOLLOW-UP이
*"call give_discoverable_user_tool … once per discrepant transaction (with … as arguments)"*라 했고,
SIGNATURE가 *"give takes only discoverable_tool_name; you also passed arguments"*로 막았다.
give는 인자를 받지 않고, 거래마다 반복하는 쪽은 손님의 `call_discoverable_user_tool`이다. **오기다.**

이후: 손님이 *"나는 도구를 못 쓴다"*로 굳음 → 이관 요청 → 이관 → DB 미충족.
(`n`에서는 같은 user-sim이 스스로 10회 호출해 통과했다.)

## 3. 실측 ③ — 레버별 노출·공발화 (산출: `x104 §A/§B`)

**계기 규약**(재현에 필요·초판 누락): 우리 문구 = 사이드카 `kind ∈ {reminder-user, tool-deny}`
(283/414행). `reminder-assistant`(131행)는 **모델이 재생성한 답변**이라 제외한다 — 섞으면 우리 문구 수가
부풀어 재현이 깨진다. 레버 정체는 본문 앞 120자의 `[TAG]`, 없으면 채널명(`unified_regen 73` 행이
그 잔여다).

| 레버 | 발화 | 레버 | 발화 |
|---|---|---|---|
| unified_regen(태그 없는 잔여) | 73 | TOOL-CHANNEL | 10 |
| CLAIM-PROVENANCE | 48 | usertoolnote | 9 |
| DISCOVERY-REQUIRED | 26 | WRITE-EVIDENCE | 7 |
| FOLLOW-UP | 19 | UNLOCKED-NOT-CALLED / givequote / GIVE-EXEC | 5 |
| SIGNATURE | 16 | **ACTION** | **3**(022 전용) |
| PROCEDURE / VALUE-ACQUIRE | 15 / 15 | 그 외 8종 | ≤4 |

공발화 상위: `SIGNATURE+VALUE-ACQUIRE 8` · `PROCEDURE+TOOL-CHANNEL 7` · `CLAIM-PROVENANCE+FOLLOW-UP 7`
· `VALUE-ACQUIRE+usertoolnote 5` · **`PROCEDURE+VALUE-ACQUIRE 2`**(=§2(b)).

⚠ 통과/실패 분할은 **교란**이다(실패 sim이 길어 기회가 많다). 표적 크기 측정에만 쓰고 인과는 §2 궤적에서만.
⚠ `x95` **정적** 감사는 §2(b) 쌍을 못 잡는다 — 두 문구는 어휘로 대립하지 않고 **도구 이름 하나**로
대립하며 그건 런타임에만 보인다. 계기 구멍이고 §5-E5가 그 대응물이다.

---

## 4. 설계 — 조정의 단위는 (레버 × 단계 × 조건)

### 4.1 세 층은 성격이 다르다

| 층 | 묻는 것 | 출처 | 판정자 |
|---|---|---|---|
| **단계**(phase) | 지금 이 대화가 어느 국면인가 | A2 선언 + 호출 이력 | `t2_phase.phase_of` |
| **조건**(condition) | 이 표적을 **지금 말해도 되는가** | A2 `procedures[].prohibits` | 발화-직전 중재 |
| **문구**(text) | 말하는 내용이 사실인가 | **정책 축자** | 오기 수정(§5-E4) |

### 4.2 레버는 표적 출처로 갈린다 — 조건을 잘못 걸면 금지를 집행한 문장이 사라진다

| 분류 | 표적을 누가 골랐나 | 예 | 조건 |
|---|---|---|---|
| **push** | 우리가 골랐다 | ACTION · VALUE-ACQUIRE · DISCOVERY-REQUIRED · FOLLOW-UP · UNLOCKED-NOT-CALLED | 활성 절차가 그 표적을 `prohibits`하면 **침묵** |
| **react** | 모델이 골랐다(자기 호출에 대한 판정) | SIGNATURE · TOOL-CHANNEL · PROCEDURE-deny · WRITE-EVIDENCE · CLAIM-PROVENANCE | 금지 조건 **대상 아님**. 같은 turn에 push와 겹치면 **react 우선** |

**이 구분은 장식이 아니다.** 초판 계기(x104 v1)는 표적을 따옴표로 추출했는데, VALUE-ACQUIRE 문구는
도구명을 맨몸으로 쓴다(*"…running get_card_last_4_digits"*). 그 결과 **표적을 놓치고 대신 그 도구를
인용한 PROCEDURE deny 자신을 침묵 대상으로 지목**했다 — 조건을 그대로 배선했다면 **금지를 집행하던
문장을 지우고 금지된 권유는 남겼을 것**이다. 정반대 처방이다. 그래서 x104는 표적을 **env 레지스트리와의
닫힌 대조**로 뽑고([[22]]), push/react를 명시 분류한다([[55]]: 계기는 부정통제 없이 신뢰 금지).

### 4.3 단계 선언표 — **측정된 칸 1개 + 가설 칸 N개** (산출: `x104 §D`)

발화 283건이 어느 단계에서 났는지 전수 계산했다. **현행 배선에서는 97%가 `open`이다**:

| 배선 | 단계 분포(발화 283) |
|---|---|
| **현행** | `open 275` · `verify 8` |
| **+state**(E2-①) | `open 197` · `decide 51` · `procedure 27` · `verify 8` |
| **+discover**(E2-② 가설 정의) | `open 191` · `decide 51` · `procedure 27` · `verify 8` · **`discover 6`** |

이 표를 초판 §4.3의 선언 가설에 대입하면:

| 레버 | 초판이 선언하려던 단계 | **그 선언의 실제 효과(+state 기준)** | 판정 |
|---|---|---|---|
| DISCOVERY-REQUIRED | `discover·procedure` | 발화 26건 중 그 단계 = **0** (open 21·decide 5) | ❌ **레버를 끄는 것** |
| VALUE-ACQUIRE | `procedure` | 15건 중 **3**만 생존, 12건 침묵 | ❌ 표적(022 3건)과 무관한 12건을 지운다 |
| FOLLOW-UP | `decide·open` | 19건 중 11 생존, `procedure` 8건 침묵 | ⚠ 근거 없음 |
| **ACTION** | `procedure·decide·open` | 3건 전부 `open` → 생존. 다만 `discover` 추가 시 새 침묵 면적 발생 | ⚠ **C17 등가는 `NOT verify`** |

⇒ **초판 §4.3 표는 폐기한다.** 남는 규칙은 둘뿐이다:
1. **측정된 칸 = ACTION의 `NOT verify` 1개**(C17 실측: 손해 9건 삭제·통과 노출 0). 선언은 단계 **열거**가
   아니라 **제외**로 적는다 — 열거는 `discover`가 추가되는 순간 조용히 침묵 면적을 넓힌다.
2. 나머지 레버는 **선언 없음(=항상 발화)**을 유지한다. 각 칸은 위 분포에서 생존 수를 먼저 계산하고,
   그 수가 표적과 일치할 때만 켠다.

⚠ `discover`의 분포(6건)는 **내 가설 정의**(활성 절차가 이름 댄 단계 도구 중 미-unlock)에서 나온 값이다.
정의를 넓히면(비활성 절차 포함) 분포는 커진다 — **정의가 곧 침묵 면적**이므로 E2-②는 정의를 먼저 고정하고
분포를 다시 재야 한다. 리뷰가 우려한 "discover가 대부분을 먹는다"는 이 정의에서는 일어나지 않았지만,
그건 정의 선택의 결과이지 안전 보장이 아니다.

---

## 5. 구현안

### 5.1 채택

- **E5(먼저) 동적 공발화 감사** — `x104_lever_arbitration_census.py`. **이미 구현·이 문서의 모든 수치가
  그 산출물이다.** `x95`의 정적 패스와 상보. 이후 모든 Δ는 이 계기로 잰다.
- **E4 오기 수정.** FOLLOW-UP 문구에서 *"once per discrepant transaction (with … as arguments)"* 제거.
  **출처 = 정책 축자**(`_note_tool_signatures`: *"Use the give_discoverable_user_tool(discoverable_tool_name)
  function"* + *"인자는 유저에게 말로 설명하라"*). ⚠초판이 "env 스키마가 출처"라 한 것은 **틀렸다** —
  env는 반대를 말한다(`env_surface.json:377` `give_discoverable_user_tool.args =
  [discoverable_tool_name, arguments]`). 1인자 제약은 정책이 출처이므로 수정은 [[23]] 통과이고, "env라서 정책 해석 불요"라는
  면제 논리는 성립하지 않는다.
  **동기화 의무**([[24]] 양방향): 이 문자열은 `banking_knowledge.specific.json:418` ·
  `banking_knowledge.gate.json:593` · `a2/split/banking_knowledge.core.json:390` **3곳**에 있다.
  정본 수정 → gate.json 바이트 동일 복사 → split 재생성 → `load_domain_a2()` 병합 확인 +
  `test_a2_three_layer.py`.
- **E3-② 금지 조건(단독).** 발화 직전에 push 레버의 표적을 활성 절차의 `prohibits`와 대조해 침묵.
  **speak-time 계약**(초판의 "새 판정 로직 0"은 철회): `prohibited(procs, names, executed)`는 `names`가
  절차 트리거이면 **그 언급만으로 절차를 활성 취급**한다. 호출 시점의 `names`는 *모델이 지금 부르는 도구*
  였지만 발화 시점의 `names`는 *우리가 권하는 표적*이라 의미가 다르다. ⇒ speak-time에서는
  **활성 판정을 `executed`로만** 하고, 표적은 **금지 대조에만** 쓴다(x104가 그 규약대로 계산한다).

### 5.2 철회

- **E3-③(면허 조건) 철회.** *"표적 ∈ A2 선언 ∪ 회수 텍스트"*는 **구조적 항등 통과**다:
  ACTION의 후보 집합 자체가 A2 선언이고(`t2_gate_patch.py:5364` `_acts = a2["action_tools"]` →
  `formalize_intent_tool(..., 후보=선언)`), `submit_transaction`은 그 선언에 있다
  (`banking_knowledge.settings.json:156`). 즉 022의 3발은 **지워지지 않는다** — 초판 §6.1의
  "ACTION 3발 전부 침묵"은 **사실이 아니다.** 철회하고 §6으로 이관한다.

### 5.3 보류(측정 선행)

- **E2 `phase_of` 완성** — ① `procedures_state` 전달(기존 함수 조합: `active_procedures` + `next_step`)
  ② `discover` 분기. **정의 고정 → x104 §D 재측정 → 그 다음.**
- **E1 `owns()` 실배선** — §4.3에 따라 켤 칸이 현재 **1개**(ACTION `NOT verify`)뿐이다. 그마저 C17이
  이미 하고 있으므로, E1의 실익은 "같은 규칙을 선언으로 옮기는 것"이다. **035의 C17 귀속 판정
  ([[18]] 격리) 전에는 켜지 않는다.**
- **`speak()`의 4단계는 각각 독립 플래그**여야 한다(단계 / 금지 / 우선순위 / (철회된)면허).
  하나의 관문으로 묶어 넣으면 E3-②만 원해도 E1이 함께 켜져 §4.3의 대량 침묵이 그대로 발생한다.
- **관문 위치**: 각 레버 자리가 아니라 `fb` 리스트가 모여 나가는 지점(`t2_gate_patch.py:6063-6069`,
  사이드카가 기록되는 바로 그 자리)이 가장 싸고 누락이 없다. 단 그 자리에는 **레버 정체·표적
  메타데이터가 없다** — 태그 문자열 파싱에 의존하게 되는데, 그 취약성은 이번에 실물로 확인됐다
  (§4.2: 계기가 표적을 반대로 지목). ⇒ **`fb.append(msg)` → `fb.append((lever, target, msg))`**로
  바꿔 메타데이터를 실은 뒤에 관문을 건다([[55]] `proc_fb` 死배선과 같은 계열의 위험 회피).

---

## 6. ACTION의 진짜 원인은 게이트가 아니다 — [[22]] 따름정리 (별도 설계)

022 turn 2의 `submit_transaction`은 **매핑 오류**다. `formalize_intent_tool`(`t2_resolve.py:207`)은
`'none'` 기권 옵션을 주는데도 리워드 감사 요청에서 그 도구를 골랐다(입력 = 마지막 user 메시지 6개·각
300자 절단). 후보 집합이 선언이라 어떤 "선언 면허" 조건도 이걸 못 막는다(§5.2).

정합적 처방은 [[22]] 따름정리 **근거-우선 formalize**다: 표적을 고를 때 **손님 발화의 축자 근거를 함께
산출**하게 하고, 검증기는 의미가 아니라 **근거만** 검사한다(핀 고정·불이행 시 기권). 같은 패턴의 선례가
이미 A2에 있다 — `axis_notes.give_quote`(*"넘기기 전 손님 발화 4단어 연속 축자 인용"*).
이건 게이트 조정이 아니라 **formalize 계약 변경**이므로 **별도 설계서**로 다룬다(이 문서 범위 밖).

## 7. 사전등록 — 무엇을 재고 무엇으로 판정하나

### 7.1 오프라인으로 이미 확정된 표적 (`x104 §C`)

| 태스크 | 레버 | 분류 | 표적 | 금지 절차 | 발화 | E3-② 적용 |
|---|---|---|---|---|---|---|
| **022** | **VALUE-ACQUIRE** | **push** | `get_card_last_4_digits` | `cash_back_dispute` | **3** | **침묵** |
| 022 | PROCEDURE | react | `get_card_last_4_digits` | `cash_back_dispute` | 3 | 유지(금지를 집행하는 문장) |
| 048·051·035·053 | VALUE-ACQUIRE | push | 〃 | (금지 절차 비활성) | 12 | 유지 |

⇒ **정확히 022의 3발만 지운다. over-block 0. 이 레버의 통과 sim 발화 = 0.**

⚠ **범위의 정직한 한계(유효 n=1)**: VALUE-ACQUIRE 15발은 전부 **동일 문구·동일 표적 하나**이고
(A2 `value_acquisition` 스펙이 1개), 5개 sim의 차이는 활성 절차가 `cash_back_dispute`인지뿐이다.
즉 E3-②는 **(레버 1 × 표적 1) 패치**이며, §7.2가 검증하는 것도 그 한 칸이다.
**"조정의 단위는 (레버×단계×조건)"이라는 프레임 자체는 이 측정이 지지하지 않는다** — 나머지 25개 레버
칸의 근거는 0이고, §4.3은 오히려 **가설 칸 대부분이 해롭다**는 쪽을 가리킨다.

### 7.2 판정 지표 (다음 스모크)

| 지표 | 기준 | 등급 |
|---|---|---|
| VALUE-ACQUIRE 침묵 | 022 3발 · 타 태스크 0발 (x104 §C와 일치) | [M] |
| **부정통제** | 금지 대상을 **의도적으로 1건 심어** 탐지되는지 왕복 확인. 배선이 죽어도 "0발"은 통과하므로 이것 없이는 신호가 아니다([[24]] 탐지자 왕복·[[55]] `proc_fb` 死배선) | [M] |
| 022 액션 이행 | `action_match` 2/12 → 상승 | [D](nt=1) |
| 통과 sim 노출 | 017·018·019·028에서 새 침묵 0 | [M] |
| 대조군 유지 | 017·018·019·028 pass 유지 | [D] |
| 공발화(x104 §B) | `PROCEDURE+VALUE-ACQUIRE` 0회 | [M] |

### 7.3 함정

- **오프라인 노출 측정은 1차 위험만 준다** — 035가 반례다(침묵 자체의 통과 노출은 0이었는데 궤적이
  갈려 2차로 통과를 잃었다). §7.1의 "over-block 0"은 *그 발화가 지워진다*는 뜻이지 *결과가 나빠지지
  않는다*는 뜻이 **아니다**.
- **침묵은 공짜가 아니다**(등대 §1.2): 켜는 것도 끄는 것도 상쇄가 있다. 칸을 한 번에 켜지 않는다.
- **user-sim 변동은 면책이 아니다**([[21]]): 022 손실을 "손님이 도구를 안 썼다"로 종결하지 않는다.

## 8. [[05]] 체크리스트 (의무 섹션)

1. **scaffold에 도메인-특화를 박나?** ❌ 아니다. E3-②가 읽는 값은 A2 `procedures[].prohibits`·
   `enter_when`과 env 레지스트리다. 엔진에 들어가는 문장은 *"활성 절차가 금지한 표적은 권하지 않는다"*
   하나뿐 — 도구명·도메인명 0.
2. **모델을 타깃 도메인에 학습하나?** ❌ 학습 0.
3. **새 도메인 = A2-swap만으로 되나?** ✅ `prohibits`는 이미 3층 스키마의 필드다. 금지 선언이 없는
   도메인에서는 조건이 항상 통과 = 현행 거동.
4. **A2 순증**: E3-② = **0**(선언은 2026-07-31에 정책 축자와 함께 이미 있다). E4 = 오기 수정(순증 0).
   ⚠ **`axis_notes.lever_phases`(E1)는 순증 1키이고 층 결정이 미해결이다**: `axis_notes`는 L1
   `base/shared.json:97`에만 있고 로더는 `merged.update(part)` **얕은 병합**(`gate_interpreter.py:99`)이라,
   도메인 파일에 `axis_notes`를 새로 쓰면 **L1의 20여 키가 통째로 사라진다**([[24]] 변형 사고).
   ⇒ E1을 켜려면 (a) L1 전용으로 두고 도메인 재정의를 포기하거나 (b) 로더에 `axis_notes` 깊은 병합을
   넣어야 한다. **(b)는 엔진 변경이므로 별도 판단**이고, 그전까지 E1은 보류(§5.3)다.

## 9. 실행 순서 (측정 근거가 있는 것만 먼저)

1. **E5 완료** ✅ — `x104_lever_arbitration_census.py` 커밋. 이후 모든 Δ의 계기.
2. **E4** — 정책 축자 출처로 문구 수정 + 3파일 동기화 + 왕복 검증.
3. **E3-② 단독** — speak-time 계약 명시, 독립 플래그, `fb`에 (레버, 표적) 메타데이터 적재 선행.
   **여기까지가 측정 근거가 있는 전부다.**
4. 스모크 1회(022 표적) → x104로 §7.2 판정.
5. **보류**: E2/E1은 (a) `discover` 정의 고정 + 분포 재측정 (b) ACTION 선언을 `NOT verify` 등가로 고정
   (c) **035의 C17 귀속 판정**([[18]] 격리) 이후. 이 셋 없이 켜면 035형 2차 손실을 더 넓은 면적에서
   반복한다.
6. ACTION 매핑 오류는 **근거-우선 formalize 별도 설계**(§6)로.
7. 그 다음 97×nt2 전수 런(전수 런 중 엔진 수정 금지).

## 10. 미결

- **035의 C17 귀속** — 격리 arm 필요. E1·E2의 선행 조건.
- **012 대책**(C8 + `axis_notes.user_tool_channel` 병기)은 §4.2의 "react 우선" 자리에서 만난다 —
  C8은 react, `usertoolnote`는 그 뒤에 붙는 부가 문구다. **react가 이겼을 때 부가 문구를 함께 싣는다**로
  통합 가능한지 확인 후 별도 패치 여부 결정.
- **`discover` 정의** — 활성 절차 한정인가, 선언 전체인가. 정의가 곧 침묵 면적이다.
