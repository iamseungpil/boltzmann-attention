# C38 — cfbsynth in-dist gather 측정 결과 · **"학습 NO-GO"는 데이터의 실패였다** (2026-07-09 새벽)

> 상위 = `RESEARCH_MASTER.md`. 선결 측정 지시 = `E6PRIME_GATHER_LEARN_DESIGN §1`.
> 재현 = `scripts/distill/tau2/c38_indist_gather_eval.py` · held-out = seed 7(학습 seed 0) · 7B + LoRA(vLLM `--lora-modules`).
> 물음: C4 학습 arm은 **(H-nolearn)** 못 배웠나, **(H-transfer)** 배웠는데 안 넘어갔나.
> **답: 둘 다 아니다. (H-no-headroom) 학습 데이터에 그 실패가 들어 있지 않았다.**

---

## 1. 결과 (held-out 합성 · gather 60 / copy 60 / native 60)

| model | cond A (학습 포맷·규칙 有) | cond B (native FC·규칙 有) | cond C (native FC·**규칙 除**) |
|---|---|---|---|
| **base 7B** | gather 0.00 · copy 0.77 | **gather 0.98** | **gather 0.40** |
| **dpo** (cfbsynth DPO) | gather 0.00 · copy **0.63** | gather 0.98 | **gather 0.33** |
| **sft** (cfbsynth native SFT) | (completion 포맷 무응답·artifact) | **gather 1.00** | **gather 1.00** |

- **DPO**: 규칙 없는 조건서 base보다 **낮다**(0.33 < 0.40) · copy도 **0.77 → 0.63**. **아무것도 안 사고 둘 다 팔았다.**
- **SFT**: 규칙 문장 없이도 **1.00** — 표적을 완벽히 배웠다. **그런데 tau2선 악화**(A_notfound .31→.41 · too_many_errors 13→**25**).

## 2. ★세 가지 발견

### 2.1 `cond A gather 0.00` 은 지표 아티팩트 — 결정은 이미 옳다
raw 출력을 읽으면 **base·dpo·sft 전부** 산문으로 답하며 **올바른 lookup 도구를 이름까지 지목**한다:
> `"To proceed with the action, I first need to look up the record using your key_cf1. I will call the **lookup_ujzd** tool..."`

즉 `0.00`은 *hermes `<tool_call>` 태그를 즉시 안 뱉음*을 잰 것이지 **조회 결정의 실패가 아니다.**

### 2.2 ★DPO 페어가 **off-policy** 였다
```
chosen   = <tool_call>{"name":"lookup_...", ...}</tool_call>
rejected = <tool_call>{"name":"act_...", "arguments":{"number":"123456"}}</tool_call>
```
**둘 다 모델이 실제로 내지 않는 출력**이다(모델은 산문을 낸다). 선호최적화의 마진이 모델의 **지지집합 밖**에 놓였다.
⇒ 결정은 안 바뀌고(`cond B` 0.98 = base), **부수 피해만 남았다**(`copy` 0.77 → **0.63**).

### 2.3 ★★합성이 **결손 자체를 제거**하고 있었다
cfbsynth의 사용자 발화는 매번 이렇게 말한다:
> `"My key_cf1 is 0epf9. **I don't have the id.**"`

| | *"나는 그 값이 없다"* 큐 |
|---|---|
| cfbsynth DPO gather 프롬프트 | **150/150 (100%)** |
| cfbsynth native FC | **150/150 (100%)** |
| **tau2 (floor 32B · 120 sim)** | **1건** |

게다가 시스템 프롬프트가 규칙을 명시한다:
> `"When an argument value is not given, obtain it by calling the tool that produces it... Never invent an id"`

그 규칙 한 문장이 **gather의 58pp를 지고 있다**(base: 규칙 有 0.98 → 규칙 除 **0.40**).

> **우리가 사려는 능력은 "내가 그 값을 갖고 있지 않음을 알아채는 것"인데, 합성은 그 신호를 사용자 발화로 공짜로 준다.**
> 그래서 base가 이미 98%를 맞히고, 어떤 손실도 gradient를 줄 수 없었다.
> cfbsynth는 **detection(탐지)** 이 아니라 **obedience(복종)** 를 시험한다.

### 2.4 ★★SFT는 **무조건 조회**를 배웠다 (퇴화 정책)
`cfbsynth_native.jsonl` 2000 궤적:

| 첫 assistant 행동 | 건수 |
|---|---|
| `lookup_*` | **2000 / 2000** |
| 바로 `act_*` (값이 이미 있어 조회 불필요) | **0** |

**음성 사례가 하나도 없다.** SFT는 조건부(`값이 없으면` 조회)가 아니라 **무조건 조회**를 학습했고,
cond C 1.00은 그 증거다. tau2에선 매번 getter를 부르다 잘못된 인자로 not-found를 맞고 포기한다
⇒ `too_many_errors` 13→**25**, `A_notfound` .31→**.41**.
(DPO에는 `copy` 페어가 있었으나 §2.2대로 **off-policy** 라 작동하지 않았다.)

## 3. 판정

- **C4의 "학습 NO-GO"는 학습의 실패가 아니라 *데이터*의 실패다.** learn 축은 **진짜 결손 위에서 시험된 적이 없다.**
- 사용자 주장(*"명확한 원인을 scaffold 꼼수로 피하지 말고 학습으로 보강하라"*)은 **기각된 적이 없다.**
- 동시에 `dpo`의 `copy` 저하(0.77→0.63)는 **off-policy DPO의 실제 부작용**이다 ⇒ *"학습은 부작용이 거의 없다"* 는
  **방법에 의존**한다. on-policy로 바꾸면 사라질 수 있으나 **측정 전에는 가정 금지.**

## 4. E6′ 데이터 수정 (필수 4항)

| # | 수정 | 근거 |
|---|---|---|
| D1 | 사용자 발화에서 **"I don't have the X" 큐 제거** — 결손을 모델이 스스로 탐지하게 | §2.3 (100% → tau2 0.8%) |
| D2 | 시스템 프롬프트에서 **규칙 문장 제거** (또는 별도 arm으로 분리) | base 0.98 → 0.40 |
| D3 | **on-policy rejected**: 모델이 실제로 낸 출력(산문·날조 write)을 rejected로 사용 | §2.2 off-policy |
| D4 | rejected 집합을 **{스키마 예시, 발명형 id, 조합형 placeholder}** 로 확장 | C39 · C36(32B는 발명함) |

추가로 **D5(다중턴·distractor)** 는 T1 가설이나, SFT 데이터가 이미 native FC였으므로 **포맷 격차는 SFT 실패의 원인이 아니다.**

## 5. 남은 셀 · 다음
- `dpo/C` · `sft/B` · `sft/C` 수집 후 본 표 완성. **`dpo/C` ≈ base/C(0.40)이면 §2.2가 확정된다.**
- 이후 순서(사용자 지시 3→2→1): **2 완료**(E11-a의 getter 정합성 = 23건 중 12건만 올바른 원천·주소는 getter가 오답) → **1**(E11-a 다중턴 본실행).
