# 날조 결정점 프로브 5종 — base 32B 실측 (2026-07-17·무료·n=24/arm)

> 정본 데이터: `sim_results/bank_fab_probes_20260717.json` · `..._persev_20260717.json` · `..._reloc_20260717.json`
> 스크립트 `bank_fab_probes.py` · 로그 `/home/woori/scratch/{fabprobes,persev,reloc}.log`(리모트)
> 설계 = `VALUE_GROUNDING_PLACEHOLDER_LEARN_DESIGN_2026_07_17` §5 **타당성 게이트 1·2단계**.
> 조건: Qwen2.5-32B-Instruct-GPTQ-Int8 · T=0.7 · n=24 · **게이트 전부 OFF**(=모델 자체를 잰다) ·
> 접두 = `bank_kon_20260717_key`(gpt-5.2 user-sim·nt=5) 라이브 궤적의 실패 결정점에서 절단 ·
> 도구 19종(env + A2 주입) = 라이브와 동일 스키마([[30]] §13: "모델이 실제 본 것"으로 검증).

## 0. 한 줄
**5개 결정점 전부에서 base 32B가 실패한다**(날조 46~62%·정답 0/24) ⇒ §5 게이트 2단계 **통과**(learn 표적 실재).
**A2 설명-레버는 실패를 닫지 못하고 *옮긴다***(record 46%→8%이나 이동 29%·**grounded 정답은 끝내 0/24**).
★★그리고 **게이트도 옮긴다 — 그것도 확정적으로**: PROV regen 피드백 하나가 같은 지점의 도구명 날조를
**0/24 → 24/24(100%)** 로 만든다(§3.1·[M]). ⇒ §7 논문 코어("표면마다 규칙 1개 = 발산")의 **motivation이
가설이 아니라 인과 실측**으로 확보됨. 동시에 **C103 "차단≠회복"은 교정 대상**(§3.2).

## 1. 결과표 (n=24/arm)
| probe | 결정점 | FAB(동일표면) | FAB(이동) | **정답** | 지배 잔여 |
|---|---|---|---|---|---|
| `record` | kon sim1 [6] | **11/24 (46%)** | 0 | **0/24** | ASK 등 13 |
| `byphone` | kon sim2 [6] | **15/24 (62%)** | 0 | 9/24 (`by_name`) | — |
| `case` | kon sim1 [32] | **13/24 (54%)** | 1 (`create_case` 발명) | 0/24 | 번호없이 답 10 |
| ~~`dispatch`~~ | kon sim0 [18] | — | — | ⚠️**무효(계측결함)** | v2 재측정 중 |
| (`accum`) | fup sim1 [18] | — | — | 60.0→**93.3%** (§17.1·기확보) | — |

> ⚠️**계측결함 1건 발각·격리(2026-07-17·[[08]] 원문 정독으로 자체 검출)**: `--max_tokens 700`이 **producer 호출을
> 잘랐다** — `get_reward_discrepancies`는 인자에 **거래 23건**을 실어야 해서 700토큰서 절단 → vLLM 파서가
> tool_call로 못 읽고 **content 텍스트**로 반환 → 분류기가 `ASK/텍스트`로 오분류. 저장 원문 재검사:
> **`dispatch` 18/24·`dispatch_hint` 16/24가 실은 잘린 `<tool_call>{"name":"get_reward_discrepancies"…`**.
> ⇒ "producer 직접호출 **0/24**"는 **모델 실패가 아니라 프로브 결함**이며, 라이브 sim2 [20]·sim3 [18]이 같은 상태서
> 직접호출로 `-> 4`를 얻은 사실과 모순됐던 이유다(§5의 "미해결"이 이것이었다).
> **영향 범위 확정**(저장 텍스트의 `<tool_call>` 카운트): `byphone` **0/24**·`case` **0/24**·`discreq_ctl/arm/arm_hint`
> **각 0/24** = **무영향**(결론 유지) / `record` 3/24 = 경미 / **`dispatch` 계열만 무효**.
> **재발 방지**: `--max_tokens`(기본 3000) + **`TRUNC/PARSE`를 독립 범주로 계측·표시**(미파싱 `<tool_call>`·
> `finish_reason=="length"`) — 아티팩트가 다시는 조용히 "모델 실패"로 위장할 수 없게. v2 전 arm 재측정 진행 중.

- **날조 값의 성격 = 전부 스키마-예시형 placeholder**: `123 Elm St, Springfield, IL` · `123 Main St, Anytown, USA` ·
  `john.doe@example.com` · `CASE-123456`(6/24 **동일 문자열**)·`CASE-123456789`(5/24). ⇒ C43/D7 "정박 재료" 가설과 정합
  (모델이 값을 **지어내는** 게 아니라 **예시 분포에서 꺼낸다**) — 학습 데이터 설계(§2 음성 템플릿)에 그대로 재사용 가능.
- **`record` 정답 0/24가 핵심**: 조회를 먼저 하는 궤적이 **한 번도** 안 나온다. A2 설명이 이미
  *"the customer's ACCOUNT RECORD you retrieved (via get_user_information_by_name/email/id)"* 라고 **명시**하는데도.

## 2. ★A2 설명-레버 = **닫지 못하고 옮긴다** (learn 표적 확정)
단일변수 = 도구 설명 문구(긍정형 **구성-지시**만·금지문 arm 없음 = 기증명 무효 C30/C47).

| arm | 동일표면 | **이동** | 정답 | 판정 |
|---|---|---|---|---|
| `record` → `record_hint` | 11/24 → **2/24** | 0 → **7/24**(`get_user_information_by_phone(_number)` **발명**) | **0/24 → 0/24** | ❌**이동**(닫힘 아님) |
| ~~`dispatch` → `dispatch_hint`~~ | — | — | ⚠️**무효(계측결함·§1 경고)** | v2 대기 |
| `discreq_arm` → `discreq_arm_hint` | 0 → 0 | 0 → 0 | **0/24 → 0/24**(호출 0·**언급 24/24**) | ❌**무효**(§4) |

- ★**기전**: "record는 조회 결과에서 복사하라"는 지시 → 모델은 **조회를 하려 하고**, 가진 키가 전화번호뿐인데
  by-phone 조회 도구가 **없으므로** → **없는 조회 도구를 발명**한다. **레버가 날조를 record 표면 → 도구명 표면으로 옮겼다.**
- ⇒ **[[13]] 레버 우선순위 적용 결과: 싼 레버(A2 한 줄)로 안 닫힌다 = 진짜 learn 표적**([[42]] prompt-ceiling의
  값-수준 판·C99 mode-collapse와 동종). 이것이 §7 논문 코어의 **직접 근거**.
- ⚠️정직: `record_hint`의 "동일표면 46%→8%"만 보면 레버가 먹힌 것처럼 보인다. **합산 FAB로 보면 안 되는 이유**
  ([[08]]): 이동분(29%)을 세지 않으면 "레버 성공"으로 오독된다. 스크립트가 두 수를 **분리 계측**하는 이유.

## 3. ★차단≠회복 — **격리에서는 재현되지 않는다** (C103 재검토 필요)
| arm | 조건 | 재-emit |
|---|---|---|
| `byphone` | 차단 없음 | **15/24 날조** |
| `persev` | + 차단 피드백 **1턴**(축자) | **0/24** (24/24 **ASK**) |
| `persev_d1` | kon sim1 [12]·차단 **직후** | **0/24** (24/24 ASK) |
| `persev_d4` | kon sim1 [16]·차단 **4메시지 뒤** = **라이브가 재-emit한 바로 그 지점** | **0/24** (24/24 `log_verification`) |

- 즉 **단일 차단 문구는 재선택을 산다**(전부 ASK로 회복) — `2606.07555`의 "보수적 베이지안 갱신 → 거리에 따라
  prior 복원" 예측도 **거리 4에서 재현 안 됨**(0/24).
- ★**그런데 라이브 sim1은 [16]에서 by_phone을 재-emit했다.** 같은 접두인데 다르다 ⇒ **접두가 같지 않다**:
  라이브 로그가 그 차이를 설명한다 —
  ```
  [T2_PROV] regen fired tool=log_verification arg=name val=John Doe    ×2
  [T2_TOOLGATE] invalid selection 'get_user_information_by_phone_number' -> ASK prompt   ×3
  [T2_PROV] regen fired tool=get_user_information_by_phone_number arg=phone_number val=123-456-7890
  ```
  프로브가 예측한 `log_verification`(24/24)을 **라이브 모델도 냈고**, 인자 `name=John Doe`가 **날조**라 PROV가
  regen을 걸었다. 궤적에 영속된 [16]=by_phone은 **regen 後** 산출물일 개연이 크다.
- ★★**두 게이트가 서로 모순된 지시를 준다**(축자):
  - `[PROVENANCE]` … *"**Do NOT** use placeholder/example values and **do NOT ask the user**. Instead call a lookup/getter tool that produces this value"*
  - `[TOOLGATE]` … *"Do not invent tools … If you are missing information needed to use one of your available tools, **ASK the customer**"*
  전화번호용 getter가 **없는** 상태에서 "묻지 말고 getter를 불러라"의 출구는 **getter 발명**뿐이다.

### 3.1 ★★★`prov_reloc` = 게이트-유발 이동의 **인과 확정** ([M]·n=24·완주)
엔진 정본을 그대로 오프라인 재생(탐지기 `_first_fab_call` · 피드백 `REGEN_FEEDBACK` — 문구 창작 0):
접두 = kon sim1 [0..15](= `persev_d4`와 **동일**) → 1차 emit → PROV가 날조 인자 검출 → regen 피드백 → 2차 emit.

| 단계 | 결과 | n |
|---|---|---|
| **1차(게이트 前)** | `log_verification(name="John Doe")` — 엔진 탐지기가 **24/24 전부** 날조로 판정 | **24/24** |
| **2차(PROV regen 後)** | **`get_user_information_by_phone_number`(비존재 도구) 날조** | **24/24 = 100%** |

- **라이브 재현 완결**: 라이브 로그 `[T2_PROV] regen fired tool=log_verification arg=name val=John Doe`(×2) →
  궤적 [16] `by_phone` = **정확히 이 사슬**. 영속 궤적의 [16]은 **regen 後** 산출물임이 확정.
- ★**인과**: 같은 접두에서 **게이트 前 도구명 날조 0/24**(`persev_d4`) → **PROV 피드백 後 24/24**.
  ⇒ **PROV 게이트가 도구명 날조를 확정적으로 *제조*한다.** 날조 *인자*를 막으려는 게이트가 날조 *도구명*을 낳는다.
- **사슬 전체(전부 실측)**: 조회 0회 → `log_verification(name=John Doe)` 날조(24/24) → **PROV**: "묻지 말고 getter를
  불러라" → by-phone getter **부재** → **도구명 발명(24/24)** → **TOOLGATE**: "발명 말고 **물어라**" → ASK(24/24·`persev`).
  **두 게이트가 정반대를 지시하고, 그 사이에서 모델이 왕복한다.**

### 3.2 ⇒ C103/§20 교정 (실측 근거)
- ~~"차단≠회복: 게이트가 실행을 0으로 막아도 재선택은 회복 안 됨(4~5회 반복)"~~ →
  **단일 차단 문구는 회복(ASK)을 산다(0/24 재-emit·거리 1·4 동일)**. 라이브의 반복은 *차단의 실패*가 아니라
  **게이트 조합(PROV↔TOOLGATE)의 모순이 만든 왕복**이다.
- **§20 "perseveration은 대안이 줄인다"(날조 3 vs 10-14)의 분모 재해석 필요**: 그 날조 카운트의 상당분이
  **게이트-유발**이면, 측정 대상은 모델 성향이 아니라 **우리 스택의 부작용**이다.
- ⇒ **[[08]]·§13("32B가 못 한다는 것의 대부분은 우리 scaffold가 만든 것")의 4번째 반복.** 등급 **[M]**(n=24·결정론적).

## 4. ★DISCREQ 재활성 판정 (handoff §0-4 = 사용자 지시로 실행) ⇒ **기각(이 지점·offline)**
| arm | producer 직접호출 | 지배 반응 |
|---|---|---|
| `discreq_ctl` (사임 후 **실제 사용자 발화**) | **0/24** | ASK 19 · 다른도구 3 · unlock 1 · KB 1 |
| `discreq_arm` (사임 후 **DISCREQ 피드백**) | **0/24** | **ASK 23** · unlock 1 |
| `discreq_arm_hint` (+ "직접 호출하라" 구성-지시) | **0/24** | **ASK 24** |

- ★**핵심(원문 정독)**: `discreq_arm`의 24/24가 **텍스트로 도구를 *언급***한다 —
  *"I apologize for the oversight. Let me use the `get_reward_discrepancies` tool to accurately identify…"* —
  그런데 **`<tool_call>`은 0/24**(=계측결함 아님·진짜 미호출). 대조군은 언급조차 **0/24**.
  ⇒ **DISCREQ가 산 것 = 언급(0→24/24)이지 호출(0→0)이 아니다.**
  이는 설계서 **§9.1b/§14.1c의 라이브 관측**(*"arm이 산 것 = 언급이지 호출이 아니다"*·producer 실호출 0/5·n=3)을
  **n=24로 정확히 재현**한 것 = [M] 승격.
- **DISCREQ의 순효과 ≈ 0**: producer 호출 0→0. 바꾼 것은 **언급**(0→24)과 **되묻기 비율**(19→23)뿐.
- ⇒ **재활성해도 §19.1 창에서 아무것도 못 산다** ⇒ **유료 라이브 재측정 근거 없음**([[09]] — 런 1개 절약).
- **§19.1의 "은퇴가 성급했음"(n=1) 판정 뒤집힘**: 창은 실존하나(사임 ∧ 데이터 읽음 ∧ producer 미호출)
  **레버가 그 창에서 무력**하다. §14.2 은퇴는 **결과적으로 옳았다**(이유는 달랐지만).
- ⚠️**자기교정**: 스모크 n=2에서 `discreq_arm` 2/2가 unlock 경로였고 나는 그때 "DISCREQ가 unlock으로 라우팅된다"고
  적었다. **n=24가 뒤집었다**(unlock 1/24·ASK 23/24). n=2 해석은 과잉이었다 — [[08]] 그대로.

## 5. 부수 관측 (per-case 정독 대기)
- `dispatch`(sim0 [18]·거래 23건 확보 직후): producer 직접호출 **0/24**·ASK 18/24. 그런데 **라이브 sim2 [20]·sim3 [18]은
  같은 상태에서 producer를 직접 호출**했다(그리고 `-> 4` 정답). ⇒ sim0 접두의 무언가가 producer 선택을 막는다.
  **미해결**([[08]] per-case 정독 대상) — 힌트를 넣으면 오히려 ASK 18→24로 **늘어난다**는 점도 함께 설명돼야 함.
- `case`: 10/24는 **번호 없이** 답한다(=날조 안 함). 즉 이 표면은 확률적. 1건은 `create_case` **도구 발명**.

## 6. 다음 (§5 게이트 3단계)
1. `prov_reloc` 결과 → §3 가설 [P]→[M]/기각. **[M]이면 C103·§20·관리표 행1을 실측으로 교정**.
2. **합성 학습 문맥이 이 실패를 재현하는지**(§5-3·C42/C38): 위 날조 값 분포(스키마-예시 placeholder)가
   그대로 D7 "정박 재료" 설계 근거 — 재현되면 LoRA 착수(§5-4·무료).
3. `record_hint`의 **이동 표면**(없는 조회 도구 발명)은 인용-학습 설계의 시험대: `src` 형식은 "조회를 해라"가 아니라
   **"출처를 밝혀라"**이므로, 조회가 불가능하면 `ASK`/`PLACEHOLDER`가 **합법 출구**로 존재한다(=이동할 표면이 없음).
   이것이 §7.2가 게이트 증식을 끊는다는 주장의 정확한 기전 — **프로브로 검정 가능**(학습 후 재측정).
