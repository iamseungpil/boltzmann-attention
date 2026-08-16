# 딥리서치 — 진단 방법론의 선행조사 (2026-08-16)

> 대상 = 권위본 §6 에 *선행조사 대기*로 등록해 둔 조각 셋:
> ⒜ 격리↔라이브 **사다리** · ⒝ 뒤집기 **통제 유효성 기계 판정** · ⒞ **잡음 바닥 코드 강제**
> (+ 오늘 관찰한 ⒟ **자기-정박**(첫 지목 후 재료 무시)).
> 결론 먼저: **방법론 단독 논문은 성립하지 않는다.** ⒟는 **정면 선점**이고, ⒞는 **완전 선점**,
> ⒝는 개념적으로 선점(다른 분야), ⒜만 부분적으로 열려 있으나 약하다.

---

## §1 ⒟ 자기-정박 = **정면 선점** (가장 중요한 발견)

**`When Agents Commit Too Soon: Diagnosing Premature Commitment in LLM Agents`**
(Aman Mehta·Snowflake AI Research·arXiv 2606.22936·22p) — 초록·본문 직독.

- 현상 정의가 우리 관찰과 같다: *"agents settle on one reading of the evidence early, then spend
  the rest of the run defending it"* · *"the trajectory can look coherent; the problem is that it has
  become coherent **around the first reading, right or wrong**"*.
- 계기는 다르다: **representational commitment** = 같은 입력을 n회 돌려 **고정 step 의 은닉상태**
  cross-run 코사인 수렴도. 우리는 **궤적 표면**(첫 지목 메시지 index ↔ 재료 도달 턴)으로 잰다.
- ⚠**그들의 개입 결과가 우리 C492 와 같은 모양이다**: *"a prompting intervention cuts behavioral
  variance by **28%** against a token-matched control while leaving **accuracy statistically
  unchanged**"* — 우리 촉구도 격리 2→16/24 를 사고 **라이브 성적은 null**(8/20↔9/20)이었다.
- 그리고 그들 스스로 경계를 긋는다: *"commitment tells us whether an agent has **settled**, not
  whether it is **right**"* · *"a diagnostic for a hidden process failure, **with clear limits rather than
  a general accuracy lever**"*.

**함의**: *"에이전트가 일찍 확정하고 그 뒤 증거를 안 쓴다"* 는 **우리 발견이 아니다**. 인용하고 양보한다.
남는 우리 델타 후보는 둘뿐이고 둘 다 아직 주장 등급이 아니다:
1. **블랙박스 계기** — 은닉상태 없이 궤적 표면만으로 같은 것을 잡는다(재현 비용이 낮다).
2. **개입-관련 종점** — *"재료가 **첫 지목 이전**에 도달했는가"* 를 **레버 설계의 1차 종점**으로 쓰는 것.
   그들은 커밋 **탐지**를 하고, 우리는 커밋 **이전에 재료를 넣는 것**을 종점으로 삼는다.
   ⚠단 이것을 주장하려면 **인과 실험**(도달률을 올리면 적중이 오르는가)이 필요하다 — 아직 없다.

### 곁가지 — 재료를 줘도 안 쓴다
`LiveBrowseComp`(2605.28721): *"Are Search Agents Searching, or Just Verifying What They Already
Know?"* — **증거 사용률이 전 모델 1/3 미만**(DeepSeek v3.2 32.2% · GLM-5.1 24.7% · MiniMax M2.5
30.8% · Kimi-K2.5 31.5%)이고 *"agents search from their own hypotheses"*. ⇒ 우리 024 관찰
(gold 문서가 검색 1위로 왔는데 안 바꿈)도 **이 계열로 이미 문서화**돼 있다.

---

## §2 ⒞ 잡음 바닥·사전등록 = **완전 선점**

- `Measuring all the noises of LLM Evals`(2512.21326): 예측 잡음이 데이터 잡음의 ~2배(MATH500) —
  줄이면 **최소검출효과(MDE)** 가 준다.
- 표본 복잡도 정리(TMLS): 5% 검정·80% 검출력·10% 불일치에서 **HumanEval 6.9pt · GPQA-D 6.3 ·
  SWE-bench Verified 4.0 · GSM8K 2.4 · MMLU 0.75** 보다 작은 차이는 못 가른다.
- `Hidden Measurement Error in LLM Pipelines`(2604.11581) · 경량 **사전등록 프로토콜**(해시 공개)까지
  이미 제안돼 있다.

⇒ 우리 `NOISE=4`·`cite()`·*"판정문을 결과보다 먼저 인쇄"* 는 **좋은 위생이지 신규성이 아니다.**
논문에서는 **인용하고 준수 사실만** 적는다.

---

## §3 ⒝ 통제 유효성 기계 판정 = **개념 선점**(다른 분야) · 우리 것은 그 적용

- 역학: `Negative controls: a tool for detecting confounding and bias`(Lipsitch 2010·PMID 20335814) ·
  `Negative controls: concepts and caveats`(PMC10515451) — *"negative controls may lack both
  specificity and sensitivity"* 라는 한계까지 명시.
- 계량경제: `Negative Control Falsification Tests for Instrumental Variable Designs`(2312.15624) —
  **위약(placebo)·falsification 검정**의 정식 계보. *"valid designs may fail negative control tests
  merely due to functional-form violations"*.
- 우리 규칙(*뒤집기 지표가 **어느 팔에서도** 안 열리면 통제 성립 불가 ⇒ 판정 보류*)의 정확한 이름은
  인과추론의 **positivity/overlap(지지) 조건 점검**이다 — 처치 하에서 그 결과가 **일어날 수 있어야**
  비교가 성립한다.

⇒ 신규성 주장 금지. **다만 실무 가치는 실측으로 증명됐다** — 같은 날 무효 통제 **2건**(x335 `D_NEG` ·
x336-070 `D_FLIP`)을 이 검사가 잡았고 유효한 1건(x336-071)은 통과시켰다. 방법 절에 **1문단 + 인용**.

---

## §4 ⒜ 사다리 = **부분 선점** (유일하게 조금 열려 있음)

이미 존재하는 것:

| 선행 | 하는 일 | 우리와의 관계 |
|---|---|---|
| `DoVer`(2512.06749) | 귀속을 **가설로 보고**, 지목 지점에 **표적 편집 후 재실행**해 마일스톤/효용 증가로 판정 | *"귀속은 재실행으로 검증한다"* = 우리 [[62]] 규칙4(성적은 라이브로만)와 **같은 정신** |
| `FALAT`(2606.00765) | 실패 귀속을 **의존성 유도 탐색**으로(평면 step 분류 → 구조적 진단) | 축 분류의 대안 |
| `The Long-Horizon Task Mirage`(2604.11978) | *"어디서·왜 깨지는가"* 진단 | 우리 per-step 분해와 같은 층 |
| `Seeing the Whole Elephant`(2604.22708)·`Who&When Pro`(2607.09996) | 실패 귀속 **벤치마크**(책임 에이전트·결정적 step) | 귀속 평가 기준 |

우리 사다리의 남는 차이: **라이브에만 있는 요인을 이름 붙여 한 칸씩 되돌린다**
(부하 → 타이밍 → 경합 → 끝맺음 → **비영속**). 특히 **비영속**(재료가 그 턴의 재생성 버퍼에만 붙고
`state.messages` 에 안 남는다)은 **스캐폴드 구현 사실**에서 나온 축이라 위 문헌에 대응물이 안 보인다.

⚠그러나 약하다: ⑴DoVer 가 이미 *개입-재실행* 을 하고 ⑵우리 사다리는 오늘 **두 칸을 지우고 한 칸은
무효**였다(`I5` 는 내가 잘못 짜서 `I2` 의 복제였다) ⑶*"비영속"* 은 **우리 구현의 성질**이지 일반 현상이
아닐 수 있다. ⇒ **단독 논문 불가**. `declfirst` 방법 절의 **한 표**로 넣는다.

---

## §5 판정 (권위본 §6 갱신용)

| 조각 | 등급 | 처분 |
|---|---|---|
| ⒟ 자기-정박(첫 지목 후 증거 무시) | **선점**(2606.22936·2605.28721) | **인용·양보**. 우리 표현은 *"블랙박스 계기로 재현"* 까지만 |
| ⒞ 잡음 바닥·사전등록 | **완전 선점** | 인용·준수 표기 |
| ⒝ 통제 유효성 검사 | **개념 선점**(negative control falsification·positivity) | 방법 1문단 + 인용. **실측 2건 잡은 사실**만 우리 기여로 |
| ⒜ 격리↔라이브 사다리 | **부분 선점**(DoVer·FALAT·Mirage) | `declfirst` 방법 절의 표 하나. 단독 논문 금지 |
| **배달-타이밍 종점**(첫 지목 이전 도달률) | **미확인·후보** | **인과 실험 후에만** 주장. 그 전엔 내부 계기로만 사용 |

⇒ **권위본 §6 의 *"미배치 조각"* 은 사실상 소멸한다.** 남는 것은 *배달-타이밍 종점* 하나이고,
그것도 실험이 서야 한다. **개별 6편 불가 + 방법론 단독 불가** ⇒ 3편·2특허 흡수 구도는 그대로다.

---

## §6 출처

- [When Agents Commit Too Soon (2606.22936)](https://arxiv.org/pdf/2606.22936) — 본문 직독
- [LiveBrowseComp (2605.28721)](https://arxiv.org/pdf/2605.28721)
- [DoVer (2512.06749)](https://arxiv.org/pdf/2512.06749) · [FALAT (2606.00765)](https://arxiv.org/html/2606.00765v1) ·
  [Long-Horizon Task Mirage (2604.11978)](https://arxiv.org/pdf/2604.11978) ·
  [Seeing the Whole Elephant (2604.22708)](https://arxiv.org/html/2604.22708v1) ·
  [Who&When Pro (2607.09996)](https://arxiv.org/html/2607.09996v1)
- [Measuring all the noises of LLM Evals (2512.21326)](https://arxiv.org/html/2512.21326) ·
  [Hidden Measurement Error in LLM Pipelines (2604.11581)](https://arxiv.org/pdf/2604.11581) ·
  [Sample Complexity of LLM Evaluation (TMLS)](https://www.tmls.nyc/research/eval-sample-complexity)
- [Negative controls (Lipsitch 2010)](https://pubmed.ncbi.nlm.nih.gov/20335814/) ·
  [Negative controls: concepts and caveats](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10515451/) ·
  [Negative Control Falsification Tests for IV (2312.15624)](https://arxiv.org/pdf/2312.15624)

⚠**검색 범위 한계**: 웹 검색 4회 + 본문 직독 1편이다. `DoVer`·`FALAT` 는 **초록 수준만** 봤다 —
사다리 축의 최종 판정 전에 그 둘은 정독이 필요하다([[41]] 규율).

---

## §7 정독 후 갱신 (2026-08-16 · DoVer·FALAT 본문 확인)

⚠§4 의 *"초록만 봤다"* 경고 해소. 둘 다 본문 축자 확인.

**DoVer**(2512.06749·Microsoft+CAS·29p): *"log-only debugging lacks validation, producing **untested
hypotheses**"* · *"single-step attribution is often **ill-posed** — **multiple distinct interventions can
independently repair** the failed task"*. 방법 = 가설 + **표적 개입**(메시지 편집·계획 변경)으로 능동
검증, 평가는 귀속 정확도가 아니라 **결과**(실패→성공 전환). 성적: 실패 시행 **18~28% 전환** ·
가설 **30~60% 검증/반증** · 다른 프레임워크(AG2·GSMPlus)에서 **49% 회복**.

**FALAT**(2606.00765·Concordia SPEAR): *"attribution cannot be treated as independent step-level
classification"* — 기대 궤적 구성 → 의심 구간 → **의존성 추적**으로 *오류 도입 step* ↔ *물려받은 step*
분리 → *그 step 을 고치면 회복되는가* 로 판정. Who&When **46.0%/29.1%** step 정확도.

### 갱신된 판정

| | 선행이 점유한 것 | 우리에게 남는 것 |
|---|---|---|
| 범주 | **개입-검증형 귀속**(DoVer) · **의존성 귀속**(FALAT) | — (양보) |
| 개입 단위 | 궤적 **내용**(메시지·계획) | **스캐폴드 전달 체제**(무엇을·언제·얼마나 오래) |
| 축 | 의존성 그래프·step | **커밋-상대 타이밍**·**비영속**·경합·방출 |
| 진단 품질 게이트 | 없음 | positivity 검사 · 잡음 바닥 코드 강제 |

⇒ **논문 판정 불변**: 단독 불가, `declfirst` 방법 절로 흡수.

### 특허 방향 (사용자 질문 2026-08-16)

**진단 방법 자체가 아니라 그것이 정당화하는 런타임 기구**로 청구를 세운다.

- 약한 형태(권고 안 함): *"격리↔라이브 사다리로 원인을 귀속하는 방법"* — 평가 방법론이라 기술적
  효과를 대기 어렵고, DoVer/FALAT 가 범주를 점유했다.
- 강한 형태(권고): **"에이전트가 *첫 확정을 내리기 전* 에 검색 재료를 생성 버퍼에 주입하는 제어기"**
  — 입력(궤적 상태·첫 지목 여부) · 판정(커밋 이전인가) · 동작(전달 시점·문서군 선택) · 효과(측정된
  종점 개선). **선행 셋 중 아무도 이것을 하지 않는다**: DoVer=사후 디버깅 · FALAT=오프라인 귀속 ·
  2606.22936=**탐지**(은닉상태)이고 그 개입은 프롬프트 한 줄로 **정확도 불변**.

⚠**전제 셋**(지키지 않으면 청구가 빈다):
1. **인과 실험 선결** — *도달률(첫 지목 이전 배달 비율)을 올리면 적중이 오르는가*. 아직 없다. 실시예의 몸통이 이것이다.
2. **별도 제3건 출원은 폐기된 결정**(2026-08-02 우산 = A·B 흡수). 형태는 **B 종속항 추가 또는 분할출원**.
3. **공개 전 출원**(A/B 명세의 신규성 조항). 그리고 **법률 판단은 변리사 몫** — 이 문서는 기술 정리다.
