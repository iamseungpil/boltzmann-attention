# C43 — 날조는 **working memory 문제가 아니다. 정박 치환(anchored substitution)이다** (2026-07-09)

> 상위 = `RESEARCH_MASTER.md`. 사용자 물음: *"결국 작은 모델의 long-context / WM 문제 아닌가?"*
> 판정: **아니다.** 전수 궤적(456 sim × 4 arm·infra 0·clean만) 기준. 재현 스크립트 = 본 doc §5.

---

## 0. 한 줄
모델은 **읽은 것을 잊지 않는다. 읽기 전에 옆에 있는 것을 집어 변형한다.**
`4127323219 → 4127323220`. 창(window)에 근접-오답을 넣을수록 더 그런다.

## 1. WM 가설의 세 갈래를 각각 기각

### H-forget — "읽었는데 긴 문맥에서 잃어버린다"
| arm | 미조회 날조 | **조회했는데도 날조(H-forget)** |
|---|---|---|
| 7B +scaffold | 60 | 7 |
| 14B floor | 84 | **3** |
| **32B floor** | 70 | **1** |
| 32B +present | 57 | 2 |

⇒ **32B는 71건 중 1건.** 잊어서 날조하는 일은 사실상 없다.

### H-load — "대화가 길어지면 무너진다"
에이전트 행동과 무관한 부하 지표로 층화 (32B floor · write 860):

| 부하 지표 | Q1 → Q4 날조율 |
|---|---|
| 사용자 턴 수 | **9.5% → 5.4%** (감소) |
| 사용자 발화 문자수 | 11.1 → 7.5 → 3.7 → 10.3 (단조 아님) |
| 문맥 전체 문자수 | 14.3 → 9.3 → 2.8 → 6.5 (감소) |
| 날조 시점 문맥 median | **6,355자** vs 정상 write **9,421자** |

⇒ **날조는 문맥이 *짧을 때* 일어난다.** 14B도 동일(사용자 턴 Q1 10.2% → Q4 6.3%).

### H-distractor — "비슷한 id가 많으면 헷갈린다"
read 호출수로 층화 후 id-유사 토큰 수의 효과 (32B floor):

| read 구간 | id수 적음 | id수 많음 | Δ |
|---|---|---|---|
| 0–3 | 27.0% | 17.6% | **−9.4pp** |
| 4–5 | 5.8% | 6.7% | +0.9pp |
| 6+ | 2.1% | 4.9% | +2.8pp |

⇒ 일관된 방향 없음. **밀도가 아니다.**

## 2. ★실제로 지배하는 변수 = "아직 안 읽었다"

| read 호출수 | 32B floor 날조율 | 14B floor |
|---|---|---|
| 0–3 | **13.6%** (하위층 27.0%) | 19.2% |
| 4–5 | 6.0% | 8.1% |
| 6–7 | **1.0%** | 3.8% |
| 8+ | 6.0% | 3.0% |

⇒ **조기 커밋(premature commit)**: 조회하기 전에 쓴다. 조회한 뒤에는 거의 안 틀린다.
(주의: read 수는 내생변수다. 그러나 외생 지표(사용자 턴·발화량)가 *반대 방향*이므로 부하 설명은 남지 않는다.)

## 3. ★★기전 — 정박 치환 (anchored substitution)

미조회 날조된 `new_item_id`의 정체:

| arm | **문맥 id의 근접 변형(edit ≤2)** | placeholder 문자열 | 무관한 발명 | 총 |
|---|---|---|---|---|
| 14B floor | 19% | **70%** | 11% | 37 |
| **32B floor** | **70%** | 10% | 20% | 30 |
| **32B +present** | 39% | 28% | 33% | **64** |

실물:
```
4127323219 → 4127323220     7166996157 → 7166996158
8054888773 → 8054888774     9385662952 → 9385662953
6117189161 → 6117189162     3799046073 → 3799046074
```
**옛 item id의 끝자리를 +1 해서 "새 변형 id"를 만든다.**

### 3.1 이 기전이 설명하는 것들
1. **합성(cfbsynth v1·v2)서 날조 0.00** (C42): 문맥에 **id가 하나도 없다** → 집을 게 없다. 그래서 7B도 완벽했다.
   ⇒ C42의 "짧으면 잘한다"는 **짧아서가 아니라 정박할 대상이 없어서**였다. **C42 §3의 load 해석은 철회.**
2. **present가 날조를 2배로 늘린다** (30 → 64): 창에 item id를 더 뿌리므로 **재료를 공급**한다(C34·C35의 기전).
3. **scale은 날조를 없애지 않고 양식을 바꾼다**: 14B = placeholder(`_cheapest`) 70% / 32B = 산술 변형 70%.
   둘 다 *"옆에 있는 걸 쓴다"* 의 변주. **C36 정련.**
4. **E12서 `short` − `full` = +0.03**: 문맥을 줄여도 그 자리의 id는 그대로 남는다 → 개선 없음. **정합.**

## 4. 처방이 뒤집힌다

| 처방 | 판정 |
|---|---|
| 외부 메모장 / 짧은 문맥 / WM 보조 | ❌ 잊는 게 아니다. E12서 +0.03 |
| **창에 근접-오답을 넣지 않기** | ✅ present는 정확히 반대를 했다 |
| **provenance 검증기** | ✅ 정박 산물은 문맥에 없는 문자열 → 100% 검출·Δspurious 0(C40) |
| **gather-before-act 강제** | ✅ read 6+ 구간 날조율 1.0% |
| 학습으로 copy-prior 억제 | **미검** — 단 합성이 날조를 유발해야 함(C42 게이트 FAIL) |

### 4.1 E6′ 데이터 v3 요건 (C42의 D1~D4에 추가)
**D7: 문맥에 *같은 형식의 근접-오답 id*를 반드시 배치**한다.
GET 갈래에서 정답 id는 없되, **정답과 같은 형식의 다른 id들**(예: 조회 전 단계의 다른 레코드 id)이 창에 있어야
base가 정박 치환을 저지른다. **없으면 타당성 게이트를 통과할 수 없다**(v2가 그래서 FAIL).

## 5. 재현
`fl14b_floor_retail_t4` · `fl32b_floor_retail_t4` · `asmregen32b_regen_retail_t4` · `base7b_assembled_retail_t3`
(전부 `user_stop`만 · infra 0). 날조 판정 = `new_item_id ∉ {사용자 발화 ∪ 도구 출력}` ∧ 해당 product 미조회.
근접 변형 = 같은 길이 10자리 · Hamming distance ≤ 2.

## 6. ★선행연구 대조 (딥리서치 `wy3wbu6o9` 완료·2026-07-09·3-vote 검증)

### 6.1 우리 가설을 지지하는 확립된 기전 (검증 통과)
- **정박 치환의 기계적 근거 = contextual entrainment** [Niu et al. ACL 2025·`2505.09338`]: LLM은 문맥에 나온 *아무 토큰*(무관·랜덤 포함)에도 logit을 유의하게 올린다 — 관련성 독립. entrainment head(3-10%) 제거 시 완화. **우리 "인접 id의 edit≤2 변형" 날조의 직접 기전.**
- **induction head** [Olsson 2022·`2209.11895`; Crosbie&Shutova NAACL 2025·`2407.07011`]: prefix-match-and-copy 알고리즘·랭크드 ablation서 인과 필요(추상 ICL −32%→랜덤 근접). ★단 "induction head가 ICL 대부분을 설명"·"copy가 scale-persistent"는 **refute(1-2)** — copy *알고리즘 자체*만 settled.
- **copy suppression head** [McDougall·`2310.04625`]: 모델이 순진한 복사를 *상쇄하려고* 전용 head를 둔다 = 복사가 교정 대상인 prior라는 방증. (GPT-2 Small·전이는 외삽.)

### 6.2 WM 가설 = 반증된 foil (검증 통과)
- [**`2506.08184` "Unable to Forget"**]: WM 병목을 **proactive interference**(덮인 옛 값 재인출)로 설명 → **문맥/간섭 누적 시 오류 증가**를 예측. **우리 데이터는 정반대**(턴↑서 날조↓·read 0-3 27% vs 6+ 2.1%·forget 1/71). **방향 예측이 우리 관측과 배치** ⇒ WM은 지지가 아니라 반증 foil.
- [`2510.05381` "Context Length Alone Hurts"]: 순수 길이 저하는 실재하나 **완벽 조회 상태서도·distractor를 whitespace로 마스킹해도** 발생 = *추론* 저하지 tool-arg 날조 아님. **다른 현상.** 단 그 처방(recite-before-solve)은 우리 gather-before-act와 정합.

### 6.3 distractor 증거 (지지·QA/RAG 유비)
- [Chroma "Context Rot"·18모델]: **단일 distractor도 저하·4개면 복리**. [Cuconasu SIGIR 2024 "Power of Noise"]: near-miss 문서가 성능 해침. ⇒ **present 날조 2배(30→64)와 일치**. (단 blog·QA도메인·모델별 차이 주장은 refute 0-3.)

### 6.4 ★off-policy DPO 실패의 기전 = C38 정확히 설명 (지지)
- [**Razin ICLR 2025·`2410.08847` likelihood displacement**]: 선호 응답 확률이 학습 중 *감소*·유사 embedding 쌍(CHES 高)서 구동. [Yan `2406.07327`]: chosen/rejected 동시 확률 하락.
- ⇒ **valid id vs fabricated id는 edit-distance 1-2 = CHES 극히 높음** = DPO가 정답 id로 질량을 *밀어내는* 정확한 regime. **C38의 "DPO off-policy 실패"를 기계적으로 설명.** on-policy(RLVR·DPO-Positive)가 대안 [`2402.13228`].

### 6.5 ★우리 원본 관찰 (선행 없음 = whitespace)
- **scale 형태 변화(7B placeholder ↔ 32B 산술변형·C36)를 확인한 소스 0** ⇒ **원본 발견**(별도 인용·실험 필요).
- **provenance 검증기·constrained decoding·retrieval-forcing이 tool-arg id 날조에 효과적**이라는 강한 선행 **없음**(유일 후보 `2505.05057`은 RAG-불충분·정량우위 둘 다 refute) ⇒ **C45(출처선언 레버)가 whitespace 점유.**
- **tool-call argument hallucination(TCH)**은 명명됨 [survey `2509.18970`] but 기전 없음·"argument extension" primitive는 refute(0-3).

## 7. 미결
- 정박 치환 `+1`이 산술인지 토큰-인접인지 (`...19 → ...20`=산술 시사·기전 프로브 가능).
- entrainment head ablation(white-box)이 production서 tool-arg 날조를 실제로 줄이나 (§6.5 open).
- on-policy(RLVR/DPO-Positive)가 copy-prior를 닫나 — off-policy는 §6.4로 실패 확정.
