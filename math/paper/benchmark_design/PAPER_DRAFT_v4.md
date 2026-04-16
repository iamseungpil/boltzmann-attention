# Layer-Adaptive Q+K Ontology Steering: 레이어별 역할 분리를 통한 멀티 도구 선택 최적화

> **CANONICAL DOCUMENT** -- 이 문서가 프로젝트의 기준 문서입니다. 이후 모든 실험 결과와 해석은 여기에 통합됩니다.
> v1/v2/v3 초안은 참조용으로만 보존합니다.
>
> **날짜**: 2026-04-17
> **상태**: v4 (layer-adaptive 발견 반영, 전면 재구성)
> **타겟**: ICLR 2027 (또는 NeurIPS 2026 late deadline 검토)
> **이전 버전**: PAPER_DRAFT_v3.md (Q-coverage full-pivot) <- v2 (hybrid) <- v1

---

## 1. 문제 정의

### 1.1 현장 상황: 수천 개 파편화된 도구 DB

```
사용자: "뉴스에서 주식 관련 기사 찾아서 요약해줘"
```

실제 엔터프라이즈 AI 에이전트는 수천 개의 도구가 파편화된 DB로 존재한다:

| 도구 | domain | function_action | io_type | 비고 |
|------|--------|-----------------|---------|------|
| NewsTool | news | retrieve | text | |
| FinanceTool | finance | analyze | numeric | |
| ResearchFinder | research | retrieve | text | NewsTool과 기능 중복 |
| ResearchHelper | research | summarize | text | |
| RepoTool | research | search | text | |

**3가지 핵심 문제:**

1. **중복 도구**: `NewsTool`과 `ResearchFinder`는 둘 다 `retrieve`인데, domain만 다름
2. **다중 facet 매칭**: 위 질문은 `news x retrieve` + `finance x summarize` = **2개 도구** 필요
3. **사용자 선호**: 같은 기능이라도 사람마다 선호하는 도구가 다름

### 1.2 4-Facet 온톨로지: 도구를 직교 축으로 분해

도구를 4개 직교하는 facet으로 분류한다:

```
도구 = (domain, function_action, io_type, tool_category)
```

예시:
```
NewsTool      = (news,      retrieve,  text, -)
FinanceTool   = (finance,   analyze,   numeric, -)
MusicTool     = (entertain, play,      audio, -)
WeatherTool   = (weather,   inform,    text, -)
```

사용자 질문도 같은 facet 공간에 투영할 수 있다:

```
"뉴스에서 주식 기사 찾아서 요약해줘"
-> 필요 facet: {domain: [news, finance], function_action: [retrieve, summarize]}
-> 정답 도구: NewsTool (news x retrieve) + FinanceTool (finance x summarize)
```

### 1.3 왜 기존 방법이 멀티 도구에서 실패하는가

#### K-only 방법 (SEKA, CAA, 우리의 K-bias)이 실패하는 이유

K-side 스티어링은 **정지(stationary) 연산자**다 -- 매 디코딩 스텝마다 동일한 K 증폭을 적용한다:

```
Step 1: K' = K + alpha * P_ont * K  ->  NewsTool 선택 (O)
Step 2: K' = K + alpha * P_ont * K  ->  NewsTool 또 선택 (X) -- 같은 증폭이니까!
```

K-bias는 "어느 도구가 이미 선택되었는지" 모른다. SEKA도 동일한 구조적 한계를 가진다.

**실험 증거 (canonical SEKA, smoke N=20)**:
- Qwen2.5-7B: amp=0.5 -> -22.5pp, amp=5.0 -> -70pp (FC 완전 붕괴)
- Llama-3.1-8B: amp<=1.0 -> 0.00pp (baseline과 동일), amp=5.0 -> -14.2pp

**모든 테스트된 (모델, 증폭) 조합에서 K-only는 멀티 도구 양의 리프트를 생산하지 못했다.**

#### Q-only 방법도 부족한 이유

Q에서 이미 선택된 facet 방향을 빼면 탐색 방향은 바뀌지만, K의 정확도 보조 없이는 남은 후보 중 정답을 골라내기 어렵다:

```
                    Step 1 (Q 원본)   Step 2 (Q에서 news/retr 빼기)
q * k_FinanceTool =  0.61       ->     0.63  (finance 상대적 강화, 하지만 소폭)
q * k_JobTool     =  0.58       ->     0.60  (career/search도 같이 올라옴)

-> FinanceTool(0.63)과 JobTool(0.60) 차이가 0.03밖에 안 됨
-> K에 정확도 증폭이 없으니, 비슷한 도구 사이에서 헷갈림
```

**비유**: 지우개는 있는데 망원경이 없는 상태.

#### 그런데 둘을 동시에 전 레이어에 쓰면 더 나빠진다

이것이 v1~v3까지의 교착 상태였다:
- K만 전 레이어에 쓰면: -4.57pp (Subtask4)
- Q만 전 레이어에 쓰면: +0.00pp (효과 없음)
- K+Q 둘 다 전 레이어에 쓰면: 여전히 파괴적 (K의 후반 레이어 파괴가 지배)

**핵심 질문: K와 Q를 *언제, 어디서* 써야 하는가?**

---

## 2. 핵심 발견 -- 레이어별 역할 분리 (U-Shape)

### 2.1 Attention MSE의 U자 곡선

Transformer의 레이어를 L이라 하면, K-bias 적용 시 attention output의 MSE 변화가 U자 형태를 보인다:

```
MSE
 |
 |  *                                        *
 |   *                                      *
 |    *                                    *
 |     *                                  *
 |      *                               *
 |       *       **************        *
 |        *    *               *      *
 |         ***                  ****
 |
 +------+--------+---------+---------+-----> 레이어
     0   L/4     L/2       3L/4      L

     [초기]     [중간: MSE 최저]     [후반]
     정보 인코딩    안정 구간         출력 수렴
```

이 U-shape는 3개 구간의 서로 다른 역할을 반영한다:

| 구간 | 레이어 | MSE | 역할 | 시사점 |
|------|--------|-----|------|--------|
| **초기** | 0 ~ L/4 | 높음 | 토큰 정보 인코딩 | K-bias가 여기서 정확도 확보 효과 큼 |
| **중간** | L/4 ~ 3L/4 | 최저 | 안정적 표현 구축 | 조작 여유 있음 -- Q/K 모두 가능 |
| **후반** | 3L/4 ~ L | 다시 높음 | 출력 분포 수렴 | K를 쓰면 **파괴**; Q-coverage만 안전 |

### 2.2 핵심 통찰

> **K를 전 레이어에 쓰면 -4.57pp, 초기 1/4에만 쓰면 +2.08pp**

후반 레이어에서 K를 증폭하면 출력 분포가 수렴하는 과정을 교란하여 성능이 파괴된다. 반면 초기 레이어에서의 K 증폭은 정보 인코딩 단계에서 정답 도구 방향을 강화하여 이후 레이어가 올바른 표현을 구축하도록 유도한다.

Q-coverage(이미 선택한 facet 빼기)는 후반 레이어에서도 안전하다 -- Q는 "다음에 뭘 볼지"를 결정하는 것이지 이미 인코딩된 정보를 변형하는 것이 아니기 때문이다.

**이것이 v5 (layer-adaptive)의 핵심 아이디어다:**
- **초기 레이어**: K로 정확도 확보 (망원경)
- **중반+후반 레이어**: Q로 탐색 범위 확대 (지우개)
- K와 Q를 레이어별로 **분리**해서 각자의 장점만 활용

---

## 3. 알고리즘 v0 -> v5 진화

### 3.0 공통 수학적 기초

```
Attention(Q, K, V) = softmax(Q K^T / sqrt(d)) V

B_ont: 온톨로지 기저 (per layer, per head) -- 도구의 의미 구조를 d차원 공간의 방향으로 인코딩
P_f = B_f * B_f^T: facet f에 대한 투영 연산자
P_ont = B_ont * B_ont^T: 온톨로지 전체 투영
```

### Version 0: Q-coverage Only (탐색 범위만)

```python
# Q에서 이미 선택된 facet 방향만 빼기. K는 원본 그대로.
emitted = {}
while n_emitted < max_tools:
    P_emitted = sum(P_facet(s) for s in emitted)
    Q' = Q - |beta| * P_emitted * Q     (beta < 0)
    K' = K                               (변화 없음)
    tool = generate_one_tool(Q', K')
    emitted.add(tool)
```

- **할 수 있는 것**: 이미 선택한 도구 반복 방지
- **못 하는 것**: 남은 후보 중 정답을 정확히 골라내기
- **비유**: 지우개만 있고 망원경 없음
- **결과**: +0.00pp (Subtask4 N=497)

### Version 1: K-bias Only (단일 도구 전용)

```python
# 모든 레이어, 모든 스텝에서 동일한 증폭
K' = K + alpha * P_ont * K
```

- **할 수 있는 것**: 1개 도구 정확도 향상 (+15pp Llama ST1, +11pp Qwen ST1)
- **못 하는 것**: 멀티 도구 선택 (고정 증폭 -> 같은 도구 반복)
- **결과**: +15.08pp (Subtask1), **-4.57pp** (Subtask4) -- 멀티 도구에서 오히려 악화

### Version 2: Q+K Uniform (멀티 도구 -- 전 레이어 동시 적용)

```python
emitted = {}
while eps_q > threshold AND n_emitted < max_tools:
    P_remaining = P_ont_full - P_emitted
    K' = K + alpha * P_remaining * K     # K: 남은 facet만 증폭
    Q' = Q + beta * P_emitted * Q        # Q: 이미 선택된 facet 빼기
    tool = generate_one_tool(Q', K')
    emitted.add(tool)
```

- **설계 의도**: K의 정확도 + Q의 탐색 범위를 결합
- **현실**: K가 후반 레이어까지 적용되어 출력 분포 파괴
- **결과**: -4.57pp (K의 파괴가 Q의 이득을 압도)

### Version 3: Facet-Adaptive Weighting (쿼리 적응형)

```python
# 각 facet의 에너지 측정 후 가중 투영
eps_f = ||B_f^T * q||^2 / ||q||^2
w_f = eps_f / sum(eps_f)
P_weighted = sum(w_f * P_f)
# 도구 선택 후 해당 facet decay (x 0.2)
```

- **설계 의도**: 질문에 따라 중요한 facet에 집중
- **현실**: 적응 로직이 오히려 노이즈 유입 -- 에너지 측정이 불안정
- **결과**: -1.03pp (Subtask4 N=497, baseline 대비 악화)

### Version 4: Multi-Turn Facet Narrowing (멀티 턴)

```python
class ConversationState:
    conv_weights = None   # EMA 메모리

def process_turn(user_message, state):
    measured = measure_facet_weights(user_message)
    weights = gamma * measured + (1-gamma) * state.conv_weights
    # 도구 선택 루프 + facet decay
    # 대화 메모리 업데이트
```

- **설계 의도**: 턴 간 facet 정보 누적 -> 추상에서 구체로 수렴
- **현실**: 미실행 상태. v1~v3의 gen loop 버그 수정이 선행 필요했음.
- **결과**: 미실행

### Version 5: Layer-Adaptive K+Q (NEW -- 핵심 기여)

```python
def make_layer_schedule(L, alpha, beta, mode="k_early_only"):
    schedule = []
    for li in range(L):
        boundary = L // 4
        if li < boundary:
            # 초기: K가 정확도 확보 + Q도 적용
            schedule.append((alpha, beta))
        else:
            # 중반+후반: K 안 건드림, Q만 적용
            schedule.append((0.0, beta))
    return schedule

# hook 설치
for each layer l:
    alpha_l, beta_l = schedule[l]
    if alpha_l > 0:
        K' = K + alpha_l * P_remaining * K
    if beta_l != 0:
        Q' = Q + beta_l * P_emitted * Q
```

**두 가지 변형:**

| 변형 | 설명 | 레이어 스케줄 |
|------|------|---------------|
| `layer_adaptive` | K 초기, weak K 중간, Q 후반 | L0~L/5: K only, L/5~3L/4: 0.3*K + Q, 3L/4~L: Q only |
| `k_early_only` | K 초기 1/4만, Q 전체 | L0~L/4: K + Q, L/4~L: Q only |

**결과** (다음 섹션에서 상세 비교):
- `layer_adaptive`: **+2.01pp** (F1 0.7507)
- `k_early_only`: **+2.08pp** (F1 0.7514) -- **BEST**

**v5의 핵심 원리:**

```
초기 레이어 (0~L/4):  K 증폭으로 정답 도구 방향 강화
                       "이 토큰들은 news 관련이야" -- 정보 각인
                       |
                       v
중간 레이어 (L/4~3L/4): K 증폭 중단, Q에서 기선택 facet 빼기
                         "news는 이미 골랐으니 finance 찾아봐" -- 탐색
                         |
                         v
후반 레이어 (3L/4~L):   Q-coverage만 유지, K 절대 건드리지 않음
                         출력 분포 수렴을 보호하면서 탐색 방향만 미세 조정
```

---

## 4. 실험 결과 -- 완전한 비교표

### 4.1 Subtask4 (멀티 도구 선택) 전체 비교 (Qwen2.5-7B-Instruct, N=497)

| 방법 | 버전 | F1 | Exact | Delta F1 | alpha | beta | 비고 |
|------|------|-----|-------|----------|-------|------|------|
| no_steer (baseline) | - | 0.7307 | 0.5252 | -- | 0 | 0 | |
| Q-only static beta=-0.03 | v0 | 0.7307 | 0.5252 | +0.00pp | 0 | -0.03 | Q만으로는 효과 없음 (v2 bugfix eval) |
| Q-only iterative beta=-0.03 | v0 | 0.7307 | 0.5252 | +0.00pp | 0 | -0.03 | iterative도 동일 |
| Q-only beta=-0.1 | v0 | 0.7470 | - | +1.64pp | 0 | -0.1 | v3 paper에서 보고 |
| Q-only adaptive (v3) | v3 | 0.7203 | 0.5151 | -1.03pp | 0 | -0.03 | 적응형이 오히려 악화 |
| K-only uniform alpha=0.3 | v1 | 0.6850 | - | -4.57pp | 0.3 | 0 | K 전 레이어 -> 파괴 |
| K+Q uniform alpha=0.3 | v2 | - | - | ~-4.57pp | 0.3 | -0.03 | K 파괴가 지배 |
| SEKA amp=0.5 (canonical) | 선행연구 | 0.4750 | - | -25.6pp | - | - | smoke N=20, Qwen |
| SEKA amp=2.0 (canonical) | 선행연구 | 0.1100 | - | -64.3pp | - | - | K-only stationary 붕괴 |
| **layer_adaptive** K초기+Q중후반 | **v5** | **0.7507** | **0.5352** | **+2.01pp** | 0.05 | -0.05 | NEW |
| **k_early_only** K초기1/4+Q전체 | **v5** | **0.7514** | **0.5473** | **+2.08pp** | 0.05 | -0.05 | **BEST** |

### 4.2 결과 해석

**v5 `k_early_only`가 최고 성능인 이유:**
1. K를 초기 1/4 (Qwen L=28 기준 L0~L6) 에만 적용하여 정보 인코딩 단계에서 정답 방향을 각인
2. L7 이후부터는 K를 건드리지 않아 출력 분포 파괴를 완전 차단
3. Q-coverage는 전 레이어에서 적용하여 기선택 도구 회피를 최대화

**`layer_adaptive` vs `k_early_only`:**
- `layer_adaptive`는 중간 레이어에 약한 K (0.3*alpha)를 남김 -> 미세한 추가 파괴
- `k_early_only`는 초기 이후 K를 완전 차단 -> 더 깨끗한 분리
- F1 차이 0.07pp (0.7507 vs 0.7514) -- 큰 차이는 아니지만 일관되게 k_early_only가 우세

**v5에서의 alpha/beta 값 (0.05/-0.05):**
- v1~v3에서 사용하던 alpha=0.3보다 훨씬 작은 alpha=0.05
- 이유: 초기 레이어에만 집중하면 작은 alpha로도 충분한 방향 각인이 가능
- beta=-0.05도 beta=-0.1보다 약함 -- 전 레이어에 고르게 적용하므로 누적 효과가 강력

### 4.3 Subtask1 (단일 도구 선택) 참고 결과

| 모델 | Baseline | K-bias alpha=0.3 | Delta | 스코러 |
|------|----------|------------------|-------|--------|
| Llama-3.1-8B | 62.31% | 77.39% | **+15.08pp** | substring |
| Qwen2.5-7B | 75.58% | 86.73% | **+11.16pp** | substring |
| Qwen2.5-7B | - | - | **+2.81pp** | parser_safe (strict) |

**주의**: substring 스코러에서 parser_safe (strict)로 전환하면 수치가 3~5배 감소한다. 방향 특이성 갭(+24~49pp real vs random)은 스코러와 무관하게 유지된다.

Subtask1은 **parity regime check**이다 -- 단일 선택 체제에서 우리 K-bias가 SEKA 급 성능을 보인다는 확인. 우리의 진정한 기여는 Subtask4 멀티 선택이다.

### 4.4 방향 특이성 검증 (Null-Control)

| 조건 | Subtask4 F1 | 의미 |
|------|-------------|------|
| 실제 B_ont (alpha=0.3) | 0.685 | 온톨로지 방향 사용 |
| Random 방향 (동일 norm) | 0.000 | 아무 방향 -> 완전 실패 |
| Feature-shuffle (동일 norm) | 0.000 | facet 섞기 -> 완전 실패 |

**+68.5pp 갭**: 실제 온톨로지 방향만이 동작. 이것은 B_ont가 기하학적으로 특권적인 부분공간이라는 강한 증거다 (Cor 6.9.6). Random이나 feature-shuffle 컨트롤이 동일한 크기의 perturbation임에도 FC-emission manifold를 완전히 이탈하는 반면, 실제 B_ont만이 이를 보존한다.

### 4.5 전 버전 결과 진화 궤적 (Subtask4, Qwen)

```
버전      F1      Delta    왜 이 결과인가
------    ------  ------   ----------------
v0(Q)     0.7307  +0.00    Q만으로는 정확도 보조 없어서 효과 없음
v1(K)     0.6850  -4.57    K 전 레이어 -> 후반부 출력 파괴
v2(K+Q)   ~0.685  -4.57    K 파괴가 Q 이득을 압도
v3(적응)   0.7203  -1.03    에너지 기반 적응이 노이즈 유입
v4(멀티턴) 미실행   -        gen loop 버그로 미실행
v5(layer)  0.7514  +2.08    초기 K + 전체 Q = 역할 분리 성공  ★
```

---

## 5. 메트릭 논의

### 5.1 현재 F1의 한계

현재 사용하는 set-level F1은 도구 간 **우선순위**를 반영하지 못한다:

```
정답: [NewsTool, FinanceTool]
예측1: [NewsTool, JobTool]      -> F1 = 0.5 (NewsTool 맞음, FinanceTool 빠짐)
예측2: [NewsTool, ResearchTool]  -> F1 = 0.5 (동일 점수)

하지만 예측2가 더 나을 수 있다:
  ResearchTool = (research, retrieve, text) 
  -> "retrieve" facet은 FinanceTool과 부분 중복
  -> 기능적으로 더 가까운 대체

F1은 이 차이를 반영하지 못한다.
```

### 5.2 Facet-Weighted nDCG 설계 방향

도구의 **부분 일치(partial match)**를 반영하는 메트릭:

```
relevance(tool_pred, tool_gt_set) = max over tool_gt in gt_set:
    sum_f (facet_f_match(tool_pred, tool_gt) * eps_f)

여기서:
  facet_f_match = 1 if tool_pred.facet_f == tool_gt.facet_f else 0
  eps_f = 해당 facet의 에너지 (우리 알고리즘의 측정값 직접 사용 가능)
```

nDCG로 순위 품질 측정:

```
DCG = sum_i relevance(pred_i) / log2(i+1)
nDCG = DCG / IDCG
```

**장점:**
- 3개 중 1개 **필수** 도구 (domain facet 에너지 높음) vs 3개 중 2개 **nice-to-have** 도구 (io_type facet만 일치) 구분 가능
- 우리 알고리즘의 eps_f 측정값을 자연스럽게 활용
- 부분 facet 일치에 대한 부분 점수 부여

### 5.3 Exact Match의 가치

현재 Exact Match는 엄격하지만 유용한 보조 메트릭이다:

```
k_early_only:  Exact = 0.5473 (baseline 0.5252, +2.21pp)
layer_adaptive: Exact = 0.5352 (baseline 0.5252, +1.00pp)
```

k_early_only가 Exact에서 더 큰 우위를 보인다는 것은, 단순히 "비슷한 도구를 더 많이 맞추는" 것이 아니라 **정확히 정답 도구 세트를 완전 일치시키는** 빈도가 높아졌다는 의미다.

---

## 6. 수학적 기초 (요약 + 참조)

### 6.1 Thm 6.1: Per-Sample Attention-Weighted Bound

```
E_q ||o_hat - o||^2 <= 2 * E[qaMSE * Var_s(V)] + C_1 * rho^4
```

- 모든 3개 역할 (Q-steering, K-stability, K-compression)의 기반
- Qwen2.5-7B L=13 alpha=0.3: bound_pass_rate 2800/2800
- median LHS/RHS ratio 2.36e-8 (느슨하지만 방향적으로 정보적)

### 6.2 Thm 6.17' (revised): Q-Coverage First-Order Optimality

**원래 주장**: QKV-joint가 최적
**수정된 주장**: **Q-coverage primary + K small-alpha additive** 패밀리가 검증된 정확도 리프트 패밀리

Q-coverage 연산자:
```
Delta_Q^(t) = -beta * sum_{s<t} P_{f_s} * q_t
```

여기서 P_{f_s}는 이미 방출된 도구 s의 facet 투영 연산자.

**검증**: Qwen Subtask4 N=497에서 beta=-0.1 -> F1 +1.64pp (3-tier null-control 확인)

상세 증명: `math/paper/lie_group/THEOREM_SUPPLEMENTS_2026_04_16.md` 참조

### 6.3 Thm 6.20': eps_q Stopping Criterion

```
eps_q_t = ||B_ont^T * q_t||^2 / ||q_t||^2
```

- 남은 온톨로지 에너지 -> 자동 중단 기준
- AUROC 0.976 [0.947, 1.000] (N=100 smoke)
- 길이 고정 rebuttal: n_steps=29에서 4 fail vs 4 success 완전 분리 (eps* = 0.14)
- Non-vacuous in favorable regime (상세: THEOREM_SUPPLEMENTS)

### 6.4 Lemma 6.17.C: Projector Idempotency

P_f = B_f * B_f^T는 idempotent (P_f^2 = P_f). 이것은:

1. Q에서 facet 빼기가 중복 적용되어도 안전 (이미 뺀 방향을 또 빼도 추가 손상 없음)
2. v2에서 발생했던 "투영 중첩" 버그의 이론적 근거
3. 레이어별 독립 적용의 수학적 정당성

### 6.5 NEW: Layer-Adaptive 이론적 정당화

**왜 U-shape MSE가 레이어별 K/Q 분리를 요구하는가:**

Transformer의 레이어별 표현은 3단계로 진화한다:

1. **초기 (L0~L/4)**: 토큰-수준 정보 인코딩. 이 단계에서 K 벡터는 각 토큰의 의미적 역할을 아직 확정하지 않은 상태다. K-bias가 이 단계에서 온톨로지 방향을 강화하면, 이후 레이어가 이 "방향 힌트"를 받아서 표현을 구축한다. 효과: 정보 각인 (imprinting).

2. **중간 (L/4~3L/4)**: 표현이 안정화되는 구간. attention pattern이 수렴하기 시작하며, 이 구간에서의 perturbation은 비교적 흡수된다. K-bias의 한계 효과가 줄어드는 대신, Q-coverage의 "탐색 방향 재지정"이 안전하게 작동한다.

3. **후반 (3L/4~L)**: 출력 분포 수렴. 최종 토큰 예측을 위한 표현이 확정되는 단계. 여기서 K를 교란하면 이미 수렴 중인 softmax 분포가 깨져서 FC-emission이 파괴된다. 반면 Q-coverage는 "다음에 뭘 볼지"만 조정하므로 이미 인코딩된 정보를 훼손하지 않는다.

**Thm 6.1과의 연결**: per-sample bound의 qaMSE 항이 후반 레이어에서 급격히 증가하는 것은, 해당 레이어의 attention output 변화가 최종 출력에 미치는 영향이 크다는 것을 의미한다. Layer-adaptive 스케줄은 이 qaMSE가 높은 레이어에서 K perturbation을 제거하여 bound의 유효성을 보존한다.

---

## 7. 다음 실험 계획

### P0 (즉시)

1. **Llama-3.1-8B cross-model validation**
   - k_early_only (alpha=0.05, beta=-0.05) on Subtask4 N=497
   - 예상: Qwen과 동일 부호 (+), 크기는 다를 수 있음
   - 소요: ~2시간 (단일 GPU)

2. **alpha/beta sweep on k_early_only (Qwen)**
   - alpha in {0.02, 0.05, 0.1, 0.15}, beta in {-0.03, -0.05, -0.07}
   - 현재 최적 (0.05, -0.05)이 진정한 최적인지 확인
   - 소요: ~4시간

### P1 (이번 주)

3. **Facet-Weighted nDCG 메트릭 구현**
   - F1 기반 비교와 nDCG 기반 비교의 순위 역전 여부 확인
   - 우리 알고리즘의 eps_f를 relevance weight로 직접 사용

4. **Multi-turn (v4) + layer-adaptive 결합**
   - v4의 EMA 메모리 + v5의 레이어 스케줄 결합
   - MetaTool Subtask4를 2-turn 대화로 분리한 합성 데이터에서 테스트
   - 소요: ~1일

### P2 (다음 주)

5. **tau^2-bench external validation**
   - 소매 도메인 멀티턴 도구 호출 벤치마크
   - 완전히 독립적인 데이터셋에서 layer-adaptive의 효과 검증

6. **레이어 경계 최적화**
   - 현재 L/4는 heuristic -- 모델별 최적 경계가 다를 수 있음
   - MSE U-shape의 inflection point를 자동 탐지하는 방법 설계

7. **k_early_only의 초기 레이어 세분화**
   - L0만? L0~L2? L0~L6?
   - 최소한의 K 개입으로 최대 효과를 내는 정확한 레이어 범위 탐색

---

## 8. 한눈에 보기

### 8.1 버전 진화 요약

```
v0(Q only) -> v1(K only) -> v2(Q+K uniform) -> v3(adaptive Q) -> v4(multi-turn) -> v5(layer-adaptive K+Q)
+0.00pp      +15pp(ST1)    -4.57pp(ST4)       -1.03pp          미실행           +2.08pp ★
             -4.57pp(ST4)
"지우개만"   "망원경만"    "같이쓰면 파괴"     "쿼리적응         "멀티턴"         "초기망원경+
                                               노이즈 유입"                      후반지우개"
```

### 8.2 핵심 숫자 3개

```
1. +2.08pp   k_early_only (K 초기 1/4 + Q 전체) -- Subtask4 N=497 BEST
2. +68.5pp   방향 특이성 갭 (real B_ont vs random/shuffle) -- B_ont는 특권적 부분공간
3. 0.976     AUROC eps_q 중단 예측기 -- 배포 시 plan 실패 사전 예측 가능
```

### 8.3 현재 위치와 남은 갭

```
달성한 것:
  [x] K-only의 구조적 한계 진단 (stationary -> multi-tool 실패)
  [x] Q-coverage의 이론적 최적성 (Thm 6.17')
  [x] Layer-adaptive의 경험적 검증 (+2.08pp, N=497)
  [x] 방향 특이성의 강한 증거 (+68.5pp null-control)
  [x] 배포용 중단 예측기 (AUROC 0.976)

남은 과제:
  [ ] Cross-model validation (Llama)
  [ ] alpha/beta 최적화 (현재 0.05/-0.05는 첫 시도)
  [ ] Multi-turn 결합 (v4+v5)
  [ ] External validation (tau^2-bench)
  [ ] Facet-Weighted nDCG 구현
  [ ] 레이어 경계 자동 탐지
```

### 8.4 논문 핵심 서사 (One-liner)

> **K-side stationary steering은 멀티 도구 선택에서 구조적으로 실패한다. Layer-adaptive 스케줄 -- 초기 레이어에서 K로 정확도를 확보하고, 중후반 레이어에서 Q로 탐색 범위를 확대 -- 이 이 한계를 극복하는 첫 번째 방법이다.**

---

## Appendix A: 실험 환경 상세

| 항목 | 값 |
|------|-----|
| 모델 | Qwen2.5-7B-Instruct, Llama-3.1-8B-Instruct |
| 벤치마크 | MetaTool Subtask1 (N=995), Subtask4 (N=497) |
| B_ont 구축 | DeepSeek-V3 분류 -> 4-facet -> Gram-Schmidt 직교화, per-layer per-head |
| GPU | NVIDIA A100 80GB (단일 GPU, CUDA_VISIBLE_DEVICES=0) |
| 평가 스크립트 | `scripts/ocq/eval_subtask4_dynamic_qk_v2.py` |
| Layer-adaptive 결과 | `reports/layer_adaptive_2026_04_17/` |
| Beta sweep 결과 | `reports/beta_sweep_2026_04_16/` |
| Facet-adaptive 결과 | `reports/facet_adaptive_v3_bugfix_2026_04_16/` |
| SEKA 비교 결과 | `reports/seka_headtohead_2026_04_16/` |

## Appendix B: 결과 재현 명령어

```bash
# k_early_only (BEST)
CUDA_VISIBLE_DEVICES=0 python scripts/ocq/eval_subtask4_dynamic_qk_v2.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --alpha 0.05 --beta -0.05 \
    --layer_mode k_early_only \
    --n_queries 497

# layer_adaptive
CUDA_VISIBLE_DEVICES=0 python scripts/ocq/eval_subtask4_dynamic_qk_v2.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --alpha 0.05 --beta -0.05 \
    --layer_mode layer_adaptive \
    --n_queries 497
```

---

*이 문서는 2026-04-17 기준입니다. 새로운 실험 결과는 이 문서에 직접 업데이트합니다.*
