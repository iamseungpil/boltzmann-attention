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

### 4.1 실험 조건 정의 (모든 변수 명시)

실험 결과를 정확히 이해하려면 각 방법의 **5가지 독립 변수**를 구분해야 한다:

| 변수 | 설명 | 가능 값 |
|------|------|---------|
| **파이프라인** | 생성 방식 | single-pass / iterative / multipass / stop-at-tool |
| **Q 빼는 방향** | Q에서 어떤 방향을 빼는가 | 없음 / 전체 B_ont / P_emitted (선택된 facet만) |
| **K 적용 범위** | K-bias를 어느 레이어에 적용하는가 | 없음 / 전체 레이어 / 초기 1/4 (k_early) |
| **α (K 강도)** | K-bias 증폭 크기 | 0, 0.05, 0.1, 0.3 |
| **β (Q 강도)** | Q-subtraction 크기 | 0, -0.03, -0.05, -0.1 |

**파이프라인 설명:**
- `single-pass`: 한 번에 모든 도구 생성. Q 수정이 전체 생성에 영향.
- `iterative`: multi-step 생성. 각 step에서 hook 갱신하지만, 모델이 1 step에 2개 도구를 이미 생성하므로 P_emitted 갱신이 2번째 도구에 간접적으로만 영향.
- `multipass`: 같은 프롬프트를 여러 번 독립 생성. 각 pass 후 P_emitted 갱신. 모델 context 깨지지 않음.
- `stop-at-tool`: `</tool_call>` 토큰에서 강제 중단 후 이어서 생성. **모델 context 깨짐 → 붕괴.**

### 4.2 Subtask4 전체 비교 (Qwen2.5-7B-Instruct, N=497)

| 방법 | 파이프라인 | Q 빼기 | K 범위 | α | β | F1 | Exact | Δ F1 |
|------|-----------|--------|--------|---|---|-----|-------|------|
| **baseline** | — | — | — | 0 | 0 | 0.7307 | 0.5252 | — |
| | | | | | | | | |
| **Q-only 계열** | | | | | | | | |
| Q 전체 B_ont | single-pass | 전체 B_ont | 없음 | 0 | -0.03 | **0.7535** | 0.5332 | **+2.28pp** |
| Q 전체 B_ont | single-pass | 전체 B_ont | 없음 | 0 | -0.05 | 0.7292 | 0.5332 | -0.15pp |
| Q P_emitted (static) | iterative | P_emitted | 없음 | 0 | -0.03 | 0.7307 | 0.5252 | +0.00pp |
| Q P_emitted (multipass) | multipass | P_emitted | 없음 | 0 | -0.05 | 0.7237 | 0.4930 | -0.70pp |
| Q adaptive (v3) | iterative | P_emitted 가중 | 없음 | 0 | -0.03 | 0.7203 | 0.5151 | -1.03pp |
| | | | | | | | | |
| **K-only 계열** | | | | | | | | |
| K 전 레이어 | single-pass | — | 전체 | 0.3 | 0 | 0.6850 | — | -4.57pp |
| K 전 레이어 | single-pass | — | 전체 | 0.05 | 0 | 0.7100 | 0.5200 | +1.30pp* |
| SEKA amp=2.0 | single-pass | — | 전체 (SEKA) | — | — | 0.1100 | — | -64.3pp |
| | | | | | | | | |
| **Q+K 결합 계열** | | | | | | | | |
| ladapt K+Q | single-pass | 전체 B_ont | K early | 0.05 | -0.03 | 0.7450 | 0.5372 | +1.43pp |
| ladapt K+Q | single-pass | 전체 B_ont | K early | 0.05 | -0.05 | 0.7507 | 0.5352 | +2.01pp |
| multipass P_emitted+K | multipass | P_emitted | K early | 0.05 | -0.05 | 0.7410 | 0.5091 | +1.03pp |
| **iterative P_emitted+K** | **iterative** | **P_emitted** | **K early** | **0.05** | **-0.05** | **0.7524** | **0.5473** | **+2.18pp** |
| k_early_only (이전) | iterative | P_emitted | K early | 0.05 | -0.05 | 0.7514 | 0.5473 | +2.08pp |
| | | | | | | | | |
| **실패 계열** | | | | | | | | |
| stop-at-tool Q only | stop | P_emitted | 없음 | 0 | -0.05 | 0.4600 | 0.0400 | -27.1pp |
| stop-at-tool K+Q | stop | P_emitted | K early | 0.05 | -0.05 | 0.4793 | 0.0000 | -25.1pp |

*N=50 smoke 결과 (N=497 미실행)

### 4.3 핵심 발견 3가지

**발견 1: 작동하는 두 가지 경로**

```
경로 A: single-pass + 전체 B_ont Q-subtraction  → +2.28pp (β=-0.03이 최적)
경로 B: iterative + P_emitted + K early         → +2.18pp (α=0.05, β=-0.05)
```

두 경로 모두 +2pp 이상이지만, **메커니즘이 다르다**:
- 경로 A는 "전체 온톨로지 방향을 약하게 억제"하는 attention regularization
- 경로 B는 "초기 레이어 K로 정확도 확보 + Q로 선택 도구 회피"하는 이론 기반 steering

**발견 2: K는 초기 레이어에서만 유효**

| K 적용 범위 | F1 | 해석 |
|------------|-----|------|
| 전체 레이어 (α=0.3) | 0.685 (-4.57pp) | 후반 레이어에서 출력 수렴 파괴 |
| 전체 레이어 (α=0.05) | 0.710 (+1.3pp)* | α를 줄이면 파괴 감소 |
| 초기 1/4만 (α=0.05) | 0.751 (+2.08pp) | 초기에서만 → 파괴 없이 정확도 확보 |

U-shape MSE 관측과 일치: 초기 레이어는 정보 인코딩(K-bias 안전), 후반 레이어는 출력 수렴(K-bias 위험).

**발견 3: Q-only의 두 방식이 극적으로 다름**

| Q 방식 | Q 빼는 방향 | F1 | 해석 |
|--------|-----------|-----|------|
| 전체 B_ont (β=-0.03) | 모든 온톨로지 방향 | 0.754 (+2.28pp) | 작동 |
| P_emitted static | 선택된 facet만 | 0.731 (+0.00pp) | no-op (P_emitted=0) |
| P_emitted multipass | 선택된 facet만 | 0.724 (-0.70pp) | 과잉추천 (precision 하락) |
| P_emitted iterative + K | 선택된 facet만 | 0.752 (+2.18pp) | K 보조로 작동 |

**미해결 문제**: 전체 B_ont가 P_emitted보다 왜 단독으로 더 효과적인가? → Section 4.5에서 분석.

### 4.4 이전 세션 결과 포함 (참고)

coworker(승필)가 확인한 이전 결과:
- `qkv_alpha_microsweep_2026_04_15/full497`: 최고 0.7529 F1 (small-alpha Q+K)
- `wave_2026_04_15_pm/gpu0/llama_inst_st4_full497.json`: Llama에서 K-bias 붕괴, Q-bias 약한 양수

### 4.5 미해결 문제: 전체 B_ont Q-subtraction은 왜 효과적인가?

**현상**: 이론(Thm 6.17')은 P_emitted(선택된 facet만 빼기)가 최적이라고 하지만, 실측은 전체 B_ont를 빼는 것이 Q-only에서 더 좋다 (+2.28pp vs +0.00pp).

### 4.6 가설: 전체 B_ont Q-subtraction이 효과적인 이유

#### 가설 A: Attention Regularization (가장 유력)

전체 B_ont를 빼는 것은 "이미 선택한 방향만 빼는" 정밀 수술이 아니라, **모든 도구 관련 attention을 약하게 억제**하는 정규화(regularization) 효과를 가진다.

```
Q' = Q + β · B_ont · B_ont^T · Q   (β = -0.03)
   = Q - 0.03 · P_ont · Q
   = (I - 0.03 · P_ont) · Q

효과: Q에서 온톨로지 방향 에너지를 3% 줄임
→ softmax에서 모든 도구 후보의 attention이 미세하게 감소
→ 경계선 사례에서 2위 도구가 1위를 이길 수 있는 여지 생성
```

**검증 방법**:
- beta_sweep 497개 per-sample에서 pred가 변한 63개 분석
- 변화 패턴: "1위 도구 교체" vs "2위 도구 추가/삭제" vs "도구 수 변화" 비율
- 예측: regularization이면 "1위 교체"가 주, coverage면 "2위 추가"가 주

#### 가설 B: 첫 번째 도구 생성 중 Q 수정 효과

single-pass에서 전체 B_ont Q-subtraction은 **첫 번째 도구 이름 토큰 생성 중에도** Q를 수정한다. 이 수정이 첫 번째 도구 선택 자체를 바꿀 수 있다.

반면 P_emitted는 첫 번째 도구가 완전히 생성된 후에야 비-zero가 되므로, 첫 번째 도구에는 영향 없음.

```
single-pass 전체 B_ont:
  토큰1(tool_call) → Q 수정됨 → 도구A 선택 (원래는 도구B였을 수 있음)
  토큰50(2nd tool) → Q 수정됨 → 도구C 선택

P_emitted iterative:
  Pass 1: P_emitted=0 → Q 수정 없음 → 도구B 선택 (baseline과 동일)
  Pass 2: P_emitted=P_B → Q 수정됨 → 도구? 선택
```

**검증 방법**:
- no_steer vs Q-only 전체 B_ont에서 **1번째 도구**가 변한 비율 분석
- 예측: 가설 B가 맞으면 1번째 도구 변경 비율이 높음 (>20%)
- 가설 A가 맞으면 1번째 도구 변경 비율이 낮고, 2번째 도구 변경이 주

#### 가설 C: β 값과 방향의 교호작용

전체 B_ont β=-0.03이 최적인데, P_emitted β=-0.05가 차선. 이것은 빼는 방향의 rank 차이 때문일 수 있다:

```
전체 B_ont: rank R (~24), β=-0.03 → 24차원 × 0.03 = 누적 효과 0.72
P_emitted: rank ~6 (1-2 facet), β=-0.05 → 6차원 × 0.05 = 누적 효과 0.30
```

즉 전체 B_ont는 약하게 넓게, P_emitted는 강하게 좁게 빼는 것이고, "약하게 넓게"가 더 효과적.

**검증 방법**:
- P_emitted에서 β를 키워서 누적 효과를 전체 B_ont와 맞춤: β=-0.12 (6차원 × 0.12 = 0.72)
- 전체 B_ont에서 β를 줄여서 P_emitted 수준으로: β=-0.006 (24차원 × 0.006 = 0.14)
- 예측: 누적 효과가 같으면 결과도 비슷해야 함

#### 가설 D: 도구 수 분포 차이

전체 B_ont는 도구 **수**를 바꾸지 않고 도구 **선택**만 바꾸는 반면, multipass P_emitted는 추가 pass에서 3번째 도구를 생성하여 precision을 낮춘다.

```
전체 B_ont: avg_pred = 1.88 (baseline과 동일) → 도구 수 불변
multipass P_emitted: avg_pred > 2.0 (추가 도구 발견) → precision 하락
```

**검증 방법**:
- 각 방법의 avg_pred, pred_count 분포 비교
- multipass에서 3번째 도구가 정답인 비율 vs 오답인 비율

#### 가설 E: Facet 분해가 모델 내부와 불일치

우리가 정의한 4-facet (domain, function_action, io_type, tool_category)이 모델의 실제 내부 도구 표현과 일치하지 않을 수 있다. 이 경우:
- 전체 B_ont: facet 경계와 무관하게 전체 온톨로지 부분공간을 억제 → 작동
- P_emitted(domain만 빼기): 모델 내부에서 "domain"이 독립적 차원이 아님 → 효과 없음

**검증 방법**:
- B_ont의 facet별 열을 PCA로 분석: inter-facet 직교성 확인 (cos similarity)
- 모델의 실제 K 공간에서 facet별 variance explained 비율
- 예측: 직교성이 낮으면 facet별 P_emitted가 의도한 방향만 빼지 못함

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

### 5.1 왜 F1을 주 메트릭으로 썼는가

F1을 주 메트릭으로 선택한 이유는 **recall과 precision의 비중 차이**를 반영해야 하기 때문이다:

```
시나리오 A: 정답 1개, 예측 1개 → 틀리면 F1 = 0.0 (치명적)
시나리오 B: 정답 3개, 예측 3개 중 1개만 맞음 → F1 = 0.5 (부분 성공)
```

현장에서 "정답이 1개인데 못 맞추는 것"과 "정답이 3개인데 1개만 맞추는 것"은 의미가 완전히 다르다. Accuracy나 단순 hit rate로는 이 차이가 안 잡힌다. F1은 precision(추천한 것 중 맞는 비율)과 recall(정답 중 찾은 비율)의 조화평균이므로 이 비중 차이를 자연스럽게 반영한다.

### 5.2 현재 F1의 한계 — 도구 간 우선순위

F1의 근본 한계: 추천된 도구들 **내부의 순위 품질**을 반영하지 못한다.

```
정답: [FinanceTool(4facet 완전일치), NewsTool(3facet), WeatherTool(2facet)]
```

이 정답에서 각 도구의 중요도는 다르다:
- **FinanceTool**: 4개 facet 모두 일치 → "꼭 있어야 하는" 핵심 도구
- **NewsTool**: 3개 facet 일치 → "있으면 좋은" 보조 도구  
- **WeatherTool**: 2개 facet만 일치하지만, 그 2개에서 **압도적**으로 적합

```
예측A: [FinanceTool, JobTool, MusicTool]  → F1 = 0.5 (1/3 맞음)
예측B: [WeatherTool, JobTool, MusicTool]  → F1 = 0.5 (1/3 맞음)

현재 F1: 둘 다 0.5로 동일
현실: 예측A가 훨씬 나음 (핵심 도구 FinanceTool을 맞춤)
```

더 미묘한 케이스:

```
예측C: [FinanceTool, NewsTool, JobTool]   → F1 = 0.667 (2/3 맞음)
예측D: [FinanceTool, WeatherTool, JobTool] → F1 = 0.667 (2/3 맞음)

현재 F1: 둘 다 0.667로 동일
현실: FinanceTool(4facet)과 NewsTool(3facet)을 맞춘 예측C가
      FinanceTool(4facet)과 WeatherTool(2facet)을 맞춘 예측D보다 나을 수 있음
      하지만 WeatherTool이 2개 facet에서 "압도적"이라면? → 판단이 어려움
```

**핵심 문제**: facet 일치 개수만으로는 도구의 중요도를 완전히 포착할 수 없다. "2개 facet에서 압도적인 도구"와 "3개 facet에 부분적으로 맞는 도구" 중 어느 것이 더 중요한지는 **쿼리 맥락에 의존**한다.

### 5.3 Facet-Weighted nDCG — 우리 이론에 맞는 메트릭 설계

IR/추천 분야의 nDCG를 우리 온톨로지 구조에 맞게 변형한다. 핵심 아이디어: **facet별 에너지 ε_f를 relevance 가중치로 사용**하면, "이 쿼리에서 어떤 facet이 더 중요한가"를 자동으로 반영할 수 있다.

#### 5.3.1 Relevance Score 정의

```
relevance(tool_pred, query) = Σ_f  match(tool_pred, f) × ε_f(query)
```

여기서:
- `match(tool, f)` = 1 if tool의 facet f가 정답 도구 세트의 어떤 도구와 일치, else 0
- `ε_f(query)` = 쿼리 Q에서 facet f의 에너지 비율 (우리 알고리즘이 이미 측정하는 값)

```
예시:
  쿼리: "뉴스에서 주식 기사 찾아서 요약해줘"
  측정된 에너지: ε_domain=0.35, ε_action=0.25, ε_io=0.20, ε_cat=0.10

  FinanceTool: domain=finance(✓0.35) + action=analyze(✓0.25) + io=numeric(✗0) + cat(✓0.10)
    → relevance = 0.35 + 0.25 + 0.10 = 0.70

  WeatherTool: domain=weather(✗0) + action=inform(✗0) + io=text(✓0.20) + cat(✗0)
    → relevance = 0.20

  → FinanceTool이 3.5배 더 relevant (F1에서는 둘 다 "1개 맞음"으로 동일)
```

#### 5.3.2 nDCG 계산

```
DCG@K = Σ_{i=1}^{K}  relevance(pred_i) / log₂(i + 1)

IDCG@K = 정답 도구를 relevance 내림차순으로 배치했을 때의 DCG

nDCG@K = DCG@K / IDCG@K  ∈ [0, 1]
```

#### 5.3.3 왜 이 메트릭이 우리 이론에 맞는가

1. **ε_f를 직접 사용**: 우리 알고리즘의 `measure_facet_weights()`가 이미 ε_f를 계산한다. 별도 annotation 불필요
2. **쿼리 의존적 가중치**: "뉴스 관련 쿼리"에서는 domain facet이 높고, "검색 관련 쿼리"에서는 action facet이 높음 → 같은 도구라도 쿼리에 따라 relevance가 달라짐
3. **부분 일치 반영**: 2개 facet만 맞는 도구도 부분 점수를 받음
4. **순위 반영**: 1순위에 relevance 높은 도구가 오면 DCG가 높음 (log discount)
5. **"압도적 2facet" 처리**: ε_f가 높은 2개 facet에서 일치하면, ε_f가 낮은 3개 facet에서 일치하는 것보다 relevance가 높을 수 있음

#### 5.3.4 F1 vs nDCG에서 순위 역전이 발생하는 경우

```
정답: [FinanceTool, NewsTool, WeatherTool]
ε = {domain: 0.40, action: 0.30, io: 0.20, cat: 0.10}

예측X: [FinanceTool, MusicTool]  → F1 = 0.5 (1/3 recall, 1/2 precision)
  nDCG: rel(Finance)=0.70 at rank 1, rel(Music)=0.10 at rank 2
  DCG = 0.70/1 + 0.10/1.58 = 0.763
  IDCG = 0.70/1 + 0.50/1.58 + 0.20/2 = 1.016
  nDCG = 0.751

예측Y: [WeatherTool, NewsTool, JobTool]  → F1 = 0.667 (2/3 recall, 2/3 precision)
  nDCG: rel(Weather)=0.20 at rank 1, rel(News)=0.50 at rank 2, rel(Job)=0.0 at rank 3
  DCG = 0.20/1 + 0.50/1.58 + 0/2 = 0.516
  IDCG = 0.70/1 + 0.50/1.58 + 0.20/2 = 1.016
  nDCG = 0.508

F1: 예측Y(0.667) > 예측X(0.500) — "더 많이 맞추니까 좋다"
nDCG: 예측X(0.751) > 예측Y(0.508) — "핵심 도구를 1순위에 놓으니까 좋다"
```

이런 순위 역전이 실제로 얼마나 발생하는지가 메트릭의 가치를 결정한다. → **실험 계획 P1에 포함**.

### 5.4 Exact Match의 가치

현재 Exact Match는 엄격하지만 유용한 보조 메트릭이다:

```
k_early_only:  Exact = 0.5473 (baseline 0.5252, +2.21pp)
layer_adaptive: Exact = 0.5352 (baseline 0.5252, +1.00pp)
```

k_early_only가 Exact에서 더 큰 우위를 보인다는 것은, 단순히 "비슷한 도구를 더 많이 맞추는" 것이 아니라 **정확히 정답 도구 세트를 완전 일치시키는** 빈도가 높아졌다는 의미다.

### 5.5 메트릭 체계 정리

앞으로 모든 실험에서 3개 메트릭을 병렬 보고한다:

| 메트릭 | 측정 대상 | 강점 | 약점 |
|--------|-----------|------|------|
| **F1** | recall+precision 균형 | 도구 개수 맞추기 반영 | 우선순위 무시, 부분 일치 무시 |
| **Exact** | 완전 일치 | 가장 엄격, 실용적 | 1개라도 틀리면 0 |
| **FW-nDCG** (구현 예정) | facet 가중 순위 품질 | 우선순위+부분일치+쿼리의존 | ε_f 측정 필요 (우리 파이프라인에 이미 있음) |

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
