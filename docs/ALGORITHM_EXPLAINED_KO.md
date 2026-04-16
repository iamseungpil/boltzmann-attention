# 온톨로지 기반 도구 선택 스티어링: Q와 K의 역할

> **목적**: coworker 가 알고리즘 v1→v4 진화를 이해하기 위한 문서
> **날짜**: 2026-04-16

---

## 1. 문제 정의: 현실의 도구 선택은 왜 어려운가

### 1.1 현장 상황

```
사용자: "뉴스에서 주식 관련 기사 찾아서 요약해줘"
```

현장에는 수천 개의 도구가 파편화된 DB로 존재한다:

| 도구 | domain | function_action | io_type | 비고 |
|------|--------|-----------------|---------|------|
| NewsTool | news | retrieve | text | |
| FinanceTool | finance | analyze | numeric | |
| ResearchFinder | research | retrieve | text | NewsTool과 기능 중복 |
| ResearchHelper | research | summarize | text | |
| RepoTool | research | search | text | |

**문제 3가지:**

1. **중복 도구**: `NewsTool`과 `ResearchFinder`는 둘 다 `retrieve`인데, domain만 다름
2. **다중 facet 매칭**: 위 질문은 `news × retrieve` + `finance × summarize` = **2개 도구** 필요
3. **사용자 선호**: 같은 기능이라도 사람마다 선호하는 도구가 다름

### 1.2 온톨로지: 도구를 직교 축으로 분해

도구를 4개 **직교하는 facet**으로 분류한다:

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

**핵심 아이디어**: 사용자 질문도 같은 facet 공간에 투영할 수 있다.

```
"뉴스에서 주식 기사 찾아서 요약해줘"
→ 필요 facet: {domain: [news, finance], function_action: [retrieve, summarize]}
→ 정답 도구: NewsTool (news × retrieve) + FinanceTool (finance × summarize)
```

### 1.3 K와 Q가 해결하는 것

이제 핵심 질문: **LLM의 attention 메커니즘에서 K와 Q를 어떻게 조작하면 도구 선택이 좋아지는가?**

| 역할 | 누가 | 비유 | 효과 |
|------|------|------|------|
| **정확도** (precision) | **K** (Key) | 망원경 초점 맞추기 | 온톨로지 방향으로 K를 증폭 → 정답 도구에 attention 집중 |
| **탐색 범위** (coverage) | **Q** (Query) | 이미 본 곳 치우기 | 이미 선택한 facet 방향을 Q에서 제거 → 다음 도구 탐색 유도 |

---

## 2. 수학적 기초: Attention에서 K와 Q가 하는 일

### 2.1 표준 Attention

```
Attention(Q, K, V) = softmax(Q K^T / √d) V
```

여기서:
- `Q ∈ R^{T×d}`: 쿼리 — "내가 뭘 찾고 있나"
- `K ∈ R^{S×d}`: 키 — "각 위치에 뭐가 있나"
- `softmax(Q K^T)`: attention 가중치 — "어디를 볼 것인가"

**도구 선택 맥락에서:**

```
Q_t = "현재 시점에 모델이 다음 토큰으로 어떤 도구를 찾고 있는가"
K_s = "프롬프트의 s번째 위치에 어떤 도구 정보가 인코딩되어 있는가"

attention_weight(t, s) = softmax(q_t · k_s / √d)
    → 이 값이 클수록 = s 위치의 도구 정보를 더 많이 참조
```

### 2.2 온톨로지 기저 B_ont

도구의 의미 구조를 **attention 공간 안의 방향(basis)**으로 인코딩한다:

```
B_ont ∈ R^{d × R}    (per layer, per head)
```

여기서 `R`은 온톨로지 rank (보통 8~16). 이 basis의 열(column)은 facet별 의미 방향:

```
B_ont = [ b_news | b_finance | b_retrieve | b_analyze | ... ]
           ↑         ↑            ↑            ↑
        domain    domain     func_action   func_action
        facet     facet        facet         facet
```

**Facet별 projection 연산자:**

```
P_f = B_f · B_f^T    (d × d 행렬)
```

여기서 `B_f`는 facet `f`에 해당하는 열들만 모은 부분행렬.

- `P_news · k` = k 벡터에서 "news" 방향 성분만 추출
- `P_retrieve · k` = k 벡터에서 "retrieve" 방향 성분만 추출

---

## 3. K-bias: 정확도를 높이는 메커니즘

### 3.1 수식

```
K' = K + α · (K · B_ont) · B_ont^T
   = K + α · P_ont · K
```

- `α > 0`: 증폭 강도 (보통 0.03 ~ 0.3)
- `P_ont = B_ont · B_ont^T`: 온톨로지 전체 방향으로의 projection

### 3.2 구체적 예제: 1개 도구 선택

```
질문: "오늘 날씨 알려줘"
정답: WeatherTool
후보: [WeatherTool, NewsTool, MusicTool, FinanceTool, ...]
```

**K 공간에서 일어나는 일:**

```
원래 K 벡터들 (프롬프트 내 각 도구 설명 위치):
  k_weather = [0.3, 0.1, 0.8, ...]   ← "weather" 방향 성분 0.8
  k_news    = [0.7, 0.2, 0.1, ...]   ← "weather" 방향 성분 0.1
  k_music   = [0.1, 0.9, 0.05, ...]  ← "weather" 방향 성분 0.05
```

**K-bias 적용 후 (α=0.3):**

```
k'_weather = k_weather + 0.3 × P_ont · k_weather
           = [0.3, 0.1, 0.8, ...] + 0.3 × [*, *, 0.8×boost, ...]
           ← "weather" 방향이 증폭됨

k'_news    = k_news + 0.3 × P_ont · k_news
           = [0.7, 0.2, 0.1, ...] + 0.3 × [*, *, 0.1×boost, ...]
           ← "weather" 방향 성분이 작아서 증폭도 작음
```

**Attention 변화:**

```
                    증폭 전          증폭 후
q · k_weather  =    0.72      →     0.89  ↑↑  (온톨로지 방향 성분 크니까 많이 증가)
q · k_news     =    0.68      →     0.71  ↑   (온톨로지 방향 성분 작으니까 조금 증가)
q · k_music    =    0.45      →     0.46  →   (거의 변화 없음)

→ softmax 후: WeatherTool의 attention 확률이 올라감 → 정답 선택률 증가
```

### 3.3 실험 결과: K-bias는 1개 도구 선택에서 강력

```
Subtask1 (1개 도구 선택, N=995):
                    Baseline    K-bias(α=0.3)    차이
  Llama-3.1-8B      62.31%       77.39%        +15.08pp ✓
  Qwen2.5-7B        75.58%       86.73%        +11.16pp ✓
```

### 3.4 K-bias의 한계: 멀티 도구에서 실패

```
질문: "뉴스에서 주식 기사 찾아서 요약해줘"
정답: [NewsTool, FinanceTool]    ← 2개 필요
```

K-bias는 **고정(stationary)**이다 — 매 디코딩 스텝마다 같은 증폭을 한다:

```
Step 1: K' = K + α · P_ont · K  →  NewsTool 선택  ✓
Step 2: K' = K + α · P_ont · K  →  NewsTool 또 선택  ✗ (같은 증폭이니까!)
```

K-bias는 "어느 도구가 이미 선택되었는지" 모른다.

```
Subtask4 (멀티 도구 선택):
                    Baseline    K-bias(α=0.3)    차이
  Qwen2.5-7B        75.33%       65.33%        -10.00pp ✗ (악화!)
  SEKA amp=2.0       -            11.00%        -64.33pp ✗ (붕괴)
```

**결론: K는 "어디에 초점을 맞출지"는 잘 하지만, "이미 본 곳을 피하기"는 못 한다.**

---

## 4. Q-coverage: 탐색 범위를 넓히는 메커니즘

### 4.1 수식

```
Q' = Q + β · P_emitted · Q      (β < 0, 보통 -0.03 ~ -0.1)
   = Q - |β| · P_emitted · Q
```

- `P_emitted = Σ_{이미 선택된 도구 s} P_{facet(s)}`: 이미 선택한 도구들의 facet 방향
- β가 음수 → **이미 선택된 방향을 Q에서 빼기**

### 4.2 구체적 예제: 2개 도구 선택

```
질문: "뉴스에서 주식 기사 찾아서 요약해줘"
정답: [NewsTool, FinanceTool]
```

**Step 1: 첫 번째 도구 생성 (아직 아무것도 선택 안 함)**

```
P_emitted = 0 (빈 집합)
Q' = Q - 0 = Q  (변화 없음)
→ 모델이 가장 강한 facet 방향으로 생성: NewsTool 선택 ✓

emitted = {NewsTool}
NewsTool의 facet: domain=news, function_action=retrieve
→ P_emitted = P_news + P_retrieve
```

**Step 2: 두 번째 도구 생성 (NewsTool 이미 선택됨)**

```
P_emitted = P_news + P_retrieve

Q' = Q - |β| · (P_news + P_retrieve) · Q
   = Q에서 "news"와 "retrieve" 방향 성분을 약화시킴

원래 Q:        [news:0.7, finance:0.5, retrieve:0.6, summarize:0.4, ...]
Q' (빼기 후):  [news:0.3, finance:0.5, retrieve:0.2, summarize:0.4, ...]
                  ↓↓                        ↓↓
              news 약해짐              retrieve 약해짐
              → finance가 상대적으로 강해짐   → summarize가 상대적으로 강해짐
```

**Attention 변화:**

```
                    Step 1 Q        Step 2 Q' (news/retrieve 약화됨)
q · k_NewsTool  =    0.89     →     0.52  ↓↓ (이미 선택됨 → 억제)
q · k_FinanceTool =  0.61     →     0.73  ↑↑ (finance/summarize 상대적 강화)
q · k_MusicTool  =   0.30     →     0.31  →  (무관한 도구 변화 적음)

→ softmax 후: FinanceTool의 attention 확률 상승 → 두 번째 정답 선택 ✓
```

### 4.3 K와 Q의 역할 분리 요약

```
┌─────────────────────────────────────────────────────────────────┐
│                     도구 선택 프로세스                            │
│                                                                 │
│  Step 1: K가 정확도 담당                                         │
│    K' = K + α · P_ont · K                                       │
│    "온톨로지 방향으로 증폭해서 정답 도구에 초점"                    │
│    → NewsTool 선택                                               │
│                                                                 │
│  Step 2: Q가 탐색 범위 담당                                      │
│    Q' = Q - |β| · P_{news+retrieve} · Q                        │
│    "이미 본 방향을 빼서 나머지 facet으로 탐색 유도"                │
│    + K가 여전히 나머지 facet에 대해 정확도 유지                    │
│    → FinanceTool 선택                                            │
│                                                                 │
│  Step 3: 더 이상 매칭되는 facet 없으면 중단                       │
│    ε_q = ||B_ont^T q||² / ||q||² < threshold                   │
│    "Q에서 온톨로지 에너지가 바닥나면 멈춤"                         │
└─────────────────────────────────────────────────────────────────┘
```

---

## 5. 버전별 알고리즘 진화

### Version 0: Q-coverage Only (탐색 범위만 — K 건드리지 않음)

```python
# Q에서 이미 선택된 facet 방향만 빼기. K는 원본 그대로.
emitted = {}
while n_emitted < max_tools:
    P_emitted = Σ P_{facet(s)} for s in emitted
    Q' = Q - |β| · P_emitted · Q     (β < 0)
    K' = K                            (변화 없음)

    tool = generate_one_tool(Q', K')
    emitted.add(tool)
```

**Q만 조작하면 어떻게 되는가:**

```
질문: "뉴스에서 주식 기사 찾아서 요약해줘"
정답: [NewsTool, FinanceTool]

Step 1: P_emitted = 0, Q' = Q (변화 없음)
  → 모델의 원래 능력으로 NewsTool 선택 ✓

Step 2: P_emitted = P_news + P_retrieve
  Q' = Q에서 news, retrieve 방향 약화
  → finance, summarize 방향이 상대적으로 강해짐
  → FinanceTool 선택 가능 ✓ ... 하지만

문제: K가 증폭되지 않아서, 원래 모델이 FinanceTool과 다른 도구를
      잘 구분 못 하면 Q를 아무리 돌려도 정확도가 낮음
```

**구체적 Attention 수치 예시:**

```
                    Step 1 (Q 원본)   Step 2 (Q에서 news/retr 빼기)
q · k_NewsTool  =    0.72       →     0.52  ↓↓ (Q에서 news 빠짐)
q · k_FinanceTool =  0.61       →     0.63  ↑  (finance 상대적 강화, 하지만 소폭)
q · k_MusicTool  =   0.45       →     0.46  →  (무관)
q · k_JobTool    =   0.58       →     0.60  ↑  (career/search도 같이 올라옴)

→ FinanceTool(0.63)과 JobTool(0.60) 차이가 0.03밖에 안 됨
→ K에 정확도 증폭이 없으니, 비슷한 도구 사이에서 헷갈림
```

- **할 수 있는 것**: 이미 선택한 도구 반복 방지 (탐색 범위 확보)
- **못 하는 것**: 남은 후보 중 정답을 정확히 골라내기 (정확도 부족)
- **비유**: 지우개는 있는데 망원경이 없는 상태 — 이미 본 곳은 지우지만, 나머지 중 어디를 봐야 하는지 초점이 안 맞음

```
[v0 메커니즘 그림]

Step 1:  Q ─────────────→ Att → NewsTool      (모델 원래 능력)
         K (원본) ────────→ Att

Step 2:  Q - |β|·P_{news,retr}·Q ──→ Att → FinanceTool? JobTool?
         K (원본) ──────────────────→ Att    (K 증폭 없어서 구분 어려움)
```

**실험 결과:**

```
Subtask4 (멀티 도구 선택):
                    Baseline    Q-only(β=-0.1)    차이
  Qwen2.5-7B        0.697        0.697          +0.0pp (변화 없음~미미)
```

Q만으로는 탐색 방향은 바꿀 수 있지만, K의 정확도 보조 없이는 유의미한 개선이 어렵다.

---

### Version 1: K-bias Only (단일 도구 전용)

```python
# 모든 레이어, 모든 스텝에서 동일한 증폭
K' = K + α · B_ont · B_ont^T · K
```

- **할 수 있는 것**: 1개 도구 정확도 향상 (+15pp Llama, +11pp Qwen)
- **못 하는 것**: 2개 이상 도구 선택 (고정 증폭이라 같은 도구 반복)
- **실험**: Subtask1 (1개 도구) 전용

```
[v1 메커니즘 그림]

Q ──────────────────→ Attention ──→ 도구 1개 선택
K ──→ K + α·P·K ──→ Attention       (항상 같은 방향 증폭)
```

---

### Version 2: Dynamic Q+K (멀티 도구 — 단일 턴)

```python
# 도구를 하나 선택할 때마다 hook을 갱신
emitted = {}
while ε_q > threshold AND n_emitted < max_tools:
    # 1. K: 아직 선택 안 된 facet만 증폭
    P_remaining = P_ont_full - P_emitted
    K' = K + α · P_remaining · K

    # 2. Q: 이미 선택된 facet을 빼기
    Q' = Q + β · P_emitted · Q        (β < 0)

    # 3. 생성 → 도구 하나 파싱
    tool = generate_one_tool(Q', K')
    emitted.add(tool)
    P_emitted += P_{facet(tool)}

    # 4. ε_q 체크 — 남은 에너지 측정
    ε_q = measure_remaining_energy(Q)
```

**v1 대비 핵심 변화:**
- K의 증폭 대상이 **동적으로 줄어든다** (이미 선택된 facet 제외)
- Q에서 **이미 선택된 방향을 빼서** 다음 도구로 유도
- **ε_q 기반 자동 중단**: 더 이상 매칭할 facet이 없으면 멈춤

```
[v2 메커니즘 그림]

Step 1:  Q ─────────────→ Att → NewsTool
         K + α·P_all·K ─→ Att    emitted = {news, retrieve}

Step 2:  Q - |β|·P_{news,retr}·Q ──→ Att → FinanceTool
         K + α·(P_all - P_{news,retr})·K → Att
                                          emitted = {news, retrieve, finance, summarize}

Step 3:  ε_q < threshold → 중단. 최종 출력: [NewsTool, FinanceTool]
```

---

### Version 3: Facet-Adaptive Weighting (쿼리 적응형)

v2의 문제: 모든 facet을 **동일 가중치**로 투영했다. 하지만 질문마다 중요한 facet이 다르다.

```python
# Step 0: 현재 쿼리에서 각 facet의 에너지 측정
ε_f(q) = ||B_f^T · q||² / ||q||²    (facet f가 쿼리에 얼마나 포함되어 있나)

# 예시: "뉴스에서 주식 기사 찾아서 요약해줘"
ε_news      = 0.35  ← Q가 news 방향에 많이 투영됨
ε_finance   = 0.25
ε_retrieve  = 0.20
ε_summarize = 0.15
ε_play      = 0.02  ← 거의 무관
ε_weather   = 0.01  ← 거의 무관

# Step 1: 가중치 기반 투영 (에너지 큰 facet에 집중)
w_f = ε_f / Σε_f    (정규화)
P_weighted = Σ w_f · P_f

# Step 2: 도구 선택 후 해당 facet 가중치를 decay
# NewsTool 선택 → w_news *= 0.2, w_retrieve *= 0.2
# → 다음 스텝에서 finance, summarize의 상대적 가중치 증가
```

**v2 대비 핵심 변화:**
- **쿼리 적응형**: 질문의 내용에 따라 어떤 facet을 더 강하게 조작할지 자동 결정
- **Soft decay**: 선택된 facet을 완전 제거하지 않고 0.2배로 약화 (중복 도구 허용)

```
[v3 메커니즘 그림]

Query → 에너지 측정 [ε_news=0.35, ε_fin=0.25, ε_retr=0.20, ...]
                           ↓
                    가중 투영: P = 0.35·P_news + 0.25·P_fin + ...
                           ↓
Step 1: Q + K + P_weighted → NewsTool
        decay: w_news × 0.2, w_retrieve × 0.2
                           ↓
Step 2: Q + K + P_reweighted → FinanceTool
        (finance, summarize가 이제 최대 가중치)
                           ↓
Step 3: ε_total < 0.05 → 중단
```

---

### Version 4: Multi-Turn Facet Narrowing (멀티 턴 대화)

**현실 시나리오**: 사용자는 한 번에 모든 것을 말하지 않는다.

```
Turn 1: "여행 계획 좀 도와줘"         ← 추상적 (domain=travel만 특정)
Turn 2: "호텔이랑 항공편 같이 검색해줘"  ← 구체적 (function=search, io=structured 추가)
Turn 3: "가격 비교해서 추천해줘"        ← 더 구체적 (function=recommend, domain=finance 추가)
```

**v4는 3단계 계층(hierarchy)으로 처리한다:**

```
┌──────────────────────────────────────────────────────────────┐
│ Level 1: 대화(Conversation) 수준 — EMA 메모리                  │
│                                                              │
│   conv_weights = γ · 이번턴측정값 + (1-γ) · 이전턴메모리       │
│   (γ=0.6: 현재 턴 60%, 과거 누적 40%)                         │
│                                                              │
│   Turn 1: conv = [travel:0.8, -:0.2]         ← 추상적        │
│   Turn 2: conv = [travel:0.5, search:0.3, structured:0.2]   │
│   Turn 3: conv = [travel:0.3, search:0.2, recommend:0.25,   │
│                    finance:0.15, ...]          ← 구체적       │
│                                                              │
│   → 턴이 진행될수록 facet이 구체화됨                           │
├──────────────────────────────────────────────────────────────┤
│ Level 2: 턴(Turn) 수준 — 도구 선택 후 facet decay              │
│                                                              │
│   Turn 2 내부:                                                │
│     도구1: HotelSearch 선택 → travel, search facet decay      │
│     도구2: FlightSearch 선택 → 이미 travel decay했으니          │
│            search만 추가 decay → 추천 facet 강화               │
├──────────────────────────────────────────────────────────────┤
│ Level 3: 토큰(Token) 수준 — 매 토큰 Q/K hook                  │
│                                                              │
│   각 디코딩 스텝마다:                                          │
│     K' = K + α · P_weighted · K     (가중치 반영 증폭)         │
│     Q' = Q + β · P_weighted · Q     (이미 본 방향 빼기)        │
└──────────────────────────────────────────────────────────────┘
```

**v3 대비 핵심 변화:**
- **EMA 메모리**: 이전 턴의 facet 정보를 누적 — 대화가 진행될수록 정확해짐
- **턴 종료 시 decay**: 한 턴에서 선택한 도구들의 facet을 다음 턴에 반영
- **추상→구체 자연스러운 수렴**: Turn 1에서 넓게 탐색, Turn 3에서 좁게 집중

### v4 알고리즘 상세 (per turn)

```python
class ConversationState:
    conv_weights = None   # 대화 수준 EMA 메모리

def process_turn(user_message, conversation_state):
    # 1. 현재 턴 쿼리에서 facet 에너지 측정
    measured = measure_facet_weights(user_message)
    # 예: Turn 2 → {travel: 0.3, search: 0.5, structured: 0.2}

    # 2. 대화 메모리와 블렌딩 (EMA)
    if conversation_state.conv_weights is None:
        weights = measured                           # 첫 턴
    else:
        weights = γ * measured + (1-γ) * conv_weights  # 후속 턴
    # Turn 1 메모리 [travel:0.8] + Turn 2 측정 [search:0.5]
    # → 블렌드: [travel:0.5, search:0.3, structured:0.12, ...]

    # 3. 가중 투영자 설치 & 도구 생성 루프
    tools_this_turn = []
    for step in range(max_tools_per_turn):
        install_hooks(weights, α, β)
        tool = generate_one_tool()
        if tool is None:
            break
        tools_this_turn.append(tool)
        # 선택된 도구의 facet을 decay
        for facet in tool.facets:
            weights[facet] *= decay    # decay=0.2

    # 4. 대화 메모리 업데이트 (decay 반영)
    conversation_state.conv_weights = weights
    # → 다음 턴에서 이미 처리된 facet은 약화된 상태로 시작

    return tools_this_turn
```

---

## 6. 실험 결과 요약

### 6.1 Single-tool (Subtask1): K-bias 효과

| 모델 | Baseline | K-bias α=0.3 | 차이 |
|------|----------|-------------|------|
| Llama-3.1-8B | 62.31% | 77.39% | **+15.08pp** |
| Qwen2.5-7B | 75.58% | 86.73% | **+11.16pp** |

**K-bias는 1개 도구 정확도에 확실히 효과적.**

### 6.2 Multi-tool (Subtask4): K-only vs Q+K

| 방법 | Qwen F1 | vs Baseline |
|------|---------|-------------|
| Baseline (no steer) | 0.753 | — |
| K-bias only (α=0.3) | 0.653 | **-10.0pp** (악화) |
| SEKA amp=2.0 | 0.110 | **-64.3pp** (붕괴) |
| Q+K dynamic (v2) | 실험중 | 목표 +2~5pp |

**K만으로는 멀티 도구 선택이 불가능 — Q의 탐색 범위 조절이 필수.**

### 6.3 방향 특이성 (Null-control)

이 알고리즘이 진짜 온톨로지 의미 방향을 사용하는지 검증:

| 조건 | Subtask4 F1 | 의미 |
|------|-------------|------|
| 실제 B_ont | 0.685 | 온톨로지 방향 사용 |
| Random 방향 | 0.000 | 아무 방향 → 완전 실패 |
| Feature-shuffle | 0.000 | facet 섞기 → 완전 실패 |

**+68.5pp 갭**: 실제 온톨로지 방향만이 동작. 랜덤이나 섞은 방향은 쓸모없음.

---

## 7. 벤치마크 테스트 계획

### 7.1 MetaTool Benchmark (현재 사용 중)

| Subtask | 테스트 내용 | 정답 도구 수 | 비고 |
|---------|------------|-------------|------|
| Subtask1 | 유사 도구 10개 중 1개 선택 | 1개 | K-bias 효과 측정 |
| Subtask4 | 복합 질문에서 여러 도구 선택 | 2~4개 | Q+K 효과 측정 |

### 7.2 버전별 테스트 매트릭스

```
         Subtask1      Subtask4        Multi-turn
         (1개 도구)    (N개 도구)      (턴별 도구)
v0(Q)    → +0pp       → +0pp          해당없음       ← 탐색만, 정확도 부족
v1(K)    ✓ +15pp      ✗ -10pp         해당없음       ← 정확도만, 탐색 불가
v2(Q+K)  ✓ +15pp      목표 +2~5pp     해당없음
v3(적응)  ✓ +15pp      목표 +5pp       해당없음
v4(멀티턴) ✓ +15pp      목표 +5pp       목표 +10pp
```

### 7.3 Multi-turn 벤치마크 설계 (v4용)

**합성 데이터 (즉시 가능):**
- MetaTool Subtask4 항목을 2-turn 대화로 분리
- Turn 1: 첫 번째 도구의 domain만 언급하는 추상적 질문
- Turn 2: 나머지 도구가 필요한 구체적 질문
- 정답: 양 턴의 도구 합집합

**실제 데이터 (추후):**
- τ²-bench: 소매 도메인 멀티턴 도구 호출 벤치마크
- BFCL v3: 멀티턴 function calling 표준 벤치마크

### 7.4 핵심 메트릭

| 메트릭 | 수식 | 의미 |
|--------|------|------|
| F1 | 2·P·R/(P+R) | 선택된 도구와 정답 도구의 일치도 |
| Exact Match | 1[pred == gt] | 정답과 완전 일치 비율 |
| ε_q (Thm 6.20) | \|\|B^T q\|\|²/\|\|q\|\|² | 남은 온톨로지 에너지 — 자동 중단 기준 |
| AUROC | - | ε_q가 성공/실패를 얼마나 잘 예측하는지 (0.976 달성) |

---

## 8. 한눈에 보는 버전 진화

```
v0 ─────→ v1 ─────→ v2 ──────→ v3 ─────────→ v4
Q only    K only    Q + K      Query-adaptive  Multi-turn
                               Q + K           Q + K + Memory

탐색만    정확도만   N도구       N도구            N도구 × M턴
+0pp     +15pp     +탐색범위↑   +쿼리적응↑        +대화맥락↑

"지우개   "망원경"  "망원경+    "자동초점        "대화하며
 만 있음"          지우개"     망원경+지우개"    점점 좁히는
                                              자동초점 망원경"
```

**핵심 메시지:**
- **K = 정확도** (온톨로지 방향으로 초점)
- **Q = 탐색 범위** (이미 본 방향을 빼서 나머지로)
- **v1→v4**: 고정 → 동적 → 적응적 → 대화 수준 메모리
- **왜 새 알고리즘이 필요한가**: 기존 방법(SEKA 등)은 K만 조작 → 멀티 도구에서 -64pp 붕괴. Q 조작이 반드시 필요.
