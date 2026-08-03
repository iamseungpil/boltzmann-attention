# 이관 지시 충실도 복원 설계 (2026-08-03)

> 근거: `AX33G_SPLIT_FORENSIC_2026_08_03.md` §7.5 · 런 `bank_ax33n_gpu{0,1}_20260803g`(64 sim 완주)
> 원자료 `sim_results/bank_ax33n_gpu*_20260803g.results.json.gz`(커밋 `583af743`)

## 0. 한 문단

오늘 실패 40 sim 정독에서 **최대 음성 신호는 `ASKED`**("Would you like to be transferred to a human
agent?")였다 — **총 43회 발화 · 있음 n=29 pass .310 vs 없음 n=35 pass .429**. 출처를 찾으니 KB도
도메인 정책도 아닌 **우리 A2**였다. 그리고 도메인 정책 원문을 읽으니, 정책은 이미 **두 개의 조건절로
가드된 형태**로 쓰여 있는데 우리 A2가 그것을 **절대문 하나로 평탄화**해 대화 중간에 주입하고 있었다.
따라서 1차 처방은 레버 추가가 아니라 **우리 문구의 삭제**다.

## 1. 관측 (전부 궤적 단위·[[08]])

| 관측 | 근거 |
|---|---|
| `ASKED` 43회 · n=29 pass .310 / n=35 pass .429 | `ax33g_rescue_scan.py` 전수 64 sim |
| 문구가 8개 이상 sim에서 **축자 동일** | 014t0·032t0·032t1·033t0·033t1·034t0·034t1·035t1 |
| 032 t0: 사용자가 [9][11][13]에서 이관 요구 → 에이전트가 [14][16][18] **3회 더 질문** | 궤적 정독 |
| 034 t1: 사용자 요구 5회 · **도구 호출 0회**로 종료 | 궤적 정독 |
| 035 t0: 엔진이 *"Proceed immediately with transfer_to_human_agents"* → KB 검색 2회 더 | 궤적 정독 |

## 2. 원인 — 정책 대비 우리 A2의 손실·추가

**도메인 정책** `prompts/components/policy_header.md` (에이전트 system prompt에 포함됨 — `results.json`
`info`에서 확인) 축자:

> **§5** *"…ask the user whether they would like to be transferred to a human agent. **If they do, invoke the
> appropriate transfer_to_human_agents tool.** Do this only if you absolutely have to… Do not transfer without
> asking the user first. **This guidance may be overridden by specific scenario-based transfer guidance in the
> knowledge base.**"*
>
> **§6** *"…try to help them first. **If the user asks for a human agent 4 times, then you may invoke the
> transfer_to_human_agents tool.** This guidance may be overridden by specific scenario-based transfer guidance
> in the knowledge base."*

**우리 A2** `a2/banking_knowledge.settings.json` `/gates[1]/ask` · `/gates[1]/notice_text`:

| 정책 요소 | 우리 A2 | 판정 |
|---|---|---|
| "Do not transfer without asking the user first" | 있음 | [[23]] 통과 — 날조 아님 |
| **"may be overridden by … knowledge base"** (§5·§6 **2회**) | **없음** | **손실** |
| **"asks for a human agent 4 times → may invoke"** | **없음** | **손실** |
| `"TRANSFER NOTICE: I have checked the knowledge base and there are no further actions I can take…"` | 있음 | **우리 저작 · 정책에 없음** |
| "send the notice at most ONCE / already asked → do NOT ask again" | 있음(**산문 예외절**) | 미집행 |

세 가지가 겹쳐 실패한다.
1. **절대문 + 산문 예외절** — 모델은 절대문을 붙잡고 예외를 흘린다([[42]]).
2. **검증 불가능한 완결성 주장** — 우리 템플릿이 *"there are no further actions I can take"* 를 선언하게
   만든다. 032/033에서 이는 **거짓**이었다(문서화된 `initial_transfer_to_human_agent_0218`이 존재).
3. **주입 위치** — 정책은 system prompt(턴 0), 우리 문구는 **대화 중간**. 최신성 우위로 우리 것이 이긴다.
   (⚠주입 지점은 `info`에 `TRANSFER NOTICE` 부재·messages에 존재로부터의 **추론**. §10-1에서 코드 확인 필요.)

**7월에 이미 탈출로로 실측해 놓았다** — `t2_gate_patch.py:6154` 주석 축자:
*"038 실측: "I will file 3 disputes…"(SAY)→**TRANSFER NOTICE로 탈출**"*. 조치 없이 유지됨.

## 3. ★[[05]] 결정-시점 3질문 (설계서 상설 의무·[[17]])

**1차 설계(삭제)에 대해:**

| # | 질문 | 답 |
|---|---|---|
| 1 | scaffold **또는 A2**의 도메인-특화를 *순증*시키나? | **NO — 감소한다.** A2에서 우리 저작 문구 2개(`notice_text`·`ask` 산문)를 **제거**하며, 어떤 텍스트도 추가하지 않는다. 정책의 조건절은 **이미 system prompt에 있으므로 복원할 필요조차 없다**(§2 검증). |
| 2 | 모델이 할 수 있는 *유동적 판단*을 결정론에 *동결*하나? | **NO — 반대다.** 우리가 동결해 두었던 "언제 이관하는가"를 정책 원문의 판단으로 **되돌려준다**. |
| 3 | scaffold가 모델 대신 *도메인 행동을 수행*하나? | **NO.** 텍스트를 빼는 것뿐. 도구 호출 0. |

⇒ 셋 다 NO. **기본=GO.** ([[05]] 메타 경고 준수: "A2로 옮긴 것=제거 아님"의 shell game이 아니라 **실제 삭제**.)

**2차 설계(가드된 조립)에 대해 — 조건부로만:**

| # | 답 |
|---|---|
| 1 | **부분 YES** — 조각·가드가 A2에 추가된다. 단 내용은 **정책 축자**이고 새 도메인 판단은 없다. |
| 2 | NO — 조각을 *빼는* 방향이라 모델 재량이 늘어난다. |
| 3 | NO. |

⇒ 하나가 YES이므로 **기본=NO. 1차 삭제 측정이 손실을 보일 때만 정당화.**

## 4. 설계

### 4.1 1차 — 삭제 (arm B″)

`a2/banking_knowledge.settings.json` `/gates[1]` 에서 제거:
- `notice_text` 전체
- `ask` 중 절대문 재진술 + 고정 템플릿 지시 + 산문 예외절

남길 것: 게이트의 **탐지·계측**(언제 이관-류 호출이 났는지)은 유지. 발화만 제거.

근거: 정책 §5·§6이 system prompt에 이미 있고, 그것은 **override 조항과 4회 임계를 모두 포함**한다.
우리 문구는 그 위에 덮인 열화 사본이다.

### 4.2 2차 — 가드된 조각 조립 (arm C, 조건부)

1차가 과잉 이관을 유발하면(§6 실패 기준) 그때만. 조각은 **정책 축자**, 가드는 **정책이 명시한 조건**:

```
fragment F_ask  := "<정책 §5 축자>"
guard  G1 := env_transfer_requests >= 4          # 정책 §6 축자 · 환경이 "Transfer request #N submitted" 로깅
guard  G2 := scenario_doc_retrieved              # 정책 §5·§6 "may be overridden by …" · 문서 id 대조
include F_ask  iff  ¬G1 ∧ ¬G2
```

**엔진은 저작된 조각을 고를 뿐 문장을 만들지 않는다** ⇒ [[10]] 선택기 역할 유지.

## 5. 가드의 형식 제약 (2607.22868 정리 6·7)

정리 6: 감소 카운터 2개면 **명세 비자명성 결정 불가** — 가드가 서로를 참조하면 "이 조각이 켜지기는
하는가"를 정적으로 판정할 수 없다.
정리 7 축자: *"Each guard tests one counter (or counter plus that amount) against a binary-encoded constant,
**never another counter**"* + 분리성 + 키-지역성 + **컴포넌트가 입력을 분할**(미해당 → bad).

⇒ **하드 규칙**: ① 가드는 단일 카운터/플래그 vs 상수 ② **가드 간 상호참조 금지** ③ 각 조각의 가드는
전면적 ④ 조각 수 N은 측정된 것만 늘린다(2^N 미검증 구성 방지).

**비-가법성 경고**: [arXiv:2604.14862](https://arxiv.org/pdf/2604.14862) 축자 *"the two channels interact
**non-additively**"* — 조각별 검증이 조립의 정당성을 주지 않는다. **N=1로 시작한다.**

## 6. 측정 (짝비교·[[19]] 합성 아님·단일변수)

| arm | 내용 |
|---|---|
| **A** | 현행 |
| **B″** | §4.1 삭제 |

- 대상: **front32 × nt2**(현 런과 동일 조건·비교 가능). 우선 **ASKED 발생 sim 29건 중 front32 교집합**.
- 1급 지표: `ASKED` 발화 수 · `transfer_to_human_agents` 호출률 · pass.
- **Δspurious(필수)**: **부당 이관**(gold가 이관을 요구하지 않는 태스크에서의 이관) 증가량.
  등대 제1원리 — 레버를 빼는 것도 레버다. Δspurious > 0이면 4.2로 간다.
- 판정 전 **궤적 전수 포렌식**([[08]]) — 집계에서 직행 금지.

## 7. 이 설계가 고치지 **않는** 것 (non-goals·정직)

- **계열 A의 liveness 부분**: "결국 도구를 호출해야 한다"는 pre-call 게이트로 집행 불가(2607.22868 정리 1·
  명제 1). 사용자가 `###STOP###`으로 끝내는 sim(003 t0·034 t1)은 **어떤 게이트도 걸 자리가 없다**.
- **회수 공백**: 032/033은 `initial_transfer_to_human_agent_0218`이 궤적에 한 번도 안 나타났다.
  삭제해도 그 이름을 모르면 못 부른다. **레지스트리×문서 조인이 병행되어야 닫힌다**([[19]]).
- 018/020/021의 인계 와이어 포맷·전송 손실(별건).

## 8. 위험·선행 부정 증거

- **과잉 이관**: 정책이 "정말 필요할 때만"이라 한 이유가 있다. Δspurious가 이를 잰다.
- **[[07]]**: 삭제는 soft 조작이라 보장이 아니다. 잔여는 측정으로만 안다.
- **[[42]]**: 프롬프트만으로는 규칙 준수가 안 된다 — 이 설계는 *준수를 얻으려는 것이 아니라 우리가
  만든 간섭을 걷는 것*이다. 이 구분을 흐리지 말 것.
- **fleet/서브에이전트 대안은 채택하지 않음**: `RATE_SUBAGENT_DESIGN_2026_07_18` §2d가 짝비교로 반증
  (서브 요청 **172,731 토큰** 컨텍스트 초과 → 폴백 → 47/47 판정불가가 discrepant 0으로 **위장**). C6 fleet
  저-ROI. 필요해지면 §4.2 이후에 재검토.

## 9. 구현 표면 ([[24]])

- 수정 = **정본 층** `a2/banking_knowledge.settings.json`(및 `split/banking_knowledge.core.json`).
  `gate.json`은 생성물이므로 직접 수정 금지(2026-08-02 P9 死코드 재발 방지).
- 편집 후 **`load_domain_a2()` 병합 확인** + `test_a2_three_layer.py`.
- 엔진 코드 변경 **0** (1차). `scaffold_guard.py` 감시 대상에 A2 조각 저장소 추가는 §4.2 착수 시 **선행 조건**.

## 10. 열린 검증 항목 (착수 전)

1. **주입 지점 확인** — `/gates[1]/ask`가 실제로 대화 중간에 주입되는지 코드로 확인(현재는 `info` 부재·
   messages 존재로부터의 추론).
2. 다른 도메인(retail/airline/telecom) A2에 동형 문구가 있는지 grep — 있으면 같은 손실이 반복 중.
3. `ASKED` 29 sim 중 front32 교집합 수 확인(측정 표본 크기).
