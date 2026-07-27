# A·B·C 처방 설계서 — 폭주 디코드 / 전환 실패 / 반복 어트랙터 (2026-07-27)

> 사용자 지시: "A,B,C 처방 설계서 + **폭주의 근본 원인**을 폭주 내용 자체로 규명 + 035/019 등 pass→fail을
> 이전 궤적과 per-step 대조 + 축 C의 반복을 per-step 분해."
> 근거 = day4b(`bank_day4bfront{A,B}_20260727`) 전수 포렌식 · day3/day2 대조 · vLLM serve 로그 · 원장 C196~C205.
> **본 문서는 설계까지.** 현행 day4b B-arm이 아직 도는 중 — 구현·배포는 완주 후.

---
## §1 축 A — 폭주 디코드의 **근본 원인** (폭주 원문 해부·[S])

### §1a 폭주 산출물의 해부 (006 m4·33,332자 원문)
```
<tool_call>\n{"name":"give_discoverable_user_tool","arguments":{"discoverable_tool_name":
"apply_for_credit_card","arguments":"{\"card_type\":\"EcoCard\",\"user_id\":\"bob_nakamoto\"}"}}</tool_call>
<tool_call>[공백 95칸]\n{ ...동일 블록... }</tool_call>        ← ×6 반복(공백 패딩이 매 회 삽입)
<tool_call>  ...열고 닫지 않음...  \r\r\r\n\r\r\r\n ...      ← **8번째 블록에서 붕괴**·31,321자 잔여·cap서 절단
```
### ★★정정(2026-07-27·재파싱 실증): **JSON 형식 위반이 원인이 아니다**
닫힌 `<tool_call>` 블록 **7개 전부 `json.loads` 유효**(전부 동일한 `give_discoverable_user_tool` 호출·인자 키도 정상).
문법이 깨진 곳은 **닫히지 않은 8번째 블록**뿐이고, 그 안에서 `\r\r\r\n` 퇴화가 31,321자 이어지다 cap에 잘렸다.
⇒ 모델은 **형식을 못 지킨 게 아니라 "멈추지" 못했다**(첫 호출 뒤 EOS 대신 같은 블록을 계속 생성).
⇒ 그리고 **vLLM hermes 파서는 all-or-nothing** — 미종결 8번째 때문에 **유효한 7개를 통째로 폐기**하고
   전체를 content로 넘긴다(serve 로그 `json.loads(match…)` 예외와 정합).

### §1b 인과 사슬 (5단·각 단계 실측 근거)
| 단계 | 내용 | 근거 |
|---|---|---|
| ①**정지 실패(핵심)** | 유효한 tool_call 1개를 낸 뒤 **EOS를 내지 않고** 같은 블록을 계속 생성 — 형식 문제가 아니라 **종료 실패** | 닫힌 블록 7/7 **JSON 유효**(재파싱 실증) |
| ②**복사 루프** | 직전 자기 출력이 다음 토큰 분포를 지배 → 동일 블록 반복(induction/copy attractor)·공백 패딩 오염 누적 | 7블록 전부 동일 호출·95칸 공백런 |
| ③**탈출 불가(진짜 뿌리)** | `temperature=0.0`(우리 스택이 결정론 위해 **의도적으로** 설정)+`repetition_penalty` 미설정(vLLM 기본 1.0=무패널티) ⇒ 샘플링 탈출구가 **원리적으로 없음**. 그리디 디코딩의 고전적 퇴화 | `t2_run_gated.py` llm_args={temperature 0.0(+seed/max_tokens)}·다른 penalty 인자 없음 |
| ④**파서 all-or-nothing** | 미종결 8번째 블록 하나 때문에 **유효한 7개 호출이 통째로 폐기**되고 33k 전체가 content로 커밋 | serve 로그 `hermes_tool_parser.py:157` 예외·궤적 `calls=[]` |
| (부가)**중첩 인자 상관** | 006·001의 반복 대상 블록이 둘 다 give의 중첩 JSON-문자열 인자형(040은 산문형) — **원인이 아니라 반복 대상의 특징**([D]) | 3건 표본 |
| ⑤**창 소멸** | 8,192토큰 덩어리가 대화에 커밋 → 3~5턴이면 48,640 초과 → `context_window_exceeded` | 006=nmsg 11서 사망·001 동형·A arm ctxover 12건 |

### §1c 왜 **지금** 보이나 (day3와의 관계)
day3에도 같은 폭주가 있었다 — 다만 max_tokens 무제한이라 **20분+ 단독 디코드 → 클라 타임아웃 → 시도 폐기(nmsg 0)** 로
끝나 궤적이 남지 않았다(C205 [S]). C205 캡은 증상을 옮긴 것이 아니라 **숨어 있던 뿌리를 관측 가능하게 만든 것**이다.
⇒ day3 infra 6건·day2 infra 7건의 상당수가 실은 이 폭주였을 개연성([M]·궤적 부재로 확정 불가).

### §1d A 처방 후보
| # | 처방 | 성격 | 리스크 |
|---|---|---|---|
| **A1** | 에이전트 요청에 `repetition_penalty≈1.05` 또는 `frequency_penalty≈0.1` | 원인 ③ 직격·러너 1줄 | **샘플링 변경 = 스택 비교성 영향**·결정론 실험(seed arm)과 상호작용 → 별도 arm 검증 필요 |
| **A2** | **봉투-퇴화 게이트**: content에 `<tool_call>` 텍스트가 있는데 파싱된 tool_calls가 **비면** → `tool_choice="required"`로 regen(구조화 디코딩 경로는 텍스트 파서를 안 타므로 이 실패가 성립 불가·기존 레버 A 실측 24/24) | 원인 ④⑤ 차단·엔진 게이트(기존 FORCE/TOOLLIST 동형) | regen 1회 비용·[[03b]] 안전(봉투=프레임워크 프로토콜이지 도메인 formalize 아님) |
| **A3** | give-계열의 **중첩 인자 예방**: A2 `tool_arg_allowlist`가 이미 여분 `arguments`를 strip 중 — 이를 **생성 이전 지시**(params 설명에 "중첩 JSON 금지·인자 없이 도구명만")로 승격 | 원인 ① 예방·A2 문구 | soft(006에서 soft 무시 전례) |
| **A4** | 절단 응답 자체를 대화에 **커밋하지 않기**(finish_reason=length면 regen) | 원인 ⑤ 완화 | 정당 장문 응답과 구분 필요(8192 캡이 있으므로 length=사실상 폭주 신호) |
### §1e ★정정이 처방에 주는 함의
- **A3(중첩 인자 예방)의 우선순위 하향** — 트리거가 형식 위반이 아니므로 예방 효과는 [D].
- **A2가 더 강하게 정당화됨** — `tool_choice="required"`(구조화 디코딩)는 문법을 **강제 종료**까지 규정하므로
  ①정지 실패·④파서 폐기 둘 다 성립 불가.
- **신규 A5(구제·salvage) 후보**: content에 닫힌 유효 블록이 N개 있는데 파싱 결과가 0이면, **닫힌 블록만 재파싱해 실행**
  (동일 호출은 D7 dedup이 1개로 접음). 프레임워크 봉투 파싱이라 [[03b]]의 도메인-formalize와 다르지만, 파서가 버린 호출을
  엔진이 되살리는 것이라 **A2보다 공격적** — A2 우선, A5는 A2 실패 시 대안.
**권고 조합 = A2(주) + A4(보조)**. A1은 효과가 가장 확실하나 **비교성 비용**이 있어 별도 arm 승인 후. A3는 무료라 동반.

---
## §2 축 B — pass→fail 역전의 per-step 원인

### §2a 035: day3 PASS vs day4b FAIL — **에이전트 행동은 동형·차이는 종결 마진** [S]
| | day3(PASS·nmsg 46) | day4b(FAIL·nmsg 20) |
|---|---|---|
| 에스컬 도구 | m8 unlock+call ✓ | m4 unlock+call ✓ |
| 그 후 | **OTP 날조 6턴 연속**(m26~m36 "will send/resend the one-time passcode") | KB 중복검색 4회(m9~m15·DUPLICATE stub) |
| 정식 notice | m38 | m18 |
| 유저 terminal | m39 `###TRANSFER###` | m19 `###TRANSFER###` |
| **직후** | **sim이 계속됨**(m40~m41) → **m42서 transfer 실호출 → PASS** | **sim 종료** → 호출 0 → FAIL |
⇒ **동일 결함**(동의 후 호출 대신 notice)인데 day3는 terminal 뒤 한 턴을 더 받아 만회했다. **035의 day3 PASS는 마진의 운**이고,
구조적으로 고쳐진 적이 없다(C200 진단 유효). day4b에선 **D2 chain이 3회 정확히 발화**했으나 regen 전부 빈손 →
`_t2_followup` cap(3) 소진 → 마지막 국면에 레버 부재.
### §2b 001: day3 PASS → day4b FAIL = **축 A**(게이트 무관)
day4b 001은 m4에서 33k자 폭주(nested-args 트리거) → ctxover, nmsg 11. 게이트·D6 무관.
### §2c 019: **pass한 적 없음**(day2/3/4b 전부 0.0) — 다만 **단계는 전진**
day3=rate 도구 미호출(KB 루프) → day4b=검출 22/23·dispute give 5회까지 도달·**유저 실행 층에서 이탈**.
D9(정규화 매칭)는 이번엔 불필요했다(KB 4회 만에 도달·발화 0=오탐 0).
### §2d B 처방 후보
| # | 처방 | 근거 |
|---|---|---|
| **B1** | **chain 예비-예산**: `_t2_followup` cap 소진 후에도 **transfer-류 requires 미이행** 시 sim당 1회 보장(D3 예비-창과 동형·A2 `reserve` 플래그로 체인별 선언) | 035 day4b: 3회 발화 후 정확히 그 순간에 레버 없음 |
| **B2** | **빈손 regen 대책 재검토**((b)채널 강제) — 전환률 day3 39%·day4b A 29%/B 39%로 정체. 세분 kind처럼 **도구가 특정되는 경우에 한해** `tool_choice="required"` | D5(a) 문구-only가 3개 런에서 전환률을 못 올림 |
| **B3** | 종결-마진 의존 자체를 계측: terminal 토큰 직후 sim 지속 여부를 결과에 기록 | 035류 flip의 pass^1 노이즈를 판정에서 분리 |

---
## §3 축 C — 반복의 per-step 분해 (**두 개의 다른 반복**)
### §3a C1 — 동일-질의 복사 루프 (012·[S])
`"how to set up a travel notification for a credit card"` **4회 완전 동일**(+근사 변형 2회) → DUPLICATE stub 3회 →
그 후 **도구 사용을 포기**하고 환각 안내로 전환(m14~m24). C194 에스컬레이션(횟수 표기+행동전환 지시)이 발화했는데도
루프가 안 끊긴 이유 = §1b③과 **같은 뿌리**(temp 0 + 무패널티에서 직전 자기 출력이 지배). 즉 C1은 축 A의 **경량판**이다.
### §3b C2 — 정형구 어트랙터 + 능력 날조 지속 (004·035day3·[S])
004의 어시스턴트 턴은 거의 전부 `"I apologize for the oversight. Let's proceed with ..."`로 시작(질의는 3개 모두 상이=C1 아님).
여기에 **존재하지 않는 OTP/SMS 기능**을 반복 약속(m14·m18; 035 day3는 m26~m36 6연속). 도구 목록에 없는 기능을
말로만 제공하는 상태가 여러 턴 지속된다.
**004의 진짜 감점 지점**: transfer를 **호출은 했다**(m24) — action_match=False = `reason` enum 불일치(day2와 동일).
### §3c C 처방 후보
| # | 처방 | 성격 |
|---|---|---|
| **C1-a** | A1/A2와 동일(뿌리 공유) — 별도 레버 불요 | 축 A에 흡수 |
| **C2-a** | **미보유 기능 약속 차단**: claim_prov의 **미래형(pending)** 판정에 "도구 목록에 없는 행동"을 kind로 추가 — 엔진은 약속된 도구명이 자기 도구목록에 있는지 **집합 대조**만(TOOLLIST 술어 재사용·리터럴 0) | 신규(소)·[[03b]] 안전 |
| **C2-b** | reason enum(004)은 **미처방 유지** — 상황→enum 매핑은 순수 semantic·게이트가 답을 주면 [[03b]] 위반([[13]] scale/learn 잔여) | 유지 |

---
## §4 [[05]] 3질문 ([[17]] 상설)
1. **도메인-특화 순증?** A2·A4·B1·B2·C2-a 전부 **프레임워크 층**(봉투 형식·게이트 예산·도구목록 집합)이라 도메인 리터럴 0.
   A3만 A2 문구 1항목. A1은 샘플링 파라미터로 도메인과 무관.
2. **유동 판단 동결?** 아니다 — 어떤 도구를 부를지·무엇을 말할지는 전부 모델. 게이트는 봉투가 깨졌을 때 채널을 바꿔
   재생성시키거나(A2), 예산을 한 번 더 주거나(B1), 없는 기능 약속을 지적할 뿐(C2-a).
3. **엔진이 도메인 행동 수행?** 아니다 — regen·피드백·집합 대조뿐. 도구 대행 0.

## §5 검증 계획 (구현 시)
1. **A2 재현 픽스처**: 006 m4 원문을 그대로 넣어 (a)봉투-퇴화 검출 (b)정상 tool_call 응답 무발화(오탐 0) (c)정상 산문 응답 무발화.
2. **A4**: finish_reason=length 모의 → regen 트리거·정상 종료는 무발화.
3. **B1**: chain cap 소진 상태 + transfer requires 미이행 → 1회 발화·소진 후 무발화(순수함수 `_cpv_window`와 동형 테스트).
4. **C2-a**: 약속 도구명이 목록에 있으면 무발화 / 없으면 unbacked(집합 대조만).
5. 회귀: test_c201_stage2·test_c204_nextrun·test_followup_chain·test_claim_pending·test_toollist·test_compute 등 12종.
6. **A1 채택 시**: 별도 arm(동일 32태스크·A1만 차이)으로 Δ측정 — 비교성 훼손을 데이터로 정량화한 뒤 본 스택 반영 여부 결정.

## §6 GO 기준 / 우선순위
- **즉시 GO(무료·비교성 무영향)**: A2·A4·A3·B1·C2-a.
- **승인 필요**: A1(샘플링 변경·결정론 arm 영향)·B2((b)채널 강제·유저-측 도구 오호출 위험은 "도구 특정 가능한 kind"로 한정해 완화).
- 기대 커버리지(day4b 기준): 축 A 6건+infra 2 개연 · 축 B 2건(035·019 실행층) · 축 C 1건(012) ≈ **9~11/15**.

## §7 미해결
- day2/day3 infra 13건 중 폭주 비중은 궤적 부재로 [M] — day4b 이후 런에서만 계측 가능.
- 040 폭주는 nested-args 없이 산문에서 시작 = 트리거가 중첩 인자 **전용은 아님**(복사 루프 자체가 일반).
- B-arm 잔여 11건 완주 후 축 분포 재확정 필요(특히 021·023·026 heavy).

---
## §8 구현 명세 (rev2·2026-07-27) — **리뷰 대상**. 승인 후 구현.

### §8-0 도달성 선검증 ([[08]]·D2/D4 실패 재발 방지·전부 코드 확인 [S])
| 필요 조건 | 확인 결과 |
|---|---|
| 응답의 `finish_reason` 접근 | ✅ `llm_utils.generate`가 `AssistantMessage.raw_data = response.to_dict()`로 보존 → `raw_data["choices"][0]["finish_reason"]` |
| 커밋 前 개입 지점 | ✅ `t2_gate_patch.py` 게이트부의 `am`(=`_gen(...)` 결과)은 **커밋 전 작업버퍼** — NOTICEREP/CLAIMPROV가 이미 여기서 regen |
| 채널 강제 regen | ✅ `_ap_regen(fbtxt, tag, tool_choice=None)` 3번째 인자 지원(레버 A 경로) |
| chain 예산 위치 | ✅ `_fu_cap = int(os.environ.get("T2_FOLLOWUP_CAP","1"))`(go_stack=3)·카운터 `self._t2_followup` |
| 도구목록 집합 | ✅ `{getattr(t,"name",None) for t in (self.tools or [])}`(TOOLLIST가 사용 중) |

### §8-1 A2 — 봉투-퇴화 게이트 (주 처방)
- **위치**: `t2_gate_patch.py` · `am` 확정 직후, NOTICEREP 블록 **앞**(가장 이른 방어선).
- **술어**(전부 구조적·도메인 리터럴 0):
  `env T2_ENVELOPE_GUARD=1` ∧ `am.tool_calls` 비어 있음 ∧ `content`에 봉투 여는 태그(`T2_ENVELOPE_TAG`, 기본 `<tool_call>`)가 **1회 이상** ∧ `self._t2_envguard < cap(T2_ENVELOPE_CAP, 기본 2)`
- **동작**: `_ap_regen(feedback, "envguard", tool_choice="required")` — 구조화 디코딩은 문법·**종료**까지 규정하므로 정지 실패가 성립 불가.
- **feedback 문구**(프레임워크 층·도메인 무관): "직전 응답의 도구 호출 봉투가 파싱되지 않아 **아무 도구도 실행되지 않았다**. 같은 호출을 반복하지 말고, 필요한 도구 **하나**를 지금 호출하라."
- **오탐 방어**: `tool_calls`가 하나라도 파싱된 정상 응답 → 술어 거짓. 봉투 태그 없는 순수 산문 → 거짓. cap 소진 후 통과(liveness).
- **replay 안전**: 작업버퍼 교체만(비커밋).

### §8-2 A4 — 절단 응답 미커밋 (보조)
- **위치**: A2 바로 뒤(순서: A2 → A4).
- **술어**: `env T2_TRUNC_GUARD=1` ∧ `finish_reason == "length"` ∧ `self._t2_truncguard < cap(T2_TRUNC_CAP, 기본 1)`
- **동작**: `_ap_regen(feedback, "truncguard")` — **채널 강제 없음**(정당한 장문 요약이 잘렸을 수도 있으므로 산문 재작성도 허용).
- **문구**: "직전 응답이 길이 상한에서 잘렸다. 반복 없이 **짧게** 다시 답하되, 행동이 필요하면 도구를 호출하라."
- **핵심 안전장치**: 재생성도 `length`면 **그대로 통과**(cap 1) — 무한 regen 금지.
- **[S] 근거**: 정당 응답이 8,192에 닿은 사례 0(최장 정당=77행 에코 ~8k·절단 안 됨) ⇒ `length`=사실상 폭주 신호.

### §8-3 B1 — chain 예비-예산
- **위치**: `follow_up_chains` 디스패치부(`_fu_cap` 검사 지점).
- **술어**: cap 소진 ∧ 발화 대상 chain이 **A2에 `reserve: true` 선언** ∧ `self._t2_fu_reserve` 미사용 → 1회 허용 후 소진 마킹.
- **A2**: 에스컬→transfer chain에 `"reserve": true`(035 표적). 다른 chain 미선언=거동 보존.
- **순수함수 분리**: `_fu_window(cap_used, cap, reserve_declared, reserve_used) -> bool`(단위테스트 공유·`_cpv_window`와 동형).

### §8-4 C2-a — 미보유 기능 약속 차단
- **위치**: `claim_prov` pending 판정부(`_claim_unbacked` 호출 인접).
- **설계**: A2 `claim_prov.question`의 `pending` 지시에 "**네가 하겠다고 말한 행동에 필요한 도구 이름**"을 함께 선언하게 하고(`{kind, what, tool}`), 엔진은 그 `tool`이 **자기 도구목록에 없으면** unbacked로 계상(집합 대조만·TOOLLIST 술어 재사용).
- **문구**: "그 기능을 제공할 도구가 네게 없다 — 할 수 있다고 말하지 말고, 가능한 대안이나 이관을 제시하라."
- **오탐 방어**: `tool` 미선언 pending은 기존 경로 그대로(거동 보존). 도구명이 목록에 있으면 무발화.
- **표적**: 004·035day3의 OTP/SMS 반복 약속.

### §8-5 순서·상호작용 (게이트 스택 내 배치)
`A2(봉투) → A4(절단) → NOTICEREP → WRITEPROV → CLAIMPROV(+C2-a) → FOLLOWUP chains(+B1)`
- A2/A4가 먼저 도는 이유: **오염된 `am`을 뒤 게이트가 판정하면 전부 오판**(폭주 텍스트에는 어떤 술어든 걸릴 수 있음).
- 전역 `T2_REGEN_BUDGET=12`는 유지 — A2/A4는 폭주 턴에서만 발화하므로 예산 잠식 미미(예상 ≤2/sim).

### §8-6 오프라인 검증 (구현과 동시·`test_c207_envelope.py` 신설)
1. **A2**: 006 m4 원문 픽스처 → 발화 ✓ / 정상 tool_call 응답 → 무발화 / 봉투 없는 산문 → 무발화 / cap 소진 후 → 무발화.
2. **A4**: `raw_data.choices[0].finish_reason="length"` 모의 → 발화 ✓ / `"stop"` → 무발화 / 재생성도 length → 통과(무한루프 0).
3. **B1**: `_fu_window` 4케이스(여유·소진+예비 선언·예비 소진·예비 미선언).
4. **C2-a**: pending `tool`이 목록 내/외 각각 무발화/unbacked.
5. 회귀 12종(test_c201_stage2·test_c204_nextrun·test_followup_chain·test_claim_pending·test_toollist·test_compute·test_notice_gate·test_sub_inject·test_c197_inputholes·test_operand_grounding·test_banking_gate·test_sg_isolate).

### §8-7 [[05]] 3질문 (레버별)
| 레버 | (1)도메인 순증 | (2)판단 동결 | (3)행동 대행 |
|---|---|---|---|
| A2 | 0 — 봉투 태그는 **서빙 포맷**(hermes) 상수·env로 노출 | 0 — 어떤 도구든 모델이 고름 | 0 — regen만 |
| A4 | 0 — `finish_reason`은 프로토콜 필드 | 0 — 재작성 내용 자유 | 0 |
| B1 | A2 플래그 1개(`reserve`) | 0 — 예산만 1회 추가 | 0 |
| C2-a | 0 — 도구목록 집합 대조 | 0 — 대안 선택은 모델 | 0 |

### §8-8 미채택(승인 대기) 재확인
- **A1**(repetition_penalty): 원인 ③ 직격이나 **샘플링 변경 = 비교성·결정론 arm 영향** → 별도 arm 승인 필요.
- **A5**(salvage): 파서가 버린 호출을 엔진이 되살림 — A2가 실패할 때만.
- **B2**(채널 강제 일반화): 유저-측 도구 오호출 위험 → "도구 특정 가능 kind" 한정안으로만.
