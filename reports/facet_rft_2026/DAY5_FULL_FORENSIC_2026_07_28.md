# day5 front32 전수 포렌식 (2026-07-28 재분석)

> 대상: `bank_day5front{A,B}_20260728` (conc 1 · timeout 2400 · `T2_AGENT_MAX_TOKENS=8192` · 전 레버 스택)
> 방법: 32 sim 전 궤적 per-step 정독 + `prompt_tokens`/`completion_tokens` 기반 문맥 분해 +
> **tau2 평가 환경 오프라인 재현으로 DB 해시 diff 복원**(유료 호출 0 · [[09]] 준수).
> 등급: [S]=궤적/코드/재현으로 확정 · [M]=강한 정황 · [?]=미확정.
> 전판 `HANDOFF_2026_07_28.md` §2a를 **부분 정정·대폭 확장**한다.

---

## 0. 스코어보드 (변동 없음)

| arm | PASS | user_stop 실패 | ctxover | infra | **실패 계** | 계 |
|---|---|---|---|---|---|---|
| A | 4 (006·017·033·034) | **9** (004·005·008·012·015·019·027·035·040) | 2 (022·028) | 1 (024) | **12** | 16 |
| B | 7 (001·002·003·016·021·025·032) | 3 (007·014·026) | 5 (018·020·023·029·041) | 1 (010) | **9** | 16 |
| 합 | **11** | **12** | **7** | **2** | **21** | 32 |

> 주의: "9"가 두 곳에 나온다 — **A의 user_stop 실패 9건**과 **B의 실패 총계 9건**은 다른 값이다.
> 본 문서 §5는 user_stop 실패 **12건**(A 9 + B 3)만 다루고, 나머지 9건(ctxover 7 + infra 2)은 §2·§3에서 다룬다.

### 0.1 분석 커버리지 (무엇을 어디까지 봤는가)

| 구분 | 건수 | per-step 궤적 정독 | 근거 |
|---|---|---|---|
| ctxover | 7 | **7/7** | 전 메시지 + pt/comp 실측 차분 분해 (§2) |
| user_stop 실패 | 12 | **12/12** | 전 메시지 정독 (§5) |
| infra | 2 | **0/2 — 원리적으로 불가** | 궤적이 존재하지 않음(`messages` 0). 로그의 retry 블록 + 코드 3지점으로 기전 확정 (§3) |
| **실패 계** | **21** | **19/21** | |

**infra 2건(024·010)에 per-step이 없는 이유**: 평가 단계에서 ValueError가 나면 tau2 러너가 sim 전체를 실패로
버리고 재시도하며, 4회 실패 뒤 **메시지를 하나도 영속하지 않는다**. 사망 기전은 확정했지만
"에이전트가 그 전에 무엇을 잘못했는지"는 이 런의 데이터로는 복원 불가다.
→ **부수 처방**: 러너가 실패 sim의 궤적도 남기도록 할 것(이 구멍이 day2~day4b infra 다수의 원인 규명을 막고 있다).

PASS 11건은 대조군으로만 썼다(coverage 교차표 §6, 페이로드 전사 검증 §6). 실패 원인 분석 대상은 아니다.

---

## 1. ★최상위 발견 — ctxover 7건은 "문맥 초과"가 아니라 **우리가 만든 천장**이다 [S]

vLLM 에러 원문(로그 A:3123 등):

```
ContextWindowExceededError: 'max_tokens' or 'max_completion_tokens' is too large: 8192.
This model's maximum context length is 48640 tokens and your ...
```

→ 실효 프롬프트 천장 = **48,640 − 8,192 = 40,448**. 모델 창이 아니라 **C205가 예약한 8,192가 창을 깎았다.**

측정된 최대 prompt_tokens (전 32 sim):

| task | 결과 | max_pt | 40,448까지 여유 |
|---|---|---|---|
| 020 | CWE | 40,189 | **259** |
| 040 | user_stop | 40,114 | 334 |
| 041 | CWE | 39,489 | 959 |
| 022 | CWE | 37,365 | 3,083 |
| 028 | CWE | 37,108 | 3,340 |
| 018 | CWE | 36,987 | 3,461 |
| 023 | CWE | 36,778 | 3,670 |
| 029 | CWE | 36,575 | 3,873 |

**7건 전부 36.5k~40.2k에서 죽었다. 48.6k 근처까지 간 sim은 하나도 없다.**

예약의 비용/편익 (assistant 생성 500회 실측):

- completion_tokens p50=74 · p90=370 · **p95=1,115** · p99=4,727 · p100=8,192
- 8,000 초과 = **1회 / 500회 (0.2%)** — 007의 폭주 1건뿐
- 즉 **매 턴 6,144토큰(창의 15%)을 상납해 0.2%를 막고 있다.**

반사실 계산(마지막 성공 pt + 그 턴 completion + 이후 tool/user 텍스트):

| task | 실패 직전 프롬프트 추정 | max_tokens=2048(천장 46,592) |
|---|---|---|
| 018 / 028 / 029 | 41.4k / 41.9k / 41.4k | 전부 통과 |
| 020 / 022 / 023 / 041 | 40.4k / 41.4k / 40.9k / 40.9k | 전부 통과 |

**7건 전부 그 스텝에서는 살아남는다.** (통과가 PASS를 뜻하진 않음 — 018/028/029는 이미 정답 compute를 끝낸 직후였다: §2 참조.)

→ **처방 1(최저비용·즉시)**: 고정 `T2_AGENT_MAX_TOKENS` 폐기, 턴마다
`max_tokens = min(8192, 48640 − prompt_tokens − margin)`. 폭주 방어(A4 TRUNCGUARD)는 그대로 유지되고
창 6k가 즉시 돌아온다. 레버를 끄는 게 아니라 조정 — [[19]] 합성-우선과 정합.

---

## 2. ctxover 7건 전수 분해 (토큰 귀속은 pt/comp 실측 차분 — 추정 아님)

방법: assistant i와 i+1 사이에 프롬프트에 더해지는 것은 (i의 completion) + (그 사이 tool/user 메시지)뿐이므로
`between = pt[i+1] − pt[i] − comp[i]` 는 **측정값**이다. 이를 char 비율로 귀속했다. 음수 잔차 = 뷰 재작성(VIEW_COMPACT).

### 2.1 합산 (7건 전체)

| 구성 | 토큰 | 비중 |
|---|---|---|
| **tool 출력: KB_search 문서 덤프** | 139,089 | **44.8%** |
| system + 도구 스키마(A2 주입 8종 포함) | 52,863 | 17.0% |
| tool 출력: DB 레코드 덤프 | 45,137 | 14.6% |
| **assistant: compute 도구 인자 재직렬화** | 24,728 | **8.0%** |
| assistant 산문 | 16,928 | 5.5% |
| tool 출력 기타/짧은 것 | 13,354 | 4.3% |
| assistant 기타 도구호출 | 10,750 | 3.5% |
| user 턴 | 7,359 | 2.4% |

### 2.2 개별

**018 · 028 · 029 (동일 시나리오 3변형 · temp 0이라 접두사 완전 동형)** [S]
- 첫 5스텝이 전부 KB_search: 15,149 / 19,357 / 18,159 / 23,203 / 23,181자 = **99,812자(≈22.3k tok, 창의 60%)**
  — 계좌 조회 전에, 전부 "도구를 찾으려고" 한 근사-중복 질의("get credit card transactions" ↔ "tool to get credit card transactions" ↔ "…by user").
- READ_DEDUP은 **완전 일치 1건만** 잡았다(6번째). 근사-중복 4건은 통과.
- 죽은 지점: `get_reward_discrepancies` 가 **47/47행 전부 판정·6건 검출**(=정답 산출)한 **직후**.
  → **이 3건은 계산을 이미 끝내고 보고 직전에 천장에 부딪혔다.** §1 처방만으로 회수 가능성이 가장 높은 후보.

**023** [S]
- KB 6회 40.2% + 60행 페이로드를 **두 번** 재직렬화(14,270자 → 10,754자).
- 2회차 페이로드는 **60행 중 45행만** 담았다(15행 유실). 출력은 똑같은 258자.

**022** [S]
- 77행 fetch(23,291자) + 페이로드 **19,768자 = completion 7,019토큰**(런 전체 최대 정당 생성).
- 페이로드가 77행 중 68행만 담고(9행 유실), **2건은 레코드가 뒤섞였다**(ebf80b에 Marriott/Travel 값이, 670323에 다른 금액이 들어감 — 11개 필드 오류).
- VIEW_COMPACT가 −10,438토큰을 회수했지만 이미 늦음.

**020** [S] — **가장 뼈아픈 케이스**
- 실제로는 **과제를 끝냈다**(마지막에 dispute 제출 완료). 그런데 pt=40,189, 천장까지 **259토큰** 남기고 사망.
- 주범: `get_reward_discrepancies`를 **동일 인자로 5회** 호출. 매번 26행 페이로드 6,032~6,654자를 다시 뱉음
  = **11,844토큰 = 창의 28.9%**.
- `[DUPLICATE-COMPUTE]` 가드는 **출력(220자)만 막았고 입력 비용은 못 막았으며, 반복 자체도 못 멈췄다**
  (첫 DUPLICATE-COMPUTE 이후에도 3회 더 같은 호출).

**041** [S]
- KB_search 12회 = 174,138자(도구 출력의 92%). VIEW_COMPACT가 −33,907토큰을 회수했는데도 39,489에서 사망.
- 4장 카드 × 4건 dispute = 16건 과제. 천장 문제 이전에 **작업량 자체가 40k 창에 안 들어간다**.

### 2.3 VIEW_COMPACT가 사실상 꺼져 있다 [S]

`t2_gate_patch.py:2959` `_compact_view(..., min_total=120000)` — **뷰 총 120,000자를 넘어야** 발동.
120,000자 ≈ 34k~40k 토큰 = **이미 죽는 지점**. 실제로 32 sim 전체에서 발동 **6회**(A 1 · B 5)뿐이고,
018/028/029는 도구 출력 총합 115,552자로 **문턱 아래에서 죽었다**.

→ **처방 3**: `min_total`을 창 기준 비율(예: 천장의 55~60% ≈ 22k토큰 ≈ 80,000자)로 낮추고,
KB_search 결과에 per-call 상한(top-k 축소 또는 문서 본문 절단)을 건다. 기대효과가 가장 큰 축(44.8%).

---

## 3. infra 2건 (024 · 010) — 우리 스캐폴드가 tau2 replay 불변식을 깬다 [S]

기전(코드로 확정):

1. `t2_prekb_patch.py:353~358` — env가 돌려준 ToolMessage에 `r.content = (c + " [GUIDANCE] …")` 로 **덧붙임**.
2. tau2 `environment.py:374~390` — 평가 시 **mutating 도구만** 깨끗한 env에서 재실행하고 content를 **정확히 비교**:
   ```
   if not self._is_mutating_tool(tool_call.name): continue
   ...
   if content != expected_content: raise ValueError(...)
   ```
3. `give_discoverable_user_tool` 은 `@is_tool(ToolType.GENERIC, mutates_state=True)` (tools.py:520) = **재실행 대상**.
   재실행 결과에는 `[GUIDANCE]`가 없다 → 불일치 → ValueError → 재시도 3회 → `infrastructure_error`.

**1회 발생만으로 그 sim은 채점 전에 사망하고 궤적도 안 남는다**(024·010 모두 messages 0).

정정 2건 (전판 자기-오진 확정):
- (i) "cap이 시도 간 유지돼 불일치" = **틀림**. 원인은 replay에 패치가 없어서.
- (ii) "모델이 `apply_for_credit_card`를 발명" = **틀림**. `apply_for_credit_card`(tools.py:4325)·`submit_referral`(4382)
  둘 다 `KnowledgeUserTools`의 **실재 WRITE 도구**다. 모델의 실제 잘못은 **채널 오분류**
  (유저-네이티브 도구를 `give_discoverable_user_tool` 디스패처로 라우팅). 우리 GUIDANCE 문구
  ("That tool name does not exist — you invented it")는 **사실과 다른 피드백**이고,
  "KB를 다시 검색하라"는 지시는 존재하지 않는 문서를 찾게 만든다.

노출 범위(전 32 sim 마커 감사): mutating 도구에 마커가 붙은 sim = **024 · 010 · 022 = 3건(9.4%)**.
022는 ctxover로 끝나 **평가 자체를 건너뛰어 우연히 살았다**(ctxover sim은 reward 0.0 직행·replay 미실행).
→ **infra 2건은 노출의 하한이다.**

→ **처방 2(평가 위생·최우선)**: mutating 도구의 ToolMessage에는 **어떤 문자열도 덧붙이지 않는다**
(피드백은 생성-레벨 버퍼로 이동). 동시에 문구를 사실에 맞게 교정: "그 이름은 discoverable 도구가 아니라
유저 네이티브 도구다 — 건네지 말고 고객이 직접 실행하도록 안내하라"(env 레지스트리 집합 대조로 판정).

---

## 4. ★C207/C2-a(미보유 기능 약속) 레버는 **한 번도 실행되지 않았다** [S]

전판 §1의 "게이트 계측: … unavail 223 …"은 **작동 계측이 아니라 전량 실패 계측**이다.

```
[T2_UNAVAIL] skipped (no-op): NameError("name 'orch' is not defined")   A 105 + B 118 = 223 / 223
```

`t2_gate_patch.py:5077~5090`:
```python
_known = _known_tool_names(getattr(self, "tools", None),
                           getattr(self, "environment", None)
                           or getattr(orch, "environment", None),   # ← orch 미정의
                           state.messages)
```
`self.environment` 가 항상 falsy라 `getattr(orch, ...)`가 평가되고 `orch`는 그 스코프에 없다 → 매번 NameError → except로 무음 통과.

**직접 피해자 004**: [14]에서 존재하지 않는 SMS 인증코드 발송을 약속하고 창을 소진하다 transfer 미호출로 실패.
이게 정확히 C2-a가 잡으라고 만든 케이스다.

→ **처방 4(1줄)**: `or getattr(orch, ...)` 제거. 그리고 **예외를 무음 통과시키지 말 것** —
223회 전량 실패가 로그에서 "정상 계측"처럼 보였다.

---

## 5. user_stop 실패 12건 per-step 원인

### 5.1 ACTION 기준 5건 — 종결 행동 미실행

| task | gold | 실제 | 확정 원인 |
|---|---|---|---|
| **004** | `transfer_to_human_agents(reason=account_ownership_dispute)` | **호출 0** | §5.1b 터미널-턴 복권의 직격 피해자: 유저 동의(###TRANSFER###) 순간 **행동 턴 0으로 즉시 종료**(drive gap 없음). RESOLVE 3 · FORCE_ACTION 3 · CLAIMPROV regen 4(전부 `tool_calls=[]`) 불발 + 없는 OTP 약속(§4). [S] |
| **008** | transfer(`customer_demands_after_unavailable_offer_refusal`) compare_args=['reason'] | 호출했고 **`reason`도 넣었다**: `customer_frustrated_demands_human` | ~~"reason 미기입"~~ **정정: enum 선택 오류 + 자기-정박**. PRE-ACTION-KB가 강제한 검색[41]이 **정답 티어표 문서(doc_042)를 문맥에 원문 배달** — "항상 최상위 티어에서 골라라" + gold enum(TIER 1)의 정의문이 008 상황과 자구까지 일치("없는 오퍼 주장→불가 안내→고객 고집→인간 요구"). 그런데 모델은 읽기 **전**[38]에 고른 TIER 3 코드(`customer_frustrated_demands_human`="일반적 짜증")를 읽은 **후**[42] **한 글자도 안 바꾸고 재발급**. 증거가 문맥에 실재해도 기존 선택을 갱신하지 않는 **forced-read-without-revision**([[18]] 자기-정박 동형). [S] |
| **012** | transfer(`kb_search_unsuccessful…`) | 호출 0 | 필요문서 = **doc_042(전이 사유표)뿐** = 이 과제의 gold 경로는 "여행알림 절차는 KB에 없음→NOTICE→transfer". 모델은 "없음"까진 정확히 판정했으나 **근거 없는 우회 안내**("앱에서 support에 메시지하라")를 제공 → user-sim의 "도움 받음" 분기 발동 → 만족-###STOP###으로 **transfer 요청 분기 자체가 소멸**. [S] |
| **014** | transfer(`unconfirmed_external_communication`) | 호출 0 | 015와 대구: 014는 고객 주장 조건이 **어느 카드와도 불일치**(필요문서 5종 대조)라 gold=transfer, 015는 5번째(Platinum)만 일치라 gold=Platinum 링크. 모델은 두 과제 모두 **첫 카드 조건을 KB와 대조 없이 그대로 승인** — 같은 한 스텝(주장-대조 생략)이 gold가 반대 방향인 두 과제를 동시에 죽임. [S] |
| **035** | unlock+call `emergency_credit_bureau_incident_transfer_1114` + transfer | **도구 호출 0회**(전 sim) | 필요문서 = `credit_cards_(general)_012`(긴급 프로토콜). **KB_search 0회** → 표준검증 면제 긴급경로의 존재를 알 기회 자체가 없었음 → 검증 거부 2턴 → 유저 터미널 → gap 0 → 즉시 종료. 발견 시도 0 + §5.1b 복합. [S] |

### 5.1b ★터미널-턴 복권 — "동의 후 행동 턴"은 coverage 모델이 추첨한다 [S]

유저의 터미널 토큰(###TRANSFER###/###STOP###)이 오면 sim은 즉시 종료된다. 예외는 하나 —
E-PLAN walk(`t2_eplan_patch.py:976`)가 gap을 찾을 때만 user_stop을 보류하고 한 턴을 더 준다:
`n = walk_required_n(...); if n <= 1 or n <= m: return`(종료 확정).

- **004·035**: 수량-ask 없음/원장 빈 상태 → gap 산출 불가 → **동의 직후 0턴 종료**. NOTICE→동의→호출 프로토콜의
  마지막 단계가 **구조적으로 실행 불가능**했다.
- **008**: `walk gap qty=10`(자구 "$10,000"/"10 business days"류 오파싱 개연·C133 spurious-gap 동형 [M])이
  **우연히** 보류를 발동 → 그 여분 턴에 transfer 실호출 성공.
- 즉 C200의 notice-레이스는 "모델이 안 불렀다"가 아니라 **"gap 없는 과제에서는 부를 턴이 주어지지 않는다"**로
  기전이 내려간다. gold 종결행동(transfer 등)의 유무는 drive의 gap 계산에 전혀 반영되지 않는다.

→ **처방 5(정정)**: NOTICE 발화를 원장에 "미이행 종결행동"으로 등록해 drive가 gap으로 인식하게 하거나,
유저 터미널 수신 시 **미호출 종결행동이 있으면 1턴 유예**를 walk과 독립으로 보장. (기존 "tool_choice=required
강제" 안은 턴 자체가 없으면 무의미 — 이 기전이 먼저다.)

### 5.2 DB 기준 7건 — **tau2 평가 환경 재현으로 정확한 diff 복원** [S]

| task | DB diff (gold vs 실제) | 확정 원인 |
|---|---|---|
| **005** | gold의 `verification_history` 행이 전 필드 `9K2X7M4P1N8Q3R5T6A` 센티널. 실제 행은 정상 기록. | **gold 자체가 깨져 있다.** 어떤 행동으로도 그 행을 만들 수 없고, 안 만들어도 불일치. → **원리상 통과 불가 과제**. 분모에서 빼거나 gold를 고쳐야 한다. |
| **007** | `credit_card_applications` EcoCard 신청행 없음 + `initial_transfer_to_human_agent_0218 CALLED` 여분 | 2중 인과 [S]: ①**검색 실패** — 필요문서 4종(ecocard_001/009·silver_001·bronze_001=카드별 프로모 문서)이 **어느 검색 결과에도 0회 등장**(다중 카드명 복합 질의를 BM25가 savings 문서로 매칭) ②**우리 A2 fit 도구가 앵커** — 첫 행동[2]이 `check_card_application_fit`인데 그 fact 스키마에 **sign-up bonus 필드가 없음** → 권위적으로 보이는 불완전 답이 먼저 깔림 → "프로모 정보는 KB에 없다" 단정 → 신청 대신 사임. (이 sim이 런 유일의 폭주 발생지: [4] completion 8,192 / content 315,822자 — TRUNCGUARD가 뷰에서 봉쇄해 후속 턴은 정상.) |
| **015** | `referrals` 가 gold=Platinum Rewards Card, 실제=Crypto-Cash Back | 고객이 주장한 5개 조건 중 **KB와 맞는 건 Platinum뿐**인데, 대조 없이 첫 카드로 링크를 발급. 014와 동형. |
| **019** | dispute 1건 누락: `txn_f093f96e2001` | compute coverage 22/23 · 로그 `quote-ground 불성립: txn_f093f96e2001(Thrive Market) → rate 드롭(abstain)`. **abstain된 그 1행이 정확히 누락된 dispute.** 인과 사슬 완결. |
| **026** | dispute 3건 + rewards 갱신 4건 누락 | compute coverage **12/26**. §6 참조. |
| **027** | **유일한 차이 = `agent_discoverable_tools` 에 `get_user_dispute_history_7291: CALLED` 하나 더 있음** | gold 4건 dispute를 **전부 정확히 수행**했다. 그 뒤 고객이 "이제 기록을 갱신해 달라"고 하자 [60~65]에서 해소 여부를 **확인하려고** dispute 이력을 1회 조회했고, "아직 submitted"라 갱신을 보류했다. 그 **조회 1회가 만든 상태 행 하나**만으로 DB 해시 불일치. 과행동 페널티 — RESEARCH_MASTER §1 "게이트 자신도 over-action 역효과"의 실측 사례이자, **판단은 합리적이었는데 채점이 벌한** 사례. |
| **040** | dispute 8건 중 7건 누락. 제출한 1건은 `card_last_4_digits=''`, provisional 플래그 2개 오설정. `user_discoverable_tools: get_card_last_4_digits GIVEN` + CALLED 2건이 **ONLY-IN-GOLD**. + `initial_transfer_to_human_agent_1822 CALLED` 여분 | **핵심 = "고객에게 도구를 건네는" 한 스텝 누락** [S]. dispute 도구가 요구하는 `card_last_4_digits`는 유저 도구 `get_card_last_4_digits`로만 얻는데, 에이전트는 `give_discoverable_user_tool`을 **끝내 호출하지 않았다**. [84]에서 env가 `Tool 'get_card_last_4_digits' has not been given to you by the agent. The agent must first use give_di…` 라고 **정답을 그대로 알려줬는데도**, [86]에서 엉뚱하게 dispute 도구를 다시 unlock했다. 대신 KB를 `get card last 4 digits`로 **8회** 검색(= §2.1의 "KB를 도구 탐색기로 쓰는" 병리). 총 KB 20회 = 202,765자(도구 출력의 81.6%), max_pt 40,114로 ctxover 천장 334토큰 앞까지 감(그 직후 뷰 재작성 −12,876으로 겨우 생존). 결국 [102]에서 `card_last_4_digits=""`로 1건만 제출하고 나머지 7건을 포기. |

부수 [M]: 027의 gold action 4건은 인자가 완전히 동일한데 `action_match=False`다.
gold `"{\"user_id\": \"755bcb4d5d\", …}"` vs 실제 `"{\"user_id\":\"755bcb4d5d\",…}"` — **중첩 JSON 문자열의 공백 차이**.
점수(basis=DB)에는 영향이 없지만 **action 단위 계측은 이 때문에 신뢰할 수 없다.**

---

## 6. ★compute coverage 붕괴 — A2 rate 표 결손 [S] + 비결정성 [M]

`get_reward_discrepancies` 는 결과에 커버리지를 병기한다(C195). 전 sim 교차표:

| task | coverage | 검출 | 결과 |
|---|---|---|---|
| 017 | 5/5 | 2 | **PASS** |
| 021 | 17/17 | 2 | **PASS** |
| 018·028·029 | 47/47 | 6 | 정답 산출 직후 ctxover |
| 027 | 25/26 | 4 (=gold 4) | dispute는 정확·과행동으로 실패 |
| 019 | 22/23 | 3 | abstain된 1행 = 누락된 dispute 1건 |
| **020** | **12/26** | 1 | 실패 |
| **026** | **12/26** | 1 | 실패 |
| 022 | 68/69 | 21 | ctxover |

020·026·027은 **같은 고객(Amara)·같은 26행**(Silver 12 + Business Silver 14)이다. 페이로드 값 검증 결과
**필드 전사 오류 0**(전 16회 호출 중 오류는 022뿐 — "값 오전사" 가설은 이 런에서는 기각).

**⚠본 문서 초판의 "A2 rate 케이스 표에 Business Silver 없음 → abstain" 결론은 오류 — 철회.**
그 case 표는 **비활성 경로**다: 라이브는 `ratefix` variant(`T2_A2_VARIANT` 로그 전 sim 발화)이고,
rate는 격리 서브가 KB에서 formalize한 per-row operand(`base_rate` 등)로 온다. operand trace
(`day5{A,B}_operands.jsonl`) 실측: **양 arm 모두** BS에 base_rate=10.0을 정상 산출 — 표 결손은 무관하다.

**진짜 사슬 [S] — 한 필드(`account_open`)의 누락/날조가 갈랐다:**

1. 격리 서브는 세 sim 모두에서 BS를 정확히 읽었다: `base_rate 10 · promo_mult 2 · window 6개월 ·
   promo기간 2024-11-14~2025-11-14`. Silver는 promo 없음(mult 1·날짜 공란).
2. ratefix op의 promo 판정은 `date_between(r.account_open, …)` + `date_in_window(anchor=r.account_open, …)` —
   **`account_open`은 모델이 페이로드에 넣어야 하는 필드**(variant params에 명시)인데, 세 sim 모두
   **`get_credit_card_accounts_by_user`를 한 번도 호출하지 않아** 개설일을 가진 적이 없다.
3. **020·026**: `account_open`을 정직하게 **생략** → promo 선언된 BS 14행 전부가 3-값 논리로 판정불가
   (Silver 12행은 promo 공란이라 개설일 불필요→판정됨) = **coverage 12/26의 정확한 기전**. 검출 1건뿐 → gold 4건 미달.
4. **027**: `account_open`을 **날조** — 전 26행에 단일값 `02/01/2025`(=첫 거래일 앵커·어느 도구 출력에도 없는 값).
   실제 BS 개설일 = **02/13/2025**(env DB 확인). 12일 차이가 promo 판정 경계를 안 건드려
   **날조 입력에서 gold 정확 일치 4/4가 우연히 재현**됐다(25/26·나머지 1=quote-ground drop).
5. 즉 이 삼형제에서 **정직한 생략(020/026)이 날조(027)보다 나쁜 점수**를 받았고, 027의 "성공"도
   운이었다. 엔진의 abstain 메시지("14 could not be verified")는 **어느 필드가 없어서인지 말하지 않아**
   모델이 자기-수복할 수 없었다(엔진은 null이 난 operand를 정확히 알고 있는데도).

019의 1행 누락도 같은 층: quote-ground가 abstain시킨 바로 그 행(txn_f093f96e2001)이 누락된 dispute다.

→ **처방 6(정정)**: (a) ~~rate 표 보강~~ 폐기 — 무관. (b) abstain 시 **결핍 operand 필드명 + 그 필드를
공급하는 getter 계열을 지목**하는 actionable 에러로 승격("account_open missing for 14 rows — fetch the
account record first"·도메인-일반 문구·A2 param명 그대로). (c) `account_open`류 **비-레코드 유래 값의
producer-binding**(값이 선언된 producer 도구 출력에 실재해야 통과 — 027 날조는 값-실재 검사로는 못 잡음:
02/01/2025는 거래일로 문맥에 실재). (d) 부분 커버리지의 error 승격(초판 (b) 유지).

---

## 7. 레버 부작용 계측 (제1원리: 하나를 사면 하나를 판다)

| 레버 | 산 것 | 판 것 (실측) |
|---|---|---|
| C205 `max_tokens=8192` | timeout 41→0 · 정당 응답 무절단 | **창 6,144토큰 상납 → ctxover 7건의 천장** (§1) |
| PRE-ACTION-KB (transfer) | 절차 미확인 transfer 차단 — 008에서는 **정답 티어표(doc_042)를 실제로 문맥에 배달** | 5회 발동 × 28,953자 KB 강제(≈6,041 tok). 배달에 성공해도 모델이 기존 선택을 갱신 안 함(008 §5.1) = **강제-읽기의 한계 실측**. 007·026은 그 뒤 `initial_transfer_*`로 이탈해 **gold transfer를 영영 못 부름** |
| `[DUPLICATE-COMPUTE]` | 중복 출력 220자로 축소 | **게이트가 생성-후에 발화**해 입력(페이로드 재직렬화) 비용은 매회 지불 + **스텁이 이전 결과를 재제시하지 않고 "earlier output 참조"만 지시** → 유저 후속질문마다 재호출 유인 → 020에서 5회·11.8k토큰(창 29%). 경고 상향은 4회째에야 |
| `[GUIDANCE]` (prekb) | 도구명 오발명 교정 유도 | **replay 위반 → infra 2건**(+022 잠재), 게다가 문구가 사실과 다름 (§3) |
| CLAIMPROV regen | 미담보 주장 차단 | 156회 중 **101회(65%)가 `tool_calls=[]`** — 창만 쓰고 행동 전환 실패 |
| VIEW_COMPACT | 022 −10.4k · 041 −33.9k 회수 | 문턱 120,000자가 사망선 위 → 32 sim 중 **6회만 발동** (§2.3) |
| A2 scaffold_get 8종 주입 | 결정론 offload 경로 확보 | base 프롬프트 7.5k토큰(천장의 18.5%) 상시 점유 |

---

## 8. 다음 작업 우선순위 (기대효과 순 · 2차 포렌식 반영)

1. **동적 max_tokens** (§1) — 1곳 수정 · 창 6k 즉시 회수 · ctxover 7건 전부가 그 스텝에서 생존. **최우선.**
2. **평가층 replay 위생** (§3) — mutating 도구 출력 불변 + GUIDANCE 문구 사실 교정. 측정 정합성이라 미룰 수 없음.
3. **터미널-턴 보장** (§5.1b) — 미호출 종결행동이 있으면 유저 터미널 후 1턴 유예(walk gap과 독립).
   004·035 직접 회수 + notice-레이스 계열(C200) 구조 봉합.
4. **abstain의 actionable화** (§6) — 결핍 operand 필드명+getter 지목. 020/026형(coverage 붕괴) 직접 회수.
5. **KB 출력 상한 + VIEW_COMPACT 문턱 하향** (§2.3) — 남은 최대 축(44.8%).
6. **compute를 참조-전달로** (§2.2·§6) — 재직렬화 비용·행 유실·account_open 누락/날조 문제를 **한 번에** 제거
   (엔진이 env에서 직접 읽으면 개설일도 레코드에서 옴).
7. `T2_UNAVAIL` NameError 1줄 수정 + 예외 무음 통과 금지 (§4).
8. DUPLICATE-COMPUTE 스텁에 이전 결과 재제시 (§7) — 020형 재호출 루프의 유인 제거.
9. **task_005는 gold가 깨졌다** — 분모/판정에서 처리 방침 결정 필요 (§5.2).
10. 러너의 실패-sim 궤적 영속 (§0.1) — infra 원인 규명 가능하게.

## 9. 정정 기록 (전판 + 본 문서 초판)

- "ctxover 7건 = KB 출력 누적"(전판) → **부분적으로만 맞다.** KB는 44.8%이고, **직접 사인은 C205의 8,192 예약**(§1).
- "unavail 223" = 작동 계측이 아니라 **전량 NameError 실패**(§4).
- infra 오진 2건은 전판이 이미 정정 — 여기서 **코드 3지점으로 확정**(§3).
- "값 오전사" 가설 — 이 런에서는 022 1건뿐(§6).
- **[초판 오류] "008 reason 인자 미기입"** → 틀림. reason은 넣었고 **티어 선택 오류 + 강제-읽기 후 무갱신**(§5.1).
- **[초판 오류] "020/026 저검출 = A2 rate 표 결손"** → 틀림. 표는 비활성 경로(ratefix variant)이고
  실기전은 **`account_open` 누락(정직) vs 날조(027·우연히 정답)**(§6).
- **[초판 미달] "NOTICE 템플릿이 묻게 만든다"** → 표층. 실기전은 **터미널-턴 복권**(§5.1b) —
  gap 없는 과제는 동의 후 행동 턴이 아예 없다.
