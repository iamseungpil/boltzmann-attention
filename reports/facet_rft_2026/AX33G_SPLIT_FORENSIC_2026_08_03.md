# AX33 run-g — 분열 8쌍 정밀 포렌식 (2026-08-03)

원자료: `sim_results/bank_ax33n_gpu{0,1}_20260803g.results.json.gz` (커밋 `583af743`)
분석기: `scripts/distill/tau2/ax33g_split_forensic.py` (커밋 `2218f776`) · 출력 `/home/woori/scratch/ax33fx/split.txt`
런: banking_knowledge · front32 × nt2 · alltools · 32B-GPTQ · user-sim gpt-5.2 reasoning low

## 0. 집계 (판정 근거 아님 — §1이 판정)

64/64 sim 완주. pass(sim) 24/64 = **0.375** · **pass^2 8/32 = 0.250** · any-pass 16/32 · **분열 8/32 = 25%**.
종료: user_stop 62 · context_window_exceeded 1 · max_steps 1.
유지율 p²/p¹ = 0.667 (참고: GPT-5.5 36.94/46.39 = 0.796 — 스위트·nt·gold 다름, 방향 지시로만).

## 1. 분열 8쌍의 최초 발산 지점 — 전수 분류

| task | 통과 trial | 최초 발산 | 분류 |
|---|---|---|---|
| 003 | t1 | t0가 호출 1회 후 종료(70s/8msg) vs t1 2회(400s/12msg) | **미행동** |
| 018 | t0 | call#0 `KB_search_bm25` 질의어 *"verify customer identity"* vs *"verify identity"* | **검색 질의어** |
| 020 | t1 | call#0 동일 질의어 차이 | **검색 질의어** |
| 021 | t0 | call#1 *"verify customer identity"* vs *"check user identity"* | **검색 질의어** |
| 027 | t1 | call#3 `verify_identity` vs 3번째 `KB_search_bm25` | **검색 질의어**(재검색 반복) |
| 028 | t0 | call#9 `get_reward_discrepancies` 인자 — t0 `@last:` **byref** vs t1 **리터럴 JSON 인라인** | **byref vs 인라인** |
| 034 | t0 | t1이 **도구 호출 0회**(22msg/177s) | **미행동** |
| 035 | t0 | call#0 `get_user_information_by_name(customer_name="April 18, 1992")` — **생년월일을 이름 자리에** | **operand 오배치** |

**분류 집계: 검색 질의어 4 · 미행동 2 · byref 1 · operand 1 = 8/8 귀속 완료.**

## 2. ★레버는 분열의 원인이 아니다 (8/8)

- **어떤 분열 쌍에서도 레버 deny가 실패 trial을 만들지 않았다.** SIG·GIVE_QUOTE·TIER·DISPATCH는 8쌍 **전부에서 미발화**.
- 발화한 레버는 `[coverage]` 하나뿐(018·020·021·027·028). 018·028은 **양 trial 동일 텍스트**로 발화 = 판별력 0.
- 020·021·027은 실패 trial에서 발화가 **더 많지만**, 그 텍스트가 가리키는 행 수·점수가 서로 다르다
  (021: t0 *"17 of 17 rows"* vs t1 *"8 of 8 rows"* · 027: t0 *"2550/1020 points"* vs t1 *"600/1499 points"*).
  즉 레버는 **이미 갈라진 문맥을 보고한 것**이지 갈림의 원인이 아니다 — 발산은 그보다 앞선 call#0~#3에서 났다.

⇒ **분산의 원천은 레버 스택 밖에 있다.** 레버를 조정해서 pass^2를 올릴 수 없다.

## 3. ★지배 원인 = 검색 질의어 (4/8)

018·020·021·027 네 쌍 모두 **동일 의미의 신원확인 정책 조회**에서 갈렸다:

```
"verify identity" / "verify customer identity" / "check user identity" / "verify user identity"
```

같은 것을 묻는 네 표현이 서로 다른 문서 집합을 회수하고, 그 뒤 궤적 전체가 달라진다.
실패 trial은 대개 **재검색을 반복**한다(021 t1: BM25 3연속 · 027 t1: BM25 3연속 · 020 t0: bm25→dense→bm25).
그리고 실패 trial이 **호출 수가 더 많다**(020 24>17 · 021 19>11 · 027 29>20) = 회수 실패 후 헤맴.

이 채널은 **모델의 자유 서술**이고 **엔진이 결정론으로 덮고 있지 않다**. 우리 레버는 전부 회수 *이후*에 건다.
`RETRIEVAL_FANOUT_DESIGN_2026_08_03.md`가 겨냥한 자리가 정확히 여기임이 실측으로 확인됐다.

## 4. 부수 확인

- **byref(028)**: `@last:` 참조로 넘긴 trial이 통과, 같은 데이터를 **리터럴 JSON으로 인라인**한 trial이 DB 불일치 + 12건 action 누락으로 실패. 핸드오프 P7(byref 결함) 처방이 **부하를 지고 있음**이 확인됐다.
- **미행동(003·034)**: 003 t0은 호출 1회 후 종료(`give_discoverable_user_tool` 미수행), 034 t1은 **호출 0회**로 종료. 의무가 열린 채 종료하는 경로에 게이트가 없다.
- **operand 오배치(035)**: `customer_name` 자리에 생년월일. 첫 호출에서 나고 궤적 전체가 어긋난다.

## 5. ⚠채점 기준 단서 (비교 시 반드시 명시)

`reward`와 `action_checks`가 **8쌍 중 3건에서 불일치**한다 — 020 t1·027 t1은 action 누락 2~4건인데 `reward=1.0`
(`reward_basis=["DB"]`), 034 t0은 DB 불일치인데 `reward=1.0`(basis가 DB 아님).
⇒ **"pass"는 "옳은 행동을 했다"와 다르다.** 태스크마다 basis가 달라, action 수준 정확도는 pass가 시사하는 것보다 낮다.
리더보드 비교·논문 수치에서 이 구분을 흐리지 말 것([[54]] 비교규격 항목에 추가).

## 6.5 ★§2 정정 — 태그 없는 엔진 개입 2종을 놓쳤다 (2026-08-03 추가)

§2의 "레버는 원인이 아니다"는 **`[T2_*]` 태그가 붙은 레버에 한해서만** 참이다. 분석기의 정규식이 태그를
요구해서, 태그 없는 엔진 개입 **2종을 통째로 못 봤다**. 궤적의 짧은 KB 응답을 축자로 열어 발견:

1. **도구명 리다이렉트** (169자) — 축자: *"`verify_identity` is not managed by `KB_search_bm25`. …
   call `verify_identity` directly"*. 평문 정책 질의가 도구명과 겹치면 정책 문서 대신 이 메시지가 나간다.
2. **`[DUPLICATE-READ]`** (572~857자) — 동일 (도구, 인자) 재호출 시 억제 노트. 3회째부터 축자로
   *"You have now issued this IDENTICAL call 3 times … Do not issue this same call again."* 로 격상.

**64 sim 상관 (무료·전수)**:

| 조건 | n | pass |
|---|---|---|
| redirect 0회 | 51 | 0.373 |
| redirect ≥1 | 13 | **0.385** |
| dup 0회 | 48 | 0.396 |
| dup ≥1 | 16 | 0.312 |
| dup ≥5 | 2 | **0.000** (평균 호출 64.0) |
| 최대-반복 ≥4 | 9 | 2/9 = 0.22 |

- **리다이렉트 가설은 기각**: 연관 없음(0.373 vs 0.385)이고 쌍 내부에서도 방향이 엇갈린다
  (018은 실패측만 발화 / 020은 **통과측만** 발화 / 021·027은 양쪽 동일). 원인이 아니다.
- **분열 8쌍은 여전히 단일 원인 미확정**: dup 개수도 쌍 내부에서 엇갈린다(020·027 실패측 多 / 028 통과측 多).
  §1의 분류는 *발산 지점*의 기술이지 *원인의 확정*이 아니다.

**★그러나 별건으로 가장 큰 것이 나왔다 — `task_041 trial0`**:
동일 인자로 `get_credit_card_accounts_by_user`를 **85회** 호출. 총 호출 97 · DUPLICATE-READ **83회** ·
"IDENTICAL call" 격상 경고 **80회** 수신 · reward 0.0. **엔진이 80번 명시적으로 "같은 호출을 또 내지 말라"고
말했고, 모델은 85번 냈다.**

⇒ 이것은 **표면화(note) 모드의 완전 실패의 존재 증명**이며, [[49]](재검색 지시해도 같은 것을 반복)·
Recuse Signal(작업 중 정지 0/40)·2607.22868의 **deny vs note 화이트스페이스 ⓐ**와 정확히 같은 지점이다.
단 dup≥5는 **n=2**이므로 비율은 존재 증명으로만 읽을 것.

## 7.5 정독 확장 — 40 실패 sim + 사용자-구조 계측 (2026-08-03 야간)

도구: `ax33g_perstep.py` · `ax33g_taskcause.py` · `ax33g_rescue_scan.py`.
⚠**§1 재교정**: gold action은 `requestor`를 갖는다(user 132 / assistant 159). MISS 87건이 user-requestor
(대부분 `call_discoverable_user_tool` 76)이며 **에이전트가 부르면 안 되는 것**이다. §1의 "NEVER ATTEMPTED"
서술 중 상당수가 이 아티팩트였다.

### 사용자-구조(rescue) 계측 — 내 과대주장 정정
정의(결정 가능): 사용자 발화가 도구 X를 명명 ∧ 그 전까지 에이전트가 X 미호출 ∧ 이후 호출.
결과: **user-taught ≥1 n=15 pass .467 / 없음 n=49 pass .347. 통과 24건 중 7건**(29%).
⇒ 내가 손으로 읽은 6건에서 "통과 대부분이 사용자 덕"이라고 적었던 것은 **표본 선택 편향**(분열 태스크만 읽음).
정확히는 **7/24이며, 이 정의는 인자 교정·재촉 같은 구조는 못 세므로 하한**이다.

### 엔진 메시지 사건 × 결과 (64 sim)
| 사건 | 총발화 | 있음 n / pass | 없음 n / pass |
|---|---|---|---|
| GRANT_ERR "has not been given to you" | 39 | 21 / **.429** | 43 / .349 |
| UNKNOWN "Unknown discoverable tool" | 13 | 11 / .545 | 53 / .340 |
| ARGERR "Unexpected/Missing parameter" | 9 | 3 / **.000** | 61 / .393 |
| **ASKED "Would you like to be transferred"** | **43** | **29 / .310** | 35 / .429 |
| DUPWRITE "already exist" | 8 | 8 / **.125** | 56 / .411 |

⚠**E2 격하**: GRANT_ERR은 실패와 **연관되지 않는다**(오히려 pass가 높다 — 인계를 *시도한* 런에서만 나오므로).
E2는 여전히 **실증된 메시지 결함이자 턴 낭비**지만, **실패의 원인으로는 미입증**이다.
실패와 연관되는 것은 **ASKED**(최대 n) · DUPWRITE · ARGERR이다.

### 실패 40 sim의 두 계열
- **계열 A — `Executed:` 0회 (약 18 sim)**: discoverable 경로에 **도달조차 못 함**.
  005·007·010·012·014·015·016·023·024·032·033·034t1·035t1·041t0·003t0·022t1. ASKED와 조기 인간이관이 지배.
- **계열 B — `Executed:` >0 (약 12 sim)**: 실행했으나 불완전·오인자.
  018t1·019·020t0·022t0·026·027t0·028t1·040·041t1.

### task_027 t0 — 커버리지 붕괴의 진짜 위치
[36] `get_reward_discrepancies` 를 **손으로 타이핑한 리터럴 거래 목록**으로 호출(byref `@last:` 아님)
→ [38] 축자 *"We have identified **one** discrepancy"* — **gold는 4건**.
⇒ **에이전트가 조기에 멈춘 게 아니라, 1건뿐이라고 보고받았다.** 이후 처리는 그 잘못된 분모에 대해 완결적이다.
`[coverage] N of M rows checked` 레버는 **주어진 입력에 대해 정직하게** "12 of 12" 를 보고하므로, **틀린 분모를
인증**한다. 같은 단계에서 020 t1(통과)은 `@last:` byref를 써 4건을 전부 얻었다.
추가: `unlock_discoverable_agent_tool` **3회**[44][62][70] · 종말부 `check dispute status` 검색 6회(DUPLICATE 2).

⇒ **byref-vs-리터럴은 P7의 인자 편의 문제가 아니라 커버리지의 상류 원인**이다.

## 8. 판정과 다음

1. **k축 모트 명제는 이 런으로 반증되지 않았다** — 결정론이 분산을 못 줄인 게 아니라, **분산이 실린 채널에
   결정론이 걸려 있지 않았다**. 레버는 회수 이후에만 작동하고 갈림은 회수에서 났다.
2. **pass^2를 올리는 유일한 큰 레버 = 회수 채널 결정화**(질의 정규화/팬아웃/중복질의 흡수). 4/8이 여기.
3. 그다음 = **미행동 종료 게이트**(2/8) · **byref 강제**(1/8·P7 기존 처방).
4. 한계: 32 태스크·2 trial·단일 도메인. 분열 8건은 소표본이므로 위 비율(4/8 등)은 **순위 지시**로만 쓸 것.
