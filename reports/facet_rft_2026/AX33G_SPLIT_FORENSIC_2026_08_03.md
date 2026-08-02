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

## 6. 판정과 다음

1. **k축 모트 명제는 이 런으로 반증되지 않았다** — 결정론이 분산을 못 줄인 게 아니라, **분산이 실린 채널에
   결정론이 걸려 있지 않았다**. 레버는 회수 이후에만 작동하고 갈림은 회수에서 났다.
2. **pass^2를 올리는 유일한 큰 레버 = 회수 채널 결정화**(질의 정규화/팬아웃/중복질의 흡수). 4/8이 여기.
3. 그다음 = **미행동 종료 게이트**(2/8) · **byref 강제**(1/8·P7 기존 처방).
4. 한계: 32 태스크·2 trial·단일 도메인. 분열 8건은 소표본이므로 위 비율(4/8 등)은 **순위 지시**로만 쓸 것.
