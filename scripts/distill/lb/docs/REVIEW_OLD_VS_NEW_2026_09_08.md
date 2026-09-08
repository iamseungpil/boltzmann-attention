# 구 코드베이스(`tau2/`) 대 신 코드베이스(`lb/`) 정밀 대조 — 2026-09-08

질문: 일곱 레버는 거의 독립인데 왜 버그가 계속 나오는가.
답: 레버끼리 얽혀 난 버그는 **0건**이다. 여섯 결함 전부 **엔진과 바깥의 경계**에서 났다 — 구 선언을 그대로 들여온 것,
구 술어는 옮기고 그 술어를 감싸던 상한·정규화를 안 옮긴 것, 인자 뷰를 잘못 고른 것. 그리고 공정 실수 하나:
라이브에 올리기 **전에** 실제 궤적으로 재현 검사를 안 돌렸다. 이 문서 이후 재현 검사(`lb_replay.py`, base 54 태스크 216 sim)는
배포 전 관문이다.

## 1. 라이브에서 잡힌 결함 여섯 — 어디서 났나

| # | 결함 | 경계 | 고침 |
|---|---|---|---|
| 1 | LB3 schema 가 디스패처 페이로드를 래퍼 인자로 읽어 280건 오거부 | 인자 뷰 | 래퍼 인자만 (`as_dict(call.arguments)`) |
| 2 | LB4 follow-up 이 decision_tools 없는 사슬에서 IndexError | 선언 결손 | 표적 대체 |
| 3 | 절차 도구를 지목하는 write-evidence 선언 5건(구 HARMFUL) 그대로 이전 | 구 선언 | 이번에 종류 자체 삭제(§2 D2) |
| 4 | claims 감사가 문장을 바꿔가며 13회 발화 | 예산 단위 | 규칙당 예산 |
| 5 | 절차가 read 를 10회 거부(048) | 술어만 이전 | `write_tools` 에만 deny |
| 6 | base 는 `max_tokens` 없음, 신 러너는 8192 | 실험 조건 | 기본 None (§2 P1) |

## 2. 구 가드 전수 판정 — 세 칸

**① 일반화되어 규칙 하나로 흡수** (코드에 들어간 것)

| 구 코드 | 신 코드 | 왜 일반인가 |
|---|---|---|
| 레버별 상한 25종 (`T2_PROCEDURE_CAP=6` `T2_VERIFY_DENY_CAP` `T2_FOLLOWUP_CAP=3` …), 소진 = 통과 | `lb_coordinator.DENY_BUDGET = 6` 규칙당·sim당, 초과분은 `lb-release` 로 기록하고 통과 | "같은 모델에게 계속 지는 규칙은 진 것이다, sim 은 살아야 한다" 는 하나의 문장 |
| `T2_REGEN_BUDGET=12` | `lb_runtime.REGEN_BUDGET = 12` sim당 (턴당 `ROUNDS=3` 은 유지) | 재생성은 sim 단위로 유한해야 한다 |
| `_ctx_has` 의 `#` 제거 | `lb3.norm()` — 대소문자·구두점 제거 | 렌더링은 증거가 아니다 |
| `_wev_expand` 의 `%g %.1f %.2f int` | `lb3.renderings()` — 수치는 관용 표기 전부 | 같은 값의 다른 표기 |
| `_hint_hit` (인자명 토큰 힌트) | `identifying.args` 에 토큰 매칭 (`txn_ref` ⊃ `txn`) | 인자 **이름** 규약; 값 모양 추정은 삭제(048 에서 날짜·금액 오거부) |
| `_ctx_fits` ((hist+len)/3.5 ≤ cap−gen−1024−11000) | `lb7.room()` — 남은 자리만큼 전달, 자리 없으면 문서 **이름만** | 컨텍스트에 안 들어갈 것은 넣지 않는다 |
| viewmax2 `T2_VIEW_COMPACT_MINTOTAL=344064` (=0.75·131072·3.5) | `lb6.FOLD_AT=0.75` × `a2.model_context` | 문턱은 모델 컨텍스트에서 유도, 상수 금지 |
| `_SRC8` 선점 사슬 | `LB_ORDER` + 표적당 명령 하나 (이미 있음) | 순서 하나로 충분 |

**② 실험 조건 동등성** (레버가 아님, base 와 같아야 하는 것)

| 항목 | base | 신 코드 |
|---|---|---|
| `max_tokens` | 없음 | 없음 (P1) — `--max_tokens` 를 명시할 때만 |
| `temperature` | 0.0 | 0.0 |
| user-sim | gpt-5.2 temp 0 low | 동일 |

**③ 옮기지 않은 것** (과제·도메인 관용구, 또는 이미 다른 규칙이 덮음)

| 구 코드 | 이유 |
|---|---|
| `DEFAULT_ARG_HINTS = ("email","name","zip",…)` | 도메인 리터럴. 필요하면 `identifying.args` 데이터로 |
| `_dup_stub_content` + bm25/kb 재검색 유도 산문 | LB6 뷰 dedup 이 부하는 이미 접는다; 산문은 검색 관용구 |
| `T2_SUPPRESS_AUTH` `T2_SG_TRUTH` `T2_KB_NOHIT_K` `T2_SEARCH_EXHAUST_TH` `T2_TRANSCRIBE_CAP` `T2_TOOLLIST_CAP` `T2_REPEAT_GOV` `T2_REGEN_KEEP_MUTATING` `T2_REGEN_WRITE_GATES` | 과제 관용구 또는 예산 하나로 흡수됨 |
| `T2_REQUIRE_DOC_DELIVER_CAP=3` | 조언 예산 2/규칙 이 덮음 |

## 3. 재현 검사가 추가로 잡은 것 (base 216 sim, 라이브 전에)

| 규칙 | 발화 | 판정 | 처리 |
|---|---|---|---|
| LB1 `absent` (핀) | 180 | 043 msg5 = 모델이 신원 확인을 **묻는** 턴에 다음 단계를 핀. 핀은 창·지문·예산을 전부 우회했다. 인계 시 미완 단계는 LB5 `steps-open` 이 이미 본다 | **D1 삭제** |
| LB3 `tokens` (write-evidence) | 53 | 049 base 통과 sim 이 `CLOSURE_OK` 없이 close 하고 reward 1.0. 판정 문자열 요구는 인용이 아니라 특정 검사의 **처방** | **D2 종류 삭제**, `_dropped` 에 기록 |
| LB3 `identifying` 값 모양 추정 | 5 → 1 | 날짜·금액 오거부 | 선언 인자만 |
| LB5 `uncalled-unlock` | 28 (빈 슬롯) | 구 `tool_unlock_hint` 문장을 빌려 `{tools}` 슬롯이 비었고 뜻도 반대(잠금 해제 **안 된** 도구용) | 규칙 자체 문장으로 교체 |

검사 후 잔여 발화: LB1 requirement 164 (그중 137 = `log_verification` 전에 `verify_identity` — LB2 검증기 요구, sim당 1회, **판단 보류: 결손 측정 전이라 유지하되 표시**), LB4 follow-up 71(조언·예산 2), LB5 28, LB7 17, LB3 2(플레이스홀더 `YOUR_USER_ID` = 정당).

## 4. 배포 전 관문 (이 순서, 매번)

1. `python tests/test_lb.py` 23/23
2. `python lb_replay.py sim_results/bank_x806_base_nt4_task_*.results.json.gz` — base 통과 sim 에서 deny 가 늘면 그 규칙을 읽는다
3. 8141 프로브: 레버별 대표 태스크 nt=1, 1분 단위 `lb_tick.py`
4. 그 뒤에만 라이브

## 5. 규칙별 손실 위험 대 이익 가능 (base 201 sim 재현, 2026-09-08 오후)

판정 기준(사용자 지시): **4/4 태스크에서 잃는 규칙은, 다른 태스크에서 얻는 것이 확실할 때만 고친다. 확실치 않으면 뺀다.**

| 규칙 | 통과 sim 발화 | 실패 sim 발화 | 구 원장 §7-1 의 양의 칸 |
|---|---|---|---|
| LB1 requirement deny | 97 | 67 | — |
| LB4 follow-up | 36 | 35 | — |
| LB5 steps-open | 20 | 2 | HANDOFF_PREDICATE **028 0/2→2/2** (음: 019·029) |
| LB1 procedure surface | 17 | 7 | — |
| LB5 uncalled-unlock | 13 | 15 | UNLOCK_QUIET **010 +1** (음: 099) |
| LB7 value-acquire | 12 | 5 | VALUE_ACQUIRE = 전제가 거짓(원장 124) |
| LB7 deliver | (호출 턴) | — | DELIVER_PRECOMMIT **024 2/4→3/4** |
| LB1 procedure deny | 0 | 10 | — |

`LB1 procedure deny` 는 통과 sim 발화 0 이다 — 위험 없는 유일한 deny.
`LB5 steps-open` 은 20 대 2 로 위험이 크다. 028(양)·029(음) 프로브로 정한다.

## 6. 구 원장이 이미 적어 둔 두 결함이 오늘 재현됐다

- `T2_CLAIM_PROV`(원장 123): *"log_verification 이 양쪽 sim 에서 실행됐는데 ledger shows no such event"*. 오늘 프로브 004 에서 같은 문장이 나왔다. 원인은 실행과 성공의 혼동이었고(`NOT_VERIFIED` 를 실패 표지로 셈), `Turn.attempted` 로 분리했다.
- `T2_TOOL_SIGNATURE`(원장 116): *"task_017 tr0 turn 53 · submit_cash_back_dispute_0589 · reward 0.0"* — gold 호출을 막았다. 우리 LB3 schema 는 래퍼 인자만 보므로 그 사거리 밖이고, 오늘 017 프로브에서 같은 호출이 두 번 통과했다.

## 7. 프로브 집합 (이익 칸과 손해 칸을 함께)

017 031 048 049 004 028 001 010 029 024 007 023 — base 성적 = 017·001·004·007·023·024 4/4 · 028 3/4 · 010·029 0/4.

## 8. 원장 근거로 규칙 하나 폐기 (2026-09-08)

**뺀 것 — LB7 `named_uncalled`** (구 `T2_HANDOFF_PREDICATE` · `named-but-not-given`)

| 근거 | 축자 |
|---|---|
| 단일변수 A/B | `2/12 ↔ 2/12` (t7308 ctl/treat) |
| 유일한 양의 칸을 원장이 부인 | *"HANDOFF 의 +2 는 기전이 레버와 무관 — 실제 변화는 env 인자 오류 소멸이지 named-but-not-given 술어가 아니다"* |
| 비용 | 지연 **1.90×**(67,949s→129,179s) · context_window_exceeded **13 ↔ 0** |
| 표적 인구 | 근거였던 62% 가 이 가족에서는 **10%** |
| 판정 | *"폐기확정"* · *"HARMFUL"* · 기본 OFF |

**남긴 것 — LB4 `claims`**, 조건부. 이득 기록이 **있다**: 017#1 에서 허위 *"I have enabled the tool"* 3회를 잡아 재생성시켜 *"reward 1.0 을 실제로 샀다"*. 단 같은 줄이 *"모델 날조 문구의 우연에 선행 의존 — 안정 능력 아님"* 이라 적는다.
반대 기록: 과거형 고발 **오발화 88%** · 빈손 재생성 65~73% · 019 붕괴 · 등급 *"효과 [?]"*.
우리 판은 표면화 전용 · 규칙당 예산 2 · 오발화의 직접 원인(실행을 성공과 혼동)을 `Turn.attempted` 로 제거했다.
**중단 기준**: 프로브에서 오발화가 참 발화보다 많으면 뺀다. 현재 거짓 1(004) · 참 1(031).

## 9. 원장 근거로 규칙 하나 더 폐기 — LB5 `uncalled-unlock` (2026-09-08 오후)

구 `T2_UNLOCK_QUIET`(C408 · x241 · n=8) 는 **잠금만 하고 안 부른 도구가 있는 동안 우리 층이 입을 다무는** 레버였다. 그 근거 측정:
*"궤적만 주면 8/8 이 그 도구를 부른다. 우리가 실제로 넣었던 문장들을 되돌리면 1/8 이고, 우리 문장 하나만 얹어도 4/8 이다. 이 상태에서 우리 조언은 도움이 아니라 경쟁 지시다."*
LB5 `uncalled-unlock` 은 정확히 그 상태(unlocked − called ≠ ∅)에서 말하는 규칙이다. 구 L2 `T2_UNCALLED_UNLOCK` 에는 양의 칸이 없고(원장 237: 표적 인구 62%↔10% 의문), 재현 census 는 통과 sim 13 · 실패 sim 15 였다. ⇒ 폐기.
`UNLOCK_QUIET` 자체와 `DECISION_ISOLATE`(결정 블록 메시지에서 상품 목록을 지우던 억제 레버 · HARMFUL) 는 이 트리에 이식된 적이 없다 — 폐기할 대상 없음.

task_010 (base 0/4 · 우리 0/1 · 개입 0): 값 계산은 전부 맞았고, 문서 조항("거절된 추천은 복구 불가")을 상황에 잘못 맺은 뒤 문서에 없는 전제("링크는 신청 전에 있어야")를 지어 "재제출 불가"라 답했다. **F3 + 근거 없는 추론** = 결정기의 자리가 아니다. 보류.

## 10. 기록 상태 근거 (LB3 `state`) 복원 — 026·027·028·029 가족 (2026-09-08 오후)

이 넷은 같은 기만 국면을 가진다: 분쟁 제출 뒤 손님이 *"모두 승인·해결됐다, 기록을 갱신해 달라"* 고 **거짓말**한다. base 는 16 sim 중 3 통과. 027·029 는 12/12 sim 이 그 말을 믿고 `update_transaction_rewards` 를 4~7회 불러 실패했다. 028 은 환경의 제출 결과가 스스로 `Status: RESOLVED - approved` 라 말하므로 갱신이 gold 다.

⇒ 규칙: **행동이 의존하는 상태는 그 값을 이름 대는 환경 출력에서 읽어야 한다. 손님의 말은 기록이 아니다.** 아침에 지운 `tokens` 종류에서 우리 검증기 판정 문자열(CLOSURE_OK·ELIGIBLE)과 도구 이름(선행 읽기 = LB1)을 걸러 내면 남는 것이 이것 하나(`RESOLVED`)다. 판단 0, substring 1.

| 태스크 | base | `state` deny (재현) |
|---|---|---|
| 028 | 3/4 | 0 |
| 027 | 0/4 | 14 |
| 029 | 0/4 | 19 |
| 026 | 0/4 | 0 |

같은 자리에서 반대 방향으로 밀던 LB4 follow-up 선언(제출 → 갱신 요구)은 삭제했다: 결정 단계 없는 맨 쓰기 요구는 대장이 볼 수 없는 조건(해결됨)을 전제한다.

## 11. LB1 · LB5 규칙별 판정 (2026-09-08 오후, 사용자 질문 "전부 폐기인가")

| 규칙 | 통과 sim 발화 | 이득 기록 | 판정 |
|---|---|---|---|
| LB1 `procedures` deny | 0 | 없음 | 유지 — 잃는 자리가 없다 |
| LB1 `prerequisites` 정책 축자 (검증 후 계좌 접근 · 분쟁 전 이력) | 27 | 없음 | 유지 — 031 에서 정상 작동 |
| LB1 `prerequisites` `verify_identity` 요구 | **137** | 없음 | **삭제** — 우리 검증기를 쓰라는 처방(`_env_reads`) |
| LB5 `steps-open` | **20** | 028 +2 (원장이 부인) | **삭제** |
| LB5 `doc-unread` · `search-exhausted` | 1 · 0 | 없음 | 유지 — 비용 0 |

`absent` = 구 `T2_PROC_ABSENT`(L8 부재종결·"K턴 무호출이면 체크리스트 표면화"·원장 *사전 기대치 null*). `uncalled-unlock` = 구 L2 `T2_UNCALLED_UNLOCK`(잠금 해제 후 미호출 이름 나열·`UNLOCK_QUIET` 측정 8/8→1/8 이 반증).

## 12. 첫 클라우드 짝 A/B 가 잡은 손실 — LB7 `deliver` 폐기 (2026-09-08 16:30)

| 태스크 | base | LB (`e723eed4`, nt=4) | 발화 |
|---|---|---|---|
| 001 | 4/4 | 4/4 | 0 |
| 004 | 4/4 | **2/4** | deliver deny 10 · claims 4 |

004 의 gold 는 `transfer_to_human_agents` 하나다. `deliver` 가 그 호출을 sim 마다 2~3회 거부하며 "정의 문서"를 실었는데, `docs_naming` 은 본문에 도구 이름이 있는 문서를 전부 집어 *Understanding Regulation E* 같은 무관한 문서를 넘겼다. 실패한 두 sim 은 끝내 이관하지 않았고, claims 감사는 그 뒤 "이관했다"는 거짓 진술을 정확히 잡았다(우리 deny 가 만든 거짓).
원장 근거: `DELIVER_PRECOMMIT` 024 2/4→3/4 (+1, 잡음 바닥 아래). x829(0/8→8/8)는 결정 **전** 격리 문맥에 재료를 준 측정이지 호출 거부가 아니다. ⇒ `deliver` 종류 삭제, `have_value` 만 남김. 004 는 큐 앞에 되돌려 재실행.

## 13. 클라우드 짝 A/B 15 태스크 (17:50) — 049 손실과 수리

| 태스크 | base | LB | 판정 |
|---|---|---|---|
| 012 · 031 | 3/4 | **4/4** | 이득 |
| 007 | 4/4 | 3/4 | 발화 0 · 손님이 신청을 안 함 = 잡음 |
| **049** | 3/4 | **0/4** | 손실 |
| 나머지 11 | 4/4 또는 3/4 | 동일 | — |

049: 4 sim 모두 모델이 *"$5 statement credit 을 드리겠다"* 고 말만 하고 `apply_statement_credit` 없이 이관. base 는 손님의 되물음 뒤 실행(3/4). 우리 절차 deny(읽기→사유→폐쇄)가 흐름을 바꿔 그 되물음 구간이 사라졌다. 이것은 LB4 `pending`(약속한 행동이 인계 전에 실행됐는가)의 자리인데 두 결함이 막았다: ① `event_map write → __effective_write__` 가 "어떤 쓰기든 하나 있으면 뒷받침" 으로 읽혀 도구 이름을 무시(`backed()` 를 도구 우선으로) ② 인계 **호출** 턴의 조언은 재생성을 안 일으켜 모델에게 닿지 않음(인계 호출은 한 번 미루고 약속을 보인다 — 거부가 아니라 재생성, 예산 2). 재실행: 049 · 004(가드).

## 14. 056 (base 3/4 → LB 2/4) per-step — F2 이되 닫을 수 없는 자리

실패 sim0 은 `open_bank_account(account_class="Silver Plus Saver")`, gold·통과 sim 은 `"Silver Plus Saver Account"`. 환경은 잘린 이름도 받아 다른 계좌 id 를 만들었다(DB 불일치). 우리 발화는 이 자리에 없었다(deny 0).
선언 `CHOICE-GROUND`(account_class 는 문서에 있어야) 는 substring 이라 잘린 이름을 통과시킨다. 정확 일치로 조이면 gold 를 막는다: gold 의 범주 이름 23개 중 6개가 문서 제목과 다르다 — `Navy Blue`(제목 Navy Blue Account) · `World Blue Account`(제목 World Blue) · `Green Account` · `Evergreen Account` · `Purple Account`. "더 긴 제목의 접두면 잘린 것" 규칙도 `Navy Blue`·`Green Account` 를 오판한다. 범주 이름 인자를 쓰는 gold 행위는 37 태스크. ⇒ 규칙 추가 없음, 결손 기록. (079·078 지갑 분실 가족은 F3 집합 선택 + 순서를 명령하는 문서 부재 — §13 참조.)

## 15. HARD 25 의 이력 (전 기록 2,197 파일 · 1,982 sim, 중복 제거)

| 태스크 | 통과 런 | 모델 |
|---|---|---|
| 029 | `ax33n_gpu1`·`b4_gpu1` (08-03) 각 2/2 · `n97` 1/2 | Qwen2.5-32B |
| 027 | 같은 세 런 각 1/2 | Qwen2.5-32B |
| 010 | `n97` 1/2 · `night2p1_t3prime` (09-01) 1/1 | Q2.5 · Q3.8 |
| 026 · 060 · 065 · 067 · 068 | 각 1~2회 (nt=1 런) | 026 Q2.5, 나머지 Q3.8 |
| 039 046 053 061 066 069 077 082~088 091 092 102 (16) | **0회** | — |

029 를 2/2 로 푼 08-03 팔의 `go_stack.sh`(`8086c8ab`)는 `T2_WRITE_EVIDENCE=1` 이 켜져 있었고 07-31 커밋 *"Stop the dispute-evidence gate from accepting a dispute the bank won"* 직후였다 ⇒ 오늘 복원한 LB3 `state` 와 같은 기전. Q3.8 에서는 미검증 — 이번 런의 027·029 가 검증.
원인 분류(§13 digest): A. 참조·추천 판단 F3 17개(지갑 분실·사기 가족 10 + 계좌 재구성 7) · B. 기만+다행 계산 3개(026 027 029, 우리 사거리) · C. 긴 다중 목표 사슬 3개(039 046 053) · D. 오독·벤치마크 오류 2개(010 102).
