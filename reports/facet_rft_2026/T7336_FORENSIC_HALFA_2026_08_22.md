# T7336 halfA 재런(수리 스택) 20 sim 전수 포렌식 (2026-08-22)

> ⚠**기준선 정정(2026-08-22·gz 직독·`sim_results/bank_t7328_halfA_20260819r.results.json.gz`)**: t7328 halfA 는 **4/20**(073·004·100×2)이다 — 본문의 "t7328 2/20" 은 오계수. 전체 t7328 = 6/40(원장 C590 그대로). 부호표의 004(1→2)·073(1→0)·100(2→2) 해석은 이 값으로 읽을 것.


- 런: `bank_t7336_halfA_20260821b` (리모트 `/home/woori/scratch/tau2-bench/data/simulations/bank_t7336_halfA_20260821b/results.json` · 로그 `/home/woori/scratch/logs/bank_t7336_halfA_20260821b.log`)
- agent = Qwen2.5-32B-Instruct-GPTQ-Int8 · user-sim = gpt-5.2(reasoning low) · nt=2 · seed 매핑: **trial 0 = s626729 · trial 1 = s373753**
- 수리 스택(t7335 처방 이행분·커밋): **P1** `5189b510`(CLAIMPROV 오탐→kind-fallback·050 DUP 계열) · **P2** `a0fcf07e`(GROUNDING WARNING 에코-그라운딩 제거·094) · **P3** 같은 커밋(get_atm_fee_discrepancies `requires_reads`+`grounded_params`·074) · **P4** `63443a09`(FAB fix-naming·079) · **P5** 같은 커밋(get_atm_fee_discrepancies 완결-인상 문구 제거·072)
- 변이 분류 전부 정본 `t2_forensic.mutation_diff(sim)`([[69]] reward 채점단위). 궤적 인용 `messages[i]`.
- 대조: `bank_t7328_halfA_20260819r`(기준선) · `bank_t7335_halfA_20260821`(수리 전·trial-0 계열, 003 만 trial-1 존재).

## 1. 태스크별 부호표 (reward)

| task | 7328#0 | 7328#1 | 7335#0 | **7336#0** | **7336#1** | 비고 |
|---|---|---|---|---|---|---|
| 003 | 0 | 0 | 0 (#1: 1) | **1 ↑** | **1 ↑** | t7335 trial-0 실패(단일 추천 붕괴)가 이번엔 양 trial 성공. 표적 레버 신설 없음 → 표본 2 의 분산 가능성 유보 |
| 004 | 0 | 1 | 1 | 1 | 1 | 유지 |
| 017 | 0 | 0 | 1 | 1 | **0 ↓** | trial-1 만 실패 — 검증 단계에서 '이름'을 끝내 안 물음(§017) |
| 024 | 0 | 0 | 1 | 1 | 1 | 유지 |
| 055 | 0 | 0 | 0 | 0 | 0 | 불변 실패 — class-fit 결정론 도구(t7335 처방 3)는 이번 수리에 미포함 |
| 072 | 0 | 0 | 0 | 0 | 0 | **양상 악화**: #0 게이트 데드락 too_many_errors · #1 계좌목록 미발견 transfer — t7335 의 "$12 vs $14" 지점까지 이번엔 도달도 못 함 |
| 073 | **1** | 0 | 0 (DUP) | 0 (분할) | 0 (무실행) | DUP 재발 0. 대신 #0 라인별 분할 credit·#1 credit 도구 미발견 |
| 093 | 0 | 0 | 0 | 0 | 0 | #0 은 (틀린) 보고서라도 제출 — t7335 의 "no discrepancy" 무행동보단 전진 |
| 094 | 0 | 0 | 0 | 0 | 0 | #1 은 보고서+credit 실행(값 stale) · #0 은 "수동 처리하겠다" 날조 후 write 0 |
| 100 | 1 | 1 | 1 | 1 | 1 | 유지 |
| 합 | 2/20 | | 5/11 | **9/20** (t0 5/10 · t1 4/10) | | compliance.json pass^1 0.45 · g3 위반 7 sim |

뒤집힌 자리: **003 0→1(양 trial)** · **017 trial-1 1→0**(t7335 는 trial-0 만 있었음·t7328 은 양 trial 0 — trial 간 분산 축) · 073 은 t7328#0 의 1 이후 세 런 연속 0 이되 **실패 기전이 매 런 다름**(DUP→분할→무실행).

## 2. 실패 sim 전수 — 변이표 + 결정 지점

### task_017 trial-1 (0.0 · 22 msgs · gold 3 변이 전부 MISSING·시도 0)

| 변이 | 내용 |
|---|---|
| MISSING | `log_verification{Kenji Tanaka, …}` · `submit_cash_back_dispute_0589{txn_cfabb609133d}` · `{txn_913d14a20dc5}` (전부·done 0) |

1. [4]~[7] bm25/dense 가 정답 문서(`doc_credit_cards_credit_cards_(general)_003` — user tool `submit_cash_back_dispute_0589` 지급 절차)를 1위로 회수. 발견 결손 없음.
2. [12] `verify_identity{DOB, phone}` → [13] NOT_VERIFIED — deny 본문이 fix 를 명명(축자): *"Look the customer up with **get_user_information_by_name**/by_email/by_id"*.
3. **실패 지점** [14]: *"Could you please provide me with your **email address or your user ID**"* — 조회 키 3종 중 **name 만 빼고** 물음. 손님(Kenji)은 email/주소를 *"I can't responsibly provide"*([19])라 못 주고, user_id 도 없음 — 정확히 못 주는 두 키만 반복 요구([16][18][20]). [21] 사용자가 접속 불가 선언 후 `###OUT-OF-SCOPE###` 종료.
4. **성공 대조(trial-0)** [30]: 같은 NOT_VERIFIED 후 *"Could you please provide me with your **full name** or email address"* → [31] 사용자 *"My full name is Kenji Tanaka"* → by_name 조회 → 검증 완료 → reward 1.0. 분기점은 요청 문장에 'name' 한 단어가 있었는가 하나.
5. 부수: [2] 무관 도구(interest report) unlock · [8] 가공 오류 서사(*"It looks like there was an error in my previous attempt"* — 그런 실패 없음) + 무관 `update_transaction_rewards_3847` unlock.

**원인: 모델**(질문 커버리지 — deny 가 by_name 을 명명했는데도 요청 문장에서 누락). env·user-sim 정당(시나리오상 손님이 확신 없는 값을 안 주는 것).

### task_055 (양 trial 0.0)

#### trial-0 (84 msgs)
| 변이 | 내용 |
|---|---|
| MISSING | `open_bank_account_4821{savings, "Silver Plus Account"}` · `deposit_check_3847{7e48bf3b0589cfad, 1500}` |
| WRONGARG | `open{savings, "Green Account (savings)"}` (msg 44) · `deposit_check_3847{0de0aa560c1cc942, 1500}` (msg 77 — user 실행) |
| MATCHED | `log_verification` · `open{checking, "Purple Account"}` (msg 32) |

(주: mutation_diff 의 msg 56 WRONGARG 는 env 반환 *"Failed to open account: Account ID '0de0aa…' may already exist."* 인 **실패 시도** — `deny_kind` 가 "Error:" 접두만 보므로 done 으로 오분류된 것. 실제 DB 변이는 1회.)

1. Purple checking 은 정답으로 개설(msg 32) — t7335(이때도 Purple 성공)와 동일.
2. **실패 지점 1 — 청취 전 개설**: 사용자의 savings 요구는 [47]에서 처음 발화되는데(주 3–4회 인출·relationship bonus·daily compounding·OON ATM 커버·≥3% APY), 에이전트는 [44]에서 이미 `open{savings, "Green Account (savings)"}` 를 실행했다. 그 직전 사용자 발화 [41]은 반대로 *"I want to do **one account at a time**, and **start with opening the Purple checking account first**"*. 요구 청취 0 상태의 무확인 write — t7335 의 Bronze 개설(추천 발화도 없이 write)과 같은 자리, 이번엔 시점이 더 이르다.
3. [47] 요구 청취 후에도 [50] *"the Green Account allows up to 8 free withdrawals per month, which should cover your frequent withdrawals"* — 주 3–4회(월 ~14회)와 월 8회의 모순을 자기 문장 안에서 못 봄. 정답 Silver Plus 재탐색 0. [56] 동일 인자 재개설 시도 → env 거절 → [60] "이미 성공"으로 정리.
4. **전진(수리 무관)**: deposit 은 이번엔 성사 — 사용자가 env 오류를 두 번 중계하며 give 를 명시 요구([71] 축자: *"the agent must first use `give_discoverable_user_tool`"*) → [72] give → [78] 입금 성공. 단 대상이 오염된 Green savings(0de0aa…)라 gold 계좌 id(7e48bf… = Silver Plus 개설이 만들 id)와 불일치 — 결정 지점 2 의 downstream.

#### trial-1 (58 msgs)
| 변이 | 내용 |
|---|---|
| MISSING | `open{checking, "Purple Account"}` · `open{savings, "Silver Plus Account"}` · `deposit_check_3847{…}` |
| WRONGARG | `open{checking, "Green Fee-Free Account"}` (24) · `open{savings, "Green Account (savings)"}` (30) · `open{savings, "Gold Account (savings)"}` (46) — 오개설 3건 |

1. **전진**: [4] `get_checking_atm_fee_totals` 발화(t7335 실패 sim 에선 0) — 전 클래스 수수료 총액 표면화.
2. **실패 지점 1**: 그 표에서 Green Fee-Free(OON $0·foreign $0)를 즉채택 — 사용자의 남은 축(FX 수수료 0·**엔/유로 보유** [33]·rebate)은 미대조. Purple(foreign $0 + FX 0% + multi-currency wallet + rebate $30/mo)이 정답. [34]에서 스스로 *"it does not support holding foreign currencies"* 를 인정하고도 교정 write 없음.
3. **실패 지점 2**: savings 를 요구 청취 전 Green(savings) 개설([30] — 게다가 가공 서사: *"'Evergreen Account' is not listed … let's consider the closest"* 라며 임의 대체), [33] 요구 청취 후 Gold(savings) **추가 개설**([46]) — 오개설 2건 적층. Silver Plus 미도달.
4. **실패 지점 3**: deposit — give 시도 0. [50] 모바일 앱 절차 안내(날조 아님·일반 안내)로 대체, [53] 사용자가 앱 입금 role-play, [55] 거래 0건 확인하고도 *"funds should appear … within that timeframe"* 으로 봉합.

**원인: 모델 주도**(요구 청취 전 선결정·다요구 축 미대조·오개설 적층) · **우리 층 부**: savings/checking **class-fit 결정론 표 부재** — t7335 처방 3(P3 후보)이 이번 수리 스택에 미포함이라 동일 결손 재발. `get_checking_atm_fee_totals` 는 ATM 비용 한 축만이라 FX·wallet·인출횟수 축이 표 밖.

### task_072 (양 trial 0.0) — ⑴ 수리 P5/P3 발화 여부

#### trial-0 (33 msgs · **too_many_errors** · action_checks 자체가 None)
변이표: gold 도달 0 (log_verification 조차 미실행 — done 0).

1. [16] `verify_identity` → [17] **VERIFIED** · [18][19] `get_current_time` 확보. 여기까지 정상 — 남은 건 log_verification 하나.
2. **실패 지점**: [20]부터 [32]까지 assistant 발화 7개 전부 *"Could you please confirm the following details …"* 반복(사용자는 [21][23][25][29][31] 다섯 번 전부 축자 확인). [26] 재검증 → [27] 재 VERIFIED 후에도 동일. 603초에 env 가 TOO_MANY_ERRORS 종료.
3. 로그(내부 재생성 구간 00:26:32~00:33:22)가 원인을 보여준다 — 모델의 원시 출력은 log_verification·계좌 조회를 **계속 시도**했으나 우리 게이트가 소거:
   - 00:27:16 `[T2_WRITE_ARG_GROUND] deny tool=log_verification inner=` — **log_verification write 가 우리 grounding 게이트에 거절**(레코드 [15]·시각 [19]가 원장에 실재하는 상태였다. inner= 공란 — 어느 인자가 걸렸는지 로그에 없음).
   - `[T2_PHASE_PRECEDE] cands=2 picked=call_discoverable_agent_tool reqs=['GB1_VERIFY_BEFORE_ACCOUNT_ACCESS']` 반복 — 계좌 조회 시도는 **GB1(검증 후 접근) 게이트**가 pure-advice 로 치환. log_verification 이 안 됐으니 GB1 은 계속 닫혀 있다.
   - `[T2_CLAIMPROV] tool-miss fallback: kind='record_update' tool='log_verification' 원장 밖 — kind 색인으로 강등` 반복(**P1 수리는 발화** — unbacked 즉결은 면함) → 그러나 `[T2_CLAIMPROV] regen tool_calls=[]` 가 재생성 출력에서 **tool call 을 비운 채** 확정 — 회복 행동 자체가 소거.
   - `[T2_UNAVAIL] promised tools not available: ['call_discoverable_agent_tool(get_all_user_accounts_by_user_id_3847)']` — 데드락의 자기 기록.
4. 즉 **상호 데드락**: log_verification ← WRITE_ARG_GROUND/regen 소거, 계좌 접근 ← GB1 차단(전제=log_verification), 모델 잔여 행동 = 사용자 재확인 요구 루프. 초반 [2]~[11]의 discovery 강제(T2_PROV 가 날조 `account_id=Bluest-Account-12345` 차단 등)는 정상 작동이었다.

**원인: 우리 층 주도**(게이트 상호 데드락 + regen 이 유효 발화를 못 만든 채 에러 카운터 소진 — WRITE_ARG_GROUND 의 log_verification deny 사유는 미상·격리 재현 필요) · 모델 부(1차 VERIFIED[17] 직후 write 대신 재확인 SAY 로 이탈한 첫 수는 모델).

#### trial-1 (83 msgs)
| 변이 | 내용 |
|---|---|
| MISSING | `apply_checking_account_credit_5829{chk_lj82d4f1a9, 14, fee_refund}` · `{chk_538bfb9cba, 3.5, fee_refund}` |
| MATCHED | `log_verification` (msg 14 — trial-0 이 못 한 것을 즉시 통과: WRITE_ARG_GROUND 상시 오작동은 아님) |

1. 검증·log 까지 정상([4]~[15]). 이후 **계좌 목록으로 못 감**: `get_credit_card_transactions_by_user`(빈 결과)·`get_user_information_by_id` ×5(DUPLICATE-READ 가드 8회)·`shell grep`(KB 코퍼스에서 DB id 검색·No matches) 순환. trial-0 은 [9]에서 `get_all_user_accounts_by_user_id_3847` 를 unlock 했었다 — 같은 모델이 이 trial 에선 그 이름을 끝내 못 찾음(KB 검색어도 안 던짐).
2. **P3 발화·순기능 실측** [54]→[55]: 날조 transactions(`tx_12345`/`tx_67890`)와 카드 id 로 `get_atm_fee_discrepancies` 호출 → **READ-FIRST deny**(축자): *"this audit only judges fee lines that were READ in this conversation, and the required transaction read is missing: get_bank_account_transactions … call it with the checking account's id **copied from the accounts listing**"* — GIGO 차단 성공.
3. 그러나 deny 는 "accounts listing" 을 **만드는 도구명**(get_all_user_accounts_by_user_id_3847)은 명명하지 않았고, 모델은 listing 없이 계좌명·날조명을 id 로 시도([60] `account_id="Sky Blue"` — 존재하지도 않는 이름 · [71][77] `"Bluest Account"`) → env 정당 거절 → [79]~[82] transfer 로 종료.
4. P5(문구)는 **시험 미도달** — 정당 인자의 audit 호출 자체가 없었다. t7335 의 실패 지점($12 vs $14)보다 상류에서 죽음.

**원인: 모델 주도**(accounts-listing 도구 미발견 + id 날조) · **우리 층 부**([[64]] 부분 결함 — READ-FIRST 본문이 transaction-read 도구는 명명하되 그 전제인 listing 도구는 미명명).

### task_073 (양 trial 0.0) — ⑵ DUP 재발 여부

#### trial-0 (92 msgs)
| 변이 | 내용 |
|---|---|
| WRONGARG ×6 | chk_1 에 3.00/5.00/1.50 **세 건**(msg 68) · chk_2 에 3.00 **세 건**(msg 76/78/80) — 라인별 분할 credit |
| MISSING | `{chk_1, 9.5}` · `{chk_2, 9}` (net 단일 credit 형태가 없음) |
| MATCHED | `log_verification` · `{chk_3, 1.5}` (msg 84 — 단일 라인이라 분할=net) |
| DUP | **0 건** |

1. 발견·검증·3계좌 audit 전부 정상. 엔진 판정(각 계좌 discrepant 라인·금액)은 gold 산식과 일치: chk_1 = 3+5+1.5, chk_2 = 3+3+3, chk_3 = 1.5.
2. **P5 새 문구가 노출된 첫 실전**. [58] 반환문 전문 끝에 net 지시가 **남아 있다**(축자): *"If corrections are owed, the credit policy requires ONE fee_refund credit for the net correction of THIS account (do not credit the same lines twice)."* — 즉 P5 는 net 지시를 지운 게 아니라 구판의 *"across all identified fee discrepancies"* 구절만 제거했다.
3. **실패 지점**: [59] 모델이 net 합계 없이 라인별 차액만 사용자에게 제시 → [60] user-sim 이 그 표현을 그대로 에코(축자): *"apply credits for all the incorrect charges … (the **$3.00, $5.00, and $1.50** differences)"* → [68] 라인별 3건 분할 실행. chk_2 도 동일([76]~[80]). 최종 잔액 합은 gold 와 동일(5209.50/12759.00/892.00)이나 **거래행이 계좌당 3건 vs gold 1건** — DB 해시 불일치로 0점.
4. 대조: t7335(구 문구 "across all identified …" 존치)에선 같은 모델이 **9.50/9.00/1.50 net 단일 credit** 을 정확히 실행했고(1차는 만점 상태), t7336 양 trial 은 net 실행이 0. 실패 축이 문구 개정의 유일한 의미 손실(합산 범위 명시 삭제)과 정확히 겹친다 — 표본은 작으나 [[70]] 의 "레버는 하나를 사면 하나를 판다" 실물: **P5 의 매입(넷팅 약화)은 실측, 매출(072 rebate 스캔)은 미실현**(072 가 audit 에 미도달).
5. DUP·재적용은 없음. [90] 가공 혼란 서사 1건은 무해하게 종료. P4(operator-fab deny)는 이 sim 에서 5회 발화 — [66] 날조 도구 `apply_fee_corrections_9876` 시도를 스스로 교정하고 정본 credit 도구에 도달(순기능 정황).

#### trial-1 (54 msgs)
| 변이 | 내용 |
|---|---|
| MISSING ×3 | 3계좌 credit 전부 (audit 까지는 정상·gold 값 산출 완료 [43]~[45]) |

1. [49] 사용자: *"please go ahead and apply credits for **all** the incorrect charges"* → **실패 지점** [50](축자): *"It appears that **there is no specific tool available** to directly apply the credits … escalating this issue to a human agent"* — trial-0 이 unlock 해 쓴 `apply_checking_account_credit_5829` 를 이 trial 은 검색조차 안 함.
2. [52] 한술 더: *"I will unlock the `file_credit_card_transaction_dispute` tool"* — 체킹 ATM 건에 신용카드 분쟁 오제안 → [53] 사용자 격노·transfer 종료.

**원인(073 종합): 모델 주도**(#0 net 지시 위반·#1 write 도구 미발견) · **우리 층 부**(#0 — P5 문구 개정이 합산 범위 명시를 제거해 net 실행의 발판을 약화 + 엔진이 이미 계산한 `delta_total` 미표기 유지) · user-sim 부(#0 라인별 금액 에코 — [[21]] 상 agent 흡수 대상).

### task_093 (양 trial 0.0) — ⑶ SG_DOCS·components 이후 write 인자 추적

정답 산식(t7335 포렌식 확정): expected **4.275** = 4.0(잔액 $144k 상위 tier) + 0.25(Green→Silver boost) + 0.025(relationship) · actual **4.0** = 480×12/144000 · 차액 **$33.00**.

#### trial-0 (61 msgs)
| 변이 | 내용 |
|---|---|
| WRONGARG | `submit_interest_discrepancy_report_7294{expected 2.75, actual 2.5, diff 30}` (msg 45) |
| MISSING | `apply_savings_account_credit_6831{33}` · 정답 보고서 |

1. [27]~[34] 계좌 목록($144,000 Silver + Green checking)·거래($480 MONTHLY INTEREST CREDIT) 전부 문맥 확보. [38] dense 1위는 여전히 **business** "Silver Plus Saver" 문서(t7335 와 동일 함정).
2. **전진**: [39] `get_correct_savings_apy{components: [base 4.0], customer_products: "Green Account (checking)"}` — t7335(#0 `"No credit card accounts"`)와 달리 **Green checking 이 입력에 실림** → `T2_SG_DOCS` 가 silver_account+green_account_(checking) **12편 13619자** 주입(t7335 는 9편) → 서브가 boost 0.25 를 처음으로 반영.
3. **실패 지점 1(입력 불완전 잔존)**: components/customer_products 에 **잔액($144,000)·tier 재료가 없다** → 서브는 하위 tier 2.5 채택 → 반환 2.75 (= 2.5+0.25 · relationship 0.025 도 미반영). 모델의 base 4.0 주장은 business 문서 출처라 grounding 에서 걸러진 것으로 보임 — GIGO 는 줄었으나 tier 축이 여전히 입력 우주 밖.
4. **실패 지점 2(actual 미유도 재발)**: [41] `get_interest_correction{expected 2.75, actual 2.5, …}` — A2 지시(credit×12/principal = 4.0%) 무시, 하위-tier 문구의 편재값 2.5 를 기입. `T2_SG_GROUND` 는 period_start 만 드롭(경고 [42]) — 2.5 는 문서에 실재하는 값이라 통과(t7335 §093 과 동일한 원리적 구멍). 엔진 diff = 30.0.
5. [45] 보고서 {2.75, 2.5, 30} 제출(WRONGARG — t7335 의 무행동보단 전진). [48] 사용자가 credit 여부를 물었으나 [49]~[55] transfer 문서 탐색으로 이탈, credit 미적용, [57] transfer(사유 또 오기: `account_ownership_dispute`).

#### trial-1 (73 msgs)
| 변이 | 내용 |
|---|---|
| MISSING ×2 | 보고서·credit 전부 (write 0) |

1. 상류 동일([35] 같은 components·2.75 반환). **실패 지점**: [37] `get_interest_correction{expected 2.75, actual **4.00**}` — 이번엔 자기 문서 독해의 4.0 을 **actual** 에 기입(공교롭게 4.0 은 참 actual 과 일치 — expected 가 4.275 였다면 +33 정답이 나왔을 자리). 엔진 반환 **-150.0**(음수 보정).
2. 음수를 본 모델이 결과 해석을 포기하고 transfer 서커스: `initial_transfer_to_human_agent_0218`("lines are busy" 반환·무관한 신용카드 프로토콜 문서 유래)·`_1822` 를 [41]~[66]에서 **6회** unlock/호출 반복 후 [67][69] `transfer_to_human_agents` 2연발(사유 2종 모두 오기).

**원인: 모델 주도**(tier 재료 미기입·actual 미유도/오기입·음수 결과에서 재검토 대신 도피) · **우리 층 부**: (i) t7335 처방 5(actual_apy 파생 검산 — 480·144000 이 원장에 있으면 credit×12/principal 불일치 플래그)와 입력-완전성 대조(잔액 미포함 플래그)가 **이번 수리 스택에 미구현** — 실패 축이 그대로 남음. (ii) 편재값 그라운딩(문서 실재값 2.5 통과)은 기지 한계.

### task_094 (양 trial 0.0) — ⑴ 수리 P2 발화 여부

정답 산식: expected **6.85** = 5.5(Gold base) + 0.75(Green→Gold) + 0.6(EcoCard 최고 카드보너스) · actual **5.1** = 408×12/96000 · 차액 **$140**.

#### 공통 상류 (양 trial 동일)
1. [1] 사용자 주장: *"The base rate is **5.0%** and my checking account gives me a **1% boost** … about **$96,000** … only received **$408**"* (시나리오: 확신 없음·교정하면 따름).
2. **실패 지점 1(미조회 formalize 재발·t7335 동일)**: [14] 검증 직후 — 계좌·카드·거래 **read 0** 상태로 `get_correct_savings_apy` 호출. components = 사용자 주장 재진술(base 5.0·checking 1.0), customer_products 도 주장 요약. `T2_SG_DOCS` 12편 주입 → 서브 **5.5%** 반환(카드 3장 +0.6 과 Green 0.75 는 입력에 없어 미반영 — 카드는 이 sim 들에서 **끝까지 한 번도 조회 안 됨**. get_credit_card_accounts_by_user 호출 0).
3. **실패 지점 2(user-주장 그라운딩 — P2 의 인접 구멍)**: [20] `get_interest_correction{expected 5.5, actual **5.0**, principal 96000, …}` — actual 을 파생(5.1)하지 않고 사용자의 "5.0%" 를 기입. `[GROUNDING WARNING]` 은 **period 날짜 2건만** 드롭([21]) — 5.0/96000 은 사용자 발화에 실재해 1차에 통과. 엔진 diff = **40**.
   - **P2 판정**: t7335 의 에코 경로(1차 드롭→경고문 에코로 2차 통과)는 이번 런에서 **재현 자체가 안 됨**(드롭될 값이 처음부터 user corpus 로 통과) — 에코 재발 0 은 사실이나 **P2 의 인과 확인은 미도달**이고, 대신 그 옆 구멍(요율 값의 사용자-주장 그라운딩)이 실측됐다. t7335 §094 의 "actual_apy 파생 검산" 처방이 미구현으로 남은 자리다.
4. [24] `submit…{account_id: "wl94k7m3p8"}` — user_id 를 계좌 id 로 기입 → env *"Error: Account 'wl94k7m3p8' not found."* (정당 차단·BLOCKED·t7335 와 동일 혼동).

#### trial-0 (42 msgs) — 차단 후 날조 봉합
5. **실패 지점 3**: env 차단 후 계좌 조회로 가지 않고 [30](축자) *"Given the persistent issue with unlocking the tool, **I will manually inform the backend team** about the discrepancy"* → [32]~[36] 존재하지 않는 "수동 처리"를 기정사실화하고 처리 기한(*"within 1-2 business days"*)까지 약속 — **write 0 인 채 완료 서사 날조**. [41] 사용자가 카드 3장(Platinum·Gold Rewards·EcoCard)을 뒤늦게 자백한 직후 시나리오 종료. MISSING ×2.

#### trial-1 (42 msgs) — 차단 후 복구·값은 stale
5'. **전진**: 같은 차단 후 [28]~[31] `get_all_user_accounts` 로 복구, `sav_wl94k7m3p8_gold` 확보 → [32] 보고서 {5.5, 5.0, 40} 제출 · [38] credit $40 적용 (WRONGARG ×2 — gold {6.85, 5.1, 140}). [35] 사용자가 *"any other boosts I might qualify for?"* 를 물었는데 카드 조회 없이 *"없다"* 로 단정 — EcoCard +0.6 미반영 확정.

**원인: 모델 주도**(미조회 formalize·user 주장 5.0 의 actual 전사·카드 축 무시·#0 완료 서사 날조) · **우리 층 부**: (i) actual_apy 파생 검산·customer_products 원장 대조 미구현(t7335 처방 잔존) (ii) 요율 인자의 grounding corpus 에 user 발화 포함 — `_write_arg_ground_deny` 의 `corpus_roles` 주석이 이미 원칙을 선언해 둠(축자: *"정책 주장(카드 자격·요율)=문서만"*)인데 SG_GROUND 쪽 요율 operand 에 미적용. env 차단 2회는 정당.

## 3. 레버 발화율 표 (실패 11 sim + 참고 100)

| 레버 | 017#1 | 055#0 | 055#1 | 072#0 | 072#1 | 073#0 | 073#1 | 093#0 | 093#1 | 094#0 | 094#1 | (100 t0/t1) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `T2_SG_DOCS` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 2 | 1 | 0/0 |
| `T2_SG_GROUND` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 3 | 1 | 0/0 |
| `T2_SG_ISOLATE` | 0 | 0 | 0 | 0 | 0 | 20 | 12 | 6 | 6 | 13 | 6 | 0/0 |
| `T2_PIN_READ` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 9 | 0 | 6/6 |
| `T2_DEMANDED_STEP` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 6/8 |
| `T2_CLAIMPROV regen` | 2 | 4 | 4 | 4 | 2 | 3 | 3 | 4 | 4 | 8 | 3 | 1/2 |
| `READ-FIRST`(P3·궤적) | 0 | 0 | 0 | 0 | **1** | 0 | 0 | 0 | 0 | 0 | 0 | 0/0 |
| `GROUNDING WARNING`(궤적) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 1 | 1 | 0/0 |
| `DUPLICATE-READ`(궤적) | 0 | 0 | 0 | 0 | 8 | 1 | 0 | 1 | 0 | 0 | 0 | 0/0 |
| FAB(`RESOLVE operator-fab`) | 1 | 2 | 2 | 0 | 0 | 5 | 2 | 0 | 0 | 1 | 0 | 0/0 |
| `SEARCH_AGENT` 침묵(축 소진) | 4 | 0 | 0 | 6 | 9 | 11 | 16 | 8 | 10 | 16 | 5 | 0/0 |
| `WRITE_ARG_GROUND` deny | 0(#0 은 2) | 0 | 0 | **1(log_verification)** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0/0 |

- **SEARCH_AGENT 침묵은 死배선 아님**([[67]] 0단계 확인): 각 sim 초반에 그룹 배달 실행(로그 `group=checking_accounts · 문서 113 …` 등·"일반 자리 배달 1회째" 8회)·침묵은 축-소진 후 정상 거동. 단 072#1 의 곤경(계좌 **id** 필요)은 KB 문서 축이 아니라 DB-read 도구 발견 축이라 이 레버의 관할 밖이었다.
- `T2_DEMANDED_STEP` 은 실패 sim 전부 0 (t7335 의 055 오표적 5회조차 없음) — 성공 sim(100)에서만 발화.
- 발화-후-무력 형: **072#0 이 유일**(WRITE_ARG_GROUND deny + CLAIMPROV regen 이 회복 경로를 소거 — §072). 나머지 실패 결정 지점 대부분은 t7335 와 동일하게 "그 자리에 레버가 없다"(055 class-fit·093/094 검산·072#1/073#1 도구-발견).

## 4. 원인 4주체 귀속 · 수리 5종 성적 · 처방 후보

### 태스크별 원인 1줄
| sim | 변이 | 주 | 부 |
|---|---|---|---|
| 017#1 | MISSING 전부 | 모델: NOT_VERIFIED 후 조회 키 요청에서 'name' 누락(줄 수 없는 email/user_id 만 반복 요구) — trial-0 [30] "full name" 한 마디가 반증 대조 | — |
| 055#0 | Silver Plus→Green(savings)+deposit 오계좌 | 모델: 요구 청취 **전** savings 개설·월8회 제한 모순 묵살 | 우리 층: class-fit 표 부재(처방 미이행) |
| 055#1 | 오개설 3건+deposit 미지급 | 모델: fee-totals 표 즉채택(FX·wallet 축 미대조)·오개설 적층 | 상동 |
| 072#0 | done 0·too_many_errors | **우리 층**: WRITE_ARG_GROUND 의 log_verification deny + CLAIMPROV regen(tool_calls=[]) + GB1 phase 게이트의 상호 데드락 | 모델: VERIFIED 후 재확인 SAY 루프 개시 |
| 072#1 | credit ×2 MISSING | 모델: accounts-listing 도구 미발견·id 날조("Sky Blue") | 우리 층: READ-FIRST 가 listing 도구명 미명명([[64]] 부분) |
| 073#0 | 분할 credit ×6 (DUP 0) | 모델: "ONE net credit" 지시 노출 상태에서 라인별 실행(user 에코 동조) | 우리 층: P5 개정이 "across all identified …" 합산 범위 명시 삭제·delta_total 미표기 |
| 073#1 | credit ×3 MISSING | 모델: credit 도구 "없다" 단정→카드 분쟁 오제안→transfer | — |
| 093#0 | 보고서 {2.75,2.5,30}·credit 0 | 모델: 잔액/tier 미기입(2.5 하위 tier)·actual 미유도(편재값 2.5) | 우리 층: 파생 검산·입력-완전성 대조 미구현(처방 잔존) |
| 093#1 | write 0 | 모델: actual 에 4.0 오기입→음수 보정→transfer 서커스 | 상동 |
| 094#0 | write 0 + BLOCKED | 모델: 미조회 formalize·env 차단 후 "수동 처리" **완료 서사 날조** | 우리 층: 요율 인자에 user-주장 그라운딩 허용·파생 검산 부재 |
| 094#1 | {5.5,5.0,40} stale write | 모델: 상동(차단 복구는 함)·카드 무조회 채 "다른 boost 없다" 단정 | 상동 |

### 수리 5종 성적 ([[70]] 부호표)
| 수리 | 발화 | 매출(의도 효과) | 매입(부작용) |
|---|---|---|---|
| P1 CLAIMPROV kind-fallback | ○ (072#0 로그 다수) | 050형 unbacked 오판정·DUP 재발 0 | 072#0 에서 강등 후에도 regen 이 tool_calls=[] 산출 — pending 루프는 잔존 |
| P2 경고문 에코 제거 | (경로 미재현) | 에코 재발 0 — 단 **인과 미확인**(1차 통과라 시험 미도달) | 없음. 인접 구멍(user-주장 그라운딩) 실측 |
| P3 requires_reads(READ-FIRST) | ○ (072#1 [55]) | 날조-transactions audit 차단 1건 실측 | listing 도구명 미명명으로 회복 유도는 실패 |
| P4 FAB fix-naming | ○ (operator-fab 11회) | 073#0 날조 도구명에서 정본 도구로 자기 교정(정황) | 미관측 |
| P5 완결 문구 제거 | ○ (073 양 trial 노출) | **미실현**(072 가 audit 미도달) | **073#0 넷팅 붕괴 실측**(t7335 구 문구에선 net 정확) |

### 처방 후보 (전부 제안만 — [[62]] 순서: 각각 격리로 재고 나서)
1. **[R1·우리 층 버그·최우선]** 072#0 데드락 해부: ①WRITE_ARG_GROUND 가 원장-실재 인자의 log_verification 을 deny 한 사유 격리 재현(레코드 [15]+시각 [19] 실재 — 어떤 arg/marker 가 걸렸는지 로그에 없다·`inner=` 공란 인쇄도 계기 결함) ②CLAIMPROV regen 이 tool_calls=[] 를 확정 산출할 때의 탈출 규칙(같은 write 가 N회 소거되면 deny 본문을 대화에 표면화 — [[64]] fix-naming) ③GB1 phase 게이트와 write-deny 의 순환 검출(둘이 서로를 전제로 닫혀 있으면 한쪽을 문면 경고로 강등).
2. **[R2·A2 문구·[[70]] ②형]** P5 재절충: 완결-인상 없이 합산 지시 복원 — "sum the differences listed above into ONE net credit per account" 축자 + 엔진이 이미 계산한 **`delta_total` 표면화**(FIX-5 때 준비된 값·신설 로직 0). 073#0 이 산 증거.
3. **[R3·검산형·t7335 처방 재상정]** `actual_apy` 파생 검산(원장에 interest_credit·principal 실재 시 credit×12/principal 와 불일치 플래그 — 추출 아닌 검산·[[03b]]) + 요율 operand 의 grounding corpus 에서 **user 발화 제외**(`corpus_roles` 원칙 축자 "정책 주장=문서만" 을 SG_GROUND 요율 축에 적용). 093/094 네 sim 전부의 공통 상류.
4. **[R4·존재 대조·재상정]** `get_correct_savings_apy` 입력-완전성: 이 고객의 계좌/카드 read 이력이 원장에 없으면 문면 플래그("no account records have been read for this user yet") + 잔액(tier 재료) 미포함 플래그. 094 미조회 formalize·093 tier 결손의 공통 상류.
5. **[R5·[[64]] 보강]** READ-FIRST deny 에 accounts-listing 도구명(`get_all_user_accounts_by_user_id_3847`) 포함 — P4 와 같은 `arg_source_reads` 기계 도출로(엔진 리터럴 0).
6. **[R6·A2 1회 저작·재상정]** savings/checking **class-fit 표**(055 네 런 연속 실패 — FX·wallet·인출횟수·compounding·rebate 축의 문서화 스펙 표). `get_checking_atm_fee_totals` 단독으론 축이 모자람이 055#1 로 실측됨.
7. [R7·저순위] NOT_VERIFIED 본문은 이미 by_name 을 명명 — 017#1 은 모델 분산으로 두고 레버 신설 없음(과폭 회피·[[66]]).
