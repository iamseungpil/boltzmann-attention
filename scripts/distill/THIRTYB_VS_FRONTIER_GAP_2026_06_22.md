# 32B vs frontier 갭 — 전수 궤적 비교 (2026-06-22·사용자 지시)

> 사용자: "32B면 fetch-first 통과하니 상용=32B. 문제는 32B→frontier 능력 갭." → 전수비교.
> 데이터: `on_n32int8_floor_retail`(32B-int8·3 trial)·`retail_gpt41_nogate`(gpt-4.1 agent·1 trial). 둘 다 no-gate=순수 모델능력.

## §1. pass (★갭의 정체 = 신뢰성)
- 32B pass^1 ≈ 0.60(§35) · **32B pass-any-of-3 = 0.77** · gpt-4.1 단일시행 = **0.82**.
- ⇒ **32B 최선-of-3 ≈ frontier 첫시행.** 갭의 큰 부분 = *raw 능력*이 아니라 **신뢰성/일관성**(32B는 풀 수 있으나 첫시행 실패·재시도 필요).

## §2. GAP = frontier 풀고 32B 3시행 전부 실패 = 15 task (진짜 frontier 능력)
전수 분류(15 task·write-action 비교):
| 유형 | 수 | 정체 |
|---|---|---|
| **DIFF-ACTION** | **8 (53%)** | 틀린/불완전 행동·루프·포기 = **flow 다단계 완수 + 복구(P7)** |
| **SAME-ACTION(operand)** | 6 (40%) | 올바른 행동·틀린 인자 = variant 선택(B1)+verbatim 복사 |
| no-write | 1 (7%) | 시도 안 함 |

## §3. 궤적 구체 (직접 확인)
- **T8 (operand/B1)**: 32B new_item_ids=1270145486→4385534692(틀린 variant) vs FR 9083642334(정답). = 의미적 variant 선택. ← B1-select 타깃.
- **T17 (operand/copy-fidelity)**: 32B "123 Elm **St**" vs FR "123 Elm **Street**". = user 문자열 verbatim 복사(32B 축약). ← operand-formalize 정밀도.
- **T38 (flow/recovery)**: 32B modify_items(new_ids=[]) **4회 루프** vs FR **cancel_pending_order**(올바른 의도). = 의도파악+루프탈출.
- **T34/T37 (flow/포기)**: 32B modify 루프 후 **cancel로 fallback**(포기) vs FR modify 완수. = 다단계 완수 못 하면 도망.
- **T41 (flow/불완전)**: 32B 2 write vs FR 4 write. = 멀티-액션 누락(불완전).

## §4. 결론 — 32B→frontier 갭 = (a)flow완수·복구 > (b)operand
1. **★최대 = flow 다단계 완수 + 복구(53%)**: frontier는 올바른 시퀀스를 *끝까지 수행*·실패시 *다르게 재시도*. 32B는 *루프*하거나 *cancel로 포기*. = §35 "7B→32B 갭=DB-state/flow/복구"가 *32B→frontier에도 잔존*(작아졌으나).
2. **operand 정밀도(40%)**: variant 선택(B1·의미)+verbatim 복사. ← B1-select가 겨눈 부분(40%만).
3. **신뢰성**: 32B≈frontier@best-of-3 → 갭은 *첫시행 완수율*. flow-복구·operand 둘 다 여기 기여.

## §5. 함의 (연구 타깃 재정렬)
- **B1-select(진행중)은 갭의 40%(operand)만 커버.** 더 큰 레버(53%)=**flow 완수+복구**.
- ⇒ 32B를 frontier로 올리는 cheap-replication 타깃 = **(1)복구/완수 controller**(루프탈출·포기금지·다르게재시도=P7·이전 C8) + **(2)operand B1(variant선택+verbatim)**. flow-복구가 우선.
- 복구는 [[42]]/딥리서치가 retry-controller/reflection/최소LoRA로 지목·§35c C8서 "retry=잘못된 레버"였으나 그건 *grounding* 맥락·여기선 *완수/포기* 맥락 → 재검토 가치.
- REVERSE(32B 풀고 frontier 실패)=10 task(frontier 약점·user-sim 노이즈 가능·부차).

## §6b. ★Opus-4.8 직접 풀이 검증 (사용자: "당신이 frontier니 직접") — T105
- 대상 = **32B(3시행)·gpt-4.1 둘 다 실패**한 both-fail 11개 중 T105(1-action·top-frontier-only). 무결성: 시나리오+실 variant 카탈로그만 보고(gold 비공개) Opus가 풀이 → 사후 gold 대조.
- **task**: 두 Tea Kettle(glass/2L/induction)을 ①{ceramic,gas} ②{1.5L,gas}로 교환. **부분-스펙 규칙**(명시 안 한 속성=현재값 유지): ①=ceramic·**2L유지**·gas=`3761330360` ②=**glass유지**·1.5L·gas=`9647374798`.
- **Opus 답 = GOLD 정확 일치** `exchange(item=[7292993796×2], new=[3761330360, 9647374798], gift_card_7245904)` 1회. ✅
- **★why gpt-4.1 실패(진단)**: gpt-4.1 *첫* 교환 new=[3761330360, 9647374798] = **gold와 동일(operand 정답!)**. 그런데 *둘째* 틀린 교환[4238115171,3909406921] 추가 → 최종상태 오염·reward0. = **operand 아닌 flow 규율 실패**(정답 후 멈춤 못 함·재시도 폭주).
- ⇒ **top-frontier(Opus) 우위 = "정답 한 번 내고 STOP"(flow 규율)**. 중상위(gpt-4.1)도 operand는 맞히나 flow서 무너짐. = §2/§4 "갭의 최대=flow완수+복구(53%)" *직접 실증*·operand(40%)는 둘 다 통과 가능.
- 함의 강화: 32B→top-frontier cheap-replication 1순위 = **flow 규율/완수/멈춤**(redundant-redo 금지·루프탈출) > operand. = C8 recovery 재부상(단 "완수/멈춤" 프레임).

## §6c. Opus-4.8 both-fail 11개 풀이 (2026-06-22 PM·★결론1=오라클-프록시·실 e2e pass 아님·⚠️정정)
방법: `frontier_solve_kit.py --task N --brief`(시나리오+실 카탈로그·gold 숨김) → Opus 추론 → `--check`(gold 대조). 32B/gpt-4.1 = `results.json` write-action+termination 추출(`/tmp/extract.sh`).

### ⚠️ 결론 1 (정정·과대주장 교정) = **오라클-프록시서 write 결정가능 11/11**. 실 e2e pass 아님.
- ★**이 "풀이"=오프라인 오라클 프록시**(brief가 시나리오+user/orders/**variant 카탈로그 pre-fetch**까지 제공·`--check`=write **집합** 대조·communicate-check 미채점·`_norm` list정렬로 pairing 무시). = [[04-current-position]] 경고 "오프라인 op-eval≠멀티턴 τ²". **leaderboard 0.91/0.92(Opus4.6 실 멀티턴 pass^1 전 태스크 평균)와 동치 아님.**
- 프록시가 우회한 실-e2e 난이도: ①멀티턴 elicitation(user-sim이 점진공개·"private") ②read 도구 자가호출(카탈로그 발견) ③**communicate/NL-assertion**(★T2/3/4 실패의 정체="가용 옵션 *카운트*"=프록시가 *미채점*·즉 그 셋은 프록시서도 write만 맞춘 *부분*해결) ④전 대화서 틀린 write 0 + pass^1 + user-sim 노이즈.
- ✅ **유효한 좁은 주장**: writes-only gold(99,100,103,109,110)=`SOLVED ✓`·full-trajectory gold(2,3,4,20,21)=write 집합 일치 → **올바른 write 시퀀스가 데이터서 결정가능** = 11개 *genuinely unsolvable/mislabeled 아님*(narrow). **단 "Opus가 실 e2e서 11/11 pass"는 미검증·1.00 비현실**(0.92 천장·info/communicate가 사각).
- ✅ **변치 않는 부분**: §아래 32B/gpt-4.1 실패유형 분류는 *실 e2e 궤적*(`results.json`) 기반 = 유효.
- ★진짜 검증 = **Opus-4.8을 *agent*로 실 tau2 멀티턴 e2e**(gpt-4.1 user-sim·실 reward incl communicate·오라클 없음)로 이 11개 돌리기. (COST GUARD: agent=Claude 15-30x.)

### 전수 분석표 (gpt-4.1 1trial·32B 3trial·gold write 대조)
| T | gold write | Opus | gpt-4.1 실패 정체(full-args 확인) | 32B 실패 정체 | 1차 갭축 |
|---|---|---|---|---|---|
| 2 | return×1(3 items) | ✓ | **선(先) 2-item return → 후 3-item return = 중복쓰기**(상태오염)+count정보 | t1 정답쓰기인데 reward0(count정보)·t2 vacuum누락 | info-count + flow-중복 |
| 3 | modify(small만) | ✓ | **write=gold 완전동일**·reward0 = **info-count(가용 옵션 수)** | XXL까지 포함(필터실패)/count | **info-count** |
| 4 | modify×2 | ✓ | **write=gold 동일**·reward0 = **info-count** | write=gold 동일·info-count | **info-count** |
| 20 | modify(4 items·max-price) | ✓ | 4중 3 정답·Makeup=2등값 5012998807($258.71) vs gold max 2882812427($261.11) = **operand 랭킹** | **processed주문 수정(eligibility위반)**+item-id환각+중복 | operand + eligibility(32B) |
| 21 | modify(shoe+kbd 한 콜) | ✓ | 1 item만(shoe 결합 누락) = **flow 결합실패** | **product-id를 item-id로**(1656367028)=id혼동/포기 | flow-결합 + id혼동(32B) |
| 99 | exchange×2(cancel 안 함) | ✓ | **전체주문 cancel**(skateboard 한 품목 위해) = **거부규율 위반(과충족)** | 동일·전체 cancel | **거부규율(과충족)** |
| 100 | modify+return | ✓ | **여분 modify_payment**(미요청)+ = 과행동 | **중복 modify**(2회·재수정 불가)+ | flow 과행동/중복 |
| 103 | 4 writes | ✓ | **write=gold 4개 동일**·reward0 = **info(취소주문 tracking#)** | return 1개 누락(완수실패)/중복 | info(gpt) + 완수(32B) |
| 105 | exchange×1 | ✓ | 정답 exchange **후 둘째 틀린 exchange = 중복 redo** | under-item/modify_user 환각 배회 | **flow 과행동(redundant redo)** |
| 109 | addr×2+modify | ✓ | flow/grounding 정답(어느주문·주소조회)·tablet=4913411651($941) vs gold 최저가 2106335193($903.95) = **operand 최저가** | **틀린 주문+방향역전**(old760 설정)+pending에 exchange | operand(gpt) + grounding방향(32B) |
| 110 | addr×2+modify | ✓ | user주소 정답·**틀린 주문 주소변경**(#W1603792 not #W1092119) = **어느주문이 wrong 판별실패** | 틀린주문/pending에 exchange | **flow 판별(which-order-wrong)** |

### ★결론 2 = gpt-4.1(중상위)의 잔여 갭 = **flow-규율 ≈ info/communicate > operand** (Opus 대비)
- **flow-규율 ~5** (T2중복·T21결합·T99과충족·T100과행동·T105 redundant-redo·T110 판별): "정확히 맞는 write를 *한 번만* 내고 STOP"·"어느 대상이 wrong인지 판별"·"요청을 *정확히* 충족(과/미달 금지)". = §4 "flow완수+복구(53%)" *재확증*·여기선 **과행동(over-action)** 형태가 지배(루프보다).
- **★info/communicate ~4 (신규축)** (T2·T3·T4 가용옵션 *세기*·T103 tracking# 보고): **write가 gold와 글자단위 동일한데 reward0** = 행동 아닌 **파생사실 산출·보고**(가용 variant 카운트=두 tshirt 상품 전부 열람 필요·취소주문 추적번호) 실패. = 기존 action-중심 census(§2)가 *놓친* 갭축. operand/flow와 별개의 결정론-친화 레버(카운트=엔진 집계).
- **operand ~2** (T20 max·T109 cheapest = 랭킹선택 off-by-one). = B1-select가 겨눈 부분이지만 *이 both-fail 셋에선 최소축*(§2 "40%"는 과대). variant 의미선택이 아니라 **순서통계(max/min)** = [[10]] B2(결정론 resolve) 영역이지 B1(LLM) 아님!

### ★결론 3 = 32B 갭 = **신뢰성/광범위** (gpt-4.1보다 깊고 잡다)
- id환각(product↔item), eligibility위반(processed 수정), 잘못된 도구(pending에 exchange), grounding 방향역전(old/new), 중복쓰기, 포기(no-write). = §1 "신뢰성" 정합. 32B는 flow-controller *와* operand/grounding 둘 다 필요.

### ★cheap-replication 타깃 확정 (이 분석의 산출)
1. **flow-규율 controller (1순위·결정론 친화)**: ①commit-once(중복 write 차단·T2/100/105) ②요청-충족 판정 후 STOP(과충족 차단·T99) ③which-target-is-wrong 판별 지원(T110). = §6b "정답 후 STOP" + over-action 차단. learn보다 scaffold-gate 공산(중복/eligibility=결정론 판정).
2. **info/communicate 집계 (신규·결정론 친화)**: "가용 옵션 수"·"tracking#" = 카탈로그/주문 *집계*를 엔진이 산출→모델이 보고. autofetch 사촌(fetch가 아니라 aggregate-and-report).
3. **operand 최소(B2-resolve)**: max/min 랭킹선택 = 결정론 resolve(B2·[[10]])지 B1-learn 아님. ⇒ **B1-select learn 우선순위 하향**(이 셋에선 의미선택 거의 없음·전부 순서통계/부분스펙).
- ⇒ **3축 모두 결정론/scaffold-gate 친화** = 둘째기둥(scale 능력 cheap-replication)이 learn보다 **engine-side**로 기운다는 강한 신호([[06-NOW]] §C4 "autofetch=유일 작동 도메인일반 레버" 정합 확장).

## §6. 다음
1. flow-규율 controller 설계(commit-once·STOP-when-satisfied·결정론 gate vs learn 비교) = 최대레버.
2. info-aggregate 레버(가용옵션 카운트 엔진) PoC.
3. B1-select 결과 회수하되 *우선순위 하향*(operand=이 셋에선 B2-resolve·최소축).

**불변**: §35·[[42]]·`FETCH_SELECT_DIVISION`(operand=B). 상위 `RULE_LEVER_COST_EFFICIENCY_PROGRAM`(C8 recovery·C10 operand 둘 다 깊음).
