# b78c(T5-C 단계B nt=1) 전수 포렌식 + S1 재설계 — 2026-07-12

> [[08]] per-case 포렌식 → S1 재설계. 소스=`sim_results/t5c_b78c2.results.json.gz`(78 task nt=1·reward 0.526·pass 41/78·infra 0·user_stop 77).
> 기준 = **DB-only**(db_check.db_match·C22·reward_basis NL_ASSERTION 혼입 배제). 스크립트 provenance = scratchpad/b78c_forensic2.sh.

## 0. ★[[08]] 자기교정 (측정 아티팩트 2건 색출)
1. **초기 "33 ZERO_WRITE" = 전면 파서 버그.** tool_call 스키마가 `tc["name"]`인데 `tc["function"]["name"]`로 읽어 전 write가 0으로 셈됨 → action_checks가 write 매치(t99 ok=2)를 보이는데 exec=0인 모순으로 색출. **폐기.**
2. 수정 파서로 재분류 = 아래. (집계 라벨 "acc_post=self-conditioning"도 sd/sc 판별자 아님을 별건서 색출·C72.) **집계→결론 직행 3회 색출 = [[08]] 규율 실증.**

## 1. 전수 분류 (35 db-fail·수정 파서)
| 클래스 | n | 실체(per-case 정독 기반) |
|---|---|---|
| **WRONG/MISSING** | 16 | exec write 있으나 gold 미매치 = ⋈ 오선택 + 다중-write 일부누락(coverage) + 값오류 혼합 |
| **OVER_ACTION** | 9 | gold에 없는 write 추가 = 대화-semantic(C25/C50) + **제어 루프**(t102 22×) |
| **ZERO_WRITE** | 8 | write 전무 = coverage/discovery + **prov over-block**(t17/t39) + calc(t20) |
| **NL_ONLY** | 2 | gold write 0인데 실행(t111/t57) = over-action의 no-write-gold 변종 |

## 2. per-case 정독 (진단 3건·레버 부작용 vs 진짜 잔여 판별)
- **t102** [OVER_ACTION·22×]: "가장 최근 2-watch 주문" 주소변경·order_id 무 → **⋈ 오선택**(#W6729841 vs gold #W4219264) **+ modify_address 22회 동일반복 = 제어 루프**. ⇒ ① ⋈=(b) 경계 ② **22× 루프=레버/제어 부작용**(cap 부재·deny-loop C[`00fa5d2`]와 동류·Δspurious 위험).
- **t17** [ZERO_WRITE]: "부분(suite만) 주소, 나머지는 기존 주문서" → 에이전트 "full address 없이는 못 함"·transfer. gold=부분수정 write. **prov-rescue가 날조는 옳게 차단하나 *정당 부분수정*(기존주소 fetch+suite 치환)을 못 함 = over-block**. ⇒ GROUND/값충실도 잔여 + prov 입도 문제(C65 PROV-RESCUE-PERARG 계열).
- **t99** [OVER_ACTION]: 사용자가 "cancellation" 요청·gold엔 cancel 없음 → 에이전트가 cancel #W8855135 실행 = **대화-semantic over-action**(요청됐으나 gold-불가·C25 8/12형·C50 NO-GO 경계).

## 3. 레버 부작용 vs 진짜 잔여 (S1 설계 입력)
| 성격 | 사례 | S1 처방 |
|---|---|---|
| **제어 부작용(고칠 것)** | t102 22× 루프·기타 반복-write | ★**write-반복 cap**(동일 (name,order,args) N회 초과 차단·무료·최우선) |
| **prov over-block(고칠 것)** | t17/t39 zero-write | PROV-RESCUE-PERARG: 부분수정 시 기존레코드 fetch+치환 허용(날조와 구분·C65) |
| **(b) 문맥-⋈ 잔여** | t102 order⋈·WRONG/MISSING 다수 | T5-C silent repair/DISAMB 1차 소진 → (b)-잔여(§판단실험) |
| **(c) 대화-semantic 경계** | t99·over-action 9·NL_ONLY 2 | **게이트 금지**(C50 NO-GO)·대화-precond controller/ASK만·대부분 P3 경계 |
| **coverage 미완** | ZERO_WRITE 일부·MISSING 다중 | E-PLAN L2/CP5(구축됨) |

**핵심 재발견**: 현 스택 실패의 상당분이 **레버 부작용**(t102 루프·t17 prov over-block)이거나 **(c) 대화-semantic 경계**(over-action 11)다. 즉 pass 상향의 다음 무료 레버 = **(1) 제어 cap + (2) prov over-block 봉합**이고, over-action 11은 대부분 **경계**(scaffold 불가·Part II 입력). 순수 addressable(⋈+coverage) = WRONG/MISSING·ZERO 중 per-case 분리 필요.

## 4. S1 재설계 (nt=1 다음 사이클)
> 원칙 불변: 무료-先·per-case·Δspurious≤0·[[05]] A2만·nt=1 누적(T5-C §0b).

**S1a — 무료 부작용 봉합(최우선·GPU 무관)**:
1. **write-반복 cap**: 동일 (tool,order_id,args) 재실행 K회(제안 2) 초과 시 차단 + "이미 시도됨" 피드백(생성-레벨·히스토리 비커밋·replay-safe). t102형 22× 제거. **반대편 계측**: 정당 재시도(다른 args) 오차단 0.
2. **PROV-RESCUE-PERARG 부분수정 경로**: 부분-필드 write(주소 suite만 등) 시 나머지 필드를 **기존 조회 레코드서 보완**(fetch 강제·날조 아님) → t17/t39. over-block Δ 계측.

**S1b — WRONG/MISSING 16 per-case 분리(무료)**: order⋈(오선택) / same-order-wrong-item(값) / missing-of-multi(coverage)로 자동+정독 분리 → 각각 T5-C(⋈)/CALC·GROUND(값)/E-PLAN(coverage) 귀속. (이번 정독 3건은 진단용·16 전수 분리는 S1b 스크립트.)

**S1c — over-action 11 경계 확정(무료 정독)**: C50 재확인(대화-불가/철회 수행이 몇인가) → 게이트-불가분은 **Part II (c) 경계**로 이관·P3 계상. gate 추가 금지([[06]]).

**S1d — nt=1 재런(소액·승인)**: S1a 봉합 스택 = COMP+census+**cap**+**prov-부분수정** → 78(or 26 표적) nt=1 → per-case → Δ(부작용 제거분) 측정. GO=pass↑ ∧ Δspurious≤0 ∧ 루프 0 ∧ over-block Δ≤0.

**도달 기대(정직·상한)**: 부작용 봉합(t102류 루프 + prov over-block)만으로 pass 몇 점 회복 가능(정확 크기=S1b 분리 후)·over-action 11은 경계라 대부분 미회복. 0.526→? 는 S1d 실측.

## 5. 미해결·리스크
- WRONG/MISSING 16·ZERO 8의 **per-case 세부 귀속 미완**(정독 3건만) → S1b가 확정. 그 전 레버-크기 단정 금지([[08]]).
- write-cap이 정당 다-write(다른 order 연속)와 충돌 안 하게 = (tool,order,args) 3중키 필수.
- over-action 11의 경계 비율은 C50 재확인이 확정(현재 t99 1건만 정독).
