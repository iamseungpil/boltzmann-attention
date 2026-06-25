# present+nested+g15 전수 실패 census + 날조 원인 확정 (2026-06-25) — [[08]] 궤적 정독

> 도구 `escape_det_census.py`(결정론 census) + 전수 에러-메시지 census + 궤적 정독. 데이터 = retail t3 (32B `on_n32int8_presentnest_g15_retail_t3`·n=342·14B 교차확인). 목적 = (1)nested arm 효과 (2)날조 원인 (3)**모든 실패 케이스 전수 분류**.

## 0. 요약 (확정)
- **nested present ≠ 에러 유발**(32B errors↓·too_many 1→0·14B 7/9 nest=0). 가설 반증.
- **nested 68-75% 발화하나 컨테이너 not_found 못 줄임**(post-lookup 구조). write-레이어 L2/L3만 32B서 약간 도움(wMatch +.021·L3 38→25)·14B neutral.
- **★지배 실패 ≠ "container 날조(C4)" — 그건 과장이었음(철회).** 실제 = **operation-semantic 정책위반(bizrule)** + **silent-wrong-write(operand)**. not_found 다수는 *양성*(유저-오류 복구).
- **레버**: 정책위반=**결정론 게이트 확장**(G5 family)·operand=present/nested 부분+잔여·날조=provenance(소수).

## 1. ★날조 원인 확정 (궤적 정독)
- **진짜 날조 = "값 없으면 ASK 대신 generic placeholder 발명" (R1B·소수·대부분 복구)**: task57 — 유저가 이메일 안 줬는데 모델이 `find_user_id_by_email("user@example.com")` 호출→not_found→곧 name+zip 물어 복구. 원인=[[43]] §0(채점이 기권<추측). 
- **not_found 다수 = 유저-sim이 틀린값 줌 + 모델 복구 (날조 아님·실패 아님)**: task47 — 유저 "order 9502126"(틀림)→모델 충실 사용→not_found→`get_user_details`로 진짜 #W9502127 복구.
- **fab 카운트 과대 보정**: tau2 실제 이메일 도메인=`@example.com`. `isabella.lopez3271@example.com`(task40)=유저가 준 *진짜* 이메일(분류기 오판). generic 날조(user@/customer@/#W0000000)는 더 소수.

## 2. ★전수 툴-에러 taxonomy (32B nest·n=342·distinct 전부)
### A. OPERATION-SEMANTIC 정책위반 (bizrule·env가 정책-무효 행동 거부) — 총 ≈132
| n | 메시지 | retail 정책 | 모델이 왜 위반 | 레버 |
|--:|---|---|---|---|
| **69** | "The number of items to be exchanged should match" | exchange/modify-items는 \|new_item_ids\|==\|item_ids\| 필수·**remove 연산 없음** | item *제거* 시도(new=[]) or 개수 불일치 | 게이트: count-match precondition |
| **46** | "Payment method should be the original payment method" | 환불/결제변경은 **원결제수단**(또는 gift card)만 | 다른/새 결제수단 지정 | 게이트: payment-original |
| **11** | "The new item id should be different from the old item id" | exchange는 변형을 *바꿔야* | new==old(no-op exchange) | 게이트: new≠old |
| **3** | "Insufficient gift card balance to pay for the price difference" | gift card가 차액 커버해야 | gift card 선택(차액 부족) | 게이트/정당 |
| **2** | "Insufficient gift card balance to pay for the order" | gift card가 주문액 커버해야 | gift card 선택(잔액 부족) | 게이트/정당 |
| **1** | "Non-pending order cannot be modified" | pending만 수정가능 | 비-pending 수정 | G5(기존)·1건 누락 |

### B. OPERAND not_found (item/variant/payment id가 대상에 없음·그라운딩) — 총 ≈72
| n | 메시지 | 의미 | 층 |
|--:|---|---|---|
| 20 | "Number of \<item_id\> not found" | item_id가 주문에 없음 | L2 |
| 17 | "Payment method not found" | payment_method_id가 유저것 아님 | operand |
| 14 | "Variant not found" | variant id가 product에 없음 | L3 |
| 11 | "Some item not found" | item_ids가 주문에 없음 | L2 |
| 8 | "New item \<ID\> not found or available" | new_item_id가 유효/가용 변형 아님 | L3 |
| 9 | "\<ID\> not found" | generic | operand |
| 2 | "Item not found" | | operand |

### C. CONTAINER not_found (user/order 그라운딩·**대부분 양성=유저-오류 복구**) — 총 ≈75
| n | 메시지 | 의미 |
|--:|---|---|
| 57 | "User not found" | email/name-zip 미해결 (유저-오류 복구 多·placeholder 날조 少) |
| 18 | "Order not found" | order_id 틀림 (유저가 틀린 order# 줌→복구 多·fab 少) |

### D. GATE 거부 (scaffold 정상작동·실패 아님) — 총 ≈35
| 32 | GATE:G4_TRANSFER_MSG | transfer-notice 집행 |
| 3 | GATE:G1_AUTH_FIRST | auth-first 집행 |

## 3. ★reward=0 전수 실패-모드 census (42 task·trial0)
- **① silent-wrong-write (실행 성공·nerr=0·reward=0) ≈14** [task 3,4,20,41,63,72,76,79,84,99,100,104,109,112]: *유효* write인데 gold와 불일치 = **operand/operation 오선택**(env가 허용했으나 틀림). **최대 단일 클래스.** 예: task79=wrong variant(L3)·task102=⋈ 주소. 게이트가 못 잡음(write 유효). 레버=present/nested 부분(32B)+comprehension/operand 잔여.
- **② 정책위반 연루 (bizrule_err>0) ≈9** [task 13,34,36,37,38,40,69,81,105]: 정책 벽 충돌(일부 다른 trial서 too_many로 루프). 레버=게이트 확장(§2A).
- **③ no-write 미도달 (did_write=False) ≈6** [task 24,39,43,64,67,68]: 상류(auth 실패·포기·communicate-only). 혼재.
- **④ not_found 연루·복구했으나 wrong ≈13** [task 10,30,46,47,57,82,91,92,95,102,103,107,111]: lookup 에러 후 복구했으나 최종 틀림(→대부분 ①과 겹침=복구 후 wrong operand).

## 4. 레버 배정 (확정)
- **정책위반(A·~132 에러·~9 task) → 결정론 게이트 확장** (G5 family·constraint precondition: count-match·payment-original·new≠old·balance·pending). decidable(args+DB state). [[05]] 준수(엔진 일반+A2 사실). **학습/present/autofetch 아님.** + too_many_errors 루프도 차단(현재 모델이 무효 op 반복).
- **operand(B·silent-wrong ①) → present/nested 부분**(32B)·잔여=C4/M-σ 클래스(학습 전이-음성·[[20]])·일부 comprehension.
- **container not_found(C) → 대부분 양성**(실패원 아님)·날조 소수=provenance 게이트(기존).
- **gate(D) → 정상.**

## 5. 함의 / 다음
- **학습 NO-GO 방향 유지·근거 정정**: "not_found=C4 지배"가 아니라 **지배 잔여=결정론-게이트-able 정책위반(A) + operand(B·C4계열)**. 둘 다 학습 아님(A=게이트·B=전이음성).
- **다음 결정론 레버 = bizrule 정책위반의 게이트화 census** — A의 6규칙을 A2 constraint-gate로 표현가능한지·G5 family 확장. present/nested/학습 아님.
- **남는 진짜 learn 후보 = silent-wrong-write①의 comprehension 분**(operand-copy 제외 후·"유저 요청을 어느 op/operand로 formalize") — 그 실재·크기는 게이트+present 적용 *후* 잔여로만 측정.
