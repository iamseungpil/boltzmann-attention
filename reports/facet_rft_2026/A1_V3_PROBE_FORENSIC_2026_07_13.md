# A1-v3 probe 실패 13건 전수 per-step 포렌식 + 대책 (2026-07-13)

> 데이터: `sim_results/genv3_probe.results.json.gz`(18-probe·A1-v3 스택·5통과/13실패) + 리모트 로그 `genv3_probe.log`(레버 발화) + 대조군 `genv2_a1v2`(A1-v2 full·arm 교차).
> 방법([[08]]): 종료사유 분포(user_stop 12·max_steps 1·crash 0) · 병렬 4에이전트 per-step 정독 13/13 · gold action_checks/d[tasks] 결정론 diff · 레버 발화는 로그 물증으로 교차(궤적엔 생성-레벨 개입 비가시) · nt=1 한계는 arm-교차·nt4(진행 중)로 보강.
> 선행: `COVERAGE_LOOP_DESIGN_2026_07_13`(8건 1차 포렌식·coverage 재설계) · `A1_V3_DESIGN` v3.1(레버 세트). 본 문서 = **13건 완결판 + v3 신규 회귀 2건의 물증 + L4a 성적표**.

## 0. 결론 (요지)
1. **v3 신규 회귀 2건 확정**: t58·t32는 **A1-v2에서 통과(db=True)했는데 v3에서 실패**. t58 = **L4a가 모델의 정답을 오답으로 덮어씀**(로그 물증). t32 = 요청-레코드 위 과잉서비스(semantic·v2 통과는 nt1 요행 가능 [?]).
2. **L4a 성적표 = 치환 2/2 오답·순효과 −1**: t58(정답→오답 파손·회귀 직접원인)·t20(오답→다른 오답·무익). keep 10(무해). **L4a 치환은 재설계 전 정지 권고**(§3-L4).
3. 13건 원인 분포: 진짜 coverage 3 · v3 자해(L4a) 1 · 순결정론 가드 가능 5 · 의미(learn/ASK) 2 · infra(user-sim) 1 · 복합 1.
4. **벤치-수준 신발견: user-sim 확인은 검증 축이 아니다** — 오확인·세탁 6건(t97·t76·t71×2·t79·t20). 신뢰 가능한 검증 = grounded DB 대조뿐.

## 1. 13건 완결 표 (도입 스텝·원인·대책)
| task | 도입 스텝(핵심 증거) | 근본원인 | 분류 | 대책(레버) |
|---|---|---|---|---|
| t41 | 주소수정 1 intent가 2주문 span·#W9583042 누락 | coverage | coverage | **COV** FIND-subset 루프 |
| t81 | 취소 2주문 중 1개만·"안 쓰는 것들"=형식화 불능 | coverage | coverage | **COV**(enumerate-ASK 경유) |
| t92 | 반품 2주문 중 #W3239882는 read 0 | coverage | coverage | **COV**(read 강제가 해독제) |
| t35 | msg23서 반증 읽고도 user 오매핑 절반 수용→item∉주문 모순 바인딩→가짜 오류 프레이밍→미존재 tool | 선택적 grounding+바인딩 | BIND | **L10** 멤버십 가드(순결정론) |
| t97 | tool call 0회로 "123 Broadway" 날조(msg24)→user yes 세탁→DB 오염. gold 주소는 미조회 #W3407479에 실재 | 날조+누락 GET | PROV | **L3** origin-prov+fetch-first(순결정론) |
| t76 | 2번째 cancel reason을 1번째서 복사·user 동기(msg51)에 mistake 0건·yes/no 확인이 세탁 | enum carryover | GATE/enum | **L11** attested 검사+개방형 ASK 강등 |
| t64 | predicate 동률 2후보{466.75, 481.50} first-fit·COMPUTED FACTS의 정답 무시 | 동률 미해소 | FIND(동률) | **L4-tie** min-price(+전수비교 강제) |
| t54 | user-sim이 known_info에 없는 "black size 8" 날조→동정 교착→transfer(user 요청)→hold-loop 149/201 msgs | user-sim 날조+비종결 | infra+종결 | **L12** 종결 가드 + **집계 infra 분리** |
| **t20** | 모델: size-9 내 argmax 오계산($148.95≠max 155.33). **L4a: 제약-무시 전역 argmax(4153505238·size 8) 치환** — 실행 인자가 user 승인값(9635758562)과도 불일치. 신발 1슬롯 외 3아이템 전부 정답 | 이중 실패(모델 제약내 극값 오계산 + L4a constraint-blind) | FIND(오치환) | **L4a 재설계**(§3)+**승인-일관성 가드** |
| **t58 ★v3회귀** | 모델이 msg18서 gold 정확 특정($2908.42·9bar)·정답 call 발행 → **로그: `[T2_L4] substituted 3815173328→3714494375`** = 랩탑의 "cheapest"가 에스프레소에 교차-누출·복합기준 무시·floor-guard 무력(정답∉극값집합) | **L4a가 정답 파괴** | 가드 자해 | **L4a 치환 정지→재설계**(§3) |
| **t32 ★v3회귀** | user "잃어버렸다" 직후 에이전트가 return 제안(msg20)→실행(msg24)=gold에 없는 spurious write. gold write 3/3·communicate 전부 성공. transfer·sim-이탈 미재현(과거 결론 교정). sibling over-action 0(reads-cond 의도대로) | 요청-레코드 위 과잉서비스(분실≠반품가능=의미) | over-action(의미) | **learn/ASK 날개** + 어휘 게이트 후보(lost/missing→write 전 개방형 재확인) [D]. nt4로 flip율 확인 |
| t71 | "most recent"가 DB에 timestamp 부재=결정불능인데 리스트 마지막을 근거 없이 바인딩(msg10)→msg18서 모순 2건(주소 이미 기본값=no-op write·desk lamp 부재) 자기 발화하고도 통과→user-sim 2회 오확인 고착 | 근거-없는 바인딩+회복 실기 | FIND(first-fit)+sim복합 | **no-op write 가드**(요청 변경값==현재값→오바인딩 신호·재disamb)+**recency 불능→내용-기반 disamb 강제**(순결정론) |
| t79 | "other 1L bottle" 참조를 미해소(잔여 2주문 미조회)한 채 색=black 가정(msg18·교체대상에 앵커링)→user-sim이 실제 red를 black으로 오확인. GET 후엔 predicate 유일해(red stainless 1L) | 교차-주문 참조 GET누락 | GET누락 | **L2-ref**: user-언급 참조 엔티티가 grounded 미매칭→잔여 read 강제(discovery 동형·순결정론) |

## 2. ★신규 교차 패턴 (13건 종합·[M])
1. **user-sim 확인 신뢰 불가(6건)**: t97(날조 승인)·t76(carryover 승인)·t71(오바인딩 2회 승인)·t79(틀린 색 오확인)·t20(승인값과 다른 값 실행됨) — **"사용자 확인"은 이 벤치에서 가드가 아니라 오염 통로**. 검증 축 = grounded DB 대조만. ⇒ 전 레버 ASK=개방형 원칙(v3.1) 재확인 + **에이전트-제시값 확인 의존 금지**.
2. **승인-일관성 위반(t20·신규)**: 결정론 치환이 **user가 승인한 값과 다른 값**을 실행 — 치환 계층의 새 harm 축. ⇒ **승인-일관성 가드**: 대화서 명시 확인된 write 인자는 치환 금지(또는 재확인 강제). 순결정론(확인 발화의 값 vs 실행 인자 대조).
3. **단일-슬롯 실패(7건)**: t20/58/64/76/79/71/97 전부 "write 인자 1개"만 오염·나머지 정답 — 실패 = predicate 파라미터 오염이지 실행능력 아님. 두-날개 프레임(formalize 정밀도) 정합.
4. **walk 채널 누설(2건·부수)**: t58 msg28("The user did not mention…" 3인칭 메타발화)·t71 msg34("internal note…") = stop-후 주입 턴이 대화에 누설 — walk 폐기 근거 추가(채널 오염).
5. **가짜 오류 프레이밍(기존 3건)**: t97/35/76 — tool call 0회+"I encountered an issue" 템플릿(1차 포렌식).

## 3. ★L4a 재설계 요건 (치환 정지 권고·물증 기반)
**성적표**: 치환 2/2 오답(t58 파손·t20 무익)·keep 10. 격리 l4probe서 "t52 회복"으로 보였으나 **합성 스택 실측서 t52는 L4 무개입 통과**(극값축=zoom≠price·후보 사실상 유일)·**price-치환이 개입한 2건은 전부 오답**.
실증된 결함 4:
- **(F1) 교차-품목 기준 누출**: 랩탑의 "cheapest"를 에스프레소 슬롯에 적용(t58). ⇒ 극값어는 **해당 품목에 대한 user 발화에 attested**일 때만 그 슬롯에 적용(per-slot criterion binding).
- **(F2) 복합기준 무시**: "cheapest **i7 or above**"·"same **size**"의 제약부를 버리고 전역 극값(t58 잠재·t20 실증). ⇒ **keep-제약/속성-제약 감지 시 치환 no-op**(보수) 또는 **constrained-argmax**(제약-매칭 부분집합 내 극값·바인딩 불확실시 no-op).
- **(F3) floor-guard 구조 무력**: cur∈극값집합 아니면 무방비 — 극값집합 자체가 틀리면(F1·F2) 정답을 파괴. ⇒ floor 조건을 "cur가 available ∧ 동일 product ∧ user-확인됨이면 keep"로 강화.
- **(F4) 승인-일관성 부재**: user 승인 후 몰래 치환(t20). ⇒ §2-2 가드.
**적용**: F1∧F2∧F4 충족 전 **L4a 치환 OFF**(keep/annotation-only). L4-tie(t64·동률 해소)는 별개 — 치환이 아니라 *모델이 비교를 안 한* 케이스의 전수비교 강제라 유지.

## 4. 대책 종합 (구현 대상·우선순위 갱신)
| 우선 | 대책 | 유형 | 근거 태스크 | 비용 |
|---|---|---|---|---|
| 0 | **L4a 치환 정지**(재설계 요건 §3 충족까지) | 스위치 | t58 회귀 제거(+1 즉시) | 0 |
| 1 | L10 멤버십 | 순결정론 | 35 | 소 |
| 2 | L3 origin-prov(+fetch-first) | 순결정론 | 97 | 소(설계 기존) |
| 3 | 승인-일관성 가드 | 순결정론 | 20(+치환 계층 전반) | 소 |
| 4 | no-op write 가드 + recency-불능→내용 disamb | 순결정론 | 71 | 소 |
| 5 | L2-ref(참조 미해소→read 강제) | 순결정론 | 79 | 중 |
| 6 | L12 종결 가드 | 순결정론(비용) | 54 | 소 |
| 7 | COV FIND-subset 루프 | 루프 | 41·92·81 | 중(정본 COVERAGE_LOOP §3) |
| 8 | L4-tie(전수비교+min-price) | 결정론(1규칙) | 64 | 소 |
| 9 | L11 enum attested+ASK 강등 | 탐지 결정론 | 76 | 소 |
| 10 | L4a constrained-argmax 재설계 | 재설계 | 20·58 | 중 |
| — | t32: learn/ASK 날개(어휘 게이트는 [D] 후보) · t54: infra 분리 표기 | 의미/infra | 32·54 | — |
- **기대치(정직·[D])**: 대책 0~9 완비 시 13건 중 폐쇄 후보 = t58(정지로 즉시)+t35·97·20*·71*·79·64·76*+t41·92·81 ≈ 8~11건. *표=모델 자체 오류 동반(t20 제약내 극값 오계산)·sim 오확인 위험(t71·79)·INFER 잔여(t76)로 부분 비보장. t32·t54=레버 밖.
- 전부 격리 probe(표적+무회귀 trivial+Δspurious)로 [M] 승격 후 편입. 무료 unit 先([[09]]).

## 5. nt=1 한계·검증 계획
- t32·t58의 "v2 통과"는 nt1 — **A1-v2 nt4(r3·r4·r5 진행 중)에서 flip율 실측**으로 회귀 판정 보강(t58은 로그 물증이라 nt 무관 확정·t32는 flip 가능 [?]).
- t20 모델-자체 오류(제약내 argmax 오계산) = L4a 정지 후에도 잔존 예상 — constrained-argmax 재설계(§3-F2)까지 미폐쇄.
- 대책 구현 후 표적 probe 구성: 13 실패 + 가드(58 무회귀 재확인·32 관찰·27/58/83 Δspurious) + trivial ≈ 18-20 task.
