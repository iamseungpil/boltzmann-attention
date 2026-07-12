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

## 2b. ★S1b 세부 분류 (write실패 33 전수·`scratchpad/s1b.sh`·gold-write 단위 disposition)
> gold 미매치 write마다: 같은-order+tool 다른args=**VALUE** / 그 order+tool write 없음=**MISSING** / gold-외 order write=**ORDER-⋈**. + refusal-language(ZERO).

| 세부 클래스 | n | tasks | 성격·레버 |
|---|---|---|---|
| **VALUE/item-⋈** | 11 | t8·36·37·38·58·59·96·97·98·100·109 | 옳은 order+tool, **틀린 item/변형 선택**. **★리뷰 정정: 이질 버킷** — **t36·t37 = 기준-계산형**(최저가/예산 = CALC-EXT·문맥-의존분은 FORMALIZE-EXEC[동결]·`NEXT_LEVER §2.2`가 t37 명시) ≠ DISAMB(서브콜에 계산 시키면 추측). 나머지 = 순수 값-⋈ → **T5-C/DISAMB** |
| **MISSING/coverage** | 8 | t34·41·66·74·76·86·110·112 | 다중-write 중 일부 누락 = **coverage** → **E-PLAN L2/CP5**(구축됨) |
| **ZERO(prov/refuse-block)** | 8 | t17·20·22·39·40·64·77·82 | write 전무+거부언어 = **prov over-block**(t17)·**calc**(t20)·⋈-refuse(t82) 혼합 |
| **ORDER-⋈(wrong order)** | 5 | t69·71·94·102·103 | gold-외 order에 write. **★리뷰 정정: "→DISAMB 일괄" 기각** — **t71 = ASK-채널 확정**(정본 `CENSUS_LEVERS §2a` V0: most_recent 사망·retail 전 tool 출력 날짜필드 0건 → "가장 최근"은 도구로 결정 불가 = C48 위계상 ASK·DISAMB 서브콜도 동일 이유로 결정 불가). **t102 = ASK 의심**(§2 자체 정독: 같은 "가장 최근" 기준·루프 부작용 병발). **t94·103 = 내용-기술 order ✅정독**(described-item 매칭으로 오선택·"내가 받은 15인치 노트북"·"같은 주문의 bookshelf+jigsaw" → **내용-매칭 order 해소 가능**·order-id 열거 C61과 무관 = 내용-DISAMB/gather-match 후보) · **t69 = framing 혼선 ✅정독**(user "received laptop → return" vs gold "pending → cancel" = 행위·상태 불일치 → 경계/user-sim 의심·DISAMB 아님) |
| genuine over-action | 1 | t99 | gold 2매치+추가 cancel = **(c) 대화-semantic 경계**(C50) |

**★핵심 재프레이밍(2b가 §2 3건-정독을 정정 → 리뷰가 레버 배분 재정정)**: "OVER_ACTION 9 = (c) 경계"는 **과대해석**이었다 — genuine (c) 경계는 **t99 1건 + NL_ONLY 2 = ~3**뿐(경계 축소는 유지·산수: OVER_ACTION 9 = ORDER-⋈ 5 + genuine 1 + VALUE/MISSING 재배치 3). **단 "addressable 16 → 전부 T5-C/DISAMB"도 같은 실수의 반복(일괄 라우팅)** — addressable **총량 ≈16은 유지**하되 레버가 갈린다: **DISAMB ≈ 9~12**(VALUE 순수분 + ORDER-⋈ 정독 후 확정분) · **CALC/FORMALIZE 2**(t36·t37 기준-계산형) · **ASK-채널 1~2**(t71 확정·t102 의심 — ASK는 addressable이나 user-sim 협조 의존 = 확률적 커버·t57형과 동류). **[[08]] 실증 2중: 소표본 정독도, 휴리스틱 전수도 각자 과대일반화 — 서로가 서로를 교정(3건-정독↔전수분류↔정본대조).**

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
1. **write-반복 cap**: 동일 (tool,order_id,args) 재실행 K회(제안 2) 초과 시 차단 + "이미 시도됨" 피드백(생성-레벨·히스토리 비커밋·replay-safe). **반대편 계측**: 정당 재시도(다른 args) 오차단 0.
   - **★F5 데이터-확정(t102 전수·`scratchpad/t102loop.sh`)**: modify_address **19회 전부 같은 order(#W6729841)·전부 err=False(성공)** = **순수 "성공-write 재발emit" 루프**(deny-loop 아님=DENY_CAP 4 무관 · env-거부 아님=C63 무관 = **신규 유형**). ⇒ cap 정당·기존 deny-cap과 독립.
   - **★재발emit 원인 조사 필수(cap 전)**: 성공 write를 19회 반복 = **regen/silent-repair가 매 턴 동일 write를 재주입하는 스택 부작용 의심** — cap은 증상 제거, **원인(스택 상호작용)은 별도 색출**(레버 부작용 2호 후보).
   - **★cap ≠ pass(F5)**: t102 root = ⋈/ASK 오선택(#W6729841 vs gold #W4219264·"가장 최근"=ASK). cap은 thrash(턴·비용·Δspurious)만 제거·pass-회복으로 계상 금지.
2. **PROV-RESCUE-PERARG 부분수정 경로**: 부분-필드 write(주소 suite만 등) 시 t17/t39형 over-block 봉합. **★[[05]] 명문화(리뷰)**: **엔진이 필드를 채우지 않는다** — 메커니즘은 `CENSUS_LEVERS §1` 정본 그대로 **deny+피드백("기존 레코드를 getter로 조회해 나머지 필드를 원문으로 쓰라") → *에이전트가* fetch → *에이전트가* write**. "기존 레코드서 보완"을 엔진-측 값 채움으로 구현하면 Q3 위반(autofetch류·write-인자 생성). over-block Δ 계측.

**S1b — 세부 분리 ✅완료(§2b·리뷰 레버-배분 정정 반영)**: addressable ≈16 · coverage 8 → E-PLAN · genuine (c) 경계 ≈3. **레버 배분(정정)**: DISAMB ≈9~12(VALUE 순수분·ORDER-⋈ 정독 후) · CALC/FORMALIZE 2(t36·t37) · ASK-채널 1~2(t71 확정·t102 의심) · ZERO/block 8은 prov/calc/⋈ 3분 정독 잔여. ⇒ **다음 스택 표적 = T5-C/DISAMB(순수 값-⋈)·E-PLAN(coverage 8)·S1a 봉합(루프+prov-block)**·기준-계산형은 CALC-EXT(기구현)/FORMALIZE(동결 V0). gate 추가 없음. **S1b-잔여(무료 정독): ~~ORDER-⋈ t69/94/103~~ ✅완료(t94/103=내용-매칭·t69=경계/framing) → 남은 것 = ZERO/block 8의 prov/calc/⋈ 3분 정독만.**

**S1c — over-action 11 경계 확정(무료 정독)**: C50 재확인(대화-불가/철회 수행이 몇인가) → 게이트-불가분은 **Part II (c) 경계**로 이관·P3 계상. gate 추가 금지([[06]]).

**S1d — nt=1 재런(소액·승인)**: S1a 봉합 스택 = COMP+census+**cap**+**prov-부분수정** → 78(or 26 표적) nt=1 → per-case → Δ(부작용 제거분) 측정. GO=pass↑ ∧ Δspurious≤0 ∧ 루프 0 ∧ over-block Δ≤0.

**도달 기대(정직·상한)**: 부작용 봉합(t102류 루프 + prov over-block)만으로 pass 몇 점 회복 가능(정확 크기=S1b 분리 후)·over-action 11은 경계라 대부분 미회복. 0.526→? 는 S1d 실측.

## 5. 미해결·리스크
- ~~WRONG/MISSING 16·ZERO 8 per-case 세부 미완~~ → **S1b ✅완료(§2b)**. 잔여 미완 = **ZERO/block 8의 prov-block vs calc vs ⋈ 3분**(refusal-language 휴리스틱만·8건 전수 정독 필요) + VALUE 11이 값-⋈(문맥해소가능)인지 순수경계인지(T5-C 적용 후 (b)-잔여로 확정).
- write-cap이 정당 다-write(다른 order 연속)와 충돌 안 하게 = (tool,order,args) 3중키 필수.
- over-action 11의 경계 비율은 C50 재확인이 확정(현재 t99 1건만 정독).
- **★F4(리뷰) VALUE 11 사정거리 정직화**: (a) **아티팩트 혼입** — 같은-order+tool·args-diff 휴리스틱은 **C26 variant-leak**(item_ids 순서 의존 db_hash·의미동일 write가 fail·레버 아님)·**C28 reason-enum**(user-sim이 gold와 모순·레버 아님)을 못 거른다 → VALUE 11 중 **1~2건 non-addressable 아티팩트 가능**(S1b-잔여 정독서 배제). (b) **경계 floor** — DISAMB(열거)가 값-⋈를 +31pp 열지만(C59) **~.34 잔여=진짜 의미/미결정**(C56 체계핵 t8 variant 4/4=thinking-flat)은 안 닫힘 → VALUE addressable도 **전량 아님**. ⇒ "DISAMB 9~12"는 상한·(b)-잔여가 learn/경계로 남음.
