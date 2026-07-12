# 보강 방법 설계서 (REMEDIATION DESIGN) — 2026-07-13

> ★A1 실패 케이스를 frontier 대조로 "scaffold+A2 복구 가능 / learn·scale 잔여"로 가르고, 이번 세션 도출 보강 레버를 전부 통합·설계.
> 파생: `A1_REGRESSION_PERSTEP_FORENSIC_2026_07_13`(per-step+ablation) · `GENERALIZED_SCAFFOLD_ARCHITECTURE_2026_07_12`(LOCK).
> 불변: [[05]] scaffold 도메인일반·A2만 · [[10]] 선택기=결정론·생성기=LLM · Δspurious≤0 · gold-independence · [[09]] 무료검증 先.

## 0. 근거가 된 실측 (이번 세션)
- **회귀=25**(strict8 + major17). 단일-요인 ablation(COMP→−present→+eplan, prov=full·cap off·nt2·회귀25).
- **eplan = 순이득 +10pp**(52→62%)·핸드오프 "eplan 과-블록 순손실"은 **sign 오류**. ⇒ **eplan 유지·harm만 정밀교정**(제거 아님).
- **present↔eplan = 대체재**(둘 다 discovery/후보 제공). present=spoon-feed(anti-drift)라 eplan/GET-forcing로 대체.
- **cap·rescue-passthrough 실발화 0**. robust 순손실 = present(t7·t92)·eplan(t43 부분)·prov-rescue(t6·46·55·확정대기).
- **filter dotted-path 수정 확증**(filter one 0→5·t102 회복).

## 1. Frontier 대조 = 복구 경계
**scaffold+A2 복구 가능** = 답이 문맥/정책서 *결정 가능* ∧ frontier도 *구조적 기전*(GET·필터·확인·전제)으로 푸는 것. **learn/scale** = 답이 의미 이해에 있어 frontier가 *능력*으로 푸는 것.

| 실패 유형 | tasks | frontier 기전 | 복구 | 레버 |
|---|---|---|---|---|
| 주문선택 ⋈(후보 다수조회) | 55·59·101 | describe-confirm/content-match([[47]]) | ✅ | L1 filter |
| 주문선택 ⋈(틀린주문만) | 83·92·112 | 전수 read 후 content 선택 | ✅ | L2 GET-forcing+L1 |
| coverage(멀티주문) | 42 | 관련 주문 전수 read | ✅ | L5 eplan(순이득) |
| 주소 날조 | 96 | 원천 주문 GET(발명 안 함) | ✅ | L3 origin-prov |
| 극값/명시속성 변형 | 52·77·110 | 변형 read 후 argmax/필터 | ✅ | L4 fexec-variants |
| 전제/정책 | 21·38 | 정책 준수(exchange=delivered·split불가) | ✅ | L7 precondition gate |
| 집합연산 | 108 | items 집합-여 | ✅ | L9 set-op |
| qty-conflation transfer | 58 | 한 주문 N품목 정상처리 | ✅ | L5b examined-safe |
| 의미적 변형(미명시) | 94·23·7 | 제품 이해 | ❌ learn/scale | (underspec→ASK 안전낙하) |
| 의도/행동(오행동) | 108·77 행동분 | NL 의도 파악 | ❌ 주로 learn/scale | (일부 L7) |
| 속성-노이즈 | 38·43 | — | N/A(노이즈/underdet) | — |
| user-sim 이탈/nt노이즈 | 32·6·46·55 | — | N/A(하네스) | — |

**대략 복구율 ≈ 15/25(60%) scaffold+A2 · 5-7(25%) learn/scale · 4(15%) 비-능력.**

## 2. 보강 레버 카탈로그
> 각 레버: 문제·방법·frontier정합·[[05]]준수·상태. 엔진=도메인일반·A2=스키마/정책분.

- **L1 filter-substitute(dotted-path)** [DONE·확증]: ≥2 후보 order를 LLM-formalize→엔진 결정론 필터(any-match·점경로). 1?치환:≥2?열거-ASK. frontier=content-match. 대상 55·59·101.
- **L2 GET-forcing(주문선택 discovery)** [설계]: 애매 order write 전, 후보 order 상세 **전수 GET 강제**(examine 안 된 후보 있으면 write 보류→read) → 그 후 L1 필터. filter 단독이 못 여는 "틀린주문만 조회"(83·92·112) 해소. frontier=전수 read. [[05]]: A2 getter-스키마·엔진 루프.
- **L3 origin-provenance(세탁 대응)** [설계·turn-order 확증]: 값이 도구출력/이전 사용자발화보다 **에이전트가 먼저** 낸 것이면(first-mention=ASSISTANT) 나중 사용자확인 무효→getter 있으면 GET 강제. t96 "123 Broadway"(first=ASST) 거절·t43 user주소(first=USER) 수용. frontier=원천 GET.
- **L4 fexec-on-variants(argmax/필터)** [설계]: L1 fexec 엔진을 **product-variant record**에 적용. "가장 줌 큰"(52)=argmax(zoom)·"가장 큰 사이즈"(77)=argmax·"7in/128/black"(110)=filter. frontier=변형 read+선택. 명시속성만(의미매칭=learn 잔여).
- **L5 eplan(coverage)** [유지·순이득 실증]: discovery-enforce. **제거 아님** — harm만 §3서 교정.
- **L5b eplan examined-safe** [구현됨·검증대기]: write 대상 order가 examined면 discovery-deny 생략. §3 t58 fix.
- **L6 eplan reads-only** [설계]: eplan은 read만 강제·sibling write 유도 금지. §3 t32.
- **L7 precondition gate** [부분존재]: A2 정책분. exchange=delivered-only(21 pending→modify 유도)·split-payment 불가(38). frontier=정책준수. gate kind=preconditions(기존 온톨로지).
- **L8 prov feedback 강도(payment/id)** [확정대기]: rescue 중립문이 약해 exchange payment_method_id 날조 미해소→포기(transfer·t6). id/payment류=directive/full 피드백. (rescue arm 확정 후 fix.)
- **L9 set-op coverage** [설계]: "X 빼고 전부"(108)=order items 집합-여 결정론. coverage 동류.

## 3. ★eplan-hurt 6건 심층 대책 (per-step 기전 → 교정)
> arm1(no eplan·pass) vs arm2(eplan·fail) 전체 콜 대조. eplan은 순이득이나 3가지 harm-mode 존재.

### 3a. harm-mode I — qty-conflation → transfer (t58·robust 2/2 fail)
- **기전**: "TWO items(커피머신+랩탑)"=한 주문(#W5838674) 요청. 에이전트가 대상 order examine 후 exchange 시도 → eplan L2가 "N=2 > 실행0, 미검토 sibling {#W2782744·#W4284542}" deny(품목수를 주문수로 오독) → 무관 주문 화해 불가 → **transfer**. arm1은 그냥 exchange(pass).
- **frontier**: 한 주문의 2품목을 그대로 exchange. "2품목=2주문" 오독 안 함.
- **대책 = L5b examined-safe**(대상 #W5838674 examined→deny 생략) + **L5c qty-guard 강화**: `qty_item_covered`가 실행 전(coverage 0)이라 불발 → **시도 write의 attempt_items(2품목)를 coverage에 선반영** → N=2 충족→deny 안 함. 둘 중 examined-safe가 더 일반(대상 검토=discovery 완료).
- [[05]]: 도메인 리터럴 0(examined=ledger 사실·attempt_items=A2 items_key).

### 3b. harm-mode II — deny가 궁지→transfer + 오주문 (t83·t69·robust 2/2 fail)
- **t83 기전**: "두 태블릿 중 **비싼 것** 반품"·4주문. arm1 tr0=4주문 전수 examine→#W9571698(비싼거) 반품(pass). arm2=#W3069600(틀린것)만 examine→deny 압력→**transfer**(tr0) or 틀린주문 반품(tr1). eplan이 discovery를 *강제*하려다 오히려 에이전트를 궁지로 몰아 포기.
- **t69 기전**: "받은 랩탑 반품"이나 gold=cancel #W2417020(pending). arm2가 주문들 훑다 deny 압력→transfer(양 trial). 오주문(#W5605613) + 도구오선(return vs cancel).
- **frontier**: 후보 주문 전수 read → 내용("비싼 태블릿")으로 선택 → 진행. 절대 transfer 안 함.
- **대책 3중**: ① **L2 GET-forcing이 discovery를 *완성***(전 후보 examine까지 유도·중도 deny로 방치 아님) → ② **L1 filter**("비싼 태블릿"=argmax price·"랩탑 든 주문"=content) 로 올바른 주문 결정 → ③ **eplan no-transfer 가드**: 에이전트가 eplan-deny 중 transfer로 이탈하려 하면 deny 해제(포기 방지). frontier의 "전수read+content선택"을 scaffold가 재현.
- 근거: arm1 tr0(전수 examine)가 pass·arm2(부분 examine+deny)가 fail = **discovery 미완성이 harm**. eplan이 "read 더 하라"만 하고 완성 못 시킴 → GET-forcing+filter가 완성.

### 3c. harm-mode III — 과-행동(sibling에 spurious write) (t32·부분 1/2)
- **기전**: gold=3 write(cancel·cancel·return). arm2 tr1이 **불필요한 return #W2692684 추가**(gold 아님). eplan L2가 sibling #W2692684 surface("read then decide") → 에이전트가 *행동*으로 오해.
- **frontier**: 요청 주문에만 행동·surface된 sibling에 안 씀.
- **대책 = L6 reads-only**: eplan 피드백 문구를 "**관련성 CHECK용 read·요청에 없으면 행동 금지**"로(현 "decide which records the request covers"가 행동 유도) + eplan은 sibling read 충족 시 deny 해제(행동 요구 아님). 부분효과(1/2=stochastic 잔여).

### 3d. non-eplan 노이즈 (t38·t43·1/2·재분류)
- **t38**: cancel **reason** "no longer needed"(gold) vs "ordered by mistake"(arm2) = 유효 사유 중 택1이 gold와 불일치. tr1은 modify(오행동). **eplan harm 아님** — 사유 underdetermined(둘 다 정책상 유효) = **노이즈/learn 잔여**. (정책이 canonical reason 강제하면 gate 가능·미검.)
- **t43**: modify_user_address 값이 arm2 tr1서 "1427 W Belmont Ave"(gold 943 Maple·오값). 1/2 = **주소-노이즈**(eplan harm 아님) → L3 origin-prov 또는 잔여.
- ⇒ **"eplan HURTS 6"의 실체 = robust harm 3(t58·83·69) + 부분 1(t32) + 노이즈 2(t38·43)**. 집계가 과대계상했음(ablation nt2도 노이즈 잔존).

## 4. 우선순위·예상복구·검증 (무료 先·[[09]])
| 레버 | 대상 | 예상 | 상태 | 검증(무료우선) |
|---|---|---|---|---|
| L1 filter dotted-path | 55·59·101·102 | +3~4 | ✅확증 | done |
| L5b examined-safe | 58(+69·83 일부) | +1~3 | 구현됨 | arm2+examined-safe probe |
| L2 GET-forcing+L1 | 83·92·112·83·69 | +3~5 | 설계 | 로컬 ledger unit → probe |
| L4 fexec-variants | 52·77·110 | +2~3 | 설계 | 로컬 variant-record unit → probe |
| L3 origin-prov | 96(+43?) | +1~2 | 설계 | turn-order unit(확증) → probe |
| L7 precondition gate | 21·38 | +1~2 | 부분존재 | A2 gate_spec → 오프라인 |
| L6 reads-only | 32 | +0~1 | 설계 | 피드백문구+probe |
| L8 prov 강도 | 6·46·55 | 확정대기 | rescue arm | arm 결과 依 |
| L9 set-op | 108 | +0~1 | 설계 | 로컬 unit |

**핵심 순위**: (1) examined-safe 검증(구현됨·즉시) (2) GET-forcing+filter(harm-mode II·최대 복구·주문선택 6+) (3) fexec-variants(변형 3) (4) origin-prov(주소).

## 5. learn/scale 잔여 (scaffold 금지 경계·[[05]])
- **의미적 변형매칭**(94 "다른 구성"·23·7 미명시·52 "버드워칭"의미분): 명시속성/극값 아니면 결정불가 → **underspec은 ASK 안전낙하**, 진짜 의미매칭은 learn/scale(7B 학습 날개). scaffold로 닫지 말 것(트릭).
- **의도/행동 이해**(108 return↔cancel·77 exchange↔신규주문): 일부 precondition gate 방어·핵심은 intent=learn/scale.
- **속성 underdetermined**(38 reason): 정책이 canonical 강제 안 하면 잔여.
- frontier도 이 영역서 실패(~16%) = scale가 닫는 지점([[46]] crossover).

## 6. Open / 미검
- **prov-rescue 확정**: A1-nt2 + COMP−present+eplan+rescue arm(진행중) → t6·46·55 rescue 인과·L8 확정.
- **examined-safe Δspurious**: eplan-help 8(21·23·33·42·52·94·108·110) 무회귀 확인 필수(과-완화 금지).
- **GET-forcing 과-read 비용**: 전수 GET이 over-read/latency 유발 안 하는지(설계 §4d over-ask tradeoff 동형).
- **L4 fexec-variants**: variant record가 order record와 구조 달라 _field_values 점경로 재검 필요.
