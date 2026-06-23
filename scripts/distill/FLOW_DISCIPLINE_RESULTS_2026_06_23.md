# Flow-discipline 밤샘 결과 (2026-06-23 아침) — cross-domain × scale, bench + compliant-pass

> 진입: [FLOW_DISCIPLINE_SCAFFOLD_DESIGN](FLOW_DISCIPLINE_SCAFFOLD_DESIGN_2026_06_22.md) §6 측정 실행 결과. 데이터: `data/simulations/ours_*` + `on_n*_floor_retail`. 도구: `morning_tables.py`. nt=1(gated arms)·nt=3(floor 기존). user-sim gpt-4.1.

## Table 2 — Retail flow-discipline × scale (bench/compliant pass^1)
| arm | 7B | 14B | 32B-int8 |
|---|---|---|---|
| floor | 0.189/0.130 | 0.468/0.404 | 0.547/0.491 |
| g14 (G1-G4) | 0.170/**0.170** | 0.416/**0.416** | 0.531/**0.531** |
| g15 (+G5) | 0.191/**0.191** | 0.440/**0.440** | 0.573/**0.573** |
| g15+retry | 0.124/0.124 | 0.343/0.343 | 0.583/0.583 |
| g5-only* | – | – | 0.568/0.513 |
| g5+retry* | – | – | 0.564/0.491 |

\* G5-isolation(G1-4 OFF). (gated arm = nt=1·floor = nt=3 → 32B는 GPU0 nt=3 denoise 진행중으로 확정 예정.)

## Table 1 — Cross-domain floor × scale (bench/compliant pass^1)
| domain | 7B | 14B | 32B-int8 |
|---|---|---|---|
| retail | 0.189/0.130 | 0.468/0.404 | 0.547/0.491 |
| airline | 0.160/0.040‡ | 0.260/0.080‡ | 0.300/0.020‡ |
| banking | 0.011/0.011 | 0.010/0.010 | (진행중) |

‡ airline compliant = auth-모델 caveat(bench 신뢰). banking floor ≈ **1%** (id-fabrication prior → 스캐폴드 강한 동기).

## 실패-클래스 census (retail·scaffold-addressable Δ)
| arm/model | n | pass | elig/wrong-tool | loop | no-write |
|---|---|---|---|---|---|
| 7B floor | 341 | 64 | 58 | **113** | 124 |
| 7B g15 | 110 | 21 | 2 | 38 | 51 |
| 7B g15+retry | 105 | 13 | 1 | 44 | 63 |
| 14B floor | 339 | 160 | 32 | 44 | 37 |
| 14B g15 | 109 | 48 | 1 | 9 | 14 |
| 14B g15+retry | 105 | 36 | 0 | 13 | 21 |
| 32B floor | 342 | 187 | 25 | 20 | 14 |
| 32B g15 | 110 | 63 | 0 | 6 | 5 |
| 32B g15+retry | 108 | 63 | 0 | 3 | 9 |

## ★핵심 결론
1. **게이트 = compliant-pass를 bench와 동일하게 만든다(위반 0)**: floor는 bench>compliant(위반 gap·32B 0.056), gated(g14/g15/g15retry)는 **bench==compliant**(런타임 게이트가 auth/confirm/ownership 위반을 0으로). = 배포-실제 지표서 게이트의 핵심 가치.
2. **G5(precondition-steering)는 전 scale서 compliant-pass↑** (g14→g15): 7B +0.021·14B +0.024·**32B +0.042**. 32B서 최대(addressable이 pass로 전환되는 지점).
3. **★retry-controller는 scale-게이트 음성**: 7B −0.067·14B −0.097(해침!)·32B +0.010(중립~약양). = §35c "retry=잘못된레버" *정량 확증*·**소형엔 retry 빼야**. ⇒ 배포: **소형=g15(G1-G5·retry 없음)·32B만 +retry**.
4. **★최적 deploy = g15**: floor compliant→g15 compliant = 7B 0.130→0.191(+0.061)·14B 0.404→0.440(+0.036)·**32B 0.491→0.573(+0.082)**. 전 scale 양성·deployment 지표(compliant)서.
5. **scaffold가 addressable 클래스를 전 scale서 닫음**(census): eligibility 7B 58→2·14B 32→1·32B 25→0 / loop 7B 113→38·14B 44→9·32B 20→6. **그러나 pass 전환은 32B서 주로**(7B는 닫아도 capability 벽서 실패·pass 0.188→0.191 무변). = §1b "addressable ~25%·나머지 capability" *확증*.
6. **H1 확증(addressable는 scale↑서 수축)**: floor loop 113→44→20·elig 58→32→25. 큰 모델일수록 그런 실수 덜 함. *그러나* scaffold의 pass-lift는 32B서 최대(닫은 게 success로 전환되는 sweet-spot).
7. **cross-domain**: airline가 retail보다 어렵고 평탄(scale 둔감)·banking floor≈1%(거의 전부 실패=id-fab prior). = 도메인별 난이도 차 큼·banking은 스캐폴드 없이는 불가.

## 정직 caveat
- gated arm nt=1 vs floor nt=3 → pass^1 비교 약노이즈. **GPU0 nt=3 denoise 진행중**(on_n32int8_g1*_retail_t3)으로 32B 행 확정.
- airline/banking compliant = compliance auth-establishment이 lookup-tool(retail형)이라 부분만(bench 1차신뢰). = Stage-2 prereq(airline user-id auth·banking gate.json·compliance auth 일반화).
- 32B banking floor = 진행중(마지막 arm).
- ABLATION 미포함(추후): false-block 직접수치(floor-pass였다 깨진 task)·#4 loop→success vs →다른실패 전환.

## ★★★★★★ 32B capability-gap 전수 분해 (2026-06-23·`/tmp/gapcensus.sh`·`/tmp/decompose.sh`)
**gap = gpt-4.1 pass ∧ 32B floor 3시행 fail = 15 task.** 실제 write 인자로 정밀 분해:
- **① wrong-ORDER 선택 (7·47%·최대)**: 여러 주문 中 *엉뚱한 주문*에 작업. T71/72 gold #W5270061(DC주문)←32B #W5782623·T41 #W4082615←#W9583042·T101/102 주문 뒤바꿈·T29/74. = "DC로 간 그 주문"/"시계 두개 주문" *식별*(상세 읽고 매칭) grounding 실패.
- **② wrong-ACTION 유형 (3·20%)**: 의도→도구. T34 주소바꿔야 하는데 아이템수정·T38 취소해야 하는데 수정·T85 수정해야 하는데 교환.
- **③ operand/verbatim (2·13%)**: T17 "123 Elm **Street**"←"123 Elm **St**"(축약)·T8 틀린 variant 2회.
- **④ 예산추론+루프 (2·13%)**: T36/37 "예산부족→어느 아이템 빼면 맞나" 추론 못 하고 빈 new_item_ids 루프.
- **⑤ 과행동 (1·7%)**: T62 어드버서리얼·gold 무행동인데 modify+cancel.
- **★2차 증상=에러루프/복구실패**: 틀린 선택 후 *같은 행동 4-6회 반복*(nerr 4-6·T34/38/41/71). 루프=근본 아닌 증상(wrong선택+재계획실패).
- **쉬운 요약**: 32B의 gap 핵심 = **"여러 주문 中 어디에·어떤 행동을" 못 고름(grounding+intent매핑) + 한번 틀리면 반복(복구불능)**. operand는 소수(13%).
- **★G5 무익 이유 확정**: G5는 *도구-status* 오류(delivered인데 modify) 잡는데, 실제 gap=wrong-ORDER+wrong-ACTION+no-recovery → G5 적용지점 아님. ⇒ **레버 재타깃**: 주문-disambiguation(grounding: 상세읽고 매칭)+복구(에러후 재계획)·intent→action. precondition 아님. = learn/scale 또는 grounding/recovery scaffold.

## ★★★★★ 정밀 원인 = 큰 에이전트 비결정성 (2026-06-23·`/tmp/precise.sh`·`/tmp/trajdiff.sh`)
- **차이가 큰 건 노이즈가 작아서가 아니라 *에이전트 비결정성이 커서***: floor 자기 3-trial = 0.596/0.482/0.561 = **spread 0.114**(vllm 배칭·conc8·seed無·17턴 누적). 단일 trial pass 신뢰불가 ±0.06.
- **g14-pass∧g15-fail 21 task 전부 G5 0발동인데 에이전트 궤적 다름**(예 T17: g14 get_order_details 거침/PASS vs g15 건너뜀/FAIL). = pass 차이=100% 에이전트 RNG·G5 무관.
- pooled(3-trial): floor 0.547·**g14 0.605**·g15 0.550·g15retry 0.574. g14-g15 0.055=~1.4σ(유의X)·게다가 G5 0발동이라 인과상 G5 불가→런타임/throttle confound.
- **★단 G1-G4 compliant 상승(+0.045)은 진짜**: bench 차이는 노이즈밴드나, compliant 상승=floor 위반율(G2-confirm~0.05)을 게이트가 *결정론적*으로 제거(위반 카운트로 측정·샘플링 아님).
- **방법론**: 현 측정 검정력 부족(노이즈~0.11)→<0.05 레버 탐지하려면 결정론 serve(seed+max-num-seqs1·[[30]]) 필요. 단 G5=0(작은게 아닌 0)이라 결론 강건.

## ★★★★ 전수 궤적 원인 확정 (2026-06-23·`/tmp/census*.sh`) = G5 인과효과 ZERO
**G5(precondition-steering)의 net 인과 = 0 (전 scale·궤적 전수 검증):**
- 발동량: **7B 19회/10task · 14B 0 · 32B 0.** (wrong-tool 에러=소형모델 현상·14B서 소멸.)
- 7B G5-발동 10task를 g15(G5) vs floor(無G5) 동일-task 비교: **도움0·해악0·동일10**(8 fail-fail·2 pass-pass). steer가 도구선택 교정해도 출력 불변.
- ⇒ **"G5 +0.042"(nt1)·"G5 −0.080"(nt3 trial-0) 둘 다 샘플링노이즈·G5 진짜효과=0.** 14/32B의 g14/g15/g15retry 차이=노이즈(G5 미발동·retry 43발동 net+1).
- **메커니즘**: wrong-tool/eligibility=capability *증상*이지 binding 원인 아님. 올바른 도구 짚어줘도 약한모델은 완수능력 없어 실패. = §1b "addressable 닫아도 capability 벽 binding" 정밀확증.
- **★결론: G5 폐기. 유일 진짜 레버 = G1-G4 compliance게이트**(결정론 위반제거→compliant 0.544→0.589). 잔여 gap(32B 0.59 vs gpt41 0.81)=순수 capability(scaffold 불가).

## ★★★ 정정 = 깨끗한 nt=3 (2026-06-23·API-throttle sim 제외)·nt=1 헤드라인 뒤집힘
**이전 nt=1(g15 compliant 0.573·"G5 +0.042")은 노이즈. 깨끗한 nt=3(throttle 제외):**
| 32B arm | bench(clean) | compliant(clean) |
|---|---|---|
| floor | 0.596 | 0.544 |
| **g14 (G1-G4)** | 0.589 | **0.589** |
| g15 (+G5) | 0.509 | **0.509** |
| g15retry (+retry) | 0.596 | **0.596** |
- ✅ **G1-G4(compliance 게이트)=진짜 레버**: compliant 0.544→**0.589**(+0.045)·위반0·bench비용~0.
- ❌ **G5(precondition)=净-음성 −0.080**(g14 0.589→g15 0.509). ★false-block=0인데도 pass↓ → 올바른 write 차단 아님·**steer+매-write get_order_details 추가읽기가 에이전트 흐름 교란**. eligibility 클래스 닫음(census 25→0)이 **pass전환 실패**(§1b capability벽 + 개입비용).
- retry=G5 손실 복구(0.509→0.596). ⇒ **최적 배포=G1-G4만·G5 제외.**
- 교훈: nt=1 task-level 신호 신뢰불가(리뷰 #4)·denoise+throttle필터 필수. (throttle: OpenRouter 429/502/503 → arm당 ~12 sim 비-모델 실패·제외함.)

## ★false-block 검증 (헤드라인 빠진 메트릭·2026-06-23 추가)
- **궤적-수준 false-block(게이트가 *gold-정답 write*를 deny) = 0** (전 arm 7B/14B/32B g15·g15retry). 게이트 deny는 전부 정당: G4 notice(transfer 전 문구)·G1 auth(인증 전)·G5(7B 4건=*비-gold* wrong-tool 차단)·RETRY_LOOP(반복실패). **write-deny가 gold와 일치한 건 0.**
- ⚠️ **task-level 1-trial "false-block"(24-32·net −22) = 전부 샘플링 노이즈**: floor pass-any-3(0.77)를 gate trial-0(0.57)과 비교한 trial-비대칭 + 1-trial churn(~20% 무작위 뒤집힘). **게이트 잘못 아님**(궤적검증 0). = 리뷰 #4("작은n/노이즈→점추정 불신") 적중·신뢰메트릭=궤적수준.
- ⇒ **게이트는 안전**(올바른 행동 0건 차단·#3 NO-GO 조건 통과) + compliant 향상(32B 0.491→0.573). 잔여 gap(0.573<0.82)=capability/operand(§1b·scaffold-addressable~25%). nt=3 denoise(GPU0 진행중)로 pass 수치 확정 예정.

## 다음
1. GPU0 nt=3 denoise 회수 → 32B 행 확정.
2. 32B banking floor 마무리.
3. (Stage-2) airline/banking 게이트 prereq 해결 → cross-domain scaffold transfer + compliant.
4. 이 결과로 coworker 요청서(72/225B) 확정·push — 특히 "g15 best·retry는 32B+" + scale-lift sweet-spot 가설을 큰 모델서 검증.
