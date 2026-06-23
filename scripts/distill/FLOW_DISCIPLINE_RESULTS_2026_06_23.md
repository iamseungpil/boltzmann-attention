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

## ★false-block 검증 (헤드라인 빠진 메트릭·2026-06-23 추가)
- **궤적-수준 false-block(게이트가 *gold-정답 write*를 deny) = 0** (전 arm 7B/14B/32B g15·g15retry). 게이트 deny는 전부 정당: G4 notice(transfer 전 문구)·G1 auth(인증 전)·G5(7B 4건=*비-gold* wrong-tool 차단)·RETRY_LOOP(반복실패). **write-deny가 gold와 일치한 건 0.**
- ⚠️ **task-level 1-trial "false-block"(24-32·net −22) = 전부 샘플링 노이즈**: floor pass-any-3(0.77)를 gate trial-0(0.57)과 비교한 trial-비대칭 + 1-trial churn(~20% 무작위 뒤집힘). **게이트 잘못 아님**(궤적검증 0). = 리뷰 #4("작은n/노이즈→점추정 불신") 적중·신뢰메트릭=궤적수준.
- ⇒ **게이트는 안전**(올바른 행동 0건 차단·#3 NO-GO 조건 통과) + compliant 향상(32B 0.491→0.573). 잔여 gap(0.573<0.82)=capability/operand(§1b·scaffold-addressable~25%). nt=3 denoise(GPU0 진행중)로 pass 수치 확정 예정.

## 다음
1. GPU0 nt=3 denoise 회수 → 32B 행 확정.
2. 32B banking floor 마무리.
3. (Stage-2) airline/banking 게이트 prereq 해결 → cross-domain scaffold transfer + compliant.
4. 이 결과로 coworker 요청서(72/225B) 확정·push — 특히 "g15 best·retry는 32B+" + scale-lift sweet-spot 가설을 큰 모델서 검증.
