# A2 Role-Sourcing — 설계 (2026-06-23·게이트 분류의 *출처*를 cost-efficiency 실험으로 결정)

> 진입: [05-fixed-vs-variable] · [A2_GENERALIZATION_DESIGN](A2_GENERALIZATION_DESIGN_2026_06_23.md)(S1 정정 포함) · 상위 [RULE_LEVER_COST_EFFICIENCY_PROGRAM](RULE_LEVER_COST_EFFICIENCY_PROGRAM_2026_06_22.md).
> 사용자 지시(2026-06-23): "A2도 노골적 도메인특화 금지·규칙 일반화·**행동탐지도 설계에 넣고 비교**·**실험으로 비용효율 검증**." 본 문서=구현 전 설계(리뷰 대상).

## 0. 문제 & 결정원칙
게이트(G1-G5)는 도메인마다 각 도구의 **role**을 알아야 함: write·user-scoped·auth·handoff·owned-entity·ownership-path·precond-status. **그 role을 *어디서* 얻는가**가 핵심 질문.
- **원칙(사용자 정정 반영)**: 도메인 지식은 세 곳에 살 수 있다 — (a)도구 정의[저자 선언] (b)A2[명시·감사가능] (c)scaffold[추측]. **(c) scaffold 추측 = 금지**(명명가정을 고정엔진에 박음·깨져도 안 보임=silent false-block). (a)구조계약 *읽기* = 정당(추측 아님). (b) 환원불가 사실은 A2에 *최소·명시*.
- **결정방식**: 어떤 role을 어떤 방법으로 소싱할지 = **단정 아닌 *cost-efficiency 실험*으로**(§4-5).

## 1. role별 도출가능성 (1차 판정)
| role | 구조신호(읽기) 존재? | 1차 판정 |
|---|---|---|
| owned-entity (order vs product) | ✅ detail-getter 반환모델에 owner 필드 有/無 | **M1 구조도출**(검증완료·추측아님) |
| user-scoped | ✅ 인자에 owner-id 보유 | **M1 구조도출** |
| ownership-path | ✅ owner-id→`get_<e>_details`→owner_field | **M1 구조도출** |
| handoff | ✅ 프레임워크 표준 도구(transfer_to_human) | **M1 구조도출** |
| auth(분류) | — | **제거**(S4: 인증=identity가 grounded·분류 불필요) |
| **write 여부** | ❌ tau2 미선언·read/write 둘다 객체반환 | **★실험 포크**(M0/M2/M3/M4) |
| **precond-status** | ❌ 이름에 status어 있으나 modify_pending_**address**=미검사(이름파싱=오분류) | **★실험 포크**(M2/M3) |
- owner_field 이름(`user_id`) = M1의 유일 가정 → A2 기본값 1개(override 가능·grep 가시).

## 2. 소싱 방법 (정의·[[05]]평가·비용 프로파일)
| 방법 | 정의 | [[05]] | build | transfer(새도메인) | 도메인-특화 surface | 실패양상 |
|---|---|---|---|---|---|---|
| **M0 naming-guess** (baseline·기각후보) | scaffold가 이름 prefix로 write 추측 | ❌위반(엔진에 명명가정) | ~0 | ~0 | 0(숨음) | **silent 오분류→false-block** |
| **M1 structural** | 도구의 *선언된* 계약 읽기(반환모델·인자·표준도구) | ✅최선 | ~0(일반코드 1회) | ~0(자동) | 0(읽기) | 규약위반시(owner 필드명 다름)→override |
| **M2 minimal-A2** | write-set·precond를 A2에 *명시* | ✅(A2=도메인사실·감사가능) | 인간 N줄 | 인간 N줄/도메인 | N(grep 가시) | 도구추가시 갱신누락(가시) |
| **M3 behavioral-probe** | 오프라인: sandbox DB서 도구실행→DB-diff면 write·wrong-status면 precond | ✅(A2=0·엔진=일반 프로브) | 프로브코드(1회·일반)+도메인당 compute | 자동(프로브 실행) | 0 | 인자커버리지(gold 미사용 도구=미분류)·side-effect |
| **M4 LLM-classify** (A2-생성기) | frontier가 도구docstring 읽고 분류(오프라인·ATA류) | △(생성A2·on-prem 데이터반출 우려·別논문) | LLM 1콜 | LLM 1콜 | 0(생성) | LLM 오분류·frontier 의존 |

## 3. role별 방법 적용 + 실험 포크
- **확정(비실험)**: owned-entity/user-scoped/ownership/handoff = **M1**(구조). auth = **제거**(S4). owner_field = A2 기본값.
- **★실험 대상(write·precond의 출처)**: **M0(baseline)·M2·M3·(M4 선택)** 를 *동일 게이트*에 꽂아 cost-efficiency 비교.
  - 가설: M1로 다 안 되는 잔여(write/precond)에 대해 M2(최소명시)와 M3(행동탐지)가 *정확도 동률*이면 → 비용(build·transfer·robustness)이 낮은 쪽 채택. M0는 정확도서 탈락(오분류 실증)이 예상.

## 4. cost-efficiency 메트릭 (생애주기·[RULE_LEVER] 정합)
각 방법 × (retail/airline/banking)에 대해:
1. **정확도**: write-set 분류가 oracle과 일치? precision/recall. (oracle=§5.)
2. **다운스트림 harm**: 그 분류로 게이트 돌렸을 때 **false-block rate**(옳은 write 차단)·pass/compliant Δ. = 오분류가 실제 해를 끼치나.
3. **build 비용**: 코드 LOC + 인간-분 + compute-초 (도메인당).
4. **★transfer 비용(⑤일반화)**: *새 도메인*(airline/banking)을 0서 분류하는 비용 = 인간노동 or compute. (연구 핵심지표.)
5. **robustness/maintenance**: 도구 rename/add 시뮬 → 깨지나·*가시적*으로 깨지나(silent vs visible).
6. **surface/감사성**: grep으로 보이는 도메인-특화 바이트.
- knee = 정확도 충족 中 생애비용 최소 방법. (M0가 정확도 미달이면 cost 무관 탈락.)

## 5. 실험 설계
- **oracle(ground truth) = 전수 행동탐지**: 모든 write-후보를 *모든 gold 궤적의 실인자*로 sandbox DB서 실행→DB-diff. (gold가 도구를 실제 호출하니 인자 확보·tau2 task data=신뢰.) = write의 객관적 정의. precond oracle = wrong-status 타깃에 실행→에러여부.
  - oracle 미커버(어떤 gold도 안 부른 도구)는 별도 표기(M3의 커버리지 한계 노출).
- **arm**: 각 방법으로 write/precond-set 산출 → §4.1 정확도(vs oracle) + §4.2 다운스트림(게이트 1-trial retail로 false-block/pass) + §4.3-6 비용 기록.
- **도메인**: retail(검증된 hand-list=2차 oracle)·airline·banking(전이 비용 실측). 3도메인 = M0의 명명-깨짐, M2의 인간비용, M3의 커버리지를 *교차도메인*서 드러냄.
- **산출**: 방법 × 도메인 × 메트릭 표 → knee(채택방법). = "scaffold-추측은 정확도/robustness서 탈락·M2 vs M3는 transfer비용 vs build복잡도 트레이드오프" 실증.

## 6. Step-by-step build + 검정 (구현 단계·리뷰 후)
- **S0** 본 설계 리뷰(현재).
- **S1'(정정)** `tool_roles.py`서 **이름-기반 write/auth 추측 제거**, M1 구조도출만 잔존(owned-entity/user-scoped/handoff/ownership). 검정: roles==hand-list(구조부분만)·`grep prefix-list in tool_roles=0`.
- **S2** 소싱 인터페이스 `write_source(method, domain)→set`: M0/M1n/a·M2(A2 파일)·M3(probe)·M4. 검정: 각 방법 실행·oracle 대조.
- **S3** oracle 프로브 `write_oracle.py`(gold 인자·DB-diff). 검정: retail write-oracle == 검증된 hand-list(7개).
- **S4** G1 auth 일반화(grounded-identity)·auth 분류 제거. 검정: airline 런타임 auth 작동.
- **S5** cost-efficiency 실험 드라이버 → §4 표. 검정: 표 산출·M0 오분류 수치.
- **S6** knee 방법 채택 → gate.json 최소화 확정·전이실증(신규 하드리스트 0 또는 최소-명시).
- 각 S = 작은 단위·검정 통과 후 다음([[03]]).

## 7. [[05]] 정합 & 정직 tradeoff
- M1(구조읽기)·auth제거 = [[05]] 강화(엔진 추측 0). M0 = 위반(기각·단 baseline 측정으로 *실증* 기각).
- write/precond = 환원불가 도메인사실 → M2(A2최소·명시) 또는 M3(행동탐지·A2 0). **둘 다 [[05]] 합치**(A2-사실 vs 일반-엔진프로브). **어느 게 나은지는 비용으로 결정**(사용자 지시) — assert 아닌 실험.
- 정직: M3는 "A2 0"이 매력적이나 build복잡도+커버리지+side-effect 비용이 숨어있음 → §4가 그걸 *드러냄*. M2는 단순·감사가능하나 transfer마다 인간노동. M4는 on-prem 데이터반출 제약(우리 배포전제와 충돌 가능)→별 평가.
- ⚠️ owner_field 이름 가정(M1) = A2 기본값+override·grep 가시(숨은 가정 방지).

## 8. 미해결(리뷰 질문)
- write oracle을 "DB-diff"로 둘 때, gold 미커버 도구 처리(보수적=non-write로? 위험). 
- M3 probe의 side-effect 격리(sandbox DB 깊은복사 비용).
- precond를 굳이 분리할지(=write∩status-gated) vs G5를 oracle-probe로만.
- M4(LLM-A2생성)를 이번 실험에 넣을지(on-prem 충돌이라 참고용만?).
