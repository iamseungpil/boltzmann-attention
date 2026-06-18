# Closure-Payoff 실험 설계 (2026-06-18) — "닫힌 유한 규칙-기저"가 ICLR 기여인지 가르는 실험

> **자립 문서**(다른 세션 리뷰용·컨텍스트 불요). 상위 = `RELWORK_AND_DIRECTION_2026_06_18.md`(§1.5 핵심주장·차별)·`GENERATOR_ALGEBRA_DESIGN_2026_06_17.md`(closure 논증)·`ma/M_A_RESULTS.md`(§21·§23A 씨앗). 메모리 = `00-thesis`·`41-relwork-rivals-whitespace`·`06-NOW`.

## 0. 왜 이 실험인가 (한 문단)
정독 종료(2026-06-18) 판정: 우리 유일한 깨끗한 신규 = **"tool-use 계획이 *닫힌(closure-justified) 유한 생성원 기저*로 환원되고, 그 *규칙 추상화*를 작은 모델이 학습·전이"**. 단 **"전이" 자체는 신규 아님**(schema-guided DST·ToolLLM 선점)·**closure 수학도 재적용**(Codd 1972·Böhm–Jacopini 1966). ⇒ ICLR-worthiness는 **"닫힘(완전성)이 *측정 가능한 전이 이득*의 원인"임을 통제 실증**하느냐에 달림(`RELWORK §6`). 명제·프레이밍만으론 taxonomy 논문 = 부족. **이 실험이 그 payoff를 만들거나 죽인다.**

## 1. 가설 (falsifiable) + 반증조건
**H (주가설)**: 작은 모델을 **닫힌 완전 생성원 기저**(flow P1-P9 + content ops)로 학습하면, **(a) open-set 모방**(ToolLLM류·기저 없음)·**(b) 비폐포/불완전 기저**(STAR류·생성원 누락) 대비 **cross-domain·cross-bench 전이·커버리지·sample-efficiency가 우월**하고, 그 이득은 **closure(완전성)에 귀속**된다.

**반증(이 중 하나면 H 기각)**:
- 불완전 기저가 완전 기저만큼 전이함 → closure 무관(완전성이 이득 아님).
- open-set 모방이 cross-bench서 closed와 동급 → 기저 구조화 무가치.
- 이득이 diversity/data/compute 교란으로 설명됨(closure 귀속 실패).

## 2. 두 비교 (상보적)
**비교 1 — 완전성 ablation (closure-내부·*가장 깨끗한 귀속*)**: 모든 것 고정(모델·offload·데이터량·diversity), **기저 완전성만** 변화.
- 씨앗 = `M_A_RESULTS §23A`(이미 함): 5-op(불완전·substitute 누락)→τ² 역전이 0.03 / 7-op(완전)→0.44. = "누락 생성원이 §17 음성 원인" 박제. **이 실험은 이를 flow+content 전 기저·다벤치로 일반화.**
- 설계: 완전 기저에서 생성원을 *체계적으로 제거*(flow에서 P_k 제거·content에서 op 제거) → 그 생성원을 요구하는 held-out 도메인서 전이 붕괴하나. **예측: 제거된 생성원을 쓰는 도메인서만 붕괴**(국소적·closure가 그 생성원을 필요로 함을 증명).

**비교 2 — regime 비교 (vs rival 패러다임)**: 동일 base·동일 예산.
- **A_closed**(우리): 닫힌 기저 라우팅(flow native tool_calls + content `resolve_selection`) + 결정론 offload + ABox.
- **B_open**(ToolLLM류·`2307.16789` 충실 재현): 기저 추상화 *없이* 벤치별 raw native tool-call 모방·도구 schema in-context로 일반화. resolve_selection·op 어휘 없음.
- **C_nonclosed**(STAR류·`2010.11853` 충실 재현): 절차를 *데이터로 제공*(per-task 순서도/schema) + 모델은 따라가기·이탈은 학습 fallback. (= 규칙이 weight 아닌 data.)
- **예측**: cross-bench 전이·sample-eff에서 A_closed > {B_open, C_nonclosed}.

## 3. 지표 (전부 결정론·LLM-judge 금지·[[10-roles-deterministic]])
- **전이 보존율** = held-out / in-domain 공식지표 비(per-bench 개별·**집계 평균 금지**·[[20-proven]] 규율). 벤치 native: τ² pass^1·SOPBench official success·TaskBench graph-F1·Synth round-trip.
- **★orphan / 신-생성원 출현율** = held-out *벤치*서 필요 연산 중 기저 *밖*인 비율. **closure 예측 = ~0**(census가 τ²서 orphan=0 이미 확인·`PRIMITIVE_COVERAGE_MATRIX`). held-out 벤치로 확장 = closure의 직접 시험.
- **sample-efficiency** = 성능 vs 학습데이터량 곡선. 예측: closed가 *적은 데이터로* 전이 도달(완전 기저=압축).
- **커버리지** = 기저 재조합만으로 풀리는 task 비율.
- **비용** = 토큰·USD(F1 장부·`EXPERIMENT_DESIGN §1.6`).

## 4. closure-귀속 통제 (이득이 교란 아님 보장)
- **동일 base 모델·동일 총 compute·arm 간 데이터량 matched.**
- **결정론 offload를 *세 arm 모두* 동일 제공**(또는 별도 ablate) → closed>open 차이가 "우린 offload 있고 쟨 없다"가 아니게. **closure 귀속은 비교 1(완전성 ablation·offload 고정)이 담당**(가장 깨끗).
- **diversity matched**(표현/구조·[[12-diversity-required]]) → 전이 차이가 surface-mapping 아티팩트(§17-18) 아니게.
- **per-domain/bench 분기 0**(grep `if domain` = 0·CI).
- **contamination 0**: held-out 벤치·도메인은 학습서 제외.
- **scaffold vs weight 분리**: flow 전이가 결정론 scaffold가 나르는지(전수본 adapter held-out≈0·`SOP:583`) vs 학습 weight인지 *분리 측정*(adapter-only arm) → "*학습된* 규칙추상화 전이" 주장의 정직 근거.

## 5. 다벤치 전이 매트릭스 (cross-bench = schema-guided DST와 차별점)
| 학습 | held-out(도메인) | held-out(★벤치) |
|---|---|---|
| SOPBench(일부 도메인) + TaskBench + Synth | SOPBench 잔여 도메인 | **τ²(retail·airline)·SOP-Bench(Amazon)** |
- **cross-domain 셀** = schema-guided DST도 함(차별 약). **cross-bench 셀**(다른 벤치·포맷·task 패러다임) = DST는 한 포맷 내라 *안 함* → **여기가 우리 고유**. 매트릭스가 "닫힌 기저는 패러다임 횡단도 전이"를 보이면 ①이 capability 기여로 섬.
- ABox-swap: A_closed는 unchanged·catalog/gate_spec(ABox)만 교체. 재학습 0.

## 6. 빌드 단계 (증분·각 단계 실측·기존 자산 재사용)
- **S0 (씨앗 확인)**: `§23A` 5op/7op 재현 = 완전성 ablation 최소사례 작동 확인(이미 양성·재실행으로 harness 검증).
- **S1 (완전성 ablation·비교1)**: flow+content 전 기저서 생성원 1개씩 제거 arm → held-out 도메인 전이 붕괴 국소성 측정. **offload·diversity·데이터 고정.**
- **S2 (baseline 구현·비교2)**: B_open(ToolLLM류)·C_nonclosed(STAR류) 충실 재현(논문 방법 그대로·`RELWORK 버킷 가/마` 인용). 공정 통제(§4).
- **S3 (전이 매트릭스)**: 세 arm × 다벤치 매트릭스 → 전이 보존율·orphan율·sample-eff.
- **S4 (귀속·autopsy)**: 이득이 closure에 귀속되나(diversity/data/offload 통제 후 잔존?)·scaffold vs weight 분리.
- 자산: 기존 LoRA(SOP·TB·Synth)·`synth_to_nativefc`·resolve wiring·e2e harness(`real_e2e_base.sh`)·census(`tau2_primitive_census`).

## 7. ★자가심사 (리뷰 안건)
- **thesis-정합**: 학습=닫힌 규칙-기저(도메인일반)·offload=decidable·ABox=swap·e2e=학습 TBox. ✅
- **치팅 방어**: orphan율·완전성 ablation은 결정론 census·per-domain 분기 0. 이득 귀속은 통제(§4)로. real 도구 미대체(offload=인자계산·`INTEGRATED_TBOX §3b`).
- **선행 재사용**: open/non-closed baseline = ToolLLM/STAR 방법 *그대로*(재발명 아님·[[41]] directive).
- **정직 scope**: closure = transactional slice 한정·층B 상대닫힘(`GENERATOR_ALGEBRA`·`ALGEBRAIC_DERIVATION_CLOSURE`). "무제한 닫힘" 주장 금지. 층B(policy-gate)는 결정론 집행이라 학습-기저 전이 시험 대상 아닐 수 있음 — 리뷰서 확정.
- **무엇이 H를 죽이나 명시(§1)** = 정직.

## 8. 리뷰 받을 미결 질문
1. **C_nonclosed 조작화**: STAR식 "절차=데이터+follow"가 깨끗한가, 아니면 "불완전 기저"(비교1과 겹침)로 통합? 둘 다 둘지.
2. **공정 예산**: open-set(ToolLLM)은 본래 대량 API 데이터로 큼 — 동일 데이터량 통제가 공정한가, 아니면 동일 compute? matched 축 확정.
3. **offload를 baseline에도 줄까**: 주면 closed-vs-open이 순수 기저효과·안 주면 시스템 전체효과. 어느 비교가 ICLR 논지에 맞나(완전성 ablation은 어차피 offload 고정).
4. **cross-bench 지표 정합**: 벤치마다 native 지표가 달라(pass^1 vs F1 vs success) 전이 보존율을 어떻게 벤치-간 비교 없이(집계금지) 보고할지.
5. **sample-eff 곡선 범위**: 어디까지 data-scaling 할지·소형 조건부(§22 width=scale 교훈) 반영.
6. **scaffold/weight 분리가 ①을 약화?**: flow 전이가 scaffold면 "*학습된* 규칙추상화 전이"는 content-op로 좁혀짐 — 그래도 ICLR 충분한가(범위 정직).

## 9. 성공/실패 판정 (사전등록)
- **성공(ICLR 간다)**: 완전성 ablation이 국소 붕괴 보임(closure 필요) + A_closed가 cross-bench서 B_open·C_nonclosed 초과 + 이득이 통제 후 closure 귀속.
- **부분(워크숍/전문 venue)**: 이득 있으나 modest·cross-domain만(cross-bench 약)·일부 confound 잔존.
- **실패(taxonomy 회귀)**: 불완전≈완전 or open≈closed → closure 무가치 → 명제 재구성 필요.

> 권위 = `RELWORK_AND_DIRECTION_2026_06_18.md`·`GENERATOR_ALGEBRA_DESIGN_2026_06_17.md`·`ma/M_A_RESULTS.md §21·§23A`·`PRIMITIVE_COVERAGE_MATRIX_2026_06_15.md`(census·orphan=0). 불변 = [[03-anti-drift]]·[[10-roles-deterministic]]·[[12-diversity-required]].
