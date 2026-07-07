# 잔여 극복 전략 재설계 — scaffold-천장 vs 학습-TBox 전이 vs 경계-지도 (2026-07-07·rev2)

> **rev1 폐기 사유(표류 자기-진단)**: rev1은 τ²-retail 실패 버킷을 보고 **A2-scaffold 레버를 손-저작**했다.
> 이는 [[13]](scale→학습→*최후*scaffold/A2) 순서 역전·[[05]]/[[46]](최소 A2·도메인-일반·moat=배분 method)
> 위반·**R1**(주입→복사=steering, paper1 §5.4서 이미 causal effect ~0로 측정=이 프로젝트서 실패한 기전)이다.
> 목표는 **τ²를 푸는 것이 아니라, 벤치-독립·도메인-독립 scaffold+TBox를 최소 A2로 확립**하는 것(사용자 교정).
> 본 rev2는 그 목표로 재정렬한다.
> **불변**: [[00]] 두 날개(결정론 scaffold + 학습 도메인-일반 TBox·ABox-swap 전이)·[[01]] four-bench TBox·
> [[05]] 고정={TBox+scaffold}·변경={ABox}·[[11]] TBox=학습벤치만·τ²=swap·[[12]] 다양성·[[13]] 학습먼저.

---

## 0. 핵심 재프레임 (한 줄)
**이번 nt=4 포렌식은 *base Qwen + scaffold*였다(학습 TBox 아님·[[01]]). 잔여(변형선택·coverage·⋈)는
scaffold가 present한 정보를 *base가 못 쓰는 것* = 도메인-일반 *스킬* gap.** 따라서 잔여는 **scaffold를 더
얹어 닫는 게 아니라(R1: steering 천장) 학습된 도메인-일반 TBox가 닫는다.** scaffold(present+gate)는 이미
할 일을 다 했다·FIXED·최소-A2. 이게 원래 두-날개 계획이고, rev1은 그걸 우회한 표류였다.

---

## 1. 잔여의 정확한 세-몫 귀속 (포렌식 버킷 → 어느 날개)
CLEAN_NT4_FAILURE_FORENSIC의 버킷을 *닫는 주체*로 재귀속:

| 포렌식 버킷 | 32B/14B | 닫는 주체 | 근거 |
|---|---|---|---|
| compliance(위반) | 0 (이미0) | **scaffold-gate**(done) | 결정론·scale-불변·최소 A2 |
| coverage(상태추적) | 17/16 | **학습-TBox**: state-tracking | TaskBench(data-flow)·SOPBench(control) 프리미티브 |
| 변형선택(criterion) | 21/26 | **학습-TBox**: content-op select | Synth COMPUTE(filter/argmax over options) |
| ⋈ cross-order 참조 | 7/10 | **학습-TBox**: reference/join | Synth cfbsynth(fetch-first/COPY) |
| conditional cascade | (36 등) | **학습-TBox**: control-flow | SOPBench |
| order-total | 6/3 | **scaffold-calc**(집계) *IF* steering 유효 / 아니면 TBox | §5 probe |
| over-action(불가능op) | 4/7 | **scaffold-gate**(precondition·done) | 결정론 |
| over-action(should-not intent) | (소량) | **잔여**(intent·over-block 위험) | 레버 금지·§4 |
| orchestration load(loop/no-write) | 13/17 | **genuine-scale/load** | plan-execute capacity·scaffold 밖 |

**★관측**: 학습-TBox가 닫을 버킷(coverage+변형+⋈+conditional)이 잔여의 **지배부분**이고, 이들은 **정확히
four-bench TBox의 학습 프리미티브**(state-tracking·content-op·reference)다. 즉 **포렌식 잔여 = 학습 타깃과
동형**. rev1이 손-저작하려던 것을 **학습이 도메인-일반으로 설치**해야 한다(그래야 airline/bank로 A2-swap 전이).

---

## 2. scaffold의 역할은 끝났다 — steering 천장 (R1)
- scaffold = **present**(후보·변형·주문·집계 노출) + **gate**(compliance). 둘 다 **결정론·도메인-일반·최소 A2**.
  present는 이미 변형·주문·총액을 노출한다 → **정보는 이미 거기 있다.**
- **R1(prior 확정)**: 주입→복사(steering)의 causal effect ~0(paper1 §5.4). 즉 **더 present/주입해도 base가
  안 쓴다.** 잔여는 "정보 부족"이 아니라 "present된 정보를 쓰는 *스킬* 부족" → **scaffold가 아니라 학습**.
- ⇒ rev1의 R-레버(변형-select 주입·cross-order 주입·order-total 주입)는 **R1 근거로 기각**. scaffold-adding
  종료.

---

## 3. make-or-break 실험 — 학습-TBox의 τ² 전이 ([[13]] 학습 먼저)
**가설**: four-bench 학습 TBox는 state-tracking·content-op·reference를 *도메인-일반 스킬*로 설치하며, 이는
τ²에 **ABox-swap 전이**돼(τ² never-trained) 포렌식 잔여를 닫는다. scaffold(present+gate)는 FIXED·A2 최소.

**설계**:
1. **학습(도메인-일반)**: SOPBench(control-flow)+TaskBench(data-flow)+Synth(COMPUTE+cfbsynth)로 통합 TBox
   학습. [[12]] 표현/구조 다양성 필수(단일템플릿 SFT=표면매핑 역전이 방지·R5). τ² 데이터 **절대 미사용**([[11]]).
2. **전이 측정**: 학습 TBox를 τ²-retail에 **동일 scaffold**(present+gate·regen·A2 그대로)로 e2e 구동. base
   Qwen(현 nt=4)과 **동일 조건 A/B**(모델 가중치만 교체).
3. **판정(포렌식-구동·[[08]])**: 잔여 버킷별 pass 변화 — coverage 17/16·변형 21/26·⋈ 7/10이 **유의하게
   닫히는가**(per-bucket + 공식 pass^1..4 same-k). 전이 스킬이 base가 못 쓰던 present-정보를 쓰게 하는가.
4. **overfit 방어(R2)**: 학습=SOPBench/TaskBench/Synth·**τ²는 held-out 도메인** → τ² pass↑는 정의상
   전이(τ²-특화 불가). A2-cost(R3) 불변(스킬 학습·손-저작 아님).

**사전등록(R4/R5·[[03]])**:
- **성공**: 전이 TBox가 지배 잔여(coverage+변형+⋈)를 유의하게 닫고 same-scale/frontier-gap 축소 → thesis
  (학습된 도메인-일반 스킬이 전이).
- **부분**: 일부 버킷만 닫힘 → 닫힌 것=전이-스킬·안 닫힌 것=§4 경계로 귀속.
- **실패**: 전이가 잔여를 못 닫음(역전이·flat) → §4 contingency(경계-지도+fleet). **강요 금지.**

---

## 4. contingency — 전이 실패 시의 "다른 방안" (R5·정직)
R5가 옳다: 학습-전이는 불안정(monolithic SFT 역전이·strict서 crossover flip 실측). **학습된 TBox도 잔여를
전이로 못 닫으면**, 그건 scaffold로도 학습으로도 안 닫히는 **genuine scale/capacity 경계**다. 그때:
- **"small=large" 강요 폐기.** 기여를 *닫기*에서 ***특성화(map)*로 전환**:
  - **능력별 경계 지도**: (a)결정론-scaffold가 닫음(compliance·given-spec) (b)학습-TBox가 전이로 닫음(닫힌
    버킷) (c)genuine-scale 잔여(안 닫힌 버킷·load). = paper1의 cost×capability×lever 지도(이미 프레임).
  - **fleet**: genuine 잔여만 frontier로 escalate(닫는 게 아니라 우회)·on-prem 기본.
- **moat 불변**: 결정론 게이트(compliance 보장·scale-불변) + cost-knee + 배분 method([[46]]). **pass-parity
  없이 성립** — 이번 forensic이 이미 그걸 지지(frontier-pass 아래여도 compliance는 우리만 보장).
- ⇒ 실패도 논문이 된다: "무엇이 scaffold-decidable·무엇이 learn-transferable·무엇이 scale-bound인가"의
  **정직한 경계 census**가 곧 기여.

---

## 5. 유일 잔존 scaffold 작업 = order-total steering probe (GO/NO-GO·무료-급)
scaffold-천장(R1)을 **이 클린 런서 재확인**하는 최소·결정적 probe:
- **가장 단순한 steering**: order-total(집계 숫자)을 READ-증강으로 주입 → 모델이 NL 리포트에 복사만 하면 됨.
- **GO/NO-GO**: 주입해도 67·68류 order-total NL-fail이 **안 닫히면** → **steering 확정 사망** → 모든
  scaffold-주입 레버 기각 확정·학습만이 유일 경로. 닫히면 → 순수-집계-복사는 여전히 유효(천장이 "자기 판단
  대신 신뢰"에만 있음)·단 변형/⋈ 주입은 여전히 리스크(R1의 서열).
- 이 하나만 A2 `calc_specs`에 추가(기존 sum 엔진)·A/B smoke. **scaffold 레버는 이것으로 끝**(더 안 얹음).

---

## 6. 실험 순서·검증
1. **order-total steering probe**(§5·무료 A/B·GO/NO-GO scaffold 천장 확인).
2. **four-bench TBox 학습 상태 점검/학습**([[01]][[12]]·τ² 미사용).
3. **TBox τ² 전이 measure**(§3 make-or-break·base와 A/B·같은 scaffold).
4. **분기**: 닫으면 thesis 확정 / 못 닫으면 §4 경계-지도+fleet로 정직 전환.
- 검증: 공식 compute_metrics pass^1..4 same-k·per-bucket 포렌식([[08]])·[[05]] census(retail 리터럴0 +
  airline/bank A2-swap 등가)·[[09]] 무료 먼저·유료 승인.

## 7. 불변·정직 경계 (R1–R5 반영)
- **R1**(steering=0): scaffold-주입 레버 기각·§5 probe로 재확인. **R2**(overfit): 학습=타벤치·τ²=held-out
  swap. **R3**(A2-cost): 스킬 학습이라 A2 불변·assembled=이동표적 아님. **R4**(over-block): G-레버 미추가로
  moot. **R5**(learned residual 불안정): 전이 실패를 §4 경계로 열어둠·"학습으로 닫힘" 단정 금지.
- **over-claim 금지**: 학습-전이 성공도, genuine-scale도 단정하지 않는다 — **측정으로 결정**. 실패면 경계-지도가
  기여(강요 아님).
- **표류 방지([[03]])**: 목표=벤치-독립·도메인-독립 scaffold+TBox+최소 A2. τ² pass 극대화 아님. rev1이 이걸
  잊어 표류했다 — rev2는 매 레버를 "도메인-일반이며 A2 안 늘리는가"로 검문한다.
