# 리뷰 — `TASK_CONSTRAINT_DESIGN.md` §8 (b) 구현 설계 (완전 검증 게더 + resolver)

> 상태: **리뷰 (구현 착수 게이트 제안)**. 작성 2026-06-02.
> 대상: `scripts/distill/TASK_CONSTRAINT_DESIGN.md` §8.1~§8.5 (binding 진단 + harness sim + (b) 구현 권위본 §8.5).
> 방법: 통독이 아니라 §8의 load-bearing 주장(수치 출처·아키텍처 함의·정합성)을 trace. 1라운드 리뷰(`TASK_CONSTRAINT_DESIGN_REVIEW.md` P1-P8/R1-R7)의 후속.
> 선행: §7 zero-train(login 비-binding) → §8.1 binding=under-verification(37/44) → §8.2 harness sim abc=37 → §8.3/8.4 realistic·cred 정정 → §8.5 (b) 구현.

---

## 0. 한 줄 결론

§8의 **binding 진단(under-verification, login 단독 binding=0)** 은 데이터로 견고하고 §7 가설을 정직하게 반증했다. 그러나 그 위의 **(b) 구현 설계(§8.5)** 는 (1) 학습-planner 논제를 잠식할 수 있는 결정론 resolver 의존, (2) 모든 천장 수치가 단일 자작 하네스(`run_scripted`)에 의존·실파이프 미교차검증, (3) 측정 아닌 정적 상한을 목표로 제시(should_F 83)라는 3개 load-bearing 위험을 안고 있다. **"§8.5 구현 확정"은 이 게이트들 전에는 과하다.** 코드 착수 전 G1(실파이프 교차검증)·G2(resolver+should_F 샘플 검증)를 권고.

---

## 1. 검증된 부분 (토대는 견고)

| 주장 | 근거 | 판정 |
|---|---|---|
| should_T 병목 = under-verification(필수 CHECK 미호출), login 단독 binding=0 | §8.1 `binding_diag.py`: 44 실패 중 37(84%) 필수 CHECK 미호출; zero-train(§7) login -59%인데 should_T 불변 | ✅ 견고 |
| root cause = teacher 게더 결함(name-dedup + condition 누락) | §8.1 `build_tbox_planner_sft.py:108` `dict.fromkeys`(동명 체크 소멸) + `p in tool_names`(condition 제외) | ✅ 코드 일치 |
| 완전 게더(A+B+C)면 dirgraph 충족 | §8.2 `run_scripted` abc=37=oracle (단 C2 단서) | ⚠️ 조건부(아래 C2) |
| cred-cheating 발견·16분해(8 도구선택+8 cred부재) | §8.3/8.4 leaderboard 통과수 + `_admincheck.py`(internal_get_database 미노출) | ✅ 정직한 정정 |

binding **방향**(완전 게더)은 정당화됐다. 이하는 그 위의 구현 설계·수치 신뢰성·논제 정합.

---

## 2. 문제점 (심각도순)

### 🔴 C1 — 결정론 resolver가 게이팅 판단을 흡수 → **학습-planner 논제 잠식** (최우선·개념)
§8.5.1: planner(LLM)는 `observed` **bool만 보고 act/STOP**, **산술·비교 안 함**; resolver(결정론 ABox)가 `balance≥amount`·`score≥임계`를 계산.
- 게이팅 **판단**이 결정론으로 이동하면 학습 TBox의 기여가 **"all-True→act / any-False→STOP"** 라는 거의 자명한 함수로 축소 → **L0(결정론 executor)로 수렴**. 본 라인 핵심 주장(HT3 학습기여·전이)을 약화.
- 프롬프트가 precond를 렌더해 보여주므로 planner의 "검증 선택"은 **열거**에 가까움 = 얇은 학습 신호.
- **요구**: §8.5/§0에 **(α) 벤치 SOTA** vs **(β) 학습-전이 논제** 중 목표를 택일 명시. β라면 "planner가 비자명하게 학습/전이하는 것"이 무엇인지 정의(예: 어떤 검증이 *필요한지*를 도메인 간 전이) + ablation(resolver만 + planner naive → 붕괴)로 학습기여 입증. 없으면 "arm-2 L0의 변형"으로 정직 프레이밍.

### 🔴 C2 — 모든 천장 수치(37/24/21/29)가 **단일 자작 하네스 `run_scripted`** 산물, 실파이프 교차검증 0
§8.2~8.3의 oracle 37·ab 24·abc 37·realistic 21/29가 전부 `run_scripted`(결정론 scripted-gather + evaluator). 이 하네스는 **이미 2회 hand-replay 수정**(content bool 복원·호출순서)을 거침 = 신뢰가 자기검증.
- 전례: `mre_bank_impossible.py`(graph-replay)도 "나열순서 아티팩트로 신뢰불가→증거금지" 폐기됨(메모리). 동형 위험.
- 정직 천장 32 = (scripted conditional realistic 29) + (3 "scripted 아티팩트") — **다리가 자작 하네스의 선택적 신뢰 위에 있음**(3개는 틀리고 나머지는 맞다고 가정).
- **게이트 G1(반나절, 최우선)**: abc scripted plan 중 **1개 config라도 실제 `run_simulation`(scripted client)→`run_evaluation`** 으로 돌려 should_T가 37과 일치하는지 확인. 불일치 시 §8.2~8.4 수치 전부 재검토. **코드 착수 전 통과 필수.**

### 🔴 C3 — should_F "31→~83"은 **측정값 아닌 정적 상한**을 목표로 제시
§8.5.5 "86 중 83 도구-탐지". 그러나 §8.2 명시: **should_F는 scripted 미검증**(항상 goal 호출). 83 = "거부 트리거가 원리상 탐지가능한 task 수"일 뿐, **모델이 게더+탐지+STOP 달성 보장 아님**.
- 회귀 전례: arm-3v2(무학습 in-context STOP)=**57** → arm-4a(학습)=**31**. **학습이 should_F를 회귀**시킨 실적. P3 fragile 14건 경고.
- **요구**: should_F 목표를 "상한 83"이 아니라 **"arm-3v2의 57 회복 + α"** 로 정직화. should_F도 scripted로 검증(현재 미검증). 재학습이 57을 넘는다는 근거 없으면 83은 aspirational로만 표기.

### 🟠 C4 — §8 내부 **반복 retraction**이 n≈1 패턴 재발
§8.1 "A+B→40 확정" 철회 → §8.2 "C 불필요" 정정 → §8.3 realistic 21 → §8.4 정정 32. 한 세션에 핵심 수치 **3회 반전**, 각각 자작 하네스 1회 실행 기반. 1라운드 리뷰가 경고한 함정 재발.
- **요구**: G1(C2) 교차검증으로 수치 lock 전엔 "확정"·"권위본" 표기 보류. §8.5 헤더 "구현 설계 확정"은 G1/G2 후로.

### 🟠 C5 — §9.4·§10-Q2가 §8.4와 **정면 모순** (R4 스테일 재발)
- §9.4/§10-Q2: "마스크=`task_constraint` 단독 충분, ∪innate_dep 불필요(P5 해소)".
- ↔ §8.2/§8.4/§8.5.3: **C(login/innate-dep) 필요**(ab 24→abc 37, +13), §8.1이 "P5 단독충분 **반증**" 명시.
- → §9.4·Q2는 **현재 틀림**. 1라운드 R4가 지적한 "정정 누적 스테일" 동일 재발. **§9.4를 "C 필요로 반전·P5 철회"로 즉시 수정.**

### 🟠 C6 — B의 condition→getter는 "induce·비-oracle"이나 **compare 로직은 hand-derived(bank)**
§8.4 "co-occurrence로 깨끗이 도출·inducible". 그러나 §8.5.3 매핑의 **비교연산·임계**(`≥amount`, `≥minimum_credit_score`)는 bank 의미론 — co-occurrence가 주지 않음. getter *식별*은 induce 가능, *비교규칙*은 도메인 수기.
- §8.5.6이 "타도메인 일반화 ⬜"로 정직히 남긴 건 좋으나 §8.4 "깨끗이 도출"은 과표현.
- **요구**: "getter 식별=induce / compare=현 bank 수기"로 정확히 구분 표기. 6도메인 일반화는 미검증 리스크로 명시.

---

## 3. 인정 (강점)

- **binding 진단**: under-verification 37/44·login 단독 binding=0 — §7 가설을 데이터로 반증, 레버 방향(완전 게더) 정당화. 견고.
- **cred-cheating 발견·16분해**: 사용자 지적 수용한 정직한 정정. PartB 8개 admin-DB-read 불가 논증 구체적(internal_get_database 미노출).
- **should_T/should_F 대칭(이유기반 STOP)**: 현 teacher "이유없는 STOP" 진단 타당, 양축 동시 교정 방향 옳음.
- **two-stage "LLM 무계산"**: 환각 산술 제거 동기는 합리(단 C1 논제 함의 confront 필요).

---

## 4. 구현 착수 게이트 (권고)

| # | 게이트 | 막는 위험 | 비용 | 통과 기준 |
|---|---|---|---|---|
| **G1** | **실파이프 교차검증**: abc scripted plan 1 config를 `run_simulation`→`run_evaluation`으로 → should_T가 run_scripted 37과 일치? | C2·C4 | 반나절, 재학습0 | 일치(±1). 불일치 시 §8.2~8.4 수치 동결·재검토 |
| **G2** | **resolver condition→getter+compare** 구현 + bank 5~6 should_F task에서 "이유기반 STOP" 샘플 실제 생성 검증(§8.5.2) | C3·C6 | 코드+검증, 재학습0 | should_F 샘플이 위반 condition 게더 후 STOP. + §9.4 스테일 정정(C5) |
| G3 | (G1·G2 통과 시만) teacher 전체 교체 → SFT 재생성 → 재학습 1회 | — | 재학습 | should_T↑(목표 32 정직), should_F **gross** gain/loss(57+α, net 금지) |

**순서**: G1(가장 싸고 결정적) → G2(사용자 제안 resolver-first와 병행) → G3. **사용자 제안 "resolver부터"는 타당하나, G1로 천장 수치를 lock하지 않으면 §8 4번째 retraction 위험.**

---

## 5. 리뷰어 결정 질문

- **Q-C1**: 목표는 (α) 벤치 SOTA인가 (β) 학습-전이 논제인가? β면 결정론 resolver가 학습기여를 남기는지 ablation 설계 필요. (가장 중요 — §8.5 전체 프레이밍을 가름.)
- **Q-C3**: should_F 목표를 83(상한)에서 57+α(arm-3v2 회복)로 내릴지.
- **Q-G1**: 코드 착수 전 실파이프 교차검증을 게이트로 둘지(권고 yes).
- **Q-C6**: B compare 로직을 bank 수기로 두고 6도메인은 후속으로 명시할지.

---

## 부록 — 검증 출처(설계서 인용 아티팩트)
`binding_diag.py`(§8.1 under-verification 37/44)·`lever_decomp.py`(§8.1 leaf 173, B-dominant)·`run_scripted.py`(§8.2~8.3 천장 37/24/21/29 — C2 단일점)·`_admincheck.py`(§8.4 internal_get_database 미노출)·`evidence_a_probe.py`(8 PartA 확정)·leaderboard 59 files(§8.4 16분해). **G1은 이 중 run_scripted를 실파이프로 교차검증하는 것.**
