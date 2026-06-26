# LOAD THEORY — 부하의 규정·분해·측정 + scaffold=load-reduction + scale의 load-tolerance (설계서)

> **상태**: 설계+Phase-L0 관측 완료(리뷰#1·#2 반영·2026-06-26). 진입=`06-NOW`·`EXPERIMENT_DESIGN §0★★`(마스터 matrix)·`MAKEORBREAK_VERDICT_2026_06_26`·`ORCHESTRATION_CAPABILITY_LEVER_DESIGN_2026_06_26`. 결과=`sim_results/load_obs_phase0_2026_06_26.txt.gz`.
> **한 줄**: "작은 모델이 *부하 상태*서 큰 모델보다 더 틀린다"의 *부하*를 측정가능 feature로 규정·분해하고, scale별 **load-response 곡선**(failure-onset)을 측정하고, **scaffold를 load-reduction 연산자**로 정식화해 thesis(small@유효load L′ ≈ large@native load L)를 *예측틀*(ΔL 독립추정 필수)로 만든다.
> **★범위 경계(엄수)**: 이건 *새 논문 주제가 아니라* `EXPERIMENT_DESIGN §0★★` scale-map의 **이론 backbone**. make-or-break 헤드라인(결정론 scaffold+base+TCO·operand SFT NO-GO)을 *대체* 아니라 *설명*. **closed-form scaling law·25-bench 동시확산=보류**(§6).
>
> **★리뷰#2 반영(2026-06-26·falsification-first)**: (1) **무료 관측 2개를 construct-validity 게이트로 격상·먼저**(기존데이터·gpt-4.1 0). 통과해야 생성·onset(L1+)으로. (2) **L1~L4 미약속**(예산위기·헤드라인 settled·load theory=modest 설명층·예산 승인 시만). (3) **§3 "증명형태"→"예측틀"**: `pass⇔L_eff<L*`는 ΔL·L* 둘 다 fit하면 동어반복→**ΔL을 scaffold의 *기계적* feature-감소량(결과 아닌 동작서)으로 독립추정**해 onset-shift를 *예측*해야 비순환. (4) **5차원=canonical 아님·후보**: 데이터가 collinear/sparse로 줄임(관측이 실제로 2로 줄임·아래). (5) 관측=약한 스크린(난이도 교락)·인과는 통제생성.
>
> **★★Phase-L0 관측 결과(2026-06-26·`load_obs.py`·n=112·gpt-4.1 0)**: **게이트 통과 — 단 좁게.** operand 공변량(r=+0.39) 통제 후 **L_len(r|op=+0.37·p<.001)·L_branch(+0.26·p<.01)만 생존**·L_state/L_interf=operand 교락으로 소멸·L_contra=무의미+희소(12/112). **유효 load축=2개**(리뷰어 붕괴예측 확증). 약한 ΔL 스크린: **현 scaffold(present+nest+g15)가 L_branch를 안 깎음**(고분기 task 잔존·조건트리 executor 공백·t37 정합)·L_len은 약하게 감소. ⇒ 이론 무근거 아님이나 **2차원으로 좁혀짐**·proof-form은 미입증(L1+ 필요).

---

## §0. 동기 (왜·표류 아님)

- 지금까지 "orchestration-under-load"를 **느슨하게** 써왔다(부하의 정의 없음). 사용자 directive: 부하를 *규정·분해·측정*하고 scale이 부하상태서 orchestration을 어떻게 키우는지 정밀화.
- **make-or-break 설명층**: 판정(operand 실행=32B 100%·SFT NO-GO·잔여=orchestration-under-load)은 *무엇이* 남는지 말했지 *왜 scale이 그걸 닫는지*는 mechanism이 없다. load theory가 그 mechanism.
- **이미 로드맵 위**: handoff §2 "스케일=능력-출현/비용 지도"=`EXPERIMENT_DESIGN §0★★` matrix. 본 설계=거기에 **load축**을 추가하는 확장(신문서 아님·[[03]] 재발명 금지).
- **thesis 정합**([[00]]): 작은 LLM이 결정론 분담으로 대형 도달. load theory=그 "분담"이 *유효부하 감소*임을 정량화 = thesis의 증명형태.

---

## §1. load 규정 — 단일량 아닌 5차원 벡터

부하 = task가 모델의 *작업기억/주의*에 거는 압력. 우리 fail-all 포렌식 + 사용자 후보(모순지시·길이망각)로 **측정가능 차원**으로 분해. 각 차원은 task에서 **계산가능한 feature**.

| 차원 | 정의 | 측정 feature(계산식) | 실증(tau2) | 인지/LLM 선행 |
|---|---|---|---|---|
| **L_len** 길이-보존 | 보존할 컨텍스트 길이·"중요한 거 잊음" | 대화+읽기 토큰 수 | t36 16388 초과·t99 주문 망각 | lost-in-the-middle·retrieval head |
| **L_state** 상태-운반 | 동시 보존 상호의존 변수 수 | Σ(미완 write의 미해결 슬롯) | t20 배치·t99 multi-order | working-memory span |
| **L_branch** 분기 | 중간결과로 갈리는 조건분기 수 | 요청 내 if/else 깊이·#분기 | t37 "split되면…아니면…" | 다단계 추론 깊이 |
| **L_interf** 간섭 | 헷갈릴 유사 엔티티 수 | #유사 주문/품목(같은 타입) | t111 엉뚱 주문 | proactive interference |
| **L_contra** 모순-개정 | 후행 턴이 선행 지시를 뒤집는 횟수 | #(턴 간 제약 revision) | t71 "GC→PayPal" | belief update·recency |

- **부하 벡터** `L = (L_len, L_state, L_branch, L_interf, L_contra)` = **후보**(canonical 아님). 스칼라화·차원선택은 *데이터가* 결정 — 선험 가중 금지.
- **load ≠ operand 난이도**: operand(어느 변형)=실행능력(이미 100% GIVEN-SPEC). load=그 실행들을 *함께 들고가는* 압력. 직교 통제(load 올리되 operand 고정).
- **★Phase-L0 관측이 5→2로 줄임(2026-06-26·n=112)**: operand 통제 후 **L_len(+0.37)·L_branch(+0.26)만 fail-예측 생존**. L_state·L_interf=operand 교락으로 소멸·collinear(L_state↔L_interf=.55). **L_contra=무의미+희소(12/112)→tau2서 폐기**(다른 벤치선 재검토). 유효 load축≈2 클러스터{길이+분기 생존 / 상태+간섭 비생존}. → 이후 통제생성은 **L_len·L_branch 2개로 한정.**

---

## §2. 측정 형태 — load-response 곡선 (closed-form 아님·★리뷰 정직)

closed-form 유도는 과욕. **측정가능 규칙성**만 주장:

- **종속변수**: `fail(L, N)` = scale N에서 부하 L일 때 robust 실패율(pass^all·다수 trial·[[06]]).
- **load-response 곡선**: 한 차원씩 통제 증가 → `fail`의 *onset*(정확도가 무너지는 load).
- **load-tolerance** `L*(N)` := `fail`이 임계(예 0.5) 넘는 load 수준. **가설 H_scale: L*(N) monotone↑ in N**(큰 모델이 더 높은 부하까지 버팀). = "scale이 load-tolerance를 키운다"의 정량.
- **iso-fail contour의 scale-shift**: 같은 실패율을 주는 load 등고선이 N↑로 우상향 이동 → scale의 효과를 *차원별*로 분해(어느 load 차원에 scale이 가장 효과적인가).
- 주장 강도 = **경험적 onset-threshold/power-law**(descriptive). "수학적 법칙" 단정 금지·관측된 규칙성으로.

---

## §3. 통합 — scaffold = load-reduction 연산자 (thesis 증명형태·★핵심)

각 scaffold가 특정 load 차원을 깎는다:

| scaffold | 깎는 차원 | 메커니즘 |
|---|---|---|
| present/autofetch | L_len↓·L_interf↓ | 읽기 외부화·후보 열거 |
| calc/compute(COMPUTE) | L_state↓·L_steps↓ | 집계를 모델 밖 결정론 |
| gate(eligibility) | L_branch↓ | 자격분기를 enforce(모델이 추적 안 함) |
| controller(plan-execute) | L_state↓·L_len↓ | orchestrator가 state 보유·atomic 단문 |
| 조건트리 executor(D) | L_branch↓ | 분기를 결정론 walk |

→ **유효부하** `L_eff = L − ΔL_scaffold`. 그러면:
> **예측(증명형태): pass ⇔ L_eff < L*(N).**
> 따라서 **small @ L_eff(=scaffold 깎은) ≈ large @ L(native).** "왜 작은+scaffold가 큰 모델만큼?"의 정량 답.

- **검증법**: scaffold ON/OFF로 *같은 task*의 failure-onset 곡선이 **수평 이동**(ΔL_scaffold)하나 측정. 이동량 = scaffold의 부하감소 효용.
- 이 한 식이 make-or-break(operand 100%지만 load서 무너짐)·scale 단조·scaffold 작동을 **하나로 설명**.

---

## §4. 기존 자산 정합 (재발명 금지·[[03]])

- **orchestration 능력 = [[02]] generator-algebra flow-축(P1–P9 제어흐름)**. 벤치별 orchestration을 *새로 규정 말고* P-primitive 조합으로 기술. control-flow **깊이/복잡도 = L_branch·L_state의 합성 지표**.
- **측정 홈 = SOPBench(control-flow 벤치)**: control-flow 복잡도를 파라미터로 load-graded 변형 생성=자연스러운 load-response 실험대(이미 학습 벤치).
- **load축 = `EXPERIMENT_DESIGN §0★★` matrix 확장**: 능력×레버×scale **×load**. 마스터 확장.
- **load-graded task 생성 = Synth/ABox 메커니즘 재사용**([[01]]): operand 고정·flow 복잡도만 증감 = 통제 변형.
- **BENCH_PORTFOLIO**: tau2+SOPBench 2개로 *먼저* load-response 검증 → portfolio 25 확산은 조건부(§6).

---

## §5. 측정 설계 (free 우선·plan_probe 확장)

- **load-graded 격리 probe**(gpt-4.1 0): `plan_probe.py` 확장 = operand 고정한 채 한 load 차원씩 통제 증가시킨 **합성 변형** 생성 → 7B/14B/32B에 격리 질의 → failure-onset.
  - L_len: 무관 주문/품목 padding.
  - L_state: 동시 수정 항목/주문 수 ↑.
  - L_branch: 조건분기 depth ↑.
  - L_interf: 유사 엔티티 수 ↑.
  - L_contra: 중간 revision 수 ↑.
- **scaffold ON/OFF ΔL**: 같은 load-graded 셋을 scaffold(present/calc/gate/controller) on/off로 → onset 곡선 수평이동 측정.
- **지표**: robust failure-rate(pass^all·다수 trial)·onset load·L*(N)·ΔL_scaffold. **pass^1 금지**([[06]]). 레버 발화율 전수([[30]]).
- **유료 게이팅**([[09]]): probe 전부 로컬. live user-sim end-to-end는 곡선 확정 후 승인후 확인만.

---

## §6. 비용규율 + 범위 경계 (엄수·★표류 차단)

- **보류(GO 아님)**: ① **closed-form scaling law** → 경험 곡선으로 대체(§2). ② **25-bench 동시확산** → tau2+SOPBench 2개 먼저, 확산은 load-response가 *측정되고 thesis-증명형태가 설 때* 승인후.
- **선행 정독 필수**([[41]] rival/whitespace): cognitive-load theory(교육)·long-context degradation/lost-in-middle·task-complexity×model-size 경험연구 *이미 존재*. **우리 novelty=load-decomposition × scaffold-load-reduction을 tool-use transfer에 묶음** — 선행 확인 전 "신이론" 주장 금지. ToolOrchestra(소형>대형 선점)와 차별=transfer + load-offload 정량.
- **헤드라인 불변**: make-or-break 판정 그대로. load theory=설명층(F3 능력-출현 지도를 *왜*로 보강).
- **[[05]]/[[11]]**: load-graded 생성=ABox/flow만 변경·도메인특화 0. tau2 학습 0(SOPBench서 학습 시 A2-swap).

---

## §7. 단계 (design-first·각 단계 게이트)

0. **Phase L0 — 무료 관측 게이트 ✅완료(2026-06-26)**: ① construct-validity(fail↔load 상관·operand 통제)=**통과 좁게**(L_len·L_branch 생존) ② 약한 ΔL 스크린(floor vs scaffold)=**L_branch 미감소·L_len 약감소**. 결과=`sim_results/load_obs_phase0`. → 이론 무근거 아님·2차원으로 좁힘.
1. **Phase L1 — load-response probe(조건부·미약속)**: **L_len·L_branch만** load-graded 격리 probe·7B/14B/32B·onset. → L*(N) monotone? **예산 승인 시만.**
2. **Phase L2 — scaffold ΔL 독립추정(조건부)**: ΔL=scaffold 기계적 feature-감소량(독립)→onset-shift 예측(비순환). **orchestration controller 빌드 시 묶음.**
3. **Phase L3 — SOPBench 일반화(조건부)**: control-flow 복잡도 load-response가 tau2와 정합 + 예산 승인 시.
4. **Phase L4 — live 확인(조건부·유료 1회)**: 곡선 확정 후 end-to-end 1 스케일.

**★현재 GO = L0(완료)까지.** L1~L4 = 미약속·예산 게이트([[09]]/[[03]]). 헤드라인 settled → load theory는 modest 설명층.

---

## §8. 판정 / 성공기준

- **성공(thesis-증명형태 성립)**: (a) L*(N) monotone↑ (b) scaffold가 onset을 수평이동(ΔL_scaffold>0) (c) small@L_eff ≈ large@L 가 데이터로. → "결정론 scaffold가 유효부하를 깎아 소형을 대형 tolerance 안으로"가 *정량 증명*.
- **부분/실패**: scaffold ΔL≈0(부하 안 깎음) → scaffold 효용이 load-reduction이 아닌 다른 것 → 재해석. L*(N) 비단조 → scale-부하 관계 재고.
- **learn 함의**: scaffold ΔL로도 L_eff > L*(N) 남는 차원 존재 → 그 차원이 **유일 learn 후보**(부하-tolerance를 학습으로 키우기·SOPBench control-flow→A2-swap). 없으면 learn NO-GO 강화.

---

## §9. 리스크 / 함정

1. **load feature 측정 타당성**: 5차원 feature가 진짜 부하를 포착하나 = 구성타당도. → onset이 feature와 단조 상관하는지로 *검증*(상관 0이면 그 feature 폐기).
2. **차원 간 교락**: load 차원들이 함께 움직임(긴 task=L_len↑∧L_state↑). → **한 번에 한 차원만** 통제증가(나머지 고정) 합성 변형.
3. **operand 누설/혼입**([[08]]): load 올릴 때 operand 난이도 같이 오르면 혼동. operand 고정 강제(GIVEN-SPEC 옵션).
4. **계측기 버그**(plan_probe # 버그 전례): load-graded 생성·채점 단위테스트 + 전수 덤프 eyeball.
5. **표류**([[03]]): "수학 이론" 매력에 끌려 측정 전 형식화 과투자 금지. **측정→규칙성→modest 형식화** 순서. 25-bench·closed-form 보류 재확인.
6. **선행 중복**([[41]]): 선행 정독 없이 novelty 주장 → reject 위험. L0서 선행 census 동반.

---

## §10. 자산

- **재사용**: `plan_probe.py`(격리 probe·확장 base)·`operand_controlled.py`(GIVEN-SPEC 통제)·`gate_interpreter.py`/`t2_gate_patch.py`(scaffold on/off)·Synth/ABox 생성기([[01]])·SOPBench 클론·gate-grid 다scale 데이터(7B/14B/32B).
- **신규**: `load_graded_gen.py`(차원별 통제 변형 생성)·`load_response_probe.py`(onset 측정)·`scaffold_delta_probe.py`(ΔL)·`LOAD_THEORY_DESIGN`(본 문서).
- **결과**: `reports/facet_rft_2026/sim_results/`(영속화·[[30]]).

---

## §11. 리뷰 질문

1. **범위 경계 동의?**: closed-form·25-bench 보류 + tau2+SOPBench 먼저 + 마스터 matrix load축 확장 = OK?
2. **5차원 충분/과다?**: L_len/L_state/L_branch/L_interf/L_contra — 합치거나 더할 차원?
3. **측정 우선순위**: Phase L1(tau2 onset) 먼저 vs L2(scaffold ΔL) 먼저? (ΔL이 thesis-증명형태 직결이라 L2 우선도 가능.)
4. **first build**: `load_graded_gen.py`(생성기) 먼저 vs 기존 tau2 task를 load-feature로 *분류만* 해서 관측연구부터(생성 없이 free·더 싸다)?
5. **선행 census**: L0에서 cognitive-load/long-context 선행 정독을 어느 깊이로?
