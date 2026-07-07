# Related work — 멀티스텝 에이전트 horizon 부하 (2026-07-07·DR#1 `wf_22bb5647`)

> RELWORK_LOAD_COT(2026-07-05·단일스텝 CoT/WM/self-consistency)의 **멀티스텝 horizon 확장**. 22소스·101클레임→
> 25검증(20확증·5기각). ★대부분 2025-26 preprint·다수 post-cutoff·미재현 → 수치=single-source 데이터점(상수 취급 금지).
> 유일 peer-reviewed=Press 2210.03350. 목적=부하-감축 아키텍처(LOAD_REDUCTION_ARCH_DESIGN) 선행근거 + Paper1 relwork.

---

## 0. 한 줄
2025-26 문헌이 **멀티스텝 실패 = execution-under-load(추론/지식과 별개)**를 직접 지지하고, **부하의 일부가
scale-invariant**(self-conditioning·compositionality gap)임을 실증. 최선 개입 = **결정론 decomposition(minimal-context
+ 외부 상태)** — 단 합성과제(Hanoi)서만 증명. **"소형+scaffold ≈ 대형-bare on 실제 tool-use" = 미확립 = 우리 whitespace.**

## 1. 확증된 핵심 (thesis 지지)

### 1.1 ★keystone — self-conditioning (2509.09677 Sinha/Geiping "Illusion of Diminishing Returns"·ICLR2026)
- **execution 격리**: 지식(in-context dict)+명시 plan 줘도 **per-step 정확도가 step 수 늘수록 하락** → 실패=추론 아닌 **실행**.
- **self-conditioning = scale-invariant**: "context에 자기 과거오류 있으면 더 틀림·**모델 크기 키워도 안 줄어듦**." 200B+도 degrade.
  단 **test-time thinking은 완화**. ⇒ **parameter-scale 축(무효) vs test-time-compute 축(유효) 분리** = "일부 부하 scale-invariant"의 정본.
- **단 scale은 horizon 길이를 삼**(clean isolated-execution서 marginal per-step↑가 지수적으로 긴 과제 완수로 compound). p^H 수치:
  99% per-step→90.4%@10·36.6%@100·0.004%@1000. **∴ scale=isolated-execution regime의 horizon만·오류진입後엔 test-time-compute.**
- ★우리 정합: **이게 "부하 두 원천"의 문헌판** — scale이 삼는 축(isolated per-step) vs 안 삼는 축(self-conditioning drift).

### 1.2 ★compositionality gap (Press 2210.03350·EMNLP2023·**peer-reviewed·유일**)
- **정의(우리 competence-performance 갭의 정본 名)**: "모든 하위문제는 격리서 맞히나 전체 합성은 실패하는 비율." GPT-3 family
  전반 **~40% flat = scale-invariant**(single-hop이 multi-hop보다 빨리 개선돼서). Yang 2402.16837 확증(hop-1만 scaling).
- **Self-Ask scaffold가 갭 축소 = 구조가 닫지 scale 아님** → 우리 thesis 직접 지지.
- caveat: GPT-3세대·2-hop QA(멀티툴 궤적 아님)·pre-CoT. o1급 미검. = 우리가 실제 tool-use로 확장.

### 1.3 super-linear 붕괴·capability≠reliability (2603.29231·post-cutoff)
- 오류 **양의 상관** → long-horizon pass@k가 **i.i.d. p^H보다 빨리** 붕괴("confused agent stays confused"·Ω(ε·e^{ρT})).
- **capability rank ≠ reliability rank**(GLM-4.5 Air short 94.9%→very-long 66.7%·1위→4위). coverage/조기종료 1%→25%.
- ⇒ **naive p^H는 하한**·실제는 더 나쁨(self-conditioning 정합). pass^1 무효·robust 지표 정당([[08]]).

### 1.4 gradual drift on REAL tool-use (2602.19008 "Capable but Unreliable"·Toolathlon·post-cutoff)
- **within-unit 자연실험**(같은 모델·같은 task): 성공 run이 실패 run보다 canonical path 준수 ↑(+0.060 Jaccard·p<0.0001·n=488)
  = 실패=**reliability/execution**이지 capability 아님(실제 벤치서 격리).
- **점진적 self-reinforcing drift**(단일 초기실수 아님): off-canonical 호출이 다음 off-canonical을 **+22.7pp**↑(전반부 50%는 성공·실패 구분무). = 우리 M3 loop·M4 drift의 기전.
- "**agent reliability는 capability scaling만으론 개선 불가**·lever=실행중 canonical-path로 drift 제약(결정론 개입)." = 우리 접근의 가장 근접한 출판 진술. (단 그 논문 자체 monitor+8.8pp는 검증서 기각.)

### 1.5 taxonomy (2604.11978 HORIZON·Dawn Song 외·3100궤적·post-cutoff)
- 7범주: Planning·**Catastrophic Forgetting**(제약이 context엔 있으나 미주목=우리 M5/M2)·**History Error Accumulation**(M3 loop)·
  Memory Limitation·Environment·Instruction·False-Assumption. = **우리 6기전과 가장 직접 대응하는 출판 taxonomy.**
- "model scaling alone unlikely to resolve dominant failures"·memory+forgetting=27.5%·breaking-region서 모델간 격차 좁혀짐.
  단 within-family scale ladder·scaffold ablation 無 → scale-invariance는 추론(우리 14B/32B가 그 공백 메움).

## 2. 개입 증거

### 2.1 ★MAKER (2511.09030 Meyerson/Cognizant·Nov2025) = 사용자 멀티에이전트 제안의 문헌판
- **maximal decomposition**: step당 **minimal-context microagent 1개 + 강제 state-output** → **per-step 오류가 horizon 늘어도 FLAT**
  (백만 step). 기전: "clear-instruction 실행"을 "insight 필요"에서 격리·minimal context가 attention dilution 제거·각 agent가
  next state 산출=외부화 상태가 tracking loss 방지. = **정확히 LOAD_REDUCTION_ARCH E1(격리)+E2(외부상태 dispatch).**
- ★★**STRONG caveat**: **Hanoi·곱셈(합성·동질 step·상태 자명 외부화)만**. **heterogeneous 실제 tool-use(tau2) 미증명.**
  flat per-step는 부분적으로 by-construction. ⇒ **우리가 실제 벤치로 옮기면 그게 novel.**

### 2.2 planner-executor·reflection
- COPE(2506.11578): small/large plan 교대. self-correction(2310.01798): **외부 피드백 없는 self-reflection은 개선 안 함**.
- ★self-conditioning 함의: **오류 context 위 naive reflection은 오히려 악화 가능** → 우리 개입은 반드시 **외부-결정론 피드백**
  (gate/verifier)이지 self-reflection 아님. 우리 gate=외부-결정론 ⇒ 정합.

## 3. ★★WHITESPACE = 우리 기여 (openQuestions·전부 미충족)
1. **"소형+결정론 scaffold ≈ 대형-bare on 실제 agentic multi-step tool-use" = 미확립.** MAKER=합성만·AppWorld scaffold
   (2604.11465)=통계 비유의(N=168)·memory-scaffold=0/10 개선. **tau2/Toolathlon급 controlled 비교 공백** = 우리 헤드라인.
2. **어느 기전이 scale-reducible/invariant인지 within-family scale ladder + 매칭 scaffold ablation으로 분리한 연구 無**
   → **우리 14B/32B same-scaffold**가 정확히 그 공백.
3. **isolated ~100% vs in-trajectory 정확도를 sub-skill별·history길이 함수로 정량한 연구 無**(Press=2-hop QA뿐) → **우리 operand
   probe(격리) vs full-run(궤적)**이 그 측정.
4. **coverage "do X for all Y"는 서술적 命名뿐**(early-failure·budget exhaustion)·**결정론 set-iteration이 model-frozen으로 닫나
   증거 無** → **우리 E2**가 그 open question 직격.
5. self-correction/critic 수렴조건 미해결(언제 drift 줄이고 언제 증폭하나).

## 4. 기각(support로 인용 금지·검증서 죽음)
- "memory scaffold가 reliability 개선한 적 없음"(과장·0-3) · "Mistral12B>Llama400B"(1-2) · "MAKER=scale이 per-step 무효"
  (overreach·0-3) · "mid-trajectory monitor +8.8pp"(0-3) · "Qwen3-8B scaffold≈33B"(비유의·N=168·1-2).

## 5. Paper1 편입 (신규 cite)
- §2 relwork "agentic horizon" 문단 신설: **2509.09677**(self-conditioning=scale-invariant execution·keystone)·
  **2210.03350**(compositionality gap=競/성 갭 名·peer-reviewed)·**2511.09030**(MAKER=decomposition)·**2604.11978**(HORIZON taxonomy)·
  **2602.19008**(Capable-but-Unreliable=실벤치 drift·"scaling만으론 불가").
- §3/§4 framing: 부하 두 원천(isolated-execution[scale가 삼] vs self-conditioning/composition[scale-invariant·구조가 삼]).
  우리 scaffold=MAKER의 실제-tool-use판 + coverage set-iteration(open Q4).
- 헤드라인 강화: whitespace #1·#2가 **정확히 우리 iso-scaffold×cross-scale on tau2** = moat 재확인([[46]]).
- ★규율: post-cutoff·single-preprint 다수 → 수치 인용 시 caveat·상수 취급 금지. peer-reviewed(Press)만 강하게.
