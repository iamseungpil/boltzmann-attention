# Deep Research — 입력 formalize/구조화가 추론↑ + 정보 floor + scale 트레이드오프

> 출처: deep-research `w2i00droj` (2026-06-16 완주·107 agents·3.9M tok). 전체 result=task output. 인용=1차 검증분([[feedback-arxiv-citation-discipline]]).
> 직결: MSC 설계(`MIN_CONTEXT_FORMALIZER_DESIGN.md`)·floor sweep 결과(`ma/M_A_RESULTS.md` §8)·[[project-decomposition-optimality-contribution]].

## 한 줄 결론
설계의 3 기둥(입력 formalize↑·정보 floor 측정가능·input-offload가 scale 대체)은 문헌이 지지하나 **scale 대체는 task-narrow서만**(일반주장 REFUTED). 4번째(우리 신규)=**deterministic scaffold가 world-state pre-formalize + tool-use exchange서 min-info floor 측정·scale-trade**는 미출판 gap.

## 검증 findings
1. **입력 formalize/구조화가 고정모델서 추론↑(확립)**: PoT(`2211.12588`·TMLR23)·PAL(`2211.10435`·ICML23) 프로그램화+결정론 인터프리터 offload = **~12% over CoT**·**PAL+Codex가 PaLM-540B+CoT를 GSM8K서 절대 +15%**. 표 serialization만으로 ±0.22(보통 ~0.05)·structural prompting +0.8~5.7. ⇒ "같은 정보, 더 형식화 → 정확도↑" 1차 지지.
2. **★정보 floor 측정가능·capability와 분리(근접선행)**: **"Sufficient Context"(Joren et al·ICLR25)** — context 충분여부 autorater. **소형=충분해도 환각(reasoning-limited)·대형=불충분해도 abstain 안 함(info-limited)** = 우리 info/reasoning 분리와 *동일 개념*. irrelevant context가 reasoning-path·산술 저하(고정 capability서). ⇒ **floor 개념 자체는 신규 아님.**
3. **★input-offload가 scale 대체 = task-narrow서만**: PAL 소형>540B(claim 3-0). 단 **일반 "작은+offload≈큰"은 REFUTED(1-2)**·350M-beats 수치 REFUTED(0-3). crossover 태스크별·보편법칙 아님. ⇒ **우리 floor 음성(L2b@7B ≪ L1@14B·MSC≠scale대체 on exchange)은 문헌과 정합.**
4. **★genuinely unclaimed(우리 gap)**: 기존 pre-formalizer/filter(Visconde reranker·DeepSieve router·sufficient-context autorater·PoT/PAL)는 **전부 NEURAL/LLM-driven**. **결정론 scaffold가 world-state pre-formalize는 없음**·**tool-use/agentic exchange서 min-info floor 측정+scale-trade도 없음.** = 우리 신규성 자리.

## 우리에의 함의
- floor sweep 음성(MSC≠scale대체)은 **버그/실패 아니라 문헌-정합**(scale대체는 task-narrow). 정직하게 보고.
- **신규성 무게 재확정**: floor *개념*=선행(Sufficient Context)·input-formalize↑=선행(PoT/PAL) → 우리 = **(i)deterministic-scaffold pre-formalize(LLM-driven 아님) (ii)typed-DAG closure 최소성 (iii)tool-use exchange서 floor+scale-trade 측정**. 측정+결정론-scaffold 결합이 gap.
- ⚠input-side(도움) vs output-side constrained decoding(해침·[[dr_constrained_decoding_vs_reasoning]]) 혼동 금지(DR 명시).
