# Deep Research — 작은 모델 reasoning↑: 분해 + 외부 step-검증

> 출처: deep-research `w2ijg45bt` (2026-06-16 완주·101 agents·3.6M tok). 전체 result=task output. 인용=1차 검증분([[feedback-arxiv-citation-discipline]]).
> 직결: 사용자 가설("작은 스텝 엄격히")·MSC/floor(`ma/M_A_RESULTS.md §8`)·[[feedback-selector-verifier-deterministic]]·thesis 증분 typed-action(§6.5).

## 한 줄 결론
**작은 모델을 큰 모델 reasoning에 근접시키는 가장 신뢰가능한 길 = step-분해 + *외부* step-검증.** 자유 CoT도·내부 self-correction도 아님. ⇒ 우리 "결정론 scaffold + 작은 LLM 스텝 + 결정론 per-step 검증" 방향을 문헌이 *직접 지지*. 그리고 **tool-use/NL→formalize per-step typed-verify는 미출판 gap**(전 문헌 math/QA).

## 검증 findings
1. **분해가 소형 reasoning↑(확립)**: least-to-most(Zhou·ICLR23) SCAN 16%→≥99%(+83pp)·easy-to-hard 일반화. distilling step-by-step(Hsieh·ACL23) **770M T5가 few-shot 540B PaLM 능가**(rationale 다중태스크 supervision·데이터 80%). ⚠"CoT는 대형서만"은 **REFUTED(0-3)** — 분해가치는 scale-gated 아님. ⚠일부 헤드라인=단일 데이터셋 cherry·분해 데모는 대형서.
2. **process supervision > outcome(확립)**: Lightman "Let's Verify Step by Step"(ICLR24) MATH 78%·OmegaPRM(DeepMind24) 긴 체인서 ORM 불충분(중간스텝 무보상)·PRM이 고침. (단 GSM8K 짧은체인선 ORM≈PRM·전부 math.)
3. **★외부 step-verifier가 소형(7B)을 대형 수준으로(핵심)**: Math-Shepherd(Wang·ACL24) **자동구축 step-PRM**(인간주석 0)으로 Mistral-7B GSM8K **89.1%**(PRM reranking). compute-optimal test-time search가 **FLOPs-matched서 14× 큰 모델 능가**(Snell).
4. **★self-correction은 reasoning *악화*·외부검증 필요(우리 설계 load-bearing)**: Huang et al "LLMs cannot self-correct reasoning yet" — 내부 self-correction 저하. Zhang et al(ACL24)·T1(2025): **소형은 *강한 외부 verifier* 필요**·내부 self-verify는 **memorization-heavy 서브스텝서 가장 실패** = **결정론 scaffold/tool이 offload하는 바로 그것.** ⇒ [[feedback-selector-verifier-deterministic]] 직접 지지(verifier=결정론).
5. **★gap(우리 신규)**: 전 증거가 **math/QA**. **tool-use / NL→formalize per-step typed-verify는 under-claimed** = 우리 결정론-scaffold 설계의 white space.

## 우리에의 함의 (라인 확정)
- **사용자 가설 "작은 스텝 엄격히" = 문헌 지지·단 *외부* 검증이 핵심**(자유 CoT/내부 self-verify 아님). 우리 floor 데이터(7B+자유CoT 0.656≈14B, but plateau)와 정합 — 자유 CoT는 일부·외부검증이 강한형.
- **우리 메커니즘 = 문헌의 검증된 승리공식**(분해+외부검증)·**우리 신규 = (i)검증이 *결정론*(PRM도 아닌 scaffold) (ii)*typed 증분 action* (iii)*tool-use/NL→formalize* 세팅**(math/QA 아님).
- 정직 caveat: "소형이 대형 능가" 헤드라인=task-narrow·math·cherry 잦음 → 우리도 exchange서 *측정*으로(보편주장 금지).

## 다음 실험 (강한형)
**7B + 결정론 scaffold가 typed 증분스텝 강제 + per-step 결정론 검증** → 자유 CoT(0.656) 넘어 32B-L2b(0.844) 닿나? = 사용자 가설 강한형 + thesis 증분 typed-action 직접 검증. (per-step 검증 = resolver/타입/사전조건 체크 = 이미 결정론 가능.)
