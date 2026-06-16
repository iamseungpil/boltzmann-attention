# Deep Research — constrained/forced-JSON decoding vs reasoning + 우리 방법 신규성

> 출처: deep-research 워크플로 `wf_3f814306` (task `whv0jinf9`·2026-06-16 완주). 104 agents·4.1M tok. 전체 result = task output file. 인용=1차 검증분([[feedback-arxiv-citation-discipline]]).
> 직결: M-A forced-JSON 교란(arm A/B)·CoT/two-call/structural_tag 처방·[[feedback-capability-vs-artifact-elicitation]]·MSC 설계 §7.

## 한 줄 결론
**Yes — schema-강제 디코딩은 추론을 측정가능하게 깎는다(JSON 포맷 자체가 아니라 *schema 제약*이 구동).** 완화법(strict 유효성 유지하며 추론 회복)은 **이미 출판**됨. 우리 selector+resolver의 일반개념은 선행에 매핑되나, **agentic NL→tool-arg에 reason-then-format + decoupled resolver를 결합**한 건 미출판 gap.

## 검증된 findings
1. **부작용 실재·정량**: schema-JSON이 GSM8K서 ~26–73% 상대 하락(Claude-3-Haiku **86.51→23.44**·GPT-3.5 75.99→49.25·LLaMA-3-8B 75.13→48.90). **JSON-without-schema는 훨씬 덜 떨어짐 → schema 제약이 구동(JSON 포맷 아님).** [Tam et al. EMNLP24 Industry·`2408.02442` Table 1]
2. **메커니즘 이중**: (i) 유한문법이 CoT 중간토큰 공간 제거(CRANE·ICML25·`2502.09061`·circuit-complexity) (ii) greedy 토큰마스킹+재정규화가 분포 왜곡(GAD·NeurIPS24·Ye et al KL-bias).
3. **모델크기 의존 reversal**: ≥14B few-shot서 unconstrained ≥ constrained(qwen2.5-14b GSM 5-shot 0.56 vs 0.36). 단 이건 그 논문의 *소수/예외* case·"GCD 일반적으로 도움" 헤드라인은 **REFUTED(0-3)**.
4. **완화법(strict 유효+추론 회복·출판됨)**:
   - **CRANE** 문법증강+제약/무제약 교대 → GSM-sym/FOLIO +10pp.
   - **in-schema reasoning-field-first**(JSONSchemaBench) — schema 안 free-text reasoning 필드 먼저.
   - **two-stage NL→Format**(무제약 CoT → 제약 추출) → 손실 거의 전부 회복(Tam et al).
   - **GAD/ASAp** 분포보존(점근·효율비용).
   - ⇒ 우리 arm Bcot(CoT)·Btwo(two-stage)·structural_tag = 이 처방들의 채택.
5. **★신규성**: "provenance-typed selector + 추론채널 + 결정론 resolver"의 *각 조각*은 선행(reason-then-format·in-schema reasoning·decoupled parser/solver)에 매핑. **단 셋을 *agentic NL→tool-arg resolution*에 결합한 surveyed source 없음 = genuine gap.**

## 우리에의 함의
- **M-A "reasoning 실패"는 schema-강제 교란 확정** — JSON 포맷이 아니라 guided_json *schema 제약*이 원인. 우리 forced arm(A/B)이 정확히 그 regime.
- **처방은 기성기술**(CRANE/two-stage/structural_tag) → *채택*이지 기여 아님([[reference-nl-formal-decouple-literature]] 정합).
- **메서드 신규성 후보** = agentic tool-arg에 결합(gap 실재)·**단 측정 기여(floor)가 더 안전**(MSC §7).
- ⚠"JSON 항상 추론 해침"·"strictness 단조 저하"는 같은 논문서 **REFUTED** — 과대일반화 금지(특정 magnitude만 유효).
