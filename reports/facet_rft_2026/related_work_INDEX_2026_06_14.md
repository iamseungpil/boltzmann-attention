# Related-Work INDEX — 6편 full-text 딥리드 통합 (2026-06-14)

> 주제별 6 relwork 에이전트(`relwork_{selector,metrics,diversity,nlformalize,determinism,arch}_2026_06_14.md`)의 **본문기반 정독** 산출을 한 곳에서 항해. 입력 = 8편 abstract-level research 보고서(179 unique arXiv). 본 라운드 = abstract-검증 → **full-text 1차검증·승격**.
> **인용규율**: 아래는 각 relwork의 verbatim-검증 결론 요약. 수치/제목/버전은 해당 relwork의 §에 1차출처 명시. **DROP/정정 표시는 설계서에 이미 반영**(§아래 "정정 적용 로그").

---

## 0. 주제별 "가장 결정적인" 논문 (load-bearing 3편씩)

| 주제 | 결정적 1 | 결정적 2 | 결정적 3 | relwork |
|---|---|---|---|---|
| **선별/MBR** | Smoothie `2412.04692`(SEL-1 직계; diagonal-cov=독립오차 가정 → 우리 source-correlation Novelty 공백 입증) | MBR bias–diversity `2410.15021`(선별=다양성 함수 이론닻; 단일모델 i.i.d. 가정=우리 이종풀 보정의 빈칸) | Coder-Reviewer `2211.16490`(SEL-4 직계; "degenerate-solution 선호" = 게이트 역선택 병리의 문헌-동형) | selector §B |
| **다양성/D-oracle** | Setlur `2502.12118` Thm 5.8(검증이득 Ω̃(H/√n)이 *correct/high-reward* trace heterogeneity에 keyed = D-oracle의 정식 옷) | Wang `2502.11027`(쌍별다양성은 **oracle 하에서만** 이득·MV서 "vanish" = 우리 통합풀 음성을 한 문장으로) | Brown `2407.21787`(gold-free 선별 plateau·coverage는 계속↑ = 병목=선별기) | diversity §6 |
| **평가지표 F1-F7** | Chen `2107.03374`(pass@k 비편향 추정량) | τ-bench `2406.12045`(pass^k = F3 동결식) | Erol `2504.13359`(cost-of-pass v=C/R = F7) | metrics §2 |
| **NL→formalize (A2)** | StepFun `2508.04440`(7B가 o3-pro/Claude-4/R1-671B 이김 = "소형≥frontier" capacity 증거 + ThinkingF S1 템플릿) | Prose2Policy `2603.15799`(최근접 과제 analog·frontier-prompt-only·76.5% 정직분모) | Do-LLMs-Game `2604.19459`(compile≠faithful = replay 사각지대 → cross-stage 검사) | nlformalize §8 |
| **추론 결정론 (ⓟ1)** | Thinking Machines blog(batch-size 변동=비결정 주범·batch-invariant 커널 1000/1000 동일) | Yuan `2506.09501`(BF16 greedy서 batch/GPU만으로 ±9%·LayerCast fix) | Zhang `2511.17826`(TP축 비결정·LLM-judge/multi-agent 명시·TBIK) | determinism §7 |
| **아키텍처/diffusion (A3)** | A3 any-order AR `2601.13228`("set 다양성에 diffusion 불요" 닻 — ⚠️공정성 정정 아래) | XGrammar `2411.15100`(guided formalized valid-floor) | SoS-distill `2404.03683`(3.10 path) | arch |

---

## 1. ★정정 적용 로그 (설계서 반영 완료 — 2026-06-14)

인용규율 핵심: full-read가 abstract-수치를 뒤집은 사례들. **전부 권위본에 반영·커밋(aa415aa)**.

1. **`2601.15808` 내용 불일치 [최중대]** — SELECTOR §0·FIELD_GAP §134/136이 "DeepVerifier·검증-측 오기각(correct→incorrect) 천장"으로 인용했으나, 실제 = Wan et al. *"Inference-Time Scaling of Verification: Self-Evolving Deep Research Agents"*(GAIA/XBench). "오기각 천장"과 무관. ⇒ **철회**, 게이트 역선택 외부동형은 `2411.17501`(Stroebl+24 imperfect-verifier 천장)로 대체. (diversity §5)
2. **StepFun `2508.04440` 7B/32B 역전** — "헤드라인 32B(7B 아님)" = **틀림**. Table 3: 7B가 o3-pro·Claude-4-thinking·R1-671B를 이기고 7B≈32B. ⇒ FIELD_GAP §5.6 헤지 제거·capacity 증거로 승격. (nlformalize §9.2)
3. **Prose2Policy `2603.15799` 정직 분모** — "95.3% compile"=371/389 post-filter(485→389, ~20% 거부) = 원입력 76.5%. 양성테스트 82.2%=LLM 자가채점. 결정론=future work. (nlformalize §9.1)
4. **GenRM `2408.15240` 수치** — GSM8K 73→**92.8%**(93.4% 아님; Gemma-9B가 Gemini-1.0-Pro 검증). (selector §A.3) — *SELECTOR 본문 미인용이라 relwork 기록·향후 RW용.*
5. **LLM-Blender `2306.02561` ranker** — PairRanker=**DeBERTa-400M**(RoBERTa 아님)·O(n²) → shortlist 압축 정당. SEL-5 행에 반영. (selector §A.6)
6. **Kamigaito `2410.15021` 부호** — (û−ū)² = **Bias − Diversity**(다양성 차감; "bias+diversity"는 오기). 제목도 "Theoretical Aspects of Bias and Diversity in MBR"(=v2 "Diversity Explains…"은 미확정). (selector §A.1-2)
7. **§1.6 metrics 3건** — cost-of-pass 분모 R:=pass^1(우리 매핑)·Jaeger `2211.15259` [A]→[F]·HELM `2211.09110` v2 날짜핀 2023-10-01. (metrics §A)
8. **`2506.09501` = LayerCast** — "BF16 최악/FP32 근결정"은 관찰; 실제 처방=LayerCast(16-bit 저장·FP32 compute)=fallback 레버. (determinism §3)
9. **vLLM 결정론 최소버전 = v0.11.1** — `VLLM_BATCH_INVARIANT`은 0.11.1 신규(0.11.0 미탑재 확인)·TP=1 필수. (determinism §5/8)
10. **A3 any-order AR `2601.13228` 공정성** — 표준 AR에 짐(TriviaQA 19.4 vs 52.1); diffusion 필적은 2B-vs-65B 효율정규화 = **framing 닻**일 뿐 강생성기 아님. (arch)

## 1b. DROP / 인용금지 (불공정·제약불일치)

- **PoLL `2404.18796`** — 패널=Command-R-35B+Claude-Haiku+GPT-3.5(proprietary). "frontier 단일 judge 불필요"의 일반근거로만, **≤7B/on-prem 주권-leg 근거로는 금지**. (selector §C) — SEL-5 행에 ⚠️ 박제.
- **MAV `2502.20379`** — "약-verifier 집계 > SC/RM"은 **GPQA/HumanEval tie 누락 시 cherry-pick**. 반드시 "축 다양성+held-in val subset 선별" 조건 동반(없으면 우리 SEL-2 음성과 모순). (selector §C)
- **2502.11027 "different-wrong jams consensus" gloss** — 논문 본문 미지지(fast-model paraphrase 아티팩트). 쌍별다양성→oracle-BoN 이득 + "MV vanish"로만 인용. (diversity §1.1)
- **ATLAS-RTC `2603.27905`** — 가짜 ID → bib drop. (arch)
- **min-p `2407.01082`** — Schaeffer 반박(`2506.13681`) 진행중 = "검증된 다양성 레버"로 인용 금지·high-temp 안정자까지만. (diversity §1.6)
- **Koehn 2004** — 전문 미추출 = *표준 paired-bootstrap 방법*에만 인용·특정 수식/숫자 금지. (metrics §C)

---

## 2. 미래-날짜 ID 감사 (suspect 전수 — 전부 resolve, 가짜 1건만)

- **다양성**: 2606.05728(DiG-Plan)·2509.07430·2509.04784·**2603.19146(D5P4)=verified 승격**·2601.15808(✓실재이나 내용불일치=§1.1) — 전부 실재.
- **nlformalize**: 2603.15799·2604.19459·2512.09629·2603.17233·2510.06296·2508.04440·2410.10135·2404.07751·2503.18666·2510.15981 — 전부 resolve.
- **determinism**: 2601.06118·2601.17768(LLM-42)·2511.00025·**2604.22411(⚠️한 render가 허위-플래그했으나 arxiv 검색 실재확인·abstract만)** — 전부 resolve.
- **arch**: **ATLAS-RTC `2603.27905` = 가짜 → drop**; PLaT `2601.21358`·OATS `2603.13426` 실재.

**순결산**: full-read한 미래ID 중 **유일한 fabrication = `2603.27905`(ATLAS-RTC)**. 1건 내용불일치(`2601.15808`). 나머지 전부 실재.

---

## 3. 차기 액션 단서 (relwork → 다음 실험)

- **A2 faithfulness 구현**: cross-stage NL-gloss↔source entailment(`2604.19459` 처방, ⓐaxiom-fabrication 포착) + round-trip FormalAlign(`2410.10135`) + ⓑsilent-mistranslation=비가역 잔차→abstain. (nlformalize §8b)
- **S1-v2 템플릿**: StepFun ThinkingF dual-stream(verifier-filtered 지식 + 템플릿-가이드 추론; **템플릿>raw frontier CoT**) = S0-v2 합성과적합 설명. (nlformalize §8c)
- **v4 retrain 게이트**: GEM(+7.6 HumanEval Pass@100 verified)/DivPO(rare∧high-quality) = 가치 있으나 **게이트 "k0 강도 ∧ 풀 D-oracle ≥ baseline" 유지**(tool-DAG 미검증). (diversity §6b)
- **ⓟ1 재개**: ≥0.11.1 + `VLLM_BATCH_INVARIANT=1` + TP=1 + CC≥8.0 → 4-trial bitwise 동일이 성공기준. (determinism §8b)
- **★P-D 판정 확정 (2026-06-14)**: **diffusion 보류 + A3 실험 기각 + 생성기-arm 전체 선별기 대비 강등**. 근거: ①DiG-Plan이 diffusion-only 직렬화 실패·하이브리드 필수 이미 확증(raw parse 게이트=settled 재검증) ②P-D0 전수부검 = 0%는 decode-붕괴 artifact(steps<gen_len)+직렬화 약점(DiG-Plan 재현)으로 분해·diffusion verdict 불가 ③A3(`2601.13228`)는 표준AR에 짐 = framing 닻만(relwork_arch §3c "distraction") ④§3.7d 결합제약=선별기·day-5 헤드라인 +12.9pp. **생성기-side 유일 즉시후보 = XGrammar validity-floor**(zero-retrain·선별기 D-oracle 분모 안정화·다양성원 아님). 상세 = `TB_DIFFUSION_PROPOSER_DESIGN.md` §3 부검·결정.
