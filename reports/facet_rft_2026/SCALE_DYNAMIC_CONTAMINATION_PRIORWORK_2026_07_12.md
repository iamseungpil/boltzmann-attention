# scale의 구매물 = 동적-오염 내성? — 선행연구 지형 정본 (2026-07-12)

> 상위 = `RESEARCH_MASTER.md`. 이 문서 = 딥리서치 synth 워크플로 `wf_ada23267-9aa`(밤샘·무료)의 **수동 종합**.
> 워크플로가 검증 106 / 결과 100(claim 179 + verdict 36 + search 17 + scout 3) 산출 후 **Synthesize 직전에 종료**(resume=same-session-only) →
> journal(`subagents/workflows/wf_ada23267-9aa/journal.jsonl`)에서 claim+verdict 복원해 손으로 종합.
> **인용 규율([[40]])**: 등급 = **[지지]=verdict 적대검증 통과(refuted=false)** · **[반박]=verdict가 반증(refuted=true)** · **[미검]=claim 추출됐으나 verdict 미실행**(워크플로 조기 종료). **[미검]을 [M]/[지지]로 쓰지 말 것** — 논문·특허 인용 전 verification 패스 필수.
> verdict 커버리지 = **36/179 claim**(축 a·snowballing·fundamental-limits에 집중). 축 b/c/d는 대부분 [미검].

---

## 0. 한 줄 판정
명제 **"scale이 사는 것 = 동적-오염 내성"** 은 선행에서 **축별로 갈린다**:
- **[지지·강함]** scale이 사는 것 = **실행-길이/horizon**(2509.09677·verdict-verified). = 우리 등대 **F6**과 동형.
- **[반박·강함]** scale이 사는 것 = **동적-오염 균일 면역**(강한형): self-conditioning은 scale-불변 잔여(2509.09677)·sycophancy는 scale로 악화(Perez/Sharma·[미검])·멀티턴 붕괴 보편(Laban·[미검])·비의미 entrainment 악화([미검]). **강한형 = 반증.**
- **whitespace(에이전트 tool-use 궤적 × 오염-축별 통제주입 × scale 사다리) = 미선점.** 단 2509.09677 = **가장 강한 인접 선행**(수렴)·반드시 인용·구분.

## 1. 연구 질문 (scout)
"scale이 사는 핵심 능력 = 동적오염(dynamic contamination) 하 성능 유지" 명제의 선행 지형을 4 오염축으로 수집·판정:
- **(a)** 자기-생성 오류 눈덩이 / self-conditioning
- **(b)** 사용자 허위단정 수용 / sycophancy
- **(c)** 문맥 내 유사-오답 distractor 정박 / contextual entrainment
- **(d)** 긴 멀티턴 상태-추적 / binding 붕괴

각 축 × {scale 강건성 곡선 · clean-vs-contaminated 격차의 scale-의존 · 반증(frontier도 붕괴) · multi-turn pass^k 곡선} + whitespace 선점 여부.

## 2. 검증된 백본 (verdict-verified [지지])

### 2509.09677 — "The Illusion of Diminishing Returns: Measuring Long Horizon Execution in LLMs" (Sinha, Arun, Goel, Staab, Geiping; 2025-09, v3 2026-03, **ICLR 2026**) ★★★ 중심소스
verdict V0-V13 전부 refuted=false·high. **실제 scale 사다리**(Qwen3 4B/8B/14B/32B·Gemma3 4B/12B/27B·frontier Kimi-K2/DeepSeek-V3/Qwen3-235B). verbatim:
- **[지지]** "larger models can correctly execute significantly more turns even when small models have near-perfect single-turn accuracy" → **scale = 실행-길이 구매**(단일턴 능력 동일해도). = 우리 **F6 horizon**.
- **[지지]** self-conditioning 명명·측정: "models become more likely to make mistakes when the context contains their errors from prior turns" ≠ 단순 long-context 한계.
- **[지지]** **"Self-conditioning does not reduce by just scaling the model size"** (Fig 5b: 유도 error-rate↑ 시 전 scale서 성능 저하) → **self-conditioning = scale-불변 잔여**. = 우리 [[45]] load=scale-불변.
- **[지지]** **"thinking mitigates self-conditioning"**: Qwen3 thinking 모델은 turn-100 정확도가 문맥 error-rate 무관 안정 → **비-scale 레버(test-time compute)가 닫음**. = 우리 **F2 symbolic**. GPT-5 ~2176 step vs Claude-4 Sonnet 432 / Grok-4 384(단일턴).
- ⚠️ **구분 필요**: 이 논문 = 우리 프레임에 *매우 근접*하나 = **단일 합성 key-value 실행 task**·**축 (a) self-conditioning 단독**. 오염 4축 통제주입 아님·실제 agentic tool-use 궤적 아님.

### 2305.13534 — "How Language Model Hallucinations Can Snowball" (Zhang, Press, Merrill, Liu, Smith; 2023, **ACL 2024 Findings**)
verdict V11-V17 refuted=false·high. 축 (a) 존재 증거:
- **[지지]** "an LM over-commits to early mistakes, leading to more mistakes that it otherwise would not make" = snowballing 형식화.
- **[지지]** ChatGPT/GPT-4가 자기 오류를 격리 질의 시 **67%/87% 인식**하나 궤적선 여전히 snowball → **capable-but-fails-under-contamination**(능력결손 아님). = 우리 명제 "추상능력 있음·간섭이 범인" 정합.
- **[미검 caveat·verdict가 명시]** scale 사다리 없음(ChatGPT vs GPT-4만) → **축 (a) 존재만 지지·scale-의존 곡선 근거 아님**.

## 3. 축별 지형

### 축 (a) self-error snowballing — [지지·백본 §2]
2305.13534(존재) + 2509.09677(scale 사다리·self-conditioning scale-불변·thinking이 닫음). **가장 견고**. 결론: 축 (a)는 **scale-불변 잔여**, 닫는 레버 = thinking(비-scale).

### 축 (b) sycophancy — [미검·반증 방향]
- **[미검]** 2212.09251 Perez "Discovering LM Behaviors"(model-written evals 154·crowd 90-100%): **sycophancy가 scale로 증가**(강한형 명제 반증). RLHF가 악화.
- **[미검]** 2310.13548 Sharma "Towards Understanding Sycophancy": 5 SOTA RLHF assistant 전반 일관·preference-model 최적화가 truthfulness↔sycophancy 맞바꿈.
- **[미검]** SYCON-Bench: multi-turn sycophancy(Turn-of-Flip·Number-of-Flip·17 LLM).
- 결론: 축 (b)는 **scale로 안 사짐·악화 가능**. 단 MCQ/opinion eval 측정 → **agentic tool-use 통제주입 아님·whitespace 미선점**.

### 축 (c) distractor / contextual entrainment — [미검]
- **[미검]** 2307.03172 "Lost in the Middle"(Liu 2023): 위치 U자 곡선(중간 정보 정확도 저하).
- **[미검]** 2302.00093 GSM-IC(Shi 2023): irrelevant context distractor.
- **[미검]** contextual entrainment 논문: 문맥 등장 토큰에 무관하게 높은 logit·entrainment heads(ablation으로 완화)·counterfactual이 factual보다 강한 pull. ★**scale 비균일**: "larger LLMs become MORE resistant to false/counterfactual claims(semantic) while MORE prone to copying arbitrary/irrelevant tokens(non-semantic)". → 우리 **C43 정박치환**([[46]])과 수렴.
- 결론: 축 (c)는 **scale 비균일**(의미 distractor 저항↑·비의미 copy↑). 강한형 균일면역 반증.

### 축 (d) multi-turn / binding 붕괴 — [미검·반증 방향]
- **[미검]** 2505.06120 Laban "LLMs Get Lost in Multi-Turn Conversation": 15 LLM·6 task·**평균 39% 저하**·멀티턴(underspecified sharded) vs 단일턴. **frontier도 소형만큼 저하·scale 곡선 없음**(단 저자 caveat: 상대저하 지표가 scale효과 가릴 수 있음). 분해 = aptitude 손실 ~16% + **reliability 붕괴 +112%**(unreliability 2배). **reasoning 모델(o3·R1)도 안 고침**. → 우리 **pass^k reliability** 프레임과 직접 수렴.
- **[미검]** MultiChallenge: instruction retention·context allocation·in-context reasoning 4 challenge.
- **[미검]** 2406.12045 τ-bench(Yao 2024)·2506.07982 τ²-bench(Barres) = 우리 벤치.
- 결론: 축 (d)는 **보편 붕괴·비-scale-gated**. reliability 붕괴 지배 = capable-but-fails 정합.

## 4. 반박된 claim (인용 금지 — verdict가 반증)
- **[반박] 2606.07937** "Hallucination Cascade: Multi-Agent"(V30/31/33): "smaller GPT-5.3" = **날조**(efficiency≠scale·GPT-5.3 크기 비공개·frontier). multi-agent≠multi-turn. cascade 깊이서 hallucination *감소*(0.422→0.272)=명제 역방향. **scale 증거로 인용 금지.**
- **[반박] 2505.17656** "Too Consistent to Detect"(Tan, EMNLP 2025)의 **self-conditioning 형식화 주장**(V20/22/23): self-consistent error = **독립 i.i.d. 샘플**(detection)·궤적 self-conditioning 아님 = **범주오류**. 단 scale 발견 자체(self-consistent error가 scale로 stable/증가·V18/19/21 [지지])는 **축 무관 유효**(scale-불변 잔여 방증).
- **[반박] 2511.12869** "On the Fundamental Limits of LLMs at Scale"(Mohsin et al.)의 **강한 주장들**(V26/27/28/29): "architectural constraint scale can't fix"가 RULER(2404.06654: 34B>6B long-context)·position-frequency 가변수정(2410.18745 StRing)·Qwen2.5(2412.15115)에 반증. exposure-bias·sycophancy는 논문에 "not found". 단 논문의 **일반 thesis(scale이 실패를 amplify)**는 [지지·V24/25] = **명제 역방향 선행**(scale-amplifies).

## 5. whitespace 선점 판정 (핵심 산출)
**미선점.** "scale이 사는 것 = 동적오염 내성"은 선행에서 **부분 지지(horizon) + 강한형 반박(균일면역) + 축별 분기**로 존재하나, 우리 whitespace =
> **에이전트 tool-use 궤적 × 오염-축별(a/b/c/d) 통제 주입 × scale 사다리**
는 어느 선행도 안 함:
- 2509.09677 = 단일 합성 실행 task·축 (a) 단독(다축 통제주입 아님·실 tool-use 아님).
- sycophancy(b) = MCQ/opinion eval.
- entrainment/lost-in-middle(c) = QA/retrieval.
- Laban(d) = 생성 task·오염-축 분해 아님.
- snowballing·CHARM = scale 사다리 없음.

**그러나 2509.09677은 선점이 아니라 수렴 증거** — 인용 필수·구분 명시("우리는 실 agentic 궤적에서 4축을 개별 통제주입해 scale 사다리 측정·2509.09677은 단일 실행축의 self-conditioning만").

## 6. 우리 결과와의 수렴 (positioning)
- **F6 horizon = scale 구매물**: 2509.09677의 "scale=실행길이" = 등대 §1 프레임 직접 지지.
- **[[45]] load=scale-불변**: self-conditioning scale-불변(2509.09677) + 멀티턴 붕괴 보편(Laban) + self-consistent error scale-안정(2505.17656) = **동적오염 잔여는 scale로 안 닫힘** 3중 수렴.
- **F2 thinking 레버**: 2509.09677 "thinking fixes self-conditioning" = 우리 present/게이트 아닌 test-time compute 레버 지지(단 Laban은 reasoning도 멀티턴 안 고침 = thinking이 만능 아님·축의존).
- **C43 정박치환([[46]])**: entrainment 논문의 "semantic 저항↑·non-semantic copy↑" = 우리 정박치환 비균일성 수렴.
- **밤샘 E-REF(OVERNIGHT §1)**: 바인딩 1.5B emergent·정적오염 강건·**동적오염이 범인** = 2509.09677 self-conditioning-scale-불변과 독립 수렴. [[00-thesis]] 강화.

## 7. provenance + 다음
- **원본**: journal `wf_ada23267-9aa`(로컬 세션 `31d57683-…`). 복원 스크립트 = scratchpad `extract_synth.py`·추출물 `synth_extract.md`(claim 179·verdict 36 전문).
- **★[미검] 처리**: 축 b/c/d 소스(Perez 2212.09251·Sharma 2310.13548·Laban 2505.06120·Lost-in-Middle 2307.03172·GSM-IC 2302.00093·entrainment·MultiChallenge·SYCON-Bench)는 **verdict 미실행**. 논문 relwork/특허 인용 전 **arXiv ID·연도·주장 정합 verification 패스**(딥리서치 재발사 or WebFetch 개별검증). 특히 Laban 2505.06120·entrainment 논문 = 축 d/c 대표라 우선.
- **인용 즉시 가능([지지])**: 2509.09677·2305.13534(+ 반박 대조군 2606.07937 인용금지·2511.12869 강한주장 인용금지).
- **whitespace 확정**: 미선점·2509.09677 수렴 구분. Paper1 relwork "동적오염 축" 절 + 특허 positioning에 반영.

## 8. ★2509.09677 심층 positioning (정독·WebFetch v3·2026-07-12)
가장 강한 인접 선행이라 별도 심층. **"The Illusion of Diminishing Returns: Measuring Long Horizon Execution in LLMs"** (Sinha, Arun, Goel, Staab, **Geiping**; 2025-09-11 v1 / v3 2026-03-13; **ICLR 2026**).

### 8.1 논문 요지 (정독)
- **task = 순수 합성 key-value 룩업 + 누적합**: 사전(5글자 단어→[-99,99] 정수)·플랜(뽑을 키 K개)을 **다 주고** 매 턴 `Sₜ=Sₜ₋₁+ΣD[kₜ]`만. 지식·계획 제공 = **실행(execution)만 격리**(추론 아님). **도구호출·대화·사용자 0**.
- 모델: Qwen3 4B/8B/14B/32B/235B · Gemma3 4B/12B/27B · frontier(GPT-5·Claude-4-Sonnet·Grok-4·Gemini-2.5·Kimi-K2 1026B·DeepSeek-V3 670B/R1).
- **5 발견**: ①scale=실행길이 구매(단일턴 완벽해도 큰 모델이 더 많은 턴·미세개선→task길이 지수증가) ②스텝정확도가 턴수와 붕괴(Qwen3-32B K=1인데 15턴 내 <50%) ③self-conditioning(문맥 오류율 0/25/50% 주입→turn100 저하·long-context 아님) ④**scale 못 고침**(200B+도 저하) ⑤**thinking 고침**(Qwen3-thinking turn100 오류율무관 안정·단일턴 horizon GPT-5 **2176**·Claude-4-Sonnet 432·Grok-4 384·비-thinking 4-6스텝 붕괴). Appendix: 실벤치(ALFWorld/GAIA/WebShop) 실패의 ~20-48%가 self-conditioning 패턴.

### 8.2 우리와 같은 점 (수렴)
| 논문 | 우리 |
|---|---|
| scale이 사는 것=실행길이/horizon | F6·C5 [S-lit] "scale 유일 축=horizon" |
| 작은 모델 단일턴 능력 보유 | C66 바인딩 1.5B emergent |
| self-conditioning=scale-불변 잔여 | C69·[[45]] 동적오염 scale-불변 부하 |
| 잔여는 비-scale 레버(thinking)가 닫음 | 잔여=비-scale scaffold가 닫음 |
| capable-but-fails(능력결손 아님) | C43·전 census |
⇒ **Paper1 헤드라인의 가장 강한 독립 선행 확증**(합성 task 엄밀 사다리).

### 8.3 우리와 다른 점 (whitespace)
1. **도메인**: 그들=합성 key-value+산술(도구·대화·사용자 0) / 우리=실제 agentic tool-use(τ²·멀티턴·DB부작용).
2. **오염축**: 그들=1축(self-conditioning=*자기* 오류) / 우리=4축 통제주입(a자기·b*사용자*sycophancy·c distractor·d 바인딩). C69 near-miss=*외생* 주입=자기생성과 오염원 다름.
3. **레버**: 그들=thinking이 THE 레버 / 우리=결정론 scaffold, **thinking은 부분레버 실증**(Laban reasoning o3/R1 멀티턴 못고침·C56 동-scale thinking ⋈ 못삼) → **thinking-fixes는 축 a(합성) 국한·축 d(agentic 바인딩/⋈)엔 무효 = 축-의존성은 우리 소유**.
4. **guarantee 차원**: 그들=실행정확도 / 우리=**compliance(위반0 게이트)** = 그들 프레임에 없는 직교축. 모트=compliant-pass crossover(C1/C2).
5. **비용 crossover**: 그들=pro-scale(클수록 길다) / 우리=소형+scaffold≈대형 신뢰성·비용 knee(pro-scaffold).

### 8.4 특허 관점
- 선행리스크 **제한적**: 이 논문=**진단**(실행길이+self-conditioning)이지 **장치** 아님. 우리 청구(provenance검증기·Δspurious≤0 게이트·오프로드·도메인-일반 TBox·ABox-swap)를 안 가르침.
- **오히려 유리**: "scale 혼자 잔여 못 닫음" 확립→비-scale scaffold **motivation 강화**(명세 §배경 인용).
- **★지뢰**: 청구항이 "test-time compute로 누적오류 오염 감소"에 읽히면 **선점당함** → **thinking 청구 금지**(우리는 애초 안 함=안전).
- **공개**: 2025-09이라 출원 前 공지자산 개시 대상·positioning="수렴 동기 인용·장치로 차별".

### 8.5 논문(Paper1) 관점
- relwork **최우선 인용**(헤드라인 독립확증). "scale buys execution length"=양보(인용·위에 쌓기).
- **살아남는 노벨티**: (a)compliance/guarantee 차원 (b)crossover+비용 knee (c)실 agentic 다축 통제주입 (d)오프로드/TBox-전이 장치.
- **★예상 반론**: "2509.09677이 이미 scale=horizon+비-scale레버 보였다·뭐가 새로운가?" **답**: 그들 레버(thinking)=*그 자체가 inference-scaling*·**합성 단일축**만 닫음. 우리=**결정론·pass-비용-0 scaffold**가 **실 다축 agentic** compliance 닫고 **비용 crossover 정량화**.
- **★정직할 긴장**: 그들 "thinking이 self-conditioning 고침" ↔ 우리 "scale/thinking 잔여 못닫음" 부분충돌 → **축별 스코프 필수**: thinking=축 a(자기오류·합성) 닫되 축 d(멀티턴 바인딩/⋈) 못닫음(Laban·C56). 이 축-의존성 명시적 소유가 안전.
