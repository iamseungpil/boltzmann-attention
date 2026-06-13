# A2 front-end 증류 설계 — NL 정책 → GATE_SPEC 컴파일러 학습 (detail, 2026-06-12)
> 📌 **구조 안내**: 모든 설계·실험 문서의 단일 마스터 = repo `scripts/distill/EXPERIMENT_DESIGN.md` (**§7 문서지도**; 목표·순서 변경은 마스터 §0-§4에서만). 처음 읽는다면 마스터부터.

> 발주 = 사용자 문답 (2026-06-12): "소형이 생성기 학습으로 frontier 정도 될 수 있나? 생성기 학습을
> 최적화하는 구조 설계 필요." 좌표 = `BENCH_PORTFOLIO` §3.8(생성기=교체부품·검증기=프로그램 불변),
> 마스터 §1.5(A2=유일 난제=thesis 상품형)·"세 컴파일러 대조군".

## 0. 왜 이건 LOCK이 아닌가 (regime 판별 — 선결)
LOCK이 죽인 것 = **실행-루프 내 결정-emission**(게더 도중 truth/derivation 생성 → fabrication).
A2 컴파일 = **오프라인·단발·닫힌 스키마·결정론 검증기 동반** 구조화 번역 — 우리가 반복 성공한
regime(v2 균형-DPO·D1 구조-DPO·v3·RFT+evaluator)의 형태. guided로 스키마 강제 가능(R1)·
K-샘플+검증기-선별 가능(R6)·검증 통화가 결정론(replay over/under-deny).

## 1. 과제 정의
- 입력: 도메인 정책 NL (τ² policy.md류·SOPBench SOP·Amazon SOP 텍스트).
- 출력: `GATE_SPEC` JSON (gate별 predicate·satisfiers{tool→required inputs}·applies_to·terminal/ask).
- **수용 기준 = 결정론 검증기**: Guard-2-동형 replay (gold 궤적 over-deny=0 ∧ 순서위반=0)
  + (가능 도메인) evaluator 대조. 검증기는 영구히 프로그램 (§3.8 불변).

## 2. ★데이터 엔진 — 역방향 렌더링 (데이터 기근 해소)
실 (NL, spec) 쌍은 ~22 도메인뿐 → 합성으로 확장하되 **GT를 구성으로 보장**:
1. **spec 샘플러 (프로그램)**: GATE_SPEC 문법 위 무작위 샘플 — predicate 종류(인증/확인/스코프/한도/
   시간창...)·satisfier 도구 시그니처(가짜 카탈로그 동반 생성)·applies_to 조합. 난이도 손잡이 =
   게이트 수·교차참조 깊이·예외절 수.
2. **NL 렌더러 (frontier, 도메인당 1회 비용 불요·무제한)**: spec → 정책 산문 K-스타일
   (격식/캐주얼/불릿/장문-교차참조/한영혼합). **spec이 먼저라 GT 완벽·검증기 불필요.**
3. **오염 통제**: 렌더 NL에 spec 용어 직노출 금지(별칭/패러프레이즈 — alias 마스킹 교훈)·
   스타일 다양성이 분포갭 완화의 본체.
- 산출 규모 목표: 5k-20k 쌍 (7B LoRA SFT 수 시간 분량).

## 3. 학습 사다리 (사전등록)
| 단계 | 방법 | 게이트 (통과 기준) |
|---|---|---|
| **S0 합성 SFT** | LoRA SFT + guided(spec JSON 스키마) | G-A2-1: held-out 합성 spec EM ≥90% ∧ **실 retail replay over+under-deny 합 ≤ frontier 단일샷** |
| **S1 실-도메인 verified distill** | frontier가 실 정책 22 도메인 컴파일 → replay 필터 통과분만 SFT 계속 | G-A2-2: **LODO** (N−1 도메인 학습 → held-out 도메인 컴파일) replay-검증 gate-F1 |
| **S2 on-policy DPO** | 자기 K-샘플 → 검증기 채점 → (통과, near-miss) 쌍 — **대조축=구조 정확성**(D1 교훈: 길이 탈교락 by 스키마 고정) | G-A2-3: S1 대비 K=1 정확도 ↑ ∧ 검증기-선별 후 동일 |
| 추론시 | K-샘플 + 검증기-선별·전원 탈락 시 **abstain→HITL** (F6 risk-coverage로 채점) | — |

## 4. 판정 프레임 — "frontier급" 주장의 정확한 형태
- 비교 단위 = **시스템** (생성기+검증기+K-선별), 단일샷 아님.
- **세 컴파일러 대조군** (마스터 등재분의 실측 무대): L0 파서 / frontier 단일샷 / 소형 K+선별 —
  동일 검증기·동일 도메인(retail+airline→22 LODO).
- 사전등록 헤드라인 예측: **소형(K=8+검증기-선별) ≥ frontier 단일샷** on held-out 도메인
  (replay 통과율·gate-F1). 근거: 과제가 닫힌 구조화 번역 + 정밀도가 검색 문제로 환원(N2 +8.8 동형)
  + 7B+scaffold>GPT-5 선례.
- 훈련의 존재 이유 명시(§3.8): 주권(망분리)·정책-Δ robustness·전이 주장 — 그 외 regime에선
  frontier 컴파일이 정답이라고 논문에 그대로 씀.

## 5. 리스크 (정직)
①합성→실 분포갭 (실 정책의 암묵·세계지식 의존 절 — 완화: S1 실-도메인 + abstain)
②검증기 커버리지: replay는 gold 궤적 범위만 검증 — gold-밖 over-deny는 미검출
  (완화: τ²류는 evaluator 대조 추가·합성은 GT 완전)
③긴 정책(10k라인 타깃)의 globality — 청크 교차참조 (2단계 stress-test로 분리, §1 RAG-대조 계획과 합류)
④spec 스키마의 표현력 한계 — 새 predicate 유형 등장 시 스키마 확장 비용 (버전 관리).

## 5b. ★NL→formalize 서베이 반영 (2026-06-14, `research_nl_formalize_2026_06_14.md` — 1차 검증·수치는 load-bearing 전 재확인)
- **★분야 표준 레시피 = generate→sound check→keep/repair/abstain→aggregate** = 우리 replay-검증기 형태와 동일 (= 방향 정합 확인).
- **★검증기 사각지대 — faithfulness (리스크② 정밀화)**: "Do LLMs Game Formalization?"(`2604.19459`)·FormalAlign(`2410.10135`) = **compile-pass ≠ NL-faithful**. 우리 replay는 *행동*(over/under-deny)만 잡고 — **fabricated gate가 우연히 replay 통과하면 못 잡음**. ⚠️정정(relwork_nlformalize §9.4): `2604.19459`는 "systematic gaming 없음"이라 명시 — 결과는 *두 특정 양상*(ⓐcross-stage 탐지가능 axiom-fabrication / ⓑ탐지회피 silent-mistranslation). **✅구현(2026-06-14, `tau2/t2_a2_faithfulness.py`)**: 각 gate의 NL-gloss ↔ source 정책 clause **entailment** 검사 → SUPPORTED/FABRICATED(ⓐ 포착·근거 clause 부재)/UNCERTAIN(ⓑ 잔차). 판정기 3중(LLM-judge `--judge` / 사전판정 `--judgments` / lexical 폴백). FABRICATED·UNCERTAIN 1건이라도 있으면 spec→**abstain→HITL**(F6). replay(level-2 행동)에 **직교·가산**되는 level-3 검사. **검증 (2026-06-14, 3단)**: ①`--selftest` PASS(lexical·결정론·무네트워크 — 주입 fabricated gate G9 포착·라우팅 abstain) ②실제 Fable-5 retail spec **`--judge` gpt-4.1**(openrouter, GPU 무관) = **G1/G2/G3 전부 SUPPORTED**(정확한 verbatim 근거 quote·conf 0.95~1.0)·**fabricated 0**·route=**TRUST**(known-good에 오경보 0) ③`retail_gate_spec_fabtest.json`(실제 3 + 주입 G9_LOYALTY_TIER) **judge E2E = 3 SUPPORTED + G9 FABRICATED(conf 1.0)**·route=ABSTAIN → **replay 사각지대 폐쇄 실증**(G9는 gold 궤적 미발화 시 behavioral replay 통과하나 cross-stage가 확실히 포착). ⚠️lexical 폴백은 형태소 약점으로 uncertain 과다 = *판정기 부재 시 보수적 하한*이지 운영 경로 아님(운영 = `--judge`). ⚠️인프라: 리모트 venv는 outbound HTTPS에 `SSL_CERT_FILE=$(python -c 'import certifi;certifi.where()')` 필요. VeriEquivBench(`2510.06296`)=ground-truth-free 등가 검사의 SOTA 천장(level-5, 미구현).
- **★S1 직계 템플릿 = StepFun-Formalizer(`2508.04440`, AAAI'26)**: 7B/32B dual-stream(ThinkingF) distill+RLVR = **유일한 검증된 소형-formalizer 선례** — S1-v2(P5 도착 후)의 레시피 모델. ❗**정정(relwork_nlformalize §9.2, Table 3 정독)**: "헤드라인 32B(7B 아님)"는 **틀림 — 역전**. **7B가 o3-pro·Claude-4-thinking·R1-671B를 이기고 7B≈32B** = A2 "소형 생성기 ≥ frontier" capacity-plausibility 최강 외부증거(크기 천장 아닌 distillation-deficit). 레시피 = verifier-filtered 지식 183K + **템플릿-가이드 추론 5.8K**(템플릿 추론 > raw frontier CoT 증류, Table 5 = S0-v2 합성과적합 설명). ⚠️math-only·in-domain 우위 = 전이는 우리 기여.
- **★A2 novelty 좌표 확정 (FIELD_GAP §5.6 박제)**: 최근접 과제 analog = **Prose2Policy(`2603.15799`, Apple — NL→Rego)** = A2와 가장 가까운 published 과제. **단 frontier-prompt-only·증류/전이/주권 無**. ❗정직분모(relwork_nlformalize §9.1): "95.3% compile"=371/389 **post-filter**(485→389, ~20% 거부) = 원입력 **76.5%**·양성테스트 자가채점·결정론=future work. **2nd-최근접 = AgentSpec(`2503.18666`, o1-규칙 recall 70.96%)** — P2P와 함께 필수 인용. ⇒ **리뷰어 필수 질문 "왜 그냥 frontier 프롬프트(P2P) 안 쓰나"의 답 = raw accuracy 아닌 ①주권(on-prem 소형) ②cross-domain 전이(LODO)** — A2 헤드라인을 이 둘로 고정. 4-way 교집합(swappable 소형생성기 + 고정 검증기를 *런타임 계약*으로 + verified-distill + SOP+주권)을 점유한 단일 논문 부재 = 검증된 공백.

## 6. 실행 순서 (큐 등재용)
P-A2-0 (zero-GPU): frontier로 retail+airline 컴파일 → replay 검증 — GT 파이프라인 생존성 + frontier 단일샷 baseline 수치 확보.
  **✅PASS (2026-06-12)**: Fable-5 in-session airline 단일샷 → 상태-추적 replay **over-deny 0/108** (PORTFOLIO §3.9).
**P-A2-0b 크기 하한 census (사용자 지시 2026-06-12 — R7을 A2 컴파일에 적용)**: 동일 프롬프트(스키마+retail 1-shot+airline 정책+A1 카탈로그)로 **7B/14B/32B-Int8(로컬)·72B(OpenRouter)** zero-shot 컴파일 → Fable-5 reference 대비 채점(게이트 매칭률·applies_to F1·핵심술어 recall) — **"Fable-5급 생성기의 모델 하한선" 확정**이 목적. 하한 위 모델=즉시 활용 가능, 하한 아래=증류 사다리(S0-S2)의 타깃·기대이득 정량화. ⚠️v1 채점은 구조·키워드 tier (생성 spec의 db_check prose replay는 DSL 후 = P-A2-1). 컨텍스트-공정성 각주는 외적-타당성 행에만 해당 — 하한 질문엔 무관(교사=컨텍스트 보유 Fable-5가 오히려 자산).
P-A2-1: spec 샘플러 + 역방향 렌더 5k → S0 SFT → G-A2-1.
  **✅부트스트랩 PASS (2026-06-12)**: ①spec 샘플러(`t2_a2_spec_sampler.py`, 프로그램·결정론·난이도손잡이) ②Fable-5 역렌더 시드(`specs/synth_seed_pairs_fable5.jsonl`, telecom-0 spec→formal/casual 정책NL) ③round-trip 검증기(`t2_a2_roundtrip.py`: 렌더NL 재컴파일→원spec 일치도=데이터청정 게이트) — self-sanity 1.0 ∧ Fable-5 재컴파일 **KEEP 2/2**(gate_recall·applies_F1·kind_match 전부 1.0). **루프 폐쇄 증명**: spec 무한생성→NL렌더→round-trip 필터로 청정 (NL,spec) 쌍 자동 확보. 다음 = 규모화(샘플러 N=5k + frontier 배치 렌더 + round-trip 필터 → S0 학습셋).
  **✅batch1 생성 완료 (2026-06-12, Fable-5 직접 작성 — 사용자 지시)**: 샘플러 seed-1 spec 30개 → **6스타일 로테이션**(formal-md/casual-prose/bullet-terse/legalese/**한국어**/mixed-crossref) 전량 수작업 렌더 → 기계적 충실성 QC(`t2_a2_join_qc.py`: 모든 gated 도구·satisfier·인자의 본문 등장 검사) — **30/30 보존** (`specs/a2_s0_batch1.jsonl`: exact 25 + paraphrase-manual-verified 5). ★QC 발견: casual 패러프레이즈("adjusting or closing a ticket")·한국어 번역("멤버십")은 QC v1 false-positive이자 **의미-매핑을 강제하는 고가치 학습사례**(alias 교훈과 동형) — QC v2=round-trip(LLM-tier)이 정식 필터. 규모화 경로: 회당 ~30쌍(Fable-5 세션) 또는 OpenRouter 배치 — S0 최소 학습셋 목표 200-500쌍.
  **✅dataset v1 = 90쌍 (2026-06-12, batch1-3 전부 Fable-5 직접 렌더)**: seed 1/2/3 × 30, 스타일 로테이션 시프트로 (spec유형×스타일) 격자 커버, batch2-3은 이중언어 토큰 병기로 QC exact 60/60 (수동감사 0). `specs/a2_s0_dataset_v1.jsonl`. 잔여 목표: +110~410쌍 → S0 SFT 발사 가능. 다양성 손잡이 다음 단계 = 샘플러 어휘 확장(현 VERBS×NOUNS 조합 재등장 시작)·게이트 수 6+·예외절(EXCEPT) 밀도.
  **✅P-A2-0b 로컬분 실측 (2026-06-12)**: 동일 프롬프트(retail 1-shot+airline 정책+카탈로그) zero-shot — **7B gate_recall 0.333**/applies_F1 0.815/9게이트(1-shot 게이트 3종 복사 + airline 적격성 G4-G9 파편 추출) · **14B 0.167**/0.698/4게이트(**1-shot 과앵커링 — airline 본체(취소적격·수정규칙·지불조성) 전부 누락**, 점수 역전이 채점 아티팩트 아님). 예측(7B<0.5) 적중·둘 다 frontier(ref 6게이트) 크게 미달 = **증류 사다리 기대이득 큼**. ⚠️n=1·키워드-tier. 32B/72B/235B = P4(coworker) 대기. 인프라 2건 수정: census `--model` 파싱(URL 콜론)·vllm 동시기동 `VLLM_PORT` 분리(EADDRINUSE).
  **✅S0 스모크 = 학습 신호 확정 (2026-06-12, dataset v3 135쌍 = 120 train/15 holdout 스타일균형)**: 7B LoRA r16 3ep SFT(`sft_runs/qwen7b_a2_s0`, census-동일 프롬프트로 빌드 `t2_a2_s0_build_sft.py`) → **held-in holdout gate_recall 0.564→1.000·applies_F1 0.974→0.996** (per-pair 구조점수 만점). ②EM 0/15(양쪽) = canonical-JSON EM이 prose 필드(predicate/ask 문구) 변주에 깨짐 — **G-A2-1의 EM 지표는 structure-EM(게이트 kind+applies_to+satisfiers exact, prose 제외)으로 정교화 필요**(사전등록 갱신은 마스터 경유). ③실도메인 airline 전이 **0.333→0.167 하락**(생성 게이트 9→5, 합성 추상-predicate 스타일로 수렴) = **합성→실 분포갭(리스크①) 실측 — S1(실-도메인 verified distill) 정당화**. 평가 하네스 = `t2_a2_s0_eval.py`·드라이버 = `driver_a2_s0_sft.sh`.
  **✅S0-v2 = 규모 효과 확정 (2026-06-12 야간, dataset v6 200쌍 = 180 train/20 holdout)**: ①held-in **structEM 0/20→14/20(70%)·canonical-EM 0→8/20(40%)**·gate_recall 0.248→1.000·applies_F1 0.982→0.997 — 135→200쌍 + 샘플러 v2 다양성으로 **정확-구조 재현이 창발**(v1은 EM 0). G-A2-1 기준(EM≥90%, structure-EM으로 재정의)에 아직 미달이나 궤적 가파름 — 규모 추가(+렌더 스타일 확장)로 도달 가능성. ②airline 전이는 **추가 악화**(applies_F1 0.876→0.528, gate_recall 0.167 동일) — 합성 데이터가 늘수록 합성 분포 과적합 심화(n=1 유의). **처방 = S1(실-도메인 verified distill) 필수**, 합성-only 규모 확대는 held-in 전용 레버. 어댑터 = `sft_runs/qwen7b_a2_s0v2`.
**★S1 스모크 설계·발사 (2026-06-13 — S0-v2 합성 과적합 처방)**: 학습셋 = 합성200(S0 held-in) + **실 도메인 (정책NL 전문, Fable-5 교사-컴파일 spec) 쌍**(retail+telecom, oversample 8) / **airline = held-out 평가축**(S0-v2 airline census 0.528과 직접 비교 = 동일 척도). telecom spec = Fable-5 in-session 컴파일 6게이트(`specs/s1_inputs/telecom_gate_spec_fable5.json` — 인증선행+G3/G4/G5 db_check[overdue·resume적격·2GB한도]+G6 transfer). **수용(설계 정신)**: 실 spec은 replay 필터 통과분만 — telecom 검증기 부재라 스모크는 **frontier 교사 신뢰**(P-A2-0 airline replay-clean 근거); P5(대형 교사)·SOPBench 도메인 합류 시 replay 정식 적용. **사전등록**: S1 airline applies_F1 **> 0.528**(=실-spec 도입이 분포갭 교정) — 미달 시 "실 2도메인으로 부족, P5 규모 필요"로 해석. 도구 = `t2_a2_s1_build.py`·드라이버 `driver_s1_sel4.sh`(NIGHT14_DONE 게이트 후 GPU0). ⚠️실 정책 길어(136~166줄) max-seq 8192 truncate 주의.
  **❌S1 스모크 기각 (2026-06-13 새벽 실행)**: airline held-out census **applies_F1 0.528 = S0-v2와 동일**(개선 0)·gate_recall 0.167·**n_gates 1**(생성 게이트 5→1개로 축소 — 실-도메인 장문 spec 모방이 오히려 출력 위축 유발 의심). **사전등록(>0.528) 명확 기각** — 실 2도메인×oversample8(16/216)로는 분포갭 교정 불가. 해석(사전 명시대로): **P5 교사-풀(대형모델×다도메인) 규모가 필요** — 실 도메인 다양성(SOPBench 7+ 합류)이 본질이지 oversample이 아님. 차기: P5 spec 도착 → replay 필터 → S1-v2 (+ 실-도메인 전용 가중/포맷 재검토: 장문 정책의 truncation은 없었음[max 2525tok], 의심축은 spec 복잡도 갭).
**★S1-v2 = dose-response 설계 (사전등록 2026-06-14, 리뷰 3순위 — 기각을 곡선으로)**: P5 teacher-pool 도착 시 real 도메인 수를 **1/3/6/9로 층화**해 4-arm 학습 → "실-도메인 노출량 → airline 전이(applies_F1)" 곡선 측정. 단일 점(성공/실패)이 아닌 곡선 = 실패해도 "얼마나 필요한가" 답 잔존·성공 시 스케일링 근거. 사전등록: 단조 증가 ∧ 도메인 6+에서 0.815(base) 회복. 하네스 = `t2_a2_s1_build.py` --real 가변(준비 완료 — P5 spec 도착 즉시 발사 가능; replay 필터 = retail/airline 검증기 + telecom-급 frontier 신뢰).
**★S1-diag (사전등록 2026-06-13 — S1 기각의 가설 3분리, P5 비의존·즉시 발사)**: 기각 부검에서 "도메인 다양성 부족" 외 경쟁가설 2개를 0~저비용으로 분리 — **H4 구체성 갭**(합성 db_check=`"<...>"` 플레이스홀더 ↔ 실 spec=구체 조건문: 모델이 구체 술어 작성을 학습한 적 없음 — 분포갭의 가장 날카로운 후보) ⇒ 처방 = **구체-술어 합성 60쌍**(`t2_a2_concrete_gen.py` — 술어·db_check·산문 전부 프로그램 인스턴스화, 렌더 비용 0) v7 병합 재학습(s1d). **H3 형식 불일치**(census=1-shot ↔ 학습=zero-shot) ⇒ `--no_oneshot` 평가 추가. **arms** = {base, s0v2, s1d}×{1shot, 0shot}. **사전등록**: ⓗ4 s1d airline applies_F1 >0.528 ∧ n_gates≥3 / ⓗ3 0-shot에서 어댑터 점수↑(base는 ↓ 예상) / 둘 다 기각 = "스타일·형식 아님 → 도메인 다양성(P5 dose-response)만 잔존"으로 가설 공간 축소. 드라이버 `driver_s1d.sh`(DAY13_DONE 게이트 후 GPU0 자동).
  **◐S1-diag 결과 (2026-06-13 day13 [G] 실행)**: 1-shot census — base 0.815/9게이트 · s0v2 0.528/6 · **s1d 0.704/1게이트**. **H4 부분 적중**: 구체-술어 합성(60쌍) 추가가 applies_F1 **0.528→0.704**(+18pp) — 구체성 갭이 정확도 손실의 *일부* 확정. **단 n_gates=1 위축 지속**(사전등록 ⓗ4 = applies_F1>0.528 ✅ ∧ n_gates≥3 ❌ = 부분). ★**새 1급 병목 부상 = 게이트-수 위축**: 실 spec(소수 게이트) oversample 모방으로 모델이 "게이트를 적게 생성"하게 학습 — 구체성과 독립 문제. 0-shot(H3): base 0.500→오히려 **22게이트 생성**(예시 없으면 과생성)·**s1d는 parse 붕괴(0/0)** = 어댑터가 1-shot 형식에 *고착*(H3 역방향 — 형식 불일치 아니라 어댑터 과적합). **차기**: ①게이트-수 위축 직접 처방(실 spec oversample↓·게이트-수 다양화 합성) ②P5 dose-response(도메인 다양성)는 여전히 유효. 어댑터 = `sft_runs/qwen7b_a2_s1d`.
P-A2-2: 실 22 도메인 verified distill + LODO → G-A2-2.
P-A2-3: on-policy DPO → G-A2-3 → 세-컴파일러 표 완성 = thesis front-end 헤드라인.
