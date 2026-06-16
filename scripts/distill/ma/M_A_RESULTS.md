# M-A 결과 (권위본) — write-벽/정밀도-벽 root cause 확정 = REASONING (NL→formalize), not fabrication — 2026-06-16

> 설계 = `../M_A_PROTOTYPE_DESIGN.md`. 도구 = `ma_resolver.py`·`ma_gold_extract.py`·`ma_eval.py`·`ma_trace.py`·`dist_overcall_trace.py`. 권위([[feedback_results_master_doc]]).
> 결론 = **선택기+resolver 아키텍처는 fabrication을 제거(필요)하나 NL→formalize *추론* 잔여는 못 닫음(불충분) → write-벽 본체 = σ(NL→formalize) 학습 대상.** 설계 사전등록 §10 "음성=diagnostic gold" 시나리오 확정.

## 1. M-A 3-arm 결과 (base Qwen2.5-7B·무재학습·retail exchange 29케이스/32items·offline 값-정확성)
| arm | item-acc | case-acc | 비고 |
|---|---|---|---|
| A concrete-emit (카탈로그서 item_id 직접) | 0.438 (14/32) | 0.379 (11/29) | parse_fail 0 |
| **B formal+resolver+xgrammar** | **0.469 (15/32)** | 0.414 (12/29) | resolver_fail 8·wrong_criteria 9 |
| C formal·xgrammar 끔 | 0.438 (14/32) | 0.414 (12/29) | parse_fail 3(=grammar가 form만 보장 재확인) |

**B ≈ A (순이득 +1 item) = 압승 아님.** C(xgrammar 끔)서 parse_fail 3 발생 = xgrammar는 *form*만 보장([[reference-nl-formal-decouple-literature]] 정합)·정확도엔 무관.

## 2. ★전수 궤적 추적 — A-vs-B 짝 대조 (32 items)
| 결과 | 수 | 의미 |
|---|---|---|
| both correct | 8 | 쉬운 케이스 |
| **B-only (아키텍처가 FIX)** | **7** | **fabrication** = A가 *의도는 맞는데* 엉뚱 item_id 선택·B는 옳은 criteria→resolver 정답 |
| A-only (B가 깨짐) | 6 | selector overhead/availability-blind |
| **both wrong** | **11** | **reasoning-bound**(아키텍처 불변·둘 다 의도 자체 틀림) |

⇒ **fabrication 몫=7 / reasoning 몫=11+** (both-wrong 11 + B의 wrong_criteria가 추론). 아키텍처는 fabrication 7을 닫지만 그만큼(6) 새로 깨 **순효과 ~0** — 결정 요인은 **reasoning 잔여**.

## 3. ★★root cause 확정 — wrong_criteria 9건 = "변경 오계산"
B가 emit한 select_by를 gold/old와 대조하면 지배 패턴 = **"X만 바꾸고 유지"를 LLM이 잘못 계산**(옛 옵션 echo·엉뚱 필드 변경):
- task 6: gold{silver,**low**,battery} ← B{white,**high**,USB} (옛것+엉뚱)
- task 18: gold{**gray,fixed**,…} ← B{**blue,none**,…} = **옛것 전부 그대로(변경 0)**
- task 45: gold{canister,**bagless**,…} ← B{canister,**bagged**,…}(옛것 유지)
- task 49: gold{blue,**6h**,IPX4} ← B{blue,**4h**,IPX4}(옛 배터리 유지)
- task 64: gold{4K,yes,**black**} ← B{4K,yes,**silver**}(옛색 유지)
- task 70: gold{M,**blue**,high} ← B{M,**red**,high}(옛색 유지)
- task 91: gold{8in,WiFi,**32GB**} ← B{8in,WiFi,**8GB**}=옛것 그대로
- **task 106·107: 정답을 *fallback*에 넣고 select_by엔 옛것** → resolver가 select_by 우선 → 오답(=선호순서 오인코딩=추론)

**= NL→formalize 추론 실패(무엇을 바꾸고 무엇을 유지·조건부 선호). resolver는 틀린 criteria도 충실히 resolve → 추론 오류가 그대로 출력.**

## 4. resolver_fail 8건 = availability-blind + fallback 미인코딩
- **task 0(대표)**: B가 {clicky,**RGB**,full} emit·**fallback("no backlight") 미인코딩** → unavailable → None. **반면 arm A는 카탈로그 availability 보고 정답 7706410293 선택**. ⇒ **selector는 availability-blind**(criteria를 가용성 모르고 사전확정)·"의도 맞는 *가용* variant 고르기"를 못 함 = pure-selector 아키텍처의 구조적 한계(resolver가 fallback 못 받으면).
- task 52/58/93/94/95(laptop): partial criteria가 비가용 조합 or 승계 충돌 → None.

## 5. ★dist 정밀도-벽 전수 추적 (별 실험·`dist_overcall_trace.py`)과 합류
- dist FINAL: **샘플당 6.02 도구 예측 vs gold 2.97(~2배)·499중 458(92%)이 과다호출**. 과다호출 정체 = **습관적 공통 action 도구 반사 emit**(send_sms +351·send_email +295·share_by_social +162·make_voice_call +109). **base는 2.65≈gold·과다호출 3/222뿐** → 과다호출은 **SFT가 유발**(subset-tool 학습 confound). [[project-nativefc-fullcatalog-collapse]]
- = node-선택 차원의 동일 결론: **concrete-emit SFT는 정확한 *선택 reasoning*을 못 가르치고 나쁜 습관(과다호출)만 induce.**

## 6. ★★통합 확정 (두 실험 합류)
**write-벽(인자값)·정밀도-벽(노드선택) 둘 다 root cause = "intent→formal *선택 reasoning*" 실패이지 fabrication/format 아님.**
- 선택기+resolver 아키텍처 = fabrication 제거(필요·M-A 7건 입증)·**but NL→formalize 추론은 못 offload**(불충분·11+ reasoning 잔여).
- concrete-emit SFT = 추론 안 가르치고 과다호출 습관 induce(dist 정밀도 0.44·완주 후퇴).
- ⇒ **thesis 방향 확정**: 학습 타깃 = **NL→formalize 선택 reasoning(P4-σ·"무엇 바꾸고 유지"·조건부 선호)**. resolver/xgrammar는 fabrication·form 가드(보조). **σ를 RLVR/SFT로 *학습*해야 벽이 닫힘**([[feedback-nl-formalize-llm-selection-deterministic]]·[[feedback-thesis-tbox-transfer-direction]]).

## 7. 한계/후속 (정직)
- offline 값-정확성(gold call 대비)·full τ² rollout 아님(M-E). base 모델 1개(7B)·29케이스(소수)·retail single-config.
- selector availability-blind 한계(§4)=설계 보정 필요(resolver가 "가용 중 best-match" 폴백 자동확장 or 모델에 availability 노출).

### ★7b. 교란변수 — "reasoning 실패" 결론은 미완 (2026-06-16 정밀화·[[feedback-capability-vs-artifact-elicitation]])
§3 "wrong_criteria=reasoning" 결론엔 **두 교란**이 남아 capability 단정 못 함:
1. **forced-JSON이 추론 억압**: arm A/B는 `guided_json`으로 JSON 즉시 강제(추론공간 0). NL→SQL 딥리서치 경고 = constrained decoding이 reasoning 떨어뜨림([[reference-nl-formal-decouple-literature]]). ⇒ **CoT-then-formalize 미시험** = "reasoning 실패"가 아티팩트일 수 있음.
2. **과다호출(별 실험·§5)은 capability 아님·SFT 손상**(base 7B 멀쩡 2.65≈gold·dist만 6.02). scale로 푸는 게 아니라 그 SFT 안 하면 사라짐.

### ★7c. 다음 실험 = scale × elicitation × arm 2D (1D scale 금지)
| | forced-JSON | **CoT-then-formalize** |
|---|---|---|
| 7B / 14B (로컬·먼저) | 기준 | ? (교란 배제) |
| 32B / 72B (coworker) | ? | ? |
- arm A(concrete) vs B(formal+resolver). **결정질문 = "결정론 offload(B)가 scale을 *대체*하나?"** — B가 7B를 32B-A 수준으로 끌면 아키텍처가 capability 대체(강한 주권결과). γ천장이 offload 무관 scale로만 닫히면 γ=환원불가 capability 핵.
- **구현됨**: `ma_eval.py` arm Acot/Bcot(CoT·trailing JSON·grammar끔)·`ma_eval_scale.sh`(7B+14B 스윕). **싼것 먼저→그 다음 coworker 32/72B**(node_run·AWQ/2-GPU·사양=A/B×scale×CoT). 교란 미배제 상태서 큰모델 compute 금지.
- **★선행연구 = 설계서 §11**: forced-format이 추론 해침(Tam EMNLP24·GCD-parser ACL25)·**CRANE**(`arXiv:2502.09061`·ICML25·자유추론+제약된최종·+10pp)·**vLLM `structural_tag`**(0.11 내장·CRANE delimiter-gating 제품화·코드검증). ⇒ arm **Bstag**(structural_tag=자유CoT+최종만 strict-grammar) 추가 예정 — Bcot(grammar끔·무보장)보다 엄격유효성 유지. **신규성 경계(정직)**: 디코딩 처방(CRANE/structural_tag)은 *채택*이지 기여 아님·기여=selector+resolver+전이 분담. 전수 신규성=딥리서치 `wf_3f814306-3e4` 확정중.

### 7d. 그 다음
- 교란 배제·scale 기울기 확인 후 → **M-σ: NL→formalize 선택 reasoning(γ-grounding) 학습**(SOPBench/TaskBench 궤적→(NL,config,target-spec) 삼중쌍·"change-X-keep-rest"·등방화로 표면덮음)→ held-out config 전이(M-D). σ=증명·γ=등방화구성+실증(§5.10).

---

## 8. ★★FLOOR SWEEP 결과 (2026-06-16·decisive) — A·Bfair·L0–L3 × {7B,14B,32B-Int8} + 비용
> 출력=concrete 고정(입력 효과 분리)·`ma_eval_scale.sh _floor`·비용계측. ⚠29케이스(±~6pp 노이즈)·32B=GPTQ-Int8·L1≈A(±2 item=프롬프트문구).

| arm (입력수준) | 7B | 14B | 32B-Int8 | tok/case |
|---|---|---|---|---|
| L0 (availability 없음) | 0.375 | 0.531 | 0.594 | ~855 |
| L1 (full+avail) | 0.531 | 0.688 | 0.750 | ~900 |
| L2a (가용필터·raw) | 0.406 | 0.656 | 0.812 | ~625 |
| **L2b (가용·표 formalized)** | 0.531 | 0.625 | **0.844** | **~508** |
| L3 (diff 주석) | 0.406 | 0.656 | 0.750 | ~628 |
| A (concrete baseline) | 0.438 | 0.719 | 0.719 | ~918 |
| Bfair (공정-정보 selector) | 0.375 | 0.500 | 0.656 | ~810 |

### 4 확정 (정직·일부 음성)
1. **정보 floor 실재·scale-불변**: L0→L1 = **+16pp 모든 scale 동일**. availability=fallback 필수=info-limited 성분 → **정보 제공(MSC/scaffold)으로 닫힘·모델크기 무관**.
2. **★MSC가 scale 대체 *못 함*(음성)**: L2b@7B=0.531 ≪ L1@14B=0.688 ≪ L2b@32B=0.844. **7B는 입력수준 무관 ~0.53 천장**(L0–L3 0.38–0.53)=**reasoning-limited·scale-bound**. ⇒ [[project-decomposition-optimality-contribution]] **조건#4(잔여추론⊆작은모델) 위배 for 7B**. sovereignty "작은+MSC≈큰" 미지지.
3. **★formalize(L2b)는 비용-Pareto 우월(양성)**: 같은/높은 정확도에 **토큰 최소**(L2b~508 vs A~918). 32B서 L2b=0.844(최고)@최저토큰. ⇒ MSC 가치 = scale 대체 아니라 **정보floor 보장 + 비용효율 + *큰* 모델 증폭**.
4. **★selector(Bfair) 공정정보로도 concrete에 짐(음성)**: Bfair < A 전 scale(14B 0.50<0.72). 카탈로그 보이면 id 직접선택 > criteria round-trip. ⇒ **selector offline 가치 음성**·가치는 *전이/multi-turn 날조방지*에만.

### ★딥리서치(`w2i00droj`) 검증·scoping
- **input-offload가 scale 대체 = task-narrow서만**(PAL 소형>PaLM-540B). **일반 "작은+offload≈큰"은 REFUTED(1-2)**·crossover 태스크별. ⇒ **우리 floor 음성(MSC≠scale대체 on exchange)은 문헌과 정합**(보편법칙 아님).
- **floor 측정 근접선행 = "Sufficient Context"(Joren et al·ICLR25)**: 충분-context autorater·**소형은 충분해도 환각(reasoning-limited)·대형은 불충분해도 abstain 안 함(info-limited)** = 우리 info/reasoning 분리와 동일. ⇒ floor *개념* 신규 아님·우리 신규=**typed-DAG closure + tool-use exchange + scale-trade**(deterministic scaffold·미출판 gap).
- **입력 formalize↑ = 확립**(PoT/PAL ~12% over CoT·표 serialization ±0.22·structural +0.8~5.7) **단 전부 *LLM-driven* pre-formalizer**(Visconde·DeepSieve·autorater) → **deterministic scaffold가 world-state pre-formalize는 미출판** = 우리 메서드 gap.

### caveat / 후속
- 32B=Int8 → **coworker 32B-bf16**이 (reasoning-floor vs Int8-cap)·(L2b +9pp 진위)·(14B→32B A 평탄 0.719=quant-cap?) 확정.
- model-내 L2a/L2b/L3 델타=노이즈(±6pp)·예외=32B L1→L2b +9pp.
- ⇒ 최적성 갱신: **분담은 *비용*서 이김(formalize=Pareto)·*capability(reasoning)*는 scale 필요**. floor가 조건#4를 reasoning-limited로 판정.

---

## 9. ★Sstep (강한형·scaffolded 증분 typed스텝+per-step 검증) 결과 (2026-06-16)
| arm | 7B | 14B | 32B-Int8 | tok/case | calls |
|---|---|---|---|---|---|
| A (forced) | 0.438 | 0.719 | 0.719 | ~918 | 1 |
| Acot | 0.531 | 0.781 | 0.719 | ~1400 | 1 |
| Atwo (2-call) | 0.656 | 0.719 | 0.781 | ~2900 | 2 |
| **Sstep** | **0.656** | **0.719** | **0.750** | **~528** | ~1.6 |

### 판정 (★capability 음성·비용 양성)
1. **Sstep는 capability 못 올림**: 7B 0.656(=Atwo)·자유CoT 최고 *안 넘고 동률*·32B-L2b 0.844 미달. **7B ~0.656 천장 = reasoning(binding) floor 지속.**
2. **★Sstep = 비용-Pareto 승리**: ~528 tok/case(Atwo 2900의 **1/6**·A의 절반)·calls 1.6. scaffold가 구조 결정론처리→LLM은 작은 typed스텝만. **best-elicitation 정확도를 최저비용에.**
3. **★리뷰 확증**: scaffold 검증(타입/vocab)은 *날조/형식*만 잡고 **grounded-but-wrong 변형선택(binding)은 못 잡음**(valid하지만 틀린 변형 emit). ⇒ capability 벽=binding·Sstep typed-검증으론 안 닫힘 → **M-σ v2 5번째 축(typed-derivation+resolver 관계계산)이 유일 장치**([[project-tau2-write-failure-rootcause]]·`M_SIGMA_DESIGN_2026_06_16.md §0b`).
- 대기: Snover(검증 OFF·Sstep≈Snover면 검증조차 무효 시사)·SCv(self-consistency)·coworker scale. `ma_overnight_summary.log`.
