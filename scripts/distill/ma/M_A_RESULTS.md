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

## 8. ★★FLOOR SWEEP 결과 (2026-06-16·decisive) — A·Bfair·L0–L3 × {7B,14B,32B-Int8,32B-bf16,72B-AWQ4} + 비용
> 출력=concrete 고정(입력 효과 분리)·`ma_eval_scale.sh _floor`·비용계측. ⚠29케이스(±~6pp 노이즈)·32B-Int8=GPTQ·L1≈A(±2 item=프롬프트문구).
> **coworker 추가 (2026-06-16, 단일 A100 80GB TP1, `autoresearch/ma-scale-260616/`): 32B-bf16 + 72B-AWQ-Int4** (72B-bf16은 145GB라 80GB 단일 GPU 불가→AWQ-Int4가 로컬 천장·노드bf16 별도). 사전검증 GATE1-5 통과(tau2 29케이스·resolver 4/4·7B 스모크 14/14 실레코드).

| arm (입력수준) | 7B | 14B | 32B-Int8 | **32B-bf16** | **72B-AWQ4** | tok/case |
|---|---|---|---|---|---|---|
| L0 (availability 없음) | 0.375 | 0.531 | 0.594 | 0.625 | 0.562 | ~855 |
| L1 (full+avail) | 0.531 | 0.688 | 0.750 | 0.719 | 0.688 | ~900 |
| L2a (가용필터·raw) | 0.406 | 0.656 | 0.812 | 0.812 | 0.781 | ~625 |
| **L2b (가용·표 formalized)** | 0.531 | 0.625 | **0.844** | **0.844** | 0.719 | **~508** |
| L3 (diff 주석) | 0.406 | 0.656 | 0.750 | 0.750 | 0.719 | ~628 |
| A (concrete baseline) | 0.438 | 0.719 | 0.719 | 0.719 | 0.688 | ~918 |
| Bfair (공정-정보 selector) | 0.375 | 0.500 | 0.656 | 0.656 | 0.594 | ~810 |

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
- ✅ **coworker 확정 (2026-06-16·32B-bf16 + 72B-AWQ4 로컬)**: (a) **14B→32B 평탄 = reasoning-FLOOR** — 32B-bf16 A=0.719=Int8=14B (양자화 무죄·Int8-cap 기각). (b) **L2b formalize 견고** — bf16 0.844=Int8 (최고·최저비용 유지). (c) **selector(Bfair) bf16서도 짐** (0.656<A 0.719). (d) ⚠️**Q2 천장 미확정(confounded)**: 72B-AWQ-Int4가 32B-bf16보다 전 arm 낮음(A 0.688·L2b 0.719=−0.125)은 **Int4 아티팩트 의심**(32B서 Int8≈bf16였으나 AWQ-Int4는 2배 공격적·formalize arm 최대 손상) → 평탄 시사하나 단정불가, **72B-bf16(H100x4 노드) 필요**. 상세=`autoresearch/ma-scale-260616/FINDINGS.md`.
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

---

## 10. ★M-σ in-dist 결과 (2026-06-16) — derivation-레벨 학습 *가능* 실증
M-σ = cfb-derivation SFT(copy-threaded args→typed `$ref`·등방화·val_loss 0.0101). in-dist($ref-emit) eval (cfb·`m_sigma_eval.py`·고친 harness):
| | name_ok | $ref-emitted | $ref-CORRECT-path |
|---|---|---|---|
| base 7B | 79/80 | 0/280 | 0/280 (0.00) |
| **M-σ** | 80/80 | 270/280 | **268/280 (0.96)** |
- **base 0% → M-σ 96%**: M-σ가 typed-derivation을 학습(어느 prior-output 필드 참조인지 정확). **v4-v7이 못한 *derivation-레벨* 학습이 가능함을 실증**(concrete-emit 아님).
- **★한계(과독 금지)**: (a) **in-dist(cfb train-set)·일반화 아님**(held-out cfb 미분리) (b) 학습 타입=**copy-threading**(cfb 유일)·**τ² selection-by-criteria 아님**(orphan) → threading-96%가 τ²-selection 전이 의미 안 함.
- **다음=M-D 전이**(held-out τ²·$ref-resolver·C8). selection-binding 데이터 v2 필요(딥리서치 `w3l415qh5`).

## 11. ★M-D 전이 결과 (C8 1차·2026-06-16) — 음성 (진단적)
M-σ(cfb-threading)를 held-out τ² exchange서 (`m_sigma_transfer_eval.py`·per-arg-type):
| arg | base | M-σ |
|---|---|---|
| used_$ref | 0/29 | 25/29 |
| order_id | 1.00 | 0.90 |
| item_ids | 0.93 | 0.79 |
| new_item_ids | 0.41 | 0.34 |
| payment_method_id | 1.00 | **0.07** |
| **all** | **0.41** | **0.03** |
- **형식 전이 ✓**($ref 25/29 emit·새 도메인) **but 내용 음성**: base가 맞히는 arg를 over-$ref로 망침(all 0.41→0.03).
- **원인 3**: (a) **over-$ref/provenance 미구분**(order_id=NL 리터럴인데 $ref 시도) — cfb는 threadable만 있어 "리터럴 vs derived" 못 가르침 (b) **payment 0.07 = harness 아티팩트**(gold pm_id를 dict 키로 둠→값-walk로 $ref불가·M-σ 불공정) (c) **new_item_ids 0.34 = selection orphan**(threading≠selection).
- **C8 1차 = 음성**(v4-v7 cfb-전이 패턴 정합). ⇒ **데이터 v2 3요건**: ①selection-by-criteria(DR `w3l415qh5`) ②**provenance 학습**(리터럴/키 vs $ref·리터럴-arg 섞기) ③harness 수정(payment=값·n 확장).

## 12. ★M-σ v4 factorial — ISO=ON half (coworker, 2026-06-16·node tb-h100-0616-factiso)
> primary = held-out τ² `$select`(new_item_ids 선택 정확률). base inference·NO τ² in training·NO openrouter(키 없음·로컬 vllm). split json = HF `factorial_iso/split_*.json`. **OFF-half(woori) 도착 시 ΔISO main effect 집계.**

| arm | ISO/NL/PROV | **$select** | all | item_ids | autopsy ok/no_avail/unresolved | emit |
|---|---|---|---|---|---|---|
| A-iso | 1/0/0 | **0.38** (11/29) | 0.38 | 0.79 | 18/9/2 | literal84·ref0 |
| C-in  | 1/1/0 | **0.41** (12/29) | 0.34 | 0.66 | 17/8/2 | literal78·ref0 |
| C-ip  | 1/0/1 | **0.41** (12/29) | 0.41 | 0.86 | 18/7/1 | ref79·over_$ref5 |
| FULL  | 1/1/1 | **0.38** (11/29) | 0.38 | 0.86 | 20/3/3 | ref81·over_$ref4 |

- **ON-half $select = 0.38–0.41 (mean ~0.40)·전 cell 평탄**(±1 case): ISO 위에 NL·PROV 추가해도 selection 정확률 불변 → NL/PROV main effect(ISO=on 하) ≈ 0.
- **잔여 실패 = M_A selection-reasoning 벽 그대로**: fail_no_available(가용성-blind 3–9)·unresolved(wrong criteria 1–3). **ISO 등방화 단독으로 이 벽 안 닫힘.**
- ⚠️ **ISO main effect 판정 보류**: ΔISO = mean(ON)−mean(OFF), OFF-half(M0/A-nl/A-prov/C-np)=woori 미도착. M0 대비 ON이 높으면 ISO 구동(Olver 실증)·M0도 ~0.40이면 ISO 무효(추상→실 갭). **§3 판독은 OFF-half 합류 후.**

## 15. ★★★7B N-sweep = thesis 직접 증거 (2026-06-17·통제 합성·`synth_depth.py`+`depth_eval.py`·raw `/home/woori/scratch/depth/depth_7B_N{5,10,20,50}.json`)
> 설계 = `../NL_PROCEDURE_OFFLOAD_THEORY_2026_06_17.md`(이론)·`../B_BUDGET_SCALE_DESIGN_2026_06_17.md`(실험). **arm A**(catalog+in-head 단독)·**arm B**(NL→연산-IR `{op,attr,among}` *명명*→결정론 엔진 `resolve_operation` 실행)·**arm D**(oracle). 각 N=op 5종(filter·argmax·argmin·rank·comparative)×50=250케이스. base Qwen2.5-7B·무재학습. **raw 전수 검증 완료(아래 분수=원본 그대로).**

| N (리스트 길이) | A in-head | **B 구조(IR+엔진)** | D oracle | 격차 B−A | rank A→B | argmax A→B | recognition(op-명명) |
|---|---|---|---|---|---|---|---|
| 5  | 0.68 (170/250) | **0.80 (200/250)** | 1.00 | **+0.12** | 0.42→1.00 | 0.82→1.00 | 0.60 (151/250) |
| 10 | 0.56 (141/250) | **0.80 (199/250)** | 1.00 | **+0.24** | 0.30→0.98 | 0.74→1.00 | 0.60 (150/250) |
| 20 | 0.50 (125/250) | **0.80 (199/250)** | 1.00 | **+0.30** | 0.12→1.00 | 0.80→1.00 | 0.60 (150/250) |
| 50 | **0.34 (86/250)** | **0.80 (200/250)** | 1.00 | **+0.46** | **0.10→1.00** | 0.40→1.00 | 0.60 (150/250) |

- **★헤드라인: in-head A는 N↑서 무너짐(0.68→0.34)·구조 B는 *N-불변 0.80*·격차 +0.12→+0.46(N로 단조 *벌어짐*).** = 유계 절차예산 B(L,width) 초과를 결정론 엔진(B=∞)이 흡수 → **문제 클수록 offload가 더 지배** = thesis 핵심 직접 증거.
- **rank(가장 깊은 절차·d 최대): N=50서 in-head 5/50=0.10(≈random) → 구조 1.00.** argmax도 N=50서 0.40→1.00. argmin N=50 0.28→1.00. = 깊은 절차일수록 in-head 붕괴가 가파르고 엔진 흡수가 완전.
- **★recognition(op-명명)도 N-불변 0.60**(150/250 전 N 고정): LLM의 본업(절차-타입 *분류/명명*)은 N에 안 무너짐. N로 무너지는 건 *실행*뿐(offload 대상). = "얕은 명명 LLM / 깊은 실행 결정론" 분담선의 직접 실측.
- **(v1 시점) 유일 잔여 = comparative 전 N서 B = 0/50 = 0.00**(A도 18/7/8/2로 붕괴). → **아래 v2서 진단·해소.**
- **함의(B-budget 스케일로 이어짐)**: 7B+구조가 N-불변인데 in-head는 대형모델이라야 무릎이 우측이동(14B d3 0.50>7B 0.20). arm C(32/72/235B in-head 매핑 임계 S\*(d,N))로 "거대모델조차 N↑서 매핑비용 지불 vs 7B+엔진 무비용"을 박을 것(coworker, 진행중).

### 15-bis. ★comparative 진단·수정 = v2 (2026-06-17·`comparative_diag.py`·`comparative_fix.py`·raw `depth_7Bv2_N*.json`)
**진단(전수)**: comparative B=0.00은 "절차의미 명명 불가"가 *아님*. 7B는 **피연산자를 완벽 추출**(`attr·among·dir·anchor_id` — 긴 랜덤 id까지 verbatim) — **틀린 건 연산자 *라벨* 하나**(op="comparative" 대신 "filter"·N=10/50 둘 다 50/50). 원인=NL 도입부 "Among items where {filter}…"가 `filter` 토큰 priming + IR spec에 연산자-*어휘 정의*(gloss) 부재 → default 라벨 붕괴.
**수정(A/B)**: spec에 op-gloss(생성원 어휘 정의) 추가 → comparative **0.00→1.00 즉시·N-불변**(`comparative_fix` B1). NL 재배열(B2)은 불요. = 병목은 *어휘/라벨링*(깊이 아님)·gloss=형식 IR의 정당한 어휘정의.

**v2 전체 N-sweep (gloss spec·per-op recognition·`depth_eval --gloss 1`)**:
| N | A in-head | **B 구조** | recognition | 격차 B−A | comparative B | filter/argmax/argmin/rank B |
|---|---|---|---|---|---|---|
| 5  | 0.68 | **0.99 (248/250)** | **1.00 (250/250)** | **+0.31** | 0.96 | 1.00/1.00/1.00/1.00 |
| 10 | 0.56 | **1.00 (250/250)** | **1.00** | **+0.44** | 1.00 | 1.00/1.00/1.00/0.98 |
| 20 | 0.50 | **0.99 (248/250)** | **1.00** | **+0.49** | 0.98 | 1.00/1.00/0.98/1.00 |
| 50 | **0.34** | **1.00 (250/250)** | **1.00** | **+0.66** | 1.00 | 1.00/1.00/1.00/1.00 |

- **comparative 0.00→0.96–1.00 전 N·다른 4 op 무회귀**(0.98–1.00) = gloss가 안 깨뜨림.
- **recognition 0.60→1.00**: (i) filter artifact 해소(gold filter엔 attr 없어 v1이 attr-mismatch로 오집계 → op-only 채점으로 교정) (ii) comparative 회복. **이제 per-op recognition 전부 1.00** = 𝔤-식별(생성원 명명)이 *어휘만 정의되면* 소형서 완전·N-불변.
- **격차 B−A = +0.31→+0.66**(N로 단조·v1 +0.12→+0.46보다 큼): comparative 회복으로 B가 ~1.00 → in-head 붕괴(0.68→0.34) 대비 구조 우위 *더 선명*. = §6 예측 (a)(b) 강화 실증.
- **이론 함의(§9-A 교정·§7f 다리)**: "절차의미 sub-personal=명명불가"는 *과장*이었음. 명명은 *어휘 정의*로 즉발(보간 아님의 방증). 단 gloss는 *in-context* 떠먹이기 = C8 시험의 *상한*. **C8 = 이 라우팅을 weight 내재화해 gloss 없는 held-out 도메인서도 되나**(진행중·`C8_PROCEDURE_ROUTING_TRANSFER_DESIGN`).

## 16. ★★★C8 1차 = 절차-라우팅 weight 내재화 → held-out 어휘 전이 = 양성 (2026-06-17·`c8_batch.sh`·`c8_summary.py`·raw `/home/woori/scratch/depth/c8/results/`)
> 설계 = `../C8_PROCEDURE_ROUTING_TRANSFER_DESIGN_2026_06_17.md`. **시험**: NL→op-IR 라우팅을 *gloss 없이* SFT(학습 prompt에 연산어휘 정의 제거) → 학습에 안 나온 *다른 seed*(새 attr 토큰·스키마=전이 도메인) held-out서 gloss 없이 라우팅되나. base Qwen2.5-7B·LoRA·합성 등방화.

**floor S0(base·gloss off)=comparative 0.00 / ceiling S1(base·gloss on)=1.00. 7 변형 전수 held-out comparative recognition:**
| 변형 | 학습 | held-out cmp recog | in-dist | 판정 |
|---|---|---|---|---|
| A1_ep1 | ep1 | **1.00** | 1.00 | TRANSFER |
| A2_ep3 | ep3 | **1.00** | 1.00 | TRANSFER |
| A3_ep6 | ep6 | **1.00** | 1.00 | TRANSFER |
| B1_cmpheavy | cmp 3x·ep3 | **1.00** | 1.00 | TRANSFER |
| B2_glossin | gloss-IN·ep3 | **0.98** | 1.00 | TRANSFER |
| B3_big | 8k·ep2 | **1.00** | 1.00 | TRANSFER |
| B4_lr3 | lr3e-4·ep3 | **1.00** | 1.00 | TRANSFER |

- **★전 변형 양성·강건**: 학습량(ep1–6)·LR·데이터량(3k/8k)·comparative집중·gloss유무 무관하게 held-out comparative 0.00→~1.00. **1 epoch이면 충분**(저차원 라우팅·§7e 예측 "𝔤-식별은 소형도 즉발·전이" 정합). in-dist=held-out=1.00 → 과적합 아닌 어휘-무관 일반화. 나머지 4 op도 전 변형 1.00.
- **= 사용자 질문 1차 답**: 절차어휘(op 라우팅)를 *gloss 없이* weight에 내재화(TBox 고정) → 새 도메인 어휘 전이 = **"절차 TBox 고정 / 도메인 ABox swap" 합성서 입증.** base가 gloss 없이 comparative 0.00인데 학습 후 1.00 = 학습이 라우팅을 분명히 내재화(floor-ceiling 대비 명확).
- ⚠️ **정직 한계(과대해석 금지·[[feedback-no-fundamental-claims-from-convenience-data]])**: (1) **합성 1차**·held-out은 새 *attr 어휘·스키마*지만 **NL 템플릿은 동일**(synth_depth op별 고정 문구) → **어휘-전이 입증·표현-전이 아님**. (2) 진짜 표현/도메인 전이 = τ²(2차·`C8_TAU2_SELECTION_TRANSFER_DESIGN`). (3) 1ep 100%는 라우팅이 *쉬운 저차원*이란 방증이자 동시에 시험이 *쉬웠단* 뜻 — τ²가 난이도 시험.
- **⇒ τ² 2차 게이트 통과**: "합성서 안 되면 τ²는 더 안 됨"의 전제가 깨짐 → τ² 어댑터(M-A static-select를 op-IR로 재무장) 구현이 의미를 얻음. 다음 = `C8_TAU2_SELECTION_TRANSFER_DESIGN` Phase 1(offline·키 불필요).

## 17. ★★C8 2차 = τ² selection 표현-전이 = 음성 (진단적·2026-06-17·`tau2_op_eval.py`·offline 키0·32 items)
> 설계 = `../C8_TAU2_SELECTION_TRANSFER_DESIGN_2026_06_17.md`. resolver sanity = oracle filter(gold_opts) **32/32(1.00)**(엔진·catalog 정합). arm = S2(C8-trained A2_ep3·gloss0)·S1(base·gloss1)·S0(base·gloss0).

| arm | overall new_item_id | substitution(13) | superlative(19) | emitted op 분포 |
|---|---|---|---|---|
| **S2 C8-trained** | **1/32 (0.03)** | 1/13 | 0/19 | comparative12·**exchange11·replace1·update1**·argmax4·argmin3 |
| S1 base+gloss | 9/32 (0.28) | 4/13 | 5/19 | comparative29·filter2·argmax1 |
| S0 base | 8/32 (0.25) | 3/13 | 5/19 | filter17·argmax6·comparative6·argmin2 |
| (참고) M-A static select_by | 15/32 (0.469·§1) | — | — | — |

- **★C8-trained = 역전이**(0.03 ≪ floor 0.25): 합성 SFT가 τ²서 *깨짐*. **결정 단서 = emitted op에 `exchange/replace/update`**(τ² NL 동사를 op 슬롯에 복사) → 합성 학습이 *절차 추상*이 아니라 **"NL 동사→op" 얕은 표면 매핑**을 내재화했을 가능성. 합성선 "select the highest"→argmax 맞았지만 τ² "exchange…for a bigger"→op=exchange로 붕괴.
- **= §16 정직 한계의 실증·폭로**: 합성 1차 양성(0.00→1.00)은 *NL 템플릿 고정* = **어휘 전이**였고, τ²의 *자유 표현*엔 안 옮음(**표현 전이 음성**). "절차어휘 TBox 고정"은 *같은 표현 분포* 내에서만 성립(이번 증거).
- **op-IR 재무장 자체도 τ²서 무이득**: base op-IR(S0 0.25)도 M-A static select_by(0.469)보다 *낮음*. τ² selection(substitution 다수·multi-categorical)에 단일-ordinal 합성 op-IR 형식이 부적합(설계서 §3 리스크 실현).
- ⚠️ **진단 범위(과대해석 금지·[[feedback-no-fundamental-claims-from-convenience-data]]·[[feedback-capability-vs-artifact-elicitation]])**: n=32·retail single-domain·**프롬프트 형식 confound 미배제**(τ² prompt가 합성과 형식 달라 C8-trained OOD 붕괴일 수 있음 — "표면 매핑" 단정 전 형식-통제 필요). ep3 1개 어댑터만(ep1 robustness 미확인).
- **⇒ 함의**: C8이 실벤치 전이로 닫히려면 (a) **합성 NL 표현 다양화** 재학습(템플릿 고정 제거→표현 전이 시험) 또는 (b) 합성↔실벤치 표현 갭 중간층. 현 단일-템플릿 합성 양성은 *필요조건이지 충분조건 아님*. = thesis에 중요한 음성(C8 핵심 미해결 재확인).

### 17b. ★형식 통제 진단 = 표면 매핑 *확정* (SFT 주입·2026-06-17·`tau2_op_eval --synth_format`)
프롬프트 구조를 합성 arm_B와 동일하게 맞춰 형식 confound 배제:
| arm | overall | emitted op |
|---|---|---|
| A2_ep3 synth-fmt | 0.09 | filter11·**exchange8·modify2**·comparative4·argmax4·argmin3 |
| A1_ep1 synth-fmt | 0.09 | comparative11·**exchange6·modify2** |
| A1_ep1 τ²-fmt | 0.00 | comparative17·exchange10 |
| **base synth-fmt** | **0.16** | filter20·rank6·comparative3·**exchange 0** |
- **op-슬롯 붕괴 = NL 동사 복사 확정**(형식 아님): 합성 형식서도 exchange8 잔존 → τ² *동사*가 원인.
- **★SFT가 표면 매핑을 *주입***: **base는 exchange 0회**(filter/rank로 감)·**trained만 exchange emit**. 게다가 base(0.16) > trained(0.09) = **좁은 단일-템플릿 SFT가 "NL동사→op" 표면매핑을 심어 역전이**.
- ⇒ **결론 = §16 합성 양성은 표면 매핑 아티팩트**(좁은 표현 분포). **처방 = 표현/구조 다양성**(어휘 등방화 불충분·깊이 비례). 다양성-전이 곡선 K×depth 설계 예정.

## 18. ★표현 다양화 → τ² 재전이 = 부분양성 (표면붕괴 끊김·정확도 미회복·2026-06-17·`render_nl_diverse`·DIV_ep3)
> 다양화 = 동사 풀 op-무관·op은 의미표현으로만 결정(`synth_depth.render_nl_diverse`)·표면-op 상관 0. 3000샘플 ep1 LoRA → τ² offline + held-out diverse.

| | 표면붕괴(op=exchange/replace 복사) | τ² 정확도 | in-dist diverse |
|---|---|---|---|
| 단일템플릿 S2 | **13/32** | 1/32 (0.03) | — |
| **다양화 DIV** | **3/32** | 2/32 (0.06) | **250/250 (1.00)** |

- **★표면붕괴 13→3**: 다양화가 "NL동사→op 복사"를 끊음(exchange 11→3·comparative/argmax 정상 emit). = 사용자 명제(표면-op 상관 끊기) **표면붕괴 차원 확증** = **Cor 6.9.6 off-manifold→on-manifold 복귀** 신호.
- **★정확도 미회복(2/32·floor 0.25 이하)**: in-dist diverse=1.00인데 τ²=0.06. on-manifold 복귀해도 *올바른 basin* 못 찾음 — over-comparative(23/32·substitution도 comparative)·인자(attr/among/anchor) 오류·합성 표현풀이 τ² 표현/구조 미커버.
- ⇒ **표면 다양성 = 필요조건 *한 겹*·충분 아님**. 정확도 전이엔 (a) 구조 다양성(과제형태·multi-attr) (b) D 증대(K-sweep) (c) 인자 reasoning 필요. = `EXPRESSION_DIVERSITY_TRANSFER_DESIGN` K-sweep 곡선으로 표면붕괴율 vs D / 정확도 vs D *분리* 추적.
- 정직: n=32·합성 표현풀 한정·over-comparative는 다양화 데이터 op-편향 가능성(별도 진단).

## 19. ★★★생성원 적합성 = substitute op-IR가 τ² exchange 전건 재현 (오라클·2026-06-17 PM·`tau2_subst_oracle.py`·offline 키0·GPU0·n=32)
> §17/§18 정확도 미회복의 **근본원인 확정 후 닫기**: over-comparative(23/32)는 학습 편향이 아니라 **5-op 어휘가 τ² substitution을 *표현 불가***였기 때문(`GENERATOR_ALGEBRA §3`·HANDOFF_PM §1: 모델이 exchange→comparative 강제 + `to:` 발명). 처방 = content 생성원 5→7(**substitute**·create 추가). substitute op-IR = `{op:substitute, anchor_id:old_item_id, set:{변경된 옵션}}`·엔진 = anchor 옵션 ⊕ set override로 target 구성 후 유일 variant 매칭(`synth_depth.resolve_operation`·`tau2_op_resolver.resolve_op_tau2`).

| 검증 | 결과 |
|---|---|
| **substitute 오라클** (gold IR → τ² retail exchange) | **32/32 (1.000)** new_item_id 재현 |
| synth round-trip (전 7-op·N∈{5,10,20}·각 60) | **420/420 (1.000)** |

- **★표현 적합성 입증**: gold substitute IR이 τ² retail exchange 32건 전부를 결정론적으로 재현. = 생성원 대수 gap(2nd-gate 음성 근본)이 **표현 차원서 닫힘** — 학습 무관·thesis offload 구조 그대로.
- **keep-rest가 진짜 난점 확증**(HANDOFF §2.3 gotcha): 변경/전체 옵션 분포 = keep-rest 진성 substitute **25건**(일부 유지: (1,2)·(1,3)·(2,3)·(2,4)·(3,5)·(4,5))·full-change(=create-shaped) 7건((3,3)·(4,4)). substitute 엔진이 두 경우 모두 처리(target=old⊕set).
- **함의**: §17/§18 음성 = *표현 부재*(𝔤 불완전)였지 다양성(𝔥) 부족이 주인이 아님 → 생성원 완전성이 선행(HANDOFF §1.2 진단 정량 확증).
- **다음(학습-전이)**: 이 오라클은 *표현* 적합성만(IR을 손으로 줌). 미해결 = **모델이 τ² NL서 substitute를 *명명*하는가**(C8-route 학습 후). = §20 다도메인 동시 전이 매트릭스(retail+airline·`tau2_op_eval` 확장·GPU 학습 필요).
- 정직: n=32·retail exchange 한정·오라클(IR 수동)=학습 전이 아님·airline 미포함(추출 진행 중).

**★교차도메인 추가 (2026-06-17 PM·`ma_gold_extract --domain airline`·`tau2_subst_oracle`):** airline cabin 차원(basic_economy<economy<business ordinal) 추출 = update_reservation_flights(cabin 변경·flights 유지=keep-rest substitute) **17** + book_reservation(create) **10**.

| 도메인 | 오라클 | op-IR |
|---|---|---|
| retail exchange | **32/32 (1.000)** | substitute(keep-rest, 25진성+7 full) |
| **airline cabin** | **27/27 (1.000)** | substitute 17 + create 10 |

- **★교차도메인 twin 입증(표현)**: *동일* op-IR 어휘·*동일* resolver(`resolve_op_tau2`·cabin을 ORD_WORDS 등록)로 retail variant-exchange와 airline cabin-update 둘 다 닫힘 = substitute가 도메인-일반(공격 "retail 특화" 표현차원 반박). HANDOFF §2 가설(substitute가 두 도메인 지배)의 표현측 확증.
- 정직: airline cabin=단일 ordinal attr → catalog 레벨 keep-rest 구조 degenerate(유지되는 "rest"=flights는 cabin-catalog 밖) → airline 케이스는 multi-attr keep-rest보다 **op-라우팅(substitute vs comparative vs create)**을 시험. retail이 multi-attr keep-rest 담당. 학습-전이(모델이 NL서 op 명명)=§20 미해결(GPU 학습 필요).

## 20. ★★★인자(`set`) 실패 궤전수조사 = 원인 2성분 확정 (2026-06-17 PM·base 7B·`tau2_arg_autopsy.py`+`tau2_reresolve.py`·offline 키0·GPU0·retail n=32/airline n=27)
> base 7B(학습0)는 op-라우팅 **recognition 높음**(retail g1 0.94·airline g1 0.93)인데 new_item_id **정확도 낮음**(0.19~0.44). = op은 맞히고 *인자*가 틀림. 전 miss를 결정론 분류(emitted `set`/`anchor_id` vs gold)해 원인 확정. **두 성분이 도메인별로 분리**:

**성분 A — anchor_id 환각 (우리 설계 결함·수정 완료·airline 지배):** op-IR이 LLM에게 `anchor_id`(수정대상 item) emit을 강제 → 모델이 **그럴듯한 id 환각**(airline=예약코드 "M05KNL"·task18 5회 전부 다른 환각·retail=카탈로그 부재 10자리). resolver가 `aid=op_ir.anchor_id or 문맥anchor`로 **환각을 grounded 문맥보다 우선** → anchor 못찾음/오인 → None. **수정 = anchor grounding**(`aid=문맥anchor or …`·concrete id=offload, LLM 결정 아님·[[feedback-nl-formalize-llm-selection-deterministic]]).

| | acc 원본 | acc grounded | 회복 |
|---|---|---|---|
| airline g0 | 0.19 | **0.37** | +5 |
| airline g1 | 0.44 | **0.78** | +9 |
| retail g0/g1 | 0.28 / 0.44 | 0.28 / 0.44 | **0** |

**성분 B — `set` 과소추출 (genuine LLM 과제·retail 지배·anchor-무관):** retail miss 지배 = **missing_key**(요청된 변경 attr 중 *일부만* 추출·g0 14/22·g1 11/17). 예: gold_set={color:silver,brightness:low,power:battery}, emit={brightness:low,power:battery}(color 누락). anchor grounding이 retail서 0 회복 = **set 자체가 불완전**(target=old⊕부분set→누락 attr는 old값 유지→오매칭). 부차: **wrong_value**(카탈로그 enum 미정규화: "Google Assistant"→"Google Home")·**no_set**(op만 명명·set 공란). = *어떤 attr를 무엇으로 바꾸고 나머지 유지*의 multi-attr delta 추출 = HANDOFF §2.3가 예고한 "keep-rest reasoning" 진짜 난점 = **formalize 단계**(offload 불가).

- **★확정**: airline 갭 ≈ 전부 성분 A(설계결함·결정론 수정으로 닫힘·thesis 정합). retail 갭 ≈ 전부 성분 B(genuine·LLM이 풀어야). 정확도-낮음≠라우팅 실패·≠모델 무능(recognition 높음)·**인자 추출/grounding 문제**.
- **함의 (다음)**: (1) **anchor grounding 적용**(완료) → airline base+gloss 0.78. §20 학습본도 재해결(`tau2_reresolve`) 필요(multidomain_route는 수정 전 pull). (2) **synth substitute가 multi-attr 과소커버**(n_change=1~2 vs retail 1~4)→3~4 attr 변경 합성 추가해야 성분 B 학습. (3) wrong_value→**값 스냅 offload**(모델은 attr+의도 명명·엔진이 카탈로그 enum 최근접 스냅) 여지. (4) coworker 스케일 질문 초점 = "스케일이 multi-attr delta 추출(성분 B)을 올리나"로 좁혀짐.
- 정직: base 7B 한정(학습본 §21 예정)·retail n=32/airline n=27·anchor grounding은 grounded 문맥 전제(τ² 실제 fetch와 정합).

## 21. ★★★§20 다도메인 라우팅 전이 매트릭스 = 도메인-일반 양성·천장은 성분 B (2026-06-18·`multidomain_route.sh`·7B 라우팅 LoRA·synth-only 학습·ep1·6000·gloss-free)
> synth(7-op 등방화 라우팅)만 학습 → retail+airline config-swap 전이(재학습0). 학습본은 grounded-anchor 재해결로 0 회복(=학습이 anchor 환각도 교정).

| | base g0 | base g0 grounded | **학습본 g0** | base g1 (gloss 천장) |
|---|---|---|---|---|
| retail | 0.28 | 0.28 | **0.44** | 0.44 |
| airline | 0.19 | 0.37 | **0.44** | 0.44 |
| synth held-out (gloss-free) | — | — | **1.00 (250/250)** | — |

- **★라우팅 내재화**: synth held-out 1.00(새 어휘/스키마·gloss-free) = NL→op 라우팅이 weight에 완전 내재화·전이.
- **★도메인-일반 확증(§20 헤드라인)**: *동일* synth-학습 라우팅이 retail·airline **동시** 상승(둘 다 0.28/0.37→0.44) = "도메인-일반 생성원" 학습-입증(한쪽만 아님). 학습본 recognition retail 27/32·airline 24/27.
- **★천장 ~0.44 = 성분 B(§20)**: 학습본 잔여 miss = missing_key(retail 12)·wrong_value(airline 6) = multi-attr `set` 과소추출. width-1~2 synth가 retail/airline의 multi-attr·값정규화를 과소커버해 못 메움. → width 실험(§22)·wider-synth·decomposition-offload 동기.
- **함의**: 전이의 *라우팅* 축은 닫힘(내재화+도메인일반). 남은 벽 = *인자(set) formalize* = width-budget. 이게 scale로 풀리나 offload(분해)가 필연이냐 = §22 width×scale(frontier 포함).
- 정직: ep1·6000·7B 한정·multidomain_route는 anchor-fix 전 resolver로 평가(재해결로 동일 0.44 확인)·n 작음(32/27).

## 22. ★★★성분 B(width)는 스케일로 해소 = offload 필요성은 *소형 모델 조건부* (2026-06-18·`width_eval.py`·통제 width substitute·n=60/width)
> 질문: multi-attr `set` 추출 벽이 *근본*(전 스케일 지속→decomposition-offload 필연)이냐, *소형 한정*(frontier가 흡수→offload 불요)이냐. frontier gpt-4.1(openrouter)로 width 1~5 측정.

| width | 1 | 2 | 3 | 4 | 5 | 패턴 |
|---|---|---|---|---|---|---|
| **gpt-4.1 SET_EXACT** (요청 k개 전부 추출) | 0.88 | 0.93 | 0.80 | 0.82 | 0.95 | **평탄·벽 없음** |
| gpt-4.1 arm A (in-head 전부) | 1.00 | 0.93 | 0.85 | 0.90 | 0.97 | 평탄 |
| 7B (τ² base, §20) | 0.64 | 0.40 | 0.29 | 0.25 | — | **급락** |

- **★확정**: frontier는 width 1→5서 SET_EXACT ~0.8~0.95 **평탄**(벽 없음)·7B는 급락. ⇒ **width 벽 = 소형 모델 현상·스케일로 해소**. (gpt-4.1 width3~4서 size_bias −0.57/−0.73 = 경미한 under-spec 잔존이나 붕괴 아님.)
- **★offload 필요성 = 주권(소형 on-prem) 조건부**: frontier 있으면 width offload(분해) *불요*(native 처리). **소형 on-prem 7B(=thesis 타깃)엔 벽이 실재→decomposition-offload가 7B를 frontier 수준으로 끌어올리는 구조**. = thesis "소형+구조=대형" **정합·강화**(frontier가 native로 하는 걸 7B는 분해로 달성). depth 축(엔진 offload)과 동형 결론.
- **다음**: (1) 7B를 *동일* synth width_eval로(τ²→synth substrate 통일·redo 후 GPU). (2) coworker 32/72/235B width 스윕(`width_scale_batch.sh`)→ S\*(width) 임계 스케일 위치. (3) **decomposition arm**(per-attr emit+엔진 조립) 구현→소형서 width 벽 우회 *충분성* 실증.
- 정직: gpt-4.1 단일 frontier(추가 모델 권장)·n=60·synth 한정·7B 비교점은 아직 τ² 기반(동일-substrate 보강 중)·SET_EXACT<1.0(frontier도 완벽 아님).

## 23. ★★★밤샘 종합 (2026-06-18·width×scale·decomp·§17 closure·wide-train·op-IR e2e) — 권위본
> `morning_readout.sh` 집계. raw=`/home/woori/scratch/depth/c8/width/*.json`·`…/multidomain/results/*`·`…/tau2-bench/data/simulations/*`.

**A. §17 closure = airtight (동일 케이스·resolver·base/5op/7op 3-way):**
| 모델 | retail acc | op_dist |
|---|---|---|
| base 7B | 0.28 | substitute 30 |
| 5-op-C8 A2_ep3 (narrow) | **0.12** | substitute 30 (값틀림) |
| 5-op-C8 DIV_ep3 (diverse) | **0.03** | cancel/argmax/comparative (substitute 못 emit) |
| **7-op-C8 MD_route** | **0.44** | substitute 27 |
→ 5-op은 base 미만(역전이 §17 재현)·7-op은 초과. **누락 생성원(substitute)이 §17 음성 원인 박제.**

**B. width×scale S\*(width) (synth `width_eval`·SET_EXACT one-shot):**
| 모델 | w1 | w3 | w5 | 패턴 |
|---|---|---|---|---|
| 1.5B | 0.00 | 0.00 | 0.00 | op-IR 포맷 붕괴(0.5B serve실패) |
| 3B/7B/14B | 0.74/0.66/0.88 | 0.71/0.52/0.62 | 0.66/0.56/0.77 | **synth선 완만**(τ² 0.64→0.25보다 약함) |
| gpt-4.1 | 0.88 | 0.80 | 0.95 | frontier 평탄 |
- **★decomposition-offload가 7B 회복**: one-shot w4 0.51 → **decomp 0.87**(Qwen 3/7/14B 회복·llama8b 무효=모델별). = offload 충분성 부분입증.
- **중요**: synth 벽 완만·τ² 벽 급함 ⇒ **τ² 성분 B 잔여는 width/arity 아니라 실-카탈로그 값-grounding/정규화**(아래 D).

**C. 다양성 (clean 7-op K1–K32):** surface-collapse→**K4서 0**·held-out recognition→**1.00**·kcenter 효율(무릎 K4)·τ² acc 캡(성분 B). 층 분리 확증.

**D. ★wide-substitute 학습 τ² 전이 실패 = *라우팅 퇴행*(궤적 전수·`tau2_arg_autopsy`):** synth SET_EXACT **1.00**(완벽) but τ² retail 0.44→**0.47**·airline 0.44→**0.30 퇴행**. 원인=set 아니라 **op-라우팅 깨짐**: retail miss 지배 **op_mismatch 9**(substitute→comparative/argmax/argmin)·recognition 27→20↓·airline **mixed_keys 10**(create를 cabin아닌 origin/dest/date로 emit)+op_mismatch 8(substitute→argmin). = **한 축(set-binding) 고치니 다른 축(routing) 손상**(skewed SFT 전이손상). 잔여 recognition-correct miss=wrong_value(값정규화)·missing_key(width4 색누락).

**E. ★op-IR 어댑터 native agent 불가 = *출력 포맷 비호환*(궤적 전수):** MD_route/widesubst를 native τ² agent로 → pass^1 **0.075/0.077**(<base 0.17). 궤적=모델이 함수콜 JSON을 **hermes tool_call 아닌 텍스트 content로** 출력→파서 미인식→도구 실행0→인자 날조(order_id `#W0000000`·이름 "John Doe")→**no_auth 31·agent_collapse**. = "Output ONLY JSON" op-IR SFT가 native tool-call 프로토콜 덮어씀. ⇒ offload 통합 시 **write-tool resolver만 끼워야**(포맷 전체 교체 금지)·agent는 native tool_call 유지.

**함의:** 라우팅(§21 내재화+도메인일반)·표현(§22 다양성 D\*)은 닫힘. 남은 τ² 벽 = **(i) 실-카탈로그 값-grounding(wrong_value 정규화·offload로 스냅)** + **(ii) write-step offload 통합**(native tool_call 유지한 채 변형 의도→resolver). wide-train·op-IR-native 둘 다 폐기(전이손상·포맷붕괴). base e2e GBW headroom = 측정 중(§24 예정).

## 24. ★★★base τ² e2e GBW headroom = GBW는 *소수*(5%)·지배는 FLOW collapse(60%) (2026-06-18·`tau2_autopsy`·40 tasks·gated pass^1)
> 질문: "GBW(변형 오선택)만 고치면 e2e 얼마 오르나?" → base 7B 풀 rollout 실패모드 전수분류로 정량.

| 실패모드 | 수/40 | 축 |
|---|---|---|
| PASS | 7 (**0.175**≈문서 0.17 재현) | — |
| **agent_collapse** | **24 (60%)** | **FLOW**(루프·max_steps·too_many_errors) |
| wrong_write (=GBW) | **2 (5%)** | CONTENT(변형 오선택) |
| premature_refuse | 2 | FLOW(조기거부) |
| over_ask | 1 | FLOW(묻기만) |
| other | 4 | — |

- **★GBW headroom = 2/40 (5%)**: GBW만 고치면 0.175→~0.225(+2). **작다.** auth provenance **grounded 39/39·날조 0**(base는 인증 grounding 정상·날조는 op-IR 어댑터 §23E 현상).
- **★e2e 지배 벽 = FLOW collapse 60%**: base 7B는 변형 선택 *이전에* 멀티턴 시퀀스서 루프/에러로 붕괴(P1–P9 orchestration·에러복구 실패). content/width offline diagnostic이 닫는 건 **e2e의 5% 슬라이스**.
- **함의(정직·중요)**: 우리 content축(substitute/width §19–23)은 **grounded-but-wrong write 결정**을 닫지만, **base 7B e2e의 본체는 FLOW축(agent_collapse)**. ⇒ e2e 큰 레버 = FLOW 생성원(P1–P9·시퀀싱·복구·non-loop) 전이. GBW가 지배가 되려면 FLOW 먼저 닫혀야(anti-fab v8/v9선 GBW 노출됐던 것과 정합: 모델별 GBW몫 다름).
- 정직: 40 tasks·gated without-L2·base 7B 단일·agent_collapse 내부 세분(어느 도구단계 루프) 미규명(후속).

## 25. ★★★§24 정정 — agent_collapse 본체 = fetchable 구체값 *날조*(offload 대상)이지 generic FLOW 아님 (2026-06-18·`tau2_collapse_autopsy`·전수 궤적·base 7B·27 collapsed)
> 사용자 교정: collapse 원인 = 파라미터 오예측. 전수확인 결과 정확. §24의 "FLOW축"은 **잘못된 일반화**였고, 본체는 **구체값 날조/오포맷 = offload-addressable**.

**collapsed 27/40·에러난 도구콜 파라미터 전수분류:**
| 오예측 파라미터 | 건수 | 성격 |
|---|---|---|
| **order_id 날조** | **79** | placeholder 발명(`#W0000001`·`#W000000123`) |
| order_id 포맷(# 누락) | 12 | `W2378156`·`100238`(user값 오포맷) |
| item_id 날조 | 15 | fetch 안 하고 발명(`1008292230`) |
| user_id/payment/기타 | ~30 | |
- 도구: `get_order_details` 84 에러 지배. **retry-same-args loop 16/27**(동일 틀린 콜 반복→too_many_errors·복구 0).
- **★본체 = tool-fetchable 구체값 날조**: order_id·item_id·payment를 *get_order/get_user에서 fetch*해야 하는데 **모델이 발명**→not found→무한재시도→collapse. base auth는 grounded(39/39·§24)나 **다운스트림 fetch-id는 날조**(다른 grounding 차원).
- **★§24 재구성**: e2e 벽 = (i) **GBW 변형선택**(new_item_ids·5%·content) + (ii) **fetch-id 날조**(order_id/item_id·collapse 구동·훨씬 큼) + (iii) **retry-same 복구실패**. (i)(ii) 둘 다 **offload 핵심**(=구체값 LLM 발명 금지·결정론 fetch/resolve). ⇒ **offload thesis가 e2e 벽의 대부분을 덮음**(5% 아님)·잔여 = 복구(iii).
- **선행 정합**: 옛 v4 write-벽 autopsy "tool-fetchable 값(order_id) 날조"와 동일 기제([[project-tau2-write-failure-rootcause]]·메모리 핸드오프) — base서 정량확인(order_id 날조 91). [[feedback-nl-formalize-llm-selection-deterministic]] 정확 정합.
- **함의(처방)**: write-step뿐 아니라 **read-step도 offload** = 모델은 "주문 조회/교환" *의도*만 명명, order_id/item_id/payment는 **결정론 resolver가 직전 tool 결과서 grounding**(날조 차단)+에러시 재-fetch(복구). = thesis "구체선택=offload"의 e2e 실증 경로.
- 정직: base 7B·40 tasks·gated·order_id 날조 vs 포맷 비율(79:12)은 user가 번호 제공한 태스크 비중에 의존·retry-loop은 복구프롬프트로 일부 완화가능(별도).

## 26. ★★★수렴 진단 — peeling이 *단일 근원(concrete-value offload)*으로 수렴 (2026-06-18·전 §의 통합)
> 사용자 관찰: "하나 풀면 다음이 나타나 결국 GBW까지 왔다." = peeling 시퀀스. 한 발 물러서 보면 **모든 벽이 한 뿌리**.

**peeling 궤적**: 0.17 write-벽 →(학습)order_id 날조·루프 →(anti-fab)GBW →(substitute)표현적합 →offline 0.44 →(autopsy)anchor환각+width →(§22)width=소형·decomp회복 →(e2e §25)**다시 order_id 날조**.

**전 벽 = 동일 근원**: write-벽(order_id/item_id 날조)=GBW(new_item_ids 오선택)=width(multi-attr binding)=collapse(fetch-id 날조) = **LLM이 구체값을 만들려다 실패**. 처방 하나: **LLM=의도/op 명명+attr별 결정만, 구체값(read-id·write-변형·조합)=결정론 resolver.**

- **무한 regress 아님·수렴**: 구체값 생산을 다 offload하면 잔여=retry-복구+진짜 routing 모호성(유계·종류 다름).
- **전략**: offline 잔여 추격 중단(점점 작은 슬라이스). **통합 offload 1회 구현→e2e 측정**이 탑(fabrication+GBW+width)을 한 번에 무너뜨림.
- **wide-학습 폐기 근거 재확인(§23D)**: binding을 모델에 학습시키면 routing 퇴행(트레이드오프). offload는 routing-clean 모델 위에서 재학습0으로 회복(§22)=트레이드오프 회피. ⇒ **학습 말고 offload.**
- 설계 = `UNIFIED_OFFLOAD_DESIGN_2026_06_18.md`(아래).

## 27. ★★★정정 (2026-06-18) — §25-26·UNIFIED는 표류·order_id 날조 = *기존* P2b/P1 벽 (권위본 재독 후)
> 사용자 교정: P9·R8 필수 결론을 재론(원점회귀)하지 말 것. THESIS_STATEMENT·NL_PROCEDURE_OFFLOAD_THEORY·PRIMITIVE_COVERAGE_MATRIX 재독 후 정정.

- **§24-25 "발견"(order_id `#W0000001` 날조→collapse)은 신규 아님 = `PRIMITIVE_COVERAGE_MATRIX §3`(2026-06-15·`tau2_rootcause_census.py`)서 이미 확립된 P2b 'fetchable 값 날조-FIRST'(17/20·스키마 example 복사)**. 처방도 이미 = **R1b provenance(arg∈{user,tool}·스키마-example 거부)+fetch-first**([[R1B_PROVENANCE_DESIGN_2026_06_14]]).
- **UNIFIED_OFFLOAD_DESIGN = 폐기(표류)**: base 7B + 손-짠 bespoke resolver·학습0 = thesis의 **학습-leg(P1-P9 도메인일반)·전이-leg(ABox-swap) 둘 다 버림**. thesis는 "**P1-P9·생성원 어휘를 *학습*·ABox로 사실제공·결정론은 decidable·전이=ABox-swap**"(THESIS §1-3). bespoke는 그 정반대.
- **이번 세션 결과의 *올바른* 자리**: §17(content 생성원 𝔤 완성·P4-content)·§21(**𝔤-식별 학습·전이=C8 양성**·heldout 1.00·도메인일반=thesis 예측 적중)·§22-23(실행=offload·명명만 학습·wide-train 퇴행이 "execution 학습 금지" 확증)·§24-25(=기존 P2b/P1·신규 아님). **전부 이론 정합·이론이 예측한 것**.
- **올바른 다음(bespoke 아님)**: 확립된 프로그램 진행 — **P2b(R1b provenance·fetch-first)·P6(confirm)·P7(recovery)를 도메인-일반 *학습* + content 생성원 + ABox-swap 전이 검증**(THESIS §7 증명경로). agent_collapse는 P2b 학습으로 닫고, GBW/width는 §21-22(명명 학습+실행 offload)로. **e2e 통합 = P-primitive 학습된 모델 + 결정론 실행·ABox config**(손-짜기 아님).
- 교훈(박제): 권위본 먼저 grep([[feedback_check_authority_before_rederive]])·"하나 풀면 다음" peeling을 *새 발견*으로 격상 말 것·각 벽을 P1-P9에 매핑부터.

## 28. ★★★facet (3) native keystone = op-IR(§21)이 native 형식서 재현 = §23E 다리 확보 (2026-06-18·`facet3_native.sh`+`synth_to_nativefc.py`+`synth_native_eval.py`·held-out 252·새 어휘)
> §21 cross-bench 양성(op-naming 전이)이 op-IR 포맷이라 native agent 깸(§23E pass^1 0.075). 격리실험 facet(3)=op-naming을 **native `resolve_selection` tool_call**로 재학습(op-IR 폐기·anchor_id 모델제외)→held-out(새 어휘) op-naming/operand 채점(formalize 출력만·e2e 아님).

| | recognition | operand_acc | no_tool_call |
|---|---|---|---|
| base (native·무학습) | 0.758 | **0.286** | 0.008 |
| **trained (native·synth-only LoRA)** | **1.000** | **1.000** | 0.000 |

- per-op 전 7개(filter/argmax/argmin/rank/comparative/substitute/create) recognition·operand **모두 1.00**·emit 균형(각 36). base는 op-name 어렴풋(0.76) but operand 붕괴(comparative/filter/substitute 0~0.03·create 0.14).
- **★§21이 native 형식서 1.00 재현**: op-IR held-out 1.00 → native resolve_selection held-out **1.00**(recognition+operand). **형식 변경(op-IR→native tool_call)이 스킬 안 깸** = §23E 다리가 *formalize 레벨*서 확보. native라 agent-호환(op-IR처럼 텍스트 안 뱉음).
- 정직: held-out **synth 새-어휘(오프라인 단발 op-naming)**·§21 synth held-out과 동급 측정(이번엔 native). **미측정**: ① native cross-bench τ²(retail+airline·§21은 op-IR로 0.44) ② e2e multi-turn agent(§23E 진짜 시험). 다음 = 둘.
- 도구: `synth_to_nativefc.py`(생성기→native·anchor_id 제외)·`synth_native_eval.py`(tool_call 파싱 채점)·adapter `facet3_native_ep1`(7B·6020·ep1).

## 29. ★facet (3) gate① = native op-naming의 진짜 τ² 전이 = 음성·역전이 (2026-06-18·`facet3_tau2_native.sh`·`tau2_op_eval --native`·base+trained × retail+airline)
> §28(synth held-out native 1.00) 후 진짜 τ² NL 전이 시험. facet3_native_ep1 vs base, new_item_id acc.

| arm | new_item_id | recognition | 비고 |
|---|---|---|---|
| base retail | **0.34**(11/32) | 0.88 | — |
| trained retail | **0.19**(6/32) | 0.88 | **base 미만(역전이)** |
| base airline | **0.59**(16/27) | 0.89 | — |
| trained airline | **0.26**(7/27) | 0.81 | **base 미만** |

- **★역전이**: 학습본 < base (retail 0.34→0.19·airline 0.59→0.26). recognition은 비슷(~0.88·op은 맞힘·retail 전부 substitute) → **operand(set)을 base보다 더 틀림** = facet 4 학습이 τ²서 *해롭다*. = **§17/§23D "좁은 synth SFT 표면매핑 역전이" 재현**·synth held-out 1.00(§28)이 τ² 전이 보장 안 함.
- **함의①**: facet 4(operand·hard multi-attr keep-rest) = **학습-전이 안 됨 → offload(§22 decomposition)**. theory-vs-리뷰어서 hard operand는 **리뷰어(offload) 편**.
- **함의②(진단 필요)**: 내 facet3_native(0.19) < §21 MD_route op-IR(0.44) → 순수 native-실패 아니라 **synth_to_nativefc 데이터가 MD_route보다 좁을 가능성**. 분리 필요: 데이터 다양성 vs native 포맷 vs operand-비학습. (op-naming 자체는 native서 살아있음 §28·recognition 0.88.)
- 정직: n 작음(32/27)·retail은 op 자명(전부 substitute)이라 operand가 차별점·airline 16→7 격차 큼.

### 29-bis. ★정정 (사용자 교정) — retail 단일-op라 "recognition 중립" 무효·op-명명은 mixed-op서만 분석
- **retail = 전부 substitute(1 op)** → recognition은 "항상 substitute 뱉기"로 trivially 높음·**op-*구분* 시험 아님**. §29의 "facet 3 native 중립" **철회**.
- **op-명명 구분의 진짜 시험 = mixed-op만**: synth held-out 7-op(§28)=trained 1.00 vs base 0.76(양성·in-substrate)·τ² airline 2-op(substitute+create)=base recog **0.89 → trained 0.81**(약간 역전이·spurious comparative emit).
- **정정된 게이트① 본질**: 학습본이 τ²서 *두 축 다 약하게 역전이* — op-구분(airline 0.89→0.81) + operand(new_item_id 크게↓·지배). = §17 reverse-transfer. synth 1.00(통제)이 τ² 보장 못 함.
- **방법론 박제**: op-명명 전이를 real 벤치서 시험하려면 **mixed-op 벤치 필요**·τ²(retail) 단일-op라 부적합 → τ² 병목=operand(facet 4)·op-구분 시험은 synth(7-op)가 담당. (벤치-타당성: 단일-op 데이터로 op-명명 분석 금지.)

## 30. ★★★operand 전수 궤적조사 = 진짜 원인 3분해 (2026-06-18·`tau2_native_operand_autopsy.py`·gate① rows·base+trained × retail/airline)
> "operand 역전이"(§29) aggregate가 아니라 케이스별 emitted_set vs gold_set 전수분류. 진짜 원인 = 3개(섞임).

| | base retail | trained retail | base airline | trained airline |
|---|---|---|---|---|
| resolved_ok | 0.34 | 0.19 | 0.59 | 0.26 |
| missing_key(과소추출) | 9 | 6 | — | — |
| mixed_error(누락+오값) | 7 | **16** | — | **10** |
| wrong_value | 3 | 2 | 8 | 7 |

**원인 1 — 과소추출(missing_key·base 지배·§20-B 진성)**: `emit={brightness:low} gold={color:silver,brightness:low,power:battery}` = 3 중 1만 추출. base는 *보수적*(추출분 맞음·나머지 놓침). = multi-attr keep-rest formalize 난점.
**원인 2 — 값 enum-정규화(wrong_value·base+trained 공통·OFFLOAD-able)**: `"Google Home"→"Google Assistant"·"basic_economy"→"business"` = 개념 맞고 카탈로그 enum 부정확. **엔진 최근접 스냅으로 풀림(§20 값-스냅 offload)**·neither 안 함.
**원인 3 — 값 환각+spurious 키(trained 전용·역전이 아티팩트)**: trained `power="AC"`(gold battery·NL에 없음)·airline `emit={reservation_id:"XEHM4B"}`(set 키에 예약번호·§23D mixed_keys). = 좁은 synth가 synth 값-분포 주입 → τ² 틀린값/비-attr키. **trained만**.

- **★결론(operand=offload냐 learned냐 정답)**: **둘 다 아님·분해됨** — (1)어느 attr 바꾸나=formalize·LLM(missing_key·§22 per-attr decomp) (2)값→카탈로그 enum=**offload**(스냅·§20) (3)monolithic set 학습=**해롭다**(환각/spurious 주입·§17/§23D). **처방 = 모델은 attr+근사값 명명(per-attr·narrow-train 금지) + 엔진이 값 enum 스냅.** trained<base는 "operand 학습 불가"가 아니라 **좁은 synth가 환각 주입**.
- 정직: n 작음(retail 32·airline 27)·wrong_value의 enum-close vs 진짜환각 자동분리는 미정밀(예시로 질적 확인). 다음=enum-snap offload 구현→스냅 후 재채점(원인2 회수율).

## 31. ★★★operand 문제 = 새벽(K-sweep·width) 실험이 *이미* 답함 — §29-30은 §23D 재현(정리+반영)
> 사용자 교정: operand(역전이·offload vs learned)는 어제 새벽 K/width로 이미 분석·실험된 문제. §29-30이 그걸 참조 안 하고 재유도(anti-drift 위반). raw 정리(summary 파일 부재였음) + §22-23에 반영.

**A. width SET_EXACT (operand=multi-attr 과소추출·decomp-offload 회복·`width/width_*.json` 정리):**
| 모델 | w1 | w2 | w3 | w4 | w5 | decomp w4 |
|---|---|---|---|---|---|---|
| 1.5B | 0 | 0 | 0 | 0 | 0 | 0(포맷붕괴) |
| 3B | 0.74 | 0.81 | 0.71 | 0.72 | 0.66 | 0.68 |
| **7B** | 0.66 | 0.64 | 0.52 | **0.51** | 0.56 | **0.87** |
| **14B** | 0.88 | 0.75 | 0.62 | 0.62 | 0.77 | **1.0** |
| gpt-4.1 | 0.88 | 0.93 | 0.80 | 0.82 | 0.95 | — (frontier native) |
| gpt4o-mini | 0.65 | 0.75 | 0.62 | 0.55 | 0.62 | 0.70 |
| llama8b | 0.78 | 0.75 | 0.52 | 0.65 | 0.43 | **0.18(실패)** |
| qwen7b-or | 0.63 | 0.70 | 0.43 | 0.50 | 0.50 | 0.65 |
| **MD_widesubst(wide-train)** | **1.0** | **1.0** | **1.0** | **1.0** | 0.99 | — |
- set_size_bias 음수 단조(7B w4 −1.82 = ~1.8 attr 누락) = **과소추출**(=§30 missing_key). **decomp_set_recall≈1.0**(per-attr 물으면 다 추출) → **under-extraction은 decomposition-offload로 회복**(7B 0.51→0.87·14B 1.0)·단 llama8b 0.18=모델별. frontier(gpt41)=평탄=width 소형현상(§22).
- **MD_widesubst(wide-train)=synth 1.0** but **τ²서 퇴행**(§23D 0.44→0.30·mixed_keys·op_mismatch) = **monolithic set 학습이 라우팅 손상.**

**B. K-sweep τ² (diversity가 operand 전이 고치나·`ksweep/tau2_K*.json` 정리):**
| K | τ²_acc | recog |
|---|---|---|
| 1 | 0.22(kc)/0.34(rd) | 0.91/0.59 |
| 2 | 0.38/0.41 | 0.78 |
| 4-32 | **캡(§23C)** | recog→높음 |
- diversity는 **op-routing recognition을 올리나 τ² new_item_id는 operand에 캡**(§23C). = 다양성≠operand 해결.

**★결론(operand 문제의 *이미-확정* 답·§22-23)**: operand = (1)과소추출=**decomposition-offload 회복**(per-attr·§22·width 0.51→0.87) (2)값 enum-정규화=**offload 스냅**(§23B) (3)monolithic 학습=**퇴행**(§23D=§29-30) (4)diversity=**캡**(§23C) (5)scale=frontier native·소형은 decomp(§22). **§29-30(facet-3 native)은 §23D를 native서 재확인한 것·신규 아님.** ⇒ 처방 확정: **per-attr 명명(decomp·monolithic 학습 금지) + 엔진 enum-스냅.** 재유도 말 것.

## 32. ★★★operand 잔여원인 *전수확정* (enum-snap 시뮬레이션·`tau2_operand_cause_confirm.py`) — "decomp+snap만"은 거짓·제3원인=wrong-value-selection
> 사용자 교정: "decomp+snap만 남았나"를 *주장* 말고 *시뮬*로 확인. enum-snap을 실제 적용해 재해결→잔여 전수분류.

| 원인 | base retail | base airline | trained retail | trained airline |
|---|---|---|---|---|
| already_ok | 11 | 16 | 6 | 7 |
| needs_DECOMP(과소추출) | **13** | 0 | 13 | 0 |
| **RESIDUAL_OTHER(틀린값 선택)** | 3 | **8** | 0 | **7** |
| spurious_key | 2 | 0 | 10 | 10 |
| **SNAP_FIXES** | **1** | 0 | 1 | 0 |
| op_wrong | 2 | 3 | 2 | 3 |

- **★enum-snap = 거의 0(1)**: §30/§31 "snap이 원인2(정규화) 푼다" **정정/철회** — 값들은 exact거나 *유효하지만 틀린* 값이라 snap 무용. **snap은 레버 아님.**
- **under-extraction(decomp) = retail 13**(base=trained 동일·진짜 operand 난점·multi-attr keep-rest). decomp가 답(§22).
- **★RESIDUAL_OTHER = 제3원인(사용자 의심 적중)**: `retail length "31 inch"→gold "28 inch"·airline cabin "cancel"→gold "business"` = **유효 enum인데 틀린값·NL 값 오독**. snap 못고침(유효)·decomp 못고침(누락 아님)·**offload 아님(값이 NL에 있음·모델 오독) = 순수 formalize 정확도 잔여.** airline 8(지배)·retail 3.
- spurious_key(trained 10/base 2) = trained 역전이 아티팩트(좁은 synth·§29-30).
- **도메인 분리**: retail 실패=under-extraction(decomp)·airline 실패=**wrong-value-selection(decomp/snap 둘다 못고침)**. = airline single-attr cabin이라 누락 없고 값 오독.
- **★확정 결론**: operand 잔여 = (i)under-extraction→**decomp** (ii)wrong-value-selection→**neither(순수 formalize 정확도·NL 값 정확히 읽기)** (iii)snap=무시. **"decomp+snap만"=거짓·제3원인(값-선택 정확도)이 airline 지배.** = facet-4의 진짜 hard core는 *값-comprehension*(offload 불가·formalize)·decomp는 retail under-extraction만.

## 33. ★확정 — width×scale(under-extraction) 전체표 (어제 openrouter 스케일 결과 finalize·n=100/width)
| 모델 | w1 | w2 | w3 | w4 | w5 | decomp w1/w4 |
|---|---|---|---|---|---|---|
| 1.5B | 0 | 0 | 0 | 0 | 0 | 0.11/0(포맷붕괴) |
| 3B | 0.74 | 0.81 | 0.71 | 0.72 | 0.66 | 0.68/0.68 |
| 7B | 0.66 | 0.64 | 0.52 | 0.51 | 0.56 | 0.33/0.87 |
| 14B | 0.88 | 0.75 | 0.62 | 0.62 | 0.77 | 0.98/1.00 |
| qwen-7b | 0.63 | 0.70 | 0.43 | 0.50 | 0.50 | 0.23/0.65 |
| llama-8b | 0.78 | 0.75 | 0.52 | 0.65 | 0.43 | 0.15/0.18(실패) |
| gpt-4o-mini | 0.65 | 0.75 | 0.62 | 0.55 | 0.62 | 0.83/0.70 |
| **gpt-4.1** | 0.88 | 0.93 | 0.80 | 0.82 | 0.95 | —(평탄·벽없음) |

- **확정**: under-extraction = **소형 조건부**(frontier gpt-4.1 평탄 0.80-0.95·미만 하락)·decomp capable만 회복(7B 0.87·14B 1.0·llama-8b 실패 0.18=모델별).
- ⚠️ **미완(정직)**: 계획된 **llama-70b·qwen-72b·gpt-4.1-mini·mistral 미완료**(width 파일 없음) → frontier 단일점(gpt-4.1)+하락추세만·중간 스케일곡선 비어있음. 14B 비단조(noisy).
- ⚠️ **scope**: under-extraction(synth width) 축뿐 — **airline keep-rest/wrong-value(§32)와 무관**(별 실험 필요).

## 34. ★★★[CONFIRMED CAUSE·§29-34 정정] airline 실패 = *하네스 추출 버그*(reason_for_call 사용)·벤치/모델 결함 아님 (2026-06-18·사용자 교정 "리더보드가 작동하는데 벤치 결함은 말이 안 됨")
> 사용자: "벤치 문제면 다른 리더보드도 다 틀린 건가? 말이 안 됨." → 실제 τ² 태스크 전수확인.

**확정**: cabin은 **항상 `task_instructions`(전체 user 지시)에 명시**돼 있는데 `ma_gold_extract`가 **`reason_for_call`(요약)만** 사용 → cabin 명시를 버림.
| task | reason_for_call(추출에 씀) | task_instructions(진짜) | gold |
|---|---|---|---|
| 7 | "cancel flights" | "**upgrade to business** then cancel" | business |
| 14 | "change reservation" | "cheapest **business** round trip" | business |
| 29 | "change to nonstop, same dates" | "cheapest **Economy** (not basic)" | economy |
- 실제 τ² 에이전트는 user-sim 대화로 `task_instructions` 받음 → cabin 도출가능·**리더보드 정상**(벤치 멀쩡). 제 오프라인 eval만 요약으로 잘라 underspecified처럼 보임 = **하네스 버그.**
- **★대규모 정정**: §32(airline=formalize 잔여)·§34(케이스 결함) **둘 다 철회** = 추출 버그. §29-31 airline 수치·§30 retail missing_key(reason_for_call이 attr 누락?)도 **오염 가능** → task_instructions로 재추출·재측정 필요.
- 교훈: 오프라인 eval의 입력이 실벤치 에이전트 입력과 다르면(요약 vs 전체 지시) 전부 무효 — "NL→graph 실현가능 천장"(EXPERIMENT_DESIGN §1) 동형(답이 입력에 있나 먼저 확인). 사용자 "리더보드 교차조사" 규율이 잡음.

## 35. ★★★[CONFIRMED·스케일 분해] 스케일이 *무엇을* 사는가 = A(provenance/recovery)·B(operand)는 스케일-불변 = 유일 학습 타깃 (2026-06-21·retail n=342 denoised·`/tmp/curve.sh`·t2_failcensus_deep)
> 사용자: "7B↔32B 격차 엄밀 분석·스케일이 가져가는 것 명확히·작은 모델에 깨끗이 학습시킬 것만 찾아라."

**스케일 곡선 (pass^1·retail·tau2 학습0):** 7B 0.240 / 14B 0.518 / 32B 0.596. knee=7B→14B(+0.28). 프롬프트 효과 = **32B만 +0.047**·7B −0.015(역효과)·14B +0.009(null) = **instruction-following 크기-의존**.

**스케일이 사는 것 (floor 실패수 분해):**
| 실패(→규칙) | 7B | 14B | 32B | 판정 |
|---|---|---|---|---|
| **A** provenance/recovery(order_id 날조·R1/R8) | **76** | 23 | **3** | ★스케일 최대선물(96%↓)·**단 decidable→엔진이 7B에 줌** |
| INFO(NL) | 47 | 11 | 7 | 스케일 |
| D(미시도) | 21 | 11 | 10 | 스케일 |
| **B** operand/selection(write 인자·facet4) | **83** | 66 | **62** | ✗**plateau=스케일-불변 잔여** |
| E(partial) | 33 | 54 | 56 | ↑(거친실패↓→이동) |

**★확정 결론 (두 갈래·역할 다름):**
- **(가) 7B→32B 격차를 만드는 것 = A**(76→3). decidable(provenance check + fetch-first + A2-producer) → **엔진 offload가 7B에 줌**(학습 아님). = `t2_gate_patch` autofetch(`T2_AUTOFETCH`)로 실증 중(S-min arm1b).
- **(나) 스케일이 *못 고치는* 잔여 = B(operand/selection)**(83→62 plateau·32B도 62 실패=모두의 천장). 스케일도 32B-프롬프트(−10)만 약간·**스케일로는 안 줄어듦**. = §32-34 operand-formalize "진짜 hard core"와 동일. forced-replay 격리시 instruction으로 0.62 = **학습/내재화 가능.**
- ⇒ **작은 모델에 깨끗이 학습시킬 유일 타깃 = B(operand-formalize). A는 엔진·나머지는 스케일.** = 사용자 최초 operand 직관 적중(중간 "flow" 흔들림 정정).

**per-규칙 promptability 발현 = 스케일 임계:** A는 14B서 발현(7B 역효과 +11 → 14B 도움 −6 → 32B base near-0)·B는 32B만(−10). ⇒ 7B는 A·B 둘 다 프롬프트로 못 받음(A는 엔진, B는 학습).
