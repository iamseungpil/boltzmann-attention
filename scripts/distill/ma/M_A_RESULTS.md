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
