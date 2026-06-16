# HANDOFF 2026-06-16 PM — M-A 진단 완결 · capability/cost 두 날개 실측 · M-σ 한 사이클(in-dist 양성·전이 음성) · 다음=M-σ v2 추상합성 ablation

> 진입점 = 이 문서. 직전 = `HANDOFF_2026_06_16.md`(AM). 마스터 = `EXPERIMENT_DESIGN.md`(§0·§2·§7 06-16 라인 링크됨).
> thesis 수렴본 = `THESIS_STATEMENT_2026_06_16.md`. 결과 권위 = `ma/M_A_RESULTS.md`(§1-11).
> 불변 = [[feedback-thesis-tbox-transfer-direction]]·[[feedback-selector-verifier-deterministic]]·[[feedback-nl-formalize-llm-selection-deterministic]]·[[feedback-capability-vs-artifact-elicitation]].

## 0. 이번 세션 한 줄
**M-A로 write-벽 원인=NL→formalize *reasoning*(날조 아님) 확정 → floor sweep로 capability(binding)는 scale-bound·cost는 formalize-Pareto 분리 → 5 딥리서치로 메커니즘(분해+외부검증) 문헌검증 → M-σ 한 사이클(census→데이터검증100%→in-dist 96%→M-D 전이 *음성*·진단) → 다음=순수추상 selection-by-criteria 합성 ablation(C8 가장깨끗한 시험).**

## 1. ★다음 세션 첫 행동
1. **M-σ v2 설계서 완성·리뷰** (사용자 지시): `ma/M_SIGMA_V2_SYNTH_DESIGN.md` §0-9 이미 초안. **다음 세션서 정련**(특히 §1 NL-grounding·§9 ablation 매트릭스). 리뷰 후 구현.
2. **딥리서치 `w3l415qh5` 수확**: binding-다양 학습벤치 + 합성 선행연구(controllable-binding) → §1 합성 충실도·§9 입력. (완료 알림 확인.)
3. **구현(리뷰 후)**: `synth_selection.py`(추상 도메인/카탈로그/NL-요청/gold + knob 플래그 `--iso/--nl/--prov/--sem` + round-trip 검증) → resolver 확장(`$select`/`$ref`/literal) → `ma_synth_ablation_batch.sh`(config별 생성→7B LoRA SFT→M-D τ² eval→집계·2-GPU·~2.5-3h).

## 2. ★이번 세션 핵심 결과 (전부 `ma/M_A_RESULTS.md`)
- **M-A 3-arm(§1-7)**: write-벽 = NL→formalize **reasoning**(변형 오선택·"X만 바꾸고 유지" 오계산)·날조 아님. selector(출력) in-domain 패배(공정정보 Bfair로도). **fabrication 제거(필요)·reasoning 잔여 못 닫음(불충분).**
- **floor sweep(§8)**: A·Bfair·L0–L3 × {7B,14B,32B-Int8,**32B-bf16**,**72B-AWQ4**(coworker)}. ①정보 floor scale-불변(+16pp) ②**MSC≠scale대체**(7B ~0.53 천장=reasoning-bound) ③**formalize(L2b)=비용-Pareto**(토큰 절반·동급↑) ④selector 음성. **★32B-bf16=32B-Int8→Int8 cap 아님·14B→32B A-평탄(0.719)=진짜 reasoning-floor·72B-AWQ4도 천장 못깸 → binding 벽은 scale로 안 풀림.**
- **Sstep/Snover/SCv(§9·자동배치)**: Sstep(분해+결정론검증)=자유CoT 동률(0.656)·**~1/6 토큰(비용Pareto)**·but capability 천장 못넘음. **Sstep≈Snover**(검증이 ~1item만·binding 못잡음)·SCv 더나쁨. = elicitation/검증/샘플 어느것도 binding 벽 못깸.
- **5 딥리서치(`deepresearch/`)**: NL→SQL(decouple·reference-emit·grammar=form-not-meaning)·det-vs-learned TCO·plan-selection·constrained-decode(CRANE·schema제약이 추론손실)·input-formalize(Sufficient-Context floor 근접선행)·**small-model-reasoning(분해+*외부*검증·self-correct는 악화·Math-Shepherd)**. 메커니즘 전부 검증·신규=미점유 교차점.
- **★M-σ 한 사이클(§10-11)**: census(SOPBench/TaskBench=binding 0·**cfb만 threading**·selection은 세벤치 orphan) → cfb-derivation 데이터(round-trip 100%) → 학습(val 0.0101) → **in-dist 양성**(base 0%→M-σ **96%** $ref-correct·derivation-레벨 학습 *가능* 실증) → **M-D 전이 음성**(τ² all 0.41→**0.03**·over-$ref로 퇴화·payment harness아티팩트·selection orphan). **C8 1차=음성·진단적.**

## 3. ★현재 thesis 위치 (THESIS_STATEMENT)
- **두 날개**: (A)capability=분해+결정론 per-step 검증 (B)cost/전이=MSC+ABox+도메인일반학습.
- **실측**: cost 날개=formalize Pareto 양성·capability=binding floor가 scale/검증/분해로 안 풀림·전이(C8)=미증명(M-σ 1차 음성).
- **확정 진단**: 막힌 건 **selection-by-criteria binding** — 모든 학습벤치에 orphan·scale로 안 풀림·cfb-threading 전이 안 됨. ⇒ **순수 추상 selection 합성이 유일 경로**(M-σ v2).

## 4. M-D 음성의 3원인 → M-σ v2가 해결할 것
1. **selection orphan**: cfb=threading만 → 추상 selection-by-criteria 직접 합성.
2. **over-$ref(provenance 미구분)**: order_id(리터럴)까지 $ref → literal+$ref+$select 혼합 학습.
3. **harness**: payment를 dict-키로 둠(값-walk $ref불가) → 값으로·n 확장.

## 5. 도구/산출물 (전부 repo 커밋·`scripts/distill/ma/`)
- 설계: `M_A_PROTOTYPE_DESIGN.md`·`MIN_CONTEXT_FORMALIZER_DESIGN.md`(MSC)·`DECOMPOSITION_OPTIMALITY.md`·`../THESIS_STATEMENT_2026_06_16.md`·`M_SIGMA_DESIGN_2026_06_16.md`·**`M_SIGMA_V2_SYNTH_DESIGN.md`(다음 작업)**.
- 결과: `M_A_RESULTS.md`(§1-11 권위).
- 코드: `ma_resolver.py`·`ma_gold_extract.py`·`ma_eval.py`(arm A/Acot/Atwo/B*/L0-L3/Sstep/Snover/SCv·비용계측)·`ma_eval_scale.sh`(GPU/PORT/suffix 인자)·`dist_overcall_trace.py`·`ma_trace.py`·`m_sigma_data.py`(cfb→derivation·round-trip)·`m_sigma_eval.py`(in-dist)·`m_sigma_transfer_eval.py`(M-D)·각 watcher/배치 `.sh`.
- coworker: `COWORKER_REQUEST_2026_06_16_scale_floor.md`(완료·§8 반영)·`node_run_ma_72b.sh`(72B-bf16 노드·진행).
- 딥리서치 5건: `deepresearch/dr_*.md`.
- 학습 어댑터: `/home/woori/scratch/sft_runs/qwen7b_msigma`(cfb-derivation·M-σ v1).

## 6. gotcha / 인프라
- eval gotcha: `tb_evaluate.py`는 `--dependency_type`(`-t` 무효)·gold=`data_dailylifeapis_evalfull_qwen7b`(평범dir=task_nodes없어 KeyError)·metric `_no_matching` 접미사. **floor/M-A는 `ma_eval_scale.sh` 사용**.
- **GPU 충돌 주의**(이번 세션 3회 냄): GPU별 분리(GPU0:8013·GPU1:8014·etc)·serve 전 해당 GPU kill·port/log 분리. watcher-체인이 안전.
- **git fileMode**: `chmod +x`가 추적파일 mode 바꿔 `git pull --ff-only` abort 유발 → `core.fileMode false` 설정됨(woori).
- woori repo HEAD가 coworker push로 뒤처질 수 있음·pull --ff-only 확인.
- n=29/32 케이스 거침(±6pp 노이즈·k/32 granularity)·헤드라인엔 케이스 확장 필요.
- **결정론 검증·selector=결정론 불변**(LLM은 생성기만). concrete는 학습타깃 금지(v4-v7 재현).

## 7. 미해결/리스크
- **C8(전이)=thesis가 서고넘어지는 곳·현 음성**. M-σ v2 추상합성이 *깨끗한* 시험이나 전이 보장 아님(추상→실 grounding 갭 가능·§6-1 NL-패러프레이즈가 처방).
- 핵심 갈림: ablation(§9) −iso vs +iso가 **등방화→전이 이론(§5.10)을 실증** — 양성이면 헤드라인·음성이면 등방화 라인 재검토.
- 정직: "작은>큰" task-narrow·math 헤드라인 多→exchange 측정으로(보편주장 금지).
