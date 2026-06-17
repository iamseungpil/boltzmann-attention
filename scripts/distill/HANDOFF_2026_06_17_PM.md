# HANDOFF 2026-06-17 PM — 다도메인 content-routing 학습 = 생성원 closure 학습 실증 (다음 세션 진입점)

> 진입점 = 이 문서 + 마스터 [`EXPERIMENT_DESIGN.md §0 PM 블록`](EXPERIMENT_DESIGN.md). 권위 = [`GENERATOR_ALGEBRA_DESIGN_2026_06_17.md`](GENERATOR_ALGEBRA_DESIGN_2026_06_17.md)(생성원)·[`EXPRESSION_DIVERSITY_TRANSFER_DESIGN_2026_06_17.md`](EXPRESSION_DIVERSITY_TRANSFER_DESIGN_2026_06_17.md)(다양성)·[`ma/M_A_RESULTS.md §15-18`](ma/M_A_RESULTS.md)(실증). 불변 = `feedback-thesis-tbox-transfer-direction`(도메인-타깃 금지)·`feedback-expression-diversity-required-for-transfer`·`feedback-nl-formalize-llm-selection-deterministic`.

## 0. 한 줄
**오늘 세션 = thesis 이론 골격 완성(생성원 2축 closure·다양성 D\*·에너지-Lie). 다음 = 가장 thesis-결정적 실증: 등방화 합성으로 content-routing(NL→생성원 op) 학습 → retail+airline *동시* config-swap 전이 = "도메인-일반 생성원" 직접 입증.**

## 1. 오늘 도달 (context)
- **3층 아키텍처**(마스터 §0 PM): 층1 *무엇*=생성원 대수 / 층2 *어떻게*=표현 다양성 D\* / 층3 *분담*=offload(P2b).
- **생성원 2축 closure**: flow(P1-P9·`PRIMITIVE_MATRIX`) + content(8 op: filter/argmax/argmin/rank/comparative/substitute/create/project). **25벤치 적대적 딥리서치**: transactional tool-orchestration scope서 닫힘(FLOW=Böhm-Jacopini·CONTENT=Codd+Libkin-Wong)·value-derivation=P2b offload(NESTful 식)·scope밖 4축 제외. 문헌 whitespace.
- **실증 사이클**(`M_A_RESULTS §15-18`): comparative gloss 0→1.00 / C8 1차 합성양성=표면아티팩트 / **τ² 2차 음성**(C8-trained 역전이·op=exchange 동사복사) / 다양성 부분양성(표면붕괴 13→3·정확도 미회복). **궤적 전수규명: 정확도 미회복 근본=생성원 𝔤 부족**(τ² substitution을 우리 5-op이 못 표상·comparative로 억지 매핑+`to:` 발명).
- **핵심 진단**: 다양성(𝔥·표면)은 표면붕괴 끊으나, **생성원(𝔤) 완전성이 선행**. = 다음 작업의 동기.

## 2. ★다음 작업 = 다도메인 content-routing 학습 (도메인-일반 직접 실증)
**가설**: 8개 content 생성원 + derivation(P2b)을 *등방화 합성*으로 라우팅 학습 → retail·airline 두 도메인에 **config swap만으로** 전이. 닫히면(두 도메인 같은 라우팅) = 도메인-일반 입증·공격 무효.

**왜 결정적**: GENERATOR_ALGEBRA §6 "다도메인 전이"가 closure를 *학습으로* 증명하는 단계. substitute가 retail·airline 둘 다 지배(실증·§3) → 동시 전이가 "retail 특화 아님" 직접 입증.

### 2.1 설계 (τ²-blind 하드룰)
- **생성원 집합**: filter·argmax·argmin·rank·comparative·**substitute**·**create**·project (8) + **derivation=P2b**(date_resolver/unit_convert tool 호출·모델은 명명만).
- **등방화 합성 데이터**: 각 생성원 라우팅·**표현 다양화**(K-sweep 교훈·`synth_expr` 축분해 L×S×P×R·단일템플릿 금지) · 등방 어휘/스키마. **표현 풀은 일반 축서 설계·τ² 역설계 금지**.
- **ABox config (도메인별)**: retail·airline catalog → attr-타입(ordinal/categorical)·도메인 어휘 vocabulary. config만 swap·재학습 0.
- **학습**: content-routing LoRA (NL+config → content op-IR). 도메인 사실 X·라우팅만.
- **전이 측정**: retail+airline **동시**(같은 모델·config swap) → 각 도메인 selection accuracy + op-routing recognition.
- **판정**: 두 도메인 닫힘(같은 라우팅·둘 다 ↑) = 도메인-일반 / 한 도메인만 = 특화(위험 신호).

### 2.2 구현 단계 (첫 행동)
1. **substitute/create 생성기 추가** → `synth_depth.py` (현 selection 5-op만). substitute = old⊕change override·create = base∅ 설정. 통제 합성(gold 라운드트립).
2. **airline 케이스 추출** → `ma_gold_extract.py` 확장 (현 retail exchange만). airline `update_reservation_flights/baggages/passengers`(substitute)·`book_reservation`(create). offline 케이스(NL+catalog+gold).
3. **resolver 확장** → `tau2_op_resolver.py`에 substitute/create 실행 (among=old⊕change·create=new record). derivation=P2b는 tool stub.
4. **등방화 content-routing 학습데이터** → `c8_build_sft.py` 확장(8 op·`render_nl_diverse`·표현 다양화). 
5. **학습** → `lora_train_chat_toolcall.py` (8-op routing).
6. **eval** → `tau2_op_eval.py` 확장(retail+airline·substitute/create 채점) → 동시 전이 매트릭스. `M_A_RESULTS §19` 박제.

### 2.3 ★gotcha (실패 함정)
- **도메인-타깃 금지**(`feedback-thesis-tbox-transfer-direction`): 생성원·표현 풀 τ²-blind·일반 축서. airline 보고 op 추가 금지(taxonomy 연역만).
- **substitute "X만 바꾸고 유지" reasoning**(M-A wrong_criteria 원인·`M_A_RESULTS §3`): among = old_options ⊕ requested-change. 모델이 *무엇 바꾸고 무엇 유지*를 정확히 — 이게 진짜 난점(다양성 아님).
- **표현 다양성 필수**(단일템플릿=표면매핑 주입·§17b 역전이): `render_nl_diverse`/`synth_expr` 사용. 동사 op-무관.
- **value-derivation=P2b**(NESTful 식·§9): 모델이 계산 X·도구 명명+threading. date_resolver/unit_convert stub 제공.
- **comparative=ordinal만**: categorical attr(color·switch)을 comparative로 라우팅 금지(궤적 규명 오류). ABox attr-타입 config가 ordinal/categorical 명시.
- **정직 경계**: retail+airline 둘 다 서비스-API 트랜잭션 편중 — 닫혀도 "그 슬라이스서 도메인-일반"·적대적 도메인(다른 content 패턴) 추가 필요.

## 3. 진행 중 (다음 세션 시작 시 확인)
- **K-sweep 배치**(GPU0=random·GPU1=kcenter·`kshot_sweep.sh`): 표현 다양성 D-전이 곡선. 학습 느림(~수시간·전수 미완). 첫 확인 = `ksweep_summary.py`. 예측 = 표면붕괴 D↓·정확도 D-무관(생성원 부족 입증).
- **coworker B-budget arm C**(32/72B): HF dataset `iamseungpil/sopbench-trackb-h200/depth_scale` (woori HF 토큰 없어 미확인). 사용자가 HF서 확인 or 토큰 설정.

## 4. 도구 (오늘 만듦·전부 repo `scripts/distill/ma/`·검증됨)
`synth_depth.py`(render_nl_diverse 표현 다양화)·`synth_expr.py`(K-제어 축분해)·`expr_diversity.py`(D=eff-rank·kcenter)·`c8_build_sft.py`(--K/--method/--diverse)·`c8_batch.sh`·`kshot_sweep.sh`·`ksweep_summary.py`·`tau2_op_resolver.py`(op-IR→item)·`tau2_op_eval.py`(--synth_format)·`comparative_diag.py`/`comparative_fix.py`·`depth_eval.py`(--gloss).

## 5. 권위 문서 (방향 흔들리면 읽을 것)
- 마스터 [`EXPERIMENT_DESIGN.md §0 PM 블록`](EXPERIMENT_DESIGN.md) — 3층·문서지도.
- [`GENERATOR_ALGEBRA_DESIGN_2026_06_17.md`](GENERATOR_ALGEBRA_DESIGN_2026_06_17.md) — 생성원 2축·25벤치 닫힘·value-derivation=P2b.
- [`EXPRESSION_DIVERSITY_TRANSFER_DESIGN_2026_06_17.md`](EXPRESSION_DIVERSITY_TRANSFER_DESIGN_2026_06_17.md) — 다양성 D\*·리뷰 반영.
- [`ma/M_A_RESULTS.md §15-18`](ma/M_A_RESULTS.md) — comparative·C8·다양성 실증.
- [`NL_PROCEDURE_OFFLOAD_THEORY §10`](NL_PROCEDURE_OFFLOAD_THEORY_2026_06_17.md) — 에너지-Lie 통합.
- 메모리 `reference-repo-energy-lie-prior-work`(boltzmann·lie_group 인벤토리·재census 금지).

## 7. ★진행 추가 (2026-06-17 PM(2) — 이 세션) = 생성원 5→7 + 표현적합성 양도메인 증명 + §20 큐잉
- **생성원 5→7 구현 완료**(전부 repo `ma/`·커밋·로컬 round-trip 420/420):
  - `synth_depth.py` = substitute(keep-rest: anchor⊕set)·create(full set) 생성기+엔진(`resolve_operation`)+diverse 표현풀(verb op-무관).
  - `depth_eval.py` = op-IR spec/gloss에 substitute/create·`build_arm_B_user` 범주형(ord 없음) 처리.
  - `tau2_op_resolver.py` = substitute/create(τ² nested options)·cabin ORD_WORDS 등록.
  - `ma_gold_extract.py` = `--domain airline`(cabin 차원: update_flights=substitute·book=create·retail과 동형 스키마).
  - `tau2_op_eval.py` = multi-domain(`--cases`)·create-aware prompt·by case_op + op-routing recognition.
- **★★표현적합성 양도메인 증명**(`tau2_subst_oracle.py`·`M_A_RESULTS §19`·키0·GPU0): **retail 32/32 + airline 27/27** = *동일* op-IR/resolver로 두 도메인 닫힘. §17/§18 over-comparative 근본=5-op 표현부재 확정·substitute가 닫음. = 생성원 도메인-일반(표현측).
- **§20 학습-전이 = 큐잉됨**(GPU-blocked·K-sweep 점유): 데이터 전부 빌드 검증(remote `…/c8/multidomain/route_sft.jsonl` 6000·7-op 균형·`c8_eval_heldout_div.jsonl` 250·retail/airline 케이스). **detached 워처 `md_wait_launch.sh`(PID 가동중)가 K-sweep 종료 자동감지→`multidomain_route.sh <GPU> 8027 1 6000` 실행**(synth-only 7-op 라우팅 LoRA→retail+airline config-swap 전이·base floor/ceiling 포함).
- **★다음 세션 첫 행동**: ① K-sweep 종료확인 + `ksweep_summary.py`(다양성 곡선). ② `…/c8/multidomain/logs/wait_launch.log` + `…/multidomain/results/MD_route_ep1__{retail,airline}_g0.json`(§20 매트릭스) 확인 → **판정: 두 도메인 동시 ↑(같은 라우팅)=도메인-일반 입증 / 한쪽만=특화 위험**. `M_A_RESULTS §20` 박제. ③ 음성 시 궤적 규명(over-comparative 잔존? substitute 미명명? airline cabin 단일attr degenerate?).
- **신규 도구**: `tau2_subst_oracle.py`·`multidomain_route.sh`·`md_wait_launch.sh`·`ma_gold_extract.py --domain airline`.
- **정직(이 세션)**: 오라클=IR 수동(학습 전이 아님)·airline cabin=단일 ordinal attr(catalog-레벨 keep-rest degenerate→op-라우팅 시험 위주)·project(8번째 content op)=read/getter-shaped라 보류(offload).

## 6. 정직 라벨 (과대주장 금지)
- finiteness/closure만·**minimality 금지**(Kozen-Tseng). scope=transactional tool-orchestration.
- 생성원 닫힘=경험검증(다도메인 전이가 시험). substitute 일반=실증(tau2 5도메인)이나 학습-전이는 미증명(이 작업이 시험).
- Lie geometry 실증/에너지 정식화-only·basin-hierarchy 미실행/T_eff GQA-rejected.
- arxiv snippet-only id(2602.*·2603.*·2508.*) 재검증 후 인용(`feedback-arxiv-citation-discipline`).
