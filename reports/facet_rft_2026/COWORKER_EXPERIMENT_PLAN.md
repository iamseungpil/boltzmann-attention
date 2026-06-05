> ⚠️ **마스터 = `scripts/distill/EXPERIMENT_DESIGN.md`** (목표·실험순서 SFT→RFT→xattn·헤드라인 지표 권위본). 이 문서는 **Track B(coworker, 32B/72B) 실행 상세** — 목표/순서 변경은 마스터에서. (32B 바닐라는 leaderboard 인용=재측정 금지.)

# Coworker 실험 계획서 — Workflow Ontology Agent on SOP-Bench

> 대상: 4× A100 80GB coworker. 공유 채널 = GitHub `iamseungpil/boltzmann-attention` branch **`facet-rft-2026`**.
> **★ 모델 분업 (확정 2026-06-01)**: **coworker = Qwen2.5-32B + Qwen2.5-72B** / **Track A(우리) = Qwen2.5-7B + Qwen2.5-14B**. 동일 arm·설정으로 돌려 모델 크기 효과 비교. coworker는 대형모델(32B/72B) arm-0~4 매트릭스에 집중; Track A는 소형(7B/14B) 파일럿·구현·검증.
> 본 계획은 `reports/EXPERIMENT_DESIGN_v1_7_facet_rft.md` **§16(SOP-Bench 피벗)**을 구현한다. **먼저 §16 + `scripts/distill/WORKFLOW_ONTOLOGY_DESIGN.md`(특히 ★§9 LLM-in-loop)를 읽을 것.** (§15.9~15.14 = tau2 기반 개념 원본, substrate만 SOP-Bench로 이전.)

> ### ★★★★★★★★★★★ v1.41 (2026-06-05 PM) — 로드맵 확정: cross-domain → should_F → B축 + Fix-3 완료(50.75%)
> **Fix-3 STOPSUCCESS LIVE**: 공식 success(리더보드 지표) **base 29.85%→…→50.75%(68/134)**, should_T full **40/48=정직천장**(잔여8=PartA버그), 회귀0. base Qwen2.5-7B 5.22%→**50.75%=오픈소스 SOTA(Llama70B 42.54%) 추월**.
> **★로드맵(확정)**: ① **cross-domain 전이**(A축 scaffold ABox-swap 재학습0 일반화, 설계 [`../../scripts/distill/CROSS_DOMAIN_TRANSFER_DESIGN.md`](../../scripts/distill/CROSS_DOMAIN_TRANSFER_DESIGN.md) 리뷰대기) → ② **should_F 거부축**(전체% 유일 잔여 레버, should_T 천장) = A축 논문 완성 → ③ **B축 weight 내재화**(verifier-DPO/RFT, C축 자기-emit은 LOCK死). **coworker 32B/72B는 ③ B축(decision-emission 스케일) 전용** — 단 ① cross-domain에 32B/72B를 같은 stack으로 태우면 "scaffold 전이 × 스케일" 매트릭스도 가능(공식 success 보고 필수). 인프라: 7도메인 ontology·도메인규칙·getter_map·task 전부 존재(induce 완료)=cross-domain authoring 0.

> ### ★★★★★★★★★★ v1.40 (2026-06-05) — H3 offload 사다리 LIVE + ★지표 정정(공식 success) + 잔여 fix
> **진입점 = [`../../scripts/distill/HANDOFF_2026_06_05_PM_argfix_dggate_ladder.md`](../../scripts/distill/HANDOFF_2026_06_05_PM_argfix_dggate_ladder.md) + [`../../scripts/distill/RESIDUAL_PREMATURE_DIAGNOSIS_2026_06_05.md`](../../scripts/distill/RESIDUAL_PREMATURE_DIAGNOSIS_2026_06_05.md).** 이번 세션(Track A, 7B):
> 1. **H3 decision-offload LIVE** (`check_permitted` 결정론 게이트 + active-H3) + 무재학습 사다리 **ARGFIX→VALFIX→KEEPTUPLE→DGGATE = BOTH(dg∧acc) 29** (실제 evaluator·회귀0; Guard-2 PASS). + **Fix1 LOGINFIRST**(login front-load)·**Fix2 LOGINCALL**(cred-absent login-call).
> 2. **★★지표 정정 (coworker 필독)**: 프로젝트 내부지표 **BOTH(dg∧acc, should_T만)는 공식 `success`(리더보드 기준)를 8~12 과대계상**. **리더보드/논문 비교·coworker 32B/72B 수치는 반드시 공식 success**(ntce∧cnv∧dbm∧acalled∧dg, 전체 134, tool_full)로 보고. 검증=오픈소스 README 정확일치. 상세 = [`../../scripts/distill/LEADERBOARD_METRIC_GROUNDING_2026_06_05.md`](../../scripts/distill/LEADERBOARD_METRIC_GROUNDING_2026_06_05.md).
> 3. **우리 공식 success**: base 29.85%→loginfirst 37.31%→logincall **40.30%(54/134)**; base Qwen2.5-7B 3.73~5.22% → tbox_v2(7B+SFT+scaffold) **40.30%**(Llama70B 42.54% 근접). **진짜 잔여 레버 = goal-call LOOPING 차단(Fix3 STOP-after-success)** + **should_F 거부축**(현 30%, 전체% 레버).
> 4. **▶ coworker(32B/72B) 반영**: B축(decision-emission) 스케일 시험 시 ① **공식 success로 보고**(BOTH 금지) ② DGGATE/LOGINFIRST/LOGINCALL는 A축 scaffold(상한, 모델능력 아님)임을 분리 ③ cred-absent 통과(login-call)는 evaluator quirk(auth-성공 아닌 call-order)·internal_get_database=react 누수(offered 아님)임을 인지=[`../../scripts/distill/INTERNAL_GET_DATABASE_GROUNDING_2026_06_05.md`](../../scripts/distill/INTERNAL_GET_DATABASE_GROUNDING_2026_06_05.md). 나머지(v1.39 A/B축 분리·LOCK·gather 스케일곡선·credential 격하)는 유효.

> ### ★★★★★★★★★ v1.39 (2026-06-04, 0a 진단 후) — credential confound 기각 + sub-7B 추가 + 매트릭스 비용역전 수정
> **Gate-A/0a 진단(zero-cost, `eval_t1c` 전수재파싱; 권위본 ★★Gate-A 기록):** v1.38 #2의 "login=False 16=credential 진짜부재"를 **정정** — eval-confound·bench-defect·credential-absence **모두 0건**, 지배원인 = **over-call 31/48**(태스크 constraint에 없는 auth 호출·실패; admin 실제비번 미제공·환각). **함의 = credential-augment는 *학습* 모델 천장 못 올림**(base 모델엔 absence 유효=측정 통제로만). 외부 리뷰(R1~R9)+0a 반영 **수정 매트릭스/순서**:
> 1. **★R1 비용역전(단 하나의 구조 수정)**: 7B가 이미 gather→임계 ≤7B → 7B–72B로 *min-scale 국소화 불가*. **gather 스케일곡선 = 0.5/1.5/3/7/14B(전부 Qwen2.5-Instruct, 싼 쪽; Track A 확장)** + **대형 32B/72B = B축(decision-emission) 전용**(LOCK이 스케일로 깨지나 — 여기만 스케일 의미). full 매트릭스를 72B에 태우지 말 것.
> 2. **★A1(능력 바닥 통제)**: 각 스케일 base-gather + valid-tool-call rate 먼저 → sub-7B "gather 못함" vs "기계적 도구조작 못함" 분리.
> 3. **★R3/A2(offload=메모장형)**: `check_permitted` = 결정론 게이트 over *모델 게더결과*(oracle 아님=upper bound). unknown→deny 사전등록. BOTH = dirgraph ∧ goal-call-correctness ∧ 게이트.
> 4. **★R4/R5/R7/R8 사전등록**: LoRA r 고정+1스케일 rank-sweep / 전이 multi-holdout(7B full-LODO ≥4)+≥70% 상대 사전등록 / tool-change 7B 파일럿 선행 / 도메인-mix 고정·seed 2.
> 5. **★credential-augment 격하**: 비번만 surface(누출 금지)·realistic 병행·base 측정 통제로만. **coworker 32B/72B는 B축만이므로 credential-augment 의존 낮음**(B축은 결정-emission NULL/break 시험). positioning(R9): offload=LLM-Modulo 인용, novelty=gather 스케일-임계+LODO 전이+도구변경 robust.
> ⚠️ 아래 v1.38 본문은 **#2(credential 16)·매트릭스 스케일이 위로 대체됨** — 나머지(A/B축 분리·LOCK·value-prop)는 유효.

> ### ★★★★★★★★ v1.38 (2026-06-04 PM) — 논문 축 = **robust gather 스케일-임계(7B/14B/32B/72B)** + 결정 offload. LOCK 후 정렬.
> **이번 세션 결과(필독, 상세=권위본 `Exp-4-rung1-{upperbound,T1c}` + `RUNG1_SOURCE_LADDER_DESIGN.md` LOCK):**
> 1. **T1c(grounded-permitted@s1) NULL**(BOTH 1<C-none 3). **LOCK 발효**: 결정 terminal에 truth/derivation emit하는 SFT 스캐폴드(treeval→inductive→T1c)=3-NULL 종결. over-refuse/over-call/early-act=MODEL회귀=SFT-positive 불가. **emission 변종 추가 금지.** (범위: gather-grounding/credential teacher/2-agent SFT는 유효=over-prune 금지.)
> 2. **게이트 진단**: login=False 19/48 중 **16=credential 진짜부재**(prompt·user_known에 비번 없음→모델 환각). = KNOWN credential-조건화 이슈, 현 base가 credential-부재 regime이라 should_T capped.
> 3. **★논문 헤드라인 재정렬 = 두 축 분리**: **A축 robust gather(도구선택+완전성+alias robust+LODO 전이)가 어느 최소 스케일서 학습되나** + **B축 결정 게이트는 SFT로 32B/72B서도 NULL인가(LOCK 스케일-시험) → check_permitted로 offload**. 클레임 = "robust gather+전이는 [min scale] 학습가능, 결정은 verifier offload"(현장 정합).
>
> **▶ coworker(32B/72B) 할 일 (v1.37 대체):**
> 1. **★gather 스케일곡선**: gather SFT(LODO holdout=bank) → **dirgraph_satisfied**(gather 1차지표)·**LODO 전이**(held-out/in-domain)·**alias on/off + 도구 rename/add/remove robust**. = Track A 7B/14B와 동일조건 → **7B/14B/32B/72B 곡선으로 "robust gather 최소 스케일" 규명.**
> 2. **★B축 LOCK 스케일-시험**: 결정-emission(treeval@s1, slot-fix HEAD)을 32B/72B서 → **BOTH가 7B처럼 NULL인가**(=LOCK 스케일-불변 확증) vs 깨지는가(=스케일-임계). *가정 말고 측정.*
> 3. **★필수 통제(안 하면 스케일비교 무효)**: ①**credential-augmented regime**(login confound 16 제거=비번 surface; Track A가 메커니즘 확정 후 공유, 모든 스케일 공통) ②bench-defect 제외(cancel_cc/pay_bill_cc ~8) ③`check_permitted` offload로 BOTH=gather-bound 측정(결정 변수 제거). ④no-400 client(HEAD≥434c515)·헤드라인 race 가드·freshness.
> 4. **버려진 축**(추구 금지): 트리-emit 정교화·inductive·depth-recurrence·getter-hint. source는 s1(배포현실) 중심.
> 5. **value-prop**: 헤드라인 = robust gather + **도구변경 robust 전이**(재학습0) + 중첩도구 disambiguation(alias). 결정/스케일-KV는 OISA·offload 영역.

> ### ★★★★★★★ v1.37 (2026-06-04) — Track A 대전환: 트리평가-형식 종료 / 병목=결정 / **T1c grounded-permitted @ source=1** + slot-fix + 2-agent
> **이번 세션(Track A, 7B) 변경 — coworker 필독. 상세 진입점 = [`scripts/distill/RUNG1_T1C_DESIGN.md`](../../scripts/distill/RUNG1_T1C_DESIGN.md) + `RUNG1_SOURCE_LADDER_DESIGN.md` + 권위본 `SOPBENCH_EXPERIMENT_RESULTS.md`(Exp-4-rung1-{v3-AB,v3ind,upperbound,T1c}).**
> 1. **트리평가-*형식* 라인 전부 NULL/종결** — 추구 금지: 단일식 grounded gate(`Exp-4-rung1-v3-AB`: "회귀"는 maxtok=24 truncation 아티팩트, maxtok=1024 재시험 BOTH 2→5=control과 동) + inductive reduction 체인(`v3ind`: BOTH 3<4, fabrication+over-gather) + depth-recurrence(deep-research: Huginn from-scratch=retrofit 불가). **조건수별 BOTH 균일 바닥(1조건도 0) → serial-depth/조건수는 병목 아님.**
> 2. **★병목 = 결정(permitted 콜드붕괴), 구조 아님** — `Exp-4-rung1-upperbound`(Agent2@oracle): 구조 제공(source=1) BOTH **3 = source=3** = 구조만으론 BOTH 무개선(게더↑ dirgraph 29→34·STOP 40→49%는 개선). 전수조사: **gathered_then_REFUSE 29**(완전게더 후 거부). → source=1은 게더만 돕고, *게더 후 ACT/STOP 결정*이 벽.
> 3. **★slot-fix** (`build_tbox_planner_sft._add_req` dedup 키에 args(pm) 포함): 같은-predicate-다른-args leaf(transfer_funds: internal_check_username_exist on username AND destination_username) 누락 버그 → required서 destination 드롭 → unknown → fallback. 수정 후 teacher 천장 **34→42**(transfer_funds 8 grounded 회복; fallback 14→6 = 순수 bench-defect cancel_cc/pay_bill_cc만, BUGREPORT 대상). **coworker 재학습 시 이 fix 포함된 HEAD 사용 필수.**
> 4. **★현 헤드라인 레버 = T1c = grounded-permitted @ source=1**: permitted을 콜드 should_succeed 추측 → *주어진 구조의 leaf-truth AND/OR 룩업*으로(= **treeval@s1**, v3 실패는 source=3 fabrication 탓·s1이 구조 제공해 차단). teacher byte-identical(s1/s3 동일 타깃, 프롬프트만 다름). **학습 중**(7B, s1+s3 slot-fixed 병렬).
> 5. **★북극성 = 2-agent (단일 base + 2 LoRA)**: Agent1(NL→dirgraph 구조추론, 검증=GT constraints) + Agent2(구조+게더→ACT/STOP, 검증=결정론 evaluator). vLLM 멀티-LoRA로 단계별 `model` 선택. (§11 of RUNG1_SOURCE_LADDER_DESIGN.)
>
> **▶ coworker(32B/72B) 할 일 (v1.36 매트릭스 대체):**
> 1. **★T1c를 32B/72B로**: `build_tbox_planner_sft.py --alias --source 1 --treeval`(slot-fix HEAD) → LODO(holdout=bank) 학습 → eval `SOPBENCH_SOURCE=1 ... PLAN_MAXTOK=1024`. **헤드라인 = BOTH**(천장 42). **대형모델이 grounded-permitted를 7B보다 잘 학습해 BOTH↑인가** = 핵심 질문(7B가 gathered_then_REFUSE 29를 ACT로 전환 못 하면 32B/72B가?).
> 2. **2×2 (slot-fixed)**: {none(permitted=should_succeed), treeval(grounded)} × {s1, s3}, 32B/72B. interaction(grounding이 s1서 더 도움)=fabrication-attribution. (none 셀 = `--source N --scratchpad` treeval 없이.)
> 3. **buggy harness 주의**: eval client `two_stage_client._resolve` **no-400 fix(HEAD≥434c515) 필수**(goal 부재 시 400→ACT 태스크 드롭). 헤드라인 python은 run_evaluation **후** 실행(nt=0 레이스). freshness 가드(eval>adapter).
> 4. **버려진 축**: getter-hint(v1.36)·트리-emit·depth-recurrence = 추구 금지. source 축은 **s1(배포현실=OISA Score-Prune-Present) 중심**.
> 5. **value-prop 정렬(현장)**: 헤드라인 클레임 = (a) 중첩도구 disambiguation(alias) (b) **도구변경 robust 전이**(도구 add/rename/remove→리스트만, 재학습0) (c) ABox-swap 전이. 스케일/KV는 OISA 영역(우리 코어 아님).

> ### ★★★★★★ v1.36 (2026-06-02) — Track A 근본원인 해결(condition→getter auto-derive) → coworker 재정렬 + 지표 교정
> **이번 세션(Track A, 7B) 변경 요약 — coworker가 알아야 할 것 (상세는 링크):**
> - **근본원인 확정 = condition→getter 맵 결손**(절차-학습성 문제 *아님*). permitted-collapse(거부 붕괴/should_T 0)는 *정책조건이 getter에 안 묶여 미게더*된 탓이었다. **ungroundable ≈ 0**(7도메인 158/158 grounded, env predicate-source로 정의적 확정; "28% ungroundable"은 휴리스틱 결함이었음). → 상세 [`SOPBENCH_EXPERIMENT_RESULTS.md`](SOPBENCH_EXPERIMENT_RESULTS.md) **Exp-4-precheck-FINAL**.
> - **auto-derive v2** (`scripts/distill/sopbench/autoderive_getter_map.py` → clone `induced/getter_map.json`): predicate 소스 정적파싱으로 condition→getter-**집합** 자동도출(구조적·전수·전이무결·토큰추측 아님; multi-getter). hand bank-map recall 8/9.
> - **teacher 수정** (`build_tbox_planner_sft.py`): getter-집합 required-set 배선 + **터미널 = GT `should_succeed`**(login/credential-block·OR 과잉거부 해소). 검증: 터미널 ACT/STOP = should 정확 수렴(bank **48/86**, 7도메인 전부). → 상세 [`SOPBENCH_EXPERIMENT_RESULTS.md`](SOPBENCH_EXPERIMENT_RESULTS.md) **Exp-4-mapwire**.
> - **설계 priority-lock**: Track A(데이터-fix) **선행** / 합성 §3.0b는 **fallback**(grounding 고유 scope 없음 확정) / **★지표 교정**. → 상세 [`scripts/distill/EXPERIMENT_DESIGN.md`](../../scripts/distill/EXPERIMENT_DESIGN.md) "Rung 1 진단·우선순위 확정" + §3.0b.
> - 7B LODO 재학습 launch됨(새 teacher 데이터).
>
> **▶ coworker(32B/72B) 재정렬 — 무엇이 바뀌나:**
> 1. **★지표 교정 (필수, 전 셀 적용)**: 헤드라인 = **ACT-recall | 충분게더**(should_T서 게더 완료 후 실제 ACT한 비율; over-refusal 붕괴를 직접 검출) + **STOP-recall 분리 보고**(should_F서 올바른 STOP). 붕괴는 비대칭(STOP-recall=1·ACT-recall=0)이라 **합친 총점/Mean Pass Rate가 ACT-recall=0을 가림** → ⚠️**Mean Pass Rate 단독 헤드라인 금지**. should_T **BOTH(dirgraph∩goal)** 유지, should_F gross. ordering-violation(조기 ACT)은 3차 가드레일.
> 2. **★★in-context 플래너 getter-hint 정렬 (중요 — 신규 prerequisite)**: 현 `two_stage_client.py:build_v2_prompt`는 **condition→getter 매핑을 프롬프트에 노출하지 않는다** → 32B/72B in-context는 7B와 똑같이 정책조건의 getter를 **cold-infer**해야 함 = **permitted-collapse 재현 위험**. 신규 ablation 축 = **getter-hint on/off**:
>    - **hint-OFF**(현행 build_v2_prompt): "강모델이 구조 맵 없이 condition→getter를 *추론*하는가" (= 7B가 못 한 것을 32B/72B는 하나).
>    - **hint-ON** (✅**구현완료·검증**: env **`SOPBENCH_GETTER_HINT=1`** → `build_v2_prompt`가 getter_map에서 "to verify [condition], call [getter(s)]" 블록을 프롬프트에 주입; 고친 7B teacher와 apples-to-apples). `SOPBENCH_GETTER_MAP` 미지정시 clone `induced/getter_map.json` 사용. **OFF시 프롬프트 byte-identical**(레거시/teacher 불변 검증).
> 3. **매트릭스 (v1.35 축 유지 + 신규축)**: {alias on/off} × {source 1/3} × **{getter-hint on/off(신규)}**, 32B·72B, bank held-out(+가능하면 LODO). 비교앵커 = 32B leaderboard 40.30(바닐라, 재측정 금지)·고친 7B SFT(재학습 후 갱신).
> 4. **합성 §3.0b는 coworker 범위 아님**(grounding scope 없음 확정; many-conditions 일반화 stress-test가 필요해지면 그때 Track A가 합성 생성).
>
> **▶ 실행**: v1.35 STEP1-4 레시피 그대로(32B/72B 서빙 + `run_simulation.py --two_stage --two_stage_v2`) + env 토글로 셀 선택:
> ```bash
> #   매트릭스 셀 = {SOPBENCH_ALIAS} × {SOPBENCH_SOURCE} × {SOPBENCH_GETTER_HINT}
> for HINT in "" "SOPBENCH_GETTER_HINT=1"; do
>  for A in "" "SOPBENCH_ALIAS=1"; do for S in "SOPBENCH_SOURCE=1" "SOPBENCH_SOURCE=3"; do
>   env $HINT $A $S SOPBENCH_VLLM_BASE_URL=http://localhost:9100/v1 \
>     <seka_env>/bin/python run_simulation.py --domain bank --assistant_model qwen2.5-32b-instruct \
>     --two_stage --two_stage_v2 --tool_list full --output_dir ./out_32b_${HINT}_${A}_${S}   # +v2-planner 플래그
>   <seka_env>/bin/python run_evaluation.py --domain bank --assistant_model qwen2.5-32b-instruct \
>     --tool_list full --output_dir ./out_32b_${HINT}_${A}_${S}
>  done; done
> done
> ```
> ⚠️ **STEP2 `apply_two_stage_patch.py <clone>` 재실행 필수**(갱신된 `two_stage_client.py`를 clone에 재배포해야 getter-hint 작동). `induced/getter_map.json`은 `autoderive_getter_map.py`로 생성(rung1 파이프라인이 자동). 헤드라인 지표(1번 항목)로 판독. 결과 → `SOPBENCH_EXPERIMENT_RESULTS.md` Exp-4* 행.
>
> ### ★★★★★ v1.35 (2026-06-01) — ★coworker 32B = **우리 arm(구조+alias) vs leaderboard 바닐라** (바닐라 재측정 금지)
> **무게중심 업데이트**: should_T 병목 해소의 (b) 구현이 **tool-name ALIAS 마스킹 + source-3(NL-only)** 로 재정렬됐다
> (설계 권위본 = `scripts/distill/TASK_CONSTRAINT_DESIGN.md` **§8.5.★**). **coworker의 역할 = 32B로 *우리 arm*만**
> (구조 planner + alias on/off + source3, end-to-end bank). **★32B/72B 바닐라(arm-1)는 leaderboard 인용 — 재측정 금지**
> (bank: Qwen32B **40.30** / Qwen72B **35.07**, ReAct/full = 아래 공식표). Track A(7B) = 같은 regime을 SFT 학습·eval(진행 중).
>
> **왜 leaderboard로 안 되고 우리 arm을 32B로 돌려야 하나 (핵심):**
> - leaderboard 40.30 = **바닐라 단일 LLM·실제 도구이름·풀 롤아웃** = "**이름을 다 보여줄 때** 32B 천장". 우리 regime은
>   **이름을 지운다(alias)** → leaderboard엔 alias 세팅이 없어 답할 수 없다. 우리가 알 것 = **강모델에서 이름을 지우면
>   얼마나 떨어지나**(= alias on/off Δ). leaderboard 40.30 = with-names 앵커, 우리 alias 런 = without-names 측정.
> - **alias 마스킹** = LODO 전이 헤드라인의 **타당성 게이트**: 도구명이 암기가능하면(`login→apply_credit_card`) 양성 전이가
>   "이름 암기"로 오염. per-task 불투명 alias(`op_7`)면 모델이 **NL 정책↔도구 설명 의미매칭**을 강제당함 = thesis 스킬 그 자체.
>   (그래프 전체 일관 alias: 술어·STATUS내 체크명·설명·history까지. 선두 이름만 가리면 needs[]/STATUS로 샌다.)
> - **source-3** = 제약-도출 STATUS('정답지') 미렌더, 도구는 **설명 + NL 정책만**. alias와 **직교축**: 진짜 anti-cheat =
>   **alias ON + source3**(alias+source1은 익명화된 dirgraph를 여전히 떠먹임).
>
> ### ▶ coworker 실험 = 우리 arm을 32B로 (end-to-end, 실제 evaluator)
> **★32B는 우리 7B SFT 어댑터를 못 받는다**(LoRA=7B 전용) → 32B는 **in-context 구조 planner**(arm-3v2, zero-shot)로 돌린다.
> 같은 `run_simulation.py --two_stage`(arm-3v2/arm-4a 레시피, `SOPBENCH_EXPERIMENT_RESULTS.md` Exp-3v2/4a) → `run_evaluation.py`,
> bank, should_T 48 / should_F 86. **추가 = `SOPBENCH_ALIAS`/`SOPBENCH_SOURCE` env 토글만** (TwoStageClient가 읽음):
> ```bash
> # 32B 서빙은 v1.34 STEP3 그대로(port 9100, served-model-name qwen2.5-32b-instruct).
> # arm-3v2 셀 4개(alias×source) — 나머지 인자는 arm-3v2/arm-4a 표준과 동일:
> for ENV in "" "SOPBENCH_SOURCE=3" "SOPBENCH_ALIAS=1" "SOPBENCH_ALIAS=1 SOPBENCH_SOURCE=3"; do
>   env $ENV SOPBENCH_VLLM_BASE_URL=http://localhost:9100/v1 \
>     <seka_env>/bin/python run_simulation.py --domain bank --assistant_model qwen2.5-32b-instruct \
>     --two_stage --tool_list full --max_num_turns 20 --output_dir ./out_32b_$ENV   # +arm-3v2 v2-planner 플래그
>   <seka_env>/bin/python run_evaluation.py --domain bank --assistant_model qwen2.5-32b-instruct \
>     --tool_list full --output_dir ./out_32b_$ENV
> done
> ```
> > 정확한 v2-planner 활성 플래그/abox 경로는 arm-3v2/arm-4a 실행 레시피(Exp-3v2/4a)와 동일하게 맞춘다(Track A가 7B로 검증한 그 호출).
>
> **비교 앵커**: 32B leaderboard arm-1 **40.30**(바닐라, 이름有) · 7B arm-4a **26.1**(학습) · 7B arm-3v2(in-context).
> **핵심 측정 = 32B의 alias on/off Δ** = 강모델에서 anti-cheat 비용 + "강모델이 구조+NL만으로 절차를 도출하는가".
>
> **(선택·싼 proxy) teacher-forced 게이트** `gate_alias.py` — 풀 롤아웃 전 빠른 사전점검(SFT 프롬프트에 32B 질의, next-op
> 정확도). 단 **어차피 32B를 쓰면 위 end-to-end가 정보량이 더 크다**(leaderboard와 직접 비교 가능). proxy로만 사용.
>
> ### ▶ G9* 판정 (재정의 — end-to-end)
> **32B alias-ON arm이 (a) alias-OFF arm 대비 큰 붕괴 없이 유지되고 (b) leaderboard 40.30 대비 합리적**이면 = alias+source3
> regime이 강모델에서 성립 → 7B SFT 학습가치 확정. **alias-ON에서 should_T가 바닥**이면 = 도구 설명·정책 신호 부족 →
> 재학습 전 **설명/정책 블록 보강**(목표 정책을 600자 truncate 말고 goal 관련 블록 전달, §8.5.★). (proxy 기준: teacher-forced
> alias+s3 ≥0.6.)
>
> ### ▶ 게이트 통과 시 → 32B/72B alias 매트릭스 (coworker 본 기여)
> **헤드라인 실험**(설계 §8.5.★ ④): ① **LODO 전이**(6도메인 학습→held-out bank, ABox swap, 재학습0 — 학습은 Track A 7B가
> 주, 32B는 in-context 전이) ② **ablation**: 빈/틀린 ABox→붕괴(온톨로지 실사용) · in-context vs L0(arm-2) · **alias on/off**
> ③ 멀티턴 user_sim pass@1. **SOTA 절대수치보다 "재학습0 전이"가 1급 결과.** (bank held-out은 §10 결함 8개 제외 분모.)
>
> ### ▶ 파일 레퍼런스 (repo `scripts/distill/sopbench/`)
> | 파일 | 역할 |
> |---|---|
> | `two_stage_client.py` | `build_v2_prompt(alias_map, source)` + `make_alias_map`(전그래프 일관 bijection) + `_plan_v2` de-alias. **env `SOPBENCH_ALIAS`/`SOPBENCH_SOURCE` 토글**(32B in-context 런의 alias on/off 스위치). 기본경로(off·s1)=레거시 byte-identical(검증). |
> | `build_tbox_planner_sft.py` | `--alias`/`--source {1,3}` → per-task TRAIN-salt alias map + 타깃 alias(7B SFT 데이터용). |
> | `gate_alias.py` | (선택) offline teacher-forced next-op 정확도 proxy. SOPBench env 無, endpoint만. |
> > **train salt ≠ eval salt (의도)**: build `train|<dom>|<task#>` / client `eval|<goal>|<toolset>` → alias 값이 train/eval
> > 달라야 "alias↔tool 암기 아님=의미매칭만 전이" 입증. 프롬프트 *형식*은 동일(§6.4). ⚠️ rr.ps1 1호출·로컬 python=스텁
> > (seka_env py3.12)·실제 RC 확인 후만 인용(§10.4).
>
> ---
>
> ### ★★★★ v1.34 (2026-06-01 밤) — arm-3 파이프라인 완성·검증 + 32B/72B TURNKEY 실행 지시 (coworker 내일 바로)
> **TL;DR**: arm-3(L1 2-stage agent) 파이프라인이 **완성·디버그·검증**되었다(7B bank N=134 저자 evaluator로 채점 완료).
> coworker는 **`MODEL` 한 변수만 바꿔** 32B/72B를 7도메인 sweep 가능. 모든 코드·스크립트·서빙레시피 아래 박제.
>
> **⚠️ 7B 결과 먼저 보고 기대치 보정 (중요):** arm-3-**naive**(operator 이름+desc만 보는 planner + 매턴 강제
> 도구호출 resolver)는 7B bank에서 **pass@1 = 0.0%** (arm-1 fc/full 3.7%·react/full 5.2%보다 **나쁨**).
> 실패의 ~90%가 **제약 순서 위반**(constraint/dirgraph violation): agent가 SOP 선행검증 없이 타깃 액션을 즉시
> 호출. → **순진한 L1은 음성이지만, 이것이 "구조적 planner(의존성 그래프 주입)"의 필요를 깨끗이 증명**한다.
> 상세 = `SOPBENCH_EXPERIMENT_RESULTS.md` **Exp-3**. coworker의 32B/72B arm-3-naive run의 목적 =
> **모델크기 × 구조 상호작용 측정**(강한 모델이 강제호출 패널티를 흡수하는지) + 구조판 v2의 비교 baseline 확보.
> **수치가 낮아도 정상** — error_statistics 분해(어느 실패가 지배적인지)가 핵심 산출물.
>
> ### ▶ TURNKEY 실행 (4 스텝, `MODEL`만 변경)
>
> **STEP 1 — 코드 받기 (repo pull):**
> ```bash
> cd <your boltzmann-attention checkout>     # branch facet-rft-2026
> git pull --rebase origin facet-rft-2026
> # 신규 파일: scripts/distill/sopbench/{two_stage_client.py, apply_two_stage_patch.py, run_arm3_sweep.sh}
> ```
>
> **STEP 2 — SOPBench 클론 + 패치 1회 (단일 명령, 멱등·.bak 백업):**
> ```bash
> git clone https://github.com/Leezekun/SOPBench.git ~/SOPBench   # 또는 기존 클론 경로
> python <repo>/scripts/distill/sopbench/apply_two_stage_patch.py ~/SOPBench
> #  → cp two_stage_client.py + run_simulation.py(--two_stage) + types.py(client 완화)
> #    + llm_handler.py(SOPBENCH_VLLM_BASE_URL endpoint) + constants.py(OSS fc 등록) 전부 적용
> # env: python3.10+ (py3.12 권장), pip install openai tqdm termcolor colorama pydantic  (vllm 모듈 불요·CLI만)
> ```
>
> **STEP 3 — 모델 서빙 (vLLM, `--served-model-name`을 짧은 표준명으로 — 이게 turnkey 핵심):**
> ```bash
> # 32B (1×80GB로 충분; 여유위해 TP=2도 가능). served-model-name = constants에 등록된 짧은 id.
> CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen2.5-32B-Instruct \
>   --served-model-name qwen2.5-32b-instruct \
>   --port 9100 --dtype bfloat16 --gpu-memory-utilization 0.90 --max-model-len 32000 \
>   --enable-auto-tool-choice --tool-call-parser hermes --trust-remote-code
>
> # 72B (TP=2, 2×80GB):
> CUDA_VISIBLE_DEVICES=0,1 vllm serve Qwen/Qwen2.5-72B-Instruct \
>   --served-model-name qwen2.5-72b-instruct \
>   --tensor-parallel-size 2 \
>   --port 9000 --dtype bfloat16 --gpu-memory-utilization 0.92 --max-model-len 32000 \
>   --enable-auto-tool-choice --tool-call-parser hermes --trust-remote-code
> ```
> > **왜 `--served-model-name` 짧은 id가 핵심**: arm-1 fc는 `model_name ∈ FUNCTION_CALLING_MODELS["vllm"]`
> > (=`qwen2.5-32b-instruct` 등 짧은명) assert를 통과해야 하고, arm-3·arm-1 모두 OpenAI API에 `model=<MODEL>`로
> > 호출한다. 서빙명을 짧은 표준명으로 맞추면 **두 arm이 동일 `--assistant_model <MODEL>`로 작동**(이름 충돌 0).
>
> **STEP 4 — sweep 실행 (`MODEL`만 변경):**
> ```bash
> cd ~/SOPBench
> # 32B 전체(arm-1 + arm-3, 7도메인, fc/full):
> MODEL=qwen2.5-32b-instruct SOPBENCH_VLLM_BASE_URL=http://localhost:9100/v1 \
>   PY=$(which python) bash <repo>/scripts/distill/sopbench/run_arm3_sweep.sh
>
> # 72B (다른 endpoint):
> MODEL=qwen2.5-72b-instruct SOPBENCH_VLLM_BASE_URL=http://localhost:9000/v1 \
>   PY=$(which python) bash <repo>/scripts/distill/sopbench/run_arm3_sweep.sh
>
> # 빠른 smoke 먼저 권장(각 5태스크): ... NUM_TASKS=5 DOMAINS="bank dmv" ARMS="arm3" ... 추가
> ```
> 끝나면 `output_coworker/summary_<MODEL>.tsv` + 콘솔에 `arm/mode/tool/domain/pass_rate/action_called` 표 출력.
> arm별 결과 json = `output_coworker/{arm1,arm3}/<domain>/...json` (저자 포맷 그대로 → 재평가·궤적검수 가능).
>
> ### ▶ 무엇을 비교하나 (arm 정의)
> - **arm-1 (LLM-alone, baseline)**: 단일 LLM, full tool list. `run_simulation.py`(--two_stage 없음).
> - **arm-3 (L1 2-stage)**: planner(LLM, **operator 이름+desc만**, concrete schema 숨김=전이가드) → resolver
>   (`tool_choice` 강제 LLM 인자채움). `run_simulation.py --two_stage`. **planner+resolver = 매턴 2 LLM 호출.**
> - **공정비교**: `run_arm3_sweep.sh`는 arm-1·arm-3 **둘 다 같은 `fc/full`**로 돌려 **같은-모드 Δ**(arm3−arm1) 산출.
>   (resolver가 fc-native라 fc로 통일. `ARMS="arm1react"`로 leaderboard react/full 앵커도 추가 가능.)
> - **headline 판독**: domain별 arm3 vs arm1 + **error_statistics 분해**(constraint/dirgraph/db/action/toolcall).
>   7B에선 constraint/dirgraph가 ~90% 지배 → 32B/72B에서 이 분포가 어떻게 바뀌는지가 관전포인트.
>
> ### ▶ 결과 기록
> - 수치 → `SOPBENCH_EXPERIMENT_RESULTS.md` Exp-3 표에 `(arm, mode, model, domain, pass@1)` 행 추가.
> - 출력 서브트리: coworker = `reports/facet_rft_2026/phase4_distill/coworker_a100/` 하위(충돌회피). summary.tsv도 여기 commit.
> - arm-3 **설계 한계·v2 방향**(planner에 의존성그래프 주입·gate·exit허용·인자환각가드) = `WORKFLOW_ONTOLOGY_DESIGN §10`
>   + `SOPBENCH_EXPERIMENT_RESULTS.md` Exp-3 "다음" 절. v2 구현 전까지 coworker는 **arm-3-naive를 그대로** 돌릴 것.
>
> ### ▶ 파일 레퍼런스 (전부 repo `scripts/distill/sopbench/`)
> | 파일 | 역할 |
> |---|---|
> | `two_stage_client.py` | arm-3 정책(planner+resolver). `OpenAIHandler.inference` 호환 client. `use_deterministic_shortcut`(기본 off). |
> | `apply_two_stage_patch.py` | 클론에 5패치 멱등 배포(client cp + run_simulation `--two_stage` + types/llm_handler/constants). |
> | `run_arm3_sweep.sh` | **메인 진입점.** `MODEL`만 바꿔 arm-1+arm-3 × 7도메인 × eval × summary.tsv. |
> | `run_two_stage.py` | ⚠️DEPRECATED(구 인라인 eval 버그). 쓰지 말 것. |
>
> ### ▶ gotcha
> - `apply_two_stage_patch.py`는 멱등(이미 패치면 skip). 클론 업데이트 후 재실행 안전.
> - arm-3는 OpenAIHandler를 우회 → `--num_gpus/--gpu_memory_utilization` 무시(이미 서빙된 endpoint 사용).
> - arm-1 fc는 `SOPBENCH_VLLM_BASE_URL` 설정 시 pre-served endpoint 사용(patch #4). 미설정 시 자체 vLLM spawn.
> - 강제 도구호출 resolver가 read-loop에서 긴 인자 환각 → max_tokens 절단 → JSON 에러 → run_simulation이 retry(최대5).
>   일부 태스크 empty-runs 가능하나 **분모에 실패로 정상 집계**(검증됨). 강한 모델에선 빈도 줄 것.
> - 72B fp16은 1×80GB에 안 들어감(~145GB) → **반드시 TP≥2**.
>
> ---
>
> ### ★★★ v1.33 (2026-06-01) — arm-1 baseline 완료 + 내일 실험 지시
> **결과 문서 (필독)**: `reports/facet_rft_2026/SOPBENCH_EXPERIMENT_RESULTS.md` — 가설·결과·해석·
> 다음 스텝 누적 기록. 이 문서가 앞으로 모든 SOPBench 실험의 결과 기록 권위본.
>
> **arm-1 baseline 확정 (Qwen2.5-7B, 7도메인, 2026-05-31):**
> | domain | fc/full | react/full (=leaderboard) | react/oracle |
> |---|--:|--:|--:|
> | bank | 3.7% | **5.2%** ✓ | 59.0% |
> | dmv | 11.3% | **21.7%** ✓ | 62.9% |
> | healthcare | 8.1% | **16.1%** ✓ | 34.7% |
> | hotel | 0.5% | **0.0%** ✓ | 16.9% |
> | library | 13.6% | **13.6%** ✓ | 47.0% |
> | online_market | 9.3% | **7.6%** ✓ | 43.0% |
> | university | 4.8% | **2.4%** ✓ | 38.1% |
> | **평균** | **7.3%** | **9.5%** | **43.1%** |
> → 공식 leaderboard Qwen-7B avg 9.7%와 일치. **harness 재현 검증 완료.**
> → **핵심 발견: oracle−full 격차가 압도적** (43% vs 9.5%). distractor 도구선택이 7B 주 병목.
>   FC vs ReAct는 full에서 미미 (7.3 vs 9.5%). 우리 표준 = **react/full** (leaderboard 정합).
>
> **★ 내일(2026-06-01) 즉시 할 실험 (Track B, coworker):**
> 1. **arm-0: oracle ceiling** — `scripts/distill/sopbench/workflow_executor.py` + `abox/` + SOPBench
>    custom agent interface. full+oracle 둘 다. ⚠️ should_succeed=False(거부) 처리 명시적 구현 필수.
>    bank 먼저 → 7도메인 확장. 목표: arm-1과 ceiling 사이 gap 정량화.
> 2. **arm-3: L1 2-stage agent** — `Agent.client` = planner(LLM, abstract operator in-context,
>    goal+추상 affordance만, concrete schema 금지) → resolver(b, ontollm). **full 표준 + bank부터.**
>    baseline(react/full 5.2%) vs arm-3(full) → ≥5%p 향상이면 구조 기여 확인 → 7도메인 확장.
>    코드: `scripts/distill/two_stage_agent.py`(tau2 기반, SOP-Bench 인터페이스로 재연결 필요).
> **표준 비교 규약**: 셀 표기 `(mode)/(tool_list)`. baseline arm = react/full. 진단 = react/oracle.
> 결과는 `SOPBENCH_EXPERIMENT_RESULTS.md` 대시보드에 채워 push.

> ### ★★ v1.32 (2026-05-31) — LLM-in-loop 재정립 + Tier-1 이중벤치 (이 배너가 무게중심을 고정)
> **★ Tier-1 이중벤치(이름 혼동 주의)**: **SOPBench**(Zekun Li, 하이픈無, arXiv **2503.08669**, 7도메인
> bank/dmv/healthcare/hotel/library/online_market/university, native 형식 operator
> `directed_action_graph`+constraints + rule oracle `env/evaluator.py`, LLM judge無) = **主, 파일럿=bank**.
> **SOP-Bench**(Amazon, 하이픈有, **2506.08119**, 12도메인, `abox/` 자산) = **보조**(breadth/2차 전이면).
> clone: `/home/woori/scratch/{SOPBench,SOP-Bench}`.
> **결정적 executor·resolver coverage% = oracle/상한 + 진단**이지 최종 기여가 아니다 (GT call-graph 보유→천장).
> **헤드라인 = 학습 도메인-일반 planner(TBox) + ABox-conditioned neural resolver(B5* xattn)의 전이** —
> N-1 도메인 학습→held-out ABox swap·**재학습 0**(SOPBench 7도메인 LODO 主, Amazon 12도메인 보조) +
> ABox-ablation 붕괴. 즉 **B5*(xattn)+B2* LODO 매트릭스+B1*(planner SFT)가 핵심**, 결정적 coverage(B2*
> rule 모드)는 진단축. 권위본 = `WORKFLOW_ONTOLOGY_DESIGN.md §9`.
> **⚠️ 본문 §1~§9의 tau2 도메인명(telecom/retail/airline)·`two_stage_agent.py`+`--user-llm`(user-sim)·
> pass^1·tau2 러너는 superseded substrate**. SOPBench(主)로 읽을 것: 도메인=7 SOPBench 도메인(bank 등),
> 러너=`swarm.Swarm`+`Agent(client=OpenAIHandler, functions=<d>_assistant.py:actions)`의 assistant를
> planner→resolver로 교체(`run_simulation.py --domain --assistant_model --tool_list{oracle,full}`),
> 지표 pass^1→**rule pass-rate(목표+constraint_not_violated+graph정합) + 거부정확도(action_should_succeed=false)
> + tool@scale(--tool_list full)**, 전이=7도메인 LODO(主)/Amazon 12도메인(보조). ⚠️`env/helpers.py`=Python≥3.10.

### ★★ SOPBench(Zekun Li) 공식 leaderboard + ★FC/ReAct 모드 주의 (coworker 필독, 2026-05-31)

**공식 pass rate(%) — 7도메인** (clone `README.md` `## Results`; pass@1=`success`=5체크 AND: no_tool_call_error ∧ constraint_not_violated ∧ database_match ∧ action_called_correctly ∧ dirgraph_satisfied. **LLM judge 0, 순수 rule oracle** `env/evaluator.py`). Avg*=7도메인 산술평균(공식표엔 없음, 우리 계산).

| Model (mode) | Bank | DMV | Health | Market | Univ | Library | Hotel | Avg* |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| GPT-5 (FC) | 71.64 | 84.54 | 76.61 | 69.77 | 88.10 | 66.67 | 67.18 | 74.9 |
| o4-mini-high (FC) | 76.87 | 83.51 | 92.74 | 89.53 | 95.24 | 34.85 | 55.90 | 75.5 |
| GPT-5-mini (FC) | 58.96 | 82.47 | 92.74 | 75.58 | 95.24 | 34.85 | 69.74 | 72.8 |
| Gemini-2.5-Flash (FC) | 67.91 | 81.44 | 87.90 | 77.91 | 83.33 | 51.52 | 42.56 | 70.4 |
| Deepseek-R1 (ReAct) | 54.48 | 81.44 | 54.03 | 70.41 | 76.19 | 54.55 | 50.77 | 63.1 |
| GPT-4.1 (FC) | 69.40 | 79.38 | 79.03 | 80.81 | 50.00 | 57.58 | 42.56 | 65.5 |
| GPT-4o (FC) | 58.96 | 80.41 | 73.39 | 61.63 | 66.67 | 60.61 | 39.49 | 63.0 |
| Claude-3-7-Sonnet (FC) | 65.67 | 70.10 | 70.97 | 56.98 | 66.67 | 27.27 | 23.59 | 54.5 |
| GPT-4.1-mini (FC) | 57.46 | 76.29 | 66.13 | 56.40 | 35.71 | 18.18 | 7.18 | 45.3 |
| Claude-3-5-Sonnet (FC) | 71.90 | 50.43 | 39.23 | 43.32 | 52.27 | 33.33 | 15.82 | 43.8 |
| GPT-4o-mini (FC) | 33.58 | 73.20 | 25.00 | 43.60 | 38.10 | 42.42 | 41.03 | 42.4 |
| Gemini-2.0-Flash (FC) | 52.99 | 51.55 | 21.77 | 38.37 | 30.95 | 19.70 | 7.18 | 31.8 |
| Llama3.1-70B-Instruct (ReAct) | 42.54 | 65.98 | 54.84 | 37.21 | 42.86 | 34.85 | 13.85 | 41.7 |
| Qwen2.5-32B-Instruct (ReAct) | 40.30 | 52.58 | 41.13 | 44.19 | 54.76 | 27.27 | 18.46 | 39.8 |
| Qwen2.5-72B-Instruct (ReAct) | 35.07 | 68.04 | 27.42 | 40.12 | 35.71 | 34.85 | 13.85 | 36.4 |
| Qwen2.5-14B-Instruct (ReAct) | 35.07 | 57.73 | 29.03 | 35.47 | 23.81 | 25.76 | 14.87 | 31.7 |
| Llama3.1-8B-Instruct (ReAct) | 14.93 | 18.56 | 20.16 | 16.28 | 23.81 | 30.30 | 0.00 | 17.7 |
| **Qwen2.5-7B-Instruct (ReAct)** | **5.22** | 20.62 | 16.94 | 9.30 | 0.00 | 15.15 | 0.51 | **9.7** |

**★ FC vs ReAct — baseline 정합의 핵심 주의:**
- **leaderboard는 모드가 섞여 있다**: **proprietary = FC**(`--tool_call_mode fc`, native function-calling), **open-source = ReAct**(`--tool_call_mode react`, 프롬프트기반 도구호출). 이유 = 원 repo의 `FUNCTION_CALLING_MODELS["vllm"]=[]`(빈 리스트) → OSS는 fc assert를 통과 못 해 저자들이 ReAct로 측정.
- **그래서 우리 baseline 모델 Qwen2.5-7B의 공식값은 ReAct, bank=5.22%/avg=9.7%** (약한모델 regime = 우리 2-stage 구조 향상 여지 최대).
- **⚠️ 우리 파일럿은 FC로 돌렸다**(Track A가 `constants.FUNCTION_CALLING_MODELS["vllm"]`에 qwen/llama 등록 + vLLM을 `--enable-auto-tool-choice --tool-call-parser hermes`로 서빙). **FC≠ReAct → 5.22%와 직접 비교 금지.** 모드를 명시하지 않고 leaderboard와 나란히 놓지 말 것.
- **★leaderboard 실제 실행설정 (확정, `scripts/simulation/<model>.sh` 실측)**: (1) **user_model 없음 = dummy user**(`user_known` 첫 메시지로 선제공, agent 단독) — **user-sim 아님**. user-sim(`--user_model gpt-4.1-mini`)은 `scripts/multi-turn/` **별도 실험**, `--user_model adv`(adversarial)는 Exp2 별도. (2) **OSS는 `--tool_call_mode react`**(qwen-7b 포함), proprietary는 `fc`. (3) **full·oracle 둘 다 실행**(Exp1 루프). **★★확정(Track A, 저자 공식 결과파일 재평가, 8모델 bank 전수 대조)**: **README leaderboard = `full` tool list, 예외 없음.** full값이 README 표값과 **정확히 일치**: GPT-5 .7164 / GPT-4o .5896 / GPT-4o-mini .3358 / Gemini-2.0 .5299 / Qwen72B .3507 / Qwen32B .4030 / Qwen14B .3507 / Llama8B .1493 / Qwen7B .0522 = 표값들. **oracle은 일관되게 훨씬 높음**(GPT-4o 59→77, Qwen32B 40→77, Qwen14B 35→65, Llama8B 15→41, Qwen7B 5→58). 7도메인 저자파일 모두 `output/<domain>/`에 존재. should_succeed split도 정상(게이밍 아님). **⇒ 우리 표준 = `full`**(leaderboard 정합). oracle은 폐기말고 **tool-selection@scale 진단축**으로 유지(oracle−full 격차 = distractor 도구선택 난이도 = 우리 planner 표적). 셀마다 `(mode)/(tool_list)` 표기 필수. (4) `--env_mode prompt`, `--num_run_per_interaction 1`. → **우리 dummy-user 설정은 leaderboard 표준과 일치**(이건 confound 아님); 남는 차이는 **모드(FC vs ReAct)와 tool_list(oracle vs full)** 둘뿐.
- **규약(coworker 매트릭스 전체에 적용)**: baseline·우리방법 **동일 tool_call_mode**로 셀을 채울 것. 권장 = **두 모드 다 보고**:
  - **ReAct 트랙**: leaderboard 정합·재현(Qwen-7B bank≈5.22% 출발점 고정). 우리 향상Δ를 공식값 위에 직접 얹어 보고.
  - **FC 트랙**: native function-calling을 우리 표준으로(현대적·tool_calls 구조화). baseline을 **우리가 직접 측정**해 같은 모드로 비교(공식표에 OSS-FC 값 없음 → 우리가 만든다).
  - 어느 경우든 셀 라벨에 `(FC)`/`(ReAct)` 표기 필수.

**baseline 재현 레시피 (Track A 검증완료, bank 파일럿):**
```bash
# 서빙: Qwen2.5-7B-Instruct, GPU0:9100, tool-calling 활성(FC용; ReAct만 할거면 tool-parser 불요)
CUDA_VISIBLE_DEVICES=0 <vllm0.11_env>/bin/vllm serve Qwen/Qwen2.5-7B-Instruct \
  --port 9100 --dtype bfloat16 --gpu-memory-utilization 0.85 --max-model-len 32000 \
  --enable-auto-tool-choice --tool-call-parser hermes --trust-remote-code
# 실행 env: seka_env(py3.12)+colorama/termcolor/anthropic로 SOPBench import OK(vllm 모듈 불요, CLI만)
# 패치2종(clone, *.py.bak 백업): (a) llm_handler._init_vllm가 env SOPBENCH_VLLM_BASE_URL 있으면 spawn 회피
#   (b) constants.FUNCTION_CALLING_MODELS["vllm"]에 qwen/llama 등록(FC assert 통과)
cd SOPBench
SOPBENCH_VLLM_BASE_URL=http://localhost:9100/v1 seka_env/bin/python run_simulation.py \
  --domain bank --assistant_model qwen2.5-7b-instruct --tool_list oracle \
  --tool_call_mode {fc|react} --max_num_turns 20 --output_dir ./output   # user_model 생략=dummy(user_known 통째 제공, agent단독)
seka_env/bin/python run_evaluation.py --domain bank --assistant_model qwen2.5-7b-instruct \
  --tool_list oracle --tool_call_mode {fc|react} --output_dir ./output    # → Mean Pass Rate + 오류분해
# bank=14 goals/134 instances(48 should_succeed=True / 86 False=거부축). num_tasks 생략=전체.
```

---

## 0.0 ★★★ 벤치마크 피벗 (2026-05-30 밤) — tau2 → SOP-Bench

tau2 재진단 결과 **tau2는 user_sim이 핵심실행, agent는 도구선택+대화지휘만** 하는 벤치 → 우리 워크플로우 온톨로지("에이전트가 SOP 전체를 자기 도구로 결정적 실행")와 구조적 mismatch(상세 §16.1). **벤치마크를 [SOP-Bench](https://github.com/amazon-science/SOP-Bench)(arXiv 2506.08119)로 이전.** 에이전트단독·single-shot(대화없음→max_steps/read-loop 실패모드 소멸)·SOP의 명시적 If-분기(=우리 trigger/branch)·12 독립도메인·정답 SOP 제공.

**개념(§15)은 전부 유지, substrate만 교체.** 아래 §1~§9의 tau2 도메인(telecom/retail/airline)·two_stage_agent·coverage 수치는 **개념 참조용(superseded substrate)**. SOP-Bench 재정의:

| 옛 (tau2) | 새 (SOP-Bench) |
|---|---|
| telecom/retail/airline 3도메인 | 12 산업도메인(customer_service/content_flagging/... ) |
| two_stage_agent.py + tau2 run | `sop_bench_loader.py` + SOP-Bench custom agent(`agents/base.py`) |
| 결정적 coverage% + pass^1(대화) | **TSR/ECR/C-TSR + Tool Accuracy**(상태기반) + per-phase coverage |
| obs_triggers/step_realization induce | `compile_sop_ontology.py`(sop.txt 직접) + `induce_ontology.py`(궤적) |
| user_sim(OpenRouter gpt-4.1) | **불요**(에이전트 단독, user 없음) → OpenRouter 비용·judge 이슈 소멸 |
| LODO 3도메인 swap | **12도메인 ABox-swap** 전이 |

**★phased plan + 특허 분리 (2026-05-31, 상세 EXPERIMENT_DESIGN §16.8 / WORKFLOW_ONTOLOGY_DESIGN §6)**:
- **목표 격상(agentic)**: "주어진 SOP 실행"→"목표만 주면 자동 도구선택". **TBox=학습 일반 planner / ABox=도메인 operator(precondition/produces/arg/achieves)**. 온톨로지는 **8 단순 call-graph 관계**(복잡 compute/decide는 함수 안). planner 3단 L0(symbolic)/L1(LLM+operator)/L2(학습 ABox-conditioned=§15.13).
- **Phase 1=SOP-Bench로 전이 정량증명**(1a executor 상한+induced↔sop.txt / **1b ★goal-only L2 N-1학습→held-out operator swap·재학습0**), **Phase 2=AppWorld**(자율·절차없음)로 확장.
- **★특허와 별개 트랙**: 본 라인=계획·전이(SOP-Bench/AppWorld). 특허=중복·다면 도구 컨텍스트 **선택**+내재화(MetaTool/ToolBench/τ²-bench). 독립 평가·보고.

### ★ Coworker 태스크 재정의 (SOP-Bench)
- **C1*. SOP→온톨로지 컴파일러 + executor** (Track A와 공동, P0): `compile_sop_ontology.py`(sop.txt→ontology) + `workflow_executor.py`(순수 agent-tool, user-side 無) → SOP-Bench custom agent. **파일럿 customer_service 1도메인 end-to-end 먼저.**
- **C2*. ★대규모 매트릭스** (coworker 핵심, A100×4 fan-out): {Baseline FC/ReAct, Ours-P0-compiled, Ours-induced} × **12 도메인** × {in-domain, ABox-swap LODO}. 셀=TSR/ToolAcc/per-phase coverage. **SOP-Bench는 single-shot이라 user_sim 비용 0 → 대량 병렬 저렴.**
- **C3*. Induce vs 작성SOP 검증**: `induce_ontology.py`로 궤적에서 온톨로지 유도 → sop.txt와 구조일치도 + TSR(=§15.14 자동induce thesis의 ground-truth 검증, tau2 불가했던 것).
- **C4*. (조건부) Neural ABox resolver (xattn, §15.13)**: SOP 분기가 결정적이지 않은(생성형·모호) step에서 rule이 못 메우는 잔차를 학습 resolver가 메우나. ABox 메모리=SOP 관계. Baseline rule-coverage gap이 클 때만 진입.
- **C5*. (조건부) Phase-planner SFT (P1)**: P0가 phase 시퀀싱에서 LLM 필요할 때만. SOP-Bench 궤적에서 phase 라벨 SFT. 32B.
- **SKIP**: tau2 32B abstract-SFT(구 B1*), tau2 two_stage 매트릭스(구 B2*), GRPO(구 B4*)는 **substrate 폐기**. 개념(Group J·Routine R1-R4)은 SOP 분기/슬롯 컴파일로 흡수.

---

## 0. ★ (참고·구 substrate) 방향 전환 (2026-05-31) — 단순 distillation → 다층 온톨로지 에이전트

> ⚠️ 이하 §0~§9는 **tau2 기반 구 계획**. 개념은 §0.0 표대로 SOP-Bench에 re-map됨. 운영 세부(도메인·러너·user_sim)는 SOP-Bench 재정의로 대체.

이전 계획(B1–B4: 32B plain/facet × full/none 4조합 SFT 매트릭스)은 **단순 goal→tool distillation** 검증용이었다. 그 thesis는 **7B에서 이미 검증 완료**(아래 §1) → coworker는 그 중간단계를 **재현하지 않는다**. 대신 그 위에 올린 **layered hierarchical agent**(planner=TBox + 결정적 온톨로지 executor + LLM fallback)를 **A100×4 우위로 대규모 검증**한다.

**핵심 구조** (§15.11):
```
  PLANNER (LLM, 학습)  : 추상 PLAN_STEP만 emit ("Plan: apply_targeted_fix") — 구체 도구 안 봄 = 순수 TBox
        │
        ▼
  EXECUTOR (결정적, 무학습) : step + 관찰상태 → 구체 (tool, args)
        = step_realizes_tool⁻¹(후보) × observation_triggers(상태매칭) × arg_source(인자 provenance)
        │  miss(후보 모호)
        ▼
  LLM FALLBACK : 후보-제한 LLM이 도구 선택
```
- **전이 = 온톨로지 파일만 swap** (planner·PLAN_STEP vocab 불변 = TBox / `obs_triggers_<dom>.json`·`step_realization_<dom>.json` = ABox).
- **헤드라인 지표 = 결정적 coverage%** (온톨로지만으로 해결된 도구선택 비율). 나머지는 LLM fallback.
- **이미 구현 완료**(Track A, repo): `scripts/distill/ontology_resolver.py`(결정적 executor) + `scripts/distill/two_stage_agent.py`(planner+resolver+fallback wrap, 4 ablation mode). 필드·API 검증 완료.

---

## 0.1 ★★ 7B에서 검증 완료 → coworker가 SKIP 할 것 (재현 불필요)

| 검증된 것 (7B) | 결과 | → coworker 스킵 |
|---|---|---|
| **efficiency thesis** (NONE 정책제거 ≥ FULL 정책유지) | 3도메인 모두 NONE≥FULL (telecom .35/.30 · retail .77/.64 · airline .40/.30) | **32B full-arm 전부 스킵.** none(내부화)만 학습 |
| **goal→tool distillation 작동** (base≈0 → SFT 상승) | F1/seq_F1 AUC 0.902, base recall~0.04→student↑ | **단순 SFT-lift 재현(구 G1) 스킵.** base vs plain-SFT 매트릭스 불필요 |
| **plain vs facet** | facet 별이득 불명, 7B는 plain만으로 thesis 입증 | **facet-SFT arm(구 G3) 폐기.** plain/abstract만 |
| **F1/seq_F1·arg_bind 지표, GRPO dense reward 설계** | scorecard·grpo_reward.py 검증 | **지표/reward 재설계 불필요.** 그대로 사용 |
| **scorecard 3도메인 일반화** | read-제외·requestor 정정 후 +0.2~0.6 | **scorer 재검증 스킵** |

= 4조합(plain/facet × full/none) × 3도메인 = 12셀 SFT 매트릭스 → **abstract-none 32B 1개 + 큰모델 fallback probe로 축소.** "큰 모델로 같은 thesis 재확인"이 아니라 **큰 모델이 아니면 못 보는 것**(아래 §3)만 한다.

---

## 1. 배경 (7B 기확립, 요약)
- tau2-bench 도구 에이전트. 큰/작은 격차 = **94% 절차(distillable)**, capability 벽 1%. goal→tool이 success/failure 최강 변별.
- NONE 실패 63% = **recall-miss / anti-loop**(정책 없이 진단만 반복, fix commit 안 함). = 결정적 executor + harness 재시도가 직접 겨냥하는 실패모드.
- **★결정적 executor의 경계 (7B inducer 분석, §15.11 v3)**:
  - **telecom류 상태머신 = 결정적 작동.** `enable_roaming ⇐ roaming_enabled=False`(prec 0.81) · `resume_line ⇐ status=Paid`(0.98) · `send_payment_request ⇐ bill.status=Draft`(0.85). 관찰상태가 도구를 일의적으로 결정.
  - **retail/airline 카탈로그-선택형 = 과적합 → LLM fallback 필요.** variant.size=S·destination=PHX 등 인스턴스-특수 trigger(소표본). 결정적 안 됨.
  - → **coverage%가 곧 "온톨로지가 어디까지 일하나"의 정량 경계.** 이게 이 트랙의 핵심 결과물.

---

## 2. 가설 & 지표

**Thesis (layered)**: (H1) goal→tool 절차를 **추상 PLAN_STEP planner(TBox)** + **결정적 온톨로지 executor(ABox)**로 분해하면, (H2) executor가 상태머신 도메인에서 **높은 결정적 coverage**로 도구를 LLM 없이 선택하고, (H3) **온톨로지 파일 swap만으로 held-out 도메인 전이**(planner 재학습 0)하며, (H4) 큰 모델은 **결정적이 못 푸는 catalogue-선택 잔차의 LLM-fallback 품질**에서 우위를 보인다.

**지표**
- **★결정적 coverage%** (headline): `deterministic / tool_call_turns`. `two_stage_agent.py`가 `coverage_<mode>_<set>_<split>.json`에 자동 저장(by_step 분해 포함).
- **pass^1** (tau2 test split) — 모드별·도메인별 최종 성능.
- **F1 / seq_F1 / arg_bind** (scorecard, 기존) — goal→tool 품질.
- **transfer Δ**: in-distribution vs ontology-swap LODO의 pass^1·coverage 차이.
- **efficiency**: none-arm 토큰/KV 절감 (7B에서 입증, 32B는 확인만).

**Ablation 6모드** (`two_stage_agent.py --mode`, §15.11 + §15.13):
| mode | 구성 | 측정 의미 | 상태 |
|---|---|---|---|
| `base` | 전체도구 LLM (planner/resolver 無) | 하한 baseline | done |
| `resolver` | planner step + 결정적 rule resolver(ABox=dict), miss시 planner 자기콜 | **순수 결정적 coverage** (LLM 추가 0) | done(§15.11) |
| `ontollm` | ABox를 프롬프트로 직렬화, LLM in-context 선택 | 프롬프트 천장 (토큰 비쌈) | 신규(b) |
| `xattn` | **ABox=cross-attn 메모리, TBox=학습 weights** | **★본 트랙 novelty**(토큰0+학습 유연성) | 신규(c)·B5* |
| `fallback` | rule resolver + miss시 후보제한 LLM | 결정적+fallback 조합 (실사용) | done(§15.11) |
| `monolithic` | abstract 모델이 Plan+구체콜 end-to-end (resolver bypass) | planner 단독 상한 | done(§15.11) |

→ 사다리 판독: `base→ontollm`=온톨로지 프롬프트-side 기여 / `resolver→xattn`=rule이 못 푼 걸 학습 attention이 메운 양(catalogue 도메인서 격차 클 것) / `ontollm vs xattn`=프롬프트 vs weights(토큰·전이·정확도) / `xattn vs monolithic`=ABox-conditioning 이득.

---

## 3. Coworker 태스크 (A100×4, layered 중심)

### B1*. 32B abstract-planner SFT  ★우선 (none-only, 1 arm)
- **목적**: 큰 planner가 추상 PLAN_STEP을 7B보다 정확히 emit → 결정적 executor의 입력 품질↑.
- **데이터**: `reports/facet_rft_2026/phase4_distill/sft_data/sft_abstract_train_all.jsonl` (Plan-step prefix 주입본, repo). **plain/facet/full 변형 없음** — abstract-none 단일.
- **trainer**: `scripts/distill/lora_train_chat_toolcall.py --system-mode none --max-seq-len 8192` (7B 레시피 그대로, 32B만 교체). base=`Qwen/Qwen2.5-32B-Instruct`.
- **GPU**: 1× 80GB (LoRA r16 + grad-ckpt) 또는 2× FSDP.
- **출력**: adapter `lora_adapters/qwen32b_abstract_none/` → HF private. (구 4조합 매트릭스 폐기.)
- **성공기준**: 수렴 + monolithic 모드 pass^1 ≥ 7B abstract.

### B2*. two_stage_agent ablation × LODO 매트릭스  ★★대규모 병렬 (coworker 핵심 기여)
- **목적**: layered agent의 결정적 coverage + 전이를 **모드·도메인·온톨로지 전부** 채운다. A100×4 fan-out 최적.
- **러너**: `scripts/distill/two_stage_agent.py` (repo). 모드별 호출:
  ```
  python scripts/distill/two_stage_agent.py --mode {base|resolver|ontollm|xattn|fallback|monolithic} \
    --domain <dom> --task-set <dom> --task-split test \
    --agent-llm openai/<served-lora> --base-url http://127.0.0.1:<port>/v1 --agent-api-key sk-noauth \
    --user-llm openai/openai/gpt-4.1 --user-base-url https://openrouter.ai/api/v1 --user-api-key $OPENROUTER_API_KEY
  ```
- **매트릭스** (각 셀 = pass^1 + 결정적 coverage%):
  - **{7B-abstract, 32B-abstract}** × **6 modes**(base/resolver/ontollm/xattn/fallback/monolithic) × **{telecom, retail, airline} in-distribution test**. (xattn 모드는 B5* 학습 완료 후 추가.)
  - **★LODO ontology-swap**: planner 불변, `--ontology-domain <other>`로 ABox만 교체. 핵심 셀 = **telecom planner + airline ontology**(전이) vs **airline planner + airline ontology**(in-dist) — coverage·pass^1 격차로 "온톨로지만 swap해도 전이되나"(H3) 판정.
  - **ABox-swap sanity**: `--ontology-domain`을 틀린 도메인으로 주면 coverage가 무너져야 함(온톨로지가 실제로 일한다는 음성대조).
- **예상 결과**(7B 기준 외삽): telecom = `resolver` 높은 coverage(상태머신), retail/airline = `resolver` 낮음 → `fallback` 비중↑. **coverage%가 도메인별로 갈리는 곡선이 메인 figure.**
- **출력**: `reports/facet_rft_2026/phase4_distill/coworker_a100/two_stage/<run>/` — results.json + coverage_*.json manifest.

### B3*. Capability-ceiling = fallback 품질 probe  (B4 재정의)
- **목적**: 결정적 executor가 **못 푸는 잔차**(retail/airline catalogue-선택, coverage의 1−x)를 **큰 모델 LLM-fallback이 메우는가**. = capability가 도움되는 지점을 layered 구조 안에서 정확히 격리.
- **방법**: `--mode fallback` 고정, fallback LLM만 {7B, 32B, 70B} 교체 → miss-turn에서의 도구선택 정확도·pass^1 비교. (planner·resolver 동일.)
- **GPU**: 70B = 2× A100(AWQ) / 4× bf16. (Qwen2.5-72B / Llama-3.3-70B, Track A 다운로드본 HF 공유.)
- **출력**: fallback-모델 크기 함수의 catalogue-도메인 pass^1 → "결정적이 못 푸는 부분 = capability냐 절차냐" 결론.

### B4*. (조건부) On-policy GRPO with Group J reward
- **선행조건**: B2* 결과 layered가 base 대비 양성(coverage 의미 + pass^1 ≥ baseline)일 때만.
- **reward**: `scripts/distill/grpo_reward.py`(검증) + **Group J 항**(repairs_state recall, distractor penalty, step penalty=anti-loop). 정책 init=B1* abstract-none 어댑터. planner의 step-emit에 dense reward.
- **trl**: seka_env(transformers 4.51.3) 충돌 → coworker는 **trl 호환 별도 venv** 권장(transformers 버전 맞춤). 안 되면 수동 GRPO 루프(Track A 방식).
- **출력**: GRPO adapter + reward curve + B2* 매트릭스 갱신.

### B5*. ★Neural ABox-conditioned resolver (cross-attn) — 본 트랙 novelty (design §15.13, v1.29)
- **목적**: §15.11의 결정적 rule resolver(코드+dict)를 **학습된 neural resolver**로 일반화. **TBox=ABox를 읽어 도구·인자를 고르는 절차(cross-attn weights, 도메인무관 고정) / ABox=온톨로지 관계 메모리(도메인별 swap)**. rule이 못 푸는 catalogue-선택형(retail/airline)까지 coverage 천장을 올리고, **온톨로지 메모리 swap만으로 전이**.
- **왜 coworker(A100)**: 아키텍처 수술(base에 cross-attn block 삽입)+학습이 무거움. 7B는 (a)rule+(b)프롬프트 baseline·gap 정량화(Track A), **(c)xattn 학습·매트릭스는 B 트랙**.
- **아키텍처(우선 C-1)**: ABox 관계를 자연어 직렬화→frozen 텍스트 인코더→메모리 M={e_1..e_N}(도메인별). executor hidden h_t(관찰+abstract step)=Query → `cross_attn(Q=h_t,K=V=M)` → head가 (tool,args) emit. **학습=cross-attn W_Q/K/V+readout=TBox / swap=M=ABox**. 토큰0(프롬프트 아님). 대안: C-2 hypernet→ABox-LoRA(공유 TBox-LoRA + per-domain ABox-LoRA), C-3 graph encoder.
- **학습**: teacher SUCCESS 궤적(telret 등). 입력=관찰상태+planner의 abstract step, 타깃=GT(tool,args). planner(B1* abstract-none 어댑터)는 freeze 권장. **ABox 인코더는 도메인무관 텍스트 인코더**(관계→자연어→frozen embed) — swap 도메인 M이 학습분포와 같은 의미슬롯이어야 전이(★최난점).
- **eval/ablation**:
  - **`two_stage_agent --mode xattn`** 신규 → B2* 매트릭스에 모드 1개 추가({7B,32B}×{base,resolver,ontollm,**xattn**,fallback,monolithic}×3도메인×{in-dist,swap}).
  - **ABox-memory swap LODO**: TBox(cross-attn weights) 불변, M_telecom→M_airline 교체만으로 held-out airline 작동?
  - **★ABox-ablation(검증가능성)**: 빈 M / 틀린 도메인 M 주입 시 성능 **붕괴**해야 "온톨로지가 실제로 일한다" 입증(attention이 ABox 무시·암기 아님). attention map으로 "어느 관계 읽었나" 해석.
- **선행조건**: Track A가 (a)resolver+(b)ontollm baseline으로 retail/airline rule-coverage gap을 정량화(=xattn이 메울 표적). gap이 크면 B5* 메인 기여로 진입.
- **출력**: `ontology_encoder.py`(관계→메모리) + cross-attn executor block + 학습된 TBox weights(HF) + 도메인별 ABox 메모리 + coverage/swap/ablation manifest (`coworker_a100/xattn/`).
- **리스크**: telret~1300 데이터로 cross-attn 신규 파라미터 학습 충분한지(→증강/32B), ABox 인코딩 분포정합(전이 핵심·최난점), 구현 복잡도((a)(b)보다 훨씬 무거움).

### B6*. ★Routine-derived layers — scenario/branch/placeholder 자동 induce (design §15.14, v1.30)
Routine(2507.14447) 4 메커니즘을 **자동 induce + 다층 executor**로 일반화. 전부 기존 induced 맵에서 추출(새 데이터 0). **우선순위 R4 > R3 > R1/R2.**
- **R4 scenario(★최대 레버, 3도메인)**: `induce_scenario_workflow.py` — fault-유형 클러스터=scenario(`fault_fix_map` 키), scenario별 workflow DAG, 초기 read→fault 시그니처 결정적 매칭. **planner 2단계(task→scenario→step)** + multi-fault 합집합 활성화(NONE 누락 직격). ABox swap·xattn 메모리를 scenario 슬롯으로.
- **R3 branch(telecom 결정적)**: `induce_branch_dag.py` — 같은 step 후 분기점+직전 read 대조 → `exclusive_choice(step,[(cond,tool)])`. 재료=observation_triggers∪distractor_for∪escalate_when(완료). mutual-exclusion + else→escalate(anti-loop 차단).
- **R1/R2 placeholder(arg_bind 계약)**: `induce_variable_slots.py` — step input/output 슬롯을 `ObservedState.by_source`(런타임 variable memory) key로 강제 채움. 빈 슬롯→miss→fallback. **인자 할루시네이션 구조적 불가**(arg_bind 0.32→계약).
- **eval**: 각 층 결정적 coverage% + marginal pass^1 + **multi-fault 누락 감소(R4)** + arg_bind 향상(R1/R2) + anti-loop 감소(R3). telecom 결정적 / retail·airline neural(→B5* xattn) 경계.
- **담당**: induce·결정적 검증=Track A(7B). neural scenario/branch(xattn 메모리에 scenario·분기 슬롯)=coworker. scenario-conditioned planner SFT는 B1* 데이터에 scenario 라벨 추가로 흡수 가능.
- **출력**: 3 inducer + scenario/branch/variable 맵(`induced/{scenario_workflow,branch_dag,variable_slots}_<dom>.json`) + two_stage_agent scenario-2단계 통합.

---

## 4. 환경 셋업 (coworker box)

```bash
# 1) 코드 + 학습데이터 (전부 git)
git clone -b facet-rft-2026 https://github.com/iamseungpil/boltzmann-attention.git bap-pi
#  → scripts/distill/{two_stage_agent,ontology_resolver,lora_train_chat_toolcall,...}.py
#  → reports/.../sft_data/sft_abstract_train_all.jsonl + induced/{obs_triggers,step_realization}_<dom>.json

# 2) ★eval용 SOP-Bench (public, tau2 대체)
git clone https://github.com/amazon-science/SOP-Bench.git && cd SOP-Bench && pip install -e .
#  → 데이터: src/amazon_sop_bench/benchmarks/data/<domain>/{sop.txt,toolspecs.json,tools.py,data.csv,test_set_with_outputs.csv,metadata.json}
#  → 우리 커스텀 agent = scripts/distill/{sop_bench_loader,compile_sop_ontology,workflow_executor}.py (Track A 제공)
#  → eval CLI: TSR/ECR/C-TSR + Tool Accuracy (src/amazon_sop_bench/cli)
#  (tau2-bench는 더 이상 필요 없음 — 개념 참조용으로만)

# 3) python env
#   학습(C5* phase-planner, 조건부): torch + transformers>=4.51 + peft + accelerate (검증: 4.51.3 / torch 2.7.0+cu126)
#   서빙: vllm==0.11.0 (--enable-lora ...) — P0(무학습)는 서빙만, API 모델 직접도 가능
#   SOP-Bench 의존: requirements.txt (pip install -e . 로 충족)

# 4) 모델: P0는 강 instruct 모델(API 또는 로컬 Qwen2.5-32B/72B). (C4* xattn / C5* SFT 시) Qwen2.5-7B/32B-Instruct.

# 5) ★user_sim 불요 (SOP-Bench는 single-shot, user 없음) → OpenRouter user_sim·judge 비용/이슈 전부 소멸.
#   LLM-fallback·생성형 step용 모델만 필요(agent-llm). OpenRouter 키는 fallback 모델 호출에만(선택).
```

---

## 5. 데이터/아티팩트 핸드오프

| 아티팩트 | 채널 | 비고 |
|---|---|---|
| abstract SFT jsonl + induced 온톨로지 맵 | **GitHub repo** | `sft_abstract_train_all.jsonl`, `induced/{obs_triggers,step_realization}_<dom>.json` |
| two_stage_agent.py / ontology_resolver.py / trainer / scorecard / grpo_reward | **GitHub repo** | `scripts/distill/` |
| eval 데이터 (domains/split/env) | **public tau2-bench** | clone+pip |
| 학습된 adapter (32B abstract, 70B) | **HF private** | repo push 금지(GB급) |
| 결과 results/coverage manifest | **GitHub** (`coworker_a100/` 하위) | 100MB↑면 manifest만 |

---

## 6. 협업 규약
- **branch**: 공유 `facet-rft-2026`. commit 전 `git pull --rebase origin facet-rft-2026`.
- **출력 서브트리**: coworker = `reports/facet_rft_2026/phase4_distill/coworker_a100/` 아래만 (Track A는 그 밖). 충돌 회피.
- **대용량**: results.json 100MB↑ commit 금지 → manifest만, 원본 HF/디스크. adapter는 HF.
- **git user**: `iamseungpil <iamseungpil@users.noreply.github.com>`. 파일 수정/추가 시 자동 commit+push.

---

## 7. 일정 (4× A100, 3주 — xattn 트랙 추가 반영)

| 주 | Track B (coworker) | Track A (우리) |
|---|---|---|
| **W1** | 셋업 + **B1* 32B abstract-none SFT** + **B2* ablation×LODO 매트릭스 착수**(base/resolver/fallback/monolithic) | B LODO(telret) 학습완료 → 7B two_stage telecom coverage + airline swap LODO |
| **W2** | **B2* 완성**(rule계열 전모드×3도메인×LODO) + **ontollm 모드** + **B3* fallback 70B probe** + (조건부)B4* GRPO | (a)resolver+(b)ontollm baseline로 **retail/airline rule-coverage gap 정량화** → B5* 표적 확정 |
| **W3** | **★B5* xattn neural resolver**(C-1 cross-attn 학습 → xattn 모드 매트릭스 + ABox-swap LODO + ABox-ablation) | 결과 종합, coverage 곡선·xattn vs rule/프롬프트 figure, 논문 표 |

---

## 8. Go/No-Go 게이트 (layered 재정의)

| 게이트 | 기준 | 판단 |
|---|---|---|
| **G1* (planner SFT)** | 32B abstract monolithic pass^1 ≥ 7B abstract, val 수렴 | 진행 / 데이터·trainer 점검 |
| **G2* (결정적 coverage)** | telecom `resolver` 모드 **coverage ≥ 60%** (상태머신서 온톨로지가 실제로 도구선택) | layered 핵심 양성 / inducer precision 재정제 |
| **G3* (ontology-swap 전이)** | telecom-planner + airline-ontology 의 pass^1·coverage가 **base 대비 +, in-dist의 ≥70% 회수** | H3 전이 입증 / ABox 재설계 |
| **G4* (fallback capability)** | catalogue 도메인(retail/airline)서 70B fallback이 7B fallback 대비 **miss-turn 정확도 +≥10%p** | capability가 잔차 메움 확인 / 결정적 확장 필요 |
| **G5* (GRPO, 조건부)** | anti-loop(step penalty)로 NONE max-step 실패 직접 감소 + pass^1 +≥5%p | 진입 / SFT로 충분 보고 |
| **G6* (xattn neural resolver, B5*)** | catalogue 도메인(retail/airline)서 `xattn` coverage·pass^1이 `resolver`(rule) 및 `ontollm`(프롬프트) **둘 다 상회** + **ABox-ablation으로 붕괴**(빈/틀린 M) + **swap LODO가 in-dist의 ≥70% 회수** | 본 트랙 novelty 입증 / 인코딩 분포정합·데이터 재설계 |
| **G7* (R4 scenario, B6*)** | scenario-2단계 planner가 평면 대비 **multi-fault task pass^1 +≥5%p**(누락 감소) + scenario 매칭 정확도 ≥80% (3도메인) | 계획층 가치 입증 / fault 클러스터 재정의 |
| **G8* (R3 branch / R1·R2 placeholder, B6*)** | branch=telecom anti-loop(max-step 실패) 감소 + placeholder=arg_bind **0.32→≥0.7**(인자 계약) | 실행층 가치 입증 / 슬롯 induce 정제 |
| **G9* (alias regime 성립, v1.35)** | **32B in-context arm(alias ON+source3, end-to-end bank)이 alias-OFF 대비 붕괴 없이 + leaderboard 40.30 대비 합리적**. (싼 proxy: `gate_alias.py` teacher-forced alias+s3 ≥0.6) | 7B SFT 학습가치 확정 / alias-ON should_T 바닥이면 도구설명·정책 신호 보강 후 재게이트 |

---

## 9. 핵심 리스크 / 주의
- **결정적 executor 경계**: retail/airline은 인스턴스-특수 trigger 과적합 → `resolver` 단독 coverage 낮을 것(예상·정상). **fallback 비중 자체가 결과** — 낮은 coverage를 실패로 보지 말 것.
- **predicate 정합성**: `ontology_resolver`의 `ObservedState` flatten/_scalar는 `induce_observation_triggers.py`와 **동일 키잉** 필수(런타임 pred가 induced pred와 글자단위 일치). 도메인 추가 시 inducer 먼저 재실행.
- **planner step 파싱**: abstract 모델이 `Plan: <step>` 포맷을 안정적으로 emit해야 resolver 동작. monolithic 모드로 emit률 먼저 확인.
- **cross-domain 분포이동**: 도메인특수성 내부화 시 전이 취약(Transmuting 100→42.7). **ABox-swap sanity(틀린 온톨로지 → coverage 붕괴)** 로 "불변 절차만 weights" 확인.
- **user_sim 비용**: OpenRouter gpt-4.1 과금. 매트릭스 셀 多(2모델×4모드×3도메인×{in-dist,LODO}) → test N≈20–40/셀로 관리, 우선순위=telecom resolver/fallback + airline swap.
- **judge**: airline/retail reward_basis=nl_assertions → LLM judge 필수. two_stage_agent도 `_route_nl_judge_via_openrouter()` 포함(pull만).
- **trl**: seka_env 충돌 → 별도 venv. vLLM 0.11.0 / tau2 버전 Track A와 일치.

---

## 10. ★SOPBench `bank` 벤치마크 결함 — coworker 협의 조치사항 (2026-06-01, 실측 확정)

> **결론(리모트 GPU0 실측·검증):** Leezekun/SOPBench `bank`의 `should_succeed=True` 48 인스턴스 중 **8개가 저자
> 자신의 strict 오라클로도 통과 불가 = 진짜 벤치마크 결함**. 모든 bank 수치 해석의 전제이므로 coworker 실험에도
> 직접 영향. 아래 조치사항을 **다음 동기 시 협의 후 진행**.

### 10.1 무엇이 결함인가 (재현·검증 완료)
- 결함 8 인스턴스 / 2 goal: **`cancel_credit_card`×6** (메서드 `return False`), **`pay_bill_with_credit_card`×2**
  (`KeyError: 'credit_limit'`). 근본원인: 태스크 데이터의 `credit_cards`는 **list-of-dict**인데 `env/domains/bank/bank.py`
  메서드(L189–190, L209–213, L254–255, L271–278)는 **dict-keyed** 가정 → 매칭 영영 실패.
- **판정 기준 = evidence-A**(저자 strict 오라클이 전제충족+GT인자로 goal 호출 시 성공하는가). `evidence_a_probe.py`로
  BEFORE 8실패 → 패치 AFTER 0실패 → revert 8실패 **리모트 실증**.
- evidence-B(53 출시모델 output 교차검증)는 14개가 전모델 0%였으나, 그중 6개(get_loan/pay_bill/set_safety_box/
  transfer_funds)는 **오라클은 통과 = 극難·결함 아님**. ⇒ **"전모델 0%"(B)는 결함의 충분조건이 아니며, "오라클
  실패"(A)가 판정 기준**임을 데이터로 확립. coworker도 이 기준을 따를 것.

### 10.2 coworker 실험에 대한 조치 (협의 항목)
1. **bank 수치는 8개 제외/플래그하여 해석** — 실효 천장 = should=True 40/48. **arm-1/L0/arm-3/arm-4a 모든 bank
   pass-rate를 "8 결함 제외" 기준으로 보고**(분모 명시). 32B/72B sweep·LODO 결과 표에도 동일 적용. 안 그러면 우리·
   coworker 숫자가 일괄 하향편향.
2. **patched vs unpatched 결정** (★협의 필요): 결함을 (a) **건드리지 않고 8개 제외 보고**(기본·리더보드 비교 가능)
   할지, (b) `fix_bank_creditcard.py`로 **패치한 클론에서 재측정**(천장 회복, 단 리더보드와 비교 불가)할지.
   → **권장: 두 트랙 분리** — 주(主)는 (a) unpatched+제외(리더보드 정합), 보조로 (b) patched에서 "결함 제거 시
   천장" 1회 측정. coworker 클론은 **기본 unpatched 유지**(현 결과 호환), 패치는 opt-in.
3. **LODO 전이 측정 시 bank가 held-out일 때 주의**: held-out=bank면 분모에 결함 8개 포함되어 전이율이 인위적으로
   낮아짐 → bank-held-out 셀은 8개 제외 분모로 별도 보고.
4. **다른 6개 도메인도 evidence-A 스윕 권장**(협의): `evidence_a_probe.py --domain <d>`를 dmv/healthcare/hotel/
   library/online_market/university에 돌려 **유사 결함 유무 선제 점검**. 결함 있으면 그 도메인 천장도 보정. (bank만
   확인된 상태 — 19도메인 LODO/통합 TBox 전에 전수 점검이 안전.)

### 10.3 저자 보고 (이슈/PR) — 분담 협의
- **이슈 = FILE-READY**: `scripts/distill/ISSUE_paste_ready_bank_creditcard.md`(제목+본문, 내부 메모 제거). 중복 재검색
  완료(이슈 #1 license만 존재 = 중복 아님). **제출은 GitHub 쓰기 인증 필요 → 누가 계정으로 올릴지 협의.**
- **PR(선택)**: `scripts/distill/sopbench/fix_bank_creditcard.py`(멱등 anchor 패치, `--check` 지원, 리모트 실증). 이슈에
  "fix PR 가능" 명시함. fork·push·PR도 인증 필요 → 분담.
- **제출 전 수동 2건**: (i) 제출 직전 이슈 재검색(중복), (ii) 선택적 1-task end-to-end 부록(john_doe cancel).

### 10.4 인프라 교훈 (coworker도 준수 — ★중요)
- **로컬 Windows python은 Store 스텁**(exit49, 미실행). **모든 측정은 리모트(rr.ps1)에서**, **RC·scanned 수 확인 후에만
  "실측"으로 인용**. (이번에 미실행/0-반환 결과를 실측이라 기록하는 fabrication 3회 발생·전부 철회 — 동일 실수 금지.)
- **스크립트 배포는 git pull**(SFTP 텍스트 업로드가 들여쓰기/구버전 손상시킨 사례 있음). 배포 후 glob·핵심 라인 확인.
- **rr.ps1은 메시지당 1호출**(paramiko가 형제 호출 일괄 취소).
- **자작 oracle-replay `mre_bank_impossible.py`는 신뢰 불가**(dirgraph 나열순서 아티팩트 → 허위 48/48). 결함 판정은
  **`evidence_a_probe.py`(전제충족 후 goal 직접 호출)** 사용. 저자 내장 evaluation 교차검증(`offline_crosscheck.py`)은
  corroboration용.
- 자산: `reports/facet_rft_2026/{evidence_a_bank.json, xcheck_bank_evidenceB.json}`, 권위본 설계=`scripts/distill/
  WORKFLOW_ONTOLOGY_DESIGN.md` §11.13/11.14.
