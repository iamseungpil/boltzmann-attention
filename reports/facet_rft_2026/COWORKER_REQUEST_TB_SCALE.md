# ▶▶ Coworker 요청서 — TaskBench scale 실험 (Qwen2.5/Qwen3 이원화, 4×A100 80GB) — 2026-06-10 **v4** (06-12 P2/P3 전면 교체)

> 🚨 **v4 액션 알림 (2026-06-12) — 착수 전 필독**:
> - **구 P2(72B gold-SFT)·구 P3(타-family 앵커) = SUPERSEDED.** 만약 구판 기준으로 이미 착수했다면 **중단 요청** (근거: P1 판정=①적중 −5.4라 구 P2의 "+ 신호" 전제 불발 + 기제 예측 72B SFT −4.4 + TaskBench 외부동결로 구 P3 질문 가치 소멸). 단, 구 P2 게이트("P1이 + 시")가 불발이므로 **착수 안 했을 것으로 추정** — 했더라도 sunk cost 작을 때 끊는 게 이득.
> - **신판: P2 = §7**(대형 결정론-leg 2×2, 추론-only) → **P3 = §9**(신규 벤치 포트폴리오 대형 arm: ODCV→Amazon SOP-Bench→τ²). 전부 추론-only(학습 0).
> - **무관·계속**: SOPBench Track-B 작업(stack32b 등 v1.42 라인)은 이번 변경과 무관 — 그대로 진행.
> 📌 **구조 안내**: 모든 설계·실험 문서의 단일 마스터 = repo `scripts/distill/EXPERIMENT_DESIGN.md` (**§7 문서지도**에서 각 문서의 역할·상태 확인; 목표·순서 변경은 마스터 §0-§4에서만). 처음 읽는다면 마스터부터.

> **★v3 변경 (2026-06-11, 궤적 전수조사 — 결과문서 §8이 권위)**: Track A가 {1.5, 7, 14}B 전이-vs-용량을 로컬 측정(held-out Δedge = **+8.7 / −0.8 / +4.3**, sub500 동일-id 정합) 후 **전수조사로 기제 확정**: held-out 효과 = **(+)참조-인덱싱 규율 전이**(`<node-j>` 자기참조 제거: 7B −83%·14B 0.478→0.010/ex) ⊕ **(−)도구-어휘 간섭**(유효 도구명 −4~−8pp, train-도메인 이름 침투) ⊕ **(0)누락-불변**. ~~"형식-간섭"~~·~~"14B=용량 효과"~~ 철회 — 14B의 +4.3은 *base의 인덱스-오류율이 7B의 2.2배*라 교정 이득이 컸던 것.
> **⇒ P1 프로토콜 변경 (필수)**: 32B SFT **착수 전에 32B base census 선행** — P0a의 32B base 예측(MM sub500)에 `scripts/distill/taskbench/tb_census.py` 시그니처(자기참조율 nself/ex·유효 도구명 비율 valid_frac)를 재서 **Δ(32B) ≈ (인덱스-교정 이득 ∝ base nself) − (어휘 간섭 ~5pp)을 사전 예측·등록**한 뒤 학습. Q2의 질문 자체가 "용량이 +로 뒤집나"에서 **"기제-예측이 32B서도 맞나(맞으면 기제 확립, 틀리면 새 변수)"**로 정련됨 — 어느 쪽이든 1급 결과.
> (참고: RFT round-1/2는 Track A 로컬 진행 — round-2 = recall+validity 보상, 결과문서 §9 분기 사전등록.)

> **★v2 (2026-06-10 PM, 사용자 결정·정정 반영): 이원화 — Qwen2.5는 비교-통제용 유지 + Qwen3는 곡선-연장용 신규.**
> - **fact**: 72B는 Qwen2.5에만 존재(Qwen3 dense는 32B 천장, 그 위는 MoE 30B-A3B/235B-A22B). SOPBench 리더보드(Track-B 기준선)도 Qwen2.5 계열 → **SOPBench 작업은 Qwen2.5 유지(전환 아님)**.
> - **Q2(전이 NULL=용량? — P1)는 동일-family 통제 필수**: 오늘 7B NULL이 Qwen2.5-7B 측정 → **P1 SFT = Qwen2.5-32B**(Qwen3-32B면 family+scale 동시 변경 = confound).
> - **Q1/Q3(곡선 연장·72B-초과)는 Qwen3**: 단일 family가 0.6B→235B 커버. Track A가 0.6–14B 재측정(전환), coworker가 32B/235B-A22B. 오늘까지의 Qwen2.5 0.5–14B 곡선은 **P0에서 32B/72B만 보태면 완성** = 2-family 증거(곡선 모양의 family-불변성 체크)로 승격.
> **⚠️전 실험 공통 통제 (신규, 필수)**: Qwen3는 hybrid thinking 모드 → **non-thinking 고정**. 방법 = 요청 body에 `"chat_template_kwargs": {"enable_thinking": false}` (vLLM OpenAI 서버 지원; TaskBench inference.py payload에 1줄 추가) 또는 user 프롬프트에 `/no_think`. **두 트랙 같은 방법 사용**(차이나면 비교 무효). thinking-on은 별도 arm으로만(선택, 비교축 오염 금지).
> **⚠️MoE 표기**: 235B-A22B는 total 235B/activated 22B — 곡선에 올리되 "MoE(A22B)" 병기(dense 곡선과 시각 구분).

> 발신: Track A (7B/14B, 리모트 2×49GB). 채널 = branch `facet-rft-2026`. 마스터 정합: `scripts/distill/FIELD_GAP_LLM_VALUE_DESIGN.md` §18.1 Exp-A/C + `scripts/distill/HANDOFF_2026_06_10_taskbench_learning.md`. 결과 권위본 = `reports/facet_rft_2026/TASKBENCH_EXPERIMENT_RESULTS.md` (오늘 수치 전부 여기).
> **보고 규율 (사전등록, 위반 금지)**: ① TaskBench LODO = **supporting 전이만**(moat-(3) 주장 금지, FIELD_GAP §17.9 리뷰7-1) ② 지표 = **edge-F1 중심**(node ~포화) + type(single/chain/dag) 층화 ③ GT=GPT-4 back-instruct → **GPT-4를 teacher/증강에 쓰지 말 것**(순환, 리뷰7-2) ④ exact-match 채점이라 천장<100(valid 대안 penalize; A-0 감사: miss의 ~27%=관례).

## 0. 왜 요청하나 (Track A 오늘 실측 — 비교 타깃)

**(a) base prompted 곡선 (Qwen2.5-Instruct, 500-sub/도메인; 7B만 full)** — edge-F1은 1.5B→3B서 emerge, **14B까지 비포화**:

| 크기 | HF n/e-F1 | MM n/e-F1 | daily n/e-F1 |
|---|---|---|---|
| 0.5B | 10.5 / 0.0 | 17.0 / 0.0 | 5.1 / 0.2 |
| 1.5B | 50.3 / 2.3 | 55.9 / 3.0 | 54.5 / 18.6 |
| 3B | 64.4 / 19.1 | 72.6 / 27.6 | 70.5 / 33.9 |
| 7B (full) | 73.6 / 32.2 | 84.4 / 50.0 | 90.8 / 68.1 |
| 14B | 77.5 / 39.6 | 89.5 / 52.8 | 93.5 / 77.4 |
| (published gpt-4) | — | 90.9 / 69.3 | — |

**(b) gold-SFT LODO — v3 갱신 (전이-vs-용량 3점 + census 기제, 상세=결과문서 §5/§8)**: held-out MM Δedge(sub500 동일-id) = **1.5B +8.7 / 7B −0.8 / 14B +4.3** (U자형). 전수조사 기제 = (+)인덱스-규율 전이(이득∝base 자기참조율) ⊕ (−)어휘-간섭(~5pp) ⊕ (0)누락-불변. **미해결 질문(v3) = 이 기제-예측이 32B에서도 맞나** — P1 step-0의 base census로 Δ를 사전 예측하고 학습으로 검증.

## 1. 질문 3개 (이 실험이 답할 것)
- **Q1 (곡선 연장)**: edge-F1이 32B/72B(Qwen2.5)·32B/235B-A22B(Qwen3)에서 계속 오르나, 어디서 published gpt-4(69.3 MM)에 접근/포화하나? + **곡선 모양이 family-불변인가**(Qwen2.5 vs Qwen3 두 곡선 비교).
- **Q2 (★핵심, v3 정련 — 기제 검증, 동일-family 통제 = Qwen2.5)**: census 기제(Δ ≈ 인덱스-교정 이득[∝base 자기참조율] − 어휘-간섭[~5pp])가 **Qwen2.5-32B에서도 예측대로 맞나?** step-0 base census로 Δ 사전 예측·박제 → SFT로 검증. 맞으면 기제 확립(전이의 함수형 확보), 틀리면 새 변수 발견 — 어느 쪽이든 1급. (→+면 72B로 3점째.)
- **Q3 (선택, frontier-adjacent 오픈 앵커)**: 235B-초과/타-family 오픈모델은 prompted로 어디까지 가나? (별도 표 — family/양자화 confound로 곡선에 못 올림.)

## 2. 실험 arms (우선순위순; Qwen3 run은 **전부 non-thinking 고정**)

**P0 — prompted baseline, 두 family (전부 500-sub ×3도메인, 각 도메인 `user_requests.json` 앞 500줄 — §4 스크립트가 자동 생성)**
- **P0a (Qwen2.5 곡선 완성)**: Qwen2.5-32B(TP1-2, bf16) + **Qwen2.5-72B(TP4, bf16 ~145GB)** — Track A 기측정 0.5–14B에 보태 Qwen2.5 곡선 완결. 견적 합 ~2-4h.
- **P0b (Qwen3 곡선 대형단)**: Qwen3-32B(dense, bf16) + Qwen3-235B-A22B(공식 AWQ/GPTQ-INT4 ~120-130GB, TP4). (선택: Qwen3-30B-A3B = dense-32B 동구간 MoE 대조점.) 견적: 32B ~30-60분, 235B-INT4 ~2-4h. Track A가 같은 라인 0.6–14B 측정 → **0.6B→235B 단일-family 곡선 완성.**
- 산출: 두 family 곡선 분리 표기(같은 그림에 겹쳐도 좋으나 family 라벨 필수).

**P1 — ★gold-SFT LODO_mm @ **Qwen2.5-32B** (Q2 본명 — 동일-family 통제)**
- **(v3 신규, step-0 필수) 32B base census 선행**: P0a 32B base의 MM sub500 pred에 `tb_census.py` 시그니처 측정(자기참조율·valid_frac; --dir_a/--dir_b에 같은 dir 넣고 aggregate-A만 읽어도 됨) → **Δedge 사전 예측 박제 후 학습 착수**. 참조점: nself/ex = 7B 0.218→Δ−0.8 · 14B 0.478→Δ+4.3 (간섭 ~5pp 공제).
- 학습 데이터 = Track A와 **동일 생성**(아래 §4 `tb_build_sft.py`, seed 42 → byte-동일 jsonl): HF(single400/chain1000/dag전부795=2194) + daily(400/1000/275=1675) = 3869.
- 레시피 = 7B와 동일 통제: **LoRA r16/alpha32, 2ep, seqlen 6144, lr 1e-4, val 2%** (`scripts/distill/lora_train_chat_toolcall.py` 그대로 사용 가능; 32B는 grad-ckpt+TP/FSDP로 2~4GPU. 자체 트레이너 쓸 경우 위 하이퍼 고정).
- 평가 = held-out **MM full**(시간 빠듯하면 sub500 먼저→full 보충) + in-domain HF/daily sub500. 비교 4열: `base-32B / 32B+gold-SFT / base-7B / 7B+gold-SFT`(7B 수치는 §0).
- 견적: 32B LoRA 7584 step seqlen6144 ≈ **12-24h**(설정 따라). **판정(v3)**: 실측 Δ를 step-0 사전예측과 대조 — ①예측 적중 = 기제 확립(이후 스케일 투자 여부는 함수형으로 계산) ②예측 대비 초과 = 인덱스 축 외 추가 전이 발견(용량 가설 부분 부활) ③미달 = 32B 고유 변수 census로 재조사. (구판 "용량-바운드 확정/방식-바운드" 이분법은 철회 — 결과문서 §8.)
- ⚠️ in-domain이 두 스케일 다 평평하면 "gold-SFT 자체 무가치(base 포화)" — 그것도 1급 결과.

**P2 — (P1이 + 신호 시) Qwen2.5-72B gold-SFT LODO_mm** ⚠️**SUPERSEDED (2026-06-12)** — P1 판정 = ①적중(Δ −5.4 vs 예측 −5.0, §8.5)이라 "+ 신호" 전제 자체가 불발 + 기제 외삽상 72B SFT는 Δpred −4.4(비권장, §7 명기). **신판 P2 = §7 (대형-모델 결정론-leg 2×2)**.

**P3 — (선택) 타-family 오픈 앵커, prompted-only** ⚠️**SUPERSEDED (2026-06-12)** — TaskBench 외부동결 판정(TB결과 §1.5: 리더보드 2023-11 동결·frontier 정체)으로 "gpt-4(69.3)를 넘나" 질문의 외부 가치 소멸. **신판 P3 = §9 (신규 벤치 포트폴리오 대형 arm)**. (구안 보존: gpt-oss-120b / Llama-3.1-405B-INT4 anchor — 필요 시 §9와 병행 가능하나 비권장.)

## 3. 지표·산출물 (고정)
- metrics: `node_micro_f1_no_matching` / `link_binary_f1` (+가능하면 `-m argument`의 t/v-F1). **type 층화**(metrics json에 single/chain/dag split 포함됨).
- 산출물: ① 위 표 형식으로 `TASKBENCH_EXPERIMENT_RESULTS.md`에 추가(§1 표에 32B/72B행, §3에 P1 4열표, P3는 별도표) ② pred/metrics json은 자체 보관+경로 공유(대용량이라 커밋 불요, metrics json만 커밋 권장).

## 3.5 ★raw 궤적 커밋 요청 (2026-06-11 신규 — Track A 전수 census용, §3의 "커밋 불요" 일부 정정)
> 목적: Track A에서 우리 census 도구(`tb_census.py`/`tb_pr_census.py`+궤적 직독)로 ①32B base nself=0·valid=1.0 **독립 재검증** ②32B 잔여 오류축 분해(누락/edge-구조/인자 — 32B+ 개선 레버 선정의 입력) ③Qwen3-32B 평탄(14B→32B) 원인 ④(P1 학습 후) Δ=−5 적중 검증+간섭 census. sub500 pred는 파일당 500행(~1-3MB)이라 git 무리 없음.

**커밋 위치**: `reports/facet_rft_2026/trackb_raw/` (구조 유지: `preds/<dir>/<tag>.json`)

| # | 파일 (노드 경로 기준) | 비고 |
|---|---|---|
| 1 | `/scratch/taskbench_runs/preds/data_{multimedia,huggingface,dailylifeapis}_sub500/qwen25_32b.json` | P0a 3도메인 (최우선) |
| 2 | 동일 경로 `qwen3_32b.json` ×3 | P0b |
| 3 | `$OUT/<dom>_sub500_eval_<tag>/metrics/<tag>.json` (위 6런 전부) | 공식수치 교차확인용, 수 KB |
| 4 | `$OUT/p1_census_prereg.json` + step-0 census 원본 md | 사전등록 동결본 |
| 5 | `preds/<dir>/<tag>.log` (inference 로그) | failed/드롭 id 확인 — 크면 `grep -E "Failed|Success" \| wc` 요약만 |
| 6 | (완료 시) `qwen25_72b`·`qwen3_235b_a22b_int4` 동일 세트 | P0 후속 |
| 7 | (P1 완료 시) 32B-SFT held-out MM sub500 + in-domain pred/metrics | Δ=−5 검증 + 간섭 census의 본체 |

## 4. 재현 절차 (전부 박제됨 — 재발견 금지)
1. **클론**: `git clone https://github.com/microsoft/JARVIS && cd JARVIS/taskbench` (Apache-2.0). 데이터 3도메인 동봉(`data_huggingface` 7458 / `data_multimedia` 5555 / `data_dailylifeapis` 4318).
2. **venv**: python3.10+; `pip install numpy scikit-learn networkx python-Levenshtein "datasets==2.14.5" "pyarrow==12.0.0" rouge_score aiohttp emoji click` + vllm(서빙용; P3 모델은 최신 vllm).
3. **inference.py 버그 패치 필수**: `loop = asyncio.get_event_loop()` 다음 줄에 `results = []` 추가 (미초기화 크래시).
4. **dependency_type**: HF/MM = `resource`, **daily = `temporal`**(tool_desc에 input-type 없음 — 틀리면 assert).
5. **500-sub 생성·serve·infer·eval 전부** = 우리 repo `scripts/distill/taskbench/` 재사용:
   - `tb_scale_curve.sh` — sub500 생성+모델 루프(serve→infer→eval)의 견본(경로 변수만 자체 클러스터로 수정).
   - `tb_build_sft.py` — P1 학습 jsonl 생성: `--tb_dir <클론> --domain data_huggingface --out sft_hf.jsonl --n_single 400 --n_chain 1000 --n_dag 0` + daily 동일(`--n_dag 0`=전부) → concat. seed 기본 42 고정.
   - `tb_build_eval.py` — **id-정렬·기형 gold skip·기형 pred sanitize 전부 내장**(직접 evaluate.py 돌리면 KeyError 함정 3종). `--tb_dir <클론> --domain <dom> --llm <tag> [--pred_file ... --dst ...]`.
   - `tb_eval_adapter.sh` — 어댑터 서빙+held-out/in-domain 평가 견본(GPU/UTIL/PORT 인자화 돼 있음).
6. **함정 요약**: pred id 순서≠gold 순서(→`tb_build_eval.py`가 처리) / `-m node`은 무효(`-m f1`이 node-F1) / metrics = `{dst}/metrics/{tag}.json`의 `overall_overall`.

## 5. 설계서·문서 맵 (cold-start용 — 처음이면 위에서 아래 순서로)

**이 요청서만 읽어도 실행 가능**하지만, 배경·판단 기준이 필요할 때 아래를 참조 (전부 repo `facet-rft-2026` branch 내 상대경로):

| 문서 | 무엇 | 이 요청과의 관계 |
|---|---|---|
| `../../scripts/distill/HANDOFF_2026_06_10_taskbench_learning.md` | **TaskBench 실험 핸드오프** — 인프라(§2)·converter(§5)·baseline(§3)·실행큐(§4) | 본 요청의 모체. 재현 gotcha 원본 |
| `reports/facet_rft_2026/TASKBENCH_EXPERIMENT_RESULTS.md` (이 폴더) | **TaskBench 결과 권위본** — full baseline·A-0 감사·scale곡선·LODO 수치 | **결과를 여기에 행 추가** (비교 타깃 수치 전부 이 문서) |
| `../../scripts/distill/FIELD_GAP_LLM_VALUE_DESIGN.md` | **thesis 설계서 (동결)** — §17.9 고정 thesis(soundness-coverage 패키지)·**§18 실행 권위**(이 요청 = §18.1 Exp-A/C의 scale 위임)·§15 위협/방어 | 보고 규율(supporting-전이·moat 금지)의 출처. 변경 금지(동결) |
| `../../scripts/distill/EXPERIMENT_DESIGN.md` | **마스터 설계서** — 목표·실험순서·헤드라인 지표 권위본 | Track A/B 분업의 상위 문서 |
| `reports/facet_rft_2026/COWORKER_EXPERIMENT_PLAN.md` (이 폴더) | **Track-B 협업 채널** — v1.42(32B SOPBench #0 sanity/#1 scaffold)·v1.43(7B transfer)·v1.44(본 요청 포인터) | 기존 SOPBench 32B 작업과의 우선순위 조정(§6 일정) |
| `reports/facet_rft_2026/SOPBENCH_EXPERIMENT_RESULTS.md` (이 폴더) | SOPBench 결과 권위본 — 게이트사다리 29/34·transfer 43-95% | "weight 전이 NULL은 SOPBench adapter-only≈0과 정합" 맥락 |
| `../../scripts/distill/SHIELDING_CBF_RELATED_WORK.md` | 관련연구 5축(shielding/CBF) — 포지셔닝 "제어-패러다임 인스턴스화" | 논문 프레이밍 배경 (실행 무관) |
| `../../scripts/distill/BITTER_LESSON_REBUTTAL_SOURCES.md` | bitter-lesson 방어 5라인 + 정직 양보 | 동상 (배경) |
| `../../scripts/distill/taskbench/` | **실행 스크립트** — `tb_build_sft.py`(SFT jsonl)·`tb_build_eval.py`(id정렬/sanitize 평가)·`tb_census.py`(궤적 전수 시그니처 — P1 step-0 필수)·`tb_typeclosure_probe.py`·`tb_rft_rollout.py`/`tb_rft_round.sh`(RFT)·`tb_scale_curve*.sh`·`tb_eval_adapter.sh`/`tb_train_lodo.sh`(BASE/PREFIX 인자화 드라이버) | §4 재현 절차가 이것들 사용 |
| `../../scripts/distill/lora_train_chat_toolcall.py` | LoRA 트레이너 (chat-format jsonl, P1 하이퍼 §2 참조) | P1/P2 학습 |

**(보강) 현재 활성 설계서 전체 — SOPBench 트랙 (Track-B 기존 작업 v1.42 #0/#1과 그 배경; TaskBench 요청 실행엔 불요, coworker가 양 트랙을 다 보므로 등재)**:

| 문서 (`../../scripts/distill/`) | 무엇 | 상태 |
|---|---|---|
| `HANDOFF_2026_06_05_PM_argfix_dggate_ladder.md` | **게이트 사다리 진입점** — ARGFIX→VALFIX→KEEPTUPLE→DGGATE 무재학습 4-fix, BOTH 15→29/34·천장34 | 활성 (scaffold 플래그 정의 원본 — Track-B #1이 쓰는 통합 scaffold) |
| `HANDOFF_2026_06_06_xdomain_full.md` | **cross-domain transfer 진입점** — held-out bank 43.3/library 75.8/healthcare 95.2%, 재학습0·scaffold가 전부 | 활성 (v1.43의 근거; "weight 전이 NULL" 맥락의 1차 출처) |
| `CROSS_DOMAIN_TRANSFER_DESIGN.md` | Exp-5(A축 transfer) 설계 — LODO/train-1·ABox-swap·지표 규율 | 활성 |
| `RUNG1_REDESIGN_2026_06_04.md` | decision-axis A/B/C 정식화(§8-9)·H3 offload 결론(§10.3)·LOCK 배경 | 참조 (결정축 격상 근거) |
| `RUNG1_SOURCE_LADDER_DESIGN.md` | **LOCK 원본** — 결정-emission SFT 3-NULL 종결("emission 변종 금지") | 참조 (B축 설계 시 위반 금지 목록) |
| `GUARD2_DIRGRAPH_MIRROR_DESIGN.md` | Guard-2(게이트=evaluator exact 재구성, OVER0/UNDER0) 검증 설계 | 완료 (precision=1 by-construction의 증명 절차) |
| `RESIDUAL11_FIX_DESIGN.md` | 잔여11 fix 설계(4 BLOCKING 가드) — KEEPTUPLE/DGGATE의 안전절차 | 완료 |
| `TASK_CONSTRAINT_DESIGN.md` | TBox/ABox 재해석·alias-마스킹(§8.5★) 설계 — **P3(이름암기 차단) alias 레시피 원본** | 활성 (TaskBench alias arm 때 재사용 예정) |
| `WORKFLOW_ONTOLOGY_DESIGN.md` | 워크플로 온톨로지 agent 설계(★§9 LLM-in-loop) — Track-B 계획서 모체 | 참조 |
| `GROUNDED_BIZ_AGENT_BENCH_DESIGN.md` | 신규(06-10 DRAFT): CDP 직접 벤치 설계(T1-T5·GT=검증기·리뷰훅 R1-R8 대기) | DRAFT (실행 전) |
| `REGULATORY_DETERMINISM_SOURCING.md` | 규제 1차원문 판정(06-10 완료): 로깅+검증으로 충족→결정론-leg 철회·검증가능성 후퇴; SR 26-2 수확 | 완료 (moat 문구에 영향 — 보고 시 "결정론 필수" 주장 금지) |
| `BUGREPORT_SOPBench_bank_impossible_tasks.md` | bank PartA8/PartB6 벤치 결함 보고 — 정직분모 34의 근거 | 완료 (천장 회계 출처) |

**한 줄 배경 (왜 이 실험인가)**: 동결 thesis = "소형 모델이 경로 *제안*(coverage) + 결정론 게이트가 soundness 보장 + 재학습0 전이" 패키지. TaskBench는 그 중 coverage/구조-예측 leg의 *supporting* 측정(실행·soundness 없음 → moat 주장 불가). 오늘 7B에서 ①edge-F1만 진짜 headroom(node 포화) ②gold-SFT의 held-out 전이 NULL을 확정 → 이 요청의 P1(32B 동일-레시피)이 "NULL=용량 한계인지"를 판정해 Track A의 다음 수(alias-마스킹 vs RFT vs 스케일 투자)를 게이트한다.

## 6. 일정 제안
P0(반나절) → P1(1-2일) → 판정 공유(채널) → P2/P3(조건부, 1-3일). **P1 결과가 Track A의 다음 수(alias vs RFT vs 스케일)를 게이트하므로 P1 우선 완주 요청.** 기존 Track-B(32B SOPBench SFT, v1.42 #0/#1)와 GPU 경합 시 — Track-B #0 sanity가 먼저, 그 다음 본 요청 P0/P1 권장(어차피 32B 서빙/학습 인프라 공유).

## 7. ★P2 — 대형-모델 결정론-leg 2×2 (2026-06-12 신규, P0/P1 완료 후 다음 큐)

> 배경: P1 사전등록 적중(Δ −5.4 vs 예측 −5.0, §8.5)으로 "대형은 SFT 잃기만 함" 확정 → 대형 모델의 레버는 결정론-leg(guided decoding)뿐이라는 게 thesis 예측. **이번 P2 = 그 예측을 32B/72B/235B에서 직접 검증** (Track A는 7B에서 guided 완료: daily +8.0·MM 합성 57.22, 결과문서 §9.5b). 전부 추론-only(학습 0) — A100×4에서 서빙만.

**키트 (repo에 전부 있음)**: `tb_guided_schema.py`(--dep resource/temporal)·`tb_guided_patch.py`(inference.py에 env-게이트 no-op 패치)·드라이버 견본 `tb_guided_mm_dpo2.sh`. vllm 요청 문법 = `extra_body={"structured_outputs": {"json": <schema>}}`(0.10.x+ 지원; 구버전이면 guided_json fallback — deprecation 경고만). 도메인당 더미 1요청으로 grammar pre-warm.

**arms (전부 MM sub500, 동일 첫-500 — 사전등록 예측 동결 2026-06-12, Track A가 trackb_raw 원본 census로 산출)**:
| # | arm | 서빙 | base census (실측) | ★사전예측 |
|---|---|---|---|---|
| P2a-1 | **lodo_mm_32b(SFT)+guided** | 32B+LoRA TP2 | SFT valid 0.952(간섭 −4.8pp) | **micro 56.5→+3~5 (어휘분 회복)** — base 61.9 미복원분 = 플랜-단축/구조축 귀속(census로 분해) |
| P2a-2 | **base-32B+guided** (통제) | 32B TP2 | valid 1.000·nself 0 | **Δ≈0** (고칠 어휘 없음 — "guided=0-cost 보험" 행) |
| P2b-1 | base-72B+guided | 72B TP4 | valid 0.998·nself 0.030·deficit +0.09 | **+0~0.5** (잔여 어휘 미세) |
| P2b-2 | base-235B+guided | 235B-INT4 TP4 | valid 1.000·**nself 0.143**(Qwen3 대형도 인덱스 오류 잔존=family-의존) | **Δ≈0** (guided는 어휘만 — nself는 못 고침 = 음성 통제) |
| P2c | base-32B+guided+**promptslim**(desc 제거 tool_desc) | 32B TP2 | 7B 실측: 목록 51% 절감=−3.1 edge | **−3.1보다 작은 손실** 예측(대형=이름만으로 의미매칭 ↑) — "names-only 사정거리 × 스케일" 곡선의 32B 점 |

**비권장 (기록만, 실행 불요)**: ①32B 균형-DPO — Track A 측정상 32B base 누락축 거의 소멸(deficit +0.024 vs 7B +0.256) → 기대이득 작음 ②72B/235B gold-SFT — 기제 예측 Δ=19.4×nself−5.0 → 72B −4.4·235B −2.2 (둘 다 음수; 235B-INT4 MoE LoRA는 비용도 비현실적).

**산출물**: trackb_raw 동일 구조로 preds/metrics push + §8.5에 행 추가. 판정 기준: P2a-1이 +3 이상이면 "대형의 결정론-leg 회복" 확정 → §10 분류의 32B 열 완성; P2c가 −1.5 이내면 비용-leg(도구폭발 컨텍스트 절감)에 대형-모델 행 추가.

**일정**: P2a(반나절, 32B 서빙 재사용) → P2b/P2c(반나절). 전부 추론-only라 Track-B 학습 잡과 GPU 경합 시 빈틈에 끼워도 됨.

## 8. ★프레임워크 목적·벤치 포트폴리오 공유 (2026-06-12 — Track-B 향후 분담 예고)
> **권위 = `../../scripts/distill/BENCH_PORTFOLIO_FRAMEWORK_DESIGN.md`** (요약 = `TASKBENCH_EXPERIMENT_RESULTS.md` §10.5).

**한 줄 요약**: 목적은 특정 벤치가 아니라 **벤치-불변 규칙(R1-R8: 심볼=컨텍스트 복사+enum 집행·gather 선행·결정=게이트 offload·의미매칭=모델/공간=제약·정책행동=양방향 on-policy만·구조선택=K+검증선별·base census→레버·궤적 census 규율)을 내재한 프레임워크**로 전 벤치를 최소 어댑터(A1-A5)로 커버하는 것. 새 벤치 비용은 A2(정책 NL→제약 구조)로 수렴하고, 그걸 학습 front-end가 대체하는 게 thesis.

**포트폴리오 (확정분)**: TaskBench·SOPBench(완료) → **τ²/τ³-bench**(신규 1순위 — 순수 NL정책=A2 끝점+유일 활성 frontier 리더보드) → **Amazon SOP-Bench**(12도메인 LODO 스케일업; ⚠️우리 SOPBench와 이름충돌 — 표기 구분) → AppWorld·ODCV-Bench(스팟).

**Track-B 함의 (P2 이후 예고 — 지금 액션 불요)**: 대형모델(32B/72B) arm은 향후 ①τ²-bench pass^k에서 "대형 base±게이트" (R3·일관성 이득) ②Amazon SOP-Bench 12도메인에서 32B base census→처방(R7) 적용이 자연 후속. P2(§7) 완료 후 구체 명세 추가 예정. TaskBench 외부 동결 판정(§ = TB결과 §1.5: 리더보드 2023-11 동결·frontier 정체 64.4·ToLeaP GPT-4o 행 인용금지)도 보고서 작성 시 참조.

## 9. ★P3 (신판, 2026-06-12) — 신규 벤치 포트폴리오 대형-모델 arm (P2 완료 후 순차)
> 구판 P3(§2, 타-family 앵커)는 supersede — TaskBench 외부동결로 질문 가치 소멸. 포트폴리오 권위 = `../../scripts/distill/EXPERIMENT_DESIGN.md` **§1.5** + `BENCH_PORTFOLIO_FRAMEWORK_DESIGN.md`(§8 요약 참조). **Track-B 몫 = 대형(32B/72B) arm만** — 어댑터(A1-A5)·정책 컴파일(A2)·7B 기준선은 Track A가 먼저 깔고 인계. 전부 추론-only(학습 0).

**P3c — ODCV-Bench 스팟 (순서상 먼저: 즉시 가능·최저비용, 40 시나리오)**
- 32B/72B base(prompted) **위반율** 측정 vs +결정론 게이트 0% — "KPI-유혹 위반은 크기로 안 풀리고 게이트로 풀린다" 단일-주장 실험. 공개 기준선: frontier 30~50% 위반 (McGill, arXiv 2512.20798, repo `McGill-DMaS/ODCV-Bench`).
- 사전예측: 대형 base도 위반 >0 (크기 비단조 가능 — KPI-유혹은 capability 축이 아님), +게이트=0.

**P3b — Amazon SOP-Bench 12도메인 대형 census + 기준선** (`amazon-science/SOP-Bench`, arXiv 2506.08119, CC-BY-NC)
- step-0 (R7 절차): 12도메인 **base census**(위반/누락/어휘 시그니처) → 처방 사전등록 → Qwen2.5-32B/72B base 성적 행렬(Task Success/Execution Completion/Tool Accuracy).
- 논문 보고 open-weights 행(DeepSeek-R1·Llama-3.3 ≈ proprietary)과 교차 검증. 대형 SFT arm은 **기제상 비권장 예상**(P1 −5.4 외삽) — census가 뒤집으면만 재고.
- ⚠️표기: 우리 SOPBench(UCSB)와 **이름충돌** — 문서·코드에서 "SOP-Bench(Amazon)" 표기 통일.

**P3a — τ²-bench 대형 arm (Track A 어댑터 인계 후 — 대기)**
- 32B/72B base ± 결정론 게이트(Track A가 컴파일한 retail 제약 구조 사용 — **Track-B는 정책 authoring 안 함**), 지표 pass^1 / **pass^k(k=4)**.
- 사전예측: ①게이트 이득은 pass^1보다 **pass^k에서 더 큼**(일관성=분산 억제 — 7B 검증 효과의 대형 재현) ②대형 base 위반 클래스는 7B와 동형(비율만 상이 — census 귀속).
- 외부 가치: 유일 활성 frontier 리더보드(~30모델)라 "오픈 대형±게이트 vs frontier"가 공인 무대에서 찍힘.

**산출물·순서**: trackb_raw 동일 구조 push + 결과는 TB결과 §8.5 형식(사전등록→실측→census). **P2(§7) → P3c → P3b(step-0 census부터) → P3a(인계 대기).** 일정 견적: P3c 반나절·P3b census+기준선 1-2일·P3a 어댑터 도착 후 1일.
