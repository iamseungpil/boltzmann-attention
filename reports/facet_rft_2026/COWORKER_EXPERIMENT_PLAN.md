# Coworker 실험 계획서 — Facet Distillation (goal→tool 절차 내부화)

> 대상: 4× A100 80GB 보유 coworker. 공유 채널 = GitHub `iamseungpil/boltzmann-attention` branch **`facet-rft-2026`**.
> 본 계획은 `reports/EXPERIMENT_DESIGN_v1_7_facet_rft.md` **§13 (v1.25)** 방향을 구현한다. 먼저 §13을 읽을 것.

---

## 0. TL;DR — 역할 분담

| Track | 담당 | GPU | 핵심 역할 |
|---|---|---|---|
| **A (우리)** | woori box | A6000 ×2 | ✅데이터·도구(trainer/scorecard/grpo_reward/semantic layers) **완비** · **7B full/none 학습 중** · eval·WISE-Flow baseline |
| **B (coworker)** | A100 box | **A100 ×4 (80GB)** | **32B 학습 · cross-domain eval 매트릭스(대규모) · on-policy GRPO · capability-ceiling probe** |

**한 줄 목표**: teacher 궤적에서 **goal→tool 선택 절차**를 student 가중치에 distill해, (1) student의 goal→tool 일치도(F1/seq_F1, base 거의 0)를 끌어올리고, (2) **재학습 없이 held-out 도메인으로 전이**되며, (3) full-prompt 대비 토큰/KV/latency를 줄이는지 검증.

---

## 0.5 ★ 업데이트 (2026-05-30) — Track A 도구 완비, 즉시 착수 가능

**Track A가 B1~B3에 필요한 도구·데이터·설계를 전부 제작·검증·commit 완료.** coworker는 clone 후 바로 32B 학습/eval/GRPO 착수 가능.

**도구 인벤토리** (`scripts/distill/`, 전부 repo):
- `lora_train_chat_toolcall.py` — chat-SFT 트레이너(멀티턴 tool-use, assistant-only 마스킹, `--system-mode full/none`). **검증·버그픽스 완료**(torch_dtype, **grad-ckpt는 peft wrap 후 + enable_input_require_grads + use_reentrant=False** — 안 하면 long-seq backward OOM). **7B full/none 현재 Track A에서 학습 중**(레시피 그대로 32B에 적용).
- `procedure_scorecard.py` — **eval 지표(F1/seq_F1 headline + recall/precision/seq_match/arg_bind/extra/order)**. fix-coverage 단독 폐기.
- `metric_mining.py` — AUC 기반 지표 발굴(F1이 최강, AUC 0.902).
- `fault_fix_induce.py`·`param_dataflow.py` — goal→tool + 파라미터 provenance semantic layer. 맵 commit: `induced/{fault_fix_map,task_required_tools,param_dataflow_{telecom,retail,airline}}.json`.
- `wiseflow_baseline.py` — prompt-side 비교군 inducer/injector.
- `grpo_reward.py` + `GRPO_REWARD_DESIGN.md` — **B3용 검증된 dense reward**.

**핵심 측정 결과(데이터-주도, 지표·reward 근거)**:
- 큰/작은 격차 = **94% 절차(distillable)**, capability 벽 1%. base 7B 실패 시 recall ~0.04 + precision 0.03~0.1(과잉진단).
- **F1/seq_F1이 success/failure 최강 변별**(AUC 0.902 teacher / 0.985 student). 3 도메인 일반화(telecom/retail/airline, read-제외·requestor 정정 후 전부 +0.2~0.6).
- **arg_bind**(파라미터 dataflow): teacher 포화(0.99)·**student 약점 노출(0.32~0.73 ID 할루시네이션)** = 학습신호.
- GRPO dense reward 검증: 실패 롤아웃 seq_F1 0.255±0.291(std>0) → all-fail group도 advantage(sparse cold-start 구제).

**바뀐 점**: 지표가 fix-coverage → **scorecard(F1/seq_F1)**. 데이터·trainer·scorer·reward 전부 **repo에 존재**(이전 "제작 예정" 해소). adapter만 HF 교환.

---

## 0.6 ★ 업데이트 (2026-05-30 PM) — 7B eval 완료(3도메인) + Group J + a→b→c 사다리

**Track A가 7B full/none 학습+eval을 3도메인 마무리.** 핵심:
- **★학습은 multi-domain**(`sft_plain_train_all.jsonl` = telecom 462 + retail 831 + airline 246 = **1539**). telecom-only 아님. → eval은 도메인별 **in-distribution held-out task**(test split). *미학습 도메인 zero-shot 전이는 LODO(아래)로 별도 측정 — 미실시*.
- **3 도메인 모두 NONE(정책제거 내부화) ≥ FULL(정책유지)**:

  | domain | test N | NONE | FULL |
  |---|---|---|---|
  | telecom | 40 | 0.350 | 0.300 |
  | retail | 40 | ~0.77 | 0.641 |
  | airline | 20 | 0.400 | 0.300 |

  → 정책 토큰 내부화 **무손실(efficiency thesis) multi-domain 재현**. retail 최고=학습량 831 최다와 정합.
- **★airline judge 버그 + 수정(coworker도 필수)**: airline `reward_basis=nl_assertions`(+communicate)뿐 → reward에 **LLM judge 필수**. tau2 `evaluator_nl_assertions`가 bare `gpt-4.1-2025-04-14` 호출 → litellm OpenAI 라우팅 → `OPENAI_API_KEY` 없음 → "Missing credentials" → airline/일부 retail reward 미집계. telecom은 db/env로 채점해 안 걸림. **수정**: `phase1_runner._route_nl_judge_via_openrouter()`(judge→`openrouter/openai/gpt-4.1`, commit 7530d14). **coworker eval에도 이미 적용됨**(pull만 하면 됨).
- **실패-주도 TBox 확장 Group J(4종)**: design doc §15. `repairs_state/diagnosis_sufficient_for/distractor_for/escalate_when`. `induce_tbox_relations.py`로 3도메인 ABox(`induced/tbox_relations_<domain>.json`) 산출 완료. NONE 실패 63%=recall-miss/anti-loop.
- **a→b→c 사다리**: (a)SFT floor[완료] → (b)offline DPO(`build_dpo_dataset.py`, 1171 preference pairs, distractor 음성쌍) → (c)GRPO(Group J reward). **★trl 설치 불가**(seka_env transformers 4.51.3과 모든 trl 버전 충돌) → **수동 DPO/GRPO**. coworker A100 박스가 trl 호환 env를 새로 만들면 trl 사용 가능(권장: transformers 맞춘 별도 venv).

---

## 1. 배경 (요약)
- tau2-bench 도구 에이전트. 큰/작은 모델 격차는 **거의 전부 절차(distillable)** 임이 실측됨: Qwen-7B telecom 실패의 **도구선택 58% + 형식 36% = 94%**, long-horizon capability 벽은 1%. student **fix-coverage = 0.06**(올바른 fix-tool을 거의 안 부름).
- **목표→도구(positive)는 성공/실패를 강하게 변별**(disc +0.36, FULL-coverage +48%p). 제약(precedes/mutex)은 변별 0(폐기).
- 따라서 distill 타깃 = **goal→tool 선택 절차**. (상세·문헌 근거: §13)

---

## 2. 검증 가설 & 지표

**Thesis**: contrastive 유도 goal-conditioned 선택 절차를 가중치에 distill(=steering이 못한 matrix 성분), goal→도구-역할 추상화로 일반화, 도메인 인스턴스(ABox)는 swap → cross-domain 전이.

**지표** (scorer=`scripts/distill/procedure_scorecard.py`, 완료·3도메인 일반화)
- **F1 / seq_F1** (핵심·headline): goal→tool 일치(recall·순서·minimality 통합). 데이터-주도 발굴서 최강 변별(AUC 0.902). base student 거의 0 → SFT 상승폭이 1차 판정.
- 분해축: recall / precision(minimality) / seq_match / **arg_bind**(인자 바인딩=student 품질) / extra_actions(over-diagnosis).
- **pass^1** (tau2 test split, 도메인별) — 최종 성능.
- **cross-domain transfer**: telecom+retail 학습 → **airline held-out** 전이 +%p.
- **efficiency**: retained pass^1 vs 절감 토큰/KV/latency (full-prompt 대비).
- **mms-chain pass^1**: capability ceiling 분리용 (multi-fault).

**비교군(baseline)**
- B0: full-prompt agent (현 phase1 baseline).
- B-WISE: WISE-Flow식 prompt-side prerequisite-워크플로 주입 (Track A 제작).
- (옵션) Graphify-RAG.

---

## 3. Coworker 태스크 상세 (B1–B4)

### B1. 32B student LoRA-SFT  ★우선
- **목적**: 큰 student + 내부화된 절차가 7B 대비 얼마나 더 메우는지(특히 mms).
- **GPU**: 2× A100 (FSDP) 또는 1× 80GB (LoRA + grad checkpointing). 
- **입력**: `reports/facet_rft_2026/phase4_distill/sft_data/sft_plain_train_all.jsonl` (repo, clone로 확보) + **chat-SFT trainer**(Track A 제공: `scripts/distill/lora_train_chat_toolcall.py`, multi-turn tool-use, assistant-only loss). base = `Qwen/Qwen2.5-32B-Instruct`.
- **변형 (2축)**: (a) **데이터**: plain(1539) / facet_L1(ontology-clean) — plain vs facet confound 격리. (b) **`--system-mode`**: `full`(정책 prompt 유지 = plain-SFT baseline, 추론 시 prompt 필요) vs **`none`**(정책 제거 = **내부화 arm**, 정책을 가중치로 흡수 → 추론 시 prompt 불필요; seq ~절반 5K로 학습도 쌈). §13 efficiency thesis 검증. → 최소 4 조합(plain/facet × full/none).
- **trainer**: `scripts/distill/lora_train_chat_toolcall.py` (repo, **제작·인코딩 검증 완료**: incremental assistant-only 마스킹, tool_call args str→dict 정규화, dual-control 처리, boundary_mismatch 0). 검증: full supervised-frac 0.18·seq max 13K / none 0.35·seq max 7.7K. **env 주의**: seka_env엔 peft 없음 → 학습은 `torch+transformers>=4.51+peft+accelerate` env 필요. `--max-seq-len` 기본 14336(telecom 정책~6K 때문; none 모드는 더 짧음).
- **출력**: adapter(`lora_adapters/qwen32b_{plain,facetL1}/`) + `train_meta.json`. adapter는 **HF private model repo로 공유**(repo push 금지, GB급).
- **성공기준**: 수렴(val loss↓), 실제 lift는 B2에서 측정.

### B2. Cross-domain transfer EVAL 매트릭스  ★대규모 병렬 (coworker 핵심 기여)
- **목적**: 학습된 student를 tau2 **test split**에서 평가하고 전이 매트릭스를 채움.
- **GPU**: vLLM 서빙(adapter 머지 후) 1 GPU/모델 + 다중 에피소드 병렬. 4 GPU면 2~3 모델 동시 + 에피소드 fan-out.
- **매트릭스**: {7B(Track A), 32B(B1)} × {baseline, plain-SFT, facet-SFT} × {full, none system-mode} × {telecom, retail, airline} test split. user_sim = **`openai/gpt-4.1` via OpenRouter** (키 공유; `OPENAI_BASE_URL`=openrouter 필수).
- **★efficiency 측정(§13 핵심)**: `none`-mode student(정책 prompt 0)의 **retained pass^1** vs `full`-prompt baseline + 절감 토큰/KV/latency. "정책을 가중치로 내부화해도 pass^1 유지되는가"가 thesis 검증.
- **러너**: `scripts/phase1_runner.py` (repo; 패치됨 — `--agent-llm`=로컬 vLLM, `--user-llm`, `--agent-api-key`). **주의(기존 버그)**: 도메인별 `--task-set <domain>` 필수(기본 telecom), airline judge는 `OPENAI_BASE_URL`/`OPENAI_API_BASE`=openrouter 설정.
- **출력**: `reports/facet_rft_2026/phase4_distill/coworker_a100/eval/<run>/results.json` + fix-coverage/pass^1 manifest.
- **성공기준**: G1/G2/G3 (아래 §8).

### B3. On-policy GRPO (학습 사다리 ③) — 조건부
- **목적**: SFT가 못 메운 mms residual을 student 탐색+reward로 보강.
- **GPU**: 4× A100 (policy + rollout 서빙). **trl 주의**: Track A seka_env(transformers 4.51.3)는 trl 설치 시 transformers 다운/업그레이드 충돌 → 우리는 **수동 GRPO 루프**. coworker는 **trl 호환 별도 venv**(transformers 버전 맞춤) 만들면 trl GRPOTrainer 사용 가능(권장).
- **reward**: `scripts/distill/grpo_reward.py`(검증 완료) — `w_pass·pass + w_proc·seq_F1 − w_extra·extra + w_arg·arg_bind`(기본 1.0/0.5/0.3/0.1). 설계·anti-hacking·ablation·통합은 **`reports/facet_rft_2026/GRPO_REWARD_DESIGN.md`** 참조. dense seq_F1이 sparse cold-start 구제(검증: 실패 롤아웃 seq_F1 0.255±0.291, std>0 → all-fail group advantage). 정책 init=SFT 어댑터(full/none).
- **선행조건**: B1/A2 SFT가 양성 lift(G1) 확인 후 진입.
- **출력**: GRPO adapter + reward curve + test 매트릭스 갱신.

### B4. Capability-ceiling probe — 조건부
- **목적**: mms(B=1%)가 절차인지 capability인지 분리. 32B/70B를 student로 두면 7B+SFT가 못 푼 mms가 풀리는가?
- **GPU**: 70B 추론 = 2× A100(AWQ) 또는 4× bf16. (Qwen2.5-72B / Llama-3.3-70B — Track A에 다운로드본 있음, HF로 공유 가능)
- **출력**: mms-chain pass^1 (모델 크기 함수) → capability vs 절차 결론.

---

## 4. 환경 셋업 (coworker box)

```bash
# 1) 코드 + 학습데이터 (전부 git)
git clone -b facet-rft-2026 https://github.com/iamseungpil/boltzmann-attention.git bap-pi
#  → scripts/distill/*, scripts/phase1_runner.py, reports/.../sft_data/*.jsonl 확보

# 2) eval용 tau2-bench (public, 별도 전송 불필요)
git clone https://github.com/sierra-research/tau2-bench.git
cd tau2-bench && pip install -e .        # tau2 패키지 + data/tau2/domains(split_tasks/policy/tools)

# 3) python env
#   학습: torch, transformers>=4.51, peft, accelerate, (B3) trl   # ★seka_env엔 peft 없음 → 별도 학습 env
#         (검증 env: transformers 4.51.3 / torch 2.7.0+cu126; flash-attn 있으면 --attn flash_attention_2)
#   서빙/eval: vllm==0.11.0  (우리 phase1 baseline과 버전 일치 권장)

# 4) 모델 (HF)
#   Qwen/Qwen2.5-32B-Instruct (학습/평가), Qwen/Qwen2.5-7B-Instruct
#   (B4) Qwen/Qwen2.5-72B-Instruct 또는 meta-llama/Llama-3.3-70B-Instruct

# 5) OpenRouter (user_sim) — 키 공유받기
export OPENROUTER_API_KEY=...   # phase1_runner: --user-llm openai/gpt-4.1 + OPENAI_BASE_URL=openrouter
```

---

## 5. 데이터/아티팩트 핸드오프 (전부 git 또는 HF)

| 아티팩트 | 채널 | 비고 |
|---|---|---|
| 학습 jsonl (plain/facet/aux) | **GitHub repo** (commit 완료) | clone로 즉시 확보. plain_all 54MB 등 |
| eval 데이터 (domains/split/env) | **public tau2-bench** | clone+pip install |
| 577MB shipped 궤적 | — | **불필요**(재생성 소스일 뿐) |
| trainer / scorecard / metric_mining / fault_fix / param_dataflow / wiseflow / grpo_reward | **GitHub repo (전부 commit 완료)** | `scripts/distill/` |
| induced 맵 (fault_fix / task_required / param_dataflow×3) | **GitHub repo (commit 완료)** | `reports/.../phase4_distill/induced/` |
| 학습된 adapter (7B↔32B 교차평가) | **HF private model repo** | repo push 금지(GB급); HF org에 공유 |

---

## 6. 협업 규약 (collision 방지)

- **branch**: 공유 `facet-rft-2026`. commit 전 항상 `git pull --rebase origin facet-rft-2026`.
- **출력 서브트리**: coworker 결과는 `reports/facet_rft_2026/phase4_distill/coworker_a100/` 아래에만 (Track A는 그 밖). 충돌 회피.
- **대용량**: results.json이 100MB↑면 commit 금지 → 요약 manifest만 commit, 원본은 HF/디스크. adapter는 HF.
- **git user**: `iamseungpil <iamseungpil@users.noreply.github.com>`. 파일 수정/추가 시 자동 commit+push 권장(우리 정책).
- **동기화**: 주 1회 결과 manifest 공유.

---

## 7. 일정 (4× A100 기준, 3주)

| 주 | Track B (coworker) | Track A (우리) |
|---|---|---|
| **W1** | 셋업 + **B1 32B plain/none-SFT** (도구 전부 repo, 즉시 착수) | ✅도구 일습 완료, **7B full/none 학습 중** → eval |
| **W2** | **B2 eval 매트릭스**(7B+32B × plain/facet × full/none × 3도메인) + B1 facet-SFT | 7B eval(scorecard F1/seq_F1), efficiency 측정, fault→fix 추가 정제 |
| **W3** | **B3 GRPO**(grpo_reward.py, 조건부) + **B4 capability probe** | 결과 종합, 논문 표/그림 |

GPU 할당 예시(W2): 2 GPU 학습 + 2 GPU eval 병렬.

---

## 8. Go/No-Go 게이트

| 게이트 | 기준 | 판단 |
|---|---|---|
| **G1 (SFT lift)** | base 대비 **seq_F1/F1 대폭↑**(base 거의 0) + arg_bind↑(ID 바인딩) + pass^1 ≥ baseline | 진행 / 데이터·trainer 점검 |
| **G2 (cross-domain 전이)** | telecom+retail 학습 → **airline held-out +≥5%p** vs baseline | thesis 핵심 양성 / ABox-swap 재설계 |
| **G3 (facet vs plain)** | facet가 plain 대비 **+≥3%p** | facet 기여 확정 / 없으면 **facet 폐기, plain만** |
| **G4 (GRPO, 조건부)** | mms-chain **+≥5%p** | capability 잔차 보강 / capability 벽 보고 |

---

## 9. 핵심 리스크 / 주의
- **fault→fix ground-truth 품질**(SkillFlow 교훈: 병목은 검색 아닌 라이브러리 품질). Track A가 단일결함/인과귀속으로 정제 후 scorer에 반영.
- **cross-domain = 분포 이동 → 도메인특수성 내부화 시 취약**(Transmuting 100→42.7 사례). 전이 평가에서 "불변 절차만 내부화" ablation 확인.
- **user_sim 비용**: eval은 OpenRouter gpt-4.1 호출(과금). test split N≈40/도메인이라 관리가능하나 매트릭스 셀 수 × N 주의.
- **★baseline 정정**: "retail/airline student baseline 없음(telecom only)"은 **base Qwen B0 baseline** 얘기였음. **distillation student(SFT)는 multi-domain**(retail 831·airline 246 학습 포함)이라 3도메인 모두 eval 완료(NONE≥FULL, §0.6). 단 이는 *in-distribution* — **미학습 도메인 zero-shot 전이는 LODO(G2)로만 측정**(아직). G2 LODO 전 base Qwen의 retail/airline B0 baseline은 비교군으로 별도 필요.
- **scorecard 도메인 특성**: telecom 가장 깨끗(F1 AUC 0.9). airline arg_bind는 복잡 중첩 params라 부분(0.6/0.69). retail/airline GT는 **read 제외·requestor=user 제외** 후 써야 정확(트레이너/scorer엔 반영됨).
- **GT actions = reward-aligned but soft order**(env-assertion 기반) → reward의 seq_F1 vs set-F1 비교 ablation(GRPO_REWARD_DESIGN §8).
- tau2 버전 일치(우리 baseline과). vLLM 0.11.0 권장.
