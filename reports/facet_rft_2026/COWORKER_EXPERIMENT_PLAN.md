# Coworker 실험 계획서 — Facet Distillation (goal→tool 절차 내부화)

> 대상: 4× A100 80GB 보유 coworker. 공유 채널 = GitHub `iamseungpil/boltzmann-attention` branch **`facet-rft-2026`**.
> 본 계획은 `reports/EXPERIMENT_DESIGN_v1_7_facet_rft.md` **§13 (v1.25)** 방향을 구현한다. 먼저 §13을 읽을 것.

---

## 0. TL;DR — 역할 분담

| Track | 담당 | GPU | 핵심 역할 |
|---|---|---|---|
| **A (우리)** | woori box | A6000 ×2 | 데이터 파이프라인(완료)·7B 학습·trainer/scorer 제작·WISE-Flow baseline·fault→fix 정제 |
| **B (coworker)** | A100 box | **A100 ×4 (80GB)** | **32B 학습 · cross-domain eval 매트릭스(대규모) · on-policy GRPO · capability-ceiling probe** |

**한 줄 목표**: teacher 궤적에서 **goal→tool 선택 절차**를 student 가중치에 distill해, (1) student의 fix-coverage를 0.06 → 높게 끌어올리고, (2) **재학습 없이 held-out 도메인으로 전이**되며, (3) full-prompt 대비 토큰/KV/latency를 줄이는지 검증.

---

## 1. 배경 (요약)
- tau2-bench 도구 에이전트. 큰/작은 모델 격차는 **거의 전부 절차(distillable)** 임이 실측됨: Qwen-7B telecom 실패의 **도구선택 58% + 형식 36% = 94%**, long-horizon capability 벽은 1%. student **fix-coverage = 0.06**(올바른 fix-tool을 거의 안 부름).
- **목표→도구(positive)는 성공/실패를 강하게 변별**(disc +0.36, FULL-coverage +48%p). 제약(precedes/mutex)은 변별 0(폐기).
- 따라서 distill 타깃 = **goal→tool 선택 절차**. (상세·문헌 근거: §13)

---

## 2. 검증 가설 & 지표

**Thesis**: contrastive 유도 goal-conditioned 선택 절차를 가중치에 distill(=steering이 못한 matrix 성분), goal→도구-역할 추상화로 일반화, 도메인 인스턴스(ABox)는 swap → cross-domain 전이.

**지표**
- **fix-coverage** (핵심): 태스크의 각 결함에 대해 student가 올바른 fix-tool을 불렀는가 (baseline 0.06). scorer = Track A 제공(`scripts/distill/score_fix_coverage.py`).
- **pass^1** (tau2 test split, 도메인별).
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
- **변형**: ① plain(1539) ② facet_L1(ontology-clean) — plain vs facet confound 격리.
- **출력**: adapter(`lora_adapters/qwen32b_{plain,facetL1}/`) + `train_meta.json`. adapter는 **HF private model repo로 공유**(repo push 금지, GB급).
- **성공기준**: 수렴(val loss↓), 실제 lift는 B2에서 측정.

### B2. Cross-domain transfer EVAL 매트릭스  ★대규모 병렬 (coworker 핵심 기여)
- **목적**: 학습된 student를 tau2 **test split**에서 평가하고 전이 매트릭스를 채움.
- **GPU**: vLLM 서빙(adapter 머지 후) 1 GPU/모델 + 다중 에피소드 병렬. 4 GPU면 2~3 모델 동시 + 에피소드 fan-out.
- **매트릭스**: {7B(Track A), 32B(B1)} × {baseline, plain-SFT, facet-SFT} × {telecom, retail, airline} test split. user_sim = **`openai/gpt-4.1` via OpenRouter** (키 공유; `OPENAI_BASE_URL`=openrouter 필수).
- **러너**: `scripts/phase1_runner.py` (repo; 패치됨 — `--agent-llm`=로컬 vLLM, `--user-llm`, `--agent-api-key`). **주의(기존 버그)**: 도메인별 `--task-set <domain>` 필수(기본 telecom), airline judge는 `OPENAI_BASE_URL`/`OPENAI_API_BASE`=openrouter 설정.
- **출력**: `reports/facet_rft_2026/phase4_distill/coworker_a100/eval/<run>/results.json` + fix-coverage/pass^1 manifest.
- **성공기준**: G1/G2/G3 (아래 §8).

### B3. On-policy GRPO (학습 사다리 ③) — 조건부
- **목적**: SFT가 못 메운 mms residual을 student 탐색+reward로 보강.
- **GPU**: 4× A100 (policy + rollout 서빙). trl 설치 필요(미설치).
- **reward**: pass/fail + (옵션) goal→tool process reward(fix-tool 호출 시 +) + ontology-violation penalty. 
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
#   학습: torch, peft, transformers, accelerate, (B3) trl
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
| trainer / scorer / WISE-Flow | GitHub (Track A 제작·push) | `scripts/distill/` |
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
| **W1** | 셋업 + **B1 32B plain-SFT** | chat-SFT trainer + fix-coverage scorer 제작·전달, 7B plain-SFT, fault→fix 정제 |
| **W2** | **B2 eval 매트릭스**(7B+32B × plain/facet × 3도메인) + B1 facet-SFT | WISE-Flow baseline, 7B facet-SFT, efficiency 측정 |
| **W3** | **B3 GRPO**(조건부) + **B4 capability probe** | 결과 종합, 논문 표/그림 |

GPU 할당 예시(W2): 2 GPU 학습 + 2 GPU eval 병렬.

---

## 8. Go/No-Go 게이트

| 게이트 | 기준 | 판단 |
|---|---|---|
| **G1 (SFT lift)** | fix-coverage 0.06 → **≥0.50 (7B), ≥0.60 (32B)**; pass^1 ≥ baseline | 진행 / 데이터·trainer 점검 |
| **G2 (cross-domain 전이)** | telecom+retail 학습 → **airline held-out +≥5%p** vs baseline | thesis 핵심 양성 / ABox-swap 재설계 |
| **G3 (facet vs plain)** | facet가 plain 대비 **+≥3%p** | facet 기여 확정 / 없으면 **facet 폐기, plain만** |
| **G4 (GRPO, 조건부)** | mms-chain **+≥5%p** | capability 잔차 보강 / capability 벽 보고 |

---

## 9. 핵심 리스크 / 주의
- **fault→fix ground-truth 품질**(SkillFlow 교훈: 병목은 검색 아닌 라이브러리 품질). Track A가 단일결함/인과귀속으로 정제 후 scorer에 반영.
- **cross-domain = 분포 이동 → 도메인특수성 내부화 시 취약**(Transmuting 100→42.7 사례). 전이 평가에서 "불변 절차만 내부화" ablation 확인.
- **user_sim 비용**: eval은 OpenRouter gpt-4.1 호출(과금). test split N≈40/도메인이라 관리가능하나 매트릭스 셀 수 × N 주의.
- tau2 버전 일치(우리 baseline과). vLLM 0.11.0 권장.
