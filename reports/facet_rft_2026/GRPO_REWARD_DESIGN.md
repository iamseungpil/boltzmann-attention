# GRPO Reward 설계 — Facet on-policy RFT (학습 사다리 ③, coworker B3)

> 참조 구현: `scripts/distill/grpo_reward.py` (검증됨). 맥락: `EXPERIMENT_DESIGN_v1_7_facet_rft.md §13(방향)·§14(구현·측정)`, 핸드오프 `project_distillation_handoff_2026_05_30`.
> 정책 초기화: SFT 어댑터(`lora_train_chat_toolcall.py`, full 또는 none). 롤아웃: tau2 env + gpt-4.1 user_sim.

**지표 근거 (metric_mining AUC 랭킹, succ>fail 확률; reward 구성의 실측 토대)**: **F1 0.902 / seq_F1 0.89~0.99**(최강, recall·precision 통합) · recall 0.88 · precision 0.86 · superset 0.88 · **extra_actions(over-diagnosis) 실패 1.27 vs 성공 0.63** · arg_bind은 teacher 포화(0.99)·**student 약점 노출(0.32~0.73)**. **3 도메인 일반화**(telecom/retail/airline). → reward = 이 변별축들의 가중합.

## 1. 동기 — sparse cold-start
7B student는 pass^1 ≈ 0.18. GRPO는 프롬프트당 G개 롤아웃을 뽑아 **advantage = (r − group_mean)/std** 로 학습하는데, pass/fail(0/1)만 쓰면 어려운 task에서 **G개 전부 pass=0 → reward 전부 0 → advantage 0 → 학습 신호 없음** (mms hard tail에서 특히). 

**해결**: GT-action 기반 **dense process reward(seq_F1)** 를 더한다. 실측(`grpo_reward.py` sanity, shipped telecom): **실패 롤아웃들 사이 seq_F1 = 0.255 ± 0.291 (std>0)** → all-fail group도 롤아웃이 구분되어 **advantage 비0 → gradient 발생**. success reward 1.384 vs failure 0.027로 분리.

## 2. Reward 함수

```
r(rollout) = w_pass · pass              # 터미널 env reward(0/1) — 지배항
           + w_proc · seq_F1            # dense goal→tool 일치(recall·순서·minimality)
           − w_extra · extra_norm       # over-diagnosis 패널티(minimality)
           + w_arg  · arg_bind          # entity 인자 바인딩(student 품질)
```
- **seq_F1** = harmonic(seq_prec, seq_match), LCS 기반 순서-aware (metric_mining에서 AUC 0.89~0.99, F1과 동급 최강).
- **extra_norm** = (필수 아닌 action 호출 수)/(|GT|+1).
- **arg_bind** = entity 인자가 선행 read에서 바인딩된 비율(param_dataflow). **teacher 포화·student 약점 신호**이므로 **학습 초반 가중↑, 후반 anneal↓**(student가 ID 할루시네이션 졸업하면 saturate→무신호).
- 기본 가중치: `{pass:1.0, proc:0.5, extra:0.3, arg:0.1}` (pass 지배, proc는 shaping).
- **★GT actions 추출 정정**(grpo_reward·scorecard 반영): GT = `evaluation_criteria.actions` 중 **read(get_*) 제외 + `requestor=='user'` 제외**(telecom dual-control의 user 행동·정보조회 read는 agent-필수 아님). 안 거르면 recall/seq_F1 과소·왜곡.

## 3. Reward hacking 방지
- **recall 아닌 seq_F1** 사용 → "모든 도구 호출"은 precision/seq_prec이 깎음.
- **extra_norm** → 엉뚱/잉여 호출 직접 패널티 (실패의 over-diagnosis 실측 1.27 vs 성공 0.63).
- **w_pass 지배** → 진짜 성공 > 어떤 부분점수. proc은 보조 shaping(λ=0.5<1).
- GRPO는 group-normalize → 절대 스케일보다 **롤아웃 간 순위**가 중요. seq_F1/extra/arg_bind가 pass=0 롤아웃들의 순위를 만든다.

## 4. 검증가능성 (PRM과의 차별)
process reward가 **tasks.json `evaluation_criteria.actions`(GT) 기반·결정적·LLM judge 불필요·무료**. 대부분 agent-PRM(LLM/MC judge)과 달리 reward hacking·judge noise 없음. (cf. §9.4.5 PRM 계열; 우리는 named-ontology GT-grounded.)

## 5. 통합 (B3, 4×A100)
1. **정책 초기화** = SFT 어댑터(multi-domain plain, telecom462+retail831+airline246). 두 arm: **full**(정책 prompt 유지) / **none**(내부화, prompt 없이 롤아웃). 
2. **롤아웃**: tau2 env(train split) + user_sim=gpt-4.1(OpenRouter). G=8~16/prompt.
3. **reward**: 각 롤아웃 messages + task GT action names → `compute_reward(...)`. pass=env-assertion, process=seq_F1 등. **★airline 등 reward_basis=nl_assertions 도메인은 judge가 OpenRouter로 라우팅돼야 pass가 0이 안 됨**(`phase1_runner._route_nl_judge_via_openrouter`, commit 7530d14; bare `gpt-4.1-2025-04-14`→OPENAI_API_KEY 필요 버그).
4. **★trl 설치 불가**(seka_env transformers 4.51.3과 모든 trl 버전이 다운/업그레이드 충돌) → **수동 GRPO 루프**(policy=SFT+trainable LoRA, ref=frozen SFT, advantage=group-normalize, KL to ref). DPOTrainer/GRPOTrainer 미사용.
5. KL 정규화 to SFT 정책(드리프트 방지).

## 6. 일반화 변형 (cross-domain)
- in-domain: GT = tasks.json actions(read·user 제외) — 가장 깨끗. **3 도메인(telecom/retail/airline) 모두 scorecard 변별 확인**(seq_F1 disc +0.22~0.63) → reward 동일 적용 가능.
- GT 없는/새 도메인: **induced fault_fix_map + param_dataflow**(ABox)로 goal→tool 기대치 구성 → 동일 reward. = facet reward의 도메인-일반(전이) 버전.
- 단 airline은 다수 task가 write-action 0개(nl-assertion 평가)·arg_bind 부분(복잡 중첩 params) → reward proc 비중·축 도메인별 조정.

## 7. Ablation & Go/No-Go
- **A. pass-only GRPO** (vanilla) vs **B. pass + seq_F1 shaped** (ours) vs **C. + extra/arg**. 
- 가설: B/C가 A보다 **수렴 빠름 + mms-chain(전부 pass=0이던 곳) lift** (dense가 sparse 구제).
- **Go**: shaped GRPO가 SFT 대비 pass^1 +Xp **그리고** mms-chain에서 vanilla-GRPO 대비 +≥5%p (dense reward 가치 입증). 
- 측정: `procedure_scorecard`(F1/seq_F1/recall) + pass^1, GRPO 전/후.

## 8. 주의
- process reward는 **shaping**(보조)이지 목적이 아님 — pass가 최종 판정. λ 과대 시 reward hacking 위험 → λ sweep.
- seq_F1은 GT 순서 가정 — tau2 reward는 env-assertion(end-state)이라 순서 일부 soft. **F1(set) AUC 0.902 vs seq_F1 0.89~0.99로 거의 동급** → proc=F1(set) vs seq_F1 ablation으로 안전한 쪽 선택(multi-action 도메인은 seq_F1이, telecom 단일-action은 set-F1이 유리할 수 있음).
- none-arm 롤아웃: 정책 prompt 없이 → SFT-none이 충분히 내부화돼야 롤아웃이 무의미 붕괴 안 함. **검증됨**: NONE eval이 3도메인 모두 FULL≥(telecom .35/.30, retail ~.82/.67, airline .40/.30)라 none-arm 롤아웃 안전.

## 9. ★v1.27 — Group J reward 항 + SFT 실측 + a→b→c 사다리
**SFT 실측(3도메인 held-out test, multi-domain SFT)**: NONE≥FULL 전부(위). 단 **NONE 실패 19건 분해**(`analyze_none_failures.py`): **63% recall-miss/anti-loop**(fix tool 미발화·max_steps 루프), wrong-tool 고착(send_payment 스팸), escalation 오타이밍. = imitation 분포이동 한계 → **순수 SFT로 못 닫음, reward shaping 필요**.

**Group J(§15.4 design doc) 기반 reward 증강** — 도메인무관, ABox는 `induced/tbox_relations_<domain>.json`(inducer 산출):
```
r += w_repair · repairs_state_recall      # 막힌 상태→fix 도구를 실제로 불렀나 (recall-miss 직접 공략)
   − w_loop   · step_penalty              # 진단 루프(max_steps 도달·과다 read) 패널티 (anti-commitment)
   − w_distr  · distractor_hit            # distractor_for 오답 도구 호출 시 패널티 (wrong-tool 고착)
   (+ diagnosis_sufficient_for 충족 후 write 미발화 시 추가 패널티 — commitment)
```
- 전부 **GT/induced 기반 결정적·도메인무관**(LLM judge 불요, §4 검증가능성 유지).
- 기존 `{pass1.0/proc0.5/extra0.3/arg0.1}`에 `{repair, loop, distr}` 추가, λ sweep로 anti-hacking.

**학습 사다리 a→b→c** (SFT floor→offline→on-policy):
- **(a) SFT** = floor(완료, NONE≥FULL).
- **(b) offline DPO**: `build_dpo_dataset.py` 1171 preference pairs(chosen=GT fix / rejected=distractor_for). wrong-tool 고착 공략. **trl 불가→수동 DPO**(ref=frozen SFT).
- **(c) GRPO**: 위 Group J reward로 anti-loop residual. 수동 루프(trl 불가).
- **전이 검증 = LODO**(§15.6): reward·ABox가 도메인무관이므로 미학습 도메인 전이 측정.
