# GRPO Reward 설계 — Facet on-policy RFT (학습 사다리 ③, coworker B3)

> 참조 구현: `scripts/distill/grpo_reward.py` (검증됨). 지표 근거: `metric_mining.py` (F1 AUC 0.902).
> 정책 초기화: SFT 어댑터(`lora_train_chat_toolcall.py`, full 또는 none). 롤아웃: tau2 env + gpt-4.1 user_sim.

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
- **arg_bind** = entity 인자가 선행 read에서 바인딩된 비율(param_dataflow).
- 기본 가중치: `{pass:1.0, proc:0.5, extra:0.3, arg:0.1}` (pass 지배, proc는 shaping).

## 3. Reward hacking 방지
- **recall 아닌 seq_F1** 사용 → "모든 도구 호출"은 precision/seq_prec이 깎음.
- **extra_norm** → 엉뚱/잉여 호출 직접 패널티 (실패의 over-diagnosis 실측 1.27 vs 성공 0.63).
- **w_pass 지배** → 진짜 성공 > 어떤 부분점수. proc은 보조 shaping(λ=0.5<1).
- GRPO는 group-normalize → 절대 스케일보다 **롤아웃 간 순위**가 중요. seq_F1/extra/arg_bind가 pass=0 롤아웃들의 순위를 만든다.

## 4. 검증가능성 (PRM과의 차별)
process reward가 **tasks.json `evaluation_criteria.actions`(GT) 기반·결정적·LLM judge 불필요·무료**. 대부분 agent-PRM(LLM/MC judge)과 달리 reward hacking·judge noise 없음. (cf. §9.4.5 PRM 계열; 우리는 named-ontology GT-grounded.)

## 5. 통합 (B3, 4×A100, trl)
1. **정책 초기화** = SFT 어댑터. 두 arm: **full**(정책 prompt 유지) / **none**(내부화, prompt 없이 롤아웃 → 정책 흡수 검증). 
2. **롤아웃**: tau2 env(telecom train split 74) + user_sim=gpt-4.1(OpenRouter). G=8~16/prompt.
3. **reward**: 각 롤아웃 messages + task GT action names → `compute_reward(...)`. pass=env-assertion, process=seq_F1 등.
4. **trl GRPOTrainer**: 4×A100(policy + 롤아웃 서빙). 32B는 coworker, 7B는 Track A.
5. KL 정규화 to SFT 정책(드리프트 방지).

## 6. 일반화 변형 (cross-domain)
- in-domain: GT = tasks.json actions (가장 깨끗).
- GT 없는/새 도메인: **induced fault_fix_map + param_dataflow**(ABox)로 goal→tool 기대치 구성 → 동일 reward. = facet reward의 도메인-일반(전이) 버전.

## 7. Ablation & Go/No-Go
- **A. pass-only GRPO** (vanilla) vs **B. pass + seq_F1 shaped** (ours) vs **C. + extra/arg**. 
- 가설: B/C가 A보다 **수렴 빠름 + mms-chain(전부 pass=0이던 곳) lift** (dense가 sparse 구제).
- **Go**: shaped GRPO가 SFT 대비 pass^1 +Xp **그리고** mms-chain에서 vanilla-GRPO 대비 +≥5%p (dense reward 가치 입증). 
- 측정: `procedure_scorecard`(F1/seq_F1/recall) + pass^1, GRPO 전/후.

## 8. 주의
- process reward는 **shaping**(보조)이지 목적이 아님 — pass가 최종 판정. λ 과대 시 reward hacking 위험 → λ sweep.
- seq_F1은 GT 순서 가정 — tau2 reward는 env-assertion(end-state)이라 순서 일부 soft. order 비중은 seq보다 set(recall/precision)이 안전할 수 있음 → proc=F1(set) vs seq_F1 비교 ablation.
- none-arm 롤아웃: 정책 prompt 없이 → SFT-none이 충분히 내부화돼야 롤아웃이 무의미 붕괴 안 함. full-arm으로 먼저 검증 권장.
