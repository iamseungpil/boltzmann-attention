# Coworker 재요청 (2026-06-18) — Lane2를 **실 τ² user-sim e2e**로 (오프라인 op-eval 폐기)

> 자기완결. 직전 결과 = `COWORKER_RESULTS_2026_06_17_scale.md`(Lane1 depth + Lane2 multidomain). 권위 = `scripts/distill/HANDOFF_2026_06_18_PM.md §0·§3`(오프라인 op-eval 폐기)·`ma/M_A_RESULTS.md §29-34`(철회)·`INTEGRATED_SCAFFOLD_IMPL_DESIGN_2026_06_18.md (v2)`.
> 불변: 추론-only(학습 0)·temp0·결정론 보상·도메인-타깃 금지·COST GUARD(user-sim=gpt-4.1·Claude 금지).

## 0. TL;DR — 무엇을·왜 다시
- **Lane1(B-budget in-head depth·synth)** = ✅ **신뢰·유지**(synth 통제 벤치·재측정 불요). 결론(in-head 235B도 깊은 연산 못 넘음·op-IR+엔진 1.0=offload 지배) 그대로 박제.
- **Lane2(multidomain content-routing·retail/airline)** = ⚠️ **오프라인 op-eval(`tau2_op_eval`) 기반 → 신뢰불가 → 실 τ² user-sim e2e로 재측정 요청.**
- **왜 다시**: (1) 오프라인 op-eval이 실 τ²를 재현 못 함(아래 §1). (2) woori가 **단독 통합 LoRA를 실 user-sim e2e로** 측정 중 → coworker scale도 **같은 축(실 e2e)**이라야 "소형+학습 vs 대형 base" 비교가 성립.

---

## 1. ★왜 기존 Lane2가 신뢰불가인가 (오프라인 op-eval의 결함 — 자세히)

**기존 측정(`tau2_op_eval.py --native`)** = 모델에 `[τ² NL + variant_catalog + ABox ordinal config]`를 **프롬프트로 한 번에 직접 주고**, 단발로 `resolve_selection`(op-IR)을 emit시켜 → resolver → `new_item_id`가 gold와 같은지 채점.

이게 **실제 τ² 벤치와 5가지로 체계적 불일치**:

| # | 축 | 오프라인 op-eval | 실제 τ² | 결과 |
|---|---|---|---|---|
| 1 | **정보 제공** | task 정보를 프롬프트로 한 번에 주입 | **user-sim이 `task_instructions`를 대화로 점진 공개**·agent가 물어서 발굴 | 오프라인은 정보 과다/과소(HANDOFF §0: reason_for_call만=부족·전체=노이즈·둘 다 불일치) |
| 2 | **catalog 출처** | `variant_catalog`를 프롬프트로 제공 | agent가 `get_product_details`를 **fetch해야** catalog 획득(못 하면 없음) | 오프라인은 fetch-first(R1b)·grounding 부담 제거 |
| 3 | **턴 수** | 1-shot op emit | 인증→fetch→confirm→exchange **멀티스텝 궤적**(한 스텝 실패=전체 fail) | 오프라인은 멀티턴 누적 실패 미반영 |
| 4 | **채점** | `new_item_id == gold` (content slot 1개만) | **최종 DB state ∧ NL-assertion ∧ communication**(compliant-pass·task 전체) | 오프라인은 "단발 슬롯 정확도"지 "task 성공률"이 아님 |
| 5 | **정책 게이트** | 없음 | auth-first·write-confirm·order-ownership 등 **gate 통과 필수** | 오프라인은 gate/provenance/인증 0 |

⇒ **오프라인 수치(retail 0.44·airline 0.74)는 "content slot 단발 정확도"이지 "실 τ² task 성공률"이 전혀 아니다.** 둘은 체계적으로 다름 — 실측 대조: **base 7B 실 user-sim e2e retail pass^1 = 0.205**(woori 측정·신뢰) vs 같은 base의 오프라인 retail op-eval ≈ 0.34. 오프라인이 실 e2e보다 *높게* 나온다(멀티턴·gate·fetch 부담이 빠져서). 이 괴리 때문에 HANDOFF가 §29-34(오프라인 op-eval 기반 τ² 분석)를 **전부 철회**했다.

**핵심**: 같은 결함을 coworker Lane2도 그대로 안고 있다(같은 `tau2_op_eval`). 그래서 Lane2 절대 수치는 인용 불가. 단 *정성적* 결론("큰 base도 retail set 추출 못 올림")만 §20-B와 정합해 참고 가치.

---

## 2. 신뢰 대체 = 실 τ² user-sim e2e (`t2_run_gated.py`)
- **agent** = base 32B/72B serve(TP) → 실 τ² 멀티턴(인증·fetch·confirm·write).
- **user-sim** = `openrouter/openai/gpt-4.1`·temp0 (COST GUARD: Claude 금지·키는 coworker 환경).
- **보상** = 결정론(τ² DB state ∧ NL-assertion ∧ communication·`t2_compliance` compliant-pass). LLM-judge 보상 아님.
- **측정** = `pass^1` (+ `pass^k` for num_trials≥2). full tasks(retail 114·airline 50) 또는 동일 `num_tasks N`.
- **= woori 7B 단독 LoRA 실 e2e와 같은 측정축** → "소형(7B)+학습+offload vs 대형(32/72B) base" 직접 비교.

---

## 3. 실행 (도구 = repo·`git pull --ff-only`)
```bash
# 전제: tau2-bench 클론 + openrouter key(coworker 환경·user-sim=gpt-4.1).
#   REPO/scripts/distill/tau2/t2_run_gated.py (재사용). PYTHONPATH=src:$REPO/scripts/distill/tau2.
cd $TAU2_BENCH && export PYTHONPATH=src:$REPO/scripts/distill/tau2
# serve 32B (TP2) → retail + airline
PY $REPO/scripts/distill/tau2/t2_run_gated.py --domain retail  --num_trials 2 --num_tasks 0 \
   --agent_model "Qwen/Qwen2.5-32B-Instruct" --agent_base http://localhost:PORT/v1 \
   --user_llm "openrouter/openai/gpt-4.1" --user_temp 0.0 --save_to scale_32B_retail
PY ... --domain airline ... --save_to scale_32B_airline
# serve 72B (TP4) → 동일 (scale_72B_*)
```
- `num_tasks 0`=전체(retail 114·airline 50). num_trials≥2(분산↓).
- **base는 순수 e2e**(resolve_selection offload는 base엔 무의미 — woori smoke 실측: base는 resolve_selection을 **0회 호출**. offload+학습은 woori 7B arm 몫). 따라서 `--resolve`·`--gate`는 **off**로 순수 base 측정(원하면 `--gate 1` arm도 별도).
- **gloss(floor/ceiling) drop**: gloss는 오프라인 프롬프트 변형 아티팩트 — 실 e2e는 도구/프롬프트 고정이라 무의미.

## 4. 회수 (woori 집계)
- per-(크기 × 도메인): `pass^1`·`pass^k`·`mean_reward`·`n` (`t2_run_gated` RESULT 라인 + `compliance.json`).
- autopsy: fail 원인 분류(order_id 날조 / gate-deny / set 오류 / 무한루프) — `tau2_autopsy.py` 재사용 가능.
- 모델 메타(L·d·params) 동봉. 235B는 기존 3겹 별표(Qwen3·MoE·FP8) 유지.

## 5. 규율 / caveat
- temp0·GPU 분리·serve 전 GPU kill·port/log 분리(`reference-remote-server-environment`).
- **COST GUARD**: user-sim=gpt-4.1 전용. `t2_run_gated`가 Claude 모델 거부(--allow-frontier 없이). 키는 coworker 자기 환경(시크릿 커밋 금지).
- ★**측정 분산**: 실 e2e는 agent 비결정으로 run-to-run **±2-3 pass(±0.05)**(woori 전수확정·동일조건 11/40 flip). → **num_trials≥2-4(pass^k)** 필수·작은 Δ는 노이즈.
- 동일 `num_tasks`·태스크 set(크기간 비교 유효). 프롬프트/도구 무편집.

## 6. 왜 이게 더 가치있나 (다시 하는 본질)
- Lane2 원질문 = "스케일이 성분 B(multi-attr set 추출)를 푸나" + "**소형+학습 > 대형+gloss**". 이건 **실 e2e task 성공률**로만 판정 가능 — 오프라인 op-eval은 content slot 단발이라 멀티턴·gate·fetch가 빠져 *질문에 답을 못 함*.
- woori 단독 통합 LoRA(실 e2e·진행 중)와 **같은 축**으로 비교 → thesis 헤드라인("소형 on-prem LLM + 학습 + 결정론 offload가 대형 base tool-use에 도달")을 **실 벤치에서** 실증. 오프라인 수치로는 이 주장을 못 한다(리뷰어가 "실 벤치 아니다"로 침).
- = HANDOFF의 핵심 교훈("실 user-sim e2e만 신뢰")을 coworker lane에도 적용.
