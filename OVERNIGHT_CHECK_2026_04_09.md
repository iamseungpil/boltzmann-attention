# 퇴근 후 체크 — 2026-04-09 저녁

## TL;DR
**Phase B Week 1 kill-switch** 검증을 위한 2개의 full 995 run 이 cuda:0/cuda:1 병렬 진행 중. 예상 완료 ~60-70분 후.

기 smoke 결과 (50 샘플 × 2 슬라이스 이미 확인):
- [0:50]: no_steer 70% → **ocq_bias α=0.3 88%** (+18pp) — determinism ✓
- [50:100]: no_steer 62% → **ocq_bias α=0.3 86%** (+24pp) — sample bias ✗ (signal even stronger on different slice)
- α curve smooth around 0.3 (0.28: +16, 0.30: +18, 0.32: +6)

**전체 995 sample kill-switch 결과가 이 두 run 에서 확정됩니다.**

## 실행 중인 background runs

### cuda:0 — **Primary kill-switch 검증** (byuch14z5)
- Script: `scripts/ocq/eval_metatool_subtask1.py`
- Model: Qwen/Qwen2.5-7B
- Samples: full 995
- Methods: `no_steer`, `ocq_bias_a0.2`, `ocq_bias_a0.25`, `ocq_bias_a0.3`, `ocq_bias_a0.35`, `ocq_bias_a0.4`
- B_ont: MetaTool catalog (28 × 4 × 128 × 24)
- Parser: fixed (earliest-position, not longest-first)
- Output: `/tmp/metatool_FULL995_alpha_sweep_cuda0.json`
- Log: `/tmp/claude-1002/-home-woori-workspace-common-boltzmann-attention/17bb0d10-2e1b-4f61-a6ea-239da93e416a/tasks/byuch14z5.output`
- 예상 런타임: ~70분 (6 methods × ~11분)

### cuda:1 — **High-α + quant ablation** (b33k75939)
- Script: `scripts/ocq/eval_metatool_subtask1.py`
- Model: Qwen/Qwen2.5-7B
- Samples: full 995
- Methods: `no_steer`, `ocq_bias_a1`, `ocq_bias_a3`, `ocq_quant` (4-bit), `ocq_quant_bias_a0.3`
- 목적: 
  - α=1 / α=3 이 parser 버그 영향이었는지, 진짜 catastrophic 인지 검증
  - ocq_quant 단독 (Claim B PPL-like) 이 얼마나 파괴적인지 full scale 로 확증
  - bias × quant 결합 시 OCQ dual-claim 이 살아남는지 확인
- Output: `/tmp/metatool_FULL995_ablations_cuda1.json`
- Log: `/tmp/claude-1002/-home-woori-workspace-common-boltzmann-attention/17bb0d10-2e1b-4f61-a6ea-239da93e416a/tasks/b33k75939.output`
- 예상 런타임: ~60분 (5 methods × ~12분; ocq_quant 은 bias 와 KIVI 둘 다 돌아서 느림)

## 결과 확인 방법

집에서 확인할 때:

```bash
# 완료 확인
tail -30 /tmp/metatool_FULL995_alpha_sweep_cuda0.log
tail -30 /tmp/metatool_FULL995_ablations_cuda1.log

# 또는 JSON 결과 직접
python -c "
import json
for f in ['/tmp/metatool_FULL995_alpha_sweep_cuda0.json',
          '/tmp/metatool_FULL995_ablations_cuda1.json']:
    try:
        d = json.load(open(f))
        print('=== ' + f + ' ===')
        for r in d['results']:
            print(f'  {r[\"method\"]:25s}  top1={r[\"top1_accuracy\"]*100:6.2f}%  '
                  f'({r[\"top1_correct\"]}/{r[\"n_queries\"]})  '
                  f'runtime={r[\"runtime_s\"]:.0f}s')
    except Exception as e:
        print(f'{f}: {e}')
"
```

## Kill-switch 해석 가이드

**`no_steer` 전체 995 top1 **:
- 예상치: 70-75% (smoke 에서 70% 와 62%)
- 극적으로 다르면 (e.g., 80% 이상 또는 50% 이하) 이상 징후

**`ocq_bias α=0.3` 전체 995 top1 Δ vs no_steer**:
- **≥3pp 양의 lift**: Phase B Week 1 kill-switch PASS → NeurIPS 경로 유지
- **0-3pp 양**: marginal, 추가 검증 필요 (α tuning, cross-model)
- **0 또는 음**: kill-switch FAIL → ICLR 2027 fallback 확정
- **+15pp 이상**: smoke 신호 확증, strong signal

**α curve 형태 (0.2, 0.25, 0.3, 0.35, 0.4 전체 995)**:
- Smooth unimodal (0.3 peak) → sweet spot 실존
- Non-monotonic noise → α calibration 필요
- Monotonic increasing → 더 큰 α 시도 값어치
- Monotonic decreasing → smaller α 시도

**α=1, α=3 (cuda:1)**:
- 여전히 catastrophic → parser 버그 아님, 원 관찰 확증
- 덜 catastrophic → parser 버그였을 수 있음, 재해석 필요

**ocq_quant 단독 (cuda:1)**:
- 50 샘플에서 38% (-32pp). 전체 995 에서 비슷하면 → **Claim B (KV compression via categorical) 는 tool selection 을 파괴함**, 단일 claim 논문 (Claim A only) 으로 축소
- 50% 이상으로 회복되면 → 하이브리드 재고려

**ocq_quant_bias (cuda:1)**:
- bias 가 quant 의 손실을 보상하면 → dual-claim 가능성
- 여전히 no_steer 이하면 → bias 와 quant 는 서로 상쇄하지 않는 독립적 메커니즘, 논문에서 trade-off 로 honest 하게 보고

## 이미 알려진 상황 (누적)

1. **Bug 1 fix** (fp16 fall-through), **Bug 2 fix** (pre-RoPE space mismatch) 모두 반영된 hook-mode eval driver (`scripts/ocq/eval_hook_mode.py`).
2. **origin/main 이 develop 와 divergence**: 99 파일 checkout 완료. iamseungpil 의 2026-04-09 CURRENT_STATUS 는 "Pre-RoPE PCA + WF f=2 SOTA 7.10/7.16/5.82" 를 **retract** ("2K eval artifact + Lloyd bug, 49K 재현 실패"). 새 paper 제목은 "The Rotation-Quantizer Interaction in KV Cache Compression" (negative-result paper).
3. **TurboQuant 2-bit = 10.47 PPL** 이 현재 코드베이스의 **실제 재현가능 2-bit 챔피언** (random rotation + Lloyd). 인용된 7.10 은 어떤 코드 경로로도 재현 불가.
4. **OCQ WT2 PPL 결과** (Qwen2.5-7B 4 windows smoke): ocq_kivi 2b=33.30, ocq_wf 2b=24.36, ocq_kivi 4b=15.48, ocq_wf 4b=15.42 — 모두 KIVI/TurboQuant 보다 나쁨. **Claim B (compression) 는 WT2 에서 지는 게 paper framing 과 일치** (WT2 는 unconditioned text, ontology resolved 안 됨).
5. **핵심 paper 전환**: Claim A (K-bias for tool selection) 이 유일하게 살아있는 claim. Claim B (quantization) 은 full 995 결과에 따라 drop 될 수 있음.

## 모든 관련 파일

### 새 코드 (develop 에 add, auto-committed)
- `scripts/ocq/quantizer.py` — OCQ + prior FOKVQ import hybrid
- `scripts/ocq/eval_hook_mode.py` — hook-mode WT2 PPL driver (5 method 지원)
- `scripts/ocq/eval_metatool_subtask1.py` — MetaTool eval driver (새로, 오늘)
- `scripts/ocq/build_metatool_ontology.py` — catalog → 4-facet ontology (ported from fokvq)
- `scripts/ocq/build_qwen_metatool_b_ont.py` — B_ont builder via k_proj hook (ported)

### Checkout from origin/main (2026-04-09 afternoon)
- `math/paper/lie_group/verify_v15_experiments.py` — SOTA compute_wf_alloc with min_bits parameter
- `math/paper/lie_group/verify_3axis_unified.py` — 3-axis verification script
- `paper/neurips2026_ko/` — latest paper draft tree
- `reports/CURRENT_STATUS_2026-04-09.md` — today's team status
- `reports/EXPERIMENT_PLAN_v23.md`, `EXPERIMENT_PLAN_v24.md` — latest plans
- `scripts/exp_correct_lloyd_vs_uniform.py`, `scripts/exp_linf_vs_l2.py`, `scripts/exp_pos0_diagnostic_v2.py`

### Updated 문서
- `reports/PHASE_B_PAPER_PLAN_v1.md` — retraction note 추가, baseline 재정의

### 신규 메모리 (persistence across sessions)
- `memory/feedback_decisive_rename.md`
- `memory/method_rename_ocq_2026_04_09.md`
- `memory/eval_arch_two_bugs_2026_04_09.md`
- `memory/pre_rope_proven_in_mse_not_ppl.md`
- `memory/origin_main_paper_pivot_2026_04_09.md`
- `memory/metatool_subtask1_first_signal_2026_04_09.md` — **kill-switch preliminary**

## 완료 후 다음 단계 (우선순위)

kill-switch 결과에 따라:

### PASS (+3pp 이상) 시나리오
1. **Cross-model 검증**: Llama-3.1-8B, Mistral-7B 용 B_ont 빌드 + 동일 eval (이 세션에서는 Qwen 만 테스트)
2. **α per-model calibration**: 모델별 sweet spot 이 다를 수 있음
3. **다른 prompt 변형**: `thought_prompt` 도 시도 (dataset 에 있음)
4. **PHASE_B_PAPER_PLAN 대대적 업데이트**: Claim A 단일 contribution 으로 재작성
5. **coworker 와 sync**: origin/main 의 rotation-quantizer 논문과 통합 검토

### FAIL (<3pp) 시나리오
1. **α tuning 심층**: 더 넓은 범위 (0.05, 0.1, 0.15, ..., 0.5)
2. **다른 B_ont 구성**: anchor sentence 확장, r_ont 조정
3. **SEKA-style targeted K-bias**: 현재는 모든 토큰에 global amplification, 특정 marker 위치만 적용해야 할 수도
4. **ICLR 2027 확정**: Path A fallback, Phase 1.x 결과를 main body 로 복귀

## 주의사항

- **α=0.3 +18pp 을 외부에 보고하지 말 것** full 995 확정 전
- 두 run 이 파일시스템에 결과 쓰는 중이므로 cleanup 하지 말 것
- cuda:0/1 둘 다 바쁠 예정이므로 다른 GPU 작업 launch 하지 말 것
- 다음 세션에서 "kill-switch 결과 확인" 으로 재개하면 이 파일 읽고 계속
