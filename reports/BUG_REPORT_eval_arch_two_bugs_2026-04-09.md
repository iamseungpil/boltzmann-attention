# Bug Report — Two Eval-Architecture Bugs in `exp4_2_standard_ppl_benchmark.py`

**To**: `iamseungpil` + Codex (coworker session on `origin/main`)
**From**: `mais` (Claude session on `develop`)
**Date**: 2026-04-09
**Subject**: `scripts/fokvq/exp4_2_standard_ppl_benchmark.py`의 두 가지 eval architecture 버그 — KIVI/TurboQuant/OCQ 재현 숫자 전부 영향. 현재 rotation-quantizer negative-result 논문의 baseline 재검토 필요.
**Urgency**: High — 논문 main claim(10.47 TurboQuant 2-bit 재현 가능 champion, 7.10/7.16/5.82 retract 분석)에 직접 영향.
**Action requested**: 아래 두 버그 재현 확인 + 영향 범위 평가 + 필요 시 측정 재실행. Main 쪽 수정은 coworker가 진행. Develop 쪽에서는 이미 hook-mode eval로 우회 완료.

---

## TL;DR

`scripts/fokvq/exp4_2_standard_ppl_benchmark.py`의 **`quantize_cache` 경로가 두 개의 독립적인 버그를 품고 있음**:

1. **Bug 1 (fall-through)**: `stride == context_len` (KIVI non-overlap protocol)일 때 모든 window가 `prefix_len == 0` branch로 빠져 **quantization이 아예 호출되지 않음**. Qwen2.5-7B WT2 ctx=2048 non-overlap에서 모든 method가 동일한 7.684 PPL을 반환하는 증상으로 확인됨.
2. **Bug 2 (pre/post-RoPE 공간 불일치)**: `past_key_values`는 HF attention forward에서 **apply_rotary_pos_emb 이후에** `past_key_value.update()` 호출로 채워짐 → **post-RoPE K**. 반면 B_ont는 `k_proj.register_forward_hook`에서 build되어 **pre-RoPE K**. Post-RoPE K에 pre-RoPE basis를 적용하는 것은 수학적으로 잘못됨 (RoPE는 position-dependent rotation이므로 basis는 더 이상 orthonormal이 아니고 의미를 보존하지 않음).

**Impact**: `quantize_cache` 경로로 측정된 모든 숫자가 영향을 받음. 구체적으로:

- KIVI 재현 7.22 (Qwen2.5-7B WT2 ctx=2048)
- OCQ 7.43 ("KIVI 12% lower bits"로 보고되던 것)
- TurboQuant 2-bit = 10.47 (현재 "재현 가능 챔피언")
- Retract된 CWF 7.10/7.16/5.82의 "2K eval artifact + Lloyd bug" 원인 분석 — Bug 1이 합쳐졌을 가능성

**Develop side 조치**: 이미 hook-mode driver (`scripts/ocq/eval_hook_mode.py`)로 pre-RoPE K에 직접 개입하는 방식으로 우회. 신규 측정은 모두 hook-mode로만 진행. Main의 `quantize_cache` 경로는 건드리지 않음.

---

## Bug 1 — Fall-through at `prefix_len == 0`

### Location
`scripts/fokvq/exp4_2_standard_ppl_benchmark.py:918-931`

```python
for begin in range(0, input_ids.size(1), stride):
    end = min(begin + context_len, input_ids.size(1))
    window = input_ids[:, begin:end]
    if window.size(1) <= 1:
        continue

    trg_len = end - prev_end
    prefix_len = max(0, window.size(1) - trg_len)
    if prefix_len == 0:
        token_nll, num_tokens = score_full_window_suffix(model, window, trg_len)
        total_nll += token_nll
        total_tokens += num_tokens
        prev_end = end
        continue   # ← quantize_cache 호출 없이 다음 window로 이동

    prefix_ids = window[:, :prefix_len]
    target_ids = window[:, prefix_len:]

    prefix_outputs = model(prefix_ids, use_cache=True)
    quantized_cache, cache_stats = quantize_cache(
        prefix_outputs.past_key_values, method, ...
    )
```

### 문제 분석
KIVI의 공식 평가 protocol은 `stride = context_len` (non-overlap). 이 경우:

- 첫 window: `begin=0, end=context_len, prev_end=0` → `trg_len = context_len - 0 = context_len` → `prefix_len = context_len - context_len = 0` → **fall-through**
- 두 번째 window: `begin=context_len, end=2*context_len, prev_end=context_len` → `trg_len = 2*context_len - context_len = context_len` → `prefix_len = 0` → **fall-through**
- 모든 window가 같은 경로

즉, non-overlap protocol로 돌리면 어떤 `method` argument를 주든 `quantize_cache`가 한 번도 호출되지 않고 `score_full_window_suffix` (fp16 full-precision)만 돈다.

### 재현 증거 (2026-04-09 오전 develop에서 관찰)
Qwen2.5-7B / WT2 / ctx=2048 / stride=2048 (non-overlap):

| Method | PPL |
|---|---|
| `uniform` 2b | 7.684 |
| `variance` 2b | 7.684 |
| `kivi` 2b | 7.684 |
| `fokvq` 2b | 7.684 |
| `turboquant` 2b | 7.684 |

모든 method가 fp16 baseline과 bit-exact 동일 → quantization이 한 번도 호출되지 않음의 결정적 증거.

### 간이 fix (정답은 아님)
Non-overlap protocol을 지원하려면 `prefix_len == 0` branch에서도 `model(prefix_ids, ...)` → `quantize_cache` → forward with quantized prefix 로직이 필요. 즉, window 전체가 "target"이어도 첫 몇 토큰을 prefix로 잡아 cache quantization을 실제로 발생시키는 protocol 재정의 필요. **또는** overlap stride (e.g., `stride = context_len // 2`)로 강제해서 non-overlap eval 지원을 포기.

단, Bug 2가 해결되기 전에는 Bug 1 수정만으로는 쓸모 없음 (post-RoPE 공간에서 측정한 숫자는 여전히 의미 없음).

---

## Bug 2 — Pre-RoPE / Post-RoPE 공간 불일치

### Location
- `quantize_cache`의 basis 적용: `scripts/fokvq/exp4_2_standard_ppl_benchmark.py:730-828`
- 특히 line 787–810의 `oc_fokvq_*` branch가 `oc_bases_2a`/`oc_bases_2b`를 `legacy_cache` (post-RoPE)에 직접 적용
- B_ont builder: `scripts/fokvq/build_qwen_metatool_b_ont.py` (또는 `scripts/ocq/build_qwen_metatool_b_ont.py:156`) — `layer.self_attn.k_proj.register_forward_hook`으로 **pre-RoPE** K 수집

### 문제 분석
HF LlamaAttention / Qwen2Attention forward의 순서:

```python
# HF transformers/models/llama/modeling_llama.py (LlamaAttention.forward)
query_states = self.q_proj(hidden_states)        # pre-RoPE Q
key_states   = self.k_proj(hidden_states)        # ← 여기가 k_proj hook 지점 (pre-RoPE)
value_states = self.v_proj(hidden_states)

cos, sin = self.rotary_emb(value_states, position_ids)
query_states, key_states = apply_rotary_pos_emb(
    query_states, key_states, cos, sin
)                                                 # ← 여기서 post-RoPE로 변환

if past_key_value is not None:
    key_states, value_states = past_key_value.update(
        key_states, value_states, self.layer_idx
    )                                             # ← cache에 저장되는 것은 post-RoPE K
```

즉:
- `k_proj.register_forward_hook`에서 읽는 output은 **pre-RoPE K** (`self.k_proj(hidden_states)`의 출력)
- `past_key_values.to_legacy_cache()`가 반환하는 key tensor는 **post-RoPE K** (`apply_rotary_pos_emb` 이후 값)

B_ont 파일 (`external/SEKA/seka_projections/ontology-qwen25-7b-metatool/B_ont.pt`)은 첫 번째 경로에서 build되었고, `quantize_cache`는 두 번째 경로에서 그것을 꺼내 쓴다. 공간이 다름.

### 왜 post-RoPE K에 pre-RoPE basis를 적용하면 안 되나
RoPE는 각 channel pair를 position-dependent angle `θ_t`로 회전시킨다:

```
K_post_rope[t] = R(t) @ K_pre_rope[t]
```

여기서 `R(t)`는 block-diagonal rotation matrix (각 channel pair마다 `[[cosθ_t, -sinθ_t],[sinθ_t, cosθ_t]]`). 이 rotation은 position `t`마다 다르다.

Pre-RoPE 공간에서 build된 ontology basis `B_ont`는 "kpre-rope 분포의 주축"을 나타낸다. Post-RoPE K에 같은 `B_ont`를 투영하면:

```
B_ont.T @ K_post_rope[t] = B_ont.T @ R(t) @ K_pre_rope[t]
```

이는 `(R(t).T @ B_ont).T @ K_pre_rope[t]`와 같고, **position-dependent하게 회전된 basis로 pre-RoPE K를 측정하는 것과 동치**. position마다 다른 basis로 측정하니, "ontology 방향"이라는 해석이 무너진다. Categorical 1-bit / top-k 선택 같은 연산은 이 회전에 covariant하지 않다 (특히 coordinate-wise quantization은 회전 불변성이 없음).

실질적으로 `B_ont @ B_ont.T` projection은 pre-RoPE 공간에서는 low-rank projector였지만 post-RoPE 공간에서는 이미 orthonormal projector가 아니다 — RoPE rotation의 결과로.

KIVI 자체에도 같은 문제: KIVI paper는 "per-channel K quantization"을 pre-RoPE K에 적용. 각 channel이 일정한 semantic 단위 (e.g., query-position-independent token feature)를 carrying할 때만 per-channel quant가 의미 있음. Post-RoPE channel은 position-dependent linear combination이므로 per-channel quant가 망가진다. `exp4_2`의 `asymmetric_quantize_seq_dim` (line 753)도 post-RoPE K에 적용되고 있어 **KIVI 재현 숫자 7.22도 wrong-space 측정**.

### Fix 방향 (develop에서 이미 적용한 것)
`scripts/ocq/eval_hook_mode.py:396-423` 참고. 핵심 idea:

```python
for layer_idx, layer in enumerate(model.model.layers):
    k_proj = layer.self_attn.k_proj

    def make_hook(li):
        def hook(module, inputs, output):
            # output: (B, T, n_kv*head_dim), pre-RoPE
            B, T, D = output.shape
            k = output.view(B, T, n_kv, head_dim).permute(0, 2, 1, 3)
            k_q = quant_fn(k.float(), li).to(dtype=output.dtype)
            return k_q.permute(0, 2, 1, 3).view(B, T, D)
        return hook

    handles.append(k_proj.register_forward_hook(make_hook(layer_idx)))
```

- Hook을 `k_proj`의 forward output에 설치해 **pre-RoPE K를 바로 quantize 후 원래 위치에 return**
- 이후 HF의 `apply_rotary_pos_emb`가 이미-quantized-pre-RoPE K에 rotation을 적용
- Cache에는 이미-quantized-pre-RoPE-then-RoPE'd K가 저장됨
- `quantize_cache` 자체를 호출할 필요 없음 → Bug 1도 동시 회피

이 방식이면 `B_ont` 파일은 재빌드 필요 없이 그대로 사용 가능 (이미 pre-RoPE 공간).

---

## 논문 claim에 미치는 영향

### Direct impact
Rotation-Quantizer Interaction 논문의 main table이 `quantize_cache` 경로로 측정되었다면:

| 기존 보고된 숫자 | 상태 | 이유 |
|---|---|---|
| KIVI Qwen2.5-7B WT2 2b 7.22 | Invalid (wrong space) | Bug 2: post-RoPE K에 per-channel quant |
| TurboQuant 2b = 10.47 "champion" | Invalid (wrong space) | Bug 2: post-RoPE K에 rotated codebook |
| FOKVQ Lloyd variants | Invalid (wrong space) | Bug 2 |
| 7.10/7.16/5.82 "SOTA" retract 분석 ("2K eval artifact + Lloyd bug") | 원인 분석 불완전 | Bug 1이 합쳐진 효과일 수 있음 — ctx=2048 non-overlap에서 모든 method가 fp16 pass-through가 되므로 "2K에서만 보이던 이득"이 실은 "non-overlap eval에서 quant가 작동 안 하던" artifact일 가능성 |

### Indirect impact (methodology 차원)
- Paper의 "negative result" framing — "rotation 없는 quant는 rotation과 interact해서 망한다"는 결론은 **여전히 valid**할 수 있음. 단, 측정 수치가 잘못된 공간에서 나온 것이라 구체적 수치를 인용할 때는 "hook-mode 재측정 후"로 qualify해야 함.
- TurboQuant 2b "champion" claim은 hook-mode 재측정 전까지 보류 권장.

### Retract 재해석 제안
"2K eval artifact + Lloyd bug"로 retract된 7.10/7.16/5.82의 실제 원인 가설:

- **Hypothesis A** (기존): 2K context에서 특정 Lloyd initialization이 numerical artifact로 낮은 PPL을 생성했다.
- **Hypothesis B** (신규): Non-overlap eval 시 Bug 1 fall-through로 해당 method들이 fp16 pass-through로 평가되었는데, longer context에서는 stride가 달라져 Bug 1이 발현 안 되어 진짜 method가 돌아 고 PPL이 나왔다.
- **Hypothesis C**: A + B의 조합.

어느 쪽이 맞는지는 `exp4_2`의 해당 재현 run을 stride 다양화하면서 재측정해보면 판별 가능.

---

## Fix reference (develop side, cherry-pick 가능)

develop branch에 다음 5개 파일이 존재하고 main에는 없음:

```
scripts/ocq/eval_hook_mode.py         (780 lines)  — hook-mode PPL eval, 5 methods
scripts/ocq/quantizer.py              (644 lines)  — OCQ + legacy FOKVQ hybrid
scripts/ocq/build_qwen_metatool_b_ont.py (278 lines) — pre-RoPE B_ont builder (k_proj hook)
scripts/ocq/build_metatool_ontology.py (404 lines)
scripts/ocq/eval_metatool_subtask1.py  (514 lines)  — MetaTool tool-selection driver
```

필요 시 `git checkout develop -- scripts/ocq/eval_hook_mode.py scripts/ocq/quantizer.py scripts/ocq/build_qwen_metatool_b_ont.py` 정도만 가져와도 main에서 hook-mode로 재측정 가능. `external/SEKA/` 전체 tree도 develop-only이므로 B_ont 파일 경로만 맞춰주면 됨.

단, OCQ code에는 `scripts/ocq/quantizer.py`가 `kivi`, `turboquant`, `uniform` 등을 다 지원하므로 rotation-quantizer 논문의 모든 baseline을 이 driver로 재측정 가능 (굳이 `exp4_2`를 고칠 필요 없이 driver 교체).

---

## Develop side에서 하고 있는 일 / 안 하는 일

### 하고 있는 것
- `scripts/ocq/eval_hook_mode.py`로 모든 신규 PPL 측정 — 이미 Qwen2.5-7B WT2 4 windows smoke로 OCQ 변종들 PPL 재측정 완료 (`memory/ocq_real_ontology_validation_2026_04_09.md` + `memory/eval_arch_two_bugs_2026_04_09.md`).
- Phase B pivot: PPL metric 자체를 main claim에서 제외. Tool selection (MetaTool Subtask1) 이 primary evaluation. 현재 full 995 run 진행 중 (cuda:0/1 병렬, ~60분 ETA).

### 안 하는 것
- **Main의 `exp4_2_standard_ppl_benchmark.py`를 우리가 직접 수정하지 않음.** Main은 coworker의 논문 작성 영역이라 건드리면 merge conflict 및 in-flight 수정과 충돌 위험.
- Main의 `reports/CURRENT_STATUS_2026-04-09.md` 및 retract 문서는 수정하지 않음. 본 bug report는 `develop/reports/`에 저장되어 merge 경로를 따로 타지 않음.
- **Bug 1 / Bug 2 발견 전 보고된 develop-side OCQ 숫자도 모두 invalid 처리.** 해당 메모리 파일에 상태 플래그 추가됨 (`memory/eval_arch_two_bugs_2026_04_09.md` 참조).

---

## Requested action

1. **Bug 1 확인**: `exp4_2_standard_ppl_benchmark.py`를 `stride=ctx_len` non-overlap protocol로 돌려 모든 method가 동일 PPL을 반환하는지 확인. (재현 data point: Qwen2.5-7B WT2 ctx=2048 → 7.684 모두 동일)
2. **Bug 2 확인**: `past_key_values.to_legacy_cache()[0][0]` 에서 꺼낸 K tensor가 post-RoPE인지 확인. 방법: `model.model.layers[0].self_attn.k_proj`에 hook 설치해서 그 output과 `past_key_values` 의 layer 0 key를 동일 forward pass에서 비교. 다르면 post-RoPE 확정.
3. **논문 숫자 재측정 결정**: hook-mode driver로 main table을 재측정할지, 또는 "measurement caveat" 형태로 paper에 qualification을 추가할지 결정. 어느 쪽이든 원 측정 경로는 retract 대상.
4. **Retract 원인 재분석**: 7.10/7.16/5.82 retract의 원인을 Bug 1 + 2K eval artifact + Lloyd bug의 조합 가설로 재검토.
5. **develop 쪽 hook-mode driver 재사용 여부**: `scripts/ocq/` 파일들을 main으로 cherry-pick할지 결정. 원하면 dev session에서 해당 파일들을 main-compatible하게 refactor하는 것 가능.

---

## How to send this report

이 파일 자체는 `develop` branch의 `reports/` 디렉토리에 저장됨. Coworker에게 전달하는 방법:

**Option A — Commit + 공유 경로 pointer**
```bash
git add reports/BUG_REPORT_eval_arch_two_bugs_2026-04-09.md
git commit -m "Bug report: two eval-architecture bugs in exp4_2 quantize_cache path"
git push origin develop
```
그리고 coworker에게 메시지: "develop branch `reports/BUG_REPORT_eval_arch_two_bugs_2026-04-09.md` 확인 부탁드립니다. `exp4_2_standard_ppl_benchmark.py`의 두 가지 eval architecture 버그 리포트입니다."

**Option B — 파일만 공유 (git push 없이)**
`reports/BUG_REPORT_eval_arch_two_bugs_2026-04-09.md` 파일을 이메일/Slack에 첨부.

**Option C — Cherry-pick branch 만들어 PR**
```bash
git checkout -b bug-report/eval-arch-bugs main
git checkout develop -- reports/BUG_REPORT_eval_arch_two_bugs_2026-04-09.md
git commit -m "docs: bug report on eval architecture bugs in exp4_2"
git push origin bug-report/eval-arch-bugs
# GitHub에서 PR 생성 → main으로 merge 요청 (fix는 coworker가, 리포트만 먼저 merge)
```

Option A가 가장 가볍고 coordination 비용 낮음. Option C는 bug report가 main에도 visible해지는 장점이 있지만 PR 리뷰 round-trip이 추가됨.

---

## References

- Develop memory: `memory/eval_arch_two_bugs_2026_04_09.md` — 본 bug report의 원본 분석 메모
- Develop memory: `memory/origin_main_paper_pivot_2026_04_09.md` — origin/main의 negative-result 논문 pivot 맥락
- Develop memory: `memory/pre_rope_proven_in_mse_not_ppl.md` — pre-RoPE / post-RoPE 관련 이론적 배경 (MSE optimality만 증명되어 있고 PPL 직접 비교는 없음)
- Develop code: `scripts/ocq/eval_hook_mode.py` — hook-mode PPL driver 레퍼런스 구현
- Develop code: `scripts/ocq/build_qwen_metatool_b_ont.py:156` — pre-RoPE B_ont builder의 hook 지점
