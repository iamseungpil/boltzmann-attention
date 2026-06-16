# M_A scale_floor — 32B-bf16 local result (2026-06-16, A100 80GB TP1)
Pre-flight GATES 1-5 PASS (tau2-bench cloned, 29 cases, resolver 4/4, 7B smoke 14/14 real records).
Full 29x7 sweep, base inference, no SFT. Output: ma_eval_Qwen2_5_32B_Instruct_bf16.jsonl (203 records).

## Verdicts (item-acc, 29 cases / 32 items, ±~6pp noise)
- Q1 HEADLINE = REASONING-FLOOR (not Int8-cap): 32B-bf16 A = 0.719 == 32B-Int8 A = 0.719 == 14B A = 0.719.
  bf16 does NOT lift 32B above Int8 -> the 14B->32B plateau is a real reasoning ceiling for concrete-emit,
  quantization exonerated. All arms within +/-1 case of Int8 (L1 -1, L0 +1 = noise).
- Q3 = formalize (L2b) robust: 0.844 bf16 == 0.844 Int8, still TOP arm AND cheapest (~505 tok/case vs A ~917).
  Input-formalization (MSC) Pareto-advantage holds under bf16.
- Q4 = selector (Bfair) still loses: 0.656 < A 0.719 at bf16 too (negative across all scales).
- Q2 (72B ceiling) = NOT answered (needs multi-GPU node; H100x4 nodes paused).

## 72B-AWQ-Int4 (local A100 80GB TP1, 2026-06-16) — Q2 CONFOUNDED
72B-bf16 (~145GB) does NOT fit a single 80GB A100; AWQ-Int4 (~36GB) is the only local option.
Result: 72B-AWQ-Int4 is BELOW 32B-bf16 on every arm (A 0.688<0.719; L2b 0.719<0.844 = -0.125).
- This is almost certainly an Int4 quant artifact, NOT 72B<32B. Q1 showed Int8~=bf16 at 32B, but
  AWQ-Int4 is 2x more aggressive than Int8 and degrades reasoning (L2b formalized arm hit hardest).
- Q2 (does reasoning keep rising to 72B) is NOT cleanly answered: 72B-AWQ ~= or below 32B is
  CONSISTENT WITH a plateau but cannot rule out a bf16-72B gain masked by Int4.
- Suggestive (not decisive): no huge latent 72B gain survived Int4, so a large bf16-72B jump is
  unlikely; plateau-at-~32B is the more probable read. Clean answer REQUIRES 72B-bf16 on a
  multi-GPU node (H100x4, currently paused).
