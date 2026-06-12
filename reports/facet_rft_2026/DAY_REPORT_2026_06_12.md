# DAY REPORT 2026-06-12 — D1 구조-표적 / D2 비용-표적 DPO (자동 생성)
> 쌍: structure 1017 / cost 376 (rft2 rollout, 어휘청정·통제 필터). base=rft2. 통제: rft2 MM full 49.0 / dpo2 55.95 / in-domain HF 51.6·daily 85.0(sub500, rft2).

## dpo_struct 공식 수치 (MM full / HF sub500 / daily sub500 순)

## dpo_cost 공식 수치 (MM full / HF sub500 / daily sub500 순)
  link_binary_f1 = 0.46997016474250874
  node_micro_f1_no_matching = 0.8217500637778344
  link_binary_f1 = 0.48384424192212094
  node_micro_f1_no_matching = 0.826605504587156
  link_binary_f1 = 0.8455846077947706
  node_micro_f1_no_matching = 0.9491408934707903

## P/R·결손 (MM full)
tb_rft2_mm       n=5547 parsed=5547 P=0.872 R=0.829 short=1017/5547 (18.3%) deficit=+0.225 (short-only 1.37)
tb_dpo2_mm       n=5546 parsed=5546 P=0.905 R=0.870 short=889/5546 (16.0%) deficit=+0.181 (short-only 1.37)

## census rft2->dpo_struct aggregate

## census rft2->dpo_cost aggregate
## aggregate
A: {"parse": 1.0, "n_nodes": 2.582477014602488, "valid_frac": 0.9514205447818219, "ntag": 1.657472507661799, "nself": 0.13827294032810528, "ndangle": 0.03461330448891293, "node_f1": 0.8438644503544444, "edge_f1": 0.6363152696711376, "links_ok": null, "argdict_frac": null}
B: {"parse": 1.0, "n_nodes": 2.4002163331530557, "valid_frac": 0.9422899121790426, "ntag": 1.4616910041463855, "nself": 0.11051018568595637, "ndangle": 0.024517757346313323, "node_f1": 0.8209553986866026, "edge_f1": 0.6263058701809378, "links_ok": null, "argdict_frac": null}

## improved: 315 (5.7%)  types={'dag': 49, 'chain': 254, 'single': 12}
## worsened: 478 (8.6%)  types={'dag': 75, 'chain': 403}
## same: 4754 (85.7%)  types={'chain': 2308, 'single': 2023, 'dag': 423}
