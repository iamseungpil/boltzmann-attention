# NIGHT REPORT 2026-06-12 (자동 생성 — tb_night_0612.sh)
> 사전예측: E8 +1~3 / E4 slim 손실<3.1 / E6 선별갭>+5 / E5 회수≥30%. 판정·박제는 Track A 아침 세션.

## E1 base+guided MM full (2×2 마지막 셀)
  link_binary_f1 = 0.501310490315157
  node_micro_f1_no_matching = 0.8426785408460353

## E8 HF held-out guided (lodo_hf, full — unguided 통제=35.0)
  link_binary_f1 = 0.5440494590417311
  node_micro_f1_no_matching = 0.8701633705932932
e-F1 0.37832857274486315 n-F1 0.7465381337878142

## E4 promptslim in-domain (dpo2+guided; 통제: HF sub500 54.10 / daily 83.64 — unguided)
hf_full e-F1 0.544 n-F1 0.8702
hf_slim e-F1 0.5034 n-F1 0.8497
dl_full e-F1 0.8504 n-F1 0.964
dl_slim e-F1 0.8442 n-F1 0.9645

## E5 스코어러 v1 (in-domain .all)
[kgate] prompts=3869 K~8 | mean(1-shot)=0.6979 gate-select=0.7369 oracle(best-of-K)=0.8702 (참고 reward-pick=0.8695 — reward는 gold-기반=oracle류)

## E6 held-out K=8 선별 (MM sub500, dpo2+guided, temp0.8)
[kgate-heldout] ids=0 K=8 | mean=0.0000 gate_v0=0.0000 gate_v1=0.0000 oracle=0.0000

원본 로그: tb_night.log·tb_guided_base.log / pred: tb_dpo2g_{hf,dl}_{full,slim}·tb_dpo2g_mmk0-7·tb_lodo_hf_guided
