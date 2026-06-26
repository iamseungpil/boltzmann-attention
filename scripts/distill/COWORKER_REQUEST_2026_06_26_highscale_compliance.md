# Coworker 요청 (2026-06-26) — high-scale compliance sweep (32B-fp16 + 72B + 235B)

> 자기완결 요청서. 권위 컨텍스트: `papers/paper1_capability_scale_lever/what_scale_buys.md` §5.3·`scripts/distill/tau2/g2_rate.py`·`t2_compliance.py`·driver `gpu1_gategrid_t3.sh`. 이전 4월 요청서 무관.
> 한 줄: **{32B-fp16, 72B, 235B} 각각 floor + g15(gated) retail을 돌려 compliance scale-invariance 곡선을 frontier-쪽으로 확장 + 양자화 confound를 통제**한다. 논문 중심주장의 단일 최대 신뢰도 향상(§7-item-2).

## 0. TL;DR — 무엇을 / 왜
논문 핵심 발견(measured, 7B–32B-**Int8**·로컬): scale은 capability를 사지만 *policy-compliance(confirm-before-write)*는 못 산다. g2(미확인 write) **per-write rate가 scale 무관히 flat**:
| scale (로컬·Int8) | g2/write | 95% CI |
|---|---|---|
| 7B | 0.103 | [0.080, 0.132] |
| 14B | 0.070 | [0.053, 0.092] |
| 32B-Int8 | 0.075 | [0.058, 0.097] |
gate(g15)는 모든 scale서 위반 0. **세 구멍이 referee를 부른다**:
1. **32B-fp16** — 로컬은 **Int8**이었다. fp16에서 g2-rate가 같으면 *quant confound 없음*(주장 robust)·다르면 정직 보고. (양자화가 결론 교란 안 함을 박는 통제점.)
2. **72B** — "32B는 frontier 아니다" 반론. 72B floor의 g2-rate가 여전히 ~0.07–0.10이면 ≤72B로 확장.
3. **235B** — 최상위 open 모델. scale-invariance가 235B까지 가나(strong) 또는 깨지나(한정). *(가족 caveat: Qwen2.5 dense는 72B가 상한·235B는 Qwen3-235B-A22B[MoE]=다른 family·"더 큰 모델" 점이지 Qwen2.5 연속 아님 — tag·본문에 명시.)*
어느 결과든 §5.3·§9에 1급 데이터.

## 1. 세 run-셋 (각 floor + g15·기존 32B와 *완전 동일* config — 비교가능성이 생명)
공통: retail · **nt=3** · 114 task · user-sim=**gpt-4.1** · `t2_run_gated.py --gate 0 --resolve 1` · 프롬프트/task/config 변경 0 · base inference만(SFT 금지).

| # | 모델 (serve) | quant | GPU 대략 | floor save_to | g15 save_to |
|---|---|---|---|---|---|
| A | `Qwen/Qwen2.5-32B-Instruct` | **bf16/fp16** | ~64GB(2×48G TP=2 또는 1×80G) | `on_n32fp16_floor_retail` | `on_n32fp16_g15_retail_t3` |
| B | `Qwen/Qwen2.5-72B-Instruct-GPTQ-Int4` | Int4(AWQ도 OK) | ~40GB(1×48G) | `on_n72int4_floor_retail` | `on_n72int4_g15_retail_t3` |
| C | `Qwen/Qwen3-235B-A22B-Instruct`(또는 가용 235B) | Int4/AWQ | 多GPU(~120GB+) | `on_n235_floor_retail` | `on_n235_g15_retail_t3` |

**gate config (정확)**: floor = `T2_GATE_KINDS=`(빈값·게이트 미적용) / g15 = `T2_GATE_KINDS=auth,confirm,ownership,notice,preconditions`.

**실행 (driver 재사용 권장)**: `gpu1_gategrid_t3.sh`(7/14/32B 생성한 그 driver)를 **MODEL/quant/tag만** 위 표로 바꿔 floor+g15 두 줄씩. serve 패턴 = `reexp_assembled.sh` L32 (`CUDA_VISIBLE_DEVICES=$GPU vllm serve "$M" --port $PORT --enable-auto-tool-choice --tool-call-parser hermes --max-model-len 16384 --gpu-memory-utilization 0.92`). 직접 호출 예(floor B):
```bash
t2_run_gated.py --gate 0 --resolve 1 --domain retail --num_trials 3 \
  --user_llm openrouter/openai/gpt-4.1 --user_temp 0.0 --max_concurrency 8 \
  --save_to on_n72int4_floor_retail
```
- TP 필요(32B-fp16·72B-bf16·235B): vllm serve 줄에 `--tensor-parallel-size N` 추가. OOM시 `--max-model-len 12288`.
- **235B MoE**: tool-call 지원 확인·`--enable-auto-tool-choice --tool-call-parser hermes`(Qwen3면 parser 다를 수 있음·실패시 모델 카드 확인). 235B serve 어려우면 **A·B 먼저 보내고 C는 별도**.

## 2. compliance 계산 (도구 committed·코드 추가 0)
각 run 쌍마다 (vllm 꺼도 됨·CPU):
```bash
cd $REPO/scripts/distill/tau2 ; export PYTHONPATH=$REPO/scripts/distill/tau2:$TB/src
python t2_compliance.py --run <save_to> --domain retail   # F3(bench)·F4(full)·g1..g4 counts·compliance.json
# ★핵심 = g2 per-write rate + CI: g2_rate.py의 RUNS에 세 floor run을 추가해 한 번에:
#   RUNS=[("32fp16","on_n32fp16_floor_retail"),("72int4","on_n72int4_floor_retail"),("235","on_n235_floor_retail")]
python g2_rate.py
```

## 3. 산출물 (git 회수·필수·[[30]])
- 각 run의 `data/simulations/<save_to>/{results.json, compliance.json}`.
- **gzip → `$REPO/reports/facet_rft_2026/sim_results/` → `git add -f` + push**(scratch gitignore·소실방지). 또는 경로 통보.
- 한 줄 요약(scale별): floor **F3·F4·g2 count·총 write 수·g2/write rate+CI**, g15 **gap=0·viol=0** 확인.

## 4. 비용 / 규율 (엄수)
- **유료 = gpt-4.1 user-sim뿐**. nt3 × 114 × 2run × 3scale ≈ **2050 대화 → ~$90–240 예상**.
- **★예산 통제(권장 순서)**: ① **floor만 nt=1**로 세 scale 먼저(g2-rate 방향 확정·~$45–120) → 결과 보고 → ② strong/한정 갈리면 그때 nt3 + g15. g15는 *결정론적*이라(gate가 enforce) 한 scale 확인이면 scale-invariance 충분 — 비용 우려시 g15는 72B 하나만.
- **우선순위**: 못 다 돌리면 **floor 72B > floor 32B-fp16 > floor 235B > g15들** 순(72B가 최대 신뢰도·32B-fp16이 quant통제·235B가 frontier).
- base inference만·config 변경 0·quant tag 명시·결과 즉시 영속화·distinct tag(덮어쓰기 금지).
- 함정: `pkill -f`가 ssh 부모 죽임→PID kill. 235B/72B 다운로드 김→setsid background.

## 5. coworker가 답하는 결정 질문
1. **32B-fp16 g2-rate가 32B-Int8(0.075)과 같나** → quant confound 부재(주장 robust) vs 있음.
2. **72B·235B floor g2-rate가 0.07–0.10 대역에 머무나** → scale-invariant ≤235B(strong) vs 떨어짐(≤32B 한정).
3. **g15 gate가 모든 scale서 위반 0** 차단(guarantee-scaffold scale-invariant 재확인).
4. (부수) F3(raw pass) 32B 0.547 → 72B/235B headroom 한 점.

## 6. 맥락 (한 단락)
논문 = 기능×scale×레버×비용 지도. 중심주장 = **compliance(보장)는 scale-invariant → frontier도 결정론 gate 필요 → 소형+gate 신뢰성 대등 → 비용우위(~23×)**. 이 sweep은 (a)양자화가 결론 안 흔든다(32B-fp16) (b)주장이 frontier까지 간다(72B·235B)를 박아 "32B는 작다·Int8 아티팩트다" 두 반론을 동시에 닫는다. strong/한정 어느 쪽이든 정직 반영.
