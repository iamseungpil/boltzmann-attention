# Coworker 요청 (2026-06-26) — 72B compliance point (논문 핵심주장 ≤32B → ≤72B 확장)

> 자기완결 요청서. 권위 컨텍스트: `papers/paper1_capability_scale_lever/what_scale_buys.md` §5.3 (compliance scale-invariance)·`scripts/distill/tau2/g2_rate.py`·`t2_compliance.py`. 이전 4월 요청서는 무관.
> 한 줄: **72B에서 floor + g15(gated) retail을 돌려, compliance scale-invariance 곡선에 72B 점을 추가**한다. 이게 논문 중심주장의 *단일 최대 신뢰도 향상*(§7-item-2).

## 0. TL;DR — 무엇을 / 왜
논문 핵심 발견(measured, 7B–32B): **scale은 capability를 사지만 *policy-compliance(confirm-before-write)*는 못 산다.** g2(미확인 write) **per-write-opportunity rate가 scale 무관히 flat**:
| scale | g2/write | 95% Wilson CI |
|---|---|---|
| 7B | 0.103 | [0.080, 0.132] |
| 14B | 0.070 | [0.053, 0.092] |
| 32B | 0.075 | [0.058, 0.097] |
그리고 결정론 gate(g15)는 **모든 scale서 위반 0**. **72B 점이 빠져 있다** — referee가 "32B는 frontier 아니다"라고 반드시 찌른다. 72B floor의 g2-rate가 *여전히 ~0.07–0.10이고 gate가 0으로* 차단하면 → 주장이 ≤72B로 확장(strong). 만약 72B에서 rate가 *떨어지면* → 주장을 ≤32B로 정직 한정. **어느 쪽이든 논문에 1급 데이터.**

## 1. 정확한 실험 (기존 32B run과 *완전 동일* config·비교가능성이 생명)
**모델**: `Qwen/Qwen2.5-72B-Instruct-GPTQ-Int4` (≈40GB·단일 A6000 48GB 가능. AWQ-Int4도 OK·tag에 명시. **bf16/Int8 불요** — 32B는 Int8이었으니 동급/상위 quant면 충분). 미캐시 → HF 다운로드 필요.

**두 run (각 nt=3·114 task·retail·user-sim=gpt-4.1)**:
1. **floor** (gate 없음 — g2 위반이 *발생*하는 baseline):
   ```bash
   T2_GATE_KINDS= (빈값)  # 게이트 미적용
   t2_run_gated.py --gate 0 --resolve 1 --domain retail --num_trials 3 \
     --user_llm openrouter/openai/gpt-4.1 --user_temp 0.0 --max_concurrency 8 \
     --save_to on_n72int4_floor_retail
   ```
2. **g15** (gated — gate가 위반을 *0으로* 차단하는지):
   ```bash
   env T2_GATE_KINDS=auth,confirm,ownership,notice,preconditions \
   t2_run_gated.py --gate 0 --resolve 1 --domain retail --num_trials 3 \
     --user_llm openrouter/openai/gpt-4.1 --user_temp 0.0 --max_concurrency 8 \
     --save_to on_n72int4_g15_retail_t3
   ```
- **driver 재사용 권장**: `scripts/distill/tau2/gpu1_gategrid_t3.sh`(7/14/32B × g14/g15 생성한 그 driver)를 **MODEL만 72B로** 바꿔 floor + g15 두 줄 실행. serve 패턴은 `reexp_assembled.sh`(L32 `CUDA_VISIBLE_DEVICES=$GPU vllm serve "$M" --port $PORT --enable-auto-tool-choice --tool-call-parser hermes --max-model-len 16384 ...`)와 동일·72B면 `--gpu-memory-utilization 0.92` 유지·OOM시 `--max-model-len 12288`.
- ⚠ **config·task·프롬프트 변경 금지**(t2_run_gated.py committed 그대로) — 32B/14B/7B와 *직접* 비교해야 함. resolve=1·nt=3 동일.

## 2. compliance 계산 (도구 committed·코드 추가 0)
두 run 끝나면 (vllm 꺼도 됨·CPU 분석):
```bash
cd $REPO/scripts/distill/tau2 ; export PYTHONPATH=$REPO/scripts/distill/tau2:$TB/src
# F3/F4 + g1..g4 violation counts (compliance.json 자동 생성됨·없으면):
python t2_compliance.py --run on_n72int4_floor_retail --domain retail   # F3(bench)·F4(full)·violation_sims
python t2_compliance.py --run on_n72int4_g15_retail_t3 --domain retail
# ★핵심: g2 per-write-opportunity rate + CI (g2_rate.py의 RUNS에 72B 한 줄 추가하거나 --run 인자화):
python g2_rate.py    # (RUNS 리스트에 ("72B","on_n72int4_floor_retail") 추가 후)
```
(만약 `t2_run_gated.py`가 자동으로 `compliance.json`을 안 쓰면, `t2_compliance.py`로 results.json→compliance.json 1회 생성.)

## 3. 산출물 (git 회수 — **필수**)
- `data/simulations/on_n72int4_floor_retail/{results.json, compliance.json}`
- `data/simulations/on_n72int4_g15_retail_t3/{results.json, compliance.json}`
- **gzip → `$REPO/reports/facet_rft_2026/sim_results/` → `git add -f` + push** (scratch는 gitignore·소실방지·[[30]]). 또는 경로 알려주면 우리가 회수.
- 한 줄 요약: 72B floor의 **F3·F4·g2 count·총 write 수·g2/write rate + CI**, g15의 **gap=0·viol=0** 확인.

## 4. 비용 / 규율 (엄수)
- **유료 = gpt-4.1 user-sim뿐**. nt3 × 114 task × 2 run ≈ 684 대화 → **~$30–80 예상**(§7-item-2 추정). **예산 초과 우려 시 floor를 nt=1로 먼저**(g2-rate는 nt1로도 방향 확정 가능·~$15–40) → 결과 보고 후 nt3.
- **base inference만**(SFT/finetune 금지·[[feedback-thesis-tbox-transfer-direction]]). config 변경 0. quant는 tag 명시.
- **결과 즉시 영속화**(위 §3). 같은 tag 재런 금지(덮어씀)·distinct tag.
- vllm 함정: `pkill -f`가 ssh 부모 죽임 → PID kill. 72B 다운로드 오래 걸림 → setsid background.

## 5. coworker가 답하는 결정 질문
1. **72B floor의 g2/write rate가 0.07–0.10 대역에 머무나**(=scale-invariant ≤72B·strong) **아니면 떨어지나**(=주장 ≤32B 한정).
2. **g15 gate가 72B서도 위반을 0으로** 차단하나(guarantee-scaffold scale-invariant 재확인).
3. (부수) 72B F3(raw pass)가 32B 0.547서 얼마나 더 오르나 = capability headroom 한 점.

## 6. 맥락 (한 단락)
논문 = "scale이 무엇을 사고 무엇을 못 사나"의 기능×scale×레버×비용 지도. 중심주장 = **compliance(보장)는 scale-invariant → frontier도 결정론 gate 필요 → 소형+gate가 신뢰성 대등 → 비용우위(~23×)**. 72B는 그 곡선의 frontier-쪽 한 점으로, "32B는 작다"는 반론을 닫는다. 결과는 strong/한정 어느 쪽이든 §5.3·§9에 정직히 반영한다.
