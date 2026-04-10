# Experiment Plan v30: Proxy-to-Storage Recovery Plan

## Decision Summary

이 버전의 계획은 의도적으로 공격적인 정리본이다. 지금 계속 들고 갈 것은 `two_pass` 하나뿐이고, 그것도 **diagnostic-only** 상태로 다룬다.

- Proceed now: `two_pass` control diagnostics
- Proceed now: 8K/16K tiny-budget frontier
- Proceed next: storage-valid selective refinement rewrite
- Hold: QDRP revised risk metric only
- Stop: TRIC
- Stop: query-aware dequantization as an end-to-end direction
- Stop: sharp temperature as a method direction

## Direction 1: Two-Pass Selective Refinement

### Intent

low-bit attention path에서 retrieval failure를 복구하는 positive signal이 실제 retrieval-region recovery인지, 아니면 쉬운 task artifact인지 판별한다.

### Hypothesis

`two_pass`가 진짜 retrieval repair라면 `recent_k`, `sink_k`, `random_k`보다 needle overlap과 NIAH가 모두 높아야 하며, 8K/16K에서도 작은 `k`에서 baseline보다 먼저 회복해야 한다.

### Verification

1. Selector diagnostics
   - same prompt grid
   - methods: `two_pass`, `recent_k`, `sink_k`, `random_k`
   - metrics: NIAH, needle hit rate, sink overlap rate, recent overlap rate, selected head-slot count
2. Harder frontier
   - contexts: `8192`, `16384`
   - budgets: `k=1,2,4,8,16`
   - models: Mistral, Qwen
   - metrics: NIAH, minimum recovery `k`, runtime, effective average bits upper bound

### Kill Criterion

`recent_k` 또는 `sink_k`가 `two_pass`와 비슷하게 나오거나, 8K/16K에서 작은 `k` 이득이 사라지면 retrieval-specific story를 접는다.

### Output Interpretation Rule

이 단계 결과는 전부 `attention_path_proxy_diagnostic`로 기록한다. storage-valid compression claim은 금지한다.

## Direction 2: Storage-Valid Selective Refinement Rewrite

### Intent

현재 proxy positive signal이 실제 compressed-cache setting에서도 남는지 검증한다.

### Hypothesis

base cache를 실제 low-bit로 저장하고 selected subset만 stored residual 또는 promoted side-buffer를 통해 복원해도, uniform 2-bit 대비 retrieval gain이 남아야 한다.

### Verification

1. Rewrite requirements
   - cache write/update path에서 base K를 실제 quantized representation으로 저장
   - selected token용 promoted side-buffer 또는 stored residual path 구현
   - decode attention은 stored representation만 사용
2. Fair-budget table
   - regimes: equal-storage, equal-HBM, equal-extra-compute
   - methods: `uniform_2bit`, `uniform_3bit`, storage-valid selective refine
3. Metrics
   - bounded NIAH
   - short PPL sanity
   - latency and realized memory accounting

### Kill Criterion

storage-valid rewrite 뒤 gain이 사라지거나, equal-budget에서 `uniform_3bit`보다 확실히 낫지 못하면 paper method로 채택하지 않는다.

## Direction 3: QDRP

### Intent

query-conditioned score uncertainty가 정말 score-only selector보다 유용한지 최소한의 real-trace evidence를 확보한다.

### Hypothesis

revised page-risk metric이 real trace에서 raw score보다 miss-page recall 또는 winner-page hit를 높여야 한다.

### Verification

1. Do not run end-to-end NIAH first.
2. First run trace-level calibration only.
3. Compare `raw_score`, `margin`, revised `risk`, and `hybrid`.

### Kill Criterion

real trace에서 raw score를 못 이기면 즉시 종료한다.

## Direction 4: TRIC

### Intent

현재는 없다. 이 방향은 닫는다.

### Hypothesis

없다. shared linear baseline을 넘지 못한 상태에서 추가 GPU 사용은 정당화되지 않는다.

### Verification

없다. 새로운 real-trace evidence가 나오기 전까지 실행하지 않는다.

### Kill Criterion

이미 충족되었다.

## Execution Order

1. Run `two_pass` control diagnostics.
2. Run 8K/16K tiny-budget frontier.
3. Decide whether the signal survives basic reviewer attacks.
4. Only then implement storage-valid selective refinement.
5. Revisit QDRP only if a revised real-trace risk metric appears.

## Autoresearch Gate

새 autoresearch는 두 조건이 모두 만족될 때만 시작한다.

1. Result payloads and report language agree that current positive runs are proxy diagnostics.
2. Next experiment directly tests a remaining objection rather than introducing a new idea.
