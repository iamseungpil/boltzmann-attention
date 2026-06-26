# Coworker 결과 (2026-06-17) — 대형 스케일 두 lane (B-budget depth + multidomain routing)

> 두 요청서(`COWORKER_REQUEST_2026_06_17_B_budget_scale.md`·`..._multidomain_scale.md`)의 coworker 절반.
> **inference-only·temp0·frozen eval(무편집)·로컬 vLLM(키 미사용).** 집계·phase diagram·박제는 woori 몫
> (B-budget→`M_A_RESULTS §15`, multidomain→`§20`). 이 노트 = 원자료 포인터 + 요약.

## 산출물 위치 (HF dataset `iamseungpil/sopbench-trackb-h200`)
- `depth_scale/depth_{32B,72B,235B_fp8}_N{5,10,20,50}.json` + `meta_*.json`
- `multidomain_scale/{32B,72B,235B_fp8}__{retail,airline}_g{0,1}.json` + `meta_*.json`
- amlt: depth/md 32+72B = `keen-bluebird`·`capital-porpoise`; 235B = `genuine-wolf`. node 스크립트 = `scripts/distill/ma/node_run_{depth_scale,multidomain,235b}.sh`.
- 모델 메타: 32B L=64/d5120 · 72B L=80/d8192 (both Qwen2.5-Instruct bf16) · 235B L=94/d4096/exp128-active8 (Qwen3-A22B-Instruct-2507 **FP8**).

## Lane 1 — B-budget in-head 깊이 (n=250/셀, arms A=in-head / B=op-IR+엔진 / D=oracle)
overall acc(N):

| N | A:32B | A:72B | A:235B | B(=op-IR+엔진) 전 크기 |
|---|---|---|---|---|
| 5 | 0.81 | 0.82 | 0.83 | **1.00** |
| 10 | 0.76 | 0.70 | 0.73 | **1.00** |
| 20 | 0.61 | 0.61 | 0.70 | **1.00** |
| 50 | 0.49 | 0.48 | 0.54 | **1.00** |

- **in-head(A)은 N으로 단조 붕괴**, 크기 키워도 거의 안 움직임(N50: 0.49→0.48→0.54). by-op에서 **comparative@N50 = 0.02 (235B 포함 전 크기)**, rank@N50 ≈ 0.30. → comparative/rank의 임계크기 **S\* > 235B**.
- **op-IR+엔진(B) = 1.00 전 크기·전 N**, op-recognition = 1.00. = 얕은 인식(소형도 가능) + 결정론 엔진이 깊은 스캔. **B ≫ A, B = D.**
- **판정**: "충분히 크면 in-head 매핑으로 푼다"는 *얕은 연산에만* 참. 깊은 연산은 235B도 못 넘고 엔진은 trivial → offload 지배.

## Lane 2 — multidomain content-routing (retail n=32 / airline n=27, base 학습 0)
overall new_item_id acc:

| 셀 | 32B | 72B | 235B |
|---|---|---|---|
| retail g0(floor) | 0.44 | 0.38 | 0.41 |
| retail g1(ceiling) | 0.50 | 0.53 | **0.41** |
| airline g0 | 0.56 | 0.41 | 0.59 |
| airline g1 | 0.74 | 0.70 | 0.70 |

- recognition 0.85~1.00 (라우팅 문제 아님 — §20 확증). 실패 = **성분 B = `set` 과소추출(missing_key)** 지배(retail).
- **성분 B는 스케일·gloss로 안 풀림**: retail miss_key가 32B 14 → 72B 18 → 235B 11 (g0)·all g1 ≈ 11~12. 235B는 **gloss가 안 먹힘**(retail g0=g1=0.41) → ceiling에서 235B(0.41) < 32B(0.50) < 72B(0.53)로 뒤집힘.
- airline은 성분 B 부담 적음(235B g1 miss_key=0). gloss는 일관되게 도움(S1>S0).
- **판정**: 큰 base가 retail multi-attr `set` 추출을 못 올림. woori 7B-routing-LoRA(§20)가 이 천장을 넘으면 **소형+학습 > 대형+gloss** 박힘.

## 종합 + caveat
- **두 lane 모두 0.5B→235B 전 사다리에서 scale-plateau** — 깊은 연산·다속성 추출이라는 핵심 하위문제를 스케일이 못 넘고, **결정론 offload(엔진 1.0 / 학습 라우팅)가 푼다** = thesis 강화.
- ★**235B 별표 3겹**(집계 시 반영): (1) **Qwen3 family**(32/72B는 Qwen2.5 = cross-family, 깨끗한 dense 연장 아님) (2) **MoE** active22B≠total235B (3) **FP8** 양자화. retail g1 역전은 이 confound 탓일 수 있음 → **load-bearing dense 사다리 = 0.5–72B(bf16 Qwen2.5)**, 235B는 확장점.
