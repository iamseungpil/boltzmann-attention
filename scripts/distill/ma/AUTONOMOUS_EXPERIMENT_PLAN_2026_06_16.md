# 자동 실험 계획 (5h 무인 배치) — 2026-06-16 PM

> 목적: 외출 5h 동안 **순수 inference 실험을 연속 자동 실행**. 학습(M-σ)·deep-research는 babysit 필요 → 제외(계획서 §다음 수동). 권위 결과 = `M_A_RESULTS.md`. 오케스트레이터 = `ma_overnight_batch.sh`.

## 자동 큐 (순서·전부 inference·committed 파이프 재사용)
| # | arm | 무엇을 답하나 | 모델 | 출력 |
|---|---|---|---|---|
| Q1 | **Sstep** (실행중) | scaffolded typed스텝+per-step 결정론검증이 자유CoT(0.656) 넘어 큰모델(0.844) 닿나 = capability 날개 | 7B·14B(GPU0)·32B-Int8(GPU1) | `_sstep` |
| Q2 | **Snover** (Sstep·검증 OFF) | **per-step 검증/피드백이 핵심인가**(분해만 vs 분해+검증). DR5 "외부검증>자유" 주장을 *우리 데이터로* 분리 | 7B·14B·32B-Int8 | `_snover` |
| Q3 | **SCv** (self-consistency·A를 N=5 majority) | **구조적 검증(Sstep) > 무차별 샘플(brute)인가** at 유사 비용 | 7B·14B·32B-Int8 | `_scv` |

- 각 job = `ma_eval_scale.sh <models> <arms> <suffix> <gpu> <port>`. 2-GPU 병렬(7B+14B=GPU0 / 32B=GPU1)·job-set 간 wait.
- 비용계측 자동(토큰·호출). 완료 시 `ma_overnight_summary.log`에 3모델×전arm 집계.

## 해석 가이드 (돌아와서 볼 것)
- **Sstep > Atwo(0.656)**: 결정론 검증이 자유CoT 넘음 = capability 날개 양성.
- **Sstep > Snover**: per-step 검증/피드백이 *핵심*(분해만으론 부족) = DR5 정합·thesis 검증주장 입증.
- **Sstep > SCv**: 구조적 결정론검증 > 무차별샘플 = "검증이 brute-force보다 효율".
- 비용: Sstep multi-call 토큰 vs 큰모델 1-call — 비용-정당성 판정.

## ★다음 수동 단계 (외출 후·자동화 안 함)
1. **결과 박제**: `M_A_RESULTS.md §9`에 Q1-Q3 + 해석.
2. **M-σ 데이터 파이프** (critical path #5·전이의 전제): SOPBench/TaskBench → (NL,config,target-spec) 삼중쌍·multi-config·등방화. *학습이라 babysit 필요.*
3. **M-D 전이** (#6·C8·thesis 핵심): M-σ 스킬 → held-out config 무재학습 전이.
4. coworker 32B-bf16/72B floor 회수.

## gotcha
- GPU0:8013 / GPU1:8014 분리(충돌 방지). 각 job 시작 시 해당 GPU procs kill.
- ssh 끊겨도 setsid detached 지속. 진척=`ma_overnight_batch.log`·결과=`ma_eval_<tag>_<suffix>.jsonl`.
- SCv는 temp>0 샘플(비결정)·N=5 majority. 나머지 temp=0.
