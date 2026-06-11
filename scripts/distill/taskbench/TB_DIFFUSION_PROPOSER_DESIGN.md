# 이종 제안기(diffusion) × 결정론 선별 — 설계 (detail, 2026-06-12)
> 📌 **구조 안내**: 모든 설계·실험 문서의 단일 마스터 = repo `scripts/distill/EXPERIMENT_DESIGN.md` (**§7 문서지도**에서 각 문서의 역할·상태 확인; 목표·순서 변경은 마스터 §0-§4에서만). 처음 읽는다면 마스터부터.

> 동기 = E6 확정 (TB결과 §8.8): best-stack(dpo2+guided) 정책은 K=8 샘플이 수렴 → 선별 갭 +1.4뿐.
> 잔여 L6는 **수렴 정책의 샘플 분포 밖** ⇒ 처방 = 제안 다양성의 인위적 복원. 외부 근거 = DiG-Plan
> (arXiv 2606.05728): 도구-셋 제안 커버리지 AR Pass@10 0.32 vs **diffusion 0.94** (단, 그들 프로토콜
> = "TaskBench-23" 501개 비표준 — 수치 이식 금지, 기제만 차용. TB결과 §1.5 인용위생 준수).

## 1. 가설·프레임 정합
- **가설 H-D**: 이종(비-AR) 제안기를 풀에 섞으면 **풀-oracle이 상승**한다 (다르게 틀리는 제안 → 정답 구조가 풀에 들어올 확률↑). 선별은 기존 결정론 게이트(R6) — 마스킹 아닌 선별이라 GAD 왜곡 무관.
- §10.5 정합: R6(K-제안+검증-선별)의 **제안자 슬롯만 교체** — propose-then-gate 구조 불변. A1-A5 어댑터 재사용(스키마·gold·평가기 동일).

## 2. 후보 제안기 (공개 가중치, 우리 GPU 적합)
| 모델 | 크기 | 비고 |
|---|---|---|
| **Dream-7B** (`Dream-org/Dream-v0-Instruct-7B`) | 7B, bf16 ~16GB | DiG-Plan이 쓴 라인. HF 공개. 자체 diffusion generate 루프(vllm 불가) |
| LLaDA-8B-Instruct (`GSAI-ML/LLaDA-8B-Instruct`) | 8B | 대안/2차 |

## 3. 실험 단계 (사전등록)
**P-D0 스모크 (반나절, GPU 1장)**: Dream-7B로 MM 프롬프트 50개 × K=4 생성 →
  ①**형식 준수율 측정이 1차 관문**: diffusion은 vllm structured_outputs 불가 → JSON 파싱율·valid_frac 실측
  (회복 경로 = name-snap v0 후처리 — MM에선 guided와 동급, §9.5b). 파싱율 <50%면 템플릿/few-shot 보강 1회 재시도, 그래도 <50%면 LLaDA로 교체.
  ②다양성 즉정: 50개 풀에서 distinct-plan율·노드셋 Jaccard.
**P-D1 본 측정 (1일, GPU 1장)**: MM sub500 × K_d=8 생성 → 풀 분석 3종 (`tb_kgate_heldout.py` 확장):
  - AR-only 풀(기존 tb_dpo2g_mmk0-7) vs **혼합 풀(AR 4 + Dream 4)** vs Dream-only 풀
  - **판정(사전등록)**: ⓐ혼합-풀 oracle > AR-풀 oracle +2 이상 = H-D 채택 → 게이트-선별 실측으로 진행
    ⓑoracle 상승 없음 = H-D 기각 (diffusion 다양성이 우리 표준 프로토콜로 전이 안 됨 — 그 자체로 DiG-Plan 비표준-프로토콜 의존성의 증거, 1급 음성결과)
  - 선별 후 공식 edge(tb_build_eval)도 병기 — census-식과 분리 보고.
**P-D2 (조건부, ⓐ시)**: 혼합-풀 + 강화 게이트(v1+) 선별의 best-stack 대비 순이득 → 패키지 갱신 여부.

## 4. 구현 노트
- 생성 루프: Dream repo의 diffusion_generate API(HF transformers 기반, trust_remote_code) — `tb_diffusion_sample.py` 신규 (프롬프트 = inference.py와 동일 문자열 재사용, 출력 = inference.py 호환 predictions jsonl로 기록 → 기존 채점·조인 도구 전부 재사용).
- 후처리 사다리: parse-fix(reformat) → name-snap(v0) → 풀 합류. guided는 불가(서빙 스택 비호환) — **불공정 비교 방지를 위해 AR 풀도 snap-기준으로 정렬한 변형 병기**.
- 자원: 다운로드 ~16GB(디스크 1.3T 여유) — **D1/D2 학습과 무충돌(네트워크/디스크만)**, GPU는 학습 종료 후.

## 5. 리스크 (정직)
①형식 준수가 최대 리스크(P-D0가 게이트) ②diffusion 추론 속도(스텝 수 × 길이 — sub500×8이면 수 시간) ③DiG-Plan 기제의 프로토콜-의존 가능성(그래서 ⓑ도 1급) ④신규 의존성(trust_remote_code) — seka_env 격리 설치.
