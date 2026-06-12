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

## 3. 실험 단계 (사전등록 — 2026-06-12 리뷰 반영 v2: P-D(-1) 신설·분해행·ⓑ전제조건·P-D2 채택기준)
**★P-D(-1) 이종-AR 풀 census (zero-GPU, P-D0 선행 — 메타규칙 "GPU 전 zero-cost 진단")**:
  디스크의 기존 sub500 MM 예측으로 "이종이면 되는가"를 공짜로 분리. 풀(사전등록):
  AR8 = `tb_dpo2g_mmk0-7`(E6 동일) / **H6** = {qwen3b, qwen14b, qwen3_4b, qwen3_14b, tb_lodo_hf, tb_lodo_daily}
  (family×size×학습변형 3축 이종; id 겹침 486-500/500 실측). 행: oracle(AR8) [E6 재현 통제] /
  oracle(AR8+H6) / oracle(AR4=k0-3 고정) / oracle(AR4+H6).
  - **판정**: Δhetero=oracle(AR8+H6)−oracle(AR8) **≥+2** → 싼-이종성으로 충분 ⇒ P-D1 성공 기준 상향
    ("Dream 한계가치 > 이종-AR 한계가치"; 배포 2-모델 비용 정당화 부담) / **<+2** → AR끼리 닮게 틀림의
    1차 증거 ⇒ H-D 본형(비-AR 기제 필요) 강화. ⚠️해석 가드: H6=greedy 단일샘플=싼-이종성의 *하한*
    (temp0.8 K-샘플 이종-AR 아님) — 음성이어도 "AR 다양성 한계 확정"으로 과대해석 금지.
  - **✅실행·판정 완료 (2026-06-12, 결과 권위 = TB결과 §8.9)**: **Δhetero=+13.6**(0.720→0.856, H6단독 0.847,
    인플레이션 가설 기각 census 통과) ⇒ 싼-이종성으로 coverage 해결. **동시 발견 = 혼합 풀에서 게이트 붕괴**
    (v0/v1 0.71→0.52-0.54 < mean) ⇒ **binding constraint는 선별기로 이동, P-D0/P-D1은 조건부 강등**
    (Dream 한계가치는 AR+H6 위에서만 의미·"싼 대안 대비 순이득" 입증 부담). H6=같은 base LoRA들=멀티-LoRA
    1서버 ≈ 추가비 0. 다음 = 이종-풀 robust 선별기 설계(신규 detail 문서로, 마스터 §7 경유).
**P-D0 스모크 (반나절, GPU 1장)**: Dream-7B로 MM 프롬프트 50개 × K=4 생성 →
  ①**형식 준수 관문(이중)**: JSON 파싱율 ≥0.5 **∧ snap-후 valid_frac ≥0.8**(게이트가 실제 소비 가능한
  비율 — 파싱돼도 구조 전파손 케이스 차단). 미달이면 템플릿/few-shot 보강 1회 재시도, 그래도 미달 시 LLaDA로 교체.
  ②다양성 측정: distinct-plan율·노드셋 Jaccard.
**P-D1 본 측정 (1일, GPU 1장)**: MM sub500 × K_d=4 생성 → 풀 분해 4행(`tb_kgate_heldout.py`, 동일 id셋):
  - **oracle(AR4=k0-3) / oracle(AR4+D4) / oracle(AR8) / oracle(AR8+D4)** + Dream-only oracle
  - **검정(사전등록)**: Dream 한계가치 = oracle(AR4+D4)−oracle(AR4) vs AR 한계가치 = oracle(AR8)−oracle(AR4).
    ⓐ Dream 한계가치 > AR 한계가치 ∧ oracle(AR8+D4)−oracle(AR8) ≥ +2 (P-D(-1)이 Δhetero≥2면 "> 이종-AR 한계가치"도 요구)
      = H-D 채택 → P-D2. **+2 임계는 paired bootstrap 95% CI(id-단위 resample) 병기** — CI가 0 걸치면 보류.
    ⓑ 상승 없음 = 기각. **단 (iii)"diffusion 다양성이 표준 프로토콜로 전이 안 됨" 해석은 전제 2개 충족 시만**:
      P-D0 형식 관문 통과 ∧ Dream-only oracle ≥ 0.7×oracle(AR8). 미달 → "형식/모델 한계로 측정불가" 강등
      (형식 사고가 음성결과로 둔갑 금지 — LODO 직렬화-범인 함정 동형).
  - 선별 후 공식 edge(tb_build_eval)도 병기 — census-식과 분리 보고.
  - 공정성 한계(명시): unguided AR K-샘플 디스크에 없음(재생성=GPU 비용) → AR8=guided. 비대칭은 Dream에
    불리한 방향=ⓐ에 보수적이므로 채택 결론엔 안전, ⓑ 해석에만 위 전제조건으로 방어.
**P-D2 (조건부, ⓐ시)**: 혼합-풀 + 게이트(v1+) 선별의 best-stack 대비 **공식 edge-F1 순이득 ≥ +1 = 채택**
  (E6 교훈: oracle↑≠실현이득 — 현 게이트 회수 18-22.6%라 oracle+2≈실현+0.4뿐; 게이트가 AR 오류형태에
  튜닝돼 diffusion 제안을 더 못 고를 위험 포함). 미달 = "oracle-only 호기심" 분류. 채택 시 패키지 보고에
  **배포 비용 열(제안기 2-모델 = 추론비 ~2×) 병기** — {소형·저비용} 주장과 충돌 방지.

> 인용위생 체크박스: DiG-Plan(2606.05728) 1차 검증 = 프로토콜 디테일(TaskBench-23 501, Pass@10 수치) 원문 확인됨
> · 논문 본문 인용 전 R8 절차(버전 명시·수치 재검증) 필수, 수치 이식 금지 유지.

## 4. 구현 노트
- 생성 루프: Dream repo의 diffusion_generate API(HF transformers 기반, trust_remote_code) — `tb_diffusion_sample.py` 신규 (프롬프트 = inference.py와 동일 문자열 재사용, 출력 = inference.py 호환 predictions jsonl로 기록 → 기존 채점·조인 도구 전부 재사용).
- 후처리 사다리: parse-fix(reformat) → name-snap(v0) → 풀 합류. guided는 불가(서빙 스택 비호환) — **불공정 비교 방지를 위해 AR 풀도 snap-기준으로 정렬한 변형 병기**.
- 자원: 다운로드 ~16GB(디스크 1.3T 여유) — **D1/D2 학습과 무충돌(네트워크/디스크만)**, GPU는 학습 종료 후.

## 5. 리스크 (정직)
①형식 준수가 최대 리스크(P-D0가 게이트) ②diffusion 추론 속도(스텝 수 × 길이 — sub500×8이면 수 시간) ③DiG-Plan 기제의 프로토콜-의존 가능성(그래서 ⓑ도 1급) ④신규 의존성(trust_remote_code) — seka_env 격리 설치.
