# 벤치-불변 프레임워크 × 벤치 포트폴리오 (detail 문서, 2026-06-12)

> ⚠️**마스터 = `EXPERIMENT_DESIGN.md`** (§1.5 요약·§7 문서지도 등재). 목표·순서 변경은 마스터에서만 — 이 문서는 규칙표·포트폴리오 선정근거·어댑터 명세의 **구현 세부**.

> **목적 명제 (사용자, 2026-06-12)**: 특정 벤치 최적화가 아니라, **다양한 벤치를 최소한의 자동 노력으로 전부 커버하는 기본 규칙 내재 프레임워크**. 벤치 서베이는 선택용이 아니라 커버리지 행렬의 타깃 목록.
> 상호링크: 규칙·근거 요약 = `../reports/facet_rft_2026/TASKBENCH_EXPERIMENT_RESULTS.md` **§10·§10.5**(레버 장부·층위 분류) · 벤치 생사 전수조사 = 동 **§1.5** · 선행연구 차별점 = `taskbench/TB_GROUNDED_COPY_V1_DESIGN.md` **§6.5** · SOPBench 실측 = `../reports/facet_rft_2026/SOPBENCH_EXPERIMENT_RESULTS.md` · 대형모델 분담 = `../reports/facet_rft_2026/COWORKER_REQUEST_TB_SCALE.md` **§7-8** · thesis 좌표 = `FIELD_GAP_LLM_VALUE_DESIGN.md` **§15.4(정정 포함)·§17.9·§18** · CDP 벤치 설계 = `GROUNDED_BIZ_AGENT_BENCH_DESIGN.md`.

## 1. 벤치-불변 규칙 R1-R8 (프레임워크 내재 — 전부 실측 근거)
| # | 규칙 | 근거 (출처) |
|---|---|---|
| R1 | 심볼(도구명·필드명)은 생성 금지, 컨텍스트 복사 — enum/문법 마스크로 집행 | TB §9.5b (guided daily +8.0·무효 0/13k) |
| R2 | 인스턴스 사실은 act 전 gather 선행 | SOP active-H3 6→15·gather LODO 전이 |
| R3 | 허가/결정 판단은 모델 emit 금지 — 결정론 게이트 offload | SOP 3-NULL LOCK·DGGATE 15→29/34 |
| R4 | 의미 매칭은 모델, 출력 공간은 제약 (하이브리드) | TB §9.5 v0/v1 경계 |
| R5 | 정책류 행동(종결·길이)은 on-policy **양방향** 선호학습만; 규율-완비 base에 모방-SFT 금지 | TB §9.6 (v1 net− ↔ v2 신기록)·§8.5 (32B −5.4 prereg 적중) |
| R6 | 구조 선택(L6)은 K-제안 + 결정론 검증-**선별** (마스킹 아님 — 분포왜곡 회피) | TB §8.7 (edge-snap NULL)·DiG-Plan Pass@10·GAD |
| R7 | 배포 전 base census → 레버 선택 (이득 = base 결핍의 함수; 크기·family 불문) | TB §8/§8.5/§8.6 (1.5B~235B 실측) |
| R8 | 측정 규율: 집계 후 즉시 궤적 census·사전등록·내부-일관 비교·인용=원문 검증 | 전 문서의 방법론 + arXiv 규율 |

## 2. 벤치-어댑터 A1-A5 (벤치당 새로 쓰는 유일한 것)
| # | 추출물 | 자동화 | 견본 |
|---|---|---|---|
| A1 | 도구 카탈로그 → enum 스키마 | 기계적 | `taskbench/tb_guided_schema.py` |
| A2 | **정책/SOP NL → 제약 구조(게이트 입력)** | **★유일 연구-난제 = thesis의 학습 front-end** | SOP Guard-2 재구성(구조 제공시)·학습 front-end(NL시) |
| A3 | 평가기 → 보상·채굴 신호 (RFT/DPO) | 래핑 | `tb_dpo_mine.py`·sopbench_reward |
| A4 | 도메인 경계 → LODO 분할 | 기계적 | lodo 스크립트군 |
| A5 | 출력 스키마 → guided 문법 | 기계적 | schema `--dep` 분기 |

**주장 형태**: 새 벤치 커버 비용 = A1/A4/A5(기계적)+A3(래핑)+**A2** — A2의 자동화 정도가 기여의 크기(per-domain authoring 비용 제거).

## 3. 실험 벤치 포트폴리오 (2026-06-12 제안 — A2 난이도 스펙트럼 구성)
| Tier | 벤치 | A2 난이도 | 축 | 역할 (실증 규칙) | 상태 |
|---|---|---|---|---|---|
| 1 | TaskBench | 없음(정책 무) | 플랜-예측 | R1·R4·R5·R6·R7 | ✅ 완료 (외부 동결 — TB §1.5; 내부-일관 유지) |
| 1 | SOPBench | 구조 제공 | 실행-준수 | R2·R3 (per-constraint 진단 = 유일 자산) | ✅ 완료 (upstream 비유지보수·Amazon SOP-Bench와 이름충돌 주의) |
| 1 | **★τ²/τ³-bench** | **순수 NL 정책** | 실행-준수 | **A2 front-end 필요성의 실증 무대** + R3 + pass^k(게이트=일관성) | 신규 1순위 — 유일 활성 frontier 리더보드(~30모델 '26-05) |
| 2 | **Amazon SOP-Bench** | SOP 텍스트 | 실행-준수 | **R7을 12도메인 LODO로** (전이 스케일업, 2,000+태스크) | 신규 2순위 (어댑터 대부분 기계적·CC-BY-NC) |
| 2 | AppWorld | 없음(암묵) | 실행-검증 | R3 타-환경 (collateral-damage 단위테스트) | 신규 3순위 ('26-02 활성) |
| 3 | ODCV-Bench | 유혹-시나리오 | 거부-축 | "게이트=KPI-유혹 위반 0" (frontier 30-50% 위반) | 스팟 (40개·저비용) |
| 조건부 | WorFBench | 그래프 생성 | 플랜-예측 | TaskBench 후계 — 플랜-축 외부비교 복원 | 구조-축 조사 도착 후 확정 |

**선정 논리**: ①A2 스펙트럼 4점(없음→구조→semi-SOP→순수NL) 완성 ②τ²=활성 리더보드(외부 비교)+NL정책(front-end 데모)+pass^k(게이트 일관성 보상) ③Amazon=12도메인 전이 ④ODCV=결정론-leg 한 줄 헤드라인.

**실행 순서**: ⑴τ² retail 어댑터(A1 기계+A3 래핑+A2 수동-1회 — 수동본이 front-end 자동화의 GT가 됨)→7B+게이트 평가 ⑵Amazon 12-도메인 LODO 행렬 ⑶AppWorld·ODCV 스팟. 대형모델(32B+) arm은 Track-B(coworker) 분담 — `COWORKER_REQUEST_TB_SCALE.md` §8.

## 4. 커버리지 행렬 (벤치 × A1-A5 가용성 × R1-R8 적용처) — [작성 중: landscape census·구조-축 조사 도착 후 완성]

## 5. 메타 (인용·측정 규율)
- 벤치 수치 인용 함정 6종 = TB §1.5 ④ (재분할·백본 드리프트·지표 약화·파이프라인 재구성·in-domain 학습 비교·도메인 전치). ToLeaP "GPT-4o" 행 인용 금지.
- 신규 벤치 투입 시 절차: base census(R7) → 사전등록 → 어댑터 → 측정 → 궤적 census(R8).
