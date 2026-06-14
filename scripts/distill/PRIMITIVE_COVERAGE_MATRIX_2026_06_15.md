# Tool-use Primitive 분류 × 벤치 커버리지 매트릭스 (2026-06-15)

> 상위 = `V7_PROACTIVE_GATHER_DESIGN_2026_06_14.md` · 동기 = "추론룰이 무한하면 whack-a-mole 아닌가"(사용자). 답 = **tool-use는 유한 primitive로 생성된다는 가설**을 매트릭스로 명시·검증. 불변 = memory `project-framework-goal-bench-invariant-rules`(R1-R8).

## 0. 핵심 명제
- **whack-a-mole의 정체 = 무한 스킬이 아니라 유한 primitive의 *불완전 커버리지*.** 2-hop gather는 새 스킬이 아니라 **P2b(gather-for-arg)** 하위형 — v6 실패는 그 하위형 미커버 탓.
- ⇒ 커버는 **primitive별(유한 ~9)**, 벤치별(∞) 아님. 전 primitive 커버 후 held-out 벤치가 무재학습 전이 = thesis. 새 벤치가 P10 요구 = 분류 미완성 신호(한 번 채우면 그 P10 쓰는 모든 벤치로 일반화).
- **scope 경계(과대주장 금지)**: 이 유한성은 **tool-calling 에이전트의 control/data-flow primitive**에 한정. 밖(장기계획·코드실행·GUI-grounding·세션간 메모리·수치연산-heavy)은 별 축·환원 불가 가능.

## 1. Primitive 분류 (R1-R8 정련 — 도메인-독립 control/data-flow 연산)
| P | primitive | 정의 | R-매핑 |
|---|---|---|---|
| **P1** | grounding/무날조 | 인자=컨텍스트서 복사(tool-name·arg-value)·지어내지 않음 | R1·R1b |
| **P2a** | gather-for-decision | getter 출력 → 사전조건/결정 입력 | R2 |
| **P2b** | gather-for-arg (2-hop) | lookup 출력 → *하류 인자*(추론+관찰) | R2+R4 |
| **P3** | 시퀀싱/의존 | DAG 순서·prereq 선행 | R4·R6 |
| **P4** | select-from-output | 관찰한 리스트/출력서 옳은 항목 추출 | R4 |
| **P5** | policy-gating | write 전 정책 사전조건 검사·결정 offload | R3 |
| **P6** | confirm-gate | 비가역 write 전 user 확인 | R3(하위) |
| **P7** | recovery | 에러/게이트 후 전략전환(re-gather/ask)·무한루프 금지 | R3(하위) |
| **P8** | provenance/auth | 값 출처=user/tool만·단일유저·인증 선행 | R1b·G1·G3 |
| **P9** | parallelism | 독립 호출 동시(DAG 레벨) | R6 |

## 2. 커버리지 매트릭스
범례: **✓**=강함/명시 · **◐**=약함/부분 · **✗**=부재 · **?**=미검증추정. (V)=우리 실측 검증 · (R)=딥리서치/문헌추정.

| | P1 | P2a | P2b | P3 | P4 | P5 | P6 | P7 | P8 | P9 |
|---|---|---|---|---|---|---|---|---|---|---|
| **SOPBench**(train,V) | ✓ | ✓ | ◐(1.9%) | ✓ | ◐ | ✓ | ◐ | ◐(success-only=thin) | ✓ | ✗ |
| **TaskBench**(train,V) | ✓ | ✗ | ◐(symbolic·plan-given) | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ |
| **ComplexFuncBench**(train,V) | ✓ | ◐ | ✓✓(grounded·inferred·100%) | ✓ | ✓ | ◐ | ✗ | ?(미검증) | ✗ | ◐ |
| **τ²**(test,V autopsy) | 필요✓ | — | **필요✓(v6 binding)** | 필요✓ | 필요✓ | 필요✓ | **필요✓(G2)** | **필요✓(retry-loop)** | 필요✓(G1/G3) | — |
| **SOP-Bench**(test,R) | 필요✓ | 필요✓ | 필요? | 필요✓ | ? | 필요✓ | ? | ? | 필요✓ | ? |
| **BFCL V3 mt**(cand,R) | ✓ | ◐ | ◐(state-val·id아님) | ✓ | ◐ | ✗ | ✗ | ? | ✗ | ◐ |
| **RestBench**(cand,R) | ✓ | ◐ | ✓(inferred id) | ✓ | ◐ | ✗ | ✗ | ✗ | ◐ | ✗ |

(기각: Seal-Tools=TaskBench와 동(P2b symbolic)·신규0 / NESTful=P2b 수학만)

## 3. ★진단 — τ²를 위해 학습이 커버한 것 vs 남은 gap
τ² 필요 = P1·P2b·P3·P4·P5·P6·P7·P8.
- **커버됨**(train 합집합): P1(SOP/TB/cfb)·P3(SOP)·P5(SOP)·P8(SOP)·**P2b(cfb 추가=v7)**·**P4(cfb)**.
- **★남은 gap (유한·2개)**: **P6 confirm-gate** + **P7 recovery**. autopsy task17이 정확히 이 둘(G2 확인 무시 + 同호출 8연타)서 실패.
- ⇒ **v7 예측(매트릭스 기반)**: P2b/P4 닫혀 order_id 날조↓·부분 개선하나, **P6+P7 미커버라 τ² 완전 돌파는 아직**(특히 confirm-gate write·에러 복구 태스크). = 테스트 가능한 사전등록 예측.
- **다음 work-list = P6, P7** (벤치 무한 아니라 primitive 2개). P6=write-confirm 궤적(SOPBench 정책게이트 활용·D5류)·P7=에러→복구 augmentation(autopsy Q2/Q3·A2 정의 retry).

## 4. ★일반화 증거 = 유한 생성집합 + 포화(saturation) 실험
- **주장**: tool-use 스킬 = 유한 primitive로 생성. 증명 = **벤치 추가 시 *새 primitive 수 → 0***.
- **실험**: 벤치를 순차 추가(SOPBench→+TaskBench→+cfb→+BFCL→+RestBench→…), 각 추가서 *새로 필요한 primitive* 카운트 → 곡선이 포화(→0)면 유한 생성 실증.
- **전이 검증**: P1-P9 전부 커버 후 **새 held-out 벤치(미학습)가 추가데이터 0으로 전이**되면 thesis 성립. 어떤 벤치가 P10 요구 = 분류에 1행 추가(그 P10 쓰는 모든 벤치로 일반화) — *벤치별 아닌 primitive별 1회*.
- **현재까지 정황**: 새 벤치가 드러낸 primitive = R1(전이실증)→P2b(cfb)→P6/P7(autopsy). 이미 *기존 R1-R8 안*에서 하위형으로 환원됨(새 R 0) = 포화 방향 시사(단 N 작음·계속 측정).

## 5. scope 경계 (정직)
- **IN**: tool-calling 에이전트의 데이터/제어 흐름(P1-P9). 헤드라인 = "이 부분공간서 유한 생성 + held-out 전이".
- **OUT(별 축)**: 장기 multi-step 계획·코드실행 에이전트·GUI/컴퓨터유즈 grounding·세션간 장기메모리·수치/기호 연산-heavy. R1-R9 환원 불보장 → thesis 범위 밖 명시.

## 6. 다음
1. P6/P7 소스 결정(SOPBench confirm-gate·error-recovery augmentation = 우리 자산서 가능 = clean·특허OK).
2. v7 결과로 §3 예측(P2b 닫힘·P6/P7 잔존) 검증.
3. saturation 곡선 시작(현 5벤치 매핑이 첫 점들) — 벤치 추가마다 새-primitive 카운트 박제.
