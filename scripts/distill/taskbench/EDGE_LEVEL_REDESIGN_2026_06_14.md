# 엣지-단위 재설계: 생성기/선택기/검증기 (2026-06-14, oracle 분석 기반)

> 사용자 근본비판 수용: plan-atomic oracle(=max 단일후보 F1)이 needle/selectable/gold-limited 분류를
> 다 어긋나게 한 근원. 정답 구조는 **엣지(부품) 단위**이고, oracle을 엣지로 분석하면 세 역할이 분리된다.
> **범위 규율(메모리 [[feedback-no-fundamental-claims-from-convenience-data]])**: 아래 *수치*는 이 임의 풀
> (dpo2g AR8 + hetero6) 한정 진단치. **설계 원칙은 일반**(엣지-단위 역할분리·빈도검증기), *수치*는 풀-종속.

## 0. oracle 재정의 — plan-atomic의 결함
- 구 oracle = `max(edge-F1(단일후보, gold))` = "계획 통째 고르기" 상한. 정답 엣지가 후보들에 *흩어지면* 과소평가.
- **조립-oracle** = (풀에 존재하는 gold 엣지)만 모은 F1 = 완벽 엣지-검증기가 있을 때의 진짜 천장.
- 실측(`tb_oracle_analyze.py`, n=293 비-단일노드): 조립 **0.858** > 단일최대 0.822 > 현 선별 **0.733**. ⇒ gold-limited("최선이 나쁨")는 **조립으로 복원 가능**한 케이스를 오분류한 것.

## 1. oracle 엣지분석 실측 (이 풀 한정 진단)
| 지표 | 값 | 역할 함의 |
|---|---|---|
| gold-엣지 커버리지 | **0.828** | **생성기 상한**: 17% gold 엣지가 풀에 부재 |
| 생성기-한계 task(cov<1) | **28%** | 이 task들은 *생성기*로만 — 없는 엣지는 선별/검증 불가 |
| 조립−단일최대 | +0.036 | 조립-선택기가 통째선택 위로 딸 여지 |
| **gold 엣지 ≥4소스** | **78.8%** | 정답 엣지는 강한 다중-지지 (선택기 실현성 高) |
| **wrong 엣지 1소스** | **74.2%** | 오답은 일회성 환각 = 빈도로 걸러짐 |
| 엣지-빈도 검증기 t=3 | **P 0.80 / R 0.85** | gold-free 엣지 검증기 실현 가능 |
| distractor 부하 | 정답 2.3 vs 오답 3.4 엣지/task | 검증기가 분리할 부담 |

**★핵심 전환**: plan 단위에선 agreement 무용(92% 만장일치)이었으나, **엣지 단위에선 빈도가 정답/오답을 강하게 변별**(74% 오답이 singleton). 분석 단위가 틀렸던 것.

## 2. 재설계 (세 역할 분리 + 선행연구 근거)

### 2a. 생성기 — 목표 = gold-엣지 커버리지 최대화
- 현 병목: 커버리지 0.828, 28% task 엣지 누락. **없는 엣지는 어떤 선별/검증도 복원 불가** = 이 풀의 진짜 하한.
- 설계: 엣지-커버리지를 올리는 **다양-정답 생성**(plan 다양 아닌 *엣지* 다양). = 생성기-다양 라인이 *엣지 커버리지* 수준에서 정당화(이전 "plan 다양 강등"과 무모순 — 단위가 다름).
- 선행: Large Language Monkeys(coverage가 샘플로 스케일)·Setlur 2502.12118(정답-trace heterogeneity가 검증이득 키 = 엣지판 D-oracle).
- 사전등록 지표: **gold-엣지 커버리지 ↑**(현 0.828)·생성기-한계 task% ↓(현 28%).

### 2b. 검증기 — 엣지 source-redundancy = gold-free 변별 신호
- 발견: 정답 엣지 79%가 ≥4소스 vs 오답 74%가 1소스 → **엣지-빈도(독립소스 수)가 P0.80/R0.85 검증기**.
- 설계: **엣지를 독립소스 수로 채점**(같은-정책 K샘플=1소스 = SEL-1 group 규율). 임계 t=3 채택(또는 redundancy를 연속 가중). plan-level agreement(무용)와 달리 **엣지 granularity서 작동**.
- 선행: Self-Consistency(Wang, exact-match voting)를 **그래프 엣지로 일반화**·"different-AND-right"(Heineman/Setlur)의 엣지판·imperfect-verifier(Stroebl)는 엣지서 FP율 낮음(오답 singleton).
- 사전등록: 엣지-빈도 검증기 P/R 곡선(현 t=3 0.80/0.85)·다른-base 추가 시 변별 ↑ 여부.

### 2c. 선택기 — 엣지-조립 (plan-atomic MBR 대체)
- 현 결함: 통째-plan MBR은 조립여지(+0.036)+선별갭(0.733→0.858)을 남김.
- 설계: **고-redundancy 엣지를 채택 → 유효 DAG로 조립**(검증기 2b로 엣지 채점 → validity 제약 하 조립). ⚠️**생성식 fusion(원칙#4 기각) 아님** — *기존 엣지 중 선택*+구조제약이라 합법 DAG 보장.
- 선행: Bertsch "MBR all the way down"(엣지 granularity)·AlphaCode filter→cluster→대표선택(엣지판)·XGrammar(조립 validity floor §7).
- 사전등록: 엣지-조립 선별 공식 link-F1 vs best-stack 0.6803(=plan-MBR). 조립-oracle 0.858 향한 회수.

## 3. 구 분류 → 엣지-단위 대체 (needle/selectable/gold-limited 폐기)
| 구 (plan-atomic, 폐기) | 신 (엣지-단위) | 누가 고치나 |
|---|---|---|
| gold-limited | 엣지 커버리지 < 1 (엣지 부재) | **생성기**만 |
| needle(통째 1출처) | 엣지 redundancy 낮음 | 생성기(커버리지) + 검증기(빈도 약) |
| selectable(통째 2+출처) | 엣지 present·multi-source인데 plan-MBR이 못 조립 | **선택기**(엣지-조립) |

## 4. 사전등록 실험 (차기, 대부분 zero-GPU)
1. ✅**엣지-조립 선택기 실행 (2026-06-14, `tb_edge_assemble.py`, zero-GPU)**: 검증기=엣지 redundancy(t 스윕)→DAG 조립→내부 edge-F1(oracle분석 척도, n=293). **결과 = plan-MBR과 동률**: best-stack 0.733 vs **엣지-조립 t=2 0.736**(+0.004·acyclic 0.738)·t=3 0.716·t=4 0.680·t=5 0.578. 조립-oracle 0.858 **헤드룸 미회수**. **★해석**: 빈도 검증기는 precision 0.71(t2)~0.89(t5)의 정밀-재현 트레이드오프에 갇혀 어느 t도 plan-MBR 벽(~0.73)을 못 넘음 = **합의/빈도는 plan이든 엣지든 ~0.73 포화**(같은 신호의 다른 granularity). ⇒ 엣지 재프레임은 *역할분리·오답=singleton*을 정확히 드러냈으나, **빈도-단독 엣지검증기로는 부족**. 0.858까지 0.12 갭 = **빈도 너머 정밀 검증기**(생성기-독립/결정론 구조체커) 필요. ⇒ 병목이 "검증기 정밀도"로 정확히 좁혀짐.
2. ✅**결정론 타입-검증기 스택 (2026-06-14, `tb_edge_assemble.py --typecheck`)**: 엣지 A→B 호환 ⇔ output-type(A)∩input-type(B)≠∅ (tool_desc 메타·결정론·무비용). **진단: gold 엣지 99.7% 통과 vs wrong 63.7%** = 오답 36% 포착·정답 보존. **빈도×타입 조립 t=2 = 0.748 = best-stack 0.733 +1.5pp = 처음으로 plan-MBR 초과**(빈도단독 0.736 위 +0.012). ⇒ **독립 gold-free 검증기 스택(빈도+결정론구조)이 합의-단독을 넘음 = 재설계 검증기 방향 입증.** 남은 갭(0.748→조립-oracle 0.858)=**타입-호환이나 틀린** 엣지(63.7%)라 의미검증(독립-base/LLM) 필요. 보너스: 결정론 구조검증 = thesis(결정론 검증기) 라인 정합. ❗**공식 metric 검증 = 반전(2026-06-14)**: 조립 엣지를 task_nodes `<node-j>` 참조로 재구성→공식 eval = **link-F1 0.6766 vs best-stack 0.6803(−0.4pp)**. **내부 edge-F1 +1.5pp는 척도 아티팩트**(내부≠공식), 공식선 동률~약간 하회. ⇒ **엣지-조립은 공식 metric서 plan-MBR 못 넘음** — 빈도든 빈도×타입이든 ~0.68 포화. **교훈: convenience-data 내부지표 과신 금지**([[feedback-no-fundamental-claims-from-convenience-data]]) — 공식검증 없었으면 +1.5pp 오보할 뻔.
3. **엣지-레벨 체인 최종(정직)**: 재프레임은 *분석적*으로 유효(역할분리·오답구조 규명)하나, **엣지-조립 선별은 공식서 plan-MBR 무초과**. same-base 합의/구조는 granularity 불문 ~0.68 포화 재확인. 헤드룸 = **독립검증(다른-base)** 또는 **생성 커버리지**(엣지 17%부재)로만 — 둘 다 새 자원(GPU/다른모델) 필요.
2. **검증기 변별 강화**: 다른-base 모델 추가 시 정답/오답 엣지 redundancy 분리 ↑? (cross-base가 singleton-오답 더 거름)
3. **생성기 커버리지**: 어떤 출처 추가가 gold-엣지 커버리지(0.828)를 가장 올리나 = 엣지-한계 28% 축소. (설계 변수)
> ⚠️전부 이 풀의 *수치*는 풀-종속 — 결론은 "엣지-단위 역할분리·빈도검증기" *원칙*으로 한정 보고.
