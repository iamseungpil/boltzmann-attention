# Content 생성원 대수 — 연역적 도메인-독립 연산자 집합 + 다중벤치 닫힘 (설계) — 2026-06-17

> 상위 = `PRIMITIVE_COVERAGE_MATRIX_2026_06_15.md`(flow 축 P1-P9·닫힘 도출됨) · `ALGEBRAIC_DERIVATION_CLOSURE` · `EXPRESSION_DIVERSITY_TRANSFER_DESIGN`(𝔥 표면축). 동기 = 사용자(2026-06-17): 생성원을 *벤치 귀납 아닌 연역*으로 규정·도메인독립·거의 모든 벤치 닫힘. + substitute "retail 특화" 공격 방어.

## 0. ★두 생성원 축 (핵심 프레임)
tool-use 절차 = 두 직교 축의 생성원으로 분해:
- **Flow 생성원 (P1-P9·`PRIMITIVE_COVERAGE_MATRIX`)**: *언제·어떻게* 도구를 엮나(sequence/branch/loop/par + provenance + gate). **대수적 닫힘 도출됨**(Böhm–Jacopini + provenance 완전분할 + 유한 게이트타입).
- **Content 생성원 (이 문서·op-IR)**: 데이터에 *무슨 연산*(select/aggregate/relative/transform). `PRIMITIVE_MATRIX §5`가 **명시적 scope-out**("transform 연산자 없음·data축=copy/select만"). = 우리 op-IR 작업이 채우는 누락 축.
- **τ² substitution 실패 = content 축**(flow는 닫혔으나 content 미도출). LLM이 content op를 *명명*·결정론 엔진이 실행(offload).

## 1. Content 생성원 연역 (벤치 아닌 연산 taxonomy에서)
출처 = 관계대수(Codd) + 함수형 변환 + 순서대수:
| 생성원 | 연산 | 연역 출처 |
|---|---|---|
| **filter** | 조건으로 항목 pin | σ select |
| **argmax/argmin** | 극값 | γ aggregate-extremum |
| **rank-k** | k번째 정렬 | sort∘select |
| **comparative** | anchor 상대 (just-above/below) | order relation |
| **substitute** | 기존 객체 일부 필드 override·나머지 유지 | record update |
| **create** | 빈 base에 필드 설정 (신규 객체) | record construct (substitute의 base=∅) |
| **project** | 출력서 필드 추출 | π project |
- **command-trivial**(toggle/invoke·인자 없음) = content-identity = **flow 축**(P-something), content 생성원 아님.
- **scope-out(정직)**: 수치/기호 *계산*(sum·format·산술)은 제외 — tool-call로 환원(`PRIMITIVE_MATRIX` seam β) 또는 별 축.

## 2. 도메인 독립성
- 생성원 = 순수 연산(item_id·zoom·mode 같은 도메인 어휘 0). **ABox config가 도메인 채움**(attr 타입·값 vocabulary).
- 학습 = 등방화 합성으로 생성원 *라우팅*(NL→어떤 content op). 도메인 사실 X. 전이 = ABox swap.

## 3. ★tau2 5도메인 실증 (substitute 도메인-일반 = 공격 반박)
| content op | retail(114) | airline(50) | telecom(2285) | banking(97) | mock(10) |
|---|---|---|---|---|---|
| selection(filter/argmax/comparative) | ✓✓(sup 62) | ◐(sup 10) | enum | ✗ | ✗ |
| **substitute** | ✓✓(exchange/modify·subst-kw 82) | **✓✓(update_reservation_*·subst-kw 18)** | ✓(set_mode=X single-field) | ✗ | ✓(update_task) |
| **create** | ✗ | ✓(book_reservation) | ✗ | ✓(apply_credit_card) | ✓(create_task) |
| command-trivial(flow) | ✗ | ◐ | ✓✓(toggle/reboot/grant) | ✓✓(call/unlock/verify) | ◐ |
- **★substitute는 retail·airline 둘 다 지배**(airline `update_reservation_flights/baggages/passengers` 전부 substitution)·telecom degenerate(single-field set)·mock도. = **트랜잭션 도메인 일반**(retail 특화 *아님*) = 사용자 공격 직접 반박.
- **content-rich(retail·airline) vs content-light(telecom·banking)**: telecom/banking = command/auth 지배(content-trivial) → flow 축(P1-P9)이 답. op-IR은 content-rich 벤치용.

## 4. 닫힘 + create 분리 검증
- **content 생성원 ~7개**(filter·argmax·argmin·rank·comparative·substitute·project·+create) 가 tau2 5도메인 content 연산을 *닫는 신호*(P10-content 부재·5도메인 전수).
- **create vs substitute 분리(leave-one-out)**: create=base∅ override·substitute=existing override. 별 생성원인지 = create 미커버 학습 → substitute서 전이되면 병합·안 되면 분리(§1.5a flow와 동일 기준).
- **닫힘 = 경험적**(선험 아님): 벤치 추가 시 새 content-op 수 → 0 곡선. SOPBench/TaskBench/CFB content 연산 매핑 추가 필요(다음).

## 5. ★공격 방어 (연역·유한·닫힘)
> "substitute는 τ² retail 특화 아니냐" → **"아니다. (1)substitute=관계대수 record-update, 벤치 보기 전 연산 taxonomy서 연역. (2)airline update_reservation·telecom set·mock update가 전부 substitute=도메인 일반 실증. (3)~7 content 생성원이 5도메인 content를 닫음(새 op 0)."**
- 핵심 = 생성원을 *벤치 귀납*(특화·공격O) 아닌 *taxonomy 연역*(일반·공격무효)으로.
- **위험 신호(정직)**: content-rich 벤치 추가마다 새 content-op면 무한→thesis 위험. 닫힘은 검증 대상.

## 6. 학습/검증 계획
1. content 생성원 집합 확정(§1) + ABox attr-타입 config(ordinal/categorical·comparative는 ordinal만).
2. 등방화 합성으로 **content-routing 학습**(NL→content op·여러 op 균형·substitute/create 포함).
3. **다도메인 전이**: 합성 학습 → retail·airline 동시(같은 라우팅·config swap). 닫히면 도메인-일반 입증.
4. **flow 축(P1-P9)과 합성**: content-rich 벤치는 content+flow 둘 다(op-IR + sequencing/gate). telecom/banking은 flow만.
5. **𝔥 표면 다양성(K-sweep)은 그 위**: content 생성원(𝔤) 완전 후 표면 다양화.

## 7. 정직 경계
- content 닫힘 = **가해(solvable) 연산 구간**(이론 §4)·무계 탐색·복잡 계산은 밖.
- tau2 5도메인은 *서비스-API 트랜잭션*에 편중 — 적대적 벤치(다른 content 패턴)로 닫힘 반증 시도 필요(`PRIMITIVE_MATRIX` §4 리뷰#6 content판).
- command-trivial=flow 분류는 경계 사례(set_mode=X는 substitute-single vs command 모호) — leave-one-out로 판정.
- SOPBench/TaskBench/CFB/BFCL content 매핑 미완(문서 기반·실데이터 분석 다음).
