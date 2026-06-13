# CROSS-BENCH TRANSFER 계획 (2026-06-14, 세션 확정 — plan X)

> 마스터 = `EXPERIMENT_DESIGN.md` §0·§1(line28 벤치-횡단)·§1.5 포트폴리오·§10.5(R1-R8×A1-A5). 불변 = memory `feedback-thesis-tbox-transfer-direction`(★★★)·`feedback-selector-verifier-deterministic`(★★★).

## 0. 목표 (헤드라인 주장)
**멀티-벤치 학습 → held-out *벤치* 전이** = "벤치-일반"(LODO보다 강함). 다중-도메인 LODO 논리(§1 line24 "혼합→공통 불변량만 추출")를 다중-*벤치*로 격상.

## 1. 벤치 배치 (plan X — SOP-Bench는 테스트 측)
| | 벤치 | 역할 | 비고 |
|---|---|---|---|
| **학습** | SOPBench | R2(gather선행)·R3(결정 offload) | rule oracle=결정론 보상 ✓ |
| **학습** | TaskBench | R1(심볼/id grounding)·R4(하이브리드)·R6(K선별) | graph-F1=결정론 보상 ✓ |
| **테스트** | SOP-Bench(Amazon) | 표현호환(8관계 공유) = *쉬운* 벤치-횡단 전이증거 | 최종상태매칭 결정론 ✓ |
| **테스트** | τ² | native tool-calling·NL정책 = *어려운* stress-test | 활성 리더보드 |

- SOP-Bench를 *학습*에 넣지 않는다: 표현호환 = 가장 깨끗한 전이증거라 테스트로 보존. (capability 부족 시에만 학습 합류 = data-decided fallback; 단 SOP-Bench도 멀티턴 대화 아님 → τ² 대화갭은 학습벤치 수가 아니라 base가 채움.)
- 보상 = 3 학습벤치 전부 결정론(LLM-judge 보상 금지 = ★★★불변).

## 2. 아키텍처 결정 = native 재학습 (Option B) — 포맷매칭 공격 제거
- **probe 확정 (2026-06-14, GPU0 t1c LoRA·retail held-out)**: SOPBench `ready;op_X` 스캐폴드 LoRA가 τ² retail을 op-포맷으로 표현 시 — ①terse 단일-op 규율 *유지*(base는 step3서 붕괴) ②auth→order→변이→결제→exchange **상태추적 순서** 정확 ③전제 충족 시 목표 ACT(op_6)·base는 재인증(상태무시). ⇒ **연산자-시퀀싱 R-규율이 SOPBench→τ² retail로 전이 = 양성**(probe-tier·spoon-fed·1태스크 한정).
- **단 probe가 확인한 건 시퀀싱뿐**: 인자 바인딩·대화(ask-user)는 스캐폴드 모델 밖.
- **포맷매칭 공격(리뷰어)**: 브리지(op→도구 번역·스캐폴드 템플릿·인자 볼트온)가 **per-domain 분기/손규칙이면 = 도메인-베이킹 = 공격 성립**(login 특별취급 금지 동형). per-bench·도메인-일반이면 A1/A5 어댑터(허용).
- **결정 = Option B (native-tool-calling 재학습)**: TBox가 native tool_call 직접 emit → 포맷매칭 브리지 *통째 제거* → 공격표면 소멸. per-벤치 비용 = A1+A2뿐(수용된 어댑터 비용). probe(스캐폴드)는 "규율 실재" proof-of-concept로 인용.
  - 즉 검증된 시퀀싱 규율 + R1/R4/R6을 **native tool-calling 포맷으로 SOPBench+TaskBench 도메인 위에 재학습**.
  - 역할-파이프라인(멀티-LoRA on 공유 base) 옵션은 유지 — 단 인터페이스도 native로.

## 3. 컴포넌트 ↔ R/A 매핑 (불변)
- 학습 TBox(weight) = R1 grounding·R2 gather·R4 매칭·R6 *제안* (행동 규율).
- 결정론 스캐폴드(불변·학습 아님) = R1 enforce(XGrammar)·R3 게이트(GATE_SPEC replay)·R6 선별·검증기. [[feedback-selector-verifier-deterministic]]
- ABox(per-domain swap) = A1(도구 카탈로그·기계)·A2(정책→GATE_SPEC·front-end 난제)·A4(LODO 분할)·A5(출력 문법).
- 대화(ask-user) = base 모델 능력(도메인-일반).

## 4. 다음 행동
1. **표현/포맷 = native tool-calling 확정** → SOPBench+TaskBench를 native 포맷 통합 학습셋으로 빌드(공통표현 정의).
2. 학습 → SOP-Bench(직접 ABox-swap)·τ²(native, 브리지 불요) 전이 테스트.
3. F1 비용장부 갱신·ABox-ablation(빈/틀린 A1/A2→붕괴) 동반.

## 5. 공격 방어 체크리스트 (리뷰어 사전대응)
- [ ] 포맷매칭에 per-domain 분기 0 (grep `if domain`).
- [ ] ABox-ablation: 빈/틀린 A1/A2 → 붕괴 실측.
- [ ] 동일 시스템 unchanged로 retail+airline+telecom 작동(per-bench 1회).
- [ ] F1 비용장부: 브리지/어댑터 LOC < A2 자동화 속도.
- [ ] 학습벤치(τ²·SOP-Bench) contamination 0 (테스트 held-out).
