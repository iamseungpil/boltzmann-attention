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

## 1.5 ★개별화 기준 + 대수적 도출 (리뷰 #1·#2 — 포화를 반증가능·유한성을 구성적으로)
### (a) 개별화 기준 — "몇 개"를 비임의로 (리뷰 #1)
> **primitive = (i) 도메인-독립 control/data-flow 연산 AND (ii) 그 커버가 *다른* primitive 커버에서 zero-shot 전이되지 *않는* 최소 단위 (= separable learnability).**
- 기준 없으면 P5/P6/P8 합치거나 P3 쪼개 임의 포화 → 반증불가. (ii)가 그걸 막음.
- **이미 통과한 전이-테스트**: P5·P8 커버(SOPBench)에도 P6이 task17서 실패 = P6은 P5/P8서 전이 안 됨 = 분리 확인(우연 아님).
- **경계 정합**: 분할은 *게이트(집행)* 아닌 *모델(coverage)* 쪽. 게이트는 G1-G4 균일 집행(soundness)·모델은 "write엔 선행 confirm 필요"를 별 스킬(P6)로 학습(coverage). = thesis(coverage=모델/soundness=게이트) 정합.
- **★실험적 검증(선언 아님)**: leave-one-primitive-out ablation — X 커버·Y 미커버로 학습→Y 전이되면 병합(분리 아님)·안 되면 분리 확정. 분류=실험 대상.

### (b) 대수적 도출 시도 — 유한성을 "못 찾음" 아닌 "구성상" (리뷰 #2·최고가치)
가설: P1-P9 = tool-use 궤적(=tool 위 dataflow 프로그램)의 **연산자 닫힘**. 두 층:
- **층 A — dataflow-program calculus**: control{순차 P3·분기 P5·루프/에러분기 P7·병렬 P9} × data{produce-consume grounding P1·출력→결정 P2a·출력→인자 P2b·리스트→선택 P4}. = 고전 프로그램 구조(seq/sel/iter + dataflow)에 대응 → *구성상 유한*.
- **층 B — policy-overlay**: 게이트 타입(G1-G4)에 대응하는 정책 연산{인증/단일유저 P8·정책-사전조건 P5·confirm P6}. 게이트 타입 유한 → 유한.
- P7 = 층A의 루프/에러분기가 층B 게이트와 상호작용(deny→복구). 
- **닫히면**: 포화는 증명 아닌 *확인*으로 강등(훨씬 강함). **안 닫히는 P가 있으면** = "그건 post-hoc 실패 카탈로그"라는 유용 신호 → 분류 정직성 점검. ⇒ **이 도출 닫힘 검증이 단일 최고가치 다음 작업.**

## 2. 커버리지 매트릭스
범례(★리뷰 #3): **데이터-존재 vs 전이-획득 분리** — **✓**=데이터 존재(벤치가 보유)·**✓!**=*전이 검증됨*(eval 후만)·**◐**=약함·**✗**=부재·**?**=미검증. (V)실측·(R)문헌. ⚠️**대부분 ✓는 "데이터 존재"이지 "전이 해결"이 아님**(TaskBench 교훈: threading 데이터 있었으나 전이 0). **현재 ✓!(전이검증)=P1뿐**; P3/P5/P8=in-dist만(τ² 전이는 v6서 0=§8b R4 gap)·cfb P2b/P4=데이터존재(v7 eval 전 미확정).

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
- **★남은 gap (잠정·2개)**: **P6 confirm-gate** + **P7 recovery**.
- **★순환성 차단 (리뷰 #4a)**: "task17 gap=P6+P7"은 동어반복 위험. 탈출 = (i)**재발**(P6/P7이 task17 아닌 *여러* held-out·v4 20태스크 전반서 반복=아티팩트 아님) + (ii)**v7 예측 적중**(사전등록). §4 포화와 연결.
- **★"정확히 2개"는 미확정 (리뷰 #4b)**: τ² 필요-집합 완전성은 *autopsy한 task*(task17+v4 20개)서 나옴 — **τ² 전 task 유형 × P1-P9 census(zero-cost) 선행** 해야 "정확히 2"가 섬. 미-autopsy task가 P-something 요구하면 무너짐.
- ⇒ **v7 예측(사전등록)**: P2b/P4 ✓→✓!(전이검증)·order_id 날조↓·부분개선하나 **P6+P7 미커버라 τ² 완전돌파 아직**. 테스트 가능.
- **★P6 vs P7 획득경로 분리 (리뷰 #5·중요)**: 대칭 아님.
  - **P6=전방(proactive)**: confirm-then-write가 *gold 궤적에 존재* → 벤치/gold 소싱·SFT 가능(SOPBench 정책게이트·D5·When2Call).
  - **P7=반응형(reactive)**: deny/error에 대한 반응 → **성공-gold엔 절대 없음**(성공경로는 deny 안 당함). static-gold로 소싱 불가 → **gate-in-loop 데이터 필요**: (a)error-injection augmentation(SFT 근사·합성설계) + (b)**deny→recovery RL/DPO(원본)=FIELD_GAP Track B 재진입.** P6과 같은 줄에 두면 계획 틀어짐.

## 4. ★일반화 증거 = 유한 생성집합 + 포화(saturation) 실험
- **주장**: tool-use 스킬 = 유한 primitive로 생성. 증명 = **벤치 추가 시 *새 primitive 수 → 0***.
- **실험**: 벤치를 순차 추가(SOPBench→+TaskBench→+cfb→+BFCL→+RestBench→…), 각 추가서 *새로 필요한 primitive* 카운트 → 곡선이 포화(→0)면 유한 생성 실증.
- **전이 검증**: P1-P9 전부 커버 후 **새 held-out 벤치(미학습)가 추가데이터 0으로 전이**되면 thesis 성립. 어떤 벤치가 P10 요구 = 분류에 1행 추가(그 P10 쓰는 모든 벤치로 일반화) — *벤치별 아닌 primitive별 1회*.
- **현재까지 정황**: 새 벤치가 드러낸 primitive = R1(전이실증)→P2b(cfb)→P6/P7(autopsy). 기존 R1-R8 안 하위형으로 환원(새 R 0) = 포화 방향 시사(단 N 작음).
- **★적대적 포화 필요 (리뷰 #6)**: "새 P→0"은 추가 벤치가 *다양*할 때만 유의미 — **7벤치 전부 서비스-API 고객플로면 포화는 자명·무정보.** 강한 증거 = **적대적 벤치 탐색**(P1-P9 *밖* control/data-flow 패턴을 일부러 가진 벤치를 찾아 *실패 유도*). 발견 = 분류 +1행(여전히 유한)·반복 미발견(다양 벤치서) = 강한 유한성. **= finiteness의 *반증 시도***(self-fulfilling 회피).

## 5. scope 경계 (정직·리뷰 #6 — 무거운 짐 자인)
- **IN(경계 있는 슬라이스)**: tool-calling 에이전트의 control/data-flow(P1-P9). 헤드라인 = **"control/data-flow tool-use라는 실재하나 *경계 있는* 슬라이스에서 유한 생성 + held-out 전이"**(과대주장 금지).
- **★finiteness의 무거운 짐 = 제외 축**: 어려운 축(장기계획·코드실행·GUI-grounding·세션간 메모리·수치/기호연산)을 *뺐기에* 나머지 유한성이 그럴듯. 정직 박제.
- **★scope-vs-타깃 갭 (플래그)**: 실타깃 CDP가 계획/메모리를 요하면 thesis는 **CDP의 *일부만* 덮음** — scope와 타깃의 갭 명시.

## 6. 다음 (리뷰 6건 반영·우선순위)
1. **[#2·최고가치] 대수적 도출 닫힘 검증** — P1-P9를 §1.5(b) 두 층(dataflow calculus × policy-overlay)서 도출 시도. 닫히면 유한성 구성적·포화=확인 강등. 안 닫히는 P=정직 신호.
2. **[#4b·"정확히 2" 방어] τ² 전 task 유형 × P1-P9 census**(zero-cost) — 필요-집합 완전성 박제.
3. **[#3] v7 eval로 P2b/P4 ✓→✓!** 전환 + §3 예측 검증.
4. **[#5] P6(gold-소싱 SFT) vs P7(error-injection SFT + gate-in-loop RL=Track B) 분리 계획.**
5. **[#1] leave-one-primitive-out ablation**(분리 학습가능성 실험적 확정) — 여력 시.
6. **[#6] 적대적 벤치 탐색**(P1-P9 밖 패턴 유도) + 헤드라인 "경계 있는 슬라이스"로 고정.
3. saturation 곡선 시작(현 5벤치 매핑이 첫 점들) — 벤치 추가마다 새-primitive 카운트 박제.
