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

### (b) 대수적 도출 — 유한성을 "못 찾음" 아닌 "구성상" (리뷰 #2·★도출 수행됨 2026-06-15)
> 형식 companion = `ALGEBRAIC_DERIVATION_CLOSURE_2026_06_15.md` (net-new 3: ①흡수메커니즘=branch/bounded-loop가 primitive 아닌 근거+§1.5a/b 통합 ②두 seam 화해=1차 live seam은 **층B 게이트유한**·data-transform은 census 조건부 2차 ③P8↔P1 provenance 중복=merge-후보). 본체가 권위.
> 가설: P1-P9 = tool-use 궤적(=tool 위 dataflow 프로그램)의 **연산자 닫힘**. 아래는 *선언이 아니라 구성*. 모델 = 궤적 = **effectful tool-call 노드의 DAG**, 각 노드의 인자슬롯은 value-source에 바인딩.

**층 A — dataflow-program calculus (control × data)**
tool-use 프로그램이 가질 수 있는 구조는 정확히 두 축뿐: (1)노드 스케줄링=control, (2)인자슬롯 바인딩=data.

- **control 축**: Böhm–Jacopini 구조정리 = 모든 제어흐름은 {순차·선택·반복}으로 생성. effectful-call DAG엔 **병렬**(동시성=관측가능) 추가 → 4 구성자.
  - 순차(의존 강제 순서) = **P3**. — DAG-edge의 *존재*(prereq 선행).
  - 선택(분기) = 조건 소스에 따라 갈림: 조건이 *데이터-술어*면 그 게더링 = **P2a**(출력→결정), 조건이 *정책-술어*면 = **P5**(층 B). ⇒ P2a와 P5 = "선택의 조건소스"의 두 맛(데이터 vs 정책).
  - 반복(루프) = tool-use서 유일 유의미 루프 = **에러/deny 재시도** → 올바른 수행(전략전환·무한루프 금지) = **P7**(층 A 루프 × 층 B verdict = 경계 연산자, 아래).
  - 병렬 = 데이터-독립 노드 동시(DAG 폭) = **P9**.
- **data 축 (produce→consume 관계)**: 인자슬롯 값의 provenance는 **완전 분할** — {context-상수, 상류출력-스칼라, 상류출력-컬렉션, *날조*}. 소비위치={인자, 조건}.
  - context-상수 → 인자 (복사) = **P1**(grounding).
  - 출력-스칼라 → 조건 = **P2a**.  · 출력-스칼라 → 인자(2-hop) = **P2b**.
  - 출력-컬렉션 → (선택 후 바인딩) = **P4**(P2a/P2b 위의 "cardinality>1" 수식자).
  - **★4번째 provenance = 날조 = ¬P1 = *실패모드*이지 새 연산자 아님.** ("주어졌거나 / 관측서 유도했거나 / 지어냈거나" — 셋째가 P1 위반.) value-*변환*(합·포맷)은 **scope서 제외**(§5 수치/기호연산) → data축=copy/select만, transform 연산자 없음.

**층 B — policy-overlay (게이트 타입당 1 primitive)**
agentic tool은 게이트로 보호됨. 게이트 타입(G1-G4 분류=auth / confirm-write / single-user·provenance / 정책-사전조건) 각각이 노드 실행 전 **사전조건 술어**를 부과. 모델측 커버 primitive = "그 사전조건을 가드된 호출 전에 확립":
- 신원·값-출처 확립(인증 선행·단일유저·값∈{user,tool}) = **P8** (G1+G3).
- 도메인 정책-사전조건 검사 = **P5** (G4).
- 비가역 write 전 user 확인 = **P6** (G2).
게이트 타입이 유한 → 층 B 유한.

**P7 = A×B 경계의 유일 연산자**: 반복(층 A) × 게이트-verdict(층 B). 이 도출이 *예측*하는 것: P7은 **반응형** → 성공-gold엔 deny가 없어 루프 본체가 절대 실행 안 됨 ⇒ **static SFT로 소싱 불가·gate-in-loop RL 필요**(§3 리뷰#5와 독립적으로 일치 — 경험 우연 아닌 calculus 귀결).

**닫힘 판정**
| 축 | 닫힘 근거 | 강도 |
|---|---|---|
| 층 A control | Böhm–Jacopini 정리 + 동시성(par) — 다섯째 제어구성자 없음 | **정리-기반(강)** |
| 층 A data | provenance 완전분할(given/scalar/collection/¬P1) · transform=scope-out | **완전분할(강)** |
| 층 B policy | 게이트-타입당 1 primitive · 타입 유한 | **유한 게이트 타입 *상대적*(약·경험적)** |

- **셀 ↔ primitive 대응 (10셀/9패밀리)**: 층A-data{P1,P2a,P2b,P4}+층A-control{P3,P9,P7경계}+층B{P5,P6,P8} = 10셀. 매트릭스 "9개"=P2a/P2b가 동일 data-관계(출력→소비, 소비위치만 차이)라 **P2 한 패밀리**로 묶임 → **도출이 lettering(P2a/P2b)을 정당화**.
- **결론**: 층 A는 **구성상 닫힘**(정리+완전분할). 층 B는 **유한 게이트-타입 상대 닫힘** — 여기만 soft. ⇒ 포화 주장의 유일 취약점 = **게이트-타입 유한성**(G5→P10 가능). 따라서 적대적 벤치 탐색(§4 리뷰#6)은 곧 **층 B 유한성의 반증시도**로 정조준됨(층 A는 정리로 잠김).
- **부산물(도출이 *예측*한 것, post-hoc 아님)**: ①날조 = "없는 4번째 data-source" = 실패모드의 구조적 위치(P1의 부정) ②P2a/P2b 묶음의 정당화 ③P7의 RL-필연성 ④P5 vs P2a = 선택의 정책/데이터 조건소스 이분.
- **남은 정직 잔여**: (i)층 B 게이트-타입 유한성=경험적 추측(적대탐색 대상) (ii)P2a vs P2b·P5 vs P2a의 *분리 학습가능성*은 구성상 별셀이나 실증은 leave-one-out(§1.5a·리뷰#1) 필요 (iii)transform 제외는 scope 자인(§5).

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
1. ~~**[#2·최고가치] 대수적 도출 닫힘 검증**~~ **✓완료(06-15)** — §1.5(b) 수행됨. **층 A 구성상 닫힘**(Böhm–Jacopini+동시성·provenance 완전분할)·**층 B 유한 게이트-타입 상대 닫힘**(유일 soft spot). 부산물=날조의 구조적 위치·P2 lettering 정당화·P7 RL-필연성. ⇒ 적대탐색(#6)=층 B 유한성 반증으로 재정조준.
2. **[#4b·"정확히 2" 방어] τ² 전 task 유형 × P1-P9 census**(zero-cost) — 필요-집합 완전성 박제. **★companion 추가: census는 두 seam 동시 시험** — (a)*계산된 인자*(obs→fn→arg) 요구 task 수=제약 S 시험 (b)G1-G4 밖 게이트 유형=층B 유한 시험. 0/0=닫힘 경험확정·>0=해당 축 P10 발화(여전히 유한).
3. **[#3] v7 eval로 P2b/P4 ✓→✓!** 전환 + §3 예측 검증.
4. **[#5] P6(gold-소싱 SFT) vs P7(error-injection SFT + gate-in-loop RL=Track B) 분리 계획.**
5. **[#1] leave-one-primitive-out ablation**(분리 학습가능성 실험적 확정) — 여력 시. **★companion 추가: 첫 정조준 쌍 = (P1, P8-provenance)** — P8의 provenance 성분=P1 무날조와 중복 ⇒ P1 커버 시 P8-provenance 전이되면 merge, auth-gate 성분은 별도 전이테스트.
6. **[#6] 적대적 벤치 탐색**(P1-P9 밖 패턴 유도) + 헤드라인 "경계 있는 슬라이스"로 고정.
3. saturation 곡선 시작(현 5벤치 매핑이 첫 점들) — 벤치 추가마다 새-primitive 카운트 박제.
