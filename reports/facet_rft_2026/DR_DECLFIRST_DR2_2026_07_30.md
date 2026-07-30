# 딥리서치 2차 — 선언-우선 잔여 문제 해법 + 공백 선점 검증 (2026-07-30)

> 실행: workflow `wf_781f7000-938`·13 agents(리서치 6·검증 6·종합 1)·오류 0·인용 24편 중 16편
> 개별 검증·exists=false 0. **종합 입력 절단으로 3개 축(whitespace·feedback·guide) 원본이
> 종합에서 누락/재구성됐던 것을 저널에서 복구·본 문서=완전판**(DR1과 동일 패턴·3회째 —
> 워크플로 개선 교훈: 종합 전 압축 단계 필요).
> 선행: `DR_BANLIST5_PRIOR_WORK_2026_07_30.md`(DR1). 적용 대상: `DECLARATION_FIRST`·
> `PATENT_SKETCH_FORMALIZE_BOUNDARY_TRIAGE`·논문 전략([[41]][[46]]).

## §0. 한 문단

**부품은 전부 선점·조합은 미발견 — 그리고 우산 명제(W5)는 open이다.** ①수요-원장: 분해
(TICK)·checklist→tool 대조(Gecko)·결정론 pre-write gate(2607.07405 +10~12pp)는 있으나
"quote-검증 원장+결정론 coverage+DONE 차단"의 runtime 합성은 미발견. ②수치: PoT/PAL이
"숫자=interpreter만"을 선점했으나 **다회턴 산문 재진술 숫자의 원장 대조는 어떤 선행도 없음**
(가장 선명한 공백). ③봉투: typed 태그가 선언→행동 불일치를 22–26%→0.7–1.4%로 붕괴시키고
(2606.00476), false success=tau2 실패의 45–48%인데 **LLM judge는 AUROC≤0.65로 못 잡고 구조
신호는 잡는다**(2606.09863·Trajel 0.908) → done_report의 in-loop 결정론 대조가 우리 델타.
④단 봉투 자신의 부작용(constraint tax·tool 호출 억제) 관리 필수 — 등대 모트 원리의 외부
실증. ⑤피드백: **누출-등급 분류법 = 미선점**(신규 기여 후보 추가).

## §1. 수요-원장 (EPLAN 수요측 대체)

- 지형: TICK/STICK(요청→checklist 분해·인간 일치 46.4→52.2%·self-refine +7.8%p
  [2410.03608])·**Gecko [2602.19218]=최근접 rival**(대화→checklist→tool-call 대조·단 LLM
  judge·시뮬레이션 refinement용)·compositional parsing(TOP/DMR/MULTI3NLU++=수량사·부정·복수의
  typed 표현 — regex 파싱 부적절성의 문헌 근거)·evidence-span 강제(Paper2Data·LAQuer).
- **결정론**: quote∈utterance=닫힌 술어 / coverage=원장 잔여 set-difference→DONE gate /
  수량사('all')의 **외연은 GET/FIND 후 DB 결과 위에서 엔진이 전개**(파서 단독으론 안 닫힘).
- LLM: 분해·추출(선점·인용 소비). 잔여 열림="quote는 진짜인데 해석 과잉"→CONFIRM/learn.
- [[21]] 정합: user-sim이 요구를 번복해도 원장 append/retract 이벤트가 진리원.
- 권고: 등재 조건=quote 필수+결정론 검증 / 2607.07405(over-action gate)의 **under-action
  대칭**으로 포지셔닝 / 필수 인용: Gecko·TICK·2606.09863(motivation).

## §2. 수치-주장 (NLNUM 대체)

- 지형: HERMAN(2020·수치 환각=별도 클래스)·FinQA(모델 61% vs 인간 91%·3+스텝 ~22% 붕괴)·
  PoT [2211.12588]/PAL [2211.10435](계산=interpreter 축출·~12% 이득)·ClaimDB·ATLAS-RTC
  [2603.27905·**전독 전 [?]**]. 주류 완화=학습 검증기의 산문 사후-스캔 = 우리가 금지한 패턴.
- **결정론**: declared number == ledger value(정규화 후 equality·정규화는 선언값에만·산문
  불독해). FinQA execution-accuracy 검사를 eval-time→runtime 봉투로 이동.
- 채널 구성이 "no mental arithmetic"을 강제: 숫자를 말하려면 먼저 원장에 존재해야(calc 도구
  경유) — PoT 원리의 다회턴 확장. learn 보완=Toolformer(라우팅 prior·보장 아님).
- ⚠인용 금지: FinAgent-RAG 2605.05409(**저자 철회**). 보조 근거: 2605.06635(frontier
  인용-사실 정확도 39–77%=수치 전용 채널 필요성).

## §3. 봉투-충실도 (아키텍처 핵심 실증)

- **[2606.00476] 제1 실증**: typed 태그 선언→행동 불일치 0.7–1.4%(free-text 추출 22–26%
  대비)·기계 강제가 규칙 오적용 13.9→6.8%. **잔여 불충실=formalize 단계(reasoning→선언)에
  집중** = 우리 "선언 충실도=유일 잔여" 명제의 외부 확인.
- **[2606.09863] motivation 정본**: false success=tau2 single-control 실패의 45–48%·AppWorld
  75.8% / **LLM judge 전 구성 AUROC≤0.65**(ground-truth spec 줘도·confident-closing 표면
  proxy에 정박) / 경량 구조 detector 0.83–0.95·~3300배 빠름 / reasoning 모델도 무방비(79%).
- [Trajel 2605.24219]: procedural hallucination(스텝 건너뜀/날조)=최대 클래스 38.5%·구조
  신호 AUC 0.908 > 학습 분류기 0.689 > LLM judge κ=0.456.
- 선례 상보: 2607.07405(행동만 검사)·OAP [2603.20953](인가 53ms·adversarial 0% vs 모델 단독
  74.6%)·Delegation Envelope [2604.25000](act/ask/escalate 타이핑·단 principal-부과).
- 충실도 측정 프로토콜 이식 가능: hint-injection(thinking 87.5% vs answer 28.6%=59pp 괴리
  [2603.26410]·원조 [2307.13702]).
- learn: 엔진 모순 검사=무료 라벨 → (선언,행동)-일관성 GRPO(2512.22631: GRPO>DPO·DPO=스타일만
  위험).

## §4. 피드백 설계 (오류 계약의 정본 — 복구 완전판)

- **누출-등급 분류법 = 미선점**(신규 기여 후보): SEAL [2605.24426]=금지 리스트만(정답 콜·
  인스턴스 인자값·gold 궤적 금지 / 오류 클래스·위반-제약 식별·공개 스키마 단서 허용) —
  분류법+회복-오염 crossover 측정은 없음.
- 실측: 구조화 진단+**admissible 대안 집합**=+42–44pp 종단 성공(2607.14167)·상세도는 단조
  유익(2606.01522). **최대 지렛대=최대 누출 위험이 같은 필드**: admissible-set은 "닫힌·
  인스턴스-독립 집합"일 때만 정당(스키마-파생) — 인스턴스 값이면 oracle 누출. **= PREKB
  스푼피드 문제의 정확한 일반 이론.**
- **ITS 사다리를 결정론 에스컬레이션으로**: 첫 실패→위반-제약 식별(why·정답 금지) / 같은
  검사 반복 실패(구조 카운터=결정가능)→한 단 승급(스키마-파생 admissible set 추가).
- 감사 가능성: 하네스-누출 포렌식(2604.11806·9벤치 28제출서 광범위 cheating 발견) → 엔진
  메시지의 결정론 self-audit(내용이 gold/인스턴스 값 포함하면 플래그) 추가 가능.

## §5. 공백 선점 판정 (원본 복구 — 종합의 재번호와 별개·발주 원안 기준)

| # | 후보 | Verdict | 최근접 | 남는 기여 포지션 |
|---|---|---|---|---|
| W1 | slot-소스 구분(producer-도구 vs 유저-발화) 기반 중복-질문 판정 | partially | LHAW·RegretBench·SAGE·TRACER(도구 측만) | **소스-타입 원장 규칙**으로 주장(중복-질문 탐지 일반으론 포화) |
| W2 | 유저-위임 실행=1급 plan 노드+이벤트-검증 진행 추적 | partially | tau2(환경만)·MUA-RL(e2e 학습·구조 노드 없음)·LangGraph interrupt(비학술) | **정식화+이벤트-검증 진행+실측 효과**를 주장(개념 아님) |
| W3 | grace-turn(K-턴 유예+typed escape 후 강제) | partially | RegretBench(하네스 측 캡)·질문-예산 sweep류 | **agent-측 결정론 정책+K ablation**은 방어 가능 |
| W4 | confirm(filled)=합법 / request(filled)=위반의 act-타입 결정론 규칙 | partially | DST act 구분(고전)·Know-Your-Mistakes·PolicyGuard(LLM 판정) | **DA 타이핑의 scaffold-집행 이식**으로·인용 두텁게 |
| **W5** | **우산: 전 집행=선언↔이벤트 정합·산문 해석 0·스키마-컴파일 문법** | **open** | AgentSpec [2503.18666](이벤트-규칙 DSL·강제 선언 없음)·Behavioral Contracts [2602.22302]·**PolicyGuard=foil**(LLM 산문-해석 집행=명시적 비결정론)·XGrammar(부품) | **조합-수준 신규성 성립** — (a)강제 typed 선언 (b)스키마-컴파일 문법 (c)선언↔이벤트-정합-만-검사 엔진의 결합은 미발표 |

- 종합 agent의 재번호 판정(수요-원장 조합·수치 선언-채널·재진술 충실도·봉투 조합)도 유효 —
  **"재진술 충실도"(다회턴 산문 숫자↔이전 tool output 대조)가 가장 깨끗한 단독 공백.**
- 선점 확정(주장 금지): 선언-divergence 측정 일반(2606.00476이 정확히 측정)·수치 provenance
  개념(2605.06635)·산문+JSON 혼합 디코딩 기법(아래 §6).

## §6. 구현 실무 확정 (guide 축 복구 완전판 — E-DECL-COMP 설계 입력)

1. **혼합 생성=기술 확정·발명 금지**: 비제약 산문 영역 + 트리거-토큰 활성 엄격 봉투 tail
   (In-Writing식 eos-트리거 > CRANE 구분자·100% 파싱). two-pass(2606.25605 권고)와 tail-제약
   단일-pass를 **E-DECL-COMP arm으로 비교**.
2. **schema-in-prompt 단독은 우리 스케일에서 비현실**: 8B서 어려운 스키마 준수 ~90→13% 붕괴
   / 문법 엔진은 유지+토큰당 ~50% 빠름+정확도 소폭 상승 → guided 필수(§6 기존 결론 강화).
3. **장기-지평 리스크=의미 준수**(문법 아님): 멀티턴 준수 평균 −39%·knows-but-violates
   8–99% — 문법이 봉투 형식을 턴-불변으로 만들고, 남는 것은 §3 모순-검사 몫.
4. **모델 제약**: logit-mask 제약 디코딩은 MoE서 붕괴(Mixtral 0.41→0.24) — dense 체크포인트
   사용(우리 Qwen2.5-32B=dense·문제 없음).
5. **엔진 주의**: JSON-Schema 기능 커버리지가 엔진 간 ~2배 차이(JSONSchemaBench·6엔진) —
   봉투 스키마는 채택 엔진의 지원 subset 안으로. + 제약 엔진 신뢰 금지=post-parse 결정론
   재검증 상설(2605.26128: 하드 제약 하 소형 91.5→48.0% 붕괴 사례).
6. **우리 무료 실험 2건이 문헌 공백**: (i) **schema-as-declarative-data vs 하드코딩의 교차-
   도메인 적응 비교 — 통제 연구 부재** = A2-swap 주장(특허 유한성·논문)의 직접 실험 기회
   (ii) 제약 봉투가 공존 산문 채널 품질을 깎는지(Δprose 계측).
7. 재생성 분산 큼(2605.28840) → 준수 측정=다중 시드·재생성 상한+ASK fallback.

## §7. 논문 지형 최종 (DR1+DR2 합본)

| 논문 후보 | 지위 |
|---|---|
| **①시스템 우산**: 선언-우선 scaffold(W5 open·2607.07405의 대칭·일반화 프레임·motivation=2606.09863 무료) | **성립** — 부품 전부 양보·조합+경계논증([[22]])이 모트·[[46]] 결 |
| **②재진술 충실도**(다회턴 산문 수치↔원장) | **성립** — 가장 깨끗한 "새 문제" |
| **③누출-등급 피드백 분류법**(회복-오염 crossover 측정) | **성립 후보**(§4·SEAL=리스트만) |
| W1~W4 각각 | 단독 ✗ — ①의 종속 기여/ablation 절로 |
| 값 라우팅(DR1 공백) | ①에 흡수 or W2 정식화로 ①의 한 축 |
| 분해·화이트리스트·오류계약 기법 자체 | 인용-소비 엔지니어링 |

- **Rival 감시**: Gecko(runtime gate로 승격해 오면 W1 잠식)·ATLAS-RTC(**전독=최우선 후속**·
  W2/수치 diff 미확정)·Paper2Data(quote 기전 abstract 미확증 — **전독 전 양보 금지**).

## §8. 인용 신뢰도

- 24편 중 16편 개별 검증 전부 실존·exists=false 0. 완전 지지(수치까지): TICK·Gecko·
  2607.07405·2606.09863·2606.00476·Trajel·OAP·HERMAN·PoT·2605.06635 등.
- **⚠금지**: FinAgent-RAG 2605.05409=철회. **⚠[?]-급(전독 필수)**: Paper2Data 기전·ATLAS-RTC
  기전·2605.26128 long-tail 세부(tradeoff 자체는 강확인). FinQA 수치=본문-수준(61.24/91.16
  정합). 미검증 저위험(기지 논문): Toolformer·PAL·DMR 등.
- 원장 등급 규율: 완전 지지=[S-lit]·본문-수준=[M]·snippet·미확증 기전=[?] — [?]를 [M]처럼
  쓰지 말 것.
