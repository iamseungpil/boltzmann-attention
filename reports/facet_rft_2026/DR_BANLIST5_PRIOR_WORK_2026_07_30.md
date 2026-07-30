# 딥리서치 — 금지선 5종의 선행연구 지형·해법 (2026-07-30)

> 실행: workflow `wf_df853bf1-7a4`(2026-07-29 발사)·11 agents(리서치 5·인용검증 5·종합 1)·오류 0.
> 방법: 주제별 웹 리서치(arXiv/ACL/벤치 정독)→인용 실재·지지 검증→종합. **종합 입력 절단으로
> §3(VALUE_ACQUIRE)·§5(GIVE_RELEVANCE) 원본이 누락됐던 것을 저널(journal.jsonl)에서 복구·본
> 문서에 완전판 반영**(리서치·검증은 5/5 완료돼 있었음).
> 인용 등급: 검증 agent가 실재+abstract 지지 확인한 것만 [V]·snippet-only/미검증은 명시.

## §0. 한 문단 결론

**다섯 문제 모두 산문 매칭으로 푸는 계보가 선행연구에 없다**(부재 자체가 폐기 지지). 표준
해법은 일관되게 **탐지→예방: 판정 대상을 산문에서 구조 이벤트로 옮기는 재구조화**다 —
(i) act-vs-talk 타이밍 = 닫힌 act 인벤토리 + 스키마-유도 eligibility(HCN action-mask·SGD·
AnyTOD) (ii) 재질문 = 구조 이벤트로만 채워지는 slot ledger 표면화(CALM·PyTOD) (iii) 환각
도구명 = blocklist가 아니라 **스키마 화이트리스트**(ToolDec·constrained decoding) (iv) 잔여
의미 판단 = LLM sub-call formalize·hidden-state probe·preference 학습(When2Call RPO·SMART-ER·
Fission-GRPO). **tool_choice=required 무조건 강제는 유해로 문서화**(jailbreak 벡터·파라미터
환각 유발) — 강제는 "구조적 eligibility 성립 ∧ ask-escape 보존" 상태에서만. 레버 4개(1·2·3·5)는
**"typed act 인벤토리 + 스키마-유도 slot/eligibility ledger" 단일 아키텍처**가 동시에 닫는다 =
우리 선언-우선 설계(`DECLARATION_FIRST`)와 동형.

## §1. FORCE_ACTION → act-timing 제어

- **지형**: 30년 된 TOD dialogue policy 문제. HCN action-masking [arXiv:1702.03274]·SGD 닫힌
  system-act 인벤토리 [AAAI 2020]·**AnyTOD neuro-symbolic 분할** [arXiv:2212.09939]·StateFlow
  FSM [arXiv:2403.11322]·plan-then-execute(ReWOO [arXiv:2305.18323]·LLMCompiler
  [arXiv:2312.04511])·When2Call [arXiv:2504.18851]. 전부 [V].
- **결정론**: eligibility 술어를 스키마+구조 이벤트(인자 접지·확인 이벤트)에서 컴파일해
  ACT-vs-SAY 마스킹. 최강 보장 = 실행 권한을 토큰 스트림 밖으로: LLM은 plan node만 방출·
  결정론 스케줄러가 실행(say-without-do 구성상 불가·LLMCompiler: ReAct 대비 정확도 ~+9%·
  지연 3.7×·비용 6.7×) — **E-PLAN/TRACK-A 방향의 외부 실증**.
- **LLM**: AnyTOD 분할 = LLM sub-call이 대화를 심볼 이벤트로 formalize·도메인-일반 심볼
  프로그램이 선언 정책을 읽어 SAY/ACT 결정 — **우리 §2 컨트롤러(user_act formalize→결정론
  라우팅)와 정확히 동형**. 신형 신호: pre-generation hidden-state 선형 probe(tool 필요성
  AUROC 0.89–0.96·불필요 호출 −48%·정확도 −1.7%) [arXiv:2605.09252].
- **learn**: When2Call RPO(4-way call/ask/abstain/answer·4B 29.7→51.0 F1·**SFT는 과보수화**)·
  ACT contrastive DPO [arXiv:2406.00222]. 데이터가 스키마-생성이라 도메인-일반([[11]] 합치).
- **★경고**: tool_choice=required 무조건 강제 = jailbreak 경로+파라미터 환각 유발 문서화
  [When2Call] → 강제 채널은 eligibility+escape 조건부로만.

## §2. HAVE_VALUE → 재질문 예방

- **지형**: slot memory 문제. Rasa CALM(LLM=DSL 번역기·결정론 dialogue manager)
  [arXiv:2402.12234]·PyTOD(실행 가능한 프로그램 상태+execution feedback) [arXiv:2508.15456]·
  MemGuide [arXiv:2505.20231]·RegretBench [arXiv:2607.21143]·NoisyToolBench [arXiv:2409.00557].
  **재질문을 산문 매칭으로 탐지하는 논문 = 전무.**
- **결정론**: ①구조 이벤트로만 채워지는 typed slot ledger("filled"=tool-call 레코드 위 닫힌
  술어·producer→slot 매핑=A2 데이터) ②**known-values 블록 표면화**("known: card_last4=1234
  (from get_card)")가 재질문을 멈추는 표준 치료 ③re-ask vs confirm 구분 = act 타입으로 구성적
  해결: request(slot)=미충전에만·confirm(slot=value)=충전-미검증에만 합법.
- **LLM**: 발화→typed act 분류 sub-call(유일한 환원 불가 의미 단계)·CLAM식 사전 게이트
  [arXiv:2212.07769].
- **learn**: SAGE — 구조적 belief-state 위 EVPI + **redundancy cost Σn_a(같은 인자 ask-횟수
  결정론 벌점)**·clarification 1.5–2.7×↓·When2Call 36.5→65.2(3B·GRPO) [arXiv:2511.08798].
- **whitespace**: "producer 도구가 이미 반환한 값" vs "유저가 이미 말한 값" 구분을 다룬 논문
  미발견 — 우리 producer→slot 스키마 매핑은 신규 기여 후보.

## §3. VALUE_ACQUIRE → 값-획득 라우팅 (복구 완전판)

- **지형**: RavenClaw/SGD의 slot-source 라우팅(진행=slot-binding 구조 이벤트·엔진 고정+선언
  task tree = **[[05]]와 동형**) [CS&L 2009·AAAI 2020]·HammerBench [arXiv:2412.16516]·
  Ask-before-Plan [arXiv:2406.12639]·IAL 무한루프 구조 정의 [arXiv:2607.01641]·**tau2-bench
  자체** [arXiv:2506.07982]. 검증 10/10 실재.
- **결정론 (3개 수입품)**:
  1. **missing-args = diff**: 매 스텝 구조적 draft call vs 선언 스키마의 rule-based 비교
     (HammerBench PN_MR/PN_HR·snapshot 방식 84% vs ask-프롬프트 68%) — 우리 봉투 `next_action`
     선언과 동형.
  2. **"사용자가 했는가"는 산문이 아니라 이벤트**: tau2 dual-control이 user-측 tool-call
     레코드·상태 변화를 이미 노출 — give 이후 (user 실행 이벤트 ∨ 대상 인자 binding) 전까지
     재질문 게이트·k턴 무-binding이면 라우팅 전환. 전부 구조.
  3. **stuck = 진행 술어**: progress=신규 binding(구조)·같은 인자 ASK k회+binding 0 = stuck
     확정(SAGE Σn_a·IAL dedup과 동형). **단 ASK가 엔진-매개 typed 이벤트로 승격돼 있어야
     카운터가 닫힌다** = 선언-우선의 ask 선언이 전제.
  - 라우팅 순서 자체도 도메인-무관 상수로 선언 가능: own-getter→호출 / user-tool 선언→give
    후 대기 게이트 / user-knows→ASK(한도付) / 한도 초과→강등·escape.
- **learn**: When2Call RPO(공개 데이터·NVIDIA)·Ask-before-Plan trajectory tuning(대화+상호작용
  기록 조건부 clarify-vs-execute·"LLM은 자발적 clarify를 못 한다" 실측).
- **whitespace**: **유저-위임 실행(give) 라우팅을 1급으로 다룬 agent-측 방법론 부재**(환경
  측은 tau2뿐) — [[41]] 방향의 신규성 후보.

## §4. UNKNOWN_REPEAT → 화이트리스트 + 오류 계약

- **결정론**: ①**스키마-유도 화이트리스트**(blocklist의 쌍대·과거 거부 기록 불요 = perseveration을
  패치가 아니라 해소): decode-time FSM(ToolDec: 환각 도구명 0·Mistral 0%→52%) [arXiv:2310.07075]
  or pre-execution membership 게이트 ②**오류 계약**: bare "Unknown tool" 대신 diagnosis+
  suggestions[](유효명 열거·최근접 이름=스키마에서 결정론 계산) → **recovery 10–20%→63–97%**
  [arXiv:2606.05037] — 우리 현행 bare 에러 = 측정상 최악 포맷 ③반복 탐지 = 연속 구조 레코드
  exact equality(산문 아님) ④산문 속 도구명(유저 지시문)은 constrained decoding 밖 →
  **구조 act(`instruct_user_run`) 승격 + 검증된 레코드에서 문구 렌더** = 선언-우선 그대로.
- **LLM**: 구조-카운트 N회 후 CRITIC sub-call(외부 증거 필수·순수 self-reflection 비신뢰)
  [arXiv:2305.11738]·AgentDebug [arXiv:2509.25370·snippet-only].
- **learn**: Fission-GRPO("repetitive invalid re-invocations" 명시 표적·error-recovery +5.7pp·
  TAU1 Retail 최대 +17.4pp 전이) [arXiv:2601.15625]·Reasoning Trap(**프롬프트 거의 무효·DPO는
  capability 저하 동반**) [arXiv:2510.22977].
- **경고**: constrained decoding의 constraint tax(정당 호출 억제) 보고 [arXiv:2606.25605] →
  Δspurious≤0 계측 필수([[19]]).

## §5. GIVE_RELEVANCE → 관련성 판단 (복구 완전판)

- **지형**: 필드 표준 = 관련성/필요성은 **LLM-측 판단**(MetaTool [arXiv:2310.03128]·API-Bank
  [arXiv:2304.08244]·BFCL irrelevance 카테고리) — 우리 (2)/(3) 분할이 필드 규범과 일치.
- **★Verifier Tax [arXiv:2603.19328·tau-bench 실측]**: 비준수-행동 **94% 차단해도 safe-success
  <5%** — 차단-후 회복이 21%→~0으로 붕괴. **회복 채널 없는 hard-block = 재앙** = 우리 모트
  규율(Δspurious·over-block 계측)과 C152("deny=soft·포기 유발")의 강력한 외부 실증. ⇒ 열린
  술어 위 deny 신설 금지 재확인·처방은 표면화/critique-regen만.
- **LLM**: Cleanlab TLM(tau2에서 trust-score+critique-조건 재생성·불필요 행동 pre-commit 포착·
  vendor-보고 실패율 −50%) — 우리 문제의 최근접 발표 사례. 단 **LLM-judge 단독 정밀도
  ~69–70%**(ToolEmu vs human [arXiv:2309.15817]) → **soft 신호로만**(threshold+regen·다중
  judge·저신뢰→ASK)·결정론 거부권 금지.
- **hybrid**: **GuardAgent [arXiv:2406.09187]·AGrail [arXiv:2502.11448] = LLM이 선언 정책을
  가드레일 코드로 formalize·엔진이 구조 레코드 위에서 결정론 실행** — [[16]] §3의 "정책→
  gate_spec 고정 컴파일러"와 아키텍처 동형(외부 실증)·단 닫힌 술어(접근제어형)에만.
- **learn**: SMART-ER(스텝별 필요성 rationale 학습·불필요 도구 사용 −60~67%·정확도 무손실)
  [arXiv:2502.11435]·step-level PRM(ToolPRMBench [arXiv:2601.12294]·2026 id 일부 미검증 주의)·
  AgentDoG(경량 가드 분류기 **~1k 궤적 샘플로 충분**) [arXiv:2601.18491·미검증 id 주의].
- **결정론이 닫는 부분집합**: eligibility 마스크(인자 미접지 도구는 give 대상서 구조적 제외)+
  give의 구조 act 승격(스키마-검증 이름)·근거-인용 접지. **관련성 본체는 열린 술어 확정.**

## §6. 공통 패턴 (전 주제 관통)

1. **단일 아키텍처 수렴**: typed act 인벤토리 + 스키마-유도 slot/eligibility ledger + LLM
   formalize sub-call(TOD 3분할)이 레버 1·2·3·5를 동시에 닫음. UNKNOWN_REPEAT만 별도(화이트
   리스트+오류 계약)이되 같은 스키마에서 파생. **레버 5개→인프라 1개**([[19]] 합성-우선 합치).
2. **3층 사다리 일관**: 결정론(구성적 예방) → LLM sub-call(구조-카운트 트리거 후 critique/
   formalize) → learn(preference·**SFT 단독은 과보수화 반복 보고**). 트리거는 언제나 구조
   이벤트 카운트·절대 산문.
3. **강제(force)는 일관되게 유해 문서화**: required=jailbreak+환각 유발·hard-block=회복 붕괴
   (Verifier Tax)·DPO refusal=capability 저하 — 필드 처방=마스킹/steering/regen. **등대 §1.3
   모트("부작용 없는 레버 없음")의 외부 실증.**
4. **산문 매칭 계보 부재** = 폐기 결정의 문헌적 지지.
5. **constrained decoding 한계**: 구조 채널 전용·membership만 보장·constraint tax — 산문 채널
   문제의 해법은 항상 "그 행위를 구조 채널로 승격".
6. **whitespace(우리 기여 후보·[[41]])**: producer-반환 vs 유저-발화 값 구분·유저-위임 실행
   노드의 agent-측 라우팅·grace-turn 정식화·verification-vs-redundancy 분리.

## §7. 인용 신뢰도

- act-timing 8/8·reask 9/9·invalid-name 10/10·arg-acquire 10/10 실재 확인(abstract 지지=
  완전/부분 구분은 각 절에 반영). relevance는 8개 우선 검증 실재·**2026 arXiv id 5개
  (ToolPRMBench·AgentDoG·TraceSafe 등)는 미fetch=미확정 취급**. Cleanlab=vendor-보고.
  exists=false = 0건.
- 종합 agent의 §3·§5 "미수신" 주석은 입력 절단 아티팩트였고 본 문서에서 원본 복구로 해소.

## §8. 우리 설계(`DECLARATION_FIRST`) 대조 — 확증 4·수정 2·추가 3

- **확증**: ①산문 매칭 폐기(계보 부재) ②§2 컨트롤러=AnyTOD/HCN/SGD 동형(30년 계보) ③선언
  스키마 화이트리스트>블랙리스트(ToolDec) ④[[16]] §3 정책-컴파일러=GuardAgent 동형.
- **수정 1 (§2 라우팅)**: "consent→ACT required"도 **무조건 강제 금지** — eligibility 성립 ∧
  ask-escape act 보존 시에만·기본은 마스킹/유예(When2Call·Verifier Tax 근거).
- **수정 2 (§1 give 근거-인용)**: 처방을 deny가 아니라 **표면화/critique-regen**으로 명시
  (Verifier Tax: hard-block 회복 붕괴).
- **추가 1 (무료·즉시)**: **오류 계약** — Unknown-tool/결핍-인자 에러를 diagnosis+suggestions
  구조 포맷으로(recovery 10–20%→63–97%·스키마에서 결정론 계산) → AXIS §4-3 결함 수정에 편입.
- **추가 2**: ask의 엔진-매개 typed 이벤트 승격 시 **per-argument ask-counter**가 닫힘 →
  stuck=ask k회∧binding 0 구조 술어(VALUE_ACQUIRE 대체 완성).
- **추가 3 (learn 데이터 스펙 정본 후보)**: When2Call RPO(공개)·SMART-ER rationale·
  Fission-GRPO recovery·Ask-before-Plan trajectory·AgentDoG ~1k — E6′/learn 날개에 등재.
