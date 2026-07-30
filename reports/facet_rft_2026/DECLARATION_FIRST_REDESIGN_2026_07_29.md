# 선언-우선(Declaration-First) 재설계 — 전면 formalize 아키텍처 + NL-패턴매칭 전면 폐기 (2026-07-29·rev1)

> 발단=사용자 지시 3건: ①"5개를 LLM이 formalize한 정형식으로 표현하고, 정형식이 맞는지만
> 결정론으로 체크" ②"턴 결정도 DAG에서 결정론으로" ③**"모든 걸 LLM formalize로 규격화·
> 결정론=형식 검증기+실행기로만·패턴 매칭 전면 폐기·formalize 유도가 학습만인지/프롬프트
> 가능인지/다른 방법 있는지 확인"(=rev1·전면화 확정)**.
> 상위: `AXIS_DECISION`(코어 동결·금지선)·`STACK_PREDICATE_AUDIT`(배터리)·[[16]] LOCK(fexec·
> FIND=근거-요구)·C45(출처선언·날조 67→0%·Δspurious 0)·[[10]](LLM=formalize·검증=결정론).
> 상태: **rev1 설계** — 리뷰 → 구현(스위치-오버 순서 §1c). 딥리서치(run `wf_df853bf1-7a4`) 대조分 추후 편입.

## §0b. 전면화 (rev1 LOCK 후보) — 아키텍처 완성형

**LLM 출력 3분**: ①선언(봉투: turn_type·ask_slot·done_report·출처 4지선다·evidence_quote)
②행동(tool_calls) ③산문(유저용·**엔진 불투명·해석 0**).
**엔진 3역**: ①형식 검증기(스키마·봉투) ②정합 검증기(선언 ⊨ 구조 이벤트·원장) ③실행기/라우터
(calc·fexec·byref·DAG policy).
**엔진의 산문-접촉 허용 3종(전부 형식 검증·비교 한쪽이 형식 객체/상수)**: (a) 정본-상수 앵커
(G1 notice 정규화·A4 토큰) (b) 선언값↔원장 접지(substring·검증 대상=선언) (c) 자기-삽입 스텁
계수. **이 3종 외 자유-NL 해석(P2-b) = 전면 폐기.**

## §0c. ★역할-분담 LOCK + formalize 스키마의 거처 = A2 (2026-07-30 사용자 지시)

> "LLM은 formalize하고, 엔진은 formalize된 걸 검증·실행만 하게 하라. formalize의 기본
> 스키마와 가이드는 A2에 기술하라."

**LOCK**: ①**LLM = formalize만** — 선언(봉투)·행동(tool_calls)·산문(유저용) 생성. 판정·선택·
집행 없음. ②**엔진 = 검증+실행만** — (a) 형식 검증기: A2 `formalize_spec`에서 **guided-decoding
문법을 컴파일**해 선언 필드의 존재·타입·enum을 디코딩에서 하드 보장(+ENVELOPE regen 백스톱)
(b) 정합 검증기: 선언 ⊨ 구조 이벤트·원장 (c) 실행기/라우터: calc·fexec·byref·turn-policy.
엔진의 다른 행동 일절 금지(산문 해석 0·§1b 폐기 목록 집행).

**정정 (구 §5-1)**: "enum·스키마=엔진 상수" → **A2 base-layer 데이터**로 이관. 엔진은 스키마의
순수 해석기가 되고, guided 문법도 A2에서 파생-컴파일된다 = [[16]] §3 "고정 컴파일러" 패턴·
특허 프레임("도메인적응=명세 공급·엔진 무수정")과 완전 정합. **단 base layer는 도메인-불변**
(모든 도메인이 같은 봉투를 씀) — 도메인 A2는 내용(producer 맵·레지스트리·slot·claim_kinds)만
공급하고 **enum 확장은 금지**(확장=온톨로지 절차·[[16]] §3 검증 필요).

## §1d. A2 `formalize_spec` 정본 초안 (base layer = 도메인-불변 데이터·엔진 동봉)

```yaml
formalize_spec:                    # ── base layer: 도메인-불변 (전 도메인 동일·수정=온톨로지 절차)
  version: 1
  envelope:                        # 매 assistant 턴 선언
    turn_type:   {enum: [ACT, ASK, CONFIRM, INFORM, DONE], required: true}
    next_action: {type: tool_ref, required_if: turn_type==ACT}
    ask:         {required_if: turn_type==ASK,
                  fields: {slot: slot_ref, reason: {enum: [missing, confirm]}}}
    done_report: {type: list[{kind: enum_from(domain.claim_kinds), what: str}],
                  required_if: terminal_turn}          # CLAIMPROV/WRITEPROV 서브콜 통합(§1b 정정)
    instruct_user_run: {type: tool_ref(domain.user_runnable), nullable: true}
    evidence_quote:   {required_for: [give, FIND-값], verify: substring_in_ledger}
  user_act:                        # 유저 발화 formalize (컨트롤러 입력)
    enum: [question, provides_slot, consent, refusal, smalltalk, mixed]
    slots_extracted: list[{slot: slot_ref, value: str, quote: str}]   # verify: quote∈발화
  guide: |                         # 시스템-프롬프트 주입 가이드(데이터·C47: placeholder-명백 예시만)
    <formalize 방법 지시문 — 엔진 불변·A2 텍스트>
  verification:                    # 엔진 정합 규칙 (전부 닫힌 술어·이 목록이 검증기의 전부)
    - ACT ⇒ tool_calls ≠ ∅
    - ask(slot, missing) ⇒ slot ∉ filled_ledger
    - done_report ⊆ executed_events(domain.event_map)
    - instruct_user_run ∈ domain.user_runnable
    - evidence_quote ∈ ledger / slots_extracted.quote ∈ utterance
  routing:                         # 결정론 turn-policy (도메인-불변 default)
    provides_slot: ledger 갱신 → eligible(read)=required 허용
    consent_token: write-게이트 해제 (eligibility ∧ ask-escape 보존 시만 required·DR §8 수정1)
    question|refusal: 강제 없음 (pending 표면화 유지)
    stuck: ask(slot) k회 ∧ binding 0 → producer-경로 표면화
domain:                            # ── 도메인 layer: 내용만 공급 (enum 확장 불가)
  slot_types: <스키마 파생>         # 거의 무료([[16]] §3)
  arg_producers: {<arg>: {tool: <t>, side: agent|user}}
  user_runnable: [<도구 레지스트리·환경 파생>]
  claim_kinds / event_map: <기존 claim_prov 재사용>
```

- **형식 보장 사슬**: `formalize_spec` → guided 문법 컴파일(결정론) → 디코딩 강제 → 파싱
  실패 시 ENVELOPE regen. 스키마가 데이터라 문법도 자동 파생 — 엔진 수정 없이 스키마 진화.
- **검증기의 완결성**: `verification` 목록 밖의 검사를 엔진이 하면 위반(§1b 집행 장치).

## §1b. P2-b 전면 폐기 목록 (rev1b·2026-07-30 **regex 전수 스윕으로 확정** — 사용자 질의 "5개 외 더 있나")

스윕 방법: 전 패치 파일(t2_gate/prekb/scaffold_get/eplan/resolve) regex·substring 사이트 전수
색출 → 매칭 대상이 {자유 NL} vs {기계-포맷/선언/스키마}인지 분류.

**폐기 확정 (자유-NL 해석)**:
| 폐기 | 현행(산문 해석) | 대체 |
|---|---|---|
| FORCE_ACTION | say-don't-do 감지 | turn_type 선언↔tool_calls 모순(§1) |
| HAVE_VALUE·VALUE_ACQUIRE | 재요청 신호 문구 | ask 선언↔slot 표(§1) |
| UNKNOWN_REPEAT | 재지시-문맥 부분문자열 | instruct_user_run 선언↔레지스트리(§1) |
| GIVE_RELEVANCE | 원장-이름 프록시 | give+evidence_quote↔원장(§1) |
| **RESOLVE instruct·action-required** | 산문 조건 트리거(발화 384=열린-측 최대) | §2 turn-policy 라우팅에 흡수 |
| **★EPLAN 수요-측 추출 전반**(스윕 확대) | intent_chains 신호(C191)만이 아니라 **유저 발화 직접 파싱 일체**: `_SCOPE_RE`(ALL/EVERY)·`_QTY_*`(수량어/범위/시간)·`_ENUM_*`(열거) — ledger 수요측이 통째로 NL-파싱 | user_act/slot formalize(근거-인용付)가 요구 목록 산출→엔진은 그 선언 위에서 walk/coverage |
| **★NLNUM `_MONEY_RE`**(스윕 신규 발견) | 어시스턴트 산문에서 금액 스캔→원장 대조→강제 regen | 산문 불투명 원칙상 폐기 — 수치 주장은 done_report/calc-재진술 선언으로·잔여=learn |
| A2 신호-문구류·PREKB 검색어 공급 | 산문-매칭 설정·스푼피드 | 삭제·사실-통지만 |
| (경계) FAB_STRIP `_PROCEDURAL_RE` | give-인자 내 "절차문" 성격 분류 regex(선언 필드 위이나 판정=의미-성격) | instruct_user_run 구조 act 승격 시 자연 소멸 |

**★정정 (폐기 목록에서 제외 — 코드 정독으로 판명)**: **CLAIMPROV·WRITEPROV의 탐지는 이미
선언-우선이다** — 산문 regex가 아니라 **LLM formalize 서브콜**(claims `{kind,what}` JSON 선언
→ A2 event_map 집합 교차·주석 명시 "텍스트 파싱 0"·창=사임∨transfer 호출=구조). 결함은
술어가 아니라 ①처방(강제 regen=3b 표면화 강등 기확정) ②별도 서브콜 비용 — **봉투
done_report로 통합(개선)이지 폐기가 아님.** 초판 §1b의 "산문 claim 스캔" 표기는 오류였다.

**허용 확인 (폐기 아님 — 오해 방지)**: 기계-포맷 파싱(도구 출력 field:value·env 에러
"Unknown tool"·coverage/unverified 블록·vLLM CWE) · 스키마/A2/KB-문서 파싱(`_SUCH_AS_RE`
예시 추출·doc header) · formalize 서브콜 출력 JSON 파싱(REF_ISO·RESOLVE fexec) · 도구명/인자
형식 검사(suffix·prefix·placeholder 패턴·byref). 전부 비교 한쪽이 형식 객체/기계 포맷.

**스윕 종결 확정 (2026-07-30 2차: `.lower()`/in 서브스트링 전수 + 미분류 레버 해소)**:
- **GUIDED = guided-decoding 문법 주입**(`t2_guided_patch.py`·vLLM grammar) → **코어 확정**
  (형식-층·구조 디코딩 = ToolDec-계 화이트리스트·DR §4 권고와 일치·감사표 "미분류 1건" 해소).
- **UNAVAIL_PROMISE** = CLAIMPROV `pending` 선언 리스트 + known-tool **membership** → 이미
  선언-기반·유지(P2-b 아님).
- `_ctx_has`·`_mentioned`(WRITE_ARG_GROUND 유저-언급 검사) = 선언값↔발화 접지(P2-a·허용).
  단 `_mentioned`의 토큰-중첩 완화(min_tok)는 false-block 실측(rall22 031 "Marriott") 이력 —
  **정규화 정비 대상**(G1형·폐기 아님).
- ⇒ **P2-b 폐기 목록은 위 표로 종결**(regex+서브스트링 전수·잔여 미분류 0).

## §1c. 스위치-오버 순서 (폐기 실행 규율)

**선언 채널 배선 → 구 트리거 제거** — 배선 전 제거는 표면화까지 죽인다. ①봉투 확장(turn_type·
ask·done_report) + 검증기 ②E-DECL-COMP 프로브(§6)로 준수율 확인 ③구 P2-b 트리거 OFF(플래그
단위·원장 기록) ④E-MFIX 후 라이브 검증. 설계서 리뷰 승인 전 구현 착수 금지([[03]]).

## §0. 원리 — 정보 흐름의 역전

금지선 5종의 공통 오류 = **엔진이 산문에서 모델의 의도를 재구성(추론)**하려 함 → 열린 프록시
(P2-b). 올바른 방향 = **모델이 의도를 닫힌 채널에 선언(formalize)**하고 엔진은 정합만 검증:

- 열린 술어 "모델이 X하려 한다" → 닫힌 술어 "**모델의 선언 D와 구조 이벤트가 모순된다**".
- 검증 대상이 산문→선언으로 바뀌면 P2-(b) 프록시가 P2-(a) 자기-생성 구조 이벤트가 된다.
- 의미 부담은 LLM 관할로 이동(정당 배치·[[10]])·잔여 = **선언 충실도** 하나(learn 표적).
- **기증명 선례**: C45 출처선언(operand 층에서 정확히 이 변환·선언 강제 자체가 행동을 바꿈=
  자기-정합 압력 실측) · [[16]] §4d FIND=근거-요구(evidence_quote ∈ source 결정론 검증) ·
  ENVELOPE_GUARD(선언 형식-층 인프라 기존재).

## §1. 5종 재설계 표

| 구 레버(금지) | 선언(LLM formalize) | 결정론 체크(엔진·전부 닫힘) | 잔여(LLM/learn) |
|---|---|---|---|
| FORCE_ACTION | 매 턴 봉투 `turn_type: ACT\|ASK\|CONFIRM\|INFORM\|DONE` (+`next_action: tool` or null) | **ACT ∧ tool_calls=∅ = 자기-선언 모순** → 형식-층 regen. `next_action` 선언 후 K턴 미이행 = 선언-이벤트 불일치 | 선언 충실도 |
| HAVE_VALUE | ask 행위를 `{ask_slot: X, reason: missing\|confirm}`으로 선언 | `reason=missing ∧ X 원장 접지 실재 = 모순` → 표면화("X는 이미 <ref>"). 확인/재요청 구분=모델 자기-선언 | 선언 충실도 |
| VALUE_ACQUIRE | 동일 ask 선언 + A2 `arg_producers` | `ask_slot=X ∧ A2 user-측 producer P ∧ P 미-give` → P 경로 표면화 (membership) | — |
| UNKNOWN_REPEAT | 유저-실행 지시를 `{instruct_user_run: T}`로 선언 | `T ∉ 실행가능 레지스트리` = membership 위반. **선언 시점**에 차단(반려-후-재지시 산문 매칭 불필요) | — |
| GIVE_RELEVANCE | `{give: P, evidence_quote: "..."}` 근거 인용 의무 | `quote ∈ 원장` substring(=§4d FIND-근거 적용). 무근거-give 소멸 | 근거의 올바름=learn |

- 다섯 모두 **새 엔진 메커니즘이 아니라 기존 코어의 확장**: 봉투(형식-층)·원장 membership·
  substring 접지 — AXIS 코어 1·3·5층 어휘로 정의됨 = Q3 어휘 테스트 통과.
- A2 신설분: turn_type enum(도메인-불변=엔진 상수)·ask 선언 스키마(불변)·`arg_producers`(기존).
  **도메인 리터럴 순증 0.**

## §2. 턴-결정의 결정론화 (사용자 질의 ② — "잔여 열림" 정정)

초안 평가("항상 required 불가능·act/talk 판정은 열림")는 **과대였다**. 두 사실이 닫는다:

1. **act vs talk는 배타가 아니다**: tau2 assistant 턴은 `content+tool_calls` 공존 가능 →
   eligible-시-required를 걸어도 같은 턴에 답변+호출 가능. "말할 차례" 문제 대부분 소멸.
   조기-write 위험만 남고 그것은 동의-토큰 게이트(A4·닫힘)가 담당.
2. **유저-발화 의존분도 formalize로 축소**: 턴 결정의 유일한 열린 입력=직전 유저 NL →
   LLM이 닫힌 enum으로 분류 `user_act: question|provides_slot|consent|refusal|smalltalk` →
   컨트롤러가 (user_act × DAG 상태 × slot 표 × 게이트 상태)에서 **결정론 라우팅**:
   - provides_slot → 표 갱신 → eligible이면 ACT(required)
   - consent(토큰) → ACT(required·write 허용)
   - 선행 read 미충족 → read required(§1.5 자유)
   - question/refusal → 응답 허용·강제 없음(단 pending 표면화 유지)
   = 고전 task-oriented dialogue의 **NLU → DST → policy** 3분할과 동형·우리 분담([[10]])과 일치.

⇒ **턴 결정은 "유저-발화 formalize 정확도"라는 좁은 잔여를 빼면 결정론으로 닫힌다.** 그
잔여는 엔진이 추측하는 것(금지)이 아니라 LLM 정당 관할·learn 표적. TRACK-A(per-step DAG
컨트롤러)의 구체안이 이것이다.

## §3. 선언-우선 아키텍처 4층 (전체 조감)

| 층 | 선언 | 상태 |
|---|---|---|
| operand | 출처 4지선다 {GET/FIND/INFER/ASK} | **기구현·실증**(C45) |
| ask | `{ask_slot, reason}` | 본 설계 |
| turn | `{turn_type, next_action}` 봉투 | 본 설계 |
| user-발화 | `user_act` enum formalize | 본 설계(컨트롤러 입력) |

## §4. 정직한 한계·리스크

1. **선언 충실도**: 모델이 CONFIRM이라 선언하고 실제로 신규 정보를 물을 수 있다 — 선언의
   참됨 자체는 의미. 단 (a) 선언↔구조 이벤트 모순은 잡히고 (b) 틀린 선언도 learn 데이터
   라벨이 되며 (c) C45 실측상 선언 강제만으로 행동이 교정되는 성분이 크다. 잔여=learn.
2. **형식 부담**: 소형 모델의 봉투 준수 비용 — ENVELOPE_GUARD가 백스톱·C45에서 32B 준수 실증.
   단 선언 필드 증가 = 파싱 실패 표면 증가 → 봉투 스키마는 최소로.
3. **게임 가능성**: required 회피를 위해 ASK만 선언하는 퇴행 — pending-표면화 + K턴 유예 후
   승급(TERM_GRANT 패턴)으로 유계. 계측 필수(Δspurious·모트 규율).
4. **검증 대기**: 본 설계는 무료 오프라인 검증(격리 프로브: 선언 준수율·모순 검출율) 가능하나
   효과 판정은 **E-MFIX(측정 고정) 후**. 딥리서치 결과로 선행 실증(TOD policy·declaration
   패턴) 대조 후 rev1.

## §6. formalize 유도 방법 — 3층 분해 (사용자 질의 ③ 확정·전부 기측정 인용)

| 층 | 문제 | 방법 | 증거 |
|---|---|---|---|
| ①형식 준수 | 봉투가 스키마대로 나오나 | **guided decoding=제3의 방법·하드 보장**(vLLM grammar/json·선언 필드 존재/타입/enum 디코딩 강제·학습 불요·결정론) + 프롬프트 + ENVELOPE regen 백스톱 | C154(우리-서빙=벤치 불변 정당성). 형식 준수는 [[42]] ceiling 대상 아님(ceiling=행동 규칙) |
| ②선언 존재·정합 | 선언을 내고 검증 통과까지 | **프롬프트+검증기+재발화로 충분 — 학습 0 실증** | **C45**(32B·날조 67→0%·over-block 0/2650·Δspurious 0·전부 프롬프트-레벨)·C42(7B 격리 4지선다 완벽)·C41(짧은 문맥 규칙 작동 0.20→0.87) |
| ③선언 충실도 | 선언=실제 의도·사실 | **여기만 learn**(SFT 설치+DPO 거짓-선언 벌점·[[42]] 처방 일치). 단 C45의 자기-정합 효과가 상당분 무료 | C30(행동 규칙 프롬프트 무효)·fexec 0.79·INFER-calibration 미확립([[16]]) |

- few-shot 형식 예시는 **C47 주의**(예시 복사 오염) — 값이 명백 placeholder인 스키마만.
- **E-DECL-COMP (무료 격리 프로브·구현 전 관문)**: 신규 선언(turn_type·ask) 3-arm =
  prompt-only / prompt+guided / prompt+regen. 측정 = 봉투 준수율·모순-검출율(의도적 모순 주입
  재생)·ASK-남발율(게임 감시). 32B·로컬 vLLM·[[18]] 정보-맞춘 구성. 통과 기준 = guided arm
  준수 100%(구조 보장 확인)·regen arm ≥ C45 수준.
- 외부 선행연구 대조 = DR1(`wf_df853bf1-7a4`)·DR2(`wf_781f7000-938`) 편입 완료.
- **★DR2 구현 확정분(정본=`DR_DECLFIRST_DR2` §6)**: ①혼합 생성=**tail-제약 단일-pass**
  (In-Writing식 eos-트리거·기술 확정) vs two-pass(2606.25605)를 E-DECL-COMP arm으로 비교
  ②schema-in-prompt 단독은 비현실(8B 90→13% 붕괴)=guided 필수 재확인 ③**post-parse 결정론
  재검증 상설**(제약 엔진 신뢰 금지·2605.26128) ④dense 모델만(MoE서 logit-mask 붕괴·우리
  32B=dense) ⑤봉투 스키마=채택 엔진 지원 subset 내(JSONSchemaBench ~2배 편차) ⑥Δprose
  계측(제약 봉투의 산문 채널 품질 영향=문헌 공백=우리 실험) ⑦오류-계약은 **누출-등급
  에스컬레이션**으로(SEAL 허용/금지 + ITS 사다리·구조 카운터·admissible-set=닫힌·인스턴스-
  독립일 때만 = PREKB 스푼피드의 일반 이론) ⑧다중 시드 측정·재생성 상한+ASK fallback.
- **★설계 정당화의 외부 실증(DR2 §3)**: typed 선언→행동 불일치 0.7–1.4%(free-text 22–26%)
  [2606.00476] · false success=tau2 실패 45–48%·**LLM judge AUROC≤0.65 vs 구조 신호 0.908**
  [2606.09863·Trajel] = "왜 결정론 원장 대조인가"의 motivation 정본.

## §7. 선행연구 대조 (딥리서치 편입·2026-07-30·정본=`DR_BANLIST5_PRIOR_WORK_2026_07_30.md`)

- **확증 4**: ①산문 매칭 계보 부재=폐기 지지 ②§2 컨트롤러=TOD 3분할(AnyTOD·HCN·SGD) 30년
  계보와 동형 ③화이트리스트>블랙리스트(ToolDec: 환각 도구명 0) ④[[16]] §3 정책-컴파일러=
  GuardAgent 동형(외부 실증).
- **★수정 1 (§2 라우팅)**: "consent→ACT required"도 무조건 강제 금지 — required는 jailbreak+
  파라미터 환각 유발 문서화(When2Call)·**eligibility 성립 ∧ ask-escape act 보존 시에만**·
  기본은 마스킹/K-턴 유예.
- **★수정 2 (§1 give)**: evidence_quote 검증의 처방=deny 금지·**표면화/critique-regen만** —
  Verifier Tax(2603.19328·tau-bench): 94% 차단해도 safe-success <5%(차단-후 회복 21%→0) =
  모트 규율의 외부 실증.
- **추가 1 (무료·즉시·AXIS §4-3 편입)**: **오류 계약** — bare "Unknown tool"/결핍-인자 에러를
  diagnosis+suggestions[] 구조 포맷으로(유효명·최근접=스키마에서 결정론 계산) → recovery
  10–20%→63–97%(2606.05037). 현행 bare 문자열=측정상 최악.
- **추가 2**: ask가 typed 이벤트로 승격되면 **per-argument ask-counter**(SAGE Σn_a)가 닫힌
  술어 → stuck = 같은 인자 ask k회 ∧ 신규 binding 0(§1 VALUE_ACQUIRE 행 완성).
- **learn 데이터 정본 후보**: When2Call RPO(공개·SFT는 과보수화)·SMART-ER(필요성 rationale·
  불필요 도구 −60~67%)·Fission-GRPO(recovery·TAU1 +17.4pp 전이)·Ask-before-Plan trajectory.
- **whitespace(기여 후보·[[41]])**: producer-반환 vs 유저-발화 값 구분·유저-위임(give) 실행
  노드의 agent-측 라우팅·grace-turn 정식화.

## §5. [[05]] 3질문 ([[17]])

1. **순증?** 엔진 메커니즘 순증 0 — 기존 코어(봉투·membership·substring)의 어휘 확장.
   금지선 5종+P2-b 폐기로 순감. **스키마·가이드=A2 base-layer 데이터**(§0c 정정)·엔진=순수
   해석기·guided 문법도 A2-파생 컴파일.
2. **경계?** 강화 — 산문 해석 전면 제거·도메인 차이는 A2 domain-layer(producer 맵·레지스트리·
   slot·claim_kinds)로만·base-layer enum 확장 금지(온톨로지 절차).
3. **수행 대체?** 없음 — 선언·행동 전부 LLM이 emit·엔진은 §1d `verification` 목록의 정합
   검증과 `routing`의 결정론 라우팅만(목록 밖 검사=위반).
