# E-PLAN compact-plan 재환기 설계 (2026-07-24)

> 상위 = `RESEARCH_MASTER.md` §3 원장 C142/C143 · [[14]] E-PLAN 우선 · [[17]] [[05]]-섹션 의무 · [[10]] plan=LLM/controller=결정론.
> 계기 = 사용자 설계(2026-07-24): "전체 계획을 먼저 세우고, 실행 중 중요 분기점에서 계획 대비 진척(N중 M완료·남은=X)을 환기."
> C143 실증: raw-doc 재투입 subcall은 35K 희석에 실패(0/5)·compact plan 목록은 본질적 focused → 이 설계가 구조적 우월.

## 0. 한 줄
현 E-PLAN 두 결함(①plan을 종료시점 오염맥락서 재도출 ②리마인더가 종료시점만)을 교정한다:
**①실행 초반(오염 前) 깨끗한 서브콜로 compact plan(필수 도구 목록) 1회 포착 → ②실행 중 분기점에서
그 compact plan 대비 진척(완료/남음)을 결정론으로 대조해 남은 필수단계를 재환기.** plan=LLM(포착)·
추적/판정/트리거=결정론([[10]])·write 날조 0(환기만·에이전트가 emit).

## 1. 왜 (측정 근거)
- **C142**: 모델은 깨끗·focused 맥락서 완전 plan(apply_flag 포함) 냄(E0 4/4)·라이브는 reactive-execution
  으로 Step 건너뜀(자기-열거 안 함).
- **C143**: "오염이력 제거"만으론 부족(35K 정책문서 clean-ctx도 0/5)·**focus가 진짜 변수**·compact plan
  목록=희석 회피. raw-doc subcall 처방 기각.
- **C136**: 종료시점·막연 리마인더는 실패(apply_flag 0) → 분기점·구체 진척이 필요(미검증 가설).

## 2. 아키텍처 (3 컴포넌트)
### 2a. PLAN-CAPTURE (LLM·1회·오염 前)
- **시점**: 신원확인 완료 & 첫 정책-관련 read 직후(=요청+정책이 문맥 진입·에이전트 write/에러 오염 前).
  트리거=결정론(신원확인 write 관측 ∧ 첫 KB/정책 read 관측 ∧ 아직 미포착).
- **입력(focused)**: 손님 요청(user 발화) + **관련 절차 문서만**(그 시점 retrieved 정책 tool출력 중
  요청-매칭·C143: 전부 넣으면 희석→요청 키워드로 상위 K개만). *에이전트 자기 tool_call/에러 제외.*
- **출력**: compact 의무 목록 JSON `[{action, id?, note}]`(OBLIGATION_PROMPT 재사용·id 날조 금지).
- **저장**: `agent._t2_plan`(불변 앵커). 실패/빈출력=미포착(폴백=기존 qty/chain 신호).

### 2b. PROGRESS-TRACK (결정론·매턴)
- ledger(기존 `build_ledger_from_messages`)로 실행된 write/read 관측 → plan 대비 diff:
  `done = plan ∩ executed`·`remaining = plan∖executed`. 순수 집합차(도메인 리터럴 0).

### 2c. BRANCH-REGROUND (결정론 트리거 + 환기)
- **분기점 정의**(재환기 트리거·결정론): (i)**종결-신호 직전**(close/finalize/transfer/"끝"류 도구 or
  종결 산문 검출 시) ∧ remaining≠∅ — *가장 중요*(사용자의 "closure 확정 前") (ii)종료(user_stop·기존
  walk 위치)·백스톱. per-분기점 1회·sim당 cap(K).
- **환기 문구**(C116 처방·compact): "계획된 N단계 중 M완료. 남은: {remaining 이름 나열}. 종결 前 완수."
  (raw 문서 재투입 X·compact plan만=focused). write 강제 0(에이전트 emit)·progress-guard(직전 환기 후
  진전0→release·§7.3 딜레마 회피).

## 3. 기존 자산과의 관계 (재사용·delta)
- **재사용**: `_cp5_replan_subcall`/OBLIGATION_PROMPT(→PLAN-CAPTURE·단 입력을 full-transcript→focused로
  교체가 핵심 delta)·`build_ledger_from_messages`(PROGRESS)·`drive_decision`+progress-guard(C131)·
  `chain_gap` 집합차 로직.
- **delta vs 현 E-PLAN**: ①plan 포착 시점=종료→**초반**(오염 前) ②포착 입력=full-transcript→**focused
  절차** ③환기 위치=종료만→**분기점(종결 직전)** ④환기 내용=qty/chain gap→**compact plan 대비 진척**.
- **intent_chains(C134)**: 정적 체크리스트=폴백(도메인 focused 절차 자동선택 실패 시)·주경로는 LLM 포착.

## 4. [[05]] 3질문 (의무·[[17]])
- **(1) 도메인-특화 순증?** 아니오. PLAN=LLM 산출(도메인지식 A2 아님)·트리거/진척=결정론 집합차·문구=일반.
  focused 절차 선택=요청 키워드 매칭(엔진 일반·도메인 리터럴 0). intent_chains 폴백만 A2(기존).
- **(2) 유동 판단 동결?** 아니오. plan은 **LLM이 매 태스크 생성**(정적 동결 아님)·controller는 추적/환기만.
  "무엇이 필수인가"=모델 판단·"무엇이 실행됐나"=결정론 사실.
- **(3) 모델 대신 행동 수행?** 아니오. write 날조/실행 0. 조기종료 보류 + "남은 단계" 통지만·에이전트가 emit.
  §1.5 write-강제 금지 준수(FORCE_ACTION 선례 동형). progress-guard가 over-drive 역효과 계측·차단.

## 5. 검증 순서 (무료 先·[[09]])
- (a) **오프라인 PLAN-CAPTURE**: 043 실궤적서 focused-입력(요청+doc_003 상위매칭)으로 서브콜→apply_flag
  포함 plan 나오나(C142 E0 재현·C143 35K 실패와 대조). **make-or-break**.
- (b) 단위: 분기점 트리거(종결신호 검출)·progress diff·환기 문구·progress-guard release selftest.
- (c) 라이브 nt1(유료·승인): 043 chain met↑·분기점 환기 발화·apply_flag 실행 전환 여부([[08]] per-step).
- **정직**: C136이 종료·막연 넛지 실패 증거 → 분기점·구체 환기의 실효는 (c)서만 확정.

## 6. 함정 (선등록)
- PLAN-CAPTURE 입력이 focused여야(C143)·요청-매칭 상위 K 선택이 관련 절차 놓치면 plan 불완전 → K·매칭
  튜닝은 (a)서 계측.
- 분기점 종결-신호 검출 과다=over-reground(over-action류)·과소=miss → progress-guard+cap 방어.
- plan 포착 시점 너무 이르면(정책 read 前) 절차 부재·너무 늦으면 오염 → 트리거 조건(신원+첫정책read) 계측.
- compact plan의 id/이름이 discoverable suffix와 불일치 가능 → 환기는 이름만(unlock은 에이전트).
