# E-PLAN 지속 구동 컨트롤러 설계 (2026-07-24 · 피벗)

> 상위 = `RESEARCH_MASTER.md` §4 TRACK-A · [[14]] E-PLAN 우선 · [[17]] [[05]]-섹션 의무.
> 계기 = 사용자 지적(2026-07-24): 031/043가 rall19~23 매 런 **재판** — 포인트 게이트는 F2/F3(참조)
> 축을 잡지만 pass를 막는 지배 잔여는 **F4 coverage·say-don't-do**이고 이건 게이트가 아니라 컨트롤러의 몫.

## 0. 한 줄
현재 E-PLAN은 **종료 시점 1회 리마인더**다. 탐지는 되나(`walk gap: qty=9 executed=1` 실측·rall23b)
**1회 보류 후 포기**한다(`_t2_eplan_walked` 하드캡). 이를 **plan 충족까지 지속 구동**(budget K·progress-guard)
으로 승격한다. 컨트롤러=결정론([[10]])·plan 추출=LLM·**write 날조 0**(에이전트가 emit·controller는 조기종료
차단+처방 재프롬프트만).

## 1. 현 상태 진단 (코드·실측)
- 배선점 = `_check_termination` wrap (`t2_eplan_patch.py:744`). `user_stop` 시 ledger 재구축→coverage_gap
  탐지→`self.done=False`+리마인더 1회 주입→**즉시 `_t2_eplan_walked=True`로 재발화 봉쇄**.
- 실측(rall23b 043): `walk gap: qty=9 executed=1`(9 필요·1 완료 앎) → 보류 1회 → 그래도 미완 → user_stop.
  close_credit_card_account 등 **결정된·동의된 write가 미실행**(say-don't-do)인데 컨트롤러가 놓아줌.
- FORCE_ACTION(say-don't-do→required)·RESOLVE도 발화하나 **단발·매루프 리셋**이라 사슬 지속 구동 못 함.

## 2. 변경 (최소 diff·기존 자산 재사용)
`_check_termination` wrap 을 다음으로 교체 (ledger/coverage_gap/cp5_reminder 로직 전부 재사용):

1. **budget K** (`T2_EPLAN_DRIVE_K` 기본 4·상한): `_t2_eplan_walked`(bool) → `_t2_eplan_drives`(int).
   drives<K 이고 gap 있으면 보류·구동.
2. **progress-guard** (무한루프/붕괴 차단 = §7.3 딜레마 회피): 직전 보류 후 **executed write 집합이
   커졌는가**(진전)만 계속 구동. 진전 0이면 즉시 release(놓아줌) — 못 고치는 걸 계속 잡지 않는다.
   `_t2_eplan_last_exec` 스냅샷 비교.
3. **directive 리마인더** (C116 처방적 구체성): 기존 generic "gap 있음" → **plan서 미실행 필수-write를
   지목**("The plan requires `{tool}` for `{entity}`, which you have not executed. Do it now."). 문구·도구=
   spec(A2)·엔진은 ledger diff만.
4. **stop-이유 보존**: transfer/quota 등 user_stop 아닌 종료는 불개입(기존).

## 3. [[05]] 3질문 (의무·[[17]])
- **(1) 도메인-특화 순증?** 아니오. 구동 로직=ledger diff(결정론)·plan/write-tool/문구=spec(A2/도메인일반).
  엔진 도메인 리터럴 0. K·progress-guard=일반 제어.
- **(2) 유동 판단을 결정론에 동결?** 아니오. **선택·실행은 여전히 에이전트**(controller는 조기종료 차단+
  어느 필수-write가 남았는지 *사실* 통지). plan 추출=LLM. coverage 판정="필수 N개 중 실행 M개"=결정론 사실.
- **(3) 모델 대신 도메인 행동 수행?** ★핵심 경계. **아니오** — controller는 write를 **날조/실행하지 않는다**.
  에이전트가 tool_call을 emit하고, controller는 (a)조기 user_stop을 보류 (b)"이 write가 계획에 있는데
  미실행" 통지만. §1.5 "write 강제 금지"=controller가 write를 *만드는* 것 금지 → 준수(에이전트가 만듦).
  say-don't-do 완주 유도=FORCE_ACTION 선례와 동형(이미 sanctioned). progress-guard가 over-drive(강제 재호출
  =over-action) 역효과를 계측·차단(Δ진전=0→release).

## 4. 검증 (무료 우선·[[09]])
- (a) 단위: `test_eplan.py` 확장 — K회 구동·progress-guard release·directive 문구·no-gap 종결.
- (b) 오프라인 replay: rall23 043/054 궤적서 "walk가 지속됐다면 close/필수-write 지목이 몇 턴에
   나왔나" 시뮬(결정론 ledger walk·라이브 재현 아닌 술어 검증).
- (c) 표적 nt=1 라이브(유료·승인 후): 043(close 사슬)·052(submit→deny 사슬)·039(8-dispute) — pass 아닌
   **완주율**(필수-write 실행 수/gold) 대조.

## 5. 함정 (선등록)
- progress-guard 없으면 = §7.3 deny/pass-through 딜레마 재현(못 고치는 사슬 무한 보류→붕괴). **필수**.
- ledger 재구축이 매 종료마다 전체 메시지 파싱 = O(n)·종료당 1회라 무해(스텝당 아님).
- K 너무 크면 user-sim이 이미 나간 뒤 공회전 → K=4·progress-guard로 자연 종료.
- **write-강제 경계 재확인**: close는 write지만 손님 동의([70] "please close it")+에이전트 결정([71]
  "I will proceed") 상태의 **미실행**을 완주시키는 것 = say-don't-do(FORCE 선례). 손님 미동의 write는
  plan에 없거나 precondition 미충족→구동 안 함.
