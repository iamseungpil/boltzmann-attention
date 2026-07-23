# E-PLAN "중간 견인"(mid-conversation coverage drive) 설계 — 2026-07-23

> 표적 = chain-reach(F4 coverage) 잔여. 스모크(task_039·§2bw 후속) 실측: E-PLAN walk가 **진짜 8-dispute
> 갭을 정확히 감지하나 user_stop서 보류(1회)** → disputes 1/11. **세기는 작동·몰기(drive)가 실패.**
> 정본 코드=`t2_eplan_patch.py` `apply()`/`_cp5_walk`. 데이터=sim_results/bank_eplansmoke_full_20260723.

## 0. 문제 (실측 확정)
`_cp5_walk`(현 배선)은 `_check_termination`을 wrap해 **`done=True` ∧ `user_stop` ∧ 미walk(cap1)** 일 때만 발화
(t2_eplan_patch.py:749-751). 발화 시 `done=False`로 **1턴 보류** + 생성-레벨 리마인더 1회. 그러나:
1. **개입 시점=종료(user_stop)** — 이미 손님이 나간 뒤. 늦음.
2. **1턴 보류로 N-item 갭 못 닫음** — dispute 1건=탐색+unlock+call 다수턴. 1 보너스턴에 8건 불가.
3. 리마인더=soft(§1.5 write강제 금지). → **coverage 1/11.**

## 1. 근본 진단 (§2bw 정합)
- **세기(count)=정확·결정론**: 갭 8=real(task_039=진짜 dispute). false-gap은 *비-dispute 태스크*(050)서만
  = 소스 문제(list_from_reads가 all-reads·§2 아래).
- **몰기(drive)=병목**: 개입이 "끝·1회"에 묶여 손님 나가기 전에 사슬을 못 끈다.

## 2. 설계 — mid-conversation coverage drive (E-PLAN v2)

### 2.1 핵심 전환: "종료 시 1회 보류" → "갭 열려있는 동안 매 드리프트 견인"
- **트리거(WHEN)**: `_check_termination`(종료) 대신, **에이전트 턴 생성 직후** 검사. 조건 3AND:
  ① **flagging-source가 표적셋 산출**(count N 결정론·§2.3) ② **executed M < N**(갭) ③ **직전 턴이 갭 미진행**
  (dispute write 아님 = 드리프트). → 손님 나가기 前, 드리프트 순간 조기 개입.
- **행동(WHAT)**: 생성-레벨 리마인더(히스토리 비커밋·채널 절대규칙) 주입: *"[source]가 N건을 dispute 표적으로
  표시했고 M건 접수됨. 남은: [ids]. 다음 건을 지금 file하라."* 에이전트가 각 dispute를 **직접 emit**(강제 0).
- **종료(STOP nudging)**: 갭 닫히면 종료 / **K턴 연속 미진행이면 중단**(무한 넛지·derail 방지·에이전트 정당사유 존중).

### 2.2 왜 "매 턴"이 아니라 "드리프트 시"인가
매 턴 넛지=대화 derail·[[73]] 위험. **갭 열림 ∧ 직전 턴이 dispute 아님**(에이전트가 다른 데로 샘) = 정확히 견인
필요 순간. 에이전트가 dispute 진행 중이면 넛지 0(방해 안 함).

### 2.3 count 소스 = flagging-source (false-gap 해소 + semantic 분담)
현 `eplan.list_from_reads=true`는 **읽은 거래 전부** 열거 → 050서 false-gap. 교정:
- count N = **flagging-source 출력에서만**: `get_reward_discrepancies`(결정론 도구가 discrepant 표시)의
  반환 transaction_id ∪ **손님-명시 dispute 대상**. A2에 `flag_source` 선언(도메인특화=A2·[[05]]).
- ⇒ **선택(어느 게 대상이냐)=도구/손님(결정론·입력)** · **세기·견인=엔진**. 순수-semantic(맨땅 판단) 케이스는
  LLM 선택분만 견인(선택 자체 불강제=F3 경계 존중).

## 3. [[05]] 3질문 (필수·[[17]])
1. **도메인-특화 순증?** — 엔진 walk/드라이브 로직=도메인일반. 도메인지식(flag_source 도구명·write_tools)=
   **A2 `eplan`에 이미 있음**(list_enumerator/write_tools) + `flag_source` 1키 추가. 엔진 리터럴0. **A2 데이터**.
2. **유동판단 동결?** — **No.** "어느 거래가 dispute 대상이냐"(semantic)=get_reward_discrepancies(결정론 도구)
   또는 손님이 정함·엔진 안 정함. 엔진=flagged 셋 **세기+coverage 견인**만. 순수판단은 LLM 몫으로 남김.
3. **스캐폴드가 write 수행?** — **No.** 엔진은 리마인더(넛지)만·각 dispute는 **에이전트가 emit**. autofetch/
   auto-write 아님. §1.5 write강제 금지 준수(soft nudge·cap).

## 4. 위험·계측 (모트 §1.3)
- **Δspurious(과다 dispute)**: count가 flag-source에 정박돼 상한. flagged 밖 dispute=계측·억제.
- **derail([[73]])**: 드리프트-시만 발화 + K턴 미진행 중단. 리마인더=비커밋(대화 오염 0).
- **over-nudge**: 진행 중이면 발화 0. cap(진행-감응).

## 5. 격리 검증 계획 (구현 前 무료)
1. **결정점 replay**(greenlight/approve_iso 동형): task_039 궤적을 *get_reward_discrepancies 출력 있음 + dispute
   1건 + 드리프트* 지점에서 얼려, mid-drive 리마인더 주입 → 재생성이 **다음 dispute를 file하나** vs 드리프트.
   대조: 무-넛지(드리프트 재현)·user_stop-보류(현재).
2. **엔진 단위**: flag_source count(=get_reward_discrepancies 반환 수)·coverage_gap·드리프트 판정 selftest.
3. 통과 시 라이브 A/B(off vs mid-drive)·nt≥3·Δspurious 계측.

## 6. 미해결·한계
- 손님이 **아주 일찍** STOP(dispute 식별 前)이면 견인 불가(대화 scope 자체 부족·별개).
- 순수-semantic 표적(flag-source 없음)=LLM 선택 신뢰·엔진은 사후 coverage만.
- 구현 위치: `_check_termination` wrap → **에이전트 `_generate_next_message` wrap**(t2_gate_patch.py 리마인더
  소비 패턴 재사용)으로 이동 필요(엔진 배관 진화).
