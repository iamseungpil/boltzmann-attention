#!/bin/bash
# ★정본 GO-STACK 런처 (single source of truth · 2026-07-25 C167)
# 목적: 스택 조성이 세션/사람 기억에 의존해 플래그가 누락되는 사고 방지([[07]] hard-constraint).
#   모든 라이브 런은 이 파일을 source한 뒤 t2_launch로 띄운다. 실험 arm은 그 위에
#   T2_* 플래그를 추가/제거하고, 그 차이를 런 태그와 원장에 명시한다.
#
# ★합성-우선 원칙(사용자 지시 2026-07-25·C168): **성공한 레버는 전부 함께 켠다** —
#   간섭은 합성 런에서만 드러나고, 드러나야 레버간 조정이 가능하다(격리 검증만으로는
#   간섭이 영원히 미지·"최종 스택"이 미검증 상태로 남음). 실증: UNIFIED=gate+prov 합성이
#   드러낸 CONFLICT의 조정물·guided pre-gate 순서=합성 라이브가 드러낸 관통의 조정물.
#   ⇒ 기본 = 전부 ON. 간섭 관측 시 레버를 끄는 게 아니라 **조정**(통합·순서·창/캡)한다.
#   개별 격리 검증은 "귀속용 실험 arm"에서만(그때 명시적으로 끄고 태그에 기록).
#
# ── 환경(정본) ──────────────────────────────────────────────────────────────
export GO_REPO=/home/woori/workspace_common/boltzmann-attention-pi
export GO_TAU2=/home/woori/scratch/tau2-bench           # ★정본 tau2([[30]]·C166 사고 재발 방지)
export PYTHONPATH=src:$GO_REPO/scripts/distill/tau2
source /home/woori/.openrouter_key                       # export OPENROUTER_API_KEY=... 형식

# ── [GO] 기본 스택 (nt4 계보·E11-e2e GO·C146~C149 아크) ────────────────────
export T2_OVERFLOW_GUARD=1
export T2_GATE_REGEN=1          # UNIFIED regen의 gate 축 (단독 아님·아래 PROV와 통합 라우팅)
export T2_GATE_REGEN_K=1        # (구 C173의 K=2는 unified 경로서 미사용=no-op으로 판명·철회.
                                #  비-unified 분기 전용 knob이라 기본 1 유지.)
export T2_PRECLOSE_CAP=2        # ★C173-corr(2026-07-25): 진짜 원인=pre-close deny가 공유
                                #  T2_EPLAN_DENY_CAP(4)을 discovery deny와 나눠 써서 044서 소진→
                                #  2번째 close 통과(CLOSED·상태오염 3). 전용 예비 예산으로 분리
                                #  (t2_gate_patch 3386~·claimprov transfer-창 §2ao 선례 동일 패턴).
export T2_PROV_REGEN=1          # 출처선언/provenance (E11 GO·C45 67→0%)
export T2_PROV_REGEN_K=4
export T2_PROV_MODE=full
export T2_GROUND=1              # P-A GROUND (T5-C rev3)
export T2_EPLAN=1               # E-PLAN ledger+walk ([[14]])
export T2_EPLAN_WALK=1
export T2_BRANCH_REGROUND=1     # C146 make-or-break GO·C149 close-차단 인과 [S]
export T2_SCAFFOLD_GET=1        # A2 scaffold_get_tools (검증기 GET)
# ★C186(2026-07-25·W7 004 부검서 발견한 **런처 누락 회귀**): 아래 둘은 2026-07-18/20에 검증돼
#   당시 런 스크립트 10여 개(run_e2e9/e2e10/r095*/hv_*/eplan_smoke)가 켜던 레버인데, go_stack이
#   정본 런처가 되면서(C167/C168) **조용히 빠졌다** = [[19]] 합성-우선 위반.
#   · T2_A2_VARIANT=ledger → verify_identity의 **record 슬롯 삭제**(match_verdict_grounded).
#     근거 `VERIFY_IDENTITY_LEDGER_BINDING_DESIGN_2026_07_18`: record 날조 46%·grounded 0/24·
#     A2 설명 레버 0/24·라이브 PROV 미포착 ⇒ 슬롯 삭제만 남음. **누락의 대가 = task_004**:
#     조회 실패("No records found") 후 모델이 record를 날조(DOB 01/15/1985·"123 Main St")했고
#     우리 도구가 그것을 모델 자신의 provided와 대조해 **VERIFIED 발급**(가짜 검증).
#     변이 ON이면 record 인자가 아예 없어 이 경로가 구조적으로 불가.
#     (ratefix = get_reward_discrepancies rate 테이블 교정본·같은 시기 검증분)
#   · T2_SG_GROUND=1 → A2 `ground` 선언 도구(check_rebate/apy/interest/closure 4종)의 operand를
#     KB/원장 대조로 검증·미검증은 드롭→abstain(가짜 정밀도 차단).
export T2_A2_VARIANT=ledger,ratefix
export T2_SG_GROUND=1

# ★★C186 검증-레버 복원(2026-07-25·[[19]] 이행) ─────────────────────────────
#  발견: go_stack은 C167서 **13개로 새로 작성**됐고(tiers GO/VAL/TGT+promotion rule) C168이
#  "성공 레버 전부 ON"을 선언했지만 **승격이 실행되지 않아** 직전 검증 런(`run_rall25_20260724`
#  =56 레버)의 **43개가 스택에서 이탈**했다. 그런데 handoff/메모리는 go_stack을 "전 레버 ON"으로
#  기록 ⇒ 문서-실제 불일치. C185 34-fail 포렌식이 "미설계"로 분류한 표적 다수가 **이미 구현된
#  레버를 끈 상태**였다: ⑤KB반복=READ_DEDUP · ⑤컨텍스트=VIEW_COMPACT · ③가공도구=UNKNOWN_NAME_BL
#  · ①라우팅=DISPATCH_ROLE/UNLOCK_NAME · ⑨값=WRITE_EVIDENCE/REF_VERIFY · ②완료날조=FOLLOWUP_*.
#  ⚠**단서(정직)**: 56-레버 시대 런(rall21~25)은 0/4였고 18-레버 go_stack이 12 pass를 냈다 —
#  즉 "많을수록 좋다"는 증거는 없다. [[19]] 대응은 끄기가 아니라 조정이므로 **복원 후 기준셋
#  (032/033/035/043/058) 재측정으로 회귀를 확인**해야 하고, 그 전에 스모크 필수([[30]]).
#  ⚠**런타임**: SG_ISOLATE/FORCE 계열은 서브콜·토큰을 늘린다(20~60분/태스크 관측).
export T2_COMPUTE=1 T2_RESOLVE=1 T2_ARG_SCHEMA=1 T2_TOOLGATE=1
export T2_SG_TRUTH=1 T2_SG_ISOLATE=1 T2_SG_ISOFB=1 T2_SG_REQREADS=1 T2_SG_TRACE=1
export T2_FAB_STRIP=1 T2_UNKNOWN_NAME_BL=1 T2_UNLOCK_NAME=1 T2_UNLOCK_PROV=1
export T2_DISPATCH_ROLE=1 T2_TOOLLIST=1 T2_PRESCRIPTION=1
# ★T2_DISPATCH_ROLE_ENVSET (2026-08-05 등재·구현은 C257·`ABSENCE_DRIVEN_PROCEDURE_DESIGN` §4):
#   give 대상의 판정 집합을 `self.tools` 소속 → **env가 실제로 넘길 수 있는 집합**으로 바꾼다.
#   구판은 *존재하지 않는 이름*을 통과시킨다 — 012가 `navigate_to_travel_notification`을 손님에게
#   건넨 자리다. x88 전수(N97B·194 sim): give 342건 중 **집합 밖 252건(12 sim·통과 0)**,
#   그리고 **gold이 요구한 give 41건 중 집합 밖 = 0건** ⇒ 오차단 0으로 등재 가능.
#   ⚠접미사 규칙을 give로 확장하지 말 것: user-discoverable 4종 중 `get_card_last_4_digits`·
#   `get_referral_link`는 무접미사라 정당한 give가 상시 차단된다. 술어는 **집합 소속**이다.
export T2_DISPATCH_ROLE_ENVSET=1
export T2_WRITE_EVIDENCE=1 T2_WEV_ROUNDS=2 T2_WRITE_ARG_GROUND=1 T2_WRITE_PROV=1
export T2_REF_VERIFY=1 T2_VALUE_ACQUIRE=1 T2_HAVE_VALUE=1 T2_HAVE_VALUE_FORCE=1
export T2_FOLLOWUP_REQUIRED=1 T2_FOLLOWUP_FORCE=1 T2_FOLLOWUP_READLOOP=1
export T2_FOLLOWUP_CAP=3 T2_FOLLOWUP_PROGRESS_REFUND=1
export T2_FORCE_ACTION=1 T2_ACTION_DENY_CAP=3 T2_ACTION_PROGRESS_REFUND=1
export T2_VERIFY_DENY_CAP=2 T2_PARAM_CAP=1 T2_PAIRCHECK=1 T2_PAIRFIX=1
export T2_STALE_STRIP=1 T2_READ_DEDUP=1 T2_VIEW_COMPACT=1 T2_VIEW_ANNOTATE=1
export T2_COV_MIDDRIVE=1 T2_COV_MIDDRIVE_K=4 T2_EPLAN_DRIVE_K=4
export T2_REGEN_BUDGET=12 T2_LLM_RETRIES=1 T2_LLM_TIMEOUT=2400
export T2_AGENT_MAX_TOKENS=8192   # ★폭주 상한(C271 실측으로 복원) — 근거는 바로 아래 주석
# ★생성 설정 표준 (2026-07-31·C269→C271) — 런처가 쥐고 있어야 런마다 표류하지 않는다.
#   ① `T2_LLM_TIMEOUT=2400` (구 480) — 480은 느린 턴을 조기 절단한다. Y1·Y2-B가 쓴 값으로 통일.
#   ② **`T2_AGENT_MAX_TOKENS=8192`** — 한 번 빼봤다가 **첫 런에서 폭주가 재현돼 되돌린 값**이다(C271).
#      캡을 빼면 프롬프트 천장 자해는 사라지지만(C208①), 폭주 응답의 상한이 **창 전체**가 된다.
#      ★2026-07-31 Y2-C 1차 발사 실측: 발사 29분에 4 sim 전부 정지 · 완료 **0/32** ·
#      서버 지문 `prompt 0.0 / gen 10.7 tok/s / Running 1`이 **117 상태라인 연속**(≈18,600토큰 단독 디코드)
#      = C205 지문과 동일. 캡 8192면 13분에 절단될 것이 timeout 2400(40분)까지 가고 재시도까지 붙는다.
#      **비대칭이 결정 근거**: 천장 손해는 `T2_DYN_MT`가 자동 회복하고(Y1 64 sim 중 8회 전부 회복),
#      폭주는 **회복 수단이 없다**. 그래서 캡을 되살린다(값은 Y1과 동일 = 짝비교도 유지).
#   ⚠**짝비교 주의**: Y1은 `max_tokens=8192`로 돌았다. 이 설정으로 도는 런을 Y1과 짝지으면
#      프롬프트 창이 다르다는 **알려진 델타**를 안고 비교하는 것이다 — 판정문에 반드시 명시할 것.
# (제외 유지: dd_fb·retry·투표 = 실측 해로움·C154/C168)

# ── 신규 레버 (합성-우선 원칙에 따라 전부 기본 ON·C168) ─────────────────────
export T2_GUIDED=1              # C162 실증·C166 체인수정(pre-gate 순서=합성 조정물)
export T2_PREKB=1               # C165 행동-키 검색 게이트
# ★C204/D7(2026-07-27): 동일-인자 계산도구 반복=결정론 stub(022 ctx초과 10회·003 5회 실측 표적).
#   evidence_from(원장-의존)·fetch_formalize(env-가변)는 자동 제외 — 005형 정당 재호출 보호.
export T2_SG_DEDUP=1
# ★C207(2026-07-27·day4b ctxover 20건): 폭주-디코드 방어 3종.
#   ENVELOPE_GUARD=봉투 파싱 실패(정지 실패)→required-channel regen · TRUNC_GUARD=length 절단 미커밋(cap 1)
#   UNAVAIL_PROMISE=미보유 기능 약속 차단(집합 대조). 전부 프레임워크 층·도메인 리터럴 0.
export T2_ENVELOPE_GUARD=1 T2_ENVELOPE_CAP=2 T2_TRUNC_GUARD=1 T2_UNAVAIL_PROMISE=1

# ── ★C208/day5 처방(DAY5_PRESCRIPTIONS_DESIGN_2026_07_28·P1~P10) ─────────────
export T2_DYN_MT=1              # P1 동적 max_tokens(CWE 파싱→축소 1회 재시도·플로어 미만=graceful)
export T2_MT_FLOOR=256 T2_DYN_MT_MARGIN=64
export T2_TERM_GRANT=1          # P3 터미널-턴 보장(notice 공표+동의+미호출→1턴 유예·required)
export T2_ABSTAIN_FIELDS=1      # P4 abstain 결핍-필드 지목(coverage에 필드명+공급 지시)
export T2_PROD_BIND=1           # P4b producer-binding(A2 grounded_params·날조=결핍 강등)
export T2_DUP_REPRESENT=1       # P8 DUP-COMPUTE 스텁에 이전 결과 재제시(상한 2·shrink 시 생략)
export T2_FAILED_PERSIST=1      # P10 실패-sim 궤적 사이드카(set_state 예외 시 덤프)
export T2_VIEW_COMPACT_MINTOTAL=60000   # P5-1 문턱 하향(구 120000=사망선 위·day5 6/32 발동)
export T2_VIEW_MSG_CAP=8000     # P5-2 per-메시지 뷰 캡(최신 배치 제외·리뷰 필수1=배치 전체 전문)
# ── ★C211/day7 처방(DAY7_PRESCRIPTIONS_DESIGN_2026_07_28·F6~F8) ──────────────
export T2_SG_BYREF=1            # F7a 참조-전달 승격(day6 W-f 실측=GO 충족·+F7b A2 equijoin)
export T2_ARG_PRODUCERS=1       # F8 필수인자-생산자 give-흐름 넛지(040/041 오도구 전환 표적)
# F6a/F6b=A2 선언·버그픽스라 스위치 없음(P4b/FAB_STRIP 기존 플래그 하위).
# OFF 유지(승격 조건 명기): T2_READ_NEARDUP(P5-3·오탐 계측 후)
export T2_CLAIM_PROV=1          # claim-날조 원장대조(사임/transfer 창·035 기전 표적)
export T2_CLAIMPROV_CAP=3       # cap=1은 빈손 regen 1회에 전소(코드 포렌식 실측)→스모크 권장 3
# ── ★C212/day7 중간-포렌식 처방(DAY8_PRESCRIPTIONS_DESIGN_2026_07_28·A1~A4/B1~B3) ──
# ★T2_DISPATCH_ROLE_NOTE 폐기(2026-07-31): 딸린 strip(`tool_arg_allowlist`)을 V7로 대체해
#   재진술할 "떼어낸 값" 자체가 없다. 021형 좌초 방지는 V7 피드백 문구가 진다(§아래).
export T2_TOOL_SIGNATURE=1      # ★V7 give 서명 deny+재발행(구 strip 대체·C151 compliance 패턴)
#   ⚠구 strip은 엔진이 호출을 대신 고쳐 로그 위반을 0으로 만들었다(Z4: strip 2 / V7 0).
#   이제 모델이 고친다 — 무한 deny 방지는 rule① RETRY_LOOP(동일-호출 반복 차단)에 의존하고,
#   통과-캡은 두지 않았다(순수 compliance 측정). 좌초 관측 시 cap 도입이 조정 후보([[19]]).
export T2_TERM_GRANT_USERDEMAND=1  # A4 유저 ###TRANSFER### 직접-방출 시 notice-요건 면제(008 [S])
export T2_COVERAGE_FOLLOWUP=1   # B1 [coverage] 미판정-행 재호출 지시 무시+사임→1회 regen(019/022/027 [S])
export T2_UNKNOWN_REPEAT_GUARD=1  # B3 Unknown-tool 반려된 이름 재지시 차단(cap 2·010/014/015/016 [S])
# A1(FOLLOWUP tool_args 이행판정)=A2 선언·엔진 하위호환이라 스위치 없음. B2=A2 ask 문구.

# ── 간섭 감시점(합성 런에서 로그 마크로 확인·관측 시 '조정'이 기본 대응) ────
#  W1 claim_prov × EPLAN drive: 둘 다 사임/user_stop 창에서 발화 → 같은 턴 이중 넛지
#     (over-steer) 여부. 마크: [T2_CLAIMPROV]·[T2_EPLAN] drive 동일 턴 공발화.
#  W2 claim_prov × PREKB: transfer 호출이 양쪽 창을 동시 트리거(생성-레벨 감사 +
#     실행-레벨 deny) → C152형 포기 유발 여부. 각각 캡 有(1/fam·1/sim)로 유계.
#  W3 guided × claim_prov 서브콜: 감사 서브콜은 tools=None → 문법 미주입(무간섭 확인됨).

# ── 폐기/실험전용 (기본 OFF 유지 — '성공한 레버'가 아님) ────────────────────
# export T2_DD_FB=1             # C154 폐기 권고(soft·교란)
# export T2_MAXPROMPT=1         # 프롬프트-한계 실험 전용

# ★리더보드 정합 기본값(2026-08-02·[[54]]·LEADERBOARD_TRACK_DESIGN):
#   retrieval_config = alltools (기본 — 보드 상위권 전부 alltools/Terminal·bm25 항목 0개)
#   user reasoning_effort = low (GPT-5.5 제출이 gpt-5.2 user-sim을 low로 돌림)
#   arm별 덮어쓰기: GO_RETRIEVAL=bm25 / GO_USER_EFFORT= (귀속 실험용)
#   ⚠alltools는 OPENAI_API_KEY 필요 — 발사 전 `source /home/woori/.openai_key`
# ── 공통 런처 함수 ──────────────────────────────────────────────────────────
# 사용: t2_launch <TAG> <PORT> <TASK_IDS> <NUM_TRIALS> [EXTRA_ARGS...]
t2_launch() {
  local TAG="$1" PORT="$2" TASKS="$3" NT="$4"; shift 4
  cd "$GO_TAU2" || return 1
  /home/woori/venvs/seka_env/bin/python -u "$GO_REPO/scripts/distill/tau2/t2_run_gated.py" \
    --domain banking_knowledge --retrieval_config "${GO_RETRIEVAL:-alltools}" \
    --gate 1 \
    --agent_model Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8 \
    --agent_base "http://localhost:${PORT}/v1" \
    --user_llm openrouter/openai/gpt-5.2 --user_temp 0.0 \
    --user_reasoning_effort "${GO_USER_EFFORT:-low}" \
    --task_ids "$TASKS" --num_trials "$NT" --max_concurrency "${GO_CONCURRENCY:-4}" \
    --max_steps 200 \
    --save_to "$TAG" "$@"
}
# ── ★C213/day9 처방(DAY9_PRESCRIPTIONS_DESIGN_2026_07_29·경계정본 §4) ─────────
export T2_GIVE_RELEVANCE_NUDGE=1  # N1 원장-미등장 give 확인 넛지(021 DB-오염 [S]·강제 금지·cap1)
# W1: EPLAN walk 강제-보류는 기본 OFF로 강등(gap=표면화만·001 [S]) — 종전 보류는 T2_EPLAN_WALK_HOLD=1(격리 arm 전용)
# G1: notice 공용 정규화 술어(gate_interpreter.notice_sent_in)=무스위치 교체(GB2·EPLAN·compliance 3층 일원화)
# R1: 제외(017=후속-턴 자체 발화 확인→learn 축). T1: 접지 기선언 확인(코드 변경 0)·undercount 조사 별도.
# ── ★C214 (day9 재발사 전 A/B 보강·DAY9 설계서 §5 추가분) ─────────────────────
export T2_UNVERIFIED_FOLLOWUP=1   # E1 unverified 조건-확정 재호출 넛지(003 [S]·비강제)
export T2_GIVE_EXEC_NUDGE=1       # E2 give 성사·user 미실행 시 실행 안내 넛지(019 [S])
export T2_SEARCH_EXHAUST_NUDGE=1  # E3 중복-검색 소진 시 전략 전환·날조 금지 넛지(012/033/032 [S])
export T2_SEARCH_EXHAUST_TH=2
# E4=A2 dispute→update 체인 resign_th=1(028 구조적 미발화)+`chain suppressed` 진단 마크

# ── ★AX33 처방 스택 (AX32_MIDRUN_PRESCRIPTIONS_DESIGN r7·2026-08-02·리뷰 승인분) ──────────
#   구현 완료분만 등재한다. 미구현(P1·P6·P8)·재설계 대기(P2·P10)·보류(P3)는 넣지 않는다.
#   ⚠P4ⓐ(rebate 윈도 산식)는 엔진 수정이므로 플래그 없음 — 수정 전까지 P8은 차단(설계서 §F).
export T2_WRITE_ARG_GROUND=1      # P9 give 내포 인자 접지(A2 write_arg_grounding에 give 추가·028/040)
export T2_ARG_SCHEMA=1            # P11 스키마-밖 최상위 키 위생(unified 이설 완료·표적 0=위생)
export T2_TOOL_CHANNEL=1          # P13 채널 오분류 — **예방형 생성-레벨**(출력-부착 금지·041 사고)
export T2_USER_TOOL_NOTE=1        # P5 user-tool 안내 표준문(018/040·생성-레벨·sim당 1회)
export T2_GIVE_QUOTE=1            # P1 give-인용 실재성(010 재현 2/2·생성-레벨·재질의 1회 fail-open)
export T2_DISPATCH_LEDGER=1       # P8 제출-완결 표면화(020/027·터미널 훅·deny 아님·1회/sim)
export T2_SG_WINDOW_ABSTAIN=1     # §4-2 미측정 윈도 abstain(023 부정-오판정 봉쇄·A2 선언 도구만)
export T2_KB_NOHIT_SURFACE=1      # P2/P10 bm25 전-0점 표면화(012=절차 날조 금지·014/015=주장 미뒷받침)
export T2_KB_NOHIT_K=2            #   연속 무득점 문턱(설계서 §P2 k=2)
export T2_TRANSFER_TIER=1         # P15 이관 사유 티어 표면화(004 실측·정책 doc_042 티어표·A2 구동)
#   P12(axis replay 가드)·P7(byref 오류 문구)·§4-1(byref 중첩 op 트리)은 플래그 없는
#   **무조건 위생**이라 등재 불요.

# ── ★N97 처방 스택 (N97_LEVER_PRESCRIPTIONS_DESIGN_2026_08_04·리뷰 2회 반영) ──────────
#   근거 = 실패 50 태스크 매턴 정독(N97_TASKWISE_FORENSIC) + 전수 194 sim 재계량(x73).
#   구현 완료분만 등재한다. P2는 폐기(술어는 정상이었다·설계서 §2), P6는 **P3 이후 재계량**이
#   GO 조건이라 아직 없다.
export T2_CALLABLE_HINT=1         # P3 통지에 접미사 포함 호출형 동봉([READ-FIRST] 44발화/18sim·051이 이것만 없어 이관)
export T2_REPEAT_CAP=8            # P4 동일 호출 K회 후 실행 중단·전환 요구(경고 182 ↔ 반복 71·084/t1 max_steps 사망)
export T2_QUOTE_HINT=1            # P5 인용 반려 시 원장 표기 지목 — **값이 원장에 실재할 때만**(046 t0/t1 자연실험)
export T2_PIN_READ=1              # P1 선행 read를 named tool_choice + 단일값 enum으로 1회 고정(x72 3/3·replay 무관)
#   ⚠P1 단독 기대 pass 증가 = **0으로 사전등록**(설계서 §0c). 관문 도달 26 sim도 전부 실패했으므로
#   1차 지표는 pass가 아니라 원인별 이동(관문 미도달 53 / 0점 410행 / shell 식별자 49 / 최대 반복 71).

# ── ★확정-미등재 처방 회수 (2026-08-05·정본 `ROOTCAUSE_LEVER_ATTRIBUTION_2026_08_05.md` §3·§4) ──
#   둘 다 **구현·검증을 마치고도 이 파일에 오른 적이 없어** 죽어 있었다([[24]] 死코드 패턴의 재발).
#   드라이버(`run_n97_nt2.sh`)는 이 파일만 source하므로, 여기 없으면 라이브에서 존재하지 않는다.
#   N97B가 그 대가를 실측했다 — 아래 각 줄의 근거는 이번 런의 궤적·인자 실물에서 나온 것이다.
export T2_QUOTE_PIN=1             # C278/C279 pin_kind 라우팅(C197 열린-술어 가드를 종류별 닫힌-검사로 교체).
#                                 #   근거: C282 라이브 — 022가 5런 전패 코어에서 PASS로 뒤집혔고 사슬 전 구간
#                                 #   확인(discrepant 10/10·`77 of 77`·드롭 0·false-apply 0). N97B에서 OFF였던
#                                 #   대가 = `txn_ba8b473f295d` 오차단으로 **022 t0/t1 2 sim 소각**.
#                                 #   ⚠회수 아닌 것: 019(Thrive Market)는 **정당 차단**이라 QP로도 안 열린다(C289).
#                                 #   판다(−): 표 밖 산문-범주는 여전히 열린 잔여(케이스 12)·핀 방향 오류 1/7(C289⑥).
export T2_KB_DOCS_DIR="$GO_TAU2/data/tau2/domains/banking_knowledge/documents"
#                                 # ⚠MATCH_COUNT의 **의존물**. 2026-08-05 스모크 실측: 플래그만 등재했더니
#                                 #   `matches:` 주석이 궤적에 **0건**이었다 — 코퍼스를 못 찾으면
#                                 #   `t2_match_count.note()`가 조용히 None을 반환한다. run_b4.sh는 이 줄을
#                                 #   갖고 있었고 go_stack은 없었다(=플래그만 옮기고 의존물을 안 옮긴 것).
export T2_MATCH_COUNT=1           # KB_search 회수 경계 표면화("N개 걸림 중 K개 표시" ↔ "전부 표시").
#                                 #   근거: x75 재계량(2026-08-05·B4 202주석) **인증 126/202=62%** = 설계서 §4
#                                 #   등재 기준(과반) 충족. P3(CALLABLE_HINT)가 만든 **0점 검색 410→1024(+150%)**의
#                                 #   직접 짝 처방 — 질의가 구(phrase)라 0점인 것을 모델이 볼 수 있게 한다.
#                                 #   replay 안전: KB_search는 비-mutating이라 P13 규약(출력-부착=읽기 전용) 준수.
#   ⛔ 같은 날 구현된 신규 5종(N2a·N2b·N1·L4·L5-a)은 **등재하지 않는다** — 라이브 발화 미검이고,
#      특히 N2b는 허가 술어의 지배 분기(`_customer_stated`)가 미교정이라 지금 켜면 측정이 무효다
#      (`THEORY_AUTHORITY_LICENCE_2026_08_05.md` §3·C295).

# ── ★절차 준수 (2026-08-05·`t2_procedure.py` + A2 L3 `procedures`) ─────────────
#   정책이 **순서를 명령**하거나 **도구를 금지**한 흐름만 A2가 index로 선언하고, 그 흐름에
#   진입한 뒤에만 검사한다. 차단 허가는 선언이 준다(`enforce`+MUST 문장 / `prohibits`+금지 문장).
#   과차단 사전 계량(x80·194 sim 전수): **0건**. 진입 개념 없이 노드 이름만으로 판정하던 1차
#   설계는 28건을 오차단했고 그 측정으로 폐기했다. 후보 rule "회수 문서가 이름을 댄 도구만
#   give"도 8건 오차단으로 기각(x81).
export T2_PROCEDURE=1
# ★D1′ 부재-구동 체크리스트 (2026-08-05·설계 `ABSENCE_DRIVEN_PROCEDURE_DESIGN` §2·게이트=x86):
#   절차에 **들어와 놓고** K턴 동안 그 절차 쪽으로 아무 호출도 없으면 선언의 체크리스트를 표면화한다.
#   차단 아님(비커밋 피드백 1건)·▶NEXT는 `enforce` ∧ 후보 유일일 때만·동렬이면 목록만([[10]]).
#   x86 전수(194 sim·K=3): 발화 54회/29 sim · ▶유일 98.1% · **gold-밖 지목의 write 0** ·
#   지목 도구의 **100%가 미-unlock**(048 livelock서 모델에게 없던 유일한 정보).
export T2_PROC_ABSENT=1
export T2_PROC_ABSENT_K=3         # 무호출 연속 assistant 턴 임계(x86 K-sweep 2/3/5 전부 write 0)
export T2_PROC_ABSENT_CAP=2       # sim당 상한(불응 시 조용히 소진 — 기존 cap 규약)
export T2_PROCEDURE_CAP=6         # sim당 deny 상한(불응 무한루프 방지·기존 cap 선례)
export T2_UNINSTRUCTABLE=1        # 실행 불가 지시 차단(012): 손님에게 도구 실행을 안내했는데 전달 이력 0.
#                                 #   술어=A2 L1 선언 토큰 포함 ∧ 전달 마커 부재(정규식 추출 0·C279 계보).
#                                 #   사전 계량(x82·194 sim): 발화 43 sim(1회/sim 캡)·그 중 17은 나중에
#                                 #   전달이 실제로 일어남(넛지가 이르지만 문구가 "먼저 전달하라"라 무해).
export T2_CHOICE_GROUND=1        # 계좌 클래스 등 열린-문자열 선택의 접지 넛지(x84: 이득 7·gold 미접지 3이라 deny 금지)
