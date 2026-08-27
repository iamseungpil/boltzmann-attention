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
# ★오프라인 전용 · 런 노브 아님(export 하지 않는다). 여기 적는 이유는 `test_flag_registry`
#   래칫이 **엔진이 읽는 T2_* 는 전부 정본에 이름이 있어야 한다**를 강제하기 때문이다.
#   T2_FORENSIC_DENY_LEGACY=1 → `t2_forensic.deny_kind` 를 2026-08-24 수리 **이전** 판정으로
#   되돌린다(우리 거절을 env 로 찍던 그 판). 수리의 양성대조 전용이고 라이브 거동과 무관하다.
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
# ★2026-08-21 등재(t7335·레지스트리 래칫): 승자 합성 + 신규 전달 레버를 정본에 선언한다.
#   · T2_SG_DOCS=1       A3 `isolate.docs` 읽기-명세를 엔진이 잘라 서브에 전달(검색 0·[[71]]).
#     격리 실측 C585: 관문1 생존 bm25 45% → 87.5%(x456 v2·n=8). 실패=종전 검색 폴백(거동보존).
#   · T2_ARG_DOC_SUB=1   선언 문서 → 격리 서브 → 값+인용·엔진은 인용 실재만 검산(C576 71/71).
#   · T2_VALUE_FORMULA=full  범주 요율 × 손님 발화 금액 − 연회비(C562 값 0.98·C580 합성 8/8).
#   · T2_CATEGORY_CITE=  기본 OFF(2026-08-20 측정: 라이브 거동 0) — 켜려면 재측정 후([[60]] 통합).
export T2_SG_DOCS=1 T2_ARG_DOC_SUB=1 T2_VALUE_FORMULA=full
# ★T2_SG_SCHEMA (2026-08-22 등재·구현 같은 날): 격리 서브의 **마감 라운드에만** A2
#   `isolate.operand_schema` 를 guided_json 으로 건다(도구 0인 라운드 — 문법과 도구를 같이
#   걸면 tool_calls 가 0 이 된다·t2_declfirst 실측 C248).
#   왜: 산문 `answer_format` 으로 형식을 부탁하면 서브가 **예시의 값을 베낀다** —
#   `{principal: 0.0, actual_apy: 0.0}` 가 t7337·t7338 두 런에서 재현됐고, 이 자리는
#   `:2be` 주석이 *"§2as 0.0-포이즈닝의 신형 재발"* 로 이미 이름 붙인 곳이다(당시 처방은
#   답 폐기 = 증상 억제였고, 그 폐기가 폴백→추측→grounding 드롭→도구 None→자기계산 write→
#   WEV deny 의 livelock 을 낳았다·093 실시간 포렌식 2026-08-22).
#   ⇒ 부탁 대신 문법으로 형식을 보장해 **베낄 예시 자체를 없앤다**. 엔진은 형식만 강제하고
#   값은 여전히 서브가 낸다([[62]]·[[10]]). 스키마 출처는 A2 하나뿐(엔진 리터럴 0·[[05]]).
#   선행 근거: declfirst 2패스 실측 — 프롬프트만 32% ↔ 도구미제공+문법 96%(C250).
#   ⚠[[70]] 무엇을 파는가: "JSON 하나만" 이 강제되면 서브가 추론할 자리를 잃는다 ⇒ 스키마의
#     **첫 required 필드를 `derivation`** 으로 두어 추론이 값보다 앞에 오게 했다(파싱은
#     `_merge_json(content, operand_keys)` 라 추가 필드를 무시한다).
export T2_SG_SCHEMA=1
export T2_CATEGORY_CITE=
# ★T2_SEARCH_REARM (2026-08-22 등재·구현 2026-08-21 커밋 992b7d53·정본 `T7336_FORENSIC_016_2026_08_21.md`·
#   격리 C591 x464): 검색 에이전트의 축-소진 키를 군 → **(군, 배달된 계열 집합)** 으로 좁히고, 배달분에
#   없던 계열이 발화에 축자 등장하면 재무장해 **그 계열의 문서 델타만** 배달(선언 id → 정확 집기·[[71]]).
#   격리(n=9/팔): 정책값 확보 0/9 → 6/9. 판 것 = 문맥 +22k자. ⚠`_search_material` 안의 하위 스위치라
#   **검색 에이전트 플래그(`SEARCH_AGENT`)가 켜진 런에서만 산다**(런처 PIN 이 켠다 — 이 파일은 그것을
#   선언하지 않는다·래칫 기준선에 미선언으로 남아 있음·여기 이름을 적으면 래칫이 "선언"으로 오인하므로
#   접두사 없이 적는다). OFF=바이트 동일. [[70]] 짝 A/B 의무.
export T2_SEARCH_REARM=1
# ★T2_DIAG_UNAMBIGUOUS (2026-08-26 등재·격리 `x554_diag_mispick_iso.py`·래칫 `test_diag_unambiguous.py`):
#   진단 서브의 답은 **이름** 단위인데 원장은 **행** 단위다. 한 이름이 여러 상태를 동시에 이면
#   *"어느 record 가 미지급인가"* 가 그 문맥에서 하나로 정해지지 않는데도 구판은
#   *"A separate check was run … It answers: X"* 로 단언한다 — 016 이 그 자리다.
#   ⇒ 답한 이름이 상태를 **하나만** 이고 있을 때만 단언한다. 여럿이면 침묵(묻지도 않는다).
#   격리(x554·6팔×5=30셀·8140): A_asis 가 라이브 오지목을 **축자 재현**하고, 문서가 적은 처방 둘
#   (`date_field` 축·거짓 유일성 교정)과 재료 조인까지 **전부 0/5** — 이 자리는 더 줘서 못 산다.
#   [[70]] 부호표(영속 로그 34 태그·08-13~08-26·발화 119): 016 22 제거(**전부 reward 0** ⇒ 손실
#   불가) · 010 12 불변(reward 1 이 5) · 098 73·099 12 는 이미 `원장 밖 = 침묵`이라 불변 ⇒ 순지배.
#   ⚠**§3 형(허위 문면 교정)으로 등재한다** — 새 거동을 더하는 것이 아니라 근거 없는 단언 하나를
#     거두는 것이고, `status_breakdown`·`status_meanings_text` 는 이 호출 **이전에** 그대로 나간다.
#   ⚠반증: 이름당 상태 1종인데 판정문이 틀린 sim 이 나오면 술어가 부족하다. OFF=바이트 동일.
#   ⛔**2026-08-27 저녁 — 끈다. 물음을 고쳤기 때문이다([[76]] 서브가 틀리면 서브를 고친다).**
#   이 침묵은 *"어느 이름도 단일 상태가 아니다"* 를 근거로 삼았는데, 그 모호는 원장의 성질이
#   아니라 **우리 물음**이 만든 것이었다. 구판 `diagnose_prompt` 의 *"One of these records did
#   not pay out"* 은 같은 블록에 실려 나가는 정의와 겹친다 —
#   `COMPLETE — … met the criteria to get the referral bonus` 아래에서 *'못 받은 것'* 은
#   COMPLETE 로도 읽힌다. 격리 `x566`(8140·4팔×5×2태그):
#       구판 그대로            **Platinum 3/5**(라이브 오지목 재현)
#       행 단위로만 바꿈       **Platinum 5/5**  ⇒ 단위는 원인이 아니었다
#       정의의 어휘로 물음     **Silver 10/10** (두 태그 전부)   ← 새 선언이 이것이다
#       부정통제              무응답 5/5
#   범위: 이 코퍼스(7 태그)에서 원장이 있는 sim 은 **016 뿐(9개)** 이고 침묵은 그 9개 전부에서
#   발동했다 ⇒ 끄는 것의 폭발 반경도 016 뿐이고, 다른 태스크는 **바이트 동일**이다.
#   ⚠술어·래칫은 남긴다(`test_diag_unambiguous.py` 초록). 서브가 다시 틀리면 그때 되살린다.
export T2_DIAG_UNAMBIGUOUS=0
# ★T2_REARM_USER_ONLY = 문서의 **A-3′** (2026-08-26 등재·측정 `x553_rearm_role_split.py`·
#   래칫 `test_rearm_user_only.py`): 재무장 재수요를 **손님 발화 · 전 접두**로 본다.
#   문서 처방 A-3(user 로 한정만)은 발화 **67/84** 를 죽이고 통과 sim 발화를 10건(반증 견딘 것
#   8) 가져간다 — 순손실이다. 창까지 되돌린 A-3′ 는 kill **27/84**, 통과 sim 3, **반증 잔존 1**
#   이면서 표적 세 태스크(016 6/7 · 055 4/4 · 057 6/7)는 A-3 와 **완전히 동일**하다 ⇒ [[70]] 절충.
#   ⚠래칫이 잠그는 것은 효과가 아니라 **동치성** — 플래그 ON 이 x553 이 잰 팔과 발화마다 같다
#     (불일치 0/84). 다르면 우리가 잰 것과 다른 것을 켠 것이다([[76]]).
#   ⚠남는 위험 1건(098@t7348). 라이브 효과는 다음 런이 산다. OFF=바이트 동일.
export T2_REARM_USER_ONLY=1
# ★T2_READ_PER_ENTITY (2026-08-27 등재·격리 `x561_read_entity_demand_iso.py`·발화면
#   `x560_read_entity_gap_scan.py`·래칫 `test_read_per_entity.py`):
#   선행 read 요건의 충족 판정이 **도구 이름만** 봐서, 다른 주체로 돈 read 가 요건을 영구히
#   닫았다. 016 실측(t7363·t7356 두 세대): 계좌 read 는 손님 자신으로만 돌았고 손님이 묻는
#   **친구**로는 끝내 안 돌았다 — 원장 15행 중 어느 행이 그 친구 것인지 아는 유일한 경로인데.
#   ⇒ 충족을 **주체별로** 본다. 인자 키·값만 비교하고 값의 뜻은 모른다([[59]]·[[22]]).
#   격리(x561·8140·3팔×4): A_asis **0/4**(라이브 축자 재현 — 거래 read 로 샌다) ·
#   B_demand **4/4**(`get_all_user_accounts_by_user_id_3847{user_id: friend_…}`) ·
#   N_len **0/4** ⇒ 길이 아님([[57]]).
#   [[70]] 부호표(t7363·t7356 두 세대·채점 33 sim): 발화 **7(21%)** — 016 2 · 072 1 · 074 1 ·
#   085 3 이고 **전부 reward 0** ⇒ 이 코퍼스에서 손실 불가. 판 것 = 발화 sim 당 read 한 턴.
#   ⚠경계(래칫이 잠근다): 술어는 **모델이 그 주체를 인자에 넣은 뒤에만** 선다 — NL 은 안 읽는다.
#     016 다섯 sim 중 둘이 그 형상이고 나머지 셋에서는 침묵한다. 손님이 말한 직후(msg[38])가
#     아니라 모델이 그 값을 쓴 뒤(msg[41])가 발화점이다.
#   ⚠반증: 발화한 자리에서 요구한 read 가 **이미 그 주체로 돌았던** sim 이 나오면 술어가 틀렸다.
#     OFF=바이트 동일.
#   ⛔**2026-08-27 저녁 — 끈다.** 격리(x561 4/4·x562 B_live 4/4)가 잰 것은 **순종**이지 매수가
#   아니었다. t7364(016 nt4)는 **0/4** 였고 궤적이 이유를 말한다: 이 술어가 주체라고 부른 값이
#   **그 read 의 주체가 아니다.**
#     016 `friend_user_5839` — env 의 users·referrals·credit_card_transaction_history **전부 빈 결과**
#     074·072 `Dark Green Account`·`Bluest Account` — 계좌 **이름**(그 read 는 `chk_ar72c5d8e3_1` 류 id 를 받는다)
#     085 `f7d3a82c91` — **user id**(Jordan Williams)가 `account_id` 자리에 들어간 것
#   ⇒ 켜면 *읽을 수 없는 것을 읽으라*고 요구한다 = 턴만 태운다([[70]] 매도).
#   ★그런데 **진단으로는 정확하다** — 네 자리 전부 §12 가 말한 *"배달된 값을 엉뚱한 엔티티에
#     묶는다"* 축이다(016 문서→Bronze · 074 금액←이웃 계좌 · 085 user id←account_id).
#     `x560` 을 그 **탐지기**로 남긴다: 발화 7/33 · 전부 reward 0.
#   코드·래칫은 남긴다(OFF=바이트 동일). 되살리려면 **주체 판정**부터 고쳐야 한다 —
#   같은 이름의 인자에 들어간 값이라는 것만으로는 그 read 의 주체가 아니다.
export T2_READ_PER_ENTITY=0
# ★T2_ARG_LABEL (2026-08-27 등재·계수 `x564_arg_producer_census.py`·격리 `x565_wrong_account_id_iso.py`
#   ·래칫 `test_arg_label.py`):
#   `_provenance_deny` 의 술어는 `_ctx_has` — *"이 문자열이 문맥 어딘가에 있나"* 라서 **출처는
#   맞고 종류가 틀린** 값이 전부 통과한다. env 는 레코드를 `필드: 값` 으로 찍으므로 종류의 답이
#   이미 문맥에 있는데 우리가 안 봤다. 085 축자: `user_id: f7d3a82c91` 이 나온 뒤 모델이 그 값을
#   `account_id` 로 넘긴다(계좌 목록이 오기 **전**·msg[24]) — 그 뒤 열 호출은 전부 옳다.
#   ⇒ 결손은 *"옳은 값을 못 고른다"* 가 아니라 **없는 값을 이웃 필드에서 빌린다** 이다.
#   계수(채점 37 sim·식별자 인자 720): 제 이름표 72% · **다른 이름표 8%** · 이름표 없음 18% ·
#   부재 2%. 잡음 둘은 선언으로 걷는다 — 덤프 머리 `ID`(079 18건) · 같은 축 동의어
#   `phone`/`phone_number`(040 17건·생산자 목록 동일성으로 판정).
#   [[70]] 부호표: 040·057·074·079·085 **다섯 태스크 12 sim · reward 1.0 인 것 0** ⇒ 손실 불가.
#   격리(x565·3팔): A_asis **4/16**(라이브 재현) · B_say **16/16**(전부 생산자 read 호출) ·
#   N_len **4/16** ⇒ 길이 아님([[57]]).
#   ⚠엔진은 **어느 값을 쓰라고 말하지 않는다** — 이름표 사실과 생산자 이름만 낸다([[62]]③④).
#   ⚠이름표는 **레코드 덤프에서만** 읽는다(`Record ID:` 표지). 스키마 줄을 먹으면 `string` 이
#     그 인자의 옳은 값이 된다(x565 배선 확인이 잡았다).
#   ⚠반증: 반려한 값이 그 인자로 **정당했던** sim 이 나오면 술어가 틀렸다. OFF=바이트 동일.
export T2_ARG_LABEL=1
# ★T2_CARD_DOCS (2026-08-27 등재·사용자 지시·격리 `x574_subject_docs_subagent_iso.py`):
#   진단 서브가 **주어를 정하면**, A3 `policy_ontology.doc_index` 가 그 주어에 대해 선언한
#   문서만 격리 서브에게 주고 그 **답만** 메인에 올린다([[71]]·[[65]]).
#   사용자 축자: *"격리 서브에이전트는 자신에게 관계된 문서만 받고 그것만 읽고 결정해야 한다.
#   그러기 위해서 A3 에 관련 문서들을 index 로 정의한 거다."*
#   격리(x574·8140·3팔×5): 그 주어 문서만 **5/5**(*"spend at least $750 within 60 days"* +
#   문서 id 인용) · 다른 주어 문서를 섞어도 **5/5**(섞임은 원인이 아니다) · 문서 없이 물으면
#   **0/5**($300 과 가짜 문서 id 를 만든다) ⇒ 병은 혼동이 아니라 **부재**다.
#   ⚠엔진은 색인을 **읽기만** 한다 — 검색·유사도·선별 0([[59]]). 답이 우리가 준 문서 id 를
#     인용하지 않으면 **침묵**한다([[22]] 근거-우선·[[25]]).
#   ⚠반증: 인용은 했는데 그 문서에 없는 수를 대는 답이 나오면 술어가 부족하다. OFF=바이트 동일.
#   ★★라이브 판정(t7367·sha 8d6cb581·016 nt4): **2/4 통과** — 이 태스크의 첫 pass.
#     대조 t7365(같은 태스크·이 레버만 없음) **0/4**. 통과 두 sim 은 gap 0 이고
#     `submit_transaction{amount: 750, credit_card_type: "Silver Rewards Card"}` 축자.
#     네 sim 전부 배달됐고 네 sim 전부 에이전트가 750 을 말했다(16·8·5·2회).
#     남은 둘은 **손님이 찍는 자리** — 하나는 먼저 $225 를 찍었고 하나는 안 찍었다.
#   ★★부호표 확정(t7368·hard-0 10×nt2·대조 t7363): 016 **0/2 → 1/2** · 나머지 아홉
#     **전부 불변** · **1→0 없음** ⇒ 순증. 배달은 016 에서만 18회 — 이 술어는
#     `ledger_metrics` 스펙 종속이라 폭발 반경이 구조적으로 그 태스크에 갇힌다.
#     016 재현: t7367 2/4 · t7368 1/2 (대조 0).
export T2_CARD_DOCS=1
# ★T2_PROMPT_DUMP (2026-08-27 등재·**계기**이지 레버가 아니다·기본 OFF):
#   모델이 실제로 본 메시지 전체를 사이드카에 남긴다(`kind=prompt`·`channel=gen`).
#   왜: 오늘 두 번 라이브 실패가 같은 접두 위 격리에서 **재현되지 않았다**
#   (`x562` B_live 4/4 · `x571` A_asis 가 옳게 답함). 라이브 프롬프트는 영속 궤적에 없다 —
#   커밋된 메시지 + **비커밋 `work` 주입** + 뷰 압축이 합쳐진 것이 실제 프롬프트이고
#   그 합은 어디에도 안 남는다. 그것 없이는 [[78]] 의 *"두 프롬프트를 찍어 diff"* 가 불가능하다.
#   ⚠턴당 30~40k자다. `T2_PROMPT_DUMP_TASKS`(부분일치·쉼표)와 `T2_PROMPT_DUMP_MAX`(기본 60000)
#     로 좁혀서만 켜라. 상시 ON 금지.
export T2_PROMPT_DUMP=0
# ★T2_FB_SIDECAR_TEXT_MAX (2026-08-27·계기 노브·기본 4000 = 종전 거동):
#   사이드카가 `text` 를 자르는 상한. 프롬프트 회수는 한 건이 25~50k자라 4000 에서 자르면
#   **시스템 메시지만** 남는다. ⚠`len` 필드는 **자르기 전 원본 길이**다 — 그 값으로
#   상한이 열렸는지 판정하면 안 된다(오늘 내가 그렇게 오독했다). 저장분은 `text` 로 재라.
export T2_FB_SIDECAR_TEXT_MAX=4000
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
# ★T2_ACTION_DENY_CAP 제거(2026-08-07·사용자 지시). 미설정=무제한.
#   이 cap이 101에서 turn 6 이후 우리 층을 영구히 침묵시켰고(원장 조회 2/20), cap이 막아 준 것은
#   측정된 적이 없다. over-action 상한이 필요하면 전역 T2_REGEN_BUDGET에 둔다(023 사고의 실제 층).
export T2_FORCE_ACTION=1 T2_ACTION_PROGRESS_REFUND=1
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
# ★발사 전 키 확인 (2026-08-08·C326). 위 §리더보드 정합 주석은 *"alltools는 OPENAI_API_KEY
#   필요 — 발사 전 source 하라"* 고 **주석으로만** 말했고, 드라이버 `run_stage1b_20260808.sh`는
#   이 파일을 source 하면서 키는 안 읽었다. 결과: 유료 2 sim이 **dense KB가 죽은 채로** 돌았고
#   (두 sim 모두 첫 호출이 `Missing credentials`), 고장난 도구가 스키마에 실린 것을 **런이 끝난
#   뒤 포렌식으로** 알았다. 사이드카와 똑같은 형태의 사고라 똑같이 **모든 라이브 런이 통과하는
#   한 자리**로 옮긴다([[07]] hard-constraint — 드라이버 기억에 맡기지 않는다).
#   dense 없이 가는 것도 정당한 선택이지만 그건 **명시된 결정**이어야 한다(`GO_RETRIEVAL=bm25`).
#   ⚠키는 절대 커밋하지 않는다 — 경로만 안다([[30]]·2026-06-16 유출 사고).
#   별도 함수인 이유: 런을 띄우지 않고 이 가드만 검정할 수 있어야 한다(유료 런으로 검정 금지).
t2_require_key() {
  [ "${GO_RETRIEVAL:-alltools}" = "alltools" ] || return 0
  [ -n "$OPENAI_API_KEY" ] && return 0
  local KP="${GO_OPENAI_KEY:-/home/woori/.openai_key}"
  [ -f "$KP" ] && . "$KP"
  if [ -z "$OPENAI_API_KEY" ]; then
    echo "[t2_launch] REFUSING: retrieval_config=alltools needs OPENAI_API_KEY (dense KB)." >&2
    echo "  키 파일이 없거나 export 하지 않는다: $KP" >&2
    echo "  dense 없이 갈 거면 그 결정을 명시하라: GO_RETRIEVAL=bm25 t2_launch ..." >&2
    return 1
  fi
  echo "[t2_launch] OPENAI_API_KEY loaded for alltools (dense KB live)"
  return 0
}

t2_launch() {
  local TAG="$1" PORT="$2" TASKS="$3" NT="$4"; shift 4
  # ★사이드카 기본 ON (2026-08-06 사고 재발 방지). 전수 런이 7시간 동안 **우리 층이 무엇을 언제
  #   말했는지 기록하지 않은 채** 돌았다 — 스모크 드라이버는 켜고 전수 드라이버는 안 켰고, 둘이
  #   같아야 한다고 말하는 코드가 없었다. 비커밋 관측이라 거동 변화 0이고(파일에만 쓴다), 없으면
  #   포렌식의 절반이 원리적으로 불가능하다. 드라이버가 기억해서 켜는 방식이 사고의 뿌리이므로
  #   **모든 라이브 런이 통과하는 이 한 자리**에 기본값을 둔다([[07]] hard-constraint).
  #   드라이버가 이미 지정했으면 그대로 쓴다(`:=`).
  : "${T2_FB_SIDECAR:=/home/woori/scratch/logs/fb_${TAG}.jsonl}"
  : "${T2_FB_SIDECAR_TEXT:=1}"
  export T2_FB_SIDECAR T2_FB_SIDECAR_TEXT
  # ★기구 발화 추적 기본 ON (2026-08-09·사용자 지시 *"어느 기구가 켜졌는지 확인할 수 있게
  #   하라"*). 사이드카는 **우리가 보낸 문장**을 남기고, 이것은 **어느 기구가 말했는가**를
  #   남긴다 — 둘은 다르다. 지금까지 후자는 런마다 다른 grep 이었고 이 세션에서만 두 번
  #   틀렸다(로그에서 셌는데 그 문구는 사이드카 채널이었다·C369 재발). 비커밋 관측이라
  #   거동 변화 0 이고, 드라이버 기억에 맡기면 또 빠지므로 **모든 라이브 런이 지나는 이
  #   한 자리**에 둔다([[07]]·사이드카와 같은 이유). 읽기 = `x196_run_trace.py <tag>`.
  : "${T2_TRACE:=/home/woori/scratch/logs/trace_${TAG}.jsonl}"
  export T2_TRACE
  t2_require_key || return 1
  cd "$GO_TAU2" || return 1
  /home/woori/venvs/seka_env/bin/python -u "$GO_REPO/scripts/distill/tau2/t2_run_gated.py" \
    --domain banking_knowledge --retrieval_config "${GO_RETRIEVAL:-alltools}" \
    --gate 1 \
    --agent_model Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8 \
    --agent_base "http://localhost:${PORT}/v1" \
    --user_llm openrouter/openai/gpt-5.2 --user_temp 0.0 \
    --user_reasoning_effort "${GO_USER_EFFORT:-low}" \
    --task_ids "$TASKS" --num_trials "$NT" --max_concurrency "${GO_CONCURRENCY:-4}" \
    --max_steps "${GO_MAX_STEPS:-200}" \
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
export T2_SG_RESULT_RANGE=1       # A8/OL-11 결과 범위 게이트(A2 `result_range` 선언 도구만·기본 ON·[[60]])
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
export T2_DISCOVERY_NAMES=1     # C11b(032): 발견 문구에 **이미 회수한 문서가 이름을 말한** 미호출 도구를 병기
export T2_VERDICT_SURFACE=1     # (2) 판정 실재+결정 도구 미호출이면 판정을 인용하고 선택은 남긴다
export T2_PHASE_OWNER=1         # C17(050): 선언된 auth 게이트 미충족 구간에선 행동-유도 레버가 침묵(단계 소유권)
export T2_SPEAK_PROHIBIT=1      # E3-②(022): push 레버가 **돌고 있는 절차가 금지한 도구**를 권하지 않는다
                                #   (표적 3발 침묵·over-block 0을 오프라인 전수로 사전 확정 — x104 §C)
export T2_PIN_READ_STEPS=1      # C15(사용자 지시): 지목한 잔여 **read**를 named tool_choice+단일값 enum으로 고정(write 제외)
export T2_PROC_PIN_REARM=1      # C15 보조: 첫 라이브라 재무장 1회만(기회비용 미측정이라 보수적으로)
export T2_TRANSFER_LEAVES_STEPS=1 # C16(048): 이관 시도 순간에 미완 절차 단계를 이름으로 표면화
export T2_UNCALLED_UNLOCK=1     # C12(053): 해제해 놓고 부르지 않은 도구를 사임 턴에 1회 표면화
export T2_PROC_ABSENT=1
export T2_PROC_ABSENT_K=3         # 무호출 연속 assistant 턴 임계(x86 K-sweep 2/3/5 전부 write 0)
# ⚠T2_PROC_ABSENT_CAP 은 2026-08-18 에 **없앴다**: 예산을 배당하지 않고 *같은 말을 두 번
#   하지 않는다*로 바꿨다(사용자 지시). 총량 상한은 t7315 050 에서 아무것도 강제할 수 없는
#   구간에 소진돼, 정작 두 조회가 열린 뒤엔 침묵하게 만든 원인이었다. 선언을 남겨 두면
#   읽는 곳 없는 노브가 되므로 지운다.
export T2_PROCEDURE_CAP=6         # sim당 deny 상한(불응 무한루프 방지·기존 cap 선례)
# ★A-1 재생성-경로 절차 재평가 (2026-08-23·`tasks__20260822/TASK_050.md` §7-①·축 E).
#   `_ap_regen` 이 낸 호출은 gate·UNLOCK_NAME·UNLOCK_PROV 만 다시 받고 절차 게이트를 **평가조차
#   받지 않은 채** 커밋됐다. t7346 050 trial 0 이 그렇게 승인을 먼저 커밋해 요청-제출을 빠뜨렸고
#   (DB 해시 갈림·reward 0.0), 같은 sha 의 trial 1 은 동일 호출이 원본 am 에 있었기에 **축자 동일한**
#   deny 를 받고 1.0 을 받았다. 새 결정론 0 — 기존 `t2_procedure.decide` 재호출뿐([[62]]).
#   ⚠[[70]] 판다: 사임-경로 regen 이 접히면 그 턴이 빈손으로 끝날 수 있다(over-action↓/no-action↑).
#   부정통제 4칸([[57]]) = 이 플래그 1↔0 × `T2_PROCEDURE` 1↔0 · 계수 = `[T2_PROCEDURE] regen-*`.
export T2_PROC_REGEN=1
export T2_UNINSTRUCTABLE=1        # 실행 불가 지시 차단(012): 손님에게 도구 실행을 안내했는데 전달 이력 0.
#                                 #   술어=A2 L1 선언 토큰 포함 ∧ 전달 마커 부재(정규식 추출 0·C279 계보).
#                                 #   사전 계량(x82·194 sim): 발화 43 sim(1회/sim 캡)·그 중 17은 나중에
#                                 #   전달이 실제로 일어남(넛지가 이르지만 문구가 "먼저 전달하라"라 무해).
export T2_CHOICE_GROUND=1        # 계좌 클래스 등 열린-문자열 선택의 접지 넛지(x84: 이득 7·gold 미접지 3이라 deny 금지)

# ★F5 전사 대조 (2026-08-05·`t2_transcribe.py`·A2 `transcription_check`·게이트=x90):
#   행 배열 인자의 손-전사 값이 그 대화가 읽은 원장(record dump)과 어긋나면 deny.
#   018 t0: rewards_earned 1113(원장 487) → 없는 불일치 → 여분 분쟁 1건 → db_match=False.
#   x90 전수(194 sim): 발화 3건/2 sim · **gold 자신이 걸린 횟수 0**(오차단 0).
#   엔진은 값을 고치지 않는다 — 어긋난 사실만 말하고 재발행은 모델이 한다([[10]]).
export T2_TRANSCRIBE=1
export T2_TRANSCRIBE_CAP=4        # sim당 상한(기존 cap 규약)

# ★G2-a 프로토콜 문서 미열람 표면화 (2026-08-05·게이트=x93·`OPEN_PREDICATE_DECOMPOSITION` §1):
#   x93 전수(194 sim): 미열람 사용 27건 / **gold이 요구한 이관인데 미열람 6건** ⇒ deny하면 정답을
#   막는다 ⇒ **표면화만·sim당 1회**. 어느 프로토콜이 맞는가(열린 술어)는 말하지 않는다.
export T2_REQUIRE_DOC=1
# ★T2_REQUIRE_DOC_DELIVER (2026-08-22 등재·정본 `T7336_FORENSIC_033_2026_08_22.md`·격리 C592 x465):
#   위와 **같은 닫힌 술어**(선언 도구 시도 ∧ 정의 문서 미열람)에서 표면화 대신 **정의 문서 전문을
#   그 턴의 재생성 버퍼에 싣는다**(헤더 = x465 축자·지시-앞·코퍼스 도출 전부·선택 0·deny 0 =
#   x93 gold-이관 6건 보호). 격리(n=7/팔): 일반 7/7 → 사슬 6/7 · 부정통제(무내용 재촉) 0/7 ⇒
#   원인은 미전달·레버는 전달뿐([[62]]②). 판 것 = 문맥 +16k자/회·sim당 CAP 회. [[70]] 짝 A/B 의무.
#   배달이 나간 턴엔 위 표면화(REQUIRE_DOC)가 비워진다(같은 턴 "검색하라"↔"여기 있다" 모순 방지).
export T2_REQUIRE_DOC_DELIVER=1
export T2_REQUIRE_DOC_DELIVER_CAP=3       # sim당 배달 상한(검색 에이전트 예산 3 선례·재료는 한 턴만 산다)
export T2_REQUIRE_DOC_DELIVER_MAX=90000   # 배달 상한 자수(x465 --maxchars 동일·절단은 표시)

# ★T2_CP2_QUEUE (2026-08-23·기본 **0**·계기 수리·정본 원장 C502 의 S1-(1)):
#   배달 채널 `_t2_cp2_pending` 은 **슬롯 하나**다. 같은 턴에 두 배달이 오면 뒤엣것이 앞엣것을
#   **버린다**. C502(t7303)가 그것으로 전달의 1/3 을 잃어 인과 실험 자체를 무효로 만들었고,
#   그때 심은 `[T2_CP2_CLOBBER]` 가 이번에 t7346 에서 다시 잡았다 —
#     098#s626729 `SEARCH_ON_PROCEED 가 미소비 배달물 243자를 버리고 247자로 덮어씀`
#   (같은 런 057 ×2 · 063 ×2 · 셋 다 0/2 · t7336 의 같은 태스크는 CLOBBER 0건 · 2/2 통과)
#   구판 anti-clobber 는 **≥10k자만** 구제해 243자는 못 잡는다. 켜면 크기 무관 이어붙임.
#   ⚠기본 0 인 이유 = 소형↔소형까지 바뀌어 ctl 바이트가 달라진다. **켜기 전에 손해도 재라**([[70]]).
export T2_CP2_QUEUE=0
# ★T2_CP2_KEEP_SURE (2026-08-24·기본 **0**·`_qcross` 의 거울): 확실히 배달될 소형 미소비 배달물이
#   가드 검사 대상(≥5,000자) 대형에게 밀려 죽는 것을 막는다. 실물 057#s373753 = 247→247→87,407자가
#   clobber·clobber·ctx_skip 으로 **셋 다 소실**. t7348 전량 clobber 10 · ctx_skip 1.
#   ctl 바이트가 달라지므로 측정 전에는 켜지 않는다([[70]]).
export T2_CP2_KEEP_SURE=0
export T2_CP2_APPEND_MAX=90000            # 이어붙임 총량 상한(초과 시 종전대로 덮어쓰되 로그로 남긴다)

# ── ★死배선 2건 등재 (2026-08-07·`t2_levers.py --audit_declared`가 검출) ───────
#   둘 다 **구현·배선을 마치고 오프라인 자기검사까지 통과했는데 이 파일에만 없어서 죽어 있었다**.
#   [[24]] 死코드 패턴의 3·4번째 재발이다(앞선 둘 = QUOTE_PIN·MATCH_COUNT).
#   레지스트리에 셀→라이브 **역방향** 감사를 넣자 바로 나왔다 — 정방향(미분류)만 보면 안 보인다.
#   ⚠의존물 확인 완료(MATCH_COUNT가 가르친 절차): A2 `ledger_metrics`가 정본·gate 양층에 실재.
export T2_LEDGER=1                # C5 이관 — 원장 산수(전사=모델·산수=엔진). `t2_ledger.py` 자기검정 OK.
#                                 #   표적: 원장 29행 받고 창_잔여 미산출 **19/22 trial**.
#                                 #   ⚠핸드오프 §7-3이 지목한 미검증분 — 첫 런에서 `[T2_LEDGER] probe:`
#                                 #   한 줄이 `flag='1'`로 바뀌는지부터 본다(구판은 flag=None을 찍었다).
# ── ★미등재 4건 해소 (2026-08-07·x127 감사 + 20260807b~e 실측) ──────────────
#   x127: 코드가 읽는 T2_* **236종** vs 런처가 주는 **130종** → 없는 것 109종.
#   그중 셀에 배치된 **레버**는 `T2_SELF_DECLARATION` 하나뿐이었지만, **메타 플래그가 꺼져
#   레버를 가두는** 형태가 따로 있었다 — 그게 아래 `T2_ARBITRATE`다.
export T2_ARBITRATE=1             # C3 중재(합병·등급). ★이걸 켜야 `T2_SOURCE`에 **도달**한다 —
#                                 #   `t2_gate_patch:5614`가 SOURCE 블록을 통째로 감싸고 있다.
#                                 #   같은 중첩이 `_reqs` UnboundLocal도 만들었다(오늘 별도 픽스).
export T2_SELF_DECLARATION=1      # 답변 근거를 스스로 선언시키고 `INFER`인데 도구가 있으면 되돌린다.
#                                 #   구현 완료·`:7844` 배선 완료인데 런처에만 없었다.
export T2_SURFACE_BUS=1           # 부착의 단일 출구(replay·정직·예산·채널 4불변식).
#                                 #   ⚠OFF가 더 위험했다: 구판 주석 축자 — 가드가 "`T2_SURFACE_BUS=1`
#                                 #   일 때만" 걸렸고 라이브(버스 OFF)는 **무가드 직접 부착**이었다.
export T2_WINDOW=1                # C6 발화 창 — resign ∪ acting ∪ instructing(`:5542`).
#                                 #   ⚠**창을 넓히는 변경**이다. 발화 기회가 늘므로 Δspurious 필수 계측.
export T2_SUPPRESS_AUTH=1         # 억제 자격 집행. 근거는 A2 **L3** `suppression_authority`
#                                 #   (banking_knowledge.specific.json — phase_owner "실패 런 9회
#                                 #   침묵·통과 런 0회" + speak_prohibit "표적 3발·over-block 0").
#                                 #   ⚠2026-08-08 L1→L3: 근거가 banking 런에서 잰 수치라 L1에 두면
#                                 #   **재본 적 없는 도메인의 억제까지 licensing** 한다. banking 거동 0.
#                                 #   미선언 레버는 **침묵시킬 수 없다**(표면화는 그대로).
#   ⚠이 5줄은 **거동을 바꾼다**. 한 번에 켜는 이유는 [[19]] 합성-우선(간섭은 합성 런에서만 드러난다)
#   이고, 대신 스모크에서 **Δspurious·발화 폭증**을 1차 지표로 본다.

export T2_SOURCE=1                # C1 출처 계약 — 주장을 형식화(LLM)하고 **출처만** 검증(엔진).
#                                 #   `t2_source.py` 자기검정 OK · 호출부 `t2_gate_patch:5639`.
#                                 #   출력은 C3 합병 경로로 합류한다(따로 내보내면 T4b 슬롯 경합 재생산).
#                                 #   1차 지표 = `[T2_SOURCE] claims=N unsourced=M` 발화 여부.

# ★G3 확정-행 미제출 표면화 (2026-08-05·게이트=x94·[[21]]의 닫힌 절반):
#   019 t1은 엔진이 확정한 3행 중 하나를 손님 산문에 설득당해 철회했다. 손님 문장은 읽지 않고
#   엔진 출력과 호출 이력만 본다. ⚠**F5 위에서만 건전** — 오염 입력으로 나온 확정 행은 제외한다
#   (x94 1차 gold 반례 2건이 전부 그것이었고, 제외 후 반례 0). 제출 강제 아님·표면화만.
export T2_WITHDRAWN_ROW=1

# ★VC **호출-트리거** (2026-08-18·C543ⓓ·설계 `VERDICT_CALL_TRIGGER_DESIGN_2026_08_18.md`).
#   push 형(`T2_VERDICT_CARRY`)은 결정점에 닿기만 하면 발화해 **고를 것이 없는 073 에서 음수**였다
#   (ctl 1.0 ↔ vconly 0.0). 트리거를 A3 의 호출-관용구(`write_arg_enum`)로 옮겨, 후보를 먹는 호출을
#   **부를 때만** LLM 자신의 판정 줄로 되돌린다 — 비-선택 태스크엔 트리거 자체가 없다.
#   ⚠기본 OFF: push 형과 **동시에 켜지 않는다**(같은 판정을 두 번 사면 귀속이 섞인다). 팔 = run_one.sh `vgate`.
# ⚠아래 두 줄은 **선언**이다(값은 종전 기본과 같은 OFF) — 주석에만 이름이 있으면 래칫이
#   '선언됐다'고 세는데 실제 스택엔 없다. 스택은 정의되어야 비교가 성립한다.
export T2_VERDICT_CARRY=0         # push 형(결정점 선적재) — 073 에서 음수(C543ⓐ)
export T2_VERDICT_GATE=0
export T2_VERDICT_GATE_CAP=1      # sim당 거부 상한(livelock 금지·052 전례)

# ★완료-주장 **격리 검증 서브** (2026-08-18·C544 예정·사용자 지시 축자: *"LLM 격리로 env 정책과
#   실행한 도구, 현재 실행했다고 주장하는 도구를 참 거짓으로 판단하게 별도의 검증 에이전트"*).
#   자리는 종전 그대로 — `claim_audit.question` 이 초안이 나가기 전에 돌고, 엔진이 원장과 대조한다.
#   바뀌는 것은 **대조의 판단**뿐이다: 이름이 원장에 있다고 통과시키던 자리에서 격리 서브가
#   *"그 호출이 그 일을 할 수 있었나"* 를 답하고, 엔진은 참/거짓 토큰과 지목의 원장 실재만 본다.
#   근거 = t7318 073(조회 도구로 환급 주장이 구제되고 환급은 끝내 미실행·reward 0.0).
#   ⚠기본 OFF — 판정 효과는 A/B 로 잰다([[57]]).
export T2_CLAIM_VERIFY=0

# ★출처 집합에 **우리 층**을 포함 (`T2_PROV_OURS`·2026-08-18 선언·정본
#   `CONFLICT_ARBITRATION_THEORY_2026_08_06` §3-T3·구현은 2026-08-06 부터 있었고 **한 번도 안 켰다**).
#   근거(t7320 라이브 실측): 읽기 루틴이 `unlock_discoverable_agent_tool` 을 두 조회 이름으로
#   고정한 그 턴에서, 같은 층의 출처 가드가 그 이름을 **operator-fab(지어낸 이름)** 으로 막았다 —
#   우리 피드백은 `role=tool, error=True` 로 나가 출처 집합(성공한 tool-result)에서 구조적으로
#   빠지기 때문이다. 그 턴에 모델이 할 수 있는 행동이 0이 된다.
#   ⚠날조 통과 위험 0: `stated_names` 가 **레지스트리 교집합**을 걸어 실재하지 않는 이름은
#     애초에 담기지 않는다(집합 구성으로 보장).
export T2_PROV_OURS=1

# ★no-record 분기 v2 (`T2_NOREC_BRANCH`·2026-08-18 선언·근거 C536ⓒ·x35③).
#   A2 에 저작된 `no_record_template_v2`(639자)가 **한 번도 안 켜져** 라이브는 v1(253자)을 쓴다.
#   v1 은 *"…then call this tool again"* 으로 닫혀 **종료 분기가 없다** — 008 실측: 조회가 영구
#   `No records found` 인데 6-호출 사이클을 30회 돌아 문맥을 초과했다. v2 는 ⑴같은 인자 반복 금지
#   + 미사용 식별자 우선(x35③: 다른 인자 재조회 성공 **73/138** = 즉시-ASK 가 놓칠 회복 상한 52.9%)
#   ⑵못 찾으면 손님에게 다른 식별자 요청 ⑶줄 수 없으면 **종결** ⑷검증 전 기록 금지.
#   ⚠옛 런처(`run_axis32_chain.sh:52`)는 켰다 — go_stack 이 정본이 되며 조용히 빠진 계열이다.
export T2_NOREC_BRANCH=1

# ★접지 헤더의 모순 문장 제거 (`T2_GROUND_HDR`·2026-08-18 선언·근거 x35②·INSTRUCTION_DEFECT §2d′).
#   공통 헤더가 **전 필드**에 *"Re-read the exact value(s) from the records"* 를 말하는데,
#   `intent_fields`(출처=손님 발화)의 개별 주석은 *"손님이 말한 적 없다"* 로 **정반대 출처**를
#   가리킨다. 실측: ledger 파라미터 회복 **38:20** ↔ user 파라미터 회복 **7:43**.
#   ⇒ 헤더에서 그 한 문장을 뺀다(각 flag 의 괄호 주석이 이미 클래스별로 정확하다).
#   엔진 분기 순증 0 · A2 순증 0 · [[55]] *"문구 모순 상존"* 계열의 수리다.
export T2_GROUND_HDR=1

# ★빈-결과 문면 (`T2_RETURN_EMPTY`·2026-08-18 선언·D4·INSTRUCTION_DEFECT §2a).
#   판정 상세가 **공집합**인데 같은 문장이 *"these require a cash back dispute"* 를 단언하던 자리.
#   표적 실측(오늘·영속 14런): `get_reward_discrepancies` **71회 중 10회가 빈 결과**(14%) ·
#   다른 상세-도구는 0. 문면은 오늘 저작했고 **완결을 주장하지 않는다**(coverage 표면화에 위임).
#   술어 = `_sg_details` 공집합(닫힘·판단 0) · 선언 없는 도구는 종전 거동.
export T2_RETURN_EMPTY=1

# ★도구 명세 되붙이기 (`T2_SPEC_AT_WRITE`·기본 OFF·2026-08-25·격리 x532 후 배선).
#   왜: t7348 085 두 sim 궤적 축자 — `unlock_discoverable_agent_tool` 의 반환문이 파라미터
#   17개와 enum 4종을 전부 담아 **msg22** 에 도착하는데 첫 오답 write 는 **msg68 / msg80**
#   이다(거리 46·58). 그 사이 모델은 `debit_card_id`·`category`·`date_first_noticed` …
#   10개를 지어내며 13턴을 태운다 ⇒ 재료 부재가 아니라 **거리**다(x509 큐 공통 진단
#   *"재료는 상류에 있고 결정점에 없다"* 와 같은 모양).
#   격리 x532(n=6 창): A_asis 1/6 ↔ **B_spec 6/6** ↔ N_neg 2/5(같은 길이 무관 블록)
#   ⇒ 산 것은 길이가 아니라 내용([[57]]) · A_asis 가 라이브 오답 키를 재현하므로 공정([[62]]2b).
#   하는 일은 **전달 하나** — env 가 앞서 보낸 응답을 자르지도 고르지도 않고 되붙인다.
#   값 선택은 전부 모델 몫이고 술어에 도메인 낱말이 0 이다([[05]] 전이).
export T2_SPEC_AT_WRITE=0
export T2_SPEC_AT_WRITE_MIN=8     # 재료가 이만큼 뒤일 때만 — 바로 앞이면 되붙일 이유가 없다

# ★스키마 enum 조회 (`T2_SCHEMA_ENUM`·기본 OFF·2026-08-25 선언 누락분 소급).
#   `t2_role.enum_of` 의 소비부. 지금 발화처가 없다 — agent 가 보는 도구 17개 중 enum 을
#   선언한 인자가 하나뿐이고 표적 도구들은 discoverable 이라 그 목록에 없다(실측).
#   ⇒ OFF 유지. 켜기 전에 발화 검정이 선행돼야 한다([[24]] 死배선 금지).
export T2_SCHEMA_ENUM=0

# ★전사 프롬프트 조립 순서 (`T2_SG_PROMPT_V2`·2026-08-25 선언 누락분 소급·074).
#   x525 계열 8런·팔 15종 이등분: `=== REFERENCE ===` 를 JSON 블록으로 주면 행이 빠지고
#   (13~15/16) 평문이면 16/16 · `answer_format` 이 재료보다 앞이면 유령 `duplicate_of` +3,
#   뒤면 정확히 16 ⇒ 조립 순서 = instructions + params + 재료 + answer_format.
#   엔진이 쓰는 문장 0 — 선언의 텍스트를 그대로 쓰고 **순서와 렌더링**만 바꾼다.
#   ⚠커버리지는 닫혔으나 초과 행 1 이 남는다(rows 17/18 · 원인 미규명).
export T2_SG_PROMPT_V2=0

# ★선언된 인자 타입 (`T2_WRITE_ARG_TYPE`·기본 OFF·2026-08-25).
#   t7354 라이브 전수: 도구가 `(boolean)` 으로 선언한 인자에 모델이 문자열 `"Yes"`/`"No"` 를 보낸다
#   — 085 접수 분쟁 **전건** · 040 은 gold 8건을 축자 접수하고도 `db_match=false`(8/8 어긋남).
#   env 가 문자열을 받아 저장하므로 호출은 성공하고 채점만 조용히 실패한다.
#   ⚠엔진은 값을 바꾸지 않는다 — 선언된 타입만 알리고 모델이 다시 낸다([[62]]③④).
export T2_WRITE_ARG_TYPE=0

# ★선언된 절차 문장을 write 결정점에 (`T2_RULE_AT_WRITE`·기본 OFF·2026-08-25·격리 x537).
#   창 그대로 0/12 ↔ 문장 한 줄 12/12 ↔ 같은 길이 무관 문장 0/12. 검색·순위 0(선언 읽기).
export T2_RULE_AT_WRITE=0

# ★자리표시자로 채운 인자 (`T2_WRITE_ARG_FAB`·기본 OFF·2026-08-25).
#   술어 셋 전부 **선언이거나 값의 모양**이고 이름 패턴 0: env 가 string 이라 선언 + 열거 아님
#   + 자리표시자 모양(연속·동일 자릿수 4) + 문맥 부재. t7354 6배치 전수 실측 20건 전부 진짜
#   날조(`card_last_4_digits='1234'` 12 · `transaction_id='TRXN123456789x'` 8)·오차단 0.
#   ⚠이름 패턴판(`identifying_arg_types.digit`)은 2026-08-25 사용자 지적으로 **철회**했다.
export T2_WRITE_ARG_FAB=0

# ★서브에게 주는 레코드 덤프의 순서 (`T2_SG_RECORD_ORDER`·기본 OFF·2026-08-25·격리 x536/x539).
#   x536(4계좌×6팔×3=72샘플): 같은 6,752자 원문을 순서만 바꾸면 갈린다 —
#   N_wire 17/17/17(기대 16) ↔ D_old_group 16/16/16 ↔ **N_scramble(무의미 순서)은 두 계좌를 부순다**
#   ⇒ 산 것은 재렌더링이 아니라 순서의 내용([[57]] 통과). 엔진이 쓰는 문장 0·값 0·판단 0.
export T2_SG_RECORD_ORDER=0

# ★전사 행 수 검산 (`T2_SG_ROW_COUNT`·기본 OFF·2026-08-28·닫힌 술어·[[22]]·[[25]]).
#   술어 = 서브가 넘긴 배열 길이 < 원천에서 센 **선언된 종류**(`isolate.row_kind`)의 레코드 수.
#   그러면 총액을 단언하지 않고 선언된 `return_template_short` 로 나간다(재공급 지시 포함).
#   실측(t7368 `task_072#s626729`): Bluest 32 레코드 중 `type: atm_withdrawal` 9 → 서브 9 →
#   delta_total **14.0 = gold** · Light Green 26 중 **10** → 서브 **9** → delta_total **5.0 ≠ 3.5**.
#   빠진 한 행이 `btxn_8c58b19a3628 (charged $0.00, documented fee $1.50, difference $-1.50)`
#   = 수수료 줄 없는 인출인데, 반환문은 `[coverage] 9 of 9 rows were checked (0 could not be
#   verified)` 였다 — **분모가 넘어온 행 수라 자기 자신을 잰다**. 그 결과 우리 층이 틀린 총액을
#   *"use it as the credit amount"* 라는 권위 문면으로 건넸다.
#   ★왜 프롬프트 수리가 아니라 검산인가: 같은 결손을 `T2_SG_PROMPT_V2` 가 프롬프트 **모양**으로
#     고쳤는데 074 chk_2 를 13~15/16 → 16/16 으로 사고 072 Light Green 을 10/10 → 9/10 으로
#     팔았다(t7348 ↔ t7363·t7368). 섭동은 태스크마다 부호가 갈리고([[07]]) 손실이 조용하다.
#     닫힌 검산은 못 본 태스크에서도 참이다 — **표본이 아니라 증명**이다.
#   ⚠적게 넘긴 것만 본다(초과·중복은 다른 술어 몫·`_omitted_rows_note` 가 걸린 함정 회피).
#   ⚠종류 미선언·원천 0건이면 판정하지 않는다. ⚠라이브 효과 미측정 — A/B 가 잰다.
export T2_SG_ROW_COUNT=0

# ★env 명세에서 도출한 타입·열거 (`T2_SPEC_ARG_FACTS`·기본 OFF·2026-08-25).
#   손 선언(write_arg_enum 값 6칸·booleans 2세트)을 **대체**한다. 등가성은 코퍼스로 쟀다 —
#   x540: 명세 블록 61 · 도구 16 · 대조 9건 전부 일치(다르다 0·대조 불가 0).
#   폭발 반경: 도출이 손 선언보다 새로 막는 것 **0건**(t7354 전 배치).
#   명세는 **도구별**로 읽는다 — card_action 은 신용 2값·직불 3값이라 이름만으로 합치면 오차단.
export T2_SPEC_ARG_FACTS=0

# ★이미 성공한 변이의 재실행을 지운다 (`T2_DUP_WRITE`·기본 OFF·2026-08-26).
#   근거: x546/x547 재생 — 중복을 전부 빼도 만점 sim 14/14 불변(비용 0)이고
#         0점 sim 142 중 8 이 1.0 으로 뒤집는다(074·073·050).
#         x548 격리 — 문면이 재발행을 4/4 → 0/4 로 막고, 이름 없는 거절·같은 길이
#         무관 문장은 못 막는다([[57]] 부정통제 통과).
#   ⛔stub 금지 — **재생성 채널로만** 나간다(2026-08-02 `failed_setstate` 사고).
export T2_DUP_WRITE=0

# ★선언 인자 ↔ 정책 행 **동일성 조인** (`T2_ARG_POLICY_AT_WRITE`·기본 OFF·2026-08-25).
#   write_rules 의 일반형: 손으로 고른 문장 대신 이 write 가 선언한 인자 이름과 A3 `axis` 가
#   같은 행을 축자로 결정점에 놓는다. 유사도 검색 아님(동일성)·순위 0·상한 넘으면 전부 안 준다.
#   조인 커버리지 실측: 신용 분쟁 인자 15 중 13 · 직불 17 중 9.
#   ★2026-08-26 ON — 격리 통과(`x551`, 040 `eligible_for_provisional_credit`):
#     A_asis **2/4** ↔ B_rule **4/4** ↔ N_len **2/4**(부정통제가 A 와 행별 답까지 동일).
#     창은 **전 접두**로 잡아 라이브 거리를 재현했다(짧은 창은 A_asis 3/4 로 결손을 지웠다).
#     같은 날 A3 의 그 축 행을 **완결**시켰다 — 종전 인용은 *"Guidelines 문서를 보고 판단하라"*
#     는 **포인터**였고(=A_asis 조건 그 자체), 이제 그 문서의 기준 5개가 축자로 실린다
#     (doc `..._015` · 827자 · [[72]] 선언은 완결 · [[23]] 출처는 정책 문서).
# ★`[OPERATOR-SCOPE]` 를 **실효 write 로 좁힌다** (2026-08-26·x550 §2·되돌리기 노브).
#   실측(최근 12런 사이드카): 발화 **61회** 중 **read 46(75%)** 이고, **61 중 49 는 그 도구가
#   끝내 실행됐다** — 반려가 선택을 바꾼 게 아니라 **턴만 태웠다**(079 26회·085 25회 집중).
#   읽기의 오선택은 회복 가능하고(다시 읽으면 된다) 쓰기의 오선택만 되돌릴 수 없다.
#   ⇒ 끄는 게 아니라 **조건부 발화**([[70]]) · 조건은 도메인 일반 닫힌 술어(`_is_effective_write`).
#   `=1` 로 두면 읽기까지 종전대로 발화한다(음성으로 판정되면 그렇게 되돌린다·[[60]]).
export T2_SCOPE_ALL=0

#   ⛔**2026-08-26 다시 OFF — t7361 이 그 경고를 실현했다.** 위 격리는 기준 **827자**만 실었는데
#     라이브 조인은 이 write 의 선언 인자 **15개 전부**를 실어 **3,033자**를 보낸다. 그리고 그
#     자리엔 이미 세 레버가 앉아 있었다. 040 의 같은 결정점을 두 런에서 재면:
#         t7360  DECIDE 285 + SPEC 2137 + RULE 74            = **2,496자** → turn 79 · 완료
#         t7361  DECIDE 6973 + SPEC 2137 + RULE 74 + AP 3033 = **12,217자(4.9x)**
#                → `toolerr` **같은 지문 26회** 반복 · turn 98 에서 중단(79분)
#     [[65]] 축자: *"재료를 메인에 올리는 것 자체가 부하다"*(x231 8/8→0/8). 내가 그 규칙을 어겼다.
#   ⚠증가분이 전부 이 레버는 아니다 — `DECIDE_BEFORE_WRITE` 재료가 285→6,973자로 24배 늘었고
#     그건 이 수리가 건드린 경로가 아니다(`src=search`) ⇒ **단독 귀속하지 않는다**.
#   ⚠**격리(x551)는 유효하다** — 규칙을 결정점에 놓으면 닫힌다(A_asis 2/4 ↔ B_rule 4/4 ↔
#     N_len 2/4). 잰 적 없는 것은 **조인된 형태의 부하**다. 그걸 재기 전에는 켜지 마라.
#   ⚠A3 그 축 행의 **완결은 유지한다**(기준 5개 축자) — 꺼져 있으면 무해하고 다시 켤 때 쓴다.
#     축 하나만 실으면 3,033 → **1,073자**다(어느 축을 고를지는 미측정 — 그래서 지금 안 한다).
#   래칫 = `test_decision_point_load.py`(결정점 합산 상한).
# ★끝내는 자리에서 **절차의 남은 칸을 이름으로** (`T2_PROCEDURE_LEFT`·기본 OFF·2026-08-26).
#   t7361 per-step 포렌식: 050·074·085 이 **같은 모양**으로 끝났고 우리 층은 매번 남은 것을
#   알고 있었다. 050 축자 — `[T2_PROCEDURE] checklist … done=5 left=['decision']` 직후
#   `[T2_CLAIMPROV] window hit(resign) claims=3 unbacked=0` → `regen tool_calls=[]`(통과).
#   기존 둘이 못 잡는 이유: WRITE_PROV 는 *완료 주장*을 전제하고(074 는 주장 없이 이관),
#   CLAIM_PROV 는 *"어떤 write 가 원장에 있나"* 를 본다(050 은 submit 이 있어 unbacked=0).
#   물음은 **절차의 남은 칸**이었고, 그 술어는 `t2_procedure.checklist` 에 이미 있다([[67]]).
#   ⚠엔진은 고르지 않는다 — 미충족 노드를 **전부** 인쇄한다([[63]] 빼기 · [[62]]④).
#   ⚠효과 미측정 — A/B 가 잰다(이 세션 지시). 격리 없이 기본 ON 금지.
# ★E-PLAN L1 이 **이미 부른 열거자를 뺀다** (`T2_EPLAN_ENUM_SUBTRACT`·기본 OFF·2026-08-26).
#   t7361 085 실물: 직불 분쟁인데 모델이 선언된 열거자 `get_credit_card_transactions_by_user`
#   를 성공 호출했고 env 가 정당하게 `No records found in 'credit_card_transaction_history'.`
#   를 돌려줬다. `listed` 는 *출력에서 id 가 뽑혔나* 라서 공집합이 됐고, L1 이 **같은 문면을
#   4회** 반복해 분쟁 write 를 막았다 → 모델은 인간 이관으로 나갔다(gold 4행 중 done 2).
#   이 파일 054 주석이 이미 이름 붙인 **충족 불가 술어**다 — 모델이 할 수 있는 일이 없다.
#   ⇒ *부른 적 있나* 와 *뭐가 나왔나* 를 가르고, 남은 열거자만 이름으로 말한다([[63]] 빼기).
#     선언된 열거자를 **전부** 불렀으면 L1 은 놓아주고 L2 로 흘려보낸다(다른 가드는 그대로).
#   ⚠효과 미측정 — A/B 가 잰다.
# ★근거 없는 손님-액션 지목을 침묵 (`T2_ACTIONREQ_GROUNDED`·기본 OFF·2026-08-26).
#   `TASK_072.md` §7-2 처방 P-A 의 구현(문서 상태: 미착수). 결함 = `formalize_intent_tool` 이
#   **이 대화에 한 번도 안 나온** 손님-측 도구를 지목하면 `[ACTION]` 이 *"'X' 는 손님이
#   실행한다"* 고 말한다 — 참이지만 무관하고, 072 t0 에서 강제-행동 경로를 통째로 죽였다.
#   같은 site 를 `x505_TASK_073_t7348_perstep.md` §2.1 이 독립 지목(단발 아님).
#   빈도 실측(최근 12런·태그별): `formalized_target` 383건 중 **29건(8%)** 이 궤적 축자 0회 ·
#   그중 **23건이 `submit_transaction`** · 태스크 040(8)·085(6)·074(5)·057(5)·063(4)·055(1)
#   ⇒ **hard-0 여섯**에 걸친다(문서 추정 둘보다 넓다).
#   술어 = 집합 소속 + 축자 대조뿐([[22]]·C45 동형) · 이름이 대화에 있으면 종전대로 발화.
#   ⚠효과 미측정 — 런이 잰다.
# ★requires_reads 를 **정본 입구**로 읽는다 (`T2_SG_REQREADS_CANON`·기본 OFF·2026-08-26).
#   `t2_precedence.declarations()` 는 자기를 *"소비자가 A2 키를 직접 읽지 않게 하는 단 하나의
#   입구"* 라 못박고 이유로 *"소비자가 각자 키를 읽으면 '정본만 고치고 소비자 미동기화' 버그"*
#   ([[24]] 2026-08-03 실측)를 든다. `t2_scaffold_get.py:2077` 이 정확히 그 우회였다.
#   실물(094·4 sim 0/4): A3 인덱스에 `get_interest_correction → [계좌목록, 거래이력]` 이
#   **정책 축자와 함께** 있는데(C586·`doc_bank_accounts_…_043`) 게이트는 도구 자신의 키만 봐
#   `None` 을 읽었고 거래 read 가 **0회**였다.
#   ⚠폭발 반경 실측 = **도구 1개** — 스캐폴드 10 중 `get_interest_correction` 만 불일치,
#     나머지 9 는 이미 같다(거동 불변).
#   ⚠[[23]]: 출처는 정책 축자다. `TASK_094` §7-P2 의 *"gold 가 정확히 이 read"* 논거는 안 쓴다.
#   ⚠[[70]] 파는 것: 계좌·거래 read 를 아직 안 한 궤적에서 계산이 한 턴 밀린다.
export T2_SG_REQREADS_CANON=0

export T2_ACTIONREQ_GROUNDED=0

export T2_EPLAN_ENUM_SUBTRACT=0

export T2_PROCEDURE_LEFT=0

export T2_ARG_POLICY_AT_WRITE=0
export T2_ARG_POLICY_CAP=4000

# ★손님-측 도구 미전달 (`T2_GIVE_REQUIRED`·기본 OFF·2026-08-26).
#   손님이 call_discoverable_user_tool 로 도구를 실행하려다 env 에 거절당했는데 에이전트가
#   give_discoverable_user_tool 을 안 부른 상태면, **정확한 호출을 지목해** 재생성 채널로 되돌린다.
#   목록은 env 레지스트리 user-side 도출(A2 저작 0·리터럴 0). 엔진이 대신 부르지는 않는다.
#   t7356 전수 표적: 017 미달 1 · 055 미달 1 · 057 미달 2(give 호출 0회) = 3 태스크.
export T2_GIVE_REQUIRED=0
export T2_GIVE_REQUIRED_CAP=2

# ★호출 형식 교정 3단계 ③ (`T2_CALL_FORM_FIX`·기본 OFF·2026-08-26·사용자 확정).
#   ②단계(T2_GIVE_REQUIRED)로 상한만큼 지목한 **뒤에만** 움직인다: 엔진이 래퍼를 바꿔 직접 부른다.
#   내용(도구·인자)은 모델 것 축자 복사 — 만들면 [[03b]] 위반이고 래칫이 그 보존을 검정한다.
export T2_CALL_FORM_FIX=0
