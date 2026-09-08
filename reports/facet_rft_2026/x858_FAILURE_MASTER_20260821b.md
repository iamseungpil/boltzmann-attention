# x858 · FAILURE_MASTER_20260821b — bank_x806_base_nt4 전수 실패 포렌식 종합

- 작성 2026-09-08 · 입력 = 태스크당 1 에이전트의 per-step 구조화 결과 20편 + 우리-층 주장에 대한 반증자(refuter) 판정 46건.
- 대상 런: `bank_x806_base_nt4` (Qwen3.8-27B-FP8 · `--gate 0` base 팔 · `--retrieval_config alltools` · nt=4 · bench sha fc0055dc · user-sim gpt-5.2/low).
- 규율: 수치는 입력에 있는 것만 · 근거 없는 승격 없음 · CONFIRMED 만 우리-층 결손으로 승격 · UNPROVEN/REFUTED(refuted) 는 등급 그대로 표에 남김 · 각 원인 축에 **반증 조건(refutation condition)** 을 병기.
- 태스크별 정본: `tasks_20260907/TASK_060.md` · `tasks_20260821b/x818_TASK_{061,062,063,065,066,067,068,069,074}.md` · `tasks_20260821b/TASK_{075,077,078}.md` · `tasks_20260905/TASK_{082,085}.md` · `tasks_20260908/TASK_083.md` · `tasks_20260830/TASK_{084,086,087,088}.md` (포인터 `tasks_20260821b/x85N_*_pointer.md`).
- 선행 확인(grep 한 곳): `ls reports/facet_rft_2026 | grep -i FAILURE_MASTER` → `FAILURE_MASTER__20260822.md` · `T7336_FAILURE_MASTER_2026_08_22.md` · `x616_FAILURE_MASTER_reg12.md` (전부 다른 런 · 이번 런 마스터는 없었다).
- 파일명: 지시 경로 `FAILURE_MASTER_20260821b.md` 는 훅 §74-b 가 신설을 거부(§7-8) → 프로브 명명.

---

## 0. 재료 정직 선언 (표를 읽기 전에)

| 항목 | 사실 |
|---|---|
| 지시 파일명 `bank_x806_base_nt4_{A,B}_20260821b.*` | **20/20 태스크에서 로컬 부재**(20260821b 명명 파일은 t7333/t7336 런뿐 — 각 정본 §0 에 검색 경로 기록). |
| x806 실물이 로컬에 있는 태스크 | **10개**: 060·061·062·063·065·066·067·068·069(`x818cloud_bank_x806_base_nt4_task_NNN.results.json.gz` + drv) · 074(drv 로그만, results 없음). |
| x806 실물이 로컬에 없는 태스크 | **10개**: 075(rep1 팔 런처 사망 로그 181B 만) · 077·078·082·083·084·085·086·087·088(대리 궤적으로 추적). 078 은 반증자가 리모트 4 sim 완주(0/4)·미영속을 확인 — «미착수» 아님. |
| 오염 | x818cloud 국면(09-07 05:33~14:43)은 shell(srt) 101/101 실패 · 완주 99 sim 중 74 오염(x817 §15-1) ⇒ **base 율 통계 근거로 쓰지 않는다**. |
| 성적 문장 | 미제공. 아래 표는 입력 JSON 의 trial 별 reward·termination 만 옮겼다. |

---

## 1. 성적 표 (입력 수치 그대로)

### 1-1. x806 실물 (10 태스크 · 40 trial)

| task | 채점 sim | reward | infra(미채점) | 종료/비고 | 채점축 |
|---|---|---|---|---|---|
| 060 | 4 | 0.0×4 | 0 | user_stop · 4/4 srt 오염 | DB |
| 061 | 4 | 0.0×4 | 0 | 4/4 contaminated 플래그 | DB |
| 062 | 4 | 0.0×4 | 0 | t2 는 shell 0회(무오염)인데도 0.0 | DB |
| 063 | 0 | — | 4 (CWE 14/16 attempt · 빈 assistant 2/16) | infrastructure_error · messages=[] | — |
| 065 | 4 | 0.0×4 | 0 | t3 는 CWE 후 R1 재시도 완주 | DB |
| 066 | 2 (t0,t3) | 0.0×2 | 2 (t1 빈 응답×4 · t2 CWE×4) | | DB |
| 067 | 1 (t3) | 0.0 | 3 (CWE 8 + 빈 응답 4 / 12 attempt) | | DB |
| 068 | 0 | — | 4 (16 attempt 전부 창 포화: CWE / max_tokens=0 / 빈 응답) | | — |
| 069 | 1 (t0) | 0.0 | 3 (빈 응답 1 · CWE 2) | | DB |
| 074 | 0 | −1(미완주 표기) | 4 (results 없음 · 하네스 전환 중 소실 · 재시도 6회 진행 중) | manifest «aborted at switchover» | — |
| **합** | **20** | **0.0 ×20** | **16** | + 074 미완주 4 | |

### 1-2. x806 실물 없음 · 대리 궤적 (10 태스크)

| task | 대리 실물 | reward | 비고 |
|---|---|---|---|
| 075 | `x818cloud_rep1_task_075_drv.log.gz`(181B, viewmax2 팔) · `bank_p1_task_075`(nt=1) | 사망 / 1.0 | x806 계열 sim 0. 유일 완주 궤적은 1.0(다른 엔진). |
| 077 | x644(base)·relane2a(ours)·t7393_laneB(ours) | 0/3 | 로컬 17 sim 중 order 도달 9 sim 전부 CLASSIC. |
| 078 | x644(base)·t7392(ours 표적) | 0/2 | 로컬 완주 13 sim 전부 0.0. 리모트 x806 4 sim 0/4(미영속). |
| 082 | t7393_laneC(ours, gate 1) | 0/1 | |
| 083 | x644(base) | 0/1 | 로컬 18 sim 전부 0. ACTION 축. |
| 084 | x644(base)·t7393_laneC·lev6b | 0/3 | 로컬 19 sim 전부 0. |
| 085 | t7393_laneC·lev6a(ours) | 0/2 | 로컬 41 결과 파일 전수 통과 0. |
| 086 | x644(base) | 0/1 | |
| 087 | x644(base)·relane2b151·t7393_laneC | 0/3 | 로컬 17 sim 전부 0. |
| 088 | x644(base)+같은 시드 5 sim | 0/6 | A 가족 3(MISSING 3) : B 가족 3(WRONGARG 2). |

---

## 2. 원인 축별 군집표 (축은 데이터에서)

귀속 주체 표기: **M**=model · **U**=user_sim · **E**=env(벤치/서빙/하네스) · **O**=our_layer. 괄호 안 sim 수는 해당 축이 «독립 충분» 또는 «지배 칸» 으로 판정된 sim. 마지막 열 = 이 축 귀속을 무너뜨릴 관측(refutation).

| 축 | 태스크 (sim) | 대표 축자 근거 | 1차 | 2차 | 반증 조건 |
|---|---|---|---|---|---|
| **A. `reason` 자유서술 인자** — close_bank_account_7392 의 gold 미전달 optional 인자를 채워 DB 행(`closure_reason`) 변조 | 060(4/4) · 061(t3) · 062(4/4, t0/t2/t3 은 이 칸 단독) · 065(4/4 단독) · 066(t0,t3; t3 단독) · 067(t3) · 069(t0) — **실물 17 sim** | 062 t1[66] ledger 재독 `closure_reason: Customer consolidating accounts…`; 교차: reason 동반 ∧ 나머지 clean 11 sim → 1.0 = 0, 미동반 clean 10/10 = 1.0(062); reason 동반 70 sim → 1.0 = 0(060 집계) / 66 sim(반증자 재집계) / 59 sim(065) | M | O 없음(T2_FREE_TEXT_ARG 는 base 설계상 OFF) | reason 동반 ∧ 나머지 gold 일치인 sim 이 DB 축 1.0 을 받는 실물 1건 |
| **B. close-first 순서 붕괴** — doc_002 «checking 14일 보유» 가 문맥에 있는데 유일 tenure checking 을 먼저 폐쇄 → open 거부/미호출 | 060(4/4) · 061(t3: 10일→«more than 14 days» 산수 오류) · 069(t0) — 실물 6 sim | 060 t1[21] 스스로 «we'll need to open your savings account before closing» → [55] «no barrier … close it for you now»; 069 t0[35] close → [40] «Error: Account eligibility requirements not met» | M | U(060 t0/t1 pushback 은 대본대로 · 굴복은 모델) | 우리-층 regen/주입이 close 턴 직전에 있었다는 로그 줄(drv `T2_`) 1건 — 현재 0/77 |
| **C. 발명된 전제조건 / 절차 단계 누락** | 061(t0,t1 «wait until posted» 발명 · t2 give_tool 미제공·doc_011 0회) · 069(t0 요건 3종 미유도→Green) · 086(card_design 질문 생략) · 087(«first noticed» 미질문) · 084(슬롯 소진 전 고객 선택 미요구) | 061 t0[71] «I'd recommend waiting until the deposit shows as posted»(문맥 $25/OPEN/No transactions); 087 x644[49] 질문 5개에 알아챈 날 없음 → [61] discovery_date=오늘 | M | — | 해당 «전제조건» 문장이 배달된 KB 문서 어딘가에 축자로 있으면 발명이 아니다(061 doc_005 grep 0) |
| **D. 파생값 인자(Reg E)** — customer_max_liability_amount · discovery_date · provisional_credit · pin | 082(5행) · 084(3행) · 085(3~4칸) · 086(7칸) · 087(2칸) — 전부 대리 | 085 laneC[47] «within 60 days of the transaction … $500»(doc_031 은 «of statement») ; 084 gold 084_7=50(amount 47.5) vs 084_9=412.88(=amount) 규칙 상이 · KB 8편 grep 0; 085_9 gold 50(disputed 14.99) 가 gate.json:11123 min 규칙 반증 | M | **E**(gold↔KB 도달 불가 칸: 083 17키↔gold 16키 · 084 412.88 · 085_9 · 082 prov/gff · 069 Gold $30) | 리모트 KB 12편 회수본에 liability 규칙이 있으면 E 2차는 철회 |
| **E. 상품/옵션 선택 오류** | 067(t3 Blue+Platinum · 카드 축 누락 7.3 동률) · 066(t0 EcoCard) · 077(card_design CLASSIC 3/3) · 086(TechWorld CNP) · 069(t0 Green) | 067[45] «The Blue + Platinum combo is the safer choice»(문맥 [11] Platinum Rewards +0.4% → 7.7% 실재); 077 x644[93] «Classic is the standard default … → CLASSIC» (doc_029 PREMIUM $0 실재) | M | U(086 skimmed 단서 0회 · 077 «fancy» 프라이밍) · O(077 relane2a STANDARD 칸만, §5) | push 없는 base 궤적이 PREMIUM/올바른 조합을 내면 O 기여는 무너진다(x644 는 CLASSIC → 유지) |
| **F. 범위·순서 사건(조기 표면화·이관·EXTRA)** | 088(x644 [55] stolen 조기 표면화 → Part 2 STOP · MISSING 3) · 087(relane2b151 doc_039 이관 과일반화) · 078(x644 freeze→close 치환 · t7392 pending lg close EXTRA) · 082(gff close EXTRA · dispute 자발) · 077(dispute ×5/6 EXTRA) | 088 같은 시드 6 sim «stolen 발화 msg < Whole Foods msg» 로 A:B = 3:3 예외 없음; 078 t7392[66] 자백 «Two pending transactions … may still post even though the card is closed» | M | U(088 Part1§9+Part2§14 병합 · 086 «Option B» 철회) | 088: stolen 을 먼저 말하고도 Part 1.5 → dispute 까지 간 sim, 또는 끝까지 안 말했는데 user 가 Part 1.5 를 안 꺼낸 sim 1건(로컬 6 sim 에 0) |
| **G. 인프라 — 컨텍스트 131,072 포화 / 빈 AssistantMessage / 하네스 전환 / 런처 사망** | 063(4) · 066(2) · 067(3) · 068(4) · 069(3) · 074(4 미완주) · 075(rep1 CRLF) — **실물 16 infra + 074 4 + 075** | `maximum context length is 131072 tokens … prompt contains at least 131073 input tokens`; 배치 059/063/064/068/071 4/4 CWE; 074 로그 끝 `0/4 complete … task_074.3(2190s)`; 075 `set: pipefail\r: invalid option name` | **E** | M(과검색: 067 t3 KB_search 35회·≈115K tok · x644 대리 117k) · **O(관측·계상 결함, §5 CONFIRMED)** | shell 정상 로컬 하네스 재런에서 같은 태스크가 CWE 없이 완주하면 E(srt) 가중은 오르고, 그래도 CWE 면 M(태스크×모델) 로 이동 |
| **H. 재료 부재(미영속·미회수·팔 불일치)** | 077·078·082·083·084·085·086·087·088 (x806 실물 0) · 075 | 078: 리모트 «bank_x806_base_nt4_task_078 2.1M 09-08 09:53 … 미영속 24건» | — | (회수 절차 공백 · §7) | 회수 후 실물 4 trial 이 대리 판정과 다른 변이 집합을 내면 대리 판정 철회 |

**군집 읽기**: 실물 채점 20 sim 은 전부 A(17)·B(6)·C/E(061 t0-t2, 066 t0, 067 t3, 069 t0)로 설명되고 축 A 단독 sim(062 ×3 · 065 ×4 · 066 t3 = 8 sim)은 «나머지 gold 일치» 다. 우리-층이 **인과**로 지목된 sim 은 실물 20 중 **0** (base 팔 · 게이트 레버 미로드).

---

## 3. 직전 런 이후 들어간 수리·레버의 실측 성적표

분류: **발화했나** / **발화하고도 못 샀나** / **발화 기회 자체가 없었나**. 死배선(발화 불가 코드)과 무효과(발화했으나 값 불변)를 구분한다([[55]]).

### 3-1. base 팔(x806 실물)에서의 게이트 레버 — 전부 «기회 없음(설계)»

`t2_run_gated.py:219 if a.gate: import t2_gate_patch` · 워커 `--gate 0`(t2_base_worker.sh:30 · x812_cloud_worker.sh:46 · lane.sh:30) · drv 로그 `T2_`·`[sim=` 줄 0/77. 따라서 T2_FREE_TEXT_ARG · T2_CLAIMPROV · T2_DEMANDED_STEP · T2_FOLLOWUP · T2_ARG_PRODUCERS · T2_RULE_AT_WRITE · T2_ARG_POLICY_AT_WRITE · T2_DISTINCT_ARGS · T2_OVERFLOW_GUARD · T2_DYN_MT · T2_P2_REGEN · T2_VIEW_COMPACT · T2_SEARCH_* 는 **기회 없음**. 단 «gate 0 = 우리 층 0» 은 반증자가 REFUTED(078-②): 하네스 층(alltools shell 노출 + `_check_sandbox_dependencies` stub + `TAU2_SANDBOX_FALLBACK`)은 base 에서도 도구면을 바꿨고 74/99 sim 오염이 그 실측이다.

### 3-2. 레버별 실측 (스택 대리 런 포함)

| 레버/수리 | x806 base | ours 대리에서의 실측 | 판정 |
|---|---|---|---|
| **T2_FREE_TEXT_ARG** (gate.json:11108 · go_stack.sh:926) | 기회 없음 | 062: lost5_viewmax2·x721_t1A·x721_t1B 발화 «reason 제거» → **1.0** / x713_nightB 미발화 → 0.0. 060: re8141p11 발화(reason 제거) → close→open 순서로 0.0 · laneB 발화 → 0.0. 065: x724·resume1p1 발화 → 1.0 | **발화·삼(062/065 계열)** · **발화하고도 못 삼(060 — 축 B 잔존)** |
| **T2_CLAIMPROV feedback_pending** (t2_gate_patch.py:15383-15493) | 기회 없음 | 077 relane2a t61 발화 «None: None ×3» → [78] «user is clearly frustrated … Default: STANDARD»(짜증 문면은 user 턴 어디에도 없음) ; 088 k8141med1 재생성 · 087 relane2b151 98 마크 오발화(무해) | **오발화·해악(077 STANDARD 칸 기여, CONFIRMED)** — reward 반전 경로는 없음(card_design 잔존) |
| **T2_DISTINCT_ARGS** (:13620-13629) | 기회 없음 | 084 re8141p11 turn55/57 stderr 발화 · 값 불변; 085 laneC 술어 참 2/3 · 전달 0. 코드에 `_dv[4]`(처방) 소비 0 · deny 분기 0 | **死배선(계기 전용 · log/deny 동일)** — 승격 금지(084 gold 와 술어 모순) |
| **T2_RULE_AT_WRITE** (write_rules gate.json:11086) | 기회 없음 | run_ours_task.sh:136 이 =1 강제 → provenance 76/76 ON 인데 소비부 :12697 이 `T2_DECIDE_BEFORE_WRITE==1` 블록 안 → 9월 fb 사이드카 70개 RULE 마크 0 | **반증자 관측: ON-but-dead** (085-[1] REFUTED 의 이유부; 등급 미부여 · §6 무료 확인 대상) |
| **T2_ARG_POLICY_AT_WRITE** (:12736) | 기회 없음 | 084 lev6b 만 탑재 → 084_7/8 일치(n=1) ; 085 lev6a A3 에 liability 행 0 → «무발화 0행» | **기회 있었으나 선언 행 부재(085) · n=1 방향만(084)** |
| **operator-scope** (t2_resolve.py:281-329 · T2_SCOPE_DENY_CAP=1) | 기회 없음 | 082 laneC 3회 발화 → 재발행 통과 · 인자 불변 · 090 같은 발화 후 1.0 | **발화·무효과(비인과 CONFIRMED)** |
| **T2_RESOLVE reference-unmatched** | 기회 없음 | 084 re8141p11 turn50/52/55 deny · fb에 `btxn_` 0 → 초안 인자 복원 불가 | 발화 · 인과 **UNPROVEN** |
| **GB2_NOTICE_BEFORE_TRANSFER / WORK-INCOMPLETE** | 기회 없음 | 087 relane2b151 deny 1(준수 후 이관) · WORK-INCOMPLETE 발화·무시 | 발화하고도 못 삼 |
| **`_install_overflow_guard`** (:7822, gate 전용) | 기회 없음 → CWE 4/4 소실(063) | 로컬 전 로그 발화 **0회**; 실효 경로는 `_gen` :8774 CWE 분기(71 sim 우아한 종료) | **死배선(백스톱)** — 063-1 CONFIRMED 의 반증자 정정 |
| **`_install_failed_persist`** (t2_run_gated.py:29-67) | opt-in 미설정 + set_state 만 래핑 → 063/074 궤적 0 보존 | — | **기회 있었으나 경로 미커버(CONFIRMED)** |
| **`--max_retries`** 미지정 | 063 16 attempt · 067 12 · 068 16 (사실) | 같은 배치 023/060/065/098 은 재시도로 완주 | 낭비 CONFIRMED · «포렌식 불가의 원인» **REFUTED**(0 이어도 messages=[]) |
| **sandbox-check stub** (t2_run_gated.py:188-190) | 060 4/4·065·066 등 shell 낭비 호출 · 모델 KB_search 복귀 | 오염 sim pass 54% vs 무오염 64%(가르지 못함) | **비원인·위생 CONFIRMED** · 2차 원인 UNPROVEN |
| **T2_AGENT_MAX_TOKENS / `_dyn_mt_target` 정규식** | 068 문면 관련 | 표본 부재 | **UNPROVEN** |
| **lane_rep1.sh 120s 가드** | — | 075: 실행자 lane_rep1.sh:27-37 에 가드 존재(bd5e041a) | «미적용 재발» REFUTED |
| **mutation_diff (t2_forensic.py:1166)** | infra sim 4 → `clean:True` | — | **계기 결함 CONFIRMED** |

---

## 4. 회귀 전용 절

직전 런의 태스크별 성적표는 입력에 **없다** → «내려갔나» 자체는 판정 불가. 아래는 입력의 `vs_prior` 에 근거한 **원인 이동**만 적고, 팔았는지 확정 못 하는 곳은 «미상» 이다([[70]]).

| task | 직전(선행 문서) | 이번 | 무엇을 팔았나 |
|---|---|---|---|
| 060 | x737 행11: D8 regen 이 순서 파괴(PLAUSIBLE) | regen 0 인데 4/4 close→open | **D8 은 필요조건 아님(반증)** — 판 것 없음, 원인 동일(모델) |
| 061 | FLOOR 07-11: FABTOOL/EARLYTR | FABTOOL 0 · EARLYTR 0 · 발명 전제조건/산수 오류 | 원인 **변경** · 회귀 여부 미상 |
| 062 | x817 «T3 stack regression / unconfirmed» | base 0/4 = reason 단독, 스택 발화 시 1.0 | **회귀 아님 → stack-rescued 재분류** |
| 063 | t7346 등: 결정 지점(Silver 배제 실패) | CWE 소실(결정 지점 도달 전) | 원인 **종류 변경**(env) · 미상 |
| 065 | FAILURE_AXIS 08-15 «1칸→0» | 4/4 reason 단독 | 동일 · 회귀 아님 |
| 066 | 08-15 «값오답·미호출» | t3 는 본체 0·reason 단독(첫 카드 정답 sim) | **개선 방향**(단 reward 0) |
| 067 | x644 base: Purple+Platinum Plus 정확·카드 미추천 | t3 Blue/Platinum 오선택 추가 · trial 0~2 CWE(x644 에 없던 형상: 검색 3회→35회) | **확장형 악화** — 검색량 10배의 이유 UNPROVEN(클라우드 서빙 차이) · p1(gate 1) 1.0 |
| 068 | FLOOR 07-11 MIXED(궤적 있음) | 4/4 결측 | 원인 변경(env) · 미상 |
| 069 | 상품명 오답 축 | 순서 붕괴+요건 미유도+reason | 원인 **변경** · 미상 |
| 074 | 산수 축(fee_refund 4칸 · 우리 서브 3/4 오류) | 완주 여부 축(하네스 중단) | 상류로 이동 · 미상 |
| 077 | x737 §1f-10 동일 | 동일(+base 도 CLASSIC) | 판 것 없음 |
| 078 | x737 감사 B: MISSING freeze/unfreeze | t7392(표적 런): MISSING 0 → EXTRA(lg unfreeze+close) | **모양 이동(누락→과잉)** · 미상 |
| 082 | t7336: EARLY_TRANSFER | 이관 0 · ARG 5행 | REACH→ARG 하강 · 미상 |
| 083 | x737 1f-12 동일(env) | 동일 + «후행 0 바이트 재현 불가» 철회 | 판 것 없음 |
| 084 | N97 71회 중복 read | dup 0 · ARG | REACH→ARG · 미상 |
| 085 | x737 동일(B 파생값) | 동일 | 판 것 없음 |
| 086 | N97 ID 날조 | 날조 0 · ARG 7칸 | REACH→ARG · 미상 |
| 087 | N97 read 오류·이관 | read 완주 · dispute 2칸 | REACH→ARG · 미상 |
| 088 | x817 «이관 막힘» | 이관 «너무 일찍 잘 함» | 원인 변경 · 미상 |

요약: 회귀로 **확정**된 태스크 0 · «판 것» 이 특정된 항목 0. 062 는 회귀 표에서 빼야 한다.

---

## 5. 반증자 판정 반영 (46건)

집계: **CONFIRMED 27 · REFUTED 11 · UNPROVEN 8**. 각 항목의 반증 조건(refutation condition)은 입력 `counter_evidence` 칸 그대로이며 여기서는 요지만 옮긴다.

### 5-1. CONFIRMED → 우리-층 결손으로 승격 (전부 계기·위생·관측 결함이거나 ours 팔 대리; x806 채점 sim 의 reward 를 바꾼 것은 0)

| # | 출처 | 결손 | 반증자 정정(문면) |
|---|---|---|---|
| C1 | 063-1 | gate 0 에 CWE 우아한 종료 없음 → sim 4/4 소실 | 실효 경로는 `_gen` :8736-8782, `_install_overflow_guard` 는 발화 0 백스톱; 워커 줄 :30-34 · export :20; «base=우리층 0» 확정 시 철회 조건부 |
| C2 | 063-2 · 074-OL74x | `_install_failed_persist` 가 set_state 만 래핑 + opt-in 미설정 → CWE/빈응답/kill 궤적 보존 0 | 074: 1차 이유는 opt-in 미설정; 빈 응답은 `message.py:287 validate`; 버려진 궤적 10(7 아님); `ls … grep fail|persist` 14건(0 아님) |
| C3 | 063-3 · 067-1a/1b · 068-2a | `--max_retries` 미지정 → 16/12/16 attempt · 결과 동일 | 절약분 ≈20분(37 아님); 트레이드오프(재시도가 023/060/065/098 은 살림) |
| C4 | 063-4 | `mutation_diff` 가 reward_info=None 에 `clean:True` | pass 로 둔갑은 아님 — 실패 종류 집계에서 0행으로 소실 |
| C5 | 060-1 · 068-1 | sandbox stub 이 srt 부재 fail-fast 를 우회 → 고장 shell 이 스키마에 실림(101/101) | «매번 KB_search 복귀» 는 5회 중 4회; «런 시작 전 정지» 는 과장 가능 |
| C6 | 060-2 | T2_FREE_TEXT_ARG 는 base 미적용(설계) · (A) 를 못 삼 | laneB 060 은 close 만(open 미호출); 런처 줄 46-50; reason 66 sim(70 아님) |
| C7 | 077-1/2/3/5 | claimprov `None: None` push → 가짜 «손님 짜증» 역추론 → 질문 생략 → STANDARD 기본값(relane2a 077 · 082 n=2) | 단독 결정론 아님(기여 원인); 반전 경로 없음 |
| C8 | 082-1/2 | operator-scope 3회 발화 · 재발행 통과 · 비인과 | 발화 횟수 3 = 하한 |
| C9 | 084-① · 085-[4] | base 팔 = 게이트 레버 부재 | 발사처 둘(iso_tau3 / /root) · gate=0 71/71 |
| C10 | 084-② · 085-[2] | T2_DISTINCT_ARGS 계기 전용 · `_dv[4]` 폐기 · deny 승격 금지 | «발화-무시» 아니라 «전달된 적 없음»; log/deny 동일 |
| C11 | 085-[3] | gate.json:11123 `_note` «min 19/19» 는 085_9 gold 가 반증 | 호출부 줄 13486→13620 드리프트 |
| C12 | 084-③a | 훅 scaffold_guard.py:200 이 `tasks_20260821b/TASK_*.md` 신설 차단 | 훅 위치 `C:/workspace/.claude/hooks/`; Bash 경로로는 신설된 파일 존재 |
| C13 | 075-1a/2c | rep1 팔 `run_ours_task.sh` CRLF 로 런처 사망 · 큐 35건 드레인 | 전송 채널 미관측 |
| C14 | 067-1d/1e | max_retries 는 067 실패 비원인 · 인용 줄 일치 | 배치 통계엔 비중립(삭제 편향) |

### 5-2. REFUTED (refuted · 등급 유지 · 승격 금지)

| 출처 | 주장 | 반증 요지 |
|---|---|---|
| 067-1c | max_retries 기본 3 이 궤적 추적 불가의 원인 | `--max_retries 0` 런 4건도 messages=[] |
| 068-2b | x812 워커 고유 결함·포렌식 불가 | 정본 t2_base_worker.sh 도 동일 · 플래그 무관 |
| 074-설계선택 | run_queue_20260901.sh 가 074 를 맨 뒤에 둠 · 워커 지목 | 실행자는 x812_cloud/lane.sh · 그 큐는 t3prime 용 · 074 뒤에 022 |
| 075-1b/2a/2b | 드레인 시점 13:44 이후 · 실행자 t2_lane_worker · 가드 미적용 | mtime 11:11(재기동 11:12 이전) · 실행자 lane_rep1.sh · 가드 존재 |
| 078-1/2/3 | 미착수 · gate 0=우리층 0 · persist() 가 «상태 4» 로 뭉갬 | 리모트 4 sim 완주 0/4 미영속 · 하네스 층 개입 실측(74/99) · «4» 는 손으로 넣은 실측 인자 |
| 084-②b | reference error 게이트 UNPROVEN | T2_RESOLVE reference-unmatched 로 특정 가능(인과는 UNPROVEN) |
| 085-[1] | T2_RULE_AT_WRITE 기본 OFF 라 미배달 | ON(run_ours_task.sh:136)인데 소비부가 T2_DECIDE_BEFORE_WRITE 블록 안 → dead wiring |

### 5-3. UNPROVEN (등급 유지)

068-1b(srt→CWE 2차 원인) · 068-3(max_tokens=0 기전) · 068-4(`_dyn_mt_target` 잠복) · 075-부수(x806 075 궤적 존재) · 077-4(t7393 «push 없음» 대조) · 082-3(재발행 인자 동일) · 082-4(x806 082 귀속) · 084-③b(지시서 자리표시자).

---

## 6. 처방 큐 (3분할 · [[62]] 순서 = 무료 → 격리 → 유료 확인)

### 6-A. 무료 수리 가능 (코드 직독·기존 데이터·단위검정으로 닫힘 · 유료 런 0)

| 순서 | 항목 | 표적 | 기대 상한 | 근거 |
|---|---|---|---|---|
| A1 | `mutation_diff` 에 reward_info=None 표식(`sidecar`/`gold:None`) | 집계 전체 | 계기 정직성(reward 무관) | C4 |
| A2 | `_install_failed_persist` 를 generate 단·kill 경로까지 (사이드카) | CWE/빈응답 sim | 포렌식 가능성(reward 무관) | C2 |
| A3 | gate.json:11123 `_note` 정정(min 규칙 철회) · 1f-12-0 083 행 철회 · x817 T3 목록 062 → stack-rescued · TASK_067 §7-4a 모순 문장 · TASK_078 «미착수»→«미영속» · TASK_074 문면 4건 | 문서 | — | C11 · 083 · 062 · 067-1c · 078-1 · 074 |
| A4 | claimprov `feedback_pending` 렌더 `None: None` 복원(kind/what 키) + 단위검정(relane2a 077 t61 fb 픽스처) | 077/082/087/088 ours | 077 STANDARD 칸만(반전 없음) | C7 · x737 D8/D9 |
| A5 | T2_RULE_AT_WRITE dead-wiring 코드 직독 확인(:12551 블록 밖으로 뺄지) — **수정은 격리 후** | 085/086/087 | 미상 | 085-[1] 반증자 관측(등급 미부여) |
| A6 | `.gitattributes eol=lf` + 리모트 `sed -i 's/\r$//' *.sh` (승인 후) · t2_lane_worker*.sh 에도 120s 가드(별건 예방) | 런처 | 재발 방지 | C13 |
| A7 | 리모트 미영속 24건(078 포함) 로컬 회수 → 대리 판정 10 태스크를 실물로 재계산 | 077~088 | 귀속 등급 상승 | 078-1 · 읽기는 자유, SSH 는 사용자 |

### 6-B. 격리 프로브 선행 필요 (새 술어·새 전달 — 격리로 잰 뒤에만 배선 · 조건 명시)

| 순서 | 항목 | 표적 | 기대 상한 | 조건(격리 통과 기준 · 반증 조건) |
|---|---|---|---|---|
| B1 | close_bank_account_7392 전제조건 선언층: 미완 personal-savings 개설 의도 + 유일 tenure checking 폐쇄 요청 → 1회 deny + 순서 이유 요구(doc_002/doc_005 축자 · write_evidence_specs) | 060(4/4) · 061 t3 · 069 t0 | 060 은 **T2_FREE_TEXT_ARG 와 동시**일 때만 1.0 가능(두 변이 독립 충분) · 069 는 KB↔gold 모순으로 상한 0 | 반대 태스크(close 가 먼저인 gold) 전수 대조 후 · 태스크 리터럴 0 · 격리에서 060 t1[55] 형 4/4 deny 이면서 close-first gold 0 오발화 |
| B2 | write 직전 사용자 명시 동의 턴 술어(설명과 실행 동일 턴 결합 차단) | 060 t1/t3 · 078 x644 | 미상 | 기존 T2_DEMANDED_STEP/FOLLOWUP 이 덮는지 스택 로그로 선확인 |
| B3 | Reg E 파생값 write-point 전달: T2_RULE_AT_WRITE 실배선(A5 뒤) + A3 에 liability/discovery_date 행, «first noticed» 필드에 오늘/거래일 복사 차단 | 085(3칸) · 086(3칸) · 087(2칸) · 082 | lev6c 2/4 · lev6b 084_7/8 이 방향 근거; **gold 도달 불가 칸(083·084_9·085 prov·086 SupplyPro) 은 상한 밖** | 084 음성 대조 필수(liability==amount gold) · DISTINCT_ARGS deny 금지 · 격리에서 doc_031 «of statement» 축자 전달 후 $50 선택률 측정 |
| B4 | close_debit_card 전 pending 거래 존재 시 반려·freeze 유지(doc025 요건4) | 078 t7392 EXTRA | 미상(n=1) | 078 x806 실물 4 trial 회수 후 |
| B5 | 카드 순이득(보너스×잔액−연회비) 비교 행을 기존 계산 레버 축에 도메인 데이터로 | 066 t0 · 067 t3 | 067 은 p1(gate 1) 1.0 이 이미 있어 base→gate1 차이 재확인이 우선 | 부호표 선행 |
| B6 | CWE 우아한 종료를 gate 밖 플래그로(`_gen` :8736-8782 분기 이식) + base 워커 `--max_retries 0` 정책 | 063/066/067/068/069 infra 16 sim | **reward 상승 아님** — 부분 궤적 채점·관측 | «base = 우리 층 0» 정의를 사용자가 먼저 결정 |
| B7 | 088 «문의 안 한 카드 보안 상태 선제 표면화 금지» 류 절차 문구 | 088 A 가족 | 미상 | 선제 고지가 gold 인 반대 태스크 전수 대조 후에만 |

### 6-C. 레버 없음(경계) — 처방 대상 아님 · 표지·분모 처리만

| 항목 | 태스크 | 처리 |
|---|---|---|
| base 팔의 `reason` 결손 | 060·061·062·065·066·067·069 | 스택에 T2_FREE_TEXT_ARG 기존재 · base 는 «상수 결손» 으로 명기(설계) |
| gold↔KB/env 도달 불가 | 083(17 required ↔ gold 16키, ACTION 축) · 084 customer_max_liability(412.88·50 규칙 상이) · 085_9 · 082(prov=true ↔ doc_032 NOT REQUIRED · gff close) · 069(Gold $30 rebate ↔ gold Silver Plus) | «doc-unreachable» 표지 · 분모 제외 여부 [[68]] 절차 |
| user_sim 대본 이탈 | 062 t1(3,500 일괄) · 086(skimmed 0회·Option B 철회) · 088(§9+§14 병합) · 087(§11 조건절 분산) · 067 t3(«and credit card» 삭제) | 양화만(user_only_review) |
| 모델 해석 고유 | 077 card_design CLASSIC(두 세대 9/9) · 088 조기 표면화 | 우리-층 문면 없음 |
| CWE 자체(과검색×131k) | 063·066~069·074 | base 에선 수리 없음 · 스택 T2_SEARCH_*/VIEW_COMPACT 가 k 를 줄이는지 재확인은 별건 |
| 미래날짜 posted 거래 시드 | 077 | 벤치 질의 |

---

## 7. 이 종합이 못 사는 것 (정직 절)

1. **x806 실물은 20 태스크 중 10** — 077~088 의 원인 진술은 대리 궤적(x644 base·ours 팔)에 대한 것이고 x806 4 trial 에 대한 귀속은 전부 **UNPROVEN** 이다(082-4 등급 그대로). 리모트에 최소 078 실물(4 sim 0/4)이 있고 미영속 24건이 있다 — 회수 전엔 이 절이 실물 판정으로 승격되지 않는다.
2. **채점된 x806 20 sim 에 우리-층 인과 결손 0** — base 팔이라 당연하나, 이는 «우리 층이 무해» 가 아니라 «측정 자리가 없음» 이다. 반대로 «gate 0 = 우리 층 0» 도 REFUTED(하네스 층 오염 74/99) 이므로 base 율 자체를 근거로 못 쓴다(x817 §15-1).
3. **회귀 판정 불가** — 직전 런 성적 미제공. §4 는 원인 이동만 적었고 «판 것» 은 0건 특정.
4. **CONFIRMED 우리-층 결손 14묶음은 전부 계기·위생·관측·문서 결함이거나 ours 대리(C7)** — 어느 하나를 고쳐도 x806 reward 는 오르지 않는다. reward 에 닿는 처방은 전부 §6-B(격리 선행)이고 **기대 상한을 잴 격리 결과가 입력에 0건**이라 상한은 «미상» 또는 n=1 방향뿐이다.
5. **축 A 의 교차 집계 수치가 세 종류(70/66/59)** — 집계 기준(첫 close 호출·팔·기간) 차이로 보이나 이 종합에서 통일하지 않았다. 방향(0 통과)은 세 집계 동일.
6. **CWE 16 sim 의 model/env 분리 불가** — 궤적 0. 067 «검색량 10배» 의 이유 UNPROVEN.
7. **반증자가 발견한 dead wiring(T2_RULE_AT_WRITE)** 은 REFUTED 판정의 이유부에서 나온 관측이라 이 규율상 승격하지 않았다 — A5 로 무료 확인 후에만 결손으로 올린다.
8. 지시 경로 `FAILURE_MASTER_20260821b.md` 는 훅 §74-b(scaffold_guard.py:200-214 «보고서 신설 차단 · 새 파일은 프로브 xNNN 만»)가 Write 를 거부해(§77 원인계약은 통과) 프로브 명명 `x858_FAILURE_MASTER_20260821b.md` 로 두었다. 지시 경로로 옮기려면 사용자 승인이 필요하다.
