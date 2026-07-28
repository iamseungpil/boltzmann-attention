# day6 전수 포렌식 + 결함 수정 설계·구현 (2026-07-28)

> 대상: `bank_day6front{A,B}_20260728` (front32·conc1·C209 P1~P10 스택·BYREF/NEARDUP OFF).
> 방법: 실패 21건 전 궤적 per-step + DB-diff 오프라인 재현(전 16 DB-basis) + 레버 발화 per-task 지도 + P10 사이드카 3건.
> 스코어: **PASS 11/32 (n=31 기준 11/31·P9(a))** — A 4(006·**008**·017·034) B 7(001·002·003·021·**023**·025·032).
> 등급: 점수 비교는 [D](pass^1·C192 user-sim 비결정) · 기전 판정은 [S].

---

## 1. 기전-소멸 판정 (C209의 1차 목표) — 달성

| 기전 | day5 | day6 | 판정 |
|---|---|---|---|
| ctxover(`is too large`) | 7 | **0** | [S] 소멸. `T2_DYN_MT` 발화 0 = **P5 뷰-예산이 선제 억제**(천장 근접 자체가 없음·P1은 백스톱 대기) |
| infra | 2 | **0** | [S] GUIDANCE-계열 재발 0(P2 뷰-채널 26회 정상 우회) |
| C2-a(unavail) | 0/223 NameError | **fired 9+7 / ok 89+93** | [S] P7 실증·LEVER_HEALTH 전량-스킵 0 |
| 실패-sim 궤적 | 무영속 | **사이드카 3건** | [S] P10 첫 실전 — 신규 위험(§3-D1)을 즉시 특정 |

점수 이동: **+023(day5 ctxover→PASS: P1/P5 반사실 예측 적중 실측)** · +008(grant 발화 후 transfer 완결·enum 정선택) /
−033·−016(§2). 총수 11 동일 — 기전은 닫혔고 잔여는 행동·값 층으로 이동.

⚠중간 점검의 "TERM_GRANT 발화 0"은 **grep 패턴 오류였다** — 실제 마크는 `[T2_EPLAN] terminal grant`·**9회 발화**
(008·015·028·033·035·010×2·018·029). 035가 사상 처음 transfer를 실호출(035_2 match=True)한 것도 grant 기여.

## 2. 실패 21건 분류 (전수)

### 2a. 신규-레버가 만든 회귀 [S] — 2건
- **033(−)**: [4] `give('submit_transaction')`→Unknown → **P2 뷰-채널 피드백 1회 노출을 모델이 무시** →
  [6] 말-안내 후퇴(day5의 in-history GUIDANCE는 준수됐던 바로 그 지점) → KB 재검색 없음 → 1822/0218
  에스컬 프로토콜 미발견 → 직행 transfer(gold=프로토콜 경유) → ACTION 0. **뷰-채널 1회 노출 < in-history**
  의 실측. grant는 발화·required도 작동(강제 transfer를 auth 게이트가 deny→re-gen 산문 = 게이트×grant 간섭 관찰).
- **016(−)**: [24]에서 정답 문서("Referral Statuses"·ERROR 상태 해설)를 **검색해 놓고** [29] 조기 에스컬 →
  ###TRANSFER###. gold=friend의 750달러 거래 제출(submit_transaction 경유). day5는 완주했었음 —
  forced-read-without-revision 계열([[18]])·직접적 레버 원인 없음 → **pass^1 변동 성분 우세 [M]**.

### 2b. 결함 2건 (중간 점검 보고분·확정)
- **D1 — rb(CHECK-FIRST) deny의 replay 위반 [S]**: `require_tool_before` 표적 4종(apply_for_credit_card·
  open_bank_account·submit_referral·pay_credit_card_from_checking)이 **전부 mutating 유저-도구** —
  유저-sim이 직접 호출한 것을 미실행 deny-스텁으로 커밋 → replay 재실행 → content 불일치 ValueError.
  day6 실측 3건(007·010·025·P10 사이드카로 특정)·전부 재시도 1회 회복(infra 0·시도 소각). rb 설계
  주석("손님 호출 포함 1회 deny")이 replay-불변식 이전에 쓰인 것.
- **D2 — TERM_GRANT 술어 ⓑ 결함 [S]**: 004 = notice→동의→transfer **호출**→PREKB deny(fam=transfer)→종료.
  ⓑ가 deny(error)된 호출도 '호출됨'으로 계상해 grant 미발화. + 동의-터미널 직후의 PREKB deny 자체가
  마지막 행동 턴을 소각(레이스 공범).

### 2c. coverage/compute 값-층 — 6건 (018·020·022·026·028·029) : **전진했으나 값·근거 층에서 실패**
day5 대비: 전원 give 도달(018_1·020_1 등 match=True)·일부 dispute 제출. 잔여 원인 3축:
1. **account_open 여전히 미공급**: P4 지목은 발화(W-f 실측: 027 재호출 args=7,514자·026=4,140자 —
   **P6 없이는 재타이핑 재호출** = 리뷰 필수3의 예측 그대로), PROD_BIND가 날조를 강등(022: 73/77행 날조 적발!
   027: 26행·026: 14행) — 그러나 모델이 **accounts를 끝내 안 읽어** promo 행 판정 결손 지속.
2. **격리 rate 품질**: 028 갱신값 오류(pred 128 vs gold 642 등 = 카테고리-부스트 rate 오선택·C185a 계열).
3. **★신규 날조 표면 — 발명 txn id로 dispute 제출**: 020(`txn_e0h5i0j0k004`·`txn_d9g4f8h9i903`) ·
   028(`txn_4c5d6e7f8g9h`) — 비-hex 문자(g~k) = 명백 발명. **유저가 실행까지 했고 env가 수용**(존재 검증 없음).
   FAB_STRIP(id-근거 검사)이 못 잡음 — give의 **중첩 arguments 안 txn id까지 근거-검사가 닿는지** 후속 확인 필요.
4. 027 = **2런 연속 유일-diff가 조회 1회 과행동**(get_user_dispute_history CALLED·gold 6행동 전부 match) —
   over-action 벌점의 가장 순수한 실측(2/2 재현·모트 서사 재료).

### 2d. 도구-선택 오류 — 2건 (040·041)
day5의 give-누락/규모 붕괴에서 **양상 전환**: last-4를 못 얻자 formal `file_credit_card_transaction_dispute_4829`
대신 **`submit_cash_back_dispute_0589`로 9건(040)·다수(041) 제출** — 사기/중복청구 분쟁을 캐시백-분쟁 도구로(오도구).
required 인자를 못 채우자 인자 없는 도구로 우회한 **경로-최소저항 행동** [S].

### 2e. 재발형(day5 §9 잔여 그대로) — 7건
004(D2)·005(gold 파손·n=31 처리)·007(사임→미신청)·012(우회안내 만족분기)·014/015(주장-KB 대조 생략)·
024(fit 실행 후에도 Gold 신청·gold=Bronze — rec 오류)·010(referral 흐름 미완+0218 이탈)·035(전진: transfer는
호출·잔여=긴급 프로토콜 unlock 미발견=KB 0회 재발)·019(2건 누락·abstain 행 계열)·026(§2c)·022(§2c).

## 3. 해결책 (F1~F5 = 이번 구현 · F6~ = 설계 대기)

| # | 결함 | 수정 (구현 완료·오프라인 PASS) |
|---|---|---|
| **F1** | D1 rb replay | 표적이 `_replay_compared`(env등록∧mutating)면 **deny-스텁 금지 → 실행 통과 + 생성-레벨 사후-점검 지시**("방금 X가 점검 Y 없이 실행됨 — 지금 Y 실행·대조·교정"). 레버는 사전→사후 점검으로 약화되나 측정 정합성 우선([[19]] 조정). 형제-스텁에도 동일 passthrough |
| **F2** | D2 ⓑ | **비-에러 응답 호출만 '호출됨' 계상** — deny된 transfer는 미호출 → grant 발화(004형 회수 경로) |
| **F4** | D2 레이스 공범 | **notice 공표 후 transfer-fam PREKB deny 면제**(A2 notice 게이트가 KB-확인 의무를 포섭·판정=notice_text 부분문자열) — 동의-턴 소각 차단 |
| **F5** | 033 뷰-채널 약화 | view-fb **K회(기본 2) 연속 생성 재노출**(`T2_FB_VIEW_K`) — 1회 노출<in-history 실측 보정 |
| (F3) | 033 required | 조사 결과 **required는 정상 작동** — 강제 호출을 auth 게이트가 deny한 것(게이트 정상 동작·수정 없음·간섭 W-g로 계측 등재) |

**F6~ (미구현·설계 대기·우선순위순)**:
- **F6 발명-id 근거검사**: give/dispatcher **중첩 arguments의 id-류 값**까지 FAB_STRIP/WRITE_EVIDENCE 근거-검사 도달
  확인·미달 시 중첩 unwrap 확장(020/028 발명 txn 직격). 기존 레버 확장이라 [[05]] 무해.
- **F7 = P6 ON**: W-f 실측 확보(재호출 7.5k/4.1k자 재타이핑) = 설계서의 GO 신호 충족 — **day7에서 `T2_SG_BYREF=1` 승격**
  + P4 지시문에 "@last 참조 가능" 결합. account_open 공급 문제도 A2 param 문구가 이미 "레코드에서 복사"를 요구.
- **F8 오도구 선택(040/041)**: dispute-계열 도구 선택이 갈리는 지점의 A2 설명 보강(cash-back dispute=캐시백 전용 명시)
  — A2 데이터 수정만. 값-층(028 rate 품질)은 isolate 개선 별도(C185a 트랙).

## 4. 판정 요약 (등대 원칙: 레버는 하나를 사면 하나를 판다)

- **산 것**: ctxover·infra·무음실패 소멸 / 023 회수·008 완결 / grant 9회 발화·035 transfer 최초 실행 /
  PROD_BIND가 대규모 날조(73행) 적발 / P10이 신규 위험을 당일 특정.
- **판 것(실측)**: P2 뷰-채널 약화(033) · rb-deny의 replay 폭탄이 P2 우회 후 잔존 계열로 노출(D1) ·
  grant×PREKB(004)·grant×auth-게이트(033) 간섭 — F1~F5로 조정(끄기 0).
- 점수 11/31은 [D] — 기전 지표는 전부 전진. 남은 큰 덩어리 = coverage 값-층(P6 ON + F6·F8)과
  행동-층 잔여(§2e = scale/learn 축 유지).
