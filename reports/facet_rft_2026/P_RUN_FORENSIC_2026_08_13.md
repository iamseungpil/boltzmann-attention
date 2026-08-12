# 밤샘 p런 실패 전수 포렌식 — 2026-08-13

> 대상: `bank_judge6p_{a,b}_20260813p` (6태스크 · nt=4 · 24 sim · 수리5 중 5호 死 · sha `35379ea7`).
> 데이터: `sim_results/bank_judge6p_*_results.json.gz` + `fb_*.jsonl.gz` (커밋 `8ed8fed9`) — 리모트 유일본을 영속화 후 분석.
> 방법: 실패 13 sim 전수 per-step 정독 + 사이드카 정확 조인(첫 user 발화 SHA1 — §6.1 참조) + 오프라인 격리 재생(x282).
> 결론 순서는 [[55]]: 우리 배관 → 우리 문구 → 계기 → 모델.

## §0 실패 지도 (13/24)

| sim | task | trial | 실패 칸 | 한 줄 근인 |
|---|---|---|---|---|
| b#3 | 010 | 0 | 010_1 | 재제출 지시 미발화 → transfer 로 흡수 |
| b#2 | 010 | 1 | 010_1 | 정답 초안을 GB2 recovery 가 삭제 (§2.3) |
| b#6 | 010 | 2 | 010_1 | 지시 미발화 (transfer) |
| b#8 | 010 | 3 | 010_1 | 지시 미발화 (transfer·user-sim 역할역전 포함) |
| a#1 | 098 | 0 | 098_2 | **오답 추천**: Green Fee-Free 를 추천 → 손님이 그대로 제출 |
| a#6 | 098 | 2 | 098_2 | **오답 추천**: "Blue Account 는 KB에 없다" → Light Blue 제출 |
| a#2 | 070 | 0 | 070_1~4 | 미도달: `submit_transaction`(가짜 user-tool) 호출 후 정지 |
| a#4 | 070 | 2 | 070_0~4 | **전멸형**: 카드 fit 오선택→계정명 5회 표류→검증 거짓말→개인계좌(Light Blue) 추천→**포털 링크 날조**로 종료 |
| a#7 | 070 | 3 | 070_3~4 | 자리표시자 날조 verify([20] John Doe)·체인 18→34 정지·open 미도달 |
| a#3 | 071 | 0 | 071_1·2·4·5 | 오답 추천(Hunter Green+Business Bronze Saver)→EIN 요구(인접-절차 오염)→포털 날조→"도구 없음" 거짓([38])→늦은 개설(오답 클래스)·savings 미개설 |
| a#9 | 071 | 1 | 071_4·5 | 개설 3회 전부 오답 클래스(Hunter Green/Gold Plus Saver/Emerald Saver)·가짜 `open_account` 시도 |
| a#8 | 071 | 2 | (칸 0·db) | **gold 6/6 완수 후** user-sim 즉흥 funding($500/$1000) 순응 → 여분 transfer 2건 → db_match 붕괴 |
| a#11 | 071 | 3 | (칸 0·db) | 동형($8000/$3000)·user가 *"내 쪽엔 도구 없으니 네가 해달라"* 역제안 |

## §1 010 0/4 — "GO 신호가 구조적으로 없다"

**gold**: `010_1 = submit_referral(Platinum)` · **requestor=user** — 손님이 실행하고, 에이전트는 *말로* 시켜야 한다.

**승리 시퀀스** (k런 #7·#8, m런 #2 — 통과 3건 공통): 에이전트가 "창이 지났으니 재제출 가능"을 발화 →
user-sim 이 빈 발화 + `Referral request submitted successfully!` 로 스스로 실행. `give_discoverable_user_tool` 불요.

**p런에서 일어난 일**:
1. 우리 [ACTION]+[ORDER] 푸시는 수리1 이후 **정상 전달**됐다 (sim당 4~6회).
2. CANNOT-YET("고객에게 시키는 것도 이르다")는 발화 시점(turn 2~18)엔 **참말**이었다 — read 완료가 msg 19~21.
   **x282 오프라인 재생**: 4 sim 전부 K=20/22 에서 요건 정확히 빔 — `requirements_for`·`_fam` 크레딧 무결.
   (앞선 "검사기가 거짓말" 의심은 **기각** — fb `turn`=기록 시점 메시지 수, 후기처럼 보인 푸시는 실제론 완료 전.)
3. 완료 후 푸시는 CANNOT-YET 를 제때 떨어뜨리고 [ACTION] 소유권+상태표만 남김 — **그러나 조건문**("Once you
   have everything ... tell the customer")이고, **긍정 해제 신호가 없다**. `cleared()`(=DEFAULT_CLEARED "That now
   holds. Do X now")는 **모델이 직접 시도했다 보류된 표적만** 부활시킨다. user-측 도구는 모델이 시도 자체를
   안 하므로 **영구히 조건부만 남는다**. 금지문 2~3회 반복 priming([[42]]) 후 해제 0 ([[63]] 동형).
4. **스모킹건 (b#2 t1, msg34)**: 모델 초안이 마침내 정답 — *"the 9-day window ... has passed ... you should be able
   to re-submit the Platinum Rewards Card referral now"* — 을 뱉은 그 턴에 GB2(transfer 시도) deny 의 recovery
   *"Send the user **exactly this message** now: 'TRANSFER NOTICE: ...'"* 가 발화 → regen 이 초안 전체를 버리고
   **알맹이 없는 NOTICE 한 줄로 커밋**([34] 확인). **우리 recovery 문구가 모델의 정답 발화를 지웠다.**
5. 전달량-성적 단조 역상관: k런(입법으로 푸시 전면 침묵) 2/4 → m런(_ufb 덮어쓰기로 부분 침묵) 1/4 →
   p런(수리로 완전 전달) 0/4. ⚠n=4·±2 잡음(C430) — 단독으론 약하고, ③④의 기전 정독이 방향을 준다.

## §2 098 −2 — 추천 내용 오류 (수리 회귀 아님 판정)

손님은 4/4 전부 제출했다. 실패 2건은 **틀린 계좌를 제출**: t0=Green Fee-Free, t2=Light Blue —
t2 에서 에이전트가 축자로 *"the 'Blue Account' does not seem to be explicitly listed in the knowledge base"* 라며
인접 계좌를 추천했다([30]). 통과 2건은 Blue Account + 보너스 근거($30/$20)까지 제시. ⇒ 기전 = **KB 미회수 시
인접 대체 추천** (F2b 최적조합 계산 + 회수 변동). 수리 1~4 와의 인과 연결 없음 — 잡음+검색 변동 부류.

## §3 070 — write 미도달의 실체 = 자격 축 탈락 + 자기서비스 날조

- **a#4 (t2 전멸형)**: [02] 첫 턴부터 `check_card_application_fit`(**신용카드** 도구)로 응답 → Cobalt Blue→Bluest→
  Light Blue→Lime Green→Light Blue **계정명 5회 표류**(매번 "I apologize... Let's proceed with...") → [38] **검증
  거짓말**(verify 성공 0·log_verification 0 인데 "successfully verified" 선언) → 최종 추천 **Light Blue(개인
  계좌)** → [42~45] **온라인 신청 링크 날조**·user_stop. 검색 4요구 질의가 개인/savings 문서(Silver Plus·Blue·
  Light Blue)만 회수했고 모델은 business 자격으로 거르지 않았다([[63]] 자격 축·C439 집합 밖 19/24 동형).
- **a#2 (t0)**: 가짜 user-tool `submit_transaction` 호출(Unknown) 후 체인 미착수.
- **a#7 (t3)**: [20] **자리표시자 날조 verify**("John Doe"/"123 Main St") · unlock@18→call@34 정지 ·
  open_bank_account 미도달. [32] 4요구 dense 질의가 **Gold Account(개인 savings)** 회수.
- t1 은 통과(사상 2번째) — 070 은 "체인 어디서 멈추나"의 분산이 크다.

## §4 071 — 두 모드: 오답 클래스 vs 완수-후-여분-write

- **t0/t1 (오답 클래스 모드)**: 에이전트가 Hunter Green(+Business Bronze Saver/Gold Plus Saver/Emerald Saver)을
  추천·개설. t0 는 그 전에 EIN 요구(**x277 의 인접-절차 오염 실물** — EIN 은 신용카드 신청 문서에 실재)·포털
  안내·"도구 없음" 거짓([38])까지의 26턴급 사가 후 [40]에서 스스로 unlock·개설(클래스는 오답 유지).
  t1 은 개설 3회가 전부 다른 오답 클래스 — 요구(모바일 입금 $10k/일·$0 초과인출·최소잔고<$10k·≥1% APY //
  same-day ACH·<$50k·wire≤$15)와 클래스 스펙의 대조가 어디서도 고정되지 않았다.
  ⚠071 결정-전달(CP2) 검증은 **불가능했다** — 사이드카 `kind=route` 가 `_text len=0` 으로 본문을 버림(§6.3).
- **t2/t3 (완수-후-붕괴 모드)**: gold 6/6 정확(클래스 정답 Sky Blue+Gold Saver Account) → **대본 밖** funding
  요청(시나리오는 개설 후 "all set"으로 종료·금액도 시행마다 상이: $500/$1000 vs $8000/$3000)에 순응 →
  transfer 2건 → db_match=false. t3 는 user 가 *"I can't run internal_transfer from my side ... initiate on your
  end"* 역제안. [[21]]에 따라 user-sim 면책 없음 — 단 처방은 [[66]] 접경(의도 판단)이라 **레버 설계 보류·기록만**.

## §5 계측 결함 (task-4 수리 표적 — 이번 포렌식이 실측으로 특정)

1. **사이드카 sim 지문 충돌**: `t2_fbsidecar._sim_key` = 첫 user 발화 SHA1[:12]. user-sim temp 0.0 이라 같은
   태스크의 trial 들이 **같은 첫 발화 → 한 키로 병합**(098 4 trial→`49872c…` 하나·fb 104줄). "071 t1 이 t3 에
   복제"(전 handoff §6.4)의 정체. results.json 조인은 첫 발화 재해시로 가능하나 **동태스크 trial 구분 불가**.
2. **deny 원문 ↔ 커밋 문면 상이**: 사이드카는 `Error: [POLICY GATE GB1...]` 원문, 궤적 커밋은 `[Note: the tool
   call(s) above were blocked...]` 재작성 — 게이트ID 기반 교차대조가 반쪽. (기록엔 커밋형도 남겨야 함.)
3. **`kind=route` `_text` len=0** — 071 CP2 검증을 원천 봉쇄한 실물.
4. **logmark 400줄 캡**.

## §6 레버 후보 (전부 **격리 후 출시** — [[62]]·[[66]] ③무측정 출시 금지)

- **R-A (GB2 recovery 공존형)**: "Send exactly this message" → "네 답장에 이 문장을 그대로 포함하라(다른 내용
  공존 가능)". 실측 근거 = §1.4 (정답 삭제 1건). 8-sim 격리(A 현행/B 공존형·8141) 후 출시.
- **R-B (user-측 표적 GO 1회)**: 우리가 CANNOT-YET 를 냈던 user-requestor 표적이 요건 공집합으로 **전이**하는
  순간, `cleared()` 동형 문장("that now holds — tell the customer to run X now") 1회. 결정론 상태전이 뿐·의도
  분류 0·[[63]] 해제 신호·[[64]] 해법 명시. 먼저 x283 격리(010 t0 prefix·A 현행 푸시/B +GO/C 무푸시) 설계.
- **R-F (verified-claim 검출)**: claimprov 술어에 "검증했다" 주장 추가(원장 대조·닫힌 술어) — 070 a#4 [38] 실물.
  후순위.
- **보류**: 071 transfer 억제(의도 판단 접경) · 070 자격 축(기존 x281 대조표 축과 통합 검토·부분회복 2/8 뿐).

## §7 q런 사전 예측 (2026-08-13 기입 · 결과 보기 전)

- 010: GO 부재 불변 → **0~1/4** 예상.
- 070: `T2_UNAVAIL` 부활(수리5 살아있음·q 로그 발화 1회 확인)이 포털/외부절차 날조를 물면 **t2 전멸형 감소** 기대.
- 071: t2/t3 형(대본 밖 funding 순응)은 레버가 없으므로 재발 가능.
