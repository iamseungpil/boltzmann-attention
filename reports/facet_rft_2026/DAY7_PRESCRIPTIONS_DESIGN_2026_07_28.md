# day7 처방 설계서 — F6·F7(BYREF ON)·F8 (2026-07-28)

> 입력: `DAY6_FORENSIC_AND_FIXES_2026_07_28.md`(C210) · day6 도달성 선검증(본 문서 §각주).
> 상태: **설계 단계** — 리뷰 후 구현(설계서→리뷰→구현). F1~F5는 구현·배포 완료(C210)·본 문서 스코프 밖.
> 명명: F6(발명-id)·F7(참조-전달 ON+조인)·F8(인자-생산자 give 흐름). day7 런 계획 §5.

---

## §0. [[05]] 결정-시점 3질문 (상설)

| 질문 | F6 | F7(a/b) | F8 |
|---|---|---|---|
| ①도메인-특화 순증? | A2 선언 +1(`grounded_params.transaction_id`=**기존 P4b 스키마 재사용**·엔진 0)+엔진 버그픽스(inner-key) | F7a=0(승격만)·**F7b=A2 조인-spec +1**(도메인 필드명은 전부 A2에·엔진=일반 equijoin 실행기) | **A2 선언 +1**(`arg_producers`)·엔진=선언 대조+문구 |
| ②유동 판단 동결? | no — 실재 검증(값이 레코드 출력에 있나) | no — 전사·조인=concrete 이동([[10]]) | no — "이 인자는 손님 도구가 만든다"는 **env 사실**의 표면화(문구 지시·give는 모델이 emit) |
| ③도메인 행동 수행? | no | no(fetch=모델·커밋된 출력만 재사용·C101 선례) | no(give 호출=모델) |

전부 측정 근거 있음: F6=020/028 발명-id 제출 [S] · F7=W-f 실측(재호출 7,514/4,140자 재타이핑)=C209 설계서의 GO 신호 충족 · F8=040/041 오도구 전환 [S]+VALUE_ACQUIRE 원리적 불발 확인.

---

## §F6. 발명 txn id 차단 — "엔진이 날조를 에코해 정당화하는" 경로 봉쇄

**근본원인(도달성 선검증 [S] — 초판 가설 2건 정정)**:
- 발명 id(`txn_e0h5i0j0k004`·`txn_4c5d6e7f8g9h`)의 **첫 등장 = compute 페이로드(assistant tool_call)** —
  모델이 존재하지 않는 거래 행을 지어 넣었고, `get_reward_discrepancies`가 그 행을 판정해 **discrepant로 에코**
  → 우리 도구의 출력이 날조에 권위를 부여 → 이후 give/유저-실행 dispute까지 전파. ~~"give 인자에서 첫 발명"~~ 정정.
- FAB_STRIP이 못 잡은 이유 2중: ⓐ `_PROCEDURAL_RE`에 `discoverable|^give_` 면제 → give/디스패처 외피가 통째 면제
  ⓑ inner-key 목록(`agent_tool_name|user_tool_name`)에 **`discoverable_tool_name` 부재** → give의 실효 이름을
  못 벗겨 외피 이름으로 면제 판정. (`t2_gate_patch.py:4413~4415`·`:1653`)

**설계 (2부·전부 기존 레버의 도달 수정)**:
1. **F6a — 페이로드 행-정체 grounding(주 방어)**: A2 ratefix `grounded_params`에
   `"transaction_id": {"producer_contains": ["credit_card_transaction_history"]}` 추가(P4b 스키마 그대로).
   엔진 소수정 1건: **id_field가 결핍(강등 포함)인 행은 판정에서 제외하고 skipped+missing_fields로 계상**
   — [소소1 반영·라인 인용] 현행 `t2_compute.py:638` `out_ids.append(r.get(idf))`는 id 값을 무검사
   append하므로 P4b 강등(id=None) 행이 discrepant로 판정되면 **None이 결과에 실린다**(선검증 [S]).
   효과: 발명 행 → "N rows could not be verified — missing/ungrounded field(s): transaction_id" 지목 →
   **에코-정당화 소멸**. txn id는 장문 유일 문자열이라 값-실재 검사가 강함(D4 폐기 사유였던
   '흔한 값 우연 실재'가 성립 안 함).
2. **F6b — FAB_STRIP give-구멍 수리(보조 방어·[리뷰 가드레일 반영] 공유 술어 불변·국소 수정)**:
   수리는 **FAB_STRIP의 지역 함수 `_fab_write_ungrounded` 안에서만** — 공유 `_PROCEDURAL_RE`(1653)와
   `_eff_tool_name`(1659)은 **불변**(둘은 `_is_effective_write`→claim_prov write축·WRITEPROV·readloop의
   공통 기반이고, give/unlock/discoverable의 procedural 면제는 2026-07-18 교정으로 의도된 의미론).
   ⓐ 4413 inner-key 목록에 `discoverable_tool_name` 추가 ⓑ **give/call 디스패처에 한해** inner가 있으면
   inner 이름으로만 면제 판정(외피 이름 면제 금지·unlock은 inner-판정 제외=안전측).
   회귀 필수: `_is_effective_write("give_discoverable_user_tool")=False` **유지** 케이스.
   → day5 022형(give 인자에 nested txn) 어시스턴트-측 차단 복원. 유저-측 실행은 차단 불가(D1 교훈:
   유저 호출 deny=replay 폭탄)·어시스턴트의 give/SAY 단계가 유일한 개입점임을 명기.

**검증(오프라인)**: 020 실페이로드(발명 2행 포함) → transaction_id 2행 지목+비-에코 확인 ·
day5 022 give-인자 픽스처 → FAB_STRIP 검출 · 정상 give(gold 인자) → 무개입 · 기존 FAB_STRIP 회귀.
**한계(정직)**: 발명 id가 **산문(SAY)으로만** 전달되고 유저가 그대로 실행하는 경로는 잔존 —
F6a가 상류(discrepant 목록)를 막아 발생 확률을 낮추지만 구조 봉쇄는 아님(후보: claimprov kind 확장·스코프 밖).

## §F7. 참조-전달 승격(BYREF ON) + A2 조인-spec — coverage 값-층의 완결형

**GO 근거**: C209 설계서 리뷰 필수3 = "W-f 실측이 P6 ON의 GO 신호" → day6 실측 확보
(027 지목 후 재호출 args=7,514자·026=4,140자 = 재타이핑 경로 실증). PROD_BIND 대규모 적발(022: 73/77행 날조)도
"재타이핑=날조 표면"임을 재확인.

**F7a — 승격**: go_stack `T2_SG_BYREF=1`. A2 param 문구는 C209 구현이 이미 조건부 부가(ON일 때만).

**F7b — 조인-spec (신규 설계·리뷰 포인트)**: byref로 해석된 rows는 **거래 덤프 유래라 `account_open`이 없다**
— C209의 "비-over 인자 참조 미지원" 결정과 promo 판정 요건이 충돌(이대로 켜면 byref 사용 시 promo 행이
P4 abstain으로 회귀). 해소 = **A2-선언 일반 equijoin**:

```json
"byref_join": {
  "account_open": {
    "from_ref_param": "account_open",            ← 모델이 account_open="@last:<accounts getter>"로 참조
    "source_selector": "credit_card_accounts",   ← 참조 출력의 정당성 검사(P4b selector 재사용)
    "match": {"row_field": "credit_card_type", "source_field": "card_type"},
    "take": "date_of_account_open"
  }
}
```
- 엔진 = **일반 equijoin 실행기**(row_field==source_field인 소스 행에서 take 필드 복사·필드명 전부 A2 데이터·
  파서는 기존 `Record ID:` 전용 재사용). 조인 불성립 행 = account_open 미기입(=P4 지목 경로·안전측).
- **[명세 보강1 반영] 복수-매칭 처리**: 소스에 같은 match 값의 행이 2개 이상이면 조인 모호 —
  "첫 행" 채택은 침묵-오값 경로(D4형)를 연다. **유일 매칭만 유효·복수 매칭=불성립**(조인 불성립과 동일
  경로=abstain 안전측). §6 테스트에 복수-매칭 픽스처 필수.
- [[05]] ①: A2 성장 +1 — 정당화 = 이 조인이 없으면 byref와 promo 판정이 양립 불가(측정: day6 BS 전멸 지속).
  ②: 조인=결정론 값-이동(판단 0). ③: 참조 대상은 **모델이 읽어 커밋한** 출력뿐.
- **비-over 스칼라/필드 참조 일반 허용은 계속 미지원**(C209 결정 유지) — 조인은 `byref_join` 선언이 있는
  필드에 한정(경계 좁게).

**검증(오프라인)**: day6 026/027 실궤적 재생 — transactions="@last:…"+account_open="@last:accounts" 시
26행 전원 판정(BS 14행 promo 포함)·gold 4건 재현(이번엔 **실개설일 02/13 기반**·027의 날조-운 제거 검증) ·
조인 불성립(카드명 불일치) 픽스처 → 해당 행만 abstain · byref OFF 회귀 무변.
**간섭 감시 W-h**: byref rows → isolate 서브 operand 병합 순서(byref 해석→isolate→P4b→op) 로그로 확인 ·
W-i: P4 지목 문구+byref 안내가 같은 턴에 겹칠 때 재호출이 참조식으로 오는지(발화율 계측=F7 효능 지표).

## §F8. 인자-생산자 give 흐름 — 040/041 오도구 전환의 상류 봉쇄

**근본원인 [S]**: formal dispute 도구가 `card_last_4_digits` 필수 → 값은 **유저-측 도구**(get_card_last_4_digits)만
생산 → 모델이 give 흐름을 못 밟고(day5: env가 자구로 알려줘도 불응·day6: 인자 없는 cash-back dispute로
**경로-최소저항 우회**) → 오도구 제출. 기존 VALUE_ACQUIRE는 **producer 출력이 이미 실재**할 때만 발화하는
술어라 원리적 불발(도달성 검증: day6 발화=035·018뿐).

**설계 — A2 `arg_producers` 선언 + 결핍-인자 트리거**:
- A2(도메인 데이터): `"arg_producers": {"card_last_4_digits": {"user_tool": "get_card_last_4_digits"}}`.
- 엔진 트리거(도메인-일반): 도구 실행 에러 content에 "required"류 + A2 선언 인자명이 **부분문자열로 등장**하면
  view-fb(K=2·F5 채널): "Argument '{arg}' is produced by the customer-side tool '{tool}'. Hand it to the
  customer with give_discoverable_user_tool, ask them to run it, then retry the SAME tool with the value —
  do NOT switch to a different dispute tool because an argument is missing."
- 마지막 문장이 040/041의 전환-우회를 직접 겨냥(도구명 미지정=선택은 모델·"same tool" 지시는 이미 모델이
  고른 도구의 유지 요구라 gold-planting 아님).
- **발화 조건 좁게**: 에러 원문에 인자명 실재(A2 선언 대조)일 때만·cap 2/sim·give 이후엔 무발화.
- **[명세 보강2 반영] 인자-공유 오탐 감시**: 트리거가 인자명 기준이라 복수 도구가 같은 인자를 공유하면
  "retry the SAME tool" 넛지가 **오선택을 고착**시킬 수 있다(모델이 틀린 도구를 골라 그 인자 에러를 냈을 때).
  현 도메인은 `card_last_4_digits`=formal dispute 전용이라 실질 무해로 보나 가정을 실측으로: §5 판정 ⑥의
  세부 지표로 **"트리거 발화 시점의 도구 = gold 도구 일치 여부"**를 계측(`[T2_ARG_PRODUCERS] fired
  tool=<명>` 로그 → 포렌식 시 gold 대조).

**검증(오프라인)**: 040 실궤적의 미싱-인자 에러 픽스처 → 발화+문구 · 인자명 무관 에러 → 무발화 ·
give 완료 후 → 무발화 · retail A2(선언 없음) → 전역 무개입.

## §4. 스코프 밖 (정직한 경계 — day6 §2e 재발형)

012/014/015(주장-KB 대조·우회안내)·016(조기 에스컬)·024(fit 후 오선택)·007(사임)·028 rate 품질(C185a 트랙)·
027 과행동 벌점(측정 재료로 보존— 교정 시 Δspurious 역효과 리스크가 더 큼) = scale/learn 축 유지.
005 = n=31 처리 확정(P9a).

## §5. day7 런 계획

- 스택: day6 + F1~F5(배포됨) + F6 + **F7a/b(BYREF ON)** + F8. OFF 유지: NEARDUP.
- 판정(기전 1차·점수 [D]): ①replay ValueError 0(F1 확증) ②004형 grant 발화→transfer 완결 전환율(F2+F4)
  ③033형 view-fb 2회 노출 후 준수율(F5) ④발명-id 지목 발화·에코 0(F6) ⑤byref 사용률·재타이핑 재호출 소멸(W-f 역전=F7)
  ⑥040/041형 give-흐름 진입(F8) ⑦coverage 완전판정률(BS promo 행 판정 수).
- 비용: front32·conc1·day6 동일(유료·발사는 별도 승인).

## §6. 오프라인 테스트 목록

`test_c211_day7rx.py`: F6a(발명행 지목·비-에코·id_field 결핍 skip)·F6b(give nested 검출+정상 무개입)·
F7b(equijoin 재생 026/027·불성립 abstain·OFF 회귀)·F8(트리거 3분기) + 기존 배터리 전체 회귀.
