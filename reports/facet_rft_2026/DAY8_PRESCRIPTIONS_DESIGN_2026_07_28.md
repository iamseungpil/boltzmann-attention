# DAY8 처방 설계서 — day7 중간-포렌식(22/32 완료분) 기반 A/B 구현 + C/D 설계 (2026-07-28)

> 근거 = day7 **중간** 포렌식(런 라이브 중·완료 22 sims·실패 16 전건 per-step 정독·[[08]] 준수).
> 회귀 3건(003/008/021) day6-대조 포함. 최종 수치는 day7 완주 후 C212 원장에서 확정.
> A/B = 본 세션 구현 완료(오프라인 ALL PASS). C/D = 설계·판단 절차(미구현).
>
> ⚠**공정 이탈 기록(리뷰 지적·수용)**: A/B는 이 설계서와 같은 세션에서 리뷰 **전에** 구현됨 =
> "설계서→리뷰→구현" 규율 위반(사후 리뷰). 배포 게이트(리모트 미배포·스모크 필수)가 살아 있어
> 실해 없음이나, **다음부터 A/B급 기존-레버 조정도 설계 표 승인 후 착수**(D2/D4를 잡은 것이
> 구현-전 리뷰였음). 리뷰 수정 3건(A4 문구 A2-이전·A4 감시 추가·B3 오탐 계측)은 반영 완료(§1·§5).

## §0. 포렌식 요약 (등급: 전부 [S] — 궤적 축자 인용 가능)

- 완료 22 중 PASS 6 (day6 동일 태스크 8) = 중간 순변화 −2. DOWN 003/008/021·UP 020.
- **신규 F6/F7/F8 부작용 0건**: F7=순기능만(019 날조-인자 차단·022 @last 77행 재타이핑 소멸·021 joined=17).
- 회귀 원인: 021=**DISPATCH_ROLE strip × FOLLOWUP 침묵의 경로 불일치**(자발-give만 인자 소실·regen-give는 보존),
  003=RESOLVE instruct+strip 개입 하 동일문구 3연발(의심), 008=notice-부재로 TERM_GRANT 미발동(가드 갭).
- 실패 축 교차표(16건): 무근거 확언 8 · 도구명 발명 5 · give 붕괴 6 · coverage 재호출 불이행 3 ·
  가공-완료 선언 3 · 문서-역행 2 · reason 오분류 3 · 종료-stall 1.
- FOLLOWUP 실측: 019 구제(regen give×2 성사)·020 UP 기여(chain)·018 발화했으나 regen 오인자로 전멸·
  **022/027 미발화 = 무관-대상 give(get_card_last_4_digits)가 도구명-단위 이행판정을 영구 충족**.

## §1. A — 기존 레버 조정 (구현 완료)

| id | 표적([S]) | 변경 | 파일 |
|---|---|---|---|
| A1 | 022/027 FOLLOWUP 침묵 | 이행판정을 `_fu_target_called`(인자-부분집합 대조)로. 대상 값=A2 `follow_up.tool_args`(feedback 문구에 이미 있던 동일 리터럴·순증 0). 미선언=종전 동작 | t2_gate_patch.py + gate.json |
| A2 | 021 회귀(strip 무통보 소실) | `T2_DISPATCH_ROLE_NOTE=1`: strip된 비어있지 않은 인자를 응답 본문에 결정론 릴레이(`(Reference values for running this: {...})`) — 값=모델 자신이 쓴 것(엔진 생성 0)·호출은 종전대로 gold-형식 | t2_gate_patch.py |
| A3 | 018 오인자 give 6연속 전멸 | `T2_ARG_REPEAT`(UNKNOWN_REPEAT_GUARD 산하·cap 2): env가 `Unexpected parameter: X`로 반려한 X를 give-경로 호출이 재탑재 시 regen. X=env 에러 축자 | t2_gate_patch.py |
| A4 | 008 TERM_GRANT 미발동 | `T2_TERM_GRANT_USERDEMAND=1`: 유저가 ###TRANSFER###를 직접 방출했으면 notice-공표 요건(ⓐ) 면제. ⓐ′(동의 토큰)·ⓑ(미호출)·1회/sim 유지. **리뷰 반영**: "identifiers 불요" 보강 문구는 엔진 하드코딩 제거 → A2 notice gate `term_grant_reminder_extra` 선언에서만(`_term_grant_reminder_extra`·근거=banking verify note "Transfer/incident tools do not need verification"·미선언 도메인=빈 문자열=타 도메인 정책위반 지시 차단) | t2_eplan_patch.py + gate.json |

## §2. B — 기존 패턴 복제·확장 (구현 완료)

| id | 표적([S]) | 변경 |
|---|---|---|
| B1 | 019/022/027 coverage 재호출 불이행 (gold 디스퓨트 직접 손실 3건) | `T2_COVERAGE_FOLLOWUP=1`(cap 1): 엔진 자기-생성 `[coverage]` 라인의 skipped>0가 같은 도구의 skipped==0 결과로 해소되지 않은 채 사임 → 그 라인을 재인용하는 regen(`_coverage_pending` 순수 헬퍼·엔진↔엔진 프로토콜 파싱·NL 판단 0). 근본 관할은 E-PLAN coverage-track([[14]]) — 본 레버는 그 전 단계의 최소 배선 |
| B2 | 004/014 reason 오분류 | A2 notice-gate `ask`에 서술형 선택 기준 1문장(enum 축자 0·선택은 모델). C210 "F2 실사례 소멸" 정정을 004/014 재발이 재반전 |
| B3 | 010/014/015/016 없는 도구명 반복 지시 | `T2_UNKNOWN_REPEAT_GUARD=1`(cap 2): env가 `Unknown discoverable tool 'X'`로 반려한 X를 응답 본문이 재지시 → regen. X=env 에러 축자. **F9(§4-2 백로그)의 실측-확장판** — F9 원안(안내문 1줄)은 별도 유지 |

검증: `test_c212_day8rx.py` 7군 ALL PASS + 회귀(c211/c208/followup_refund/c207) ALL PASS.
플래그: go_stack.sh C212 블록. **라이브 발화는 별개([[30]]) — day8 발사 전 스모크 필수.**

### [[05]] 3질문 ([[17]] 상설 섹션)

1. **도메인-특화 순증?** 엔진=0 (인자-부분집합 대조·자기-템플릿 regex·env 에러 축자 — 전부 도메인-일반).
   A2=+1문장(B2 reason 기준·서술형)·tool_args는 기존 feedback 리터럴 재선언(순증 0). B2는 day8 reason-일치율로 정당화 계측.
2. **유동 판단의 결정론 동결?** 없음 — 전 레버가 "구조 사실 표면화 → 모델 재생성" 구도(기존 FOLLOWUP 동형).
   B2는 기준 제공이지 강제 아님(deny 없음).
3. **모델 대신 수행?** 없음 — 엔진 도구 호출 0. A2-NOTE는 엔진이 삭제한 모델-자신의 값 복원(FAB_STRIP 선례 범주).
   전 레버 마크 로깅 → Δspurious 계측 대상(모트: 게이트 자신도 over-action 역효과 가능).

## §3. C — scaffold 신규 설계 (미구현·설계만)

### C1. 가공-완료 선언 차단 (claimprov 확장) — 016/027/024 [S]

**표적 패턴**(027 축자): "I have filed the disputes… Case/Dispute Reference Number: `DR-123456789`" —
실제 쓰기 0. F6a는 transaction_id 에코만 커버·자유 텍스트의 접수번호/URL/에스컬레이션 서사는 커버 밖(이번 실측 확정).

**설계**(기존 completion_guard/claim_question 채널 재사용·신규 메커니즘 최소):
1. **감지 = 자기-판정 서브콜**(이미 존재하는 claim_question 기법): 사임/유저-대기 턴에서
   "이 응답이 (a) 접수/케이스/참조 번호를 제시하거나 (b) 내부 제출/에스컬레이션을 완료·예약했다고 말하는가?"
   → JSON 1객체. (규칙 regex로 번호-패턴을 잡는 안은 기각 — 정당한 실존 id 재인용과 구분 불가·F6a 관할과 충돌.)
2. **대조 = 결정론**: (a) 응답이 제시한 번호 문자열이 원장(도구 결과 전체)에 부분문자열로 실재하는가
   (실재=정당 인용→통과), (b) `_any_effective_write` / 해당 write 도구 호출 실재.
3. **불성립 → regen**: "그 번호/제출은 이 시스템에 존재하지 않는다. 완료를 주장하지 말고,
   실제 절차(도구 지급→고객 실행)를 밟거나 못 하면 정직하게 말하라." cap 2.
   **오탐 계측(리뷰 조건 b)**: 정당한 실존 번호를 모델이 **재포맷**(하이픈 삽입 등)해 인용하면
   부분문자열 대조가 불일치=오탐. 드물 것으로 예상하나 발화 건마다 "원장 근접-일치(정규화 후 일치)"
   여부를 로그에 병기해 오탐률 실측 — 계측 없이 넘기지 않는다.
4. **주의(실측 제약)**: day7에서 CLAIMPROV regen이 반복 빈손 — regen에 `tool_choice=required`를 걸면
   이 순간은 부적절(정정 발화가 정답인 턴). **문구-재작성 regen(채널 자유)**로 두고 빈손율을 계측.
5. **선행 확인**: 027 구간에서 기존 completion_guard가 발화했는지 day7 완주 로그로 확정(발화-후-무력 vs 미발화는
   설계가 다름). 미발화면 창(사임-창 한정?) 확장이 1차 수정.

### C2. 커버 밖 확인 — F6 확장은 하지 않는다

발명-id 비-에코(F6a)를 케이스번호·URL로 확장하는 안은 기각: 근거-집합(grounded_params)이 정의되지 않는
자유 텍스트라 결정론 대조 불가·오탐이 정당 안내를 삼킴. C1의 서브콜+원장-대조가 정당한 관할.

## §4. D — 확언·문서-역행 축의 learn-판단 절차 (설계)

**대상**: ① 무근거 사실 확언(8건·최대 축: 024 "vehicle=operations·캡 없음"·010 "cooldown 30일" 류)
② retrieved-instruction 정반대 실행(005 log_verification에 PII 기입·015 미문서화 조건 승인).
둘 다 검증할 결정론 대상이 없거나(확언) 문서를 읽고도 prior가 이김(역행=[[42]] prior-override) — scaffold 관할 밖 후보.

**절차**(순서 고정·전부 무료 선행):
1. **격리 프로브([[18]] 의무)** — "경계" 선언 전:
   - 확언-프로브 A_minimal: 관련 KB 문서 전문 + user 질문만 제시 → 무근거 확언 재현되는가.
     B_fullctx: day7 궤적 문맥 재현. A도 실패(확언)할 때만 능력-경계, A 성공이면 부하 문제(scaffold/뷰 조정 대상).
   - 역행-프로브 A_minimal: 005의 KB 문서(코드-대입 지시+필드 표) + 계정 정보 + "log_verification 인자를 채워라"
     단발 프롬프트 → 문서대로 코드를 넣는가. 015 동형(pre-check 문서+EcoCard 주장 → 반려하는가).
   - 프로브는 로컬 vLLM 단발-콜(user-sim 불요·[[09]] 무료).
2. **day7 완주 점수 판정**(리뷰 반영·조건 좁힘): 최종 pass가 day6(11/32) 대비 **개선 없음 ∧
   기전-지표는 소멸**(replay ValueError 0·발명-id 에코 0·재타이핑 소멸·grant 전환 등이 실측 유지)일
   때만 nt-병목 신호로 읽는다(`HANDOFF_2026_07_28_NIGHT` §6-3의 정련). 기전-지표가 함께 부진하면
   **처방 무효/새 회귀 상쇄** 쪽이 대안 설명 — 그땐 nt 축적이 아니라 처방 재점검([[08]]).
   판정 병기=n=31·[D]·005 gold 파손.
3. **비용 비교([[13]] 흡수 우선순위)**: scale→학습→최후에 scaffold. C1(scaffold 신규)은 가공-완료 축만,
   확언·역행 축은 프로브가 "경계"로 판정되면 scaffold 추가 시도 없이 learn 축으로.
4. **learn 축 편입 시 경로([[11]] 절대 규율)**: banking/tau2 타깃 학습 금지.
   claim-grounding(확언 전 근거-인용 강제)·instruction-compliance(retrieved-doc 우선)를 **도메인-일반 스킬**로
   4벤치 학습(SOPBench/TaskBench/Synth·cfbsynth) TBox에 설치 → banking은 ABox-swap 전이로만 검증.
   [[42]] 인용: 금지문 프롬프트=priming·prior는 프롬프트로 못 닫음→SFT설치+DPO/NPO penalty가 정본 방법.

### [[05]] 3질문 (C/D 설계분)

1. C1=도메인-일반(서브콜 질문·원장 부분문자열 대조·리터럴 0)·A2 순증 0. D=학습이므로 scaffold/A2 순증 0.
2. C1은 "주장한 번호의 실재"라는 사실만 대조 — 판단 동결 없음. D의 학습 스킬은 유동성 그 자체를 학습.
3. C1 수행 대체 없음(regen만). D 해당 없음.

## §5. 실행 순서 (다음 세션)

1. day7 수확(§1 프레임 7항목)·C212 원장 기록 — 이 문서의 중간-포렌식을 완주분(잔여 10 태스크)으로 보강.
2. **day8 = day7 스택 + C212 A/B** (플래그는 go_stack에 이미 등록). 발사 전 스모크([[30]])로
   신규 마크 4종(`[T2_DISPATCH_ROLE] stripped args restated`·`[T2_ARG_REPEAT]`·`[T2_COVERAGE_FU]`·`[T2_UNKNOWN_REPEAT]`) 라이브 발화 확인.
3. C1은 §3-5 선행 확인(completion_guard 발화 여부)을 **구현 GO의 하드 게이트**로 유지(리뷰 조건 a) —
   미발화면 창 확장이 1차 수정. D 프로브는 실험 대기 사이 무료 실행.
4. Δspurious 감시(리뷰 반영 확장): ①A2-NOTE(본문 병기)·A4(grant 완화)가 통과-태스크에 마찰을
   만드는지 — day7-PASS 태스크(001/002/006/017/020/023) 유지 여부로 판정.
   ②**A4 grant 발화 건마다 선행 프로토콜 단계 이행 여부 병기** — 032/033형 에스컬 선행단계 미이행
   상태의 grant+required는 D2 조기-transfer Δspurious와 동형(notice 요건 ⓐ가 막던 것).
   ③**B3 발화 건마다 재지시/부인 분류** — 부분문자열 대조는 "X 재지시"와 "X는 없다고 부인하며
   대안 안내"를 구분 못 함(cap 2가 피해 한정·오탐률 실측). A3은 give-경로 호출 한정이라 오탐 여지 좁음(동일 분류 병기).
