# C축(learn 잔여)·D축(하네스/gold 아티팩트) 설계서 (초안·승인 전 구현/학습 금지) — 2026-07-29

> 발단=실패-예상 23건 per-step 재분류: **A(결정론) 6 / B(넛지) 4 / C(learn) 9 / D(아티팩트) 2**.
> A·B는 C213/C214로 구현·day9c 검증 중. 이 문서는 **C·D 두 축**을 다룬다.
> ⚠**명명 주의**: `DAY8_PRESCRIPTIONS_DESIGN`의 C(claimprov 확장)·D(learn 판단 절차)와 **글자가 겹치지만 다른 축**이다. 본 문서 기준 = 23건 분류의 C(learn 잔여)·D(아티팩트). DAY8의 C1(가공-완료 차단)은 여기 C축의 §3-3 유형과 표적이 겹치므로 §4에서 관계를 명시.

## §1. C축 대상 9건과 하위 유형

| 유형 | 태스크 | 관찰 (전부 [S]·궤적 인용 가능) |
|---|---|---|
| **C-i 무근거 확언** | 024·022·010·004 | 024 "vehicle=operations·캡 없음·총 $1,000"(소스 문서 0회 열람)→user가 그 확언대로 **오선택 write** / 022 KB 미검색 상태로 "dispute 도구 없음" 단정 / 010 "cooldown 30일" 날조 / 004 존재하지 않는 SMS 코드 절차 |
| **C-ii 문서-역행** | 015·018·(005 부분) | 015 pre-check 문서("미문서화 조건이면 반려·도구 제공 금지")를 쥐고 EcoCard 승인 / 018 KB가 명시한 dispute 도구를 "없다"고 부인 / 005 "코드를 넣어라·개인정보 금지" 문서를 3회 읽고 PII 기입 |
| **C-iii 오독·발명** | 016·027·040 | 016 IN_PROGRESS를 COMPLETE로 오독→"지급 지연" 프레임 고착 / 027 가짜 케이스번호 3개(DR-\*)+가공 타임라인 / 040 실재 base명+타 문서 서픽스 접합(`..._7834`) |

공통점: **정답이 이미 컨텍스트에 있는 상태에서 실패**(028형과 동형). 040은 DUPLICATE 격상 경고에도 불응.

## §2. 왜 scaffold 밖인가 (경계 정본 §3 대조)

- C-i: "이 확언이 근거가 있는가"는 **열린 술어** — 대조할 결정론적 대상이 정의되지 않음(자유 텍스트의 사실 주장). 부분 예외=수치·id·접수번호처럼 **원장 대조가 가능한 조각**(→C1 관할·§4 참조).
- C-ii: 술어("문서를 읽었는가")는 닫히지만 **처방**("문서대로 하라")은 내용 해석이라 강제 불가. 문서를 재부각(BRANCH_REGROUND)해도 day7 015/018에서 불응 실측.
- C-iii: 오독은 의미 판정, 발명 도구명은 **호출 전에는** 대조 대상이 없음(B3는 env가 반려한 **후**에만 작동·040은 호출 자체가 0회라 커버 밖).
⇒ [[45]]/[[13]] 정합: scale-비의존 잔여가 아니라 **모델 내부 절차 준수**의 문제 → learn 축 후보.

## §3. 격리 프로브 (구현 전 의무·[[18]]·전부 무료 로컬 단발콜)

목적: **능력 경계 vs 부하** 판별. A_minimal도 실패해야 "경계"(learn 표적), A 성공이면 부하(뷰·scaffold 조정 대상).

| 프로브 | A_minimal 구성 | B_fullctx | 판정 |
|---|---|---|---|
| P-i (확언) | 해당 KB 문서 전문 + user 질문 1개 + "근거 없으면 모른다고 답하라" 없이 중립 지시 | day7/8 궤적 문맥 재현 | A에서도 무근거 확언 → 경계 |
| P-ii (역행) | 005 KB 문서(코드-대입 지시+필드표) + 계정 레코드 + "log_verification 인자를 채워라" 단발 / 015 pre-check 문서 + EcoCard 주장 | 동 궤적 | A에서도 문서 반대 실행 → 경계 |
| P-iii (오독) | referral 레코드 15건(최신=IN_PROGRESS) + "보너스 지급됐나?" | 동 궤적 | A에서도 COMPLETE 오독 → 경계 |
| P-iv (발명) | 도구 목록 없이 "dispute 도구 이름을 KB에서 찾아라" + 서픽스 다른 문서 2개 | 동 궤적 | A에서도 서픽스 접합 → 경계 |

### §3.1 표본 설계 (리뷰 필수1 반영·초판의 "3변형×3회" 폐기)

초판은 `n≥9 = 문구 3변형 × 3회`였으나 **우리 스택은 temp 0(그리디)** 이라 같은 프롬프트 반복은 배칭 수치 노이즈를 빼면 동일 출력 = **실질 표본 3**. 반복 차원이 정보를 주려면 temp>0이어야 하고 그러면 배포 레짐과 달라져 대표성이 깨진다. ⇒ **n≥9 = 서로 다른 변형 9개**(문구 3 × 문서 배치/제시 순서 3의 교차), **각 1회·temp 0 유지**. [[12]] 다양성 요건의 취지와도 이쪽이 정합.

### §3.2 판정 임계 (리뷰 필수1-①)

A_minimal 9변형 중 실패 건수 기준: **0 = 부하**(scaffold/뷰 조정 대상) · **1~4 = 혼합**(별도 보고·유형 분해 후 재프로브) · **≥5(과반) = 경계**(learn 표적). 혼합은 learn 편입 근거로 쓰지 않는다.

### §3.3 B_fullctx의 역할 = 원 관찰 재현 확인 (리뷰 필수1-②)

B에서 day7/8의 원 실패가 **재현되지 않으면** 그 관찰 자체가 표집 노이즈일 수 있다(C192 user-sim 비결정 전례). 이 경우 **경계/부하 판정 이전에 원 관찰의 등급을 강등**([S]→[M] 또는 [?])하고 판정 보류 — 프로브 결과로 learn을 논하지 않는다.

- 실행: 로컬 vLLM 단발콜(비용 0). **⚠day9c와 같은 vLLM 인스턴스면 큐 경합**(day6 §6B 큐-포화 아티팩트=양 arm 동시 timeout의 재현 경로) → **직렬·스로틀 실행 또는 day9c 완주 후**. 총 36콜이라도 infra 원인을 우리 손으로 만들지 않는다.
- **선행 필수**: 프로브 없이 "learn 필요" 단정 금지(day5~8에서 반복 교정된 규율).

## §4. C1(가공-완료 차단)과의 관계 — 중복 방지

C-iii의 027(가짜 번호)은 **DAY8 §3의 C1이 관할**(서브콜 감지 + 원장 부분문자열 대조 + regen). 즉 C축 9건 중 **027은 scaffold 시도가 먼저**이고, C1이 실패하거나 오탐이 크면 그때 learn으로 넘긴다. 024의 가짜 URL·016의 가공 에스컬레이션도 같은 처리(원장 대조 가능한 조각). **learn 전용으로 남는 것은 순수 사실-확언(C-i의 rate/자격 판단)·문서-역행(C-ii)·의미 오독(C-iii 016)**.

## §5. learn 편입 경로 ([[11]] 절대 규율)

1. **금지**: banking/tau2 타깃 학습·gold 맞추기([[03b]]). 프로브 데이터도 banking 문항 그대로 학습 금지.
2. **스킬 2종(도메인-일반)**: ⓐ **claim-grounding** = 사실 주장 전 근거(도구 출력·문서 축자)를 인용하거나 기권 ⓑ **instruction-compliance** = 회수된 절차 문서가 파라미터/금지를 명시하면 prior보다 우선.
3. **학습 벤치**: SOPBench(control-flow)·TaskBench(data-flow)·Synth(content-op)+cfbsynth — [[01]] 4벤치에서 위 두 스킬의 P-primitive를 생성·학습 → 통합 TBox → **banking은 ABox-swap 전이로만 평가**.
4. **방법**([[42]] 정본): 프롬프트/금지문으로는 prior가 안 닫힘(실측·priming 역효과) → **SFT 설치 + DPO/NPO penalty**(확언-무근거 vs 기권 쌍, 문서-역행 vs 준수 쌍). 표현·구조 다양성 필수([[12]]).
5. **게이트**: §3 프로브가 "경계"로 판정된 유형만 편입. 부하로 나오면 scaffold/뷰 조정으로 회귀.

## §6. D축 — 하네스/gold 아티팩트 2건

### D-1. 005 gold 파손 (확정·재조사 금지)
원리상 통과 불가. 처리=**n=31 병기**(P9a 채택분) — 모든 보고에 유지.

### D-2. **[S 확정·신규] nested-JSON 인자의 원시 문자열 비교**

**근거 발췌(리뷰 보강2·provenance 문서-내 완결)** — `tau2-bench/src/tau2/data_model/tasks.py:178`:
```python
    def compare_with_tool_call(self, tool_call: ToolCall) -> bool:
        if self.name != tool_call.name:
            return False
        if self.compare_args is None:
            compare_args = tool_call.arguments.keys()
        else:
            compare_args = self.compare_args
        if len(compare_args) == 0:
            return True
        tool_args = {k: v for k, v in tool_call.arguments.items() if k in compare_args}
        action_args = {k: v for k, v in self.arguments.items() if k in compare_args}
        return tool_args == action_args        # ← 중첩 JSON은 '문자열'로 비교됨
```
`compare_args=None`이면 **인자 dict 전체를 `==`로 비교**한다. `call_discoverable_user_tool`의 `arguments`는 **JSON 문자열**이므로,
`'{"user_id": "890389b165", "transaction_id": "txn_…"}'`(gold·공백 있음) vs `'{"user_id":"890389b165",…}'`(실제·공백 없음)은 **의미가 같아도 불일치** → `action_match=false`.
day7 028 실측: user가 dispute 6건을 전부 성사시켰는데 028_2~7이 전부 false.

**처치 규율(중요)**:
- ⛔ **평가기 수정 금지** — 벤치 불변식([[03b]]·리더보드 비교성).
- ⛔ **에이전트에게 gold 공백/키순서를 맞추도록 지시·학습 금지** — gold-fitting=cheating.
- ✅ **계량 후 병기 보고**: ⓐ census(무료) = 전 태스크 gold action 중 `call_discoverable_*` + `arguments` 문자열 + `compare_args=null`인 건수, 그중 **reward_basis에 ACTION 포함**인 태스크 목록 ⓑ 해당 태스크는 결과 표에 각주 ⓒ db_match-basis 태스크는 무영향(DB 상태 비교)이므로 구분 표기.
- ✅ **counterfactual 재채점 열(리뷰 보강1)**: 과거 궤적의 gold vs 실제 인자를 **`json.loads` 후 dict 비교로 재채점**한 열을 census에 병기(평가기 수정이 아니라 별도 계측이므로 규율 안). 각주가 "영향 가능"→**"영향 실측 N건"**으로 강해지고, 깎기/살려주기 양방향도 같은 계산에서 나온다.
- **1차 실측(029/041 포렌식 [S])**: 전 결과파일 **nested-arg 액션 328건 중 매치 6건(1.8%)** — 매치된 사례(day6B/021)는 실제 호출이 gold와 동일하게 **공백이 있는** JSON이었다. 029_2~7 6건, 041_4~7 4건이 **전부 공백-단독 불일치**로 확인. ⇒ **action-기반 지표는 사실상 무의미**하고, 이 문자열은 **user-sim이 생성**하므로 에이전트가 통제할 수도 없다(=gold-fitting 대상조차 못 됨).

### D-3. **[M·신규·중대] db_match 해시에 read-only discoverable CALLED가 포함**

029는 **6건 디스퓨트 제출 집합이 gold와 정확히 일치**하고 rewards 미갱신도 gold와 같은데 `db_match=false`다. 권위본(`dbdiff_task.py` docstring·C149)은 db_match를 **strict full-DB 해시**로 규정하고 banking은 `agent_discoverable_tools`(호출된 discoverable 도구 CALLED)가 해시에 포함된다고 명시한다. 029에서 gold 대비 남는 유일한 DB-영향 차이는 **`get_user_dispute_history_7291` CALLED 2건**(gold action list에는 어떤 상태확인 read도 없음).
교차검증(전부 반증 0): 반복 give·성공 unlock·실패 call은 해시에 무영향(task_020 db=True 유지)·dispute id는 결정론(3런 동일)이라 순서/난수 가설 배제.
- ⚠**자기모순 gold**: 029의 gold notes는 "Agent must verify dispute status"를 요구하는데 gold action에는 그 read가 없다 → **시키는 대로 확인하면 해시가 깨진다.**
- ⚠**실험 해석 오염 위험**: read 레버를 켤수록 db_match가 떨어지는 **역상관**이 생길 수 있다 — [[19]] 합성 런 해석 시 반드시 통제. day6~9 결과 재해석 시 이 요인을 분리할 것.
- 처치 규율은 D-2와 동일(평가기·db_check 수정 금지) + **계량**: read-only CALLED만 다른 태스크 수를 census에 포함.

## §6.5. 029/041 포렌식이 낳은 **A축 신규 표적** (이 문서 밖·day10 처방 후보)

C/D 분류 중 나온 결정론-가능 항목이라 여기 기록만 하고 처방은 별도 설계서로:
1. **★쓰기-인자 근거 결속 — 귀속 정정 후 재설계(최우선·단 착수 게이트 있음)**
   041 관찰: 16행 거래목록이 컨텍스트에 2회 있었는데도 **12건의 transaction_id를 `txn_1234abcd` 류 자리표시자로 창작**+2건 오배정 후 **16회 실쓰기**. env는 미존재 id를 검증 없이 수락.
   ⚠**초판 귀속 2건 정정(리뷰 필수·코드/로그 실측)**:
   - ⓐ `T2_WRITEPROV`는 **선언-기반 완료-주장 게이트**(`claim_question`·`_any_effective_write`)이지 인자-단위 provenance가 아니다 → 041의 관할 레버가 아님. 인자-단위 관할은 **FAB_STRIP**이고, 이미 중첩 `arguments`를 `json.loads`로 파싱하며 inner-key에 `agent_tool_name`이 처음부터 있다(`t2_gate_patch.py:4522~4553`). 즉 사정거리 안.
   - ⓑ 포렌식의 "148스텝 provenance 개입 **0건**"은 **오류**. day7B 최종 로그 실측: 041 구간(4015행~)에 **`[T2_FAB_STRIP] dropped 1 ungrounded write call(s)` ×3**(로그 전체 FAB_STRIP 3건이 전부 041) + `[T2_PROV]`15·`[T2_WRITE_EVIDENCE]`9·`[T2_WRITEPROV]`7·`[T2_VALUE_ACQUIRE]`3.
   ⇒ **진짜 질문 = "왜 16건 중 3건만 잡았나"**(무발화가 아니라 **부분 발화**). 규명 갈래(day10 착수 전 하드 게이트):
   1. `T2_FAB_STRIP=1` — go_stack에 실재 ✔(확인 완료)
   2. requestor — 041의 16회 쓰기는 assistant `call_discoverable_agent_tool` ✔(FAB_STRIP의 `am.tool_calls` 사정거리 안)
   3. `_hint_hit(k, hints)`가 중첩 dict의 `transaction_id`를 잡는가 — 미확인
   4. **★유력 가설 = ctx 세탁**: `_ctx_from_messages`(1042행)는 **user·tool 메시지**를 ctx에 넣는다. 041에서 ⓐ env 성공 응답이 `Transaction: txn_1234abcd`를 **에코**하고 ⓑ user-sim이 발명 id를 [79][91][95][109][117][131][139]에서 **복창**했다 → 최초 1~3건은 걸리고(=실측 3건) 이후 같은/유사 id는 **ctx에 실재**하게 되어 통과. 즉 구멍은 "미적용"이 아니라 **근거-원천의 오염**이며, C211/F6a가 compute 페이로드 에코에 대해 고친 것과 **동형 문제의 다른 채널**([[21]] user-sim 임의 반응 원칙과 직결).
   ⇒ 처방 방향(설계는 별도): id형 인자의 grounding 원천을 **도구 출력(레코드)로 한정**하고 user 산문·env 성공 에코는 근거에서 배제 — 단 user가 정당하게 제공하는 값(이메일·DOB 등)은 유지해야 하므로 **필드-종류별 원천 규칙**이 필요. 착수 전 갈래 3·4 실측 확정 필수.
2. **DUPLICATE-READ 경고 → 하드 차단 (조건 보강·리뷰 권고 반영)**: 029 [12][14][16] 동일 쿼리 3연속·리다이렉트 3회 무시 실측([[07]] 재확증). **단 무조건 차단은 D-3와 충돌** — 상태-확인 read는 상태가 바뀌면 정당한 재호출이고, 029 gold notes가 바로 그 verify를 요구한다. ⇒ 술어를 **"동일 쿼리 3회 ∧ 그 사이 실효 write 0(상태 불변)"** 으로 좁힌다(상태 불변=원장에서 닫히게 판정 가능). 이 조건이 폴링-정당 케이스와 029형 루프를 가른다.
3. 날짜 형식 강제(041 `issue_noticed_date="unknown"` 12건 vs 스키마 `MM/DD/YYYY`·`get_current_time` 결과 실재), 필드-문서 의존성 강제(스키마가 "이 문서를 근거로 판단하라"고 명시한 필드는 그 문서 획득 전 쓰기 차단·[[16]] GET/FIND 루프 정합), 미부여 도구 실행 지시 차단.
4. **B급**: "사용자가 모른다고 한 값"의 처리 — 041에서 날짜는 시스템 값을 써야 정답, 주소도 DB 값이 정답이었으나 무조건 강제는 PII 맥락에서 역효과 ⇒ 넛지.

## §7. [[05]] 3질문

1. 순증: 0(프로브·census는 계측·learn은 4벤치 경로). 2. 동결: 없음(오히려 learn이 유동성 담당). 3. 수행 대체: 없음.

## §8. 실행 순서·리뷰 요청

1. **무료 선행**: ⓐ D-2 census(즉시 가능) ⓑ §3 프로브 4종(로컬·비용 0) — day9c 대기 중 진행 가능.
2. 프로브 결과가 "경계"면 §5 학습 데이터 설계서(별도)로, "부하"면 scaffold 조정으로 회귀.
3. 027 등 원장-대조 가능 조각은 **C1 우선**(DAY8 §3-5 선행 확인 후).
- **리뷰 결과 반영 완료**: (a) 프로브 = 9변형·판정 임계·B 재현확인 반영 후 동의 (b) 스킬 2종 유지 — 하위 분해는 프로브 결과가 나온 뒤 데이터 설계서에서(지금 쪼개면 근거 없는 선험 분류) (c) D-2 규율 동의 + counterfactual 열·근거 발췌 보강 (d) census 즉시·프로브는 스로틀/완주 후.
- **신규 리뷰 요청 답변 반영(2026-07-29 2차)**:
  - **ⓐ dbdiff 실행 = 승인·즉시**(LLM 호출 0·저장 궤적 DB diff·day9c vLLM 큐 무관). 같은 실행에서 **"read-only CALLED만 다른" 태스크 census**를 함께 산출 → D-3 [M]→[S].
  - **ⓑ day10 1순위 = 조건부 동의**: 표적·A급 분류는 유지하되 **§6.5-1의 규명 갈래 3·4(hints 적중·ctx 세탁) 실측을 착수 하드 게이트**로 건다. 결과가 "기존 레버의 도달 구멍"이면 처방=F6b형 **소수리**, "구조적 사각(근거-원천 오염)"이면 그때 **원천 규칙 신설**이 정당 — 어느 쪽이냐로 설계가 갈린다.
  - **ⓒ 소급 범위 = 3단계 한정**: ①D-2([S])는 day5~9에 **counterfactual 열 병기**(원 수치 보존·전면 재계산 아님·[[47]] 비교성)·이후 인용 수치는 병기 기준 명시 ②D-3는 **dbdiff [S] 확정 후에만** read-역상관 통제를 소급(지금 재해석하면 증거원장 위반) ③전면 소급 대신 **"action/db 지표가 레버 GO/NO-GO 판단에 쓰인 결정 목록"**을 먼저 만들고 그 결정만 재검토. 1순위 재검토=**[[19]] 합성 런의 read-계열 레버(REQREADS·discovery) 효과가 db_match 하락과 상쇄돼 과소평가됐을 가능성**.
