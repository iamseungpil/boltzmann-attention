# day5 처방 설계서 — C208 원인 10건에 대한 해결책 (2026-07-28)

> 입력: `DAY5_FULL_FORENSIC_2026_07_28.md`(2-pass 전수 포렌식) · 원장 C208.
> 상태: **설계 단계** — 사용자 리뷰 후 구현(설계서→리뷰→구현).
> 명명: P1~P10 = C208 처방 우선순위 번호와 일치.

---

## §0. [[05]] 결정-시점 3질문 (상설 · 설계 전체에 대한 답)

각 처방의 개별 답은 각 절에 있고, 여기는 설계 전체의 총괄이다.

| 질문 | 답 |
|---|---|
| ①scaffold/A2의 도메인-특화 **순증**? | **P1~P3·P5·P7~P10 = 순증 0**(하네스/엔진 도메인-일반·도구명/도메인명 리터럴 0). P4는 기존 A2 param 텍스트를 문구만 보강(신규 선언 1: `account_open`의 producer 계열 — verify_identity `ledger` 변이(C186)와 동형의 기존 패턴). P6은 A2 param 설명 1문장 추가. |
| ②모델의 유동 판단을 결정론에 **동결**? | 전부 NO — P3는 턴을 **주는** 것(행동은 모델이 emit)·P4는 결핍 사실 통지·P6은 전사(판단 아님)의 기계화·나머지는 인프라 위생. write 강제 0 유지(등대 §1.5). |
| ③scaffold가 모델 대신 **도메인 행동 수행**? | P6이 유일한 쟁점(→§P6 쟁점 절에서 정면으로 다룸): fetch는 **여전히 모델이 수행**하고 엔진은 이미 커밋된 도구 출력을 기계-파싱해 재사용만 한다(autofetch 아님·E-PLAN `_extract_entity_ids`(C101) 선례). 나머지는 해당 없음. |

검증 규율: 구현 후 `grep "if domain"=0` · scaffold 내 도메인 도구명 grep 0 · retail/airline 회귀 무변화.

---

## §1. 전체 지도 — 원인축 ↔ 처방 ↔ 기대 회수

| 원인축 (C208) | 처방 | 직접 표적 sim | 성격 |
|---|---|---|---|
| ①천장 자해(max_tokens 8192 예약) | **P1** 동적 max_tokens | ctxover 7 전부(특히 018/028/029=정답 직후 사망) | 하네스 1곳 |
| ②replay 위반(mutating 출력 변형) | **P2** replay 위생+문구 사실화 | 024·010(+022 잠재) | 측정 정합성 |
| ③터미널-턴 복권 | **P3** 종결행동 1턴 보장 | 004·035(+C200 계열 032·016류) | 엔진 게이트 |
| ④abstain 침묵(결핍 필드 미지목)+날조 비대칭 | **P4**(+P4b) abstain actionable화+producer-binding | 020·026(·027 위생) | 엔진+A2 |
| ⑤KB 덤프 44.8%+COMPACT 불발동 | **P5** 뷰-예산 재설계 | ctxover 7·040 | 엔진 파라미터+뷰 |
| ⑥재직렬화 8%+행 유실 | **P6** compute 참조-전달(opt-in) | 020·022·023 | **리뷰 필수** |
| ⑦C2-a 전량 NameError | **P7** 1줄 수정+무음실패 금지 | 004(OTP 약속) | 버그픽스 |
| ⑧DUP 스텁이 결과 미재제시 | **P8** 이전 결과 재제시 | 020 루프 | 엔진 문구 |
| ⑨gold 파손 | **P9** 005 처리방침 | 005 | **사용자 결정** |
| ⑩실패-sim 궤적 무영속 | **P10** 사이드카 영속 | 향후 infra 전부 | 하네스 |

비처방 잔여(§9): 008 티어선택/무갱신·014/015 주장-대조 생략·012 근거없는 우회안내·040 give-스텝 무시·027 과행동 → scale/learn 축([[13]]/[[45]]) + 후보 메모만.

---

## §P1. 동적 max_tokens — 천장 자해 제거 (최우선·최저비용)

**근본원인**(C208①): 고정 `T2_AGENT_MAX_TOKENS=8192`가 매 턴 프롬프트 천장을 40,448로 깎음.
completion 실측 p95=1,115·8k 초과 1/500. 7건 전부 36.5~40.2k에서 사망 — 모델 창(48,640) 초과 0건.

**설계 — 반응형 축소-재시도(추정 없음·결정론)**:
`t2_run_gated.py`의 LLM 호출 래퍼(`_gen`/llm_args 지점) 1곳.

1. 평시 `max_tokens = T2_AGENT_MAX_TOKENS`(기본 8192 유지 — 폭주-방어 A4와 짝).
2. `ContextWindowExceededError` 캐치 시 에러 원문에서 두 수를 파싱
   (`maximum context length is (\d+)` · `your .* resulted in (\d+) tokens` — vLLM 고정 포맷):
   `new_max = model_max − prompt_actual − MARGIN(64)`.
3. `new_max ≥ T2_MT_FLOOR`(기본 256)이면 **그 값으로 1회 재시도**. 미만이면 진짜 창 소진 →
   기존 OVERFLOW_GUARD graceful-stop 경로 그대로(변경 0).
4. 계측: `[T2_DYN_MT] shrink 8192→{new_max} (pt={prompt_actual})` — 발화 수가 곧 "천장 근접" 신호.

프롬프트-토큰 사전 추정은 채택하지 않음(chars/3.3 추정=비결정·오차): 에러가 정확한 수를 주므로 반응형이 더 단순·정확.

- **[[05]]**: 서빙 파라미터·도메인 무관. ①②③ 전부 no.
- **검증(오프라인)**: 가짜 CWE 에러 문자열 → 파싱·재시도 경로 단위테스트(`test_dyn_mt`)·플로어 경계·파싱 실패 시 기존 경로 폴백.
- **반사실 근거**: cap 2048 시 7/7 그 스텝 생존(포렌식 §1). 동적화는 2048 고정보다 우월 — 평시 8192 유지로 A4 절단-미커밋과 007형 폭주 방어 불변.
- **간섭 감시(W-c)**: 축소-재시도 → OVERFLOW_GUARD 순서 — shrink가 먼저, 실패 시에만 guard. regen 계열(claimprov 등)이 재시도 안에서 또 CWE 나면 즉시 guard(2중 재시도 금지).

## §P2. 평가층 replay 위생 + GUIDANCE 사실 교정 (측정 정합성·P1과 동급 우선)

**근본원인**(C208②): `t2_prekb_patch.py`가 **mutating 도구**의 ToolMessage.content에 `[GUIDANCE]`를 append →
tau2 replay(`environment.py:374~390`)는 mutating 도구만 재실행해 content **정확 비교** → 불일치 ValueError → sim 사망.
비-mutating(KB 등)은 replay가 스킵하므로 append 무해. 스캐폴드 주입 도구(우리 것)는 env에 없어 replay 스킵 = 전부 무해.

**설계 — 2원칙 + 문구 사실화**:

1. **불변식**: `_is_replay_compared(name) = env.(tools|user_tools).has_tool(name) and tool_mutates_state(name)` 이면
   **content 불변**. 판정은 env toolkit 질의(도메인-일반·리터럴 0). 헬퍼 1개를 만들어 content-변형이 있는
   **전 지점에 일괄 적용**(prekb utool/atool feedback·gate_patch 4297 "[Note: …]" 등 — 구현 시 grep으로 전수:
   `r.content =`·`.content +` 패턴 감사, 포렌식 §3의 마커 감사 목록 활용).
2. **mutating일 때의 피드백 채널 = 생성-레벨**: 기존 reminder 채널 패턴(`ag._t2_eplan_reminder` 동형)으로
   다음 생성 뷰에만 주입·히스토리 미커밋(repo 원칙 "생성-레벨=작업버퍼=replay-clean"과 정합).
3. **문구 사실화**(채널 오분류 처방·handoff §2b-1 병합): "Unknown discoverable tool 'X'" 시 3-분기 —
   - `user_tools.has_tool(X)` → "X is not a discoverable tool — it is a tool the customer runs directly.
     Do NOT try to give it; instruct the customer to run X themselves."(도구명=env 레지스트리에서·리터럴 0)
   - `tools.has_tool(X)` → "X is one of your own tools — call it directly."
   - 둘 다 없음 → 기존 "does not exist" 유지(이때만 '발명' 표현 정당).

- **[[05]]**: 판정=env 레지스트리 멤버십·문구 도메인-일반. ①②③ no.
- **검증(오프라인·필수)**: ①위생 — day5 024/010의 재현 시나리오(가짜 히스토리에 GUIDANCE 부착 mutating 결과)
  → 패치 후엔 append 자체가 안 일어나 `set_state` replay 무예외 통과(`test_replay_hygiene`).
  ②문구 — apply_for_credit_card(유저-네이티브)/실제 미존재명 각각 분기 확인.
  ③잔존 감사 — 헬퍼 우회하는 content-변형 grep 0.
- **기대**: infra 2건 소멸 + 022형 잠재 제거. day2~day4b infra 다수 소급 설명([M]은 P10 이후 검증 가능).

## §P3. 터미널-턴 보장 — 동의 후 행동 턴 0 봉합

**근본원인**(C208③): 유저 터미널 토큰 → 즉시 종료. 유일 예외=E-PLAN walk gap 보류인데 gap 산출은
coverage 모델 의존(004·035=gap 0→턴 0·008=spurious gap이 우연히 구제). gold 종결행동 유무는 반영 안 됨.

**설계 — "공표-미이행 종결행동" 1턴 유예(walk과 독립)**:
`t2_eplan_patch._check_termination` wrap 내부, drive 판정과 **별도 분기**(drive 앞에 배치).

- **술어**(전부 결정론·기존 데이터만): `termination_reason=user_stop` ∧ ⓐ notice 공표됨
  (assistant 메시지에 A2 `notice_text` 부분문자열 실재 — A2 gate spec에서 읽음) ∧ ⓑ 해당 게이트의
  대상 도구(A2 spec의 tool·family)가 **한 번도 호출 안 됨**(툴콜 원장 대조) ∧ ⓒ 본 유예 미사용(1회/sim).
- **동작**: `done=False·termination_reason=None`(drive 보류와 동일 기법) + 생성-레벨 reminder
  "The customer has already agreed. Do not repeat the notice — CALL {tool} now with an appropriate summary."
  + 그 턴에 한해 **기존 FORCE_ACTION 레버로 `tool_choice=required`**(재-notice 산문 봉쇄·신규 강제 아님·기존 레버 합성).
- **비표적 확인**: 이미 호출했으면(040·008 성공 후) ⓑ 불성립=무개입. notice 미공표(만족-STOP류: 012·014·019)면 ⓐ 불성립=무개입 —
  **이 레버는 notice-레이스 계열만 표적**(012/014형은 §9 잔여).

- **[[05]]**: notice_text·대상 도구=A2에서 읽음(기존 선언 재사용·신규 선언 0)·엔진=부분문자열+원장 대조. ②동결 아님 — 호출은 모델이 emit(write 강제 0·요구는 "지금 불러라" 통지+required는 산문 차단일 뿐 도구·인자 미지정). ③no.
- **검증(오프라인)**: 004/035 말미 재현 시나리오(공표+미호출+user_stop→유예 1회·reminder 문구)·
  040/008 말미(호출 있음→무개입)·2회째 터미널→무개입·drive와 동시 성립 시 순서(`test_terminal_grant`).
- **간섭 감시(W-a)**: drive 보류와 본 유예가 **같은 sim에서 연쇄**되면 종료 지연 최대 2턴 — 합산 상한
  `T2_TERM_HOLD_MAX=2`(drive K와 별도지만 합산 로그로 감시). user-sim이 유예 턴에 또 터미널을 내면 ⓒ로 종료 — 무한루프 없음.

## §P4. abstain의 actionable화 + P4b producer-binding — coverage 붕괴와 날조 비대칭 동시 봉합

**근본원인**(C208④): 3-값 판정불가 시 "N could not be verified"만 출력 — **어느 필드 결핍인지 미지목**(엔진은 안다).
020/026(정직 생략)=14행 전멸, 027(날조 02/01/2025)=우연히 gold 재현. 정직<날조 역전.

**설계 P4 — 결핍 지목**:
`t2_scaffold_get.py`의 op 실행부 — 3-값 전파에서 행이 탈락할 때 **null이 난 ref의 필드명을 수집**
(`r.account_open` → `account_open`). coverage 병기부를 확장:

> `[coverage] 12 of 26 rows were checked (14 could not be verified — missing input field(s): account_open (14 rows)). Provide the missing field for those rows — read it from the records that contain it — and call again with the completed input.`

- 필드명=A2 params 키 그대로(도메인 데이터)·문구=도메인-일반. `judged==0`이면 `error=True` 승격(기존 C195 방향 유지).

**설계 P4b — 비-레코드-유래 operand의 producer-binding**(C186 `ledger` 변이 선례 동형):
- A2 ratefix variant에 선언 1개 추가: `"grounded_params": {"account_open": {"producer_contains": ["credit_card_accounts"]}}`
  (해당 값이 **producer 출력 조각을 담은 커밋된 tool 출력** 안에 부분문자열로 실재해야 유효 —
  producer 식별=출력 내용의 selector 문자열·A2 데이터).
- 엔진: 값이 selector-일치 출력 안에 없으면 그 필드를 **missing으로 강등** → P4 경로와 합류(같은 지목 문구).
  ⇒ 027의 날조(02/01/2025는 거래 출력에는 있으나 accounts 출력에 없음)가 **정직 생략과 동일한 abstain+지시**로 수렴.
- 한계 명기: 값-부분문자열 검사라 "accounts를 읽었고 그 안의 아무 날짜"면 통과(약함) — 그러나 현 실패군(아예 안 읽음/전량 날조)은 전부 잡힘. 강한 필드-단위 바인딩은 후속.

- **[[05]]**: 신규 A2 선언 1(producer selector=도메인 데이터)·엔진=부분문자열 실재 검사(기존 `_val_grounded`/ledger 계열 재사용). ②판정불가→통지이지 값 대입 아님. ③no.
- **검증(오프라인)**: 020 실제 페이로드(account_open 無)→"missing: account_open(14 rows)" 지목·
  027 실제 페이로드(날조)→P4b 강등으로 동일 지목·완전 페이로드→무개입·019 quote-ground 행은 별도 사유 표기(`test_abstain_actionable`).
- **기대**: 020/026형 직접 회수 + 027의 운-의존 제거. P3와 독립·P6과 합성 시 시너지(§P6).

## §P5. 뷰-예산 재설계 — KB 44.8% 축

**근본원인**(C208①·⑤): KB 1회 15~29k자·근사-중복 질의 dedup 회피·VIEW_COMPACT 문턱(120k자)이 사망선 위(6/32만 발동).

**설계(3부·전부 뷰-측=커밋 히스토리 불변=replay-safe)**:

1. **문턱 하향**: `T2_VIEW_COMPACT_MINTOTAL` 기본 120,000→**60,000자**(≈17k tok·천장의 40%선에서 개입 시작).
   코드 기본값+go_stack 동시 수정(문서-실제 불일치 방지·C186 교훈).
2. **per-메시지 뷰 캡(신설·`T2_VIEW_MSG_CAP` 기본 8,000자)**: **최신 tool 출력 1개를 제외한** 모든 비에러
  tool 출력이 캡 초과면 뷰에서 head 300+tail 150 다이제스트(기존 `_compact_view` 다이제스트 함수 재사용).
  모델은 도착 턴에 전문을 봤고, 이후 턴부터 다이제스트 — read 주체=모델 유지([[05]]③ autofetch-류 아님·
  기존 VIEW_COMPACT과 같은 원리의 강화이며 keep_recent 의미를 "최신 1개 전문"으로 좁힘).
3. **근사-중복 질의 안내(기본 OFF·`T2_READ_NEARDUP=1`)**: KB류 read의 질의를 정규화(소문자·불용어 제거·토큰
  집합)해 Jaccard≥0.8이면 스텁: "This query is nearly identical to an earlier one (differs only in: {diff
  tokens}); the earlier output is above. Refine with genuinely NEW terms, or proceed with what you have."
  — 차단이 아니라 안내(1회분 출력은 그대로 줌: 018형 "재검색 5연발"의 2회째부터 스텁). 정당 재검색 오탐 리스크가
  있어 **기본 OFF·격리 arm에서 발화·오탐 계측 후 승격**.

- **[[05]]**: 전부 도메인-일반(문자열 길이·토큰 집합). ①②③ no.
- **검증(오프라인)**: 018 실제 궤적 재생 — 문턱 60k+MSG_CAP 8k였다면 최종 pt 추정 재계산(목표: 40k 여유 확보)·
  다이제스트 후 재열람 탈출구(동일 호출 재발행=READ_DEDUP 면제) 회귀 유지·retail 회귀(`test_view_budget`).
- **간섭 감시(W-e)**: MSG_CAP이 isolate 서브의 문서 주입과는 무관함을 확인(isolate는 자체 컨텍스트)·
  P1 동적 mt와 합성 시 "천장 근접 전에 뷰가 먼저 줄어드는" 순서가 되는지 계측으로 확인.

## §P6. compute 참조-전달 (opt-in·리뷰 필수 — [[03b]]/[[05]] 쟁점 정면)

**근본원인**(C208①·④): 대량-행 compute가 pass-by-value — 재직렬화 8%(020 11.8k tok)·행 유실(022 9행·023 15행)·
필드 누락(account_open)·거대 completion 필요(022=7,019 tok)의 공통 뿌리.

**설계 — 커밋-출력 참조 해석(`T2_SG_BYREF=1`·기본 OFF)**:
- A2 params 문구에 1문장 추가: "Instead of retyping, you MAY pass `transactions="@last:<the tool you used to
  read the transactions>"` — the deterministic system will reuse that exact earlier output."
- 엔진: `@last:<tool명>` (또는 `@call:<tool_call_id>`)을 **이미 커밋된 그 도구의 최신 비에러 출력**으로 해석,
  env 레코드 덤프의 기계 포맷(`Record ID: … field: value`)을 결정론 파싱해 rows 구성(파서=포렌식 §6 검증에 쓴
  것과 동일 로직·isolate `row_fields`가 필드 목록 제공). account_open은 P4 지시에 따라 모델이 accounts getter를
  읽은 뒤 `account_open="@last:<accounts getter>"`로 참조 — 엔진이 card_type 결정론 join.

**쟁점 (정직 표기 — 리뷰 포인트)**:
- ⓐ [[05]]③ "scaffold가 도메인 행동 수행?" — **fetch는 수행 안 함**(모델이 읽어 커밋한 출력만 재사용).
  autofetch 기각 사유(엔진이 fetch 수행+read 주체 이전)와 구분됨. 선례: E-PLAN ledger가 도구 출력을 기계-파싱(C101).
- ⓑ [[03b]] "엔진-formalize(문자열파싱)=cheating" — 본 파싱 대상은 **NL이 아니라 env의 고정 기계 포맷**
  (LLM 몫의 formalize를 대체하지 않음: 전사는 판단이 아님). 단 이 경계 해석이 넓어지면 위험하므로
  **파서는 `Record ID:` 포맷 전용·다른 텍스트에 불가**로 못 박음.
- ⓒ 학습 관점 부작용: 전사-스킬이 훈련 신호에서 사라짐 — 그러나 [[10]] "concrete=offload"와 정합
  (전사=concrete 값 이동). 반론 여지 있음 → **기본 OFF·arm 대조로 도입**.

- **검증(오프라인)**: 022 실제 궤적에서 참조-해석이 77행 무손실 재구성(포렌식 파서와 대조)·`@last` 미존재/에러
  출력 참조 시 명확한 에러·pass-by-value 경로 무회귀(`test_sg_byref`).
- **기대**: 020형 재직렬화 소멸·행 유실 0·P8의 재호출 비용도 무해화. **P4/P4b와 합성이 완결형**
  (결핍 지목→모델이 getter 읽음→참조로 공급).

## §P7. `T2_UNAVAIL` 수정 + 레버 무음실패 금지 (즉시)

1. `t2_gate_patch.py:5081` — `or getattr(orch, "environment", None)` 제거(스코프에 `orch` 없음=전량 NameError).
   env 해석은 `getattr(self, "environment", None)` + `getattr(getattr(self, "agent", None), "environment", None)`
   순 폴백(구현 시 해당 wrap의 self 실체 확인 후 확정).
2. **무음실패 금지(일반 규율)**: 레버 try/except의 skipped 카운터를 런 종료 요약에 집계 —
   `[T2_LEVER_HEALTH] unavail: fired=0 skipped=223` 형태로 **0-발화·전량-스킵 레버를 런 요약이 자동 고발**
   (C208⑥ "정상 위장" 재발 방지). 구현=마커 카운터 dict 1개+종료 시 출력.
- 검증: 단위(`test_unavail_env`) — env 해석 성공 경로·004형(미보유 기능 약속) 시나리오에서 발화 확인.

## §P8. DUPLICATE-COMPUTE 스텁에 이전 결과 재제시

**근본원인**(C208①): 스텁이 "refer to the earlier output"만 지시 — 유저 후속질문마다 재호출 유인 존치(020 5회).

**설계**: 엔진이 (tool, args-hash)→마지막 출력 텍스트 캐시(우리 주입 도구 한정=env 무관=replay-safe).
스텁 = 기존 경고 + `Previous result (unchanged): {cached}`. 재제시 상한 2회(3회째부터 기존 STOP 문구만).
**천장 근접 시 생략**: P1 계측(`T2_DYN_MT` shrink 발생 후)이면 재제시 생략(작은 창에서 재제시가 역효과·W-d).
- [[05]]: 캐시=자기 출력 재게시·도메인 무관. 검증: 020 시나리오 — 2회째 호출에 결과 포함 확인(`test_dup_represent`).

## §P9. task_005 — gold 파손 처리방침 (사용자 결정 필요)

사실(C208⑦): gold `log_verification` 행이 전 필드 센티널 `9K2X7M4P1N8Q3R5T6A` — 어떤 행동으로도 그 DB 행을
만들 수 없어 **원리상 통과 불가**(basis=DB). 선택지:

| 안 | 내용 | 비용/리스크 |
|---|---|---|
| **(a) 권고** | 스코어보드에서 **주석부 제외**: "n=31 유효 + 005 gold-파손 1" 병기(수치 조작 아님·매 보고 명시) | 이전 런과 분모 불일치 — 소급 재표기 필요(day2~5 전부 005 fail이었으므로 순위 무영향) |
| (b) | gold JSON 로컬 패치(센티널→실값) | 벤치 개변=비교성·상류 재현성 훼손 — 비권고 |
| (c) | 현행 유지(분모 포함) | pass율이 구조적으로 1/32 저평가 지속 |

## §P10. 실패-sim 궤적 영속 (사이드카)

**근본원인**(C208②): 러너가 4회 재시도 소진 시 메시지 무영속 → infra 원인 규명 불가(024/010·과거 run 다수).
**설계**: 하네스 레벨(tau2 코어 비수정) — `run_with_retry` 몽키패치로 최종 실패 시 그 시도의 메시지·예외 원문을
`failed_<task>_<trial>.json.gz` 사이드카로 저장(있는 데이터만·best-effort·실패해도 러너 무영향).
persist 데몬 수집 대상에 포함. 검증: 강제 실패 주입 단위테스트(`test_failed_persist`).

---

## §7. 배포·검증 계획

**구현 순서**(의존·효과 순): ①P7(1줄+계측) ②P1 ③P2 ④P10 → 오프라인 배터리 → ⑤P3 ⑥P4(+P4b) ⑦P8 → 배터리 →
⑧P5(1·2만·3은 OFF) → **P6은 리뷰 결론 후 별도**. P9는 사용자 결정만.

**오프라인 게이트(전부 무료·[[09]])**: 신규 테스트 7종(`test_dyn_mt`·`test_replay_hygiene`·`test_terminal_grant`·
`test_abstain_actionable`·`test_view_budget`·`test_unavail_env`·`test_dup_represent`(+`test_sg_byref`·`test_failed_persist`))
+ 기존 회귀 12종 + day5 실궤적 재생 검정(018 뷰-예산·020 페이로드 지목·024 replay·004 유예).

**go_stack 갱신**([[19]]·C186 교훈=코드 기본값과 동시): 신규 `T2_DYN_MT=1 T2_TERM_GRANT=1 T2_ABSTAIN_FIELDS=1
T2_PROD_BIND=1 T2_DUP_REPRESENT=1 T2_FAILED_PERSIST=1` + `T2_VIEW_COMPACT_MINTOTAL=60000 T2_VIEW_MSG_CAP=8000`.
OFF 유지: `T2_READ_NEARDUP`·`T2_SG_BYREF`(승격 조건 명기). **간섭 감시점**: W-a(§P3)·W-c(§P1)·W-d(§P8)·W-e(§P5).

**day6 판정 프레임**(front 32·conc 1·비교성: 레버 추가는 [[19]] 합성-우선 정합):
- 회수 기대(정직·과약속 금지): ctxover 7→0~2(P1+P5) · infra 2→0(P2) · 004/035형 +1~2(P3+P7) ·
  020/026형 +0~2(P4·모델이 지시를 따라 accounts를 읽는지는 모델 몫) ⇒ **PASS 11 → 14~17 범위**(P6 제외).
- **[[08]] 강건성 한계(명시)**: day5=단일 trial(pass^1 점추정)·user-sim(gpt-5.2)=서버 비결정(C192 실측 — 054가
  16→8→2로 출렁인 전례). 위 기대치는 **기전-회수 예측이지 점수 예측이 아니다** — day6 판정은 점수 증감이 아니라
  **기전 소멸 여부**(ctxover 에러 원문 0건·replay ValueError 0건·유예 발화→호출 전환·abstain 지목 후 재호출)로
  1차 판정하고, 점수는 nt≥2 누적 전까지 [D] 등급으로만 인용한다. 035 flip=마진의 운(C206) 재확인 —
  P3의 효과 판정도 "턴이 주어졌고 그 턴에 호출했나"(발화율)로, pass flip으로 하지 않는다.
- 부작용 계측(Δspurious≤0 원칙): 유예 턴의 오발화 0 목표·NEARDUP 오탐(OFF지만 격리 arm)·P4 지시 후 재호출 횟수.

## §8. 각 처방의 [[05]] 3질문 개별 답 (요약표)

| 처방 | ①특화 순증 | ②유동성 동결 | ③도메인 행동 수행 |
|---|---|---|---|
| P1/P7/P10 | 0 | no | no |
| P2 | 0(레지스트리 질의) | no | no |
| P3 | 0(기존 A2 notice 재사용) | no(호출=모델 emit) | no |
| P4 | 문구 보강만 | no(통지) | no |
| P4b | A2 선언 +1(C186 선례) | no(검증) | no |
| P5 | 0 | no(read 주체=모델) | no |
| P6 | A2 문구 +1 | **전사=판단 아님(쟁점 ⓒ)** | **쟁점 ⓐⓑ — §P6 명기·기본 OFF** |
| P8 | 0 | no | no |

## §9. 이 설계서가 다루지 않는 잔여 (정직한 경계)

- **008 티어선택+forced-read-without-revision**: 정답 표가 문맥에 있어도 무갱신 — soft 넛지(강제-읽기 후 "인자
  재도출" 지시)는 [[42]] prompt-ceiling상 기대 낮음 → 후보 메모만·**scale/learn 축**([[45]]). enum-제약
  (guided decoding으로 reason을 KB-티어표에서 추출한 enum grammar로) 은 A2-특화 성장+gold-맞추기 위험 — 기각.
- **014/015 주장-대조 생략·012 근거없는 우회안내**: "고객 주장→KB 검증 후 발화" 일반 게이트는 claimprov의
  확장 영역이나 이번 스코프 밖(설계 미착수·별도 설계서).
- **040 give-스텝 무시(env 자구 지시 불응)·027 과행동**: 행동층 잔여. give-flow는 기존 FOLLOWUP/RESOLVE 계열이
  이미 표적 — day6에서 P2 문구 사실화의 효과 관측 후 판단.
- **A1 repetition_penalty**: 여전히 승인 대기(비교성 비용) — P1과 독립.
