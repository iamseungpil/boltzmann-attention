# day2 front 실패 7건 전수 포렌식 + rate-서브 원인 [S] 확정 (2026-07-26)

> 정본. 원장 C196·C197의 근거 문서. 런=`bank_day2front{A,B}_20260726`(user-sim gpt-5.2·bm25·64-레버 go_stack
> +C193~C195·T2_LLM_TIMEOUT=1200·conc3). 궤적 덤프=리모트 `/home/woori/scratch/day2_fail7.txt`(1434줄).
> operand 트레이스=`/home/woori/scratch/day2{A,B}_operands.jsonl`(T2_SG_ISOLATE_TRACE·C195 상시화分).
> 리플레이 스크립트=본문 §3(리모트 즉석 실행·t2_compute.apply_op 직접 호출). 스택 커밋=remote `f749eb6f`.

## §0 요약
- front 중간(16/32 시점) 실패 7건(004·005·012·014·015·019·020) 전수 per-step 정독 → 3축 분류(§1).
- rate-서브 2건(019·020)은 **서브·엔진 결백** — 원인=에이전트 입력-결함이 엔진을 **침묵 통과**(§2·전부 [S] 리플레이 재현).
- 처방 4종 구현+오프라인 검증(§4·C197). **현행 97-런 리모트에는 미배포**(단일-스택 비교성 보존·다음 런부터).

## §1 실패 7건 분류 (전수 per-step 정독)
| task | gold 요지 | 실패 지점 | 축 |
|---|---|---|---|
| 004 | transfer(reason=account_ownership_dispute) | 이메일 불일치·검증불가 상황을 소유권 분쟁으로 개념화 못함 → reason=customer_demands_… 오선택(m30). 부가: verify_identity에 DB-fetch 값 날조 투입 → A2 대화-인지 검증기가 차단(phone만 인정) | 종결-선택 |
| 005 | bypass 코드 KB 절차(log_verification 전필드=코드)→email 변경 | 코드 국면 KB_search 0회·m42 절차 감지했으나 코드 대신 개인정보 요구(KB 정반대)·3턴 자기-반전→transfer. 전판 §2b 결론 유지 | 외부-주장 KB검증 |
| 012 | transfer(reason=kb_search_unsuccessful_…) | KB 4회 전부 무관 문서(미발견 확정)인데 이관 대신 "인앱 챗 접수·1주 전 제출" 환각 안내→유저 만족 STOP. **transfer 호출 자체가 없어 PRE-ACTION-KB(C190) 미발화** — pre-gate는 미이관-환각을 구조적으로 못 잡음 | 기권-대신-환각([[43]])·종결-선택 |
| 014 | transfer(reason=unconfirmed_external_communication) | doc_002 "각 카드 조건 상세 없음"을 읽고도(m6) 편지 오퍼($100/$500/2mo)를 사실 승인·Crypto 링크 유도 | 외부-주장 |
| 015 | 유저가 get_referral_link(card=**Platinum**) 실행 | KB 문서 찾고도 오퍼↔카드 매핑 미대조·편지의 Crypto에 자기-정박. give(인자 없음)=match O | 외부-주장 |
| 019 | dispute 4건(f093/5807/d398/37b5) | §2A — "(none)" 침묵(입력 직렬화 슬립) 후 환각 캐스케이드(rate 발명·+310/+2642 발명·**발명 txn ID** txn_adjustment_gold/ecocard로 dispute 제출=DB 오염) | rate-서브 입력결함 |
| 020 | dispute 4건(403/506/410/411) | §2B — account_open 누락→promo 침묵 해제→8 FP+403 FN. 에이전트는 도구 출력에 충실(11건 제출=DB 오염) | rate-서브 입력결함 |
- 축 집계: 외부-주장(005·014·015) 3 · rate-서브(019·020) 2 · 종결-선택(004·012) 2. E-계열 처방각도([[18]] 격리 프로브 先)·F-계열 관찰 유지.

## §2 rate-서브 원인 확정 (전부 [S])
### §2A task_019 — 인자 직렬화 슬립 → 엔진 3중 침묵 경로
1. **1차 시도**: 정상 인자 → isolate 23/23 산출(run_day2A.log:1936·Gold 5배치+Eco 5배치=operands.jsonl 실측)
   → LLM timeout(12:32·log)으로 크래시 → **R1 재시도**(log:2174 `task_019.0(26s R1)`), 영속 궤적=재시도.
2. **재시도(영속)**: 에이전트가 transactions를 **python-repr + leading-zero 숫자**(JSON·Python 모두 파싱 불가) 문자열로 emit →
   ① `t2_scaffold_get` ctx-빌드 json.loads 실패·str 잔류(침묵) ② isolate rows-not-list 조기 return None(로그 0줄)
   ③ `select_discrepant` recs-not-list → **stats 설정 前 [] 반환 = C195 coverage 우회** → "(none)"이 깨끗한 빈 결과로 위장.
3. **재현**: 실제 인자 문자열 리플레이 = json.loads FAIL → apply_op []·stats None (관측과 동일). 교차검증: 전날 밤(정상 파싱) 019=3/4 검출(C195 note)과 정합.
4. **부가 결함 2건**: (a) 재시도 인자에서 Uber(txn_5807…) card를 EcoCard로 오전사(DB=Gold — 파싱됐어도 이 행 FN)
   (b) 서브가 Thrive Market(txn_f093…·Green $175)에 ThredUp 제외문을 오적용(operands 실측 `exclusion_quote:"Thrift and Resale Markets\n- ThredUp"`·rate 1)=이웃-상인 혼동 — 밤-런 3/4의 잔여 1건 원인.

### §2B task_020 — `account_open` 누락 단독 원인
1. 서브 operand **전부 정확**(operands.jsonl 실측): BizSilver base 10/1·promo_mult 2·window 6mo·2024-11-14~2025-11-14·
   exclusion 인용(Microsoft·Coursera) 정확.
2. 에이전트가 ratefix params 계약의 `account_open`을 26행 전부 누락(019 호출엔 있었음=확률적 슬립) →
   엔진 `_date_between(None,…)`/`_date_in_window(None,…)`가 **False로 강제**(구판) → mult=1 → BizSilver 8 FP(전부 correct=recorded÷2)+**403 FN**.
3. **재현**: as-is 리플레이=관측 11건 정확 재현(coverage 26/26). **반사실**(DB `date_of_account_open` 02/13/2025·01/20/2025 주입)=
   **gold 4건(403·506·410·411) 정확 재현**(403 expected 6300 vs recorded 3150). 411 expected 1499.9=기지 gold 저작버그(_note_rounding)·판정 무영향.

### §2C 공통 구조 구멍
입력-결함(파싱불가 인자·필수 leaf 누락)이 abstain/피드백 없이 **침묵-오판정**으로 흐른다. C195 coverage는 stats-설정
경로에만 붙어 not-a-list 조기반환이 우회. (019의 "(none)"=무판정의 위장·020의 11건=결여 leaf의 위장 판정.)

## §3 리플레이 재현 절차 (provenance)
리모트 `workspace_common/boltzmann-attention-pi/scripts/distill/tau2`에서 `t2_compute.apply_op`에
(a) 019: results.json의 transactions 인자 그대로 → json.loads FAIL·[]·stats None
(b) 020 as-is: 인자(json.loads OK)+day2B_operands 병합 → 11건 (c) 020 반사실: +account_open(DB 개설일) → 4건.
결과 원문=본 문서 §2 수치. (세션 스크립트 /tmp/rc20·rc22·rc24, 2026-07-26.)

## §4 처방 C197 (구현 완료·오프라인 검증·[[03b]]/[[05]] 안전)
| # | 내용 | 파일 | 검증 |
|---|---|---|---|
| 1 | 목록형 op(over 선언) 인자가 json.loads 실패로 str 잔류 → **[ARGS-FORMAT] 에러 반환**(유효 JSON 재송신 요구·엔진 대리파싱 없음=formalize는 LLM 몫) | t2_scaffold_get.py exec2 | py_compile·경로 로직=REQREADS 동형 |
| 2 | select_discrepant not-a-list → stats{0,0,0} 계상(+judged=0 전용 문구 "NOTHING was actually checked") = C195 우회 봉쇄 | t2_compute.py·t2_scaffold_get.py | test_c197 t3 |
| 3 | `_date_between`/`_date_in_window` 입력누락=None(미확정·3-값 논리 전파) + if_then cond-미확정·양분기 동일값→그 값(무-프로모 행 보존) | t2_compute.py | test_c197 t1/t2/t4/t5/t6 (t4: 누락=skipped 2·판정 1 / t5: 개설일 주입=403 검출·401 비플래그) |
| 4 | exclusion 강등 quote를 엔진 결정론 대조(quote∈주입문서 축자 ∧ 행의 `quote_must_contain_field` 값∈quote·불성립=rate 드롭 abstain·값 생성 0). A2 선언 시만 활성 | t2_scaffold_get.py `_sub_inject`·A2 ratefix isolate `quote_must_contain_field:"merchant_name"` | test_sub_inject ②b (Thrive/ThredUp형 드롭·날조 인용 드롭·rate 외 병합 유지) |
- 회귀: test_c197 6/6·test_sub_inject·test_sg_isolate·test_discovery_dispatch·test_toolgate_requestor 전부 PASS.
- **배포**: 현행 97-런에는 미배포(단일-스택 비교성·[[47]] 동형 규율). 다음 런(재실험/B-계열 재검) 시 리모트 pull.
- 한계: #4는 오검출 강등을 **정직한 판정불가**로 바꿀 뿐 정답 rate를 만들지 않음(coverage+재확인 지시로 에이전트에 환류·[[03b]] 준수).
  #1은 재송신 요구 1회분 턴 비용. Δspurious 계측=다음 런 판정 프레임에 포함할 것(게이트 자신의 역효과 원칙·RESEARCH_MASTER §1).

## §5 남는 관찰 (판정 대기·front 완주 후)
- 회귀 3(005 완료=KB-확인 층·032/035 대기)·pass 4 유지·infra 12 완주 여부 = handoff §2 프레임.
- PASS 인과 확정分: 002(min_cashback 발명 소멸·catalog v2 eligible 정확·CHECK-FIRST 첫 행동) · 003(동일 경로) — 클러스터① 라이브 2연속.
