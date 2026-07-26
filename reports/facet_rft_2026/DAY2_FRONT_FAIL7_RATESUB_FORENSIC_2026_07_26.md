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

## §6 front 19/32 시점 신호 3건 포렌식 (2026-07-26 오후·C198)
### §6A 021 회귀 = C197 침묵 경로 그대로 [S] (Δspurious 혐의 해소)
- 021 1차 시도 timeout 크래시 → **R1 재시도 영속**(log `task_021.0(…R1)`·Retry 1회). 재시도 인자 = python-repr 문자열
  (`json.loads` FAIL 재현) → isolate 무언 skip → **coverage 없는 "(none)"**(전문 확인) = 019와 완전 동형.
- gold 분쟁 2건(txn_ccbb948ffa10 Chipotle·txn_5b30a52ac9d6)은 무판정에 매몰·에이전트는 WeWork(txn_fa793baabcf4)만
  dispute(오귀속·유저-sim이 Chipotle을 에이전트의 "dining 1pt/$2" 환각 설명에 근거해 자진 철회하는 2차 피해 포함).
- ⇒ 어제 flip-pass(021)를 깬 것은 C193~C195 부작용이 아니라 **C197 입력-결함 구멍**. fix#1([ARGS-FORMAT])이 직접 차단.
- **[D] 관찰**: python-repr 슬립 = 재시도 시도 2/2(019 R1·021 R1)·정상 1차 시도(017/020/027)=전부 유효 JSON. retry-상관 후보(표본 2).

### §6B infra None 5건(001/006/007/008/010) = vLLM 큐 포화 [M] — "timeout 1200 확정" 기각 방향
- 5건 전부 termination=infrastructure_error·nmsg 0(attempt 폐기)·**agent 모델(Qwen-32B vLLM) litellm.Timeout 3/3 소진**
  (user-sim 아님·에러 문맥 model=Qwen 확인). B: 001/007/010 · A: 006/008.
- 결정 단서: 12:30~12:33 **양-arm 8태스크 동시 Retry-1**(001/003/006/019/022/023/024/025) = 일시 큐-포화 이벤트.
  rate-태스크 isolate 문서주입 폭주(대형 프롬프트 다발) 시간대와 일치 의심([M]·타임스탬프 부분 대조).
- 사후 헬스: 양 포트 /v1/models 8ms 정상·GPU0 100%/GPU1 36% — 영구 stall 아님.
- ⇒ handoff §2-3 "timeout 1200 처방 확정" **기각 방향**: 포화 창에서는 1200도 소진. 다음-런 후보: isolate 서브 스로틀/
  전용 인스턴스·conc 축소·arm 시차 기동. (final 판정은 front 완주 후 infra 전수와 함께.)

### §6C 027 = C197 아님·context-overflow 축(023/t95형) [S 관찰]
- rate 호출 **건강**: 유효 JSON·**account_open 26행 포함**·coverage 26/26·**403 검출(6300 vs 3150)** — 020이 놓친 FN을
  같은 유저·같은 배치에서 잡음(= C197 §2B 원인 귀속의 대조군 재확인: 차이는 오직 account_open 유무).
- 실패 = termination **context_window_exceeded**(reward 0.0·nmsg 79): 26-txn 다단계(분쟁 11건→상태확인 루프→갱신)에서
  전체 txn 재-fetch 반복+대형 echo로 창 초과. 별개 축 = coverage/discovery-load.
- 처방 = 신규 발명 불요·기존 설계 적용 후보: rate 도구 fetch-first isolate(ref_params·§2ah 동형)·E-PLAN([[14]]).

## §7 front 25/32 시점 신호 4건 정밀 포렌식 (2026-07-26 저녁·C199)
> 궤적 덤프=리모트 `day2_sig4.txt`(032/033/022/023/029 전문·action_checks 대조).

### §7A 032 회귀 [S] = notice-레이스 아님 → **말-완결(claimed-completion) 슬립**
- notice 미등장(noticerep 0 정합)=C193 표적 밖 — **어제 야간 분류(032=notice-레이스·C192 A그룹)는 이 건에 한해 재분류**.
- 기전 3단: ①미끼 접미사 `initial_transfer_to_human_agent_1822` 언락·2회 호출("단계 미달" decoy 반환에 고착)
  ②불요 신원확인 우회로 3턴(transfer에 검증 불요·0-match에도 지속) ③0218 2회 호출(gold=3회) 후
  `transfer_to_human_agents` **무호출·"연결해 드렸습니다" 단언**(m38)→유저 STOP. action_checks: transfer만 X.

### §7B 033 PASS [S] = 032의 살아있는 대조군
- 같은 프로토콜 계열: 1822(033 gold 포함)→0218→busy→**TRANSFER NOTICE→동의→transfer 실호출**(m18~m21) 완결.
- C193 표적 흐름(notice 1회→동의→즉시 호출)의 라이브 정상 작동 실증. 032와의 차이=마지막 실호출 유무뿐.

### §7C 022/023/029 [S] — 도구 결백·실패층 3색
- **022**: rate 도구 **9/10 gold 검출·과잉 0·coverage 55/55**(77 fetch 중 55 입력 — 미검출 txn_ffeede…=입력 커버리지 갭 의심).
  실패=유저 dispute 실행 0 — 에이전트가 "조정 게시" **날조 완결 선언**(발명 타임스탬프·조정 엔트리명 포함)→유저 audit-note
  만족 STOP. 기존 completion_guard **워딩 우회**(claim_question이 "dispute 제출" 주장만 겨냥·"adjustment 게시" 통과).
- **023**: `check_rebate_qualification` 정상(DOES NOT QUALIFY 결정론·60행). 실패=피벗 국면 — 유저 "간단한 카드(Silver?)"에
  **미발화 제약 4종 발명 투입**(max_annual_fee='95000'=소득 오전사·max_fx_fee 3·max_min_payment_pct 2·credit_score 700)
  +대안 비교/제시 없이 Silver 승인→유저 Silver 신청(gold=**Diamond Elite**). D-축(발명 형식화)+선택 축.
- **029**: rate-축 아님(도구 미도달) — **`johndoe@example.com` 플레이스홀더 이메일 날조 조회**(m12)+NOT_VERIFIED 가드의
  명시 지시("이름/이메일/ID로 조회")에서 이름-경로 불이행(끝내 안 물음)→검증 데드엔드→transfer. 신원-조회 경로 슬립.

### §7D 재시도 빈발 [M 정련] = 단발 폭풍 아님·**포화-마진 상시 운전**
- 타임아웃 시간당 1~4건 지속(12~15시 A:1/1/4/1·B:3/2/3/2)·isolate 주입 81회(A34/B47)+메인 6판+user-sim 병행.
- 긴 대화(infra-헤비)에 집중·**재시도=전체 대화 재실행**이라 비용 제곱(008: 12:32→13:33→14:43·~65분 주기 재소진).
- 레버 후보: isolate 전용 인스턴스/스로틀·arm 시차 기동·resume-형 재시도(러너 개조).

### §7E 교차 발견 — **말-완결 슬립 = 결정적 실패 3건의 공통 결정타** (C199 후보 레버)
- 032(transfer 연결 단언)·022(adjustment 게시 단언)·[§2A] 019(발명-txn 흐름 인접) — 유저-sim이 단언을 신뢰하고 STOP하면
  gold 행동이 영영 미실행되는 종결 기전.
- 레버 후보=completion-claim 게이트 일반화: (a) transfer-류 적용 확대 (b) claim_question 술어를 "제출" 한정→"행동-완료 주장
  일반"(게시/조정/연결)로 확대. [[03b]] 안전(주장-사실 대조만·행동 미지정)·[[19]] 합성-우선으로 다음 런 스택에.
- 별개 층: 발명-operand(023)·플레이스홀더-날조(029)=D-축 동형 — C197 fix#1(형식 검사)과 직교(형식 유효·내용 발명).

## §8 035 포렌식 — 회귀 3 판정 완결 (2026-07-26 밤·C200)
### §8A 035 [S] = notice-선행동의 레이스의 **첫-정식-notice 변형** (C193 술어의 구조적 사각)
- 절차 전반 성공: KB→`emergency_credit_bureau_incident_transfer_1114` 언락·호출(gold O·O)·출력="Proceed **immediately**
  with transfer_to_human_agents". 그 직후 m20서 호출 대신 **정식 TRANSFER NOTICE 첫 발화**→m21 유저 재동의(###TRANSFER###)
  →sim 종료(user_stop·nmsg 22)→transfer 영영 미호출(유일 X).
- C193 미발화 이유 확정: 술어={notice_text 포함 ∧ **이미 송신(재발화)** ∧ 무호출} — m8 "transferred now?"는 비정식
  패러프레이즈(notice_text 불일치)·m20은 **첫** 정식 notice → 술어 통과. noticerep=0 정합.
- 어제 035(재발화형)와 오늘 035(첫-notice형)=같은 레이스의 두 변형. 공통 구조=**동의(m9 선확보)·도구의 즉시-호출 지시가
  모두 실재하는데 호출 대신 notice 발화 → 유저 터미널 토큰으로 행동 창 소멸**.

### §8B 회귀 3(005/032/035) 최종 귀속 — "C193=원인-수정 확정" 기각
| task | 실제 사인 | C193 표적(재발화) 재현? |
|---|---|---|
| 005 | bypass-코드 국면 KB-확인 미실시(외부-주장 E-축·§1) | 아님 |
| 032 | 미끼 접미사 1822 고착+말-완결 슬립(§7A) | 아님 |
| 035 | notice-선행동의 레이스·첫-notice 변형(§8A) | 변형만(술어 사각) |
- 재발화형 0/3 재현. C193 자체는 무해(033 notice→동의→즉시 호출 정상 완결 실증·§7B·Δspurious 관찰 0) — 단 오늘 회귀
  3건의 사인이 아니며, 회귀 3의 실체=**3개 이질 층**.

### §8C 레버 후보 (기존 자산 적용·신규 발명 불요·[[19]] 합성)
1. **A2 `follow_up`을 1114류 에스컬 도구에 선언**(→transfer_to_human_agents) — 도구 출력이 지시한 후속 호출의 결정론 집행
   (give_discoverable follow_up과 동형=가장 깨끗한 기존-메커니즘 재사용).
2. C193 술어 확장: {notice 발화 ∧ 직전 유저 턴 동의 실재(∨ 직전 도구 출력에 transfer 지시) ∧ 무호출}→regen.
3. §7E 말-완결 게이트와 합성 — 032·035가 한 스택에서 함께 닫히는 구성.
