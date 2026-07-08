# QwQ agentic floor 실패 전수 포렌식 (2026-07-08·per-case·[[08]])

> 대상: `qwq32b_floor_retail_t4`(nt=4·456 sim·infra=0·gpt-4.1 user-sim) vs 대조 `fl32b_floor_retail_t4`(Qwen2.5-32B floor).
> 질문: QwQ(reasoning-RL 32B)가 same-size base보다 agentic pass 낮은(0.443<0.557) **실제 원인**을 case-by-case로.
> 도구: `scratchpad/forensic_step{1..5}` (repo 미커밋 분석·수치는 본 doc에 영속).

## 0. 결론 (한 줄)
QwQ의 deficit은 **행동선택이 나빠서가 아니라 실행/지속(execution/persistence)이 나빠서**다. 행동을 할 때는 base와 대등
(오히려 약간 나음). 초과 실패 전부가 **non-execution/premature-transfer**(추론이 "불가→transfer"로 귀결 + 출력 포맷 누수).

## 1. aggregate (clean)
- QwQ pass^1 0.443(202/456) vs base 0.557(254/456). termination 둘 다 456 all `user_stop`(crash/infra=0)·reward 이진(partial 0).
- **task 재분배(균일 열화 아님)**: base>QwQ **52 task**(t5/6/23/30/106 4→0) · QwQ>base **28 task**(t17/40 0→4·t63 0→3·t93 1→4).
  reasoning이 어떤 task를 열고 어떤 task를 깬다. 순 −24 task(−52 sim).

## 2. zero-reward 실행상태 분해 (correct tool_call 추출·`tool_calls[].name`)
| | QwQ (254 zero) | base (202 zero) |
|---|---|---|
| **non-execution**(write 0회·gold는 write 필요) | **73 (29%)** [69 needed] | **12 (6%)** |
| **wrong-execution**(write ≥1·args 틀림/미완) | 181 (71%) | 190 (94%) |
| ended transfer_to_human | 60 (24%) | 27 (13%) |
| mean agent turns / tool-calls | 13.5 / 6.3 | 17.8 / 8.4 |
- **★deficit 인과**: QwQ 254 = 73+181, base 202 = 12+190. 차 +52 = **non-exec +61 − wrong-exec 9**. ⇒ **deficit 전량이
  non-execution 초과**. wrong-execution은 QwQ가 오히려 적음(181<190).

## 3. wrong-execution(181) 오류형 — isolated 버킷과 정합
gold write action 미스매치 유형(QwQ 실제 write args vs gold):
- **order_id 틀림 ~79**(27 order_id + 26 order_id,item + 17 삼중 + …) = **⋈ cross-order**(틀린 주문 선택).
- item_ids/new_item_ids ~60(19 item+new · 15 new · …) = **variant/criterion + item 선택**.
- missed_this_write 79 = multi-write task 일부만(미완).
- payment 6 · reason 6 · address 9 = 소수 operand.
→ **⋈(order_id)이 wrong-execution 지배**. isolated ⋈ 경계(0.40·scale/budget/CoT/RL 다 실패)와 **정확히 정합** —
  QwQ의 방대한 추론(1.6k tok)도 ⋈을 못 풂. reasoning은 hard-operand(⋈)을 열지 못한다(§isolated 확증).

## 4. non-execution/transfer(QwQ-특이 초과) — per-case
- **t6**(4→0·wrong-exec 예): 인증→주문조회→변형조회 후 user "go ahead"에 `exchange_delivered_order_items` **호출은 함**
  ·단 args 틀림(desk-lamp 변형/item). = wrong-execution(⋈/variant), 실행실패 아님.
- **t30**(non-exec): user가 "태블릿 주문했다" 주장·QwQ 못 찾자 "system error 가능→transfer 권함" 반복→transfer. **성급한 포기**.
- **t106**(non-exec): email·name+zip 인증 실패→즉시 transfer. base는 더 끈질기게 재시도(transfer 13% only).
- **★포맷 누수 관찰**: QwQ content에 `</tool_call>`·`<|im_start|>` 리터럴 누수(t30 tail) — reasoning+chat-template이
  hermes 파서와 간헐 충돌→tool call 미등록 가능. non-execution 일부의 **배포 아티팩트**(순수 capability 아님).

## 5. 판정 (진짜 원인)
1. **wrong-execution(다수·71%)**: base와 공유하는 hard-operand(⋈ order·variant). QwQ reasoning이 못 개선(⋈ 경계 정합).
   → 이건 "thinking이 나쁘게 함"이 아니라 "thinking이 이 축을 못 엶"(isolated와 동일).
2. **non-execution/transfer(QwQ-특이·deficit 전량)**: reasoning-RL 모델이 **난관서 성급히 포기·transfer**(24% vs 13%) +
   **출력 포맷 누수**로 실행 누락. ⇒ **thinking이 추가한 *새* 실패모드 = 실행/지속 저하**(행동선택 저하 아님).
3. ⇒ 종합: **reasoning은 QwQ의 결정을 나쁘게 하지 않는다. 실행을 나쁘게 한다**(과숙고→포기/포맷깨짐). isolated criterion
   이득(0.86)이 배포서 사라지는 이유 = (a)⋈ 미해결(공유 난관) + (b)실행/지속 저하(신규 모드).

## 6b. ★사용자 도전(2026-07-08): "실행문제는 scaffold/serving으로 커버 가능 아닌가" → 실측=YES (§5 결론 약화·재판정)
non-execution 73 분해(누수마커·transfer-vs-base 교차):
| 범주 | 수 | 고칠 수 있나 |
|---|---|---|
| **leaked write-JSON in content**(write 생성했으나 파서가 놓침) | **12** | **YES·파서/서빙**(`--reasoning-parser` 부재가 원인) |
| **premature transfer**(base가 그 task ≥3/4 통과=avoidable) | **19** | **YES·scaffold gate**(transfer 억제) |
| mid transfer(base 1-2/4) | 17 | 부분 |
| clean no-action(추론후 무행동·누수 없음) | 10 | 어려움(행동) |
| other leak, no write | 8 | 애매(파서 가능) |
| legit transfer(base도 0/4) | 7 | 아니오(진짜 난관·base도 실패) |
- **누수는 systematic**: 92/456(20%) content에 `</tool_call>`·`<|im_start|>` 마커(passing 16·wrong-exec 53 포함). **내 QwQ 서빙이
  `--reasoning-parser`(deepseek_r1) 없이 hermes만** → `<think>` 미분리로 tool-call 추출 파손. **=방법 결함**(QwQ capability 아님).
- **★재판정**: QwQ deficit 52 = non-exec 초과(+61). 그 중 **~30-40(파서 12 + premature-transfer 19 + other-leak 8)이 배관-fixable**.
  QwQ wrong-exec는 이미 base ≤(181<190). ⇒ **"thinking이 agentic을 해친다"는 robust하지 않음**(내 서빙 handicap + transfer 성향에
  교란). 공정한 판정 = **reasoning-parser 서빙 + 경량 실행 scaffold(누수 tool-call 재추출 + premature-transfer gate)** 재run 필요.
- **thesis 정합(중요)**: 이 그림은 오히려 우리 논지에 부합 — **scaffold가 실행/포맷 실패를 커버하고 모델은 추론 제공**. QwQ+파서+실행-
  scaffold가 base 대등/초과면 = "scaffold가 thinking을 배포가능하게 함"(offload 논지). **유일 non-fixable 잔여 = ⋈ operand 경계**
  (단 base와 공유·thinking-특이 아님) + 소량(clean-noaction 10·legit-transfer 7).

## 6c. ★Step1/2 실측 = 재판정 확증 (2026-07-08·서빙 수정)
- **Step1(무료·구조 테스트)**: QwQ를 `--reasoning-parser deepseek_r1`(+hermes)로 재서빙 → `<think>`가 reasoning_content로
  분리·content 청정(누수 마커 0)·tool_calls 정상 파싱("go ahead"에 exchange write call 방출). 서빙이 누수 원인 확인.
- **Step2(유료·nt=1 full·`qwq_rparser_floor_nt1`)**: avg_reward **0.443→0.526**(+8.3pp)·**leak 20%→1%(1/114)**·non-exec
  29%→11%·transfer 24%→7%. **base 0.557과 3.1pp 차=nt=1 노이즈 안**. ⇒ **deficit의 ~⅔가 내 서빙 handicap**(reasoning-parser
  부재)이었음이 실증. **"thinking이 agentic 해친다"=철회 확정**. 올바로 서빙하면 reasoning-model ≈ base(+QwQ가 28 task는 base
  초과). 잔여 3pp=noise 내·(옵션)transfer-gate로 추가 회복 여지. **정확 pass 확정=nt=4 full 재run(Step3) 필요**(nt=1 noise).
- thesis 함의: **scaffold/serving이 실행실패 커버·모델은 추론 제공** 실증. 유일 non-fixable=⋈ operand 경계(base 공유).

## 7. ★"reasoning-parser로도 왜 base를 못 넘나" 스텝별 전수 (2026-07-08·qwq_rparser_floor_nt1 vs base·per-case)
> 제약: **reasoning_content 미저장**(tau2가 버림) → 내부 추론 불가·**행동/발화 스텝**만 대조. nt=1(단일 trial·noise 有·Step3 nt=4가 확정).
- **landscape**: RP-LOSS 12(QwQ 0/1·base≥3/4) = **RP-WIN 12**(QwQ 1/1·base≤1/4) → **12승=12패=순0=parity**(0.526 vs 0.557).
- **WIN 12 패턴**: base가 **wrong-execution**(잘못된 operand/args)로 실패(11/12)·t33은 base 6-write **over-action**. QwQ 추론이
  **단일 결정을 정확히**(올바른 operand·scope·over-action 회피) → 1 correct write. = **reasoning 가치=per-decision 정확도**.
- **LOSS 12 스텝별 분류**(gold write vs QwQ 실제 write vs base 실제 write):
  | 유형 | task | 내용 |
  |---|---|---|
  | premature transfer(포기) | t13·t30·t106 | 복잡 multi-write(3~4 write)·인증서 즉시 transfer. base는 완수 |
  | non-exec(읽고 무행동) | t47·t49·t87 | 주문 여러개 read만·write 0. base는 완수 |
  | incomplete multi-write | t42(2/4)·t55(1/4) | 일부 write만·나머지 누락. base 전부 완수 |
  | over-action(더 함) | t5(exchange×2 스퓨리어스+return+transfer)·t62 | 안 시킨 변경 추가 |
  | ⋈ wrong-order | t6(#W7800651≠gold #W6390527) | 틀린 주문 선택 |
  | variant wrong | t58(new_item 3709608322≠gold 3815173328) | 틀린 변형 |
- **★핵심(집계)**: LOSS 지배 = **실행/지속 실패 on 복잡 multi-action task = 8/12**(transfer 3 + non-exec 3 + incomplete 2).
  operand(⋈/variant) 단 2/12·over-action 2/12. ⇒ **잔여는 operand-추론이 아니라 orchestration/persistence**.
- **★답(왜 순이득 0)**: **reasoning은 per-decision 정확도를 사지만(WIN 12) multi-step 실행/지속을 희생한다(LOSS 8)**. 복잡
  task서 QwQ는 과숙고→포기(transfer)·미완(부분 write)·무행동(read-only). 근거 정합: QwQ turns/calls 적음(6.3<8.4)=**조기종료
  편향**(reasoning-RL이 "think→answer once"용이라 긴 tool-loop 지속에 역행). base(instruction-tuned)가 multi-write 완주 더 잘함.
  agentic 벤치가 multi-action 지배라 (a)결정정확도↑ vs (b)실행지속↓이 상쇄 → parity.
- **fixability/thesis**: LOSS 8 orchestration은 **completion-enforcing scaffold**(premature-transfer gate + "요청 변경 전부 완료까지 지속")로
  대부분 복구 가능 → QwQ+scaffold가 base 초과 가능(thesis: scaffold=orchestration·모델=reasoning). non-fixable=⋈/variant 2.
- caveat: nt=1 단일trial(Step3 nt=4 확정)·reasoning_content 미저장(행동레벨 추론)·QwQ≠Qwen2.5(RL/템플릿).

## 7b. ★내부 추론 분석 (reasoning_content = `raw_data.choices[0].message.reasoning_content`·재실행 불요·[[09]])
give-up/non-exec LOSS 6건의 결정지점 추론 정독 → **일관 패턴 = "friction 만나면 policy-literal escalation/give-up"**:
- **t13**(transfer): *"if I can't handle the request within the tools provided, I should transfer"* — get_order_details 1회 실패를 unrecoverable로 단정. base는 완수(복구가능했음).
- **t30**(transfer): *"the user might have made a mistake... transferring is the best option"* — **최신 주문 하나만** 보고 못 찾자 포기. gold는 다른 주문들(#W2692684/9373487/7449508). **전수 탐색 안 함**.
- **t47**(non-exec): *"despite correct ID... perhaps a data entry error... I should transfer"* — lookup 실패를 system-error로 오귀속. base는 찾음.
- **t106**(transfer): *"I can't authenticate her... policy says to deny requests that can't be authenticated"* — 인증 friction서 정책-literal deny. base 인증 성공.
- **t49**(non-exec): *"all checked orders don't have earbuds... I can't proceed"* — item↔order 매칭 실패를 "물건 없음"으로 단정·포기(gold item은 #W3470184에 있었음). = ⋈/item-match 실패를 give-up으로.
- **t87**(non-exec): *"no orders with a Washington DC address"* — **mis-formalize**: 목표주소로 *변경*할 task를 그 주소를 *가진* 주문 *검색*으로 오해→못 찾음→포기.
- **★핵심**: QwQ 추론이 friction(lookup 실패·인증난관·item 불일치)서 **"정책상 못 하면 escalate/deny"로 스스로를 설득**한다. 숙고할수록
  give-up을 정당화. = **reasoning-RL 서명**(단일문제 careful-correct용 학습이 agentic friction을 "못 풂→escalate"로 처리). base는
  추론 적어 그냥 계속 tool 호출→우연히 완수. **즉 reasoning이 persistence를 *깎는다***(WIN의 decision-정확도 이득을 상쇄).
- **fixability**: 대부분 give-up은 **completion-forcing scaffold**(transfer 차단·"exhaust 전 escalate 금지"·전수탐색 강제)로 복구. 단
  mis-formalize(t87·t49 일부)는 task-해석 오류라 harder. ⇒ scaffold가 persistence 담당·모델은 decision → base 초과 경로 확인.

## 7c. ★게이트-관련 실측 영속 (2026-07-08·외부 리뷰 A 지적 반영 — 인용만 하고 미영속이던 수치)
데이터=`qwq_rparser_floor_nt1`(nt=1·114) + 대조 `fl32b_floor_retail_t4`(nt=4·456). 스크립트=scratchpad `verify_review`(재현 가능).
- **(A) naive 게이트 over-block(=passing sim 발화)**: 조건 `transfer OR (writes < base-max-writes)` → **passing 19건 발화**.
  내역: transfer-type **5**(t10·t12·t25·t46·t50 = **transfer가 gold**·확실 파손) + `nw<base-max` **14**(t1·t11·t14·t22·t28·t33·
  t45·t48·t50·t56·t57·t64·t65·t66·t83류). ★**14는 나쁜 프록시**(valid한 짧은 해법을 미완으로 오판) ⇒ **정밀 게이트의 over-block은
  여전히 미측정**. ★§6b의 premature-transfer **19**(old qwq nt=4·*benefit*)와 **수치 우연 일치**(별개 측정·혼동 주의).
- **(D) over-action 기준선(spurious write = gold action에 없는 write)**:
  | | spurious 총계 | **passing sim 내** | ≥1 spurious 보유 sim |
  |---|---|---|---|
  | QwQ-rp (nt=1·114) | 12 | **0** | 8 (7.0%) |
  | base (nt=4·456) | 130 | **47** | 80 (17.5%) |
  ⇒ **QwQ의 낮은 over-action = give-up 성향의 뒷면.** 완결게이트가 종료를 막으면 base 쪽(over-action)으로 밀어낼 구조적 위험.
  선례가 이 축을 명시 금지(`NEXT_DET_LEVERS:131` "do NOT gate" · `NEXT_LEVERS:33` "레버 금지").
- **(C) LOSS-12 슬라이스 정정 원장**(writes/xfer/spurious 실측): ①transfer{t13,t30,t106} ①'무행동종료{t47} ②미완{t42(2/4),
  t55(1/4)} ③mis-formalize{**t87 단독**} operand{t6,t58,**t49**} over-action{t5(sp2),t62(sp1)}. ⇒ **게이트-addressable=6**
  (옛 "실행/지속 8/12"는 t87·t49를 잘못 포함·철회).

## 6. caveat / [[08]] 규율기록
- **QwQ≠Qwen2.5-32B**(RL-튜닝 다른 모델): transfer-성향·포맷누수는 QwQ 특유 학습/템플릿일 수 있음("thinking 자체" 단정 불가).
  clean thinking-격리 = base-8k 통제(같은 weights·⋈ 0 이득)가 담당·본 forensic은 *reasoning-model-as-agent* 판정.
- **포맷 누수 = 서빙/파서 상호작용**(배포 아티팩트)이지 순수 capability 경계 아님.
- **자기수정 2회 기록(방법 정직)**: ①`tools=[None]`=내 `function.name` 오추출(실은 top-level `name`)·tool call 정상.
  ②writes()가 `m["name"]`(assistant엔 없음) 읽어 "100% non-execution" 오산출 → pass-sims-0%-write 새너티가 적발 →
  `tool_calls[].name`로 수정 후 실측(29% non-exec). **집계→결론 직행이 아니라 새너티+재추출이 두 오판을 막음.**
