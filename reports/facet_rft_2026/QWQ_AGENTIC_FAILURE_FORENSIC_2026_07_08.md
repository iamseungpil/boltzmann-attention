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

## 6. caveat / [[08]] 규율기록
- **QwQ≠Qwen2.5-32B**(RL-튜닝 다른 모델): transfer-성향·포맷누수는 QwQ 특유 학습/템플릿일 수 있음("thinking 자체" 단정 불가).
  clean thinking-격리 = base-8k 통제(같은 weights·⋈ 0 이득)가 담당·본 forensic은 *reasoning-model-as-agent* 판정.
- **포맷 누수 = 서빙/파서 상호작용**(배포 아티팩트)이지 순수 capability 경계 아님.
- **자기수정 2회 기록(방법 정직)**: ①`tools=[None]`=내 `function.name` 오추출(실은 top-level `name`)·tool call 정상.
  ②writes()가 `m["name"]`(assistant엔 없음) 읽어 "100% non-execution" 오산출 → pass-sims-0%-write 새너티가 적발 →
  `tool_calls[].name`로 수정 후 실측(29% non-exec). **집계→결론 직행이 아니라 새너티+재추출이 두 오판을 막음.**
