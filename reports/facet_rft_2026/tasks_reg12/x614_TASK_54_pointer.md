# TASK_54 (t7391_reg12) — 본문 위치 포인터

정본 본문 = `reports/facet_rft_2026/tasks__20260829/TASK_54.md`

이 디렉터리(`tasks_reg12/`)에 `TASK_54.md` 를 직접 두지 못한 이유:
`.claude/hooks/scaffold_guard.py:200-201` 의 런별-포렌식 예외 술어가
```python
_is_run_task_report = re.match(r"^TASK_\d+[a-z]?\.md$", base) and re.search(r"/tasks_+\d{8}/", fp)
```
라 `tasks_reg12/TASK_54.md` 는 `§74-b 보고서 신설 차단`(exit 2)에 걸린다. 훅을 우회하지 않고
정본 명명 `tasks__<날짜>/TASK_<id>.md` 를 따랐다(형제 `x602_TASK_3_pointer.md` …
`x613_TASK_28_pointer.md` 와 동일 선례).

딸린 격리 프로브: `reports/facet_rft_2026/x614_t7391_task54_confirm_iso.py`
(모델 호출 0 · 프롬프트 저작 0 · gold 무참조 · G2 반사실 4 + 행동결속 부정통제 + 코퍼스/총액 감사)

⚠데이터 파일명 정정: 태스크 지시문의 `bank_t7391_retail_20260829_undefined_reg12.results.json.gz`
는 로컬에 없다. 실제 = `sim_results/t7391_reg12.results.json.gz` (로그·사이드카·trace 는 **미회수**
⇒ `[T2_*]` 마커 계수·`t2_liveness` 불가).

---

## 한 줄 요약

`reward 0.0 = DB 0.0 × NL_ASSERTION 0.0` — **두 축이 서로 다른 지점에서 각각 죽었다** (독립 실패).

**A축(DB · msg 29→30)**: 변이 집합 = **MISSING 2 · WRONGARG 2 · EXTRA 0 · DUP 0 · BLOCKED 1**.
`order_id` 2/2 정확, **`reason` 필드만** 틀렸다 — 보낸 값 `"ordered by mistake"` ↔ gold
`"no longer needed"`. 경로: 손님 msg 1 축자 *"I've had a **financial issue**"* → 모델이 그 낱말을
`reason` 에 전사(msg 26 제시 *"Reason: Financial issue"* · msg 28 호출) → env 가 **`Error: Invalid
reason`(19자·허용값 미고지·[[64]] 위반)** → msg 30 에서 **손님에게 묻지 않고** 정책 열거문의 두
값 중 하나를 자기가 골랐다. 정책은 그 자리에서 *"The user needs to confirm the order id **and the
reason**"* 을 요구했다. 두 허용값은 msg 0~29 코퍼스에 **각 0건**(`x614 E_CORPUS`).
우리 층: `T2_RULE_AT_WRITE=1` 인데 retail `write_rules` **부재**(`_declared_rules_for → None`) ·
`write_arg_enum` **부재** · `tool_error_specs` 에 `invalid reason` 매치 **부재**.

**B축(NL · msg 37 → 42 → 43)**: gold `$3,646.68` ↔ 보고 `$2,460.21`. **오퍼랜드 2/3** —
`1186.47` 이 빠졌는데 그 값은 msg 43 시점 문맥에 **msgs [19, 38] 2회 실재**했다(재료 결손 아님).
경로: msg 37 이 **본문 `''`** 로 `return_delivered_order_items(#W4597054)` 를 실행했고(이것은
**gold**·`action_match=true`), 손님이 msg 42 에서 *"**what is #W4597054? I didn't ask to return
anything else**"* 로 되묻자 모델이 msg 43 에서 *"we will not proceed with the return request"* 라며
**자기 gold write 를 부인**하고 총액에서 뺐다.

**막았어야 할 게이트**: `G2_CONFIRM_WRITE` 는 `applies_to` 에 `return_delivered_order_items` 를
**명시적으로 포함**하는데, msg 37 을 통과시킨 확인 토큰은 **10 메시지 전 msg 27 의 `"Yes"`**,
그 문장은 축자로 *"Yes — **cancel both pending orders** #W4836353 and #W7342738"* 이다.
코드 경로 = `gate_interpreter.py:387-390` (confirm 분기가 `args` 를 **한 번도 읽지 않는다**) +
`:16-18` CONFIRM_RE + `t2_gate_patch.py:6938-6944` `_regen_last_user`.
격리 `x614 A_LIVE` 가 이 통과를 **정확히 재현**하고, `B_ACTBIND`(도구 어간 결속)가
**cancel 3 통과 / return 1 차단** 으로 뒤집는다([[57]] 부정통제 포함) ⇒ **NL 축 격리 100%**.
⚠DB 축은 격리 **부분**이다 — `B_WAG` 가 오답을 잡지만 `N_NEG2` 에서 **정답 `"no longer needed"`
도 같은 이유로 잡는다**(그 비용은 `TASK_28.md §7 P2` 가 이미 측정·재유도 금지).

**대조군 대조**: `hist_gpt52_reg12_PASS` task 54 = **같은 32B 모델 · 같은 seed 626729 · trial 0 ·
reward 1.0**. 갈린 곳 둘 — ⑴ 손님 첫 발화가 *"Money's tight right now."* 라 `reason` 자리에
전사할 낱말이 없었고 모델이 msg 18/20 에서 `"no longer needed"` 를 한 번에 냈다 ⑵ 반품을
msg 28 에 나열해 msg 29 손님 축자 *"**Yes, confirm the return for those items from #W4597054**"*
를 받은 뒤 실행했고, msg 32 에서 *"$2,460.21 + $1,186.47 = **$3,646.68**"* 로 닫았다.
⚠sha 상이(`5ebebbe8` ↔ `fc0055dc`) ⇒ 통제 실험이 아니라 참조.

**선행 대조**: `TASK_28.md §3-4` 와 **같은 G2 구멍**이되 **새 손해 형태**다 — 28 은 *gold 아닌
write 를 통과*시켰고, 54 는 ***gold write 를 손님 모르게 통과시켜 나중에 부인하게 만들었다***.
`TASK_12`(확인 아닌 발화가 확인으로) 와 합쳐 **같은 게이트의 구멍 3종**. `TASK_28.md §7 P2` 가
*"대조군에서도 gold 취소 2건(task 16 msg 10 · **task 54 msg 18**)이 거부된다"* 로 이 태스크를
이미 측정해 두었다 — 인용만 하고 재유도하지 않았다([[74]]).
`grep -rln "task 54|#W4597054|amelia_silva_7726" reports/facet_rft_2026/` → task 54 **전용** 선행
보고서는 **없다**(히트는 `TASK_28.md` 와 이 세션 프로브뿐).
