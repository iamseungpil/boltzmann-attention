# TASK_24 (t7391_reg12) — 본문 위치 포인터

정본 본문 = `reports/facet_rft_2026/tasks__20260829/TASK_24.md`

이 디렉터리(`tasks_reg12/`)에 `TASK_24.md` 를 직접 두지 못한 이유:
`.claude/hooks/scaffold_guard.py:200-201` 의 런별-포렌식 예외 술어가
```python
_is_run_task_report = re.match(r"^TASK_\d+[a-z]?\.md$", base) and re.search(r"/tasks_+\d{8}/", fp)
```
라 `tasks_reg12/TASK_24.md` 는 `§74-b 보고서 신설 차단`(exit 2)에 걸린다. 훅을 우회하지 않고
정본 명명 `tasks__<날짜>/TASK_<id>.md` 를 따랐다(형제 `x602_TASK_3_pointer.md` ·
`x603_TASK_4_pointer.md` · `x603_TASK_9_pointer.md` · `x611_TASK_12_pointer.md` ·
`x612_TASK_22_pointer.md` 와 동일 선례).

딸린 격리 프로브: `reports/facet_rft_2026/x613_t7391_task24_gate_iso.py`
(모델 호출 0 · 프롬프트 저작 0 · 게이트 반사실 3 + 후보요약/operand요약 생성 확인 + calc_specs 감사)

⚠데이터 파일명 정정: 태스크 지시문의 `bank_t7391_retail_20260829_undefined_reg12.results.json.gz`
는 로컬에 없다. 실제 = `sim_results/t7391_reg12.results.json.gz` (로그·사이드카·trace 는 **미회수**).

---

## 한 줄 요약

`reward 0.0 = DB 0.0 × NL_ASSERTION 0.0` — **두 점수축이 동시에 죽었다.**

**A축(DB)**: gold write 0건인데 msg 13 에서 `cancel_pending_order(#W3561391)` 을 **확인 없이**
실행했다. 태스크 설계는 *"if the agent asks you to confirm, you regret and want to keep it"* 이라
확인만 물었으면 통과였다. 막았어야 할 `G2_CONFIRM_WRITE` 가 **인증 턴의 "Sure"** 에 열렸다
(`gate_interpreter.py:16-18` CONFIRM_RE · `:387-390` confirm 술어 ·
`t2_gate_patch.py:1091`·`:1276-1283` `_last_user_text`). 격리 재현(`x613`):
last_user=msg[3] → **allow**, last_user=msg[1] → **deny(G2)**. 두 번째 기회였던
`G6_SELECT_CONFIRM` 은 리졸버 침묵으로 미발화(런 전수 0 · 기전 UNPROVEN).

**B축(NL)**: msg 19 에서 **서로 다른 두 주문의 티셔츠를 한 짝으로 접합**했다 —
`#W6876713`(T-Shirt 1개·cotton)을 답으로 고르고, 거기에 `#W9609649` 의 blue 티셔츠를 붙여
*"둘 다 cotton"* 이라 보고. 정답 `#W9609649` (cotton + **polyester**) 의 전문은 msg 11 에
**1,605자로 문맥에 있었다** ⇒ 커버리지 결손 아님 · 닫힌 술어(`count(name=='T-Shirt')==2`) 실패 =
**모델 귀속**. 대조군에서 이 선택을 지탱한 `[DISAMBIGUATION NOTE]`(3,035자)는 이 런에 **0자**
들어갔다(`T2_PRESENT_READS`·`T2_PRESENT_NESTED` 러너 미수출 · `t2_gate_patch.py:1096`·`:1100`).

**대조군 대조**: `hist_gpt52_reg12_PASS` task 24 = **같은 32B 모델 · 같은 seed 626729 · reward 1.0**.
tool 응답 꼬리에 우리 층 문면 msg9 +3,035자 / msg13 +1,265자가 붙었고, 모델은 `get_order_details`
를 **한 번도 부르기 전에** *"the order #W9609649 contains two t-shirts"* 라고 답했다.
