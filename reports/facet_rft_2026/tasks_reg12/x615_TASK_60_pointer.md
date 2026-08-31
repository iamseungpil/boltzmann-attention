# TASK_60 (t7391_reg12) — 본문 위치 포인터

정본 본문 = `reports/facet_rft_2026/tasks__20260829/TASK_60.md`

이 디렉터리(`tasks_reg12/`)에 `TASK_60.md` 를 직접 두지 못한 이유:
`C:\workspace\.claude\hooks\scaffold_guard.py:200-201` 의 런별-포렌식 예외 술어가

```python
_is_run_task_report = re.match(r"^TASK_\d+[a-z]?\.md$", base) and re.search(r"/tasks_+\d{8}/", fp)
```

라 `tasks_reg12/TASK_60.md` 는 `§74-b 보고서 신설 차단`(exit 2)에 걸린다. 훅을 우회하지 않고
정본 명명 `tasks__<날짜>/TASK_<id>.md` 를 따랐다(형제 `x602_TASK_3_pointer.md` …
`x614_TASK_54_pointer.md` 와 동일 선례).

딸린 격리 프로브: `reports/facet_rft_2026/x615_t7391_task60_confirm_iso.py`
(모델 호출 0 · 프롬프트 저작 0 · gold 는 오차단 계수에만 · N_NEG 부정통제 2종 + 팔 A/B/C/D 전수)

⚠**데이터 파일명 정정**: 태스크 지시문의 `bank_t7391_retail_20260829_undefined_reg12.results.json.gz`
는 로컬에 없다. 실제 = `sim_results/t7391_reg12.results.json.gz`. 로그·사이드카·trace 는 **미회수**
⇒ `[T2_*]` 마커 계수·`t2_liveness` 불가. 대조군 지시 `undefined.results.json.gz` 도 부재 —
대신 `hist_gpt52_reg12_PASS.results.json.gz`(task 60 = **reward 1.0** · 같은 seed 626729)를 참조로 썼다.

⚠**sha 핀 불가**: 런 sha `fc0055dc…` 가 로컬 repo 에 없다(`git cat-file -t` 실패 · HEAD=`0b612169`).
차선으로 worktree↔HEAD 대조 — `a2/retail.gate.json` 의 `gates`·`present_specs` 는 HEAD 와 바이트
동일(워크트리 수정은 `failure_markers` 추가분뿐) · `gate_interpreter.py`/`t2_gate_patch.py` 수정 0.

---

## 한 줄 요약

`reward 0.0 = DB 0.0 × NL_ASSERTION 0.0`. 변이 = **MISSING 1 · WRONGARG 1 · EXTRA 0 · DUP 0 ·
BLOCKED 0**. 4개 인자 중 **`new_item_ids` 한 필드만** 틀렸다 — `8555936349`($226.49) ↔ gold
`6077640618`($242.92). NL 축이 요구한 `$242.92` 도 같은 선택의 그림자다.

**결정점 = msg[8] 단 하나.** 모델이 *"I am now proceeding to make the change."* 라는 본문과 write
도구호출을 **한 메시지에 함께** 내면서 손님 턴을 건너뛰었고, 그 write 가 **게이트를 하나도 안 맞고**
실행됐다.

**우리 층(재현 100%)**: `G2_CONFIRM_WRITE` 가 통과시켰다. 통과 토큰은 손님 **최초 요청** msg[1] 의
`sure` — 축자 *"Please **make sure** the price is the same or lower … and **confirm that explicitly
before making the change**."* **확인을 요구하는 문장이 확인 게이트를 열었다.**
코드 = `gate_interpreter.py:16-18`(`CONFIRM_RE`) · `:387-390`(confirm 분기가 `args` 미참조) ·
`t2_gate_patch.py:6938-6944`(`_regen_last_user`).
격리 `x615`: A_LIVE `(True,None,None)` 재현 · 낱말 2개만 치환한 N_NEG1 과 무관 발화 N_NEG2 는
둘 다 `(False,'G2_CONFIRM_WRITE')`.

**⚠게이트 단독으로는 이 태스크를 못 산다.** 시나리오 축자 *"**If and only if the agent provides
several options**, you want the option without water resistance."* — 정답의 판별 술어가 **손님에게서
유도되어야 하는 값**이다. 후보는 3개(blue ∧ available ∧ ≤$256.67)였고 그중 `not resistant` 는
`6077640618` 유일. 대조군은 msg[10] 에 2안을 나열한 **뒤에야** msg[11] 에서 손님 선호를 받았다.

**선행 대조**: 원인은 `TASK_12.md:257-259` 가 이미 `x611b` 센서스로 지목했다(*"task 60 이 이 결함의
순수형"*). 새 원인 주장 **없음**. 이 보고서의 신규 소득은 둘 —
① **`TASK_12 §9 P1` 의 값싼 조작화(`prevTxt`)를 반증**: 팔 B 가 팔 A 와 **41 write 전수 동일**
(task 60 의 `prevTxt=True` 는 msg[0] 인사말 *"Hi! How can I help you today?"* 때문).
② **손님-조건부 선호 유도** 기전 — 형제 12편 어디에도 없다.
