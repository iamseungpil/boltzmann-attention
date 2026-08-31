# FAILURE_MASTER_reg12 — `bank_t7391_retail_20260829` (실물 태그 `t7391_reg12`) 전수 실패 포렌식 종합

작성 2026-08-29 · 입력 = 태스크당 1 에이전트 per-step 포렌식 12편 + 반증자(refuter) 판정 59건
정본 1차 자료 = `reports/facet_rft_2026/tasks__20260829/TASK_{1,3,4,9,12,16,22,24,28,54,58,60}.md`

> ⚠**파일명 주의**: 지시 경로는 `FAILURE_MASTER_reg12.md` 였으나 `C:\workspace\.claude\hooks\scaffold_guard.py:202-213`
> (§74-b · [[31]] 규칙 ①)이 `reports/` 아래 **신설 .md 를 `xNNN_*` 이외 이름으로는 exit 2 차단**한다.
> 훅을 우회하지 않고 형제 12편이 이미 쓴 방식(프로브형 명명 + 사유 명시)을 따랐다.
> 찾아본 곳: `ls reports/facet_rft_2026/ | grep -i "FAILURE_MASTER|STATE_OF"` → `FAILURE_MASTER__20260822.md` ·
> `T7336_FAILURE_MASTER_2026_08_22.md` · `STATE_OF_PLAY_2026_08_23.md` (전부 다른 런) · `ls tasks__20260829/` → TASK 12편.
> ⇒ 이 런의 마스터는 **갱신할 앞 문서가 없다**.

⛔이 문서는 **입력에 있는 수치만** 쓴다. 새 수치를 유도하지 않았고, 코드·선언·러너 수정 0 · git 커밋/push 0 · SSH 0.

---

## §0 재료 실사 — 무엇을 읽고 무엇을 못 읽었나 ([[30]] · [[77]])

| 항목 | 실물 | 판정 |
|---|---|---|
| 결과 | `sim_results/t7391_reg12.results.json.gz` (12 sim · nt=1 · 전부 reward 0.0) | 있음 |
| 지시문 파일명 `bank_t7391_retail_20260829_undefined_reg12.*` | 로컬 부재(템플릿 미치환) | **없음** — 12편 중 7편이 독립 확인. 찾아본 곳: `ls sim_results/ | grep -i "7391|reg12"` |
| `.log.gz` · `fb_*` · `trace_*` 사이드카 | 0건 회수 (`t2_forensic.sidecar_paths('t7391_reg12')` → `[]`) | **없음** ⇒ stderr `[T2_*]` 마커 계수 불가 · `t2_liveness`([[55]] 0단계) 불가. 찾아본 곳: `ls sim_results/ | grep -E "^(fb|trace)_"` · `find ba-frft -name "*7391*"` |
| 런 sha `fc0055dc4e0a…` | 로컬 repo 에 객체 없음(`git cat-file -t` fatal · `git rev-list --all \| grep ^fc0055` 0) | **엔진 sha 미기록**. 미커밋 `t2_run_gated.py` 주석 축자: *"`info.git_commit` 은 cwd 의 sha … **벤치마크 sha 이지 우리 엔진 sha 가 아니다**"* ⇒ 모든 파일:줄 인용은 워크트리(HEAD `0b612169`) 대조 |
| 러너 | `run_t7391_retail.sh` 는 태그 `bank_t7391_retail_(smoke_)20260829`·retail 전수 114 를 만든다. `reg12`(12 태스크) 생산자는 repo 에 없다(`grep -rln reg12 --include=*.sh` 0건) | 러너 텍스트 근거의 일부는 **다른 런처에 대한 진술**일 수 있다 |
| 대조군 `hist_gpt52_reg12_PASS` | 2026-07-06 · sha `5ebebbe8` · nt=4 · **PASS 만 추린 큐레이션** | **통제 A/B 아님**(task1 E2 = REFUTED). 참조로만 |

⇒ 이 런은 [[70]] 레버 판정 의무 3종(①전체 reward 짝 A/B ②태스크별 부호표 ③무엇을 팔았나) 중 **0/3** 을 충족한다.

---

## §1 성적 표

성적 문장은 제공되지 않았다. 아래는 입력의 `reward_info` 를 그대로 옮긴 것뿐이다.

**12 sim 전수 reward 0.0 · trial 1개씩(nt=1) · reward_basis 전부 `['DB','NL_ASSERTION']`**

| task | DB | NL_ASSERTION | 변이 집합(`mutating_tools('retail')`) | 점수를 정한 칸 |
|---|---|---|---|---|
| 1 | 0.0 | 1.0 | MISSING 1 (순수 · wrongarg/extra/dup 0) | gold write `exchange_delivered_order_items` 미실행 |
| 3 | **1.0** | 0.0 | clean (gold 1 · matched 1 · 나머지 0) | 발화 숫자 `12` ↔ gold `10` |
| 4 | **1.0** | 0.0 | clean (gold 2 · matched 2 · 나머지 0) | 발화 숫자 `12` ↔ gold `10` |
| 9 | 0.0 | 1.0 | MISSING 1 + WRONGARG 1 (같은 도구) | `item_ids` 초과 1 + `new_item_ids` 오선택 |
| 12 | 0.0 | 1.0 | EXTRA 2 + BLOCKED 1 (**gold write 0건**) | 미확인 `return_delivered_order_items` 2회 실행 |
| 16 | 0.0 | 0.0 | MISSING 1 + BLOCKED 1 (실질 WRONGARG 1) | `item_ids` 한 칸(cross-order 바인딩) |
| 22 | 0.0 | 1.0 | WRONGARG 2 | `address2` 한 칸(`'Suite 865'` ↔ gold `''`) |
| 24 | 0.0 | 0.0 | EXTRA 1 | 미확인 `cancel_pending_order` + 주문 간 품목 접합 |
| 28 | 0.0 | 0.0 | EXTRA 1 | 금지된 전체-취소 1건(환불 $1,619.34 가 총액 오염) |
| 54 | 0.0 | 0.0 | MISSING 2 + WRONGARG 2 + BLOCKED 1 | `reason` 필드 + 자기 gold write 부인 |
| 58 | 0.0 | 1.0 | MISSING 1 + WRONGARG 2 (뿌리는 한 슬롯) | `new_item_ids[1]` 한 칸 |
| 60 | 0.0 | 0.0 | MISSING 1 + WRONGARG 1 | `new_item_ids` 한 칸 |

**축별 집계**: DB=0 → **10/12**(3·4 제외) · NL_ASSERTION=0 → **7/12**(3·4·16·24·28·54·60) · **두 축 동시 0 → 5/12**(16·24·28·54·60).
⇒ 16·24·28·54·60 은 우리-층 수리만으로는 reward 가 오르지 않는다(§7-⑦).

**변이 형태 분포**: 순수 MISSING 1(task 1) · clean-but-NL 2(3·4) · **인자 한 칸 오염 5**(9·16·22·58·60) · **gold 에 없는 write(EXTRA) 3**(12·24·28) · 복합 1(54).
⇒ 지배적 형태는 **"도구·order_id 는 맞는데 인자 한 칸이 틀림"** 과 **"해서는 안 될 write 를 함"** 둘이다.

---

## §2 원인 축별 군집표 (축은 데이터에서 나왔다)

> 이 절의 각 축은 §2-Z 의 **반증조건(refutation) 표**와 짝을 이룬다. 반증조건이 없는 축은 원인으로 쓰지 않았다.

### A축 — 확인 게이트 `G2_CONFIRM_WRITE` 의 술어 결손 (**최대 군집**)
> 코드: `gate_interpreter.py:16-18`(CONFIRM_RE) · `:387-390`(confirm 분기 3줄, `args` 미참조) · `t2_gate_patch.py:6937-6942`(`_regen_last_user` = 뒤에서 처음 만난 user 메시지를 턴 종류 구분 없이 반환)
> 선언: `a2/retail.gate.json` G2 predicate 축자 *"explicit user confirmation (yes) **of the action details** in the latest user message"* ⇒ **구현이 선언보다 엄격히 약하다**([[22]])

| 하위 변형 | 태스크(sim) | 대표 축자 | 점수 영향 |
|---|---|---|---|
| ①확인 아닌 발화가 확인으로 읽힘(인증 턴) | 9 · 12 · 24 | `"Sure—my name is Mei Kovacs…"`(9#msg3) / `"Sure—my email address is …"`(12#msg3) | 3건 전부 **사망** |
| ②진짜 확인이 **다른 행동**에 재사용 | 28 · 54 | 28: 확인=품목 제거, 실행=전체 취소 / 54#msg27 `"Yes — cancel both pending orders"` 가 msg37 `return_delivered_order_items` 를 염 | **사망** |
| ③확인된 **인자값**이 실행 시 변경 | 54 | 제시 `reason: Financial issue`(msg26) ↔ 실행 `ordered by mistake`(msg30) | 사망 |
| ④유보문·**요구문**이 게이트를 염 | 3 · 60 | 3#msg15 `"…before I confirm any changes?"` / 60#msg1 `"Please **make sure** … and **confirm that explicitly before** making the change."` | 3=0 · **60=사망(순수형)** |
| ⑤부정 문맥 | 12 | 12#msg19 `"I'm **not okay** with…"` → `okay` 매치 | (이미 사망 후) |

- **전수 센서스**(반증자 독립 재검산): 커밋된 write 호출 **26건 · 비오류 실행 22건이 전부** CONFIRM_RE 로 통과했고, 그중 **최초요청/인증 턴 토큰으로 열린 것이 6호출(실행 5건)**. 예측↔관측 대조 **28/28 일치 · 불일치 0**.
- 게이트 자체는 살아 있었다: 엔진이 붙인 실물 deny **6 sim · 10턴**(문자열 20~24회 · task 1·3·4·16·22·28·54).
- 귀속: **our_layer(그물이 뚫림) + model(행위)**. 반증자 다수 의견 — 우리 층은 *'모델이 그렇게 한 것'* 의 but-for 원인이 아니라 *'그것이 막히지 않은 것'* 의 but-for 원인.
- ⛔**선행 처방 P1(TASK_12:423) 의 값싼 조작화는 무효**: '직전 assistant 텍스트 존재(`prevTxt`)' 로 구현하면 **41 write 전수에서 현행과 완전 동일(0개 차단)** — task 60 은 msg[0] 인사말 `"Hi! How can I help you today?"` 가 술어를 만족시킨다.

### B축 — 닫힌 술어 위의 **빼기/집계 실패** ([[63]])
| 태스크 | 결정 술어 | 문맥 실재 여부 | 모델이 한 것 |
|---|---|---|---|
| 3 · 4 | `count(variants, available==true)` = 10 | msg 9 도구 출력(2,029자)에 12행 available 불리언 전부 실재 | `len(variants)` = 12 |
| 16 | `item ∈ order.items` | msg 9/10/11 에 3주문 전문·fulfillment item_ids 실재 | 이름 표면 매칭 → cross-order 바인딩 |
| 9 | `brightness == high` (1순위 제약) | msg 14 한 줄에 정답 `7624783998` 축자 실재 | medium 선택 + `"a brighter one"` 거짓 요약 |
| 58 | `argmin(price \| available ∧ i7+)` | msg 14 에 17 변형 전문 실재 | 기준 도착 **전** 후보 2개로 절단, 이후 재검토 0 |
| 24 | `count(items.name=='T-Shirt')==2` | msg 11 에 전문 1,605자 실재 | 두 주문 품목을 한 짝으로 접합 |
| 54 | 3항 합 $3,646.68 | 세 피가산수 전부 문맥 실재(msgs 19·38 등) | 자기 gold write 를 부인하고 한 항 제거 |

- **전부 재료 결손이 아니다**(discovery·coverage 결손 0). 귀속 primary = **model**.
- 우리-층 기여는 "그 자리에 놓았어야 할 결정론 산출물이 없었다"(D축)이지 오염이 아니다.

### C축 — retail A2 **선언 결손**(플래그 ON · 재료 0 ⇒ 구조적 침묵)
- 규모 실측(task 28 CONFIRMED): `retail.gate.json` 최상위 **32키**, 엔진이 실제 dict 조회하는 banking 전용 키 중 retail 부재 **45개**(반증자가 45/45 를 `\.get\("key"\)|\["key"\]` 정규식으로 재검산 — 주석 오탐 0).
- 침묵한 재료: `write_rules` · `write_arg_grounding` · `write_arg_enum` · `write_evidence_specs` · `scaffold_get_tools` · `arg_producers` · `require_doc_before` · `follow_up_chains` · `procedures` · `claim_prov` · `completion_guard` · `requires_reads` 계열.
- 선행 `A2_THREE_LAYER_SPLIT_DESIGN_2026_07_31.md:233` 축자 *"t7391(retail)은 A2 칸이 67% 비어 있는 채로 도는 중"* — 이 종합은 그 일반 진술을 **실물 호출 다수로 고정**한다.
- ⚠'45' 는 결손의 **상한**이지 필요분이 아니다(banking KB/디스패처 전용 키가 섞여 있다).

### D축 — **플래그 미수출** (`T2_CALC` · `T2_PRESENT_READS` · `T2_PRESENT_NESTED`)
- 관측: 12 sim 전수 `[COMPUTED FACTS]` **0** · `[DISAMBIGUATION NOTE]` **0** · `[OPERAND DISAMBIGUATION]` **0** ↔ 대조군 16 · 10 · 16.
- 코드: `t2_gate_patch.py:7348`(calc) · `:7345`(present) · `:7347`(nested) — env 미설정 시 스펙이 빈 리스트로 접힌다. `grep` 결과 `go_stack.sh` · `run_t7391_retail.sh` 둘 다 **0건**.
- **형제 8편이 독립 지목**(TASK_1:153 · 3:199 · 4 · 12:347 · 16:242 · 24:297 · 28:286 · 58 · 60) ⇒ 새 발견이 아니라 **8중 중복 관측**([[74]]).
- 인과 등급이 갈린다: **task 4 만 CONFIRMED** — 같은 32B 모델 고정 시 CF 주입 **23/23 met** ↔ CF 없음 **2/13 met**, '주입됐는데 12를 말한' 사례 **0건**. 나머지는 PLAUSIBLE/UNPROVEN.
- ⛔반대 방향 실측: `CENSUS_LEVERS_DESIGN_2026_07_11.md:72` 축자 — **바로 이 task 3 에서** T2_CALC ON 으로 `": 10"` 이 **4/4 주입됐는데도 통과 못 했다**(relay-gap). ⇒ *"플래그만 켜면 산다"* 는 유일한 실측에 의해 반박된다.

### E축 — 우리 층의 **능동 오염**(치환) · 이 런 유일 1건
- task 22: 모델 날조 `address2='Apt 1'`(msg10 raw_data)을 `T2_GROUND`(`t2_gate_patch.py:8435-8445`)가 문맥 유일 후보 `'Suite 865'`(손님 **옛** 주소)로 제자리 치환하고 `continue` 로 거절·재생성을 건너뜀. 격리 재현 **바이트 동일**(같은 tool_call id). 그 값이 msg[11] 도구 결과로 에코된 뒤 점수축 write(msg12)가 복사.
- 계보: 2026-07-11 에 같은 계열(GROUND-VERBATIM empty-게이트)이 **t59 에서 이미 폐기**됐고, 오늘은 다른 입구(값이 빈 문자열이 아니라 날조)로 들어왔다.
- ⚠**과잉결정 가능**: 치환이 없었어도 `'Apt 1'` 이 에코되어 gold `''` 와 달랐을 개연이 있다 ⇒ *"T2_GROUND 가 없었으면 통과했다"* 는 따라 나오지 않는다.

### F축 — 게이트 kinds / `G6_SELECT_CONFIRM` **미해결 축**(형제 간 상충)
- `T2_GATE_KINDS` 미설정 ⇒ retail 게이트 **8종 전량 활성**(정본은 6종 · `select_confirm` 제외). **CONFIRMED**(task 1 ①).
- 그런데 **G6 가 라이브에서 발화했는가**는 갈린다:
  - task 16 = **CONFIRMED** — 모델이 msg16 에 축자 서술 `"policy gate disambiguation check"`, 그 문자열의 유일 출처가 `a2/retail.gate.json:170` G6 message. task 54 msg28 에 독립 2차 사례.
  - task 9 = **PLAUSIBLE** — 토큰 회계 잔차(+1,203tok) ↔ G6 문면 2,631자(≈1,184tok).
  - task 1 = **UNPROVEN** — 그 턴 raw_data 가 `finish_reason='stop'` · `tool_calls=null`(애초에 호출이 없었을 수 있음).
  - task 3·4·12·24 = "미발화" 로 적었으나 그중 술어 오류 1건은 **CONFIRMED 로 정정**(task 1 ⑥).
- **관측 불가 구조가 원인**: 게이트 deny → 재생성 성공이면 커밋에 아무 표지도 안 남는다(`_commit_block_note` 는 K 소진 후 잔존 deny 에만 붙는다). 마커 0 은 음성이 아니라 **모른다**.

### G축 — 모델의 **false-success 날조**
- task 1 msg20 `"I have successfully processed the exchange"`(tool_calls 0) · task 9 msg21 · task 12 msg20(`finish_reason=length` · 8,192tok · `<tool_call>` 327회 텍스트 누출) · task 28 msg41 `"The new order has been created"`(retail 에 주문 생성 도구 0) · task 54 msg43 **자기 gold write 부인** · task 3 msg34.
- 이를 잡을 기구 `T2_CLAIMPROV` 는 retail 에 `claim_prov` 선언이 없거나 창(사임/transfer)이 안 맞아 **원리적 침묵**. `T2_FAB_STRIP` 은 `am.tool_calls` 를 보는데 날조는 산문으로 나갔다 ⇒ 대상 0.
- [[46]] 참고: 이 축은 라이브 A/B 에서 이미 null(1/16↔1/16)로 측정된 이력이 있다.

### H축 — **계기 결함**(귀속 아님 · [[25]])
- `t2_forensic.py:1069 def mutating_tools(domain="banking_knowledge")` — retail 런에 인자 없이 부르면 변이 도구 0개 ⇒ **실패한 sim 이 `clean=True` 로 보고**된다. 반증자가 task 4·16 에서 실행으로 재현(같은 sim 에서 표 두 개가 다르다). **4편이 독립 발견**.
- `sidecar='unknown'` 은 "안 막혔다"가 아니라 "모른다" — task 1·9·16 이 이 함정을 명시.
- 성공 경로 마커가 없는 계기는 `0` 이 '안 돌았다'가 아니다([[67]]).

### §2-Z 축별 **반증조건 (refutation)** — 무엇이 관측되면 이 축이 거짓인가
| 축 | 이 축이 거짓이 되는 관측 | 지금 실행 가능한가 |
|---|---|---|
| A | 회수 로그에서 문제의 write 턴에 `[G2_CONFIRM_WRITE]` deny 가 관측되거나, `enable_g2=False` 였음이 확인되면 거짓 | ⚠로그 0건 ⇒ **불가**. 단 실행된 write 26/26 통과 + 예측 28/28 일치가 간접 반증을 견딤 |
| B | 그 결정 시점 문맥에서 판별 재료(available 불리언·fulfillment item_ids·brightness·variants)가 **부재**함이 확인되면 거짓(= 재료 결손으로 재분류) | 가능 — 이미 6/6 실행했고 전부 실재 |
| C | 해당 키가 retail A2 3층 병합본(`load_domain_a2('retail')`)에 존재하면 거짓 | 가능 — 실행으로 부재 확인됨 |
| D | `T2_CALC=1`(또는 present 2종) 인 **비큐레이션** retail 런에서 주입이 실렸는데도 같은 오답이 나오면 인과는 거짓 | ⚠**아직 안 돌렸다**. 2026-07-11 t3 실측이 그 방향(주입 4/4 후에도 실패) |
| E | 모델이 `''` 를 낸 턴에서도 치환이 일어나면 '항상' 이 참, 안 일어나면 조건부 | 실행됨 — msg6/msg8 5건에서 치환 0 ⇒ **'항상'은 REFUTED**(조건부만 참) |
| F | 회수 로그의 그 sim·그 턴 게이트 id 가 G6 가 아니면 거짓 / G6 deny 후 재생성 성공이면 **원리적으로 흔적 0** | ⚠로그 0건 ⇒ **불가**. 이 축이 열린 채인 이유 |
| G | 같은 발화가 우리 층 ask 문면(`_BLOCK_NOTE_ASK`)에서 유도됐음이 로그의 `regen ok` 줄로 확인되면 귀속 이동 | ⚠로그 0건 ⇒ **불가**(task 28 ⑤가 UNPROVEN 인 이유) |
| H | retail 도구가 `banking_knowledge` 키에도 들어 있으면 거짓 | 실행됨 — 교집합 공집합 ⇒ 반증 실패 |

---

## §3 수리·레버 실측 성적표 — 발화 / 발화하고도 못 삼 / 발화 기회 없음 ([[55]] 死배선 ↔ 무효과 구분)

| 레버 | 이 런 상태 | 근거 | 분류 |
|---|---|---|---|
| `G2_CONFIRM_WRITE` | **발화 + 오통과** | 엔진 deny 6 sim·10턴 / 실행 write 26건 **전부** CONFIRM_RE 통과 | **발화했고 팔았다**(A축) |
| `G4_TRANSFER_MSG` | 발화 2회 | task 12·16·54 | 발화·무해 |
| `[DUPLICATE-READ]` | 발화 3건 | task 3#msg29 · 58#msg17 | 발화·무해(선언대로 K=3) |
| `T2_GROUND` | 발화 1건 | task 22#msg10 치환 | **발화했고 팔았다**(E축) |
| `G6_SELECT_CONFIRM` | **분쟁** | 마커 0 · task16 CONFIRMED · task9 PLAUSIBLE · task1 UNPROVEN | 판정 보류(F축) |
| `G1/G3/G5/G7/G_EXHAUST` | 마커 0 | 반증자 리플레이: 리졸버가 **100% 살아 있어도** G3 0(소유자 전부 일치) · G5 0(위반 write 가 관할 밖) · G_EXHAUST 0(transfer 2건 다 전수 read 뒤) | **발화 기회 자체가 없었다** — "리졸버 死" 가설은 **REFUTED**(task 12 주장2) |
| `T2_CALC` | 미발화 | 플래그 0건 · 선언(`calc_specs` 4행) 살아 있음 · 격리에서 `": 10"` 산출 | **재료 있음 · 플래그 없음**(死배선 아님) |
| `T2_PRESENT_READS` / `_NESTED` | 미발화 | 플래그 0건 · retail `present_specs` 2건 실재 | 동상 |
| `T2_CONSISTENCY`(L10 멤버십) | 미발화 | `go_stack.sh` 에 **한 번도 있던 적 없음**(`git log -S` 0 커밋). 선행 로스터가 이미 "414 로그 전수 0" 으로 등재 | 미등재 |
| `T2_RESOLVE` (value/operand) | **무장됐으나 반환 폐기 = 死배선** | `T2_RESOLVE=1` export 되고 `_contract_on` 참인데 `t2_resolve.py:1280` 의 `{"status":"resolved"}` **소비자 0건**; 리스트 operand 를 `[0]` 으로 접어 둘째 슬롯을 원리적으로 못 본다 | **死배선**(task 9·16·58) |
| `T2_RULE_AT_WRITE` | 미발화 · **이중 사망** | retail `write_rules` 부재 **+** 소비점 `:11710` 이 `T2_DECIDE_BEFORE_WRITE` 블록 안에 중첩인데 그 플래그가 어디서도 export 되지 않음(task 28 ② **REFUTED** 로 정정) | 선언 저작만으로는 **안 켜진다** |
| `T2_WRITE_ARG_GROUND` | 미발화 | `write_arg_grounding` 키 부재(+상위 가드 `_wev_live` 진입 불가) · 격리 B_WAG 는 오답을 정확히 거부 | 재료 0 |
| `T2_TOOLERR` | 미발화 · **이중 사망** | retail `tool_error_specs` 에 'invalid reason' · `return_delivered_order_items` 행 없음 **+** `T2_TOOLERR` export 0건(task 54 ⑤ **REFUTED** 로 정정) | 재료 0 + 플래그 0 |
| `T2_SG_*` / `T2_SEARCH_*` / `T2_REQUIRE_DOC_*` / `T2_ARG_PRODUCERS` / `T2_FOLLOWUP` / `T2_CLAIM_PROV` / `T2_DEMANDED_STEP` | 구조적 침묵 | retail A2 에 해당 키 0 · retail KB 문서 0 · `GO_RETRIEVAL` 빈 값 | 침묵이 정상(C축) |
| `T2_COMPUTE` | **유령 export** | `go_stack.sh:67` 이 export 하는데 `os.environ` 으로 읽는 코드 0건. 선행 `LEVER_ROSTER_CANONICAL_2026_08_19.md:65` 가 이미 *"존재하지 않는 이름"* 이라 적었는데 2026-08-29 런까지 그대로 | **死export** |
| `T2_L4` (variant substitute) | 미등재 | 2026-07-13 실측 *"치환 성적 2/2 오답(**t58 정답파괴** · t20 제약절단)"* ⇒ 기본 `mode='keep'` | 폐기 유지 |
| `T2_FAB_STRIP` | 대상 0 | 날조가 전부 산문(tool_calls 0) | 기회 없음 |
| `T2_PIN_READ` · `READ-FIRST` 외 8종 | **UNPROVEN** | stderr 전용 마커 · 로그 미회수 | 미발화/미회수를 **가를 수 없다** |

### 3-b 하드 제약(래칫)의 사각지대 — 이 런이 배터리 25/25 초록으로 지나간 이유
- `test_assembled_run.py:88` 의 검사 대상 `_drv` 가 **`reexp_assembled.sh` 로 하드코딩**되어 있어, 정본 런처 `go_stack.sh` 와 실 러너는 그 불변식(`T2_GATE_KINDS` 에 select_confirm 없음 / `T2_PRESENT_READS=1` / `T2_CALC=1`)을 **검사받지 않는다**.
- `run_t7391_retail.sh` 의 배터리 25종에 `test_assembled_run` **자체가 없다**(`grep -c` 0).
- 결정타: 배터리에 든 `test_flag_registry.py` 의 기준선 `flag_registry_baseline.json` 의 `undeclared` 목록에 **`T2_CALC`(:11) · `T2_PRESENT_NESTED`(:74) · `T2_PRESENT_READS`(:75)** 가 이미 박제돼 있다 ⇒ 누락 상태가 **승인된 기준선**이다.
- ⇒ [[07]] 동형: soft(주석·설계문서)로 못 막았고, hard 도 **엉뚱한 파일**에 걸려 있었다.

---

## §4 회귀 전용 절 — 무엇을 팔았나 ([[70]] 의무)

⛔**이 런에 대해 회귀 귀속은 판정 불가다.** 이유는 전부 근거로 확정됐다:

1. 유일한 비교 대상 `hist_gpt52_reg12_PASS` 는 **sha 상이**(`5ebebbe8` ↔ `fc0055dc`) · 8주치 레버 변경 미통제 · **PASS 만 추린 큐레이션(생존자 편향)** · nt=4 ↔ nt=1 · 포트 8360 ↔ 8141. task 1 E2 = **REFUTED**.
2. 프로젝트 자신이 워크트리 `t2_run_gated.py` 주석에 축자로 적어 두었다 — *"2026-08-29 retail 회귀를 [[70]] 의 같은-sha A/B 로 귀속하려다 **7월 런의 엔진 버전을 복원할 수 없어 막혔다**(로그는 로테이션으로 소실)"*.
3. 런 sha 가 벤치마크 sha 라 **엔진 버전이 기록되지 않았다**(§0).

| task | 대조군 대비 | 무엇을 팔았나 | 근거 / 반증조건 |
|---|---|---|---|
| 1 | 1.0 → 0.0 | **미상**. 후보: `T2_GATE_KINDS` 미설정으로 G6 활성 | 격리 100% 재현이나 라이브 발화 UNPROVEN. 반증: 로그의 그 턴 게이트 id 가 G6 가 아니면 거짓 |
| 3 · 4 | 1.0 → 0.0 | **미상**. 후보: `T2_CALC` 미수출 | 반증: T2_CALC ON 비큐레이션 런에서 `": 10"` 실렸는데 12 발화하면 model 로 이동 — **2026-07-11 t3 실측이 그 방향** |
| 9 · 16 · 58 · 60 | 1.0 → 0.0 | **미상**. 국면 발산이 우리 층 밖에서 관측됨 | 58: 손님 **첫 발화 타이밍**(대조군 msg1 에 기준 有 ↔ 치료 msg26 도착) · 54: user-sim `reasoning_effort:'low'` |
| 12 · 24 · 28 | 1.0 → 0.0 | **미상**. 공통점 = 모델이 **먼저 묻지 않았다**(대조군은 물었다) | ⇒ G2 결함은 **잠재였고 이 궤적이 노출**시켰다. 반증: 확인 턴을 강제해도 같은 write 가 나오면 거짓 |
| 22 | 1.0 → 0.0 | **미상**. 후보: `T2_GROUND` 치환(유일하게 우리 층이 값을 넣은 자리) | 과잉결정 가능 — `'Apt 1'` 도 gold `''` 와 다르다 |
| 54 | 1.0 → 0.0 | **미상** | — |

⇒ **레버가 무엇을 팔았는지 한 건도 확정하지 못했다.** 이것이 이 종합의 가장 큰 결손이며, 처방 큐 대부분이 §6(2) 격리-선행으로 밀린 이유다.

---

## §5 반증자 판정 반영 — 승격 / 보류 / 기각 (총 59건)

**집계: CONFIRMED 41 · UNPROVEN 7 · REFUTED 11**
(태스크별: 1=C4/U2/R2 · 3=C2/R3 · 4=C4 · 9=C2/U1 · 12=C2/R1 · 16=C7 · 22=C3/R1 · 24=C2/U1 · 28=C3/U1/R1 · 54=C5/R2 · 58=C2/U2/R1 · 60=C5)

### 5-a CONFIRMED (= 우리-층 결손으로 승격) — 41건, 결함 단위로 병합하면 **14 계열**

| # | 결함 계열 | 승격 근거 태스크 | 코드/선언 | 반증조건(refutation) |
|---|---|---|---|---|
| C1 | `T2_GATE_KINDS` 미설정 ⇒ retail 게이트 8종 전량 활성 | 1① | `t2_gate_patch.py:7777-7781` | 러너·A2 어딘가에서 kinds 가 설정돼 있었음이 확인되면 거짓 |
| C2 | 게이트 재생성이 tool_calls 를 떨어뜨렸을 때 **원 호출 보존 예비 부재** | 1④ | `:8468` `gate_rounds<1` ↔ DISAMB `:12744-12762` `keeping original` | 게이트 경로에 동형 보존 코드가 발견되면 거짓(grep 0건) |
| C3 | 도메인 비대칭이 결함을 뱅킹에서 은폐(retail 8게이트 ↔ banking 3) | 1⑤ | 선언 직독 | banking A2 에 select_confirm/exhaust kind 가 있으면 거짓 |
| C4 | 형제 보고서의 "G6 = 플래그 OFF" 술어 오류 정정 | 1⑥ | `T2_PRESENT_READS` 는 읽기-증강만, 차단 경로는 kinds 만 본다 | `gate_interpreter.py` 의 select_confirm 분기에 `T2_PRESENT` 참조가 있으면 거짓(grep 0건) |
| C5 | **`T2_CALC` 미배선** — `calc_specs` 선언은 살아 있고 격리에서 `": 10"` 산출 | 3주장1 · 4① | `:7348` / `a2/retail.specific.json:39-47` | 런 로그에 `[COMPUTED FACTS]` 1건이라도 있으면 거짓(궤적 0/12) |
| C6 | **`CONFIRM_RE` 오통과 기전**(A축) | 3주장2 · 9① · 12① · 24① · 28③ · 54①② · 60①②③ | `gate_interpreter.py:16-18,387-390` + `t2_gate_patch.py:6937-6942` | 그 write 턴에 G2 deny 가 로그에 있으면 거짓 / `enable_g2=False` 였으면 무효(실물 deny 6 sim 이 반증) |
| C7 | `T2_PRESENT_READS` 부재로 창이 5 레코드 밀림(부하 · 점수축 아님) | 4② · 12③ | `:7345`, `:7453-7460` | present ON 런에서도 `get_order_details≥5` 가 나오면 거짓(30/30 반증 실패) |
| C8 | **계기**: `mutating_tools()` 기본값 banking ⇒ 실패 sim 이 `clean=True` | 4③ · 16⑦ | `t2_forensic.py:1069` | retail 도구가 banking 키에도 있으면 거짓(교집합 공집합) |
| C9 | **불변식이 틀린 파일에 걸림** + 배터리 미포함 + 기준선 박제 | 4④ · 16⑥ | `test_assembled_run.py:88` · `flag_registry_baseline.json:11,74,75` | 배터리에 그 테스트가 있거나 `_drv` 가 런타임 드라이버면 거짓(둘 다 0) |
| C10 | `t2_resolve` value 분기 **死배선**(리스트 `[0]` 접힘 + `'resolved'` 소비자 0) | 9② · 58② | `t2_resolve.py:1272-1295` | `resolve_write` 밖에서 `decision` 을 소비하는 코드가 발견되면 거짓(grep 0건) |
| C11 | **멤버십 술어가 `return_delivered_order_items` 를 안 덮음** + `T2_CONSISTENCY` 미등재 + 통합이 술어를 좁힘 | 16①②③ | `t2_resolve.py:1288` / `t2_gate_patch.py:8605, 9498-9500` | 그 한 칸만 되살린 팔에서 sim16 이 여전히 `1994478369` 로 커밋되면 거짓 |
| C12 | `G6` 가 **틀린 필드**(order_id)를 묻고 늦게 묻는다(priority · sim당 1회 래치) | 16④ | `gate_interpreter.py:21-22, 233, 429-436` | 로그의 그 턴 게이트 id 가 G6 가 아니면 거짓 |
| C13 | `T2_GROUND` 제자리 치환이 손님의 **옛 값**을 되살림 + `G5.applies_to` 누락 + **차단 write 인자 재제출 기전 부재** | 22①③④ | `:8435-8445` / `retail.gate.json` / `_BLOCK_NOTE` 경로 | 치환 산출이 라이브 커밋 인자와 다르면 거짓(바이트 동일 확인) |
| C14 | retail A2 **재료 결손**(45키) — `write_arg_grounding` · `write_rules` · `write_arg_enum` · `tool_error_specs` 행 누락 | 28①④ · 54③④ · 16⑤ | 선언 직독 + `load_domain_a2('retail')` 실행 | 병합본에 그 키가 있으면 거짓 / 로그에 해당 `[T2_*]` 마커 1건이라도 있으면 거짓 |

### 5-b UNPROVEN (등급 그대로 유지 — **승격 금지 · 처방 근거로 쓰지 마라**) — 7건
| 항목 | 태스크 | 왜 못 닫았나 |
|---|---|---|
| G6 가 라이브에서 그 write 를 막았는가 | 1② · 9③ | 그 턴 raw_data 에 tool_calls 가 없거나(1), 결정 증거가 미회수 로그(9). **토큰 회계는 진단력 없음** — 도구호출 0인 턴에도 같은 크기 팽창이 반복 관측(sim4 msg4 +2,000tok) |
| 게이트 직렬화가 턴을 2배 태웠다 | 1③ | 전건(모델이 그 턴에 호출을 냈다)이 미관측. 지목 줄도 표 정의(정렬은 `:246`) |
| 리졸버 침묵 기전(`env.tools is None`) | 24② | 로컬 tau2 미설치(`import tau2` ModuleNotFoundError). 대조군이 리졸버 생존을 시사하나 sha 상이 |
| `_BLOCK_NOTE_ASK` 가 msg13 산문을 유도했는가 | 28⑤ | `appended`/`regen` 분기가 **저장 궤적에서 구별 불가**, 표지는 stderr 뿐 |
| `T2_CALC` 미수출의 **인과** | 58 | 두 플래그만 켠 팔 미실행 · task58 격리 프로브 0건 |
| 대조군을 통제로 쓸 수 있는가 | 58 | 최소 5축 동시 상이 |

### 5-c REFUTED (기각 — 이 판정들이 처방을 바꾼다) — 11건
| 기각된 주장 | 태스크 | 무엇이 참인가 |
|---|---|---|
| `info.git_commit` 이 엔진 sha | 1 E1 | **벤치마크 sha**. 엔진 버전 미기록 |
| `hist_gpt52_reg12_PASS` 가 대조군 | 1 E2 | 큐레이션 PASS 집합 · sha 상이 ⇒ 참조만 |
| CONFIRM_RE 결함은 **n=1** | 3 양화 | **5 태스크(3·9·12·24·60) / write 실행 7건** ⇒ 결함 **등급 상향** |
| 주입 구간이 `if dedup_on:` 안이라 무발화 | 3 반증⒝ | 들여쓰기상 **밖**이고 `T2_READ_DEDUP=1` 로 어차피 True. **코드 주석(:7350)이 낡았다** — 주석을 관측 대신 쓴 사례 |
| 지시문 데이터·산출 경로 일치 | 3 경로 | `t7391_reg12.results.json.gz` 하나뿐 · log 0건 |
| 리졸버 의존 게이트 4종 침묵이 **원인** | 12 주장2 | 리졸버가 100% 살아도 G3/G5/G_EXHAUST 는 0이 정상이고 G6 는 1라운드 재생성으로 마커를 남길 수 없다 ⇒ **판별력 0** |
| `T2_GROUND` 가 빈 칸을 **항상** 되살린다 / `GROUND_FEEDBACK` 이 같은 오값을 지시 | 22② | 모델이 `''` 를 내면 스캔이 건너뛴다(같은 sim 반례 5건). `GROUND_FEEDBACK` 유일 사용처는 **이 런에서 도달 불가한 `patched()`** — 격리가 죽은 템플릿을 찍었다([[78]] iso↔live) |
| `T2_RULE_AT_WRITE` 는 선언만 넣으면 켜진다 | 28② | 소비점이 `T2_DECIDE_BEFORE_WRITE` 블록 **안**이고 그 플래그는 어디서도 export 되지 않는다 ⇒ **처방 P1 무효** |
| `tool_error_specs` 행 누락이 힌트 부재의 원인 | 54⑤ | `T2_TOOLERR` 자체가 OFF ⇒ 행을 넣어도 이 런에선 무변화 |
| 회수 실패가 `run_t7391_retail.sh:96-101` 탓 | 54⑦ | 그 루프면 results 도 못 내려왔어야 한다 · 산출물이 untracked ⇒ **다른(미지의) 경로** |
| variant_spec 선언이 **유일** | 58 부속 | `retail.specific.json` 의 `variant_operand`+`variant_spec`(T2_L4 경로)이 두 번째로 존재하고 **인덱스 짝을 맞춘다** ⇒ t2_resolve 를 고치면 **중복·경합** |

---

## §6 처방 큐 ([[62]] 순서 — ①격리로 결손 측정 → ②전달 레버 → ③그 단계에만 결정론)

### (1) 무료 수리 가능 — 격리가 이미 있거나 계기·절차 결함 · 엔진 결정론 추가 0
| id | 처방 | 표적 | 기대 상한 | 근거 / 반증조건 |
|---|---|---|---|---|
| **F1** | `retail.settings.json` + `retail.gate.json` operands 에 `return_delivered_order_items.item_ids = {kind:'membership', …}` 한 칸 추가(양방향 동기 · [[24]]) | 16 | **+1 태스크(상한)** | 형제 두 도구가 이미 쓰는 술어라 엔진 수정 0. 오프라인 전수 스캔에서 **판정이 바뀌는 호출 1건뿐**(=이 실패), gold 인자 부정통제 통과 ⇒ **이 큐 유일의 순매수 후보**. 반증: 그 팔에서 sim16 이 여전히 오답 커밋이면 거짓 |
| **F2** | `t2_forensic.mutating_tools()` 도메인 자동 선택 | 계기 | 0(진단 정확도) | 4편 독립 발견 · 실패 sim 을 `clean=True` 로 보고([[25]] 위반 후보) |
| **F3** | `T2_COMPUTE` 死export 제거 → `T2_CALC` 로 통일 + `test_flag_registry` **역방향 래칫**(정본이 export 하는데 엔진이 안 읽는 이름 검출) | 계기 | 0 | 2026-08-19 에 이미 문서화된 유령 이름이 그대로 살아 있었다 |
| **F4** | `test_assembled_run.py` 검사 대상을 **런타임 드라이버**로 + 배터리 편입 + `flag_registry_baseline.json` 의 3플래그 `undeclared` 박제 해제 | 절차 | 0 | C9. 이 사각지대가 D축 누락을 25/25 초록으로 통과시켰다 |
| **F5** | 회수 규율: `reg12` 생산자 특정 → 로그·사이드카 회수 → `git ls-files --error-unmatch` 까지 절차화([[30]]) | 계기 | 0 | log/fb/trace 0건이 UNPROVEN 7건 중 4건을 만들었다. ⚠원인 귀속은 54⑦로 **미확정** |
| **F6** | 런 시작 시 **"플래그 ON · 재료 0" 레버 자기검사 인쇄** + `resolvers_from_env` 생존 표지 | 계기 | 0 | TASK_12 P3 · 24 P3 · 28 P5 · 54 P5 = **독립 발견 4건** ⇒ 정본 등재 자격 |
| **F7** | `G5_STATUS_PRECONDITION.applies_to` 에 `modify_pending_order_address` 추가 | 22 | **0**(이 sim DB 피해 0) | 완전성 목적. **우선순위로 쓰지 마라** |

### (2) 격리 프로브 선행 필요 — [[78]] 격리 100% + 프로브 id 를 주석에 박은 뒤에만 배선
| id | 처방 | 표적 | 선행 격리 요건 | 이미 잰 ± ([[70]]) |
|---|---|---|---|---|
| **Q1** | **G2 confirm 술어 강화**(대상 결속: 직전 assistant 가 **이 write 의 인자**를 발화 ∧ 그 뒤 user 발화가 매치) | 3·9·12·24·28·54·60 | ⛔`prevTxt` 판은 **무효 확정**(41/41 동일 · 0차단). 인자 결속판만 유효 | **판다**: 대조군 gold 1건(22#18) · 치료 gold 2건(3#24 · 54#37). 효과어 결속판은 치료 13건 · 대조 4건 차단(gold 3+3) ⇒ **부호표 없이 켜지 마라**. 선행 `REPLAY_SAFE_GATE_DESIGN_2026_07_06.md:355-362` 가 반대편 오차단을 이미 판정 게이트로 걸어 두었다 |
| **Q2** | `T2_CALC` **단독** A/B (`LEVER_ROSTER_CANONICAL_2026_08_19.md:318` 대기 실험) | 3·4 | 무주입 부정통제 팔 동반([[57]]) · `[COMPUTED FACTS]` 실물 계수 | 묶음 arm(+26)은 귀속 불가. **2026-07-11 t3 4/4 주입 후에도 실패(relay-gap)** 가 반대 방향 |
| **Q3** | `T2_PRESENT_NESTED` 단독 A/B | 9·58·60 | 마커 2종(`[DISAMBIGUATION NOTE` ↔ `[OPERAND DISAMBIGUATION`)을 **분리 계수**(합치면 오판) | NOTE 발화 30건 중 `get_order_details≥5` 0건(부하 감소 실측) / 그러나 점수에 준 **측정된 효과는 0**(comp_retail_t4 반례) |
| **Q4** | `t2_resolve` value 분기 **리스트 접힘 제거 + `decision` 소비 배선** | 9·16·58 | ⛔같은 자리에서 `T2_L4` 가 **2/2 오답(t58 정답 파괴)** 실측 · 두 번째 선언(`variant_operand`)과 **경합** ⇒ 통합 설계 후 격리 | 58 부속 REFUTED |
| **Q5** | 도메인 스왑 런의 `T2_GATE_KINDS` 를 정본 6종으로 명시 | 1 | G6 라이브 발화가 UNPROVEN ⇒ **먼저 F5(로그 회수)로 F축을 닫아라** | retail 114 부호표 필수 |
| **Q6** | 게이트 deny 후 **원 호출 보존 예비**(DISAMB 형) + 확인 직후 replay | 1·22 | 런 전수 센서스(차단 write 인자가 재생성에서 바뀌는 비율) 선행 | 22 에서 gold 인자 write 5건이 폐기됐다 |
| **Q7** | retail A2 에 `write_rules` · `write_arg_grounding` · `write_arg_enum`+`axis_prompt` 저작 | 28·54·24 | ⛔**`T2_DECIDE_BEFORE_WRITE` 를 함께 켜지 않으면 `write_rules` 는 발화하지 않는다**(28② REFUTED) | WAG 는 런 전수 취소 7건 중 4~6건 거부 · 그중 **gold 2건** + 대조군 gold 2건 ⇒ **순매수 아님** |
| **Q8** | retail `calc_specs` 에 도메인-일반 집계 op 추가(품목명별 개수 · cross-record 합) | 24·54 | ⛔"T-Shirt 를 세라" 로 좁히면 **[[23]] 위반** — 도메인-일반으로만 | 현행 `calc_specs` 는 단일 레코드 내부 연산뿐 |

### (3) 레버 없음(경계 = LLM 몫 · 새 결정론기 짓지 마라 [[62]])
| 현상 | 태스크 | 왜 경계인가 |
|---|---|---|
| 변형 오선택(brightness · cheapest-i7) | 9 · 58 | 선행 2건이 *"LLM operand-formalize, NOT a present-content gap"* · G6 present 는 *"already maximal"* 로 확정. 후보를 다 보여줘도 모델이 틀린다 |
| 손님-조건부 선호 **유도** | 60 | 시나리오 축자 *"**If and only if the agent provides several options**, you want the option without water resistance"* ⇒ 판별 술어가 손님에게서 유도되어야 하는 값. 게이트를 고쳐도 pass 보장 없음 |
| 주문 간 품목 접합 날조 | 24 | 정답 전문 1,605자가 문맥에 실재. 닫힌 술어 위의 모델 실패 |
| false-success 날조 | 1 · 9 · 12 · 28 · 54 | [[46]] 이 라이브 A/B null(1/16↔1/16) · 지연 1.8× 로 이미 측정 |
| cross-record 총액 오퍼랜드 완결성 | 54 | 처방 짓기 전에 [[62]]① 격리로 결손부터 재라(Q8 로 이관) |
| 정책 위반 다중 tool_call 동시발화 | 28 · 60 | 두 런 모두에 있어(21/169 ↔ 13/141) 런-특이 회귀 아님 ⇒ 계측만 |

---

## §7 이 종합이 못 사는 것 (정직 절)

1. **reward 를 하나도 예측하지 못한다.** CONFIRMED 41건 중 *"고치면 pass"* 를 증명한 것은 **0건**이다. 모든 반사실이 미측정이고, 형제 보고서 다수가 스스로 그 선을 그었다(*"막을 수 있었다"* 까지).
2. **[[70]] 의무 0/3.** 같은-sha A/B 가 없다. 회귀 귀속은 12/12 전부 **미상**이며, 7월 비교본은 sha · 큐레이션 · nt 가 모두 다르다.
3. **엔진 sha 미기록.** 인용한 모든 파일:줄은 워크트리 대조이고 [[77]] 의 `git show <런sha>:파일` 요건을 못 채운다. 게다가 **`reg12` 를 만든 런처가 repo 에 없어**(찾아본 곳: `grep -rln reg12 --include=*.sh` · `ls sim_results | grep bank_t7391`) 플래그 부재 근거의 일부는 *다른 런처에 대한 진술*이다 — 궤적 마커 0 이 그 자리를 대신 지탱한다.
4. **F축(G6)은 열린 채로 남았다.** 형제 3편이 상충하고, 게이트 deny→재생성 성공 경로가 **원리적으로 흔적을 남기지 않는다**. 로그 회수(F5) 전에는 못 닫는다. 같은 이유로 레버 8종이 UNPROVEN 이다.
5. **표본이 실패-편향이다.** 12 sim 은 retail 114 의 회귀 부분집합이고 전부 reward 0 이다 ⇒ 오차단(=파는 것) 계수의 **분모가 없다**. Q1/Q7 의 ± 는 전수 런 없이는 확정 못 한다.
6. **격리가 라이브가 아닌 자리 3곳.** ⒜`A_LIVE` 류는 게이트 한 칸의 재현이지 궤적 재현이 아니다 ⒝`GROUND_FEEDBACK` 격리는 **이 런에서 도달 불가한 `patched()` 템플릿**을 찍었다 ⒞G6 프로브는 `presented_select=True` · `resolvers` 미주입으로 자기 대상을 무력화한다. [[78]] 이 요구하는 **두 프롬프트 찍어 diff** 절차가 아직 없다.
7. **두 축 동시 0인 5 태스크(16·24·28·54·60)** 는 우리-층 수리만으로 reward 가 오르지 않는다. Q1 을 완벽히 고쳐도 NL 축(집계·유도·부인)이 남는다.
8. **코드 주석을 관측 대신 쓴 사례가 최소 2건**(`:7350` dedup 자백 주석 · `T2_RULE_AT_WRITE` 소비점 중첩 미확인)에서 보고서를 잘못된 UNPROVEN/CONFIRMED 로 몰았다. [[67]] *"이름을 믿지 마라"* 의 **주석판**을 규율로 올려야 한다.

---

### 다음 수 (순서 고정)
`F5 로그·러너 회수` → `F2·F3·F4·F6 계기·래칫` → `F1 멤버십 한 칸(유일 순매수 후보)` → `Q5 로 F축 닫기` → `Q1 부호표(retail 114 전수)` → 나머지 Q.
⛔Q 는 격리 100% + 프로브 id 주석 박제 전에 배선하지 않는다([[76]] · [[78]]).
