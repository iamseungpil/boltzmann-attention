# OPERAND 레버 최종 판정표 (2026-08-19)

> 대상 = tau2 `banking_knowledge` 스택의 **operand(인자 값/종류) 관련 레버 43건**.
> 배경 실측 = t7326(20태스크·40 sim) 실패 18태스크 = **WRONGARG 7 · MISSING 7 · EXTRA/대체 2 · ACTION 채점 2**.
> 채점 = sim 의 `reward_basis`(DB 35 · ACTION 4 · 없음 1). **DB 채점이면 성공한 변이(mutating) 호출 집합만 점수를 만든다. read 는 무관.**
> 레버 발화 판정 = ①stderr 태그 ②궤적/사이드카의 효과 문자열. **둘 다 0이어야 dark.**
>
> ⛔ 이 문서는 **끄기 권고를 담지 않는다**([[60]] 레버는 전부 항상 켠다). 각 레버가 **무엇을 사고 무엇을 파는지**, 그리고 **무엇을 재야 판정이 갈리는지**만 적는다.
> 규율: 축자 근거 + 파일:줄. "효과 있다"는 **측정된 대조**가 있을 때만. 못 찾은 것은 **미검증**으로 남긴다([[08]]·[[57]]·[[62]]).

---

## 1. 요약표

| 판정 | 개수 | 플래그 |
|---|---:|---|
| **VALID_MEASURED** | **0** | — (pass/reward 축에서 **이득 방향으로 측정된 레버는 한 건도 없다**) |
| **PLAUSIBLE_UNMEASURED** | **9** | `T2_WRITE_ARG_ENUM` · `T2_WRITE_ARG_GROUND` · `T2_WRITE_SUB` · `T2_GROUND_HDR` · `T2_QUOTE_PIN` · `T2_PROD_BIND` · `T2_HAVE_VALUE_FORCE` · `T2_REF_VERIFY` · `T2_DECIDE_BEFORE_WRITE` |
| **DARK** | **11** | `T2_ARG_REPEAT` · `T2_WRITE_DEDUP` · `T2_SCALAR_ARRAY` · `T2_FIT_DIFF` · `T2_GROUNDING_SPEC` · `T2_GROUND_DROP_NAVKEYS` · `T2_PROV_GROUND` · `T2_QUOTE_HINT` · `T2_CHOICE_GROUND` · `T2_HAVE_VALUE` · `T2_DOCS_AT_WRITE` |
| **HARMFUL** | **12** | `T2_ARG_AXIS` · `T2_ARG_PRODUCERS` · `T2_WRITE_EVIDENCE` · `T2_RESOLVE` · `T2_RESOLVE_CAP` · `T2_COMPUTE` · `T2_MATCH_COUNT` · `T2_GROUND` · `T2_SG_GROUND` · `T2_OPERATOR_PINPOINT` · `T2_VALUE_ACQUIRE` · `T2_REF_ISO` |
| **NOT_OPERAND** | **11** | `T2_ARG_SCHEMA` · `T2_ARGSCHEMA` · `T2_ARG_EMPTY` · `T2_WRITE_PROV` · `T2_WRITEPROV` · `T2_PAIRCHECK` · `T2_PAIRFIX` · `T2_GROUND_LOG` · `T2_GIVE_QUOTE` · `T2_BRANCH_REGROUND` · `T2_SG_ARGS` |
| **UNKNOWN** | **0** | — |
| **합계** | **43** | |

### 1-1. 분모 정정 — 43 중 4건은 레버가 아니다

| 이름 | 실체 | 확인 |
|---|---|---|
| `T2_ARGSCHEMA` | `T2_ARG_SCHEMA` 의 stderr **태그 문자열** | `grep -rn 'environ.get("T2_ARGSCHEMA")' *.py` = **0건**; x44_lever_coverage.py:83 수동 등재가 이중계수를 만들었다 |
| `T2_WRITEPROV` | `T2_WRITE_PROV` 의 **태그 문자열** | `grep -rn 'os.environ.get("T2_WRITEPROV"' --include=*.py` = **0건**; 정본 매핑 x44_lever_coverage.py:79 `"T2_WRITE_PROV": {"[T2_WRITEPROV]"},` |
| `T2_SG_ARGS` | t2_scaffold_get.py:1689 의 **인쇄 문자열**(실제 게이트 = `T2_SCAFFOLD_GET`) | `environ.get("T2_SG_ARGS")` **0건** |
| `T2_GROUND_LOG` | **계기**(JSONL append 전용·거동 변화 0) | t2_resolve_patch.py:274-276 축자 *"§7 계측 … 미설정이면 무동작"* |

⇒ 실효 레버 분모 = **39**. (`T2_WRITE_DEDUP` 은 플래그 이름은 존재하나 **구현이 없다** — 아래 §3 참조. 이를 빼면 38.)
이 이중계수가 곧 "레버 N종" 통계와 dark 비율을 부풀린 원인이다(C434 *"인쇄이지 레버가 아니다"* 계열).

---

## 2. VALID_MEASURED — **0건**

> **가장 중요한 결과: operand 축 39개 실효 레버 중, pass/reward 를 이득 방향으로 움직였다는 것이 부정통제를 갖춘 대조로 측정된 레버는 0건이다.**

측정된 대조를 **갖고 있기는 한** 레버는 16건이고, 그 부호는 **음성·null·무효(gold 대조)·비-pass 종점** 중 하나다. 전수는 아래와 같다.

| 레버 | 대조 설계 | 수치 | 부호 | 출처(축자 위치) | 등급/한계 |
|---|---|---|---|---|---|
| `T2_RESOLVE` | 라이브 A/B (표준 gpt-5.2 user-sim·apply-heavy 5태스크·nt1) | **G(레버 無) 3/5 ↔ GR(레버 有) 0/5** | **음성** | BANK_ACTIONREQ_PROBE_FORENSIC_2026_07_13.md:80-99 §6d — 축자 *"레버가 표준서 순수 손해(0/5 vs 3/5·Δspurious 강한 음성)"* | [M]·n=5 |
| `T2_OPERATOR_PINPOINT` | 격리 5팔 n=24(8×3·사전 고정 판정선) | **A_REF 24/24 ↔ B_PINPOINT 0/24**, 부정통제 **E_NEG 0/24** | **음성**(파괴적) | t2_resolve.py:100-106 · x322_operator_scope_iso.py · RESEARCH_MASTER.md:424 C489 | [M]·**원 출력 파일 부재**(스크립트만 존재) |
| `T2_ARG_AXIS` | 격리 x275 5팔 n=8 + 부정통제 N1_PERSONAL 0/8 | 격리 **8/8** | 격리 양성 ↔ **라이브 반증** | 격리=[M]; 라이브 = gold 축 거부 9건(bank_t7299_{ctl,treat}_20260816b 등) | 전이 실패·라이브 A/B 0 |
| `T2_WRITE_SUB` | 격리 사슬 x307~x310 (부정통제 D_NOBASIS 0/8) | x307 **0/8 ↔ 7/8** · x308 7~8/8 · x309 8/8 · x310 오제안 순응 0/8 | **결손 측정 양성** | RESEARCH_MASTER.md:409-410 C472 | 차 7~8 ≥ C483 잡음 바닥 5 ⇒ 인용 가능. **단 [[62]]① 결손 측정이지 레버 효과 아님** |
| `T2_MATCH_COUNT` | 시드-맞춤 짝비교 **64/64**(arm A `bank_ax33n_gpu{0,1}_20260803g` ↔ B4 `bank_b4_gpu{0,1}_20260803h`) | **24/64 ↔ 24/64**, flip 16(8↑8↓) | **null** | B4_CAUSE_LEVEL_FORENSIC_2026_08_04.md:20-25 — 축자 *"pass 동률 24 vs 24는 무변화가 아니라 8↑8↓ 상쇄"* | ①+④ 묶음·단독 귀속 아님 |
| `T2_DOCS_AT_WRITE` | **단일-변수** A/B (run_t7304_20260816j.sh:2-20 축자 *"처치 = `T2_DOCS_AT_WRITE=1` 하나"*) | 055 **0/8 ↔ 0/8** · aux **3/8 ↔ 4/8**(사전 GO 바 +5) | **null**(사전등록 프레임 미달) | 원본 results 집계 + RESEARCH_MASTER.md:475 C505 | 기전도 반증(C505⒟ 격리 8/8 오답) |
| `T2_HAVE_VALUE` | 라이브 ON/OFF (`bank_hve2e9_{base,hv}_20260723`·같은 8태스크·nt1) | base **1/8** → hv **0/8**, 레버 발화 **0회** | **null/무효** | 원본 results 집계 · C117 축자 *"031 base1/hv0=user-sim 변동"* | 발화 0이라 Δ는 레버 효과 아님 |
| `T2_WRITE_DEDUP` | **사전 고정 게이트** (x294_dup_write_probe.py:14) | 요구 A_ASIS ≥6/8 ↔ 실측 **0/8** | **부결** | X291_CHECKING_FIT_DESIGN_2026_08_13.md:179 축자 *"A_ASIS 0/8 — 재현 실패·T2_WRITE_DEDUP 보류"* | 격리에서 결손 재현 실패 ⇒ deny 면허 없음 |
| `T2_BRANCH_REGROUND` | ①오프라인 3-arm n=6 ②라이브 matched pair(`bank_reg043fix_{base,treat}`) | ① R_none 0/6·0/6 ↔ R_reground **5/6·6/6** ② treat **close x0** ↔ base **close x1** | **행동 축 양성** | RESEARCH_MASTER.md:622 C146 · :628 C149 | ★**pass 는 못 샀다** — C147 축자 *"task 실패(정직): db_match=False 둘다"* |
| `T2_QUOTE_PIN` | 동일-조건 ON/OFF 2 arm (`x30 --user af0581dcbf --tag 019`) | **ON과 OFF 완전 동일**(discrepant 4건 동일·coverage 23/23·드롭 0) | **null** | RESEARCH_MASTER.md:777 C286 | 승격 근거 C282 는 n=1 before/after·**원 근거 파일 디스크 부재** |
| `T2_RESOLVE_CAP` | cap=3 ↔ 무제한 | 발화 6→**100**, 요건 집합 **100회 전부 동일**(전진 0) | 반대 가설 기각(종점이 pass 아님) | t2_gate_patch.py:3784-3791 | **원 산출 로그 미확인** |
| `T2_RESOLVE_CAP` | C540 시드-맞춤 자연실험(x382/x383) | 098 14/15→**0/4** · 100 →**0/4** · 073 4/5→**0/1** | **음성** | REGRESSION_2026_08_18_CAP_LATCH.md:29-36 · RESEARCH_MASTER.md:457 | 회귀 사실 [S] / 귀속 [M]~[D](C538 자인 *"과장 금지"*) |
| `T2_COMPUTE` | Δspurious 표 + 오프라인 replay 240 sim | 오치환 **27/431(6.3%)** ↔ 교정 375/414 = 순 +348; replay 교정→gold 일치 90.9% | **[[23]] 무효** | BANK_COMPUTE_OP_KEYSTONE_DESIGN_2026_07_13.md:205,:233-236 | **전 수치의 종점이 gold**·임계 T1 도 gold 재현율로 선택(A2 `_note_compute_ops_removed_2026_08_19` 자백) |
| `T2_REF_ISO` | 라이브 per-switch 포렌식 n=34(rall21/22) | switched 4 · memo-switch 6 · keep 9 · unsure 15; **gold→wrong 1건** | **음성** | `bank_rall22a_20260724.log.gz` 축자 `switched param=transaction_id txn_adea68821a1d->txn_9a72b84326d1` + 그 sim reward 0.0 | 성적 대조는 전부 0/2 |
| `T2_REF_VERIFY` | 오프라인 결정론 replay(`efiso_detmatch_proof.py`·본 감사에서 재실행) | 슬립 **8/8 검출** · gold **25/25 통과**(false-block 0) | 오프라인 양성 | C128 재현 | ★**현행 6인자 판본을 검정하지 못한다** — `test_ref_verify.py:67` 3인자 호출 → `TypeError` |
| `T2_GIVE_QUOTE` | 사전등록 지표(철회율) — 본 감사 재계산 | retract=1 **188** / retract=0 179 = **51.2%**(367) | **효과 아님** | t2_gate_patch.py:11220 종료조건 | 처치 arm 내부 관측·(런,태스크) 82쌍 reward 조인 0.211(n=19) vs 0.158(n=19) = 잡음 |

**해석 (등대 §1 모트 관점).**
- "레버는 하나를 사면 하나를 판다" 의 **판 쪽만 측정돼 있다**. 산 쪽(pass 이동)을 부정통제와 함께 잰 사례는 operand 축에 0건이다.
- 원장이 인정하는 라이브 pass 이동 실측 4종(LEVER_CONSOLIDATION_2026_08_19.md:760 축자 *"`PROV_REGEN`(C53·456 sim) · `QUOTE_PIN`(C282·n=1) · `READ_DEDUP`(C114·n=1) · `WRITE_SUB`(C475)"*) 중 이 배치에 속한 둘(`QUOTE_PIN`·`WRITE_SUB`)은 본 감사에서 각각 **null(C286)** 과 **재현 실패(t7313 1.0↔0.0 · t7326 0.0/0.0)** 로 뒤집혔다.
- ⇒ **operand 축에서 "이 레버가 pass 를 산다"고 말할 수 있는 것은 현재 하나도 없다.**

---

## 3. DARK — 두 신호 모두 0 (T2_FN_ISOLATE 형 사고 후보)

| 레버 | 신호 | dark 의 성질 | 원인(축자) |
|---|---|---|---|
| `T2_ARG_REPEAT` | 태그 0 / 효과문자열 0 (1139 전수) | **구조적 死코드** | `_rejected_params`(t2_gate_patch.py:4054-4064) 가 `not getattr(m,"error",True)` 로 skip 하는데 대상 tool 메시지 **285건 전부 `error=False`** ⇒ 원리상 발화 불가. 전제조건은 풍부(`Unexpected parameter` 553회/41런), 형제 `[T2_UNKNOWN_REPEAT]` 는 177회 발화 |
| `T2_WRITE_DEDUP` | 태그 0 / 효과문자열 0 | **구현 부재 + 옳은 보류** | `environ.get("T2_WRITE_DEDUP")` **0건**·registry 미등재. 사전 고정 게이트 미달(A_ASIS 0/8). 양성통제 `[DUPLICATE-READ]` 2,301회로 스캐너 무결 |
| `T2_SCALAR_ARRAY` | 자기 태그 없음 / 효과문자열 0/352 | **부착 0** | ON 이었던 유일 런(ax32)의 1,047 호출에 정본 술어 통과 → 발화했을 자리 **0건**; t7326 908 호출도 0 |
| `T2_FIT_DIFF` | 자기 태그 없음 / 효과문자열 0/352 | **표적 있는데 부착 0** | ON 런에서 표적 15건(전부 `check_card_application_fit`)인데 `[T2_AXIS]` 6회는 전부 `call_/unlock_discoverable_agent_tool` = fit 도구 부착 0 |
| `T2_GROUNDING_SPEC` | 태그 0/414 · 궤적 0/352 | **3중 도달 불가** | `--resolve` 가 go_stack.sh 에 0건 ⇒ `t2_resolve_patch.apply()` 미호출; env 도 안 읽힘(t2_run_gated.py:175-177 항상 명시 경로); `banking_knowledge.grounding.json` **파일 부재** |
| `T2_GROUND_DROP_NAVKEYS` | 태그 없음 / 효과문자열 없음 | **상위 모듈 死** | 런처 grep repo 전체 **0건**; read 지점이 `resolve_selection` 경로 안(위 항목과 동일하게 死). Y2_DESIGN_2026_07_31.md:105 축자 *"완전 미문서 4개"* 에 등재 |
| `T2_PROV_GROUND` | 태그 0 · 궤적 0 | **켜면 프로세스 사망** | t2_run_gated.py:222-223 축자 `raise SystemExit("[t2_run] T2_PROV_GROUND is not supported in unified mode (E-COMP scope). Use T2_GROUND=1.")` — go_stack 이 unified 이므로 항상 차단 |
| `T2_QUOTE_HINT` | beat 0/414 · 궤적 0/352 | ★**표적 부재가 아님(재현으로 반증)** | 생성 이후 런 `bank_smoke8b_pin_20260805` task_061 t0 에서 가드가 `outstanding_balance='0'` 드롭 → 그 sim 실제 출력으로 `t2_quote_hint.hint('0', outs)` 돌리면 **199자 힌트 + beat 발화**. 원인은 상류(원격 플래그 적용·모듈 배포·A2 병합)이고 로컬로는 미확정 |
| `T2_CHOICE_GROUND` | 태그 0/414 · 효과문자열 0/352 | **중첩 인자 미처리** | `_args_dict`(t2_gate_patch.py:387-397)가 바깥 dict 만 반환 ⇒ `_ar_cg.get(...)` 항상 None ⇒ `continue`(:11045). `_parse_nested_args`(t2_resolve.py:775)가 이미 있는데 이 자리에서 안 쓴다 |
| `T2_HAVE_VALUE` | 태그 0 · 효과 0 · **would-fire 억제 인쇄도 0** | ★**(a)표적 미발생 확정** | 관측 전용 `elif`(t2_gate_patch.py:7138-7155)가 배타 체인에 밀린 턴도 재평가해 `would-fire but suppressed by=` 를 찍는데 그것도 0 ⇒ 술어가 참이 된 적 없음. 형제 VALUE_ACQUIRE 1,272회 발화로 배선 생존 확인. selftest 7/7 PASS |
| `T2_DOCS_AT_WRITE` | 자체 태그·효과문자열 **둘 다 코드에 없음** | **본체 양팔 0회** | RESEARCH_MASTER.md:475 C505⒝ 축자 *"처치의 본체(`T2_DOCS_AT_WRITE` → write-hold)는 양팔 0회 발화했다 … 자리는 옛 자리 그대로였다"* |

**T2_FN_ISOLATE 형 사고의 재발 조건 3종(이 11건에서 반복 확인).**
1. **가드 술어가 라이브 스키마와 어긋난다** — `error=False` 전건 skip(ARG_REPEAT), 중첩 `arguments` 미파싱(CHOICE_GROUND).
2. **상위 배선이 죽어 하위 플래그가 도달 불가** — `--resolve` 미전달(GROUNDING_SPEC → DROP_NAVKEYS), unified 모드 SystemExit(PROV_GROUND).
3. **자기 태그/효과 문자열이 아예 없어 관측 창이 0** — DOCS_AT_WRITE 가 여기서 **오판을 한 번 만들었다**(C504 '배선 통과' → C505 코드 직독으로 철회).

⇒ 재야 할 것: **모든 레버에 (a)자기 태그 (b)would-fire-but-suppressed 관측 분기** 두 개. `T2_HAVE_VALUE` 만이 (b)를 갖고 있어서 "표적 부재"와 "배타 체인 억제"를 갈랐다. 나머지 10건은 그 구분을 못 한다.

---

## 4. HARMFUL — 정답을 지우거나 잘못 거부한 **실측** 사례

| 레버 | 실측 피해 | 축자 근거 |
|---|---|---|
| `T2_ARG_AXIS` | **gold 축을 거부 9건**(task_055 15건 중 9) + want 집합 **자기모순 11/26(42%)** + 같은 task+seed 에 매번 다른 답 | `bank_t7299_ctl_20260816b.log.gz` 축자 `[T2_ARG_AXIS] deny got=checking want=['business_checking','savings']`; 055 gold = `Purple/checking` + `Silver Plus/savings` |
| `T2_ARG_PRODUCERS` | 넛지 705회 중 지목 도구가 **전부 검색/read**(`KB_search_bm25` 483 · `KB_search_dense` 110 · `KB_search` 82 · `unlock_…` 21 · `shell` 9) = **gold 변이 도구 0건**; 표적(040/041) 적중 **1/705** | 술어가 도구 결과 산문 substring(`if tool and arg in c`) — [[59]] 위반. DAY7_PRESCRIPTIONS_DESIGN_2026_07_28.md:109-112 가 사전등록한 위험의 실현 |
| `T2_WRITE_EVIDENCE` | t7326 deny 25 / **gold 이름 표적 22 / 미회복 5**(전부 reward 0.0). 런 전체 gold 오차단 56 중 **22 = 39% 단일 최대** | TWO_KERNEL_DESIGN_2026_08_19.md:342 축자 *"094 t1 turn71 의 gold write 를 차단·미회복시킨 것 [S]"*; task_040 t0 turn 39/43/45/47 동일 이름 4연타 후 영영 미실행 |
| `T2_RESOLVE` | 라이브 A/B **3/5 → 0/5**; gold 오차단 24(operator-fab 12 + operator-scope 12); 깨끗한 흐름 탈선(t003·t023) | BANK_ACTIONREQ_PROBE_FORENSIC §6d · LEVER_CONSOLIDATION_2026_08_19.md:74 |
| `T2_RESOLVE_CAP` | 캡 정지가 계약 경로를 3~7배 죽임(098 2→6·100 0→6·073 12→25 / T2_RESOLVE 8→3·12→4·21→3) → 098 14/15→0/4 | REGRESSION_2026_08_18_CAP_LATCH.md:29-36 축자 *"쓰기 전에 조회하라는 요구가 사라지자 모델이 추측으로 인자를 채운다"* |
| `T2_COMPUTE` | **에이전트가 이미 맞춘 값을 엔진이 틀리게 덮어쓴 건수 27/431(6.3%)**; `provisional_credit_eligible` 은 순손해 −4 | BANK_COMPUTE_OP_KEYSTONE_DESIGN_2026_07_13.md:205. ★더 근본적으로 **산출물이 채점되는 gold 인자 자체** = [[62]] 측정 대상 소멸 |
| `T2_MATCH_COUNT` | **우리 도구가 모델에게 거짓 완결 인증** — t7326 '전부 표시' 102건 중 **19건이 반증 가능하게 거짓**(주장 31·23·23·20·20·15·14 … vs 실제 반환 10) | `shown_in()`(t2_match_count.py:88-91)의 `^\s*(\d+)\.\s` 가 회수 문서 **본문 안 번호 목록**까지 셈 — task_055 실제 10 ↔ 보고 34. [[25]] 정면 |
| `T2_GROUND` | **치환 385건 중 301건(78%)이 `arg=agent_tool_name` → 고객 이름**; 궤적 생존 110건/11 sim, **11/11 전부 reward 0.0** | `bank_all97_nt1_v2_20260718.log.gz` 축자 `substituted arg=agent_tool_name val=cancel_debit_card_7823 -> LIANG JINHAI`; task_077 `Unknown agent tool 'LIANG JINHAI'` **79회 반복** |
| `T2_SG_GROUND` | **참인 값을 드롭** — task_046/t0 `outstanding_balance=0.00` 드롭 → 11턴 루프 → 잔여 gold 소실 | N97_TASKWISE_FORENSIC_2026_08_04.md:115. ★단 t7326 드롭은 내가 검산해 **거짓 드롭 없음** 확인(레코드 실재 0) |
| `T2_OPERATOR_PINPOINT` | 격리 **24/24 → 0/24**(부정통제 E_NEG 0/24 = 가짜 도구명에도 순응); 라이브 C485 = **성공한 write 를 부정 → 같은 계좌에 $9.50 두 번** → DB 불일치 reward 0 | RESEARCH_MASTER.md:420 축자 *"못 해서 실패한 게 아니라 한 번 더 해서 실패했다"* |
| `T2_VALUE_ACQUIRE` | 전달된 넛지 **80/80 이 표적 밖 태스크**(031/039/053 = 0건); 그 중 20건에서 `get_card_last_4_digits` 접촉, **20/20 reward 0.0**, 첫 접촉이 전부 첫 넛지 **이후** | gate.json:2386 문구가 사실 아닌 전제를 단정(*"they are NOT stored in the account records … file the dispute with that value"*). 발화 조건이 산문 문자열뿐(t2_gate_patch.py:1644-1650 축자 *"조건을 걸지 않는다"*) |
| `T2_REF_ISO` | **gold 인자값을 오답으로 치환 + 3회 메모이즈** — task_031 `txn_adea68821a1d`(gold) → `txn_9a72b84326d1`(손님 미언급), action_match False·reward 0.0 | `bank_rall22a_20260724.log.gz` 축자 `[T2_REF_ISO] switched param=transaction_id …` · RESEARCH_MASTER.md:588 C129 *"LLM 재선택 shipping 불가(해로움)"* |

**공통 기전 3종.**
- **(H1) 지목형 개입** — "정답은 X" 를 말하는 순간 모델은 그 이름에 순응한다(E_NEG 0/24). `OPERATOR_PINPOINT` · `RESOLVE` · `GROUND` · `REF_ISO` · `WRITE_ARG_ENUM` 동봉 채널이 같은 부류다. [[62]] 가 생긴 자리.
- **(H2) fail-closed deny 가 실패 모드를 이동시킨다** — *빈 값 기록* → *기록 없음*(C425), gold write 차단→미회복(WEV 5건). DB 채점은 **성공한 변이만** 점수를 만들므로 이 이동은 곧 pass 소멸이다.
- **(H3) 술어가 도메인 산문을 substring/정규식으로 뜯는다** — `ARG_PRODUCERS`(705/705 오지목) · `MATCH_COUNT`(카운터 고장) · `FIT_DIFF`(`'card'` 리터럴). [[59]]·[[05]] 위반이자 전이 불안정의 원인.

---

## 5. PLAUSIBLE_UNMEASURED — 주장은 있는데 수치 원본이 없다 (9건)

각 행의 마지막 칸이 **"무엇을 재면 유효/무효가 갈리는가"** 다.

| 레버 | 살아 있는 주장 | 왜 미측정인가 | **재야 할 단 하나** |
|---|---|---|---|
| `T2_WRITE_ARG_ENUM` | over-block 0 — 거부 198건 전수 대조에서 **후보 명단 안 이름을 막은 적 0/198** | 유일 ON/OFF A/B(run_lever_20260812b)가 **레버 死 상태로 돌았다**(run_lever_20260812c.sh:6-9 축자 *"직전 …b 는 죽은 코드로 돌았다"*); 이후 전부 ON 팔만 | **동봉 채널의 정확도** — 거부문에 실리는 `It answers: X` 가 gold 와 일치한 비율(055 에서 `Gold Account` 32회 ↔ gold `Silver Plus`; 071 에서는 정답 동봉 = 동전 던지기). 동봉 제거 arm 과의 pass 대조 |
| `T2_WRITE_ARG_GROUND` | 미접지 인자 차단·발화 315행 | `T2_WRITE_ARG_GROUND=0` **repo 전체 grep 0건** = OFF 팔이 역사상 없음. 유일 전후 수치(rall10→rall11 031 0.0 2/5→1.0 3/5)는 **C217 이 [D] 로 강등** | **ON/OFF 짝비교에서 gold 이름 deny 수와 미회복 수** — t7326 deny 2건이 **둘 다 gold 이름 표적**(비-gold deny 0)이라 현행 순기여가 오차단 쪽으로만 보인다 |
| `T2_WRITE_SUB` | 격리 사슬 x307~x310 (부정통제 포함·차 7~8) | 라이브 귀속이 **재현 실패** — t7313 ctl 073 **1.0** ↔ treat **0.0**(양팔 `T2_WRITE_SUB=3` 동일) · t7326 073 **[0.0, 0.0]** | **같은 시드·같은 스택에서 `=0` vs `=3` 짝비교 n≥8**, 종점 073/075 pass. C483 잡음 바닥(차 ≥5) 초과 여부가 판정선 |
| `T2_GROUND_HDR` | 지시문 분리로 intent_fields 모순 제거 | ON 런이 전부 다른 레버와 동시 전환(run_axis32_chain.sh:49-56에서 10종 동시 ON). ★dossier 의 "t7326 이 첫 분리" 는 **반증**(ax32p1/p2·axmicro 가 2026-08-02 이미 분리) | **설계서가 스스로 사전등록한 `38/58` 의 사후 대조** — INSTRUCTION_DEFECT_REDESIGN_2026_08_01.md:273 축자 *"(현 회복 38/58) ⇒ scalar 회복률을 사전/사후 대조한다"*. repo 전체에 사후 수치 0건 |
| `T2_QUOTE_PIN` | 표적 실재·15회 발화 | **고유 기전이 판정을 바꾼 발화가 0건** — 15/15 가 핀 빈 문자열(1층 `quote_unverbatim`). C289 축자 *"표가 잡은 게 아니다 … 표의 순이득은 0"*. C282 원 근거 파일(`bank_qpsmoke_20260801` 등) **디스크 부재** | **pin_kind 라우팅·`policy_group_rows` 멤버십이 판정을 바꾼 발화 수**(현재 0) 대 **rate 드롭으로 잃은 행 수**(C289: 019 t0 reward 0·db False) |
| `T2_PROD_BIND` | 날조 행 강등 44회·규모 73/72/47행 | 검증이 **오프라인 한정**(DAY5_PRESCRIPTIONS_DESIGN_2026_07_28.md:146-147). 하류 문자열이 정직한 생략과 공유돼 궤적 귀속 불가 | **전-행 오강등률** — t2_scaffold_get.py:1807-1816 에서 `_cands=[]` 면 빈 `any()` 가 False ⇒ **모든 행 강등**. 강등된 행 중 실제로 producer 출력에 존재했던 행의 비율 |
| `T2_HAVE_VALUE_FORCE` | 기본 ON·라이브 활발(hv_fb 재사용으로 VALUE_ACQUIRE 1,272회 뒤마다 점화) | **전용 계기가 0개** — 언제 걸렸는지 원리상 셀 수 없다. run_rall19 는 '단일변수'라 적고 5종을 묶어 켠다 | **force 턴 수 + 그 턴에 죽은 `_gen_action_sub` 수** — t2_gate_patch.py:9468-9469 가 `not force_required` 를 요구하므로 이 노브는 **그 턴의 ACTION_SUB 를 판다**. 인쇄부터 심는 것이 선결 |
| `T2_REF_VERIFY` | 오프라인 결정론 8/8 검출·25/25 통과(본 감사 재현) | ★**인증 회귀 2종이 깨져 있다** — `test_ref_verify.py:67`·`test_ref_verify_replay.py:85` 3인자 호출 → `TypeError`(현행 `_ref_verify_deny(agent, la, UserMessage, messages, tc, specs)`·:1281). 라이브에서 이 플래그가 변수였던 런 0 | **현행 판본에 대한 replay 재인증** + **서브 fail-open skip 비율**(t2_gate_patch.py:1320-1330 이 `t2_search.sub_records` 로 LLM 서브콜을 돌리는데 실패 시 `rec_val=None → skip` 침묵). 그리고 deny 받은 sim 9건 중 8건 0.0·최대 5회 반복 deny 의 **교정 성공률** |
| `T2_DECIDE_BEFORE_WRITE` | 교정 후 20회 발화·사이드카 배달 12레코드/6 sim 확인 | 유일 사전등록 A/B(C439)가 **정반대 조건 가드**로 돌아 0회 발화 = 무효(RESEARCH_MASTER.md:374 축자 *"ON/OFF 는 이 레버에 무효"*). 이후 런은 전부 양팔 공통 PIN | **유예 후 인자가 재료대로 바뀌었는가**(전/후 tool_call 인자 diff). 현재 배달 6건 **전부 reward 0.0** 이고 '고쳐 썼다'는 사례가 표본에 없다. 부수로 **배타 체인 상호배제**(C505⒝ 축자 *"문서를 실제로 나른 경로가 바로 그 `rw_fb`(action-deny)라 상호 배제된다"*) 해소 여부 |

---

## 6. 2026-08-19 실패 부류와 레버 배치

### 6-1. WRONGARG 7 (값/종류 오선택) — **핵심 공백이 여기 있다**

| 하위 부류 | 배치된 레버 | 상태 |
|---|---|---|
| **집합 밖 이름 = 날조** | `T2_WRITE_ARG_ENUM`(deny 114·`[OFFICIAL-NAME]` 198) · `T2_WRITE_ARG_GROUND`(315) · `T2_PROD_BIND`(44) · `T2_SG_GROUND`(496·드롭) | 발화 활발 / 효과 미측정 |
| **참조 대상 오선택(transaction_id 등)** | `T2_REF_VERIFY`(41회·정면 표적) · `T2_REF_ISO`(死배선·HARMFUL) | 유일 정면 레버가 **인증 깨짐** |
| **축/종류 오선택(checking↔savings)** | `T2_ARG_AXIS`(deny 26·formalize 389) | **HARMFUL**(gold 축 거부 9·want 자기모순 42%) |
| **후보 판별 실패(fit 결과에서 다른 카드)** | `T2_FIT_DIFF` | **DARK**(부착 0·켜면 [[59]] 위반) |
| **선택 자체를 접지**(값이 대화에 없음) | `T2_CHOICE_GROUND` | **DARK**(중첩 인자 미처리) |
| **⛔ 집합 內 실재하는 이름·값 중 오답 선택** | **없음 — 레버 0** | ← **다음에 지을 것** |

**⛔ 빈 칸 명시.** t7326 의 WRONGARG 잔여(055·057·063)는 전부 **집합 內 실재 이름**이다. 이 칸을 덮는 레버는 하나도 없고, 인접 레버들이 스스로 사정거리 밖임을 자인한다:
- `T2_WRITE_ARG_ENUM` — 집합 밖만 차단(over-block 0/198 = 집합 內은 원리상 통과).
- `T2_WRITE_SUB` — C472⒡ 축자 *"검산은 `\"Blue Account\"` 가 텍스트 실재라 통과시켰다 — **날조는 잡고 오답은 못 잡는다**"*.
- `T2_WRITE_ARG_GROUND` — 코퍼스에 실재하는 틀린 값은 통과(원리적 상한).
- `T2_REF_VERIFY` — A2 `_note` 축자 *"merchant-absence는 cross-merchant만·동일상점 내 오선택(4 Costco 중 오답)은 amount 스펙 별도 필요"*.

⇒ [[62]] 순서대로: **먼저 격리 프로브로 "집합 內 오답 선택" 결손을 재라.** 격리에서 되면 레버는 전달(부하 축소)뿐이고, 격리에서도 실패하는 단계에만 결정론이다. **엔진이 "정답은 X" 를 내는 형태는 (H1) 실측대로 파괴적이다**(E_NEG 0/24).

### 6-2. MISSING 7 (변이 미실행)

| 방향 | 레버 | 실측 |
|---|---|---|
| **사는 쪽(변이를 밀어냄)** | `T2_WRITE_SUB`(pre-draft 전달 91·검산 통과 560) · `T2_BRANCH_REGROUND`(pre-close 100·walk 490) · `T2_WRITE_PROV`(완료-주장 대조·실발화 141) · `T2_VALUE_ACQUIRE`(획득 경로 1,272) · `T2_HAVE_VALUE`(DARK) | BRANCH_REGROUND 만 행동 축 대조 보유(close x1→x0), **pass 는 0** |
| **⚠파는 쪽(MISSING 을 만듦)** | `T2_WRITE_EVIDENCE`(미회복 5·전부 0.0) · `T2_ARG_EMPTY`(C425 *"실패 모드만 이동 — 빈 값 기록 → 기록 없음"*) · `T2_SG_GROUND`(operand 를 고치지 않고 **뺀다** → abstain) · `T2_RESOLVE_CAP`(계약 경로 3~7배 침묵) · `T2_PROD_BIND`(전-행 오강등 경로) | 전부 실측 |

**MISSING 칸의 진짜 문제는 레버 부재가 아니라 부호다.** WRONGARG 를 잡는 fail-closed deny 가 그 호출을 **MISSING 으로 옮기고**, DB 채점은 성공한 변이만 점수를 만들므로 그 이동은 순손실이다. ⇒ 재야 할 것: **모든 deny 레버에 대해 `deny 후 later_ok` 비율**(회복률). 현재 이 계기를 가진 것은 x392_block_join 한 건뿐이고 t7326 한 런에만 돌았다.

### 6-3. EXTRA/대체 2

| 레버 | 발화 | 상태 |
|---|---|---|
| `T2_GIVE_QUOTE` | stderr 734 / beat 314 / 넛지 철회율 51.2% | 철회는 **개입의 크기이지 성공이 아니다** |
| `T2_RESOLVE` operator-scope | 406회 | 격리 A=C(24/24 ↔ 24/24) = **개입 없음 대비 순이득 0** |
| `T2_OPERATOR_PINPOINT` | 플래그 설정 0 (구판은 `reason=operator-find` 429회) | HARMFUL |

**⛔ 계측 공백:** 철회된 give 가 **gold 요구였는지 검사하는 계기가 없다**. EXTRA 2건을 줄이는 대가로 MISSING 7(최대 부류)을 살 수 있는데 그 방향은 한 번도 재지 않았다. ⇒ 재야 할 것: **철회 188건 × (그 give 가 gold action 에 있었는가)**.

### 6-4. ACTION 채점 2 — **레버 0**

배치된 레버가 하나도 없다. ACTION 채점 sim 은 write 집합이 아니라 **행동 순서/호출 자체**를 보는데, 이 배치 43건 중 그 종점을 겨냥한 것은 없다(`T2_PAIRCHECK`/`T2_PAIRFIX` 는 하네스 보전이지 행동 축이 아니다). ⇒ 명시적 공백.

### 6-5. 부류-무관 손실: read 자리에 쓰이는 레버

DB 채점에서 **read 는 점수를 만들 수 없다**(PERTASK_FAILURE_ONSET_2026_08_19.md 축자). 그런데:
- `T2_ARG_EMPTY` 발화 26건 중 **21건(81%)이 read 도구**(`check_card_application_fit` 13 · `get_reward_discrepancies` 7 · `log_verification` 5).
- `T2_ARG_PRODUCERS` 705건 **전부** 검색/read 도구 지목.
- `T2_MATCH_COUNT` 2,043건 전부 회수 경계.
⇒ 재야 할 것: **레버 발화의 도구별 분포 × 그 도구가 gold 변이 집합에 있는가**. 이 한 지표가 "점수를 만들 수 없는 자리에 쓰인 예산" 을 바로 준다.

---

## 7. 계기 한계 — 이 감사가 못 본 것

**(L1) 커밋 궤적만 스캔하면 regen 채널은 원리상 0으로 보인다.**
`results.json.gz` 는 커밋된 호출만 담으므로 regen 안에서 소비되는 거부문은 **못 들어간다**(C440⒜ 축자 *"거부는 재생성 안에서 소비돼 커밋 궤적엔 교정된 호출만 남는다"*). 이 아티팩트가 만든 dark 오탐 = `T2_ARG_SCHEMA`·`T2_ARG_EMPTY`·`T2_WRITE_ARG_ENUM`·`T2_ARG_AXIS`. 양성통제로 쓰이던 `[GROUNDING WARNING]` 은 커밋 채널이라 통제가 성립하지 않았다.

**(L2) t7326 의 `.log.gz` 가 로컬에 없다.**
`run_t7326_stage1_nt2_20260819.sh:104` 이 stderr 를 리모트 `$LOG` 로만 보내고 :98-101 은 `results.json` 만 영속한다. ⇒ **stderr 가 유일 신호인 레버들의 "t7326 0발화" 는 확인된 진술이 아니다** — `T2_GROUND`(전 이력 385회) · `T2_PROD_BIND`(44) · `T2_BRANCH_REGROUND`(100/490) · `T2_DECIDE_BEFORE_WRITE`(20). LEVER_CONSOLIDATION 의 C군(死배선) 분류 중 이 4건은 재판정이 필요하다.

**(L3) fb/trace 사이드카는 일부 런에만 존재한다.**
`[VALUE-ACQUIRE]` 291건·`[DECIDE-FIRST]` 12건은 사이드카를 남긴 런에서만 배달이 검증됐다. `T2_REF_VERIFY` 41회가 난 런들에는 사이드카가 없어 **배달 여부 미확인**. ⇒ 발화 ≠ 전달([[55]]).

**(L4) 공용 태그 오귀속 4건.**
- `[T2_AXIS]` = 4레버 공용(x44_lever_coverage.py:74 축자 *"발화>0은 존재 증명 수준"*) — `SCALAR_ARRAY`·`FIT_DIFF`·`TERMINAL_TURN`·`TOOL_CHANNEL` 이 같은 6건을 각자 전부 자기 것으로 셌다.
- `[GROUNDING WARNING]` = `SG_GROUND` ↔ `GROUND_HDR` 공용(x44:80-81).
- `[T2_GROUND]` = `GROUND` ↔ `PROV_GROUND` 공용(단 후자는 SystemExit 로 차단돼 실제 오귀속은 없었다).
- `[T2_COMPUTE]` 269회는 **전부 다른 경로**(`select_discrepant` 판정불가)인데 x44 가 이를 T2_COMPUTE 발화로 세고 '정상' 판정을 냈다.

**(L5) beat 하드코딩 오귀속.**
`t2_gate_patch.py:7062`·`:7110` 의 `_lbeat("T2_WRITE_EVIDENCE", …)` 가 `_wtag` 와 무관하게 고정돼 있어 **`WRITE_ARG_GROUND`·`ARG_EMPTY` 발화가 `T2_WRITE_EVIDENCE` 로 계수된다**. `[T2_LEVER]` 기반 커버리지 집계는 이 배치에서 신뢰할 수 없다.

**(L6) 태그 계수 ≠ 발화 계수.**
`T2_WRITE_PROV`: `window hit` **12,038** ↔ 실제 `regen tool_calls=` **141** = **85:1**. 이 이름으로 인용된 수치(C121 '70'·C215 '91')의 99% 가 발화가 아니다([[08]] 위험).

**(L7) 파일 집합 자체가 새고 있었다.**
results 는 352 가 아니라 **419**개다(`_results.json.gz` 명명 변형 누락). 이 때문에 `[HAVE-VALUE]`·`[ARGS-FORMAT]` 계수가 과소 보고됐다(후자 54회/32파일 → 실제 **75회/42파일**).

**(L8) 원 산출물 부재 — 재검증 불가 7건.**
`x322`(operator scope 격리) · `x275`(ARG_AXIS 격리) · `x294`(dup-write) · `x84`(choice grounding census) · `x390`(scalar array census) 프로브의 **저장된 출력이 0건**. 근거 런 파일도 부재: `bank_dbw_off_20260812` · `bank_qpsmoke_20260801` · `x30run/smoke_trace.jsonl` · `logs/qpsmoke.log`. 수치는 코드 주석·2차 문서에만 산다. ⇒ **프로브 출력 영속화가 인프라 부채**(등대 갱신 프로토콜: scratchpad-only 인용 금지).

**(L9) 배타 체인·캡 공유가 상류로 하류를 가린다.**
`wd` 한 변수를 6종(WEV·WAG·ARG_EMPTY·REF_VERIFY·ASK_UNKNOWN_BOOL·HANDOFF_ARG_GROUND)이 t2_gate_patch.py:7049-7105 폴스루로 공유하고, `T2_WRITE_ARG_ENUM_CAP` 을 ENUM·AXIS 가 나눠 쓴다. ⇒ **"0발화" 가 표적 부재인지 상류 억제인지 대부분 구분 불가**. 유일하게 구분한 것은 `T2_HAVE_VALUE`(would-fire-but-suppressed 관측 분기).

**(L10) 선택편향 — 발화 sim 의 reward 평균은 인과가 아니다.**
레버는 모델이 틀릴 때만 발화하므로 "발화 sim 이 더 낮다" 는 당연하다(`T2_GROUND` 7/7 0.0 vs 비발화 0.260 등). 본 문서의 그런 수치는 전부 **관측 대조**로만 읽어야 하고 [[57]] 부정통제를 대체하지 않는다.

**(L11) 우리 층 자신이 근거원을 오염시킨 사례가 2건 확인됐다([[25]]).**
`T2_MATCH_COUNT` 의 거짓 완결 인증 19건, `T2_QUOTE_HINT` 의 substring 매칭(값 `"0"` → 주소 `80210` 지목). 계기 결함이 곧 모델 입력 결함이다.

---

## 8. 다음에 잴 것 (우선순위)

1. **[[62]]① — WRONGARG '집합 內 오답 선택' 격리 프로브.** 055/057/063 을 정보-맞춘 A_minimal vs B_fullctx 로. 격리에서 되면 레버는 **전달뿐**, 안 되면 그 단계에만 결정론. 지금 이 칸에 레버가 0개인 것은 결손을 안 재고 지었기 때문이 아니라 **재고 나서 아직 안 지은 자리**로 남겨야 한다.
2. **deny 레버 공통 계기 — `deny → later_ok` 회복률.** x392_block_join 를 정본화해 전 런에 상설. (H2) 실패-모드 이동을 상시 계측.
3. **레버 발화 × 도구 gold 멤버십.** read 자리에 쓰인 예산(ARG_EMPTY 81% · ARG_PRODUCERS 100%)을 한 지표로.
4. **would-fire-but-suppressed 분기를 전 레버에 이식.** dark 11건 중 10건이 (a)표적 부재 / (b)상류 억제를 못 가른다.
5. **깨진 인증 회귀 복구** — `test_ref_verify.py` / `test_ref_verify_replay.py` 3인자 호출(TypeError). 지금 인용되는 6/6·9/9·26/26 은 출고본을 검정하지 못한다.
6. **동봉/지목 채널의 정확도 감사** — `WRITE_ARG_ENUM` 의 `It answers: X`, `WRITE_SUB` 의 pre-draft, `DECIDE_BEFORE_WRITE` 의 `It has now been made:`. 세 곳 모두 우리 층이 **값을 선언**한다. 정확도가 동전 던지기면 (H1) 부류다.
7. **프로브 출력 영속화 규칙.** x-스크립트는 결과 JSON 을 `reports/facet_rft_2026/` 에 provenance 와 함께 남긴다. 현재 5개 프로브의 수치가 코드 주석에만 산다.

---

### 부록 A — 판정 기준

- **VALID_MEASURED**: pass/reward 를 종점으로, 부정통제 또는 동일-조건 대조군을 갖춘 대조에서 **이득 방향** 수치가 있고 그 산출 파일이 실재.
- **PLAUSIBLE_UNMEASURED**: 표적·발화·기전이 실재하나 위 조건의 수치가 없거나 원본 미확인.
- **DARK**: stderr 태그 · 효과 문자열 **둘 다 0**.
- **HARMFUL**: 정답(gold 인자·gold write)을 지웠거나 오답을 주입한 **실측 사례**가 1건 이상.
- **NOT_OPERAND**: 겨냥 결손이 인자 값/종류 오선택이 아님(하네스·계기·행동 국면·형식·say-don't-do), 또는 플래그가 아님.
