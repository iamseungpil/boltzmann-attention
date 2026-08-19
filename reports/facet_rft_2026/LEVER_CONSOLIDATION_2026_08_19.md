# LEVER_CONSOLIDATION — 레버 정리 권위본 (2026-08-19)

> 권위 범위: 이 문서는 **레버 지형·통합 원소·매핑·런 체계**의 정본이다. 등대(`RESEARCH_MASTER.md`)의 §1 프레임 LOCK·증거원장·실험큐에 종속하며, 그것과 충돌하면 등대가 이긴다.
> 기계가독 매핑: `lever_consolidation_map_2026_08_19.json` (274 레버 전수).
> 규율: 문장 단위 **[S] 확정 / [D] 추정 / [?] 미측정** 표기 · 축자 인용 유지 · 새 실패-분류 이름 만들지 않음([[48]]) · 상태값에 "끈다" 없음([[60]]).

---

## 1. 한 줄 요지

**레버가 많은 것이 병이 아니다. ⑴판정 근거가 *우리가 쓴 선언*에 몰려 있어 도메인을 바꾸면 술어가 통째로 거짓이 되고, ⑵그 판정들이 배타 `elif` 사슬 하나를 나눠 써 서로를 지우며, ⑶그 결과 우리 층이 우리가 지목한 이름을 우리가 막는다 — 이 셋을 고친다.**

바꾸는 방향은 **끄기가 아니라 재기저화(re-grounding)**다. 판정의 근거를 "A2 선언"에서 **원장**(env 가 스스로 선언한 스키마 + 이 대화가 스스로 남긴 기록)으로 옮기고, 선언에는 원장이 **원리상 못 아는 것**(정책 조건·절차 순서·문구)만 남긴다. 출구는 배타 선택이 아니라 **병합**으로 바꾼다(명령은 하나, 사실은 합집합).

- 산 것: 자기차단 56건(gold 오차단 44 · 미회복 8)의 **구조적 소멸** [S 기전 / ? 효과], ARGDIFF 교정 문구의 rank 12칸 선점 해소 [S 기전 / D 효과], 그리고 무엇보다 **측정 가능성** — 레버 274종 단일 스택에서는 기여 귀속이 원리상 0건이다 [S].
- 사지 못한 것: NEVER 89 중 **정책 문장이 축자 도달했는데도 안 부른 21건**. 이것은 지식·회수·이름의 결손이 아니라 **이행**이고, 이행 촉구는 이미 라이브 null 로 측정됐다(C492 · C529) [S]. 이 문서는 그 축을 산다고 주장하지 않는다.

---

## 2. 현황 수치 (전부 2026-08-19 실측)

### 2-a. 코드 지형 [S]

| 항목 | 값 | 근거 |
|---|---:|---|
| 엔진·모듈이 `environ.get` 하는 `T2_*` | **269** | `grep -rho 'environ.get("T2_[A-Z0-9_]*"' t2_*.py \| sort -u \| wc -l` |
| 본 문서 매핑 대상 레버(死배선·모순 포함) | **274** | 매핑 JSON `total_levers` |
| `t2_*.py` 모듈 수 | **77** | `ls t2_*.py \| wc -l` |
| `t2_gate_patch.py` 줄 수 (모놀리스) | **11,415** | `wc -l` |
| 상위 7파일 합계 | **17,930** | gate_patch 11,415 · scaffold_get 2,091 · eplan_patch 1,208 · resolve 1,153 · compute 965 · levers 700 · run_gated 398 |
| `t2_levers.py` 레지스트리 등재 `T2_*` | **139** | ⇒ 엔진이 읽는 269 중 **130종이 레지스트리 밖** |
| `go_stack.sh` 의 `export T2_` 줄 | **105** | 자칭 "정본 GO-STACK 런처(single source of truth)" |
| `run_t7326` PIN 항목 | **27** (ON 15 · OFF 12) | **PIN ON 15종 전부가 go_stack 에 없다** ⇒ 두 스택은 비교 불가 |
| 테스트 `test_*.py` | **179** | 레버↔테스트 대응표 없음 |
| 프로브 `x*.py` | **410** | [[67]] "사본 금지" 의 실질 규모 |
| 런처 `run_*.sh` | **122** | 런처 정본 문제의 원인 |

### 2-b. t7326 발화·죽음 [S]

- trace 13,195행(halfA 6,242 + halfB 6,953) · sim 40 전수.
- **발화 마크 108종** / 런타임 리터럴 마크 174종 / **40 sim 전부에서 발화한 마크는 13종뿐**.
- **死배선 후보 C군 11종** = 노브 ON + 인쇄 지점이 효과 경로 + 40 sim 0발화: `BRANCH_REGROUND` · `CHOICE_GROUND` · `KB_NOHIT_SURFACE` · `PHASE_OWNER` · `PROD_BIND` · `SG_TRUTH` · `SG_WINDOW_ABSTAIN` · `WITHDRAWN_ROW` · `GIVE_RELEVANCE` 외.
- **coverage 가족이 이미 죽어 있다**: `COV_MIDDRIVE` 0 · `COV` 0 · `READALL` 0 · `DISPATCH_LEDGER` 2 · `COVERAGE_FU` 2 = **합 4발화**. 그런데 다중요구가 gold 결말의 **150/289(52%)** 다.
- 억제 총량 **606회**를 네 개의 서로 다른 기구가 나눠 집행: 지문창 억제 150 · deny 본문 접힘 116 · 지침 drop 84 · `ARBITRATE` 72 · `PHASE_PRECEDE` 184. `route ≠ chose` **76턴**.
- 재료 배달 정지 **393회**: `resolve_cap` 225 · `other_lever(gate)` 68 · `(prov)` 62 · `(wev)` 32 · `(eplan)` 6.

### 2-c. 도달(사이드카) — **인벤토리 오류 정정** [S]

인벤토리는 `T2_FB_SIDECAR` 를 "어느 런처에도 없음(OFF)"으로 등재했고, 진단은 "도달을 못 잰다"고 결론했다. **둘 다 틀렸다.** t7326 라이브 사이드카가 실재한다:

```
/home/woori/scratch/logs/fb_bank_t7326_half{A,B}_20260819q.jsonl
합 2,153행 · sim 40 전수 · text 필드 포함
kind    : reminder-user 684 · reminder-assistant 654 · tool-deny 491 · route 317 · speak-prohibit 7
channel : unified_regen 1412(66%) · writesub 172 · claimprov 134 · uncalled_unlock 21 · usertoolnote 19
          · channel 10 · searchexhaust 10 · followup_chain 8 · verdict_surface 8 · speak_gate 7
          · givexec 6 · signature 5 · uninstructable 4 · givequote 4 · writeprov 3 · transfertier 2
          · unkrepeat 2 · covfollowup 2 · truncguard 2 · envguard 2 · followup 1 · selfdecl 1 · followup_decision 1
```

함의 3가지 [S]:
1. `T2_CLAIMPROV` 마크 **1,124** ↔ 도달 **134** = **8.4:1**. 마크는 도달이 아니다.
2. `T2_TOOL_SIGNATURE` 마크 44 · deny 18 ↔ 도달 **5**. 세 설계안이 각각 문단을 할애해 논쟁한 레버의 라이브 노출이 40 sim 에서 5회다.
3. 개별 레버 문면 대부분이 **한 자릿수 도달**이고 `unified_regen` 이 66% 를 먹는다 ⇒ "레버 통합으로 pass 를 산다"의 상한이 애초에 작다. **표적은 개별 레버가 아니라 regen 채널 자체일 수 있다** [D].
4. 개입 강도의 sim별 편차 **4 ~ 118 (30배)** ⇒ 어떤 A/B 에서도 교락. 층화 또는 공변량 필수.

### 2-d. 결말·차단 [S]

- **pass 7/40**. 2/2 = 098 · 100. 갈림 = 017 · 024 · 050.
- gold 액션 단위 결말: **MATCH 142 · NEVER 89 · ARGDIFF 53 · OURS 2 · ENV_REJECT 3**.
- deny **173** = JOIN 114 + TARGET 49 (= 핸드오프의 "163") + ARGVAL 10.
- gold 이름 표적 **56** · 회복 45 · **미회복 11**(040×5 · 063×2 · 072 · 079 · 093 · 094).
- **자기차단**(우리 층이 같은 sim 에서 먼저 지목한 이름을 우리가 막음) = deny 173 중 **56 (32%)** · gold 56 중 **44 (79%)** · 미회복 11 중 **8**.
- gold 오차단 56 의 마크별 귀속: `WRITE_EVIDENCE` 22 · `RESOLVE` operator-fab 12 · operator-scope 12 · `DISPATCH_ROLE` 3 · `WRITE_ARG_GROUND` 2 · `UNLOCK_PROV` 2 · `UNLOCK_NAME` 2 · `TOOL_SIGNATURE` 1.
- **deny 0 인 두 태스크(098·100)만 2/2** — 상관이지 인과가 아니다(deny 는 고전하는 sim 에서 더 나온다 = 교락) [S].

### 2-e. 분모 오염 3건 — 인용 전 반드시 교정 [S]

1. **NEVER 89 → 사건 35~36개.** unique (sim, 도구명) = 36 조합. 다중요구 31 조합이 83행을 만든다. `get_bank_account_transactions_9173` 한 사건이 26행. ⇒ "NEVER 89 를 표적 크기"로 쓰면 2.5배 과대.
2. **ARGDIFF 53 중 47(89%)이 다중요구 그룹 내부**이고 id 오결속 33 · 필수 인자 `None` 24. 단일 호출 위반이 아니라 **호출들 사이의 관계** 위반이다.
3. **감사 도구가 위반을 1.8배 과대 보고한다.** `x6h_engine_literal_audit.py:267-268 selftest_range()` 가 `if __name__` 블록만 테스트 구역으로 봐서 `def selftest():` 안 픽스처 60건을 live 위반으로 인쇄한다(136 → 정정 **76**). **감사를 고치기 전 감사 수치 인용 금지.**
4. **종점 프록시 오염**: reward 1.0 인데 gold 결말이 완전 MATCH 가 아닌 sim 2건(017 t1 = NEVER 2 · 050 t1 = ARGDIFF 1). tau2 채점은 DB 상태이지 액션 매칭이 아니다. 그리고 `task_079#t1`(LOOP·steps 150·gold 0)은 `x392.ends` 에 **행이 하나도 없다**(40 중 39 sim 만 등재) ⇒ **가장 나쁜 sim 이 분모에서 증발한다.**

### 2-f. 도메인 오염 — 전이 성립 여부 [S]

- **FATAL-1**: banking 디스패처 3종(`give_/call_/unlock_discoverable_*`)이 **판정 술어 라이브 8곳**에 하드코딩: `t2_gate_patch.py:11101 / 11135 / 11230 / 11248 / 6670 / 6761` · `t2_axis_levers.py:99 / 104 / 106` · `t2_search.py:383`. t7326 실발화 **≥51회 / ≥19 sim**. ⇒ **"같은 스택을 도메인만 바꿔 돌렸다"는 주장이 지금은 성립하지 않는다.**
- **FATAL-2**: `t2_compute.py:301-521` `catalog_filter` 에 banking 카드 필드 **42 리터럴 · ~220줄 전부 판정**.
- **FATAL-3**: `t2_axis_levers.py:136/140/141` 이 도구 출력 산문을 정규식으로 뜯는다 = [[59]] 정면. 현재 `T2_FIT_DIFF` OFF 라 실험 무효는 아니나 **켜는 순간 무효**.
- **FATAL-4**: `t2_gate_patch.py:2349 _PROCEDURAL_RE` 의 `^log_ ^verify_ _verification$ ^kb_ ^shell$` 5개. `transfer_to_human` 만 5/5 도메인 공통이라 정당.
- 반면 repo 는 **이미 정답 경로를 두 곳에 갖고 있다**: `t2_gate_patch.py:2514 _dispatch_tools()` 는 스키마 형상(`agent_tool_name` ∈ props · `arguments` 유무)으로 디스패처를 도출하고, `:2566 _is_read_tool()` 은 축자 *"`__tool_type__` is what tau2's own metrics use to split writes from reads"* 로 env 자기선언을 읽는다. **새 기구를 지을 필요가 없다 — 이미 옳게 지어진 두 함수를 판정면 전체로 확대하는 일이다.**

### 2-g. 선언 지형 [S]

- `policy_ontology.rows` **153행 전부** `{doc, quote, quote_match}` 보유 · 원격 코퍼스 축자 검증 **153/153 통과**(exact 152 · normalized 1). ⇒ 이것이 [[23]] 모범 형식이고 나머지가 따라야 할 규격.
- 그러나 선언 단위 80개 중 **25개(31%)가 출처 표기 없음**(`field_ops` 4/4 · `compute_ops` 2/2 · `discoverable_name_check` 2/2 …).
- `scaffold_get_tools[7]/[9].op.table` **23행**의 `source` 가 산문 글롭(`"doc platinum_rewards_card_*"`)이라 **기계 검증 불가**.
- retail specific 10키 중 2키만 출처 표기 · **airline specific 0키** ⇒ 전이 arm 은 [[23]] 감사 대상이 아직 없다.

---

## 3. 통합 아키텍처

### 3-0. 왜 이 형태인가 (설계 3안 심사 결과)

승자 = **원장 우선(ledger-first)**. 세 안의 차이는 하나로 요약된다 — 1안·2안은 하드코딩을 **선언으로 옮기는** 안이고, 승자는 **env 가 이미 자기선언한 것을 읽어 선언 자체를 지우는** 안이다. 그 전제가 희망이 아니라 코드에 실재함을 §2-f 에서 확인했다.

패자에서 **이식(graft)한 것** — 이식 없이는 승자에 구멍이 있다:

| 출처 | 이식 항목 | 왜 필요한가 |
|---|---|---|
| 안1 | **INFER 기본 `none` 규율** | 승자는 `L4`·`PRINCIPLE_DEFAULT`·`DISAMB` 를 "선언 잔류"로 밀어놓기만 하고 **선언이 없을 때 무엇이 일어나는지 규정하지 않았다**. 미선언 = 실행 안 함 = ASK 낙하 |
| 안1 | **`P10_ASK` 원소 신설** | 승자에 권위 이전이 **아예 없었다**. [[48]] `권한 월권`(12/14 실측·레버 없음) 과 [[52]] "ASK 가 skeptical 보다 강하다"가 통째로 빠진다 |
| 안1 | **승격 조건 = 발화가 아니라 *평가* 확인** | [[60]] 조용한 끄기의 유일한 방어 |
| 안1 | **음성 실측 보유 레버는 기본 침묵 + 근거 이관** | 원소 안으로 접으면 개별 근거 상태가 안 보이게 된다 |
| 안2 | **`t2_dominance.requirements_for()` / `merged_text()` 를 `P9_REQUIRE` 로 그대로 재사용** | 승자의 `P5_SETGAP` 은 집합 차만 내고 **"지금 실행 가능한 첫 단계"가 없다** — [[64]] 의 "다음 한 수"를 원리상 못 만든다. 이미 라이브 배선(`t2_gate_patch.py:7537/7589/7952`)이고 자기 문서에 **"새 A2 키 0"** ⇒ 비용 0 |
| 안2 | **병합 규율: 명령은 하나, 사실은 합집합** | 초판이 전부 명령형 나열이라 task_101 이 첫 항목만 집고 나머지를 흘린 실측 |
| 안2 | **래칫 검정: 레버 → 원소 대응이 비면 CI 실패** | [[60]] 의 유일한 *강제* 수단 |

**채택하지 않은 것**: 안2 의 "결정점 밖 무개입". 040형(영구 `No records` → 6-호출 사이클 ×30 → turn 104 소진)과 008형(문맥 초과)은 **결정점에 도달하기 전에 죽는다** [S]. 재료 배달·부재 종결·하네스는 관문 밖에서도 살아야 한다.

### 3-1. 원소 계약표

⛔공통 불변식 3개 (모든 원소에 적용):
1. **엔진은 정답을 고르지 않는다** — 집합 밖이라는 사실, 다르다는 사실, 아직 안 했다는 사실만 말한다([[62]] · x322 지목 24/24 → **0/24**).
2. **거부는 이름과 다음 한 수를 담는다** — 발화를 보류해도 본문은 지우지 않는다([[64]] · 이름 없는 *"먼저 해소하라"* 가 3회↑ 나온 6 sim = **6/6 실패**).
3. **메인에는 답만** — 재료는 서브 안에서 끝낸다([[65]] · x231 `ineligible_text` 한 줄로 task_100 8/8 → 0/8).

| 원소 | 엔진이 보는 것 (닫힌 술어) | LLM 이 내는 것 | 출력 | 금지 |
|---|---|---|---|---|
| **P1 KIND** | env `__tool_type__` (`READ/WRITE/GENERIC/THINK`) · 접근자 `_is_read_tool(env,name)` | — | `is_write` / `is_read` | 이름 철자 판정(`^log_` 류 접두사) |
| **P2 EVENT** | `tool_calls.id` ↔ `role=="tool"` 조인 · `error` · `content.startswith("Error:")` · `_call_key` 동일성 · `_stale_call_ids` | 호출 | EXECUTED / COUNT / ERR / SAMECALL / STALE | 판정 0 — 계수만 |
| **P3 NAME** | 한 이름의 **7비트**: (a)레지스트리 실재 (b)이 대화 도구 출력에 등장 (c)unlock 인자 등장 (d)give 성사 (e)호출됨 (f)env 반려 이력 (g)**우리가 말함** | 이름 선택 — `formalize_intent_tool` **결정점당 1회** | 튜플 + 호출 가능 형식 | 두 후보집합으로 formalize 를 두 번 부르는 것 · 어느 이름이 옳은지 지목 |
| **P4 GROUND** | `_flatten(_args_dict(tc))` leaf ↔ `_context_text` 부분문자열(`_ctx_has`) · `_shared_span` n-그램 · 열거/축 소속 · 공백 · 수치 캡 · 유효창 | 값 + **근거 인용**(evidence_quote) | 미접지 leaf 목록 + 후보 record | 유사도·의미 대조 · 옳은 값 제시 |
| **P5 SETGAP** | LISTED∖EXAMINED · TARGETS∖SUBMITTED · GIVEN∖RAN · 미판정 행 · **대상별 분화**(gold 다중요구 대상 집합 ∖ 이미 소비한 대상) | — | 남은 원소 id 목록 | 아무것도 막지 않는다 |
| **P6 SPEECH** | 본문 ∩ 레지스트리 − CALLED · tool_calls 빈 턴 · `_quiet_turns` | — | 미이행 이름 + 정체 턴수 | 이름과 형식까지만 |
| **P7 CHANNEL** | `finish_reason=="length"` · 봉투 태그 · 토큰 수 · vLLM 에러 원문 · call↔result 쌍 | — | 커밋 금지 / 재생성 / mt 축소 / 쌍 교정 | 도메인 주장 0 |
| **P8 GOV** | 이번 턴 각 원소 판정 목록 · 직전 발화 지문 · **인자·요건 집합이 변했는가** | — | **병합 문면 1개**(명령 하나 + 사실 합집합) + 채널(평서/regen/required) | 배타 선택 · 횟수 캡 · 이름 없는 보류 |
| **P9 REQUIRE** | A2 요건 그래프(`gates[]`·`require_tool_before`·`requires_reads`·`eplan`·`procedures`) × 실행 원장 | — | `[{id, predicate, satisfiers[1]}]` = **지금 실행 가능한 첫 단계** | 사슬의 끝을 명령하기 |
| **P10 ASK** | 후보 카디널리티 ≠ 1 · A2 `authority` 가 손님 지정 · `required` ∧ 값 부재 · `on_ambiguous` 모드 | 질문 문면 · 손님이 실행할 인자값 | 되묻기 + 손님-실행 채널 정합 | 손님 몫 판정을 우리가 대신 하기 · over-ask 무계측 |
| **D DECL** | (엔진 아님) A2/A3 선언면 | — | 정책 조건·절차 순서·문구·정책산문 enum | 출처(문서 id + 축자 인용) 없는 항목 |
| **INSTR** | 거동 불변 인쇄 | — | 계측 | **레버 분모에 넣지 말 것**(C434 "인쇄이지 레버가 아니다 ⇒ 감사 대상 46 → 8") |
| **ARM** | 측정 노브 | — | 귀속 실험 | 기본 스택에 넣지 말 것 |

### 3-2. 무엇이 실제로 바뀌는가 (기전 3개)

**⑴ 자기지목–자기차단이 정의상 성립 불가가 된다** [S 기전].
현재 원인은 코드 두 곳으로 특정돼 있다: `_t2_our_names` 에 쓰는 곳이 읽기-루틴 핀 **한 곳뿐**(`t2_gate_patch.py:2656-2658`)이고, 지침이 `rw_fb[0] is None` 경로에서 **비커밋 UserMessage**(`:9328-9344`)로 나가 `stated_names`(role=="tool" ∧ error=True 만 훑음 · `t2_resolve.py:86`)에 **구조적으로 안 잡힌다**. P3 가 (b)(c)(f)(g) 를 **한 튜플에서** 읽으면 실물 ②(`task_093` t1 *"submit_interest_discrepancy_report_7294 를 unlock 하라"* → t3 그 호출 차단)가 재현 불가가 된다.

**⑵ ARGDIFF 교정 문구의 12칸 선점이 사라진다** [S 기전 / D 효과].
`_SRC8`(`t2_gate_patch.py:9105-9110`)은 정확히 17칸이고 `wev` 는 색인 5(rank 8) · `write_enum` 은 색인 16(**rank 19**)이다. 실물 ④ `task_063#s626729 t44`: `[T2_ARG_AXIS] deny got=savings want=['business_savings']` 와 `[T2_ROUTE] open_bank_account 경합 2 → resolve_write 승 · 밀림 write_enum` 이 같은 턴에 있고 결말이 `account_type: savings≠checking`. **교정 문구는 실재했으나 밀렸다.** P8 병합 출구에는 rank 가 없다.

**⑶ 전이가 처음으로 성립한다** [S].
`write_tools` · `dispatcher_role_check` · `discoverable_name_check` · `tool_signatures` 선언이 **사라진다**(env 가 대체). 전이 최소 생존선 = 도메인당 **2줄**(`entity_key` + `failure_markers`).

코드 자신이 이미 이 진단을 적어 놓았다 — `t2_gate_patch.py:9222-9226` 축자:
> *"위 `elif` 는 같은 tool_call 에 대해 **하나만** 내보낸다 … 오프라인 32/32 인 문장이 라이브에서 3/6 만 닿았고, 원인의 절반이 이 배타성인 것을 오늘에야 코드 추적으로 알았다 — **계수가 없어서 몰랐다.**"*

---

## 4. 전 레버 매핑표 (274 전수)

⛔상태값에 **"끈다"가 없다**([[60]]). `폐기확정`은 *음성 실측을 가진 것*이고 반드시 원장 인용을 동반한다 — 끄기와 다른 범주다.
⛔`死배선복구`는 **구현·측정은 있는데 라이브에서 한 번도 안 돈 것**이다. 통합의 부수 효과로 자동 점등되므로, 켜기 전에 [[62]] 4문을 답한다.

총 274 레버 · 원소 13 · 상태 9종 [S] (2026-08-19 실측 인벤토리 기준)

| 원소 | 건수 |
|---|---:|
| `P1_KIND` | 2 |
| `P2_EVENT` | 16 |
| `P3_NAME` | 26 |
| `P4_GROUND` | 58 |
| `P5_SETGAP` | 17 |
| `P6_SPEECH` | 10 |
| `P7_CHANNEL` | 29 |
| `P8_GOV` | 57 |
| `P9_REQUIRE` | 18 |
| `P10_ASK` | 3 |
| `D_DECL` | 14 |
| `INSTR` | 16 |
| `ARM` | 8 |

| 상태 | 건수 |
|---|---:|
| 통합 | 103 |
| 유지 | 53 |
| 死배선복구 | 42 |
| 폐기확정 | 17 |
| 근거보강필요 | 16 |
| 계기재분류 | 14 |
| 실험노브 | 12 |
| 선언이설 | 10 |
| 모순확정필요 | 7 |

### P1_KIND — 행위 종류 — env 자기선언(__tool_type__) 로 write/read/generic 판정  (2)

| 레버 | 상태 | 근거·주 |
|---|---|---|
| `T2_FAB_STRIP` | 통합 | _PROCEDURAL_RE 5접두사(FATAL-4) 폐기 -> __tool_type__==WRITE. t7326 8발화/4 sim |
| `T2_STALE_STRIP` | 통합 | committed 성공 write 재호출 strip. _PROCEDURAL_RE 간접 의존 -> __tool_type__ 로 |

### P2_EVENT — 실행 원장 — tool_call.id 조인·계수·동일호출·실패반복  (16)

| 레버 | 상태 | 근거·주 |
|---|---|---|
| `T2_CLAIM_BLOCK` | 死배선복구 | 194 sim 술어 4종 재측정·최종안 과차단 0·pass 0 [S]. 어느 런처에도 없음 = [[60]] 진짜 표적 |
| `T2_CLAIM_PROV` | 근거보강필요 | 마크 1,124회(발화 2위)인데 사이드카 도달 134 (8.4:1) [S]·원장 등급 축자 '효과 [?]'(C341) |
| `T2_CLAIM_VERIFY` | 통합 | 격리 완료주장 검증. spec 미설정이면 return [] = 선언 종속(옳은 형태) |
| `T2_CONS_NOOP` | 통합 | noop_write 감지. 코드 기본 ON 인데 go_stack 미선언 = DEFAULT_ON 미등재 사각 [S] |
| `T2_DUP_REPRESENT` | 통합 | 축약이 지운 재료 복원 = 원장 정규화의 역방향 |
| `T2_LEDGER` | 유지 | 원장 등재. 뷰 큐잉은 P8 로 분리(뷰 예산이 사실을 지우면 안 된다) |
| `T2_NOW_SELFCALL` | 유지 | 상수 반환 self-call(DB 무접촉·READ·순위/지목 문장 0) = P2 계약의 교과서. C488 '성적이 0 만큼 움직였다'·C494 부정통제 24<->24 |
| `T2_NO_DIGEST_REEXEC` | 통합 | digest 캐시 pop = 원장 조인. 런처 0 |
| `T2_READ_DEDUP` | 유지 | 라이브 pass 이동 실측 4종 중 하나. C114 [S] rall17 050 t1=13/13 첫 완전 PASS |
| `T2_READ_NEARDUP` | 死배선복구 | Jaccard 근사 중복 read. 런처 0 |
| `T2_RETRY_CONTROLLER` | 死배선복구 | C261 'gated 는 이 스택에서 설치되지 않는다 => 켜도 발화 불가·켜면 무음 실패' |
| `T2_SEARCH_EXHAUST_NUDGE` | 통합 | 엔진 자기 스텁 계수(닫힘). C214/E3 012=8회 전패 후 앱 절차 날조 |
| `T2_SG_DEDUP` | 통합 | 서브 도구 출력 dedup |
| `T2_TOOLERR` | 死배선복구 | 도구 에러 분류. 런처 0 |
| `T2_WRITE_CAP` | 死배선복구 | C261 'gated 미설치 => 켜도 발화 불가'. t102형 19x 성공-write 재emit 이 표적 |
| `T2_WRITE_PROV` | 통합 | 사임 시 write 주장 provenance |

### P3_NAME — 이름 원장 — 한 이름의 7비트 튜플(실재/회수/해제/전달/호출/반려/우리가말함)  (26)

| 레버 | 상태 | 근거·주 |
|---|---|---|
| `T2_ACTION_INDEX` | 통합 | 행동 색인 43줄 표면화. t7326 26발화/26 sim. 엔진은 고르지 않음 |
| `T2_ARG_PRODUCERS` | 통합 | 인자 값의 출처 도구 매핑 = 이름 원장 조회 |
| `T2_ARG_SCHEMA` | 死배선복구 | 주석 축자 '이 블록은 patched() 안이고 라이브 러너는 unified() 를 설치 = 現 死코드(P11 이설 대상)'. go_stack=1 인데 t7326 0발화 [S] |
| `T2_CALLABLE_HINT` | 통합 | base 이름 -> _NNNN 실명 = env 레지스트리 조회 |
| `T2_CALL_FORM` | 통합 | 호출 가능 형식 변환. C419 격리 [S] 16/16 / 라이브 [?]. 배타 체인 11위(C427) -> P3 단일 출구 |
| `T2_DISCOVERY_NAMES` | 통합 | 회수 문서가 이름을 말한 미호출 도구. t7326 162발화. C303 '그 가지에 애초에 들어가지 않는다'(사정거리) |
| `T2_DISCOVERY_STEP2` | 통합 | **자기차단의 원천**: formalize_intent_tool 을 후보집합만 바꿔 두 번 부른다(t2_resolve.py:189 회수집합 <-> :455 레지스트리 폴백) [S]. P3 는 튜플만 내고 formalize 1회 |
| `T2_DISPATCH_ROLE` | 통합 | 실행 주체를 레지스트리에서 도출. t7326 6발화/6 deny·미회복 11 중 1건(063 t35) |
| `T2_DISPATCH_ROLE_ENVSET` | 통합 | _user_discoverable 대조 = 튜플 (d)비트 |
| `T2_HANDOFF_PREDICATE` | 폐기확정 | C529 [S] '표적이 거의 없고 pass 는 null 이며 부호까지 뒤집혔다 => 폐기'. 술어((d)비트)만 P3 에 남긴다 |
| `T2_OPERATOR_PINPOINT` | 폐기확정 | x322 지목 24/24 -> 0/24 [S]. 기본(=범위 표면화)이 P3 계약 그 자체 — 지목 모드는 재론 금지 |
| `T2_PENDING_DISCOVERED` | 死배선복구 | C539 [S] '_ts 는 다른 함수의 지역 별칭이라 처음부터 죽은 레버'. 런타임 discoverable = 레지스트리 갱신 |
| `T2_PROV_OURS` | 유지 | **자기차단 수리의 핵심**: _t2_our_names 등재점이 읽기루틴 한 곳뿐(:2656) => P3 가 (g)비트를 단일 관리 |
| `T2_RESOLVE` | 근거보강필요 | 최대 판정기(519발화·92 deny·gold 오차단 24) 인데 원장 등급 축자 '효과는 [?]'(C324) |
| `T2_SG_TRUTH` | 死배선복구 | 노브 ON·t7326 0발화(C군). 인터페이스-사실 정정 |
| `T2_TOOLGATE` | 유지 | _t2_known_tools = 튜플 (a)비트 |
| `T2_TOOLLIST` | 통합 | 보이는 도구 목록 표면화(다른 피드백 전부 없을 때만) = 배타 체인 종속 |
| `T2_TOOL_CHANNEL` | 통합 | t2_axis_levers.py:99/104/106 이 give/call/unlock 3리터럴을 판정 술어에 박음(FATAL-1) -> P3 튜플로 소멸 |
| `T2_TOOL_SIGNATURE` | 근거보강필요 | **원장이 명시적으로 OFF 를 권고한 유일 레버**. C267 [S] 'V7이 금지하는 형태가 DB를 맞춘 경로다 … 승격 금지·다음 런에서 OFF 권고 … DB 축의 대가는 한 번도 재지 않았다' <-> go_stack:132 ON·t7326 44발화/18 deny/도달 5 |
| `T2_UNAVAIL_PROMISE` | 통합 | C207 3종 중 유일하게 도메인 판정. 레지스트리 차집합 |
| `T2_UNCALLED_UNLOCK` | 통합 | (c)∧¬(e). C12 053: gold 16개 중 15 맞추고 남은 하나가 unlock 만 하고 호출 없음 |
| `T2_UNKNOWN_NAME_BL` | 통합 | 채널 수정판 C264 [S] 'gold 차단 sim 2 -> 0'·회귀 8/8. 구판은 task_017 gold give 18회 차단 |
| `T2_UNLOCK_NAME` | 통합 | 구판은 '이름에 _숫자 없음' 철자 규칙으로 x99 7발 7오발화 => 트리거 교체됨 |
| `T2_UNLOCK_PROV` | 통합 | 미회복 11 중 1건(063 t19). _uv not in _ctx2 and _uv not in _ours2 가 우리 핀을 막던 자리 -> (b)∨(g) 를 같은 튜플에서 |
| `T2_UNLOCK_QUIET` | 폐기확정 | 격리 E_ISO 2/8 · G_ISO_STATE 6/8 < A_FREE 8/8 => 격리가 잘하는 종류가 아니다. 근거가 코드 주석에만 있어 원장 이관 필요 |
| `T2_USER_TOOL_NOTE` | 통합 | 018/040: 대화-내 user-tool 실행을 'portal/app 제출'로 오설명 -> 2회 거부 -> 이관 -> gold write 0. FATAL-1 리터럴 1곳 |

### P4_GROUND — 접지 — 인자 leaf 가 대화 텍스트에 축자 실재하는가  (58)

| 레버 | 상태 | 근거·주 |
|---|---|---|
| `T2_ABSTAIN_FIELDS` | 통합 | 결핍 필드를 출처별로 갈라 문구화 = 기권도 근거를 댄다 |
| `T2_ARG_AXIS` | 통합 | '다르다'만 알림·어느 축이 옳은지 판정 안 함 = P4 계약의 모범. sim당 1회 하드캡은 P8 로 |
| `T2_ARG_EMPTY` | 근거보강필요 | C425 [S] '표적을 사지 않았다'로 철회. 그러나 t7326 ARGDIFF 53 중 None 인자 24건 실재·040 발화 0(선점) [S] => 철회 근거와 실측이 충돌·재판정 필요 |
| `T2_AUTOFETCH` | 폐기확정 | C34 [M] '엔진이 주문마다 대신 호출해 주입' = 규칙 0 위반. 재론 금지 |
| `T2_CALC` | 死배선복구 | A2 calc_specs 구동. banking_knowledge 에 표적 0 => 전이 시 부활 대상(C261) |
| `T2_CHOICE_GROUND` | 死배선복구 | 노브 ON·0발화. gold 도 3건 미접지라 deny 면 오차단 => 넛지 강도 유지 |
| `T2_COMPUTE` | 통합 | A2 compute_ops 결정론 수리. catalog_filter 42리터럴 A2 이설이 선결(FATAL-2) |
| `T2_CONSISTENCY` | 死배선복구 | A2 일관성 deny. 런처 0 |
| `T2_DELIVER_PRECOMMIT` | 폐기확정 | C502 [S] '전달의 3분의 1이 우리 슬롯에서 소리 없이 사라졌고 1차 종점은 순환'·지연 1.38x => 특허 실시예 기록 금지 |
| `T2_DOCS_AT_WRITE` | 폐기확정 | C505 [S] '축별 배달은 반증됐다 — 굶은 축이 맞히고 먹은 축이 틀렸다' |
| `T2_ELIG_LINE` | 근거보강필요 | C517 [M] 사전 고정 바 통과·부정통제 유효(자격 뒤집기 오선택 3->26) [S] 인데 t7326 PIN=0 = [[60]] 위반 실물 |
| `T2_FEXEC` | 死배선복구 | 형식화-실행 모듈 진입. 런처 0 |
| `T2_FIT_DIFF` | 폐기확정 | t2_axis_levers.py:136-143 이 도구 출력 산문을 정규식으로 뜯는다 = [[59]] 정면 위반. 켜기 전 폐기·구조화 dict 경로로 대체 |
| `T2_FN_ISOLATE` | 死배선복구 | 함수 단위 격리 wrap. 런처 0 |
| `T2_GIVE_QUOTE` | 통합 | give 직전 손님 말 축자 실재(fail-open). 인용을 인자에 얹는 것은 금지(여분 키=evaluator 파괴) |
| `T2_GIVE_RELEVANCE_NUDGE` | 통합 | give 대상이 원장에 미등장 => 넛지(강제 금지·'무관하다'는 열린 술어) |
| `T2_GROUND` | 유지 | P-A GROUND(T5-C rev3) |
| `T2_GROUND_DROP_NAVKEYS` | 死배선복구 | 네비 키 접지 제외 = 오차단 완화. 런처 0 |
| `T2_HANDOFF_ARG_GROUND` | 死배선복구 | N1 실측: give 80회 중 75회가 도구명만 싣고 값은 본문에 — 손님 실행 인자값 157 중 142(90%)가 산문에 축자 [S]. 런처 0 |
| `T2_L4` | 폐기확정 | 치환 성적 2/2 오답(t58 정답파괴·t20 제약절단). A2 선언 없으면 실행 불가(INFER=none 규율) |
| `T2_MATERIAL_BYPASS` | 死배선복구 | 재료 배달을 판정 예산에서 떼는 유일한 기존 구현. t7326 배달 정지 393회(resolve_cap 225·other_lever 168) [S] |
| `T2_MATERIAL_RESERVE` | 폐기확정 | C499 [S] '무동작이었고, 그 진단의 전제였던 내 로그 독해가 틀렸다' |
| `T2_NLNUM_PROV` | 死배선복구 | 최종 발화 금액 provenance. 런처 0 |
| `T2_NOREC_BRANCH` | 근거보강필요 | 040형(영구 No records -> 6-호출 사이클 x30 -> ctx 초과) 표적. t7326 이 첫 라이브·C536 [D] 라이브 인과 미측정 |
| `T2_PRESENT_NESTED` | 死배선복구 | nested list/dict 를 명시 choice-set 으로. 런처 0 |
| `T2_PRESENT_READS` | 폐기확정 | C34 [M] '엔진이 대신 호출해 주입' 계보로 폐기 |
| `T2_PROCEED_DOCBODY` | 폐기확정 | t7304 심사 3인 일치 '유료 런으로 배관 사실을 사는 형태'(t7303 동형) |
| `T2_PROD_BIND` | 死배선복구 | 노브 ON·t7326 0발화(C군) |
| `T2_PROVENANCE` | 통합 | 구세대 PROV 축. 후계=PROV_REGEN |
| `T2_PROV_ADDR_FULL` | 死배선복구 | t43/96 주소 날조 통과가 표적. _FULL 접미사라 audit_unset 에 지워져 감사에서 안 보임 |
| `T2_PROV_BADWORDS` | 死배선복구 | placeholder 금칙어 축. 런처 0 |
| `T2_PROV_GROUND` | 死배선복구 | provenance 접지 축. 런처 0 |
| `T2_PROV_MODE` | 유지 | full\|rescue. rescue 스킵은 env-검증 id 만 |
| `T2_PROV_ORIGIN` | 死배선복구 | 최초 출처 날조 탐지. 런처 0 |
| `T2_PROV_REGEN` | 유지 | 라이브 pass 이동 실측 4종 중 하나. C53 [S] 456 sim·reward 0.580>floor 0.547(+3.3pp)·t17 4/4 db_fail->db_pass·over-block 0 |
| `T2_QUOTE_HINT` | 유지 | 모델이 제시한 값이 원장에 실재할 때만 지목(무조건이면 C226 spoonfeed) |
| `T2_QUOTE_PIN` | 유지 | [[66]] 이 의도분류 대체물로 지정한 인용-근거 축. C282 [S] 사슬 / [D] pass(n=1) |
| `T2_REF_ISO` | 死배선복구 | C124/C125 참조 슬립(039 wrong-pick=전사 슬립+자기 정박 부하). 런처 0 |
| `T2_REF_VERIFY` | 통합 | 결정론 참조-검증기(도구/필드/문구=A2). WEV 블록 합류 = wd 폴스루 6종 중 하나 |
| `T2_SCALAR_ARRAY` | 死배선복구 | 스칼라/배열 불일치 노트. 런처 0 |
| `T2_SEARCH_AGENT` | 유지 | [[67]] 검색은 t2_search 하나로 고정. C418/C432 [S] 프로덕션 2축xn=8 / [?] 라이브 |
| `T2_SEARCH_ON_PROCEED` | 통합 | P(SEARCH_AGENT\|SEARCH_ON_PROCEED)=1.00 (52/52) [S] = 독립 레버가 아니라 하위 분기 |
| `T2_SELF_DECLARATION` | 모순확정필요 | t2_levers NOT_LAUNCHED <-> go_stack.sh:341 export=1 [S]. 정반대 |
| `T2_SG_BYREF` | 유지 | 수리판. C531 [S] 수리 / '[D] pass 이동을 기대할 근거는 없다'. 구판은 거짓 deny 49건/11 sim(C526) |
| `T2_SG_GROUND` | 통합 | A2 ground 선언으로 서브 산출 operand 접지 |
| `T2_SG_ISOFB` | 유지 | r095g: 서브가 실패를 모른 채 종료 — checking 값-없는 인용 4-trial 반복 |
| `T2_SG_ISOLATE` | 유지 | 실패=None -> 메인 폴백(거동 변화 0) = 옳은 배치 계약 |
| `T2_SG_WINDOW_ABSTAIN` | 死배선복구 | 노브 ON·t7326 0발화(C군). 유효창 밖 기권 |
| `T2_SOURCE` | 유지 | 행동을 좌우하는 사실 주장은 출처를 대야 한다(C1). t7326 62발화/13 sim |
| `T2_SOURCE_QUALIFY` | 폐기확정 | 켠 arm 라이브 회귀 실측(102 db_match 2/2 -> 0/2·제출 1·1 -> 5·3). 근거가 코드 주석에만 있어 원장 이관 필요 |
| `T2_TRANSCRIBE` | 유지 | 엔진은 고치지 않고 어긋난 사실만 말한다(inject 자기일관 0.969@32B) |
| `T2_UNKNOWN_UNVERIFIED` | 死배선복구 | 미검증 항목 unknown 표기(N2 fixture). 런처 0 |
| `T2_VERDICT_CARRY` | 근거보강필요 | C515 [M] n=25 A 8 -> B 15(Δ+7)·McNemar p=.092·D_NEG 2 => 사전 고정 바 통과인데 t7326 PIN=0 = [[60]] 위반 실물 |
| `T2_VERDICT_GATE` | 실험노브 | VC 의 호출-트리거 판. run_one.sh 주석 'VC 와 VG 를 같이 켜는 팔은 없다 — 같은 판정을 두 번 사면 귀속이 섞인다' |
| `T2_WRITE_ARG_ENUM` | 통합 | **ARGDIFF 직격**: en_fb 는 rank 19(최후미)·wev_fb 는 rank 8 로 12칸 격차 [S]. 063 t44 '경합 2 -> resolve_write 승 · 밀림 write_enum' -> 결말 account_type: savings≠checking |
| `T2_WRITE_ARG_GROUND` | 통합 | 040 실측: ARG_SCHEMA 는 최상위 키만 봐서 못 잡는 자리. wd 폴스루 2번째 |
| `T2_WRITE_EVIDENCE` | 근거보강필요 | **gold 오차단 최대 기여자 22/56 [S]**·미회복 11 중 5건(040 x4·094 x1). wd 한 변수를 6종이 폴스루로 공유(:7049-7105) |
| `T2_WRITE_SUB` | 유지 | 라이브 pass 이동 실측 4종 중 하나. C472/C475 [M] 격리 8/8 <-> 근거 제거 0/8·부정통제 2종·비용 89->23. t7326 863발화/도달 172 |

### P5_SETGAP — 집합 차 — listed-examined / targets-submitted / given-ran / 대상별 분화(binding)  (17)

| 레버 | 상태 | 근거·주 |
|---|---|---|
| `T2_COV` | 통합 | COV_MIDDRIVE 구판. t7326 0발화 [S] |
| `T2_COVERAGE_FOLLOWUP` | 근거보강필요 | 엔진 자기생성 라인 재인용(판단 0). t7326 2발화 [S] — 다중요구가 결말의 52%인데 이 축 전체가 4발화 |
| `T2_COV_MIDDRIVE` | 근거보강필요 | t7326 0발화 [S]. C118 한계 실측 '보류 1회 후 통과 -> 막지 못했다' |
| `T2_DISCOVERY_REQUIRED` | 모순확정필요 | t2_levers NOT_LAUNCHED 등재·런처 0. analysis_producers 미호출 = 집합 차 |
| `T2_DISPATCH_LEDGER` | 근거보강필요 | 다건-write 대상 집합 등재(판단 0). t7326 2발화 [S] |
| `T2_EPLAN_EXAMINED_SAFE` | 통합 | 오차단 완화 조건 = 집합 술어 |
| `T2_EPLAN_REPLAN` | 死배선복구 | coverage_gap replan 격리 서브콜. 런처 0 |
| `T2_FOLLOWUP_REQUIRED` | 통합 | 사임 창 10종 중 하나. 억제 실측 8건이 지문창에 먹힘 [S] |
| `T2_GIVE_EXEC_NUDGE` | 통합 | GIVEN\RAN. C214/E2 019: 포털 불가라던 손님에게 실행 가능을 끝내 안 알림 |
| `T2_PROC_ABSENT` | 통합 | 절차 진입 턴 체크리스트 1회. 아무도 안 부른 도구의 첫 지목 위치 중앙값이 대화의 0.63 |
| `T2_READALL` | 死배선복구 | readall_unread(listed, examined) = P5 술어의 원형. t7326 0발화 |
| `T2_TERM_GRANT` | 유지 | 종료 허가 = 남은 절차 단계 유무(피로·횟수 아님) |
| `T2_TERM_GRANT_USERDEMAND` | 유지 | C212/A4 day7 008 회귀: 공식 notice 없이 ###TRANSFER### -> grant 0 -> 미호출 종료 |
| `T2_TRANSFER_LEAVES_STEPS` | 통합 | 이관이 남기는 미완 단계 1회 표면화 |
| `T2_UNVERIFIED_FOLLOWUP` | 통합 | C214/E1 day8 003: Silver fx_fee 가 premium 조건부라 unverified 인데 재실행 0 |
| `T2_VERDICT_SURFACE` | 통합 | 우리 판정이 실재 ∧ 결정 도구 미호출. 051 은 approve/deny 중 하나를 골라야 했고 둘 다 안 불렀다 |
| `T2_WITHDRAWN_ROW` | 死배선복구 | 노브 ON·t7326 0발화(C군). 철회된 행 표면화 |

### P6_SPEECH — 말 vs 행동 — 언급했으나 미호출·사임 등가·정체 턴수  (10)

| 레버 | 상태 | 근거·주 |
|---|---|---|
| `T2_ACTION_SUB` | 유지 | 트리거=P6, 서브콜=엔진 잔류. C406 24sim ON 9/12 <-> OFF 7/12 [S]·팔 오염검사 준-NC |
| `T2_ACT_DEMAND` | 폐기확정 | C492 [M] 배선 통과·성적 null·부작용 확정(촉구 231<->0·9/20<->12/20=잡음). 술어는 P6 에 남기고 발화는 안 함 |
| `T2_DECIDE_ANY` | 통합 | action_tools 를 누구든 밀고 있으면 결정 재료 생성. 엔진이 보는 것은 멤버십뿐 |
| `T2_DECISION_CARRY` | 통합 | 결정 이후 이월. t7326 192발화/37 sim |
| `T2_DECISION_ISOLATE` | 폐기확정 | C403 [S] 24sim '이득 없음, 그리고 배제 근거까지 빼는 것으로 보인다' |
| `T2_FOLLOWUP_READLOOP` | 통합 | post-submit read-루프를 사임 등가로 계상(rall7 050: 사임 이벤트 0) |
| `T2_FORCE_ACTION` | 근거보강필요 | t7326 214발화인데 C330 '말로는 따르고 호출은 0'·효과 [?]. 어느 도구인지는 모델 몫(디코딩 제약만) |
| `T2_HAVE_VALUE` | 통합 | 값을 쥐고도 행동 안 함 = 접지된 값 ∧ 미호출 |
| `T2_UNINSTRUCTABLE` | 통합 | 012: 실행 불가능한 지시 위에 존재하지 않는 도구명·앱 경로 날조(코퍼스 grep 0건) |
| `T2_VALUE_ACQUIRE` | 통합 | C119 [M] n=8 give 커밋 -> 유저 도구 실행 -> 5320 획득 |

### P7_CHANNEL — 하네스 — 봉투/절단/토큰/타임아웃/쌍 무결성. 도메인 무주장  (29)

| 레버 | 상태 | 근거·주 |
|---|---|---|
| `T2_AGENT_MAX_TOKENS` | 유지 | agent(vLLM) 전용 디코드 예산 |
| `T2_DYN_MT` | 유지 | vLLM 에러 원문 파싱(추정 0). day5 ctxover 7건 전부 36.5~40.2k 사망 |
| `T2_DYN_MT_MARGIN` | 유지 | 하네스 파라미터 |
| `T2_ENVELOPE_CAP` | 유지 | 봉투 regen 상한 |
| `T2_ENVELOPE_GUARD` | 계기재분류 | 셀 '조건 게이트' -> 하네스로 이관. 봉투는 우리가 요구한 출력 형식 = 채널 사실 |
| `T2_ENVELOPE_TAG` | 유지 | 서빙 포맷 상수 |
| `T2_ENVELOPE_TRUNC` | 유지 | regen 프롬프트 content 절단 |
| `T2_FAILED_PERSIST` | 유지 | stdout 출력이라 trace 마크 감사에 원리상 안 잡힘 [S] — 계측 채널 통일 대상 |
| `T2_FORCE_MIN_TOKENS` | 유지 | 강제 tool-call JSON 절단 방지(vLLM #19051/#36794) |
| `T2_GATE_REGEN` | 유지 | replay-safe 게이트 설치·unified 경로 라우팅 |
| `T2_GUIDED` | 유지 | 반드시 gate **이전** 적용(C166: 나중이면 regen 이 문법 우회·032 관통) |
| `T2_KB_NOHIT_SURFACE` | 死배선복구 | 노브 ON·t7326 0발화. Score: 정규식은 env 고정 포맷 전사 = 채널. startswith('KB_search') 는 A2 search_tools 로 |
| `T2_LLM_RETRIES` | 유지 | num_retries 주입 |
| `T2_LLM_TIMEOUT` | 유지 | 미설정=litellm 기본 ~40분 조용한 stall(097 실측) |
| `T2_MATCH_COUNT` | 근거보강필요 | 회수 경계 표면화. t7326 마크표에 없다 — 0발화인지 마크 미배선인지 못 갈랐다 [?] |
| `T2_MT_FLOOR` | 유지 | 플로어 미만=진짜 창 소진 -> graceful-stop |
| `T2_PAIRCHECK` | 유지 | call<->result 불변식 라이브 검사(로그 전용) |
| `T2_PAIRFIX` | 유지 | 순서-스왑 교정(의미론 no-op) |
| `T2_READ_DEDUP_MIN` | 유지 | dedup 대상 최소 출력 길이 |
| `T2_REGEN_BUDGET` | 유지 | over-action 사고(023 ctx 초과)를 막는 유일한 전역 층 |
| `T2_SALVAGE` | 死배선복구 | 오프라인 42건 중 복제형 38/38=100% 회수 [S]. C248 형이 실패 census 를 오염시킨다. 런처 0 |
| `T2_SG_SUB_TOOLCAP` | 유지 | rall11 097: 서브 52854 tok > 48640 max -> CWE -> 메인 오추측 95000 |
| `T2_TRUNC_GUARD` | 유지 | finish_reason=='length' 는 채널 사실이지 도메인 판정이 아니다 |
| `T2_VIEW_ANNOTATE` | 유지 | 생성-뷰에만 부가(비커밋) |
| `T2_VIEW_COMPACT` | 유지 | 생성-뷰만 압축·커밋 히스토리 원문 유지 = replay-safe |
| `T2_VIEW_COMPACT_KEEP` | 유지 | 하네스 파라미터 |
| `T2_VIEW_COMPACT_MINLEN` | 유지 | 하네스 파라미터 |
| `T2_VIEW_COMPACT_MINTOTAL` | 유지 | 구 120,000 은 사망선 위라 day5 에서 6/32 만 발동 |
| `T2_VIEW_MSG_CAP` | 유지 | per-메시지 상한 |

### P8_GOV — 발화 거버너 — 병합 출구(명령 하나·사실 합집합) + 인자변화 재무장  (57)

| 레버 | 상태 | 근거·주 |
|---|---|---|
| `T2_ACTION_DENY_CAP` | 통합 | 횟수 예산 -> 인자변화 재무장. 101/102 부검: 발화 3회가 turn 4·6·8 소진, 첫 요건 충족 turn 11 |
| `T2_ACTION_PROGRESS_REFUND` | 통합 | 환급 개념은 재무장 술어로 흡수(집합 교집합 -> 인자변화) |
| `T2_ARBITRATE` | 유지 | C3 합병기 = P8 의 숙주. _SRC8 17칸을 전부 받아 한 번에 낸다 |
| `T2_AXIS_NOTE_CAP` | 통합 | 원 근거가 P8 정당화 자체: '서사가 반복되면 안내도 매번 붙어 026 에서 55회 = 레버가 팽창을 만든다' |
| `T2_CLAIMPROV_CAP` | 통합 | 완료주장 발화 예산 -> 재무장 술어 |
| `T2_COV_MIDDRIVE_K` | 통합 | 무진행 중단 임계 -> 재무장 술어 |
| `T2_DD_CAP` | 통합 | discovery-dispatch deny 예산 |
| `T2_DEFERRED_VIEW_KEEP` | 통합 | 뷰 유지 턴수 -> 재무장 파라미터 |
| `T2_DISPATCH_ROLE_CAP` | 통합 | deny 상한 |
| `T2_EPLAN_DENY_CAP` | 통합 | C173-corr: pre-close deny 와 공유 예산을 나눠 써 044 소진 => 예산 공유가 병 |
| `T2_EPLAN_DRIVE_K` | 통합 | progress-guard 가 곧 재무장 술어의 원형 |
| `T2_FB_VIEW_K` | 통합 | 피드백 뷰 유지 턴수 |
| `T2_FOLLOWUP_CAP` | 모순확정필요 | registry PARAMS·go_stack·주석 세 곳에 있는데 environ.get 자리를 코드에서 찾지 못했다 [S] |
| `T2_FOLLOWUP_FORCE` | 통합 | tool_choice=required = 채널 옵션 |
| `T2_FOLLOWUP_PROGRESS_REFUND` | 통합 | cap3 < 사슬 6단계의 땜질 => 재무장 술어로 대체 |
| `T2_FOLLOWUP_RESIGN_TH` | 통합 | 사임 몇 회째부터. 오프라인 replay: 2회째가 실패 4/4 커버 |
| `T2_GATE_REGEN_K` | 통합 | 구 C173 의 K=2 는 unified 경로서 no-op 판명·철회 |
| `T2_HANDOFF_CAP` | 통합 | 잔소리 루프 억제 |
| `T2_HAVE_VALUE_CAP` | 통합 | 재질의 상한 |
| `T2_HAVE_VALUE_FORCE` | 통합 | 코드 기본 ON·go_stack 도 1. DEFAULT_ON 미등재 사각 |
| `T2_KB_NOHIT_K` | 통합 | 연속 전-0점 임계 |
| `T2_KEEP_DENY_BODY` | 유지 | [[64]] 불변식. 이름 없는 '먼저 해소하라' 가 3회↑ 나온 6 sim = 6/6 실패 [S] |
| `T2_LEDGER_VIEW_KEEP` | 통합 | 원장 행 뷰 유지 턴수 |
| `T2_MAIN_ANSWERS_ONLY` | 死배선복구 | [[65]] 정본 원리의 유일한 코드 구현체(x231 8/8->0/8 계보) 인데 어느 런처에도 없다 [S]. P8 의 기본 거동으로 승격 |
| `T2_PARAM_CAP_CAP` | 통합 | deny 상한 |
| `T2_PRECLOSE_CAP` | 통합 | C173-corr 로 분리했던 예비 예산 => 재무장 술어로 통합 |
| `T2_PRESCRIPTION_CAP` | 통합 | 발화 상한 |
| `T2_PROCEDURE_CAP` | 통합 | 절차 deny 상한 |
| `T2_PROC_ABSENT_CAP` | 모순확정필요 | registry PARAMS 에만 있고 코드 read 자리 0 [S] |
| `T2_PROC_ABSENT_K` | 통합 | 발화 임계 |
| `T2_PROC_PIN_REARM` | 근거보강필요 | 원 근거('deny 해도 이행 안 한다')가 F19 로 소멸 — 그 deny 는 모델에게 간 적이 없었다. 재무장 정책 재측정 대상 |
| `T2_PROV_REGEN_K` | 통합 | PROV 예산 |
| `T2_READALL_CAP` | 통합 | deny 상한 |
| `T2_READ_DEDUP_LOOP_K` | 통합 | 루프 판정 임계 |
| `T2_READ_NEARDUP_J` | 통합 | 임계 |
| `T2_REF_ISO_CAP` | 통합 | 서브콜 예산 |
| `T2_REPEAT_CAP` | 모순확정필요 | t2_levers.py:489 RETIRED([[57]] 위반 폐기) <-> go_stack.sh:248 export=8 [S]. 정반대 |
| `T2_REPEAT_GOV` | 死배선복구 | 반복 채널 거버너 = P8 의 기존 구현. 런처 0(레거시 경로 사용 중) |
| `T2_RESOLVE_CAP` | 폐기확정 | C540 [S] 'stop=resolve_cap 098 2->6·100 0->6·073 12->25 … 계약 경로가 3~7배로 죽고 그 자리에 캡 정지가 들어찼다' + C537 래치 뒤 3,082 메시지. 횟수 캡 -> 인자변화 술어로 교체 |
| `T2_RETRY_K` | 통합 | 연속 실패 K회 가드 |
| `T2_SEARCH_EXHAUST_TH` | 통합 | 임계 |
| `T2_SUPPRESS_AUTH` | 유지 | 억제하려는 레버가 근거를 대야 한다(C13: 반복-억제가 gold 를 만들던 반복을 침묵시킴) |
| `T2_SURFACE_BUS` | 유지 | 부착의 단일 출구·불변식 4종. 주석 축자 'OFF 가 더 위험했다' |
| `T2_TOOLLIST_CAP` | 통합 | 발화 예산 |
| `T2_TOOL_SIGNATURE_CAP` | 통합 | 발화 예산 |
| `T2_TRANSCRIBE_CAP` | 통합 | 발화 예산 |
| `T2_UNKNOWN_NAME_BL_CAP` | 통합 | 발화 예산 |
| `T2_UNKNOWN_REPEAT_GUARD` | 유지 | '무엇이 바뀌었는가'를 보지 '몇 번 했는가'를 보지 않음 = P8 재무장 술어의 기존 구현 |
| `T2_UNLOCK_NAME_CAP` | 통합 | 발화 예산 |
| `T2_VALUE_ACQUIRE_CAP` | 통합 | 발화 예산 |
| `T2_VERDICT_GATE_CAP` | 통합 | sim당 상한 |
| `T2_VERIFY_DENY_CAP` | 통합 | 검증 지속 예산 |
| `T2_WEV_CAP` | 통합 | write-evidence deny 상한 = wd 폴스루 6종의 공유 예산 |
| `T2_WEV_ROUNDS` | 통합 | 라운드 예산 |
| `T2_WINDOW` | 계기재분류 | C434 축자 '인쇄이지 레버가 아니다 => 감사 대상 46 -> 8'. 창 판정은 P8 로, 인쇄는 분모에서 제외 |
| `T2_WRITE_ARG_ENUM_CAP` | 통합 | 발화 예산 |
| `T2_WRITE_CAP_K` | 통합 | 동일-write 허용 성공 횟수 |

### P9_REQUIRE — 요건과 다음 한 수 — t2_dominance.requirements_for / merged_text 재사용  (18)

| 레버 | 상태 | 근거·주 |
|---|---|---|
| `T2_BRANCH_REGROUND` | 死배선복구 | 노브 ON 인데 t7326 0발화(C군 11종). pre-close 후 재접지 = 요건 재제시 |
| `T2_DECIDE_BEFORE_WRITE` | 근거보강필요 | 앞 14종이 전부 None 일 때만 평가 -> t7326 3발화. C439 축자 'ON/OFF 는 이 레버에 무효' [S] = 배타 체인 병의 순수 표본 |
| `T2_DISCOVERY_DISPATCH` | 死배선복구 | ep_spec.dispatch_tool 기반 발견 사슬. 런처 0 |
| `T2_EPLAN` | 유지 | 원장의 원형. requirements_for 와 같은 선언(gates/require_tool_before/requires_reads)에서 읽는다 |
| `T2_EPLAN_READS_ONLY` | 통합 | A1-v2 실패분석: 무조건 행동금지가 coverage 누락 8건 유발 => in-scope sibling 허용을 계약에 |
| `T2_EPLAN_WALK` | 유지 | 사슬 역행. 터미널 훅 진입 조건 |
| `T2_PHASE_OWNER` | 死배선복구 | 노브 ON·t7326 0발화 = 이중 무근거(원장 근거 0 + 발화 0) |
| `T2_PIN_READ` | 통합 | require_tool_before 가 강제 못 하던 자리(C210 3건·gold 관문 요구 79 sim 중 53 미호출). 효과 [?] |
| `T2_PIN_READ_STEPS` | 통합 | 절차 단계별 읽기 핀. t7326 9발화 — 실패 60%가 READ_MISS 인데 sim당 0.7회 |
| `T2_PREKB` | 통합 | C165: 문제-기반 쿼리는 절차 문서를 못 찾고 행동-기반 쿼리는 1~2위. 문서를 찾게만 하고 답은 안 준다 |
| `T2_PRESCRIPTION` | 통합 | A2 처방 표면화(다른 피드백 전부 없을 때만) = 배타 체인 종속 |
| `T2_PROCEDURE` | 통합 | A2 절차 선언 구동. t7326 4 deny/017 전량 |
| `T2_REQUIRE_DOC` | 통합 | 문서 미열람 종결 행동에 1회. t7326 10발화/10 sim |
| `T2_SCAFFOLD_GET` | 유지 | A2 scaffold_get_tools 주입 전체 스위치 |
| `T2_SG_REQREADS` | 근거보강필요 | t7326 1발화 [S]. 실패의 60%가 READ_MISS 인데 이 배선이 sim당 0.025회 |
| `T2_SPEAK_PROHIBIT` | 통합 | 금지 도구를 추천하는 것까지 막음. 022 가 같은 턴에 추천과 금지를 3회 동시 수신 = P8 병합 부재의 증상 |
| `T2_SUB_REQUIREMENT` | 폐기확정 | C508 [M] '기전은 반증됐고 서브 정확도는 오히려 떨어졌다(gold 4->0)'·지연 2.4x => 스택 승격 금지 |
| `T2_TRANSFER_PREREQ` | 死배선복구 | 194 sim 중 이관 전 KB_search 0회 = 9건/9 sim [S]. 런처 0 |

### P10_ASK — 권위 이전 — 후보 카디널리티 != 1 이면 손님에게 묻는다  (3)

| 레버 | 상태 | 근거·주 |
|---|---|---|
| `T2_ASK_UNKNOWN_BOOL` | 死배선복구 | 194 sim 실측: 불리언 136건 중 106 false/None·35건은 대화에 주제조차 없음 [S]. 어느 런처에도 없음 |
| `T2_DISAMB` | 死배선복구 | C60 [M] 456/456 p2 +1.5·p3 +2.6·p4 +1.8pp = 원장에 남은 유일한 pass^k 개선 실측인데 어느 런처에도 없다 [S] |
| `T2_DISAMB_MODE` | 통합 | dialog\|subcall 되묻기 배치. _MODE 접미사라 audit_unset 필터에 지워짐 |

### D_DECL — 선언면(A2/A3) — 정책 조건·절차 순서·문구·정책산문 enum  (14)

| 레버 | 상태 | 근거·주 |
|---|---|---|
| `T2_A2_VARIANT` | 선언이설 | 도구 슬롯 변이=선언면. t7326 발화 1위 1,668회인데 원장에 효과 항목 0 [S]. 효과는 선언 A/B 로만 잰다 |
| `T2_DECLFIRST` | 실험노브 | 선언-우선 2패스 = 계약 통지층. 집행은 P9/P4 가 하므로 통지 전용으로 고정 |
| `T2_DECLFIRST_GUIDE` | 실험노브 | 코드 기본 ON 인데 go_stack 미선언 = DEFAULT_ON 미등재 사각 [S] |
| `T2_DECLFIRST_GUIDE_FIX` | 실험노브 | 가이드 문면 수리판. PIN=0 |
| `T2_DISAMB_ORDER` | 선언이설 | 엔진에 order/order_id 리터럴(:4524) = 도메인 리터럴. A2 disamb_args 로 |
| `T2_GROUNDING_SPEC` | 선언이설 | 접지 spec 경로. _SPEC 접미사라 audit_unset 필터에 지워짐 |
| `T2_GROUND_HDR` | 선언이설 | D3 헤더-상세 모순(x35② ledger 38:20 <-> user 7:43). 2026-08-18 신규 선언 |
| `T2_KB_DOCS_DIR` | 선언이설 | 런처 값이 도메인 고정(banking_knowledge/documents). 전이 시 이 한 줄만 간다 = 옳은 모양 |
| `T2_NOTICE_REPEAT` | 선언이설 | 코드 기본 ON 인데 go_stack 미선언 = 조용히 라이브. notice 문구는 A2 |
| `T2_PARAM_CAP` | 선언이설 | 엔진에 account_id/card_type/credit_limit 폴백 리터럴(:1141/1149/1155) — 폴백 제거·미선언=침묵 |
| `T2_PRINCIPLE_DEFAULT` | 폐기확정 | LOCK anti-drift 4 'default 개념 금지 — 정책-강제(gate) or ASK 로만'. _DEFAULT 접미사라 감사에서 영구 비가시 |
| `T2_RETURN_EMPTY` | 선언이설 | D4 빈-결과 문면. 2026-08-18 신규 선언 |
| `T2_TERMINAL_TURN` | 선언이설 | 엔진에 'transfer' 부분문자열 판정 -> A2 transfer_tokens |
| `T2_TRANSFER_TIER` | 선언이설 | 엔진 판정은 닫힌 둘(마커 실재·티어 정수 비교)·티어 표는 A2. C520 '도달 != 효과' |

### INSTR — 계기(인쇄·거동 불변) — 레버 분모에서 제외(C434)  (16)

| 레버 | 상태 | 근거·주 |
|---|---|---|
| `T2_DD_FB` | 모순확정필요 | t2_levers.py:489 RETIRED 등재인데 코드·런처 어디서도 안 읽음 = 존재 확정 필요 |
| `T2_DECLFIRST_ENFORCE` | 계기재분류 | 1차 마일스톤이 검출 전용(=0)·deny 안 함 = 계기 |
| `T2_FAILED_DIR` | 계기재분류 | 궤적 덤프 경로 |
| `T2_FB_SIDECAR` | 死배선복구 | **인벤토리 오류 확정**: '어느 런처에도 없음' 이었으나 t7326 라이브 2,153행·24채널 실재 [S]. 도달을 재는 유일한 계기 — 삭제 금지 |
| `T2_FB_SIDECAR_TEXT` | 死배선복구 | 본문 기록. 위와 동일 |
| `T2_GROUND_LOG` | 계기재분류 | 접지 판정 로그 |
| `T2_GUIDED_VERBOSE` | 계기재분류 | guided decoding 로그 |
| `T2_OVERFLOW_GUARD` | 모순확정필요 | go_stack.sh:21 export 인데 코드 전체에 environ.get 자리 0 [S] = 순수 노이즈. read 자리 신설 or export 삭제 |
| `T2_ROUTE_TRACE` | 계기재분류 | 코드 기본 ON·go_stack 미선언. 배타성 계수를 만든 계기 — 유지하되 분모에서 제외 |
| `T2_SG_ISOLATE_TRACE` | 계기재분류 | 격리 디스패치 trace |
| `T2_SG_TRACE` | 계기재분류 | 침묵-스킵 불가능하게 하는 계기(r095e) |
| `T2_STACK_OBSERVE` | 계기재분류 | [[48]] 7층 관측. 코드 기본 ON·go_stack 미선언 |
| `T2_STACK_WINDOW` | 계기재분류 | 관측 창 |
| `T2_TOOL_SIGNATURE_OBSERVE` | 계기재분류 | OFF 여도 술어 평가·로그만 = 상쇄-arm 모집단 실측용 |
| `T2_TRACE` | 계기재분류 | 레버 비트 trace |
| `T2_TRACE_LINECAP` | 계기재분류 | trace 줄 길이 상한 |

### ARM — 측정 노브 — 귀속 실험 전용. 레버 아님  (8)

| 레버 | 상태 | 근거·주 |
|---|---|---|
| `T2_EPLAN_WALK_HOLD` | 실험노브 | 001 [S] 로 기본 OFF 강등·go_stack:217 주석 처리 = ARM_ONLY 명시 |
| `T2_GATE_KINDS` | 실험노브 | gate kind 화이트리스트 = 측정-격리 |
| `T2_L4_MODE` | 실험노브 | keep<->substitute. _MODE 접미사라 감사에서 지워짐 — 술어 기준으로 갈라야 함 |
| `T2_MAXPROMPT` | 실험노브 | [[42]] prompt-ceiling 실험 전용 |
| `T2_MAXPROMPT_POS` | 실험노브 | 프롬프트 위치 실험 |
| `T2_OFF_CELLS` | 실험노브 | [[60]] 비상구. t7326 미설정=전 셀 ON. 쓰면 태그 기록 의무 |
| `T2_RULES_PROMPT` | 실험노브 | 규칙 프롬프트 파일 경로 |
| `T2_SG_EXCLUDE` | 실험노브 | 단일 변수 대조(대안 도구 유/무) |

---

## 5. A2/A3 선언 스키마 초안

### 5-0. 원칙 — **선언은 줄고, 남은 것은 검증 가능해진다**

| 지금 선언으로 사는 것 | 이후 | 근거 |
|---|---|---|
| `write_tools` / confirm-gate `applies_to`(=write 집합) | **삭제** | env `__tool_type__` (P1) |
| `dispatcher_role_check.{unlock_tool, name_args}` · `eplan.{dispatch_tool, list_tool}` | **삭제** | 스키마 형상 도출 `_dispatch_tools` (P3) |
| `discoverable_name_check.tools` · `user_tool_channel_args` | **삭제** | 레지스트리 3집합 `_agent_discoverable` / `_user_discoverable` / `_user_all_tools` (P3) |
| `tool_signatures` 인자 구성 | **삭제** | env 스키마 `properties` (P3) |
| 엔진 리터럴 8곳 + `_PROCEDURAL_RE` 5접두사 | **코드에서 삭제** | FATAL-1 · FATAL-4 |

⇒ §2-g 가 센 "출처 표기 없는 선언 25건" 중 `discoverable_name_check`(2) · `dispatcher_role_check.name_args`(1) · `tool_signatures`(1) · `relations.by_tool`(1) = **5건이 감사 대상에서 아예 빠진다**.

### 5-1. 출처 강제 (모든 선언 항목의 필수 형식)

`policy_ontology.rows` 153행이 이미 쓰고 있고 원격 축자 검증 **153/153 통과**한 형식을 **전 선언으로 통일**한다 [S].

```jsonc
"source": {
  "doc":   "doc_checking_accounts_gold_years_account_002",   // 문서 id (파일 basename·글롭 금지)
  "quote": "Foreign ATM withdrawal fee: $3.50 dollars",       // 축자 인용
  "quote_match": "exact"                                      // exact | normalized  (normalized 는 사유 필수)
}
```

- ⛔**금지**: 현행 `scaffold_get_tools[7]/[9].op.table` 의 산문 글롭 `"doc platinum_rewards_card_*"` (23행). 값 자체는 코퍼스에 실재하나 **형식이 검증을 막는다** — 다음 편집자가 gold 로 채워도 아무도 못 잡는다 [S].
- ⛔**금지**: A2 필드를 A2 필드의 출처로 대는 자기참조(`scaffold_get_tools[6]._note_requires_reads` 가 "이 도구의 description 축자"를 출처로 든다). [[23]] 이 막으려던 순환의 형태.
- **검정 상설화**: `verify_a2.py` 형(원격 코퍼스 축자 대조)을 래칫에 넣는다. 실패 시 CI red.

### 5-2. 신규·필수 키 (전이 최소 생존선 2줄 포함)

| 키 | 소속 | 왜 | 미선언 시 |
|---|---|---|---|
| **`failure_markers`** | 도메인 최상위 | env 가 실패를 `error=False` + 본문으로 돌려준다(축자: `NOT_VERIFIED — only 1 of the required 2 values … match` · `Failed to log verification: Record may already exist.`). 미선언이면 **실패한 호출이 '실행됨'으로 계상되어 P2 원장 전체가 조용히 거짓이 된다** | ⛔**전이 금지** — 이 고장은 예외도 로그도 안 남긴다 |
| **`entity_key` / `items_key`** | 도메인 최상위 | P5 의 집합을 만드는 열쇠. 없으면 `_idlike`(len≥2 ∧ 숫자 포함)만 남아 금액·날짜·전화번호가 record id 로 섞인다 | P5 과발화 |
| `tool_roles` | 도메인 | `read/write/procedural/dispatch/give/unlock/call/transfer`. **P1 이 덮지 못하는 잔여만** | 해당 술어 침묵 |
| `procedural_prefixes` | 도메인 | `_PROCEDURAL_RE` 5개 이설. `transfer_to_human` 만 엔진 잔류(5/5 공통·자체 주석 근거 보유) | 접두사 판정 없음 |
| `search_tools` / `search_output.score_line` | 도메인 | `.startswith("KB_search")` 3곳(`:976` · `t2_prekb_patch.py:184` · `t2_transfer_prereq.py:22`) 과 `_KB_SCORE_RE`(`:509`) 대체 | 검색 인식 없음 |
| `op.constraints[{ctx_key,row_key,sense}]` · `segment_key` · `gate_key` · `identity_keys` | 도구별 | `catalog_filter` 42 리터럴 승격(FATAL-2). 표 스키마는 이미 A2 데이터 | 카드 자격 판정 침묵 |
| `arg_contract[leaf]{source_required, enum, axis, cap{field,ratio}, window, producer}` | write 도구별 | 엔진 폴백(`account_id`·`card_type`·`credit_limit` `:1141/1149/1155` · `merchant_name` `:1313` · `transaction_id`/`("date","amount")` `t2_resolve.py:919/992/1011/1048`) 제거 | **그 필드 침묵**(폴백 금지) |
| `resolve.infer` | operand별 | **기본 `none`**. 미선언 = 실행 안 함 = ASK 낙하 | ASK |
| `authority` / `on_ambiguous` | operand별 | P10 ASK 의 권위자·나열 여부(banking=비나열·프라이버시) | ASK 침묵 |
| `transfer_tokens` | 도메인 | 엔진 `"transfer"` 부분문자열 판정(`T2_TERMINAL_TURN`) 이설 | 터미널 인식 없음 |
| `contract_feedback{blocked_name, what_is_wrong, next_step}` | 도메인-일반 | [[64]] 문면을 A2 로. 엔진은 이름만 채운다 | 기본 문면 |

**폴백 규범 (이미 코드에 존재)** — `t2_resolve.py:990-991` 축자: *"미선언이면 dispatcher 개념이 없는 도메인이므로 이 레버를 **끈다**(안전측·B3 교훈)"*. 이것을 전 인자 계약으로 확대한다. 얇은 선언 도메인에서는 통과만 시킨다 = **성능은 낮아지되 오차단은 늘지 않는다**.

### 5-3. 기존 `t2_procedure` 선언과의 차이

| 축 | 현행 `procedures` | 이후 |
|---|---|---|
| 요건 표현 | `nodes[].requires` / `prohibits` / `applies_to` 를 **절차 전용** 스키마로 | 그대로 **유지**. 다만 P9 가 `gates[]` · `require_tool_before` · `requires_reads` · `eplan` 과 **같은 반환형**(`[{id,predicate,satisfiers[1]}]`)으로 합류시킨다 |
| 출구 | `proc_fb` 가 `_SRC8` 색인 7(rank 9) 에서 단독 발화 | P8 병합에 합류 — `022` 가 같은 턴에 `[VALUE-ACQUIRE]` 추천과 `[PROCEDURE]` 금지를 **3회 동시 수신**한 자기모순이 사라진다 |
| 예산 | `T2_PROCEDURE_CAP=6` 등 자기 캡 | P8 재무장 술어(인자 변화)로 흡수 |
| 출처 | `_note_requires` 일부가 **gold 경유**(x91 "gold 2태스크를 막았다"·`_note_choice_grounding` "gold 3건 ⇒ 넛지") | §9-B 로 이관 — 사용자 판정 필요 |
| 도메인 리터럴 | `self._t2_proc_pin = ("call_discoverable_agent_tool", ...)` (`:6670`) | `eplan.dispatch_tool` 또는 P3 형상 도출 |

---

## 6. 코드 정리 계획

⛔전제: **F1(도메인 리터럴 8곳) 이설이 통합보다 먼저다.** 순서를 바꾸면 통합 후 전이 arm 에서 무엇이 죽었는지 귀속 불가.
⛔각 단계는 **관찰 모드 1런 → 회귀 확인 → 승격**. 승격 조건은 발화가 아니라 **그 술어가 여전히 *평가*되는가**([[60]] 방어).

| # | 단계 | 위험도 | 위험의 구체 | 되돌리기 |
|---|---|---|---|---|
| **S0** | `x6h_engine_literal_audit.selftest_range()` 를 `def selftest/_selftest/test_*` 함수 범위까지 확장 | **낮음** | 감사 전용·라이브 무영향 | 파일 1개 revert |
| **S1** | 인벤토리 확정: 엔진 269 ↔ registry 139 ↔ go_stack 105 ↔ PIN 27 의 4자 대조표를 `flag_registry_baseline.json` 에 고정 | **낮음** | 없음(문서) | baseline revert |
| **S2** | **모순 7건 확정**: `OVERFLOW_GUARD`(read 자리 0) · `ARG_SCHEMA`(死코드) · `REPEAT_CAP`(RETIRED↔export=8) · `SELF_DECLARATION`(NOT_LAUNCHED↔export=1) · `FOLLOWUP_CAP` · `PROC_ABSENT_CAP` · `DD_FB` | **낮음** | 결정만·코드 변경 최소 | 각 1줄 |
| **S3** | **F1 이설**: give/call/unlock 8곳 → `_a2_procedural(a2)`(`:2367`) · `eplan.*_tool` · P3 형상 도출. `t2_axis_levers.channel_note` 는 시그니처에 `a2` 추가(선례 `_a2_of(obj)` `:2374`) | **중** | 5레버(t7326 ≥51발화)의 술어가 바뀐다. 오이설 시 조용히 침묵 | 함수별 커밋 분리 + 오프라인 replay 로 발화 집합 차 0 확인 |
| **S4** | **F4 이설**: `_PROCEDURAL_RE` 5접두사 → `a2.procedural_prefixes`. `transfer_to_human` 만 잔류 | **중** | `_is_effective_write` 가 바뀌면 WEV·PROV·CLAIM·STALE_STRIP 이 동시에 움직인다 | 접두사 리스트를 A2 에 **현행과 동일하게** 먼저 넣어 바이트 등가 확인 후 이설 |
| **S5** | **P1 KIND 전환**: `_is_effective_write` 를 `__tool_type__` 기반으로 | **⛔높음** | `t2_gate_patch.py:5315` 주석 축자 *"`call_discoverable_agent_tool` 은 `@is_tool(ToolType.WRITE)` 로"* — **env 는 디스패처를 WRITE 로 선언하는데 현행 우리 판정은 procedural(=write 아님)이다. 정반대.** banking 은 거의 모든 호출이 디스패처 경유라 write 집합이 대량 True 로 뒤집히고 WEV·PROV·CLAIM·confirm-gate 가 **동시 발화 폭발**할 수 있다 | **오프라인 replay 로 두 판정의 차집합을 세기 전에는 켜지 않는다.** 전환은 플래그가 아니라 별도 arm |
| **S6** | **F2 이설**: `catalog_filter` 42 리터럴 → `op.constraints` 튜플 | **중** | 카드 자격 판정 전체가 이 함수에 있다. 튜플 누락 = 자격 오판 | 이설 전후 동일 입력 13행 × 10행 표로 출력 바이트 등가 회귀 |
| **S7** | **F3 폐기**: `fit_diff_note` 산문 정규식 3줄 삭제, `catalog_filter` 가 구조화 dict 를 직접 반환 | **낮음** | `T2_FIT_DIFF` 는 현재 OFF·0발화 | 삭제 revert |
| **S8** | **P8 병합 출구**: `_SRC8` 17칸 `elif` → `t2_arbitrate` 로 전량 수집 후 `t2_dominance.merged_text` 로 1문면 | **⛔높음** | 병합 자체가 [[65]] 부하가 된다(x231 한 줄로 8/8→0/8). 그리고 **병합을 '선택'으로 구현하면 이름만 바꾼 동일 병** | ①`T2_ROUTE_TRACE` 로 밀린 문장 수를 상시 계수 ②Δ메시지·Δ턴 동반 계측 ③레거시 경로를 arm 으로 남긴다 |
| **S9** | **P8 예산 통일**: `_CAP/_K/_TH` 32종 → 인자변화 재무장 1개 | **중** | 종료 조건 설계가 이 안에서 **가장 덜 풀린 자리**. 무한 루프 ↔ C540 회귀 사이 | `hard_stop` 을 null 이 아닌 큰 값으로 두고 도달 횟수를 계측 |
| **S10** | **모놀리스 분리**: `t2_gate_patch.py` 11,415줄 → 원소별 모듈(`t2_kind` / `t2_event` / `t2_name` / `t2_ground` / `t2_setgap` / `t2_speech` / `t2_gov`). **의미 변경 0의 순수 이동만** | **중** | 이동 중 조건절 하나가 흡수되면 그 레버가 조용히 죽는다(실증: `PENDING_DISCOVERED` 가 `_ts` 지역 별칭 하나로 처음부터 죽어 있었다·C539) | 파일당 커밋 1개 + 이동 전후 `git diff --stat` 이 순수 이동인지 확인 + 래칫(아래 S11) |
| **S11** | **래칫 신설**: `test_lever_element_map.py` — 매핑 JSON 의 274 레버 각각에 대해 ①`environ.get` 자리 존재 ②원소 배정 존재 ③`死배선복구`/`모순확정필요` 는 해소 기한 필드 보유. 하나라도 비면 CI red | **낮음** | 없음 | 테스트 revert |
| **S12** | **[[67]] 정본 강화**: `x*.py` 410 중 정본 5종(`t2_subcall`·`t2_forensic`·`t2_liveness`·`t2_search`·`t2_levers`)과 기능이 겹치는 것을 목록화하고, 신규 프로브는 정본에 **추가**만 | **낮음** | 기존 프로브 삭제는 하지 않는다(재현성) | 목록만 |
| **S13** | **런처 통합**: `run_*.sh` 122 → `go_stack.sh`(기본 스택) + `run_one.sh`(arm) 둘. PIN ON 15종을 go_stack 으로 승격하거나 go_stack 이 "정본" 자칭을 철회 | **중** | 지금 상태로는 **어떤 A/B 도 무엇과 비교했는지 모른다** | 기존 런처는 `archive/` 로 이동(삭제 금지) |

**되돌리기 공통 규약**: 각 단계는 **단일 커밋 · 단일 관심사**. 라이브 런 사이에 두 단계를 겹치지 않는다. 되돌림은 `git revert <sha>` 하나로 가능해야 하고, 그것이 불가능한 단계는 착수 금지.

---

## 7. 실행(런) 체계 재설정

### 7-1. 기본 스택 (`go_stack.sh` 승격판)

- **전 레버 ON 이 기본**([[60]]). `T2_OFF_CELLS` 는 귀속 실험에서만, 쓰면 **태그에 기록 의무**.
- PIN ON 15종(`ACTION_SUB`·`KEEP_DENY_BODY`·`CALL_FORM`·`ARG_EMPTY`·`SEARCH_AGENT`·`DECIDE_ANY`·`WRITE_ARG_ENUM`·`DECIDE_BEFORE_WRITE`·`DECISION_CARRY`·`DISCOVERY_STEP2`·`ARG_AXIS`·`ACTION_INDEX`·`NOW_SELFCALL`·`SEARCH_ON_PROCEED`·`WRITE_SUB=3`)을 **go_stack 으로 승격**한다. 그때까지 t7326 계열과 go_stack 계열은 **비교 불가**로 표기 [S].
- **음성 실측 보유 레버는 기본 침묵 + 원장 근거 인용 의무**: `TOOL_SIGNATURE`(C267 OFF 권고) · `SOURCE_QUALIFY`(102 회귀) · `L4`(2/2 오답) · `AUTOFETCH`(C34) · `HANDOFF_PREDICATE`(C529) · `DELIVER_PRECOMMIT`(C502) · `DOCS_AT_WRITE`(C505) · `MATERIAL_RESERVE`(C499) · `DECISION_ISOLATE`(C403) · `SUB_REQUIREMENT`(C508) · `PROCEED_DOCBODY` · `PRESENT_READS` · `PRINCIPLE_DEFAULT` · `OPERATOR_PINPOINT`(x322) · `UNLOCK_QUIET` · `FIT_DIFF`([[59]]) · `RESOLVE_CAP`(C540).
- **사이드카는 항상 켠다** — `T2_FB_SIDECAR` + `_TEXT`. 도달을 재는 유일한 계기이고, 이번에 이것이 꺼진 줄 알고 세 보고서가 "못 잰다"로 결론했다 [S].

### 7-2. 로스터 단계

| 단계 | 로스터 | 목적 | nt |
|---|---|---|---|
| **L0 배선** | 3 태스크 × nt1 (스모크) | `t2_liveness` 로 원소 8+2 생존 확인([[55]] 0단계). **pass 를 보지 않는다** | 1 |
| **L1 귀속** | 20 태스크 × nt2 = 40 sim (t7326 로스터 고정) | 사전등록 A/B 1축 | 2 |
| **L2 확인** | 실효 95 (005·102 제외·[[68]]) | L1 에서 바를 통과한 것만 | 2~4 |

- **분모 규약**: L1 은 **40 sim 고정**. `ends` 행이 비는 sim(예: `task_079#t1`)도 **실패로 계상**한다 — 이번처럼 최악 sim 이 표에서 증발하는 일을 금지한다 [S].
- **비교 금지**: 20 태스크 로스터 결과를 실효 95 나 리더보드 수치와 나란히 놓지 않는다([[54]]).

### 7-3. 태그 규칙 (런 하나 = 태그 하나)

```
t<NNNN>_<도메인>_<로스터>_<nt>_<arm>_<YYYYMMDD><seq>
  arm ∈ {base, treat_<원소>, dneg_<원소>, obs}
  base   = 승격된 go_stack 그대로
  treat  = base + 그 원소 하나만 바뀜
  dneg   = 부정통제([[57]]) — 같은 배선·무내용 문면
  obs    = 관찰 모드(술어 평가·발화 0)
```
- 태그에 **`T2_OFF_CELLS` 사용 여부와 값**을 반드시 넣는다.
- 로그 회수 체크리스트(규율 신설): **trace + sidecar + log + results + meta 4종을 항상 함께 내린다.** 이번에 사이드카를 안 내려 세 보고서가 도달을 못 쟀다 [S].

### 7-4. ⛔지금 중단할 것

1. **단일 스택 전 레버 ON 런으로 "효과"를 논하는 것.** t7326 형은 계수·포렌식용으로만 돌린다. 기여 귀속은 원리상 0건이다 [S].
2. **`x6h` 출력 인용** — S0 수리 전까지(위반 1.8배 과대) [S].
3. **`NEVER 89` · `ARGDIFF 53` 을 표적 크기로 인용하는 것** — 다중요구 가중이 걸려 있다(사건 35~36 / 47 이 그룹 내부) [S].
4. **MATCH/NEVER/ARGDIFF 를 종점처럼 쓰는 것** — reward 와 어긋난다(017 t1 · 050 t1) [S]. 1차 종점은 `pass^1` 하나.
5. **go_stack 과 PIN 스택의 결과를 나란히 놓는 것** — S13 전까지 비교 불가 [S].
6. **`T2_PROCEED_DOCBODY` / `DELIVER_PRECOMMIT` 계열의 유료 런** — 배관 사실을 유료로 사는 형태(t7303/t7304 심사 3인 일치).
7. **탐색 목적 full-run** — [[09]]. 로컬 무료 검증(격리 프로브·오프라인 replay·기존 데이터)을 다 끝내고 **확인용으로만**.

---

## 8. 측정 우선 큐 (다음 5개)

⛔공통: **정의는 런 전에 확정**한다. 런 후 정의를 바꾸면 그 항목은 무효.
⛔공통: 어떤 항목도 *"deny 를 줄였더니 좋아졌다"* 로 주장할 수 없다 — deny 는 고전하는 sim 에서 더 나오는 교락이다 [S].
⛔공통: 개입 강도 sim당 4~118(30배) 편차를 **층화 또는 공변량**으로 다룬다 [S].

### M1 (안전 종점·최우선) — 도달 래칫 [무료]

- **세는 것**: 사이드카 `channel × sim` 도달 행 수.
- **기준선 [S]**: 총 2,153행 · 24채널 · `unified_regen` 1,412(66%) · `claimprov` 134 · `signature` 5 · `covfollowup` 2.
- **사전 고정 종점**: 통합 후 **기준선에서 도달>0 이던 24채널 중 0 이 되는 채널 = 0개**. 하나라도 0이면 **효과와 무관하게 롤백**([[60]] 래칫).
- **부정통제**: 불필요(안전 종점). 이것을 통과해야 효과를 논한다.
- **비용**: 오프라인 replay + L0 스모크. 유료 0.

### M2 (1차 효과 종점) — 대상별 인자 분화 [L1 1축]

- **세는 것**: gold 가 같은 도구를 N≥2 회 요구하는 (sim, 도구)마다
  `DISTINCT-RATIO = |에이전트가 그 도구에 쓴 서로 다른 대상 id| / |gold 대상 id|`.
- **기준선 [S·3건만 확인]**: 074 t0 `apply_checking_account_credit` = **1/5** · 085 t1 `file_debit_card_dispute` = **1/3** · 040 `file_credit_card_dispute` = **1/9(전부 None)**. ⚠다중요구 89 조합 전수 분포는 **아직 안 셌다** — M2 착수 전 무료로 센다.
- **사전 고정 종점**: 다중요구 조합의 DISTINCT-RATIO 평균 **≥ +0.15** ∧ **spurious write(gold 밖 write) 증가 ≤ 0**. 둘 다 만족해야 통과.
- **부정통제**: 대상 집합을 **무작위로 뒤섞어** 제시하는 팔. 여기서도 오르면 산 것은 표면화가 아니라 재시도다.
- **왜 1차인가**: ARGDIFF 47/53 · NEVER 83/89 가 이 축이고, 세 설계안 **어느 것도 이 축을 다루지 않았다** [S].

### M3 (자해 종점) — 자기차단 [L1 동반]

- **세는 것**: deny 중 `val` 이 같은 sim 에서 우리 층이 먼저 말한 이름인 건수 / gold 오차단 / 미회복. **사건 단위로도 함께** 낸다(NEVER 89↔36 과 동형 문제가 deny 에도 있는지 미확인 [?]).
- **기준선 [S]**: deny 173(JOIN 114 · TARGET 49 · ARGVAL 10) · gold 56 · later_ok 45 · **미회복 11** · 자기차단 56.
- **사전 고정 종점**: **미회복 11 → ≤ 4** ∧ **spurious 통과 증가 ≤ 0**. 후자 없이 전자만이면 "막기를 그만둔 것"이지 고친 것이 아니다.
- **부정통제**: `_t2_our_names` 를 **무작위 이름으로 채운** 팔 — 출처 집합을 넓힌 것 자체가 효과인지 우리 지목이 맞아서인지 가른다.

### M4 (비용 종점) — 개입 비용 [L1 동반·항상]

- **세는 것**: sim당 `steps` · 사이드카 도달 행 수 · wall-clock.
- **기준선 [S]**: steps 150(079 t1 LOOP·stops 17) · 142(057 t0) · 124(079 t0) · 118(085 t1) · 도달 4~118/sim · 캡 래치 뒤 3,082 메시지(C537).
- **사전 고정 종점**: 중앙값 steps **증가 ≤ 0%** · 95퍼센타일 steps **증가 ≤ 10%** · 지연 **≤ 1.2×**. 초과하면 **pass 가 올라도 승격 금지**(C502 1.38× · C508 2.4× · [[46]] t7296 1.8× 전례).

### M5 (배선 진실) — `unified_regen` 1,412 의 내부 구성 [무료]

- **세는 것**: 도달의 66% 를 먹는 이 채널이 **어느 레버의 문면을 실어 나르는지** 갈라낸다(사이드카 `text` 지문 ↔ 레버 문면 템플릿 대조).
- **왜 여기 있나**: 이것을 모르면 M1~M4 의 어떤 결과도 "어느 레버가 했나"로 귀속할 수 없다. **개별 레버 통합보다 이 채널 하나가 더 큰 표적일 수 있다** [D].
- **사전 고정 종점**: 없음(탐색). **유료 런 금지** — 기존 t7326 사이드카 2,153행으로 끝난다.

---

## 9. 폐기·재론 금지 목록

### 9-A. 확정 — 다시 논쟁하지 않는다

| # | 확정 사항 | 근거(원장·실측) |
|---|---|---|
| 1 | **엔진이 답을 지목하지 않는다** — 범위 표면화까지 | x322 지목 24/24 → **0/24** |
| 2 | **엔진이 도메인 산문을 정규식으로 뜯지 않는다** | [[59]] · `parse_records` 형 · `fit_diff_note` 3줄 폐기 |
| 3 | **엔진이 대신 조회해 주입하지 않는다**(autofetch·present_reads 계보) | C34 [M] 규칙 0 위반 |
| 4 | **횟수 캡은 억제 수단이 아니다** — 인자 변화 술어로 | [[57]] · C537 "처방은 종료가 아니라 리셋 술어다" · C540 `stop=resolve_cap` 098 2→6 · 100 0→6 · 073 12→25 |
| 5 | **거부는 이름과 다음 한 수를 담는다** | [[64]] · 이름 없는 문구 3회↑ 6 sim = **6/6 실패** |
| 6 | **`MATERIAL_GATE`·`WINDOW`·`SELFDECL`·`STACK`·`LEVER`·`ROUTE_TRACE` 는 레버가 아니라 계기** | C434 "인쇄이지 레버가 아니다 ⇒ 감사 대상 46 → 8" |
| 7 | **`DELIVER_PRECOMMIT` 는 특허 실시예로 기록하지 않는다** | C502 [S] 전달 1/3 소실·1차 종점 순환·지연 1.38× |
| 8 | **`DOCS_AT_WRITE` 의 축별 배달은 반증됐다** | C505 [S] "굶은 축이 맞히고 먹은 축이 틀렸다" |
| 9 | **`SUB_REQUIREMENT` 는 스택에 승격하지 않는다** | C508 [M] gold 4→0 · 지연 2.4× |
| 10 | **`HANDOFF_PREDICATE` 는 폐기** — 술어만 P3 에 남는다 | C529 [S] 표적 희소·pass null·부호 반전 |
| 11 | **`DECISION_ISOLATE`(R8b) 는 이득 없음** | C403 [S] 24 sim "배제 근거까지 빼는 것으로 보인다" |
| 12 | **`MATERIAL_RESERVE` 는 무동작이었다** | C499 [S] "그 진단의 전제였던 내 로그 독해가 틀렸다" |
| 13 | **`L4` 치환 금지** | 2/2 오답(t58 정답파괴·t20 제약절단) |
| 14 | **`SOURCE_QUALIFY` 는 켜지 않는다** | 102 db_match 2/2 → 0/2 라이브 회귀 |
| 15 | **`t7326` 형 단일 스택은 효과 측정 도구가 아니다** | 기여 귀속 원리상 0건 |
| 16 | **005·102 는 분모에서 제외 · 069 는 표적 금지** | [[68]] · `TASK_LEVER_MAP_AND_EXCLUSIONS_2026_08_16.md` |
| 17 | **`TASK_LEVER_MAP` 은 [[05]] 위반이 아니다** — 결손 축 지도이지 태스크별 레버 스위칭이 아니고, 엔진은 태스크 id 를 읽지 않는다 | 엔진 48파일 grep 0건 · 런처가 두 팔에 동일 PIN 을 export |

### 9-B. ⛔사용자 판정 필요 (내가 확정하지 않는다)

`_note` 안 gold 참조 중 **강도(deny↔넛지)와 필수/선택 여부를 gold 가 정한 3건** — 내용 도출은 정책에서 왔으나 **부작용 계측을 넘어 설계를 결정**했다 [D]:
1. `procedures[2].nodes[5]._note_requires` — *"구판은 log_reason 을 필수로 걸어 **gold 2태스크를 막았다**(x91)"* ⇒ 그래서 완화.
2. `_note_choice_grounding` — *"agent 7건 / **gold 3건** ⇒ deny 는 오차단이라 넛지 1회로 둔다"* ⇒ 강도를 gold 가 정함.
3. `procedures[1]._note_prohibits` — *"gold 20건 중 last4 를 요구하는 것은 0건"*.

⇒ [[23]] 통과인지 gold-fitting 인지 **사용자 확정 전까지 이 셋에 손대지 않는다**.

---

## 10. 미해결 · [?] 목록

### 10-A. 근거 없이 통용되던 주장 (그리고 그것을 닫을 실험)

| # | 주장 | 실제 상태 | 닫는 방법 |
|---|---|---|---|
| 1 | `TOOL_SIGNATURE` 가 give 서명을 고친다 | **원장이 정반대**: C267 [S] *"V7 이 금지하는 형태가 DB 를 맞춘 경로다 … 승격 금지·다음 런에서 OFF 권고 … **DB 축의 대가는 한 번도 재지 않았다**"* ↔ `go_stack:132` ON · t7326 44발화/18 deny/**도달 5** | C267 이 지정한 그 측정: give-다발 태스크에서 V7 on/off 의 Δdb_match. 그때까지 모순을 문서에 명기 |
| 2 | `A2_VARIANT` 가 004 날조를 막는다 | 발화 **1위 1,668회**인데 원장에 **효과 항목 0** | 선언 A/B(변이 유/무) + Δspurious |
| 3 | `CLAIMPROV` 가 완료 날조를 막는다 | 등급열 축자 *"효과 [?]"*(C341) · 마크 1,124 ↔ **도달 134** | 도달 기준 A/B. 1,124회의 대가(문맥·턴)를 M4 로 |
| 4 | `FORCE_ACTION` 이 say-don't-do 를 닫는다 | C330 *"말로는 따르고 호출은 0"* · 효과 [?] · t7326 214발화 | 채널별(평서/regen/required) Δ 귀속 |
| 5 | `RESOLVE` 가 이름을 산다 | 최대 판정기(519발화·92 deny·gold 오차단 24)인데 원장 축자 *"효과는 [?]"*(C324) | P3 통합 전후 M3 |
| 6 | 캡·억제가 낭비를 막는다 | C537 래치 뒤 3,082 메시지 · C540 계약 경로 3~7배 사망 = **캡이 pass 를 팔았다** | M4 + S9 재무장 술어 A/B |
| 7 | 098·100 이 deny 0 이라 통과했다 | **상관이지 인과 아님**(교락) | M3 의 부정통제 |
| 8 | `[[60]]` 이 지켜지고 있다 | **지켜지지 않는다**: 사전 고정 바를 통과한 `ELIG_LINE`(C517)·`VERDICT_CARRY`(C515)가 PIN=0 이고, 원장에 남은 유일한 pass^k 개선 실측 `DISAMB`(C60 p2~p4 +1.5~2.6pp)는 **어느 런처에도 없다** [S] | S13 런처 통합 + 세 레버 재점등 A/B |
| 9 | "우리 스택이 pass 를 산다" | 라이브 pass 이동 실측은 **4종뿐**: `PROV_REGEN`(C53·456 sim) · `QUOTE_PIN`(C282·n=1) · `READ_DEDUP`(C114·n=1) · `WRITE_SUB`(C475). 나머지 ~270 은 배선·발화·격리까지만 왔다 | M2/M3 를 원소 단위로 |
| 10 | `ARG_EMPTY` 는 표적을 사지 않았다(C425 철회) | **실측과 충돌**: ARGDIFF 53 중 `None` 인자 **24건 실재**·040 에서 이 레버 발화 **0**(선점) [S] | 040 오프라인 replay 로 선점 제거 시 발화 여부 |

### 10-B. 세지 못한 것 (추정으로 메우지 않는다)

1. **다중요구 89 조합의 DISTINCT-RATIO 전수 분포** — M2 기준선. 3건만 확인했다.
2. **deny 173 에 다중요구 가중이 걸려 있는지** — 자기차단 56 이 사건 단위로 몇 건인지 미확인.
3. **사이드카 `sha`·`len` 으로 중복 문면을 접었을 때의 순 도달 수** — 위 channel 계수는 행 단위다.
4. **`unified_regen` 1,412 의 내부 구성** — M5. 도달의 66% 가 여기 있다.
5. **reward 의 내부 구성(db check ↔ communicate check)** — 017 t1 이 NEVER 2 로도 통과한 이유를 못 갈랐다.
6. **halfA/halfB 사이드카의 sim 중복 여부** — `simtag` 교집합 미검사.
7. **테스트 179개의 현재 통과 여부** — 실행하지 않았다(이번 작업은 문서·표만).
8. **전이 arm 에서 F1 술어가 몇 레버를 죽이는지** — airline/retail 라이브 로그가 없어 침묵 수를 실측 못 했다.
9. **`T2_MATCH_COUNT` 의 t7326 발화** — 0발화인지 마크 미배선인지 못 갈랐다.
10. **`FOLLOWUP_CAP`·`PROC_ABSENT_CAP`·`DD_FB`** — registry·주석에는 있는데 `environ.get` 자리를 코드에서 찾지 못했다(S2 대상).
11. **마크 68종 ↔ 플래그 전수 대응표** — 이름 유사도로 일부만 짝지었다. ⚠언더바 하나로 갈린 쌍이 여럿이라(`TRUNCGUARD`↔`TRUNC_GUARD` · `NOTICEREP`↔`NOTICE_REPEAT` · `ENVGUARD`↔`ENVELOPE_GUARD` · `COVERAGE_FU`↔`COVERAGE_FOLLOWUP` · `UNVERIFIED_FU`↔`UNVERIFIED_FOLLOWUP`) **grep 감사가 조용히 빗나간다**.
12. **원장 결번 36개**(C1~C8 등)의 초기 근거 — 압축으로 사라져 검증 불가.
13. **원장 밖 근거 5건** — `SOURCE_QUALIFY`(102 회귀) · `UNLOCK_QUIET`(격리 2/8·6/8·8/8) · `MAIN_ANSWERS_ONLY`(x231 8/8→0/8) · `L4`(2/2 오답) · `OPERATOR_PINPOINT`(x322)는 **코드 주석에만** 있다. 통합하면 그 주석이 붙은 코드가 사라지고 **근거도 같이 사라진다** ⇒ 원장 이관이 S3 이전 필수.

---

## 부록 A. 즉시 착수 3건 (비용 0 · 라이브 무영향)

| # | 작업 | 왜 지금 | 위험 |
|---|---|---|---|
| **A1** | 원장 밖 근거 5건 이관 + 모순 7건 확정(S2) | 통합이 근거를 지우기 전에 | 없음(문서) |
| **A2** | `x6h.selftest_range` 수리(S0) + 인벤토리 4자 대조 고정(S1) | 감사 수치가 1.8배 과대인 채로는 전후 비교가 오염된다 | 없음(감사 전용) |
| **A3** | M5(`unified_regen` 1,412 분해) + M2 기준선 전수 계산 | 기존 사이드카·trace 로 끝난다. **유료 0** · 이것 없이는 M2 종점을 고정할 수 없다 | 없음 |

---

*작성 2026-08-19. 이 문서의 수치는 전부 `x392_block_join.json` · `x384_fail_anatomy.json` · `x393_policy_reach.json` · `trace_bank_t7326_half{A,B}_20260819q.jsonl` · `fb_bank_t7326_half{A,B}_20260819q.jsonl` · 코드 축자 grep 에서 재현 가능하다. 인과는 하나도 없다 — t7326 은 단일 스택이다.*
