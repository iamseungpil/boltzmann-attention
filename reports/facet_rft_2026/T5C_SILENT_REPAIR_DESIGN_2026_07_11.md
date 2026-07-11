# T5-C — 채점 시스템 감사 + "조용한 개선(silent repair)" 재설계 (2026-07-11)

> 발단(사용자, 2026-07-11): ① "부작용은 잘못된 접근 때문이지 근원적인 게 아니다. 열거가 잘못된 선택을
> 열 수는 없다. 명시적 히스토리 기록으로 replay 채점 시 문제가 될 뿐 아닌가 — 채점 시스템을 엄밀 점검하라."
> ② "턴을 버리는 방식이 말이 안 된다. 조용히 개선하면 될 것 같다."
> 선행 정본: `E_AMB_MEASUREMENT_PLAN_2026_07_10.md` §7i (C61) · `RETAIL_PASS_COMPOSITION_DESIGN` §3c (C53-보강).

---

## §1. 채점 시스템 감사 — "replay 채점 아티팩트" 가설의 판정 [M]

채점 경로 (tau2-bench 코드 정독, `/home/woori/scratch/tau2-bench/src/tau2`):
1. **DB 채점 = 커밋된 메시지 히스토리의 replay**. `evaluator/evaluator_env.py:85-125` —
   `predicted_environment.set_state(히스토리)`가 **mutating tool call만 재실행**하고(비-mutating skip·
   hallucinated tool은 no-op), 재실행 응답을 기록된 ToolMessage와 비교, 불일치면 `ValueError`
   (→ infrastructure_error). 최종 DB hash를 gold-적용 DB와 비교(`environment.py:360-390`).
2. NL 축 = `evaluator_nl_assertions.py` LLM judge. 우리 런 reward_basis = DB + NL_ASSERTION (C19).

검증 결과:
- **커밋 히스토리 오염 0**: routerv1 17,771 msgs · prov 15,959 msgs 전수 스캔 —
  `[DISAMBIGUATE]`/`[PROVENANCE]`/`re-check pending`/gate 마커 **0건**. (개입은 전부 작업버퍼(dwork)
  로컬 — 중간 am·합성 ToolMessage는 히스토리에 안 들어감.) infra 0/456 양 arm.
- ⇒ **가설 기각: 손상은 채점 아티팩트가 아니다.** 채점은 "집행된 것"을 충실히 재생하고, 문제는
  개입이 "집행되는 것 자체"를 바꿨다는 데 있다 (write가 그 턴에서 파기되어 live에서도 실행 안 됨).
- 단 채점축 뉘앙스 실재: **db_match=True ∧ reward=0**(NL축 실패) trial — router 11 vs prov 19
  (router가 NL축은 오히려 개선). t46 trial2/3이 이 유형 (write·DB 정상인데 NL로 사망).

## §2. episode-level 손상 재분해 [M] — 그리고 사용자 명제의 판정

3-arm 전수 (456 trials each · 분류: PASS / db_match=True인 NL-only 실패 / write 호출 0 실패 /
write 있으나 오답 실패 · WRITES={exchange_,return_,modify_,cancel_}):

| arm | PASS | FAIL_NO_WRITE | FAIL_WRONG_WRITE | FAIL_NL_ONLY | write 호출 총량 |
|---|---|---|---|---|---|
| fl32b floor | 254 | **12** | 173 | 17 | **860** |
| prov (C53) | 263 | **22 (+10)** | 152 | 19 | 833 |
| routerv1 (C60) | 260 | **25 (+3)** | 160 | 11 | 809 |

- **★prov 자체가 무-write 실패를 배증**(12→22)·write 총량 단조 감소(860→833→809). 코드상 같은 채널
  존재(prov 재생성 루프도 최종 am이 텍스트-only면 무조건 수락). 단 **[[08]] 정독 3건(t92/69/40 trial별)이
  이 기전을 지지하지 않음**: 형상은 전부 조기 escalation(transfer_to_human ×2)·오결론-후-종료("바꿀 것
  없음" 합의 ×1) — 기계적 write-삼킴이 아니라 **대화-발산의 하류**(C61 H-E "재생성 분산"의 얼굴) 또는
  기저 flip. **등급 [P]·기전 미확정** — 발화-join(run stderr) 없이는 재생성 루프 직접 귀속 불가.
  신규 no-write 태스크 16개 분산(t31·34·69·74·92·40…).
- DISAMB 손상의 episode-level 지배형은 **WRONG_WRITE**(전무-write는 +3뿐) — §7i step-수준 "write-소멸
  39건"과 양립(다수는 *부분* 소멸 후 오답 재발화: t95 router trial2 exchange 1회 vs prov 2-3회).
- **사용자 명제 "열거가 잘못된 선택을 열 수는 없다" — 지지**: DISAMB 발화 1,274 중 switch 26(2%)·
  손상 기전은 스위치-오답이 아니라 **턴-파기(write 유실)와 대화 교란**. 정보(열거)는 무해했고
  **전달 기전(재생성)이 유해**했다. 이는 "부작용=같은 힘의 양면" 프레임의 **정정**이다: 열거 레버의
  부작용은 본질이 아니라 구현이 만든 것. (단, in-dialogue 개입이 대화 경로를 가르는 나비효과 자체는
  전달 기전 고유 비용으로 남는다 — silent 설계가 이것까지 제거.)

## §3. 적용된 수정 (커밋 `07337a3` · 2026-07-11 오전)

T5-C 스펙 #1 (handoff 2026-07-11 §4): `t2_gate_patch.py` 양 분기(prov-disamb·unified) —
재확인 응답 am2가 **tool_calls 없는 텍스트-only면 원 호출(am) 유지**(카운터 `_t2_disamb_nowrite_keep`).
DISAMB 원값은 문맥-실재라 유지 무해. **banking full-stack arm(bankxfer_switch3)은 이 수정 포함으로
발사됨**(mini-smoke 게이트 내장). 스펙 #2(REGEN_FEEDBACK 예시 중립화)·#3(구조대 모드)은 레버 버전
보존을 위해 미적용 — §4로 이관.

## §4. silent repair 설계 (다음 구현 — 사용자 방향)

**원리 4조** (개입의 불변량):
(i) 커밋될 턴을 파기하지 않는다 (write 보존) · (ii) 대화에 새 텍스트/턴을 넣지 않는다 (나비효과 0)
· (iii) 실행=기록 (replay-clean·§1이 보장 근거) · (iv) 레버 ≥ floor pointwise (실패 시 폴백 = 무개입).

| 경로 | 내용 | 근거 | 상태 |
|---|---|---|---|
| **P-A GROUND 이식** | \|C\|=1이면 재생성 없이 tool call 인자 제자리 치환(`t2_gate_patch.py:575-583` 기존 구현·unified 분기 미지원 → 이식). 후보 원천 = 에이전트 자신이 조회한 tool 출력만(DB 주입 0·규칙0 클린) | P2b/c: prov가 payment \|C\|=1 날조 0/319로 닫음(C57) | 구현 소 |
| **P-B DISAMB-silent** | \|C\|≥2: in-dialogue 재확인 폐지 → **격리 서브콜**(동결된 현재 문맥 + 후보 열거 → 선택만 반환) 후 서브콜 답 ≠ 원값일 때만 인자 제자리 치환. 원턴·대화 완전 불변 | C59 격리 열거 .657(+31pp) — "격리에서 검증된 이득을 격리된 채로 소비" | 구현 중 |
| **P-C prov 구조대 모드** | 사전 재생성 축소: env가 어차피 거부하는 id-날조는 개입 생략(C61 H-D: 70/70 env-차단 중복), free-text(주소 등 env가 못 잡는 타입·C24)만 사전 개입 유지 + env-거부 후 회복 유도 | C61 H-D·H-E(죽임 74) · §2 no-write 배증 [P] | 설계 |

**계측(제1원리·GO 조건)**: 치환률·switch 정오표(gold 대비) · **Δspurious ≤ 0**(치환이 정답-write를
오답으로 뒤집은 수) · no-write 실패 ≤ floor(12) · p1 ∧ **p4**(1급 축) 동시 보고.
**리스크**: gold∉C 3.7%(C55)에서 제자리 치환은 env-수락 오답 write를 만들 수 있다(floor라면 env-거부됐을
것) → id-형 인자는 "env-거부 예정"일 때만 치환하는 조건 분기 검토. 서브콜이 원값(정답)을 오스위치하는
비율은 c51 데이터로 사전 추정 가능(무료).

## §5. 대기 결정 (사용자)

1. **T5-C 재런**(retail 456×nt4·유료): 최소형 = §3 수정만(산술 상한 ≈ +27시행·p4 환매 확대) vs
   silent형 = P-A/P-B 구현 후. **권고: silent형까지 구현·스모크 후 1회 재런**(재런 2회 방지).
2. P-C(prov 구조대)를 같은 재런에 합류시킬지 (arm 수 증가 없이 단일 arm 통합이 E-COMP 정신).
3. banking arm은 §3 수정 포함 자동 진행 중 — 개입 불요.

---

# 구현 설계서 (rev1 · 2026-07-11 · 리뷰 대상)

> 절차: 본 설계 → 적대 리뷰(블로킹/비블로킹) → 반영(rev2) → 구현+V1 → (V0 GPU 확보 시) → V2/V3는 승인 후.
> 터치 파일 = `scripts/distill/tau2/t2_gate_patch.py` 단일. 기존 경로(C53/C60 재현)는 전부 보존, 신규는 opt-in env.

## §6. 구현 명세

### §6.0 공통 헬퍼 (신규)

```
_candidate_records(arg_key, orig_value, msgs, limit=6) -> list[(value, snippet)]
```
- `_grounded_candidates`와 동일한 후보 열거(기존 로직 재사용) + 각 후보에 대해 **그 후보가 등장한
  가장 최근 tool 출력에서 최소 enclosing JSON 객체**를 추출해 snippet(≤500자·초과 시 스칼라 필드만)으로 동봉.
- 근거: **E-ISO ③(C61) — id-only 열거는 판별정보가 없어 역효과**(order-⋈ C .21 < A/B .32).
  C59의 GO는 내용-매칭 열거였음. ⇒ 열거는 반드시 내용 동봉.
- 규칙0: 원천은 `_parse_tool_outputs(msgs)` = **에이전트 자신이 조회한 tool 출력만**. DB 접근 0.
- 도메인-일반: JSON 구조 연산만·스키마/필드명 리터럴 0 ([[05]] 통과).

### §6.1 P-A — GROUND 제자리 치환의 unified 이식 (기본 ON: `T2_GROUND=1`)

- 위치: `apply_unified_regen`의 prov 루프( fab 감지 직후·regen 前 ) — `apply_provenance_regen`의
  기존 575-583 블록과 동일 로직 복제:
  `cands = _grounded_candidates(k, s, state.messages); if len(cands)==1 and cands[0]!=s → tc.arguments[k]=cands[0]; subs+=1; continue`
- |C|=1 = 판단 없음 = decidable(§1.5 Q1) → 결정론 치환 정당. P2b/c(C57): prov가 payment |C|=1 날조를
  0/319로 닫음 = 이 칸의 정답률 근거.
- 턴 불파기·대화 불변·실행=기록. 치환 상한 subs<8/턴(기존과 동일). 카운터 `_t2_ground_sub`.
- **★[B1 반영] 원소-치환 알고리즘 확정** — 신규 헬퍼 `_subst_arg_value(tc, k, old, new) -> bool`:
  `d = _args_dict(tc)`(str-JSON이면 새 dict 반환이므로 **반드시 `tc.arguments = d` 재할당**·N2) 후
  `d[k]`가 (a) 스칼라·`str(d[k]).strip()==old` → `d[k]=new` (b) **리스트 → old와 일치하는 원소만
  위치 보존 교체**·new가 이미 리스트에 존재하면(중복 발생) **no-op·False** (c) nested dict/그 외 →
  no-op·False. 성공 시에만 카운터/`continue`. **기존 GROUND 블록(575-583)의 `d[k]=cands[0]` 통짜
  치환도 이 헬퍼로 교체**(list-인자 파괴 버그의 원천 수정 — GROUND는 e2e 미측정 레버라 버전 침해 없음).

### §6.2 P-B — DISAMB-silent: 격리 서브콜 + 인자 제자리 치환 (`T2_DISAMB_MODE=subcall`)

fire 조건(기존과 동일): confirm-write 도구 · arg가 hints 매칭 · **값이 문맥 실재** · `_grounded_candidates ≥2`
· `(tool,k,val)` memo 미방문. fire 시 — in-dialogue 재확인(dwork 주입·재생성) **전면 폐지**, 대신:

```
records = _candidate_records(k, s, state.messages)
transcript = user/assistant 텍스트 턴 전사(tool 원문 제외·_BLOCK_NOTE 등 개입 메타텍스트 필터=N5)
sub_msgs = [UserMessage(role="user", content=SUBCALL_SYS + "\n\n" + transcript
                        + "\n\nCandidates for '{k}':\n" + enumerate(records)
                        + "\n원값: '{s}'. Which candidate does the user intend? "
                          "Answer EXACTLY one candidate value, or UNSURE.")]   # ★[B4] kwargs 생성·단일 user 턴(SystemMessage 미검증 클래스 회피)
kw = {k2: v for k2, v in self.llm_args.items() if "tool" not in k2}            # ★[B4] tool_choice류 strip
sub = la.generate(model=self.llm, tools=None, messages=sub_msgs,
                  call_name="disamb_subcall", **kw)                            # ★[B4] tools=None(빈 [] 금지)
txt = (getattr(sub, "content", None) or "").strip()                           # ★[B4] None 가드
ans = 파싱(txt): ① strip(따옴표·공백·구두점) 후 txt 전체가 정확히 후보 1개 → 수락(N3 우선규칙)
                 ② 아니면 경계-인식 부분검색서 유일 매치만 수락 ③ 그 외 UNSURE
if ans == s or ans is UNSURE:  no-op (원턴 그대로)          # _t2_subcall_keep/_unsure
elif ans in cands and _key_tokens(k) ∩ A2.disamb_sub_args:
    _subst_arg_value(tc, k, s, ans)                          # _t2_subcall_switch (B1 헬퍼·실패 시 no-op)
else: no-op(로깅만)                                          # confirm-only 타입
```
서브콜 전체를 try/except로 감싸 **어떤 예외도 no-op**(에이전트 크래시=episode infra 사망 금지·B4).

- **원턴·대화 완전 불변**(서브콜은 히스토리 밖) · 재생성 0 · num_errors 0 · replay-clean(§1 보장).
- **★[B2 반영] arg-type 게이팅은 A2 config로**: 엔진은 "후보 arg의 `_key_tokens`가
  `a2.disamb_sub_args`(신규 필드·key-token 리스트)와 교집합일 때만 치환" 규칙만 가짐 — **엔진에 도메인
  어휘 0**. retail 초기값 = `["item"]`(c51: item .079→.545·new_item .116→.658 [M]) — V0가 갱신.
  order-id형 등 미등재 타입 = confirm-only(치환 OFF·로깅만). 타 도메인 전이 시 = 그 도메인 V0-동형
  측정으로 필드 채움(§7). 근거: E-ISO ③(id-열거 역효과)·[[05]](A2만 변경).
- **비대칭 안전성(핵심 리스크)**: fire 지점의 원값 정답률은 높다(1,274 fire 중 실패-관여는 소수) —
  블라인드 치환은 유해 가능. 서브콜이 **불일치일 때만** 치환하고, V0에서
  **P(서브콜 정답 | 불일치) > P(원값 정답 | 불일치)** 를 c51 데이터로 선검증(§7 V0). V0 불통과 타입은 confirm-only.
- 문맥 전사 = user/assistant 텍스트만(도구 원문 제외·후보 records가 내용을 담당) — E-ISO CP1/CP2
  오염-경로(20%)를 서브콜에 재주입하지 않기 위한 선택. 리뷰 포인트(§9-R3).
- memo·세션당 서브콜 상한 32(폭주 방지). latency ≈ 짧은 생성 1회/fire.

### §6.3 P-C — prov 구조대 모드 (`T2_PROV_MODE=rescue`; 기본은 기존 `full`)

fab 감지 시 순서: **(a)** P-A 제자리 치환(|C|=1) → **(b)** rescue 조건 분기:
- **★[B3 반영] env-검증형 판별자**: "인자→producer 매핑"은 A2에 존재하지 않음(producers=auth 단일).
  대체 = **A2 `preconditions` 게이트들의 `resolver_path[0]` 인자명 집합**(예: `[order_id,
  get_order_details, status]` → `order_id`) = env가 lookup으로 검증하는 id-형의 기존-A2-파생 집합
  (`gate_interpreter.py:240-250`이 이미 이 경로로 live-read). 도메인 리터럴 신규 주입 0.
  보조: `_sig(s) in {hashid, numid}`(id-형 시그니처)와 AND.
- **개입 생략(pass-through)**: 인자가 위 env-검증형 **∧** 에러-루프 아님 → 그대로 커밋(환경이 거부 —
  C61 H-D: 70/70 차단·중복). 카운터 `_t2_prov_skipped_envdup`.
- **개입 유지(기존 regen)**: free-text형(env-검증형 집합 밖·env가 못 잡음·C24) 또는 **에러-루프 중**.
  **★[N6] 에러-루프 감지** = committed `state.messages`의 최근 **tool-result 쌍 6개** 안에
  같은 tool 이름의 `error=True` ToolMessage 존재 — assistant `tool_calls.id ↔ ToolMessage.id` join으로
  tool 이름 복원(`_iter_tc_result_pairs` 패턴). unified 경로선 deny가 히스토리에 안 남으므로
  "error=True = env-오류" 동치 성립(리뷰 검증).
- REGEN_FEEDBACK **중립화**(스펙#2): 예시 나열("payment methods, or addresses") 삭제 — t61형 오도([P]·
  [[42]] priming 동형) 제거. GROUND_FEEDBACK의 괄호 예시("which order contains that item")도 동일 처리.
- 폴백 불변량: regen 최종 am이 텍스트-only면 — **fab가 id-형이면 원 am 유지**(env 거부=floor 동형),
  **free-text형이면 현행 유지(am2 수락)** — free-text는 원턴 복원이 날조-write 커밋이 되는 비대칭(§3) 때문.

### §6.4 arm 정의·플래그 (재현성)

| arm | env | 비고 |
|---|---|---|
| routerv1 (C60·보존) | `T2_PROV_REGEN=1 T2_DISAMB=1` | 기존 경로 그대로(+§3 fix) |
| **routerv2 = T5-C** | `T2_PROV_REGEN=1 T2_PROV_MODE=rescue T2_GROUND=1 T2_DISAMB=1 T2_DISAMB_MODE=subcall` | 단일 arm(P-A+P-B+P-C 통합) |
- unified(gate) 분기에도 동일 플래그 해석 이식(E-COMP/banking 계열 후속용).
- 레버 버전 라벨: prov-v2/disamb-v2 (P_OC rev4 — 셀 불변·버전 갱신).
- **신규 플래그 전부 opt-in·기본값서 v1 동작과 바이트-동일**(리모트 드라이버가 persist 단계서
  `git pull --rebase`로 코드를 당기므로 — 실행 중/대기 중 arm(banking full-stack 포함)에 무영향 필수·V1 회귀로 보증).
- [N9] orchestrator-레벨 `apply()` 게이트와 **상호배타**(`_deny_msg` 히스토리 커밋=replay-파괴 기지·`:681-686`).

### §6.5 계측 (stderr 마커·census 호환)

`_t2_ground_sub` · `_t2_subcall_{fired,keep,switch,unsure}` · `_t2_prov_skipped_envdup` ·
`_t2_disamb_nowrite_keep`(§3) — 각각 `[T2_*]` stderr 마커 동반(기존 eamb7/checkpoint census가 집계 가능).

## §7. 검증 계획

| 단계 | 내용 | 비용 | GO |
|---|---|---|---|
| **V0** | **★[B5 확정 절차]** (a) c51_disamb_results.jsonl에는 원값 컬럼이 없음 → `fl32b_floor_retail_t4.results.json.gz`와 (task,trial,idx,arg) **join으로 에이전트 원값 복원** (b) P-B 프로토콜 그대로(텍스트 턴 전사+records·R5 해소) 로컬 vLLM 재현 런 → 타입별 **P(sub 정답\|불일치)−P(orig 정답\|불일치)** (c) **정량 GO: 차이>0 ∧ 95% CI 하한>−0.05 ∧ n≥30/타입** (d) 편향 명기: c51 수집이 gold∈C 조건화(`c51_disambig_boundary.py:92-93`) → live(gold∉C 3.7%·C55) 대비 과대추정 ⇒ 임계 보수 적용 (e) [N8] 불일치∩both-wrong에서 env-거부형→env-수락형 이동(spurious 순증) 계수 | 무료(GPU 대기) | (c) 통과 타입만 A2 `disamb_sub_args` 등재 |
| **V1** | 단위테스트: (i) silent 경로서 반환 am.tool_calls ≥ 원 am (ii) 대화 텍스트 불변 (iii) 치환=fire 인자 1개만 (iv) list-원소 치환 왕복 (v) 기존 플래그 조합 회귀(§3 fix 포함) (vi) rescue 분기표 전수 | 무료 | 전부 PASS |
| **V2** | 스모크 10task×1(routerv2) — 발화 카운터>0·crash 0·공식 채점 통과(replay-clean 실증) | 유료 소액(승인 후) | 라이브 발화 확인 |
| **★V2.5** | **표적 복구 실험**(사용자 지시 2026-07-11: "전체 재실험 전에 실패 케이스 대표만 골라 복구 확인") — **13태스크×nt4=52 sims**(≈full의 11%)·COMP+D-v2 config. 기전별 표적+대조군·per-task로 **모든 설계 기능의 복구를 개별 검정** | 유료 소액 | **아래 기전별 GO 전부 통과 시에만 V3** |
| **V3** | full 456×nt4 = **COMP+D-v2**(rev3) | 유료(승인·**V2.5 게이트 후**) | **p1≥COMP .634 ∧ p4≥.316 ∧ Δspurious≤0 ∧ no-write≤12 ∧ 위반0** |

### §7b. V2.5 표적 명세 (기전 → 태스크 → 판정)
| 기전 | 표적 태스크(근거) | 복구 판정(per-task·vs 기록된 COMP/routerv1/floor 패턴) |
|---|---|---|
| **fix#1+P-B** write-loss 소멸 | **t46·t47**(C60/C62 write-loss·derail) · **t95**(write-loss+constraints) | write 호출 재출현(no-write trial 0) ∧ pass ≥ COMP 패턴 |
| **R-β** constraints 제외 | **t95**(COMP 0/4·floor 3/4·over-steer) | **≥2/4**(steer 소멸 시 floor-형 복원) |
| **P-B** silent-DISAMB 회복력 | **t61**(naive DISAMB가 0/4→4/4 회복했던 것—silent도 되나) · **t2·t101·t103**(⋈/변형 표적) | t61 **≥3/4** ∧ 나머지 COMP 대비 비퇴행·합계 ≥1 개선 |
| **P-C** prov 무-write 부작용 제거 | **t40·t69·t92**(prov no-write 배증 피해·C62②[P]) | no-write trial ≤ floor 패턴(0-1) |
| **P-A/P-C** 날조 억제 유지 | **t17**(prov 정본·4/4) | **4/4 유지**(rescue 모드가 free-text 개입 보존 증명) |
| **무회귀 대조군** | **t0·t28**(COMP 신규-robust) | **4/4 유지** |
- 발화 카운터 필수: `_t2_subcall_fired>0`·`_t2_prov_skipped_envdup>0`(rescue 실발화)·`_t2_disamb_nowrite_keep` 관찰·crash/infra 0.
- 판정 원칙: 기전별 독립 판정(하나라도 실패 → 그 기전 수정 후 V2.5만 재실행·V3 금지). 52 sims라 pass^k 통계 아님 — **per-task 패턴 대조**가 판정 단위([[08]]).

## §8. [[05]]·제1원리 점검 + 리스크

- [[05]] 3문: 도메인-특화 순증 0(JSON-구조 연산·A2 기존 매핑 재사용·리터럴 0) / 유동성 동결 —
  P-A는 |C|=1(판단 없음=decidable)만·P-B는 **선택을 LLM(서브콜)에 유지**·전달만 결정론 / 대행 수행 0
  (조회·행동 없음, 에이전트 자신의 호출·자신이 조회한 문맥만).
- 제1원리(부작용 계측): 치환의 반대편 = **정답→오답 스위치**(Δspurious 동형) — V3 per-case로 계수,
  `switch` 전건 로깅으로 사후 감사 가능. P-C의 반대편 = pass-through로 인한 에러예산 소모(tme) — V3서 tme 계측.
- 잔여 리스크: ① gold∉C 3.7%(C55)서 P-A 치환이 env-수락 오답 write 생성 가능(id-형은 env-거부 예정
  케이스만 치환되므로 실질 free-text 칸 — P-A는 id-형 위주라 소폭) ② 서브콜 파싱 실패 → UNSURE(안전측)
  ③ user-sim 나비효과는 **P-B에서 구조적으로 0**(대화 불변) — 이것이 v1 대비 핵심 개선.

## §9. 리뷰 결과 (rev1 → rev2 · 적대 리뷰 2026-07-11 · 판정: 조건부 GO)

**블로킹 5 — 전부 rev2 본문 반영**:
- **B1** list-인자 통짜 치환 버그(기존 GROUND 575-583 포함·fire 지배형이 `item_ids`=리스트라 주경로)
  → `_subst_arg_value` 원소-치환 헬퍼 확정(§6.1)·기존 블록도 교체(GROUND e2e-미측정=버전 침해 없음).
- **B2** 화이트리스트 `{item}` 하드코딩=[[05]] 위반 → A2 신규 필드 `disamb_sub_args`로 이관(§6.2).
- **B3** "인자→producer 매핑" 부재(A2 producers=auth 단일) → preconditions `resolver_path[0]` 파생
  집합으로 대체(§6.3). P-C는 이 해소 전 NO-GO였음.
- **B4** 서브콜 API 미검증(SystemMessage 존재 미확인·위치인자 생성·tools=[] 거부 가능·content None·
  tool_choice 충돌) → 단일 user 턴·kwargs·tools=None·가드·llm_args strip·전체 try/except(§6.2)
  + V1에 리모트 실-generate 프로브 1콜 추가.
- **B5** V0 계산 불가(원값 컬럼 부재)+GO 비정량 → fl32b join·정량 GO·gold∈C 편향 명기(§7).

**비블로킹 9 — 반영/예정**: N1 GROUND 블록은 unified 루프서 `fab` 감지 직후·`_denied_calls` **앞**
배치(게이트 check가 상태-변이(one-shot select_confirm)라 버려질 반복서 소진 금지) · N2 `_args_dict`
str-경로 재할당(§6.1 반영) · N3 파싱 우선규칙(§6.2 반영) · N4 `_parse_tool_outputs`가 augment-append
content서 침묵 실패 → leading-JSON 파싱 방어(구현) · N5 전사서 `_BLOCK_NOTE` 필터(§6.2 반영) ·
N6 에러-루프 감지 명세(§6.3 반영) · N7 텍스트-콜 불일치는 NL축 신규 채널 후보 — V2서 관찰·필요시
content 동일-치환 옵션 · N8 V0에 spurious-이동 항(§7 반영) · N9 orchestrator-레벨 `apply()`와
상호배타(§6.4에 명기: replay-파괴 기지).

**리뷰어 판정**: P-A/P-B = B1·B2·B4 반영 후 GO / P-C = B3 해소로 부활(V3 arm 포함 여부는 V1 후 확정)
/ V0 = B5 반영 전 화이트리스트 입력으로 사용 불가.

## §10. 최종 리뷰 반영 (rev3 · 2026-07-11 · 사용자 "새 실험에 추가" 지시)

> 맥락 정정: 사용자가 **중단한 것은 naive COMP+D**(게이트6종+calc/nested+prov+disamb·7/456서 정지)이고,
> "재개"의 대상은 그 **개선판**이다. rev2의 `routerv2`(게이트 없는 prov+disamb)는 이 목표와 불일치.

### R-α (헤드라인 재정의·블로킹) — 새 실험 = **COMP+D-v2**, routerv2는 격리 ablation로 강등
- **COMP+D-v2** = 전체 unified 스택 + **silent** prov/disamb: `T2_GATE_REGEN=1 T2_GATE_KINDS=<§R-β> T2_PRESENT_NESTED=1 T2_CALC=1 T2_PROV_REGEN=1 T2_PROV_MODE=rescue T2_GROUND=1 T2_DISAMB=1 T2_DISAMB_MODE=subcall`.
- **비교 기준선 = C62 COMP(reward .634/.480/.382/.316·db .665/../.375·위반0)** — prov(.577) 아님.
  합성 GO질문 = "silent DISAMB/prov가 게이트 위 COMP에 **순증**하나(p1↑) ∧ **p4 비퇴행**(≥.316, silent가 naive의 재생성분산 제거)".
- ⇒ **P-A/P-B/P-C의 unified 분기 이식이 이번 범위**(rev2 §6.4 "후속용" 철회). §6.1~6.3 로직은 prov-disamb 분기와 동일·unified의 `unified()` 함수에 동형 배치. routerv2(게이트 무)는 **budget 허용 시에만** 격리 ablation(silent가 게이트와 독립임을 증명·기준선 routerv1/C60).
- 근거: 사용자가 정지한 것이 COMP+D·C62가 COMP까지만 측정(DISAMB 미측정)·retail-pass 최종 산출물은 COMP+D.

### R-β (게이트 부작용·보정) — COMP+D-v2에서 `constraints` kind 제외
- C62 Δspurious 후보 **t95**(floor 3/4→COMP 0/4) = `constraints`(equal_len) 게이트 over-steer(gold=2주문 각1item 교환인데 steer가 1주문 중복 유도). retail.gate.json G7 `_note` 자인: "env가 이미 강제=env-mirror·false-block 0" ⇒ **결과적으로 redundant인데 1건 유해**.
- 처방: COMP+D-v2 kinds = `auth,confirm,ownership,notice,preconditions`(**constraints 제외**). env가 disjoint/equal_len 여전히 집행(정확성 무손실·[[13]] scaffold 감소). V3 per-case서 t95 회복 확인 = 이 판단의 검정. (constraints steer의 순이득은 미측정이나 유일 측정치가 유해라 제외가 보수적.)

### R-γ (계측·정합) — V3 GO 기준선·per-case를 C62로 재정박
- V3 GO(§7 갱신): **p1 ≥ COMP .634 ∧ p4 ≥ COMP .316 ∧ Δspurious ≤ 0 ∧ no-write ≤ floor(12) ∧ 위반0 유지(compliant=bench)** + per-case **t95 복구(constraints 제외 효과)·t61 4/4 유지·t46/47 write-loss 복구·t17 4/4 유지**.
- routerv2 ablation(선택)은 rev2 §7 기준(vs prov .577/.281) 유지 — silent 기전의 게이트-독립 증명용.
- **불변**: V0(무료)·V1(무료)은 R-α와 무관하게 rev2대로 선행(서브콜 per-type 정답·replay-clean 단위). V0 통과 타입만 `disamb_sub_args` 등재는 그대로.

### 리뷰 잔여(비블로킹·기록)
- N10: 서브콜이 same-32B라 E-ISO C-정확도(ITEMS .44)가 상한 — 오답 치환은 V0 보수임계+"불일치 시만" 가드로 억제하나 gold∉C(3.7%)서 잔여 spurious 가능(§8-①과 동일·수용). N11: COMP+D-v2는 calc/nested 포함이라 서브콜 문맥 전사에 calc 주입텍스트([COMPUTED FACTS]) 혼입 가능 → N5 필터에 `[COMPUTED FACTS]`·`[OPERAND DISAMBIGUATION]`·`[DISAMBIGUATION NOTE]` 마커 추가(silent 서브콜은 augment-전 원문만 봐야). N12: routerv1(C60)은 naive DISAMB 버그 포함 수치 → C60 인용 시 "버그-포함" 명기(수정판과 비교 금지).

**rev3 판정**: R-α/R-β/R-γ 반영 시 새 실험 = **COMP+D-v2 단일 V3 arm**(headline) + routerv2(optional ablation). 구현 범위 = §6.1~6.3 로직의 prov-disamb·**unified 양 분기**(R-α) + N11 필터 + kinds 조정(R-β). V0/V1 무료 선행 후 V3 승인.
