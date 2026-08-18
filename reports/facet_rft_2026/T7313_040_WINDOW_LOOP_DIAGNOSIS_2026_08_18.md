# t7313 `task_040` 창 순환 — 원인 진단 (2026-08-18 · 무료 · GPU 0)

> 등대 = `RESEARCH_MASTER.md`. 원장 = **C536**. 진단 순서 = [[55]](배관 → 문구 → 계기 → 모델).
> 재료 = 라이브 로그·사이드카 **축자**(리모트 읽기만) + 소스 직독. **엔진 수정 0**(동결 중).

## 0. 관측 (t7313 treat · `task_040#s626729`)

- `[T2_MATERIAL_GATE] stop=resolve_cap(정체 3회)` 가 **turn 56 → 104, 짝수 턴마다 59회**.
- 그 sim 하나가 **2,780초 · turn 104**. `GO_MAX_STEPS=150` 이 없었으면 t7307(C527)처럼
  런이 죽었을 자리다. ctl 도 같은 병을 앓는다(`task_057` **26회**·1,025초).
- 사이드카에서 **가장 많이 반복된 문면**(그 sim 안):

```
[6x] Error: [SIGNATURE] give_discoverable_user_tool takes only `discoverable_tool_name`
     in this domain; you also passed `arguments`. Re-issue the call with the dec…
[5x] Error: resolve the flagged call(s) first; do not call this tool yet.     ← 이름이 없다
[4x] Error: [WRITE-GROUNDING] the value 'your_credit_card_account_id' … does not appear …
[3x] [VALUE-ACQUIRE] The customer cannot read the last 4 digits off the card …
```

## 1. ⒜ 이름 없는 거부 본문이 아직 한 자리 남아 있다 — [[64]] 정면

`t2_gate_patch.py:8770-8771`

```python
else:
    content = _FB_GENERIC          # "Error: resolve the flagged call(s) first; …"
```

- C416 이 만든 `_sibling_wait(tag, flagged, what)` — *어느 호출이 막혔는지 이름을 대고 다음 한
  수를 준다* — 는 **네 자리에만** 붙어 있다(`:4354` PROVENANCE · `:4408` ARG-SCHEMA ·
  `:4459` DISAMBIGUATE · `:5185` POLICY GATE). **fb 조립의 `else` 는 빠졌다.**
- 그 문구의 해악은 이미 격리로 쟀다: **일반 문구 3회 = 정체 3/8 ↔ 원본 본문 = 0/8**(x246·C414),
  라이브 30 sim 에서 3회↑ 나온 6건 **6/6 실패**(C413). 040 에선 **5회**.
- 수리 형태: 같은 턴에서 **특정 본문을 받은 호출들의 이름**을 대고 그 뒤에 한 수를 붙인다.
  거동은 fail-closed 그대로(그 호출은 여전히 실행되지 않는다) · 새 결정론 0 · 도메인 어휘 0 ·
  `T2_KEEP_DENY_BODY` 되돌리기 경로 유지.

## 2. ⒝ ★이미 저작된 문면이 라이브에서 한 번도 안 켜졌다

| | 문면 | 길이 | 라이브 |
|---|---|---|---|
| v1 | `no_record_template` — *"…call get_user_information_by_name/by_email/by_id, **then call this tool again**."* | 253자 | **쓰임** |
| v2 | `no_record_template_v2` — 같은 인자 재조회 금지 · **못 찾으면 손님에게 말하고 다른 식별자** · 줄 수 없으면 **종결** · 검증 통과 **후에만** 기록 · *"right now the next step is the lookup"* | 639자 | **안 쓰임** |

- 문 = `t2_compute.py:674` `if os.environ.get("T2_NOREC_BRANCH") == "1"`.
- **`go_stack.sh` 도 런처도 그 플래그를 안 켠다.** 옛 `run_axis32_chain.sh:52` 는 켰다.
- v1 은 *"then call this tool again"* 으로 닫혀 **종료 분기가 없다** — x33·x34 가 D1(지시 결함)
  으로 기록한 바로 그 문구이고, 핸드오프 §5⑵ 의 072(같은 확인 **6회** 반복 → 손님 이탈)가 그 모양이다.
- ⇒ **여기서 살 것은 새 저작이 아니라 이미 있는 문면**이다. 측정 = `x379_norec_wording_iso.py`
  (A_REF v1 / **B_V2** / C_NAME(v2+다음 행동 이름) / D_NEG · 컷 25 · 8141 전용).

## 3. ⒞ resolve_cap 래치 — 손해 **미측정**(거동 안 건드린다)

`_resolve_cap_ok`(`:3513`)는 *정체*(직전 발화 이후 새 실행 도구 0 ∧ 새 회수 이름 0)에만 과금하고
캡(3)에 걸리면 **계약 경로 전체가 침묵**한다(`:6882`). 그런데 그 침묵이 *"지금 X 를 하라"* 를
없애므로 새 실행이 안 생기고, 새 실행이 없으니 **리셋 조건이 영영 오지 않는다** — 래치다.

- 040 은 그 래치 뒤로 **48턴**을 태웠다. 양팔 공통(ctl 057 26회).
- C530 이 *"손해 미확정 ⇒ 관측만 수리"* 로 남긴 자리 그대로다 ⇒ **[[62]] 대로 재기 전엔 안 고친다**.
- 무료 측정 설계(다음): 영속 런 전수에서 *래치 sim* ↔ *비-래치 sim* 의 종료사유·turn·pass 를
  짝지어 본다. 래치가 **pass 를 깎지 않고 시간만 태운다**면 처방은 레버가 아니라 **조기 종료**다.

## 4. 감사의 사각 — 세 번째 방향을 정본에 추가했다

`t2_levers.audit()`는 라이브→셀, `audit_declared()`는 셀→라이브만 본다. **셀에도 런처에도 없는
플래그는 어느 쪽에도 안 잡힌다** — 두 감사가 `0건 ✓` 를 인쇄하는 동안 `T2_NOREC_BRANCH` 가
꺼져 있었다. ⇒ `t2_levers.audit_unset()` 신설([[67]] 정본에 추가·사본 0).

- 결과: **56종**(파라미터 접미사·코드 기본 ON·RETIRED·NOT_LAUNCHED·ARM_ONLY 제외).
- 그중 **41종은 원장에 이름조차 없다** = 어떤 판정도 받은 적 없다:
  `ASK_UNKNOWN_BOOL · CLAIM_BLOCK · CONSISTENCY · DECISION_ISOLATE · DECLFIRST(3) · ENVELOPE_TAG ·
  ENVELOPE_TRUNC · EPLAN_EXAMINED_SAFE · EPLAN_READS_ONLY · FEXEC · FIT_DIFF · FN_ISOLATE ·
  FORCE_MIN_TOKENS · GROUND_DROP_NAVKEYS · GROUND_HDR · HANDOFF_ARG_GROUND · L4 · NLNUM_PROV ·
  NOREC_BRANCH · NO_DIGEST_REEXEC · OPERATOR_PINPOINT · PROVENANCE · PROV_GROUND · PROV_ORIGIN ·
  READALL · READ_NEARDUP · REPEAT_GOV · RETURN_EMPTY · RULES_PROMPT · SALVAGE · SCALAR_ARRAY ·
  SG_EXCLUDE · SG_SUB_TOOLCAP · SOURCE_QUALIFY · TERMINAL_TURN · TOOLERR · TRANSFER_PREREQ ·
  UNKNOWN_UNVERIFIED · UNLOCK_QUIET`
- ⚠**이것은 "41개를 켜라"가 아니다.** 상당수는 후계 레버로 흡수됐을 수 있다(예 `T2_CALC`↔
  `T2_COMPUTE` · `T2_COV`↔`T2_COV_MIDDRIVE` · `T2_TERMINAL_TURN`↔`T2_TERM_GRANT` · `T2_PROVENANCE`↔
  `T2_PROV_REGEN`). [[60]] 은 *"끄지 마라"* 이지 *"전부 켜라"* 가 아니다 — **하나씩 후계를 대거나
  판정을 붙이는 것**이 부채다. 착수 순서 = 라이브 실패에 닿아 있는 것부터(`NOREC_BRANCH` 가 1번).

## 5. 실행 제약

⛔⒜·⒝ 는 **엔진·A2 변경**이라 `frozen_hash = b6a64de79158a0ed` 가 바뀐다 ⇒ 런북 **STEP 4** 에서만,
그리고 적용하면 **1단계부터 다시**(스모크 4 포함 20 태스크). 그 전에 x379 로 ⒝ 를 재는 것이
[[62]] 순서다. ⒞ 는 측정 전 손대지 않는다.
