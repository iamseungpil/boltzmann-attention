# 레버 97개 → 계약 5개 배정 (초안·2026-08-07)

> 사용자 지시: *"레버는 5개만이다."* · *"C1이 근거 확보, C2가 선행 검증."*
> 계약 정의 정본 = `GENERAL_CONTRACTS_DESIGN_2026_08_06.md` §2.

## §0 먼저 — 어제 무엇이 통합됐는가 [M]

기록 대조 결과 **통합은 문서에서만 일어났다**:

| 주장 | 실측 |
|---|---|
| "79 레버를 5개로" | **79는 레버가 아니라 A2 키**다(설계서 §1 축자: *"'79키' = gate 44 + specific 35의 두 층 합산"*) |
| A2 키 79 → 계약 5 | 현재 **82키**(gate 46 + specific 36). 감소 0 |
| 흡수 커밋 `9eb0a946` | **`GENERAL_CONTRACTS_DESIGN_2026_08_06.md` 1개 파일만 변경.** 코드 0줄 |
| 엔진 레버 | `go_stack.sh` T2_* **128종 = 레버 97 + 파라미터·설정 31**. 감소 0, arm이 3개 추가 |

설계서 §6b(a)가 *"30.2%→63.4%를 만든 것은 계약 5개 어디에도 들어가지 않는다 ⇒ '계약 밖이지만 유지'"*
로 빠져나갔고, §8이 실제로 버린 것은 3항목뿐이다. **"5개"는 목표였고 실행된 적이 없다.**

---

## §1 배정 (초안 — 근거는 각 레버의 자기 설명 1줄)

### C1 출처 — *주장이 출처집합 안에 있는가* (막는다)
`PROV_REGEN` `PROV_MODE` `GROUND` `SG_GROUND` `WRITE_PROV` `WRITE_EVIDENCE` `WRITE_ARG_GROUND`
`CLAIM_PROV` `TRANSCRIBE` `GIVE_QUOTE` `QUOTE_HINT` `QUOTE_PIN` `UNLOCK_PROV` `UNKNOWN_NAME_BL`
`UNLOCK_NAME` `FAB_STRIP` `CHOICE_GROUND` `REQUIRE_DOC` `KB_NOHIT_SURFACE`
`SEARCH_EXHAUST_NUDGE` `UNAVAIL_PROMISE`
`SG_TRUTH` `SG_WINDOW_ABSTAIN` `ABSTAIN_FIELDS` `PROD_BIND` `REF_VERIFY` `TRANSFER_TIER` — **27**

### ★C7 발견 — *출처 안에 있는데 아직 안 쓴 것* (보여준다·2026-08-07 신설)
`DISCOVERY_NAMES` `UNCALLED_UNLOCK` `VERDICT_SURFACE` `TRANSFER_LEAVES_STEPS` `MATCH_COUNT` — **5**

> C1에서 3개(`DISCOVERY_NAMES`·`UNCALLED_UNLOCK`·`VERDICT_SURFACE`·`MATCH_COUNT`),
> C2에서 1개(`TRANSFER_LEAVES_STEPS`)를 옮겨 왔다. 방향이 반대라서다 — C1은 출처 **밖**을 막고,
> C7은 출처 **안**에 있는데 안 쓴 것을 짚는다. `KB_NOHIT_SURFACE`(출처에 **없다**)는 C1에 남는다.

### C2 선행 — *미충족 조상이 먼저 말한다* (deny · replace · pin)
`FORCE_ACTION` `PROCEDURE` `PROC_ABSENT` `PROC_PIN_REARM` `PIN_READ` `PIN_READ_STEPS` `EPLAN`
`EPLAN_WALK` `FOLLOWUP_REQUIRED` `FOLLOWUP_FORCE` `FOLLOWUP_READLOOP` `COVERAGE_FOLLOWUP`
`UNVERIFIED_FOLLOWUP` `SCAFFOLD_GET` `SG_REQREADS` `PREKB` `TERM_GRANT` `TERM_GRANT_USERDEMAND`
`UNINSTRUCTABLE` `GIVE_EXEC_NUDGE` `GIVE_RELEVANCE_NUDGE` `COV_MIDDRIVE`
`TOOLGATE` `CALLABLE_HINT` `WITHDRAWN_ROW` `DISPATCH_LEDGER` — **26**

### C3 중재 — *등급 높은 쪽이 자기 범위 안에서만 명령·진 쪽은 치환*
`PHASE_OWNER` `SPEAK_PROHIBIT` `GATE_REGEN` `BRANCH_REGROUND` `UNKNOWN_REPEAT_GUARD` `REPEAT_CAP` — **6**

### C4 역할 — *실행 주체·인자 소비자는 레지스트리에서 도출*
`DISPATCH_ROLE` `DISPATCH_ROLE_ENVSET` `TOOL_CHANNEL` `USER_TOOL_NOTE` `VALUE_ACQUIRE` `HAVE_VALUE`
`HAVE_VALUE_FORCE` `ARG_PRODUCERS` `TOOL_SIGNATURE` `TOOLLIST` `SG_BYREF` `ARG_SCHEMA` — **12**

### C5 이관 — *엔진은 연산자·서식, 도메인은 필드 목록*
`COMPUTE` `RESOLVE` `SG_ISOLATE` `SG_ISOFB` `SG_TRACE` `SG_DEDUP` `PRESCRIPTION` `LEDGER`(신규) — **8**

### 계약 밖 — **레버가 아니다**(런타임 보호·뷰·사고 대응)
`OVERFLOW_GUARD` `TRUNC_GUARD` `ENVELOPE_GUARD` `DYN_MT` `VIEW_COMPACT` `VIEW_ANNOTATE`
`STALE_STRIP` `READ_DEDUP` `PAIRCHECK` `PAIRFIX` `DUP_REPRESENT` `FAILED_PERSIST` `MAXPROMPT`
`GUIDED` `ACTION_PROGRESS_REFUND` `FOLLOWUP_PROGRESS_REFUND` `A2_VARIANT` — **17**

**합 = 27+26+6+12+8+5+17 = 101** (일부 플래그가 두 술어에 걸쳐 중복 계수됨 — §2 참조)

---

## §2 이 표를 그대로 믿으면 안 되는 이유 [[55]]

- 배정 근거가 **`go_stack.sh`의 한 줄 주석**이다. 주석은 그 레버가 *무엇을 막는지* 말하지, *어떤 술어로
  판정하는지*는 말하지 않는다. **코드 술어를 읽어야 확정된다.**
- 경계가 겹치는 것이 최소 **8개**: `UNKNOWN_NAME_BL`·`UNLOCK_NAME`(이름 출처 C1 ↔ 소속 C4) ·
  `CALLABLE_HINT`·`PIN_READ`(선행 C2 ↔ 채널 C4) · `TRANSFER_TIER`(정책 출처 C1 ↔ 절차 C2) ·
  `REPEAT_CAP`·`UNKNOWN_REPEAT_GUARD`(C3 억제 ↔ 애초에 [[57]]로 **삭제 후보**).
- **`REPEAT_CAP` 계열은 계약이 아니라 폐기 후보다.** [[57]]: *반복 억제는 '횟수'가 아니라 '인자 변화'로*.
  C3의 **억제 자격**(정책 축자 또는 사전등록 계량) 규칙을 적용하면 자격을 못 댈 가능성이 높다.

## §2b ★arm-5 라이브 결과 — **계약은 끄고 켤 수 있는 단위가 아니다** [M]

`run_arm5_20260807.sh`로 T2_* **120개를 unset**하고 `T2_SOURCE`(C1)·`T2_ARBITRATE`(C3)만 남겨 돌렸다
(101·102 nt=1). 결과:

| 마커 | 발화 |
|---|---|
| `[T2_SOURCE]` | **0** |
| `[T2_ARBITRATE]` | **0** |
| `[T2_RESOLVE]` | **0** |
| 사이드카 파일 | **없음** |

**두 계약이 켜져 있는데 한 번도 발화하지 않았다.** 원인은 `t2_gate_patch.py:5390`:

```python
if (os.environ.get("T2_RESOLVE") == "1" and a2 is not None ...   # ← 바깥 조건
    ...
    if os.environ.get("T2_ARBITRATE") == "1":                     # ← C3가 그 안에
        if os.environ.get("T2_SOURCE") == "1":                    # ← C1이 다시 그 안에
```

⇒ **C1·C3는 계약이 아니라 `T2_RESOLVE`라는 기존 레버 안의 가지다.** 껍데기를 끄면 함께 죽는다.
계약이라는 이름의 **독립된 코드 단위가 존재하지 않는다** — 그래서 "레버 5개만"은 플래그 조작으로는
도달 불가이고, **C1~C5를 각각 자기 진입점을 가진 모듈로 세우는 구현**이 선행되어야 한다.
(설계서 §3이 이미 그렇게 적어 뒀다: `t2_precedence.py`·`t2_arbitrate.py`·`t2_offload.py` **신규**.)

**부수 소득 — 이 런은 [[57]] 부정통제로는 유효하다**(우리 층 전체 OFF):
101 gold 2종 **전부 누락** · 102 `Sky Blue` 누락 · DB **0/2** · 제출이 전부 신용카드 계열로 붕괴
(`Business Bronze/Gold/Silver Rewards Card`·`Green Fee-Free`). 97 레버가 무언가는 지탱하고 있었다.
⚠n=2라 [D]. 바닥(59/93)과 비교 불가.

---

## §3 그래서 "5개만"의 실행은 이것이다

배정이 맞더라도 **플래그 97개가 5개가 되지는 않는다** — 배정은 *분류*이지 *합병*이 아니다.
합병이 되려면 각 계약이 **하나의 술어 + A2 데이터**로 서고, 개별 플래그가 사라져야 한다.

따라서 검정은 하나다:

```
arm-7 :  C1 SOURCE · C2 PRECEDENCE · C3 ARBITRATE · C4(무플래그) · C5 OFFLOAD
         C6 WINDOW · **C7 DISCOVERY**(DISCOVERY_NAMES·UNCALLED_UNLOCK·VERDICT_SURFACE·
                       TRANSFER_LEAVES_STEPS·MATCH_COUNT)
         나머지 OFF · 설정 유지 + 호스트(GATE_REGEN·PROV_REGEN)
★arm-5(C7 없음)는 **63.4%를 만든 축을 꺼 놓고** 돌린 것이라 바닥 비교가 성립하지 않았다(DB 0/2).
사전등록 바닥 : action_match ≥ 59/93 (63.4%) — 설계서 §6b(a)
```

- 바닥을 지키면 **5개로 선다**가 [M]이 된다.
- 무너지면 **어느 계약이 무엇을 못 흡수했는지**가 그 자리에서 드러난다 — 그것이 다음 설계 입력이다.

⚠현재 C2(`t2_precedence`)·C5(`t2_offload`)는 **모듈이 없다**(설계서 §3: C2 부분·C5 미착수).
arm-5를 돌리려면 그 둘을 먼저 세워야 하고, 그것이 지금 시점의 진짜 다음 작업이다.
