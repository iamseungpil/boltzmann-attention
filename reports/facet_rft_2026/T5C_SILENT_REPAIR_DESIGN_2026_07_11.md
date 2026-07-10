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
