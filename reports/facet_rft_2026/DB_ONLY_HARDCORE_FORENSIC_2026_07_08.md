# DB-only 재선별 hard-core · 스텝별 per-case 정독 (2026-07-08 밤·무료)

> 상위 = `RESEARCH_MASTER.md`. 지시: handoff `HANDOFF_2026_07_08.md` §0.2 —
> **gold action_checks가 아니라 DB-only 기준**으로 hard-core를 재선별하고, `operand 정밀도(+10)`·`over-action(+9)` 를 per-case 정독.
> 데이터: `sim_results/asmregen32b_regen_retail_t4.results.json.gz`(456 sim) vs tau2 공식
> `o4-mini-2025-04-16_retail_default_gpt-4.1..._4trials.json` · `gpt-4.1-2025-04-14_retail_default_..._4trials.json`.
> 기준: `reward_info.db_check.db_match` (C19가 확인한 **유일한 공통 채점축**).
> 재현 스크립트: `scripts/distill/tau2/dbonly_forensic.py` (본 커밋 동봉).

### 0.0 유효성 선검사 ([[08]] 게이트 — 결론 전 필수)
| arm | sim | termination_reason | infrastructure_error | hallucination_retries |
|---|---|---|---|---|
| ours | 456 | `user_stop` 456/456 | **0** | 0 |
| o4-mini | 456 | `user_stop` 456/456 | **0** | 0 |
| gpt-4.1 | 456 | `user_stop` 456/456 | **0** | 0 |

⇒ crash·infra·max_steps 오염 **없음**. 아래 계수는 전부 **4-trial sim 단위 전수**이며 pass^1 점추정이 아니다.
hard-core 선별은 `≥3/4 ∧ ≤1/4` = robust 기준.

---

## 0. 결론 요약 (전부 [M]·단일 clean run·456 sim)

1. **DB-only hard-core = 7 task**(기존 reward기준 10에서 3 탈락). **역방향 0 task**는 DB 기준으로도 유지.
2. **`operand 정밀도 +10`은 단일 축이 아니다.** 클래스별로 쪼개면 **⋈(+8)·item집합 혼합(+6)·주소(+5)** 가 잔여이고,
   **F2 변형선택(−4)·payment_method(−11)** 는 **우리가 앞선다**(단 §5 선택편향 주의).
3. **★free-text 주소 날조 = ours 5 / o4-mini 0 / gpt-4.1 0**, 5건 전부 db_fail, **passing-spurious 0**.
   환경이 id 날조는 32/32 거부하지만(C11) **free-text 날조는 거부할 수 없다** ⇒ **E9의 NO-GO는 free-text로 확장되지 않는다.**
4. **★over-action 12건 중 10건은 "안 시킨 write"가 아니라 "시켰지만 정책상 불가능·이미 철회된 write"**(decidable precondition).
   순수 unrequested scope 과확장은 **2건**. ⇒ [[06]]의 "over-action 게이트 금지" 선례가 겨눈 대상과 **다르다**.
5. **★`§1.7 #4` (repair 최우선·"all-errored 16 vs 1 = +15") 근거 붕괴.** 공통 버킷(=성공 write 0)으로 재면 **24 vs 20 = +4**.
   o4-mini의 `never-attempted 19`를 빼고 비교한 프레이밍 아티팩트였다.
6. **★tau2-bench 코드 버그**: `modify_pending_order_items`가 DB를 **인자 리스트 순서에 의존**하게 만든다(§4). 우리 1 sim 피해.
7. **reason enum 불일치는 arm 공통 노이즈**(ours 8 / o4 5 / g41 4), 절반은 **user-sim이 gold와 모순**. 레버 아님.

---

## 1. DB-only hard-core 재선별

선별식: `ours db_pass ≤1/4 ∧ o4-mini ≥3/4 ∧ gpt-4.1 ≥3/4`, `db_pass ≡ db_check.db_match is True`.

| task | ours | o4-mini | gpt-4.1 |
|---|---|---|---|
| t17 | 0/4 | 4/4 | 4/4 |
| t37 | 0/4 | 4/4 | 3/4 |
| t57 | 1/4 | 4/4 | 3/4 |
| t63 | 1/4 | 3/4 | 4/4 |
| t86 | 1/4 | 3/4 | 4/4 |
| t91 | 1/4 | 4/4 | 3/4 |
| t111 | 1/4 | 3/4 | 3/4 |

**탈락 3 task** (기존 `HARDCORE_STEP_FORENSIC §1`의 reward기준 core):

| task | DB 기준 | 함의 |
|---|---|---|
| t40 | ours **4/4** · o4 4/4 · g41 4/4 | DB 완전 통과. 순수 NL 실패 ⇒ **C19 채점기준 불일치 구역** = 비교 불가 |
| t68 | ours **4/4** · o4 4/4 · g41 4/4 | 동일 |
| t105 | ours **2/4** | ≤1/4 미충족 |

⇒ **`HARDCORE_STEP_FORENSIC` 표의 N4(도구 거부·t105)와 NL 사례(t40·t68)는 DB 기준선 밖**이다.
N4는 여전히 all-errored 16의 근인 후보이나 **hard-core 근거로는 쓸 수 없다**.

## 2. ★방법 교정 — index-pairing 아티팩트 (자기교정 #9)

gold write열과 실제 write열을 **인덱스 순서로 짝지으면** 같은 집합을 다른 순서로 실행한 sim이 전부 "⋈ 틀린 주문"으로 오분류된다.
예) t76 tr2: gold `[cancel #W8367380, cancel #W1242543]` vs ours `[cancel #W1242543, cancel #W8367380]` = **동일 집합**.

**최소-diff 순열 매칭**으로 교정한 결과:

| | index-pairing | 최소-diff 매칭 |
|---|---|---|
| ⋈ order_id | 28 | **23** |
| "필드 전부 일치인데 DB fail" | 14 | **1** |

⇒ 13건은 순열 아티팩트였다. **집계된 원인표는 짝짓기 규칙에 민감하다**([[08]]).

## 3. SAME-count 버킷(=`operand 정밀도`)의 클래스별 전수 분해

대상: `db_match=False ∧ gold write 수 == 성공 write 수 ∧ gold write ≥1`. ours 70 sim / o4-mini 60 / gpt-4.1 56.
단위 = **write 쌍**(sim당 여러 개 가능).

| 클래스 | ours | o4-mini | gpt-4.1 | Δo4 | Δg41 |
|---|---|---|---|---|---|
| **⋈ order_id (틀린 주문)** | 23 | 15 | 10 | **+8** | **+13** |
| **item_ids+new_item_ids 혼합** | 9 | 3 | 3 | **+6** | **+6** |
| **주소 필드** | 8 | 3 | 2 | **+5** | **+6** |
| ★op(연산자) 불일치 | 10 | 7 | 17 | +3 | −7 |
| item 누락 | 2 | 0 | 1 | +2 | +1 |
| item 오선택(동수) | 2 | 1 | 0 | +1 | +2 |
| reason enum | 6 | 5 | 4 | +1 | +2 |
| 전부 일치인데 DB fail(§4 버그) | 1 | 0 | 0 | +1 | +1 |
| item 과포함 | 1 | 1 | 0 | 0 | +1 |
| **F2 변형선택(new_item_ids)** | 17 | 21 | 19 | **−4** | −2 |
| **payment_method** | 3 | 14 | 6 | **−11** | −3 |
| **합계** | 82 | 70 | 62 | **+12** | +20 |

- **잔여의 이름은 `precision`이 아니라 `⋈ + 집합범위 + 미조회 날조`이다.** "정확히 그 값"(F2 변형·payment)은 우리가 오히려 앞선다.
- `op 불일치` 10건은 per-case 정독 결과 **연산자 선택 실패가 아니다**(§6).

## 4. ★tau2-bench 버그 — `modify_pending_order_items`의 variant leak

`src/tau2/domains/retail/tools.py:531-537`
```python
for item_id, new_item_id in zip(item_ids, new_item_ids):
    item = next(item for item in order.items if item.item_id == item_id)
    item.item_id = new_item_id
    item.price   = variant.price      # ← variant는 앞 검증 루프의 '마지막' 값
    item.options = variant.options    # ← 동일
```
`variant`가 앞 루프에서 새므로 **수정된 모든 item이 마지막 변형의 price/options를 받는다**.
`db_match`는 `get_dict_hash(db.model_dump())` 해시 동등성(`utils/utils.py:44`, `json.dumps(sort_keys=True)`)이므로
**의미적으로 동일한 write라도 `item_ids` 나열 순서가 gold와 다르면 DB 해시가 달라져 fail**한다.

- 실측 피해: ours **1 sim**(t100 tr2 — 쌍은 gold와 동일, 순서만 반대) · o4-mini 0 · gpt-4.1 0.
- **격차 설명력 없음(1/456)**. 다만 벤치 자체의 결함이므로 기록하고 upstream 보고 후보로 남긴다.

## 5. ★F2·payment의 "우세"에 붙는 선택편향 (banking 금지)

§3의 클래스 계수는 **성공한 write**만 본다. 우리는 `성공 write 0`인 sim이 24개(o4-mini 20), 그중 16개는
**모든 write 시도가 ERROR**(C15)다. 이 sim들의 변형선택 오류는 **계수에 나타나지 않는다**.
⇒ **"F2 변형선택은 우리가 앞선다"로 banking 금지.** 올바른 진술:

> **성공한 write에 한해** F2 변형 오류는 ours 17 < o4-mini 21이며, 이는 C13(`p_traj` .762 vs frontier .908 = 15pp 능력격차)와
> **긴장 관계**에 있다. C13의 프로브는 결정 지점 전체를, 본 계수는 성공 write만 센다. **E1′ Phase A가 이 긴장을 판정한다.**

## 6. over-action 12건 per-case 재분류

| sim | gold write | 실제 | 무엇이 일어났나 | 분류 |
|---|---|---|---|---|
| t57 tr0 | 0 | cancel 전체주문 | 사용자: *"기프트카드 환불 안 되면 **취소하지 마라**"* → 취소함 | **철회 무시** |
| t57 tr1 | 0 | cancel 전체주문 | 동일(명시 철회) | **철회 무시** |
| t57 tr3 | 0 | cancel 전체주문 | 사용자 요청했으나 **기프트카드 환불=정책 불가** | 정책-불가능 수행 |
| t10 tr1 | 0 | return ×2 | 사용자: 두 주문 환불수단 **교차** 요구(정책 불가) → 그냥 실행 | 정책-불가능 수행 |
| t12 tr3 | 0 | return | 사용자: PayPal 환불 아니면 **상담원 연결** → 실행 후 escalate 실패 | 정책-불가능 + F5 |
| t99 tr1 | 2 | +cancel 전체주문 | 사용자: *"스케이트보드만 취소"* → **주문 전체 취소**(부분취소 불가) | 정책-불가능 수행 |
| t99 tr3 | 2 | +cancel 전체주문 | 동일 | 정책-불가능 수행 |
| t32 tr0 | 3 | +return | 분실 태블릿 반품(gold=불가) — 사용자가 "시도해봐"라 함 | 정책-불가능 수행 |
| t43 tr3 | 1 | +exchange | gold=주소변경만. 사용자 요청은 있었음 | 대상 불일치 |
| t54 tr2 | 3 | +return `item_ids: []` | **빈 리스트 write를 환경이 수용** | degenerate write |
| t79 tr1 | 1 | +modify(다른 주문) | **⋈ 틀린 주문**에 먼저 write | ⋈의 부산물 |
| **t111 tr3** | 3 | +modify_address(두 번째 주문) | 같은 주소를 **요청 안 한 주문에도** 적용 | **순수 scope 과확장** |

**집계: 정책-불가능/철회 수행 8 · ⋈·대상불일치 2 · degenerate 1 · 순수 unrequested 1.**

> [[06]]/`NEXT_DET_LEVERS:131` "do NOT gate over-action" 선례는 **LLM scope residual**(=t111형, 1/12)을 겨눈 것이다.
> 실측 다수(8/12)는 **정책 precondition 위반** — `§1.5 Q1` 상 **decidable ∧ 미집행**이다.
> ★단 C12를 잊지 말 것: decidable이 곧 이득은 아니다. **차단만 하면 write가 안 일어나 여전히 fail**인 경우와,
> **환경이 이미 막고 있는** 경우를 per-case로 갈라야 한다. t57/t99/t10은 gold write=0이므로 **차단이 곧 정답**이다(예외적).

## 7. ★free-text 날조 — E9가 죽인 적 없는 통로

전수 계수(주소 write 시점 이전 문맥에 `address1` 문자열이 존재했는가):

| arm | 주소 write | 문맥에 없던 `address1` | 그중 db_fail |
|---|---|---|---|
| **ours** | 128 | **5 (4%)** | **5 (100%)** |
| o4-mini | 105 | **0** | 0 |
| gpt-4.1 | 126 | **0** | 0 |

사례: t17 tr0-3 `"123 Elm St"`(gold `"123 Elm Street"`) · t39 tr0 `"123 Palm Tree Lane"`.

**t17 전수 정독**: msg0–msg7 어디에도 "Elm"이 없다. 에이전트는 `find_user_id_by_name_zip`만 호출해 id를 얻고,
주문·유저를 **조회하지 않은 채** `address1/city/state/country`를 통째로 생성했다(zip만 사용자 발화). 4/4 동일 = systematic.

- ⇒ `HARDCORE_STEP_FORENSIC` **N1 "값 충실도(약어)" 철회**. 약어는 날조의 부산물이지 verbatim 실패가 아니다.
- ⇒ **C11/C12 정련**: 환경은 *id* 날조를 32/32 거부하지만 **free-text 날조는 타입상 거부할 수 없다**. 여기서는 날조가 **진짜 근인**이다.
- ⇒ **C16 정련**: "present가 주문정보를 주입하므로 읽기 불필요"는 *읽었을 때* 성립. t17은 present가 발화하지 않았고
  **정보가 실제로 부재**했다. "도구호출 부재 ≠ 정보 부재"는 **평균의 진술**이지 per-case 진술이 아니다.

**E9′ 후보 (E9와 다름)**: `write operand의 free-text 값은 이전 문맥(도구 출력 ∨ 사용자 발화)에 존재해야 한다`.
- decidable(문자열 포함) · 미집행(환경이 타입 검사 없음) · **frontier 위양성 0/231** · **우리 passing-spurious 0**.
- 처방은 **차단이 아니라 repair**: 위반 시 `get_order_details`/`get_user_details`를 강제하고 재발화(= §1.5 Q5의 **읽기 강제만 안전**과 정합).
- **정직한 상한 = 5 sim = 1.1pp** (그중 4건이 t17 한 task). 작다. 그러나 무료·부작용 0·hard-core 1개를 직접 연다.

## 8. reason enum = arm 공통 노이즈

실행된 `cancel_pending_order`(gold에도 있는 주문) 중 `reason` 불일치:

| arm | 실행 | 불일치 | 그중 **사용자 발화가 우리 값을 지지** |
|---|---|---|---|
| ours | 93 | 8 | 4 |
| o4-mini | 84 | 5 | **5 (전부)** |
| gpt-4.1 | 97 | 4 | 1 |

- t38 tr0 사용자 원문: *"Ordered by mistake. Please cancel it now."* ↔ gold `reason="no longer needed"`.
- t76 tr3 사용자 원문: *"The reason is that I no longer need it."* ↔ gold `reason="ordered by mistake"`.

⇒ **user-sim(gpt-4.1)이 gold와 모순되는 발화를 하면, 사용자에게 충실한 에이전트가 DB fail**한다.
o4-mini는 5/5가 이 경우다. **격차 설명력 Δ+3(vs o4) / +4(vs g41)** 이고 절반이 하네스 노이즈 ⇒ **레버 아님**.
(이것은 pass^k 표본분산의 알려진 원천과 같은 계열이다.)

## 9. ★`§1.7 #4` 재검 — repair 레버의 상한 (자기교정 #10)

C15는 `"모든 write 시도가 ERROR" 16 vs 1 = +15`를 **최대 조각**으로 지목했다. 그러나 o4-mini는 같은 실패를
`never-attempted 19`로 낸다. **공통 버킷 = "성공한 write 0"** 으로 재면:

| | ours | o4-mini | Δ |
|---|---|---|---|
| ZERO write (never-attempted + all-errored) | **24** | **20** | **+4** |

⇒ **"repair가 최대 조각"은 비교 프레이밍 아티팩트.** DB-only 격차 +23(vs o4-mini)의 실제 구성:

| 조각 | Δ |
|---|---|
| **over-action (MORE+EXTRA)** | **+9** |
| ⋈ order_id | +8 (write쌍) |
| item집합 혼합 | +6 |
| 주소 날조·필드 | +5 |
| ZERO write (repair 대상) | +4 |
| FEWER (미완) | **0** |
| payment_method | **−11** |
| F2 변형선택 | **−4** |

## 10. ★gather-before-act — 프롬프트 천장과 진짜 상한 (사용자 제기·2026-07-08 밤)

### 10.1 프롬프트로는 안 된다 (측정됨)
`data/tau2/domains/retail/policy.md:18` 이 **이미** 명령한다:
> *"You should not make up any information or knowledge or procedures not provided by the user or the tools"*

같은 시스템 프롬프트 아래 순수 날조(도구출력·사용자발화 어디에도 없고 DB에도 없는 값):
**ours 91 / o4-mini 1 / gpt-4.1 3** (write id-값 2597 / 2079 / 2366 중).
⇒ **[[42]] scale-emergent prior-override의 우리 데이터 내 실증.** 금지문은 32B에서 작동하지 않는다.
또한 날조된 값 **93/93 전부 환경이 ERR로 거부**(수용 0) ⇒ **C12 재확인**: 날조를 *막는* 것은 이미 되고 있다.

### 10.2 ★자기교정 #12 — 잘못된 gather 지표
처음에 `gather-recoverable`을 **"날조된 그 문자열이 DB에 있나"** 로 쟀고 4/2597(0.2%)를 얻어 "gather는 살 게 없다"고 결론지을 뻔했다.
**틀렸다.** t37의 `6117189162`는 실재 id `6117189161`에 **+1** 한 값이다. 모델이 *원한* 값(그 제품의 진짜 변형)은 DB에 있다.
올바른 술어: **날조 시점에 그 제품의 변형 목록을 조회했었나.**

### 10.3 올바른 측정 (`new_item_id` 값 단위)
| arm | 실재 변형 선택 | **미조회 상태로 날조** | 조회했는데도 날조 |
|---|---|---|---|
| **ours 32B** | 374 (85.2%) | **63 (14.4%)** | 2 |
| o4-mini | 340 (99.7%) | **1 (0.3%)** | 0 |
| gpt-4.1 | 416 (100%) | **0** | 0 |

reads/sim: ours **3.59** · o4-mini 5.32 · gpt-4.1 5.92.
⇒ **frontier는 조회하지 않은 operand를 사실상 발화하지 않는다.** 날조는 gather 실패의 *하류 증상*이다.

### 10.4 sim 단위 상한
`미조회 날조 ≥1`인 sim = **28 / 456 (6.1%)** — 그중 **db_fail 19** (write는 했으나 fail 16 · zero-write 3) · **db_pass 9**.

- **정직한 상한 = 19 sim = 4.2pp.** vs o4-mini DB격차가 23 sim(5.0pp)이므로 **격차의 ~83%가 이 축 위에 있다.**
- **db_pass 9 sim은 Δspurious 위험**: 날조했지만 환경 거부 후 회복해 통과했다. 게이트가 여기서 발화한다 ⇒ 반드시 계측.
- **차단만으론 부족**(C12 · E9 PhaseA: 날조 12/15는 복구 후에도 실패). 게이트는 **변형 목록을 *공급*** 해야 한다(repair).
- 대상 task: `{17:4, 37:3, 103:3, 20:2, 74:2, 98:2, 111:2, 0·6·36·39·52·66·85·86·99·107:1}`

### 10.5 present vs gather — 학습 신호의 관점
`present`는 정보를 **주입**한다: grounding을 사고 **읽기 주도성을 판다**(C16·reads 3.59 vs frontier 5.9).
그런데 그 부작용은 성능만이 아니다 — **present는 궤적에서 `read → act` 시퀀스를 지운다.**
따라서 **learn-wing(P4·C7)이 배워야 할 바로 그 감독 신호를 present가 파괴한다.**
⇒ 학습된 TBox를 목표로 한다면(=[[00]]·[[11]]), **주입(present)보다 강제 조회(gather)가 정합적**이다.

> 단 `§1.5 Q1`은 여전히 **결정론 scaffold 먼저**를 지시한다(decidable ∧ 미집행). 두 경로는 경쟁이 아니다:
> **결정론 gather 게이트가 위반 궤적을 라벨링해 주고, 그것이 E6(learn) 의 감독 신호가 된다.**

### 10.6 신설: E11 gather-before-act 게이트
술어(도메인-일반): *write 인자 중 entity-id 타입 값은, 그 값을 산출한 선행 read 결과에 provenance가 있어야 한다.*
위반 시 **차단이 아니라** 해당 타입의 열거 read를 강제 호출하고 재발화.
- decidable ∧ 미집행 ⇒ `§1.5 Q1` 통과. A2 = `{write arg type → 열거 read tool}` 매핑만 도메인별([[05]] 준수).
- **GO 조건**: db_pass 9 sim에서 **Δspurious ≤ 0** ∧ over-block 0 ∧ 턴예산 초과 0 (제1원리: 강제 조회는 **턴·문맥을 판다**).
- **E9′(free-text provenance)를 흡수**한다(주소 5 sim은 같은 술어의 특수경우).

## 11. 미결 · 다음

- **E1 Phase B (실행중)**: 완결/persistence 게이트 closed 판정 + Δspurious. §6이 시사하는 바 — 게이트가 겨눠야 할 것은
  coverage가 아니라 **precondition(정책 불가능성)** 일 수 있다.
- **E9′ 오프라인 스모크(무료)**: free-text provenance 위반 검출기 + 강제 read repair. GO 조건 = per-case 복구 ∧ over-block 0 ∧ Δspurious ≤ 0.
- **E1′ Phase A**: §5의 긴장(성공-write 기준 F2 우세 vs C13 15pp)을 판정. 우선순위는 §3·§9에 비추어 **하향 검토**.
- **Step 3 (QwQ nt=4, 유료·실행중)**: 사용자 가설 *"precision ↔ over-action = thinking 트레이드오프"* 검정.
  현 증거는 **반대**(thinking은 둘 다 개선·persistence를 판다). §6이 over-action의 실체를 정책위반으로 바꾸었으므로 재해석 필요.
- **t34/t36 재검**: 두 sim의 `cancel`은 에이전트 op 선택 오류가 아니다 — t36은 `"..._cheapest"` placeholder **날조** 후
  사용자가 전체취소를 요청했고, t34는 user-sim이 gold(주소변경)와 무관하게 진행했다. **user-sim 이탈**의 별도 census 필요.
