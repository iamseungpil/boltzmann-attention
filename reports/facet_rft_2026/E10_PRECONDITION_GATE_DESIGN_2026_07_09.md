# E10 — 정책-precondition 게이트 설계·타당성 (2026-07-09)

> 등대 `RESEARCH_MASTER.md` §4 큐 E10(2순위·무료→소액) 파생. 닫는 것 = **C25 over-action +9**(DB-only 격차 최대 조각).
> 규율: [[08]] per-case 先·[[05]] 도메인-일반 gate_spec·[[06]] scope-게이트 금지 선례·[[13]] 게이트 증식 경계·모트 §1.3 **Δspurious≤0 필수**.
> **★상태(2026-07-09 확정): §5 무료 격리검증 결과 = NO-GO. 유료런 착수 안 함([[09]] 절약). 상세 §5.1·§6.**

---

## 0. 한 줄
over-action(+9)의 실체는 "안 시킨 write"(scope 잔여)가 아니라 **"시켰지만 정책상 불가능·이미 철회된 write"**(C25: 8/12).
이는 `§1.5 Q1` 상 **decidable ∧ 미집행**(환경이 안 막음)이므로 결정론 precondition 게이트의 후보다. **단 절반은 ⋈-coupled·compound**여서
순수-decidable 슬라이스는 작고, 검증된 `refund_target` 규칙(Lever A·2026-06-27·DB 5/5)의 일반화로 재유도 없이 얹는다.

## 1. 근거 (C25 per-case · 정본 `DB_ONLY_HARDCORE_FORENSIC §6`)

over-action 12건 재분류: **정책-불가능/철회 수행 8 · ⋈·대상불일치 2 · degenerate 1 · 순수 unrequested 1.**

| sim | gold write | over-action | 술어 유형 | decidable? | 환경 집행? |
|---|---|---|---|---|---|
| t57 tr0/1 | 0 | cancel 전체 | 사용자 명시 철회("기프트카드 환불 안 되면 취소 마라") | **예**(refund-eligibility→withdrawal) | ✗(실행됨·db_fail) |
| t57 tr3 | 0 | cancel 전체 | 기프트카드 환불=정책 불가 | **예**(refund-eligibility) | ✗ |
| t10 tr1 | 0 | return ×2 | 교차-주문 환불수단(정책 불가) | **예**(refund-target) | ✗ |
| t12 tr3 | 0 | return | PayPal 환불 불가→escalate | **예**(refund-eligibility)+F5 | ✗ |
| t99 tr1/3 | 2 | +cancel 전체 | "스케이트보드만 취소"→전체(부분취소 불가) | **compound**(tool-semantics ∧ ⋈-intent) | ✗ |
| t32 tr0 | 3 | +return | 분실 태블릿 반품(status 불가) | **예**(returnable-status) | ✗ |
| t43 tr3 | 1 | +exchange | 대상 불일치 | ⋈ | — |
| t54 tr2 | 3 | +return `[]` | 빈 리스트 write | degenerate | 수용 |
| t79 tr1 | 1 | +modify(다른 주문) | ⋈ 틀린 주문 | ⋈ | — |
| t111 tr3 | 3 | +modify_address(요청 안 한 주문) | 순수 scope 과확장 | ✗(semantic scope) | — |

**핵심 판정 3가지:**
1. **decidable ∧ 미집행 확정**: 8건 전부 나쁜 write가 *실제 실행*돼 db_fail → **환경이 안 막는다**(id 날조의 C12 함정과 반대·환경은 id를 32/32 거부하나 이 술어들은 통과시킴). ⇒ 결정론 게이트가 **비-redundant**.
2. **차단=정답 확실 슬라이스**: t57·t10(gold write=0)은 해당 write가 gold에 없으므로 **차단이 곧 정답**(upside 확실·예외적). t99·t32(gold>0)는 *특정 나쁜 write만* 차단하고 정답 write는 보존해야 upside.
3. **[[06]] 선례와 다른 술어**: [[06]]/`NEXT_DET_LEVERS:104` "over-action 게이트 금지"는 **t111형 scope 잔여**(1/12·semantic)를 겨눈 것. E10은 **정책 precondition**(decidable)을 겨눈다. **⇒ t111형은 E10 대상 아님**(재유도 금지).

## 2. 술어 taxonomy → 순수-decidable vs compound (정직 분리)

| 술어 | 정의 (decidable predicate) | 근거 | 케이스 | 등급 |
|---|---|---|---|---|
| **P1 refund-target** | 환불 `payment_method_id` ∈ 해당 주문 `payment_history` 원결제 ∪ 사용자 gift-card | Lever A·**DB 5/5 MATCH 검증됨**(2026-06-27) | t10·t12·t57 | **순수-decidable**(단 gift-card over-block 위험) |
| **P2 status-eligibility** | action(return/cancel/modify)이 주문·항목 status ∈ 허용집합일 때만 (예: delivered→return·pending→cancel) | tau2 policy·DB status | t32 | **순수-decidable** |
| **P3 withdrawal-conditional** | 사용자 "X 안 되면 Y 마라"의 X가 P1/P2로 false → Y 차단 | t57(=P1 false→cancel 철회) | t57 | **decidable**(P1/P2에 환원) |
| **P4 partial-cancel** | cancel은 전체-주문 단위인데 사용자가 부분을 지목 | tool-semantics ∧ **어느 항목=⋈/NL** | t99 | **compound → 게이트 아님·ASK** |

**⇒ 순수-decidable = {P1, P2, P3} (t10·t12·t57·t32) · compound = {P4} (t99) → ASK/cardinality로 라우팅(하드 게이트 금지·over-block).**

## 3. 설계 (재유도 없이·기존 엔진 위에)

### 3.1 엔진 (FIXED·[[05]])
`t2_gate_patch.py`의 GateInterpreter는 **실행-전 tool-call 검사 → deny 시 replay-safe regen**(`apply_gate_regen`·K=MAX_REGEN·R8 종단·`num_errors++` 예산)이 이미 존재. `T2_GATE_KINDS` 화이트리스트로 kind 격리 측정 가능. **E10은 새 엔진이 아니라 새 kind 1개**(`precondition`)를 이 디스패치에 추가.

### 3.2 gate_spec (도메인-일반 kind + retail ABox 인스턴스)
```yaml
kind: precondition
applies_to: [cancel_pending_order, return_delivered_order, exchange_delivered_order, modify_pending_order_payment]
checks:
  - name: refund_target            # P1
    when: tool in {return_*, exchange_*}  # 환불수단 인자 존재
    require: refund_pm ∈ fetch_record(order_id).payment_history.originals ∪ user.gift_cards
    on_fail: DENY("refund method not on this order's payment history; do not invent/route cross-order")
  - name: status_eligibility       # P2
    require: fetch_record(order_id).status ∈ allowed_status[tool]
    on_fail: DENY("order/item status does not permit this action")
  # P3(withdrawal)는 P1/P2 fail을 사용자 조건과 AND — 별도 상태 불요(P1/P2가 이미 차단)
```
- **`fetch_record`만 사용**(허용 resolver·gate1 통과·**새 join-resolver 금지**·[[05]]). 도메인-특이 값(허용 status 집합·gift-card 정책)은 **ABox(gate_spec)** 에, 엔진은 리터럴 0.
- **P4는 gate_spec에 넣지 않는다** — cardinality-mismatch(사용자가 항목 지목 ∧ cancel=전체)면 **ASK 트리거**(기존 escape-ask 경로)로 라우팅. 하드 deny는 over-block(정당한 전체취소를 막음).

### 3.3 모트 §1.3 필수 계측 (부작용 없는 레버는 없다)
게이트 자신이 **정당한 write를 over-block**할 수 있다(특히 P1 gift-card = Lever A shelved 사유). 따라서 GO 판정에 **의무**:
- **over-block = 0** (전수·정당한 환불/취소를 deny한 건수). gift-card 정당 환불·정상 status 액션이 표적.
- **Δspurious ≤ 0** (게이트 켠 arm의 신규 파손 ≤ 0·per-case).
- **turn-예산 초과 0** (deny→regen이 `too_many_errors`를 늘리지 않는가·C38 교훈).

## 4. ★타당성 한계 (재유도 방지 — Lever A 교훈 계승)

1. **⋈-coupling**: `NEXT_DET_LEVERS §Lever A`가 실증 — PAYMENT 잔여의 다수는 **⋈의 하류 증상**(틀린 주문 선택→틀린 카드). precondition 게이트는 *주문이 정해진 뒤* 일관성을 강제하나 **order-choice는 못 고친다**. ⇒ 게이트가 틀린 주문 위에서 "일관"되게 통과시키면 여전히 fail. **마진 상한 < 8.**
2. **gift-card over-block 위험**: P1은 정당한 gift-card 환불을 막을 수 있어 Lever A가 shelved된 직접 사유. 계측 없이는 net-음성 가능.
3. **소표본**: 순수-decidable 슬라이스 = 4 task(t10·t12·t57·t32)·~6 sim-trial/456. 헤드라인 이동은 작다(over-action +9는 격차 *구성*이지 pass 상한 아님).
4. **decidable ≠ 이득(C12)**: 차단만 하고 정답 write가 안 일어나면 여전히 fail(t12=block+escalate 필요·F5). block 단독으론 t12 미해결.

## 5. ▶ 무료 격리검증 (paid full-run 前 게이트·[[09]])

**목적**: gate_spec를 코딩·유료런 하기 *전에*, 기존 궤적에서 (a)술어가 실제로 fire하고 (b)over-block 0인지 확정.

**절차(전부 GPU-free·기존 sim json + `db.json`)**:
1. 4 task(t10·t12·t57·t32)의 over-action write를 floor/ours sim에서 추출(도구·인자·order_id).
2. P1/P2 술어를 `db.json`으로 결정론 평가 → 표적 write가 **DENY로 판정되는가**(true-positive).
3. **동일 arm의 정당한 return/exchange/cancel 전수**에 같은 술어 적용 → **over-block(false-positive) = 0인가**(특히 gift-card 환불·정상 status).
4. 판정: TP≥표적 ∧ FP=0 → **BUILD**(gate_spec 코딩→smoke→유료 nt=4). 아니면 **NO-GO**(Lever A처럼 특성화·map).

**GO 조건**: (per-case 표적 차단 확인) ∧ (over-block=0) ∧ (⋈-독립 슬라이스 ≥ 2 task). **미달 시 = 경계 지도**(over-action 잔여는 ⋈+scope로 귀속·게이트 아님).

### 5.1 ★결과 (2026-07-09 시행·NO-GO) — provenance 영속

**소스**: arm ours=`sim_results/asmregen32b_regen_retail_t4.results.json.gz` · `retail/db.json` · 스크립트 `scripts/distill/tau2/e10_precond_probe.py`(`--stage cases|scan`·GPU-free·기준 = `reward_info.db_check.db_match`).

**(a) per-case 정독 ([[08]] 궤적)** — over-action 5 task(t10·t12·t32·t57·t99) 전 trial:
| case | over-action write | DB 판정 | 술어 발화? |
|---|---|---|---|
| t10 tr1 (gold=0) | return ×2 | pm=credit_card_3124723·paypal_9497703 = **주문 pm_orig와 일치**·status=delivered | **P1·P2 둘 다 미발화** |
| t12 tr3 (gold=0) | return | pm=credit_card_3124723 = **주문 pm_orig 일치**·delivered | **미발화** |
| t57 tr0/1/3 (gold=0) | cancel | status=**pending(eligible)**·pm_orig=credit_card(gift-card 무관) | **미발화** |
| t32 tr0 (gold=3·+1) | return(분실 태블릿) | pm 일치·status=**delivered(eligible)** | **미발화**(불가능성=대화상 "분실") |
| t99 (gold=2·+cancel) | exchange/cancel | status=eligible·부분취소 불가 | **compound(⋈/intent)** |

⇒ **DB-결정론 술어(P1 refund-target·P2 status)가 표적 over-action의 0건에 발화**. 불가능성은 전부 DB-state가 아니라 **대화(철회·부분의도·"분실"·교차-라우팅)** 에 있다.

**(b) arm 전수 교차표 (대조군)**:
| 술어 | 측정 | 판정 |
|---|---|---|
| **P1 refund-target** | 총 refund write **269** · 위반&db_**fail**(TP)=**5**(전부 t99 exchange) · 위반&db_**pass**(over-block FP)=**6**(t52 ×4·t99 tr0 ×2) | **TP(5) < FP(6)·비판별**(t99 tr0=pass가 db_fail과 *동일* PM 패턴) |
| **P2 status-eligibility** | 실행 write 중 status-ineligible = **0/602** | **환경이 이미 집행**(C12·redundant·레버 아님) |

**판정 = NO-GO.** P1은 순-이득 음성(over-block > TP)이고 판별력 없음(t99 pass/fail 동일 PM). P2는 환경이 이미 집행(Q1b: enforced → 레버 아님). Lever A NO-GO(2026-06-27·payment 잔여=⋈ 하류)와 **동형 재확인**.

## 6. 결론 (2026-07-09 확정·NO-GO)
- **E10(DB-결정론 precondition 게이트) = NO-GO.** §5.1 실측: P1 순-이득 음성(over-block 6 > TP 5·비판별) · P2 환경 이미 집행(0/602·redundant).
- **재프레임 (C25 정련)**: over-action의 "precondition"은 **decidable하되 DB-state가 아니라 대화(policy+intent)** 에 산다 — 사용자 조건부 철회("X 안 되면 Y 마라")·부분 의도("스케이트보드만")·불가 설명("분실")은 전부 **semantic**(대화 상태 추적). ⇒ [[06]] "over-action=게이트 금지 축"이 **DB-게이트에 한해** 재확인된다. C25 §6의 "decidable ∧ 미집행"은 *policy-decidable*이지 *DB-decidable*이 아니었음(정련).
- **남는 처방 후보**(게이트 아님): (i) **대화-precondition을 추적하는 결정론 controller**(사용자 철회/조건을 상태로 유지·부하 축·§1.5 Q2) — 단 "어느 항목·무슨 조건"은 LLM formalize. (ii) **cardinality-mismatch → ASK**(사용자가 항목 지목 ∧ 도구=전체단위 → 되물음). 둘 다 신규 실험(E-controller/E-ASK)이지 E10 게이트 아님.
- Lever A(2026-06-27·refund-target)와 **동형 NO-GO** 재확인: payment/over-action 잔여는 대체로 **⋈·intent의 하류**이지 독립 결정론 술어가 아니다.

## 7. 원장·다음
- **원장**: 등대 §3에 신규 C-엔트리(E10 NO-GO) 등록·§4 큐 E10 상태 NO-GO. 수치 provenance = §5.1(스크립트 `e10_precond_probe.py`·arm asmregen32b·db.json).
- **다음 무료 후보**(별개 실험): E-ASK(cardinality-mismatch·C48 위계·단 clarification 벤치 필요) · 대화-precondition controller 설계(부하 축). **E10 자체는 종료**(재유도 금지).
