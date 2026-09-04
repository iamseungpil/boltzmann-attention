# x737 — 다음 97 실행 계획서 + 설계서 (2026-09-04)

> **자리**: 이 문서는 정본이 아니다. 정본 작업 순서는 `x509_axis_queue_2026_08_24.json` 이고,
> 여기서 제안하는 두 수리는 그 큐의 **새 단계 후보**다. 프레임 LOCK 은 `RESEARCH_MASTER.md` §1.
> **범위**: 진행 중인 banking 97 캠페인이 끝난 **뒤**에 무엇을 고치고, 그 수리를 어떻게 97 태스크
> A/B 로 재는가. 캠페인 중에는 아무것도 배선하지 않는다([[54]]).

---

## ★ 리뷰 진입점 (2026-09-04 최종)

### ⛔⛔ 먼저 읽을 한 문단 — 이 문서의 결론이 바뀌었다

**이 캠페인은 어떤 것도 판정하지 못했다.** `§1e` 가 전역 부정통제를 돌린 결과다:

```
flip 바닥(scaffold 0개인 base 팔을 두 번 돌렸을 때 뒤집히는 비율)  = **18.8%** (16쌍)
  실증: task_008 (x599 1.0 -> x644 0.0) · task_012 (1.0->0.0) · task_017 (0.0->1.0)
  사용자 축자 "원래 25% 정도는 pass/fail 을 반복했다" 와 정합

회귀 10/42 = 23.8%   =>  flip 바닥 **안**.  P(X>=10) = 0.255 (vs 18.8%) · 0.629 (vs 25%)
그리고 결정적으로: **base-PASS 42 태스크가 전부 n_sims = 1** 이다
  => 42개의 단일 베르누이 시행 위에서 "회귀" 라는 서술은 만들어질 수 없다
회복 13/29 = 44.8% (base 바닥 대비 P=0.0012) 이지만 ours 팔 flip 43.2% 와 겹친다 => 미판정
부호검정 (10 잃고 13 얻음) **p = 0.678**  =>  팔 전체의 순효과는 **0과 구분되지 않는다**

188 기전 전수 검정 + 라벨 순열 300회:
  관측 p<0.05 = **6**   우연 기대치 = 7.4   순열에서 6 이상 나온 비율 = **0.200**
  => **어떤 기전도 우연 이상으로 pass/fail 을 가르지 못한다**
```

⇒ **`§1d`(회귀 태스크 단위 귀속)와 `§1c`·`§1b` 의 태스크별 판정은 "이 sim 에서 무슨 일이 있었나"
의 기록으로만 읽어라. *"우리가 깼다"* 의 근거로 쓰지 마라.**
⇒ **reward 를 근거로 살아남은 수리 후보는 지금 하나도 없다.** 남은 근거는 전부
[[23]](gold 무참조 위반) · [[64]](거절이 처방을 못 준다) · **계기 무결성** 쪽이다.

### 등급 (2026-09-04 최종)

```
[[23]]/[[64]] 근거로 유지 (reward 대응은 전부 미판정)
    D12  user_action_feedback 오부착 — t2_gate_patch.py:10524·10961
    D13  정책 근거 없는 [ORDER] 강제 — gate.json:4653 이 스스로 "정책 문장은 없다" 고 자백
    D14  ★★신설(1f · 029+048+**027** 통합) — **재생성 산출물이 쓰기 게이트 밖에서 커밋된다.**
         027 추가(1f-7): 같은 write 를 정상경로 5회 DENY 한 게이트를 searchexhaust 1발이 우회.
         `_ap_regen` 29채널 전부가 wtag 6종(WEV 포함)을 원리적으로 우회 + `_fab_only` 두 번째 문.
         029: 모델의 옳은 거절을 넛지 3연발이 뒤집어 **금지 write 5건 커밋**(오프라인 재실행 5/5 DENY).
         ⚠반사실 reward 예측 0 — 수리 근거는 [[25]](자기 금지 우회 금지)다. §1f-2
    D15  신설(약화) — deny 처방 <-> dedup 억제 상호모순 (039 · 인과 불성립 · §1f-2)
    D16  신설(1f-9 · 010) — **[ACTION] 선언 불완결(referent 무구속)**: 인자명만 주고 «누구의
         user_id 인가»를 안 묶어 유일한 §7-무장 fail 을 죽였다. CONFIRMED·선재(07-22~08-10 도입).
         수리 = env 축자 스니펫 동봉([[71]]/[[23]]/[[58]] 안전) · P10 격리 후 배선
    D8   ★**원인 확정 · 승격** — 이름이 빈 게 아니라 **출력 스키마가 소비부 이름과 어긋났다**
         (`f6224e26` 09-01: `what`->`claim` 개명 · pending 에서 `kind` 제거).
         전송 문면 **73/73 이 "None: None"** · unb_p>=1 **158/158** · 날짜 절벽 09-01↔09-03.
         처방은 «침묵» 이 아니라 **스키마를 소비부에 맞추기**(한 줄 급). §1c-5 D8 재작성
계기 (레버 아님 · 선행)
    D9   폐기 원문 원장 — **강화**. [BLOCKED] 희생자 이름이 영속 궤적 0/133 ⇒ D4 는 D9 없이 판정 불가
    D10  declaration failed — ⛔**강등 · 자연실험 철회 확정**. OFF 팔 n=0 이었고, 채점된 sim 만
         다시 재니 declfail 7 @0.5714 ↔ 정상 42 @0.5714 = **차이 정확히 0**(로컬 재현).
         남는 근거는 토큰 누수뿐. ⚠D8 과 혼동 금지 — D8 은 **파싱 성공 판**의 결손이다
표적 재정의 후 상향
    D4   [BLOCKED] -> **unlock 경로로 좁히면** 코퍼스 최대 신호와 겹친다
         (unlock_* 차단자 1/6 p=0.057  ↔  T2_UNCALLED_UNLOCK 3/11 p=0.018 · 독립 수렴)
    L2   recommend_formalize — T2_FB_VIEW 40/75 · 50.0% (Δ-24.3pp · p=0.036)
⛔ 강등
    D1(겨눈 기전이 pass 쪽 +29pp) · D2 · D6(자격 1런 · x548 --target 051 필수) ·
    D7(발화 12/12 이 회복 · reward 근거 없음) · D10
중립·측정 불가
    D3(본체 표본부족 · 계열은 강화) · D11(단위가 sim 이 아니라 재생성 호출) · L1(기대수익 0)
```

### 이 문서의 절

| 절 | 내용 | 상태 |
|---|---|---|
| §0 | 선행 확인 | 확정 |
| §1 | task_049 원인 진술 | 분모 주의(한 태그·n=2) |
| §1b | 실패 20건 + **자기 CONFIRMED 철회** | 기록으로만 |
| §1c | 새 실패 3건 · D7·D8·D9 출처 | 기록으로만 |
| §1d | 회귀 10건 per-step · D11·D12·D13 출처 · **D6 강등** | 기록으로만 |
| **§1e** | **기전 인구조사 + 전역 부정통제** | ★**판정의 근거는 여기다** |
| **§1f** | 새 실패 6건(029·039·048·060·063·084) 발견자+반증자 짝 포렌식 · **D14·D15 출처** · M1~M6 | 반증 통과분만 |
| §2~§8 | 후보·프로브·스모크·실험 규격·순서 | 1차 리뷰 반영 완료 · **§1e 로 재정렬 필요** |

### 2차 리뷰(2026-09-04) — 리뷰어 자기정정 2건을 내가 코드로 확인했다

| 리뷰어의 1차 주장 | 판정 | 확인한 근거 |
|---|---|---|
| `C2` *"221건 부호를 `mutation_diff` 로 지금 무료·결정론으로 매길 수 있다"* | **철회** | `t2_forensic.py:1183-1186` — `gold_mutations` 가 gold 를 `mut_key` 로 **중복 제거**(`if k in seen: continue`) ⇒ *"gold 가 같은 행을 두 번 요구"*(051형)를 **표현할 수 없다** |
| `B1` *"엔진이 `tool` 만 보나 `tool_any` 도 보나에 따라 갈린다"* | **불필요한 분기** | `t2_procedure.py:38 _tools_of` = `t = node.get("tool"); if t: return [t]; return list(node.get("tool_any") or [])` ⇒ **항상 동일 취급**. 내 3행 표의 *"tool 만 인정 → terminal 1"* 은 **존재하지 않는 구현 분기**였다 |

⇒ **P1 에서 이것을 찍을 필요가 없다.** `_tools_of` 가 유일한 해석 경로이고 `_satisfied`·`checklist`·
`next_step`·`absent_note` 가 전부 그것을 쓴다.

### ★★2차 리뷰를 적용하다가 원인이 하나 확정됐다 — **D8 (2026-09-04)**

리뷰어는 D8 의 명명이 흔들린다며 *"식별 불가"* 를 **kind 라벨 결손**으로 고쳐 부르고
«침묵 vs tool-렌더» 2팔 대조를 설계하라고 했다. 그 지시대로 실물을 파 보니 **명명도 처방도
둘 다 틀렸다** — 라벨은 결손된 게 아니라 **우리가 뽑지 못하게 막아 놓았다**.

```
A2 질문   gate.json:4740          {"kind", "what", "tool"} 을 요구
스키마    t2_run_gated.py:392-400 {"claim", "tool", "kind"} / pending 은 {"claim","tool"}
소비부    t2_gate_patch.py:14986  "%s: %s" % (c.get("kind"), c.get("what"))
                                  ⇒ what 은 개명당해 영원히 None · pending 의 kind 는 아예 없다
검정      test_terse_schema.py    리터럴 '"claim"' 을 단언해 **오답을 못박았다**
```

실측: 전송 문면 **73/73 이 `None: None`** · `unb_p>=1` **158/158** · 날짜 절벽
(09-01 까지 nonNone 전량 ↔ 09-03 부터 None 전량 · 경계의 유일 변경 = `f6224e26`).
파생 피해 3종(예비-창 사문화 · 050-DUP 구제 무력화 · user 롤 점유)은 §1c-5 D8 에 있다.

⇒ **이것이 이 문서에서 유일하게 원인까지 닫힌 우리-층 결함**이다. reward 효과는 여전히
미판정이지만([[57]] 부정통제 미확보), **[[64]]·계기 무결성 근거로는 무조건 수리**다.
그리고 [[81]]/[[84]] 의 재발이다 — *"고쳐 놓은 레버가 실은 다른 계약과 짝이 아니었다"*.

### ⛔ 근거 교체 — 「회귀 ≠ 인과」의 근거를 x725 에서 flip 정본으로

처음엔 `task_055` 의 x725 통과(1/3)를 근거로 삼았으나 그 짝은 **조건 혼합**이다:
x725 는 **런 전체가 claimprov 무효**였고(`agent_claimprov` 24/24 `declaration failed`),
`results.info.git_commit`(fc0055dc)은 **tau2-bench sha** 라 우리 층 동일성을 보증하지 않으며,
x725(09-01 16:01)와 실패 런들 사이에 커밋 **`117f02e5`**(09-01 22:54 · claimprov 수리 + A2 3파일)가
끼어 있다. **결론(회귀≠인과)은 유지하되 근거를 바꾼다** —
`C249`(nt=2 · 9/31 = **29%** · *"이후 모든 arm 비교의 판정 임계"*) ·
`C292`(16/64 = **25%** · *"pass 24↔24 는 무변화가 아니라 8↑8↓ 상쇄"*).
(부수: claimprov 전멸 런이 055 를 통과한 것 자체는 D8 방향의 **약한 n=1 신호**다 — 교란 있음·인용 금지.)

### `§1e` 는 신설이 아니라 **재측정**이다

정본에 이미 있다. `§1e` 는 아래 다섯을 **인용하고 Q38 에 재사용**하는 것으로 프레임해야
[[74]]/[[40]] 이 선다:
`C251`(*"두 trial 의 실패 분류가 동일 8/상이 9 ⇒ 처방을 태스크 단위가 아니라 **축 단위**로"*) ·
`C292`(*"원인 클래스 구성은 불변·명단만 churn — 태스크 단위 귀속은 **시뮬레이터를 재는 것**"*) ·
`C273`(기전-준결정 13/20 분리법) ·
`C498ⓖ` / `DEFECT_LEVER_COVERAGE_2026_08_23.md:99`(*"1차 종점을 성적이 아니라 **기전 계수**로"*) ·
방법 골격 `A1_V2_NT2_FORENSIC`(PP/FF/flip 교차표 → 기전 분해).

### 1차 리뷰(2026-09-04)에 대한 응답 — 커밋 `e5976454`

`B1 B2 B3 · C1 C2 C3 · D1~D5 약점 · D6-op · D7-op · §7 10단 교체` 전부 반영. `B1` 은 제 손으로
재현했고 **리뷰보다 나쁘게** 나왔다(§2 「B1 실측과 수리」).

### 리뷰에서 봐 주었으면 하는 것 (최종)

1. **§1e-0 의 flip 바닥이 옳게 잡혔나** — base 팔 16쌍은 얇다. 이 바닥이 판정 전체를 좌우한다.
2. **§1e-2 순열 부정통제** — 상관 구조를 보존한 라벨 순열이 맞는 설계인가.
3. **§1e-5 재정렬** — *"가르지 못하는 기전을 겨눈 후보는 강등"* 규칙이 과한가.
   D12·D13 처럼 **reward 가 아니라 [[23]]/[[64]] 로 서는 후보**를 어떻게 다룰 것인가.
4. **§1e-6** — 무엇을 더 재야 판정이 서는가. 특히 **태스크당 n 을 늘리는 것**([[57]]).

---

## 0. 선행 확인 — 어디를 찾아봤나 ([[74]] · §77 (4))

```
grep -rl "readloop|retention_offer" reports/facet_rft_2026/
  -> DAY7_PRESCRIPTIONS_DESIGN_2026_07_28.md · N97B_FIX_LEDGER_2026_08_05.md
     PAPER_TRACKC_DRAFT_v0_2026_07_24.md · RATE_SUBAGENT_DESIGN_2026_07_18.md
     RESEARCH_MASTER.md · STAGE2_GATE_DESIGN_2026_07_26.md · x540_spec_derivation_2026_08_25.json
grep -rn "readloop|resignation" scripts/distill/tau2/*.py
  -> t2_gate_patch.py:5477 · :13756 · :13759 · :13760 · :14550
     test_c207_envelope.py:99,151 · x60_l4_predicate.py:12
read  reports/facet_rft_2026/x509_axis_queue_2026_08_24.json (axis_table · steps S0~S2 · status_2026_08_29)
read  scripts/distill/tau2/a2/banking_knowledge.specific.json:4742-4850
```

**이 자리는 이미 두 번 다뤄졌다. 재발명하지 마라.**

| 선행 | 무엇을 확정했나 |
|---|---|
| `N97B_FIX_LEDGER_2026_08_05.md:323` | `close.requires` 에서 `retention_offer`·`log_reason` 을 **일부러 뺐다**. 정책 축자 *"If records are found within the past year … skip retention offers and proceed directly to processing the closure"* ⇒ 조건부 단계는 **표면화만** 한다 |
| `banking_knowledge.specific.json` `_note_nodes` | 구판이 `close` 에 `retention_offer` 를 필수로 걸어 **gold 경로를 6 태스크·47곳에서 막았다(x91)** |
| RESEARCH_MASTER `C157`·`C158` | 같은 절차의 premature-close 를 오프라인 사다리(n=12/16)로 측정. ①순수 reasoning 은 억제 실패 ②억제는 *"닫기 외 대안"* 빈 슬롯이 산다 ③올바른 회상은 A2 범주명(`retention_offers`)이 있을 때만. C158 축자: *"유일 고장=close 순서"* |

⇒ **`close.requires` 를 되돌리는 처방은 이미 폐기된 것이다. 이 문서는 그것을 제안하지 않는다.**

---

## 1. 원인 진술 ([[77]] 4칸)

> ⛔⛔**P0 를 리뷰어가 직접 쳤다 — 이 절의 양화가 무너진다 (2026-09-04 2차 리뷰)**
> `fb_bank_049ctl2` 실측: `[PROCEDURE] You are inside` 는 **런 전체 6행**(turn 30·35·50·55·59)이고
> **종결 후는 sim 당 1~2건**이다. 내가 쓴 *"56턴 중 34턴 동일 상태"* 는 `[T2_PROCEDURE]` **계기 전용
> 로그 줄**(35줄이 같은 turn 에 몰림)이었지 표면화 발화가 아니다. 그리고 엔진은 **이미 묶고 있다** —
> `[T2_PROC_ABSENT] surface … quiet>=3`.
> ⇒ **R2 는 절반 성립**: 발화가 실재하므로 이 절을 폐기(0건)하지는 않는다. 그러나 **D1 이 제거할
> 압력의 실물은 34가 아니라 2다.** §1c-4 의 *"중립~약화"* 에서 **한 칸 더 내려간다**.
>
> ⛔**[[74]] — 049 는 이미 A/B 가 돌았다.** `bank_049treat_20260903_1913` = **`T2_ACT_DEMAND` 팔**
> (28 발화) · `ctl`/`ctl2` 와 함께 **셋 다 `results.sims` 0**(마감 사망 · reward 없음). 이 절은
> `ctl2` 만 인용했다. 게다가 `T2_ACT_DEMAND` 는 **`C492` 가 이미 null+과행동으로 판정**한 레버다 —
> **treat 설계 자체가 선행과 충돌했다.**
>
> ⚠**분모 주의 (2026-09-04 1차 리뷰 D4 · §1c 와 같은 경고를 여기에도 붙인다)**
> 이 절의 근거는 **`bank_049ctl2_20260904_0534` 한 태그 · `task_049` 한 태스크 · n=2 sim** 이다.
> 다른 절과 태그가 다르다 — §1b = `bank_g97151p11_viewmax2`(20건) · §1c = `bank_k8141med1`(3건) ·
> D6 = `bank_k8143med1`(1건). **합산하지 마라.**
> 그리고 **§1c 의 3 sim 에서 이 병리(절차 정체·readloop)는 재현되지 않았다** — D1·D2 의 근거는
> 지금도 **한 태스크·두 sim** 이 전부다.

### (1) 주장 + 양화

`bank_049ctl2_20260904_0534` 의 sim **#s626729** 에서, `close_credit_card_account_7834` 가 성공
실행된 **뒤에도** 절차 `credit_card_closure_retention` 의 잔여 노드 `retention_offer` 가 계속
"다음에 할 일"로 표면화되어, 체크리스트가 **56턴 중 34턴** 동일 상태
(`done=5 left=['retention_offer']`)로 머물고 KB 검색 루프가 이어졌다. 자매 sim **#s373753** 은
48턴 중 13턴. 두 sim 은 각각 **23회·18회** `readloop-turn` 으로 집계됐고, agent 컨텍스트가
**1,552 tok/턴**(#s626729 최근 15턴 실측)으로 자라 131,072 천장까지 **약 26턴**을 남긴 상태에서
마감으로 중단됐다. n=2 sim.

### (2) 근거 — 축자 인용 + 파일:줄

```
로그  bank_049ctl2_20260904_0534.log
  [sim=task_049#s626729] [T2_PROCEDURE] checklist proc=credit_card_closure_retention
                          nodes=6 done=5 left=['retention_offer']            <- 34회 동일
  [sim=task_049#s626729] [T2_FOLLOWUP] readloop-turn counted as resignation  <- 23회
  전이 3->4 직전:
  [sim=task_049#s626729] [T2_PROCEDURE] unmet tool=close_credit_card_account_7834 missing=[] unobservable=[]
  [sim=task_049#s626729] [T2_TOOL_OBS] Credit card account closed successfully.

표면화 문구  a2/banking_knowledge.specific.json
             procedures.credit_card_closure_retention.feedback.absent 축자:
  "[PROCEDURE] You are inside {procedure} and its next step has not been taken
   ({done} of {total} done): {checklist}  NEXT: {next}. {unlock_hint}Do that step before continuing."

설계 의도    같은 파일 _note_nodes 축자:
  "보유-제안은 조건이 도구 출력 내용에 달려 있어 우리에겐 닫히지 않으므로 표면화만 한다."

계기 구현    scripts/distill/tau2/t2_gate_patch.py:13759   self._t2_fu_readloop_turn = True
             scripts/distill/tau2/t2_gate_patch.py:13760   print("[T2_FOLLOWUP] readloop-turn ...")
             scripts/distill/tau2/t2_gate_patch.py:14550   _fu_genuine = not ..._fu_readloop_turn
```

⇒ **선언은 "표면화만", 실제 문구는 명령형("Do that step before continuing")** 이다. 그리고 readloop
플래그는 **chain 예비-예산 소비를 막는 데만** 쓰이고(`:14550`), 모델에게는 아무 말도 하지 않는다.

### (3) 반증 조건 / refutation conditions — 무엇이 관측되면 이 주장이 거짓이 되는가

주장과 **동시에** 적는다. 아래 셋 중 하나라도 관측되면 위 원인 진술은 무너진다.

- **R1 (refute by isolation)**: 종결 노드 실행 뒤 표면화를 끈 격리 조건에서도 KB 읽기 루프가 같은
  빈도로 나오면, 원인은 표면화가 아니다(모델의 절차 종료 판단 결손).
- **R2 (refute the premise)**: `feedback.absent` 가 종결 뒤 실제로는 발화하지 않았고 로그의
  `checklist` 줄이 표면화가 아니라 **계기 전용 출력**이라면, 이 원인 진술 전체가 거짓이 된다.
  ⇒ **P0 에서 먼저 확인한다**.
- **R3 (refute by history)**: 같은 태스크의 과거 `context_window_exceeded` 2건
  (`n97_gpu0_main_20260805` 109분 · `n97_gpu0_main_20260806b` 82분) 궤적에 종결-후 표면화가
  **없었는데도** 같은 루프가 있었다면, 표면화는 충분조건이 아니다.

### (4) 선행 확인

§0 의 grep 경로 목록 그대로. `close.requires` 완화는 **이미 판정된 사안**이며 되돌리지 않는다.

> ⚠ **이 진술은 아직 가설이다.** refutation(R1~R3)을 거치기 전에는 처방의 근거로 쓰지 않는다([[77]]).

---

## 1b. 캠페인 실패 20건 전수 per-step 포렌식 (2026-09-04 08:00)

> ⛔⛔ **이 절 전체에 붙는 경고 — `action_checks` 는 채점 단위가 아니다** ([[69]])
> 이 캠페인의 실패 20건 중 **19건이 `reward_basis: ['DB']`** 다. 그 태스크의 reward 는
> `db_check.db_match`(궤적 재실행 후 **DB 해시 비교**)에서 나오고, `action_checks` 는 **진단용**이다.
> 실증(n=1 sim · task_051): `051_6` 이 `match=False` 인데 그 호출은 **실제로 실행됐고 DB 를 바꿨다** —
> `msg60` 축자 *"Payment processed successfully! - Payment Amount: $3000.00 - New Checking Balance:
> $2000.00"*. 불일치의 정체는 중첩 payload 문자열 비교(`3000` ↔ `3000.00`)뿐이었다
> (`tau2-bench/src/tau2/data_model/tasks.py:195` `return tool_args == action_args`).
> ⇒ **아래의 칸 단위 수치(81칸·45/32/4·34칸 값 대조)는 "어디를 볼지"의 지도이지 실패 귀속이 아니다.**
> 귀속은 **변이 집합(MISSING/WRONGARG/EXTRA)** 으로 다시 세야 한다([[69]]).
> ⚠같은 이유로 내가 한때 낸 *"81칸 중 12칸이 직렬화 문제 · base 는 345칸 중 204칸"* 은 **철회한다** —
> `action_checks` 를 실패 단위로 놓은 데서 나온 수치다.



채점 55/97 · pass 35 · **fail 20**. 20건 **전부 `user_stop`** 종료다(크래시 0 · 컨텍스트 소진 0).

### 실패 칸의 기전 분해 (81 칸)

각 실패 칸의 도구 이름이 궤적에 ①아예 안 나왔나 ②나왔는데 안 불렀나 ③불렀는데 불일치인가:

```
③ 불렀는데 인자/값 불일치   45 칸 (55.6%)
② 이름은 나왔는데 미호출     32 칸 (39.5%)
① 이름이 안 나옴(미발견)      4 칸 ( 4.9%)   <- 검색은 병목이 아니다([[79]] 와 정합)
```

첫 실패 칸(연쇄의 머리)의 requestor = **assistant 11 · user 8**. 도구는
`call_discoverable_agent_tool` 6 · `call_discoverable_user_tool` 5 · `unlock_discoverable_agent_tool` 4.
전체 실패 칸의 45/81 이 `assistant/call_discoverable_agent_tool` 다.

### 우리 층 거절 130건의 절반이 **부수 차단**이다

```
tool-deny 130 = 원발 65 + [BLOCKED] 부수 65 (50%)
원발 문구: resolve-the-flagged-call 22 · [SIGNATURE] 16 · [OPERATOR-SCOPE] 6
           [POLICY GATE GB2_NOTICE_BEFORE_TRANSFER] 5 · [DUPLICATE-WRITE] 4
           [WRITE-EVIDENCE] 4 · [REFERENCE] 3 · [E-PLAN] 2 · [PROCEDURE] 1
```

`task_041` 한 건이 130 중 **67**을 차지하고, **한 턴에 최대 21건**이 동반 차단됐다(turn 44).

### §1b-refute — 내가 CONFIRMED 라 적었다가 **스스로 반증한 것** (2026-09-04)

**주장(원래)**: `task_041` 의 dispute **8 칸**을 우리 `[REFERENCE]` 게이트가 죽였다. n=8 칸 · sim 1개.

**반증 근거 — 축자 + 위치(sim#turn / action_id)**
```
같은 sim 의 같은 게이트 아래에서 dispute 6 칸이 **통과**했다:
  041_9  match=True  txn_645286a3dd13 digits=0652     041_19 match=True txn_dd095dee227f
  041_10 match=True  txn_1b4cc30a928e digits=3081     041_20~041_24 match=True (4칸 더)
연쇄의 머리는 따로 있다 — 손님 액션이 안 났다:
  041_3 requestor=assistant give_discoverable_user_tool(get_card_last_4_digits)  match=True
  041_4~041_7 requestor=user  call_discoverable_user_tool(get_card_last_4_digits) **match=False ×4**
그리고 우리 게이트는 그 결손을 **정확히 지적**했다 (task_041 turn 42·44 축자):
  "[WRITE-EVIDENCE] no tool output in this conversation shows the card's last 4 digits (2716).
   Do NOT guess or fabricate the digits ... call give_discoverable_user_tool to give the customer
   the get_card_last_4_digits ..."
```
⇒ **도구를 넘기는 것(041_3)은 우리가 했고, 손님이 그것을 호출하지 않았다(041_4~7).** 자릿수 없이
분쟁을 걸려는 시도를 우리 층이 막은 것은 **정당하다**. 같은 게이트 아래 6 칸이 통과했으므로
*"게이트가 8 칸을 죽였다"* 는 인과는 **성립하지 않는다**.

**남는 것(축소된 주장)**: `[REFERENCE]` 문면이 *"does not appear in any record returned by the
tools"* 라고 말하는데 그 8개 id 는 msg 17(role=tool)에 있었다 ⇒ **문면이 거짓**이다. 이것만이
확정이고, 처방은 D3 의 **문면·술어 일치**로 한정된다.

**반증 조건 / refutation (남은 주장에 대해)**: `apply_op` 가 집합을 돌려주는데 호출부가 스칼라로
비교하는 것이면 수리 위치가 `t2_compute` 다. 그리고 041 의 8개 id 중 criteria 부합이 1개뿐이라면
게이트 술어는 옳고 문면만 고치면 된다 ⇒ P3a·P3b.

**선행 확인**: `grep -rn "does not appear in any record" scripts/distill/tau2/`(`t2_gate_patch.py:2851`
`:9410`) · `grep -n "def resolve_reference_filter" -A 90 t2_resolve.py` · 회수된
`fb_bank_g97151p11_viewmax2_20260903_1924.jsonl` · 해당 sim 의 `action_checks` 전문.

### 원래 적었던 관찰 (위 반증을 붙여 읽어라) — `[REFERENCE]` 게이트의 오판

**주장 + 양화 (n=8 칸 · sim 1개)**: gold 요구 `file_credit_card_transaction_dispute_4829`
8 칸(041_11~041_18)이 실패했고 같은 턴에서 우리 게이트가 이들을 거부·동반차단했다.

**근거 (축자 + 위치)**
```
원발 deny 축자:
  "Error: [REFERENCE] the transaction_id you named does not appear in any record
   returned by the tools in this conversation."
부수 deny 축자 (같은 턴 나머지 전부):
  "Error: [BLOCKED] this call was not run because another call in the same turn was
   blocked: 'call_discoverable_agent_tool(file_credit_card_transaction_dispute_4829)'"

그런데 그 8개 transaction_id 는 모두 이미 대화에 있었다:
  txn_107c4fa829bd · txn_3880720b4409 · txn_816986054539 · txn_4f6e48543e07
  txn_b4f90f6ee392 · txn_5e6ad271fefb · txn_a42ce2e4156d · txn_c7a1c5fad26b
  최초 등장 = 전부 **메시지 17 (role=tool)**
  dispute 호출이 나간 메시지 = **23 · 64**   => 차단 시점에 이미 6~47 메시지 전부터 존재
```

**반증 / refutation**: 그 id 들이 msg 17 이 아니라 차단 **이후**에 처음 나왔다면 게이트가 옳고
이 귀속은 무너진다. 위 index 측정이 그 반증을 이미 쳤다(8/8 이 msg 17 · 호출은 23·64).

**선행 확인**: §0 의 grep 경로 + `fb_bank_g97151p11_*.jsonl`(회수분) + 해당 sim 의 messages 전문.

**⛔내가 처음 세운 가설은 틀렸다 — 코드를 읽어 반증했다.** *"게이트가 view 창만 보느라 msg 17 을
못 봤다"* 고 적었으나, `t2_gate_patch.py:9398` 는 `state.messages` **전사**를 넘긴다. 창은 무관하다.
**진짜 버그는 술어 자체**이고 §2 의 D3 에 적었다 — 게이트는 *"지목한 id 가 기록에 있는가"* 가 아니라
*"내가 계산한 단 하나의 id 와 같은가"* 를 검사하며, 손님이 8건을 분쟁하는 이 태스크에서는 정의상
7개가 거부된다.

---

### 1c. 새 실패 3건 정밀 포렌식 (2026-09-04 09:00 · 059 · 064 · 088)

> **채점 단위 선언 ([[69]] ①)** — 세 태스크 모두 `reward_basis: ['DB']` 이고 셋 다 `db_check.db_match=false` ·
> `nl_assertions=null` · `env_assertions=[]` 다. 따라서 **아래 서술의 실패 단위는 전부 DB 변이 집합
> (MISSING / WRONGARG / EXTRA / DUP / MATCHED)** 이고, 정본 `t2_forensic.mutation_diff(sim, tag=TAG)` 로
> 산출·재현했다. `action_checks`(059 1/6 · 064 2/4 · 088 3/17 실패)는 **어디를 볼지의 지도일 뿐 실패 귀속이
> 아니다** — §1b 머리말의 경고가 그대로 적용된다. 아래에서 action_checks 수치는 한 번도 실패 단위로 쓰지 않았다.
>
> **절차** — 태스크마다 per-step 포렌식 1회 + **적대적 반증 1회**를 돌렸다. 반증에서 무너진 주장은 **지우지
> 않고 "⛔철회" 로 남긴다**([[73]] · §1b-refute 와 같은 규율). 각 문장에 **[CONFIRMED] / [PLAUSIBLE] /
> [미판정]** 을 붙였다.
>
> **분모 주의** — 태그는 `bank_k8141med1_20260903_2256` (results.json 12 sim)이며 §1b 의 20건과 **합산하지
> 마라**. 두 집계가 겹치는지는 **[미판정]**.
>
> **라벨 충돌 주의** — 088 반증 문서가 자체 라벨 `D1~D3` 을 썼는데 이 계획서의 `D1~D6` 과 **다른 것**이다.
> 이 절에서는 그 항목들을 **D8·D9** 로 재명명해 인용한다.
>
> **시간 수치 주의 ([[83]] · [[54]])** — ours 는 `Concurrency: 4`(`bank_k8141med1_20260903_2256.log` 축자
> *"Save: bank_k8141med1_20260903_2256  Concurrency: 4"*), base x644 는 축자 *"규격 : alltools · seed 300 ·
> max-steps 200 · timeout 7200 · **concurrency 1** · port 8143"* 다. **벽시계 분은 배선 비용과 배치 조건이
> 섞인 값이므로 원인 진술에 쓰지 않는다.** 아래 분 수치는 §5 ② 축의 참고값으로만 적는다.

---

#### 1c-0. 세 건 공통 (교차 확인된 것)

| 사실 | 등급 | 근거 |
|---|---|---|
| 세 건 모두 **검색 결손이 아니다** — 필요한 KB 문서가 궤적에 배달됐는데 값·건수·전달이 어긋났다 | **CONFIRMED** | 059 msg32(두 절차 문서) · 064 msg31(`check_card_application_fit` 로스터) · 088 `doc_..._031` **4회 배달**. [[79]] *"Q38 의 잔여는 retrieval 이 아니다"* 와 정합 |
| **우리 층 거절이 gold mutating 호출을 막은 사례 0건** | **CONFIRMED** | 059 deny 3건 전문 · 064 deny 8건 전수 + 형제-통과 대조 · 088 deny 6건 전수(`[OPERATOR-SCOPE]` 는 **지연**시켰고 turn 67 축자 *"[T2_RESOLVE] operator-scope 상한 초과(2회) — 통과시킨다"* 로 통과) |
| 그러나 **"우리 층이 값을 저작하지 않았다"** 는 059·064 에서만 유지되고 **088 에서는 무너졌다** | **CONFIRMED** | 088 msg 66 = 우리 claimprov 재생성 산출물(1256B, 바이트 일치) — 1c-3 참조 |
| 세 건 모두 `max_tokens=8192` 상한 미충돌 ⇒ [[82]] 폭주 아님 | **CONFIRMED** | 059 최대 completion **4,998**(`…log:5714` `gen=4998 prompt=58499`) |
| 세 건 모두 **절차 체크리스트 정체·readloop 없음** (§1 의 task_049 병리가 재현되지 않음) | **PLAUSIBLE** | 059 msg62→66 절차 소진 · 088 KB 검색 15회 **질의 중복 0**. ⚠같은 포렌식이 다른 칸(재생성 카운트)에서 오류를 냈으므로 등급을 낮춘다 |

---

#### 1c-1. task_059 — `account_class` 한 인자 (sim `task_059#s626729` · n=1 · 72 msg · 291분)

**실패 단위 [CONFIRMED]** — `MISSING 1 · WRONGARG 1(같은 gold 행의 짝) · EXTRA 0 · DUP 0 · MATCHED 2 · BLOCKED 0`.

```
MATCHED  msg51 log_verification              -> "Verification logged successfully. - User: Casey Rivera (ID: cr59b4d8e3)"
MATCHED  msg56 apply_for_credit_card(card_type="Silver Rewards Card")
WRONGARG msg68 open_bank_account_4821 account_class = "Green Account (savings)"   (gold "Green Account")
```
그 호출은 **거절되지 않고 실행됐다** [CONFIRMED] — msg69 축자: *"Bank account opened successfully! - Account ID:
f9386249cd4ade09 - Account Type: savings - **Account Class: Green Account (savings)** - Status: OPEN"*.
⇒ 이 태스크의 DB 실패 전체 = 문자열 `" (savings)"` 6글자.

**연쇄의 머리 [수정됨]** — **msg 40** 이다(msg 68 아님). 28 메시지 앞에서 이미 같은 문자열을 쓰고 출처를 KB 문서
id 로 자인했다:
```
msg40 get_correct_savings_apy {"savings_account_type": "Green Account (savings)",
      "source": "Green Account (savings) FAQ: '... 4.0%' (doc_savings_accounts_green_account__savings__005)"}
```
그리고 그 표기는 **유추가 아니라 KB 의 상품명 축자**다 [CONFIRMED] — msg3 *"doc_savings_accounts_green_account__savings__001.md
- **Green Account (savings)** specifications and requirements"*, msg7 표제 *"# Green Account (savings) specifications
and requirements"*, msg4 *"3. Evergreen Account (checking) + **Green Account (savings)**"* (msgs 3·4·7·9·10·13).
`"Green Account"` 와 `"Green Account (savings)"` 가 **둘 다 KB 에 실재**해 KB 접지로는 가를 수 없다.
⇒ 남는 진술: **표기 직렬화 분산** [PLAUSIBLE]. 양쪽 런 어디에도 선택 이유가 남지 않았다 [CONFIRMED].

**우리 층 개입 [수정됨]** — *"손대지 않았다"* 는 **문장 그대로는 거짓**이다. 문제의 생성(trace turn 65) 직전·직후로
우리 문장이 뷰에 들어갔다:
```
trace turn=63 / 65 / 67   [T2_FB_VIEW] 1 queued feedback item(s) injected in view
t2_gate_patch.py:8939-8956  _t2_view_fb 를 UserMessage 로 작업버퍼에만 부착(비커밋)
큐잉 원천 = T2_LEDGER (trace turn 61 · T2_LEDGER_VIEW_KEEP=3 -> 63·65·67 정확히 3회)
```
그러나 **주입된 내용은 값과 무관하다** [CONFIRMED] — 정본 `t2_ledger.facts_text` 를 실제 rows 로 렌더한 전문에
`account_class`·상품명·괄호 언급이 **0회**다(*"[COMPUTED FACTS] Counted from the accounts above (arithmetic, not a
recommendation): 1 account(s). …This is elapsed time only. It is NOT a threshold."*). 그 턴의 우리 층 발화 전량
(`T2_SUBWIN` · `T2_SUBCALL cache hit` · `T2_SIBLING_PAREN`(print) · `T2_A2_VARIANT`×2 · `T2_FB_VIEW`)에 인자를
고치는 경로가 없다. `attempted_mutations` 상 `open_bank_account_4821` 시도는 **msg68 단 1회 · ok=True**.
⇒ **우리 층은 `account_class` 값을 저작·변경하지 않았다** [CONFIRMED · 근거 교체됨].

**우리 층 계기 결함 (원인 아님) [CONFIRMED]** — `T2_SIBLING_PAREN` 이 이 호출을 정확히 탐지하고 고칠 값까지
이름 붙였는데(런 전체 8,300여 줄에서 **유일한 발화** · `logs/bank_k8141med1_20260903_2256.log:7764`) **집행되지
않는다**:
```
t2_gate_patch.py:13306  "★§T-8 계기 ... **거동 변화 0**. ... 반려(`deny`)는 이 수를 보고 붙인다"
t2_gate_patch.py:13311  if os.environ.get("T2_SIBLING_PAREN") in ("log","deny") ...   <- 분기는 print 한 줄뿐
arms/{t8log,t2prime,t3prime,viewmax2}.env  전부 "=log"     repo 전체 "=deny" 0건
```
인용한 코드 = 실제로 돈 코드임을 확인했다 [CONFIRMED]: `provenance.json` 이 `engine_dirty: true` 인데
`git status --porcelain | grep -v '^??'` → **0줄**(전부 untracked reports/).

**base 대조 [강등]** — x644(`sim_results/bank_x644_q38base_bank78_20260830.results.json.gz`)에서 task_059 는
**reward 1.0 · db_match True · 47 msg · 14.7분** 이고 보낸 값이 `"Green Account"` 다. ours 는 `0.0 · 72 msg ·
291분`. **그러나 "회귀" 로 쓰지 마라** — 코퍼스 전량(정본 `F.iter_all_sims(want_tasks={"task_059"})` · **14 sim**)에서
gold 문자열은 **x644 단 1회**이고, 호출을 한 7 sim 중 6 sim 은 아예 다른 상품(Diamond Elite ×7 · Platinum Savings …)을
골랐다. 이번 런은 **상품을 처음으로 맞히고 표기만 틀린 최초 사례**다. ⇒ 판정: **n=1 분산** [PLAUSIBLE], 회귀
[미판정]. (08-04~08-06 sim 은 Q2.5 레거시라 [[79]] 상 직접 비교 불가.)

**⛔ 철회 목록 (059)**

| 철회한 주장 | 왜 |
|---|---|
| *"사이드카 turn 집합 = [0,29,…,57] · turn≥58 은 0건 ⇒ 우리 층 미개입"* | 계기 파손. 33행 중 **17행이 `turn=0`이고 전부 `kind='subcall'`** — turn 이 subcall 행에 채워지지 않는다. 그 중 4행은 시간상 turn 57 이후이고, 하나는 `{"tool": "open_bank_account_4821"}` 로 **msg67 unlock 이후에만 존재하는 이름**을 담고 있다 |
| *"turn 57 < turn 68 이므로 개입 없음"* | **turn 축 3종 혼용**(msg 색인 / 사이드카 turn / trace turn). 문제의 생성은 trace turn **65** 다 |
| *"우리 층은 그 호출에 아예 손을 대지 않았다"* | `T2_FB_VIEW` 주입 3회가 사이드카·영속 궤적 **어디에도 안 남는다**. 결론은 유지하되 근거를 D1 계열(내용 무관)로 교체 |
| msg66 assistant reasoning 축자(*"…but that doesn't end in 'Account.' Hmm. … By analogy…"*) | **궤적에 존재하지 않는다.** 우리 런 assistant 28개 전부 `reasoning_content` 0B. `'By analogy'`·`"doesn't end in"`·`'Hmm'` 모두 검색 결과 `[]` |
| base x644 msg41 reasoning 축자(*"…account_type disambiguates…"*) | 동일. base assistant 중 `reasoning_content` 보유 **0개**, base msg41 은 content 0B 의 맨 도구 호출 |
| *"checking 예시로부터 유추해 술어를 뒤집었다"* | KB 가 그 상품을 그렇게 부른다(위 msgs 3·4·7·9·10·13). 유추 아님 |
| *"머리 = msg68 말단 단일 인자 오류"* | 첫 발화는 **msg40** |
| *"우리 층이 개입한 가지는 정답이 됐다(apply_for_credit_card MATCHED)"* | 그 호출은 **`role=user`** 다 — user-sim 이 실행했고 `annual_income=55000` 도 user-sim 이 채웠다(agent 는 msg55 에서 *"I left this blank for you to fill in"*). base 도 동일(`msg25 role=user`, 인자 바이트 동일) |
| *"turn 49 deny 대상 = apply_for_credit_card / check_card_application_fit"* | turn 49 의 대상은 `call_discoverable_agent_tool` 이고 사유는 `[T2_PHASE_PRECEDE] … reqs=['GB1_VERIFY_BEFORE_ACCOUNT_ACCESS']` — **gold 호출과 같은 래퍼**다. ⚠§2 (철회됨)D5 의 *"우리 층은 이 호출을 건드린 적이 없다"* 줄도 이 좁은 표현으로 읽어라 |
| *"T2_SIBLING_PAREN 이 이 결함을 막을 수 있었던 **유일한** 결정론 장치"* | 거짓. `[OFFICIAL-NAME]`/`T2_WRITE_ARG_ENUM` 계열이 존재하고 과거 fb 파일 **56개**에서 발화했다(이번 런 0건 — L1 참조) |
| 최대 completion `4,613` | 실측 **4,998**. 결론(8192 미충돌)은 불변 |

**⛔[[74]] 위반 (059 포렌식의 방법 결함) [CONFIRMED]** — `prior_checked` 12칸이 전부 `scripts/`·`logs/`·`sim_results` 이고
**`reports/` 를 한 번도 grep 하지 않았다.** 결론의 절반이 3주 전 정본에 이미 있다:
```
FAILURE_AXIS_AND_FIX_ORDER_2026_08_15.md:2478-2481 §T-8
  gold : open_bank_account_4821{account_class:"Green Account", ...}
  done : open_bank_account_4821{account_class:"Green Account (savings)", ...}
:2490  "**KB 접지** — `Green Account` 와 `Green Account (savings)` **둘 다 KB 에 실재**해 못 가른다."
:2508  ⛔W-5 (블로커) "모델이 반려를 받고도 같은 값을 다시 보낸다 ... 한 sim 최다 반복 18회 ...
        회복되지 않으면 이 레버는 **오답을 스텝 소진으로 바꾼다**"
:2521  "⛔엔진이 값을 고치지 않는다. ... 지금은 **관측+부호표**이고, 효과는 켠 팔과 안 켠 팔로 재야 한다."
DEFECT_LEVER_COVERAGE_2026_08_23.md §ⓒ
  open_bank_account_4821 WRONGARG 160건 중 **159건이 단일 인자 account_class**
  (Green Account (savings) ↔ Silver Plus Account 25 ...)
```

---

#### 1c-2. task_064 — operand 2칸(`account_class` · `card_type`) (sim `c65c88f6…` = `task_064#s626729` · n=1 · 92 msg · 301분)

**실패 단위 [CONFIRMED · 독립 재현 일치]** — `gold mutating 3 · MATCHED 1 · WRONGARG 2 · MISSING 2(같은 gold 행의
미매칭 쌍) · EXTRA 0 · DUP 0 · BLOCKED 0 · regen_blocked 8`.
```
MATCHED  msg53 log_verification            -> "Verification logged successfully. - User: Jordan Lee (ID: jl72b4e9d1)"
WRONGARG msg84 open_bank_account_4821  account_class "Silver Account"   (gold "Gold Plus Account")
         msg85 축자 "Bank account opened successfully! ... **Account Class: Silver Account** - Status: OPEN"
WRONGARG msg87 apply_for_credit_card   card_type "EcoCard"              (gold "Gold Rewards Card") · requestor=user(gold 도 user)
         msg88 축자 "Credit card application submitted: Your application has been successfully submitted."
```
두 호출 모두 **실행됐다**. 나머지 인자(`user_id`·`account_type`·`customer_name`·`annual_income`·
`rho_bank_subscription`)는 gold 와 동일 ⇒ **operand 값 선택 오류 2칸**이고 열거·전달·실행은 성공했다.

**연쇄의 머리 [수정됨]** — msg 35 송출본(689B, 권고표 소실) → **msg 47** → msg 80 → 집행. 다만 msg 47 은
**모델의 자발적 전환이 아니라 우리가 만든 턴**이다 [CONFIRMED]:
```
fb idx=27 kind=reminder-user turn=47 channel=claimprov len=306 축자:
  "Note: [CLAIM-PROVENANCE] tool ownership — the following are in YOUR OWN tool list, not the
   customer's: None (tool: verify_identity); None (tool: log_verification); None (tool: give_discoverable_user_tool)."
msg 47 reasoning_content 축자:
  "**This appears to be a system reminder/note rather than a genuine customer message.**"
  "Since there's no actionable customer request, **I'll send a brief message keeping the ball in their court.**"
```
그 턴의 채움말이 *"lock in the **EcoCard + Silver Account** combination"* 이었다. **심의된 operand 결정이 아니다.**
그리고 msg 80 송출본(1503B)은 두 금액을 **나란히** 제시했다 — *"**Reply \"1\"** → I open the **Silver Account** now
(best score-independent combo: $1,817.50 …)"* / *"**Reply \"2\"** → I open the **Gold Plus Account** now (best overall
combo: $1,905.00 …, if your score is 720+)"* — 고객은 그것을 보고 1을 골랐다(msg 81). ⇒ 머리의 최소 진술:
**닫힌 목적함수(연이자−연회비 최대화)의 operand 2칸이 우리가 점유한 턴에서 굳었고, 금액 병치 뒤에도 열린 조건
(미지 신용점수) 회피가 이겼다** [PLAUSIBLE · n=1].

**우리 층 개입 = yes [CONFIRMED]**, 단 세 갈래로 나눠 적는다.

1. **[B] `get_correct_savings_apy` grounding 이 `파일명: '인용'` 접두를 못 읽는다 — CONFIRMED, 기전 확정**
```
t2_scaffold_get.py:724     src_ok = bool(ns) and any(ns in nc for nc in norm_corpus if nc)     <- 순수 substring
t2_scaffold_get.py:121-123 _norm_ground = re.sub(r"[^a-z0-9%]+", " ", str(s).lower())
msg59 source="doc_savings_accounts_silver_account_003.md: 'At or above threshold | At least $10,000 | 4.0%'"
  -> msg60 "[GROUNDING WARNING] 3 input value(s) could not be verified ... (could not compute — check your arguments)"
msg64 접두만 제거한 **같은 인용**
  -> msg65 "Correct savings APY computed from the components supplied in this call: 6.2250000000000005%"
계약 축자 a2/banking_knowledge.gate.json:1092  "source": "<verbatim quote from the KB doc stating this value>"
```
   반증조건 [B'](코퍼스 미도달) **소거됨**: msg 59↔64 사이 새 read **0건**. 비용 = 4콜 1라운드 전손, 유효 수치
   6개를 얻는 데 14콜. 거절 문면 *"source not found in the knowledge base"* 는 **무엇을 하면 풀리는지 말하지
   않는다** ⇒ [[64]] 위반 [CONFIRMED].
2. **[A] RECOMMEND-OFFER 는 원인이 아니라 추인 — 단 근거가 바뀌었다**. 우리 서브가 gold 와 다른 값을 확언했고
   (`recommend_formalize` → `{"applies": true, "card_type": "EcoCard"}`) 명령형으로 뷰에 넣었다. 그러나 그
   산출물은 **사용자에게 전달되지 않았다** [CONFIRMED]: 621B 문면이 궤적에 0건, `give_discoverable_user_tool`
   영속 호출 0건, `driver.log:8255 [T2_RESOLVE] recommendation-offer deny`, 송출 msg 86(634B)에 카드명 0회.
3. **[N2] 같은 턴에서 우리 층 두 문면이 시키고 막았다 — CONFIRMED · [[55]] 대상**
```
8227 [T2_RESOLVE] recommendation-offer deny  -> "Offer it now by calling 'give_discoverable_user_tool' … card_type='EcoCard'"
8235 agent_response_unified_regen … tool_calls=1                 <- 시킨 대로 호출
8247 [T2_TOOL_SIGNATURE] would-deny tool=give_discoverable_user_tool but preempted-by=resolve_write
8249 [T2_TOOL_OBS] err=True -> "Error: [ACTION] 'apply_for_credit_card' is run by the CUSTOMER, not by you."
8255 [T2_RESOLVE] recommendation-offer deny                       <- 또 같은 요구
8263 [T2_MATERIAL_GATE] stop=resolve_cap(정체 3회) turn=86
```
   루프는 판단이 아니라 **정체 캡으로만** 멈췄다.
   부수: **[N3]** `[ARG-EMPTY]` 거절(fb idx=6, turn=30)이 날조를 유도했고(msg30 reasoning 축자 *"I shouldn't
   fabricate constraints. **But the tool requires non-empty values.**"*, args `"credit_score":"850"`), 우리
   grounding 이 msg31 에서 다시 드롭했다(*"credit_score=850 (the customer never mentioned this kind of
   requirement — do not add limits they did not state)"*). [F](우리 도구가 Gold 우세 근거를 쥐어줬다)는 사실이나
   **거절→날조→드롭 1왕복 뒤**에 얻은 것이다.

**gold 호출 차단 여부 [CONFIRMED · 형제 통과 전수 대조]** — 0건. 같은 `[READ-FIRST]` 게이트가 msg40 4콜을
막았으나 msg59 4콜·msg64 4콜·msg77 2콜을 통과시켰고, `[ACTION]` 라우팅은 gold(`064_1 requestor=user`)와
**일치**하며 사용자가 msg87 에 실제 실행했다. turn 51 의 `resolve the flagged call(s) first` 는
`GB1_VERIFY_BEFORE_ACCOUNT_ACCESS` 선행이고 이후 msg53 MATCHED · msg84 실행 성공.

**base 대조 [CONFIRMED · 회귀]** — x644 `task_064` = **reward 1.0 · 68 msg · 20.25분(1215.13s) · 같은 시드
626729 · 첫 사용자 발화 동일**. base 결정 축자(msg 28): *"**My recommendation: Gold Plus Savings + Gold Rewards
Card**"* → user msg 29 *"…let's do it"*. ours 는 `0.0 · 92 msg`. **pass→fail 성립**.

**⛔ 철회 목록 (064)**

| 철회한 주장 | 왜 |
|---|---|
| *"검증 패스(`recommend_operand_verify`)도 EcoCard 를 통과시켰다"* | **독립 검증이 아니다.** 두 프롬프트(plen 5935 / 5900)는 35자만 다른 **같은 재료**이고, `Option details` 블록에 `annual_fee`·`min_score`·`cashback`·`"Gold Rewards"`·`"Gold Plus"` 가 **각 0회**다. 앞선 "none" 6회도 카드가 0회 등장하는 프롬프트에서 나온 **강제값**이다 ⇒ [[78]] 재료 결손 |
| *"그 값이 그대로 W2 가 됐다"* | 산출물 **미전달**(위 [A]). 결론('추인')은 유지, 근거 교체 |
| *"월클럭 14.9배"* | **배치 산물**(conc 4 ↔ 1). 배치 불변 지표로 다시 재면 `ours comp 52,800 tok / prompt 2,137,899` ↔ `base comp 21,267 / prompt 2,338,132` ⇒ **생성 2.48배, 프롬프트는 오히려 우리가 적다**. `msg 92↔68` 과 `reward 1.0→0.0` 만 유효 |
| 계기 수치 *"생성 45콜 · 재생성 31% · recommend_formalize ×7"* | 실측 **79콜 · 재생성 13/79 = 16.5% · recommend_formalize 실생성 5회**(사이드카 8행 중 캐시 3) |
| msg 80 축자(*"…$87.50 more…"*) | **폐기된 초안**(fb idx=39, len=3117)의 문장이고 송출 msg 80 에는 없다. 송출본은 두 금액 병치(위) — 결론은 오히려 강화 |
| *"msg 47 에서 에이전트가 수치 하나 없이 스스로 전환했다"* | 그 턴은 **우리 claimprov note 가 점유한 턴**이다(위 축자) |
| *"[T2_CLAIM_PROV] 오작동 1건"* | **3건**(idx 17 turn 35 · idx 27 turn 47 · idx 43 turn 80) |
| *"regen 이 정답 초안을 2회 폐기 · EcoCard 대체안은 regen 산물"* | 폐기는 **3회**(turn 30·35·80)이고, turn 35 **초안 자체**가 이미 *"(The EcoCard has no score requirement, which is why it's the fallback.)"* 를 담고 있었다 ⇒ 대체안 프레임은 regen 이 만든 게 아니다. [C] 는 **상관까지만** |

---

#### 1c-3. task_088 — EXTRA 1 + WRONGARG 2 (sim `task_088#s626729` · n=1 · 93 msg · 242.2분)

**실패 단위 [CONFIRMED · 독립 재현 일치]** — `GOLD 4 · DONE 5 · MATCHED 2 · MISSING 2 · WRONGARG 2 · EXTRA 1 ·
DUP 0 · BLOCKED 0`.
```
MATCHED  msg13 log_verification · msg74 close_debit_card_4721(reason=fraud_suspected)
WRONGARG msg70 file_debit_card_transaction_dispute_6281  (17 인자 중 15 일치)
           transaction_type              gold 'signature_purchase' ↔ 'pin_purchase'
           customer_max_liability_amount gold 50                   ↔ 500
           실행 증거 msg71 "Dispute ID: dsp_76b0f2bc26c3 … Provisional Credit: ISSUED - $347.99"
WRONGARG msg78 order_debit_card_5739
           delivery_option              gold 'STANDARD' ↔ 'EXPEDITED'
           excess_replacement_fee       gold 미선언   ↔ 0 명시           (DB 영향 [미판정])
           실행 증거 msg79 "Debit Card Order Confirmed … Delivery Option: EXPEDITED"
EXTRA    msg49 transfer_funds_between_bank_accounts_7291(blue->green, 100)  gold 17 action 어디에도 없음
           실행 증거 msg50 "Transfer completed successfully! - Amount: $100.00 - From: chk_..._blue
                            (new balance: $1150.00) - To: chk_..._green (new balance: $480.47)"
```
`transfer_to_human_agents`(088_16)는 **`mutating_tools`(44종) 밖**이라 DB 단위에 들어오지 않는다 [CONFIRMED ·
`F.mutating_tools()` 직접 확인].

**연쇄의 머리 [수정됨]** — 두 사슬이다.

- **사슬 A (EXTRA)**: msg 45 무청구 제안 → msg 46 승낙 → msg 49 실행. user 대본에 이 분기는 없다
  ([CONFIRMED] `task_088.json` notes 축자 *"informational resolution"*; msg 46 은 대본 §11 문장 *"Okay, I
  understand now. So basically I need to wait a couple more days…"* 과 즉흥 승낙을 **한 메시지에** 담고 있다) ⇒
  제안의 저자는 에이전트다([[21]]). 단 KB 가 그 제안을 허용한다 [CONFIRMED]: `doc_checking_accounts_…_006`
  CODE 51 §3 *"If they want to transfer, help them do so."*
  ⚠**그 턴의 실행 경로는 우리가 코치했다** — 아래 철회 참조.
- **사슬 B (`delivery_option`)**: **msg 66 은 모델의 작문이 아니라 우리 claimprov 재생성 산출물이다**
  [CONFIRMED · 바이트 일치].
```
trace turn 65
  [T2_GEN_TRACE] call=agent_response          -> gen=2059 reason=5476B **content=2325B** tool_calls=0
  [T2_CLAIMPROV] window hit(resign) claims=12 unbacked=0 pending=3 **unb_p=3 [None, None, None]**
  [T2_CLAIMPROV] owner split: agent=0 user=0 **unknown=3**
  [T2_GUIDED] guided applied (call=agent_response_claimprov tools=27)
  [T2_GEN_TRACE] call=agent_response_claimprov -> gen=966 reason=2932B **content=1256B** tool_calls=0
  [T2_CLAIMPROV] regen tool_calls=[]
궤적 msg 66 = **1256자**  (원본 2325B 는 폐기 · -46%)
```
  그 재생성본이 배달 메뉴에서 **STANDARD 를 통째로 빠뜨렸다**: *"3. **Delivery speed:** Free expedited (3–5
  business days) or Rush (1–2 business days, $35)?"* → 대본 §18 *"Just the standard option is fine."* 인 user 는
  준 메뉴에서 고를 수밖에 없었다(msg 67) → msg 78 `EXPEDITED`. 그리고 그 메시지의 "안 했다 + 번호 매긴 재질의"
  골격은 우리 문구가 축자로 요구한 것이다 — reminder-user turn 66: *"Do not end your involvement by describing the
  work as done or under way — either call the tool now, or **state explicitly that it has NOT been performed.**"*
  ⚠**폐기된 2325B 원본에도 STANDARD 가 없었는지는 원리적으로 확인 불가**(→ D9) ⇒ 이 자리는 **[미판정]** 으로
  남긴다. 원 포렌식이 *"모델의 열거 결손"* 으로 **닫은 것 자체가 잘못**이다.

**우리 층 개입 = yes [철회된 판정]** — 아래 참조. 다만 **거절이 gold 호출을 막지는 않았다** [CONFIRMED]:
`[OPERATOR-SCOPE]`(turn 68)의 대상은 gold `088_9` 의 unlock 이었으나 turn 67 축자 *"[T2_RESOLVE] operator-scope
상한 초과(2회) — 통과시킨다"* 로 한 턴 **지연 후 통과**했고(msg69 *"Tool unlocked: …"*), unlock 은 GRANTS 라 DB 를
바꾸지 않는다. `[POLICY GATE GB2_NOTICE_BEFORE_TRANSFER]`(turn 85)도 이전을 막지 않았다(msg90 *"Transfer
successful"*).

**값이 어긋난 자리의 KB 출처 (gold 역산 없음 · [[23]] 준수) [CONFIRMED]** — 필요 문서는 배달됐다
(`doc_bank_accounts_…_031` **4회**):
```
_031  "- Reported within 2 business days of statement: Maximum liability $50
       - Reported within 60 days of statement: Maximum liability $500"      -> 모델은 500 선택
      msg55 축자 "That charge was on **11/09/2025** — about **five days ago**."  <- 거래일 앵커
_031  "9. transaction_type ... 'pin_purchase': In-store purchase with PIN
                             'signature_purchase': In-store purchase with signature"
      같은 호출에 pin_compromised='no' 를 넣고 msg58 에서 "the classic signature of a **cloned (counterfeit) card**"
      라고 진단해 놓고 pin_purchase 를 골랐다 (자기 인자와 모순)
_029  "PREMIUM TIER: ... (delivery_fee: $0 for both STANDARD and EXPEDITED) - Rush shipping ... ($35)"
      => EXPEDITED $0 은 **정책상 허용**된다. 정책 위반이 아니라 메뉴 결손이다
```
*"pin_compromised='no' ⇒ transaction_type≠'pin_purchase'"* 를 세울 KB 축자는 **없다** ⇒ 닫힌 술어 후보로
올릴 수 없다 [미판정 · [[23]]].

**base 대조 [CONFIRMED · 회귀 아님]** — x644 `task_088#s626729`(같은 시드) = **reward 0.0 · db_match False ·
61 msg · 9.3분(556.49s)** 이고 변이집합은 `MISSING 3 · WRONGARG 0 · EXTRA 0 · MATCHED 1` — dispute·close·order 를
**한 건도 실행하지 않았다**. ⇒ *"base 는 됐는데 우리가 깨뜨렸다"* 는 성립하지 않는다. [[79]] 그대로 **행동 부재
(base) → 값·건수 어긋남(ours)** 으로 이동했고 reward 는 둘 다 0.0. 대가는 msg 61→93, 생성 호출 110회 ·
프롬프트 누적 4.07M 토큰 [PLAUSIBLE — 이 두 수치는 재검증하지 않았다].

**⛔ 철회 목록 (088)**

| 철회한 주장 | 왜 |
|---|---|
| **`our_layer_involved: "no"`** | 판정 자체가 무너졌다(아래 두 줄) |
| *"msg 66 의 불완전 열거 = 모델의 열거 결손([[49]])"* | msg 66 은 **우리 claimprov 재생성본**(2325B→1256B, 바이트 일치)이고, 메뉴 결손은 그 안에서만 존재한다. 원본 확인 불가 ⇒ **[미판정]** |
| *"msg 45–49 구간에 우리 층 개입이 0건"* | **거짓.** fb reminder-user turn=47 channel=`channel` 축자 *"Error: [TOOL-CHANNEL] `transfer_funds_between_bank_accounts_7291` has not been unlocked yet. Call `unlock_discoverable_agent_tool(...)` first…"* + trace turn 46 `[T2_TOOL_CHANNEL] pre-call regen` ⇒ msg 47 = **0자 · tool_calls=1(unlock)** 이 우리 재생성 산출이다 |
| *"재생성 6회(모두 unified_regen)"* | 실측 **12회 · 11턴 · 4채널**(unified_regen 7 · claimprov 3 · channel 2). 교체 수지 4건 전부 바이트 일치 확인 |
| *"EXTRA 하나만으로 db_match=False 가 확정된다"* | 두 계좌 잔액이 실제로 바뀐 것은 확정이나, 세 변이 각각의 해시 기여도는 **재실행하지 않았다** ⇒ **[미판정]** (강한 [PLAUSIBLE]) |
| *"우리 층이 EXTRA 를 만들었다"* 로 읽힐 여지 | 만들지 않았다 [CONFIRMED · 형제 통과]: `T2_TOOL_CHANNEL` 은 EXTRA transfer(turn 47)와 **gold `close_debit_card_4721`**(turn 72, 같은 문면)에 똑같이 발화했고 둘 다 통과했다. 우리가 한 것은 **실행 경로 코치**이지 변이 저작이 아니다. ⚠단 `T2_CLAIM_PROV` 에는 **형제가 없다** — acting 구간에서 발화 조건이 성립한 자리는 turn 65 하나뿐이고 그것이 정확히 어긋난 메시지 위에 떨어졌다 |
| turn 68 `[BLOCKED]` 2건 = unlock 이라는 판정 | 추론으로 강등. 근거는 여전히 강하다(content=205B *"Let me get all three tools ready and execute everything."* · `tool_calls=3` = 잠긴 3종 · unlocked 카운터 6→7→8 한 턴에 하나). 그러나 **폐기된 tool_calls 인자 원문이 존재하지 않으므로**(D9) 원 보고서의 반증조건 #1 은 **원리적으로 시험 불가** |

---

#### 1c-4. 수리 후보에 미치는 영향 (D1 · D2 · D3 · D4 · D6 · L1)

| 후보 | 이 3건의 효과 | 근거 | 등급 |
|---|---|---|---|
| **D1** (종결 후 표면화 중지) | **중립~약화** — 세 건 어디에도 §1 의 절차 정체·표면화 루프가 **재현되지 않았다**. D1 의 적용 폭은 캠페인 전체가 아니라 049 계열로 좁다 | 059 msg62→66 절차 소진 · 088 KB 질의 중복 0 · 세 건 모두 `readloop-turn` 집계 없음 | PLAUSIBLE |
| **D2** (읽기 루프에 이름과 출구) | **약화** — 064 의 `get_correct_savings_apy` 14콜 중 6콜만 유효한 낭비는 **읽기 루프가 아니라 grounding 접두 드롭**이 원인이다. D2 는 이 칸을 사지 못하고 **D7 이 산다** | `t2_scaffold_get.py:724` · msg59↔64 대조 | CONFIRMED |
| **D3** (reference-filter 문면·술어 일치) | **그 게이트 자체는 미발화(중립)**, 그러나 **상위 계열 주장이 강화**된다 — *"deny 문면이 검사한 것과 다르거나 처방을 못 준다"* 의 **새 독립 사례 2건**: ① 064 grounding *"source not found in the knowledge base"*(무엇을 하면 풀리는지 없음) ② 088/064 claimprov *"None: None"*(이름 자체가 없음) | 위 축자 · [[64]] | CONFIRMED (계열) / 미판정 (D3 본체) |
| **D4** ([BLOCKED] 을 의존 호출로만) | **약화** — 이번 3건의 동반 차단은 **gold 를 죽이지 않고 지연만** 시켰다. 088 turn 68(unlock 3콜 → 한 턴에 하나씩 3턴에 전부 성공) · 064 turn 40(4콜 → msg59/64/77 에 재발행 통과). ⇒ D4 가 사는 것은 **턴 수**이지 reward 가 아닐 수 있다. 게다가 **D4 의 효과 측정 자체가 D9 에 의존한다**(폐기된 인자 원문이 없으면 무엇이 막혔는지 사후 확인 불가) | 위 · trace turn 67·69·73·77 | PLAUSIBLE |
| **D6** (중복 억제를 선언된 write 로 한정) | **중립 + ⊖ 신호 1개.** `[DUPLICATE-WRITE]` 는 이 3건에 미발화다. 그러나 **088 의 EXTRA 1건이 DB 채점을 무너뜨릴 수 있음을 보였다**(잔액 2계좌 실제 변경). D6 는 deny 를 309→88(**221건 소멸**)로 여는 레버이므로, §D6 선행조건 3(태스크별 부호표)의 **필요성이 커졌다** — "보호 상실"의 실물 형태가 EXTRA 다 | mutation_diff(088) · §2 D6 반사실 재현 | PLAUSIBLE (기여도 미분리 · [미판정]) |
| **L1** (꺼진 열거 레버 조사) | **전제 강화 · 기대수익 약화.** 강화: 이 런에서도 `[OFFICIAL-NAME]`/`T2_WRITE_ARG_ENUM` 발화 **0건**이고 `provenance.json.levers_on` 195개에 이름이 **없다**(과거 fb 파일 **56개**에는 있다) ⇒ [[81]] 배선 회귀 **CONFIRMED**. 약화: `x509_axis_queue_2026_08_24.json:183` 축자 *"모델이 낸 `account_class` 69건 중 집합 안 68(98.6%) … 게이트가 겨누는 것은 1.4% 뿐"* 이고 `"Green Account (savings)"` 는 **KB 에 실재하는 이름**이므로 켜져 있었어도 059 를 통과시켰을 공산이 크다 | 위 · 059 msgs 3·4·7 | 전제 CONFIRMED / 059 무효 PLAUSIBLE (`write_arg_enum[0].group_map` 실물 미확인 = **[미판정]**) |

**⇒ P4(L1 격리)의 exit 를 한 칸 늘려라**: *"`group_map` 에 `"… (savings)"` 형 변형이 들어 있는가"* 를 먼저
찍는다. 들어 있으면 L1 은 059 를 못 산다(그 자체가 폐기 사유는 아니고 **기대수익 0** 기록).

---

#### 1c-5. 새 후보 — D7 · D8 · D9(계기) · L2(조사) · 4칸 계약

> ⛔[[62]]: 넷 다 **격리 프로브 전에는 배선하지 않는다**. D9 는 레버가 아니라 **원장**이고, 나머지 셋의 판정
> 가능성이 D9 에 걸려 있으므로 **먼저 한다**.

##### D7 — grounding 출처 검사가 `파일명: '인용'` 접두를 삼키지 못한다 (**CONFIRMED 우리-층**)

1. **주장 + 양화** — n=1 sim(`task_064#s626729`) · 4콜 1라운드 전손 · 성분 3개 드롭. 같은 대화·같은 도구·같은
   축자 인용에서 **접두 유무로만** 결과가 갈렸다. 형제 통과 10콜이 있으므로 표적은 **문자열 형식 하나**로 좁다.
2. **근거 (축자 + 파일:줄)** — `t2_scaffold_get.py:724` `src_ok = bool(ns) and any(ns in nc for nc in norm_corpus
   if nc)` · `:121-123` `_norm_ground = re.sub(r"[^a-z0-9%]+", " ", str(s).lower())` · msg59 → msg60 *"[GROUNDING
   WARNING] 3 input value(s) could not be verified … (could not compute — check your arguments)"* · msg64(접두만
   제거) → msg65 *"Correct savings APY computed … 6.2250000000000005%"* · 계약 `a2/banking_knowledge.gate.json:1092`
   *"source": "<verbatim quote from the KB doc stating this value>"*.
   **규칙(2단)**: ① **문면 수리(무조건)** — 거절문에 요구 형식을 축자로 넣는다(무엇이 틀렸나 + 무엇을 하면
   풀리나 · [[64]]). ② **술어 완화(격리 후)** — 검사 전에 선행 `"<파일명>: "` 접두를 제거하는 정규화. 도메인
   리터럴 0, 값 선택 0([[59]] · [[23]]).
   ⛔**①이 §1c-5 머리말(*"넷 다 격리 전 배선 금지"*)의 예외인 이유 — 명시한다** (2차 리뷰 지정질문 ④):
   ①은 **거절문의 글자만** 바꾼다. `src_ok` 술어도, 무엇을 거절하는가도, 몇 건을 거절하는가도
   **불변**이다 ⇒ [[62]] 가 재라고 한 「결손」의 크기가 이 수리로 변하지 않으므로 측정 대상을 훼손하지
   않는다. 반대로 ②는 `src_ok` 를 **완화**해 거절 집합을 바꾸므로 **P7 게이트 뒤**다.
   이 면제는 [[64]](*"거부는 무엇이 틀렸나 + 무엇을 하면 풀리나 둘 다"*)가 이미 요구하는 것이고,
   그 자체가 레버가 아니다. **면제는 D7-① 하나뿐이다** — D8-④(스키마 이름) 역시 술어를 바꾸지
   않지만 **계기가 되살아나 거동이 바뀌므로** 면제가 아니라 P8 대상이다.
3. **반증 조건** — 격리에서 (i)접두형·(ii)순수 인용을 같은 components 로 넣어 **둘 다 통과하면** 이 귀속은
   거짓이다(원인은 다른 요인). / msg59↔64 사이에 새 read 가 있었다면 "코퍼스 미도달" 가설로 돌아간다(**이미
   소거: 0건**). / 접두형을 권장하는 계약 문구가 실제로 없다면 ②는 폐기하고 ①만 남는다.
4. **선행 확인** — ⛔**C3 (2026-09-04 리뷰): 이 칸이 `reports/` 를 빠뜨렸다.** 같은 거절 문면을
   **이미 소유한 선행이 있다**: `grep -rn "source not found in the knowledge base" reports/` →
   `RESEARCH_MASTER.md:854`(**C581**) · `:855`(**C582**) · `FAILURE_AXIS_AND_FIX_ORDER_2026_08_15.md:1761`.
   **C582 는 같은 문면에 다른 원인을 확정해 뒀다** — 축자 *"bm25 는 어느 조합이 자격이 되나 문서는
   주지만 그 클래스의 base APY 를 명시한 문서는 안 준다"* ⇒ 처방도 *"새 레버가 아니라 선언 수정
   (`isolate.getter_tools` 를 bm25 → shell)"* 로 **이미 정해져 있다**.
   ★D7 이 더 강한 주장인 이유는 **반증조건이 그 귀속을 실제로 깨기 때문**이다 — `msg59↔64` 사이
   **새 read 0건**이므로 *"문서를 안 줬다"* 로는 설명되지 않는다. 그러나 [[56]](근거 확보한 쪽이
   우세)·[[40]](확정 사실 인용) 상 **선행을 인용하고 무엇이 다른지 대야** 성립한다.
   ⚠이것은 §1c-1 이 059 포렌식에 대해 자백한 결함(*"prior_checked 12칸이 전부 `scripts/`·`logs/`·
   `sim_results` 이고 `reports/` 를 한 번도 grep 하지 않았다"*)이 **같은 문서 안에서 D7·D8 에 그대로
   재발한 것**이다. D8 의 「선행 확인」에도 같은 grep 을 붙인다.
   그 밖: `t2_scaffold_get.py:121-123·724` · `a2/banking_knowledge.gate.json:1092` · 형제 통과 6콜
   (msg64 4 · msg77 2) · [[22]] 근거-우선 formalize · [[64]] · §1b-refute(문면-술어 불일치 계열).

##### D8 — **선언 스키마가 소비부 계약과 어긋나 문면이 이름을 잃었다** (**CONFIRMED 우리-층 · 원인 확정**)

> ⛔**2차 리뷰 반영 중 원인이 바뀌었다.** 리뷰어는 *"식별 불가"* 라는 내 명명이 흔들린다며
> **kind 라벨 결손**으로 고쳐 부르고 *"침묵 vs tool-렌더"* 2팔을 설계하라고 했다. 그 지적을
> 따라 실물을 파 보니 **둘 다 처방이 아니다** — 라벨은 결손된 게 아니라 **우리가 뽑지 못하게
> 막아 놓았다**. 원인이 확정됐으므로 2팔 대조도 불필요하다(아래 ⑸).
> 이전 초안(*"항목이 전부 None 이면 침묵한다"*)은 **철회**한다.

1. **주장 + 양화**
```
캠페인 24런 · trace 실측
  [T2_CLAIMPROV] window hit   421행
  그중 unb_p>=1                158행
    'None' 포함                158/158 = **100%**      <- 리뷰어 양화 확인
    항목 **전부** None          151/158 (7행은 unbacked 쪽에 실명 1개: record_update · give)

캠페인 23 사이드카 · **모델에 실제로 간 문면**
  "ledger shows it was never actually executed: {claims}" 전송  **73건**
    그중 "None: None" 포함                                     **73건 = 100% · 예외 0**
```
로그 렌더가 아니라 **전송 문면**이 100% 이름을 잃었다.

2. **근거 (축자 + 파일:줄)** — 세 계약이 서로 어긋나 있고, 어긋난 지점이 셋 다 코드에 있다.
```
A2 질문        a2/banking_knowledge.gate.json:4740
               {"kind": "...", "what": "<5-10 words>", "tool": "..."}      <- 모델에게 요구한 이름
출력 스키마    t2_run_gated.py:386-404   (커밋 f6224e26 · 2026-09-01 13:40)
               claims  items.properties = {"claim", "tool", "kind"}  required=["claim"]
               pending items.properties = {"claim", "tool"}          <- **kind 가 아예 없다**
소비부 렌더    t2_gate_patch.py:14986-14988
               "%s: %s" % (c.get("kind"), str(c.get("what"))[:60])   <- **what 을 읽는다**
```
⇒ 스키마가 `what` 을 **`claim` 으로 개명**했으므로 `c.get("what")` 은 **영원히 None** 이고,
pending 에는 `kind` 자체가 없으므로 `c.get("kind")` 도 **영원히 None** 이다. 그래서 문면이
축자 *"None: None; None: None"* 이 된다.

**⑵-b 날짜 절벽이 이 귀속을 잠근다** (trace 전수 · 로컬 재현):
```
2026-09-01 까지 전 런   kind-index rescued: kind=<실명>   nonNone 전량 · None **0**
   x713_nightB 142 · x721_t1B_ctl 100 · x722_t2A 87 · x722_t2B 91 · x004_base2 75 …
2026-09-03 부터 전 런   kind=None                        nonNone **0** · None 전량
   g97151p11 349 · lost5 392 · k8143med1 461 · 049treat 286 · smoke049ad 265 …
```
경계에 있는 유일한 변경이 `f6224e26` 이다(커밋 메시지 축자: *"give the claim-provenance probe an
output schema so truncation stops disabling it"*).

**⑵-c 검정이 오히려 잘못된 이름을 못박았다** — `test_terse_schema.py` 는
`for k in ('"claims"', '"pending"', '"tool"', '"claim"')` 로 **`claim` 이 있는지**를 단언하고,
소비부 쪽은 `_j2["claims"]`·`_j2.get("pending")` 이라는 **컨테이너 이름만** 검사한다.
**항목 필드 이름은 어느 쪽도 대조하지 않는다.** 그래서 세 계약이 어긋난 채 통과했다.

3. **파생 피해 — 이건 문면 하나가 아니다** (같은 결손에서 갈라진다)
```
① 문면       "None: None"                    73/73 전송        [[64]] 정면 위반 · 라이브
② 예비-창    _claim_has_kind(_unb_p, _cpv_rsv_kinds)  :14970-14971
             pending 에 kind 가 없으므로 **행동-kind 예비 창은 열릴 수 없다**(C201/D3 사문화)
③ 과거형 구제 kind_fallback_on_miss=True 경로 :5252-5262
             지목이 원장 밖 -> kind 색인으로 강등 -> k="" -> emap.get("") = None -> **continue**
             = 항목이 **조용히 통과**한다. 050-DUP 수리(2026-08-21)가 무력화돼 있다
④ 2차 피해   문면이 user 롤을 점유해 에이전트가 없는 사용자 턴을 상상했다 —
             064 msg47 *"This appears to be a system reminder/note rather than a genuine customer
             message."* · msg86 *"the user's message is empty… This probably means the user executed…"*
             -> 송출 *"Thanks for running it."* (실행은 msg87 · 사용자는 그때 하지 않았다)
```

4. **수리 — 이름 있는 한 줄** ([[64]])
```
t2_run_gated.py:392-400   "claim"  ->  "what"           (A2 질문·소비부와 같은 이름으로)
                          pending items 에 "kind" 추가   (claims 와 같은 열거)
                          required 는 ["what"] 로        (모델이 이름을 반드시 낸다)
test_terse_schema.py      항목 필드 이름을 **소비부에서 뽑아** 대조하게 바꾼다
                          (지금은 리터럴 '"claim"' 을 단언해 오답을 못박는다)
```
도메인 리터럴 0 · 값 선택 0 · 술어 변경 0([[59]] · [[23]] · [[05]]). **억제가 아니라 복구**다.

5. **⑸ 2팔 대조가 불필요해진 이유** — 리뷰어가 제안한 «침묵» 팔은 정당한 미이행-약속 탐지를
   통째로 버리고, «tool-렌더» 팔은 `tool` 만 남겨 *"무엇을"* 을 여전히 잃는다. 원인이
   **스키마-소비부 이름 불일치**로 확정된 이상 둘 다 증상 우회다. 다만 **P8(격리)** 은 그대로
   한다 — 스키마를 고친 팔에서 문면에 실명이 실제로 들어가는지 확인해야 배선 자격이다([[81]]).

6. **반증 조건**
   - 스키마를 고쳐도 `what` 이 여전히 비면 원인은 스키마가 아니라 모델 산출이다(그때는 프롬프트 축).
   - 2026-09-01 이전 런에서 `pending` 항목의 kind 가 **원래도** None 이었음이 보이면 ⑵-b 절벽은
     claims 축만의 이야기이고 pending 결손은 더 오래된 것이다(그래도 ①②③은 그대로 성립한다).
   - `f6224e26` 이전 런에 `"None: None"` 전송이 하나라도 있으면 날짜 귀속이 깨진다.

7. **선행 확인** — ⛔C3(1차 리뷰)가 지적한 대로 이 칸이 `reports/` 를 빠뜨렸었다. 보완:
   `CLAIM_DEMAND_ISO_VS_LIVE_AUDIT_2026_08_22.md`(격리↔라이브 괴리 정본 · **격리에서 되는데
   라이브에서 안 되면 배선을 의심하라**는 그 문서의 결론이 바로 이 사례다) ·
   `git log -S TERSE` -> `f6224e26`·`2fc0225d`·`39e541a0` · [[81]] 고쳐 놓고 켠 적 없는 레버 ·
   [[84]] **레버가 강제하는 표면형은 서버 설정과 짝이다**(같은 병 — 이번엔 표면형이 아니라
   **출력 스키마**가 소비부와 짝이 아니었다) · [[64]] 거부는 이름을 대야 한다 ·
   `t2_gate_patch.py:9837`(WORK-INCOMPLETE 인접) · `fb_/trace_` 캠페인 24런 실측.

##### D9 — 재생성 폐기 원문을 **모든 채널**에서 원장에 남긴다 (계기 · 배선 자격의 선행조건) (**CONFIRMED**)

1. **주장 + 양화** — 이 태그 전역 `reminder-assistant`(폐기 원문 보존) **53행 / 53행 전부 `channel=unified_regen`**
   인 반면 `reminder-user` 59행의 채널은 `unified_regen 22 · claimprov 18 · usertoolnote 3 · channel 3 ·
   selfdecl 2 …` 다 ⇒ **claimprov·channel 재생성은 폐기 원문을 한 줄도 남기지 않는다.**
2. **근거 (축자 + 위치)** — 088 trace turn 65 `content=2325B` → `agent_response_claimprov … content=1256B`
   (= 궤적 msg 66 **1256자**, 바이트 일치)인데 **2325B 의 내용은 어디에도 없다** / 088 turn 46 `content=44B
   tool_calls=1` → `agent_response_channel … content=0B tool_calls=1`(= msg 47) / 088 turn 68 `[BLOCKED]` 2건의
   폐기 인자 원문도 없다. **귀결**: 1c-3 의 두 자리가 **확증도 반증도 불가**가 됐고, 059 포렌식이 존재하지 않는
   reasoning 을 인용한 것도 같은 구멍(영속 궤적 `reasoning_content` 0B)에서 나왔다.
3. **반증 조건** — claimprov·channel 재생성의 폐기 원문이 다른 산출물(`driver.log` 의 본문 덤프 등)에 **전량**
   남아 있음을 보이면 이 결손은 없다.
4. **선행 확인** — [[76]](서브는 진리다 — 검증 가능해야 한다) · [[70]] 판정 의무 3종 · `x509_axis_queue…` §방법_교훈
   *"레버 원장 상설화"* · `fb_/trace_` 채널 분포 실측.

##### L2 — `recommend_formalize` 격리 (서브가 **확언으로 오답**을 낸 2건 + 자재결손 1건) (**조사 · 레버 아님**)

> ★1f-7 의 067 이 **세 번째 유형**을 보탠다: 오답이 아니라 **자재결손의 옳은 답** —
> `applies=false` ×3 은 카드 로스터가 `ctx[-8:]` 창(t2_resolve.py:1076) 밖이라 나온 답이고,
> offer 는 `applies ∧ correct` 동시 필요(:1149)라 이중 불발. 재격리(로스터+기준 주입 →
> `applies=true`+Platinum 산출 여부)가 재료결손↔판단결손을 가른다. ⚠`recommendation_verify` 는
> gate.json:271 에서 이미 «DISCARD 제거 대상» 플래그 — 수리 승격 전 x509 ⑦유도 판정 인용 필수.

1. **주장 + 양화** — n=2 sim. 059 사이드카 row0·row3 → `{"applies": true, "card_type": "Gold Rewards Card"}`
   (gold = Silver Rewards Card)이고 turn 29 에 **명령형으로 뷰 주입**: *"…'card_type=Gold Rewards Card' is the
   match. **Offer it now** by calling 'give_discoverable_user_tool' …"*. 064 #31·#32 → `EcoCard`(gold = Gold
   Rewards Card). 064 프롬프트 실물(plen 5935)의 `Option details` 에 `annual_fee`·`min_score`·`cashback` **각 0회**.
2. **근거** — 위 축자 + `[T2_RESOLVE] recommendation-offer deny`(059 trace turn 27·41 · 064 driver.log 8227·8255).
   059 는 모델이 무시해서 DB 가 살았을 뿐이다 ⇒ **오답 자체가 수리 대상**([[76]]).
3. **반증 조건** — 다른 sim 의 recommend 프롬프트 `Option details` 에 `check_card_application_fit` 로스터가 실려
   있으면 "재료 결손"은 국소 사고다(그때는 판단 결손). / 재료를 채운 격리에서 **여전히** 오답이면 결손이 아니라
   판단이고, [[76]] 대로 서브를 고치거나 폐기한다. / 재료를 채웠더니 정답이면 **전달 경로**만 남는다.
4. **선행 확인** — ⚠**로스터 주입을 새 레버로 올리기 전에 반드시 인용할 판정**: `x509_axis_queue_2026_08_24.json`
   `status_2026_08_24_pm.⑦유도` 축자 *"x516(후보집합)·x517(질문 프레임) **둘 다 gold 0/39** ⇒ **경로 없음**"*.
   무엇이 다른지 대지 못하면 제안 금지([[40]] · [[74]]). 그 외 `Option details` 를 채우는 **코드 경로는 아직
   grep 하지 않았다 — [미판정]**.

##### ⛔ 새 후보로 올리지 **않는** 것 (재유도 금지)

- **`T2_SIBLING_PAREN` 의 deny 승격** — ★1f-7 의 068 이 신 sim 을 보탠다: 계기가 `'Green Account
  (checking)'` 를 **정확히 탐지하고 수리값까지 지목**했는데 log-only 미무장([[81]]). env-KB 가
  괄호형을 유효 공식명으로 못박고 있어 **deny 무장은 재발화-루프 위험 — 처방 후보는 결정론
  괄호-STRIP** 이고 gold-손상 여부는 §T-8 A/B 게이트 실측 전 미확정. 그 게이트의 정본은 그대로:
  `FAILURE_AXIS_AND_FIX_ORDER_2026_08_15.md` §T-8(:2476-2586)이 **이미**
  결함·KB 접지 불가·처방 후보·블로커 W-5(*"모델이 반려를 받고도 같은 값을 다시 보낸다 … 한 sim 최다 18회 …
  오답을 스텝 소진으로 바꾼다"*)까지 확정해 두었다. 승격 여부는 **§T-8 이 정한 게이트**(반대 팔 A/B + 반려 후
  괄호 제거율 부호표)를 그대로 따른다.
- **`account_class` 열거 검사** — 이미 `D5(철회됨)` → `L1` 로 처리된 자리다.
- **088 의 `customer_max_liability_amount`(①금액) · `transaction_type`(②범주) 에 표를 더 주는 처방** —
  x509 축자 *"②범주: x512 경계 판정 철회 · x513 표를 줘도 057·063 0/6"* ⇒ **이미 측정돼 실패한 경로**다.
- **088 의 이체 제안을 ⑦유도 축으로 접기** — ⑦유도는 `requestor=user` 축인데 이 EXTRA 는 **에이전트 자신이**
  실행했다. 접으면 오분류.

---

#### 1c-6. §5(비용 축)에 반영할 정정

- **② `sim 당 벽시계 분` 은 이번 캠페인에서 교란돼 있다** [CONFIRMED] — ours conc **4** ↔ base x644 conc **1**
  (양쪽 축자 확보). 배선 비용을 재려면 **배치 불변 지표**를 함께 적어라: task_064 `생성 토큰 52,800 ↔ 21,267
  (2.48배)` · `프롬프트 2,137,899 ↔ 2,338,132 (ours 가 더 적다)`.
- **③ 생성 호출 배수** — §5 의 task_064 분해(`agent_response 29 ↔ 부수 30`)는 08:15 미완 시점 값이다. **완주 후
  실측 79콜**(agent_response 32 · unified_regen 10 · claimprov 3 · source_claim_formalize 6 ·
  recommend_formalize 5 · intent_operator_formalize 5 · selfdecl 5 · agent_claimprov 5 · 기타)로 갱신하라.
  재생성 비율은 **13/79 = 16.5%**.
- **§5 표의 `task_064 ours 4.7시간째 미완`** → **완료: reward 0.0 · 92 msg · 301.0분**.

---

#### 1c-7. 아직 모르는 것 (원인 진술에 쓰지 마라)

- 088 turn 65 에서 폐기된 **2325B 원본의 내용** — D9 때문에 **영구 복구 불가**.
- 088 세 변이(EXTRA · dispute · order) **각각의 `db_match` 기여도** — db_check 해시 내부 미개봉, 재실행 미수행.
- 088 `excess_replacement_fee="0"` 명시가 DB 행을 바꾸는지.
- 059 `write_arg_enum[0].group_map` 에 `"… (savings)"` 형 변형이 있는지(→ L1 기대수익).
- 064 `Option details (from lookups)` 를 **무엇이 채우는지** — 프롬프트 실물만 봤고 코드 경로 미확인.
- `_t2_view_fb` 큐잉 원천이 059 turn 61 `T2_LEDGER` 하나뿐인지(다른 마커가 큐잉하면 1c-1 의 D1 렌더 근거가 흔들린다).

---

### 1d. 회귀 10건 per-step 대조 (2026-09-04)

> **회귀 = base pass ∩ ours 전량 실패**. 이 캠페인에서 **정확히 10건**이고 base PASS 42 는 전부
> 소진됐다(2026-09-04 12:0x · `x738:34-36` 와 교집합) ⇒ **더 이상 회귀가 나올 수 없다**.
>
> ⛔**F10 정합 — 「ours 전량」의 범위(scope)를 명시한다** (2차 리뷰): 이 정의는 범위를 안 적어
> 두 번 다르게 읽혔다. 확정 범위는 **이 캠페인 = `mtime >= 2026-09-03` 인 banking·Qwen3.8·
> arm=viewmax2 런의 채점 sim** 이다(§1e-1 인구조사 분모와 동일). 그보다 넓게(코퍼스 전량)
> 읽으면 `055`(x725 통과)·`059`(x644 통과)가 회귀에서 빠져 **10 이 8 이 된다** — 아래
> *"회귀는 사실이지 인과가 아니다"* 가 정확히 그 이야기다. 좁게/넓게를 섞어 쓰지 마라.
> `007 014 015 038 051 054 055 059 064 079`
>
> ⛔⛔**가장 중요한 교훈 — "회귀" 는 사실이지 인과가 아니다.**
> `1d-055` 가 그 실증이다: 같은 seed·같은 gold·같은 sha 에서 **ours 가 통과한 런이 있다**
> (`bank_x725_t3prime_A_20260901` task_055 **reward 1.0 · db_match True**). Q38+ours 에서 055 는
> **1/3** 이다. ⇒ *"우리 스캐폴드가 base-pass 를 깼다"* 는 **n=1 회귀로는 성립하지 않는다.**
> 059 도 같은 이유로 등급이 내려갔다(코퍼스 14 sim 중 gold 문자열은 x644 단 1회 · §1c-1).
> **모든 회귀 항목에 「코퍼스 전량에서 ours 가 이 태스크를 몇 번 통과했나」를 붙여라.**
>
> 채점 단위는 DB 변이다([[69]]). `1d-055` 가 그 실증도 겸한다 — **base 도 `action_checks` 2칸이
> False 인데 reward 1.0** 이다.


> **범위**: 위 머리말의 회귀 10건 중 `1d-055` 를 뺀 **9건** — `007 014 015 038 051 054 059 064 079`.
> 전부 **base PASS ↔ ours FAIL** 이다(회귀 집합의 정의는 머리말 · `x738:34-36` 교집합).
> 9건 각각에 **독립 반증 서브 1개**를 붙였다. 아래는 원 대조 보고가 아니라 **그 반증을 통과한 것만** 남긴 결과다.
>
> ⛔⛔ **머리말의 교훈을 여기에도 그대로 적용한다 — 회귀는 사실이지 인과가 아니다.**
> 「코퍼스 전량에서 ours 가 이 태스크를 몇 번 통과했나」를 실제로 붙일 수 있었던 것은 **051 뿐**이다
> (전 롤아웃 42 중 `db_match=True` **0건**, 그중 41 이 Qwen2.5-32B 레거시 ⇒ **Q38 비교쌍은 1:1**). 나머지
> 8 건은 **미측정**이다 ⇒ 아래 등급은 전부 *"이 짝에서"* 의 등급이고, 055 처럼 **한 번의 통과 런이 나오면
> 즉시 강등된다**. ★[[57]] 후속작업으로 남긴다.

#### 1d-0. 서술 단위와 판정 계약

**서술 단위는 DB 변이다** ([[69]]). 9건 중 **8건이 `reward_basis: ['DB']`** 이고, 예외는 **`task_014` 하나뿐**
(`reward_basis: ['ACTION']` · `reward_breakdown: {'ACTION': 0.0}` · `compare_args: ["reason"]`). 따라서 아래는
전부 **MISSING / WRONGARG / EXTRA** 로 세고, `action_checks` 는 어디서도 실패 단위로 쓰지 않는다.
079 는 evaluator 를 그대로 재현해 `gold d4984a97f7daf847… / base 동일(match True) / ours d6996ccc376de19e…`
를 재생산한 뒤 DeepDiff 로 변이를 뽑았고, 038 은 궤적 재실행 DB 를 표별로 diff 한 뒤 **인자 한 칸만 뒤집는
반사실**(`EXPEDITED=T match True`)까지 돌렸다 ⇒ 이 둘의 실패 단위는 **추론이 아니라 측정**이다.

**귀속 4조건** (이 절의 계약 · 넷을 다 못 채우면 `unknown`):

| | 조건 |
|---|---|
| (a) | 첫 분기점을 **메시지 단위로** 특정한다 |
| (b) | 그 분기 **직전에** 우리 층 발화(tool-deny · `[T2_*]` 표면화 · 재생성)가 있었다 |
| (c) | 그 발화가 **base 가 실제로 한 행동**을 금지·전환시켰음을 축자로 보인다 |
| (d) | 같은 sim 안에 **같은 게이트 아래 통과한 형제 호출이 없다** |

**등급 어휘** — `CONFIRMED`(반증을 걸었고 살아남음) · `PLAUSIBLE`(사슬은 서지만 반례가 인과를 약화) ·
`우리층 무관`(반증이 우리 층 쪽을 죽임) · `미판정`(4칸 미충족 · 추정으로 yes 를 쓰지 않는다).

⛔ **태그를 합산하지 마라.** ours 팔은 네 태그에 흩어져 있다 — `bank_k8141med1_20260903_2256`(007·059·064) ·
`bank_k8141med2_20260903_2256`(038) · `bank_k8143med1_20260904_0135`(015·051·054·079) ·
`bank_re151med1_20260904_0255`(014). base 대조본이 `bank_x644_q38base_bank78_20260830` 임을 **축자로 확인한
것은 007·014·051 세 건**이고, 나머지는 머리말의 base PASS 42 집합에 의존한다.

---

#### 1d-1. 판정 표

| 태스크 | 첫 분기 msg | 분기 **직전** 우리 층 발화 | DB 실패 단위 (변이 집합) | 최종 귀속 | 수리 후보 |
|---|---|---|---|---|---|
| **007** | **m1** (user-sim 개시 발화 · 에이전트 행동 **이전**) | **없음** (m33 생성에 deny 0·regen 0) | MISSING 1 = `007_0 apply_for_credit_card`(requestor=user, write) | **우리층 무관** (표적 레버 채널) · 상시 guidance 버스는 **미판정** | L2 (n↑) · reward 대응 **없음** |
| **014** | **turn 43** (transfer 초안 1·2라운드) | **있음** — `[ACTION] 'submit_referral' … do not transfer for this.` | ⚠**ACTION 채점** · MISSING 1 = `014_0 transfer_to_human_agents` | **CONFIRMED 우리층**(gold 호출 제거) + **미판정**(제거가 없었으면 1.0 이었나) | **D12(신규)** |
| **015** | **t26** (pre-give 재생성) | **있음** — `usertoolnote`·`givequote` 재생성 | MISSING 2 = `get_referral_link` 가 만드는 `referrals` 행 + `user_discoverable_tool_calls` 행 | **PLAUSIBLE** (기전은 CONFIRMED · (d) 런 수준 붕괴) | **D11(신규)** · D9 |
| **038** | **m45** (배송속도 양보) | **없음** (trace turn=40 단일 생성 · deny 0·regen 0) | WRONGARG 1 = `credit_card_orders/ccord_6ed1a491c036/expedited_shipping` `False`↔`True` (측정) | **미판정** (채점 칸) · 별건 **CONFIRMED 우리층** 1건 | **D11(신규)** · [[70]] SIGNATURE 절충표 |
| **051** | **turn 60 / m61** | **있음** — `[T2_DUP_WRITE] deny` ×4 + `[T2_PIN_READ]` + 강제 `tool_choice` | MISSING 2 = `051_8 unlock approve_credit_limit_increase_5847` · `051_9 call approve…{"new_credit_limit":5000}` (DB 델타 `credit_limit 4000→5000`) | **미판정** (두 기전 모두 4칸 미충족) | **D6 전제 재검토(필수)** · x548 `--target 051` |
| **054** | **m34** (ours) ↔ **m31** (base) | **없음** (t33 fb 0행 · `[T2_CLAIMPROV] reserve window: no action-kind claim — skip`) | MISSING 2 = `submit_credit_limit_increase_request_7392` · `approve_credit_limit_increase_5847` (키 자체 부재) | **우리층 무관** (근거 교체) · 별건 **CONFIRMED 우리층** 1건 | D8 (n↑) · reward 대응 **없음** |
| **059** | **turn 27** (recommendation-offer deny) → 하류 m68 | **있음** (간접) — deny 로 갈아끼운 m29 → m31 `cat` → m32 | WRONGARG 1 = `account_class` `"Green Account (savings)"` ↔ gold `"Green Account"` | **미판정** ((a)(b)(c) 충족 · **(d) 불충족**) | §T-8 게이트(`T2_SIBLING_PAREN`) · L2 (n↑) |
| **064** | **t30** (초안 폐기) → m35 (점수 추궁 착지) | **있음** — `[ACTION]+[ORDER]` 1262자 + `[ARG-EMPTY] credit_score` | WRONGARG 2 = `account_class` `"Silver Account"`↔`"Gold Plus Account"` · `card_type` `"EcoCard"`↔`"Gold Rewards Card"` | **CONFIRMED 우리층**(분기 **개시**) + **PLAUSIBLE**(두 값의 반사실 원인) | **D13(신규)** · L2 (n↑) · D8 (n↑) |
| **079** | **turn 44 → m45** (branch B) | **있음** — `T2_UNCALLED_UNLOCK` + `T2_GATE_REGEN claimprov` | **branch A** MISSING 8칸(freeze/unfreeze 6 + `agent_discoverable_tools` 2행) · **branch B** EXTRA 2 + WRONGARG(order 4필드) + MISSING(`btxn… -35.0`, `dcord RUSH/PREMIUM/35`) | **CONFIRMED 우리층 (branch B)** / **미판정 (branch A)** / 태스크 수준 **복구 불가** | **없음** (D8 로 안 잡힘 · [[70]] 부호표 선행) · D9 |

---

#### 1d-2. task_007 — 분기가 에이전트 행동보다 앞에 있다 (**우리층 무관**)

**(a) 첫 분기 = m1.** 양팔 m0 은 바이트 동일(`Hi! How can I help you today?`)이고 그 다음 **user-sim 개시
발화**가 갈린다 — base m1 은 `I'm Jordan Mitchell` 로 이름을 대고, ours m1 은 끝까지 대지 않는다. **에이전트
입력이 동일한 지점에서 갈렸다** ⇒ 롤아웃 사건.

**(c) 는 유보가 아니라 반증됐다.** base 의 승리 수는 m17 클로징 *"Would you like me to: 1. Walk you through
how to apply for the **EcoCard**…"* 하나인데, ours 는 **deny 전에도 후에도 그 수를 둔 적이 없다** — 폐기 초안
(fb `sha=1fa092bc3374`, len=1781)의 클로징도 배달된 m33 의 클로징도 똑같이 *"personal or business"* 자격
질문 2개다. **우리 층이 없앤 것이 없다.**

**런-내 부정통제([[57]]).** 같은 런 12 sim 중 `recommendation-offer deny` 가 난 7 sim 에서 **reward 1.0 이 4건**
(003·006·024·025). deny 는 실패의 필요조건도 충분조건도 아니다.

**⛔ 철회 (원 대조 보고)**

| 철회 | 왜 |
|---|---|
| *"m33 의 오답 승격은 우리 reminder 가 문맥에 남긴 결과"* | **토큰 산수로 기각.** GEN_TRACE `prompt=` 3연 = 초안 `34616` → 재생성 `35256`(+640) → **m33 생성 `34682`(+66)**. +66 은 m31 shell 호출 + m32 `(no output)` 뿐 ⇒ reminder 도 폐기 초안도 m33 프롬프트에 **없다** |
| *"첫 구조적 분기 = m5"* | 진짜 첫 분기는 **m1** (위) |
| *"EcoCard 를 $20 로 깎은 것이 실패 요인"* | base 도 축자로 똑같이 깎고(`the bonus is modest`) **이겼다** |
| *"(c) 는 판단 유보"* | (c) 는 **반증**이다 (위) |

**남는 CONFIRMED 우리-층 결함 2종 (reward 원인 아님 · L2 로 접는다)**
① `recommend_formalize` 게이트에 **"구별 질문을 던진다"는 탈출구가 없다** — card_type 을 정하는 데 필요한
personal/business 를 묻는 정당한 초안을 `only describing options in text (or deflecting)` 로 규정해 폐기했다.
② 같은 turn 배터리 안에서 `recommend_formalize(applies=true)` 와 `intent_operator_formalize({"tool":"none"})`
가 **서로 모순**한다. 그 결과 성공 4건에서 2~4회 발화한 `[T2_RESOLVE] user-action instruct` 가 **007 에서만 0회**다.
③ reminder 이행률 **0/1** — *"Offer it now by calling 'give_discoverable_user_tool'…"* 에 모델은
`shell grep "apply_for_credit_card"` 로 답했다(m31→m32 `(no output)`).

**계기 결손(별건)**: `perstep_diff.py` 의 sim 필터가 **메시지-레벨 deny 를 못 본다** — fb 행이 `"sim": "nouser"` /
`"sim": "ab42bebb3d8a"` 이고 task_007 은 `simtag` 에만 있다. §[5]/[6] 에 메시지 deny 채널을 추가해야 한다.

---

#### 1d-3. task_014 — 우리 문면이 **자기가 말하지 않은 도구**를 죽였다 (**CONFIRMED 제거 / 미판정 복구**)

이 9건에서 **유일한 ACTION 채점**이다. `action_checks=[{action_id:"014_0", name:"transfer_to_human_agents",
compare_args:["reason"], action_match:false}]` 이고 ours 21 tool_calls 전수 스캔에서 `transfer*` **0건**.

**(b)(c) 충족 — 3라운드의 실물.**
```
1R  trace L905 turn=40 GEN_TRACE tool_calls=1 → L908 [T2_MATERIAL_GATE] stop=other_lever(gate) calls=transfer_to_human_agents
                                              → L911 [T2_TOOL_OBS] err=True [POLICY GATE GB2_NOTICE_BEFORE_TRANSFER]
2R  L933 agent_response_unified_regen tool_calls=1 · L935 [T2_REQUIRE_DOC] surface transfer_to_human_agents
                                              → L949 Error: [ACTION] 'submit_referral' … do not transfer for this.
3R  L971 tool_calls=0 · L972 stop=resolve_cap(정체 3회) prose=True
```
**결정적 배선 2줄(로컬 재확인)** — `t2_gate_patch.py:10524` 기본 `user_action_feedback` 문면에
`"and do not transfer for this. "` 가 **조건 없이** 들어 있고, `t2_gate_patch.py:10961`
`rw_fb = ((am.tool_calls or [None])[0], _ufb) if _ufb else None` 이 그 문면을 **초안의 첫 tool_call 이 무엇이든**
그 호출의 오류 관측으로 붙인다. 그래서 `submit_referral` 얘기가 `transfer_to_human_agents` 를 죽였다.

**완화 레버가 켜져 있었는데 조건이 안 걸렸다** — `provenance.json.levers_on` 에 `T2_ACTIONREQ_GROUNDED` 가
있는데도 *"에이전트가 직접 할 수 있는 일이 남아 있지 않다 …(target=submit_referral)"* 가 **10회** 찍혔다.
원인은 `t2_gate_patch.py:3151 _delivered_unused_agent_tools` 가 **discoverable 레지스트리만** 본다는 것이다
(로컬 확인: `reg = _agent_discoverable(...)`). 손님이 명시적으로 요구한 미실행 이관도구가 있는데 집합이 비어
"남은 일 없음"으로 오판됐다 — 같은 파일 `:3175 _transfer_tools(a2)` 가 **이미 그 집합을 A2 에서 도출한다**.

**⛔ 철회**

| 철회 | 왜 |
|---|---|
| *"(d) GB2 도 [ACTION] 도 이 sim 에서 단 한 번도 통과를 허용하지 않았다"* | **거짓.** 2R 에는 `stop=other_lever(gate)` 도 GB2 오류도 없다 — 모델이 GB2 복구지시를 이행했고(`TRANSFER NOTICE: …` 축자 포함) **게이트가 열렸다**. 치명적 deny 는 **한 번**이고 그것은 GB2 가 아니라 [ACTION] 이다 |
| *"(d) 형제 통과 0건"* | **거짓.** m24 `get_referrals_by_user` · m29 `unlock_discoverable_agent_tool` · m31 `get_credit_card_accounts_by_user` 가 같은 레버 활성 중 통과 |
| *"인자까지 맞을 궤도였다"* | **근거 없음.** 모델이 낸 `reason` 값은 results/fb/trace/log **어디에도 없고**, `tools.py:48-68 TransferReasonLiteral` 은 **19지 enum** 이며 TIER-1 경쟁 코드가 존재한다 ⇒ *"우리 층이 1.0 을 앗았다"* 는 성립하지 않고, 성립하는 것은 *"그 자리의 유일한 기회를 없앴다"* 까지다 |
| *"m43 전송 거부가 우리 문면 때문"* | **절반 무너짐.** `doc_credit_cards_..._009` 축자 *"Do not transfer to a human in these cases."* 를 **양팔이 모두 회수**했고, **base 는 우리 층 0 으로 같은 거절을 스스로 썼다**(base m24) ⇒ 거부는 모델의 자율 레퍼토리 안에 있다 |
| *"git 은 base 와 동일 `fc0055dc4e0a`"* | **오독.** `engine_sha: a208c8e0` · **`engine_dirty: true`** · `bench_sha_cwd: fc0055d`(tau2-bench 쪽) |
| *"세 이름이 m18·m24·m29 로 그대로 실행됐다"* | 셋째 거짓 — m29 는 **unlock 만** 했다(`[UNLOCKED-NOT-CALLED]` 리마인더 → m35 사과) |
| *"turn 14 분기가 load-bearing"* | 아니다. 궤적이 재수렴한다(ours m37≈base m22 · ours m38≈base m23/25). 실효 비용은 **손님 턴 1칸** |

⇒ **등급: CONFIRMED(우리 층이 gold 호출을 제거) + 미판정(제거가 없었으면 1.0 이었는지).** (d) 미충족.
**계기 선행수리**: `[T2_MATERIAL_GATE]` 가 `calls=` 옆에 **인자 해시/값**을 남기지 않으면 이 태스크의 인과는
영원히 미판정이다.

---

#### 1d-4. task_015 — pre-give 재생성이 호출을 삼킨다 (**PLAUSIBLE · 기전은 CONFIRMED**)

**우리 층이 조건을 만든 것은 CONFIRMED** — `"has not been given to you"` 에러는 **base 78 sim 에 0건**이고
ours 두 sim 에만 있다.

**그러나 원 보고가 지목한 원인은 틀렸다.** `T2_TOOL_SIGNATURE deny` 는 이 런에서 **3 sim** 에 났고
**task_020 1.0 · task_021 1.0 · task_015 0.0** ⇒ 런 수준 base rate 1/3. 진짜 표적은 **pre-give 재생성이
tool_call 을 보존하지 않는다**는 것이다(런 전체 집계):

```
agent_response_usertoolnote     DROP(1->0)=2  KEEP(>=1)=1
agent_response_givequote        DROP(1->0)=2  KEEP(>=1)=1
agent_response_unified_regen    DROP(1->0)=5  KEEP(>=1)=30
```
SIGNATURE 와 독립인 사례도 있다 — `task_020 t31 agent_response_claimprov(tc=1) -> agent_response_givequote(tc=0)`
(그 sim 의 deny 는 t35·t54 뿐). **task_015 이 유일하게 진 이유 = give 를 두 번 연속 잃은 유일한 sim** 이다:

| | t26 usertoolnote | t28 givequote | 결과 |
|---|---|---|---|
| task_015 | `tool_calls=0` | `retract=1 (give_present_after_reask=0)` | 거짓 진술 **2회** · 에러 2회 · 손님 재시도 **없음** |
| task_021 | `tool_calls=0` | `retract=0 (give_present_after_reask=1)` | 거짓 진술 1회 · 에러 2회 · 손님 **실행 → 1.0** |

**⛔ 철회**

| 철회 | 왜 |
|---|---|
| *"SIGNATURE deny 가 치명타"* | 같은 게이트 3 sim 중 2건이 **reward 1.0**. deny 는 원인이 아니라 **배수** — 재생성 캐스케이드를 한 번 더 열어 2/3 확률의 주사위를 다시 굴리게 한다 |
| *"우리가 유발한 env 에러 2번이 손님을 포기시켰다"* | **완전 대조군 task_021** 이 같은 에러 2회 + 거의 축자 동일한 복구 문구 뒤에 **복구했다**(1/2) |
| *"(c) 우리 게이트 전제가 환경 사실과 어긋난다"* | 절반 오류. env 는 관대하지만(`tools.py:533-534` `arguments` 선언) **정책은 반대**다 — `additional_instructions.md:15,17` 이 give 를 **1인자**로 두 번 적었다 ⇒ 게이트는 정책상 옳다([[23]] 통과) |
| *"`(no arguments)` 라 손님이 못 돌렸다"* | **경험적 기각.** task_021 m38 이 `Arguments: (no arguments)` 인데 m41/m43 이 성공했다 |

⇒ **등급: PLAUSIBLE.** (a)(b)(c) 충족 · **(d) 는 sim 내부로만 성립**하고 런 수준에서 무너진다. 마지막 한 칸
(손님 미재시도)은 **shared** 로 내린다. 남는 CONFIRMED 는 **기전**(D11)이지 reward 귀속이 아니다.

---

#### 1d-5. task_038 — 채점 칸에는 우리 층이 발화한 적이 없다 (**미판정** + 별건 CONFIRMED)

**실패 단위는 측정이다.** DB 표별 diff 결과 **다른 칸이 하나뿐**이고(`expedited_shipping pred=False gold=True`),
그 인자만 뒤집어 재실행하면 `EXPEDITED=T match True`. ⇒ **WRONGARG 1 · MISSING 0 · EXTRA 0.**

**yes 로 못 미는 이유** — 양보가 일어난 m45 는 `trace turn=40 [T2_GEN_TRACE] call=agent_response … tool_calls=0`
**단 한 줄**이다(deny 0 · regen 0). fb 의 `expedited` 17히트는 **전부 `kind=subcall` 프롬프트 payload**,
trace 0, 런 로그 0 ⇒ **우리 층이 배송속도를 언급한 횟수 = 0.** 정책 축자
(*"strongly recommend expedited shipping"*)는 양팔 다 회수했고 base 만 되받았다.
**no 로 못 미는 이유** — 선택지를 제시한 m43 자체가 `[SIGNATURE] deny → unified_regen → givexec → claimprov`
캐스케이드의 산물이다. **무대는 우리가 만들었고 대사는 모델이 썼다.**

**⛔ 철회**

| 철회 | 왜 |
|---|---|
| *"(d) 형제 통과 있음 → 인과 약함"* | **shape 혼동.** deny 된 네 번은 전부 `arguments` 를 실은 give 이고 **0/4 통과**, 실행된 give 2회는 **인자 없는 별개 형태**다 ⇒ [SIGNATURE] 에 대해 (d) 는 **충족** |
| *"(c) 우리 발화가 base 행동을 금지하지 않았다"* | **좁은 축으로만 검사했다.** base m31/m37 은 `arguments` 를 실은 give 를 실제로 했고 우리가 그것을 4회 금지했다 ⇒ **give 축에서는 (c) 충족**(채점 칸 축에서는 여전히 불충족) |
| *"[SIGNATURE] 은 채점을 위해 필요하다"* | `t2_signature.py:10-11` 의 근거는 **action_checks 논거**다. env 의 give 는 DB 에 `{"tool_name","status"}` 두 칸만 쓰므로(`tools.py:582-585`) `arguments` 는 **원리적으로 DB 해시를 못 움직인다** ⇒ DB-basis 태스크에서 이 레버는 **채점축에 아무 것도 사주지 않으면서** 16 메시지를 팔았다([[70]]) |

**별건 CONFIRMED 우리-층 결함** — turn 28 에서 `usertoolnote` 재생성이 **이미 규격에 맞는(=gold `038_1` 과
동일 형태) give 호출을 파괴**했다. 대가 = 6 메시지 우회 + `Unknown discoverable tool 'retrieve_last4'` /
`'get_card_last4'` **2회**. ⇒ **015 와 같은 기전이 다른 태그에서 재발**했다 ⇒ **D11.**

---

#### 1d-6. task_051 — 이 절의 최대 산출은 **D6 의 출시 전제가 거짓이 됐다**는 것 (**미판정**)

**실패 단위**는 base/ours `action_checks` 를 전 항목 대조해 좁혔다: `051_0`~`051_7` 은 **두 팔 값이 완전히
동일**하고 갈리는 것은 `051_8`·`051_9` 뿐(DB 델타 `credit_limit 4000→5000`, base m64 축자
*"Previous Limit: $4000.00 - New Limit: $5000.00"*). 지운 두 번째 submit 은 **어떤 채점 단위도 잃지 않았다**
(`ours 051_7 action_match=True`).

**⛔ D6 에 직접 걸리는 것 (로컬에서 내 손으로 재확인)**

1. `t2_gate_patch.py:12263-12264` 축자 — *"⚠**알려진 노출**: 051 은 gold 가 거절·상환 뒤 같은 인자 재제출을
   요구한다 … 이 가드는 그것도 막는다 … **051 은 코퍼스 전 sim 이 0점이라 실제로 잃은 점수는 없다.**"*
   ⇒ **그 출시 전제가 지금 거짓이다.** `base x644 task_051 reward=1.0`(2026-08-30)이 코퍼스에 들어왔다.
2. `t2_gate_patch.py:12256-12257`·`go_stack.sh:692-694` 는 부정통제를 *"재발행 4/4 → 0/4 … 이름 없는 거절(4/4)·
   같은 길이 무관 문장(4/4)은 못 막는다"* 로 인용하는데, **정본 JSON 은 그 숫자를 담고 있지 않다** —
   `x548_dup_deny_iso_2026_08_26.json` 의 유일한 행(`rows` 길이 **1** · `target:"074"` · n=4) tally 는
   `A_live/B_bare/C_proceed/D_escape 전부 {reissue:0, acted:0}` · `N_len {reissue:0, acted:4}` 다.
   **팔을 가르는 양은 `reissue` 가 아니라 `acted`(출시 문면 0/4 ↔ 길이 맞춘 통제 4/4)** 이고, `4/4` 기준선은
   파일에 없다 ⇒ **인용 수치 정정 필요**([[40]]).
3. **배선 불일치([[81]]·[[54]])** — 정본은 `go_stack.sh:695 export T2_DUP_WRITE=0`(기본 OFF)인데, 라이브
   런처는 전부 `=1` 이다: `run_ours_task.sh:128` · `run_night_ab.sh:65` · `run_t7363_night.sh:57` ·
   `run_t7364.sh:64` · `run_t7365.sh:63`(로컬 grep). **이 회귀 런은 정본 스택이 아니다.**

**⛔ 철회**

| 철회 | 왜 |
|---|---|
| *"현재 없는 부정통제다([[57]])"* | **거짓.** `x548_dup_deny_iso.py`(444줄 · 팔 5개)가 2026-08-26 부터 있고 **051 케이스 빌더까지 이미 있다**(`:284`). 없는 것은 프로브가 아니라 **051 행**이다 ⇒ [[74]] 위반 |
| *"가장 싼 수리 = 문구를 다음 행동 지명으로 바꾼다"* | **이미 저작됐고 0/4 로 실패했다** — `DENY_ESCAPE` 축자 존재 · `D_escape {reissue:0, acted:0}` ⇒ [[40]] 재유도 |
| *"CONFIRMED / 강등 논쟁"* | 노출은 **런 이전에 소스에 적혀 있었다**(위 1) ⇒ 재발명 |
| *"deny 8건 / route 5회"* | 계수 오류. DUPLICATE-WRITE 는 `tool-deny` 8 중 **4**(turn 61·63·65·67) · route `dup_write` **4** · trace deny **4** |
| *"분기 직전 우리 층 발화 없음"* | 놓쳤다. turn 60 에 `[T2_DUP_WRITE] deny` + `[T2_PIN_READ] pinned call_discoverable_agent_tool(agent_tool_name=[4개])` + **강제 `tool_choice`** 가 동시 발화했고, 결손 gold `051_8` 은 그 화이트리스트에 **없다** ⇒ 그 turn 에 **디코딩 제약으로 선택 불가**였다 |
| *"git 은 base 와 동일 `fc0055dc4e0a`"* | `git cat-file -t` → `fatal: Not a valid object name` ⇒ **검증 불가** |

**4칸 판정** — dup_write 기전: (a)✅ (b)✅ (c)**협의만**(지운 호출은 DB 0행) (d)✅ / PIN_READ 기전: (a)✅ (b)✅
(c)**하드 충족** (d)**❌**(`unlock_discoverable_agent_tool` 이 turn 42·52·57 에서 자유 통과). ⇒ **미판정.**
**결정적 시험은 이제 싸고 인프라도 있다**: `x548_dup_deny_iso.py --target 051`, 창을 base m52 로 잡고
모델을 **Qwen3.8-27B-FP8 로 맞춰서**(기존 x548 은 Qwen2.5-32B).

---

#### 1d-7. task_054 — 결론은 살고 **근거가 통째로 교체됐다** (**우리층 무관**)

**(b)(c) 불충족은 재검증됐다** — t33 fb 비-subcall 0행, `[T2_CLAIMPROV] reserve window: no action-kind claim —
skip` · `[T2_UNAVAIL] promised tools not available: []`, 그리고 t33 의 `agent_response` GEN_TRACE 는 **1회뿐**
(`content=1042B` ≈ m34 1048B). **m34 시점 컨텍스트에 영속하는 우리 층 문자열은 KB 출력 꼬리의
`[axis] matches: 4 documents …` 건수 한 줄뿐**이다(m0~m34 전수 스캔 히트 0).

**⛔ 철회**

| 철회 | 왜 |
|---|---|
| *"유저 질문이 base m30 과 같은 취지로 m33 에 왔다"* | **축자로 거짓 — 대조의 전제가 깨진다.** base m30 은 CLI 질문 **하나만** 담았고, ours m33 은 *"Please go straight to disputing it."* + *"send the replacement card to …"* 라는 **사기 처리 명령 2건이 선행**하고 CLI 질문은 맨 끝 부록이다. 이것이 분기의 가장 강한 대안설명이다 |
| *"재생성 없음"* | **turn 33 만 참.** sim 전체로는 t6·t9·t11 에서 우리 층이 모델 텍스트(715B·587B·521B)를 버리고 도구호출로 교체했다 |
| *"잔여 채널은 axis note 와 guided grammar 둘뿐"* | 열거 누락(`T2_SURFACE_BUS guidance:attached` ×2 · `T2_A2_VARIANT` · 매 턴 서브 5종) |
| *사전 판정* *"우리 [OPERATOR-SCOPE] 가 그 오선택을 지적했다"* | **틀렸다.** deny 는 t36 이고 갈래는 m34/m35 에서 이미 끝났으며, deny 가 지목한 두 도구는 gold `054_13~054_16` 이 전부 `match=True` 인 **정답 도구**다 |

**내가 세웠다가 스스로 죽인 대안가설 2건** — ① *"우리 층이 shell 을 억제했다"*: base **471회/58태스크(74%)**
↔ ours **52회/9태스크(75%)** ⇒ 억제 없음. ② *"CLI 적격 사실을 몰랐다"*: ours m34 축자 *"**Yes, technically we
could do the CLI first.**"* ⇒ 지식 결손이 아니라 판단 차이.

**별건 CONFIRMED 우리-층 결함 (D8 강화 · reward 원인 아님)** — claimprov 가 **`unbacked=0` 인데** 3회 발화해
텍스트를 폐기했고, 전송 문면의 항목 이름이 전부 `None` 이다: *"…never actually executed: **None: None; None:
None**"* / *"…are in YOUR OWN tool list, not the customer's: **None** (tool: get_credit_card_transactions_by_user)"*.
강제된 산출물은 손님이 말한 적 없는 `customer_name="Alex Morgan"` ×3 → 전부 `No records found in 'users'.`
(읽기 전용 · DB 무영향 · 111초). **reward 원인은 아니다** — 두 팔 모두 신원확인이 m14 에 착지하고 m16~m30 이
도구 정체 기준 완전 일치한다.

---

#### 1d-8. task_059 — `no` 에서 **`미판정`** 으로 올린다 (간접 사슬 (a)(b)(c) 충족 · (d) 불충족)

§1c-1 과 같은 sim 이다. **직접 인과는 여전히 없다** — m68 을 만든 생성 직전의 우리 층 발화는 `[COMPUTED FACTS]`
원장 주입뿐이고 상품명·`account_class` 를 말하지 않는다. 그러나 **간접 경로가 축자로 선다**:

```
turn=27 [T2_RESOLVE] recommendation-offer deny
turn=27 GEN_TRACE agent_response            content=2772B tool_calls=0   ← 폐기된 원본
turn=27 GEN_TRACE agent_response_unified_regen content=138B tool_calls=1  ← 살아남은 m29
```
폐기된 2772B 는 fb 에 남아 있고(`sha a87199805434`) *"## My recommendation … **Green Savings Account + Silver
Rewards Card**"* 다 — **base 가 실제로 한 행동**(base m24 추천 → m25 손님 실행)과 같은 수다. ours 궤적에는 그
메시지가 **한 줄도 없다**(`"My recommendation"` 등 3종 문자열 0건). 그 자리에 들어간 m29 는
`KB_search_bm25("apply for credit card tool apply_for_credit_card …")` 였고 → m31 `cat …general__001` →
**m32 에 두 팔 통틀어 유일한 줄**이 들어온다: *"- Personal checking account_class options must use the full
official name ending with 'Account' (e.g., 'Blue Account', **'Green Account (checking)'**)."*
(`account_class options` = **BASE 0건 / OURS 1건**).

**⛔ 철회** — *"분기 직전 우리 층 발화 없음"*(→ `[T2_FB_VIEW]` 원장 주입이 turn 63·65·67 **3회**, 큐잉과
주입을 혼동했다) · *"우리 층 발화 3건(deny)은 전부 다른 자리"*(→ 재생성 **6회**, 그중 2회가 추천 국면을 통째로
삭제) · *"첫 분기는 m2 retrieval coin-flip 이고 무게가 없다"*(→ turn 27 에 무게 있는 분기가 따로 있다).

**그럼에도 (d) 가 불충족이고, 대안 설명이 셋 살아 있다**: ① 괄호 노출 자체는 원인이 아니다(base 도 m3·m4·m5·
m10·m19·m20 에서 반복 봤고 자기 추천문에도 썼는데 m41 은 `"Green Account"` 였다) ② **정답 규칙이 같은 메시지
안에 있었다**(*"It must be the full official name ending with \"Account\""*) ③ m32 → m68 사이 거리가 **37
메시지**이고 그 사이 이 인자를 향한 계기가 0이다 ④ 같은 게이트가 turn 54 에 `passed` 로 통과했다.
⇒ **미판정.** 넷 중 셋만 충족.

**부수 (우리 층 오답 · [[25]] 위반)** — `recommend_formalize` 가 turn 29·43 에 두 번 다
`{"applies": true, "card_type": "Gold Rewards Card"}` 를 내고 *"'card_type=Gold Rewards Card' is the match.
**Offer it now**"* 로 명령형 주입했다. gold·base 는 **Silver Rewards Card** 다. 모델이 무시해 reward 손해가
없었을 뿐 ⇒ **L2 의 n 을 3(007·059·064)으로 올린다.**

**`T2_SIBLING_PAREN` 은 계기다(레버 아님)** — `t2_gate_patch.py:13311` 의 `in ("log","deny")` 본문은
`print(..., file=stderr)` 한 줄뿐이고, 배선 grep 은 `arms/*.env` 4건 + 검정 1건이며 **`go_stack.sh` 0건**.
런 전체 발화 **정확히 1회**(task_059) ⇒ [[81]] 사례. **승격 여부는 §1c-5 대로 `FAILURE_AXIS_AND_FIX_ORDER
_2026_08_15.md` §T-8 이 정한 게이트를 따른다. 새 후보로 올리지 않는다.**

---

#### 1d-9. task_064 — 분기의 **개시자는 우리가 주입한 도구**다 (**CONFIRMED 개시 / PLAUSIBLE 값**)

**⛔ 철회 (원 대조 보고의 핵심 축)** — *"t30 이전 우리 층 발화는 없다 / 신용점수 질문은 모델 자신의 것"*:
**무너졌다.** `[T2_SCAFFOLD_GET] injected name=check_card_application_fit … params=[… 'credit_score' …]` 이
sim 시작부터 서 있었고, 그 설명 첫 문장이 `banking_knowledge.gate.json:3254` 축자
*"**MANDATORY before recommending or applying for ANY credit card**: formalize the customer's stated
constraints (… **their credit score** …) and call this tool."* 다. 모델이 **m17 reasoning(t30 보다 55분 전)**
에서 *"**The check_card_application_fit tool requires formalizing the constraints.**"* 로 그 도구를 이름으로
지목한다. base 78/78 에는 이 문자열이 **0건**이다.

**부정통제가 반대 방향으로 결정적이다** — 같은 KB·같은 seed·같은 모델·같은 인덱스에서 정반대 결정:
base m44 *"**I can't check their credit score; the application will handle that.**"* ↔
ours m28 *"**I need to ask about their credit score to confirm eligibility.**"*

**사슬 6단(전부 축자)** — ①주입 → ②채택(m17) → ③삭제(t30 `[ACTION]+[ORDER]` 1262자가 *"🏆 Best combination:
Gold Plus savings + Gold Rewards Card"* 초안 2481자를 지움) → ④강제(`[ARG-EMPTY] … 'credit_score' … Re-issue
the call with … filled in` → 모델이 `credit_score="850"` 을 **지어냄**) → ⑤박탈(`[GROUNDING WARNING] …
credit_score=850 (the customer never mentioned this…)`) → ⑥착지(m35 점수 추궁 → m36 손님 거절 → m81
*"let's go with \"1\" — open the **Silver Account**"* → m84/m87 두 WRONGARG).

**그 [ORDER] 게이트는 정책 근거가 없다 — 우리 정본이 자백한다.** `banking_knowledge.gate.json:4653`
`_note_require_tool_before` 축자(로컬 재확인): *"나머지 체인(`apply_for_credit_card ←
check_card_application_fit` 등)의 **선행 도구는 우리가 만든 scaffold GET 도구**이고, **그 체인을 요구하는
정책 문장은 없다** … ⇒ **[[23]] 소급 대상**"*. ⇒ **D13.**

**⛔ 그 밖 철회** — *"(c) 는 단계를 추가한 것"*(범주 오류 — base 가 한 행동은 *추천 발화 송출*이고 우리가 그
발화를 지웠다 ⇒ (c) 충족) · *"(d) `SIBLING apply_for_credit_card passed_before=[73]`"*(**grep 자기매칭 위양성**
— m73 은 shell 이고 명령에 그 문자열이 들어 있다; `[ORDER]` 태그는 이 sim 전체에서 **1회**뿐 ⇒ (d) 충족).
**계기 정정 2건**: *"m40 의 4회는 [READ-FIRST] 로 반려"* → 실제는 `[ARG-EMPTY]×1 + [BLOCKED]×3` ·
*"`check_card_application_fit` 29회"* → **문자열 등장 수**이고 **호출은 1회(m30)**.

**값까지 yes 로 확장하지 않는 이유** — 지워진 t30 초안의 마지막 줄이 이미
*"**roughly what's your credit score**, and do you already have a Rho Bank Plus subscription?"* 였고,
페르소나 검증정보에 **신용점수가 없다**(Name/Phone/Email/DOB/Address/Annual Income 뿐). 초안이 그대로
착지했어도 같은 거절이 왔을 개연성이 높다. 게다가 m80 은 Gold Plus 를 숫자째 유지했는데도 손님이 "1"을 골랐다.
⇒ **분기 개시 = CONFIRMED · 두 값의 반사실 원인 = PLAUSIBLE.**

**별건**: `[RECOMMEND-OFFER]` 가 t86 에 *"'card_type=EcoCard' is the match. Offer it now"* 로 **gold 와 반대
값을 승인**했다(L2 n↑) · claimprov `None` 렌더 결함 2건(D8 n↑).

---

#### 1d-10. task_079 — **우리 층이 실패 변이를 직접 만들었지만, 우리 층만 고쳐도 통과하지 못한다**

evaluator 를 재현해 `base match True / ours match False` 를 재생산한 뒤 DeepDiff 로 갈랐다. 변이가 **두
branch** 이고 **각 branch 가 단독으로 `db_match=False`** 를 만든다(AND 채점).

**branch B = 우리 층 (CONFIRMED).** 두 자리에서 모델의 원안은 **도구 없는 말-턴**이었고 우리 재생성이
도구-턴으로 바꿨다:
```
turn=35 agent_response content=1965B tool_calls=0 → [T2_CLAIMPROV] window hit(resign) → agent_response_claimprov content=178B tool_calls=2
turn=44 agent_response content=1227B tool_calls=0 → [T2_UNCALLED_UNLOCK] surface order_debit_card_5739
                                                  → agent_response_claimprov content=217B tool_calls=1
```
영속 m45(217자 축자 일치): *"**You're right — I should have executed the orders immediately rather than
waiting.**"* — **그 "You're right" 의 수신자는 우리 층뿐이다**(m38↔m53 사이 user 메시지 **0건**;
fb turn=45 *"[UNLOCKED-NOT-CALLED] … call it now with its arguments"* · *"Do the promised work NOW …"*).
base 는 같은 자리에서 정책을 되묻고(m60) 손님 확인을 받은 뒤(m61) **RUSH/PREMIUM/$35** 로 발주했다(m67).
ours 는 확인 없이 **3건을 Standard/Classic 로** 질렀다(m47/49/51) — 그러고 **m55 에서 스스로 정답을 말한다**
(*"the quickest option … is **RUSH** … **$35 fee**. I already placed your Evergreen order with expedited."*).

**branch A 는 강등한다 — `no` → 미판정.** freeze/unfreeze 8칸의 원인으로 지목했던 "손님 발화 차이"의 그
손님 발화(m6)가 답한 상대 메시지 **m5 자체가 claimprov 재생성 산출물**이다(`turn=3 agent_response 749B tc=0
→ agent_response_claimprov 462B tc=0`, 영속 m5 = 462자). **원안 749B 가 유실돼 방향을 증명할 수 없다**(D9).

**⛔ 철회** — *"분기 직전 우리 층 발화 없음"*(fb 만 보고 내렸다. **claimprov·channel 재생성은 fb 에 흔적을 남기지
않는다** — trace 를 봐야 한다) · *"원안 복원 불가라 부차가설 기각"*(부분 거짓 — **본문은 없지만 `tool_calls` 수와
바이트는 남아 있고** 그 둘이 0→N 을 증명한다) · *"WRONGARG 는 검색 결손 때문"*(보조 원인일 뿐. m55 축자가 반박).

**D9 의 사정거리를 좁힌다** — 없는 것은 *존재*가 아니라 **본문**이다. `T2_GEN_TRACE` 는 `content=B tool_calls=N`
을 채널별로 남기므로 **호출 집합의 변화는 사후 판정 가능**하다. D9 의 요구는 *"폐기 원문"* 에 한정한다.

⇒ **등급: branch B CONFIRMED 우리층 / branch A 미판정 / 태스크 수준 = 우리 층 수리로 복구 불가(CONFIRMED).**
**수리 후보는 만들지 않는다** — branch B 를 막는 술어(*"실행 압박 재생성이 새 write 를 만들지 않는다"*)는
claimprov 의 존재 이유를 통째로 파는 것이라 [[70]] 절충이지 결함 수리가 아니다. 선행은 **claimprov ON/OFF
태스크별 부호표**다. (런 전체 base rate: claimprov 재생성 18회 중 원안 0콜 15회, 그중 **8/15 가 0→≥1**.)

---

#### 1d-11. ⇒ 회귀 9건 중 **우리 층 수리로 되돌릴 수 있는 것은 몇 건인가**

**지금 근거로 확정할 수 있는 것은 `0`건이다.** 하나도 "고치면 1.0 이 돌아온다"를 반증까지 통과시키지 못했다.
그 아래를 이유별로 가른다:

| 구획 | 건수 | 태스크 | 근거 |
|---|---|---|---|
| **A. 되돌릴 수 있음이 확정** | **0** | — | 9건 중 (a)(b)(c)(d) 를 다 채운 채 **반사실까지 확인된 것이 없다** |
| **B. 시험을 통과하면 돌아올 수 있는 후보** | **5** | 014 · 015 · 038 · 051 · 059 | 각각 결정적 시험이 아래에 있다. 전부 **미검증** |
| **C. 우리 층 무관** | **2** | 007 · 054 | 분기가 **에이전트 행동 이전**(007 m1)이거나 **분기 자리에 우리 층 발화 0**(054 t33). 054 는 손님 턴 내용 자체가 갈렸다 |
| **D. 우리 층 개시는 CONFIRMED 이나 반사실이 부정적** | **1** | 064 | 지워진 초안이 **이미 신용점수를 묻고 있었고** 페르소나에 그 값이 없다 ⇒ 삭제를 되돌려도 같은 거절이 올 개연성이 높다 |
| **E. 우리 층을 고쳐도 통과 불가** | **1** | 079 | branch A(freeze 8칸)가 **단독으로 `db_match=False`** 를 만든다. branch B 만 고쳐도 0.0 |

**B 5건의 결정적 시험 (전부 [[57]] 부정통제 · 무료 격리가 먼저)**

| 태스크 | 시험 | 통과 판정 |
|---|---|---|
| 014 | `[ACTION] user-action feedback` **만** 끈 팔로 turn 43 재생성 + **`reason` 값 기록** | transfer 가 나가고 `reason` 이 gold `unconfirmed_external_communication` 이면 A 로 승격. **선행 수리**: `[T2_MATERIAL_GATE]` 에 인자 값/해시 기록(없으면 영구 미판정) |
| 015 | D11 을 넣어 pre-give 재생성 보존율을 100% 로 만든 뒤 **같은 sha·seed** 재실행 | `has not been given to you` 0건 + `get_referral_link` 실행이면 A. 쌍둥이 021 이 1/2 로 복구했으므로 **n=1 로는 못 세운다** |
| 038 | `T2_TOOL_SIGNATURE=0` **만** 끈 대조 팔(seed 626729) | `expedited=true` 로 끝나면 A, `false` 면 **C 로 확정**(우리층 무관) |
| 051 | `x548_dup_deny_iso.py --target 051`, 창=base m52, **모델 Qwen3.8-27B-FP8** | `C_proceed` 가 base 의 m59→m61 전환을 얼리면 B 유지·수리 표적 확정. 그리고 **D6 의 출시 전제(*"051 은 잃을 점수가 없다"*)를 문서에서 먼저 정정한다** |
| 059 | §T-8 게이트대로 `T2_SIBLING_PAREN` 반대 팔 A/B + 반려 후 괄호 제거율 부호표 | 한 칸이 reward 0→1 이고 런 전체 발화 1회라 부작용 위험은 실측상 최소. **§T-8 밖에서 승격 금지** |

---

#### 1d-12. 신규 수리 후보 3개 — 4칸 계약 ([[77]])

> ⛔ 셋 다 **격리 프로브 전에는 배선하지 않는다**([[62]]·[[78]]). 셋 다 도메인 리터럴 0 · 값 선택 0 ·
> gold 미접촉([[05]]·[[23]]·[[59]]).

##### D11 — 재생성이 원본의 **env-변이 호출을 잃으면 교체를 기각**한다 (015 · 038)

1. **주장 + 양화** — pre-give 재생성 두 채널이 초안의 `give_discoverable_user_tool` 을 떨군다.
   `bank_k8143med1_20260904_0135` 전역: `usertoolnote` DROP(1→0)=2 / KEEP=1 · `givequote` DROP=2 / KEEP=1
   (= **pre-give 4/6 = 66.7%**) · `unified_regen` DROP=5 / KEEP=30(14.3%). SIGNATURE 와 **독립인 사례 1건**
   (`task_020 t31`). 다른 태그(`bank_k8141med2`)의 **038 에서 같은 형태가 재발**했다(deny 4회 전부 arguments
   실은 give · 0/4 통과 · 재발행된 규격 give 를 `usertoolnote` 가 파괴).
2. **근거 (축자 + 파일:줄 · 로컬 재확인)** — `t2_gate_patch.py:15207-15224`
   `_new5 = _ap_regen("Note: " + _tpl5.format(tool=_want5), "usertoolnote")` / `if _new5 is not None: am = _new5`
   ⇒ **호출을 잃어도 무조건 교체**. `:15335-15364` 도 같은 구조(`am = _new1p`)이고 바로 아래에
   `print("[T2_GIVE_QUOTE] retract=%d (give_present_after_reask=%d)")` — **계기가 그 손실을 이미 세고 있는데
   아무 것도 하지 않는다**. 실측 축자: `t26 agent_response_usertoolnote … content=1428B tool_calls=0` ·
   `t28 retract=1 (give_present_after_reask=0)` ↔ `task_021 t32 retract=0 (…=1)` · 038 `t28
   agent_response_unified_regen (required STRIPPED) content=0B tool_calls=1` → `agent_response_usertoolnote
   content=419B tool_calls=0`. 하류: `Error: Unknown discoverable tool 'retrieve_last4'` / `'get_card_last4'` ·
   `has not been given to you`(**base 78 sim 0건**).
   **규칙(닫힌 술어) — ⛔2026-09-04 §1f 에서 양방향으로 재작성**:
   ⓐ (구판·잃음 방향) *재생성 전 `tool_calls` 의 env-변이 부분집합 ⊄ 재생성 후 `tool_calls` → 교체 기각*
   ⓑ (신설·더함 방향) *재생성 후에 **새로 생긴** env-변이 호출은 **원 게이트 체인을 통과해야** 커밋*
   ★근거: §1f 배치의 피해 7건은 전부 ⓑ 방향(초안 0 -> 재생성이 gold 밖 write 를 더함)이고,
   구판 ⓐ 만으로는 **한 건도 못 잡는다**(비평 4-3). ⓑ 는 D14 와 같은 표적의 술어 표현이다.
   (도구 이름 열거 0 · 변이 집합은 A2 도출).
3. **반증 조건** — ⑴ 격리에서 pending give 를 실은 프롬프트로 두 채널을 돌려 **보존율이 이미 100%** 면 결손이
   없다. ⑵ 보존형으로 고친 뒤 같은 sha·seed 에서 015 가 여전히 `db_match=False` 면 이 후보는 015 를 못 산다
   (038 도 동일). ⑶ 038 turn 28 의 `unified_regen` 산출 `tool_calls=1` 이 give 가 **아니었음**을 보이면 038
   사례가 무너진다(현재는 게이트 가드가 give 를 요구하고 로그가 그 인자 `_want5='get_card_last_4_digits'` 를
   출력했으므로 성립). ⑷ 교체 기각이 SIGNATURE 문면을 무력화해 `arguments` 실은 give 가 다시 늘면
   [[70]] 절충으로 내려간다.
4. **선행 확인** — `t2_gate_patch.py:15207-15224·15335-15364`(로컬) · `t2_signature.py:10-11`(레버 자기 근거가
   **action_checks** 논거임) · env `tools.py:533-534·582-585·4149·4245·4517-4566` ·
   `banking_knowledge/prompts/components/additional_instructions.md:15,17,27` · [[81]] ·
   **아직 안 한 것**: `git log -S "_ap_regen"` + `LEVER_ROSTER_CANONICAL` 대조 = **[미판정]**.

##### D12 — `user_action_feedback` 를 **초안의 아무 호출에나 붙이지 않는다** (014)

1. **주장 + 양화** — n=1 sim(`task_014#s626729`)에서 `submit_referral` 을 겨눈 문면이
   `transfer_to_human_agents` 초안에 붙어 **gold 호출을 궤적에서 0회로 만들었다**. 같은 sim 에서 그 문면 안의
   무조건절이 **fb 4행(13·28·38·117)에 전부** 실렸다(substring 검산 소스 True). 부수로 `[T2_ACTIONREQ]
   침묵 안 함 …(target=submit_referral)` 이 **10회** 오판됐다.
2. **근거 (축자 + 파일:줄 · 로컬 재확인)** — ⓐ `t2_gate_patch.py:10524` 기본 문면에
   `"and do not transfer for this. "` 가 **조건 없이** 들어 있다. ⓑ `t2_gate_patch.py:10961`
   `rw_fb = ((am.tool_calls or [None])[0], _ufb) if _ufb else None` ⇒ **첫 tool_call 이 무엇이든** 그 호출의
   오류 관측으로 붙는다. ⓒ `t2_gate_patch.py:3151 _delivered_unused_agent_tools` 가 `_agent_discoverable`
   레지스트리만 보므로 **미실행 이관도구**가 침묵 자격 계산에서 빠진다 — 같은 파일 `:3175 _transfer_tools(a2)`
   가 이미 그 집합을 A2 에서 도출한다. 실측 축자: `fb…jsonl:117 turn=43 kind=tool-deny`
   *"Error: Error: [ACTION] 'submit_referral' … do not transfer for this."*
   **규칙(닫힌 술어)**: ⓐ 그 절을 `_utgt` 조건절로 좁히거나 삭제 · ⓑ `_ufb` 는 `_utgt` 계열 호출에만 부착
   (특히 `_transfer_tools(a2)` 원소에는 금지) · ⓒ 침묵 자격 집합에 `_transfer_tools(a2)` ∩ 손님 요구분 포함.
3. **반증 조건** — ⑴ ⓐⓑ 를 고친 팔에서도 turn 43 에 transfer 가 안 나가면 원인은 다른 곳이다.
   ⑵ 나갔는데 `reason` 이 gold 와 다르면 **수리해도 reward 는 안 돌아온다**(19지 enum · TIER-1 경쟁 코드
   `customer_demands_after_unavailable_offer_refusal` 존재) ⇒ 수리 근거는 유지하되 회귀 복구 주장은 폐기.
   ⑶ ⓒ 를 넓혔더니 `[T2_ACTIONREQ]` 침묵이 과도해져 다른 태스크에서 user-action 지목이 사라지면 [[70]] 절충.
4. **선행 확인** — `grep -rl "do not transfer for this" reports/` → `refute_2026_08_24/refute_016.json` ·
   `refute_072.json` · `refute_073.json` · `x505_TASK_073_t7348_perstep.md` ⇒ **같은 결함 가족이 이미 CONFIRMED
   로 박제돼 있다**(072/073: `intent_operator_formalize` 오바인딩 + 동일 문면). **014 는 그 가족의 새 표적이지
   새 축이 아니다.** · `x509_axis_queue_2026_08_24.json.per_task_required` 에 **014 없음**(축 재발명 아님) ·
   `N97_TASK_ROOT_CAUSE_2026_08_06.md:276` 의 014 진단은 **다른 실패 모드**(이번 런은 링크를 주지 않았다).

##### D13 — **정책 근거 없는 scaffold-GET 선행조건**을 `[ORDER]` 로 강제하지 않는다 (064)

1. **주장 + 양화** — n=1 sim(`task_064#s626729`)에서 `[ORDER]` 가 *"the prior read(s) this action requires have
   been done (do it with: **check_card_application_fit**)"* 로 정답 추천 초안(2481B)을 지웠고, 이어진
   `[ARG-EMPTY]` 가 **`credit_score` 를 채우라고 강제**해 모델이 값을 **지어냈으며**(`"850"`),
   `[GROUNDING WARNING]` 이 그것을 되돌려 **손님에게 묻는 것 외에 길이 없게** 만들었다. 이 도구는
   base 78/78 에 **0회** 등장한다. `[ORDER]` 태그는 이 sim 전체에서 **1회**(t30) ⇒ (d) 충족.
2. **근거 (축자 + 파일:줄 · 로컬 재확인)** — `a2/banking_knowledge.gate.json:4653` `_note_require_tool_before`
   축자: *"정책 축자로 licensing 되는 것은 **unlock→call 순서**뿐이다 … 나머지 체인(`apply_for_credit_card ←
   check_card_application_fit` 등)의 **선행 도구는 우리가 만든 scaffold GET 도구**이고, **그 체인을 요구하는
   정책 문장은 없다** … ⇒ **[[23]] 소급 대상**"*. 같은 파일 `:3254` 도구 설명: *"**MANDATORY before
   recommending or applying for ANY credit card** … **their credit score** …"*. 모델 채택 축자 m17
   *"The check_card_application_fit tool requires formalizing the constraints."*
   **규칙(닫힌 술어)**: `require_tool_before` 의 선행 노드가 **`scaffold_get_tools` 원소**이면
   `[ORDER]` deny 를 **표면화로 강등**한다(정책 축자로 licensing 된 `unlock→call` 만 deny 로 남긴다).
   부속: 그 도구의 `description` 에서 `MANDATORY` 및 정책 미근거 필수 파라미터 문구를 뺀다.
3. **반증 조건** — ⑴ `apply_for_credit_card ← check_card_application_fit` 를 요구하는 **정책·KB 축자를 실제로
   찾으면** 이 후보는 폐기다(그때 `:4653` 의 자백을 지운다). ⑵ 강등한 팔에서 070·099 계열의 gold MISS 가
   되살아나면(구 주석이 근거로 든 관측) [[70]] 절충으로 내리고 표면화 문면을 손본다.
   ⑶ 강등해도 064 가 여전히 손님에게 점수를 물으면(초안이 이미 묻고 있었다) **회귀 복구 주장은 성립하지
   않는다** — 그때도 [[23]] 준수 근거만으로 수리는 유지한다.
4. **선행 확인** — `banking_knowledge.gate.json:3254·4653`(로컬) · `[T2_SCAFFOLD_GET] injected` 로그 ·
   `grep -rn check_card_application_fit --include=*.py --include=*.json`(repo) · [[23]]·[[05]]·[[66]] ·
   **아직 안 한 것**: `reports/` 에 이 체인의 정책 근거를 찾는 grep(`require_tool_before` 계열 판정) =
   **[미판정]** ⇒ 프로브 exit 의 첫 칸으로 둔다.

---

#### 1d-13. 기존 후보에 미치는 영향 (표)

| 후보 | 이 9건의 효과 | 등급 |
|---|---|---|
| **D1** (종결 후 표면화 중지) | **중립** — 9건 어디에도 §1 의 절차 정체·`readloop-turn` 이 재현되지 않았다 | PLAUSIBLE (적용 폭은 049 계열로 좁다) |
| **D2** (읽기 루프에 이름·출구) | **중립~약화** — 결손이 읽기 부족인 사례가 없다. 054 는 shell 억제 가설이 실측으로 기각됐다(base 74% ↔ ours 75%) | 약화 |
| **D3** (문면·술어 일치) | **계열 강화** — 014 의 *"do not transfer for this"* 오부착이 **가장 강한 사례**다(문면이 검사한 것과 다른 호출을 죽였다). 059·054·064 의 `None` 문면도 같은 계열 | 계열 CONFIRMED / 본체 미판정 |
| **D4** (`[BLOCKED]` 좁히기) | **약화 재확인** — 051 turn 14 의 `[BLOCKED]` 는 gold 를 죽이지 않았고, 064 turn 40 의 3건도 m59/64/77 에 재발행 통과 | PLAUSIBLE |
| **D6** (중복 창 리셋) | ⛔**전제 붕괴 — 선행조건이 하나 더 는다.** ①출시 노출 문구(*"051 은 잃을 점수가 없다"*)가 **거짓이 됐다**(base x644 051 = 1.0) ②소스·`go_stack` 이 인용한 부정통제 수치가 **정본 JSON 에서 재현되지 않는다** ③정본 OFF ↔ 라이브 런처 전부 ON([[81]]·[[54]]) | **배선 전 문서 정정 필수** |
| **D7** (grounding 접두) | **중립** — 9건에서 재발 없음(064 는 grounding 이 **의도대로** 지어낸 값을 되돌린 사례다) | 유지 |
| **D8** (claimprov `None` 금지) | **강화 · n 증가** — 054(3발화 `[None, None]` · `unbacked=0`) · 064(2건) · 079(부분: `['record_update', None, None]`) ⇒ **079 는 D8 로 안 잡힌다**(식별 가능한 항목이 1개 있다) | CONFIRMED · 사정거리 정정 |
| **D9** (폐기 초안 원장) | **강화 + 사정거리 축소** — 079·054·051 에서 (c) 판정이 폐기 원문 부재에 걸렸다. 단 **`T2_GEN_TRACE` 가 `content=B tool_calls=N` 을 채널별로 남기므로 호출 집합 변화는 사후 판정 가능** ⇒ D9 의 요구는 **본문**에 한정 | CONFIRMED (범위 축소) |
| **D10** (선언 실패 fail-open 침묵) | **중립** — 9건에서 별도 사례를 세지 않았다 | 유지 |
| **L1** (꺼진 열거 레버) | **기대수익 0 재확인** — 059 의 오답 `"Green Account (savings)"` 도 064 의 `"Silver Account"` 도 **KB 실재 이름**이라 *'실재하는 이름인가'* 술어를 통과한다 | 전제 CONFIRMED / 기대수익 0 |
| **L2** (`recommend_formalize` 격리) | **n 을 3으로 올린다 — 007 · 059 · 064.** 셋 다 서브가 **확언으로 오답**을 냈고(007 은 `intent_operator_formalize` 와 **서로 모순**), 007 에서는 그 게이트에 **구별 질문 탈출구가 없어** 정당한 초안을 폐기했다 | 조사 · 우선순위 상향 |

---

#### 1d-14. 이 절의 한계 (원인 진술에 쓰지 마라)

- **8/9 건에 「코퍼스 통과 횟수」가 없다.** 055 가 보여준 대로 **한 번의 통과 런이 나오면 등급이 즉시 내려간다.**
  051 만 붙였다(0/42 · 41 레거시).
- **014 의 `reason` 값**, **079 branch A 의 원안 749B**, **051 초안 인자 원문** — 전부 **영구 복구 불가**(D9).
- **007 의 상시 guidance 버스**(도구 10개 주입 · `[T2_GUIDED] guided applied`)는 [[57]] 부정통제 없이 분리
  불가 ⇒ `미판정`. 038 도 같은 이유로 `no` 로 못 민다.
- **분석 단위 경고 재확인** — 머리말 ⑵ 대로 이 절은 **태스크 단위**다. `§1e`(기전별 per-step · flip 25% 바닥
  부정통제)가 나오면 **B 구획 5건의 상당수가 잡음으로 흡수될 수 있다.** 리뷰 비중은 §1e 에 둬라.

**주요 경로** — 리모트 ours: `/home/woori/scratch/tau2-bench/data/simulations/{bank_k8141med1_20260903_2256,
bank_k8141med2_20260903_2256, bank_k8143med1_20260904_0135, bank_re151med1_20260904_0255}/results.json` ·
`/home/woori/scratch/logs/{<tag>.log, fb_<tag>.jsonl, trace_<tag>.jsonl}` ·
재현 스크립트 `/home/woori/scratch/regrun/{xr038_db.py, xr038_cf.py, z4.py, xz051_dump.py, xz051b.py, xz548b.py,
x054_*.py, g1.py–cc.py}` · 덤프 `/home/woori/scratch/regrun/x014/{ours_014.txt, base_014.txt, fb014.txt, tr014.txt}`
· base gz `reports/facet_rft_2026/sim_results/bank_x644_q38base_bank78_20260830.results.json.gz`
로컬(이 절에서 내가 직접 재확인한 것): `scripts\distill\tau2\t2_gate_patch.py:3151,3175,10524,10961,12256-12264,
15207-15224,15335-15364` · `scripts\distill\tau2\go_stack.sh:689-695` ·
`scripts\distill\tau2\{run_ours_task.sh:128, run_night_ab.sh:65, run_t7363_night.sh:57, run_t7364.sh:64,
run_t7365.sh:63}` · `scripts\distill\tau2\a2\banking_knowledge.gate.json:3254,4653` ·
`scripts\distill\tau2\x548_dup_deny_iso.py:86,284` · `reports\facet_rft_2026\x548_dup_deny_iso_2026_08_26.json`
(`rows` 길이 1 · target 074 · tally 확인)

---

##### 1d-055 — 유일하게 미분석이던 회귀 (base PASS ↔ ours FAIL)

**왜 이 칸이 남아 있었나.** 이 캠페인의 회귀(= base pass ∩ ours 전량 실패)는 **정확히 10건**이다 [CONFIRMED] — `regrun/.perstep_simindex.json` 에서 mtime ≥ 2026-09-03 인 런 **30개 · 고유 72 태스크**를 모아 ours 전량 0.0 인 24 태스크를 뽑고, `x738_q38_base97_census_2026_09_04.md:34-36` 의 base pass 42 와 교집합:

```
007 014 015 038 051 054 055 059 064 079
```

이 중 **9건은 `/home/woori/scratch/regrun/` 에 per-step 덤프가 이미 있었고**(`_007_out.txt` `_014_full.txt` `_015_out.txt` `_038_out.txt` `_051_*` `_054_*` `_059_full.txt` `_064_full.txt` `_079_*`) **055 만 없었다.** 이 소절이 그 칸을 채운다.

> ⛔ **채점 단위는 DB 변이다** ([[69]]). `reward_basis: ['DB']` · `reward_breakdown: {"DB": 0.0}` · `db_check: {"db_match": false, "db_reward": 0.0}`. `action_checks` 는 진단용이고, 이 태스크가 그 실증이다 — **base 도 `055_6`·`055_7` 두 칸이 `action_match=False` 인데 reward 1.0** 이다(base m82 가 `give_discoverable_user_tool` 에 `arguments` 를 얹어 보냈다). 아래는 전부 **MISSING / WRONGARG / EXTRA** 로 센다.

**대조 표** — 같은 시드 · 같은 sha · 같은 모델 · 같은 검색 설정 [CONFIRMED]

| | base `bank_x617_iso_q38_bank20_20260830` | ours `bank_k8143long3_20260904_0839` (arm=**viewmax2**) |
|---|---|---|
| reward / db_match | **1.0** / True | **0.0** / False |
| 벽시계 | 131.5분 (7,888.91s) | 73.6분 (4,416.80s) |
| msg / tool call | 100 / 45 | 84 / 37 |
| seed · git · llm · retrieval | s626729 · `fc0055dc4e0a…` · `Qwen/Qwen3.8-27B-FP8` · `alltools` | **동일** |
| 저장·savings 결정 msg | **m58** 단일 지목 | **m41** 3지선다(1번 Gold) → m46 확정 → m63 기록 |
| 분기 직전 우리 층 발화 | **없음** (런 로그 `[T2_` **0줄**) | **있음** (fb `reminder-user/claimprov` turn=41 · log:803-806) |

⚠ **분은 비교하지 마라** — 배치 조건이 다르다(§1c-6). 배치 불변 지표(메시지 usage 합, 서브콜 제외): 생성 **20,970 ↔ 35,358 토큰(1.69배)** · 프롬프트 **2,568,388 ↔ 1,770,790**(ours 가 더 적다) — §1c-6 의 064 패턴과 같은 방향이다.

**결정 턴 축자.** base m58: *"## My recommendation: **Silver Plus Account** (savings)"*. ours m41(= 이 sim 에서 savings 클래스 이름이 처음 나오는 어시스턴트 발화 · 앞 41 메시지에 Gold/Silver 언급 0):

> `1. **Gold Account + Gold Rewards Card** — my top pick. $5,000 minimum (no fee at your $5–6k balance), ~6.25%+ APY …`
> `3. **Silver Plus Account** — no-fee fallback. $2,500 minimum, ~3.275% APY, 15 withdrawals/mo …`

손님 발화 차이로는 설명되지 않는다 [CONFIRMED] — base m49 *"A friend mentioned like Gold and Green accounts?"* ↔ ours m15 *"A friend mentioned something about **Gold** and **Green** accounts, but I'm honestly lost."* 로 **양쪽 다 Gold 를 먼저 꺼낸다**. ours m66 의 수락(*"keep the Gold Account as-is for now"*)은 m63 **이후**라 원인이 될 수 없다 — `DEFECT_LEVER_COVERAGE_2026_08_23.md:356` 축자 *"손님이 고른 것이 아니라 우리가 틀리게 추천한 것을 손님이 받아들였다"* 와 동형.

**DB 실패 단위 [CONFIRMED]** — write 도구는 `ToolType.WRITE` 셋뿐이다(`tools.py:481 log_verification` · `:2372 open_bank_account_4821` · `:4326 deposit_check_3847`).

```
MATCHED   m52 log_verification                    -> "Verification logged successfully."
MATCHED   m61 open_bank_account_4821 checking="Purple Account"
WRONGARG  m63 open_bank_account_4821 savings account_class="Gold Account"   (gold "Silver Plus Account")
          실행됨: m64 "Bank account opened successfully! - Account ID: 1bc7064aea2ca2d3 …
                       - Account Class: Gold Account - Status: OPEN"
WRONGARG  m77 deposit_check_3847 account_id="1bc7064aea2ca2d3"  (gold "7e48bf3b0589cfad")  ← 순수 하류
          실행됨: m78 "Check deposited successfully …"
MISSING   0      EXTRA(write) 0      DUP 0
```

EXTRA 가 0 인 근거: m71 손님 호출은 **에러로 죽었다**(m72 *"Error: Tool 'deposit_check_3847' has not been given to you by the agent."*) ⇒ 변이 0. m48 `verify_identity` 는 우리 A2 도구다 — env `tools.py` 에 정의 **0건**이고 로그가 `[T2_A2_VARIANT] verify_identity ← 'ledger' (params=['provided'] op=match_verdict_grounded)` 로 자인한다 ⇒ DB 무관. 나머지 EXTRA 는 전부 `shell`/`KB_search` read.
⇒ **단일 결손 = `open_bank_account_4821.account_class` 한 칸.** `account_id` 는 클래스의 함수이므로 둘째 칸은 하류다(`tasks__20260824/TASK_055.md:80` 재현).

**귀속 4조건**

| | 조건 | 판정 | 근거 |
|---|---|---|---|
| (a) | 첫 분기 msg 특정 | **✅ m41** | 이 sim 최초의 클래스 지목. 손님 발화 대조로 배제 완료(위) |
| (b) | 분기 **직전** 우리 층 발화 | **✅** | log:758 `agent_response … content=2069B`(초안) → :803 `[T2_CLAIMPROV] window hit(resign) claims=17 **unbacked=0 pending=4 unb_p=4 [None, None, None, None]`* → :806 `[T2_LEVER] T2_GATE_REGEN … claimprov` → :829 `agent_response_claimprov … content=1082B`(= m41). m46 도 동일(:881→:902→:928) |
| (c) | 그 발화가 base 가 한 행동을 금지·전환시킴을 축자로 | **❌** | 전송 문면에 `account_class`·상품명이 **0회**다(내용은 미이행 약속·도구 소유권, 그것도 이름이 전부 `None`). base m58 의 *단일 지목* 을 우리가 막았다고 말하려면 **폐기된 2069B 초안 본문**이 필요한데 어디에도 없다(**D9**) |
| (d) | 같은 게이트 아래 통과한 형제 없음 | **❌** | 같은 CLAIMPROV window 가 regen 없이 **4회 통과**(log:296 `claims=5 … pending=0` · :1417 `claims=8` · :1570 `claims=20` · :1644 `claims=9`). 더 결정적으로 **보상을 잃은 write 자체(m61·m63)에는 우리 층이 한 번도 발화하지 않았다** — `T2_LEVER` 가 log:906 다음 **:1469 까지 공백**이다 |

## ⇒ 최종 등급: **미판정** (`our_layer = unknown`)

(c)(d) 미충족. 추정으로 'yes' 를 쓰지 않는다. 확정적으로 남는 두 문장:

1. **우리 층은 결정 자리에 아무것도 하지 않았다** [CONFIRMED] — `account_class` 를 보는 게이트·서브가 라이브에 없다(`T2_WRITE_ARG_ENUM` = 이 로그 **0회** · `go_stack.sh`·`arms/*.env` grep **0건**). `recommend_formalize` 는 3회 발화했으나 전부 `{"applies": false, "card_type": "none"}` 이다.
2. **우리 층은 결정 턴 두 개를 오발화로 덮어썼다** [CONFIRMED · 결함 / 인과 미판정] — `unbacked=0` 인데 `pending=4`, 항목 이름이 전부 `None`, resign 중이 아닌데 resign 문면. 그 결과 m41·m46 이 면책문으로 시작하고, 손님이 m42 에서 *"I really don't want to compare three. Can you please just tell me **which ONE**"* 이라 되물었는데 m46 은 다시 검증 얘기로 갔다. 부호 판정은 **claimprov ON/OFF × 같은 팔(viewmax2) × 같은 seed** A/B 로만 선다.

**부수 (reward 무관 · 턴 비용만)** — `[SIGNATURE] give_discoverable_user_tool takes only discoverable_tool_name in this domain; you also passed arguments` 가 turn 70·72(deny) + 72·74(route)에서 걸려 m70–m76 **6 메시지**를 태웠다. 그러나 **같은 거부가 통과 런 x725 에도 2회 났고 reward 1.0** 이다 ⇒ DB 무관 [CONFIRMED].

---

**정본과의 관계 — 재현이다. 새 후보를 만들지 않는다.**

- **②범주(`account_class`) 의 재현** — `x506_hard0_rootcause_2026_08_24.json` `answer_3_six_axes[1]` 축자: `axis "② 범주 소속" · fields ["account_class","card_type"] · tasks ["055","057","063"]`. 오답쌍까지 코퍼스 표의 **첫 줄 그대로**다 — `DEFECT_LEVER_COVERAGE_2026_08_23.md:344` `Gold Account ↔ Silver Plus Account 27`.
- **다만 32B 의 `canonical_cause`(*"의미 소속 판정 불가(경계)"*)는 재현되지 않았다** — 모델은 m20 에서 `doc_savings_accounts_silver_plus_account_*` 를 cat 했고, m41 에 Silver Plus 스펙을 **표로 적었고**, m65 에서 *"the **Silver Plus Account** … is the cleaner fit for your actual balance"* 라고 스스로 말한다. 이름·표·비교가 다 있다. 이는 정본이 이미 내린 판정과 같은 방향이다 — `x509_axis_queue_2026_08_24.json:82` `boundary_RETRACTED` · `DEFECT_LEVER_COVERAGE:361` 축자 *"이 축은 `의미 소속 판정 불가`(경계)가 **아니다**. 스펙 표와 손님의 수치 요구를 맞추는 **검산 미실행**이다."*
- **기전 이름도 정본이 이미 붙여 뒀다: 「미합류 조건부 칸」.** `x506` ② `status` 축자 *"잔여를 **우리 층 둘**(미합류 조건부 칸·`absent` 토큰 충돌)로 좁혔다"* · `x509…json:256` 축자 *"라이브 엔진은 `exists|absent` op enum 을 모델에게 제시하지 않고 **조건부 표를 안 읽는다**"*. 이번 실패가 정확히 그 자리다 — 모델은 m41 에서 Gold 최소잔액을 **`$5,000`(조건부 행)** 으로 쓰고, m65 에서 *"(The $5,000 minimum I mentioned earlier only applies if you also hold a Gold Rewards Card, **which you don't currently**.)"* 라고 자기 정정한다. 궤적의 원문은 m34 축자 *"Opening deposit minimum: $5,000 · Ongoing minimum balance: $10,000"* 다.
- **⑥ 식별자 전사는 재현되지 않았다** — `x506` ⑥ 의 `tasks` 에 055 가 있지만, m64 가 발급한 `1bc7064aea2ca2d3` 는 m65·m66·m77 에서 **한 글자도 어긋나지 않는다**. 틀린 것은 *틀린 계좌의 옳은 id* 다 ⇒ `TASK_055.md:80`(*"`account_id` 는 클래스의 함수"*)이 Q38 프레임에서 재확인됐다. 축표의 *"②+⑥ 동시 필요"* 는 여기서도 **필드 인구조사이지 인과가 아니다**.

**⛔ 철회 목록 (055)**

| 철회한 주장 | 왜 |
|---|---|
| *"base PASS ↔ ours FAIL ⇒ **우리 스캐폴드가 Q38 base-pass 를 깼다**"* | 회귀 **사실**은 맞지만 **인과 진술로는 철회**. 같은 seed·같은 gold·같은 sha 에서 ours 가 **통과한 런이 있다** — `bank_x725_t3prime_A_20260901` task_055 **reward 1.0 · db_match True** |
| *"Q38 + ours 에서 055 를 잰 적이 없다"* | **3회** 쟀다: x725(t3prime) 1.0 · night2p1(실제 arm=**viewmax2**) 0.0 · k8143long3(viewmax2) 0.0 ⇒ **1/3** |
| *"`x725` 는 base 다"* (조사 배경 브리프) | **ours 다.** `fb_`/`trace_` 사이드카 존재 · trace 의 `task_055#s626729` 행 339 · 런 로그 `[T2_` **1,528줄** · `[T2_LEVER] T2_GATE_REGEN … usertoolnote/givequote/givexec` |
| *"`messages` 안 `[T2_` 마커 0 ⇒ base"* (같은 브리프의 판별식) | **무효.** 확정 ours 인 k8143long3·x725·night2p1 모두 `messages` 기준 0이다 — 마커는 영속 궤적이 아니라 **stderr 로그/trace** 로 간다 |
| *"⇒ 그러므로 `x738` census 를 재검증해야 한다"* | **철회.** `x738:63-71` 의 판별식은 처음부터 **런 로그** 기준이고(*"판별자: 런 로그의 `[T2_` 마커 유무"*), 실측이 그것을 지지한다(x617 log **0** · x725 log **1,528**). x738 은 x725 를 ours 로 옳게 분류한다 |
| *"claimprov 문법×스키마 충돌은 **새** CONFIRMED 배선 결함"* | **철회.** 2026-09-01 커밋 `117f02e5` 가 고쳤고 회귀 검정 `scripts/distill/tau2/test_guided_schema_conflict.py` 가 축자로 그 사유를 담고 있다(*"스키마가 걸린 콜에는 문법을 붙이지 않는다"*). [[74]] 위반이었다 |
| *"night2p1 은 `silver_plus` 가 로그에 3회뿐"* | 수치 교체: 궤적 기준 `Silver Plus` **32회**다. 결론은 유지 — 전부 KB 인덱스·검색 결과 리스트이고 **어시스턴트 발화 0회 · `silver_plus` 문서 cat 0회** |
| *"네 롤아웃 전부 같은 Gold/Green 언급"* | **거짓.** night2p1 의 손님은 `Gold` 를 **한 번도** 말하지 않는다. 손님 발화 대조가 성립하는 짝은 **base m49 ↔ k8143long3 m15** 뿐이다 |

---

**수리 후보에 미치는 영향**

| 후보 | 이 건의 효과 | 근거 |
|---|---|---|
| **D1** (종결 후 표면화 중지) | **중립** — 이 sim 의 `T2_LEVER` 전량이 claimprov 2 · usertoolnote 1 · givequote 1 · givexec 1 · EPLAN_WALK 1 이고 표면화 루프 마커가 없다 | log `sim=task_055` LEVER 전수 |
| **D2** (읽기 루프에 이름·출구) | **중립~약화** — 결손은 읽기 부족이 아니다. Silver Plus 문서를 m20 에서 cat 하고 m41 에 스펙을 표로 적고도 틀렸다 | m20 · m41 |
| **D3** (문면·술어 일치) | **본체 미발화 / 계열 강화** — *"deny 문면이 이름을 못 댄다"* 의 새 독립 사례 2건(`None: None; None: None; None: None`) | fb turn 41·46 |
| **D4** (`[BLOCKED]` 좁히기) | **약화 재확인** — 이 sim 의 동반 차단은 turn 50 `resolve the flagged call(s) first` **1건**이고 gold 를 죽이지 않았다 | fb tool-deny 전수 |
| **D6** (중복 창 리셋) | **중립** — `[DUPLICATE-WRITE]` 이 sim 발화 **0** | grep 0 |
| **D7** (grounding 접두) | **중립** — `[GROUNDING…]` 이 sim 발화 **0** | grep 0 |
| **D8** (claimprov `None` 금지) | **강화 · n 증가** — 055 에서 **2 sim · 4 발화 전부** `None`(k8143long3 turn 41·46 · night2p1 turn 25·48). 게다가 여기서는 그 발화가 **결정 턴 두 개를 덮어썼고**, 손님의 *"which ONE"* 질문에 답하지 못했다 | fb 4행 · log:803·902 |
| **D9** (폐기 초안 원장) | **결정적 강화 — 이 소절의 (c) 가 D9 때문에 원리상 미충족이다.** 2069B·1720B 두 초안 본문이 어디에도 없다(faildump 없음 · trace 없음 · 바이트 수만 남음) | log:758·881 |
| **L1** (꺼진 열거 레버) | **기대수익 0 재확인** — 오답 `Gold Account` 는 KB 실재 상품명이다(m12·m13 에서 cat). `T2_WRITE_ARG_ENUM` 술어(*'실재하는 이름인가'*)는 이 칸을 통과시킨다 ⇒ `x509…json:183` 축자 *"게이트가 겨누는 것은 1.4% 뿐이다"* 의 실증 사례 하나 추가. 전제(레버 OFF)는 여전히 CONFIRMED | grep 0건 |

**새 후보 D10 — 선언 서브가 실패하면 게이트가 조용히 꺼진다 (계기·비교 규격 · 레버 아님)**

> ⛔⛔**2026-09-04 2차 리뷰 — D10↔D8 자연 실험은 성립하지 않는다. 철회한다.**
> 아래 1.의 *"055 의 ours 1/3 은 레버 대조가 아니다"* 는 **비교 규격 손상의 지적으로만** 살고,
> **reward 근거로는 죽는다.** 채점된 sim 만으로 다시 재면(로컬 재현 · `/tmp/d10.py`):
> ```
> 캠페인 24런 · reward_info 가 있는 sim 49개
>   declaration failed 가 난 태스크의 sim   n= 7   평균 reward 0.5714
>   그 밖의 sim                             n=42   평균 reward 0.5714
>                                         차이 = **정확히 0**
> ```
> 이전에 본 0.48 격차는 **채점되지 않은 sim(phantom)** 을 분모에 넣어 생긴 인공물이었다.
> ⇒ 반증조건 ⑴(*"reward 분포가 다르지 않으면 비교 규격 손상은 무해하다"*)이 **충족됐다.**
> D10 은 **계기·토큰 누수 항목**으로만 남는다([[57]] 부정통제로 자기 주장을 죽인 사례).
> ⚠단 **D8 과는 다른 이야기다** — D8 의 원인은 `declaration failed` 가 아니라
> **파싱에 성공한 판에서도 이름이 비는 스키마 불일치**이고, 그건 158/158·73/73 로 살아 있다.

1. **주장 + 양화** — 현 캠페인(2026-09-03 이후) 런 로그 **32개 중 9개 · 18회**에서 `[T2_CLAIMPROV] declaration failed (no-op)` 가 났고, 그 sim 에서 claimprov 는 **꺼진 것과 같다**(뒤이어 `window hit claims=None pending=None`). 055 의 **통과 런 x725 는 task_055 안에서만 6회** 나서 그 sim 전 구간 무효였다 ⇒ *"ours 1/3"* 은 **레버 대조가 아니다**([[54]]).
2. **근거 (축자 + 위치)** — `logs/bank_k8143med1_20260904_0135.log:896-897` `call=agent_claimprov max_tokens=8192 tb=None [TERSE] -> gen=8192 … **TRUNC** … content=29666B` → `declaration failed (no-op): JSONDecodeError("Expecting ',' delimiter: line 1 column 29646 (char 29645)")` / 침묵 지점 `t2_gate_patch.py:14875` `print("[T2_CLAIMPROV] declaration failed (no-op): %r" % (_ce2,), …)`.
3. **반증 조건** — ⑴ 선언 실패가 난 sim 의 reward 분포가 성공 sim 과 다르지 않으면 비교 규격 손상은 무해하다(그때는 계기만 남긴다). ⑵ `JSONDecodeError` 가 8192 절단이 아닌 다른 원인이면 처방이 바뀐다 — `**TRUNC**` 동반 여부로 가른다(현 실측은 **동반 1:1**). ⑶ 실패 시 재발사 경로가 이미 있는데 로그만 안 남는 것이면 결손 없음.
4. **선행 확인** — ⛔**[[74]] 보완 (2차 리뷰)**: `model_profiles/Qwen__Qwen3.8-27B-FP8.env:60-67`
   이 **같은 절단을 2026-09-02 에 이미 재고 처방까지 적어 뒀다** — 축자 *"2048 도 부족했다 …
   `call=agent_claimprov max_tokens=2048 … **TRUNC** … content=6970B` → `declaration failed:
   JSONDecodeError` … 주장 목록은 궤적 길이에 비례한다 … ⇒ **8192**"*. 즉 상한 인상은 **이미
   한 번 처방됐고 실행됐다**(`export T2_PROBE_MAX_TOKENS=8192`). 캠페인의 18회는 **그 8192 에서
   또 난 것**이므로, 남은 처방은 *"상한을 더 올린다"* 가 **아니라** 선언 blob 크기를 줄이는
   쪽이다([[82]] — 상한은 모델별로 다시 재라 · 그리고 폭주의 실효 처방은 상한이다).
   그 밖: `test_terse_schema.py` §T-12 가 **같은 기전을 512 상한에서 이미 확정했다**(축자 *"TRUNC 1 ↔ `declaration failed (no-op)` 1 로 **1:1 대응**했다"*) · 문법×스키마 원인은 `test_guided_schema_conflict.py` + 커밋 `117f02e5`(2026-09-01)로 **수리 완료** · [[84]] · [[54]] · [[81]].
   ⇒ **새 기전이 아니다.** 델타는 두 개뿐이다: ⓐ 같은 절단이 **8192 상한에서 현 sha 로 계속 난다**(캠페인 18회), ⓑ 실패가 **fail-open 침묵**이라 레버 켠 런과 꺼진 런이 같은 태그로 섞인다. 레버가 아니라 **D9 와 같은 계기 항목**으로 둔다.

**주요 경로** — 리모트: `/home/woori/scratch/tau2-bench/data/simulations/bank_k8143long3_20260904_0839/results.json` · `/home/woori/scratch/logs/bank_k8143long3_20260904_0839.log`(:296·758·803-806·829·881·902·928·1417) · `fb_bank_k8143long3_20260904_0839.jsonl` · `/home/woori/scratch/logs/bank_x725_t3prime_A_20260901.log:207` · `bank_k8143med1_20260904_0135.log:896` · `/home/woori/scratch/regrun/v055*.py` · `.perstep_simindex.json` · `sim_results/bank_x617_iso_q38_bank20_20260830.results.json.gz`
로컬: `C:\workspace\ba-frft\reports\facet_rft_2026\x506_hard0_rootcause_2026_08_24.json` · `…\x509_axis_queue_2026_08_24.json:82,183,256` · `…\DEFECT_LEVER_COVERAGE_2026_08_23.md:344,356,361` · `…\tasks__20260824\TASK_055.md:80` · `…\x738_q38_base97_census_2026_09_04.md:34-36,63-71` · `C:\workspace\ba-frft\scripts\distill\tau2\go_stack.sh:134,279` · `…\test_terse_schema.py` · `…\test_guided_schema_conflict.py`

---

### 1e. 기전 인구조사 — 태스크가 아니라 병증을 센다 (2026-09-04)

#### 1e-0. 왜 단위를 바꿨나 — 그리고 flip 바닥이 회귀를 삼키는가

**사용자 지시 (2026-09-04 · 축자)**

> *"회귀가 아니라 시나리오가 달라져서 pass 못한 거면, 기존 레버의 원인과 같은 건지 아닌지를 판단하면 된다. 태스크별로 회귀를 따지는 게 아니라, 태스크의 **기전별로 per step 별로** 따져야 한다. 여러 병증이 태스크별로 **돌아가면서 돌아다닌다**."*
> *"원래 **25% 정도는 pass/fail 을 반복**했다."*

**실증 2건 (이 지시의 근거)**

| 사실 | 값 |
|---|---|
| `task_055` 는 같은 seed·gold 에서 ours 통과 런이 있다 | `x725_t3prime_A_20260901` **1.0** / `night2p1_t3prime_20260901` 0.0 / `k8143long3_20260904` 0.0 ⇒ **1/3** |
| `task_059` gold 문자열은 코퍼스 14 sim 중 1회 | (§1c-1) |

⇒ 회귀 명단에 오른 태스크가 **같은 팔에서 통과한 기록**을 갖는다. 태스크 단위 서술은 이 코퍼스에서 성립하지 않는다.

**★flip 바닥 (부정통제 · [[57]]) — 먼저 잰다**

모집단: `sim_results/` 573→746 results.gz 중 `domain=banking_knowledge ∧ llm=Qwen3.8-27B-FP8` = 70 런. ours/base 판별자는 런 로그의 `[T2_` 마커 유무(`x738_q38_base97_census_2026_09_04.md:66` 축자: *"판별자: 런 로그의 `[T2_` 마커 유무."*). ours 62 런 · 채점 156 sim.

| 층 | 값 | n | 95% CI (Wilson) |
|---|---|---|---|
| **base 팔 flip 바닥 (쌍 단위 · scaffold 0개)** | **18.8%** | 16 쌍 | [6.6, 43.0] |
| base 팔 flip (태스크 단위) | 21.4% | 14 | [7.6, 47.6] |
| base 런 내부 trial 반복 | **0.0%** | 2 | — |
| 같은 `engine_sha` · 다른 런 | 0.0% | 1 | — |
| **sha 가로지름** | **36.0%** | 111 쌍 | — |
| ours 태스크 단위 flip | 43.6% | 39 | [29.3, 59.0] |
| ours 시행 단위 소수파 | 18.1% | 105 | [11.9, 26.5] |

base flip 실증 3건: `task_008` (x599=1.0 → x644=0.0) · `task_012` (1.0→0.0) · `task_017` (x599=0.0 → x617=1.0). **scaffold 를 하나도 켜지 않아도 같은 base 를 두 번 돌리면 3/16 이 뒤집힌다.** 사용자 축자 *"25% 정도"* 와 정합한다.

⛔ **sha 층화는 이 코퍼스에서 원리적으로 불가능하다.** `engine_sha` 를 남긴 24 런이 **24/24 `dirty=True`** 다 — 축자(`bank_010ctl_20260904_0007.log.gz`):

```
[t2_run] provenance -> data/simulations/bank_010ctl_20260904_0007/provenance.json (engine_sha=a208c8e0 dirty=True ctx=131072)
```

⇒ sha 동일 ≠ 코드 동일. 그래서 바닥을 **다른 축**(scaffold 0개 팔)에서 잡았다.

**⇒ 판정 — 회귀 10/42 는 flip 바닥 안이다 (CONFIRMED)**

| 대조 | 값 | 이항/z |
|---|---|---|
| 회귀 관측 | **10/42 = 23.8%** [13.5, 38.5] | — |
| vs base 팔 flip 바닥 18.8% | | P(X≥10) = **0.255** |
| vs ours 시행 단위 소수파 18.1% | | P(X≥10) = **0.218** |
| vs base 팔 태스크 flip 21.4% | | P(X≥10) = **0.410** |
| vs 사용자 인용 25% | | P(X≥10) = **0.629** |

그리고 결정적인 것은 통계 이전의 **설계**다:

```
campaign covers base-PASS tasks n=42 -> ours fail(all sims) 10  => REGRESSION 10/42 = 23.8%
   regressed: ['task_007','task_014','task_015','task_038','task_051','task_054','task_055','task_059','task_064','task_079']
  -> of the 10, 10 were judged on a SINGLE sim
  base-PASS coverage in campaign: n_sims histogram Counter({1: 42})
```

**base-pass 42 태스크 전체가 `n_sims=1` 이다.** 42개의 단일 베르누이 시행 위에서 "회귀"라는 서술은 만들어질 수 없다. 반복이 있는 유일한 건(`055`)은 1/3 로 통과한다 ⇒ 그 한 건은 회귀가 아니라 flip 이다.

**⇒ 반대편 — 회복은 바닥 밖이지만 바닥의 선택에 걸린다 (미판정)**

| | 값 | vs base 바닥 18.8% | vs ours base-FAIL flip 43.2% |
|---|---|---|---|
| 회복 (base-FAIL → ours pass) | **13/29 = 44.8%** [28.4, 62.5] | P(X≥13) = **0.0012** | 겹친다 |

**부호검정 (10 잃고 13 얻음, 불일치 23):** p = **0.678**. ⇒ **팔 전체의 순효과는 0과 구분되지 않는다.** 어느 바닥이 옳은지(base 팔 18.8% 냐 ours 팔 43.2% 냐)는 이 코퍼스가 결정하지 못한다 — ours 팔 flip 에는 레버 변화가 섞여 있고 base 팔은 쌍이 16 개다. **[미판정]**

⇒ ⚠**`§1d`(태스크 단위 귀속)의 격을 내린다.** §1d 의 10건 귀속은 전부 n=1 위에 서 있고, 그 개수 자체가 잡음과 구분되지 않는다. §1d 는 *"이 sim 에서 무슨 일이 있었나"* 의 기록으로만 읽어라 — *"우리가 깼다"* 의 근거로 쓰지 마라.

---

#### 1e-1. 인구조사표 — 기전 × sim

**모집단** (정본 `/home/woori/scratch/mechrun/census/mech_census.py` · 경로 결의는 `t2_forensic.path_for / log_text / sidecar_rows / simtag` 위임 · 사본 0 [[67]])

```
arm=viewmax2 · banking_knowledge · driver 로그 mtime >= 2026-09-03
런(arm 일치) 29 · sim>0 런 18 · **채점 sim 75** · 태스크 73 · pass 46 (61.3%) / fail 29
사이드카 present 런 26 / absent 3 — 채점 sim 75/75 전량이 present 런 소속
탈락: {'no_results_file': 2, 'unscored': 9} · 로그없음 [] · simtag 충돌 0
태스크당 sim 수: {3: 1, 1: 72}  → 반복이 있는 태스크는 ['task_010'] 뿐
종료사유: {'user_stop': 75}   (max_steps 0)
기전 총 188 종 · {'M': 103, 'LEVER': 48, 'DENY': 18, 'DET': 11, 'FB': 5, 'REGEN_CALL': 1, 'DECLFAIL': 1, 'TRUNC': 1}
```

⛔ 영속 `.results.json.gz` 가 빈 껍데기인 런 9개가 있다(`bank_c55_s1_viewmax2` 영속 0 sim ↔ 라이브 98,018 B). `t2_forensic` 위임이 없으면 모집단이 무너진다.

**① 상존 16종 — 변별 정보 0**

`DET:claimprov_window_resign · FB:reminder-assistant · FB:subcall · LEVER:T2_GUIDED · M:T2_A2_VARIANT · M:T2_ACTIONREQ · M:T2_CLAIMPROV · M:T2_GEN_TRACE · M:T2_GUIDED · M:T2_LEVER · M:T2_SCAFFOLD_GET · M:T2_SELFDECL · M:T2_SUBWIN · M:T2_TOOL_OBS · M:T2_WINDOW · REGEN_CALL` — 75/75 sim 발화 ⇒ pass% 가 정의상 기저 61.3%. **이 16종을 겨눈 판정은 원리적으로 불가능하다.**

**② 변동 기전 (2 ≤ n_sim ≤ 45) 상위 — 돌아다님 내림차순**

```
기전                                sim/분모  task     ev  pass  fail    p%   Δ기저  roam
M:T2_RESOLVE                         40/75     40    264    21    19   52%    -9%  1.00
M:T2_FB_VIEW                         40/75     38    153    20    20   50%   -11%  0.95
M:T2_MATERIAL_GATE                   38/75     37     91    25    13   66%    +4%  0.97
M:T2_UNAVAIL                         30/75     30     68    14    16   47%   -15%  1.00
DENY:BLOCKED                         28/75     28    133    16    12   57%    -4%  1.00
M:T2_TOOL_SIGNATURE                  23/75     23     55    14     9   61%    -0%  1.00
M:T2_PROCEDURE                       22/75     22    637    18     4   82%   +20%  1.00
LEVER:T2_USER_TOOL_NOTE              19/75     19     19    10     9   53%    -9%  1.00
M:T2_SG_ISOLATE                      19/75     19    162    16     3   84%   +23%  1.00
DENY:POLICY GATE GB2_NOTICE_BEFORE_T 19/75     18     22    12     7   63%    +2%  0.95
M:T2_SOURCE / M:T2_ARBITRATE         18/75     18  86/111    11     7   61%    -0%  1.00
LEVER:T2_GATE_REGEN/usertoolnote     18/75     18     18    10     8   56%    -6%  1.00
LEVER:T2_GIVE_QUOTE / M:T2_GIVE_EXEC 17/75     17  17/17     9     8   53%    -8%  1.00
DENY:SIGNATURE                       17/75     17     41     9     8   53%    -8%  1.00
M:T2_TRANSFER_LEAVES_STEPS           17/75     16     24    10     7   59%    -3%  0.94
```

**③ 희소 기전 (n_sim ≤ 12) — 사용자가 말한 "돌아다니는 병증" 구간**

```
M:T2_SEARCH_EXHAUST                  12/75     12     22     7     5   58%    -3%
M:T2_PIN_READ_STEPS                  12/75     12     18     9     3   75%   +14%
M:T2_PROC_ABSENT                     12/75     12     36     9     3   75%   +14%
M:T2_SG_GROUND / DET:grounding_warn  12/75     12   24/18     9     3   75%   +14%
M:T2_DEFERRED / LEVER:T2_DEFERRED    11/75     11     12     6     5   55%    -7%
M:T2_UNCALLED_UNLOCK                 11/75     11     11     3     8   27%   -34%   ★
M:T2_SUB_RECORDS                     11/75     11     93     4     7   36%   -25%   ★
DET:gen_at_cap_8192                  11/75     11     19     8     3   73%   +11%
DECLFAIL:T2_CLAIMPROV                11/75     11     18     8     3   73%   +11%
M:T2_FAILDUMP                        10/75     10     19     8     2   80%   +19%
M:T2_DISPATCH_LEDGER                 10/75     10     13     9     1   90%   +29%
M:T2_FOLLOWUP                         9/75      9     30     7     2   78%   +16%
M:T2_ARG_EMPTY / DENY:ARG-EMPTY       9/75      9     13     7     2   78%   +16%
M:T2_TRANSCRIBE                       9/75      9     12     8     1   89%   +28%
M:T2_SG_BYREF                         9/75      9     23     8     1   89%   +28%
M:T2_CALLABLE_FRONTIER / CALL_FORM    8/75      8     40     3     5   38%   -24%   ★
```

**④ tool-deny 문면 머리 전수 (18종)**

```
DENY:(no-marker) resolve the flagged  61/75  59   79  35/26  57%
DENY:BLOCKED                          28/75  28  133  16/12  57%
DENY:POLICY GATE GB2_NOTICE_BEFORE_T  19/75  18   22  12/ 7  63%
DENY:SIGNATURE                        17/75  17   41   9/ 8  53%
DENY:OPERATOR-SCOPE                   14/75  14   16   7/ 7  50%
DENY:ARG-EMPTY                         9/75   9   13   7/ 2  78%
DENY:POLICY GATE GB1_VERIFY_BEFORE_A   8/75   8    8   6/ 2  75%
DENY:ACTION 6 · WRITE-EVIDENCE 5 · PROVENANCE 4 · PROCEDURE 4 · REFERENCE 2 · PRESCRIPTION 2
DENY:DUPLICATE-WRITE 1 (0/1) · OPERATOR-PROVENANCE 1 · E-PLAN 1
```

⚠**최대 deny(`resolve the flagged` 61/75)에는 머리 표지가 없다.** 어느 게이트가 냈는지 이 계기로는 안 갈린다 — 이것이 1e-3 의 `[BLOCKED]` 철회 사유와 같은 뿌리다.

**돌아다님 — 수치로, 그리고 왜 이 코퍼스에서 못 쓰는가**

| 기전 | n_sim | n_task | roam = task/sim | 최다 태스크 점유 | 발생 런 | 최다 런 점유 |
|---|---|---|---|---|---|---|
| `M:T2_GUIDED` (상존 계기) | 75 | 73 | 0.97 | 4.0% | 18 | — |
| `DENY:BLOCKED` (판정 대상) | 28 | 28 | 1.00 | 3.6% | 13 | 25.0% |
| `DET:grounding_warning` | 12 | 12 | 1.00 | 8.3% | 8 | 33.3% |
| `M:T2_UNCALLED_UNLOCK` | 11 | 11 | 1.00 | 9.1% | 7 | 45.5% |
| `DECLFAIL:T2_CLAIMPROV` | 11 | 11 | 1.00 | 9.1% | 9 | 18.2% |

⛔ **roam·집중도(HHI)는 이 코퍼스에서 잴 수 없는 양이다.** 태스크당 sim 이 73 중 72 가 1 이므로 `n_task ≈ n_sim` 이 되어 roam 이 구조적으로 ~1.00 에 붙는다. **항상 켜진 상존 계기와 판정 대상이 같은 값을 낸다** ⇒ *"돌아다니므로 태스크 성질이 아니다"* 라는 논증은 **무내용**이다. 그 자리를 대신하는 것은 **⑧ 발생 런 수**다.

**⑧ 런-교락**: 기전당 발생 런 수 분포 `{1:29, 2:23, 3:12, 4:22, 5:12, 6:8, 7:5, 8:8, 9:7, 10:8, 11:6, 12:2, 13:3, 14:1, 15:2, 16:9, 17:2, 18:29}`. 변동 기전 상위 34종의 최다 런 점유는 8~45% ⇒ 런과 분리된다. 반대로 **29/188 종은 단일 런에서만** 찍혔고 그 돌아다님은 런과 분리되지 않는다(`DENY:DUPLICATE-WRITE · DENY:E-PLAN · DENY:REFERENCE · TRUNC · M:T2_DUP_WRITE · M:T2_SG_SCHEMA · LEVER:T2_CALLABLE_HINT · LEVER:T2_DUP_WRITE …`).

**⑥ 태스크당 기전 수 분포**

```
태스크 73 · min 24 · p25 44 · median 53 · p75 58 · max 75 · mean 51.8 · sd 10.4
  20-24 :  1  #            50-54 : 14  ##############
  35-39 :  9  #########    55-59 : 17  #################
  40-44 : 10  ##########   60-64 :  7  #######
  45-49 :  7  #######      65-69 :  4  ####   70-74 : 3 ###   75-79 : 1 #
최소 5: t007=24 · t003=35 · t002=35 · t070=35 · t034=36
최대 5: t102=75 · t023=74 · t016=72 · t010=70 · t022=68
sim 당 기전 수: pass n=46 median 53 mean 50.7 ↔ fail n=29 median 52 mean 52.7
```

⇒ **pass sim 과 fail sim 의 기전 수 분포가 사실상 같다.** *"병증이 많아서 실패했다"* 는 이 코퍼스에서 성립하지 않는다.

**⑤ 지목 기전 점호** (사용자가 준 후보 목록)

```
DET:claimprov_kind_None      74/75 · task 72 · ev 3262 · 45/29 (61%)
DET:claimprov_window_resign  75/75 · task 73 · ev  368 · 46/29 (61%)
DET:claimprov_pending_None   56/75 · task 56 · ev  125 · 34/22 (61%)
DET:claimprov_regen_empty    48/75 · task 48 · ev   65 · 31/17 (65%)
DET:claimprov_regen_rejected  7/75 · task  7 · ev    7 ·  5/2  (71%)
DET:grounding_warning        12/75 · task 12 · ev   18 ·  9/3  (75%)
DET:gen_at_cap_8192          11/75 · task 11 · ev   19 ·  8/3  (73%)
DECLFAIL:T2_CLAIMPROV        11/75 · task 11 · ev   18 ·  8/3  (73%)
M:T2_PHASE_PRECEDE           61/75 · task 59 · ev  406 · 35/26 (57%)
M:T2_TOOL_SIGNATURE          23/75 · task 23 · ev   55 · 14/9  (61%)
M:T2_WRITE_EVIDENCE           5/75 · task  5 · ev    8 ·  2/3  (40%)
DET:readloop_turn             2/75 · DET:sibling_paren 2/75 · DET:truncated_tool_call 2/75
M:T2_DUP_WRITE                1/75 · TRUNC 1/75 · DET:truncguard_fired 1/75
M:T2_WRITE_ARG_ENUM          **발화 0** · M:T2_READLOOP **발화 0**
```

⚠ **표지 단위로만 세면 후보 절반이 "발화 0" 으로 오독된다.** `readloop-turn`·`GROUNDING WARNING`·`kind=None`·상한도달 `gen=8192` 는 대괄호 표지가 아니라 **표지 문면 안쪽**이다 — `M:T2_READLOOP` 은 0 인데 `readloop-turn counted as resignation` 은 실재한다. DET 계열 11종은 그래서 만들었다. 축자(검산 완료):

```
bank_049ctl2_20260904_0534.log.gz
  [sim=task_049#s373753] [T2_FOLLOWUP] readloop-turn counted as resignation
bank_010ctl_20260904_0007.log.gz
  [sim=task_010#s626729] [T2_CLAIMPROV] kind-index rescued: kind=None tool='verify_identity' 원장에 있다
  [sim=task_010#s626729] [T2_CLAIMPROV] tool-miss fallback: kind=None tool='transfer_to_human_agents' 원장 밖 — kind 색인으로 강등
  [sim=task_010#s626729] [T2_SELFDECL] declared=(none — no-op)
bank_lost5_viewmax2_20260903_1610.log.gz
  [sim=task_093#s626729] [T2_TOOL_OBS] id=chatcmpl-tool- err=False -> [GROUNDING WARNING] 3 input value(s) could not be
  verified against the account records / knowledge base and were dropped: base=4.0 (source not found in the knowledge base); …
```

`M:T2_WRITE_ARG_ENUM` 발화 0 은 **검색 경로 3개 전부**에서 확인했다 — 로그 정규식 `(?i)write_arg_enum` 0줄/18런 · `[T2_WRITE_ARG_ENUM]` 0 · 사이드카 DENY 머리 18종에 없음. `LEVER_ROSTER_CANONICAL` 의 *"attested 54 에 없다"* 와 정합 ⇒ **L1 전제 CONFIRMED**.

---

#### 1e-2. ★전역 부정통제 — 188 기전 전수 검정 + 라벨 순열 (신규 · 이 절의 최대 산출)

기전을 **하나씩** 골라 2×2 를 내면 다중비교가 숨는다. 188 종 전부를 같은 방식으로 검정하고, 그 결과를 **reward 라벨 순열**과 비교했다(기전 동시발생 구조는 그대로 두고 라벨만 섞는다 ⇒ 상관 구조 보존).

```
검정 가능한 기전 수(2 <= n_sim <= 74): 148 / 전체 188
관측 p<0.05: 6    p<0.01: 0    최소 p: 0.0183
우연 기대치(148 × 0.05) = 7.4
라벨 순열 300회: p<0.05 개수 median 2 · p90 8 · max 23
                최소 p median 0.0233 · p10 0.0038 · min 0.00011
순열에서 관측(6) 이상 나온 비율 = **0.200**
순열 최소p 가 관측 최소p(0.0183) 이하인 비율 = **0.370**
```

**⇒ CONFIRMED: 이 코퍼스에서 reward 를 가르는 기전은 없다.** *"어떤 기전이든 하나는 가른다"* 라는 가족단위 가설의 p 는 **0.37** 이다. 관측된 유의 칸 6개는 라벨을 무작위로 섞어도 20% 의 확률로 나온다. 그리고 p<0.01 은 **한 칸도 없다**.

(순열 중앙값이 7.4 가 아니라 2인 이유: 기전들이 강하게 공기(共起)해 유효 독립 검정 수가 148 보다 훨씬 작다. 즉 **실제 다중비교 보정은 Bonferroni 148 보다 관대하고, 그런데도 관측이 순열 안에 있다.**)

**그럼에도 순위는 산출이다** — 아래 6칸이 이 코퍼스가 고른 상위 후보다(전부 **[미판정]**, 인과 아님):

| 기전 | present | pass | pass% | Δ기저 | Fisher | 태스크 | 런 | 최다런 |
|---|---|---|---|---|---|---|---|---|
| `M:T2_UNCALLED_UNLOCK` | 11 | 3 | 27.3% [10,57] | **−39.9pp** | **0.018** | 11 | 7 | 5 |
| `M:T2_PROCEDURE` | 22 | 18 | 81.8% [61,93] | **+29.0pp** | 0.021 | 22 | 10 | 5 |
| `M:T2_LEDGER` | 47 | 24 | 51.1% [37,65] | −27.5pp | 0.027 | 45 | 16 | 6 |
| `M:T2_SG_ISOLATE` | 19 | 16 | 84.2% [62,94] | **+30.6pp** | 0.028 | 19 | 10 | 4 |
| `M:T2_FB_VIEW` | 40 | 20 | 50.0% [35,65] | −24.3pp | 0.036 | 38 | 14 | 5 |
| `M:T2_VALUE_ACQUIRE` | 8 | 2 | 25.0% [7,59] | **−40.7pp** | 0.049 | 8 | 7 | 2 |

⛔ **이 상위 6칸 중 어느 것도 §1b~§1d 포렌식이 조사한 기전이 아니다.** 조사된 6종(claimprov-None · declaration failed · grounding 접두 · DUP/REFERENCE · BLOCKED · readloop/표면화)은 전부 이 순위 밖에 있다. 축자(발화 확인):

```
bank_k8141med1_20260903_2256.log.gz
  [sim=task_079#s626729] [T2_UNCALLED_UNLOCK] surface order_debit_card_5739
  [sim=task_079#s626729] [T2_STACK] audit route=[('출처 근거 확보','claim','T2_CLAIM_PROV'), …]
                                    chose=[('resolve_write','order_debit_card')] differs=True suppressed=['T2_UNCALLED_UNLOCK']
bank_g97151p11_viewmax2_20260903_1924.log.gz
  [sim=task_036#s626729] [T2_VALUE_ACQUIRE] consumers card_last_4_digits=1
```

`M:T2_UNCALLED_UNLOCK` 태스크 명단 `014,016,033,041,049,053,061,062,079,092,102`.

**이 표의 한계 (원인 진술에 쓰지 마라)**: ⑴ 태스크 난이도 미보정 ⑵ 자격 분모 미적용 ⑶ 11 sim 전부 n=1 ⑷ 전역 순열이 null 이므로 개별 p 는 **선택 후 순위**이지 유의성이 아니다.

---

#### 1e-3. 기전별 판정 — 가른다 / 안 가른다 / 표본 부족 / **철회**

| 기전 | 관측 (present pass/n) | Δ기저 | Fisher | 판정 | 등급 |
|---|---|---|---|---|---|
| claimprov-None (V1 rescued) | 45/74 | −39.2pp* | 1.000 | **검정 불가** (absent n=1) | 미판정 |
| claimprov-None (V2 tool-miss) | 13/25 (52.0%) | −14.0pp | 0.316 | **미탐지** (근거 교체) | 미판정 |
| claimprov-None (V3/V4) | 36/59 · 8/11 | −1.5 / +13.4 | 1.000 / 0.513 | 안 가른다 | 미판정 |
| `declaration failed (no-op)` | 8/11 (72.7%) | **+13.4pp** | 0.513 | **미탐지 · 부호 반대** | 미판정 |
| grounding 접두 드롭 | 9/12 (75.0%) | +16.3pp | 0.349 | ⛔**철회 — 분모 오류** | 미판정 |
| `[REFERENCE]` reference-unmatched | 2/11 (18.2%) | −31.0pp → **+4.2pp** (태스크 고정) | 0.113 | 안 가른다 (MH Σ(a−E)=+0.00) | 미판정 |
| `[DUPLICATE-WRITE]` | 0/1 | — | 1.000 | **표본 부족** (자격 1 런) | 미판정 |
| `[BLOCKED]` 부수차단 | 16/28 (57.1%) | −6.7pp | 0.628 | ⛔**철회 — 별칭 누락** | 미판정 |
| readloop-turn | 0/2 | −63.0pp | 0.146 | ⛔**철회 — 채점 검열** | 미판정 |
| 절차 표면화 잔존 (B_core) | 9/12 (75.0%) | +16.3pp | 0.349 | 안 가른다 (base 도 9/12) | 미판정 |

<sub>* V1 은 74/75 발화라 absent 칸이 1 sim(`task_050` reward 1.0)뿐 — 부호는 무의미.</sub>

**⛔ 철회 3건 — 무엇이 무너졌나 (반증 결과 · 그대로 남긴다)**

**⑴ grounding 접두 드롭 — *"안 가른다 / 닫아도 얻을 게 없다"* 를 철회한다.**
absent 칸의 **91%가 "안 났다"가 아니라 "날 수 없다"** 였다. 게이트 축자 (`t2_scaffold_get.py:2759`):

```
if os.environ.get("T2_SG_GROUND") == "1" and d.get("ground"):
```

A2 `ground` 선언 도구는 10개 중 **5개**(`check_rebate_qualification · get_correct_savings_apy · get_interest_correction · check_card_closure_eligibility · check_card_application_fit`)뿐이고, **그 도구를 한 번이라도 부른 sim 은 18/78 (23.1%)** 이다.

```
GW (분모 전체)      present 12 (9/3) 75.0%   absent 66 (37/29) 56.1%   Δ=+18.9pp  p=0.340
GW (분모 자격 sim)  present 12 (9/3) 75.0%   absent  6 ( 5/ 1) 83.3%   Δ= −8.3pp  p=1.000
자격 없는 60 sim 의 pass = 32/60 = 53.3%   ← 이들이 absent 칸을 끌어내렸다
```

⇒ 자격을 맞추면 **부호가 뒤집히고** absent 는 n=6 이 되어 검정 자체가 성립하지 않는다. **[미판정]** 로 내린다. (같은 자격 검사를 `[BLOCKED]`(다중호출 턴 보유 77/78)와 `declaration failed`(`agent_claimprov` 호출 78/78)에 걸면 분모가 안 줄어든다 ⇒ 이 둘은 통과.)

**⑵ `[BLOCKED]` 부수차단 — 하나의 코드 경로가 두 이름으로 나간다.** 축자 (`t2_gate_patch.py`, 로컬 미러 재확인):

```
:12786   _FB_GENERIC = "Error: resolve the flagged call(s) first; do not call this tool yet."
:12861       content = _FB_GENERIC
:12902   if content == _FB_GENERIC and os.environ.get("T2_KEEP_DENY_BODY") == "1":
:12903       _flag8 = next((x for x in (am.tool_calls or []) …), None)
:12910           _body8 = _sibling_wait("BLOCKED", _flag8, "what to fix")
:4536    return ("Error: [%s] this call was not run because another call in the same turn was blocked: "
                 "'%s' (see its own error for %s). Fix that one first, then re-issue this call."
go_stack.sh:816   export T2_KEEP_DENY_BODY=1
```

같은 `else` 로 떨어진 호출이 **형제 객체 `_flag8` 을 찾으면 `[BLOCKED]`, 못 찾으면 이름 없는 문면**으로 나간다. 기존 판정은 앞쪽만 세고 뒤쪽을 absent 에 넣었다:

```
[BLOCKED] 발화 sim 30 · 이름없는 _FB_GENERIC 발화 sim 64 · 둘 다 29 · 어느 것도 없음 13
★기존 absent 48 sim 중 35 (73%) 가 같은 코드 경로의 이름없는 출구를 갖고 있다

BLOCKED (기존 정의)   present 30 (16/14) 53.3%  absent 48 (30/18) 62.5%  Δ= −9.2pp  p=0.482
합집합 (코드 경로)     present 65 (35/30) 53.8%  absent 13 (11/ 2) 84.6%  Δ=−30.8pp  p=0.062
```

⇒ **−9.2pp 가 −30.8pp 로 바뀐다.** ⚠자인: 합집합의 absent 13 sim 은 *"게이트 마찰이 0인 sim"* 과 분리되지 않고, p=0.062 는 1e-2 의 전역 순열을 통과하지 못한다(그리고 사후 재정의다). **[미판정]** — 그러나 *"안 가른다"* 는 **정의 선택의 산물**이었다는 것은 확정이다.

부수 — repo 자신이 남긴 용량 주장은 **이 코퍼스에서 재현되지 않는다**. `t2_gate_patch.py:12871` 축자 *"실측(3 런·30 sim): 그 문구가 한 sim 에 3회 이상 나온 6건은 **6/6 전부 실패**"* ↔ 실측:

```
이름없는 문구 >= 3회   present 4 (2/2) 50.0%  absent 74 (44/30) 59.5%  Δ=−9.5pp p=1.000
   task_037 bank_k8143long3 gen=3 reward=0.0   /   task_047 bank_k8143med2 gen=3 reward=1.0
```

**⑶ readloop — 판정이 검열의 산물이다.** 로그에 등장한 (run,sim) 130 중 채점 78 (60.0%) · **미채점 52 (40.0%)** (`{'no_row': 44, 'reward_null': 8}` — 44건은 `results.json` 에 행 자체가 없다).

```
mech         scored/sim  unscored/sim   ratio
readloop           0.06        1.83     28.50x   ★
PROCEDURE          8.49        8.37      0.99x
gate_deny          1.05        0.92      0.88x
GW                 0.54        0.10      0.18x
BLOCKED            0.45        0.12      0.26x
lines            368.05      217.19      0.59x
미채점에만 존재하는 태스크 8: 027,048,063,067,068,069,077,084
```

⇒ readloop 은 미채점 sim 에서 **28.5배** 난다(`bank_049ctl2` 는 결과 파일에 sim 0개, `task_048` 은 24줄인데 행이 없다). **`n=2` 는 기전의 희소성이 아니라 채점 파이프라인의 결과다.** 길이 대리 가설은 같은 표에서 깨진다(`PROCEDURE 0.99x` · `gate_deny 0.88x` · 줄 수 0.59x). ⇒ **판정 불가(계기 결손)**. GW·BLOCKED·declfail 은 반대로 채점층에 몰려 있어 이 공격을 통과한다.

**⑷ claimprov V2 — 결론은 서지만 근거를 교체한다.** V2 의 present 태스크는 base 팔에서 **더 쉬운** 집합이다(base 기대 pass 59.6% ↔ absent 51.9%). 난이도 보정 시 Δ 가 −14.0pp → **−21.2pp 로 커지고**, base 난이도 2층 층화에서 **두 층 부호가 같다**(base-PASS −10.9pp · base-FAIL −32.9pp · **MH OR 0.349**). 즉 기존 보고는 *"flip 바닥과 비교"* 를 하면서 **교락으로 축소된 쪽**을 썼다. 그럼에도 p 는 여전히 0.05 밖이고 전역 순열 안이다 ⇒ **미탐지**.

**⑸ 검출력 자인 — 여섯 판정 모두 *"안 가른다"* 를 발화할 자격이 없다.** 각 present n 을 고정하고 해악 방향으로 p<0.05 가 되는 최대 pass 를 역산하면:

```
기전                    관측 present    배제된 최소 해악   탐지에 필요한 여분 실패
claimprov V2           13/25 = 52.0%   −26.0pp            6.5 건
DECLFAIL                8/11 = 72.7%   −41.2pp            4.5 건
GROUNDING(전체분모)      9/12 = 75.0%   −42.1pp            5.0 건
REFERENCE               2/11 = 18.2%   −38.2pp            4.2 건
BLOCKED                16/28 = 57.1%   −28.1pp            7.9 건
표면화 B_core            9/12 = 75.0%   −42.1pp            5.0 건
readloop                0/ 2 =  0.0%   해악 방향 어떤 값도 p<0.05 불가 = **검출력 0**
```

설명해야 하는 크기는 **회귀 10건**이다. present 12 sim 기전이 여분 실패 3건(회귀의 30%)을 냈다면 관측은 `present 8/12 vs absent 37/63, p=0.752` — *"안 가른다"* 라고 쓴 바로 그 칸이다. ⇒ **이 표본에서 쓸 수 있는 결론은 「미탐지」뿐이고, `안 가른다`·`원인이 아니다`·`닫아도 얻을 게 없다` 는 데이터가 지지하지 않는다.** 1e-2 의 전역 순열만이 *"전체로서 가르는 것이 없다"* 를 말할 수 있다.

**⑹ 살아남은 것 — 기저율 함정은 실재한다 (CONFIRMED)**

| 관측 | 값 |
|---|---|
| `declaration failed` 11건 중 **pass 8** | 72.7% (기저 61.3% 위) |
| GW 발화 12 sim 이 **전부 회복** — `064` GW×4 → ok×10 · `093` GW×2 → ok×2 | 12/12 |
| 동일 차단자 `give_discoverable_user_tool(submit_cash_back_dispute_0589)` 부수차단 | 7 태스크 **7/7 통과** (`task_022` 는 18회 받고 reward 1.0) |
| `[REFERENCE]` 7 태스크의 **base 팔** pass | 1/7 = 14.3% (원래 어려운 태스크) |
| pass sim 의 기전 수 median 53 ↔ fail sim median 52 | 차이 없음 |

---

#### 1e-4. 자연 실험 — `declaration failed` 는 A/B 가 아니었다

**설계 의도**: 선언 파싱이 실패하면 claimprov 가 no-op 이 되므로 *"레버가 꺼진 sim"* 이 공짜로 생긴다 ⇒ ON/OFF 자연 실험.

**결과: 잴 대상이 없었다. OFF 팔의 n = 0.**

```
total agent_claimprov gen calls: 420
  parsed OK (window hit(...)): 390     declaration failed: 18     gen **TRUNC**: 17
sims with ZERO successful claimprov window: **0**  []

sim                run                                      rew   ok fail empty claims  unb
task_005#s626729   bank_k151med1_20260903_2257              1.0    8    1     1     35    3
task_008#s626729   bank_re8143p11_20260904_1053             1.0   12    1     1    108    0
task_015#s626729   bank_k8143med1_20260904_0135             0.0    4    3     3     18    0
task_022#s626729   bank_lost5_viewmax2_20260903_1610        1.0   10    1     1    215    0
task_081#s626729   bank_g97151p11_viewmax2_20260903_1924    1.0    8    4     4    128    0
task_102#s626729   bank_g97151p11_viewmax2_20260903_1924    0.0    3    2     2     21    0
```

**⇒ declfail 은 sim 배정이 아니라 창(window) 단위 4.3% 결손(18/420)이다.** sim 을 "있음/없음"으로 가르면 *"창 하나를 놓친 sim"* 을 *"레버가 꺼진 sim"* 으로 오분류한다.

**코드가 그 이유를 정한다 — 창은 다시 열린다 (CONFIRMED 우리-층 사실)**

```
t2_gate_patch.py:14875   print("[T2_CLAIMPROV] declaration failed (no-op): %r" % (_ce2,), …)
t2_gate_patch.py:14877   if not _cl and not _pd:
t2_gate_patch.py:14878       print("[T2_CLAIMPROV] window hit claims=%s pending=%s" % (_cl, _pd), …)
t2_gate_patch.py:14979   self._t2_claimprov = getattr(self, "_t2_claimprov", 0) + 1   ← break 하류·`if _unbacked or _unb_p:` 안
```

예산 카운터가 오르지 않으므로 다음 사임 턴에 창이 그대로 재개된다. 로그가 교대를 보여준다(`bank_k8143med1_20260904_0135`, 축자 검산 완료):

```
[sim=task_015#s626729] [T2_GEN_TRACE] call=agent_claimprov max_tokens=8192 tb=None tool_choice=None [TERSE]
                                      -> gen=8192 prompt=22765 **TRUNC** reason=0B content=29666B tool_calls=0
[sim=task_015#s626729] [T2_CLAIMPROV] declaration failed (no-op): JSONDecodeError("Expecting ',' delimiter: line 1 column 29646 (char 29645)")
[sim=task_015#s626729] [T2_CLAIMPROV] declaration failed (no-op): JSONDecodeError("Expecting ',' delimiter: line 1 column 31224 (char 31223)")
```

**한계 (4종)**

1. **OFF 팔 n=0** ⇒ 자연 실험으로는 영영 안 된다. 스위치는 이미 있다: `t2_gate_patch.py:14853` `if (os.environ.get("T2_CLAIM_PROV") == "1" and (_resign or _cpv_transfer) …` ⇒ **선언적 A/B 짝 런**이 유일한 길이다.
2. **`TRUNC 1:1` 은 정확히는 틀렸다** — 18 중 17. 반례 2개: `task_028` 은 모집단 최장 프롬프트(102,566)에서 **절단 없이** 실패(`gen=7197 content=29669B`), `task_021` 은 `agent_selfdecl` 이 **절단됐는데 파싱 성공**. 진짜 불변량은 절단이 아니라 **선언 blob 28.2~39.1 KB 폭주**다. 폭주는 claimprov 질문에 특이적이다: `agent_response` TRUNC 0/1891 · `agent_claimprov` 17/420 (4.0%) · `agent_selfdecl` 1/395 (0.3%).
3. **컨텍스트 길이로 층화할 수 없다** — TRUNC 호출의 프롬프트 중앙값(38,178)이 비-TRUNC(40,752)보다 **짧다**.
4. **층화해도 n 이 안 남는다** — DECLFAIL 11 sim 이 전부 서로 다른 태스크의 단일 시행이고, 관측 fail율 27.3% [9.7, 56.6] 은 flip 바닥 18.8% 와 기저 fail율 38.7% **사이**에 있다.

**부수 산출 — 계기가 죽어 있다 (CONFIRMED · 별건)**

- **kind 는 이 코퍼스에서 상수다.** `kind-index rescued` 3262/3262 · `tool-miss fallback` 100/100 이 전부 `kind=None`. 선언 arm 은 09-01 이전에만 존재(`bank_x721_t1B_ctl_20260901_0945` nonNone=100/None=0 ↔ `night1p1_t3prime_20260901_2337` 이후 전 런 None-only). ⇒ **D8 의 *"런 전역 22/22"* 는 분모가 상수라 무정보다.**
  ★**그리고 2026-09-04 에 그 상수의 원인을 찾았다** — 상수가 된 것이 결론이 아니라 **증상**이었다.
  `f6224e26`(09-01 13:40)이 건 출력 스키마가 소비부가 읽는 `what` 을 `claim` 으로 개명하고
  pending 에서 `kind` 를 통째로 뺐다. 날짜 절벽(09-01 nonNone 전량 ↔ 09-03 None 전량)이 그
  커밋 하나에만 걸린다. **D8 은 이 발견으로 재작성됐다**(§1c-5 D8) — 처방이 *"침묵"* 에서
  **"스키마를 소비부 이름에 맞춘다"** 로 바뀌었다. 이 항목이 그 단서를 이미 들고 있었는데
  내가 *"무정보"* 로 닫고 지나갔다([[08]] 집계에서 결론 직행 금지 — 여기서는 반대로 **집계에서
  멈춘 것**이 실수였다).
- **`[T2_TRANSCRIBE]` deny 가 자기 로그 줄에서 죽는다.** 축자(코드 · 로컬 재확인):

```
t2_gate_patch.py:9674     _bad = _TR.mismatches(_sp, _args_dict(c), _byid)
t2_gate_patch.py:9683     _msg = _TR.note(_trs.get("_feedback"), _bad, getattr(c, "name", None))
t2_gate_patch.py:9685-9691  if not _msg:  _unk = _TR.unknown_ids(...)  →  _msg 재설정   ← _bad 는 여전히 []
t2_gate_patch.py:9716         % (getattr(c, "name", None), len(_bad), _bad[0][:2]),      ← IndexError
t2_gate_patch.py:9722-9723  tr_fb = None ;  print("[T2_TRANSCRIBE] error (no-op): %r" % (_tre,), …)
```

로그 축자(검산 완료 · 3회 발화):

```
[sim=task_024#s626729] [T2_TRANSCRIBE] live tool=get_reward_discrepancies rows=1 records=0
[sim=task_024#s626729] [T2_TRANSCRIBE] error (no-op): IndexError('list index out of range')
```

`records=0` ⇒ `_byid` 공집합 ⇒ `mismatches()==[]` ⇒ unknown-id 분기 확정. 모집단 18런에서 `[T2_TRANSCRIBE] deny` 발화 **0회**. **이름 있는 수리**([[64]]): `:9716` 을 `_bad[0][:2] if _bad else None` 로. 한 줄. 등급 **CONFIRMED 우리-층 (계기 · reward 원인 아님)**.

---

#### 1e-5. 수리 후보 재정렬 — 인구조사가 강화/약화/중립 한 것

> 규칙: **가르지 못하는 기전(또는 가름을 잴 수 없는 기전)을 겨눈 후보는 등급을 내린다.** 등급을 유지하는 것은 근거가 reward 가 아니라 [[23]]·[[64]]·계기 무결성인 경우뿐이다.

| 후보 | 겨눈 기전 (census 값) | 인구조사 효과 | 새 등급 |
|---|---|---|---|
| **D1** 절차 종결 후 표면화 중지 | `M:T2_PROCEDURE` 22/75 **82% (+29pp)** · `M:T2_PROC_ABSENT` 12/75 75% · `DET:readloop_turn` 2/75 (검열) | ⛔**강등** — 겨눈 기전이 **pass 쪽으로 기운다**. base 팔도 같은 12 태스크에서 9/12. readloop 다리는 판정 불가(28.5x 검열) | PLAUSIBLE → **미판정 (보류)** |
| **D2** 읽기 루프에 이름·출구 | `M:T2_SEARCH_EXHAUST` 12/75 58% (−3pp) · readloop(검열) | **약화 재확인** — 겨눈 칸이 중립이고 근거 다리는 검열됐다. [[63]] 과 어긋난다는 기존 지적 유지 | 약화 → **미판정 (보류)** |
| **D3** reference-filter 문면·술어 일치 | `DENY:REFERENCE` 2/75 · `M:T2_RESOLVE` 40/75 52.5% (p=0.104) | **중립.** 본체는 표본 부족. 계열 주장(*"문면이 검사한 것과 다르다"*)은 1e-3 ⑵ 의 `[BLOCKED]`/`_FB_GENERIC` 별칭이 **새 독립 사례**로 강화 | 계열 CONFIRMED / 본체 **미판정** |
| **D4** `[BLOCKED]` 을 의존 호출로만 | `DENY:BLOCKED` 28/75 (−6.7pp, p=0.628) → **별칭 합집합 65/78 (−30.8pp, p=0.062)** | ⚠**강등이 아니라 재정의.** 표적을 부수차단 전반에서 **`unlock` 경로**로 좁히면 이 코퍼스의 최대 신호와 겹친다: 하위분류 `unlock_*` 차단자 **1/6 (p=0.057)** ↔ census `M:T2_UNCALLED_UNLOCK` **3/11 (p=0.018 · Δ−39.9pp)** — 서로 독립 계기에서 같은 곳을 가리킨다 | **미판정 · 표적 재정의 후 우선순위 상향** |
| **D5** (철회됨) | — | 변화 없음 | 철회 유지 |
| **D6** DUP-WRITE 창 리셋 | `DENY:DUPLICATE-WRITE` 1/75 · `M:T2_DUP_WRITE` 1/75 | ⛔**강등 확정.** 자격 자체가 1 런이다 — `go_stack.sh:695 export T2_DUP_WRITE=0` 이고 `bank_k8143med1_20260904_0135` 한 런이 덮었다. 축자: `[sim=task_051#s626729] [T2_DUP_WRITE] deny tool=submit_credit_limit_increase_request (앞선 성공 msg=23)`. 인구조사는 §1d-6 의 강등을 **표본 부족**으로 재확인한다 | 미판정 유지 · **배선 전 `x548 --target 051` 필수** |
| **D7** grounding 접두 | `DET:grounding_warning` 12/75 **75% (+14pp)** · 자격분모 18 → Δ−8.3pp p=1.000 (absent n=6) | ⛔**강등.** 발화 12/12 sim 이 뒤이어 회복하고(`064` GW×4→ok×10), 실패 3건의 MISSING/WRONGARG 은 **GW 를 내지 않은 write 도구** 위에 있다. **reward 근거는 없다** — 남는 근거는 [[64]](거절이 처방을 못 준다)·[[23]] 뿐 | CONFIRMED(문면) → **미판정 (reward 무관) · 우선순위 하향** |
| **D8** claimprov `None` 금지 | `DET:claimprov_kind_None` 74/75 (**검정 불가**) · V2 26/78 (−13.5pp, 보정 후 −21.2pp · MH OR 0.349) | **중립 + 양화 정정.** *"22/22"* 는 kind 가 코퍼스 상수(100% None)라 무정보다. V2 만이 유일한 잔여 후보이고 그마저 전역 순열 안. 코드상 발화의 **3262/3362 (97.0%)** 는 kind 를 읽기 전에 찍히는 불활성 로그 | CONFIRMED(문면) 유지 / **인과 미판정 · 양화 재작성 필수** |
| **D9** 폐기 원문 원장 | — | **강화.** `[BLOCKED]` 희생자 이름이 영속 궤적 **0/133**, 사이드카 `tool-deny` 행에 `call_name` 이 **없다**(키 `channel,kind,len,sha,sim,simtag,text,turn`) ⇒ **부수차단의 표적은 측정 불가**. D4 의 판정이 D9 에 걸려 있다 | CONFIRMED · **선행 순위 유지** |
| **D10** declaration failed 침묵 | `DECLFAIL:T2_CLAIMPROV` 11/75 **73% (+11pp)** | ⛔**강등 + 근거 교체.** 부호가 반대이고, no-op 은 예산을 안 태운다(카운터가 `break` 하류 · `081×4`·`015×3` 반복 실증) ⇒ *"꺼진 것과 같다"* 는 **거짓**. 남는 근거는 reward 가 아니라 **토큰**: 18회 × ~8192 디코드가 아무 감사 없이 버려진다 | 계기 → **비용 누수 (reward 근거 없음)** |
| **D11** 재생성이 env-변이 호출 잃으면 기각 | `LEVER:T2_GATE_REGEN/usertoolnote` 18/75 56% (−6pp) · `givequote` 10/75 60% (−1pp) | **중립 · 단위 불일치.** D11 의 단위는 sim 이 아니라 **재생성 호출**(DROP 4/6 vs 5/35)이므로 sim 단위 인구조사는 판정할 수 없다. 인구조사는 이 후보를 **지지도 반박도 못 한다** | PLAUSIBLE 유지 · **측정 단위 별도 명시 필요** |
| **D12** `user_action_feedback` 오부착 | `M:T2_ACTIONREQ` **75/75 상존** | **중립 (측정 불가).** 상존 계기라 pass% 가 정의상 기저다. n=1 태스크 귀속이고 인구조사가 확인해 줄 것이 없다 | CONFIRMED(문면) 유지 / **reward 대응 미판정** |
| **D13** 정책 근거 없는 `[ORDER]` 강제 | `M:T2_SCAFFOLD_GET` **75/75 상존** | **중립 (측정 불가).** 근거는 [[23]] 뿐 — *"그 체인을 요구하는 정책 문장은 없다"*(`banking_knowledge.gate.json:4653`) | CONFIRMED([[23]] 위반) 유지 / **reward 대응 미판정** |
| **L1** 꺼진 열거 레버 | `M:T2_WRITE_ARG_ENUM` **발화 0** (3경로 확인) | **전제 CONFIRMED 재확인 / 기대수익 0 재확인.** 배선 회귀는 사실이고, 켜도 059·064 를 못 산다는 기존 판정은 그대로 | 전제 CONFIRMED / 기대수익 0 |
| **L2** `recommend_formalize` 격리 | `M:T2_FB_VIEW` 40/75 **50.0% (−24.3pp · p=0.036)** | ⚠**강화.** 뷰 주입 경로가 1e-2 상위 6칸 중 하나다. L2 의 오답이 이 경로로 전달된다 ⇒ 인구조사가 **독립적으로 같은 곳을 가리킨다**. 단 `T2_FB_VIEW` 는 L2 전용이 아니므로 귀속 아님 | 조사 · **우선순위 상향** |

**신규 후보 (인구조사가 고른 것 · 4칸 미충족 ⇒ 아직 후보 아님 · [미판정])**

| 축 | census | 왜 봐야 하나 |
|---|---|---|
| `M:T2_UNCALLED_UNLOCK` | 11/75 · **3/8 (27%)** · p=0.018 · 7런 | 코퍼스 최대 |Δ| 이고 **D4 하위분류(`unlock_*` 1/6)와 독립 수렴**. 축자: `[T2_UNCALLED_UNLOCK] surface order_debit_card_5739` · 같은 sim 에서 `[T2_STACK] … suppressed=['T2_UNCALLED_UNLOCK']` |
| `M:T2_VALUE_ACQUIRE` | 8/75 · 2/6 (25%) · p=0.049 · 7런 | 값 획득 경로. 축자: `[T2_VALUE_ACQUIRE] consumers card_last_4_digits=1` |
| `M:T2_LEDGER` | 47/75 · 51.1% · p=0.027 · 16런 | n 이 커서 자격분모·난이도 보정을 견딜 유일한 칸 |
| `M:T2_SUB_RECORDS` | 11/75 · 36% (Δ−25pp) | `[REFERENCE]` 와 **12/12 완전 공기** ⇒ 두 기전은 이 코퍼스에서 분리 불가 |

---

#### 1e-6. 다음에 재야 할 것 — 무엇이 더 있어야 판정되는가

**⑴ 반복 (n) — 이것 없이는 어떤 공격도 최종 방어 불가**
- base-PASS 42 태스크를 **같은 sha·같은 레버로 최소 3회**. 지금 코퍼스에 같은 조건 반복 쌍은 **3개**(같은 런 2 + 같은 `engine_sha` 1)뿐이고 그 3쌍의 불일치는 0/3 이다.
- 목표 검출력: 1e-3 ⑸ 의 "여분 실패 4.2~7.9건"을 잡으려면 **팔당 sim 수를 3배**로. 예: `[BLOCKED]` 의 6.7pp 를 flip 위에서 80% 검정력으로 잡으려면 팔당 약 **834 sim** — 즉 **그 크기의 효과는 영영 못 잰다. 재려면 표적을 좁혀야 한다.**

**⑵ 조건 식별자 — sha 가 못 쓴다**
24/24 런이 `dirty=True`. **레버 선언 스냅샷(`levers_on` 전량 + 표면형 문법 + `max_tokens`)을 `provenance.json` 에 남기는 것이 다음 런의 선결 조건**이다([[54]]·[[84]]).

**⑶ 채점 파이프라인 — 40%를 삼킨다**
미채점 52 (`no_row` 44 · `reward_null` 8). readloop 계열은 이걸 고치기 전에는 **판정 불가**다. `task_048`·`task_049`·`task_053` 을 채점되게 돌리는 것이 선결.

**⑷ 자격 분모를 모든 2×2 에 붙인다 (신규 계약)**
- GW = `ground` 선언 도구 호출 sim **18/78**
- DUP-WRITE = `T2_DUP_WRITE=1` 런 **1/18**
- V2 = `agent_claimprov` 호출 sim 78/78 (분모 안 줄어듦)
⇒ 자격 없는 sim 을 absent 에 넣으면 부호가 뒤집힌다(GW 실증: +18.9pp → −8.3pp).

**⑸ 기전은 문자열이 아니라 코드 경로로 정의한다 (신규 계약)**
`[BLOCKED]`/`_FB_GENERIC` 처럼 한 `else` 가 두 이름으로 나가는 자리가 최소 하나 실재한다. `t2_stack.TAG_TO_FLAG` 를 **저작 지점(파일:줄) 축**으로 재색인해야 한다.

**⑹ 격리 ([[78]]) — 지금 무료로 칠 수 있는 것 3개**
- **P-declfail**: `agent_claimprov` 만 `max_tokens` 를 올려 파싱률 전후 비교(격리 프로브 하나). [[82]] 재측정(Q3.8 정상 p99=385 토큰인데 이 서브콜만 4.0% 폭주).
- **P-unlock**: `M:T2_UNCALLED_UNLOCK` 11 태스크에서 `[T2_STACK] suppressed` 가 무엇을 눌렀는지 per-step. D4 재정의의 exit.
- **P-transcribe**: `:9716` 한 줄 수리 + 래칫. deny 0/18런 → 발화 확인.

**⑺ 부정통제 계약 ([[57]]) — 앞으로 모든 기전 판정에 의무**
① flip 바닥(base 팔) 대조 ② 자격 분모 ③ 태스크 난이도 보정(base 팔 기대치) ④ 동반 기전 분리칸 크기 ⑤ **1e-2 라벨 순열** — 개별 p 를 발화하기 전에 전역 순열을 먼저 통과해야 한다.

---

#### 1e-7. 원인 진술 4칸 ([[77]])

**(1) 주장 + 양화**
banking·Qwen3.8·arm=viewmax2·2026-09-03 이후 채점 sim **75**(29런 중 sim>0 18런 · 73 태스크 · pass 46 = 61.3%)에서, **188 기전 중 검정 가능한 148 종 어느 것도 reward 를 가르지 못한다**: p<0.05 가 6칸(기대 7.4) · p<0.01 **0칸** · 최소 p 0.0183 인데, reward 라벨을 300회 순열하면 **20%** 가 6칸 이상을 내고 **37%** 가 그보다 작은 최소 p 를 낸다 ⇒ 가족단위 p = **0.37**. 그 위에서 캠페인 회귀 **10/42 = 23.8%** [13.5, 38.5] 는 scaffold 를 끈 base 팔의 flip 바닥 **3/16 = 18.8%** [6.6, 43.0] 과 구분되지 않고(P(X≥10)=0.255 · 사용자 인용 25% 기준 P=0.629), **회귀 10건이 전부 n=1 sim** 이며 base-PASS 42 태스크의 채점 횟수 히스토그램이 `Counter({1: 42})` 다. 유일하게 반복이 있는 `055` 는 ours 에서 **1/3** 로 통과한다. 반대편 회복 13/29 = 44.8% 는 base 바닥으로 재면 밖(P=0.0012)이지만 ours 팔 base-FAIL flip 43.2% 와 겹치고, 팔 전체의 순효과는 부호검정 **p=0.678** 로 0과 구분되지 않는다. 개별 기전 판정 6종 중 **3종(grounding 접두·`[BLOCKED]` 부수차단·readloop)은 분모·정의·검열 오류로 철회**되고, 나머지 3종은 *"안 가른다"* 가 아니라 **"미탐지"** 다(검출력이 26~58pp).

**(2) 근거 — 축자 + 파일:줄 (전량 재검산)**

```
소스 (로컬 미러 C:\workspace\ba-frft\scripts\distill\tau2\ · sed 로 줄 확인)
  t2_gate_patch.py:4536   return ("Error: [%s] this call was not run because another call in the same turn was blocked: "
  t2_gate_patch.py:12786  _FB_GENERIC = "Error: resolve the flagged call(s) first; do not call this tool yet."
  t2_gate_patch.py:12902  if content == _FB_GENERIC and os.environ.get("T2_KEEP_DENY_BODY") == "1":
  t2_gate_patch.py:12910      _body8 = _sibling_wait("BLOCKED", _flag8, "what to fix")
  t2_gate_patch.py:12264  "051 은 코퍼스 전 sim 이 0점이라 실제로 잃은 점수는 없다."      <- Q38 base 1.0 으로 거짓
  t2_gate_patch.py:12871  "실측(3 런·30 sim): 그 문구가 한 sim 에 3회 이상 나온 6건은 **6/6 전부 실패**"  <- 재현 실패(2/4)
  t2_gate_patch.py:14875  print("[T2_CLAIMPROV] declaration failed (no-op): %r" % (_ce2,), …)
  t2_gate_patch.py:14979  self._t2_claimprov = getattr(self, "_t2_claimprov", 0) + 1    <- break 하류 = cap 미소진
  t2_gate_patch.py:13741  if (os.environ.get("T2_FOLLOWUP_READLOOP") == "1" and not _resign
  t2_gate_patch.py:13760      print("[T2_FOLLOWUP] readloop-turn counted as resignation", …)
  t2_gate_patch.py:9674   _bad = _TR.mismatches(_sp, _args_dict(c), _byid)
  t2_gate_patch.py:9716       % (getattr(c, "name", None), len(_bad), _bad[0][:2]),      <- IndexError
  t2_scaffold_get.py:2759 if os.environ.get("T2_SG_GROUND") == "1" and d.get("ground"):
  t2_scaffold_get.py:3267 _txt = ("[GROUNDING WARNING] %d input value(s) could not be verified against the "
  go_stack.sh:229  export T2_FOLLOWUP_REQUIRED=1 T2_FOLLOWUP_FORCE=1 T2_FOLLOWUP_READLOOP=1
  go_stack.sh:695  export T2_DUP_WRITE=0            go_stack.sh:816  export T2_KEEP_DENY_BODY=1

로그 (746 gz + 1160 plain 전수 스캔 · 전부 존재 확인 · 파일명 명시)
  bank_010ctl_20260904_0007.log.gz
    [t2_run] provenance -> data/simulations/bank_010ctl_20260904_0007/provenance.json (engine_sha=a208c8e0 dirty=True ctx=131072)
    [sim=task_010#s626729] [T2_CLAIMPROV] kind-index rescued: kind=None tool='verify_identity' 원장에 있다      (전량 988줄)
    [sim=task_010#s626729] [T2_CLAIMPROV] tool-miss fallback: kind=None tool='transfer_to_human_agents' 원장 밖 — kind 색인으로 강등
    [sim=task_010#s626729] [T2_SELFDECL] declared=(none — no-op)
    [sim=task_010#s373753] [T2_TOOL_OBS] id=chatcmpl-tool- err=True -> Error: resolve the flagged call(s) first; do not call this tool yet.
  bank_049ctl2_20260904_0534.log.gz
    [sim=task_049#s373753] [T2_FOLLOWUP] readloop-turn counted as resignation                                    (전량 7840줄)
    [sim=task_049#s373753] [T2_TOOL_OBS] … -> Error: [BLOCKED] this call was not run because another call in the same
      turn was blocked: 'call_discoverable_agent_tool(get_closure_reason_history_8293)' (see its own error for what to fix).
    [sim=task_049#s373753] [T2_PROCEDURE] checklist proc=credit_card_closure_retention nodes=6 done=0
      left=['disputes','pending_replacement','prior_attempts','log_reason','retention_offer','close']
  bank_k8143med1_20260904_0135.log.gz
    [sim=task_015#s626729] [T2_GEN_TRACE] call=agent_claimprov max_tokens=8192 … -> gen=8192 prompt=22765 **TRUNC** content=29666B tool_calls=0
    [sim=task_015#s626729] [T2_CLAIMPROV] declaration failed (no-op): JSONDecodeError("Expecting ',' delimiter: line 1 column 29646 (char 29645)")
    [sim=task_051#s626729] [T2_DUP_WRITE] deny tool=submit_credit_limit_increase_request (앞선 성공 msg=23)
  bank_g97151p11_viewmax2_20260903_1924.log.gz
    [sim=task_041#s626729] [T2_RESOLVE] deny reference-unmatched param=transaction_id (치환 폐기·표면화 배달)
    [sim=task_036#s626729] [T2_VALUE_ACQUIRE] consumers card_last_4_digits=1
  bank_k8141med1_20260903_2256.log.gz
    [sim=task_079#s626729] [T2_UNCALLED_UNLOCK] surface order_debit_card_5739
    [sim=task_024#s626729] [T2_TRANSCRIBE] live tool=get_reward_discrepancies rows=1 records=0
    [sim=task_024#s626729] [T2_TRANSCRIBE] error (no-op): IndexError('list index out of range')
  bank_lost5_viewmax2_20260903_1610.log.gz
    [sim=task_093#s626729] [T2_TOOL_OBS] … -> [GROUNDING WARNING] 3 input value(s) could not be verified against the
      account records / knowledge base and were dropped: base=4.0 (source not found in the knowledge base); …

정본 문서
  x738_q38_base97_census_2026_09_04.md:13  "정본 = bank_x644_q38base_bank78_20260830  (78 태스크)"
                                     :17  "pass 42 · fail 55"
                                     :66  "판별자: 런 로그의 `[T2_` 마커 유무."
  x737_next_run_plan_2026_09_04.md:121     "§1c 의 3 sim 에서 이 병리(절차 정체·readloop)는 재현되지 않았다"
```

**(3) 반증 조건 (동시 기재)**

- **R1** — base-PASS 42 태스크를 **같은 sha·같은 레버로 3회 이상** 재채점했을 때 회귀 10건이 3/3 로 재현되면 잡음 귀속이 무너지고 진짜 회귀가 된다. 지금은 10/10 이 1/1 이다.
- **R2** — base 팔 flip 3건(`008`·`012`·`017`)이 x599 의 **다른 서빙 조건**(포트·concurrency·모델 이어받기 [[30]])에서 온 것이면 18.8% 바닥은 과대평가이고 23.8% 가 신호가 된다. ⇒ **x599 의 `/v1/models` 대조 로그 확인이 선결**.
- **R3** — 라벨 순열을 **런 안에서만**(런을 층으로 고정) 돌렸을 때 관측 6칸이 순열 상위 5% 밖으로 나가면 1e-2 의 전역 null 은 무너진다. 현재 순열은 런 구조를 깨므로 이 검사가 남아 있다.
- **R4** — 자격 분모와 태스크 난이도를 동시에 보정한 뒤 `M:T2_UNCALLED_UNLOCK` 이 여전히 p<0.05 면 그 칸은 **가른다**로 승격된다. 지금은 두 보정 모두 안 걸렸다.
- **R5** — `[BLOCKED]`/`_FB_GENERIC` 이 서로 다른 원인이라는 것을 코드로 보이면(= `_flag8` 이 None 인 칸이 다른 게이트에서 온다는 증명) 별칭 철회가 무너진다. `:12903` 의 후보 집합은 `denied_by_objid ∪ main_prov ∪ _SRC8` 하나뿐이라 현재 반례가 없다.
- **R6** — 미채점 52 sim 이 채점된 것과 **같은 기전 분포**를 가지면 readloop 철회는 약해진다. 다만 같은 표에서 `PROCEDURE 0.99x`·`gate_deny 0.88x` 로 길이 대리 가설은 이미 깨진다.
- **R7 (분모 표류)** — 이 절의 census 는 75 sim, 반증 재계산은 78 sim 이다(런이 진행 중 3건 추가). **도는 런 위에서 잰 표**이므로 모든 절대 수치가 다시 움직인다([[54]]: 도는 런의 조건은 바꾸지 않되, 그 위에서 잰 표는 재잰다).

**(4) 선행 확인 (실제로 찾아본 경로)**

```
ls   reports/facet_rft_2026/ | grep 2026_09        -> x737_next_run_plan · x738_q38_base97_census (2건)
read reports/facet_rft_2026/x737_next_run_plan_2026_09_04.md:10-90, 600-712, 1166-1280   (D1~D13·L1·L2 정의·등급)
read reports/facet_rft_2026/x738_q38_base97_census_2026_09_04.md:1-90                    (정본 base 42/55 · 팔 판별자)
grep -rn "10/42|회귀 10|23.8"  reports/facet_rft_2026/*.md  _cdp_private_local/*.md      -> repo 문서에 없음(라이브 집계뿐)
grep -rn "declaration failed" --include=*.py .   -> 저작 3곳 (14760 WRITEPROV / 14875 CLAIMPROV / 15082 SELFDECL)
grep -rn "not run because another call" --include=*.py .   -> t2_gate_patch.py:4536 단일 저작 지점
grep -n  "_sibling_wait" t2_gate_patch.py        -> 정의 4498 · 호출 5자리(PROVENANCE/ARG-SCHEMA/DISAMBIGUATE/POLICY GATE/BLOCKED)
grep -n  "_gflags" t2_scaffold_get.py            -> 6회 전수 · **침묵 드롭 경로 0** (드롭 ⟺ 플래그 ⟺ 접두)
grep -n  "_bad" t2_gate_patch.py                 -> 9674 대입 ↔ 9716 사용 사이 재대입 없음
grep -n  "T2_DUP_WRITE|T2_KEEP_DENY_BODY|T2_FOLLOWUP_READLOOP|T2_PROCEDURE=|T2_PROC_ABSENT" go_stack.sh
a2/*.gate.json 전수에서 `ground` 선언 추출        -> scaffold_get_tools 10개 중 5개
로그 전수 스캔 (746 gz + 1160 plain · 19 패턴)     -> 인용 축자 19/19 존재 확인 (파일·줄 예시 첨부)
정본 계기 위임: t2_forensic.path_for / log_text / sidecar_rows / simtag / mutation_diff / reward_basis  (사본 0 · [[67]])
```

---

### 1f. 새 실패 6건 per-step 포렌식 — 워크플로 13에이전트 (2026-09-04 오후 · `wf_14f8393a`)

> 대상 = `029 039 048 060 063 084` (11:36~14:41 완료 · §1e 분모에 없던 것들 · 전부 `['DB']` · n=1).
> 방법 = 태스크당 «발견자 + 독립 반증자» 짝 6 + 완결성 비평 1. 축자 인용 검산 오류 **0건**.
> Q3.8 이력: **여섯 다 재실험**이다 — 029·060·063 은 ours 통과 이력이 있고(flip 권역),
> 039·048·084 는 팔을 갈아도 전부 실패(지속 실패). ⛔n=1 — 태스크 단위 인과 서술 금지.

#### 1f-0. 판정 표 (반증 후)

| task | 발견자 등급 | 반증 후 | 층 | 무엇이 있었나 |
|---|---|---|---|---|
| **029** | CONFIRMED | **CONFIRMED 유지** | our_layer | ★모델이 msg83 에서 **정확히 옳게 거절**(*"I can't update rewards based on the discrepancy list alone"* · tool_calls=0)했는데, 우리 넛지 3연발(`uncalled_unlock`→`searchexhaust`→`claimprov`)이 그 턴을 재생성해 **금지된 write 5건을 커밋**(EXTRA 5). 반증자가 `_wev_deny_msgs` 를 오프라인 재실행 — **5/5 DENY**(게이트가 봤다면 다 막았다). 단 MISSING 1(txn_e647… 분쟁 미제출)은 **우리층무관**(모델이 스스로 과다지급 1건을 뺐다 — 우리 도구는 6건 전부·그 txn 을 첫째로 나열했다) ⇒ EXTRA 5 를 다 막아도 reward 는 안 산다(반사실 미판정) |
| **039** | 우리층무관 | **우리층무관 유지** | user_sim/model | WRONGARG 본체 = `issue_noticed_date` **8/8 이 "09/04/2026"**(gold "11/14/2025" — 오늘 날짜를 사고 날짜 칸에). `FAILURE_MASTER:220` 이 이미 이 단독 수리의 매수를 0 으로 판정. D3 는 3회 발화(문면 거짓 · 세 번째 독립 사례) 후 상한 소진 — 8/8 dispute 는 결국 실행됐다 |
| **048** | CONFIRMED | **CONFIRMED 유지** | our_layer | ★`T2_UNCALLED_UNLOCK` 술어가 (unlock ∧ 미호출) 만 보고 **호출이 금지인지 안 본다** — 금지 표적(`log_closure_reason` green)을 호출로 강제했고 그 재생성 라운드에서 금지(WEV)가 꺼져 커밋. 029 와 **같은 구멍의 두 얼굴** ⇒ D14 로 통합. 정정 1건: turn29 재생성 소유자는 D1(절차 체크리스트)이 아니라 **regen 채널** |
| **060** | CONFIRMED | ⛔**PLAUSIBLE 강등** | our_layer? | D8 `"None: None"` 문면 재생성이 close unlock 을 밀어 **순서 파괴**. 강등 사유 = 같은 sim turn3 에 같은 압력·무전환 형제(`regen tool_calls=[]`) + 폐기 초안 1089B 가 D9 구멍으로 소실 — 초안이 이미 그 행동이었는지 확인 불가 |
| **063** | 우리층무관 | **우리층무관 유지** | model | account_class Gold 고정(x509 ②범주 축). READ-FIRST 4건은 막았지만 [48]→[52] 로 정상 회수(형제 통과) ⇒ 원인 아님. 별건 관측: `get_correct_savings_apy` 가 **자격(최소잔액 $10,000 · 손님 $8,000) 검사 없이** Gold 5.7% 를 확인해 줬다 — §1f-5 #7 측정 대상 |
| **084** | 미판정 | ⛔**우리층무관 강등** | model | D3 3회 발화 후 dispute **3/3 실행**(형제 통과 — 발견자가 성공한 read 를 "폐기 draft" 로 오독했고 그 오독이 유일한 우리층 서술 근거였다). 모델은 msg44 에서 gold 값을 정확히 유도했다가 msg46 에서 **스스로 뒤집었다** |

#### 1f-1. ★M1 — 사임-창 재생성이 「말할 턴」을 「쓸 턴」으로 바꾼다 (4/6 태스크 · 10턴 · 전수)

모델 초안 `tool_calls=0`(산문)을 우리 재생성이 폐기하고 호출 턴으로 교체한 사건 전수:

```
산출 = gold 밖 EXTRA write 7 · gold 요구 write/unlock 2 · 순서 파괴 1 · 거부 1 · read 3
채널 = claimprov 4 · unified_regen 3 · uncalled_unlock 2 · followup_chain 2
술어 = t2_gate_patch.py:13735  _resign = (not tool_calls and content)   <- "말만 하는 턴" 이 표적
```

⛔**[[70]] 부호표 — 이 레버는 순해악이 아니다**: 048 t63(gold `pay_credit_card`)·t123(gold unlock)은
재생성이 **만들어 낸 gold 칸**이다. 029·048 발견자 둘 다 순해악으로 서술했고 이 표가 그것을 정정한다.

#### 1f-2. ★M2 — 그 재생성 호출은 **쓰기 게이트를 원리적으로 지나지 않는다** (D14 의 코드 근거)

```
_wev_deny_msgs 실호출부 = 전 코드베이스 단 1곳  t2_gate_patch.py:9955
    wd = None if _fab_only else _wev_deny_msgs(...)          <- while True: 루프 안 (9124~13220)
_ap_regen 정의 = :13765 (루프 밖) · 실호출부 29곳
_ap_regen 내부 재검사 = _denied_calls(gate) · T2_PROCEDURE · T2_UNLOCK_NAME · T2_UNLOCK_PROV  넷뿐
=> 재생성 산출 호출은 T2_WRITE_EVIDENCE · T2_WRITE_ARG_GROUND · T2_ARG_EMPTY · T2_REF_VERIFY ·
  T2_ASK_UNKNOWN_BOOL · T2_HANDOFF_ARG_GROUND (wtag 6종) 를 지나지 않는다 — 29채널 전부
_fab_only 는 루프 안의 두 번째 문 — 이 6 sim 에서 그 예외가 보전한 것 = 0
  (T2_WRITE_ARG_GROUND 발화 0 · T2_REF_VERIFY 0 · WAG_DECOUPLED 0 — 켜져 있는데도)
```

**⇒ D14 신설 — 재생성 산출물이 쓰기 게이트 밖에서 커밋된다** (CONFIRMED 우리-층 · 029+048 통합).
⚠**반사실 reward 예측 = 이 6건에서 0** (비평자 계산 · [미측정]): 029 는 MISSING 1 이 남고,
048 eco 는 forbid 술어 불성립(msg43 *"No closure reason records found"*)이며, 060 `close_bank_account`
는 `write_evidence_specs` 14개 밖이다. ⇒ 수리 순위는 reward 가 아니라 **[[25]] 정본 의무**(우리 층이
금지를 스스로 우회하면 안 된다)로 선다. §1f-5 #1(3판 db_check 재실행)이 선행.

**D15 (약화 등재)** — 039 의 deny 처방(*"re-fetch"* 조건절) ↔ `[DUPLICATE-READ]` 억제 상호모순.
실재하나 인과 불성립(문면이 조건절이라 명령 격상 금지 · 비평 4-5).

#### 1f-3. 기존 후보 이동 (6건 계수)

```
D8   ★최강 강화 — rescue 445/445 kind=None · 창 17/17 · "None: None" 전송 10 ·
     그리고 그 문면이 write 를 민 사건 3 (029 t72 · 048 t29 · 060 t59)
     => 피해 등급 「낭비」->「오작동 write 의 저작」
D9   ★구속 조건 확정 — 5/6 에서 폐기 원문 소실이 판정을 막았다 (060 강등의 유일 사유 포함)
     D9 없이는 D4·D11·D14 를 다음 런에서도 판정 못 한다 => 수리 1순위
D4   빈도 강화(24건) / 피해 미측정 (희생 인자가 D9 구멍으로 전멸)
D3   4/4 sim 상한 3 소진 · 4/4 최종 write 통과 (041·081·039·084) — 술어 결함 강화 · reward 인과 재약화
D7   강화 + 범위 2중 정정: ①063 은 접두가 아니라 접미((doc_….md)) ②048 오검출 — 손님이 msg69 에서
     실제로 말한 3.25% 를 "never mentioned" 로 드롭
D1   인과 약화 (048 turn29 소유자 정정) / 관측 유지 (checklist 78행)
D6   ⛔0/6 발화 — "오늘 가장 확실한 우리-층 결함" 표현을 내린다
D11  ★방향이 반대 — 현행 술어(변이 호출을 잃으면 기각)는 이 배치 피해 7건(더한 것)을 못 잡는다
D13  거울상 — 060 은 순서 태스크인데 [ORDER] 0건 (과발화 불만 <-> 여기선 미발화)
D12 무증거 0/6 · D10 1건 · L1 기대수익 0 재확인 · L2 연결 축자 0
```

#### 1f-4. 규율 정정 2건 (이 문서 자신에 대한)

- **채점 단위 함정은 양방향이다**: 051_6(match=False 인데 DB 움직임)의 반대편 — **060 은 7/7 match=True
  인데 db_match=False**(msg89 인자가 gold 와 바이트 동일 · env 가 *"Account eligibility requirements not
  met"* 로 거부). `action_match` 는 도구 성공 여부를 보지 않는다.
- **귀속 (d) 를 레버 종류로 가른다**: 차단 게이트 = 「같은 게이트를 통과한 형제」 / **push 레버**
  (claimprov·uncalled_unlock·followup) = 「같은 압력을 받고도 행동이 안 바뀐 형제」. push 는 sim 당
  1회 상한 때문에 구판 (d)가 **자동 충족**되고(048), 상한 있는 차단 게이트는 fail-open 때문에
  **자동 불충족**된다(039·084) — 규율 결손이 두 방향의 등급 오류를 만든다.

#### 1f-5. 재야 할 것 (§1e-6 에 추가 · 비평자 11항)

```
#1  ★db_check 3판 재실행 (원본 / EXTRA 제거 / WRONGARG 교정) — 반사실 reward 0/6 미측정
#2  048 eco 에 _wev_deny_msgs 오프라인 재실행 (forbid 불성립 예측 검증)
#3  ★D9 를 먼저 — 이것 없이 D4·D11·D14 다음 런에서도 판정 불가
#4  _ap_regen 29채널 × 전 코퍼스 「초안0->재생성>0」 gold 안/밖 비율 census ([[70]] 코퍼스 부호표)
#5  029 t72 3넛지 개별 OFF 부정통제 ([[57]])
#6  039 base 팔 user-sim 이 (9/4/2026) 를 말하는가 — user_sim 귀속의 반증 조건
#7  063 get_correct_savings_apy 자격 술어 부재 측정 (최소잔액 검사 0)
    ⚠1f-7: 067(자격 충족·정상 산출)·068(None 반환=술어 작동)에서 미재현 — 발현 n=1 · D16 미자격
#8  084 customer_max_liability 412.88 의 KB 유도 경로 (KB 결손 vs gold 임의값)
#9  084 동일날짜 중복쌍 tie-break — KB 에 규칙 없음 · 모델이 맞게 유도했다 스스로 뒤집음
#10 T2_WEV_ROUNDS 코드 경로 (발견자·반증자 둘 다 모름)
#11 engine_sha · n_sims 회수 ([[85]] — 미회수 상태로 타 런 대조 금지)
```

#### 1f-7. 신규 실패 4건 — 027 · 067 · 068 · 086 (워크플로 9에이전트 · `wf_bdb58e3f` · 반증 4/4 생존)

> 전부 flip 권역 재실험(086 만 base 0·첫 ours)·n=1·`['DB']`. args_equal 필터로 거짓음성을
> 걸렀다(027 에서 whitespace 위양성 3칸 제거 — 진짜 변이만 셌다).

| task | 등급 | 층 | 진짜 변이 | 무엇이 있었나 |
|---|---|---|---|---|
| **027** | **CONFIRMED** | our_layer | MISSING 1 + **EXTRA 1** | ★**D14 신 사례(3번째)**: WEV 게이트가 같은 write(update e403)를 정상경로에서 **5회 live-DENY**(t55·57·63·65·71) 했는데 `searchexhaust` 재생성 1발(t73)이 WEV 재검 없이 **커밋**. MISSING(과다지급 e506 미분쟁)은 029 와 동일 패턴의 모델 판단(도구는 *"each needs a cash back dispute"* 로 4건 전부 나열) — 우리층무관. ⚠반사실 reward 미판정(EXTRA 막아도 MISSING 잔존) |
| **067** | 우리층무관 | model | MISSING 1 (`apply_for_credit_card` · user) | 007 정확 재현(카드 미추천). **L2 자재결손 n↑**: 카드 로스터가 서브의 `ctx[-8:]` 창(t2_resolve.py:1076) 밖 → `applies=false` ×3 은 서브의 옳은 답이지만 재료가 없었다. 063 자격술어와 **무관 배제**($100k=Platinum Plus min 충족·7.3% 정상) |
| **068** | 우리층무관 | model | WRONGARG 1 (`account_class`) | `'Green Account (checking)'` ↔ gold `'Green Account'` — **env-KB 함정**(msg7 이 괄호형을 유효 공식명 예시로 못박음). ★`T2_SIBLING_PAREN` 이 **정확히 탐지하고 수리값까지 지목**했는데 log-only 미무장([[81]]). 처방은 deny 가 아니라 **괄호 STRIP**(deny 는 재발화-루프 위험) — 무차별 STRIP 의 gold-손상은 §T-8 A/B 실측 전 미확정 |
| **086** | 우리층무관 | model | WRONGARG 6 (4× `customer_max_liability` 0↔50 등) | **P5 파생값 측정 자리 신 sim**(085 쌍 · compute_ops 는 08-19 의도적 공집합). 정정: *"우리층이 안 닿았다"* 가 아니라 — `write_rules`(specific:10318 · 격리 20/20)·`distinct_args`(:10344) 레버가 **실재하되 write-point 전달 레버가 기본 OFF·미발화**([[81]] delivered-where) |

**★코드 결합 발견 (P9 에 박제)**: `searchexhaust` 재생성은 `t2_gate_patch.py:14432`
`_resign or _srchex_mid` 로 게이트된다 — **M1 의 사임-창이 D14 채널의 트리거 표면을 공유**한다.

**★횡단 명제 — 이 4건의 공통 형상은 「억제-원인」이 아니라 「remedy 공백」이다**: 3/4 가
우리층무관이고, 그 옆에 도울 수 있던 레버가 **미무장(068)·미발화(086)·자재결손(067)** 상태로
서 있었다. 다음 런의 물음은 *"우리가 무엇을 막았나"* 만이 아니라 **"도울 수 있던 레버를 왜 안
전달했나"**(무장·발화·급양)다.

**D16 승격 기각**: 063 의 자격술어-부재(get_correct_savings_apy 가 최소잔액 검사 0)는 067(자격
충족·정상 산출)·068(None 반환 = 술어 작동) 에서 **미재현** — 발현 n=1 이라 §1f-5 #7 단독 측정
대상으로 유지.

#### 1f-7c. 미분류 8건 분류 확정 — 040 053 056 066 071 078 082 085 (워크플로 wf_e364035a · 9에이전트 · 축자 검산 18/18)

## wf_e364035a — 8건 A/B/C/D 분류 (§1f-7b 덧붙임용)

**분모 규율 확인**: 8/8 전부 `reward_basis=['DB']`·`db_match=false`. action_checks 는 지도로만 썼고 args_equal 필터 적용(040_4/5 는 공백 직렬화 거짓음성 — 실호출 msg42/44 인자 gold 축자 동일·실행 성공 → 변이집합에서 제외). 085 sim1 은 `infrastructure_error`·msgs 0·reward None(내용 없음 — 채점 sim 은 sim0 뿐). ⛔전건 n=1 — 회귀·인과 서술 없음([[85]]).

### 8행 표

| 태스크 | 군 | 축/결함 | 한 줄 근거 (갈림 msg · 직전 우리층 발화 · 축자) |
|---|---|---|---|
| 040 | **B** | 파생값 선택(provisional 자격 2칸) | 8/8 dispute 실행·6/8 전칸 일치, 잔여=`eligible_for_provisional_credit` 2칸(e503·e510 True↔gold False — gold 는 «prior 1건+제출 누적>2» 누적 계수 의미론, 모델은 정적 계수+사유범주만 적용). 갈림 msg49 직전 우리층 발화 **«없음»**(t38 이후 FB 0건). 정책 축자 실재: *"has not filed more than 2 disputes in the past 12 months"*(msg4) |
| 053 | **D** | 미판정(DB diff 필드 미확정) | gold 16/16 이 인자 **바이트 동일**로 실행됨(dispute 14키 전칸·CLI 7500→22500 일치·전 호출 무오류) — MISSING 0·WRONGARG 0. 잔여 후보=유저 `get_card_last_4_digits` 실행 1회(msg50·gold 는 give 까지만·대본 §4 *"If the agent provides you with the get_card_last_4_digits tool, use it"* 로 실행 강제) — 이 EXTRA 의 DB 효과는 env 소스 리모트라 로컬 검증 불가. 반증조건: 리모트 `dbdiff_task.py <tag> task_053` 1회가 확정한다 |
| 056 | **B** | 값-선택(저축상품) + 요건 미유도 | 갈림 msg61: *"My recommendation: Emerald Saver"* — 현재 이체액 $3,200 에 정박해 Silver Plus 를 *"❌ Below minimum"* 으로 배제. 대본 §7 예치 요건(*"I can probably put $5,000-$10,000 in there to start"*)은 msg49→61 사이 미발화(agent 미질의). 이체 오목적지(dc2e…)는 동일 선택의 연쇄. 직전 우리층 발화 **«없음»**(t45-89 FB 는 t48 claimprov 뿐) |
| 066 | **B** (+C 병존) | 값-선택(카드) · D-계열 SIBLING_PAREN 미무장 | 카드칸: EcoCard↔gold Silver Rewards — genesis 는 모델 자신 msg63(푸시 이전), t75 `[RECOMMEND-OFFER] "'card_type=EcoCard' is the match"` 는 동방향 재지목(서브 LLM 산출·레버 OFF 반사실에서도 갈림 유지 → 락은 레버가 아님). 병존 C: t58 `T2_SIBLING_PAREN` 이 *"'Green Account (savings)' 에서 '(savings)' 를 빼라"* 로 고칠 값까지 댔으나 전 arm `=log`(x737:445 — 집행 0) — 닫힌 술어·단 **단독 수리 매수 0**(카드칸 잔존) |
| 071 | **B** | 값-선택(상품) + 자발 EXTRA write | `account_class` Lime Green↔gold Sky Blue(msg59) + gold 에 없는 transfer $3,000(msg66) — 이체는 모델의 자발 제안 msg62 *"Would you like me to transfer an opening deposit…?"* 을 유저심이 수락(대본 §7 에 자금이체 없음). 두 갈림 모두 직전 우리층 내용성 발화 **«없음»**(t56/58 FB_VIEW 는 원장 사실 주입 — 059 선례상 상품 무관 D1 계열) |
| 078 | **B** | 절차 선택(freeze-first 생략) — 우리층 반대방향 관철 | 유저 msg1 축자 *"I need to freeze all of them right away and get replacements."* 에도 lb·green 은 freeze/unfreeze 5액션 전량 미실행·직행 close. 갈림(t36-39 close 시도)에서 우리층 `[OPERATOR-SCOPE]` 가 **반대방향**으로 freeze/close 범위를 표면화·반려했으나 모델이 재발행 관철. close 자체는 gold 내 액션이라 cap 통과가 오답을 낳은 구조(③축) 아님 |
| 082 | **B** (+A①′ 병존) | Reg E 파생값 4칸 지배 · 푸시 조기-제출 부수 | `customer_max_liability_amount` 4/4 오답(500·89.99·100·500↔gold 50 — 085 동형·푸시 무관 기전) 이 단독으로도 r=0. 병존 ①′ 실측: t41 claimprov·t43 `[UNLOCKED-NOT-CALLED]` 푸시 직후 msg41 *"I'll use reasonable defaults for the few details I was still confirming (you still have both cards in your possession…)"* → 유도 4칸(possession·pin·discovery×2) 추가 오염 — ①′ 부호표 «잃는 편» 짝으로 등재 |
| 085 | **B** | Reg E 파생값(50) 3칸 + provisional 1칸 | msg41/47/64 dispute 3건 모두 id·범주·날짜 정확, 오답은 `customer_max_liability_amount` 3칸(=disputed_amount 복사)+`provisional_credit_eligible` 1칸(085_7 False↔gold True·반대방향). 우리층 `T2_DISTINCT_ARGS` 가 3회 정확 탐지(*"== disputed_amount (100.0) — 선언상 달라야 한다"*)했으나 **trace 전용 — turns 38-66 FB 전달 0건**(x737:2330 «write-point 전달 레버 기본 OFF» 정합). 대본 §5/§6 금액-불일치 조건절은 정확 항행(3번 스킵·4번 $14.99 제출) |

### 정정 목록

1. **A①축 반대편(040·085) 등재 — 확인 + 한정 정정**: 오지목은 현행 런에도 실재한다(040: t26·26·28·32·34·36 `formalized_target=submit_transaction` **7회** / 085: t61 **1회**) 그러나 **전량 `T2_ACTIONREQ_GROUNDED` 침묵** — 축자 *"침묵: formalized_target=submit_transaction 가 이 대화 축자에 0회 — 근거 없는 지목은 하지 않는다 (TASK_072 §7-2)"*. ⇒ go_stack:747 의 29건 반대편 **실측 등재는 유지**하되, 두 태스크의 **현행 락 분류는 A① 이 아니라 B** — 억제 술어(집합 소속+축자 대조·닫힌)가 이미 그 편을 지키고 있고 잔여 실패는 파생값 칸이다. §2398 괄호를 태스크 자체의 군 배정으로 읽으면 오독.
2. **040 선행(FAILURE_MASTER:220) 대비 차이**: t7346 의 `address` 7/8·`issue_noticed_date` 8/8 오답 축은 이번 런에서 **소멸**(8/8 전부 gold 일치) — 잔여가 `eligible` 2칸으로 수렴. «issue_noticed_date 단독 수리 매수 0» 판정과 정합(그 축이 사라져도 r=0).
3. **085 선행(x737 §1f-7·3329-65) 대비 차이**: 3+1칸 목록은 동일 sim 의 기지 사실 — 신규 실측은 ① write-window 우리층 **전달 0건**(탐지-미전달 확정) ② t61 오지목 침묵. «다시 계산하면 [[23]]·[[62]] 위반» 판시에 따라 B 유지.
4. **D 남발 점검**: D 는 053 하나 — gold 전 액션 바이트-일치를 확인하고도 DB 갈림 필드를 로컬 재료로 못 닫는 경우라 «모른다» 가 정답([[77]]). 유일 후보(유저 도구 실행 EXTRA·give 는 gold 에 있음)와 반증 절차(리모트 dbdiff 1회)를 명기했다. 참고로 gold 가 give 만 두고 실행을 안 두는데 대본이 실행을 강제하므로, 유저 실행이 해시에 든다면 **대본-정합 궤적으로는 통과 불능**(env/gold 정합성 문제) — dbdiff 로 확정할 것.
5. **판별식 일관성(검산 ②)**: 닫힌 갈림인데 A 로 간 것 0건·열린 갈림인데 C 로 간 것 0건. C-성 결함은 066 SIBLING_PAREN(닫힌·수리처 명명됨) 하나였으나 병존 B 칸 때문에 군은 B(+C 병존·단독 수리 매수 0) — C600 «버그를 트레이드오프로 부르지 마라» 와 «매수 0 이면 락이 아님» 둘 다 충족. A-성 병존은 082 의 ①′ 뿐이며 부호표 짝으로만 등재.
6. **축자 검산(검산 ①)**: 인용 18개 전부 표본 재개봉 substring PASS(18/18).
7. **관찰(원인 아님·[[84]] D8 계열)**: claimprov 빈 문면 *"None: None"* 이 8건 중 6건 런에 상존(040 t5·056 t48·066 t54·071 t66·078 t36·082 t5/32/41) — 078 에선 원장이 promised=[close×2, freeze_lg] 를 알고 있었으므로 문면이 채워졌어도 같은 오경로를 밀었을 것(인과 아님 확인), 그러나 렌더 결함 자체는 살아 있다.

파일: 번들 `…\scratchpad\x742\task_{040,053,056,066,071,078,082,085}.json.gz` · 검산 스크립트 `…\x742\verify_quotes.py`·`diffgen.py` · 선행 `C:\workspace\ba-frft\reports\facet_rft_2026\x737_next_run_plan_2026_09_04.md:2394-2419,3329-3365,430-460` · `FAILURE_MASTER__20260822.md:220` · `C:\workspace\ba-frft\scripts\distill\tau2\go_stack.sh:735-760` · `t2_gate_patch.py:7483,13306-13322` · `dbdiff_task.py:8-15`.

**★접기 두 건 (내가 보탠다)**:
- **078 = 080-핀의 두 번째 표적**: 유저 축자 *"I need to freeze all of them right away"* 에 직행
  close + [OPERATOR-SCOPE] 반려를 모델이 재발행 관철 — C601 핀(가역/비가역 선택의 고객-회부·자답
  금지 = P11)의 사정권. 군은 B 유지(현행 레버 기준)·P11 부호표 «사는 편» 짝으로 등재.
- **053 의 D 는 벤치-정합성 후보**: gold 는 give 까지만 두는데 대본 §4 가 유저 실행을 강제 —
  유저 실행 행이 해시에 들면 **대본-정합 궤적으로는 통과 불능**. §1f-5 #17 로: 리모트
  `dbdiff_task.py <tag> task_053` 1회가 확정한다(env/gold 층 판정 — [[21]]/[[68]] 절차).

**§1f-7b 최종 계수 (실패 42 기준 · 083/087/097 비행 제외)**:
```
A 내재절충 10 (010 027 029 038 039 041 048 060 061 084)
B 원리락  20 (007 026 037 040 046 054 055 056 063 064 066 067 069 071 077 078 082 085 086 101)
C 버그     5 (014 015 051 068 092)     D 미판정 2 (053 059)
M 측정중   1 (049)                     E 분모제외 1 (102)
판별식-통과 후보(핀 대기) 1 (080 · P11 — 078 이 부호표 짝)
```

#### 1f-9. task_010 — 캠페인 0/5 정밀 포렌식 (워크플로 10에이전트 · `wf_bfb0cc8f` · 반증 감사 반영)

> 대상 = ctl×2(viewmax2) · treat×2(viewmax2_actdemand) · g97×1, 대조 = night2p1(09-01 ·
> viewmax2 동족 · PASS) · x723(다른 팔 · PASS). prior 2/3 ↔ 캠페인 0/5 (초기하 P=3/28≈10.7% ·
> 반우연은 약함). 반증 감사가 **1차 분석의 주장 7건을 죽이거나 정정**했다 — 아래는 생존분만.

**한 줄**: gold 010_1 은 «유저-쓰기»(user 가 **본인 id** 로 `submit_referral`)다. 0/5 는
**레버 회귀가 아니라** ①모델층이 duplicate 조항(platinum_008 — 지원 중복 조항)을 referral
재제출에 오적용 + 발명 규칙(*"attribution before application"* — pass 에도 존재) →
*"retrying is possible"* 을 부정 → ②**user-sim 스크립트 §7 의 전건이 소멸**(*"…and retrying
is possible, go ahead and submit a new referral"*) → escalation 기본값(###TRANSFER### ×4 ·
장외 이탈 ×1)의 사슬이다. pass 는 유저심이 §7 전건을 날짜 산수로 **자가충족한 관대 적용**
(에이전트가 *"permanently lost"* 라 했는데도 제출).

**레버 회귀 기각의 근거** — 창(09-01 23:41→09-04 00:07) 내 엔진 커밋 19개 전수 diff:
referral/ACTIONREQ/[ACTION]/digest/GB2 경로 커밋 **0건**. 유일한 전-팔 거동 커밋
`39e541a0`(PROBE_MAX_TOKENS 2048→8192)은 갈림 턴 4/5 가 우리층 무개입 순수 prose 라 인과
경로 미성립. ⚠단서: engine_sha 24/24 dirty([[85]]) — 커밋 이력 기준 배제이지 작업트리 증명
아님. / ⚠[[54]] 단서: fail 번들에만 `seed` 키 실재 — user-sim 하니스 델타 가능성 미해소
(§아래 재측정 #1이 판정 전제).

**우리층 인과는 5 중 1 (idx1 ctl-t1) — 단 회귀 아닌 선재 결함**:
M1 재생성 ×3(t23/27/29)이 모델의 거부 초안을 안내로 전환(방향은 gold 편!)했는데, 산출물이
`user_id` 의 referent 를 **«친구 id»로 오지정** — 직전 reasoning 축자 *"Which user_id? Could
be referred person's? Or referrer? Need infer."* 유저심 §5 는 본인 id 를 스크립트했으므로 이
오지정 하나가 §7-무장 유일 fail 을 죽였다(손님이 친구 이메일 받으러 ###STOP### 이탈).

**⇒ D16 신설 — «[ACTION] 선언 불완결 (referent 무구속)»** (CONFIRMED · **선재**: 도입
`a84041e1` 07-22 · `cff28b00` 08-03 · `839c10a4` 08-10 — 전부 창 밖):
`t2_gate_patch.py:10560` *"Arguments of {tool}: {args}"* 가 **인자명만** 주고 referent 를 안
묶는다. 수리 = 그 도구를 언급하는 **env 문서 축자 스니펫**(예: doc_009 *"Tell the customer to
pass their own user_id"*) 동봉 — 엔진은 읽어 전달만([[71]]) · 출처 env 뿐([[23]]) · 원장-표면화라
[[58]] 안전. ⚠[[31]]: `t2_levers.py` 정본 원장에 동종 항목 유무 대조 후 번호 확정. 격리 프로브
(P10: 같은 t27 재료 + 스니펫 동봉 뷰) 통과 후 배선.

**[[64]] 일반 규칙 수리 후보 — 무명 계수 transfer notice**: GB2/WORK-INCOMPLETE 가
*"7 item(s) … 0 acted on"* 을 세면서 **미이행 유저-행동 도구명을 안 댄다**(종결부 4/5 개입 —
반증 A2 정정: idx0 도 turn46 발화·transfer 종결). pending_user 원장은 env 소유권표 유래라
[[23]] 안전. 부기: env doc_009 는 *"Do not transfer to a human in these cases"* 라 적어
무전송을 문서로 승인 — 이 조항 표면화도 같은 수리의 재료.

**관찰 (원인 아님)**: idx4 의 `[T2_ACTIONREQ] 침묵: …대화 축자에 0회` — formalize 가
submit_referral 을 만들고도 침묵(5/5 반복 공백 · [[64]] 방향 보강 후보). ⚠원인이 아닌 이유:
instruct 가 실제로 발화한 idx1 도 r=0.0 (referent 오지정으로). / ctl↔treat 의 T2_CALC
발화 여부가 갈림 — A/B 1-변수 규격 깨짐([[54]] · 별도 보고).

**다음에 재야 할 것**: #1 user-sim 동일성 대조(최우선·판정 전제 — `seed` 키 델타) ·
#2 t3prime 팔 현행 sha n≥5 재발사([[09]] 승인 필요 — ≥3/5 pass 면 «요행» 기각·0/N transfer-동형이면
회귀 재개) · #3 idx1 t27 뷰 실측 diff([[78]] — doc_009 문면이 뷰에 있었나) · #4 P10 격리.

#### 1f-7b. 태스크별 락 분류 — 내재 절충 / 원리 락 / 버그 (2026-09-04 밤 · 사용자 질문 «트레이드오프로 더 못 올리는 태스크»)

```
A. 내재 트레이드오프 락 (~10) — 끄면 반대편 태스크가 되돌아온다. [[70]] 부호표 대상 유일 집합
   ①  push 지목 ↔ 오지목 억제      010            반대편 실측: 오지목 29건(040 085 074 057 063 055)
   ①′ push 강도 ↔ 잔여 EXTRA      027 029 048 060  D14 수리 후에도 절충 잔존(048 eco spec 공백)
   ②  deny 턴 비용                038            ACTION-basis 방어 ↔ DB-basis 16메시지
   ③  cap fail-open ↔ deny 루프   039 041 084     W-5: cap 없으면 18회 루프 · 있으면 소진 후 오답 통과
   ④  연좌 차단 ↔ 부분 실행        079            §1d-10 «우리층만 고쳐도 통과 못함»
B. 원리 락 (~14) — [[23]]/[[58]] 금지선. 모델 값-선택/판단. scaffold 매수 = 실험 무효. [[13]] 학습 축
   007 026 037 054 055 063 064 067 086 101 + 지속실패 일부
C. 버그 (= 락 아님 · ~5) — 014(D12) 051(D6) 015(D11ⓑ) 068(STRIP) 060(D8) — 수리 후보 ①묶음
```
**미분류 18건 해소 (2026-09-04 밤 · 사용자 지시 «전수 분류»)** — 기존 포렌식 즉시 배정 6:
```
088 → B   금액·범주 유도(customer_max_liability·transaction_type) — x509 «표를 줘도 0/6» 기측정-실패 경로
092 → C   D14 계열 — 모델이 gold 호출 시도 → [OPERATOR-SCOPE] deny → 재생성이 방향 전환·재시도 0
          («미호출» 아닌 «시도-차단-미재시도» · x749b CONFIRMED)
061 → A①′ push 시점 정본 사례 — uncalled_unlock 재생성이 수수료-대화 창을 건너뛰고 즉시 해지
          (게이트 무관 통과 경로라 D14 수리로 안 잡힘 · 인과 PLAUSIBLE — D9 병기)
059 → D   신설 «D. 미판정» — 간접 사슬 3/4 충족·(d) 불충족·폐기 원문 소실(D9). 추정 금지가 정답
049 → M   신설 «M. 측정 진행» — 049ctl2/049treat A/B(nt2)가 지금 돌고 있다 — 그 결과가 판정
102 → E   신설 «E. 분모 제외» — [[68]] 리더보드 분모 제외 태스크
```
잔여 12건 = 워크플로 2개가 판정 중: `wf_891fd63f`(046 069 077 080) · `wf_e364035a`(040 053 056
066 071 078 082 085 — A/B/C/D 분류 전용·판별식 적용). 완료 시 이 절의 계수를 최종 확정한다.

⛔**이력 열의 함정 (2026-09-04 22:4x 사용자 정정으로 발견)**: x756 표와 이 절의 «prior ours» 는
**로컬 회수분 한정**이라 과소집계다 — [[30]] 역방향 함정(회수 안 된 것을 부재로 단정).
리모트 전수 재검 실측: **099 는 Q2.5 42/112 통과**(내 «지속 실패» 분류 오류 — 그리고 21:57 에
viewmax2 로 실제 통과) · **091 은 Q2.5 0/21**(«이력 없음» 아님 — 미측정은 base-Q3.8 뿐) ·
**096 은 Q2.5 0/24 → Q3.8 1/2 = 세대 첫 승**. 진짜 전-세대 hard-0 블록 = 046 069 077 082 083
087 097(Q2.5 0/23~34 · x509 hard-0 축 정합). ⇒ 규칙: **«없다/지속 실패»는 리모트 전수 스캔
후에만 발화한다** — 이력 주장 전 `iso_tau3`+`scratch` 두 클론의 simulations 를 다 긁는다.

처방 원칙: A 는 끄기가 아니라 **술어 정밀화 + 도메인 일반 조건**([[19]]·[[70]]) — 각 축의 양편
태스크를 부호표의 짝으로 등재한다. B 는 이 실험의 **경계 측정 그 자체**(논문 가치)로 남긴다.
D(미판정)는 D9 수리 후 재판정 큐로, M(측정 진행)은 결과 도착 시 즉시 재배정한다.

**★A군의 발생 원리 (논문 프레임 · 2026-09-04 밤)** — 넷은 한 원리의 네 얼굴이다:
scaffold 는 닫힌 술어([[50]] 3조건)만 쓸 수 있는데 «이 개입이 지금 옳은가»는 열린 술어다.
열린 집합의 닫힌 근사는 위양성·위음성을 동시에 갖고, 레버는 총량이 아니라 **교환비**만 고른다.
태스크가 락되는 조건 = gold 경로가 그 근사의 오류 영역을 통과할 때.
```
①  지목 옳음 = 열린 술어 · 축자-근거 요구는 «이름을 아직 모르는 도구»에서 구조적 위음성(닭-달걀)
①′ push 방아쇠 = 닫힌 대리(tool_calls=0 ∧ 원장-미이행) ≠ 원조건(«부당한 멈춤») — 029 msg83 의
    정당한 거절을 구분 못한다 (Goodhart)
③  게이트는 거부만 하고 생산은 금지([[62]] 저작 금지) ⇒ 처방 완결 불가([[64]] 와의 긴장) ⇒
    비수렴 생성기 앞에서 종단 정책은 무한거부(루프) 아니면 k회 후 개방뿐 — cap 은 저작-금지의 그림자
②  액추에이터가 채점되는 채널 그 자체([[25]]) — 사이드채널 없음 ⇒ 보장은 대역폭으로 지불
④  호출 묶음의 의존 구조는 모델 의도 안(열린) · 엔진은 동시-발생(닫힌)만 본다
```
**따름정리(A/C 판별식)**: 갈림 조건이 닫힌 술어로 표현 가능하면 절충은 버그로 해소된다 —
051 실증(갈림=상태 변화, 닫힌 술어 ⇒ D6 리셋으로 C군 이동) ↔ 010(갈림=지목 관련성, 열린 ⇒ A군 잔류).
새 절충이 나오면 이 질문을 먼저 던진다.

#### 1f-8. P9 프로브 신설 (§3 에 편입)

- **P9 — D14 격리**: 팔 = 선언 오버라이드 한 칸(`regen_calls_reenter_write_gates = on/off`).
  재료 = 029 t72 · 048 t36/t55 · 027 t73 메시지 전량. **exit** = off 에서 커밋 ∧ on 에서 deny.
  ⛔[[70]] 부호표 필수 — **048 t63(gold pay)·t123(gold unlock)이 함께 죽지 않는지** 센다
  (예측: `pay_credit_card_from_checking` 은 `write_evidence_specs` 밖이라 무영향 · [미측정]).
- 스모크 추가: **`_ap_regen` 산출 호출 중 게이트 체인 밖 커밋 = 0** (§4 (3) 표에 D14 행).

---

#### 1f-10. 잔여 4건 per-step 포렌식 — 046 · 069 · 077 · 080 (워크플로 `wf_891fd63f` · 발견자+반증자 짝 4 · 반증 4/4 생존)

> 대상 = §1f-7b 잔여 12건 중 4건. 이력: 069 지속실패(0/3째) · 077 첫 ours · 080 flip 권역(prior 1/1 PASS) · 046 지속실패(0/4째). 전부 `['DB']` · n=1 · seed 626729. ⛔n=1 — 태스크 단위 인과·회귀 서술 금지([[85]]).
> 등급 변경 0건 — 그러나 반증자가 발견자 논거를 3건에서 교체·1건에서 프롱을 죽였다(아래 ③). **4/4 가 model 층** — 이 배치에서 우리층 원인 0, «remedy 공백» 횡단 명제(§1f-7) 재확인.

##### 1f-10-0. 판정 표 (반증 후)

| task | 발견자 등급 | 반증 후 | 층 | 무엇이 있었나 |
|---|---|---|---|---|
| **069** | 우리층무관 | **유지 — 단 논거 교체** | model | close-first 로 14일 tenure checking 자멸(msg52 회수불능) → savings MISSING + Green EXTRA. ★반증자의 진짜 방어 = **클래스 수렴 반사실 봉쇄**: 사용자가 모든 분기에서 Gold savings 를 명시 승인(msg71·89·101)·gold 는 Silver Plus ⇒ 순서를 우리 레버로 완벽히 고쳤어도 db_match false — 우리층 수리로 flip 할 경로가 이 sim 에 없다. ★발견자가 [[68]] **표적 금지 069**(노트 "Gold NO ATM rebates" ↔ KB "$30/월 rebate")를 재발견만 하고 미인용([[74]]·[[40]] 위반) |
| **077** | CONFIRMED(model) | **유지 — 자재결손 프롱 사망** | model | 두 축 독립 치명: ①EXTRA dispute ×5(+provisional $1,200) = 모델 자발 범위확장(env-bait 미래일자 인출, 우리층 발화 0) ②WRONGARG STANDARD/CLASSIC(gold RUSH/PREMIUM) = 완전 정보 하 모델 기본값 + D8 push PLAUSIBLE 상한. ★모델 자필 축자([63] reasoning): *"Given the system wants me to DO the work, I should use reasonable defaults: STANDARD delivery (free) and CLASSIC design (free)"* + 허위 기억 *"I asked but they didn't answer"*(§8 질문은 debit 쪽에서 0회) — **D8→행동의 in-instance 내용 인과가 모델 자필로 남은 첫 실물** |
| **080** | 우리층무관 | **유지** | model | 사용자 freeze 명령 2회(msg1·49)를 close-직행으로 대체 → green 비가역 CLOSED. 선회 근거는 env 자료(doc_026 *"recommend closing instead of freezing"*). 우리층은 역방향: OPERATOR-SCOPE 가 freeze 이름+범위 병기 deny 3회(t58/70/75) — 3/3 무전환, t123 대칭 작동(gold-일치 무변경 재발화 = 레버 무선택성 방증). freeze deny·차단 0회 |
| **046** | 우리층무관 | **유지 — 강화** | model | EXTRA 1(pay $125)이 유일 해시 이동원·gold 3/3 완수. 갈림 msg[29](잔액 선표면화) 직전 claimprov 는 pending=0 **무개입** — 지불 제안은 압력 0 의 모델 자생. ★반증자가 완화 각주를 죽임: doc_003 disputes-우선 체크리스트가 msg[3]·[4]·[18] 로 [29] **이전 3회 문맥 실재** — "모델이 순서 명시 문서를 보유한 채 역행". ★반사실 양팔 무PASS: claimprov 무개입 팔은 resign→046_1/2 미실행=MISSING — **7291 이름의 최초 등장이 우리 PROCEDURE deny 문면[56]**: dispute 체크 자체가 우리 레버 산물(deny 의 gold-칸 창출 실물, [[70]] +쪽) |

##### 1f-10-1. ① 기존 기전 이동 (4건 계수)

| 기전 | 이동 | 근거 (이 4건) |
|---|---|---|
| **D8** | ★★최강 강화 (연속 2배치째) | **4/4 sim 발화**(069 t45/96·077 t43/63·080 t5/30·046 t44) · "None: None" 전송 전건. 신규 2면: ⓐ077 [63] 자필 reasoning — push→기본값 기입의 내용 인과 축자(수리 우선순위 논거 최상급) ⓑ080 — 원장은 freeze 약속을 추적(`locked=['freeze_debit_card_3892'×2,…]`)했는데 렌더가 이름을 소거해 '이름 실린 상기'가 OPERATOR-SCOPE 경유뿐이 됨. ⚠단 reward 인과는 **4/4 비결정**(069 클래스수렴 봉쇄·077 형제 t41 반례·080 (d) 3/3·046 양팔 무PASS) — 등급 구조는 §1f-3 「오작동 write 의 저작」 그대로 |
| **D9** | ★강화 — 구속 사례 +3 | 069 t45 폐기 1774B 초안 소실(claimprov 가속 반사실 PLAUSIBLE 로만 남은 유일 사유) · 080 t30 규탄 초안 원문 소실 · 046 t44 resign 원문 소실. 수리 1순위 재확인 — D9 없이는 이 배치에서도 push 인과 판정이 막혔다 |
| **M1** | 강화 + **매수면 신규** | 069 t45: tool_calls=0 1774B 초안(말할 턴)→write 2건(쓸 턴) 교체, 개입~실행 사이 타 입력 0. 077 t63: 강행 전환. 반례: 046 t44 산출은 KB 검색(쓰기 아님) · 077 t41 은 질문 응답 — M1 은 결정론이 아니라 창-조건부. 매수면: 046 은 **deny(PROCEDURE)가 gold 칸 2개를 창출** — §1f-1 부수 매수 원장에 채널 구분(regen발/deny발) 추가해 +1 |
| **D13** | 거울상 강화 +2 | 069(tenure 순서)·080(freeze 순서) 둘 다 순서-치명 태스크인데 [ORDER] 발화 0 — 060 포함 3건 수렴. 단 순서 규칙의 출처가 env 문서라 A2 선언은 [[23]]/[[58]] 심사 대상(«remedy 공백» 처리, 아래) |
| **«remedy 공백»(1f-7)** | ★강화 +2 | 046: 절차 레버(nodes=6)가 disputes 미이행을 t26 부터 알고도 mapped-tool 호출 전 **구조적 침묵**(발화점 공백) + pay 도구가 절차 노드 밖이라 무게이트. 069: 순서/tenure 제약의 우리층 선언 부재(A2 relations.edges 는 get_all_user_accounts 뿐) |
| D16 | 주변 관측 +1 | 077 VALUE-ACQUIRE 가 debit 활성화 창에서 credit-dispute 문면 3연발(모델 3회 저항·무해) — referent/창 무구속의 동류이나 채널 상이. 080 t119 도 동형. 승격 근거 아님(둘 다 비원인) |
| D1 | 중립 | 046 절차 체크리스트는 무장돼 있었으나 침묵 — 인과 발화 0 (결함은 발화점, «remedy 공백» 쪽) |
| D6 | 재약화 | 0/4 발화 — §1f-3 의 0/6 에 누적, **0/10** |
| D2·D3·D4·D7·D11·D12·D14·D15·L1·L2 | 중립 | 이 4건에서 인과 발화 0. L1: 069 'Gold Account' 는 실재-이름이라 ARG_ENUM 무관(x509 98.6% 통과 재확인). D14: 게이트-거부-후-재생성-커밋 형상 자체가 이 4건에 없음(069·077 의 regen 발 write 는 어떤 게이트의 금지 표적도 아니었다) |

**신규 후보 (4칸 계약 전 — 번호는 `t2_levers.py` 정본 대조 후 확정 [[31]])**:
- **«핀-집행 provenance 공백»** (046 · CONFIRMED 우리층 · 비원인·지연만): `T2_PIN_READ` 가 핀+`tool_choice` 로 **우리 자신이 강제한 호출**을 operator-fab(t2_resolve.py:215)이 2회 거부 — `_t2_proc_pin`(t2_gate_patch.py:9519-9527)이 `_t2_our_names` 미등재(등재자는 4229·593 둘뿐). t7324-050·OL-02 와 같은 류 3번째 발화자 — 수리 = 등재 한 줄.
- **«RECOMMEND-OFFER 원장 무대조»** (069 · CONFIRMED 우리층 · 비인과): user 의 기실행(msg69 gold-일치)을 `_offered_in_history`(t2_resolve.py:1147-1152)만 보고 push → 우리 gate(banking gate.json:4553)가 deny 하는 push↔deny 자기모순·regen 1회 낭비.
- **«모델 자발 범위확장(gold 밖 정당-write)»** (077 EXTRA dispute) — 기존 명단 무대응. ⛔신설 보류: model 층이고 dispute 는 정책상 정당해 scaffold 처방이 [[23]]/[[58]] 금지선에 걸린다. §1f ⓑ방향(재생성발 EXTRA)과 **다른 것**(재생성 아닌 산문 제안발)임만 명단에 각주.

##### 1f-10-2. ② 4건 횡단 반복 기전

1. **「보유한 계획·지시·문서의 실행-시점 방기」가 4/4 공통 형상**: 069 는 reasoning(msg32/35/38)에서 tenure-안전 순서를 도출해 놓고 burst 속에 방기 · 080 은 freeze 명령 2회를 close 로 대체 · 046 은 disputes-우선 문서 3회 보유 채 잔액 선표면화 · 077 은 §8 질문 의무를 기본값 기입으로 대체. retrieval 결손이 아니다 — 자료는 전부 컨텍스트에 있었다([[79]] "Q38 잔여는 retrieval 이 아니다" 정합). [[13]] 학습 축.
2. **user-sim 조건절·즉흥 동의가 4/4 에서 증폭기**: gold 인자·gold 분기가 조건절 뒤에 봉인돼 있고(077 §8 ONLY-IF → RUSH/PREMIUM · 080 step4 "After debit cards are frozen" → green 구출), 에이전트가 전건을 안 만들면 소멸하고 스스로 만들면(046 [29]→§3 지불 요청) 이탈이 발화한다. 080·069 에선 스크립트 밖 즉흥 동의가 gold-이탈 DB 를 사용자-승인으로 잠갔다. [[21]]상 종결 귀속은 4/4 model 유지 — 단 010 §7 발견의 일반화: **조건절 전건 관리가 per-step 판정의 상설 항목**이어야 한다.
3. **반사실 봉쇄가 (d) 형제보다 먼저 판정을 끝낸다**: 069(클래스 수렴)·046(양팔 무PASS)·077(두 축 독립 치명) 전부 «우리층 수리로 reward 가 뒤집힐 경로가 있는가»가 등급을 확정했고, (d) 형제는 보조였다(069 에선 형제 칸이 아예 부적격이었는데도 등급이 섰다). 규율 승격: 귀속 절차에서 **반사실-경로 검사를 (d) 앞에** 둔다.
4. **D8 은 매 sim 발화하는데 매 sim 비결정**: push 인과의 판정 불능이 3/4 에서 D9(폐기 원문 소실) 탓 — §1f-3 의 「D9 수리 1순위」가 세 번째 배치에서도 재확인.

##### 1f-10-3. ③ 채점단위·축자 오류 (발견자 오류 원장 — 반증자 적발분)

- **046 축자-거짓 1**: "doc_003 은 msg[61]에야 등장" — 실제 msg[3](bm25 #8)·[4](dense #2)·[18] 실재. 완화 각주 삭제·"보유 채 역행"으로 강화 기재.
- **069 실질 허위 1 + 과잉단정 1 + 선행 미인용 1**: "act-before-read"(doc 002 전문이 msg8 BM25 로 기실재) / "close-first 는 약속에 이미 있었음"(원장은 도구명만 저장 — 안전/치명 계획 판별 불가·D9 로 미확인인데 단정, §77 위반) / [[68]] 표적 금지 069 미인용([[74]]·[[40]]).
- **077 프롱 사망 1 + 파일명 오기 1**: "KB 스니펫 절단 자재결손" — 절단은 라이브가 아니라 **read.py show() cap=4000 표시 절단**(M[42] 실측 12,202자·완전한 절 char 3959 실재·같은 표가 6곳 노출). 자재결손 프롱·unknowns(1) 삭제. / 생산부 스키마 위치는 `t2_run_gated.py:387-400`(발견자 표기 `_t2_terse_schemas` 오기).
- **080 수치 오기 2**: action_checks "19 true"→실측 **26 true**/8 false(총 34 · 번들 재검산 완료) · "write 19 일치"→**17**. false 8 식별은 정확 — 판정 비개입.
- **채점 단위 자체는 4/4 건전**: 직렬화 거짓음성 0·060형 거짓양성 0(069 거부 2건·046 err=True 0건은 DB 무변이라 대조 칸 부재) · 046 의 C297 allowlist 적용(unlock 무변화·비allowlist read 스킵) 정확. 잔존 추정 1: 077_6 여벌 키 `expedited_shipping:false` 의 DB 등가는 결과 문면("Shipping Method: Standard")로만 — env 코드 검증 항목.
- **⇒ 계기-층 교훈**: 077 프롱 사망의 원인은 포렌식 리더 자신의 표시 상한이었다. `read.py` 절단부에 `[TRUNCATED n/total]` 명시 필수 — 안 하면 다음 배치에서도 '라이브 절단' 오독이 재생산된다([[64]] 형: 이름 없는 절단은 자기 원인을 재생산).

##### 1f-10-4. ④ 설계서 반영 (절 · 무엇)

| 절 | 변경 |
|---|---|
| §1f-3 | D8 계수 갱신(+4 sim · 077 자필 축자 병기) · D9 구속 사례 +3 · D6 0/10 · D13 거울상 +2(069·080) · M1 에 «창-조건부» 단서와 deny-발 매수 채널 추가 |
| §1f-5 | 항목 추가: #12 `read.py` 표시 절단 명시(계기 수리 — 선행) · #13 `_t2_proc_pin`→`_t2_our_names` 등재(t2_levers.py 대조 후) · #14 RECOMMEND-OFFER 원장 대조 · #15 077_6 여벌 키 DB 등가 env 검증 · #16 080 prior-PASS 궤적의 freeze 경로 여부(gz 회수) |
| §1f-4 | 규율 3번 추가: **반사실-경로 검사를 (d) 앞에**(1f-10-2 #3) · per-step 표준 항목에 «user-sim 조건절 전건 관리» 상설화 · prior_check 의무 범위에 **메모리 색인 grep**([[68]] 미인용 재발 방지) 명시 |
| §1f-7 | «remedy 공백» 명단 +2(046 발화점 공백 · 069 순서선언 부재) · 신규 후보 3건(핀-provenance·RECOMMEND-OFFER·범위확장 각주) 등재 · D16 주변 관측 +2 병기 |
| §1f-7b | 계수 확정 4건(아래 1f-10-6) — 잔여 12→8(`wf_e364035a` 몫) |
| §1c-5 D8 | 077 [63] 자필 reasoning 축자를 D8 항의 «내용 인과 증거» 칸에 박제 — 수리 우선순위 논거의 현재 최상급 |

##### 1f-10-5. ⑤ 080 — flip vs 인과 (prior ours 1/1 PASS → 이번 fail)

- **인과 주장 불가**: prior PASS 는 n=1 · flip 바닥 18.8~25%([[85]]) 안 — "회귀/우리가 깼다" 서술 금지. 이 sim 내 서술만 성립한다.
- **이 sim 내에서 우리층이 PASS 를 깬 경로는 부재**: freeze 호출/unlock 0회·freeze 를 죽인 deny 0건(반증조건 ii 기각) · 우리층은 오히려 freeze 이름+범위를 3회 병기 push(3/3 무전환·t123 대칭 작동). 갈림은 모델의 close-대체 판단이고 근거는 env 자료(doc_026)다.
- **user-sim 조건절 개입 — 실재하되 종결 귀속 아님**: 스크립트 분기 구조가 freeze-선행을 전제(step 4 전건 "After debit cards are frozen" 이 참이 된 적 없음 → green 구출 분기 영구 봉인)하고, 즉흥 동의 2건("Yes, close and replace all five" · "keep the Green Account replacement")이 gold-이탈 DB 를 재비준으로 잠갔다. [[21]]상 귀속은 model(freeze 명령이 msg1·49 문면에 존속).
- **n=1 로 말할 수 있는 것**: 이 태스크의 flip 감수성 원천은 «doc_026 의 close-권고 조항 × user-sim 의 freeze-전제 숨은 분기» 구조다 — 모델이 env-근거로 정당하게 선회할 수 있는 한 PASS 는 문자적 순종에 조건부다. 이는 [[85]] flip 바닥의 **개별 기전 실물 후보**이지 회귀 증거가 아니다. prior PASS 가 freeze 경로를 실제 탔는지(§1f-5 #16)가 이 서술의 다음 검증 항목.

##### 1f-10-5b. ⛔080 재분류 — «동전»이 아니라 «이름 붙는 이탈»이었다 (2026-09-04 밤 · 사용자 2연 지적 · C601)

위 1f-10-5 의 «모델이 env-근거로 정당하게 선회할 수 있는 한…» 서술은 **철회한다**. 축자 사슬:
정책 동사는 *recommend*(권고=발화 행위·실행 면허 아님) → msg48 권고는 준수 →
**msg49 손님이 freeze 재천명**(*"please freeze all three debit cards immediately"* · close 는
«교체 절차» 한정 동의 · 미철회) → **msg56 일방 재해석**(*"I'll go straight to closing …
rather than freezing"*) → **msg58 우리 레버가 갈림을 flag 했는데 모델이 자답**하고 통과 →
msg81 반전(미도난 판명·영구 close). **판별식 통과**: 갈림 조건 3요소(가역/비가역 형제쌍 ·
가역 쪽 축자 요청 실재 · 비가역 커밋 시도)가 전부 닫힌 술어 ⇒ B(원리락)가 아니라
**판별식-통과 후보**. 수리 = 탐지(기존재)가 아니라 **종단 정책**: flag 자답 금지 · 두 옵션 +
가역성 라벨 고객 회부. [[70]] 파는 것 = 정당 업그레이드 확인 턴 — 부호표·프로브(P11) 전 배선 금지.
★일반화(C601): flip 바닥은 미포착 기전 상계 — flip 실물마다 이 검사를 표준 산출로 한다.

**보강 (사용자 3연 지적 «권고 조항에 조건이 있을 것» — 축자 확인)**: 조항은 조건부가 맞다 —
*"If customer **confirms** the card is lost/stolen"* 이고, 근거도 문서 안에 있다(*"Freezing does
NOT affect ATM access if the customer has their PIN"* — 도난 확정 카드에 freeze 는 무방비).
같은 문서의 freeze 사유 목록이 정답 경로를 이름 붙여 둔다: *"Suspicious activity noticed, wants
to **investigate before closing**"*. CLOSE 의 실행 전건은 *"customer wants to cancel"* — 080 손님은
취소를 원한 적 없다. ⇒ 모델의 죄목 정밀화: ①미확정 «확인»의 확정 승격(3장 중 1장은 실제 비도난
— 시나리오의 시험 지점) ②권고→실행 면허 승격 ③미철회 지시 재해석. 셋 다 닫힌 검사 사정권
(①: confirm 발화 축자 실재 검사 — A2 선언 가능).

##### 1f-10-6. §1f-7b 계수 확정 (4건)

```
069 → B   원리 락 — 모델 값-선택(Gold 고정·순서 방기) + 클래스-수렴 반사실 봉쇄.
          ★[[68]] «표적 금지» 병기(노트↔KB 모순 · [[21]] 방어 불가) — 표적 수리 자체 금지
077 → B   원리 락 — EXTRA 축(자발 범위확장·정책상 정당)은 scaffold 처방 불가([[23]]/[[58]]).
          ⚠D8 수리(C 묶음)가 WRONGARG 축을 도울 수 있으나 EXTRA 축 잔존으로 flip 기대 낮음
080 → ⛔재분류(1f-10-5b·C601) — B 아님. **판별식-통과 후보**: 가역/비가역 대체의 고객-회부
          (레버 기존재·자답 허용이 결손 · P11 + [[70]] 부호표 전 배선 금지)
046 → B   원리 락 — 순서 역전은 모델 판단·반사실 양팔 무PASS. ⚠절차레버의 pay 확장-게이트는
          A①″ 후보(정당 지불 태스크를 판다 — [[70]] 부호표 실측 전 처방 금지)
```
4/4 B — «우리층 원인 0 · 옆에 선 우리층 결함(D8·핀-provenance·RECOMMEND-OFFER·발화점 공백)은 전부 비원인»이 이 배치의 한 줄이다. C-버그 수리(D8·D9·핀 등재·원장 대조)는 이 4건의 flip 을 약속하지 않는다 — 수리 근거는 [[25]](자기모순 금지)와 판정-가능성(D9) 확보다.

---

#### §1f-11 C601 census — flip 바닥의 기전 해부 전수 표 (x742 · 2026-09-04 · 반증 감사 통과분)

**전제 규율:** 전 행 n=1 짝([[85]]) — 회귀·인과 서술 없음. 채점단위는 각 행 reward_basis 선확인([[69]]①, 008·012 는 ACTION-basis). 모든 pin 은 격리·계기·측정 선행 조건부이며 [[81]](정본 런처 등재+첫 런 발화 확인까지 한 작업) 적용.

### ① flip 실물 전수 표 (14행)

| # | 실물 | 기전 (감사-확정 축자 기반) | 판별 (T3) | pin | 파는 것 [[70]] |
|---|---|---|---|---|---|
| 1 | **080** (견본) | 정책 동사 recommend=발화행위 · 손님 freeze 재천명 미철회 · 우리 flag 를 모델이 자답 | **닫힘** (가역/비가역 형제쌍 · 가역 축자요청 · 비가역 커밋 — 3요소 전수열거) | flag 의 고객-회부(자답 금지) — **P11 기배정**(x737:2389) | 정당 업그레이드 확인 턴 (부호표 짝=078 · 배선 금지 유지) |
| 2 | **037** | 유저심 프레임-날짜 누출("today (09/03/2026)") × **모순 인지 후 권위-중재 오판** — FAIL[38] reasoning 축자 "That's inconsistent. But the user is the source of truth … I'll proceed with the user's stated date." (감사 정정: 축자-복사 아님 — 알고도 진행, [36] 허위 자기보고 동반) | **닫힘** — P1 날짜형 스키마 ∧ P2 지시어-리터럴 공기(유한 어휘) ∧ P3 env 시계 존재 ∧ P4 해석≠리터럴(산수) | 지시어-날짜 정합 가드(provenance 존재-술어 `_ctx_has` 의 형제 정합-술어, t2_gate_patch ~L11880-11900). **감사 정정 반영:** 종단 = 1회 deny + 지시어-미포함 닫힌 재확인 → **사용자 최종답 수용**(env 날짜 강제기입 금지 — 권위는 'today' 해석에만) + (인자,값)쌍당 발화 1회 cap | regen 1턴 + 지시어가 정당 비-today 리터럴과 우연 공기하는 희귀 오발화(P2 인접-공기로 유계) |
| 3 | **101** | 결정캐리 argmax("It answers: Bluest" — referee 4인 미수집 상태) × [ORDER] regen 이 gold 전건 draft 2회 kill(t=31·41, `submit_referral` 축자 0회 draft 실측 2,119/2,687자) — [[62]] "측정 대상이 사라진다"의 실물 | **닫힘** — P1 settled row 의 referred-party 상수 유무(선언 유계) ∧ P2 draft 내 target 툴명 substring 0회(:10502 §7-2 술어 실재) | ① 캐리 가드: referee 값 0개면 "It answers" 대신 결손 표면화 — **전제조건(감사 보완): A2 party 태그 1키 순증**(현행은 desc 자유문 "the REFERRED party must…" — 영어 파싱 금지, [[72]] 완결 저작) ② [ORDER] regen 은 target 축자 지목 draft 만 kill | ① 참조인-단독 완결 sim 의 옳은 단언(선례 085#1·016·101 은 전부 해악 쪽 — 이득 칸 실측 의무) ② 조기-지시 억제력 일부 |
| 4 | **055** | 요건 미수집 배치-커밋(msg43→45 고객 턴 0) + 형제 무언 삭제 — **감사 정정: 무언급 substring 술어는 검산 실패**('Silver Plus Account' 가 FAIL 도구 출력에 10회) → 닫힌 재표현 = **형제 문서 read 0회**(FAIL 0 ↔ PASS 5문서 cat, 분리 성공) | **부분** — 닫힌 요소 = ①′ **최종 명명 턴→쓰기** 고객 수락 창(최초-명명 기준은 FAIL 도 충족 — 감사 정정) + ②′ 형제 read 0회. **열린 잔여(본문 승격): 올바른 클래스 도달** — 대본 §8 "single savings recommendation: Accept it" 이 오답 확인도 수락할 개연 | 레버 A confirm-before-open(정책 축자 general_002 step3 유계 · [[23]] 안전) + 레버 B′ 형제-read 결손 시 내부 read 표면화(고객-대면 메뉴 금지) | +1턴/개설 · 배치 개설 포기 · **§8 이 닫힌 요소의 효과를 «발화 기회 창»까지로 한정**(§6 재점화 미보장) · B′ 잘못 시공 시 pushback 루프 |
| 5 | **008** | tier 중재 미실행 — FAIL[52] summary 가 Tier-1 전건을 자인하며 Tier-2 enum 방출(가설-grep 자기확인 + 능력-결손 서사 축적, 규칙 "always select from the highest tier that applies" 는 31msg 상류 기수신) | **닫힘(조건부 — 감사 강등)**: SELECT=argmin_tier{applies} 에서 Codes·tier 순서는 닫힘, **applies 열거 완전성은 [[49]] 선행 박제상 미보장** — 커밋-시 형식화가 Tier-2 만 열거하면 min-tier 도 Tier-2 를 답한다 | enumerate-then-arbitrate: LLM 이 적용 코드 전수+문서 행 축자 인용([[66]] substring 검산) · 선택은 엔진 min-tier([[10]]) · 일반 계약형 "순서 규칙이 문서에 없으면 no-op"([[58]]) — **출시 게이트: 격리에서 Tier-1 열거율 측정 + [[57]] 부정통제 선행** | enum-커밋 토큰 비용 · 상위-tier 적용성 환각 시 과-격상(인용 검산으로 유계·잔여 존재) · 무문서 enum 이득 0 |
| 6 | **012** | 유저심 turn-1 페르소나 붕괴 — msg[1] "Hi Marcus — …" (유저심이 자기 캐릭터를 호격으로 부르며 어시스턴트로 발화) → gold 전건(손님의 전이 요청 발화행위) 구조적 도달 불능 · agent 층 결함 0(반사실 폐색) | **부분** — agent 층 **열림**(닫을 술어 없음 — reason 정의 축자 "customer then requests transfer") / sim 층 닫힌 부분집합 실재 | 하니스 층 sim-유효성 게이트([[05]] 고정분 불변): 자기캐릭터 호격 검출 — **감사 정정: 1인칭 도입 마커 {I'm, I am, this is, my name is} 비공기 조건 필수**(없으면 "Hi, I'm Marcus" 위양성) · 개시문 앵커=플래그 · **PASS·FAIL 대칭 적용** | 재롤 비용 · sim 저확률 행동 검열(대칭 적용이 조건 — fail-만 적용은 DV 선택편향) · 앵커 위음성 |
| 7 | **017** | grant id 전달-축자성 유실(B1: [29] 표시명 패러프레이즈만·리터럴 0회) × 결정론적 이름-불일치를 시스템 장애로 오진해 transfer 봉인(B2: [37]→[42] reason='technical_system_error') | **닫힘** — P1 grant 직후 발화에 X substring(런타임 grant 원장 유한·[[23]] 강화: KB 문서 축자 "Tell the customer to run the tool" 실재) ∧ P2 오류문면 "Unknown discoverable tool" ∧ Y∉granted ∧ ∃X∈granted | 레버 A grant-relay 축자 검증(mention_note 계보 역방향) + 게이트 B 이름불일치 transfer 거부 — **감사 정정: fix 는 상태-의존** «X 재지시 미발화면 '재지시하라' / 기발화면 '고객에게 X 로 재시도 청유(시스템 오류 아님)'» (미규정 시 W-5 모순-지시 루프) | A 오탐 0 기대이나 실측 의무 · B 진짜 도구 소실 시 transfer 1턴 지연 |
| 8 | **010** | duplicate 조항(platinum_008) 오적용 + 발명 규칙 "attribution before application"(**pass 에도 존재 — 분리 실패**) → 유저심 §7 전건 소멸 | **열림** — 조항-범위 관련성 판단 = [[59]] 금지 · §1f-7b 축자 "010(갈림=지목 관련성, 열린 ⇒ A군 잔류)"(x737:2495) | 부분 pin 만: ① D16 env 스니펫 동봉(**P10 기배정** · 격리 후 배선) ② [[64]] 무명계수 transfer-notice(미이행 도구명+doc_009 "Do not transfer" 표면화) | 지목 push 강화 → 오지목 29건(040 085 074 057 063 055) 되삼 |
| 9 | **027** | ⓐ EXTRA1: 정상경로 5회 live-DENY 된 write 를 searchexhaust 재생성 1발이 재검 없이 커밋(D14) ⓑ MISSING1: 도구 축자 "each needs a cash back dispute" 전수 나열에서 1건 자체 제외 | **닫힘** — ⓐ 재생성 산출도 wtag 6종 재진입(순수 엔진 술어·`:14432` 트리거 공유 박제) ⓑ 집합-차이(요구집합이 우리 도구 축자 양화 "each" — [[25]] 정본·[[23]] 안전) | ⓐ D14 수리 + **P9**(`regen_calls_reenter_write_gates` — 기배정) ⓑ 종결부 완결성 검사: (열거집합−제출집합)≠∅ 이면 미제출 txn 실명 상기([[64]] · deny 아님) | ⓐ만 고치면 MISSING 잔존(반사실 reward 불산) · 완결성 상기=push 레버라 같은 축 EXTRA 되삼+턴 비용 · gold 가 열거항목 의도-제외하는 자리 위양성 |
| 10 | **029** | 옳은 거절(msg83 "I can't update rewards based on the discrepancy list alone")을 넛지 3연발이 재생성 → 금지 write 5건(재실행 5/5 DENY) + MISSING1 | **부분** — EXTRA5 닫힘(D14 공용) · push 원조건 **열림**(§1f-7b ①′ Goodhart 축자: 닫힌 대리 ≠ «부당한 멈춤» — 정당 거절 구분 불가, x737:2487) · MISSING 조건부 닫힘(도구 문면 양화사 축자 미인용 — 궤적 재개봉 선행) | D14+P9 공용 · §1f-5 #5 넛지 개별 OFF 부정통제([[57]]) 선행 · 완결성 검사 027 공용 | push 사멸 시 048 t63·t123 의 **재생성이 만든 gold 칸**을 판다(§1f-1 부호표가 순해악 서술 정정) |
| 11 | **060** | D8 "None: None" 문면 재생성이 close 를 밀어 순서 파괴 — 7/7 action_match=True ∧ db_match=False(env "Account eligibility requirements not met") · PLAUSIBLE(D9 폐기 초안 소실) | **부분(조건부 닫힘)** — env 거부문이 전제조건 실재 자백·[[50]] 3조건 형이나, `close_bank_account` 가 `write_evidence_specs` 14키 밖 + [ORDER] 정책 문장 부재 자백(D13 · gate.json:4653) ⇒ env 축자 확보 전제로만 성립 · 인과는 D9 로 미확정 | ① D8 스키마 정합(`what`→`claim` — C군 기확정) ② D9 폐기 원문 원장(판정 전제·수리 1순위) ③ close 전제조건의 write_evidence_specs 등재(env 축자 출처 · 선언층 [[05]] 적법 · [[72]] 완결) | 전제 기충족 상태의 위양성 deny(턴 비용) + push 축 잔존(D14 수리 후에도) |
| 12 | **063** | account_class 값-고정(Gold) — READ-FIRST 4건 정상 회수로 원인 배제 · 별건 자격술어(최소잔액) 미검사 | **열림(B 원리락)** — 범주 선택=모델 값-판단([[13]]) · 닫힌 근사 기측정-실패: x509 축자 "표를 줘도 057·063 0/6" · D5 철회 완료 — 더하기로 안 닫힘([[63]]) | pin 없음 — 유일 닫힌 잔여 = **제거형** 도구측 자격술어(잔액≥KB 최소 미달 시 확인 거절+부족액 실명 [[64]])는 발현 n=1·미재현으로 §1f-5 #7 **측정 선행** | what-if 문의 거절 → W-5 재발화 루프 · 기측정 0/6 경로 재제안 금지([[40]]/[[74]]) |
| 13 | **067** | 자재-실재 — 카드 로스터가 서브 `ctx[-8:]` 창 밖(t2_resolve.py:1076) → `applies=false`×3 은 자재결손의 **옳은 답** → MISSING1 (007 정확 재현) | **부분(격리조건부)** — «선언 재료의 서브 도달» 술어는 닫힘([[71]] 선언 id→cat) · 갈림(재료↔판단 결손)은 **재격리가 판정**: 로스터+기준 주입 → Platinum 산출이면 닫힌 배선 결손([[78]]), 오답 지속이면 열림(B 락 · [[76]] 서브 수리) | 재격리 통과 시 = 선언 급양 배선(창 확장 아님). 승격 전 의무 인용: x509 ⑦ "x516·x517 둘 다 gold 0/39 ⇒ 경로 없음" + 차이 소명(모델 주입 아닌 **격리 서브 급양** — 소비부가 다름) · gate.json:271 DISCARD 선인용 | 컨텍스트/prefix 캐시 비용([[83]]) · 소명 실패 시 기실패 경로 재구매 |
| 14 | **068** | WRONGARG1 'Green Account (checking)' ↔ gold 'Green Account' — env-KB 함정(msg7 이 괄호형을 유효 예시로 못박음) · `T2_SIBLING_PAREN` 정확 탐지+수리값 지목했으나 **log-only 미무장**([[81]] · :13306-13316) | **닫힘(기판정 유지 · C군 «STRIP»)** — 괄호↔무괄호 형제쌍 = DB 실명 대조 문자열 술어 · 정본 §T-8(FAILURE_AXIS :2476-2586) 재유도 금지 | 결정론 괄호-STRIP **무장**(deny 금지 — W-5 축자 "반려를 받고도 같은 값 … 최다 18회") · 무장 게이트 = §T-8 반대 팔 A/B + 반려 후 괄호 제거율 부호표 — **실측 전 무장 금지** | gold 가 괄호 포함 공식명을 요구하는 자리의 손상 — env 스스로 괄호형 유효를 못박아 반대편 실재 |

### ② 집계 — «flip 바닥 중 압축 가능분» 첫 추정

- **닫힘 7** / 14: 080 · 037 · 101 · 008(조건부: applies 열거 완전성) · 017 · 027 · 068(조건부: §T-8 실측 게이트)
- **부분 5** / 14: 055 · 012 · 029 · 060 · 067 — 각각 닫힌 요소가 갈림의 **일부**(발화 기회 창 · sim 층 · EXTRA 축 · 순서 조건 · 급양 배선)만 산다
- **열림 2** / 14: 010(조항-관련성 = [[59]] 금지) · 063(범주 값-판단 = B 원리락 · 표 0/6 기측정)

**추정 진술(4칸, [[77]]):** ①주장 — flip 실물 14건 중 갈림 지점이 닫힌 술어로 지배되는 것 7건(하한), 부분의 닫힌 요소 포함 12건(상한): «판별식 미적용 기전 목록»이 바닥의 다수를 설명한다. ②근거 — 본 표 14행 각각의 축자 좌표(감사 전수 검산 통과). ③반증조건 — 어느 행이든 pin 격리/A/B 에서 닫힌 술어가 pass↔fail 를 분리하지 못하면 그 행은 열림으로 강등된다(055 ② 술어가 실제로 그렇게 강등·재규정됐다). ④선행확인 — x742 번들 6건 전건 재개봉 + x737 §1f-0~9 정독 + 엔진/gate.json/x509/FAILURE_AXIS 좌표 전수 실검(감사 로그). **단서: n=1/행 — 이 비율은 기전 census 이지 바닥 감소율 예측이 아니다. "18.8%×7/14" 류 산술 금지 — 각 pin 의 바닥 기여는 [[70]] A/B 부호표 후에만 확정.**

### ③ base 짝 3종이 말하는 것 (scaffold 0 분산의 원천)

세 짝 모두 fb/trace 0 — [[85]] 18.8% 바닥의 원표본이며, 분산 원천이 서로 다른 층에서 셋으로 해부됐다:

1. **008 = 모델 정보-취득·중재 스타일 분산** — cat-전문 vs 가설-grep 자기확인, 커밋 시 «상위 tier 선택» 중재의 실행/생략. 판별 정보는 두 팔이 다 받았다.
2. **017 = 모델 전달-축자성 샘플링 분산 × 오진 연쇄** — 같은 grant 컨텍스트에서 리터럴 포함/패러프레이즈가 갈리고, 그 결과인 결정론적 오류를 시스템 장애로 오진해 봉인.
3. **012 = 유저심 층 붕괴** — turn-1 페르소나 인스턴스화 실패. agent 는 양팔 정책-적합·반사실 폐색: **agent 측 어떤 pin 도 이 건을 못 산다**(하니스 sim-게이트만 산다).

⇒ 이 3건에서 «진짜 열린 값-판단이 갈림을 지배한» 자연상수 성분은 관측 0. 셋 다 명명 가능한 기전(2건 agent-측 닫힌 술어 · 1건 하니스-측 sim 술어)이다. 잔여 불확실: x599↔x644 서빙 조건(/v1/models 대조·[[30]] 포트 이어받기)이 미배제 — R2 선결 그대로.

### ④ 횡단 결손 — 해제(release) 술어 (감사 횡단 관찰의 승격)

**모든 pin 공통 전제:** deny/상기 발화 «이후» 상태의 닫힌 종단이 스케치에 없으면 W-5 형("반려를 받고도 같은 값 … 최다 18회 … 오답을 스텝 소진으로 바꾼다") 루프를 산다. 승격 전 각 가드에 다음 2요소 명기 의무 —
- (인자,값)쌍당 발화 **1회 cap**
- **사후 상태별 fix 분기**: 037 = 재확인 답 재충족 시 사용자 최종답 수용 / 017 = 재지시 기발화 시 "고객 재시도 청유" / 055 = 확인 후 §8 수락 시 종결(재deny 금지)

### ⑤ 프로브 번호 배정 (P11 이후) + [[70]] 부호표 짝

P9(027·029 regen 재진입)·P10(010 D16)·P11(080, 부호표 짝=078)은 기배정 — 재배정 금지. 신규:

| 프로브 | 실물·내용 | 선행 조건 | [[70]] 부호표 짝 (반대편 명시) |
|---|---|---|---|
| **P12** | 037 지시어-날짜 정합 가드 (P1∧P2∧P3∧P4 격리 → 배선) | 해제 술어(1회 cap+사용자 최종답 수용) 명기 | 발화 전수 × pass/fail 부호표 · 반대편 = 고객이 정당 과거 날짜를 지시어와 병기한 정상 발화의 오발화율 |
| **P13** | 101 캐리 referee-결손 가드 | **A2 party 태그 1키 순증이 전제**(desc 영어파싱 금지 · [[72]]) | T2_DECISION_CARRY 발화 전수 × settled-row referred-party 상수 유무 × pass/fail — 선례 3건(085#1·016·101) 전부 해악 쪽, 이득 칸(참조인-단독 완결 sim) 실측 |
| **P14** | 101 [ORDER] regen 자격 (target 축자 0회 draft 보존) | :10502 술어 재사용 확인 | kill 된 축자-0회 draft 전수 × pass/fail · 반대편 = 보존된 조기-지시 draft 가 만드는 EXTRA |
| **P15** | 055 confirm-before-open (최종 명명→쓰기 고객 턴 창) | §8 "Accept it" 오답-수락률 동시 실측(효과 상한 확정용) | 추천-유래 open 전수 × 고객 턴 0 여부 × pass/fail · 반대편 = +1턴·배치 개설 포기 |
| **P16** | 055 형제 문서 read 0회 표면화(내부 read 지시) | recommend_formalize 결정창 skip 배선 규명([[81]] 형) 선행 | 결정 턴 × 형제 read 결손 × pass/fail · 반대편 = 고객-대면 노출 오시공 시 pushback 루프 발생률 |
| **P17** | 008 enumerate-then-arbitrate (min-tier 결정론 선택기) | **격리에서 Tier-1 열거율 측정 + [[57]] 부정통제** 통과 후만 | 우선순위-문서화 enum 커밋 전수 × pass/fail · 반대편 = 과-격상(fraud 오적용) 발화율 |
| **P18** | 012 sim-유효성 게이트(하니스 층 — 호격 ∧ 1인칭 마커 비공기 · 앵커=플래그) | PASS·FAIL **대칭 적용** 설계 고정 | 전 런 turn-1 대칭 스캔 — pass 팔 히트 = 위양성 실측 · 반대편 = 정상 sim 재롤 비용·저확률 행동 검열 |
| **P19** | 017-A grant-relay 축자 검증 | 해제 술어(재지시 기이행 분기) 명기 | base 런 전수의 grant-후 발화 리터럴 누락률 census × pass/fail · 반대편 = 올바른 발화 오탐(0 기대·실측 의무) |
| **P20** | 017-B 이름불일치 transfer 게이트(상태-의존 fix) | P19 와 짝 등재 | 'Unknown discoverable tool' 후 transfer(technical_system_error) 전수 · 반대편 = 진짜 도구 소실 시 transfer 1턴 지연 |
| **P21** | 027/029 종결부 완결성 상기((열거−제출)≠∅ 실명 지목) | **029 도구 문면 양화사 축자 재확인(궤적 재개봉)** · §1f-5 #5 넛지 OFF 부정통제 | 해당 종결 전수 × pass/fail · 반대편 = push 축 EXTRA 되삼 + gold 의도-제외 자리 위양성 |
| **P22** | 060 close 전제조건 write_evidence_specs 등재 | **D8 스키마 정합 + D9 폐기 원문 원장이 판정 전제**(수리 1순위) | close 류 write 전수 · 반대편 = 전제 기충족 위양성 deny 턴 비용 |
| **P23** | 067 **재격리(판정 프로브 — pin 아님)**: 로스터+기준 선언 급양 → applies/correct 산출 | x509 ⑦ 축자 인용 + 차이 소명(격리 서브 급양 ≠ 모델 주입) 선기재 | 결과 자체가 판정: 통과=닫힌 배선 결손(그때 pin 승격) / 오답 지속=B 락 잔류 — B 명단 재배정의 유일 근거 |
| **P24** | 010 [[64]] 무명계수 transfer-notice(부분 pin) | P10(D16) 과 독립 등재 | transfer 직전 미이행 원장 표면화 × pass/fail · 반대편 = 지목 push 오지목 29건 축 |
| **P25** | 068 괄호-STRIP 무장 | **§T-8 반대 팔 A/B + 반려 후 괄호 제거율 실측 완료가 무장 전제**(정본 §T-8 — 재유도 금지) | 괄호형 WRONGARG 전수 · 반대편 = gold 가 괄호 공식명을 요구하는 자리 손상(env 가 괄호형 유효 못박음) |

063 은 프로브 미배정(열림 — §1f-5 #7 측정 선행이 유일 다음 수) · 010 본체도 미배정(열림 — 부분 pin P10·P24 만).

---

**감사 반영 요약(반증 반영분):** 037 기전 재서술(권위-중재 오판)+[[25]] 종단 기각→사용자 최종답 수용 · 055 술어② 재규정(무언급→형제 read 0회, 원 술어는 축자 검산 실패)+«닫힌 요소 지배» 강등(§8 위험 본문 승격)+PASS 낭독 위치 msg23 확정 · 008 «닫힘(열거 완전성 조건부)» 강등+격리 게이트 · 012 호격 술어에 1인칭 마커 배제 추가 · 017 게이트 B fix 상태-의존 재규정+[[23]] KB 축자 강화 · 101 party 태그 A2 1키를 pin 전제로 승격+좌표 정정(:2738·:2386) · 횡단: 해제 술어 2요소 전 가드 의무. [[23]]/[[58]] 위반 0(감사 확인). 전 행 n=1 — pin 의 reward 주장은 각 프로브 A/B 후에만.

---

### 1g. 기전별 base↔ours 트레이드오프 원장 — 도구별 per-step (2026-09-04 저녁)

> **왜 이 절인가** (사용자 지시 축자): *"트레이드오프와 사이드 이펙트를 per step 도구별로 정밀 비교
> 분석하라. base 대비 회귀한 10개 비교 포렌식한 분석을 포함해서 전체적으로 기전별로 다시 분석하라.
> 태스크별 분석보다. **트레이드오프와 사이드이펙트를 개선할 방법이 없는지가 중요하다.**"*
> 방법 = 짝 48태스크(base x644+x617 첫 sim ↔ ours 캠페인 첫 sim) · 같은 evaluator 셀 지도 446칸을
> **진단 지도**로([[69]] — 채점 아님) + 궤적 직접 계수. 부호검정은 태스크 짝 단위(셀 합산 아님).
> ✅검증 완료 (`wf_a466f436` · 5에이전트 · 130 도구호출): 분류기 36/37 재현 ·
> **재제출 귀속 반전 확정** · **0/N 블록은 채점기 artifact 로 반전** — 아래 각 절에 반영됨.

#### 1g-0. 원장 요약 — 무엇을 사고 무엇을 팔았나 ([[70]])

> ⛔⛔**명명 정정 (2026-09-04 사용자 지적: «트레이드오프가 아니라 버그 아닌가»)** — 맞다.
> 이 절이 «판 것»이라 부른 항목의 대부분은 [[70]] 의 ± 절충이 아니라 **이름 있는 버그**다:
> D14(재검사 목록 누락) · D8(필드 개명) · D12(오부착) · D6(리셋 조건 누락) · D3(거짓 문면 하드코딩) ·
> D7(정규화 누락) — 전부 **수리해도 레버가 사는 것을 팔지 않는다**.
> **내재 트레이드오프로 남는 것은 3종뿐**: ⑴ push 레버의 시점 판단 불가(061 — 술어·게이트 다
> 통과하는 «옳은 도구, 이른 시점») ⑵ deny 발화 자체의 턴·컨텍스트 비용 ⑶ cap fail-open(폭주 ↔ 커버리지).
>
> 따라서 이 원장의 올바른 독법: **산 층(선언·전달·격리)과 판 층(게이트·재생성)이 같은 레버가
> 아니다.** «하나를 사면 하나를 판다»의 절충 사례가 아니라, **한 층이 번 것을 다른 층의 버그가
> 결정 칸에서 까먹은** 구조 — 그것이 reward 순효과 0(부호검정 p=0.678)의 정체다.
> 매수의 실재는 flip 으로 설명 불가(24:2 · p<1e-4)이고 reward 실물도 있다(020·022 base 0 → ours 1.0).
> ⇒ 처방의 성격도 바뀐다: [[70]] 절충표가 필요한 것은 위 3종뿐이고, 나머지는 **그냥 고친다**
> (각자 프로브 + 발화∩PASS 위험군 스모크는 유지 — 버그 수리도 배선이므로 [[62]]/[[81]]).

```
기전 (짝 부호검정 n=48)                  ours↑  ours↓  동률     p     판정
DIFF 인자상이   (↓ 가 매수)                2     24    22   <1e-4   ★샀다 — 본체 †
OK   셀 정답                            24      6    18   0.0014  ★샀다
EXTRA 여분 비-read 호출 (↑ 가 매도)       37      5     6   <1e-4   ★팔았다
동일인자 재제출                           16      4    28   0.012   ⛔귀속 반전 — 아래 1g-3
MISS 미호출                              7      3    38   0.34    방향 나쁨·비유의 — 단 결정 칸 집중
transfer 탈출                            7      1    40   0.07    방향 나쁨
reads 검색량                            24     20     4   0.65    불변 (검색 부하는 안 팔았다)
```

†**검증 정정 (분류기 · PARTIAL)**: CALLED-DIFF 는 「값이 틀림」이 아니라 **strict 문자열 불일치**다 —
공백·float 직렬화(9.50→9.5) 위양성이 섞여 있다. ↑2 소수측의 073 은 그 위양성(parsed 동치 · DB pass).
**↓24 다수측 방향은 불변**. 기전 귀속 때는 `t2_forensic.args_equal`(parsed) 필터를 먼저 통과시킨다.
표본 37셀 재검산 36/37 일치 · 유일 불일치는 014 base transfer(빈 gold 필드 와일드카드 규칙 — 암묵이라 명문화 필요).

#### 1g-1. 산 것 (도구별 · base OK/DIFF/MISS → ours)

```
get_bank_account_transactions   8/11/0 -> 19/0/0     get_debit_cards      5/6/0 -> 11/0/0
get_all_user_accounts           9/ 9/0 -> 17/1/0     close_debit_card     2/7/0 ->  8/1/0
close_bank_account              0/ 5/0 ->  4/1/0     get_closure_reason   3/4/0 ->  6/1/0
file_credit_card_dispute        3/20/0 -> 12/11/0    order_debit_card     1/5/0 ->  4/2/0
give_discoverable_user_tool     0/ 7/0 ->  7/0/0     ← 전달 레버 완승
합계(assistant 셀): DIFF 132->46 (−86) · OK 247->312 (+65)
```
**출처 층** = A2 선언·scaffold GET·인자 유도(§5 층 선언의 표면화/선언 층). 아래 매도 기전의
수리는 전부 **다른 층**(게이트 술어·재생성 층)이라 **이 매수를 건드리지 않고 좁힐 수 있다** — 그것이
개선 경로의 존재 근거다. 부수 매수 2: **재생성이 만든 gold 칸**(048 t63 pay · t123 unlock — §1f-1).

#### 1g-2. 판 것 ① — MISS 미호출 +21셀: **억제 술어 4종이 결정 칸을 죽였다** (회귀 10 접힘)

MISS 증가 8태스크를 §1d 귀속과 접으면 **회귀 10건이 기전 4개로 접힌다**:

| 기전 | 태스크(회귀) | 죽인 도구 | 수리 (이미 설계됨) | 프로브 |
|---|---|---|---|---|
| **D12** feedback 오부착 | 014 | `transfer_to_human_agents` | 부착을 표적 호출로만 | — (문면) |
| **D6** 중복창 비리셋 | 051 | `approve_credit_limit` unlock+call | 상태 변화로 창 리셋 | P6d |
| **D4·D9** 동반차단(+원장 소실) | 079 | `freeze/unfreeze_debit_card` 4칸 | 의존 호출로 좁힘 · D9 선행 | P3c |
| **D11ⓑ/D14** pre-give 삼킴·재생성 | 015 · 049 | `get_referral_link` · `apply_statement_credit` | 양방향 술어 | P9 |
| (아래 분해 참조) | 007 038 054 055 059 064* | — | reward 귀속으론 수리 근거 없음 — **단 층이 다 다르다** | — |

⛔**«우리층 무관 = 모델 문제» 가 아니다 — 다섯의 층 분해** (2026-09-04 사용자 질문으로 정정):

```
007  user-sim/롤아웃   첫 분기 = m1 손님 개시발화 차이(이름 제시 여부) — 에이전트 입력이 같은
                      지점에서 갈렸다. 런-내 부정통제: 같은 deny 7 sim 중 pass 4
038  ⛔미판정          양보 발화는 모델 단독(우리 발화 0) — 단 그 무대 m43 은 [SIGNATURE]
                      캐스케이드 산물. "무대는 우리가 만들었고 대사는 모델이 썼다"
054  모델 판단+경로차   ours m33 에 사기 명령 2건 선행(base m30 은 CLI 단독) — 최강 대안설명.
                      지식 결손 아님("Yes, technically we could do the CLI first")
055  flip/시행 변동    같은 seed·gold·sha ours 통과 런 실재(x725 · 1/3) — 층 귀속 불성립
059  ⛔미판정          우리 재생성이 base 승수와 동일한 추천문 2772B 를 삭제 +
                      account_class options 오염 줄 BASE 0/OURS 1 — 간접 사슬 (a)(b)(c) 충족
```

그리고 이 다섯의 **별건 CONFIRMED 우리-층 결함은 수리 큐에 이미 있다**: 007→L2(구별 질문 탈출구
없음·서브 모순) · 038→D11(gold 규격 give 파괴)+[SIGNATURE] 강등(DB-basis 매수 0·16메시지 매도) ·
054→D8(`unbacked=0` 3회 발화) · 059→L2 오답(Gold Rewards 확언·[[25]] 위반).
«수리 대상 아님»은 **reward 귀속 축**에만 참이다.

**MISS 신규 4건(비회귀 태스크)의 검증 결과** — 층이 넷 다 다르다:

```
092  ★우리층 CONFIRMED — 라벨 자체가 오분류: 「미호출」이 아니라 「시도-차단-미재시도」다.
     모델이 gold 호출(reset_debit_card_pin unlock)을 실제로 시도 → [OPERATOR-SCOPE] deny
     + 형제 [BLOCKED] → unified_regen 이 그 턴을 다른 도구(close_debit_card)로 교체 →
     재시도 0회. deny 문면 축자 "call it again unchanged and it will proceed" 를 모델이
     안 따랐고 재생성이 방향을 틀었다 — D14 계열(재생성이 deny 후 경로를 바꿈)
061  우리층 PLAUSIBLE — uncalled_unlock 재생성이 산문 초안(1953B)을 즉시-해지 호출로 교체,
     수수료 대화 창(손님 대본 §9-10 의 deposit_check 발화 조건)을 건너뜀. 폐기 원문
     미보존(D9)이라 인과 확정 불가
010  우리층 발화 없음 — 조향은 env KB 문서("REJECTED: Do not retry immediately") ·
     T2_ACTIONREQ 는 의도적 침묵("근거 없는 지목은 하지 않는다")
049  우리층 발화 없음 + 오히려 turn 69·100 에서 빠진 도구를 2회 지목·촉구 —
     모델이 구두 제안만 하고($5 credit) 손님 거절 후 미호출
```

*064 는 D13 개시 CONFIRMED 지만 값 반사실은 PLAUSIBLE. ⇒ **회귀 10 중 우리-층 수리가 겨눌 수 있는
것은 4~5건이고 전부 「끄기」가 아니라 「술어 좁히기」다**([[19]]·[[60]] — 조정, not OFF).

#### 1g-3. 판 것 ② — EXTRA +56·재제출 +23: **범인은 거절문이 아니라 재생성이다** (★가설 반전)

동일인자 재제출의 귀속을 양팔 같은 잣대로 전수 계수(x750):

```
                우리deny후   env오류후    자발     합계
base                0        11        18      29
ours                2        17        33      52     ← 우리 거절문 후는 2/52 뿐
```
⇒ **W-5(반려-재제출 루프) 가설 반증 — 검증에서 한 겹 더 뒤집혔다** (REFUTED · 표본 정독 9건):

```
검증 재계수 (requestor 분해 · 96/96 재현)
  deny 후 재제출          ours 1/52 (1.9%)  ·  base 0/29
  assistant 쪽 dup        ours 24  ↔  base 24   = 동수 (우리 층 무관)
  ★잉여 +23 = 전부 user-sim 쪽 재호출 (ours 28 ↔ base 5)
     내역: 대본 반복 구간 도달(034·081 — ours 가 더 오래 살아 반복 구간에 더 깊이 들어감) ·
           give-전-실행지시 순서 실수 → env 오류 10발 → give 후 재실행 (022) ·
           한 메시지 일괄 동일인자 + env ID 충돌 재시도 (101)
  deny→재제출 실체 3건은 전부 읽기 도구([DUPLICATE-READ]·[GROUNDING]) — 지표(비읽기) 밖
```
⇒ 재제출 축은 **우리 층이 판 것이 아니다** — 절반은 «ours 가 더 오래 산 것»의 그림자다(생존 편향).
deny 문면 수리(D3·D7①)는 [[25]] 오염 제거로는 옳지만 재제출 처방이 아니다.
남는 관측 후보 1: **give-전-실행지시 순서**(022 형 · env 오류 10발) — 측정만, 수리 후보 아님.

**EXTRA 는 다르다 — requestor 로 갈라도 assistant 쪽이 본체다**(x752):
```
EXTRA(비-read·gold 초과)   assistant  base 97 -> ours 136   짝 ↑33 ↓6   p<1e-4
                           user       base  8 -> ours  25   짝 ↑9 ↓3    p=0.15   (비유의)
```

⛔**범주 정정 (사용자 지적: «EXTRA 가 왜 트레이드오프인가? 얻는 게 있나?») — 없다.**
EXTRA = gold **초과** 호출이므로 이득은 정의상 이 열에 못 찍힌다 — 같은 push 채널이 만든
gold 칸(048 t63·t123)은 OK 열에 찍힌다. 즉 트레이드오프인 것은 EXTRA 가 아니라 **push 기전**이고,
EXTRA 는 그 **손해쪽 계기**다. 내부이름 해제 재계산(x753)으로 손해의 순도도 갈랐다:
```
assistant EXTRA   읽기형(discoverable getter 류)  59 -> 83   17:4  p=0.007   DB 무해 · 컨텍스트 비용만
                  쓰기형(DB 를 건드림)             44 -> 67   32:5  p<1e-4   진짜 손해 +23
```
쓰기형 정독 표본(gold-외 write 7)의 분해 = **D14 버그 5 · D8 방아쇠 1 · 게이트 spec 공백 1**
⇒ 손해의 대부분도 버그 산출이다. **버그 수리 후 남는 쓰기형 잔여**만이 push 의 내재 비용이고,
그때 이득쪽 계기(gold-칸 창출 — 표본 2 · 코퍼스는 §1f-5 #4 미측정)와 [[70]] 절충표를 짠다.
지금 수치로 절충을 논하면 버그를 비용으로 계상하는 오류다.
그리고 그 assistant 쪽의 우리-층 채널이 **재생성**이다(x751 · GEN_TRACE 전수): «초안 tool_calls=0 → 재생성 >0» 전환
**96건/캠페인** (claimprov 47 · unified_regen 24 · followup_chain 12 · uncalled_unlock 8 · 기타 5).
pass 태스크 32 · fail 태스크 53. 6-sim 정독 표본(§1f-1)에서 그 전환의 산출은 **gold 밖 write 7 :
gold 칸 2**. 재생성이 만든 dup 은 x750 의 「자발」로 위장된다(거절문이 흐름에 안 남는 채널).
최다 태스크 = **049 (전환 14건)** — §1 의 원인 진술과 합류한다.

**⇒ 개선 경로 (매수 보존 조건 포함)**

| 매도 기전 | 수리 | 매수를 건드리나 | 게이트 |
|---|---|---|---|
| EXTRA(재생성발) | **D14/D11ⓑ — 재생성 산출이 원 게이트 체인을 재진입해야 커밋** | gold 칸 2 는 게이트를 정상 통과하므로 보존(예측·P9 부호표로 검증) | P9 + 스모크 「게이트 밖 커밋 0」 |
| EXTRA(재생성발) 의 방아쇠 | **D8 스키마 수리** — `"None: None"` 문면이 write 를 민 3건 제거 | pass 태스크의 전환 32건 중 유익분 — 코퍼스 gold 안/밖 census(§1f-5 #4) 후 판단 | P8a/b/c |
| MISS(억제발) | D12·D6·D4·D1 술어 좁히기 (1g-2 표) | 인자 정확도 층과 무관(층 분리) | 각자 프로브 + **발화∩PASS 위험군 스모크** |
| 재제출(env오류후 17) | **수리 없음** — user tool env 오류 재시도는 모델/env 층 ⇒ [[13]] 흡수 순서(scale/학습) | — | — |
| 재제출(자발 잔여) | 재생성분 제거 후 재계수 → 남으면 모델 층 | — | — |
| transfer 증가 | D8 수리의 2차 피해(user 롤 점유) 제거 후 재계수 | — | — |

#### 1g-4. ⛔«양팔 공통 0/N 블록» 은 레버 사정거리 밖이 아니었다 — **채점기 artifact** (검증 반전)

초판은 이 블록을 「모델/env 층 · 수리 불가」로 적었다. 검증(48셀 전수 재채점)이 뒤집었다:

```
본체 = 채점기 층. tasks.py:195 가 중첩 arguments JSON 을 **문자열째** 비교 —
  이미 정본 박제: test_args_equal.py "6런 gold_nested 1,222건 중 121건" (reward 무영향)
  gold 직렬화는 공백 있음 · user-sim 직렬화는 compact ⇒ 이 셀 가족(call_discoverable_*)은
  어느 팔이든 구조적으로 OK 가 될 수 없다
parsed 재채점 (submit_cash_back 48셀):  원본 0/24 vs 0/24  →  **base 15/24 vs ours 23/24 OK**
  ⇒ 블록 안에 ours 우세 8셀이 숨어 있었다 (020×4 · 022×4 — 두 태스크 reward 0→1)
"seed 종속 비교불가" 가설 = 반증 (같은 seed · txn id 축자 동일)
진짜 양팔 공통 실수 = **1셀뿐** (029_7 · 과지급 건 분쟁 억제 — 모델 층, 엔진은 정답을 전달했다)
```

⇒ 규칙: **셀 부호검정 분모에서 이 가족을 빼거나 `t2_forensic.args_equal` 로 재채점한 뒤 센다.**
초판의 「수리 기대수익 분모에서 제외」 지시는 **철회** — 제외가 아니라 재채점이 맞다.

#### 1g-5. 반증 조건

- 짝이 태스크당 n=1 이라 **기전 부호도 flip 의 영향을 받는다** — 단 24:2(DIFF)·37:5(EXTRA)는
  flip 25% 로 설명되지 않는다(이항 p<1e-4). MISS 7:3 은 flip 안이다 — 결정 칸 집중이 유일 근거.
- base 첫 sim 선택이 자의적이다 — x617 은 nt2 라 둘째 sim 으로 짝을 다시 짜면 표가 흔들릴 수 있다.
- CALLED-DIFF/NOTCALLED 분류기가 중첩 JSON 인자를 오독하면 도구별 표가 바뀐다. [검증중]
- x751 의 96건 전환 census 는 **이름 없는 계수**다(GEN_TRACE 에 호출명이 없다) — gold 안/밖
  비율은 6-sim 표본(7:2)의 외삽이고, 코퍼스 확정은 §1f-5 #4 를 D9 수리 후에만 할 수 있다.

---

#### 재현 산출물 (리모트 절대경로)

```
/home/woori/scratch/mechrun/census/   mech_census.py · mech_report.py · census.json · report.txt   (인구조사 정본)
/home/woori/scratch/mechrun/vq/       vq.py(축자 19패턴 전수 검산) · fx3.py(148 기전 Fisher) · fx4.py(★라벨 순열 300회)
                                      vq.out (축자 검산 원장)
/home/woori/scratch/mechrun/flip/     prefilter.py · extract.py · final.py · rewards.json · esha.json   (flip 바닥)
/home/woori/scratch/mechrun/cp/       claimprov V1~V4 (recon/main/step2~step7)
/home/woori/scratch/mechrun/declfail/ declfail 2×2 (grab/an/db/mut/fin)
/home/woori/scratch/mechrun/declfail2/ ★자연 실험 (s2·s4·s6·s7·s8) — OFF 팔 n=0 증명
/home/woori/scratch/mechrun/gpd/      grounding (gpd·gpd5 궤적권위·gpd2·gpd3·gpd6)
/home/woori/scratch/mechrun/dupref/   DUP-WRITE / REFERENCE (extract·report·report2·r4)
/home/woori/scratch/mechrun/blk/      BLOCKED 부수차단 (lit·main·mut·vict·mh·sub)
/home/woori/scratch/mechrun/rs/       readloop / 표면화 (x901~x910)
/home/woori/scratch/mechrun/atk/      ★반증 (p1 모집단구멍 · p4 자격분모 · p6 검출력 · p9 별칭 · p10 검열 · p11 난이도 · p12 MH)
```

**한 줄 판정**: 이 코퍼스에서 **가르는 기전은 하나도 확인되지 않았고, 회귀 10/42 는 flip 바닥과 구분되지 않는다.** 인구조사가 실제로 산출한 것은 기전의 인과가 아니라 ⑴ **판정 계약**(자격 분모 · 난이도 보정 · 코드 경로 정의 · 전역 순열)과 ⑵ **순위**(`unlock` 경로 · `FB_VIEW` · `LEDGER`)와 ⑶ **계기 결손 3건**(채점 40% 유실 · `dirty=True` 100% · `[T2_TRANSCRIBE]` IndexError)이다. 반복(n≥3)과 자격 분모 없이는 다음 런도 같은 자리에 선다.

---

## 2. 설계 — 수리 후보 전 명단 · 파생값·오선택은 측정만

> ⛔**F3 정합 교정 (2026-09-04)**: 이 제목은 한때 *"수리 후보 다섯(D1~D4 · D6) + 조사 하나(L1)"*
> 이었는데, 그 뒤 §1c-5 가 D7·D8·D9·L2 를, §1d-12 가 D11·D12·D13 을 추가해 **명단이 어긋났다**.
> 정본 명단은 **진입점 「등급」 블록 하나**이고, 아래 절들은 그 명단의 상세다.
> D6 은 §1e 에서 **강등**됐다(자격 1런 · `x548 --target 051` 필수) — 아래 D6 절을 그 전제로 읽어라.

### [[05]] 3질문 (설계서 상설 의무 · [[17]]) — **후보별 재작성** (B3 · 2026-09-04)

⛔**초안은 *"두 수리 모두 엔진 층 · A2 는 손대지 않는다"* 였고 지금은 거짓이다.** 후보가 다섯 +
조사 둘 + 계기 하나로 늘었고 **그중 셋이 선언 층에 닿는다.**

| 후보 | ①고정 | ②변경 | ③도메인-특화 섞이나 |
|---|---|---|---|
| **D1** | 엔진(절차 표면화) | **없음** — 술어만 바꾼다(`requires`·`tool`/`tool_any` 는 기존 필드) | 아니오 |
| **D2** | 엔진(readloop 발화) | 없음 · 단 **상수 K** 가 새로 생긴다 ⇒ gold 로 고르면 [[23]] 위반 | 아니오 |
| **D3** | 엔진(reference-filter) | 없음 | 아니오 |
| **D4** | 엔진(동반 차단) | 없음 — 의존은 기존 `arg_source_reads` | 아니오 |
| **D6** | 엔진(중복 억제) | ★**A2 변경** — `write_once_keys` 에 **3건 추가**(선행조건 1) | 값은 KB 축자에서만([[23]]) |
| **D7** | 엔진(grounding 검사) | 문면 수리는 없음 · 술어 완화도 엔진 | 아니오 |
| **D8** | 엔진(claimprov 발화) | 침묵 술어가 **선언 층에 닿는다**(무엇이 식별 가능인가) | 아니오 |
| **L1** | — (조사) | ★A2 `write_arg_enum` 을 **발화시키는 것** + `go_stack` 등재 | 아니오 |
| **D9** | 계기(원장) | 없음 | 아니오 |

⚠**A2 를 만지는 순간 [[24]] 양방향 의무가 붙는다** — 정본만 고치고 `gate.json` 을 동기화하지 않으면
등가 게이트가 FAIL 이다. §4 (5)가 그 게이트를 갖고 있으나, §2 가 *"A2 미변경"* 을 선언해 두면
**그 게이트를 건너뛸 유인**을 만든다. 그래서 이 표로 대체했다.
편집 후 `load_domain_a2` 확인 + `test_a2_three_layer.py` 를 돌린다([[24]]).

### D1 — 절차가 종결되면 잔여 노드 표면화를 멈춘다

> ⛔**B1 (2026-09-04 리뷰 · 실측으로 반영) — 아래 초안 술어는 외연이 거의 비어 있고 규칙 문면이 위험하다.**
> 정정은 이 절 끝 「B1 실측과 수리」에 있다. 초안은 지우지 않고 남긴다.

```
종결 노드(terminal) := 그 절차의 노드 중
                        (a) 다른 어떤 노드의 requires 에 등장하지 않고
                        (b) mutating write 인 노드
규칙 : 종결 노드의 도구가 성공 실행되면 그 절차 인스턴스를 closed 로 표시하고,
       이후 feedback.absent / absent_many 를 그 인스턴스에 대해 발화하지 않는다.
불변 : feedback.unmet (다른 도구의 선행 차단) 은 그대로 둔다 — 그건 표면화가 아니라 게이트다.
```

- **도출 가능**: (a)(b) 둘 다 A2 에 이미 있는 필드(`requires` · `mutates`)로 닫힌다. 새 선언 0.
- **x91 재발 없음**: `close.requires` 를 건드리지 않으므로 gold 경로를 다시 막지 않는다.
- **위험**: 종결 뒤에도 정당한 후속 단계가 있는 절차에서는 표면화가 사라진다.

#### B1 실측과 수리 — 술어를 A2 실물에 돌린 결과 (2026-09-04)

**주장 + 양화 (절차 n=6 · banking 전수)**: 초안 술어의 외연은 **1~3/6** 이고, 그마저 구현 세부
(`tool` 만 보나 `tool_any` 도 보나)에 따라 갈린다. 그리고 한 절차에서 **실행 가능한 terminal 이
둘**이라 규칙 문면이 gold 를 죽일 수 있다.

**근거 — 실측표** (`a2/banking_knowledge.specific.json` `procedures` + `a2/env_surface.json`)
```
절차                                노드   not-required 노드 (id · tool · mutates)
credit_limit_increase                7    decision · [tool_any] approve_credit_limit_increase_5847 · true
cash_back_dispute                    0    (없음)
credit_card_closure_retention        6    retention_offer · [tool_any] apply_statement_credit_8472 · true
                                          close           · [tool]     close_credit_card_account_7834 · true
incident_transfer_order              3    complete          · transfer_to_human_agents · **false**
decline_transfer_count               2    fourth_standard   · transfer_to_human_agents · **false**
credit_bureau_incident_escalation    2    complete_transfer · transfer_to_human_agents · **false**

(b)=mutating write   -> terminal **3**
(b)=노드가 tool/tool_any 를 가진다 -> terminal **6**
```
⛔**2차 리뷰 정정**: 초안에 있던 *"`tool` 만 인정 -> terminal 1"* 행은 **삭제한다. 존재하지 않는
구현 분기였다.** 엔진은 `t2_procedure.py:38 _tools_of` **하나**로 둘을 항상 동일 취급하고
(`t = node.get("tool"); if t: return [t]; return list(node.get("tool_any") or [])`),
`_satisfied`·`checklist`·`next_step`·`absent_note` 가 전부 그것을 쓴다. 근거는 *"구현 우연"* 이
아니라 **확정 사실**로 적는다.

**`tool_any` 전체 열거 (초안이 첫 항목만 적어 축약됐다)**
```
retention_offer.tool_any = apply_statement_credit_8472 · apply_credit_card_account_flag_6147
                           (둘 다 agent tool · mutating)
                         + apply_for_credit_card        (user tool)
decision.tool_any        = approve_credit_limit_increase_5847 · deny_credit_limit_increase_5848
A2 _note 축자: "세 도구 중 어느 것이든 하나면 충족"
```
⇒ **다중-terminal 트리거 면이 문서 서술의 3배다.**
전이(轉移) 절차 3개가 전부 탈락하는 이유는 축자 하나다 — `env_surface.json` 상
`transfer_to_human_agents.mutates = false`. 이 사실은 이 문서 §1c-3 이 088_16 에 대해 **스스로
확인해 둔 것**이다(*"`mutating_tools`(44종) 밖이라 DB 단위에 들어오지 않는다"*).

**수리 ①(술어)**: (b)를 **"노드가 tool(또는 tool_any)을 가진다"** 로 바꾼다. 종결의 의미는
*"DB 를 바꿨다"* 가 아니라 ***"이 절차에서 더 할 일이 없다"*** 이므로 술어도 그쪽이 맞다.
`tool` / `tool_any` 를 **둘 다** 인정한다고 명시한다 — 안 하면 외연이 구현 우연에 매달린다.

**수리 ②(문면 · 더 급하다)**: 지금 규칙은 *"종결 노드의 도구가 성공 실행되면 인스턴스를 closed"* 다.
`credit_card_closure_retention` 은 실행 가능한 terminal 이 **둘**(`retention_offer`·`close`)이라
**먼저 실행된 하나가 나머지를 덮는다**. `retention_offer` 가 먼저 나면 `close` 표면화가 사라진다.
⇒ **문면을 「실행 가능한 terminal 이 유일할 때만 closed」로 확정한다** (2차 리뷰).

⛔**두 대안은 등가가 아니다 — 초안이 등가로 나란히 뒀다.** *"모든 terminal 이 소진되면 closed"* 는
현행 침묵 조건(`t2_procedure.py:371` — `next`/`ready` 가 비면 `absent_note=None`)과 **거의 동치**라
순증이 ~0 이다. 유일한 차이는 **순서 위반 궤적**뿐이다(terminal 이 다 done 이어도 `_satisfied` 가
`requires` 를 안 보므로 `ready` 가 남는다 — 실측 반례 `['disputes']`).

**필수임이 실측으로 확정됐다**: `log_reason` 까지 실행된 상태를 엔진에 넣으면
**`ready: ['retention_offer','close']` 가 동시**로 나오고 `absent_many` 가 발화한다.
정책 출처는 `close` 의 `_note_requires` 축자 — *"Step 3~5 가 **조건부로 건너뛰어진다**"*.

**함의(기대치)**: D1 이 살 수 있는 태스크는 **`049` 계열로 한정**된다 — §1c-4 가 *"적용 폭이 049
계열로 좁다"* 라고 **추정만** 한 것을 이 표가 수치로 확정한다.

**반증 / refutation (2차 리뷰로 재설계)**
- 초안의 *"엔진이 `tool_any` 를 읽는다면 무효"* 는 **이미 해소**됐다(`_tools_of` · 위).
- 초안의 *"다른 도메인에 terminal 둘 이상인 절차가 없으면 수리 ②는 과잉"* 은 **폐기한다** —
  `a2/` 전수에서 `procedures` 를 가진 파일은 **banking 하나뿐**이라, 문자 그대로 적용하면
  **필수 수리를 폐기시킨다**. 재설계한 반증조건:
  ```
  R-B1a  log_reason 실행 상태를 엔진에 넣었을 때 ready 가 하나만 나오면 수리 ②는 불필요
         (실측은 반대다 — ['retention_offer','close'] 동시)
  R-B1b  "유일할 때만" 규칙이 049 계열에서 close 표면화를 지우면(즉 retention_offer 가 먼저 ready
         가 되어 유일성이 깨지면) 규칙이 과잉이다 — P1 에서 순서까지 재현해 확인한다
  R-B1c  다른 도메인에 procedures 가 생기면 그때 재검토한다(현재 표본 1)
  ```

**선행 확인**: `a2/banking_knowledge.specific.json` `procedures`(6) · `a2/env_surface.json`
`transfer_to_human_agents.mutates` · 이 문서 §1c-3(088_16 · `F.mutating_tools()` 44종) ·
§1c-4(D1 중립~약화).

⛔**이 표는 §4 스모크 (2)가 "눈으로 확인"하라고 미뤄 둔 바로 그 덤프다. 무료였고 설계 시점에
찍혔어야 했다**([[62]] 레버 전에 결손을 재라). §4 (2)는 이제 회귀 검사로만 남긴다.

### D2 — 읽기 루프에 이름과 출구를 준다 ([[64]])

현재 `_t2_fu_readloop_turn`(`t2_gate_patch.py:13759`)은 예비-예산 보호에만 쓰인다. 여기에 발화를
붙인다.

```
조건 : 같은 절차 인스턴스에서 readloop 턴이 연속 K회를 넘고 그동안 체크리스트 상태가 불변
발화 : (1) 무엇이 틀렸나 — 최근 K턴 동안 절차가 한 칸도 전진하지 않았다 (관측 사실만)
       (2) 무엇을 하면 풀리나 — 이미 있는 unlock_hint 축자를 이 문맥에도 붙인다:
           "Do not search the knowledge base for it: the name above is complete, and each
            search returns a large amount of text that will crowd out the conversation."
       (3) 남은 노드가 조건부(우회 가능)이면 그 사실을 서술형으로 알린다 (명령형 금지)
```

- **K 는 상수다. gold 로 고르지 마라([[23]]).** 격리(P2)에서 정하고 그 근거를 이 문서에 적는다.
- ⛔**사전 기대치를 낮춰 적는다([[63]] · 2026-09-04 리뷰 D1)**: D2 는 **문장을 하나 더 붙이는
  순수 "더하기" 레버**다. 그런데 정본 결과는 *"모델은 더하기·지시로는 안 닫히고 **제거만** 닫는다
  (0/8↔8/8)"* 다. D1 은 빼기라 [[63]] 정합이지만 **D2 는 정면으로 어긋난다.**
  ⇒ 우선순위를 **D7·D8 뒤로** 내리고, P2 의 exit(*"어떤 K 에서도 안 끊기면 폐기"*)를 그대로 두되
  기대 이득을 낮게 기록한다.
- (3)의 "조건부" 는 A2 의 `requires` 구조로 닫힌다 — 내용 해석 없음([[59]]).

---

### D3 — reference-filter 의 문면과 술어를 **일치**시킨다 (§1b-refute 로 축소된 주장)

**주장 + 양화 (n=8 칸 · sim 1개)**: `task_041` 의 `file_credit_card_transaction_dispute` 8 칸이
이 게이트에 막혔다. **창(view) 때문이 아니다** — 게이트는 `state.messages` 전사를 받는다.
⛔**F9 정합 — 양화를 정정한다 (2026-09-04 · 로컬 재현)**: 이 게이트에는 **sim 당 3회 상한**
(`_t2_reffilter < 3` · `:9395`, 증가 `:9404`)이 있고, **실측이 그 상한에 정확히 닿는다**:
```
trace 캠페인 24런 · "[T2_RESOLVE] deny reference-unmatched" 를 simtag 로 집계
    task_041#s626729   **3**        task_081#s626729   **3**      (그 외 sim 0)
fb 사이드카의 [REFERENCE] 행은 6 인데, 3 deny × 2 레코드(`tool-deny` + `route`)다 — 중복이다
turn 분포: 40 · 44 (tool-deny)  /  57 · 59 · 61 (route)
```
⇒ *"8 칸이 이 게이트에 막혔다"* 는 **틀렸다.** 막힌 것은 **3콜**이고, 나머지는 상한 소진 후
**fail-open 통과**했다. ★§1f 갱신: 상한 3 정확 소진 sim 이 **4개**가 됐다(041·081·039·084) —
4/4 에서 소진 후 최종 write 가 통과했다. 술어 결함(문면 거짓)은 강화 · reward 인과는 재약화. ①의 인과 가설은 그만큼 더 약해진다(§1b-refute 와 같은 방향).
반대로 ②(문면이 검사한 내용과 다르다)는 **상한과 무관하게** 산다 — 3번 다 거짓을 말했다.
P3b 는 「criteria 부합 수」와 함께 **상한 소진 시점**을 같이 찍어야 한다.

**근거 — 축자 + 파일:줄**
```
t2_gate_patch.py:9398   _rz_rf.resolve_reference_filter(am, state.messages, a2, self, la, UserMessage)
                                                            ^^^^^^^^^^^^^ 전사 (창 가설 반증됨)
t2_gate_patch.py:9395   and getattr(self, "_t2_reffilter", 0) < 3):     ← **sim 당 3회 상한**
t2_gate_patch.py:9404   self._t2_reffilter = getattr(self, "_t2_reffilter", 0) + 1
t2_resolve.py:1258-1264
  correct = _c.apply_op({"op":"filter","over":"records","return":keyf,"match":match, ...})
  if correct and str(correct) != str(chosen):
      return {"status":"deny", ...}
```
검사는 *"지목한 id 가 기록에 있는가"* 가 **아니라** *"내가 계산한 단 하나의 id 와 같은가"* 다.
`formalize_reference_criteria` 는 criteria **하나**를 뽑고 `apply_op(filter)` 는 id **하나**를
돌려준다. 그런데 041 의 손님은 거래 8건을 분쟁한다 ⇒ 8개 중 7개는 정의상 `chosen != correct`
가 되어 거부된다. ⚠단 §1b-refute 대로 **이것이 041 실패의 원인이라는 인과는 성립하지 않는다**
(같은 게이트 아래 dispute 6 칸이 통과했다). 확정된 결함은 아래 ②(문면 거짓)이고, ①은 P3b 가
criteria 부합 수를 세기 전까지 **가설**이다.

**두 번째 결함 — 거부 문면이 검사한 내용과 다르다** (`t2_gate_patch.py:9410` 하드코딩)
```
"[REFERENCE] the %s you named does not appear in any record returned by the tools in this
 conversation. Re-read the records you already fetched and name a %s that appears in one of them"
```
`t2_resolve.py` 가 만든 `REF_FILTER_FB` 를 **쓰지 않고** 이 문장으로 대체한다(2026-08-19 의
*"치환 폐기"* 결정의 부산물). 그래서 문면이 사실과 다르다 — 041 의 8개 id 는 모두 msg 17 의
도구 출력에 있었다. 모델은 이미 시킨 대로 하고 있었으므로 msg 64 에서 **같은 배치를 재발행**했고
또 막혔다. [[64]] 위반이다 — 억제가 이름을 대되 **그 이름이 틀린 처방**이다.

```
규칙 : deny 조건을 "계산한 하나와 다르다" 에서
       **"지목한 id 가 criteria 에 부합하는 레코드 집합에 없다"** 로 바꾼다
       (집합 소속 검사 — 여전히 닫힌 술어이고 옳은 값을 흘리지 않는다 · [[59]] · [[23]])
문면 : 검사한 것과 같은 말을 한다. 집합에 없을 때만 "기록에 없다" 고 말한다.
```

⚠ **[[70]] 무엇을 파는가 (2026-09-04 리뷰 D2 · 초안에 없던 칸)**: 술어를 *"계산한 하나와 일치"* →
*"집합 소속"* 으로 완화하면 **집합 안의 엉뚱한 레코드를 지목해도 통과한다.** 그리고 그것이 정확히
「오선택 14칸」 축의 실패형이다 — `041_4` GOLD `…_gold` ↔ OURS `…_crypto` · `041_5` 가 **서로 뒤바뀐
꼴**. 즉 D3 는 ②범주 실패를 게이트로 잡던 것을 **놓는 방향**이다.
⇒ 태스크별 부호표에 **"D3 완화로 통과한 오선택 건수"** 칸을 신설한다. 그 칸 없이는 판정 불가.

**반증 / refutation**: `apply_op` 가 실제로는 집합을 돌려주는데 호출부가 스칼라로 비교하는
것이라면 수리 위치가 다르다(`t2_compute` 쪽) ⇒ P3 에서 반환형을 먼저 찍는다. 그리고 041 의
8개 id 중 criteria 부합이 실제로 1개뿐이라면 이 귀속은 무너진다.

**선행 확인**: `grep -rn "does not appear in any record" scripts/distill/tau2/` →
`t2_gate_patch.py:2851 · :9410` · `grep -n "def resolve_reference_filter" -A 90 t2_resolve.py` ·
`a2/banking_knowledge.specific.json` 의 `reference_filter` 2 스펙(debit · credit).
⛔**[[74]] 보완 (2차 리뷰)**: `reports/facet_rft_2026/refute_2026_08_24/` 를 빠뜨렸다.
그 디렉터리는 **같은 계열 귀속을 이미 한 번 반증한 기록**이고, D4 쪽에는 형제-통과 반례가
`refute_055.json:31` 에 축자로 남아 있다. D3·D4 의 인과 주장은 그 반례를 **인용하고 무엇이
다른지 대야** 성립한다([[56]] 근거 확보한 쪽이 우세 · [[40]]).

### D4 — 턴 동반 차단(`[BLOCKED]`)을 **의존 호출로만** 좁힌다

**주장 + 양화 (n=130 deny · sim 20개)**: `tool-deny` 130 중 **65(50%)** 가 부수 차단이고,
`task_041` 은 한 턴에 21건(turn 44) · 17건(turn 40)이 함께 죽었다. 그 대상이 gold 요구 8 칸이다.

**근거 — 축자**
```
"Error: [BLOCKED] this call was not run because another call in the same turn was blocked:
 'call_discoverable_agent_tool(file_credit_card_transaction_dispute_4829)'"
```
```
규칙 : 플래그된 호출만 막고 나머지는 실행한다.
예외 : 막힌 호출의 **출력에 의존하는** 호출만 함께 막는다 — 그 의존은 A2 `arg_source_reads`
       로 이미 닫혀 있다(새 선언 0).
```
⚠ [[70]] 무엇을 파는가: 부분 실행은 한 턴의 일부만 반영된 상태를 만든다(원자성 상실). 그 대가를
**태스크별 부호표**로 세지 않으면 판정 불가다.

**반증 / refutation**: D3 를 고치면 원발 거부가 사라져 부수 차단도 함께 사라질 수 있다. 그러면
D4 는 불필요하다 ⇒ **D3 만 켠 팔을 먼저** 보고 D4 단독 효과를 판단한다.

**선행 확인**: `grep -rn "BLOCKED" t2_gate_patch.py` · 회수된
`fb_bank_g97151p11_viewmax2_20260903_1924.jsonl` 의 turn 별 deny 집계.

### D6 — `[DUPLICATE-WRITE]` 의 중복 창을 **상태 변화로 리셋**한다
> ⛔(2026-09-04 §1f) *"오늘 가장 확실한 우리-층 결함"* 표현을 **내린다** — 새 실패 6건에서 발화 **0/6**.
> §1e 강등(자격 1런)과 합쳐, D6 은 「확실」이 아니라 「표본 희소」다.

**주장 + 양화 (n=1 sim · gold 칸 2개 직접 사망)**: `task_051`(`bank_k8143med1_20260904_0135`)에서
gold 가 **같은 write 를 두 번** 요구하는데 우리 게이트가 두 번째를 막았고, 그 하류 2 칸이 죽었다.

**근거 — 축자 + 위치**
```
gold 051_2 {"agent_tool_name":"submit_credit_limit_increase_request_7392",
            "arguments":"{\"credit_card_account_id\":\"cc_5e4c1a83b0_bronze\",
                          \"user_id\":\"5e4c1a83b0\",\"requested_increase_amount\":1000}"}
gold 051_7 위와 **바이트 단위로 동일**            <- 같은 호출을 두 번 요구한다
궤적의 실제 submit 호출: msg23 **한 번뿐**

우리 게이트 축자 (task_051 turn 61·63):
  "[DUPLICATE-WRITE] This exact call (same tool, same arguments) already succeeded earlier in this
   conversation, so this call was REMOVED and not run ... It ran at message 23 and returned:
   Credit limit increase request submitted successfully ... Request ID: cli_e33db0778663 ...
   That change is already done. **Do NOT attempt this change again and do not do anything further
   about it.**"

에이전트가 손님에게 (msg65): "시스템이 방금 처리·거절된 동일 요청으로 인식해서 새로 제출하지 못하게
                              하고 있습니다"
손님 (msg66): "몇 분 기다릴게요. 시스템이 허용하면 $1,000 증액을 **다시 제출**해 주세요"
죽은 하류: 051_8 · 051_9 (approve_credit_limit_increase_5847) = match False
```
gold 흐름은 **신청 → 조회 → 거절 → 대금 완납 → 재신청 → 승인**이다. 완납이 상태를 바꿨으므로 두 번째
신청은 **다른 요청**인데, 우리 게이트는 "같은 도구·같은 인자"만 보고 동일하다고 판정한다.

### D6 의 술어 — 초안을 폐기하고 좁힌다 (2026-09-04 · 사용자 지적)

**초안(폐기)**: *"직전 동일 write 이후 다른 mutating 호출이 있었으면 중복이 아니다."*
⛔**너무 느슨하다 — 진짜 중복 실행까지 열어 준다.** 폐기한다.

**대안으로 검토했다가 기각한 것**: *"인자가 바뀐 것만 통과."* ⛔**051 을 못 산다** — gold `051_2` 와
`051_7` 은 **바이트 단위로 동일**하다(§위 축자). 인자 변화는 이 자리의 판별자가 아니다.

**실제 버그는 반대편에 있다 — 억제가 선언 없이 기본으로 걸린다.**
```
t2_gate_patch.py:6121   for k in (_mut_key_of(tc), _once_key_of(tc, a2w)):
t2_gate_patch.py:6042   _mut_key_of: "변이 하나의 동일성 = 실행 이름 + **인자 전체**(문자열 접기)"
t2_gate_patch.py:6052   _once_key_of: "A2 `write_once_keys` 가 선언한 **정책의 유일성 키** (없으면 None)"
A2 축자 `_note_write_once_keys`:
   "정책이 선언한 **유일성 키**. 엔진은 이 이름들의 값을 읽어 이어 붙일 뿐이고
    **무엇이 유일한지는 여기서만 정한다**([[05]])."
```
선언은 *"여기서만 정한다"* 인데, 엔진은 **선언과 무관하게 `_mut_key_of` 도 함께 등록**한다. 그 결과
*"어떤 write 도 같은 인자로 반복될 수 없다"* 는 **정책이 말한 적 없는 유일성 규칙**이 전역으로 걸린다.
`write_once_keys` 에는 현재 `apply_checking_account_credit` **한 건만** 선언돼 있는데,
`submit_credit_limit_increase_request` 는 선언이 **없는데도** 막혔다 — 051 이 그 실물이다.

```
규칙 : 중복 억제는 **`write_once_keys` 가 선언한 write 에만** 적용한다(`_once_key_of`).
       선언이 없는 write 에 `_mut_key_of` 로 억제하지 않는다.
```
- **진짜 중복은 그대로 막힌다**: `apply_checking_account_credit` 는 선언돼 있다 —
  도구 설명 축자 *"may only be called ONCE per checking account per customer interaction"*.
- **051 은 통과한다**: 그 도구엔 유일성 선언이 없다. 정책이 반복을 금지한 적이 없다.
- **[[05]] 정합**: 무엇이 유일한지는 **선언(A2)** 이 정하고 엔진은 집합 소속만 본다([[59]]).

⚠ [[70]] 무엇을 파는가: **아직 선언되지 않았지만 반복하면 해로운 write** 가 보호를 잃는다.
완화책은 창을 여는 게 아니라 **선언을 채우는 것**이다. ⇒ **P6a 로 실제 감사했다(아래).**

#### P6a 결과 (1차) — ⛔**아래 결론은 P6c 가 반증했다. 정정은 이 절 끝에 있다.**

**주장 + 양화 (WRITE 도구 n=42 · KB 문서 전수)**: banking 도메인의 write 도구 42개 중 KB 가
유일성을 말하는 것은 **`apply_checking_account_credit_5829` 하나뿐이고, 그것은 이미 선언돼 있다.**

**근거 — 축자 + 출처**
```
유일성 문면이 있는 KB 문서 5개 · 그중 write 유일성은 1건
  doc_bank_accounts_bank_accounts_(general)_017 축자:
    "The apply_checking_account_credit_5829 tool may only be called ONCE per checking account
     per customer interaction. After a credit is applied to a checking account, the system
     enforces a 14-day cooldown period before another credit can be applied to that same account."
  => A2 write_once_keys 에 **이미 선언됨**(keys=["agent_tool_name","account_id"])

같이 걸린 나머지 1건은 write 유일성이 **아니다** — read 로 확인하라는 선행 조건이다:
  doc_credit_cards_credit_card_account_logistics_007 축자:
    "Cooldown Period: Use the get_credit_limit_increase_history_4829 tool to check if the customer
     has submitted a request within the cooldown period for their card tier."
  => 이것은 **점검(read)** 지시이지 재제출 금지가 아니다. 051 이 정확히 이 경우다 —
     정책은 "확인하라"고 하고 우리는 "하지 마라"로 막았다.
```

**보호를 잃는 write 목록 (41개 · 정책 근거 없음)**: `submit_credit_limit_increase_request_7392` ·
`approve_credit_limit_increase_5847` · `pay_credit_card_from_checking_9182` · `open_bank_account_4821` ·
`close_bank_account_7392` · `order_debit_card_5739` · `freeze_debit_card_3892` · `close_debit_card_4721` ·
`file_credit_card_transaction_dispute_4829` · `file_debit_card_transaction_dispute_6281` ·
`log_verification` · `submit_referral` · `apply_for_credit_card` … (전 41개)
⇒ **정책이 반복을 금지한 적 없는 도구들**이다. 기본 억제는 이들에게 근거가 없다.

**반증 / refutation**: 내 검색어가 놓친 표현이 있으면 이 "0개" 는 거짓이 된다. 쓴 패턴:
`only be called ONCE` · `may only be called` · `only once` · `ONCE per` · `a second time/request` ·
`cannot be called/submitted/applied again|twice` · `cooldown period` · `one per` ·
`single request/submission per` · `duplicate request/submission`. 다른 표현이 나오면 **선언을 먼저
채우고** D6 를 켠다. (⚠파이썬 docstring 만 뒤진 1차 감사는 enum 값 `duplicate_charge` ·
`cooldown_period_active` 에 걸린 **오탐 3건**을 냈고 정작 017 문서를 놓쳤다 — 출처는 **KB 문서**다.)

**선행 확인**: `grep -rln "may only be called ONCE" tau2-bench/` →
`documents/doc_bank_accounts_bank_accounts_(general)_017.json` · A2 `write_once_keys`(1건) ·
`_note_write_once_keys` · `env_surface.json`(banking 엔 유일성 문면 없음 · retail 만 2건).

#### ⛔P6a 결론 철회 — P6c 가 반증했다 (2026-09-04 · 워크플로 `wf_63c350dd`)

> 위 *"채울 선언 0개 · D6 으로 잃는 보호는 없다"* 는 **거짓이다.** 지우지 않고 남겨 둔다 —
> 내가 쓴 반증 조건(*"검색어가 놓친 표현이 있으면 이 0개는 거짓"*)이 그대로 성립했다.

**주장 + 양화 (미선언 유일성 write n=3 · 노출 12 태스크)**: banking KB 에 유일성을 말하는 write
도구가 **최소 3개 더** 있고 전부 `write_once_keys` 에 없다. 선언된 1건의 노출이 3 태스크인데
미선언 3건의 노출은 **12 태스크**다.

**근거 — 축자 + 파일:줄**
```
request_temporary_debit_card_limit_increase_8374
  doc_bank_accounts_bank_accounts_(general)_040.json:13
    "- **Frequency**: Only one temporary increase per 24-hour period per card"
  tools.py:4124  "- Only one temporary increase is allowed per 24-hour period"
  env 자체 가드: **없음** (본문 3986-4185 에 재호출 차단 분기 0)      노출 1 태스크(089)

order_replacement_credit_card_7291
  doc_credit_cards_credit_card_replacements_004.json:2
    "- You cannot submit another replacement while an existing request is still being processed."
  env 가드 부분적 — tools.py:1468 "Error: Order may have already been placed ..."
    단 `reason` 만 바꾸면 우회                                        노출 8 태스크
      (036·037·038·039·054·077·080·081)

deposit_check_3847
  doc_bank_accounts_bank_accounts_(general)_011.json:40
    "- Duplicate deposit detected: Do not attempt to redeposit."
  env 가드 **0건** ⇒ `_mut_key_of` 가 **유일한 보호**였다               노출 3 태스크(055·057·061)
```
**내 1차 감사가 놓친 이유**: 정규식에 `only once` · `ONCE per` · `one per` 는 넣었으나
**`only one` · `cannot submit another` · `duplicate deposit`** 을 넣지 않았다.

**증거 등급의 역전(부수 발견)**: 유일하게 선언된 `apply_checking_account_credit` 의
`_note_source_` 는 출처를 *"도구 설명 축자(env 출력)"* 라 적었는데 `tools.py` 에
`ONCE per checking` 은 **0건**이고 그 축자는 KB `doc_..._017:48` 에만 있다. 반면 `_040` 건은
**KB + env 양쪽**에 있다 — 선언된 사례보다 근거가 하나 더 많은 write 가 미선언 상태다.

#### D6 은 공짜가 아니다 — 반사실 재현 (실 sim n=9,468 · 2026-08-29 이후)

```
총 deny   현행 309  →  D6 88        **221건(71.5%) 소멸**
  정당한 회복: submit_credit_limit_increase_request 24 (051 계열)
  그 외:      open_bank_account 45 · update_transaction_rewards 17 · order_debit_card 15 …
once-키만 잡는 것도 실재: apply_checking_account_credit 49건 중 **18건** (mut_hit=False)
retail 도 exchange_delivered_order_items 23 · modify_pending_order_items 12 가 once 전용
```

#### P6c 좁은 질문은 PASS (실측)

리모트에서 repo 모듈을 `importlib` 로 직접 불러 돌린 프로브(`/home/woori/scratch/p6crun/x_p6c_keys.py`
· 모듈 sha `0b7d703` · HEAD 와 `t2_gate_patch.py` diff 없음 · 모델 0 · GPU 0):
```
(1) 선언 케이스 A(amount 14.5) vs B(30.0) 의 once-key 동일?   True  -> 계속 막힌다
(2) 다른 계좌 C 의 once-key 는 다른가?                        True  -> 오차단 없음
(3) 미선언 write D1/D2 의 once-key 가 None 인가?              True  -> 051 통과
★ A.mut != B.mut  — 지금의 전체-인자 키는 "같은 계좌·다른 금액"을 **못 잡는다**.
  once-키만 잡는다 ⇒ D6 는 막던 것을 잃지 않는다.
```
관련 위치: `t2_gate_patch.py:6121`(등록) · `:12278`(조회) · `:12285-12286`(문면 분기) ·
`:6065` docstring *"선언이 없으면 None 을 돌려 종전 거동(인자 전체 키)을 그대로 둔다 = fail-open"*.
⛔`_mut_key_of` **함수 자체를 무력화하면 안 된다** — `:12101` 이 `T2_WRITE_ARG_TYPE` 의 sim-당 cap
키로 별개 사용 중이고, 무력화하면 2026-08-28 에 고친 t7376 task_040 회귀가 되돌아온다.

#### 반증에서 살아남지 못한 우려들 (= 기우였다)

중첩 JSON 파싱은 **된다**(실 궤적 `apply_checking_account_credit` 호출 325/325 가 `account_id`
담은 키 생성 · `:6080-6081`) · `_a2_of` 도달함(`unlock_`/`give_` deny 0건 · `:3637-3648`) ·
등록/조회 접두가 달라 충돌 불가 · **레버는 라이브에 켜져 있다**(`/proc/<pid>/environ` 에
`T2_DUP_WRITE=1`; 정본 `go_stack.sh:695` 는 `0` 인데 `run_ours_task.sh:128` 이 덮어쓴다).

#### ⛔C1 (2026-09-04 리뷰) — P6c 의 사정거리를 좁힌다

P6c 가 **실제로 보인 것**: 선언 케이스 A(14.5) vs B(30.0) 에서 `A.mut != B.mut` 이고 once-키는
같다 ⇒ **선언된 write 안에서는** once-키가 mut-키를 완전히 대체한다. 여기까지가 참이다.

그런데 초안은 이 문장을 **D6 전체의 판정 근거**로 썼다. 같은 절 두 곳이 그것을 스스로 반박한다:
```
deposit_check_3847 — "env 가드 0건 ⇒ _mut_key_of 가 **유일한 보호**였다"
반사실 재현        — deny 309 → 88 (**221건 소멸**)
```
미선언 write 에서 `_mut_key_of` 는 **바이트 동일 반복을 막던 유일한 장치**이고 D6 는 그것을 통째로
뗀다. 두 진술은 화해되지 않은 채 나란히 있었다.
⇒ **"막던 것을 잃지 않는다" 를 "선언된 write 에서는 잃지 않는다" 로 좁힌다.**

#### ⛔⛔C2 **철회·재작성** (2026-09-04 2차 리뷰 · 리뷰어 자기정정)

> **아래 초안은 틀렸다. "지금·무료·정본" 세 낱말이 다 성립하지 않는다.** 지우지 않고 남긴다.
> **게이트 지위는 유지하되 조건을 바꾼다.**

| 초안의 전제 | 왜 틀렸나 (코드로 확인) |
|---|---|
| *"`mutation_diff` 로 ①051형을 가른다"* | `t2_forensic.py:1183-1186` 이 gold 를 `mut_key` 로 **중복 제거**한다(`if k in seen: continue`) ⇒ ***"gold 가 같은 행을 두 번 요구"* 를 표현할 칸이 코드에 없다** |
| *"221건의 (도구,인자)를 이미 갖고 있다"* | **deny 된 호출은 궤적에서 제거되어 `attempted_mutations` 에 없다** ⇒ `mutation_diff` 는 221건을 **애초에 못 본다**. 사이드카도 `target`+deny 본문만 담고 **인자를 안 남긴다**(키 = `channel,kind,len,sha,sim,simtag,text,turn`) ⇒ once-키 복원 불가 |
| *"②088형 = EXTRA"* | **정의상 불가**. dup 재실행은 앞선 성공과 **같은 key** 라 `EXTRA`(name not in gnames)가 될 수 없다 |
| *"EXTRA 1건 → db_match 붕괴"* | **순환**. §1c-3 이 그것을 **[미판정]** 으로 남겼는데 전제로 승격했다. env `tools.py`/`db_check` 계산부가 로컬에 없어 **멱등 중복 판정도 불가** |
| 반사실 산출물 `x707.py` | **repo 에 없다** |
| 선행조건 2(airline) | `mutation_diff` 기본 변이집합이 **banking 전용** ⇒ 그대로 돌리면 **조용한 축소 채점** |

**⇒ 재작성된 게이트**: 부호는 **정적 재분류로 서지 않는다.** `D9`(원장) 배선으로 **deny 된 호출의
인자를 보존한 뒤**, `D6` 를 켠 **격리 재실행**으로만 선다 — deny 가 **궤적 자체를 바꾸기** 때문이다
(051 의 msg65~70 이 deny 산물임을 반증 에이전트가 실측했다).
⇒ **`C2` 는 이제 `D9` 의 하류다.**

**그리고 문면 수리로는 안 된다는 근거가 이미 있다**: `x548_dup_deny_iso.py` 의 **051 케이스 빌더가
2026-08-26 부터 실재**하고, 그 주석이 *"탈출 단서를 붙인 판도 열지 못했다(0/4)"* 를 박제해 뒀다
⇒ **D6 만 이 자리를 산다.**

---

#### (초안 · 철회됨) C2 — 221건의 가격은 지금 무료로 매길 수 있다

초안은 선행조건 3(태스크별 부호표)을 **런 이후**에 걸어 두었다. 그럴 필요가 없다:
```
채점 단위가 DB 해시다([[69]]). 어떤 중복 write 든 실행되면 EXTRA 이고,
088 이 EXTRA 1건으로 db_match=False 가 되는 모습을 이 문서가 이미 보였다(§1c-3).
반사실 재현은 각 deny 의 (도구, 인자)를 **이미 갖고 있다**.
정본 t2_forensic.mutation_diff 로 각 건을 결정론으로 가른다:
   ① 051형 — gold 가 같은 행을 두 번 요구한다        => 정당한 회복(이득)
   ② 088형 — gold 에 없는 변이가 하나 더 생긴다      => EXTRA(손실)
모델 0 · GPU 0 · 무료.
```
`open_bank_account` **45** · `update_transaction_rewards` **17** · `order_debit_card` **15** 가
①이면 큰 이득이고 ②면 **그대로 실패 45건**이다. 부호를 모르는 채 켜면 [[70]] 판정 의무 3종 중
*"무엇을 팔았나"* 를 **런 값으로 사후 추정**하게 된다.
⇒ **선행조건 3 을 "배선 전 게이트" 로 승격한다.**

⛔**논거 교정**: *"KB 에 유일성 문면이 없다"* 는 ***"반복이 무해하다"*** 가 아니다. DB 해시 채점에서
**정책의 침묵은 보호를 정당화하지도, 해제를 정당화하지도 않는다.** D6 의 정당화는 침묵이 아니라
**051 의 gold 가 동일 호출을 두 번 요구한다는 축자**에서 온다.

#### ⛔D6-op (2026-09-04 리뷰) — **A(대조)가 정의되지 않았다**

```
scripts/distill/tau2/go_stack.sh:695       export T2_DUP_WRITE=0    <- 정본은 OFF
scripts/distill/tau2/run_ours_task.sh:128  export T2_RULE_AT_WRITE=1 T2_DUP_WRITE=1   <- 태스크 런처가 켠다
run_ours_task.sh:14 축자: "⛔[[19]]/[[60]] 레버는 전부 켠다 — go_stack.sh 를 source 하고
                          여기서는 **실험 축만** 얹는다"
```
그런데 얹고 있는 것은 실험 축이 아니라 **레버**다. §3 「배선 조건」이 *"go_stack 등재까지가 한 작업"*
이라면, **D6 를 만지기 전에 이 레버의 정본이 어느 파일인지부터 정해야** `A(대조) = 현행 sha` 가
무엇인지 정의된다. ([[81]] 의 **거울상** — 정본에선 꺼져 있는데 라이브에선 켜져 있다.)

#### ⇒ 판정: **PASS-with-precondition**

```
선행조건 1  미선언 3건을 write_once_keys 에 추가한다 (키 선정은 KB/env 축자로 따로 정당화 · [[23]])
            우선순위: deposit_check_3847 (env 가드 0)
선행조건 2  airline 은 선언 **0건**이다. run_t7390_airline.sh:58 이 T2_DUP_WRITE=1 로 돌리므로
            D6 이후 그 도메인에서 이 레버는 **전면 무발화**가 된다.
            airline KB 에 유일성 문면이 있는지 — **모른다. 확인 안 했다.**
선행조건 3  [[70]] 부호표: 221건 손실 중 무엇이 정당한 회복이고 무엇이 보호 상실인지 태스크별로.
```

**반증 / refutation**: 위 3건 외에 또 다른 표현이 나오면 이 목록도 여전히 불완전하다.
airline·telecom KB 를 같은 방식으로 훑기 전에는 *"banking 만 3건"* 이라고 말할 수 없다.

**선행 확인**: 워크플로 `wf_63c350dd` 저널 · `x901_census.py`/`x902_dump.py`/`x903_writexkb.py`/
`x707.py`(반사실) · `grep -rn "T2_DUP_WRITE" scripts/distill/tau2/`(go_stack.sh:695 · run_ours_task.sh:128
· run_t7389.sh:61 · run_t7390_airline.sh:58 · run_t7391_retail.sh:52).

**반증 / refutation**: gold `051_7` 이 `051_2` 와 인자가 달랐다면 이 귀속은 무너진다 —
**동일하다**(위 축자). 재신청이 `DUPLICATE-WRITE` 아닌 다른 이유로 막혔다면 무너진다 —
deny 문면이 그 이름을 달고 있다. 그리고 D6 를 켠 격리에서 재신청이 통과해도 **승인까지 가지 못하면**
이 태스크는 여전히 안 산다.

**선행 확인**: `grep -rn "DUPLICATE-WRITE" scripts/distill/tau2/` · A2 `write_once_keys` ·
`_note_write_once_keys` · 회수된 `fb_bank_k8143med1_20260904_0135.jsonl` turn 61·63 ·
해당 sim 의 `messages` msg20·23·57·59·60·65·66.

---

### L1 — **꺼진 레버 조사** (D5 를 철회하고 이것으로 대체한다 · 2026-09-04)

> ⛔**D5 는 재발명이었다. 철회한다.** 아래 원문은 근거로 남겨 두되 **새 레버로 올리지 마라.**

**주장 + 양화 (n=3 런 · 발화 0)**: 내가 D5 로 제안한 열거값 검사는 **이미 존재한다**. A2
`write_arg_enum` 에 **9 개 선언**이 있고 그 **0번이 `open_bank_account.account_class`** 다.
그런데 이번 캠페인 3개 런에서 **발화 0회**다.

**근거 — 축자 + 파일:줄**
```
A2      banking_knowledge.specific.json  "write_arg_enum" (9 항목)
        [0] applies_to=call_discoverable_agent_tool · applies_when.prefix=open_bank_account
            arg=account_class · group_arg=account_type · group_map={...}
        [3] prefix=file_credit_card_transaction_dispute
            booleans=["contacted_merchant","eligible_for_provisional_credit"]   <- 파생값 10칸의 그 인자
엔진    t2_gate_patch.py:12004  _ens = (a2 or {}).get("write_arg_enum") or []
        t2_gate_patch.py:12005  if os.environ.get("T2_WRITE_ARG_ENUM") == "1" and _ens:
스위치  go_stack.sh 에 T2_WRITE_ARG_ENUM  **없음**
라이브  bank_k8141med1 · bank_g97151p11 · bank_re151med1  발화 각 **0**
과거 런 축자 (CAUSE_STEP_FORENSIC_RAW_2026_08_23.json:188):
  "[sim=task_055#s363271] [T2_WRITE_ARG_ENUM] deny val='Beige Savings Account'
   group=savings_accounts (후보 9)"
```
⇒ **레버는 있고, 예전엔 발화했고, 지금은 꺼져 있다**([[81]]). 할 일은 새 게이트를 짓는 것이 아니라
*"언제·왜 꺼졌나, 켜면 무엇이 달라지나"* 를 재는 것이다.

**⚠선행이 이미 경고한다 — 그냥 켜지 마라.** `refute_2026_08_23/refute_1.json` 축자:
*"⑵`T2_WRITE_ARG_ENUM_CAP` fail-open. 단 [[70]] 판정 의무 3종이 아직 안 채워졌다(레버 ON/OFF
reward 짝 없음·태스크별 부호표 없음) … **격리 프로브 없이 손대지 말 것**([[62]] ②③)."*
같은 문서가 이미 셋을 박제해 뒀다: ⓐgold 값 오거부(2026-08-13 FIX-6 로 수리됨) ⓑ**CAP(기본 3)
소진 후 fail-open** ⓒdeny 본문이 **영속 궤적에 안 남아** `messages` 만 보는 포렌식엔 안 보임.

**D7-op 실측 (2026-09-04 리뷰 · 전제 확인됨)**: `grep -rn "T2_WRITE_ARG_ENUM" scripts/distill/tau2/`
→ `run_one.sh:54` · `run_night_ab.sh:63` · `run_lever_*` 다수에 **있고** `go_stack.sh` 에는 **없다**.
⇒ [[81]] 배선 회귀 **CONFIRMED** 로 유지한다.

**반증 / refutation**: `T2_WRITE_ARG_ENUM=1` 로 켠 팔에서 059·066·071 의 값이 그대로 통과하면
이 레버는 그 칸들을 사지 못한다 ⇒ L1 폐기. 그리고 CAP 3 이 sim 당 소진되면 fail-open 이 되어
**켠 것과 안 켠 것이 같아진다** — 그 경우도 폐기다.

**선행 확인**: `grep -rn "T2_WRITE_ARG_ENUM" scripts/distill/tau2/` · `go_stack.sh`(부재 확인) ·
`reports/facet_rft_2026/CAUSE_STEP_FORENSIC_RAW_2026_08_23.json`(:188 · :251 · :271) ·
`reports/facet_rft_2026/refute_2026_08_23/refute_1.json`(:7 · :31 · :55) ·
`reports/facet_rft_2026/lever_consolidation_map_2026_08_19.json`(:1661 · :1667).

---

### P5 — 파생값 17칸은 **수리가 아니라 측정**이다

> ★1f-7 신 sim: **086** — 4× `customer_max_liability_amount` 0↔50 (085 쌍). 정정 포함:
> *"우리층이 안 닿았다"* 가 아니라 `write_rules`(specific:10318 · 50/500/-1 만 방출 가능 — 모델의
> '0' 은 못 낸다)·`distinct_args`(:10344)가 **실재하되 write-point 전달 레버(T2_RULE_AT_WRITE 류·
> 기본 OFF·gate_patch:12442)가 미발화**([[81]]). 등급은 우리층무관 유지(의도적 측정 자리).

**주장 + 양화 (n=17 칸 · sim 5개)**: 값만 틀린 34 칸 중 **17 칸**이 정책 파생값이다.
```
eligible_for_provisional_credit  10칸 (041×8 · 040×2)  GOLD False ↔ OURS True   (전부 한 방향·과다 인정)
customer_max_liability_amount     3칸 (085)            GOLD 50    ↔ 100.0 · 89.99 · 14.99
new_rewards_earned                2칸 (026)            GOLD 1020 · 1500 ↔ 6300
provisional_credit_eligible       1칸 (085_7)          GOLD True  ↔ False        (반대 방향)
expedited_shipping                1칸 (038_4)          GOLD True  ↔ False
=> 12/17 이 불리언, 그중 10칸이 같은 인자를 같은 방향으로 틀린다.
```

**⛔ 이 자리는 일부러 비워 둔 자리다 — 계산 레버를 되살리면 실험이 무효다.**
`a2/banking_knowledge.specific.json` 의 `compute_ops` 는 `{}` 이고 옆의
`_note_compute_ops_removed_2026_08_19` 축자:

> *"REMOVED (user decision 2026-08-19, plan A). Two ops were deleted because **the engine was
> producing values that the benchmark scores as gold arguments, which erases the very deficit we
> measure** ([[62]]), and because one constant was fitted to gold ([[23]]). (1)
> `file_debit_card_transaction_dispute.customer_max_liability_amount` used thr=30 days while the
> policy text says 'within 2 business days of statement'; the threshold was chosen by **gold
> reproduction rate** (T1=2 → 73.6% vs T1=30 → 89.4%) … Live evidence in run
> `bank_t7326_*_20260819q`: `'[T2_RESOLVE] compute silent-repair customer_max_liability_amount
> -1->50' fired 8 times in **task_085**."*

⇒ **085 의 그 3칸은 예전에 엔진이 채워 주던 바로 그 칸이다.** 다시 계산하면 [[23]]·[[62]] 위반이다.

**경계는 같은 노트가 그어 뒀다** — 축자:
> *"The policy tables themselves stay legal as **DELIVERED TEXT** (surface the doc_036/_031
> wording to the model); **what is forbidden is the engine writing the value into the call**."*

정책 조건 자체는 KB 에 있다: 책임 상한 `doc_036/_031` *"within 2 business days of statement→$50 /
within 60 days→$500 / after→전액"* · 구조 `min(disputed_amount, tier_cap)`; 임시 크레딧 `doc_032`
`ALL{timely ≤ 60일, category ∈ 5종, written_statement, account OPEN}`. 085 의 OURS(100.0·89.99·
14.99)는 **거래 금액 자체**로 보이고(티어 표 미적용), 041 의 10칸은 `ALL{}` 을 **평가하지 않고
True 로 넘긴** 모습이다.

### R1 정밀 분석 — **gold 없이 닫히는가? 부분적으로 그렇다** (2026-09-04)

*"파생값은 gold 를 볼 수밖에 없나"* 에 대한 답. **아니다.** 규칙은 KB 에 있다. 2026-08-19 의 위반은
**규칙을 몰라서가 아니라 상수를 gold 재현율로 고른 것**이었다. 파라미터별로 가른다 (n=17 칸).

```
[A] 규칙이 KB 축자에 있다 — gold 불필요
  eligible_for_provisional_credit (10칸)
    KB 축자 doc_credit_cards_credit_cards_(general)_015:
      "Previous Disputes: The customer has not filed more than 2 disputes in the past 12 months"
      (이 축자는 A2 `_note` 에 **이미 인용돼 있다** — 새 사실 0)
    => 규칙은 "직전 12개월 분쟁 2건 이하" + 카테고리 적격 + 60일 이내.
       분쟁 이력은 get_user_dispute_history_7291 로 **관측된다**. 계좌 OPEN 여부도 관측된다.
  customer_max_liability_amount (3칸)
    KB 축자 doc_036/_031: "within 2 business days of statement -> $50 / within 60 days -> $500 /
                           after -> 전액", 구조 min(disputed_amount, tier_cap)

[B] 관측이 아니라 **손님 발화 해석**이 정한다 — LLM 몫([[52]])
  8건 중 어느 3건에 임시 크레딧을 줄 것인가. 041 의 손님은 "가장 큰 금액들"이라고 말한다.
  => 순위·선택은 해석이다. 엔진이 고르면 [[62]] 위반.

[C] 상수가 정책 문면과 어긋난다 — **여기가 2026-08-19 의 죄**
  구판은 thr=30일을 썼는데 정책 축자는 "within 2 business days".
  그 30일은 **gold 재현율로 선택**됐다(T1=2 -> 73.6% vs T1=30 -> 89.4%) => [[23]] 위반.
  ⇒ 상수를 다시 고를 때 gold 를 보면 같은 죄를 반복한다. **정책 축자 외의 출처 금지.**
```

⇒ **결론: gold 는 필요 없다. 필요한 것은 ⓐKB 축자를 전달하고 ⓑ선택은 모델에게 남기는 것**이다.
같은 노트가 이미 그 경계를 그어 뒀다 — *"policy tables stay legal as DELIVERED TEXT … forbidden is
the engine writing the value into the call."*

⚠**주의 — 이 규칙을 `tasks.json` 에서 읽지 마라.** 벤치의 태스크 주석에 *"maximum 3 disputes can
receive provisional credit … only 3 can actually receive provisional credit"* 라는 **해설이 들어
있다**. 그것은 gold 주석이지 KB 가 아니다([[23]]). 위 [A]의 출처는 **`documents/` 아래 KB 문서**여야
한다.

### 격리로는 안 되는가 — **된다. 그리고 그것이 정확히 옳은 도구다**

이 물음은 [[62]] 2b 가 이미 정한 형태다: *격리에서 되면 결손은 전달(부하)이고, 격리에서도 안 되면
능력 경계다.* 무료이고 gold 를 안 본다.

```
P5-iso : 041 · 040 · 085 의 결정 시점에서 모델이 **실제로 받은 재료**만 주고
         + KB 축자(doc_015 · doc_032 · doc_036/_031)를 앞쪽에 두고
         => eligible_for_provisional_credit / customer_max_liability_amount 를 산출시킨다
부정통제 : 같은 길이의 무내용 문구를 넣은 팔([[57]])
exit    : 격리에서 닫히면 => 결손은 **전달**이고 합법 레버는 "표를 앞쪽에 전달"이다(값은 안 쓴다)
          격리에서도 안 닫히면 => **능력 경계**로 기록하고 이 17칸을 수리 대상에서 내린다
```
전례가 있다: `x511` 이 ①금액 축에서 같은 실험을 했고 *"B_policy(궤적과 같은 자리·앞쪽) **8/8** ·
C_policy_last(요구 직전) 8/8 합치되 산수가 깨진다"* 를 얻었다 — **표를 어디에 두느냐가 결과를 갈랐다**.
이 자리도 같은 설계를 쓴다.

**반증 / refutation**: 격리에서 닫히는데 라이브에서 안 닫히면 iso↔live 차이를 **프롬프트 두 개를 찍어
diff** 해야 한다([[78]]) — 추정 금지.

**선행 확인**: `_note_compute_ops`(PROVENANCE 축자 · doc_036/_031 · doc_032 인용) ·
`_note_compute_ops_removed_2026_08_19` · `x509_axis_queue` 의 `steps[S2].result.isolation_x511` ·
`grep -rn "provisional" tau2-bench/data/tau2/domains/banking_knowledge/documents/`.

**그래서 P5 는 측정만 한다 (무료·오프라인)**
```
P5a  041 · 040 · 085 의 궤적에 그 정책 표(doc_032 · doc_036/_031)가 **실제로 전달됐는가**
     전달됐는데 틀렸다 => 능력 경계(모델 몫) · 전달 안 됐다 => 전달 레버가 자리다(값은 안 쓴다)
P5b  L1 을 켠 팔에서 write_arg_enum[3].booleans 가 그 10칸에 발화하는가
P5c  CAP(기본 3) 소진 시점과 그 뒤 통과 여부 (fail-open 재현)
```
**exit**: P5a 가 "전달됨"이면 이 17칸은 **수리 대상에서 내린다**(측정값으로만 기록).

**반증 / refutation**: 표가 전달되지 않았음이 확인되면 *"모델 몫"* 이라는 접기는 거짓이 되고,
전달 레버가 정당한 후보가 된다. 반대로 전달됐는데도 틀렸다면 어떤 우리-층 처방도 이 칸을 못 산다.

**선행 확인**: `_note_compute_ops_removed_2026_08_19` · `_note_compute_ops`(PROVENANCE 축자) ·
`grep -rn "compute_ops" scripts/distill/tau2/a2/*.json`(specific:67 · gate:260 모두 `{}`) ·
`t2_resolve.py:1281 resolve_compute_params`(선언 없으면 no-op).

---

### 오선택 14칸 — **분류만 한다. 수리 대상이 아니다**

**주장 + 양화 (n=14 칸 · sim 5개)**: `credit_card_account_id`(041×4 · 040×2) ·
`card_id` `_green↔_blue`(092) · `_lb/_green↔_lg`(078) · `transaction_id`(026×4).
지목한 **레코드가 다르다** — 값 형식도 계산도 아니다.

**근거 — 축자(값 대조)**
```
041_4  GOLD cc_a6a7d745b2_gold   OURS cc_a6a7d745b2_crypto
041_5  GOLD cc_a6a7d745b2_crypto OURS cc_a6a7d745b2_gold      <- 서로 뒤바뀐 꼴
092_13 GOLD dbc_rw42b8d3e1_green OURS dbc_rw42b8d3e1_blue
078_3  GOLD dbc_mc78a5b9d2_lb    OURS dbc_mc78a5b9d2_lg
```

이 축은 x509 큐의 **②범주**이고 `x512`(경계 판정 철회) · `x513`(*"표를 줘도 057·063 은 0/6"*)이
**이미 판정한 자리**다. 여기서 새 처방을 만들지 마라([[74]]).

**필요한 것은 격리 프로브 하나**: *같은 종류의 카드·계좌가 여럿일 때 손님 발화의 지시체를
고르는가*. [[18]] 상 F3/경계 판정 전에는 **정보-맞춘 격리**가 선행이고, 이 문서는 그 프로브를
**기술만 하고 설계하지 않는다**(큐 밖 작업 금지 · §74-d).

**반증 / refutation**: 격리에서 지시체 선택이 닫히면 이것은 능력 경계가 아니라 전달 부하이고,
그때는 ②범주 축의 판정을 되돌려야 한다.

**선행 확인**: `x509_axis_queue_2026_08_24.json`(`axis_table.boundary_RETRACTED` · `status_2026_08_24_pm.②범주`) ·
`grep -rn "x512\|x513" reports/facet_rft_2026/`.

---

### (철회됨) D5 — `[ARG-ENUM]`: 선언된 값 집합에서만 오는 인자를 검사한다

**주장 + 양화 (n=1 칸 · sim 1개)**: `task_059`(`bank_k8141med1_20260903_2256`)는 gold 6 칸 중
**5 칸을 통과하고 059_4 한 칸**으로 떨어졌다. 그 한 칸의 차이는 문자열 하나다.

**근거 — 축자 + 파일:줄**
```
GOLD  059_4 : account_class = "Green Account"
OURS        : account_class = "Green Account (savings)"          <- 모델이 "(savings)" 를 덧붙였다

도구 문서 축자  tau2-bench/src/tau2/domains/banking_knowledge/tools.py:2384
  "account_class (string): The full official account class name"
정책 문서의 등급 열거 (tasks/정책 JSON 등장 횟수)
  Green Account 194 · Sky Blue Account 18 · Gold Account 16 · Silver Plus Account 16 · Bronze Account 6

우리 층은 이 호출을 건드린 적이 없다 — task_059 의 tool-deny 는 3건뿐이고 전부 다른 자리다:
  turn 49  "resolve the flagged call(s) first"
  turn 53  "[ACTION] 'apply_for_credit_card' is run by the CUSTOMER, not by you"
  turn 53  "[ARG-EMPTY] ... left the required argument(s) ... as an empty string"
```

⇒ **059 는 우리 스택의 부작용이 아니다**(같은 sim 의 다른 `open_bank_account` 칸 059_3 은 통과).
그러나 **우리가 잡을 수 있었던 결함**이다: `account_class` 는 자유 문자열이 아니라 **열거값**이고,
지목한 값이 그 집합에 속하는지는 **닫힌 술어**다([[22]] 변이 불변 · [[59]] 문자열 소속만).

```
규칙 : 인자의 허용값 집합이 선언돼 있으면(A2/정책 열거), 집합에 없는 값을 거부한다.
문면 : 무엇이 틀렸나(집합 밖) + 무엇을 하면 풀리나(선언된 값 중 하나를 쓰라).
       ⛔옳은 값을 골라 주지 않는다 — 고르는 순간 측정 대상이 사라진다([[62]] · [[23]]).
```
같은 계열이 **이미 둘 있다**: `[ARG-EMPTY]`(빈 문자열) · `[SIGNATURE]`(선언 안 된 인자).
D5 는 그 형제이고 엔진에 도메인 리터럴을 박지 않는다([[58]]) — 집합은 A2 에서 온다.

**반증 / refutation**: 그 열거가 A2·정책에서 **닫히지 않으면**(등급이 문서마다 다르거나 자유 서술이면)
이 술어는 성립하지 않고 D5 는 폐기다. 그리고 059 의 `"Green Account (savings)"` 를 거부했을 때
모델이 올바른 값으로 재발행하지 못하면 **레버가 사는 것은 0**이다 ⇒ P4 에서 격리로 확인한다.

**선행 확인**: `grep -rn "account_class" tau2-bench/data/tau2/domains/banking_knowledge/*.json` ·
`grep -rn -A3 "account_class" tau2-bench/src/.../tools.py`(:2377·:2384·:2394) ·
`grep -rn "ARG-EMPTY|SIGNATURE" scripts/distill/tau2/` (기존 형제 게이트 확인) ·
회수된 `fb_bank_k8141med1_20260903_2256.jsonl` 의 task_059 deny 3건 전문.

---

## 3. 격리 먼저, 배선은 그 다음 ([[62]] · [[78]])

### P0 (선결·무료) — R2 를 먼저 친다

`feedback.absent` 가 종결 뒤 실제로 발화했는지 확인한다. 회수해 둔
`sim_results/bank_049ctl2_20260904_0534.log.gz` 와 `fb_*.jsonl.gz` 에서
`[PROCEDURE] You are inside` 의 turn 별 출현을 센다.
**0건이면 §1 의 원인 진술을 폐기하고 이 문서를 여기서 멈춘다.**

### P1 — D1 격리

- 프로브는 프롬프트를 쓰지 않고 **엔진 빌더를 부른다**([[78]]). 팔은 선언 오버라이드 한 칸
  (`terminal_closes_procedure = on/off`).
- 재료: 그 sim 이 종결 시점에 실제로 받은 메시지 전량(축자 재생).
- **exit**: off 에서 읽기 루프 재현 ∧ on 에서 소멸 ⇒ D1 배선 자격. 둘 다 루프면 R1 성립 ⇒ D1 폐기.
- 부정통제 필수([[57]]): 같은 길이의 무내용 문구를 붙인 팔.

### P3 — D3/D4 격리 (2026-09-04 신설)

- **P3a**: `apply_op(filter)` 의 반환형을 찍는다(스칼라 vs 집합). 집합이면 수리 위치가 `t2_compute` 다.
- **P3b**: 041 의 8개 id 각각이 formalize 된 criteria 에 부합하는지 결정론으로 센다.
  8개 다 부합하면 D3 확정 · 1개만 부합하면 이 귀속은 무너진다.
- **P3c**: D3 만 켠 팔 vs D3+D4 팔을 같은 재료로 돌려 D4 의 단독 기여를 잰다([[57]] 부정통제 포함).
- **exit**: D3 가 041 의 8 칸을 통과시키는가 · D4 가 그 위에 무엇을 더 사는가.

### P4 — L1 격리 (꺼진 열거 레버)

- **P4a**: `T2_WRITE_ARG_ENUM` 이 **언제 꺼졌는지** git 이력으로 찾는다(`git log -S`). 의도적 OFF 면 그 이유를 인용한다.
- **P4b**: 켠 팔에서 059·066·071 의 값이 실제로 거부되는가, 그리고 모델이 **선언된 값으로 재발행하는가**.
- **P4c**: `T2_WRITE_ARG_ENUM_CAP`(기본 3) 소진 후 fail-open 재현 — 켠 것과 안 켠 것이 같아지는지.
- **exit**: 재발행 성공 ∧ CAP 소진 전 발화 ⇒ L1(켜기) 자격. 부정통제 필수([[57]]).
  ⛔[[70]] 판정 의무 3종(ON/OFF reward 짝 · 태스크별 부호표 · 무엇을 팔았나)을 채우기 전엔 켜지 마라.

### P2 — D2 격리 + K 결정

- K ∈ {2, 3, 5} 를 각각 재고, 루프가 끊기는 **가장 작은 K** 를 쓴다. gold 무참조.
- **exit**: 어떤 K 에서도 안 끊기면 D2 폐기(경로 없음으로 기록).

### P6 — D6 격리 (중복-쓰기 창 리셋)

- **P6a/P6c 는 이미 돌았다**(§2 D6 절). 여기 남는 것은 **배선 자격 프로브**다.
- **P6d**: `x548_dup_deny_iso.py`(2026-08-26 존재 · [[74]] 위반으로 뒤늦게 발견)의 051 케이스
  빌더에 **상태-변화 리셋 술어**를 넣은 팔과 안 넣은 팔을 돌린다. 재료는 gold 무참조.
- **exit**: 리셋 팔에서 gold 051_2/051_7(바이트 동일 재제출)이 통과하고, 반사실 재현 9,468 sim 중
  **선언된 write 키에서만** 억제가 사라지는가. 다른 키의 억제가 함께 풀리면 술어가 넓다 -> 재설계.
- 부정통제([[57]]): 무내용 재시도 팔.

### P7 — D7-② 격리 (grounding 접두 정규화)

- 팔은 **선언 오버라이드 한 칸**(`ground_strip_source_prefix = on/off`), 프롬프트 미사용([[78]]).
- 재료: `task_064#s626729` msg59 의 components 축자 + 같은 코퍼스.
- **exit**: (i)`파일명: '인용'` 접두형과 (ii)순수 인용을 같은 components 로 넣어
  **off 에서 (i)만 거절 ∧ on 에서 둘 다 통과**면 D7-② 배선 자격.
  **둘 다 통과하면 이 귀속은 거짓**이고(§1c-5 D7 반증조건 (i)), **둘 다 거절되면** 원인은 접두가
  아니라 코퍼스 도달이다 -> C582 의 선행 처방(`isolate.getter_tools` bm25->shell)으로 돌아간다.
- ⛔C582 가 같은 문면에 **다른 원인**을 이미 확정해 뒀으므로([[56]]·[[40]]), 이 프로브는
  **C582 를 반증하는 형태로** 설계돼야 한다 — 그래서 (ii)순수 인용 팔이 필수다.

### P8 — D8 격리 (선언 스키마 ↔ 소비부 이름)

- **P8a (무료·오프라인)**: 스키마를 고친 뒤 `test_terse_schema.py` 를 **소비부에서 필드 이름을
  뽑아 대조**하도록 바꾸고 돌린다. 지금 판은 리터럴 `'"claim"'` 을 단언해 **오답을 못박는다**.
- **P8b (격리)**: `agent_claimprov` 호출 하나를 격리에서 재생하고 산출 JSON 의 `what`·`kind`
  점유율을 잰다. **exit**: `what` 비율이 0% -> 유의하게 상승하지 않으면 원인은 스키마가 아니다.
- **P8c (스모크)**: 고친 팔 첫 런에서 전송 문면에 `"None: None"` 이 **0건**인지 센다([[81]]
  — 고쳐 놓고 켠 적 없는 레버). 계기 이름 = `fb_*.jsonl` 의 `channel=claimprov` 텍스트.
- 부정통제([[57]]): 스키마만 바꾸고 렌더는 그대로 둔 팔(= 이름이 살아나는가만 본다).

### P9 — D14 격리 (재생성 산출물의 게이트 재진입)

- 팔 = 선언 오버라이드 한 칸(`regen_calls_reenter_write_gates = on/off`) · 프롬프트 미사용([[78]]).
- 재료 = 029 t72 · 048 t36/t55 · **027 t73(searchexhaust 채널 · 정상경로 5회 DENY 뒤 우회 커밋)** 의 메시지 전량(축자 재생).
- ★채널 트리거 결합: `t2_gate_patch.py:14432` `_resign or _srchex_mid` — **M1 사임-창이 이 채널을 연다**(1f-7).
- **exit**: off 에서 금지 write 커밋 재현 ∧ on 에서 deny ⇒ D14 배선 자격.
- ⛔[[70]] 부호표 필수: 048 t63(gold `pay_credit_card`)·t123(gold unlock) — **재생성이 만들어 낸
  gold 칸 2개가 함께 죽지 않는지** 센다(예측: `pay` 는 `write_evidence_specs` 밖이라 무영향 · [미측정]).
- 부정통제([[57]]): 029 t72 의 넛지 3연발(`uncalled_unlock`→`searchexhaust`→`claimprov`)을
  하나씩 끈 4팔 — 어느 넛지가 결정적인지 아직 아무도 못 갈랐다(§1f-5 #5).

### 배선 조건

⛔**「P1·P2 를 통과한 것만」은 자기 후보 명단과 어긋나 있었다** — D3·D4·D6·D7·D8·L1 은
P1·P2 의 대상이 아니다. 정정: **각 후보는 자기 프로브(P1~P8)를 통과해야 배선한다.**
프로브가 없는 후보는 **배선하지 않는다**(D9 는 레버가 아니라 원장이므로 이 규칙 밖이고,
D7-① 는 위 면제 논거로 프로브 밖이다). 통과 후 `go_stack.sh` 정본 런처에 **등재까지가 한 작업**이고,
첫 런에서 **실발화를 확인**한다([[81]] — 고쳐 놓고 켠 적 없는 레버가 실재한다).

---

## 4. 스모크 게이트 ([[73]])

full-run 전에 반드시 통과시킨다. 단위테스트 통과 ≠ 라이브 발화.

```
(1) --num_tasks 10 --num_trials 1  (~6분)   크래시 0
(2) 전 절차 terminal 덤프 — **설계 시점에 이미 찍었다(§2 B1 실측)**. 여기서는 **회귀 검사**로만:
    banking terminal 수가 B1 표와 같은가 · 다른 도메인에 terminal 이 둘 이상인 절차가 있는가
(3) 계기 이름을 명시해 센다 — ⛔"D1 발화 카운트" 는 틀린 계기다. D1 은 **억제** 레버라 발화가
    *감소*한다. 세야 할 것은 **절차 인스턴스 `closed` 마킹 수 > 0** 이고, 짝으로
    `feedback.absent` 발화가 종결 뒤 **0** 인지 본다. D2·D6·D7·D8 도 각자 계기 이름을 적는다([[81]])
(4) 기존 배터리: test_a2_three_layer.py · test_c207_envelope.py · test_lever_reachable.py
(5) 등가 게이트: 정본 A2 만 고치고 gate.json 미동기화면 FAIL ([[24]])
```

---

## 5. 실험 — 97 태스크 A/B

### 팔

```
A (대조) : 현행 sha (수리 전)   ⚠**정의가 아직 안 섰다 — §2 「⛔D6-op」 참조(이 절보다 위에 있다)**
B (처치) : A + **격리를 통과한 것 전부 합성**([[19]] 합성-우선 · [[60]] 레버는 전부 켠다)
```

⛔**B2 (2026-09-04 리뷰) — 초안의 `A + D1 + D2` 는 폐기한다.** 그 팔은 §1c-4 가 **약화시킨 둘만**
담고 §1c 가 **승격시킨 셋(D6·D7·D8)** 을 뺐다:
```
D1  §1c-4 판정 = 중립~약화 (3건에 재현 0)      초안 팔에 있음
D2  §1c-4 판정 = 약화 (그 칸은 D7 이 산다)      초안 팔에 있음
D6·D7·D8  CONFIRMED 우리-층                    초안 팔에 **없음**
```
[[19]]/[[60]] 상 **격리 통과분은 합성해서 켜는 것이 기본**이고, 무엇을 끄려면 그것이 판정이어야 한다.
⇒ 팔 확정은 **§1d(회귀 대조)와 각 프로브의 exit 가 나온 뒤**에 한다. 그때까지 이 칸은 열어 둔다.

#### ⛔2차 리뷰 — 「합성 팔」 해석은 맞지만 정본이 준 해답 둘을 안 적었다

**⑴ 내가 만든 순환**: *"무엇을 끄려면 그것이 판정이어야"* 라고 [[19]]/[[60]] 보다 **강하게** 적었는데,
[[19]]-3 은 **"귀속용 실험 arm" 예외를 명시**한다. 인용하지 않아 *끄려면 판정 · 판정하려면 꺼야* 하는
순환을 스스로 만들었다. ⇒ [[19]]-3 을 인용하고 **귀속 arm 은 예외**로 적는다.

**⑵ [[70]]① 은 원리상 못 채운다**: 판정 의무 ①이 **"단일변수"** 를 괄호로 명시하는데 8~9 레버 팔은
그것을 만족할 수 없다. `LEVER_ROSTER_CANONICAL` 의 VALID 정의도 단일변수 요구라 **155건 중 VALID 0**
이고, 합성 팔은 그 0 을 **구조적으로 유지**시킨다. `C217` 은 *"스택 매일 변경"* 을 **캠페인 무효 사유**로
박제해 뒀다.

**⇒ 정본의 해답 둘을 채택한다**
```
[[73]]         합성 스택으로 돌리되 Δ 귀속은 **per-step 포렌식**으로 (§7 에 per_task_forensics 추가)
E-ABL-NUDGE    **층-단위 1회 판정** — D1~D9·L1 이 각각 어느 층인지 선언한다
```

**층 선언 (채워야 할 칸)**
```
표면화/절차 층 : D1 · D2
게이트 문면 층 : D3 · D7 · D8 · D12
게이트 술어 층 : D4 · D6 · D13
계기/원장 층   : D9 · D10
배선(ON/OFF)   : L1
```

**⑶ 부호표 단위가 문서 안에서 충돌한다** — §5 *"태스크별"* ↔ 진입점 *"기전별 per step"*.
⇒ **기전별로 통일한다**(정본 `C498ⓖ` / `DEFECT_LEVER_COVERAGE:99` — *"1차 종점을 성적이 아니라
기전 계수로"*).

**⑷ nt=1 은 판정 도구가 아니다**
```
C467  "nt=1 은 판정 도구가 아님 — 레버 효과 판정은 nt=4 로만" (074 가 연속 런 9/13 -> 0/13)
C249  nt=2 · 9/31 = 29%  "이후 모든 arm 비교의 판정 임계"
C292  16/64 = 25%        "pass 24<->24 는 무변화가 아니라 8↑8↓ 상쇄"
C548  "Δ<4/40 은 시행 변동과 구별 불가"  <- 4/40 은 **바닥이지 유의선이 아니다**
```
⇒ **§5 의 194 sim = nt=1 설계를 폐기하고 nt>=2 로 간다.** 그러면 **flip 을 병산**할 수 있다 —
Q38 자체 flip 은 같은-태그 반복이 **4 태스크뿐**이라 아직 미측정이고, 그것이 §1e 의 실측 경로다.

**⑸ Δ>=10/97 은 값은 방어되나 유도가 틀렸다**
비례 환산(4/40 -> 10/97)이 아니라 **flip-불일치쌍 산수**로 다시 쓴다:
97 태스크 · flip 25~29% 에서 **McNemar 2σ = 9.7~10.4** ⇒ 값은 같지만 근거가 다르다.

### ⛔F6 정합 — *"지배적 실패 모드가 컨텍스트 소진"* 은 §1b 와 어긋난다 (2026-09-04)

§1b 전수 포렌식은 **컨텍스트 소진 0건**이라고 적었다. 그런데 아래 재실행 논거가 한때
*"지배적 실패 모드가 컨텍스트 소진"* 이라고 썼다 — **같은 문서 안의 모순**이다. 실물은 이렇다:
```
컨텍스트 **소진**(창을 다 써서 잘림)          §1b 전수 20건 중 **0건**
선언 blob **폭주**(28.2~39.1KB · TRUNC)      agent_claimprov 17/420 = 4.0%  (§1e-4)
sim 벽시계 장기화(242~301분)                 §1c 3건 전부 · 원인은 배치([[83]]) 와 재-prefill
```
⇒ 재실행 논거는 **컨텍스트 소진이 아니라 ⑴배치 조건 혼합([[83]]·[[54]]) ⑵`f6224e26` 이후
D8 결손 상시화** 두 가지다. 아래 문단은 그 전제로 읽어라.

### ⚠ 대조군을 재실행해야 하는가 — 판단과 대가

지금 도는 캠페인(2026-09-03~04)은 **비교 규격이 깨져 있다**:

- `max_concurrency` 가 런마다 **4 와 2 로 섞였다**
- 서버가 `.151` 과 `.153` **두 대**로 갈렸고 `.151` 은 2026-09-04 07:00 에 반납했다
- 일부 태스크는 죽은 런의 잔여를 다른 태그에서 이어 돌았다

**권고: 대조군을 재실행한다.** reward 자체는 conc 에 불변일 **가능성이 높지만**(축출은 재계산일
뿐 토큰을 바꾸지 않는다), 이 태스크군의 지배적 실패 모드가 **컨텍스트 소진**이라 배치 조건이 종료
시점에 개입할 여지가 있고, 그 여지를 남긴 채로 Δ 를 주장할 수 없다([[54]]).

**무료 대안**: 이번 캠페인을 대조군으로 재사용한다. 그 경우 *"대조군은 혼합 배치에서 수집됨"* 을
명시하고, Δ 판정 시 **컨텍스트 소진으로 끝난 sim 을 따로 센다**. 비용을 아끼려면 이쪽.

### 배치 설계 (`.151` 반납 반영) — ★2026-09-04 07:40 측정으로 규칙을 바꿨다

```
엔진 2개 (.153 GPU0=8141 · GPU1=8143)
레인 = 엔진당 1개 (kvlane.sh · nb() 는 HOST:PORT 로 센다 — 포트만으론 엔진이 식별되지 않는다)
MAXB=1

★배치 규칙: conc 숫자가 아니라 **비행 컨텍스트 합 <= kv_cache_size_tokens (171,749)**
```

**왜 conc 가 기준이 아닌가 — 실측(§5a)**. `conc 2` 는 태스크가 짧을 때만 맞는 근사다.
92k 짜리 태스크는 **혼자 돌려야** 예산 안이고, 40k 짜리 둘은 같이 돌려도 된다.

⇒ 큐를 짤 때 각 묶음의 **태스크별 최대 `agent_response` prompt 실측치**로 묶는다. 그 값은
회수된 런 로그에서 무료로 뽑힌다(`[T2_GEN_TRACE] call=agent_response ... prompt=<N>` 의 최댓값).

### §5a 근거 — prefix 캐시 붕괴 측정 (2026-09-04 07:4x · 4분 구간)

같은 sha · 같은 팔(`viewmax2`) · 같은 모델인데 두 엔진의 **prefix 적중률이 16배** 갈렸다.

```
포트 8141 (비행 4 sim)  질의 265,321 블록 · 적중   9,408 -> 구간 적중률  3.5%  · 축출 +8
포트 8143 (비행 3 sim)  질의 198,782 블록 · 적중 112,112 -> 구간 적중률 56.4%  · 축출 +2

비행 컨텍스트 합 (sim 별 agent_response 최대 prompt)
  8141   92k + 89k + 87k + 77k = 346,733  = 예산의 2.0배  -> 적중률  3.5%
  8143   99k + 52k + 37k       = 190,779  = 예산의 1.1배  -> 적중률 56.4%
```

**기전**: sim 이 user-sim(gpt-5.2) 응답을 기다리는 동안 엔진에서 내려가고, 그 사이 다른 sim 들이
그 sim 의 캐시된 prefix 블록을 밀어낸다. 다음 턴에 돌아오면 90k 를 처음부터 다시 계산한다.
**이 퇴출은 `num_preemptions_total` 에 한 줄도 안 남는다** — 그 계수기는 *비행 중* 요청의 선점만
센다. 위 구간의 축출이 +8/+2 로 거의 0인데 적중률이 3.5% 인 것이 그 증거다.

**반증 / refutation**: 비행 합을 예산 아래로 넣었는데도 적중률이 60% 를 밑돌면 이 설명은 부족하고,
남은 몫은 다른 곳(예: 재생성 채널)에 있다.

⚠**D5 (2026-09-04 리뷰) — 예산선 171,749 는 외삽이다.** 측정점이 **2.0배(3.5%)·1.1배(56.4%) 둘뿐**이고
**1.0배 이하 점이 없다.** 그런데 §5 「배치 설계」는 이미 그 규칙으로 묶음을 짜라고 지시한다
(구판은 *"§7 7a"* 를 가리켰는데 §7 이 2차 리뷰의 10단으로 교체되면서 **그 항이 없어졌다**) ⇒ 첫 배치에서
**1.0배 이하 점을 반드시 찍어** 선을 확정한다.
⚠**그리고 sim 최대 prompt 로 묶으면 초반 턴에 GPU 를 크게 놀린다** — 최대치는 sim 후반에만 난다.
⇒ 실무 규칙: **최대 기준으로 묶되 진행에 따라 순차 투입(admission)** 한다. 즉 슬롯을 미리 비워 두지
말고, 비행 중 컨텍스트 합이 예산 아래인 동안만 다음 sim 을 들인다.

**재생성 채널(`_ap_regen`)은 주범이 아니다 — 반증됨. ⛔단 이 반증의 대상은 «prefix 캐시 붕괴»뿐이다**
(§1f-2 가 확정한 «쓰기 게이트 우회»는 별개 결함이고 이 문장으로 기각되지 않는다). 재생성 비율은 8141 **7.6%**(619 중 47) ·
8143 **11.1%**(162 중 18) 로, **재생성이 더 잦은 쪽의 적중률이 16배 높다**. 상관이 반대다.
다만 8143 의 56.4% 도 순수 append 대화의 기대치(90%대)에는 못 미치므로, **비행 합이 예산 아래로
완전히 들어간 상태에서 한 번 더 재서** 잔여를 귀속시킨다(무료).

### 비용·기간 (정직하게)

```
sim 수 : 97 × 2 arms = 194 (대조군 재실행 시) / 97 (재사용 시)
처리율 : 엔진 2개 · conc2 실측 sim 당 12~65분 (n=3 · bank_k8143med1)
         => 낙관 4 sim/h · 비관 2 sim/h
기간   : 194 sim -> 48~97시간 / 97 sim -> 24~48시간
비용   : user-sim = openrouter gpt-5.2 ([[30]] 권장표준) — [[09]] 사용자 승인 필요
```

⚠ 위 처리율은 **비행 합이 예산의 1.1배였던 구간**(적중률 56%)에서 잰 것이다. §5a 규칙대로 예산
아래로 묶으면 재-prefill 이 줄어 빨라질 여지가 있으나 **그 상태를 아직 재보지 않았다** — 추정에
반영하지 않았다. 첫 배치에서 재고 이 표를 갱신한다.

### 판정 기준

- 1차 지표는 **reward**(궤적 재실행 후 DB 해시 비교 · [[69]]). 집계 metric 에서 결론 직행 금지([[08]]).
- **Δ ≥ 10/97** 을 유의로 본다([[73]] 의 Δ≥4/40 관례를 97 로 환산).
- **태스크별 부호표 필수**([[70]]) — 무엇을 샀고 **무엇을 팔았나**. D1 이 표면화를 줄이므로 종결
  전 단계를 놓치는 태스크가 생길 수 있다. 그 손실을 세지 않으면 판정이 아니다.
- 우리 층 귀속은 per-step 포렌식 + 적대적 refutation 을 거친 것만 CONFIRMED([[73]]).

### ★비용 축 — reward 만 보면 "무엇을 팔았나"가 안 보인다 ([[70]])

**주장 + 양화 (n=2 sim · base 대조)**: 같은 태스크를 base 는 분 단위로 통과하는데 ours 는 시간
단위를 쓴다. 두 사례 모두 base 팔이 **pass** 한 태스크다.

**근거 — 축자 + 위치**
```
task_059   base(x644) reward=1.0 ·  15분 · msg 47      ours reward=0.0 · 291분 · msg 72
task_064   base(x644) reward=1.0 ·  20분 · msg 68      ours **reward 0.0 · 92 msg · 301.0분**(완주·실패)

task_064 의 생성 호출 분해 (bank_k8141med1_20260903_2256.log · [T2_GEN_TRACE] call=... 집계)
  ⛔**아래 29↔30 은 §1c-6 이 정정하라고 적어 둔 수다 — 정정본을 쓴다: 총 79콜 · 부수 16.5%.**
  (초안) agent_response 29  ↔  부수 생성 30
  (intent_operator_formalize 5 · source_claim_formalize 5 · recommend_formalize 4 ·
   agent_response_unified_regen 6 · claimprov 6 · selfdecl 3 · sg_arg_docs 2 · 기타)
=> 실질 턴당 생성이 2배 이상이다.
```

그러므로 판정표에 **세 칸을 더 적는다**(태스크별 부호표와 같은 줄에):

```
① reward 짝 (A/B)                      <- 지금 유일하게 보고 있는 것
② sim 당 벽시계 분                       <- 우리가 파는 것
③ base 대비 **생성 호출 배수** = (ours 총 생성 호출) / (base 총 생성 호출)
   ⛔초안의 *"(agent_response + 부수 생성) / base 의 **turn 수**"* 는 **단위 불일치**다
     (분자 = 콜 수, 분모 = 턴 수 · 2026-09-04 리뷰 D3). 콜/콜 로 맞추거나 **턴당 생성**으로 적는다.
```
②③ 없이 Δ 를 보고하면 *"정확도를 샀다"* 만 남고 **대가가 장부에서 사라진다**. base 팔의 분/턴은
`x738_q38_base97_census_2026_09_04.md` 의 두 런에서 무료로 뽑힌다.

**반증 / refutation**: ②③ 의 격차가 **KV 경합만으로** 설명되면(예산 안에서 돌린 배치에서 격차가
사라지면) 이건 우리 레버의 대가가 아니라 배치 문제다 ⇒ §5a 규칙대로 묶은 첫 배치에서 다시 잰다.

**선행 확인**: `grep -rn "T2_GEN_TRACE" scripts/distill/tau2/` · 회수된 base 런
`bank_x644_q38base_bank78_20260830.results.json.gz`(duration·messages) · 이 캠페인 로그의 호출 집계.

---

## 6. 중단 조건

| 신호 | 조치 |
|---|---|
| P0 에서 `[PROCEDURE] You are inside` 0건 | 이 문서 폐기. 원인 진술이 틀렸다 |
| P1 on/off 둘 다 루프 (R1) | D1 폐기 · *"표면화는 원인이 아니다"* 로 기록 |
| P2 어떤 K 에서도 안 끊김 | D2 폐기 · 경로 없음으로 기록 |
| P6d 리셋 팔에서 **선언 밖 키의 억제까지 풀림** | D6 술어가 넓다. 재설계 전 배선 금지 |
| P7 에서 접두형·순수인용 **둘 다 통과** | D7-② 귀속 거짓 · 폐기 (①만 남긴다) |
| P7 에서 **둘 다 거절** | 원인은 접두가 아니라 코퍼스 도달 ⇒ C582 선행 처방으로 복귀 |
| P8b 에서 `what` 점유율이 안 오름 | D8 원인은 스키마가 아니다 ⇒ 프롬프트 축으로 이관 |
| P8c 에서 `"None: None"` 전송 > 0 | 수리가 라이브에 안 닿았다. 런 금지([[81]]) |
| P4 에서 CAP 소진 전 발화 0 | L1 배선 자격 없음 · *"켜도 안 닿는다"* 로 기록 |
| **기전별** 부호표에서 손실 > 이득 | 배선 철회. 끄지 말고 조건을 조정한다([[19]] · [[70]]) |

⛔**F8 정합 — 「스모크 (3) 발화 0 ⇒ 런 금지」는 D1 에 그대로 쓰면 틀린다.** §4(3)이 이미
경고하듯 **D1 은 억제 레버라 발화가 감소한다**. 그래서 이 행은 **후보별 계기 이름**으로 갈린다:

| 후보 | 스모크에서 세는 것 (0 이면 런 금지) | 방향 |
|---|---|---|
| D1 | 절차 인스턴스 `closed` 마킹 수 | **증가** (`feedback.absent` 발화는 **감소**해야 정상) |
| D2 | 읽기-루프 이름/출구 문면 발화 수 | 증가 |
| D3 | `[REFERENCE]` 거절 수 | **감소** (문면-술어 일치 후 오거절이 준다) |
| D4 | `[BLOCKED]` 중 **의존 호출** 비율 | 비율 상승 · 총량 감소 |
| D6 | `[DUPLICATE-WRITE]` 억제 수 | **감소**(선언된 write 키에서만 남는다) |
| D7 | `[GROUNDING WARNING]` 문면에 요구형식 포함 수 | 증가(①) · 경고 총량 감소(②) |
| D8 | `fb_*.jsonl` `channel=claimprov` 텍스트의 `"None: None"` | **0 이어야 한다** |
| D14 | `_ap_regen` 산출 호출 중 **게이트 체인 밖 커밋** 수 | **0 이어야 한다**(역방향 계기) |
| D9 | `reminder-assistant` 행의 채널 종류 수 | 증가(1 -> 3 이상) |
| D10 | `declaration failed` 수 | 감소 |
| L1 | `T2_WRITE_ARG_ENUM` 발화 수 | 증가(0 -> N) |

---

## 7. 실행 순서 — **2차 리뷰의 10단으로 교체** (2026-09-04)

```
[x] 1  D9   원장 — C2(부호표)의 선행조건. **문서 반영 완료 · 코드 배선은 캠페인 완주 후**
[x] 2  P0   종결 후 표면화 실물 = **sim 당 1~2건**(런 전체 6행). §1 양화 교체 · D1 기대치 하향
[x] 3  B1   3행 표 -> 2행 · tool_any 전체 열거 · 수리 ②는
             **"실행 가능한 terminal 이 유일할 때만"** 한쪽으로 확정
[x] 4  §3   P6·P7·P8 프로브 신설 · 「배선 조건」을 **후보별 자기 프로브**로 교정 · §6 exit 7행 추가
[x] 5  D7   ① 문면 수리 = **면제 논거 명시**([[64]] 문면 수리는 술어 불변) · ② 는 **P7 뒤**
[x] 6  D8   ★**양화 갱신을 넘어 원인이 확정됐다** — 158/158 · **전송 문면 73/73** ·
             날짜 절벽(09-01↔09-03) · 원인 = `f6224e26` 스키마가 소비부 이름과 불일치.
             ⇒ *"침묵 vs tool-렌더 2팔"* 은 **불필요**(둘 다 증상 우회). §1c-5 D8 전면 재작성
[x] 7  C2   재작성 — *"무료·결정론"* 삭제 · **D9-후 · 격리-재실행** 조건부로
[x] 8  §5   층 선언 · [[19]]-3 귀속 arm 조항 · **nt>=2(flip 병산)** · Δ>=10/97 유도 교체
[x] 9  §1e  정본 5종 인용 프레임(C251·C292·C273·C498ⓖ·A1_V2_NT2_FORENSIC)
[x]10  정합 청소 — F3·F4·F6·F8·F9·F10 + 진입점 근거를 x725 -> C249/C292 로 교체
             (F9 는 정합이 아니라 **양화 정정**이 됐다: D3 의 "8칸" -> **실측 3콜**)
[x]11  D10  자연실험 **철회 확정** — 채점 sim 만: declfail 7 @0.5714 ↔ 정상 42 @0.5714 = **차이 0**
[x]12  [[74]] 인용 보완 — D8:`CLAIM_DEMAND_ISO_VS_LIVE_AUDIT` · D3·D4:`refute_2026_08_24/`
             (`refute_055.json:31` 형제-통과 반례) · D10:`model_profiles/Qwen__Qwen3.8-27B-FP8.env:60-67`
```

**남은 것 (캠페인 완주가 선행)**
```
[ ] A  캠페인 완주 — 2026-09-04 14:35 현재 78/97 · pass 46 · 19 태스크 남음
[ ] B  task_091 base 재실행 (`--gate 0` · 1태스크 1시행) — x738 §4 R3
[ ] C  코드 배선: D8 스키마 수리 -> P8a/P8b/P8c -> D9 원장 -> D7-① 문면
[ ] D  [[73]] **per-step 포렌식**을 §7 의 정식 단계로 — 합성 스택 런의 Δ 귀속 도구
       (산출물 이름 `per_task_forensics` · §5 「층 선언」과 짝)
[ ] E  프로브 P6d·P7·P4 실행 후 배선 자격 판정
```

⛔ **캠페인 중에는 아무것도 배선하지 않는다**([[54]]). 1~3·7·9·10 은 오프라인이라 지금 해도 되고,
**엔진/A2 편집과 런은 완주 후**다.

⛔ **§7 5·6단이 「무조건 수리」로 읽히면 §1c-5 머리말(*"넷 다 격리 프로브 전에는 배선하지 않는다"*)과
정면 모순이다.** D7-① 만 면제이고 그 논거는 **"[[64]] 문면 수리는 술어를 바꾸지 않는다"** 이다 —
술어를 건드리는 D7-② 는 **P7 게이트 뒤**다.

⛔ **[[73]] per-step 포렌식이 §7 에 없었다 — 넣는다.** 합성 스택으로 돌리되 Δ 귀속은
`per_task_forensics` 로 한다(아래 §5 「층 선언」과 짝).

---

## 8. 이 캠페인의 기준선 (2026-09-04 06:07 시점 · 진행 중)

```
arm=viewmax2 · 2026-09-03 이후
  완료 sim 55 · 고유 태스크 53/97 · pass 33 (채점분의 62%)

관측된 배치 병리 (재발 방지 대상)
  포트당 비행 토큰이 예산의 4.4배 -> Waiting 5 · KV 94% · 생성 20.5 tok/s 고착
  과부하 런 하나를 빼자 같은 엔진이 KV 30% · Waiting 0 · 42.6 tok/s
  ★그리고 그 값의 정체는 prefix 캐시였다 — 비행 합 2.0배에서 구간 적중률 3.5%,
    1.1배에서 56.4% (§5a). conc 4 로 발사해도 Running 은 늘 2 였다 — 엔진이 못 돌린 게
    아니라 매 턴 90k 를 다시 계산하느라 못 나아갔다.
```
