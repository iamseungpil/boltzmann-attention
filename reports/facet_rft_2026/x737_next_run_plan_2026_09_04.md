# x737 — 다음 97 실행 계획서 + 설계서 (2026-09-04)

> **자리**: 이 문서는 정본이 아니다. 정본 작업 순서는 `x509_axis_queue_2026_08_24.json` 이고,
> 여기서 제안하는 두 수리는 그 큐의 **새 단계 후보**다. 프레임 LOCK 은 `RESEARCH_MASTER.md` §1.
> **범위**: 진행 중인 banking 97 캠페인이 끝난 **뒤**에 무엇을 고치고, 그 수리를 어떻게 97 태스크
> A/B 로 재는가. 캠페인 중에는 아무것도 배선하지 않는다([[54]]).

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

## 2. 설계 — 수리 후보 둘

### [[05]] 3질문 (설계서 상설 의무 · [[17]])

1. **무엇이 고정인가**: TBox weights + Scaffold 엔진. 두 수리 모두 **엔진 층**이고 도메인 상수를
   담지 않는다.
2. **무엇이 변경인가**: 없음. A2/ABox 는 **손대지 않는다** — `close.requires` 도 노드 목록도 그대로.
3. **도메인-특화가 섞이는가**: 아니다. 아래 술어는 banking 도구 이름도 태스크 id 도 담지 않는다
   ([[58]] · [[05]]). `credit_card_closure_retention` 은 **관측 대상**이지 조건이 아니다.

### D1 — 절차가 종결되면 잔여 노드 표면화를 멈춘다

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
- **위험**: 종결 뒤에도 정당한 후속 단계가 있는 절차에서는 표면화가 사라진다. ⇒ 스모크에서
  전 절차의 종결 노드 목록을 덤프해 눈으로 확인한다(§4 (2)).

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
- (3)의 "조건부" 는 A2 의 `requires` 구조로 닫힌다 — 내용 해석 없음([[59]]).

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

### P2 — D2 격리 + K 결정

- K ∈ {2, 3, 5} 를 각각 재고, 루프가 끊기는 **가장 작은 K** 를 쓴다. gold 무참조.
- **exit**: 어떤 K 에서도 안 끊기면 D2 폐기(경로 없음으로 기록).

### 배선 조건

P1·P2 를 통과한 것만 배선한다. 통과 후 `go_stack.sh` 정본 런처에 **등재까지가 한 작업**이고,
첫 런에서 **실발화를 확인**한다([[81]] — 고쳐 놓고 켠 적 없는 레버가 실재한다).

---

## 4. 스모크 게이트 ([[73]])

full-run 전에 반드시 통과시킨다. 단위테스트 통과 ≠ 라이브 발화.

```
(1) --num_tasks 10 --num_trials 1  (~6분)   크래시 0
(2) 전 절차의 terminal 노드 목록 덤프 — 눈으로 확인 (D1 이 엉뚱한 노드를 종결로 잡지 않는가)
(3) D1 발화 카운트 > 0 · D2 발화 카운트 > 0   <- 0 이면 배선 경로가 틀렸다([[81]])
(4) 기존 배터리: test_a2_three_layer.py · test_c207_envelope.py · test_lever_reachable.py
(5) 등가 게이트: 정본 A2 만 고치고 gate.json 미동기화면 FAIL ([[24]])
```

---

## 5. 실험 — 97 태스크 A/B

### 팔

```
A (대조) : 현행 sha (수리 전)
B (처치) : A + D1 + D2   (P1/P2 를 통과한 것만)
```

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

### 배치 설계 (`.151` 반납 반영)

```
엔진 2개 (.153 GPU0=8141 · GPU1=8143)
레인 = 엔진당 1개 (kvlane.sh · nb() 는 HOST:PORT 로 센다 — 포트만으론 엔진이 식별되지 않는다)
MAXB=1 · conc 2                      => GPU당 비행 sim 2
근거: kv_cache_size_tokens = 171,749 · banking sim 실측 컨텍스트 40~94k
      => 긴 태스크 2개가 겹치면 예산의 92% (2026-09-04 05:07 실측 93,827 + 69,731)
```

⚠ **긴 묶음(`long`)은 conc 1 을 고려한다** — 위 실측이 conc 2 로도 초과가 난다는 것을 보였다.

### 비용·기간 (정직하게)

```
sim 수 : 97 × 2 arms = 194 (대조군 재실행 시) / 97 (재사용 시)
처리율 : 엔진 2개 × conc 2 · conc2 실측 sim 당 12~65분 (n=3 · bank_k8143med1)
         => 낙관 4 sim/h · 비관 2 sim/h
기간   : 194 sim -> 48~97시간 / 97 sim -> 24~48시간
비용   : user-sim = openrouter gpt-5.2 ([[30]] 권장표준) — [[09]] 사용자 승인 필요
```

### 판정 기준

- 1차 지표는 **reward**(궤적 재실행 후 DB 해시 비교 · [[69]]). 집계 metric 에서 결론 직행 금지([[08]]).
- **Δ ≥ 10/97** 을 유의로 본다([[73]] 의 Δ≥4/40 관례를 97 로 환산).
- **태스크별 부호표 필수**([[70]]) — 무엇을 샀고 **무엇을 팔았나**. D1 이 표면화를 줄이므로 종결
  전 단계를 놓치는 태스크가 생길 수 있다. 그 손실을 세지 않으면 판정이 아니다.
- 우리 층 귀속은 per-step 포렌식 + 적대적 refutation 을 거친 것만 CONFIRMED([[73]]).

---

## 6. 중단 조건

| 신호 | 조치 |
|---|---|
| P0 에서 `[PROCEDURE] You are inside` 0건 | 이 문서 폐기. 원인 진술이 틀렸다 |
| P1 on/off 둘 다 루프 (R1) | D1 폐기 · *"표면화는 원인이 아니다"* 로 기록 |
| P2 어떤 K 에서도 안 끊김 | D2 폐기 · 경로 없음으로 기록 |
| 스모크 (3) 발화 0 | 배선 경로가 틀렸다. 런 금지([[81]]) |
| 태스크별 부호표에서 손실 > 이득 | 배선 철회. 끄지 말고 조건을 조정한다([[19]] · [[70]]) |

---

## 7. 실행 순서 (체크리스트)

```
[ ] 0. 진행 중 97 캠페인 완주 · 전 태그 회수 (gz -> sim_results -> git add -f -> ls-files 확인)
[ ] 1. P0  종결-후 표면화 실재 확인            <- 여기서 폐기될 수 있다
[ ] 2. P1  D1 격리 (+ 부정통제)
[ ] 3. P2  D2 격리 · K 결정
[ ] 4. 통과분만 배선 + go_stack.sh 등재 + 단위테스트
[ ] 5. 스모크 게이트 5칸
[ ] 6. x509 큐에 단계 등재 (정본 갱신 — 새 문서 만들지 마라)
[ ] 7. 97 A/B 런 (배치: 엔진당 레인 1 · MAXB 1 · conc 2)
[ ] 8. per-step 포렌식 -> 적대적 refutation -> 태스크별 부호표 -> 판정
```

---

## 8. 이 캠페인의 기준선 (2026-09-04 06:07 시점 · 진행 중)

```
arm=viewmax2 · 2026-09-03 이후
  완료 sim 55 · 고유 태스크 53/97 · pass 33 (채점분의 62%)

관측된 배치 병리 (재발 방지 대상)
  포트당 비행 토큰이 예산의 4.4배 -> Waiting 5 · KV 94% · 생성 20.5 tok/s 고착
  과부하 런 하나를 빼자 같은 엔진이 KV 30% · Waiting 0 · 42.6 tok/s
```
