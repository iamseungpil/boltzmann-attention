# thinking 보강 — completion/persistence scaffold 설계 (2026-07-08 · **rev2: 외부 리뷰 반영**)

> **위치**: `QWQ_AGENTIC_FAILURE_FORENSIC_2026_07_08`(thinking 약점) + `LOAD_REDUCTION_ARCH_DESIGN §4`(E2) + DR
> (`RELWORK_AGENTIC_HORIZON`: 조기종료·external-verifier) 수렴. **rev2 = 외부 세션 설계리뷰(A~G) 검증·반영.**
> **불변**: [[05]] 도메인-일반·게이트 증식금지 · [[13]] scaffold 최소 · [[08]] per-case·offline먼저 · [[09]] 무료先
> · [[03]]#9 대칭크레딧·실측前 banking금지 · [[10]] verifier=결정론 · [[06]] **over-action=레버 금지 선례**.

---

## 0. 한 줄 (rev2 수정: 성공기준 교체)
thinking은 friction서 **policy-literal escalation/give-up**으로 완결 실패한다. 외부 결정론 완결게이트가 종료-시도를 가로챈다.
**단 게이트 자신이 역효과(over-action)를 갖는다.** ⇒ 성공기준 = ~~"QwQ+게이트 > base 0.557"~~ **철회**(nt=1 노이즈 안) →
**per-case: ①② 슬라이스 건별 복구 ∧ spurious-write Δ ≤ 0 ∧ over-block = 0.** pass 비교는 Step3(nt=4) 후 *부차* 지표.

## 0b. ★rev2 정정 원장 (리뷰 A~G 검증 결과·실측 재현)
| # | 리뷰 지적 | 판정 | 실측 |
|---|---|---|---|
| A | "over-block 19(실측)" 출처없음·benefit을 harm으로 오귀속 | **절차 ✅ / 실체 ❌** | 재실행: naive 게이트가 passing 19에 **발화**(rparser nt=1·독립측정). §6b premature-transfer 19(old qwq nt=4·benefit)와 **우연 일치**. **단 어느 doc에도 미영속 → "(실측)" 표기 부당.** 19 중 **확실 파손=transfer-type 5**뿐·나머지 14는 `nw<base-max`(나쁜 프록시). ⇒ **정밀 게이트 over-block = 미측정**(Phase A가 최초). |
| B | LOSS12가 nt=1 단일시행 위 | **✅** | n=114 1σ≈±4.7pp > 목표 gap 3.1pp. 게이트-addressable 6 → 최대 +5.3pp. **완벽 복구해도 여유 0** ⇒ pass-목표 철회. |
| C | 슬라이스 장부 오류 | **✅** | 아래 §1 재작성. addressable=**6**(8 아님). |
| D | **over-action 역풍(선례가 게이트 금지)** | **✅ 격상=1순위** | 선례 verbatim(`do NOT gate`·`레버 금지`). **실측: QwQ-rp passing-sim spurious write=0 · base=47 (≥1 spurious 보유 7% vs 17.5%)** ⇒ QwQ의 낮은 over-action은 give-up의 뒷면. 게이트가 종료를 막으면 base 영역으로 밀어냄. |
| E | offline은 GO문턱 불충분 | **✅**(미세반론: base=Qwen2.5는 `<think>` 없어 rparser 교란 아님·진짜 교란=**QwQ≠Qwen2.5**) | offline=발화(shaped)만·deny 후 재행동 불가시. |
| F | P/C 통합하되 mode별 회계분리 | **✅** | — |
| G | t49 오분류 · ③→learn이 settled NO-GO 재오픈 우려 | **✅** | §6 재작성. |

## 1. 문제 — LOSS 12 슬라이스 원장 (rev2 정정·근거=forensic §7/§7b + write/xfer 실측)
| 슬라이스 | task | 실측(writes/xfer/spurious) | 게이트 대상? |
|---|---|---|---|
| **① premature-transfer** | t13·t30·t106 | writes=[]·xfer=T | ✅ |
| **①' terminate-without-action** | **t47** | writes=[]·xfer=F (무행동 종료) | ✅ (게이트가 transfer뿐 아니라 *무행동 종료*도 커버해야) |
| **② coverage 미완** | t42(2/4)·t55(1/4) | 일부 write | ✅ |
| **③ mis-formalize** | **t87 단독** | writes=[] (목표주소 *변경*을 *검색*으로 오해) | ❌ learn(§6) |
| **operand 경계** | t6(⋈)·t58(variant)·**t49**(item-match 실패→give-up) | 1 write or 무행동 | ❌ F3 경계 |
| **⚠️ over-action** | **t5**(spurious 2·xfer=T)·**t62**(spurious 1·질문만한 주문 cancel=파괴적) | 과잉 write | ❌ **게이트가 *악화*시킬 축** |
- **게이트-addressable = ① + ①' + ② = 6** (t13·t30·t106·t47·t42·t55). ~~8~~ 철회.
- **t87 이중배정 해소**(②에서 제거·③ 단독) · **t49 → operand**(§7b "⋈/item-match 실패를 give-up으로") · **t47 편입**.

## 2. ★게이트 자신의 역효과 (rev2 신설 — 최대 리스크)
**우리 antagonism 법칙(MASTER_FRAME §3)은 게이트에도 적용된다.** 완결게이트는 *완결*을 사고 *scope 규율*을 판다.
- **기전**: 종료 차단 → 모델이 행동을 더 함 → 요청 안 한 write(over-action) 증가.
- **선례가 명시 금지**: `NEXT_DET_LEVERS:131` *"unrequested over-action = LLM scope residual (**do NOT gate**, [[06]])"* ·
  `NEXT_LEVERS:33` *"over-action(should-not intent) | 잔여(intent·over-block 위험) | **레버 금지**"*.
- **실측 기준선(rev2)**: passing sim 내 spurious write — **QwQ-rp 0 · base 47**. ≥1 spurious 보유 sim: QwQ 7% · base 17.5%.
  ⇒ **QwQ의 conservatism과 give-up은 같은 동전**. 게이트가 이를 base 쪽으로 이동시킬 구조적 위험.
- **파괴성**: t62 = 사용자가 *질문만* 한 주문을 cancel(06-NOW "파괴적 spurious"). over-action은 pass 손실이 아니라 **실제 피해**.
- ⇒ **필수 계측**: `spurious_write_count`(gold action에 없는 write) · **GO 조건에 `Δspurious ≤ 0` 추가**. 없으면 게이트가
  pass를 살리며 파괴적 write를 늘리는 것을 못 본다.

## 3. 공통 primitive — entity-coverage (도메인-일반·유지)
①①'② 공통: 사용자 관련 엔티티를 **결정론 enumerate → 커버리지 추적 → 미커버 상태의 종료-시도를 deny**.
- ① ①' = **inspection 커버리지**(포기 전 후보를 다 *봤나*) · ② = **action 커버리지**(완료 전 대상을 다 *처리했나*).
- working-set = `coverage_spec{entity, predicate_source}` A2-구동 · **enumerate=코드·predicate formalize=LLM·gold 미접근**.
- **통합(리뷰 F)**: 단일 게이트 kind=`coverage`·**mode ∈ {inspect, act}**. 메커니즘 통합·**over-block/spurious 회계는 mode별 분리**
  (안 그러면 회귀 원인을 못 가림).

## 4. Gate mode=inspect (①·①')
- **트리거**: `transfer_to_human_agents` 시도 **또는 무행동 종료**(t47류 — transfer만 보면 놓침).
- **DENY 조건(보수)**: 후보 엔티티 중 **미검사** 존재 ∧ 종료사유가 "not found/can't" 계열.
- **행동**: deny + regen 넌지 · `gen_gated` deny경로.
- **anti-cheat**: **transfer가 gold인 task 절대 미차단**(실측: passing 중 transfer 5건 = t10·t12·t25·t46·t50). 전 후보를 이미
  검사한 뒤의 transfer는 **통과**(정당 escalation). 애매 → gate off.

## 5. Gate mode=act (②·E2 재사용)
- **트리거**: user turn에 **명시 universal 양화사**(all/both/every/each … orders/items) — 좁게.
- **DENY**: 종료 시도 시 working-set에 **미-action 엔티티 잔존**.
- **anti-cheat**: 명시 양화사 + 올바른 엔티티레벨만 · 암묵 scope(단수·"고쳐줘")=미발화 · predicate 애매→gate off.
- ⚠️ `nw < base-max`류 프록시 **금지**(valid 짧은 해법 오판·A의 14건 원인).

## 6. 범위 밖 (rev2 정정)
- **③ mis-formalize = t87 단독**. → learn 귀속. **★settled 관계 명시(리뷰 G)**: MAKEORBREAK가 확정한 NO-GO는
  **τ²-타깃 faithful-formalize SFT**([[03]] 재유도 금지 대상)이다. 여기서 말하는 learn은 **four-bench 도메인-일반 TBox 학습 →
  τ² ABox-swap 전이**([[01]]/[[11]]·τ² 미학습)로 **다른 대상**이다. 이 구분 없이 ③을 "learn"이라 쓰면 settled 음성의 조용한
  재오픈이다. **본 설계는 ③에 대해 어떤 learn도 GO하지 않는다** — 경계 표기만.
- **operand 경계(t6·t58·t49)** = F3. 게이트 아님.
- **over-action(t5·t62)** = 게이트 금지 축([[06]] 선례). **게이트의 부작용 계측 대상**(§2)이지 타깃 아님.
- prompt-only persistence 넌지 단독 금지([[42]]·DR self-reflection 무효). 넌지는 deny와 결합해서만.

## 7. anti-cheating / 도메인-일반 감사 ([[05]])
1. enumerate = agent 보유 정보(DB fetch·present)만 · gold/eval 미접근.
2. 게이트 = **종료 차단만**(답·operand 미주입).
3. 트리거·enumerate·넌지 = 도메인-일반. `grep "if domain"=0`.
4. 도메인 추가 = `coverage_spec` A2-swap만.

## 8. 측정 프로토콜 (rev2: 지표·문턱 재정의)
**지표(전부 mode별 분리 회계)**:
- (a) **recall**: ①①'② 6-task에서 게이트 정확 발화
- (b) **over-block**: passing control에서 **부당 deny** — ★**현재 미측정**(Phase A가 최초). naive 프록시는 19 발화(확실 파손 5).
- (c) **★Δspurious_write ≤ 0** (게이트 전후·비요청 write 증가 금지·§2)
- (d) **turn/step 예산**: 전수검사 강제가 예산 초과를 유발하는지(QwQ tool-call 6.3 < base 8.4 — 여유 적음)
- (e) flail: deny 후 무의미 반복

| Phase | 무엇 | 볼 수 있는 것 | 볼 수 **없는** 것 |
|---|---|---|---|
| **A offline**(무료·replay) | 게이트 조건을 기존 궤적에 얹기 | (a) recall · (b) **발화**(shaped) | deny 후 재행동 ⇒ **(c)(d)(e) 불가·"over-block=0"도 shaped** |
| **B live-smoke**(무료 user-sim) | 배선 후 fail-set 6 + control(passing 10·transfer-gold 5 포함) | (b)(c)(d)(e) **closed** · 건별 복구 | pass 통계(표본 작음) |
| **C 확인**(유료·승인) | **nt=4 필수**(nt=1은 3pp 미분해) · 대조=동일 프로토콜 base | pass^1..4 | QwQ≠Qwen2.5 교란은 잔존 |

## 9. GO / NO-GO (rev2: shaped/closed 분리·성공기준 교체)
| Phase | GO 조건 | 실패 시 |
|---|---|---|
| A(shaped) | recall 유의(6 중 다수 발화) ∧ 발화가 passing-transfer-gold 5건을 **비-발화** | 트리거 축소·재측정 |
| **B(closed·진짜 문턱)** | **건별 복구 > 0 ∧ over-block = 0 ∧ Δspurious ≤ 0 ∧ turn-예산 초과 0** | 게이트 **폐기 or mode 분리 재설계** |
| C | (부차) pass^1..4 개선 — **주장 근거 아님**(노이즈) | — |
- **핵심**: 성공은 **per-case 메커니즘 주장**(노이즈-robust)으로 판정. pass 비교는 Step3 nt=4 이후 보조.

## 10. base와의 관계 (일반성)
mode=act(②)는 원래 base coverage 잔여(M1)용 설계. thinking은 같은 클래스를 증폭. ⇒ 동일 도메인-일반 게이트가 양쪽 커버 =
QwQ-특이 패치 아님. **단 base는 이미 over-action 47(passing)** — base에 게이트를 얹으면 Δspurious 위험이 더 큼. E4(base 회귀)서 필수 계측.

## 11. 리뷰 대기 (rev2 갱신)
1. **exhaustion 정의**(mode=inspect): "후보 엔티티"를 user의 전 orders로? 요청 품목 포함가능 orders로? → Phase A가 답.
2. ~~P/C 통합?~~ **채택**(kind=coverage·mode 2종)·**회계는 mode별 분리**(리뷰 F).
3. override 강도: deny+regen 유지.
4. **over-block 허용치 = 0 strict**(transfer-gold 5건 파손은 즉시 폐기 사유).
5. **★Δspurious ≤ 0을 GO에 넣었으나, 게이트가 원리적으로 over-action을 유발한다면 설계 자체가 자기모순인가?** — 열린 질문.
   완화안: deny를 "행동 강요"가 아니라 **"검사 강요"(mode=inspect)** 로 한정하면 over-action 압력이 낮음. mode=act(행동 강요)가
   위험. ⇒ **Phase B를 mode=inspect 먼저** 권장.
6. 테스트베드: QwQ 먼저·GO 시 base 회귀(E4·Δspurious 필수).

## 12. 시퀀싱 (rev2)
1. **[먼저] 장부 정정 완료**(본 rev2) — 틀린 타깃집합에 recall 재는 것 방지.
2. **Phase A offline(무료)** — recall + shaped 발화 · transfer-gold 5건 비-발화 확인.
3. **Phase B live-smoke(무료)** — **mode=inspect 먼저**(over-action 압력 낮음) → Δspurious·over-block·turn 예산 closed 측정.
4. mode=act는 B-inspect가 Δspurious≤0을 보인 뒤에만.
5. Step3(nt=4) 회수 후에야 pass 비교. GO → Phase C(유료 1회·승인).
