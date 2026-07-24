# 순서-계획(order-planning) learn 학습벤치 설계 — 054 knowing-doing gap (2026-07-22)

> 상위 = `RESEARCH_MASTER.md`(등대). learn 축 정본 = `LEARN_RESIDUAL_MASTER_2026_07_22.md` **§2.2**(본 문서가 그 §2.2의 설계 초안).
> 선행 필독 = `C38_INDIST_GATHER_RESULT_2026_07_09.md`(데이터 타당성 게이트·D1~D4·off-policy 교훈) ·
> `E6PRIME_GATHER_LEARN_DESIGN_2026_07_08.md`(learn 설계 골격) · 데이터 템플릿 참조(읽기전용) = `scripts/distill/tau2/cfbsynth_dpo_pairs.py`(도메인-일반 익명도구·per-traj 랜덤 id·{prompt,chosen,rejected} 포맷).
> 규율: **[[11]]** 학습=학습벤치(synth)서만·banking=eval 전용·ABox-swap 전이 · **[[12]]** 표현/구조 다양성 필수 · **[[03b]]** gold-맞추기·엔진 리터럴·규칙 이식 금지 · **[[05]]** 도메인-일반 · **[[13]]** scale→learn→scaffold · **[[42]]** prior-override=SFT 설치+DPO/NPO penalty · **[[09]]** 무료검증 우선 · **[[08]]** 집계→결론 금지·격리로 판정.
> **병행 트랙 경계**: 다른 작업자가 054 scaffold 픽스(`t2_gate_patch.py`·`a2/*.json`·`run_rall*`·`probe_*`) 구현 중. 본 트랙은 learn 설계만·그 파일 불가침.
>
> **본 문서는 설계 초안이다.** 구현·데이터 제작·학습·평가는 등대 실험큐(P4/E6′ 계열) 정렬 및 사용자 승인 후.

---

## 0. 요약 — 무엇을 왜 사려는가

054는 learn 축의 **확실한 단일 활성 잔여**(`LEARN_RESIDUAL_MASTER §2.2`)이자 crossover 모트의 가장 깨끗한 실증이다. 격리 프로브가 그 성격을 **knowing-doing gap**으로 확정했다: 모델은 규칙과 인과를 **완벽히 알지만**(C1 9/9·M5 9/9), 그 지식을 **순서로 적용하지 못한다**(M1-M4 전부 0/9). 원인은 능력 부족도 규칙 무지도 아니라 **강한 경쟁 prior("보안/사기대응 먼저")가 규칙-추론을 압도**하는 prior-override([[42]]·scale-emergent)다.

사려는 것은 **054 그 자체가 아니다**([[11]] 도메인-타깃 금지). 사려는 것은 **도메인-일반 순서-계획 P-primitive** — "여러 write 요청을 받았을 때, 실행 전에 요청 간 부작용-차단 관계를 검토하여 심사-완결형(차단당하는) 작업을 먼저 배치하고, 경쟁 salience prior의 당김에 저항하는" 추론 습관 — 을 학습벤치(synth)서 SFT+DPO/NPO로 설치하는 것이다. banking 규칙은 **가르치지 않는다**. 054 eval 시 모델은 규칙을 자기 prior(C1)에서 공급하고, 설치된 순서-적용 습관과 **합성**해서 올바른 순서를 낸다. 이것이 전이 주장이다.

**정직한 한계**(master §2.2 정합): ①상식-규칙 지식 자체는 scale 몫(C1이 이미 있음을 보증). 우리가 설치하는 것은 지식→순서 **적용-추론**(C2·doing이 scale-gated인 부분)뿐이다.

---

## 1. 문제 정의 — knowing-doing gap과 prior-override

### 1.1 054 태스크 구조 (실측·[S]·`LEARN_RESIDUAL_MASTER §2.2`·tools.py 확정)
한 대화에서 고객이 세 write를 요청: ①사기 dispute 접수 ②교체카드 발급(replacement) ③한도증액(CLI approve). env 히든룰: **"pending dispute 또는 진행중 replacement가 있으면 CLI approve 거부."** 이 규칙은 **KB 698문서·도구설명·에러문구 어디에도 없다**(census·역grep 0 = agent-가시 채널 전무). gold 순서 = **CLI 완결 먼저** → replacement → dispute. dispute를 먼저 접수하면 그 sim의 CLI는 **회복 불가**.

구조 요약: 요청 A(dispute)의 **부작용**(pending dispute 생성)이 요청 B(CLI approve)의 **precondition을 위반**시켜 B를 영구 차단. 고객은 A를 먼저 시킨다. 올바른 계획 = B(차단당하는 심사-완결형)를 먼저 실행.

### 1.2 격리 실측 — dissociation 확정 (`probe_054_prior_iso`·`probe_054c_elicit`)
| 프로브 | 결과 | 의미 |
|---|---|---|
| **C1** 규칙 상식(규칙 미제공) | **9/9 "wait"** | 규칙-prior를 32B도 scale-불변으로 **완벽히 앎**(이유까지 정확) |
| **M5** 인과(dispute먼저→CLI승인?) | **9/9 "no, blocked"** | 인과도 **완벽히 앎** |
| M1 pairwise(규칙 미제공) | **0/9** | 2택 분해로도 dispute-먼저 |
| M2 pairwise(**규칙 명시**) | **0/9** | 규칙 코앞에 줘도 순서 안 바뀜 |
| M3 greedy 첫선택(규칙 명시) | **0/9** | "지금 하나만" 골라도 dispute |
| M4 chain 단계분해 | **0/9** | **(b)="CLI 불가" 자답하고도 (c)=dispute** = 자기추론 모순 |

**knowing(C1·M5) 9/9 ↔ doing(M1-M4) 0/9의 완전 해리**가 핵심 데이터. 어떤 elicitation(명시·분해·chain·greedy)도 순서로 연결 못함 = **scaffold 최종 불가**(M2/M3/M4 게이트 다 0/9). 원인 = "보안 먼저(dispute 선처리=계좌 보호)" 경쟁 prior가 규칙/인과-추론을 이김. 이것이 [[42]] prior-override의 교과서 인스턴스 — **프롬프트(규칙 명시 M2)로는 못 닫힘**·닫개는 scale 또는 weight-update(SFT 설치+penalty 억제).

### 1.3 성격 규정
- **부하 아님**: 격리(단순 프롬프트)서도 0/9 = 대화 길이 탓 아님(097/052와 다름).
- **능력 아님(순수)**: C1·M5 9/9 = 지식·인과 추론 능력 有. 결손은 **적용-연결**뿐.
- **활성화 실패 아님(038과 다름)**: 038은 명시 질의 8/8로 닫힘. 054는 M2(명시)·M3·M4 다 0/9 = 질의 게이트 무효.
- ⇒ **순수 learn/scale 영역**. 논문 crossover 모트: knowing=scale-불변·doing(적용추론)=scale-gated(frontier 054 1/4만 계획-먼저 습관으로 이김).

---

## 2. 학습 목표 행동 — 도메인-일반 순서-계획 P-primitive

### 2.1 P-primitive 정의 (`[[02]]` 생성원 대수의 flow 축)
**이름(가칭)**: `P-SEQ`(order-dependent sequencing / plan-first). content-op가 아니라 **flow 생성원**(P1-P9 계열)에 속하는 계획 primitive.

**정의**: 한 세션에서 다중 write 요청을 수령하면,
1. **실행 전** 각 요청의 부작용을 열거하고,
2. 요청 쌍 (X,Y)에 대해 "X의 부작용이 Y의 precondition을 위반해 Y를 차단하는가"를 판정하고(**interaction review**),
3. 차단 관계가 있고 Y도 여전히 요망되면, **차단당하는(심사-완결형) Y를 X보다 먼저 배치**하며,
4. 이때 "먼저 처리하고 싶은" 경쟁 salience(보안·긴급·고객우선)의 당김에 **저항**한다(prior-override 방지).

**조건부성이 핵심**(C38 §2.4 퇴화정책 교훈): 차단 관계가 **없으면** salience 순서를 유지해야 한다. 즉 "항상 덜 급한 걸 먼저"의 blanket-flip이 아니라 **부작용-차단 그래프에 따른 위상정렬(topological ordering)**을 배운다. 무조건-reorder는 SFT가 배우기 쉬운 표면 heuristic이므로(cfbsynth SFT가 "무조건 조회" 배운 것과 동형) 반드시 음성/대조로 차단.

### 2.2 054와의 연결(전이 논리·[[03b]] 준수)
- synth는 차단 관계를 **에피소드 내에서 가시**하게(또는 상식-명백하게) 제공한다 → 우리가 시험/설치하는 것은 **규칙 지식이 아니라 적용-추론**(054에서 결손인 바로 그것). banking 히든룰을 A2/데이터에 이식하지 않으므로 [[03b]] cheating 아님.
- 054 eval 시: 모델은 은닉 규칙을 **자기 C1-prior에서 recall**(scale 몫·이미 보유) + 설치된 P-SEQ 적용습관 → 합성. 규칙 recall 단계까지 synth로 강제하려면 "차단 관계 상식-암묵" 다양성 축(§3.2)이 054 히든룰 조건의 최근접 아날로그.

### 2.3 E-PLAN([[14]])과의 직렬 관계
learn(P-SEQ) = **순서 판단**(어떤 순서가 feasible한가). E-PLAN controller = **순서 집행**(결정론적 실행). 경쟁 아니라 직렬: learn이 위상정렬 결정을 내면 controller가 그 순서를 결정론으로 강제. 본 설계는 판단측만 다룬다.

---

## 3. 데이터 설계 — 학습벤치(도메인-일반)

포맷 = `cfbsynth_dpo_pairs.py` 계열 상속: 익명 도구·per-traj 랜덤 id·다도메인 표면·{prompt, chosen, rejected}(DPO) / {messages}(SFT). tau2 궤적·frontier 증류 **금지**([[11]]). banking 어휘·규칙 **금지**.

### 3.1 4-종 데이터
1. **양성 궤적(SFT positive)**: 다중 요청 → interaction review(부작용 열거·차단쌍 식별을 명시적 reasoning으로) → 위상정렬된 실행 순서 → 완주. reasoning 궤적이 "심사-완결형 먼저" 습관을 시연(demo>지시·[[42]]).
2. **변조쌍(tampered/order-flip)**: 동일 시나리오에서 **순서를 뒤집으면 차단으로 실패**하는 짝. chosen=올바른 위상순서 첫수, rejected=차단행동(부작용-생성 write) 첫수. 차이가 **첫 write 선택 토큰**뿐 → margin이 순서-결정을 정조준(cfbsynth의 "차이가 id 토큰뿐" 설계 동형).
3. **음성(prior-따름 misorder)**: 경쟁 prior("안전/보안/긴급 먼저")를 따라 차단행동을 앞세운 궤적 = **가장 강하게 penalize할 rejected**. salience를 명시적으로 심어(문구가 "urgent fraud"·"protect first"류) prior를 활성화한 뒤 그것을 벌준다([[42]] prior 억제).
4. **조건부 대조(control·필수)**: 차단 관계가 **없는** 다중 요청 → 이때는 salience 순서 유지가 정답. chosen=salient-먼저, rejected=불필요한 reorder. **blanket-flip heuristic 차단**(C38 §2.4 무조건-정책 재발 방지). 없으면 설계 무효(§7 반증).

### 3.2 다양성 축 ([[12]] 필수·단일템플릿=역전이)
| 축 | 값 |
|---|---|
| 도메인 표면(비-banking 다수) | 물류/배송, IT 프로비저닝, 여행예약, 의료 스케줄링, 법무 접수, 제조 주문, HR 온보딩, 클라우드 리소스, 재고… |
| 요청 수 / 차단쌍 수 | 2·3·4+ 요청 · 0·1·2 차단쌍(0=대조군) |
| 차단행동의 user-진술 위치 | user가 **차단행동을 먼저** 말하는 경우 다수(054 동형·prior 당김 최대) + 뒤에 말하는 경우 |
| **차단 관계 가시성** | {도구설명 명시 / 오순서-에러 명시 / **상식-암묵**(도구엔 없고 세계지식으로만)} — 상식-암묵이 054 히든룰 최근접 아날로그 |
| 경쟁 prior 유형(salience) | 보안/사기, 긴급, 고객-우선/정중, "빠른 완수 먼저" |
| 가역성 프레이밍 | 차단이 회복불가 / 회복가능(검토 필요) — 검토 습관 학습 |
| 표면 실현 | NL 렌더 다양화(동사 op-무관·render_nl_diverse 계열)·reasoning 문체 다양 |

### 3.3 배합 비율(초안·타당성 게이트 후 조정)
- SFT set: 양성 궤적 60% + **조건부 대조 40%**(무조건-reorder 퇴화 방지 위해 대조 비중 높게).
- DPO/NPO set: 변조쌍 50% + prior-따름 음성 50%(음성이 prior penalty 주신호).
- **선호 위계**(3-원): 올바른 위상순서 ≻ 임의(무관) 순서 ≻ prior-따름 순서. prior-따름을 최저(최대 margin)로 배치해 [[42]] 정합.
- 대조군(차단 0)이 전체의 ≥30% 유지 = 조건부성 보증.

---

## 4. §0 데이터 타당성 게이트 (선행 필수·이거 없이 착수 금지)

**C38 교훈**: cfbsynth는 결손 큐를 100% 제공해 base가 이미 0.98 → gradient 0 → 측정 무효. 순서-계획도 **합성이 실패를 재현하는지 먼저 확인**해야 한다. 무료·32B·수분.

### 4.1 3-검문 (전부 통과해야 학습 착수)
- **G1 base 실패 재현**: base 32B를 synth 순서-결정점에서 샘플 → **차단행동을 앞세우는(prior-따름) 비율이 높은가**(banking M1-M4 0/9에 상응하는 저조한 순서-정확도). 목표: synth order-accuracy가 **낮아야** 함(≈0.0~0.3). 이미 높으면(cfbsynth 0.98처럼) synth가 054 실패를 재현 못함 → **재설계**.
- **G2 기제 동형(knowing-doing gap 재현)**: synth 세계에서 base에 "행동 X가 Y를 차단하는가"를 **명시 질의**(M5/C1 동형) → **높아야** 함(≈0.9). G1 낮음 ∧ G2 높음 = synth가 **적용-추론 결손**을 재현(지식 결손 아님). G2도 낮으면 = synth 실패가 **지식 gap** = 잘못된 기제(지식 가르치기 = 054 gap 아님·[[03b]] 위험) → 재설계.
- **G3 gradient 존재**: G1 낮음 = headroom 있음. cfbsynth의 0.98(gradient 없음) 대척.

### 4.2 규율
- G1/G2는 **격리 프로브로 판정**([[08]]·궤적→결론 직행 금지). 054의 C1/M5-vs-M1-M4 dissociation을 **synth에 이식 재현**하는 것이 게이트의 본질.
- COMPLETION_EVIDENCE가 이 게이트서 FAIL(0/36)로 종결된 선례 = 게이트의 실효성 실증. 054 순서-계획은 **격리서 이미 0/9 실패 확인** = base 실패 실물 확보(C38이 없던 것). synth 재현만 남음.

---

## 5. 학습 방법

### 5.1 손실 조합 ([[42]] 정합)
- **SFT(순서 설치)**: 양성 궤적 + 조건부 대조로 interaction-review→위상정렬 **행동을 추가 설치**(demo=ICL 실채널). 대조 포함이 조건부성 보증.
- **DPO/NPO(prior 억제)**: prior-따름 misorder를 penalty. [[42]]: SFT=추가·DPO/NPO=prior 억제 → **둘 다 필요**(설치+억제). NPO(collapse-free unlearn)로 "보안 먼저" prior를 무너뜨리지 않고 억제.
- **선호 위계 3-원**으로 margin 구조화(올바른≻임의≻prior).

### 5.2 on-policy rejected (C38 D3·C39 필수)
- off-policy DPO는 지지집합 밖 margin = likelihood displacement 실패(cfbsynth copy 0.77→0.63 부작용). **우리 32B를 실 순서-결정점서 샘플** → 그 자연출력(prior-따름=차단행동 앞세움)을 rejected로.
- rejected 다양화(C39·32B 실패양식): {차단행동 먼저·salience-정당화 동반·"먼저 처리하겠다" 서술형} — 32B가 실제로 내는 순서-오류 양식으로.

### 5.3 32B·망각 게이트
- **32B LoRA**(우리 타깃 tier·C36 scale). 일반 tool-use/능력 데이터 혼합.
- **망각 게이트**: 학습 전후 base 능력 회귀 측정(pass^1 비열등 요구). cfbsynth SFT의 tme 13→25·A_notfound .31→.41 재발 시 즉시 중단. 대조군 포함 SFT라 무조건-정책 퇴화 위험은 낮으나 계측.

---

## 6. 평가

### 6.1 격리(무료·선행)
- **synth held-out 순서-정확도**(seed 분리·in-dist): 학습이 실제로 배웠는지(H-nolearn vs H-transfer 구분·E6′ §1 골격).
- **M1-M4 재실행**(banking 054·`probe_054c_elicit` 동형·eval 전용): 학습 후 0/9 → 상승하는가. **M5/C1 비열등**(knowing 유지 확인). **조건부 대조 프로브**(차단 없는 순서)서 blanket-flip 안 하는지.

### 6.2 e2e(유료·승인·[[09]] 최소 scope)
- **banking 054 ABox-swap 전이**(eval 전용·[[11]]): 학습된 TBox가 054에서 CLI-먼저 순서를 내는가. banking 재학습 0.
- 회귀: 다른 tau2 sim pass^k 비열등(망각 0)·Δspurious≤0(다른 태스크서 불필요 reorder로 인한 오작동 없음).
- 대조: floor 32B · floor+scaffold(순서 게이트 0/9 = 무효 확인된 상한 하한).

### 6.3 성공/실패 기준
- **GO**: synth 순서-정확도 ↑ ∧ M1-M4 054 상승(0/9→유의) ∧ M5/C1·조건부대조 비열등 ∧ 054 e2e 전이 ∧ 망각 0.
- **NO-GO**: §7 반증 중 하나라도.

---

## 7. 반증 예측 사전등록 (무엇이 관찰되면 이 설계가 틀린가)

1. **base가 synth 순서를 이미 맞힘**(G1 높음): synth가 054 실패 재현 못함 = cfbsynth 재발 → 측정 무효·설계 NO-GO.
2. **synth 명시-질의도 낮음**(G2 낮음): synth 실패 = 지식 gap이지 적용-추론 gap 아님 = 054 knowing-doing gap과 다른 기제 → 잘못된 표적(지식 가르치기 = 도메인-cheating 위험).
3. **학습 후 M1-M4 054는 오르나 조건부 대조가 떨어짐**: 무조건-reorder 표면 heuristic 설치(C38 §2.4 퇴화정책 동형) = 스킬 아님 → NO-GO.
4. **synth 순서-정확도 ↑·054 전이 0·일반 pass 회귀**: 망각/표면매핑 역전이([[12]]) → 데이터 다양성/손실 재설계.
5. **054만 오르고 다른 held-out 순서 태스크 전무**: overfit/누출 의심 → held-out 다도메인 순서 세트로 재검.
6. **frontier도 격리 M1-M4서 054 실패**(우리 32B와 무차별): crossover 서사(knowing scale-불변·doing scale-gated) 약화 → §2.2 crossover 재검(단 master §2bg는 frontier 계획-먼저 습관으로 1/4 통과를 이미 관찰).
7. **on-policy로도 DPO 부작용(copy/일반능력 저하) 재발**: prior 억제가 순서 밖 능력을 매도 = 레버 부작용(모트 §제1원리) → NPO β·데이터 혼합 재조정 후에도 지속 시 learn 강등([[13]]).

---

## 8. [[05]] 3질문 답 (고정 vs 변경 경계·상설 의무·[[17]])

**Q1. 도메인-특화 순증이 있는가?** 없음. 학습 대상 = 도메인-일반 P-SEQ(부작용-차단 위상정렬)이고 synth는 익명 도구·다도메인(비-banking 다수)·per-traj 랜덤 id. **banking 히든룰("dispute→CLI 차단")은 데이터·A2·엔진 어디에도 넣지 않는다**([[03b]]). banking-특화 요소는 오직 **eval**뿐이며, 규칙은 모델이 자기 C1-prior에서 공급(scale 몫). 변경되는 것 = **TBox weights**(학습벤치서만·[[11]]), 고정 = Scaffold 엔진.

**Q2. 유동 판단을 동결하는가?** 아니오. "CLI-먼저" 같은 도메인 순서를 룩업으로 박지 않는다. 설치하는 것은 "interaction review → feasibility-보존 위상정렬" **추론 습관**이고, **조건부 대조군**(차단 없으면 salience 순서 유지)이 판단의 유동성을 보증한다. 무조건-flip(동결형 heuristic)은 §3.1·§7-3에서 명시적으로 차단. 판단은 런타임에 부작용 그래프에 따라 매번 결정된다.

**Q3. scaffold가 도메인 행동을 수행하는가?** 아니오. 이것은 **learn(weights) 개입**이지 scaffold가 아니다. 실제로 master §2.2가 scaffold 불가를 확정(M2/M3/M4 게이트 다 0/9). E-PLAN controller는 learn이 **판단한** 순서를 결정론으로 **집행**만 하며(§2.3 직렬), 순서 판단(도메인 semantic)은 하지 않는다. 규칙 A2 이식은 [[03b]] cheating으로 금지.

---

## 9. 상태·다음

- **상태**: 설계 초안(본 문서). 데이터 미제작·학습 미착수.
- **선결(무료·[[09]])**: §4 데이터 타당성 게이트(G1/G2/G3). synth 생성기(`cfbsynth_dpo_pairs.py` 계열 신규·별 파일) 초안 → base 32B 격리로 G1/G2 재현 확인 후에만 학습.
- **실험큐 정렬**: learn 트랙 §2.2 = P4(Learned TBox Transfer)·E6′ 계열의 형제 축. 등대 실험큐 정렬 및 사용자 승인 후 구현.
- **불가침**: 054 scaffold 픽스 파일(`t2_gate_patch.py`·`a2/*.json`·`run_rall*`·`probe_054*`)은 병행 트랙 소유·읽기만.
