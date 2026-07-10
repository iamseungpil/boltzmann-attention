# E6′ — gather-before-act 학습 재설계 (2026-07-08 밤)

> 상위 = `RESEARCH_MASTER.md`. 선행 = `SCAFFOLD_AUDIT_RULE0_2026_07_08.md`(꼼수 색출) ·
> `C4_LEARN_FETCHFIRST_CROSSOVER_DESIGN_2026_06_22.md`(원 설계) · `cfbsynth_dpo_pairs.py`(데이터).
> 사용자 지시: *"명확한 원인을 scaffold 꼼수로 피하지 말고 학습으로 보강하라. gather 학습은 부작용이 거의 없다."*

---

## 0. 착수 전 정정 — 이건 새 실험이 아니라 **실패한 실험의 재시도**다

**C37 철회(자기교정 #14).** "gather 학습은 시도된 적 없다"는 틀렸다. `cfbsynth_dpo_pairs.py`는 이미
`gather` 페어(값 없음 → **getter 호출**이 chosen, 예시값 consumer가 rejected)와
`copy` 페어(값 있음 → 실값 consumer가 chosen)를 **둘 다** 만든다. 조건부 양쪽을 가르쳤다.

### 0.1 실측 재검 — ★"역전이"는 성립하지 않는다 (자기교정 #15)

처음에 나는 pass^1로 "학습이 역전이했다"고 단정했다. **[[08]] 종료사유 전수 검문에서 무너졌다.**

**(i) C4의 pass^1은 해석 불가.** 전 arm sim=114. `too_many_errors`(=행동 실패)와 infra가 arm마다 다르고,
무엇보다 **동일한 7B base가 두 런서 pass 21 vs 32**로 갈린다(`c4ff_retail_base` vs `c4dpo_retail_base`,
같은 모델·같은 도메인·af=0·nt=1·gpt-4.1 user-sim). **학습 효과 전체가 이 편차 안에 있다.**

**(ii) 기전 *비율*(분모=nfail)로 보면 DPO는 표적을 눌렀다.**

| arm | nfail | `no_gather` rate | **`schema_copy` rate** | pass/114 | too_many_errors |
|---|---|---|---|---|---|
| dpo base | 82 | .439 | **.439** | 32 | 12 |
| **DPO `dpo-pure`** | 93 | .398 | **.376** | 20 | 13 |
| DPO `dpo-deny` | 89 | .404 | .404 | 23 | 23 |
| **autofetch(꼼수) `dpo-perform`** | 84 | **.286** | .274 | 25 | 22 |

- **DPO는 자기가 `rejected`로 벌준 것(스키마 예시 복사)을 실제로 −6.3pp 눌렀고, 도구호출은 무손상**(tme 12→13).
- **SFT(`learn`)는 진짜로 해로웠다**: A_notfound .31→**.41**(`learn-pure` **.49**) · tme 13→**25**.

⇒ **정확한 진술**: *SFT는 해로웠다. DPO 경로는 살아 있고 표본이 부족했을 뿐이다.*
사용자의 *"gather 학습은 부작용이 거의 없다"* 는 **DPO에 한해 현 데이터와 모순되지 않는다**(tme·기전 모두 무손상).

### 0.2 ★C39 — DPO는 `rejected`에 넣은 것만 배운다
`rejected` = 스키마 예시값 ⇒ 줄어든 것도 `schema_copy`뿐(−6.3pp), 일반 날조는 거의 그대로.
그런데 **32B의 실패 양식은 예시복사가 아니라 발명**이다(93건: 예시복사 18 · **발명형 10자리 id 48** · 조합형 16 · C36).
**7B용 rejected 집합은 32B의 실패를 벌하지 않는다.** 이것이 E6′의 첫 번째 수정점이다.

## 1. ★E6′의 선결 측정 (C38·이거 없이는 착수 금지)

> **DPO/SFT 모델의 in-dist(합성 held-out) gather 정확도가 로그에 없다.**

두 가설이 관측적으로 구분되지 않는다:
- **H-transfer**: 합성서는 배웠는데(in-dist ↑) tau2로 안 넘어갔다 ⇒ [[12]] 다양성/구조 문제.
- **H-nolearn**: 애초에 못 배웠다(in-dist flat) ⇒ 데이터·손실 설계 문제.

**처방이 정반대다.** 먼저 `dpo`·`c4ff` 체크포인트로 **held-out 합성 gather 정확도**를 잰다(무료·GPU 1대·수분).
- H-nolearn이면 → 데이터/손실 재설계.
- H-transfer면 → 아래 §2.

## 2. 전이 실패의 후보 원인 (H-transfer 가정 시)

| # | 후보 | 근거 | 처방 |
|---|---|---|---|
| T1 | **구조 격차**: 합성=1턴 hermes 텍스트 / tau2=30턴 native FC·16 도구·사용자 대화 | `cfbsynth_dpo_pairs.py:12` `_tc()` = 텍스트 tool_call | 다중턴·native FC·distractor 도구·사용자 개입 포함 합성 |
| T2 | **망각**: DPO 3000쌍 1턴 텍스트 → 30턴 에이전트 분포 붕괴 (`dpo-pure` 32→20) | pass^1 대폭 하락 | 일반 능력 데이터 혼합 · KL/β 상향 · LoRA 소용량 |
| T3 | **표적 오조준**: rejected가 *스키마 예시값*뿐 | 32B 날조 93건 중 예시복사는 **18**, 그럴듯한 id **발명 48**(C36) | rejected에 **발명형 id**(랜덤 실재-형태) 추가 |
| T4 | **scale**: 7B에서만 측정 | 미조회 날조 7B 38.8% → 32B 6.7%(C36) | **32B에서 재측정** (우리 타깃 tier) |

★T3는 이번 감사에서 처음 나온 것이다. **7B는 예시를 복사하고 32B는 id를 발명한다**(C36).
7B용 페어(rejected=예시값)는 **32B의 실제 실패 양식을 벌하지 않는다.**

## 3. 설계 (수정)

### 3.1 데이터 (`cfbsynth_dpo_pairs.py` v2 — 도메인-일반 유지)
- **rejected 3종**으로 확장: ① 스키마 예시값(기존) ② **랜덤 실재-형태 id 발명**(T3·32B 양식)
  ③ **조합형 placeholder**(`{real_id}_cheapest`류·32B서 16건)
- **구조 다양성**(T1·[[12]]): 다중턴 · native function-calling 포맷 · distractor 도구 5~15개 · 사용자 개입 턴
- **음성 조건부 유지**: 값이 이미 문맥에 있으면 chosen=바로 write (재조회는 rejected) — **"항상 읽어라" 퇴화 방지**
- ⛔ tau2 궤적·frontier 증류 **금지**([[11]]). tau2 = held-out.

### 3.2 학습 (T2)
- **32B LoRA**(T4) · 일반 tool-use 데이터 혼합 · DPO β 상향 or SFT+NPO
- **망각 게이트**: 학습 전후 **base 능력 회귀 측정**(pass^1 하락 0 요구). `dpo-pure` 32→20 재발 시 즉시 중단.

### 3.3 평가 (규칙 0 엄수)
- **모든 supply 꼼수 OFF**: `T2_PRESENT_READS=0` · `T2_AUTOFETCH=0`(영구 금지·C34).
- 주 지표 = **미조회 날조율**(C29 술어) · A_notfound · reads/sim. 부차 = pass^k.
- 대조군: floor · floor+E11-a · floor+autofetch(**상한 참조용**·꼼수임을 명시).

## 4. 성공/실패 기준
- **GO**: in-dist gather ↑ **∧** tau2 미조회 날조율 **6.7% → frontier(0.0~0.3%) 방향으로 유의 감소**
  **∧ 망각 0**(pass^1 비열등) **∧ reads/sim 증가**.
- **NO-GO**: 역전이 재발(A_notfound ↑) 또는 망각. 이 경우 **§0.1이 두 번째 독립 반증** ⇒
  gather는 *학습으로 싸게 사지지 않는* 축으로 [[13]] 우선순위에서 강등하고, C36의 scale-불연속(32B 6.7% vs frontier 0.0%)을
  **경계(boundary)** 로 등재한다.

## 5. 순서 (사용자 승인 = a → b)
1. **(a) E11-a 격리 프로브** (실행중·무료·32B) — deny/hint가 32B서도 무효인지. 무효면 scaffold 경로 종결.
2. **(b) E6′** — 단 §1의 선결 측정(in-dist gather 정확도) **먼저**. H-nolearn/H-transfer 판정 후 §3.
