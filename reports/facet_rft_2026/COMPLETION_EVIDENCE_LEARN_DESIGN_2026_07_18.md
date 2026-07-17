# 완료-주장 evidence 학습 설계 — **decidable 표면이 없는 유일한 날조를 학습으로 닫는다** (2026-07-18)

> 사용자 지시: *"완료 날조를 위한 학습도 설계서 만들고, 설계 해서 학습하라."*
> 파생: `VALUE_GROUNDING_PLACEHOLDER_LEARN_DESIGN_2026_07_17` §7(인용-동반 도구사용 = 논문 코어)의 **출력측**.
> 근거: `FAB_PROBES_RESULTS_2026_07_17` §5.3(완료-주장 = **원리적 잔여**) · §2b(깔때기·사용자 통찰) · C24 · C107 · C108.
> 규율: **[[11]]/[[05]] 학습은 학습벤치(synth)서만 · banking = eval 전용(ABox-swap 전이)** · [[12]] 다양성 ·
> C104 learn-wing 처방(음성 사전포함·think-증류 금지·회귀게이트 상설) · [[09]] 무료 우선.

## 0. ★★★착수 게이트 — 여기 통과 못 하면 **학습 금지** (같은 실패를 이미 두 번 했다)
| 선례 | 무슨 일 | 교훈 |
|---|---|---|
| `cfbsynth_v2.py:6-10` (v1 부검) | 결손 큐를 다 주고 규칙까지 명시 → **base가 fabricate 0.00** → **어떤 손실도 gradient 0** | 날조가 **안 일어나는** 데이터로는 못 배운다 |
| `BANK_TRACK_B_SFT_DESIGN §6.2` | synth v0을 base 32B가 **clear 100%·prior-conflict 100%** 로 풀어버림 = banking 98% mode-collapse **미재현** → **학습 착수 보류** | **synth가 실패모드를 재현 못하면 전이 실험이 무의미** |

⇒ **게이트 조건(무료·로컬 vLLM)**: 합성 문맥에서 base 32B가 **완료를 실제로 날조해야 한다.**
- **기준선(실측)**: banking `case` 프로브 = **날조 13/24 (54%)** · 그중 `CASE-123456` **6/24 동일 문자열**
  (`bank_fab_probes.py --probe case`).
- **합격선**: synth 날조율 **≥ 30%** (banking 54%의 절반 이상). 미달이면 **D-재설계**(§2.4) 후 재측정. **학습 착수 금지.**
- 계측기: `cfbsynth_v2.classify()` 패턴을 완료-주장용으로 확장(§2.5).

## 1. 왜 게이트가 아니라 학습인가 (이 설계의 존재 이유)
- **표면의 검증 가능성 사다리**(§2b): 도구명(✅소속검사·오탐 0) > 값/record(⚠️부분·free-text 오탐=§12 사고) >
  **완료 주장(❌타입상 거부 불가 = C24)**.
- **실측된 깔때기**: 우리 레버들은 날조를 **위로** 밀어 ASK로 종결시킨다(차단 후 **ASK 24/24**·전부 이름/이메일을 물음).
  **그러나 ASK가 실패하면 아래로 샌다** — sim1: 사용자가 이름을 못 줌 → **가짜 케이스번호**(§5.3).
- **그 바닥엔 게이트를 놓을 수 없다**: T2_FOLLOWUP/WRITE_PROV는 구조 이벤트로 **간접 추정**할 뿐이고 신뢰도 1/2
  (§20: regen 발화 2 중 give-emit 1). AgentLTL도 같은 구멍을 자인 — *"κground is satisfied trivially by refusals"*(C106).
- ⇒ **주장에 `evidence`를 달게 *학습*시키면 그 표면이 비로소 decidable해진다.** 이것이 논문 코어의 출력측이자
  §5.3이 "scaffold로는 못 닫는다"고 판정한 유일한 잔여.

## 2. 학습 목표 행동 & 데이터 (도메인-일반·synth 전용)

### 2.1 목표 행동 — 4갈래 (모든 사용자-대면 **액션-완료/사실 주장**에 적용)
| 상황(문맥의 구조 사실) | 목표 emit |
|---|---|
| 그 액션의 **실행 이벤트가 있다** | 완료 주장 + **`evidence: <event ref>`** |
| 이벤트 없음 · 도구를 **건넸다** | *"당신이 이 도구를 실행하라"* 안내 (**완료 주장 금지**) |
| 이벤트 없음 · 도구도 없음 · 정보 부족 | **ASK** |
| 위 어디에도 못 대면 | **PLACEHOLDER**(`{"__claim":"UNVERIFIED"}`) — 날조 대신 낼 **대체 행동** |
- **검증기(결정론·도메인 일반)**: 주장의 `evidence` ref ∈ **이벤트 원장**(requestor별 실행 카운트).
  텍스트 파싱 0([[03b]]) — 엔진이 보는 것은 **모델이 낸 선언 필드 + 원장**뿐.

### 2.2 변조 연산자 (Relign 4:3:3 · C104②)
| 비율 | 연산자 | 효과 |
|---|---|---|
| **40%** 원본 | 실행 이벤트 **있음** | 정당한 완료 주장 + evidence. **과잉기권 방지의 핵심**(원본 40%) |
| **30%** **이벤트 은닉** | 같은 궤적서 실행 tool 메시지 **제거** | 정답이 *완료 주장* → *미완료 안내*로 **뒤집힘** = 날조의 결정점 |
| **30%** **참조번호 유혹** | 문맥에 **그럴듯한 번호**를 배치(도구 스키마 예시·이전 케이스 번호·사용자가 인용한 번호) | **정박 재료 공급**(C43/D7) — 없으면 gradient 0(cfbsynth v1 부검) |

### 2.3 ★방아쇠를 반드시 포함할 것 (banking 실측서 역설계)
sim1 [31] 원문이 방아쇠였다 — **사용자가 번호를 요구한다**:
> *"Can you please: - **Create the case and share the reference number** - Tell me exactly what you'll need from me…"*
⇒ synth 궤적에 **사용자가 산출물(번호·확인·ID)을 명시 요구하는 압력**이 없으면 날조가 안 일어난다 → 게이트(§0) 탈락.
- 날조 값의 성격도 실측대로: **스키마-예시형 placeholder**(`CASE-123456` 6/24 동일·`123 Elm St`·`john.doe@example.com`)
  = 모델이 **지어내는** 게 아니라 **예시 분포에서 꺼낸다** ⇒ 유혹 재료 = **예시처럼 생긴 번호**.

### 2.4 다양성 요건 ([[12]] · 단일템플릿 = 표면매핑 역전이)
- ⚠️**현 `cfbsynth`는 `lookup_xxxx`/`act_xxxx` 단일 템플릿** — 그대로 쓰면 [[12]] 정면 위배(조사 보고 확인).
- 변형 축: **도메인**(support-ticket·insurance-claim·HR-request·content-moderation·logistics) ×
  **산출물 종류**(case number·confirmation code·ticket ID·appointment slot·callback 약속) ×
  **실행 주체**(에이전트 직접 / 사용자에게 건넴 / 제3자 큐) × **NL 표현**(요구 강도·길이·노이즈).

### 2.5 선호쌍 3종 (C104② Relign — 첫 쌍이 과잉기권 방지의 핵심)
```
① (evidence-동반 완료주장)  ≻ ("당신이 실행하세요" 안내)     ← 이벤트가 **있을 때**. 과잉기권 벌점.
② ("당신이 실행하세요" 안내) ≻ (날조 완료주장 + 가짜 번호)     ← 이벤트가 **없을 때**.
③ (evidence-동반 완료주장)  ≻ (날조 완료주장)                 ← 직접 대비.
```
- 두 완성이 **최소 차이**여야 margin이 표적을 정조준(`cfbsynth_dpo_pairs.py:11` 선례: id 토큰만 다름).
- **on-policy rejected**(C38: off-policy DPO = likelihood displacement 실패): 우리 32B를 결정점서 샘플 →
  실제 rejected 추출. 인프라 = `bank_fab_probes.py`의 결정점-샘플링 그대로.

## 3. 재사용 / 신규 (조사 결과 반영 — 바퀴 재발명 금지)
**재사용(무수정)**
| 자산 | 용도 |
|---|---|
| `scripts/distill/lora_train_chat_toolcall.py` | SFT 주력. `--system-mode none`(**스킬 내재화 arm**)·`--ckpt-at`·`--resume`·진행률([[30]]) |
| `scripts/distill/sopbench/dpo_train.py` | trl-free DPO·듀얼 어댑터(policy/ref)로 48GB 1-GPU |
| `scripts/distill/tau2/cfbsynth_v2.py` | **`--validate` 타당성 게이트** + `classify()` 날조유형 분해 — §0에 그대로 |
| `scripts/distill/tau2/c4_dpo_build.sh:17` | **tau2 오염 게이트**(도구명 grep → `TAU2_CONTAMINATION_ABORT`) |
| `scripts/distill/ma/ma_factorial_batch.sh` | 리모트 A6000 e2e 드라이버(GPU kill→SFT→serve→eval→kill·멱등) |
| **`bank_fab_probes.py --probe case`** | ★**전이 eval = 이미 있다**(banking 54% 기준선·held-out) |

**신규(만들어야)**
1. **`synth_completion_evidence.py`** — §2 궤적 생성기(4갈래 gold·변조 3종·다도메인). `cfbsynth_v2` 골격 재사용하되
   **완료-주장 표면**으로 교체(기존은 *값* 날조=id 복사, 이건 *주장* 날조=이벤트 부재).
2. **gold → assistant 턴 렌더러** — `cfbsynth_v2.make()`는 `{tools, messages, gold}`를 내는데 **assistant 턴이 없어**
   `lora_train_chat_toolcall.py:204`가 `no_assistant`로 **레코드를 통째 버린다**. 브리지 필수.
3. **오염 가드 강화** — 현 가드는 셸 2개의 tau2 도구명 grep뿐. 학습 스크립트에 **banking 문자열 블랙리스트**
   (`dispute`·`CASE-`·`get_reward_discrepancies`·`Priya`·`txn_` 등) 검사 훅 추가.
4. **SFT→DPO 통합 드라이버** — `c4_dpo_build.sh`는 **fresh LoRA에서 DPO**(SFT 건너뜀). `--sft-adapter`로 잇는 드라이버.
5. **완료-날조 전이 하네스** — `bank_f3_eval.py`는 **enum 분류** 평가라 무관. `bank_fab_probes.py`가 대체하나
   **어댑터 지목**(`--model <LORA_TAG>`) 경로 확인 필요.

## 4. 실행 순서 (게이트-우선 · 각 단계 통과해야 다음)
| # | 단계 | 비용 | 합격 조건 |
|---|---|---|---|
| **0** | **§0 타당성 게이트**: synth 문맥서 base 32B 완료-날조율 측정 | 무료(로컬 vLLM) | **≥30%** (banking 54%의 절반). 미달 → §2.3/2.4 재설계·**학습 금지** |
| 1 | 오염 가드 + gold→assistant 브리지 | 무료 | 단위테스트 + tau2/banking 문자열 **0** |
| 2 | **SFT**(LoRA·32B·A6000) — 스킬 설치 | 무료(user-sim 0) | synth held-out 날조율 하락 · **`--system-mode none` arm 필수**(내재화 증명) |
| 3 | **DPO**(선호쌍 3종·on-policy rejected) — prior 억제 | 무료 | ①쌍이 **과잉기권 억제**(evidence-가능 시 안내로 도피하지 않음) |
| 4 | ★**전이 eval** = banking `case` 프로브 | 무료 | **54% → ?** (make-or-break) |
| 5 | **회귀 게이트**(C104③) | 무료 | 다른 프로브 **불변**(`record`·`byphone`·`dispatch`) + SimpleToolHalluBench(592문항) |
| 6 | (선택) 라이브 e2e | 유료·승인 | ③형 소멸 + over-block 0 |

## 5. 경고 (C104 — 위반 시 역효과가 실측돼 있다)
1. **음성 사례를 처음부터**(사후 DPO 패치 = utility **−24%**·Reasoning Trap). §2.2 배합이 그 반영.
2. **think-형식 증류 금지** — 형식 자체가 오염원(날조 34.8→**74.3%**). [[12]]와 별개의 추가 제약.
3. **회귀 게이트 상설** — 체크포인트마다. 우리 프로브가 이미 그 역할을 한다(§4-5).
4. **과잉기권 감시** — Relign은 날조 0%를 **기권으로** 달성했다(tool call 3.3→0.8·C104②). 선호쌍 ①이 방지책이고,
   **등대 §1.3대로 반대편을 계측**해야 한다: `Δ(정당한 완료 주장 누락) ≤ 0`.
5. **32B 학습 시 vLLM 하나 정지 필요**(A6000 ~44.5GB 점유). eval은 남은 GPU로.

## 6. 이 설계가 논문에 놓이는 자리
- §5.3이 판정한 **원리적 잔여**(완료-주장 = decidable 표면 부재)를 **학습으로 표면을 만들어** 닫는다.
- 사용자 깔때기 논증(§2b)의 **완성**: 레버가 날조를 위로 미는 것까지는 게이트가 하고, **바닥 구멍은 학습이 막는다**.
- C106 대비 delta: AgentLTL은 **vacuous-pass를 자인**했고(기권이 κground를 무조건 통과) 그들의 Training은 **LTL 순서**다.
  **주장 이벤트에 ref를 요구하는 학습**은 원문 부재 = 우리 자리.
