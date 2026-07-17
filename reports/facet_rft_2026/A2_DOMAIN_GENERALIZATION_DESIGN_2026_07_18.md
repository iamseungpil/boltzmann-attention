# A2 도메인 일반화 설계 — **task_019 하드코딩을 env 열거로** (2026-07-18)

> 지시: *"A2로 97개 태스크 모두 동작하게 하라."* 목적 = **리더보드 비교 가능한 banking 점수**(97 태스크·`alltools`).
> ⚠️**§8 하드룰 1**: 랭크 前 **반증 예측**(§5). ⚠️**[[05]]**: A2=ABox(가변) 허용이나 **task-특화는 정답 주입**이다.

## §PROD. ★★producer op는 도메인 일반화 **불가** — 규칙이 KB-분산·시간-조건부다 (2026-07-18·97런 per-step 확정)
> 계기: 97런 실패 per-step 포렌식(사용자 지시) → 실패 주축이 verify/give가 아니라 **`get_reward_discrepancies`가
> 틀린 거래집합을 냄**. 사용자 판단 = **①KB에서 규칙 도출해 op `cases` 확장**. ⇒ **데이터가 ①의 전제를 반증**(§8 하드룰).

**① 증상**(task_026·task_020 등 reward-discrepancy 태스크군): producer가 gold와 **다른 거래집합**을 discrepant로 판정.
task_026: gold 대상 4건 vs 우리 반환 6건 = **교집합 1건**(`f506`)뿐. `reward_basis=DB`라 거래 틀리면 fail.
에이전트는 결백 — producer가 준 걸 충실히 제출/실행함(give·user-exec 다 정상).

**② 원인**: op `cases`는 **카드 이름만** 본다(`Silver=4·Gold=2.5…`). task_026 입력의 `Business Silver Rewards Card`가
없어 `default:null`. ⇒ 표를 채우면 될 것 같으나 —

**③ ★반증**(gold 수치 ÷ amount로 배율 역산): **같은 `Business Silver / Travel`이 배율 20 vs 10으로 갈린다.**
| card / category | amount | gold | 배율 |
|---|---|---|---|
| Business Silver / Travel | 315 | 6300 | **20** |
| Business Silver / Travel | 380 | 3800 | **10** |
카드×카테고리 2차원으로도 rate가 안 정해진다.

**④ ★무엇이 20과 10을 가르나 = KB의 시간-조건부 프로모**(축자):
> *"New Business Silver Rewards Card customers can earn **2x cash back on all purchases for the first 6 months** …
> Start date: 2024-11-14, End date: **2025-11-14**"* (`Double Cash Back Promo` 문서)
⇒ rate = 기본 10% **× (거래일 ∈ 프로모기간 ∧ 신규고객이면 2배)**. **최소 4차원**(카드·카테고리·**거래날짜**·**계정자격**)·
여러 KB 문서 분산·**시간 조건부**.

**⑤ ⇒ [[05]]/[[03b]] 판정 — op로 담을 수 없다**:
- gold 수치로 표를 채우면 = **정답 주입([[05]] 위반)**.
- KB 산문의 프로모 조건을 **엔진이 파싱**해 채우면 = **엔진-formalize([[03b]] 위반)** + 날짜/자격 대조는 **유동적 판단**.
- ⇒ **[[05]] 정본 "결정론↔유동성" 경계 그대로**: 이 rate 판정은 KB를 읽어 프로모기간·자격을 대조하는 **유동적
  판단**이라 **결정론 offload가 아니라 모델의 몫**이다. task_019는 마침 프로모 없는 단순 케이스라 작동했을 뿐.

**⑥ ★측정된 결론(부정적이나 값짐)**: ***우리 결정론 offload(producer)는 규칙이 문맥-독립일 때만 작동하고,
banking 보상률처럼 KB-분산·시간-조건부 규칙은 담을 수 없다.*** task_019 성공 = 과적합의 우연.
⇒ **producer 확장 폐기**(①·②·③ 다). reward-discrepancy 태스크군은 우리 결정론 레버가 원리적으로 안 닿는 곳.
- **잔여 정당한 기여**(변함없음): verify(84%)·give/완료(22%)·(a1)·TOOLGATE — **compute가 아닌 표면들**.
- **[[45]] 대조**: 이건 scale이 못 푸는 부하가 아니라 **offload가 못 담는 부하**(KB-문맥 의존)다. 새 축.

## 0. ★조사 결과 — **A2는 생각만큼 task-튜닝돼 있지 않았다** (무료·env/tasks 직독)
### 0.1 env 구조 (banking)
| | 값 |
|---|---|
| `@is_discoverable_tool` 전수 | **48** (WRITE 34 · READ 11 · GENERIC 3) |
| ★**agent-side**(`KnowledgeTools`) | **44** — unlock → `call_discoverable_agent_tool` |
| ★**user-side**(`KnowledgeUserTools`) | **4** — `submit_cash_back_dispute_0589` · `get_referral_link` · `get_card_last_4_digits` · `deposit_check_3847` |
- **구분이 클래스-구조적**(축자 주석): *"These tools represent actions users take in the real world. The agent
  **gives them to the user** via `give_discoverable_user_tool`, and the user calls them directly.
  They are **NOT included in the default tool list**."*
- ★**엔진이 물어볼 수 있다**: `env.user_tools.get_discoverable_tools()`
  (`ToolKitBase.get_discoverable_tools` = `DISCOVERABLE_ATTR` 필터) ⇒ **도메인 리터럴 0.**
- ★**`mutates_state` 속성 존재**(`= tool_type == ToolType.WRITE`) ⇒ **"상태 변경 실행이 있었나"를 구조적으로 판정 가능.**

### 0.2 97 태스크가 실제로 요구하는 것 (gold `evaluation_criteria.actions` 전수)
| 요구 | 태스크 수 | 우리 레버 |
|---|---|---|
| **`log_verification`(신원확인)** | **81/97 = 84%** | ★**GB1 게이트 + `verify_identity`** — **이미 도메인 일반** |
| agent-side unlock/call | **69/97 = 71%** | env 네이티브 — 우리 scaffold 무관 |
| **`give`(user-side)** | **21/97 = 22%** | follow_up — **4개 중 1개만 하드코딩** |
- requestor 분포: assistant **825** · user **102**.
- gold 상위 도구: `get_bank_account_transactions_9173`(73) · `get_all_user_accounts_by_user_id_3847`(72) ·
  `submit_cash_back_dispute_0589`(58) · `file_credit_card_transaction_dispute_4829`(56) · `open_bank_account_4821`(47).

### 0.3 ⇒ A2는 **세 조각**이고, 둘은 이미 일반이다
1. ✅**`verify_identity` + GB1** → **81/97(84%)**. 태스크 리터럴 **없음**. ⇒ ★**오늘 설계한 원장-결합
   (`VERIFY_IDENTITY_LEDGER_BINDING_DESIGN`·record 슬롯 삭제)은 도메인의 84%에 걸린다** — task_019 국소 수정이 아니다.
2. ⚠️**follow_up/give** → 21/97(22%). **하드코딩 1개** → **env 열거 4개**로 일반화(§2).
3. ❌**producer(`get_reward_discrepancies`)** → **task_019 전용**. ★**일반화하지 않는다**(§3).

## 1. 무엇을 만들지 않는가 (경계 선언·[[05]])
- ❌**97개 태스크에 producer/워크플로를 써 넣지 않는다.** 그건 ABox가 아니라 **태스크별 정답 주입**이고,
  점수가 *"일반화했다"*가 아니라 *"97개를 개별 튜닝했다"*가 된다. **§8 프레임-동기추론의 최고 난이도 버전.**
- ❌**gold를 읽지 않는다.** 엔진/A2는 `evaluation_criteria`에 접근하지 않는다(위 0.2는 **조사용 통계**일 뿐).
- ✅**env 구조(레지스트리·`mutates_state`)만 읽는다** — 그건 에이전트도 런타임에 볼 수 있는 사실이다.

## 2. 설계 — **user-side 하드코딩 제거**
### 2.1 현행 (task_019 리터럴)
```
a2.scaffold_get_tools[get_reward_discrepancies].follow_up = {
  tool: "give_discoverable_user_tool",
  feedback: "... call 'give_discoverable_user_tool' with discoverable_tool_name='submit_cash_back_dispute_0589' ..."
}                                                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^ task_019 전용
```
**트리거도 리터럴**이다: *"producer를 불렀으면"* = task_019에만 존재하는 조건.

### 2.2 신규 — **완료-주장 ∧ 상태변경 실행 0** (도메인 일반·트리거 불요)
- **엔진이 아는 것**(전부 구조적): ①원장 = 실제 호출 이력 ②각 도구의 `mutates_state`
  ③`env.user_tools.get_discoverable_tools()` = user-side 집합.
- **규칙**: 에이전트 응답이 **행위 완료를 주장**하는데 **원장에 상태변경 실행이 0**이면 → 피드백.
  - user-side 도구가 필요한 태스크(21/97): 사용자가 실행해야 하는데 **0회** → 주장은 거짓.
  - agent-side 태스크(69/97): 에이전트가 WRITE를 실행했으면 **주장은 참** → 발화 안 함(오탐 방지).
  ⇒ **트리거가 태스크에 의존하지 않는다.** producer 개념도 필요 없다.
- **"완료를 주장하나"의 판정**: 기존 `completion_guard.claim_question`(A2 문구·모델 자기보고) 재사용.
  ⚠️**이게 이 설계의 최약점**이다 — §6.

### 2.3 A2 변경 (리터럴 제거)
- `follow_up`을 producer에서 **떼어내** 도메인 수준 `completion_guard`로 승격.
- `discoverable_tool_name='submit_cash_back_dispute_0589'` 문구 **삭제** → 피드백이 **env 열거 집합**을 렌더.
- **엔진 신규**: `_user_discoverable(env)` = `set(env.user_tools.get_discoverable_tools())` (try/except·미지원 도메인=∅).

## 3. producer는 왜 일반화하지 않나 (정직)
`get_reward_discrepancies`는 **env 도구가 아니라 우리 A2 compute 도구**다(task_019의 보상-불일치 계산).
나머지 96개는 **env 네이티브 도구를 직접** 쓴다 ⇒ producer가 **필요 없다**. 96개에 producer를 만드는 것 = §1 금지.
⇒ **97-태스크 점수에서 우리 기여는 ①verify(84%) + ②give/완료(22%)이고, producer(task_019)는 1/97이다.**
**이걸 미리 적어둔다** — 점수가 낮게 나왔을 때 *"producer가 없어서"*라고 사후 변명하지 않기 위해.

## 4. ⚠️새로 보이는 위험 — **(a1)이 도메인 전역 버그일 수 있다**
`verify_identity`는 **97개 전부에 주입**된다. 그런데 (a1) 실측: 모델이 `unlock_discoverable_agent_tool`로
**우리 주입 도구**를 부르면 env가 *"This tool is not available"*(거짓)을 반환하고 → producer 직접호출이 **6%**로,
**ASK(포기) 56%**로 간다(`FAB_PROBES §5.2-A`).
★**69/97이 unlock/call 경로를 쓰는 도메인**이라 모델에 **unlock 습관**이 있다 ⇒ 우리 주입 도구(`verify_identity`)에도
unlock을 시도할 개연 ⇒ **(a1)은 task_019 국소 수정이 아니라 도메인 전역**일 수 있다.
- **[S] 현 데이터로 확인 불가**(우리 banking 런은 전부 task_019). **97-태스크 런이 처음으로 잰다.**
- ⇒ **97-런 로그에서 `unlock…(verify_identity)` 빈도를 반드시 센다**(무료·§5 F3).

## 5. ★반증 예측 — 사전등록 (결과 보기 前)
> **이 설계가 옳다면**: user-side 하드코딩을 env 열거로 바꿔도 **task_019 성적이 안 떨어지고**(회귀 0),
> **give가 필요한 21/97**서 `give` 누락·완료-날조가 **줄어야** 한다.
>
> **반증조건(하나라도 걸리면 폐기·기함 교체 금지)**:
> - **(F1) task_019 회귀**: 페어 비교서 pass가 떨어지면 → 일반화가 task_019용 정밀도를 팔았다 ⇒ 폐기.
> - **(F2) 오탐**: agent-side 태스크(69/97)서 `completion_guard`가 **참인 주장**에 발화하면(에이전트가 WRITE를
>   실행했는데도) ⇒ 규칙이 틀렸다 ⇒ 폐기. **Δspurious 계측 필수.**
> - **(F3) claim_question 미준수**: 모델이 자기보고를 안 하거나 거짓 보고하면 → NabaOS Lim 1과 같은 벽
>   (그들 실측 **오픈웨이트 준수 85%**). **준수율을 먼저 재고**, 낮으면 이 설계는 **프롬프트 천장**에 걸린 것.
> - **(F4) 84%가 안 걸린다**: `verify_identity`가 81/97서 실제로 호출되지 않으면 → 0.3①의 전제가 거짓 ⇒
>   **"A2가 84%를 덮는다"는 주장 철회.** ★**97-런 로그로 즉시 검정 가능.**

## 6. 정직한 최약점 — **완료-주장 판정이 모델 자기보고다**
`claim_question`은 **모델에게 "네 답이 완료를 주장하니?"라고 묻는다** = **NabaOS self-tagging과 같은 형**
(`NABAOS_PREEMPTION_AUDIT §1.1`). 그들 실측: 준수율 **92% Claude / 88% GPT-4 / 85% 오픈웨이트**,
미준수는 *"conservative fallback"*으로 흡수. **우리는 오픈웨이트 32B = 그들 최악 구간.**
- **대안(텍스트 파싱)은 금지**([[03b]]) — 그건 우리가 NabaOS 감사서 함정이라 판정한 길이다.
- ⇒ **(F3)이 이 최약점을 정면으로 시험한다.** 걸리면 **"게이트로는 못 닫는다"**가 실측되는 것이고,
  그게 §4 학습 트랙의 **진짜 근거**가 된다(현재 §0 게이트 FAIL로 정지 중).

## 7. 순서
1. **엔진 `_user_discoverable(env)` + completion_guard 일반화** 구현 · **기본 OFF**(`T2_WRITE_PROV=1`로 opt-in —
   현재 런 스크립트에 없어서 **0회 발화**였다·`HANDOFF_LEVER_DESIGN §1`).
2. **task_019 회귀 페어**(무료 프로브 우선) → (F1)/(F2).
3. **97-태스크 nt=1 · `alltools` · `--max_retries 0` · `--auto_resume`** → (F3)/(F4) + (a1) 전역성 계측.
4. nt=4까지 누적 → **리더보드 Custom 탭 비교**(현 유일 경쟁자 **Distyl ButtonAgent 31.2%**·`TAU2_FRONTIER §3b`).
