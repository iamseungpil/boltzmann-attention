# `verify_identity` 원장-결합 설계 — **날조 슬롯을 없앤다** (2026-07-18)

> 목적: **τ² pass 올리기**(논문 신규성 아님). NabaOS(`2603.10060`) 설계를 **의도적으로 베낀다** —
> 신규성 주장 0([[41]]/C111·`NABAOS_PREEMPTION_AUDIT §1.1`). 실용만 본다.
> 선행 provenance = `SOURCE_PROVENANCE.md §1`. 상위 감사 = `NABAOS_PREEMPTION_AUDIT_2026_07_18`.
> ⚠️**§8 하드룰 1 적용**: **랭크 前에 반증 예측을 먼저 적는다**(§6). 그게 없으면 이 설계도 오늘 죽은 프레임 3개와 같은 길.

## 0. 왜 (실측 근거)
| 실측 | 값 | 출처 |
|---|---|---|
| `record` 날조율 | **46%** | `FAB_PROBES §2`·`bank_fab_probes --probe record` n=24 |
| **grounded 정답** | **0/24** | 동 |
| A2 **설명 레버**로 고쳐지나 | **0/24**(이동 29%만) | `record_hint` arm |
| 라이브 PROV가 이 표면을 보나 | ❌**못 봄** | `_hint_hit('record') = False`([M] 실행확인·감사 §7-2) |

⇒ **설명으로도 안 되고, 게이트도 못 본다.** 남은 건 **슬롯 자체를 없애는 것**뿐.

## 1. 진단 — **우리가 만든 순환** (감사 §2.1-v3 ⑤)
```
verify_identity(provided={...}, record={...})  ← LLM이 주장과 증거를 둘 다 공급
op: match_verdict(a=provided, b=record, fields=[dob,address,phone,email], threshold=2)
```
- 엔진은 **LLM이 준 두 dict를 서로** 비교한다. **양쪽 다 LLM 것이면 비교는 공허하다.**
- **우리는 이미 알고 있었다**: `cls_record`에 축자로
  `if set(vals) <= provvals: return ("★record=provided복사(순환)", "FAB")` — **공허함을 인코딩해 두고 현상으로 보고할 뻔했다.**
- ⇒ **46%는 세상에 대한 발견이 아니라 우리 A2에 대한 버그 리포트다.**

## 2. 처방 — **`record` 슬롯 삭제 · 엔진이 원장을 본다**
```
verify_identity(provided={date_of_birth?, address?, phone_number?, email?})   ← record 파라미터 제거
```
**엔진 판정(필드별)**:
| 조건 | 무엇을 보나 | 왜 |
|---|---|---|
| `said_by_user(v)` | v가 **user 메시지**에 등장 | 고객이 실제로 말했나 (주장의 근거) |
| `in_record(v)` | v가 **선언된 producer 도구의 출력(ToolMessage)**에 등장 | 계정 기록과 일치하나 (증거) |
| **match** | **둘 다 참** | |
`VERIFIED  ⟺  count(match) ≥ threshold(2)`

- **날조 슬롯이 없다** → `record` 날조 **구조적으로 불가**(막는 게 아니라 **소멸**).
- **조회를 강제한다** → producer 출력이 없으면 `in_record`가 전부 거짓 → **0 match → NOT_VERIFIED**
  → 피드백이 *"아직 기록을 조회하지 않았다. 이름/이메일/user ID를 물어 `get_user_information_by_*`를 부르라"*.
  ⇒ **gold action(실제 lookup)까지 같이 산다.**
- **순환 소멸** → `provided`만으로는 자기 자신을 검증 못 한다(record 측 증거가 **엔진 소유**이므로).

## 3. 왜 [[03b]] 안전한가 — **파싱·추출 0**
- ⚠️함정: 엔진이 도구 출력에서 record를 **추출**하면 그게 **엔진-formalize = 구현 속임**([[03b]]).
- ✅**우리는 추출하지 않는다. 멤버십만 테스트한다**: LLM이 operand(`provided[f]`)를 **formalize해서 주고**,
  엔진은 *"이 문자열이 저 영수증에 등장하나"*만 묻는다 = 기존 `_ctx_has`와 **동일 술어**(PROV가 이미 쓰는 것).
- ✅**실측 전제 확인**(값이 축자로 등장하는가·[M] 궤적 직독):
  ```
  get_user_information_by_name -> "Found 1 record(s) in 'users':
     1. Record ID: af0581dcbf
        address: 562 Riverside Drive, Chicago, IL 60611
        email: sparkle_queen_99@yahoo.com
        phone_number: 312-555-0481
        date_of_birth: 11/03/1990"
  ```
  ⇒ `11/03/1990`·`312-555-0481`이 **축자로 등장** → 멤버십으로 충분. **파서 불필요.**
- **NabaOS 대응**: 그들 `pratyakṣa` 검사(*"compare claimed facts against the facts field"*·`315-316`)와 **같은 형**.
  단 그들은 어댑터로 **추출**(`286-287`)하고, 우리는 **멤버십**만 한다 = **더 약하지만 더 안전**(추출 코드 0).

## 4. 왜 [[05]] 안전한가 — 엔진=일반 · 도메인=A2
| | 무엇 | 어디 |
|---|---|---|
| **엔진(고정·도메인일반)** | 역할-제한 멤버십(user 발화 / producer 출력) · 카운트 · threshold 비교 | `t2_compute` 새 op |
| **A2(가변·ABox)** | **어느 도구가 record producer인가**(`evidence_from`) · fields · threshold · 문구 | `banking_knowledge.gate.json` |
- 엔진에 **도메인 도구명 리터럴 0**. banking→다른 도메인 = **A2의 `evidence_from`만 교체**.
- A2에 **이미 자리가 있다**: `producers` 키가 **비어 있고**(`{}`), 엔진은 이미
  `a2["_producer"] = (a2.get("producers") or {}).get("authenticated_user_record")`를 읽는다 ⇒ **그 자리를 채우는 것**.

## 5. 구현 (3곳·전부 작은 변경)
1. **`t2_compute.py`** — 새 op `match_verdict_grounded`:
   ```
   {"op":"match_verdict_grounded", "a":"provided",
    "evidence_from":["get_user_information_by_name","get_user_information_by_email","get_user_information_by_id"],
    "fields":[...], "threshold":2, "met_template":..., "unmet_template":..., "no_record_template":...}
   ```
   ⚠️ctx 2종(user 발화 / producer 출력)이 필요 → `t2_scaffold_get.exec2`가 **op ctx에 주입**(현재는 인자만 넘김).
2. **`t2_scaffold_get.py`** — `exec2`에서 `state.messages`로 두 ctx를 만들어 `_ctx`에 실어 보냄.
   role-제한 = **user content** / producer 호출의 **ToolMessage content**(tool_call_id 매칭).
3. **`a2/banking_knowledge.gate.json`** — `verify_identity`의 `params.record` **삭제** · `op` 교체 ·
   description에서 record 문장 삭제 · `examples` 교체.

## 6. ★반증 예측 — **사전등록** (§8 하드룰 1 · 결과 보기 前에 확정)
> **이 설계가 옳다면**: `bank_fab_probes --probe record` (n=24)에서
> ① **`★record-날조` = 0**(슬롯이 없으니 **구조적으로** 0 — 이건 시험이 아니라 정의)
> ② ★**진짜 시험 = `record-grounded` 정답이 `0/24` → 유의하게 상승**해야 한다.
>
> **반증조건(하나라도 걸리면 설계 폐기 · 기함 교체 금지)**:
> - **(F1) grounded 정답 ≤ 2/24** — 슬롯을 없앴는데도 안 오르면 **표면이 사라진 게 아니라 이동**한 것
>   (§2b 깔때기: 아래로 샜나 확인). ⇒ **폐기·정직 보고.**
> - **(F2) Δspurious > 0** — **정당한 값이 NOT_VERIFIED로 오버블록**되면(포맷 차이 등) 레버가 **하나 사고 하나 판** 것.
>   모트 제1원리상 **계측 필수**·순증 아니면 폐기.
> - **(F3) ASK 폭증** — 모델이 검증을 포기하고 되묻기만 하면(과잉기권) pass는 안 오른다.
>
> **이 예측을 지금 박제한다. 결과가 나온 뒤 기준을 옮기지 않는다.**

## 7. 정직한 위험
- ★**포맷 불일치** — ✅**전수 사전측정 완료**(2026-07-18·[[08]] 훅이 잡음: 초안은 **궤적 1건**으로 낙관했었다):
  영속 banking sim **전건**서 조회 성공 **43 sim** · 기록 값이 사용자 발화에 축자 등장 **86/176 = 49%**.
  ★**그런데 49%는 포맷 문제가 아니다** — 분해하면:
  | 필드 | 값 | `said_by_user` |
  |---|---|---|
  | `date_of_birth` | `11/03/1990` | **True** |
  | `phone_number` | `312-555-0481` | **True** |
  | `address` | `562 Riverside Drive, Chicago, IL 60611` | False |
  | `email` | `sparkle_queen_99@yahoo.com` | False |
  ⇒ **고객이 4개 중 2개만 말했기 때문**이고, 그게 **정확히 정책이 요구하는 것**(threshold=2)이다.
  **고객이 실제로 준 두 값은 축자 일치** ⇒ `said_by_user ∧ in_record` = 2 ≥ 2 → **VERIFIED**. **엔진 멤버십이 그대로 선다.**
  ⇒ **(F2) 포맷 위험 = task_019에 한해 낮음**([M]).
- ⚠️★**단, 이 증거를 과대평가 금지**([[08]] 분모 규율): 43 sim은 **전부 task_019 = 동일 고객 1명**이다.
  중복 제거하면 **distinct record = 1**. `86/176`은 n=176처럼 보이지만 **레코드 1개 × 43 trial × 4 필드**다.
  ⇒ **"포맷은 안전하다"는 task_019 밖으로 일반화되지 않는다.** 타 도메인/레코드 전이 시 **(F2) 재측정 필수.**
- **부분일치 오탐**: `_ctx_has`는 부분문자열이다 → 짧은 값 우연 매칭. **len≥4 관례 유지**(기존 `_first_fab_call`과 동일).
- **`said_by_user` 엄격성**: 사용자가 다른 형태로 말했으면 실패. **1차는 이 조건을 lenient로 두고 (F2) 보며 조인다.**
- **거동 변화**: 게이트 판정이 바뀐다 = **모든 arm 비교 기준선이 바뀐다** ⇒ **진행 중 nt=20 수거 後 적용**(§8).

## 8. 순서 (전부 무료 · 유료 arm과 분리)
1. **nt=20 수거 완료 대기**(진행 중·워처 자동 push). **그 전엔 엔진 손대지 않는다.**
2. **오프라인 프로브 먼저**: `bank_fab_probes --probe record`를 새 시그니처로 n=24 → §6 반증조건 판정. **무료**(GPU만).
3. 통과 시 **단일변수 라이브 arm**(kon과 이 변경 1개만 차이).
4. 같은 창에 `_hint_hit` 다중토큰 픽스(감사 §7-1)도 함께 — **단 별개 변수라 arm은 따로**.

## 9. 이 설계에서 우리가 주장하지 않는 것
- **신규성 0.** claim→evidence 결합·런타임 소유 원장·LLM이 위조 못 하는 증거 = **NabaOS Stage 4/§3.2 그대로**
  (`309-311`·`230-232`·`17`). 우리는 **HMAC만 생략**한다(엔진이 원장 소유자라 위조가 애초에 불가).
- 이건 **엔지니어링 수리**다. 논문 코어 후보가 아니다(감사 §2.1-v5·§4).
