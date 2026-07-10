# Scaffold 전수 감사 — 규칙 0 위반(꼼수) 색출 (2026-07-08 밤)

> 상위 = `RESEARCH_MASTER.md`. 사용자 지시(2026-07-08):
> **"tau²의 0번 규칙 = DB 등 내용은 반드시 도구를 통해 얻고 날조하면 안 된다. present 같은 꼼수를 다 찾아 제거하라.
> learn으로 가면 명확한 부분을 scaffold 꼼수로 피하면 안 된다."**
>
> **판정 기준**: *에이전트가 스스로 호출하지 않은 도구의 출력을 엔진이 대신 가져와 문맥에 넣으면 = 규칙 0을 에이전트 대신 우회 = 꼼수.*
> 반대로 **에이전트 자신이 이미 가져온 내용 위에서** 검증·재제시·계산하는 것은 위반이 아니다([[10]] 결정론 분담).

---

## 1. 감사표

| 기계 | 플래그 | 동작 | 새 DB 내용 주입? | 판정 |
|---|---|---|---|---|
| `candidate_summary` (`gate_interpreter.py:321`) | **`T2_PRESENT_READS`** | 에이전트가 `user_producer`(=`get_user_details`)를 부르면, **엔진이 그 사용자의 *모든* 주문에 대해 `detail_producer`(=`get_order_details`)를 대신 호출**하고 `present_fields`(status·address·items)를 `[DISAMBIGUATION NOTE]`로 응답에 덧붙임 | **YES** — 에이전트는 `get_order_details`를 부른 적 없다 | ❌ **꼼수·제거** |
| `_autofetch_text` (`t2_gate_patch.py:149`) | **`T2_AUTOFETCH`** | provenance-deny 시 **엔진이 A2 producer를 호출해 실제 레코드를 텍스트로 반환**. 주석: *"모델에 실값 제공… 날조-FIRST default를 엔진이 결정론으로 메움"* | **YES** | ❌ **꼼수·제거** |
| `nested_candidate_summary` (`:350`) | `T2_PRESENT_NESTED` | **에이전트 자신의 응답 레코드**(`_rec`)의 nested list를 `[OPERAND DISAMBIGUATION]` choice-set으로 재제시 | NO | ⚠️ 재제시(허용·단 §3) |
| `compute_facts` (`:385`) | `T2_CALC` | **에이전트 자신이 부른 도구 출력**(`_rec`) 위에서 `count_where/count/sum/lookup` | NO | ⚠️ 계산 offload(허용·단 §3) |
| gate kinds `auth·confirm·ownership·notice·preconditions·constraints` | `T2_GATE_KINDS` | 호출 거부만. 메시지 = `Error: [POLICY GATE {name}] {reason}` (값 미노출). 내부 `resolve_field`는 판정용 read-only | NO | ✅ 결정론 검증기 |
| `REGEN_FEEDBACK` (`:373`) | **`T2_PROV_REGEN`** | 날조 인자 감지 → *"invented 값이다. **getter를 불러 실값을 읽어라**"* + 재생성. **정보 주입 0** | NO | ✅ **= E11-a 그 자체** |
| `_grounded_candidates` (`:467`) | `T2_PROV_GROUND` | **이전 도구 출력에서** 후보 추출해 제시 | NO | ⚠️ 재제시 |
| `_static_blacklist` (`:99`) | `T2_PROV_BADWORDS` | 도구 docstring의 placeholder 토큰을 logit ban | NO | ⚠️ 능력 아님(강제 억제) |

## 2. 발견

1. **정도(正道)가 구현돼 있는데 한 번도 안 켰다.** `T2_PROV_REGEN`은 어떤 `reexp_*.sh`에도 없다.
   대신 `reexp_assembled.sh`는 `T2_PRESENT_READS=1 T2_PRESENT_NESTED=1 T2_CALC=1`을 켰다 —
   **꼼수를 켜고 정도를 껐다.**
2. ~~**C4 학습 평가가 autofetch로 오염됐다**~~ → **❌ 철회 (자기교정 #13).**
   `AF`는 위치 인자이고, 로그 확인 결과 `af=1`은 **`engine` · `dpo-perform` · `scaffold-perform`** 세 arm뿐이다.
   학습 평가 본체(`dpo-pure` · `dpo-deny` · `learn` · `learn-pure`)는 전부 `af=0`. **오염 없음.**
   *(스크립트에서 플래그를 보고 "학습 결과 오염"으로 승격한 것 — 오늘 반복된 대리신호→인과 승격 패턴.)*

## 2b. ★그 대신 나온 것 — deny도 프롬프트도 gather를 못 산다 (7B·retail·nt=1·선행 실측)

`c4_prompt_mechanism.py` 정의: `no_gather` = 날조 id 사용 ∧ (스키마-예시 패턴 ∨ producer 미호출).
`schema_copy` ⊂ `no_gather` (예시 패턴). `gather_wrong` = producer는 불렀으나 그 출력에 없는 id.

| arm | pass^1 | nfail | **no_gather** | schema_copy | gather_wrong |
|---|---|---|---|---|---|
| base 7B | 32/114 | 82 | 36 | 36 | 3 |
| dpo-pure (학습) | 20/113 | 93 | 37 | 35 | 11 |
| **dpo-deny** (provenance 거부만·af=0) | 23/112 | 89 | **36** | 36 | 11 |
| **dpo-perform** (+autofetch 꼼수) | 25/109 | 84 | **24** | **23** | 9 |

C3 프롬프트 스윕(7B·af=0): base 23 · nofab 27 · **fetchfirst 23** · fewshot 26 · structured 26 (/114).
C4 학습(cfbsynth SFT): `learn` 13/110 · `learn-pure` 15/112 — **base(21)보다 나쁨**.

**읽는 법**
- **프롬프트로 gather 강제 = 무효** (`fetchfirst` = base). ⇒ C30 재확인, 7B서도.
- **deny(=E11-a류)만으로도 무효**: `no_gather` **36 → 36**. *"지어낸 값이다, getter를 불러라"* 라고 말해줘도 안 부른다.
- **autofetch(꼼수)만 작동**: `no_gather` 36→24, `schema_copy` 36→23.
- 7B base의 `no_gather` 36 중 **schema_copy 36 = 전부** ⇒ 날조 = **스키마 예시 복사**([[42]] induction head).

> ### ★기전 (이 감사의 핵심)
> **autofetch도 present도 gather를 가르치지 않는다. 더 가까운 *복사 대상*을 문맥에 놓아줄 뿐이다.**
> 모델은 여전히 복사한다 — 다만 스키마 예시 대신 진짜 값을 복사한다. 그래서 pass는 오르고 조회는 죽는다.
> ⇒ **두 꼼수가 켜진 채로는 gather를 측정조차 할 수 없다.** 그리고 이 둘은 induction head를 *고치는* 게 아니라 *이용한다*.
>
> ※ 단 **32B에서는 날조의 *형태*가 다르다**: 93건 중 스키마-예시 복사 18 · 그럴듯한 10자리 id 발명 48 ·
> 조합형 placeholder(`X_cheapest`) 16. **scale은 복사를 발명으로 바꾼다**(둘 다 환경이 100% 거부). [M]

## 3. present 꼼수가 실제로 무엇을 했나 (32B-int8·t3·clean sim)

| arm | pass^1 | db_pass | **order 조회/sim** | product 조회/sim | **미조회 날조** |
|---|---|---|---|---|---|
| floor | 0.547 | 0.596 | **2.62** | 0.79 | **5.6%** |
| present 단독 | 0.576 | 0.626 | 1.29 | 0.77 | 6.9% |
| g15 단독 | 0.550 | 0.578 | 2.44 | 0.78 | 7.0% |
| **present+g15** | 0.594 | 0.678 | **0.48** | 0.76 | **10.4%** |

- present는 pass를 **사고**(+4.7pp), **조회 습관을 5.5× 죽이고**, **날조를 1.9× 늘린다**.
- 즉 present는 **frontier 격차의 83%를 차지하는 실패(C29)를 스스로 제조한다**. 국소 최적이며 진짜 수리를 막는다.
- ★기전은 [D]: order 조회만 죽고 product 조회는 불변(0.79→0.76)인데 *변형* 날조가 2배가 된다.
  present가 item_id를 나열해 주자 모델이 **변형 id도 조합해 만들 수 있다고 착각**하는 것으로 보인다. **미확정.**

### 3.1 `nested`·`calc`는 규칙 0을 어기지 않는다 — 그러나 학습 신호는 지운다
둘 다 **에이전트 자신이 가져온** 레코드 위에서 동작하므로 규칙 0 위반이 아니다.
그러나 **C31**(present가 `read→act` 감독 신호를 파괴)의 약한 형태를 공유한다: 모델이 *선택*과 *계산*을 배울 기회가 사라진다.
- `calc`: [[10]]/[[00]]이 **명시적으로 승인한 결정론 분담**(계산기는 LLM 몫이 아니다) ⇒ **유지**.
- `nested`: 재제시일 뿐 정보 추가가 없다 ⇒ **E11 실험에서는 OFF**(gather 효과를 오염 없이 재려면).

## 4. 정화된 스택 (제안)

| | 기존 `asmregen` | **정화 후** |
|---|---|---|
| `T2_PRESENT_READS` | 1 | **0 (제거)** |
| `T2_AUTOFETCH` | 0 (단 c3/c4 probe선 1) | **0·영구 금지** |
| `T2_PRESENT_NESTED` | 1 | **0** (E11 측정 중) |
| `T2_CALC` | 1 | 1 (유지·[[10]]) |
| `T2_GATE_KINDS` | auth,confirm,ownership,notice,preconditions,constraints | 동일 |
| **`T2_PROV_REGEN`** | **없음** | **1 = E11-a** |
| `T2_PROV_BADWORDS` | — | **0** (능력이 아니라 억제·별도 arm) |
| `T2_PROV_GROUND` | — | 0 (별도 arm = E11-b) |

## 5. 이 감사가 바꾸는 주장

- **C4a `present+g15 +12.3pp`**: 수치는 유효하나 **규칙 0 우회로 산 것**. 논문에서 단독 인용 금지 —
  반드시 *"조회 5.5× 억제·날조 1.9× 증가를 대가로"* 와 함께.
- **C16 "도구호출 부재 ≠ 정보 부재"**: 맞다. 그런데 그 정보를 **엔진이 대신 가져왔다**는 것이 요점이었다.
  즉 C16은 present가 꼼수라는 증거였는데 우리는 그것을 *에이전트 변호*로 읽었다.
- **C29 gather 격차**: present를 끄면 floor에서 상한이 **8.8pp**로 커진다(present 스택선 5.0pp).
- **[[20]] C4 "학습 실패"**: 오염 아님(af=0). 그러나 **그 학습의 표적이 gather가 아니었다** — cfbsynth `$ref` **copy**였다.
  ⇒ *"학습은 실패했다"* 가 아니라 **"gather 학습은 시도된 적이 없다"** 가 정확한 진술.

## 5b. ★결론 — 여기가 learn 축이다

§2b가 보여주는 것: 이 결손에 대해
**프롬프트(무효) → deny/hint(무효) → 꼼수 supply(작동)** 의 순서다.
`§1.5` 절차대로 읽으면:

- Q1b: 날조 *차단*은 **환경이 이미 집행**(93/93 ERR·C12) ⇒ 차단 게이트 = 죽은 레버.
- 남은 scaffold = **supply(present/autofetch)** 뿐인데 이것은 **규칙 0 우회**이며 **induction head를 이용**할 뿐이다.
- Q3/Q4: 프롬프트·hint로 안 열리고 scale은 7B→14B만 산다(38.8%→7.0%) · 14B→32B 정체 · frontier 0.0%는 **불연속**.

⇒ **명확한 원인(gather-before-act 미수행)을 scaffold 꼼수로 우회하지 말고 학습으로 보강해야 한다.**
그리고 **gather 학습은 부작용이 거의 없다**: 검출 술어의 Δspurious = 0(§6.1), 규칙 0을 어기지 않으며,
도메인-일반(`read → act`)이라 **ABox-swap 전이**([[11]])의 정직한 시험대다. ⇒ **E6의 표적을 copy에서 gather로 재지정.**

## 6. 다음
1. **꼼수 제거**: `T2_PRESENT_READS` 영구 OFF · `T2_AUTOFETCH` 영구 금지. 이후 baseline = **floor**.
2. **E11-a (`T2_PROV_REGEN=1`) × 32B floor** — *희망*이 아니라 **통제군**으로 돌린다.
   7B서 deny는 `no_gather`를 못 줄였다(36→36). 32B서도 못 줄이면 **scaffold 경로 종결 → learn 확정**.
   Δspurious는 이미 0으로 측정됨(§6.1: db_pass sim 발화 write 14/14·22/22·1/1 전부 환경이 ERR로 거부).
3. **E6′ = gather-before-act 학습.** 감독 신호 = `위반 지점 → read 호출 → 올바른 write`.
   ★표적을 copy가 아니라 **read 호출 결정**으로 둔다(C4의 실패 원인). 평가는 **모든 supply 꼼수 OFF**로만.

### 6.1 Δspurious 실측 (E11 술어)
| arm | db_pass sim서 발화한 write | 결말 |
|---|---|---|
| 32B floor | 14 | **14/14 ERR** (환경이 어차피 거부) |
| 32B +present스택 | 22 | **22/22 ERR** |
| o4-mini | 1 | 1/1 ERR |

**옳은 write를 막는 사례 0.** 술어는 *어차피 실패할 write*에만 발화한다 ⇒ 검출 자체는 부작용 0.
