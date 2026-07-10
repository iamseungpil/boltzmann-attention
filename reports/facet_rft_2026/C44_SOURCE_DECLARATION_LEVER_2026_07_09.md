# C44-C48 — 출처 선언 레버(4지선다 + provenance 검증)가 날조를 닫는다. 단 ⋈ 오선택은 못 닫는다 (2026-07-09 밤·무료)

> 상위 = `RESEARCH_MASTER.md`. 선행 = `C43_ANCHORED_SUBSTITUTION_NOT_WM`(날조=정박치환).
> 사용자 물음: (1) 예시를 태그로 무시시킬 수 있나 (2) 4지선다 프롬프트로 싸게 되나 (3) 과잉 ASK 분석 (4) 검증기 D′.
> 재현 = `scripts/distill/tau2/c45_four_arms.py` · `c47_dprime.py` · `c48_dprime_full.py` · `dbonly_forensic.py`.
> 대상 = `fl32b_floor_retail_t4` (user_stop만·infra 0). 결정점 = 원 궤적서 날조가 난 write 지점(FAB) + grounded였던 지점(CLEAN).

---

## 0. 한 줄
**날조는 능력 결손이 아니라 "출처를 대지 않아도 되는 인터페이스"의 산물이다.**
출처를 선언·검증하게 하면 67% → 0%. 단 남는 것은 ⋈ 오선택(F3 경계)이며 그것은 이 레버로 안 닫힌다.

## 1. 4-arm 비교 (C45 · 32B · 날조 지점 30 · 짝지음)

| arm | 날조 | 기전 |
|---|---|---|
| A 원본 | **20/30 (67%)** | 예시값 9 · 근접변형 8 · placeholder 3 |
| B 예시값 제거 | 11/30 (37%) | 예시값 **0** · 근접변형 7 |
| C 예시 `<EXAMPLE>` 태그 + "무시하라" 지시 | 17/30 (57%) | 예시값 **8**(거의 안 줄어듦) |
| **D 4지선다 강제** | **3/30 (10%)** | ASK 20 · FIND 6 · **FIND→날조 3** |

- **C 기각**: 태그+금지문으로 예시 복사가 9→8. **보고도 무시하라는 지시는 안 먹는다**([[42]] induction-copy·금지문=priming).
- **B**: 예시 제거는 원천 ①(47%)만 없앤다. 근접변형(②)은 그대로 — **과제가 창에 넣는 id라 뺄 수 없다**.
- **D**: 날조를 *기권(ASK)* 으로 바꾼다. 그런데 그 ASK가 문제였다(§2).

## 2. ★D의 ASK는 과잉이었다 (C46·C47 gold 재계수)

**gold 라벨 버그 교정(자기교정 #17)**: `gold_value=None`을 ASK로 뭉갰다. 결정 시점 접두 기준 + NO-WRITE 분리 후:

| gold (날조 지점 30) | 건수 |
|---|---|
| GET | 16 |
| FIND | 7 |
| NO-WRITE (그 write 자체가 gold에 없음=과행동) | 7 |
| **ASK (사용자만 아는 값)** | **0** |

⇒ **D의 ASK 20건은 전부 오답.** 질문 원문: *"Could you please provide the item ID for the Fleece Jacket..."* — **고객이 알 리 없는 내부 id를 사용자에게 요구.**
몇몇은 *"If you're unsure, I can help you find it"* 이라 덧붙임 = **조회할 수 있음을 알면서 물음.**
⇒ **D는 우리를 o4-mini의 실패 양식(기권형·C21)으로 옮긴다.** 격차를 닫는 행동은 GET인데 D는 30건 중 1건만 GET.

### 2.1 ★over-block 전수 census (LLM 0회·편향 없음)
`fl32b_floor` write 인자 결정점 **2,650**: FIND **79.9%** · NO-WRITE 16.4% · GET 3.7% · **ASK 0.0%(0건)**.
14B floor도 **ASK 0/3065**. ⇒ **tau2 retail은 write 인자를 사용자에게 물을 일이 없다.**
∴ 검증기의 "산출 도구가 있으면 ASK 금지" 규칙은 **정당한 ASK를 하나도 막지 않는다(Δspurious_ASK = 0).**

## 3. D′ / D″ — 갈래별 결정론 검증기 + 재발화 + GET 폴백 (C47·C48)

검증기(전부 decidable·**DB 내용 주입 0**):
- FIND: 값이 {사용자 발화 ∪ 도구 출력}에 실재 · GET: 지목 도구가 그 인자의 producer · **ASK: producer가 없어야 함**
- 소진 시 **GET 강제 폴백**.

| 모집단 | n | **날조** | FIND-wrong(오선택) | GET폴백 | 재발화 tries≥2 |
|---|---|---|---|---|---|
| **FAB** (원래 날조 지점) | 30 | **0 (0%)** | 3 | 1 | 24 |
| **CLEAN** (원래 grounded 지점) | 30 | **0 (0%)** | 3 | 1 | 6 |

- **날조 67% → 0%.** DB 주입 0·도구 대신호출 0·학습 0. **"출처를 대라"만으로.**
- **Δspurious = 파손 3/30 (10%)** — GO 조건 위반. 정독하니 **전부 F3 ⋈ 오선택**:
```
t83  D''=credit_card_1565124  gold=gift_card_7250692   (결제수단 여럿)
t72  D''=gift_card_1711656    gold=paypal_7729105      (결제수단 여럿)
t109 D''=760 Elm Avenue       gold=592 Elm Avenue       (주소 여럿)
```

## 4. ★핵심 판정 — 두 실패는 다르다

| 실패 | 무엇 | 출처-선언 레버 | 정체 |
|---|---|---|---|
| **날조** | 없는 값 발명(정박 치환) | **67% → 0%** | engineering·닫힘 |
| **FIND-wrong** | 문맥의 *틀린* 값 선택 | 안 닫힘(3/30) | **F3 ⋈ 의미 경계(C3b)** |

**provenance 검증은 날조를 닫고 ⋈를 격리한다.** 지금까지 둘을 섞어 "operand 정밀도"로 보던 것을 갈라냈다(C23 정련).
남는 잔여는 **오직 ⋈**이며 이는 이미 "어떤 레버도 못 연 경계"로 등재됨(C3b·C9).

### 4.1 ★Δspurious 확정 = 0 (원 호출 대조 완료)
파손 3건 전부 **문맥에 후보 정확히 2개**가 있는 ⋈ 지점이고, **원 궤적도 그 지점에서 틀렸다**:
```
t83  문맥 후보 {credit_card_1565124, gift_card_7250692} · 원본이 둘 다 write(틀림+맞음) · D''=credit(틀림)
t72  문맥 후보 {gift_card_1711656, paypal_7729105}      · 원본이 둘 다 write               · D''=gift(틀림)
t109 문맥 후보 {592 Elm Avenue, 760 Elm Avenue}         · 원본도 760(틀림)               · D''=760(틀림)
```
⇒ **CLEAN "파손"은 D″의 신규 오류가 아니다**(내 CLEAN 정의 = grounded지 gold-정답 아님·자기교정 #18).
**진짜 Δspurious = 0.** D″는 멀쩡한 결정을 하나도 깨지 않았다. 파손처럼 보인 3건은 전부 F3 ⋈ 경계(원본도 못 풂).
∴ **GO 조건 충족**: 날조 0 · over-block 0(전수) · Δspurious 0. 남은 잔여 = ⋈ (레버 아님·경계).

## 5. 레버 성격 (특허·[[05]] 관점)
- **엔진**: 결정점 생성(write 직전·인자 단위) · 출처 선언 강제 · 검증(provenance·producer 존재) · 재발화 · 예산.
- **A2(도메인별)**: `{인자 → producer 도구}` 매핑 **하나뿐**(스키마 도출). 값·내용은 주입 안 함.
- present/autofetch와 대조: **DB를 대신 읽지 않는다.** 규칙 0 위반 없음(C34). Δspurious_ASK=0(전수).
- ⇒ **"operand 출처를 유한 선택으로 선언시키고 provenance로 집행"** = 도메인-일반·[[11]] ABox-swap 후보.

## 5b. ★선행연구 대조 (딥리서치 `wb07r5hi7` 완료·2026-07-09·3-vote 검증)

### 5b.1 우리 프레이밍의 신규성 (지지)
- **4지선다 완전체(GET/FIND/INFER/ASK)를 정식화한 선행 0** ⇒ **원본 프레이밍.** 각 분기는 조각으로 연구됨:
  ASK-vs-retrieve/fabricate만 명시 벤치화(ToolDial `2503.00564`·SAGE-Agent `2511.08798`·Learning-to-Ask `2409.00557`·**BFCL v3 "Missing Parameters"**·When2Call `2504.18851`).
- **★INFER 분기(문맥 값에서 유도·최저가·합계)는 어디에도 라벨된 결정으로 없음** — 선행이 ASK나 GET으로 뭉갬. **whitespace.**
- **"산출 도구 있으면 ASK 금지"(우리 C48 검증기 규칙)를 인코딩한 선행 미확인** — BFCL이 그렇다는 주장은 **refute(1-2)**. ⇒ **검증기 규칙도 신규 후보.**

### 5b.2 우리 관측을 지지하는 확립 사실 (검증 통과)
- **날조 = default 실패**: "명시 결정 없으면 결측 인자를 임의 생성" [Learning-to-Ask·When2Call: *"still often hallucinate a tool call with the missing parameters"*]. ⇒ **C45 전제 확립.**
- **결정이 scale로 안 열림**: ToolDial 프롬프트 frontier <70%(action+param) · When2Call 70B **~33-38 F1**(BFCL AST 68 대비) · **tau-bench pass^8 <25%(retail·우리 도메인)**. ⇒ **출처 결정은 규모-저항** — 우리 32B 관측과 정합.
- **over-asking 실증**(우리 "ASK로의 도피"): Learning-to-Ask `Re` 지표 · SAGE-Agent가 clarification **1.5-2.7× 감소**·완전명세 쿼리서도 baseline 2.1-2.9회 질문. ★단 **When2Call은 Qwen/xLAM서 *under*-asking 우세**(refute 1-2) ⇒ **over-asking은 모델·세팅 의존**, 보편법칙 아님.

### 5b.3 완화법 전이 (지지·우리 D″와 정합)
- **명시 ask-결정 프롬프트**: Learning-to-Ask서 ask-정확도 0.52→**0.90** but **최종 call 정확도 0.48→0.58만**. ⇒ **결정 단계는 asking을 고치지 grounding을 거의 못 고침** = 우리 D″가 날조는 닫되(67→0%) ⋈는 못 닫음(C46)과 **정확히 정합**.
- **abstention/irrelevance는 학습 가능**: Hammer `2410.04587` — 부정 예시(정답 함수 제외·라벨=빈 리스트) ~7,500개·최적 ~10% 혼합. **단 실행정확도와 trade-off**(균형 가능). ⇒ E6′ 데이터 설계에 참고(음성 사례 필수·C38과 정합).

### 5b.4 미지지·우리 추론으로 남는 것
- **"tool-field 매칭이 병목"(C40)은 선행 미지지**: 기전은 구축됨(ToolDial API 그래프·AutoTool 의존성 역추적) but **병목이라는 수치 격리 없음**(AutoTool GET 22-77%/FIND 25-71% 주장은 refute 0-3). ⇒ **우리 자체 ablation 필요**(oracle-matching vs model-matching).
- caveat: 대다수 2025 단일 preprint·저자 self-bench. tau-bench <50% 등 일부 refute(0-3).

## 6. 다음 (무료)
1. **§4.1**: CLEAN 원 호출 vs gold 대조 → 진짜 Δspurious.
2. FIND-wrong(⋈)을 검증기로 못 닫음을 확인 → **경계 재확인**(C3b 강화·레버 아님).
3. 다중턴 e2e: GET 선언 후 실제 올바른 도구·인자로 호출하고 올바른 변형 고르는지 (32B·user-sim 14B 무료).
4. 딥리서치 `wy3wbu6o9`(정박치환) · `wb07r5hi7`(argument source resolution 선행) 회수 후 §5 상한 대조.
