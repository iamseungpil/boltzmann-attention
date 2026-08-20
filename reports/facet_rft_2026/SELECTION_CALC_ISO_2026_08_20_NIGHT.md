# 선택 근거의 계산 — 부정통제가 이 측정을 무효로 만들었다, 그리고 기전은 그래도 보인다

> 2026-08-20 밤 · 사용자 지시 *"진행하라"* · 계기 `x440_selection_calc_iso.py` · 산출 `x440_calc1.json`
> 팔 `A_min`(후보표 + 손님 발화 중 **수가 든 문장만**) / `B_full`(후보표 + 발화 전문) / **`C_nopat`**(후보표만·부정통제)

## 0. 한 줄

**이 프로브의 채점은 무효다** — 부정통제 `C_nopat`(손님 정보를 **뺀** 팔)이 3/4 로 두 정보 팔(0/4·1/4)을
이겼다. 이유는 표본의 gold 가 4 중 3 이 같은 카드(`Silver Rewards Card`)라 *"항상 Silver"* 라고만 해도
3/4 가 나오기 때문이다. ⇒ **hit 수치는 인용 금지**. 다만 **답의 근거 문장**은 오염되지 않았고, 거기서
기전이 보인다: 모델은 랭킹 문제를 **필터 문제로 되돌리거나 가장 큰 요율 하나를 고른다**.

## 1. 부정통제가 한 일 ([[57]])

`C_nopat` 의 답은 네 사례에서 **글자까지 같다**:
```
winner: "Silver Rewards Card"
why   : "It offers high cashback on top categories without an annual fee."
```
즉 그것은 **상수**다 — 손님 정보가 없을 때 이 모델의 기본 카드. 그런데 표본의 gold 는
003 t0·003 t1·063 t0 이 전부 `Silver Rewards Card` 라서 그 상수가 **3/4 적중**으로 채점됐다.
⇒ 부정통제가 없었다면 나는 *"정보를 주면 오히려 나빠진다"* 를 결론으로 적었을 것이다. 그것은
**표본의 gold 편중**을 잰 것이다([[57]] 가 요구하는 자리·이번엔 통제가 실제로 결론을 막았다).

## 2. 그래도 남는 관측 — 근거 문장 (오염 없음)

| 사례 | 정보 팔이 고른 것 | 모델이 댄 근거(축자) |
|---|---|---|
| 003 t0·t1 | `Platinum Rewards Card` | *"meets all the customer's requirements including no foreign transaction fees, purchase protection, and a credit limit of at least $100,000"* |
| 024 t0 | `Business Platinum Rewards Card` | *"offers the **highest base cashback rate of 1.5%** and purchase protection, suitable for a large business expense like a work truck"* |
| 063 t0 (A_min) | `Bronze Rewards Card` | *"no annual fee, zero foreign transaction fees … meets the requirement for paper statements"* |

두 가지가 반복된다.
1. **랭킹을 필터로 되돌린다** — 003 은 *"요구를 전부 만족한다"* 로 끝난다. 요구를 만족하는 카드는
   여럿인데 그중 **무엇이 이 손님에게 더 이득인지**는 계산하지 않는다.
2. **가장 큰 요율 하나를 고른다** — 024 는 *"base cashback 1.5% 가 최고"* 라고 말한다. 그러나 이 태스크는
   금액이 **확정**돼 있다($40,000). 표만으로 순이익이 결정된다:

```
Business Bronze     base 1.0% · 연회비    0.0 → 40000×1% − 0     = **400.00**   (gold)
Green Rewards       base 1.0% · 연회비  100.0 →                    300.00
Business Silver     base 1.0% · 연회비  122.5 →                    277.50
Business Gold       base 1.0% · 연회비  200.0 →                    200.00
Business Platinum   base 1.5% · 연회비  450.0 → 40000×1.5% − 450 = **150.00**   ← 모델의 선택
```
⇒ 모델이 고른 카드는 **순이익 최하위**다. 요율은 가장 크고 순이익은 가장 작다 — **연회비와 금액을
   식에 넣지 않았다**. 이것은 산술 실패가 아니라 **목적함수를 세우지 않은 것**이고, x424 의 선행 실측
   (*"피연산자 맞음·결과 틀림 **0**/144"*)과 같은 방향이다.

⚠단 이 관측은 **024 한 사례**다(다른 셋은 gold 편중 때문에 채점이 무효). n=1.

## 3. 계기 결함과 다음 판의 조건

```
결함   표본 4 중 gold 3 이 동일 카드 ⇒ 상수 답이 3/4 를 얻는다. gold 로 채점하는 한 이 표본은 못 쓴다.
고칠 것 ⒜ gold 가 갈리는 표본으로 넓히거나
        ⒝ **결정론 참조**(손님이 준 수 × 표의 요율 − 연회비)로 채점하고 gold 는 참조 검증에만 쓴다
        ⒞ 부정통제는 그대로 유지 — 이번에 그것만이 오독을 막았다
```

## 4. ⛔인용 규율

```
"A_min 0/4 · B_full 1/4 · C_nopat 3/4"   **인용 금지** — 표본 gold 편중이 만든 수다
024 의 순이익 표(400 vs 150)               인용 가능(우리 A2 표 + 손님이 말한 $40,000 만으로 결정)
"모델은 목적함수를 세우지 않는다"           **n=1 관측**이다. 주장으로 올리려면 표본을 넓혀야 한다
```
