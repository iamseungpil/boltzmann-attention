# D6 설계서 rev2 — 카테고리 실효요율 주석 (클러스터① 최종-비교 층) · 2026-07-27

> **rev2(리뷰 반영)**: 결함1(동의어 토큰의 침묵 오표기)·사실정정2(9행·Platinum scope 모순)·명세보강3(엔진 변경 지점/규모·정보는 이미 노출됨·회귀 목록)을 [[08]] 규율대로 **코드 직접 검증 후** 전량 반영. ⚠리뷰 1건은 과장으로 확인·정정: `catalog_filter`를 직접 참조하는 테스트는 `test_compute.py` **하나뿐**(`test_compute_params.py`는 미참조·grep 확인).

> 사용자 지시(2026-07-27): "설계서 작성하고, 결과 보고 GO 결정한다."
> ⇒ **본 문서는 설계까지만. 구현 착수는 day3 완주 후 §6 GO 기준 충족 시.**
> 근거 = `DAY2_FRONT_FAIL7_RATESUB_FORENSIC_2026_07_26.md` · `STAGE2_GATE_DESIGN_2026_07_26.md` §11~§12 · 원장 C188/C189/C199/C202/C203.

## §1 문제 (측정된 것)
**필터는 맞는데 최종 비교에서 틀린다** — 클러스터①(002·003·006·024·063)의 공통 형태.
| 사례 | 필터 결과 | 실패 |
|---|---|---|
| 003(day3·[S]) | eligible에 Gold·Silver 둘 다(양쪽 다 하드제약 충족·연회비 $0) | 손님이 "지출 대부분 여행"이라 말했고 gold 정답=Silver(travel 4%) — 에이전트가 **여행 요율 비교를 제시하지 않아** 유저가 Gold(전 구매 2.5%) 선택 |
| 024(day2·C189) | 동일 구조 | 평면 `cashback` 열이 Business Silver 10%를 최고로 보이게 함(실제로는 travel/software 한정·해당 구매는 1%) |
| 003(day2) | 동일 | **통과** — 같은 상황에서 모델이 우연히 잘함 |
⇒ 구조가 보장하지 않고 **모델 변동에 맡겨져 있다**. day2 pass/day3 fail이 그 증거(pass^1).

**★rev2 핵심 정정 — 정보는 이미 "노출"돼 있었다**: `t2_compute.py:347`의 `facts` dict는 eligible/unverified
엔트리에 행의 전 필드(=`cashback`·`cashback_scope`·`base_cashback`)를 **이미 반환한다**. 003 day3 에이전트는
그 출력을 손에 쥐고도 여행 요율 비교를 제시하지 않았다. ⇒ **D6의 delta는 정보 추가가 아니라 "손님 카테고리에
맞춰 계산된 수치의 salience"**이며, "산문형 scope 문자열은 무시되지만 계산된 수치 병기는 다르다"는 **[M] 가설**이다
(006에서 soft 주석이 무시된 전례가 있으므로 낙관 금지). §8의 지표(`spend_category` 채움률·eligible 내 선택
정확도)가 이 가설의 판정기다.

**데이터 사실**: 판단에 필요한 값은 **이미 A2 표에 있다**(`cashback`·`cashback_scope`·`base_cashback`·C189가 넣음).
빠진 것은 "손님이 말한 카테고리에 각 카드가 실제로 몇 %인가"를 **계산해서 보여주는 단계**뿐이다.

**기존 레버가 못 닫는 이유**
- C189 = 데이터 + soft 주석. 006에서 같은 계열 soft 주석이 발화하고도 무시됨([[07]] 한계 실측).
- `recommendation_verify`(A2 실재) = `{offer_tool, action_tool, operand, research_tool}` — 추천/신청 **일치**만 봄. 003처럼 둘 다 eligible이면 무력. go_stack 활성 env도 없음.

## §2 설계 — 엔진은 **고르지 않고 계산만** 한다
1. `check_card_application_fit`에 **선택적 operand `spend_category`** 추가(모델이 손님 발화에서 formalize·[[10]]).
2. 엔진은 eligible 각 카드에 **그 카테고리의 실효 요율**을 붙여 반환:
   `Silver Rewards Card — rate for 'travel': 4.0% (other categories 1.0%)` / `Gold Rewards Card — rate for 'travel': 2.5%`
3. 규칙(전부 A2 선언값의 조회): 카드의 `category_rates`에 그 카테고리가 있으면 그 값, 없으면 `base_cashback`,
   `base_cashback`도 없으면 `cashback`(scope=all), 셋 다 없으면 **`unverified`**(추측 금지).
3b. **★rev2 결함1 수정 — 동의어 토큰의 침묵 오표기 차단**: 규칙 3의 base 폴백은 두 상황을 구분하지 못한다 —
   (a)그 카테고리가 실제로 base 요율(정당) vs (b)**모델 토큰이 키와 문자열 불일치**(손님이 "flights"라 해서
   `spend_category='flights'` → `{travel:4.0}`에 키가 없어 Silver가 1.0%로 **오표기**). (b)는 D4를 폐기시킨
   침묵-실패와 동종이고, 이번엔 **오표기 방향이 안전하지 않다**(정답 카드 요율을 깎아 gold 배제를 유도).
   ⇒ 두 겹 처방: **(필수)** A2 `params.spend_category` 설명에 **문서화된 카테고리 어휘를 열거**하고 그 중 하나로
   정규화하라고 지시(정규화=모델 몫·[[10]]·어휘=A2 소관) · **(권장)** 토큰이 **어느 카드의 `category_rates`에도
   없으면** base 폴백 대신 `no documented category-specific rate for '<token>'`으로 정직 표기(표에 없는 것을
   있는 것처럼 계산하지 않는 3버킷 철학·C185 동형).
4. `spend_category` 미제공 시 = **현행 그대로**(주석 없음·거동 변화 0).

### §2a A2 데이터 재구조화 (필수 선행)
현행 `cashback_scope`는 `"top_categories(travel/software)"` 같은 **문자열**이다. 엔진이 이걸 파싱하면
**엔진-formalize = [[03b]] 위반**이므로, 표를 선언형으로 바꾼다:
```
Silver Rewards Card: category_rates {travel: 4.0, software: 4.0}, base_cashback 1.0
Gold Rewards Card:   category_rates {},                            base_cashback 2.5   (scope=all의 선언형)
Green Rewards Card:  category_rates {...KB 축자...},               base_cashback 1.0
EcoCard / Business Gold / Business Platinum / Silver Zoom: 미문서 → 필드 없음(=unverified 유지)
```
- 대상 13행 중 **요율이 문서화된 9행**(rev2 정정: Platinum·Gold·Silver·Bronze·Crypto·Diamond·Business Bronze·
  Business Silver·Green / 미문서 4행=EcoCard·Business Gold·Business Platinum·Silver Zoom)만 손댄다.
  값은 **KB 축자**만(새 사실 발명 0·C189 원칙 유지).
- `cashback_scope` 문자열은 표시용으로 남기되 엔진은 `category_rates`/`base_cashback`만 읽는다.
- ⚠**rev2 정정 — KB 재확인 대상은 Green만이 아니다**: 표의 `_note_v2` 안에서 **C187 문구는 "Platinum 10%=
  travel/software", C189 문구는 "Platinum 10%는 전 구매", 행의 `scope`는 `all`** — provenance가 서로 모순이다.
  현행 scope 문자열을 그대로 옮기면 **모순이 선언형으로 굳는다**. ⇒ 재구조화 시 **9행 전수를 KB 원문과 대조**
  (최소 Platinum·Green은 필수)하고, 확정 불가한 행은 `category_rates`를 넣지 않고 unverified로 남긴다.

### §2b ★변경 지점·규모 (rev2 보강 — D4의 "엔진 무변경" 오기 재발 방지)
| 대상 | 위치 | 규모 |
|---|---|---|
| 엔진 | `t2_compute.py` `catalog_filter`(269행~) 내부의 **eligible/unverified 엔트리 구성부**(`facts` 산출 직후·347행 부근) | ~15줄(딕셔너리 조회+주석 문자열 1개·**리터럴 0**) |
| A2 | `check_card_application_fit.params`에 `spend_category` 항목 추가(어휘 열거 포함·§2-3b) | 1항목 |
| A2 | 표 재구조화: `cashback_scope` 문자열 → `category_rates`/`base_cashback` 선언형 | **9행**(§2a) |
- ctx-키→행-필드 매핑은 기존 관행과 동형(`max_annual_fee→annual_fee` 등 306~310행) — 새 패턴 도입 아님.

## §3 무엇을 하지 않는가 (경계)
- 카드를 **고르지 않는다**. 순위·추천·정렬도 하지 않는다 — 요율 주석만 붙인다.
- 손님의 카테고리를 엔진이 **추론하지 않는다**(모델이 operand로 준다). 미제공이면 아무 일도 안 한다.
- 요율을 **발명하지 않는다** — 표에 없으면 `unverified`로 정직하게 남긴다.

## §4 [[05]] 3질문 (설계서 상설 의무·[[17]])
1. **도메인-특화 순증?** A2 표의 **재구조화**가 주(문자열 scope → 선언형 category_rates)이고 새 도메인 규칙 추가는 없다.
   순증분 = 카테고리별 요율 9행(전부 KB 축자·이미 `cashback_scope`로 같은 사실이 표현돼 있던 것). 엔진 리터럴 0.
   전이 시 ABox-swap 대상(도메인마다 표만 교체·엔진 불변).
2. **유동 판단 동결?** 아니다 — (a) 손님 발화에서 어떤 카테고리가 결정축인지 = 모델 (b) 어느 카드를 추천할지 = 모델.
   엔진은 "선언된 표에서 그 카테고리의 값을 꺼내 보여주기"만 한다(산수도 아닌 조회).
3. **엔진이 도메인 행동 수행?** 아니다 — 반환 문자열에 사실을 노출할 뿐 도구를 호출하지 않는다.
⇒ 3질문 전부 no. 단 §7 Δspurious 계측 동반(레버는 하나 사면 하나 판다·등대 §1).

## §5 대안 검토 (기각 사유)
| 대안 | 기각 이유 |
|---|---|
| soft 주석 강화(“카테고리 요율을 비교하라”) | C189 주석이 006에서 무시됨 — 같은 층의 반복은 기대효과 없음([[07]]) |
| 엔진이 최적 카드 선택 | **답-주입**([[03b]] 위반)·모트 훼손 |
| `recommendation_verify` 확장(추천 전 카테고리 비교 문장 요구) | "비교 문장이 있는가"를 문자열로 판정 = 취약·오탐 위험. 정보를 **주는** 쪽이 견고 |
| 학습(learn)으로 닫기 | 유효하나 이번 사이클 범위 밖([[13]] 우선순위: scaffold/A2 최소 → 학습) |

## §6 ★GO / NO-GO 기준 (day3 완주 후 판정)
클러스터① 5건(002·003·006·024·063)의 day3 결과를 전수 확인하고:
- **GO**: 이 기전(필터 통과 후 최종-비교/제시 실패)으로 죽은 것이 **2건 이상**.
- **NO-GO(관찰 유지)**: 1건 이하 — 확률적 변동으로 두고 D6는 큐에 보관.
- 판정은 집계가 아니라 **궤적 전수 정독**으로([[08]]): 각 건이 (a)필터 미호출 (b)eligible 부정확 (c)**eligible 내 오선택**
  중 어디인지 분류하고, (c)만 D6 대상으로 센다.
- 현 시점 확정 재료: 003=(c) 1건. 006=발명 제약으로 eligible 왜곡(=D4′ 대상·D6 아님). 024·063·002는 실행 대기/무효.

## §7 오프라인 검증 계획 (GO 시)
1. 표 재구조화 후 **기존 필터 회귀 0**: 하드 제약 판정 결과가 재구조화 전후 동일(13행 전수 대조).
2. `spend_category='travel'` → Silver 4.0·Gold 2.5 주석, Business Silver 10.0(travel), EcoCard `unverified`.
2b. **★rev2 (b)경로**: 동의어 토큰(`spend_category='flights'`) → base 폴백으로 **오표기하지 않고**
   `no documented category-specific rate for 'flights'` 표기(결함1 처방의 직접 검정).
3. `spend_category` 미제공 → 반환 문자열 **바이트 동일**(거동 변화 0).
4. 미문서 카드에 카테고리 조회 → 추측 없이 `unverified`.
5. 회귀(rev2 보강): test_banking_gate·test_reffilter_credit·test_recommend_verify·test_c201_stage2·
   test_operand_grounding **+ `test_compute.py`**(`catalog_filter` 직접 검증·grep 확인된 유일 파일.
   ⚠리뷰가 든 `test_compute_params.py`는 catalog_filter 미참조로 확인 → 목록 제외).
6. 003 리플레이: 실제 인자 + `spend_category='travel'`로 도구를 호출해 **Silver가 최고 요율로 표시되는지** 확인
   (선택은 모델 몫이므로 여기서 검증하는 것은 *정보가 제대로 노출되는가*까지).

## §8 라이브 판정 지표 (다음 런)
- 클러스터① 태스크에서 `spend_category`가 실제로 채워지는 비율(모델이 이 채널을 쓰는가).
- 주석 노출 후 **eligible 내 선택 정확도** 변화.
- **Δspurious**: (a) 카테고리 오지정으로 엉뚱한 요율이 강조되는 사례 (b) 주석 추가로 반환문이 길어져 생기는 컨텍스트 비용
  (027형 창 초과 재발 여부) (c) 미문서 카드가 `unverified`로 밀려 후보에서 부당하게 배제되는지.

## §9 미해결
- 재구조화 9행의 KB 원문 대조(특히 **Platinum scope 모순**·Green `top_categories` 목록) = 구현 착수 시 선행(§2a).
- 손님이 카테고리를 **여러 개** 말한 경우(주 카테고리 판정)는 모델 몫으로 남긴다 — 엔진은 준 것만 조회.
- 이 레버는 **정보 제공**이므로, 모델이 정보를 보고도 틀리면 잔여는 scale/learn 축([[13]]/[[45]]·C202와 동형 논리).
