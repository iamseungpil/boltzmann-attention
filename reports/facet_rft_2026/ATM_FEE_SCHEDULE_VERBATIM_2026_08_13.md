# ATM 수수료 스케줄 축자 추출 — get_atm_fee_discrepancies 저작 근거 (2026-08-13)

> 출처 규율([[23]]): **documents/ 정책 문서만** 열람(tasks.json/gold 미접촉·추출 에이전트 지시문에
> 명시·준수 확인). db.json 은 레코드 스키마 확인 용도만. 모든 수치에 문서 ID 병기.
> 소비처: `scaffold_get_tools` 신규 항목 `get_atm_fee_discrepancies` 의 op 상수( x288 판정
> A_DOCS 0/8 → (B) 확정 — `PROBE_X283_X285_DESIGN` §x288 ).

## 클래스별 스케줄 (요약 — 축자 인용 전문은 §끝 부록 참조)

| 클래스 | out-of-network | foreign | 리베이트 | 출처 문서 |
|---|---|---|---|---|
| blue_account | 1%·max $3.00 | max(3%, $5.00)/건·USD 환산 기준 | 없음 | blue_001·blue_012 |
| bluest_account | $2.00 flat | **$0.00** | 월 $50 캡(third-party) | bluest_010·_003·_007 |
| green_account_(checking) | $3.00 flat | max(3%, $5.00) | 없음 | green_010(_001/_002 동일)·green_012 |
| light_green_account | 월 4회 무료 후 $1.50 | tier: ≤$100→$2.00·≤$300→$3.50·>$300→$5.00 (경계=하위 tier) | 없음 | lg_001·lg_013 |
| light_blue_account | 월 2회 무료 후 $2.50 | 월 2회 무료 후 $4.00/건(금액 무관) | 없음 | lb_004·lb_006 |
| purple_account | $2.50 flat | **$0.00**·FX 0% | operator fee 월 $30 캡 | purple_012·_001·_004·_010 |
| dark_green_account | 1%·min $1.50 | 2.5%·max $6.00 | 없음 | dg_001·dg_002 |
| evergreen_account | 1%·max $2.50 | 2%·min $3.00 | 없음 | eg_008·eg_001 |

(문서 ID 접두 `doc_checking_accounts_` 생략 표기.)

## 거래 레코드 스키마 (doc_bank_accounts_bank_accounts_(general)_018 + db 실측 대조)

- 필드: `transaction_id`·`account_id`·`date`(MM/DD/YYYY)·`description`·`amount`(음수=출금)·`type`·`status`
- ATM 관련 type: `atm_withdrawal`·`atm_fee`·`fee_rebate`(·`rebate_credit`·`fee_refund`)
- **fee 라인 description 이 적용 공식을 자기-기술**(17종 실존): `"NON-RHO ATM FEE - 1% (MAX $3.00)"`
  ·`"FOREIGN ATM FEE - 2.5% (MAX $6.00)"`·`"FOREIGN ATM FEE - TIER 2 ($101-$300)"` 등
- 인출 description 에 네트워크 식별: `"ATM WITHDRAWAL - RHO-BANK #4521 ..."` vs `"- CHASE BANK ..."`
- **env 자체가 심은 불일치 레코드 = `_err` 접미 id**(예: `btxn_ar_dg_02f_err` — RHO-BANK 인출에
  NON-RHO FEE) ⇒ **오프라인 검증 기준으로 사용 가능(gold 불요·env 기계도출)**

## 저작 시 모호점 11 (추출 에이전트 원문 유지 — op 범위 결정에 사용)

1. fee↔withdrawal 링크 필드 부재(같은 날짜·인접 추론) 2. % 기준=|amount|(반올림 규칙 문서 없음)
3. in-network $0 조항 부재(purple_007 상담 스크립트 발화뿐) 4. bluest 리베이트 대상(third-party vs
aggregate 문서 간 긴장) 5. purple operator-surcharge 합산형 식별 불가 6. 리베이트=별도 fee_rebate
라인(감액 아님·건별/월합 미규정) 7. calendar month vs statement cycle 8. light_blue OON/foreign
무료 풀 공유 여부 미규정 9. foreign+OON 중복 부과 미규정(db 실측=건당 atm_fee 1건)
10. light_green 무료 4회의 foreign 적용 여부 미규정 11. 리베이트 지연/누락 완충 조항
(general_019 — 부재≠위반)

## op 설계 결정 (모호점 반영·최소 범위)

- **1차 술어 = 라벨-대 스케줄 대조**: fee 라인 description 의 자기-기술 공식(패턴 17종) vs
  해당 계좌 클래스의 정답 스케줄 — 클래스 오적용(타 클래스 공식 라벨)·NON-RHO fee 인데 짝
  인출이 RHO-BANK 인 경우(네트워크 모순)를 discrepant 로.
- **2차 술어 = 금액-공식 재계산**: 라벨이 옳아도 amount ≠ 공식(짝 인출 |amount| 기준) 이면
  discrepant. 짝 = 같은 계좌·같은 날짜의 직전 atm_withdrawal(모호점 1 — 미짝 fee 는 판정 보류
  로 반환하지 않음·과차단 방지).
- ~~**무료 횟수/리베이트 캡(모호점 4·7·8·10)은 1판 범위 밖**~~ — **2026-08-24 해제**(아래 §갱신).
  반환은 rewards 판 선례대로 **id 목록만**(금액 합산은 모델 몫 — [[62]] 최소 결정론).
- 검증: db.json 의 `_err` 집합과 오프라인 대조(gold 미접촉).


---

## 갱신 2026-08-24 — 유보했던 축이 열렸다 (무료 횟수 · 리베이트 캡 · 부재)

위 §op 설계 결정이 *"1판 범위 밖"* 으로 미뤄 둔 축을 `get_atm_fee_discrepancies` 에 넣었다.
**새로 읽은 문서는 없다** — 아래 값은 전부 이 문서가 2026-08-13 에 이미 표로 적어 둔 것이다.

| 축 | 값 | 출처(이 문서 §클래스별 스케줄) |
|---|---|---|
| light_green 타행 무료 | 월 4회 → 이후 $1.50 | lg_001 |
| light_blue 타행 무료 | 월 2회 → 이후 $2.50 | lb_004 |
| light_blue 해외 무료 | 월 2회 → 이후 $4.00/건 | lb_006 |
| bluest 리베이트 | 월 $50 캡 | bluest_001·_003·_007 |
| purple 리베이트 | 월 $30 캡 | purple_001·_004·_010 |

### 모호점 처리 (추측으로 채운 것이 아니라 문서 구조가 답을 준 것)

```
⑧ light_blue 타행·해외 무료 풀 공유 → **분리**. 두 축이 서로 다른 문서에 각각 '2 free' 로 선언돼 있다
⑩ light_green 무료 4회의 해외 적용   → **타행에만**. 해외는 lg_013 의 TIER 표가 전액을 규정한다
⑥⑦ 리베이트 건별/월합·cycle 경계     → **인출 1건당 문서요율**만큼이 채무·상한은 레코드 순서로 누적
                                     (코퍼스는 전부 단월이라 calendar↔statement 경계는 행사되지 않는다)
④ bluest 'third-party' 문언의 대상    → 우리 요율 라인. 원장이 NON-RHO ATM FEE 5건에 1:1 로 REBATE 를 단다
①  fee↔withdrawal 링크 부재           → 여전히 **LLM 몫**(인접 레코드 페어링). 엔진은 페어링을 안 만든다
```

### 부재(줄이 아예 없는 인출)

이 문서 §op 설계 결정의 *"미짝 fee 는 판정 보류"* 는 **fee → withdrawal** 방향이었다.
반대 방향(**withdrawal → fee 없음**)이 실패의 절반이었고, 그래서 입력 단위를 `atm_fee` 라인에서
**ATM 인출**로 올렸다. 요율이 0 이 아닌데 수수료 줄이 없으면 그 차액은 **음수 보정**이다.

### 검산

`scripts/distill/tau2/test_atm_ledger_close.py` — 072·073·074 **9계좌 전수**가 gold 와 정확히 일치.
⚠규칙은 위 표(2026-08-13 축자)로 짓고 **나서** gold 로 대조했다. 9계좌가 한 번에 맞았다([[23]]·[[69]]).

### 표적 재계수 (2026-08-24)

ATM 거래가 있는 계좌 = **9개 · 등급 8종** = 이 문서가 스케줄을 가진 8종과 정확히 일치.
Silver·Bronze·Navy Blue·Cobalt Blue 는 계좌는 있으나 ATM 거래 **0건** ⇒ 스케줄을 저작할 표적이 없다
(`default: null` 이라 오면 기권한다·[[31]] 규칙 4 "표적 실재 선행").
