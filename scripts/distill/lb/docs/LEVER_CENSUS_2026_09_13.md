# 레버 센서스 — LB 코드베이스 전 팔 (2026-09-13)

> 스크립트 `scripts/distill/lb/lb_lever_census.py` · 입력 리모트 `x768/out_lb`(결과 gz 427 · 사이드카 408) + base `bank_x806_base_nt4`(`x818cloud_*` 제외).
> JOIN: sims 1,642 · 사이드카 행 있는 sim 1,218 · 사이드카 없는 셀 18 · infra 셀 1 제외.
> 읽는 법은 `DESIGN_60PCT_2026_09_13.md §1`. 여기는 표 원문.

## 팔 vs base (같은 태스크 · 양쪽 4-sim 셀)

```
lb    93 +4 | n     52 +6 | sw    47 -19 | fs    29 +5 | dr    14 -1 | v2    14 -11 | c098  13 -11 | nc18  13 -5
cb    10 -15| ub     8 -10| mn     6 -8  | tb     6 -11| nc17   5 -4 | nc23   5 -4  | nc25   4 -1  | v5     4 -6
x8     4 -5 | fix    3 -2 | nc18r  3 +2  | nc19   3 -4 | nc23r  3 -1 | sb     3 -4  | sc     3 -3  | t723   3 0
nc20   2 0  | nc21   2 -2 | nc22   2 0   | nc22r  2 -2 | nc11   2 -6 | nc12   2 -7  | nc     2 -3  | (1-2 태스크 팔 생략)
```

nc 팔 태스크별 (ours/base):
```
nc18   004:4/4 008:3/4 016:1/4 018:2/2 036:2/2 038:1/1 040:1/2 047:4/4 049:1/3 054:1/2 059:3/2 066:3/0 081:1/2
nc17   003:4/2 004:1/4 007:4/4 016:2/4 070:1/2
nc23   004:3/4 008:4/4 049:1/3 054:3/2 081:0/2      nc23r 008:4/4 054:3/2 081:0/2
nc25   004:4/4 036:0/2 049:1/3 066:3/0              nc18r 040:3/2 049:0/3 066:4/0
nc19   004:4/4 016:2/4 040:0/2                      nc19r 040:1/2
nc20   016:3/4 066:1/0                              nc20r 066:4/0
nc21   043:4/4 049:1/3
nc22   036:2/2 038:1/1                              nc22r 036:0/2 038:1/1
nc11   004:1/4 016:1/4   nc12 004:0/4 016:1/4   nc 007:4/4 016:1/4
fs     007:1/4 012:4/3 016:1/4 018:3/2 019:3/4 023:4/4 028:2/3 031:4/3 033:3/2 036:2/2 038:2/1 040:0/2 043:4/4 045:3/1 048:0/3 049:4/3 054:1/2 058:4/4 070:1/2 072:4/3 073:4/3 074:4/0 075:4/0 079:2/0 081:0/2 094:2/0 098:4/4 099:2/2
```

## 레버 (pooled = 모든 팔 운/조용 · paired = 같은 팔·태스크 안 짝 · 부호검정 양측)

```
LB   lever                                     fired  win%  | quiet  win% | pairs  dpp   bet/wor  p     판정
LB1  advice procedure ([PROCEDURE]/[ORDER])      299  51.2  | 1343  52.8  |  62   +5.9   19/19   1.00  잡음
LB1  deny procedure                               96  28.1  | 1546  54.0  |  17   -7.8    2/3    1.00  잡음(049 9%/36% · 048 0/16 vs 4/12)
LB2  tool verify_identity                        666  44.1  |  976  58.2  |   6  +27.8    3/0     .25  짝 6 — 팔 A/B 로만
LB2  tool check_card_application_fit             153  75.8  | 1489  50.1  |  30   -6.1    7/10    .63  부작용 기울기
LB2  tool check_savings_account_fit               80  73.8  | 1562  51.4  |   6  +33.3    3/1     .62  잡음
LB2  tool check_credit_dispute_provisional_credit 77  29.9  | 1565  53.6  |  25   +4.7    9/7     .80  잡음
LB2  tool get_reward_discrepancies (+inject)      70  41.4  | 1572  53.0  |   2  -50.0    0/1     -    측정불가
LB2  tool get_debit_dispute_liability_cap         43   2.3  | 1599  53.8  |   3    0.0    0/0     -    측정불가(HARD 전용)
LB2  tool check_checking_account_fit              41  75.6  | 1601  51.9  |   5  +13.3    2/2    1.00  잡음
LB2  tool check_card_closure_eligibility          37  45.9  | 1605  52.6  |  25   +8.7    9/5     .42  효과 기울기
LB2  tool check_rebate_qualification              37  89.2  | 1605  51.7  |   1  +66.7    1/0     -    측정불가
LB2  tool check_business_checking_fit             30  53.3  | 1612  52.5  |   0     -     0/0     -    측정불가
LB2  tool check_referral_options                  29  69.0  | 1613  52.2  |   6  -11.1    1/2    1.00  잡음
LB2  tool get_correct_savings_apy                 21  61.9  | 1621  52.4  |  12  -20.8    1/4     .38  부작용 기울기
LB2  tool get_atm_fee_discrepancies               18 100.0  | 1624  52.0  |   1    0.0    0/0     -    측정불가
LB2  tool check_business_savings_fit               8  62.5  | 1634  52.4  |   0     -     0/0     -    측정불가
LB2  tool check_cli_eligibility                    3 100.0  | 1639  52.4  |   3  +44.4    2/0     .50  측정불가
LB2  tool get_checking_atm_fee_totals              2 100.0  | 1640  52.4  |   2    0.0    0/0     -    측정불가
LB3  deny grounding                                79  48.1  | 1563  52.7  |   8  +10.4    3/1     .62  잡음
LB3  release grounding (WRITE-EVIDENCE)            11  45.5  | 1631  52.5  |   5  +50.0    3/0     .25  효과 기울기
LB3  deny FREE-TEXT-DEFAULT                        52  55.8  | 1590  52.4  |   0     -     -       -    셀 전부 발화 → 팔 A/B(066)
LB3  deny name-registry                             5  20.0  | 1637  52.6  |   5   +6.7    1/1    1.00  잡음
LB4  advice claims ([CLAIM-PROVENANCE])           323  40.9  | 1319  55.3  | 130   -5.8   29/37    .39  부작용 기울기
LB4  advice settled-rows                           70  41.4  | 1572  53.0  |   2  -50.0    0/1    1.00  측정불가
LB4  advice follow-up                              14  64.3  | 1628  52.4  |   2  -50.0    0/1    1.00  측정불가
LB5  advice leaving ([LEDGER])                    399  49.9  | 1243  53.3  |  28   -4.2    7/10    .63  잡음
LB5  deny search-repeat                             4  25.0  | 1638  52.6  |   3  -50.0    0/2     .50  측정불가
LB5  advice PROTOCOL                                2   0.0  | 1640  52.6  |   2    0.0    0/0     -    측정불가
LB6  fold                                         331  34.1  | 1311  57.1  | 104   -2.4   28/30    .90  잡음(pooled 차는 선택 효과)
LB7  advice write-rule                            244  41.4  | 1398  54.4  |  25   +7.3   10/7     .63  효과 기울기
LB7  advice action-index                          164  42.1  | 1478  53.7  |  23  -10.1    5/10    .30  부작용 기울기
LB7  advice value-acquire                          62  35.5  | 1580  53.2  |  26   +1.3    7/7    1.00  잡음
배관  tools(주입)                                 1128  52.0  |  514  53.7  |   0     -     -       -    상존 · 검정 불가
배관  ask                                        1118  52.1  |  524  53.2  |   0     -     -       -    상존 · 검정 불가
배관  regen                                       697  45.6  |  945  57.6  |  63   -3.4   14/16    .86  잡음
```

## 태스크 (모든 팔 합산 ours n≥8 vs base · |Δ|≥25pp · Fisher 한쪽꼬리)

```
GAIN  045 1/4→6/12 .392 | 055 2/4→7/8 .236 | 059 2/4→10/12 .245 | 062 0/4→2/8 .424 | 066 0/4→18/28 .028 | 067 0/4→3/8 .255
      069 0/4→2/8 .424 | 072 3/4→12/12 .250 | 074 0/4→7/8 .010 | 075 0/4→12/12 .001 | 079 0/4→4/8 .141 | 094 0/4→4/8 .141
LOSS  004 4/4→21/36 .138 | 007 4/4→58/80 .289 | 014 3/4→5/12 .285 | 016 4/4→46/120 .025 | 019 4/4→18/24 .357
      028 3/4→7/16 .291 | 044 4/4→9/12 .393 | 048 3/4→10/44 .055 | 049 3/4→16/94 .022 | 081 2/4→4/32 .121
pooled ours 862/1642 = 52.5%  vs base 같은 태스크 192/380 = 50.5%
```

## 태스크 안 2×2 (운 sim 승/전 · 조용 sim 승/전) — 강건 6 + 004 007

```
066 (28 sim, 18 승; lb 0/4 n 3/4 nc18 3/4 nc18r 4/4 nc20 1/4 nc20r 4/4 nc25 3/4)
  verify_identity 18/24 vs 0/4 · check_savings_account_fit 18/22 vs 0/2 · leaving 14/19 vs 1/1 · write-rule 8/11 vs 7/9 · fold 1/4 vs 10/12 · action-index 0/2 vs 7/10
074 (8, 7; fs 4/4 lb 3/4)   get_atm_fee_discrepancies 4/4 vs 3/4 · action-index 4/4 vs -
075 (12, 12; fs lb n 4/4)   check_checking_account_fit 8/8 vs - · 전부 승
016 (120, 46; 30 팔 0~3/4)  verify_identity 26/76 vs 2/4 · leaving 2/10 vs 4/10 · procedure 7/11 vs 9/31 · fold 5/9 vs 4/17
049 (94, 16; fs 4/4 그 외 0~2/4) deny:LB1:procedure 4/43 vs 4/11 · fold 12/73 vs 4/9 · leaving 13/42 vs 0/10 · write-rule 16/59 vs 0/1
048 (44, 10; 0~2/4)         deny:LB1:procedure 0/16 vs 4/12 · check_card_closure_eligibility 5/6 vs 5/38 · procedure 7/25 vs 3/15
004 (36, 21; lb nc18 nc19 nc25 4/4 · nc11 1 nc12 0 nc17 1 sw 0) leaving 16/31 vs 1/1 · procedure 17/30 vs 0/2 · claims 7/16 vs 14/20
007 (80, 58)                check_card_application_fit 14/23 vs 11/17 · 우리 문장 거의 0
```

## 한계
- 레버 수준 유의 0 — 53 레버에서 p<.05 우연 기대 2~3(메모리 85). 아무것도 갈리지 않았다는 것이 결과다.
- 상존 레버(주입·ask·verify_identity)는 짝이 없어 검정 불가. 근거는 팔 A/B(nc9/15/17)뿐.
- 팔은 코드가 다르다 — pooled 합은 「우리 층이라는 부류」의 값이지 특정 sha 의 값이 아니다(`allarms2.py` 주의와 같음).
- 사이드카 없는 sim 424 는 「조용」으로 세었다(회수 누락 18 셀 포함) — 보수적 방향.
