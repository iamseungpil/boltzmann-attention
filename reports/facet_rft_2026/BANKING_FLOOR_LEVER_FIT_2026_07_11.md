# BANKING FLOOR × RETAIL 레버 스택 — LEVER-FIT 전수 분류 (2026-07-11)

> **질문**: banking floor(`bankxfer_floor_bank_t4` trial 0·1 = nt2 정본) 실패 170건을, 현재 retail canonical 전-레버 스택(GB1 게이트 + prov per-arg + DISAMB nested/calc)이 "발화해 결과를 바꿨을" 것 vs 아닌 것으로 엄밀 분류.
> **방법**([[08]]): 스크립트 1차 버킷(`scripts/distill/tau2/banking_lever_fit_census.py` · per-case JSON `sim_results/banking_lever_fit_percase.json`) → **12개 task-trial 전문 정독 + 4개 상세 부분정독**으로 판정 검증·정정(v2 플래그로 반영). 정독이 v1 판정을 5곳에서 뒤집음(§2.7).
> **데이터**: n=192(trial 0·1), pass 9(4.7%), fail 170, **infra(reward None) 13 제외 계상**. gold = gz 내 `tasks` 키(simulations[].task_id 조인·확보 성공). trial 2(19 sims)는 비정본·미사용.
> **레버 발화조건** = 과제 지시서의 기계 판정 기준 10종. **발화 ≠ 복구** — '발화확실+복구개연'과 '발화만'을 분리했고, sim-폐쇄는 all-or-nothing DB 채점이라 **모든** blocking diff가 커버돼야 함.

---

## §0 요약표

### 0.1 레버별 발화·닫힘-후보 (170 fails · sim 단위 다중라벨)

| 레버 | 발화(확실) sims | % | 그중 복구개연 | 정독 검증 | 판정 |
|---|---|---|---|---|---|
| **1 PERARG(현행 엔진·전-호출 스캔)** | **71** | **41.8%** | 부분(아래 분해) | 016·012·038·041·065·096 | **최대 발화 레버 — 단 복구는 조건부** |
| ├ 도구명 날조(agent/discoverable_tool_name·'name'힌트) | 61 | 35.9% | KB재검색 필요·개연 중 | 012·016·096·065 | 발화확실. env도 error를 주는데 못 살아났음 → deny+fetch-지시의 한계 개연 |
| ├ placeholder id(user123·계정id 등) | ~10 | ~6% | getter 존재·개연 상 | 016(user123)·065(rp65a7b3c4-lightblue) | 발화확실+복구개연. 단 065는 env-error로 **자가복구**했음(레버 이득≈턴 절약) |
| └ gold-diff 기준(엄밀): fired 19 / 복구개연 4 | 19 | 11.2% | 4 (2.4%) | 041 | txn_id 날조 등 — gold값이 문맥에 없어 regen 단독으론 부족·fetch 경로는 존재 |
| **2 GATE GB1(auth·유일 게이트)** | 6 | 3.5% | 폐쇄-단독 0 | 016·096 | 발화확실 소수 — 실패 대다수가 gated 도구 **도달 전** 사망. 016 t0은 연쇄-재궤도 개연(§2.2) |
| **3 CALC-EXT(argmax/count/sum)** | 수치-diff 20 | 11.8% | **진성 calc-shape 0~2** | 038·041·062·021 | **정독이 대부분 뒤집음**: 수치 diff의 실체 = last4 placeholder 날조(fetch-형)·미확인 가정(confirm-형)·계산이 아님 |
| **4 DISAMB(\|C\|≥2 silent P-B)** | 36 | 21.2% | 불확실(기준이 정책-semantic) | 038(card_action)·065(account_class)·051 | 발화후보 실재하나 다수는 문맥-실재 후보 간 **정책 기준** 선택 → subcall 정답률 미지수 |
| **5 P2 원리-디폴트** | **0 (구조적)** | 0% | — | — | banking A2에 default_specs 부재. 단 **잠재 spec 발견**: issue_noticed_date←현재시각(041) |
| **6-8 E-PLAN L1/L2/walk** | 27/38/66 | 16/22/39% | **대부분 spurious** | 082·021·026 | 정독: RPTGAP의 지배 원인은 조기이관·발견실패에 **종속** — 열거자-강제로 안 닫힘. banking의 진짜 E-PLAN 표적 = **discovery-read 강제**([[14]] 정합) |
| **9 NOTICE/EXCLUSIVITY** | 0 (구조적) | 0% | — | — | banking A2 notice 미인스턴스·정책에 고정문구 없음 |
| **10-신규 LOGV_TIME_FAB** (A2 identifying_arg_types 확장 표적) | **59** | **34.7%** | producer=get_current_time 결정론 | 043·065 | time_verified 날조(과거시각 지어냄) = DB 단독-치명. 현행 힌트 사거리 **밖**·ABox-only 확장으로 편입 가능 |
| **10-신규 EARLY_TRANSFER** (retail G_EXHAUST kind 확장 표적) | **36** | **21.2%** | 이관차단≠체인완주 | 082·016 | gold에 없는 transfer 실행. exhaust 게이트는 이관은 막아도 15-step 체인 완주는 별개 |

### 0.2 실패 1차 버킷 (스크립트·정독 검증)

| 버킷 | sims | % | C52(frontier 17모델) 대조 |
|---|---|---|---|
| REACH/조립(미실행≥⅓) | 130 | 76.5% | frontier 발견/조기중단 31% → **모델-기울기 재현**(최약 gemini25pro 68% 미실행·§3.2f(5)) |
| ARGDIFF(완주-후-불일치) | 23 | 13.5% | frontier 45% (coverage≥0.8 기준으론 우리 16% vs frontier 45%) |
| MIXED | 16 | 9.4% | — |
| EXTRA_WRITE 단독 | 1 | 0.6% | 신규 마이크로 유형(§3.3) |

### 0.3 ★sim-폐쇄 정직 상한 (all-or-nothing: 모든 blocker 커버 필요)

| 티어 | 폐쇄 상한 | 비고 |
|---|---|---|
| **T1 현행 스택 그대로**(GB1+prov+DISAMB) | **0 / 170** | 발화는 71 sims에서 일어나나, 잔여 blocker(미실행 체인·초과 write)가 전 sim에 공존 |
| **T2 +A2-only 확장**(time힌트·calc·exhaust) | **≤1 → 실질 0** | 기계 판정 1건(task_062 t0)도 정독상 동일-key 초과-transfer가 DB를 이미 오염 → 불폐쇄 |
| 연쇄(cascade) 개연 — 정적 분석 밖 | 소수(~2-4 sims) | deny→재궤도가 체인 전체를 살리는 시나리오: 016(placeholder deny→verify→올바른 조회), 038(last4 deny→user-tool 체인 유도) |

**결론 한 줄**: retail 레버 스택은 banking floor에서 **자주 발화하지만(fail의 ~42%) 거의 닫지 못한다(정적 0, 연쇄 포함 낙관 ~1-2%p)** — 지배 잔여는 REACH/조립(발견 체인)로, DOMAIN_TRANSFER §1.4의 예측(BC2 = 결정론 controller 필요·게이트 아님)을 per-case로 **확정**한다.

---

## §1 per-task 지도 (91 실패 task × trial 0/1)

표기: `버킷 m미실행/골드[플래그]`. 플래그: GATE=GB1 발화 · PROVall=PERARG 전-호출 발화 · FABTOOL=도구명 날조 · TIMEFAB=time_verified 날조 · EARLYTR=gold-밖 transfer · DISAMB=|C|≥2 후보 · NUMDIFF=수치 diff · RPTGAP=gold 반복-write 미충족 · USERarg=유저측 write 인자오류. (전체 per-diff 세부 = `sim_results/banking_lever_fit_percase.json`)

| task | trial 0 | trial 1 |
|---|---|---|
| task_001 | ARGDIFF m0/1[USERarg] | ARGDIFF m0/1[USERarg] |
| task_003 | ARGDIFF m0/1[USERarg] | ARGDIFF m0/1[USERarg] |
| task_004 | REACH m1/1[PROVall,FABTOOL] | ARGDIFF m0/1[PROVall,FABTOOL] |
| task_005 | REACH m2/3[TIMEFAB] | ARGDIFF m0/3[TIMEFAB] |
| task_007 | ARGDIFF m0/1[USERarg] | ARGDIFF m0/1[USERarg] |
| task_008 | ARGDIFF m0/1[-] | ARGDIFF m0/1[PROVall,FABTOOL] |
| task_010 | EXTRAW m0/2[-] | — (pass) |
| task_012 | REACH m1/1[PROVall,FABTOOL] | REACH m1/1[-] |
| task_014 | REACH m1/1[PROVall,FABTOOL] | REACH m1/1[PROVall,FABTOOL] |
| task_015 | ARGDIFF m0/2[PROVall,USERarg] | ARGDIFF m0/2[PROVall,USERarg] |
| task_016 | REACH m1/2[GATE,PROVall,FABTOOL,EARLYTR] | REACH m1/2[-] |
| task_017 | REACH m4/4[EARLYTR,RPTGAP] | REACH m4/4[GATE,RPTGAP] |
| task_018 | REACH m8/8[PROVall,EARLYTR,RPTGAP] | REACH m8/8[EARLYTR,RPTGAP] |
| task_019 | REACH m6/6[EARLYTR,RPTGAP] | REACH m6/6[GATE,PROVall,EARLYTR,RPTGAP] |
| task_020 | REACH m6/6[EARLYTR,RPTGAP] | REACH m6/6[EARLYTR,RPTGAP] |
| task_021 | REACH m3/4[RPTGAP] | REACH m3/4[RPTGAP] |
| task_022 | REACH m12/12[GATE,PROVall,EARLYTR,RPTGAP] | REACH m12/12[EARLYTR,RPTGAP] |
| task_023 | REACH m2/2[PROVall,EARLYTR] | REACH m1/2[PROVall,FABTOOL] |
| task_024 | ARGDIFF m0/1[USERarg] | ARGDIFF m0/1[USERarg] |
| task_025 | ARGDIFF m0/1[USERarg] | ARGDIFF m0/1[USERarg] |
| task_026 | REACH m11/11[RPTGAP] | REACH m11/11[EARLYTR,RPTGAP] |
| task_027 | REACH m6/6[EARLYTR,RPTGAP] | REACH m5/6[RPTGAP] |
| task_028 | REACH m13/15[GATE,DISAMB,NUMDIFF,RPTGAP] | REACH m15/15[EARLYTR,RPTGAP] |
| task_029 | REACH m6/8[RPTGAP] | REACH m8/8[EARLYTR,RPTGAP] |
| task_031 | REACH m3/5[TIMEFAB] | REACH m3/5[TIMEFAB] |
| task_032 | REACH m4/5[RPTGAP] | REACH m4/5[RPTGAP] |
| task_035 | REACH m2/3[PROVall] | REACH m2/3[PROVall,FABTOOL] |
| task_036 | REACH m2/3[PROVall,FABTOOL,TIMEFAB] | REACH m2/3[PROVall,FABTOOL,TIMEFAB,EARLYTR] |
| task_037 | MIXED m2/8[PROVall,FABTOOL,TIMEFAB,NUMDIFF] | REACH m7/8[TIMEFAB,RPTGAP] |
| task_038 | MIXED m2/9[DISAMB,NUMDIFF] | MIXED m2/9[PROVall,FABTOOL,DISAMB,NUMDIFF] |
| task_040 | REACH m5/15[TIMEFAB,DISAMB,NUMDIFF,RPTGAP] | REACH m5/15[TIMEFAB,DISAMB,NUMDIFF,RPTGAP] |
| task_041 | MIXED m7/25[TIMEFAB,DISAMB,NUMDIFF,RPTGAP] | MIXED m7/25[TIMEFAB,DISAMB,NUMDIFF,RPTGAP] |
| task_043 | REACH m14/15[TIMEFAB,EARLYTR] | REACH m14/15[PROVall,FABTOOL,TIMEFAB] |
| task_044 | REACH m9/10[-] | REACH m9/10[PROVall,FABTOOL,EARLYTR] |
| task_045 | REACH m10/15[PROVall,FABTOOL,TIMEFAB] | REACH m14/15[PROVall,FABTOOL,TIMEFAB] |
| task_046 | REACH m2/3[PROVall,FABTOOL] | REACH m2/3[PROVall,FABTOOL,TIMEFAB] |
| task_047 | REACH m12/15[PROVall,FABTOOL,RPTGAP] | REACH m14/15[RPTGAP] |
| task_048 | REACH m17/24[TIMEFAB,RPTGAP] | — (infra) |
| task_049 | — (infra) | REACH m14/19[RPTGAP] |
| task_050 | REACH m12/13[-] | REACH m12/13[TIMEFAB] |
| task_051 | REACH m16/20[DISAMB,NUMDIFF] | REACH m16/20[DISAMB,NUMDIFF] |
| task_052 | REACH m10/13[TIMEFAB] | REACH m10/13[TIMEFAB] |
| task_053 | REACH m11/16[PROVall,FABTOOL,NUMDIFF] | REACH m15/16[-] |
| task_054 | REACH m12/17[PROVall,FABTOOL,TIMEFAB,DISAMB,NUMDIFF] | REACH m10/17[PROVall,FABTOOL,TIMEFAB,DISAMB,NUMDIFF] |
| task_055 | — (infra) | REACH m4/8[TIMEFAB,EARLYTR,DISAMB] |
| task_056 | REACH m6/8[RPTGAP] | — (infra) |
| task_057 | MIXED m2/7[TIMEFAB,DISAMB] | REACH m4/7[PROVall,FABTOOL,DISAMB] |
| task_058 | REACH m2/4[PROVall,FABTOOL,USERarg] | REACH m3/4[-] |
| task_059 | REACH m5/6[PROVall,FABTOOL] | REACH m6/6[PROVall,FABTOOL] |
| task_060 | MIXED m2/7[DISAMB] | ARGDIFF m0/7[DISAMB] |
| task_061 | REACH m3/9[PROVall,FABTOOL,EARLYTR] | REACH m6/9[PROVall,FABTOOL,EARLYTR] |
| task_062 | ARGDIFF m0/13[NUMDIFF] | MIXED m3/13[TIMEFAB,RPTGAP] |
| task_063 | ARGDIFF m0/4[PROVall,FABTOOL,DISAMB,USERarg] | REACH m2/4[PROVall,FABTOOL,USERarg] |
| task_064 | REACH m3/4[PROVall,FABTOOL,USERarg] | REACH m3/4[PROVall,FABTOOL] |
| task_065 | ARGDIFF m0/6[TIMEFAB,DISAMB] | ARGDIFF m0/6[TIMEFAB,DISAMB] |
| task_066 | MIXED m1/9[TIMEFAB,DISAMB] | REACH m3/9[PROVall,FABTOOL,TIMEFAB,DISAMB] |
| task_067 | REACH m4/9[PROVall,DISAMB] | — (infra) |
| task_068 | MIXED m2/7[PROVall,FABTOOL,TIMEFAB,DISAMB,USERarg] | MIXED m2/7[PROVall,FABTOOL,TIMEFAB,DISAMB,USERarg] |
| task_069 | MIXED m2/7[PROVall,TIMEFAB,DISAMB,USERarg] | MIXED m2/7[PROVall,FABTOOL,TIMEFAB,DISAMB,USERarg] |
| task_070 | REACH m4/5[PROVall,FABTOOL] | REACH m4/5[-] |
| task_071 | — (infra) | REACH m2/6[-] |
| task_072 | REACH m8/9[TIMEFAB,RPTGAP] | REACH m3/9[TIMEFAB,EARLYTR,RPTGAP] |
| task_073 | — (infra) | REACH m6/11[RPTGAP] |
| task_074 | REACH m12/13[TIMEFAB,EARLYTR,RPTGAP] | — (infra) |
| task_075 | REACH m2/3[PROVall,FABTOOL] | ARGDIFF m0/3[PROVall,FABTOOL] |
| task_076 | REACH m3/3[PROVall,FABTOOL] | REACH m2/3[TIMEFAB] |
| task_077 | REACH m14/23[PROVall,FABTOOL,DISAMB,RPTGAP] | REACH m13/23[TIMEFAB,DISAMB,RPTGAP] |
| task_078 | REACH m15/21[TIMEFAB,RPTGAP] | MIXED m4/21[TIMEFAB,DISAMB,NUMDIFF,RPTGAP] |
| task_079 | MIXED m4/24[TIMEFAB,DISAMB,RPTGAP] | MIXED m7/24[TIMEFAB,RPTGAP] |
| task_081 | REACH m28/33[RPTGAP] | REACH m30/33[RPTGAP] |
| task_082 | REACH m15/15[EARLYTR,RPTGAP] | REACH m15/15[EARLYTR,RPTGAP] |
| task_083 | REACH m9/10[PROVall,FABTOOL,RPTGAP] | REACH m7/10[DISAMB,RPTGAP] |
| task_084 | REACH m7/12[PROVall,FABTOOL,TIMEFAB,DISAMB,NUMDIFF] | REACH m5/12[PROVall,FABTOOL,DISAMB,NUMDIFF] |
| task_085 | REACH m9/10[PROVall,FABTOOL,TIMEFAB,RPTGAP] | REACH m9/10[PROVall,FABTOOL,TIMEFAB,EARLYTR,RPTGAP] |
| task_086 | REACH m11/18[PROVall,FABTOOL,TIMEFAB,DISAMB,NUMDIFF,RPTGAP] | REACH m11/18[TIMEFAB,DISAMB,NUMDIFF,RPTGAP] |
| task_087 | REACH m19/20[PROVall,FABTOOL,TIMEFAB,EARLYTR,RPTGAP] | REACH m19/20[PROVall,FABTOOL,TIMEFAB,EARLYTR,RPTGAP] |
| task_088 | REACH m15/17[PROVall,FABTOOL,RPTGAP] | REACH m11/17[PROVall,FABTOOL,RPTGAP] |
| task_089 | REACH m12/13[PROVall,FABTOOL,EARLYTR,RPTGAP] | REACH m12/13[PROVall,FABTOOL,TIMEFAB,EARLYTR,RPTGAP] |
| task_090 | REACH m10/16[PROVall,FABTOOL,NUMDIFF,RPTGAP] | REACH m15/16[PROVall,FABTOOL,RPTGAP] |
| task_091 | REACH m24/25[PROVall,FABTOOL,TIMEFAB,EARLYTR,DISAMB,RPTGAP] | — (infra) |
| task_092 | REACH m17/21[TIMEFAB,RPTGAP] | REACH m10/21[PROVall,FABTOOL,TIMEFAB,RPTGAP] |
| task_093 | REACH m4/9[TIMEFAB] | REACH m4/9[PROVall,FABTOOL,TIMEFAB,EARLYTR] |
| task_094 | REACH m8/9[EARLYTR] | REACH m8/9[TIMEFAB] |
| task_095 | REACH m6/9[PROVall,EARLYTR] | — (infra) |
| task_096 | REACH m12/12[GATE,PROVall,FABTOOL,RPTGAP] | — (infra) |
| task_097 | REACH m17/18[PROVall,FABTOOL,EARLYTR,RPTGAP] | REACH m17/18[TIMEFAB,EARLYTR,RPTGAP] |
| task_098 | REACH m2/3[USERarg] | REACH m2/3[USERarg] |
| task_099 | REACH m4/5[USERarg] | REACH m4/5[USERarg] |
| task_100 | REACH m3/4[-] | REACH m3/4[-] |
| task_101 | REACH m2/4[USERarg] | REACH m2/4[USERarg] |
| task_102 | REACH m1/3[TIMEFAB,USERarg] | ARGDIFF m0/3[USERarg] |

pass 태스크(9 sims): 002(×2)·006(×2)·010(t1)·033(×2)·034(×2). 미실행-key 유형(REACH 내용물): call_discoverable 500 · unlock 352 · call_user_tool 118 · give_user_tool 36 · **log_verification 35** · 기타.

---

## §2 레버별 근거 (정독 인용)

### 2.1 PERARG — 최대 발화 레버, 복구는 조건부 (71 sims·41.8%)
- **도구명 날조(61 sims·현행 힌트 사거리 안)** — 엔진 `_provenance_deny`는 모든 호출의 raw args를 검사하고 `agent_tool_name`/`discoverable_tool_name`은 'name' 힌트에 걸린다. 문맥(KB출력)에 없는 이름 = deny.
  - task_012 t0: KB에 없는 `set_travel_notification`을 지어내 give 시도 → env "Unknown discoverable tool" → 이후 **앱 조작 절차를 통째로 날조**해 안내(gold=정직 인정+transfer). deny가 떠도 "정직하게 모른다고 말하기"로 이어질진 불확실 — **발화확실·복구 불확실**(epistemic 잔여 §3.4).
  - task_016 t0: `investigate_referral_status`·`view_transaction_history` 연속 날조 → 포기 → 이관. env-error에도 다른 이름을 또 지어냄 = **error-신호만으론 복구 안 됨의 실측** — deny의 "OBTAIN the real value first" 지시가 KB재검색으로 유도할 개연은 있으나 보장 없음.
  - task_096 t0: `link_checking_savings_accounts`·`verify_account_apy_settings` 등 4개 연속 날조 — 진짜 체인(get_bank_account_transactions_9173 등)은 전혀 미발견.
- **placeholder id** — task_016 t0: 검증 전 `get_referrals_by_user(user_id="user123")` → "No records" → 스토리 전체 탈선(이후 본인 ID로 재조회 안 함). **GATE+PERARG 동시 발화 지점** — deny→verify-first→record의 실 id 습득이 자연 경로라 **연쇄-재궤도 개연 최상위 사례**. task_065 t0: `account_id="rp65a7b3c4-lightblue"` 날조 → env error 4턴 방황 후 **자가복구**(전수조회로 실 id 획득) — 레버 이득 = 턴 절약뿐, 결과 동일.
- **gold-diff 기준 엄밀치**: fired 19 sims / gold값이 문맥에 실재해 regen 즉시 복구개연인 것 4 sims (2.4%). task_041 t0: `transaction_id`를 `business_gold_10142025_623_45` 형식으로 **16회 합성-날조**(실 txn id는 한 번도 fetch 안 함) — deny 시 getter(get_credit_card_transactions_by_user)가 존재하므로 fetch-경로 복구개연, 단 last4 등 나머지 blocker 잔존.

### 2.2 GATE GB1 — 발화 6 sims(3.5%)·단독폐쇄 0
- 발화조건(gold에 log_verification ∧ gated 도구를 검증 전 시도)이 좁은 이유: 실패 대다수는 **gated 도구까지 도달 자체를 못 한다**(REACH 사망이 검증 이전). log_verification 미실행 35 sims 중 gated-시도 동반은 6.
- task_016 t0(위)·task_096 t0: 검증 없이 날조-도구 시도 — GB1 deny → verify-first 강제 = 미실행 gold 1개(log_verification)를 채우고 재궤도 개연. **단 나머지 체인 blocker는 그대로** → 단독폐쇄 0.
- 면제(applies_when not_in) 정합: transfer/incident 태스크(015·032·033·035)에서 위양성 발화 0 — **pass 9 sims에서도 발화 0**(§4 Δspurious).

### 2.3 CALC-EXT — 진성 계산형은 우리 floor에선 거의 등장 안 함 (0~2 sims)
- 수치-diff 20 sims의 정독 실체:
  - task_038 t0 / 041: `card_last_4_digits="1234"/"5678"/"1235"` = **placeholder 날조**(정답 last4는 discoverable **user 도구** get_card_last_4_digits로만 획득 — gold에 give+user-call 체인 존재). 계산 아님·fetch-형. 현행 힌트('digits'·'amount' 미포함) **사거리 밖**.
  - task_062 t0: $3,500을 물어보지 않고 1750/1750 임의 분할 — 계산 오류가 아니라 **미확인 가정**(confirm-형·§3.3).
  - frontier가 보여준 진성 집계형(C52 task_074 fee-refund 합산)은 우리 floor에선 그 단계까지 도달한 sim 자체가 희소.
- ⇒ CALC-EXT는 banking floor에선 **발화 표적이 사실상 없다**(frontier·pass↑ 후에나 잔여로 등장할 것 — C52 모델 기울기와 정합).

### 2.4 DISAMB — 후보 36 sims(21.2%)·기준이 정책-semantic이라 복구 불확실
- task_038 t0: `card_action` gold=cancel_and_reissue vs 실행=keep_active(3번째 dispute) — 둘 다 문맥 실재(|C|≥2·스키마/선행 콜). "fraud로 카드 재발급 중이면 cancel_and_reissue" = 정책 추론 — subcall이 맞출 개연 중간.
- task_065: account_class 선택(gold Green/Evergreen vs Bluest/Platinum/Gold…) — 후보는 전부 문맥 실재하나 정답 기준 = **KB 정책 ⋈**(APY×적격성×페어링·잔액$6,000) → DISAMB subcall로 닫힐 성질 아님(§3.2).

### 2.5 P2/NOTICE/EXCLUSIVITY — 구조적 0
- banking A2(`a2/banking_knowledge.gate.json`): gates=[GB1 auth 1개]·producers={}·**default_specs/eplan/notice/confirm 키 부재**(\_meta not_instantiated 명기) → P2·NOTICE 발화 0은 A2 구성상 확정.
- 정독이 찾은 **잠재 default_spec 후보**: `issue_noticed_date` ← 현재시각(041에서 gold=11/14/2025(오늘) vs 실행=거래일짜) · `time_verified` ← get_current_time(§2.6).

### 2.6 ★신규 — LOGV_TIME_FAB (59 sims·34.7%) & EARLY_TRANSFER (36 sims·21.2%)
- **time_verified 날조**: get_current_time을 안 부르고 `"2023-04-15 14:20:00 EST"`(task_043 t0)·`"2023-10-10 14:30:00 EST"`(task_065 t0) 같은 과거시각을 지어냄. env는 그대로 수용(로그 성공) → **DB 행이 gold와 달라져 단독-치명**. 59/170 (34.7%). 현행 힌트에 'time' 없음 → **A2 `identifying_arg_types` 추가(ABox-only) + producer(get_current_time)로 결정론 폐쇄 가능한 클래스**. 단 이것이 유일 diff인 sim은 **0** — 폐쇄 기여는 상한 0, "미래에 다른 blocker가 닫힌 뒤" 살아나는 지뢰 제거 성격.
- **조기/과잉 이관**: task_082 t0 — 유저가 "human agent" 요구하자 2턴 만에 transfer(gold=15-step dispute 체인 자체 수행). retail `G_EXHAUST(exhaust_before_escalate)` kind가 정확히 이 모양 — banking A2 미인스턴스. 단 **이관을 막아도 체인을 완주한다는 보장은 전무**(막힌 뒤 발견 체인은 여전히 REACH 문제).

### 2.7 정독이 스크립트 판정을 뒤집은 사례 (v1→v2 정정 기록)
1. **task_012**: v1 "REACH·플래그 없음" → 정독: **도구명·절차 날조(epistemic)**. v1이 도구명을 매칭-key로 흡수해 PERARG 스캔에서 누락 → v2 PERARG_ALLCALL/FAB_TOOLNAME 신설(+61 sims 발화 재계상).
2. **task_082 등 EPLAN 플래그**: v1 EPL1/L2 발화 → 정독: 조기이관이 2턴째라 열거 단계 자체 미도달 — E-PLAN 발화조건 불성립(**spurious**). EARLY_TRANSFER로 재귀속.
3. **task_038·041 CALC-derived**: v1 계산형 → 정독: last4 placeholder **날조(fetch-형)** — CALC 오탐.
4. **task_062 t0 CALC-selection**: v1 계산형/T2-폐쇄 → 정독: **미확인 가정(confirm-형)** + 동일-key 초과-transfer가 DB 오염(matcher가 gold-key 중복이라 extra로 못 셈) → T2 폐쇄 1건도 기각, 실질 0.
5. **task_043·065**: time_verified 날조 발견(v1 사거리 밖) → LOGV_TIME_FAB 신설(59 sims).
6. **task_021**: v1 EPL2 → 정독: 원인은 "어느 txn이 오계산인지"의 **KB정책 ⋈ 판별 + 절차선택 오류**(직접 update_transaction_rewards로 '정상 예외'(WeWork 코워킹)를 멋대로 수정·진짜 오류 2건은 미식별). 열거-강제는 무력.
7. **task_005 t1**: v1 ARGDIFF → 정독: **사회공학 수용**(가짜 bypass code로 change_user_email 실행) + 온파일 이메일 누설 — 스택 밖 신규 유형(§3.4).

교차검증: 자체 matcher vs tau2 `action_checks` — 170 fails 중 16건 ±1 불일치(전수 확인: tau2는 raw-dict 완전일치라 잉여 `arguments:{}` 키도 불일치 처리 — 우리가 관대한 쪽·레버 판정 목적엔 우리 쪽이 옳음. 예: task_015 give에 빈 arguments 동봉).

---

## §3 "안 닫히는 잔여"의 정체 (C52 대조·신규 발견)

### 3.1 C52/C54 재현 여부 — **재현 + 모델-기울기 이동**
| C52 주장(frontier 17모델) | 우리 floor(32B·pass 4.7%) | 판정 |
|---|---|---|
| REACH 지배(24-48%) · unlock 체인 80% 태스크 | REACH-지배 버킷 76.5% · 미실행 key의 85%가 unlock/call 체인 | **재현·증폭**(§3.2f(5) "최약 모델은 미실행 68% 지배" 기울기의 연장선에 정확히 위치) |
| KB검색은 함(119/125) | **164/170 fails가 KB_search 호출** | 재현 — 검색은 하나 **조립**을 못 함 |
| horizon 8단계 | gold median 8 (max 33) | 재현 |
| 완주-후-불일치 45%(coverage≥0.8) | **16%** (coverage<0.5가 65%·med 0.33) | 불일치 아님 — **실패 구성의 모델-사다리 재배치**(C52 §3.2f(5) 예측 그대로: 약한 모델일수록 미실행이 흡수) |
| 조기 transfer(발견/조기중단형 내) | EARLY_TRANSFER 36 sims로 정량화 | 재현·정밀화 |

### 3.2 잔여 유형학 (서술형 이름 — [[48]] 준수·새 코드 없음)
1. **발견-체인 미조립** (지배·~130 sims): KB는 부르나 "이 절차의 내부 도구를 KB에서 찾아 unlock→call로 잇는" 결합을 못 함. 하위 모드: (a) 도구명 환각(61) (b) 자기-능력 오인("이체 능력이 없다"며 셀프서비스 안내로 도피 — task_043) (c) **대기-교착**: DB가 저절로 바뀌길 기다리며 get_current_time/재조회 루프(task_043의 $75 잔액 무한 대기 — env 시계도 안 흐름).
2. **유저측 오유도** (~28 sims·001/003/007/015/024/025/098/099/101/102…): gold write가 user-요청자(추천 따라 apply/submit). 실패 근원 = **추천 답변의 semantic argmax/⋈**(예: task_098 — "최대 합산 보너스 계좌"에 KB 1-hit로 **신용카드**(EcoCard)를 추천·gold=Blue Account·검증/이력조회도 생략). agent-write 게이트로는 못 닿고, 답변-내용 레버는 스택에 없음.
3. **미확인-가정 write + all-or-nothing** (task_062·010): 062 t0 = 금액 분할을 묻지 않고 실행 → 초과 transaction 행이 DB에 영구 잔존 — **되돌려도 안 닫히는 벤치 특성**(수정-불가능성). 010 t0 = gold 전 액션 일치·**잉여 give 1건**만으로 db_match false = 신규 마이크로 유형 "무해해 보이는 초과 액션의 DB 오염"(t1은 pass — 재현성 있는 확률적 근접-미스).
4. **순서-의존 절차 파괴**(task_065): 유저가 요구한 순서(폐쇄 먼저)를 정책 근거 없이 수용 → 적격성 영구 상실 → 동일-write 8연속 맹목 재시도. preconditions-kind 게이트의 표적이나 그 사실이 **KB 문서 semantic**이라 gate_spec의 결정가능 술어로 컴파일 어려움.
5. **epistemic/사회공학**(012·014 계열, 005): KB-부재를 인정 못 하고 내용 날조 / 가짜 bypass code 수용·PII 누설. [[43]]의 실측 사례 — 스택 밖.

### 3.3 함의 — 예측 확정
DOMAIN_TRANSFER §1.4 "banking G2 REACH=BC2 부하 → **결정론 controller(발견/조립)**·게이트 아님"이 per-case에서 확정됐다. E-PLAN이 banking에 닿으려면 retail형 열거자(L1/L2)가 아니라 **discovery-read 강제 + plan↔execute 분리**([[14]]의 원안 그대로)여야 하고, 이는 지배 질량(~76%)을 직접 겨눈다.

---

## §4 gate-arm 재실측 시 기대치 (정직 상한)

**pass^1 델타(주지표) 기대 = +0 ~ +2 sims (4.7% → 4.7~5.7%)**
- 근거: 정적 폐쇄 상한 T1=0/170·T2≤1(정독 기각→0). 상승 여지는 오직 **deny→재궤도 연쇄**(정적 분석 밖): 최유력 task_016(placeholder deny→verify→본인 referral 조회 성공 시 gold 2-act 충족 가능), 차순위 038(last4 deny→user-tool 체인 유도)·마이너 확률적 근접-미스(010 유형). 낙관 시나리오도 nt2 기준 +1~2%p 절대를 넘기 어렵다.
- **레버 스택으로 banking pass는 못 산다** — 이 arm의 가치는 pass가 아니라 아래 compliance 계측이다.

**compliance/부작용 지표 기대 (이건 크고 측정가치 있음)**
- 날조-regen 발화: fail의 41.8%에서 최소 1회 (도구명 61·placeholder id·gold-diff 19) → **날조율 감소를 직접 계측 가능**(47-tau2의 compliance-drop 모트 서사에 banking 열 추가).
- GB1 위반(검증 전 고객데이터 접근) 교정: 6 sims + 발화-무관 위반 census는 t2_compliance로 별도.
- **Δspurious ≤ 0 계측 기반 확보**: floor의 pass 9 sims 전수에서 GB1·PERARG·TIMEFAB **발화 0** (측정) → 게이트 귀속 fail-전환 위험의 사전 추정 0. 단 regen이 대화 흐름을 바꾸는 2차 효과는 실런만 안다(등대 제1원리: 게이트 자신의 over-action 역효과 계측 필수).
- tme/장문화: banking은 이미 sim당 ~4분(32k) — deny-regen K=4가 턴을 늘림 → **비용 +10~20% 각오**.
- 참고: `bankxfer_gate_bank_t1_partial_oldstack`(구 스택·13 sims·pass 1)은 n이 작아 무정보 — 비교 사용 금지.

**A2-only 확장(엔진 불변·[[05]] 합치) 우선순위 제안** — 재실측과 별개 트랙:
1. `identifying_arg_types: ["time_verified"]` + producer get_current_time — 59-sim 지뢰 클래스 결정론 제거(폐쇄 기여 즉시 0이나 REACH가 닫힌 뒤 병목으로 승격될 1순위).
2. exhaust_before_escalate → transfer_to_human_agents/request_human_agent_transfer (36 sims 표적·정책 §"transfer 조건" 실재 여부 재확인 후).
3. confirm-kind는 banking 정책에 전역 확인규칙이 없어(\_meta) **정책-근거 없는 인스턴스화 = over-action 위험** — 보류.

---
*산출물: 본 doc · `scripts/distill/tau2/banking_lever_fit_census.py`(재현: `py -3 banking_lever_fit_census.py`) · per-case `reports/facet_rft_2026/sim_results/banking_lever_fit_percase.json`. 정독 12전문+4부분: 012·043·082·016·065·062·038·007·005t1·010·021·098 (전문) / 041·096·008·015 (per-diff 상세). 커밋하지 않음(사용자 지시).*
