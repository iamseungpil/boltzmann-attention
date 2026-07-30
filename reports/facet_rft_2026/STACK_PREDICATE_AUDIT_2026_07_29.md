# 스택 술어 감사 — 닫힘 판정 배터리(P1-P3/R/Q3) + 전 레버 분류 + 발화 실측 (2026-07-29)

> 발단=사용자 지시: "열린 술어 부분을 정밀하게 측정하라. 닫힌/열린 판정 기준을 만들 수 없나?"
> = `AXIS_DECISION_DETERMINISM_LEARN_2026_07_29` rev1 §4-4(스택 감사)의 실행 + §2 기준의 조작화.
> 방법: go_stack ON 전 레버의 트리거 코드 정독(t2_gate_patch·t2_prekb_patch·t2_scaffold_get) +
> day6~9c 로그 8개(bank_day{6,7,8,9c}{A,B} `sim_results/*.log.gz`) 마커 발화 실측.
> 상태: 설계+측정 [M]·리뷰 대기.

## §1. 판정 배터리 — 닫힘은 "구현의 기계성"이 아니라 "의도 술어의 정의가능성"이다

**핵심 정리**: 모든 레버는 코드이므로 구현은 전부 기계적이다. 여는 것은 **의도 술어**
("재요청 중이다"·"무관하다"·"이행 안 했다")가 의미-판단이라 구현이 그 **문자열-프록시**가
되는 경우다 — 프록시의 충실도는 시나리오-의존이라 변이가 곧 오탐/미탐이 된다(실증: C191
"new personal checking" 부분일치 실패·UNKNOWN_REPEAT 016 날조-이름 부활 [S]). 따라서 기준은
기계적으로 적용 가능한 4검사로 조작화된다:

**P1. 정의 검사 (술어)** — 의도 술어를 아래 어휘만으로, 판단어 없이 진술할 수 있는가?
- 구조-이벤트 어휘: tool_call 레코드(name·args·id·requestor) / 도구 출력 원문 / 발화 원문
  (문자열로서) / A2·스키마 상수 / KB 문서 원문 / **엔진 자기-삽입 토큰·스텁·마커**
- 허용 연산: 정규화 후 문자열 동등·포함 / 집합 연산 / 계수 / 산술 / 스키마 검증
- 진술에 판단어(무관·적절·이행·의도·재요청·관련·충분)가 잔존 → **열림**.

**P2. 프록시 검사 (구현)** — 자유 NL에 대한 substring/regex 매칭인가?
- (a) 매칭 대상 = 엔진 자기-삽입 토큰/스텁(`[DUPLICATE-READ]` 계수·`###TRANSFER###`·
  `Executed:`·`unverified` 자기-마커) → **닫힘**(자기-생성 구조 이벤트).
- (b) 매칭 대상 = 모델/유저 자유 NL(재요청 "신호 문구"·say-don't-do 선언·의도 표현) →
  **프록시=열림**. 예외 하나: 비교 한쪽이 엔진 보유 **정본 상수**인 정규화 비교(G1 notice)는 닫힘.

**P3. 변이 테스트 (경계 사례의 최종심·측정 가능)** — 의미-동일·표현-상이 변이 M개에서
verdict flip율 = 0인가? 프로토콜: 트리거 시점 문맥 고정 + 트리거 원인 텍스트만 패러프레이즈
치환([[18]] 정보-맞춘 격리 프로브와 동형·무료·오프라인). **flip율>0 → 열림 확정.**
P1/P2가 갈리는 경계 사례를 이 프로브가 기계 판정한다 — "기준을 만들 수 없나"의 답:
**P1/P2=코드 정독으로, P3=flip-프로브로 결정 가능하다.**

**R. 처방 검사** — 위반 확정 시 강제 행동이 닫힌 메뉴에 속하고 인자까지 유일 결정되는가?
닫힌 메뉴 = {①형식-층 전체 재생성/재송신 ②결정론 치환(치환값 유일 계산) ③deny+위반사실
명시(cap 유계) ④표면화(행동 요구 0) ⑤**read-계열 호출 강제**(§1.5·대상 도구 유일 결정 시)}.
"고쳐라/전환하라/안내하라/(조건부)재호출하라" = 열림.

**Q3.** AXIS rev1 §2(정당화의 일반성·정의-기준). **강제 허용 = P∧R∧Q3 전부 통과.**

### §1b. 개방도의 정량 정식화 — h(π)와 비대칭 판정 규칙 (2026-07-29 사용자 질의)

- **정의**: 상황 s의 **변이 궤도** O(s)=의미-보존 변환(패러프레이즈·서식·순서·user-sim
  재표현)의 동치류. 술어 π의 **개방도** h(π)=E_s[H(π(S′)|S′∈O(s))] — verdict의 궤도-조건부
  엔트로피. **닫힘 ⟺ h(π)=0**(모든 궤도에서 상수). P3 flip-테스트=이것의 추정기.
- **추정**: 시드 N×변형 M·flip율 p̂(궤도 다수결 기준)·Wilson 구간. 변형 생성=LLM·flip
  계산=결정론([[10]] 분담: 생성기=LLM·판정=엔진·변형의 의미-보존성은 표본 감사).
- **★판정 규칙은 비대칭이다**:
  - **기각(열림 확정)**: 재현 flip≥1(Wilson 하한>0) → 열림 [M]. **건전(sound)** — 측정만으로 확정.
  - **수용(닫힘 인정)**: Ĥ≈0은 닫힘을 증명 못 한다(표본 밖 변이·**안정적으로-틀린 프록시**
    가능) → 닫힘 인정은 **P1 정의-증명으로만**. 측정은 불완전(incomplete).
  - ⇒ **측정은 닫힘을 기각할 수만 있고, 수립은 정의로만 된다**(결정가능성 증명과 동형).
- **h_min의 실제 자리 = 처방-등급별 강제 예산**(단일 문턱 아님): 기대 피해=발화율×flip
  상한×피해를 모트 규율(Δspurious≤0)에 넣으면 — **강제(deny/regen)는 ε≈0=사실상 P1 증명
  요구 / read-강제(R⑤)·표면화·넛지는 완화 임계 허용.**
- **궤도 출처 2종**: 합성 궤도(통제 패러프레이즈·[[18]] 정보-맞춘 격리·무료)=정본 /
  자연 궤도(user-sim 시나리오 재생성 0.77)=실질 변경 혼입이라 보조 [D]만.
- **6층 리스트의 지위(사용자 질의 ①)**: 6층=배터리의 현-스택 열거(P1 증명이 존재하는
  집합)이지 정의가 아니다. (a) 코어 밖 닫힌 술어 신설 가능(온톨로지 확장·gate kinds 지위·
  보편 미증명) (b) 층 안이라도 구현이 엶(023·PREKB·VIEW 실증=닫힘은 술어+처방+**구현
  충실도**의 속성) (c) 닫힘=강제의 필요조건이지 충분조건 아님(C12). 현 스택에 한해
  "A층 밖=배터리 탈락"은 감사 결과로서 참 [M].

## §2. 전 레버 분류 (go_stack ON 기준·코드 정독)

### A. 코어 — P닫힘∧R닫힘∧Q3 (유지·동결)
| 층 | 레버 | 비고 |
|---|---|---|
| 형식 | OVERFLOW/ENVELOPE/TRUNC_GUARD·DYN_MT·ARG_SCHEMA·PARAM_CAP·STALE_STRIP·PAIRCHECK/PAIRFIX | |
| 형식(캡·뷰) | VIEW_COMPACT/ANNOTATE/MSG_CAP·SUB_TOOLCAP | ⚠술어 닫힘이나 **정보-손실 부작용 실측**(SUB_TOOLCAP 절단=023 공범 [S]) → Δ손실 계측 의무·캡 값 재검(AXIS §4-3-①) |
| 전사 | SG_BYREF | |
| write-접지 | PROV/GATE_REGEN(UNIFIED)·WRITE_EVIDENCE·REF_VERIFY·WRITE_ARG_GROUND·SG_GROUND·PROD_BIND·FAB_STRIP·UNLOCK_NAME/PROV·UNKNOWN_NAME_BL·TOOLLIST | 이름=집합 membership·증거=원장 공존 |
| 게이트 | gate_spec·PRECLOSE_CAP·BRANCH_REGROUND·TERM_GRANT(+USERDEMAND=A4)·require_tool_before·SG_REQREADS·TOOLGATE·DISPATCH_ROLE(+NOTE)·SG_TRUTH·G1 notice | A4=Δspurious 계측 유지. **PREKB=조건부**: 게이트 술어(검색 실행 여부)는 닫힘이나 피드백 문구가 검색어 축자 공급=스푼피드 실증(032 [S]) → 문구를 사실 통지로 축소 후 잔류 |
| read-강제(R⑤) | FOLLOWUP_REQUIRED/FORCE/READLOOP·COVERAGE_FU(결함 §4-3-④ 수정 전제) | 사임∧미호출 판정도구=구조·대상 유일·빈손 0% [S] |
| controller | SG_ISOLATE/ISOFB/FN_ISOLATE·SCAFFOLD_GET·RESOLVE-compute/reffilter(fexec) | Q2 부하 처방·실행=결정론(formalize만 LLM) |
| 표면화 | READ_DEDUP·SG_DEDUP(+DUP_REPRESENT)·COVERAGE 표·ABSTAIN_FIELDS·EPLAN ledger/walk 표면화 | |
| 계측(비개입) | SG_TRACE·PAIRDUMP·FB_VIEW·FAILED_PERSIST·LEVER_HEALTH·LLM_DIAG | 행동 변경 0 |

### B. 협곡 — P닫힘∧**R열림** (넛지층·일괄 ablation 대상)
| 레버 | 술어(닫힘 근거) | 처방(열림 근거) | 발화(8런) |
|---|---|---|---|
| CLAIMPROV | claim 원장대조 | 강제 regen=교정 방향 미결정 → **3b 강등 확정**(AXIS rev1) | **482** |
| WRITEPROV | write-주장 대조 | 동일 — **3b 강등 후보**(빈손 91% [S]) | 32 |
| RESOLVE action-required | ⚠술어도 프록시(say-don't-do 선언) | required 강제 | **216** |
| RESOLVE user-action instruct | user-측 도구=A2 구조 | 지시 문구·시점(단 D-4 [S]: 설득이 유일 경로 → ablation서 판정) | **168** |
| GIVE_EXEC_NUDGE | give 성사∧user 호출 0(**error_flag 버그 수정 전제** [S]) | 안내 넛지 | 20 |
| SEARCH_EXHAUST | 자기-스텁 계수(P2a) | 전략 전환 권고 | 4 |
| UNVERIFIED_FU | 자기-마커 'unverified'(P2a) | 조건부 재호출 권고 | 1 |
| ARG_PRODUCERS | 결핍 에러(구조) | give-흐름 안내 | 35 |
| EPLAN drive·COV_MIDDRIVE | coverage 갭 | 견인 지시(비강제) | 41 |

### C. 금지선 — **P열림(프록시)** ∨ Q3 실격 (재설계·OFF 후보·learn 이관)
| 레버 | 실격 사유 | 발화(8런) |
|---|---|---|
| FORCE_ACTION | say-don't-do=NL 프록시(P2b) ∧ required **강제** = 이중 위반 | **211** |
| HAVE_VALUE(+FORCE) | "재요청"=NL 신호 프록시(P2b)·FORCE=강제 | ~0 |
| VALUE_ACQUIRE | "재요청" 프록시(P2b)·넛지 | 54 |
| UNKNOWN_REPEAT_GUARD | Q3 실격·악화 실증(016 [S]) | 20 |
| GIVE_RELEVANCE(N1) | "무관"=열린 술어·rev1 발화 0=무효 → learn 이관 확정 | 0 |
| 미분류 | GUIDED(pre-gate 순서) — 코드 확인 1건 잔여 | — |

## §3. 발화 실측 (day6~9c·A+B 8런·개입-행위 라인만: deny/regen/nudge/repair/strip/fired/stub/force/instruct)

**출처**: `sim_results/bank_day{6,7,8,9c}{A,B}_2026072{8,9}.log.gz` stderr 마커([[08]] 레버
발화는 stderr만·handoff §8). 운영 로그 제외 필터 적용. 총 개입-행위 ≈ **2,776건**.

| | day6 | day7 | day8 | day9c |
|---|---|---|---|---|
| 코어(A) | 455 | 282 | 598 | 225 |
| 협곡+금지선(B+C·RESOLVE 제외) | 190 | 192 | 215 | 220 |
| RESOLVE 실행-견인(req+instruct) | 98 | 94 | 100 | 92 |
| **열린-측 비중** | **39%** | **50%** | **34%**¹ | **58%** |

¹ day8 코어 급증(598)은 DISPATCH_ROLE·READ_DEDUP 대량 발화(모델 flail 반응) — 코어 발화는
모델 상태의 함수라 일자 간 변동이 크고, **열린-측 발화는 ~300/일로 상수**(레버 구성의 함수).

- **열린-측(B+C) 합계 ≈ 1,201건 = 전체 개입의 43%.** day9c 기준 58%·sim당 ~4.9건.
- **최대 단일 채널 = CLAIMPROV 482(열린-측의 40%)** — 3b 강등(AXIS rev1)만으로 열린-측
  개입의 40%·전체의 17%가 제거된다.
- 다음 = RESOLVE 실행-견인 384 + FORCE_ACTION 211 = 강제·견인 계열 595(열린-측의 50%).
  **FORCE_ACTION은 이중 위반(프록시 술어∧강제)으로 금지선 위** — ablation 최상위 표적.
- 빈손율 [S](handoff §2-4·인용): 비강제 regen 60~91%(CLAIMPROV 68·WRITEPROV 91)·강제 0%·
  day6에도 64%로 상수.
- ⚠**한계(정직)**: 발화량=개입 노출량이지 피해량이 아니다. 피해 인과는 역인과 원리
  (AXIS §1b-1)로 관측 불가 → 층-단위 ablation([S])만이 판정한다. 본 표는 ablation의
  **사전 규모 산정**이다.

## §4. 요약 답 (사용자 질문 2건)

1. **"열린 술어 부분" 정량**: 현 스택 레버 수 기준 — 코어(P∧R∧Q3) ~35 / 협곡(P닫힘·R열림)
   9 / 금지선(P열림∨Q3실격) 5 / 계측 6 / 미분류 1(GUIDED). **개입 발화량 기준 — 열린-측 43%
   (day9c 58%)·그중 CLAIMPROV 40%·강제/견인 계열 50%.** 열린-측 발화는 ~300/일 상수로 스택
   구성의 함수이며, day7~9 신규 처방이 열린-측 비중을 끌어올렸다(39→58%).
2. **판정 기준**: 만들 수 있다 — §1 배터리. P1(정의)·P2(프록시)·R(메뉴)·Q3는 코드 정독으로
   기계 판정, 경계 사례는 P3 flip-프로브(무료·오프라인·결정적)가 최종심. **신규 레버는 이
   배터리 통과 없이 도입 금지**(AXIS §4-5 게이트의 구체 절차).

## §5. 후속 (AXIS §4 순서에 편입)

- ablation arm 정의 구체화: 코어-only(A층) vs 코어+{B+C}(현행) — C층은 arm 정의상 B와 묶어
  1회 판정(레버별 해상도 없음·§3 한계).
- GUIDED 코드 확인 1건 → 분류 확정.
- P3 flip-프로브 구현(무료): C층 프록시 술어 3종(FORCE_ACTION·HAVE_VALUE·VALUE_ACQUIRE)의
  flip율 실측 → 금지선 판정을 [M]로 승격.
- PREKB 문구 축소·VIEW/캡 Δ손실 계측(§2-A 비고)·GIVE_EXEC error_flag 필터(AXIS §4-3-②).

## §6. [[05]] 3질문 ([[17]])

1. 순증 없음 — 분류·측정·판정 절차뿐(레버 0개 추가). 2. 고정 강화 — 코어 축소·동결의 집행
   도구. 3. 수행 대체 없음 — 프로브는 판정용 오프라인 도구.
