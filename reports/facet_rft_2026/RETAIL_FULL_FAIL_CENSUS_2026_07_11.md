# RETAIL 전수 실패 census — COMP 456 sims 전 실패의 per-case 분류·복구 라우팅 (2026-07-11)

> 사용자 지시: "6표적 외에 더 있는지 전수 확인·모든 실패를 복구하고 간다."
> 입력 = `sim_results/comp_retail_t4.results.json.gz`(C62 COMP arm·456/456·오늘 새벽 완료) + `t5c_v25e.results.json.gz`(silent 6표적 nt=4).
> 도구 = `ecomp_fail_census.py --dump` → `t5c_taskmap.py`(per-task 지도) → `t5c_taskdiff.py`(gold-diff 정독). [[08]] 전수·per-case.

---

## 0. 헤드라인
- **실패는 6표적이 아니라 task 78개 / 167 sims** (COMP 456 중 36.6%). 6표적(0,17,40,47,61,95)은 T5-C 회귀-표적 부분집합이었다.
- 구조: **SYSTEMIC(0/4) 15** · MOSTLY(1/4) 11 · FLAKY(2–3/4) 52.
- 기지 13표적(0,2,17,28,40,46,47,61,69,92,95,101,103) 중 COMP서 실패 지속 = 9뿐. **신규 SYSTEMIC 13개**: 20,34,36,37,41,57,71,76,79,99,100,102,111.
- **신규 발견(정정 완료)**: 게이트-deny 발화 31 sims 중 31/31이 transfer 동반·19 sims reward=0 — 그러나 **전문 정독 3/3서 게이트-부작용 반증**: deny↔transfer 상관 = **impasse 표지**·게이트는 날조-escape write를 옳게 차단 (§4).
- v25e(silent 스택) 판정: **t0 4/4·t61 4/4(P2 GO)·t47 3/4** / 잔존 t17(값충실도)·t40(NL)·t95(discovery+NL).

## 1. per-task 지도 (실패 78 task 전량)
> 산출 재현: `t5c_taskmap.py --dump comp_fail.jsonl --results comp_retail_t4.gz --ref prov_e2e_retail_t4.gz`. ref=prov arm.

**SYSTEMIC 15**: t20(ITEMS×4) t34(OVER×4) t36(OVER×4) t37(ITEMS×4) t41(MISSED×4) t57(OVER×4) t71(REF×4) t76(REF/MISS 혼합) t79(ITEMS×4 동일오답) t95(ZERO_ATT×4) t99(OVER×4) t100(ITEMS/MISS) t102(REF/MISS) t103(REF×3+MISS) t111(MISSED×3+ZERO)
**MOSTLY 11**: t3 t8 t27 t35 t39 t72 t81 t86 t93 t98 t109
**FLAKY 52**: 16,19,22,29,33,38,40,46,53,54,58,60,61,64,66,69,82,94,101,105,110,112(2/4) · 4,10,12,15,17,21,24,31,32,42,47,49,52,56,59,62,63,73,74,77,83,84,87,89,91,96,97,104,107,108(3/4)
버킷 합계(167 sims): WRONG_ITEMS 27 · WRONG_REF_ORDER 27 · MISSED_WRITE 26 · OVER_ACTION 26 · ZERO_WRITE_NEV 21 · NL_ONLY 13 · WRONG_ADDRESS 9 · OTHER_ARG 8 · WRONG_PAYMENT 5 · ZERO_WRITE_ATT 5. disamb-도달(문맥에 gold·오답 공존) 63.

## 2. 원인 클래스 (per-case 정독 근거·SYS+MOST 26 전수 + v25e 3)

### A. coverage/discovery — 멀티엔티티 중 둘째를 못 찾거나 안 함 (≈8 task)
t41(둘째 주문 주소 4/4 누락) · t76(둘째 cancel 누락) · **t81**(2주문 cancel 중 1개만·3회) · **t95**(둘째 laptop이 *다른 주문*임을 미발견 → 같은 주문에 재시도 → env 거부 → 포기) · t100(부분) · t102(부분) · t103 tr0 · t111(주소 write 누락 3/4).
**처방 = E-PLAN**(discovery-read 강제 + coverage-walk·`E_PLAN_LIVE_WIRING_DESIGN`). silent 불가(없는 write=[[05]] Q3).

### B. 대화-조건/정책-불가 수행 over-action — C25/C50 클래스 (≈5 task)
t34(조건체인 결과=주소변경인데 cancel 수행 4/4) · t36(체인 결과=다운그레이드 modify인데 cancel 4/4) · t57(체인 끝=no-op인데 cancel·"gift card로 환불했다" 날조 발화 포함) · **t99**(사용자 "스케이트보드 취소는 *내가 직접* 웹서 하겠다" 명시 → 에이전트가 cancel 수행 = 철회-요청 수행 그 자체) · t27(반품+교환 동시요청·"하나만 되면 교환 선호" → 반품 먼저 실행 → status 변경으로 교환 불가).
**처방 분화**: t27 = **L3 feasibility**(동일주문 반품∧교환 배타 = 도구스키마-결정가능 → 사전 DISAMBIGUATE/ask). t34/36/57/99 = **대화-precondition controller/ASK**(C50: DB-게이트 불가·대화 semantic·ENDGAME R1) — 잔여 semantic 절반은 정직하게 경계.

### C. compound-criterion → formalize→결정론 실행 (F2b·≈6 task + NL 축)
t20("최고가 변형·신발은 같은 사이즈" argmax 4/4 부분오답) · t36/t37("총액≤크레딧 되게 다운그레이드" 예산조합·t37은 item 집합 과포함) · t71("최근 주문"=argmax 날짜·C56 체계핵) · t79("다른 1L 병과 같은 색" attribute 매칭·4/4 동일오답) · t35("13인치·i5>i7·silver>black 선호순위").
**처방 = calc/formalize 확장**(A2 calc_specs에 argmax/filter/attr-lookup 일반연산 — 엔진은 도메인일반·[[05]]). C58 T6h FORCED=정책-도출 규칙과 동선.

### D. GET-chain / cross-order lookup — "값은 안 밝히지만 어딘가에 있다" (≈5 task)
t86(DC 주소=다른 주문에·Dallas 주소 오복사 2/4+포기 1/4) · t102(NY 주소=다른 주문에·트라이얼2는 Seattle 주소 오복사) · t109(신주소=luggage 주문에·구주소 오복사 3/4) · t39(주소=recent 주문에·**address1 빈 문자열** write 3/4) · t71/t72(default 주소=프로필에·+주문 ⋈ 오염).
**처방** = 결정론 filter-lookup(도시명 등 사용자 단서 → 주문 주소 전수 필터 = C59 내용-매칭 열거 동형) + DISAMB(|C|=2 구주소/신주소). C48 GET-위계의 실전 모집단.

### E. DISAMB/변형 잔여 — COMP엔 DISAMB 없음 → COMP+D-v2 표적 (≈10+ task)
t8(램프 변형) t79 t93(두 laptop 주문 ⋈·t95 자매) t98(payment 타인카드+테마 변형) t100 t103(bookshelf+jigsaw 주문 내용-매칭) + FLAKY 다수(t29,42,49,56,58,60,64,107,110…). census disamb-도달 63 sims.

### F. 값 충실도(copy-fidelity) — v25e 신규 확정 (≈2 task)
**t17**: "Suite 641만 바꿔달라" → address1을 "123 Elm St"로 **축약 복사**(gold "123 Elm Street"·4/4 동일). t39(빈 문자열)도 동형.
**처방 = GROUND 확장**: 문맥-실재 원문과 edit-distance 미세 diff(축약/공백)면 **원문 그대로 치환**(|C|=1 제자리·silent P-A 동형·DB 안 읽음). 화이트리스트에 address 추가 검토(V0 재실행).

### G. NL_ONLY — db 통과·NL 채점 실패 (13 sims)
t3(옵션 개수 count) t19 t24 t40(gift-card 적용 가능답) t46 t47 t95-tr2(**총액**) t104 t105 t108.
**처방 = calc의 NL-지원 확장**(count/총액 결정론 계산 제시). C62 census: NL_ONLY 16/19 calc-사정거리. (judge 의존 잔여는 C19 구역=정직 보고.)

### H. 게이트-deny↔transfer 상관 = impasse 표지 (§4·인과 반증됨) — 별도 레버 불필요·상류 클래스(A/D/E)가 진짜 표적.

### I. 잔여 노이즈/경계
t76 tr1 reason enum(C28 하네스 노이즈) · FLAKY 52 task 질량의 상당수=C60 paraphrase-brittleness(개별 수리 아니라 robust-레버로) · B클래스 semantic 절반 = P3 경계.

## 3. 복구 라우팅 (레버 → 커버 sims 추정·구현 순서)
> **★설계 구체화 완료(2026-07-11)**: 행 2~7의 설계 명세 = **`CENSUS_LEVERS_DESIGN_2026_07_11`**(§1 GROUND-VERBATIM·§2 CALC-EXT+relay-gap 분리·§3 EXCLUSIVITY/NOTICE·§4 DISAMB-ADDR·§5 B-잔여·§6 단계B 편입) + **`E_PLAN_LIVE_WIRING_DESIGN`**(v1.2·A클래스+D-미조회분).

| # | 레버 | 클래스 | 표적 sims(대략) | 상태 | 비용 |
|---|---|---|---|---|---|
| 1 | ~~게이트 deny-피드백 수정/cooldown~~ → **철회**(§4·3/3 정독 반증: 게이트는 옳게 차단·근인은 상류) | H | — (deny 마커는 진단용으로만) | 철회 | — |
| 2 | **COMP+D-v2 = T5-C silent 스택**(P-A/P-B/P-C/P2) | E·F(일부) | disamb-도달 63 중 상당분 + t61형 | 구현완료·단계 B 대기 | 사이클 nt=1 |
| 3 | **GROUND 확장(원문-치환·address)** | F | t17·t39형(≈7) | V0 화이트리스트 재검증 | 무료 |
| 4 | **calc/formalize 확장**(argmax/most_recent/diff_sum) — ⚠️**후속 실측 정정**: t3 4-trial 프로브서 기존 calc **4/4 정확 발화**·실패는 **relay-gap**(전달 누락 2/4)+write-loss(1/4) ⇒ G의 절반은 calc로 못 닫음·"≈25–30" 과대추정 | C·G-compute | C≈15 + G-compute≈6 (relay분 제외) | `CENSUS_LEVERS_DESIGN §2` | 무료 구현 |
| 5 | **E-PLAN**(discovery+coverage-walk) | A | MISSED+ZERO 52 중 멀티엔티티분(≈20) | 설계 v1.1 완료·구현 착수 | 무료 구현 |
| 6 | **L3 feasibility**(반품∧교환 배타 등) | B(t27형) | ≈5 | ENDGAME L3 설계 있음 | 무료 구현 |
| 7 | 대화-precondition controller/ASK | B(잔여) | ≈12 | ENDGAME R1 [D] | 설계 필요 |
| — | 경계/노이즈 정직 보고 | I | reason-enum·judge·semantic 잔여 | P3 | — |

순서 원칙: [[13]] 결정론-먼저·무료-먼저. 1·3·4·6은 기존 엔진(A2/게이트/calc/GROUND) 위 증분 = T5-C 사이클에 합류 가능. 5는 별도 arm(합산 금지·ENDGAME 규칙). **모든 레버 반대편 계측 필수**(Δspurious≤0·Δtme≤0·§1.3).

## 4. 게이트-deny↔transfer 상관 — 집계-오도의 교정 기록 (C63·[M]·[[08]] 실증)
- **집계**: COMP 456 중 게이트-deny 노트 발화 **31 sims → 31/31(100%) transfer 동반**·reward<1 = 19 sims. 집계만 보면 "게이트가 포기 유발"(Δspurious 채널)로 읽힌다. **초안도 그렇게 썼다.**
- **전문 정독 3/3이 인과를 반증**:
  - **t93.0**: 사용자가 *잘못 짚은 주문*(#W3826449=pending·안엔 earbuds뿐) → exchange 불가 → 에이전트가 **"수동으로 status를 delivered로 변경" 날조-escape 제안·시도** → 게이트가 옳게 차단 → transfer. 근인=⋈ 오선택(내용 검증 안 함).
  - **t8.0**: 결제 impasse(사용자가 credit card 고집·계정엔 PayPal뿐·gold는 paypal) → **사용자가 직접 transfer 요청**. deny는 마지막 턴 부수.
  - **t95.0**: 둘째 laptop이 딴 주문임을 미발견 → 같은 주문에 item_id **중복**(`["3478699712","3478699712"]`) 시도 → **env가 3회 거부**(게이트 아님) → "manual adjustment" 날조-escape → 게이트 차단 → transfer. 근인=discovery(A클래스).
- **결론**: deny↔transfer 100% 상관 = **impasse 표지**(상류 실패가 궁지를 만들고, 궁지에서 에이전트가 날조-escape write를 시도하며, 게이트가 그걸 차단하고, 에이전트가 포기). 게이트는 compliance 일을 정확히 함(위반0 유지·12/31은 pass 유지·over-block 아님). **transfer-cooldown/문구 처방 = 불필요·철회.** 진짜 레버 = 상류 A(discovery)/D(lookup)/E(DISAMB).
- **부산물 서명**: "impasse-시 발명형 escape-write"(수동조정·status조작 등 없는 능력 발명) = C36 발명형 날조의 행동판. 게이트-deny 노트가 이 지점의 **무료 검출 마커**로 쓸 수 있다(진단 가치).
- **방법 교훈**: 집계(31/31)→인과 직행은 이번에도 오도였다. [[08]] 가드 훅이 커밋 직전에 잡음.

## 5. v25e 판정 영속 (T5-C 단계 A 결과)
| task | v25e | 판정 |
|---|---|---|
| t0 | 4/4 | ✅ 닫힘 |
| t61 | 4/4 | ✅ **P2(원리-디폴트) GO** — gift_card→원결제 교정 실증 |
| t47 | 3/4 | ✅ write-loss 복구(fix#1)·잔여 1 trial은 NL |
| t17 | 0/4 | ❌ ~~값충실도~~ → **재진단(V0+코드 확정)**: read 0회 **미조회 자유텍스트 날조** + prov rescue per-call `break` 구멍(`#`-접두 거짓양성 fab이 선점) — `CENSUS_LEVERS_DESIGN §1` PROV-RESCUE-PERARG가 표적 |
| t40 | 0/4 | ❌ 2 trial db=True·NL축(gift-card 적용답) → §2G calc-NL |
| t95 | 0/4 | ❌ discovery(E-PLAN) + tr2는 db=True·NL 총액(calc) — **이중 결손 확정** |

## 6. 다음 액션
1. T5-C 단계 B 표적 재구성: 기존 13 + 신규 SYSTEMIC 13 = **26 task × nt=1 사이클**(§0b 프로토콜) — 레버 2(COMP+D-v2)·3(GROUND 확장)·4(calc 확장)·6(L3 feasibility) 편입 후.
2. E-PLAN 구현(§L4·별도 arm) — A클래스(discovery)·§4가 근인 재확인(t95.0 중복-id=미발견의 직접 증거).
3. census 결과를 `RESEARCH_MASTER §3`(C63·C64)·§4 큐에 반영. ✅
4. B클래스 잔여(대화-precondition) = ENDGAME R1 설계·경계 몫은 P3 정직 보고.
