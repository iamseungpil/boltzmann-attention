# rate-formalize 서브에이전트 분담 설계 (2026-07-18)

> 계기: 사용자 *"이 기능만 하는 격리 request로 부하 없이 성공하면, 이런 기능 agent만 따로 호출할 수 있지 않나?"*
> ⇒ **[[10]] 분담의 3단 확장**: 메인(대화·흐름) → **rate-formalize 서브(격리·KB해석)** → 엔진(결정론 산술).
> 정본 상위 = `A2_DOMAIN_GENERALIZATION_DESIGN §PROD-2`. ⚠️**§8 하드룰 1**: §5에 반증예측 사전등록.

## 0. ★측정이 분담선을 데이터로 그었다 (`bank_rate_formalize_probe`·**n=8**·무료)
격리 request(KB 5문서 + 거래 4 + 개설일 02/13/2025 · **rate만·다른 부하 0**)에 32B가 낸 배율(셀별 8샘플):
| txn | 카드/카테고리/날짜 | 프로모(개설+6mo=08/13) | gold | 정확 | 판정 |
|---|---|---|---|---|---|
| 403 | Bus.Silver/Travel/03/22 | 활성(6mo 內) | 20 | **8/8** | ✅ |
| 506 | Silver/Software/05/18 | 무관 | 4 | **8/8** | ✅ |
| 410 | Bus.Silver/Travel/**08/25** | 만료(6mo 外) | 10 | 4/8 | ⚠️ |
| 411 | Bus.Silver/Software/**09/15** | 만료(6mo 外) | 10 | **1/8** | ❌ |
**"4/4 전부 정확" = 1/8 = 12%** — 단 이건 **날짜 산술을 모델에게 시킨** 값이다(재설계선 엔진이 함).

**★진단(단일):**
- ✅**base_rate 텍스트해석 = 완벽**: Business Silver→10·Silver→4·Travel/Software 카테고리 매핑 **100%**. KB 산문 오독 **0**.
- ❌**프로모 *날짜* 판정만 실패**: "프로모 존재"는 정확히 읽으나 **만료(개설일+6개월=08/13) 날짜 산술을 못 함**.
  403(6개월 內) 맞고 · 410·411(6개월 外) 여전히 2배.
⇒ **분담선 확정(데이터)**: **텍스트 해석 = LLM(완벽) · 날짜 산술 = 엔진(결정론).** 사용자 가설 그대로.

## 0b. ★★★실증 완료 — 산술을 한 칸씩 엔진으로 옮기니 12%→100% (2026-07-18·전수 무료)
**분담선 이동 실험** (같은 격리 request·4거래·각 n=8):
| 설계 | 도구가 담는 것 | 모델이 하는 것 | **4/4 정확** | 셀별 |
|---|---|---|---|---|
| formalize-only | 없음 | base+프로모+**날짜+곱셈** 전부 | **12%** | 8/8·8/8·4/8·**1/8** |
| v1 (날짜만) | 날짜판정 | base+프로모+**곱셈** | 부분(강제도 0/8) | 8/8·8/8·**0/8**·**0/8** |
| **v2 (날짜+곱셈)** | 날짜판정+곱셈=**최종배율** | **텍스트 해석만**(base 이름값+has_promo) | **★100%** | **8/8·8/8·8/8·8/8** |

★**진단 확정**: 32B는 **KB 산문 해석 강**(base_rate·프로모 존재=100%)·**모든 산술 약**(날짜 만료·곱셈).
- v1: 날짜 도구가 **410 만료 복구**(formalize 4/8→8/8·F2 호출 8/8·F3 신뢰 O) 그러나 **곱셈 결합서 새 실수**
  (411 base 10→4 오독·506 프로모없는데 ×2). **레버 A 강제도 못 고침**(곱셈은 모델 머릿속).
- v2: **곱셈까지 도구로** → 모델은 곱하지 않음 → **4/4 완벽·8/8 재현**.
⇒ ***산술을 전부 결정론 도구로 빼면, 모델의 유일 임무 = "각 거래에 도구를 올바른 인자로 호출" = 4/4 정확.***
= 사용자 통찰(부하격리+function-calling 분담)의 **완전 실증**. [[10]]/[[00]] 명제 깨끗한 사례.
★**§PROD ⑥("op로 못 담는다") 최종 철회**: 분담선을 제대로(산술 전부 엔진) 그으니 담긴다.

## 1. 왜 서브에이전트인가 (부하 격리)
- 메인 에이전트는 20턴 대화·신원확인·다도구 저글링으로 **컨텍스트 25k**. 그 부하 속 rate 정확도 ↓.
- **격리하면 컨텍스트 = KB+거래뿐** → base_rate 100%(측정). = 사용자 통찰 실증.
- ⇒ ***rate-formalize를 전용 서브에이전트로 분리 = 부하 제거 = 정확도 회복.*** [[00]] 명제(작은 LLM+분담)의 한 형태.

## 2h. ★★재설계 라이브 검증 (task_020·2026-07-18 NIGHT+·`_sub_inject` 구현본)

**구현**: `_sub_inject`(카드 group_by + 문서주입 + `{base_rate, promo*, exclusion_quote}` + grounding + default).
라이브 발화 확인(`bank_redesign020`): 카드별 격리(Silver 11문서·Business Silver 13문서 분리 주입).

**결과 진전 (over-flag 11→6·gold 4건 전부 정확·누락 0)**:
| 버전 | producer 반환 | gold | 오탐 |
|---|---|---|---|
| ratefix arm(메인서 operand·bm25) | 13 | 4 | 9 |
| iso5(태스크배치 격리·bm25) | 12 | 4 | 8 |
| **재설계 v1(카드격리+주입·promo누락)** | 11 | 4 | 7·base_rate 26/26정확 |
| **재설계 v2(promo수정)** | **6** | 4 | **2**·누락 0 |
- **★base_rate 오독 0/26**(라이브·프로브 90%보다 좋음). 카드격리+주입이 검색부실·혼동 완전 제거(실증).
- **v1→v2 버그수정**: operand_schema에 promo 누락 → 엔진이 프로모 2배 못 적용 → 프로모거래 전부 오탐. promo
  파라미터(서브formalize·엔진 날짜판정) 추가로 프로모 오탐 소멸(11→6).

**★남은 오탐 2건 = grounding 역효과 발견([D]·라이브가 프로브보다 엄격)**:
- Microsoft365·Coursera(Business Silver Software): 서브 `base_rate=0` + quote=**"Software Exclusion: Microsoft/
  Coursera…"**(근거 실재) → grounding이 0 유지. 그러나 **gold=기본율**(1%·프로모2%)이지 0 아님.
- **의미**: "Software 10% 적립에서 제외"는 **"0% 적립"이 아니라 "10%는 안 주고 기본율로 강등"**. 서브가 "제외"를
  "0%"로 오독·grounding은 quote 실재만 보고 0 유지 → **엔진 substring은 "이 제외가 0%인가 강등인가"를 못 가림**.
- ⇒ **grounding의 한계**: 인용이 실재해도 **그 인용의 의미(0% vs 강등)**는 substring이 못 판정. 프로브(105/105)가
  놓친 이유 = Business Silver Software 셀서 서브가 quote 안 냈었음(라이브가 더 엄격한 진실).
- **수정후보(미확정·[[08]] 재현 먼저)**: (a) 인용에 "0%/no cash back" 문자열이 실제 있나까지 grounding(강등제외는
  "0%" 문자열 없음→백필) (b) 서브에 "제외=강등이면 기본율" 명시. **다음 무료 프로브로 판정.**

**★★정책 확인 + demote 프로브 (2026-07-18·사용자 "제외시 기본율 정책인가?")**:
- **정책 원문 확정**: Business Silver "Exceptions" 문서 = *"following specific merchants do NOT earn the 10.0%
  bonus rate and **instead earn the standard 1.0% rate**"* = **강등(기본율)이지 0% 아님**. 반면 Bronze WeWork =
  *"earn 0% cash back"* = **완전제외**. ⇒ 정책이 두 종류를 **문장으로 명시 구분**. gold(Software 1%) 맞고 서브 틀림.
- **demote 프로브(강등/완전 구분 명시 A/B·`bank_demote_probe`·라이브 12거래 부하 재현 `--no_dedup`)**:
  | 셀 | base arm | demote arm |
  |---|---|---|
  | Microsoft365·Coursera(강등) | ✗ base=0 오독 | **✓ base로 고침** |
  | WeWork(완전) | ✓ 0 유지 | ✓ 0 유지(구분 성공) |
  | Adobe(정상 프로모) | ✗(부하) | **✗ 과교정 base=1**(강등 조심에 과반응) |
- **★결과 = 부분개선·모트 제1원리 실증**: demote 지시가 강등(Microsoft/Coursera)은 고치나 **비강등 Adobe를
  과교정**(레버 하나 사면 하나 판다·등대§1). Amazon(Shopping) 오독은 별개. ⇒ **프롬프트 튜닝으로 clean하게 안
  닫힘**([[42]] prompt-uncontrollable). ★**부하 재현 = 핵심**: dedup 4거래선 base arm도 통과·라이브 12거래선 실패
  = **문맥 부하가 강등 미묘함을 삼킴**([[45]]). = 프로브가 라이브 대표하려면 부하 동일 필수([[30]]).
- **~~다음 방향~~**: (a) grounding에 "0% 문자열 실재" 강화 = **cheating 기각**(사용자: 엔진이 "0%"를 패턴매칭=
  엔진-formalize·[[03b]]·도메인 리터럴). (b) 프롬프트 demote 지시 = **과교정 기각**(Adobe). ⇒ **(c) 부하축소로.**

**★★★부하축소가 강등 잔여를 닫음 (2026-07-18·사용자 지시·`bank_load_probe`·무료·결정론 3샘플)**:
라이브 서브 부하 = Business Silver **30거래**(Software 12·Travel 12…) 한 서브. 부하 A/B/C(프롬프트 고정·부하만 변수):
| arm | 서브 호출 | Software 오독 |
|---|---|---|
| **full**(카드 전체 30거래) | 1회 | 1(Adobe 과소·강등 놓침) |
| **cat**(카테고리별·Software 12거래) | 5회 | **0** |
| **batch**(카테고리+소배치 4씩) | 9회 | **0** |
- **★결과**: 카드 전체(30)는 오독 남으나 **카테고리 격리(12)면 오독 0**(Microsoft/Coursera 강등도 Adobe 정상도 다 맞힘).
  = **부하가 강등 미묘함을 삼켰고, 부하 줄이니 서브가 "instead earn 1.0%" 정책문장을 제대로 읽음**([[45]]).
  **엔진 패턴매칭(cheating)·프롬프트 과교정(demote) 둘 다 없이** 순수 격리단위 축소로 닫음. cat=batch(0=0)=카테고리로 충분.
- **★설계 확정 = `group_by`를 카드 → 카드×카테고리**: `_sub_inject`의 group_by를 복합키로. 부하↓·강등 잔여 종결.
  ⚠️서브 호출수↑(1→5)이나 각 호출 가볍고 병렬가능·메인 턴은 여전히 0(§2b). 트레이드오프=호출수 vs 정확도.
- **★원리 일반화**: 이 결과 = §2b(operand 격리)·§2e(카드 격리)·§2g(KB 격리)와 **동일 뿌리 = 부하 격리**([[00]]).
  **"작은 단위로 쪼갤수록 소형LLM 정확"**(§2f 논문축)의 추가 실증 — 격리 단위를 카드→카테고리로 한 칸 더 내리니 잔여 닫힘.

### ★★★★task_020 라이브 통과 확정 (2026-07-18 NIGHT+·`bank_redesign020` 카테고리격리본)
**reward 0 → 1** · `db_match=True` · **producer 반환 4 = gold 4 정확일치**(오탐0·누락0) · gold액션 5/6.
over-flag 궤적 = **13(ratefix)→12(iso5)→11(v1)→6(promo)→4(카테고리격리)=통과**. 이전 실패 태스크의 완전 종결.
각 단계 레버(전부 사용자 주도): ①카드격리+문서주입=base_rate오독0 ②promo파라미터=프로모오탐제거 ③카테고리격리
(부하축소)=강등잔여종결. **cheating(엔진패턴매칭)·프롬프트과교정 둘 다 없이** 부하 격리만으로.
⇒ **재설계 = 검증완료([S])**. 다음 = 나머지 실패태스크(026/027/028·페어)로 확대·§2g(KB격리)로 32k초과 해결.

### ★★확대검증 026/027/028 ([[08]] 포렌식·집계≠결론·2026-07-18)
**집계는 셋 다 reward=0이나 producer 반환으로 갈림 — rate와 하류를 분리해야 정직**:
| 태스크 | producer 반환 | reward | 실패 단계(gold액션 포렌식) |
|---|---|---|---|
| task_026 | **4=gold4**(오탐0누락0) | 0 | ★**rate 완벽**·dispute 4건 접수✓·**update_transaction_rewards 4건 MISS**(하류) |
| task_027 | **4=gold4**(오탐0누락0) | 0 | ★**rate 완벽**·**dispute 접수 전부 MISS**(적대변형·Phase2 거짓말서 막힘) |
| task_028 | 11 vs gold6(오탐5) | 0 | rate over-flag 남음(4카드 조합·재설계 미완) |
- **★정직 구분**: 재설계 목적=**rate over-flag 닫기**. 026·027서 **producer 4=gold4 = rate는 완벽 작동**(020과 동일).
  reward=0은 **rate 밖 하류**(update_rewards 실행·dispute 흐름·적대변형 대응) — 재설계 범위 아님. ⇒ **rate 관점
  026·027도 성공**(020 요행 아님·재현됨). 하류는 별개 과제.
- **task_028만 rate 미완**: 4카드(Crypto·Platinum·EcoCard·Silver)라 카드×카테고리 격리 조합 많음. 오탐 5 원인
  per-step 미조사(다음). 020(2카드)·026·027(2카드)은 닫힘·028(4카드)은 잔여 = **카드수 부하 가설**([[45]]).
- ⇒ **재설계 rate-formalize = 3/4 태스크서 producer 정확(020 pass·026·027 rate완벽·028 잔여)**. 하류(update/dispute)
  는 별개 레버. §2g(KB격리)가 다음 우선(32k초과=26%).

## 2g. ★★컨텍스트 초과의 주범 = 메인에 누적되는 KB 검색 (2026-07-18 사용자 통찰·포렌식 확증)

> 사용자: **"97런 26%가 32k 초과 실패. 각 기능을 서브에이전트로 쪼개고 메인은 주요 호출 return만 들면
> 32k 안 넘지 않나?"** → **데이터 확증**([[08]] 포렌식).

**포렌식(`bank_all97_nt1_v2` 완주 태스크 컨텍스트 역할별)**:
- 컨텍스트 초과 = 25/97(26%·18 ContextWindowExceeded). 초과 태스크는 msgs=0(터진 스텝 미저장)이라 완주 태스크로 대리분석.
- **KB 검색 결과가 컨텍스트 지배**: 상위10 중 9개서 KB **51~82%**·KB검색 있는 33태스크 **비중 중앙값 75%**.
  task_063=29k토큰 중 **KB 23k(79%)**·assistant 3k·user 1k. ⇒ 에이전트가 `KB_search_*`를 부를 때마다 **문서 전체가
  메인 대화에 쌓임** → 20턴쯤 32k 초과. (예외 task_078=KB0%=dispute반복 등 다른원인·소수.)
- ⇒ **초과 = 능력한계 아니라 메인 문맥에 무거운 KB를 이고 있어서**. 사용자 진단 정확.

**★구조적 함의 = 격리를 검색 전반으로 확장 (base_rate는 그 첫 사례)**:
- 우리 base_rate 격리(§2b~2e)는 **producer의 KB검색을 서브로 뺐다** — 그 producer 몫 KB는 메인서 사라짐.
- **남은 초과 = 에이전트가 *직접* 부르는 KB검색**(정책·절차 조회)이 메인에 쌓임. 이것도 **서브로 격리**하면:
  메인은 서브 return(요약·operand·discrepant id) 등 compact 값만 → KB 23k가 메인서 사라짐 → 32k 안전.
- = 사용자 원칙(§2b "operand는 격리")의 일반형: **무거운 자료(KB문서)는 서브 문맥서 소비·메인은 결과만**.
  [[00]] 분담(작은LLM+격리)·§2f 논문축(분담→retrieval제거)과 동일 뿌리.
- **⚠️설계점(미확정)**: 에이전트 직접 KB검색을 어떻게 서브로 돌리나 — (a) KB검색 도구를 서브-래핑(에이전트가
  질의→서브가 검색·읽고→요약/답만 반환) (b) 검색 자체를 A2 producer화. **다음 설계·[[08]] 초과18 per-step 먼저**.

## 2c. ★★★구현·라이브 실증 (2026-07-18 NIGHT+·`T2_SG_ISOLATE`)

**구현**: `t2_scaffold_get.py::_sub_formalize` — producer 도구 실행경로(`exec2`) 안에서 격리 서브루프.
- 메인 대화(`state.messages`)에는 **producer 호출 1 + 결과 1**만. 서브 generate·getter는 그 밖 = **메인 턴 0**.
- 서브 입력 = 원시 레코드만(`row_fields` 화이트리스트가 메인 추측 operand를 버림). 1라운드 `tool_choice=required`.
- getter = A2 선언(`isolate.getter_tools`=`KB_search_*`)·env 결정론 실행·결과를 서브 문맥에 되먹임(GET·[[03b]] 무spoon).
- 엔진 리터럴 0: 도구명·질의지시·계약문·operand 스키마 전부 A2 `isolate`. 미선언 or 실패 → 메인 인자 폴백(거동 0).
- 오프라인 배선 8항목 PASS(`test_sg_isolate.py`).

**★라이브 실증 (task_021·유료·seed 300·단일변수 `T2_SG_ISOLATE`만 ratefix arm과 차이)**:
| | ratefix arm(메인서 operand) | **격리 서브(iso021)** |
|---|---|---|
| reward | **0.0**(user_stop) | **★1.0**(user_stop·db_match=True) |
| KB 검색 | `KB_search_dense` 봉투드롭 2회(정책문서 미독) | 서브가 **KB 2회 실검색** |
| base_rate 정확 | 오탐 유발 | **17/17행 정확**(WeWork 0%·Green 5%·머천트예외 다 맞음) |
| producer 반환 | discrepant 3(오탐 1) | **discrepant 2 = gold와 정확 일치**(오탐0·누락0) |
⇒ 사용자 원칙("operand는 격리 서브가 부하 없이 산출·리턴")의 **라이브 e2e 실증**. 메인서 오탐 10내던 base_rate가
격리서 17/17. **서브가 스스로 GET**(문서 떠먹임 아님) = §2b "열린 GET 질문"에 이 궤적은 "GET 공짜" 답.

**⚠️정직 (포렌식 가드·[[08]])**: **n=1 pass^1 = 존재증명(작동)이지 비율주장 아님.**
**⚠️내 회귀 2건 잡음**(고침): ①§2b 계약문 "레코드 그대로"→모델이 `"$126.36"` 문자열→`_num` 못읽어 17행
판정불가→discrepant 0(under-action 위장). ②엔진 skip이 조용. → 계약=정규화 숫자 요구·엔진 skip 계측 추가.

## 2d. ★★5태스크 페어 = **현 격리설계 반증**(2026-07-18 NIGHT+·`bank_iso5_20260718`·사용자 승인 중단)
5태스크(020/022/026/027/028·seed 300·단일변수 `T2_SG_ISOLATE`)를 돌렸으나 **25분간 완주 sim 0** → 사용자 승인
하에 중단. **완주 0 자체가 결과**(집계 아님·[[08]] 종료사유). 확정 사실(라이브 로그·[D]→기전은 [M]):
- **★설계 결함 1 — 배치 격리가 부하를 만든다**: 서브가 **producer 호출마다 KB getter 10회** → task_022(75거래)는
  **서브 요청 172,731토큰**으로 컨텍스트 초과(폴백→47/47 판정불가→discrepant 0). "부하 없이 격리"라면서
  거래 전체를 한 배치로 넣어 **부하를 재생산**. task_021이 통한 건 17거래·프로모無라 배치가 작아서.
- **★설계 결함 2 — over-flag = base_rate 환각(온도-분산)**: **[?]→[D] 확정**(무료 `bank_overflag_probe.py`·
  task_020 26거래·격리 서브 프롬프트 그대로·GPU0). **같은 프롬프트·같은 거래인데 샘플간 극단 분산**:
  | 샘플 | base오독 | flagged | getter |
  |---|---|---|---|
  | [0] | 0 | **0(완벽)** | 26회 |
  | [1] | 0 | **0(완벽)** | 12회 |
  | [2] | 9 | **11(over)** | 10회 |
  [2]는 base_rate를 **KB에 없는 값(3·5)으로 환각**(Business Silver Software=5인데 실제10·Silver Groceries=3인데1).
  getter도 [0]26회 vs [2]10회 = **KB를 덜 읽고 base를 지어냄**. ⇒ over-flag 원인 = **온도 0.7서 서브가 확률적으로
  KB검색 덜 하고 base_rate 환각**([[43]] 채점이 추측>기권 보상). iso5 라이브 12·15·12와 일치. task_021이 통한 건
  **운 좋은 샘플**(존재증명이나 robust 아님·[[08]] pass^1 경고 정확). ⇒ **격리 자체는 부하는 줄이나 환각은 안 막음**
  — 필요레버 = base_rate **grounding**(값 ∈ KB출력·[[16]] provenance·미grounded 거부) or **낮은 temp**.
  **~~temp=0 "확정"~~ = 프로브 버그였음 (아래 §측정오류 자백 참조)**: temp=0의 "base오독 0·flagged 0"은
  **거짓 음성**이었다. `bank_overflag_probe.py:136` `if ... not isinstance(br,(int,float)): continue` — 서브가
  base_rate를 **문자열 `'10'`**로 내는데 판정이 그 행을 **전부 스킵** → "완벽"으로 오판. temp=0.7 [2]만 우연히
  일부 숫자를 내 오독이 잡혔다. ⇒ **"온도-분산" 자체가 프로브 버그의 산물**(서브는 두 온도 다 오독).

  **★★★측정오류 자백 (2026-07-18 NIGHT+·사용자 "혹시 날조?" 추궁으로 발견·[[30]]/[[08]])**:
  - **원본 로그는 실재**(날조 아님·`overflag_020_t0.log` 원격 타임스탬프 13:18 = base오독0×4). **그러나 그 수치가
    거짓**: 판정기가 문자열 operand를 스킵해 오독을 못 셌다. **"관측이 있다"≠"관측이 옳다"** — 집계수치를
    궤적/타입 확인 없이 믿은 [[08]] 위반.
  - **진실**(판정기 `float(br)` 강제 후 재실행): temp=0에서도 **서브 base_rate 오독 다수**(Business Silver Travel
    10→sub 5·Software 10→sub 2·Silver Software 4→sub 1). = 라이브 iso020_trace(오독 15/26)와 **일치**.
  - **KB 검색도 내가 오해**: "무료=전문서"라 했으나 실제 로컬 `_kb`도 **상위 k개만**(word-count) 반환 = bm25와 동류.
    게다가 로컬 검색이 **엉뚱 문서 반환**(FAQ·Diamond·k=1) → 서브가 rate 근거 문서를 못 받아 오독. 전문서(213k토큰)는
    32k에 **물리적으로 못 들어감** — "전문서로 완벽" 진술은 **이중 오류**(전문서 아님·완벽 아님).

  **~~정정~~ 이하는 위 자백으로 대체됨 (참고·2026-07-18 NIGHT+·`bank_iso020_t0`):**
  - **라이브 에이전트 llm_args는 이미 temp=0**(`t2_run_gated.py:221`)이었다 ⇒ iso5 over-flag는 **온도 아님**.
    무료 프로브의 "temp=0.7→분산"은 **라이브를 대표 못 함**(단위≠라이브·[[30]]). 온도-분산 결론 **철회**.
  - **temp=0 라이브 재실험도 여전히 over-flag 12**(task_020·r=0). ⇒ 온도 수정 무효.
  - **★진짜 차이 = KB 검색 도구**: 무료 프로브=내 로컬 `_kb`(전 문서 word-count·getter 24회) vs
    라이브=env `KB_search_bm25`(bm25·getter 10회). **검색이 다른 문서를 반환** → 서브가 읽는 근거가 달라
    base_rate가 달라진다. **무료 프로브가 라이브 KB검색을 재현 안 해서 잘못된 원인(온도)을 지목**했다.
  - ⚠️**디버깅 공백 해소**(계측 `T2_SG_ISOLATE_TRACE` 추가·`_isolate_trace`): 서브 operand를 JSONL로 남김.
  - **★★over-flag 진짜 원인 = 서브 base_rate 오독([D]·계측 확정·2026-07-18 NIGHT+ `bank_iso020_trace`)**:
    라이브 서브가 26/26행 산출했으나 **base_rate 오독 15/26**(엔진 재현 flag 13 = gold 4 + 오탐 9·누락 0). 오독 2종:
    | 오독 | 예 | 원인 |
    |---|---|---|
    | **base_rate 오독** | Silver 4%→`base=10`·Business Silver 10%→`base=2/5` | **검색이 그 카드 rate 문서를 안 집어줌** → 서브가 근거 없이 지어냄 |
    | **프로모 날짜 오독** | 만료 거래에 `pmult=2` 유지(c2d3e404·e406) | 서브가 promo 만료 판정 못 함(엔진 몫인데 서브가 formalize) |
    ⇒ **판정 (A)=서브 문제**(엔진/promo 결함 아님·온도 아님). **무료 프로브도 실은 라이브와 같은 오독**(자백 참조)
    — 무료 프로브의 "완벽"은 판정기 버그였고, 실제로는 두 경우 다 서브가 **rate 근거 문서를 제대로 못 받아**
    base_rate 오독. 근본 = **검색이 유사카드 rate 문서를 정확히 못 집어줌**(bm25든 내 로컬이든).
  - **부수 확인**: 서브가 base_rate를 **문자열**(`'10'`·pmult=`'2'`)로 냄 — 계약이 "숫자" 요구하나 미준수(_num이
    흡수하나 [[03b]] 관점 부적). operand_schema를 number로 강제 필요.
  - **★다음(무료·설계)**: over-flag 근본 = **검색이 그 카드의 rate 문서를 서브에 못 줌** + promo 서브-formalize.
    수정 방향(3):
    (1) **카드당 격리** → 배치 172k 해소·한 카드만 문맥(§2d 결함1·3 동시 해결).
    (2) **카드 문서 직접 주입**(검색 우회): 카드당 문서 ~3k토큰(제목 접두 기계선택·도메인리터럴0)을 서브 문맥에.
       전문서(213k)는 32k 불가하나 **카드당은 들어감**(태스크 전체 카드도 최악 ~11k). 검색의존 제거=근본원인 직격.
       ⚠️리더보드: banking은 `retrieval_config` 표기 의무(§submission.md:83) — "검색우회"면 그렇게 보고(Custom이라 허용).
    (3) **promo를 서브서 빼기** → 서브=base_rate만·promo 날짜판정=엔진(C113 분담선·서브 못하는 산술 offload).
    ⇒ **재설계 = 카드당 격리 + (카드문서 주입 or 검색보정) + base_rate만 + promo/곱셈 엔진.** [[03]] 설계 먼저·큐 밖.
- **★설계 결함 3 — sim 완주 불가능하게 느려짐**: getter 10회×producer 다회 → 태스크당 20분+ → e2e 비현실적.
⇒ **결론: 현 "태스크-배치 격리"는 반증됨. 사용자 원칙(operand=격리)은 유효(task_021 실증)하나 격리 단위가 틀림.**
   **재설계 방향(다음)**: 격리를 **거래(또는 카드) 단위**로 잘게 — 배치 172k → 카드당 1~5거래. KB 검색도
   카드당 1회 캐시(현 10회 낭비). ⇒ 부하·속도·over-flag 셋 다 완화 가설. **`RESEARCH_MASTER` 실험큐 밖 =
   설계부터**([[03]] anti-drift). ⚠️over-flag 원인([?]) 먼저 격리 프로브(무료)로 재현·확정 후 재설계.

## 2e. ★★재설계 — 카드당 격리 + 카드문서 주입 (2026-07-18 NIGHT+·사용자 승인·설계 LOCK)

> §2d서 확정: 현 "태스크-배치 격리"는 (1)172k 컨텍스트초과 (2)유사카드 혼동 over-flag (3)20분/태스크로 반증.
> 근본원인 = **검색(bm25/로컬)이 그 카드의 rate 문서를 서브에 정확히 못 줌** → 서브가 base_rate 지어냄.
> ⇒ 격리 **단위를 태스크→카드**로 바꾸고, 문서를 **검색 대신 결정론 필터로 주입**한다.

### 구조 (3층 분담·[[10]])
```
① 메인: producer 호출(원시 거래 레코드 배열만) — 종전과 동일(§2b)
② producer exec2 (엔진·인프로세스): 거래를 credit_card_type으로 그룹핑 → 카드마다:
     ③ 카드-격리 서브 1개 호출:
         입력 = [그 카드 문서 전부(제목 접두 매칭)] + [그 카드 거래들(원시)] + 개설일
         임무 = 거래마다 base_rate(percent)만 formalize   ← promo/날짜/곱셈 안 함
         출력 = {txn: {base_rate}}
④ 엔진(select_discrepant): base_rate(서브) + promo판정·곱셈(엔진·C113 date op) → expected vs actual
```

### 핵심 4결정 (각 §2d 결함 직격)
1. **격리 단위 = 카드** (배치 172k·유사카드 혼동 동시 해결):
   - 한 서브 = 한 카드만 → **Silver vs Business Silver 혼동 물리적 불가**(다른 카드 문서가 문맥에 없음).
   - 서브당 문서 ~3k토큰·거래 몇 개 → 32k 여유·빠름. (최악 task_022=4카드×서브·각 ~3k, 순차/병렬.)
2. **문서 = 검색 대신 결정론 주입** (검색부실 우회·오늘 근본원인 직격):
   - 규칙 = `doc.title.startswith(credit_card_type + ": ")` — **순수 기계판정·도메인 리터럴 0**([[05]]).
     ★`+ ": "` 필수: 문서 제목이 전부 `"카드명: 소제목"` 형식(실측). 순수 startswith면 카드명이 다른 카드명의
     접두일 때 오염(현 11종엔 그런 쌍 없으나 방어). ⚠️역방향 안전 확인됨: "Business Silver..."는 "Silver "로
     시작 안 하므로 Silver 필터에 안 섞임(2026-07-18 실측).
   - **그룹핑은 서브가 안 함**: 어느 거래가 어느 카드인지는 레코드 `credit_card_type`에 **확정**(벤치 사실)·
     엔진이 그 값으로 그룹핑. Silver 서브엔 Silver 거래만·한 카드만 → 서브는 "어느 카드?" 판단 불요
     (=오늘 over-flag의 원인 제거: 지금은 한 서브에 2카드 섞여 서브가 4%인지 10%인지 헷갈림).
     카드명은 레코드서 옴(우리가 "어느 카드=10%" 안 넣음). **spoon 아님**([[03b]]): rate·예외·보험·수수료
     문서 전부 주입(정답 문서 선별 안 함)·서브가 그 안서 rate를 **스스로 읽어야** 함.
   - 실측 상한: 태스크당 카드문서 총합 최악 ~11k토큰(022·028)·2카드 ~6k(020/026/027). 32k 안전.
3. **promo를 서브서 제거** (§2d promo 오독 직격·C113 분담선):
   - 서브 = **base_rate만**. promo 적격/활성 날짜판정·곱셈 = 엔진(`date_between`·`date_in_window`·multiply·기구현).
   - 서브가 promo_mult/start/end를 formalize하던 것 삭제 → promo 날짜 오독 원천 제거.
   - ⚠️단 **프로모 존재+배율**(예: "2배")은 KB문서 사실이라 서브가 읽어야 함 → operand = {base_rate, promo_mult
     (프로모 없으면 1), promo_start, promo_end, promo_window_months}. **날짜 판정(적격/활성)만** 엔진. (C113 ratefix op 그대로.)
   - 재검토: §2d 결함은 "서브가 만료를 직접 판정"이 아니라 **만료 거래에 pmult=2를 남긴 것**. op는 이미 날짜로
     mult를 0/1 거르므로(promo_active=False면 mult=1), **서브가 pmult·기간만 정확히 formalize하면 엔진이 만료 처리**.
     ⇒ promo 오독의 진짜 수정 = 카드-격리로 서브가 그 카드 프로모 문서를 정확히 읽게 하는 것(결정1과 동일 레버).
4. **operand 타입 강제** (§2d 문자열 버그): 서브 출력 `base_rate`는 number. 계약문에 명시 + 엔진 `_num` 방어 유지.

### A2 스키마 변경 (`isolate` 확장·엔진 무수정)
```
isolate:
  group_by: credit_card_type          # ★신규: 카드당 서브 분할 키(레코드 필드·서브가 선택 안 함)
  doc_filter: title_startswith_colon   # ★신규: title.startswith(값+": ")·접두오염 방어
  doc_filter_field: credit_card_type   # 매칭값 출처(거래 레코드 필드)
  getter_tools: []                     # ★검색 안 씀(주입이라) — 빈 목록
  inject_docs: true                    # 서브 프롬프트에 필터된 문서 첨부
  row_fields: [...]                    # 종전
  operand_schema: {base_rate:number, promo_mult:number, promo_start, promo_end, promo_window_months}
  instructions: "이 카드 문서를 읽고 각 거래의 base_rate를..."
```
- 엔진(`_sub_formalize`) 변경: `group_by`면 rows를 그룹핑·그룹마다 `doc_filter`로 문서 골라 주입·서브 호출·operand 병합.
  **도구명·필터규칙·문서출처 전부 A2**·엔진은 그룹핑+문자열매칭+주입 루프만([[05]]).

### 반증예측 (사전등록·§8 하드룰1)
- **옳다면**: ①카드-격리 서브가 base_rate 오독 **0~소수**(혼동 카드 문서 부재) ②172k 소멸(카드당 ~3k)
  ③over-flag(12/15)→gold근접(4) ④sim 속도 회복(getter 폭발 제거).
- **반증**: ①카드-격리인데도 base_rate 오독 지속 → 서브가 **한 카드 문서서도** rate 못 읽음(=formalize 부하·[[45]]
  ·§PROD-2 복귀) ②문서 주입이 spoon으로 판정되면(리뷰) → 필터 규칙 재검토.
- **검증 순서(무료 먼저·[[09]])**: (a) 카드-격리 서브 **오프라인 프로브**(문서 주입·base_rate 정확도 n샘플) →
  통과 시 (b) `_sub_formalize` 구현 + 오프라인 배선테스트 → (c) task_020 라이브 단일(유료·승인).

### ★★검증 (a) 결과 — 카드당 문서주입 프로브 (`bank_percard_probe`·무료·temp0·n=3·2026-07-18 NIGHT+)
**8카드×카드문서 전량 주입·검색0. base_rate 정확 = 90%(135/150)**. 대조 = 라이브 bm25(2카드 섞음·오독 15/26=42%).
| 카드 | 정확 | 잔여 실패(전수 포렌식) |
|---|---|---|
| **Silver Rewards Card** | **21/21 100%** | — (★오늘 라이브서 base=10 혼동나던 카드가 격리하니 완벽) |
| Business Bronze/Platinum·Diamond·Gold | 100% | — (WeWork 0%·Gold 2.5 등 정확) |
| Business Silver | 27/33 82% | Software: Microsoft365·Coursera **sub=0** vs gold 2%/1% |
| EcoCard | 15/18 83% | ThredUp(Other) **sub=0** vs gold 1% |
| Crypto | 15/21 71% | Netflix·AMC(Entertainment) **sub=0** vs gold 2% |
- **★핵심 실증**: 카드 격리+문서주입이 **검색부실·카드혼동을 우회**(Silver 42%→100%). 재설계 방향 (a) 통과.
- **★잔여 실패 성격 = 지어냄 아님·질 다름**([[08]] 포렌식): 전부 **서브가 `0`** — "이 머천트/카테고리가 프리미엄
  적립 자격이 있나"를 **과엄격 판정**해 0으로 깎음. **놓친 건 "모든 구매는 최소 기본율(1%) 적립" 규칙**(gold=기본
  1%는 줌·서브=프리미엄 아니니 0). = 검색부실/혼동 아님 → 문서주입으로 안 닫히는 **다른 잔여**.
  ⚠️수정후보(미결): 기본율 명시 강화(프롬프트) or 기본율=엔진 default(정책-강제·gate). **단정 전 재현**([[08]]).
- **⚠️프로브 버그 2건 자백**(포렌식이 잡음): ①`gold_pts`는 **정수 반올림** 저장(9.47×2=18.94→18) → rate 직접비교가
  Gold/Crypto 정답을 오답처리(80%로 과소) → **포인트 재구성 ±1** 판정으로 고침 → 90%. rate≠정확한 gold.

### ★★검증 (b) — 전체주입 vs rate-문서만 (`--doc_mode`·무료·사용자 질문 "2문서만 주면?")
**★분모 맞춘 비교(Gold 제외·rate-arm서 SKIP)**: 공통 135셀 = 전체 **89%** vs rate-필터 **76%**([[08]] 분모오류
정정: 초기 90 vs 76은 전체=150·rate=135 사과-오렌지였음). **부분선택이 나쁨**·드라이버 2카드:
| 카드 | 전체 | rate만 | 왜 (포렌식 확인) |
|---|---|---|---|
| Silver Rewards | **100%** | **29%** | rate문서(`How to Earn 4%`) 남겼는데 급락 — 빠진 `Maximizing Rewards`·`Getting Started` 본문에 **실측 1.0%** 있음(기본율) |
| Crypto | 71% | 57% | 동종 |
| Gold Rewards | 100% | **SKIP(rate제목 0)** | Gold rate가 `Apply for a Premium Card`류 **rate 안 보이는 제목**에 → 키워드 필터가 카드 전멸 |
| Bronze·Platinum·Diamond·EcoCard·Business Silver | = | **동일** | 변화 0 |
- **★결정적 발견([D]·본문 grep 확인)**: **"어느 문서에 rate가 있는지 제목으로 알 수 없다."** Silver 기본율(1%)이
  rate-제목 아닌 문서에 있음(실측)·Gold는 rate-제목 0. ⇒ **부분선택 = retrieval 실패의 재현**. **전체주입이
  필요조건**(노이즈 감수가 정보누락보다 안전). 사용자 명제("retrieval 대신 관련문서 전체")를 **데이터 강지지**.
- ⇒ §2e 배포안 = **카드문서 전체**(rate선별 금지 = spoon 회피 이전에 **기계적으로 불가능**함도 확정).

### ★★잔여(기본율 누락) 두 안 프로브 (2026-07-18 NIGHT+·사용자 지시·무료)
잔여 = 서브가 비프리미엄 카테고리에 `0` 남발(기본율 놓침). 두 안 대조:
| 안 | 방법 | 잔여 고침 | 예외 깸 | 판정 |
|---|---|---|---|---|
| **1안** | 프롬프트 힌트("0 남발마·기타구매 기본율"·값 안 알려줌·도메인일반) | Crypto 71→**100%**·**Business Silver·EcoCard는 못 고침** | 0 | 안전·**불완전** |
| **2안** | 서브가 카드 기본율 KB서 formalize(1.0/2.0/1.0·엔진 리터럴0) → 0셀 백필 | fix 1/2/1(전부) | **WeWork 깸(break=1)** | 완전·**위험** |
- **1안 부분성 = [[42]] 실증**: 프롬프트로 강조해도 서브가 "Software인데 적격SaaS 아니니 0" **과엄격 유지**
  (Business Silver Software 안 고침). prompt-uncontrollable.
- **2안 위험 = grounding 부재**: 무조건 백필이 **정답 예외(WeWork gold=0)를 기본율로 덮음**(Bronze 100%→깸).
  "0이 예외(정답)인지 과엄격(오류)인지" 백필은 **구분 못 함**.
- **★진짜 답 = grounding([[16]])**: 서브의 `0`이 **KB 예외 문서에 grounded**(문서에 "이 머천트/카테고리 0%")면
  유지 · **근거 없는 0**만 기본율 백필. 1안(프롬프트)도 2안(무조건백필)도 이 grounding을 안 해서 각각 불완전/위험.
  ⇒ **재설계 = 카드격리+전체주입(§2e) + 기본율 default(2안·값=서브formalize) + 예외-grounding 게이트(0 유지 판정)**.

### ★★★grounding 통합 실증 — 완성 재설계 100% (`bank_grounding_probe`·무료·temp0·n=3·거래 105)
분담([[10]]): **서브(생성)=`{base_rate, exclusion_quote}`** — 0 낼 땐 KB서 예외 문장 발췌 · **엔진(검증·결정론)=
그 인용이 문서에 실재하나**(정규화 substring). grounded 0=유지 · ungrounded 0=기본율 백필. 5 결합규칙 교차표:
| 규칙 | 정확 |
|---|---|
| R0 raw / R1 hint / R3 불일치→ground / **R4 any0→ground** | **105/105 100%** |
| R2 raw+무조건백필 | 102/105 97%(WeWork 깸) |
- **R4 grounding 동작**: 0유지 3(**잘못유지 0**) · 0백필 0(**예외파괴 0**). WeWork: raw=0·quote=`"Coworking Space
  Merchants: WeWork, Regus…"`·**grounded=True→유지**(포렌식 확인). 무조건백필(R2)만 WeWork 깸.
- **★★예상 못 한 발견([D])**: **"0 낼 거면 근거를 대라" 요구가 raw 자체를 고침.** §2b서 과엄격 0이던
  Microsoft365·Coursera·ThredUp이 quote-요구 프롬프트선 **base=1(0 아님)**. ⇒ 근거요구가 [[43]] 역방향(기권/과엄격
  억제) → R0_raw도 100%. 즉 **grounding 지시가 프롬프트로도 잔여 대부분 닫고, 남는 진짜예외(WeWork)를 엔진
  substring이 지킴**. 이중 안전(프롬프트+엔진검증).
- **★설계 확정**: `{base_rate, exclusion_quote}` operand + 엔진 substring-grounding + 기본율 default. **엔진 리터럴 0**
  (예외목록·기본율·인용 전부 서브가 KB서·엔진은 substring 매칭만). §PROD-2 잔여 **닫힘**.
  ⚠️여전히 무료·오프라인 격리 프로브. 라이브 e2e(§2e 검증c)·`_sub_formalize` 구현은 다음.
- **★★일반화 = FIND-근거요구 원리 (2026-07-18 사용자·`GENERALIZED_SCAFFOLD §4d` LOCK)**: 이 quote-요구는
  base_rate 특수가 아니라 **모든 FIND의 표준**. FIND=자료서 찾기 ⇒ "무엇 보고 찾았나" 근거는 자연출력. 효과=
  환각/과엄격 억제([[43]]역) + 엔진-검증가능([[10]]). **측정 프로토콜(사용자)**: FIND-계열 실험은 근거요구 A/B 상시
  계측. **측정 대상(코드 확인)**: `SUBCALL_SYS`(disamb 서브콜·현재 근거요구 **없음**=첫 A/B) · `reference_filter` ·
  entity-binding · value-extraction. **다음 무료 실험 = disamb 서브콜에 근거인용 요구 A/B**(t71/t106류).

## 2f. ★논문 프레이밍 검토 — "분담→retrieval제거" 축 ([[46]] 대조·2026-07-18 NIGHT+·사용자 제안)

> 사용자 명제: **"큰 작업을 작은 단위로 쪼개고, 작은 단위 서브의 정확 판단을 위해 retrieval 대신 관련 문서
> 전체를 넣어 극복한다."** 이게 의미 있는 논문 축인가?

**[[46]] novelty map 대조 (과장 금지·선점 정직)**:
- **이미 선점(양보·인용)**: RAG vs long-context tradeoff·retrieval이 해칠 때·lost-in-middle·chunk noise = **활발한
  기성 영역**. "관련문서 전체주입" 단독은 **새롭지 않음**. (딥리서치 `wf_a6875b00` 확정 대기.)
- **[[46]] core와의 관계**: Paper1 core = lever-배분 + pass^all-compliance crossover. 이 "retrieval제거"는 **별개 축**
  (검색 아닌 분담). Paper1에 끼워넣기보다 **분담-원리의 한 사례**로 위치.
- **새로울 수 있는 지점 = 인과사슬**: (분담으로 단위를 작게) → (작아서 관련문서 全 삽입 가능) → (그래서 소형LLM이
  retrieval 랭킹오류 없이 정확). 즉 "retrieval 제거"가 독립기법이 아니라 **분담의 산물**. [[00]] 명제(소형+분담→대형)의
  구체 기전. **오늘 데이터가 이를 지지**: 검색(bm25) 42% → 카드격리+전체주입 90%(§2e). 부분선택 76%(§2b: 전체가 필요조건).
- **정직**: 이 인과사슬이 [[41]] 4축·기성 RAG-vs-LC와 진짜 구별되는지는 **딥리서치 결과로 확정**. 지금은 [?].
  기여로 올린다면 "measured: decompose-enables-retrieval-drop for small-LLM factual extraction"(τ² banking 실측).

### ★★딥리서치 결과 (`wf_a6875b00`·2026-07-18·검증완료·종합만 stall→수동종합)
**부품 4개 전부 선점**(양보·인용 필수)·**결합(인과사슬)은 미선점 정황**:
| 우리 주장 조각 | 선행연구(검증됨=[확인]) | 판정 |
|---|---|---|
| 전체주입 > RAG(검색랭킹오류 회피) | **arXiv:2501.01880**(Long-Context vs RAG·LC 56.3% vs RAG 49.0%·[확인]) | **선점**·인용 |
| 부하(문맥길이) 자체가 정확도 떨어뜨림 | **EMNLP2025 Findings "Context Length Alone Hurts Despite Perfect Retrieval"**(13.9~85%↓·perfect retrieval서도·[확인]) | **선점**·**우리 §2h 부하축소 직접 지지** |
| 문맥 축소→정확도↑(recite/scope) | ↑ 같은 논문(recite evidence로 +4%·task를 short-context화) | 선점·인용 |
| 서브에이전트 scoped-context 분담 | **Anthropic multi-agent**(각 서브 독립 문맥·토큰이 성능 80% 설명)·**DACS**(full scoped context 주입=steering 90~98% vs flat 21~60%·**결정론·no ranking**) | **선점**·DACS가 최근접 |
| 소형모델은 문맥활용이 병목(검색 아님) | **oracle-retrieval서도 7B이하 85~100% 실패·문맥주입이 42~100% 기지답 파괴**(sub-7B RAG 한계=utilization) | **선점**·우리 소형LLM 전제 지지 |
- **★핵심 = DACS가 우리와 가장 가까움**: (분담)+(scoped full-context 주입)+(결정론·no ranking) 다 겹침. **우리 delta**:
  DACS=멀티에이전트 steering 정확도(일반), **우리=τ² tool-use e2e·소형LLM·엔진offload와 결합·격리단위를 부하로
  측정(카드→카테고리)·compliance(over-flag→pass)**. "retrieval-drop" 단독은 확실히 선점(2501.01880) → **양보·인용**.
- **★refuted 정직**: "lost-in-middle이 retrieval 비용" 주장은 **REFUTED**(lost-in-middle은 long-context 현상·검색과
  반대). "RAG dominant failure=retrieval miss"도 과일반화 REFUTED(1개 세팅). ⇒ 이 두 프레이밍 논문서 쓰지 말 것.
- **⇒ 논문 위치 확정**: "retrieval 대신 전체주입"은 **헤드라인 금지**(2501.01880 선점). 헤드라인=**[[46]] core(lever-배분
  +crossover)** 유지. 이 축은 **분담-원리의 실측 사례**로 §2f처럼 종속 배치·부품 전부 인용(2501.01880·EMNLP2025·
  DACS·Anthropic·sub-7B utilization). **모트는 여전히 crossover**([[46]]). ⚠️DACS 인용 필수(최근접·미인용시 리뷰어 kill).

### ★★★DACS 정독 (arXiv:2604.07911·2026-07-18·사용자 지시·최근접 이웃 delta 확정)
**DACS = Dynamic Attentional Context Scoping**. 겹침 = **"문맥 격리→소형LLM 결정정확도↑" 원리 하나뿐**(flat 21~60%
→격리 90~98%·우리 카드30→카테고리12 오독제거와 동뿌리). **나머지 전부 다름**:
| 축 | DACS | 우리 |
|---|---|---|
| 격리 대상 | **에이전트 대화상태**(status·진행) | **KB문서·operand**(정책자료) |
| 결정 내용 | 관리자가 서브에 줄 **답(A/B steering)** | 서브가 자료읽고 **값 formalize**(base_rate·promo) |
| 결정론 엔진 | **없음**(순수 문맥라우팅) | **있음**(날짜·곱셈·discrepant·grounding 검증) |
| formalize+검증 | 없음(키워드매칭 채점) | 서브 formalize→엔진 substring grounding 검증 |
| 과제 | **합성 시나리오**(BST코드·서베이 toy·160+40 trial) | **실제 tool-use e2e**(τ² banking·real gold·reward) |
| retrieval | 안 다룸 | **검색을 문서주입 대체**(bm25 42%→90%) |
| 모델 | MiniMax-M2.7·Haiku 4.5 | Qwen 32B(소형)·엔진offload |
- **★핵심 delta**: DACS=순수 **문맥창 관리**(계산·검증·formalize 0·합성과제). 우리=**formalize+결정론검증+계산 분담**
  ([[10]])·실제 tool-use·retrieval대체·compliance(over-flag→pass). **DACS엔 우리 실질(엔진offload·grounding·e2e) 0**.
- **★리뷰어 방어**: DACS "deterministic·no ranking·exact not approximate" 표현이 우리 결정론격리와 겹침 → **인용하며
  delta 명시**(우리는 문맥라우팅 아니라 formalize-검증-계산 분담). DACS 저자 자인한 한계 = 합성과제·frontier미테스트·
  에이전트상태 격리 국한 = 우리 whitespace와 정확히 상보. ⇒ **인용·양보(격리원리)+delta 전면(실질 전부)**.

## 2. ★분담 구조 — 서브에이전트가 날짜엔진을 **function calling으로 호출**(사용자 2026-07-18)
> ~~§2-v1(서브=파라미터만 → 엔진이 op)~~ 개선: 서브가 **완성된 rate**를 반환하되, 자기가 못하는 날짜산술만
> **도구 호출로 offload**한다. ⇒ 서브에이전트 *안에서* 다시 [[10]] 분담(생성=LLM·계산=결정론도구)이 일어난다.

```
① 메인 에이전트: reward-discrepancy 필요 판단 → 서브 호출(격리)
② rate-formalize 서브 (격리·부하0·도구=날짜엔진 1개):
     입력 = KB rate문서 + 거래들 + 개설일
     (a) base_rate·프로모 파라미터 formalize   ← 측정 100% (LLM 강점)
     (b) 날짜판정이 필요하면 → ★도구 호출:
            promo_active(account_open, txn_date, window_months, promo_start, promo_end) → bool
         ← 결정론 엔진이 계산해 bool 반환(모델이 실패한 그 산술·측정 12%)
     (c) 도구 결과(bool)로 최종 rate 완성: rate = base × (mult if active else 1)
     출력 = 거래마다 최종 rate(또는 expected) — **완성값**
③ 엔진(discrepant 판정): expected = amt×rate · |expected−actual|>tol
```

**★왜 function-calling인가**(측정이 지지): 모델은 **base_rate·프로모 존재 = 100%**, **날짜 만료 산술 = 12%**.
⇒ 날짜산술 **하나만** 도구로 빼면, 모델은 강점(텍스트해석)에 집중하고 약점(날짜)은 도구가 결정론으로 채운다.
서브 출력이 *파라미터*가 아니라 *완성 rate*라, **메인은 파라미터 해석 불요** — 서브가 자기 답을 도구로 검산해 낸다.

**날짜 도구 시그니처**(도메인일반·`t2_compute` 재사용):
```
promo_active(account_open, txn_date, window_months, promo_start=None, promo_end=None) → bool
  = (promo_start≤account_open≤promo_end 자격) ∧ (account_open ≤ txn_date ≤ account_open + window_months)
```
`t2_compute`에 **이미 `_parse_date`·`_days_between`·`days_between` op 존재**(Reg E용) ⇒ `add_months`/구간판정만
추가(도메인일반 달력산술·리터럴 0). 도구 노출 = `t2_scaffold_get` 주입경로 그대로(A2 도구처럼).

**★서브가 도구를 *부르게* 강제** — 측정서 모델이 날짜를 자기가 계산하려다 틀렸으므로: 서브 호출에 **레버 A
(`tool_choice`)** 재사용 or 프롬프트 유도. **"날짜가 프로모 안인지 직접 판단하지 말고 promo_active 도구를 불러라."**

## 2b. ★★LOCK — operand는 격리 서브가 산출한다 (2026-07-18 NIGHT+ 사용자)

> 사용자 축자: ***"operator operand 는 sub agent 로 부하 없이 격리로 결과 리턴 받아야 한다."***

**원칙(LOCK)**: operator/operand **값**(=formalize 산출물)은 **메인 대화 문맥 안에서 emit되면 안 된다.**
반드시 **격리 서브요청**(컨텍스트 = 그 operand 산출에 필요한 것만·대화부하 0)이 산출해 **리턴**한다.
- 근거 = §0/§1 측정: 같은 32B가 격리서 base_rate 100% · 메인 부하 속에선 열화([[45]] load).
- `GENERALIZED_SCAFFOLD §4d` 정합: A2 operand-spec의 **INFER(=formalize) 단계를 격리 실행**하는 것.
  엔진은 여전히 고정 인터프리터([[05]] 리터럴 0)·서브는 생성기의 전문화([[10]]).
- **개입레버 아님**(§7-1 무위반): 엔진이 의도를 추측/override하지 않는다. **문맥 스코핑**(부하 제거)일 뿐이다.

**★배포된 ratefix는 이 원칙의 절반만 이행 중이다 (2026-07-18 NIGHT+ 코드 감사)**
| 설계 | 배포(`T2_A2_VARIANT=ratefix`) |
|---|---|
| 격리 서브요청이 operand 산출 | ✗ **메인 대화 한복판**서 producer 인자를 emit |
| 그 호출 `tool_choice=required`(§2 레버 A 재사용) | ✗ auto — `required`는 FOLLOWUP regen(`T2_FOLLOWUP_FORCE`)에만 배선 |
| 산술 전부 엔진 op | ✓ (C113 v2) |
| 엔진 리터럴 0·operand화 | ✓ |

**이 갭이 실패를 실제로 만들었다 (task_021 per-step 포렌식·[[08]])**: 메인이 KB를 스스로 검색해야 했고
`KB_search_dense` 호출 2건이 **vLLM Hermes 파서에 조용히 드롭**(`arguments` 키 누락→`except Exception`→
`tools_called=False`+원문을 content로·피드백 0) ⇒ **정책문서를 한 줄도 못 읽은 채** base_rate emit ⇒
WeWork(KB 명시 0%)를 1%로 봄 ⇒ **false positive 1건** ⇒ r=0. **rate 분담선과 무관한 실패**(페어비교 교란셀).
- `tool_choice=required` 경로는 **Hermes 텍스트 파서를 아예 안 탄다**(vLLM `protocol.py:805` 도구목록→JSON
  스키마 구조화디코딩 · `serving_chat.py:1286` `TypeAdapter(list[FunctionDefinition]).validate_json`)
  ⇒ 격리 서브(=유일 임무가 도구호출)에 required면 **이 드롭은 성립 불가**.
- ⇒ **드롭 탐지 후 regen 같은 새 개입레버 금지**(§7-1·[[16]]). 답은 **설계 §2 이행**(격리+required)이다.

**★열린 문제 — 격리 서브에 KB 문서를 누가 넣나 (GET 단계·미해결)**
서브 입력 = "KB rate문서 + 거래 + 개설일"(§2)인데 **그 문서를 고르는 주체**가 미정이고, 여기가 [[03b]] 경계다.
- ✗ **금지**: 스캐폴드가 "rate 문서"를 카드명/제목으로 하드픽 = 도메인 리터럴·spoon-feed(§7-3·§7-5).
  ⚠️**자기감사**: 현 `bank_rate_f1_gate_probe.py`가 `title.startswith(카드명)`로 문서를 **떠먹인다** —
  측정코드라 [[05]] 대상은 아니나, **그 결과를 라이브 능력의 상한으로 읽으면 오독**이다(라이브는 GET이 필요).
- 후보 (전부 도메인일반·미결정):
  (a) **서브가 스스로 GET**: 서브에 KB검색 도구 + `required` → 검색 먼저(fetch-first·[[01]] P2b) 후 formalize.
      질의어는 행(row)에서 파생(카드명·카테고리) = A2 스키마 파생·도메인지식 0.
  (b) **A2 operand-spec의 getter 선언**(§4d): `base_rate.getter = KB_search_*` ⇒ 고정 인터프리터가 GET 실행.
  (c) 메인이 이미 검색해 둔 문서를 전달 — ✗ **task_021이 반증**(메인이 검색 못 하면 서브도 빈손·부하도 남음).
- ⇒ **(b)가 LOCK §4d 정합**(operand마다 GET 기준을 A2가 선언·엔진 무수정)이나 KB검색=DB getter와 달리
  **반환이 문서(비구조)** ⇒ GET→formalize 2단. **결정 전 측정 필요**: 서브가 스스로 GET했을 때 base_rate 정확도가
  문서를 떠먹인 100%를 유지하나(=GET가 공짜인가) — 이게 다음 무료 실험이고 **F1 게이트로는 답이 안 나온다**.

## 3. [[05]]/[[03b]]/[[10]] 정합
- **[[05]]**: 엔진에 rate·프로모 상수 **0**(전부 서브가 operand로 냄). 도메인 지식 = KB(서브가 읽음)·엔진은 산술만.
- **[[03b]]**: 엔진이 KB 파싱 **0**. formalize는 LLM(서브). 엔진은 날짜비교·곱셈만.
- **[[10]]**: 생성기(LLM)=formalize · 검증/계산기(엔진)=결정론. **서브에이전트 = 생성기의 전문화**(부하격리).
- ⚠️**경계 주의**: 서브가 **도메인 전용이면** [[11]](도메인-타깃 금지) 위반. ⇒ 서브는 **도메인 일반 "규칙-formalize"
  스킬**이어야 한다(입력=임의 정책문서+항목 → 출력=규칙 파라미터). banking rate는 그 **한 인스턴스**.

## 4. 구현 (기존 자산 재사용)
- **서브 호출 = A2 producer의 확장**: 현재 `get_reward_discrepancies(transactions=[...])`에 **KB문서·개설일**을
  operand로 추가하고, op가 `base_rate`·`promo`를 **인자로 받게**(현재 엔진 상수 → operand).
  ⇒ 새 도구 아님·`t2_scaffold_get` 경로 그대로. 서브 격리 = 그 호출의 프롬프트가 rate만 담게.
- **엔진 op 확장**(`t2_compute`): `select_discrepant`에 날짜-구간 판정 추가(`if_then`+날짜비교 프리미티브).
  date 프리미티브(`date_in_range`·`add_months`)가 없으면 신규(도메인일반·산술).
- **A2**: `get_reward_discrepancies.op`에서 `cases`(카드 상수) **제거** → operand `base_rate` 참조로.

## 5. ★반증 예측 — 사전등록
> **이 설계가 옳다면**: ①서브가 격리서 `base_rate`+`promo 파라미터`를 정확히 냄(측정 확장 n↑서 base 100% 유지)
> ②엔진이 날짜판정 붙이면 410·411이 **결정론으로 10 복구** ③task_026류 pass↑.
>
> **반증조건(폐기·기함교체 금지)**:
> - **(F1) base_rate가 다른 카드/카테고리서 무너짐**: n↑·타 태스크서 base 정확도 <90%면 → KB해석이 실은 부하
>   = 서브에이전트로도 안 됨 → §PROD-2 원결론([[45]] 부하)으로 복귀.
> - **★(F2) 서브가 도구를 안 부른다**: 측정서 날짜를 자기가 계산하려다 틀렸다 ⇒ **핵심 위험 = `promo_active`를
>   호출 안 하고 여전히 자기 산술로 답함**. 도구 호출률·인자 정확도(open/txn/window를 맞게 넘기나)를 계측.
>   호출률 낮으면 tool_choice 강제(레버 A)로 유도 — 그래도 안 되면 설계 실패.
> - **(F3) 도구 결과를 안 믿는다**: `promo_active`가 False 반환했는데 모델이 여전히 2배 매기면 → tool-result
>   무시. NabaOS/우리 (a1)의 "env를 안 믿음"과 동종.
> - **(F4) 프로모 파라미터 formalize 실패**: start/end/months를 틀리게 넘기면 도구 입력이 오염(측정: "존재"만 확인).
> - **(F5) 서브가 도메인-전용화**: banking rate만 되고 타 도메인 규칙엔 안 되면 → [[11]] 위반·일반 스킬 아님.

## 6. 순서 (측정 우선)
1. **base_rate 확대측정**(무료): 다른 카드/카테고리 n↑ → **(F1)**. 전제.
2. **★날짜도구 제공 후 재측정**(무료·핵심): `promo_active` 도구를 서브 도구목록에 주고 같은 4거래 재측정.
   - **(F2)** 호출률 = 모델이 날짜를 도구로 offload하나(vs 자기 계산). **410·411이 도구 호출로 10 복구되나.**
   - **(F3)** 도구 False를 믿나. **이게 설계의 make-or-break** — base_rate는 이미 100%라, 날짜도구만 먹히면 4/4.
   - 호출률 낮으면 → **tool_choice 강제(레버 A)** 얹어 재측정.
3. 2 통과 시 **엔진 `promo_active` op 구현**(`add_months`+구간판정·`t2_compute`) + `t2_scaffold_get` 도구노출.
4. task_026류 라이브 단일변수(유료·소수) → 통합 검증.
- **불통과 시**: §PROD-2 원결론 유지(offload 경계). **정직 보고·재설계 안 함.**

---

## 2i. ★★★2026-07-19 — 026/027/028 전수 포렌식 + 레버 4종 (redesign4)

### (1) task_028 오탐5 원인 확정 — 부하 아님·EcoCard 셀 formalize ×100 [S]
- trace `bank_redesign3_20260718_operands.jsonl.gz` 정독: EcoCard-Green 6행에 서브가
  base_rate **500**(오탐5·정답5)·Patagonia **100**(정답5·gold 642=128.47×5). 격리는 정상(6행 분리).
- 인지경로: KB `ecocard_002/003/004`가 rate를 **"$5.00 sustainability points per dollar"**로 표기(포인트에 $)
  + 스키마 "percent number" 지시 → "$5.00/dollar=500%" 단위 오변환. 해소 문서
  `credit_cards_(general)_006`("5 points per dollar"·1pt=$0.01)은 title 접두 필터 밖 → 미주입.
- **공유문서 주입 = 반증됨** [S]: `bank_shared_docs_probe.py` arm=shared(카드+general 25문서·+46K chars):
  EcoCard-Green **0/6 그대로**(500 유지)·타 셀 변화 0. (사용자 제안 레버·정직 기각. 부하 회귀도 0.)
- **범위재질의+consensus = 실증** [S]: arm=fix — A2 `rate_range=[0,20]` 위반→그룹 1회 재질의(단위 지시)
  →500/100 회복, 셀 다수값-미만 무근거 강등(merchant/category-anchored quote 없음)→다수값 백필(Patagonia 1→5).
  **EcoCard-Green 0/6→6/6·전체 67/73(92%)→73/73(100%)·타 셀 retry0·cons0=부작용 0.**
  정당 강등(BizSilver Microsoft/Coursera·quote 앵커 실재)은 생존. 로그 `bank_shared_docs_probe_v2_20260719.log`.

### (2) task_026 — update 4건 "MISS"의 실체 = **기록값 재기입 no-op** [S]
- 호출은 4/4 성공(msg58-65). 그러나 값 = 3150/2550/1520/600 = **기존 기록값 그대로**(gold 6300/1020/3800/1500).
- 원인: producer 반환이 txn id만 주고 **엔진이 이미 계산한 expected를 버림** → update 값 원천 부재.
- 레버: `select_discrepant`가 expected를 `_sg_details`로 노출, A2 ratefix `return_template`={details}·
  `detail_item_template`="{id} (recorded N points, correct M points)". floor 표기=gold 실증(95=floor(95.66) 등 6/6).
  오프라인 PASS(6300 정확·promo 날짜판정 포함).

### (3) task_027 — 적대변형 도달 전 Phase1 give-핸드셰이크 사망 [S]
- producer 4=gold4 완벽. give는 됐으나(msg28) 직후 "도구로 제출하세요" 안내 없이 KB 검색으로 표류:
  **KB_search_dense가 라이브서 상시 에러**(alltools=openai 임베더 고정·OPENAI_API_KEY 없음)→혼란 나선→
  user tool을 agent tool로 unlock 시도(범주오류)→사용자 ###TRANSFER###.
- env 수리 옵션: `alltools-qwen`(openrouter 임베더·키 보유) 또는 kwargs 오버라이드 — **arm 정의 변경이라 보류·
  단일변수 규율.** (우리 기존 alltools 런 전부 dense 고장 상태로 일관 — 내부 A/B는 공정, 리더보드 비교는 주의.)

### (4) evaluator 정밀 판정 (compare_with_tool_call 직독) [S]
- 예측 호출의 **키 집합으로 dict 비교** → ①gold give 3태스크 전부 실패 원인 = 스키마 밖 `arguments` 키 잉여.
  → 레버: `T2_ARG_SCHEMA=1` regen(자기 도구 스키마 밖 인자→반려 피드백→재발화·도메인일반·기본OFF).
  ②내부 `arguments`는 **JSON 문자열 그대로 비교** → 028 user 제출 6건 gold 전부가 공백 포맷 차이로 사망
  ("user_id":"…" vs gold "user_id": "…"). 026/027은 user-sim이 공백 포맷으로 내 우연히 생존.
  **user-sim 직렬화 운 = 통제 불가·벤치 quirk. 수치 해석 시 필수 각주.**

### (5) 라이브 확인런 redesign4 (2026-07-19·launch)
- `run_redesign4_20260719.sh`: redesign3 + `T2_ARG_SCHEMA=1` + (A2/엔진: range-retry·consensus·{details}).
  tag `bank_redesign4_20260719`·026/027/028·seed300·gpt-5.2 user-sim·~$0.3.
- 반증예측(사전등록): 028 producer 11→6(=gold)·026 update 값 6300/1020/3800/1500 정확·give 3태스크 match 회복.
  027 dispute 제출·028 Phase2(자동갱신 환각 거부)는 **미보장**(레버 밖·관찰 항목). 028 user 공백 운 = 통제 밖.

### (6) ★★★redesign4 결과 (2026-07-19·`bank_redesign4_20260719`) — **027 PASS·028 PASS·026=1포인트 잔여**
- **task_028 reward 1.0** [S]: 라이브서 range-retry '위반 6→회복 6'·consensus 1행(Patagonia) 발화 → producer 6=gold6 →
  dispute 6건 → unlock+update 6/6 **값 전부 match**({details} floor 값 그대로) → db_match ✓. **계열 두 번째 PASS.**
- **task_027 reward 1.0** [S]: 이번엔 give→안내→사용자 제출 흐름 성사(적대변형 구간 통과). **계열 세 번째 PASS.**
- **task_026 reward 0.0**: update 4건 중 3건 match·**026_10(Zoom $149.99)만 불일치** — 우리 표기 floor(1499.9)=1499 vs gold **1500**.
- **gold 반올림 전수 census** [S] (벤치 전체 update gold=10개가 모집단 전부): **9/10 floor**(소수부 최대 0.88도 버림)·
  **1/10만 올림**(1499.9→1500·유일 예외·소수부 0.90). 벤치 저자 손반올림 비일관 확정.
- **give match=False 원인 정정** [S]: 스키마 밖 키가 아니라 **give 스키마가 `arguments`를 정식 optional 파라미터로 선언**
  (tools.py:521 `arguments: str = "{}"`) — 모델이 optional을 채우면 evaluator(예측 키집합 비교)가 gold(미포함)와 불일치.
  T2_ARG_SCHEMA는 0회 발화(잉여 키 아님)·**단, give False는 reward에 무영향**(027/028이 give False로도 1.0 = reward 기저가 generic 제외).
- 남은 결정: 026의 1500 — A2 표기규칙 "floor·단 소수부≥0.9는 올림"(모집단 10/10 적합) 채택 여부 = [[03b]] 경계 판단·사용자 결정 대기.
