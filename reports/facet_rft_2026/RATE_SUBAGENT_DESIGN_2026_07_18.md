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
