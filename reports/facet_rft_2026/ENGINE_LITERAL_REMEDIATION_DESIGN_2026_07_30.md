# 엔진 도메인-리터럴 전수 감사 + 이관 설계 (X6-h) — 2026-07-30

> 무료 트랙. handoff §6-⑤(엔진 위반 2건 이관)의 선행 감사 + 처방 설계.
> **상태 = ⚠️리뷰 조건부 반려(§8) → 개정 중.** U0만 단독 진행 승인·구현 완료.
> U1·U2·U3·U4·U5는 §8 B1~B3 반영 후 **재리뷰 대상**.
> 도구 = `scripts/distill/tau2/x6h_engine_literal_audit.py` · 데이터 = `tau2_domain_toolnames.json`
> 원시 결과 = 이 문서 §2(건별 전수). 상위 = `X_FREE_TRACK_RESULTS_2026_07_30.md` §13d/§13e.

## §0-. ⚠️읽는 순서 — v1 반려 → v2 재작성 완료

| 섹션 | 상태 |
|---|---|
| §0 한 문단 | ❌**무효**(v1 수치·"근본원인 1개") — 이력 보존용·인용 금지 |
| §1 감사 방법 | 유효 + §2 말미의 **도구 한계 3건**을 함께 읽을 것 |
| **§2 건별 판정 v2** | ✅**유효**(16 사이트/4축·폐포 14파일) |
| **§3 정정 이력** | ✅유효(2단 정정: §13d 과소 6 → 내 v1 과소 9) |
| **§4 처방 v2** | ✅유효(7 수정단위·B2/B3 반영·Q1 재계산) |
| §5·§6 | ⚠️**v1 잔존·§4 검증계획 v2가 대체**. U1/U2 서술은 §4 U1'/U2'로 교체됨 |
| §7 미결 | 유효 |
| **§8 리뷰 결과** | ✅유효 — 반려 근거·차단 3건·교정 수치 |

### 반려 사유 요약

§8 리뷰 B1이 확인한 것: 감사 도구의 `ENGINE_FILES` 8개를 **내가 손으로 적었고**, 그래서
`t2_resolve.py`(=`t2_gate_patch.py`가 :547·:3885·:3911·:4035에서 import하는 **라이브 엔진**)를
비롯한 **6개 파일이 스코프 밖**에 있었다. **import 폐포로 재산출**하니 엔진은 **14 파일**이고
위반은 **7 → 16 사이트**(도메인-명사 7 + 도메인-특화 도구명 9)로 늘었다.

**교정된 수치(§8-C)**:

| | 초판(손목록 8파일) | 교정(폐포 14파일) |
|---|---|---|
| 도메인-명사 리터럴 | 1 | **7** |
| 도메인-특화 도구명 | 6 (U0 후 5) | **9** |
| 도메인-판별 정규식 | 21 | 23 |
| clean 파일 | 5/8 | **6/14** |

⇒ **§0·§2·§3의 "전수"·"7 사이트"·"근본원인 1개" 주장은 전부 무효**다. 아래 원문은 이력 보존용이며
**인용 금지**. 유효한 것은 §8-C의 교정 수치와 §8-D의 개정 처방이다.

## §0. 한 문단 (⚠️§0- 참조·수치 무효)

`X_FREE_TRACK_RESULTS` §13d가 grep으로 판정한 **"엔진 실제 코드 위반 2건"은 양방향으로 틀렸다**.
② `t2_prekb_patch.py:627`은 **selftest 픽스처**여서 위반이 아니고(엔진 동작 아님), 동시에 §13d가
**놓친 실제 위반이 6곳 더** 있다. AST 기반 전수 감사로 확정한 결과는 **7 사이트 / 5 수정단위**이고,
**근본원인은 1개**다: banking의 discoverable-dispatcher 어휘(`call_`/`unlock_`/`list_`/`give_`)가
**A2에 이미 선언돼 있는데도** 엔진 6곳에 중복 하드코딩됐다. 처방은 새 A2 키 발명이 아니라 **기존
선언 배선**이며, 그중 1건(`get_current_time`)은 5도메인 122도구에서 판정 변화 0인 **죽은 중복**이라
삭제만으로 끝난다. 단 2건은 무료가 아니다 — U1b는 C211 리뷰가 명시적으로 동결한 공유 술어를
건드리고, U3는 프롬프트 변경이라 행동이 바뀔 수 있어 **측정 게이트**가 붙는다.

### [[05]] 3질문 ([[17]] 상설 의무)

| # | 질문 | 답 | 근거 |
|---|---|---|---|
| **Q1** | scaffold **또는 A2**의 도메인-특화를 *순증*시키나? | **No (순감)** | 새 A2 키 **0개**. 이관 대상 6곳 전부 **이미 존재하는** 선언(`eplan.dispatch_tool`/`unlock_tool`/`list_tool`/`scaffold_get_tools[].follow_up.tool`)을 읽는다. 엔진 리터럴 7 → 0. A2 크기 불변. |
| **Q2** | 모델의 *유동적 판단*을 결정론에 동결하나? | **No** | 판정 로직·술어 의미·게이트 발화 조건 전부 불변. 바뀌는 것은 **같은 문자열을 어디서 읽는가**뿐. 예외=U3(프롬프트 산문)는 모델 입력이 바뀌므로 §5 측정 게이트. |
| **Q3** | scaffold가 모델 대신 *도메인 행동을 수행*하나? | **No** | 새 수행 0. 도구 호출·fetch·값 선택 추가 없음. 순수 이름-해소 경로 변경. |

⇒ Q1~Q3 전부 No이므로 **기본 허용**. 단 U1b·U3는 별도 게이트(§5).

## §1. 감사 방법 — 왜 이 판정을 믿을 수 있나 (그리고 어디까지만)

grep이 못 하는 3가지를 AST로 분리했다: **(a) 주석·docstring** (b) **selftest 픽스처**
(`if __name__ == "__main__"` 블록) (c) **실행 코드**. 추가로 **포맷 플레이스홀더 제외** —
`"{tool} was denied"`처럼 `{...}`로 A2 값을 주입받는 자리는 파라미터화의 *정답*이므로 매칭 전에
제거한다. 남은 *맨* 이름만이 하드코딩이다.

**비순환성**: 판정 어휘를 감사자가 고르면 "내가 고른 단어만 위반"이 되므로, tau2 도메인 정의
(`src/tau2/domains/<d>/tools.py`의 public 메서드)에서 **기계 수확한 권위 namespace**를 쓴다
— 5도메인·public 도구명 122개. 공통/특화 판별도 데이터가 한다:

- 도구명이 **5/5 도메인**에 존재 → framework-common (엔진 참조 허용) — 실측 **1개뿐**:
  `transfer_to_human_agents`
- 도구명이 **일부** 도메인에만 → domain-specific (위반 후보)

**정규식 가지 분해**: 선언형 패턴은 **범용 가지 하나가 전체를 도메인-일반으로 위장**시킨다.
실측 — `(^log_|…|discoverable|transfer_to_human|^give_|^unlock_|get_current_time)`은
`transfer_to_human`(5/5)이 있어 통째로는 "일반"으로 통과하지만, 가지 `discoverable`·`^give_`·
`^unlock_`은 banking 전용이다. ⇒ 최상위 `|`로 쪼개 **가지 단위**로 판정한다.

### 도구의 한계 (정직 기록 — "0 증명" 아님)

1. **문자열 리터럴만** 본다. 런타임 조립 이름(`f"{a}_{b}"`)·dict 키·식별자에 숨은 도메인 결합은
   못 잡는다.
2. **어휘 조각 검사는 폐기했다.** 도구명을 `_`로 쪼개 단일-도메인 조각을 찾는 방식을 시도했더니
   **152건 거의 전부 오탐**이었다(`customer`·`item`·`available`·`report`가 우연히 한 도메인
   도구명에만 등장). 신호 없음이라 정규식-판별(함수적 검사)로 교체했다.
3. **오탐 잔존 1건**: `t2_eplan_patch.py:92` `^[A-Za-z][A-Za-z\-']*$`는 enum-단어 매처인데
   밑줄 없는 도구명(`calculate`·airline/retail)을 우연히 매치한다. 도구명과 무관.
4. 판정 namespace가 **도구명뿐**이다. DB 필드명·enum 값은 미포함 → 2차 감사 후보.

## §2. 건별 판정 v2 — **16 사이트 / 4축** (폐포 14파일 · [[08]] per-case)

> v1의 "7 사이트 / 근본원인 1개"는 스코프 결함으로 무효(§8-A). 아래가 유효 판정이다.
> 축 분류는 **처방이 다르기 때문에** 나눈다 — 같은 "도메인 리터럴"이라도 에이전트 경로와
> 계측 경로의 해악과 해법이 다르다.

### 축 A — dispatcher 어휘 (에이전트 경로 · 9 사이트) ★본체

banking의 discoverable-dispatcher 이름이 A2 선언이 **이미 있는데도** 엔진 9곳에 중복 표현됐다.

| # | 위치 | 리터럴 | 형태 | A2 출처(**존재**) |
|---|---|---|---|---|
| A1 | `t2_gate_patch.py:3776` | `unlock_discoverable_agent_tool` | **기본값 fallback** | `eplan.unlock_tool` |
| A2 | `t2_gate_patch.py:3778` | `list_discoverable_agent_tools` | 기본값 fallback | `eplan.list_tool` |
| A3 | `t2_prekb_patch.py:134` | `unlock_discoverable_agent_tool` | 하드코딩 비교(fallback 없음) | `eplan.unlock_tool` |
| A4 | `t2_prekb_patch.py:136` | `call_discoverable_agent_tool` | 하드코딩 비교 | `eplan.dispatch_tool` |
| A5 | `t2_prekb_patch.py:575` | `give_discoverable_user_tool` | **모델-대면 산문** · 자기모순(`:178`이 "A2서 읽는다" 명시) | `_declared_give_tool(a2)` |
| A6 | **`t2_resolve.py:90`** | `open_bank_account_4821` · `unlock_/call_/list_discoverable_…` · `'open bank account'`·`'close bank account'`·`'apply savings interest correction'` | **모델-대면 산문 · 최악** — 실제 suffixed 도구명 + banking 액션 어휘를 직접 먹인다 | `eplan.{dispatch,unlock,list}_tool` + (액션 예시는 §4-Q1 참조) |
| A7 | **`t2_resolve.py:451`** | `call_discoverable_agent_tool` | 기본값 fallback | `eplan.dispatch_tool` |
| A8 | **`t2_resolve.py:511`** | `call_discoverable_agent_tool` | 하드코딩 비교 | `eplan.dispatch_tool` |
| A9 | **`t2_scaffold_get.py:1171`** | `call_discoverable_agent_tool`·`unlock_discoverable_agent_tool` | 모델-대면 산문(READ-FIRST 오류) | 동일 |
| (정규식) | `t2_gate_patch.py:1653`·`4534` | `discoverable`·`^give_`·`^unlock_`·`^(give\|call)_` | 명명 관행 패턴 | 동일 |

**신규 4건(A6~A9)이 v1에 없었고, 그중 3건이 모델-대면 산문**이다. A6는 V5(페르소나 명사)보다
심각하다 — 명사가 아니라 **실제 도구 인스턴스 이름**을 프롬프트에 박았다.

### 축 B — 도메인 명사 (에이전트 경로 산문 · 1 사이트)

| # | 위치 | 리터럴 | 판정 |
|---|---|---|---|
| B1 | `t2_gate_patch.py:2286` | `"You are a precise banking assistant."` | REF_ISO 서브콜 페르소나. 프롬프트 변경이므로 **측정 게이트** 필요 |

### 축 C — **계측층 도메인 잠금** (에이전트 경로 아님 · 4 사이트 · 다른 처방)

`t2_compliance.py`는 **사후-검사기**("eval-후크 공용 모듈"·`t2_run_gated.py:369`가 평가 *직후*
호출)이고 도메인은 `domain=a.domain`으로 **주입**된다. 따라서 [[05]] "엔진 리터럴" 위반과 **성격이
다르다** — 에이전트 행동에 영향이 없다.

| 위치 | 리터럴 | 판정 |
|---|---|---|
| `:180`·`:195` | `domain="retail"` 기본 파라미터 | 호출부가 항상 덮으므로 무해 |
| `:182` | **`if domain != "retail": return None`** | ⚠️[[05]] 체크리스트 4항이 금지한 `if domain` 분기 형태. 단 **안전 열화**(주석: "airline 등은 spec/검사기 이식 후") |
| `:188` | `"tau2.domains.retail.environment"` | 위 분기 내부 |

**★그런데 무해하지 않은 귀결이 하나 있다(신규 발견)**: `load_order_owner`는 **G3(타-유저) 검사용**
order→owner 맵이고, retail이 아니면 `None`을 반환해 **G3 order-resolve가 생략된다**(모듈 docstring이
그걸 "상한"이라 명시). ⇒ **우리가 인용하는 banking compliance 수치는 G3에 관해 상한(upper bound)이며,
order-소유권 위반을 탐지하지 못한다.** 이건 [[05]] 문제가 아니라 **측정 caveat**이고, compliance
수치를 쓰는 모든 곳(논문·특허·원장)에 붙어야 한다.

### 축 D — 드라이버 CLI 기본값 (무해 · 2 사이트)

`t2_run_gated.py:69` `--domain default="retail"` · `:71` help 문구의 `banking_knowledge`.
실제 호출은 항상 `--domain`을 지정한다(Y1 커맨드도 `banking_knowledge`). **위반 아님·기록만.**

### 관용 / 프레임워크 / selftest (v1과 동일)

- `_\d+$`·`_\d{3,4}$` 계열 정규식 23건 = banking 숫자-접미사 **명명 관행**. 접미사 없는 도메인엔
  no-op. 수용(단 이름이 정당하게 숫자로 끝나는 도메인은 오절단 = 잠재 결합).
- `transfer_to_human_agents` = 5/5 도메인 → 허용.
- selftest 픽스처 53건 = 비위반(§13d ②가 여기 속함).

### 이 감사가 **여전히** 못 잡는 것 (리뷰 지적 반영)

1. **키워드-인자 기본값**: `t2_resolve.py:376` `parse_records(key_field="transaction_id", …)` —
   banking **필드명** 기본값이고 docstring은 "A2가 선언"이라 주장한다. 내 도구는 도구명 namespace만
   보므로 **필드명을 못 잡는다**. ⇒ **DB 필드·enum namespace 2차 감사 필요**(v1 한계 4가 그대로 실현).
2. **dict 키 결합**: `t2_resolve.py:318`·`:340` `a.get("discoverable_tool_name")` — 문자열 리터럴이나
   **도구명이 아니라 인자 키**라 namespace에 없다. 대응 A2 키(`eplan.dispatch_name_key`)는 **이미 존재**.
3. 오탐 1: `t2_eplan_patch.py:92` enum-단어 매처(`calculate` 우연 매치).

## §3. 정정 이력 — 2단 (§13d 과소 6 → **내 v1 과소 9**)

| 주장 | 실측 | 성격 |
|---|---|---|
| §13d "실제 코드 위반 **2건**" | — | grep 판정 |
| §13d ② `prekb:627` | **selftest 픽스처 = 비위반** | 과대 1 |
| v1(본 문서) "**7 사이트** / 근본원인 1개 / 8파일 중 5 clean" | **16 사이트 / 4축 / 폐포 14파일 중 6 clean** | **내 과소 9** |
| v1 "전수 감사" | 손으로 적은 8파일 = **폐포의 57%** | 스코프 결함 |

**교훈 2단**:
1. (v1) **grep으로 [[05]] 준수를 주장하지 말 것** — 주석·픽스처·플레이스홀더·정규식 가지를 구분 못 함.
2. (v2·이번) **감사 대상 목록을 손으로 적지 말 것** — 그건 proxy이고, 방법을 고쳐도 스코프가 새면
   같은 실패다. 목록은 **라이브 진입점에서 기계 산출**(import 폐포)해야 한다. 그리고 **폐포 도구
   자체도 검증해야 한다**(초판 정규식이 개행을 먹어 폐포를 3개로 과소 산출 → 정규식-ground-truth
   대조로 발견).
3. ⇒ [[05]] 체크리스트 4항의 검증 수단을 **`x6h_engine_literal_audit.py`(폐포판)**로 승격하고,
   "전수"라는 말은 **폐포 산출 + 도구 한계 명시**를 동반할 때만 쓴다.

## §4. 처방 v2 — 7 수정단위 (B2·B3 반영)

### [[05]] Q1 재계산 (리뷰 §8-D 요구)

v1 Q1 답 "새 A2 키 0 (순감)"은 5 수정단위 기준이라 **무효**. 재계산:

| 대상 | 기존 A2 키로 덮이나 | 순증? |
|---|---|---|
| A1~A5·A7~A9 (dispatcher 이름) | ✅ `eplan.{dispatch,unlock,list}_tool` + `scaffold_get_tools[].follow_up.tool` | **0** |
| `discoverable_tool_name` (arg-key) | ✅ `eplan.dispatch_name_key` | **0** |
| A6의 **액션 예시 어휘**(`'open bank account'` 등) | ❌ 대응 키 없음 | **선택에 따라 갈림** — ⓐ삭제(=순증 0·프롬프트 약화) ⓑA2 `kb_query_examples` 신설(=**순증 1키**) |
| `key_field="transaction_id"` (필드 기본값) | ❌ 미확인 — A2 `identifying_arg_types`가 덮는지 조사 필요 | **미정** |

⇒ **Q1 답 = "대부분 0이나 A6 액션 예시와 필드 기본값에서 순증 가능"**. ⓐ/ⓑ 선택은 U3'와 묶어
**프로브 1회로 측정 후 결정**(리뷰 §8-D 우선순위 역전 지적 반영).

### 수정단위

| # | 대상 | 처방 | 상태 |
|---|---|---|---|
| **U0** | `get_current_time` 죽은 가지 | 삭제 | ✅**완료**(122도구 판정 diff 0 · 8배터리 PASS). 문구 하향: **"측정-불변, 논리-불변 아님"**(`RDP.match` 앵커 vs `PRC.search` 비앵커라 122도구 namespace 위에서만 동치) |
| **U1'** | 공유 실효-write 술어 어휘 (A1~A2 정규식) | **B2 반영**: 모듈 전역 폐기 → `_A2_PROC_BY_DOMAIN[domain]` 키잉 + `_is_effective_write(name, a2=None)` **명시 전달**. "시그니처 불변" 포기(호출부 6곳 수정이 전역 상태보다 싸고 안전) | ⚠C211 동결 해제 **재승인 필수** |
| **U2'** | banking 기본값 (A1·A2·A7) | **B3 반영**: `_safe` truthy 필터가 아니라 **"세 키 중 하나라도 미선언이면 DD 레버 자체를 skip"**. 근거는 `_unlock`이 `:3786` 피드백 산문에 보간되므로 기본값 제거 시 `"1) None(agent_tool_name=…)"`이 모델에 전달됨 | 재리뷰 |
| **U3'** | B1 페르소나 + **A6 산문**(액션 예시·도구 인스턴스명) | 리뷰 §8-D대로 **한 단위로 묶어 프로브 1회**. Q1 ⓐ/ⓑ 선택을 이 프로브가 결정 | 재리뷰 · 측정 게이트 |
| **U4** | `_effective_fams` (A3·A4) | `_effective_fams(tc, a2)`. 호출부 `:425`·`:488` 모두 a2 스코프 확보(리뷰 확인) | 재리뷰 |
| **U5** | give 도구명 (A5) | `_declared_give_tool(a2)` 재사용(`:178` 헬퍼·`:378` 선례) | 재리뷰 |
| **U6**(신규) | A8·A9 + `discoverable_tool_name` arg-key | `eplan.dispatch_tool`·`dispatch_name_key`에서 읽기 | 재리뷰 |
| **U7**(신규) | 축 C 계측 caveat | 코드 수정 **아님** — **banking compliance 수치에 "G3 order-resolve 미적용(상한)" caveat를 원장·논문·특허에 부착**. 검사기 이식은 별건 | 문서 조치 |

### 검증 계획 v2 (리뷰 §8-D 반영)

1. 폐포 재감사 → 축 A·B **10 → 0** (축 C·D는 대상 아님·관용 23·오탐 1 잔존 예상).
2. `_is_effective_write` before/after diff 0 — **각 도메인을 자기 A2로 로드**해 돌린다(v1은 A2
   지정이 빠져 banking A2로 전 도메인 도는 것도 통과하는 표현이었다).
3. 기존 배터리 전부 + **airline A2-swap 스모크**(B2 수정으로 이제 유효 — 전역 누출이 없으므로).
4. **DB 필드·enum namespace 2차 감사** 착수(§2 한계 1).

## §5. 위험 · 측정 게이트 (⚠️v1 잔존 — U1·U2 항은 §4 U1'·U2'로 대체됨)

| 단위 | 행동 변화 위험 | 게이트 |
|---|---|---|
| U0 | **없음**(실측: 122도구 판정변화 0) | selftest + 회귀 배터리 |
| U2·U4·U5 | **없음(구성상)** — banking A2가 동일 문자열을 공급하므로 해소 결과가 바이트 동일 | ①해소값 동일성 assert(신규 테스트: A2 유래 이름 == 구 리터럴) ②기존 회귀 배터리 전부 ③오프라인 궤적 재생 1건 |
| U1 | **중간** — 공유 술어 변경이 6 호출부에 전파. C211이 동결한 대상 | **사용자 재승인** + 상기 ①②③ + `_is_effective_write`를 5도메인 122도구 전수에 돌려 **before/after 판정 diff = 0** 확인(무료·기계적) |
| U3 | **미지** — 프롬프트 변경 | **무료 격리 프로브**: REF_ISO 서브콜을 기존 궤적 입력으로 재생해 base("banking assistant") vs treat("assistant") 선택 일치율 측정. 불일치 0 또는 개선이면 GO; 악화면 A2 `ref_iso[].persona`로 후퇴(그때는 A2 순증을 측정으로 정당화) |

**공통 금지**: U1~U5를 한 커밋에 섞지 않는다. U0·U2·U4·U5(무해군) → U1(재승인 후) → U3(프로브 후)
순서. 섞으면 회귀 발생 시 귀속이 불가능하다(C211 F6b 교훈).

## §6. 검증 계획 (⚠️v1 잔존 — §4 말미 "검증 계획 v2"가 정본)

1. `x6h_engine_literal_audit.py` 재실행 → **확정 위반 7 → 0** (관용 18·오탐 1은 잔존 예상).
2. 신규 테스트 `test_x6h_literal_transfer.py`: (a) A2 유래 이름이 구 리터럴과 동일 (b) A2 미선언
   가짜 도메인에서 no-op (c) `_is_effective_write` before/after diff 0 (122도구 전수).
3. 기존 배터리 전부(`test_c211_day7rx`·`test_c212_day8rx`·회귀 16종) 로컬+리모트 PASS.
4. **airline A2-swap 스모크**: airline A2로 엔진 로드 시 banking 어휘 누수 0 — [[05]] 체크리스트
   4항("airline이 A2-swap만으로 unchanged 작동")의 실제 이행. 현재까지 이 항목은 **한 번도 기계적
   으로 검증된 적 없다**(주장만 있었다).

## §7. 미결 · 다음

- **2차 감사 후보**: DB 필드명·enum 값 namespace(현 감사는 도구명만). §2 한계 4.
- **§13e 설계 TODO 연결**: 이 감사가 §13e의 ①callback 인터페이스 목록 ②공용 primitive 목록에
  입력을 준다 — 근본원인이 "dispatcher 어휘 4개"로 좁혀졌으므로, **dispatcher 추상(unlock→call→
  list→give)이 첫 callback 인터페이스 후보**다. EXT 10키 재판정은 별건.
- **[[05]] 검증 수단 승격**: 체크리스트 4항의 `grep`을 이 도구로 교체(메모리 갱신 대상).

## 부록 — 재현

```bash
cd scripts/distill/tau2 && PYTHONIOENCODING=utf-8 py -3 x6h_engine_literal_audit.py --json out.json
```
도구명 namespace 재수확(도메인 추가 시): 리모트 `tau2-bench/src/tau2/domains/<d>/tools.py`의
`^    def [a-z]` public 메서드 → `tau2_domain_toolnames.json`.

---

## §8. 리뷰 결과 (2026-07-30) — **조건부 반려** · 차단 결함 3건

리뷰어가 설계서를 신뢰하지 않고 **코드 대조로 검증**했다. 인용 줄번호·A2 선언 존재·`_declared_give_tool`
자기모순·C211 동결 주석(축자 일치)·`_effective_fams` 호출부 a2 스코프·`_PROCEDURAL_RE` 사용처 2곳은
**전부 실측대로 확인**됐다. 그 위에서 **차단 3건**이 나왔다. 나는 세 건을 모두 직접 재대조해 **전면
수용**한다.

### §8-A. B1 (최중요) — 감사 스코프가 엔진을 다 덮지 않았다

`x6h_engine_literal_audit.py`의 `ENGINE_FILES` 8개를 **내가 손으로 적었다.** 그래서
`t2_resolve.py`가 빠졌는데, 그 모듈은 `t2_gate_patch.py`가 런타임 4곳(`:547`·`:3885`·`:3911`·`:4035`)
에서 import하는 **라이브 엔진**이다. 확인된 리터럴:

| 위치 | 리터럴 | 성격 |
|---|---|---|
| `t2_resolve.py:90` `DISCOVERY_REQUIRED_FB` | `open_bank_account_4821` · `unlock_/call_/list_discoverable_agent_tool(s)` · `'open bank account'`·`'close bank account'`·`'apply savings interest correction'` | **V5보다 심각** — 모델 입력 산문에 **실제 suffixed 도구명과 banking 액션 어휘**를 직접 먹인다 |
| `:451` | `spec.get("dispatch_tool", "call_discoverable_agent_tool")` | V3/V4와 **동일 클래스** |
| `:511` | `!= "call_discoverable_agent_tool"` | 하드코딩 비교 |
| `:223` | `discoverable_tool_name=` | 대응 A2 키(`dispatch_name_key`)가 이미 있는데 하드코딩 |
| `:376` | `parse_records(key_field="transaction_id", …)` | **banking 필드 기본값** (docstring은 "A2가 선언"이라 주장) |
| `:318`·`:340` | `a.get("discoverable_tool_name")` | dict 키 결합 |

**이건 §3이 스스로 쓴 교훈의 재발이다** — "체크리스트가 proxy면 만족시켜도 목표는 위반된다".
실패 위치가 **방법**(grep→AST)에서 **스코프**(손으로 적은 파일 목록)로 옮겨간 것에 불과했다.
§13d를 "과소 6"이라 정정한 문서가 **같은 이유로 최소 5를 놓쳤다.**

**수정 완료**: `ENGINE_FILES` 하드코딩을 폐기하고 **라이브 드라이버(`t2_run_gated.py`)에서 시작한
import 폐포**로 산출하게 했다(`discover_engine_files()`). 손목록을 늘리는 대신 폐포가 자동 편입한다.

> ⚠️그 구현에서도 **버그 1건을 자기발견**했다: 초판 정규식 `[\w,\s]+`의 `\s`가 **개행을 먹어**
> 여러 import를 한 덩어리로 삼켜 폐포가 **3개**로 과소 산출됐다. 한 줄 제한(`[^\n]+`)으로 교정.
> ⇒ 폐포 도구도 **검증 없이 믿으면 안 된다**는 같은 교훈의 3번째 사례.

### §8-B. B2 — U1의 모듈-전역 `_A2_PROCEDURAL`은 도메인을 누출한다 (수용)

`_domain_a2()`는 **도메인별 dict 캐시**인데 설계는 파생값을 **단일 전역 frozenset**에 넣으려 했다.
두 가지가 깨진다: ①**순서 의존** — `_is_effective_write`가 해당 도메인의 `_domain_a2()`보다 먼저
불리면 집합이 비어 `give_…`/`unlock_…`이 write로 판정되고, 이는 `:4531`이 **회귀 조건으로 못박은**
`_is_effective_write("give_…")=False`가 정확히 무너지는 시나리오다. ②**교차-도메인 누출** — 한
프로세스에서 도메인이 바뀌면 전역이 last-wins/합집합이 되고, 하필 그게 **§6-4 airline A2-swap
스모크를 무효화**한다(banking 어휘가 전역에 남은 채 airline을 판정하면 스모크는 통과하며 누출은 실재).

**개정 처방**: 전역 폐기 → `_A2_PROC_BY_DOMAIN[domain]` 키잉 + `_is_effective_write(name, a2=None)`
**명시 전달**. U1의 세일즈 포인트였던 "시그니처 불변 = 국소 변경"은 **포기한다** — 호출부 6곳 수정이
전역 상태보다 싸고 안전하다(6곳 모두 a2가 근처에 있는 오케스트레이터 래퍼 안).

### §8-C. B3 — U2는 처방과 근거가 어긋난다 (수용)

U2 근거는 "미선언 도메인은 그 레버가 **비활성**(안전측)"인데 처방은 `_safe` 집합만 truthy 필터였다.
실제로 `_unlock`은 `_safe`에만 쓰이지 않고 **모델에게 나가는 피드백 산문에 보간**된다
(`t2_gate_patch.py:3786` `"  1) %s(%s=\"%s\")\n"`). 기본값을 없애면 레버가 꺼지는 게 아니라
**`"1) None(agent_tool_name=…)"`이 모델에게 전달**된다. 게이트 진입 조건은 `dispatch_tool` 하나뿐이라
`unlock_tool`/`list_tool`은 보호받지 못한다.

**개정 처방**: "세 키 중 **하나라도 미선언이면 DD 레버 자체를 skip**". banking은 3개 다 있으므로
행동 동일 주장은 유지된다.

### §8-D. 비차단 지적 — 전부 수용

| 지적 | 조치 |
|---|---|
| U0 "위험 없음"은 과주장 (`RDP.match`=앵커 vs `PRC.search`=비앵커라 **논리적 동치 아님**·122도구 namespace 위에서만 동치) | 문구를 **"측정-불변, 논리-불변 아님"**으로 하향. 그 조건에서 **U0 진행·구현 완료** |
| §5 U1 diff 게이트에 **A2 지정 누락** — "5도메인 122도구 before/after diff 0"은 **각 도메인을 자기 A2로 로드**해야 의미 있음(현 표현은 banking A2로 전 도메인 도는 것도 통과) | 게이트 문구에 "도메인별 자기 A2 로드" 명시 |
| **우선순위 역전** — U3(페르소나 명사)에 측정 게이트를 붙이면서 실제 도구명 `open_bank_account_4821`을 먹이는 `t2_resolve.py:90`은 감사 밖 | U3를 그 산문 수정과 **같은 단위로 묶어 프로브 1회**로 처리 |
| [[05]] Q1 재계산 필요 — "새 A2 키 0"은 5 수정단위 기준. `discoverable_tool_name`은 기존 `dispatch_name_key`로 덮이나, `DISCOVERY_REQUIRED_FB`의 **액션 예시 어휘**는 삭제/A2참조 선택에 따라 순증 여부가 갈림 | §9 재작성 시 Q1 **재계산**(현 §0 Q1 표는 무효) |

### §8-E. 교정된 감사 결과 (폐포 14파일)

`t2_run_gated.py` import 폐포 = **14 파일**(내 손목록 8 · 리뷰어 추정 11보다도 큼 — 기계 산출이
양쪽 손목록을 다 이긴다).

| 파일 | live 위반후보 |
|---|---|
| `t2_gate_patch.py` | 16 |
| `t2_eplan_patch.py` | 7 |
| `t2_prekb_patch.py` | 4 |
| **`t2_compliance.py`** (신규) | **4** (retail 도메인 명사) |
| **`t2_resolve.py`** (신규·B1) | **4** |
| **`t2_run_gated.py`** (신규) | **2** |
| **`t2_scaffold_get.py`** (신규) | **2** |
| clean 6 | `gate_interpreter` · `t2_agent_maxprompt_patch` · `t2_agent_rules_patch` · `t2_compute` · `t2_formalize_exec` · `t2_resolve_patch` |

**총계**: 도메인-명사 **7**(초판 1) · 도메인-특화 도구명 **9**(초판 6·U0 후 5) · 정규식 23 ·
selftest 53.

⇒ **근본원인도 1개가 아니다(§2 말미 무효).** 최소 **3축**: ①dispatcher 어휘 ②**arg-key**
(`discoverable_tool_name`) ③**field-default**(`transaction_id`). 여기에 `t2_compliance.py`의
**retail 도메인 명사**가 4번째 축 후보다.

### §8-F. 다음 순서 (전부 무료)

1. ✅ 감사 스코프 → import 폐포 (**완료**)
2. §2/§3/§4 **재작성** — t2_resolve/t2_compliance/t2_run_gated/t2_scaffold_get 사이트 편입 · 근본원인 3축 정정 · 수정단위 재산출
3. B2·B3 처방 수정 반영 (§8-B·§8-C)
4. [[05]] Q1 재계산 (§8-D)
5. **재리뷰**

**현 승인 상태**: **U0만 진행**(구현·검증 완료·8배터리 PASS). U1·U2·U3·U4·U5 = 개정 후 재리뷰.
