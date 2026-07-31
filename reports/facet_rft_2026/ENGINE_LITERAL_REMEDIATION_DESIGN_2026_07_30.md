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
| **§9 2차 감사 처방** | ✅**유효·리뷰 대기**(필드 namespace 23건 건별 판정 6클래스 + 수정단위 V0~V6 + 검증 C1~C6) |
| §9-5 3차 축(산문 명사) | 📝**기록 전용·처방 없음** — 축 존재만 확정(육안 5건)·열거 불가(토큰 교차 161건 대부분 오탐)·실익 0 실측(§10) |
| **§11 V6 결정** | ✅**확정**(2026-07-31) — 적용하되 **수치 불변 증명이 조건**·retail g3 수율 0.2% 실측·`g3_measurable` 신설 |
| **§12 V7 `give` 서명** | ✅**신설·초판 판정 철회** — 정책이 서명을 명시하므로 A2 EXT 선언은 **정당**(gaming 아님)·실측 82/105(78%) 위반·**opex = banking EXT +1 / 타 도메인 0** |
| **§10 U3′ 측정 게이트** | ⚠️**부분 이행**(32B on 8141) — **페르소나 축만** 측정: 행동 변화 0/81·값 축 분모 = 케이스 8/궤적 2. **`DISCOVERY_REQUIRED_FB` 어휘 축은 미측정**(§10-5). 원장 C246 |

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
| **U3'** | B1 페르소나 + **A6 산문**(액션 예시·도구 인스턴스명) | 리뷰 §8-D대로 **한 단위로 묶어 프로브 1회**. Q1 ⓐ/ⓑ 선택을 이 프로브가 결정 | ⚠️**부분 이행**(§10·C246) — **B1 페르소나 축 = 32B 행동 변화 0 [M]**(케이스 8·궤적 2) / **A6 `DISCOVERY_REQUIRED_FB` 어휘 축 = 미측정**(프로브가 페르소나만 대조했다·§10-5에 잔여 게이트 명시) |
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

---

# §9. 2차 감사(필드 namespace 축) — 건별 판정 + 처방 설계 (2026-07-30 야간)

> 전판 handoff §5가 "23 사이트 열림·처방은 다음 세션"으로 남긴 것을 여기서 판정한다.
> 감사 재현 = `x6h_engine_literal_audit.py`(폐포 14파일) · live 도메인-필드명 = **23건**
> (`t2_gate_patch` 8 · `t2_resolve` 7 · `gate_interpreter` 6 · `t2_compliance` 2).
> **판정 방법 = 23건 전건 코드 정독 + 각 키의 A2 실선언 대조.** 집계로 판정하지 않는다([[08]]).

## §9-0. 한 문단

23건은 **한 종류가 아니다** — 6클래스로 갈리고, 그중 **진짜 미배선은 1건**뿐이다. 최대 덩어리인
**13건은 A2가 이미 선언한 키의 엔진 fallback**이고, 3도메인 A2를 대조하니 **전건이 정적으로 죽은
값**이다(선언한 도메인만 그 레버를 갖고, 갖지 않은 도메인은 스펙 자체가 없어 경로가 안 열린다).
따라서 처방은 값 변경이 아니라 **fallback 삭제 + 미선언이면 skip**(U2′ 원칙의 확장)이고 **예상
diff 0**이며, 그 diff 0은 주장이 아니라 **도구로 증명**한다(§9-3). 나머지는 **동음이의어 오탐 4건**
(`membership` = 엔진 operand-kind 이름이지 airline 필드가 아니다), **문서화된 도메인-일반 토큰 1건**,
**retail 어휘 2건**(`T2_DISAMB_ORDER` 경로 — 정본 스택 `go_stack.sh`에 없어 라이브 OFF),
**계측-지표 2건**(compliance G3 정의 = U7 caveat와 같은 뿌리), **하드코딩 1건**(`t2_resolve.py:539`).
⇒ **[[05]] 위반으로 고쳐야 하는 것은 13+2+1 = 16건**이고, 계측 2건은 지표 정의라 별 축으로 다룬다.

### [[05]] 3질문 ([[17]] 상설 의무)

| # | 질문 | 답 | 근거 |
|---|---|---|---|
| **Q1** | scaffold **또는 A2**의 도메인-특화를 *순증*시키나? | **No (순감)** | fallback 13건의 키는 3도메인이 이미 선언(§9-1 대조표) = 새 키 0. V4는 새 키 1개(`disamb_sub_args_optin`)를 쓰지만 엔진 리터럴 2 제거와 교환. V5는 기존 `compute_ops` 스펙에 필드 2개 추가. 엔진 필드 리터럴 16 → 0. |
| **Q2** | 모델의 *유동적 판단*을 결정론에 동결하나? | **No** | 판정 술어·발화 조건·프롬프트 산문 전부 불변. 바뀌는 것은 **같은 문자열을 어디서 읽는가**와 **미선언 시 켜지느냐 꺼지느냐**뿐. 모델 입력 변경 0(U3와 달리 측정 게이트 불요). |
| **Q3** | scaffold가 모델 대신 *도메인 행동을 수행*하나? | **No** | 새 도구 호출·fetch·값 선택 0. |

⇒ 전부 No = 기본 허용. 단 **V6(계측)**은 지표 정의를 건드리므로 수치 재산출 의무가 붙는다(§9-2).

### [[22]] 닫힌/열린 술어 점검 (상설 항목)

바뀌는 술어는 전부 **닫힘**이다: "이 키가 A2에 선언됐는가"(문자열 존재 여부) · "이 게이트 kind가
있는가". 열린 술어(자연어 해석·의도 판단)를 새로 만들지 않는다. 미선언 시 **skip = 안전측**이라
열린 판단으로 메우지 않는다.

## §9-1. 건별 판정 (23건 · 6클래스)

### 클래스 A — 동음이의어 오탐 4건 (**수정 대상 아님**)

| 사이트 | 코드 | 판정 |
|---|---|---|
| `t2_resolve.py:580` | `if kind == "membership":` | `kind` = 엔진 operand 분류 |
| `t2_resolve.py:587` | `"reason": "membership"` | deny 이유 코드 |
| `t2_resolve.py:627` | `order = {"provenance":0,"membership":1,"value":2,"operator":3}` | 엔진 고정 해소 순서 |
| `t2_gate_patch.py:3875` | `print("[T2_CONS] membership deny …")` | 계측 로그 문자열 |

`resolve_operand`의 docstring이 명시하듯 `kind ∈ {operator, membership, provenance, value}`는
**엔진 자신의 operand 분류 어휘**다. airline A2에 `membership` 필드가 있어 필드-namespace가 같은
문자열에 걸린 것이고, 도메인 어휘 누수가 아니다. ⇒ **오탐**.

⚠단 이 4건을 감사 도구에서 *지우지 않는다*. 허용목록으로 숨기면 새 occurrence도 같이 숨는다.
처방 = **판정 원장 파일**(literal+함수명+근거)로 박제하고, 도구는 계속 보고하되 원장에 있는 건만
"판정됨"으로 접는다(§9-2 V0).

### 클래스 B — A2 선언 키의 엔진 fallback 13건 (**전건 정적 死값**)

| 사이트 | 코드(요약) | 키 | 선언 도메인 | 미선언 도메인에서 경로가 열리나? |
|---|---|---|---|---|
| `gate_interpreter.py:154` | `src.get("user_id_arg","user_id")` | `entity_source.user_id_arg` | retail(G_EXHAUST) | ❌ `exhaust_before_escalate` 게이트 없음 |
| `gate_interpreter.py:161` | `gate.get("owner_field","user_id")` | `owner_field` | retail G3 · airline G3 | ❌ `ownership` 게이트 없음 |
| `gate_interpreter.py:181` | `gate.get("user_id_arg","user_id")` | `user_id_arg` | retail G6 | ❌ `select_confirm` 없음 |
| `gate_interpreter.py:185` | `gate.get("detail_id_arg","order_id")` | `detail_id_arg` | retail G6 | ❌ 동일 |
| `gate_interpreter.py:367` | `gate.get("user_id_arg","user_id")` | 동일 | retail G6 | ❌ 동일 |
| `gate_interpreter.py:371` | `gate.get("detail_id_arg","order_id")` | 동일 | retail G6 | ❌ 동일 |
| `t2_gate_patch.py:407` | `g6.get("user_id_arg","user_id")` | 동일 | retail G6 | ❌ `g6 is None` 가드가 선행 |
| `t2_gate_patch.py:3063` | 동일(두 번째 배선점) | 동일 | retail G6 | ❌ 동일 |
| `t2_gate_patch.py:574` | `sp.get("record_key_field","account_id")` | `param_cap_check[].record_key_field` | banking | ❌ `param_cap_check` 스펙 없음 |
| `t2_gate_patch.py:719` | `sp.get("record_field","merchant_name")` | `ref_verify[].record_field` | banking | ❌ `ref_verify` 없음 |
| `t2_resolve.py:403` | `def parse_records(text, key_field="transaction_id", …)` | 호출측이 항상 전달 | banking | ❌ 기본값 도달 경로 없음(§9-3 C2에서 확인) |
| `t2_resolve.py:483` | `spec.get("param","transaction_id")` | `reference_filter[].param` | banking | ❌ `reference_filter` 없음 |
| `t2_resolve.py:502` | `spec.get("criteria_fields") or [date,merchant,transaction_type]` | `reference_filter[].criteria_fields` | banking | ❌ 동일 |

**대조 근거**(실측): banking = `key_field`·`param`·`criteria_fields`·`record_key_field`·
`record_field`·`record_require` 전부 선언 / retail = `user_id_arg`(G6·G_EXHAUST)·`detail_id_arg`·
`owner_field` 선언 / airline = `owner_field` 선언. **어떤 도메인도 fallback에 의존하지 않는다.**

⇒ 위험은 "지금 틀린 값이 쓰인다"가 아니라 **"새 도메인이 키를 빼먹으면 엔진이 조용히 banking/
retail 어휘로 동작한다"**다. B3/U2′가 세운 원칙(미선언 = 레버 skip)의 정반대. 그래서 고친다.

### 클래스 C — 문서화된 도메인-일반 토큰 1건 (**유지**)

`t2_gate_patch.py:62` `DEFAULT_ARG_HINTS = ("email","name","zip","user_id","username","id",…)`.
2026-07-13 [[05]] 감사가 도메인-특화 어휘를 A2로 이관하고 **도메인-일반 식별 토큰만** 남긴 결과다
(주석에 근거 보존). `user_id`는 3도메인 공통 식별자. ⇒ 유지·원장 박제(V0).

### 클래스 D — retail 어휘 2건 (**A2로 이관**·라이브 OFF)

`t2_gate_patch.py:2471`·`:3359` `if T2_DISAMB_ORDER: sub_args |= {"order","order_id"}`.
바로 위 줄이 `sub_args = A2["disamb_sub_args"]`(retail=`["item"]`)인데, **환경 플래그가 A2를
우회해 retail 어휘를 엔진에서 주입**한다. `T2_DISAMB_ORDER`는 `generalized_*.sh`(구 retail 캠페인)
에만 있고 **정본 `go_stack.sh`에 없다** ⇒ 라이브 banking 스택 발화 0. 위험 0·부채 확정.

### 클래스 E — 계측 지표 정의 2건 (**별 축**)

`t2_compliance.py:139` `args.get("user_id")` · `:143` `args.get("order_id")` — G3(타-유저 접근)
위반 판정. 즉 **우리가 논문·특허에 인용하는 compliance 지표의 정의**가 retail/banking 필드명에
묶여 있다. U7 caveat("banking은 G3 order-resolve 생략 = 상한")와 **같은 뿌리**다. 에이전트-면
레버가 아니므로 [[05]] 위반은 아니지만 **지표의 도메인-일반성** 문제라 처방 축이 다르다.

### 클래스 F — 진짜 하드코딩 1건 (**배선**)

`t2_resolve.py:539` `recs = _gathered_records(msgs, "transaction_id", ("date","amount"))` —
`resolve_compute_params` 안. **A2 override 경로가 아예 없다**(클래스 B와 결정적 차이). 같은 함수가
U6b에서 dispatcher 이름은 A2로 옮겼는데 이 record-키는 남았다.

**부수 발견(도구가 못 잡음)**: `t2_resolve.py:484` `tuple(spec.get("require") or ("date","amount"))`
의 기본값 `date`/`amount`는 A2 필드 namespace(37개)에 없어 **감사에 안 걸렸다**. §5 교훈("축을 하나
더 열면 clean이 뒤집힌다")의 재확인이고, V5에 함께 넣는다.

## §9-2. 처방 (수정단위 V0~V6)

| 단위 | 대상 | 처방 | 비용/위험 |
|---|---|---|---|
| **V0** | 클래스 A 4건 + C 1건 | **판정 원장** `engine_literal_adjudications.json` 신설(키 = literal+함수명+근거·line-anchor는 취약해 배제). 감사 도구가 원장 대조 후 "판정됨/신규"로 분리 출력 | 무료·행동 0 |
| **V1** | `gate_interpreter.py` 6건 | fallback 삭제 → **미선언이면 그 레버 skip**(owner 해소 불가 = 기존 `None` 보수 경로 · select_confirm 미선언 = `None` 반환) | 무료·diff 0 예상(§9-3 C1) |
| **V2** | `t2_gate_patch.py` 4건(407·574·719·3063) | 동일. 스펙은 있는데 키가 없으면 **그 레버만 skip + 1회 통지** | 무료·diff 0 예상 |
| **V3** | `t2_resolve.py` 3건(403·483·502) | `parse_records`의 `key_field`를 **필수 인자로 승격**(기본값 삭제) · `param`/`criteria_fields` 미선언 시 해당 spec skip | 무료·호출부 전수 확인 선결(C2) |
| **V4** | `t2_gate_patch.py` 2건(2471·3359) | 엔진 리터럴 삭제. 플래그는 **A2 `disamb_sub_args_optin`**(기존 배열의 opt-in 부분집합)을 켜는 역할로 축소 | 무료·라이브 OFF |
| **V5** | `t2_resolve.py:539`(+484) | `resolve_compute_params`가 A2 `compute_ops[].record_key_field`/`record_require`를 읽고 **미선언이면 records 수집 생략** | 무료·**행동 변화 가능** ⇒ C3 선결 |
| **V6** | `t2_compliance.py` 2건 | G3 필드를 **A2 게이트 선언에서 도출**(`gates[kind=ownership].owner_field`·`detail_id_arg`). banking은 ownership 게이트가 없으므로 **G3 = 측정불가로 명시 산출**(현 caveat를 각주에서 **기계-도출 값**으로 승격) | 무료·**수치 재산출 의무**(U7 문구 갱신·논문/특허 전파) |

**순서**: V0 → V1·V2·V3(동질·diff 0 증명 묶음) → V4 → **V5·V6은 별 승인**(전자는 행동 변화 가능,
후자는 인용 수치 변경).

## §9-3. 검증 계획 (make-or-break)

| # | 검사 | 통과 기준 | 성격 |
|---|---|---|---|
| **C1** | **정적 死값 증명 도구** 신설: 3도메인 A2 × 13사이트에 "해당 레버 스펙 존재 ⇒ 키 선언 존재" 전수 검사 | **13/13 참**(반례 0) ⇒ V1~V3의 diff 0이 *주장이 아니라 증명* | 필수·무료 |
| **C2** | `parse_records` **호출부 전수**(폐포 14파일 + selftest) — `key_field` 미전달 호출 0 | 0건 | 필수 |
| **C3** | `resolve_compute_params`에서 `recs`가 실제로 쓰이는 op 경로 확인(records 없으면 어떤 op가 죽나) | 죽는 op 목록 확정 → V5 확정 or 보류 | 필수([[14]] 동형: 조용한 사망 금지) |
| **C4** | 회귀 배터리 12종 + `test_c241_u1_predicate` | 전부 PASS | 필수 |
| **C5** | A2-swap diff(`x6_a2_swap_diff.py`) 재실행 | 기능군 diff 불변 | 권고 |
| **C6** | V6 후 compliance 수치 재산출 · banking G3 = "측정불가" 명시 | 기존 수치와의 차이 전건 설명 | V6 조건 |

⚠**배포 시점**: Y1 본런이 8140에서 진행 중이다. 실행 중 프로세스는 코드를 이미 로드했으므로 로컬
커밋은 무해하나, **리모트 배포·재기동은 Y1 완주 후**로 미룬다(측정 위생·[[30]]).

## §9-4. 미결 / 리스크

1. **V5가 행동을 바꿀 수 있다** — C3 결과에 따라 "미선언 시 records 생략"이 무해한지 판정. banking은
   선언할 것이므로 실질 영향은 새 도메인뿐이지만, 그 주장도 C1처럼 **증명**해야 한다.
2. **V6은 수치 축** — compliance는 논문·특허 인용 대상이다. 재산출 없이 코드만 고치면 인용과 코드가
   갈린다. V6은 반드시 C6과 묶어서만 진행.
3. **축이 또 열릴 수 있다** — 부수 발견(`require=("date","amount")`)이 보여준 대로 필드 namespace도
   완전하지 않다(A2 유래 37개 기준). **enum 값·상태 문자열 축**이 다음 후보다. "clean"을 선언하지 말 것.

## §9-5. 기록 전용 — 3차 축(엔진 모델-대면 산문의 도메인 명사). **처방 없음**

> 이 절은 **부채 기록**이다. 수정단위를 만들지 않는다. 근거는 §10의 실측이다 — 페르소나 명사
> 제거/유지가 32B 행동을 **바꾸지 않았다**(분기 0/81) ⇒ 이 축의 명사는 **정확도를 사지 않는다**.
> 사지 않는 것을 고치려고 A2 키를 늘리면 opex만 오른다(제1원리: 레버는 하나를 사면 하나를 판다).

### 축의 존재 (확정 사례)

도구명·필드명 namespace 둘 다 못 잡는 형태가 있다 — **프롬프트/피드백 산문 안의 평문 도메인 명사**.
`transaction_id`는 필드 namespace에 있지만 `transaction`은 없다. 육안 확정분:

| 위치 | 산문 | 명사 |
|---|---|---|
| `t2_resolve.py:302` | "A user is talking to a **bank** agent. …" | bank |
| `t2_resolve.py:442` | "The user is referencing a specific **transaction**/record. …" | transaction |
| `gate_interpreter.py:275` | "not permitted: this **order**'s status is '{cur}' … Do NOT retry … on this **order**." | order(retail) |
| `t2_compute.py:388` | "NOTE: no **card** is in 'eligible'. Do NOT conclude that nothing fits …" | card |
| `t2_scaffold_get.py:1442` | "could not be verified against the **account** records / knowledge base …" | account |

성격이 페르소나와 다른 건 하나 있다: 페르소나는 **정보를 더하지 않아 삭제로 끝나는데**, 위 사례들은
문장이 "무엇을 대상으로 하라"를 지시하므로 **대상 명사가 문장 기능의 일부**다. 그래서 만약 훗날
이 축을 산다면 형태는 자유산문 A2 키가 아니라 **닫힌 라벨**(예: `reference_filter[].record_noun`·
`domain_label` 한 단어를 엔진 고정 템플릿에 보간)이어야 한다 — EXT 화이트리스트가 닫힌 3종만
허용하고, 산문 선언은 AXIS 금지선의 "열린 처방"에 걸린다. **단 지금은 사지 않는다.**

### 감사 방법의 한계 (열거 불가 — 정직 기록)

이 축을 도구화하려고 시도했다: 엔진 폐포 14파일의 문자열 중 **모델-대면 영문 문장**(길이·공백
기준·한글 docstring 제외·selftest 제외)을 뽑고, tau2 5도메인 도구명을 `_`로 쪼갠 **단일-도메인
토큰**과 교차했다. 결과 **161건**이 나왔으나 **대부분 오탐**이다 — `deny`·`report`·`request`·
`customer`·`from`·`last`·`state` 같은 범용어가 우연히 한 도메인 도구명에만 등장해 토큰 집합에
들어간다. **이미 폐기한 "어휘 조각 검사"(152건 거의 전부 오탐·§1 도구 한계 2)와 같은 실패 형태**다.

⇒ **이 축은 열거하지 않았다.** 위 표는 전수가 아니라 육안 확정분이고, "산문 명사 N건"이라는
수치를 만들지 않는다. 축의 **존재만** 기록으로 남긴다([[08]]: 노이즈 집계로 판정 금지).

### 이 기록을 언제 다시 꺼내야 하나

1. **다중턴 측정에서 도메인 프레이밍이 정확도를 사는 것이 보일 때** — 그때 형태는 위 닫힌 라벨.
   현 근거는 §10의 단발 서브콜 조건이고 분모가 케이스 8·궤적 2라 강한 일반화가 아니다.
2. **airline/telecom 등 4번째 도메인을 실제로 붙일 때** — 그 도메인에서 위 문장들이 어색하거나
   오도하면(예: airline에 "this order's status") 그 시점의 실패가 이 기록을 처방으로 승격시킨다.

---

# §10. U3′ 측정 게이트 — 32B 이행 결과 (2026-07-30 야간·원장 C246)

> §4 U3′ 행의 "⚠**측정 게이트 미이행**"을 여기서 닫는다.
> 도구 = `x9_refiso_persona_probe.py`(프롬프트 축자 복제) + 판정 `x9b_refiso_adjudicate.py`
> 데이터 = `sim_results/x9_refiso_persona_32B.jsonl.gz`(영속·평문 사본 없음)
> 조건 = **32B GPTQ-Int8 · 8141(GPU1) · 8140과 동일 서빙 스펙** · 케이스 27 × 시드 3 · temp 0 · 에러 0

## §10-1. 선결 조건이 어떻게 열렸나

전판은 "8140이 Y1 점유 → 32B arm 불가"로 기록했으나, **실제 차단 원인은 프로브의 PORTS
하드코딩**이었다(32B → 8140 고정). `--base_url` 오버라이드를 넣고 **GPU1의 7B(8142)를 내리고
32B를 8141에 올려** 게이트를 열었다. 7B에는 작업이 0건이었으므로 손실은 없다.

## §10-2. 판정

| 칸 | 쌍 | 비율 |
|---|---|---|
| 둘 다 기권(행동 동일) | 57 | 70.4% |
| 둘 다 같은 값(행동 동일) | 24 | 29.6% |
| **기권 결정이 갈림(행동 변화)** | **0** | 0% |
| **값이 갈림(행동 변화)** | **0** | 0% |

⇒ **행동 변화 0/81.** 기권 결정 81/81 일치 · 양쪽이 답한 24쌍의 값 24/24 동일. 기권율도
base 57/81 = treat 57/81로 같다. **REF_ISO 서브콜에서 페르소나 명사(`banking`) 제거는
32B의 선택을 바꾸지 않는다 = U3′ GO [M].**

## §10-3. 분모를 정직하게 (이 절이 이 결과의 한계다)

- **7B는 해상도 0**이었다(27케이스 전 시드 양쪽 기권 = 전판의 "무효" 판정). 32B는 **케이스 8개**에서
  양쪽이 실제로 값을 답한다 ⇒ **바닥 효과는 부분적으로만 해소**됐다.
- 값 축 결론의 분모는 **쌍 81이 아니라 케이스 8**이고, 그 8케이스는 **궤적 2개**에서 나온다.
  나머지 19케이스는 전 시드 양쪽 기권 = **구별력 없음**(no-information)이며 **불변의 증거가 아니다**.
- temp 0이므로 시드 3은 독립 표본이 아니다. 강한 불변 주장 금지 — **"이 조건에서 분기 0"**이 결론이다.
- **정확도는 판정하지 않았다**(어느 arm이 gold에 가까운가). gold 대조는 별건이며, 이 프로브는
  **분기 여부만** 본다.
- 70%가 기권한다는 사실 자체가 프로브가 라이브 REF_ISO 조건(궤적 전체 문맥·다중턴)을 다 재현하지
  못한다는 신호일 수 있다(**C42·X5 천장과 동형**). 배포-유사 조건에서의 재확인은 열려 있다.

## §10-4. 이 게이트에서 잡은 내 도구 결함 2건

1. **x9 요약이 응답 전문 동등으로 일치를 셌다** — 그래서 "일치율 44.4% · 불일치 45건"이 나왔다.
   실체는 **양쪽 다 UNSURE인데 뒤 산문만 다른 쌍**이다. 엔진 관점에서 행동 분기는 ①기권 여부
   ②값 선택 둘뿐이므로 축을 분리했고, 그러자 불일치는 **0**이 됐다. 7B 판 각주가 지적했던 그
   아티팩트가 32B에서 45건으로 커진 것이다.
2. **첫 분모 집계가 케이스를 `sim`으로 묶어 "케이스 4개"를 냈다** — 한 궤적에 REF_ISO 케이스가
   여럿이므로 케이스 축은 `i`다. 케이스(27)와 궤적(4)을 **둘 다 보고**하도록 교체했다.

⇒ 교훈은 §5·§8과 같은 형태다: **집계는 그럴듯한데 사례를 읽으면 다르다.** 이번에는 사례가 아니라
**집계 정의**가 틀렸고, 그것도 사례(응답 원문)를 읽어서 잡았다.

## §10-5. ★게이트의 미이행 잔여 — 이 프로브는 U3′의 절반만 쟀다

U3′ 수정단위는 **두 축**이다(§4 표): **B1 페르소나 명사** 제거 + **A6 `DISCOVERY_REQUIRED_FB`
banking 어휘** 제거(도구 인스턴스명 `open_bank_account_4821`과 액션 예시 `'open bank account'`·
`'close bank account'`·`'apply savings interest correction'`를 빼고 `{getter}/{unlock}/{call}/{list}`
플레이스홀더로 전환). **이 프로브는 전자만 대조했다** — `x9`의 base/treat 차이는 페르소나 한 문장뿐이다.

따라서 정확한 판정은 이렇다:

| 축 | 상태 |
|---|---|
| B1 페르소나 | ✅ 32B 행동 변화 0/81 [M](분모 = 케이스 8·궤적 2) |
| **A6 `DISCOVERY_REQUIRED_FB` 어휘** | ❌ **미측정** |

왜 같은 방식으로 못 재는가: 페르소나는 **격리된 서브콜 프롬프트**라 같은 입력에 두 문구를 넣어
바로 비교할 수 있다. 반면 `DISCOVERY_REQUIRED_FB`는 **궤적 중간에 게이트 피드백으로 주입**되고,
그 다음 에이전트가 discovery 체인(getter → unlock → call)을 **여러 턴에 걸쳐** 수행하는지가 결과다.
단발 프롬프트 대조로는 그 행동이 재현되지 않는다(**C42·X5 천장과 동형** — 짧은 문맥 프로브는
결손을 재현하지 못한다).

⇒ **잔여 게이트**: A6 축은 **다중턴·배포-유사 조건**에서만 판정 가능하다. 후보는 (a)X3형 two-pass
프로브에 discovery-required 케이스를 넣기 (b)8140 해제 후 **동일 스택 2-arm**(구 문구 ON/OFF)
소규모 런. 후자는 유료이므로 [[09]] 승인 대상이고, **Y1이 세우는 flip 임계(현재 4/14 = 29%)보다
작은 차이는 판정 불가**라는 제약을 미리 안고 있다.

⚠따라서 "U3′ 안전 확인"이라고 쓰지 말 것. 정확한 문장은 **"U3′의 페르소나 축은 32B에서 분기 0,
FB 어휘 축은 미측정"**이다.

## §11. V6 결정 — compliance G3 지표 (2026-07-31 · 결정 확정)

> §9-2 표에서 "별 승인"으로 분리했던 항목. **Y2 이전에 정해야** 하는 이유는 V6이 코드가 아니라
> **Y2를 채점하는 자**를 바꾸기 때문이다(적용 후 재산출하면 과거 수치와 기준이 갈린다).

### §11-1. 무엇이 문제였나

`t2_compliance.py:139·143`이 G3(타-유저 접근) 판정을 `args.get("user_id")`·`args.get("order_id")`로
**필드명 하드코딩**한다. 그리고 `load_order_owner`는 **retail에서만** 소유자 맵을 만들어,
다른 도메인은 **order-ownership 간접 경로가 생략**된다(U7 caveat = "banking 수치는 G3에 관해 상한").

### §11-2. 결정 전에 잰 것 (근거)

| 사실 | 값 | 출처 |
|---|---|---|
| G3의 **직접 경로**(호출 인자의 `user_id` ≠ 인증 유저)는 **도메인 무관하게 동작** | — | 코드 정독(`:139`) |
| 생략되는 것은 **간접 경로**(order → owner 조회)뿐 | — | 코드 정독(`:143`·`load_order_owner`) |
| **retail에서도 g3 발화율이 극히 낮다** | **1 / 456 sim = 0.2%**(`fl14b_floor_retail_t4`) · 나머지 3런은 **0** | 영속 compliance 4파일 |
| **banking compliance.json 영속본 0개** | — | `sim_results/*compliance.json` = retail 4개뿐 |

⇒ **U7 caveat는 참이지만 크기가 작다**(간접 경로의 관측 수율 0.2%). 그리고 우리가 인용하는 banking
compliance 수치의 **출처가 영속본이 아니다** — 이건 별도로 정리해야 할 문제다(§11-5).

### §11-3. 결정 = **V6 적용**(단 "수치 불변"을 먼저 증명한다)

| | |
|---|---|
| **적용 범위** | ①필드명을 **A2 게이트 선언에서 도출**(`gates[kind=ownership].owner_field`·`detail_id_arg`) ②산출물에 **`g3_measurable`** 필드 추가 — 그 도메인에 ownership 게이트가 **선언됐는지**를 기계가 기록 |
| **하지 않는 것** | 검사기 이식(다른 도메인용 소유자 맵 구현)은 **범위 밖**. 그건 새 기능이고 [[05]] Q1을 건드린다 |
| **사전 예측** | **수치 변화 0** — banking엔 ownership 게이트가 없으므로 간접 경로는 **원래 안 돌았다**. retail은 선언이 있으므로 같은 값을 읽는다 |
| **검증** | 영속 compliance 4파일(retail)을 **재산출해 diff 0** 확인. 값이 바뀌면 **적용 중단하고 원인 조사** |
| **문서 조치** | U7의 산문 caveat를 **기계-도출 필드(`g3_measurable`)로 대체**. 논문·특허·원장의 banking 수치에는 "G3 간접 경로 미적용(관측 수율 retail 0.2%)"을 병기 |

**왜 적용하나** — 수치가 안 바뀌는데 왜 하냐는 반론이 가능하다. 이유는 둘이다:
①`if domain != "retail"`라는 **도메인 분기**가 계측층에 남아 있는 한 "전이 = ABox swap" 주장이
계측에서 깨진다(감사 축 C) ②caveat를 **사람이 각주로 관리**하는 상태는 누락된다 — 오늘 U7이
바로 그 위험을 보여줬다(코드에만 있고 인용에는 안 붙어 있었다).

### §11-4. Y2와의 관계

V6은 **수치 불변**이 검증되면 Y2 이전 적용해도 **임계 전이 가정을 늘리지 않는다**(채점 규칙이
같으므로). diff가 0이 아니면 **Y2 이후로 미룬다**.

### §11-5. 파생 미결 (별건)

**우리가 인용해 온 banking compliance 수치의 출처가 `sim_results`의 영속본이 아니다** —
영속 compliance는 retail 4개뿐이다. 인용 시점의 계산본이 어디 있는지, 재현 가능한지 확인해야
한다. 재현 불가면 **인용을 [D]로 강등**해야 한다([[08]]·증거원장 규율).

## §12. V7 — `give` 호출 서명의 A2(EXT) 선언 (2026-07-31 · **초판 판정 철회**)

### §12-0. 내 판정을 뒤집는다

C251 ⑦에 "`give_discoverable_user_tool` 스키마가 `arguments`를 **허용**하므로 엔진이 떼면
metric-gaming ⇒ **정당한 처방 없음**"이라고 적었다. **틀렸다** — 스키마만 보고 **정책서를 안 읽었다**.
banking 도메인 정책이 호출 서명을 **두 번 명시**한다(축자):

```
- Use the `give_discoverable_user_tool(discoverable_tool_name)` function
- Explain to the user what the tool does and how to use it, and what arguments to provide.
  Just explaining isn't enough, you must use the `give_discoverable_user_tool(discoverable_tool_name)`
  function.
```

⇒ **"인자는 유저에게 말로 설명하고, 호출은 이름만"**이 **정책이 규정한 도메인 사실**이다. gold에서
읽은 게 아니다. **따라서 A2 선언은 정당하다**([[03b]] 경계 통과 — 정답지가 아니라 정책이 출처).

### §12-1. 실측 규모

| | |
|---|---|
| `give_discoverable_user_tool` 호출 | **105회** (Y1 64 sim) |
| 그중 `arguments`를 실은 것 | **82회 = 78%** |
| 영향 태스크 | **14개**(020·021·022·023·027·028·029·033·003·016·018·019·035·041) |

채점 규약상 이 여분 키는 **자동 불일치**를 만든다(C245 `PRED_EXTRA_KEY`: 예측 키가 비교 집합이
되므로 gold에 없는 키가 있으면 **반드시** 실패).

### §12-2. 형태 — **A2 EXT 선언 + compliance 게이트**(엔진은 조작하지 않는다)

| 층 | 내용 |
|---|---|
| **A2 EXT**(banking) | `give_tool.call_signature = ["discoverable_tool_name"]` — **스키마 상수**(EXT 닫힌 3종 중 하나) |
| **엔진**(도메인 일반) | 선언된 서명 밖 키가 실리면 **deny + 재발행 요구**. 모델이 다시 낸다 |
| **금지** | 엔진이 `arguments`를 **조용히 떼는 것** = tool_call 조작 = C151에서 이미 기각한 gaming |

deny+regen은 C151이 "compliance 게이트(공격표면 최소·regen이 방어적)"로 채택한 패턴이고, **선택은
여전히 모델**이 한다([[10]] 정합).

### §12-3. ★비용 회계 (사용자 지시: "비용 측면만 분명하면 된다")

특허 §3.4의 capex/opex 프레임으로 정직 계상한다:

| 항목 | 값 |
|---|---|
| **capex**(엔진·인터페이스) | **0** — 서명 검증은 도메인-일반 로직 1개. 새 도구·새 인터페이스 없음 |
| **opex: banking** | **EXT 키 +1**(`give_tool.call_signature`) |
| **opex: retail·airline** | **0** — 두 도메인엔 dispatcher/give 구조 자체가 없다(미선언 = 레버 skip) |
| 유한성 | **새 기능군 요구 0** — 기존 EXT 3종(계산식·도구셋·**스키마 상수**)의 세 번째 칸에 들어간다 |

**왜 도메인 특화가 불가피한가**(사용자 지적): discoverable-dispatcher 구조는 **banking에만** 있다.
"이 도메인에서 give는 이름만 받는다"는 사실은 **그 도메인 정책에서만** 나온다 ⇒ 도메인-일반 엔진이
스스로 알 수 없다. **그래서 A2에 선언한다.** [[05]]는 도메인 특화를 금지하는 규율이 아니라
**어디에 두는지**를 정하는 규율이다 — 엔진에 두면 위반, **A2에 두면 정상**이고 비용만 정직하게 센다.

### §12-4. 대조 — 같은 실패 축이지만 **A2 비용이 0인** 것

`032·033·035`의 **채널 오분류**(agent-callable 도구를 user-give로 넘김)는 A2 선언이 **불필요**하다.
엔진이 이미 `_user_discoverable(env)`로 **env 레지스트리에서 user-side 집합을 읽는다** ⇒ "이 도구가
user-side인가"는 **기계-도출 사실**이다. ⇒ **opex 0**.

⇒ 두 축을 나란히 두면 원칙이 보인다: **환경에서 읽히면 A2 비용 0, 정책 산문에만 있으면 A2 EXT +1**.

### §12-5. 상한 주의 (pass 상승을 약속하지 않는다)

- 14태스크 중 **12개가 DB-basis**다. `give`에 인자를 실은 것이 **DB 해시를 바꾸는지는 미측정**이다
  (give 자체는 write가 아닐 수 있다) ⇒ **ACTION-basis 2개(033·035)** 외에는 pass 효과가 **불확실**.
- 따라서 이 처방의 1차 산출은 **`PRED_EXTRA_KEY` 소멸**(기전 지표)이고, pass는 부산물이다.
- Y1 임계 28%(≈9태스크)를 넘길 것으로 **예단하지 말 것**.

### §12-6. 검증

| # | 검사 | 기준 |
|---|---|---|
| V7-a | 정책 문구 실재 | ✅ 완료(§12-0 축자 인용) |
| V7-b | 미선언 도메인 skip | retail·airline 발화 0 |
| V7-c | 오탐 0 | 서명대로 부른 호출(23회)에 미발화 |
| V7-d | Δspurious | deny+regen이 **다른 것을 깨지 않는가** — Y2 arm에서 계측 |
| V7-e | 기전 | `PRED_EXTRA_KEY` 발생 수 82 → 0 확인(라이브) |
