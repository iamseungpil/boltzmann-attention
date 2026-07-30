# 엔진 도메인-리터럴 전수 감사 + 이관 설계 (X6-h) — 2026-07-30

> 무료 트랙. handoff §6-⑤(엔진 위반 2건 이관)의 선행 감사 + 처방 설계.
> **상태 = 리뷰 대기.** 엔진 변경이므로 승인 전 구현 금지([[03]]-7).
> 도구 = `scripts/distill/tau2/x6h_engine_literal_audit.py` · 데이터 = `tau2_domain_toolnames.json`
> 원시 결과 = 이 문서 §2(건별 전수). 상위 = `X_FREE_TRACK_RESULTS_2026_07_30.md` §13d/§13e.

## §0. 한 문단

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

## §2. 건별 판정 (live 실행 코드 전수 · [[08]] per-case)

### 확정 위반 — A2 이관 필요 (7 사이트)

| # | 위치 | 리터럴 | 판정 | A2 출처 (**이미 존재**) |
|---|---|---|---|---|
| V1 | `t2_gate_patch.py:1653` | `discoverable`·`^give_`·`^unlock_` (`_PROCEDURAL_RE` 가지) | 공유 실효-write 술어의 **어휘가 banking 명명 관행 의존** | `eplan.{dispatch,unlock,list}_tool` + `_declared_give_tool(a2)` |
| V2 | `t2_gate_patch.py:4534` | `^(give\|call)_` | 동일 근본원인(F6b 국소 판정) | 동일 |
| V3 | `t2_gate_patch.py:3771` | `"unlock_discoverable_agent_tool"` (기본값) | **banking 기본값 fallback** — 엔진 주석 `:59`가 스스로 경고한 어휘 누수 | `eplan.unlock_tool` |
| V4 | `t2_gate_patch.py:3773` | `"list_discoverable_agent_tools"` (기본값) | 동일 | `eplan.list_tool` |
| V5 | `t2_gate_patch.py:2281` | `"You are a precise banking assistant."` | REF_ISO 서브콜 프롬프트에 **도메인 명사** (§13d ① 확인) | — (§4 U3: 제거가 최소비용) |
| V6 | `t2_prekb_patch.py:134` | `"unlock_discoverable_agent_tool"` | `_effective_fams` 하드코딩 (**fallback조차 없음**) | `eplan.unlock_tool` |
| V7 | `t2_prekb_patch.py:136` | `"call_discoverable_agent_tool"` | 동일 | `eplan.dispatch_tool` |
| V8 | `t2_prekb_patch.py:575` | `"give_discoverable_user_tool"` (넛지 산문) | **자기모순** — 같은 파일 `:178` `_declared_give_tool()`가 "[[05]] 위반이라 A2서 읽는다"고 명시했는데 이 자리는 안 함 | `_declared_give_tool(a2)` (헬퍼 존재·`:378`서 이미 사용) |

> **근본원인 1개**: V1·V2·V3·V4·V6·V7·V8 = **전부 banking discoverable-dispatcher 어휘**.
> A2 선언이 **이미 4개 다 있는데도** 엔진 6곳이 각자 하드코딩·기본값·정규식으로 중복 표현했다.
> ⇒ §13e의 "엔진 고정 + A2 선언" 주장은 **구조적으로 옳고, 배선이 미완**인 상태다.

### 무해 확정 — 삭제만 (1 사이트)

| 위치 | 리터럴 | 근거 |
|---|---|---|
| `t2_gate_patch.py:1653` | `get_current_time` | `_READ_PREFIX_RE`(`^get_`)가 **이미 잡는다**. `_is_effective_write = not READ and not PROC`이므로 이 가지는 **죽은 중복**. 실측: 5도메인 122도구 전수에서 이 가지 제거로 `is_write` 판정이 바뀌는 이름 **0개**. 두 사용처(`:1818`·`:4534`) 논리 동일(`RDP.match or PRC.search`)이라 양쪽 무해. |

### 관용 — 기록하되 미수정 (18 사이트 / 1 개념)

| 리터럴 | 사이트 | 판정 |
|---|---|---|
| `_\d+$` ×16 · `_\d{3,4}$` ×2 (gate·eplan·prekb) | 18 | banking의 **숫자-접미사 명명 관행** 의존(`close_credit_card_account_7834`). 도구 *이름*이 아니고, 접미사 없는 도메인에선 **no-op**이라 전이를 깨지 않는다. 엔진 주석 `:1665`가 이미 근거를 기술("명명 관행·도메인 리터럴 아님") — 수용. **단 잠재 결합 기록**: 이름이 정당하게 숫자로 끝나는 도메인이 오면 잘못 절단한다. |

### 프레임워크-공통 — 허용 (1 사이트)

`transfer_to_human_agents` (prekb `FRAMEWORK_FINAL`) — 5/5 도메인 존재. 엔진 참조 정당.

### selftest 픽스처 — 비위반 (52 사이트)

prekb 48 · guided 4. `if __name__ == "__main__":` 블록의 테스트 데이터. **§13d ②(`:627`)가
여기 속한다** — 엔진 동작이 아니므로 위반이 아니다.

## §3. §13d 정정 (자기정정 7건째)

| §13d 주장 | 실측 | 성격 |
|---|---|---|
| "실제 코드 위반 **2건**" | **7 사이트 / 5 수정단위** | **과소 6** |
| 위반 ② = `t2_prekb_patch.py:627` | **selftest 픽스처 = 비위반** | **과대 1** |
| "[[05]] 엔진 리터럴 0은 대체로 지켜졌다" | 방향은 맞음(8파일 중 5개 완전 clean·근본원인 1개) — 단 **"대체로"의 근거가 grep이었고 grep은 셋을 구분 못 한다** | 근거 교체 |

**교훈(반복 방지)**: **grep 판정으로 [[05]] 준수를 주장하지 말 것.** [[05]] 체크리스트 4항이
`grep "<도메인 도구명>"=0`을 검증 수단으로 적고 있는데, grep은 주석·픽스처·플레이스홀더·정규식
가지를 구분하지 못해 **양방향 오류**를 낸다. 검증 수단을 `x6h_engine_literal_audit.py`로 승격한다.
(이는 [[05]] §메타 실패모드의 "proxy-game"·"letter≠spirit" 계열 — 체크리스트가 proxy면 만족시켜도
목표는 위반된다.)

## §4. 처방 — 5 수정단위

공통 원칙: **새 A2 키 0**. 기존 선언을 읽고, 미선언 도메인이면 **안전한 no-op**(prekb `:181-186`이
이미 확립한 관례 — "미선언 도메인이면 None → 채널 교정 자체를 하지 않음").

### U0 — `get_current_time` 가지 삭제 (무해·독립)
`_PROCEDURAL_RE`에서 `|get_current_time` 제거. 판정 불변(§2 실측). **U1과 분리해 단독 커밋**
가능 — U1이 승인 안 나도 이건 진행 가능.

### U1 — 공유 실효-write 술어의 어휘를 A2에서 (V1·V2) ⚠게이트 있음
- `_PROCEDURAL_RE`는 **범용 가지만** 남긴다: `^log_|^verify_|_verification$|^kb_|^shell$|transfer_to_human`.
  (`^log_`·`_verification$`는 실측상 banking 전용이지만 **개념이 도메인-일반**[로깅·검증]이라 유지.
  이 판단은 명시적 선택이며, 반증되면 A2로 내린다.)
- 도메인 어휘는 모듈 전역 `_A2_PROCEDURAL: frozenset`으로 분리. `_domain_a2()`가 A2를 캐시할 때
  (`:91-100`) `eplan.{dispatch,unlock,list}_tool` + `_declared_give_tool(a2)`에서 **파생**해 채운다.
- `_is_effective_write(name)`은 `PROC 정규식 ∪ _A2_PROCEDURAL` 둘 다 본다. 호출부 6곳
  (`:1826`·`2864`·`2913`·`2945`·`2971`·`4864`) **시그니처 불변** = 국소 변경.
- V2(`^(give|call)_`)도 같은 집합을 쓴다.
- ⚠**C211 리뷰 가드레일 위반**: `:4527` 주석이 *"공유 `_PROCEDURAL_RE`/`_eff_tool_name` 불변"*을
  F6b 승인 조건으로 명시했다. U1은 그 동결을 푸는 것이므로 **사용자 재승인 필수**. 승인 없으면
  U1 보류(U0·U2·U4·U5만 진행).

### U2 — banking 기본값 제거 (V3·V4)
`ep_spec.get("unlock_tool", "unlock_…")` → `ep_spec.get("unlock_tool")`. `_safe` 집합은 **truthy만**
넣는다. banking은 A2가 3개 다 공급하므로 **행동 동일**; 새 도메인은 조용히 banking 이름을
물려받는 대신 그 레버가 비활성된다(안전측). 엔진 주석 `:59`가 요구한 형태와 일치.

### U3 — REF_ISO 프롬프트 도메인 명사 제거 (V5) ⚠게이트 있음
`"You are a precise banking assistant."` → `"You are a precise assistant."`
- **A2 키 추가 안 함**(persona를 A2로 옮기는 건 순증이고, 서브콜 과제[제시된 listing에서 참조
  해소]는 도메인 명사로부터 정보를 얻지 않는다).
- ⚠**이것만 프롬프트 변경 = 모델 입력 변화 = 행동이 바뀔 수 있다.** [[03b]] 규율상 "무해"라고
  주장하지 않는다. §5 측정 게이트.

### U4 — `_effective_fams` dispatcher 이름 A2화 (V6·V7)
`_effective_fams(tc)` → `_effective_fams(tc, a2)`. 호출부 2곳(`:425`·`:488`) 모두 `a2`가 스코프
안에 있다(`:407`이 `_trigger_fams(a2)` 호출). A2 미선언 시 현행처럼 `[_fam(nm)]` 반환(no-op).
selftest 4개 assert(`:636`·`:637`·`:750`·`:759`) 인자 추가 필요.

### U5 — arg-producer 넛지의 give 도구명 (V8)
`_declared_give_tool(a2)` 호출로 교체(같은 파일 `:178`·이미 `:378`서 사용). None이면 **넛지 자체를
건너뛴다**(`:181-186` 관례). banking은 A2 `scaffold_get_tools[].follow_up.tool`이 공급하므로
문자열 동일 = 행동 동일.

## §5. 위험 · 측정 게이트

| 단위 | 행동 변화 위험 | 게이트 |
|---|---|---|
| U0 | **없음**(실측: 122도구 판정변화 0) | selftest + 회귀 배터리 |
| U2·U4·U5 | **없음(구성상)** — banking A2가 동일 문자열을 공급하므로 해소 결과가 바이트 동일 | ①해소값 동일성 assert(신규 테스트: A2 유래 이름 == 구 리터럴) ②기존 회귀 배터리 전부 ③오프라인 궤적 재생 1건 |
| U1 | **중간** — 공유 술어 변경이 6 호출부에 전파. C211이 동결한 대상 | **사용자 재승인** + 상기 ①②③ + `_is_effective_write`를 5도메인 122도구 전수에 돌려 **before/after 판정 diff = 0** 확인(무료·기계적) |
| U3 | **미지** — 프롬프트 변경 | **무료 격리 프로브**: REF_ISO 서브콜을 기존 궤적 입력으로 재생해 base("banking assistant") vs treat("assistant") 선택 일치율 측정. 불일치 0 또는 개선이면 GO; 악화면 A2 `ref_iso[].persona`로 후퇴(그때는 A2 순증을 측정으로 정당화) |

**공통 금지**: U1~U5를 한 커밋에 섞지 않는다. U0·U2·U4·U5(무해군) → U1(재승인 후) → U3(프로브 후)
순서. 섞으면 회귀 발생 시 귀속이 불가능하다(C211 F6b 교훈).

## §6. 검증 계획 (전부 무료)

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
