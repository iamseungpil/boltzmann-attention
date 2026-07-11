# FABRICATION 레버 비용-효율 비교 정본 — 전 케이스 taxonomy × 닫는 방법 7종 (2026-07-11)

> **목적**: fabrication(= epistemic 규율 결손)을 닫는 **모든 방법**의 비용-효율 비교 정본. Part 1 = fabrication 전 케이스
> taxonomy(전수·실측 인용)·Part 2 = 방법 7종 × 케이스 비용-효율 비교·**Part 3 = ★전 실패-클래스 통합 배분표(결론부·특허 B
> 실시예 본체)**·Part 4 = 미측정 칸과 최소-비용 실험 설계(⑥ fleet-위임 하이브리드가 핵심 신규)·Part 5 = fabrication
> 케이스별 승자 권고·Part 6 = "scale-내재 vs 이전-가능" 긴장·Part 7 = 논문·특허 연결.
> **상태 = [D] 비교-정본 v1.0 (미커밋).** 측정 칸은 전부 C-번호+등급 인용(RESEARCH_MASTER §3)·**미측정 칸은 `[?]`로 명시**.
> 규율: [[03]]#9 대칭크레딧 · [[08]] per-case · [[09]] 무료 先 · 갱신 프로토콜(§3 원장 등급을 실측처럼 인용 금지 — [D]/[?]≠[M]).
> 인용 정본: RESEARCH_MASTER §3(C-원장) · [[42]] 프롬프트 한계 · [[43]] 환각=인센티브 산물 · [[45]] scale-invariant 부하.

---

## 0. 정의와 범위

**fabrication의 작업 정의(본 문서)**: **검증 안 된 단정(assertion)이 시스템에 들어오는 모든 지점.**
- "시스템에 들어옴" = tool-call 인자·tool-call 이름·사용자향 발화(NL)·내부 계획의 전제·**타자(user-sim) 단정의 수용** —
  다섯 진입로 전부.
- 좁은 정의("없는 id 발명")는 이 중 한 케이스((a))일 뿐이다. C24(free-text)·C63(escape-write)·t86(커버리지 환각)·
  banking 61/59(도구명·시간)가 같은 규율 결손의 다른 표면임이 per-case로 확정됐다.
- **기전 정본**: 날조는 WM/능력 결손이 아니라 **정박 치환**(anchored substitution — 문맥 인접 토큰의 변형·C43 [M])이며,
  **"출처를 안 대도 되는 인터페이스"의 산물**이다(C45). 인센티브 이론([[43]]: 채점이 추측>기권 보상 → 날조=default)과 정합.
- **프롬프트 채널은 죽었다(전제)**: `retail/policy.md:18`이 이미 날조를 금지하는데 순수 날조 ours 91 vs frontier 1~3
  (C30 [M]) · 금지문+예시태그 무효 9→8(C47 [M]) · 같은 규칙이 짧은 문맥선만 작동 0.87(C41 [M]) · 짧은 합성선 base 7B가
  4지선다를 완벽 해결·fabricate 0.00(C42 [M]). ⇒ [[42]] scale-emergent prior-override. **이하 비교에서 prompt는 방법이 아니다.**

---

# PART 1 — fabrication 전 케이스 taxonomy (전수·실측)

**분류 축 = 발생 지점 × 단정 대상.**

| 발생 지점 | 단정 대상 후보 |
|---|---|
| P1 tool-call 인자 | id · free-text 값 · 시간 |
| P2 tool-call 이름 | 도구명(존재하지 않는 도구) |
| P3 사용자향 발화 | 수치 · 정책 · 완료/커버리지 · 자기-능력 |
| P4 내부 계획·전제 | DB-state · 환경-동역학 |
| P5 수용(입력측) | 타자(user-sim)-단정 — 자기-생성 아님·별도 표기 |

### 케이스 전수표

| # | 케이스 | 발생 지점 | 실측 사례·기전 | 오염되는 기능(발현 실패클래스) | 빈도(측정치) | 현행 스택 커버 | 등급 |
|---|---|---|---|---|---|---|---|
| **(a)** | **tool-인자 id 날조** | P1 | 정박 치환: 32B 미조회 날조 new_item_id의 70%가 문맥 내 기존 id의 edit≤2 변형(`…3219→…3220`)·14B는 placeholder 70%·7B는 스키마-예시 복사 36/36 — **scale은 날조의 *형태*를 바꾼다**(복사→발명) (C43·C36) | 잘못된 값 write → db_fail·에러-루프(env 거부 후 미회복) | 미조회 날조율 사다리 **7B 38.8% → 14B 7.0% → 32B 6.7% → frontier 0.0%**(C36) · ours 63/439 write 14.4% vs o4-mini 1/341·gpt-4.1 0/416(C29) · sim 단위 28/456·상한 4.2pp(C29) · floor fab 이벤트 70/456 — **전부 env-차단·수락 0**(C61) | ✅ **닫힘**: env가 id형은 93/93 거부(C12) + **출처선언 prov**(C45: 격리 67→0%) + e2e GO(C53: +3.0pp·tme 1/456) + COMP 합성이 p4 회복(C62) | **[M]~[S]** |
| **(b)** | **tool-인자 free-text 날조** | P1 | t17: read 0회 상태서 `"123 Elm St"` 주소 **자유생성**(정답 원문이 문맥 어디에도 부재·실주소 근사=오염 의심). env는 타입상 거부 불가 | 주소 등 자유필드 오염 write → db_fail (5/5) | ours 5/456 vs o4-mini 0·gpt-4.1 0 · passing-spurious 0 · 상한 1.1pp (C24) | ⚠️ **입도 구멍**: prov가 검사는 하나 ① substring-관대(`"123 elm st"⊂"123 elm street"` 통과) ② rescue 분기 per-call `break`가 둘째 fab 인자 미검사(`t2_gate_patch.py:881-888`·`#`-접두 거짓양성이 방아쇠) — **PROV-RESCUE-PERARG**가 봉합 설계(C65) | **[M] 진단·[D] 봉합** |
| **(c)** | **도구명 날조** | P2 | banking: KB에 없는 `agent_tool_name`/`discoverable_tool_name` 발명. **env가 error를 주는데도 못 살아남**(deny+fetch-지시의 한계 실증) · task_012는 도구명+**절차** 동시 날조 | 발견-체인 미조립(REACH 실패의 하위 모드 a) → 절차 진행 불능 | **61/170 fail sims(35.9%)** (`BANKING_FLOOR_LEVER_FIT` [M]) · retail선 희소(미계상 [?]) | ⚠️ **검출만**: prov 'name' 힌트가 deny하나 **회복 실패 실측**(61 sims가 deny 후에도 못 살아남) — 해결 레버 부재. E-XGRAMMAR(디코드-제약)가 1차 표적 [D] | **[M] 검출·[?] 해결** |
| **(d)** | **DB-state 단정** | P4 | t20: 주문 status 혼동(frontier 대조서 "우리와 달리 status-혼동 없음" — `T27_T103_PERSTEP` 부록) · t93.0: 사용자가 잘못 짚은 pending 주문에 exchange 불가 → **"수동으로 status를 delivered로 변경" 제안·시도**(C63) · t103: gold에 없는 return이 status를 소비(`delivered→return requested`)해 gold exchange 비가역 파괴(`T27_T103_PERSTEP`) | 잘못된 전제 위 계획 → WRONG-OP·비가역 write 순서(ORDER-IRREVERSIBLE) | 전수 빈도 **미계상 [?]** (per-case 사례 [M]) | ⚠️ **하류 차단만**: 게이트가 t93 escape를 옳게 차단(C63)·상태-전제 자체의 검증 레버 없음. E10 DB-게이트는 NO-GO(C50: 불가능성은 DB 아니라 대화 semantic) | **[M] 사례·[?] 빈도** |
| **(e)** | **자기-능력 날조 (양방향)** | P3/P4 | **과대(escape-write)**: impasse-시 "manual adjustment"류 발명형 write — C63 서명(정독 3/3: t93 수동 status·t95 manual adjustment) · **과소(능력 부인)**: banking task_043 "이체 능력이 없다"며 셀프서비스 안내로 도피(실제론 KB에 도구 존재) | 과대→불법 write 시도(게이트/env deny) · 과소→조기 transfer/포기(REACH·PERSISTENCE) | deny 31 sims 전원 transfer·19 fail — 단 **게이트-유발 아님·impasse 표지**(C63 정독 3/3) · banking 자기-능력 오인 = 발견-체인 미조립 하위 모드 b | ⚠️ **차단만**: 게이트=compliance 정상작동(위반0·12/31 pass 유지)·deny 노트=**무료 검출 마커**. 근인은 상류(discovery/lookup/DISAMB — C63) | **[M]** |
| **(f)** | **발화-수치 날조** | P3 | t47: 문맥에 없는 **$928.13** 암산 총액 발화 → user-sim이 그 위에서 행동. `_unverified_amounts` 술어로 재현 ✓ | NL 축 오염 + user-sim 오도(하류 write 방향 왜곡) | V0 census(comp 456 전수·본 문서 재현 2026-07-11): passing서 가상-fire **128/289 sims(44.3%)·발화 208건·금액 322개** / failing 88/167(52.7%)·발화 164건 — **over-fire 지배** = 미검증 금액 발화 대부분은 정당(가격 diff 등) | ❌ **미커버(정직)**: NL-NUM-PROV 구현됐으나 over-fire 44.3%로 **기본 off·표적 실측으로만 재론**(NEXT_LEVER 부록 Z). 술어 정밀화 필요 | **[M] census·[?] 레버** |
| **(g)** | **발화-정책 날조** | P3 | t57: 조건체인 끝=no-op가 gold인데 취소 실행 + **"gift card로 환불했다" 허위 발화**(C64 census·4/4) | 정책-사실 오전달 → user-sim 오유도·over-action | t57형 SYSTEMIC 4/4 (C64) · 전수 빈도 미계상 [?] | ❌ **미커버**: NOTICE-PERGATE G8(무조건 고지)은 over-block 16.6%>2%로 **A2 미부착**(부록 Z — t57 정직 잔여). **NL-POLICY-PROV**(허위-주장 시만 발화하는 트리거-반전판)가 [D] 등재(부록 W) | **[M] 사례·[D] 레버** |
| **(h)** | **커버리지/완료 환각** | P3 | t86 [36]: `get_user_details`만 부르고 **"I've reviewed your order history, and none are in DC"** — 5주문 중 3주문 미조회(replay 확정·`APRIME_REGRESSION_FORENSIC` E-표 [S]) | coverage 실패로 발현(전 도메인 40~57%·최강 불변 — C52)·잘못된 "없음" 결론 → write 미수행 | 사례 [S] · "모두 확인" 류 전수 빈도 미계상 [?] | ⚠️ **부분**: E-PLAN L2 deny가 정확히 이 지점을 겨눔("unexamined siblings — get_order_details first") — 단 **불응 실측**(t95 deny 4회 전부 무시·t103 수십 회 무시) = 피드백은 있으나 순응이 병목 | **[M] 부분** |
| **(i)** | **시간 날조** | P1 | banking: `get_current_time`을 안 부르고 `time_verified="2023-04-15 14:20:00 EST"` 과거시각 발명 — env 그대로 수용(로그 성공)·**DB 행이 gold와 달라져 단독-치명** | DB-diff 지뢰(현재는 다른 blocker에 가려짐 — 단독-diff sim 0) | **59/170 fail sims(34.7%)** (`BANKING_FLOOR_LEVER_FIT` LOGV_TIME_FAB) | ❌ **사거리 밖**: 현행 힌트에 'time' 없음. **A2 `identifying_arg_types` 확장 + producer=get_current_time**으로 결정론 폐쇄 가능(**ABox-only**·[[05]] 클린) [D] | **[M] census·[D] 레버** |
| **(j)** | **confabulation-수용** (타자-단정의 무검증 수용 — **자기-생성 아님·epistemic 동류로 별도 표기**) | P5 | t95: user-sim 허위 단정("두 노트북이 같은 주문")을 수용 → **frame-lock** → 정답 경로를 명시한 L2 deny **4회 전부 불응**(APRIME [S]) · t86: user-sim 오답 포인터 #W7739115 수용 · t103: 동일 deny 수십 회 불응·201 msgs·max_steps 소진 · t71: user-sim 오확인이 오답을 **고착**(4/4·C56) · t61: regen-유발 재선택이 user-sim 오확인과 복리(C53/C56) | ⋈ 오선택 고착·write 방향 오염·교정-피드백 무효화(스캐폴드 피드백까지 삼킴) | flip 최초 분기 **17/17 = user 발화**(C60-분기·[M]) · 단정-수용 전수 census 미계상 [?] | ❌ **미커버**: 검증기(prov·DISAMB)는 *자기* 단정만 검사 — 타자-단정 진입로는 열려 있음. gpt-4.1은 같은 표현-분산서 12/15 robust(C60-분기) = **scale/능력이 사는 축 후보** | **[M] 사례·[?] 전수** |
| **(k)** | ★신규 명명: **행위-선언 불이행** (declared-act-not-done) | P3 | t3 tr1: "modify 하겠다" 선언 후 **write 미실행** · 동류 relay-gap: calc가 4/4 정확 발화했는데 에이전트가 사용자에게 끝내 전달 안 함(t3 tr2/3·CENSUS §2b) | "했다/하겠다"가 시스템 행위와 괴리 → no-write 실패·NL 미답 | no-write 실패 floor 12 → prov 22 → router 25(C62-replay·**prov 대화-교란의 하류**) · relay-gap [M](t3 4-trial 프로브) | ❌ **미커버**: calc로 못 닫음(compute-gap 아님). 후보 = E-PLAN CP5 walk의 communicate-의무 확장 [D](CENSUS §2b) | **[M] 사례** |
| **(l)** | ★신규 명명: **환경-동역학 단정** (world-dynamics assumption) | P4 | banking task_043: **DB가 저절로 바뀌길 기다리며** `get_current_time`/재조회 무한 루프($75 잔액 대기 — env 시계도 안 흐름) = "기다리면 변한다"는 미검증 동역학 가정 | 대기-교착 → 턴 소진·REACH 실패(발견-체인 미조립 하위 모드 c) | banking 발견-체인 ~130 sims의 하위 모드(개별 계상 [?]) | ❌ **미커버**: 어떤 레버도 이 전제를 검사 안 함. 결정론 검출은 가능(동일-조회 반복 n회 = decidable 표지) [D] | **[M] 사례·[?] 빈도** |

**taxonomy 주석**
1. (a)의 세 형태(복사/발명/placeholder)는 별도 케이스가 아니라 **scale-의존 표현형**(C36) — 레버 설계엔 중요(⑤/④의
   rejected 집합이 형태를 벌함 — C39), 케이스 축엔 아님.
2. (d)·(e)·(l)은 모두 **impasse-국면**에서 집중 발생(C63 서명: impasse-시 발명형 escape) — 근인은 상류(discovery/⋈)이고
   fabrication은 그 **증상이자 무료 검출 마커**다. 레버 평가에서 "차단"과 "해결"을 분리 계상해야 한다(C61 회계:
   격리 이득 = P(교정|표적) ≠ e2e 이득).
3. (j)는 발생 주체가 다르므로(수용) 자기-생성 (a)~(i)와 분리하되, **같은 규율 결손**(단정을 검증 없이 시스템에 들임)의
   대칭형이다. 우리 검증 스택 전체가 "자기-출력"만 보고 "입력"을 안 보는 구조적 비대칭 — 이것이 (j)를 열어둔 원인.
4. 커버 요약: **닫힘 1**((a)) · **봉합 설계 1**((b)) · **검출만 3**((c)(d)(e)) · **미커버 6**((f)(g)(i)(j)(k)(l)) ·
   부분 1((h)). — "fabrication은 닫혔다"는 (a)에 한정된 참이다.

---

# PART 2 — 닫는 방법 7종 × 비용-효율 비교

## 2.1 방법 정의

| # | 방법 | 정의 | 현 상태 |
|---|---|---|---|
| ① | **scaffold-검증** (prov/PERARG·현행) | 출처선언 4지선다{GET·FIND·INFER·ASK} + 갈래별 결정론 검증기 + regen·GET폴백(C45) + rescue per-arg 수정(C65) + 게이트/notice 계열 | (a) GO·(b) 봉합 설계·(f)(g)(i) 파생 레버 [D] |
| ② | **scale-업** (32B→72B/frontier API) | 모델 tier 상향 — 사다리 실측이 근거(C36) | frontier 0.0% 실측·72B 점 미측정 |
| ③ | **thinking** (QwQ류) | test-time compute·전-궤적 CoT | 날조율 직접 측정 없음 |
| ④ | **learn-DPO** (E6′ v3) | on-policy DPO·rejected={예시값·발명형·조합형}(C39)·D7 데이터(근접-오답 배치·C43) | 데이터 게이트 미통과(C38) |
| ⑤ | **learn-SFT** | 궤적 SFT (gather 시연) | 측정된 유해(C38) |
| ⑥ | ★신규 설계: **fleet-위임** | **PERARG가 flag한 지점만** 큰 tier 격리 서브콜로 해결 — **검출=무료(결정론 술어)·해결=선별 과금**. C6 fleet(⋈/horizon 위임·저-ROI)과 다른 표적: fabrication-flag 이벤트만 | [D]·본 문서가 프로브 설계(§4.1) |
| ⑦ | **decode-제약** (xgrammar) | 디코드-시점 guided decoding — 도구명/id-집합 수준의 유한집합 제약. 채널 (f)=현 스택 유일 미점유(전 개입이 생성-후) | [D] 등재만(RESEARCH_MASTER §4 E-XGRAMMAR·스택 동결) |

## 2.2 방법 × 축 비교표

등급: [S] settled · [M] measured · [P] promise · [D] design · [?] 미측정. **[?]는 Part 4에서 실험으로 배선.**

| 축 | ① scaffold-검증 | ② scale-업 | ③ thinking | ④ learn-DPO | ⑤ learn-SFT | ⑥ fleet-위임 | ⑦ decode-제약 |
|---|---|---|---|---|---|---|---|
| **커버: (a) id** | **67→0%**(C45 [M])·e2e +3.0pp(C53 [M]) | frontier **0.0%**(C36 [M])·72B 점 [?] | [?] (QwQ 날조율 미계상) | 복사형만 −6.3pp(C37′ [M])·발명형 미벌(C39 [M])·진짜 결손 위 미시험(C38) | **유해**: tme 13→25·무조건조회 퇴화(C38 [M]) | [?] (flag→서브콜 교정률) | id-집합 제약 가능하나 **ASK-봉쇄 위험**(§1.5 Q5·write 강제 금지) [D] |
| **커버: (b) free-text** | PERARG 봉합 [D]·검출 구멍 진단 [M](C65) | frontier **0**(C24 [M]) | [?] | [?] | [?] (유해 추정) | [?] — free-text는 서브콜도 "조회 지시"만 가능(값 발명 불가) | ✗ 원리상 불가(자유텍스트=제약 집합 없음) |
| **커버: (c) 도구명** | 검출 ✅·**회복 실패**(61 sims·deny 후 못 살아남 [M]) | [?] (banking frontier 도구명 fab 미계상) | [?] | [?] | [?] | [P] 후보: flag→big-tier가 KB 재검색·정명 반환 | ★**정확 표적**: 유한집합·ASK 불필요=위험 최소(E-XGRAMMAR [D]) |
| **커버: (d)(e) 상태·능력 단정** | **차단만**(C63 [M])·해결 ✗ | [?] (frontier는 t20 status-혼동 없음 — 부분 시사 [M]) | [?] | ✗ 표적 미정의 | ✗ | [P] 후보: impasse-flag(deny 노트)→big-tier 재계획 서브콜 | ✗ |
| **커버: (f)(g)(h) 발화-주장** | (f) over-fire 44.3%→기본 off [M] · (g) NL-POLICY-PROV [D] · (h) E-PLAN 부분·불응 잔여 [M] | [?] | [?] | [?] | [?] | [P]: 발화-주장 검증은 semantic-비교 필요 → big-tier 판정 서브콜 후보 | ✗ |
| **커버: (i) 시간** | **A2 확장으로 결정론 폐쇄**(producer 존재) [D·ABox-only] | [?] | [?] | ✗ | ✗ | 불필요(①이 더 쌈) | 가능하나 ①이 우월(producer 호출 강제가 정답) |
| **커버: (j) 수용** | ✗ (자기-출력만 검사하는 구조적 비대칭) | **부분 시사**: gpt-4.1 같은 표현-분산서 12/15 robust(C60-분기 [M]) | [?] | [P]: paraphrase-invariance 표적(E6′ 신규 표적·재현 가능) | ✗ | [P]: user-단정 vs 기조회 사실 충돌 판정 서브콜 | ✗ |
| **구현비용** | **0**(기구현·GO)·파생 레버도 엔진-증분 무료 | 0 | 0 | 데이터 v3 재설계(D7) + 학습 파이프 | 동일 | **소**(flag 술어=기존 PERARG 재사용·서브콜 배선=`_t5c_disamb_subcall` 채널 (e) 재사용) | **소**(vLLM xgrammar·tau2→litellm per-call `extra_body`) |
| **추론시 비용(토큰/latency)** | regen 재발화 — 실측 미미(**tme 1/456**·C53 [M]) | latency는 오히려 ↓(30.5s vs 178s·C8) — 비용이 문제 | thinking 토큰 大 + `finish_reason=length` infra 결측 2.2%(C33 [M]) | 0 (배포 후) | 0 | **f_fab ≈ 0.15 이벤트/sim**(floor 70/456·C61) × ~2k tok/서브콜 ≈ 궤적 토큰의 **<3%** [추정] — C6 전면-fleet(1.15~2.3×)와 자릿수 다름 | **0 또는 음수**(제약이 디코드 공간 축소) |
| **학습비용** | 0 | 0 | 0 | **大**(GPU·리모트) + 망각/역전이 리스크(C4/M-σ) | 大 + 측정된 퇴화 | 0 | 0 |
| **운영 TCO** | on-prem 불변($0.0019/req·C8) | **~23×**(C8 [EST]: $0.044 vs $0.0019·범위 16–40×) + on-prem 이탈·감사불능 | on-prem 불변·토큰 증가분 | on-prem 불변 | 동일 | blended ≈ (1−f)·c_small + f·c_big — **f가 이벤트-희소라 ≈ on-prem + ε** [?·실측 필요] | on-prem 불변 |
| **부작용(측정치)** | **p1을 사고 p4를 판다**: prov p4 −5.3pp(C53 [M]) → **COMP 합성이 회복**(reward 0.316·db 0.375=floor·C62 [M]) · over-block 0/2650(C45) | frontier도 compliance-drop −2.2~−5.0pp(게이트 없으면·C1/[[45]]: guarantee는 scale-invariant) · 날조 0%의 p4세 [?] | **F4·F5 매도 → 순 0**(C4b [M]) · 단 passing-spurious 0 vs 47(C4d [M] — scope 규율은 삼) | 망각·역전이 [?]·off-policy면 likelihood displacement(C43+선행) | **A_notfound .31→.41·tme 13→25**(C38 [M]) | Δspurious [?]·오형식화-고착 리스크(서브콜 오답의 확신-주입) — silent-leak은 없음(검출=결정론) | **abstain-봉쇄**: 제약이 "쓸 수 없음"을 표현 못 하면 write 강제와 동형(§1.5: p≈0.44<0.5면 기대-유해) → **ASK/null 분기 필수**(C48·C58 동형 경계) |
| **도메인-전이성([[05]] A2-swap)** | ★**최상**: 엔진 도메인-일반·A2=`{인자→producer}` 매핑뿐(C45)·banking 스모크 UNI_OK([[05]] 재감사)·(i)는 ABox-only 확장 | 자명(모델 교체) — 단 TCO·on-prem 이탈이 전 도메인에 복제 | 자명 | **TBox 학습 = 도메인-일반이어야**([[11]])·D7 다양성 필수([[12]]) — 전이성은 설계 조건이지 자동 아님 | 동일 + 퇴화 정책이 전이(무조건조회) | 검출=엔진·해결=tier 스왑 — A2 불변·전이 구조 ①과 동일 | 제약 집합=env 스키마서 자동 도출(A2조차 불필요) — 전이성 최상 후보 [D] |
| **증거등급(종합)** | **[M]~[S]** ((a) 한정·나머지 [D]) | [M] (사다리·(a)(b) 한정) | **[?]** (fabrication 직접 측정 0) | [M] (부정적 부분증거)·표적 위 [?] | **[M] NO-GO** | **[?]** (전 칸 미측정 — §4.1) | **[D]** |

## 2.3 방법 × 케이스 커버리지 매트릭스 (압축)

✅=측정된 해결 · ◐=부분/차단만 · ▷=설계/후보([D]/[P]) · ✗=원리상 부적합 · ?=미측정

| 케이스 | ①scaffold | ②scale | ③think | ④DPO | ⑤SFT | ⑥fleet | ⑦decode |
|---|---|---|---|---|---|---|---|
| (a) id | ✅ C45/C53/C62 | ✅ C36(23×) | ? | ◐ C37′/C39 | ✗ C38 | ? | ▷(위험) |
| (b) free-text | ▷ C65 | ✅ C24(23×) | ? | ? | ✗ | ? | ✗ |
| (c) 도구명 | ◐ 검출만 | ? | ? | ? | ? | ▷ | ★▷ |
| (d) DB-state | ◐ 차단만 | ◐ 시사 | ? | ✗ | ✗ | ▷ | ✗ |
| (e) 자기-능력 | ◐ 차단만 C63 | ? | ? | ✗ | ✗ | ▷ | ✗ |
| (f) 발화-수치 | ◐ off(over-fire) | ? | ? | ? | ? | ▷ | ✗ |
| (g) 발화-정책 | ▷ 부록W | ? | ? | ? | ? | ▷ | ✗ |
| (h) 커버리지 환각 | ◐ E-PLAN 불응 | ? | ? | ? | ? | ▷ | ✗ |
| (i) 시간 | ★▷ ABox-only | ? | ? | ✗ | ✗ | 불필요 | ◐ |
| (j) 수용 | ✗ 구조적 | ◐ C60-분기 | ? | ▷ E6′ | ✗ | ▷ | ✗ |
| (k) 선언-불이행 | ▷ CP5 | ? | ? | ? | ? | ? | ✗ |
| (l) 동역학 단정 | ▷ 검출가능 | ? | ? | ✗ | ✗ | ▷ | ✗ |

---

# PART 3 — ★전 실패-클래스 통합 배분표 (결론부 · 특허 B[배분] 실시예 본체)

> **생성 규칙 = RESEARCH_MASTER §1.5 결정절차** — 이 표는 그 절차(Q1 decidable→결정론 scaffold · Q1b 이미-집행→레버 아님 ·
> Q2 부하→결정론 controller · Q3 symbolic→thinking · Q4 scale-민감→scale/fleet · 잔여→learn[미검]/경계)를 **전 실패-클래스에
> 실행한 결과**다. fabrication(Part 1)을 넘어 선택-오류·수량/coverage·조건체인·형식화/바인딩·horizon/reach까지 —
> 우리 corpus의 전 실패 클래스를 한 좌표계에 놓는다. 중복 셀은 Part 1/2 참조. **각 행 = 3단 배분: ① 결정론이 사는 몫 ·
> ② learn으로 이관되는 몫(정확한 학습 표적 명세) · ③ scale로 이관되는 몫(어느 tier가 사는지 실측).**

## 3.0 대책 어휘 표준화 (사용자 명명 · 이하 표의 기법명)

| 표준 명칭 | 정의 | 엔진 구현(현행) |
|---|---|---|
| **출처-제공-요구** | 인자 출처 4지선다{GET·FIND·INFER·ASK} 선언 강제 + 갈래별 결정론 검증 + regen | prov(C45)·PERARG(C65 [D]) |
| **선택-객관식화** | 후보를 결정론으로 열거 → 모델은 *선택만*(생성→선택 바인딩 이동) | DISAMB 서브콜(C59/C60)·CALC-ANCHOR(부록 W [D]) |
| **선택-갯수-확인** | \|C\|·planned/examined 원장으로 개수·완결을 결정론 대조 | E-PLAN ledger·eamb census(C55) |
| **read-강제** | 미조회 대상에 대한 write를 deny(읽기만 강제·쓰기 강제 절대 금지) | E-PLAN L2(cap=4) |
| **silent-치환** | 인자만 제자리 교체·대화/턴 완전 불변·replay-clean | P-A/P-B(C62-replay)·P2 원리-디폴트 |
| **결정론-주석** | 에이전트-기조회 데이터 위 결정론 계산을 부착(규칙0 클린) | calc/nested(argmax_where 등·CENSUS §2a) |
| **notice** | write 前 정책-사실 고지를 강제(에이전트 판단 경유·write 강제 0) | NOTICE-PERGATE(엔진 GO·G8 A2 미부착) |
| **decode-제약** | 디코드-시점 유한집합 제약(guided decoding) | E-XGRAMMAR [D·미점유 채널] |

## 3.1 ★통합 배분표

표기: 상태 = **GO**(실측 통과)/**부분**/**설계**([D])/**공백**. 비용등급 = 0(기구현·엔진증분)/저/중. TCO 참조 = C8(~23×).

| 실패-클래스 | ① 결정론 대책 (기법·상태·비용·증거) | ② learn-이관분 (학습 표적 명세·1회 학습비) | ③ scale-이관분 (어느 tier가 사나·실측) | 최저비용 승자 | 결정론이 멈추는 이유 (1줄) |
|---|---|---|---|---|---|
| **F-(a) id 날조** | 출처-제공-요구 — **GO**·비용 0·C45(67→0%)/C53(+3.0pp)/C62(p4 회복) | D7 데이터(근접-오답 창 배치 C43 + 발명형·조합형 rejected C39 + on-policy) — 게이트 미통과(C38)·비용 중 | frontier가 삼(0.0%·C36 [M])·**23×** | **결정론** | 멈추지 않음 — 완전 decidable(문맥-실재 검사) |
| **F-(b) free-text 날조** | 출처-제공-요구+PERARG — **설계**(구멍 2점 코드-확정)·비용 0·C65/C24 | — (표적 미정의) | frontier 0(C24 [M])·23× | **결정론**(X7 조건부) | 검증은 decidable·**값 생성은 불가**(조회-지시 반환까지) |
| **F-(c) 도구명 날조** | decode-제약 — **설계**·비용 0(E-XGRAMMAR) / prov 검출은 GO·회복 실패 61 sims [M] | 발견-체인 조립 학습 — 표적 미명세 [?] | [?] (banking frontier 미계상) | **decode-제약**(X4 조건부) | 이름 집합=decidable — **회복(체인 조립)은 능력** |
| **F-(d)(e) 상태·능력 단정** | 게이트 하류 차단 — **GO**(C63·위반0)·전제-검증은 **공백** | — | frontier 시사(t20 status-혼동 없음·`T27` 부록 [M]) | 차단=결정론·해결=**공백(정직)** | "안 읽고 단정" 검출은 decidable — 단정 *내용*의 진위 판단은 semantic |
| **F-(f) 발화-수치** | NL-NUM-PROV — **NO-GO**(over-fire 44.3%·부록 Z) / 결정론-주석(calc-NL)이 compute-gap만 | — | [?] | **공백** | 발화 속 수치의 '주장성' 판별 = semantic(over-fire가 그 증거) |
| **F-(g) 발화-정책** | NL-POLICY-PROV — **설계**(트리거-반전·부록 W)·비용 0 | — | **gpt-4.1-tier가 삼**(t57·`T27` 부록 [M]) | 결정론(X5) vs scale 경합 — X5가 판정 | 트리거(토큰∧기조회-pm 불일치)=decidable·주장 의미론은 경계 |
| **F-(h) 커버리지 환각** | read-강제 L2 — **부분**(피드백 정확·불응 잔여: deny 4회/수십 회 무시 [S]) | ★**verifier-신뢰 학습**(결정론 deny 피드백에 순응하는 성향 — E6′ 확장 후보·명세 미작성) | [?] | 결정론 + 잔여는 learn/⑥fleet | 피드백 *순응*은 모델 prior([[42]] 동형 — 프롬프트-채널의 한계가 deny-채널에도) |
| **F-(i) 시간 날조** | 출처-제공-요구 A2 확장(producer=get_current_time) — **설계·ABox-only**·비용 0 | — | 불필요 | **결정론** | 멈추지 않음 — producer 존재 = 완전 decidable |
| **F-(j) confabulation-수용** | **공백**(검출 census=X6 先·충돌 표지는 decidable) | ★**paraphrase-invariance + 기권-선호**(등가-표현 변주에 동일 결정·타자-단정에 검증-먼저 — E6′ C60 표적·재현 가능) | gpt-4.1 같은 분산서 12/15 robust(C60-분기 [M]) | **미결정** — X6→X1 순 | 타자-단정의 진위 = semantic·모순-검출 표지만 decidable |
| **F-(k) 선언-불이행** | 선택-갯수-확인(CP5 communicate-의무 확장) — **설계**(CENSUS §2b) | — | [?] | 결정론(잠정) | 의무 추출=근사-decidable·NL-이행 판정의 결정론 한계 정직 평가 先 |
| **S-1 선택: order-⋈** | 선택-객관식화(DISAMB 서브콜) — **GO-조건부**(p4 +1.8pp·t61 복구·C60) + silent-치환이 write-소멸 결함 수리(C62-replay·T5-C) — 단 **id-형 열거는 역효과**(C61③: 판별정보 없음) | paraphrase-invariance(C60·flip 17/17=user 발화 분기) — E6′·비용 중 | **scale이 삼**(C51: F3⋈ 3.1→0.9 단조)·잔여 진짜 경계 7.2%(C61) | 결정론(내용-매칭 열거) → 잔여 경계 | 내용-매칭 정보가 없는 대상엔 열거도 무효 — 진짜 semantic 경계 |
| **S-2 선택: 변형-⋈(F2)** | 선택-객관식화 — **GO-격리**(C59: .116→.658 +31pp) + CALC-ANCHOR(**설계**·부록 W: anchor 차원별 유지-argmax 열거) | D7(근접-오답 변형 배치) | **챔피언-tier(397B+think)만**(C51③ F2=0.4%)·동-scale thinking 무효(C56: .145=.143)·gpt-4.1도 잔여(F2 9~16건) | **결정론(열거)** — scale 대비 압도적 싸다 | 열거 후 잔여 ~.34 = 미결정/의미(C59)·frontier 공유 |
| **S-3 선택: payment/원리-디폴트** | silent-치환 P2 — **GO**(t61 4/4·v25e) + 정책-강제 슬롯=결정론 규칙 **1.000**(C58 FORCED 164/164) | — | 프롬프트·열거로 안 닫힘(C61: 정책 포함 열거도 .38) — scale 데이터 없음·결정론이 이미 0비용 해결 | **결정론** | 사용자-선택 슬롯(.840)만 잔여 = 진짜 선호 → **ASK 관할**(C48 위계) |
| **Q-1 수량/coverage** | read-강제 L2 + walk + 나열-힌트(t81 직격 회복·aprime2 [M]) + 선택-갯수-확인(ledger) + 품목-coverage 가드(conflation 수리·2026-07-11 커밋 공지·**본 문서 미검증** [D]) | (이관분 없음 — 게이트 축·[[45]] 부하-내성 학습은 thesis 역행) | ★**못 삼**: F4 scale-invariant(C1 [S]·17≈16·thinking은 악화 C4b·frontier도 banking coverage MISS 20~26%·C52) | **결정론 확정** | write-강제만 금지(§1.5 Q5: p<0.5면 기대-유해) — 그 선까지가 전부이자 충분 |
| **CC-1 조건체인(t57형)** | notice G8 — **NO-GO**(over-block 16.6%·부록 Z) → NL-POLICY-PROV(**설계**) + EXCLUSIVITY deny-once+ask(**설계**·CENSUS §3·t27형) | — | **gpt-4.1-tier가 삼**(t57·`T27` 부록 [M]) | 경합: 결정론-설계(비용 0·X5) vs scale(23×) — X5 실측이 판정 | 체인의 조건 *평가* = 대화-semantic(C50 동형: DB-decidable 아님) |
| **FB-1 형식화/참조-바인딩(t20형)** | FORMALIZE-EXEC — **공백 실측**(V0 full-EM 0.00·constraints 0%·부록 Z: 미편입) — 결정론 실행기는 준비됐으나 입력(형식화)이 안 됨 | ★형식화 자체가 학습 표적 후보(문맥-의존 constraints 추출 — op/field는 0.68로 살아있음·미설계) | **frontier도 공유**(t20: gpt-4.1 0/4·claude-3.7 1/4)·챔피언-tier 후보/미결정 성분(C57 scatter 최상위)·**E-REF P0/P1/P2가 경계 측정 중** | **미결정(정직)** — 현 최심 잔여 | 형식화가 안 되면 결정론 실행기가 받을 입력이 없다(NL→formalize=LLM 몫·[[10]]) |
| **HR-1 horizon/reach(banking 절차조립)** | plan/execute 분리 controller — **부하 실증**(C14 [M]·banking G2=부하: KB검색 119/125 후 *조립* 실패·C54) + read-강제 | (이관분 없음 — 부하-내성 SFT는 결정론 controller 일 침범·[[45]]) | ★**scale의 유일한 정당 구매물**(C5 [S-lit]: horizon) — 싸게는 fleet(단 C6: 전면-fleet 1.15~2.3×·tau2-retail 잔여엔 저-ROI) | **controller 先 → 잔여 복리만 scale** | 부하(Q2: p_iso>p_traj)까지가 결정론 몫 — 잔여 복리 p^H는 어떤 scaffold도 못 삼 |

## 3.2 배분 요약 (표의 독법 · 특허 B 연결)

1. **17행 중 결정론 승자 확정 6**(F-a·F-i·S-3·Q-1 + 차단-한정 F-d/e·부분 F-h) · **결정론 잠정(검증 대기) 6**(F-b·F-c·
   F-g·F-k·S-1·S-2) · **공백/미결정 4**(F-f·F-j·FB-1·CC-1 경합) · **scale 고유 몫 1**(HR-1 잔여 복리).
2. **scale-이관 열의 실측 계층이 곧 knee 곡선**: gpt-4.1-tier가 삼(t57·조건체인) < 챔피언-tier만(F2 변형) <
   frontier도 공유(t20·⋈ 잔여 7.2%·coverage) — "어느 tier를 사야 하나"가 클래스별로 **측정돼 있고**, 그 위에 결정론
   대책의 비용 0 대안이 행마다 명시된다. **이 표 자체가 특허 B(기능→최저비용 레버 배분·knee) 청구의 실시예 본체**이며,
   Part 2의 방법×케이스 매트릭스는 그 fabrication-부분집합이다.
3. **learn 열이 비어 있지 않은 행은 4개뿐**(F-a·F-h·F-j/S-1·FB-1) — learn의 정당한 표적은 "결정론이 semantic에서 멈추고
   scale이 너무 비싼 좁은 틈"(verifier-신뢰·paraphrase-invariance·형식화)이지 규율 일반이 아니다. 전부 D7-급 데이터
   재설계가 선행 게이트(C38)·[[13]] 순서(scale→학습→scaffold 최후가 아니라 **흡수 우선순위**) 준수.
4. **"결정론이 멈추는 이유" 열의 패턴**: 멈추는 지점은 예외 없이 (i) semantic 판정(진위·의미·선호) (ii) 모델-prior
   (피드백 불응) (iii) 형식화 자체 — 셋 다 §1.5의 Q3/Q4/경계 분기와 일치. **표가 결정절차의 실행 결과라는 주장의 검증.**

---

# PART 4 — 미측정 칸 전수 + 최소-비용 실험 설계

## 4.0 미측정 칸 목록 (우선순위순)

| # | 빈 칸 | 채우는 실험 | 비용 |
|---|---|---|---|
| G1 | ⑥ fleet-위임 전 칸 (교정률·Δspurious·토큰) | **X1 격리 프로브(핵심 신규·§4.1)** | 무료(로컬 arm) + 선별 소액(frontier arm·승인) |
| G2 | ③ thinking의 날조율 (a)(b) | X2 | **무료**(기존 gz 재분석) |
| G3 | ② 72B 사다리 점 (C36의 32B↔frontier 사이 불연속 위치) | X3 | 무료(리모트 GPU) |
| G4 | ⑦ 도구명 decode-제약 효과 (c) | X4 | 무료(격리 재디코드) |
| G5 | (g) 발화-정책 census + NL-POLICY-PROV V0 | X5 | 무료 |
| G6 | (j) confabulation-수용 전수 빈도 + decidable 검출률 | X6 | 무료 |
| G7 | (b) PERARG 봉합 검증 | X7 | 무료 |
| G8 | ② scale의 p4세 — frontier 0% 날조에 robust 세금이 없는가 | X8 | 무료(공개 데이터) |
| G9 | ④ DPO 표적 위 시험 (D7 데이터 게이트) | X9 | 중(마지막) |

## 4.1 ★X1 — fleet-위임 하이브리드 격리 프로브 (핵심 신규 설계)

**아이디어**: PERARG/prov/게이트-deny가 flag한 결정점만 큰 tier 서브콜로 해결. **검출은 이미 무료·결정론으로 작동
중**(prov deny·게이트 deny 노트=C63 무료 마커) — 남은 질문은 "**flag된 지점을 큰 tier가 32B 자신(regen)보다 얼마나 더
잘 교정하나**"뿐이다. C6의 전면-fleet 저-ROI 판정은 horizon/⋈-위임 기준(f≈0.065 토큰-비중·1.15~2.3×)이었고,
**fabrication-flag는 이벤트-희소**(floor 70건/456 sims·C61)라 비용 구조가 다르다 — 별도 실측이 정당하다.

**무료 케이스 생성 — flag 지점은 오늘 데이터에 전부 있다**:
- retail: `sim_results/prov_e2e_retail_t4.results.json.gz`·`comp_retail_t4.results.json.gz`의 prov-regen/deny 이벤트 +
  floor fab 이벤트 70건(C61·`eamb7_dilution_census.py` 재사용)
- banking: `sim_results/banking_lever_fit_percase.json`의 FABTOOL(61)·TIMEFAB(59) 플래그
- 프로토콜 재사용: `e11a_isolated_probe.py`(정보-맞춘 단일턴 격리·C40)·`ecomp_iso_probe.py`(E-ISO 3단·C61)

**설계**:
1. **표본**: flag 결정점을 케이스타입별 층화 추출 — (a) id 70 · (b) free-text 5(전수) · (c) 도구명 61 · (e) impasse-deny
   31 · (i) 시간 59. 각 지점의 **에이전트가 실제 갖고 있던 문맥**을 동결(정보-맞춘 격리 — §1.4 측정 규율).
2. **arm 3개** (동일 프롬프트 = C45 4지선다 + "flag된 인자의 출처를 선언하고 값을 확정하라"):
   - **arm S** (self·기준선): 32B가 자기-교정 — 현행 prov regen과 동형. 무료.
   - **arm B** (big-로컬): **72B 서브콜**. 리모트 GPU([[30]])서 Qwen2.5-72B-Instruct GPTQ-int4(가중치 ~40GB) —
     48GB 카드 1장이면 KV 빠듯·2장 TP면 안정. **서버 스펙 확인 = 선행 항목**(불가 시 arm F로 대체). 무료.
   - **arm F** (frontier API): 동일 지점 서브콜. ~150~250 지점 × ~2k tok ≈ **총 $1 미만** — 단 [[09]]:
     arm S/B 결과 확인 후 **승인·최소 scope**로만.
3. **지표**:
   - **P(교정|flag)** per 케이스타입 — arm 간 차가 곧 "scale이 이 케이스를 사는가"의 직접 측정(§1.5 Q4를
     fabrication에 재실행).
   - **Δspurious**: 서브콜이 원래 맞던 값을 뒤집은 수(≤0 게이트 — 제1원리 반대편 계측).
   - 토큰/지점·f_fab(이벤트율) → blended TCO 셀 실계산(§2.2 [?] 칸 채움).
   - e2e 상한 = **P(표적) × P(교정) × P(타결함 없음) − 부작용**(C61 회계 공식 — 격리 이득의 희석을 선반영).
4. **GO 조건**: P(교정)_B − P(교정)_S ≥ +15pp(케이스타입별) ∧ Δspurious ≤ 0 ∧ $/해결 ≪ 순수-frontier 전환비(~23×·C8).
   특히 **(c) 도구명·(e) impasse**가 유망 표적 — 32B regen이 실측 실패한 지점(61 sims 못 살아남·deny 4회 불응)이라
   self-arm 천장이 낮다.
5. **[[05]] 적합성**: 검출=기존 엔진 술어(도메인-일반·변경 0)·A2 불변·해결=tier 스왑 = §1.6 scale/fleet 행의 위임 3조건
   ((i) 큰 tier가 측정상 더 잘함 — 본 프로브가 측정 (ii) 격리 서브콜 토큰-쌈 — 이벤트-희소로 충족 (iii) 이산 결정점 —
   flag가 정의) 검정 그 자체.
6. **경계 예상(정직)**: (j) 수용·(d) 상태-단정은 flag 자체가 없어 이 프로브 밖(X6이 검출률부터) · (b) free-text는
   서브콜도 값을 발명할 수 없어 "조회 지시" 반환까지만 — 교정률이 아니라 **회복-지시 순응률**로 지표 대체.

## 4.2 나머지 실험 (전부 기존 자산 재사용)

| # | 설계 | 재료 | GO/산출 |
|---|---|---|---|
| **X2** | QwQ 456-sim gz에 미조회-날조 census 스크립트 적용 → **thinking 날조율** vs base 6.7% | `dbonly_forensic.py` §10.3 로직 + E2 QwQ gz(clean 0.547) | ③ (a) 칸 확정 — C36 사다리에 "동-scale thinking" 점 추가 |
| **X3** | fab 결정점 70건을 72B로 격리 replay(E-ISO 프로토콜) → **72B 날조율** | X1 arm B와 좌석 공유(같은 런) | C36 사다리 72B 점 — 32B 6.7%→frontier 0.0% 불연속이 어디서 꺾이나 |
| **X4** | banking 61 sims의 첫 도구명-fab 호출 지점을 **guided decoding**(도구명=KB-실재 집합 제약)으로 재디코드 vs 자유 디코드 — 격리 | vLLM xgrammar·tau2→litellm `extra_body`(E-XGRAMMAR 배선안) | ⑦ (c) 칸: 제약-디코드 정명률 + **ASK-봉쇄 계측**(제약이 기권을 막은 수 — 필수 반대편) |
| **X5** | COMP 456 전수서 "정책-사실 주장"(refund+결제수단 토큰 ∧ ≠기조회 pm) census — 부록 W V0 그대로 | comp gz + A2 어휘 | (g) 빈도 확정 + NL-POLICY-PROV over-fire(passing) — 44.3% 같은 함정 선검증 |
| **X6** | user-sim 단정 vs 에이전트-기조회 사실의 **충돌 census**: user 발화의 id/상태 주장 ∧ 기조회 tool 출력과 모순 = decidable 표지. C60-분기 17쌍 + t71/t86/t95로 검증 | 기존 gz 전수·`ecomp_divergence_census.py` 확장 | (j) 전수 빈도 + **decidable 검출률** — 검출이 서면 ⑥의 표적으로 승격 |
| **X7** | PERARG 단위(다중-fab 순회·`#`정규화) + v25e t17 4-trial 오프라인 재현(fab 검출이 address1에 닿는지) — C65 검증 계획 그대로 | `t2_gate_patch.py` 수정 2점 | (b) 봉합 [D]→[M] |
| **X8** | frontier 공개 궤적([[47]] 경로: baseline 공개·pass=submission.json)서 **pass^k robust 분석** — 날조 0%인 frontier가 p4서 우리 COMP(0.316) 대비 세금을 내는가 | 공개 데이터·무료 | §5 긴장 해소의 결정 증거: "규율의 p4세"가 설치-방식(scale vs scaffold) 의존인지 |
| **X9** | E6′ D7 데이터 게이트: 근접-오답 id 배치 합성서 **base가 tau2-수준 날조를 재현**해야 학습 착수(C38 필수 조건) → 통과 시 on-policy DPO | E6′ 설계 그대로 | ④ 표적 위 첫 시험 — **순서 최후**([[13]]: scale→learn→scaffold 역순 아님·learn은 무망각 조건부) |

**실행 순서 권고**: X2·X5·X6·X7(순수 무료 재분석·병렬) → X1 arm S/B + X3·X4(로컬 GPU) → X8 → X1 arm F(승인) → X9(게이트 통과 시만).
현 스택 동결(S4/S5)과 양립: 전부 오프라인·스택-불변([[09]]·NEXT_LEVER §3 동일 논리).

---

# PART 5 — fabrication 케이스별 최저비용 승자 권고표 (현재 증거 기준 · Part 3의 fabrication-행 상세판)

| 케이스 | 승자(현 증거) | 근거·비용 | 등급 | 차점/보험 |
|---|---|---|---|---|
| (a) id 날조 | **① scaffold-검증(prov+COMP)** | 67→0%·pass비용 0·e2e +3.0pp·p4는 COMP가 회복 — scale(23×·C8)과 동일 결과를 ~0비용에 | **[M]~[S] 확정** | ② scale(이미 frontier 0% 실측) |
| (b) free-text | **① PERARG** (잠정) | 무료·decidable·구멍 2점 코드-확정 — X7 통과 조건부 | [D]→X7 | ② scale(C24 실측·23×) |
| (c) 도구명 | **⑦ decode-제약** (잠정) | 유한집합=ASK 불필요·위험 최소·구현 무료 — ①은 검출 후 회복 실패 실측(61 sims) | [D]→X4 | ⑥ fleet(X1이 판정) |
| (d) DB-state 단정 | **승자 없음(정직)** — ① 하류 차단만 | 상태-전제 검증 레버 미존재. 상류=discovery/E-PLAN | [?] | ⑥ 재계획 서브콜(X1 확장) |
| (e) 자기-능력 | **① 차단 + 상류 E-PLAN** | 게이트=정상작동(C63)·근인은 fabrication 밖(discovery) | [M] 차단 | ⑥ impasse-위임(X1) |
| (f) 발화-수치 | **승자 없음** — 현행 술어 over-fire 44.3% | 표적 실측으로만 재론(부록 Z 판정 준수) | [M] NO-GO(현행) | 술어 정밀화 후 ①-변형 |
| (g) 발화-정책 | **① NL-POLICY-PROV** (잠정) | 트리거-반전(허위-주장 시만)·decidable·무료 — X5 V0 조건부 | [D]→X5 | — |
| (h) 커버리지 환각 | **① E-PLAN L2** (부분) + 불응 잔여는 ⑥ 후보 | 피드백은 정확·순응이 병목(deny 4회/수십 회 불응) — 불응 지점이 정확히 big-tier 위임 후보 | [M] 부분 | ⑥(X1)·② |
| (i) 시간 | **① A2 확장** | producer 존재=결정론 폐쇄·ABox-only·무료 — 전 방법 중 유일하게 완전-decidable | [D·확실성 높음] | — |
| (j) confabulation-수용 | **미해결 — X6 검출 census 先** | 유일한 P5 진입로·현 스택 구조적 사각. scale 시사(C60-분기 gpt-4.1 12/15)와 ④ paraphrase-invariance(E6′)가 장기 후보 | [?] | ②/⑥/④ |
| (k) 선언-불이행 | **① CP5 communicate-의무** (잠정) | E-PLAN 좌석 공유·[D]·격리 프로브 先(CENSUS §2b) | [D] | — |
| (l) 동역학 단정 | **① 결정론 검출**(동일-조회 반복 표지) 신설 후보 | decidable 표지 존재·레버 미설계 | [?] | ⑥ |

**총평**:
1. **① scaffold-검증이 7/12 케이스의 승자 또는 유일 후보** — 단 그 중 [M] 확정은 (a) 하나뿐이고 나머지는 [D].
   "scaffold가 fabrication을 닫았다"의 정직한 형태는 **"id-날조를 닫았고, 나머지는 같은 원리의 미검증 파생"**이다.
2. **③ thinking과 ⑤ SFT는 어느 케이스에서도 후보가 아니다** — ⑤는 측정된 유해(C38), ③은 측정 부재 + 인접 증거가
   전부 비관(C4b 순0·C56 |C|≥2 무효·F5 매도). X2가 무료로 확정한다.
3. **② scale은 (a)(b)를 실측으로 닫지만 ~23× TCO + on-prem 이탈 + guarantee는 여전히 못 삼**([[45]]: frontier도
   compliance-drop) — "닫을 수 있다"와 "최저비용"은 다르다. scale의 정당한 자리는 ⑥을 통한 **선별 구매**다.
4. **⑥ fleet-위임은 전 칸 미측정이나 유일하게 '검출-무료·해결-선별과금' 구조** — 이벤트-희소성(f_fab≈0.15/sim) 때문에
   C6의 저-ROI 판정이 적용되지 않는 별개 가설. X1이 최소비용으로 판정한다.

---

# PART 6 — "scale이 작아서 생기는 문제"인가: C36 단조 vs C45 설치-가능의 긴장

**긴장의 두 극**:
- **C36 [M]**: 미조회 날조율이 scale에 단조 감소(38.8→7.0→6.7→**0.0%**)하고 형태도 scale이 바꾼다(복사→발명→소멸).
  표면 독해 = "fabrication은 scale 결손·클수록 낫는다."
- **C45/C53 [M]**: 같은 32B에 출처선언 인터페이스를 붙이면 날조 67→**0%** — 학습 0·DB주입 0·pass비용 0. 표면 독해 =
  "fabrication은 인터페이스 결손·scale 무관하게 닫힌다."

**해소 — 규율은 scale-내재가 아니라 이전-가능(transferable)이다**:
1. **기전이 능력이 아니다**: 날조=정박 치환(C43 — 문맥 인접 id의 edit≤2 변형·contextual entrainment+induction head·
   C43+선행 [S-lit]). 읽은 걸 잊는 게 아니라 **안 읽고 옆 것을 집는다**(read 0–3회 27%→6+회 2.1%). 이것은 "무엇이
   검증된 주장인가"를 표상하는 규율의 부재이지 표상 능력의 부재가 아니다 — C42가 결정 증거: **base 7B조차 짧은
   문맥선 4지선다를 완벽 해결·fabricate 0.00**. 능력은 있다. 실행-중 규율이 없다.
2. **scale 사다리는 규율의 '설치 경로' 중 하나일 뿐**: frontier의 0%는 [[43]]의 역-사례가 아니라 동-사례다 — 대형
   모델은 사전학습 prior-override 능력([[42]] scale-emergent)과 RLHF-설치된 기권/확인 행동으로 같은 규율을 **가중치에**
   설치받았다. C45는 같은 규율을 **인터페이스에** 설치한다(선언→결정론 검증→재발화). 설치 위치가 다를 뿐 설치물이 같다.
   ⇒ "scale이 작아서 생기는 문제"의 정확한 형태: **"규율이 설치 안 된 채 출고되는 것이 소형의 기본값"**(C36 단조는
   설치도의 단조)이지, 소형에 설치 불가능하다는 뜻이 아니다(C45가 반례).
3. **단 이전은 공짜가 아니다 — 제1원리**: scaffold-설치 규율은 p4를 판다(C53 −5.3pp·regen-유발 재선택) → 합성(COMP)이
   회복(C62). scale-설치 규율의 등가 세금은 **미측정**(X8) — frontier가 p4서 세금 없이 0%라면 "설치 방식 간 비용 차"가
   scale의 진짜 구매물이 된다. 이 칸이 P1의 crossover 논증을 완성하거나 제한한다.
4. **남는 scale-단서 2개(정직)**:
   - (c)(j)에서 scale-우위 *시사*가 있다(banking 도구명은 발견-체인 조립력과 얽힘·C60-분기 gpt-4.1 robust 12/15) —
     이 케이스들은 규율이 아니라 **능력(semantic 판정·절차 조립)과 혼합**이라 인터페이스 이전이 (a)처럼 깨끗하지
     않을 수 있다. X1/X6이 분리한다.
   - C36의 형태 변화(복사→발명)는 rejected-집합 설계(④)가 scale별로 달라야 함을 뜻한다(C39) — learn 경로의 이전은
     scale-조건부다.

**결론(현 증거)**: fabrication은 **scale-내재 결손이 아니라 미설치 규율**이며, 최저비용 설치 경로는 (a)에서 실증된
인터페이스-설치(①)다. scale은 같은 것을 ~23×에 사는 대체 경로이고, 그 정당한 역할은 ⑥을 통한 **잔여-선별 구매**로
좁혀진다. 이 명제의 반증 가능 지점 = X1(위임 우위면 능력-혼합 케이스는 scale 몫)·X8(scale의 무-세금이면 crossover 약화).

---

# PART 7 — 논문·특허 연결

1. **P1 *What Scale Buys* — fabrication-절의 골격이 이 문서다**: C36 사다리("scale이 사는 것") × C45/C53/C62("같은
   것을 non-scaling 레버가 pass비용 0에 삼") = [[46]]의 core 노벨티인 **lever-배분 + crossover**의 fabrication 실시예.
   Part 6의 긴장-해소가 절의 논증 구조·X8이 마지막 빈 칸. (C2 compliant-pass crossover와 병렬 배치: guarantee에 이어
   fabrication도 "scale-invariant하게 사는 법"의 두 번째 기둥.)
2. **특허 B(배분·knee) 실시예**: Part 2의 방법×케이스 배분표 + Part 4의 승자표 자체가 "기능→최저비용 레버 배분"
   청구의 구체 실시예(fabrication 서브타입별 배분). Part 1 taxonomy는 특허 taxonomy 부록 X(채널축 (a)~(f)·로컬 전용
   [[32]])와 직교하는 **대상축** — 채널×대상 격자가 배분 청구의 좌표계. E-XGRAMMAR(⑦)=채널-분화 제3실증 축(§4 큐 명기).
3. **특허 D 후보(출처선언·C45)**: Part 1의 (a)(b)(f)(g)(i)가 전부 "출처 선언+검증" 원리의 대상 확장 — 청구 범위 설계
   재료. present 선행과의 차별(DB 안 읽음)은 C34/C45 그대로.
4. **[[32]] 준수**: 특허 세부·덱은 `_cdp_private_local` — 본 문서는 repo-안전 포인터만 둔다.

---

## 부록 A — provenance·재현

- 수치 전거: RESEARCH_MASTER §3 원장(C11·C12·C24·C29·C30·C36·C37′·C38·C39·C41·C42·C43·C45·C47·C48·C53·C56·C60·C61·
  C62·C63·C64·C65·C4b·C4d·C6·C8) · `BANKING_FLOOR_LEVER_FIT_2026_07_11`(61/59) · `APRIME_REGRESSION_FORENSIC_2026_07_11`
  (t86/t95/t103 [S]-표) · `T27_T103_PERSTEP_2026_07_11`(t20 부록·ORDER-IRREVERSIBLE) · `RETAIL_FULL_FAIL_CENSUS_2026_07_11`
  (C64) · `CENSUS_LEVERS_DESIGN_2026_07_11`(C65·§2b) · `NEXT_LEVER_GEN_DESIGN_2026_07_11` 부록 W/Z ·
  `FLEET_FUNCTION_DELEGATION_DESIGN_2026_07_07` §4/4b(rev) · `scripts/distill/TCO_TABLE_DESIGN_2026_06_23`(C8).
- (f) V0 수치(208발화·44.3%)는 본 문서 작성 시 `nlnum_offline_census.py`를 `sim_results/comp_retail_t4.results.json.gz`에
  재실행해 독립 재현함(2026-07-11): passing 128/289 sims·발화 208·금액 322 / failing 88/167·발화 164·금액 252.
- 재사용 스크립트: `e11a_isolated_probe.py` · `ecomp_iso_probe.py` · `eamb7_dilution_census.py` · `dbonly_forensic.py` ·
  `ecomp_divergence_census.py` · `nlnum_offline_census.py`.
- 갱신 규율: X1~X9 결과는 먼저 RESEARCH_MASTER §3에 C-번호로 영속 → 본 문서 해당 [?] 칸 갱신. **[D]/[?] 칸을 [M]처럼
  인용 금지.**
