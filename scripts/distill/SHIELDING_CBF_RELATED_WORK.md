# SHIELDING / SAFETY-FILTER 관련연구 축 — novelty 검증 5축 추가 (2026-06-10)

> **지위**: §10-6 novelty 적대검증(2026-06-06, 4축: 프로세스그래프·도구그래프·KG·distill전이)의 **5번째 축 추가** — safe-RL shielding / 제어 safety-filter / LLM-agent runtime-enforcement 계열. 동기 = "propose-then-shield는 제어에서 기정립 패러다임"이므로 미인용 시 공격 표면, 선제 인용 시 "검증된 패러다임의 계승" 포지션 근거.
> **검증 방법**: 2026-06-10 WebSearch/WebFetch 1차 fetch (ShieldAgent는 abs 정독 5문항 체크).
> **판정 요약**: ★**우리 결합 좌표는 이 축에서도 비점유** — 단 ShieldAgent·AgentSpec은 related-work 필수 인용(가장 가까운 이웃), 그리고 포지셔닝 문구를 "새 패러다임 발명"이 아니라 **"안전-결정적 제어에서 검증된 propose-then-shield 패러다임의, NL-구동 orchestration 계층 인스턴스화 + 빠진 두 조각(NL→spec 학습 컴파일·소형 proposer 전이)을 학습으로 채움"**으로 고정할 것.

## §1. 계보 1 — 제어/RL의 원-패러다임 (인용 닻)

| 출처 | 내용 | 우리와의 관계 |
|---|---|---|
| **Alshiekh et al., "Safe RL via Shielding" (AAAI 2018)** | LTL 안전명세→결정론 오토마타→safety game→shield 합성. pre-shielding(안전 행동만 노출) / post-shielding(제안 거부·교정). | 구조 동형의 원형: 학습 정책=제안, 결정론 shield=집행. **우리 게이트=post-shielding의 LLM-도구호출 대응물 + active 구동 확장.** |
| Jansen et al., "Safe RL via Probabilistic Shields" (1807.06096) | 확률적 환경에서 shield 완화. | 변종 계보 표기용. |
| **Wabersich & Zeilinger, predictive safety filter (Automatica 2021; pCBF 2105.10241)** | 학습-기반 제어기의 제안 입력을 제약 위반 여부로 검사·수정하는 **safety filter** — "어떤 학습 제어기든 뒤에 붙이는 안전층". | "필터는 제안자와 독립·모듈적"이라는 우리 주장(게이트=도메인-일반, 모델 교체 가능)의 제어이론 선례. |
| **Annual Reviews "The Safety Filter: A Unified View" (2023)** | CBF·예측필터·HJ-reachability 통합 프레임 survey. | 패러다임이 *성숙*했음의 증거 = "우리가 발명" 금지·"계승" 포지션의 근거. |

## §2. 계보 2 — LLM-agent 인스턴스화 (2024–25, 직접 경쟁 후보)

| 출처 | 무엇 | 5문항 체크 (결정론?/보장?/능동구동?/proposer학습?/전이?) | 좌표 충돌 |
|---|---|---|---|
| **ShieldAgent (arXiv 2503.22738, 2025)** ★최근접 | 정책문서→verifiable rules 추출→**확률적 rule circuits**→guardrail *LLM agent*가 타 에이전트 궤적의 정책준수를 검증 | ✗확률적(LLM 추론+probabilistic circuits) / ✗경험적(recall 90.1%, **precision=1 아님**) / ✗차단·플래그만 / ✗없음 / ✗없음 (abs 정독 5문항 확인 2026-06-10) | **비충돌**. 단 "정책문서→규칙 추출" 부분이 우리 Exp-B(NL SOP→구조)와 *기능상 유사* → related-work 1순위 인용 + 차별 = 우리는 추출물이 **결정론 집행기**(by-construction 보장)이고 그들은 확률 검증기(보장 없음). |
| **AgentSpec (arXiv 2503.18666, 2025)** | trigger/predicate/enforcement **DSL**로 runtime 제약을 손-작성, 결정론 집행 (code/embodied/AV, ms 오버헤드) | ✓결정론 / △규칙커버 범위 내 보장 / ✗차단·수정만 / ✗없음 / ✗없음(규칙=도메인별 손-작성) | **비충돌**. 차별 = 정확히 §15.2-②의 대조물: **per-domain 규칙 손설계** — 우리 thesis의 front-end 학습 leg가 제거하려는 비용 그 자체. + 능동구동·coverage 헤드라인 없음. |
| **Formal-LLM (arXiv 2402.00798)** | 개발자가 제약을 **오토마타로 손-작성**→스택 기반 생성 감독으로 plan이 제약 충족 | ✓생성-제약 결정론 / △ / △생성 중 제약(사후 차단보다 강함) / ✗ / ✗ | **비충돌**(스펙 손-작성·전이 없음). 생성-제약 변종 계보로 인용. |
| ProbGuard (arXiv 2508.00500) | 확률적 runtime 모니터링 | ✗확률적 | 비충돌, 한 줄 인용. |

## §3. 좌표 판정 — 무엇이 여전히 비어 있나

이 축의 어떤 것도 다음 **결합**을 갖지 않음 (개별 조각은 전부 존재 — 결합이 기여):
1. **by-construction 결정론 게이트** (evaluator-exact 재구성, Guard-2 OVER0/UNDER0) — ShieldAgent는 확률적, AgentSpec/Formal-LLM은 결정론이나 spec 손-작성;
2. **수동 차단이 아닌 능동 선행조건 구동** (active-H3: 게이트가 미충족 establishing을 결정론 실행, BOTH 6→15 실측) — 전 계열 부재 (shielding 용어로: post-shield의 veto를 넘어 *repair*);
3. **소형 학습 proposer + coverage-at-precision=1 헤드라인** — 전 계열은 안전층만 기여, 제안자 학습/coverage 최대화는 범위 밖;
4. **재학습0 도메인 전이** (scaffold 7도메인 실측) — 전 계열 부재;
5. **NL SOP→spec 학습 컴파일** (Exp-B, 미완) — ShieldAgent가 *기능상* 가장 근접(정책문서→규칙 추출)하나 산출물이 확률 검증기.

## §4. 포지셔닝 문구 (P1 related-work용 고정)

> "propose-then-shield는 safety-critical 제어에서 검증된 패러다임이다(shielding [Alshiekh'18]·safety filter [Wabersich'21, AnnRev'23]). 최근 LLM-agent로의 이식이 시작됐으나(ShieldAgent, AgentSpec, Formal-LLM) 안전층 단독 기여에 머문다 — spec은 손-작성이거나(AgentSpec/Formal-LLM) 집행이 확률적이며(ShieldAgent), shield는 차단만 하고, 제안자는 frontier 호출로 남는다. 우리는 이 패러다임을 NL-구동 도구-orchestration에 완성형으로 가져온다: **결정론 by-construction 게이트가 차단을 넘어 선행조건을 능동 구동하고, 소형 학습 proposer가 보장 하의 coverage를 최대화하며, 패키지가 재학습0로 전이된다.**"

## §5. 정직 caveat
1. **Exp-B(NL→spec) 실패 시 차별 축 5가 소멸** → AgentSpec과의 거리가 "능동구동+학습 proposer+전이"로 좁아짐(여전히 비충돌이나 마진 감소) — §15.2-②·bitter-lesson §7-3과 동일 조건부.
2. ShieldAgent-Bench는 *공격-방어* 프레임(악성 지시) — 우리 substrate는 *절차 준수* 프레임. 벤치 비교는 비대응이므로 수치 비교 시도 금지.
3. landscape time-sensitive (2503.x 두 편 모두 2025) — P1 직전 재스윕 필요(§10-6 미세-caveat ①과 동일).

## 출처 (fetch 검증)
[Alshiekh et al. AAAI 2018](https://ojs.aaai.org/index.php/AAAI/article/view/11797) · [Probabilistic Shields](https://arxiv.org/pdf/1807.06096) · [Wabersich & Zeilinger pCBF](https://arxiv.org/abs/2105.10241) · [Safety Filter unified view (Annual Reviews)](https://www.annualreviews.org/content/journals/10.1146/annurev-control-071723-102940) · [ShieldAgent](https://arxiv.org/abs/2503.22738) · [AgentSpec](https://arxiv.org/abs/2503.18666) · [Formal-LLM](https://arxiv.org/abs/2402.00798) · [ProbGuard](https://arxiv.org/pdf/2508.00500)
