# Scaffold 상태-라우터 설계 — per-step 결핍 게이팅 (2026-07-12)

> 소유: 개입레버의 spurious-misdirection을 제거하는 **결정론 per-step 라우터** 설계 + when/what/how 3축 통합 실험.
> 상위: `TRIVIAL_REGRESSION_ABLATION`(오염원=개입레버·Δspurious 실증) · `INTERVENTION_LEVER_CONDITIONALIZATION`(레버별 결핍-조건·arm3) · NIGHT §5 · `RESEARCH_MASTER §3`.
> 불변: [[05]] A2만·엔진 도메인일반 · [[10]] 선택기=결정론·생성기=LLM · Δspurious≤0 계측(모트) · gold-independence.

## 0. 동기 — 36/78은 창발이지 입력이 아니다
- "36 trivial"은 *COMP 통과*(결과)로 정의 → **런타임 관측불가**(gold 없음) → 태스크-분류 라우터 불가.
- 대신 **per-step 결핍 게이팅**: 결핍 없음→레버 무발화(=COMP-only·36 자연낙하)·결핍 감지→원인-매칭 레버(78). **36/78 = 게이트의 창발 출력.**
- 핵심(task106 증거): **"태스크 난이도(애매성)"가 아니라 "에이전트 실패(무효출력)"에 게이트.** 애매성-트리거(현 DISAMB=후보≥2)는 trivial-106에 오발화→깨뜨림. 실패-트리거(무효/미검토/미검증)는 valid 선택인 106에 무발화→COMP 유지.

## 1. 3축 직교 분해 (사용자 프레임)
| 축 | 질문 | 선택지 |
|---|---|---|
| **WHEN** | 언제 발화 | always(현 full) / **결핍-게이트**(per-step 감지) |
| **WHAT** | 무슨 레버 | all-5(현 full) / **원인-매칭 1개** |
| **HOW** | 어떻게 전달 | **override**(엔진 강제·현 subcall) / **advise**(프롬프트 힌트·에이전트 선택) / conditional-override |

현 full = (always·all-5·override) = 최악 부작용. 목표 = (결핍-게이트·원인-매칭·advise or 조건부-override).

## 2. 상태 머신 (WHEN×WHAT)
스텝마다 아래를 순서 평가·**첫 매칭 상태의 레버만** 후보(중복발화 금지):

| 상태 | 감지 신호 (gold-free·도메인일반) | 레버 | 근거 |
|---|---|---|---|
| **S0 정상** | 아래 결핍 신호 전무 | **없음(COMP-only)** | 36 여기 낙하 |
| **S1 무효-엔티티** | write arg값 ∉ 조회레코드의 가용후보집합 | DISAMB | 무효 id/변형 |
| **S2 미검토-coverage** | 요청-스코프 내 A2-relevant 레코드 미조회 존재 | EPLAN | discovery gap |
| **S3 미검증-값** | write 값이 어떤 조회출처에도 부재(=날조 의심) | GROUND/PROV | fabrication |
| **S4 누락-파라미터** | 정책-요구 필드 공란/부재 | PRINCIPLE_DEFAULT | policy default |

- **결정론 선택기**([[10]]): 상태판정=엔진 규칙(A2-spec가 "가용후보 출처·relevant 정의·요구 필드"를 도메인일반 표기). LLM은 생성만.
- **S0가 관건**: trivial의 valid-완주는 S1-S4 신호를 안 켬 → 무발화 → COMP-only. (현 full은 S1을 애매성으로 오정의해 켜버림.)

## 3. 전달 축 (HOW·레버별)
- **advise**: 결핍시점에 변동 프롬프트 힌트 주입("후보 2개 재확인하라")·선택은 에이전트. 안전(valid 유지)·단 준수 scale-의존([[42]]·IFScale)→약할 수 있음.
- **override**: 엔진이 값 교체/차단. 강함·단 valid도 뒤집을 위험(현 회귀원).
- **conditional-override**: 결핍(S1-S4) 확정시만 override = 위험 국소화.
- 가설: S0 무발화가 trivial을 지키면, S1-S4서 advise/조건부-override 선택은 hard-catch 강도 트레이드오프.

## 4. 근본 한계 (정직·모트)
- S1-S4는 **무효/미검토/미검증/누락**만 잡음 = 구조적 결핍. **valid-but-wrong-for-user**(task106 red-L·order-⋈=valid 주문 오선택)는 어떤 gold-free 신호로도 미감지 → **semantic 경계 = 잔여 = E7 learn/map**(C56 체계핵·thinking-flat 정합).
- ⇒ 라우터 도달지형: **36 무회귀(부작용0) + 78의 결정가능-부분 수리 + semantic-⋈ 잔여(P3 boundary 주장)**. 이게 정직한 상한.

## 5. 통합 실험 설계 (when×what×how 한번에)
**목표 metric**(비결정성 커버·rate): 태스크셋별 db pass-rate + **S0 무발화율**(trivial) + **레버 발화-후-수리율**(hard) + Δspurious.

**arm 매트릭스** (태스크셋 = trivial-36 + hard-78 부분집합·nt4):
| arm | WHEN | WHAT | HOW | 상태 |
|---|---|---|---|---|
| comp | — | — | — | ✅B |
| full | always | all-5 | override | ✅B |
| guard | always | 가드만 | — | ✅B |
| **prompt-rx** | always | all-5 | **advise** | 🔜 GPU1(즉시·`--rules_prompt`) |
| **router-ovr** | **결핍-게이트** | 매칭 | override | 코드 필요 |
| **router-adv** | **결핍-게이트** | 매칭 | advise | 코드 필요 |

**판독**:
- prompt-rx가 trivial 무회귀 → 부작용원은 **override 기전**(전달)·advise가 해소. hard 이득 있으면 advise로 충분.
- prompt-rx가 여전히 회귀 → always-load도 harm → **게이트(WHEN) 필수** → router로.
- router-*가 trivial 무회귀 ∧ hard 이득 → **결핍-게이트 성립**(make-or-break). S0 분리도 = 핵심 수치.

**구현 스테이징**: prompt-rx=즉시(`--rules_prompt` 기존 인프라·GPU1 병렬). router-*=결핍-게이트 코드(S1-S4 감지 = `t2_gate_patch` 확장·§2 신호)→오프라인 유닛(오발화0)→절단.

## 6. 무료 선행 (GPU 무관)
1. **결핍-감지 오프라인 유닛**: trivial 36 궤적서 S1-S4 신호 발화율 측정(목표 S0=대부분·오발화 0). hard서 발화율↑ 확인. = 라우터 성립성 **무료 예측**.
2. prompt-rx 규칙파일(`a2/RULES_PROMPT_DV2.txt`) = §2 처방의 advise 문안.

## 7. 다음
- 본 설계 리뷰 → §6.1 오프라인 유닛(무료·분리도 예측) → prompt-rx(GPU1 즉시) → 성립성 보고 → router 코드 → 통합 arm nt4.
- B(comp/full/guard)가 기저 확정 → prompt-rx/router가 4-6번째 arm으로 붙음.
