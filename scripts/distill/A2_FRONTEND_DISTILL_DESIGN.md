# A2 front-end 증류 설계 — NL 정책 → GATE_SPEC 컴파일러 학습 (detail, 2026-06-12)
> 📌 **구조 안내**: 모든 설계·실험 문서의 단일 마스터 = repo `scripts/distill/EXPERIMENT_DESIGN.md` (**§7 문서지도**; 목표·순서 변경은 마스터 §0-§4에서만). 처음 읽는다면 마스터부터.

> 발주 = 사용자 문답 (2026-06-12): "소형이 생성기 학습으로 frontier 정도 될 수 있나? 생성기 학습을
> 최적화하는 구조 설계 필요." 좌표 = `BENCH_PORTFOLIO` §3.8(생성기=교체부품·검증기=프로그램 불변),
> 마스터 §1.5(A2=유일 난제=thesis 상품형)·"세 컴파일러 대조군".

## 0. 왜 이건 LOCK이 아닌가 (regime 판별 — 선결)
LOCK이 죽인 것 = **실행-루프 내 결정-emission**(게더 도중 truth/derivation 생성 → fabrication).
A2 컴파일 = **오프라인·단발·닫힌 스키마·결정론 검증기 동반** 구조화 번역 — 우리가 반복 성공한
regime(v2 균형-DPO·D1 구조-DPO·v3·RFT+evaluator)의 형태. guided로 스키마 강제 가능(R1)·
K-샘플+검증기-선별 가능(R6)·검증 통화가 결정론(replay over/under-deny).

## 1. 과제 정의
- 입력: 도메인 정책 NL (τ² policy.md류·SOPBench SOP·Amazon SOP 텍스트).
- 출력: `GATE_SPEC` JSON (gate별 predicate·satisfiers{tool→required inputs}·applies_to·terminal/ask).
- **수용 기준 = 결정론 검증기**: Guard-2-동형 replay (gold 궤적 over-deny=0 ∧ 순서위반=0)
  + (가능 도메인) evaluator 대조. 검증기는 영구히 프로그램 (§3.8 불변).

## 2. ★데이터 엔진 — 역방향 렌더링 (데이터 기근 해소)
실 (NL, spec) 쌍은 ~22 도메인뿐 → 합성으로 확장하되 **GT를 구성으로 보장**:
1. **spec 샘플러 (프로그램)**: GATE_SPEC 문법 위 무작위 샘플 — predicate 종류(인증/확인/스코프/한도/
   시간창...)·satisfier 도구 시그니처(가짜 카탈로그 동반 생성)·applies_to 조합. 난이도 손잡이 =
   게이트 수·교차참조 깊이·예외절 수.
2. **NL 렌더러 (frontier, 도메인당 1회 비용 불요·무제한)**: spec → 정책 산문 K-스타일
   (격식/캐주얼/불릿/장문-교차참조/한영혼합). **spec이 먼저라 GT 완벽·검증기 불필요.**
3. **오염 통제**: 렌더 NL에 spec 용어 직노출 금지(별칭/패러프레이즈 — alias 마스킹 교훈)·
   스타일 다양성이 분포갭 완화의 본체.
- 산출 규모 목표: 5k-20k 쌍 (7B LoRA SFT 수 시간 분량).

## 3. 학습 사다리 (사전등록)
| 단계 | 방법 | 게이트 (통과 기준) |
|---|---|---|
| **S0 합성 SFT** | LoRA SFT + guided(spec JSON 스키마) | G-A2-1: held-out 합성 spec EM ≥90% ∧ **실 retail replay over+under-deny 합 ≤ frontier 단일샷** |
| **S1 실-도메인 verified distill** | frontier가 실 정책 22 도메인 컴파일 → replay 필터 통과분만 SFT 계속 | G-A2-2: **LODO** (N−1 도메인 학습 → held-out 도메인 컴파일) replay-검증 gate-F1 |
| **S2 on-policy DPO** | 자기 K-샘플 → 검증기 채점 → (통과, near-miss) 쌍 — **대조축=구조 정확성**(D1 교훈: 길이 탈교락 by 스키마 고정) | G-A2-3: S1 대비 K=1 정확도 ↑ ∧ 검증기-선별 후 동일 |
| 추론시 | K-샘플 + 검증기-선별·전원 탈락 시 **abstain→HITL** (F6 risk-coverage로 채점) | — |

## 4. 판정 프레임 — "frontier급" 주장의 정확한 형태
- 비교 단위 = **시스템** (생성기+검증기+K-선별), 단일샷 아님.
- **세 컴파일러 대조군** (마스터 등재분의 실측 무대): L0 파서 / frontier 단일샷 / 소형 K+선별 —
  동일 검증기·동일 도메인(retail+airline→22 LODO).
- 사전등록 헤드라인 예측: **소형(K=8+검증기-선별) ≥ frontier 단일샷** on held-out 도메인
  (replay 통과율·gate-F1). 근거: 과제가 닫힌 구조화 번역 + 정밀도가 검색 문제로 환원(N2 +8.8 동형)
  + 7B+scaffold>GPT-5 선례.
- 훈련의 존재 이유 명시(§3.8): 주권(망분리)·정책-Δ robustness·전이 주장 — 그 외 regime에선
  frontier 컴파일이 정답이라고 논문에 그대로 씀.

## 5. 리스크 (정직)
①합성→실 분포갭 (실 정책의 암묵·세계지식 의존 절 — 완화: S1 실-도메인 + abstain)
②검증기 커버리지: replay는 gold 궤적 범위만 검증 — gold-밖 over-deny는 미검출
  (완화: τ²류는 evaluator 대조 추가·합성은 GT 완전)
③긴 정책(10k라인 타깃)의 globality — 청크 교차참조 (2단계 stress-test로 분리, §1 RAG-대조 계획과 합류)
④spec 스키마의 표현력 한계 — 새 predicate 유형 등장 시 스키마 확장 비용 (버전 관리).

## 6. 실행 순서 (큐 등재용)
P-A2-0 (zero-GPU): frontier로 retail+airline 컴파일 → replay 검증 — GT 파이프라인 생존성 + frontier 단일샷 baseline 수치 확보.
P-A2-1: spec 샘플러 + 역방향 렌더 5k → S0 SFT → G-A2-1.
P-A2-2: 실 22 도메인 verified distill + LODO → G-A2-2.
P-A2-3: on-policy DPO → G-A2-3 → 세-컴파일러 표 완성 = thesis front-end 헤드라인.
