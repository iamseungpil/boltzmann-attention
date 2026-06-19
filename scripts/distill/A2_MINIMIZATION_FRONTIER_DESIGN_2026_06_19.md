# A2-최소화 프론티어 설계 — 모델크기 × 최소 A2 × 변경비용 (2026-06-19)

> **자립 문서·핵심 실험 프로그램**(리뷰용). 목표 = on-prem 배포 가치(비용·재사용)의 정량 실증. 권위 = `dr_deterministic_vs_learned_tradeoff`(비용·change-surface)·`dr_nl_to_formal_interface_granularity`(formalize=선행·차별=target)·`COWORKER_RESULTS_2026_06_17_scale`(scale 인프라). 메모리 = `00-thesis`·`05-fixed-vs-variable`·`41-relwork`·`30-remote-env`.
> 분리: A2 자동생성기(frontier·ATA-like) = 별도 논문 / 본 문서 = **실행 + A2-최소화 측정**(A2는 주어진 것으로 가정).

## 0. 목표 (배포 제약이 정의)
on-prem·**frontier 호출 불가**·**데이터 반출/frontier-학습 불가**·**제약 위반 0** 환경에 **기 학습 모델을 들고 들어감**. → ToolOrchestra(frontier 위임) 자격 없음. 가치 = **도메인당 A2(설정) 최소화 + scaffold 불변 재사용**(유지보수·비용).

## 1. 연구 질문 (정량)
> **주요 규칙을 LLM이 학습하고 그 규칙이 쓰는 *세부정보만* A2로 줄 때 — *고정 scaffold*로 얼마나 많은 다른 도메인/벤치를, *얼마나 작고 변경에 강한 A2*로 푸는가. 그리고 그걸 *최소화하는 모델 크기*는?**

스펙트럼: **ATA(A2=전체·학습 0) ←──── 우리(중간) ────→ frontier(A2≈0·전부 가중치)**. 1.5B→72B를 이 위에 매핑.

## 2. 실험 축 (3축 × 모델 sweep)
- **★모델 크기 sweep**: Qwen2.5 **1.5B / 7B / 14B / 32B / 72B** (동일 family·깨끗한 scaling). 각 크기를 *같은 규칙 데이터*로 학습.
- **A2-ablation**: 풀 A2 → 컴포넌트 제거 → 커버리지 유지? *제거가능=모델이 학습 / 잔여=최소 A2.*
- **도메인/벤치**: retail·airline·SOPBench 도메인들·TaskBench. **같은 모델+scaffold·A2만 swap** = 재사용/전이.
- **변경 이벤트**(§6): A2 perturb → 유지보수 비용.

## 3. A2 분해 (제거가능 컴포넌트 = 측정 단위)
- 카탈로그/스키마(도구·아이템) · gate_spec(정책 규칙) · 의존맵(which-producer) · 속성키(catalog option) · 절차 템플릿
- **"A2 크기" = 필요한 컴포넌트 집합 + 각 복잡도(필드·규칙·교차참조 수).** 최소-A2 = 커버리지 유지하는 최소 부분집합.

## 4. 메트릭 (정적 + ★변경 + 비용)
- **정적**: 최소-A2 크기/복잡도(컴포넌트·필드·coupling).
- **생성난이도**: frontier 자동생성 성공률(ATA식 76%→94.2%)·human 교정분.
- **★변경-흡수(핵심·A2가 자주 바뀌므로)**: **변경-이벤트당 A2-edit 수(blast-radius) + 재학습 필요 여부.** 학습모델(규칙 가중치) = 변경이 *사실-편집*으로 축소·재학습 0 / 결정론(규칙 config) = *절차 편집*·큼.
- **커버리지**: 도메인별 공식지표 임계 통과율(per-bench·집계금지).
- **재사용**: 같은 모델+scaffold가 커버하는 도메인 수.
- **inference 비용**: 모델크기별 토큰/지연/$.
- **★총비용-최적 크기**: `총비용 = inference + A2-저작·유지보수`. **A2는 크기↓(큰 모델)이나 inference↑** → 트레이드오프 knee = 답.

## 5. 핵심 결과물 = 두 곡선
1. **최소-A2(및 변경비용) vs 모델크기** (1.5B→72B), 도메인 횡단. 어디서 A2가 충분히 작아지나(knee)·plateau 지점(닫힘 floor=사실은 못 줄임).
2. **총비용-최적 모델크기**: inference vs A2-유지보수 트레이드. on-prem서 어느 크기가 최저 TCO.

## 6. 변경 시뮬레이션 (사전등록·게이밍 차단)
DR 문서화된 변경 타입을 벤치에 perturb → A2-edit·커버리지·재학습필요 측정:
- 도구 추가(새 API) · 속성 rename(`user_id`→`userId`) · 정책값 변경(한도 $100→200) · 카탈로그 variant 추가.
- 각각: (a)A2 diff 크기 (b)커버리지 유지 (c)재학습 없이 흡수. × 모델크기 × {학습모델·결정론-bespoke(L0)·ATA-style}.

## 7. closure = A2-축소의 메커니즘 (왜 가능한가)
닫힌 도메인-일반 생성원 기저라 **규칙이 도메인-불변 → 한 번 학습·가중치 흡수 → A2 = 사실만**. 열린 target(ATA의 ad-hoc FOL·semparse SQL)은 도메인마다 재형식화 → A2 안 줄음. (`dr_nl_to_formal`: formalize 방법=선행·차별=닫힌 target.)

## 8. 고정 scaffold 불변식 (치팅 차단)
- scaffold(범용 결정론 엔진) = 도메인·모델크기 무관 **불변.** `grep "if domain"=0`·CI.
- 도메인 적응은 **오직 A2 swap·재학습 0**(in-env 제약). 모델 학습 = offline·도메인-일반만.

## 9. 자가심사 (anti-drift 규칙7)
- **게이밍**: "A2 크기" = 컴포넌트 단위·사전등록·ablation. 변경타입 사전등록. 커버리지=공식지표·per-bench.
- **치팅면**: scaffold 불변(grep)·A2 swap만·재학습 0·결정론 게이트 compliance 보장(real 도구 미대체).
- **thesis정합**: 학습=도메인-일반 규칙(closure 기저)/A2=사실/scaffold=고정. ([[05-fixed-vs-variable]])
- **정직**: A2는 0 안 됨(사실 floor)·큰 모델도 깊은선택 못함(235B@N50=0.02→offload 잔존)·human-hours/$=soft 프록시(상대 edits/change·auto-gen율로 대체).

## 10. Falsifiable + GO/NO-GO
- **H**: 모델크기↑ → 최소-A2↓·변경-edit↓ (규칙 흡수). 적당한 크기(32B?)서 A2가 ATA(전체) 대비 *유의 작고* 도메인 횡단 재사용.
- **GO**: (a)최소-A2가 모델크기로 단조↓·plateau 보임 (b)32B/72B서 A2가 L0/ATA 대비 컴포넌트 수 유의↓ (c)변경-이벤트당 edit이 결정론 대비 유의↓·재학습 0 (d)같은 모델+scaffold가 ≥3 도메인 커버.
- **NO-GO**: 모델크기 무관 A2 안 줄거나(규칙 학습-전이 실패=앞 세션 위험 재현) · 변경-흡수 이득 없음(결정론과 동일 edit) → 학습 가치 없음·thesis 음성.

## 11. 선행 대비 (차별 정량)
- **ATA**: A2=전체 FOL(도메인별·+human)·학습 0. → 우리 = "모델크기로 A2를 ATA 아래 *얼마나*"를 측정. 곡선의 ATA-끝.
- **ToolOrchestra**: frontier 위임 → on-prem 자격 없음. 비교 무의미.
- **frontier(대형 on-prem)**: A2≈0이나 inference 막대·compliance 무보장·도메인 변경시 재학습. → 우리 = 총비용·보장·재학습0서 우위 주장. 곡선의 frontier-끝.
- **L0(결정론-bespoke)**: A2=전체 로직(도메인별 손코딩). → 우리 A2-축소·변경-흡수의 비교 baseline.

## 12. 빌드 단계
- **S0**: 규칙 학습 데이터(도메인-일반·closure 기저·다양성) + 고정 scaffold + A2 컴포넌트 분해 정의.
- **S1**: 1.5B/7B 먼저 (싸게) — 최소-A2 ablation + 변경 시뮬 + retail/airline 재사용. 곡선 하단 + 방법 검증.
- **S2**: 14B/32B/72B (coworker scale 인프라) — 곡선 완성·knee·총비용-최적.
- **S3**: 도메인 매트릭스 확장(SOPBench·TaskBench) + L0/ATA-style baseline 대비.
- 자산: scale 인프라(32/72B)·tau2 harness·SOPBench·기 학습 LoRA·A2 분해.

## 핵심 한 줄
**고정 범용 scaffold 위에서, 모델크기(1.5B→72B)에 따라 *최소·변경강건 A2*가 어떻게 줄고 도메인 횡단 재사용되나 — ATA(A2=전체)와 frontier(A2≈0) 사이를 어느 모델크기가 최저 TCO로 메우나.** = on-prem 배포 가치의 정량 프론티어.
