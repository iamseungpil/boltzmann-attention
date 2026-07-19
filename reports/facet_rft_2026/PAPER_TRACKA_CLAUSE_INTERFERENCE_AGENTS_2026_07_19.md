# Track A — 시스템/에이전트 논문 골격 (2026-07-19 · 사용자 지시)

**가제**: "Same-Rule Interference: Why Batched Per-Item Judgments Fail in LLM Agent Pipelines, and a Structural Fix"
**한 줄 논지**: 문서-기반 per-item 판단을 배치로 시키면, **같은 규칙 조항을 추론-결합하는 항목들끼리만** 서로 간섭해
(k*=2 계단 임계) 애매 항목이 기본값으로 후퇴한다. 프롬프트 완화는 전멸하고, **기전-도출 구조 분리(batch≤2)**가
실전 에이전트 파이프라인(τ²-bench banking)의 태스크 실패를 100% 닫는다.

## 기여 (전부 실측 완료 [S])
1. **실전 실패의 발견·격리**: τ² banking rate-formalize에서 특정 항목만 위치-의존 실패 (task_028 실측→프로브 재현).
2. **4단 배제 사슬**: 토큰 부하 ✗(길이 불변) → 절대 토큰 위치 ✗(비단조·350tok 창) → 생성-순서 ✗(입/출력 분리 arm 2/4)
   → 범주-유사성 ✗ / **조항-유사성 ✓** (§2m).
3. **이중 해리**: 쿼터(5배 소비)·판단-소모·범주-단서 3가설 기각 — 간섭원 = **같은 조항을 이름-일치 없이 추론-결합하는
   선행 항목** (B/C arm 통과·A arm 실패·합성 confound 기각).
4. **계단 임계 k*=2** (두 target 동일·임계점 출력 불안정화=단위 슬립 관찰).
5. **구조적 완화**: batch≤2 → 프로브 38/38·전 카드 무회귀 166/167·라이브 018/021 PASS. 프롬프트 보강 2종 실패
   실측(선행들의 "프롬프트 완화 전멸"과 정합·[[42]]/[[07]] 공명). prefix-cache로 비용 상쇄 실측(hit rate 로그).
6. 해석틀: modern Hopfield 연상기억(상관 패턴 간섭) 접속 — 확정 주장 없이 프레임으로만 (기전 확정은 Track B).

## 관련연구 (전부 인용·양보 — §2m 보강 종합)
- SPE-선택편향: Guo & Vosoughi ACL'25F(2406.15981)·Wang+ EMNLP'23. Batch prompting: Cheng+'23·BPE(순열+투표).
- IFScale'25(primacy 편애). PI: Unable to Forget(2506.08184)·Remember First Forget Last(2603.00270).
- LiM: Liu+'23(우리는 실험으로 분리). 부하: EMNLP'25 Context Length Alone Hurts. DACS·2501.01880(주입>RAG).
- 민간 관행 "7개+ 분리" → 우리는 임계·기전·무회귀 검증으로 격상.

## 남은 필수 실험 (Track A 완결 조건)
- [ ] **A① 통계 강건성**: 지시문 패러프레이즈 ×3 · 개입 브랜드-세트 ×3 · 행 순서 셔플 ×3 (32B :8140·무료)
      → k∈{0,2,4} 실패율 CI. (temp0 결정론이므로 구성-변형으로 분산 확보.)
- [ ] **A② k*-크기 스윕**: 0.5B/1.5B/3B/14B/32B — k* (또는 k=0 baseline 성립 여부) vs 크기.
      예측: 작은 모델은 k=0부터 실패(Δ 자체가 작음)·클수록 k*↑ (UF "큰 모델 저항"·[[46]] crossover 접속). **진행 중.**
- [ ] A③ 태스크-일반성 스팟체크: banking 외 1개 도메인(예: retail 유사 per-item 판단) 미니 재현 — 도메인-일반성 주장용.
- [ ] 라이브 최종표: redesign7b(022/028/029) 판정 포함 계열 스코어보드.

## 그림 계획
F1 파이프라인+실패 사례 / F2 배제 사슬(위치 스윕·입/출력 분리) / F3 이중 해리 막대 / F4 k-스윕 계단(크기별 겹침)
/ F5 batch≤2 무회귀+라이브 / T1 선행 대비표.

## 정본 데이터 소스
`RATE_SUBAGENT_DESIGN_2026_07_18.md` §2i~2m · sim_results/bank_redesign{4,5,6,7b} · rot/ksweep/mech/ioexp 로그(scratch→영속 예정).
⚠️scratch 로그들( rot_serial·mech·ksweep·ioexp·rpi )을 sim_results로 gzip 영속할 것(등대 갱신 프로토콜).

## 벤처/포지셔닝 메모
- Paper1([[46]] crossover)과의 관계: 본 논문은 독립 출고 가능. Paper1에는 §사례로 축약 인용.
- 후보 벤류: agentic/systems 트랙(ICLR/NeurIPS D&B·ACL industry) — 실전 파이프라인+구조 완화 강조.
