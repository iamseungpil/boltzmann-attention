# PATH-SELECTION 축 설계 — 도구폭발 하 경로선택을 깊은 추상으로 (CDP·2026-06-19)

> **자립 문서·별도 논문/특허 방향**(리뷰용). 타깃 = patent의 **CDP 마케팅 경로선택**(실제 풀 문제). tau2/closure-payoff와 *다른 축*. 권위 = `ma/M_A_RESULTS §15`(depth-scale·comparative@N50)·`PRIMITIVE_COVERAGE_MATRIX`(P3/P4/P9)·`COWORKER_RESULTS_2026_06_17_scale.md`(Lane1). 메모리 = [[00-thesis]]·[[10-roles-deterministic]]·[[01-four-bench-tbox]]·[[40-settled-cite-only]].
> 관계 = `CLOSURE_PAYOFF`(생성원=알파벳·grounding 축) ⊥ 본 문서(생성원 합성 위 *탐색*=언어 축). 상보·분리.

## 0. 문제 — CDP 경로선택 = 진짜 타깃, tau2는 *다른* 축
실제 풀 문제 = patent CDP 마케팅: **한 스텝에 같은 성격 도구 수십~수백 개 + 경로 조합 폭발** 하에서 **제약·선호를 만족하는 *가장 적절한 경로*** 찾기. 이건 *값 grounding*이 아니라 **탐색/계획(search/planning)**.

★구분(이번 세션 실증): tau2-retail 병목 = **provenance(P2b/P4·id fetch·복사)** = tau2의 *얕음*(도구 ~15·얕은 경로)이 드러낸 *grounding* 실패. **CDP 핵심 아님.** CDP = *경로탐색* 실패. **두 실패모드를 섞으면 안 됨.**

## 1. 두 축 분리 (실패모드 직교)
| 축 | 벤치 대표 | 병목 | primitive | 처방 |
|---|---|---|---|---|
| **provenance(grounding)** | tau2-retail | id 날조·fetch 누락 | P1·P2b·P4-filter | copy-grounding 학습(cfbsynth)·가드 |
| **★path-search** | CDP·TravelPlanner | 조합폭발서 경로선택 | P3·P4-argmax·P9·*탐색* | 본 문서 |
- 같은 "P4"라도 **filter(매칭·copy·grounding 축)** vs **argmax/rank over 큰 N(계산·선택·path 축)** = 다른 갈래([[CFBSYNTH_P2B_P4_DESIGN_2026_06_19]] COPY/COMPUTE 경계와 정합).

## 2. 경로선택을 framework에 매핑
- **한 스텝(수백 중 1)** = 큰-N 동질 COMPUTE-selection → **offload**. ★settled 증거: depth-scale `comparative@N50=0.02`(235B 포함)·`rank@N50≈0.30`([[40-settled-cite-only]]·Lane1) = **큰 N 선택은 in-head로 *스케일 안 됨*·엔진은 trivial(1.00).** 즉 "수백 중 고르기"는 추상으로 *직접* 불가(이미 실증).
- **다단계(경로 조합)** = 순차결정(MDP/POMDP류): 선택이 미래 옵션을 바꿈·부분관측·value/선호 의존. **생성원 closure는 *알파벳*(무슨 연산)을 주나, 어느 *경로*냐는 조합공간 *탐색*** = "알파벳 닫았지 *언어* 아님"(thesis risk C)의 CDP 구체화.
- **⇒ 미해결 질문**: 경로선택이 **깊은 (학습된) 추상으로 가능한가**, 아니면 본질상 *탐색*(외부 offload)인가.

## 3. ★핵심 연구질문 + 분담 가설
**Q**: 작은 LLM의 *학습된 도메인-일반 추상*이, 도구폭발 신규 도메인서 **좋은 경로를 선택**하게 하나 — 아니면 경로탐색은 결정론 offload(탐색+메모리)이고 LLM은 *formalize·휴리스틱*만인가.

**분담 가설(thesis-정합·[[10-roles-deterministic]])**:
- **LLM이 *학습*(도메인-일반·ABox-swap 전이)**: ① 목적·제약 **formalize**(무엇이 좋은 경로인가) ② 상태별 **적용가능 연산 인식**(생성원 라우팅) ③ **value/heuristic 추정**(이 부분경로가 유망한가).
- **결정론/외부(offload·decidable)**: ④ 경로 **탐색**(MCTS·beam·A*) given 휴리스틱 ⑤ **메모리**(실행 성공/실패·사용자 선호) = value 입력. (patent의 외부 방법들 = 이 leg.)
- **★novel·전이 기여 후보 = ②③** — 탐색을 *tractable*하게 만드는 **전이가능한 학습 휴리스틱/formalize**가, 재학습 0(ABox-swap)으로 신규 도메인 경로탐색을 이끄나.

## 4. Falsifiable 가설 + 무엇이 가른다
- **H_offload**: 경로탐색=순전히 offload. 학습 추상은 휴리스틱 못 줌(depth-scale 외삽: 순차 큰-N은 단일보다 *더* 안 됨) → 기여 = "경로=offload" 분리 자체(약).
- **H_abstract**: 깊은 학습 추상이 **전이가능 휴리스틱**을 줘 — (a) 무휴리스틱 탐색(naive MCTS) (b) 대형 in-head 경로선택 **둘 다 초과** → 기여 = "소형+전이휴리스틱 > 탐색단독·대형단독"(강·헤드라인).
- **가르는 실험**: heuristic **ablation**(LLM-value 유/무·같은 탐색예산) + **전이**(학습벤치 휴리스틱 → CDP/TravelPlanner ABox-swap·재학습0) + **scale 대조**(휴리스틱 없는 235B in-head vs 7B+휴리스틱+탐색).

## 5. 벤치 계획 — path-search 격리·tau2와 2축 비교 (★수치 검증 필요)
- **TravelPlanner**(Xie+2024·주벤치): 카테고리당 *수백 동질 옵션*(항공·호텔·식당)+제약·선호 하 **여정(경로) 선택**·조합폭발. LLM 악명높게 어려움(GPT-4류 final-pass *극저*·검증요). **병목=경로탐색/제약충족·provenance 아님** → CDP 동형. 가장 깨끗.
- **AppWorld**(Trivedi+2024): 9앱·~457 API·*stateful 깊은* 상호작용 = 경로+상태.
- **NATURAL PLAN**(Google): 제약 하 조합계획·도구경량.
- **2축 비교**: 같은 TBox를 **tau2(provenance)** vs **TravelPlanner(path-search)**에 → "추상이 경로선택을 *돕나*(②③ 전이) vs 경로는 순전히 offload(LLM=④⑤ 무관)"를 가린다. = CDP 직결 첫 데이터.

## 6. 측정 (결정론·[[10-roles-deterministic]])
- **경로 품질**: 제약충족률·선호 만족·최종 task pass(벤치 공식).
- **탐색 비용**: 노드 전개수·토큰·USD — 휴리스틱이 *탐색을 줄이나*(효율 기여).
- **전이**: 학습벤치 휴리스틱 → 타깃 ABox-swap·재학습0 보존율(per-bench·[[20-proven-results]] 집계금지).
- **휴리스틱 기여 ablation**: LLM-value on/off·탐색예산 고정 → Δ.
- **scale 대조**: 7B+휴리스틱+탐색 vs 대형 in-head(depth-scale 연장).

## 7. 외부 vs 내부 (MCTS·메모리·선호) = offload 위치·학습 기여 격리
patent의 외부방법(실행 성공/실패 메모리·선호 메모리·MCTS·경로추정)은 **전부 offload/scaffold leg(④⑤)**. 본 설계의 *학습* 기여 = **②③(formalize·전이휴리스틱)뿐** — 그래서 "외부든 내부든"은 *구현*이고, 측정해야 할 novelty = **학습 추상이 그 탐색·메모리를 *전이가능하게* 인도하나.** (외부 탐색을 줘도 휴리스틱이 도메인마다 재학습 필요하면 기여 약·ABox-swap로 전이하면 강.)

## 8. Thesis-정합 + closure-payoff와의 관계
- **알파벳 vs 언어**: closure-payoff = 생성원 기저(알파벳)·grounding 전이. 본 문서 = 그 합성 위 *탐색*(언어). **상보**: 경로는 생성원으로 *구성*되나 어느 경로냐는 탐색 = closure가 *못 닫는* 부분([[03-anti-drift]] risk C). 분리 정직.
- **두 날개**([[00-thesis]]): 본 축도 (A)capability=분해+결정론(탐색=offload) (B)전이=formalize+ABox-swap(휴리스틱 전이). 정합.

## 9. 자가심사 (anti-drift 규칙7)
- **치팅**: 휴리스틱 기여=ablation으로 격리(탐색예산 고정)·전이=재학습0 ABox-swap·per-bench. real 도구 미대체. ✅
- **thesis정합**: 학습=formalize·휴리스틱(도메인일반)/offload=탐색·메모리/ABox=도메인 옵션·제약. ✅
- **선행 재사용**: MCTS·LATS·Tree-of-Thought·success/preference memory = *그대로*(재발명 아님·[[41-relwork-rivals-whitespace]] directive). 기여=*전이가능 휴리스틱*이지 탐색기 아님.
- **scope 정직**: 경로선택이 *순수 offload*로 판명나도(H_offload) = 정직한 음성·thesis 경계 확인.

## 10. 논문/특허 분리 근거 + 위험
- **분리 근거**: grounding 축(closure-payoff)과 *직교* 기여(탐색 위 전이휴리스틱)·CDP 응용=특허 직결·"소형+전이휴리스틱>탐색단독·대형단독"=독립 헤드라인.
- **위험**: (i)H_offload면(휴리스틱 무전이) 헤드라인 약→"경로=offload 분리"로 후퇴 (ii)벤치 셋업(TravelPlanner sandbox·AppWorld stateful) LoRA arm보다 무거움 (iii)전이휴리스틱이 도메인-특정이면 ABox-swap 안 됨=기여 붕괴(핵심 리스크) (iv)patent 내용과의 정합(특허 명세 확인 필요).
- **미결(리뷰안건)**: ① 전이휴리스틱을 *어떻게* 학습·표현(value head? in-context? 생성원-라우팅 확률?) ② TravelPlanner가 CDP 충분 동형인가·CDP 자체 mini-bench 필요? ③ depth-scale(단일선택) → 순차경로 외삽이 정당한가 ④ 학습/offload 경계서 ②(인식)·③(value)가 진짜 학습-전이하나, 아니면 그것도 decidable(=offload)인가.

## 11. 다음 (설계 후)
1. 리뷰(이 문서·특히 §3 분담·§4 가른다·§10 분리근거).
2. 벤치 1개(TravelPlanner) 셋업·수치 검증(GPT-4류 floor·우리 floor).
3. 사전등록: H_offload vs H_abstract 판정선(§4 ablation·전이).
4. tau2(provenance)와 2축 대조표.
