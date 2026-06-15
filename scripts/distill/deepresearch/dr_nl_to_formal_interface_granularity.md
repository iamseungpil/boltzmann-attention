# Deep Research — NL→formal 인터페이스 granularity & decouple-then-resolve (NL→SQL/semparse)

> 출처: deep-research 워크플로 `wf_d5e29165-dea` (2026-06-16 재실행·완주). 105 agents·22 sources·97 claims 추출→25 검증(3-vote 적대)→**24 confirmed·1 refuted**·7 synthesized. 인용=1차 출처 검증분만([[feedback-arxiv-citation-discipline]]).
> 설계 직결: `ABOX_CONFIG_FORMALIZATION_DESIGN_2026_06_15.md` §1.5/§6.5/§7·M-A 프로토타입. selector 접점(reference emit) = sketch-then-fill 계보.

## 한 줄 결론
**분야 문헌은 우리 설계의 decouple 패턴으로 강하게 수렴한다**: 모델은 **추상·값-없는 구조**(sketch/IR/reference로 schema요소 선택)를 방출하고, **결정론 하류**가 concrete 값/엔티티를 resolve·실행. 이 패턴(IRNet/SemQL·coarse-to-fine·sketch-based)은 정확도 + **cross-domain/zero-shot 전이**(=config/ABox-swap 직접 analog)를 반복적으로 개선한다. grammar-constrained decoding은 **form만 보장·meaning 아님**(CFG 천장) → grammar 너머 결정론 엔진 필수.

## 검증된 7 findings (전부 3-0, 명시 예외 제외)

1. **Decouple-structure-then-resolve = 분야 지배 패턴** (high). 학습모델이 low-level(엔티티·술어·인자·변수명) 생략한 추상 구조 방출 + 별도 하류가 concrete 채움. IRNet/SemQL(3-phase: schema-link→grammar SemQL synthesis→**결정론 SQL inference**)·coarse-to-fine(Dong&Lapata)·sketch-based. [1905.08205·P19-1444·P18-1068·1909.00574]

2. **★reference/placeholder 방출(literal 아님) = 정확도+전이 양성** (high). decoder가 schema column/table을 **pointer로 reference 선택**(literal 생성 아님)·value-free sketch는 "basic meaning 같고 detail만 다른" 예제 간 **공유**(전이 근거). 측정: sketch-first GEO 88.2 vs 85.0·ATIS 87.7 vs 85.3·DJANGO 74.1 vs 69.5. = **M-A의 정확한 thesis**(LLM이 order_id 아닌 entity-ref/variant-select criteria emit). [1905.08205·P18-1068]

3. **Grammar/schema-constrained decoding = well-formedness·정확도 개선** (high). PICARD: T5-3B 68-71%→**75.1%**·execution error 12%→2%. 동기=무제약 LLM이 invalid SQL 빈발. = xgrammar 정당화. [2021.emnlp-main.779·1905.08205]

4. **★Grammar는 FORM 보장·MEANING 아님 = 이론적 천장** (high). CFG는 context-sensitive 제약(술어 arity·변수 scope·declare-before-use) 못잡음(Chomsky 위계) → syntactic-valid-but-semantic-invalid 잔존·executable rate <1.0 → **별도 결정론 엔진/solver 필수**. = "xgrammar=type강제·결정기 resolver=semantic 책임" 분담 직접근거. [2025.acl-industry.34·2021.emnlp-main.779]

5. **★최적 분담 = LLM-as-parser + 결정론 엔진/solver(reasoning·value-resolution·실행)** (high). LLM을 NL→symbolic parsing에 국한·reasoning을 symbolic solver에 위임하면 **제약의 이득 > reasoning 저하**. execution-guided decoding(학습 verifier 아닌 **결정론 실행엔진**)이 faulty program 필터·WikiSQL 83.8%. = thesis 핵심. [2025.acl-industry.34·1807.03100]

6. **★NEW schema zero-shot 전이 = schema-as-input + relation-aware linking** (high·**=ABox-swap analog**). RAT-SQL relation-aware self-attention(schema encoding+linking+alignment) → Spider 57.2%(+8.7)·BERT 65.6%. IRNet IR: cross-domain +19.5~27.0. **schema/config를 입력으로 주고 reference-selection** 설계가 schema swap 간 전이. [1911.04942·1905.08205·P19-1444]

7. **★Schema-linking/value resolution = 결정론/하이브리드 최적** (high). NL멘션→schema/cell값 grounding은 결정론 n-gram/string match·DB/ConceptNet lookup·학습모델은 link 신호를 feature로 소비(relation-aware). = **reference SELECTION은 학습(주어진 schema 위 pointer)·concrete VALUE RESOLUTION은 결정론 lookup**. = 설계의 P4-σ(기준=학습) vs P4-γ(매칭=결정론 offload) 분리 정확 일치. [1905.08205·1911.04942]

## REFUTED (1-2·과장 박제)
- "GCD-as-parser가 13개 LLM 전부서 syntactic+downstream **semantic** accuracy 일관 개선" = **기각**. 강건 이득=well-formedness·structured-output 신뢰성뿐·semantic 정확도는 결정론 엔진 필요. ⇒ **xgrammar 단독이 값-정확성 고치리라 과장 금지**. [2025.acl-industry.34]

## ★Caveat (정직·우리 세팅 직결)
- **증거 대부분 pre-LLM(2018-2021)**. LLM-era(2023-26)는 **qualify·overturn 아님**: ①제약 decoding은 base모델 강해질수록 diminishing returns(GPT-4급은 스스로 valid SQL) ②form-vs-meaning 천장(#4)은 LLM-era 결과가 **재확인**.
- **★커버리지 갭(중대)**: NL→API/tool-plan 가지(API-Bank·ToolLLM·RestGPT·Gorilla·NL2API)·compositional-gen(SCAN/COGS/CFQ)·Spider2.0/BIRD는 **생존 검증 claim에 미포함** → API/tool-calling 세팅(=우리 세팅)의 value-vs-reference 결론은 **text-to-SQL서 유추**, 직접소싱 아님. ⇒ **M-A가 메우는 바로 그 빈칸**(아래 openQ).
- sketch-based(1909.00574)의 value-filling은 그 자체로 *학습*=stage분리 지지하나 "value resolution=결정론" 단독입증 아님(IRNet/RAT-SQL/EG에 의존).

## Open Questions (M-A 동기)
1. **NL→API/tool-plan 세팅서 추상 인자-reference+결정론 value resolution이 literal-인자 직접생성을 이기나?** SQL/logical-form엔 증거 있으나 API-Bank/ToolLLM/Gorilla 1차증거 없음 = **타깃 설계의 가장 가까운 analog·여기 미소싱** → **M-A가 직접 실증할 빈칸**.
2. 현대 instruction-tuned LLM(in-context schema·Spider2.0/BIRD 스케일)서도 sketch/IR decouple·schema-as-input 전이 이득이 유지되나, 강한 LLM이 IR 상대이점을 지우나? (증거는 T5급서 멈춤)
3. 추상구조의 경험적 최적 **granularity**=얼마를 결정론 phase에 숨기나(IRNet은 FROM/JOIN/GROUP BY 숨기되 column reference 유지)? 직접 비교 claim 없음.
4. SCAN/COGS/CFQ 진짜 compositional-gen서 placeholder-then-resolve/pointer 설계가 측정이득, grammar-constrained와 상호작용?

## 인용 (1차 검증분)
IRNet/SemQL 1905.08205·P19-1444 / coarse-to-fine P18-1068 / sketch-based 1909.00574 / PICARD 2021.emnlp-main.779 / RAT-SQL 1911.04942 / execution-guided 1807.03100 / GCD-as-parser(form-not-meaning) 2025.acl-industry.34 / ToolLLM 2307.16789·Gorilla 2305.15334·API-Bank 2304.08244(소싱됐으나 검증 claim 미생존).
