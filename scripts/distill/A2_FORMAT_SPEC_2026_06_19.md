# A2 포맷 스펙 — LLM이 결정론(form+meaning)으로 동작하는 타입드 합성 스키마 (2026-06-19)

> **자립 문서**(리뷰용). 목적 = A2(도메인 config)의 *포맷*을 정해 LLM→엔진 핸드오프를 결정론으로. 상위 = `A2_MINIMIZATION_FRONTIER_DESIGN`(A2를 *최소화*·여기선 *포맷*). 근거 = `dr_nl_to_formal_interface_granularity`(reference 방출·grammar=form·schema-transfer 24 confirmed)·ATA `2510.16381`(균일 FOL 대조). 메모리 = `00-thesis`·`05-fixed-vs-variable`·`02-generator-algebra`.

## 0. 목적 — "LLM이 결정론적으로"의 실체
LLM 자체는 결정론 아님(샘플링). 대신 **출력을 *제약***: ①LLM이 **닫힌 타입드 op-vocab으로 reference(pointer) emit** ②**grammar-constrained decoding이 *형식(form)* 보장** ③**엔진이 A2 위에서 reference를 *결정론* resolve(meaning)**. = form은 grammar·meaning은 엔진, 둘 다 보장.

## 1. A2 = 3부 타입드 합성 스키마 (단일 포맷 아님·digraph는 백본)
| 부분 | 포맷 | 담당 P | 엔진 연산 |
|---|---|---|---|
| **① dataflow** | **타입드 DAG**: 도구=노드·typed I/O 포트·`produces/consumes` 엣지 | P2b·P3·P9·P4-grounding | 의존 topo-sort·producer 선행 강제·which-producer 도출(type-match) |
| **② catalog** | **relational**: items × attributes (option 키 명시) | P4-select | filter/argmax/rank over rows(주어진 기준) |
| **③ policy** | **rule-set**: `precond ⇒ allowed`·irreversible-flag·auth/scope | P5·P6·P8 | gate 집행(soundness G1-G4) |
- **digraph(①)가 중심**(=TaskBench 형식화)·②③이 거기 붙음. (P1 무날조·P7 recovery = ①② 위 *연산 규약*이지 별 데이터 아님.)

## 2. LLM formalize 출력 = 닫힌 타입드 op + reference (literal 금지)
- **op ∈ 닫힌 유한 vocab**(생성원 기저: flow P1-P9 + content 8-op). = grammar 단말.
- **인자 = A2 원소로의 *reference/pointer***(도구명·value-type·catalog attr·op·기준) — **literal 값 생성 *금지*** (= P1 무날조·dr_nl_to_formal "reference 방출" 패턴·order_id 날조 차단).
- 예: `resolve_selection(op=argmin, attr=<catalog.price>, among={<catalog.type>: "smart"})` — 전부 A2 reference + 닫힌 op. concrete item_id는 *엔진이* resolve.

## 3. 파이프라인 (form/meaning 분리)
```
NL  ──(LLM·grammar-constrained decoding)──▶  typed-op + A2-reference   [form 보장]
                                                      │
                                            (엔진·결정론 resolve over A2)
                                                      ▼
   ①DAG: producer 선행·grounding   ②catalog: select   ③policy: gate    [meaning 보장]
                                                      ▼
                                              concrete action (or abstain)
```
- LLM이 reference를 ground 못 할 상태(상품 미fetch 등) → 엔진이 abstain 신호 → 학습된 P7 복구(fetch/ask) → loop-bound escalate. (= `ABSTENTION` 분해.)

## 4. vs ATA (균일 FOL/SMT) — 왜 타입드 합성이 나은가
- ATA = 전부 FOL·z3·**열린 target**(임의 공식·LLM 오답 가능·grammar 제약 불가·closure 없음)·SMT 무거움·도메인별 재형식화.
- 우리 = **구조 정합**(dataflow=DAG·catalog=relational·policy=rules) + **닫힌 op-vocab**(grammar-constrained 가능·환각↓·type-check) + **경량 엔진**(graph+relational+rule, full SMT 불요) + **schema-as-input 전이**(RAT-SQL/IRNet식 cross-domain·=ABox-swap).

## 5. closure = 이 포맷을 *가능케 하는* 성질
유한 타입드 op-vocab이라야 — formalize target이 **닫힌 grammar**(제약가능)·**완전**(census orphan=0)·**groundable**(엔진 resolve). **열린 포맷(ATA FOL·open API)은 grammar 제약·완전성·deterministic-resolve 불가.** ⇒ 우리 closure 주장의 *공학적 귀결* = "A2를 제약-디코딩+결정론-resolve 포맷으로 만들 수 있다."

## 6. P1-P9 → A2부분 × 엔진연산 매핑
- P1(무날조): 모든 인자=reference 강제(literal 거부) — ①② 가드.
- P2b(gather-for-arg): ① DAG서 needed-type의 producer 선행 호출 강제.
- P3(sequence): ① DAG topo-order.
- P4(select): LLM op+기준(reference) → ② relational select(결정론).
- P5/P6/P8(gate): ③ rule-set 집행.
- P7(recovery): ① abstain 신호 → 학습 복구.  P9: ① 독립노드 병렬.
- **★NL→formalize(op-recognition + P4 기준) = LLM 환원불가** / P1-9 실행 = ①②③ 위 결정론.

## 7. 자가심사
- **치팅**: literal 금지(reference만)=날조 구조차단. op∈닫힌vocab=grammar/type-check. 엔진 resolve=결정론·soundness 게이트 별도. real 도구 미대체.
- **thesis정합**: 학습=formalize(reference emit)+coverage / A2=타입드 사실(3부) / scaffold=엔진+게이트 고정. ([[05-fixed-vs-variable]])
- **정직**: "가장 효과적 포맷"은 *원리*(타입드·닫힘·referenceable·grammar+엔진) 확정·구체 스키마(필드명세)는 도메인별 미정·실증 필요. grammar-constrained decoding 이득은 base 강할수록 diminishing(dr caveat).

## 8. 미결 / 다음
- 3부 스키마의 *구체 필드 명세*(tool I/O 타입·catalog attr·rule DSL) — 한 도메인(retail) 1차.
- formalize 출력 grammar(닫힌 op + reference 문법) 정의 → xgrammar/제약 디코딩 연결.
- 엔진 resolve 규약을 ①②③별로 (현 `t2_resolve_patch`는 ② 일부만·①DAG·③rule 미통합).
- = `A2_MINIMIZATION` S0(decidable-ablation)의 *구현 전제*: 엔진이 주어진 A2(이 포맷)를 물어야 grounding 성공.
