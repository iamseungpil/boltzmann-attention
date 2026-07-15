# 설계서(확정본 v4) — banking per-step regime-partition probe (voting-solvable vs verify vs ASK)

> 상태: **설계 [D]·구현 v4·Phase0 게이트 통과·Phase1 대기.** v1→v2(6결함+§13+규율2)→v3(decidability 조작화·기회≠이득·⋈-first)→**v4(리뷰❶❷❸❹: decidability oracle/runtime 2분할·voting% universe 정정·selection bias 승격·abstain 분리)**. 2026-07-14→15.
> 큐: **E-REGIME**(신규·C88 파생·§4 등대 큐 등록). 목적 = C88 라우터{voting|verify|ASK}의 per-step regime partition 실측.
> 규율: [[08]] per-case·집계직행 금지·**교차표 1급** · [[05]] 엔진 도메인일반·측정만 · [[09]] 무료로컬(32B·gpt-4.1=0)·**Phase0 go/no-go** · [[47]] provenance · 사전등록 예측.
> 입력: C88 · `DRAFT_verifyorask_as_perstep_mechanism` · probe6(+0%·8/8) · C79(formalize)·C80(coverage)·C81(compute).

---

## 1. 답할 질문 → prefix/phase 매핑 (★fix③ 핵심)
| Q | 질문 | prefix | phase | 등급 |
|---|---|---|---|---|
| **Q2/Q3** (primary) | 각 실패 스텝을 {voting|verify|ASK} 어디로? 런타임 gold-free로 가능? | **actual-prefix**(재구성 0·라우터가 실제 보는 컨텍스트) | **1** | [M] 목표 |
| **Q1** (science) | 그 스텝 오류가 *내재적* systematic인가 self-conditioning인가 | gold-prefix(선행 gold 가정) | **2** | [D]·귀속전용 |

**primary 결과를 gold-prefix 재구성에 걸지 않는다.** actual-prefix만으로 within-step partition 완결. gold-prefix는 Q1(self-conditioning 귀속·horizon C82) 전용 mechanism 실험.

## 2. 두 correlation 분리 (범주오류 방지·유지)
⚠ within-step 재샘플상관(voting-vacuity·이 probe 신호) ≠ across-step self-conditioning(horizon·`2509.09677`·`2505.17656` 범주오류). **분리**: actual(Q2/Q3·라우터 현실) vs gold(Q1·내재 귀속). actual선 verify-needed인데 gold선 voting-solvable이면 그 스텝 실패원인=self-conditioning.
- ★해석 명시: **actual-prefix의 verify/ASK%는 self-conditioning 유발분을 포함**(라우터가 실제 맞닥뜨리는 need). Q2/Q3에 옳은 값. 내재적 하한은 gold-prefix(Phase2).

## 3. 측정 단위 = field-level (★fix §13a: coverage 제외)
- 단위 = gold dispute의 **target field 값**(C79/C81 field-level 정합).
- 버킷: **⋈**(`transaction_id`) · **compute**(`liability`·`provisional_credit`·`partial_refund`·`card_action`) · **gather**(`pin`·`police_report`·`card_last_4`…).
- **★coverage(action-emission/under-action/reach) 제외** — "어느 값?" resample 질문 아니라 plan/horizon 레버([[05]] 정합) → **E-PLAN([[14]])으로 이관**. 억지로 같은 표 금지.

## 4. 실패 universe 조건화 + greedy×maj@k 2×2 (★fix①: denominator가 C88의 전부)
- **분류는 "실패 스텝" 위에서만.** 실패 universe = **in-situ 실패 스텝**(C79/C80 포렌식 실패셋 상속·probe6 조건화 명시 계승). = "voting이 실패를 구제했나"의 분모.
- 각 스텝: greedy(T=0) 1샘플 → `greedy_ok` · k 유효샘플(T=0.7) → `maj@k` → `maj_ok`.
- **★1급 산출물 = greedy×maj@k 2×2 교차표**([[08]] 교차표 요건):

| | maj@k ok | maj@k wrong |
|---|---|---|
| **greedy ok** | A=easy(**partition 제외**) | B=voting-hurts(희귀·보고) |
| **greedy wrong** | **C=voting-win(MAKER셀)** | **D=voting-fail** |

- **voting% = C/(C+D)** (greedy-wrong 분모). easy(A)가 voting-solvable로 새어들어 부풀리는 것 차단.
- partition은 {C+D} 위에서: voting=C · verify=D∩decidable · ASK=D∩non-decidable.
- probe6의 +0% = C/(C+D)≈0(⋈)로 명시 상속.

## 5. 샘플 유효성 — infra 배제 / malformed 별도 (★fix⑤)
- **infra_error/timeout/crash = 환경 → 제외**(`k_valid` 기록). `k_valid<5`면 "측정불가" 별도.
- **★malformed(모델이 뱉은 파싱불가) = 노이즈 아님·별도 실패 카테고리**(can't-formalize·C79 영역). **버킷별 malformed율 + 난이도 상관 보고**(하드스텝 편중 시 partition을 인위적으로 깨끗하게 만드는 편향 = MAKER red-flagging은 이걸 신호로 취급). malformed는 partition서 제외하지 말고 자체 실패로 계상.

## 6. 측정 프로토콜 (★fix②: k·T)
- 모델: Qwen2.5-32B 서빙(localhost:8140·C79 재사용)·gpt-4.1=0.
- **k=8 primary**(probe6 비교) + **⋈에 k=32 필수 서브샘플**("<20% voting-solvable이 k↑에도 유지?" = MAKER "샘플부족" 반박 차단). **⋈ k-curve(k=2,4,8,16,32) 보고**(asymptote 실측=반박 종결).
- **T=0.7 primary + T=1.0 1점 서브샘플**(분산이 신호 → T-민감도 bound 필수).
- 산출(유효샘플만): `H_k`(field-값 엔트로피·gold불요) · `top_freq` · `maj_ok` · `gold∈support?` · `p_gold` · `greedy_ok` · `k_valid`·malformed율.

## 7. 분류 규칙 = decidability × voting-recoverability (★v4 리뷰❶: decidability 2분할)
- **★두 decidability를 분리**(v3가 하나로 합친 결함 정정):
  - **oracle_decidable**(=science 천장·gold 필요): gold파생 criteria로 filter가 gold 유일도달? = 구조적 resolvability(true-dup의 역수). **객관적이나 gold 필요=라우터 신호 아님.** verify vs ASK의 *이론 천장*·C78과 일치.
  - **runtime_decidable**(=gold-free 라우터 신호): 모델 formalize criteria로 filter가 **real id 산출**(maj가 abstain/malformed 아님). gold 불요·라우터가 실제 쓰는 신호.
  - **★gap = oracle_decidable ∧ ¬(runtime이 gold 도달) = formalization error = 라우터 맹점.** 이 gap이 §8의 핵심수치(v3 "자동 정합·감사 리스크 소멸"은 과장·철회).
- greedy-wrong(D) 스텝 → verify(oracle_decidable)/ASK(true-dup). verify 세부 abstain vs confident-wrong 분리(❸).
- greedy-wrong(D) 스텝:
  - decidable → **verify**. 세부(rerank vs compute): **gold∈support 서브분할은 k=8서 저파워**(진확률 5%→E[count 8]=0.4→대부분 ∉support로 오판→"compute" 편향). ⇒ **서브분할은 k=32에서만**·k=8선 gold∈support를 **하한(lower bound)으로만** 보고.
  - non-decidable → **ASK**(resample 무관·정보 모델 밖). resample은 guessing(고H_k) vs confident-default(저H_k) 확인용.

## 8. ★런타임 gold-free 신호 검증 (§13e 1급 승격·★fix④ 재설계)
라우터는 gold 모름 → gold-free 신호가 oracle-regime을 예측하나 = **§8 핵심 = oracle vs runtime의 gap(§7)**:
- **★1급 수치 = formalization gap**: oracle_decidable ∧ 32B-greedy-wrong 중, 런타임(runtime_decidable ∧ maj_ok)이 못 닿는 비율. gold-free 라우터가 이 gap을 못 좁히면 §8 실현성 미달·특허 라우터 물음표. (스모크에서 이미 산출: `report()` GAP 블록.)
- gold-free 신호 후보: ①**runtime_decidable**(maj가 real id·gold불요) ②**H_k**(엔트로피) ③**결정론 재계산 일치**(gate_spec 값 vs maj).
- **★H_k 단독 = strawman**: systematic-confident-wrong은 H_k≈0(gather confident-default와 동일) → 저엔트로피가 voting-solvable 구분 못 함. **H_k 단독 AUROC 나쁜 건 예정·라우터 실패 아님.**
- **성공조건**: {runtime_decidable, H_k} **결합 AUROC**(oracle-regime 예측) + **runtime_decidable 위 H_k의 marginal AUROC**. full-run서 산출.

## 9. self-conditioning 귀속 (Phase 2·gold-prefix·[D]·★fix③)
- gold-prefix 재구성(선행 gold action canonical 순서) → 같은 §4-8 측정. actual vs gold의 regime 이동 = self-conditioning 몫(Q1·horizon C82 귀속).
- ⚠ **재구성=distribution-shift 교란**(canonical 순서 가정 틀리면 off-distribution서 측정). ∴ **[D]등급·귀속 전용·primary(Q2/Q3) 결론을 여기 걸지 않음**.

## 10. 데이터 소스
- 궤적 `sim_results` + gold(3904 dispute·C80). 파서·결정점 = `bank_xmatch_forensic.py`·`bank_keystone_formalize.py` 재사용. 실패셋 = C79/C80 포렌식 상속.

## 11. 사전등록 예측 ([[08]]·post-hoc 금지)
- **⋈**: voting% = C/(C+D) **<20%** 예측(probe6 정합)·k=32서도 유지. → verify(C78)/ASK.
- **compute**: decidable·systematic 지배(같은 오공식 반복) 예측 → verify. 일부 stochastic slip=voting.
- **gather**: non-decidable → ASK(정의상)·resample=guessing 고H_k or confident-default.
- **gold-free(§8)**: 재계산일치가 decidable regime의 주 신호·H_k marginal 작을 것.

## 12. 반증조건 + 교란·caveat
- **반증(정직 보고 의무)**: ⋈/compute의 **큰 비율이 voting-win(C)** 이면 → voting이 잔여 덮음 → C88 약화·voting 편입 확대. / actual vs gold 차이 크면 → 실패=self-conditioning(within-step 아님)→horizon 재분류.
- **★1급 selection bias(v4·리뷰❷)**: ⋈ 케이스셋(853)=**frontier-failed ⋈점**(chosen≠gold)만 = frontier가 틀린 *더 어려운* ⋈점 편중 → **voting%을 아래로 편향(C88 과대평가 위험)**. voting%은 이 스코프서만 valid("frontier-failed ⋈점 위 32B voting%")·전체 ⋈ 모집단 일반화 금지. 순수 within-32B 청구엔 frontier-succeeded ⋈점 재추출 필요(미보유).
- **★1급 model mismatch(v4)**: probe 모델=32B ≠ 궤적 모델=frontier(gemini/gpt). 측정명제 = "32B-greedy-wrong 위 32B-voting" (2×2가 32B로 조건화·§16). 원-agent 실패는 universe 선택에만 관여(위 selection bias).
- **★fix⑥ per-cell CI/min-n**: 버킷×2prefix×{v/vf/ask} 셀 급감 → **셀별 최소 n 사전등록**·미만이면 %아니라 **count** 보고. partition %에 이항 CI.
- **★v3 경계 명시(기회≠실현이득·[[08]])**: E-REGIME은 per-action·offline(C79 상속)이라 **"voting이 못 닫는 잔여의 *크기*(기회)"를 잴 뿐, 라우터를 붙였을 때의 완주율↑(실현이득)이 아니다.** 실현이득은 **E-PLAN e2e([[14]])에서만**. "이만큼 개선"으로 과장 금지. C88 방어엔 기회-크기로 충분.
- k=8 엔트로피 소표본·field 파싱 노이즈(C79 상속)·per-action이 live보다 엄격(C79)·decidability=gate_spec 산출가능(엔진 커버리지에 의존).

## 13. 비용·시퀀싱 + ★Phase 0 go/no-go 게이트 (규율1)
0. **Phase 0(무료 스모크·n~30 ⋈)**: probe6 재현(8/8 systematic·voting%≈0) + 파이프라인 검증. **★GATE: 재현 실패 시 Phase 1 중단**([[09]]·깨진전제 추격 금지·설계·서버·파서 재점검).
1. **Phase 1(무료·actual-prefix)**: **★⋈ end-to-end 먼저**(k=8+k=32+2×2+partition+gold-free) → 방법검증+C88 헤드라인 답. **★GATE: ⋈ 결과 나오기 전 compute/gather 착수 금지**([[03]] 표류방지·전체 매트릭스 선빌드 금지). ⋈ GO 후 compute.
2. **Phase 2(무료·gold-prefix)**: self-conditioning 귀속(Q1·[D]).
3. **Phase 3(무료·gather 저순위)** + **분석**: 버킷 partition 표·gold-free AUROC·per-case 6건([[08]]·decidability 라벨 감사 포함).
- 전부 32B 로컬·gpt-4.1=0. 리모트 서버([[30]]) 상태 확인 선행.

## 14. 산출물
- `bank_regime_partition.py`(신규·엔진 도메인일반·resample+분류만).
- `..._results.jsonl`(per-step: bucket·prefix·k·H_k·greedy_ok·maj_ok·gold∈support·regime·k_valid·malformed).
- 표: 버킷 × {greedy×maj@k 2×2} × {voting%/verify%/ASK%}(CI/count) × {actual/gold} · ⋈ k-curve · gold-free AUROC(결합+marginal).
- ledger C88 partition 실측 갱신 · E-REGIME 상태.

## 16. 구현·검증 로그
- 스크립트: `scripts/distill/tau2/bank_regime_partition.py`(⋈·`--dry` 오프라인/full resample·기존 filter 재사용·v4).
- **`--dry` 오프라인(무료)**: `bank_xmatch_cases`(853) · **oracle_decidable 699(82.0%) = C78 81.9% 재현** → **oracle-decidability 엔진이 C78과 일치**(정의검증 = *오라클* decidability 한정·라우팅 정의검증 아님·리뷰❶). true-dup(ASK 천장) 154(18.0%). ⚠❹: **다중-dispute 맥락(n_disputes≥2)=852는 true-dup율 18%와 별개 개념**(전자=유저가 여러 건 dispute·후자=record 속성충돌).
- ⚠853 = C77-79 추출 ⋈셋·C80 궤적 coverage 오염분·**전부 frontier-failed(§12 selection bias 1급)**. E-REGIME=formalize→filter 스텝 voting-solvability(궤적 귀속=C80 별건). 측정명제="32B-greedy-wrong 위 32B-voting이 gold 도달?"(2×2로 조건화·§12 model mismatch).
- **★Phase 0 게이트 통과(2026-07-15·서버 up·n=30·k=8·무료)**: **voting% = 0/29 = 0.0%** [CI 0–11.7]·**gold∈support 0/29**(정답이 8샘플에 단 한 번도 안 나옴=최대 systematic)·malformed 0 → **probe6(+0%·systematic) 재현**. partition: voting 0·verify 25·ASK 4. ⚠v4 재런으로 oracle/runtime gap·abstain 분리 산출 예정.
- **다음 = v4 재런(Phase 0 재확인·gap 산출) → Phase 1**(⋈ full·k=8+k=32·per-case 정독).

## 15. 리뷰 확정 (§13 사용자 답변 반영·CLOSED)
(a) field-level primary·**coverage 제외→E-PLAN** ✅ / (b) actual 먼저·gold=Phase2[D] ✅ / (c) k=8+⋈k=32·T=0.7+T=1.0 ✅ / (d) ⋈>compute>gather(coverage 이관) ✅ / (e) §8 1급·결합예측+marginal+재계산신호 ✅.
