# Rung1 v3 — grounded 트리평가 게이트: 선행연구(적대검증) + 레시피

> 2026-06-03. deep-research(110 에이전트·27 소스→112 주장→25 검증, 23 confirmed/2 killed) 종합.
> 동기: Exp-4-rung1-T1T2에서 **AND(preconds)는 0오류로 작동, `permitted`만 콜드붕괴**(조건수↑서 악화, 21% OR). 처방 후보 = permitted을 "per-leaf truth→트리집계 derivation"으로 grounded SFT.
> 권위본 결과 = `reports/facet_rft_2026/SOPBENCH_EXPERIMENT_RESULTS.md` Exp-4-rung1-T1T2. 이 문서 = 그 처방의 *문헌 근거 + 구현 레시피*.

## 1. 한 줄 결론
**"grounded 트리평가 + derivation supervision"의 *평가* 절반은 문헌이 강하게 지지**(우리 실측까지 이론이 직접 설명). **"NL→트리 추론 + held-out 전이 + 깊이 일반화" 절반은 미검증 = 진짜 리스크.**

## 2. 검증된 선행연구
| # | 논문 (arXiv, venue) | 결과 | 판정 |
|---|---|---|---|
| 1 | Kim & Suzuki 2410.08633 (ICLR25 Oral) | 중간단계 loss 포함시 parity 1-step 학습, 없으면 불가 = 증명된 효율분리 | ✅ SUPPORTED(analogical) |
| 2 | Feng 2305.15408 (NeurIPS23) · 2402.12875 (ICLR24) | no-CoT const-depth = AC0/TC0(serial boolean 불가) / CoT면 size-T boolean-circuit 평가(CVP 실증). cf Merrill&Sabharwal 2207.00729 | ✅ SUPPORTED(expressivity) |
| 3 | Bhattamishra 2211.12316 (ACL23) · Wang 2412.02823 | low-sensitivity 편향: flat-AND(O(1))=거의완벽/parity(고민감도)=붕괴. ICL정확도↔boolean식복잡도 r=−0.88(Qwen2-7B) | ✅✅ 우리 실측 직접 설명 |
| 4 | Abbe 2406.06467 (NeurIPS24) | globality 장벽; inductive(구조적) scratchpad만 깸(agnostic 안 됨, educated도 일부만 OOD) | ✅ SUPPORTED |
| 5 | He 2512.02677 · Lakretz 2101.02258 | 깊이 일반화=길이와 별개 실패축, depth↑서 chance 붕괴(PCC −0.92). ⚠️GPT-2급 preprint·fix는 inference-time | ⚠️ RISK |
| 6 | Dziri 2305.18654 (NeurIPS23) | 트랜스포머="linearized subgraph matching", 복잡도↑서 error→1. 분해 *필요성* 지지 | ✅(negative) |
| 7 | Evans 1802.08535 (ICLR18) · Bowman 1406.1827 | 트리구조 net이 logic syntax 더 활용; 의미열거(PossibleWorldNets) 최고. 단 from-scratch·LLM/전이 아님 | ◐ MEDIUM |
| 8 | Jackson et al. (random monotone DNF, RANDOM08) | OR(disjunctive) 구조 복원이 난점의 핵심(co-occurrence↔low-order Fourier). 고전 PAC이론(신경망 아님) | ◐ MEDIUM |

**기각(2)**: RASP-L 길이일반화 예측기준 + "기본적으로 길이일반화 실패"(Zhou 2310.16028) = 1-2 kill. 예측기준으로 신뢰 금지.

## 3. 우리 설계 — SUPPORTED vs ANALOGICAL/UNSUPPORTED (정직)
| 설계 선택 | 판정 | 근거 |
|---|---|---|
| grounded step별 derivation > 콜드 게이트 | ✅ SUPPORTED | #1,2,4 |
| flat-AND 쉬움 / 콜드·고민감도 붕괴(조건수↑ 악화) | ✅✅ 직접 | #3 |
| 구조적(inductive) scratchpad(free-form 아님) | ✅ SUPPORTED | #4 |
| held-out 도메인 zero-retrain 전이 | ❌ 미검증 | 트리평가 전이 실증 없음 |
| NL→트리 구조 추론(구조 마스킹·anti-cheat) | ❌ 미검증 | parse-from-NL 절반 검증주장 0 |
| 깊이/조건수 robustness | ⚠️ RISK | #5 rapid-decay |

⚠️**과대주장 경계**: 대부분 1-layer/from-scratch 합성(parity/CVP/Boolean algebra)·GPT-2급 — **7B LoRA tool-use 전이 아님**. expressivity 증명 ≠ SFT가 그 해를 복원한다는 보장. parity=XOR는 AND/OR의 *최악 케이스*(AC0 밖)라 "콜드 고민감도 붕괴"는 지지하나 grounded flat-AND를 직접 bound 안 함.

## 4. ★재활용 레시피 (v3 teacher 처방)
리포트 추출 + 우리 코드 정합:
1. **per-leaf truth를 먼저 emit → 그 다음 트리(AND/OR/chain)대로 집계** (inductive/educated scratchpad; Abbe). 예: `cond[credit]=T; cond[balance]=T; gate = AND(credit,balance) = T; ACT`.
2. **중간 집계결과를 SFT loss에 포함**(assistant-target; Kim&Suzuki — 효율분리의 핵심 기제).
3. **leaf 읽기는 *기록값*(게더 RESULT) 룩업**으로 — 재추론 금지(현 `permitted` 콜드추측 폐기).
4. **재귀 locate-and-evaluate 분해**를 derivation 템플릿으로(He): 트리 잎부터 위로.
5. **현 2-토큰 통합**: `permitted`을 콜드 should_succeed 라벨이 아니라 **per-leaf truth의 트리집계 derivation**으로 생성 → preconds와 동형(둘 다 grounded). OR/chain은 트리집계서 자연 처리(현 flat-AND 21% 오류 해소).

## 5. 진짜 리스크 = 다음에 *측정*할 것 (열린 질문)
1. **NL→트리 추론 + 전이** (최대 미커버): 7B가 구조-마스킹 NL서 AND/OR/chain을 추론하고 held-out 전이하는지 — 선행 무 → **우리가 1급 결과로 증명/반증해야**. (마스킹이 일반화를 *강제*하는지 *저하*시키는지도 미지.)
2. **깊이/조건수 일반화**: grounded read-back 집계가 depth-decay(#5)를 피하는가 — 기록값 룩업이라 피할 *가능성*, 미검증 → ablation(조건수 2/4/6/8).
3. **OR이 grounded 평가서도 더 어려운가**: 민감도/DNF이론은 *구조복원*이 어렵다 하나 *주어진 OR-트리 평가*는 별개 → 21% OR 직접 측정.
4. **process vs outcome(게이트 한정)**: PRM/least-to-most/Jia 반례는 검증주장 0으로 남음 → parity 효율분리로 간접지지만, 게이트 직접 PRM 증거는 우리 실험 몫.

## 6. 다음 단계
v3 teacher 구현(§4 레시피) → bank LODO 재학습 → 측정: BOTH(헤드라인) + **ACT-recall|게더 + STOP-recall 분리** + **조건수별 분해**(2/4/6/8로 depth-decay 검사) + **OR-케이스 정확도**(21% 직접). 게이트: BOTH가 G-SFT(≥15) 넘고 조건수-곡선이 평평하면 grounded 트리평가 성공.

## 7. 소스 (primary, arXiv)
2410.08633 · 2305.15408 · 2402.12875 · 2406.06467(NeurIPS proc) · 2211.12316 · 2412.02823 · 2512.02677 · 2305.18654 · 1802.08535 · 1406.1827 · 2207.00729 · (DNF: Jackson/Lee/Servedio/Wan RANDOM08) · 2101.02258. 기각: 2310.16028.
