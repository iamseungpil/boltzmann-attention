# HANDOFF 2026-06-14 — day-6 (★결정론 불변 확립·thesis 구조 재확인·선별기 same-base 포화·생성기=레버·결정론 라인 강등·프레임 천장 실측)

> 📌 마스터 = `EXPERIMENT_DESIGN.md`(§0 목표·§7 문서지도·§1.6 메트릭). 결과 권위 = `reports/facet_rft_2026/TASKBENCH_EXPERIMENT_RESULTS.md`·`BENCH_PORTFOLIO_FRAMEWORK_DESIGN.md` §3.7. 선별기 = `SELECTOR_DESIGN.md`. 엣지 재설계 = `taskbench/EDGE_LEVEL_REDESIGN_2026_06_14.md`. 리모트 규칙 = memory `reference-remote-server-environment`. 직전 = `HANDOFF_2026_06_14_day5.md`(승계).

## 0. ★★★ 최우선 불변 (이번 세션 확립 — 매 세션 적용)
**선택기·검증기 = 둘 다 *결정론 머신*. LLM은 생성기(swappable)에만.** LLM-judge/logprob/reverse-likelihood/self-cert 선별·검증 신호 **제안·구현 금지**(실증 전멸 §1·thesis 위배). "독립 검증기"=다른-base LLM 아닌 **결정론 검증기 심화**(타입→인자-타입→사전조건→replay). = memory `feedback-selector-verifier-deterministic`(★★★)·SELECTOR §1 불변.

## 1. ★첫 행동 (순서대로)
1. **agent 격리 결정론 테스트 수확 (비핵심)**: `cat /home/woori/scratch/agent_det.log`(미완 시 재발사 불요 — 결정론 라인 강등[§아래]으로 *기제 확인용*일 뿐). 8% near-tie 커널 비결정 확인되면 PORTFOLIO §3.7d 기제 1줄.
2. **★A2 학습 front-end 착수 (§0 thesis core — 이번 세션 *유일 미착수* 핵심)**: 지금까지 frontier(Fable-5)로 A2 GATE_SPEC 수동 컴파일 → 프레임 천장만 확인(§1 천장). **다음 = 작은 모델이 NL 정책→GATE_SPEC 자동생성(TBox)** = §0 상품형태. 부트스트랩 자산 존재(`tau2/t2_a2_*` — spec 샘플러·역렌더·round-trip·**faithfulness 게이트(이번 세션 신규)**). S0 합성→S1 verified-distill(StepFun ThinkingF 템플릿)→S2 DPO.
3. **잔여 큐**: ①frontier 천장(pass^1 0.81)을 **학습 7B-A2로 재현** 시도 ②선별기 헤드룸은 생성기-side(아래 §1)라 — diffusion/hot-AR 엣지-커버리지 실험(matched-entropy AR 대조·greedy 금지) ③τ² G4 deny-게이트 검증.

## 2. day-6 확정 결과 (전부 권위본·커밋)
- **★relwork 6편 수확·정정 통합**: 인용규율 정정 **10건**(`2601.15808` 철회·StepFun 7B 역전·P2P 76.5% 분모·§1.6 metrics·PORTFOLIO determinism 레시피 등)·**유일 fabrication=ATLAS-RTC `2603.27905` drop**·`related_work_INDEX_2026_06_14.md` 작성.
- **★A2 faithfulness 게이트 (`tau2/t2_a2_faithfulness.py`)**: cross-stage NL-gloss↔source entailment. selftest PASS·retail judge 3/3 SUPPORTED·**fabtest G9 FABRICATED(conf1.0)=replay 사각지대 폐쇄 실증**. (검증기는 LLM-judge지만 = *A2 데이터청정/faithfulness*용이지 런타임 선별기 아님 — 불변과 무충돌.)
- **★diffusion/A3/생성기 라인 재정렬**: P-D0 전수부검=0%는 decode-붕괴 artifact(steps<gen_len)+직렬화약점(DiG-Plan 재현)으로 분해·**diffusion verdict 불가**. A3 any-order 실험=기각(relwork_arch §3c). **단 oracle 엣지분석 후 재검토(§아래): 생성기가 *진짜 레버* → diffusion-as-generator 재정당화**(0.943=합성토이 정정·실제 Oracle@10 +0.052·tool-SET강/엣지약 → 엣지-커버리지를 matched-entropy AR 대조 측정). = `TB_DIFFUSION §3 재검토`.
- **★선별기 same-base 신호 *전수 음성* (천장 진단 라운드)**: SEL-5 pairwise judge **기각**(0.669<SEL-1 0.672)·XGrammar floor **한계**(+0.16 cand/id·validity≠D-oracle)·**C 갭분해**(독립-group 기준: selectable 50.7%/needle 42%/gold-limited 7% — distinct 기준의 3.6%는 오분류)·**B1 self-certainty 음성**(agreement 0.673≈SEL-1·logprob 0.664<SEL-1, #3 퇴화해). = SELECTOR §7·§8·TB §8.9i.
- **★엣지-레벨 재설계 + 한계**(`EDGE_LEVEL_REDESIGN`): oracle을 plan-atomic→엣지로 재정의(조립-oracle 0.858>단일 0.822). **엣지-조립 선별 = 공식 link-F1 0.6766 ≈ plan-MBR 0.6803(무초과)** — 내부 +1.5pp는 척도 아티팩트. **★결정론 검증기는 "지시-맹목"**(graph_desc=타입-호환과 동일·gold 99.7%/wrong 63.7%): 구조검증으론 task-부적절 엣지 못 거름 → **지시-그라운딩은 생성기만 보유 = 헤드룸은 생성기-side**.
- **★thesis 구조 재확인 (사용자 framing 검증)**: §0=작은모델이 **NL정책→결정론 GATE_SPEC 컴파일(TBox)**·ABox-swap 재학습0 전이. 프레임=**A2(학습 컴파일) → 결정론 검증기(=GATE_SPEC replay)+선택기(R6 합의)+XGrammar(validity) → 다양생성기 K-계획 → compliant pass^k(F4b)**. 헤드라인 차별=**주권+LODO 전이**(raw accuracy 아님). 사용자 framing ✅(정밀화: A2=검증기 *자체*·선택기/XGrammar는 동반 결정론 기계).
- **★★결정론(ⓟ1) 라인 *강등***: det 런 = ⓓ2(seqs1+temp0+seed로도 4-trial **0/39 동일**). **전수 궤적 census = 비결정 92% gpt-4.1 user-sim API / 8% agent vLLM**(이전 "user-sim 분산 아님"=user temp 한정·API비결정 놓침 정정). **재구성(사용자)**: ①다양-생성기 프레임은 검증기가 compliance를 생성기-비결정 무관 보장 → **LLM 결정화 핵심 비요구** ②pass^k 재현성=프레임 가치지표 아님(가치=compliance+generate-K-select oracle@K) ③determinism 실험 원래이유=F3 "게이트=분산제거"의 2차주장 ⓟ1 측정-위생인데 분산 92%가 user-sim이라 측정불가 근접. ⇒ **pass^4 낮음=다양성 설계의도. 결정론은 리더보드 pass^k비교·RL off-policy·디버깅 한정.** = PORTFOLIO §3.7d·EXPERIMENT §1.6.
- **★프레임 천장 실측 (§3.7f)**: **gpt-4.1 + Fable-5 A2 gate × K=4 = 위반0·compliant pass^1 0.81/pass^4 0.65**(vs 7B+gate 0.17/0.026·oracle@4 0.41). ⇒ **결정론 A2 gate=compliance 천장 보장(model-agnostic)·compliant-pass 천장=생성기 품질** = 생성기가 레버 재확인. 프레임 작동 실증(frontier-A2 한정·다음=학습-A2 재현).

## 3. 인프라 gotcha (day-6)
1. **ssh_run cwd**: git 명령이 cwd를 ba-frft로 남김 → 다음 ssh_run `py -3 ssh_run.py`가 "파일없음". **ssh_run 호출은 항상 `cd /c/workspace` 선행**(별도 Bash 호출 분리).
2. **t2_run_gated 기본 temp=0 전부**(agent/user/judge). agent_seed는 로컬-vllm 전용(gpt-4.1 미적용). 다양성 통제하려면 명시적 agent temp>0 또는 다양 어댑터/seed.
3. **gpt-4.1 user-sim = temp0서도 비결정(API)** → 멀티턴 궤적 결정론엔 user-sim도 결정론화(scripted/cached) 필요. agent batch-invariant만으론 8%만.
4. **리모트 venv outbound HTTPS** = `SSL_CERT_FILE=$(python -c "import certifi;certifi.where()")` 필요(openrouter judge).
5. **det serve 설정** = `--enforce-eager --max-num-seqs 1 --seed 0` (port 8351). 단 0% 동일 = 불충분(batch-invariant ≥0.11.1 필요).

## 4. 메타 (day-6 규율)
- **★불변 확립의 값**: LLM 선별/검증 신호 5종(SEL-2/4/5·B1 agreement/logprob) *전부* 실증 음성 → "선택기·검증기=결정론" 불변을 박제. drift 차단.
- **★zero-cost 진단이 GPU 실험 예측-대체**: 사용자 "agreement zero-GPU 먼저" → B1 logprob(GPU) 음성을 정확히 예측. C 갭분해(독립-group 보정)가 "selectable 3.6%→50.7%" 뒤집음.
- **★편의-데이터 과대해석 3연속 교정**(memory `feedback-no-fundamental-claims-from-convenience-data`): "천장·비가역"은 임의 풀 한정 진단치·기제만 일반화. 양적 결론 일반화 금지.
- **★사용자 재구성이 라인 강등 2건**: ①생성기 강등→재정당화(생성기=레버) ②결정론 라인 강등(핵심 비요구). 옛 데이터서 천장 읽기 금지·설계변수로 다루기.
