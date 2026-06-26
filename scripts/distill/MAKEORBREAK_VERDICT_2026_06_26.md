# Make-or-break 종결 판정 (2026-06-26) — tau2 retail·통제 실험·gpt-4.1 0원

> 질문: scaffold(present/gate/calc/constraint) 다 붙인 뒤 *남는* operand 실패가 **학습(faithful-formalize SFT)해야 할 capability gap**인가? 전 과정 로컬 Qwen + 기존 데이터 + 통제 isolated probe로 답(gpt-4.1 0). 정본 데이터=`reports/facet_rft_2026/sim_results/`(영속화).

## 통제 실험 (user-sim 입력=통제변인)
| 조건 | 정답률 | 해석 |
|---|--:|---|
| 변형-pick **GIVEN-SPEC**(gold 옵션 명시→매칭) | **88/88 (100%)** | **순수 operand 실행능력=완벽** |
| 변형-pick GOAL(reason 목표만) | 62/88 (70%) | +criterion 해석 |
| **갭** | **30%** | 목표→spec 해석 부하(argmax/argmin=compute·다항목reason=mis-attribute) |
| genuine-⋈(유저 id 안줌·묘사·fair) | 7/13 (54%) | 진짜 disambiguation(단 present-addressable·t71 full-flow서 닫힘) |
| ⋈ ID-GIVEN(제외) | 11건 | inference-probe 불공정(이전 49%/56% 오염원) |

## 결론 (rigorous)
1. **operand 실행 capability gap = 없음 (32B 100% given-spec).** spec 주면 작은 모델이 매번 정확.
2. **30% 갭 = criterion-해석** = 결정론 compute(argmax/argmin) + 대화 fidelity(충실 user-sim/present면 100% 근접). **learn 아님.**
3. **genuine-⋈ 54%(n=13) = 유일 실제 잔여**·present-개선으로 닫히는 부분(t71 full-flow 실증). learn 아니라 disambiguation/present.
4. ⇒ **faithful-formalize SFT = NO-GO 확정.** **헤드라인 = 결정론 scaffold(present/resolve/compute) + base 모델 innate skill + TCO**([[06]] 정합).

## ★측정 교훈 (이 arc 반복)
operand "실패"를 정밀 측정할 때마다 confound 노출: calc 버그(미발화)→t71 1건 성급결론→⋈ probe 아티팩트(id-given)→통제하니 실행=100%. **operand 잔여는 robust 에이전트 gap이 아니라 측정/대화 fidelity 산물.** ([[08]] 전수포렌식·집계직행금지의 반복 실증.)

## TBox 함의 (원 계획 TBox+A2+scaffold)
- **TBox 두 의미 구분**: (a) *모델이 NL→formalize→select 하는 번역자 역할* = 본질·base가 이미 함(100% given-spec) (b) *잔여 닫으려 SFT로 설치하는 도메인-일반 스킬* = tau2서 **불요**(닫을 잔여가 없음).
- ⇒ **(b) TBox-학습 wing은 tau2 make-or-break서 carry 안 함.** 계획이 "결정론 + 학습된TBox"에서 **"A2(수작성)+scaffold(결정론 offload)+base 모델(innate 번역자·SFT 불요)+TCO"**로 수렴. = thesis §2(LLM=boundary translator)와 정합·learn-wing은 tau2서 미검증/불요.
- **범위 정직**: 32B 결과(7B는 given-spec도 낮을 수 → 소형선 TBox 여지 잔존·미측정). criterion-해석의 messy-NL-parse 부분에 작은 learn 여지 가능. = 후속 확인거리이지 tau2 operand make-or-break의 결론(SFT NO-GO)은 robust.
