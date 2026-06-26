# 리더보드 지표 근거 + 정당 비교 기준 (2026-06-05, 06-06 quirk 분석 추가)
> 📌 **구조 안내**: 모든 설계·실험 문서의 단일 마스터 = repo `scripts/distill/EXPERIMENT_DESIGN.md` (**§7 문서지도**에서 각 문서의 역할·상태 확인; 목표·순서 변경은 마스터 §0-§4에서만). 처음 읽는다면 마스터부터.

> 질문: 리더보드가 공식 success를 쓰나? 논문/리더보드 비교 시 정당한 기준인가?
> 결론: **리더보드 = 공식 `success` pass rate %, 전체 134 태스크(should_T 48 + should_F 86), tool_full.** 프로젝트 헤드라인 BOTH(dg∧acc, should_T만)는 **리더보드와 지표·분모 둘 다 달라 비교 불가**. 정당 비교 기준 = 공식 success.

## ★★0. quirk(failed-login-but-passed) 분석 — honest 비교 (2026-06-06, `diag_quirk_rescore.py`)
**login-quirk** = evaluator dirgraph가 login을 "성공"이 아니라 "호출(call-order)"로 카운트(`dfscheck_called_functions`) → 가짜 비번 login(False)도 login 노드 충족. (상세 = `INTERNAL_GET_DATABASE_GROUNDING`/사용자 논의.)
- **quirk-악용의 정의(좁게)**: `should_T` 태스크가 login을 **호출했으나 전부 False**인데도 success. (should_F의 failed-login은 **정당한 거부**=success이지 quirk 아님 → should_T로 한정해야 함.)
- **★리더보드 모델 should_T quirk ≈ 0** (전 모델 0~2): 리더보드는 quirk로 should_T를 부풀리지 않음. failed-login은 전부 should_F 정당거부. ⇒ 리더보드 공식 success ≈ honest success.
- **★우리 S1 should_T 40 중 quirk 8** (LOGINCALL이 만든 것; loginfirst 1→logincall 4→S1 8). LOGINCALL의 official +3(should_T 25→28)은 **전부 quirk**(honest should_T 24 불변). ⇒ **우리만 quirk 사용 = 불공정 → LOGINCALL 드롭.**
- **★honest(quirk-out) 비교**: 우리 S1 official **50.75%(68/134)** → **honest 44.78%(60/134)** (should_T quirk 8 제외). 리더보드는 quirk≈0이라 official≈honest. **honest 44.78%도 오픈소스 SOTA(Llama3.1-70B 42.54%) 추월.** ⇒ **quirk 없이도 7B가 오픈소스 70B 추월** = 더 깨끗한 헤드라인.
- **헤드라인 권고**: honest(quirk-out) **44.78%(60/134)** 를 1급, official(quirk-in) 50.75% 는 참조. **LOGINCALL 드롭**(quirk 제거)이 정직·공정.
- ⚠️ 방법론 교훈(자가 2회 정정, 사용자 교정): quirk 측정은 ① should_T/should_F **분리 필수**(should_F failed-login=정당거부) ② login은 OR-분기라 "그래프에 login 노드 있음=필수" 오판 금지(v1 과잉처벌) ③ 좁은 시그니처(호출됨∧전부False∧should_T)만 신뢰.

## 1. 리더보드 지표 = 공식 success (확정)
- `env/evaluator.py:277` `success = no_tool_call_error ∧ constraint_not_violated ∧ database_match ∧ action_called_correctly ∧ dirgraph_satisfied`.
- `interaction_statistics`/`domain_statistics`: `pass@1`/`mean_pass_rate` = `eval_res["success"]` 평균. README "model **pass rates (%)**".
- **검증**: 오픈소스 전 모델 README 값 = 우리 재계산(공식 success, 134, tool_full) **정확 일치**: Qwen2.5-7B ReAct 5.22%(7/134)·Llama70B 42.54%(57/134)·Qwen72B 35.07%·Qwen32B 40.30%·Qwen14B 35.07%·Llama8B 14.93%. ⇒ 의심의 여지 없음.
- tool_full = 리더보드 설정(oracle 변형은 별도·더 높음). react/fc/act-only 중 README 오픈소스=react tool_full.

## 2. 프로젝트 BOTH(dg∧acc)는 리더보드 비교 불가
- BOTH = `dirgraph_satisfied ∧ action_successfully_called`, **should_T 48만**. 공식 success는 **+cnv+dbm+ntce**, **전체 134**.
- BOTH는 success를 8~12 과대계상(goal-call looping이 cnv/dbm 깨도 dg∧acc는 통과; should_F 미포함).
- ⇒ 사다리 헤드라인(15→…→40 BOTH)은 **리더보드 숫자가 아님**. 논문/리더보드 대조는 공식 success로만.

## 3. 우리 런 = 공식 success (같은 기준)
| run | all%(134) | should_T | should_F |
|---|---|---|---|
| base_noaug (scaffold, no fix) | 29.85% (40) | 22/48 | 18/86 |
| + Fix1 loginfirst | 37.31% (50) | 25/48 | 25/86 |
| + Fix2 logincall | **40.30% (54)** | 28/48 | 26/86 |
- base 모델 Qwen2.5-7B-Instruct: FC tool_full **3.73%**(5/134)·ReAct **5.22%**(7/134).
- 우리 tbox_v2(Qwen2.5-7B + SFT + DGGATE/LOGINFIRST/LOGINCALL/ARGFIX/VALFIX/KEEPTUPLE) = **40.30%**.
- ⇒ **7B base ~5% → 40.30%** (공식 지표). Llama3.1-70B(42.54%) 근접·동급 7B로 대형 추월권.

## 4. 정당성·caveat
- **정당 비교 기준 = 공식 success pass rate(134, tool_full).** 이걸로만 리더보드/논문과 같은 표에 올릴 수 있다.
- **profile skew**: 우리는 should_T 강함(28/48=58%, 대형 모델급)·**should_F 약함(26/86=30%)**. 리더보드 상위는 should_F가 높음(claude react 65/86, gpt-5 78/86). 전체% 끌어올리려면 should_F(거부축)도 개선 필요.
- **scaffold 비교 caveat**: 우리는 scaffold+SFT 7B vs 리더보드의 unscaffolded 모델. "7B+scaffold가 공식 지표 40.30%" 주장은 정당하나, scaffold가 큰 기여(논문서 명시 필요).
- **Fix2 logincall**은 공식 success +3(50→54)만(BOTH +7은 허상). looping 차단(Fix3)이 should_T full_success의 최대 잔여 레버.

스크립트: `diag_leaderboard.py`(공식 success pass rate 전 모델+우리).
