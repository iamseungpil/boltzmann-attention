# Cross-Domain Transfer 설계서 — A축 scaffold 도메인 전이 (로드맵 1단계)

> **상태: DRAFT — 리뷰 대기. 리뷰 통과 후 구현·실행.** 마스터 = `EXPERIMENT_DESIGN.md`.
> **로드맵 위치**: ① **cross-domain(본 문서) = A축 전이 입증** → ② should_F(거부축) = A축 논문 완성 → ③ B축 weight 내재화.
> 메타규칙: 강한 주장 reliable test 후 박제 · GPU 전 zero-cost 사전점검 · 공식 success(리더보드 지표)로만 보고 · scaffold 도메인-하드코딩 금지.

---

## §0. 핵심 질문 + 클레임
- **질문**: bank에서 설계·튜닝한 A축 scaffold(결정론 게이트 + fix들)가 **다른 도메인에서도 ABox-swap만으로 재학습 0**으로 동작하는가?
- **클레임(검증 대상)**: "도메인-일반 SOP-verifier(도메인 NL정책/도구 온톨로지에서 action-graph 재구성·precondition 결정 offload) + 소형 7B가, **per-domain 튜닝/재학습 없이** N 도메인서 SOP를 따른다. 가중치 아니라 온톨로지만 swap."
- **이게 성립하면 A축 단독으로 systems/agent 논문 가치**(LLM-Modulo의 cross-domain SOP-following 일반화). 붕괴하면 scaffold가 bank-overfit → B축(학습)만 일반성 구제.

## §1. 무엇이 고정 / 무엇이 swap (재학습 0의 정의)
- **고정 (재학습 0, 도메인 무관)**: ① SFT LoRA 어댑터 `qwen7b_tbox_t1c_lodo_bank`(현행) ② scaffold **코드 전체**(flag 로직: OFFLOAD/ACTIVE/DGGATE/ARGFIX/VALFIX/KEEPTUPLE/LOGINFIRST/LOGINCALL/STOPSUCCESS) — **per-domain 분기 추가 금지**.
- **swap (도메인별 입력)**: ABox = `induced/ontology_<domain>.json`(전 7도메인 이미 존재) + 벤치의 도메인 규칙(`domain_assistant_keys[domain]`: constraint_links/processes/default_dep — 전 7도메인 존재) + `getter_map[domain]`(전 7도메인 domain-keyed 존재) + task set(`data/<domain>_tasks.json`).
- ⇒ **인프라 사전 확인됨(2026-06-05)**: 7도메인 전부 ontology·도메인규칙·getter_map·task 존재 → 새 authoring 0(induce 파이프라인이 이미 추출). 전이의 "사람 작업"도 낮음(ABox 자동추출분 사용).

## §2. 두 테스트 (전이 강도별)
### T-A (primary, 재학습 0) — scaffold 도메인-일반성
- 현 `lodo_bank` 어댑터 + bank-설계 scaffold(전 flag)를 **다른 6 도메인**(dmv·healthcare·hotel·library·online_market·university)에 ABox-swap, **재학습 0** 실행.
- **분리 논리**: 이 어댑터는 LODO(bank held-out)=**6 non-bank로 학습** → 6 도메인은 어댑터-gather엔 *in-domain*. 그러나 **scaffold fix들은 bank failure만 보고 설계**됨 → 6 도메인은 scaffold엔 *cross-domain(미관찰)*. ⇒ T-A는 **"scaffold(결정론 코드)가 도메인 일반인가"**를 격리 측정(A축 핵심 클레임). 어댑터-gather 일반성은 별도(기존 LODO 주장).
- 비교: 각 도메인 **base Qwen2.5-7B vs 우리 stack vs 리더보드**(공식 success).

### T-B (strong, 1~2회 LODO 재학습) — full held-out 전이
- 타깃 도메인 D(예: online_market) **held-out**으로 LODO 어댑터 재학습 → bank-설계 scaffold 적용 → D 평가. **D가 어댑터·scaffold 양쪽서 held-out** = 가장 강한 전이 주장.
- 비용: D당 1회 학습(~4h). T-A 성공 후 1~2 도메인만.

> 1단계 = **T-A(재학습 0) 먼저**. T-B는 T-A 양성 시 확증용.

## §3. 지표 (공식 success only)
- **공식 `success` pass rate %**, 도메인별 전체 task(should_T+should_F), tool_full — 리더보드와 **동일 기준**(`LEADERBOARD_METRIC_GROUNDING_2026_06_05.md`). **BOTH(dg∧acc) 헤드라인 금지.**
- 보고 3열/도메인: **base-7B / 우리 stack / 리더보드(해당 도메인 open-source 및 max)**. should_T·should_F 분해 병기.
- 누적 지표 = end-to-end full-stack(per-fix delta 합산 금지, Fix-3 리뷰 교훈).

## §4. ★zero-cost 사전점검 (GPU 전 필수 — 도메인-readiness audit)
각 타깃 도메인에 대해 **GPU 런 전** 다음을 offline 확인(`diag_xdomain_audit.py`). 하나라도 실패하면 해당 도메인은 scaffold가 그대로는 안 도는 것 → 정직 범위로 분리/표기(per-domain 하드코딩 금지).
1. **DGGATE Guard-2 재구성 == evaluator** (도메인별 `dfsgather_invfunccalldirgraph(constraints_original,...,opt=full)` vs `task["directed_action_graph"]`, **OVER=0 ∧ UNDER=0**). = DGGATE가 도메인 일반인지 단위검증(bank서 PASS였던 그 검사를 6도메인 확장).
2. **getter_map[domain] 커버리지**: VALFIX/조건 leaf의 getter route 존재율. 없으면 VALFIX no_evidence_route 과잉-deny 가능 → 표기.
3. **LOGINFIRST/LOGINCALL 적용성**: 도메인 login_user의 **credential arg 이름**이 bank의 `identification`과 같은가? (다르면 LOGINFIRST 하드코딩 키 일반화 실패 → arg를 ontology서 끌게 일반화 or 도메인 표기). hotel=login 無→no-op(정상).
4. **ontology↔도구 정합**: `ontology_<domain>.json`의 op/predicate가 벤치 도구셋과 매칭(induce 품질). alias 매핑 깨짐 없는지.
- **판정**: audit 통과 도메인 = T-A 대상. 부분 실패 도메인 = 원인 표기 후 포함(전이의 정직 범위 = "어디서 그대로 되고 어디서 안 되나"가 결과의 일부).

## §5. BLOCKING 가드 (사전등록)
1. **B-1 scaffold 무변경**: 전 도메인 **동일 flag·동일 코드**. per-domain `if domain==` 분기 **금지**. (LOGINFIRST credential-arg 같은 bank-리터럴이 발견되면 → ontology서 끌도록 *일반화*하되 도메인-분기 아님; §4.3.)
2. **B-2 공식 success only**: 모든 수치 공식 success(134-eq, tool_full). BOTH 보고 금지.
3. **B-3 base 통제**: 각 도메인 base-7B(무 scaffold, 동일 tool_full) 동시 측정 = Δ의 분모. "stack이 base보다"가 1급 비교(리더보드는 2급 참조, scaffold caveat).
4. **B-4 정직 범위 분리**: ① LOGINCALL quirk-의존분(login-call 없이도 도메인별 재측정 = quirk 기여 격리) ② audit 부분실패 도메인 표기 ③ 어댑터 in-domain(6) vs T-B held-out 구분 명시.
5. **B-5 회귀**: bank 공식 success(50.75%, S1)는 동일 stack 재실행서 불변(±noise) 확인(파이프 무결성).

## §6. 사전등록 성공기준
- **1급(A축 전이 성립)**: 우리 stack이 **≥4/6 도메인서 base-7B를 유의하게 상회**(공식 success) ∧ audit-통과 도메인서 **per-domain 튜닝 0**으로 작동.
- **2급(경쟁력)**: 상회 도메인서 **open-source 리더보드 중앙값 이상**(가능하면 SOTA 근접). bank 50.75%가 7B로 open-source 70B(42.54%) 추월한 패턴이 ≥1 타 도메인서 재현.
- **실패/부분**: <4/6 상회 or audit 대량실패 → "scaffold는 bank-tuned, 일반성 제한" 정직 보고 → B축 의존도↑.

## §7. 정직 범위 / threats (논문 명시)
- **scaffold 기여 vs 모델**: stack 수치는 "7B+SFT+결정론 scaffold"이지 raw 7B 아님. base 통제(B-3)로 분리.
- **LOGINCALL = evaluator quirk**(login을 call-order로 카운트): 도메인별 quirk-기여 격리(B-4①). quirk 없는 도메인(login 無 hotel 등)서 결과가 핵심.
- **"재학습 0 ≠ 사람작업 0"**: ABox는 induce 자동추출분 사용(7도메인 존재) → 사람작업 낮음을 근거로 제시. 단 induce 품질(§4.4) 검증 병기.
- **T-A 어댑터 in-domain**: 6 도메인은 어댑터-gather엔 in-domain → T-A는 *scaffold* 전이 격리; full 전이는 T-B(held-out 재학습).

## §8. 실행 계획
- 드라이버 `xdomain_eval.sh`(augment OFF, full stack incl DGGATE+LOGINFIRST+LOGINCALL+STOPSUCCESS): 한 vllm 서버(어댑터 1개)로 **도메인 루프**(sim+eval), 각 도메인 fresh OUT/OFFLOG, `--domain <d>`. + 각 도메인 **base-7B 대조 런**(scaffold flag off, tool_full).
- 분석 `diag_leaderboard.py` 확장(도메인 인자) → base/stack/리더보드 3열 표.
- T-A 6도메인 → 결과 → (양성 시) T-B 1~2 도메인 LODO 재학습.

## §9. 리뷰 체크리스트
- [ ] §4 audit(특히 Guard-2 재구성 OVER0/UNDER0)를 GPU 전 전 도메인 실행했는가?
- [ ] scaffold 무변경(B-1) — LOGINFIRST credential-arg 등 bank 리터럴을 ontology-derived로 일반화(도메인 분기 아님)?
- [ ] 공식 success only(B-2)·base 통제(B-3)·LOGINCALL quirk 격리(B-4①)?
- [ ] T-A(scaffold 격리) vs T-B(full held-out) 구분 명시?
- [ ] 성공기준(≥4/6 상회) 사전등록 고정?

## §10. 산출물 (리뷰 통과 후)
1. `diag_xdomain_audit.py`: §4 도메인-readiness(Guard-2 재구성·getter_map·login-arg·ontology 정합).
2. scaffold 일반화 패치(필요시): LOGINFIRST credential-arg를 ontology/도구 시그니처서 derive(도메인 분기 없이).
3. `xdomain_eval.sh`: 도메인 루프 stack + base 대조.
4. 결과 → 본 문서 §11 + 마스터 §2 + `LEADERBOARD_METRIC_GROUNDING` 도메인 확장.
