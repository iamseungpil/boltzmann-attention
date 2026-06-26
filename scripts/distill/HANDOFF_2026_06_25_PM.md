# HANDOFF 2026-06-25 PM — autofetch arm: 하네스버그→수정→결정론 재측정 + 방법론 전환(pass^1 폐기→결정론 행동지표) + [[05]]present수정 + [[08]]hook

> **진입 = `06-NOW` + 이 handoff + `EPISTEMIC_A2_THESIS_2026_06_23`(§2 translator·§3 SOAR·delta-ⓓ).** 직전 = `HANDOFF_2026_06_25`(AM). ★새 규율 [[08]](결론 전 전수포렌식)·hook 강제됨.

## 0. 오늘 PM 서사 (한 단락)
SOAR/relwork 딥리서치(23/25 검증·§10.3) — 2602.05073(tau2서 LLM 자기confidence≈random=§0 backbone)·2604.19459(Case177 탐지불가 오역=§3 silent-residual frontier 실증·단 unfaithful은 소수·multi-hop 미입증으로 *정정*)·TRUST(소형>대형 공동선점자·NL2GenSym은 그 청구 *killed*)·NL2CA(Qwen0.6B NL→ACT-R=A2_FRONTEND 선행). NL2GenSym=오프라인 생성+결정론 추론(per-task 아님)=A2_FRONTEND 자리. → **autofetch arm(make-or-break 전환검증)**: deny-방식 첫 run=**88→14 대붕괴** → *전수 포렌식*: **stateful deny가 tau2 reward-replay 깸**(environment.py set_state가 mutating write 재실행·기록된 G6-deny≠재실행결과=infra_error 283/342)=*측정 무효*(모델 결과 아님). → **수정: deny→읽기-증강**(candidate_summary를 get_user_details 응답에 append·읽기는 replay서 skip→안전) → 재run **infra_error 0·유효**. → **★방법론 전환**: pass^1은 user-sim 편차(~0.11)로 *노이즈 지배*(전환/회귀 대부분 mixed flip)→"14B 도움/scale의존" *반증* → **결정론 order-pick 지표(user-sim 무관)**: present가 order-pick *robust 개선*(+0.063 32B>+0.024 14B·over-action↓)·ⓑ 71/72/74 개선·**101/102=order 고침·⋈주소 못고침**(pass=0 유지 이유 확정). 잔여=operator(L0·최대)·over-action(OVER)·item/variant·⋈로 *흩어짐*. → 학습없는 수정 **present+g15(precondition)** 진행 중. + [[05]] present 하드코딩 수정(generic). + [[08]] hook 추가.

## 1. ★최상위 결론
- **autofetch arm 첫 결과는 무효였음**(하네스버그) → 수정 후 유효. **교훈: 유효write를 deny하면 reward-replay 깨짐**(무효write deny=기존 게이트는 안전). **replay-safe = 읽기-증강(deny 아님).**
- **★방법론 전환(이 세션 최대 산출)**: **pass^1 점추정은 user-sim 노이즈 지배 → 폐기.** 반드시 **결정론 행동지표**(order/operator-correct·over-action·gold⊆agent) + pass^k. [[06]] "pass^1 무효"의 *구체 실천*. ([[08]] hook으로 강제.)
- **present 효과(결정론·정직)**: ⓑ formalize/select를 **진짜·robust 개선**(order-pick +0.06/+0.02·over-action↓·32B가 *더*)·**단 작음**·pass 전환은 *다른 층 잔여*(operator/over-action/operand/⋈)가 막음. **부분-GO**(present는 보조·silver bullet 아님).
- **101/102 ⋈ 정체 확정**: present가 (a)어느주문=고침·(b)다른주문 주소가져오기=못고침(cross-entity 조인). 후보리스트는 *선택*만 돕고 *값전송*은 모델 날조("123 Elm St").
- **잔여(order 고친 뒤)** = operator(24/16·최대)·over-action(14/12)·item(14/11)·address⋈(14/14)·variant(7/7)로 흩어짐 → operator/over-action=결정론게이트(g5)·operand=grounding·⋈=조인·원천=학습.

## 2. ★다음 세션 (우선순위)
1. **present+g15 결정론 포렌식**(실행 중·회수): precondition이 operator/over-action 잔여를 닫나 — **결정론 지표(operator-correct·over-action↓)**로 floor/g15/present/present+g15 누적. **+ [[06]] "G5=0"을 *결정론으로* 첫 재검**(pass^1 노이즈였을 수). 회수=`reexp_g15_*.log`·마커 `REEXP_G15_*_DONE`·`presentg15_retail_t3`.
2. **operand grounding(item/variant) 학습없이**: present를 `get_order_details`(items)·`get_product_details`(variants) 읽기에 확장(replay-safe 동일패턴·grounding.json candidate/anchor_source 있음). 코드 소량.
3. **⋈ 주소**: present에 *전체주소* 포함(지금 일부) or 값-autofetch. 부분 시도 가능.
4. **★진짜 make-or-break = faithful-formalize *학습***(잔여 원천): A2-규칙사용 SFT(벤치·도메인일반·gold 채점)→tau2 A2-swap. 딥리서치 방법(IDK/defer·다른세션 회수) 필요. + **A2_FRONTEND**(NL→A2 생성기·NL2CA[Qwen0.6B]로 해금·`A2_FRONTEND_DISTILL_DESIGN`).

## 3. 자산 (commit·branch facet-rft-2026)
- **코드**(`scripts/distill/tau2/`): `gate_interpreter.py`(candidate_summary=**generic·[[05]]수정됨**·select_confirm kind) · `t2_gate_patch.py`(T2_PRESENT_READS read-augment) · `a2/retail.gate.json`(G6 select_confirm·present_label) · 드라이버 `reexp_present.sh`·`reexp_present_g15.sh`. 진단 `escape_scope_diag.py`·`escape_layer_decomp.py`·`escape_arm2_probe{,_v2}.py`.
- **문서**: `ESCAPE_SCOPE_STAGE1_CATALOG`(Arm-II Probe-B 7/7·정정)·`..LAYERS_AUG`(층→레버·§9 SOAR·§6.5 트리거)·`AUTOFETCH_SIGMA_ARM_DESIGN`(§0.5 [[05]]가드)·`RELWORK_AND_DIRECTION §10.3`(2025-26 신규·delta방어).
- **데이터**(`data/simulations/`): `*_presentread_retail_t3`(유효·infra0)·`*_g14present_*`·`*_presentg15_*`(진행). baseline floor/g14/g15 t3.
- **hook**(`/c/workspace/.claude/`): `scaffold_guard.py`+`scaffold_rules.json`=**[[05]]+[[08]] 강제**(forensic_* 추가). 메모리 [[08]] 신설·[[41]] §SOAR갱신.

## 4. ★불변·함정
- **★pass^1 노이즈 지배(user-sim ~0.11·flip 절반)** → **결정론 행동지표(order/operator-correct·census)+pass^k.** 큰 음성/양성=아티팩트 의심. **[[08]] hook이 결론doc/metric+결론 시 강제.** ([[06]] 정합·실천.)
- **★present(scaffold)=고정 엔진·A2만 도메인특화**([[05]]): candidate_summary가 retail 필드 하드코딩했다 *수정됨*(generic dump). scaffold 작성 시 *매번* "예쁘게 포맷"이 도메인구조 새는지 grep.
- **replay-safety**: *유효* write deny 금지(reward-replay 깸=infra_error). 읽기-증강만 안전.
- **present=절차offload([[05]] Q3=yes)→측정으로만 정당화**(default 아님)·비교 시 *대형에도 같은 present*(불공정 금지)·모델 기여 측정.
- [[11]] tau2 학습0·전이=A2-swap · [[03]] 집계로 갈아엎기 금지 · [[30]] 리모트 `cd /c/workspace`후 ssh_run·CRLF→LF·시크릿금지·user-sim=gpt-4.1 COST GUARD·throttle 가능.
