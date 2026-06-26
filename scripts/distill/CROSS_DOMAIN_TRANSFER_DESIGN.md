> ⚠️ **방법 SUPERSEDED by plan X** — scaffold-bridge cross-domain 전이는 native-FC + ABox-swap(plan X)이 대체. **단 Exp-5 결과(77.3% LODO 전이)는 여전히 헤드라인 증거**로 유효. 전이 주장은 cross-domain→cross-bench로 격상.

# Cross-Domain Transfer 설계서 — A축 scaffold 도메인 전이 (로드맵 1단계)
> 📌 **구조 안내**: 모든 설계·실험 문서의 단일 마스터 = repo `scripts/distill/EXPERIMENT_DESIGN.md` (**§7 문서지도**에서 각 문서의 역할·상태 확인; 목표·순서 변경은 마스터 §0-§4에서만). 처음 읽는다면 마스터부터.

> **상태: ★결과 진행 중 (2026-06-06) — §11 transfer 확정 채워지는 중.** audit(§4.5) PASS·login-arg 일반화 완료 → train1 6 held-out 확정(avg 77.3%, LB-max 추월 3/6) + LODO 3 확정(healthcare 95.9%·library 75.8% 추월); LODO 4 + train1 6 학습 큐 진행. **결과 = §11**(권위본 `../../reports/facet_rft_2026/SOPBENCH_EXPERIMENT_RESULTS.md` Exp-5). 진입점 = `HANDOFF_2026_06_06_xdomain_full.md`. 마스터 = `EXPERIMENT_DESIGN.md`. (coworker plan = `../../reports/facet_rft_2026/COWORKER_EXPERIMENT_PLAN.md`.)
> 〔리뷰 반영 이력: BLOCKING-1(통합 stack 정합·login 3중 compose-audit §5 B-5) + BLOCKING-2(adapter-only 4열 scaffold Δ 격리 §6) + S1(VALFIX oracle-정당성 §4.2) + S2(login-arg unit-verify §4.3) + S3(바=리더보드-상대 §6) — APPROVED w/ changes.〕
> **로드맵 위치**: ① **cross-domain(본 문서) = A축 전이 입증** → ② should_F(거부축) = A축 논문 완성 → ③ B축 weight 내재화.
> 메타규칙: 강한 주장 reliable test 후 박제 · GPU 전 zero-cost 사전점검 · 공식 success(리더보드 지표)로만 보고 · scaffold 도메인-하드코딩 금지.

---

## §0. 핵심 질문 + 클레임
- **질문**: bank에서 설계·튜닝한 A축 scaffold(결정론 게이트 + fix들)가 **다른 도메인에서도 ABox-swap만으로 재학습 0**으로 동작하는가?
- **클레임(검증 대상)**: "도메인-일반 SOP-verifier(도메인 NL정책/도구 온톨로지에서 action-graph 재구성·precondition 결정 offload) + 소형 7B가, **per-domain 튜닝/재학습 없이** N 도메인서 SOP를 따른다. 가중치 아니라 온톨로지만 swap."
- **이게 성립하면 A축 단독으로 systems/agent 논문 가치**(LLM-Modulo의 cross-domain SOP-following 일반화). 붕괴하면 scaffold가 bank-overfit → B축(학습)만 일반성 구제.

## §1. 무엇이 고정 / 무엇이 swap (재학습 0의 정의)
- **고정 (재학습 0, 도메인 무관)**: ① SFT LoRA 어댑터 `qwen7b_tbox_t1c_lodo_bank`(현행) ② scaffold **코드 전체**(flag 로직: OFFLOAD/ACTIVE/DGGATE/ARGFIX/VALFIX/KEEPTUPLE/LOGINFIRST/STOPSUCCESS) — **per-domain 분기 추가 금지**.
  - **★LOGINCALL 드롭 (2026-06-06)**: LOGINCALL은 cred-absent에 dummy-login으로 quirk(failed-login-but-passed)를 악용 → 우리 should_T 8건이 quirk였고 리더보드는 안 씀(should_T quirk≈0)=불공정. **cross-domain 스택서 LOGINCALL 제외**(honest). cred-absent should_T는 정직하게 실패. honest 44.78%도 오픈소스 70B 추월이라 quirk 불요. (login은 ①②의 일반 prereq-driving + arg-binding으로 cred-present만 정당 처리 — 사용자 통찰대로 특별취급 제거.)
- **swap (도메인별 입력)**: ABox = `induced/ontology_<domain>.json`(전 7도메인 이미 존재) + 벤치의 도메인 규칙(`domain_assistant_keys[domain]`: constraint_links/processes/default_dep — 전 7도메인 존재) + `getter_map[domain]`(전 7도메인 domain-keyed 존재) + task set(`data/<domain>_tasks.json`).
- ⇒ **인프라 사전 확인됨(2026-06-05)**: 7도메인 전부 ontology·도메인규칙·getter_map·task 존재 → 새 authoring 0(induce 파이프라인이 이미 추출). 전이의 "사람 작업"도 낮음(ABox 자동추출분 사용).

## §2.0 ★★타당성 교정 (2026-06-06, 사용자 지적) — held-out 도메인에서만 transfer 측정
현 어댑터 `lodo_bank`는 **6 non-bank로 학습**(bank held-out). ⇒ **6 non-bank를 테스트하면 in-domain(어댑터가 본 도메인)=transfer 아님.** (초기 T-A가 6 학습도메인을 테스트한 것은 무효 — library 77%는 in-domain 수치, 과대.) **transfer는 held-out 도메인에서만**:
- **유일 valid 기존 점 = bank**(어댑터 미관찰, held-out): bank stack honest 43.28% = 진짜 transfer 1점.
- **6 학습도메인 런 = "scaffold 도메인-전반 기능 + login 일반화 작동" in-domain 체크로 강등**(transfer 헤드라인 아님; in-domain *상한* 참조로만 유용).
- **추가 transfer 점 = held-out 어댑터 재학습 (둘 다 진행, 2026-06-06)**:
  - **(1) LODO-per-target** (thesis-정합, 다중도메인 혼합 유지): 타깃 X held-out, 나머지 6 학습, X 테스트. X=library·healthcare (+bank 기존) → `xdomain_train.sh t1c_lodo_<X>`.
  - **(2) train-1-test-6** (사용자 원안, 극저자원 transfer): 1 도메인 학습, 나머지 6 held-out 테스트. train=bank → `t1c_train1_bank`. ⚠️ thesis 설계는 다중도메인 혼합으로 *공통 절차스킬* 추출 → train-1은 학습다양성 부족으로 under-training과 transfer-실패 혼동 가능(해석 시 주의). LODO와 같은 test 도메인(library/healthcare)서 비교하면 다양성 효과 분리.

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
2. **getter_map[domain] + ★VALFIX oracle-정당성 (리뷰 S1)**: 커버리지뿐 아니라 — VALFIX는 "getter 없음 ⟺ value-restriction(외부상태 미참조)"이라는 inducer 계약에 의존. 타 도메인서 getter_map이 **상태-읽는 조건의 getter를 놓치면** VALFIX가 DB를 직접 compute = **oracle 누출**. ⇒ per-domain으로 "no-getter 조건이 **진짜 value-restriction(상태 미참조, params만으로 결정)**인지" 검증(누출 도메인은 VALFIX off 또는 표기). 단순 커버리지율 아님.
3. **LOGINFIRST/LOGINCALL 적용성 + ★login-arg unit-verify (리뷰 S2)**: 도메인 login_user의 **credential arg 이름**(bank=`identification`)을 ontology/도구 시그니처서 derive하도록 일반화하되, **derive된 arg == 그 도메인 실제 login 시그니처인지 Guard-2식 단위검증**(틀리면 LOGINFIRST가 조용히 over-deny). 일반화 패치에 검증 동반. hotel=login 無→no-op(정상).
4. **ontology↔도구 정합**: `ontology_<domain>.json`의 op/predicate가 벤치 도구셋과 매칭(induce 품질). alias 매핑 깨짐 없는지.
- **판정**: audit 통과 도메인 = T-A 대상. 부분 실패 도메인 = 원인 표기 후 포함(전이의 정직 범위 = "어디서 그대로 되고 어디서 안 되나"가 결과의 일부).

## §4.5 ★audit 1차 결과 (2026-06-06, `diag_xdomain_audit.py`, zero-cost) — cross-domain 게이트 통과 (조건부)
| 도메인 | n | OVER | UNDER | login_arg | 판정 |
|---|---|---|---|---|---|
| bank·dmv·healthcare·hotel | 134/97/124/195 | **0** | **0** | identification (hotel=NO-LOGIN) | ✓ DGGATE 재구성=evaluator 정확일치 |
| library·online_market·university | 66/172/42 | =n | =n | **password** | ✗ **login-arg만** 불일치 |
- **불일치 원인 = 유일, login_user credential arg 이름**: 재구성=`login_user(identification,username)`(bank-ism) vs evaluator=`login_user(password,username)`. **다른 모든 노드 정확일치**(OVER=UNDER=태스크당 정확히 1 = login 노드뿐).
- ⇒ **DGGATE dirgraph 구조 재구성은 도메인-일반 확증**(4/7 exact; 3/7은 login-arg 한 곳). **수정 = login credential param을 `action_parameters[login_user]−{username}`서 derive**(도메인 분기 아님=B-1 OK; 리뷰 S2 정확 적중). 수정 후 7/7 OVER=0 재검증 = 구현 1차.
- getter_map: 7도메인 전부 존재(14~39 entries). VALFIX oracle-정당성(§4.2/S1)은 도메인별 manual 후속.

## §5. BLOCKING 가드 (사전등록)
1. **B-1 scaffold 무변경**: 전 도메인 **동일 flag·동일 코드**. per-domain `if domain==` 분기 **금지**. (LOGINFIRST credential-arg 같은 bank-리터럴이 발견되면 → ontology서 끌도록 *일반화*하되 도메인-분기 아님; §4.3.)
2. **B-2 공식 success only**: 모든 수치 공식 success(134-eq, tool_full). BOTH 보고 금지.
3. **B-3 base 통제**: 각 도메인 base-7B(무 scaffold, 동일 tool_full) 동시 측정 = Δ의 분모. "stack이 base보다"가 1급 비교(리더보드는 2급 참조, scaffold caveat).
4. **B-4 정직 범위 분리**: ① LOGINCALL quirk-의존분(login-call 없이도 도메인별 재측정 = quirk 기여 격리) ② audit 부분실패 도메인 표기 ③ 어댑터 in-domain(6) vs T-B held-out 구분 명시.
5. **B-5 통합 stack 정합 + 회귀 (cross-domain 前 BLOCKING, 리뷰 BLOCKING-1)**: bank 50.75%(S1)가 **어느 flag 집합**인지 못박는다 — **S1 = 전 fix 통합**(`ARGFIX VALFIX KEEPTUPLE DGGATE LOGINFIRST LOGINCALL STOPSUCCESS`, `offload_stopsuccess.sh` COMMON+STOPSUCCESS, augment OFF) = **통합 stack은 이미 bank서 함께 검증됨(회귀 0)**. ⚠️**login 3중 처리 compose-audit**: DGGATE(login을 dirgraph prereq로 establishing-구동)·LOGINFIRST(login front-load)·LOGINCALL(login을 call-order로 카운트)가 login_user를 각자 다룸 → **`_active_driven` 가드가 동일 도구 이중구동 차단**(설계상 1회). cross-domain 前 **bank 1회 재실행으로 ① S1=50.75% 재현(±noise) ② login_user 호출수=태스크당 ≤1(이중구동 없음) ③ 내 DGGATE-era BOTH 29/34 = 공식 success 몇 %인지 동시기록**(BOTH↔success 정합). 이게 통과해야 cross-domain.
  - **★이 bank 런 = LOGINCALL-OFF + login-일반화 통합 stack** (2026-06-06): honest 44.78%는 LOGINCALL-ON re-score(잠정) → 이 런이 **확정 헤드라인**(official=honest, quirk 0) + cross-domain T-A의 bank 기준선. S1은 이미 통합 full-stack 확인됨(COMMON에 ARGFIX/VALFIX/KEEPTUPLE/DGGATE 포함)이라, 이 런은 거기서 LOGINCALL만 끄고 login을 일반 prereq-driving(DGGATE deepest-first)+arg-binding으로 돌린 것.

## §6. 측정 = 4열 + 사전등록 성공기준
**★4열 (리뷰 BLOCKING-2 — scaffold Δ 격리, 도메인당, 공식 success):**
```
base(raw 7B)  →  adapter-only(SFT, scaffold OFF)  →  stack(SFT + 전 scaffold flag)  →  리더보드
              └ 어댑터-gather 기여(6도메인 in-domain)   └ ★scaffold cross-domain 기여 = A축 클레임 = stack − adapter-only
```
- **adapter-only** = 어댑터 서빙 + **offload/게이트 flag 전부 OFF**(모델이 직접 도구콜; two-stage offload 없음). = 어댑터 raw 행동. 비용 = 도메인당 eval 1회 추가(무재학습).
- **scaffold Δ = stack − adapter-only**가 T-A의 진짜 측정치. "stack > base"만으론 어댑터 in-domain 기여와 섞여 scaffold 전이를 격리 못 함.
- **★1급 성공기준 (리뷰 S3 — 리더보드-상대로 격상)**: ① **scaffold Δ(stack−adapter-only) > 0 유의, ≥4/6 도메인** (scaffold가 cross-domain 기여) ∧ ② **stack이 해당 도메인 open-source 리더보드 중앙값 이상, ≥3/6** (가능하면 bank의 "7B-stack > open-source 70B" 패턴 재현 ≥1) ∧ ③ per-domain 튜닝 0.
- **2급**: base→adapter-only Δ(어댑터 LODO-gather 전이, 기존 주장 재확인).
- **실패/부분**: scaffold Δ≈0(다수) or audit 대량실패 → "scaffold bank-tuned, 일반성 제한" 정직 보고 → B축 의존도↑.

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
- [x] **BLOCKING-1**: 통합 stack(전 fix) bank 정합·회귀·login 3중 compose를 cross-domain 前 재실행(§5 B-5).
- [x] **BLOCKING-2**: base가 아니라 **adapter-only 4열**로 scaffold Δ 격리(§6).
- [x] S1 VALFIX oracle-정당성·S2 login-arg unit-verify를 audit에(§4.2/4.3). S3 바=리더보드-상대(§6).
- [ ] §4 audit(Guard-2 재구성 OVER0/UNDER0 6도메인)를 GPU 전 실행 = **첫 게이트**.
- [ ] scaffold 무변경(B-1) — bank 리터럴(login-arg 등)을 ontology-derived 일반화(도메인 분기 아님)?
- [ ] T-A(scaffold 격리, adapter-only 대조) vs T-B(full held-out) 구분?

## §10. 산출물 (리뷰 통과 후)
1. `diag_xdomain_audit.py`: §4 도메인-readiness(Guard-2 재구성·getter_map·login-arg·ontology 정합).
2. scaffold 일반화 패치(필요시): LOGINFIRST credential-arg를 ontology/도구 시그니처서 derive(도메인 분기 없이).
3. `xdomain_eval.sh`: 도메인 루프 stack + base 대조.
4. 결과 → 본 문서 §11 + 마스터 §2 + `LEADERBOARD_METRIC_GROUNDING` 도메인 확장.

## §11. ★결과 (2026-06-06) — held-out transfer 확정 (재학습 0·honest·quirk≈0)
> 권위본 = `../../reports/facet_rft_2026/SOPBENCH_EXPERIMENT_RESULTS.md` **Exp-5**. 집계 = `sopbench/diag_heldout_summary.py`(LODO-7 + train1 7×6 매트릭스). 지표 = 공식 success(`evaluator.py:277`, tool_full)·honest(LOGINCALL off). transfer = **held-out only**(어댑터 미관찰 도메인).

### §11.1 train-1 (단일 도메인 학습 → 6 held-out, 극저자원) ★확정
`qwen7b_tbox_t1c_train1_bank` = **bank 한 도메인만** 학습(혼합 아님) → scaffold 그대로 + ABox-swap, 6 held-out STACK:

| held-out | STACK success | should_T | LB-max(any) | vs LB |
|---|---|---|---|---|
| dmv | 71.1% (69/97) | 35/36 | 86.7 | below |
| healthcare | 64.5% (80/124) | **44/44** | 92.7 | below |
| hotel | **83.6% (163/195)** | 58/67 | 69.7 | **추월** |
| library | **71.4% (40/56)** | 14/21 | 66.7 | **추월** |
| online_market | 73.8% (127/172) | 53/59 | 89.5 | below |
| university | **97.6% (41/42)** | 6/6 | 95.2 | **추월** |
| **avg** | **77.3%** | | | **3/6 리더보드-MAX 추월** |

- **헤드라인**: 단일 도메인(bank)만 학습 → 재학습 0·ABox-swap으로 6 안 본 도메인 평균 77.3%, 프런티어(GPT-5/o4-mini) 추월 3/6. base 7B 0~21%.
- should_T 거의 천장(44/44·35/36·6/6) → 낮은 도메인은 should_F-bound(bank 결론 일치). honest(LOGINCALL off).

### §11.2 LODO-per-target (다도메인 혼합, 타깃 held-out) 〔진행 중〕
`qwen7b_tbox_t1c_lodo_<X>` = X 제외 6학습 → X 평가. 확정 3: **bank 43.3%(58/134) · healthcare 95.9%(118/123, sT 44/44, >LB 92.7) · library 75.8%(50/66, >LB 66.7)**. dmv·hotel·online_market·university = 학습 큐(GPU 2병렬, `xdomain_train_queue.sh`) → `xdomain_eval_heldout.sh`(stack+adapter-only 4열) 대기.
- **scaffold Δ**: adapter-only ~0% → stack 75~95%(어댑터 안 본 도메인) = §6 BLOCKING-2 4열로 격리(LODO 행, eval 대기).

### §11.3 핵심 관찰
1. **A축 클레임 1차 입증**: per-domain 분기 0 scaffold가 held-out서 재학습 0 작동 — train1·LODO 양쪽서 다도메인 리더보드 추월.
2. **학습 다양성 효과(혼합 vs 단일)**: healthcare LODO 95.9% ≫ train1 64.5%(+31pp) = 혼합이 should_F 전이 강화. 단 hotel·library·university는 train1만으로도 LB-max 추월 → 다양성 효과 도메인-의존.
3. **정직 범위**: ① train1은 STACK만(adapter-only 대조 미측정); LODO 행에서 scaffold Δ 격리. ② 전 도메인 should_T 강·should_F 약 = bank 결론 도메인-일반 재현 → **로드맵 #2 should_F가 전이서도 전체% 레버**.

### §11.4 다음
학습 큐 완료 → `xdomain_eval_heldout.sh` 1회(LODO 4 held-out + train1 6×6) → `diag_heldout_summary.py` 전체 매트릭스 → ① LODO 추월 다도메인 재현 ② train-diversity(혼합 vs 단일 7×6) 정량 → 로드맵 #2 should_F.
