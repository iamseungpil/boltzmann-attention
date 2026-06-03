# RUNG1 — source ablation 사다리 + gather-종료 + 거부편향: 설계서 (다음 세션 진입점)

> 2026-06-04. 작성=Opus(이번 세션). **다른 세션에서 ①리뷰 → ②구현 → ③실험.**
> 마스터=`EXPERIMENT_DESIGN.md §3`. 결과권위본=`reports/facet_rft_2026/SOPBENCH_EXPERIMENT_RESULTS.md`.
> 직전 결과=`Exp-4-rung1-v3-AB`(truncation 정정) + `Exp-4-rung1-v3ind`(inductive NULL).
> 선행근거=`RUNG1_V3_TREE_EVAL_LITREVIEW.md §8`(AND/OR 트리평가)·`SEARCH_INTERNALIZATION_LITREVIEW.md §9`(depth-recurrence)·`RUNG1_V3_INDUCTIVE_DESIGN.md`(직전 설계).

## 0. 한 줄
트리평가-*형식* 라인(단일식·inductive·depth-recurrence) 종료 = NULL. 전수조사 = 병목은 **serial-depth/조건수가 아니라 gather/결정 정책 + 모델이 구조를 *추론/emit*하게 둔 것**. → **무엇을 외부 제공(source)하고 무엇을 정책으로 고치면 BOTH가 오르나**를 ablation 사다리로 분리 측정.

## 1. 동기 — 무엇을 배웠나 (evidence chain, 전부 권위본 기록됨)
1. **Exp-4-rung1-T1T2**: BOTH 4/48. permitted 콜드붕괴 진단.
2. **Exp-4-rung1-v3-AB**: "v3 회귀"는 **planner max_tokens=24 truncation 아티팩트**(0/29 terminal 도달→루프). 무재학습 maxtok=1024 재시험 → **BOTH 2→5, control=5 = 무개선**. 단일-스텝 whole-expression 한계.
3. **Exp-4-rung1-v3ind** (inductive reduction 체인): **inductive가 행동을 무행동/과잉거부로 크게 이동**(over-refuse 25→**39**·acted 17→**8**·STOP 33→**21%** = 9–14 태스크 이동 = **크고 실재**). **BOTH는 무개선**(3 vs 단일식 4 vs baseline 5 = flat-to-down, n_T~46서 1–2태스크 차 ≈ eval 비결정성 **노이즈 범위, 정밀 주장 금지**). ★판정의 근거 = scalar BOTH 델타가 아니라 **행동 델타(거부↑·행동↓)와 메커니즘 census**(아래 §1.4).
4. **★전 궤적 전수조사 근본원인**:
   - **구조 fabrication**: rich derivation 타깃이 모델로 하여금 실제보다 큰 트리 환각(pay_bill 평균 10op/실제~3·set_account_info 9/2). **STOP chain 17/17이 fabricate된 `=false` leaf 포함** → `gate=false; STOP`.
   - **over-gather**: 자기 구성 트리를 채우려 step cap(10)까지 게더(median 10 vs 단일식 6). planner호출 94.5%가 gather, **~73 태스크가 terminal 없이 소진**.
   - **★조건수별 BOTH = 균일 바닥**(treeval `1c:0/3 2c:0/7 3c:1/9 4c:2/10 5c:1/9 6c:0/6` / treevalind `1c:0/3 2c:1/8 3c:1/10 4c:0/11 5c:1/9 6c:0/6`): **최단순 1-2조건도 0**. depth/fan-in 병목이면 저조건수서 높고 decay해야 함 → **serial-depth/조건수는 병목 아님.**
5. **선행연구 3종 종합**: ①grounded AND/OR 트리평가+전이=선행 무(novel) ②depth-recurrence는 결정론 트리평가에 이론적 정답(Xu&Sato)이나 pretrained 7B retrofit 불가(Huginn from-scratch) ③Searchformer식 trace 증류가 검증가능-도메인 교사초과 경로. **공통 함의: 형식/깊이가 아니라 구조-충실성·gather-정책이 병목.**

**결론**: 트리를 *모델이 추론/emit*하게 두는 한 fabrication 불가피. **트리-emit 타깃 폐기. 구조를 *제공*하고(source 사다리), gather-종료·거부편향을 정책으로 고친다.**

## 2. 가설
- **H1(구조제공)**: 태스크별 구조를 프롬프트에 제공(source=1)하면 fabrication 차단 → over-refuse↓·BOTH↑. (트리-emit 불요.)
- **H2(shape-만)**: 구조 *내용*은 숨기되 **조건수 budget + 종료조건**만 주면(중간 rung) fabrication(개수 부풀리기)·over-gather가 차단되어 BOTH 회복 — 7B가 *내용 추론*은 가능함을 시사.
- **H3(정책)**: gather-종료(T1c) + 거부편향 교정(DPO)이 source와 독립적으로 BOTH를 올린다.
- **반(反)가설(검증필요)**: source=1이 너무 쉬워 학습스킬이 trivial → **전이(ABox-swap)·도구변경 ablation으로 비-trivial 확인**(구조 제공해도 게더+집계+결정+의미매칭은 실제 일).

## 3. 실험 설계 — 2-axis 그리드 (source 외부화 × 결정 레버)
> ★**§11(2-agent)이 단일 헤드라인이고, 이 §3는 그 안의 *외부화-정도 축*이다**(리뷰 A4): rung A=단일-agent baseline / rung C=Agent2@oracle(천장) / rung B=중간 진단. **그리고 두 번째 축 = 결정 레버 {none, T1c, DPO}를 *직교*로 건다**(리뷰 A2) — 아래 이유로 결정-레버가 BOTH를 움직일 가장 유력한 변수라, source 효과와 분리하지 않으면 교란된다.

### 축1 — source 외부화 정도
| Rung | 프롬프트가 주는 것 | 모델이 하는 것 | 실패 모드(예상) |
|---|---|---|---|
| **A (=s3, 단일-agent baseline)** | NL 정책 + (alias) 도구설명 | 구조 추론 + 게더 + 결정 | ⚠️**리뷰 A3 정정: tree-emit OFF면 fabrication 아님 = T1T2 regime** — **게더는 양호(dirgraph~31/48)**, 실패 = **permitted 콜드붕괴 / over-refuse~30**(결정). (fabrication/over-gather는 tree-emit 타깃의 아티팩트였음, 폐기.) **fresh 재측정 필수.** |
| **B (s3+budget+term)** | A + **조건수 N + 깊이 + 종료규칙** | *내용* 추론(어떤 N개) + 게더 + 결정 | 내용 추론 오류(틀린 N개) |
| **C (=s1, =Agent2@oracle)** | **태스크별 구조(익명 dirgraph `op_7=>VERIFY op_3`) 제공** | 게더 + 집계 + 결정 (구조 추론 불요) | 구조 줘도 **결정(permitted) 콜드붕괴 잔존 가능** ← H1 핵심 미검증 |

### 축2 — 결정 레버 (★직교, 리뷰 A2)
- **none**: 기본 two-gate(`ready=true; preconds_verified; permitted; ACT/STOP`).
- **T1c (gather-종료 grounding)**: readiness=true는 "required 전부 게더"일 때만. ★**그리고 permitted을 *주어진 구조의 leaf-truth AND 룩업*으로 강등**(콜드 should_succeed 추측 아님). = source=1과 시너지: 구조가 주어지면 permitted=AND-over-given-structure는 *룩업*이 되어 콜드붕괴 소멸. **이게 H1이 결정에 닿는 메커니즘**(구조제공→permitted을 룩업으로).
- **DPO (거부편향 교정)**: should_T `permitted=false;STOP` dispreferred. prior 1.9× false 편향이 T1c 후에도 남으면 직접 누름.

**★왜 결정 레버를 직교로? (리뷰 A2, 받아들임 + 정제)**: 권위본 line 474 = "permitted 콜드붕괴는 v2부터 미해결 동일범인"(preconds AND는 항상 true, **단독 실패축=permitted=콜드 정책추측, false 편향**). **source=1은 *어떤 조건을 게더*할지(구조)는 주지만, *게더된 truth로 ACT/STOP*은 여전히 다운스트림 결정.** 구조만 주고 결정 레버가 없으면 → C가 여전히 over-refuse → "구조가 안 도움"으로 *오독* 위험. ∴ source(A/B/C) × lever(none/T1c/DPO)를 격자로 돌려 **source 효과와 결정 효과를 분리**.

**판정 그리드 (어디서 BOTH 회복?)**:
- **C+none에서 회복** → 구조 제공만으로 충분(permitted이 구조 보면 풀림).
- **C는 T1c/DPO 켜야 회복** → 진짜 레버는 *결정*(permitted), 구조는 보조 → 결정 레버가 헤드라인.
- **B에서 회복** → shape(조건수)+종료만으로 됨(내용 추론 가능, 가벼운 승리).
- **A+T1c/DPO만으로 회복** → 구조 제공조차 불요, 결정 레버가 전부.
- **C+T1c+DPO에서도 안 오름** → 집계/결정 자체 한계 → Searchformer식 trace 증류(§9).

## 4. 지표 (분리지표 필수, Mean Pass Rate 단독 금지)
- **헤드라인 BOTH(48)** = dirgraph∩action_called_correctly. + acted·goal·over-refuse(noact)·premature(acted∧¬dirgraph, 직접계산).
- **should_F(86) STOP-recall** (기준선 = nt 42%, single-step 33% 아님).
- **gather-depth**(should_T 스텝 median·max) — over-gather 모니터.
- **fabrication 지표**(RLLOG): terminal chain의 distinct-op 수 vs 실제 #leaf — rung A/B서 과대생성 여부.
- **terminal-reach%**(RLLOG ready=true 중 ACT/STOP 도달) — 비수렴 모니터.
- **★조건수별 BOTH 분해(2/4/6/8)** — decay 재확인(균일이면 여전히 정책 병목).
- (전이) **LODO held-out BOTH** + **도구변경 ablation**.

## 5. 구현 변경 (트리-emit 제거 + rung B 스캐폴드)
- `build_tbox_planner_sft.py`:
  - **트리-emit(`--treeval`/`--treeval_inductive`) 폐기**(기본 OFF; 코드 보존하되 헤드라인서 미사용).
  - **rung B 신규 플래그 `--shape_budget`**: 프롬프트에 goal의 `n_conditions`/`depth` + 종료규칙 렌더(구조 *내용*은 source=3대로 숨김). teacher 타깃에 `verified=k/N` 진행 표시 + N 도달 시 ready=true.
  - **T1c**(`--gather_complete_gate`): readiness=true 조건 = required-set 전부 executed(미게더면 ready=false;<tool>). 현 scratchpad 게이트 강화.
- `build_v2_prompt`(`two_stage_client.py`): source=1 경로 그대로(구조 렌더), rung B용 budget/termination 렌더 추가.
- 드라이버 `rung1_source_ladder.sh`(신규): 3 rung(A/B/C) × (T1c on/off) 학습+eval, maxtok=1024, 분리지표+조건수분해+fabrication+전이.

## 6. ★필수 선행 버그 수정 (실험 전 — 이 라인서 결과오염 버그 3번째)
> 직전 두 버그(truncation "회귀", nt-race)가 결과를 오염시킨 전력 → 버그수정은 blocking.
> **단 v3ind NULL은 이 버그에 강건**(리뷰 A8·권위본 정합): tool_choice 드롭은 **ACT 많이 한 arm(treeval)만** 타격 = **treeval 불리 confound**(드롭된 3개가 잠재 성공이면 treeval을 *과소*평가) → 그럼에도 treeval > treevalind → 결론 불변. **추가 안전장치: 버그수정 후 tree-emit arm 1개를 clean-harness서 1회 재실행**(NULL이 truncation처럼 또 버그-아티팩트 아님을 확인 후에만 tree-emit 라인 영구 종료).
1. **eval ACT-call `tool_choice`/tools 미스매치**(`two_stage_client.py:562-566` `_resolve` rung(b)): ACT 실행 호출이 `tools=[chosen_spec]`(단일)인데 `tool_choice={goal}` — chosen_spec이 goal과 불일치 시 400. action_name이 tools서 못 찾으면 `tools[0]`(=login_user 등 엉뚱)로 폴백(L545-547)이 원인. → goal tool을 tools에 포함하거나 폴백 제거·검증. 드롭 0 확인.
2. **헤드라인 python 레이스**: 드라이버가 eval(run_evaluation) 완료 전 헤드라인 계산 → nt=0 무효. `wait` 후 JSON freshness(eval > adapter) 가드 + evaluations 비어있으면 재대기/재계산.
3. **alias/source 직교 확인**: alias-ON + source=1 = 익명 dirgraph 제공(`op_7=>VERIFY op_3`) 정상 렌더 검증.

## 7. 현장 정합 / value prop (OISA CDP 분석, `/workspace_common/CDP/poc/docs/architecture/v8.0`)
- 현장 통증 = **도구 폭발 + 동음이의어(중첩 유사도구) + 다단계 조합**(OISA Patent1 §3: 45도구=82K토큰·동음이의어 58%).
- **source=1 = 배포 현실 + ★Agent1이 그 "구조 제공자"다(리뷰 A7, 모순 해소)**: OISA Score-Prune-Present도 pruned 구조를 프롬프트에 제시(=구조 제공). **단 OISA는 구조를 도메인-특화 파이프라인(코드 AST induce)으로 제공·도구변경 시 재구축**; **우리 Agent1(§11)이 그 구조를 NL서 *학습된·전이가능*하게 제공** = "source=1이되, 구조를 Agent1이 댄다". → 모순 아님: source=1=구조가 *제공됨*(by Agent1), Agent2는 그 위에서 실행. 차별 = 도구/도메인 변경 시 **리스트·NL만 갱신(재학습0)** vs OISA per-domain 재학습/재구축.
- **결정론 실행기 = 천장(오라클 37/48)이자 baseline**; LLM 우위 = 유지보수·불완전구조·NL·전이. 단순 실행정확도 경쟁 아님.
- **헤드라인 클레임** = (a) 중첩도구 disambiguation(alias) (b) **도구변경 robust 전이**(add/rename/remove → 리스트만, 재학습0) (c) ABox-swap 전이. 스케일(R1/R6 KV)은 OISA LKCA 영역, 우리 논문 코어 아님(분리).

## 8. 전이 (헤드라인) — source=1이 trivial 아님을 증명
- **LODO**: 6 도메인 학습 → held-out bank(새 온톨로지가 구조 제공) → 재학습0 → BOTH ≥ in-domain의 70%.
- **도구변경 ablation**: held-out서 도구 1개 추가/이름변경/제거 → 리스트만 갱신 → BOTH 유지 vs 결정론 워크플로 수작업.
- **분리증명**: 빈/틀린 구조 주입 → 붕괴(구조 제공이 no-op 아님 확인).

## 9. 선행연구 근거 (요약, 상세=litreview)
- **트리평가 학습 형식**: 중간단계 supervise(Kim&Suzuki)·globality(Abbe)·CoT serial(Feng) — *단, 우리 실측은 형식이 아니라 fabrication/gather가 병목임을 보임 → 형식 처방 보류*.
- **fan-in/depth decay**(Beam Tree arg-count) — **조건수 분해서 균일실패로 반증**(우리는 decay 아닌 균일 바닥).
- **depth-recurrence**(Xu&Sato 결정론 트리=loop유리·but Huginn retrofit불가) — **serial-depth가 병목 아님 확정으로 보류**.
- **검증가능-trace 증류**(Searchformer/ReST-MCTS*) — C에서도 안 오르면 다음 후보(검증기=결정론 evaluator).

## 10. 코드/파일 맵
| 파일 | 역할 |
|---|---|
| `build_tbox_planner_sft.py` | teacher (트리-emit OFF, `--shape_budget`·`--gather_complete_gate` 신규) |
| `two_stage_client.py` | `build_v2_prompt`(source 1/3·budget 렌더)·`SOPBENCH_PLAN_MAXTOK`(=1024) |
| `rung1_source_ladder.sh` | 신규 드라이버 (A/B/C × T1c, 분리지표+조건수+전이) |
| `rung1_v3_maxtok_retest.sh`·`rung1_v3ind_train_eval.sh` | 직전(참조·버그수정 대상) |
| `RUNG1_V3_INDUCTIVE_DESIGN.md` | 직전 설계(treeval_reduce) |

## 11. ★★단일 헤드라인 = 2-agent (구조추론 + 실행) = 단일 base + 2 LoRA
> (리뷰 A4) 이것이 **유일한 헤드라인**이고 §3는 그 안의 외부화-축이다(rung C = Agent2@oracle 천장, rung A = 단일-agent baseline, rung B = 중간). 전수조사 근본원인(한 agent가 구조추론+실행 동시)을 **구조적으로** 제거하는 분해.

**구성**:
| | Agent1 (구조추론/Parser) | Agent2 (실행, 현 planner) |
|---|---|---|
| 입력 | NL 정책 + (alias) 도구설명 (=s3) | NL 요청 + 도구 + **Agent1 dirgraph** (=s1) |
| 출력 | dirgraph (canonical 직렬화) | gather → ACT/STOP |
| 호출 빈도 | 태스크당 **1회** | 게더-until-resolved **루프** |
| **검증기** | **GT `task["constraints"]` 트리매치** | **결정론 evaluator**(dirgraph∩correct) |
| teacher | s3 프롬프트 → dirgraph 직렬화 | s1 프롬프트(구조 제공) → 게더/결정 |

**서빙 = vLLM 네이티브 멀티-LoRA(단일 base, adapter 2개)**:
```
vllm serve Qwen/Qwen2.5-7B-Instruct --enable-lora --max-lora-rank 16 --max-loras 2 \
  --lora-modules struct=<struct_adapter> exec=<exec_adapter> ...
```
오케스트레이터(two_stage_client): Stage1 `model=struct`(NL→dirgraph) → Stage2 `model=exec`(dirgraph 렌더 후 게더/결정, 루프). 같은 base 1회 로드, adapter ~20MB×2, 단계별 `model` 필드로 선택. 저렴.

**★인터페이스 계약**: **하나의 canonical dirgraph 직렬화**(=source=1 렌더 `op_7=>VERIFY op_3`)를 Agent1 타깃 ∧ Agent2 입력으로 공유. 포맷 불일치 = 파이프라인 깨짐 → 단일 포맷 함수로 강제.

**Agent1 = 학습된 구조 추론기 (★induce_ontology의 *런타임 포팅이 아님*, 리뷰 A5)**: `induce_ontology`는 도메인 **코드(AST)를 파싱**해 구조 도출 = Agent1의 검증 GT(`task["constraints"]`)가 그 코드서 유래. **Agent1은 NL 정책+도구설명만 봄(정보 strictly less)** → **코드-유래 구조를 NL만으로 복원 = 오프라인 inducer보다 *더 어려움***(동격 아님). 이것이 novelty이자 최대 리스크. → **새 도메인 = NL만 주면 Agent1이 dirgraph 생성 → Agent2 실행, 재학습0**(전이 강화)이되, Agent1 정확도가 천장.

**★Agent1 출력 robustness (리뷰 A6)**: Agent1은 학습된 7B → 추론 시 malformed/near-miss 직렬화를 emit함(계약 함수는 *타깃* 강제일 뿐 모델 출력 보장 못 함). 필수: **(i) parse/repair 스텝**(Agent1 출력→canonical dirgraph 파싱, 실패 시 보정/재시도) + **(ii) Agent2의 noisy-구조 robustness 측정**(틀린/부분 dirgraph 주입 시 Agent2 degradation 곡선). end-to-end ≈ A1×A2가 직렬화 노이즈로 더 깎이지 않게.

**upper-bound-first 프로토콜**:
1. **Agent2 천장**: oracle 구조(=GT dirgraph 직접 주입, s1) → Agent2@oracle BOTH. fabrication 제거 시 실행이 오라클(37/48)에 근접하나? = 가장 중요한 첫 측정.
2. **Agent1 측정**: NL→dirgraph 트리매치 정확도.
3. **파이프라인**: Agent1→Agent2 end-to-end. 천장과의 격차 = **구조추론 비용**(고립·정량화).

**필수 ablation**: **2-adapter(분리, 기본)** vs **1-adapter(멀티태스크, 모드 프롬프트)** — 분리가 fabrication을 실제로 격리하는지(2>1이면 분해 정당). + Agent2 단독 단일agent(s3) = fabrication baseline(이미 NULL).

**정직 경계**: ①오차복합(파이프라인≈A1×A2) — 단 어려운 부분(구조추론) 고립·측정이 장점. ②Agent1=NL서 코드-유래 구조 복원=오프라인 inducer보다 어려움(A5), 7B dirgraph 트리매치 실측 필요. ③서빙=단일base+2LoRA로 저렴. ④Agent1 malformed 출력→parse/repair+robustness(A6).

## 12. 다음 세션 첫 행동
1. **이 설계서 리뷰**(§11 2-agent=헤드라인·§3 2축 그리드·§5 구현·§6 버그·§4 지표). 적대검증.
2. **§6 버그 3종 수정** + 검증(ACT-call tool_choice 드롭0·헤드라인 레이스 해소) + **tree-emit arm 1회 clean 재실행**(NULL 재확인 후 라인 종료).
3. **★upper-bound 먼저(§11)**: **Agent2@oracle = rung C(s1, GT dirgraph 주입) × {none, T1c}** — 트리-emit OFF, T1c는 permitted을 *주어진 구조 leaf-truth AND 룩업*으로. **구조+grounded-결정 시 BOTH가 오라클(37/48)에 근접하나? = 전체 가설 1차 게이트.** (none vs T1c로 결정-레버 기여 분리.)
4. 천장 유의미 → **Agent1(NL→dirgraph) 학습**(canonical 직렬화 타깃·트리매치 검증·parse/repair) → **2-LoRA 파이프라인**(§11). Agent1 정확도·noisy-robustness 측정.
5. **§4 판정**(천장·결정레버 기여·Agent1정확도·파이프라인 격차) → 권위본 `Exp-4-rung1-2agent` 기록.
6. 회복 시 **전이(§8)** + **도구변경 ablation** + 2-vs-1 adapter ablation → 논문 헤드라인.
7. (보조 진단) rung A(s3)+T1c/DPO·rung B(조건수budget+종료) 격자로 "구조 없이 결정-레버만으로 어디까지" 병행.
