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
3. **Exp-4-rung1-v3ind** (inductive reduction 체인): **BOTH 3 < 단일식 4 < baseline 5 = NULL, 더 나쁨**. 거부↑(25→39)·행동↓(17→8)·STOP↓(33→21%).
4. **★전 궤적 전수조사 근본원인**:
   - **구조 fabrication**: rich derivation 타깃이 모델로 하여금 실제보다 큰 트리 환각(pay_bill 평균 10op/실제~3·set_account_info 9/2). **STOP chain 17/17이 fabricate된 `=false` leaf 포함** → `gate=false; STOP`.
   - **over-gather**: 자기 구성 트리를 채우려 step cap(10)까지 게더(median 10 vs 단일식 6). planner호출 94.5%가 gather, **~73 태스크가 terminal 없이 소진**.
   - **★조건수별 BOTH = 균일 바닥**(treeval `1c:0/3 2c:0/7 3c:1/9 4c:2/10 5c:1/9` / treevalind 유사): **최단순 1조건도 0**. depth/fan-in 병목이면 저조건수서 높고 decay해야 함 → **serial-depth/조건수는 병목 아님.**
5. **선행연구 3종 종합**: ①grounded AND/OR 트리평가+전이=선행 무(novel) ②depth-recurrence는 결정론 트리평가에 이론적 정답(Xu&Sato)이나 pretrained 7B retrofit 불가(Huginn from-scratch) ③Searchformer식 trace 증류가 검증가능-도메인 교사초과 경로. **공통 함의: 형식/깊이가 아니라 구조-충실성·gather-정책이 병목.**

**결론**: 트리를 *모델이 추론/emit*하게 두는 한 fabrication 불가피. **트리-emit 타깃 폐기. 구조를 *제공*하고(source 사다리), gather-종료·거부편향을 정책으로 고친다.**

## 2. 가설
- **H1(구조제공)**: 태스크별 구조를 프롬프트에 제공(source=1)하면 fabrication 차단 → over-refuse↓·BOTH↑. (트리-emit 불요.)
- **H2(shape-만)**: 구조 *내용*은 숨기되 **조건수 budget + 종료조건**만 주면(중간 rung) fabrication(개수 부풀리기)·over-gather가 차단되어 BOTH 회복 — 7B가 *내용 추론*은 가능함을 시사.
- **H3(정책)**: gather-종료(T1c) + 거부편향 교정(DPO)이 source와 독립적으로 BOTH를 올린다.
- **반(反)가설(검증필요)**: source=1이 너무 쉬워 학습스킬이 trivial → **전이(ABox-swap)·도구변경 ablation으로 비-trivial 확인**(구조 제공해도 게더+집계+결정+의미매칭은 실제 일).

## 3. 실험 설계 — ablation 사다리 (헤드라인)
**핵심 질문**: "7B가 BOTH를 내려면 *무엇을 외부 제공*해야 하나?" = Opus의 self-termination/faithfulness 중 무엇을 외부 보철?

| Rung | 프롬프트가 주는 것 | 모델이 하는 것 | teacher 타깃 | 외부화 정도 |
|---|---|---|---|---|
| **A (=s3, 현 baseline)** | NL 정책 + (alias) 도구설명 | 구조 추론 + 게더 + 결정 | gather→ready→ACT/STOP (트리-emit 폐기) | 없음 (fabrication+over-gather 발생) |
| **B (s3+budget+term)** | A + **조건수 N + 깊이 + 종료규칙**("N개 검증되면 멈추고 결정") | *내용* 추론(어떤 N개) + 게더 + 결정 | gather(≤N)→`verified k/N`→ready→ACT/STOP | shape만 |
| **C (=s1)** | **태스크별 구조(익명 dirgraph `op_7=>VERIFY op_3`) 제공** | 게더 + 집계 + 결정 (구조 추론 불요) | 동일(구조는 프롬프트서 읽음) | 내용까지 |

- **공통**: alias=ON(이름 마스킹·의미매칭, value prop 필수), source 외 조건 고정(alias_s, ep3, r16, LODO holdout=bank, SOLO).
- **+T1c (gather-종료 정책, 모든 rung 적용 가능)**: per-step readiness가 "필요 조건 전부 게더됨"일 때만 ready=true; 미게더면 다음 게더. over-gather/premature 동시 차단. (rung B의 종료규칙이 이것의 명시 버전.)
- **+DPO (보조)**: should_T에서 `permitted=false;STOP`(거부) dispreferred. rung 결과 본 후 적용.

**판정 (어느 rung서 BOTH 회복?)**:
- **B에서 회복** → 7B는 *shape 바운드+종료*만 외부화하면 됨(내용 추론 가능) = 가볍고 강한 결과.
- **C에서만 회복** → *내용*(어떤 조건)까지 줘야 함 = 7B 구조추론 불가, source=1이 정답(현장 정합).
- **C에서도 안 오름** → 병목이 구조 아닌 *집계/결정* 자체 → T1c/DPO 또는 Searchformer식 trace 증류로.

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

## 6. ★필수 선행 버그 수정 (실험 전)
1. **eval ACT-call `tool_choice`/tools 미스매치**: 두 스테이지 ACT 실행 호출이 tools=[일부]인데 tool_choice=goal → 400 BadRequest, ACT 태스크 일부 드롭(treeval n_T 48→45). ACT-call에 goal tool을 tools에 포함하거나 tool_choice=auto+강제 검증. → 드롭 0 확인.
2. **헤드라인 python 레이스**: 드라이버가 nt eval(run_evaluation) 완료 전 헤드라인 계산 → nt=0 무효. `wait` 후 JSON freshness(eval > adapter) 가드 + evaluations 비어있으면 재대기.
3. **alias/source 직교 확인**: alias-ON + source=1 = 익명 dirgraph 제공(`op_7=>VERIFY op_3`) 정상 렌더 검증.

## 7. 현장 정합 / value prop (OISA CDP 분석, `/workspace_common/CDP/poc/docs/architecture/v8.0`)
- 현장 통증 = **도구 폭발 + 동음이의어(중첩 유사도구) + 다단계 조합**(OISA Patent1 §3: 45도구=82K토큰·동음이의어 58%).
- **source=1 = 배포 현실**: OISA Score-Prune-Present도 pruned 구조를 프롬프트에 제시(=구조 제공). 우리 차별 = **도메인-비특화 원칙 → 도구변경 시 리스트만 갱신(재학습0)** vs OISA per-domain adapter 재학습.
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

## 12. ★타깃 아키텍처 — 2-agent (구조추론 + 실행) = 단일 base + 2 LoRA
전수조사 근본원인(한 agent가 구조추론+실행 동시 → fabrication)을 **구조적으로** 제거하는 분해. source 사다리를 포섭한다(rung C = Agent2 천장, rung A/B = 단일 agent로는 불가 baseline).

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

**Agent1 = 학습된 온톨로지 inducer**: `induce_ontology`가 오프라인에 하는 걸 NL서 런타임·새도메인 일반화. → **새 도메인 = NL 정책만 주면 Agent1이 dirgraph 생성 → Agent2 실행, 재학습0**(전이 강화).

**upper-bound-first 프로토콜**:
1. **Agent2 천장**: oracle 구조(=GT dirgraph 직접 주입, s1) → Agent2@oracle BOTH. fabrication 제거 시 실행이 오라클(37/48)에 근접하나? = 가장 중요한 첫 측정.
2. **Agent1 측정**: NL→dirgraph 트리매치 정확도.
3. **파이프라인**: Agent1→Agent2 end-to-end. 천장과의 격차 = **구조추론 비용**(고립·정량화).

**필수 ablation**: **2-adapter(분리, 기본)** vs **1-adapter(멀티태스크, 모드 프롬프트)** — 분리가 fabrication을 실제로 격리하는지(2>1이면 분해 정당). + Agent2 단독 단일agent(s3) = fabrication baseline(이미 NULL).

**정직 경계**: ①오차복합(파이프라인≈A1×A2) — 단 어려운 부분(구조추론) 고립·측정이 장점. ②Agent1 학습성=supervised라 tractable(무감독 발견 아님), 단 7B의 dirgraph 트리매치 실측 필요. ③서빙=단일base+2LoRA로 저렴.

## 13. 다음 세션 첫 행동
1. **이 설계서 리뷰**(특히 §3 사다리·§12 2-agent·§5 구현·§6 버그·§4 지표). 적대검증.
2. **§6 버그 3종 수정** + 검증(ACT-call tool_choice 드롭0, nt 헤드라인 레이스 해소).
3. **upper-bound 먼저(§12-1)**: oracle 구조(s1) 주입 → **Agent2 천장 BOTH** 측정 = 트리-emit OFF·단순 게더/결정 타깃. (실행이 깨끗한 구조서 학습되는지 = 전체 가설의 1차 게이트.)
4. 천장 유의미하면 → **Agent1(NL→dirgraph) 학습**(canonical 직렬화 타깃, 트리매치 검증) → **2-LoRA 파이프라인**(§12 서빙).
5. **§4 판정**(천장·Agent1정확도·파이프라인 격차) → 권위본 `Exp-4-rung1-2agent` 기록.
6. 회복되면 **전이(§8)** + **도구변경 ablation** + 2-vs-1 adapter ablation → 논문 헤드라인.
7. (보조) source 사다리 rung B(조건수budget+종료)는 "단일agent를 구조제공 없이 얼마나 구할 수 있나"의 진단으로 병행 가능.
