# 핸드오프 (2026-06-03) — v3 grounded 트리평가 A/B 실행 중 + 다음 세션 진입점

> 진입점. 마스터=`EXPERIMENT_DESIGN.md`(§3.10 북극성·§3 사다리). 결과권위본=`reports/facet_rft_2026/SOPBENCH_EXPERIMENT_RESULTS.md`.
> 근거 detail: `RUNG1_V3_TREE_EVAL_LITREVIEW.md`(트리평가 학습), `SEARCH_INTERNALIZATION_LITREVIEW.md`(탐색 내재화), `RUNG1_IMPL_HANDOFF_2026_06_02.md`(T1/T2).

## 0. 지금 무엇이 돌고 있나 (★다음 세션 첫 행동 = 이거 폴링)
**v3 grounded 트리평가 A/B**가 리모트 nohup으로 실행 중(harness 알림 없음 → **폴링 필요**).
- driver: `bash scripts/distill/sopbench/rung1_v3_train_eval.sh` (pid ~3533177), 시작 14:56 KST.
- **A/B**: GPU0=**control**(비-treeval, should_succeed 터미널) `qwen7b_tbox_alias_s3_nt_lodo_bank` / GPU1=**v3**(treeval, grounded AND/OR derivation) `qwen7b_tbox_alias_s3_treeval_lodo_bank`. 둘 다 key-fixed·alias_s3(헤드라인).
- **ETA ~19:00–19:20 KST** (학습 ~3.9h @ 51예제/분 × 11,940예제 + eval ~25분). ⚠️**step=예제인덱스(batch=1)** — epoch=3980step, 3ep≈4h (절대 "1시간" 아님).
- 결과 파일: `/home/woori/scratch/sft_alias_run/RUNG1_V3_AB_RESULTS.txt` (driver가 끝에 헤드라인 자동 기록).

## 1. 모니터링 명령 (복붙)
**진행 점검**:
```
& C:\workspace\.ssh_helper\rr.ps1 -TimeoutSec 80 -Command @'
OUT=/home/woori/scratch/sft_alias_run
date "+%H:%M:%S"; echo "driver=$(pgrep -fc "rung1_v3""_train_eval") train=$(pgrep -fc "lora_train""_chat_toolcall")"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
for r in nt treeval; do echo "--$r--"; tail -2 $OUT/train_alias_s3_${r}.log; done
tail -20 $OUT/RUNG1_V3_AB_RESULTS.txt
'@
```
**완료 판정**: `RUNG1_V3_AB_RESULTS.txt`에 "=== RUNG1 v3 A/B HEADLINE ===" 블록 + "DONE"이 찍히면 끝. (train_meta 2/2 후 eval→헤드라인.)
**freshness**: eval JSON(`eval_alias_s3_{nt,treeval}/bank/*.json`)이 adapter(`.../qwen7b_tbox_alias_s3_{nt,treeval}_lodo_bank/adapter_model.safetensors`)보다 **최신**이어야 유효(드라이버에 STALE 가드 있음).

## 2. 결과 해석 + 판정 (★헤드라인 = v3 vs control 분리지표)
드라이버가 자동 출력하는 지표(arm = nt=control / treeval=v3):
- **should_T(48)**: dirgraph·acted·goal·**BOTH** + **ACT-recall|게더**(=BOTH/dirgraph) + over-refuse(noact) + premature.
- **should_F(86)**: **STOP-recall**.
- baseline(T1/T2): BOTH=4, over-refuse~30, STOP~46.

**v3 가설 판정**:
- **v3 성공** = treeval가 control 대비 **BOTH↑ AND ACT-recall|게더↑**(과잉거부↓), should_F STOP 비회귀 → grounded-permitted derivation이 콜드붕괴 해소. → §3 분기 A.
- **v3 무효** = treeval ≈ control(BOTH 4±) → grounding이 안 먹힘(또는 13% tail이 가림). → §3 분기 B.
- ⚠️Mean Pass Rate 단독 판정 금지(거부 부풀림).

## 3. 결과 후 분기 (decision tree)
- **A. v3가 BOTH 올림** → 13% tail을 마저 ground해 더 밀기: **T1c(slot 수정)**(getter 올바른 인자 호출 → 의존-undefined·slot-아티팩트 해소) + **멀티턴 user_sim eval로 Point-2(ask/cannot-verify)** 학습. + 조건수별 분해로 깊이-decay 확인.
- **B. v3 ≈ control(무효)** → grounding이 충분히 안 됨이 원인일 수 있음 → **T1c 선결**(13% 중 slot분 제거) 후 v3 재측정 / 또는 **②DPO**(should_T→permitted=false;STOP dispreferred)로 우회.
- **C. 둘 다 BOTH 낮으면** → 북극성(§3.10) 경로로: **방법 A(cost-aware 탐색-trace 증류, Searchformer식)** 착수(검증기로 trace 생성→증류).

## 4. v3 설계 결정 (왜 이렇게 했나 — 재논의 방지)
- **decision = should_succeed(권위 GT)**, **derivation = grounded 트리평가**(per-leaf 기록 truth → AND/OR/chain 집계). 일치(87%)면 grounded emit, 불일치(13%)면 옛 permitted= **fallback**(모순 회피).
- **collision(같은-pred-다-args, transfer_funds) 체크완료 = 실재 안 함**(induced 트리가 중복-동일args). (pred,args) 키는 유지(원칙상 정확).
- **arg-only 주입 = no-op**(불일치 unknown은 arg-only가 아니라 **의존-undefined**(예 no_credit_card_balance, 카드 없으면 정의안됨) + **slot-아티팩트**).
- **13% fallback 정체 = T1c(slot) + 의존-undefined + 벤치결함 혼합** → A/B가 그 영향까지 통제.
- **Point-2(정보부족→행동 학습)**: 현 단일턴 벤치는 ask/cannot-verify를 보상 못 함 → **멀티턴 eval 필요**(분기 A에서).
- 코드: `build_tbox_planner_sft.py` `--treeval`(treeval_expr·est_failed·obs2·emit), 드라이버 `rung1_v3_train_eval.sh`(A/B).

## 5. 이번 세션(06-03) 종합 — 권위본에 기록됨
- **Exp-4-rung1-T1T2**: login-uniform(T1)+종료(T2) → BOTH 4/4, **근본원인=permitted 콜드붕괴**(2게이트 비-grounding, conjunction over #conditions). AND(preconds)는 항상 작동(0 false)=이전과 일관.
- **Exp-4-rung1-CAST**: conditional steering=**NULL**(BOTH 4→4, bias조차 안 움직임) → steering은 grounding/compute 못 함, derivation/xattn이 답.
- **litreview #1**(트리평가 학습가능: Kim&Suzuki·Abbe·Feng) + **#2**(탐색 내재화 교사초과: Searchformer 26.8%↓·TS-LLM depth-64).
- **설계 §3.10 북극성**(graph-guided 자율 agent 내재화) + **xattn-ABox+LoRA 공동학습(B5*)** 박제.
- **v3 teacher**(grounded 트리평가) 구현·검증·A/B 실행(진행 중).

## 6. ★인프라 교훈 (이번 세션 사고)
- **step=예제인덱스(batch=1)** → 학습시간 = n_train×ep / ~51예제분 (≈4h for 3980×3). "step=옵티마이저" 오독 금지.
- **vllm은 tau2_vllm_env에만** (seka_env엔 없음). gated steering 서버도 tau2 python. monkey-patch는 vllm 0.11.0서 정상.
- **SSH-drop이 git pull 중단 → 리모트 파일 옛 상태로 남음**(잘못된 드라이버 실행됨). 작업 전 **HEAD + 파일 마커 확인**(`git log -1`, `grep <marker> <file>`). 복구=`git reset --hard origin/facet-rft-2026`.
- **긴 잡 = 리모트 nohup**(foreground+rr.ps1 긴TimeoutSec는 paramiko 안정성 risk) → nohup 후 **폴링**(harness 알림 없음). rr.ps1 메시지당 1호출. pkill/pgrep 자기-self match는 문자열 split(`"a""b"`).
- **apply_two_stage_patch constants.py AssertionError**는 이미-패치 클론서 무해(eval 정상).

## 7. 코드/문서 레퍼런스
| 파일 | 역할 |
|---|---|
| `build_tbox_planner_sft.py` | teacher (T1/T2 + `--treeval` v3 + (pred,args) 키 + arg-only) |
| `rung1_v3_train_eval.sh` | v3 A/B 드라이버 (control vs treeval, 분리지표) |
| `rung1_train_eval.sh` | T1/T2 드라이버 (baseline) |
| `cast_extract_actvec.py`·`cast_sweep_eval.sh` | CAST probe (null) |
| `autoderive_getter_map.py` | getter_map 재생성(158/158 groundable) |
| `EXPERIMENT_DESIGN.md` §3.10 | 북극성 target architecture |
| `SOPBENCH_EXPERIMENT_RESULTS.md` | Exp-4-rung1-{T1T2,CAST} |

## 8. 다음 세션 첫 3행동
1. §1 명령으로 **A/B 진행/완료 폴링**. 완료면 §2로 헤드라인 읽기.
2. **v3 vs control 분리지표 판정**(§2) → 권위본에 `Exp-4-rung1-v3-AB` 행 기록(commit+push).
3. §3 분기로 다음 레버 결정(A=T1c+멀티턴 / B=T1c선결·DPO / C=탐색증류).
