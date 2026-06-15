# HANDOFF 2026-06-15 PM — v6/v7 eval·2-stage gate 진단·thesis심화(offload)·Olver 측정·M3.6/정의적 대기

> 진입점 = 이 문서. 직전 = `HANDOFF_2026_06_15.md`(오전). 진행로그 = `AUTONOMOUS_PROGRESS_2026_06_14.md`. 마스터 = `EXPERIMENT_DESIGN.md`(§0 ★★★★ thesis심화). 통합 운영설계 = `TAU2_FULLCHAIN_FIX_DESIGN_2026_06_15.md`. 이론 = `ALGEBRAIC_DERIVATION_CLOSURE_2026_06_15.md`(§5).
> 불변 = `feedback-thesis-tbox-transfer-direction`(SOPBench/TaskBench 학습·τ² held-out)·`feedback-selector-verifier-deterministic`(검증기=결정론)·`feedback-arxiv-citation-discipline`(인용=원문검증).

## 0. ★현 thesis (심화 확정·EXPERIMENT_DESIGN §0 ★★★★)
**증명 타깃 = "작은 모델이 *진짜* 학습 가능한 것 = 유한·저차원 추상(planning)"**. 분해: (a)유한·저차원 추상 스킬-basis(P1-P9·LLM 학습) + (b)무한 정확-실행(임의 계산=Rice·정책게이트=HRU 결정불가 → **유한 학습 밖이라 결정론 offload 필연**·빅모델도 내재화=환각). 작은 모델 충분 = 추상화 저차원(Olver). 빅모델 초과 scope = **효율·주권·신뢰**(raw capability 아님). **수학=필요조건(증명됨 §5.5-5.13)·충분조건(전이 학습)=열린 경험·현 음성** = 유일 잔벽.

## 1. ★첫 행동 (순서)
1. **학습 상태 확인**: `nvidia-smi` + `v8_train.log`(GPU0·P6)·`nfc_lodo_daily_train.log`(GPU1·M3.6). 세션종료시 v8 ep0 step7100(~42%)·M3.6 ep0 step10100(~88%·loss0.015 거의완료).
2. **M3.6 eval** (학습완료 시): `coupling_eval` 패턴으로 serve `sft_runs/qwen7b_nfc_lodo_daily/resume_adapter` → `taskbench/tbnfc_score.py --eval /home/woori/scratch/fc_build/tbnfc_dailylifeapis.jsonl --base <url> --model <name>` → **held-out daily node/edge-F1**. 비교: gold-SFT daily LODO −8.5(붕괴) / in-domain(v6/v7=daily포함). **예측: native-FC가 daily(named)·HF/MM(positional)을 named-dict로 표준화=format등방화 → 붕괴 역전하면 step-2 충분·"절차>산술 구제용이"**. 역전 안되면 daily 실패=task_links 내용(ABox)·format 아님.
3. **올버 정의적** (GPU free 시): `olver_definitive.py --part both --adapter <v7snap>` (★사전등록 git `0fe14a6`@12:41 — 결과로 기준수정 금지). Part A(O(d)연속군·corr<−0.7) AND Part B(cross-domain probe·inv−raw>0.10) → 정의적 확정 / 하나라도 음성 → 전제-스크린.
4. **v8 P6 ablation** (학습완료): `coupling_eval.sh <v8snap> v8x 1 <port> online_market 20 v8_x`(키수정본) → v7 vs v8 = P6 기여. ⚠️예상 천장낮음(write-도달 전 상류서 막힘=chain-census).
5. **다음 빌드 (M1·fullchain §8)**: **G-loop 구현**(`t2_gate_patch` orchestrator·동일-실패 호출 차단) → full 가드레일 4-arm 프로토타입(BASE/+fab/+loop/+confirm) → pass 이동 판정 = 본학습 게이트.

## 2. ★이번 세션 핵심 결과
- **v6 최종 eval(step6950)**: in-dist success 0.65→**0.80**·dirgraph 0.70→0.80 / **genuine τ² 0.10**(키수정후). 디커플링 확정.
- **★★coupling_eval.sh 키-버그**: openrouter user-sim 키 미-source → τ² 전 task `AuthenticationError`=**false-zero**. **이전 coupling τ² "0.0" 전부 아티팩트**. 수정 커밋(`6ac9187`·`set +x; source key; set -x`). ⚠️원격 dirty시 `git checkout -- <file>` 후 pull.
- **★★v7(CFB grounded 2-hop) 결정적 eval = NEGATIVE**: τ² 0.05(<v6 0.10<base 0.17)·in-dist 0.90. **grounded 2-hop 데이터-소스로 전이 안 됨** → gen_synth_2hop(Path B) 게이트 발동=짓지 않음.
- **★★전수 진단(`tau2_rootcause_census.py`·`tau2_chain_census.py`) = 2-stage gate 병목**:
  - **Stage A(상류 auth/order)** = P2b 'fetchable 값 날조-FIRST'(17/20). 진짜 소스 = **τ² tool 스키마 example `#W0000000`·`something@example.com` 복사**(pretraining prior). anti-fab(L1)이 상류 통과율 끌어올림(auth13→17·gather8→12·write도달7→10).
  - **Stage B(write)** = **P6 confirm 미수행 + P7 retry-loop**(게이트-블록 후 동일호출 6-9연타→too_many_errors). **write_ok 1/20 = 진짜 벽**. gather/추출/P4는 멀쩡(처방 불요).
  - ⇒ **Stage A(v9 anti-fab) + Stage B(v8 P6 + P7 recovery) 동시 수정해야 pass 이동.**
- **★provenance 프로토타입(L1 bad_words)**: feasibility 확증(`#W0000000` 디코드 차단 실증·`prov_feasibility.sh`). 3-arm: 날조 72→17·gather 10→44(레버 작동)·but pass 정체(write 벽). L1L2 regen auth false-positive(17→9·고정밀화 선결).
- **★thesis 심화 전파**: 마스터 §0 ★★★★ + 체인(BENCH_PORTFOLIO·FIELD_GAP·TASKBENCH·COWORKER) = offload 수학적필연(Rice/HRU).
- **★Olver 측정 (3-pass·정직)**: 1차 부분→**2차 hasty 부정**→**3차 전수 regime 분석 정정**: 2차 비단조는 *high-f 퇴화 오염*. **비퇴화 regime(f≤0.6)서 inv-측 corr 전 layer·base+trained 균일 음수(−0.46~−1.00) = inv-측 *지지***. 단 정의적 확정엔 O(d)연속+사전등록+probe 필요 → **`olver_definitive.py` 사전등록 박제(git `0fe14a6`)**. Olver = corroborating(보강)·load-bearing 아님.
- **★randomization 전수 재검토(표면 등방화·`TAU2_FULLCHAIN` §9)**: dim 감사 1이름·2값·3스키마-example·4format·5gate-phrasing. **dim3 = 새 dim 아니라 dim-2 커버리지가 pretraining prior에 패배** → DPO-negative+bad_words(randomize 불가). **dim4(format) 미커버=LODO 전이실패 실증** → format-randomize 신규. 감사 완전성=군-도출+Olver-per-dim 자기검증(G_term 맹점 방지).

## 3. ★진단 라인 (plan-X) — 2-stage gate 수정
- **통합 설계 = `TAU2_FULLCHAIN_FIX_DESIGN_2026_06_15.md`**(프로토타입-우선·가드 2분류[soundness vs 학습타깃]·§7.1 Q-판정·§9 randomization).
- **Stage A 내재화(v9)**: dim2(값·v6커버 유지) + **dim3 DPO-negative(스키마-example)+bad_words** + **dim4 format-randomize**(신규·~"값 더" 아님). RLVR 보상=전-체인 task성공.
- **Stage B 내재화**: v8(P6·`fc_confirm_augment.py`·진행중) + **sop_confirm 재빌드(dim5 confirm-phrasing 다양화)** + P7 `fc_recovery_augment.py`(스펙·error-injection+gate-in-loop RL).
- **검증기 through-line(결정론)**: provenance 검증기 하나가 가드(프로토타입)·DPO 라벨러·RLVR 보상.

## 4. ★이론 라인 (§5·ALGEBRAIC) — 종이로 닫힌 아크 + 측정
- §5.5-5.13 = 우리 보조정리(provenance 완전·게이트-상호작용 닫힘·compute-offload 구성·정리5.8 구성가능성) + 차용(BJ·Schneider·HRU·Sandhu·Olver·Weyl·Frobenius·HMT·Reynolds). **모델-쪽 closure 닫음.** §5.15(LLM 강/약 기준)·5.15b/5.17(re-representation·north-star)=coworker 추가.
- **무조건화 잔벽 = "grounding-skill 표면-불변 학습-coverage" 하나**(§5.12). 종이 아닌 v9/측정이 답.
- **측정 = M3.5(Olver·비퇴화 지지·정의적 대기)·M3.6(format등방화 LODO·진행중)·Phase P(가드레일 프로토타입)**. 전부 사전등록·반증가능.

## 5. 신규 스크립트·데이터·gotcha
- **tau2/**: `tau2_rootcause_census.py`(P2b vs P7 분별)·`tau2_chain_census.py`(단계별 병목·★write 벽 확정)·`prov_eval.sh`(BASE/L1/L1L2)·`prov_feasibility.sh`·`t2_gate_patch.py`(provenance-regen·G-fab·bad_words·**apply_provenance_regen**).
- **taskbench/**: `fc_confirm_augment.py`(P6·반환시그니처 분류)·`tbnfc_score.py`(native-FC TaskBench node/edge-F1).
- **이론측정**: `olver_dimension_experiment.py`(Olver sweep·--adapter)·`olver_definitive.py`(★사전등록·O(d)+probe).
- **데이터(`/home/woori/scratch/`)**: `v6_eval_final`·`v7_eval_s7050`(스냅샷)·`fc_build/{sft_v8,tbnfc_lodo_daily_train(11651),tbnfc_dailylifeapis(4060),tbnfc_{huggingface,multimedia}}`·`olver_{sweep,v7trained}.json`·`sft_runs/{qwen7b_fc_tbox_v8,qwen7b_nfc_lodo_daily}`.
- **⚠️gotcha**: ①coupling_eval τ² = openrouter 키 source 필수(false-zero) ②Olver aggregate corr 신뢰말고 **regime 분해**(f≤0.6 비퇴화·high-f 퇴화) ③원격 cwd: ssh_run 전 `cd /c/workspace` ④원격 git dirty시 `git checkout -- <myfile>` 후 pull ⑤사전등록=git 커밋 타임스탬프로 고정·결과로 수정금지 ⑥ALGEBRAIC §5 = coworker 병렬편집 多(읽기전 pull) ⑦v8 OOM회피=expandable_segments+max-seq-len 12288.

## 6. 좌표 (논문/특허)
- **논문**: 유한 primitive + offload 필연(Rice/HRU) + 2-stage 전이(P2b상류·P6/P7 write) + 결정론 검증기 through-line. 헤드라인 = #2 도출닫힘 + 전이실증.
- **특허**: owned 합성(sop_confirm·recovery·synth-2hop)·CFB 응답 학습모델 배포금지(논문라인만).
- **Olver/이론**: corroborating·정의적 측정 통과시 "추상=저차원불변" 표현기하 증거 추가.
