# HANDOFF 2026-06-13 — day-4 종료 (D1채택/D2기각·v3신기록·선별기 1급양성·A2 front-end 라인 가동)
> 📌 **구조 안내**: 모든 설계·실험 문서의 단일 마스터 = repo `scripts/distill/EXPERIMENT_DESIGN.md` (**§7 문서지도**; 목표·순서 변경은 마스터 §0-§4에서만). 처음 읽는다면 마스터부터.

> **다음 세션 진입점.** 결과 권위 = `reports/facet_rft_2026/TASKBENCH_EXPERIMENT_RESULTS.md`(TB) **§8.9/8.9b/8.10/8.10b + Day-4 한눈표(§8.10 머리)** · τ²/A2 = `scripts/distill/BENCH_PORTFOLIO_FRAMEWORK_DESIGN.md` **§3.6-3.9** · A2 학습 = `scripts/distill/A2_FRONTEND_DISTILL_DESIGN.md`. 리모트 규칙 = memory `reference-remote-server-environment` (ssh stdin 파이프·cd /c/workspace 먼저).

## 0. ★첫 행동 (순서대로) — [2026-06-12 저녁 세션이 1·2·3 상당 부분 완료, §0b 참조]
1. ~~N3 수확~~ ✅완료 → **PORTFOLIO §3.7b** (사전등록 conj FAIL이나 기준① 오캘리브레이션 판명 — 헤드라인 = 게이트 pass^1 무비용화: matched 0.1853 vs nogate 0.1830·write 44→0·deny 한계피해 +7.9pp→−0.7pp 소거. 복구 *행동*은 r1에서도 96% — 메시지는 복구 *품질*을 바꿈 4/41→9/36).
2. ~~P-A2-0b 로컬 + S0 스모크~~ ✅완료 → **A2_FRONTEND §6** (7B 0.333 > 14B 0.167 = 1-shot 과앵커링·둘 다 frontier 미달 / S0: holdout gate_recall 0.564→**1.000**·airline 전이 0.333→0.167 = 분포갭 실측·S1 정당화·structure-EM 지표 필요).
3. **데이터 생성 계속**: ✅batch5b(15)+batch6(30, sampler v2 어휘+게이트7) 전부 QC 무손실 → **dataset v5 = 180쌍**. 잔여 = **batch7 +20 → 200 목표** (seed 7 → 6스타일 로테이션 → join_qc → v6 병합).
4. **딥 리서치 2건**: ✅완료·검수·커밋 (`reports/facet_rft_2026/research_selector_lit_2026_06_12.md` 31검증인용 · `research_framework_metrics_2026_06_12.md` 26검증인용). **다음 = 이 둘을 합류시켜 ①선별기 설계서(detail, 마스터 §7 경유 — 1순위 레버: Smoothie-prior 가중 prop-MBR(0원)→soft-approval votes→7B reverse-likelihood; novelty=상관-소스 보정·게이트 역선택 미보고) ②§1.6 v2 동결(F3=τ pass^k n=4 확정·F4=0/N+rule-of-three·구조적0/표본적0 분리·ⓟ1=paired bootstrap DiD·F6=AURC/E-AURC·F7=cost-of-pass)**. ⚠️드리프트: τ² 리더보드 user-sim gpt-5.2 권장(우리 gpt-4.1) — 외부비교 시 4-tuple 명시(PORTFOLIO §3.7b 박제). 잔여: τ-bench pass^k 원문 PDF 눈검증 1회 권고.
5. **coworker**: P4(A2 크기 census, 32B/72B/235B) — trackb_raw/p4_a2_census 도착 감시 (06-12 저녁 기준 미도착).
6. **잔여 GPU 큐**: ~~S0-v2~~ ✅완료(200쌍: held-in **structEM 70%·EM 40% 창발**·airline 전이 추가악화 0.528 → **S1 실-도메인 verified distill이 다음 rung**, A2_FRONTEND §6) · v4 재채굴 사이클 검토 · 선별기 합성(v3+guided 57.90 위에 N2-식 이종풀+선별) · P-D0 diffusion(조건부 강등 유지). GPU 양쪽 해제 상태.

### 0a0. ★★야간 0613→14 수확 완료 + 전수 부검 (2026-06-13 아침 — 최신)
- **τ² gate r3 (G4 deny-게이트) = PASS**: 위반 0/0/0/0 집행(G4 deny 65)·pass^1 0.1952(r2 동등+)·**pass^2-4 회복**(pass^4 0.018→0.054)·compliant FULL +5.3pp. PORTFOLIO §3.7c.
- **★SEL-4 신기록**: dpo2g-풀+7B Reviewer = **공식 0.6803**(SEL-1 0.6722 → +0.81pp, 사전등록 적중) = k0 대비 **+10.3pp**. best-stack=SEL-1+SEL-4. TB §8.9d/SEL-4 행.
- **음성 2건 + S1 기각**: 풀 확장(ND 0.6703)·v3g 풀(NC 0.660) 둘 다 기각 / S1 스모크 0.528 변화0(n_gates 5→1).
- **★전수 궤적 부검 3건 (사용자 발주 — PORTFOLIO §3.7d)**: ①**pass^4 정체 = user-sim 분산**(fail 종결 user_stop 압도, 게이트 무관 — ⓟ1 측정 불가·base 능력 라인 이관) ②**r3 성공 = transfer 차단의 net 양성**(transfer 실행 63→0, 원래 11% 성공 경로라 차단=이득; 단 DOWN 20 = G4 deny가 transfer-필요 태스크 죽임 → **G4 offload 전환이 데이터로 정당화**) ③**v3g 음성 = oracle 동일(0.896)·다양성 −33%**(0.024→0.016, DPO가 후보 동질화 → 선별 headroom 붕괴 = "선별=다양성 함수" 확정).
- **다음 1순위(부검 도출)**: ①G4 deny→**offload 전환**(transfer-필요 태스크 구제, PORTFOLIO §3.8 큐 승격) ②선별 풀 **다양성 명시 증대**(이종 temp/base) + SEL-4 상시화 ③F3 user-sim seed 분산 보고 ④P5 도착 → S1-v2(규모).

### 0a. ★심야 추가분 (2026-06-12 심야 — 다음 세션 1순위 갱신)
- ✅**§1.6 v2 동결**(마스터 — F3 τ pass^k n=4·F4 0/N+rule-of-three+구조적/표본적0 분리·ⓟ3 게이트-관할 census 신설·4-tuple 인용 규율) + ✅**`SELECTOR_DESIGN.md` 신설**(SEL-1~5 사다리, 마스터 §7 등재).
- ✅**SEL-1~3 즉시 실측 (0원, `tb_selector_v2.py`)**: **SEL-1 채택** — Smoothie-식 proposer-prior 가중(β=2) SIG, **공식 link F1 66.48→67.22 = k0 대비 +9.5pp 갱신** / SEL-2 기각(ns) / SEL-3 작동(confidence=승자합의 u1 — 갭-기반은 만장일치 역전 결함, top-20% risk 절반). 권위 = TB결과 **§8.9c**.
- ✅**F4b compliant-pass 신설+상시화 (2026-06-13 새벽, 사용자 발의)**: pass∧위반-무 — **FULL-clean(G1+G2+G3+G4)에서 nogate 0.1425 vs gate_r2 0.1908 = 게이트 +4.8pp 우위**(parity 아님). nogate 신규 census: G2 미확인-WRITE 44 sims(제2 위반축). 공용 모듈 `tau2/t2_compliance.py` + `t2_run_gated` 자동 후크(compliance.json 사이드카) — 이후 모든 τ² 평가에 공짜 병기. 마스터 §1.6 F4b(사후-정의 플래그)·PORTFOLIO §3.7b 표. **airline에서 사전등록 재검증이 논문 절차.**
- **다음 세션 1순위 후보**: ①**S1 설계+발사**(A2 실-도메인 verified distill — S0-v2가 합성 과적합 확정) ②SEL-4(7B reverse-likelihood) 판단 ③P4 도착 감시 ④batch8+(200쌍 달성으로 완화) ⑤F4b 비교주장용 paired bootstrap CI 추가 ⑥**G4 deny-게이트 검증 (구현 완료 2026-06-13 — PORTFOLIO §3.8)**: 차선책(의무→사전조건 변환, 문구 미송신 transfer를 deny) 코드 반영됨 — 다음 τ² gate 실행(airline 번들)에서 사전등록 검증(G4≈0·pass ±0). offload(scaffold 직접송신)는 보류 대안. ⑦**★0원 발견: 선별기 풀 확장 — Track-B 기존 예측 재사용**: `trackb_raw/preds/data_multimedia_sub500/`에 32B/72B/235B(±guided) 예측 이미 커밋돼 있음 — `tb_selector_v2.py`/`tb_select_official.py` HM 풀에 합류만 하면 됨(신규 요청 불요). 예측: oracle 천장 0.896↑·SEL-1 prior가 대형모델에 고가중→선별 67.2↑. ⑧coworker 요청서 **v5** 발행됨: P4 보강(로컬 결과·앵커링 주의)+**P5 신설**(S1 교사-풀 컴파일, `node_run_s1_compile_p5.sh`·s1_inputs/telecom 커밋됨 — S1 크리티컬 패스라 P3보다 우선 권고).

### 0c. ★야간 배치 0613→14 가동 중 (다음 세션 첫 행동 = 수확)
- **드라이버 = `taskbench/tb_night_0614.sh`** (detached, 사전등록 머리 동결) · log `/home/woori/scratch/tb_night14.log` · sentinel **NIGHT14_DONE** (부분: ND/NB/NC/NA_DONE).
- **NA (GPU0+OpenRouter)**: τ² retail **gate r3** = G4 deny-게이트+중립템플릿 검증 — 예측 ①G4 위반 0 ②G4 deny 1~3건 ③pass^1 r2 동등 ④G1-G3 위반 0. compliance.json 자동(후크 첫 실전). 수확 = `t2_compliance.py`/`t2_gate_r2_verdict.py` --arms에 r3 추가.
- **NB (GPU1)**: v3mix+guided **K=8 temp0.8 샘플링** sub500 → `tb_v3g_mmk0-7`. **NC (CPU)**: C0=v3g k0 단일 / C1=v3g-AR8+H6 선별(예측 ≥68) / C2=C1+Track-B 6종.
- **★후속 드라이버 = `tau2/driver_s1_sel4.sh`** (NIGHT14_DONE 게이트 후 이어받기 — GPU 끊김 0) · log `/home/woori/scratch/a2_s1/s1_sel4_driver.log` · sentinel **S1_SEL4_ALL_DONE**.
  - **S1 (GPU0)**: 실-도메인 verified-distill 스모크 — 합성200+실(retail 3게이트+telecom 6게이트 Fable-5 spec, oversample8)=216쌍 LoRA SFT → **airline held-out census** (예측 applies_F1 **>0.528**=S0-v2 = 실-spec이 분포갭 교정). 수확 = `$OUT/s1_census.txt`. 설계 = A2_FRONTEND §S1.
  - **SEL-4 (GPU1)**: 7B reverse-likelihood Reviewer(`tb_reviewer_select.py`, Coder-Reviewer p(instr\|plan)) — v3g 풀(NB 산출) 위 MBR+Reviewer z-합성 재선별 → 공식 채점(예측 ≥SEL-1 단독). v3g 미생성 시 dpo2g 폴백. 수확 = `$OUT/sel4_*.txt`.
- **ND ✅완료 (조기 결과 — 사전등록 미달 정직 기록)**: 기존 dpo2g-풀 + **Track-B 6 proposer 확장 = link F1 0.6703 < 무확장 0.6722** (예측 ≥68 기각) — hetero-선택 89→120으로 늘었는데 공식 F1 정체/소폭 하락 = **대형 단일샷 proposer는 합의와 중복, 풀 확장 단독 무이득**(MBR bias-diversity의 bias 항 해석 후보). 72B prior 0.768=최고(품질 서열은 정확) — 선별 천장은 풀 크기가 아니라 **후보 다양성**의 함수라는 N2 기제 재확인. NC C2와 교차 확인 예정.

### 0b. 2026-06-12 저녁 세션 산출물 (이 핸드오프 §0 실행분)
- N3 판정 스크립트 `tau2/t2_gate_r2_verdict.py` (pass^k+deny-census+F4 replay+복구행동 census+matched 비교) — 결과 권위 = PORTFOLIO §3.7b.
- P-A2-0b 하네스 가동(파싱 버그 수정)·S0 파이프라인 신규: `t2_a2_s0_build_sft.py`(census-프롬프트 일치 빌드)·`t2_a2_s0_eval.py`(holdout 평가)·`driver_a2_s0_sft.sh`·어댑터 `sft_runs/qwen7b_a2_s0`.
- 렌더 part: `synth_b5_renders_fable5_part4/5.jsonl`·`synth_b6_renders_fable5_part1-3.jsonl`. 샘플러 v2(어휘 확장+게이트≤7).
- τ² 영구실패 4 sims = OpenRouter infrastructure_error(빈 대화) — 모델 무관, n=452.

## 1. day-4 확정 결과 (Day-4 한눈표 = TB §8.10 머리; 상세 각 §)
- **★D1 구조-DPO 채택**(3관문 적중: in-dom +1.9/+2.7·**held-out MM +4.5**·nself −77%/ndangle −93%) / **★D2 비용-DPO 기각**(전수 부검: edge −2.0·R −4.8 = brevity prior, 학습어휘 겹침 10%) ⇒ **★기제 명제(논문감): DPO는 쌍의 대조축을 도메인-일반 prior로 전이** (배선=이롭게/길이=해롭게). 비용은 ε-타이브레이크 선별로(노드 −3%에 edge −0.2).
- **★v3 혼합(균형714+구조1017) 신기록 56.46**(예측 ≥max 적중) → **N1 v3+guided 57.90 = 신 best-stack**(+0.68; 예측 ≥58에 0.1 미달 정직 기록·guided 가산 +1.44 안정).
- **★P-D(-1) Δhetero +13.6**(0.720→0.856, 기존 파일만) → diffusion 조건부 강등·**선별기=병목**(게이트 역선택 0.54<mean). **선별기 사다리**: raw MBR 0.716→**proposer-1표 0.751**→+validity필터 **0.753**(회수 44%) → **★N2 공식-척도 확정 +8.8**(선별 66.5 vs k0 57.7, sub500 내부-일관). zero-cost 메타규칙 최강 사례로 memory 박제.
- **★τ² 1차 (리더보드 표준 user-sim=gpt-4.1 via OpenRouter)**: 게이트 **F4 write-위반 43→1(98% 차단)** ↔ F3 pass^1 −3.7pp(**ⓟ1 기각** — deny→fail 92%, SOPBench passive-H3 동형) ⇒ 처방=deny-복구 메시지(N3, §0.1) + §3.8 **A2 spec-데이터/R3 불변템플릿 분리**(수동 프롬프트 0).
- **★A2 front-end 라인 가동**: 설계서(`A2_FRONTEND_DISTILL_DESIGN.md` — 역렌더 데이터엔진·S0-S2 사다리·시스템-vs-frontier 판정·LOCK 비적용 논증) · **P-A2-0 PASS**(Fable-5가 미답 airline 166줄→6게이트, 상태-추적 replay **over-deny 0/108**) · **P-A2-1 부트스트랩+135쌍**(round-trip KEEP·QC 무손실·스타일 6종×난이도 상승) · P-A2-0b 하한 census 하네스+P4 분담.
- coworker P2 적중(32B SFT+guided +3.9, 예측 +3~5)·**메트릭 배터리 §1.6 사전등록**(2-tier 규율·F1-F7·ⓟ1/ⓟ2)·MEMORY.md 35→6.5KB 압축.

## 2. 인프라 gotchas (day-4 추가분 — 재발견 금지)
1. **vllm serve↔adapter 저장 레이스**: 학습 [done] 직후 serve = safetensors 미완 로드 → 엔진 초기화 실패. 가드 = size-stability 확인(`tb_v3_mix.sh` 구현 참조).
2. **EngineCore 고아 반복**: APIServer 사망/종료 후 EngineCore 42-45GB 잔존(eval 스크립트는 시작 시만 kill) — 새 배치 전 `nvidia-smi` PID 직접 kill.
3. **tau2 3함정 (전부 수정 커밋)**: ①NL-assertion judge 하드 gpt-4.1 기본값(40/114 태스크 키부재 영구실패) → `t2_run_gated`가 재바인딩 ②judge가 json.loads 직접 → response_format json_object 강제 ③기존 저장 dir = 대화형 resume 프롬프트 → EOF 크래시 — 재실행 전 simulations/<name> 삭제.
4. **OpenRouter**: 키 = `source /home/woori/.openrouter_key`(export문 형식 — cat 금지). 표준 user-sim = `openrouter/openai/gpt-4.1`(= gpt-4.1-2025-04-14 정확 해석). qwen 7B/72B 슬러그 OK, 14B/32B 슬러그 상이.
5. **데스크톱 앱 업그레이드 = 로컬 폴러·워크플로 전멸**: 리모트 detached+git 채널은 생존. workflow resume은 **same-session only** → clear 후엔 재발사.
6. printf %·heredoc 이스케이프 충돌 잦음 — 분석은 전부 repo 스크립트로 커밋 후 실행. pkill은 EXIT -1 시 드롭 — 상태 검증 후 재시도.
7. 원격 push 충돌 잦음(coworker·자동push) — `git pull --rebase` 또는 stash 시퀀스, 원격 미스테이징 잔재는 stash 우회(커밋 금지).
8. **vllm 2개 동시 기동 = torch.distributed 내부 포트 레이스** (둘 다 8010 선점 시도 → 한쪽 EADDRINUSE 즉사, 2026-06-12 실측): API `--port`만 달라선 부족 — **`VLLM_PORT`를 인스턴스별 분리**(예: 8100/8200) 또는 순차 기동.

## 3. 메타 (day-4 규율 수확)
- **zero-cost 진단 최강 사례 박제**: P-D(-1)이 GPU 1.5일 diffusion 라인을 0원 census로 선별기 연구로 교체 — GPU 설계 사전등록에 "기존 산출물로 분리 가능?" 체크 강제 (memory `feedback-zero-cost-diagnosis-strongest-case`).
- 사전등록 즉시-기각 조항이 D2를 하루 안에 정리; 예측 0.1 미달(N1)도 그대로 기록 — 예측 동결의 값.
- 부검 도구의 재사용성: `tb_d2_autopsy.py` 하나가 D2 기각·D1 채택·v3 검증 3판정을 동일 잣대로.
- A2 산출물+검증기 인터페이스 = 생성기(프로그램/frontier/소형)를 교체부품화 — 질문이 "누가 만드나"에서 "검증기를 통과하나"로 이동.
