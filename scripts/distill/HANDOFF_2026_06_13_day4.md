# HANDOFF 2026-06-13 — day-4 종료 (D1채택/D2기각·v3신기록·선별기 1급양성·A2 front-end 라인 가동)
> 📌 **구조 안내**: 모든 설계·실험 문서의 단일 마스터 = repo `scripts/distill/EXPERIMENT_DESIGN.md` (**§7 문서지도**; 목표·순서 변경은 마스터 §0-§4에서만). 처음 읽는다면 마스터부터.

> **다음 세션 진입점.** 결과 권위 = `reports/facet_rft_2026/TASKBENCH_EXPERIMENT_RESULTS.md`(TB) **§8.9/8.9b/8.10/8.10b + Day-4 한눈표(§8.10 머리)** · τ²/A2 = `scripts/distill/BENCH_PORTFOLIO_FRAMEWORK_DESIGN.md` **§3.6-3.9** · A2 학습 = `scripts/distill/A2_FRONTEND_DISTILL_DESIGN.md`. 리모트 규칙 = memory `reference-remote-server-environment` (ssh stdin 파이프·cd /c/workspace 먼저).

## 0. ★첫 행동 (순서대로)
1. **N3 수확 (τ² G1 deny-복구 메시지 게이트 재실행)**: clear 시점 438/456(96%)·영구실패 3. `grep -a "t2_run RESULT" /home/woori/scratch/t2_gate_r2.log`. **판정(사전등록)**: deny→fail 92%→**<50%** ∧ gate pass^1 0.147→**≥0.184**(nogate 동등 회복) ∧ write-차단 ~98% 유지. census는 `t2_passk_census.py`가 arm명 nogate/gate 하드코딩 — gate_r2는 results.json 직접 분석(데이터 = `tau2-bench/data/simulations/retail_7b_gate_r2/results.json`). 영구실패 3건 부검 포함. 결과 → PORTFOLIO §3.7에 행 추가.
2. **GPU 회수 후 (N3 종료 시 양쪽 빔)**: ①**P-A2-0b 로컬 7B/14B census** (`t2_a2_size_census.py` — vllm serve 후 `--model name:http://localhost:PORT/v1:HFmodel`, ref=`specs/*_gate_spec_fable5.json`, 예측 7B<0.5) ②**S0 스모크 학습**: 135쌍(`specs/a2_s0_dataset_v3.jsonl`)으로 7B LoRA SFT(기존 dpo_train 아닌 SFT — build 스크립트 신규 필요: prompt=정책NL+카탈로그, target=spec JSON, guided는 추론시) → 학습 전후 7B census 비교 = **front-end 첫 학습 신호**. 순서: census 먼저(baseline)→S0→재census.
3. **데이터 생성 계속 (Fable-5 직접, 사용자 지시 누적)**: batch5b = `specs_synth_b5.jsonl`(리모트)의 id 15-29 잔여 15개 렌더 → batch6+(seed 6+) → **목표 200쌍**. 절차: 샘플 fetch→6스타일 로테이션 렌더(part jsonl)→`t2_a2_join_qc.py`(이중언어 토큰 병기=exact pass)→dataset 병합 push. 누적 135 전부 QC 무손실.
4. **딥 리서치 2건 재발사 (clear로 소실 — resume은 same-session only)**: ①이종-풀 robust 선별기 문헌(브리프 핵심=TB §8.9b: MBR 0.753·proposer-1표·게이트 역선택·gold-free/결정론/7B 제약) ②framework-tier 메트릭(브리프=마스터 §1.6 표: pass_hat_k 추정량·0-위반 CI·AURC·비용곡선 표준). 합류 후 → 선별기 설계서(detail, 마스터 §7 경유) + §1.6 v2 동결 (task #7이었음).
5. **coworker**: P4(A2 크기 census, 32B/72B/235B) 요청 발행됨(요청서 §10·턴키 `node_run_a2_census_p4.sh`) — trackb_raw/p4_a2_census 도착 감시. P2 적중(+3.9) 도착 확인됨. P3a 사전예측에 run7 게이트 발견 주석 전달됨.
6. **잔여 GPU 큐**: v4 재채굴 사이클 검토(v3 rollout K=8→재채굴) · 선별기 합성(v3+guided 57.90 위에 N2-식 이종풀+선별 = 72B 앵커 63.5 추격) · P-D0 diffusion(조건부 강등 유지).

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
