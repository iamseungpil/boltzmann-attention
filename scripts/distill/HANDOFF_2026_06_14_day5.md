# HANDOFF 2026-06-14 — day-5 (선별기 헤드라인 확정·compliant-pass 체계·다양성 법칙 부검·관련연구 full-read 라운드)
> 📌 마스터 = repo `scripts/distill/EXPERIMENT_DESIGN.md` (§7 문서지도·§1.6 v2 메트릭). 결과 권위 = `reports/facet_rft_2026/TASKBENCH_EXPERIMENT_RESULTS.md`(TB §8.9c~h)·`scripts/distill/BENCH_PORTFOLIO_FRAMEWORK_DESIGN.md`(τ²/A2 §3.7b~e). 리모트 규칙 = memory `reference-remote-server-environment`. 직전 핸드오프 = `HANDOFF_2026_06_13_day4.md`(승계).

## 0. ★첫 행동 (순서대로)
1. **관련연구 full-read 6편 수확·통합 (clear 전 발사 — 결과는 디스크에 남음)**: `reports/facet_rft_2026/relwork_{selector,metrics,diversity,arch,nlformalize,determinism}_2026_06_14.md` 도착 확인(ls). **각 파일 끝의 (a)설계서 정정사항 (b)결정적 논문 (c)버릴 불공정 인용**을 읽어 → ①틀린 인용/수치 정정(인용규율! — DiG-Plan 0.32/0.94 정정 선례) ②`related_work_INDEX_2026_06_14.md` 통합 작성 ③각 설계서(SELECTOR/§1.6/A2/FIELD_GAP §5.6/TB_DIFFUSION) 반영. ⚠️미래-날짜 ID(2025-26) 중 "안 풀리는 것=suspect" 플래그 확인 필수. + diffusion 전반 보고서(`research_diffusion_planning_2026_06_14.md`) 도착 시 합류.
2. **determinism 실험 수확 (DET_DONE)**: `cat /home/woori/scratch/det_autopsy.txt` (또는 day13_summary.txt). 4-trial seq 동일률 0%→**≥70%면 배칭이 ⓟ1 비결정 주범 확정·ⓟ1 재개 경로** / <70%면 batch-invariant vLLM(별도 venv·VLLM_BATCH_INVARIANT, 0.11.0 미탑재) 도입 차기. 결과 → PORTFOLIO §3.7d.
3. **P-D 착수 판단 (수확 종합 후)**: P-D0 형식게이트(`/home/woori/scratch/plora_pd0_pd0.txt` — diffusion JSON 파싱율·valid_frac) + relwork_arch의 **A3 any-order AR 공정성 판정**(diffusion-not-required 닻 강화/약화) + DiG-Plan 정독(§3c: 실제 +10%·단일샷 무승부) 종합 → diffusion 돌릴지 vs A3로 갈지. **주 지표 = unique-correct(D-oracle), hot-T AR 비교**(greedy 금지).
4. **잔여 큐**: ①A2 faithfulness 검사 구현(NL-gloss↔source 대조 — replay 사각지대, relwork_nlformalize 처방) ②S1-v2(P5 도착 → dose-response 1/3/6/9·StepFun dual-dataset 템플릿) ③τ² G4 deny-게이트는 다음 gate 실행서 검증 대기 ④coworker P4/P5 도착 감시(trackb_raw).

## 0.1 clear 시점 백그라운드 상태 (재발사 불요·결과만 수확)
- **GPU**: GPU1 P-D0(Dream diffusion, ~2.8h timeout 거의 종료) → 종료 시 **determinism 드라이버 자동 시작**(GPU0, PLORA_PD0_DONE 게이트)→DET_DONE. day13 ✅완료.
- **에이전트 7개**(6 relwork + diffusion 전반): clear로 세션 소실되나 **별도 .md 파일은 디스크 잔존**. 완료분만 파일 존재 — ls로 확인 후 통합. ✅**relwork_arch 완료**(A3 정독 정정 반영됨: 표준AR에 짐·diffusion 필적은 2B-vs-65B 효율정규화 한정 = framing 닻만 / **ATLAS-RTC `2603.27905` 미해결=가짜→bib drop** / PLaT `2601.21358`·OATS `2603.13426` 실재 확인 / top-2=XGrammar·SoS-distill). 나머지 5편(selector·metrics·diversity·nlformalize·determinism) 도착 대기.

## 1. day-5 확정 결과 (전부 권위본 기록·push 완료)
- **★선별기 헤드라인 자격 = 둘째-기판 재현 (TB §8.9e)**: hf sub500 k0 0.350→**SEL-1 0.479(+12.9pp)**(사전등록 4배 초과·LODO 전이설정). MM +10.3pp가 기판-특이 아님. ⚠️SEL-4는 hf서 −0.3pp = 기판-의존(SEL-1이 코어).
- **★다양성 법칙 + 곱-가정 부검 (TB §8.9f/g/h)**: P-lora(목적-다양 어댑터 8종) 다양성 0.1535(단일정책 10배)·**회귀 gain~diversity +0.077 SIG** = 예측법칙 승격. **단 통합22풀 음성**(0.626<best 0.680): mean ↑·oracle 불변·**회수율 61→40% 붕괴** = "곱 가정 기각, P-lora가 D-oracle≈0(다르게 틀림)으로 MBR 합의 교란". ⇒ **풀 admission = 쌍별다양성 아닌 D-oracle 게이트**. 최적 풀 = dpo2g+H6(0.680) 유지. 부검기 = `tb_unified_autopsy.py`.
- **★F4b compliant-pass 체계 (PORTFOLIO §3.7b-e)**: pass∧위반-무. FULL-clean(G1+G2+G3+G4) nogate 0.1425 vs gate_r2 0.1908(+4.8pp). 벤치 pass는 compliance-blind(위반-pass 6/21). eval-후크 상시화(`t2_compliance.py`). **frontier F4b(§3.7e)**: gpt-4.1 위반 G2 4건뿐 = "frontier도 크게 깎인다" 강형 기각·게이트 model-agnostic 입증. **τ² gate r3(§3.7c)**: G4 deny-게이트 PASS(위반 0·pass^2-4 회복).
- **★ⓟ1 root-cause 확정 (PORTFOLIO §3.7d, `t2_p1_autopsy.py`)**: temp0인데 4-trial seq 0/111 동일 = **에이전트 생성 비결정성(vLLM 배칭)**이 pass^4 분산 주범, 채점잡음 0%·공통실패 45%(능력바닥). user-sim 분산 아님. ⓟ1=인프라-조건부 재개(batch-invariant 시).
- **★DiG-Plan 정독 정정 (TB_DIFFUSION §3c, 인용규율)**: 0.32/0.94 = **합성 토이(TaskBench 아님)**·실제 +10%·단일샷 무승부·greedy-AR tilt. 최종=diffusion+AR-refiner 하이브리드(=사용자 제안 일치). 수학: AR order-penalty(XLNet/ARDM)·quadratic error(2204.01171) 증명가능, 단 이득은 디코딩-regime이지 모델클래스 아님.
- **★A2 novelty 좌표 (FIELD_GAP §5.6)**: 최근접=Prose2Policy(2603.15799, frontier-prompt-only)·S1템플릿=StepFun(2508.04440). **A2 헤드라인 방어 = raw accuracy 아닌 주권+LODO 전이**. replay 사각지대=fabricated gate → faithfulness 검사 추가.
- **★A3 any-order AR(2601.13228)** = "set 다양성에 diffusion 불요" 닻 (TB_DIFFUSION §3d, relwork_arch 공정성 판정 대기).

## 2. 인프라 gotcha (day-5 추가 — 재발견 금지)
1. **vLLM 동시 multi-LoRA 서빙 라우팅 버그**: 8 LoRA를 한 serve `--lora-modules`로 올리면 마지막 1개만 채워지고 나머지 빈 출력(day13 P-lora 사고). **→ 순차 서빙(어댑터당 serve→infer→kill)** = `driver_plora_pd0.sh`/`driver_plora_select.sh` 패턴.
2. **vLLM 0.11.0 = batch-invariant 미탑재**(VLLM_BATCH_INVARIANT 없음·소스 grep NONE). temp0도 비결정(배칭). 결정론 측정 = `--enforce-eager --max-num-seqs 1` + client concurrency 1(배칭 격리). 완전 재현은 batch-invariant 지원 버전 별도 venv.
3. **divgen/census 단일후보 가드**: validity 필터 후 후보 1개면 쌍별 계산 ZeroDivision — skip 가드 필수(`tb_divgen_analyze` 수정분).
4. **data.json gold = tool_links(이중 JSON 인코딩)**·predictions = user_request — diffusion 샘플러는 instruction 키. 키 혼동 주의.
5. **선별기 풀 그룹**: 같은-정책 K샘플=1그룹(ar_group) / 이종 어댑터=각 독립그룹(--ar_group_per_slot) — proposer-1표 가중의 전제.

## 3. 메타 (day-5 규율 수확)
- **인용규율 작동 사례**: DiG-Plan 0.32/0.94를 그대로 박았으면 허위 레퍼런스. 정독으로 "합성 토이"임을 잡아 정정 = 원문검증·수치이식금지 규율의 값.
- **D-oracle > 쌍별다양성**: "다르게 틀림"(쌍별거리)이 아니라 "다르게 맞음"(oracle 기여)이 선별 레버 — 곱-가정 부검·P-lora 음성이 정량 확정. 풀 admission·다양성원(diffusion/A3) 판정의 단일 기준.
- **full-read 라운드 = abstract-검증 너머**: 8 research 보고서(179 unique arXiv) → 주제별 6 relwork 에이전트 full-text 정독 = thesis 관련연구를 1차-검증·본문기반으로 승격.
