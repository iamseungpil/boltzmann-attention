# HANDOFF 2026-06-12 — day-3 종료 (프레임워크 확정·레버 비선형 발견·D1/D2 DPO in-flight)
> 📌 **구조 안내**: 모든 설계·실험 문서의 단일 마스터 = repo `scripts/distill/EXPERIMENT_DESIGN.md` (**§7 문서지도**; 목표·순서 변경은 마스터 §0-§4에서만). 처음 읽는다면 마스터부터.

> **다음 세션 진입점.** 결과 권위 = `reports/facet_rft_2026/TASKBENCH_EXPERIMENT_RESULTS.md`(이하 TB) **§8.5-8.8·§9.5b·§9.6·§10·§10.5·§1.5**. 포트폴리오 = 마스터 §1.5 + `BENCH_PORTFOLIO_FRAMEWORK_DESIGN.md`. 리모트 규칙 = memory `reference-remote-server-environment` (**ssh_run --cmd에 큰따옴표 금지 — stdin 파이프**, `@'...'@ | py -3 ssh_run.py`).

## 0. ★첫 행동 (순서대로)
1. **DAY_REPORT 확인**: `git pull` 후 `reports/facet_rft_2026/DAY_REPORT_2026_06_12.md` 존재 확인 (주간 배치가 자동 push; PUSH_FAIL이면 ssh로 `cat /home/woori/scratch/tb_day.log` tail + 원격 `git stash→pull --rebase→push→stash pop` 수확).
   - 내용 = **D1 구조-표적 DPO**(1017쌍, GPU1)·**D2 비용-표적 DPO**(376쌍, GPU0) — rft2 위, v2 하이퍼 동일. 자동 체인: 학습→병렬 평가(MM full+in-domain sub500×2)→P/R·궤적 census→보고서.
2. **판정 (사전등록, `tb_day_0612.sh` 머리에 동결)**:
   - D1: ⓐin-domain edge +1~3? ⓑheld-out MM Δ vs rft2 49.0(방향 미커밋 — 본 실험의 질문) ⓒnself/dangle ↓?
   - D2: ⓐ평균 n_nodes ↓ ⓑP ↑·edge ±1 내 ⓒ**short/deficit 악화 감시**(v1 거울상 — 악화면 즉시 기각).
   - 통제: rft2 49.0 / dpo2(v2) 55.95 / in-dom HF 51.6·daily 85.0. census는 `census_rft2_to_dpo_{struct,cost}.md`.
3. **판정 후 분기**: D1 성공(held-out +) → 혼합쌍(균형714+구조1017) best-stack v3 재학습 검토. D1 in-domain만 → RFT-동급 기록(보상-side 한계 정합). D2 성공 → parsimony를 v3 쌍에 합류.
4. **GPU 비면 → P-D0 diffusion 스모크** (`TB_DIFFUSION_PROPOSER_DESIGN.md` 사전등록): Dream-7B 다운로드됨(`dl_dream.log`/DREAM_DL_DONE 확인) — 50개×K=4, **형식 준수율이 1차 관문**.
5. **coworker P2 확인**: 그들 노드서 5-arm 진행 중(`node_run_taskbench_p2.sh`, 951eb71 — 리뷰 노트 v4 §7에 전달: 235B+#18819 리스크). trackb_raw 신규 push·§8.5 행 추가 여부 — 도착 시 P2a-1(+3~5 예측) 대조.

## 1. day-3 확정 결과 (전부 박제·push — 상세는 TB 해당 §)
- **★2×2 factorial 완결 (E1, §9.5b)**: base 50.00/base+guided **50.13(+0.13)**/FT 55.95/FT+guided **57.22(+1.27)** — 상호작용 ~10× = "CD=FT-간섭 회복장치" factorial 증명. 사전예측 적중.
- **★레버 비선형 (E2/E5/E6, §8.8 — day-3 최대 발견)**: in-domain 선별 헤드룸 +17.2 ↔ **best-stack 위 held-out 갭 +1.4** — guided·DPO가 분산을 이미 흡수, **잔여 L6는 수렴 정책의 K-분포 밖** ⇒ 처방 = ④구조-표적 DPO(D1, in-flight)·이종 제안기(P-D0 대기). 스코어러 v1(그래프-멤버십) 회수 18→22.6%뿐.
- **guided 3회전 완성 (E8, §9.5b)**: held-out MM +1.27·HF **+2.8**(35.0→37.83)·daily +8.0 — 효과 ∝ 간섭 질량.
- **promptslim 변수 = 이름 자기서술성 (E4, §9.5b)**: daily −0.6(자기서술 API명) vs HF −4.1(불투명 모델id) — in/held-out 아님. guided는 in-domain에도 +0.3~+1.4.
- **edge-snap v0 NULL (§8.7)**: 값-인라이닝 질량 없음(대형 0건) — 관례분은 "다른 유효 플랜"이라 정준화 불가.
- **P1 32B prereg 적중 + 독립 재검증 (§8.5)**: Δ −5.4 vs 예측 −5.0·간섭 −4.8pp 그대로·trackb_raw 원본을 우리 도구로 15/15 재현. **32B 누락축 소멸**(deficit +0.024). §8.6 = Track-B 전수 census(Qwen3 평탄=L6 구조축·in-domain SFT도 간섭 −6pp·temporal 형식 대형 무결).
- **TaskBench 외부 동결 확정 (§1.5, 3-agent ~150편 전수)**: 표준 프로토콜 gpt-4 미돌파·GPT-4o 서브셋 64.4=frontier 정체·ToLeaP "GPT-4o"행=도메인-전치 복사(**인용 금지**)·gpt-4 공식수치 두 벌 함정.
- **프레임워크 §10.5**: 벤치-불변 R1-R8 × 어댑터 A1-A5 — **A2(정책 NL→제약)가 유일 난제=thesis front-end**. 포트폴리오 = τ²(1순위, retail 게이트 4종이 SOPBench 동형 실측 — `BENCH_PORTFOLIO` §3.5)→Amazon SOP-Bench(12도메인)→AppWorld·ODCV. 마스터 §1.5 등재.
- **선행연구 §6.5 (5-agent 전문 적대검증 후 정정)**: ToolDec도 FT-stacking 했음("complementary" 자인) — 차별점 = same-base 통제 2×2+census 귀속+발생론. ToolDec 인용=v3·"names-only"=v1 한정. **arXiv 규율**(허위 레퍼런스=전저자 1개월 제재) memory+§9 박제.
- coworker: P0 완주(72B 63.5·235B 56.4)·P2 드라이버 가동·구 P2/P3 supersede 정리(v4 배너·§7=현행 P2 명시).

## 2. 실행 큐 (TB §9 권위 — day-3 갱신분)
1. DAY_REPORT 판정 (§0) → v3 재학습 여부.
2. P-D0 diffusion 스모크 → P-D1 혼합-풀 oracle (ⓐ+2↑ 채택/ⓑ기각 둘 다 1급).
3. **τ² 어댑터 (포트폴리오 1순위)**: A1 도구추출 스크립트 + retail 정책 게이트 4종 수동 컴파일(Guard-2 절차 재사용) → 7B±게이트 pass^1/pass^k. 클론 = `/home/woori/scratch/tau2-bench`.
4. coworker P2 결과 합류 → P3c(ODCV)→P3b(Amazon census)→P3a(τ² 대형, 우리 어댑터 인계 후).
5. HF 트랙 best-stack(rollout→균형 DPO→guided) — 전 도메인 헤드라인 표.
6. 논문 정리: §10 골격·§6.5 related-work·§1.5 인용위생·R8 규율.

## 3. 인프라 gotchas (day-3 추가분 — 재발견 금지)
1. **새 GPU 배치 전 양쪽 GPU vllm 전체 kill 필수** — day 배치 첫 기동이 야간 잔여 vllm(41.5GB)로 OOM (드라이버에 사전-kill 넣을 것).
2. **원격 자동-push는 PUSH_FAIL 대비**: 네트워크 단절 시 보고서가 git에 안 옴 — 수확 = 원격 `git stash→pull --rebase→push→stash pop` (원격에 미스테이징 SOPBench 잔재 3파일 있음 — 커밋 금지, stash로 우회).
3. **SSH 야간 장애 패턴**: TCP/인증 성공·출력 stall — 작은 출력(grep -c)은 통과, 큰 출력 실패. 대응 = 작은 조각 명령 + git 채널 폴링(원격 자체-push 설계 덕에 데이터 안전).
4. **도메인 `data.json`은 gold 조인 불가**(task_nodes 비표준) — kgate류 조인은 **eval-dir의 data.json** 사용.
5. **inference.py `--llm` = served-name이자 pred 파일명** — arm별 파일이 필요하면 served-name으로 돌리고 `mv` (promptslim/night/day 배치 패턴).
6. K-샘플 = inference.py를 `--temperature 0.8`로 K회 + per-k mv (별도 샘플러 불요).
7. PS argv 따옴표 절단 → **ssh_run stdin 파이프**가 정식 경로 (ssh_run.py에 utf-8-sig+CRLF 정규화 적용됨).

## 4. 메타 (day-3 규율 수확)
- **사전예측이 깨진 곳이 가장 비쌌다**: E6 갭 +1.4(예측 >+5)→레버 비선형 발견 / E4 HF −4.1(예측 <3.1)→자기서술성 변수. 예측 동결 없이는 둘 다 "그냥 수치"로 지나갔을 것.
- 문서 규칙: 새 "권위" 문서 금지 — 마스터 §7 경유(1회 위반 후 정정, memory 박제). 전 활성 문서에 구조 배너.
- 조건-게이트 arm은 전제 불발 시 **명시적으로 닫기** (구 P2 모호성 사건).
