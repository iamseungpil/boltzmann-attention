# Coworker 실험 요청 — Scale 축 확장 (32B-fp16 · 72B · ~225B) × 다도메인 (retail·airline·banking) (2026-06-22)

> 요청자 컨텍스트 = 아래 §참조 링크. **한 줄 요청**: 큰 GPU에서 **32B-fp16 · 72B · ~225B** 세 모델을 tau2 **retail·airline·banking** e2e로 돌려 — **(A) 전 도메인 floor(scaffold 없음) bench-pass** [즉시 가능·핵심 scale×domain 그리드] + **(B) retail full-scaffold(deploy) + compliant-pass** [즉시 가능] 을 측정하고 census 회수해 주세요. 코드·하네스 준비됨(`--agent_model`/`--domain`/TP만 교체). **★주의: airline/banking의 *scaffold(게이트)* arm은 아직 prereq 차단(아래 §3b)** — 그쪽은 floor만 돌리고, 게이트 transfer는 우리가 엔진 배선 후 phase-2.

---

## 1. 왜 이 실험인가 (1문단)
우리 thesis 둘째 기둥 = "소형 모델 + 최소 *결정론 scaffold* 로 frontier tool-use 능력을 싸게 따라잡는다". 32B-int8서 측정 중인 **flow-discipline scaffold**(precondition-steering gate G5 + retry-controller)가 32B의 *첫시행 신뢰성* 갭(eligibility/wrong-tool/loop 실패)을 닫는지 보고 있습니다. **빠진 건 scale 축**(현재 1.5/7/14/32B-int8 측정·72B↑ 미측정). 이 요청이 그 축을 채웁니다. 검증할 가설:
- **H1 (addressable 수축)**: scaffold가 닫는 실패 클래스(eligibility/wrong-tool/loop = floor의 ~25%)가 **모델이 클수록 작아진다**(큰 모델은 애초에 그런 실수를 덜 함). → scaffold 한계기여가 scale↑서 감소하는 곡선.
- **H2 (cheap-replication / L-vs-E 크로스오버)**: "소형 + scaffold" 가 "한 단계 큰 모델 floor"에 도달하는가? = fleet-sizing 의사결정(`CAPABILITY_LEVER §10`)의 실측 점.
- **H3 (양자화 confound)**: 우리 32B-int8 baseline(bench pass^1 ≈ 0.55, compliant ≈ 0.49)이 **fp16서 다른가** = int8이 결과를 오염시켰나.
- **H4 (frontier 도달)**: 어느 scale + scaffold 가 frontier(gpt-4.1 0.82 / Claude Opus4.6 0.92 retail leaderboard)에 닿는가.

## 2. 모델 & serve (fp16·tool-calling 필수)
| tag | 체크포인트 | serve 주의 |
|---|---|---|
| `n32fp16` | `Qwen/Qwen2.5-32B-Instruct` (fp16) | 단일 80GB 또는 TP2. **우리 int8 대조군** |
| `n72` | `Qwen/Qwen2.5-72B-Instruct` (fp16) | TP2~4 |
| `n225` | **최대 체크포인트(~225B)** — 정확 ID 확인 요(예 `Qwen/Qwen3-235B-A22B` MoE, 또는 보유 최대) | TP4~8. ★tool-parser 확인(아래) |

serve recipe (검증됨·tool-calling 필수):
```bash
VLLM=/home/woori/venvs/tau2_vllm_env/bin/vllm
$VLLM serve <MODEL> --port 8360 --tensor-parallel-size <N> \
  --enable-auto-tool-choice --tool-call-parser hermes \
  --max-model-len 16384 --enforce-eager --gpu-memory-utilization 0.92
# healthcheck: curl -s localhost:8360/v1/models | grep <MODEL>
```
- **★tool-parser**: Qwen2.5(32B/72B)=`hermes` 검증됨. **비-Qwen2.5(225B/Qwen3 등)는 tool-parser가 다를 수 있음** → serve 후 **1-task 스모크**로 tool-call이 파싱되는지 먼저 확인(파싱 깨지면 전 sim이 빈 응답 → 무효). Qwen3는 `--tool-call-parser hermes` 또는 전용 파서 시도.
- `--enforce-eager` 유지(결정론 진단 관행). fp16 72B/225B는 KV 캐시 위해 TP/`--max-model-len` 조정 가능(retail 대화는 16k 충분).

## 3b. ★도메인별 readiness (정직 — 무엇이 지금 돌아가나)
| 도메인 | floor bench-pass | scaffold(게이트) arm | compliant-pass | 비고 |
|---|---|---|---|---|
| **retail** | ✅ 즉시 | ✅ G1-G5+retry (완비·검증) | ✅ full(G1-G4) 검증·회귀무손상 | 완전 ready |
| **airline** | ✅ 즉시 (`--domain airline`) | ⛔ phase-2 | ⚠️ 부분 | 게이트 G1-G3 gate.json 있으나 **런타임 auth가 user-제공-id 모델**(retail lookup과 다름)이라 엔진 미배선→G1 오작동. G5=no-op(airline write에 status-precond 없음). |
| **banking** | ✅ 즉시 (`--domain banking_knowledge`) | ⛔ phase-2 | ⚠️ 부분 | **banking.gate.json 부재**(작성 필요). 실행분기(no_knowledge+sandbox stub)는 `t2_run_gated`에 기존재. |
- ⇒ **coworker가 지금 돌릴 것 = (A) 3 도메인 × 3 스케일 × floor [bench-pass·핵심] + (B) retail × 3 스케일 × {floor, deploy} + compliant-pass.**
- airline/banking scaffold transfer(같은 엔진·gate.json swap이 lift하나)는 **우리가 phase-2 prereq 해결 후**(airline user-id auth 런타임 배선·banking.gate.json 작성·compliance auth-모델 일반화). 그때 동일 요청서로 추가.
- ★compliant-pass는 retail만 완전 신뢰. airline/banking은 `compliance.json` 나오나 auth-모델 caveat(bench-pass를 1차 신뢰).

## 3. Arm 매트릭스 (도메인 × 모델) — env 플래그로만 분기, 코드수정 0
**도메인 선택 = `--domain {retail|airline|banking_knowledge}`** (banking은 t2_run_gated가 no_knowledge 변종+sandbox stub 자동 적용). 아래 arm은 도메인마다 동일 env 스킴(단 §3b readiness대로 airline/banking은 floor만).
하네스 = `t2_run_gated.py`. gate 활성 시 `t2_gate_patch` 적용·env 플래그로 어느 게이트/레버를 켤지 결정.

| arm | 설명 | `--gate` | env | save_to |
|---|---|---|---|---|
| **floor** (필수) | scaffold 없음 = raw 모델 | `0` | — | `cw_<tag>_floor_retail` |
| **deploy** (필수) | full scaffold = G1-G4(auth/confirm/own/notice) + **G5(precondition-steering)** + **retry** | `1` | `T2_GATE_KINDS=auth,confirm,ownership,notice,preconditions T2_RETRY_CONTROLLER=1 T2_RETRY_K=3` | `cw_<tag>_deploy_retail` |
| g14 (여유 시) | compliant 베이스(G5/retry 없음) = G5 한계기여 분리용 | `1` | `T2_GATE_KINDS=auth,confirm,ownership,notice` | `cw_<tag>_g14_retail` |
| g15 (여유 시) | +G5 (retry 없음) | `1` | `T2_GATE_KINDS=auth,confirm,ownership,notice,preconditions` | `cw_<tag>_g15_retail` |

- **필수 = floor + deploy** (모델당 2 arm). 분해(g14/g15)는 GPU 여유 시.
- 우리 `flow_disc_fullgate.sh`가 정확히 이 패턴(`--agent_model`/TP만 바꾸면 그대로 재사용 가능). 복사해 모델·TP·tag만 교체 권장.

실행 예 (deploy arm·gpt-4.1 user-sim):
```bash
cd /home/woori/scratch/tau2-bench
source /home/woori/.openrouter_key                       # = export OPENROUTER_API_KEY=...  (cat 금지!)
export SSL_CERT_FILE=$(python -c "import certifi;print(certifi.where())")
export PYTHONPATH=src:$REPO/scripts/distill/tau2
rm -rf data/simulations/cw_n72_deploy_retail             # ★stale dir 제거(detached resume-prompt EOFError 방지)
env T2_GATE_KINDS=auth,confirm,ownership,notice,preconditions T2_RETRY_CONTROLLER=1 T2_RETRY_K=3 \
  $PY $REPO/scripts/distill/tau2/t2_run_gated.py --gate 1 --domain retail \
  --agent_model Qwen/Qwen2.5-72B-Instruct --agent_base http://localhost:8360/v1 \
  --user_llm openrouter/openai/gpt-4.1 --user_temp 0.0 \
  --num_trials 1 --max_concurrency 8 --save_to cw_n72_deploy_retail
```

## 4. 메트릭 (둘 다 필수) & 산출
- **표준 pass + compliant-pass = 자동**: `t2_run_gated` 가 평가 후 `compliance.json` 사이드카 산출(각 save 디렉토리·**도메인 gate.json서 제약 도출=A2-구동**). 변형: `bench`(=표준 pass^k) / `write` / `strict` / **`full` = compliant-pass^k = pass ∧ G1-G4 무위반**(배포-실제 지표). 회수 = `cat data/simulations/<save>/compliance.json`.
- ★**compliant-pass 신뢰도 (도메인별)**: **retail = full 신뢰**(검증). **airline/banking = bench-pass 1차 신뢰**(compliant도 산출되나 auth-모델이 retail형이라 G1/strict 해석 주의·`compliance.json`의 `_constants_nonempty`·warn 로그 확인). 즉 다도메인 1차 비교축 = **bench-pass**.
- **실패-클래스 census** (`scripts/distill/tau2/t2_failcensus.py`): floor 대비 deploy서 eligibility/wrong-tool·loop·no-write 실패가 줄었는지. (우리 census 스크립트 패턴은 `THIRTYB §6c` 참조.)
- **false-block 점검**: deploy서 *floor-pass였던 task가 깨졌나*(scaffold over-block). 있으면 보고(우리 NO-GO 신호).

## 5. 로지스틱스 & 함정
- **repo**: `/home/woori/workspace_common/boltzmann-attention-pi` branch `facet-rft-2026` → `git pull --ff-only` 먼저(remote-side 커밋 있으면 fetch+rebase).
- **⛔ COST GUARD (최우선)**: user-sim = **`openrouter/openai/gpt-4.1` 고정**. **Claude/Anthropic 절대 금지**(2026-06-16 공유키 ~$600 유출 사고). `t2_run_gated`가 Claude 모델이면 거부(`--allow-frontier` 쓰지 말 것). agent 모델은 로컬 vllm이라 키 무관.
- **gpt-4.1 비용 스케일**: sim당 ~수십 gpt-4.1 호출. 비용 = trials × tasks(114) × arms × models. **num_trials=1 로 먼저** 신호 본 뒤 양성이면 3으로 denoise 권장.
- **num_tasks**: 미지정(None) = retail 전체 114. (`--num_tasks 0` = "No tasks" 함정·쓰지 말 것.)
- **stale dir**: detached 실행 시 기존 save 디렉토리 있으면 tau2가 resume 프롬프트(`console.input`) → `</dev/null`서 EOFError 크래시. **실행 전 `rm -rf data/simulations/<save>`**.
- **진행률 가시**: `setsid bash drv.sh </dev/null >$LOG 2>&1 &` + `tail -f $LOG`(또는 마커 grep). `| tail` 파이프 금지(버퍼).
- **GPU**: 우리는 49GB A6000 2장(GPU0/1)·int8만 가능. 당신 환경의 fp16/TP·VRAM에 맞게 `--tensor-parallel-size` 조정.

## 6. 회수물 (요청)
**save_to 명명 = `cw_<tag>_<arm>_<domain>`** (예 `cw_n72_floor_airline`, `cw_n32fp16_deploy_retail`). 우선순위:
1. **(A) 핵심 그리드 = 3 도메인 × 3 스케일 × floor** (= 9 런·bench-pass). + **(B) retail × 3 스케일 × {floor, deploy} + compliant**.
2. 산출물: `data/simulations/cw_*/` (results.json + compliance.json) — repo 커밋 또는 경로 공유.
3. **요약 표**: domain × model × arm × {bench pass^1, (retail만)compliant full pass^1, eligibility/wrong-tool 실패수, loop 실패수, false-block 수}.
4. tool-parser 스모크 결과(특히 225B·도메인별 tool-call 파싱 확인) + serve config(TP/VRAM).

이걸로 H1(addressable 수축 곡선·retail)·H2(크로스오버)·H3(int8 confound·32fp16 vs 우리 int8)·H4(frontier 도달) + **도메인별 scale-gradient**(난이도·frontier 거리)를 채웁니다.

## 7. 참조 (현재까지 방향·결과 — 링크)
- **갭 분석 + 실패-클래스 census(이 실험의 동기)**: [THIRTYB_VS_FRONTIER_GAP §6c](scripts/distill/THIRTYB_VS_FRONTIER_GAP_2026_06_22.md) — 32B→gpt-4.1 갭=flow-discipline+info/communicate>operand; both-fail 11 분석; scaffold-addressable ~25%.
- **flow-discipline scaffold 설계(arm 정의·G5·retry·GO 기준·measurement plan)**: [FLOW_DISCIPLINE_SCAFFOLD_DESIGN](scripts/distill/FLOW_DISCIPLINE_SCAFFOLD_DESIGN_2026_06_22.md) — §6 측정계획·§1b census(floor bench 0.60/compliant 분해).
- **상위 프로그램(레버×비용)**: [RULE_LEVER_COST_EFFICIENCY_PROGRAM](scripts/distill/RULE_LEVER_COST_EFFICIENCY_PROGRAM_2026_06_22.md) · [CAPABILITY_LEVER_ALLOCATION_DESIGN §10 L-vs-E 크로스오버/fleet](scripts/distill/CAPABILITY_LEVER_ALLOCATION_DESIGN_2026_06_21.md).
- **scale 분해 기존결과(7/14/32B)**: [M_A_RESULTS §35](scripts/distill/ma/M_A_RESULTS.md).
- **전체 실험 프레임**: [EXPERIMENT_DESIGN](scripts/distill/EXPERIMENT_DESIGN.md).
- **코드**: 러너 [t2_run_gated.py](scripts/distill/tau2/t2_run_gated.py) · 게이트엔진 [gate_interpreter.py](scripts/distill/tau2/gate_interpreter.py) · wiring [t2_gate_patch.py](scripts/distill/tau2/t2_gate_patch.py) · A2 [retail.gate.json](scripts/distill/tau2/a2/retail.gate.json) · compliant-pass [t2_compliance.py](scripts/distill/tau2/t2_compliance.py) · census [t2_failcensus.py](scripts/distill/tau2/t2_failcensus.py) · 드라이버 템플릿 [flow_disc_fullgate.sh](scripts/distill/tau2/flow_disc_fullgate.sh).

## 8. 현재 진행 상태 (참고)
- 32B-int8 retail: floor bench pass^1 ≈ 0.55 / **compliant(full) 0.491** (gap 주로 G2 confirm 위반). flow-discipline arm(G5-only·G5+retry) 측정 *진행 중*(G5 라이브서 wrong-tool→steer 작동 확인). full-gate(G1-G5) compliant arm 후속.
- leaderboard(6/22): **retail** Claude Opus4.6 0.92 · gpt-4.1 0.82 · Qwen32B 미등재(Simia-Tau FT 0.617) / **airline** LongCat 0.765 · GPT-5.1 0.67.
- **★우리(요청자)가 처리할 phase-2 prereq**(이거 끝나면 airline/banking scaffold arm 추가 요청): (1) airline **user-제공-id auth** 런타임 배선(GateInterpreter가 `satisfied_by:user_provided_user_id` 처리) (2) **banking.gate.json** 작성(auth=get_user_information_by_name/email·write/ownership) (3) compliance auth-establishment 도메인-일반화(현재 lookup-tool 모델). compliance *상수 도출*은 이미 A2-구동화 완료(retail 회귀 무손상).
- **불변**: tau2 학습 금지(scaffold만·도메인-일반)·user-sim=gpt-4.1(Claude 금지)·grep if-domain=0.
