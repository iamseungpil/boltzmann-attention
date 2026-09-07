# x812 — 클라우드 GPU(vLLM) 셋업 계획서 (2026-09-07)

**왜**: 2026-09-07 사용자 지시로 우리 GPU 는 `.153` GPU0 **하나뿐**([[30]] 개정). 실측 병목은 KV —
도는 엔진 지표 축자 `kv_cache_size_tokens=171,749` · `kv_cache_max_concurrency=1.31` ·
`num_requests_running=2.0`(conc 4 요청인데 **2개만** 배치에 든다). 남은 일이 1 GPU 로 **1주일 이상**이다.
**무엇**: 80GB GPU 를 빌려 **같은 vLLM 스택**을 세우고 base 잔여 + A/B 를 거기서 돈다.
**왜 OpenRouter 가 아닌가**: 우리 팔이 vLLM 전용 확장 5종에 의존한다(§6). OpenRouter 는 못 한다.

---

## 1. 재현할 환경 (전부 실측)

| 항목 | 현재 값 |
|---|---|
| OS | CentOS Stream 9 |
| NVIDIA 드라이버 | 580.95.05 |
| **vLLM** | **0.27.1** |
| torch | 2.13.0 |
| transformers | 5.15.1 |
| 러너 python | 3.12.12 (`seka_env` · `iso_tau3/venv`) |
| 모델 | `Qwen/Qwen3.8-27B-FP8` · HF 캐시 **29GB** |
| repo | `boltzmann-attention-pi` @ `facet-rft-2026` (현재 `5e269699`) |
| GPU(현행) | RTX A6000 48GB · **Ampere sm_86** |

**⇒ 대여 목표**: **A100 80GB (Ampere sm_80)**. 같은 Ampere 계열이라 FP8 dequant 커널 경로가
같다 — L40S(sm_89)·H100(sm_90)은 **네이티브 FP8** 이라 커널이 갈린다([[54]]·[[84]] 위험).

---

## 2. vLLM 발사 명령 (현행 그대로 · 보존본에서 가져옴)

```bash
vllm serve Qwen/Qwen3.8-27B-FP8 \
  --port 8141 --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
  --max-model-len 131072 --gpu-memory-utilization 0.9 \
  --enable-prefix-caching --max-num-seqs 128
```
출처: `/home/woori/scratch/x768/relaunch_8143.cmd`(실행 중이던 프로세스 cmdline 원본).

**⛔ 한 칸도 바꾸지 않는다** — `--tool-call-parser qwen3_coder` 는 [[84]] 가 박제한 짝이다.
80GB 에서 `--gpu-memory-utilization 0.9` 이면 KV 가 **~480k 토큰**(현행 171,749 의 ~2.8배)이 되어
`kv_cache_max_concurrency` 가 **~3.7** 로 오른다 ⇒ nt=4 가 **한 배치에 들어간다**.

⚠`.151` 에서 필요했던 환경변수(참고): `VLLM_USE_FLASHINFER_SAMPLER=0`(CUDA 11.8 서버였기 때문) ·
`VLLM_CACHE_ROOT`. 클라우드는 보통 CUDA 12.x 라 **필요 없을 것**이나, 점화 실패 시 1차 처방이다.

---

## 3. 설치 절차 (30~60분 · 순서 고정)

```
① 인스턴스: A100 80GB ×1(또는 ×2) · 디스크 ≥ 150GB · CUDA 12.x 이미지
   ⚠디스크: 모델 29GB + tau2 클론 + sim 결과. 100GB 는 빠듯하다.
② 시스템:  git, python3.12, curl
③ venv 두 벌
   - 엔진용:  pip install vllm==0.27.1          (torch 2.13.0 이 함께 온다)
   - 러너용:  tau2-bench 의존성 (+ litellm)
④ 모델:    huggingface-cli download Qwen/Qwen3.8-27B-FP8   (29GB · 10~20분)
⑤ repo:    git clone -b facet-rft-2026 <repo>   ⛔[[32]] CDP 자산 없음 확인
⑥ tau2:    tau2-bench 클론 + `alltools` retrieval 변형 등록 확인(§7 함정②)
⑦ 키:      OpenRouter 키를 **파일로만** 주입 — `~/.openrouter_key` (0600)
           ⛔[[30]] 커밋 절대 금지(2026-06-16 유출로 ~$600 abuse 사고)
⑧ 엔진 점화 → `/v1/models` 로 id 대조(⛔[[30]] 포트만으로 엔진 식별 금지)
```

---

## 4. 착수 전 게이트 (통과 못 하면 런 금지)

| # | 게이트 | 통과 기준 |
|---|---|---|
| G1 | `/v1/models` id | `Qwen/Qwen3.8-27B-FP8` **문자열 일치** |
| G2 | KV 실측 | `curl /metrics` 의 `kv_cache_size_tokens` 이 **171,749 보다 크다**(80GB 확인). 작으면 메모리가 기대와 다른 것 |
| G3 | **툴콜 파싱 스모크** | 3 태스크 · nt=1. 로그에 `SALVAGED=` **0** · `**TRUNC**` **0**([[84]]·[[30]] 게이트) |
| G4 | **겹침 대조** | 로컬에서 이미 **4/4** 인 태스크 **3개**(예 `001` `002` `005`)를 nt=4 로 재실행 → **12/12 통과**여야 [[54]] 짝 성립. 하나라도 어긋나면 **base 잔여를 여기서 돌리지 않는다**(A/B 만 돌린다) |
| G5 | 영속 왕복 | 결과 gzip → `git add -f` → push → 로컬에서 pull 로 회수 **1건 성공** |

⚠**G4 를 건너뛰지 마라.** 같은 Ampere 라도 가정이다 — x738 정본(42/97)과의 짝이 여기 걸려 있다.

---

## 5. 영속 절차 (⛔ 인스턴스는 언제든 죽는다)

[[30]] 실측 사고: *"sim 결과는 gitignored → GitHub 에 없음 → 복구불가"* · *"하룻밤 6런 24 sim 이
`sim_results/` 에 놓인 채 전부 untracked 였다"*. **마켓플레이스 인스턴스에서는 이 위험이 훨씬 크다.**

⇒ **워커에 내장한다** — 태스크 하나가 끝날 때마다:
```
gzip -c <sim>/results.json > $REPO/reports/facet_rft_2026/sim_results/<tag>.results.json.gz
git add -f <그 파일> && git commit -m "results: <tag>" && git push
git ls-files --error-unmatch <그 파일>        # ← tracked 확인까지가 절차
```
로그·사이드카(`fb_*.jsonl`)도 같이 올린다 — 오늘 x808·x810 이 전부 사이드카로 나왔다.

---

## 6. 왜 OpenRouter 로는 안 되는가 (기각 근거 · 실측)

| vLLM 전용 확장 | 위치 | 빈도 |
|---|---|---|
| `extra_body.structured_outputs.grammar` | `t2_guided_patch.py:224` (`T2_GUIDED=1`) | **sim당 29~41회** |
| `guided_choice` | `x157_entrainment_lambda.py:53` ← **x166 의 측정 그 자체** | 팔마다 |
| `extra_body.chat_template_kwargs.enable_thinking` | `t2_run_gated.py:723` | 프로브마다 |
| `extra_body.include_stop_str_in_output` | `t2_run_gated.py:748` | 절단 재샘플 |
| `extra_body.bad_words` | `t2_gate_patch.py:7134` | 금칙어 재생성 |

옮기면 **조용히 무시**된다(에러가 아니라 no-op) ⇒ 우리 팔이 아닌 팔을 재게 된다([[81]]·[[84]]).

---

## 7. 알려진 함정 (x738 §4 · [[30]] — 그대로 만난다)

1. `--gate 0` 경로의 `sys` 미임포트 크래시 → 우회 = **`T2_MAX_MODEL_LEN` 선언**
2. **`alltools` 변형이 클론마다 다르다** — base 는 `iso_tau3` 계열 클론에서 돌아야 했다. 새 인스턴스에서 **G3 전에 확인**
3. 키 미주입 401 → `source ~/.openrouter_key`
4. `get_response_cost` 의 *"model isn't mapped"* ERROR 는 **무해**(비용 산정만)
5. `pkill -f` 금지 — **PID 명시 kill**
6. 툴 타임아웃 ≠ 원격 종료 → 재발사 전 `ps -eo cmd | grep "[t]2_run_gated"`
7. 장기 잡은 `setsid … &` + 로그 리다이렉트(채널로만 보내면 타임아웃에 유실)

---

## 8. 비용·일정

실측 처리량(로컬 A6000·conc 1.31): 태스크당 ~2시간 = **188 sim ≈ 94 GPU-시간**.
80GB 로 conc ~3.7 이면 **~2.8배** + [[83]] prefix 적중률 회복(3.5%→56%) ⇒ 보수적으로 **~30시간**.

| 구성 | 시간 | 비용 |
|---|---|---|
| A100-80 ×1 @ $0.67 | ~30h | **≈ $20** |
| A100-80 ×2 @ $1.34 | ~15h | ≈ $20 |
| (참고) H100 @ $1.49 | ~15h | ≈ $22 |

이어서 **C-DIRECT A/B**(우리 팔 2팔 × 97 · nt=1 = 194 sim) ≈ 20~25시간 → **≈ $27**.
`x166` 재측정 <1시간 → ~$1. **총 ≈ $48**.

⚠ 시간 추정은 **외삽**이다. G2(KV 실측)와 G3(스모크 3 태스크)에서 **실측으로 갱신**한다.
⚠ 스토리지는 **인스턴스가 멈춰 있어도 과금**된다 — 끝나면 **stop 이 아니라 destroy**.

---

## 9. 실행 순서

```
0  계정·크레딧 준비(§10) — $50 선입금이면 충분
1  인스턴스 확보 → §3 설치 → G1·G2
2  G3 스모크(3 태스크 nt=1) → G4 겹침 대조(3 태스크 nt=4) → G5 영속 왕복
3  ⛔여기서 멈추고 G4 결과를 보고한다. 통과해야 base 잔여를 옮긴다
4  base 잔여 47 태스크 투입(큐 파일 그대로 이관) · 로컬 GPU0 은 병렬로 계속
5  끝나면 x166 재측정 → C-DIRECT A/B
6  종료 시 **destroy**(stop 아님) + 영속 확인
```

⛔ 3단계에서 반드시 멈춘다 — G4 가 어긋나면 base 는 로컬에 남기고 **A/B 만** 클라우드에서 돈다.
