# 클라우드 엔진 동시 사망 — 원인·오염·복구 (2026-09-11)

## 무슨 일이 있었나

```
20:13:19 KST  task_081 이 9141 에서 연결오류 0건으로 완주            ← 성한 마지막
20:16:50~51   클라우드의 vLLM 두 개가 동시에 죽음
20:16~20:24   068 091 092 102 가 죽은 엔진에 물려 감 (각 70~76 연결오류)
20:37         레인이 새로 발사되며 가드가 거부 — 잘못된 수치는 만들어지지 않았다
21:31         엔진 재기동 · 21:33 레인 넷 자동 복귀
```
공백 **약 1시간 15분**. 그동안 사내 `8141` 한 대만 돌았다.

## 주장

두 vLLM 프로세스가 **2026-09-11 11:16:50~51 UTC**(= 20:16:50~51 KST)에
**`CUDA error: unspecified launch failure`(`cudaErrorLaunchFailure`)** 로 동시에 죽었다.
**장치/드라이버 층의 고장**이다.

## 축자

`/root/logs/vllm_8143.died_1116.log:75119`
```
(EngineCore pid=11533) ERROR 09-11 11:16:50 [core.py:1351]
RuntimeError: CUDA driver error: unspecified launch failure
```
`/root/logs/vllm_8141.died_1116.log:68204`
```
(EngineCore pid=11605) ERROR 09-11 11:16:51 [core.py:1351]
torch.AcceleratorError: CUDA error: unspecified launch failure
```
죽기 5초 전까지 정상 처리 중이었다 —
`INFO 09-11 11:16:46 [loggers.py:310] Engine 000: Avg prompt throughput: 53.7 tokens/s`.
터진 자리는 평범한 `torch.zeros(...)` 다. 특정 커널의 버그가 아니라 **컨텍스트가 이미 죽어 있었다.**

## 배제한 것

| 후보 | 반증 |
|---|---|
| 메모리 부족 | `free -g` 총 1007 GB · 사용 170 · 가용 837 |
| 디스크 | `/` 39% (93 GB 여유) |
| 인스턴스 정지·재부팅 | `up 30 days`. 역터널 `rtunnel.sh`(pid 31900)·그 ssh(pid 60448)가 **끊기지 않고 생존** |
| vast 가 서비스 종료 | supervisor 의 `vllm` 서비스는 09-07부터 `STOPPED` — 우리 엔진은 수동 기동이라 관리 밖 |
| 우리 층·하네스 | 사내 8141 은 같은 시각 `fs_task_098` 완주 |
| user-sim(OpenRouter) | 오류가 `OpenAIException` = litellm 이 **OpenAI 호환 로컬 엔드포인트**를 부르는 이름 |

## 반증조건

호스트의 `dmesg`·`nvidia-smi -q -d ERROR` 에 11:16 UTC 부근 **Xid/ECC 기록이 없고** 컨테이너
재구성 기록만 있다면 이 진술은 틀렸다. 실제로 **같은 분에 `/.launch`·`/etc/forward_port` 가 다시
쓰였다**(`/`·`/etc` mtime `09-11 11:16`) — 원인인지 결과인지는 컨테이너 안에서 가릴 수 없다
(`dmesg` 가 비어 있다). vast 문의 문구: *"Instance 50125302, 2026-09-11 11:16 UTC, both GPUs threw
cudaErrorLaunchFailure simultaneously — any Xid / GPU reset / host event?"*

⚠ **이 호스트는 공유다.** 엔진 기동 전 `nvidia-smi` 에 우리 것이 아닌 프로세스 둘(757 MiB · 1043 MiB,
이름 `[Not Found]` = 다른 PID 네임스페이스)이 있었고 **MIG 는 꺼져 있다**. 옆 테넌트의 사고가 장치
전체의 컨텍스트를 날릴 수 있고, 두 GPU 가 동시에 간 모양과 맞는다.

## 오염 — 넷은 측정이 아니다

`068 091 092 102` 는 **16 sim 전부 `infrastructure_error`**. `out_lb` 에서 `*.infra_void.gz` 로 격리,
`sim_results` 에서 제거·푸시, 큐 뒤로 되돌림. 전수 감사 결과 **오염은 이 넷뿐**이고 손해 15 와 043 은
전부 `user_stop` 이라 **−15 판정은 유효하다.**

## 이제 걸려 있는 것

| 어디 | 무엇 | 하는 일 |
|---|---|---|
| 인스턴스 | `/root/engine_keeper.sh` (pid 파일 `engine_keeper.pid`) | 60초마다 두 포트의 모델 id 확인. **응답이 없고 그 포트의 vllm 프로세스도 없을 때만** 재기동(적재 중에는 안 건드린다). 죽은 로그는 `vllm_<port>.died_<ts>.log` 로 **보존**. 포트당 5회 상한 |
| 인스턴스 | `/root/rtunnel.sh` | 역터널 `while true` 재연결 (원래 있던 것) |
| `.153` | `x768/wait_engines_fs.sh` | 포트가 기대 모델을 서빙하면 레인 넷 자동 부착 |
| `.153` | `lane_lb.sh` 의 발사 가드 | 모델 id 가 다르면 **발사 거부** — 오늘 이 가드가 잘못된 수치를 막았다 |

## 접근 메모 (다음에 헤매지 않도록)

- vast **계정 SSH 키는 「새로 만드는 인스턴스」에만** 내려간다. 도는 인스턴스에는 안 들어간다.
- 이 PC(`C:\Users\승원\.ssh\id_ed25519` = `claude-ssh-woori`)는 인스턴스 생성 시점에 등록돼 있어
  **처음부터 들어갈 수 있었다.** `.153` 의 `id_vast`(`woori153-to-vast`)는 없었고, 2026-09-11 에
  `authorized_keys` 에 추가했다(두 줄).
- 접속: `ssh -i <키> -p 31463 root@104.37.174.34` (프록시 `ssh3.vast.ai:15303` 도 같은 키를 본다).
- ⚠ `authorized_keys` 에 `>>` 로 붙일 때 **원본에 개행이 없으면 두 키가 한 줄로 붙어 둘 다 깨진다.**
  오늘 실제로 그랬고 `sed` 로 갈랐다. 붙이기 전에 `tail -c1` 로 개행을 확인할 것.
- 엔진 로그는 `/root/vllm8141.log` 가 아니라 **`/root/logs/vllm_8141.log`** 다.
