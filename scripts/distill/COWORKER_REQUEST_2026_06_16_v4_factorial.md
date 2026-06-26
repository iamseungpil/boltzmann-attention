# Coworker 요청 (2026-06-16) — M-σ v4 순수-synth factorial (mechanism leg·exp0와 병행 독립)

> 자기완결. 직전 `COWORKER_REQUEST_2026_06_16_factorial_transfer.md`(v3·ISO=ON half)는 **이걸로 대체**(v4 통일코퍼스·harness v4). `COWORKER_REQUEST_2026_06_16_scale_floor.md`(floor)는 완료·무관.
> 권위: `scripts/distill/ma/M_SIGMA_V4_UNION_CORPUS_DESIGN.md`(§7 factorial)·`M_SIGMA_V3_TRANSFER_FACTORIAL_DESIGN.md`(2³ 상세)·`M_SIGMA_V4_SUBTRACT_MAP.md`(provenance 버킷).
> 불변(必): [[feedback-thesis-tbox-transfer-direction]](**τ²로 학습/튜닝 절대 금지**·τ²=전이 타깃)·[[feedback-selector-verifier-deterministic]]·[[feedback-nl-formalize-llm-selection-deterministic]](resolver=결정론·concrete는 학습타깃 아님).

## 0. TL;DR — 분담 (3 레인 병렬)
- **나**: exp0(cfb matched-pair·GPU0 가동중) + 공유 생성기 `synth_selection.py` 커밋 + **factorial ISO=OFF half {M0·A-nl·A-prov·C-np} (GPU1)** + 이후 union-ablation·bridge.
- **coworker(이 요청)**: **factorial ISO=ON half {A-iso·C-in·C-ip·FULL} (4 arm)** — 다른 GPU/노드. 합치면 2³ 완성.
- **왜 독립인가**: factorial은 순수 추상 substrate만 학습·held-out τ² eval. exp0(cfb)·union(실벤치)와 데이터·GPU 안 겹침. v4 §7 = "union 부호 무관·독립". GPU1(나)+coworker로 8 arm 반반 = 순차 대비 절반 시간.

## 1. 측정 (mechanism)
전이 구동 후보 3축 {**ISO** 등방화·**NL** grounding·**PROV** provenance}를 직교 축으로 2³ 완전요인 → **main effect**(각 축 *단독* 전이?) + **interaction**(*합쳐서* 시너지?). 순수 추상 selection-by-criteria substrate(도메인 0) 위. = "전이를 만드는 게 어느 축인가"를 비교군으로.

| 축 | OFF | ON |
|---|---|---|
| ISO | 고정 스키마명/값 | 예제마다 랜덤(속성명/값-vocab·관계 보존·per-primitive) |
| NL | literal `attr=val` | 자연어 패러프레이즈("X 바꾸고 유지·없으면 Y") |
| PROV | $select-only | literal/$ref/$select 혼합 |

## 2. ★실행 = 2³ 8 arm (synth_selection.py 커밋 후 — 의존)
> ⚠️ **선행 의존**: 공유 생성기 `synth_selection.py` + `ma_factorial_batch.sh`를 **내가 단일소스로 커밋**한다(harness v4·`ma_resolver.py` $select는 이미 커밋·검증됨). coworker는 **`git pull` 후 `ls scripts/distill/ma/ma_factorial_batch.sh` 확인** 후 시작. 없으면 대기·핑.

```bash
cd <REPO> && git pull --ff-only
ls scripts/distill/ma/ma_factorial_batch.sh   # 없으면 대기

# ISO=ON half 4 arm. GPU/PORT는 빈 자원·arm마다 분리. 순차 권장.
for ARM in A-iso FULL C-in C-ip ; do            # A-iso·FULL(헤드라인) 먼저, 조합 다음
  bash scripts/distill/ma/ma_factorial_batch.sh $ARM <GPU> <PORT>
done
```
- arm 매핑(전체 2³·**coworker=ISO=ON 4개**): M0=000·A-iso=**100**·A-nl=010·A-prov=001·C-in=**110**·C-ip=**101**·C-np=011·FULL=**111**. (나=ISO=OFF {M0·A-nl·A-prov·C-np} GPU1.)
- `ma_factorial_batch.sh <arm> <gpu> <port>` = synth 생성(arm→`--iso/--nl/--prov` 매핑·**시드 고정**·round-trip 검증) → 7B LoRA SFT(**frozen 레시피**·전 arm 동일) → **harness v4**(`m_sigma_transfer_eval_v4.py`·held-out τ²·per-provenance split) → 집계.
- 비용: arm당 합성 빠름 + SFT ~1h + eval ~10분. tier1 5 arm 순차 ~5h·병렬이면 ~1.5h. tier2 +3.

## 3. coworker가 답할 것 (집계는 내가)
- arm별 τ² **new_item_ids selection 정확률**(primary·SUBTRACT_MAP §2 $select 버킷)·all-arg·over-$ref율·$select autopsy(lexical vs structural).
- **main effect** ΔISO=mean(ON셀)−mean(OFF셀)(NL·PROV 동일)·**interaction** FULL−[M0+ΣΔsingle].
- 판독(v3 §7): FULL 양성∧단일 음성∧interaction 양 = **조합이 전이 구동** / 한 단일 양성 = **그 축 단독 구동**(예 ISO=Olver 표면군 실증) / 전 arm 음성 = 추상→실 갭(1급 음성).

## 4. ★하지 말 것 / 규율 (factorial 타당성 사활)
- **`ma_factorial_batch.sh`·`synth_selection.py`·SFT 하이퍼·harness 편집 절대 금지** — arm 간 동일해야 factorial 유효(축 플래그만 변동). 바꿀 일 있으면 나에게 먼저.
- **τ²로 학습/튜닝 금지**: 학습=*순수 추상 합성*만·τ²=전이 타깃. 어댑터가 τ² 보면 실험 무효([[feedback-thesis-tbox-transfer-direction]]).
- **resolver=결정론**($select/$ref 해결은 코드·`ma_resolver.py`/harness)·concrete item_id는 학습타깃 아님([[feedback-nl-formalize-llm-selection-deterministic]]).
- **시드/하이퍼/arm 매핑 고정**·over-$ref·autopsy 라벨 꼭 같이 회수(원인분석).
- 7B base 동일·양자화/모델 변경 금지.

## 5. 산출물 (git 회수) + 인프라
- arm별 `/scratch/.../factorial_<arm>.json`(harness v4 split 출력: per-arg·buckets·emit·over_ref·select_autopsy)·SFT 어댑터 경로·집계 로그(`=== V4 SPLIT [<arm>] ===`).
- repo commit 또는 경로 통보 → 내가 main effect/interaction 표 집계(`M_A_RESULTS.md §12`).
- **GPU별 분리**(GPU별 포트·serve 전 kill·log 분리·[[reference-remote-server-environment]])·`git pull --ff-only`(fileMode false). **내 exp0가 GPU0 점유 중** — coworker는 **다른 GPU/노드** 사용.

## 6. 범위 밖 (혼동 방지)
- **union-ablation**(실벤치 재추출+synth·§6)·**synth-FULL vs union-FULL 격상시험**(§7)은 **내가**(union 코퍼스 필요). coworker는 **순수-synth factorial만**.
- **2차-타깃 전이**(§8)는 딥리서치(`w3eqx44at`·진행중) 후보 검증 후 별도.
- 음성도 1급(추상→실 전이한계 박제). cherry/튜닝 금지·깨끗이 측정만.

## 7. 조율
- 선행(synth_selection.py + batch) 커밋되면 핑 → 시작. 질문(GPU 자원·경로)·중간 결과 git/통보.
