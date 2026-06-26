> ⚠️ **대체됨(2026-06-16)**: 이 v3 요청(ISO=ON half)은 **`COWORKER_REQUEST_2026_06_16_v4_factorial.md`로 대체**(v4 통일코퍼스·harness v4·순수-synth factorial 전체). 이 파일은 이력용.

# Coworker 요청 (2026-06-16) — M-σ v3 전이 factorial: ISO=ON 절반 4 arm 실행

> 자기완결 요청서. 직전 요청서(`COWORKER_REQUEST_2026_06_16_scale_floor.md`·floor/scale)는 **완료·무관**. 이것만 보면 됨.
> 권위 컨텍스트: `scripts/distill/ma/M_SIGMA_V3_TRANSFER_FACTORIAL_DESIGN.md`(설계 권위)·`M_A_RESULTS.md §10-11`(M-σ in-dist 양성·M-D 전이 음성).
> 불변(必): [[feedback-thesis-tbox-transfer-direction]](τ²는 *전이 타깃*·**τ²로 학습/튜닝 절대 금지**)·[[feedback-selector-verifier-deterministic]](resolver·selector=결정론)·[[feedback-nl-formalize-llm-selection-deterministic]].

## 0. TL;DR — 무엇을 / 왜
**전이를 *한 레시피*가 아니라 *비교 실험군*으로 측정한다.** 전이 구동 후보 3축 **{ISO 등방화·NL grounding·PROV provenance}**를 직교 축으로 두고, *순수 추상* selection substrate 위 **2³ 완전요인**(8 arm)으로 각 축의 *단독 전이* + *조합 전이*를 held-out τ²로 측정.
- **나눔**: 내가 공유 인프라(합성 생성기·resolver·하니스·batch 스크립트)를 **단일 소스로 커밋**·**ISO=OFF 절반 4 arm** 실행. **coworker = ISO=ON 절반 4 arm**(아래 §3). 합쳐서 2³ 완성·main effect/interaction 집계.
- **왜 분담이 까다로운가**: factorial 타당성 = **모든 arm이 축 플래그 외 전부 동일**(#예제·step·LR·rank·난이도분포·eval셋·seed). ⇒ 코드/하이퍼는 **내가 커밋한 `ma_factorial_batch.sh`에 고정(frozen)**. coworker는 **arm 플래그만** 바꿔 실행·**스크립트/하이퍼 편집 금지**.

## 1. 설계 한 장 (맥락)
M-σ in-dist는 양성이었다(base 0%→**96%** $ref-correct·derivation-레벨 학습 *가능*). 하지만 그 데이터(cfb-threading)를 held-out τ²에 전이하니 **음성**(all-arg 0.41→0.03·over-$ref로 base를 *망침*). 음성 3원인 = ①selection-by-criteria가 orphan(cfb=threading뿐) ②over-$ref(order_id 리터럴까지 $ref) ③payment harness 아티팩트.
⇒ v3 = **순수 추상 selection-by-criteria** 합성으로 ①을 정면 합성·**provenance 타입**(literal/$ref/$select)으로 ②교정·하니스 ③수정. 그리고 "무엇이 전이를 만드나"를 한 축에 베팅 않고 factorial로 *비교*.

## 2. 전이 구동 후보 = 직교 축 (factorial 축)
| 축 | OFF | ON | 가설 |
|---|---|---|---|
| **ISO**(등방화) | 고정 스키마명/값 | 예제마다 랜덤 스키마/도구명/필드명/값 | 표면군 저차원 불변량 강제 → 과적합 차단 |
| **NL**(grounding) | literal `attr=val` | 자연어 패러프레이즈("X 바꾸고 나머지 유지·없으면 Y") | 구조-prose→criteria 파싱 학습 |
| **PROV**(provenance) | $select-only | literal/$ref/$select 혼합 | over-$ref 교정 |

substrate(추상 selection)·resolver(결정론 offload)·학습량은 **전 arm 고정**.

## 3. ★coworker 실행 = ISO=ON 절반 4 arm
| arm id | ISO | NL | PROV | (셀코드) |
|---|---|---|---|---|
| **A-iso** | ON | OFF | OFF | 100 |
| **C-in** | ON | ON | OFF | 110 |
| **C-ip** | ON | OFF | ON | 101 |
| **FULL** | ON | ON | ON | 111 |

(나=ISO=OFF 절반: M0/000·A-nl/010·A-prov/001·C-np/011 + 참조 R0 base·R1 cfb-Mσ.)

각 arm = **합성 생성(축 플래그) → 7B LoRA SFT → held-out τ² 전이 eval**. 전부 `ma_factorial_batch.sh` 한 줄로 캡슐화됨(아래).

## 4. 정확한 실행 (★`synth_selection.py`+하니스 커밋 후 — 의존)
> ⚠️ **선행 의존**: 공유 인프라(`synth_selection.py`·`ma_resolver.py` $select 확장·`m_sigma_transfer_eval.py` 하니스 수정·`ma_factorial_batch.sh`)를 **내가 먼저 커밋**한다. coworker는 **`git pull` 후 `ls scripts/distill/ma/ma_factorial_batch.sh` 존재 확인**하고 시작. (없으면 아직 안 올라온 것 — 핑 주면 알림.)

```bash
cd <REPO> && git pull --ff-only
ls scripts/distill/ma/ma_factorial_batch.sh   # 없으면 대기

# ISO=ON 절반 4 arm (GPU/PORT는 빈 자원으로; arm마다 분리)
bash scripts/distill/ma/ma_factorial_batch.sh A-iso  <GPU> <PORT>
bash scripts/distill/ma/ma_factorial_batch.sh C-in   <GPU> <PORT>
bash scripts/distill/ma/ma_factorial_batch.sh C-ip   <GPU> <PORT>
bash scripts/distill/ma/ma_factorial_batch.sh FULL   <GPU> <PORT>
```
- `ma_factorial_batch.sh <arm> <gpu> <port>` = ① `synth_selection.py --iso/--nl/--prov`(arm 매핑·**시드 고정**) 합성 → round-trip 검증(100%만 통과) → ② 7B LoRA SFT(**frozen 레시피**=qwen7b_msigma와 동일 step/LR/rank) → ③ `m_sigma_transfer_eval.py`(held-out τ²·payment=값·n≥50) → ④ per-arg + over-$ref + 구조/어휘 autopsy 집계.
- **순차 실행 권장**(arm마다 GPU 1장·VRAM 점유). 4 arm 병렬 가능하면 GPU/PORT 전부 분리(§6 충돌주의).
- 예상: arm당 합성 빠름 + SFT ~1h + eval ~10분 → 4 arm 순차 ~4-5h·병렬이면 ~1.5h.

## 5. coworker가 답할 것 (집계는 내가)
- 4 arm 각각의 τ² **new_item_ids selection 정확률**(primary)·**all-arg**·**over-$ref율**.
- 내 ISO=OFF 4 arm + 참조와 합쳐 **main effect**(ΔISO/ΔNL/ΔPROV) + **interaction**(FULL − [M0+ΣΔsingle]) 계산.
- 핵심 질문: ISO=ON 절반이 ISO=OFF 절반보다 전이↑면 **등방화가 구동축**(Olver 표면군 이론 실증·헤드라인).

## 6. 산출물 (git 회수) + 인프라 규율
- 각 arm 출력: `/scratch/.../msigma_v3_<arm>.jsonl`(per-case·literal/$ref/$select emit + resolve + gold 대비)·SFT 어댑터 경로·집계 로그(`=== FACTORIAL <arm> ===` 블록).
- **이 jsonl + 로그를 repo commit**(또는 경로 통보) → 내가 집계 표 작성(`M_A_RESULTS.md §12`).
- 인프라: **GPU별 분리**(GPU0:8021·GPU1:8022 등)·serve 전 해당 GPU kill·port/log 분리([[reference-remote-server-environment]]). `git pull --ff-only` 확인(fileMode false 설정됨).

## 7. ★하지 말 것 / 규율 (factorial 타당성 사활)
- **`ma_factorial_batch.sh`·`synth_selection.py`·SFT 하이퍼·하니스 편집 절대 금지** — arm 간 동일해야 factorial 유효(축 플래그만 변동). 바꾸고 싶으면 나에게 먼저.
- **τ²로 학습/튜닝 금지**·**도메인-fit 금지**: 학습은 *순수 추상 합성*만·τ²는 *전이 타깃*([[feedback-thesis-tbox-transfer-direction]]). 어댑터가 τ² 보면 실험 무효.
- **resolver=결정론 유지**: $select/$ref 해결은 코드(LLM 아님). concrete item_id는 학습타깃 아님([[feedback-nl-formalize-llm-selection-deterministic]]).
- **시드/하이퍼 변경 금지**·arm 매핑(§3) 그대로. over-$ref율·autopsy 라벨 꼭 같이 회수(원인분석용).
- 양자화/모델 변경 금지(7B base 동일).

## 8. 조율
- 선행 인프라 커밋되면 핑 → coworker 시작. 질문(예 GPU 자원·경로)·중간 결과는 git/통보로.
- 음성도 1급 결과(추상→실 전이한계 박제) — 깨끗이 측정만 하면 됨. cherry/튜닝 금지.
