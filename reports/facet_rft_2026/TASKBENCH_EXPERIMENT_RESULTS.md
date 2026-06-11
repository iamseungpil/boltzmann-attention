# TASKBENCH 실험 결과 (권위본) — Exp-A/C (FIELD_GAP §18.1, HANDOFF_2026_06_10)

> 시작 2026-06-10. SOPBench 결과 권위본(`SOPBENCH_EXPERIMENT_RESULTS.md`)의 TaskBench 대응물.
> 보고 규율: TaskBench LODO = **supporting 전이만**(moat-(3) 주장 금지, FIELD_GAP §17.9 리뷰7-1).
> 인프라/재현 = `HANDOFF_2026_06_10_taskbench_learning.md` §2·§5 + `scripts/distill/taskbench/`.

## 1. Full 3도메인 base baseline (Qwen2.5-7B-Instruct, prompted, 2026-06-10) ✅
inference: vllm 0.11.0, temp/top_p inference.py 기본값, multiworker 8. 커버리지 99.7%+ (HF 7527/7546·MM 5572/5584·daily 4313/4320; 잔여=영구 실패/기형 gold 제외).
metrics: `{domain}_evalfull_qwen7b/metrics/qwen7b.json` `overall_overall`.

| 도메인 | n(matched) | node-F1 | edge-F1 | t-F1(arg name) | v-F1(arg value) |
|---|---|---|---|---|---|
| data_huggingface | 7436 | 73.6 | **32.2** | 55.1 | 35.3 |
| data_multimedia | 5540 | 84.4 | **50.0** | 71.1 | 52.3 |
| data_dailylifeapis | 4313 | 90.8 | **68.1** | 86.5 | 58.3 |
| (참고 published gpt-4, MM) | — | 90.9 | 69.3 | 87.1 | 72.3 |

- **150-subset 실측(§17.8: MM 83.3/49.3)과 full(84.4/50.0) 일치** → subset 노이즈 우려 해소.
- **edge-F1이 전 도메인 진짜 headroom** (HF 32 ≪ MM 50 < daily 68). HF 최저 = human-verified 10.8%(GT 약함)·도구 다수와 정합 — LODO 시 도메인 가중 주의(§17.8 caveat).
- 재현 gotcha 추가분: ①daily는 `--dependency_type temporal`(tool_desc에 input-type 없음) ②기형 gold(tool_nodes가 dict/no-task) HF2·MM3건 skip ③모델 emit 기형 link(source/target 결손, daily 153줄)는 sanitize(매칭불가 link 제거, task는 유지) — `tb_build_eval.py`.

## 2. A-0 edge-miss 수동 감사 (zero-GPU, RFT 전 BLOCKING — FIELD_GAP §18.1) ✅ 2026-06-10
- 대상: MM full baseline, edge-miss 보유 1852/5540(33.4%) 중 seed=42 무작위 30케이스 전수 수동 판정. 추출기 `tb_a0_audit.py`(evaluate.py 링크재구성 복제: `<node-j>` 태그·`_`→` `), 원본 `/home/woori/scratch/tb_a0_audit_mm.md`.
- **헤드라인: real-error ≈ 22/30 (73%) vs valid-대안/GT-관례 ≈ 8/30 (27%)**.
  - valid-대안 8 = 순서교환(번역↔요약), 의존성 값-인라이닝(`<node-j>` 대신 "positive" 직접 기입 = 의미 동일·표기 페널티), gold의 중복단계 생략(주어진 키워드에 Keyword Extractor 재적용 등), 양쪽 다 정당한 해석(stabilize 전/후 프레임 추출), gold 자체 결함(자기참조 인덱스로 gold link 소실, ≥2건).
- **real-error 22의 하위구조 (전부 학습가능 축)**:
  1. **`<node-j>` 인덱스 오류** (off-by-one·자기참조·dangling ref) ~7건(32%) — 형식/symbolic-binding, gold-SFT가 직접 교정하는 유형.
  2. **요청된 단계 누락** (downloader/search/voice-changer 등 명시 요구 드롭) ~10건.
  3. **도구명 환각/유사명** ("Text Extractor"→Image-to-Text, "Audio Extraction"→Video-to-Audio, "Text Article Spinner") ~3건 — 의미매칭/alias 축.
  4. 비정합 플랜 ~2건.
- **보상 설계 결론 (사전등록 이행)**: exact edge-F1 보상의 GT-관례 오염은 **실재하나 소수(~27%)** → headroom 20pt 중 ~3/4은 진짜 학습가능. ⇒ **outcome-RFT 보상 = exact node+edge-F1로 진행 가능**(차단 사유 아님), 단 ①관례-overfit caveat 보고 병기 ②달성천장 < 100(gold 관례 수준; gpt-4 MM 69가 그 반영) ③matching-F1 전환은 불요(P2: matching도 대안 credit 안 함 — 전환 이득 없음).
- 부수 발견: pred의 **값-인라이닝**(참조 대신 계산값 기입)은 의미상 옳아도 edge 미스로 잡힘 — RFT가 이를 "참조 표기 관례"로 교정하는 것은 정당(벤치 표기 규약 준수)하나, 이 부분은 capability가 아닌 관례 학습임을 보고 시 구분.

## 3. gold-SFT LODO ✅ lodo_mm 완료 (2026-06-10 18:02) — **학습 大·전이 0 분해 확정**

**lodo_mm (HF+daily 학습 → held-out MM)**, Qwen2.5-7B, LoRA r16 2ep:

| 도메인 | 역할 | base node/edge-F1 | gold-SFT node/edge-F1 | Δedge |
|---|---|---|---|---|
| data_huggingface | in-domain (sub500) | 73.6 / 32.2 (full) | 84.4 / **47.8** | **+15.6** |
| data_dailylifeapis | in-domain (sub500) | 90.8 / 68.1 (full) | 96.0 / **75.9** | **+7.8** |
| data_multimedia | **held-out (full 5548)** | 84.4 / 50.0 | 82.5 / **48.3** | **−1.7** |

- **판정: gold-SFT는 in-domain edge를 크게 올리나(+8~16) held-out 전이는 0(소폭 회귀)** = 배운 것이 도메인-특정 grounding. "학습 불가" 아님·"표현 불가" 아님 — **전이할 공통분이 이 방식으론 안 실림**. SOPBench adapter-only≈0(전이=scaffold가 전부)과 정합 = 2벤치 일관 → **"절차/구조는 weight가 아니라 구조-leg로 전이된다" thesis 보조 증거로 사용 가능.**
- in-domain 비교의 sub500-vs-full caveat(±수 pt)는 +15.6/+7.8 크기상 결론 불변. held-out은 full-vs-full 동일조건.
- **다음 분기 (Q2, coworker 위임 = `COWORKER_REQUEST_TB_SCALE.md` P1)**: Qwen2.5-32B 동일-레시피가 held-out을 +로 뒤집나 → 용량-바운드(스케일 투자) vs 방식-바운드(alias-마스킹/RFT 피벗).
- (보관) 원계획 명세: train = HF 2194(single400/chain1000/dag795) + daily 1675(400/1000/275) = 3869(3792/77 val, 0 overlong), 프롬프트=inference.py 원형(`tb_build_sft.py`), 어댑터 `sft_runs/qwen7b_tb_lodo_mm`. **보고 = supporting 전이**(moat-(3) 금지). alias arm 후속(P3).

### ★LODO 3×3 매트릭스 완성 (2026-06-10 20:37, 3회전 전부) — **in-domain +8~18 vs held-out 평균 −2.5**

edge-F1 (base → gold-SFT, Δ). held-out=full, in-domain=sub500:

| 회전 (학습 → held-out) | held-out Δedge | in-domain Δedge |
|---|---|---|
| lodo_mm (HF+daily → **MM**) | 50.0→48.3 (**−1.7**) | HF +15.6 · daily +7.8 |
| lodo_hf (MM+daily → **HF**) | 32.2→35.0 (**+2.8**) | MM +17.7 · daily +13.6 |
| lodo_daily (HF+MM → **daily**) | 68.1→59.6 (**−8.5**) | HF +17.0 · MM +15.1 |
| **평균** | **−2.5** | **+14.5** |

- **종합 판정: gold-SFT in-domain 학습은 견고(+8~18 edge, 전 회전·전 도메인) — held-out 전이는 0~음(평균 −2.5).** "학습 불가" 아님, "표현 불가" 아님 — **배운 것이 도메인-특정 + 형식-특정.**
- **★형식-간섭 패턴 (회전 간 비대칭의 설명)** ⚠️**§3.6 전수 census로 메커니즘 정정됨(필독)** — "출력-관례(문법) 전환 실패"는 **반증**, 실제 = ①스키마-경계 직렬화 사고(15% 샘플 손실) + ②미훈련 필드(task_links) 내용 열화: held-out 피해가 가장 큰 곳 = daily(**−8.5**) = 유일한 **temporal 형식**(다른 출력 스키마: name/value args+task_links). lodo_daily는 resource 2도메인만 학습 → temporal 형식이 학습에 없던 유일한 회전. 반면 held-out이 resource이고 학습에 temporal+resource가 섞인 두 회전은 ±소폭(−1.7/+2.8). ⇒ 전이 신호는 "held-out *형식*이 학습에 표현됐는가"와 상관 — 단 그 간섭의 *작동 방식*은 §3.6 census가 확정.
- lodo_hf의 +2.8은 base edge가 최저(32.2, headroom 최대)인 도메인에서의 미세 양(+) — in-domain(+14~18) 대비 한 자릿수 분율이라 "전이≈0" 결론 불변.
- **Q2(coworker P1, Qwen2.5-32B)에의 함의**: 용량-바운드 가설이 약화됨(형식-간섭은 용량으로 안 풀림) — 단 32B 측정은 여전히 가치(간섭 자체가 용량↑로 줄 수 있음). **Track A 다음 수 우선순위 갱신: ①alias-마스킹보다 형식-혼합/형식-불변 타깃이 먼저**(temporal/resource 둘 다 학습 포함은 기본, 출력-관례 분리) **②RFT(in-domain 동일도메인 보상이라 간섭 무관)** ③alias(이름암기 통제는 여전히 P3 위생).
- 어댑터: `sft_runs/qwen7b_tb_{lodo_mm,lodo_hf,lodo_daily}`. 평가 dir: `{dom}_evalfull_tb_<name>`·`{dom}_sub500_eval_tb_<name>`.
- 레시피: **gold-SFT**(teacher 호출 0 — GT-generator(GPT-4) 순환 caveat 원천 회피, §17.9 리뷰7-2 명명 준수). train = HF 2194(single400/chain1000/dag795) + daily 1675(400/1000/275) = 3869(3792 train/77 val, 0 overlong), held-out = MM 전체.
- 프롬프트 = inference.py 원형 복제(`tb_build_sft.py`), target = gold graph JSON. LoRA r16, 2ep, seqlen 6144, GPU1. 어댑터 `sft_runs/qwen7b_tb_lodo_mm`.
- 평가 예정: held-out MM full + in-domain 500-sub sanity (`tb_eval_adapter.sh`). 지표 = edge-F1 중심 + type 층화. **보고 = supporting 전이**. alias-마스킹 arm은 후속(P3).

### §3.6 ★형식-간섭 전수 census (2026-06-10 밤, `tb_format_census.py`, zero-GPU) — 가설을 로그로 검증: **문법-붕괴 반증, 실 메커니즘 2개 확정**
> 동기 = "점수 패턴 추론으로 두지 말고 prediction 로그 전수 조사"(사용자, 메타규칙 '강한 주장은 reliable test 후 박제' 정합). 대상 = 3회전 held-out full 예측 전부(base `qwen7b.json` vs SFT `tb_lodo_*.json`, 도메인당 3.6k–7.4k).

**판정① — "출력-관례(문법) 전환 실패" = 반증.** lodo_daily SFT의 daily 출력(파싱 성공분)은 temporal 관례를 거의 완벽 유지: dict-args **99.3%**(base 99.6%)·`<node-j>` 태그 **0건**·gold-linked 예제서 task_links 누락 **0.8%**(24/3035). resource 회전들도 역방향 오염 0(dict-args 0%·task_links emit 0건). **모델은 프롬프트가 시키는 문법을 따랐다.**

**판정② — 실 메커니즘 A = 스키마-경계 직렬화 사고 → 15% 샘플 손실 (lodo_daily 전용).** SFT pred n=**3653 vs base 4313 = 666 예제(15.4%) 통째 누락**(base는 7건). 로그 전수: Failed 667, 실패 시그니처 = **task_nodes 직후 JSON 객체를 조기 폐쇄 → 그 *바깥에* `, "task_links": [...]}`를 덧붙임** → ContentFormatError(재시도 소진→드롭). 원인 = resource 학습 타깃에 task_links 필드가 **아예 없어서**("객체는 task_nodes서 끝난다"를 학습) temporal 프롬프트의 task_links 요구와 충돌. 누락 구성 = single 640/chain 24/dag 2.
- ⚠️**보고치 과소피해 caveat**: eval은 pred-id 조인이라 누락 666은 **제외**되고 채점됨 → held-out 59.6은 생존 서브셋 기준 = **실제 전이 피해는 −8.5보다 큼**(특히 single 다수 누락 = node-F1도 영향).

**판정③ — 실 메커니즘 B = 미훈련 필드의 내용 열화 (파싱 성공분에서도 −13pt).** 같은 예제들에서 edge micro **P 0.749→0.619 · R 0.703→0.573**: 링크를 올바른 *문법*으로 쓰지만 *내용*(source→target 선택)이 나빠짐. 학습 2도메인(resource) 타깃에 task_links가 없어 링크-생성 스킬이 한 번도 훈련되지 않은 채 열화. resource 회전들은 내용-수준 변화만: lodo_mm P −7.6(과예측)·lodo_hf R +6.6/P −3.8(엣지 과생성 8776→11415 = +2.8과 정합).

**⇒ 처방 정밀화 (Track A 우선순위 갱신의 구체화)**:
1. **최소 수정이 먼저**: "형식-불변 타깃 설계"의 최저비용 형태 = **resource SFT 타깃에도 `task_links` 필드 항상 포함**(gold `sampled_links` 이미 존재, `tb_build_sft.py` 1줄) = uniform output schema. 메커니즘 A(직렬화 사고)는 이것만으로 제거 예상; B(내용 열화)도 링크-생성이 전 도메인서 훈련되므로 직접 완화.
2. **Q2(32B) 해석 프로토콜 재정련**: A(사고)는 용량↑로 줄 수 있으나(instruction-following 보존) B(내용)는 데이터 구성 문제라 용량과 직교 추정 → 32B 결과는 **이 census 스크립트로 A/B 분리 측정** 후 해석(점수만 보면 또 conflate).
3. 보고 규율: lodo_daily 행에 누락-제외 caveat 병기(n=3653/4318).

## 4. Exp-C scale 곡선 ✅ 2026-06-10 (500-sub/도메인; 7B행만 full)
| 크기 | HF n/e-F1 | MM n/e-F1 | daily n/e-F1 |
|---|---|---|---|
| 0.5B | 10.5 / 0.0 | 17.0 / 0.0 | 5.1 / 0.2 |
| 1.5B | 50.3 / 2.3 | 55.9 / 3.0 | 54.5 / 18.6 |
| 3B | 64.4 / 19.1 | 72.6 / 27.6 | 70.5 / 33.9 |
| **7B (full-domain)** | 73.6 / 32.2 | 84.4 / 50.0 | 90.8 / 68.1 |
| 14B | 77.5 / 39.6 | 89.5 / 52.8 | 93.5 / 77.4 |
| **32B ★Track-B (P0a, H100 노드, 2026-06-11)** | 79.7 / **43.9** | 87.3 / **61.9** | 94.9 / **80.6** |

- **edge-구조 emerge 지점 = 1.5B→3B**(전 도메인 edge 0–18→19–34), 이후 **14B까지 가파른 비포화 상승**; node는 7B 이후 수확체감(+~4pt). ⇒ "node~포화·edge=변별 스킬" (§1·§17.8) 의 scale-축 확증. 0.5B는 과제 수행 불능(포맷 붕괴 수준).
- 14B조차 published gpt-4(MM e-F1 69.3)에 16pt 미달 → 7B 학습으로 edge를 끌어올리는 Exp-A의 가치 공간 확인.
- caveat: 0.5–14B는 500-sub 단일run(7B만 full); 모델별 동일 sub500이므로 곡선 내 비교는 공정, 절대값은 full 대비 ±수 pt 가능.

## 5. ★전이-vs-용량 (lodo_mm 프로토콜, held-out MM, **sub500 동일-id 정합 비교**) ✅ 2026-06-11 AM
| scale | base edge | gold-SFT edge | **Δ held-out** | in-domain Δ (HF/daily) |
|---|---|---|---|---|
| 1.5B | 3.0 | 11.7 | **+8.7** | +37.0 / +50.7 |
| 7B | 48.3 | 47.5 | **−0.8** | +15.6 / +7.8 |
| 14B | 52.8 | 57.1 | **+4.3** | +11.9 / +2.2 |

- **★U자형 비단조**: 1.5B 양(+8.7) → 7B ~0(−0.8) → 14B 양(+4.3). 해석: ①1.5B는 base가 형식조차 못 함 → SFT가 *형식 스킬*(도메인-일반)을 가르쳐 전이 양 ②7B는 형식 기보유 → 도메인-특정만 학습 → 0 ③**14B서 +4.3 재상승 = 용량-바운드 가설 부분 부활**(형식 너머의 전이가능 구조 규칙성 흡수 시작?) → **coworker P1(32B)이 진짜 결정적**(+가 커지면 용량-스케일 투자 정당화).
- 측정 공정성: 전 행 동일 첫-500 id(`*_sub500x_eval_*`), full-vs-sub 혼용 제거(7B full −1.7 → sub-정합 −0.8).

## 6. RFT round-1 (RAFT: K=8·reward 0.3node+0.7edge·keep≥0.8·warm-start) ✅ 2026-06-11 AM
- rollout: 3869 프롬프트, kept 2845(73.5%, HF 1391/daily 1454 균형), mean best reward 0.855. 학습: winners 2ep lr5e-5, `qwen7b_tb_rft_mm`.
- **결과 (vs lodo_mm SFT)**: in-domain **daily 75.9→85.2(+9.3, SFT 너머)** · HF 47.8→46.3(−1.5) · held-out MM(sub500x) 47.5→48.4(+0.9, base 48.3 회복 수준).
- 판정: **outcome-RFT는 보상이 깨끗한 도메인(daily)에서 SFT 천장을 추가로 밀고**, held-out 회귀를 base 수준으로 복원. HF 정체는 보상-노이즈(관례-mismatch 분율, A-0의 27%) 가설 — round-2는 ①HF-전용 round or ②min_reward 상향/도메인별 임계 검토. 전이는 RFT로도 발생 안 함(예상 내, in-domain 레버).

### RFT round-2 ✅ 2026-06-11 13:08 (보상 v2 = +recall 0.25·+validity 0.10, r1에서 warm-start)
| 도메인 | SFT | r1 | **r2** |
|---|---|---|---|
| held-out MM (full) | 48.3 | 49.6 | 49.0 (≈불변) |
| in-domain HF (sub500) | 47.8 | 46.3 | **51.6 (+5.3 vs r1)** |
| in-domain daily (sub500) | 75.9 | 85.2 | 85.0 (유지) |

- **★r1의 HF 정체 해소** — census 귀속: HF node P +1.5pp/R +1.6pp(도구 *선택* 개선) + edge macro +2.1pp; daily는 r1 이득 유지. **v2 보상의 in-domain 효과 확인.**
- **사전등록 판정 (둘 다 불발)**: ⓐheld-out valid_frac 0.952→0.951(어휘-간섭, 보상-side로 held-out 못 닿음 — rollout이 in-domain이므로 구조적 한계) ⓑ누락-길이축 불변(deficit +0.23/+0.20·short율 그대로 — recall 이득은 길이가 아니라 이름 정확도에서 옴). ⇒ **§9 분기 발동: 누락축→L2 DPO(.all 데이터 준비됨)·어휘축(held-out)→grounded-copy.** RFT 수확체감 가시화(r2 held-out·daily 평탄) — in-domain 레버로서의 RFT는 r1+r2로 대부분 수확된 것으로 판단.

### ★round-2 궤적 전수 정밀분석 (2026-06-11 PM — 위 판정의 3가지 정밀 수정, 버킷+예시 직독)
1. **HF +5.3의 분해 (정직 — 일부는 관례-수렴)**: SFT→r2 improved 버킷(47/487) 시그니처 = valid_frac 0.871→**0.974**(+10.3pp, validity 항 실작동) + n_nodes +0.17 + node_f1 0.678→0.889. 예시 직독: ⓐ"Table Classification"→"Tabular Classification" 류 **무효-이름 교정 = 진짜 개선** ⓑ"ASR→Audio Classification"·"Audio Emotion Analysis→Audio Classification" 류 = **gold 도구-선택 관례로의 수렴**(기능적으론 A안이 동등/우월할 수 있음 — A-0의 27% 관례축이 보상을 통해 학습됨). ⇒ +5.3 중 일부는 capability가 아니라 **GT-관례 적합** — 보고 시 분리 명기.
2. **held-out wash(421↔451)는 노이즈가 아니라 체계적 양방향 재추첨**: improved 버킷 = valid +16pp·nself 반감 / worsened 버킷 = valid **−13pp**·nself 0.18→**0.64** — 같은 크기·반대 방향·거의 같은 수. 즉 r2 정책은 held-out에서 케이스별 행동을 **재추첨**(불안정)하며 평균만 보존. ⇒ 보상-쉐이핑 추가로 held-out을 미는 것은 무망 — **held-out 처방은 decode/추론-side(grounded-copy·L3 게이트)여야 함**의 직접 증거.
3. **누락축 "무반응" 완화**: 집계 결손 불변이지만 improved 버킷 내에서는 플랜 길이 +0.17 — recall 항이 **일부(47/487) 케이스에선 작동**. L2 DPO는 이 부분 효과를 전 케이스로 확장하는 시도로 정당.
4. ⚠️측정 주의: r1/r2 rollout 보상 통계(mean_best·kept%)는 **보상 정의가 달라 round 간 직접 비교 불가** — 정책 개선의 증거는 eval 표만 사용.

## 7. Qwen3 곡선 (sub500; 8B만 full. non-thinking 고정 — family-불변성 체크) ✅ 완료 2026-06-11 PM
| 크기 | HF n/e | MM n/e | daily n/e |
|---|---|---|---|
| Qwen3-0.6B | 41.9 / 0.6 | 42.2 / 4.2 | 63.8 / 25.8 |
| Qwen3-1.7B | 62.7 / 9.2 | 71.8 / 8.9 | 72.2 / 37.8 |
| Qwen3-4B | 79.8 / 27.1 | 81.9 / 46.4 | 90.4 / 72.0 |
| **Qwen3-8B (full)** | 77.2 / 39.8 | 83.5 / 51.4 | 93.2 / 79.2 |
| Qwen3-14B | 80.6 / 42.2 | 87.2 / 59.1 | 95.0 / 79.9 |
| **Qwen3-32B ★Track-B (P0b, H100 노드, 2026-06-11, non-thinking)** | 81.2 / **45.6** | 87.1 / **58.7** | 94.9 / **79.8** |
- 동급 대비 Qwen3 ≥ Qwen2.5 경향(특히 daily edge: 0.6B가 이미 25.8 vs Qwen2.5-0.5B 0.2; 14B edge MM 59.1 vs 52.8) — **곡선 모양(edge 후발 emerge·비포화)은 family-불변** 1차 확인.
- **★Qwen3-4B ≈ Qwen2.5-7B 동급**(79.8/27.1·81.9/46.4·90.4/72.0 vs 73.6/32.2·84.4/50.0·90.8/68.1) = 이 과제에서 family 세대교체가 ~2x 파라미터 효율 — "{소형·저비용}" leg에 유리한 재료(같은 coverage를 절반 크기로).
- **Qwen3-8B(full-vs-full 직접 비교) > Qwen2.5-7B**: edge HF +7.6·daily +11.1·MM +1.4 — 세대 이득은 주로 **edge(구조) 축**에 실림. Qwen3-8B daily edge 79.2는 Qwen2.5-14B(77.4)도 추월. 14B 점(sub500 59.1 MM)과 함께 Qwen3 곡선도 비포화 — gpt-4(69.3 MM) 격차는 Qwen3-14B 기준 ~10pt로 축소.

## 8. ★궤적 전수조사 (2026-06-11, `tb_census.py` — §3/§5/§6의 해석 *정정*, 이 §이 권위)
6개 비교쌍 전수(per-id 시그니처: 파싱·도구명 유효율·`<node-j>` 태그/자기참조/dangling·per-id F1·temporal 형식 플래그) + worsened/improved 버킷 궤적 직독. 원본 `/home/woori/scratch/census_*.md`.

**★통일 기제 — gold-SFT의 held-out 효과 = 독립적 두 힘의 합:**
1. **(+) 참조-인덱싱 규율 전이 (resource 전용, 도메인-일반 — 실제로 weight로 전이됨)**: 태그 채택(1.5B base 0.47→SFT 1.44개/ex)·자기참조 제거(7B 0.218→0.038 = −83%·14B 0.478→**0.010**; 14B improved 버킷 nself 2.0→0.0, edge 0.16→0.85). 이득 크기 ∝ **base의 인덱스-오류율**.
2. **(−) 도구-어휘 간섭 (보편)**: 유효 도구명 비율 −4~−8pp(7B 0.987→0.946·14B 0.997→0.956·daily 0.978→0.900). 궤적 직독: "Text Paraphraser"→"Paraphrase"·"play_movie_by_title"→"watch_movie" 등 **무효/변형 도구명 침투**. worsened 버킷 공통 시그니처(valid_frac 0.99→0.78~0.83).

**이 두 힘이 §5 U-커브와 §3 daily 붕괴를 전부 설명:** 1.5B +8.7=태그채택≫간섭 · 7B −0.8=소폭 인덱스교정↔간섭 상쇄 · 14B +4.3=base 자기참조율이 7B의 2.2배라 교정이득>간섭 · **daily −8.5=temporal엔 태그-인덱싱 축이 없어(ntag=0) 이득 0, 간섭만 수령.**

**정정/철회 (박제):**
- ❌ §3 "형식-간섭(출력-관례)" **기각**: daily held-out의 task_links 형식 99.6%·args dict 형식 99.6% 무손상 — 깨진 건 형식이 아니라 **어휘**(valid −7.8pp·node-F1 −5.7pp).
- ❌ §5 "14B=용량-바운드 부활" **기제 정정**: 용량이 아니라 **base 인덱스-오류율의 함수**. ⇒ 32B(coworker P1) 예측은 "용량"이 아니라 **32B base의 self-ref율·valid_frac census를 먼저 재서** Δ≈(인덱스교정 이득)−(어휘간섭 ~5pp)로 사전 추정 가능.
- 🔄 "weight 전이 0" **정정**: 도메인-일반 *참조-인덱싱 규율*은 실제 전이됨 — 어휘-간섭에 가려졌을 뿐. 정확한 문구 = "net 전이 ≈ 0~소폭은 (+규율 전이)−(−어휘 간섭)의 상쇄."
- macro-micro 화해: daily SFT는 single의 가짜링크 제거(improved 886 중 single 506)=micro 무기여, chain 진짜링크 손실=micro 타격 → 공식(micro) −8.5와 census(macro) 양상 모순 없음.
- RFT 재확인: daily=chain task_links 진짜 구조개선(improved 66 vs worsened 16, 형식변화 0)·HF=wash(25 vs 29).

**(보강 2026-06-11, 리뷰 질문 "JSON 미리 닫음 포함?" 답) 두 가지 조기-종결 축의 위치:**
- **깨진-JSON형(파싱 불가)**: census **미포함이 맞으나 규모 0.2%** — inference.py의 reformat 루프가 수리하고, 영구 실패는 id 드롭(MM full: base 12·SFT 4·RFT 6건/5584; sub500엔 0건) → 해석 불변.
- **누락형(valid JSON이나 플랜 조기 종결)**: census node_f1에 섞여 있던 것을 분해 — node P/R = base 0.863/0.776(recall-약세=누락 우세), **gold보다 짧은 예측 107/499(21%)·평균 결손 +0.26노드**, SFT 0.814/0.743(+0.23)·RFT 0.823/0.751(+0.24)·14B-SFT 0.883/0.800(+0.25) ⇒ **누락 축은 base부터 존재하고 SFT/RFT가 거의 못 건드림(Δ≈0)** = 두-힘 기제에 더해지는 **제3 불변 축**. 어제 A-0(base miss의 원인 분류: 누락 ~10/22)와 오늘 census(Δ 분석)가 다르게 보인 이유 = 누락은 Δ가 0이라 improved/worsened 버킷에 안 잡혔던 것 — 모순 아님, 이제 정량 분리됨. ⇒ 누락 축의 처방은 별도(coverage-지향 보상: 노드 recall 가점 or 길이-정규화) — RFT round-2 보상 설계에 ①어휘-유효성 페널티와 함께 ②node-recall 항 검토.

**처방 갱신 (1순위 레버 교체):** 어휘-간섭 억제가 본명 — ①**RFT round-2 보상에 도구명-유효성 페널티 추가**(구현 쉬움: valid_frac<1 감점) ②grounded-copy(도구명은 컨텍스트 tool list에서 복사 강제) ③alias-마스킹은 여전히 P3 위생(이름암기 통제)이나 간섭 직접 처방 아님.

## 8.5 ★Track-B (coworker, H100×4 노드) — P0 32B-class 완료 + **P1 step-0 census 사전등록** (2026-06-11)

**P0 진행**: Qwen2.5-32B ✅(§4 표에 행 추가) · Qwen3-32B ✅(§7 표에 행 추가) · Qwen2.5-72B 추론 중(TP4) · Qwen3-235B-A22B-INT4 대기. 전부 동일 첫-500 sub500, Qwen3는 non-thinking 고정(inference.py payload patch, Track A 동일 방법).

**★P1 step-0 (v3 필수 절차) — 32B base census 측정 + Δ 사전등록 (학습 착수 전 HF 박제, 2026-06-11T09:38Z)**:
- 32B base (MM sub500, n=496): **nself/ex = 0.000 · valid_frac = 1.000** (참조: 7B 0.218/0.987 · 14B 0.478/0.997 — base 인덱스-오류가 scale 비단조였는데 32B서 소멸).
- §8 기제 그대로 적용: 인덱스-교정 이득 ∝ base nself = **0** → **Δedge 사전예측 = 19.4×0.0 − 5.0 = −5.0pp** (어휘-간섭만 수령).
- **이 prereg가 §5-vs-§8을 가르는 판별 실험이 됨**: §5의 구판 "14B +4.3 = 용량-부활" 해석이 맞다면 Δ(32B) > +4.3로 더 커져야 하고, §8(인덱스-오류율의 함수)이 맞다면 Δ(32B) ≈ −5. 32B SFT(r16/a32/2ep/seq6144, 학습 중) 완료 후 실측 대조.
- 판정 규칙(사전등록): ①≈−5 적중 = 기제 확립 ②Δ > 0 = 인덱스 축 외 추가 전이 발견(용량 가설 부분 부활) ③−5보다 더 나쁨 = 32B 고유 변수 재조사.
- 부수: 32B base가 valid_frac 1.0 → **어휘-간섭도 base엔 없음** — SFT가 주입하는 순수 학습-부작용임을 32B가 가장 깨끗하게 보여줄 표본. P1 eval 후 §9.5 name-snap을 32B-SFT pred에 적용해 간섭 분리 확인 예정(zero-GPU).

**Qwen3-32B 관찰 (§7 곡선의 꼭대기)**: Qwen3는 **14B→32B가 사실상 평탄**(MM edge 59.1→58.7·daily 79.9→79.8, HF만 42.2→45.6) — Qwen2.5의 14B→32B(+9.1 MM)와 대조. 세대-이득(§7 "Qwen3-4B≈Qwen2.5-7B")이 32B-class에선 소멸: **Qwen2.5-32B(61.9) > Qwen3-32B(58.7) on MM edge**. 곡선-모양 family-불변 주장은 14B까지만 안전 — 32B-class 분기는 72B/235B 점이 더 말해줄 것.

**SOPBench Track-B #0 sanity (v1.42, 같은 노드)**: react/full/bank **44.78%** (리더보드 40.30 대비 **+4.5pp — ±2pp 재현 밴드 밖**, ⚠️serving 차이(vLLM 0.10.2/bf16/TP2) 추정, 원인 메모 후 4열표에선 우리 서빙 기준 내부-일관 비교로 사용) · fc/full/bank **12.69%** (32B FC base 앵커 신규, 7B 참조 3.7).

무효 도구명을 tool list 최근접 유효명으로 스냅(`tb_name_snap.py`, difflib cutoff 0.6) 후 공식 재채점:

| held-out pred | 원본 edge | +snap | Δ |
|---|---|---|---|
| **RFT2 (MM full)** | 49.0 | **52.5** | **+3.5 → base 50.0을 +2.5 추월 (첫 held-out 순이득)** |
| SFT (MM full) | 48.3 | 50.1 | +1.8 (어휘-간섭 손상 복구 = base 동급) |
| lodo_daily (daily full) | 59.6 | 59.6 | 0 (아래 경계) |
| base 통제 (MM) | 50.0 | 50.4 | +0.4 (통제 통과 — 스냅 인플레 없음) |

- **★의미: weight-학습(in-domain coverage) + 추론-side 결정론 보정(held-out 어휘)의 *패키지*가 처음으로 held-out 순이득** — thesis의 propose+결정론-보정 구조의 TaskBench 인스턴스. 스냅 규모: SFT preds 무효명 7.2%(912+138) vs base 1.7%(228+22) = census 간섭 발견의 독립 재확인.
- **★v0/v1 경계 실측 (daily가 그어줌)**: daily 미스냅 689건의 정체 = 오타가 아니라 **의미적 패러프레이즈**("install software"→`software_management`·"watch movie"→`play_movie_by_title`·"pay bill"→bill-payment류, top15 전수확인) → 문자열 매칭 사정거리 밖. **처방 v1 = 추론-시 제약 선택(constrained/guided decoding: valid 이름 집합 안에서만 생성 — 의미 매칭은 모델 자신이 수행)** — MM/HF의 형태-변형은 v0로 충분, daily의 의미-변형은 v1 필요.

## 9.6 L2 DPO (조기종결 쌍) ✅ 측정 완료 2026-06-11 PM — **누락축 첫 가동·but 단방향 overshoot로 패키지 무이득**
- 채굴: `.all`(K=8 전샘플)에서 [완전·고보상 chosen / 조기종결 rejected] = **318쌍**(3869 프롬프트 중 — 'no_short' 2528 = 정책이 in-domain 샘플링에선 조기종결을 드물게 냄 = 누락 질량은 greedy-선택/held-out 측이라는 진단과 정합). `tb_dpo_mine.py`.
- 학습: rft2 위 DPO(beta 0.1, lr 5e-6, 2ep). ⚠️인프라: 2×7B(policy+ref)가 48GB서 OOM → **dpo_train 4-fix**(단일 base+이중 어댑터·completion-span logits·`.train()` 필수·grad-ckpt)로 18.4GB 안정.
- **★판정 (held-out MM full, 어댑터 `qwen7b_tb_dpo_mm`)**:
  - **①누락축 = 첫 가동 성공**: short 18.3%→**9.4%**(1017→523)·deficit +0.225→**−0.072**·node-R 0.829→**0.865**(+3.6pp) — SFT/RFT/스케일이 전부 Δ≈0이던 제3축을 DPO가 처음 움직임 (`tb_pr_census.py`).
  - **②but 패키지 무이득**: 공식 edge 49.0→**48.46**(−0.5)·+snap **50.65** vs rft2+snap 52.5(−1.9). node-P 0.872→0.851(−2.1pp).
  - **③census 귀속 (`census_rft2_to_dpo_mm.md`)**: improved 460=원래 짧고-깨진 케이스 정확 수리(edge 0.20→0.72·nself 0.50→0.16·valid 0.86→0.97) ↔ worsened 549=**이미 완전하던 플랜 과잉-연장**(n_nodes 2.88→3.85·valid 0.97→0.90·ndangle ×9·edge 0.86→0.28). 같은 81.8%.
- **★기제: 단방향 쌍의 거울상 trade** — rejected가 조기종결뿐이라 "길게"만 학습 → 완전 케이스에 무효명·junk 노드 추가. SOPBench Gate-B 교훈(DPO 쌍 **양방향** 카운트) 그대로 재현. ⇒ 처방 = **균형-쌍 v2**(`tb_dpo_mine.py --balance`: chosen=gold-길이 정확·고보상, rejected=조기종결 **및 과잉연장** 양쪽), 채굴 zero-GPU(.all 재사용).
- 사전등록 분기(§2 핸드오프)는 "결손↓=best-stack 확정"을 가정했으나 결손↓∧패키지↓ 동시 발생 = 분기 미커버 → 기록 후 최소-프로브(균형-쌍 v2) 선행, L3 게이트 이관은 v2 판정 후.
- **★in-domain 대폭 회귀 (sanity가 본판정 뒤집음)**: sub500 edge daily 84.97→**69.81(−15.2)**·HF 51.61→**47.29(−4.3)** = v1 DPO가 RFT in-domain 이득을 되감음. held-out 보다 in-domain 손상이 훨씬 큼 ⇒ **v1 단방향 DPO = net-negative 확정** (누락축 이동은 실재하나 길이-편향 전역 주입의 부수손상이 지배). ⚠️daily sub500은 PS argv 따옴표-절단으로 eval서 탈락 → `tb_dpo_daily_sub500.sh`로 보완 (재발방지: ssh_run stdin이 utf-8-sig+CRLF 정규화하도록 수정 — 따옴표 포함 명령은 stdin으로).
- **진행: 균형-쌍 v2 학습 중** (`dpo_balance.jsonl` 714쌍=short 313+long 401 — 과잉연장 질량이 rollout에 실재. chosen=gold-길이 정확. `qwen7b_tb_dpo2_mm`, GPU1, ~3h). **사전등록 판정**: ⓐin-domain sub500 회귀 소멸(daily ≥84·HF ≥51 수준 복원) ∧ ⓑheld-out short율<18.3%·deficit<+0.225 유지 ∧ ⓒ패키지(edge+snap)≥52.5 — ⓐ 실패 시 길이-DPO 레버 폐기→L3 게이트 이관.

## 9. ★실행 큐 (2026-06-11 AM 갱신 — census §8 처방 기준, 이 §이 TaskBench 실행 권위)

**진행 중:**
1. **RFT round-2** (GPU1, 체인: rollout→train→eval, ~15-16시 완료 예상): 보상 v2 = node-F1 0.10 + **recall 0.25**(누락축) + edge 0.55 + **validity 0.10**(어휘축). `--save_all`로 L2용 전샘플 동시 수집. **사전등록 판정**: held-out MM census에서 ⓐvalid_frac 회복(어휘) ⓑnode-recall 상승(누락)을 *각각* 측정 — 어느 항이 일했는지 census로 귀속.
2. Qwen3 4B/8B 보완 (GPU0, 다운로드→곡선 4B→8B full 자동 체인).

**round-2 결과 분기 (사전 명시):**
- ⓐⓑ 모두 + → round-3 여부 판단 or 측정 종료·보고 정리.
- 누락축 무반응 → **L2: 조기종결 DPO**(rollout `.all`에서 [완전=chosen, 조기종결=rejected] 채굴 — 데이터 이미 수집 중, `dpo_train.py` 재사용).
- 어휘축 무반응 → **grounded-copy**(도구명 컨텍스트-복사 강제; validity-보상의 강한 버전).

**대기 큐 (우선순위 순):**
3. coworker P0/P1 (`COWORKER_REQUEST_TB_SCALE.md` v2; ★P1 착수 전 **32B base census**(self-ref율·valid_frac) 선행 — §8 정정에 따라 Δ 사전예측 가능).
4. **L3 type-closure 게이트 통합**: probe 완료(탐지 47%/수리 1.7% → "탐지→flag/재샘플" 형태) — 통합 시 평가축 신설 필요(coverage@gate, abstain율 동시 보고). TaskBench를 propose-then-gate 패키지의 2번째 실증으로 격상하는 본명 작업.
5. alias-마스킹 arm (P3 위생 — 간섭 직접 처방 아님, §8 정정).
6. (조건부) 형식-혼합 재학습 — ~~형식-간섭~~ 철회됐으므로 우선순위 강등; 어휘-간섭 처방(2·grounded-copy)이 daily 회복을 못 하면만 재고.

**원칙 (승계)**: 한 번에 한 변수·census로 귀속·강한 주장은 궤적 전수 후 박제. zero-GPU 병렬(§18.2)은 전부 ✅(규제·bitter-lesson·shielding).
