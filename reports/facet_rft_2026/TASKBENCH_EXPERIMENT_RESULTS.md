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

## 7. Qwen3 곡선 (sub500, non-thinking 고정 — family-불변성 체크) 🔄 부분완료 2026-06-11
| 크기 | HF n/e | MM n/e | daily n/e |
|---|---|---|---|
| Qwen3-0.6B | 41.9 / 0.6 | 42.2 / 4.2 | 63.8 / 25.8 |
| Qwen3-1.7B | 62.7 / 9.2 | 71.8 / 8.9 | 72.2 / 37.8 |
| Qwen3-4B | (다운로드 미완 SERVE_FAIL → 재실행 중) | | |
| Qwen3-8B | (동상, full 3도메인 예정) | | |
| Qwen3-14B | 80.6 / 42.2 | 87.2 / 59.1 | 95.0 / 79.9 |
- 동급 대비 Qwen3 ≥ Qwen2.5 경향(특히 daily edge: 0.6B가 이미 25.8 vs Qwen2.5-0.5B 0.2; 14B edge MM 59.1 vs 52.8) — **곡선 모양(edge 후발 emerge·비포화)은 family-불변** 1차 확인.

## 8. 다음
- LODO_mm eval → (edge-F1 lift 시) LODO_hf/LODO_daily 회전 → outcome-RFT(§2 결론 보상) → alias arm.
- zero-GPU 병렬(§18.2): 규제 1차원문 sourcing(사활)·bitter-lesson — 별도 세션/딥리서치.
