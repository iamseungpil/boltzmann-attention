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

## 3. gold-SFT LODO_mm (학습 중, 2026-06-10 ~12:10 KST 시작)
- 레시피: **gold-SFT**(teacher 호출 0 — GT-generator(GPT-4) 순환 caveat 원천 회피, §17.9 리뷰7-2 명명 준수). train = HF 2194(single400/chain1000/dag795) + daily 1675(400/1000/275) = 3869(3792 train/77 val, 0 overlong), held-out = MM 전체.
- 프롬프트 = inference.py 원형 복제(`tb_build_sft.py`), target = gold graph JSON. LoRA r16, 2ep, seqlen 6144, GPU1. 어댑터 `sft_runs/qwen7b_tb_lodo_mm`.
- 평가 예정: held-out MM full + in-domain 500-sub sanity (`tb_eval_adapter.sh`). 지표 = edge-F1 중심 + type 층화. **보고 = supporting 전이**. alias-마스킹 arm은 후속(P3).

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

## 5. 다음
- LODO_mm eval → (edge-F1 lift 시) LODO_hf/LODO_daily 회전 → outcome-RFT(§2 결론 보상) → alias arm.
- zero-GPU 병렬(§18.2): 규제 1차원문 sourcing(사활)·bitter-lesson — 별도 세션/딥리서치.
