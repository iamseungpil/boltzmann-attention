# TASKBENCH 실험 결과 (권위본) — Exp-A/C (FIELD_GAP §18.1, HANDOFF_2026_06_10)
> 📌 **구조 안내**: 모든 설계·실험 문서의 단일 마스터 = repo `scripts/distill/EXPERIMENT_DESIGN.md` (**§7 문서지도**에서 각 문서의 역할·상태 확인; 목표·순서 변경은 마스터 §0-§4에서만). 처음 읽는다면 마스터부터.

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

## 1.5 ★TaskBench 외부 수치 전수조사 (2026-06-12, 3-agent 적대조사 — 인용 ~150편 전수 스크리닝)
> 동기: "리더보드 69.3은 2023 수치 — 이후 팔로우업 전수 조사"(사용자). 결론 먼저: **벤치는 외부적으로 사실상 동결(dead)** — 단 내부-일관 비교·기제 발견은 유효, 외부 비교가능성만 사망.

**①벤치 생사**: 리더보드 2023-11 동결·maintainer 무응답(2024 결과요청 이슈 방치)·**공식 evaluation.py에 도메인-이름 뒤바뀜 버그 open(2025-07 이슈)**·커뮤니티는 후속 벤치(WorFBench/UltraTool/ToolHop 등)로 이동.

**②수치 보고 논문 = ~150편 중 단 8-9편** (나머지는 cite-only/파생 벤치; 스크리닝 음성 목록 박제됨). 전부 표:
| 라인 | 프로토콜 | 최고 edge (MM/HF/daily) | 비교가능성 |
|---|---|---|---|
| **원판 gpt-4 (2023)** | full·zero-shot | 69.27 / 54.70 / 80.53 | 기준 |
| GNN4Plan(NeurIPS'24)→GNNVerifier('26) 계보 | **500-test 재분할(≥2-task 필터)**+GPT-4-turbo/4o+학습된 GNN/검증루프 | **73.73 / 60.79 / 87.61** (GPT-4o+ReAct+verifier) | ✗ 재분할·추가 추론·학습 컴포넌트 — 본인들도 리더보드 추월 주장 안 함 |
| GTool('25) | 자체 분할(726/597/558)·in-domain 학습·Llama-2-7B | 68.92 / 54.03 / 83.75 | ✗ (7B로 gpt-4 급은 주목할 만 — 단 학습+자체분할) |
| GRAFT·DiG-Plan('26) | chain-only 선형화 EM / "TaskBench-23" 501 풀드·EdgeRec(recall-only) | 비표준 지표 | ✗✗ e-F1 매핑 불가 |
| ToLeaP('25, 41모델) | 원판 harness 재런 | 오픈 최고 link ~38(Qwen2.5-32B) | ⚠️**인용 금지**: "GPT-4o" 행=원판 gpt-4 수치의 도메인-전치 복사(공식 스크립트 버그 시그니처)·전모델 일률붕괴=파서깨짐 신호 |

**③판정**: ⓐ**엄밀 원판 프로토콜에서 2023 gpt-4 행을 넘은 보고 없음**(to our knowledge — ICPE'26 1편 페이월 미확인) ⓑ유일한 fresh frontier 점 = GNNVerifier의 GPT-4o Direct **64.36 MM(서브셋)** = 원판 gpt-4보다 *낮음* — **frontier가 이 벤치에서 2023 수준 정체** ⓒGemini/o1/DeepSeek/405B의 수치는 어디에도 없음(부재 자체가 발견).

**④인용 위생 (논문 작성 시 필수)**: gpt-4 "공식" 수치 **두 벌** 존재 — README(54.70/69.27/**80.53**) vs arXiv v4 표(55.73/69.29/**83.47**) — 출처 명시 必. 프로토콜 함정 6종(재분할/필터·백본 드리프트·지표 약화(EdgeRec≠e-F1)·파이프라인 재구성·in-domain 학습 vs prompting 비교·도메인 전치) 체크 후 인용.

**⑤우리 위치 재산정 (서브셋-차이 명시 전제)**: 72B base 63.5·32B base 61.9 ≈ GPT-4o 64.4(서브셋) — **오픈웨이트 대형 base = 현대 frontier 동급**; 7B best-stack 57.3도 사정거리. 보고 문구는 "frozen 2023 리더보드 대비"+"내부-일관 비교" 프레임 유지, GNNVerifier-계보와 비교 시 프로토콜 차이 병기.

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
| **72B ★Track-B (P0a, TP4, 2026-06-11)** | 80.6 / **45.8** | 88.7 / **63.5** | 95.4 / **83.1** |

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
| **Qwen3-235B-A22B-INT4 ★Track-B (P0b, MoE(A22B)·GPTQ-Int4·TP4, non-thinking)** | 82.8 / **45.6** | 86.7 / **56.4** | 95.8 / **80.5** |
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

**P0 ✅ 전체 완료 (4모델, 2026-06-11)**: Qwen2.5-{32B,72B}(§4 행) + Qwen3-{32B,235B-A22B-INT4}(§7 행). 전부 동일 첫-500 sub500, Qwen3는 non-thinking 고정(inference.py payload patch, Track A 동일 방법). **raw 궤적/metrics/prereg = `trackb_raw/` 커밋 (§3.5 이행)**.
- **Q1 판정 재료**: Qwen2.5 곡선은 72B에서도 오르나 **기울기 급감**(MM edge 61.9→63.5, 2.25× 파라미터에 +1.6) — gpt-4(69.3)에 72B로도 −5.8pt 미달 = prompted-만의 천장 시사. **Qwen3 곡선은 14B 이후 평탄~역행**(MM 59.1→58.7→**56.4**(235B-A22B); MoE·INT4 confound 병기) — **곡선 모양 family-불변 가설은 14B-이하에서만 성립**, 대형단은 family-의존.

**★P1 step-0 (v3 필수 절차) — 32B base census 측정 + Δ 사전등록 (학습 착수 전 HF 박제, 2026-06-11T09:38Z)**:
- 32B base (MM sub500, n=496): **nself/ex = 0.000 · valid_frac = 1.000** (참조: 7B 0.218/0.987 · 14B 0.478/0.997 — base 인덱스-오류가 scale 비단조였는데 32B서 소멸).
- §8 기제 그대로 적용: 인덱스-교정 이득 ∝ base nself = **0** → **Δedge 사전예측 = 19.4×0.0 − 5.0 = −5.0pp** (어휘-간섭만 수령).
- **이 prereg가 §5-vs-§8을 가르는 판별 실험이 됨**: §5의 구판 "14B +4.3 = 용량-부활" 해석이 맞다면 Δ(32B) > +4.3로 더 커져야 하고, §8(인덱스-오류율의 함수)이 맞다면 Δ(32B) ≈ −5. 32B SFT(r16/a32/2ep/seq6144, 학습 중) 완료 후 실측 대조.
- 판정 규칙(사전등록): ①≈−5 적중 = 기제 확립 ②Δ > 0 = 인덱스 축 외 추가 전이 발견(용량 가설 부분 부활) ③−5보다 더 나쁨 = 32B 고유 변수 재조사.
- 부수: 32B base가 valid_frac 1.0 → **어휘-간섭도 base엔 없음** — SFT가 주입하는 순수 학습-부작용임을 32B가 가장 깨끗하게 보여줄 표본. P1 eval 후 §9.5 name-snap을 32B-SFT pred에 적용해 간섭 분리 확인 예정(zero-GPU).

**★P1 판정 (2026-06-11 13:10 실측 — 사전등록 ① 적중, 기제 확립)**:

| 평가 | base-32B | 32B+gold-SFT | Δ | (7B 참조: base→SFT) |
|---|---|---|---|---|
| **held-out MM sub500 (동일-id)** | **61.9** | **56.5** | **−5.4 (예측 −5.0)** | 48.3→47.5 (−0.8) |
| in-domain HF sub500 | 43.9 | 49.0 | +5.1 | 32.2→47.8 (+15.6) |
| in-domain daily sub500 | 80.6 | 84.0 | +3.4 | 68.1→75.9 (+7.8) |
| held-out MM full (어댑터) | (base full 측정 중) | 58.8 | — | 50.0→48.3 |

- **Δ실측 −5.4 vs 사전예측 −5.0 (오차 0.4pp) = 판정① 기제 확립.** §8 두-힘 기제가 1.5B/7B/14B/32B 4점에서 함수형으로 확인됨: Δ ≈ 19.4×nself(base) − 5.0. **§5 구판 "14B +4.3=용량-부활" 최종 기각** — 32B에서 Δ가 +4.3보다 커지긴커녕 −5.4.
- **기제 시그니처 census 확인 (base vs SFT, MM sub500 n=498)**: valid_frac **1.000→0.952 (−4.8pp = 예측한 ~5pp 어휘-간섭 그대로**; 7B −4.1pp·14B −4.1pp와 동일 크기) · nself 0.0→0.018(교정할 인덱스 오류가 없으니 이득 0, SFT가 미세 자기참조를 오히려 주입).
- 뉘앙스(macro-micro 분해, §8 화해 패턴 재현): census per-id macro edge는 0.639→0.690(+5.1)인데 공식 micro link_binary_f1는 −5.4 — 손실이 링크-多 예제에 집중(n_nodes 2.65→2.53 플랜 단축 + 무효명이 링크 통째 kill). improved 69 vs worsened 52.
- in-domain 이득도 7B(+15.6/+7.8) 대비 1/3 수준(+5.1/+3.4) — base가 높을수록 gold-SFT 가치 자체가 줄어듦(§2-P1 "in-domain 평평=gold-SFT 무가치" 경계의 중간 지점).
- **함의: 스케일 투자로 held-out 전이는 안 열림(기제상 32B+는 잃기만 함) → Track A 처방(§8/§9.5) 정합 — 전이는 추론-side(grounded-copy/L3 게이트) 레버가 본명.** 32B-SFT pred에 name-snap 적용(zero-GPU)으로 −5.4 중 어휘분 복구 검증 예정.
- **★Track A 독립 재검증 (2026-06-12, trackb_raw 원본 궤적 → 우리 빌더+gold+census 재계산)**: base valid_frac **1.0000·nself 0.0000** 재현 ✓·SFT 0.9521/0.018 ✓·micro 61.89→56.47(−5.4) ✓·macro +6.0/improved 76·worsened 52 ✓ — 집계 아티팩트 아님 확정(`mm_sub500_verify_*`, `census_verify_32b.md`).
- **★32B 누락축 신규 측정 (`tb_pr_census`, Track A 추가)**: base-32B deficit **+0.024**·short 13.9% (7B base +0.256/21.7%) = **조기종결 축도 32B base서 거의 소멸** ⇒ 7B 처방(균형-DPO·L3 게이트)의 32B 기대이득 작음 — "32B+ 개선 레버" 질문의 정량 답: 잔여 headroom(61.9→gold)은 누락도 어휘도 아닌 **구조-선택(edge 조합)·gold-관례 정합 축**. 단 SFT가 deficit을 +0.152로 늘림(parsimony 과교정의 미세 거울상 — 7B DPO-v1 교훈의 약한 재현).

**Qwen3-32B 관찰 (§7 곡선의 꼭대기)**: Qwen3는 **14B→32B가 사실상 평탄**(MM edge 59.1→58.7·daily 79.9→79.8, HF만 42.2→45.6) — Qwen2.5의 14B→32B(+9.1 MM)와 대조. 세대-이득(§7 "Qwen3-4B≈Qwen2.5-7B")이 32B-class에선 소멸: **Qwen2.5-32B(61.9) > Qwen3-32B(58.7) on MM edge**. 곡선-모양 family-불변 주장은 14B까지만 안전 — 32B-class 분기는 72B/235B 점이 더 말해줄 것.

**SOPBench Track-B #0 sanity (v1.42, 같은 노드)**: react/full/bank **44.78%** (리더보드 40.30 대비 **+4.5pp — ±2pp 재현 밴드 밖**, ⚠️serving 차이(vLLM 0.10.2/bf16/TP2) 추정, 원인 메모 후 4열표에선 우리 서빙 기준 내부-일관 비교로 사용) · fc/full/bank **12.69%** (32B FC base 앵커 신규, 7B 참조 3.7).

## 8.6 ★Track-B 전수 궤적 census (Track A 분석, 2026-06-12 — trackb_raw 16 preds × 우리 빌더·gold·census 전수)
> 보고서 원본: 리모트 `census_tb_{mm_q25_vs_q3_32b, mm_32b_vs_72b, hf_base_vs_sft32b, daily_base_vs_sft32b, daily_32b_vs_235b}.md` + verify-dir 15개(`{dom}_verify_{model}`).

1. **무결성 — 15/15 공식수치 원본 재현**: 모델 5(q25-32B/72B·q3-32B/235B·SFT) × 도메인 3의 micro edge 전부 coworker 표와 일치 — Track-B 표는 전수 검증됨.
2. **누락축 × 스케일 × 도메인 (PR census 15셋)**: deficit — MM·daily는 전 대형모델 ≈0(조기종결 소멸) **but HF만 +0.09~+0.14 잔존(235B도 +0.135)** = 누락축은 도메인-의존(HF=GT-약함·최난 도메인). +**SFT가 전 도메인서 deficit 증가**(MM +0.024→+0.152·HF +0.119→+0.178) = parsimony 과교정의 공통 시그니처(7B DPO-v1 거울상의 약형).
3. **★Qwen3 평탄의 정체 (q25-32B→q3-32B, MM 궤적 직독)**: 시그니처 차이 미미(valid 0.994·nself 0.036) — worsened 45(chain 31) 원문 = **유효명 안에서의 구조-선택 차이**(노드 재배열·그럴듯한 추가 스텝 삽입(Video Downloader)·유효 도구 치환(Text Downloader→Keyword 직행)). macro +2.4 ↔ micro −3.2(손실이 링크-多 체인에 집중). ⇒ **family 격차 = L6(구성 구조) 축 — L1/L5 시그니처로 설명 안 됨** = guided/snap이 못 고치는 축(P2b-2 음성통제 설계 정합).
4. **32B→72B 스케일 step**: improved 68 vs worsened 27·macro +8.0·**P 0.873→0.908** — 스케일은 정밀도·macro를 계속 올리나 micro edge는 +1.6뿐(개선이 single/저링크 예제에 집중 = §4 "edge 비포화"의 미세구조).
5. **in-domain SFT @32B 분해**: HF +5.1 micro/+12.4 macro(improved 104 vs 25 = 진짜 구조 이득) — **★단 valid 0.995→0.935(−6.0pp): 어휘-간섭은 in-domain에서도 발생**(P2 함의: SFT+guided는 in-domain HF에도 +α 예측). daily +3.4(valid −0.8pp뿐).
6. **temporal 형식 @대형**: 32B/72B/235B 전부 links_ok 1.000·argdict 1.000 — 7B의 형식 사고(§3.6 직렬화/링크열화)는 대형 base에 전무 = 형식축도 base-결핍 함수.

무효 도구명을 tool list 최근접 유효명으로 스냅(`tb_name_snap.py`, difflib cutoff 0.6) 후 공식 재채점:

| held-out pred | 원본 edge | +snap | Δ |
|---|---|---|---|
| **RFT2 (MM full)** | 49.0 | **52.5** | **+3.5 → base 50.0을 +2.5 추월 (첫 held-out 순이득)** |
| SFT (MM full) | 48.3 | 50.1 | +1.8 (어휘-간섭 손상 복구 = base 동급) |
| lodo_daily (daily full) | 59.6 | 59.6 | 0 (아래 경계) |
| base 통제 (MM) | 50.0 | 50.4 | +0.4 (통제 통과 — 스냅 인플레 없음) |

- **★의미: weight-학습(in-domain coverage) + 추론-side 결정론 보정(held-out 어휘)의 *패키지*가 처음으로 held-out 순이득** — thesis의 propose+결정론-보정 구조의 TaskBench 인스턴스. 스냅 규모: SFT preds 무효명 7.2%(912+138) vs base 1.7%(228+22) = census 간섭 발견의 독립 재확인.
- **★v0/v1 경계 실측 (daily가 그어줌)**: daily 미스냅 689건의 정체 = 오타가 아니라 **의미적 패러프레이즈**("install software"→`software_management`·"watch movie"→`play_movie_by_title`·"pay bill"→bill-payment류, top15 전수확인) → 문자열 매칭 사정거리 밖. **처방 v1 = 추론-시 제약 선택(constrained/guided decoding: valid 이름 집합 안에서만 생성 — 의미 매칭은 모델 자신이 수행)** — MM/HF의 형태-변형은 v0로 충분, daily의 의미-변형은 v1 필요.

## 9.5b ★grounded-copy v1 (guided decoding, 추론-시 제약) ✅ 2026-06-11 PM — **daily 어휘-붕괴 완전 회복 (+8.0, snap-불가 축)**
vllm 0.11 `structured_outputs`(xgrammar, per-request)로 도구명 슬롯(task/links source·target)만 enum-제약. 구현: `tb_guided_{schema,patch}.py`+`tb_guided_daily.sh`(inference.py에 env-게이트 no-op 패치, A/B 동일 파이프). 설계·출처 = `TB_GROUNDED_COPY_V1_DESIGN.md`.

| lodo_daily, daily full (held-out) | edge | node | parse | valid_frac |
|---|---|---|---|---|
| unguided (§3) | 59.55 | 88.24 | 3651/4320 | 0.900 |
| **+guided v1** | **67.60 (+8.05)** | **90.90 (+2.65)** | **4314/4320** | **1.000** |
| base 참조 (§4) | 68.1 | 90.8 | — | 0.978 |

- **★의미: snap(v0)이 Δ0이던 의미-패러프레이즈 축이 제약-선택으로 완전 해소** — guided가 SFT의 daily 붕괴(−8.5, §3·§8 어휘-간섭)를 **base 수준(−0.5)까지 전량 회복**. daily는 태그-인덱싱 축이 없어(ntag=0) SFT 규율-이득 자체가 0인 도메인 → base 회복 = 이 처방의 이론적 상한 달성.
- **귀속 (census `census_guided_daily.md` + `tb_pr_census.py`)**: ①valid_frac 0.900→**1.000**(공유 id, 무효명 0/13k+) ②**parse 회복 663건**(문법이 valid JSON 보장 — 깨진-JSON 축도 동시 해소) ③공유-id macro edge 0.650→0.717·node 0.822→0.892, improved 505 vs worsened **77** ④P/R 동반 상승 0.835/0.813→0.912/0.891 = 정밀도 손실 없는 회복(DPO v1과 대조). 라이브 중간검증: 3035노드 무효 0건.
- **★thesis 격상**: propose(weight-학습 SFT) + 결정론-gate(디코딩 제약)의 **2번째 held-out 실증**(1번째=RFT2+snap 52.5) — §10.2 L4(하이브리드: 모델이 enum 안에서 의미 선택) 행의 직접 증거. 비용: 도메인당 schema 1개+grammar 컴파일 1회(캐시), 추론 오버헤드 체감 없음(4320건 ~50분, unguided와 동급).
- **★MM 합성 실측 (2026-06-11 밤, `tb_guided_mm_dpo2.sh`)**: dpo2+guided MM full edge **57.22**/node 87.37 = dpo2+snap 57.30과 동급(−0.08)·raw 55.95 대비 +1.27 ⇒ **guided = snap 완전 상위호환 확정** (MM 동급 + daily +8.0(snap 0) + parse 보장) — 패키지의 결정론-leg를 guided 하나로 통일 가능. **최종 best-stack = rft2+dpo2+guided = 57.2~57.3 (base 50.0 대비 +7.2~+7.3).**
- **★2×2 factorial 완결 (E1, 2026-06-12 02:39 — `tb_guided_base.sh`, 사전예측 50.4±0.3 적중)**: 동일 base·동일 데이터 — base 50.00 / **base+guided 50.13(+0.13)** / FT 55.95 / FT+guided **57.22(+1.27)** ⇒ **상호작용항 양수(CD 기여가 FT 위에서 ~10×)** = "CD는 FT-주입 간섭의 회복장치"의 factorial 증명 — §6.5 ToolDec-대조 마지막 셀 채움(선행은 전부 ±FT축 base 상이). node 84.27.
- **★프롬프트-절감×guided (2026-06-12 0시, `tb_guided_promptslim.sh`, MM sub500 held-out·dpo2+guided)**: 도구목록 desc 제거(=목록 51% 절감) 시 edge 56.19→53.12/53.35(2-arm 재현 ±0.2)·node 86.66→84.4 = **절감 51%의 가격 −2.8~−3.1 edge**. 해석: 집행(유효명)은 마스크가 공짜 대체하나 **의미 정보(desc)는 held-out서 ~3pt 값** — ToolDec-v1 "names-only 충분" 주장(v2서 철회)의 TaskBench 반례 데이터. in-domain(weight가 의미 보유)은 미실측 — 절감 폭이 더 클 것으로 예상(도구폭발 82K 페인포인트의 비용-leg 후보, CDP Score-Prune-Present와 상보).
- **★E8: HF held-out guided = guided 3회전 완성 (2026-06-12 야간)**: lodo_hf+guided HF full **37.83**/n-F1 74.65 (unguided 35.0 → **+2.8**, 예측 +1~3 ✓). guided held-out 효과 3회전 전부: **MM +1.27(FT 위)·daily +8.0·HF +2.8** — 효과 크기 ∝ 그 회전의 어휘-간섭량(daily 의미-변형 最大·HF 중간·MM 소).
- **★E4: promptslim in-domain (2026-06-12 야간) — "self-descriptiveness"가 진짜 변수**: dpo2+guided에서 desc 제거 시 — daily **−0.6**(85.04→84.42, 예측 ✓) but **HF −4.1**(54.40→50.34, 예측 ✗). 정정: 절감 비용의 변수는 in/held-out이 아니라 **도구명의 자기서술성**(HF 도구명=모델id로 불투명→desc가 정보 본체 / daily=자기서술 API명). 보너스: **guided는 in-domain에도 +0.3(HF)~+1.4(daily)**. ⇒ 비용-leg 처방: 이름이 자기서술적인 카탈로그(전형적 기업 API)에선 desc 제거 ~공짜, 불투명 이름(모델 허브류)은 desc 필수.
- **선행연구 (5-에이전트 전문 적대검증 완료 2026-06-11, `TB_GROUNDED_COPY_V1_DESIGN.md` §6·§6.5 권위)**: 기제(이름-enum 마스킹)는 GENRE(ICLR'21)→PICARD/Synchromesh→ToolDec(v3'24, 최근접)→FANTASE(EMNLP'24)→ToolGen(ICLR'25)+xgrammar/OpenAI 상품화로 **확립 — novelty 주장 금지**. ⚠️FT+CD *결합 자체도* 선행에 있음(ToolDec Table1 stacking·§5 "complementary" 자인·FANTASE SFT±SCD) — 미점유는 좁은 형태: ①**TaskBench 표준 프로토콜에 inference-time·training-free CD 첫 수치**(GRAFT=학습 토큰·DiG-Plan=diffusion 구분 명시) ②**same-base 통제 2×2+census 귀속+발생론**(선행은 전부 ±FT축 base 상이·기제 부재 — "complementary" 관찰을 정량 분해로 완성) ③제약 득실의 task-수준 조건 실증(GAD=KL·소규모뿐↔daily +8.0·worsened 2.1%).

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
- **★★균형-쌍 v2 = 합격 (2026-06-11 PM, 사전등록 3기준)** (`dpo_balance.jsonl` 714쌍=short 313+long 401, chosen=gold-길이 정확, rft2 위 DPO, `qwen7b_tb_dpo2_mm`):
  | 지표 | rft2 | dpo v1 | **dpo v2** | 기준 |
  |---|---|---|---|---|
  | held-out MM full edge | 49.0 | 48.5 | **55.95** | — |
  | +snap (패키지) | 52.5 | 50.65 | **57.30** | ⓒ≥52.5 **✓** |
  | in-domain HF sub500 | 51.61 | 47.29 | **54.10** | ⓐ≥51 **✓ (초과회복)** |
  | in-domain daily sub500 | 84.97 | 69.81 | **83.64** | ⓐ≥84 △(−0.36 소폭미달=사실상 복원) |
  | short율 / deficit | 18.3%/+0.225 | 9.4%/−0.072 | **16.0%/+0.181** | ⓑ<기준 ✓ |
  | node P / R | 0.872/0.829 | 0.851/0.865 | **0.905/0.870** | (P·R 동반상승) |
- **★census 귀속 (`census_rft2_to_dpo2_mm.md`)**: improved 570 vs worsened **216**(v1 460/549에서 역전). v1 overshoot 소멸(n_nodes 2.58→2.63 완만, v1은 2.88 폭주). **부수이득 2개**: valid_frac 0.951→**0.985**(무효명 스스로 감소 — snap 규모 738→278)·nself 0.138→**0.067**(자기참조 절반). 기제: chosen=gold-길이-정확·고보상 샘플로의 holistic shift가 길이만이 아니라 어휘·참조 청결까지 동반 학습.
- **★판정: held-out 신기록 — raw 55.95가 base(50.0)를 +6.0, 패키지 57.30이 종전 최고(52.5)를 +4.8 추월.** 채굴은 train-도메인 rollout만 사용(MM 무접촉)=누출 없음, 공식/macro/P-R 3계측 정합. **§10.4-②확정: 정책류(종결 캘리브레이션)는 weight로 학습 가능, 단 신호는 양방향 필수** — v1(단방향)=net음수 ↔ v2(양방향)=전축 동반 개선이 깨끗한 대조실험. best-stack = **rft2+dpo2(+snap or guided)** — 다음: dpo2+guided 합성(MM full) 측정 → 전 도메인 held-out 재측정 헤드라인.

## 8.7 edge-snap v0 = NULL (2026-06-12 0:42, 음성 결과 박제 — `tb_edge_snap.py`)
값-인라이닝→`<node-j>` 정준화(4중 보수 가드: 태그-보유 불간섭·단일-입력-슬롯만·요청문-등장 제외·생산자 유일)를 MM 5타깃에 적용: **재작성 7B 80/5572행·dpo2g 30·32B base 0·32B-SFT 1·72B 0** — Δedge +0.16(7B)/−0.06(dpo2g)/0/+0.1/0 = 노이즈 수준. skip-census: has_tag 62%(이미 참조 사용)·잔여는 소스-노드(요청문 인자) 또는 생산자 모호. **판정: A-0 관례분(~27%)의 지배 하위유형은 값-인라이닝이 아니라 순서-대안·중복단계 생략 = "다른 유효 플랜"이라 표기-정준화로 회수 불가(oracle 필요). 대형 모델은 인라이닝 자체를 안 함(0건).** ⇒ ⓐ축 레버 폐기, L6 잔여는 ②K+게이트 선별·③그래프-멤버십·④구조-DPO로.

## 8.8 ★K+게이트 선별 헤드룸 = 실재·대형 (E2, 2026-06-12 — `tb_kgate_select.py`, zero-GPU, 기존 rft2 rollout K=8 재사용)
| 선택 방식 | edge (in-domain 3869 프롬프트, census-식 링크 채점) |
|---|---|
| 단일샘플 평균 (1-shot 기대) | 0.698 |
| **gold-free 게이트 선별 v0** (파싱>valid>self/dangle>ntag) | 0.730 (+3.2) |
| **oracle best-of-8 (선별 상한)** | **0.870 (+17.2)** |
| (sanity: reward-pick≈oracle ✓) | 0.869 |
- **판정: L6 선별 레버(§10.5 R6) 실재** — 제안 분포에 +17.2pt(DiG-Plan Pass@K의 우리-데이터 재현), v0 게이트는 그중 18%만 회수 ⇒ **다음 정밀 타깃 = 게이트 스코어 강화(그래프-멤버십·타입-호환을 스코어러로 — 마스킹 아닌 선별이라 GAD 왜곡 무관)**. ⚠️caveat: in-domain(train 도메인) 프롬프트·census-식 채점(상대 갭만 신뢰·공식수치와 비교 금지); held-out 확장은 K-샘플 추론 필요(GPU).
- **★E5/E6 야간 확정 (2026-06-12) — 선별 레버의 결정적 한정**: ①E5 스코어러 v1(+그래프-멤버십): 회수 18%→**22.6%**(예측 ≥30% ✗ — 멤버십 기여 소폭) ②**E6 held-out (dpo2+guided 위, MM sub500, K=8 temp0.8): mean 70.6 / gate 71.1~71.4 / oracle 72.0 = 선별 갭 +1.4뿐**(예측 >+5 ✗). **해석(중요): in-domain rft2 rollout의 +17.2 갭은 대부분 "우리 레버가 이미 제거한 분산"** — guided=어휘 분산·DPO=길이/정책 분산 제거로 정책이 수렴해 K-샘플이 서로 닮음. ⇒ **레버는 비선형 합성: 선별은 best-stack 위에선 천장 +1.4** — 잔여 L6(57→관례천장 ~69)는 수렴 정책의 K-분포 밖 = ④구조-표적 DPO(weight) 또는 제안 다양화(고온/이종 제안기 — DiG-Plan의 diffusion 논거)가 필요. E6 분석버그 1건 수리(도메인 data.json은 task_nodes 비표준 → eval-dir gold로 조인).

## 8.9 ★P-D(-1) 이종-AR 풀 census (2026-06-12 — zero-GPU, 리뷰 권고 신설 단계; `TB_DIFFUSION_PROPOSER_DESIGN.md` §3 v2)
디스크의 기존 sub500 MM 예측만으로 "제안 이종성이 풀-oracle을 올리는가"를 diffusion 없이 분리 (E6 동일 채점·동일 id셋 499·gold=eval-dir 조인).
| 풀 | mean | gate_v0 | gate_v1 | oracle |
|---|---|---|---|---|
| AR8 (`tb_dpo2g_mmk0-7`, E6 통제 재현) | 0.706 | 0.711 | 0.714 | 0.720 |
| AR4 (k0-3 고정) | 0.707 | 0.712 | 0.712 | 0.718 |
| **AR8+H6** | 0.671 | 0.525 | 0.541 | **0.856** |
| AR4+H6 | 0.657 | 0.526 | 0.541 | 0.856 |
| H6 단독 | 0.624 | 0.523 | 0.540 | 0.847 |

H6 = {qwen3b 0.348, qwen14b 0.656, qwen3_4b 0.537, qwen3_14b 0.682, **tb_lodo_hf 0.757, tb_lodo_daily 0.745**} (solo mean; lodo 둘은 MM-포함 학습-혼합 상이 어댑터).
- **판정 ①: Δhetero = +13.6 (0.720→0.856) ≫ 사전등록 임계 +2** — E6의 "수렴 정책 K-분포 밖" 잔여 L6는 **이종-AR 제안만으로 대부분 도달 가능**. 검증 census(R8): H6-승리 112/499 중 empty-gold 24뿐(88=비자명 gold, mean Δ0.61) → f1(빈,빈)=1 인플레이션 가설 기각. CI 생략(Δ≫임계).
- **판정 ②(신규 병목): 혼합 풀에서 게이트 붕괴** — v0/v1이 0.71→0.52-0.54로 **mean보다도 낮아짐**(이종 후보 사이에서 역선택). oracle 헤드룸 +13.6이 있어도 현 스코어러로는 실현 0 이하. ⇒ **L6의 binding constraint = 제안 다양성(해결됨, 공짜)이 아니라 선별기**.
- **처방 재배열**: ⑴**선별기 연구가 1순위로 승격**(이종-풀 위 robust 스코어러 — 후보: 타입-호환·실행-가능성·pairwise 비교·verifier) ⑵P-D0/P-D1(diffusion)은 **조건부 강등**: Dream의 한계가치는 AR8+H6 위(0.856→cap)에서만 측정 의미 — "더 싼 대안 대비 순이득" 입증 부담(설계서 v2 사전등록대로). ⑶배포 비용: H6 = 같은 base의 LoRA 어댑터들 = vllm 멀티-LoRA 1서버로 서빙 가능 ≈ 추가비 ~0 (diffusion 2-모델 대비 압도).
- ⚠️척도 주의: census-식 링크 채점(상대 비교만, 공식수치 비교 금지)·empty-gold 259/499 포함(양 풀 동일 적용). 공식 edge-F1 실현이득은 선별기 개선 후 P-D2-형 측정에서.

### 8.9b ★선별기 1차 zero-cost 실측 (2026-06-12 — MBR/합의 계열, 전부 gold-free·학습0·결정론)
AR8+H6 (동일 id 499, census-edge): **mean 0.671 / gate_v1 0.541(역선택) / raw MBR(edge-F1 utility) 0.716 / 풀dedup MBR 0.711 / ★proposer-가중 MBR 0.751 / ★validity-필터+prop-MBR 0.753 / oracle 0.856** — 회수율 44%(동질 풀 v1의 22.6% 대비 ~2×), 기존 최고 실현치(동질 gate_v1 0.714)도 +3.9 추월.
- **기제 분해(3-변형 사다리)**: ①raw MBR=다수-블록 편향(동일정책 8표가 합의 지배: AR8+H6 0.716 < H6only 0.731) ②full dedup=과교정(**다중성=증거** 소거 — 서로 다른 모델의 일치가 신호인데 1표화: H6only 0.731→0.471 붕괴) ③**proposer-당 1표(상관 샘플=합산 1표, 이종 모델 일치=독립 증거)가 정답** — 0.751. MBR utility는 edge-F1(평가척도 동형)이 node-Jaccard보다 우월.
- **v0 특징의 재배치**: lexicographic *서열*로는 이종 풀에서 역선택(0.54), 후보 *필터*(invalid명·self/dangle 제거 후 MBR)로는 +0.2~1.4 한계 기여 = "검증 특징은 거부권, 선택권은 합의에" 분업.
- 동질 AR8에선 MBR=mean(0.706, 변별 불가) — E6 수렴 소견과 정합; 이 선별기는 이종-풀 전용 레버.
- 다음 = deep-research(이종-풀 선별 문헌) 합류 후 선별기 설계서(detail) — 잔여 후보: proposer 품질 prior(보정 필요)·pairwise 7B judge·실행-가능성 체크·공식 edge-F1 확정 측정.

## 8.10 D1 구조-표적 / D2 비용-표적 DPO (2026-06-12 day 배치 — 사전등록 판정)
배경 = §8.8 레버 비선형의 처방 ④. 쌍 채굴 `tb_dpo_mine.py` (structure 1017 / cost 376), base=rft2, v2 하이퍼.
- **★D2 (비용-표적) = 기각 (1급 음성)**: ⓐ평균 n_nodes 2.582→2.400 ✓(parsimony 신호 학습됨) BUT ⓑ**공식 edge MM full 49.0→47.0 (−2.0, 사전등록 ±1 밖)** ✗ · held-out HF 51.6→48.4(−3.2) · daily 85.0→84.6 · census node_f1 −2.3·improved 315 < worsened 478. **해석: edge-동률·여분-노드 rejected 쌍이 "노드 수 줄이기"를 가르치되 필요 구조까지 깎음 = v1 거울상(사전등록 감시 항목 그대로 발현) — 비용-leg은 DPO 쌍이 아니라 promptslim/추론-side로.**
- **D1 (구조-표적)**: 학습 완료(ep1 step120). ⚠️1차 평가 실패 = vllm serve가 adapter 저장과 동시 기동(10:51 레이스)→엔진 초기화 실패. 12:00 재평가 진행 중 — 수치는 도착 후 이 행에 추가.
- 인프라 사건 3건(핸드오프 박제 예정): ①day 배치 이중 기동(1차=야간 잔여 vllm OOM→재기동 래퍼가 2차 기동, 1차 시체가 빈 보고서 push) ②adapter-저장↔serve 레이스(드라이버에 저장 완료 후 sleep/검증 필요) ③eval 후 EngineCore 고아 42GB 잔존(tb_eval_adapter가 시작 시만 kill — 종료 후 정리 추가 필요).

## 9. ★실행 큐 (2026-06-12 0시 전면 갱신 — §9.6 v2 합격·§9.5b guided·§8.5 P1 적중 이후, 이 §이 TaskBench 실행 권위)

**✅ 이번 사이클 완료 (06-11~12)**: RFT r2 → DPO v1(net−) → **균형-DPO v2 합격(55.95/57.30)** → guided v1 daily(+8.0)·MM 합성(57.22=snap 상위호환) → promptslim(−51% 목록=−3 edge) → P1 32B prereg **적중**(−5.4 vs −5.0)+Track A 독립 재검증 → 선행연구 5-agent 적대검증·§6.5 차별점 경화. §10 분류의 판별실험 2개 모두 확정.

**대기 큐 (우선순위 순) — "더 할 것":**
1. **base+guided 1런** (MM full, GPU0 ~50분): §6.5 ToolDec-대조 2×2의 마지막 빈 셀 — related-work 방어 완결. 예측: base+snap +0.4와 동급.
2. **전 도메인 best-stack 헤드라인**: rft2+dpo2+guided를 HF held-out 트랙에도 — 필요물: lodo_hf 트랙 RFT rollout(.all)→균형쌍 채굴→DPO→guided. daily 트랙은 lodo_daily+guided(67.6) 완료, DPO만 잔여. 논문 헤드라인 표(3도메인 패키지)의 본체.
3. **coworker P2 제안 (32B 후속, `COWORKER_REQUEST_TB_SCALE.md`에 추가할 것)**: ①**32B-SFT+guided/snap**(간섭 −4.8pp 회복 검증 — §8.5 예고 항목, 예측 ≈base 복원) ②(선택) 32B 균형-DPO 재현 — 단 §8.5 누락축 측정상 32B 기대이득 작음(deficit +0.024) → 우선순위 낮음·기제 확인용.
4. **promptslim in-domain arm** (zero-GPU에 가까움, sub500 2런): in-domain서 desc 제거 비용 측정 — "weight가 의미를 알면 프롬프트 더 절감" 가설 완결(비용-leg).
5. **★이종-풀 robust 선별기 (§8.9로 1순위 승격)**: oracle 헤드룸 +13.6 실재·현 게이트는 역선택(0.52<mean) — 후보: 타입-호환/실행-가능성/pairwise/verifier 스코어러. P-D0/P-D1(diffusion)은 이것 이후 조건부. L3 type-closure 게이트(K=4 재샘플)는 이 항목에 흡수.
6. **E2 복귀·논문 정리** (FIELD_GAP §18.3): §10 분류 + 2-벤치 실증 + census→처방 절차를 본문 골격으로. **⚠️작성 규율(2026-06-12, coworker 전달)**: arXiv 제재 강화 — AI-작성 미검토 문장·허위 레퍼런스 시 전저자 1개월 제출 금지. **모든 인용 = 원문 검증 후**(5-agent 검증 패턴 표준화: §6.5처럼 버전·venue·인용문 박제), 수치는 결과문서 §번호 역추적 가능해야.
7. alias-마스킹 arm (P3 위생) · 형식-혼합 재학습 (조건부) — 변동 없음, 후순위.

**원칙 (승계)**: 한 번에 한 변수·census로 귀속·강한 주장은 궤적 전수 후 박제·**인용은 원문 검증 후 버전 명시**. zero-GPU 병렬(§18.2)은 전부 ✅.

## 10. ★층위별 분업 종합 — "무엇을 weight로 학습하고, 무엇을 결정론으로 하나" (2026-06-11 PM 박제)

> SOPBench(무재학습 사다리 15→29/34·3-NULL LOCK·offload)와 TaskBench(SFT/RFT/DPO/snap/guided 전 레버 측정)의 증거를 단일 분류로 통합. thesis(propose+결정론-gate)의 정량 장부. 빈칸 2개(P1·DPO-v2)는 판별 실험 결과로 채움.

### 10.1 레버 장부 — "학습 전부 무용"은 절반만 참
| 레버 | 산 것 | 못 산 것 |
|---|---|---|
| SFT | in-domain coverage 大(+18~27) · **참조-인덱싱 규율 held-out 전이(실재)** · SOPBench gather 스킬 LODO 전이 | held-out net≈0 (규율 이득 − 어휘 간섭 상쇄, §8) |
| RFT | in-domain 진짜 구조 개선(daily chain) | held-out 재추첨(±450 거울상, §6) |
| DPO | **누락축 첫 가동** + **v2 균형쌍 = held-out 신기록**(raw 55.95·패키지 57.30, P/R 동반상승·in-domain 회귀 소멸, §9.6) | v1 단방향=overshoot net 음수 — **신호 양방향 필수**가 합격 조건 |
| 결정론 (snap/guided/offload/DGGATE) | **held-out 첫 base 추월(52.5)** · SOPBench 15→29/34 · **guided v1: daily 붕괴 완전 회복 59.6→67.6(+8.0, base−0.5까지, §9.5b)** | 문자열-snap은 의미 매칭 불가(daily 689건) → guided가 의미 선택을 모델에 위임해 해소 |

**★핵심: held-out에서 이긴 것은 전부 "학습+결정론 패키지"였고 단독은 없음** (RFT2 단독 49.0<base / RFT2+snap 52.5>base · SOPBench adapter-only≈0 / scaffold+stack 75–95%). 학습 = propose-측 절반(결정론이 보정할 좋은 제안을 만드는 역할).

### 10.2 층위 분류 (표면→심층; 증거 박힌 것만)
| 층위 | 내용 | 승자 | 증거 |
|---|---|---|---|
| L1 도메인 심볼 | 도구명·credential·ID·파라미터값 | **결정론**(컨텍스트 복사·enum 제약) | 어휘간섭 −4~−8pp(§8)·32B base 이미 1.0(§8.5)·snap +3.5(§9.5)·guided 무효 0/3035 |
| L2 인스턴스 사실 | DB 상태·유저-특정 값 | **결정론**(gather 실행·retrieval) | 정의상 weight 불가(=ABox) |
| L3 게이트 연산 | permitted?·제약 트리 평가 | **결정론**(offload) | SOPBench 3-NULL LOCK(모델이 faithful gate *생성* 불가)·offload ACT 3→19·DGGATE +3 |
| L4 의미 매칭 | 패러프레이즈→정준 도구 | **하이브리드**(모델이 제안, 결정론이 출력공간 제한) | daily 689건=문자열 매칭 밖(§9.5)·**guided v1 확정: +8.0, base 수준 완전 회복·worsened 77/3647뿐(§9.5b)** |
| L5 절차 규율 | 참조-인덱싱·gather-먼저-act·종결 캘리브레이션·형식 준수 | **weight ✓** | 태그채택 1.5B 0.47→1.44·자기참조 −83~−98%(§8)·gather LODO 전이(SOPBench)·DPO 종결축 이동(§9.6) |
| L6 구성 구조 | edge 연결(어느 노드를 어떻게) | weight(in-domain)+scale 비포화 | RFT daily chain 개선·edge=후발 emerge 스킬(§4·§7)·held-out 구조이득 미해결 |

### 10.3 결정 변수와 비대칭
- 분류 축 = "깊이" 자체가 아니라 **(도메인-일반성)×(절차성)×(인스턴스-독립성)**: weight가 이기는 것=인스턴스 불변 절차 규율(L5), 지는 것=인스턴스에 닿는 모든 것(L1–L3).
- **이득 크기 ∝ base의 해당-규율 결핍** (§8 기제): 1.5B +8.7(결핍 大) ↔ 32B 사전예측 −5.0(규율 완비→간섭만, §8.5). ⇒ **weight-보강 = 규율-결핍 모델의 주입 수단**: 대형=결정론만 얹음 / 소형=L5 학습 후 결정론 얹음. ⚠️(2026-06-11 정정, FIELD_GAP §15.4 정정 참조) 주권-leg는 "내부망=sLLM" 아님(B200급 on-prem=32B+ 가능) — 본질은 **오픈웨이트×on-prem(크기 무관)**, 크기는 경제 변수. 이 census→처방 절차가 곧 "어느 크기+어느 레버" 배포 의사결정 도구.
- L5 정책류 학습은 **신호가 양방향이어야 함**(DPO v1 단방향 overshoot = SOPBench Gate-B 교훈 재현).
- xattn-LoRA(RUNG1 §3.10 B5*)의 위치: 구조를 출력 어휘에 굽지 않고 별도 conditioning 채널로 → L5(규율)는 weight·L1(심볼)은 채널 복사로 분리 = 어휘-간섭의 구조적 원인(규율과 심볼이 같은 토큰 스트림에 혼합 학습) 차단. 단 retrofit 비용 → guided(추론-side)가 같은 효과를 공짜로 내는지 먼저 측정(진행 중).

### 10.4 미결 — 이 분류를 판가름할 라이브 판별 실험 2개
1. **coworker P1 (32B SFT, §8.5 사전등록)**: ✅ **확정 (2026-06-11 실측·06-12 Track A 독립 재검증)** — Δ실측 −5.4 vs 예측 −5.0 (오차 0.4pp) = 판정① 기제 확립. **"이득=base 결핍의 함수" Δ≈19.4×nself−5.0이 1.5B/7B/14B/32B 4점 함수형으로 성립**, "14B=용량-부활" 최종 기각. 간섭 −4.8pp도 예측 그대로. L1/L5 행과 §10.3 비대칭 전부 정량 근거 완성.
2. **DPO v2 균형쌍 (§9.6 사전등록)**: ✅ **확정 (2026-06-11 PM)** — 3기준 합격(패키지 57.30 신기록·in-domain 회귀 소멸·P/R 동반상승). **정책류(종결 캘리브레이션)는 weight로 학습 가능, 단 신호는 양방향 필수** — v1 단방향(net음수) ↔ v2 양방향(전축 개선)의 깨끗한 대조로 박제. L5 행 확정.

### 10.5 ★벤치-불변 규칙 vs 벤치-어댑터 분리 (2026-06-12, 사용자 목적 재정렬 — "벤치 선택"이 아니라 "전 벤치 커버 프레임워크")
> **목적 명제 (사용자)**: 다양한 벤치를 **최소한의 자동 노력**으로 전부 커버하는, 기본 규칙을 내재한 프레임워크. 벤치 전수조사(§1.5·실행-축 보고서)는 선택용이 아니라 **커버리지 행렬의 타깃 목록**. (τ2/SOPBench/SOP-Bench/AppWorld는 기검토 — 재발견 금지.)
> **★권위본 = `scripts/distill/BENCH_PORTFOLIO_FRAMEWORK_DESIGN.md`** — R1-R8×A1-A5 전체표 + **벤치 포트폴리오 6+1**(완료 TaskBench·SOPBench / 신규 ★τ²=NL정책 front-end 무대·Amazon SOP-Bench=12도메인 LODO·AppWorld·ODCV 스팟 / 조건부 WorFBench) + 실행순서(τ² retail 어댑터→Amazon 행렬→스팟) + 커버리지 행렬. 아래는 요약.

**벤치-불변 규칙 R1-R8 (프레임워크 내재 — 전부 본 문서·SOPBench 결과로 실측 근거 보유):**
| # | 규칙 | 근거 |
|---|---|---|
| R1 | 심볼(도구명·필드명)은 생성하지 않고 컨텍스트에서 베낀다 — enum/문법 마스크로 집행 | §9.5b guided +8.0·무효 0/13k |
| R2 | 인스턴스 사실은 act 전에 gather (정보수집 선행) | SOPBench active-H3 6→15·gather LODO 전이 |
| R3 | 허가/결정 판단은 모델 emit 금지 — 결정론 게이트 offload | SOPBench 3-NULL LOCK·DGGATE 15→29 |
| R4 | 의미 매칭은 모델이, 출력 공간은 제약이 (하이브리드) | daily 의미-패러프레이즈 해소(§9.5b) |
| R5 | 정책류 행동(종결·길이)은 on-policy **양방향** 선호학습으로만 — 규율-완비 base에 모방-SFT 금지 | DPO v1/v2 대조(§9.6)·P1 −5.4(§8.5) |
| R6 | 구조 선택(L6)은 K-제안+결정론 검증-**선별**(마스킹 아님 — 분포왜곡 회피) | DiG-Plan Pass@10 0.94·GAD 이론·edge-snap NULL(§8.7) |
| R7 | 배포 전 base census→레버 선택 (이득=결핍 함수, 크기·family 불문) | 4점 함수형(§8.5)·§8.6 |
| R8 | 측정 규율: 집계 후 즉시 궤적 census·사전등록·내부-일관 비교 | 본 문서 전체의 방법 |

**벤치-어댑터 A1-A5 (벤치당 최소 추출물 — 자동화 대상; 이것만 새로 쓰면 커버):**
| # | 추출물 | 자동화 수준 | 견본 |
|---|---|---|---|
| A1 | 도구/심볼 카탈로그 → enum 스키마 | **기계적** (tool_desc/OpenAPI/MCP 파싱) | `tb_guided_schema.py` |
| A2 | 정책/SOP 텍스트 → 제약 구조(게이트 입력) | **★유일한 연구-난제 = thesis의 NL→구조 front-end 그 자체** (SOPBench은 구조 제공돼 우회했음; τ2는 NL정책뿐=front-end 필요성의 실증 무대) | Guard-2 재구성·학습 front-end |
| A3 | 평가기/검증기 → 보상·채굴 신호 (RFT/DPO 쌍) | 래핑 (벤치 제공물 재사용) | `tb_dpo_mine.py`·sopbench_reward |
| A4 | 도메인 경계 → LODO 분할 | 기계적 | `tb_train_lodo` 류 |
| A5 | 출력 스키마 → guided 문법 (형식 분기 포함) | 기계적 | schema --dep 분기 |

**⇒ 프레임워크 주장 형태**: "새 벤치 커버 비용 = A1/A4/A5(기계적)+A3(래핑)+A2" — A2가 병목이고, **A2를 학습 front-end가 대체하는 것이 thesis의 상품 가치**(per-domain authoring 비용 제거). 커버리지 행렬(벤치 × A1-A5 가용성 × R1-R8 적용처)은 landscape census 도착 후 작성.
