# HANDOFF 2026-06-10 — TaskBench 학습 실험 (thesis 동결 후 *측정* 단계)

> **진입점.** §14–17 설계는 **동결**(`FIELD_GAP_LLM_VALUE_DESIGN.md` §17.9 = 고정 thesis). **리뷰3(2026-06-10) 반영: 실행 권위 = 설계서 §18**(E0/E2/E5 disposition·§15.4 사활 open 3건[규제원문 sourcing·bitter-lesson·erosion] zero-GPU 병렬 배정 — 특히 **규제 1차원문 sourcing은 moat-leg 사활이라 Exp-A와 병렬 필수**). 이 핸드오프 = 다음 세션이 *측정*(학습 실험)을 바로 실행하기 위한 자족 문서. **메타규칙: 더 이상 thesis 정제 금지(수확체감) — 코드/측정으로 전환.**

---

## 0. TL;DR (고정 thesis 한 줄)
**고정 도구 + 사전 결정론 compute 위에서, 소형 모델이 도구-호출 경로를 *제안*(=coverage, 어려운 부분) + 결정론·검사가능 게이트가 soundness 보장(audited 제약모델 대비; 불확실하면 abstain) + 재학습0 전이.** 헤드라인 = **보장 soundness 하 *높은 coverage*를 {소형·저비용}×{감사가능 게이트}×{전이} 패키지로** (= precision=1서 recall 최대화). capability·최적성(#2)=supporting/deferred. (상세 §17.9.)

## 1. 벤치 역할분담 (forward guard 통과만)
- **TaskBench** = 충실성 *반쪽*(NL→구조 node/edge-F1, soft-match, **실행·soundness 없음 → #1 주장 불가**). = *coverage/구조-예측* leg.
- **SOPBench/SOP-Bench** = **soundness + 제약 + 전이 = #1의 진짜 자리**(실행·게이트·거부).
- **통합(NL→구조→게이트실행→success+전이)** = SOPBench 풀파이프라인 + **blind E1**(= 진짜 통합; 현 Exp-5는 §1-노출판). PRAXIS=실세계(후행).
- forward guard: 벤치는 "소형이 frontier 이기나"로 평가 금지 → (1)충실성 (2)soundness/제약 (3)전이. (1)-only=supporting; moat=(2)∨(3). **AppWorld 탈락**(코드모달리티·capability·셋 다 ✗).

## 2. ★인프라 READY (TaskBench, 리모트 — 그대로 재사용)
- **clone**: `/home/woori/scratch/JARVIS_tb/taskbench` (Apache-2.0). 3도메인: `data_huggingface`(7458)·`data_multimedia`(5555)·`data_dailylifeapis`(4318).
- **eval venv**: `/home/woori/scratch/tbeval_venv` (numpy·sklearn·networkx·Levenshtein·datasets2.14.5·pyarrow12·rouge_score·aiohttp·emoji). **이걸로 evaluate.py·inference.py 둘 다 됨.**
- **★재현 gotcha (필독, 안 하면 깨짐)**:
  1. **필드변환**: 원본 `data.json`=`tool_nodes`/`sampled_links`/`tool_steps` → evaluate.py는 `task_nodes`/`task_links`/`task_steps` 기대. **변환 전처리 필수**(아래 converter).
  2. **메트릭명**: node-F1=`-m f1`, edge-F1=`-m link`, param=`-m argument` (`-m node`은 *무효*). step(rouge/bert)은 우리 불요.
  3. **id 정렬**: `user_requests.json`(inference 입력)과 `data.json`(gold)의 **id 순서가 다름** → pred를 gold와 *id로 정렬* 안 하면 일부만 매칭됨(예: 150중 19). pred id 기준으로 gold 빌드할 것.
  4. **inference.py 버그**: `results` 미초기화 → `loop = asyncio.get_event_loop()` 다음 줄에 `    results = []` 추가(이미 패치됨).
  5. pred 경로 = `{data_dir}/{prediction_dir}/{llm}.json`; metrics 저장 = `{data_dir}/metrics/{llm}.json`(overall = `["overall_overall"]` 키).
- **vllm**: `/home/woori/venvs/tau2_vllm_env/bin/vllm` (0.11.0). 모델 캐시: Qwen2.5-{0.5,1.5,3,7,14,32}B-Instruct 전부 있음(=scale 곡선 가능). 서빙:
  ```
  CUDA_VISIBLE_DEVICES=0 setsid nohup $VLLM serve Qwen/Qwen2.5-7B-Instruct \
    --port 8000 --served-model-name qwen7b --max-model-len 8192 --gpu-memory-utilization 0.85 \
    > /home/woori/scratch/vllm_qwen7b.log 2>&1 &
  ```
  inference: `python inference.py --data_dir <sub> --api_addr localhost --api_port 8000 --api_key dummy --llm qwen7b --multiworker 8 --dependency_type resource`
- **GPU**: GPU0/1 ~free(ollama GPU1 idle, kill금지). 협업자 H200 Track-B(32B SFT)와 충돌 주의.

## 3. ★Baseline 실측 (DONE, §17.8) — frontier 비-포화·edge가 진짜 축
| 모델 | n-F1 | e-F1 | t-F1 | v-F1 |
|---|---|---|---|---|
| gpt-4 (published) | 90.9 | 69.3 | 87.1 | 72.3 |
| gpt-3.5 | 72.8 | 44.0 | 65.9 | 40.8 |
| **Qwen2.5-7B-Instruct (base, prompted, 150 MM 실측)** | **83.3** | **49.3** | **72.5** | **54.0** |
| codellama-7b / vicuna-7b (published 2023) | 53/46 | 15/4 | — | — |
- ★**현대 base 7B(Qwen2.5)는 이미 gpt-3.5/claude-2 급(n-F1 83)** → **node headroom 작음**. **진짜 headroom = edge-F1(의존구조): 49 vs 69 = 20pt.** ⇒ 학습은 *node-선택*(base가 이미 함)이 아니라 **edge/의존-구조 + 전이**를 노려야(= §15.4 capability-침식 실증).
- ⚠️ caveat: 150 subset·1도메인·단일run = 노이즈. **첫 할 일 = full 3도메인 안정 baseline.**

## 4. 실행 큐 (우선순위)
**Exp-A (1순위) — §16 학습 on TaskBench → LODO (★supporting 전이 — moat-(3) 주장 금지, 설계서 §17.9 리뷰7-1 사전등록)**
- **(A-0, zero-GPU, RFT 전 BLOCKING)**: 7B edge-miss **~30개 수동 감사** → real-error vs valid-대안 분율 추정(P2: sim()=exact라 대안 penalize → edge 20pt headroom 실제 크기 미지; 결과로 보상 설계 확정).
- 레시피(§16 정합, ★명명 정정): **gold-SFT**(TaskBench 17K 전 샘플에 gold graph 존재 → teacher 호출 불요; "distill"은 gold 없는 SOPBench-E1용 논거) → **outcome-RFT(보상=node/edge-F1)**. ⚠️**GT-generator 순환 caveat**: GT=back-instruct(GPT-4 생성) → gpt-4를 teacher/증강에 쓰면 frontier-비교 부분 순환; teacher 필요 시 비-GPT-4 frontier.
- 세팅: 2도메인 학습 → **held-out 1도메인 평가**(LODO). 지표 = **edge-F1 중심**(node는 거의 saturated) + type(single/chain/dag) 층화.
- arms: ①base-Qwen7B(prompted, baseline) ②learned(gold-SFT+RFT) ③frontier(published).
- **판정**: learned가 base보다 *edge-F1* 올리고 held-out 전이 유지하나. **보고 = "supporting 전이"**(동일벤치·n=3·alias 조건부) — moat-(3)은 cross-bench(SOPBench→SOP-Bench)로만.
- ⚠️ **보상 설계 caveat**: exact-match F1 보상은 GT-특이성 overfit 위험(P2 실측: sim()=exact, 대안 penalize). → matching-mode-F1 보상 검토 OR soundness가 필요하면 SOPBench-실행보상(아래 Exp-B).
- ⚠️ **P3(전이 오염)**: 도구명이 NL-기술·암기가능 → **alias-마스킹**(의미매칭 강제, SOPBench alias 교훈) 적용해 "이름암기 전이" 배제.

**Exp-B (병렬/2순위) — SOPBench 풀파이프라인 / blind E1 재개 (soundness leg = #1)**
- TaskBench로는 #1(soundness) 측정 불가. **#1의 진짜 측정 = SOPBench 게이트 실행(precision=1서 coverage) + 거부(should_F)**.
- **blind E1**(paused) = NL서 구조 도출→게이트→전이 = *진짜 통합*. (현 Exp-5는 구조화입력=§1-노출, 통합 아님.) §13.5/§6 E1 참조. ⚠️§1.1 inherited-structure 선결게이트.

**Exp-C (선택) — scale 곡선**: Qwen2.5-{0.5,1.5,3,7,14}B로 node/edge-F1 곡선(이미 모델 캐시됨) = "어느 크기서 edge-구조 emerge."

## 5. converter (필드변환, 복붙용)
```python
import json,os,shutil
def pj(v): return json.loads(v) if isinstance(v,str) else v
def build_eval(src_data, pred_file, dst):  # src_data=원본 data.json, pred_file=inference 출력
    os.makedirs(f"{dst}/predictions",exist_ok=True)
    pred_ids=[json.loads(l)["id"] for l in open(pred_file)]; pset=set(pred_ids)
    gold={}
    for l in open(src_data):
        d=json.loads(l)
        if d["id"] in pset:
            tn=pj(d["tool_nodes"]); tl=pj(d.get("sampled_links","[]")); ts=pj(d["tool_steps"])
            gold[d["id"]]={"id":d["id"],"type":d.get("type","single"),
              "task_nodes":[{"task":x["task"],"arguments":x.get("arguments",[])} for x in tn],
              "task_links":[{"source":x["source"],"target":x["target"]} for x in tl] if tl else [],
              "task_steps":ts}
    ids=[i for i in pred_ids if i in gold]
    with open(f"{dst}/data.json","w") as fg:
        for i in ids: fg.write(json.dumps(gold[i])+"\n")
    shutil.copy(pred_file, f"{dst}/predictions/qwen7b.json")  # rename llm as needed
# eval: python evaluate.py --data_dir <dst> --prediction_dir predictions --llm <llm> --splits all --n_tools all --mode add --dependency_type resource -m f1 -m link
# overall: json.load(".../metrics/<llm>.json")["overall_overall"]["node_micro_f1_no_matching" / "link_binary_f1"]
```

## 6. 헤드라인 지표 (측정 시 보고)
- **coverage @ precision=1** (soundness 하 valid-path률) = SOPBench(실행). TaskBench는 *proxy*(node/edge-F1, soundness 없음).
- **edge-F1 전이**(LODO held-out): base vs learned vs frontier.
- **전이 유지율**(in-domain vs held-out).
- ⚠️ frontier를 "이김"이 아니라 *frontier-comparable coverage를 싸게+전이*(=패키지)로 프레이밍(forward guard).

## 7. 정직 caveat (승계)
- TaskBench eval=exact-match(대안 penalize, 천장<100; gpt-4 n-F1 90.9가 반영). human-verified 도메인편차(MM 62.7% vs HF 10.8%).
- "100% soundness"=*인코딩 제약 대비*만(§13.7: 게이트도 틀릴 수 있음). moat=soundness *검증가능성*.
- 새 학습신호 0이 오래됨 → **측정 우선, 정제 금지.**

---
**다음 세션 첫 행동 = Exp-A 착수**(full 3도메인 base baseline → distill+RFT → LODO edge-F1). 인프라(§2)·converter(§5) 그대로. 동결된 thesis(§17.9)에 *측정으로* 답할 때.
