# 자율 진행 로그 (2026-06-14 PM, 사용자 외출 ~5h) — R1b 아키텍처 구현

> 상위 = `R1B_PROVENANCE_DESIGN_2026_06_14.md`. 사용자 복귀 시 검토용. 시각순 추가.

## 13:00 Stage A — L2 A/B 판정 (예비)
- **fcq3b (v3 ask-user, 150-up, without-L2) = compliant-pass 0.15** (50-up 0.10 → 150-up 0.15·base 0.17 근접). **L3(ask-user)가 작동 — 학습할수록 회복** (50-up의 날조 90%는 초기 노이즈).
- **retail_l2 (v3, with-L2) = 컨텍스트 오버플로 에러**: L2 deny → 모델이 *복구(ask) 안 하고 다른 placeholder로 재시도* → deny 루프 → 16384 초과. ⇒ **L2-deny 단독은 (복구 미학습 모델을) 루프시킨다.**
- **판정**: ①L3(ask-user) 효과 양성·느림 ②L2는 L3(복구) *함께* 필요 ③root-cause = **placeholder 값 날조**(값 memorization).

## 13:15 처방 구현 (Stage B)
- **값-randomization** (`fc_value_randomize.py`): user-제공 식별값을 포맷-보존 랜덤토큰으로 일관치환(user발화+tool-call+tool출력) → memorize 불가 → **컨텍스트서 복사 강제**(도구명 alias의 값 버전). SOPBench 4543/5028 적용. 검증: `new_user_123→uda_xihh_218`(user·call 동일).
- **L2 deny 메시지 directive화**: "STOP·ask the user now·Do NOT retry with guessed value" → deny-루프 차단.
- **sft_v4** = 값-랜덤 SOPBench(6288) + 값-랜덤 ask-user(1761) + TaskBench(7000) = 13789·QC깨끗.

## 13:15 발사 — v4 대규모 재학습 + L2 A/B
- **학습**: `qwen7b_fc_tbox_v4` (GPU0·flash-attn·grad-accum4·ep2·save-every50).
- **자동테스트**(`driver_v4.sh`·GPU1): 3rd 체크포인트(~opt-step150) → 동일 어댑터 **without-L2(retail_v4) vs with-L2(retail_v4_L2)** A/B → `v4test.log`·sentinel V4TEST_DONE. 예상 ~14:15.
- **핵심 질문**: 값-randomization(복사강제) + ask-user(복구) + L2(enforce·directive)가 날조를 없애고 compliant-pass를 base(0.17)→frontier(0.81) 쪽으로 올리나.

## L1 (decode-mask) 평가 — 정직
- L1(인자값을 컨텍스트-후보로 디코딩 제약)은 **vLLM OpenAI 엔드포인트(litellm 경유)서 컨텍스트-의존 logits-processor를 per-request로 넣기가 어렵다** — guided_json/choice는 정적, 동적 컨텍스트-제약은 커스텀 서버 패치 필요. **자율로 안정 구현 위험 큼.**
- **L2(사후 게이트)가 기능적 등가**(날조 차단→복구 강제)이고 구현됨. L1은 *효율/보장-강도* 업그레이드(턴 절약·구조적0). ⇒ **자율 범위 = L2-enforcement로 진행, L1은 production 업그레이드로 설계 보존**(R1B §3c). 시간·명확경로 있으면 scoped 시도.

## 다음 (루프 자율)
1. v4 A/B 수확·판정(~14:15): with-L2가 날조↓·pass↑면 **R1b 아키텍처(L2+값랜덤+ask-user) 검증**.
2. 양성 → 학습 수렴까지(ep2) 후속 체크포인트 재테스트·날조율 추세.
3. 음성/막힘 → 기제 진단·다음 안 박제.
4. 전이(SOP-Bench·τ²) 측정은 v4 검증 후.

## 14:02 ★Stage B/C 결과 — 값-randomization이 날조 제거 (핵심 성공)
- **v4 (값-random + ask-user) ~150-up**: without-L2 pass **0.10·날조 0%**(grounded 19/20) / with-L2 pass **0.15·날조 5%**.
- vs v3(ask-user만) 날조 40-90% → **v4 날조 0-5%** = **값-randomization이 placeholder 날조 제거**(모델이 컨텍스트서 값 복사). root-cause 수정 실증.
- L2도 약간 도움(0.10→0.15·base 0.17 근접)·위반 0 유지.
- ⇒ **R1b 학습-측(값랜덤+ask-user) 작동 확정.** 남은 gap(→base/frontier)은 날조 아닌 task-해결 능력.

## 16:53 v4 최신 체크포인트(opt-step~1200) 재테스트
- v4 ep0 계속 학습(step4850·24체크포인트). 최신 체크포인트 A/B(driver_v4b.sh) → 더 학습 시 pass↑·날조 유지? 결과 ~17:10·sentinel V4BTEST_DONE.

## ★D5 대조쌍 (fetch-우선 게이트) 구현·빌드 완료 (day-7 PM 이어서)
- **근본원인(사용자 교정)**: τ² over-ask = "fetch 우선이어야 하는데 ask함". v4 ask-aug(`fc_askuser_augment.py`)이 키워드 휴리스틱(dob/birthday/income/amount/member...)으로 무차별 ask → getter-output 슬롯까지 ask 학습 → always-ask 붕괴(R1b §3a D5).
- **구현(`fc_d5_contrastive.py`·커밋)**: ask/fetch 분기를 **카탈로그-결정론**으로. 게이트 = **provenance(값∈user 발화 & ∉tool 출력·정확 신호) + getter_map(arg_key 토큰 ⊆ getter 생산-슬롯이면 fetchable=ask 금지)**. 휴리스틱 폐기.
- **빌드 census(sop_rand 5028)**: 자연 fetch-then-use **3024 traj(60%)** = fetch 분기 풍부 / ask-적격 4414 / **OVER-ASK=0 ✓**(ask한 키 중 fetchable 0). 1st-call provenance = user 7750·none 86·tool 0(첫 호출=순수 identity).
- **★대조 신호 확인**: user는 ID/타입/identity 제공(ask) vs 시스템은 details/status fetch — `product_id`(ask)↔`product_details`(fetch)·`order_id`↔`order_details/history`·`room_type`↔`room_assignment`·`test_type`↔`test_details`. 새 게이트가 구 휴리스틱이 **놓친** 정당 user-param(room_type·check_in_date·plate_num·foreign_currency_type 등)도 ask.
- **★sft_v5 빌드 완료**(`fc_build/sft_v5.jsonl`·13781 traj): = sop_rand(5028) + **sop_d5_ask40(1753·게이트 ask, frac0.40)** + tb(7000). **v4(13789)와 동일 구조, ask만 휴리스틱(1761)→D5 게이트(1753) 교체 = 깨끗한 A/B**(volume 일치, 선택만 변경). QC 클린·d5_branch 라벨 보존(ask 1753).
- **fetch/upfront 예시 = sop_rand 자체**(이미 자연 fetch-then-use 포함) → 별도 합성 불요(날조 위험 회피). 대조는 데이터셋 레벨(ask-when-no-getter ∪ fetch-when-getter) + over_ask=0 보장.
- **★v5 학습 launch (GPU1·woori)**: GPU1 실제 free 확인(41MiB·21GB는 v4btest vllm 잔여로 이미 종료) → **v5 학습 시작**(PID 3977030·`sft_runs/qwen7b_fc_tbox_v5`·`v5_train.log`). 설정 = **v4와 동일**(epochs2·lora-r16·alpha32·grad-accum4·max-seq-len14336·flash_attn2·CUDA_VISIBLE_DEVICES=1·save-every50·val-frac0.02) → 데이터만 sft_v5 = 깨끗한 A/B. v4(GPU0) 결과 후 v5(GPU1)와 τ² 비교(over-ask율·compliant-pass).
- **★coworker 동기화 (캐노니컬)**: ①`getter_map.json` repo 박제(7도메인) ②`node_run_planx.sh`에 §2b 값-randomize+§2c D5 대조 ask 반영·§4 보류해제 ③`COWORKER_REQUEST_TB_SCALE.md` **v8**(v7 보류 해제·"지금 돌려라"). coworker = 전 teacher×7도메인 대규모. 전부 push.
- **다음**: ①v5 인코딩→학습 수렴 모니터 ②v4 vs v5 τ² A/B(driver 패턴·over-ask율·compliant-pass) ③coworker 캐노니컬 산출 합류 ④전이(SOP-Bench·τ² held-out).

## ★v4 정지 + 학습량 vs τ² 전이 곡선 (20:45~20:51)
- **loss 판단**: v4 train-loss 구간평균 0-1k 0.35 → 2.5-4.5k 0.19 → 4.5-6.5k **0.16(평탄)** → 6.5-9k **0.21(소폭상승)**. **step~4500서 수렴·이후 학습신호 소진**. val-loss 미측정(트레이너 인코딩만). → **ep2 무의미·과적합 위험**(부검 "더학습=망각 단조" 사전증거) 판단으로 **v4 정지**(ep0 step10199서·snapshot `v4_final_adapter`).
- **★v4_final τ² eval(GPU0·v4 정지로 해제·v5 무중단)**: `tau2_eval_adapter.sh`(repo·git전송)로 serve→t2_run_gated without-L2 n=20 → **pass^1=0.10**.
- **★학습량 vs τ² 곡선 (전부 v4=value-random+휴리스틱ask)**: step150 **0.10** / step1200 **0.11** / step10199 **0.10** = **완전 평탄·base 0.17 미달**. **학습 68배 늘려도 τ² 전이 0 개선** → 정지 판단 실증·ep2 불요 확정.
- **★해석(중요)**: v4는 이미 value-random이라 **날조는 잡힘**(fab 0-5%·핸드오프). 그런데 pass 0.10<base 0.17 정체 = **잔여 갭은 날조 아님·SFT가 τ² capability를 오히려 저하**(over-ask + task-해결 능력). ⇒ **v5(D5 fetch-우선 게이트)가 결정적 테스트**: D5가 over-ask 잡아 0.10→0.17+ 회복하면 fetch-우선 처방 작동 입증. v5도 ~0.10이면 ask/fetch보다 깊은 capability 문제(generator-gap).
- **현재**: GPU0 free·v5 GPU1 무중단(ep0 step3550·~26%). v5 ep0 완료 후 동일 `tau2_eval_adapter.sh`로 A/B.

## ★★v4 τ² 전수 궤적 autopsy — 근본원인 확정 (`tau2_autopsy.py`·20개 전수)
- **pass 2/20=0.10. ★두 PASS 모두 write 불요(read-only) 태스크. write 필요 태스크는 전멸**(gold write 매치 거의 전부 0/X).
- **실패분포**: fab_auth 6·agent_collapse 5·no_auth 2·**over_ask 2**·premature_refuse 2·wrong_write 1.
- **★진짜 근본원인 (dump 확정·핸드오프 §54 해석 정정)**:
  1. **tool-fetchable 값의 *날조* (지배적)** — fetch해야 하는데 placeholder를 지어냄. task13: 인증성공 후 `get_order_details(order_id='#W0000000')` 날조, user가 "주문번호 안 줬다·이메일로 찾아달라" 명시해도 **get_user_details로 order 목록 fetch할 줄 모르고** 동일 날조 무한반복→too_many_errors. task17: email·order_id·주소(`123 Main St` 통째) 날조(현재주소 fetch 안 함·user는 suite만 변경 원함). **★value-random은 identity(email/name)만 grounded(10/18)·order_id·주소 등 비-identity tool-fetchable 값 날조는 그대로** = **사용자 지목 "fetch 우선" 문제의 진짜 형태(ask 아니라 placeholder 날조)**.
  2. **에러/게이트 후 재시도 루프 붕괴 (5건)** — task17: G2_CONFIRM_WRITE(user yes 필요)에 막히자 **확인 없이 같은 modify 8연타**→too_many_errors. 에러 시 전략전환(fetch/ask) 없이 동일호출 반복.
  3. **read 과수집·write 미도달** ~13/20.
- **★왜 LoRA(0.10)<base(0.17)**: SFT가 gather/read 규율은 강화했으나 ①없는 값 fetch-체인(get_user_details→order_id) ②에러 복구 미학습 → placeholder 날조+read 루프 *증폭* → base보다 나빠짐.
- **★D5/v5 함의(정직)**: D5 fetch-우선 원칙은 (1)을 정조준=방향 맞음. **단 현 value-random/D5는 identity에만 적용 → order_id·주소 등 *전 tool-fetchable 값*으로 확장 + "없으면 get_user_details로 fetch"하는 체인 학습 필요.** **(2)재시도-루프 붕괴는 D5 미해결 → 별도 처방(L2 deny→recover·에러 시 전략전환·RL).** ⇒ v5는 (1) 부분개선 기대·0.10→0.17+ 도약엔 (1)확장+(2) 동반 필요.
- 도구 = `scripts/distill/tau2/tau2_autopsy.py`(repo·`--dump TASK`·`--full`).

## ★★★v6 — fetch-to-obtain-arg 스킬이 학습데이터에 부재함을 발견·복원 (벤치-무관 처방)
- **사용자 방향(불변)**: tau2-도메인 특화 함수 처방 금지. R(카탈로그)에 도구 있으면 placeholder 날조 금지하는 **R1/R3 벤치-무관 규율**을 SOPBench/TaskBench서 학습→전이.
- **★진단(전수 검증)**: "tool 출력서 값 가져와 인자로 쓰는"(fetch-to-obtain-arg) 스킬이 **학습데이터에 사실상 부재**. SOPBench=**1.9%**(getter 대부분 fetch-to-**DECIDE**=조건게이팅이지 obtain-arg 아님)·TaskBench=**0%**(변환기가 출력을 합성 `ok/ref` 스텁으로·인자는 instruction 복사). = **order_id 날조의 진짜 뿌리: 모델이 그 스킬을 배운 적 없음.**
- **★발견 = TaskBench `<node-N>`**: HF **57%**·MM **59%** 궤적이 `<node-N>`(상류 출력→하류 입력) 체인 보유하나 **구 변환기가 전부 버림**. = fetch-to-obtain-arg의 벤치-무관 천연 출처.
- **★변환기 수정**(`fc_convert_taskbench.py`): `<node-N>`을 **상류 출력 ref(res_xxx)로 threading**. ★규약 정정(사용자: R 자기참조 금지 규율): `<node-N>`=리스트인덱스 아님 → **링크(의존성 source)로 해석**(self-ref 제거). 재변환 tb_all_v3 = fetch-chain **41%**(6473)·self/forward-ref 13개만(드롭). ref=md5 궤적-고유=비-memorizable→**copy 강제**(placeholder 날조 시 gold 불일치로 패널티).
- **★fetchable randomize**(`fc_randomize_fetchable.py`): SOPBench tool-출력서 와 재사용되는 값도 randomize(identity+fetchable). sop만은 1.9%로 희소 → 주 신호는 TaskBench threading.
- **★sft_v6 = sop_rand2 + sop_d5_ask2 + tb_all_v3c**(13780). fetch-to-obtain-arg 신호 강함(tb 41%). **GPU0서 v6 학습 시작**(PID 3983897·`qwen7b_fc_tbox_v6`·v4/v5 동일 config). **GPU1=v5(D5 ask만) 계속** → v4(0.10)/v5(D5)/v6(fetch-teaching) 3-way.
- **★설계질문 답(벤치-무관 전제)**: ⓠ1 "전부 randomize하면 fetch?" = **부분 Yes·필요충분 아님**: randomize는 memorize 차단일 뿐, **fetch STRUCTURE(출력→인자)가 데이터에 있어야** 함 — 없었음(1.9%/0%) → **threading으로 *생성*** 후 randomize=copy강제. ⓠ2 "에러 재시도 룰 벤치-무관 학습?" = **Yes**: 룰=R3 일반("에러 시 동일호출 금지·re-gather/ask"). 단 success-rollout은 에러 희소→**에러→복구 augmentation 필요**(정답값 사용=날조0)+결정론 가드(동일-실패 반복차단). ⓠ3 "재시도 A2 정의대로?" = **Yes·최고 thesis-순수**: 게이트 복구절차=정책(A2) 정의·모델은 "게이트 막히면 A2 명시 전제 충족 후 retry" 일반스킬 학습·ABox-swap 전이. (ⓠ2/ⓠ3=v7 후속, v6은 ⓠ1 fetch 부재 복원에 집중.)
- **★변환기 정합성 수정**(사용자 지적: 결정론 알고리즘인데 self-ref 왜?): 원인=**TaskBench 원본 데이터 비정합**(알고리즘 버그 아님). 84/20404 ref-노드(0.41%): no_link_source 62(`<node-N>`있는데 링크 없음)·dup_task 12·cycle/same-level 10. → 변환기에 **post-check 결정론 drop**(literal `<node-` 잔존 or res_ 산출-전-사용 시 trajectory 통째 drop) 추가. 재변환 tb_all_v4 = **residual 0·fetch-chain 41% 유지**(15711·dirty 99 drop).
- **★v6 클린 재시작**: 첫 v6(PID 3983897)는 tb_all_v3c(~0.26% literal 노이즈)·인코딩만 하고 정지 → **클린 tb_all_v4로 sft_v6 재빌드·재시작**(PID 3984552·GPU0). v5(GPU1) 계속.
- **★coworker 동기화**: node_run_planx.sh = §변환기 threading 자동 + §2b `fc_randomize_fetchable`로 교체 + v6 배너. 요청서 **v9**(fetch-to-obtain-arg 복원·sentinel rm 안내). 전부 push.
- **다음**: ①v6/v5 ep0 후 `tau2_eval_adapter.sh` 3-way A/B + `tau2_autopsy.py`(order_id 날조율↓?) ②양성이면 ⓠ2/ⓠ3(에러-복구·A2 retry) v7.

## ★v5 조기-eval (step4999·D5 ask-only) — autopsy 예측 적중
- **v5 pass^1 = 0.105 ≈ v4(0.10)**: **D5 ask-게이트 단독은 τ² 개선 0.** loss 평탄(step~1500 수렴) 확인 후 stop→eval→`--resume`(`tau2_eval_resume_v5.sh`·GPU1·v6 무영향).
- v5 autopsy = v4와 동일 프로파일: PASS 2(write 있는 태스크)·나머지 **agent_collapse(too_many_errors·aw=10=재시도 루프 망치질)+fab_auth 지배**. ⇒ **ask/fetch *선택*은 병목 아님**(over_ask 2/20 예측대로)·**병목=fetch-실행(v6)+에러-복구(v7)**.
- **3-way 현황**: v4(휴리스틱 ask) 0.10 / v5(D5 ask) 0.105 / **v6(fetch-teaching) 학습중**. v5는 ≈v4 확정(dead-end) → 완주 무의미·GPU1은 v6 eval에 쓰는 게 나음(권고).

## ★★v6 중간 eval (step2599·fetch-teaching) — 부분성공·진짜 블로커 노출
- **v6 pass^1 = 0.05** (v4/v5 0.10보다 낮음·단 step2599=ep0 19% 중간). GPU1 eval·v6 GPU0 무중단.
- **★identity 날조 잡힘**: auth provenance **grounded 18/19**(v4/v5는 fab ~8-10) = **fetch-teaching/fetchable-randomize가 identity 값 grounding 성공**.
- **★그러나 order_id 여전히 `#W0000000` 날조**(task5 dump): user가 "주문ID 없음·이름/zip으로 찾아달라"→ find_user 성공(user_id 획득)→ **get_user_details로 주문목록 fetch 안 하고** get_order_details('#W0000000') 날조→error→반복→transfer(포기).
- **★진짜 블로커 = 2-hop proactive gather 부재**: threading은 "이전 출력 ref를 인자로 *복사*"를 가르침. 하지만 τ² order_id는 **"없는 값→생산 도구(get_user_details)를 *능동 선택해 먼저 호출*→출력서 order 선택"**(R2 gather + R4 select)이 필요. SOPBench getter는 username(기지)만 받아 2-hop 아님·TaskBench는 그래프 given이라 *선택* 학습 안 됨. = **이 스킬이 학습데이터에 없음**(autopsy SOP 1.9%/TB 0% 발견의 더 깊은 층).
- **pass 하락 기제**: 모델이 *더 정직*해져(날조 후 진행 대신 막혀 멈춤/포기) collapse↑(12). = honest-but-stuck.
- **함의/다음(v7)**: ①**proactive 2-hop gather 학습**(없는 arg→생산 getter 선택→호출→출력서 select). threading을 *guided*서 *unguided 선택*으로(예: user 발화서 값 빼고 getter 호출 강제). ②에러-복구(#W0000000 error→get_user_details 전환). ③order_id류 placeholder도 randomize 대상에. **v6 ep0 완주 후 재eval하되 기제상 이 2-hop 미해결 시 0.17 미달 예상.**

## ★★SOPBench in-dist eval (사용자 진단: 전이 아니라 in-dist도 떨어졌나?)
- **v6 in-dist online_market(N=10·step2599): Mean Pass 0.60·success 0.33·action-called 0.83·db-match 0.67·dirgraph 0.33.**
- **vs base 7B 0~21% → in-dist 안 떨어짐. 모델은 학습벤치 스킬 제대로 학습.** ⇒ **τ² 0.05는 "in-dist 미학습"이 아니라 *순수 전이 문제(R4)* 확정**(모델 정상).
- census 정합: in-dist 성공 본체=gather-to-decide+user-arg write(2-hop 1.9% 희소). **dirgraph 0.33 최약=시퀀싱이 in-dist도 부분학습=R4 전이 타깃**(step2599 중간·성장여지).
- 도구 = `scripts/distill/tau2/sopbench_indist_eval.sh`(run_simulation+SOPBENCH_VLLM_BASE_URL·OSS_MODELS+FCM[vllm] lora 등록). 채점 = `run_evaluation.py`(★output_v2/서 읽음·run_simulation은 output/에 씀 → **cp 필요**, 메모리 "run_evaluation 크래시" 원인).
- ⚠️ **coworker 파일 사고·복원**: SOPBench `swarm/constants.py`의 arm-3 FCM[vllm](qwen/llama 등록)이 *uncommitted*였는데 내 `git checkout`이 되돌림 → **arm-3 내용 복원 완료**(+v6tbox). SOPBench 클론 파일 `git checkout` 금지(coworker dirty). FCM[vllm]에 v6tbox 중복 1개=cosmetic.

## ★★커플링 실험(A) + 3번째 벤치 딥리서치(B) — 둘 다 v7=3벤치로 수렴 (2026-06-15)
- **A 커플링(v6 step2599·step3999·N=20·`coupling_eval.sh`)**: 두 점 모두 **SOPBench in-dist success 0.65·dirgraph 0.70(잘 배움)·τ² 0.0(전이 완전실패)**. = **현 상태 디커플링: in-dist 높아도 τ² 0**. (앞선 N=10 dirgraph 0.33=small-N 노이즈·N=20서 0.70 안정.) 기제 = τ²는 **(ii)2-hop id-lookup binding**(v6 auth는 grounding=부분전이·order_id fetch 못해 stuck). ⇒ **in-dist 시퀀싱↑가 τ²↑로 커플되려면 2-hop 선결**(census 논증→실증). caveat: 두 ckpt 근접→*궤적*아닌 *엔드포인트* 디커플링(저-dirgraph 초기점 미스냅샷). v6 τ²0.0<v4 0.10=honest-but-stuck(autopsy).
- **B 딥리서치(3번째 벤치)**: **1순위 Seal-Tools**(현실 서비스-API 엔티티·`API_call_N` 출력→arg·586 nested·Apache-2.0·gold JSON 결과합성·변환 LOW-MED·**value-randomize 필수**=심볼참조). 2순위 BFCL V3 multi-turn(grounded·rollout 필요). 보조 NESTful(수학만·gap불충족). 기각: ToolBench(CC-BY-NC=주권충돌)·AppWorld(REPL+Amazon중첩)·API-Bank(API검색≠값fetch).
- **⇒ v7 = SOPBench + TaskBench + Seal-Tools**(2-hop 소스). 헤드라인 = R4 커플전이 + 부재 2-hop 소싱. 커플링 재검증 = v7 후. 설계서 §8b 박제.
- 도구: `scripts/distill/tau2/coupling_eval.sh`(어댑터1개 SOPBench채점+τ² 동시). ⚠️lora명 재사용시 SOPBench output(ast_<name>) 충돌주의(순차 OK).

## ★★v7 = ComplexFuncBench(grounded 2-hop) 합류·발사 (2026-06-15)
- **결정(사용자)**: 3번째 벤치 = **ComplexFuncBench(논문용)**. Seal-Tools 드롭(단발-심볼형=TaskBench 동류·v6 불충분 입증). ToU = 논문 LOW·**특허/프로덕션은 clean 소스 재생성 필수**(우리 user-sim withholding 등·설계서 §8c-BLOCKING#2).
- **변환기 `fc_convert_complexfuncbench.py`**: conversations(user/assistant.function_call/observation/final)→native FC. functions→tools·observation→tool(녹화 Booking 응답)·parallel-call obs 분할. **850 traj(Flights 150 제외=airline 근접)·avg 5.02 call·observe→arg fetch-chain 100%**(`Search_Car_Location→obs{lat}→Search_Car_Rentals(pick_up_latitude=lat)` = grounded 2-hop·v6이 못한 그 스킬).
- **value-randomize**: cfb 3x 재-randomize(seed42/43/44=복사본마다 다른 랜덤값·암기불가) → copy 강제. fetchable-vals 4042/traj.
- **sft_v7 = sop_rand2 + d5_ask2 + tb_all_v4 + cfb×3 = 16054**(tb7000·sop6780·**cfb 2274~14%**). QC bad-args 294(randomizer가 cfb 복잡JSON ~11% 손상→드롭·v8 harden). cfb 토큰 median 9452·**23% >14336**(skip-overlong 드롭=최장 2-hop 손실·v8 value-aware truncation으로 회수).
- **발사**: v7 GPU1(PID 3997341·v4/v5/v6 동일 config). v6 GPU0 계속(in-dist baseline 앵커).
- **다음 ④**: v7 ep 후 `coupling_eval.sh`(SOPBench dirgraph+τ²) + `tau2_autopsy.py` — **핵심: order_id가 이제 observe→fetch되나(#W0000000 날조↓·get_user_details 호출↑)·τ²가 0 넘나.** 양성이면 "grounded 2-hop 소싱이 전이 갭 닫음" 헤드라인. 음성이면 어느 층(관찰·추출·의미매핑) binding인지 진단.
- 도구: `scripts/distill/taskbench/fc_convert_complexfuncbench.py`(repo). 데이터 = `/home/woori/scratch/ComplexFuncBench/`(HF zai-org).

## ★★2026-06-15 (세션재개) — zero-GPU 최고가치 2건 완료 (#2 도출 닫힘 + #4b census)
- **상태 확인**: v6(GPU0)·v7(GPU1) 학습 계속(세션재개 시 v6 step6200/v7 step600 → 작업 중 v6 6600/v7 1000). v7 cfb 장궤적=느림·loss 0.16~0.58 미수렴. **v7 eval(#3)은 step600에선 시기상조** → v6-성숙도(~step3000)까지 대기·백그라운드 모니터(step3000 도달 알림) 가동. 그동안 zero-GPU 진행.
- **★#2 대수적 도출 닫힘 완료**(매트릭스 §1.5b 확장 + 형식 companion `ALGEBRAIC_DERIVATION_CLOSURE_2026_06_15.md`): **층 A(control×data) = 구성상 닫힘**(Böhm–Jacopini+동시성·provenance 완전분할·흡수 메커니즘). **층 B = 유한 게이트-타입 상대 닫힘 = 유일 live seam(α)**. seam β(transform)=census로 해소(아래). 부산물: ①날조=¬P1의 구조적 위치 ②P2 lettering 정당화 ③P7 RL-필연성 ④**교차층 6+2+2**(P7=iter×verdict·P8=provenance×auth-gate가 가장 어려운 primitive=구조서 도출). coworker가 companion에 net-new 3 작성(흡수/seam화해/P8↔P1 merge)→내 census 실증·교차층 통합.
- **★#4b τ² primitive census 완료**(`tau2/tau2_primitive_census.py`·정적·zero-cost): 도구 분류=도메인 `tools.py`+`user_tools.py`의 `@is_tool(ToolType.WRITE)` 동적 파싱(반환시그니처 원칙). **전 τ² 도메인 orphan=0**: retail114·airline50·telecom2285(dual-control)·mock = **~2450 task 모든 gold 도구가 P1-P9 매핑·P10 없음.** "분류 밖 연산 0" 전수 실증.
  - retail 요구분포: P1 112·P2b **110(96%)**·P5/P6 **104(91%)**·P3 92·P8 66·P2a 52·P4 28·**P7 0(gold)/89 잠재·P9 0**. ⇒ **gap=P6+P7이 task17 아닌 전수 지배**(리뷰#4a/#4b 동어반복 탈출).
  - **seam β 해소**: 유일 변환도구 `calculate`(13)도 tool-call→P2b 환원 = in-model 변환 primitive 불요. **live seam=α(층B 게이트유한) 하나로 확정.**
  - **P7 구조부재 확증**: 전 도메인 gold P7=0(reactive·성공-gold에 deny 없음) = 도출 예측 census 독립확인 → SFT 소싱불가·gate-in-loop RL(리뷰#5).
  - ★telecom: device-actuation write(toggle_*/reboot)도 P5/P6 = GUI-인접이나 tool-call control/data-flow → 새 primitive 아님(scope §5 자인).
- **gotcha**: census 파서 = 멀티라인 def 시그니처(`def f(`+다음줄 `self`) 처리·`@is_tool(ToolType.X)` 데코레이터 타입으로 분류·dual-control은 `user_tools.py` 병합 필수(telecom 누락 시 device-write가 가짜 orphan).
- **다음**: ①v7 step3000 도달 시 `coupling_eval.sh`+`tau2_autopsy.py`(order_id fetch?·τ²>0?·#3 P2b/P4 ✓→✓!) ②적대탐색(#6)=층B 게이트유한 반증=out-of-genre P10 사냥(τ²동일장르 saturation=self-fulfilling) ③P6/P7 합성(리뷰 후).

## ★★2026-06-15 (세션재개 #2) — v6 정지·eval(정정)·P6 합성 실험(v8) 발사
- **v6 정지·최종 eval(step6950)**: 사용자 지시로 v6 트레이너 정지(GPU0 free)·`v6_eval_final` 스냅샷.
  - **★in-dist 향상**: SOPBench online_market success **0.65→0.80**·dirgraph **0.70→0.80**(더 학습=in-dist 계속↑·과적합 아님).
  - **★진짜 τ² = 0.10**(genuine·violations 0·no_reward 0). v4(0.10)와 동급. **디커플링 확정**: in-dist 0.80 ≫ τ² 0.10(<base 0.17). 더 학습해도 τ² 정체=2-hop binding(P2b) gap.
- **★★coupling_eval.sh 키-버그 발견·정정 (중요)**: `coupling_eval.sh`가 openrouter user-sim 키(`/home/woori/.openrouter_key`)를 source 안 해 τ² 전 task `AuthenticationError(401)`→`infrastructure_error`(0 calls)=**false 0.0**. **이전 coupling 런(v6_s2599/s3999)의 "τ² 0.0"도 전부 동일 키-버그 아티팩트**(autopsy n_auth_call=0 확인) — 진짜 모델 0 아님. (단 디커플링 결론은 유효: 키 source하는 `tau2_eval_adapter.sh` 진짜 τ²[v4 0.10·v5 0.105·v6 0.10]도 ≪ in-dist.) **수정**: `set +x; source key; SSL_CERT_FILE; set -x`(로그 노출 방지) 커밋 `6ac9187`. ⚠️gotcha: 원격 워킹카피 coupling_eval.sh가 0-내용(line-ending) M로 ff-pull 차단(launch 시 "중지함")→`git checkout -- <file>` 후 pull 필요.
- **★P6 confirm-gate 합성 실험(v8) — GPU0 발사**: 매트릭스 census 지목 gap(P6=91% write-task) 직격. **v7(P2b만·GPU1) ∥ v8(P2b+P6·GPU0) = P6 ablation.**
  - **`fc_confirm_augment.py`(신규·커밋)**: SOPBench FC 궤적에 confirm-then-write 주입(pos 902·neg 462). **★write 분류=반환시그니처**(궤적 tool-output shape: bare bool=write·tuple `(bool,val)`=getter) **+ read-술어 prefix 배제**(internal_/is_/check_… — QC 반증: `internal_is_loyalty_member→False`=read를 write 오분류·수정). get_loan→bare bool=write 정확. sop_confirm=**1364**(7도메인 실 action만·예: exchange_product·transfer_funds·get_loan·book_room).
  - **sft_v8 = sft_v7(16054) + sop_confirm(1364) = 17418**(7.8% P6·중복 안 함=암기회피·val-random 일관). config=v6/v7 동일. **GPU0 학습중**(`v8_train.log`).
  - **다음**: v8 ep0 후 `coupling_eval.sh`(키수정본) → v7 vs v8 P6 준수율(write 전 user-yes)·neg-준수율(no→미실행)·τ² 비교.
- **gotcha**: ①coupling_eval τ² 키 필수(위) ②P6 write분류=output-shape+read-prefix배제(prefix 단독 금지) ③sft_v8 셔플 빌드(`fc_build/sft_v8.jsonl`).

## ★★★2026-06-15 (세션재개 #3) — v7 결정적 eval = NEGATIVE (CFB 2-hop 전이 실패·gen_synth_2hop 게이트 발동)
- **v7 정지(step7100)·키수정 coupling_eval로 결정적 eval(`v7_eval_s7050`·GPU1)**:
  - **in-dist 최고**: SOPBench online_market success **0.90**·dirgraph **0.95**(v6 0.80·v4보다↑·v7 in-dist 완전학습).
  - **★τ² = 0.05** (genuine·auth_errors 0·violations 0). **v6(0.10)/v4(0.10)보다 낮음·base 0.17 미달.**
- **★★결정적 진단 = grounded 2-hop CFB가 τ² 전이 실패 (NEGATIVE·사전등록 예측 음성분기)**:
  - **order_id/product_id 날조 여전**: `get_order_details` 64회 중 **44회 `#W000` 날조**·`get_user_details`(fetch 경로) **8회뿐**. autopsy: fab_auth 10·agent_collapse 9·PASS 1·auth provenance grounded 9/fab 10(v6 step2599 grounded 18/19보다 **퇴행**).
  - **궤적 확정(task1)**: product_id `6086499569` 날조→"not found"→**동일 호출 10연타**(전략전환 0)→too_many_errors. 인증·주문fetch 없이 ID 지어내 루프.
  - **기제**: CFB = *linear observe→use*(값이 출력에 이미 있어 복사). τ² = *proactive gather*(없는 arg 인지→**생산 getter 능동선택**→호출→출력서 **select**). **'생산도구 능동선택+select'(R4) 층이 미학습·CFB가 안 가르침.** = 데이터-소스 문제 아닌 **R4 의미-전이** 문제.
- **★gen_synth_2hop(Path B) 게이트 발동 = 짓지 않음** (SYNTHESIS_IMPL_SPEC §C 사전등록: "CFB 전이 안 되면=문제는 데이터-소스 아니라 R4 의미-전이→같은 primitive 다른 소스도 무효→진단 선행"). **매몰비용 회피 확정.**
- **★재방향(잠정·전수 census로 정정됨 아래 ★★ 참조)**: ~~P7이 차기 최고가치(retry-loop 9/20)~~ → **틀림. 아래 root-cause census가 P7 기각·P2b(스키마-example 날조)로 정정.**

## ★★★2026-06-15 (세션재개 #4) — 전수 root-cause census = 근본 P7 아니라 P2b(스키마-example 날조) 확정·정정
> 사용자 지시("전수 궤적 조사해 P7 문제인지 정확히 재확정"). `tau2_rootcause_census.py`(신규·per-traj 첫에러+에러후행동). **이전 #3의 "P7 차기 최고가치"를 기각·정정.**
- **★root 분포(n=20)**: **날조-trigger(P2b) 17/20**(auth_fab 7·fab_then_switch 8·fab_then_loop 2)·gate-trigger 2·pass 1. **에러후 행동: 동일호출 하드루프(P7 미작동) 3뿐·다른시도(P7 작동) 9.**
- **★근본 = P2b 'fetchable 값 날조-FIRST'(P7 아님)**: 모델이 없는 값에 **τ² tool 스키마의 example 값을 복사**(`tools.py`: `order_id ... such as '#W0000000'`·`email ... such as 'something@example.com'`) → `#W0000000`(order_id)·`jane_doe@example.com`(email·7회)·`6086499569`(product_id) 날조. = R1/P1 provenance 위반(스키마 example은 합법 소스 아님).
- **★결정적 반례(task6 dump)**: 모델은 proactive 2-hop gather도 P7 복구도 *할 줄 안다* — email 날조→"not found"→**name+zip 요청(P7 복구 작동)**→성공→order_id `#W0000000` 날조→"not found"→user "다른 방법?"→**`get_user_details`→`get_order_details('#W6390527')` 진짜 order_id로 성공(2-hop gather 작동)**. **문제=날조를 *먼저* 하는 기본행동**(턴 낭비→user_stop 전 미완성)이지 능력부재 아님.
- **★정정된 처방 우선순위**: ~~P7 recovery~~ **기각**(9/20 작동·하드루프 3뿐). **진짜 타깃 = 날조-FIRST 차단 = R1b provenance**(arg값 ∈ {user 발화, tool 출력}만·**스키마 example 값 거부**) + D5 fetch-first를 *전* fetchable값(order_id·email)으로 확장(현 D5/value-random은 identity만). 디코드-제약(스키마-example 블록) or 학습(없는값→gather/ask-first 기본화). [[R1B_PROVENANCE_DESIGN_2026_06_14]]가 정조준.
- **함의**: CFB(v7) 실패도 이 렌즈서 재해석 — CFB는 observe→use 가르치나 **'없는 값→날조 안 하고 gather'**를 안 가르침(스키마 example 유혹은 inference-time). 도구=`scripts/distill/tau2/tau2_rootcause_census.py`.

## 인프라 메모(구)
- **매트릭스 갱신**: cfb P2b = 데이터존재 ✓ but **전이 ✗(검증됨)** — 리뷰#3 ✓!→✗! 전환. gap 재확정 = P2b(R4 의미전이)·P6·P7.
- 도구: 결과 = `coupling_v7_s7050.log`·`retail_v7_s7050`·autopsy `tau2_autopsy.py`.

## 인프라 메모
- ⚠️ **모든 스크립트/문서 = git push/pull 전송**(사용자 지시 2026-06-14). 리모트는 pull만. eval 드라이버도 repo(`scripts/distill/tau2/tau2_eval_adapter.sh`). base64/직접전송 금지.
- ⚠️ eval 드라이버 `set -x` + `source .openrouter_key` → 로그에 키 노출. 차후 키 라인 `set +x`로 감쌀 것.
- ⚠️ git: 원격 워크스페이스 cat-append 커밋이 백틱 명령치환 + rebase 충돌 유발 → **진행로그는 로컬 클론서만 편집**(원격은 pull). 원격 dirty(offload_*.sh)는 coworker 것 — 건드리지 않음.
- ⚠️ git: 원격 워크스페이스 cat-append 커밋이 백틱 명령치환 + rebase 충돌 유발 → **진행로그는 로컬 클론서만 편집**(원격은 pull). 원격 dirty(offload_*.sh)는 coworker 것 — 건드리지 않음.
