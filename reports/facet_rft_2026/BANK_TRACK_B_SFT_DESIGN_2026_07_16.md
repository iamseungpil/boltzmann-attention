# Track B 상세 설계 — F3 스키마-분류 스킬 SFT (전부 무료·리모트 GPU·2026-07-16)

> ⚠️ **2026-07-16 재프레임(사용자·최우선)**: Track B는 **SFT 아니라 LOOP**로 먼저 시도. F3 실패=검색결과와 비교 안 하고
> prototype(fraud)로 one-shot 점프 → **COMPARE-or-ASK 결정론 loop**(GET/FIND 후보→후보별 y/n 격리비교→유일매칭 select·else ASK)가
> 구조적으로 강제(학습 불요·[[10]]/[[13]] scaffold-before-learn·프레임 F2 "격리 sub-call+결정론 실행" 동형). **Track A와 한 loop로 통합.**
> **다음 = `bank_f3_eval` per-candidate 비교 loop 확장→32B로 fraud-collapse 깨지나 테스트(무료).** 깨지면 F3 loop로 닫힘=SFT 불요.
> 아래 SFT/synth 설계(§1-7)는 **loop가 실패할 때의 fallback**으로 강등. synth v0 무효(§6.2)는 유효.


> 사용자 지적 정정: **Track B는 유료 아님**. SFT=리모트 A6000·eval=로컬 vLLM·**user-sim(gpt-5.2) 안 씀 = API 0 = 무료**.
> few-shot 실험이 표적을 좁힘(§0). 입력: C99 base-eval·few-shot(dispute_reason 98% fraud mode-collapse·프롬프트 무효)·[[11]]/[[42]]/[[12]].

## 0. 표적 (few-shot 실험이 확정)
- **닫는 것 = 강한-prior 서사 enum**(dispute_reason형): 프롬프트(정의·anti-prior·few-shot 반례) 전부 무효·98% "fraud" mode-collapse. **사실-도출 enum**(dispute_category)은 few-shot로 이미 열림(55→81.7%)=**제외**.
- **스킬(도메인일반)**: "제공된 enum 스키마 정의를 읽고 NL 분류·salient prior로 안 덮기". banking엔 ABox-swap 전이(스키마=eval서만 공급·학습에 0·[[11]]).

## 1. 비용 (정정)
| 단계 | 자원 | 비용 |
|---|---|---|
| synth 생성 | 로컬/리모트 CPU | 무료 |
| LoRA SFT + DPO | 리모트 A6000 | **무료**(자체 GPU·API 0) |
| 전이 eval (bank_f3_eval) | 리모트 vLLM | **무료**(user-sim 없음) |
| (선택) tau2 e2e 최종확인 | gpt-5.2 user-sim | 유료([[09]]·make-or-break 아님) |
- ⚠️ **GPU 제약**: 두 A6000 ~44.5GB 점유(vLLM 8140/8141). 32B LoRA 학습 = vLLM 하나 정지(GPU 확보) 필요·[[30]] 조율.

## 2. 학습 데이터 = 도메인일반 스키마-분류 (벤치·banking 미학습·[[11]])
- **생성기**: 다양한 *합성 taxonomy*(banking 아님) × NL 상황 → 정답 enum. 각 taxonomy = 5~10 카테고리 + 정의.
  - 예 도메인: support-ticket 유형·product-defect·insurance-claim·HR-request·content-moderation 등 — **다도메인**([[12]] 다양성).
- **★필수 = prior-conflict 케이스**(few-shot 실험이 이게 핵심임을 실증): surface-plausible(직관·salient) ≠ 정의상 정답. 강한-prior 유발 카테고리(각 taxonomy에 "가장 흔한/직관적" 카테고리를 두고, NL이 그걸 암시하나 정의상 다른 답).
- **다양성([[12]])**: taxonomy 구조·카테고리 수·NL 표현·prior-conflict 유형 변형. 단일템플릿 금지(표면매핑 역전이).
- **재사용**: `t2_a2_concrete_gen`·`cfbsynth_v2`(합성 스캐폴드)·`t2_formalize_exec`(NL→formalize 골격) 확장.

## 3. 방법 ([[42]] 처방)
- **SFT**: (schema 정의 + NL) → 정답 enum. diverse synth. LoRA 32B(`lora_train_metatool_v3` 재사용·진행률 가시 [[30]]).
- **prior-suppression DPO/NPO**: pair (정답 enum) ≻ (prior-default enum). `cfbsynth_dpo_pairs.py` 패턴 재사용. mode-collapse의 salient-default에 페널티.
- SFT 먼저(스킬 설치) → DPO(prior 억제). 각 단계 후 eval.

## 4. 전이 검증 (무료·make-or-break)
- SFT/DPO'd 32B → **`bank_f3_eval`**(banking F3·banking 스키마 학습에 0·held-out 전이).
- **지표**: ① dispute_reason 정확도 base 35%→? (majority 39% 초과·98% fraud mode-collapse 붕괴) ② 예측분포(fraud 편중 해소) ③ dispute_category 무회귀(≥55%) ④ 미학습 banking 스키마 전이.
- **대조군**: base 32B(35%/55%)·zero/strict/few-shot(전부 35%).

## 5. 성공기준·make-or-break
- **GO**: dispute_reason > majority·fraud-편중 붕괴·미학습 banking 전이·dispute_category 무회귀. = **소형+학습 스킬이 프롬프트-불가 F3를 연다**([[41]] 헤드라인·frontier도 이 스킬 없음).
- **NO-GO**: SFT 후에도 mode-collapse 지속 or banking 전이 실패(과적합) → F3 강한-prior=진짜 경계(learn 축까지 닫힘)·명제는 결정론+사실-도출-F3(few-shot)로 유지.
- **부분**: dispute_reason 개선하나 <frontier — 경계-완화 정량.

## 6. 순서 (전부 무료 리모트)
1. **synth 생성기 v0**(다도메인 taxonomy + prior-conflict·로컬 무료·다양성 QC).
2. **base eval on synth**(held-out synth 스키마·SFT 前 baseline·prior-conflict서 mode-collapse 재현 확인).
3. **GPU 확보**(vLLM 하나 정지·[[30]]) → **LoRA SFT** → synth eval → **DPO** → synth eval.
4. **전이 eval**(bank_f3_eval·banking held-out) = make-or-break.
5. (선택·유료) tau2 banking e2e 최종.

## 6.2 ★synth v0 검증 = banking 실패모드 미재현 (중요 [[08]] 발견·2026-07-16)
`synth_schema_classify.py`(5 도메인일반 taxonomy·generic evoker prior-conflict·held-out) v0를 base 32B로 검증(§6.2 게이트·리모트 8140·n=200):
- **clear 100%·prior-conflict 100%·salient-default 예측율 0%**. ⇒ **base가 synth를 완벽히 풂 = banking의 98% mode-collapse 미재현.**
- **원인**: synth 판별자 너무 명시적(즉시 정답)·NL 짧고 clean. banking = (i) NL 길고 노이지(대화 1500자·판별자 매몰) (ii) "fraud" 금융맥락 극강 prior (iii) 판별자 미묘(not_as_described↔fraud=정책 뉘앙스).
- **★함의(fork)**: banking F3 실패가 **도메인일반 스킬 갭**(→synth 강화로 학습)인가 vs **banking-특화 강한-prior**([[11]] 긴장)인가. **synth가 실패모드 재현 못하면 Track B 전이 실험 무의미**(잘못된 스킬 학습).
- **다음 옵션**: (a) synth 강화(salient-framing + 노이즈/길이 + 미묘 판별자로 collapse 유발) (b) banking collapse 원인 규명(길이/노이즈/prior강도 ablation) (c) Track B 전제 재검. **v0 학습 착수 보류**(synth 재현 확인이 선결).
- ★[[08]]/[[12]] 성과: 학습 前 synth 검증이 무효 학습을 차단(guard 규율).

## 8. ★★online-H_min 결정 모듈 실측 (compare loop 재설계·2026-07-16·C100·[M] n=160)
> handoff §0 compare loop을 구현·실행 → [[08]] 포렌식이 측정오염 발각 → 권위본(§14~§18) 대조 후 **재앵커 + online-H_min 결정기**로 재설계·재측정. 정본 수치.

### 8.1 [[08]] 측정오염 발각·교정 (재앵커)
- **v1 compare loop**(후보별 격리 이진 y/n → union → 유일매칭 select·else ASK): n=40서 **dispute_reason 90% ASK·0% correct**.
- **per-case 포렌식**: dispute_reason 케이스 **100% txn-없음**(dispute 액션이 transaction_id만 담고 denormalize 필드 없음) + NL=전체 multi-dispute 대화(1500자 clip). ⇒ "이 *거래*의 reason?"이 아니라 "여러 dispute 섞인 blob이 reason X와 맞나?"를 물음 = 이진 many-yes가 정당 = **F3=⋈참조앵커링 표면화**(§14.3·C79). one-shot fraud-collapse(35%/98%)도 같은 오염.
- **교정**: `parse_txn_records`(tool결과 레코드 transaction_id-join)로 txn 앵커 **0%→92%** 주입.

### 8.2 online-H_min 결정기 (이진→등급·§16/§18)
- 이진 y/n은 **margin을 버려** union→과-ASK. → **등급채점**(definitely/probably/unlikely/no=3/2/1/0·후보별 격리·앵커 강조로 multi-dispute NL scope).
- 분포 형성 후 **margin threshold**(§18.1 엔트로피-게이트): top1−top2 ≥ margin(그리고 top1>0) → **SELECT top1**(DERIVE, ASK불요) · else(동점/전부no) → **ASK**(진짜 잔여 애매성·bounded). 도메인일반([[05]]·[[10]] LLM=채점기·결정=결정론).
- 구현: `bank_f3_eval.py --hmin --margin N`. `run_hmin`.

### 8.3 결과 (n=160·각 필드 n=80·margin≥1·리모트 32B 8140)
| 필드 | correct | ASK | wrong | margin 분포 {gap:n} |
|---|---|---|---|---|
| dispute_reason (강한-prior 서사) | 28% | 58% | **15%** | {0:46, 1:30, 2:4} |
| dispute_category (사실-도출) | 39% | 56% | **5%** | {0:45, 1:10, 2:25} |
- SELECT 분포 dispute_reason: {fraud:12, incorrect_amount:7, duplicate:6, reversal:5} · ask-recoverable(gold∈tied): reason 61%·category 58%.

### 8.4 안정 신호 3 + 정직 종합
1. **fraud-collapse 깨짐**(fraud 40%·98%아님·확신-오답 65%→소멸)=견고.
2. **dispute_reason 시그니처=wrong율 3배**(15% vs 5%)=잔존 prior간섭(§17.3 miscalibration)=진짜 SFT표적.
3. **★자기교정([[08]])**: dispute_category **few-shot one-shot 81.7% ≫ H_min 격리채점 39%** → 격리채점은 파생필드엔 열등(후보-간 비교맥락 상실) = **H_min 모듈은 category엔 틀린 도구**.
- **H_min 정합(사용자 지적 해소)**: dispute_reason 58% ASK=위반 아님(§16.2 dispute_reason=CREDIT ASK필드 2.49bit≈질문1개·모듈은 필드당 ASK 1개·union 남발 아님·user_stop 예산 내).
- **정직 결론**: online-H_min 가치=파생필드 닫기 아니라 **강한-prior 확신-오파일링을 bounded-ASK로 전환**(few-shot 35%정답/65%확신-오답 → 28%정답/**15%만 오답**/58%질문1개=오-파일링 비싼 실환경서 net-safe). 올바른 컨트롤러 = **category→few-shot derive·reason→online-H_min ASK**(§16.4 verify-or-ASK 구체화).
- **n=40 과대 회수([[08]])**: 초판 "category H_min으로 닫힘 50%"=과대·실제 39%·둘 다 ~57% ASK로 자율 미닫힘.

### 8.5 전략·다음
- **전략(§14.5/14.8)**: dispute_reason=작은 slice(지배 레버=coverage 25.6%·compute 16.7%·⋈ 4%) → Track B SFT-on-reason 상한 제한적(닫아도 banking 헤드라인 거의 무변). ask-recoverable 61%(서사에 답 있음)=SFT 여지나 전략 우선순위 낮음.
- **잔여 15% wrong=calibration** → §18.3 conformal(작은 calib set·보장된 애매성) 또는 SFT.
- provenance: `f3_hmin_n160.log`(리모트 `scratch/f3_results`·gzip)·`bank_f3_eval.py`(`parse_txn_records`·`run_hmin`·재앵커 f3_cases.jsonl 커밋). caveat: 실패-sim 편향(FLOOR)·margin=1 미튜닝(risk-length knob §16.1).

### 8.6 ★비교랭킹+confidence-gate ablation = confidence가 reason collapse 못 막음 (2026-07-16·C100 후속·[M] n=160)
> C100 자기교정("격리채점=비교맥락 손실")을 검정: 격리 등급채점 대신 **비교 one-shot(전체 enum 대령) + confidence-gate**(PICK+CONFIDENCE·low→ASK). `bank_f3_eval.py --rankconf`·`f3_rankconf_n160.log`.

| 필드 | correct | ASK | wrong | (격리 H_min §8.3) |
|---|---|---|---|---|
| dispute_reason | 42% | 38% | **20%** | (28 / 58 / 15) |
| dispute_category | 56% | 34% | 10% | (39 / 56 / 5) |
- SELECT 분포: dispute_reason **{fraud:31, incorrect:7, duplicate:7, goods:3}=fraud 65% 재-collapse** / dispute_category {duplicate:16, incorrect:10, atm:9, card_present:7}=분산.
- confidence 분포: reason {low:30, med:7, **high:43**} · category {low:27, med:7, high:46}.

**두 가설 확증**:
1. **비교맥락이 category 회복**(56% vs 격리 39%)·confidence-gate 작동(고-conf 대개 정답)·ASK 34%. ⇒ **사실-도출 F3엔 비교 one-shot+confidence가 실용적**(격리채점보다 우월).
2. **비교 one-shot이 reason을 confident-fraud로 재-collapse**(SELECT 65% fraud·42% correct는 fraud=다수 gold class 착시·20% wrong).
3. **★결정적 = confidence가 reason collapse 못 막음**: reason conf=high 43건인데 **20% wrong = confident-wrong**(§17.3 spurious minimum에 뾰족한데 틀림). confidence-gate/margin 어느 프롬프트 방법도 강한-prior 필드 못 구제.

**종합(Track B 표적 선명화)**: dispute_reason 병 = **prior에 대한 miscalibrated 과확신**(확신하며 틀림) → [[42]] prompt-ceiling 재확증 + 정확한 **SFT+DPO(prior-suppression)** 또는 **conformal(§18.3·분포무가정 보장된 애매성)** 표적. 반면 category(사실-도출)=비교 one-shot+confidence로 닫힘(scaffold). **최종 필드타입별 처방**: category→비교 one-shot+conf-gate(56/34/10) · reason→(현재) 격리 H_min bounded-ASK or (표적) SFT/conformal.

## 7. 규율 가드
- [[11]] 벤치(synth)서만·banking 스키마 학습에 0(eval서만)·전이=ABox-swap. [[12]] 다양성 필수·단일템플릿=역전이. [[42]] SFT설치+DPO. [[30]] 진행률 가시·결과 gzip 영속·GPU 충돌금지. [[05]] 스킬=도메인일반·엔진 리터럴0. [[08]] SFT 후 예측분포 전수(mode-collapse 붕괴 실증)·집계직행 금지.
- **모트 계측**: 과-분류(prior 억제 역효과=over-correction) 계측·held-out 역전이 0 확인.
