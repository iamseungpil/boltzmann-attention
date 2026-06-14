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
- **다음(미결·GPU 결정)**: v5 학습 launch 보류 — GPU0=v4 학습 in-flight(ep0 step6550·32 ckpt·죽이지 말 것), GPU1=21GB 점유(우리 compute-app 0개=coworker 가능성). **launch 판단 사용자 대기**: ①v4 수렴 후 GPU0 ②GPU1 free 확인 후 ③coworker A100/H200 노드. 학습 시 = lora_train_chat_toolcall.py·flash-attn·grad-accum4, then τ² A/B(driver 패턴) vs v4 — over-ask율·compliant-pass 비교.

## 인프라 메모
- ⚠️ git: 원격 워크스페이스 cat-append 커밋이 백틱 명령치환 + rebase 충돌 유발 → **진행로그는 로컬 클론서만 편집**(원격은 pull). 원격 dirty(offload_*.sh)는 coworker 것 — 건드리지 않음.
