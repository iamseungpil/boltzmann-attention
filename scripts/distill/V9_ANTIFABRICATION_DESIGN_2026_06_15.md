# V9 설계 (리뷰용 DRAFT) — fetchable 값 날조 제거: randomization + DPO/RLVR (검증기 through-line)

> 상태 = **리뷰용 DRAFT**(승인 후 구현). 상위 = `R1B_PROVENANCE_DESIGN_2026_06_14.md`(L1/L2/L3 3층)·`SYNTHESIS_IMPL_SPEC_2026_06_15.md`(P6/P7). 진입점 = `HANDOFF_2026_06_15.md`. 불변 = memory `feedback-thesis-tbox-transfer-direction`(SOPBench/TaskBench서만 학습·τ² held-out 전이)·`feedback-selector-verifier-deterministic`(검증기=결정론·LLM은 생성기만).
> 목표 = v7 NEGATIVE의 확정 근본원인(P2b 'fetchable 값 날조-FIRST')을 **학습으로 내재화**해 τ² 전이 0.05→돌파.

## 1. 동기 — 확정된 근본원인 (이번 세션 실증)
- **v7(ComplexFuncBench grounded 2-hop) eval = NEGATIVE**: τ² pass^1 **0.05**(v4/v6 0.10보다↓·base 0.17 미달)·in-dist는 최고(success 0.90·dirgraph 0.95). ⇒ grounded 2-hop 데이터-소스 추가만으론 전이 안 됨.
- **전수 root-cause census(`tau2_rootcause_census.py`·n=20)**: **날조-trigger 17/20**(auth_fab 7·fab→switch 8·fab→loop 2)·gate 2·pass 1. **에러후 동일루프(P7 미작동)=3뿐·다른시도(P7 작동)=9.** ⇒ **근본=P2b 'fetchable 값 날조', P7 아님.**
- **날조 정체 = τ² tool 스키마 example 값 복사**: `tools.py`의 `order_id ... such as '#W0000000'`·`email ... such as 'something@example.com'` → 모델이 값 없을 때 그 **example을 실제 인자로 복사**(`#W0000000` 44/64·`jane_doe@example.com` 7회). = R1/P1 provenance 위반.
- **★결정적 반례(task6 dump)**: 모델은 **proactive 2-hop gather도(get_user_details→진짜 `#W6390527`) P7 복구도(name+zip 요청) *할 줄 안다*.** 문제 = **날조를 *먼저* 하는 기본행동**(턴 낭비→user_stop 전 미완성). = 능력부재 아닌 **나쁜 default ordering**.
- **함의**: 처방은 "새 능력 주입"이 아니라 **"날조-FIRST 기본행동을 fetch-FIRST로 교체"**. 날조는 *알려진* 값(스키마-example)서 시작 → 부정 신호로 직격 가능.

## 2. 핵심 명제 (처방 = 긍정+부정 결합)
- **긍정(어디서 값을 얻나)**: 전 fetchable 값을 **format-보존 randomize + tool 출력에만 등장** → 암기 불가 → **fetch-then-copy 강제**(맞는 값 내려면 getter 먼저). [현 value-random은 identity만 → 확장.]
- **부정(무엇을 쓰면 안 되나)**: **스키마-example/날조 값 사용에 페널티** → DPO/RLVR. SFT-randomization은 *부정 예시를 못 봄*(randomized 데이터에 스키마-example 부재)이라 inference-time 유혹에 약함 — 부정 신호가 그 갭을 메움.
- **★검증기 = through-line(결정론)**: provenance 검증기(arg값 ∈ {user 발화 ∪ tool 출력}? 스키마-example/날조면 reject)가 **①런타임 가드(prototype) ②DPO 라벨러 ③RLVR 보상** 세 곳에 재사용. LLM-judge 아님 = 불변 정합.

## 3. 구성 요소

### A. provenance 검증기 (결정론·보상/라벨/가드 공용)
- 입력 = tool_call(name, args) + 대화 컨텍스트. 판정 = 각 identifying arg값(order_id·email·payment_method_id·item_id 등)이 컨텍스트에 등장? 없으면 fabricated.
- 구현 존재 = `t2_gate_patch.py`(`_provenance_deny`·`_args_dict` robust). 강화 필요: ①PROV_ARG_HINT 커버리지(전 fetchable arg) ②**스키마-example literal 명시 blocklist**(`#W0000000` 등 = false-positive 0 고정밀) ③false-positive 감사(컨텍스트에 있는데 포맷 달라 오판하는 경우).
- **출력 신호**: `is_fabricated`(부정 보상)·`is_provenance_clean`(긍정)·`used_getter_before_arg`(fetch-first 여부).

### B. L1 런타임 (bad_words 디코드-마스크) — prototype·검증완료
- vLLM `bad_words`(extra_body·feasibility PASS: `#W0000000` 디코드 차단 실증). 정적 블랙리스트(스키마-example·placeholder 16개) + 동적(검증기 flagged → 세션 블랙리스트). 변형(`#W0000001`)은 검증기+동적-추가+내부재생성으로 ratchet.
- **역할 = 학습 아님·가설 검증/프로덕션 가드**. "날조 막으면 pass 오르나"를 무재학습 측정 → v9 학습투자 정당화. (3-arm BASE/L1/L1L2 eval 진행중.)

### C. 확장 value-randomization (SFT 긍정·`fc_randomize_fetchable` 확장)
- 현 = identity(email·name)만. **확장 = order_id·payment_method_id·product_id·item_id·address 등 전 tool-fetchable 값**을 format-보존 랜덤치환(traj-고유·md5 류).
- **불변식**: 랜덤값이 **user 발화엔 없고 tool 출력에만** 존재 → 모델이 맞추려면 **반드시 getter 먼저 호출→출력서 복사**. = 날조 차단 + fetch-first가 *한 번에* 구조적 강제.
- 소스 = SOPBench(τ²와 구조 최근접) 우선 + TaskBench threading. 3x 재-randomize(복사본마다 다른 값=암기불가).

### D. DPO (부정·합성 쌍·1순위·싸다)
- **쌍 합성(결정론·날조0)**: SOPBench fetch-then-use 궤적에서
  - **chosen** = getter 호출→출력서 진짜 값 복사→사용 (원본).
  - **rejected** = 동일 궤적, 단 그 값을 **스키마-example/placeholder로 치환**(`#W0000000`·`something@example.com`) = 날조 버전.
- 검증기가 자동 라벨(chosen=clean·rejected=fabricated). **양방향**(R5·DPO v1 단방향 net− 교훈).
- 변형: rejected에 "묻기-먼저"(fetch 가능한데 ask)도 포함 → over-ask도 페널티(D5 정합).

### E. RLVR (부정·on-policy·Track B·더 강함)
- rollout(SOPBench-류) → **결정론 보상 = (task 성공) ∧ (날조 인자 0) ∧ (fetch-before-use)**. GRPO.
- = **gate-in-loop RL**(검증기/게이트가 보상 루프 안) = FIELD_GAP Track B 재진입. P7(reactive) 처방과 동일 기제.
- **reward-hacking 가드**: "날조 없음"만 보상하면 *호출 회피=over-abstain*. ⇒ task-성공 동시 요구 필수. 부분보상=fetch-first 단계 shaping.

## 4. 학습/데이터 계획 (전이 규율 준수)
- **학습 = SOPBench(+TaskBench)서만.** order_id-류 fetchable 값 randomize + 스키마-example-형 negative. **τ² held-out**(거기서 학습 금지·전이만 측정). [[feedback-thesis-tbox-transfer-direction]].
- **v9 레시피(잠정)** = sft_v7 베이스 + **확장-randomize SOPBench**(C) + **DPO 쌍**(D) → (양성 시) **RLVR**(E). P6 confirm(v8)·P7 recovery는 직교 합류.
- 베이스 모델 = Qwen2.5-7B(동일). config = v6/v7 계열.

## 5. 평가 / 사전등록 (census-tier 보조 + 헤드라인 τ²)
- **헤드라인**: τ² pass^1 (held-out·키 source 필수·`coupling_eval` 키수정본).
- **기제 지표(사전등록)**: ①**fab율**(스키마-example/날조 인자 호출수·`tau2_rootcause_census.py`) ②**fetch-first율**(getter-before-arg) ③auth provenance grounded율 ④과수집/루프율.
- **사전등록 예측**: 확장-randomize+DPO → fab율 17/20→↓·fetch-first율↑·τ² 0.05→base(0.17) 접근. RLVR 추가 시 추가 개선. **단 write-완성(payment_method_id·과잉거부=task6 천장)은 별 처방 필요할 수 있음**(정직 천장).
- ablation: randomize-only vs +DPO vs +RLVR (각 신호 기여 분리·#1 leave-one 정신).

## 6. scope / caveat (정직)
- **prototype(L1/L2)는 가설검증·프로덕션 가드**이지 thesis 핵심 아님 — 핵심 = **학습 내재화(C/D/E)**.
- **transfer 미보장**: v7(CFB)도 grounded-2hop+randomize였으나 전이 실패 → randomization *필요조건이나 충분 아닐* 수 있음(다른 도구형식·스키마-example 유혹). **부정 신호(D/E)가 그 갭의 가설**. 음성이면 = 더 깊은 R4 의미전이 진단.
- **write-완성 천장**: 날조 차단이 *too_many_errors 타입*은 살려도 *user_stop/write-미완 타입*(task6)은 별도(P6 confirm·payment fetch·과잉거부). 천장 명시.
- 보상=결정론 검증기라 LLM-judge 음성이력 회피·재현성 확보.

## 7. ★열린 질문 (리뷰 훅)
1. **randomize-only로 충분한가, DPO/RLVR 필수인가?** — v7 실패가 "randomize 불충분"인지 "도구형식 전이갭"인지. ablation 설계로 분리(C-only vs C+D).
2. **DPO vs RLVR 순서/필요성** — DPO(싸고 합성쉬움) 먼저가 맞나, 아니면 reactive 행동은 RLVR(on-policy)이라야 전이되나? (R5: 양방향 on-policy 강조 → DPO도 on-policy 쪽이 나을 수도.)
3. **rejected 합성의 충실성** — 스키마-example 치환만으로 충분한 negative인가, 아니면 model-rollout서 실제 날조를 negative로 채굴해야(on-policy distribution 일치)?
4. **스키마-example 일반화** — 날조는 inference-time τ² 스키마서 옴. SOPBench 학습이 *못 본* τ² 스키마-example을 어떻게 거부 일반화하나? (학습 신호 = "스키마 텍스트의 값은 무효"라는 *추상 규칙*을 배워야 — 특정 literal 암기 아님.) 이게 핵심 전이 가설.
5. **검증기 false-positive 비용** — 학습 보상으로 쓸 때 오판(legit 값 reject)이 모델을 잘못 학습시킴. 감사·고정밀(스키마-example literal blocklist) 선행.
6. **write-완성 천장 분리** — v9는 P2b 전담, write-완성(P6/payment)은 v8/별도. 헤드라인서 혼선 방지.
7. **★(M0 신설) 날조가 유일 병목 아님** — prototype 실측: 날조 차단해도 gather-완성 6/20·write-완성 1/20. ⇒ anti-fab 단독 pass 기여 제한적. **v9 = 전-체인(gather→use→write) RLVR(보상=task성공) 필요? 아니면 anti-fab(v9) + write-완성(v8/P6/payment) *동반*?** 이게 최대 설계 결정.

## ★M0 결과 (2026-06-15·3-arm prototype 완료) — 메커니즘 확정·but pass 정체·천장=다단계 복합
| arm | pass^1 | 날조(#W000/email) | get_user_details |
|---|---|---|---|
| BASE | 0.05 | 72 | 10 |
| L1(bad_words 정적) | 0.05 | 58 | **44** |
| L1L2(+동적+regen) | 0.0* | **17** | 19 |
- **★레버 작동 확정**: 날조 차단 → **gather로 전환**(get_user_details 10→44·날조 72→17). 행동이 정확히 의도대로 바뀜.
- **★그러나 pass 정체(~0·*n20 노이즈*)**: L1 arm 정밀분석 — **날조-trigger 여전 17/20**(정적 bad_words가 `#W0000000`은 막아도 모델이 **변형 `#W0000001`·새 이메일 날조** = 정적 구멍 → 동적 L2 필요)·**gather 성공(read≥2) 6/20·write 매치 1/20**.
- **★확정 함의(V9 재형성)**: **날조 차단은 필요하나 *불충분*.** τ² 천장 = **다단계 복합**: ①날조-변형(동적 차단 필요) ②**gather-완성**(맞는 값 추출·6/20만) ③**write-완성**(payment_method_id 2-hop·과잉거부·1/20=task6). 날조만 막으면 병목이 *다운스트림으로 이동*하지 pass로 안 감.
- **⇒ V9 방향 조정**: 단일 anti-fab(C-only/D) 아니라 **전-체인 신호**(gather→extract→use→write 전체 정확)가 필요 → **RLVR 보상 = task 성공**(날조-무만 아님)이 전-체인을 on-policy 최적화 = 사용자 RLVR 아이디어 강화. randomization은 fetch-first 강제(긍정)·DPO는 변형-날조 페널티(부정)·**RLVR이 전-체인 묶음**. ⓠ"날조가 *유일* 병목 아니라면 v9 anti-fab의 pass 기여는 제한적 — write-완성(P6/payment)과 *동반* 필요"를 리뷰 훅 #7로.

### ★★M0 체인-census (전수·`tau2_chain_census.py`) — 진짜 병목 = write 단계의 P6+P7 (확정)
| 단계 통과 | BASE | **L1** | L1L2 |
|---|---|---|---|
| auth | 13 | **17** | 9 |
| gather | 8 | **12** | 9 |
| real_order 추출 | 8 | **12** | 9 |
| write 시도 | 7 | **10** | 8 |
| **write 성공(PASS)** | 1 | **1** | 0 |
- **★①anti-fab가 상류를 뚫음**: L1이 auth·gather·추출·write도달 *전부* 개선(7→10 write도달) = **날조 차단이 더 많은 태스크를 write까지 보냄**(레버 유효).
- **★②gather는 100% 생산적**: gather한 태스크 전부 real-order 추출·**gather 후 날조 0**. 추출/select 실패 *없음*(P4 문제 아님).
- **★③진짜 벽 = write 단계(write_ok 1/20)**, 기제 = **dump 확정**:
  - **P6 confirm 미수행**(task3: `G2_CONFIRM_WRITE blocked` → confirm 없이 **동일 write 6연타** → too_many_errors).
  - **P5 정책 위반**(task0: `Non-delivered cannot be exchanged` → **동일 호출 9연타**).
  - **= P7 retry-loop가 write 단계 지배**(상류선 P7 작동했으나 *게이트-블록 write*엔 무력).
- **★결론(병목의 *단계 분리*)**: 체인에 **두 게이트 단계**: ⓐ상류 auth/order = **P2b 날조**(→v9 anti-fab) ⓑwrite = **P6 confirm + P7 recovery**(게이트-블록 후 루프)(→v8 P6 + P7 recovery). **둘 다 고쳐야 pass 이동** — anti-fab만으론 write까지 보내고 거기서 막힘. ⇒ **v9(상류) + v8/P6+P7(write) *동반*이 정답**(리뷰 훅 #7 해소: 동반 필요 확정).
- ⚠️L1L2 auth 17→9 퇴행 = **regen/동적-블랙리스트의 auth false-positive**(legit name/email 차단 의심) → 검증기 고정밀화 선결(§7 #5).

## 8. 마일스톤 (리뷰 후 갱신)
- ~~**M0**: 3-arm prototype~~ **✅완료(위)** — 레버 확정·pass 정체·천장 다단계. C/D/E 투자는 *전-체인* 타깃으로 조정.
- **M1**: `fc_randomize_fetchable` 확장(전 fetchable 값·SOPBench) + QC. 검증기 강화(literal blocklist·false-pos 감사).
- **M2**: DPO 쌍 합성(chosen/rejected·검증기 라벨) + 양방향 DPO 학습(v9-dpo).
- **M3**: v9-dpo τ² 전이 eval(fab율·fetch-first율·pass) + ablation(randomize-only vs +DPO).
- **M4 (양성 시)**: RLVR(gate-in-loop·GRPO·보상=검증기) = Track B.
- **M5**: 논문/특허 라인 — provenance 검증기(결정론) through-line·전이 헤드라인.
