# 구현 명세 (리뷰용): fc_confirm_augment(P6) · fc_recovery_augment(P7) · gen_synth_2hop(P2b)

> 상위 = `SYNTHESIS_DESIGN_PRIMITIVES_2026_06_15.md`(설계·왜) · 이 문서 = 구현 명세(어떻게·정확). **상태: 리뷰용 DRAFT — 승인 후 코드.**
> 공통: 출력=native-FC(`{tools, messages, _meta}`·`_supervise`=assistant) · 후속 `fc_randomize_fetchable` · QC=`fc_build_sft` 합류. 생성기 LLM 허용·선별/검증 결정론(불변).

---

## A. fc_confirm_augment.py (P6 confirm-gate)
### 입력/출력
- in: SOPBench FC 궤적(`sop_all.jsonl` 또는 `sop_rand.jsonl`). out: `sop_confirm.jsonl`.
- args: `--in --out --frac_pos 0.8 --frac_neg 0.2 --seed 42 --sample N`.

### write/read 분류 (결정론·핵심)
- ~~read 접두어 집합 = {`get_`,...}~~ → **★폐기(2026-06-15 리뷰, 데이터 반증)**: name-prefix는 틀림. bank 실측 **`get_loan(username, loan_amount)->bool` = 비가역 write인데 `get_` 접두어 → read로 오분류**. confirm 누락 → P6 오염.
- **★올바른 분류기 = 반환 시그니처**(bank_assistant 실측): getter/read = `tuple[bool,<value>]`(get_account_balance→`tuple[bool,float]`·get_credit_cards→`tuple[bool,list]`·internal_get_*) / action·write·auth = `bool`(transfer_funds·**get_loan**·pay_loan·login_user 전부 `->bool`).
  - **규칙**: `bool` 반환 ∧ **auth-carveout**{`login_user`,`logout_user`,`authenticate_admin_password`} 제외 = **write**. → get_loan 정확히 write.
  - 잔여 엣지(수동 확인): `exchange_foreign_currency->tuple[bool,float]`(견적 vs 실행 모호) 등 소수.
  - **이중 안전**: 반환시그니처 + evaluator write-set(또는 `<dom>_assistant.py` action 목록) 대조 = 최종 결정론 분류.

### 알고리즘
1. 첫 write tool_call 인덱스 i 탐색. 없으면 skip.
2. **pos(frac_pos)**: i 앞에 삽입 →
   ```
   [assistant _sup] "To confirm: I'll <humanize(name)> with <k=v, ...(write 인자)>. Shall I proceed?"
   [user]           "Yes, please proceed."
   [원래 i..끝]      (write 그대로)
   ```
3. **neg(frac_neg)**: 동일 confirm 후 →
   ```
   [user]           "Actually, no — please don't."
   [assistant _sup] "Understood, I won't proceed. Is there anything else?"  (+ exit_conversation 호출 or 종료)
   ```
   = **write tool_call 제거**(미확인→미실행 학습). 원 write 이후 턴도 절단.
- key params = write 인자 dict(긴 값 truncate 표시). humanize=name.replace("_"," ").
- _meta: `{p6_confirm:true, branch:"pos"|"neg", write_name}`.

### QC/엣지
- pos: write 직전에 [assistant confirm?]+[user yes] 쌍 존재. neg: write tool_call 부재.
- 다중 write: **v1=첫 write만** confirm(단순). (리뷰: 각 write마다 confirm으로 확장?)
- value-randomize 후속(write 인자값).

### 열린 질문 → ★결정 (2026-06-15 리뷰)
- ①첫-write vs every-write → **every-write**. 정책은 *각* 비가역 행동에 confirm 요구. 첫-write만 하면 "한 번 confirm→이후 free" 오학습. 궤적 길이만 캡.
- ②neg frac 0.2 → **0.35~0.4로 상향**. neg(user "no"→미실행)가 confirm을 진짜 게이트로 만드는 **반사실**. 적으면 "항상 confirm→항상 proceed"=ritual화(SEL-2/게이트 교훈 동형). **+eval에 neg-준수율 별도 지표**(거절 시 write 안 하나).
- ③문구 다양화 → **템플릿 N개 + LLM 패러프레이즈**(단일 템플릿=리터럴 overfit, LODO 형식-간섭 교훈). confirm의 *의미행위* 학습 목표.
- ④read/write 분류 → **위 §write/read 분류로 해소**(반환시그니처+auth-carveout, get_loan 버그 박제).

---

## B. fc_recovery_augment.py (P7 recovery)
### 입력/출력
- in: SOPBench FC 성공 rollout. out: `sop_recovery.jsonl`.
- args: `--in --out --getter_map --classes A,B,C --frac 0.5 --reflect 1 --seed 42`.

### 3 클래스 (우리 고가치 = A,B — Tool-Reflection-Bench 결핍분)
- **A. not-found→lookup**(order_id 루프 처방): 대상 tool_call(인자 V·V가 *getter-fetchable*=getter_map 슬롯 매치 또는 id-류). 주입:
  ```
  [assistant _sup] <같은 tool>(arg=V')           # V'=rand_like(V)·≠V
  [tool]           "Error: <V'> not found"
  [assistant _sup] (reflect?) <V 생산 getter> 호출   # ★V' 반복 금지
  [tool]           <V 포함 출력>
  [assistant _sup] <원래 tool>(arg=V) → [tool] <원 성공 결과>
  ```
- **B. policy-gate-block**(G2류·P6 연계): write가 사전조건 미충족. 주입:
  ```
  [assistant _sup] <write>(...) → [tool] "Error: [POLICY GATE] requires <precond>"
  [assistant _sup] (reflect?) <precond 충족: confirm 요청 or 선행 getter>  → (필요 user/tool 턴)
  [assistant _sup] <write> 재시도 → [tool] <성공>
  ```
- **C. argument-error**(일반·Tool-Reflection 포맷): 임의 인자 V'→error→reflect→V. (보강용)
- **reflect 블록**(--reflect): assistant content에 1문장 진단("The previous call failed because <V'> isn't a valid <key>; I'll obtain it from <source>."). Tool-Reflection-Bench 포맷 차용. off면 corrected-call만.

### 결정론/provenance
- V=원 rollout 실제값(복구=정답·날조0). V'≠V 보장(rand_like 후 동일하면 재생성). 복구 행동 ≠ 실패 행동(no-loop 신호).
- A의 "V 생산 getter" = getter_map서 V 슬롯 생산 도구(없으면 ask-user 분기).
- **★B precond 추출 = 결정론 가능 확정(2026-06-15 리뷰, 데이터 검증) — 단 소스 2개 분업**:
  - **precond 자체**(write 전 무엇이 충족돼야) = task `directed_action_graph`/`constraints_original` = **DGGATE 재구성**(Guard-2 OVER=0/UNDER=0 검증 기계 재사용).
  - **satisfier**(그 precond를 어느 도구가 생산) = **getter_map**.
  - ⇒ getter_map 단독 불가(satisfier만 줌)·**dirgraph=precond·getter_map=satisfier**. 도메인별 수작업 불요.

### QC/엣지
- error 턴 1개 + 복구 행동이 실패 행동과 다름 + 최종 성공. tool_call_id 페어링.
- 대상 못 찾으면(getter-fetchable 인자 없음) A skip→B/C 시도. value-randomize 후속.

### 열린 질문 → ★결정 (2026-06-15 리뷰)
- ①reflect 블록 → **ablation arm으로 포함**(on/off 비교, Tool-Reflection 포맷 차용). 토큰↑vs신호↑ 실측 판정.
- ②클래스 믹스 → **A 0.5 / B 0.3 / C 0.2**. A(not-found→lookup)=order_id 루프 직격 최고가치, B(policy-gate)=P6 연계, C(arg-error)=일반 보강.
- ③B precond 결정론 추출 → **가능 확정**(위 §결정론: dirgraph=precond·getter_map=satisfier). 수작업 불요.
- ④에러 문구 → 현실 포맷 템플릿(`"Error: <V'> not found"`·`"[POLICY GATE] requires <precond>"`) + 소폭 다양화.

### ★사전등록 caveat (P7 = 형태 vs 반응성, 박제)
error-injection은 복구 **형태**(error→reflect→다른행동→성공)를 static-gold로 가르침 — 게이트상태에 대한 **진짜 반응성**(주입 안 한 novel-error/게이트상태 일반화)은 아닐 수 있음(SFT vs RL 경계). 부분전이 시 잔여 = **Track-B(게이트-in-loop DPO/RL) 후속**. 단 "8연타 루프"(에러 후 동일행동)는 "에러 후 *반드시 다른* 행동" injection으로 직격 → no-loop 신호는 잘 타깃됨.

---

## C. gen_synth_2hop.py (P2b Path B · 자체 합성 2-hop 도메인 · 완전소유)
### 목적
SOPBench에 없는 **id-lookup 2-hop**(τ² find_user→get_orders→order_id→act 미러)을 우리가 생성·소유(특허 clean·CFB 대체).

### 도메인 스키마(K개·도메인-일반)
각 도메인 = (entity E, sub-entity S, action):
```
search_<E>(name:str) -> {"<E>_id": <id>}                          # 1-hop
get_<E>_<Ss>(<E>_id) -> {"<Ss>":[{"<S>_id":<sid>, <fields>}...]}   # 2-hop(목록·sub_id 포함)
<action>(<S>_id, <params>) -> {"status":"ok"}                      # act(sub_id 사용)
```
- 후보 K(≥5·표면 다양·구조 동형): e-commerce(user→orders→order_id→cancel/modify)·library(member→loans→loan_id→renew)·clinic(patient→appointments→appt_id→reschedule)·support(account→tickets→ticket_id→escalate)·travel(traveler→bookings→booking_id→change). ABox-swap.

### task 생성(결정론 코어 + 선택 LLM 자연화)
1. 랜덤 인스턴스: name·<E>_id·[<S> 목록(각 sid+fields)] 전부 합성·랜덤(포맷-보존).
2. user 발화 = goal + **name만**(id·sid는 *withhold*). (자연화: 템플릿 or LLM-rewrite·선택)
3. gold 궤적(결정론):
   ```
   [user] "<goal> for <name>."
   [A] search_<E>(name) → [tool] {<E>_id}
   [A] get_<E>_<Ss>(<E>_id) → [tool] {목록 incl 타깃 <S>_id}
   [A] <action>(<S>_id, <params>) → [tool] {ok}      # <S>_id는 *오직 직전 출력에만* 존재→fetch강제
   ```
4. 선택 분기: 목록 다항목 → 옳은 항목 select(P4 동시 학습)·params도 출력서 추출.

### 소유/clean/스케일
- 도구·응답 전부 우리 정의 = 외부 ToU 0 = **특허 OK**. 양=무제한(M task × K domain). 도구명 per-traj alias(R1). value-randomize(sid는 출력에만).
- out: `synth2hop.jsonl`·`_meta:{bench:"synth2hop", domain}`.

### 열린 질문 → ★결정 (2026-06-15 리뷰)
> **★★우선순위 게이트(필독): gen_synth_2hop는 *지금 짓지 않는다*.** v7가 이미 GPU1서 P2b via CFB 학습 중 → CFB 전이결과에 **게이트**. CFB P2b 전이되면(order_id 날조↓) → Path B를 patent-clean 대체재로 빌드 정당. **전이 안 되면 = 문제는 데이터-소스 아니라 R4 의미-전이 → 같은 primitive 다른 소스(Path B)도 무효 → 진단 선행**(매몰비용 회피).
- ①K 도메인 → 5개(e-commerce/library/clinic/support/travel) 구조-동형 확정·ABox-swap. (게이트 후 확정.)
- ②user 자연화 → **템플릿(gold tool-call 결정론 유지) + user 발화만 선택적 LLM 패러프레이즈**. gold 궤적은 절대 LLM 비결정 금지.
- ③select(P4) 난이도 → **유사-distractor 포함**(이름 비슷한 항목 多)·목록 길이 가변. 단일항목이면 select 자명 → P4 미학습.
- ④params도 fetch-dependent → **예**(일부 action params를 get_E_Ss 출력서 추출 = P2b 강화).
- ⑤scale → CFB와 공정비교 위해 **M≈200/도메인×5≈1000**서 시작(CFB 규모 매칭).

---

## D. 빌드/믹스/eval (양 라인)
- **patent-line sft** = sop_rand + d5_ask + tb_all_v4 + **synth2hop + sop_confirm + sop_recovery** (CFB 제외). 
- **논문-line(v7+)** = + CFB(P2b)[+permissive 외부: When2Call(ask)·Tool-Reflection(P7) 선택].
- upweight 각 신규 ~10-15%·3x 재-randomize. 
- eval(결정론·`tau2_autopsy` 확장): P6 confirm 준수율(write 전 user-yes)·P7 recovery율(에러후 no-repeat&성공)·P2b 2-hop fetch율(sid가 get_details서). 전이=τ²(G2/루프/order_id)+SOP-Bench·`coupling_eval`.

## E. 마일스톤 (★리뷰 후 갱신 2026-06-15 — 우선순위 게이트 반영)
- **빌드 순서 = P6 → P7 즉시 / gen_synth_2hop(C)는 CFB 전이결과 게이트.** 근거: P6/P7 = 매트릭스 지목 실제 잔여 gap·둘 다 SOPBench rollout augmentation(동일 파이프라인). P2b는 CFB(v7·GPU1 학습중)가 *전이여부 자체*를 테스트 중 → C는 그 결과 후.
- **P6 먼저**(confirm 삽입·단순 = augmentation 파이프라인 de-risk) → **P7**(3클래스·error-injection·dirgraph precond 재사용).
- M1 ~~read/write 분류 검증~~ **✅DONE(리뷰: 반환시그니처+auth-carveout, get_loan 버그)** + ~~precond 추출 가능성~~ **✅DONE(dirgraph=precond·getter_map=satisfier)** → M2 A·B 구현 → M3 파일럿(각 50·QC+육안) → M4 patent-line sft(synth2hop 자리는 CFB 게이트 후) → M5 eval(준수율+전이·neg-준수율 포함) + 논문/특허 라인 비교.
- **C(gen_synth_2hop)**: CFB P2b 전이 양성 확인 시 별도 마일스톤으로 착수.
