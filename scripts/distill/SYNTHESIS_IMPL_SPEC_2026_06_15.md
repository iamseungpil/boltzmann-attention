# 구현 명세 (리뷰용): fc_confirm_augment(P6) · fc_recovery_augment(P7) · gen_synth_2hop(P2b)

> 상위 = `SYNTHESIS_DESIGN_PRIMITIVES_2026_06_15.md`(설계·왜) · 이 문서 = 구현 명세(어떻게·정확). **상태: 리뷰용 DRAFT — 승인 후 코드.**
> 공통: 출력=native-FC(`{tools, messages, _meta}`·`_supervise`=assistant) · 후속 `fc_randomize_fetchable` · QC=`fc_build_sft` 합류. 생성기 LLM 허용·선별/검증 결정론(불변).

---

## A. fc_confirm_augment.py (P6 confirm-gate)
### 입력/출력
- in: SOPBench FC 궤적(`sop_all.jsonl` 또는 `sop_rand.jsonl`). out: `sop_confirm.jsonl`.
- args: `--in --out --frac_pos 0.8 --frac_neg 0.2 --seed 42 --sample N`.

### write/read 분류 (결정론·핵심)
- **read 접두어 집합** = {`get_`,`view_`,`internal_get_`,`internal_check_`,`login_user`,`logout_user`,`call_get_database`}. exit_conversation=제외.
- tool_call.name이 read-접두어 매치 → read / 아니면 **write**(비가역). 
- ⚠️리뷰: 도메인별 actions(`<dom>_assistant.py`)와 대조해 오분류 점검 필요(예: `pay_loan`·`apply_*`·`transfer_*`·`cancel_*`·`modify_*`·`open_*`·`schedule_*`·`submit_*`·`update_*`·`book_*` = write 확인).

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

### 열린 질문(리뷰)
①첫-write-only vs every-write ②neg frac(0.2 적정?) ③confirm 문구 다양화(템플릿 N개 or LLM-rewrite) ④read-접두어 집합 도메인 검증.

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
- A의 "V 생산 getter" = getter_map서 V 슬롯 생산 도구(없으면 ask-user 분기). B의 precond = 해당 write의 dirgraph/getter_map 사전조건.

### QC/엣지
- error 턴 1개 + 복구 행동이 실패 행동과 다름 + 최종 성공. tool_call_id 페어링.
- 대상 못 찾으면(getter-fetchable 인자 없음) A skip→B/C 시도. value-randomize 후속.

### 열린 질문(리뷰)
①reflect 블록 포함 여부(토큰↑ vs 신호↑) ②클래스 믹스 비율(A:B:C) ③B의 precond 추출을 getter_map서 결정론 가능한가(미가능시 도메인별 수작업?) ④에러 메시지 문구 현실성.

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

### 열린 질문(리뷰)
①K 도메인 목록·각 스키마 확정 ②user 자연화=템플릿(완전결정론·단조) vs LLM-rewrite(다양·생성기LLM) ③select(P4) 난이도(목록 길이·유사 distractor) ④params도 fetch-dependent로? ⑤scale(M·K) 목표.

---

## D. 빌드/믹스/eval (양 라인)
- **patent-line sft** = sop_rand + d5_ask + tb_all_v4 + **synth2hop + sop_confirm + sop_recovery** (CFB 제외). 
- **논문-line(v7+)** = + CFB(P2b)[+permissive 외부: When2Call(ask)·Tool-Reflection(P7) 선택].
- upweight 각 신규 ~10-15%·3x 재-randomize. 
- eval(결정론·`tau2_autopsy` 확장): P6 confirm 준수율(write 전 user-yes)·P7 recovery율(에러후 no-repeat&성공)·P2b 2-hop fetch율(sid가 get_details서). 전이=τ²(G2/루프/order_id)+SOP-Bench·`coupling_eval`.

## E. 마일스톤 (리뷰 후)
M1 read/write 분류 검증(A) + getter_map precond 추출 가능성(B) + K 도메인 스키마 확정(C) → M2 세 스크립트 구현 → M3 소량 파일럿(각 50·QC+샘플 육안) → M4 patent-line sft 빌드·학습 → M5 eval(준수율+전이) + 논문/특허 라인 비교.
