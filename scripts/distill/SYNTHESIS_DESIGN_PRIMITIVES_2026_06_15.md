# 합성 설계: P2b·P6·P7 clean-owned 소스 (patent-line, 2026-06-15)

> 상위 = `PRIMITIVE_COVERAGE_MATRIX_2026_06_15.md`(P1-P9·gap=P2b clean·P6·P7) · `V7_PROACTIVE_GATHER_DESIGN §8c-BLOCKING#2`(특허=clean 소스 필수) · 불변 = `feedback-thesis-tbox-transfer-direction`·`feedback-selector-verifier-deterministic`.
> **목적**: ComplexFuncBench(P2b·논문용·ToU)와 *별개로*, **특허/프로덕션용 clean-owned 데이터**로 P2b·P6·P7을 생성. 전부 우리 소유(SOPBench executor / 자체 합성) = 외부 ToU 0 = 주권·특허 보존. 딥리서치가 clean 3rd-party 벤치를 찾으면 대체/보완 가능.

## 공통 원칙
- **생성기=LLM teacher 허용**(생성에만)·**선별/검증=결정론**(불변). 가능한 곳은 결정론 템플릿(LLM 불요).
- **value-randomization 필수**(`fc_randomize_fetchable`): 합성 값은 포맷-보존 랜덤·궤적-고유 → 암기불가·copy 강제.
- 출력 = native-FC 궤적(`fc_*` 포맷). loss-mask=assistant.
- QC = 스키마·tool-call∈tools·페어링·역할순서.

---

## P6 — confirm-gate (결정론 augmentation·SOPBench 자산) — `fc_confirm_augment.py`
**스킬**: 비가역 write 전 user 확인 획득·미확인 시 실행 안 함.
**소스**: SOPBench FC 성공 rollout(소유). write 액션 = `<dom>_assistant.py` actions 중 read/getter(get_/view_/internal_check_/login) 아닌 것(결정론 분류).
**구성(결정론 템플릿)**: rollout서 첫 write tool_call(인덱스 i) 앞에 삽입:
```
[assistant] "To confirm: I will <write_name> with <key params>. Shall I proceed?"   (_supervise)
[user]      "Yes, please proceed."
[assistant] <원래 write tool_call ...>   (그대로 이어붙임)
```
- **대조(필수)**: ① read-only 액션은 confirm 없음(자연) → 게이트가 *write 한정* 학습. ② **부정 예시**: 일부 변형서 user "No, cancel that." → assistant write **안 하고** 종료/대안 → "미확인 시 실행 금지" 학습.
- **provenance**: write 인자값은 원 rollout서 그대로(grounded). value-randomize 후속.
- **전이 타깃**: τ² G2_CONFIRM_WRITE(autopsy task17 실패점). clean·특허 OK.
- **★딥리서치 확인(2026-06-15)**: clean 전용 FC 벤치 *없음* → SOPBench 합성이 정답(검증됨). 정제:
  - **결정론 검증기 참조 = ST-WebAgentBench `is_ask_the_user`**(Apache-2.0·"User Consent: 비가역 전 명시승인"·gold궤적 없음·web포맷이라 학습 불가, *평가기 설계 참조*로만).
  - **궤적 모양 템플릿 = τ²-bench airline `policy.md`**("list action details → obtain explicit user confirmation(yes) → execute") — 인용만·학습 금지(테스트 오염).
  - **ask-측 별 소스(P6 아님) = When2Call**(NVIDIA·CC-BY-4.0 *상업OK*·FC-native·~15k SFT+9k DPO): "ask-when-param-missing"=ask-gather/D5 prior → **ask 분기 clean 학습원 후보**(P6=confirm과 구분 명시).
  - ⚠️ open: SOPBench 라이선스 repo 명시 확인 필요(우리 학습벤치라 신규 오염 0이나 재배포시 확인).

## P7 — recovery / 에러복구 (결정론 error-injection·SOPBench 자산) — `fc_recovery_augment.py`
**스킬**: tool 에러/게이트-블록 시 동일호출 반복 금지 → 전략전환(re-fetch/ask/대안) → 성공.
**소스**: SOPBench FC 성공 rollout. 한 tool_call(인자값 V·tool 성공)을 골라:
**구성(결정론)**: 정답 호출 앞에 *실패→복구* 세그먼트 주입:
```
[assistant] <같은 도구>(arg=V')   # V'=rand_like(V)·V와 다름(=틀린 값)
[tool]      "Error: <V'> not found / invalid"
[assistant] <복구>:                # ★동일 V' 반복 금지
   - V가 user-제공: "That didn't match — could you re-confirm your <key>?"  →[user] "<V>"
   - V가 getter-fetchable: <getter 호출> →[tool] <V 포함 출력>
[assistant] <원래 정답 호출>(arg=V) →[tool] <원 성공 결과>
```
- **핵심 신호**: 에러 후 **다른 행동**(V'≠V·re-fetch/ask) = no-loop. 복구는 *원 rollout의 실제 V* 사용 = 날조 0.
- **게이트-복구 변형(P6 연계·A2)**: write가 confirm-gate 에러 반환 → 복구=확인 요청 후 retry(=R3·A2 정의 retry). autopsy task17(8연타) 직접 처방.
- value-randomize 후속. clean·특허 OK.
- **★딥리서치 확인(2026-06-15)**: clean-ish 전용 소스 *존재*하나 우리 핵심 클래스 결핍 → **합성 유지**.
  - **포맷 템플릿/seed = Tool-Reflection-Bench**(MeiGen·arXiv 2509.18847·**OpenAI FC JSON**·4,928 train·`error→<reflect>진단→corrected tool_call→success`·헤드라인 예시="이전 tool 출력값 안 쓰고 잘못된 값"=우리 autopsy 그대로). **단 에러분류=Argument/Call-Order/Missing/Redundant 뿐 → not-found→lookup·policy-gate-block(우리 최고가치) 없음**·data 라이선스 미명시(특허=템플릿으로만). ⇒ **이 포맷(reflect+corrected-call)을 차용 + §위 우리 error-injection으로 빠진 클래스(not-found→lookup·policy-block) 주입** = 학습벤치 유지·특허 clean.
  - #2 ToolBench-R(3625·Python포맷·host미확인), #3 BFCL(recovery gold 없음=avoidance·Apache=P7-adjacent eval). 기각: CFB/ToolACE(clean success only)·ToolEmu(LLM-judge)·MINT/InterCode(code).

## P2b — gather-for-arg(2-hop) clean (CFB 대체·특허용)
SOPBench는 P2b 구조 희소(census 1.9%·getter=decision 위주)라 **두 경로 병행**:

### Path A (SOPBench withholding+oversample·보조) — `fc_withhold_2hop.py`
- getter_map으로 **write-arg가 getter-생산인 케이스**(1.9% + 명시 getter-체인: pay_loan(amount=owed_balance)류) 식별.
- user-sim **withholding**(V7 §3): 그 값을 `user_known`서 제거 → teacher가 getter 호출해 획득→arg. = clean 2-hop.
- **oversample**(희소 보완) + value-randomize. 한계: SOPBench 구조상 양 제한.

### ★Path B (자체 합성 2-hop 도메인·주력·완전소유) — `gen_synth_2hop.py`
SOPBench에 없는 **id-lookup 2-hop을 우리가 소유·생성**(τ²의 find_user→get_orders→order_id→act 패턴 미러·100% 우리 IP):
- **도메인 스키마(K개·서비스에이전트류)**: 각 도메인 =
  ```
  search_<entity>(name) -> {<entity>_id}                      # 1-hop: 이름→id
  get_<entity>_details(<entity>_id) -> {fields..., <sub>_id}  # 2-hop: id→상세(하위 id 포함)
  <action>(<sub>_id, <params>) -> ok                          # act: 하위 id 사용
  ```
  예: e-commerce(search_user→user_id→get_orders(user_id)→[order_id]→modify_order(order_id))·library·clinic 등 도메인-일반 변종.
- **task 생성(결정론)**: user 발화 = goal + **이름만**(id·하위id는 *안* 줌·withhold). gold 궤적 = search→get_details→**출력서 sub_id 추출**→action. 합성 응답 = 템플릿 JSON(필드 랜덤). 
- **value-randomize**: sub_id는 *오직 get_details 출력에만* 존재 → fetch 강제(암기불가). 도구명도 per-traj alias(R1).
- **abundant·clean·결정론·특허OK**: 양 무제한·외부 ToU 0·우리 IP. CFB의 grounded observe-then-use를 소유 버전으로 재현.
- **도메인-일반성**: K개 변종 + ABox-swap → 전이 타깃(τ²)과 *구조 동형·표면 무관* = thesis 부합.

**권고**: 특허-라인 P2b = **Path B 주력**(abundant·완전소유) + Path A 보조(SOPBench grounding 현실감). 논문-라인은 CFB 유지.

---

## 빌드/믹스
- 생성 → `fc_convert_*`(이미 native-FC면 직접) → `fc_randomize_fetchable` → `fc_build_sft` 합류.
- **patent-line sft** = sop_rand + d5_ask + tb_all_v4 + **synth_2hop(P2b) + confirm_aug(P6) + recovery_aug(P7)** (CFB 제외=ToU). 
- **논문-line sft(v7)** = + CFB(P2b). 두 라인 분리 유지.
- upweight: 각 신규 primitive ~10-15% 목표(묻히지 않게)·3x 재-randomize.

## eval (불변)
- 결정론: confirm 준수율(write 전 user-yes?)·recovery율(에러 후 no-repeat & 성공)·2-hop fetch율(sub_id가 get_details서 옴?). `tau2_autopsy.py` 확장.
- 전이: τ²(P6=G2·P7=루프·P2b=order_id) + SOP-Bench. coupling_eval로 in-dist↔전이.

## 마일스톤
M1 write/read 액션 분류(결정론·getter_map+actions) → M2 `fc_confirm_augment`·`fc_recovery_augment`(SOPBench·템플릿) → M3 `gen_synth_2hop`(Path B 도메인 스키마+생성기) → M4 patent-line sft 빌드+학습 → M5 eval(P6/P7/P2b 준수율+전이). 딥리서치 clean-bench 결과 도착 시 해당 primitive는 그 벤치로 대체/보완 판단.
