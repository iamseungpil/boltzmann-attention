# C8 2차 — τ² retail selection 전이 (합성 op-IR 라우팅 → 실벤치 ABox-swap) 설계 — 2026-06-17

> 상위 = `C8_PROCEDURE_ROUTING_TRANSFER_DESIGN_2026_06_17.md`(1차·합성) · `NL_PROCEDURE_OFFLOAD_THEORY §7e/§7f` · `M_A_PROTOTYPE_DESIGN.md`(하니스) · `reference-abox-config-formalization-architecture`. 불변 = [[feedback-thesis-tbox-transfer-direction]]·[[feedback-nl-formalize-llm-selection-deterministic]]·[[feedback-secrets-never-commit-openrouter-leak]]·[[feedback-no-fundamental-claims-from-convenience-data]].

## 0. 사용자 질문의 실벤치 답
1차(합성)는 "절차어휘가 weight 내재화되나"를 *격리*해 답한다. 2차(이건) = **그 내재화된 라우팅이 τ²의 *실제* selection에 전이되나** = "절차어휘 TBox 고정 / 도메인어휘 ABox swap"의 실벤치 증명. τ²는 학습에 안 쓴다(전이 타깃·불변).

## 1. 범위 (정직 — write-벽의 *selection 부분*만)
- 대상 = τ² retail exchange의 **variant selection 결정**(`exchange_delivered_order_items`의 `new_item_ids`). `ma_gold_extract.py`가 이미 추출하는 **offline 케이스 29**(NL request + old item options + product 변형 catalog + gold new_item_id).
- **범위 밖**: tool sequencing·grounding(order_id fetch)·confirm/recovery. = write-벽 전체가 아니라 selection 한 조각([[project-tau2-write-failure-rootcause]]: new_item_ids만 틀리고 order_id 등은 맞았다 = selection이 잔여 병목).
- **키 불필요(Phase 1)**: offline value-accuracy는 DB 정답 대조라 user-sim 없이 측정. full rollout(Phase 2)만 키.

## 2. 아키텍처 = M-A를 op-IR로 재무장
현 M-A arm B는 **static `select_by` {opt:val}**(정적 criteria 강제) → superlative서 "전개된 결과를 LLM에 강제"(군-실행 떠넘김·§15/이론 §2-3). 교체:
| 층 | 현 M-A | C8-2차 (op-IR) |
|---|---|---|
| LLM (TBox·C8-trained 7B) | NL→`select_by`(정적) | NL+config→**op-IR** `{op,attr,among,dir,anchor,k}`(연산 *명명*) |
| 결정론 resolver | `select_by`→item_id | **op-IR engine**(filter/argmax/argmin/rank/comparative) on τ² 변형 catalog →new_item_id |
| ABox (도메인·swap) | (암묵) | **τ² config**: attr 분류(ordinal: price·zoom·resolution / categorical: color·material) + 도메인 gloss(NL "밝은"→brightness↑·"저렴"→price↓) |

- **TBox=op 라우팅**(합성서 학습·고정)·**ABox=τ² config**(어떤 attr이 ordinal·도메인어휘→attr). config만 swap·재학습 0 = 전이.
- resolver = `synth_depth.resolve_operation`을 τ² catalog 키로 어댑트(or `ma_resolver`를 op-IR 입력 받게 확장).

## 3. ★핵심 리스크 = τ² selection이 5-op과 동형인가 (케이스 타입 분해)
τ² selection을 op-IR로 표상하면 3 타입(예측 분포):
| 타입 | 예 | op-IR | op-IR 이득 |
|---|---|---|---|
| **constrained-substitution**(다수) | "이 키보드를 silver·low battery로, 나머지 유지" | `{op:filter, among: old_attrs ⊕ 변경분}` | static과 유사(superlative 아님)·**이득 작을 수 있음** |
| **superlative**(소수) | "max zoom waterproof", "cheapest" | `{op:argmax/argmin, attr:zoom/price, among:{...}}` | **큰 이득**(static이 못 푸는 군-실행을 engine이) |
| **comparative**(드묾) | "더 밝은 것", "한 단계 위 용량" | `{op:comparative, attr, dir, anchor:current}` | **큰 이득**(1차서 0→1.00 회복한 그 케이스) |
- **정직 예측**: op-IR 전이 이득은 *superlative/comparative*에 집중·constrained-substitution은 filter로 표상되나 static 대비 한계적. **통합 가치 = LLM이 케이스타입을 *판별*(filter vs argmax)** = 𝔤-사영(§7e §6). ⇒ 결과를 **케이스타입별로 분해 보고**(전체 평균이 substitution 다수에 가려지지 않게).
- constrained-substitution 표상의 난점 = "X만 바꾸고 유지"를 among에 정확히(old 복사 + 변경 override) — 이게 M-A wrong_criteria 9건의 원인([[project-tau2-write-failure-rootcause]]). op-IR이 이걸 *더 잘* 표상하는지(among 명시)가 부수 측정.

## 4. arm (offline·Phase 1)
| arm | LLM | 정체 | 비교 |
|---|---|---|---|
| A | base | concrete item_id 직접 | M-A 현 0.438 |
| B_static | base | static select_by + ma_resolver | M-A 현 0.41 |
| **B_op·S1** | base + **gloss config(in-context)** | op-IR + engine | op-IR 상한(어휘 떠먹임) |
| **B_op·S2** | **C8-trained(gloss-free)** | op-IR + engine | **전이 측정**(합성 라우팅이 τ² 도메인에) |
| D | — | oracle(gold op-IR + engine) | sanity |
- **판정**: (i) B_op·S2 > B_static = op-IR 재무장이 정적 criteria 넘나(특히 superlative/comparative 케이스). (ii) S2 ≈ S1 = 합성 라우팅이 τ² 어휘로 전이(gloss 없이) = TBox 고정 입증. (iii) 케이스타입별 분해.
- **음성 해석**: S2 ≤ B_static = 합성 라우팅이 실어휘로 안 옮음(표면결합) or τ² selection이 5-op 밖(constrained-substitution이 본질이고 op-IR 무익). 둘 구분 = 케이스타입 분해가 함.

## 5. Phase 2 (키 필요·나중) — end-to-end rollout
- offline selection 양성이면 → full τ² rollout(user-sim gpt-4.1·OpenRouter). selection 정확이 *pass*로 이어지나(grounding·sequencing 동반 필요).
- **키 규율**([[feedback-secrets-never-commit-openrouter-leak]]): 재발급 키 → 원격 `~/.openrouter_key`만(커밋·하드코딩·채팅 금지)·`t2_run_gated` 비용가드·gpt-4.1 전용.
- 정직 경계: pass엔 selection 외 다수 요인. selection 전이는 *필요조건*이지 충분조건 아님.

## 6. 구현 단계 (설계만·미착수)
1. `tau2_config_extract.py`: τ² product catalog → ABox config(attr ordinal/categorical 분류 + 도메인 gloss 소량 수기). product_details에서 자동 + retail 도메인 어휘 매핑.
2. `tau2_op_resolver.py`: op-IR + τ² 변형 catalog → new_item_id(synth engine 어댑트). constrained-substitution용 among=old⊕change 헬퍼.
3. `ma_eval.py` 확장: arm B_op(op-IR emit·config 주입·gloss 0/1)·케이스타입 태깅·타입별 분해 출력.
4. eval: 1차 합성 C8-trained 어댑터로 B_op·S2 vs base S1/B_static. `M_A_RESULTS §17`(τ² selection 전이) 박제.
- **선행조건**: 1차 합성 C8 배치가 *양성*(TRANSFER)일 것. 음성이면 2차 무의미(합성서 안 되면 τ²는 더 안 됨) → 1차 결과 *먼저*.

## 7. 위치
ABox-config formalization 아키텍처(reference)의 구체 실벤치 인스턴스 = M-A static-select를 op-IR로 교체 + C8 라우팅 주입. 양성이면 = **소형 on-prem 7B가 절차-라우팅을 TBox로 들고 τ² 도메인만 config swap해 selection을 결정론 offload** = 주권-leg 분담의 실벤치 직접 증거. = 사용자 질문("절차 TBox 고정 / 도메인 ABox swap")의 종착 답.
```
1차 합성(진행중·키0) ─TRANSFER?─► 2차 τ² offline(이 설계·키0) ─양성?─► Phase2 rollout(키 필요)
```
