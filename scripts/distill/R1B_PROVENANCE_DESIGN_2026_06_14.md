# 설계서 v2: R1b 값-provenance 집행 (리뷰 반영) — 2026-06-14

> 상위 = `CROSS_BENCH_TRANSFER_PLAN_2026_06_14.md` §2c · 규칙 = `TASKBENCH_EXPERIMENT_RESULTS.md` §10.5 R1b · 변환기 의존 = `NATIVE_FC_CONVERTER_DESIGN_2026_06_14.md` · 불변 = `feedback-selector-verifier-deterministic`.
> **v2 변경(리뷰)**: ①진단을 *망각* vs *규칙결손* 분리(ask-rate eval 추가) ②novelty 정직화(provenance=필요조건·인증게이트와 직교) ③L1 보장 = verbatim 타입·추출기 recall 조건부(구현전 recall 실측 BLOCKING) ④D4 verbatim-only 스코프 ⑤L3 SFT 데이터의 converter 의존 명시. D1=별도-합성·D2=타입별·D5=대조쌍 확정.

## 0. 동기 + ★두 근본원인 분리 (리뷰1)
τ² 실증: plan-X 7B TBox가 인자값 날조 → compliant-pass **0.10(50-up)→0.05(250-up) < base 0.17**. **★provenance 실측(첫 인증-call 인자 grounded vs fabricated)**: base-7B **fab 7%**(grounded 88%) → fctbox 50-up **fab 40%** → 250-up **fab 60%** = **날조율 단조 증가** = **파국적 *망각* 확정**(모델이 grounding을 *갖고 있었는데* 학습으로 *잃음*). (⚠️앞선 "ask 156/160"은 greeting `?` 오염·폐기 — 올바른 지표는 인자 provenance.)
- ⇒ **두 원인 분리**: ⓐ**망각**(τ² *숫자* 붕괴의 주 기제) — 처방 = **L3(ask-user 재학습/replay)이 숫자 회복의 주역.** ⓑ**provenance 규칙 부재**(자가생성 가능성) — 처방 = **L1/L2가 *결정론 보장*을 위에 얹음.**
- **프레이밍 규율**: **L3가 τ² 숫자를 회복시키고, L1/L2는 그 위에 결정론 보장을 추가한다.** L1/L2를 "τ² 점수 수정책"으로 팔지 말 것. 어느 게 숫자를 고치는지는 **측정으로 가른다**(§7 ask-rate).

## 1. 규칙 R1b
> 모든 인자 *값*은 출처 필수 — (a)user 발화 또는 (b)도구 출력. 부재 시 획득(read-tool/ask-user) 후 사용·자가생성 금지. 두 출처 = (b)tool-fetch(DB-파생) / (a)ask-user(user-only).

## 2. 3-레이어 (직교·역할 분리)
| L | 역할 | 종류 | 무엇을 |
|---|---|---|---|
| **L3 학습된 복구순서** | 값 부재 시 fetch-우선→없으면 ask | 학습 | **τ² *숫자* 회복** (주역) |
| **L2 provenance 검증기** | 출처 없으면 reject·플래그 | 결정론·검출 | 잔여 날조 포착 + 학습 보상신호 |
| **L1 XGrammar 디코딩-마스크** | 인자값을 컨텍스트-후보로 제약 | 결정론·하드 | **날조 보장**(조건부, §3c) |

## 3. 컴포넌트
### 3a. L3 — 학습된 복구순서 (fetch→ask) + ★소스 의존 (리뷰6)
- tools= 카탈로그(A1)에 **그 슬롯을 생산하는 read-tool이 있나**로 결정론 분기: 있으면 tool-fetch(R2 gather) / 없으면 ask-user.
- **SFT 데이터**:
  - tool-fetch-then-use = **FC-rollout 변환 궤적**(NATIVE_FC §3a). ★**의존 명시**: 이 데이터는 *converter가 인자-운반 궤적을 내놓아야* 존재 — **t1c(단발·인자0)면 없음**. 우리는 이미 **FC 성공 rollout**(실인자/결과)로 전환했으므로 충족. **R1B-L3 데이터 = converter의 P0(rollout 소스) 산물** — 두 설계서가 이 한 점에서 만남.
  - ask-user-then-use = augmentation(`fc_askuser_augment.py`·user-only 키만 물음). v3서 검증.
  - **대조쌍(D5)**: 같은 값이 한 도메인엔 fetcher 有·한 도메인엔 無 → 모델이 "카탈로그 보고 분기" 학습(없으면 always-ask/always-fetch 붕괴). **분기는 카탈로그서 결정론 도출**(휴리스틱 금지).
- RL/DPO(후속): L2 신호로 (성공복구, 날조시도) 쌍.

### 3b. L2 — provenance 검증기 (결정론·별도)
- 각 인자값 v: (이전 user ∪ 이전 tool 출력)의 **타입별 매치**인가? 아니면 fabricated→reject + 복구 메시지.
- **D1 = 별도 검증기 + 게이트 합성**(R3에 병합 금지): provenance를 정책-게이트(G1-G4)와 **직교** 유지 → "게이트당 한 속성"·독립 ablation 보존. 같은 인터셉션 지점에서 합성.

### 3c. L1 — XGrammar 컨텍스트-제약 (★보장은 조건부, 리뷰3)
- per-request 동적 제약: 인자값을 **컨텍스트서 추출한 타입별 후보**(D2)로 마스크. 후보 비면 호출 불가→L3 강제.
- **★보장의 정직한 형태**: "**verbatim-copyable 타입(id·email·name·account#)에 한해, *추출기 recall 조건부*로 구조적 0**" — 무조건 아님. 추출기가 valid span을 놓치면(recall<100%) **합법 호출 false-block → 날조보다 나쁨**(net-negative).
- **★BLOCKING(구현 전 zero-cost)**: 실제 τ² 인자값에서 **타입별 추출기 recall 선실측.** recall 85%면 합법 15% false-block = L1 net-negative → 그 타입은 L1 제외, L2(사후·관대)로만. (§13.7 "인코딩 제약 대비만 sound" 규율 동형.)
- ⚠️ vLLM 기본 flag 아님 = 커스텀(per-request 문법).

## 4. 단계 (staging)
1. **L3 SFT (지금·v3)**: ask-user augmentation. **판정 = ① ask-user rate 회복(§7, 망각 가설 직접) ② compliant-pass 0.10/0.05→base 0.17 회복.** 둘 다 = L3가 숫자 회복 확정.
2. **L2 검증기**: 타입별 provenance + 복구 deny. RL 보상신호.
3. **L1 XGrammar**: 추출기 recall 실측(BLOCKING) 후 verbatim 타입만 원천차단.

## 5. 정직한 한계 (6)
1. **정규화 값**: date/amount는 verbatim 아님 → L1/L2 보장 제외(D4). locale·상대날짜("next Tuesday")서 정규화기 깨짐.
2. **validity ≠ correctness**: L1/L2는 *날조*만 막음, *옳은 span 선택*은 R4·gather.
3. **tool-output 값은 호출 후에만 후보** → gather 선행(R2) 필수.
4. **augmentation 품질**: 템플릿 어법 어색 = v1. 부족 시 frontier-rewrite.
5. **잔여 망각**: ask-user 비율·instruction 혼합·epoch로 통제 — v3로 비율 판정.
6. **★L3 소스 의존(리뷰6)**: fetch-then-use 데이터는 converter의 인자-운반(FC-rollout) 산물 — converter P0 충족 시 존재. 두 설계서 결합 의존.

## 6. 논문 기여 (★정직화 — 리뷰2)
- **provenance = compliance의 *필요조건*이지 충분조건 아님.** 반례: user가 *남의* 이메일 발화 제공 → provenance-clean(user 출처) but **G3(단일-유저) 위반.** ⇒ **provenance-0 ≠ 정책준수.** 인증/인가 게이트(G1/G3)는 여전히 load-bearing·**직교 가산.** ("게이트 단독 ≠ novelty"[AgentSpec]와 동형: provenance 단독 ≠ compliance.)
- 성립하는 기여: **no-fabrication을 *결정론 보장*(L1, verbatim·recall조건부)으로 + fetch/ask 복구순서 *학습*(L3)으로** 묶은 칸이 빔(AgentSpec/guardrail은 런타임 규칙만). compliant-pass에 "값-provenance 위반 0" 축 = **인증 게이트와 *직교 추가* 속성**(대체 아님).
- R1a(닫힌 심볼) → R1b(열린 값 provenance) = grounding 규율 일반화.

## 7. eval (★ask-rate 추가 — 리뷰: 진단 가르는 결정타)
- **★provenance 날조율** (첫 인증-call 인자가 user/tool 출처에 없으면 fabricated): **base 7% → fctbox 50-up 40% → 250-up 60%**(단조=망각). **L3(v3) 후 = base 7% 쪽 회복?** — 이 지표가 "묻는 법을 잊음(망각)" vs "알지만 날조"를 가르고 L3 효과를 분리. (greeting 오염 주의 — ask 텍스트율 아닌 *인자 provenance*로 측정. 계측 = `t2_run_gated` 후 첫-AUTH-call 인자 ∈ 이전 user 발화?)
- compliant-pass 회복(L3) / provenance-위반 검출율(L2, 주입 날조) / L1 false-block율(verbatim 타입).
- 전이: SOP-Bench·τ² held-out 동일 측정.

## 8. 결정 D1-D5 (확정·리뷰)
- **D1 = 별도 검증기 + 게이트 합성** (R3 병합 금지·provenance 직교·독립 ablation).
- **D2 = 타입별 추출기**(email/id/amount) — 정규화 타입-특이·타입당 recall 측정·도메인-일반(per-domain 분기 없음).
- **D3 = augmentation frac/instruction 혼합 = 주 레버 승격**(망각이 주 기제면 replay 비율이 붕괴 방지 본체). "ask-user 보존"을 측정량으로(§7).
- **D4 = verbatim-only 스코프**: id/email/name/account#만 L1/L2 구조적 0, date/amount는 L3+R4 + 알려진 갭 정직보고. (절반의 진짜 보장 > 전체의 새는 보장.)
- **D5 = 대조쌍 합성 추가**(yes·강력): 분기는 카탈로그서 결정론 도출.
