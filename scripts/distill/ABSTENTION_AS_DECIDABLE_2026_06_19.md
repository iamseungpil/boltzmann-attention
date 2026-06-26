# Abstention as Decidable — 기권을 구조조건으로 재캐스팅 (2026-06-19)

> **자립 델타 문서**(리뷰용). 동기 = 기권(abstention)을 "모델이 자기 무능을 앎"(LLM이 못하는 self-certification)에서 **"엔진이 구조조건을 탐지"(decidable·offload)**로 옮긴다 = thesis의 *decidable→offload*를 abstention에 적용 + fabrication을 *구조적으로 불가능*하게.
> 상위 참조 = `A2_MINIMIZATION_FRONTIER_DESIGN`(이 문서를 참조·"correct-or-abstain 보장" 축의 정밀화) · `A2_GROUNDING_WIRING_DESIGN §5a`(fetch-vs-ask 라우팅의 구현) · `A2_FORMAT_SPEC §3`(abstain 신호→P7 복구).
> 불변 = `00-thesis`(decidable=offload) · `PRIMITIVE_COVERAGE_MATRIX`(P1-P9·:94-95 P7 recovery RL) · `03-anti-drift`. 선행 = `2603.20449`(게이트=NL정책→런타임 차단) · ReDAct `2604.07036`(calibrated deferral) · Self-Healing `2606.01416`(recover-verify).

## 0. 핵심 주장 (한 줄)
**Calibrated abstention layer의 *대부분*을 P-기저로 구축 가능** — 기권을 decidable 구조조건(P1/P2/P7·resolve-비유일·P5·P6·verify)으로 재캐스팅하면, 모델이 자기 무능을 알 필요 없이 **엔진이 탐지**한다. 진짜 잔여(확률 calibration)는 가역·유일·합법·미묘오류 — *작게 격리*되고 거기만 selective-prediction.

## 1. ★보장의 정확한 범위 = 구조적-sound-or-abstain ≠ correct-or-abstain (위험A)
엔진이 탐지하는 구조조건으로 막는 건 **구조적 오류**(fabrication·미인증·미확인·미해결)뿐이다. 그러나:
- **구조적으론 멀쩡한데 의미적으로 틀린 경우** — grounded·unique·gate-pass·reversible인데 *잘못된 valid item* 선택·NL 의도 오독 — 는 **어떤 구조조건도 안 걸림**(다 통과).
- ⇒ 보장 = **"구조적-sound-or-abstain"**(증명가능·fabrication 0) **≠ "correct-or-abstain"**(과주장·쓰지 말 것).
- ★**fabrication을 구조적으로 없애고 나면 의미-오류 잔여가 *지배적일 수 있음***(τ² 실패 상당수 = 날조 아니라 wrong-valid-selection). 잔여를 "작다"로 가정 = 미검증 경험 주장 → **측정 대상**(§4-②).
- **헤드라인** = "구조적 무결 보장(fabrication 0) + 의미 잔여는 격리된 selective-prediction". *correct-or-abstain*이라 쓰지 않는다.

## 2. abstention-trigger → P-조건 매핑표
| trigger (기권/막힘 사유) | P-조건 | 판정 | 처방 |
|---|---|---|---|
| 값이 user·tool 어디에도 없음(literal 미제공) | **P1 provenance-fail** | **decidable**(provenance ∈{U,T,K} 분할·`ABOX §1.5a`) | ASK(user에게) |
| 필요 producer 출력이 trace에 없음 | **P2(producer-부재)** | **decidable**(A2 스키마: producer 존재?) | **FETCH**(존재시)·ASK(부재시) — §3 |
| resolve가 후보 다수(비유일) | resolve-비유일(`len>1`) | **decidable**(엔진 카운트) | ASK(clarify) 또는 FETCH(필터용 누락 producer) |
| 비가역 write 전 정책 미충족 | **P5 gate** | **decidable**(gate_spec 집행·G1-G4) | block→정책충족 행동 라우팅 |
| 비가역 write 전 confirm 없음 | **P6 confirm** | **decidable**(trace에 confirm turn?) | confirm 요청 |
| 실행결과 사후검증 실패 | verify-fail | **decidable**(type/precond/replay) | 차단·재라우팅 |
| ───────────────── | ───── | ───── | ───── |
| 가역·유일·합법인데 *미묘오류*(valid 중 틀림) | — | **NOT decidable**(의미층) | 확률 잔여·selective-prediction(§1) |
- 위 6 trigger = **구조조건·엔진이 결정론 탐지**(모델 self-cert 불요). 마지막 1행 = **진짜 확률 잔여**(작게 격리·여기만 calibration).

## 3. ★fetch-vs-ask 분기 (위험B — bare ask면 τ²서 깨짐)
`ground_OK=0`/막힘엔 **두 원인이 섞이고 결정론으로 구분 가능**:
```
막힘(ground_OK=0) → 필요 producer 출력이 trace에 있나?
   ├ 아니오 + producer가 A2-스키마에 존재(fetchable)  → FETCH 신호("producer를 먼저 불러라")  ← ask 아님·P2b
   ├ 아니오 + producer 부재 ∧ user 미제공             → ASK/escalate(P7)
   └ 예지만 비유일/anchor 불명                        → ASK(clarify) 또는 FETCH(누락 producer)
```
- τ²의 order_id 미ground는 **거의 fetchable-but-not-fetched**(`order_id ← get_user_details` 미호출). 둘 다 ask로 보내면 user-sim이 order_id 모름 → **거짓-실패**.
- ⇒ abstention-as-decidable이 **A2(producer-존재)에 의존** — 그 producer-존재 체크(스키마-decidable)가 ask-vs-fetch를 가른다. **bare `ground_OK=0→ask` 프로토타입 금지·반드시 fetch-우선.**

## 4. fabrication 제거의 지위 + 측정 (위험C/E)
- **fabrication→0 = offload-인프라지 학습-기여 아님**: ground-실패 라우팅은 결정론 엔진 변경이라 fabrication을 trivially 0으로(미ground 값을 엔진이 절대 안 흘림). = offload 다리(`2603.20449`/R1b-gate 동류·known)지 *모델이 뭘 학습했나*가 아님. **"fabrication 사라짐"을 thesis 결과로 오독 금지** — 필요 인프라지 novelty 아님.
- **recovery는 여전히 hard·범위 한정**: ASK 후 user 답 받아 *올바르게 재시도*하는 multi-turn recovery = 반응형 P7 — gold에 deny 없어 static-SFT 불가(`PRIMITIVE_MATRIX:94-95`·RL 필요·미해결). **abstention은 *막힌 단일 스텝의 ASK/FETCH까지*만 깨끗**·답 받아 복구는 별개. "구조적 무결" 주장 범위를 **단일-스텝 차단**으로 정직히 한정.
- **★측정 3개**(프로토타입 즉시):
  1. **fabrication율 → 0** (예상·결정론·기여 아님).
  2. **★의미-오류 잔여율**(구조 통과했는데 gold 불일치) = §1의 진짜 질문·잔여가 "작은가" 실측.
  3. **wrong-abstention율**(fetchable인데 ask로 보냄 = §3 분기 실패) = 분기가 맞나.

## 5. thesis 정합 + novelty
- 이 축이 thesis에 **"구조적-sound-or-abstain 보장"**을 더한다 — P-기저가 그걸 *대부분 결정론*으로 떠받침(decidable→offload 적용).
- **novelty는 fabrication 제거(인프라)가 아니라**: ① abstention을 *decidable 구조조건으로 분해*했다는 프레임(self-cert 불요) ② 그 위에서 **A2-formalize가 전이하나** ③ 의미-잔여를 *격리·측정*하는 정직.
- **선행 정합**: 게이트(`2603.20449`)·deferral(ReDAct)·recover-verify(Self-Healing)는 *채택·인용*(발명 아님). 우리 델타 = **P-기저 분해 + 구조/의미 잔여 분리 + A2-의존 fetch-vs-ask**.

## 6. 자가심사 (anti-drift 룰7)
- **과주장 가드**: "correct-or-abstain" 금지(§1)·"잔여 작다" 가정 금지(측정)·"fabrication 제거=novelty" 금지(§4).
- **치팅면**: 구조조건은 전부 decidable·엔진 탐지(모델 self-cert 0)·real 도구 미대체. fetch-vs-ask는 A2-스키마 decidable(하드코딩 0).
- **범위 정직**: 단일-스텝 차단까지·multi-turn recovery는 RL 별개·미해결(§4). 의미 잔여=확률·격리.

## 7. 한 줄
**기권 = (모델이 자기 무능 앎) → (엔진이 decidable 구조조건 탐지)로 재캐스팅 → fabrication을 구조적으로 0 + 막힘을 fetch-우선으로 라우팅. 보장은 구조적-sound이지 correct 아님 — 의미 잔여는 격리·측정 대상이고, 그게 A2-formalize 전이라는 진짜 질문을 데이터로 드러낸다.**
