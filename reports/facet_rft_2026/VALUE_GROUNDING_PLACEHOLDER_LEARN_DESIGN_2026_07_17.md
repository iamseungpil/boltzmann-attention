# 값-grounding placeholder 학습 설계 — FIND/GET/COMPUTE/ASK 외 값 주입 금지 스킬 (2026-07-17)

> 사용자 제안: *"record grounding을 포함한 모든 값 인용은 FIND(사용자/정책)·GET(도구)·ASK(사용자)·COMPUTE 외에는
> 값을 못 집어넣게 **placeholder를 학습**할 수 있나?"* — **가능·이것이 [[16]] 유일 잔여 learn(INFER-calibration)의
> 일반화 완성형.** 파생: C92(연산 오분류 통일이론)·C104(learn-wing 처방)·이 세션 실패 5종(§19·§14.3).
> 규율: [[11]] 학습벤치서만·ABox-swap 전이 / [[12]] 다양성 / C38·C42 데이터 타당성 게이트 / [[42]] prompt 무효의 귀결.

## 1. 학습 목표 행동 = 값 슬롯의 타입 규율
모든 값 슬롯(도구 인자·사용자-대면 주장)에서 허용되는 emit은 5가지뿐:
`FIND`(문맥 그대로 복사) · `GET`(산출 도구 호출) · `COMPUTE`(계산 도구 호출) · `ASK`(질문) ·
★**placeholder**(위 넷 불가 시 — 예: `{"__source":"ASK"}` / `<UNKNOWN:field>`) — **날조 대신 낼 대체 행동을 설치**.
- 채널 논리: 금지("하지 마라")=억제 채널=무효(C30/C47/`2606.07555`) / placeholder=**target 채널 공급**
  (Relign "우유부단 행동 부재가 날조의 원인"의 값-수준 판).
- ★**scaffold 접점**: placeholder는 결정론으로 탐지 가능한 **타입** → 엔진이 잡아 ASK 턴/GET 라우팅.
  학습(모델)과 게이트(엔진)가 같은 인터페이스에서 만남 = 두 날개([[00]])의 정확한 분담.

## 2. 데이터 설계 (도메인-일반·학습벤치서만)
| 성분 | 구성 | 출처·근거 |
|---|---|---|
| 양성 | 실 궤적의 grounded 값-채움(FIND 복사·GET 후 사용) | 4벤치+synth |
| **변조쌍** | grounding 원천 은닉(발화서 값 제거·도구반환 가림) → 정답이 채움→placeholder/ASK로 **뒤집히는 짝** | Relign 연산자 2종·배합 **4:3:3**(원본 40%=기권붕괴 방지) |
| **음성** | 이 세션 실측 날조 5종 템플릿: ①날조 record(placeholder 주소) ②가짜 케이스번호 ③가짜 도구명 ④가짜 프로세스(콜백/이메일) ⑤by_phone류 | §19·§14.3·**C43/D7: 창에 정박 재료(근접 id·그럴듯 값) 배치 필수**(C42: 재료 없으면 gradient 0) |
| 선호쌍 3종 | `grounded-채움 > placeholder/ASK > 날조-채움` — 첫 쌍=**과잉기권 방지 핵심** | Relign(C104) |
| **on-policy rejected** | 우리 32B를 실 결정점서 샘플→rejected 추출 (`bank_accum_probe` 인프라 일반화) | C38(off-policy DPO=likelihood displacement 실패) |

## 3. 왜 학습인가 (이 세션의 논리적 귀결)
프롬프트/금지문 무효(C30·C47·C99 98% mode-collapse·[[42]]) → 게이트=실행만 차단·재선택 못 삼(core) →
대안 공급=perseveration↓만·새 표면 엶(§20) ⇒ **잔여 = 행동 스킬 설치**. 값-grounding 규율=C92 연산 분류
=도메인-일반 → **ABox-swap 전이 대상**([[11]]·도메인-타깃 학습 아님).

## 4. 경고 (C104·위반 시 역효과)
1. think-형식 증류 경계(단독으로 날조 34.8→74.3%·Reasoning Trap). 2. 음성 사례 **처음부터**(사후 DPO=utility −24%).
3. 회귀 게이트 상설: SimpleToolHalluBench(592·무료) + 우리 실패-결정점 프로브를 체크포인트마다.

## 5. 착수 조건 = ★데이터 타당성 게이트 (C42/C38 교훈·순서 고정)
1. **eval 먼저**(무료): 실패 결정점 프로브 세트 — ①record-날조(§19.2 지점) ②케이스번호(③형 지점)
   ③by_phone(검증벽) ④인자누적(§17.1 지점·기확보) ⑤디스패처-prior(§19.1 지점).
2. base 32B가 eval서 **실제로 실패**함을 확인(라이브는 이미 실증·프로브로 정량).
3. **합성 학습 문맥이 그 실패를 재현**하는지 확인 — 재현율 낮으면 D7 재설계(정박 재료 보강).
4. 통과 후: LoRA SFT→DPO(리모트 A6000·**무료**·user-sim 불요) → held-out 결정점 전이 평가([[41]] 재사용).

## 6. 관계
- **record-grounding 검사(§19.3-1)와 상보**: 게이트=런타임 안전망(결정론) / 학습=행동 자체 교정.
  둘 다 같은 술어("값은 grounded여야")의 다른 집행 층 — 특허 D의 "선언+집행+학습" 3층 구조 후보.
- Track B(BANK_TRACK_B_SFT_DESIGN·F3 스키마-분류)와 인프라 공유(lora_train_metatool_v3·eval 패턴) —
  단 표적이 다름(Track B=NL→enum 분류 / 본 설계=값-grounding 규율). 통합 여부는 데이터 게이트 후 판단.
