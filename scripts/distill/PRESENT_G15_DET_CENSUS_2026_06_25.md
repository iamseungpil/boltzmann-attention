# present+g15 결정론 포렌식 회수 (2026-06-25) — [[08]] 전 trial first-divergence census (★clean·crash 배제)

> 회수 도구 `tau2/escape_det_census.py` (gold write ↔ 궤적 write 정렬·first-divergence 층·전 trial·user-sim 무관·`--clean`=non-user_stop 배제). 데이터 = retail t3 (101 gold-write task × 3 trial = 342 sim/조건). 비교 = floor / g15(precondition 단독) / present(σ-present 단독) / present+g15.

## ★포렌식 가드가 잡은 오염 (먼저)
- 초기(dirty) census는 **precondition-gate 조건에 집중된 infra_error를 MISS로 오집계**했다: termination=`infrastructure_error` (≠ 내가 처음 본 `infra_error`) — floor=0·present=0·**g15=12~13·present+g15=15~19**. 이 sim들은 **messages=0·reward_info=None = 빈 sim(아예 미실행)** → traj_writes 공집합 → 모든 gold write가 MISS로 계산됨.
- 결과: dirty에서 "present+g15 MISS 30→55 = 올바른 abstain"이라 본 것은 **artifact**. **clean하면 MISS=30=floor**(over-blocking 0). → 아래 표·결론은 모두 **clean(--clean)** 기준.
- infra 원인: 빈 sim·gate 조건에만 집중 → precondition-gate 경로의 *sparse* infra 결함 의심. 단 PM deny-replay-break(283/342 규모)와 다르고(여기 12~19/342), **배제가 올바른 처리**(측정 무효화 아님). [[추후] 게이트 init/replay 예외 조사 권장.]

## 결과 (clean·결정론 지표)

**32B (on_n32int8)**  *[excl=배제 sim 수]*
```
label        opCorr ordPick wMatch  over%   L0  L1  L2  L3  MISS MATCH  pass1 passAll  excl
floor         .887   .798    .695   30.4%   26  44  13  38   30   344   .528   .257     0
g15           .820   .747    .633   24.1%   36  35  14  41   50   303   .519   .290    12
present       .905   .859    .689   41.6%   22  23  16  68   25   341   .521   .267     0
present+g15   .902   .842    .731   30.2%   16  28  13  39   30   342   .601   .380    15
```
**14B (on_n14b / ours_n14b)**
```
floor         .803   .699    .583   27.9%   35  50  17  39   60   281   .444   .188     6
g15           .784   .662    .550   24.7%   39  56  20  31   60   252   .443   .180    16
present       .854   .782    .624   34.0%   31  35  16  60   39   300   .483   .267     9
present+g15   .857   .781    .658   29.9%   32  36  15  43   36   312   .497   .232    15
```
(passAll은 조건별 배제 task가 달라 *방향성*만·결정론 write 지표가 1차 신호.)

## 포렌식 판독 (조건 교차·메커니즘·양 스케일 일치)
1. **g15(precondition) 단독 = 순-음성** (양 스케일). over-action↓(32B -6.3pp·14B -3.2pp, 실재)이나 **L0↑**(wrong-operator 32B 26→36·14B 35→39)+wMatch↓ → 단독 게이트는 operator 오라우팅. pass≈flat.
2. **present(σ-present) 단독 = order-pick 결정론 robust 개선**(32B +.061·14B +.083·MATCH↑·L0 약간↓) **그러나 over-action 대폭 inflation**(32B +11.2pp·14B +6.1pp) — 후보리스트가 *여분 write* 유발(신규 발견). wMatch는 flat(이득이 over로 샘).
3. **present+g15 = net 승자(단조 아님)**: present의 order-pick 이득 유지(ordPick +.044/+.081) + g15가 present의 over-action inflation 상쇄(**over=floor 수준·32B -0.2pp·14B +1.9pp**) + **L0 최저**(32B 26→16) + **wMatch 최고**(+.036/+.075) + **MISS=floor**(over-block 0) + **robust pass 최고**(32B passAll .257→.380=**+12.3pp**·pass1 +.073/+.052).
4. 메커니즘: present=order-pick 공급·g15=present의 over-action 부작용 제거 + wrong-operator write 절감. 둘이 *상보적*. 단독으론 각각 부작용, 결합서만 순효과.

## 구체 궤적 정독 (5건·[[08]] 가드 항목4·메커니즘 확증·32B trial0)
- **task 2 (grounding 이득 실재)**: floor reward=0·**write 0건(MISS·미도달)** → present·present+g15 = 정답 return 1건(reward 1.0). 후보 제시가 *없었으면 안 했을* 올바른 write를 수행시킴 = present grounding 이득은 진짜 모델 행동.
- **task 58 (present+g15 회귀·비단조)**: floor·present=정답 → **present+g15 = wrong variant**(new_item_ids 3815173328→**3709608322**·L3)→reward 0. present+g15는 *단조 개선 아님* — variant(L3) grounding서 일부 회귀(census L3=39 잔여의 실체).
- **task 6 (present over-action 유해 실재)**: floor=정답 1건 → **present = exchange 2건(2번째가 *다른* payment: credit_card→paypal)**→reward 1.0→**0** (pass→fail). present over-action inflation(+11pp)은 진짜·때로 유해.
- **task 1 (over-action benign)**: present = *동일* exchange 2회(중복)·reward 1.0 유지.
- **★g15 메커니즘 확정**: precondition status-lock이 "이미 처리된 주문 재수정" 2번째 write를 차단 → present의 over-action(task6류)을 정확히 억제 = 결합서 over=floor 수준 회복의 메커니즘.

## [[06]] "G5=0" 결정론 재검
- pass^1로 ≈0이던 G5(precondition/eligibility 게이트)는 **결정론적으로 null 아님**: over-action 실감(-6.3pp)·write분포 변경. net-pass≈0은 over-action 억제와 L0↑가 *상쇄*된 것. → [[06]] "G5=0"은 pass^1이 가린 상쇄·결정론으론 실효(단 단독=net음성·present와 결합서 순기능).

## 부분-GO 판정 (make-or-break 맥락)
- scaffold(present+g15)=**작지만 진짜·robust한 결정론 이득**(order-pick·L0 최저·over 통제·wMatch 최고) + **유일하게 robust pass 개선**(passAll +12.3pp 32B). silver bullet 아님.
- **잔여(닫히지 않음)**: operand grounding(L2 item·L3 variant — present+g15서도 32B L3=39·14B L3=43 잔존)·⋈ 주소(101/102)·14B operator(L0≈floor). → 핸드오프 §2.4 "진짜 make-or-break = faithful-formalize *학습*" 재확인. scaffold=부하 offload·작동 스킬 carry(thesis §2 translator)·잔여 원천=학습.

## 다음 (핸드오프 §2 순)
2. operand grounding(item/variant) 학습없이: present를 `get_order_details`(items)·`get_product_details`(variants) 읽기로 확장(replay-safe 동일패턴) → L2/L3 잔여 닫나.
3. ⋈ 주소(present에 전체주소).
4. ★faithful-formalize 학습(잔여 원천·A2-규칙사용 SFT→tau2 A2-swap).
- (부수) precondition-gate sparse infra_error(빈 sim 12~19) 원인 조사.
- ★범위: A2_FRONTEND(NL→A2 자동생성)=별도 논문(2026-06-25 결정)→현 논문 제외. 현 논문=A2 수작성·고정 전제.
