# 다음 실험 설계 — test-time-compute(thinking)를 cost-optimal 능력 레버로 (noise-free isolated) 2026-07-07

> **위치**: `LEARNED_WING_MECHANISM_DESIGN` §4/§8의 **G1(prompted-CoT 버킷별)**을 (a)사용자 reframe(thinking=정직한
> 비용대안·1급 레버) + (b)이번 세션 발견(user-sim 노이즈→**isolated 측정이 노이즈-free**)로 격상·확장한 정본.
> **불변**: [[09]] 무료 isolated 먼저·[[13]] thinking은 학습보다 *테스트-비용*이 싸므로 *먼저* 검증(배포-싸다 아님)·
> [[05]] thinking=모델레벨 레버·캐논 base=Qwen2.5 불변·[[03]] 측정 먼저·build 금지·[[08]] per-case.
> **★rev2**: (1)A2=QwQ-32B (2)GO 2단계 (3)버킷 isolated-valid/load-only 분할 (4)§5 break-even $N^*$ (5)A1 고정규격.
> **★rev3(2026-07-07·2차 리뷰 반영)**: (A)**QwQ "순수 thinking/base 변화 0" 오버클레임 삭제** — QwQ=Qwen2.5 init+RL드리프트
> 라 A2−A0=thinking+RL *상한*·순수격리는 A1뿐(QwQ 유지·락·Qwen3은 교락 더 심함) (E)**⋈ Δ=0=경계-*의심*뿐**(대칭가드·
> under-spec null≠confirmed boundary·better-spec probe 후 fleet) (F)**A2−A1 토큰-매칭**(native>prompted가 토큰예산 교락)
> (§10)**산출=promise-맵+scoped 유료런 제안**(최종배분 아님·free-isolated 단독≠배포결론·coverage=범위밖).

## 0. 한 줄
잔여 pass(=부하/formalize)를 **test-time-compute(thinking)**가 싸게 닫는지를, **user-sim 노이즈가 없는 isolated-
decision 수준**에서 버킷별로 측정하고 **cost-per-pass-point**로 {thinking·scaffold·scale·learn} 배분을 확정한다.
compliance moat는 이미 settled(게이트=낙폭0)·이건 **capability 축**의 cost-optimal 답.

## 1. 동기 (이번 세션에서 확정된 것 위에서)
- **settled**: (a)compliance moat=게이트(frontier도 준수 흘림·게이트만 0). (b)잔여 pass=부하/formalize·**full-run
  pass^k는 user-sim 노이즈 지배**(67% flaky·flip 95% user-sim seed). (c)isolated 잔여=criterion-formalize 32B 72%/
  14B 67%(~scale-flat)·⋈ 32B 49%/14B 40%(scale-reducible)·argmax-기계적은 미미(sort 레버 dead).
- **사용자 reframe(정직한 최선)**: thinking이 부하를 싸게 완화하면 **1급 레버**. 근거: DR "test-time thinking이
  self-conditioning 완화"(2509.09677)·probe5 "CoT symbolic +17/+35"·**o4-mini(우리 frontier ref)=thinking 모델**
  → 우리 ~9pt pass 갭의 일부가 곧 "thinking". compliance는 thinking과 **직교**(reasoning 모델도 정책위반·게이트 필요).
- **★노이즈 해소 통찰**: user-sim 노이즈는 *full 멀티턴*서만 문다(reactive user-sim). **isolated 단일턴 decision은
  user-sim이 루프에 없어 재현가능·안정** → capability 효과(thinking이 닫나)는 여기서 노이즈 0로 측정. (full-run
  pass 비교는 common-random-user[§6] 인프라 필요·별도·후순위.)

## 2. 질문 (make-or-break for capability lever)
버킷별로: **test-time-compute가 isolated 잔여를 닫는가, 얼마의 토큰비용으로, scaffold/scale/learn 대비 최저인가?**
- **★2단계 판정([[08]] 재범 방지·isolated서 deployment 결론 도약 금지)**:
  - **(i) isolated Δ(thinking)>0 = *promise 신호*뿐** (진행 근거). isolated-valid 버킷(criterion·⋈)에서만.
  - **(ii) "learn/scale 불요" 같은 deployment 결론은 §8 (c) robust-partition full-run 확인 후로 유보.** isolated서
    thinking이 버킷 닫아도 **full-run pass 전이 보장 없음**(MAKEORBREAK: 잔여 지배=orchestration-under-load·isolated
    불가시). 지금 (i) 신호로 (ii) 결론 선언 = 반 칸 앞섬([[06]] 리뷰규율).
- **★못 닫음(Δ=0)도 2단계·대칭 가드([[03]]#9·null≠경계 직결 금지)**: **isolated Δ=0 = "경계-*의심*"뿐**. 특히 ⋈는
  probe under-spec(§4·40/49%=하한)이라 null이 probe 아티팩트일 수 있음. **confirmed boundary = CoT-delta 판별 +
  better-spec ⋈ probe 통과 후에만**. 그 전 fleet/scale 직결 금지(promise 방향에만 가드 걸고 경계 방향 무가드 = 06-NOW
  대칭규율 위반).

## 3. arms (isolated·residual decision 위·★전부 Qwen2.5-32B *계열*=base 불변)
| arm | 정체 | 격리하는 것 |
|---|---|---|
| **A0 base** | Qwen2.5-32B 직답(현 operand_probe·no CoT) | baseline |
| **A1 prompted-CoT** | 같은 모델 + 고정 CoT 프롬프트 | prompt-유도 test-time-compute(모델 불변) |
| **A2 native thinking** | **QwQ-32B**(Qwen2.5-32B init + RL/SFT 후학습 long-CoT) | **배포가능·최근접 lineage native-thinking 상한**(=thinking + QwQ RL드리프트·순수 thinking 아님) |
| (ref) o4-mini | 실-리더보드 frontier thinking(이미 데이터 有·신규런 0) | thinking 천장(비교 기준·probe엔 미사용·유료 회피) |
- **★순수 격리는 A1이 담당(A2 아님·rev3 오버클레임 정정)**: A1=**실제 동일 가중치 + 프롬프트**뿐 → *유일한* 순수
  test-time-compute arm. QwQ는 Qwen2.5-32B init을 공유하나 대규모 RL/SFT 후학습으로 가중치가 이동한 **별개 모델** →
  **A2−A0 = thinking + QwQ 고유 RL 드리프트**(thinking 단독 격리 아님·1라운드 base-change 교락과 같은 계열·lineage
  공유라 더 약함). ∴ 옛 "순수 thinking/base 변화 0/동일 base 순효과" **삭제** → A2=**"배포가능·lineage 최근접
  native-thinking *상한* probe"**로 재캡션. **A2−A0=상한(thinking+RL)·A2−A1="native가 prompted 넘나"(단 F).**
- **★A2=QwQ 확정(락·[[03]] 오실레이션·모델 3번째 왕복)**: Qwen3-32B로 가면 pretraining 계보까지 달라 A2−A0 교락이
  *더 심함* → 발견 A는 "Qwen3 가라" 신호가 아니라 "캡션 고쳐라" 신호. QwQ가 우리 base lineage 최근접이라 상한 probe로
  최적. 59.6=자기보고(rev2 폐기). (Qwen3 하이브리드 ON/OFF의 "드리프트0 native-thinking delta"가 필요하면 Phase B
  GPU 여유시 *보조 ref*로만 2점 추가·A1 대체 아님·필수 아님 — A1[순수]+QwQ[상한]+o4-mini[천장]로 질문 이미 닫힘.)
- isolated 단일턴 probe는 tool-call 불요 → QwQ agentic-포맷 약점 무관. 항상-ON thinking도 A2(상한) 목적에 부합.
- **★F(토큰예산 교락)**: A1(900 고정) < QwQ(수천 토큰)이면 "A2>A1"이 native-RL 덕인지 토큰 더 씀인지 미구분. ∴ "native
  > prompted" 주장엔 **토큰-매칭 비교**(A1 고-토큰예산 변형 1개) 또는 **Δpass-vs-토큰 커브**(§5) 필수.
- **★A1 규격(재현·cost 계산)**: 고정 `max_tokens`(예 900·probe5와 동일)·고정 CoT 프롬프트·truncation 카운트(0 확인).
  "충분 토큰"식 미정의 금지.
- **★A1 null≠thinking 무효([[42]])**: prompted-CoT는 prompt-adherence 천장에 걸릴 수 있음(소형=규칙프롬프트 불복종).
  따라서 **A2(native)는 확인용이 아니라 필수 교차** — A1이 0이어도 A2가 닫으면 thinking-레버는 살아있음(설치 필요).
- **o4-mini 위치**: 유료 API·on-prem 배포불가 → probe arm 부적합([[09]]). 이미 보유한 pass^1..4+compliant로 **frontier
  thinking 천장**만 담당(사용자 "리더보드값 모델과 비교" 제약을 o4-mini가 충족).

## 4. 버킷 + ★isolated-유효성 분할 (결론 범위를 버킷별로 제한·[[08]])
각 버킷 decision을 isolated 단일턴으로 A0/A1/A2에 → **정답률 + thinking 토큰수**. **단 isolated가 유효한 계측기인
버킷과 아닌 버킷을 먼저 가른다** (MAKEORBREAK: GIVEN-SPEC operand=100%·잔여 지배=orchestration-under-load라
isolated서 안 보임):
| 버킷 | isolated 유효? | 근거 |
|---|---|---|
| **criterion/variant-formalize(GOAL)** | **✅ isolated-valid** | MAKEORBREAK도 GOAL 70%(vs GIVEN-SPEC 100%) 인정=criterion 해석은 단일턴서 관측됨. isolated 계측 정당. |
| **⋈ reference-match** | ◐ 부분 | 후보 present되면 단일턴 관측 가능(operand_probe 40/49%)·단 probe under-spec 주의(하한). |
| **coverage(for-all 전수)** | ✗ load-only | "모든 X 전수"는 본질적으로 멀티스텝 부하 → isolated 단일턴서 재현 안 됨. thinking 효과는 **full-run서만** 판정. |
| **calc(order-total)** | (음성대조) | 이미 결정론 scaffold(토큰0). **thinking arm의 음성 대조** — 여기서 thinking 이득=scaffold 미발화 신호지 thinking 승리 아님. |
- ⇒ **thinking 결론은 isolated-valid 버킷(criterion·⋈)에만**. coverage는 isolated 결과로 판정 금지(load-only).
- **⋈ 판별자 = CoT delta**(투표 아님): probe6 "self-consistency +0·8/8 동일오답"은 **투표=vacuous** 증거이지 ⋈-
  reachability 판정 아님(R-1 정정·06-NOW). 판별=A1/A2 CoT가 여는가.

## 5. 지표 (cost-optimal 핵심·★쿼리-볼륨 amortization)
- **버킷별 isolated 정답률**(A0/A1/A2) + **Δ(thinking 순효과)**. per-case([[08]])·noise-free(재현가능).
- **cost-per-pass-point** = thinking 토큰비용 / Δpass. **★단 thinking과 learn은 비용 성격이 달라 쿼리-볼륨 축 필수**:
  - **thinking = 쿼리마다 반복 토큰비용**(추론시 CoT 토큰). learn(SFT) = **1회 학습비 + 무료(짧은) 추론**. scaffold = 결정론(토큰≈0). scale = 큰모델 $/토큰.
  - **break-even 쿼리수** $N^*$ = SFT 1회학습비 / (thinking 쿼리당 추가토큰비 − SFT 쿼리당비). **$N < N^*$면 thinking, $N > N^*$면 learn**이 싸다. 쿼리-볼륨 없이 "thinking=토큰비" vs "learn=1회+무료" 나란히 두면 thinking이 인위적으로 싸 보임 → 반드시 $N^*$ 병기.
- **★F(A2−A1 토큰 교락)**: "native(A2) > prompted(A1)" 주장은 A1(900) < QwQ(수천) 토큰차와 교락. ∴ (i)A1 **고-토큰예산
  변형 1개** 추가(토큰-매칭) 또는 (ii)**Δpass-vs-토큰 커브**로 토큰당 이득을 arm 간 비교. 둘 없이 A2>A1을 native-학습 덕으로 귀속 금지.
- **판독**: symbolic criterion=CoT-reachable 예상(닫힘·thinking 유력)·⋈ Δ=0은 **경계-의심**(under-spec probe 하한 →
  confirmed 아님·better-spec ⋈ probe 후 판정·§2 대칭가드).

## 6. 시퀀싱 (무료 먼저·[[09]])
1. **Phase A (무료·즉시·GPU有)**: A0 vs **A1 prompted-CoT**를 residual 버킷에 (operand_probe에 CoT 배선). 32B 서버
   이미 up(GPU0)·gpt-4.1=0. per-bucket Δ + 토큰수. = **G1 실행**.
2. **Phase B (저·GPU1 free)**: **A2 native thinking = QwQ-32B** 서빙(GPU1)·같은 isolated decision. gpt-4.1=0.
   native thinking이 A1(prompted)보다 더 닫나(A1 prompt-ceiling 우회) + 토큰비용.
3. **Phase C (분석)**: cost-optimal 레버 맵 — isolated-valid 버킷마다 {thinking/scaffold/scale/learn} 비용배분 + break-even $N^*$(§5).
4. **★2단계 GO/NO-GO (§2)**:
   - **(i) isolated promise**: isolated-valid 버킷서 Δ(A1 또는 A2)>0 = **진행 신호**뿐(deployment 결론 아님).
   - **(ii) deployment 판정 = §8 (c) robust-partition full-run 확인 후**. "thinking으로 capability 닫음·learn 불요"는
     여기서만 선언. coverage(load-only)는 full-run서만 판정. **⋈ Δ=0=경계-의심뿐**(under-spec)→better-spec ⋈ probe
     통과 후에만 thinking·RLVR 둘 다 의문(fleet)로 확정(§2 대칭가드).

## 7. 실험 안정성(user-sim 노이즈)과의 관계 — 이 설계가 해소
- **capability 측정=isolated → 노이즈 0**(이 설계). 결론은 재현가능.
- **full-run pass 비교(scaffold/thinking 효과의 end-to-end 확인)**만 user-sim 노이즈 문제. → **common-random-user
  paired 프로토콜**(§8) 필요·단 **어떤 레버가 isolated서 promise 보인 뒤**로 후순위([[09]]). 지금 안 지음.

## 8. (부속·후순위) full-run 안정성 프로토콜 = common-random-user
- 문제: user-sim(gpt-4.1) temp0도 비결정→arm 간 다른 대화(flip 95% seed). full-run pass^k ±5pp 노이즈.
- 후보: (a)**teacher-forced user replay**(floor의 user turn을 scaffold arm에 재생·divergence 전까지)=부분상쇄·reactive
  한계. (b)**pass^1 + paired bootstrap CI**(다수 trial·평균 안정)·nt↑. (c)**robust-partition 신호**(scaffold가
  robust-fail→robust-pass 몇 개 전환·flaky 중간층 제외). 권장 조합=(b)+(c)·(a)는 연구노트. **isolated 레버 GO 후 착수.**

## 9. 자가감사
- [[05]]: thinking=모델레벨(도메인 scaffold 아님·grep if domain 무관). **캐논 base=Qwen2.5 불변**·A2(QwQ)는 base
  교체가 아니라 "native thinking이면 상한 얼마"의 비교 probe(QwQ=Qwen2.5-32B lineage 최근접이나 RL드리프트 有 →
  A2−A0=thinking+RL 상한·순수격리는 A1). clean.
- [[13]] **"먼저"의 정확한 의미 = 테스트-비용이 가장 싼 레버 먼저**(thinking probe=무료 GPU / SFT=GPU학습). **배포-비용(TCO)이
  싸다는 뜻 아님** — 배포선 쿼리량 크면 SFT가 amortize해 thinking을 이길 수 있음(§5 $N^*$). 순서=검증비용 기준.
- [[03]]: build 아님·측정. LEARNED_WING G1 격상(재발명 아님)·G2(RLVR)의 **싼 상계**(thinking⊂reachability: thinking
  으로도 안 열리면 RLVR도 의문). #8 forensic-정합(잔여 버킷 = clean-nt4 forensic).
- [[09]]: Phase A/B 전부 로컬(gpt-4.1=0)·full-run 유료는 최종·common-random-user 후.
- **thesis 정합**: cost-optimal 레버 배분(특허 목적문)에 **thinking을 1급으로 정직 편입**·compliance moat 불변(직교).
- **cheating-surface**: isolated probe는 present 후보만·gold 미접근(operand_probe 동일 규율). thinking=모델 자체 추론
  (scaffold가 답 안 박음).

## 10. 산출물 (★범위 정직·rev3 하향)
**Phase A/B(무료·isolated)의 산출물 = *promise-맵* + *scoped 유료런 제안*이지 최종 배분 아님.** 근거: (ii) deployment
판정은 §8 full-run(=gpt-4.1 유료·[[09]] 예산킬러) 뒤라, **무료 isolated 실험은 자체로 배포결론에 절대 도달 못 하고
유료 full-run을 greenlight/scope만 한다**(정직).
- **산출**: isolated-valid 버킷(criterion·⋈) × arm(A0/A1/A2/o4-mini-ref) 정답률표 + Δpass-vs-토큰 커브(§5·F) + break-even
  $N^*$ 추정 → **promise-맵**(어느 버킷이 thinking으로 열릴 *가능성* + 그 버킷 full-run 확인의 scope/예산).
- **범위 밖 명기**: **coverage(clean-nt4 잔여 17/16)=load-only → 이 실험 범위 밖**(full-run서만·§4). 최종
  "thinking/scaffold/scale/learn 배분"은 full-run robust-partition(§8) 후에만 확정.
- 도구=`operand_probe_n100.py`+고정 CoT 플래그(§3)·`cot_probe5/6.py`(기존). QwQ 서빙=GPU1.
