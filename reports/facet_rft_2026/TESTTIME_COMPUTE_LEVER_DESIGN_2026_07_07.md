# 다음 실험 설계 — test-time-compute(thinking)를 cost-optimal 능력 레버로 (noise-free isolated) 2026-07-07

> **위치**: `LEARNED_WING_MECHANISM_DESIGN` §4/§8의 **G1(prompted-CoT 버킷별)**을 (a)사용자 reframe(thinking=정직한
> 비용대안·1급 레버) + (b)이번 세션 발견(user-sim 노이즈→**isolated 측정이 노이즈-free**)로 격상·확장한 정본.
> **불변**: [[09]] 무료 isolated 먼저·[[13]] thinking은 학습(G2 SFT/RLVR)보다 싸므로 *먼저* 검증·[[05]] thinking=
> 모델레벨 레버(도메인 scaffold 아님)·[[03]] 측정 먼저·build 금지·[[08]] per-case.

---

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
- 닫으면(cheap) → capability 답=thinking + 게이트(compliance). 학습(G2)·scale 불요.
- 못 닫으면(⋈ systematic·probe6 self-consistency +0) → genuine 경계 → learn(G2 reachability)·scale·fleet.

## 3. arms (isolated·residual decision 위)
| arm | 정체 | 격리하는 것 |
|---|---|---|
| **A0 base** | Qwen2.5-32B/14B 직답(현 operand_probe) | baseline |
| **A1 prompted-CoT** | 같은 모델 + "step-by-step 신중 추론"(충분 토큰·probe5 style) | **순수 test-time-compute**(모델 불변) |
| **A2 native thinking** | QwQ-32B 또는 Qwen3-32B(thinking on) | 배포-현실적 thinking 모델(모델+학습 변화) |
| (ref) A3 scale | 더 큰 base(있으면) | scale 대조 |
- A1=깨끗한 레버 격리(동일 모델). A2=실사용 옵션. non-thinking base와 A/B로 thinking 순효과 측정.

## 4. 버킷 (clean-nt4 forensic·`operand_probe`·isolated)
criterion/variant-formalize(지배·scale-flat) · ⋈ reference-match(scale-reducible·systematic 의심) · coverage(for-all
전수) · calc(order-total·이미 scaffold). 각 버킷 decision을 isolated 단일턴으로 A0/A1/A2에 → **정답률 + thinking 토큰수**.

## 5. 지표 (cost-optimal 핵심)
- **버킷별 isolated 정답률**(A0/A1/A2) + **Δ(thinking 순효과)**. per-case([[08]])·noise-free(재현가능).
- **cost-per-pass-point** = thinking 토큰비용 / Δpass. scaffold(≈결정론·토큰0)·scale($/토큰)·learn(1회학습+무료추론)과 비교.
- **판독**: symbolic criterion=CoT-reachable 예상(닫힘·thinking 유력)·⋈=systematic이면 thinking도 못 닫음(경계).

## 6. 시퀀싱 (무료 먼저·[[09]])
1. **Phase A (무료·즉시·GPU有)**: A0 vs **A1 prompted-CoT**를 residual 버킷에 (operand_probe에 CoT 배선). 32B 서버
   이미 up(GPU0)·gpt-4.1=0. per-bucket Δ + 토큰수. = **G1 실행**.
2. **Phase B (저·GPU1 free)**: **A2 native thinking**(QwQ-32B 또는 Qwen3-32B-thinking) 서빙(GPU1)·같은 decision.
   gpt-4.1=0. thinking 모델이 A1보다 더 닫나 + 비용.
3. **Phase C (분석)**: cost-optimal 레버 맵 — 버킷마다 {thinking/scaffold/scale/learn} 최저비용 배분 확정.
4. **GO/NO-GO**: thinking이 버킷 닫음 → capability=thinking+게이트(learn G2 불요·저비용). 못 닫음 → G2 reachability(RLVR)·
   또는 경계(fleet). ⋈-systematic이 핵심 분기.

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
- [[05]]: thinking=모델레벨(도메인 scaffold 아님·grep if domain 무관)·QwQ/Qwen3=모델 swap. clean.
- [[13]]: thinking(추론시 비용)은 학습(G2 SFT/RLVR·GPU학습)보다 싸므로 **먼저**·scale은 대조. 순서 정합.
- [[03]]: build 아님·측정. LEARNED_WING G1 격상(재발명 아님)·G2(RLVR)의 **싼 상계**(thinking⊂reachability: thinking
  으로도 안 열리면 RLVR도 의문). #8 forensic-정합(잔여 버킷 = clean-nt4 forensic).
- [[09]]: Phase A/B 전부 로컬(gpt-4.1=0)·full-run 유료는 최종·common-random-user 후.
- **thesis 정합**: cost-optimal 레버 배분(특허 목적문)에 **thinking을 1급으로 정직 편입**·compliance moat 불변(직교).
- **cheating-surface**: isolated probe는 present 후보만·gold 미접근(operand_probe 동일 규율). thinking=모델 자체 추론
  (scaffold가 답 안 박음).

## 10. 산출물
버킷×arm 정답률표 + cost-per-pass-point 맵 + GO/NO-GO(버킷별 thinking/scaffold/scale/learn 배분). 정본 doc + 덱/특허
"cost-optimal 레버 맵(thinking 포함)" 갱신 근거. 도구=`operand_probe_n100.py`+CoT 플래그·`cot_probe5/6.py`(기존).
