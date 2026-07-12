# tool-use 레버 생사 지도 — thinking의 경계·horizon 재프레임·비용효율 조합 (설계 2026-07-12)

> 상위 = `RESEARCH_MASTER.md`(F1-F6 프레임·제1원리 상쇄). 선행 = `SCALE_DYNAMIC_CONTAMINATION_PRIORWORK §8`(2509.09677 정독) · `OVERNIGHT_RESULTS §결과4`(C69 동적바인딩) · C56(동-scale thinking ⋈ 못삼) · C53/C62(prov·게이트 e2e).
> 규율: [[05]] A2만 변경·엔진 도메인-일반 · [[08]] 집계→결론 前 per-case · [[09]] 무료 우선(로컬 사다리·결정론gold·user-sim 0). **본 문서 = 설계 + 무료 실험 큐**(유료 full-run 없음).
>
> **★★정렬 배너 (2026-07-12·후속 실험 반영·[[48]] 명명통일·문서 동시정렬)**:
> - **(A·확정 [M])** 본문의 **"self-conditioning" 기전 라벨은 전부 "상태-발산(state-divergence)"으로 교체**(F9 재분석·inject 자기일관 0.969@32B). 정본 = `UNIFIED_TAXONOMY 근본기능6 지속`의 2기전 중 sd. 2509.09677 self-conditioning은 distinct mode·인용만. **명명 = 근본기능 서술형(M-코드 금지·[[48]]).**
> - **(B·[진행·재검토])** 본문 §1·§10의 **"thinking은 외부오염(near-miss)엔 무력·축a만"은 E-THINK2 예비가 반증**(파싱수정후 4b L4 0.28→0.97). **8b/14b 확정 시 F3 분기②(비용-우위 서사)로 §1 decidability 3분할·방화벽 표 축c/thinking행 동시 갱신**. 그전까지 이 주장 [진행]·[M] 단정 금지(F8 교훈).

## 0. 재프레임 (사용자 지시 2026-07-12·정밀화)
1. **우리는 선행의 역(逆)이다.** 선행(2509.09677·METR·scaling law)은 "**scale이 능력 X를 어떻게 사는가**"를 묻는다. 우리는 반대다 — "**tool-use의 어느 세부능력이 scale보다 *싼 것*으로 사지는가, 그리고 그 조합을 어떻게 비용-최적화하는가**". 핵심 주장 = **특정 tool-use 영역에서 scale을 {retrieval·verify-scaffold·결정론 게이트·도메인-일반 learn}으로 대체하는 것이 비용-효율적이다.**
2. **★주장 정밀화(중요·기존 정정)**: "scale이 horizon을 *못* 산다"는 **틀린 강한형**이다. 정확히는 —
   - scale은 per-step을 **개선한다**(banking gemini2.5pro 0.748→GPT-5.5 0.887·retail operand→포화). "전혀 못 산다" 아님.
   - **그러나 (a) 느리고 (b) 상용 임계(banking pass=0.7엔 p≥0.956)에 못 미치며 (c) 비싸다**(frontier API·요청당 영구비용·폐쇄망 불가).
   - ⇒ **주장 = "scale로는 상용 수준을 *충분히·싸게* 살 수 없다 → 같은 per-step을 다른 방법(scaffold/verify/retrieval)으로 *더 명확하고 싸게* 산다"**. horizon은 "scale 영토"가 아니라 "**더 싼 레버로 극복하는 영역**".
3. **필요한 증거(E-HORIZON/E-VERIFY)**: per-step 정확도 곡선 — scale(완만·상용 미달) vs scaffold/verify(급격·상용 도달·pass-cost 0).

## 1. thinking의 경계 = decidability 3분할 (핵심 구조)
실패를 유형으로 가르면 각 레버의 생사가 결정된다. **thinking = 내부 test-time compute → 재계산으로 잡히는 자기-오류만 닫는다.**

| 실패 유형 | thinking | verify-scaffold | retrieval | scale | learn | 근거 |
|---|---|---|---|---|---|---|
| **자기-생성 계산오류**(running-sum 오산·조립 순서) | ✅ 재계산 | ✅ 재계산-체크(교정) | — | ✅ | ✅ | 2509.09677 |
| **참조/검색 오류**(near-miss 값 오바인딩·미조회 날조) | ❌ *틀린 anchor 위 더 생각해도 안 고침* | ✅ DB/provenance 대조 | ✅ fetch-first | ❌(C69·C36) | 부분(cfbsynth) | C69·C43·C29 |
| **의미 모호성**(사용자 의도 ⋈·criterion) | ❌(C56) | ❌ *대조할 ground-truth 없음* | — | 부분(포화 ~.44) | 부분 | C56·C3b·C51 |

**왜 이렇게 갈리나(제1원리):** thinking은 *더 계산*할 뿐 *없는 정보를 만들거나*(retrieval) *외부 오염을 밀어내거나*(near-miss) *진짜 모호를 해소*(⋈)하지 못한다. verify는 *ground-truth 대조*가 가능한 곳(계산 재현·참조 실재)만 닫고, 대조 대상이 없는 의미모호엔 무력하다. ⇒ **thinking과 verify는 서로 다른 decidability 조각을 닫는다**(thinking=계산-재현가능·verify=참조-대조가능). 교집합=자기-생성 계산오류, 여집합=참조오류(verify만)·의미모호(둘 다 무력·scale/learn/ASK 잔여).

## 2. horizon 재프레임 (사용자 가설의 형식화·실증 설계)
### 2.1 선행 모델과 우리 반론
2509.09677: `P(H단계 완주) ≈ ∏ₜ p_t`. scale이 `p_step`을 사서 horizon을 산다(미세 p 개선→지수적 길이). **단 self-conditioning이 `p_t = p_0 − f(문맥 내 누적오류)`로 단계마다 하락시켜 *초-지수* 붕괴**(2509.09677 Fig5b).

### 2.2 사용자 가설 = "짧은 단계로 정확히 verify하면 긴 horizon 생존?" → 기전적 YES
per-step 결정론 verifier(recall r·교정)가 오류를 **문맥 진입 전 제거** →
- 문맥 유입 오류율 = `(1−p)(1−r)`. r→1이면 **self-conditioning 루프 미발동** → `p_t = p_0` 유지 → 최악에도 독립 `p_0^H`(clean 곡선)·교정하면 유효 `p→1` → **임의 길이 생존**.
- **scale과 무관**(verifier=결정론) → 소형+verify도 긴 horizon.

### 2.3 2509.09677과 수렴·차별
그들 "thinking이 self-conditioning 고침"과 **같은 기전(루프 절단)**: thinking=자기오류를 *commit 안 함* / 우리 verify=*교정으로* 끊음. **둘 다 self-cond 절단이나 verify가 (a) 더 싸고(pass-cost 0) (b) scale-불변 (c) thinking이 못 닫는 참조오류도 닫는다**(§1). ⇒ **horizon = scale 영토가 아니라 verify-scaffold 영토.**

### 2.4 ★통합 주장 (신규·아직 명시 안 함)
per-step 결정론 검증이 **하나의 장치로 둘을 산다**: (a) 준수 보장(compliance·C1) + (b) self-conditioning 절단으로 horizon 구매. 우리 기존 scaffold(provenance verify·게이트·calc-offload)가 *이미* per-step verifier다 → **"compliance-scaffold가 horizon도 산다"**는 우리 결과의 재해석. C53(prov e2e)·C62(COMP)의 pass^k가 이 관점서 재분석 대상.

## 3. tool-use 레버 생사 지도 (목적 = 비용효율 조합)
세부능력별 최저비용 생존 레버('?'=실험 대상):

| 세부능력(근본기능) | scale | thinking | retrieval | verify-scaffold | 게이트 | learn | 현 판정 |
|---|---|---|---|---|---|---|---|
| operand 실행(F2 실행) | 포화 32B | — | — | — | — | — | **base**(C-B§3.2 88/88) |
| gather/검색·날조억제 | frontier만 0 | ❌ | ✅ fetch-first | ✅ provenance | — | 부분 | **retrieval/verify**(C29·C45) |
| horizon/self-conditioning | 산다 | 산다(비쌈) | — | ✅ **가설(E-HORIZON)** | — | — | **verify(가설)** vs scale/thinking |
| compliance(F1) | ❌ 불변(C1) | ❌ 직교 | — | — | ✅ pass-cost0 | — | **게이트**(C1/C53) |
| near-miss 바인딩(F5 참조·동적) | ❌(C69) | **? E-THINK** | 부분(prov) | **? E-VERIFY** | — | ? | **미확정→실험** |
| ⋈/의미참조(F3) | 부분 포화 .44 | ❌(C56) | — | ❌ | — | 부분(경계) | **scale/learn 잔여**(C3b) |
| coverage/집합완결(F4) | 평탄(C-B) | ? | — | — | ✅ controller/E-PLAN | — | **게이트/controller** |

'?' 칸 = §4 실험이 닫는다.

## 4. 무료 실험 큐 (E-REF 인프라 재사용·결정론gold·user-sim 0·[[09]] 무료)
### E-THINK-1 — thinking × 오염축 (near-miss가 thinking으로 닫히나)
- **무엇**: C69 E-REF near-miss/paraphrase 프로브를 **thinking arm**(QwQ-32B·Qwen3-thinking 사다리)으로 재실행 vs base.
- **가설/예측**: thinking은 near-miss(외생 오염)를 **못 닫는다**(§1 참조오류=thinking 무력) — 재계산할 자기-오류가 아니라 틀린 anchor 포획. **닫으면**(부분 재-read 효과) 그것도 정보 — thinking이 "재조회 트리거"로 작동하는지 판별.
- **비용**: 무료(로컬·결정론gold·기존 `eref_probe.py` + thinking 모델).

### E-HORIZON-1 — per-step verify × horizon (사용자 핵심 가설)
- **무엇**: 2509.09677 running-sum key-value task 복제(소형 사다리 0.5B–7B 보유). arm = {base · +thinking · **+per-step 결정론 verify-and-correct** · +scale}. horizon(임계 <80%/<50%까지 단계수) 측정.
- **가설**: **base+verify ≈ large/thinking horizon**을 비용의 일부로. 확인 시 = "horizon은 verify-scaffold로 산다" 실증.
- **비용**: 무료(합성·결정론).

### E-HORIZON-2 — self-conditioning 절단 기전 격리 (verify가 *왜* 사는가)
- **무엇**: verify arm ablation — (a) detect-only(플래그만·오류는 문맥 진입) vs (b) detect-and-correct(진입 전 제거). (b)≫(a)면 **기전=self-conditioning 절단**(단순 정확도 아님) 증명.
- **비용**: 무료.

### E-LEVER-BAKE — 레버 생사 전수 bake-off
- **무엇**: E-REF 바인딩 task × {오염축 a/b/c/d} × {base·thinking·retrieval·verify·gate·scale-사다리} × 비용. §3 '?' 칸 전부 채움 → 비용효율 조합 지도 확정.
- **비용**: 무료(대부분 기존 프로브 조합).

## 5. 비용효율 조합 원리 (지도의 결론·가설 포함)
세부능력별 승리 배치(§3 지도):
- 검색/gather → **결정론 fetch-first**(scale 아님·C29)
- horizon/self-conditioning → **per-step 결정론 verify**(scale 아님·thinking보다 쌈·E-HORIZON 확인 시)
- compliance → **결정론 게이트**(scale 아님·C1)
- operand → **base**(포화)
- coverage → **controller/게이트**(E-PLAN)
- ⋈/의미참조 → **유일한 진짜 scale/learn 잔여**(그 외 전부 싼 레버로 대체)

⇒ **"scale을 싼 레버로 대체" 명제의 완성형**: horizon을 "scale 영토"에서 "verify-scaffold 영토"로 이동시키면, scale이 진짜 필요한 곳은 **의미참조(⋈) 소수 잔여뿐**. 이게 우리가 선행의 역(§0-1)인 이유 — 선행은 scale 사는 법, 우리는 scale 대체하는 법.

## 6. 다음 (우선순위·전부 무료 설계·구현)
1. **E-HORIZON-1/2 우선**(사용자 핵심 가설·2509.09677 정면 대응·synth라 빠름). 확인 시 = 논문/특허의 horizon 재프레임 근거.
2. **E-THINK-1**(thinking 경계 실증·near-miss). §1 3분할 검정.
3. **E-LEVER-BAKE**(조합 지도 완성).
4. 확인 후 → RESEARCH_MASTER §3 원장 등재 · 논문/특허/덱 horizon 재프레임 반영(현 "scale이 horizon 산다"→"verify가 horizon 산다·scale은 ⋈ 잔여만").
- ⚠️ verify-scaffold = 결정론 per-step 검증기 = **[[05]] 고정 엔진**(도메인-일반·running-sum은 재계산·tool-use는 provenance/gate). 도메인 리터럴 0 유지.

## 7. ★banking = horizon 재프레임의 실증 증거 (기존 데이터 재해석·2026-07-12·[[47]] 재런 없음)
2509.09677은 *합성* task서 "scale이 horizon 산다"를 보였다. **banking(실제 tool-use·τ³·17 frontier 모델)은 그 반대의 실증** — scale이 긴 horizon을 *못* 산다. 데이터 = `TAU2_FRONTIER..._MASTER §3.2f`([M]·17모델·4632 실패·[[08]] 완주·`banking_forensic.py`/`banking_argdiff_census.py`).

### 7.1 관측 (기존 [M])
- frontier banking pass **0.098~0.384**(17모델·gpt-5.2 sim·GPT-5.5 최강 0.384·Opus4.5 0.245·gemini2.5pro 최약 0.098). gold 절차 **median 8**(universal-fail 12.5)=4도메인 최장.
- **28/97(29%) universal-fail**(17모델×4trial 전패)·전부 `unlock_discoverable_tool→call` 발견체인 요구.
- pass가 gold 길이에 단조 급락: 1-3act 0.442→10+ **0.079**. 종료=user_stop 92%(crash 아님·"다 했다" 오종료).
- **★규정(기존)**: banking 저-pass = **3중 부하의 곱**(긴 horizon × 발견체인 × all-or-nothing DB) ⇒ **per-step p<1의 지수붕괴 p^H**.

### 7.2 ★per-step 엄밀 도출 (pass = p_step^H·H=8·**전 18 frontier 모델**·submission.json 캐시)
전 2026 frontier submission.json서 banking_knowledge pass_1 추출·per-step=pass^(1/8) 도출(`bk2.sh`·데이터 `sim_results/banking_perstep_frontier_2026_07_12.txt`):

| 모델(대표) | pass_1 | 함의 p_step | per-step err |
|---|---|---|---|
| qwen3.5-397b·glm-5 (retail 챔피언·banking 최약) | 0.098 | 0.748 | 25.2% |
| gemini-2.5-pro / grok-4 | 0.12~0.18 | 0.77~0.81 | ~20% |
| Opus 4.5/4.6/4.7 · GPT-5.2 | 0.21~0.25 | 0.83~0.84 | ~16% |
| **GPT-5.5 (18모델 최강)** | 0.374 | **0.884** | **11.6%** |
| *상용 pass=0.7 도달 필요* | 0.700 | **0.956** | **4.4%** |

- **★전-frontier per-step = 0.748~0.884 (spread 13.6pp)**. per-step은 frontier 진전으로 *개선된다*(느리게·실측). "전혀 안 준다"는 틀림([[08]] 정직).
- **★그러나 상용 미달·전멸**: **0/18이 pass=0.7 도달**(p=0.956 필요)·**0/18이 pass=0.5도 미달**(p=0.917). 최강 gpt-5-5도 err 11.6%=상용(4.4%)의 **2.6×**. 0.884⁸=0.373에 갇힘(H=8). ⇒ **전 frontier가 상용 임계 아래**.
- **★능력축 도메인-특이(C52 정합)**: retail 챔피언(qwen3.5-397b 0.855·glm-5)이 banking **최약(0.098)** — scale이 사는 축이 도메인마다 다름(단 glm/qwen banking엔 too_many_errors 아티팩트 성분·§3.2f).
- **대조 retail**: p_step~1.0(operand 포화·C-B §3.2)·H~4-5 → pass~0.95. 차이는 전적으로 banking per-step이 ~0.88서 **정체**(구조적 실패·§7.3).

### 7.3 ★정체 원인 = 구조적 per-step 실패 (scale-불변 잔여)
frontier가 못하는 것(§3.2f-5 census·모델 기울기): 규모·신형화는 **미실행(reach/조립)을 *부분* 사되**(최약 미실행 68%→최강 gpt55 미실행 아님), **결정가능-인자 오류(calc·enum·schema·copy)가 잔여로 남는다**. 둘 다 **구조적**(무작위 계산정확도 아님):
- **reach/unlock 발견체인**(universal-fail 28/28) = §1 참조/조립 오류 → thinking ❌·**verify/controller ✅**
- **결정가능-인자**(≈40%·calc/enum/id/schema) = §1 계산·참조 오류 → **calc-offload·provenance ✅**
이 둘이 정확히 **결정론 scaffold가 p→1로 만드는 것**(gather/unlock controller·calc·provenance·coverage 게이트). scale은 못 만든다(잔여로 남음).

### 7.4 결론 — banking이 §2 정밀화된 주장을 확증 (★기존 주장 정정)
**★기존 주장 정정([[40]]·[[08]])**: C71 초판·§7 헤더의 "scale이 horizon을 *못* 산다"는 **틀린 강한형**이었다. banking 데이터가 실제로 보이는 것 = **scale은 per-step을 개선하되(0.748→0.887·err 25%→11%) 상용 임계(p≥0.956·err 4.4%)에 *한참 못 미친다*** — 17 frontier 최강조차 err 11.3%=2.6× 초과. 즉 **"불가능"이 아니라 "느리고·불충분하고·비쌈"**.
- 정체 잔여 per-step 실패 = **구조적**(reach/unlock 발견체인 + 결정가능-인자). 구조적 per-step은 **결정론 scaffold가 p→1로·pass-cost 0·scale-불변**하게 만든다(gather/unlock controller·calc·provenance·coverage).
- ⇒ **주장 = "scale로는 banking 상용 수준을 충분히·싸게 못 산다 → 같은 per-step을 scaffold로 더 명확하고 싸게 산다"**(§2-2). horizon = "더 싼 레버로 극복하는 영역".
- **2509.09677 대비 방향**: 합성 task는 per-step 실패=계산오류라 scale/thinking이 삼 → "scale이 horizon 산다"처럼 보임. banking(실 tool-use)은 per-step 실패=구조라 scale이 느리고·불충분 → **scaffold가 더 싸게 사는 게 옳음**. **E-HORIZON(synth)=통제-인과 짝·banking=실-도메인 관측**. 덱/논문/특허 horizon 재프레임의 실측 앵커.

## 8. ★E-HORIZON 결과 [M] — verify가 per-step을 scale보다 급격·싸게 산다 (2026-07-12·무료·GPU1)
직접-증분 running-sum(dict-free·단일-스텝 easy·실패=순수 누적)·H=30·runs=12·Qwen2.5 사다리·결정론 gold. `eref_horizon.py`·`sim_results/ehoriz_qwen25-*.jsonl`.

| 모델 | base(scale) | verify | detect | verify−base | self-cond acc_post(base/verify) |
|---|---|---|---|---|---|
| 0.5B | 0.006 | 0.019 | 0.000 | +0.014 | 0.006/0.020 |
| 1.5B | 0.069 | 0.361 | 0.072 | +0.292 | 0.000/0.329 |
| 3B | 0.039 | 0.167 | 0.039 | +0.128 | 0.003/0.139 |
| 7B | 0.258 | 0.825 | 0.258 | +0.567 | 0.010/0.795 |
| **14B** | **0.322** | **0.911** | 0.344 | **+0.589** | **0.044/0.910** |

- **scale=완만**: 0.5B→14B(28× 파라미터) per-step 0.006→**0.322**(상용 미달·retail operand 포화와 대조).
- **verify=급격**: 14B 0.322→**0.911**(+0.589·pass-cost 0·결정론). verify_gain이 scale과 함께 커짐(단일-스텝 되는 모델서 verify가 더 크게 산다).
- **★기전=self-conditioning 절단(E-HORIZON-2)**: base acc_post(첫 오류 후) 0.04 = 고착(한번 틀리면 문맥 오염→cascade) vs verify 0.91 = 교정으로 루프 절단(14B 20× 회복). **detect≈base**(0.32) = *탐지*로는 안 되고 *교정*(오류 문맥진입 前 제거)이 핵심.
- ⇒ **"scale이 느리게 개선하는 per-step을 verify(싼 결정론)가 더 명확히 개선"**(사용자 지시 실증). horizon=per-step^H이므로 verify가 per-step→0.91로 올리면 horizon도 산다·scale-불변·pass-cost 0.
- caveats([[08]]): n=12(잡음권·1.5B>3B는 전이대역 noise)·synth(2509.09677 동형)·verify는 첫 오류는 못 막고 *cascade*를 막음(=self-conditioning 절단 정확). thinking arm(E-THINK)·in-vivo tool-use는 별도.

## 9. ★horizon = 동적오염의 시간축 (통합 이론·사용자 통찰 형식화)
**사용자 통찰(2026-07-12)**: "horizon 문제는 동적오염과 직접 연관 — 길게 이어지면 혼란스러운 지시가 누적돼 일관성 유지가 힘들다." → **형식화·확증**:
- 과제 성공 = `∏ₜ pₜ`(스텝 정확도 곱). 독립이면 `p^H`(2509.09677). **그러나 pₜ는 시간에 따라 하락**: `pₜ = p₀ − f(Cₜ)`, Cₜ = 스텝 t까지 문맥에 누적된 **동적오염**.
- **Cₜ의 원천 = 오염 축들(누적)**: (a) self-conditioning(모델 자기오류·E-HORIZON acc_post 0.04=cascade) + (d) 멀티턴 상충 지시(사용자 지시 축적→일관성 부하) + (c) distractor 누적(near-miss anchor 밀도↑·C69).
- ⇒ **horizon 붕괴 = 동적오염의 스텝-적분**. 길수록 오염 누적→pₜ 하락→성공 초-지수 붕괴(p^H보다 급함). **horizon(시간축)과 동적오염(기전)은 같은 현상의 두 얼굴** — "길면 어렵다"의 *이유*가 "동적오염이 누적된다".
- **증거**: E-HORIZON(축 a: verify 절단→pₜ 유지·[M]) · C69(축 c: near-miss 바인딩 파손·[M]) · Laban 2505.06120(축 d: 멀티턴 reliability +112% 보편·[미검]).

### 9.1 ★극복법 (사용자 "이걸 극복할 방법이 있나?" → YES·오염-방화벽)
오염은 *문맥에 누적*된다 → pₜ≈p₀ 유지하려면 **오염이 문맥에 쌓이기 전에 각 스텝서 제거**. 축별 결정론 레버 = **오염 방화벽**:
| 오염 축 | 극복 레버(결정론·pass-cost 0) | 증거 |
|---|---|---|
| (a) self-conditioning | **per-step verify-correct**(오류 문맥진입 前 교정) | E-HORIZON [M](acc_post 0.04→0.91) |
| (d) 상충 지시/멀티턴 | **canonical-state controller**(정본 상태를 LLM 문맥 밖 결정론 유지·LLM은 턴 delta만 번역·오염된 히스토리 미참조) | E-PLAN/controller [D] |
| (c) distractor 정박 | **provenance/gather 게이트**(검증된 출처 레코드서만 바인딩) | C45/C69 [M] |
| (정책) | **compliance 게이트** | C1/C53 [M/S] |
| — thinking(부분·비쌈) | 축 a만(자기오류 commit 안 함)·축 c/d 무력 | 2509.09677·C56 |
⇒ **오염-방화벽이 pₜ를 horizon 길이·scale과 무관하게 p₀ 근처로 유지** = horizon을 scale 아니라 scaffold로 산다.

## 10. ★큰 서사 + tool-use 스코핑 (사용자 지시: 논문/특허 주장 가능성)
### 10.1 서사 (동적오염×horizon×thinking×scale 통합)
> **tool-use 에이전트는 추론 부족이 아니라 *누적 간섭 하 실행*으로 실패한다.** 과제가 길어질수록(horizon) 문맥에 동적오염이 쌓인다 — 모델 자기오류(self-conditioning)·사용자 상충지시(멀티턴)·distractor 레코드. 각 오염이 스텝 정확도 pₜ를 떨어뜨리고, 성공=pₜ^H인데 pₜ 자체가 오염 누적으로 하락하니 horizon이 초-지수 붕괴한다. **scale**은 기저 pₜ를 올리나 *느리고 상용 미달*(banking 0/18)·비쌈. **thinking**은 자기-오염만 끊되(축 a) 외부 오염(축 c/d)엔 무력·비쌈(inference-scaling). **비용효율 답 = 결정론 오염-방화벽**(per-step verify + canonical-state controller + provenance/compliance 게이트)으로 pₜ를 horizon·scale 무관 p₀ 근처 유지·pass-cost 0. **scale은 추상이 지배하는 다른 분야선 여전히 답이나, 간섭이 지배하는 tool-use선 scaffold-조합이 답이다.**

### 10.2 ★스코핑 = tool-use 도메인 한정 (필수·정직·모트 강화)
- **주장 범위 = tool-use 25 벤치(τ²/SOPBench/TaskBench/Synth 등)로 명시 한정.** "scale 무용론" *아님* — scale은 추론·지식·일반능력서 답. 우리 주장 = **간섭-지배 tool-use 영역서 per-step 잔여(compliance·reach·coverage·binding-under-contamination)가 *구조적*이라 scaffold가 scale보다 싸게 닫는다**.
- **왜 정직한 좁힘이 모트를 강화하나**: (1) 과대주장("소형=대형") 회피→"버금가는"([[46]]) (2) scale이 이기는 곳 인정→신뢰 (3) 모트가 도메인-스코프됨=tool-use=저추상·고간섭(추상능력 1.5B emergent·C66 / 격차=동적오염·C69).
### 10.3 논문·특허 주장 가능성 = **YES (조건부)**
- **가능(지금 증거로)**: compliance scale-불변(C1 [S]) · banking 0/18 frontier 상용미달(C71 [M]) · E-HORIZON verify≫scale per-step(§8 [M]) · C69 동적오염 바인딩파손([M]) · thinking≠⋈(C56 [M]). ⇒ "tool-use서 scale은 비용효율 답 아님·scaffold-조합이 답"을 **측정 근거로 주장 가능**.
- **표현 규율([[40]])**: "tool-use 도메인서"를 항상 명시·"scale은 X 분야선 답이나 tool-use 간섭영역선 아니다" 대조 프레임. horizon은 "scale 영토"가 아니라 "오염-방화벽으로 극복하는 영역"으로 기술.
- **반영 대상**: Paper1 §1 동기·§6 lever allocation(오염-방화벽 통합) · 특허 A§3.5 헤드라인(오염축별 레버)·B§5.A(horizon=동적오염 스코프) · 덱 §서사 슬라이드. **[미검] 축 d(canonical-state)·in-vivo는 표기.**

## 7.5 caveats([[08]]·[[40]])
- pass=p^H는 근사(단계 독립 가정·self-conditioning 있으면 실제 더 급함). banking universal-fail은 H=12.5라 p 함의가 H=8보다 높게 나옴(장기일수록 완주가 더 정보적) — 도메인 median H=8 기준 채택.
- banking_knowledge는 τ³ KB-검색 축(우리 operand 프레임과 부분 다름·§3.2c caveat)·EXTRA_read ~100%=탐색 비인과. all-or-nothing DB 채점 가혹성 실재(단 완주-후-불일치 45%=진짜 인자오류·§3.2f-4 per-case 3건 확정).
- **미측정**: banking에 결정론 scaffold(controller+calc+prov+coverage) 실붙임 후 per-step→1·pass 상향 = **E-XFER-bank gate arm**(유료·승인대기). 본 §은 *frontier가 못함*의 기록이지 *우리가 함*의 증명 아님(후자=E-XFER-bank).
