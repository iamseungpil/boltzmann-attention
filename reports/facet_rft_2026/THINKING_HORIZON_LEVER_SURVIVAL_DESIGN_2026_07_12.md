# tool-use 레버 생사 지도 — thinking의 경계·horizon 재프레임·비용효율 조합 (설계 2026-07-12)

> 상위 = `RESEARCH_MASTER.md`(F1-F6 프레임·제1원리 상쇄). 선행 = `SCALE_DYNAMIC_CONTAMINATION_PRIORWORK §8`(2509.09677 정독) · `OVERNIGHT_RESULTS §결과4`(C69 동적바인딩) · C56(동-scale thinking ⋈ 못삼) · C53/C62(prov·게이트 e2e).
> 규율: [[05]] A2만 변경·엔진 도메인-일반 · [[08]] 집계→결론 前 per-case · [[09]] 무료 우선(로컬 사다리·결정론gold·user-sim 0). **본 문서 = 설계 + 무료 실험 큐**(유료 full-run 없음).

## 0. 두 재프레임 (사용자 지시 2026-07-12)
1. **우리는 선행의 역(逆)이다.** 선행(2509.09677·METR·scaling law)은 "**scale이 능력 X를 어떻게 사는가**"를 묻는다. 우리는 반대다 — "**tool-use의 어느 세부능력이 scale보다 *싼 것*으로 사지는가, 그리고 그 조합을 어떻게 비용-최적화하는가**". 핵심 주장 = **특정 tool-use 영역에서 scale을 {retrieval·verify-scaffold·결정론 게이트·도메인-일반 learn}으로 대체하는 것이 비용-효율적이다.**
2. **horizon은 scale의 영토가 아니라 "scale 없이도 극복 가능한 영역"이다.** (§2 상술)

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
