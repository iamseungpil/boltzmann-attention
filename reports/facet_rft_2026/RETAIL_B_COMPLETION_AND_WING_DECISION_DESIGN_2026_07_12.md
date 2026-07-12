# retail-B 완성 + 날개-판단 실험 설계 (scaffold-C → {banking 전이 | synth 학습} 분기) — 2026-07-12

> 등대 §4 큐(E-COMP·T5-C·E-PLAN·E-XFER-bank·E6′) + 기존 설계서 **시퀀싱**. 재발명 금지([[48]]).
> **이 문서 = 오케스트레이션 설계**: 개별 레버 명세는 각 정본 doc(참조)에 있고, 여기서는 (I) 무료-先 순서로 retail-B를 한계까지 밀고 (II) **C-잔여로 두 날개(전이/학습)를 가르는 판단 실험을 사전등록**한다.
> 상태: **[D] v1 — 리뷰 대기**(독립 적대리뷰 spawn). 유료(C·banking·learn)는 승인 후.
> 불변: [[05]] A2만·엔진 도메인일반 · [[08]] per-case·집계금지 · [[09]] 무료先·유료 최소 · [[13]] scaffold 소진→그다음 learn · 제1원리 Δspurious≤0.

---

## 0. 한 줄 · 두 질문

retail을 **결정론 scaffold로 소진**(닫히는 건 다 닫고 ⋈는 경계로 지도)한 뒤 남는 잔여의 *성격*이
두 날개를 가른다:
- 잔여 = **경계(⋈·미도달)** → 더 닫을 게 없음 → **banking 전이**로 "고정 scaffold의 도메인-일반성" 증명(P1/특허 core).
- 잔여 = **도달가능(reachable)하나 scaffold-불가** → **synth 학습**(P4 둘째 날개)이 정당 → E6′ 데이터 v3.

**Q1 (Part I)**: retail-B를 한계까지 미는 무료-先 실험 순서·GO 조건.
**Q2 (Part II)**: C-잔여를 어떻게 분해해 위 분기를 *사전등록된 기준*으로 판정하나 (= 판단 실험).

---

# PART I — retail-B 완성 (scaffold 소진 → C)

## 1. 잔여 레버 인벤토리 (C64 클래스 → 설계 → 상태)

> **★상태 정정(2026-07-12·코드 직접 확인·[[08]] 문서는 시점 스냅샷)**: 참조 설계서(E-PLAN §8 "다음 액션")보다 **코드가 앞서 있음** — E-PLAN + census 4종이 **이미 구축·배선**됨. 커밋 근거: `0f753a9`(E-PLAN unified() 배선: ledger+L1/L2 deny+CP5 walk) · `00fa5d2`(**deny 무한루프 발견·cap = 레버 부작용 1호**·t103/t27) · `d33db23`(census 4종 구현+V0) · `f9e7591`(L2 conflation 수리·과발화 −45%·**92 checks PASS**). ⇒ retail-B의 현 단계 = **"구축부터"가 아니라 "부작용 디버그 → 격리검증 → nt=1 표적"**.

| C64 클래스(sims) | 레버 | 정본 doc | 현 상태(코드확인) | 비용 |
|---|---|---|---|---|
| **A coverage/discovery (≈8)** | E-PLAN L1/L2 + CP5 재-plan walk | `E_PLAN_LIVE_WIRING_DESIGN` v1.3 | **구축·unified() 배선·92단위 PASS**·deny-loop cap됨 → **격리검증 §5(d)④⑤ + nt=1 미완** | V0·격리 무료 |
| C compound-criterion (≈6) | CALC-EXT (argmax/argmin/diff) | `CENSUS_LEVERS §2a` | **구현+V0 완료**(t20 3/4·28단위 PASS·most_recent→ASK) | 무료 |
| F 값충실도 (t17·t39) | **PROV-RESCUE-PERARG**(GROUND-VERBATIM 폐기) | `CENSUS_LEVERS §1` | **구현됨**(`d33db23`)·엔진수정 V0 재확인 필요 | 무료 |
| D GET-chain lookup (≈5) | DISAMB-ADDR(P-B 확장) + 미조회분=E-PLAN L2 | `CENSUS_LEVERS §4` | **구현됨**(`d33db23`)·V0 분해(gold-실재/미조회) 재확인 | 무료 |
| B over-action decidable (≈5) | **t27형**=EXCLUSIVITY(관측-ledger) · **t57형=G8 NOTICE-PERGATE**(CENSUS §3b는 G8로 superseded — 이중구현 금지) | `CENSUS_LEVERS §3a`·`NEXT_LEVER_GEN §1` | EXCLUSIVITY 구현됨·ledger 분리 확인 / **G8=동결 레버**(스택 편입 S4/S5 후·오프라인 준비물[census·단위]은 양립) | 무료 |
| **C-잔여 문맥-의존 제약** (t20 Running Shoes·t79) | **FORMALIZE-EXEC**(CALC-EXT 정적 spec의 구조적 MISS 칸 — §2a V0가 실증) | `NEXT_LEVER_GEN §2` | [D] **동결 레버**·단 V0 격리 측정(형식화-정확도 EM)은 무료·동결 양립(§0b.2) | V0 무료 |
| E DISAMB 잔여 (63) | T5-C COMP+D-v2 | `T5C_SILENT_REPAIR` | **단계 A 완료**(t0/t61 4/4)·단계 B 대기 | V0 무료·nt1 소액 |
| G-relay (t3형) | CP5 communicate-의무 확장 | `CENSUS_LEVERS §2b` | [D] 프로브先·**learn 대상 아님** | 무료 프로브 |
| B semantic 잔여·H·I | (경계·미개척) | `CENSUS_LEVERS §5`·C63 | P3 경계 계상 | — |

**핵심(정정)**: A(E-PLAN)가 최대 headroom(MISSED+ZERO_NEV 47 sims=10.3pp·`SCAFFOLD_ENDGAME §L4`)·지배 레버이며 **이미 구축·배선됨**. 현 병목 = **레버 부작용 소진**(deny-loop cap이 1호·제1원리 Δspurious 계측)과 **격리검증 §5(d)⑤(CP5 재-plan 생사)** + **nt=1 표적 사이클** — "구현"이 아니라 "검증·튜닝".

## 2. 무료-先 실행 순서 (병렬 가능 지점 표시)

각 레버 = **V0 오프라인 census(무료·COMP 167 fail에 적용) → 단위 → GO판정 → 스택 편입 → nt=1 사이클**. 순서:

```
[S0] E-PLAN 부작용 소진 + 격리검증(무료)  ┐  ★구축·배선·92단위 완료 — 남은 것:
     - 부작용 소진: deny-loop cap(1호·`00fa5d2`) 외 잔여 부작용 census(Δspurious/Δtme·제1원리)
     - 격리검증 §5(d)④⑤ (GPU 한가할 때·⑤=CP5 재-plan 주축 생사 = 최우선)
[S0'] 엔진-증분 레버 V0 재확인(무료·구현됨 `d33db23`)  ┘  (E-PLAN과 병렬)
     - PROV-RESCUE-PERARG: 단위(#정규화·다중fab순회) → v25e t17 4trial 오프라인(fab이 address1에 닿나)
     - DISAMB-ADDR: V0 분해 = 주소-오복사 전수 {gold-실재 판정 + 실재분 서브콜 재현}(D 크기 확정)
     - EXCLUSIVITY: 관측-전용 ledger 분리 확인(§3a·arm 충돌 해소)
     - (CALC-EXT는 V0 완료 — 스택 편입만)
         │  각 레버 GO = V0 fix≥임계 ∧ break=0 ∧ 단위 PASS ∧ **부작용 census 통과**
         ▼
[S1] 단계 B nt=1 사이클(소액·승인)  — T5-C §0b 프로토콜(nt≥2 한방 금지·nt1 누적)
     - arm 구성: COMP(기지·C62 GO) + {GO된 census 레버들} = 개입-합성 스택
     - E-PLAN 개입레버(discovery·walk)는 **별도 arm**(합산 금지·루프 아키텍처 변경)
     - 표적 26 task(기지 13 + SYSTEMIC 13) × nt=1 → per-case([[08]]) → 사이클 반복
     - GO(공통): per-case 복구 ∧ Δspurious≤0 ∧ over-block=0 ∧ Δtme≤0 ∧ 위반0 ∧ 짝flip 순증
         ▼
[S2] C 검증(유료·승인·456)  — 스택 확정 후 nt=1 사이클 누적(nt4 한방 폐기·T5-C 프로토콜)
     - 2 arm: (a) COMP+census 스택  (b) COMP+E-PLAN(별도)  → 필요시 통합 스택
     - 공식 pass^1..4 · compliant · Δ낙폭 · robust core
     - **스모크 필수**([[30]]): num_tasks 10 nt1로 전 레버 라이브 발화(stderr 마커) 검증
```

**도달 목표**(`SCAFFOLD_ENDGAME §3`): R0 후 0.63±0.03 → **0.66~0.70**. frontier(0.741) 잔여 = 경계 = Part II 입력.

## 3. "한계" 판정 (retail-B가 소진됐다의 정의)

retail-B 소진 = **다음이 동시 성립**:
1. C64 결정가능 클래스(A/C/D-실재/F/B-decidable) 각각 GO 레버 스택 편입·C에서 발화 확인.
2. 신규 census 레버의 V0 fix가 **수렴**(추가 레버가 새 non-empty fix 클래스를 못 냄 = E-P류 "새 C-류 0 수렴" 동형).
3. 잔여가 **{⋈ 경계 · 대화-semantic(C50) · NL-relay(C64-G) · FLAKY 분산}**으로만 구성(per-case census 확인).
4. **(리뷰 추가) 동결-레버 계상**: G8·FORMALIZE-EXEC은 동결 중이라 스택 밖일 수 있음 — 그 표적(t57·t20잔여·t79)은 **"경계"가 아니라 "설계-완료·동결-대기"로 별도 계상**(소진 판정·Part II 입력에서 제외). 이걸 안 하면 설계된 레버가 있는 잔여를 경계로 오분류 → 분기 판정 오염.

⇒ 이 시점의 잔여 = Part II 판단 실험의 입력. **주의**: "0.70 도달"은 목표지 소진 기준 아님 — 소진 기준은 **레버-수렴 + 잔여-성격**(pass 수치 아님·[[08]]).

---

# PART II — 날개-판단 실험 (C-잔여 → {전이 | 학습})

## 4. 판단 원리 (§1.5 라우팅의 잔여-적용)

C-잔여의 각 실패를 **§1.5 결정절차 + reachability 게이트**로 라우팅해, 잔여의 *성격 분포*가 분기를 결정한다.
learn(synth)이 정당한 필요충분 = **"scaffold로 안 닫히는데 *도달가능*한(sampling이 정답에 닿는) 잔여 버킷이 유의미하게 존재"**.

```
C-잔여 실패 f 마다:
  R1. decidable ∧ 미scaffold?  → scaffold 갭(retail-B 미완·S0로 복귀·판단대상 아님)
  R2. 부하(load)?  E-ISO C-프로토콜: p_iso(A 궤적재현) vs B(격리원문) vs C(격리형식화)
       ├ B≫A            → 궤적-간섭 부하 → controller(scaffold) → 학습 아님
       ├ B≈A ∧ C≫B      → 형식화-부하   → 직렬화(FORMALIZE-EXEC·scaffold) → 학습 아님
       └ C 낮음          → R3
  R3. reachability 게이트: best-of-N-with-CoT(temp>0·N≥8·four-bench+τ² 격리케이스)
       ├ pass@N > θ_reach ∧ 비체계(오답 분산)  → ★도달가능·scaffold-불가 → **learn 후보**
       └ pass@N ≈ 0 ∧ 체계(전-trial 동일오답·C56)  → **경계(⋈·map)** → 학습도 못 닫음
```

**사전등록 판정 기준 (motivated-filling 방지·F3 규율)**:
- θ_reach·유의미 임계 = **learn-후보 버킷이 robust하게 ≥ N_min sims**(제안 N_min = 4·robust=전-trial 도달) ∧ **그 버킷이 형식화-부하(R2)로 이미 설명 안 됨**.
- **분기 A (learn GO)**: R3 도달가능 버킷 ≥ N_min ∧ E-ISO C 낮음(부하 아님) → **synth 학습 정당** → E6′ 데이터 v3 설계(그 버킷 표적·C38 3결함 수정: (i)결손 재현 데이터 D7 근접오답 (ii)무조건조회 퇴화 방지 음성사례 (iii)on-policy rejected). **★[[11]] 가드(리뷰 추가)**: learn 표적 = 버킷의 **스킬-클래스 추상화**(도메인-일반 P-primitive로 승격 → synth/cfbsynth 합성·[[12]] 다양성) — retail 엔티티/템플릿/사례를 학습 데이터에 넣는 것은 도메인-타깃 학습 = **금지**. 분기 A조차 four-bench→τ² swap 구도 불변.
- **분기 B (learn NO-GO·전이 우선)**: 잔여가 {경계 + 이미-scaffold된 부하}로만 → **learning 못 닫음** → **banking 전이(E-XFER-bank)** 우선 = "고정 scaffold 도메인-일반성" 증명 + ⋈ 경계 = P3 정직 주장.

## 5. 사전 prior (판정 前 명시·대칭크레딧 [[03]]#9)

측정 前 prior는 **분기 B(전이) 쪽**이 강함 — 단 판단 실험이 *반증 가능*하게 설계돼야 함:
- C38: learn 미확립(cfbsynth 결손 재현 실패·SFT 퇴화·DPO off-policy) = 데이터 실패지 원리 실패 아님(재현 가능).
- C56: thinking이 ⋈ 못 닫음(동-scale). C59: 열거가 격리서 ⋈ 엶(+31pp) **but** C60/C61: e2e-0(전달 기전이 유해).
- ⇒ ⋈ 잔여는 **격리선 도달가능처럼 보이나 e2e 미도달** = reachability 게이트가 *격리 pass@N*과 *e2e*를 **둘 다** 봐야 함(격리 도달가능 ≠ learn e2e 이득 — C60 교훈). **판정에 e2e 축 필수**(격리 pass@N > θ 단독으로 learn GO 금지).
- **반증 조건**(prior를 뒤집는 관측): C-잔여에 "전-trial 도달가능(best-of-N pass) ∧ 격리 아닌 실-궤적서도 CoT로 회복 ∧ 형식화-부하 아님" 버킷이 ≥ N_min → 분기 A. 이건 실측으로만 갈림.

## 6. 판단 실험 = banking이 *일부* 겸함 (전이 자체가 판정의 절반)

banking 전이는 분기 결과이자 **판정 도구**이기도 하다(잔여 일반화 검정):
- E-XFER-bank floor(확보: nt2 0.050·`bankxfer_floor`) + 최종 scaffold arm → banking **잔여**를 C64-census 동형으로 분해.
- banking 잔여가 **⋈-경계 동형**(retail과 같은 성격) → **도메인-일반 경계**(강한 P3) → 어느 도메인서도 learn만이 후보 → 그때 R3 reachability로 최종 판정.
- banking 잔여가 **reach/coverage/horizon**(C52·scaffold가 닫는 것) → **전이 성공**(P1) → retail-특이 잔여(⋈)는 도메인-일반 아닐 수 있음 → learn 표적 축소.
- ∴ 순서: **retail-C 잔여 R3 게이트(무료 격리) → banking 전이(유료) 잔여 분해 → 두 잔여의 교집합이 진짜 learn 표적**.

## 7. 판단 실험 실행 순서·비용

| 단계 | 내용 | 비용 | 산출 |
|---|---|---|---|
| J0 | C-잔여 per-case census(C64 도구 재사용) → R1/R2/R3 라우팅 라벨 | 무료 | 잔여 성격 분포 |
| J1 | E-ISO C-프로토콜(A/B/C)을 C-잔여 결정점에 재실행(`ecomp_iso_probe.py`·C61 확장) | 무료(32B 로컬) | 부하 몫 제거 |
| J2 | reachability 게이트(best-of-N-CoT·격리 + **실-궤적 재현**) | 무료(로컬) | 도달가능 버킷 |
| J2′ | **(리뷰 추가·정직) true-e2e CoT 확인** — §5/§8의 "e2e 축 필수"는 user-sim 필요 = **유료**. J2의 실-궤적 재현(기록 궤적에 CoT 주입·무료)은 **e2e 근사일 뿐**(C60 교훈 자체가 격리→e2e 붕괴). learn GO 최종 확정 前 도달가능 버킷 대표 sims만 최소-scope e2e(승인) | 유료(최소·승인) | learn GO 확정 증거 |
| J3 | **분기 판정**(사전등록 §4 기준) → learn GO / NO-GO | — | 결정 |
| J4a | (분기 A) E6′ 데이터 v3 설계·타당성 게이트(base가 결손 재현) | 무료 설계·학습 유료 | learn 표적 |
| J4b | (분기 B) E-XFER-bank 최종 scaffold arm(banking 전이) | 유료(승인·재시퀀싱됨) | 전이 증명 + banking 잔여 |
| J5 | retail∩banking 잔여 교집합 = 진짜 도메인-일반 learn 표적(있으면) | — | 최종 날개 결정 |

## 8. GO/판정 요약 (사전등록)

- **retail-B 소진**(Part I §3): 레버-수렴 ∧ 잔여={경계·semantic·relay·flaky}.
- **learn GO**(분기 A): R3 도달가능 버킷 ≥ 4 robust sims ∧ 형식화-부하 아님 ∧ **e2e CoT 회복 실재**(격리 단독 금지).
- **전이 우선**(분기 B): 위 미충족 → banking 전이로 도메인-일반성 증명 + 경계 정직 주장.
- 어느 분기든 **⋈ 경계는 map**(강제 학습·강제 write 금지·§1.5).

---

## 9. [[05]] 3질문 (설계 전체)
1. 고정=TBox+scaffold 엔진 / 변경=ABox? → ✅ 전 레버 A2만(각 doc 감사 완료)·판단 실험은 측정만(개입 0).
2. 도메인-특화 scaffold 금지? → ✅ E-PLAN controller·census op 전부 도메인일반·retail 매핑만 A2.
3. 도메인-타깃 학습? → ✅ 학습은 분기 A에서만·four-bench(도메인일반)→τ² swap([[11]])·retail-타깃 학습 금지.

## 10. 리스크·미해결
- **R-a**: retail-B "소진" 판정이 주관적 → §3 3조건(수렴+성격)으로 객관화·per-case.
- **R-b**: reachability 격리 pass@N이 e2e 이득 과대추정(C60) → §5 e2e 축 필수·격리 단독 금지.
- **R-c**: E-PLAN 미구현이 전체 병목 → S0 최우선·무료·GPU 무관(스켈레톤/단위/오프라인).
- **R-d**: 비용 — C·banking·learn 전부 유료 → 무료 J0~J3로 분기를 *먼저* 좁힌 뒤 유료 최소 scope([[09]]).
- **소유권**: E-PLAN(coverage-walk)·T5-C(silent repair)·E-SPEC(오케스트레이터 재설계) 좌석 공유 — 중복 구현 금지·본 doc은 시퀀싱만.
