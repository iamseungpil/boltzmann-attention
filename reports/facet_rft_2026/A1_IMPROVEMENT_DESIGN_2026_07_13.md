# A1 기능 개선 설계서 (A1-v2) — 2026-07-13

> ★A1(generalized_stack·db 0.623) → A1-v2. 근거=ablation+per-step forensic+`REMEDIATION_DESIGN_2026_07_13`.
> 목표: 확정된 회귀 원인만 *구현된·저위험* 레버로 교정 → nt=1 재실험으로 방향 확인(vs 0.623).
> 불변: [[05]] 도메인일반·A2만 · [[10]] 선택기=결정론 · Δspurious≤0 · [[09]] 무료검증 先.

## 1. A1 현재 = gate+prov(rescue)+calc+eplan+cap · 회귀 원인 확정판
| 원인 | tasks | ablation 판정 | A1-v2 조치 |
|---|---|---|---|
| eplan qty-conflation→transfer | 58 (robust) | eplan harm-I | ✅ examined-safe |
| eplan 궁지→transfer/오주문 | 83·69 (robust) | eplan harm-II | △ filter(부분)·GET-forcing(미구현) |
| eplan 과-행동 | 32 (부분) | eplan harm-III | ✅ reads-only |
| 주문선택 ⋈ | 55·59·101·92·112 | present↔eplan 대체·filter 대상 | ✅ filter(dotted-fixed) |
| present 제거 손실 | 7·92 | eplan/filter가 대체복구 | (present 미추가·anti-drift) |
| variance | 6·46·55 | rescue arm 통과·cap 0발화 = **비-델타** | 조치 없음(노이즈) |
| 변형/의도/주소 | 52·77·110·94·23·96·108·38·43 | learn/scale or 미구현 레버 | 이번 범위 밖(§4) |

**핵심 방침**: ablation이 **eplan=순이득(+10pp)** 실증 → **eplan 유지·harm만 정밀교정**(핸드오프 "eplan 제거"는 sign-오류). prov-rescue·cap = 회귀원인 아님(무죄)→불변.

## 2. A1-v2 스펙 (구현된·저위험 레버만)
```
A1-v2 = A1 + { EXAMINED_SAFE, READS_ONLY, DISAMB-filter(dotted-fixed) }
```
| # | 변경 | 토글 | 근거 | 구현 | Δspurious 리스크 |
|---|---|---|---|---|---|
| 1 | **eplan examined-safe** | T2_EPLAN_EXAMINED_SAFE=1 | 대상 order examined면 discovery-deny 생략(harm-I t58) | ✅+unit | 낮음(검토된 write만 통과·미검토 deny 불변) |
| 2 | **eplan reads-only** | T2_EPLAN_READS_ONLY=1 | L2 피드백=관련성 확인용·요청 밖 write 금지(harm-III t32) | ✅ | 낮음(문구만·강제 불변) |
| 3 | **filter-substitute** | T2_DISAMB=1 MODE=enumerate DISAMB_ORDER=1 | ≥2 후보 order를 formalize→결정론 필터(점경로 수정·t102 회복 확증) | ✅+probe | 중(over-ask·trivial 무회귀 확인 필요) |

**명시적 제외 (리뷰서 기각)**:
- prov-addr-full: **철회** — t96 주소는 user-confirm laundering이라 prov 구조적 불가(REMEDIATION §1b).
- present 재추가: anti-drift(spoon-feed·C59)·eplan/filter가 discovery 대체.
- prov→full: rescue 무죄·full과 무차(rescue arm 실측)·generalized 기본 유지.
- GET-forcing(L2)·origin-prov(L3)·fexec-variants(L4): **미구현**→다음 이터레이션(§4).

## 3. ★리뷰 (self-critique·[[08]]/[[05]])
1. **Δspurious 필수 확인**: examined-safe/reads-only가 **eplan-help 8(21·23·33·42·52·94·108·110)** 을 회귀시키면 안 됨. reads-only는 문구만이라 coverage 강제 불변(walk off)·examined-safe는 미검토 deny 유지 → 이론상 help 보존. **단 nt=1 재실험서 help-8 per-case 확인 필수**(과-완화 금지·모트 §Δspurious).
2. **번들 혼재 주의**: 3 레버 동시 → nt=1 재run은 "전체 개선 방향"만 측정(per-레버 귀속 아님). 각 레버는 **독립 근거 보유**(examined-safe=t58 per-step·reads-only=t32 per-step·filter=t102 probe) → 맹목 번들 아님. per-레버 순효과는 기존 ablation/probe가 커버.
3. **[[05]] 준수**: 세 토글 다 도메인 리터럴 0 — examined=ledger 사실·reads-only=문구·filter=A2 items_key/getter 스키마. scaffold 확장 아님(examined-safe/reads-only=deny *완화*=scaffold 축소·filter=기존 fexec 재사용).
4. **nt=1 노이즈 한계**: 회귀의 ~절반이 flaky(§variance)라 nt=1 단일시행 db는 ±노이즈. **집계 db(vs 0.623) + 표적 per-case(58 transfer해소·32 과행동해소·55/59/101 order해소)** 병행 판독. 방향만 확인·확정은 nt≥2.
5. **위험 시나리오**: filter가 trivial서 over-ask 유발(안전하나 gold 무-질문 기대 시 감점) → trivial 셋 per-case 확인. examined-safe가 진짜 coverage 필요 태스크(t42형)서 2번째 write 못 막아 miss → help-8의 42 확인.

## 4. 다음 이터레이션 (미구현·이번 범위 밖·후속 설계)
- **GET-forcing(L2)**: 애매 order write 전 후보 전수 GET 강제 → harm-II(83·69) 완성. 로컬 ledger unit 先.
- **origin-provenance(L3)**: first-mention 추적 → 주소 세탁(96). turn-order unit 확증됨→구현.
- **fexec-variants(L4)**: fexec를 variant record에 → 극값변형(52·77·110). _field_values 점경로 재검.
- **precondition gate(L7)**: exchange=delivered(21)·split불가(38).

## 5. nt=1 재실험 계획
- **드라이버**: `generalized_stack_v2.sh`(=A1 + 3토글). ALL 114 · nt=1 · port 8141(GPU1 free).
- **판독**: (a) 집계 db vs A1 0.623 (b) 표적 per-case(58·32·55·59·101 해소?) (c) help-8 무회귀(Δspurious≤0).
- **영속화**: gz→sim_results→commit([[30]]).
- 후속: 방향 양성이면 nt≥2 확정 + per-레버 ablation.
