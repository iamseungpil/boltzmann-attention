# E-COMP: retail pass 상향 — 검증기 합성 arm 설계 (2026-07-10 · [D] · 리뷰 대기)

> **목표**: 32B retail bench-pass를 규칙0-준수 상태로 최대화. 현 위치 floor 0.557 / floor+prov 0.5768(C53·[M]) /
> 게이트스택 0.640(★present 포함=C34 폐기 역사수치·규칙0-준수로 재구성 필요) / frontier gpt-4.1 0.741.
> **핵심 관찰: 이미 GO된 레버들(게이트·prov·DISAMB·calc)이 서로 합성된 적이 없다** — 코드가 이중패치 가드로
> `T2_GATE_REGEN`↔`T2_PROV_*`를 상호배타 처리(`t2_run_gated.py` CONFLICT guard).
> 등급 [D]. 근거 원장: C53(prov e2e +3.0pp) · C51(frontier 잔여=F2 변형) · C59(열거가 ⋈를 엶·격리 +31pp) ·
> C56②③(동-scale thinking 무효→직렬화·체계핵=DISAMBIGUATE+calc) · C4d(게이트 자기-역효과→Δspurious 계측) ·
> C10(부작용은 scope에서·측정된 상쇄로만 합성).

## 0. 제1원리 적합성 (등대 §1.3)
합성은 무한후퇴가 아니라 **측정된 상쇄**여야 한다. 이 설계의 모든 레버는 **생성-레벨 검증기**(deny/재확인 후 재생성)로
같은 scope에 살고, 각각 반대편 계측을 단다:
| 레버 | 산다(+) | 팔 위험(−) | 반대편 계측 |
|---|---|---|---|
| 게이트(auth·confirm·precond·constraints) | F1 위반 0 | over-block·regen 예산→tme | false-block census·Δtme |
| prov 검증기 | 날조 0(C45) | 재발화 예산 | Δtme(C53서 1/456=미발현) |
| DISAMB(|C|≥2 재확인) | ⋈/변형 오선택(C59) | 이미 맞은 선택을 흔듦 | **switched-away-from-gold census** |
| calc(count/sum) | F2b 보고 정확도 | (읽기증강뿐) | Δspurious≤0 |

## 1. 레버 구성 (전부 기존 GO/구현물 — 신규 발명 없음)
- **A. 게이트**: `apply_gate_regen`(replay-safe·생성-레벨 deny+regen+R8 strip). kinds=auth,confirm,ownership,notice,preconditions,constraints. **present(T2_PRESENT_READS)=제외(C34 규칙0 위반·영구 폐기)**. nested(T2_PRESENT_NESTED)+calc(T2_CALC)=포함(에이전트가 가져온 내용 위 동작=규칙0 준수·C34 명시).
- **B. prov 검증기**: `apply_provenance_regen`의 L2(검증기+재발화·K=4). badwords/GROUND/autofetch=OFF(C53 동일).
- **C. DISAMB**: `T2_DISAMB` 기존 구현(문맥-실재값인데 같은-형식 후보 2+개→1회 재확인·후보=에이전트 기조회 출력만=DB주입 0).
- **D. (선택·소형) calc 기준형 확장**: 체계핵 t71 "最近 주문"=argmax(날짜) 오적용(C56④). 현 calc은 per-record count/sum뿐. **cross-record 기준형(argmax_over_fetched)은 엔진 확장이 필요**(에이전트가 fetch한 record들에 한정 누적=규칙0 준수) — 질량 ~1.4%(체계핵)라 **Phase 2로 분리**(본 실험 블로킹 아님).

## 2. 구현 설계 — 이중패치 통합 (유일한 실질 엔지니어링)
문제: `apply_gate_regen`과 `apply_provenance_regen`이 둘 다 `LLMAgent._generate_next_message`를 패치 → 나중 것이 덮음.
**설계 = 단일 생성-레벨 검증 체인** (`apply_unified_regen(gate_cfg, prov_cfg, disamb_cfg)`):
```
_generate_next_message(message, state):
  commit(message); rebuild_gate_state(committed); ctx = ctx_from(committed)
  am = generate(base)
  for round in 1..K_UNIFIED(=4):
      fb = []
      fb += prov_violations(am, ctx)        # 날조 인자 → REGEN_FEEDBACK
      fb += gate_denials(am, gate)          # 정책 deny → POLICY GATE 피드백
      if not fb: break
      num_errors++  (라운드당 1회 — 이중과금 금지·기존 두 경로와 동일 예산압박)
      am = generate(base + [am] + fb-as-tool-errors)
  # 종단
  gate-denied mutating 호출 → R8 strip + _BLOCK_NOTE   (replay-safe 필수·기존 게이트 semantics)
  prov-fab 잔존 호출 → 통과(실행)                        (기존 prov semantics·env가 id 거부=C12)
  if clean and DISAMB: 1회 재확인 규칙 그대로(기존 코드 이식·재확인 후 prov 정화 2회 한도)
  return am
```
- **순서**: prov와 gate 피드백을 **같은 라운드에 병합**(핑퐁 방지·토큰 절약). DISAMB는 클린 확정 후 마지막(기존과 동일).
- **exec-side**: `_install_regen_exec`(observe+nested+calc 읽기증강) 그대로 재사용 — 생성-레벨과 직교.
- **게이트 상태**: 라운드 내 재생성된 am은 커밋 전이므로 gate state는 라운드 간 불변(기존 gen_gated와 동일).
- t2_run_gated: CONFLICT guard 제거 → `T2_GATE_REGEN=1 ∧ T2_PROV_REGEN=1` 시 unified 경로. 단독 플래그는 기존 경로 유지(회귀 0).
- [[05]] 3질문: 도메인-특화 순증 0(전부 기존 A2 소비·wiring만)·유동성 동결 0(선택은 모델)·도메인 행동 수행 0(deny/재확인만).

## 3. Arms & 측정 (retail·32B GPTQ-Int8·gpt-4.1 user-sim·temp0·nt=4·456 sims/arm·32k serve)
| arm | 구성 | 신규런? |
|---|---|---|
| floor | — | 재사용(0.557/0.411/0.358/0.333) |
| prov-only | C53 | 재사용(0.5768) |
| **COMP** | 게이트(kinds 6종)+nested+calc+prov | **★신규** |
| **COMP+D** | COMP + T2_DISAMB | **★신규** |
- 산출: 공식 pass^1..4 · compliant-pass(t2_compliance) · **레버 실발화 census**(gate deny 라운드·prov regen 수·disamb 발화/switch 수 — stderr 카운터를 러너가 sim별 로그) · tme/infra · Δspurious per-case(floor-pass→arm-fail 전건 정독·레버 귀속) · DISAMB switched-from-gold census.
- **판정**: ①COMP가 prov-only(0.577) 대비 ≥+2pp ∧ compliant=bench(위반0) ∧ Δspurious≤0 → 합성 GO ②COMP+D가 COMP 대비 ⋈/변형 버킷 감소(per-case) ∧ switched-from-gold ≤ switched-to-gold → DISAMB e2e GO(=T5-B 종결·C59 e2e 승격). pass^1 노이즈 주의(67% flaky) → 주장은 pass^1..4 병기+버킷 per-case로.

## 3b. ★사전 census — prov arm 실패 193건 전수 + 레버-도달가능성 (2026-07-10 · [M] · 무료)
> `ecomp_fail_census.py` · 대상=`prov_e2e_retail_t4.results.json.gz`(456 sims·C53 canon·pass 263=0.577) ·
> disamb-도달 = 잘못 쓴 인자의 gold 값이 그 write *이전* 조회 문맥에 실재 ∧ 쓴 값도 실재(=|C|≥2 근사).

| 관찰 버킷 | n | %fail | disamb-도달 | gate-제약위반 | retry루프 |
|---|---|---|---|---|---|
| WRONG_ITEMS (변형/집합) | 34 | 17.6% | **33** | 5 | 0 |
| OVER_ACTION | 30 | 15.5% | 2 | 10 | 4 |
| WRONG_REF_ORDER (⋈) | 30 | 15.5% | **30** | 4 | 0 |
| MISSED_WRITE (부분 미완) | 25 | 13.0% | 0 | 1 | 1 |
| ZERO_WRITE 미시도 | 22 | 11.4% | 0 | 0 | 0 |
| NL_ONLY (db는 통과) | 19 | 9.8% | 0 | 2 | 2 |
| ZERO_WRITE 전부에러 | 13 | 6.7% | 0 | 4 | 6 |
| WRONG_PAYMENT | 8 | 4.1% | **8** | 0 | 1 |
| WRONG_ADDRESS | 8 | 4.1% | 4 | 1 | 0 |
| OTHER_ARG | 4 | 2.1% | 0 | 3 | 1 |
- **합계: disamb-도달 77/193(=16.9pp 천장) · gate-제약위반 30 · retry루프 15 · ★prov-잔존 날조 0**(C45/C53 정합 — 날조 축은 이미 닫힘).
- per-case 검증(2건 정독): t2=8회 조회 후 item 오선택·gold 문맥 실재(교정 가능형) / t71=체계핵(C56④ "최근 주문" argmax 오적용)·후보 제시만으론 부분 저항 → **체계핵은 calc-argmax(Phase 2) 몫·DISAMB 전환율을 100%로 못 봄**.
- **★trial-일관성 교차표([[08]] guard 보완)**: disamb-도달 77 = **전패(0/4) 태스크 몫 31**(체계적·안정 질량 — 태스크 101/103/109/71/76=⋈·61/98=payment·37=items·20. ★robust-core 9태스크[8,17,20,34,36,37,71,101,109] 중 **5개+가 disamb-도달**=pass^4도 개선 가능) + **일부통과 태스크 몫 46**(user-sim seed 공변·전환 노이즈). 전패 fail sims 총 76 중 31이 disamb-도달.
- **기대치([D]·판정은 실측)**: 체계 질량 31×전환 ~40%(t71류 체계핵은 후보제시 부분저항·정독 확인) ≈ +2.7pp + flaky 질량 46×~30% ≈ +3pp ⇒ **DISAMB +4~7pp** · 게이트(제약steer 30·retry 15) **+1~3pp** · 합성 COMP+D = **0.577 → 0.63±0.03**(pass^1·노이즈 1pp≈4.56 sims·주장은 pass^1..4+버킷 per-case). 규칙0-준수로 역사수치 0.640 동급+ 기대. C59 격리 전환율 47%를 e2e 기대의 상한으로.
- **COMP+D가 못 닫는 잔여** = MISSED(25)+ZERO미시도(22)+NL(19)+OVER_ACTION(~28) ≈ 94 sims ≈ **20.6pp** — coverage/persistence(C32 Δ=0 미확인)·대화-semantic(C50 NO-GO)·NL 채점 축. 이번 scope 밖(정직 명기).

## 4. 실행 계획 (비용·순서)
1. **구현+단위테스트(무료·로컬)**: unified regen·기존 두 경로 회귀 테스트(retail A2·mock 시나리오)·이중과금/R8/DISAMB 이식 검증.
2. **스모크(소액)**: 10태스크 nt=1 COMP+D — 레버 3종 실발화 각≥1 확인([[30]]·미발화시 중단), 크래시 0, tme 폭증 없음.
3. **full 2런(유료·승인必)**: COMP → COMP+D 순차. 456×2=912 sims ≈ C53 규모 2배. **GPU: banking full 완료 후 GPU1**(타 세션 GPU0 불가침). distinct tag `comp_retail_t4`·`compd_retail_t4`·런별 즉시 persist.
4. 종료: 원장 C-신규(합성·DISAMB e2e)·§4 갱신·덱 결과 그래프(규칙0-준수 스택으로 0.640 대체 여부 판단).

## 5. 리스크·중단조건
- **regen 예산 상호잠식**: 게이트+prov 피드백이 같은 K를 나눠 씀 → tme↑ 가능. 스모크서 tme>10% 시 K_UNIFIED 6으로 1회 조정(그 이상은 중단·설계 재검토).
- **DISAMB 역효과**: 맞은 선택을 흔들어 switch-from-gold>switch-to-gold면 **DISAMB만 제거**(COMP는 독립 판정).
- **0.640과의 비교 함정**: 그 수치는 present 포함 — COMP가 0.640에 못 미쳐도 실패 아님(**비교 기준은 prov-only 0.577과 floor**). 덱·문서에 present 철회 명기.
- 게이트 이득이 bench-pass에 안 나올 수 있음(C32 짝맞춤 Δ=0 소표본) — 그 경우에도 compliant-pass=bench(위반0)가 산출물(모트 주장 유지·C2 계열).

## 6. 상태
- [D] 설계·**사용자 리뷰 대기**. 리뷰 통과 → §4.1 구현 착수. full-run은 banking(E-XFER-bank) 완료 후.
