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
  am = generate(base); gate_rounds = prov_rounds = 0
  loop:
      fab    = first_fab_call(am, ctx)                       # 날조 인자
      denied = gate_denials(am, gate)                        # 정책 deny
      do_gate = denied and gate_rounds < 1                   # ★게이트 피드백 라운드 = 최대 1 (기존 K=1 승계)
      do_prov = fab and prov_rounds < 4                      # ★prov = 최대 4 (기존 K=4 승계)
                and not (do_gate and fab-call ∈ denied)      # 같은 콜엔 게이트 피드백 우선(이중피드백 금지)
      if not do_gate and not do_prov: break
      if do_gate: gate_rounds++; num_errors++                # ★게이트 라운드만 과금 (블로킹1: prov=무과금=C53 semantics)
      if do_prov: prov_rounds++                              # 무과금
      am = generate(base + [am] + 병합-피드백(gate reason 우선, prov, 나머지 hold))
  # 종단
  gate-denied 잔존 호출 → R8 strip + _BLOCK_NOTE (재과금 없음·replay-safe·기존 semantics)
  prov-fab 잔존 호출 → 통과(실행) (기존 semantics·env가 id 거부=C12)
  if DISAMB and clean:
      am2 = 1회 재확인(기존 규칙·prov 정화 2회 한도)
      ★if gate_denials(am2): am2 폐기·원 am 유지            # (블로킹2: 재확인 switch가 게이트-deny 호출을 들여오는 구멍 봉쇄)
      else: am = am2
  return am
```
- **★예산 semantics = 두 GO arm 그대로 승계**(리뷰 블로킹1 반영): 게이트 deny가 포함된 라운드만 tick·게이트 피드백 최대 1회(그 후 잔존은 즉시 R8 경로)·prov는 무과금·최대 4. ⇒ **COMP = 두 GO arm의 정확한 중첩·나눠 쓰는 것은 토큰뿐**(K_UNIFIED 논점 소거·§5의 K=6 에스컬레이션 조항 폐기).
- **순서**: prov와 gate 피드백을 **같은 라운드에 병합**(핑퐁 방지·토큰 절약)·같은 콜에는 게이트 reason 우선. DISAMB는 클린 확정 후 마지막·**채택 전 게이트 재검사 필수**(블로킹2).
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
- ★metric 명시(리뷰 반영): §0의 floor 0.557=**pass^1**·C53 floor 0.547=**reward 평균**·prov-only 0.5768=**reward 평균**. **판정 ①의 "+2pp"는 reward 기준(0.577 대비)**·pass^1..4는 병기. arm 표 대조: floor reward 0.547 / prov-only reward 0.5768.
- 산출: 공식 pass^1..4 + reward 평균 · compliant-pass(t2_compliance) · **레버 실발화 census**(gate deny 라운드·R8 strip·prov regen·disamb 발화/switch — stderr 카운터·**preconditions/constraints deny는 env-error와의 중복 여부 표기**=env-mirror 귀속·리뷰 소견) · tme/infra · **★짝지은 flip census를 1차 증거로**(같은 456 task×trial에서 prov-only-pass→COMP-fail / 역방향·C32 교훈: 비-짝 Δ=구성 아티팩트) · Δspurious per-case 정독 · DISAMB switched-from-gold census.
- **판정**: ①COMP가 prov-only(reward 0.577) 대비 ≥+2pp(짝 flip census 1차·집계 부차) ∧ compliant=bench(위반0) ∧ Δspurious≤0 → 합성 GO ②COMP+D가 COMP 대비 ⋈/변형 버킷 감소(per-case) ∧ switched-from-gold ≤ switched-to-gold → DISAMB e2e GO(=T5-B 종결·C59 e2e 승격). pass^1 노이즈 주의(67% flaky).

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
- **★수정(2026-07-10 NL_ONLY 19건 전수 분류 + 궤적 정독 3건)**: NL_ONLY는 judge 노이즈가 아니라 **진짜 미/오보고**. **count-계열 11건(tasks 2/3/4×trials) = calc 사정거리 확인** — t3/0 정독: `get_product_details` 2회 조회(=calc 트리거 실발화) 후 "**12** available options" **오산 보고**(gold=10) = 정확히 `count_where available` spec이 고치는 F2b 실례. t4/0: 조회 1회·미보고 = calc 주입 후 report-conversion 조건부. **총액/환불액 5건** = sum spec 조건부 커버. **밖 3건**(t40 지불수단 보고·t104 tracking#·t105 coverage형). **prov arm은 calc OFF·COMP arm은 T2_CALC=1** → 최대 +2~3pp 상향 요인(조건부). 진짜 scope-밖 잔여 ≈ **78~81 sims(≈17pp)** = MISSED+ZERO미시도+OVER_ACTION+기타. 후속 레버 지도(plan/execute C1·feasibility·retry)=`FIXABLE_FAIL_CENSUS_2026_07_06` + 본 census 교차.

## 4. 실행 계획 (비용·순서)
1. **구현+단위테스트(무료·로컬)**: unified regen — per-lever 예산(게이트 1회 tick·prov 무과금 4회)·R8 strip·같은-콜 이중피드백 금지·DISAMB 채택-전 게이트 재검사·기존 단독 경로 무변경(회귀). tau2 stub 로컬 하네스 + 리모트 실 tau2 재실행.
2. **스모크(소액)**: 10태스크 nt=1 COMP+D — **★태스크는 발화 조건 보장형으로 의도 선택**(리뷰 반영): t17(날조 정본=prov)·게이트-deny 이력 태스크·DISAMB 후보 다수(⋈ 전패 101/103/109·payment 61/98). 레버 3종 실발화 각≥1 미달 시 중단([[30]])·크래시 0·Δtme 계측.
3. **full 2런(유료·승인됨)**: **COMP 456 완료 → ★중간 체크포인트**(리뷰 반영: per-case 정독 先 — Δspurious>0 or GO 대폭 미달 시 COMP+D 발사 전 재검토·[[09]] 최악 456 sims 절약) → COMP+D 456. **GPU: banking full(E-XFER-bank) 완료 후 GPU1**. distinct tag `comp_retail_t4`·`compd_retail_t4`·런별 즉시 persist.
4. 종료: 원장 C-신규(합성·DISAMB e2e)·§4 갱신·덱 결과 그래프(규칙0-준수 스택으로 0.640 대체 여부 판단).

## 5. 리스크·중단조건
- **regen 상호잠식**: per-lever 예산 승계(블로킹1)로 과금·한도는 기존과 동일 — 나눠 쓰는 것은 토큰뿐. 잔여 리스크는 스모크 Δtme 계측이 커버. ~~tme>10%시 K=6 조정~~ **폐기**(리뷰: K↑는 prov semantics를 C53에서 멀어지게 함).
- **DISAMB 역효과**: 맞은 선택을 흔들어 switch-from-gold>switch-to-gold면 **DISAMB만 제거**(COMP는 독립 판정).
- **0.640과의 비교 함정**: 그 수치는 present 포함 — COMP가 0.640에 못 미쳐도 실패 아님(**비교 기준은 prov-only 0.577과 floor**). 덱·문서에 present 철회 명기.
- 게이트 이득이 bench-pass에 안 나올 수 있음(C32 짝맞춤 Δ=0 소표본) — 그 경우에도 compliant-pass=bench(위반0)가 산출물(모트 주장 유지·C2 계열).

## 6. 상태
- [D] 설계·**사용자 리뷰 대기**. 리뷰 통과 → §4.1 구현 착수. full-run은 banking(E-XFER-bank) 완료 후.
