# Trivial-회귀 오염원 절단 — 개입 vs 가드 이분 (2026-07-12)

> 소유: b78c(COMP+D-v2+cap) 스택이 COMP-강건 trivial 태스크를 회귀시키는 오염원을 양방향 절단으로 특정.
> 방법론: [[08]] per-case·집계-전-포렌식 · [[09]] 무료우선(절단=격리 프로브·소액) · temp=0(agent+user) · db_match 기준(C22·NL 혼입 배제).
> provenance = §7. 상위 = `RESEARCH_MASTER §3`(C73 예정) · `B78C_FORENSIC_AND_S1_REDESIGN` · NIGHT 핸드오프 §0z.

## 0. TL;DR
1. **trivial36_check 회귀 실재**: COMP=1.000(36 trivial·nt4 144/144)인데 **b78c(full)=30/36=0.833**(nt1). 6-fail{1,6,7,23,75,106} 전부 COMP nt4 **4/4 robust** → flaky 아닌 **진짜 회귀**.
2. **양(load) 아님·질(misdirection)임**: COMP vs b78c **시스템프롬프트 바이트 동일**(6699 chars=1674 tok). C66("정적부하 20k 강건")와 모순 0 — 양이 안 늘었으니까. 회귀는 **런타임 엔진 인터셉트**가 관측/행동을 바꿔 발생.
3. **오염원 = 개입(intervention) 레버, 무죄 = 가드(guard) 레버**. 에이전트 operand/값/discovery 선택을 *바꾸는* 레버(DISAMB·EPLAN·GROUND·PRINCIPLE)가 유죄. *막기만* 하는 레버(PROV-rescue·cap)는 양방향 무해.
4. **단일범인 아님·중복 상호작용**: DISAMB 혼자 충분(additive 0/2)하나 빼도 안 고쳐짐(subtractive)·타 개입레버가 대신 깸. 어느 하나 빼도 2/2 회복 불가.
5. **= 등대 Δspurious 원리 정밀 실증**. 하드-78 도우려 넣은 개입레버가 trivial에 spurious 발화 = "레버는 하나 사면 하나 판다".
6. **처방** → `INTERVENTION_LEVER_CONDITIONALIZATION_DESIGN_2026_07_12`(개입레버를 결핍-조건 발화로 전환·도메인일반).

## 1. 회귀 실재 (trivial36_check)
- COMP `T2_GATE_REGEN·PROV_REGEN(K4)·PRESENT_NESTED·CALC` = 36 trivial always-pass(nt4·db 1.000).
- b78c(full+cap)·nt1·36 trivial → **db 30/36=0.833**. 6 db-fail = task 1(max_steps 정치성 루프)·106/7(변형-⋈)·23(coverage 누락)·6(transfer)·75(오exchange).
- **대조**: `comp_retail_t4` 파일서 6-fail 전건 COMP **4/4 db=True**(16/16) = COMP-강건. ⇒ 회귀는 b78c 스택 탓·flaky 아님.

## 2. 양 아님 (시스템프롬프트 실측)
- COMP task106 sys-prompt = b78c task106 sys-prompt = **6699 chars(≈1674 tok)·바이트 동일**(diff 0라인).
- D-v2 레버는 프롬프트텍스트를 안 늘림 = 전부 **런타임 엔진 인터셉트**(PRESENT_NESTED=툴출력 확장·GROUND/PRINCIPLE=write-arg 치환·DISAMB=subcall 재해소·PROV=regen·EPLAN=deny). ⇒ 회귀는 **관측/행동 스트림 변경**(질적), 프롬프트 부하(양) 아님.
- 선행연구 정합(딥리서치): scale-의존 instruction-load(IFScale 2507.11538·ManyIFEval 2509.21051)는 *프롬프트 콘텐츠*에 관한 것 → **본 기전(엔진-레버 spurious)과 다름**·본 construct는 whitespace. lost-in-middle(2307.03172)·multi-turn(2505.06120)은 scale-agnostic이라 인용주의(기전=multi-turn 누적만 지지).

## 3. 양방향 절단 (task 106·nt=2·db_match)
> temp=0이나 vLLM 배칭 비결정성 존재(1/2 flip 다수)=under-sampled rate. 정밀은 §실세계 nt4.

**Additive (COMP + 레버 하나):**
| config | +레버 | t0 | t1 |
|---|---|---|---|
| c0 | (base COMP) | ✓ | ✓ (2/2 대조) |
| c1 | PROV rescue | ✓ | ✓ (무해) |
| **c3** | **DISAMB subcall** | ✗ | ✗ (**0/2·단독 충분**) |
| c2 | GROUND | ✓ | ✗ (1/2) |
| c4 | PRINCIPLE_DEFAULT | ✓ | ✗ (1/2) |
| c5 | EPLAN walk | ✗ | ✓ (1/2) |
| cf | full | ✗ | ✓ (1/2) |

**Subtractive (full − 레버 하나):**
| config | −레버 | t0 | t1 |
|---|---|---|---|
| sfull | (full) | ✗ | ✗ (0/2 대조) |
| no_disamb | −DISAMB | ✗ | ✓ (1/2 부분회복) |
| no_eplan | −EPLAN | ✓ | ✗ (1/2 부분회복) |
| no_ground/no_princ/no_prov/no_cap | 각 제거 | ✗ | ✗ (0/2 무변화) |

## 4. 개입 vs 가드 이분 + 상호작용
| 레버 | additive | subtractive | 유형 | 판정 |
|---|---|---|---|---|
| DISAMB subcall | 0/2 | 1/2 | 개입 | **주 오염원** |
| EPLAN walk | 1/2 | 1/2 | 개입 | 기여자 |
| GROUND | 1/2 | 0/2 | 개입 | 2차(중복) |
| PRINCIPLE_DEFAULT | 1/2 | 0/2 | 개입 | 2차(중복) |
| PROV rescue | 2/2 | 0/2 | 가드 | **무죄** |
| cap | — | 0/2 | 가드 | **무죄** |

- **이분**: 행동을 *바꾸는* 개입레버 = 유죄 / *막기만* 하는 가드레버 = 무죄.
- **상호작용**: DISAMB 단독 충분하나 제거로 안 고쳐짐 = **중복 spurious 원인**. 최선 단일제거(DISAMB/EPLAN)도 1/2. ⇒ "나쁜 레버 하나 제거"로 안 됨.

## 5. 방법론 caveat — 비결정성은 커버할 대상 (제거 대상 아님)
- vLLM 0.11.0(enforce-eager ON·prefix-caching ON·batch-invariant 미탑재)·temp=0이나 **배치조성 FP로 비결정** → 1/2 flip.
- **결정성화는 잘못된 목표**: 배포도 비결정(배칭+실사용자)이라 결정성화=인위적 단일경로·분산 은폐. 레버 해악=misdirect *확률* 상승 → **rate 추정이 옳음**(프로젝트 pass^k·C60 정합).
- ⇒ 1/2 = under-sampled 진짜 확률. 정밀 metric = **Δ(pass-rate)**·CI. → §8 실세계 nt4.

## 6. 처방 (→ 별도 설계doc)
개입레버를 **결핍-조건 발화**로 전환(도메인일반·[[05]]): 발화 트리거를 *가능한 애매성*(과발화)→*검증가능 결핍*(무효/누락/미검토)으로. 상세 = `INTERVENTION_LEVER_CONDITIONALIZATION_DESIGN_2026_07_12`.

## 7. Provenance
- 회귀검증: `sim_results/trivial36_check.results.json.gz`(nt1·36·`trivial36_check_run.sh` exports) · COMP 대조 `sim_results/comp_retail_t4.results.json.gz`.
- 시스템프롬프트 실측: 두 gz의 task106 policy 필드(6699 chars 동일).
- 절단: `scripts/distill/tau2/abl_sysprompt_106.sh`(additive)·`abl_sysprompt_106_sub.sh`(subtractive)·`sim_results` 미영속(진단·요약 본 doc §3). 절단 결과 회수 = 세션 폴러 `bakh4duov`.
- COMP exports = `comp_full.log` (GATE_REGEN K1·PROV_REGEN K4·PRESENT_NESTED·CALC / GROUND·DISAMB·PRINCIPLE·EPLAN·PROV_MODE 없음).

## 8. 실세계 분포 확정 (nt4·진행중)
> `abl_realworld_6fail.sh` = 6-fail × {comp·full·guard-only} × nt4 = 72 sim. persist `sim_results/bstack_{comp,full,guard}.results.json.gz`.
> guard-only = COMP + PROV-rescue + cap (개입레버 0) = "regression-safe 최소스택" 가설 검정.
> GO = full < comp(개입 회귀 실재) ∧ guard ≈ comp(가드 안전). **[결과 회수 후 기입]**
