# Assembled-Stack 잔여 census 설계 (2026-06-25) — 결정론 천장 + operand 잔여 3분해 (make-or-break 계기)

> 결정 = (b) [사용자 2026-06-25]: constraint-gate(pass-임팩트 ≈8 천장·대부분 hygiene·정적 gold-read로 확정) GPU 측정 *건너뛰고*, **조립 스택 잔여 census**에 GPU를 써서 마스터 질문에 직결: **"결정론으로 어디까지 닫히고, 그 아래 operand 잔여가 (고칠 수 있는)present-개선인가 / capability인가 / learn인가."** whack-a-mole 바닥을 측정으로 확정. [[03]]설계먼저·[[05]]·[[08]]·pass^1금지.

## 0. 왜 marginal 아닌 조립인가
개별 arm marginal(present·g15·nested·constraint 각각)은 상호작용·중복을 놓침. 결정론으로 *동시에 닫을 수 있는 최대*를 보려면 **모든 깨끗한 결정론 레버를 한 스택**으로 돌려 잔여를 census. 그 잔여가 진짜 질문(learn/capability).

## 1. Phase-1 — Assembled-stack run
- **스택 = present + nested + 전체 게이트(auth·confirm·ownership·notice·preconditions) + new≠old(disjoint)**. (=현 present+nest+g15에 깨끗한 disjoint constraint 1개 추가. count-match·payment는 §3 사유로 제외.)
- 드라이버: 기존 `reexp_present_nested.sh` 패턴 + `T2_GATE_KINDS=...,constraints` (A2엔 disjoint 인스턴스만 활성). 32B+14B·retail t3·gpt-4.1 user-sim·replay-safe.
- 회수: `escape_det_census.py --clean`(L0-L3·MATCH·over·pass^k) + 종료분포 + 전수 에러 taxonomy(`NESTED_ARM_FAILURE_CENSUS` 도구 재사용).

## 2. Phase-2 — 잔여 태깅 (정밀화 1·오염 방지)
reward=0 task(trial 다수)를 실패원으로 태깅 — **이미 있는 분류기 재사용**:
- **`constraint-addressable`** (기지·미빌드): 실패가 count-match(refuse-gold 제외·exchange-gold만) 또는 payment 위반 = "결정론으로 더 닫을 수 *있었으나* 안 빌드". 정적 회계 ~8 task. **operand에 섞지 말 것**(operand 과대계상 차단).
- **`operand`** (진짜 질문): silent-wrong-write L2/L3(wrong item/variant) 또는 operand not_found(item/variant id 그라운딩). → Phase-3로.
- **`other`**: MISS/상류(auth·communicate)·bizrule-기타·comprehension-비operand.
- 출력: 잔여의 {constraint-addressable / operand / other} 분포 = "결정론 천장 위치".

## 3. Phase-3 — ★operand 잔여 3분해 (정밀화 2·crux·Probe-B 모순 해소)
**모순**: Probe-B=후보 떠먹이면 select 7/7 작동 ↔ present-arm=결정점 present해도 L2/L3 약간만(32B wMatch+.021·14B neutral). ⇒ operand 잔여를 learn/capability로 *뭉치면 오류*. 3분해:

**계기 = operand-잔여 케이스마다 Probe-B-품질 격리 진단 + 라이브 present 발화여부 교차:**
1. 케이스 추출: 그 write의 NL 기준(유저가 어느 item/variant) + gold target(정답 id) + **clean 격리 choice-set**(주문의 전 item / product의 전 variant·라벨링·Probe-B 형식).
2. 격리 probe 실행: "유저 요청=<NL>. 후보=<clean choice-set>. 어느 id?" (orchestration 0·단일턴) — 같은 모델(7B/14B/32B).
3. 라이브 교차: 그 결정점서 present/nested가 *발화*했나? 라이브서 맞췄나/틀렸나?

**분류 (배정):**
| 격리 probe | 라이브 present | 진단 | 레버 |
|---|---|---|---|
| **PASS** | 발화O·라이브 틀림 | **(i) present-형식 약함** | **결정론**: 라이브 present를 Probe-B 품질로 개선(buried→clean choice-set at decision point) |
| **PASS** | 발화X(미커버) | **(ii) coverage/orchestration** | 결정론: present 커버 확장 / or 라이브-부하 한계(capability) |
| **FAIL** (clean 줘도 틀림) | — | **(iii) comprehension** | **learn**: NL→item 매핑 자체 오류 = 진짜 learn 타깃(있다면) |

- **핵심 산출 = (i):(ii):(iii) 비율.** (iii)comprehension이 크면 → learn GO(유일 정당 근거). 작고 (i)/(ii) 지배면 → **learn NO-GO + 다음 결정론 레버 = present를 Probe-B 품질로 개선**(format-weakness 닫기). = make-or-break의 *측정된* 답.
- ⚠️ (iii) 내에서도 operand-copy(C4계열·전이음성·[[20]]) vs criterion-formalize(σ로 못 만드는 기준)를 추가 구분 — C4계열이면 learn도 음성·기약 capability.

## 4. 측정 규율 ([[08]])
- 결정론 지표(escape_det_census·층 census)·다수 trial·pass^k. pass^1 점추정 단독 금지.
- crash/infra/too_many 배제(--clean)·종료분포 먼저.
- Phase-3 격리 probe는 user-sim 무관(단일턴·gpt-4.1 user-sim 불요·COST GUARD).
- over-deny 체크(new≠old가 양성 write 막나·false-block 0 기대지만 확인).

## 5. 빌드 범위 (정밀화 3)
- **동승(빌드)**: new≠old(disjoint·3 task·false-block 0·decidable 확실)만 constraint로 스택에 추가.
- **제외(정적 천장만 doc 기록·`CONSTRAINT_GATE_DESIGN` §accounting)**: count-match(hygiene·refuse-gold 10/loop-death0)·payment(marginal~4·다수 이미 복구·[[05]] 파생필드 _orig_payment/_gift_cards 위험·policy.md 미확인).

## 6. 산출 / 논문 기여
- **결정론 천장 = present+nest+gate+disjoint 후 pass·잔여 분포**(constraint-addressable/operand/other).
- **operand 잔여의 (i)present-형식/(ii)capability/(iii)learn 분해** = "소형 도달의 마지막 잔여가 무엇이고 어느 레버인가"의 *측정된* 답 = make-or-break.
- **정적 회계 기여**(별도): "126 bizrule 에러 → pass-flip 천장 ≈8·대부분 hygiene"을 GPU 없이 gold-read로 deprioritize = 능력→레버 배정 가이드라인 실증([[03]]/[[06]]).

## 7. NO-GO / 분기
- (iii)comprehension-learn 잔여가 의미있고 *non-C4*(criterion-formalize)면 → **learn GO**(priority-4 SFT·present-불가 잔여 한정·prep doc §4.1).
- (i)/(ii) 지배면 → **learn NO-GO·다음=present 품질 개선(결정론)**.
- operand 잔여가 (iii)지만 C4-copy계열이면 → learn도 음성 = 기약 capability 경계(scale/escalate·[[13]]).
