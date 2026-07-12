# A1 회귀 전수 per-step 포렌식 (2026-07-13) — 핸드오프 §2 귀속 교정

> ★사용자 지시로 A1 회귀를 per-step 전수 재분석([[08]]). **핸드오프 §2 기전표의 절반이 오귀속**임을 확증.
> 데이터: `sim_results/gen_A1.results.json.gz`(nt=1) vs `comp_retail_t4`(nt=4). 스크립트 `forensic17.py`·`dump_full.py`.
> 상위 = `GENERALIZED_SCAFFOLD_ARCHITECTURE_2026_07_12`(LOCK) · 갱신 대상 = HANDOFF §2·§6.

## 0. 회귀 집합 (gz 정본·핸드오프 "17"은 부정확 큐레이션)
- **strict (COMP 4/4 ∧ A1 fail) = 8**: 6·7·23·43·46·55·92·108
- **major (COMP ≥50% ∧ A1 fail) = 25 (strict 포함)**: +21·32·33·38·42·52·58·59·69·77·83·84·94·96·101·110·112
- A1 db = 71/114. **핸드오프 §2 "17"은 strict 8 + major 임의 9의 혼합**(임계값 아님) → 본 문서가 정본.

## 1. ★핵심 교정 — 오귀속 2건 (사용자 의심 적중)
### 1a. "eplan 과-블록"(t21·32·42) = **오귀속**. 실제 = 에이전트 자발 transfer / coverage
- **committed 블록 사유 전수 = `G4_TRANSFER_MSG`**(에이전트가 `transfer_to_human` 호출→게이트가 처리). eplan deny는 생성-레벨(비커밋)이라 gz에 안 남음 → gz의 "policy gate 블록"=진짜 게이트(transfer). **eplan이 write를 막은 증거 gz에 0.**
- **t32**: user-sim 이탈(태스크=cancel/return인데 "분실 태블릿 환불"로 즉흥→정책상 불가→transfer 타당). = **하네스/user-sim 아티팩트**.
- **t21**: pending 주문에 `exchange_delivered`(오도구) 시도→불가 인지 못하고 루프→transfer. gold=`modify_pending_order_items`. = **도구선택 오류(base)**.
- **t42**: #W4082615는 정확 수행. gold는 **#W9583042도** + modify_user_address = **coverage 누락**(선택오류 아님).
- ⇒ **`T2_EPLAN_EXAMINED_SAFE`(A2-v2 (2))는 이 셋을 못 고침** — 막힌 write가 없음.

### 1b. "prov 주소날조"(t43·96) = **오귀속/구조적 불가**. prov로 못 잡음
- **t43**: 사용자가 주소 명시("1234 S Michigan Ave…") → 에이전트 그대로 사용 = **날조 아님**. db-fail 근본=별개(과제 의도=default 변경 정책/배송지-확인). = **오분류**.
- **t96**: NYC 주소는 **다른 주문 #W3407479에 존재**하나 에이전트가 그 주문을 **조회 안 하고 "123 Broadway" 날조**→사용자에 제시→user-sim이 그대로 확인 재발화("123 Broadway 맞다"). ⇒ write 시점엔 값이 **ctx(사용자 발화)에 실재** → **prov(provenance=ctx有無)가 구조적으로 못 잡음**. `T2_PROV_ADDR_FULL`(A2-v2 (3)) 무효.
- ⇒ **근본 원인 = 누락된 GET**(원천 주문 미조회). 처방=조회 강제/ NL-주소 provenance(발화-laundering 대응)·prov-advise 아님. **사용자 지적 정확: getter 하면 안 생김.**

## 2. 25 회귀 정본 원인 분류 (per-step 확증)
| 카테고리 | tasks | 근본 원인 | A2-v2 커버? |
|---|---|---|---|
| **A. 자발 transfer/포기** | 6·21·32·33·58·69 (6) | 에이전트 give-up(다항목/정책복잡)·t32=user-sim이탈·t21=도구선택 | ❌ (eplan-safe 무효) |
| **B. 주문-선택 ⋈ (후보 다수 조회)** | 55·59·101 (3) | 조회한 여러 주문 중 오선택 | ✅ **filter-substitute**(내용제약 有 시) |
| **C. 주문-선택 ⋈ (틀린 주문만 조회)** | 83·92·112 (3) | 맞는 주문 **미조회**→선택불가 | ⚠️ **GET-강제 선행 필요**(filter 단독 무효) |
| **D. coverage 누락** | 42 (1) | 한 주문만 수행·2번째+user-addr 누락 | ❌ (walk off) |
| **E. 변형-선택(new_item_ids)** | 7·23·52·94·110 (5) | 교체 variant 오선택(색/사이즈/config·cheapest) | ❌ content-match/INFER |
| **F. 주소** | 43·96 (2) | 43=오분류·96=날조(누락 GET) | ❌ (§1b) |
| **G. 오행동/오도구/미수행** | 77·84·108·38·46 (5) | 77=신규주문(교체아님)·108=cancel(return아님)·38=split미수행·46=return품목오선택·84=partial | ❌ base/learn |

**요약**: A2-v2가 **정당히 고칠 잠재 = B(3) + (GET-강제 추가 시) C(3) = 최대 6/25**. eplan-safe·prov-addr(5 태스크 타깃)는 **오귀속으로 무효**.

## 3. 진짜 레버 (교정된 처방)
- **B/C (주문선택 6)**: 설계 GET→FIND 루프 정합 — **①GET-강제**(애매 주문 write 전 후보 주문 상세 전수 조회) **②FIND=filter-substitute**(내용제약→결정론). filter만으론 C 불가(record<2 폴백 실증). = 다음 구현 1순위.
- **E (변형선택 5)**: content-match/INFER(cheapest/색/config) — cheapest류=결정론 DB-getter(LOCK §8)·색/사이즈=formalize-filter 확장(variant record)·잔여=learn.
- **F-96 (주소 날조)**: 누락 GET(원천 주문 조회 강제) 또는 NL-주소 provenance(발화-laundering 대응·NL-NUM-PROV 동형). prov-advise 아님.
- **A (transfer 6)·G (5)**: base/도구선택/coverage — 상당수 user-sim변동·정책네비 = scaffold 아닌 learn/scale 또는 harness 정제.

## 3b. ★★단일-요인 ablation 실측 (2026-07-13·GPU0/1·nt2·회귀25) — 핸드오프 §2 sign 반박
> arm1=`compabl_noP`(COMP−present·EPLAN0)·arm2=`compabl_noP_eplan`(+eplan). prov=full·cap off·present off 고정, **eplan만 변수**(단일-요인). 데이터 = `sim_results/compabl_noP*.gz`.

**COMP(present·nt4) → [−present] → arm1(52%) → [+eplan] → arm2(62%) → [+prov-rescue+cap] → A1(0%*)**

| 델타 | 효과 | 판정 |
|---|---|---|
| **+eplan** (arm1 52%→arm2 62%) | **NET +10pp (+5 sim)** · HELPS 8(21·23·33·42·52·94·108·110=coverage/멀티엔티티) · HURTS 6(32·38·43·58·69·83) | ⚠️ **핸드오프 "eplan 과-블록 순손실" = sign 오류. eplan은 순이득.** |
| **−present** (COMP~100%→arm1 52%) | present↓ 9(7·21·23·33·42·52·101·110·112) | 실제 해악·단 **대부분 eplan이 복구**(substitute) |
| **cap** | A1서 0발화(기존 확정) | null |
| **prov full→rescue** | 미검(arm 없음) | arm2 62%→A1 0% 갭의 잔여 용의자 |

**★핵심 발견 = present↔eplan 상호작용(substitute)**: 21·23·33·42·52·110은 COMP=pass → −present=fail → +eplan=pass. **present(후보 surfacing)와 eplan(discovery 강제)이 기능 중복** — 하나만 있어도 그 태스크 해결. ⇒ OFAT가 경고한 상호작용 실재. present-only 순손실(eplan도 복구 못함) = **7·101·112 3개뿐**.
**★"A1 0%*"의 함정**: 회귀25는 **A1 nt=1 fail로 선택**된 집합(선택편향). arm2가 nt2서 62% 복구 = **상당수 회귀는 nt=1 노이즈**. arm2(62%)→A1(0%) 갭은 대부분 선택 아티팩트 + (미검)prov-rescue. **진짜 회귀 규모 확정 = A1 nt2 재측정 필요.**
**★eplan-firing≠eplan-causing 확증**: §2에서 eplan L2가 14/25 회귀에 발화했으나, 통제 ablation서 eplan은 순이득. 발화는 대개 양성(discovery). 관측 발화→인과 직행이 오류였음(통제실험이 교정).

## 4. A2-v2 처분 (정직·ablation 반영)
- **유지**: filter-substitute **dotted-path 수정**(라이브 8/8 fallback 버그·확증) — B(3) 겨냥·real bug fix.
- **재고/보류**: `T2_EPLAN_EXAMINED_SAFE`·`T2_PROV_ADDR_FULL` = **오귀속 기반**·타깃 태스크 미교정 → **full-run 편입 전 근거 상실**. 토글은 두되 default off·B/C의 GET-강제로 대체 설계.
- **다음**: GET-강제(주문선택 discovery) 설계·구현 → B+C 6 태스크 겨냥 → 격리 probe(무료 우선·[[09]]).
