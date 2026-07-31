# BANK E-PLAN 전-액션 균일 loop 라이브 구현 설계서 (2026-07-16)

> [[03]] 설계먼저·[[05]] 리터럴0. 정본 상위설계 = `E_PLAN_LIVE_WIRING_DESIGN_2026_07_11`(v1.3·3컴포넌트)
> + `BANK_EPLAN_CONTROLLER_DESIGN`(5원소 all-action) + `BANK_DAG_CONTROLLER_LIVE_WIRING`(dag_plan 도메인일반 분류).
> **이 문서 = 위 개념설계 → t2_eplan_patch 구체 코드변경 매핑.** 리뷰 후 구현.

## 0. 오류 진단 + 레버 tier 프레임 (BANK_TWO_TRACK §0·§6 정본)
- 2026-07-16 (a)(b)(c) 라이브 배선 시 **write_tools를 dispute 3개로 좁힘** → dispute 태스크(6/43)서만 발화. 설계는 **전-액션 균일 loop**(dag_plan: read=FIND·非read非procedural=write·도메인일반 read-prefix).
- **★per-STEP ≠ per-SIM (핵심 교정·재범 금지)**: C94 "COVERAGE 40.7%+FIND 27.2%=68%"는 **per-step 빈도**. 다층(78.3%)이라 한 레버로 sim pass 안 됨. **정직한 per-sim 극복 봉투(BANK_TWO_TRACK §0·리뷰 ❺·C97)**:

| tier | 레버 | per-sim 상한 | 성격 (`_OP_TIER`) |
|---|---|---|---|
| **HARD** | **FIND(discovery/reach) · COMPUTE · GET-⋈** | **9.9%**(관측 12.9%) | 강제가능·[[16]] 준수 = **진짜 지배 실현레버** |
| SOFT | +COVERAGE(write-emit) | +2.1% → 12.0% | **리마인더만·write강제금지·[[07]] soft 불신** |
| Track B | +F3-enum(상황→정책) | +17.3% → 29.3% | 스킬 학습(이 문서 밖) |
| — | 잔여 GATHER(user-데이터)·judgment | 47.2% | ASK/경계 |
| — | pure-DB blind | 23.5% | 오프라인 밖 |

- **⇒ 진짜 HARD 지배 = FIND/discovery-enforce(read 강제)** — 사용자 "reach부터". CP5 COVERAGE = SOFT(+2.1%·리마인더). 내 초판이 COVERAGE를 지배로 오판 = C97이 교정한 3중낙관 재범.
- **실현 봉투(❹§7)** = HARD 9.9% × **선택술어 정확도**(disputes 82%·card/account-ops ≥49%) × gold-free복원 89.6%. **68% 아님.** 종료 100% user_stop → H_min-stop이 메타 closer.

## 1. 목표·불변 (tier 순서)
- t2_eplan_patch를 dispute-scoped → **전-액션 균일 loop**으로 교정. **구현 우선순위 = tier 순**:
  1. **FIND/discovery-enforce (HARD·주레버·reach)**: L1(목록 미열거)·L2(상세 미검토)가 **전 write 시도**서 read 강제. 2단 열거(account→상세).
  2. **COMPUTE/GET-⋈ (HARD·inner)**: 이미 배선(keystone liability·reference_filter). 유지·전-액션 연결 확인.
  3. **COVERAGE (SOFT·리마인더)**: CP5 walk — write강제금지·리마인더만. soft-bet 계측.
  4. F3-enum = Track B(이 문서 밖).
- 불변: [[05]] 엔진 리터럴0(read-prefix=도메인일반 API-verb·dag_plan 선례·ABox field_ops) · [[14]] **read/discovery만 강제·write 강제 금지**(COVERAGE=soft 리마인더·abstain→forced-act 금지) · 생성-레벨 개입(히스토리 비커밋·REPLAY_SAFE) · **Δspurious≤0·over-action Δ≤0 계측(모트)**.
- **실현성 관문(❹§7)**: 선택술어(gold-free 타깃 복원) 정확도. disputes=reference_filter 82%·card/account-ops ≥49%. FIND는 술어가 타깃을 못 집으면 무효.

## 2. banking 도구 사실 (분류 검증·전수)
- **READ**(get/search/list/lookup/find/retrieve/read/view/check_): get_all_user_accounts_by_user_id·get_bank_account_transactions·get_debit_cards_by_account_id·get_user_dispute_history·get_pending_replacement_orders 등.
- **WRITE**(非read非proc·다수): file_*_dispute·open/close_bank_account·order/freeze/unfreeze/close_debit_card·apply_for_credit_card·order_replacement_credit_card·pay_credit_card_from_checking·update_transaction_rewards·submit_*_report·approve/submit_credit_limit_increase·apply_*_credit·submit_referral.
- **PROCEDURAL**(denylist): log_verification·log_*(closure)·call/give_discoverable_user_tool·transfer_to_human/initial_transfer·unlock_·kb_·shell.
- **write entity 인자(write별 다름·핵심)**: dispute→`transaction_id`·card ops→`card_id`·account ops→`account_id`·credit→`credit_card_account_id`. `user_id`=행위자(entity 아님).

## 3. 구현 변경 (t2_eplan_patch + a2)

### 3.1 도메인일반 write 검출 (dispute 리스트 폐기·핵심)
- 엔진에 dag_plan 미러(도메인일반·[[05]] OK):
  ```
  _READ_PREFIX = ^(get|search|list|lookup|find|retrieve|read|view|check)_
  _PROCEDURAL  = (^log_|_verification$|^kb_|^search_|^shell$|discoverable|transfer_to_human|give_|unlock_)
  _is_write(name) = name and not _READ_PREFIX.match(name) and not _PROCEDURAL.search(name)
  ```
- `build_ledger_from_messages`: `if eff_nm in wt_all` → `if _is_write(eff_nm)`. (wt_all = optional ABox override로 잔존·기본은 _is_write.)
- gate_patch L1/L2 loop: 동일 `_is_write(eff_nm)` 사용(dispatcher unwrap 후).
- ABox eplan `write_tools` = 제거(또는 optional allowlist override로 강등). read_prefix/procedural = 엔진(도메인일반).

### 3.2 entity 처리 (write-type별·도메인일반)
- 문제: 현 `entity_key="transaction_id"` 고정 → 非dispute write entity=None → coverage 매칭 실패.
- **해법(도메인일반)**: entity = nested args의 **primary id-like 값**(`*_id` 끝 ∧ ≠user_id·우선순위 = ABox `entity_keys` 리스트 or 첫 매칭). 헬퍼 `_entity_of(args, spec)`.
  - ABox eplan에 `entity_keys: ["transaction_id","card_id","account_id","credit_card_account_id"]`(우선순위·도메인 매핑=A2·[[05]] §3).
  - `note_write(intent, _entity_of(eff_ar, spec), items)`.
- coverage_gap 매칭(`_covers`): intent(substring) ∧ (entity 양쪽 있으면 일치 · 없으면 intent-count qty 매칭). 기존 _covers는 entity 필수 → **entity 없는 write도 커버**하도록 완화(intent-only fallback).

### 3.3 ★discovery-enforce (FIND·reach) = HARD 주레버·전-액션화 (2단 열거)
> BANK_TWO_TRACK §0: HARD 지배 = FIND. §1.2(b): 2단(account 열거→per-account 상세)·entity_key 갱신.
- **2단 열거 구조**: banking reach는 계좌→상세 2홉. list_enumerator=[`get_all_user_accounts_by_user_id`(계좌 열거)] → detail_reader=[`get_bank_account_transactions`·`get_debit_cards_by_account_id`·`get_credit_card_transactions_by_user`(per-account/user 상세)]. listed=열거된 account/card/txn id·examined=상세 조회한 것.
- **L1(목록-수준)**: 멀티엔티티 의도(SCOPE_TOKEN/수량≥2 or _enum_items≥3) ∧ list-enumerator 미호출 ∧ write 시도 → deny "계좌목록 먼저". read만 강제(t81형).
- **L2(상세-수준)**: 요구 N > 검토 M(distinct examined) ∧ 미검토 sibling(listed−examined) ∧ write 시도 → deny "미검토 [ids] 상세 먼저"(t95 ⓐ형). ids=agent 자신 열거출력서(규칙0 클린).
- **entity-type 혼재 caveat(R1)**: L2 sibling은 **동일 entity-type 내**서만 의미(txn↔txn·card↔card). 혼재 시 type별 그룹핑 필요 — 초판은 `entity_keys` 우선순위로 write의 type 판별 후 **동일-type listed/examined만 L2 대조**. 미구현 시 L2는 txn 중심(주 dispute) + L1(계좌 열거)이 타-type reach 커버.
- **실현성 관문(❹§7)**: FIND가 열거해도 **선택술어**(어느 record가 타깃?)가 gold-free로 못 집으면 write 못함 = 무효. disputes=reference_filter(82%)·card/account-ops=scope-all/name-link(≥49%). **선택술어는 별개 레버**(t2_resolve/reference_filter)·FIND는 열거만.
- **EXAMINED-SAFE(기존 T2_EPLAN_EXAMINED_SAFE)**: write 대상이 이미 examined면 L2 deny 생략(A1 부작용 교정 유지).

### 3.4 CP5 coverage-walk (obligation subcall) 전-액션화
- structured_replan_prompt: "every WRITE obligation" 추출(이미 전-write·intent-general). entity_key 프롬프트 = 제네릭 "the target id (transaction/card/account)".
- write 검출이 도메인일반이면 executed 정확 귀속 → coverage_gap 전-액션 정확.

## 4. Δ-계측 (설계 §4·모트·필수)
| 부작용 | 계측 | GO |
|---|---|---|
| over-read (불필요 열거) | `_eplan_reads_added`/sim·Δtme | Δtme≤0(too_many_errors 미증) |
| over-action (walk가 안 시킨 write 유도) | Δspurious vs floor | **Δspurious≤0** |
| 멀쩡한 종결 흔듦(C53 p4) | 짝 flip(pass→fail) | robust 상실≤획득 |
- **절대선**: read만 강제·write 강제 0(abstain→forced-act 금지·§1.5).

## 5. 테스트 (설계 §5·[[09]] 무료 先)
1. 단위 `test_eplan.py` 무회귀(ALL PASS 92) + _is_write·_entity_of·intent-only _covers 신규 케이스.
2. **오프라인 스모크**(무료·`bank_eplan_coverage_probe` 확장): 전-액션 build_ledger — executed/coverage_gap이 **전 write-type**(dispute+card+account)서 정확 귀속·매칭. floor 43태스크(37 비-dispute 포함)서 gap 검출율 재측정.
3. **라이브 스모크**(소액): 非dispute write 태스크(예 close_account·order_card)서 L1/CP5 발화 마커(`[T2_EPLAN]`) 확인.
4. **표적 nt=1**: floor 43태스크(전-액션 loop) vs 기존 floor(9.3%). db_match 기준.

## 6. arm 설계·성공기준 (BANK_TWO_TRACK §5·§1.3)
- **treatment = full E-PLAN 전-액션**(gate1 + T2_EPLAN=1 WALK=1 REPLAN=1): **FIND/L1/L2 discovery-enforce(HARD 주레버) + COMPUTE/GET(HARD inner) + CP5 coverage(SOFT)**, 전 write. vs **bare floor 9.3%**(기존·43태스크 nt=1·同 agent 32B·同 user-sim gpt-5.2).
- **성공기준(§5·정직)**: 라이브 pass 상승이 **HARD 실현분**(9.9% 상한 × 선택술어 정확도 × gold-free복원 89.6%)에 **부합**(초과 아님)·**over-action Δ≤0**(FIND 게이트 자기역효과 계측)·SOFT(COVERAGE) 별도 계측. **상한≠실현·per-sim 봉투(9.9% HARD)가 정직선**. gate 판정=db_match(NL-calc 이중결손 분리).
- 배치 = floor 43태스크 × nt=1 ×4 라운드 밤샘(사용자). user-sim=`openrouter/openai/gpt-5.2`·agent=로컬 32B(8140).
- **gate0 vs gate1**: L1/L2(FIND 주레버)는 **gate 1 필수**(unified() 내 발화). ∴ 순수 CP5(gate0)는 지배 HARD 레버(FIND) 제외 → **gate1 full 필수**(FIND 없이는 봉투의 대부분 누락).

## 7. 리스크·미해결
- **R1 entity 혼재**: write별 entity-type 다름 → L2(sibling)는 type 내에서만·타type coverage는 intent-count로. 초판 한계 명시.
- **R2 read-prefix 오분류**: banking 도구명 검증됨(§2)·단 신규 도구는 denylist 갱신 필요. false-write=deny 1회(안전측).
- **R3 obligation subcall 32B 품질**: 전-write 추출이 32B서 부정확하면 coverage_gap 오탐/누락. 오프라인 스모크서 재현율 측정·낮으면 결정론 ledger(qty>executed) fallback(설계 v1.3 ⑤).
- **R4 over-read 폭증**: 전-액션 L1이 매 write마다 열거 강제하면 turn 낭비 → cap(T2_EPLAN_DENY_CAP=4·기존) 유지.
- **R5 Δspurious**: 전-액션 리마인더가 over-action 유발 가능 → 계측 필수·GO 게이트.

## 8. 구현 순서 (리뷰 반영판)
1. **Ⓑ entity = per-family ABox 맵**(`entity_keys` 전역리스트 아님·write별 정확 id) + **listed/examined 단일(txn) 유지 명시** + `_covers` intent-only fallback(非txn=intent-count). build_ledger/L1L2 `_is_write` 교체.
2. ABox eplan: `entity_key_by_tool`(family→id key·[[05]] §3)·write_tools 강등·read_prefix/procedural=엔진.
3. test_eplan 무회귀(92) + Ⓑ 신규 단위(multi-id write entity·intent-only covers).
4. **Ⓐ 오프라인 R3 게이트(무료·최우선)**: 43 floor 궤적에 replan 서브콜(로컬 32B) → **의무-recall(vs gold coverage) 측정**. 임계 미달 시 결정론 fallback(qty>executed·v1.3 ⑤) 없이 **밤샘 금지**.
5. 오프라인 스모크(전-액션 executed/coverage 귀속 정확성).
6. **Ⓒ per-lever 마커 로깅 + 사후 교차표**(CP5/L1/L2 발화 sim의 pass율) — 무료 귀속.
7. 라이브 스모크(비-dispute write 발화·소액).
8. **Ⓓ floor 인터리브 재실행**(동일 gpt-5.2·라운드별 floor/treatment 교차) + 표적 nt=1×4 밤샘(승인).

## 9. ★리뷰 반영 (사용자 실행 리뷰 Ⓐ~Ⓕ·2026-07-16·전부 수용)
- **Ⓐ [BLOCKER] R3 미검증 → 무료 오프라인 게이트**: 지배 레버 실현자=32B replan 의무추출(gold-free). §5.2 스모크는 executed 귀속만 재고 R3 recall 못 잼(유료서만 드러남). **agent=로컬 32B=무료** → 43 floor 궤적서 replan 오프라인 실행·의무-recall(vs gold) 측정 = Phase-0 게이트(§8.4). 미달 시 결정론 fallback 필수·밤샘 금지.
- **Ⓑ [BLOCKER] entity under-scope**: entity_key 전역단일(_extract_entity_ids·note_write·parse_obligations). §3.2가 note_write만 바꿈=listed/examined 불일치. multi-id write(pay_credit_card: account_id vs credit_card_account_id)에서 전역 우선순위=오선택→false gap→over-action→Δspurious 트립. **수정: per-tool-family entity 맵(ABox)·listed/examined 단일(txn) 유지 명시·非txn coverage=intent-count only(정밀도 하락 정직 bound)·`_covers`(376-377) intent-only 조건부 완화**.
- **Ⓒ [SHOULD] arm 번들링 → 무료 per-lever 귀속**: full 한덩어리라 레버별 실현/역효과 못 가림. 2nd arm=유료2배 비추천 → **per-lever [T2_EPLAN] 마커 로깅 + 사후 교차표**(CP5-발화 sim pass율 vs L1 vs 무발화) 1급 산출물.
  > ⚠**2026-07-31 정정(C262)**: 이 교차표를 **효과 귀속으로 읽지 말 것** — "발화 sim pass율 vs 무발화"는 **처치-후 조건화(collider)**다. 레버는 *이미 실패 중인* 궤적에서 발화하므로 발화군의 낮은 pass는 레버의 해악이 아니라 선택 효과다(retry 실증: 발화층 p=3.5e-9 ↔ 무조건부 ITT p=0.065). **마커 로깅은 유지**(발화율·죽은 레버 탐지·단일변수 확인엔 유효), **효과 부호·크기는 arm 간 무조건부 짝비교로만**. 층 정의에 쓴 변수가 어느 arm 것인지 반드시 명시.
- **Ⓓ [SHOULD] floor 비교가능성**: floor "9.3%(stale)"는 nt=1 Bernoulli 분산 ±4~5%·1라운드 → lift 오염. **동일 gpt-5.2·라운드별 floor/treatment 인터리브 재실행**.
- **Ⓔ [NIT·이월] [[05]] field→op 부채**: 이 doc write-검출(_is_write)은 클린. 단 dag_plan/inner-router의 `_field_op`은 ABox field_ops로 이관됨(❶ 완료). COMPUTE/F3 라우팅 배선 시 dag_plan 오프라인 리터럴만 남는지 §7 이월 확인.
- **Ⓕ intent-only _covers 위험**: 非txn 단수 의무는 accumulate_qty=0 가능→coverage가 R3에 전적 의존→Ⓐ 게이트 필수성 강화.

## 10. ★★유료 실험 = full 스택 (전 이전 실험 결과·대책 포함·사용자 지시)
> 사용자: "유료 실험에서 모든 이전 실험 결과·대책 포함." BANK_TWO_TRACK §3 합성 = 전체 매커니즘. 유료 밤샘 arm = 단일 레버 아니라 **누적 스택**.

**full-stack treatment arm** (전 HARD 레버 + 선택술어 + compute):
| 대책 | 정본 | 배선 | 상태 |
|---|---|---|---|
| **FIND/discovery + COVERAGE** | 이 doc·E_PLAN_LIVE_WIRING | `T2_EPLAN=1 WALK=1 REPLAN=1`+gate1 | 이 doc 구현 |
| **reference_filter(⋈ 선택술어)** | C78·gate.json reference_filter | `--resolve 1`(t2_resolve_patch) | 배선됨·전이검증 |
| **COMPUTE 키스톤(liability·interest)** | `KEYSTONE §8`·gate.json compute_ops(2) | `T2_COMPUTE=1`(t2_gate_patch) | 오프라인 755replay 90.9%·라이브 미검 |
| F1 compliance gate | (banking confirm게이트 없음) | — | N/A |
| **F3(dispute_reason)** | C100·Track B | SFT(별개·미완) | 유료 arm서 제외(few-shot로 category만·reason=경계) |

- **full arm env**: `--gate 1 --resolve 1 --domain banking_knowledge T2_EPLAN=1 T2_EPLAN_WALK=1 T2_EPLAN_REPLAN=1 T2_COMPUTE=1`.
- **분리계측(Ⓒ)**: full-stack이 floor 대비 총 lift·마커 교차표로 레버별 기여 사후 귀속. (레버별 arm=유료 N배라 회피·마커로 무료 분해.)
- **선결(무료)**: ①compute 라이브 발화 스모크(755replay=오프라인만) ②reference_filter 선택술어 정확도(❹§7·검증됨 82%/49%) ③Ⓐ R3 recall. 셋 다 통과해야 full-stack 밤샘 의미.
- **정직 봉투**: full-stack 상한 = HARD 9.9%(FIND/COMPUTE/GET) + reference_filter 실현분 + compute(liability slice 16.7%×재현). per-sim·db_match·상한≠실현.
