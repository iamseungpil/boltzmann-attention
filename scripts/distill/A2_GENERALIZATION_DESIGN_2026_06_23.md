# A2 최소화·일반화 설계 (2026-06-23·사용자 지시: A2도 도메인특화 노골화 금지·규칙 일반화·step-by-step·검정가능)

> 진입: [05-fixed-vs-variable] + [FLOW_DISCIPLINE_SCAFFOLD_DESIGN](FLOW_DISCIPLINE_SCAFFOLD_DESIGN_2026_06_22.md). 동기: 현 `gate.json`이 retail tool-name 리스트(applies_to/satisfiers/resolver_path/status)를 *노골적으로* 박음 = A2지만 도메인특화 과다. **규칙을 도구-속성 술어로 일반화 → A2를 도출불가 최소치로.**

## 0. 원칙
- **규칙(엔진·고정·도메인-일반) = 도구 *속성*에 대한 술어.** 도구 속성 = 이름규약 + openai_schema(인자) + (가능시)반환타입.
- **A2(가변) = 도출 불가한 *최소* 사실만.** tool-name 하드리스트 금지(도출로 대체).
- **검정가능**: 모든 도출은 *현 hand-list 재현*으로 검증(회귀) + 동작불변(--validate·census) + `grep tool-name in gate.json → ~0` + 전이(airline/banking이 신규 하드리스트 0).

## 1. 도출 기반 (★실증 완료·2026-06-23·retail+airline 정확 일치)
`/tmp/derive2.sh`: 명명규약+스키마 분류기 출력 = 현 gate.json applies_to와 **양 도메인 100% 일치**.
- **read** = name starts {get_,find_,list_,search_,calculate}.
- **write** = name starts {modify_,cancel_,exchange_,return_,book_,update_,send_,change_,open_,close_,submit_,apply_,…} ∧ ¬read.
- **auth/identity-producer** = 반환/이름이 user 식별자 생산 (retail `find_user_id_*`; 도출=name `find_user_id*` 또는 반환타입 user_id). airline=∅(user-provided).
- **user-scoped** = 인자에 {user_id, order_id, reservation_id, account_id, …owner-id} 보유.
- **handoff** = name contains `transfer_to_human`.
- **precondition status** = write tool 이름에 인코딩된 상태어 (modify_**pending**·exchange_**delivered**·return_**delivered**·cancel_**pending**) → 정규식 파싱.
- **owner resolver** = owner-id 인자 → 동명 entity-getter (order_id→get_order_details·reservation_id→get_reservation_details) → `.user_id`. (규약: get_<entity>_details(<entity>_id)→user_id.)

검증값(실측): retail WRITE=7·USER_SCOPED=9·AUTH=2 = hand-list 동일 / airline WRITE=6·USER_SCOPED=8 = 동일.

## 2. 게이트별 일반화 (규칙 | 도출원 | 잔여 A2)
| 게이트 | 일반 규칙 | 도출원(엔진) | 잔여 A2(최소) |
|---|---|---|---|
| **G1 auth** | user-scoped 행동 전, *식별 identity가 grounded*여야 | user-scoped=인자스키마 · grounded=provenance(도구출력 OR 사용자발화) | **∅** (retail lookup·airline user-provided 둘 다 provenance로 통합) |
| **G2 confirm** | write 전, 최신 사용자 턴이 확인(yes) | write=명명규약 · confirm=CONFIRM_RE | ∅ |
| **G3 ownership** | write 대상이 인증 user 소유 | owner-id 인자→entity-getter 규약→user_id | owner_field 명(기본 `user_id`=scaffold DEFAULT) |
| **G4 notice** | handoff 전, 고정 안내문구 송신 | handoff=명명규약 | **notice_text 문자열**(정책 명령·도출불가=유일 진짜 A2) |
| **G5 precond** | write 대상 상태가 그 write 허용 | 허용상태=write 이름서 파싱·현재상태=entity-getter | ∅ (이름에 없으면 G5 미적용=airline no-op 자동) |

⇒ **이상적 gate.json = 거의 비어있음**: G4 notice_text + (예외시) override 뿐. applies_to/satisfiers/resolver_path/status-list 전부 삭제(도출).

## 3. Step-by-step (각 단계 검정 동반·작은 단위)
**S1. 도구 분류기 `tool_roles.py`** (엔진·도메인0): `roles(tools) -> {write,user_scoped,auth,handoff, precond_status:{tool:allowed_status}, owner_path:{tool:[id_arg,getter,field]}}`. 명명규약+스키마+entity-getter 규약.
  - ✅검정 S1: `roles(retail)` ⊇⊆ 현 retail.gate.json applies_to/satisfiers·`roles(airline)` 동일 = 단위테스트 `test_roles_match_handlists.py`(assert 집합일치). (실증 완료→코드화.)

**S2. GateInterpreter가 roles 소비** (가변=A2 아닌 도출): gate 정의가 tool-name 리스트 대신 *kind만* 참조, applies_to/satisfiers/resolver_path는 `roles()`서 주입.
  - ✅검정 S2: `--validate` retail/airline PassA/B=0(무회귀)·retail census(elig/loop) 불변·로컬 스모크 동일.

**S3. gate.json 축소**: applies_to/satisfiers/resolver_path/steer-status 삭제 → notice_text(+필요시 override)만 잔존.
  - ✅검정 S3: `grep -E "modify_|exchange_|get_.*_details|find_user|order_id|pending|delivered" a2/*.gate.json` → 0(또는 override만)·동작불변.

**S4. G1 auth 일반화(=이전 위반 #2 해소)**: auth-establishment를 *grounded-identity*로 — lookup-도구 출력 OR 사용자발화서 user_id가 provenance-grounded되면 인증. (retail+airline 통합·엔진서 `satisfied_by` 분기 제거.)
  - ✅검정 S4: airline 런타임 G1 실제 작동(user-provided-id 시드 후 user-scoped 허용·미인증시 deny)·retail 무회귀.

**S5. G5 status 도출 + 일반 steer**: 허용상태=이름파싱·steer=일반템플릿("상태 X는 {tool} 불가; 상태에 맞는 도구 사용"). 이름에 상태어 없는 write=G5 미적용(airline 자동 no-op).
  - ✅검정 S5: retail census(elig 25→0 @32B) 재현·"pending (item modified)"→_acted 재현·airline은 G5 deny 0(무영향).

**S6. 전이 실증(무신규-하드리스트)**: airline/banking을 *동일 엔진* + (거의 빈) gate.json으로 — 신규 tool-name 하드리스트 **0**개로 G1-G4 작동.
  - ✅검정 S6: `grep tool-name in airline.gate.json/banking.gate.json` ≈ 0·airline G1-G4 deny 동작·banking 동작.

## 4. 검정 하네스 (한 곳)
`test_a2_general.py`: ①roles==handlist(retail+airline) ②grep tool-name in gate.json 카운트 ③--validate 양도메인 PassA/B=0 ④retail census 회귀(elig/loop 동일) ⑤airline G1 런타임 auth. CI식 1커맨드.

## 5. [05] 정합
- 이건 [05] *강화*: A2 도메인특화 *순감*(하드리스트→도출)·엔진은 여전히 일반(도출규약=도메인무관)·전이=빈 A2-swap에 근접. 위반 #2(auth-bake) S4서 해소.
- 잔여 진짜 A2 = notice_text(정책 문자열) 1개 + 예외 override. = 최소.
- ⚠️ 도출=규약 의존(명명·entity-getter). 규약 깨지는 도메인이면 override(A2)로 명시 — 단 override는 *예외*고 grep로 가시화(노골 하드리스트화 방지). 규약 자체는 4벤치(retail/airline/banking/telecom) 명명서 검증.

## 6. 진행
S1부터(분류기+단위테스트=실증 코드화). 이후 S2-S6 순차·각 검정 통과 후 다음. 권위=이 문서·[05]·[FLOW_DISCIPLINE_SCAFFOLD_DESIGN].
