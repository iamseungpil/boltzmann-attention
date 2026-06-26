# A2-구동 scaffold 키스톤 — 구현 design-delta (2026-06-21)

> **선결 keystone** (HANDOFF_2026_06_21 §2(1)·§3.9·[[05-fixed-vs-variable]]). 전이/일반화 측정의 *binding* 전제. 이거 전엔 C10·전이 무의미.
> 게이트 코어 authority = `GATE_INTERPRETER_UNIFICATION_DESIGN_2026_06_18`. 본 문서 = 그 설계를 **provenance-side(autofetch·arg-types·placeholders·GATE_DOMAINS)까지 확장**한 구현 delta.

## 0. 목표 (검증가능 종료조건)
런타임 scaffold(`t2_gate.py`·`t2_gate_patch.py`)서 **도메인-하드코딩 0**:
- `grep -nE "if .*domain|if .*bench|RetailGate|get_user_details|\"retail\"|'retail'" <엔진코드>` = **0** (엔진 한정·분석도구/주석 제외).
- 도구셋(AUTH/WRITE/USER_SCOPED)·producer(autofetch)·식별arg-types·placeholder·GATE_DOMAINS = 전부 **A2 파일서 로드**.
- **airline = A2-swap만으로** 무수정 작동(`--domain airline --gate 1`이 airline gate_spec 로드·엔진 코드 동일).

## 1. A2 통합 파일 = `a2/<domain>.gate.json` (변경부=데이터)
기존 `specs/<domain>_gate_spec_fable5.json`(게이트만) → scaffold-config까지 통합한 단일 A2:
```jsonc
{
  "gates": [                                  // = 기존 gate_spec + kind 필드 추가
    {"id":"G1_AUTH_FIRST","kind":"auth",
     "predicate":"authenticated user identity",
     "satisfiers":{"find_user_id_by_email":["email"],
                   "find_user_id_by_name_zip":["first_name","last_name","zip"]},
     "applies_to":[...USER_SCOPED...]},
    {"id":"G2_CONFIRM_WRITE","kind":"confirm","applies_to":[...WRITE...],"ask":"..."},
    {"id":"G3_SINGLE_USER","kind":"ownership","applies_to":[...USER_SCOPED...],
     "resolver_path":["order_id","get_order_details","user_id"],  // target_arg→producer→owner_field (선언적)
     "terminal":"..."},
    {"id":"G4_TRANSFER_MSG","kind":"notice","applies_to":["transfer_to_human_agents"],
     "notice_text":"YOU ARE BEING TRANSFERRED ..."}
  ],
  "producers": {                              // autofetch: 의미역할→getter (T2_AUTOFETCH)
    "authenticated_user_record": {"tool":"get_user_details","args_from":{"user_id":"@auth_user"}}
  },
  "identifying_arg_types": ["email","name","zip","user_id","order_id","username","id",
                            "payment","address","phone","item"],   // = PROV_ARG_HINT
  "placeholders": ["#W0000000","johndoe@example.com", ...]         // = COMMON_PLACEHOLDERS (도메인-일반 기본 + 스키마-유래는 동적 유지)
}
```
- `@auth_user` = 엔진이 게이트 state서 채우는 placeholder(autofetch 인자). 도구명 외 하드코딩 없음.
- airline `producers.authenticated_user_record.tool` = `get_user_details`(동일)·resolver_path = reservation_id→get_reservation_details→user_id. **값만 다름.**
- `identifying_arg_types`/`placeholders`는 도메인-일반 기본을 공유하되 A2서 override 가능(=A2-swap 표면).

## 2. 엔진 = `gate_interpreter.py` (신규·FIXED·절대 도메인분기 0)
06-18 §3 그대로 + loader:
```python
def load_domain_a2(domain) -> dict        # a2/<domain>.gate.json (없으면 None=게이트 비활성)
class GateInterpreter:
    def __init__(self, gates, resolvers=None)   # gates=A2 list, resolvers=엔진제공 결정론 lookup
    def check(self, name, args, ctx) -> (ok, gate_id, reason)   # kind-dispatch (auth/confirm/ownership/notice)
    def observe(self, name, args, result, ok)                   # satisfier 성공→state.mark
```
- **kind dispatch만**(auth/confirm/ownership/notice; preconditions=SOP는 후속). `if domain` 0.
- ownership = `resolvers["owner_of"](resolver_path, args)` 콜백(엔진이 주입·도구 실행은 patch층서). 인터프리터는 path를 *모름*=데이터.
- `render_recovery`(기존·일반)·`AUTH_TOOLS`/`TRANSFER_MSG` 등 module-global 폐기 → A2 유래.

## 3. Migration (구체·blast-radius 통제)
1. **`gate_interpreter.py` 신규**: GateInterpreter + load_domain_a2 + render_recovery 이식.
2. **`t2_gate.py`**: GATE_SPEC(retail) → `specs`서 로드하거나 A2로 이전. `RetailGate`=얇은 **deprecated 호환 alias**(=`GateInterpreter(load_domain_a2("retail").gates, retail_resolvers)`)로 유지 → 분석도구(`t2_compliance`·`t2_passk_autopsy`·`tau2_primitive_census`·`t2_gate_r2_verdict`) 무수정. `validate()` 유지(retail GT replay). `AUTH_TOOLS` = A2 G1 satisfiers 키서 도출(호환 export).
3. **`t2_gate_patch.py`**: ① `GATE_DOMAINS = {d for d if load_domain_a2(d)}`(=A2 존재=활성). ② gate = `GateInterpreter(load_domain_a2(domain).gates, resolvers_from_env(env))`(RetailGate 호출 제거). ③ `_autofetch_text`: A2 `producers.authenticated_user_record`서 도구·인자 도출(get_user_details 하드코딩 제거). ④ `PROV_ARG_HINT`→A2 `identifying_arg_types`. ⑤ `COMMON_PLACEHOLDERS`→A2 `placeholders`(+ 스키마유래 동적 유지). **public API(`apply`/`apply_provenance_regen`) 시그니처 불변.**
4. ownership resolver: `resolvers_from_env(env)` = resolver_path[target_arg,producer,owner_field]를 받아 `env` getter로 owner 도출(도메인-일반·env.tools 반영적 호출 또는 db 접근). retail order·airline reservation 동일 코드.

## 4. 검증 (종료 게이트·CI)
1. `grep -nE "if .*domain|RetailGate|get_user_details|[\"']retail[\"']|[\"']airline[\"']" gate_interpreter.py t2_gate_patch.py`(런타임 분기·도구명) = **0** (import/주석/호환 alias 정의 제외).
2. `python t2_gate.py --validate --domain retail` = G1/G3 over-deny 0 (기존 GT replay 회귀 무변).
3. **airline A2-swap**: `load_domain_a2("airline")` 로드 + GateInterpreter unchanged로 airline 게이트 작동(드라이런: airline gold actions replay서 G1 over-deny 0). 코드수정 0.
4. ABox-ablation: 빈 gates → 전부 allow / 오타 applies_to → 붕괴 = spec 실사용 증명.
5. retail e2e 회귀(작게): `--gate 1 --domain retail` 한 배치가 리팩터 전 pass와 동등(±노이즈).

## 5. 정직·경계
- airline G4/G5/G6 = `db_check` DSL(서술)·본 키스톤서 **auth/confirm/ownership/notice 4 kind만** 엔진화. db_check kind(preconditions/eligibility)=후속(airline 완전화 시)·SOP preconditions와 함께. **이번 종료조건 = retail 4게이트 + airline auth/confirm/ownership swap-작동.**
- producer-map(autofetch)은 도구명 1개를 A2로 올림 = 진짜 A2-swap 표면(airline=동일 get_user_details). identifying_arg_types는 도메인-일반(스키마서 더 도출 가능·후속).
- 이 리팩터 *후* facet TBox 전이(조건②)·C10 = "고정 scaffold + A2-swap" 진짜 성립한 채 측정.
