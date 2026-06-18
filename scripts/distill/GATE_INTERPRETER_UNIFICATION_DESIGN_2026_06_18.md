# GateInterpreter 통일 설계 (2026-06-18) — scaffold 벤치-일반화 (조건 ③ keystone)

> 목표 = 벤치별 게이트(`tau2/t2_gate.py RetailGate` + SOPBench gate)를 **하나의 `GateInterpreter`**로 통일. 도구멤버십·정책은 `gate_spec`(ABox)서 *읽고*, 집행 로직은 고정. **per-bench 분기 0**(grep `if bench`/`if domain` = 0). = `FIXED_VS_VARIABLE.md` 조건 ③·`EXPERIMENT_DESIGN §4`·`ISOLATION_EXPERIMENTS §S`.
> 근거: `t2_gate.py`는 **이미 90% spec-driven**(`GATE_SPEC` dict = ABox A2 실물·`render_recovery`가 spec서 메시지 생성·도메인 문자열 0). 위반 = `AUTH_TOOLS`/`WRITE_TOOLS`가 코드 전역에 샘 + `RetailGate.check`가 그 전역 직접 참조 + 클래스명 "Retail". ⇒ 통일 = 리팩터(발명 아님).

## 1. ★핵심 = 유한 gate-KIND 어휘 (layer-B closure·일반화의 열쇠)
게이트는 무한 정책이 아니라 **유한 종류(kind)**로 닫힘(`PRIMITIVE_COVERAGE_MATRIX` 층 B·P5/P6/P8·census P10=0). 각 kind = 1 해석기 코드경로(도메인-일반). 정책 차이는 전부 *데이터*(어느 도구·어느 사전조건).

| kind | 정체(primitive) | 충족 조건(state 대비) | tau2 | SOPBench |
|---|---|---|---|---|
| **auth** | 인증 확립(P8) | satisfier 도구 성공 관측됨 | G1_AUTH | login |
| **confirm** | 쓰기-전 확인(P6) | 직전 user 턴 = 명시 yes | G2_CONFIRM | — |
| **ownership** | 대상이 auth user 소유(P5) | resolver-path(target→owner) == auth_user | G3_SINGLE | — |
| **notice** | 의무 고지 송신(P5 obligation) | 고정문구가 assistant 발화로 송신됨 | G4_TRANSFER | — |
| **preconditions** | goal 사전조건 전부 확립(P5 sequence) | required_set ⊆ established_set | — | dirgraph/gather-first |

**= 5 kind가 transactional 게이트 closure.** 새 벤치가 새 kind 요구 = 해석기 +1 dispatch(그 kind 쓰는 *전* 벤치로 일반화·여전히 유한·census가 반증탐색).

## 2. gate_spec (ABox A2) 스키마 — 데이터
```jsonc
// retail_gate_spec  (airline_gate_spec / sop_<domain>_gate_spec = 같은 스키마·값만 다름)
[ { "id": "G1_AUTH", "kind": "auth",
    "predicate": "authenticated user identity",
    "applies_to": ["get_user_details","get_order_details", ...WRITE...],   // ← 도구멤버십=데이터
    "satisfiers": {"find_user_id_by_email": ["email"],
                   "find_user_id_by_name_zip": ["first_name","last_name","zip"]} },
  { "id": "G2_CONFIRM", "kind": "confirm",
    "applies_to": ["cancel_pending_order","exchange_delivered_order_items", ...] },
  { "id": "G3_OWNERSHIP", "kind": "ownership",
    "applies_to": [...USER_SCOPED...],
    "resolver_path": ["order_id","get_order_details","user_id"] },   // target→owner 도출 경로(선언적)
  { "id": "G4_NOTICE", "kind": "notice",
    "applies_to": ["transfer_to_human_agents"],
    "notice_text": "YOU ARE BEING TRANSFERRED ..." } ]
```
- 현 `GATE_SPEC` dict가 이미 이 구조(predicate·satisfiers·applies_to·ask/terminal) — **`kind` 필드 추가 + `applies_to`에 도구셋 인라인**(현재 module global을 데이터로 올림)만 하면 됨.

## 3. GateInterpreter (고정 코드·per-bench 분기 0)
```python
class GateInterpreter:                       # FIXED·벤치 무관·절대 미수정
    def __init__(self, gate_spec, resolvers=None):
        self.spec = gate_spec                 # ABox (swap)
        self.state = GateState()              # auth_user·confirmed·notice_sent·established_set
        self.resolvers = resolvers or {}      # ownership/preconditions용 결정론 lookup(엔진 제공)

    def check(self, call, ctx):               # 실행 전
        for g in self.spec:
            if call.name in g["applies_to"]:
                ok = self._satisfied(g, call, ctx)
                if not ok:
                    return Deny(g["id"], render_recovery(g))    # render_recovery=기존·일반
        return Allow()

    def _satisfied(self, g, call, ctx):       # ★dispatch on KIND (유한·도메인일반·if bench 0)
        k = g["kind"]
        if k == "auth":          return self.state.authed
        if k == "confirm":       return _yes_in_last_user(ctx)
        if k == "ownership":     return _resolve_path(g["resolver_path"], call, self.resolvers) == self.state.auth_user
        if k == "notice":        return self.state.notice_sent
        if k == "preconditions": return set(self._required(g, call, ctx)) <= self.state.established
        raise UnknownGateKind(k)              # 새 kind = +1 경로(유한)

    def observe(self, call, result, ok=True): # 실행 후 state 갱신 (kind별)
        for g in self.spec:
            if call.name in g.get("satisfiers", {}) and ok:
                self.state.mark(g["kind"], call, result)
```
**분기 = `g["kind"]`(5종·도메인일반)뿐. `if domain`/`if bench` = 0.**

## 4. Migration (구체)
1. **RetailGate → GateInterpreter(retail_gate_spec)**: `AUTH_TOOLS`/`WRITE_TOOLS`/`USER_SCOPED` module global → `applies_to` 데이터로 이동(이미 거기 참조). G1-G4 → kind=auth/confirm/ownership/notice. db(G3) → ownership `resolver_path`(order_id→get_order_details→user_id)·`resolvers`에 결정론 lookup. `t2_gate_patch.py`의 `RetailGate(db=...)` → `GateInterpreter(load_gate_spec("retail"), resolvers)`.
2. **airline_gate_spec** = 같은 스키마·값만(update_reservation_* = write·cabin attr). **무수정 GateInterpreter로 작동 = 통일 검증.**
3. **SOPBench gate → GateInterpreter(sop_<domain>_gate_spec)**: kind=**preconditions**(required_set = goal operator precond from dirgraph·`_required`가 resolvers["dirgraph"]서 읽음) + auth(login = 평범한 satisfier). dirgraph는 *모델 출력*(ABox 아님)·gate_spec은 "precond source=dirgraph"만 명시.
4. `render_recovery`(기존·일반)·`GateState` 신규(작음). 폐기 = `RetailGate` 클래스·module global 도구셋.

## 5. 검증 (조건 ③ = 통일 실증)
- **grep `if bench`/`if domain`/`if.*retail`/`if.*airline` in GateInterpreter = 0** (자동 체크·CI).
- **동일 GateInterpreter unchanged**로 retail+airline+SOP 작동(gate_spec만 swap·재학습0·코드수정0).
- **ABox-ablation**: 빈 gate_spec → 게이트 0(전부 allow)·틀린 gate_spec(applies_to 오타) → 붕괴 = spec 실사용 증명.
- **"그냥 결정론 프로그램" 방어**(마스터 §1): 해석기는 *정책을 모름*(gate_spec 데이터가 정책)·procedure는 모델(TBox)·해석기는 fact-gate 집행만 = fact-offload(OK)·procedure-offload 아님.

## 6. 비용 (실무·작음)
- 통일 = **리팩터**(구조 이미 spec-driven). 작업 = ①도구셋 데이터화 ②kind-dispatch 추출 ③SOPBench gate를 preconditions-kind로 표현(주 작업·dirgraph→required_set 매핑). 발명 아님.
- 재발 비용 = **gate_spec 컴파일(A2)뿐**(airline/telecom 이미 Fable-5 0줄·F1 장부). GateInterpreter는 데이터만 읽으므로 per-bench 코드 = 0.

## 7. 정직 (경계)
- 유한 gate-kind closure가 bound — P10 census=0이나 **적대탐색 지속**(out-of-genre 게이트 사냥). 새 kind=+1(유한·일반화).
- **ownership resolver_path**가 가장 도메인-가까움 → *선언적 path*(데이터)로 유지·손코딩 분기 금지(path를 gate_spec에).
- **preconditions kind**는 dirgraph(모델 출력)에 의존 → gate_spec은 "source=dirgraph"만·dirgraph 자체는 TBox 출력(ABox 아님).
- 통일 *후* facet TBox 전이(조건 ②) 재측정 = 이번엔 "고정 scaffold + ABox swap"이 진짜 성립한 채로.
