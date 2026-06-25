# Constraint-Gate 설계 (2026-06-25) — operation-semantic 정책위반의 결정론 게이트화

> 근거 = `NESTED_ARM_FAILURE_CENSUS_2026_06_25.md` §2A: 지배 실패 = operation-semantic 정책위반(bizrule ~132). 전부 (proposed args + DB state)로 *decidable* → 결정론 게이트가 *사후 env 거부+모델 루프*를 *사전 steer*로 전환. [[05]] 준수(엔진=일반 연산·A2=retail 규칙)·[[03]] 설계먼저·flag-gated 측정 arm.

## ★측정-전 deprioritize (2026-06-25 정적 gold-read 회계·결정 (b)) — [[03]]/[[06]] 실증
GPU 측정 *전* 정적 gold-read로 pass-lever 천장 확정 → **count-match·payment 빌드 보류·new≠old만 동승**(`ASSEMBLED_STACK_CENSUS_DESIGN` §5).

| 규칙 | err | uniq task | LEVER(gold=유효행동) | hygiene | LEVER∧현재실패(진짜 flip후보) |
|---|--:|--:|--:|--:|---|
| count-match | 69 | 14 | 4 | **10** | t36/37/63+t56부분 ≈3-4·**단 operand로 더 깎임**(count만 강제·어느item 못강제) |
| new≠old(disjoint) | 11 | 4 | 3 | 1 | t20/36/63 ≈3 (36·63 count-match와 중복) |
| payment-original | 46 | 11 | 9 | 2 | t13/53/82/100 ≈4 (단 t11/14/51 *이미 pass*=에러후 복구) |

- **"126 bizrule 에러" → pass-flip 천장 ≈8 고유 task**(union·114중 ~7pp·operand 장벽으로 더 깎임). present+g15(+12.3pp)보다 작은 레버.
- count-match=**순수 hygiene**(refuse-gold 10·loop-death 0). payment=marginal~4(다수 이미 복구·[[05]] 파생필드 _orig_payment/_gift_cards 위험·policy.md 미확인). → **둘 다 빌드 보류·정적 천장만 기록.** new≠old(3·깨끗·false-block0)만 census arm 동승.
- 이 정적 deprioritize 자체가 기여(측정 전 능력→레버 배정·GPU 절약).

## 0. 무엇을 게이트하나 (census → 스코프)
| census 규칙 | n | decidable 출처 | 게이트화 | 비고 |
|---|---|---|---|---|
| count-match (\|new\|==\|item\|) | 69 | **순수 args** | ✅ op=`equal_len` | "remove 연산 없음" 교시 |
| new≠old (변형 바꿔야) | 11 | **순수 args** | ✅ op=`disjoint` | |
| payment-original | 46 | args + order read | ✅ op=`member_of` | allowed-set 도출 정책 확인 선결(§5) |
| insufficient-balance | 5 | 가격 산술 필요 | ❌ 제외 | env 몫(가격 replicate 안 함·저가치) |
| non-pending | 1 | status | ➖ G5 기존 | 신규 불요 |

→ **게이트 대상 = 3규칙(count-match·new≠old·payment-original)·~126 에러.** 순수-args 2개(80건)는 무조건 GO·payment는 allowed-set 확인 후.

## 1. 엔진 확장 (도메인-일반·`gate_interpreter.py`) — 새 kind `constraints`
- `_KIND_PRIORITY`에 `"constraints": 4.5` 추가 (preconditions 후·select_confirm 전 — write 직전 마지막 정합성).
- `check()`에 분기 추가. **연산 vocabulary(엔진 고정·도메인 무관)**:
  - **`equal_len`**: `fields=[a,b]` → `len(args[a]) != len(args[b])` 면 deny. (args 둘 다 있을 때만·없으면 skip=false-block 회피)
  - **`disjoint`**: `fields=[a,b]` → `set(args[a]) & set(args[b])` 비지 않으면 deny.
  - **`member_of`**: `arg=X`, `allowed_from=[resolver_path,...]` → resolver로 allowed-set 도출, `args[X] not in allowed` 면 deny. (resolve 실패=skip)
- 공통 규율(preconditions와 동일): **read-only resolver만**(replay-safe)·**resolve 못 하면 deny 안 함**(false-block 회피·리뷰#2/R4)·deny 메시지=`A2 steer 텍스트`(엔진 하드코딩 0).

```python
elif kind == "constraints":
    for chk in (g.get("checks") or []):
        if tool_name not in (chk.get("applies_to") or []): continue
        op = chk.get("op")
        if op == "equal_len":
            a,b = chk["fields"]
            if a in args and b in args and args[a] is not None and args[b] is not None \
               and len(args[a]) != len(args[b]):
                return False, g["id"], chk.get("steer")
        elif op == "disjoint":
            a,b = chk["fields"]
            if a in args and b in args and set(map(str,args[a])) & set(map(str,args[b])):
                return False, g["id"], chk.get("steer")
        elif op == "member_of":
            fn = self.resolvers.get("resolve_field")
            allowed = _resolve_set(fn, chk.get("allowed_from"), args)  # 도메인-일반 set 도출
            v = args.get(chk["arg"])
            if allowed is not None and v is not None and str(v) not in allowed:
                return False, g["id"], chk.get("steer")
```
- `_resolve_set`: allowed_from = resolver_path 목록(예: [order→payment_history→method_ids] ∪ [user→gift_cards]) → 합집합 set. 도메인-일반(경로=A2).

## 2. A2 (`retail.gate.json`) — retail 규칙 인스턴스 (도메인 사실)
```json
{ "id": "G7_OP_CONSTRAINTS", "kind": "constraints",
  "applies_to": ["modify_pending_order_items","exchange_delivered_order_items","modify_pending_order_payment"],
  "checks": [
    { "op": "equal_len", "fields": ["item_ids","new_item_ids"],
      "applies_to": ["modify_pending_order_items","exchange_delivered_order_items"],
      "steer": "[constraint] This tool EXCHANGES items 1-for-1: new_item_ids must have the SAME count as item_ids. There is NO operation to simply remove an item. If the customer wants to drop an item without replacement, that is not possible here — explain this; do NOT retry with an empty/short new_item_ids." },
    { "op": "disjoint", "fields": ["item_ids","new_item_ids"],
      "applies_to": ["modify_pending_order_items","exchange_delivered_order_items"],
      "steer": "[constraint] Each new_item_id must be a DIFFERENT variant than the item being exchanged; you reused an id from item_ids. Pick a different variant (e.g. a different size/color) or do not exchange that item." },
    { "op": "member_of", "arg": "payment_method_id",
      "applies_to": ["modify_pending_order_payment","exchange_delivered_order_items","return_delivered_order_items"],
      "allowed_from": [["order_id","get_order_details","_orig_payment"], ["user_id","get_user_details","_gift_cards"]],
      "steer": "[constraint] Refunds/payment changes must use the order's ORIGINAL payment method or a gift card. '<value>' is neither. Use the original payment method shown in the order." }
  ] }
```
- ⚠️ payment `allowed_from`의 `_orig_payment`/`_gift_cards`는 record서 도출하는 *파생 필드* → resolver `fetch_record` + 소량 추출 헬퍼 필요(payment_history[0].payment_method_id·gift_card_* 키). 도메인-일반화: A2가 "어느 record의 어느 path"를 지정, 엔진이 generic 추출.

## 3. [[05]] 준수 논증
- 엔진 = `equal_len`/`disjoint`/`member_of` = **완전 도메인-일반 연산**(airline swap = 같은 연산·다른 인스턴스). retail 도구명/필드/steer = 전부 A2. grep retail in engine = 0 (목표).
- **모델 판단 동결 아님**: 게이트는 *무효 op 차단 + 정책 교시*(steer)만. 어느 item/variant/op인지 *선택*은 모델. = 절차-제약 offload(scaffold 역할·thesis §2).
- 게이트 증식 우려([[06]]): ONE 새 kind(일반)·measured·flag-gated. G5 family 자연 확장(status-precond → arg/state-constraint).

## 4. 측정 (결정론·[[08]]·pass^1 금지)
- arm: `T2_GATE_KINDS=auth,confirm,ownership,notice,preconditions,constraints` (+present+nested 유지) vs **baseline = present+nest+g15**(`*_presentnest_g15_retail_t3`). 32B+14B.
- **결정론 지표**:
  1. **bizrule 에러 ↓** (count-match 69·payment 46·new≠old 11 → 게이트가 사전차단하면 env-error 소멸).
  2. **too_many_errors ↓** (무효-op 루프 차단).
  3. **escape_det_census --clean**: write-layer(L0-L3)·MATCH·pass^k 변화. **핵심 = bizrule이 pass로 전환되나** vs *단지 다른 층으로 이동/refuse-전환*.
  4. **false-block 체크**: 정당한 write를 게이트가 잘못 막았나(over-deny) — gold-write가 deny된 케이스 census.
- 궤적 정독: count-match deny 후 모델이 (a)올바른 교환 (b)올바른 refuse/redirect (c)여전히 루프 중 무엇인지.

## 5. 열린 결정 / 선결 (구현 전)
- **D-payment(선결)**: payment-original allowed-set = {원결제수단}만인가 {원결제 ∪ gift_card}인가 — retail `policy.md` 확인 필요(env 메시지만으론 불명). 확인 전 payment 규칙 보류·순수-args 2개(count-match·new≠old·80건) 먼저 구현/측정 가능.
- **D-scope**: 3규칙 동시 vs 순수-args 먼저(저위험·고확실) 단계적. **권장 = 순수-args 2개 먼저**(count-match·new≠old·decidable 확실·false-block 위험0) → 측정 → payment 추가.
- **D-steer 효과 가설**: count-match steer가 "remove 불가"를 교시하면 → 모델이 (gold가 그렇듯) *올바른 redirect/refuse*로 전환할 것. 단 NO-GO: steer 알아도 task 자체가 불가능(gold도 거부)이면 pass 전환 0일 수 있음(=게이트는 루프만 차단·pass 무관) → 그래도 too_many↓·정직 행동↑은 가치.

## 6. NO-GO
- (a) false-block이 양성 write를 막으면(over-deny>이득) → 롤백.
- (b) bizrule 차단해도 pass 전환 0이고 단지 refuse로 이동하면 → 게이트는 "루프 위생"일 뿐(pass 레버 아님)·정직 기록(과대평가 금지·[[06]] lever-type≠resolution).
