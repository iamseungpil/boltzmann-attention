#!/usr/bin/env python
"""D5: fetch-우선 대조쌍 SFT 빌드 (R1b L3 복구순서, 2026-06-14).

근본원인(이전 세션 부검): v4 ask-user augmentation(`fc_askuser_augment.py`)이
  키워드 휴리스틱(USER_PROVIDED_HINT: dob/birthday/income/amount/member...)으로 *무차별 ask* →
  카탈로그에 getter가 있어 *fetch 해야 하는 슬롯까지 ask*를 학습 → τ²서 "fetch 우선이어야 하는데 ask".
  = always-ask 붕괴(R1b §3a D5: "없으면 always-ask/always-fetch 붕괴").

처방(D5 대조쌍): ask/fetch 분기를 *카탈로그서 결정론 도출*(휴리스틱 금지).
  - **fetch 분기**: 값이 getter 출력서 옴(=tool-output provenance) 또는 카탈로그에 그 슬롯 getter 有 → fetch-then-use.
  - **ask 분기**: 값이 user 발화서만 옴(=user provenance) AND 카탈로그에 그 슬롯 getter 無 → ask-then-use.
  분기 기준 = **provenance(정확 신호) + getter_map(카탈로그 가용성)** 둘 다 결정론.

대조 학습: fetch-then-use(sop_rand 원본에 자연 존재) + 게이트된 ask-then-use(이 스크립트) 를 함께 학습 →
  모델이 "tools= 카탈로그 보고 분기"(getter 있으면 fetch / 없으면 ask) 학습.

이 스크립트는 **게이트된 ask 예시**(sop_rand_aug 대체)와 **대조 census**를 산출.
  fetch/upfront 예시 = sop_rand 원본 자체(이미 자연 fetch-then-use 포함) → sft 믹스서 그대로 사용.

Usage:
  fc_d5_contrastive.py --in sop_rand.jsonl --getter_map SOPBench/induced/getter_map.json \
      --out_ask sop_d5_ask.jsonl [--frac 1.0] [--seed 42] [--sample N]
"""
import argparse, json, random, re
from collections import Counter

# getter 접두어 (precheck_getter_groundability.is_getter / autoderive is_obs_tool 기준)
_GETTER_PREFIXES = ("internal_get_", "get_", "view_", "internal_")
# 구 키워드 휴리스틱 (fc_askuser_augment.USER_PROVIDED_HINT) — 비교/회귀용
_HEUR = ("username", "identification", "password", "email", "first_name", "last_name",
         "zip", "name", "phone", "dob", "birthday", "income", "asset", "license", "ssn",
         "amount", "id_number", "member")


def _tok(s):
    return set(t for t in re.split(r"[_\W]+", str(s).lower()) if t)


def produced_slot(getter_name):
    """getter 도구명 -> 생산 슬롯 토큰. e.g. internal_get_credit_score -> credit_score."""
    n = getter_name
    for p in _GETTER_PREFIXES:
        if n.startswith(p):
            return n[len(p):]
    return n


def load_getters(getter_map_path):
    """getter_map.json -> 도메인별 (getter 도구명 set, 생산-슬롯 토큰 set)."""
    gm = json.load(open(getter_map_path, encoding="utf-8"))
    names_by_dom, slots_by_dom = {}, {}
    for dom, cond_map in gm.items():
        names = set()
        for getter_list in cond_map.values():
            names.update(getter_list)
        names_by_dom[dom] = names
        slots_by_dom[dom] = {produced_slot(g) for g in names}
    return names_by_dom, slots_by_dom


def key_is_fetchable(arg_key, prod_slots):
    """arg_key 가 어떤 getter 생산-슬롯과 강하게 매치되나 = 카탈로그서 fetch 가능.
    매치 = arg_key 토큰집합 ⊆ 슬롯 토큰집합 (약한 공유토큰 'user' 등 오탐 방지)."""
    akt = _tok(arg_key)
    if not akt:
        return False
    return any(akt <= _tok(slot) for slot in prod_slots)


def first_call(messages):
    """첫 assistant tool_call (인덱스, 인자dict)."""
    for i, m in enumerate(messages):
        if m.get("role") == "assistant" and m.get("tool_calls"):
            try:
                args = json.loads(m["tool_calls"][0]["function"]["arguments"])
            except Exception:
                return i, None
            return i, (args if isinstance(args, dict) else None)
    return None, None


def prior_text(messages, upto, role):
    return " \n ".join(str(m.get("content") or "") for m in messages[:upto] if m.get("role") == role)


def fetch_slots(messages, getter_names):
    """궤적이 호출한 getter들의 생산-슬롯 = 자연 fetch-then-use 커버리지."""
    out = set()
    for m in messages:
        if m.get("role") == "assistant":
            for tc in m.get("tool_calls") or []:
                if tc["function"]["name"] in getter_names:
                    out.add(produced_slot(tc["function"]["name"]))
    return out


def goal_phrase(goal):
    return (goal or "complete my request").replace("_", " ")


def build_ask(ex, ask_items, first_a):
    msgs = ex["messages"]
    params = ", ".join(k for k, _ in ask_items)
    provided = ", ".join("%s: %s" % (k, v) for k, v in ask_items)
    new = [msgs[0],
           {"role": "user", "content": "Hi! I'd like to %s." % goal_phrase(ex["_meta"].get("goal"))},
           {"role": "assistant",
            "content": "I can help with that. To proceed I'll need to verify your identity — could you provide your %s?" % params,
            "_supervise": True},
           {"role": "user", "content": "Sure — %s." % provided}]
    new += msgs[first_a:]
    out = dict(ex)
    out["messages"] = new
    out["_meta"] = dict(ex["_meta"])
    out["_meta"]["d5_branch"] = "ask"
    out["_meta"]["d5_ask_keys"] = [k for k, _ in ask_items]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--getter_map", required=True)
    ap.add_argument("--out_ask", required=True)
    ap.add_argument("--frac", type=float, default=1.0, help="ask-적격 궤적 중 합성 비율")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sample", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)

    names_by_dom, slots_by_dom = load_getters(a.getter_map)
    rows = [json.loads(l) for l in open(a.inp, encoding="utf-8")]

    ask_out = []
    n_ask_eligible = n_emitted = n_has_fetch = 0
    asked_keys = Counter()        # 새 게이트가 물은 키
    heuristic_keys = Counter()    # 구 휴리스틱이 물었을 키
    gated_out = Counter()         # 게이트가 ask서 제외(fetch 귀속)했고 구 휴리스틱이면 잘못 물었을 키
    fetched_slots = Counter()     # getter로 fetch된 슬롯
    over_ask = Counter()          # ask했으나 fetchable (게이트 버그 검출 — 0이어야 함)
    prov_dist = Counter()

    for ex in rows:
        msgs = ex.get("messages") or []
        if not msgs or msgs[0].get("role") != "system":
            continue
        dom = ex["_meta"].get("domain")
        dom_getters = names_by_dom.get(dom, set())
        catalog_getters = {t["function"]["name"] for t in ex.get("tools", [])
                           if t["function"]["name"] in dom_getters}
        # fetch 가능 슬롯 = 이 궤적 카탈로그의 getter 슬롯 (∩ 도메인맵)
        catalog_slots = {produced_slot(g) for g in catalog_getters}

        fs = fetch_slots(msgs, dom_getters)
        if fs:
            n_has_fetch += 1
            for s in fs:
                fetched_slots[s] += 1

        first_a, args = first_call(msgs)
        if first_a is None or not args:
            continue
        user_text = prior_text(msgs, first_a, "user")
        tool_text = prior_text(msgs, first_a, "tool")

        ask_items = []
        for k, v in args.items():
            if not isinstance(v, (str, int, float)):
                continue
            v_str = str(v)
            if len(v_str) < 3:
                continue
            in_tool = v_str in tool_text
            in_user = v_str in user_text
            prov = "tool" if in_tool else ("user" if in_user else "none")
            prov_dist[prov] += 1
            is_heur = any(h in k.lower() for h in _HEUR)
            if is_heur:
                heuristic_keys[k] += 1
            fetchable = key_is_fetchable(k, catalog_slots)
            # D5 게이트: user provenance & not tool-fetched & not 카탈로그-fetchable -> ask
            if prov == "user" and not fetchable:
                ask_items.append((k, v_str))
                asked_keys[k] += 1
            else:
                # fetch로 귀속됨. 구 휴리스틱이었으면 잘못 물었을 키를 기록
                if is_heur and (prov == "tool" or fetchable):
                    gated_out[k] += 1

        # 게이트 사후검증: ask한 키 중 fetchable이 있으면 버그
        for k, _ in ask_items:
            if key_is_fetchable(k, catalog_slots):
                over_ask[k] += 1

        if ask_items:
            n_ask_eligible += 1
            if rng.random() < a.frac:
                ask_out.append(build_ask(ex, ask_items, first_a))
                n_emitted += 1

    with open(a.out_ask, "w", encoding="utf-8") as f:
        for ex in ask_out:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    # 대조 커버리지: ask된 키 토큰 ∩ fetch된 슬롯 토큰 = 같은 개념이 양 분기에 = 진짜 대조쌍
    ask_tokens = set().union(*[_tok(k) for k in asked_keys]) if asked_keys else set()
    fetch_tokens = set().union(*[_tok(s) for s in fetched_slots]) if fetched_slots else set()
    overlap = sorted(ask_tokens & fetch_tokens)

    print("=== D5 CONTRASTIVE BUILD ===")
    print("input trajectories       : %d" % len(rows))
    print("traj w/ natural fetch     : %d (fetch-then-use 분기 자연 존재)" % n_has_fetch)
    print("ask-eligible traj         : %d   emitted(ask) : %d (frac=%.2f)" % (n_ask_eligible, n_emitted, a.frac))
    print("1st-call arg provenance   :", dict(prov_dist))
    print("\n-- 새 게이트가 물은 키 (user-prov & not-fetchable) --")
    for k, c in asked_keys.most_common(25):
        print("   %-26s %d" % (k, c))
    print("\n-- 구 키워드 휴리스틱이 물었을 키 (비교) --")
    for k, c in heuristic_keys.most_common(25):
        print("   %-26s %d" % (k, c))
    print("\n-- ★게이트가 ask서 제외(=fetch 귀속)한 키 = 구 휴리스틱이면 잘못 ask했을 것 --")
    if gated_out:
        for k, c in gated_out.most_common(25):
            print("   %-26s %d  (fetch-우선)" % (k, c))
    else:
        print("   (없음 — 첫-call이 대부분 순수 identity)")
    print("\n-- getter로 fetch된 슬롯 (fetch 분기 커버리지) --")
    for s, c in fetched_slots.most_common(25):
        print("   %-28s %d" % (s, c))
    print("\n대조(ask-키 ∩ fetch-슬롯 토큰):", overlap or "(없음 = identity/data 깔끔 분리)")
    print("OVER-ASK (ask했으나 fetchable·0이어야 함):", dict(over_ask) or "0 ✓")
    print("\nwrote", a.out_ask, "(%d ask exemplars)" % len(ask_out))

    if a.sample:
        for g in ask_out[:a.sample]:
            print("\n=== ask goal=%s keys=%s ===" % (g["_meta"].get("goal"), g["_meta"].get("d5_ask_keys")))
            for m in g["messages"][:6]:
                if m["role"] == "assistant" and m.get("tool_calls"):
                    print("  [A] CALL", [t["function"]["name"] for t in m["tool_calls"]])
                else:
                    print("  [%s] %s" % (m["role"], str(m.get("content"))[:90]))


if __name__ == "__main__":
    main()
