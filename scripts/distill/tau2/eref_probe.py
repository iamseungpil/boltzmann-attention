#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""eref_probe.py — E-REF 1차 프로브: 참조-바인딩(hop 수) 통제 사다리 P0/P1/P2 (무료·32B).

설계 정본 = reports/facet_rft_2026/E_REF_BOUNDARY_DESIGN_2026_07_11.md §2.
질의를 q = f(g(X))로 분해: g = 참조-바인딩(deixis→문맥 레코드 필드값·semantic),
f = 결정론 실행(argmax/filter·op-library·t2_formalize_exec.execute_formalized 동형).
셀 = hop 수만 통제(과제·후보·f 동일):
  P0 상수-제약: 제약값이 발화에 명시 ("whose size is 9, the most expensive")
  P1 1-hop:    "same size as mine" + anchor 레코드 1개를 tool 출력으로 제공
  P2 2-hop:    anchor를 주문-더미(K개 타-상품 레코드)에서 상품명으로 식별해야

케이스 = comp gz(sim_results/comp_retail_t4.results.json.gz)의 실 product/variants서
자동 생성·**셀 간 짝지음**(같은 (product,key,value) 시나리오·표면만 상이).
gold = 결정론 계산(자가-채점): spec_gold = {op:argmax, field:price, cons:[{key,value}]},
answer_gold = argmax(price) among {variants: options[key]==value} (유일 argmax·available
gold·availability-무관 동일답이 되도록 시나리오 필터 — available 관용과 정합).

채점 (★채점-정규화 = 오늘 fexec V0 아티팩트 교정·설계서 §2.3):
  - field 정규화: lowercase·strip·"options." 접두 제거·공백 정규화·{cost→price}
  - available 관용: {available,availability,in stock,...} 제약은 EM/실행 양쪽서 무시
  - 값 유형 정규화: 숫자 동치(9=="9"==9.0)·bool 관용·case-insensitive
  - strict(구-규약) EM도 병기 → 아티팩트 크기 정량화
지표: op/field/cons/full-EM · bind(표적 (key,value) 제약 재현 = g 그 자체) ·
  answer(결정론 실행기 f를 모델 spec에 적용한 최종답 == gold item_id) · 셀 간 짝 flip.

usage:
  py -3 eref_probe.py --gz <comp gz> --dry --out eref_cases.jsonl        # 생성만
  py -3 eref_probe.py --gz <comp gz> --base http://localhost:8140/v1 \
        --model Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8 --out eref_v1_32b.jsonl
표준 stdlib만 사용(리모트 단독 실행 가능). 커밋 전 상태 — 세션 지시 "커밋 금지".
"""
import argparse
import gzip
import json
import random
import re
import sys

_MISSING = object()

# ─────────────────────────── 1. 데이터: comp gz → products ───────────────────────────

def load_products(gz_path):
    op = gzip.open if gz_path.endswith(".gz") else open
    with op(gz_path, "rt", encoding="utf-8") as f:
        data = json.load(f)
    dec = json.JSONDecoder()
    prods = {}
    for s in data["simulations"]:
        for m in s.get("messages") or []:
            c = m.get("content")
            if m.get("role") != "tool" or not isinstance(c, str) or "variants" not in c:
                continue
            try:
                obj, _ = dec.raw_decode(c)  # COMP arm 주석 tail(EXTRA DATA) 절단
            except Exception:
                continue
            if isinstance(obj, dict) and obj.get("product_id") and isinstance(obj.get("variants"), dict):
                prods[obj["product_id"]] = obj
    return prods


# ─────────────────────────── 2. 시나리오 (짝지은 (product,key,value)) ───────────────────────────

def build_scenarios(prods, n_target=36, seed=20260711):
    """시나리오 조건(availability-무관 동일답 보장·설계서 §2.2):
    - key의 distinct 값 ≥2 (필터 비자명)
    - 매칭 variants ≥2 · available 매칭 ≥2 (랭킹 비자명)
    - gold = 매칭 전체(비가용 포함) 중 유일 price-argmax ∧ available
      → available 제약을 넣든 빼든 답 동일 = 관용과 정합
    - anchor = available 매칭 중 최저가(≠ gold)
    - P2 distractor: 같은 key를 가진 타 상품 ≥3 (그중 ≥2는 값이 target과 상이)"""
    cands = []
    for pid in sorted(prods):
        pr = prods[pid]
        vs = pr["variants"]
        keys = sorted({k for v in vs.values() for k in (v.get("options") or {})})
        for key in keys:
            vals = sorted({str((v.get("options") or {}).get(key)) for v in vs.values()
                           if key in (v.get("options") or {})})
            if len(vals) < 2:
                continue
            for val in vals:
                grp = [v for v in vs.values() if str((v.get("options") or {}).get(key)) == val]
                if len(grp) < 2:
                    continue
                avail = [v for v in grp if v.get("available") is True]
                if len(avail) < 2:
                    continue
                mx = max(v["price"] for v in grp)
                top = [v for v in grp if v["price"] == mx]
                if len(top) != 1 or top[0].get("available") is not True:
                    continue
                gold = top[0]
                anchor = min((v for v in avail if v["item_id"] != gold["item_id"]),
                             key=lambda v: v["price"], default=None)
                if anchor is None:
                    continue
                cands.append({"pid": pid, "name": pr["name"], "key": key, "val": val,
                              "gold_item": gold["item_id"], "gold_price": mx,
                              "anchor": anchor, "n_match": len(grp)})
    rng = random.Random(seed)
    rng.shuffle(cands)
    # distractor 풀: key → [(pid, name, variant)] (해당 key 보유 상품들)
    bykey = {}
    for pid in sorted(prods):
        pr = prods[pid]
        for v in pr["variants"].values():
            for k in (v.get("options") or {}):
                bykey.setdefault(k, []).append((pid, pr["name"], v))
    picked, per_prod = [], {}
    for sc in cands:
        if per_prod.get(sc["pid"], 0) >= 2:
            continue
        pool = [(p, n, v) for (p, n, v) in bykey.get(sc["key"], []) if p != sc["pid"]]
        # 상품당 1 variant·target과 값이 다른 것 우선
        seen, dis = set(), []
        for p, n, v in sorted(pool, key=lambda t: (str((t[2].get("options") or {}).get(sc["key"])) == sc["val"], t[0], t[2]["item_id"])):
            if p in seen:
                continue
            seen.add(p)
            dis.append((p, n, v))
        diff = [d for d in dis if str((d[2].get("options") or {}).get(sc["key"])) != sc["val"]]
        if len(dis) < 3 or len(diff) < 2:
            continue
        sc["distractors"] = dis[:4]
        picked.append(sc)
        per_prod[sc["pid"]] = per_prod.get(sc["pid"], 0) + 1
        if len(picked) >= n_target:
            break
    return picked


# ─────────────────────────── 3. 프롬프트 (라이브 FORMALIZE_SYS 동형·자족 사본) ───────────────────────────
# 사본 출처 = scripts/distill/tau2/t2_formalize_exec.py FORMALIZE_SYS (리모트 단독 실행용).
FORMALIZE_SYS = (
    "You are formalizing the SELECTION CRITERION for ONE ambiguous tool-call argument in a "
    "customer-service conversation. Read the transcript and the candidate records, then express "
    "the criterion the user stated as ONE JSON object with EXACTLY this schema:\n"
    '{"op": "argmax" | "argmin" | "filter" | "none" | "unresolvable",\n'
    ' "field": "<record field the criterion ranks (argmax/argmin) — null for filter/none>",\n'
    ' "constraints": [{"field": "<record field>", "op": "eq|ne|le|ge|lt|gt", "value": <required value>}, ...]}\n'
    "- argmax/argmin: the user asks for the candidate with the LARGEST/SMALLEST value of 'field' "
    "(e.g. most expensive, cheapest, biggest) among candidates satisfying the constraints.\n"
    "- filter: the criterion only restricts attributes (constraints), no ranking.\n"
    "- none: the user stated no formalizable criterion (a pure pick among candidates).\n"
    "- unresolvable: the user DID state a criterion, but the data it needs does not appear in "
    "the candidate records at all.\n"
    "Use ONLY field names that actually appear in the candidate records. Constraint values must be "
    "CONCRETE values resolved from the conversation and tool outputs ('op' defaults to 'eq' and may "
    "be omitted). Output the JSON object and NOTHING else."
)

T_P0 = ["I'd like to exchange my {name}. Among the variants whose {key} is {val}, give me the most expensive one.",
        "About my {name}: please swap it to the highest-priced variant whose {key} is {val}.",
        "For the {name} I bought, I want the variant with {key} {val} that costs the most."]
T_P1 = ["I'd like to exchange my {name} for the most expensive variant that has the same {key} as the one I currently have.",
        "About my {name}: please swap it to the highest-priced variant that keeps the same {key} as mine.",
        "For the {name} I bought, I want the variant that costs the most among the ones with the same {key} as my current one."]
T_P2 = ["I'd like an exchange: for the {name} somewhere in my orders, give me the most expensive variant that has the same {key} as the one I ordered.",
        "Please look at my orders. Swap the {name} I ordered to the highest-priced variant that keeps the same {key}.",
        "In one of my orders there is a {name}. I want to exchange it for the variant that costs the most among the ones with the same {key} as it."]


def _order_rec(order_id, name, pid, variant):
    return {"order_id": order_id, "status": "delivered",
            "items": [{"name": name, "product_id": pid, "item_id": variant["item_id"],
                       "price": variant["price"], "options": variant.get("options") or {}}]}


def build_case(sc, cell, idx, prods):
    rng = random.Random(hash((sc["pid"], sc["key"], sc["val"], cell)) & 0xFFFFFFFF)
    tmpl = {"P0": T_P0, "P1": T_P1, "P2": T_P2}[cell][idx % 3]
    utter = tmpl.format(name=sc["name"], key=sc["key"], val=sc["val"])
    lines = ["USER: " + utter]
    oid = lambda: "#W%07d" % rng.randrange(10 ** 7)
    if cell == "P1":
        lines.append("TOOL (get_order_details): " + json.dumps(
            _order_rec(oid(), sc["name"], sc["pid"], sc["anchor"]), ensure_ascii=False))
    elif cell == "P2":
        recs = [_order_rec(oid(), n, p, v) for p, n, v in sc["distractors"]]
        recs.append(_order_rec(oid(), sc["name"], sc["pid"], sc["anchor"]))
        rng.shuffle(recs)
        lines.append("TOOL (get_user_orders): " + json.dumps({"orders": recs}, ensure_ascii=False))
    prod = prods[sc["pid"]]
    lines.append("TOOL (get_product_details): " + json.dumps(prod, ensure_ascii=False))
    rec_lines = ["- %s   | record: %s" % (vid, json.dumps(v, ensure_ascii=False))
                 for vid, v in sorted(prod["variants"].items())]
    prompt = (FORMALIZE_SYS + "\n\n=== Conversation ===\n" + "\n".join(lines)
              + "\n\n=== Candidate records for 'new_item_id' ===\n" + "\n".join(rec_lines)
              + "\n\nThe agent must now choose 'new_item_id'. Formalize the user's selection "
                "criterion for this argument as the JSON object.")
    return {"cell": cell, "pid": sc["pid"], "name": sc["name"], "key": sc["key"],
            "val": sc["val"], "gold_item": sc["gold_item"],
            "anchor_item": sc["anchor"]["item_id"], "n_match": sc["n_match"],
            "prompt": prompt}


# ─────────────────────────── 4. 채점 정규화 + 결정론 실행기 f ───────────────────────────
AVAIL_FIELDS = {"available", "availability", "in stock", "in_stock", "instock", "is_available"}
FIELD_ALIAS = {"cost": "price", "prices": "price"}


def norm_field(f):
    if not isinstance(f, str):
        return None
    s = re.sub(r"\s+", " ", f.strip().lower().replace("_", " ").replace("-", " "))
    s = re.sub(r"^options?\s*[.\]>:]\s*", "", s)
    s = re.sub(r"^options?\[[\"']?", "", s).rstrip("]\"'")
    return FIELD_ALIAS.get(s, s)


def norm_value(v):
    if isinstance(v, bool):
        return ("b", v)
    try:
        return ("n", round(float(v), 6))
    except (TypeError, ValueError):
        pass
    s = str(v).strip().lower()
    if s in ("true", "yes"):
        return ("b", True)
    if s in ("false", "no"):
        return ("b", False)
    return ("s", s)


def parse_spec(txt):
    """t2_formalize_exec.parse_formalize 동형(자족 사본·검증 동일)."""
    if not isinstance(txt, str):
        return None
    i = txt.find("{")
    if i < 0:
        return None
    try:
        obj, _ = json.JSONDecoder().raw_decode(txt[i:])
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    op = str(obj.get("op") or "").strip().lower()
    if op not in ("argmax", "argmin", "filter", "none", "unresolvable"):
        return None
    field = obj.get("field")
    if field is not None and not isinstance(field, str):
        return None
    cons = []
    if obj.get("constraints") is not None:
        if not isinstance(obj["constraints"], list):
            return None
        for c in obj["constraints"]:
            if not isinstance(c, dict) or not isinstance(c.get("field"), str) or "value" not in c:
                return None
            cop = str(c.get("op") or "eq").strip().lower()
            if cop not in ("eq", "ne", "le", "ge", "lt", "gt"):
                return None
            cons.append({"field": c["field"], "op": cop, "value": c["value"]})
    if op in ("argmax", "argmin") and not field:
        return None
    return {"op": op, "field": field, "constraints": cons}


def _field_lookup(rec, field):
    """BFS·case-insensitive (t2_formalize_exec._field_lookup 동형)."""
    if not isinstance(field, str):
        return _MISSING
    fl = field.strip().lower()
    queue = [rec]
    while queue:
        cur = queue.pop(0)
        if isinstance(cur, dict):
            for kk, vv in cur.items():
                if str(kk).strip().lower() == fl:
                    return vv
            queue.extend(v for v in cur.values() if isinstance(v, (dict, list)))
        elif isinstance(cur, list):
            queue.extend(x for x in cur if isinstance(x, (dict, list)))
    return _MISSING


def execute_spec(spec, records):
    """f = 결정론 실행기 (t2_formalize_exec.execute_formalized 동형·정규화-후 spec 입력).
    records = [(item_id, variant record dict)]. availability 제약은 채점-관용으로 사전 제거됨."""
    if spec["op"] in ("none", "unresolvable"):
        return {"status": spec["op"], "ids": []}
    recs = list(records)
    for cons in spec["constraints"]:
        vals = [(c, _field_lookup(r, cons["field"])) for c, r in recs]
        if all(v is _MISSING for _, v in vals):
            return {"status": "unresolvable", "ids": []}
        keep = set()
        for c, v in vals:
            if v is _MISSING:
                continue
            eq = norm_value(v) == norm_value(cons["value"])
            if {"eq": eq, "ne": not eq}.get(cons["op"], False):
                keep.add(c)
            elif cons["op"] in ("le", "ge", "lt", "gt"):
                try:
                    fv, fw = float(v), float(cons["value"])
                    if {"le": fv <= fw, "ge": fv >= fw, "lt": fv < fw, "gt": fv > fw}[cons["op"]]:
                        keep.add(c)
                except (TypeError, ValueError):
                    pass
        recs = [(c, r) for c, r in recs if c in keep]
        if not recs:
            return {"status": "empty", "ids": []}
    if spec["op"] == "filter":
        return {"status": "ok", "ids": [c for c, _ in recs]}
    ranked = []
    for c, r in recs:
        try:
            ranked.append((float(_field_lookup(r, spec["field"])), c))
        except (TypeError, ValueError):
            pass
    if not ranked:
        return {"status": "unresolvable", "ids": []}
    best = (max if spec["op"] == "argmax" else min)(f for f, _ in ranked)
    return {"status": "ok", "ids": [c for f, c in ranked if f == best]}


def score(case, spec, prods):
    gold_key, gold_val = norm_field(case["key"]), norm_value(case["val"])
    out = {"parsed": spec is not None, "op": False, "field": False, "cons": False,
           "em": False, "bind": False, "answer": False,
           "op_strict": False, "cons_strict": False, "em_strict": False,
           "bound_vals": [], "spec": spec}
    if spec is None:
        return out
    # strict(구-규약): field 문자열 그대로·availability 포함·값 str 비교
    s_cons = {(c["field"], str(c["value"])) for c in spec["constraints"]}
    out["op_strict"] = spec["op"] == "argmax"
    out["cons_strict"] = s_cons == {(case["key"], str(case["val"]))}
    out["em_strict"] = out["op_strict"] and (spec.get("field") == "price") and out["cons_strict"]
    # normalized: available 관용 + 정규화
    n_cons = []
    for c in spec["constraints"]:
        nf = norm_field(c["field"])
        if nf in AVAIL_FIELDS:
            continue
        n_cons.append({"field": nf, "op": c["op"], "value": c["value"]})
    out["op"] = spec["op"] == "argmax"
    out["field"] = norm_field(spec.get("field")) == "price"
    cset = {(c["field"], norm_value(c["value"])) for c in n_cons if c["op"] == "eq"}
    out["cons"] = cset == {(gold_key, gold_val)} and all(c["op"] == "eq" for c in n_cons)
    out["bind"] = (gold_key, gold_val) in cset
    out["em"] = out["op"] and out["field"] and out["cons"]
    out["bound_vals"] = [[c["field"], str(c["value"])] for c in n_cons]
    # 최종답 = f(정규화-후 spec) — field도 정규화해 실행
    ex_spec = {"op": spec["op"], "field": norm_field(spec.get("field")), "constraints": n_cons}
    records = sorted(prods[case["pid"]]["variants"].items())
    res = execute_spec(ex_spec, records)
    out["answer"] = res["status"] == "ok" and set(res["ids"]) == {case["gold_item"]}
    out["exec_status"] = res["status"]
    return out


# ─────────────────────────── 5. 실행·집계 ───────────────────────────

def call_server(base, model, prompt, timeout=240):
    import urllib.request
    body = json.dumps({"model": model, "temperature": 0.0, "max_tokens": 500,
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(base.rstrip("/") + "/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    out = json.load(urllib.request.urlopen(req, timeout=timeout))
    return out["choices"][0]["message"]["content"]


# ─────────────────────── V2 오염 축 (직접실행·2026-07-11) ───────────────────────
# R1(clean)=1.00 기준에 소음 주입. gold 불변·bind/answer는 기존 score() 그대로.
# 구현 축: C(문맥-부하·Context-Rot) · A(후보-distractor 밀도·C43 정박치환).
# 유보: B(key-토큰 부재 패러프레이즈=상품별 표현 필요) · D(실궤적 replay=comp 추출) — 별도 spec.
def _pad_text(rng, nchars, prods):
    """무관 catalog JSON을 nchars까지 — 부하 주입(내용은 결정과 무관·field 이름 회피)."""
    pids = list(prods.keys())
    buf = []
    tot = 0
    while tot < nchars:
        pid = pids[rng.randrange(len(pids))]
        blob = json.dumps({"unrelated_catalog_entry": pid,
                           "info": prods[pid].get("name", "item")}, ensure_ascii=False)
        buf.append(blob)
        tot += len(blob)
    return "\n".join(buf)


def build_case_v2(sc, idx, prods, axis, level):
    """P1(1-hop·clean 1.00) 기반 + 축별 소음. base = build_case(P1)."""
    base = build_case(sc, "P1", idx, prods)
    rng = random.Random(hash((sc["pid"], sc["key"], axis, level)) & 0xFFFFFFFF)
    p = base["prompt"]
    if axis == "C":  # 문맥-부하: 후보 리스트 앞에 무관 텍스트 level자 삽입
        pad = _pad_text(rng, level, prods)
        p = p.replace("=== Candidate records",
                      "=== Unrelated context (ignore) ===\n" + pad
                      + "\n\n=== Candidate records", 1)
    elif axis == "A":  # 후보-distractor 밀도: 다른 상품 변형을 가짜 후보로 level개 주입
        others = [pp for pp in prods if pp != sc["pid"]]
        extra = []
        for _ in range(level):
            op = others[rng.randrange(len(others))]
            ov = prods[op]["variants"]
            if not ov:
                continue
            vid, v = list(ov.items())[rng.randrange(len(ov))]
            extra.append("- %s   | record: %s" % (vid, json.dumps(v, ensure_ascii=False)))
        if extra:  # 후보 블록 끝(선택 지시 앞)에 삽입
            p = p.replace("\n\nThe agent must now choose",
                          "\n" + "\n".join(extra) + "\n\nThe agent must now choose", 1)
    base["prompt"] = p
    base["axis"] = axis
    base["level"] = level
    return base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gz", required=True)
    ap.add_argument("--n", type=int, default=36)
    ap.add_argument("--cells", default="P0,P1,P2")
    ap.add_argument("--v2", default=None,
                    help="오염 사다리: 'C:2000,8000,20000' or 'A:2,5,10' (axis:levels)")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--base", default="http://localhost:8140/v1")
    ap.add_argument("--model", default="Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8")
    ap.add_argument("--workers", type=int, default=4, help="동시 요청 수(공유 GPU 배려·기본 4)")
    a = ap.parse_args()
    prods = load_products(a.gz)
    scens = build_scenarios(prods, n_target=a.n)
    cells = [c.strip() for c in a.cells.split(",") if c.strip()]
    if a.v2:  # V2 오염 사다리 — R1(P1 clean) + 축별 level 사다리
        axis, lv = a.v2.split(":")
        levels = [int(x) for x in lv.split(",")]
        cases = ([build_case(sc, "P1", i, prods) for i, sc in enumerate(scens)]  # level 0 = clean
                 + [build_case_v2(sc, i, prods, axis, L)
                    for L in levels for i, sc in enumerate(scens)])
        for c in cases:
            c.setdefault("axis", axis)
            c.setdefault("level", 0)
        print("V2 axis=%s levels=[0]+%s n/level=%d total=%d"
              % (axis, levels, len(scens), len(cases)), flush=True)
    else:
        print("products=%d paired-scenarios=%d cells=%s" % (len(prods), len(scens), cells), flush=True)
        cases = [build_case(sc, cell, i, prods) for i, sc in enumerate(scens) for cell in cells]
    if a.dry:
        if a.out:
            with open(a.out, "w", encoding="utf-8") as f:
                for c in cases:
                    f.write(json.dumps(c, ensure_ascii=False) + "\n")
            print("[dry] %d cases -> %s" % (len(cases), a.out))
        for c in cases[:3]:
            print("---", c["cell"], c["name"], c["key"], c["val"], "gold=", c["gold_item"])
        return
    from concurrent.futures import ThreadPoolExecutor
    def run_one(c):
        try:
            return c, call_server(a.base, a.model, c["prompt"]), None
        except Exception as e:
            return c, None, type(e).__name__
    rows = []
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for i, (c, ans, err) in enumerate(ex.map(run_one, cases)):
            if err:
                print("  [%d/%d] SERVER_ERR %s" % (i + 1, len(cases), err), flush=True)
                continue
            sc = score(c, parse_spec(ans), prods)
            row = dict(c)
            row.pop("prompt")
            row.update(sc)
            row["raw"] = ans[:800]
            rows.append(row)
            print("  [%d/%d] %s %s/%s=%s bind=%s em=%s ans=%s" % (
                i + 1, len(cases), c["cell"], c["name"][:14], c["key"], c["val"],
                sc["bind"], sc["em"], sc["answer"]), flush=True)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    # V2 사다리 집계 (level별 낙폭)
    if a.v2:
        axis = a.v2.split(":")[0]
        print("\n== E-REF V2 오염 사다리 (axis=%s·R1=level0 기준) ==" % axis)
        print("level    n   parse  bind   fullEM answer")
        bylv = {}
        for r in rows:
            bylv.setdefault(r.get("level", 0), []).append(r)
        for L in sorted(bylv):
            rs = bylv[L]
            n = len(rs)
            f = lambda k: sum(1 for r in rs if r.get(k)) / n
            print("%-8s %3d   %.2f   %.2f   %.2f   %.2f"
                  % (L, n, f("parsed"), f("bind"), f("em"), f("answer")))
        print("★bind 낙폭 = 소음이 참조-바인딩을 부수는가 (부수면 scaffold/scale/learn 축 갈림)")
        return
    # 집계
    print("\n== E-REF V1 집계 (채점-정규화 후 / strict 병기) ==")
    hdr = "cell  n   parse  op    field cons  fullEM bind  answer | EMstrict"
    print(hdr)
    bycell = {}
    for r in rows:
        bycell.setdefault(r["cell"], []).append(r)
    for cell in cells:
        rs = bycell.get(cell, [])
        n = len(rs)
        if not n:
            continue
        f = lambda k: sum(1 for r in rs if r.get(k)) / n
        print("%-4s %3d   %.2f  %.2f  %.2f  %.2f  %.2f   %.2f  %.2f  |  %.2f"
              % (cell, n, f("parsed"), f("op"), f("field"), f("cons"), f("em"),
                 f("bind"), f("answer"), f("em_strict")))
    # 짝 flip (같은 시나리오·셀 간)
    if len(cells) >= 2:
        key = lambda r: (r["pid"], r["key"], r["val"])
        maps = {cell: {key(r): r for r in bycell.get(cell, [])} for cell in cells}
        for a_, b_ in zip(cells, cells[1:]):
            common = set(maps[a_]) & set(maps[b_])
            for metric in ("bind", "answer"):
                lost = sum(1 for k in common if maps[a_][k][metric] and not maps[b_][k][metric])
                gain = sum(1 for k in common if not maps[a_][k][metric] and maps[b_][k][metric])
                print("paired %s->%s %s: kept-base=%d lost=%d gained=%d (n=%d)"
                      % (a_, b_, metric,
                         sum(1 for k in common if maps[a_][k][metric]), lost, gain, len(common)))
    # P1/P2 오답 바인딩 포렌식: 묶인 값이 distractor의 값인가
    for cell in ("P1", "P2"):
        rs = [r for r in bycell.get(cell, []) if not r["bind"] and r["parsed"]]
        if rs:
            print("\n-- %s bind-fail forensics (%d) --" % (cell, len(rs)))
            for r in rs[:12]:
                print("  %s %s/%s gold_val=%s bound=%s" % (r["name"][:16], r["key"], r["val"],
                                                           r["val"], r["bound_vals"]))


if __name__ == "__main__":
    main()
