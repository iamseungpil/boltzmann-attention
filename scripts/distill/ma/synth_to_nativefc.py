#!/usr/bin/env python
"""facet (3) keystone — synth content-op routing을 *native tool-call* 포맷으로 직생성.

격리실험 §facet(3): 유일하게 증명된 cross-bench 학습-전이(§21·op-IR)가 native 포맷서
살아남나(§23E 다리). op-IR "Output ONLY JSON" 폐기 → 모델이 native `resolve_selection`
tool_call을 emit. anchor_id는 모델-가시 인자서 *제외*(offload가 context anchor grounding·
order_id 날조 재수입 차단·`tau2_op_resolver.py:74-77`).

출력 = native-FC jsonl(tbnfc 포맷 동형): {tools:[resolve_selection], messages:[system,user,
assistant(tool_calls), tool(result=gold_id)], _meta}.

Usage: synth_to_nativefc.py --out route_native.jsonl --n_per_op 800 --N 5,10,20 [--diverse] [--seed 0]
"""
import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import synth_depth as sd  # noqa: E402

# resolve_selection 도구 스키마 = tau2_op_resolver가 소비하는 {op,attr?,among?,dir?,k?,set?}
# (anchor_id 제외 = 모델이 채우지 않음·offload가 context로 grounding).
RESOLVE_SELECTION_SCHEMA = {
    "type": "function",
    "function": {
        "name": "resolve_selection",
        "description": ("Identify the catalog item(s) the user is asking for by NAMING the selection "
                        "operation. Do NOT compute the answer or invent any item id — name the operation "
                        "and its operands; the engine resolves the concrete item over the full catalog."),
        "parameters": {
            "type": "object",
            "properties": {
                "op": {"type": "string",
                       "enum": ["filter", "argmax", "argmin", "rank", "comparative", "substitute", "create"],
                       "description": "the selection operation the query asks for"},
                "attr": {"type": "string", "description": "the ordinal/numeric attribute (argmax/argmin/rank/comparative)"},
                "among": {"type": "object", "description": "categorical filter constraints {attr: value}"},
                "dir": {"type": "string", "enum": ["greater", "less"], "description": "comparative direction"},
                "k": {"type": "integer", "description": "rank position (1=top)"},
                "set": {"type": "object", "description": "substitute/create: changed attribute values {attr: value}"},
            },
            "required": ["op"],
        },
    },
}

SYSTEM = ("You are a tool-using assistant. Call resolve_selection to identify the item the user wants "
          "by naming the operation. Never invent item ids; name the operation and operands only.")


def _user_content(ex):
    attrs = list(ex.get("cat_attrs") or [])
    ordk = ex.get("ord_attr")
    # ordinal attr는 실제로 채워진 경우(filter/argmax/.../comparative)에만 노출 — substitute/create는 categorical-only
    ord_used = bool(ordk) and any(it.get(ordk) is not None for it in ex.get("items", []))
    lines = [ex["nl"], ""]
    allattrs = attrs + ([ordk] if ord_used and ordk not in attrs else [])
    lines.append(f"Catalog item attributes: {allattrs}.")
    if ord_used:
        lines.append(f"The ordinal/numeric attribute is '{ordk}'.")
    return "\n".join(lines)


def _emit_args(op_ir):
    """op_ir → 모델 emit 인자 (anchor_id 제외)."""
    return {k: v for k, v in op_ir.items() if k != "anchor_id"}


def to_native(ex):
    args = _emit_args(ex["op_ir"])
    tcid = "call_1"
    return {
        "tools": [RESOLVE_SELECTION_SCHEMA],
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": _user_content(ex)},
            {"role": "assistant", "tool_calls": [
                {"id": tcid, "type": "function",
                 "function": {"name": "resolve_selection", "arguments": json.dumps(args, ensure_ascii=False)}}]},
            {"role": "tool", "tool_call_id": tcid,
             "content": json.dumps({"status": "ok", "tool": "resolve_selection", "result": ex["gold_id"]},
                                   ensure_ascii=False)},
        ],
        "_meta": {"op": ex["op"], "N": len(ex["items"]), "label": ex.get("label"),
                  "gold_op_ir": ex["op_ir"]},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_per_op", type=int, default=800)
    ap.add_argument("--N", default="5,10,20", help="comma list of catalog sizes")
    ap.add_argument("--ops", default="filter,argmax,argmin,rank,comparative,substitute,create")
    ap.add_argument("--width", type=int, default=2, help="substitute/create max changed attrs")
    ap.add_argument("--diverse", action="store_true", help="render_nl_diverse (expression isotropy)")
    ap.add_argument("--iso", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    Ns = [int(x) for x in a.N.split(",")]
    ops = a.ops.split(",")
    n = 0
    with open(a.out, "w", encoding="utf-8") as f:
        for op in ops:
            made = 0
            tries = 0
            while made < a.n_per_op and tries < a.n_per_op * 40:
                tries += 1
                N = rng.choice(Ns)
                w = rng.randint(1, a.width) if op in ("substitute", "create") else 0
                ex = sd.gen_example(rng, a.iso, op, N, width=w)
                if ex is None:
                    continue
                ex["nl"] = sd.render_nl_diverse(ex, rng) if a.diverse else sd.render_nl(ex, a.iso)
                f.write(json.dumps(to_native(ex), ensure_ascii=False) + "\n")
                made += 1
                n += 1
            print(f"[synth_native] op={op} made={made} tries={tries}", file=sys.stderr)
    print(f"[synth_native] wrote {n} → {a.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
