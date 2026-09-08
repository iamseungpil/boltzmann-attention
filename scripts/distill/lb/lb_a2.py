# -*- coding: utf-8 -*-
"""A2 declarations for the LB engines: one file per domain, a2/<domain>.lb.json, seven sections.

load(domain)      read a2/<domain>.lb.json
migrate(domain)   build it once from the legacy three-layer files (settings + specific), data to data:
                  no judgement, no new sentences - only keys move. Task-specific cases stay data.

    python lb_a2.py migrate banking_knowledge
"""

import io
import json
import os

from lb_coordinator import fam
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
A2_DIR = os.path.join(HERE, "a2")


def _read(name):
    p = os.path.join(A2_DIR, name)
    return json.load(io.open(p, encoding="utf-8")) if os.path.exists(p) else None


def load(domain):
    return _read("%s.lb.json" % domain)


def _resolve_refs(node, table):
    """{"a3": [axis, subject]} references become the ontology value (exact match, else KeyError)."""
    if isinstance(node, dict):
        ref = node.get("a3")
        if isinstance(ref, list) and len(ref) == 2 and len(node) == 1:
            return table[(ref[0], ref[1])]
        return {k: _resolve_refs(v, table) for k, v in node.items()}
    if isinstance(node, list):
        return [_resolve_refs(v, table) for v in node]
    return node


def migrate(domain):
    src = {}
    for part in ("settings", "specific"):
        src.update(_read("%s.%s.json" % (domain, part)) or {})
    if not src:
        raise SystemExit("no legacy A2 for %s" % domain)
    table = {(r.get("axis"), r.get("subject")): r.get("value")
             for r in (src.get("policy_ontology") or {}).get("rows") or []}
    src = {k: (v if k == "policy_ontology" else _resolve_refs(v, table)) for k, v in src.items()}
    d, ep, arb = src.get("dispatcher_role_check") or {}, src.get("eplan") or {}, src.get("arbitration") or {}
    names = src.get("discoverable_name_check") or {}
    procedures = [_procedure(p) for p in src.get("procedures") or []]
    metrics = {m.get("trigger_tool"): m for m in src.get("ledger_metrics") or []}
    base = _read(os.path.join("base", "shared.json")) or {}
    audit, bind = base.get("claim_audit") or {}, src.get("claim_bindings") or {}
    for sp in src.get("prescription_redirect") or []:
        procedures.append(_prescription(sp))
    out = {
        "domain": domain,
        "model_context": 131072,   # the served model's context; LB6 folds and LB7 delivers against it
        "dispatch": {"agent_call": d.get("agent_call"), "user_call": d.get("user_call"),
                     "unlock_tool": d.get("unlock_tool"), "give_tool": d.get("give_tool"),
                     "name_args": d.get("name_args") or {}, "payload_key": ep.get("dispatch_args_key") or "arguments"},
        "failure_markers": src.get("failure_markers") or [],
        "LB1": {
            "prerequisites": [{"dep": x.get("dep"), "reads": x.get("reads") or [], "source": x.get("source")}
                              for x in (src.get("relations") or {}).get("declarations") or []],
            "gates": [{"id": g.get("id"), "predicate": g.get("predicate"), "satisfiers": sorted(g.get("satisfiers") or {}),
                       "applies_to": g.get("applies_to") or [], "exempt": (g.get("applies_when") or {}).get("not_in") or []}
                      for g in src.get("gates") or [] if g.get("satisfiers")],
            "procedures": procedures,
            "write_tools": ep.get("write_tools") or [],
            "feedback": {"single": arb.get("dominated_push_feedback"), "merged": arb.get("merged_requirement_feedback")},
        },
        "LB2": {"tools": [_tool(t) for t in src.get("scaffold_get_tools") or []],
                "derived": [_derived(n, metrics) for n in src.get("derived") or []],
                "a3_rows": [{"axis": r.get("axis"), "subject": r.get("subject"), "value": r.get("value")}
                            for r in (src.get("policy_ontology") or {}).get("rows") or []],
                "computations":
                [dict(kind="select", applies_to=r.get("dispatch_tool"), when={"arg": d.get("name_args", {}).get(r.get("dispatch_tool"), "agent_tool_name"), "prefix": r.get("tool_prefix")},
                      param=r.get("param"), key_field=r.get("key_field"), require=r.get("require"),
                      criteria_fields=r.get("criteria_fields"), match=r.get("match"), on_ambiguous=r.get("on_ambiguous", "none"))
                 for r in src.get("reference_filter") or []]
                + [dict(kind="ratio_cap", applies_to=s.get("applies_to"), when=_when(s), param=s.get("param"),
                      record_key=s.get("record_key_field"), limit_field=s.get("limit_field"), pct_by=s.get("pct_by"),
                      feedback=s.get("feedback")) for s in src.get("param_cap_check") or []]
                + [dict(kind="distinct", tool=t, pairs=s.get("pairs"), feedback=s.get("fail_feedback"))
                   for t, s in (src.get("distinct_args") or {}).items()]},
        "LB3": {
            "grounding":
                [dict(applies_to=s.get("applies_to"), when=_when(s), arg=a, sources=["records", "customer"],
                      feedback=s.get("feedback")) for s in src.get("write_arg_grounding") or []
                 for a in s.get("grounded_args") or []]
                + [dict(applies_to=s.get("applies_to"), when=_when(s), arg=s.get("id_key"), field=s.get("record_field"),
                        sources=["customer"], feedback=s.get("feedback")) for s in src.get("ref_verify") or []]
                + [dict(applies_to=s.get("tool"), arg=s.get("arg"), sources=["records"], feedback=s.get("feedback"))
                   for s in src.get("choice_grounding") or []]
                + [dict(applies_to=s.get("applies_to"), when=_when(s), arg=s.get("id_key"), state=s["require_tokens"],
                        feedback=s.get("feedback"))
                   for s in src.get("write_evidence_specs") or [] if _record_state(s.get("require_tokens"), src)],
            "names": {"feedback_wrong_suffix": names.get("feedback_wrong_suffix"),
                      "feedback_not_discoverable": names.get("feedback_not_discoverable"),
                      "feedback_rejected": names.get("feedback_rejected") or REJECTED},
            "schema": src.get("tool_signatures") or {},
            "identifying": {"args": sorted(set((src.get("field_ops") or {}).get("id_ref") or [])
                                           | set(src.get("identifying_arg_types") or [])), "feedback": UNGROUNDED},
        },
        "LB4": {"sets":
                [dict(kind="follow_up", after=c.get("after"), requires=c.get("requires"), decision_tools=c.get("decision_tools"),
                      feedback=c.get("feedback"), decision_feedback=c.get("decision_feedback"))
                 for c in src.get("follow_up_chains") or [] if not _bare_write_nudge(c, ep.get("write_tools") or [])]
                + ([dict(kind="settled_rows", settle_tool=w.get("settle_tool"), submit_tool=w.get("submit_tool"),
                         id_key=w.get("id_key") or "transaction_id", feedback=w.get("feedback"))
                    for w in [src.get("withdrawn_row_check")] if w and w.get("settle_tool")])
                + [dict(kind="once", applies_to=s.get("applies_to"), when=_when(s), keys=s.get("keys"), feedback=s.get("feedback"))
                   for s in src.get("write_once_keys") or []]
                + ([dict(kind="ledger", entity_key=ep.get("entity_key"), list_tools=_list(ep.get("list_enumerator")),
                         write_tools=ep.get("write_tools") or [], finalize_writes=ep.get("finalize_writes") or [],
                         feedback=ep.get("coverage_feedback") or COVERAGE)] if ep.get("entity_key") else [])
                + ([dict(kind="claims", question=audit.get("question"), kinds=bind.get("kinds", ""),
                         kind_guidance=bind.get("kind_guidance", ""), event_map=bind.get("event_map") or {},
                         write_tools=ep.get("write_tools") or [], transfer_tools=(src.get("require_doc_before") or {}).get("tools") or [],
                         feedback=audit.get("feedback"), feedback_pending=audit.get("feedback_pending"))]
                   if audit.get("question") and bind else [])},
        "LB5": {"transfer_tools": (src.get("require_doc_before") or {}).get("tools") or [],
                "doc_feedback": (src.get("require_doc_before") or {}).get("feedback"),
                "search_tools": src.get("search_tools") or [], "search_feedback": src.get("search_exhaust_escalation"),
                "steps_feedback": STEPS},
        "LB6": {"annotations": [{"field": a.get("field"), "note": a.get("note")}
                                for a in src.get("view_field_annotations") or [] if a.get("field") and a.get("note")]},
        "LB7": {"deliver_for": (src.get("require_doc_before") or {}).get("tools") or [], "max_chars": 90000,
                "have_value": _have_value(src)},
    }
    path = os.path.join(A2_DIR, "%s.lb.json" % domain)
    io.open(path, "w", encoding="utf-8").write(json.dumps(out, ensure_ascii=False, indent=1) + "\n")
    return path


REJECTED = ("Error: the environment already rejected '{name}' as unknown earlier in this conversation; that exact "
            "name does not exist. Do not reuse it - find the exact registered name first.")
COVERAGE = ("[COVERAGE] The request is not complete - these records were asked about and no successful action "
            "covers them yet: {missing}. Complete them with real tool calls before ending.")
STEPS = ("Error: [PROCEDURE-INCOMPLETE] you are about to hand this conversation off, but the procedure you entered "
         "still has steps nobody has done: {steps}. A transfer does not perform them.")


UNGROUNDED = ("Error: [GROUNDING] the value '{val}' you passed for {arg} does not appear in any tool output or "
              "customer message in this conversation - record values must be read from the records or given by the "
              "customer, never invented. Look it up (or ask), then retry with the actual value.")
VARIANTS = ("ledger", "ratefix")          # the live arm's declaration variants, applied once here
CATALOG_CONSTRAINTS = [                    # what the old catalog_filter hard-coded; now data
    {"param": "max_annual_fee", "field": "annual_fee", "sense": "le"},
    {"param": "max_fx_fee", "field": "fx_fee", "sense": "le"},
    {"param": "max_min_payment_pct", "field": "min_payment_pct", "sense": "le"},
    {"param": "min_cashback", "field": "cashback", "sense": "ge"},
    {"param": "min_credit_limit", "field": "limit_max", "sense": "ge"},
    {"param": "needs_virtual_card", "field": "virtual_card", "sense": "flag"},
    {"param": "needs_purchase_protection", "field": "purchase_protection", "sense": "flag"},
    {"param": "credit_score", "field": "min_score", "sense": "le"},
    {"param": "invited", "field": "invite_only", "sense": "unless"},
]


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items() if not str(k).startswith("_note")}
    if isinstance(o, list):
        return [_clean(v) for v in o]
    return o


def _tool(t):
    d = dict(t)
    have = d.pop("variants", None) or {}
    hit = next((v for v in VARIANTS if isinstance(have.get(v), dict)), None)
    if hit:
        d.update(have[hit])
    d = _clean(d)
    op = d.get("op") or {}
    if op.get("op") == "catalog_filter":
        op = dict(op, constraints=CATALOG_CONSTRAINTS, segment={"param": "business", "field": "business"}, label_field="card")
        d["op"] = op
    keep = ("name", "description", "params", "optional", "examples", "op", "ground", "isolate", "requires_reads",
            "return_template", "return_template_empty", "missing_hint", "result_round", "result_range",
            "result_range_feedback", "grounded_params")
    return {k: d[k] for k in keep if k in d}


def _derived(n, metrics):
    node = _clean(dict(n))
    src = next((i[5:] for i in node.get("inputs") or [] if i.startswith("tool:")), None)
    m = metrics.get(src) or next(iter(metrics.values()), {}) if metrics else {}
    if node.get("op") == "formalize" and node.get("prompt") in m:
        node["prompt"] = m[node["prompt"]]
    texts = {"window_remaining": "window_text", "days_since_earliest": "age_text", "subtract_by_group": "exhausted_text"}
    key = texts.get(node.get("op"))
    text = next((mm.get(key) for mm in metrics.values() if key and mm.get(key)), None)
    if text:
        node["text"] = text.replace("{remaining}", "{remaining_groups}") if node["op"] == "subtract_by_group" else text
    return node


def _have_value(src):
    out = {}
    for s in src.get("have_value_reask") or []:
        marker = s.get("value_pattern") or ""
        out[s.get("write")] = {"write": s.get("write"), "arg": s.get("arg"), "producer_marker": s.get("producer_marker"),
                               "value_after": marker.split("\\s*")[0].strip() if marker else None,
                               "reask_signals": s.get("reask_signals"), "feedback": s.get("feedback")}
    for s in src.get("value_acquisition") or []:
        e = out.setdefault(s.get("write"), {"write": s.get("write"), "arg": s.get("arg"),
                                            "producer_marker": s.get("producer_marker"), "reask_signals": s.get("reask_signals")})
        e.update({"acquire_tool": s.get("acquire_tool"), "give_tool": s.get("give_tool"), "acquire_feedback": s.get("feedback")})
    return list(out.values())


def _record_state(tokens, src):
    """A write-evidence token is a record state only if it is neither a tool's name (that is a
    prerequisite, LB1's kind) nor a sentence one of our own verifiers prints (that would demand our
    verdict, a prescription). What survives is a word the environment writes into a record."""
    if not tokens:
        return False
    verdicts = " ".join(str(v) for t in src.get("scaffold_get_tools") or [] for k, v in t.items() if "template" in k)
    elsewhere = json.dumps({k: v for k, v in src.items() if k != "write_evidence_specs"})
    return all(tok not in verdicts and ('"%s' % tok) not in elsewhere for tok in tokens)


def _bare_write_nudge(chain, write_tools):
    """A follow-up may demand a read or a decision step; a bare state-changing write with no decision
    is a nudge to act on a condition the ledger cannot see (submitted is not resolved)."""
    req = {fam(x) for x in chain.get("requires") or []}
    return bool(req & {fam(w) for w in write_tools}) and not chain.get("decision_tools")


def _when(s):
    w = s.get("applies_when") or {}
    return {"arg": w.get("arg"), "prefix": w.get("prefix")} if w.get("arg") else {}


def _list(v):
    return list(v) if isinstance(v, list) else ([v] if v else [])


PROC_FEEDBACK = ("unmet",)          # the only feedback the walker still uses


def _procedure(p):
    q = {k: v for k, v in p.items() if not k.startswith("_note")}
    q["feedback"] = {k: v for k, v in (p.get("feedback") or {}).items() if k in PROC_FEEDBACK}
    q["prohibits"] = {n: {"quote": s.get("_quote")} for n, s in (p.get("prohibits") or {}).items() if isinstance(s, dict)}
    q["nodes"] = [{k: v for k, v in n.items() if k in ("id", "tool", "tool_any", "tool_prefix", "requires", "min_count")}
                  for n in p.get("nodes") or []]
    return q


def _prescription(sp):
    """prescription_redirect -> a procedure: customer signal opens it; the prescribed tool precedes the target."""
    return {"id": "prescription:" + str(sp.get("prefix")), "enforce": True,
            "_quote_order": "AUTHORED (migrated from prescription_redirect 2026-09-08; source docs in _source)",
            "_source": sp.get("_source") or [], "enter_when": {"signals": sp.get("signals") or []},
            "nodes": [{"id": "prescribed", "tool_any": [sp.get("requires_absent_tool")]},
                      {"id": "target", "tool_prefix": sp.get("prefix"), "requires": ["prescribed"]}],
            "feedback": {"unmet": sp.get("feedback")}}


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "migrate":
        print("wrote", migrate(sys.argv[2]))
    else:
        print(__doc__)
