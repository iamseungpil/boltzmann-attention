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
    for sp in src.get("prescription_redirect") or []:
        procedures.append(_prescription(sp))
    out = {
        "domain": domain,
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
            "feedback": {"single": arb.get("dominated_push_feedback"), "merged": arb.get("merged_requirement_feedback")},
        },
        "LB2": {"computations":
                [dict(kind="ratio_cap", applies_to=s.get("applies_to"), when=_when(s), param=s.get("param"),
                      record_key=s.get("record_key_field"), limit_field=s.get("limit_field"), pct_by=s.get("pct_by"),
                      feedback=s.get("feedback")) for s in src.get("param_cap_check") or []]
                + [dict(kind="distinct", tool=t, pairs=s.get("pairs"), feedback=s.get("fail_feedback"))
                   for t, s in (src.get("distinct_args") or {}).items()]},
        "LB3": {
            "grounding":
                [dict(applies_to=s.get("applies_to"), when=_when(s), arg=a, sources=["records", "customer"],
                      feedback=s.get("feedback")) for s in src.get("write_arg_grounding") or []
                 for a in s.get("grounded_args") or []]
                + [dict(applies_to=s.get("applies_to"), when=_when(s), arg=s.get("id_key"), sources=["records"],
                        tokens=s.get("require_tokens"), feedback=s.get("feedback"))
                   for s in src.get("write_evidence_specs") or [] if s.get("require_tokens")]
                + [dict(applies_to=s.get("applies_to"), when=_when(s), arg=s.get("id_key"), field=s.get("record_field"),
                        sources=["customer"], feedback=s.get("feedback")) for s in src.get("ref_verify") or []]
                + [dict(applies_to=s.get("tool"), arg=s.get("arg"), sources=["records"], feedback=s.get("feedback"))
                   for s in src.get("choice_grounding") or []],
            "names": {"feedback_wrong_suffix": names.get("feedback_wrong_suffix"),
                      "feedback_not_discoverable": names.get("feedback_not_discoverable"),
                      "feedback_rejected": names.get("feedback_rejected") or REJECTED},
            "schema": src.get("tool_signatures") or {},
        },
        "LB4": {"sets":
                [dict(kind="follow_up", after=c.get("after"), requires=c.get("requires"), decision_tools=c.get("decision_tools"),
                      feedback=c.get("feedback"), decision_feedback=c.get("decision_feedback"))
                 for c in src.get("follow_up_chains") or []]
                + ([dict(kind="settled_rows", settle_tool=w.get("settle_tool"), submit_tool=w.get("submit_tool"),
                         id_key=w.get("id_key") or "transaction_id", feedback=w.get("feedback"))
                    for w in [src.get("withdrawn_row_check")] if w and w.get("settle_tool")])
                + [dict(kind="once", applies_to=s.get("applies_to"), when=_when(s), keys=s.get("keys"), feedback=s.get("feedback"))
                   for s in src.get("write_once_keys") or []]
                + ([dict(kind="ledger", entity_key=ep.get("entity_key"), list_tools=_list(ep.get("list_enumerator")),
                         write_tools=ep.get("write_tools") or [], finalize_writes=ep.get("finalize_writes") or [],
                         feedback=ep.get("coverage_feedback") or COVERAGE)] if ep.get("entity_key") else [])},
        "LB5": {"transfer_tools": (src.get("require_doc_before") or {}).get("tools") or [],
                "doc_feedback": (src.get("require_doc_before") or {}).get("feedback"),
                "search_tools": src.get("search_tools") or [], "search_feedback": src.get("search_exhaust_escalation"),
                "unlock_feedback": src.get("tool_unlock_hint"), "steps_feedback": STEPS},
        "LB6": {"annotations": [{"field": a.get("field"), "note": a.get("note")}
                                for a in src.get("view_field_annotations") or [] if a.get("field") and a.get("note")]},
        "LB7": {"deliver_for": (src.get("require_doc_before") or {}).get("tools") or [], "max_chars": 90000,
                "names_feedback": NAMES},
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
NAMES = ("Documents you already retrieved name these tools, and none has been called: {names}. If one of them is "
         "the step you need, unlock and call it by exactly that name.")


def _when(s):
    w = s.get("applies_when") or {}
    return {"arg": w.get("arg"), "prefix": w.get("prefix")} if w.get("arg") else {}


def _list(v):
    return list(v) if isinstance(v, list) else ([v] if v else [])


def _procedure(p):
    q = {k: v for k, v in p.items() if not k.startswith("_note")}
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
