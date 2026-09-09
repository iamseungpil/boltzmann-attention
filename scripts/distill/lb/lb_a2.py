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
    d, ep = src.get("dispatcher_role_check") or {}, src.get("eplan") or {}
    names = src.get("discoverable_name_check") or {}
    procedures = [_procedure(p) for p in src.get("procedures") or []]
    metrics = {m.get("trigger_tool"): m for m in src.get("ledger_metrics") or []}
    base = _read(os.path.join("base", "shared.json")) or {}
    audit, bind = base.get("claim_audit") or {}, src.get("claim_bindings") or {}
    for sp in src.get("prescription_redirect") or []:
        procedures.append(_prescription(sp))
    order = ORDER     # the walker fills {tool} and {missing}; the old arbitration text used other slot names
    for x in (src.get("relations") or {}).get("declarations") or []:
        reads = _env_reads(x.get("reads"), src)
        if reads:
            procedures.append(_edge_procedure("requires:" + x.get("dep"), x.get("dep"), reads, x.get("source"), order))
    for g in src.get("gates") or []:
        sat = sorted(g.get("satisfiers") or {})
        exempt = set((g.get("applies_when") or {}).get("not_in") or [])
        for t in g.get("applies_to") or []:
            if sat and t not in exempt:
                procedures.append(_edge_procedure("%s:%s" % (g.get("id"), t), t, sat, g.get("predicate"), order))
    out = {
        "domain": domain,
        "model_context": 131072,   # the served model's context; LB6 folds the view against it
        "dispatch": {"agent_call": d.get("agent_call"), "user_call": d.get("user_call"),
                     "unlock_tool": d.get("unlock_tool"), "give_tool": d.get("give_tool"),
                     "name_args": d.get("name_args") or {}, "payload_key": ep.get("dispatch_args_key") or "arguments"},
        "failure_markers": src.get("failure_markers") or [],
        "LB1": {
            "procedures": procedures,
            "write_tools": ep.get("write_tools") or [],
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
                # a free-text argument the environment already defaults: the model fills its own
                # sentence, that sentence lands in the record, and a DB-judged task fails on it.
                # Authored 2026-08-31 with the tasks it was measured on, then never wired to an
                # engine - 060 061 062 065 066 067 068 069 all passed a written reason and all
                # scored 0 on the run of 2026-09-10.
                + [dict(applies_to=d.get("agent_call"),
                        when={"arg": (d.get("name_args") or {}).get(d.get("agent_call")) or "agent_tool_name",
                              "prefix": tool},
                        arg=a, sources=["records", "customer"], feedback=FREE_TEXT_DEFAULT)
                   for tool, args in (src.get("free_text_defaults") or {}).items() for a in args]
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
                         kind_guidance=bind.get("kind_guidance", ""), event_map=_specific(bind.get("event_map") or {}),
                         write_tools=ep.get("write_tools") or [], transfer_tools=(src.get("require_doc_before") or {}).get("tools") or [],
                         feedback=audit.get("feedback"), feedback_pending=audit.get("feedback_pending"))]
                   if audit.get("question") and bind else [])},
        "LB5": {"transfer_tools": (src.get("require_doc_before") or {}).get("tools") or [],
                "doc_feedback": (src.get("require_doc_before") or {}).get("feedback"),
                "search_tools": src.get("search_tools") or [], "search_feedback": src.get("search_exhaust_escalation"),
                },
        "LB6": {"annotations": [{"field": a.get("field"), "note": a.get("note")}
                                for a in src.get("view_field_annotations") or [] if a.get("field") and a.get("note")],
                # what a folded search result must still show: the ids, so the model can read the one
                # document it wants instead of running the search again
                "keep": {t: list(src.get("view_keep_lines") or []) for t in src.get("search_tools") or []
                         if src.get("view_keep_lines")}},
        # the action index is machine-derived from the environment's own files (titles + the tools
        # each document names), so it moves as data. It was authored and then reached no engine: it
        # appears zero times in the built A2 of 2026-09-10, and the model spent its turns grepping.
        "LB7": {"have_value": _have_value(src),
                "action_index": {"text": (src.get("policy_ontology") or {}).get("action_index_text"),
                                 "rows": (src.get("policy_ontology") or {}).get("action_index") or []},
                "write_rules": [{"applies_to": w.get("applies_to"), "text": w.get("text")}
                                for w in src.get("write_rules") or [] if w.get("text")]},
    }
    path = os.path.join(A2_DIR, "%s.lb.json" % domain)
    io.open(path, "w", encoding="utf-8").write(json.dumps(out, ensure_ascii=False, indent=1) + "\n")
    return path


ORDER = ("Error: [ORDER] '{tool}' cannot be carried out yet - not by you, and not by the customer acting on your "
         "instruction. This has to hold first: {missing}. Do that now with the real tool calls.")
# A procedure reaches the model on two paths and only the first one blocks the call. The refusal
# sentence above belongs to that path alone; on the other it reports a refusal that never happened.
ORDER_SURFACE = "[ORDER] '{tool}' normally comes after {missing}."
UNGROUNDED = ("Error: [GROUNDING] the value '{val}' you passed for {arg} does not appear in any tool output or "
              "customer message in this conversation - record values must be read from the records or given by the "
              "customer, never invented. Look it up (or ask), then retry with the actual value.")
REJECTED = ("Error: the environment already rejected '{name}' as unknown earlier in this conversation; that exact "
            "name does not exist. Do not reuse it - find the exact registered name first.")
COVERAGE = ("[COVERAGE] The request is not complete - these records were asked about and no successful action "
            "covers them yet: {missing}. Complete them with real tool calls before ending.")
VARIANTS = ("ledger", "ratefix")          # the live arm's declaration variants, applied once here


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


def _edge_procedure(pid, dep, reads, quote, order):
    """A policy prerequisite as a procedure: always active, the dependent step requires the reads."""
    return {"id": pid, "enforce": True, "_quote_order": quote or "", "_source": [],
            "nodes": [{"id": r, "tool_prefix": r} for r in reads] + [{"id": dep, "tool_prefix": dep, "requires": list(reads)}],
            "prohibits": {}, "feedback": {"unmet": order, "unmet_surface": ORDER_SURFACE}}


def _specific(emap):
    """Event patterns that name something. "__effective_write__" (any write) is evidence of nothing."""
    out = {}
    for k, v in emap.items():
        v = v if isinstance(v, list) else [v]
        v = [x for x in v if x and x != "__effective_write__"]
        if v:
            out[k] = v
    return out


def _env_reads(reads, src):
    """A prerequisite read must be the environment's tool. One of our own verifiers as a prerequisite
    is a prescription to use it: 137 of 164 requirement denies on the base census were that one line."""
    ours = {t.get("name") for t in src.get("scaffold_get_tools") or []}
    return [r for r in reads or [] if fam(r) not in ours]


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


FREE_TEXT_DEFAULT = (
    "Error: [FREE-TEXT-DEFAULT] '{val}' is not a value this conversation established for {arg} - no "
    "tool output and nothing the customer said contains it, so it is a sentence you composed. This "
    "argument has an environment default and the record is meant to keep it. Re-issue the same call "
    "with {arg} left out entirely; do not substitute another wording.")

PROC_FEEDBACK = ("unmet", "unmet_surface")   # the blocking sentence and the one that blocks nothing


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
