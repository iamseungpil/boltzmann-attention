#!/usr/bin/env python3
"""§5b GBW catch — DETERMINISTIC diff-grounding verifier (P4_FALLBACK_IR_DESIGN §5b).

Idea: the resolver alone CANNOT catch grounded-but-wrong (GBW) — the model's criteria
resolve cleanly to a real-but-wrong variant. A SECOND deterministic verifier checks the
RESULT: every attribute that DIFFERS between old_options and the chosen variant must have
its NEW value attested in the NL request. If a changed attr's value is NOT NL-grounded ->
REJECT (GBW caught). No LLM judge (feedback-selector-verifier-deterministic).

Honest scope: grounding precision depends on the value->surface map (ABox). Pure substring
mis-rejects synonym/negation fallbacks ("no backlight" -> value "none"). So we accept an
optional --synonyms map (the ABox hook) and ALWAYS report BOTH:
  - RECALL  on GBW cases (kind==ok_wrong_variant): caught / missed
  - FALSE-REJECT on gold-correct cases: a useful verifier must NOT reject correct answers.
This measures the verifier itself, and quantifies the "catchable GBW" = thesis escape hatch
(P4 §0 strategic linkage): catchable GBW closes with retry on a SMALL model.

Standalone (offline, no GPU): join a --dump jsonl (from m_sigma_transfer_eval_v4.py --dump)
with the --cases jsonl, compute the verifier confusion matrix.
"""
import json, argparse


def _norm(s):
    return str(s).strip().lower()


def changed_attrs(old_options, chosen_options):
    """attrs whose value differs old -> chosen (the changes the criteria effected)."""
    return {k: chosen_options[k] for k in chosen_options
            if _norm(old_options.get(k)) != _norm(chosen_options.get(k))}


def value_grounded(val, nl, synonyms=None):
    """is `val` attested in the NL? substring (normalized) OR an ABox synonym surface form."""
    nln = _norm(nl)
    if _norm(val) and _norm(val) in nln:
        return True
    if synonyms:
        for surface in synonyms.get(_norm(val), []):
            if _norm(surface) in nln:
                return True
    return False


def verify(old_options, chosen_options, nl, synonyms=None):
    """(accept, ungrounded_changes). accept=True -> all changed attrs NL-grounded (pass).
    accept=False -> a changed attr's value not attested -> GBW caught -> REJECT."""
    ch = changed_attrs(old_options, chosen_options)
    ungrounded = [(k, v) for k, v in ch.items() if not value_grounded(v, nl, synonyms)]
    return (len(ungrounded) == 0, ungrounded)


# ----------------------------- offline runner -----------------------------
def _variant_options(case, exch_i, item_id):
    """look up the options dict of a resolved variant item_id within exchange exch_i."""
    if item_id is None or exch_i >= len(case["exchanges"]):
        return None
    for v in case["exchanges"][exch_i]["variant_catalog"]:
        if v["item_id"] == item_id:
            return v["options"]
    return None


def run(dump_path, cases_path, synonyms=None):
    cases = {str(json.loads(l)["task_id"]): json.loads(l) for l in open(cases_path, encoding="utf-8")}
    # confusion over $select ELEMENTS that resolved to a concrete variant
    stat = {"gbw_caught": 0, "gbw_missed": 0, "correct_accept": 0, "correct_falsereject": 0,
            "other": 0, "no_variant": 0}
    examples = {"caught": [], "falsereject": [], "missed": []}
    for line in open(dump_path, encoding="utf-8"):
        d = json.loads(line)
        case = cases.get(str(d.get("task_id")))
        if case is None:
            continue
        nl = case["nl"]
        for det in d.get("select_detail", []):
            i = det.get("i", 0)
            rid = det.get("resolved")
            chosen = _variant_options(case, i, rid)
            if chosen is None:
                stat["no_variant"] += 1
                continue
            old = case["exchanges"][i]["old_options"]
            accept, ung = verify(old, chosen, nl, synonyms)
            kind = det.get("kind", "")
            if kind == "ok_wrong_variant":                 # this is GBW (ground truth)
                if not accept:
                    stat["gbw_caught"] += 1
                    if len(examples["caught"]) < 5:
                        examples["caught"].append({"task": d.get("task_id"), "ungrounded": ung})
                else:
                    stat["gbw_missed"] += 1
                    if len(examples["missed"]) < 5:
                        examples["missed"].append({"task": d.get("task_id"), "chosen": chosen})
            elif det.get("correct"):                        # gold-correct
                if accept:
                    stat["correct_accept"] += 1
                else:
                    stat["correct_falsereject"] += 1
                    if len(examples["falsereject"]) < 5:
                        examples["falsereject"].append({"task": d.get("task_id"), "ungrounded": ung})
            else:
                stat["other"] += 1
    gbw_total = stat["gbw_caught"] + stat["gbw_missed"]
    corr_total = stat["correct_accept"] + stat["correct_falsereject"]
    print("=== diff-grounding verifier confusion ===")
    print(f"  GBW (ok_wrong_variant): caught={stat['gbw_caught']}/{gbw_total} "
          f"(recall={stat['gbw_caught']/max(gbw_total,1):.2f})  missed={stat['gbw_missed']}")
    print(f"  GOLD-CORRECT: false-reject={stat['correct_falsereject']}/{corr_total} "
          f"(should be ~0; FP rate={stat['correct_falsereject']/max(corr_total,1):.2f})")
    print(f"  other-kind={stat['other']}  unresolvable-variant={stat['no_variant']}")
    if examples["falsereject"]:
        print("  !! FALSE-REJECTS (verifier rejecting correct -> needs synonym map):")
        for e in examples["falsereject"]:
            print("     ", e)
    if examples["caught"]:
        print("  GBW caught examples:", examples["caught"][:3])
    return stat


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True, help="per-case dump jsonl from m_sigma_transfer_eval_v4 --dump")
    ap.add_argument("--cases", default="/home/woori/scratch/ma_eval_cases.jsonl")
    ap.add_argument("--synonyms", default="", help="optional ABox value->surface-forms json")
    args = ap.parse_args()
    syn = json.load(open(args.synonyms, encoding="utf-8")) if args.synonyms else None
    run(args.dump, args.cases, syn)
