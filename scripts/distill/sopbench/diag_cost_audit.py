#!/usr/bin/env python3
"""L0 COST AUDIT (FIELD_GAP_LLM_VALUE_DESIGN §11.1) — GPU-free.

Quantifies the "deterministic program per-domain authoring cost" vs our approach, to
(a) give §1 a concrete rebuttal and (b) force the §10-1 found/authored/inherited classification.

Deterministic per-domain procedural authoring = env/domains/<d>/{<d>.py(backend) , <d>_assistant.py(SOP
encoding: actions+constraints+directed graphs)}.  Our ABox = induced/ontology_<d>.json + getter_map.json.

★Δ-LOC-per-change proxy = det procedural LOC / (#actions + #constraints) = ~lines a deterministic
program must edit to add one operator/condition (the real change-axis bullet, not static total).

★CRITICAL classification (read induce_ontology_zekun.py inputs): our ontology is INDUCED FROM the
benchmark's STRUCTURED artifacts (domain_assistant_keys, each task's directed_action_graph, dep_full)
— NOT from NL policy text. => our "0 authored LOC" is INHERITED (benchmark did NL->structure), not
FOUND. This is printed explicitly; it gates E1 (NL-only) honesty.
"""
import json, sys
CL = "/home/woori/scratch/SOPBench"
IND = f"{CL}/induced"
sys.path.insert(0, CL)
DOMAINS = ["bank", "dmv", "healthcare", "hotel", "library", "online_market", "university"]

def loc(p):
    try: return sum(1 for _ in open(p, encoding="utf-8", errors="ignore"))
    except Exception: return -1

def sized(v): return len(v) if hasattr(v, "__len__") else v

# structured per-domain spec (same source induce reads)
try:
    from env.variables import domain_assistant_keys
except Exception as e:
    print("WARN import domain_assistant_keys:", e); domain_assistant_keys = {}

# show the structure of one domain's assistant_keys so #constraints/#actions counting is grounded
if "bank" in domain_assistant_keys:
    A = domain_assistant_keys["bank"]
    print("=== domain_assistant_keys['bank'] structure ===")
    if isinstance(A, dict):
        for k, v in A.items():
            print(f"   {k:<24} -> {type(v).__name__} (len={sized(v)})")
    else:
        print("   type:", type(A).__name__)
    print()

def n_actions_constraints(d):
    """best-effort: #agent-callable actions + #constraint links from structured spec."""
    A = domain_assistant_keys.get(d)
    nact = ncon = -1
    if isinstance(A, dict):
        for key in ("default_dep", "dep", "actions", "action_dep"):
            if key in A and hasattr(A[key], "__len__"): nact = len(A[key]); break
        for key in ("constraint_links", "constraints", "constraint", "links"):
            if key in A and hasattr(A[key], "__len__"): ncon = len(A[key]); break
    return nact, ncon

print(f"{'domain':<15}{'backend.py':>11}{'assist.py':>11}{'det_LOC':>9}{'#act':>6}{'#con':>6}{'LOC/unit':>9}{'ont_act':>9}{'ont_KB':>8}")
print("-" * 92)
tot_det = tot_assist = 0
for d in DOMAINS:
    bp = f"{CL}/env/domains/{d}/{d}.py"
    ap = f"{CL}/env/domains/{d}/{d}_assistant.py"
    l1, l2 = loc(bp), loc(ap)
    det = (l1 if l1 > 0 else 0) + (l2 if l2 > 0 else 0)
    nact, ncon = n_actions_constraints(d)
    # fall back to induced ontology for #actions if structured count failed
    ont_act = -1; ont_kb = -1
    try:
        ont = json.load(open(f"{IND}/ontology_{d}.json")); ont_act = len(ont.get("actions", []))
        import os; ont_kb = os.path.getsize(f"{IND}/ontology_{d}.json") // 1024
    except Exception: pass
    if nact < 0: nact = ont_act
    units = (nact if nact > 0 else 0) + (ncon if ncon > 0 else 0)
    lpu = round(l2 / units, 1) if units > 0 else -1   # Δ-proxy uses assistant(SOP) LOC only
    tot_det += det; tot_assist += (l2 if l2 > 0 else 0)
    print(f"{d:<15}{l1:>11}{l2:>11}{det:>9}{nact:>6}{ncon:>6}{lpu:>9}{ont_act:>9}{ont_kb:>8}")
print("-" * 92)
print(f"{'TOTAL':<15}{'':>11}{'':>11}{tot_det:>9}")
print(f"\n7-domain deterministic procedural authoring (backend+assistant) = {tot_det} LOC")
print(f"   assistant.py only (pure SOP encoding) = {tot_assist} LOC")

# getter_map
try:
    gm = json.load(open(f"{IND}/getter_map.json"))
    if isinstance(gm, dict):
        per = {k: (len(v) if hasattr(v, "__len__") else 1) for k, v in gm.items()}
        print(f"\ngetter_map.json: top-level keys={list(gm.keys())[:8]}{'...' if len(gm)>8 else ''} (n={len(gm)})")
except Exception as e:
    print("getter_map read err:", e)

print("\n=== ★FOUND vs AUTHORED vs INHERITED classification (§10-1) ===")
print("  ontology_<d>.json : INHERITED  (induce_ontology_zekun reads domain_assistant_keys +")
print("                       each task's directed_action_graph + dep_full = STRUCTURED bench")
print("                       artifacts, NOT NL policy text). => 'our 0 authored LOC' borrows the")
print("                       benchmark's NL->structure work. E1(NL-only) BLOCKED until an")
print("                       NL-policy-source induce path exists.")
print("  getter_map.json   : AUTO-DERIVED from predicate source (autoderive_getter_map.py) — check")
print("                       if predicate source = structured (inherited) or signatures (found).")
print("  data/<d>_tasks    : benchmark-provided (shared by both approaches).")
print("\n=== interpretation ===")
print("  Honest claim is NOT 'we author 0 from NL' (induce is inherited). It IS: 'a DEPLOYABLE")
print("  deterministic SOP-executor needs ~{} LOC/domain of hand-authored procedure; whether an".format(round(tot_assist/7)))
print("  LLM can recover that from NL (vs inherited structure) is exactly E1, gated by the")
print("  found/inherited separation.' Static LOC is the floor; the change-axis (delta-LOC per")
print("  added condition ~= LOC/unit column) is the amortization bullet (§9).")
