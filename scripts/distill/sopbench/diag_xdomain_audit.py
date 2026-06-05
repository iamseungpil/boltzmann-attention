#!/usr/bin/env python3
"""Cross-domain readiness audit (§4 of CROSS_DOMAIN_TRANSFER_DESIGN) — zero-cost, GPU 전 첫 게이트.
Per domain:
  §4.1 DGGATE generality: reconstructed dirgraph (dfsgather_invfunccalldirgraph from constraints_original
        + domain rules, opt=full) vs task["directed_action_graph"] -> OVER/UNDER (Guard-2 generalized).
        OVER=0 over all tasks => DGGATE never over-requires in that domain (ship-safe).
  §4.3 login-arg: credential arg name(s) of login_user / authenticate_admin_password per domain.
  §4.2 getter_map coverage: size + predicates (VALFIX route presence; oracle-justification = manual follow-up).
directed_action_graph is comparison target ONLY (never a reconstruction input) -> not oracle."""
import json, collections, sys
sys.path.insert(0, "/home/woori/scratch/SOPBench")
from env.variables import domain_assistant_keys, domain_keys
import env.helpers as H

DOMAINS = ["bank","dmv","healthcare","hotel","library","online_market","university"]
GM = json.load(open("/home/woori/scratch/SOPBench/induced/getter_map.json"))

def node_key(n):
    if isinstance(n, str): return n
    name, args = n
    return (name, tuple(sorted((args or {}).items())))

def load_tasks(domain):
    d = json.load(open(f"/home/woori/scratch/SOPBench/data/{domain}_tasks.json"))
    out=[]
    if isinstance(d, dict):
        for v in d.values(): out += v if isinstance(v,list) else [v]
    else: out = d
    return [t for t in out if isinstance(t,dict) and "directed_action_graph" in t and "constraints_original" in t]

print(f"{'domain':<15}{'n':>5}{'OVER_nodes':>11}{'OVER_tasks':>11}{'UNDER_nodes':>12}{'UNDER_tasks':>12}  login_arg / admin")
print("="*95)
for dom in DOMAINS:
    da = domain_assistant_keys[dom]; ds = domain_keys[dom]()
    cl, cp = da.constraint_links, da.constraint_processes
    add = H.gather_action_default_dependencies(da.action_required_dependencies, da.action_customizable_dependencies, default_dependency_option="full")
    ap = H.get_action_parameters(ds, da)
    tasks = load_tasks(dom)
    over_n=under_n=over_t=under_t=0; errs=0
    for t in tasks:
        goal=t["user_goal"]
        if goal not in ap: errs+=1; continue
        try:
            rb=H.dfsgather_invfunccalldirgraph(t["constraints_original"], cl, cp, add, ap, (goal,{k:k for k in ap[goal]}))
            cr=collections.Counter(node_key(n) for n in rb["nodes"])
            ct=collections.Counter(node_key(n) for n in t["directed_action_graph"]["nodes"])
        except Exception:
            errs+=1; continue
        over=list((cr-ct).elements()); under=list((ct-cr).elements())
        over_n+=len(over); under_n+=len(under); over_t+=(1 if over else 0); under_t+=(1 if under else 0)
    # login arg names
    la = sorted(set(ap.get("login_user",{}).keys()) - {"username"}) if "login_user" in ap else []
    aa = sorted(set(ap.get("authenticate_admin_password",{}).keys()) - {"username"}) if "authenticate_admin_password" in ap else []
    tag = f"{la or 'NO-LOGIN'} / {aa or '-'}"
    flag = "" if over_n==0 else "  <<OVER!=0 (over-deny risk)"
    print(f"{dom:<15}{len(tasks):>5}{over_n:>11}{over_t:>11}{under_n:>12}{under_t:>12}  {tag}{flag}  errs={errs}")

print("\n=== getter_map coverage (VALFIX route; oracle-justification = manual S1) ===")
for dom in DOMAINS:
    m=GM.get(dom,{})
    print(f"  {dom:<15} getter_map entries={len(m)}  sample={list(m.keys())[:4]}")
print("\n판정: OVER=0 도메인 = DGGATE 구조적 일반(ship-safe). login_arg != 'identification' 도메인 = LOGINFIRST arg 일반화 필요(S2).")
