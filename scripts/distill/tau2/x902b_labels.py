# -*- coding: utf-8 -*-
"""x902b — M1: _record_labels/_LABEL_RE(:1088) → _label_mismatch_deny 실측."""
import sys, os, json, gzip, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import t2_gate_patch as G
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "x902_a1_audit.py"),
          encoding="utf-8").read().split("sims = json.loads")[0].split('CORP = sys.argv[1]')[1])
CORP = sys.argv[1]
sims = json.loads(gzip.open(CORP, "rt", encoding="utf-8").read())
for s in sims:
    s["M"] = [M(d) for d in s["msgs"]]

a2 = G._domain_a2("banking_knowledge")
asr = {k: v for k, v in ((a2 or {}).get("arg_source_reads") or {}).items()
       if not k.startswith("_") and isinstance(v, list)}
print("arg_source_reads keys:", sorted(asr)[:40], "n=", len(asr))

# ── ①`_LABEL_RE` 가 덤프에서 뽑는 '필드 이름'이 실제로 필드인가 ─────────────
allfields = {}
nonfield_examples = []
for s in sims:
    labs = G._record_labels(Orch(s["M"]))
    for f, vs in labs.items():
        allfields.setdefault(f, set()).update(vs)
print("\n[M1-a] 덤프에서 뽑힌 서로 다른 '필드 이름' 수 =", len(allfields))
# env 가 실제로 쓰는 필드 이름 후보 = A2 arg_source_reads 키 ∪ env tool args
env = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "a2", "env_surface.json"), encoding="utf-8"))
envargs = set()
for t, v in env["banking_knowledge"]["tools"].items():
    envargs |= set(v.get("args") or [])
known = envargs | set(asr)
unknown = sorted(f for f in allfields if f not in known)
print("[M1-b] env 인자명/A2 키에 없는 '필드' =", len(unknown), "/", len(allfields))
print("        표본:", unknown[:40])
for f in unknown[:12]:
    print("   %-28s vals=%s" % (f, sorted(allfields[f])[:4]))
