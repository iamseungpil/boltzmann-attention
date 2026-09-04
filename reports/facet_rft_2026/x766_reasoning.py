# x766g — locate the "Which user_id" reasoning verbatim inside the ctl bundle and print
# the enclosing message coordinates + surrounding text.
import gzip, json, os, re

ROOT = r"C:\workspace\ba-frft\reports\facet_rft_2026\sim_results"
FN = "bank_010ctl_20260904_0007.results.json.gz"
NEEDLE = "Which user_id"

with gzip.open(os.path.join(ROOT, FN), "rb") as fh:
    obj = json.load(fh)


def walk(node, path):
    if isinstance(node, str):
        if NEEDLE in node:
            yield path, node
    elif isinstance(node, dict):
        for k, v in node.items():
            yield from walk(v, path + [str(k)])
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from walk(v, path + [str(i)])


for s in obj["simulations"]:
    if s.get("task_id") != "task_010":
        continue
    for i, m in enumerate(s.get("messages") or []):
        for path, txt in walk(m, []):
            k = txt.find(NEEDLE)
            print("=" * 100)
            print("seed=%s trial=%s  msg_index=%d  role=%s  field_path=%s  field_len=%d"
                  % (s.get("seed"), s.get("trial"), i, m.get("role"), "/".join(path), len(txt)))
            print("--- verbatim window ---")
            print(txt[max(0, k - 1200): k + 900])
