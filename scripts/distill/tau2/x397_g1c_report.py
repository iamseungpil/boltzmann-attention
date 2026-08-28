# -*- coding: utf-8 -*-
r"""x397 G1-c 보고 - 인수 축자 노출률 + 팔 정보량(길이/레코드 id). LLM 0."""
import io, json, os, re, sys, statistics as st
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import x397_argexposure as G

ARMS = ["A_min", "C_neg", "B_tail4", "B_tail8", "B_tail16", "B_tail32", "B_full"]
LEDGER_RE = re.compile(r"\b((?:chk|sav|dbc|txn|cc|acc)_[A-Za-z0-9_]+)\b")
BROAD_RE = re.compile(r"\b((?:b?txn|chk|sav|dbc|cc|acc|ccx)_[A-Za-z0-9_]+)\b")
HEX_RE = re.compile(r"\b([0-9a-f]{10,16})\b")

try:
    from transformers import AutoTokenizer
    TOK = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-32B-Instruct", local_files_only=True)
except Exception as e:
    TOK = None
    sys.stderr.write("tokenizer unavailable: %s\n" % e)


def ntok(s):
    return len(TOK(s)["input_ids"]) if TOK else -1


def klass(key):
    k = key.lower()
    if k.endswith("_id") or k == "id":
        return "ID"
    if "amount" in k:
        return "AMOUNT"
    if "date" in k:
        return "DATE"
    return "ENUM"


def gold_values(gold_args):
    vals, seen = [], set()
    for g in gold_args:
        raw = g["outer"].get("arguments")
        if isinstance(raw, str):
            try:
                d = json.loads(raw)
            except Exception:
                continue
        elif isinstance(raw, dict):
            d = raw
        else:
            continue
        for k, v in d.items():
            if isinstance(v, bool) or v is None:
                continue
            if isinstance(v, (int, float)):
                s = ("%g" % v)
            else:
                s = str(v).strip()
                if not s:
                    continue
            key = (k, s)
            if key in seen:
                continue
            seen.add(key)
            vals.append((k, s, v, klass(k)))
    return vals


def present(prompt, s, v):
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        cands = {("%g" % v)}
        if float(v) == int(float(v)):
            cands |= {str(int(float(v))), "%.1f" % v, "%.2f" % v}
        else:
            cands |= {"%.2f" % v}
        return any(re.search(r"(?<![\w.])" + re.escape(c) + r"(?![\w.])", prompt) for c in cands)
    return s in prompt


def main():
    docs, TOOLS, cases = G.build_cases(12)
    rows = []
    for c in cases:
        ga = G.gold_args_for(c["sim"], c["tool"])
        P = G.build_prompts(c, TOOLS)
        rows.append({"task": c["task"], "trial": c["trial"], "tool": c["tool"],
                     "vals": gold_values(ga), "P": P, "lines": c["lines"]})

    print("## TABLE1 arg-verbatim exposure")
    hdr = "%-9s %-38s %4s %4s | " % ("task", "target_tool", "vals", "ID")
    hdr += " ".join("%-12s" % a for a in ARMS)
    print(hdr)
    agg = {a: [0, 0] for a in ARMS}
    aggid = {a: [0, 0] for a in ARMS}
    for r in rows:
        vs = r["vals"]
        nid = sum(1 for v in vs if v[3] == "ID")
        cells = []
        for a in ARMS:
            p = r["P"][a]
            hit = sum(1 for (k, s, v, cl) in vs if present(p, s, v))
            hitid = sum(1 for (k, s, v, cl) in vs if cl == "ID" and present(p, s, v))
            agg[a][0] += hit
            agg[a][1] += len(vs)
            aggid[a][0] += hitid
            aggid[a][1] += nid
            cells.append("%-12s" % ("%d/%d=%.2f" % (hit, len(vs), (hit / float(len(vs))) if vs else 0)))
        print("%-9s %-38s %4d %4d | %s" % (r["task"], r["tool"][:38], len(vs), nid, " ".join(cells)))
    print("%-9s %-38s %4d %4d | %s" % ("ALL", "(all arg values)", agg[ARMS[0]][1], aggid[ARMS[0]][1],
          " ".join("%-12s" % ("%d/%d=%.2f" % (agg[a][0], agg[a][1], agg[a][0] / float(agg[a][1]))) for a in ARMS)))
    print("%-9s %-38s %4s %4s | %s" % ("ALL", "(ID args only)", "", "",
          " ".join("%-12s" % ("%d/%d=%.2f" % (aggid[a][0], aggid[a][1], aggid[a][0] / float(max(1, aggid[a][1])))) for a in ARMS)))

    print("")
    print("## TABLE1b targets fully exposed (all gold arg values present) /12")
    print(" ".join("%s=%d" % (a, sum(1 for r in rows if r["vals"] and all(present(r["P"][a], s, v) for (k, s, v, cl) in r["vals"]))) for a in ARMS))

    print("")
    print("## TABLE2 prompt information volume (12 targets, median[min-max])")
    print("%-9s %16s %16s %14s %14s %14s %7s" % ("arm", "chars", "tokens", "ledgerID", "broadID", "hexID", "name"))
    for a in ARMS:
        ch = [len(r["P"][a]) for r in rows]
        tk = [ntok(r["P"][a]) for r in rows]
        lid = [len(set(LEDGER_RE.findall(r["P"][a]))) for r in rows]
        bid = [len(set(BROAD_RE.findall(r["P"][a]))) for r in rows]
        hid = [len(set(HEX_RE.findall(r["P"][a]))) for r in rows]
        nm = 0
        for r in rows:
            body = r["P"][a].split("\n\n", 1)[1] if "\n\n" in r["P"][a] else ""
            if r["tool"] in body.split("# 질문")[0]:
                nm += 1
        print("%-9s %16s %16s %14s %14s %14s %7d"
              % (a, "%d[%d-%d]" % (st.median(ch), min(ch), max(ch)),
                 "%d[%d-%d]" % (st.median(tk), min(tk), max(tk)),
                 "%.1f[%d-%d]" % (sum(lid) / float(len(lid)), min(lid), max(lid)),
                 "%.1f[%d-%d]" % (sum(bid) / float(len(bid)), min(bid), max(bid)),
                 "%.1f[%d-%d]" % (sum(hid) / float(len(hid)), min(hid), max(hid)), nm))

    print("")
    print("## TABLE2b component lengths (median chars)")

    def med(f):
        return st.median([f(r["P"]["_parts"]) for r in rows])
    print("tools=%d ask=%d ledger=%d proc=%d filler=%d convo_full=%d tail4=%d tail8=%d tail16=%d tail32=%d"
          % (med(lambda p: len(p["tools"])), med(lambda p: len(p["ask"])), med(lambda p: len(p["led"])),
             med(lambda p: len(p["proc"])), med(lambda p: len(p["neg"])), med(lambda p: len(p["convo_full"])),
             med(lambda p: len(p["convo_tail"][4])), med(lambda p: len(p["convo_tail"][8])),
             med(lambda p: len(p["convo_tail"][16])), med(lambda p: len(p["convo_tail"][32]))))

    print("")
    print("## C_neg vs A_min")
    for r in rows:
        A, C = r["P"]["A_min"], r["P"]["C_neg"]
        print("  %-9s A=%d C=%d proc=%d filler=%d prefix_identical=%s"
              % (r["task"], len(A), len(C), len(r["P"]["_parts"]["proc"]), len(r["P"]["_parts"]["neg"]),
                 A.split("# 정책 절차")[0] == C.split("# 안내")[0]))
    print("")
    print("## proc lines containing target tool name verbatim: %d/%d"
          % (sum(1 for r in rows if r["tool"] in r["P"]["_parts"]["proc"]), len(rows)))
    print("## proc lines containing gold arg values: %s"
          % ("; ".join("%s %d/%d" % (r["task"], sum(1 for (k, s, v, cl) in r["vals"] if present(r["P"]["_parts"]["proc"], s, v)), len(r["vals"])) for r in rows)))

    print("")
    print("## TABLE2c does the A_min ledger carry the gold ID args?")
    for r in rows:
        led = r["P"]["_parts"]["led"]
        ids = [(k, s) for (k, s, v, cl) in r["vals"] if cl == "ID"]
        got = [k for (k, s) in ids if s in led]
        print("  %-9s %-38s IDargs=%d in_ledger=%d %s"
              % (r["task"], r["tool"][:38], len(ids), len(got), ",".join(sorted(set(got)))))
    io.open("/home/woori/scratch/x397_g1c_rows.json", "w", encoding="utf-8").write(
        json.dumps([{"task": r["task"], "tool": r["tool"],
                     "vals": [(k, s, cl) for (k, s, v, cl) in r["vals"]],
                     "expose": {a: [bool(present(r["P"][a], s, v)) for (k, s, v, cl) in r["vals"]] for a in ARMS},
                     "chars": {a: len(r["P"][a]) for a in ARMS}} for r in rows], ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
