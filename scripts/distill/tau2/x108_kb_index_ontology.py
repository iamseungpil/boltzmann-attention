# -*- coding: utf-8 -*-
"""An index over the knowledge base, derived from its own contents, and the question of
whether referencing it would close the station where most failures stand.

The sweep's dominant failure is a tool the run never unlocked because it never learned the
name (30 of 43 at that station searched the knowledge base and the name did not come back).
The names are in the corpus — every one of them, in one to four documents. So the defect is
not that the knowledge is missing; it is that **ranked retrieval returns ten documents out of
698 and guarantees nothing**. A node can hold 47.

The proposal under test is completeness by construction: build an index from the corpus —
document → node, title, the tools it names — and let a step reference it instead of hoping the
ranker ranks. Two things have to be true for that to work, and only one of them is obvious:

  ① 색인이 답을 갖고 있는가        trivially yes if the name is in some document — measured anyway
  ② 그 답을 **무엇으로 찾는가**     the open question. `x53` measured node selection from the
                                   customer's first utterance at 6–12%, so a node-keyed index
                                   inherits that front door unless the key is something else.

So this derives the index and then measures ② under three candidate keys, in increasing order
of how much they assume:

  K1 노드-완결      the run's own searches touched some node; return that node's documents in full
  K2 도구-역인덱스  the step names a tool family in plain words; the index maps words → tools → docs
  K3 문서-제목      the agent's own query text against document titles (not node labels)

Everything here is derived from the environment's corpus and registry — no domain literal is
authored, so a new domain is the same code over a different corpus ([[05]] Q3).

  usage: x108_kb_index_ontology.py [--build out.json] [--measure tag]
"""

import argparse
import collections
import glob
import gzip
import io
import json
import os
import re
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
DOCS = os.environ.get(
    "T2_KB_DOCS_DIR",
    "/home/woori/scratch/tau2-bench/data/tau2/domains/banking_knowledge/documents")
SIM = ("/home/woori/workspace_common/boltzmann-attention-pi/"
       "reports/facet_rft_2026/sim_results")
STOP = set("the a an of for to and or in on with your you this that is are be by from it as "
           "please can if not do does when what which how use using".split())


def node_of(fname):
    """파일명이 인코딩한 2층 분류. 도메인 어휘를 우리가 적지 않는다 — 파일명에서 읽는다."""
    b = re.sub(r"\.json$", "", os.path.basename(fname))
    b = re.sub(r"^doc_", "", b)
    return re.sub(r"_\d+$", "", b)


def load_docs():
    out = []
    for p in sorted(glob.glob(os.path.join(DOCS, "*.json"))):
        try:
            d = json.load(io.open(p, encoding="utf-8"))
        except Exception:
            continue
        txt = d if isinstance(d, str) else json.dumps(d, ensure_ascii=False)
        title = ""
        if isinstance(d, dict):
            title = str(d.get("title") or d.get("name") or "")
            if not title:
                m = re.search(r"#\s*([^\n\"]{4,80})", txt)
                title = m.group(1).strip() if m else ""
        out.append({"id": os.path.basename(p), "node": node_of(p), "title": title, "text": txt})
    return out


def registry_names():
    """도구명 어휘 = env 레지스트리(닫힌 집합). 철자 규칙으로 뽑지 않는다([[22]])."""
    p = os.path.join(HERE, "a2", "env_surface.json")
    d = (json.load(io.open(p, encoding="utf-8")) or {}).get("banking_knowledge") or {}
    return set(d.get("tools") or {}) | set(d.get("exposed") or []) | \
        set(d.get("discoverable_user_tools") or [])


def build():
    docs = load_docs()
    names = registry_names()
    idx = {"documents": {}, "by_node": collections.defaultdict(list),
           "by_tool": collections.defaultdict(list), "nodes": {}}
    for d in docs:
        hit = sorted(n for n in names
                     if re.search(r"(?<![A-Za-z0-9_])%s(?![A-Za-z0-9_])" % re.escape(n), d["text"]))
        idx["documents"][d["id"]] = {"node": d["node"], "title": d["title"], "tools": hit}
        idx["by_node"][d["node"]].append(d["id"])
        for n in hit:
            idx["by_tool"][n].append(d["id"])
    for n, ids in idx["by_node"].items():
        idx["nodes"][n] = {"n_docs": len(ids),
                           "tools": sorted({t for i in ids for t in idx["documents"][i]["tools"]})}
    idx["by_node"] = dict(idx["by_node"])
    idx["by_tool"] = dict(idx["by_tool"])
    return idx


def report_build(idx):
    print("== §1 색인 (코퍼스에서 기계 도출) ==")
    print("  문서 %d · 노드 %d · 도구를 언급하는 문서 %d"
          % (len(idx["documents"]), len(idx["by_node"]),
             sum(1 for v in idx["documents"].values() if v["tools"])))
    print("  색인된 도구 %d종 · 도구당 문서 수 중앙값 %s"
          % (len(idx["by_tool"]),
             sorted(len(v) for v in idx["by_tool"].values())[len(idx["by_tool"]) // 2]
             if idx["by_tool"] else 0))
    big = sorted(idx["nodes"].items(), key=lambda kv: -kv[1]["n_docs"])[:5]
    print("  큰 노드: " + " · ".join("%s(%d문서·도구%d)" % (k[:34], v["n_docs"], len(v["tools"]))
                                   for k, v in big))
    only1 = [t for t, v in idx["by_tool"].items() if len(v) == 1]
    print("  ★단 하나의 문서에만 나오는 도구 %d종 — 랭커가 그 한 건을 놓치면 이름을 못 얻는다"
          % len(only1))


def measure(idx, tag):
    """실패 궤적에 색인을 대입: 세 열쇠(K1/K2/K3)가 gold 도구명을 실제로 짚었겠는가."""
    files = sorted(glob.glob(os.path.join(SIM, "bank_n97_gpu*_%s.results.json.gz" % tag)))
    if not files:
        files = sorted(glob.glob(os.path.join(
            HERE, "..", "..", "..", "reports", "facet_rft_2026", "sim_results",
            "bank_n97_gpu*_%s.results.json.gz" % tag)))
    doc_re = re.compile(r"ID:\s*(doc_[A-Za-z0-9_()\-]+)")
    res = collections.Counter()
    rows = []
    for f in files:
        d = json.load(gzip.open(f, "rt", encoding="utf-8"))
        for s in d["simulations"]:
            ri = s["reward_info"]
            if ri["reward"] == 1.0:
                continue
            acts = ri.get("action_checks") or []
            k = next((i for i, a in enumerate(acts) if not a["action_match"]), None)
            if k is None or acts[k]["action"]["name"] != "unlock_discoverable_agent_tool":
                continue
            gold = (acts[k]["action"]["arguments"] or {}).get("agent_tool_name")
            if not gold:
                continue
            gold_docs = idx["by_tool"].get(gold) or []
            seen_docs, queries = set(), []
            for m in s["messages"]:
                for tc in (m.get("tool_calls") or []):
                    if str(tc.get("name", "")).startswith("KB_search"):
                        a = tc.get("arguments")
                        if isinstance(a, str):
                            try:
                                a = json.loads(a)
                            except Exception:
                                a = {}
                        queries.append(str((a or {}).get("query") or ""))
                if m.get("role") == "tool":
                    for did in doc_re.findall(str(m.get("content") or "")):
                        seen_docs.add(did if did.endswith(".json") else did + ".json")
            seen_nodes = {idx["documents"][d0]["node"] for d0 in seen_docs if d0 in idx["documents"]}
            gold_nodes = {idx["documents"][d0]["node"] for d0 in gold_docs if d0 in idx["documents"]}
            k1 = bool(seen_nodes & gold_nodes)                     # 노드-완결 반환이면 닿는다
            qtok = {w for q in queries for w in re.findall(r"[a-z]{3,}", q.lower())} - STOP
            gtok = set(re.findall(r"[a-z]{3,}", gold.lower())) - STOP
            k2 = bool(qtok & gtok)                                 # 질의어가 도구 이름의 말과 겹치나
            k3 = any(qtok & ((set(re.findall(r"[a-z]{3,}",
                                             (idx["documents"].get(d0) or {}).get("title", "").lower()))
                              - STOP)) for d0 in gold_docs)
            res[("K1", k1)] += 1
            res[("K2", k2)] += 1
            res[("K3", k3)] += 1
            rows.append((s["task_id"], s.get("trial"), gold, len(gold_docs), k1, k2, k3))
    print("\n== §2 반사실 — 색인을 어떤 열쇠로 여는가 (unlock 역 실패 %d건) ==" % len(rows))
    for key, label in (("K1", "노드-완결(그 run이 이미 닿은 노드의 문서를 전부 반환)"),
                       ("K2", "도구-역인덱스(질의어 ∩ 도구명 어휘)"),
                       ("K3", "문서-제목(질의어 ∩ gold 문서 제목)")):
        hit = res[(key, True)]
        tot = hit + res[(key, False)]
        print("  %-4s %-46s %3d/%3d = %.0f%%" % (key, label, hit, tot, 100.0 * hit / max(tot, 1)))
    print("\n  (표본) 태스크별:")
    for t, tr, g, nd, k1, k2, k3 in sorted(rows)[:14]:
        print("    %-10s t%s  gold=%-40s 문서%d  K1=%s K2=%s K3=%s"
              % (t, tr, g, nd, "Y" if k1 else "N", "Y" if k2 else "N", "Y" if k3 else "N"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", default=None, help="색인을 이 경로에 저장")
    ap.add_argument("--measure", default=None, help="이 태그의 실패에 색인을 대입")
    a = ap.parse_args()
    idx = build()
    report_build(idx)
    if a.build:
        json.dump(idx, io.open(a.build, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print("  → 저장 %s" % a.build)
    if a.measure:
        measure(idx, a.measure)


if __name__ == "__main__":
    main()
