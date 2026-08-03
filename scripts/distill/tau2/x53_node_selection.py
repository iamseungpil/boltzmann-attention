"""Can the right node be chosen, and would choosing it have helped?

The corpus is 698 documents — far past the context window — but the filenames already
encode a two-level classification, so "have I read everything relevant" (open, endless)
can be restated as "have I read node X's nine documents" (closed, countable) at zero
authoring cost. The unresolved part is the step before that: node selection is still an
open predicate, merely bounded to a 71-way choice.

This measures the step rather than assuming it:

  §1  the node structure, mechanically from filenames
  §2  for each failure, which node holds what the run needed
  §3  which nodes the agent's own searches actually landed in
  §4  whether a deterministic selector over node labels would have picked the right one

§4 is the make-or-break. If the label of the right node cannot be picked from the
customer's own words, the 71-way choice is not obviously easier than the 698-way one,
and the completeness claim rests on a step that has not been shown to work.
"""

import argparse
import collections
import glob
import gzip
import json
import math
import os
import re

SIM = ("/home/woori/workspace_common/boltzmann-attention-pi/"
       "reports/facet_rft_2026/sim_results")
DOCS = "/home/woori/scratch/tau2-bench/data/tau2/domains/banking_knowledge/documents"

ARMS = {
    "A":  "bank_ax33n_gpu*_20260803g",
    "B4": "bank_b4_gpu*_20260803h",
}

STOP = set(
    "a an the of for for to in on at by with and or is are was were be been being do "
    "does did how what when where which who this that these those i you we they it my "
    "your our can could should would will shall may might must if then than there here "
    "about from into over under after before during any all each card cards".split())


def norm(s):
    return re.sub(r"[^a-z0-9 ]", " ", str(s or "").lower())


def node_of(fname):
    """(category, subcategory) straight from the filename — no authoring, no judgement.

    Parsed by splitting, not by a regex: the obvious pattern for this shape
    (`doc_(cat)_(sub)_(\\d+)`) has two adjacent unbounded groups over the same
    alphabet and backtracks catastrophically on the parenthesised names — it hung
    for minutes at full CPU on this very corpus, twice.
    """
    stem = fname[4:].rsplit(".", 1)[0].rsplit("_", 1)[0]   # drop 'doc_', '.json', '_017'
    return stem.split("_")[0], stem


def load_corpus():
    docs = {}
    for p in sorted(glob.glob(os.path.join(DOCS, "*"))):
        f = os.path.basename(p)
        try:
            with open(p, encoding="utf-8", errors="replace") as fh:
                docs[f] = fh.read()
        except Exception:
            pass
    return docs


def sims(pattern):
    out = []
    for p in sorted(glob.glob(f"{SIM}/{pattern}.results.json.gz")):
        out.extend(json.load(gzip.open(p, "rt", encoding="utf-8")).get("simulations") or [])
    return out


def norm_args(a):
    if isinstance(a, str):
        try:
            return json.loads(a)
        except Exception:
            return {}
    return a if isinstance(a, dict) else {}


def fam(n):
    return re.sub(r"_\d{3,4}$", "", n or "")


def needed_tokens(sim):
    """What the run had to know and did not produce: gold calls it never made."""
    called = set()
    for m in sim.get("messages") or []:
        if m.get("role") != "assistant":
            continue
        for tc in m.get("tool_calls") or []:
            n = tc.get("name") or (tc.get("function") or {}).get("name")
            args = norm_args(tc.get("arguments") if tc.get("arguments") is not None
                             else (tc.get("function") or {}).get("arguments"))
            called.add(fam(n))
            inner = (args.get("agent_tool_name") or args.get("discoverable_tool_name")
                     or args.get("user_tool_name"))
            if inner:
                called.add(fam(inner))
    out = []
    for c in (sim.get("reward_info") or {}).get("action_checks") or []:
        a = c.get("action") or {}
        if c.get("action_match") or a.get("requestor") != "assistant":
            continue
        if fam(a.get("name")) not in called:
            out.append(a.get("name"))
    return out


def searched_docs(sim):
    """Document ids the run's KB searches actually returned."""
    out = set()
    for m in sim.get("messages") or []:
        if m.get("role") == "tool" and isinstance(m.get("content"), str):
            out |= set(re.findall(r"ID:\s*(doc_[a-z_()0-9]+)", m["content"]))
    return out


def first_utterance(sim):
    for m in sim.get("messages") or []:
        if m.get("role") == "user" and isinstance(m.get("content"), str) and m["content"].strip():
            return m["content"]
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="B4", choices=sorted(ARMS))
    args = ap.parse_args()

    docs = load_corpus()
    nodes = collections.defaultdict(list)
    for f in docs:
        nodes[node_of(f)].append(f)

    print(f"§1 구조 — 문서 {len(docs)} · 2층 노드 {len(nodes)}")
    sizes = sorted(len(v) for v in nodes.values())
    print(f"   노드당 문서: 최소 {sizes[0]} · 중앙 {sizes[len(sizes) // 2]} · 최대 {sizes[-1]}")
    label_tokens = {k: set(norm(f"{k[0]} {k[1]}").split()) - STOP for k in nodes}
    print(f"   노드 라벨 총 토큰 {sum(len(v) for v in label_tokens.values())}")

    # Which node documents a given tool name / value.
    doc_node = {f: node_of(f) for f in docs}
    lower = {f: t.lower() for f, t in docs.items()}

    run = sims(ARMS[args.arm])
    fails = [s for s in run if ((s.get("reward_info") or {}).get("reward") or 0.0) != 1.0]

    print(f"\n§2 실패 {len(fails)}건이 필요로 한 것이 어느 노드에 있나")
    reach, unreachable, spread = [], [], []
    for s in sorted(fails, key=lambda x: (x["task_id"], x.get("trial") or 0)):
        toks = needed_tokens(s)
        if not toks:
            continue
        key = f"{s['task_id']}/t{s.get('trial')}"
        got = searched_docs(s)
        got_nodes = {doc_node[d + ".json"] for d in got if d + ".json" in doc_node}
        for tok in toks:
            hits = [f for f, t in lower.items() if tok.lower() in t]
            hn = {doc_node[f] for f in hits}
            if not hits:
                unreachable.append((key, tok))
                continue
            hit_node = sorted(hn)[0]
            (reach if len(hn) == 1 else spread).append((key, tok, hn, hit_node in got_nodes))
            print(f"  {key:16s} {tok:44s} 노드 {len(hn)}개"
                  f"{' ★단일' if len(hn) == 1 else ''}"
                  f"  검색이 그 노드에 닿았나: {'YES' if hit_node in got_nodes else 'NO'}"
                  f"  ({sorted(hn)[0][0]}/{sorted(hn)[0][1]})")

    print(f"\n  단일 노드에 있음 {len(reach)} · 여러 노드에 흩어짐 {len(spread)} · "
          f"코퍼스에 아예 없음 {len(unreachable)}")
    if unreachable:
        print(f"  코퍼스 부재(회수로 못 얻음): {unreachable}")
    touched = sum(1 for _, _, _, t in reach + spread if t)
    print(f"  ★검색이 정답 노드에 실제로 닿은 비율: {touched}/{len(reach) + len(spread)}")

    # A node's name is three or four words; a customer describes their problem in
    # their own. Terms that are frequent inside a node and rare outside it are the
    # node's own vocabulary, and they are derived from the corpus, not authored —
    # so declaring them costs nothing and stays clear of gold ([[23]]).
    print("\n§5 노드 특성 — 파일명 라벨 + 코퍼스에서 기계 도출한 변별어")
    node_terms = {}
    node_df = collections.Counter()
    for k, files in nodes.items():
        c = collections.Counter()
        for f in files:
            c.update(set(norm(docs[f]).split()) - STOP)
        node_terms[k] = c
        for t in c:
            node_df[t] += 1
    n_nodes = len(nodes)
    descriptors = {}
    for k, c in node_terms.items():
        scored = sorted(c, key=lambda t: -(c[t] / len(nodes[k]))
                        * math.log(n_nodes / (1 + node_df[t])))
        descriptors[k] = [t for t in scored if len(t) > 3][:12]
    for k in list(sorted(nodes, key=lambda x: -len(nodes[x])))[:6]:
        print(f"  {k[1][:44]:46s} n={len(nodes[k]):2d}  {', '.join(descriptors[k][:8])}")
    total_desc = sum(len(v) for v in descriptors.values())
    print(f"  전체 선언 크기: 노드 {n_nodes} × 변별어 ≤12 = {total_desc} 토큰"
          f" (라벨 포함 ≈{total_desc + sum(len(v) for v in label_tokens.values())})")

    print("\n§4 결정론 선택기 — 손님 첫 발화의 단어로 노드를 고를 수 있나")
    # Plain idf-weighted overlap between the customer's words and the node label.
    df = collections.Counter()
    for toks in label_tokens.values():
        for t in toks:
            df[t] += 1
    ok = ok_desc = tried = 0
    for s in sorted(fails, key=lambda x: (x["task_id"], x.get("trial") or 0)):
        toks = needed_tokens(s)
        if not toks:
            continue
        want = set()
        for tok in toks:
            for f, t in lower.items():
                if tok.lower() in t:
                    want.add(doc_node[f])
        if not want:
            continue
        q = set(norm(first_utterance(s)).split()) - STOP

        def rank(keyset):
            return sorted(nodes, key=lambda k: -sum(
                math.log(n_nodes / (1 + df[t])) for t in (keyset(k) & q)))

        by_label = rank(lambda k: label_tokens[k])[:3]
        by_both = rank(lambda k: label_tokens[k] | set(descriptors[k]))[:3]
        tried += 1
        h_label = any(k in want for k in by_label)
        h_both = any(k in want for k in by_both)
        ok += h_label
        ok_desc += h_both
        print(f"  {s['task_id']}/t{s.get('trial'):<2} 정답 노드 {len(want):2d}개  "
              f"라벨만 {'HIT ' if h_label else 'MISS'}  라벨+변별어 "
              f"{'HIT ' if h_both else 'MISS'}  top3={[k[1][:22] for k in by_both]}")
    print(f"\n  ★첫 발화 → top-3 안에 정답 노드  라벨만 {ok}/{tried}"
          + (f" = {ok / tried:.0%}" if tried else "")
          + f"   라벨+변별어 {ok_desc}/{tried}"
          + (f" = {ok_desc / tried:.0%}" if tried else ""))


if __name__ == "__main__":
    main()
