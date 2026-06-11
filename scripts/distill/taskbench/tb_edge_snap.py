#!/usr/bin/env python
"""edge-snap v0 (post-hoc, zero-GPU): canonicalize value-inlined references into
<node-j> tags (A-0 ~27% 관례분 중 값-인라이닝 하위유형의 회수 — name-snap의 edge판).

Conservative rules (over-rewrite 방지가 1순위):
  node i를 재작성하는 조건 전부:
  1. node i의 어느 인자에도 <node-> 태그가 없음 (이미 참조하면 불간섭)
  2. tool(i)가 tool_desc에 있고 input-type이 정확히 1종 (단일-입력-슬롯)
  3. 후보 인자 = user_request에 등장하지 않는 비어있지 않은 str 인자 — 정확히 1개
     (요청문에 있으면 사용자-제공 literal로 간주, 불간섭)
  4. 앞쪽 노드(j<i) 중 output-type이 tool(i)의 input-type과 교차하는 것이 정확히 1개
  ⇒ 그 인자를 "<node-j>"로 재작성. 그 외 전부 skip(사유 카운트).

Usage: python tb_edge_snap.py --pred <pred.json> --tool_desc <tool_desc.json> --out <out.json>
"""
import argparse, json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--tool_desc", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    tools = {t["id"]: t for t in json.load(open(a.tool_desc))["nodes"]}

    stats = {"nodes": 0, "rewritten": 0, "has_tag": 0, "no_tool": 0,
             "multi_input": 0, "cand_args!=1": 0, "producer!=1": 0}
    n_rows = 0
    with open(a.out, "w") as wf:
        for l in open(a.pred):
            d = json.loads(l)
            n_rows += 1
            res = d.get("result", {})
            nodes = res.get("task_nodes") if isinstance(res, dict) else None
            req = str(d.get("user_request", ""))
            if isinstance(nodes, list):
                for i, node in enumerate(nodes):
                    if not (isinstance(node, dict) and "task" in node):
                        continue
                    stats["nodes"] += 1
                    args_ = node.get("arguments")
                    if not isinstance(args_, list):
                        continue
                    if any(isinstance(x, str) and "<node-" in x for x in args_):
                        stats["has_tag"] += 1
                        continue
                    tool = tools.get(node["task"])
                    if tool is None:
                        stats["no_tool"] += 1
                        continue
                    itypes = tool.get("input-type") or []
                    if len(itypes) != 1:
                        stats["multi_input"] += 1
                        continue
                    cands = [k for k, x in enumerate(args_)
                             if isinstance(x, str) and x.strip() and x.strip() not in req]
                    if len(cands) != 1:
                        stats["cand_args!=1"] += 1
                        continue
                    prods = []
                    for j in range(i):
                        tj = tools.get(nodes[j].get("task")) if isinstance(nodes[j], dict) else None
                        if tj and set(tj.get("output-type") or []) & set(itypes):
                            prods.append(j)
                    if len(prods) != 1:
                        stats["producer!=1"] += 1
                        continue
                    args_[cands[0]] = f"<node-{prods[0]}>"
                    stats["rewritten"] += 1
            wf.write(json.dumps(d) + "\n")
    print(f"[edge-snap] rows={n_rows} {stats}")


if __name__ == "__main__":
    main()
