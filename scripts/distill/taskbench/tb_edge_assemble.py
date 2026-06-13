#!/usr/bin/env python
"""엣지-조립 선택기 (EDGE_LEVEL_REDESIGN §4-1, zero-GPU) — plan-atomic MBR 대체 검증.

검증기 = 엣지 source-redundancy(독립소스 수; 같은-정책 K샘플=1소스). 채택 규칙 t 스윕:
  redund≥t 엣지만 모아 DAG 조립 → edge-F1(조립, gold). oracle분석과 동일 내부척도라
  best-stack 선택(0.733)·단일최대(0.822)·조립-oracle(0.858)과 직접 비교 = "엣지-조립이 통째선택 위로
  얼마나 회수하나". --acyclic 면 사이클 제거(deploy 유효 DAG; 메트릭은 set-F1이라 무관하나 정직).

Usage: tb_edge_assemble.py --tb_dir <TB> --ar_tag tb_dpo2g_mmk --ar_group dpo2g \
         --selected <TBPRED>/tb_sel4_dpo2g.json [--acyclic]
"""
import argparse, json
from collections import defaultdict
from tb_kgate_select import norm, f1
from tb_select_official import HM, load_records, sig


def load_gold(tb_dir, domain):
    gold = {}
    for l in open(f"{tb_dir}/{domain}/data.json", encoding="utf-8"):
        d = json.loads(l)
        links = d.get("tool_links") or d.get("sampled_links") or []
        if isinstance(links, str):
            links = json.loads(links)
        gold[d["id"]] = {(norm(e["source"]), norm(e["target"])) for e in links
                         if isinstance(e, dict) and "source" in e and "target" in e}
    return gold


def make_acyclic(edges, score):
    """redundancy 높은 엣지부터 추가하되 사이클 생기면 스킵 (deploy 유효 DAG)."""
    adj = defaultdict(set)

    def reaches(a, b, seen):
        if a == b:
            return True
        for nb in adj[a]:
            if nb not in seen:
                seen.add(nb)
                if reaches(nb, b, seen):
                    return True
        return False

    kept = set()
    for e in sorted(edges, key=lambda e: -score[e]):
        s, t = e
        if s == t:
            continue
        if reaches(t, s, {t}):   # t→...→s 이미 있으면 s→t 추가는 사이클
            continue
        adj[s].add(t)
        kept.add(e)
    return kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tb_dir", required=True)
    ap.add_argument("--ar_tag", default="tb_dpo2g_mmk")
    ap.add_argument("--ar_group", default="dpo2g")
    ap.add_argument("--domain", default="data_multimedia")
    ap.add_argument("--hm", default=None)
    ap.add_argument("--selected", default=None)
    ap.add_argument("--acyclic", action="store_true")
    ap.add_argument("--out_pred", default=None, help="빈도×타입 조립을 공식-eval용 예측본으로 출력")
    ap.add_argument("--out_t", type=int, default=2, help="out_pred 채택 임계 (기본 2)")
    a = ap.parse_args()
    TB, D = a.tb_dir, a.domain
    hm_list = a.hm.split(",") if a.hm else HM
    nodes = json.load(open(f"{TB}/{D}/tool_desc.json"))["nodes"]
    valid = {norm(t["id"]) for t in nodes}
    # 결정론 타입 검증기: 엣지 A→B 호환 ⇔ output-type(A) ∩ input-type(B) ≠ ∅
    otype = {norm(t["id"]): set(t.get("output-type", [])) for t in nodes}
    itype = {norm(t["id"]): set(t.get("input-type", [])) for t in nodes}

    def type_ok(e):
        return bool(otype.get(e[0], set()) & itype.get(e[1], set()))

    gold = load_gold(TB, D)

    pools, groups = [], []
    for k in range(8):
        pools.append(load_records(f"{TB}/{D}_sub500/predictions/{a.ar_tag}{k}.json"))
        groups.append(a.ar_group)
    for m in hm_list:
        try:
            pools.append(load_records(f"{TB}/{D}_sub500_eval_{m}/predictions/{m}.json"))
            groups.append(m)
        except FileNotFoundError:
            pass
    selected = load_records(a.selected) if a.selected else None
    ids = [i for i in sorted(set.intersection(*[set(p) for p in pools[:8]]) & set(gold)) if gold[i]]

    ts = [1, 2, 3, 4, 5]
    asm_sum = {t: 0.0 for t in ts}
    asm_type_sum = {t: 0.0 for t in ts}   # 빈도 ∧ 타입-호환
    sel_sum = best_sum = asmoracle_sum = 0.0
    g_tok = g_tot = w_tok = w_tot = 0     # 타입-호환 진단 (gold vs wrong)
    n = 0
    for i in ids:
        G = gold[i]
        edge_src = defaultdict(set)
        best_single = 0.0
        for p, g in zip(pools, groups):
            rec = p.get(i)
            s = sig(rec, valid) if rec else None
            if s is None:
                continue
            best_single = max(best_single, f1(s[0], G))
            for e in s[0]:
                if e[0] != e[1]:
                    edge_src[e].add(g)
        if not edge_src:
            continue
        n += 1
        best_sum += best_single
        asmoracle_sum += f1(G & set(edge_src), G)
        score = {e: len(srcs) for e, srcs in edge_src.items()}
        # 타입-호환 진단: 풀에 존재하는 gold/wrong 엣지가 타입검증을 통과하나
        for e in set(edge_src):
            if e in G:
                g_tot += 1; g_tok += type_ok(e)
            else:
                w_tot += 1; w_tok += type_ok(e)
        for t in ts:
            cand = {e for e, srcs in edge_src.items() if len(srcs) >= t}
            ctype = {e for e in cand if type_ok(e)}
            if a.acyclic:
                cand = make_acyclic(cand, score)
                ctype = make_acyclic(ctype, score)
            asm_sum[t] += f1(cand, G)
            asm_type_sum[t] += f1(ctype, G)
        if selected:
            ss = sig(selected.get(i, {}), valid)
            sel_sum += f1(ss[0], G) if ss else 0.0

    print(f"[엣지-조립] n={n} (비-단일노드){' ·acyclic' if a.acyclic else ''}")
    print(f"  기준선:  best-stack 선택 {sel_sum/n:.3f}  |  단일최대 {best_sum/n:.3f}  |  "
          f"조립-oracle {asmoracle_sum/n:.3f}")
    print(f"  ★타입-호환 진단(결정론 검증기 변별): "
          f"gold {g_tok/max(g_tot,1)*100:.1f}% vs wrong {w_tok/max(w_tot,1)*100:.1f}% 통과 "
          f"(gold-wrong 분리 클수록 타입검증 유효)")
    print(f"  엣지-조립(redund≥t):  [빈도단독]  [빈도∧타입]")
    for t in ts:
        print(f"    t={t}: {asm_sum[t]/n:.3f} ({asm_sum[t]/n-sel_sum/n:+.3f})   "
              f"{asm_type_sum[t]/n:.3f} ({asm_type_sum[t]/n-sel_sum/n:+.3f} vs 선택)")

    # 공식-eval용 예측본 출력 (전 id; 빈도≥out_t ∧ 타입 ∧ acyclic; 엣지 없으면 best-stack 폴백)
    if a.out_pred:
        allids = sorted(set.intersection(*[set(p) for p in pools[:8]]))
        nfb = 0
        with open(a.out_pred, "w") as w:
            for i in allids:
                edge_src = defaultdict(set)
                for p, g in zip(pools, groups):
                    s = sig(p.get(i), valid) if p.get(i) else None
                    if s:
                        for e in s[0]:
                            if e[0] != e[1]:
                                edge_src[e].add(g)
                score = {e: len(s2) for e, s2 in edge_src.items()}
                cand = {e for e, s2 in edge_src.items() if len(s2) >= a.out_t and type_ok(e)}
                cand = make_acyclic(cand, score)
                if cand:
                    tools = {x for e in cand for x in e}
                    # 공식 eval은 node arguments의 <node-j> 참조로 엣지 복원 → 토폴로지 순서+인덱스 인코딩
                    from collections import deque
                    adj = defaultdict(list); indeg = {x: 0 for x in tools}
                    for s, t in cand:
                        adj[s].append(t); indeg[t] += 1
                    q = deque([x for x in tools if indeg[x] == 0]); order = []
                    while q:
                        x = q.popleft(); order.append(x)
                        for nb in adj[x]:
                            indeg[nb] -= 1
                            if indeg[nb] == 0:
                                q.append(nb)
                    for x in tools:
                        if x not in order:
                            order.append(x)
                    idx = {t: j for j, t in enumerate(order)}
                    tnodes = [{"task": t, "arguments": [f"<node-{idx[s]}>" for (s, tt) in cand if tt == t]}
                              for t in order]
                    rec = {"id": i, "user_request": (selected.get(i, {}) or {}).get("user_request", ""),
                           "result": {"task_steps": [],
                                      "task_nodes": tnodes,
                                      "task_links": [{"source": e[0], "target": e[1]} for e in cand]}}
                else:  # 엣지 없음(단일노드/전부탈락) = best-stack 폴백
                    nfb += 1
                    rec = selected.get(i) or pools[0][i]
                w.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[out_pred] wrote {a.out_pred} (t={a.out_t}∧type∧acyclic; best-stack 폴백 {nfb}/{len(allids)})")


if __name__ == "__main__":
    main()
