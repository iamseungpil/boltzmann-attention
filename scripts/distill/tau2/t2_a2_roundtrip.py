#!/usr/bin/env python
"""P-A2-1 데이터엔진 품질 게이트: round-trip 검증 (합성 (NL, spec) 쌍).

역방향 렌더의 GT는 원 spec. 품질 = 렌더 NL을 *재컴파일*한 spec이 원 spec과 일치하는가.
재컴파일은 외부(frontier/학습모델)가 채우고, 이 스크립트는 두 spec의 일치도를 채점:
  gate_recall(원 술어 포착)·applies_F1(게이트별 적용도구)·kind_match(satisfier/ask/terminal).
일치 높음 = NL이 spec을 충실히 운반(렌더 무손실) = 학습쌍으로 사용 가능.
낮음 = 렌더가 정보를 흘림 → 그 쌍 폐기 (데이터 청정).

Usage: t2_a2_roundtrip.py --pairs synth_seed_pairs_fable5.jsonl --recompiled recompiled.jsonl
  (recompiled = {"id":.., "style":.., "spec_recompiled": {...}} 줄들)
재컴파일 없이 --self 면 원 spec를 그대로 넣어 채점기 sanity(완전일치=1.0) 확인.
"""
import argparse, json, re


def kws(text):
    return {w for w in re.findall(r"[a-z_]+", text.lower()) if len(w) > 3 and w not in
            {"this", "that", "user", "must", "have", "been", "with", "before", "they"}}


def kind_of(g):
    return ("satisfier" if g.get("satisfiers") else
            "ask" if g.get("ask") else "terminal" if g.get("terminal") else "?")


def score(ref, gen):
    refg = {k: v for k, v in ref.items() if not k.startswith("_")}
    geng = {k: v for k, v in (gen or {}).items() if not k.startswith("_")}
    gentext = json.dumps(geng).lower()
    genwords = set(re.findall(r"[a-z_]+", gentext))
    recall = 0
    f1s, kinds = [], 0
    for g, rv in refg.items():
        rkw = kws(rv.get("predicate", "") + " " + g)
        if rkw and len(rkw & genwords) >= max(1, len(rkw) // 2):
            recall += 1
        rt = set(rv.get("applies_to", []))
        best, best_g = 0.0, None
        for gk, gv in geng.items():
            gt = set(gv.get("applies_to", []) or [])
            inter = len(rt & gt)
            if inter:
                p, r = inter / max(len(gt), 1), inter / max(len(rt), 1)
                f = 2 * p * r / (p + r)
                if f > best:
                    best, best_g = f, gv
        f1s.append(best)
        if best_g is not None and kind_of(best_g) == kind_of(rv):
            kinds += 1
    n = max(len(refg), 1)
    return recall / n, sum(f1s) / n, kinds / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--recompiled", default=None)
    ap.add_argument("--self", action="store_true", help="원 spec 자기채점 (sanity=1.0)")
    a = ap.parse_args()

    pairs = {}
    for l in open(a.pairs):
        d = json.loads(l)
        pairs[(d["id"], d.get("style", ""))] = d

    recompiled = {}
    if a.recompiled:
        for l in open(a.recompiled):
            d = json.loads(l)
            recompiled[(d["id"], d.get("style", ""))] = d["spec_recompiled"]

    rows = []
    for k, d in pairs.items():
        ref = d["spec"]
        gen = ref if a.self else recompiled.get(k)
        if gen is None:
            print(f"id={k}: NO recompiled spec"); continue
        gr, af, km = score(ref, gen)
        keep = gr >= 0.8 and af >= 0.7 and km >= 0.8
        rows.append(keep)
        print(f"id={k} style={d.get('style')}: gate_recall={gr:.2f} applies_F1={af:.2f} "
              f"kind_match={km:.2f} -> {'KEEP' if keep else 'DROP'}")
    if rows:
        print(f"[round-trip] kept {sum(rows)}/{len(rows)} "
              f"({'sanity OK (self=1.0)' if a.self else 'data-clean filter'})")


if __name__ == "__main__":
    main()
