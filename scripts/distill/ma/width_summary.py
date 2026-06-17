#!/usr/bin/env python3
"""Morning readout: collate all width_*.json (S*(width) ladder) into scale x width tables.
Prints SET_EXACT (one-shot formalize), arm_A (in-head), and DECOMP (offload sufficiency) per model,
so the critical-scale onset and the decomposition recovery are visible at a glance."""
import json, glob, argparse, os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/home/woori/scratch/depth/c8/width")
    args = ap.parse_args()
    files = sorted(glob.glob(os.path.join(args.dir, "width_*.json")))
    rows = []
    for f in files:
        try:
            d = json.load(open(f))
        except Exception:
            continue
        rows.append((d.get("tag") or os.path.basename(f), d.get("by_width", {})))
    widths = sorted({int(w) for _, bw in rows for w in bw})

    def table(metric, title):
        print(f"\n=== {title} (by width) ===")
        print("model".ljust(14) + "".join(f"  w{w}".rjust(7) for w in widths))
        for tag, bw in rows:
            cells = []
            for w in widths:
                v = (bw.get(str(w)) or bw.get(w) or {}).get(metric)
                cells.append(f"{v:.2f}".rjust(7) if isinstance(v, (int, float)) else "   -   ")
            print(tag.ljust(14) + "".join(cells))

    table("set_exact", "SET_EXACT — one-shot multi-attr extraction (the width wall)")
    table("arm_A_item", "arm A — in-head end-to-end")
    table("decomp_item", "DECOMP item — per-attr offload (sufficiency)")
    table("decomp_set_exact", "DECOMP set-exact — offload formalize")
    print(f"\n{len(rows)} models: {[t for t,_ in rows]}")


if __name__ == "__main__":
    main()
