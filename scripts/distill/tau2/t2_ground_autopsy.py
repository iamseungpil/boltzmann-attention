#!/usr/bin/env python
"""ask_refine(=후보 있으나 resolve→None) 전수 분해 — spec-fail의 *실제 원인*을 5분류.

§7이 "spec-fail 지배"를 가리킨 뒤, 그 안을 가른다(처방이 다름):
  KEY_MISMATCH   : among 키가 카탈로그 option 키에 없음 → 포맷/키 정합 결함(엔진/spec)
  VALUE_MISMATCH : 키는 맞고 *값*이 그 키의 enum에 없음 → 0-match. 값 정규화(offload-snap) or 모델 wrong-value
  UNDER_DET      : among 매칭 행 >1 → 모델 under-extraction(제약 부족·decomp) or 진짜 ambiguous(ask)
  ANCHOR_MISSING : substitute인데 anchor 미해소
  UNIQUE_OK      : 사실 유일(=엔진 버그 후보)
저장된 sim을 재생: 각 resolve_selection 호출 직전 tool출력으로 _ground 재구성→among 적용→분류.
도메인-일반(_ground가 spec 구동). retail/airline 공통.

Run: py -3 t2_ground_autopsy.py --sim data/simulations/<save_to> --spec a2/<domain>.grounding.json
"""
import argparse
import json
import os
import sys
from collections import Counter

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import t2_resolve_patch as P  # noqa: E402


def _outs_before(msgs, idx):
    """idx 이전 tool 메시지들을 _tool_outputs와 동일 순서(최근→과거)로 파싱."""
    outs = []
    for m in reversed(msgs[:idx]):
        if m.get("role") != "tool" or m.get("error"):
            continue
        c = m.get("content")
        if not isinstance(c, str):
            continue
        try:
            outs.append(json.loads(c))
        except Exception:
            pass
    return outs


def _classify(args, cat, anchor, present, cs):
    if not present or not cat:
        return "NO_CATALOG", {}
    among = args.get("among") or {}
    optkeys = set()
    valsets = {}
    for row in cat:
        for k, v in (row.get("options") or {}).items():
            optkeys.add(k)
            valsets.setdefault(k, set()).add(str(v))
    # key mismatch
    bad_keys = [k for k in among if k not in optkeys]
    if bad_keys:
        return "KEY_MISMATCH", {"bad_keys": bad_keys, "optkeys": sorted(optkeys)}
    # value mismatch (key ok, value not in that key's enum)
    bad_vals = {k: v for k, v in among.items() if str(v) not in valsets.get(k, set())}
    # apply among (string match, like resolve_op_tau2._among_match)
    matched = [r for r in cat if all(str((r.get("options") or {}).get(k)) == str(v) for k, v in among.items())]
    avail = [r for r in matched if r.get("available")]
    pool = avail or matched
    op = args.get("op")
    if op == "substitute" and anchor is None:
        return "ANCHOR_MISSING", {}
    if len(pool) == 0:
        if bad_vals:
            return "VALUE_MISMATCH", {"bad_vals": bad_vals}
        return "VALUE_MISMATCH", {"note": "0-match (over-constrained combo)", "among": among}
    if len(pool) > 1:
        return "UNDER_DET", {"n_match": len(pool), "among": among}
    return "UNIQUE_OK", {"among": among}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", required=True)
    ap.add_argument("--spec", required=True)
    ap.add_argument("--show", type=int, default=8)
    a = ap.parse_args()
    with open(a.spec, encoding="utf-8") as f:
        P._GSPEC = json.load(f)
    cs = P._GSPEC["candidate_source"]
    with open(os.path.join(a.sim, "results.json"), encoding="utf-8") as f:
        sims = json.load(f).get("simulations", [])

    cls = Counter()
    samples = []
    for s in sims:
        msgs = s.get("messages", [])
        for i, m in enumerate(msgs):
            if m.get("role") != "assistant":
                continue
            for tc in (m.get("tool_calls") or []):
                if tc.get("name") != "resolve_selection":
                    continue
                args = tc.get("arguments")
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        args = {}
                outs = _outs_before(msgs, i)
                cat, anchor, present = P._ground(outs, P._GSPEC)
                c, info = _classify(args or {}, cat, anchor, present, cs)
                cls[c] += 1
                if c in ("VALUE_MISMATCH", "KEY_MISMATCH", "UNDER_DET") and len(samples) < a.show:
                    samples.append((c, {k: args.get(k) for k in ("op", "attr", "among", "set")}, info))

    tot = sum(cls.values())
    print(f"=== ask_refine/spec-fail 분해 ({a.sim}) · resolve 호출 {tot} ===")
    for k, n in cls.most_common():
        print(f"  {k:15s} {n:3d}  ({n/max(tot,1):.2f})")
    print("\n── 샘플 ──")
    for c, args, info in samples:
        print(f"  [{c}] args={json.dumps(args, ensure_ascii=False)[:140]}")
        print(f"       {json.dumps(info, ensure_ascii=False)[:200]}")
    print("\n[처방] VALUE_MISMATCH 지배 → 값 정규화(offload catalog-snap·§23B) or 모델 wrong-value(formalize 정확도·측정)."
          " UNDER_DET 지배 → under-extraction(decomp·더 많은 attr) or ask. KEY_MISMATCH → spec/엔진 키 정합.")


if __name__ == "__main__":
    main()
