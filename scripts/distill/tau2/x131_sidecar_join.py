# -*- coding: utf-8 -*-
"""x131 — 사이드카(우리 층이 무엇을 말했나)를 **태스크에 붙인다** (유료 0·모델 0).

우리 층의 발화는 비커밋이라 궤적에 없다 — 그래서 사이드카가 유일한 기록인데, 그 파일의
`sim` 필드는 tau2의 sim id가 아니라 **첫 유저 발화의 해시**다(`t2_fbsidecar._sim_key`).
그래서 "이 발화가 어느 태스크의 것인가"를 눈으로 맞출 수 없었고, led_j 분석에서 실제로
막혔다(사이드카 sim 2개 vs 저장된 sim 3개, 대응 불명).

같은 해시를 결과 쪽에서도 계산해 붙인다. 그러면 물을 수 있는 것:

  · 이 태스크에서 우리 층이 **무엇을 요구했나**, **몇 번째 턴에**
  · 손님이 도구를 실행하기 **전에** 요구가 있었나 (없었으면 게이트 미발화)
  · 같은 요구가 반복됐나 (접힘이 듣고 있나)

usage: x131_sidecar_join.py --dir bank_stack_led_20260807j --sidecar /path/fb_*.jsonl
                            [--tasks task_100] [--grep 계좌] [--show 3]
"""

import argparse
import glob
import gzip
import io
import json
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, HERE)

import t2_fbsidecar as FB    # noqa: E402 — 같은 해시 함수를 쓴다(재구현하면 갈린다)


class _M(object):
    """`_sim_key`가 보는 것만 — role 과 content."""

    def __init__(self, role, content):
        self.role = role
        self.content = content


def _load(dirname):
    cands = [os.path.join(REPO, "reports", "facet_rft_2026", "sim_results", dirname + ".json.gz")]
    cands += glob.glob(os.path.join(os.path.expanduser("~"), "scratch", "tau2-bench",
                                    "data", "simulations", dirname, "results.json"))
    for p in cands:
        if os.path.exists(p):
            op = gzip.open if p.endswith(".gz") else open
            with op(p, "rt", encoding="utf-8", errors="replace") as fh:
                return json.load(fh), p
    raise SystemExit("no results for %r" % dirname)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--sidecar", default="")
    ap.add_argument("--tasks", default="")
    ap.add_argument("--grep", default="")
    ap.add_argument("--show", type=int, default=0, help="본문 앞머리 N자 (0=제목만)")
    a = ap.parse_args()

    side = a.sidecar or os.path.join(os.path.expanduser("~"), "scratch", "logs",
                                     "fb_%s.jsonl" % a.dir)
    if not os.path.exists(side):
        raise SystemExit("사이드카 없음: %s" % side)

    data, src = _load(a.dir)
    key2sim = {}
    for s in data.get("simulations") or []:
        msgs = [_M(m.get("role"), m.get("content")) for m in (s.get("messages") or [])]
        key2sim.setdefault(FB._sim_key(msgs), []).append((s.get("task_id"), s.get("trial")))
    print("source: %s\n사이드카: %s\n지문↔태스크 대응 %d건\n" % (src, side, len(key2sim)))

    want = set(t.strip() for t in a.tasks.split(",") if t.strip())
    rows = {}
    for ln in open(side, encoding="utf-8", errors="replace"):
        try:
            d = json.loads(ln)
        except Exception:
            continue
        who = key2sim.get(d.get("sim"))
        label = "·".join("%s t%s" % (t, tr) for t, tr in who) if who else "(대응 없음 %s)" % d.get("sim")
        if want and not any(t in want for t, _tr in (who or [])):
            continue
        text = " ".join(str(d.get("text") or "").split())
        if a.grep and a.grep not in text:
            continue
        rows.setdefault(label, []).append((d.get("turn"), d.get("channel"), d.get("sha"), text))

    for label, items in rows.items():
        print("=" * 96)
        print("== %s ==  발화 %d건" % (label, len(items)))
        for turn, ch, sha, text in sorted(items, key=lambda r: (r[0] if r[0] is not None else -1)):
            head = text[:a.show] if a.show else ""
            print("   턴%-4s %-14s %s %s" % (turn, ch, sha[:8] if sha else "?", head))


if __name__ == "__main__":
    main()
