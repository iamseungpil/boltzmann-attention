# -*- coding: utf-8 -*-
"""사이드카(`T2_FB_SIDECAR`)를 **sim별·turn순**으로 갈라 읽는다.

왜 필요한가 (2026-08-08·C324): 라이브 로그는 동시 실행 sim이 **인터리브**되는데 `sim=` 태그는
`[T2_LEVER]` 줄에만 붙는다. 그래서 *"어느 sim에서 무엇이 언제 나갔나"* 를 로그 줄만으로는
가를 수 없고, 실제로 한 sim의 침묵을 다른 sim의 발화로 오독할 뻔했다. 사이드카는 레코드마다
`sim`/`turn`을 지니므로 **귀속이 여기서 닫힌다** — 포렌식은 로그가 아니라 이걸 먼저 본다([[08]]).

사이드카는 비커밋 관측이라 궤적에 남지 않는다(뷰-채널 피드백은 `messages`에 없다) ⇒
*우리 층이 무엇을 말했는지*의 유일한 기록이다.

실행: py -3 x147_sidecar_by_sim.py <사이드카.jsonl> [--full] [--sim <id>]
      (원격 기본 경로: /home/woori/scratch/logs/fb_<TAG>.jsonl)
"""
import argparse
import collections
import json
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--full", action="store_true", help="본문 전체(기본=앞 110자)")
    ap.add_argument("--sim", default=None, help="이 sim만")
    ap.add_argument("--channel", default=None, help="이 channel만")
    a = ap.parse_args()

    recs = []
    with open(a.path, encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                try:
                    recs.append(json.loads(ln))
                except Exception:
                    pass
    print("레코드 %d" % len(recs))
    bysim = collections.OrderedDict()
    for r in recs:
        bysim.setdefault(r.get("sim"), []).append(r)
    print("sim %d개: %s" % (len(bysim),
                            ", ".join("%s(%d)" % (k, len(v)) for k, v in bysim.items())))

    for sim, rs in bysim.items():
        if a.sim and sim != a.sim:
            continue
        print("\n===== sim=%s  (%d건)" % (sim, len(rs)))
        ch = collections.Counter(r.get("channel") for r in rs)
        print("  channel: %s" % ", ".join("%s %d" % kv for kv in ch.most_common()))
        for r in sorted(rs, key=lambda x: (x.get("turn") or 0)):
            if a.channel and r.get("channel") != a.channel:
                continue
            t = " ".join(str(r.get("text") or "").split())
            print("  turn %-4s %-20s %-16s %s"
                  % (r.get("turn"), r.get("kind"), r.get("channel"),
                     t if a.full else t[:110]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
