# -*- coding: utf-8 -*-
"""x153 — 통과-집합 레버가 **라이브에서 실제로 발화했나**, 그리고 그 턴에 무슨 일이 있었나.

왜 따로 만드나 ([[09]]·[[30]] *"천장/결론 주장 전 레버 실발화율 전수확인"*): 단위 검정 통과는
라이브 발화가 아니다 — calc 레버가 단위 OK인데 라이브 342 sim 중 **31회만** 발화한 전례가 있다.
그리고 [[08]]: 집계(pass/fail)에서 결론으로 직행하지 않는다. 그래서 이 도구는 sim마다
**⒜발화 여부·턴 ⒝그 문장에 실린 통과 집합 ⒞제출된 계좌 ⒟gold ⒠종료 사유**를 한 줄에 모은다.

읽는 곳 둘:
  · 사이드카(`fb_<TAG>.jsonl`) = **우리 층이 무엇을 말했는지의 유일한 기록**(비커밋이라 궤적에 없다)
  · 결과(`results.json` 또는 영속화된 `<TAG>.json.gz`) = 궤적·보상

실행: py -3 x153_eligible_live_audit.py <fb_TAG.jsonl> <results.json[.gz]>
"""
import argparse
import collections
import gzip
import io
import json
import re
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

MARK = "Policy constants on record, for the products not already ruled out"


def load_jsonl(p):
    out = []
    op = gzip.open if p.endswith(".gz") else io.open
    with op(p, "rt", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                try:
                    out.append(json.loads(ln))
                except Exception:
                    pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sidecar")
    ap.add_argument("results")
    ap.add_argument("--rows", action="store_true", help="통과 집합 행 전체를 찍는다")
    a = ap.parse_args()

    recs = load_jsonl(a.sidecar)
    fired = collections.defaultdict(list)
    for r in recs:
        body = str(r.get("text") or r.get("body") or json.dumps(r, ensure_ascii=False))
        if MARK in body:
            fired[r.get("sim")].append((r.get("turn"), body))
    print("사이드카 레코드 %d · 통과-집합 발화 sim %d개" % (len(recs), len(fired)))

    op = gzip.open if a.results.endswith(".gz") else io.open
    with op(a.results, "rt", encoding="utf-8") as f:
        res = json.load(f)
    sims = res.get("simulations") or []
    print("sim %d개\n" % len(sims))

    agg = collections.Counter()
    for s in sims:
        sid = s.get("id") or s.get("simulation_id")
        task = s.get("task_id")
        rw = (s.get("reward_info") or {}).get("reward")
        term = s.get("termination_reason")
        subs = []
        for m in (s.get("messages") or []):
            for tc in (m.get("tool_calls") or []):
                if str(tc.get("name") or "").startswith("submit_referral"):
                    args = tc.get("arguments")
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except Exception:
                            args = {}
                    subs.append((args or {}).get("account_type"))
        hits = fired.get(sid) or []
        # 사이드카가 sim id 대신 task 이름을 쓸 수도 있다 — 둘 다 본다.
        if not hits:
            hits = fired.get(task) or []
        agg[(task, bool(hits), rw)] += 1
        print("%-10s reward=%-5s 발화=%-3s turns=%-14s 제출=%-28s 종료=%s"
              % (task, rw, ("%d회" % len(hits)) if hits else "0",
                 ",".join(str(t) for t, _ in hits)[:14],
                 ",".join(str(x) for x in subs)[:28], term))
        if a.rows and hits:
            body = hits[-1][1]
            for ln in body.splitlines():
                if ln.startswith("  "):
                    print("        " + ln.strip()[:110])

    print("\n=== 교차표 (task, 발화, reward) ===")
    for k, v in sorted(agg.items(), key=lambda kv: str(kv[0])):
        print("  %-10s 발화=%-6s reward=%-5s : %d" % (k[0], k[1], k[2], v))
    return 0


if __name__ == "__main__":
    sys.exit(main())
