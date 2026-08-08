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


def sim_key(sim):
    """사이드카의 `sim` 키를 궤적에서 **재계산**한다 — 정확 조인의 유일한 길.

    `t2_fbsidecar._sim_key` 가 쓰는 지문 = **첫 손님 발화의 sha1 앞 12자**다(엔진 내부 id 를
    안 쓰려는 설계). 그래서 결과 파일의 UUID 와는 영영 안 맞고, 본문 대조도 안 된다 —
    사이드카는 **비커밋 채널**이라 그 문장이 궤적에 남지 않기 때문이다(그게 설계다).
    턴 번호로 추정하는 길도 있으나 같은 태스크의 두 trial 이 같은 구간을 가지면 갈리지 않는다.
    ⇒ 지문을 그대로 다시 계산한다. 결정론이고 모호함이 없다.
    """
    import hashlib
    for m in (sim.get("messages") or []):
        if m.get("role") == "user":
            c = m.get("content")
            if isinstance(c, str) and c.strip():
                return hashlib.sha1(c.strip().encode("utf-8")).hexdigest()[:12]
    return "nouser"


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

    # ⚠**지문 충돌을 먼저 말한다** (2026-08-08 실측): `_sim_key` 는 첫 손님 발화의 해시라,
    #   같은 태스크의 두 trial 이 **같은 문장으로 시작하면 한 지문을 공유한다**(run g: 6 sim →
    #   5 지문). 그 지문의 레코드는 두 sim 이 섞인 것이라 **trial 단위 귀속이 불가**하다.
    #   조용히 넘어가면 한 sim 의 발화를 다른 sim 것으로 읽는다 — C325 와 같은 병이다.
    keys = collections.Counter(sim_key(s) for s in sims)
    dup = {k: n for k, n in keys.items() if n > 1}
    if dup:
        print("⚠지문 충돌 %s — 이 지문의 발화는 **sim 단위로 귀속할 수 없다**(첫 발화가 같다)\n"
              % ", ".join("%s×%d" % (k, n) for k, n in sorted(dup.items())))

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
        hits = fired.get(sim_key(s)) or []
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
