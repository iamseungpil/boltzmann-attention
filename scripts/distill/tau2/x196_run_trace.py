# -*- coding: utf-8 -*-
r"""x196 — 한 런에서 **어느 기구가 어느 sim 에서 켜졌는가**를 한 장으로 (유료 0).

## 왜

사용자 지시 2026-08-09: *"앞으로 새로운 런할 때는 로그나 사이드카나 다른 수단으로 궤적 추적해서
어느 기구가 켜졌는지 확인할 수 있게 하라."*

지금까지 그 확인은 **런마다 다른 grep** 이었고, 이 세션에서만 두 번 틀렸다 — 우리 주입 문구는
`fb_*.jsonl` 채널로 나가는데 로그에서 세어 *"발화 0"* 이라고 읽었다(C369 의 재발). 그래서 세
출처를 **한 자리에서 합쳐** 읽는다:

  · `trace_<tag>.jsonl`  모든 `[T2_*]` stderr 마크 (기구 발화·`t2_launch` 가 기본 ON)
  · `fb_<tag>.jsonl`     우리가 **손님/에이전트에게 실제로 보낸 문장** (사이드카)
  · `results.json`       보상·종료사유 (sim 단위)

⚠**발화 ≠ 전달 ≠ 효과.** 이 표는 앞의 둘만 말한다. 효과는 궤적 정독이다([[08]]).

실행: python x196_run_trace.py <tag> [--sims <dir>] [--logs <dir>]
"""
import argparse
import collections
import glob
import gzip
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

SIMS = os.environ.get("GO_TAU2", "/home/woori/scratch/tau2-bench") + "/data/simulations"
LOGS = "/home/woori/scratch/logs"
# 우리가 보내는 문장의 **종류**는 A2 템플릿에서 오므로, 여기서는 그 문구의 고정 머리만 쓴다.
KINDS = [("결정블록", "A separate check was run"),
         ("통과집합", "Policy constants on record"),
         ("상태별세기", "grouped by the status"),
         ("창산수", "each one against the ones before it"),
         ("소진", "no room left this year"),
         ("미대조", "which was NOT checked against any allowance"),
         ("출처요구", "[SOURCE]"),
         ("순서", "[ORDER]"),
         ("검색소진", "[SEARCH-EXHAUST]")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("--sims", default=SIMS)
    ap.add_argument("--logs", default=LOGS)
    a = ap.parse_args()

    # ── 보상 ────────────────────────────────────────────────────────────────
    rw, term = {}, {}
    for p in (os.path.join(a.sims, a.tag, "results.json"),
              os.path.join(a.sims, a.tag, "results.json.gz")):
        if not os.path.exists(p):
            continue
        op = gzip.open if p.endswith(".gz") else open
        d = json.load(op(p, "rt", encoding="utf-8"))
        for s in d.get("simulations", ()):
            key = "%s#t%s" % (s["task_id"], s.get("trial"))
            rw[key] = s["reward_info"]["reward"]
            term[key] = s.get("termination_reason")
        break

    # ── 기구 발화 (stderr 마크) ─────────────────────────────────────────────
    fired = collections.defaultdict(collections.Counter)
    tp = os.path.join(a.logs, "trace_%s.jsonl" % a.tag)
    n_tr = 0
    if os.path.exists(tp):
        for line in open(tp, encoding="utf-8", errors="replace"):
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            n_tr += 1
            fired[r.get("sim") or "(무기명)"][r.get("mark")] += 1

    # ── 실제로 보낸 문장 (사이드카) ────────────────────────────────────────
    sent = collections.defaultdict(collections.Counter)
    fp = os.path.join(a.logs, "fb_%s.jsonl" % a.tag)
    n_fb = 0
    if os.path.exists(fp):
        for line in open(fp, encoding="utf-8", errors="replace"):
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            n_fb += 1
            t = r.get("text") or ""
            for label, mark in KINDS:
                if mark in t:
                    sent[r.get("sim") or "(무기명)"][label] += 1

    print("== %s ==  보상 %d sim · 마크 %d줄 · 사이드카 %d건" % (a.tag, len(rw), n_tr, n_fb))
    if not n_tr:
        print("⚠ trace 파일이 없다(%s) — `T2_TRACE` 가 꺼진 채 돌았다. 기구 발화는 이 표로 "
              "확인할 수 없다." % tp)

    print("\n§1 sim 별 결과")
    for k in sorted(rw):
        print("  %-18s reward=%-4s %s" % (k, rw[k], term.get(k)))

    # ⚠시행 태그가 안 붙는 경우가 있다 (2026-08-09 실측: orchestrator 가 `trial` 을 그 이름으로
    #   들고 있지 않으면 `task_010` 까지만 달린다). 그러면 **한 태스크의 두 시행이 한 줄로
    #   합쳐진다** — 표에 그 사실을 적는다. 없는 분해를 있는 척하지 않는다([[25]]).
    tagged = [s for s in fired if s and "#t" in s]
    if fired and not tagged:
        print("\n⚠ 마크에 시행 번호가 없다 — 아래 §2 는 **태스크 단위 합계**이고 시행별 분해가"
              " 아니다(같은 태스크의 %d 시행이 한 줄로 합쳐져 있다)."
              % max([sum(1 for k in rw if k.startswith(s)) for s in fired if s] or [1]))

    print("\n§2 기구 발화 (stderr 마크 · sim 별)")
    allmarks = sorted({m for c in fired.values() for m in c})
    if allmarks:
        for sim in sorted(fired):
            top = ", ".join("%s×%d" % (m, n) for m, n in fired[sim].most_common(10))
            print("  %-18s %s" % (sim, top))
        print("\n  전 sim 합계:")
        tot = collections.Counter()
        for c in fired.values():
            tot.update(c)
        for m, n in tot.most_common():
            print("    %-26s %d" % (m, n))

    print("\n§3 우리가 **실제로 보낸** 문장 (사이드카 · 종류별)")
    if sent:
        for sim in sorted(sent):
            print("  %-18s %s" % (sim, ", ".join("%s×%d" % (k, v)
                                                 for k, v in sent[sim].most_common())))
        tot2 = collections.Counter()
        for c in sent.values():
            tot2.update(c)
        print("\n  전 sim 합계: %s" % (", ".join("%s×%d" % kv for kv in tot2.most_common())
                                     or "(없음)"))
    else:
        print("  (사이드카 없음 또는 해당 문구 0)")

    print("\n※ 발화 ≠ 전달 ≠ 효과. 이 표는 앞의 둘만 말한다 — 효과는 궤적 정독이다([[08]]).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
