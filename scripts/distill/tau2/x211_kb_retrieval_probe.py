# -*- coding: utf-8 -*-
r"""x211 — 010 의 답이 든 문서는 **에이전트의 질의로 떠오르는가** (유료 0 · 검색만).

## 왜 (사용자 지시 2026-08-10 · C393 뒤)

x210 이 확정했다 — 010 에서 갈리는 것은 꼬리말이 아니라 **상태 정의 문서의 도달**뿐이다
(문서 없으면 세 꼬리말 다 0/8, 있으면 다 8/8). 그리고 라이브 궤적을 보면 에이전트는 매번
그 문서를 **찾으려다 실패하고 상담원 이관**으로 끝난다:

    KB_search_bm25("retrieve reason codes for referral statuses")  → 못 찾음 → 이관
    KB_search_dense("details on Platinum referral rejection")      → 못 찾음 → 이관

답은 `doc_credit_cards_credit_cards_(general)_001` 에 축자로 있다:
*"REJECTED — the user has too many referral processes going on"*.

## 이 프로브가 가르는 것

  ⒜ **그 문서가 애초에 검색으로 잡히나** — 이상적 질의로도 안 나오면 그것은 **환경 사실**이고
     우리가 지시로 덮을 일이 아니다.
  ⒝ **에이전트가 실제로 쓴 질의**로는 몇 위인가 — 잡히는데 질의가 나쁜 것이면, 질의를 만드는
     일은 **격리 서브**가 할 수 있다(규칙 E: 지시는 서브에만·최소로).

⚠질의는 **궤적에서 그대로 꺼낸다**(프로브가 저작하지 않는다). 이상적 질의만 우리가 쓴다 —
  그건 *"검색기가 이 문서를 낼 능력이 있는가"* 를 재는 상한이지 처방이 아니다.

실행 (리모트·tau2 env 필요):
    cd /home/woori/scratch/tau2-bench && python <this> [--variant alltools]
"""
import argparse
import collections
import glob
import gzip
import json
import os
import re
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

TARGET = "doc_credit_cards_credit_cards_(general)_001"
SIMS = os.environ.get("T2_SIM_DIRS", "/home/woori/scratch/tau2-bench/data/simulations")
GZ = os.environ.get("T2_SIM_GZ",
                    "/home/woori/workspace_common/boltzmann-attention-pi/reports/"
                    "facet_rft_2026/sim_results")
# 상한 측정용 — 문서 제목·본문 낱말을 그대로 쓴 질의(처방이 아니라 검색기 능력의 천장)
IDEAL = ["Understanding Credit Card Referral Statuses",
         "referral status meanings COMPLETE IN_PROGRESS REJECTED",
         "what does REJECTED mean for a credit card referral",
         "too many referral processes going on"]


def live_queries():
    """010 궤적에서 **에이전트가 실제로 보낸** KB 질의를 전부 꺼낸다."""
    out = collections.Counter()
    paths = glob.glob(os.path.join(SIMS, "*", "results.json")) + \
        glob.glob(os.path.join(GZ, "*.json.gz"))
    for p in paths:
        try:
            d = (json.load(gzip.open(p, "rt", encoding="utf-8")) if p.endswith(".gz")
                 else json.load(open(p, encoding="utf-8")))
        except Exception:
            continue
        if not isinstance(d, dict):
            continue
        for s in d.get("simulations") or []:
            if not isinstance(s, dict) or s.get("task_id") != "task_010":
                continue
            for m in s.get("messages") or []:
                for tc in (m.get("tool_calls") or []):
                    f = tc.get("function") or tc
                    if not str(f.get("name") or "").startswith("KB_search"):
                        continue
                    a = f.get("arguments")
                    try:
                        a = json.loads(a) if isinstance(a, str) else (a or {})
                    except Exception:
                        a = {}
                    q = str(a.get("query") or "").strip()
                    if q:
                        out[(str(f.get("name")), q)] += 1
    return out


def rank_of(text, target):
    """검색 결과 문자열에서 대상 문서의 **순위**(1부터). 없으면 None."""
    ids = re.findall(r"ID:\s*(\S+)", text or "")
    for i, x in enumerate(ids, 1):
        if x == target:
            return i, len(ids)
    return None, len(ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default=os.environ.get("T2_RETRIEVAL", "alltools"))
    ap.add_argument("--k", type=int, default=10)
    a = ap.parse_args()

    sys.path.insert(0, "/home/woori/scratch/tau2-bench/src")
    from tau2.domains.banking_knowledge.environment import get_environment   # noqa: E402
    env = get_environment(retrieval_variant=a.variant)
    tk = env.tools
    fns = {}
    for name in ("KB_search", "KB_search_bm25", "KB_search_dense"):
        if hasattr(tk, name):
            fns[name] = getattr(tk, name)
    print("변이=%s · 가용 검색 도구: %s" % (a.variant, sorted(fns)))
    if not fns:
        print("⚠검색 도구가 없다 — 변이를 확인하라.")
        return 1

    print("\n§1 상한 — 이 검색기가 그 문서를 **낼 수는 있나** (우리가 쓴 이상적 질의)")
    for q in IDEAL:
        for nm, fn in sorted(fns.items()):
            try:
                r = fn(q, a.k) if nm != "KB_search" else fn(q)
            except TypeError:
                r = fn(q)
            except Exception as e:
                r = "ERR %r" % (e,)
            pos, n = rank_of(r, TARGET)
            print("  %-16s %-52s → %s / %d" % (nm, q[:52], ("%d위" % pos) if pos else "없음", n))

    print("\n§2 실제 — **에이전트가 보낸 질의**로는 몇 위인가 (궤적 축자)")
    qs = live_queries()
    if not qs:
        print("  (010 궤적에서 KB 질의를 못 찾았다)")
    hit = miss = 0
    for (nm, q), cnt in qs.most_common(24):
        fn = fns.get(nm) or fns.get("KB_search_bm25") or list(fns.values())[0]
        try:
            r = fn(q, a.k)
        except TypeError:
            r = fn(q)
        except Exception as e:
            r = "ERR %r" % (e,)
        pos, n = rank_of(r, TARGET)
        hit, miss = (hit + 1, miss) if pos else (hit, miss + 1)
        print("  %-16s x%-2d %-46s → %s / %d"
              % (nm, cnt, q[:46], ("%d위" % pos) if pos else "**없음**", n))
    print("\n  실제 질의 %d개 중 그 문서를 낸 것: **%d개**" % (hit + miss, hit))
    print("\n※ §1 이 전부 '없음' 이면 → 검색기가 못 내는 것이고 **환경 사실**이다(지시로 덮지 않는다)."
          "\n  §1 은 나오는데 §2 가 '없음' 이면 → 질의 문제이고, 질의 생성은 **격리 서브**가 할 수 있다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
