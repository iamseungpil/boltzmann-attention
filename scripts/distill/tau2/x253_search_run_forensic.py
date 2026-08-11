# -*- coding: utf-8 -*-
r"""x253 — 검색 에이전트 런 포렌식 (070·071 · 사전 등록 P0~P4 · 유료 0 추가).

## 읽는 순서 (런처가 발사 전에 등록한 그대로 — **성적은 마지막**)

  P0 팔 오염   `[T2_SEARCH_AGENT]` 가 찍혔는가 · 고른 군 · **뺀 문서에 014/016 이 있는가**
               (그 제거가 엔진이 하는 **유일한 일**이다. 안 찍혔으면 나머지는 읽을 필요 없다.)
  P3 재료 도달 사이드카에 `decided_by_docs_text` 축자가 실렸는가 — 우리 층이 **실제로 말했는가**
  P2 표적 칸   gold `open_bank_account_4821` 의 `account_class` 가 맞았는가 (칸 단위)
  P4 Δspurious 게이트 거부 수 · gold 밖 쓰기 호출
  P1 성적      태스크별 pass (기준선 `bank_m3_20260810s` 0/2) — **3 sim×2 는 총점을 못 가른다**

⚠P0 가 0 이면 판정은 *"기구가 안 돌았다"* 이지 *"레버가 소용없다"* 가 아니다(死배선 ↔ 무효과).

실행(리모트): python x253_search_run_forensic.py [태그]
"""
import collections
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import x238_action_forensic as X                                  # noqa: E402
import t2_fbsidecar as FB                                         # noqa: E402

LOGS = "/home/woori/scratch/logs"
EXPIRED = ("doc_bank_accounts_bank_accounts_(general)_014",
           "doc_bank_accounts_bank_accounts_(general)_016")
GOLD_CLASS = {"task_070": "Sky Blue", "task_071": ("Sky Blue", "Gold Saver Account")}
DECIDED_SIG = "A separate check was run on the policy documents on record"


class _M(object):
    def __init__(self, r, c):
        self.role, self.content = r, c


def sidecar(tag, sims):
    key = {}
    for s in sims:
        ms = [_M(m.get("role"), m.get("content")) for m in s["messages"]]
        key[FB._sim_key(ms)] = (s["task_id"], s.get("trial"),
                                (s.get("reward_info") or {}).get("reward"))
    rec = collections.defaultdict(list)
    p = os.path.join(LOGS, "fb_%s.jsonl" % tag)
    try:
        for ln in open(p, encoding="utf-8", errors="replace"):
            o = json.loads(ln)
            rec[o["sim"]].append(o)
    except Exception as e:
        print("사이드카 없음: %r" % (e,))
    return key, rec


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "bank_sa_20260811"
    sims = X.load(tag)
    log = os.path.join(LOGS, "%s.log" % tag)
    src = open(log, encoding="utf-8", errors="replace").read() if os.path.exists(log) else ""

    print("=" * 96)
    print("P0 팔 오염 — 기구가 돌았는가")
    marks = re.findall(r"\[T2_SEARCH_AGENT\][^\n]*", src)
    print("  [T2_SEARCH_AGENT] %d회" % len(marks))
    for m in marks[:6]:
        print("    " + m[:150])
    dropped = [d for d in EXPIRED if any(d in m for m in marks)]
    print("  만료 제거 확인: %s" % (", ".join(dropped) or "★없음 — 엔진의 유일한 일이 안 일어났다"))
    for tagname in ("T2_DOCGROUP", "T2_DOCDECIDE"):
        got = re.findall(r"\[%s\][^\n]*" % tagname, src)
        print("  [%s] %d회%s" % (tagname, len(got),
                                 ("  예: " + got[0][:110]) if got else ""))
    if not marks:
        print("  ⇒ **기구가 안 돌았다**. 아래는 참고로만 읽고, 판정은 '배선/조건'이지 '레버'가 아니다.")

    key, rec = sidecar(tag, sims)
    print("\n" + "=" * 96)
    print("P3 재료 도달 — 우리 층이 실제로 말했는가")
    for k, v in sorted(rec.items(), key=lambda kv: str(key.get(kv[0]))):
        who = key.get(k, ("?",) * 3)
        turns = [r["turn"] for r in v if DECIDED_SIG in (r.get("text") or "")]
        print("  %-9s t%s rew=%s | 결정문 %s" % (who[0], who[1], who[2], turns or "없음"))

    print("\n" + "=" * 96)
    print("P2 표적 칸 — gold 액션의 account_class")
    for s in sorted(sims, key=lambda x: (x["task_id"], str(x.get("trial")))):
        ri = s.get("reward_info") or {}
        rows = []
        for c in (ri.get("action_checks") or []):
            a = c.get("action") or {}
            if "open_bank_account" in str(a.get("name")):
                rows.append("%s=%s" % ((a.get("arguments") or {}).get("account_class"),
                                       "✓" if c.get("action_match") else "✗"))
        got = []
        for m in s["messages"]:
            for tc in (m.get("tool_calls") or []):
                args = str((tc.get("function") or {}).get("arguments") or tc.get("arguments") or "")
                mm = re.search(r'"account_class"\s*:\s*"([^"]+)"', args)
                if mm:
                    got.append(mm.group(1))
        print("  %-9s t%s rew=%s | gold칸 %s | 실제 제출 %s"
              % (s["task_id"], s.get("trial"), (ri.get("reward")), rows or "-", got or "없음"))

    print("\n" + "=" * 96)
    print("P4 Δspurious · P1 성적")
    acts = X.a2_action_tools()
    per = collections.Counter()
    for s in sims:
        gnames, _ = X.gold_actions(s["task_id"])
        agent_calls = [n for r, n in X.calls(s) if r == "assistant"]
        per["spurious"] += sum(1 for n in agent_calls if n in acts and n not in gnames)
        per["denials"] += sum(1 for m in s["messages"] if m.get("role") == "tool"
                              and str(m.get("content") or "").lstrip().startswith("Error: ["))
    print("  gold 밖 액션 %d · 게이트 거부 %d" % (per["spurious"], per["denials"]))
    tally = collections.Counter()
    for s in sims:
        tally[s["task_id"]] += 1 if (s.get("reward_info") or {}).get("reward") == 1 else 0
        tally[s["task_id"] + "_n"] += 1
    for t in sorted(x for x in tally if not x.endswith("_n")):
        print("  %-9s %d/%d   (기준선 bank_m3_20260810s = 0/2 스모크)"
              % (t, tally[t], tally[t + "_n"]))
    print("\n※ P0 가 0 이면 '레버 무효'가 아니라 '기구 미발화'다 — 조건부터 본다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
