# -*- coding: utf-8 -*-
"""원장을 받아 놓고 **수로 줄였는가** — 줄였다면 그 수가 맞았는가.

이 두 태스크에서 무엇을 제출할 수 있는가는 두 개의 수로 결정된다. 둘 다
`get_referrals_by_user`가 돌려준 행과 정책 문서의 상수만으로 나오고, 대화의 어떤 말에도
의존하지 않는다:

    창_잔여   = 2 − |{원장 행 : 오늘 − 9일 ≤ 날짜 ≤ 오늘}|      (체킹 계좌 롤링 9일 창)
    연간_잔여(유형) = 상한(유형) − |{원장 행 : 그 유형}|

그래서 실패를 "모델이 판단을 잘못했다"로 적기 전에, 이 도구가 세 가지를 가른다:

    (a) 모델이 그 수를 **아예 말하지 않았다**            → 환원 자체가 없다
    (b) 말했는데 **틀렸다**                              → 환원은 했는데 산수가 틀렸다
    (c) 맞게 말했는데 **행동이 그 수를 따르지 않았다**    → 계산이 아니라 집행의 문제

셋은 서로 다른 레버를 부른다. 앞선 판(x120 §E)은 `6/6` 같은 **표기 형태**를 찾는 정규식이라
산문으로 쓴 "Made 6 referrals for the Gold Years Account"를 놓쳤다 — 그래서 '미발화'가
과대계수됐다. 여기서는 유형 이름 주변의 수를 전부 걷어 채점한다.

usage: x121_ledger_reduction_audit.py --dirs a,b --tasks task_101,task_102 [--show]
"""

import collections
import datetime
import glob
import gzip
import io
import json
import os
import re
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
TAU2 = os.environ.get("GO_TAU2", "/home/woori/scratch/tau2-bench")
DOMAIN = os.path.join(TAU2, "data", "tau2", "domains", "banking_knowledge")
SIMBASES = [os.path.join(TAU2, "data", "simulations"),
            os.path.join(REPO, "reports", "facet_rft_2026", "sim_results")]
WINDOW_DAYS, WINDOW_MAX = 9, 2


def arg(n, d=None):
    return sys.argv[sys.argv.index(n) + 1] if n in sys.argv else d


DIRS = [d for d in (arg("--dirs") or "").split(",") if d]
TASKS = [t for t in (arg("--tasks") or "task_101,task_102").split(",") if t]
SHOW = "--show" in sys.argv


def jopen(p):
    op = gzip.open if p.endswith(".gz") else io.open
    with op(p, "rt", encoding="utf-8", errors="replace") as fh:
        return json.load(fh)


def load_task(tid):
    p = os.path.join(DOMAIN, "tasks", tid + ".json")
    d = jopen(p)
    return d if d.get("id") else (d.get("tasks") or [d])[0]


def load_sims():
    out = []
    for base in SIMBASES:
        for d in DIRS:
            for p in glob.glob(os.path.join(base, d, "results.json")):
                for s in (jopen(p).get("simulations") or []):
                    s["_src"] = d
                    out.append(s)
    return out


def caps_from_docs():
    """상한은 **정책 문서**에서만 온다 — 태스크 notes에서 가져오면 gold 경유다([[23]]).

    코퍼스가 같은 사실을 네 가지 문형으로 쓴다. 넷 다 받는다.
    """
    pats = [re.compile(r"Annual limit\**:?\**\s*(\d+)\s*referral", re.I),
            re.compile(r"up to (\d+) referral bonuses per calendar year", re.I),
            re.compile(r"Annual (?:maximum|cap)\**:?\**\s*(\d+)\s*referral", re.I),
            re.compile(r"Maximum referrals(?: per year)?\s*[:|]\s*(\d+)", re.I),
            re.compile(r"Maximum referrals:\s*(\d+) per calendar year", re.I)]
    caps = {}
    for p in glob.glob(os.path.join(DOMAIN, "documents", "*")):
        try:
            txt = io.open(p, encoding="utf-8", errors="replace").read()
        except Exception:
            continue
        for rx in pats:
            m = rx.search(txt)
            if m:
                caps.setdefault(os.path.basename(p), int(m.group(1)))
                break
    return caps


def cap_for(type_name, doccaps):
    """유형 이름 → 그 유형 문서의 상한. 문서 파일명에 유형 슬러그가 들어 있다."""
    slug = type_name.lower().replace(" account", "").replace(" ", "_")
    best = None
    for fn, n in doccaps.items():
        if slug in fn.lower():
            best = n if best is None else min(best, n)
    return best


def ledger(task):
    ini = ((task.get("initial_state") or {}).get("initialization_data") or {}).get("agent_data") or {}
    rows = list(((ini.get("referrals") or {}).get("data") or {}).values())
    out = []
    for r in rows:
        d = None
        for f in ("%m/%d/%Y", "%Y-%m-%d"):
            try:
                d = datetime.datetime.strptime(r.get("date") or "", f).date()
                break
            except Exception:
                pass
        out.append((d, r.get("referred_account_type")))
    return out


NUM = r"(\d+)"
WORDNUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
           "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12}
EXHAUST = re.compile(r"(maxed|max(?:imum)? (?:out|reached)|limit (?:has been )?(?:reached|hit|exceeded|met)|"
                     r"reached (?:the|your) (?:annual )?(?:limit|cap|maximum)|no (?:more|remaining|further)|"
                     r"cap (?:has been )?reached|fully used|exhausted|not (?:be )?(?:available|eligible) .{0,30}limit)", re.I)


def claims_about(txt, type_name):
    """유형 이름 주변 ±90자에서 수 주장과 소진 주장을 걷는다."""
    got = []
    for m in re.finditer(re.escape(type_name.replace(" Account", "")), txt, re.I):
        seg = txt[max(0, m.start() - 90): m.end() + 90]
        for mm in re.finditer(r"(\d+)\s*(?:of|/|out of)\s*(\d+)", seg):
            got.append(("used_of_cap", int(mm.group(1)), int(mm.group(2)), seg))
        for mm in re.finditer(r"\b(?:made|used|done|submitted|have|had)\b[^.]{0,30}?\b(\d+)\b", seg, re.I):
            got.append(("used", int(mm.group(1)), None, seg))
        for w, v in WORDNUM.items():
            if re.search(r"\b%s\b[^.]{0,25}referral" % w, seg, re.I):
                got.append(("used", v, None, seg))
        if EXHAUST.search(seg):
            got.append(("exhausted", None, None, seg))
    return got


def main():
    doccaps = caps_from_docs()
    sims = load_sims()
    for tid in TASKS:
        task = load_task(tid)
        mine = [s for s in sims if s.get("task_id") == tid]
        if not mine:
            continue
        rows = ledger(task)
        used = collections.Counter(t for _, t in rows)
        latest = max((d for d, _ in rows if d), default=None)
        today = latest  # 창 계산의 기준일은 런의 오늘(=2025-11-14)이지만, 시딩 최신일이 그 이내인지가 관건
        run_today = datetime.date(2025, 11, 14)
        in_window = [(d, t) for d, t in rows if d and (run_today - d).days <= WINDOW_DAYS]
        types = sorted(used)
        print("=" * 100)
        print("== %s == 기계 진리 (원장 %d행 · 오늘=%s)" % (tid, len(rows), run_today))
        print("   창 안(≤%d일) %d건 %s ⇒ 창_잔여 = %d"
              % (WINDOW_DAYS, len(in_window), [(str(d), t) for d, t in in_window], WINDOW_MAX - len(in_window)))
        for t in types:
            c = cap_for(t, doccaps)
            print("   %-24s 사용 %d / 상한 %s ⇒ 연간_잔여 %s"
                  % (t, used[t], c if c is not None else "?", (c - used[t]) if c is not None else "?"))
        tot = collections.Counter()
        for s in sorted(mine, key=lambda x: (x["_src"], x.get("trial") or 0)):
            prose = [str(m.get("content") or "") for m in (s.get("messages") or [])
                     if m.get("role") == "assistant" and m.get("content")]
            txt = "\n".join(prose)
            said_window = bool(re.search(r"rolling (?:\d+[- ]day )?window|9[- ]day", txt, re.I))
            right = wrong = 0
            details = []
            for t in types:
                c = cap_for(t, doccaps)
                for kind, a, b, seg in claims_about(txt, t):
                    if kind == "used_of_cap":
                        ok = (a == used[t] and (c is None or b == c))
                    elif kind == "used":
                        ok = (a == used[t])
                        if a in (0,) or a > 50:
                            continue
                    else:  # exhausted
                        ok = (c is not None and used[t] >= c)
                    (details.append((t, kind, a, b, ok, seg)))
                    if ok:
                        right += 1
                    else:
                        wrong += 1
            tot["trial"] += 1
            tot["창언급" if said_window else "창미언급"] += 1
            tot["주장있음" if (right + wrong) else "주장없음"] += 1
            tot["맞음"] += right
            tot["틀림"] += wrong
            print("  [%-22s t%s] 창언급=%-5s 수주장 %d건(맞음 %d/틀림 %d)"
                  % (s["_src"], s.get("trial"), said_window, right + wrong, right, wrong))
            if SHOW:
                for t, kind, a, b, ok, seg in details:
                    if not ok:
                        print("      ✗ %-22s %s %s/%s  «%s»" % (t, kind, a, b,
                                                                re.sub(r"\s+", " ", seg)[:150]))
        print("  ── 합계 %s" % dict(tot))


if __name__ == "__main__":
    main()
