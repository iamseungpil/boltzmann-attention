# -*- coding: utf-8 -*-
"""One task, everything needed to name its cause and then name the lever.

The batch forensics say where failures stand; they cannot say *why* a particular turn went
the way it did, and a table row like "부재 판정 실패" is a label, not a cause. This prints,
for one task, the six things an honest attribution needs:

  §0 gold        what the task actually asks for — actions with their arguments, the
                 information the agent had to communicate, and the scenario the customer
                 was given (the customer's claim is an outside claim, [[21]]).
  §1 채점        per trial, which gold action first missed, and on what basis it was scored.
  §2 궤적        every turn: the model's prose, its calls with arguments, what came back,
                 and **who said it** — env or us (source-based, never by wording: x106 got
                 that wrong in both directions when it guessed from the text).
  §3 우리 층     the sidecar for this simulation — the reminders that never reach the
                 trajectory. Without it "our layer said nothing" is not an available
                 conclusion ([[55]]).
  §4 KB 대조     for every search the run made, whether the corpus actually holds the
                 words. This is the difference between "the model failed to find it" and
                 "it is not there" — 012 turns on exactly that distinction.
  §5 레버        which levers fired in this simulation, so the write-up can say whether an
                 existing lever missed, spoke and was ignored, or does not exist yet.

  usage:  x109_task_dossier.py task_012[,task_014] [--tag 20260806] [--wide] [--no-trace]
          (repo 디렉터리에서 실행할 것 — A2·엔진 소스가 없으면 출처 판정이 조용히 빈다)
"""

import collections
import glob
import gzip
import io
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from x106_n97live_batch_forensic import our_templates, eff          # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
TAU2 = os.environ.get("GO_TAU2", "/home/woori/scratch/tau2-bench")
DOMAIN = os.path.join(TAU2, "data", "tau2", "domains", "banking_knowledge")
SIMDIRS = [os.path.join(TAU2, "data", "simulations"),
           os.path.join(REPO, "reports", "facet_rft_2026", "sim_results")]
SIDECARS = ["/home/woori/scratch/logs/fb_*.jsonl",
            os.path.join(REPO, "reports", "facet_rft_2026", "sim_results", "fb_*.jsonl.gz")]
TAGRE = re.compile(r"\[([A-Z][A-Z0-9_\- ]{2,60})\]")
STOP = set("the a an and or of to for my me i in on is are was were with please can you it "
           "that this what how do does did my have has need want would like get".split())

ARGS = [a for a in sys.argv[1:] if not a.startswith("-")]
TAG = "20260806"
if "--tag" in sys.argv:
    TAG = sys.argv[sys.argv.index("--tag") + 1]
WIDE = "--wide" in sys.argv
NOTRACE = "--no-trace" in sys.argv
CUT = 1200 if WIDE else 420
TASKS = [t if t.startswith("task_") else "task_" + t for t in (ARGS[0].split(",") if ARGS else [])]


def jopen(path):
    op = gzip.open if path.endswith(".gz") else io.open
    with op(path, "rt", encoding="utf-8", errors="replace") as fh:
        return json.load(fh)


def load_sims():
    out = []
    seen = set()
    for base in SIMDIRS:
        pats = [os.path.join(base, "bank_n97_gpu*" + TAG + "*", "results.json"),
                os.path.join(base, "bank_n97_gpu*" + TAG + "*.results.json.gz")]
        for pat in pats:
            for p in sorted(glob.glob(pat)):
                src = os.path.basename(p).replace(".results.json.gz", "")
                if src == "results.json":
                    src = os.path.basename(os.path.dirname(p))
                if src in seen:
                    continue
                seen.add(src)
                try:
                    d = jopen(p)
                except Exception as e:
                    print("  (읽기 실패 %s: %s)" % (p, e))
                    continue
                for s in d.get("simulations") or []:
                    s["_src"] = src
                    out.append(s)
    return out


def load_tasks():
    by = {}
    for p in glob.glob(os.path.join(DOMAIN, "tasks*.json")) + \
            glob.glob(os.path.join(DOMAIN, "tasks", "*.json")):
        try:
            d = jopen(p)
        except Exception:
            continue
        for t in (d.get("tasks") if isinstance(d, dict) else d) or []:
            if isinstance(t, dict) and t.get("id"):
                by.setdefault(t["id"], (os.path.basename(p), t))
    return by


def load_sidecar():
    by = collections.defaultdict(list)
    for pat in SIDECARS:
        for p in sorted(glob.glob(pat)):
            op = gzip.open if p.endswith(".gz") else io.open
            try:
                with op(p, "rt", encoding="utf-8", errors="replace") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            r = json.loads(line)
                        except Exception:
                            continue
                        if r.get("sim"):
                            r["_file"] = os.path.basename(p)
                            by[r["sim"]].append(r)
            except Exception:
                continue
    return by


def corpus():
    docs = {}
    for p in glob.glob(os.path.join(DOMAIN, "documents", "*")):
        if os.path.isdir(p):
            continue
        try:
            docs[os.path.basename(p)] = io.open(p, encoding="utf-8", errors="replace").read()
        except Exception:
            pass
    return docs


def short(x, n=None):
    n = n or CUT
    s = x if isinstance(x, str) else json.dumps(x, ensure_ascii=False, sort_keys=True)
    s = re.sub(r"[ \t]+", " ", s.replace("\n", " ⏎ ")).strip()
    return s if len(s) <= n else s[:n] + " …(+%d자)" % (len(s) - n)


def is_ours(text, ours):
    body = text.lstrip()
    if body.startswith("Error:"):
        body = body[len("Error:"):].lstrip()
    return any(body.startswith(pre[:40]) or pre[:40] in body[:400] for pre in ours)


def gold_actions(task):
    ec = task.get("evaluation_criteria") or {}
    return ec.get("actions") or [], ec.get("communicate_info") or [], ec.get("nl_assertions") or []


def print_task_def(tid, tasks):
    print("=" * 100)
    print("== %s ==" % tid)
    if tid not in tasks:
        print("  (태스크 정의를 찾지 못함 — DOMAIN=%s)" % DOMAIN)
        return None
    fname, t = tasks[tid]
    print("  정의 파일: %s" % fname)
    ui = t.get("user_scenario") or {}
    ins = ui.get("instructions") or {}
    print("\n-- §0a 손님이 받은 시나리오(=외부 주장의 출처) --")
    for k in ("domain", "reason_for_call", "task_instructions", "known_info", "unknown_info"):
        if ins.get(k):
            print("  %-18s %s" % (k, short(ins[k], 1600)))
    if ui.get("persona"):
        print("  %-18s %s" % ("persona", short(ui["persona"], 400)))
    if t.get("description"):
        print("  %-18s %s" % ("description", short(t["description"], 600)))
    acts, comm, nls = gold_actions(t)
    print("\n-- §0b gold 액션 (%d) --" % len(acts))
    for i, a in enumerate(acts):
        print("  %2d. %-38s %s" % (i + 1, a.get("name"), short(a.get("arguments"), 700)))
        if a.get("requestor") and a["requestor"] != "assistant":
            print("      requestor=%s" % a["requestor"])
    if comm:
        print("-- §0c 전달해야 하는 정보 --")
        for c in comm:
            print("   · %s" % short(c, 300))
    if nls:
        print("-- §0d nl 어서션 --")
        for c in nls:
            print("   · %s" % short(c, 300))
    return t


def print_scoring(s):
    ri = s.get("reward_info") or {}
    print("\n-- §1 채점 (trial %s · %s · %s) --" % (s.get("trial"), s["_src"], s.get("id")))
    print("  reward=%s  종료=%s  메시지=%d  소요=%.0fs"
          % (ri.get("reward"), s.get("termination_reason"), len(s.get("messages") or []),
             s.get("duration") or 0))
    db = ri.get("db_check")
    if db is not None:
        print("  db_check=%s %s" % (getattr(db, "get", lambda k: None)("db_match") if isinstance(db, dict) else db,
                                    short((db or {}).get("db_reward") if isinstance(db, dict) else "", 200)))
    for i, a in enumerate(ri.get("action_checks") or []):
        print("  %s gold#%d %-36s %s" % ("✓" if a.get("action_match") else "✗", i + 1,
                                         (a.get("action") or {}).get("name"),
                                         "" if a.get("action_match") else short((a.get("action") or {}).get("arguments"), 240)))
    for key in ("nl_assertions", "communicate_checks"):
        for c in (ri.get(key) or []):
            if isinstance(c, dict) and c.get("met") is False:
                print("  ✗ %s: %s" % (key, short(c.get("nl_assertion") or c.get("note") or c, 240)))


def print_trace(s, ours):
    print("\n-- §2 궤적 --")
    turn = 0
    for m in s.get("messages") or []:
        role = m.get("role")
        if role == "assistant":
            turn += 1
            txt = m.get("content")
            if txt:
                print("  [%02d A] %s" % (turn, short(txt)))
            for tc in (m.get("tool_calls") or []):
                print("  [%02d A→call] %s %s" % (turn, eff(tc), short(tc.get("arguments"), CUT)))
        elif role == "user":
            txt = m.get("content")
            if txt:
                print("  [%02d U] %s" % (turn, short(txt)))
            for tc in (m.get("tool_calls") or []):
                print("  [%02d U→call] %s %s" % (turn, eff(tc), short(tc.get("arguments"), CUT)))
        elif role == "tool":
            txt = str(m.get("content") or "")
            src = "우리" if is_ours(txt, ours) else "env"
            mo = TAGRE.search(txt[:200])
            tag = ("[%s]" % mo.group(1)) if mo else ""
            head = "%s%s" % (src, tag)
            print("  [%02d T:%s] %s" % (turn, head, short(txt, CUT if src == "env" else max(CUT, 900))))


def print_sidecar(s, side):
    rows = side.get(s.get("id")) or []
    print("\n-- §3 우리 층(사이드카) — %d건 --" % len(rows))
    if not rows:
        print("  (이 sim의 사이드카 없음 — '우리 층이 말하지 않았다'는 결론 불가)")
        return
    kinds = collections.Counter(r.get("kind") for r in rows)
    print("  종류: %s" % dict(kinds))
    for r in rows:
        t = str(r.get("text") or "")
        if not t.strip():
            continue
        mo = TAGRE.search(t[:200])
        print("  turn %-3s %-18s %s%s" % (r.get("turn"), r.get("kind"),
                                          ("[%s] " % mo.group(1)) if mo else "", short(t, 700)))


def print_kb(s, docs):
    """검색 질의가 코퍼스에 실재하는가 — '못 찾은 것'과 '없는 것'을 가른다."""
    qs = []
    for m in s.get("messages") or []:
        for tc in (m.get("tool_calls") or []):
            a = tc.get("arguments")
            if isinstance(a, str):
                try:
                    a = json.loads(a)
                except Exception:
                    a = {}
            if isinstance(a, dict):
                for k in ("query", "search_query", "q", "keywords"):
                    if a.get(k):
                        qs.append((eff(tc), str(a[k])))
    if not qs:
        return
    print("\n-- §4 검색 질의 × 코퍼스 실재 (%d질의 · 문서 %d) --" % (len(qs), len(docs)))
    for name, q in qs:
        toks = [w for w in re.findall(r"[a-zA-Z_][a-zA-Z_0-9]{2,}", q.lower()) if w not in STOP]
        hits = {}
        for fn, txt in docs.items():
            low = txt.lower()
            n = sum(1 for w in set(toks) if w in low)
            if n:
                hits[fn] = n
        best = sorted(hits.items(), key=lambda kv: -kv[1])[:3]
        allw = [w for w in set(toks) if not any(w in t.lower() for t in docs.values())]
        print("  %-18s %-60s 최다일치 %s" % (name, short(q, 60),
                                           ", ".join("%s(%d/%d)" % (f, n, len(set(toks))) for f, n in best) or "없음"))
        if allw:
            print("      ⚠코퍼스에 **한 문서에도 없는 단어**: %s" % ", ".join(sorted(allw)[:12]))


def main():
    ours = our_templates(HERE)
    if not ours:
        print("⚠A2·엔진 소스에서 우리 문구를 하나도 못 읽었다 — 출처 판정 불가(repo에서 실행할 것)")
    sims = load_sims()
    tasks = load_tasks()
    docs = corpus()
    side = load_sidecar()
    if not TASKS:
        bytask = collections.defaultdict(list)
        for s in sims:
            bytask[s["task_id"]].append(s)
        print("태스크 인자가 없다. 사용 가능한 태스크 %d종 (sim %d)" % (len(bytask), len(sims)))
        fails = [(t, [x for x in v if (x.get("reward_info") or {}).get("reward") != 1.0])
                 for t, v in sorted(bytask.items())]
        print("실패 trial이 있는 태스크: %s"
              % ", ".join("%s(%d/%d)" % (t, len(f), len(bytask[t])) for t, f in fails if f))
        return
    for tid in TASKS:
        print_task_def(tid, tasks)
        mine = sorted([s for s in sims if s.get("task_id") == tid], key=lambda x: x.get("trial") or 0)
        if not mine:
            print("  (이 태그에 sim 없음: tag=%s)" % TAG)
            continue
        for s in mine:
            print_scoring(s)
            print_sidecar(s, side)
            print_kb(s, docs)
            if not NOTRACE:
                print_trace(s, ours)


if __name__ == "__main__":
    main()
