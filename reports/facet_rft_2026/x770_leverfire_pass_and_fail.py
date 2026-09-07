#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""x770 — 레버군 6종 발화의 **부호표**(pass 51 ↔ fail 46) + EPLAN 마커 전수 (2026-09-05).

x769 는 fail 46 만 봤다. [[70]] 은 **파는 것**을 요구하므로 통과 sim 에서도 같은 술어를 잰다.
캠페인 정본 집합을 여기서 재구성한다(태스크당 최신 sim · 2026-09-03T14 ~ 09-05T03).

⛔ 판정하지 않는다. 센다. ⛔ 코드 수정 0.
"""
import collections, io, json, os, re, sys
from pathlib import Path
from loguru import logger

logger.remove()
try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
except Exception:
    pass

from tau2.data_model.simulation import Results   # noqa: E402
import gate_interpreter as _GI                   # noqa: E402
import t2_gate_patch as G                        # noqa: E402
import t2_procedure as PROC                      # noqa: E402
import t2_eplan_patch as EP                      # noqa: E402

DOMAIN = "banking_knowledge"
SIMROOT = "/home/woori/scratch/tau2-bench/data/simulations"
LOGDIR = "/home/woori/scratch/logs"

A2 = _GI.load_domain_a2(DOMAIN) or {}
PROCS = A2.get("procedures") or []
ESPEC = EP.load_eplan_spec(DOMAIN) or {}
_le = ESPEC.get("list_enumerator")
ENUMS = set(_le) if isinstance(_le, (list, tuple)) else ({_le} if _le else set())

# ── ① 캠페인 정본 집합 재구성 ────────────────────────────────────────────────
TAGRE = re.compile(r"^bank_.*_(202609(?:03|04|05))_(\d{4})$")
LO, HI = ("20260903", 1400), ("20260905", 330)
tags = []
for d in sorted(os.listdir(SIMROOT)):
    m = TAGRE.match(d)
    if not m:
        continue
    day, hhmm = m.group(1), int(m.group(2))
    if day == LO[0] and hhmm < LO[1]:
        continue
    if day == HI[0] and hhmm > HI[1]:
        continue
    tags.append(d)

best = {}          # task -> (mtime, tag, simid, reward)
for tag in tags:
    p = "%s/%s/results.json" % (SIMROOT, tag)
    if not os.path.exists(p):
        continue
    mt = os.path.getmtime(p)
    try:
        res = Results.load(Path(p))
    except Exception:
        continue
    for s in res.simulations:
        tid = getattr(s, "task_id", None)
        if not tid:
            continue
        ts = getattr(s, "end_time", None) or getattr(s, "start_time", None) or ""
        key = (str(ts), mt)
        rw = getattr(getattr(s, "reward_info", None), "reward", None)
        if tid not in best or key > best[tid][0]:
            best[tid] = (key, tag, s.id, rw)

fails = sorted(t for t, v in best.items() if not (v[3] and v[3] >= 1.0))
passes = sorted(t for t, v in best.items() if (v[3] and v[3] >= 1.0))
print("CAMPAIGN tags=%d tasks=%d pass=%d fail=%d" % (len(tags), len(best), len(passes), len(fails)))
pin = set(ln.split()[1] for ln in open(sys.argv[1]).read().strip().splitlines() if ln.strip())
print("FAILSET match_x768=%s  onlyhere=%s  onlyx768=%s"
      % (set(fails) == pin, sorted(set(fails) - pin)[:8], sorted(pin - set(fails))[:8]))
print()

# ── ② 로그 패턴 ──────────────────────────────────────────────────────────────
RX_EPLAN = re.compile(r"\[T2_EPLAN\] (.{0,44})")
RX_TOOLOBS = re.compile(r"\[T2_TOOL_OBS\] id=\S* err=True -> (.*)")
RX_RESIGN = re.compile(r"window hit ?\(resign\)|window hit \(no effective write")
RX_FOLD = re.compile(r"\[T2_STACK\] window folded fb tag=(\S+)")
RX_PROC = re.compile(r"\[T2_PROCEDURE\] checklist proc=(\S+) nodes=\d+ done=\d+ left=(\[.*\])")
OURS_RE = re.compile(r"^Error: \[([A-Z0-9_\- ]+)\]")
FB_GENERIC = "Error: resolve the flagged call(s) first; do not call this tool yet."

HOWTO = {}


def howto(name, params_on=False):
    k = (name, params_on)
    if k in HOWTO:
        return HOWTO[k]
    old = os.environ.get("T2_DENY_HOWTO_PARAMS")
    os.environ["T2_DENY_HOWTO_PARAMS"] = "1" if params_on else "0"
    try:
        v = G._decl_howto(name, A2)
    except Exception:
        v = ""
    if old is None:
        os.environ.pop("T2_DENY_HOWTO_PARAMS", None)
    else:
        os.environ["T2_DENY_HOWTO_PARAMS"] = old
    HOWTO[k] = v
    return v


def sim_lines(tag, task):
    p = os.path.join(LOGDIR, "%s.log" % tag)
    if not os.path.exists(p):
        return None
    key = "[sim=%s#" % task
    with open(p, encoding="utf-8", errors="replace") as f:
        return [ln.rstrip("\n") for ln in f if key in ln]


def calls_of(m):
    return list(getattr(m, "tool_calls", None) or [])


cache = {}
ROWS = []
ALLNAMES = set()
EPMARK = collections.Counter()
DENYTAG = collections.Counter()
DENY_WITH_HOWTO = collections.Counter()

for tid in sorted(best):
    (_k, tag, simid, rw) = best[tid]
    ok = bool(rw and rw >= 1.0)
    if tag not in cache:
        try:
            cache[tag] = Results.load(Path("%s/%s/results.json" % (SIMROOT, tag)))
        except Exception:
            cache[tag] = None
    res = cache[tag]
    if res is None:
        continue
    sim = next((s for s in res.simulations if s.id == simid), None)
    if sim is None:
        continue
    msgs = list(sim.messages or [])

    # PROCEDURE_LEFT
    resign_pts, fire = 0, None
    for i, m in enumerate(msgs):
        if str(getattr(m, "role", "")) != "assistant" or calls_of(m):
            continue
        c = getattr(m, "content", None)
        if not (isinstance(c, str) and c.strip()):
            continue
        resign_pts += 1
        done = G._executed_tool_counts(msgs[:i])
        rows, pids = [], []
        for p in PROC.active_procedures(PROCS, done):
            for nid, tools, okk in PROC.checklist(p, done):
                if okk is False:
                    rows.append(nid)
                    if p.get("id") not in pids:
                        pids.append(p.get("id"))
        if rows and fire is None:
            fire = (i, resign_pts, pids, rows)

    # enum / cdut / parallel / names
    enum_n, cdut, gave, par, mx, asst = 0, [], [], 0, 0, 0
    for i, m in enumerate(msgs):
        for tc in calls_of(m):
            nm = G._eff_tool_name(tc)
            ALLNAMES.add(nm)
            ALLNAMES.add(str(getattr(tc, "name", "")))
            if nm in ENUMS or str(getattr(tc, "name", "")) in ENUMS:
                enum_n += 1
            n = str(getattr(tc, "name", ""))
            if n == "call_discoverable_user_tool":
                cdut.append(str((G._args_dict(tc) or {}).get("discoverable_tool_name") or ""))
            if n == "give_discoverable_user_tool":
                gave.append(str((G._args_dict(tc) or {}).get("discoverable_tool_name") or ""))
        if str(getattr(m, "role", "")) == "assistant":
            asst += 1
            k = len(calls_of(m))
            mx = max(mx, k)
            par += 1 if k >= 2 else 0

    lines = sim_lines(tag, tid)
    l1 = l2 = walk = 0
    resign_live = 0
    denies = []
    procleft_live = 0
    if lines is not None:
        for ln in lines:
            mm = RX_EPLAN.search(ln)
            if mm:
                txt = mm.group(1)
                EPMARK[txt.split(":")[0][:34]] += 1
                if txt.startswith("L1 deny"):
                    l1 += 1
                elif txt.startswith("L2 deny"):
                    l2 += 1
                elif txt.startswith("walk gap"):
                    walk += 1
            if RX_RESIGN.search(ln):
                resign_live += 1
            mp = RX_PROC.search(ln)
            if mp and mp.group(2) != "[]":
                procleft_live += 1
            mo = RX_TOOLOBS.search(ln)
            if mo:
                denies.append(mo.group(1))
    ROWS.append(dict(task=tid, ok=ok, tag=tag, asst=asst, resign=resign_pts, fire=fire,
                     enum=enum_n, cdut=cdut, gave=gave, par=par, mx=mx,
                     l1=l1, l2=l2, walk=walk, resign_live=resign_live,
                     procleft_live=procleft_live, denies=denies, nolog=(lines is None)))

# ── ③ deny 본문 → 도구 이름 추출 → howto 길이 ────────────────────────────────
NAMES = sorted((n for n in ALLNAMES if n and len(n) > 6), key=len, reverse=True)
for r in ROWS:
    hit = 0
    for b in r["denies"]:
        t = OURS_RE.match(b)
        key = ("[%s]" % t.group(1)) if t else ("_FB_GENERIC" if b.startswith(FB_GENERIC[:40]) else "env")
        DENYTAG[key] += 1
        if key == "env":
            continue
        nm = next((n for n in NAMES if n in b), None)
        if nm and howto(nm):
            hit += 1
            DENY_WITH_HOWTO[key] += 1
    r["howto_hits"] = hit

# ── ④ 출력 ───────────────────────────────────────────────────────────────────
print("%-9s %-4s %5s %6s %5s %5s %5s %5s %6s %7s %6s  %s"
      % ("task", "ok", "asst", "resign", "fire", "enum", "L1", "walk", "par", "maxcall", "deny", "howto_hits"))
for r in sorted(ROWS, key=lambda x: (not x["ok"], x["task"])):
    print("%-9s %-4s %5d %6d %5s %5d %5d %5d %6d %7d %6d  %d%s"
          % (r["task"], "P" if r["ok"] else "F", r["asst"], r["resign"],
             ("%s:%s" % (r["fire"][2][0][:14], ",".join(r["fire"][3])[:22])) if r["fire"] else "-",
             r["enum"], r["l1"], r["walk"], r["par"], r["mx"],
             len([b for b in r["denies"]]), r["howto_hits"],
             " NOLOG" if r["nolog"] else ""))

print()


def agg(sel, label):
    S = [r for r in ROWS if sel(r)]
    print("== %s (n=%d) ==" % (label, len(S)))
    print("  PROCLEFT fire tasks      %d" % sum(1 for r in S if r["fire"]))
    print("  PROCLEFT live left!=[]   %d tasks" % sum(1 for r in S if r["procleft_live"]))
    print("  EPLAN L1 deny            %d events / %d tasks"
          % (sum(r["l1"] for r in S), sum(1 for r in S if r["l1"])))
    print("  EPLAN walk gap           %d events / %d tasks"
          % (sum(r["walk"] for r in S), sum(1 for r in S if r["walk"])))
    print("  cdut(user-tool wrapper)  %d tasks" % sum(1 for r in S if r["cdut"]))
    print("  parallel turns           %d / %d asst turns · tasks %d"
          % (sum(r["par"] for r in S), sum(r["asst"] for r in S),
             sum(1 for r in S if r["par"])))
    print("  deny events (all)        %d / tasks %d"
          % (sum(len(r["denies"]) for r in S), sum(1 for r in S if r["denies"])))
    print("  deny w/ named howto      %d / tasks %d"
          % (sum(r["howto_hits"] for r in S), sum(1 for r in S if r["howto_hits"])))
    print("  no log                   %d" % sum(1 for r in S if r["nolog"]))


agg(lambda r: not r["ok"], "FAIL")
agg(lambda r: r["ok"], "PASS")
print()
print("== EPLAN 마커 전수(97 sim) ==")
for k, v in EPMARK.most_common(20):
    print("  %-40s %d" % (k, v))
print()
print("== deny 태그 전수 ==")
for k, v in DENYTAG.most_common(30):
    print("  %-46s %5d   (howto-named %d)" % (k, v, DENY_WITH_HOWTO.get(k, 0)))
print()
print("== howto 길이(궤적 도구) ==")
for n in sorted({k[0] for k in HOWTO}):
    a, b = len(howto(n, False)), len(howto(n, True))
    if a or b:
        print("  %-46s base=%4d params=%4d" % (n, a, b))
