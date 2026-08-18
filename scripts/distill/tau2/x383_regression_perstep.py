# -*- coding: utf-8 -*-
r"""x383 — **회귀 태스크의 pass 궤적 ↔ fail 궤적 per-step 대조**(사용자 지시 2026-08-18 ·
무료 · CPU · LLM 0 · GPU 0 · 엔진 수정 0).

## 왜 (x382 가 연 자리)

같은 시드(626729)에서 **런별로** 떨어진 태스크들이 있다 — `098`(08-17 4/4 → 08-18 0/1) ·
`100`(08-15 3/3 → 0/1) · `073`(08-15 treat 4/5 → 0/1). 양팔 모두 죽었으니 S2 레버가 아니라
**공유 스택 변경**이고, 창은 **08-17 04:10 ~ 08-18 11:29** 이다.

집계로는 여기서 더 못 간다([[08]]). 궤적을 **턴 단위로 나란히 놓고** ⑴어디서 갈렸는지
⑵그 자리에 **우리 층 마커가 무엇이 달라졌는지**를 본다.

## 무엇을 대조하나 (전부 결정론 · gold 는 방향만·C486)

  ① gold 도구 이름 집합 ↔ 각 궤적이 **실제 실행한** 이름 — 어느 gold 가 통째로 빠졌나
  ② 턴 단위 호출 나열 — **첫 분기 턴**
  ③ 그 sim 의 **우리 층 마커 집합 차이**(로그) — fail 에만 있는 것 / pass 에만 있는 것
  ④ 사이드카 문면 차이 — fail 에서만 나간 문장(있으면 그것이 처방 표적)

사용(리모트):
  python x383_regression_perstep.py task_098 bank_t7305_ctlaux_20260817a bank_t7310_ctl_20260818e
"""
import collections
import io
import json
import os
import re
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = "/home/woori/scratch/tau2-bench/data/simulations"
LOGS = "/home/woori/scratch/logs"
SEED = "626729"


def load(tag, task, seed=SEED):
    p = os.path.join(ROOT, tag, "results.json")
    doc = json.load(io.open(p, encoding="utf-8"))
    for s in (doc.get("simulations") or doc.get("results") or []):
        if str(s.get("task_id")) == task and str(s.get("seed")) == str(seed):
            return s
    return None


def nameof(tc):
    f = tc.get("function") or tc
    return str(f.get("name") or "")


def argsof(tc):
    f = tc.get("function") or tc
    a = f.get("arguments")
    if isinstance(a, str):
        try:
            a = json.loads(a)
        except Exception:
            a = {"_raw": a}
    return a if isinstance(a, dict) else {}


def inner(a):
    return str(a.get("agent_tool_name") or a.get("user_tool_name")
               or a.get("discoverable_tool_name") or "")


def calls(sim):
    """(turn, 표시이름, 인자요약) 나열 — 디스패처는 안쪽 이름까지 편다."""
    out = []
    for i, m in enumerate(sim.get("messages") or []):
        for tc in (m.get("tool_calls") or []):
            a = argsof(tc)
            nm = inner(a) or nameof(tc)
            keys = ",".join("%s=%s" % (k, str(v)[:18]) for k, v in sorted(a.items())
                            if k not in ("agent_tool_name", "user_tool_name",
                                         "discoverable_tool_name"))[:70]
            out.append((m.get("turn_idx", i), nm, keys))
    return out


def gold_names(sim):
    out = []
    for ck in ((sim.get("reward_info") or {}).get("action_checks") or []):
        a = ck.get("action") or {}
        ar = a.get("arguments") or {}
        out.append(str(inner(ar) or a.get("name") or "?"))
    return out


def markers(tag, task):
    """그 sim 의 우리 층 마커 카운터(로그가 있으면)."""
    c = collections.Counter()
    p = os.path.join(LOGS, tag + ".log")
    if not os.path.exists(p):
        return c, False
    rx = re.compile(r"\[(T2_[A-Z_]+)\]")
    key = "sim=%s#" % task
    for ln in io.open(p, encoding="utf-8", errors="replace"):
        if key not in ln:
            continue
        for m in rx.finditer(ln):
            c[m.group(1)] += 1
    return c, True


def sidecar(tag, task):
    rows = []
    p = os.path.join(LOGS, "fb_" + tag + ".jsonl")
    if not os.path.exists(p):
        return rows, False
    for ln in io.open(p, encoding="utf-8", errors="replace"):
        try:
            r = json.loads(ln)
        except Exception:
            continue
        if str(r.get("simtag", "")).startswith(task):
            rows.append(r)
    return rows, True


def main():
    task, tag_pass, tag_fail = sys.argv[1], sys.argv[2], sys.argv[3]
    a, b = load(tag_pass, task), load(tag_fail, task)
    if a is None or b is None:
        print("⛔궤적 없음 (pass=%s fail=%s)" % (a is not None, b is not None))
        return 1
    ra = (a.get("reward_info") or {}).get("reward")
    rb = (b.get("reward_info") or {}).get("reward")
    print("=" * 104)
    print("x383  %s   PASS=%s(%s)  ↔  FAIL=%s(%s)   seed=%s" % (task, tag_pass, ra, tag_fail, rb, SEED))
    print("=" * 104)

    ga, gb = gold_names(a), gold_names(b)
    ca, cb = calls(a), calls(b)
    da = [n for _t, n, _k in ca]
    db = [n for _t, n, _k in cb]
    print("① gold 도구 %d개 · 실행 이름 — pass %d · fail %d" % (len(ga), len(set(da)), len(set(db))))
    miss_a = [g for g in dict.fromkeys(ga) if g not in da]
    miss_b = [g for g in dict.fromkeys(ga) if g not in db]
    print("   gold 인데 **pass 도 안 부른 것**: %s" % (", ".join(miss_a) or "없음"))
    print("   gold 인데 **fail 이 안 부른 것**: %s" % (", ".join(miss_b) or "없음"))
    print("")

    print("② 턴 단위 호출 (좌 pass · 우 fail) — 첫 분기에 ★")
    star = False
    for i in range(max(len(ca), len(cb))):
        la = ("t%-3s %-34s %s" % (ca[i][0], ca[i][1][:34], ca[i][2][:26])) if i < len(ca) else ""
        lb = ("t%-3s %-34s %s" % (cb[i][0], cb[i][1][:34], cb[i][2][:26])) if i < len(cb) else ""
        mark = ""
        if not star and (i >= len(ca) or i >= len(cb) or ca[i][1] != cb[i][1]):
            mark, star = " ★", True
        print("   %-64s | %-64s%s" % (la[:64], lb[:64], mark))
    print("")

    ma, oka = markers(tag_pass, task)
    mb2, okb = markers(tag_fail, task)
    print("③ 우리 층 마커 차이 (로그 pass=%s fail=%s)" % (oka, okb))
    if oka and okb:
        keys = sorted(set(ma) | set(mb2))
        for k in keys:
            if ma[k] != mb2[k]:
                print("   %-26s pass %3d ↔ fail %3d %s" % (k, ma[k], mb2[k],
                                                           "← fail 에만/더" if mb2[k] > ma[k] else ""))
    else:
        print("   (한쪽 로그 없음 — 비교 생략)")
    print("")

    sa, oksa = sidecar(tag_pass, task)
    sb, oksb = sidecar(tag_fail, task)
    print("④ 사이드카 문면 (pass %d줄 · fail %d줄)" % (len(sa), len(sb)))
    if oksa and oksb:
        fa = collections.Counter(" ".join(str(r.get("text", "")).split())[:70] for r in sa)
        fb = collections.Counter(" ".join(str(r.get("text", "")).split())[:70] for r in sb)
        only_fail = [(t, n) for t, n in fb.most_common() if t not in fa]
        only_pass = [(t, n) for t, n in fa.most_common() if t not in fb]
        print("   ▸fail 에만 나간 문장 %d종" % len(only_fail))
        for t, n in only_fail[:6]:
            print("      [%dx] %s" % (n, t))
        print("   ▸pass 에만 나간 문장 %d종" % len(only_pass))
        for t, n in only_pass[:4]:
            print("      [%dx] %s" % (n, t))
    else:
        print("   (한쪽 사이드카 없음 — 비교 생략)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
