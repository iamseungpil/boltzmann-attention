# -*- coding: utf-8 -*-
"""실패 sim 전수 정밀 포렌식 ([[08]]) — t7285(072~075 nt=4) + t7286(G대표 12).

집계(pass/match)에서 결론 직행 금지. sim 마다 **gold 액션 한 줄씩** 다음을 가른다:

  MISS-NOTCALLED   그 도구 이름이 궤적에 **한 번도** 안 나옴        (도달 실패)
  MISS-UNLOCKONLY  unlock 만 하고 **호출 안 함**                    (착수 실패)
  MISS-ARGDIFF     호출했는데 인자 불일치                            (F2/F3 축)
  OK               인자까지 일치

+ 종료사유·마지막 손님-가시 본문·우리 채널(사이드카 `fb_<tag>.jsonl`) 발화 분포.
+ 교차표: 태스크 × 실패종류 · 종료사유 · 첫 이탈 도구.

사용(리모트·사이드카 있는 곳): py bank_fail_forensic_all.py <tag> [<tag>...] > report.txt
로컬은 사이드카가 없어 채널 칸이 비고 나머지는 동일하게 나온다.
"""
import collections
import gzip
import io
import json
import os
import sys

try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026", "sim_results")
FBDIR = "/home/woori/scratch/logs"
UNLOCK = "unlock_discoverable_agent_tool"
GIVE = "give_discoverable_user_tool"
CALLA = "call_discoverable_agent_tool"
CALLU = "call_discoverable_user_tool"
DONE_HINT = ("has been applied", "have been applied", "processed successfully", "successfully "
             "opened", "has been opened", "credited back", "been completed", "is now complete",
             "have been credited", "has been submitted", "been updated")


def jload(tag):
    with gzip.open(os.path.join(BASE, tag + "_results.json.gz"), "rt", encoding="utf-8") as f:
        return json.load(f)


def fb_for(tag):
    """사이드카를 sim-key 별로 모은다(없으면 빈 dict)."""
    p = os.path.join(FBDIR, "fb_%s.jsonl" % tag)
    out = collections.defaultdict(list)
    if not os.path.exists(p):
        return out
    for ln in io.open(p, encoding="utf-8", errors="replace"):
        try:
            o = json.loads(ln)
        except Exception:
            continue
        out[o.get("simtag") or "?"].append(o)
    return out


def argsof(tc):
    a = (tc.get("function") or {}).get("arguments", tc.get("arguments"))
    if isinstance(a, str):
        try:
            a = json.loads(a)
        except Exception:
            a = {"_raw": a}
    return a if isinstance(a, dict) else {}


def nameof(tc):
    return (tc.get("function") or {}).get("name") or tc.get("name") or ""


def inner_name(args):
    return (args.get("agent_tool_name") or args.get("user_tool_name")
            or args.get("tool_name") or "")


def label(name, args):
    """unlock/give/call 은 **대상 도구까지** 붙여야 의미가 있다(래퍼 이름만으론 무정보)."""
    t = inner_name(args)
    if not t:
        return name
    pre = {UNLOCK: "unlock", GIVE: "give", CALLA: "call", CALLU: "callu"}.get(name)
    return "%s:%s" % (pre, t) if pre else name


def norm(v):
    """중첩 dict/list 는 키 정렬로, 수치는 float 로 — 표기 차이를 인자 불일치로 오판하지 않게."""
    if isinstance(v, (dict, list)):
        return json.dumps(v, ensure_ascii=False, sort_keys=True).lower()
    s = " ".join(str(v).strip().lower().split())
    try:
        return "%.6g" % float(s.replace("$", "").replace(",", ""))
    except (TypeError, ValueError):
        return s


def arg_hit(gold_args, got_args):
    """gold 인자가 전부 같은가(부분집합 비교 — 채점기와 같은 방향)."""
    for k, v in (gold_args or {}).items():
        if k not in got_args or norm(got_args[k]) != norm(v):
            return False
    return True


def run(tags):
    tot = collections.Counter()
    xtab = collections.Counter()
    firstmiss = collections.Counter()
    endtab = collections.Counter()
    for tag in tags:
        d = jload(tag)
        fb = fb_for(tag)
        print("#" * 92)
        print("# %s" % tag)
        for s in d.get("simulations", []):
            tid, tr = s.get("task_id"), s.get("trial")
            rw = (s.get("reward_info") or {}).get("reward")
            term = s.get("termination_reason")
            msgs = s.get("messages") or []
            # 에이전트 실호출 수집
            calls, called = [], collections.defaultdict(list)
            last_txt = ""
            for i, m in enumerate(msgs):
                if m.get("role") != "assistant":
                    continue
                for tc in (m.get("tool_calls") or []):
                    nm, ar = nameof(tc), argsof(tc)
                    lb = label(nm, ar)
                    calls.append((i, lb))
                    # 래퍼는 **대상 도구별로** 색인한다(래퍼 이름만 보면 전부 같아 보인다)
                    called[lb].append((i, ar))
                    if nm in (UNLOCK, GIVE):
                        called["<unlocked>" + inner_name(ar)].append((i, ar))
                c = m.get("content")
                if isinstance(c, str) and c.strip():
                    last_txt = c.strip()
            ri = s.get("reward_info") or {}
            checks = ri.get("action_checks") or []
            lines, kinds, first = [], collections.Counter(), None
            for ci, ck in enumerate(checks):
                a = ck.get("action") or {}
                nm, ar = a.get("name"), a.get("arguments") or {}
                who = a.get("requestor")
                ok = bool(ck.get("action_match"))
                lab = label(nm, ar)
                tgt = inner_name(ar) or nm
                if ok:
                    kind = "OK"
                elif who != "assistant":
                    kind = "MISS-USER"          # 손님 몫(user-sim 이 해야 하는 액션)
                elif lab not in called:
                    kind = ("MISS-UNLOCKONLY" if called.get("<unlocked>" + tgt)
                            else "MISS-NOTCALLED")
                elif not any(arg_hit(ar, g) for _, g in called[lab]):
                    kind = "MISS-ARGDIFF"
                else:
                    kind = "MISS-JUDGE"        # 인자는 맞는데 채점 불일치(순서·중복 등)
                kinds[kind] += 1
                if kind.startswith("MISS") and first is None and who == "assistant":
                    first = lab
                got = ""
                if kind == "MISS-ARGDIFF":
                    # gold 와 **가장 가까운** 실호출을 보여준다(첫 호출이 아니라)
                    cand = sorted(called[lab], key=lambda g: -sum(
                        1 for k in ar if norm(g[1].get(k)) == norm(ar.get(k))))
                    gi, ga = cand[0]
                    diff = {k: {"gold": ar.get(k), "got": ga.get(k)} for k in ar
                            if norm(ga.get(k)) != norm(ar.get(k))}
                    got = "  ← 실호출[%d](%d회 중) diff=%s" % (
                        gi, len(called[lab]), json.dumps(diff, ensure_ascii=False)[:300])
                lines.append("    %-2d %-4s %-52s%s" % (ci, "OK" if ok else "✗", lab[:52], got))
            key = "%s#t%s" % (tid, tr)
            fbrows = []
            for k, v in fb.items():
                if k.startswith(tid):
                    fbrows.extend(v)
            ch = collections.Counter(r.get("channel") or r.get("kind") for r in fbrows)
            nl = [a for a in (ri.get("nl_assertions") or []) if not a.get("met", True)]
            fabr = (kinds.get("MISS-NOTCALLED", 0) + kinds.get("MISS-UNLOCKONLY", 0) > 0
                    and any(h in last_txt.lower() for h in DONE_HINT))
            print("-" * 92)
            print("%s  reward=%s  term=%s  msgs=%d  calls=%d  gold=%d(OK %d)" % (
                key, rw, term, len(msgs), len(calls), len(checks), kinds.get("OK", 0)))
            print("  종류: %s" % dict(kinds))
            print("  첫 이탈: %s" % (first or "-"))
            for ln in lines:
                print(ln)
            if nl:
                print("  NL 실패: %s" % json.dumps(nl, ensure_ascii=False)[:300])
            if ch:
                print("  채널: %s" % dict(ch.most_common(8)))
            print("  종료 직전 본문%s: %s" % (" ⚠완료날조 후보" if fabr else "",
                                          " ".join(last_txt.split())[:200]))
            tot[term] += 1
            endtab[(tid, term)] += 1
            for k, v in kinds.items():
                xtab[(tid, k)] += v
            if first:
                firstmiss[first] += 1
    print("#" * 92)
    print("# 교차표 1 — 태스크 × 판정")
    ks = ["OK", "MISS-NOTCALLED", "MISS-UNLOCKONLY", "MISS-ARGDIFF", "MISS-JUDGE", "MISS-USER"]
    print("  %-10s %s" % ("task", " ".join("%-16s" % k for k in ks)))
    for t in sorted({k[0] for k in xtab}):
        print("  %-10s %s" % (t, " ".join("%-16d" % xtab.get((t, k), 0) for k in ks)))
    print("# 교차표 2 — 첫 이탈 도구")
    for k, v in firstmiss.most_common():
        print("  %-4d %s" % (v, k))
    print("# 교차표 3 — 종료사유")
    for k, v in tot.most_common():
        print("  %-4d %s" % (v, k))


if __name__ == "__main__":
    run(sys.argv[1:] or ["bank_t7285_a_20260814g", "bank_t7285_b_20260814g",
                         "bank_t7286_a_20260814h", "bank_t7286_b_20260814h"])
