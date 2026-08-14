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

import t2_forensic as F                                            # noqa: E402

# 로딩·래퍼 해제는 정본 라이브러리에 있다(사본 금지·`t2_subcall` 과 같은 규율).
HERE, BASE, FBDIR = F.HERE, F.BASE, F.FBDIR
UNLOCK, GIVE, CALLA, CALLU = F.UNLOCK, F.GIVE, F.CALLA, F.CALLU
DONE_HINT = ("has been applied", "have been applied", "processed successfully", "successfully "
             "opened", "has been opened", "credited back", "been completed", "is now complete",
             "have been credited", "has been submitted", "been updated")


def jload(tag):
    return F.load(tag)


def fb_for(tag):
    """사이드카를 sim-key 별로(정본=`t2_forensic.sidecar`)."""
    return F.sidecar(tag)


def argsof(tc):
    return F.argsof(tc)


def nameof(tc):
    return F.nameof(tc)


def inner_name(args):
    return F.inner_name(args)


def label(name, args):
    """unlock/give/call 은 대상 도구까지(정본=`t2_forensic.label`)."""
    return F.label(name, args)


def canon(v):
    """표기 차이를 인자 불일치로 오판하지 않게 — 중첩 JSON **문자열**까지 풀고 수치는 float.

    ⚠이 정규화가 없으면 gold `amount: 9.50` 과 실호출 `9.5` 가 다른 값으로 잡힌다(초판 결함:
    073 의 **성공 실행**을 인자 불일치로 오분류했다).
    """
    if isinstance(v, dict):
        return {str(k): canon(x) for k, x in v.items()}
    if isinstance(v, list):
        return [canon(x) for x in v]
    if isinstance(v, bool) or v is None:
        return v
    if isinstance(v, (int, float)):
        return round(float(v), 6)
    s = " ".join(str(v).strip().split())
    t = s.strip()
    if t[:1] in "{[" :
        try:
            return canon(json.loads(t))
        except Exception:
            pass
    try:
        return round(float(t.replace("$", "").replace(",", "")), 6)
    except (TypeError, ValueError):
        return s.lower()


def norm(v):
    return json.dumps(canon(v), ensure_ascii=False, sort_keys=True)


def arg_hit(gold_args, got_args):
    """gold 인자가 전부 같은가(부분집합 비교 — 채점기와 같은 방향)."""
    for k, v in (gold_args or {}).items():
        if k not in got_args or norm(got_args[k]) != norm(v):
            return False
    return True


def steps(tags, only_fail=True):
    """per-step 정독 (사용자 지시 2026-08-14). 한 줄 = 한 메시지.

    우리 **비커밋 채널**(사이드카)은 궤적에 없다(C298) — 같은 turn 자리에 `>>` 로 끼워 넣어야
    *"우리가 그 자리에 무엇을 말했나"* 가 보인다. 이 병합 없이 읽으면 원인을 모델에 오귀속한다.
    """
    for tag in tags:
        d = jload(tag)
        fb = fb_for(tag)
        for s in d.get("simulations", []):
            ri = s.get("reward_info") or {}
            if only_fail and ri.get("reward"):
                continue
            tid = s.get("task_id")
            simtag = "%s#s%s" % (tid, s.get("seed"))
            byturn = collections.defaultdict(list)
            for r in (fb.get(simtag) or []):
                byturn[r.get("turn")].append(r)
            print("=" * 100)
            print("%s trial=%s reward=%s term=%s msgs=%d  [사이드카 %s]" % (
                tid, s.get("trial"), ri.get("reward"), s.get("termination_reason"),
                len(s.get("messages") or []), simtag if fb.get(simtag) else "매칭 없음"))
            res = {m.get("id"): m for m in (s.get("messages") or []) if m.get("role") == "tool"}
            for i, m in enumerate(s.get("messages") or []):
                role, c = m.get("role"), m.get("content")
                for tc in (m.get("tool_calls") or []):
                    r = res.get(tc.get("id")) or {}
                    bad = r.get("error") or str(r.get("content") or "").lstrip().startswith("Error:")
                    print("[%3d] %-9s CALL %-46s %s %s" % (
                        i, role, label(nameof(tc), argsof(tc))[:46],
                        "✗" if bad else "·",
                        json.dumps(argsof(tc), ensure_ascii=False)[:150]))
                if role == "tool":
                    bad = m.get("error") or str(c or "").lstrip().startswith("Error:")
                    print("[%3d] %-9s %s %s" % (i, role, "‼DENY" if bad else "res  ",
                                                " ".join(str(c or "").split())[:170]))
                elif c and isinstance(c, str) and c.strip():
                    print("[%3d] %-9s %s" % (i, role, " ".join(c.split())[:170]))
                for r in byturn.get(i, []):
                    print("      >> [우리·%s/%s] %s" % (
                        r.get("kind"), r.get("channel"),
                        " ".join((r.get("text") or "").split())[:210]))


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
            # 도구 결과를 id 로 이어 붙인다 — **시도**와 **실행**은 다르다(073 실측)
            res = {m.get("id"): m for m in msgs if m.get("role") == "tool"}
            # 에이전트 실호출 수집
            calls, called = [], collections.defaultdict(list)
            last_txt = ""
            for i, m in enumerate(msgs):
                if m.get("role") != "assistant":
                    continue
                for tc in (m.get("tool_calls") or []):
                    nm, ar = nameof(tc), argsof(tc)
                    lb = label(nm, ar)
                    r = res.get(tc.get("id")) or {}
                    okx = not r.get("error") and not str(r.get("content") or "").lstrip(
                        ).startswith("Error:")
                    calls.append((i, lb, okx))
                    # 래퍼는 **대상 도구별로** 색인한다(래퍼 이름만 보면 전부 같아 보인다)
                    called[lb].append((i, ar, okx))
                    if nm in (UNLOCK, GIVE):
                        called["<unlocked>" + inner_name(ar)].append((i, ar, okx))
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
                hits = [(gi, ga, gok) for gi, ga, gok in called.get(lab, [])
                        if arg_hit(ar, ga)]
                done = [h for h in hits if h[2]]
                if ok:
                    kind = "OK"
                elif who != "assistant":
                    kind = "MISS-USER"          # 손님 몫(user-sim 이 해야 하는 액션)
                elif lab not in called:
                    kind = ("MISS-UNLOCKONLY" if called.get("<unlocked>" + tgt)
                            else "MISS-NOTCALLED")
                elif done:
                    # gold 인자 그대로 **성공 실행**했는데 채점은 불일치 = 우리가 설명해야 할 칸
                    kind = "MISS-EXECUTED%s" % ("-DUP" if len(done) > 1 else "")
                elif hits:
                    kind = "MISS-DENIED"       # 인자는 맞았는데 실행이 거부/실패
                elif not any(arg_hit(ar, g) for _, g, _ in called[lab]):
                    kind = "MISS-ARGDIFF"
                else:
                    kind = "MISS-JUDGE"
                kinds[kind] += 1
                if kind.startswith("MISS") and first is None and who == "assistant":
                    first = lab
                got = ""
                if kind.startswith("MISS-EXECUTED"):
                    got = "  ← **성공 실행** msg=%s (동일 인자 %d회)" % (
                        [h[0] for h in done], len(done))
                elif kind == "MISS-DENIED":
                    got = "  ← 시도 msg=%s 전부 실패/거부" % [h[0] for h in hits]
                elif kind == "MISS-ARGDIFF":
                    # gold 와 **가장 가까운** 실호출을 보여준다(첫 호출이 아니라)
                    cand = sorted(called[lab], key=lambda g: -sum(
                        1 for k in ar if norm(g[1].get(k)) == norm(ar.get(k))))
                    gi, ga = cand[0][0], cand[0][1]
                    diff = {k: {"gold": ar.get(k), "got": ga.get(k)} for k in ar
                            if norm(ga.get(k)) != norm(ar.get(k))}
                    got = "  ← 실호출[%d](%d회 중) diff=%s" % (
                        gi, len(called[lab]), json.dumps(diff, ensure_ascii=False)[:300])
                lines.append("    %-2d %-4s %-52s%s" % (ci, "OK" if ok else "✗", lab[:52], got))
            key = "%s#t%s" % (tid, tr)
            # ⚠sim 단위로 매칭한다(태스크 접두로 모으면 시행이 서로 오염된다 — 초판 결함)
            simtag = "%s#s%s" % (tid, s.get("seed"))
            fbrows = list(fb.get(simtag) or [])
            if not fbrows:
                for k, v in fb.items():
                    if k.startswith(tid):
                        fbrows.extend(v)
                simtag += "(태스크합계·seed 매칭 실패)"
            ch = collections.Counter(r.get("channel") or r.get("kind") for r in fbrows)
            nl = [a for a in (ri.get("nl_assertions") or []) if not a.get("met", True)]
            fabr = (kinds.get("MISS-NOTCALLED", 0) + kinds.get("MISS-UNLOCKONLY", 0) > 0
                    and any(h in last_txt.lower() for h in DONE_HINT))
            print("-" * 92)
            print("%s  reward=%s  term=%s  msgs=%d  calls=%d  gold=%d(OK %d)" % (
                key, rw, term, len(msgs), len(calls), len(checks), kinds.get("OK", 0)))
            print("  종류: %s" % dict(kinds))
            print("  첫 이탈: %s" % (first or "-"))
            dup = {"%s %s" % (lb, json.dumps(a, ensure_ascii=False)[:80]): len(g)
                   for lb, v in called.items() if not lb.startswith("<")
                   for a, g in [(v[0][1], [x for x in v if x[2]
                                           and norm(x[1]) == norm(v[0][1])])]
                   if len(g) > 1}
            if dup:
                print("  ⚠중복 성공 실행: %s" % json.dumps(dup, ensure_ascii=False)[:300])
            for ln in lines:
                print(ln)
            if nl:
                print("  NL 실패: %s" % json.dumps(nl, ensure_ascii=False)[:300])
            if ch:
                print("  채널(%s): %s" % (simtag, dict(ch.most_common(8))))
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
    ks = ["OK", "MISS-NOTCALLED", "MISS-UNLOCKONLY", "MISS-ARGDIFF", "MISS-EXECUTED",
          "MISS-EXECUTED-DUP", "MISS-DENIED", "MISS-JUDGE", "MISS-USER"]
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
    ALL = ["bank_t7285_a_20260814g", "bank_t7285_b_20260814g",
           "bank_t7286_a_20260814h", "bank_t7286_b_20260814h"]
    a = [x for x in sys.argv[1:] if x != "--steps"]
    (steps if "--steps" in sys.argv[1:] else run)(a or ALL)
