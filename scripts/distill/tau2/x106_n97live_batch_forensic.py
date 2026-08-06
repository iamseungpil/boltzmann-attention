# -*- coding: utf-8 -*-
"""Per-step forensic over the simulations a live sweep has already finished.

The 97-task sweep persists only when a whole block ends, so for hours there is data and no
way to read it. This reads the in-progress `results.json` directly and produces the batch
forensic in the form `N97_TASKWISE_FORENSIC_2026_08_04` established: cause per task, from
turns, not from a rate.

Three warnings are printed before any number, because each of them has produced a wrong
conclusion in this project before:

  완주 순서 편향   finished simulations are the fast ones. A partial mean is not the mean.
  nt2 짝 미완      a task whose two trials have not both landed cannot be called stable or flaky.
  사이드카 부재    this driver does not set `T2_FB_SIDECAR`, so the reminder channel is invisible.
                   Only the deny messages — which are committed as tool results — can be counted,
                   and "our layer said nothing" is therefore **not** an available conclusion.

  usage: x106_n97live_batch_forensic.py [tag_date] [--full]
"""

import collections
import glob
import io
import json
import os
import re
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

DATE = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else "20260806"
FULL = "--full" in sys.argv
OURS = []
SIMDIRS = ["/home/woori/scratch/tau2-bench/data/simulations",
           os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "..", "..", "reports", "facet_rft_2026", "sim_results")]
# ★1차 실행 교정: `{2,30}`은 `[POLICY GATE GB1_VERIFY_BEFORE_ACCOUNT_ACCESS]`(44자)를 놓쳐
#   그 발화를 전부 '무태그'로 몰았다. 그리고 tool 메시지의 `Error:`에는 **환경 오류도 섞인다** —
#   우리 문구와 env 오류를 한 칸에 세면 "우리가 얼마나 말했나"가 곧바로 틀린다([[55]]).
TAGRE = re.compile(r"\[([A-Z][A-Z0-9_\- ]{2,60})\]")
# ★2차 교정: "env 오류인가"를 **문면으로 추측하면 틀린다**(1차에서 양방향으로 틀렸다 —
#   `[READ-FIRST]`(우리 것)를 env로 보내고, `Unexpected parameter`(env 것)를 우리 것으로 셌다).
#   판정은 출처로 한다: **우리 A2와 엔진 소스에 그 문자열이 있으면 우리 것**이다. 없으면 env.
def our_templates(here):
    """A2 선언 + 엔진 리터럴에서 '우리가 보낼 수 있는 문구'의 접두 지문을 만든다."""
    pref = set()

    def add(s):
        s = str(s or "").lstrip()
        if len(s) < 30:
            return
        # placeholder 앞까지만 취해 접두 대조(값이 바뀌어도 같은 문구로 묶인다)
        cut = re.split(r"\{|%s|%d|%\(", s)[0].strip()
        if len(cut) >= 24:
            pref.add(cut[:60])

    def walk(n):
        if isinstance(n, str):
            add(n)
        elif isinstance(n, dict):
            for v in n.values():
                walk(v)
        elif isinstance(n, list):
            for v in n:
                walk(v)

    for p in glob.glob(os.path.join(here, "a2", "*.json")) + \
            glob.glob(os.path.join(here, "a2", "base", "*.json")):
        try:
            walk(json.load(io.open(p, encoding="utf-8")))
        except Exception:
            pass
    for p in glob.glob(os.path.join(here, "t2_*.py")) + [os.path.join(here, "gate_interpreter.py")]:
        try:
            src = io.open(p, encoding="utf-8", errors="replace").read()
        except Exception:
            continue
        for m in re.finditer(r'"((?:[^"\\]|\\.){30,300})"|\'((?:[^\'\\]|\\.){30,300})\'', src):
            add((m.group(1) or m.group(2)).replace('\\n', ' ').replace('\\"', '"'))
    return sorted(pref, key=len, reverse=True)


def load():
    sims = []
    for base in SIMDIRS:
        for p in sorted(glob.glob(os.path.join(base, "bank_n97_gpu*_%s" % DATE, "results.json"))):
            try:
                d = json.load(io.open(p, encoding="utf-8"))
            except Exception as e:
                print("  (읽기 실패 %s: %s)" % (p, e))
                continue
            for s in d.get("simulations") or []:
                s["_src"] = os.path.basename(os.path.dirname(p))
                sims.append(s)
        if sims:
            break
    return sims


def eff(tc):
    a = tc.get("arguments")
    if isinstance(a, str):
        try:
            a = json.loads(a)
        except Exception:
            a = {}
    if isinstance(a, dict):
        for k in ("agent_tool_name", "discoverable_tool_name"):
            if a.get(k):
                return str(a[k])
    return tc.get("name")


def main():
    global OURS
    OURS = our_templates(os.path.dirname(os.path.abspath(__file__)))
    sims = load()
    if not sims:
        print("데이터 없음 (tag=%s)" % DATE)
        return
    print("== 계측 규약과 편향(먼저 읽을 것) ==")
    print("  · **완주 순서 편향**: 지금 끝난 sim은 *빨리 끝난* sim이다. 부분 평균은 평균이 아니다.")
    print("  · **nt2 짝 미완**: 두 trial이 다 오지 않은 태스크는 안정/흔들림을 말할 수 없다.")
    print("  · **사이드카 부재**: 이 드라이버는 `T2_FB_SIDECAR`를 켜지 않는다 — reminder 채널은")
    print("    보이지 않고 **deny만** 궤적에 남는다. '우리 층이 말하지 않았다'는 결론은 불가.")
    print("  · 수치는 전부 [D](nt2 미완)이고, 원인 진술만 궤적 근거 [M]이다.\n")

    print("== §1 종료사유·완주 현황 ==")
    print("  완주 sim %d개" % len(sims))
    term = collections.Counter(s.get("termination_reason") for s in sims)
    print("  종료사유:", dict(term))
    bad = [s for s in sims if s.get("termination_reason") not in ("user_stop", "agent_stop")]
    print("  ⇒ 크래시/인프라 계열 %d건 (있으면 원인 집계에서 제외해야 한다)" % len(bad))
    bytask = collections.defaultdict(list)
    for s in sims:
        bytask[s["task_id"]].append(s)
    pairs = [t for t, v in bytask.items() if len(v) >= 2]
    print("  태스크 %d종 · 두 trial 다 온 태스크 %d종" % (len(bytask), len(pairs)))

    print("\n== §2 시간축 — 폭주 후보 ==")
    durs = sorted(((s.get("duration") or 0), s["task_id"], s.get("trial"),
                   len(s.get("messages") or []), s["reward_info"]["reward"]) for s in sims)
    med = durs[len(durs) // 2][0] if durs else 0
    print("  소요시간 중앙값 %.0fs · 최장 %.0fs" % (med, durs[-1][0] if durs else 0))
    print("  중앙값의 5배를 넘는 sim(= 폭주 후보):")
    runaway = [d for d in durs if med and d[0] > 5 * med]
    for d, t, tr, n, r in runaway:
        print("    %-10s t%s  %6.0fs  메시지 %3d  reward %.0f" % (t, tr, d, n, r))
    if not runaway:
        print("    (없음)")

    print("\n== §3 결과 — 채점 기준별 ==")
    basis = collections.Counter()
    for s in sims:
        ri = s["reward_info"]
        b = "DB" if ri.get("db_check") else ""
        b += "+ACTION" if ri.get("action_checks") else ""
        basis[(b or "?", ri["reward"] == 1.0)] += 1
    for (b, ok), n in sorted(basis.items()):
        print("  %-10s %s %d" % (b, "pass" if ok else "fail", n))
    npass = sum(1 for s in sims if s["reward_info"]["reward"] == 1.0)
    print("  현재 pass %d/%d (%.1f%%) — ⚠부분 집합이라 최종 수치 아님" % (npass, len(sims), 100.0 * npass / len(sims)))

    print("\n== §4 실패 sim의 첫 이탈 지점 (gold 순서 대비) ==")
    div = collections.Counter()
    rows = []
    for s in sims:
        ri = s["reward_info"]
        if ri["reward"] == 1.0:
            continue
        acts = ri.get("action_checks") or []
        if not acts:
            div["gold 액션 없음(DB 단독 채점)"] += 1
            continue
        k = next((i for i, a in enumerate(acts) if not a["action_match"]), None)
        if k is None:
            div["gold 전건 일치인데 실패(=DB 상태)"] += 1
            rows.append((s["task_id"], s.get("trial"), "DB", len(acts), ""))
        else:
            tag = "첫 miss #%d/%d" % (k + 1, len(acts))
            div[("모든 gold 미실행" if k == 0 else "중도 이탈")] += 1
            rows.append((s["task_id"], s.get("trial"), tag, len(acts),
                         acts[k]["action"]["name"]))
    for k, v in div.most_common():
        print("  %-28s %d" % (k, v))
    if FULL:
        for r in sorted(rows):
            print("    %-10s t%s %-16s gold %2d  첫 miss=%s" % r)

    print("\n== §5 우리 문구(궤적에 남는 deny만) × 결과 ==")
    tagcnt = collections.defaultdict(lambda: [0, 0, 0])      # tag -> [발화, 통과sim, 실패sim]
    persim_rep = []
    untagged = collections.Counter()
    envcnt = collections.Counter()
    for s in sims:
        ok = s["reward_info"]["reward"] == 1.0
        c = collections.Counter()
        for m in s.get("messages") or []:
            if m.get("role") != "tool":
                continue
            t = str(m.get("content") or "")
            if not t.lstrip().startswith("Error:"):
                continue
            body = t.lstrip()
            body = body[len("Error:"):].lstrip() if body.startswith("Error:") else body
            if not any(body.startswith(pre[:40]) or pre[:40] in body[:200] for pre in OURS):
                envcnt[re.sub(r"'[^']*'", "'…'", t.lstrip()[:60])] += 1
                continue                                   # 우리 소스에 없는 문자열 = env
            mo = TAGRE.search(t[:160])
            key = mo.group(1) if mo else "(무태그)"
            if not mo:
                untagged[re.sub(r"'[^']*'", "'…'", t.lstrip()[:70])] += 1
            c[key] += 1
        for k, v in c.items():
            tagcnt[k][0] += v
            tagcnt[k][1 if ok else 2] += v
        if c:
            top = c.most_common(1)[0]
            if top[1] >= 5:
                persim_rep.append((top[1], s["task_id"], s.get("trial"), top[0], ok))
    print("  %-46s %6s %8s %8s" % ("deny 태그(우리 문구만)", "발화", "통과sim", "실패sim"))
    for k, (a, p, f) in sorted(tagcnt.items(), key=lambda kv: -kv[1][0])[:18]:
        print("  %-46s %6d %8d %8d" % (k, a, p, f))
    print("\n  · 무태그 우리 문구 상위(문면으로 식별):")
    for t, n in untagged.most_common(8):
        print("      %4d  %r" % (n, t))
    print("  · **env 오류**(우리 문구 아님·비교용) 상위:")
    for t, n in envcnt.most_common(6):
        print("      %4d  %r" % (n, t))

    print("\n== §6 한 sim 안에서 같은 deny가 5회 이상 (022형 루프 판별선) ==")
    for n, t, tr, k, ok in sorted(persim_rep, reverse=True)[:20]:
        print("    %-10s t%s  %-24s %3d회  %s" % (t, tr, k, n, "pass" if ok else "fail"))
    if not persim_rep:
        print("    (없음)")

    print("\n== §7 반복 호출 (같은 도구·같은 인자) 상위 ==")
    rep = []
    for s in sims:
        c = collections.Counter()
        for m in s.get("messages") or []:
            for tc in (m.get("tool_calls") or []):
                c[(eff(tc), json.dumps(tc.get("arguments"), ensure_ascii=False, sort_keys=True)[:120])] += 1
        if c:
            (nm, _), n = c.most_common(1)[0]
            if n >= 5:
                rep.append((n, s["task_id"], s.get("trial"), nm, s["reward_info"]["reward"] == 1.0))
    for n, t, tr, nm, ok in sorted(rep, reverse=True)[:20]:
        print("    %-10s t%s  %-42s %3d회  %s" % (t, tr, nm, n, "pass" if ok else "fail"))
    if not rep:
        print("    (없음)")


if __name__ == "__main__":
    main()
