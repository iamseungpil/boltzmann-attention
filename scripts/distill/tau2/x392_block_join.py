# -*- coding: utf-8 -*-
r"""x392 — **막힌 자리 조인**: 우리 층 deny 의 *표적을 복원*하고 gold 액션마다 **결말**을 붙인다.

## 왜 이 모양인가 (계기 결함 4건을 여기서 고친다)

1. `x391` 은 로그 deny 줄만 세고 *무엇을 막았는지* 를 못 말했다 — 그 줄의 `arg=` 는 **인자 이름**
   (`agent_tool_name`)이지 값이 아니다. ⇒ 마크마다 값이 어디 찍히는지 **축자 인벤토리**로 갈랐다.
2. `x391` 의 로그 파싱은 `[sim=task_074#s…]` 의 **seed 를 버려** nt=2 두 시행을 한 통에 섞었다
   (trial0=s626729·trial1=s373753). ⇒ 조인 키는 `F.simtag()` 전체(C491⒠ 와 같은 함정).
3. `"deny" 문자열 포함` 으로 세면 **위양성**이 절반이다 — 이 런 실측:
   `[T2_STACK] … deny stays, body kept`(창 접힘 주석) · `[T2_SEARCH_ON_PROCEED] deny 아님`
   · `[T2_EPLAN] deny cap N reached`(상한 주석) · `[T2_PREKB] deny waived`. ⇒ **줄머리 `] deny`**
   + 주석 계열 축자 제외.
4. **마크마다 값의 뜻이 다르다.** `[T2_DISCOVERY_STEP2] deny name=X` 의 X 는 *우리가 짚어 준*
   이름(푸시)이지 막힌 이름이 아니다. `[T2_WRITE_ARG_ENUM] deny val='Bronze Savings Account'` 의
   val 은 **인자 값**이다. ⇒ TARGET / PUSH / ARGVAL / JOIN 으로 나눠 센다.

## 결말 코드 (첫 매치·새 이름 만들지 않는다 [[48]])

    NEVER       그 이름으로 호출 0회
    OURS        호출은 했는데 **그 턴에 우리 층 deny** 가 있고 실행된 적이 없다
    ENV_REJECT  호출은 했고 **환경이 거절**했다 (Unknown tool / Invalid arguments / …)
    ARGDIFF     실행됐는데 인자가 **의미까지 다르다**
    NOTATION    실행됐고 `norm_args` 로는 같은 실행인데 `action_match=false` (C486 표기 artifact)
    MATCH       채점기가 매치로 인정

⚠결말은 **표**이지 판정이 아니다. OURS·deny 동반이 곧 원인은 아니다 — 회복(`이후실행`)을 같이 찍는다.

사용: py -3 x392_block_join.py <tag> [<tag> …]
"""
import collections
import io
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import t2_forensic as F  # noqa: E402

SUF = ".results.json.gz"

DENY_RE = re.compile(r"^\[(T2_[A-Z0-9_]+)\] deny\b")
REASON_RE = re.compile(r"reason=(\S+)")
KV_RE = re.compile(r"\b(inner|name|val|tool|call)=(?:'([^']*)'|(\S+))")
# 주석·상한·면제 — 호출을 막은 줄이 아니다(축자 제외 근거는 §왜 3).
NOT_A_DENY = ("deny 아님", "deny cap", "deny waived", "deny stays")

# 마크 → (값의 뜻, 값이 찍히는 키 순서). 근거 = 각 print 사이트 축자(2026-08-19 인벤토리).
#   TARGET 막힌 대상 도구 이름 / PUSH 우리가 짚어 준 이름(차단 아님) / ARGVAL 인자 값 /
#   JOIN   줄에 대상이 없다(래퍼만) → 그 턴 호출로 조인
MARK_SPEC = {
    "T2_WRITE_EVIDENCE": ("TARGET", ("inner", "tool")),
    "T2_WRITE_ARG_GROUND": ("TARGET", ("tool",)),
    "T2_ARG_EMPTY": ("TARGET", ("tool",)),
    "T2_DISPATCH_ROLE": ("TARGET", ("name",)),
    "T2_UNLOCK_NAME": ("TARGET", ("val",)),
    "T2_UNLOCK_PROV": ("TARGET", ("val",)),
    "T2_UNKNOWN_NAME_BL": ("TARGET", ("val",)),
    "T2_READALL": ("TARGET", ("tool",)),
    "T2_TOOLERR": ("TARGET", ("tool",)),
    "T2_TOOLLIST": ("TARGET", ("tool",)),
    "T2_PRESCRIPTION": ("TARGET", ("tool",)),
    "T2_DISCOVERY_DISPATCH": ("TARGET", ("call",)),
    "T2_RESOLVE": ("JOIN", ()),          # tool= 은 래퍼(unlock/call) — 대상은 안 찍힌다
    "T2_TOOL_SIGNATURE": ("JOIN", ()),   # tool= 은 give/call 래퍼
    "T2_PROCEDURE": ("JOIN", ()),
    "T2_PARAM_CAP": ("JOIN", ()),
    "T2_TRANSCRIBE": ("JOIN", ()),
    "T2_DISCOVERY_STEP2": ("PUSH", ("name",)),
    "T2_WRITE_ARG_ENUM": ("ARGVAL", ("val",)),
    "T2_VERDICT_GATE": ("ARGVAL", ("val",)),
    "T2_ARG_AXIS": ("ARGVAL", ("got",)),
}

# 환경(벤치)이 낸 거절 — 우리 문면이 아니다. 축자 접두사만 본다(해석 0).
ENV_PAT = ("Error: Unknown agent tool", "Error: Unknown discoverable tool", "Error: Unknown tool",
           "Error: Invalid arguments", "Error: Missing required parameter",
           "Error: Unexpected parameter", "has not been given to you", "Error: Invalid ",
           "not found.", "Error: Account eligibility requirements not met",
           "Error: There is already", "cannot be closed")


def diff_keys(actual, gold):
    """**키 단위** 대조 — 어느 인자가 갈렸는지만 인쇄한다(값 해석 0)."""
    a = F.norm_args(actual)
    if isinstance(a, dict) and "arguments" in a:
        inner = a["arguments"]
        if isinstance(inner, dict):
            a = inner
        elif isinstance(inner, str):
            return "비교불가(중첩 인자 문자열 미해석): %s" % inner[:40]
    b = F.norm_args(gold)
    if not isinstance(a, dict) or not isinstance(b, dict):
        return "형태불일치"
    out = []
    for k in sorted(set(a) | set(b)):
        if k in ("agent_tool_name", "user_tool_name", "discoverable_tool_name"):
            continue
        if a.get(k) != b.get(k):
            out.append("%s: %s≠%s" % (k, str(a.get(k))[:22], str(b.get(k))[:22]))
    return " · ".join(out) if out else "(키 동일·래퍼차)"


def turn_of(m, i):
    t = m.get("turn_idx", i)
    try:
        return int(t)
    except Exception:
        return i


def calls_by_turn(sim):
    """turn -> [(tool_call, 응답본문)]"""
    out = collections.defaultdict(list)
    msgs = sim.get("messages") or []
    resp = {}
    for m in msgs:
        if m.get("role") == "tool" and m.get("id"):
            resp[m["id"]] = " ".join(str(m.get("content") or "").split())
    for i, m in enumerate(msgs):
        for tc in (m.get("tool_calls") or []):
            out[turn_of(m, i)].append((tc, resp.get(tc.get("id"), "")))
    return out


def gold_rows(sim):
    """gold 액션 — **action_id 단위**로 접는다.

    ⚠채점표는 같은 액션을 `tool_type` 별로 여러 줄(generic+write) 싣는다. 접지 않으면 write 가 많은
      태스크가 자동으로 무거워져 계수가 부풀고, 태스크 간 비교가 깨진다(2026-08-19 실측: 276→…).
    """
    out = []
    for ck in ((sim.get("reward_info") or {}).get("action_checks") or []):
        a = ck.get("action") or {}
        ar = a.get("arguments") or {}
        nm = (ar.get("agent_tool_name") or ar.get("user_tool_name")
              or ar.get("discoverable_tool_name") or a.get("name") or "?")
        inner = ar.get("arguments", None)
        out.append({"name": str(nm), "type": ck.get("tool_type"), "match": bool(ck.get("action_match")),
                    "args": ar if inner is None else inner,
                    "aid": str(a.get("action_id") or ""), "req": a.get("requestor")})
    ded, seen = [], set()
    for g in out:
        k = g["aid"] or (g["name"] + json.dumps(g["args"], ensure_ascii=False, sort_keys=True))
        if k in seen:
            continue
        seen.add(k)
        ded.append(g)
    return ded


def deny_rows(tag):
    """**sim 태그 전체** -> {turn: [(마크, 종류, 값, reason)]}."""
    out = collections.defaultdict(lambda: collections.defaultdict(list))
    unknown = collections.Counter()
    for d in (F.trace(tag) or []):
        ln = str(d.get("line") or "")
        m = DENY_RE.match(ln)
        if not m or any(p in ln for p in NOT_A_DENY):
            continue
        if not d.get("sim") or not isinstance(d.get("turn"), int):
            continue
        mark = m.group(1)
        kind, keys = MARK_SPEC.get(mark, ("JOIN?", ()))
        if kind == "JOIN?":
            unknown[mark] += 1
        kv = {k: (q or v) for k, q, v in KV_RE.findall(ln)}
        val = ""
        for k in keys:
            if kv.get(k):
                val = kv[k]
                break
        rs = REASON_RE.search(ln)
        out[str(d["sim"])][d["turn"]].append((mark, kind, val, rs.group(1) if rs else "?"))
    if unknown:
        print("⚠MARK_SPEC 미등재 deny 마크(=JOIN 취급): %s" % dict(unknown))
    return out


def main():
    tags = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not tags:
        print("usage: x392_block_join.py <tag> ...")
        return 2

    rows, pushes, ends = [], [], []
    for tag in tags:
        dt = deny_rows(tag)
        for sim in F.scored(tag, SUF):
            task, rw = F.task_id(sim), (sim.get("reward_info") or {}).get("reward")
            stag = F.simtag(sim)
            cbt = calls_by_turn(sim)
            gold = gold_rows(sim)
            gnames = {g["name"] for g in gold}
            hist = collections.defaultdict(list)     # 이름 -> [(turn, 인자, 응답)]
            for t in sorted(cbt):
                for tc, resp in cbt[t]:
                    a = F.argsof(tc)
                    hist[str(F.inner_name(a) or F.nameof(tc))].append((t, a, resp))

            def ran_after(nm, t):
                return any(tt > t for tt, _a, r in hist.get(nm, [])
                           if r and not any(p in r for p in ENV_PAT))

            blocked_turns = collections.defaultdict(set)   # 이름 -> {turn}
            for t, marks in sorted(dt.get(stag, {}).items()):
                cand = cbt.get(t) or cbt.get(t - 1) or []
                vj, wrapper = "", ""
                if cand:
                    tc, _r = cand[-1]
                    vj, wrapper = str(F.inner_name(F.argsof(tc)) or ""), F.nameof(tc)
                for mark, kind, val, reason in marks:
                    if kind == "PUSH":
                        pushes.append({"tag": tag, "task": task, "trial": sim.get("trial"),
                                       "turn": t, "name": val, "gold": val in gnames,
                                       "later_ok": ran_after(val, t), "reward": rw})
                        continue
                    if kind == "ARGVAL":
                        rows.append({"tag": tag, "task": task, "trial": sim.get("trial"), "turn": t,
                                     "mark": mark, "kind": kind, "reason": reason, "val": "",
                                     "argval": val, "src": "line", "gold": False,
                                     "later_ok": False, "reward": rw})
                        continue
                    v = val if (kind == "TARGET" and val) else vj
                    src = "line" if (kind == "TARGET" and val) else ("join" if vj else "-")
                    if v:
                        blocked_turns[v].add(t)
                    rows.append({"tag": tag, "task": task, "trial": sim.get("trial"), "turn": t,
                                 "mark": mark, "kind": kind, "reason": reason, "val": v,
                                 "argval": "", "src": src, "wrapper": wrapper,
                                 "gold": v in gnames, "later_ok": ran_after(v, t), "reward": rw})

            for g in gold:
                nm, att = g["name"], hist.get(g["name"], [])
                bl = sorted(blocked_turns.get(nm, set()))
                if g["match"]:
                    code, why = "MATCH", ""
                elif not att and not bl:
                    code, why = "NEVER", ""
                else:
                    envrej = [r for _t, _a, r in att if any(p in r for p in ENV_PAT)]
                    okrun = [(t, a) for t, a, r in att if r and not any(p in r for p in ENV_PAT)]
                    if bl and not okrun:
                        code, why = "OURS", "turn=%s" % ",".join(str(x) for x in bl[:4])
                    elif envrej and not okrun:
                        code, why = "ENV_REJECT", envrej[0][:46]
                    elif okrun:
                        sem = any(F.args_equal(a, g["args"]) or F.args_equal(
                            (a or {}).get("arguments", a), g["args"]) for _t, a in okrun)
                        code = "NOTATION" if sem else "ARGDIFF"
                        why = "" if sem else diff_keys(okrun[-1][1], g["args"])
                    else:
                        code, why = "NEVER", "(호출 흔적만·응답 0)"
                ends.append({"tag": tag, "task": task, "trial": sim.get("trial"), "reward": rw,
                             "name": nm, "type": g["type"], "code": code, "why": why,
                             "blocked": len(bl), "attempts": len(att)})

    print("=" * 116)
    print("x392 · deny 표적 복원 + gold 결말 · 태그 %s · sim %d"
          % (", ".join(tags), len({(e["task"], e["trial"], e["tag"]) for e in ends})))
    print("=" * 116)

    real = [r for r in rows if r["kind"] != "ARGVAL"]
    print("\n## §A 우리 층이 **호출을 막은** 건 %d (인자값 deny %d 는 따로)"
          % (len(real), len(rows) - len(real)))
    print("%-22s %-7s %6s %6s %6s %s" % ("mark", "종류", "gold✓", "gold✗", "값없음", "이후실행(gold건)"))
    for mark in sorted({r["mark"] for r in real}):
        g = [r for r in real if r["mark"] == mark]
        gg = [r for r in g if r["gold"]]
        print("%-22s %-7s %6d %6d %6d %s"
              % (mark, g[0]["kind"], len(gg), len([r for r in g if r["val"] and not r["gold"]]),
                 len([r for r in g if not r["val"]]),
                 "%d/%d" % (len([r for r in gg if r["later_ok"]]), len(gg)) if gg else "-"))
    gold_deny = [r for r in real if r["gold"]]
    print("  ⇒ **gold 액션 이름을 막은 것 %d건** · 그중 이후 실행 회복 %d · 미회복 %d"
          % (len(gold_deny), len([r for r in gold_deny if r["later_ok"]]),
             len([r for r in gold_deny if not r["later_ok"]])))
    print("  값 출처: 줄 직접 %d · 턴 조인 %d · 복원 실패 %d"
          % (len([r for r in real if r["src"] == "line"]),
             len([r for r in real if r["src"] == "join"]),
             len([r for r in real if not r["val"]])))

    print("\n## §A-2 gold 를 막고 **끝내 실행 안 된** 건 (여기만 실패의 후보다)")
    print("%-9s %-3s %-5s %-40s %-20s %s" % ("task", "tr", "turn", "표적", "mark/reason", "reward"))
    for r in sorted([x for x in gold_deny if not x["later_ok"]],
                    key=lambda x: (x["task"], str(x["trial"]), x["turn"])):
        print("%-9s %-3s %-5d %-40s %-20s %s"
              % (r["task"], r["trial"], r["turn"], r["val"][:40],
                 (r["mark"].replace("T2_", "") + "/" + r["reason"])[:20], r["reward"]))

    print("\n## §A-3 우리가 **짚어 준** 이름(T2_DISCOVERY_STEP2 푸시 %d건 · 차단 아님)" % len(pushes))
    pg = [p for p in pushes if p["gold"]]
    print("  gold 이름 지목 %d · gold 아닌 이름 지목 %d · 지목 후 실행됨 %d/%d"
          % (len(pg), len(pushes) - len(pg), len([p for p in pg if p["later_ok"]]), len(pg)))
    c = collections.Counter((p["name"], p["gold"]) for p in pushes)
    for (n, g), k in c.most_common(10):
        print("  %-46s gold=%-5s %d" % (n[:46], g, k))

    print("\n## §B gold 액션 결말 (action_id 단위)")
    cb = collections.Counter(e["code"] for e in ends)
    ov = collections.Counter(e["code"] for e in ends if e["blocked"])
    for code in ("MATCH", "NEVER", "OURS", "ENV_REJECT", "ARGDIFF", "NOTATION"):
        if cb[code]:
            print("  %-11s %-4d (그중 같은 이름에 우리 deny 가 있던 것 %d)" % (code, cb[code], ov[code]))

    print("\n## §C 태스크×시행 요약")
    print("%-9s %-3s %-5s %-4s %s" % ("task", "tr", "rw", "gold", "결말 분포 · 우리deny(gold표적) · 푸시"))
    keys = sorted({(e["task"], str(e["trial"])) for e in ends})
    for tk, tr in keys:
        es = [e for e in ends if e["task"] == tk and str(e["trial"]) == tr]
        dd = collections.Counter(e["code"] for e in es)
        dn = [r for r in real if r["task"] == tk and str(r["trial"]) == tr]
        pu = [p for p in pushes if p["task"] == tk and str(p["trial"]) == tr]
        print("%-9s %-3s %-5s %-4d %s | deny %d(gold %d·미회복 %d) | 푸시 %d(gold %d)"
              % (tk, tr, es[0]["reward"], len(es),
                 " ".join("%s=%d" % (k, v) for k, v in sorted(dd.items())),
                 len(dn), len([r for r in dn if r["gold"]]),
                 len([r for r in dn if r["gold"] and not r["later_ok"]]),
                 len(pu), len([p for p in pu if p["gold"]])))

    print("\n## §D 실패 sim 의 미매치 gold 전량 (per gold action)")
    print("%-9s %-3s %-42s %-6s %-11s %-2s %s"
          % ("task", "tr", "gold 액션", "유형", "결말", "막", "근거(키 단위 대조)"))
    for e in sorted(ends, key=lambda x: (x["task"], str(x["trial"]), x["code"])):
        if e["code"] == "MATCH" or (e["reward"] or 0) >= 1.0:
            continue
        print("%-9s %-3s %-42s %-6s %-11s %-2d %s"
              % (e["task"], e["trial"], e["name"][:42], str(e["type"])[:6], e["code"],
                 e["blocked"], e["why"][:64]))

    out = os.path.normpath(os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026",
                                        "x392_block_join.json"))
    io.open(out, "w", encoding="utf-8").write(json.dumps(
        {"deny": rows, "push": pushes, "ends": ends}, ensure_ascii=False, indent=1))
    print("\n원자료: %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
