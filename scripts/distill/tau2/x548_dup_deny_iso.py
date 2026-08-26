# -*- coding: utf-8 -*-
r"""x548 - 격리: **이유와 조치를 실은 거절**이 중복 실행을 멈추는가, 정당한 재제출은 살리는가.

## 왜 (2026-08-26 · x547 다음)

x547 이 재생으로 확정했다:

    비용 갈래 14 sim | 중복 실행을 **전부** 빼도 깎인 점수 **0**
    이득 갈래 142 sim | 전부 빼면 **8** 이 살고, *순수 반복만* 빼는 정제 술어는 **1** 만 산다

⇒ 술어를 좁힐 이유가 없다. 남은 위험은 **051 하나**다: gold 가 같은 인자의 재제출을 진짜로
   요구한다([2] 제출 -> [12] **거절** -> [16] 상환 -> [17] **같은 인자 재제출** -> [19] 승인).
   그런데 코퍼스의 051 sim 은 전부 0점이라 그 비용은 **관측된 적이 없다**(반증된 게 아니다).

사용자 지적(2026-08-26): *"차단할 때 차단 이유와 조치할 수 있는 방법을 알려주어서 같은 실수
반복하지 않게 하는거 아니었나?"* — 그렇다([[64]]). 그러면 051 의 탈출구는 술어를 좁히는 것이
아니라 **문면**이다. 이 프로브가 그 문면을 시험한다.

## 재생이 못 답하는 자리 (그래서 격리다)

재생은 *실행을 뺐을 때의 점수*(G1/G2)만 답한다. **거절당한 모델이 다음에 무엇을 하는가**(G3)는
답할 수 없다(`x515` §경계). 여기가 그 G3 이다.

## 표적 둘 — 같은 문면이 **반대 방향**으로 작동해야 한다

    074  이미 성공한 크레딧 4건을 msg97 에서 통째로 재실행했다 -> 거절이 **멈춰야** 한다
    051  거절·상환 뒤의 재제출은 gold 가 요구한다             -> 거절이 **막아서는 안 된다**

## 팔 ([[57]] 부정통제 포함)

    A_live   창 그대로                       <- 라이브의 재실행을 **재현해야** 한다([[62]] 2b)
    B_bare   창 + 시도 + **이름 없는 거절**    <- [[64]] 의 반대 극 (*"할 수 없습니다"* 뿐)
    C_named  창 + 시도 + **이유·앞선 결과·조치**  <- 수리 후보
    N_len    창 + 시도 + 같은 길이 무관 문장    <- 길이가 아니라 내용임을 가른다

## 채점 - 닫힌 술어 · gold 미접촉 ([[23]])

*이미 성공한* 대상(074=계좌 id · 051=카드 id)이 답에 다시 나오는지만 센다. 어느 쪽이 옳은지는
표적마다 반대이므로 **판정은 두 표적을 함께** 읽는다. 정답표를 프롬프트에 넣지 않는다.

사용: (리모트·cwd=scripts/distill/tau2) py -3 x548_dup_deny_iso.py --port 8141 --n 4
"""
import argparse
import collections
import gzip
import json
import os
import re
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import t2_forensic as F   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SIMS = os.path.abspath(os.path.join(HERE, "..", "..", "..",
                                    "reports", "facet_rft_2026", "sim_results"))
REP = os.path.abspath(os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026"))
MODEL = "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8"
NL = chr(10)
W = 14

# ★거절 문면. 도메인 낱말 0 — 대상 이름과 앞선 결과는 **궤적에서 축자로** 실린다([[23]]).
#   [[64]] 두 가지를 다 담는다: 무엇이 틀렸나 + 무엇을 하면 풀리나(+ 풀리는 조건).
DENY_NAMED = (
    "Error: [DUPLICATE-WRITE] This exact call (same tool, same arguments) already succeeded "
    "earlier in this conversation. It ran at message {at} and returned:" + NL + NL +
    "{result}" + NL + NL +
    "It has NOT been run again, because running it twice would apply the same change twice. "
    "If nothing has changed since that message, do not send this call again - use the result "
    "above to answer. If something HAS changed since then that makes this call correct a "
    "second time, state what changed and send the call again; it will go through.")
DENY_BARE = "Error: This call could not be completed. Please continue."


def gen(port, body, maxtok=200):
    payload = {"model": MODEL, "temperature": 0.0, "max_tokens": maxtok,
               "messages": [{"role": "user", "content": body}]}
    req = urllib.request.Request("http://127.0.0.1:%d/v1/chat/completions" % port,
                                 data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read().decode("utf-8"))["choices"][0]["message"]["content"]


def load(tag):
    p = os.path.join(SIMS, tag + ".results.json.gz")
    if not os.path.exists(p):
        return []
    d = json.load(gzip.open(p, "rt", encoding="utf-8", errors="replace"))
    return d.get("simulations") or []


def inner_args(tc):
    a = F.argsof(tc) or {}
    inner = a.get("arguments")
    if isinstance(inner, str):
        try:
            inner = json.loads(inner)
        except Exception:
            inner = {}
    return a, (inner if isinstance(inner, dict) else {})


def result_of(ms, i, tcid):
    for j in range(i + 1, len(ms)):
        m = ms[j]
        if str(m.get("role")) == "tool" and str(m.get("id") or "") == tcid:
            return str(m.get("content") or "")
    return ""


def window(ms, upto):
    txt = []
    for mm in ms[max(0, upto - W):upto]:
        cc = str(mm.get("content") or "").strip()
        if cc:
            txt.append("[%s] %s" % (mm.get("role"), cc[:1500]))
    return (NL + NL).join(txt)


def filler(ms, want):
    best = None
    for m in ms:
        if str(m.get("role")) != "tool":
            continue
        for s in re.split(r"(?<=\.)\s+", str(m.get("content") or "")):
            s = s.strip()
            if not (80 < len(s) < 1200):
                continue
            d = abs(len(s) - want)
            if best is None or d < best[0]:
                best = (d, s)
    return best[1] if best else None


def find_repeat(sim, toolpat, idkey, needs_between=None):
    """이 sim 에서 **이미 성공한 호출을 다시 낸 자리**를 찾는다.

    반환 = (재시도 msg 인덱스, 대상 id 집합, 앞선 성공 (msg, 결과)). 닫힌 규칙뿐."""
    ms = sim.get("messages") or []
    first, between_ok = {}, (needs_between is None)
    for i, m in enumerate(ms):
        if str(m.get("role")) != "assistant":
            continue
        for tc in (m.get("tool_calls") or []):
            a, inner = inner_args(tc)
            tool = str(a.get("agent_tool_name") or a.get("user_tool_name")
                       or a.get("discoverable_tool_name") or F.nameof(tc) or "")
            tcid = str((tc.get("id") if isinstance(tc, dict) else "") or "")
            res = result_of(ms, i, tcid)
            ok = bool(res) and not res.lstrip().startswith("Error:")
            if needs_between and needs_between in tool and ok:
                between_ok = True
            if toolpat not in tool:
                continue
            tid = str(inner.get(idkey) or a.get(idkey) or "")
            if not tid:
                continue
            if tid in first:
                if between_ok:
                    return i, {tid}, first[tid]
            elif ok:
                first[tid] = (i, res)
    return None, set(), None


def target_074(tag="bank_t7358_d074_20260826"):
    """074: msg97 이 이미 성공한 네 계좌를 통째로 재실행한 자리."""
    out = []
    for s in load(tag):
        ms = s.get("messages") or []
        done = {}
        for i, m in enumerate(ms):
            if str(m.get("role")) != "assistant":
                continue
            ids = []
            for tc in (m.get("tool_calls") or []):
                a, inner = inner_args(tc)
                tool = str(a.get("agent_tool_name") or "")
                if "apply_checking_account_credit" not in tool:
                    continue
                acct = str(inner.get("account_id") or "")
                tcid = str((tc.get("id") if isinstance(tc, dict) else "") or "")
                res = result_of(ms, i, tcid)
                if not acct:
                    continue
                if acct in done:
                    ids.append(acct)
                elif res and not res.lstrip().startswith("Error:"):
                    done[acct] = (i, res)
            if ids and len(ids) >= 2:
                at, res = done[ids[0]]
                out.append({"target": "074", "sim": "%s#s%s" % (s.get("task_id"), s.get("seed")),
                            "tag": tag, "msg": i, "ids": sorted(set(ids)),
                            "prior_at": at, "prior_result": res[:700],
                            "win": window(ms, i), "filler": filler(ms, len(DENY_NAMED))})
                break
    return out


def target_051(tags):
    """051: **거절이 있고 난 뒤** 같은 카드로 재제출한 자리(gold 가 요구하는 모양)."""
    out = []
    for tag in tags:
        for s in load(tag):
            if str(s.get("task_id")) != "task_051":
                continue
            ms = s.get("messages") or []
            i, ids, prior = find_repeat(s, "submit_credit_limit_increase_request",
                                        "credit_card_account_id",
                                        needs_between="deny_credit_limit_increase")
            if i is None:
                continue
            at, res = prior
            out.append({"target": "051", "sim": "%s#s%s" % (s.get("task_id"), s.get("seed")),
                        "tag": tag, "msg": i, "ids": sorted(ids),
                        "prior_at": at, "prior_result": res[:700],
                        "win": window(ms, i), "filler": filler(ms, len(DENY_NAMED))})
            if len(out) >= 3:
                return out
    return out


ASK = (NL + NL + "[system] Decide your next step. Reply with ONLY the tool calls you will make "
       "now, one per line, in the form name(arg=value, ...). If you will not call any tool, "
       "reply with exactly NONE.")


def arms(case):
    """네 팔의 **본문**. 창은 동일하고 마지막 블록만 다르다."""
    attempt = ("[assistant] (calling %s again for %s)"
               % ("the same tool", ", ".join(case["ids"])))
    named = DENY_NAMED.format(at=case["prior_at"], result=case["prior_result"])
    fil = case.get("filler") or ("This account was reviewed earlier in the conversation. " * 6)
    return collections.OrderedDict((
        ("A_live", case["win"]),
        ("B_bare", case["win"] + NL + NL + attempt + NL + "[tool] " + DENY_BARE),
        ("C_named", case["win"] + NL + NL + attempt + NL + "[tool] " + named),
        ("N_len", case["win"] + NL + NL + attempt + NL + "[tool] Error: " + fil[:len(named)]),
    ))


def reissued(ans, ids):
    """답에 **이미 성공한 대상 id** 가 몇 개 다시 나오나. 형식 준수를 요구하지 않는다."""
    low = str(ans or "")
    return sum(1 for t in ids if t and t in low)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8141)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--tags051", default="bank_n97_gpu1_main_20260806,"
                                         "bank_n97_gpu1_main_20260805,"
                                         "bank_all97_nt1_v2_20260718")
    a = ap.parse_args(argv)

    cases = target_074() + target_051([t.strip() for t in a.tags051.split(",") if t.strip()])
    print("=" * 100)
    print("x548 격리 — 표적 %d (074 %d · 051 %d)"
          % (len(cases), sum(1 for c in cases if c["target"] == "074"),
             sum(1 for c in cases if c["target"] == "051")))
    print("=" * 100, flush=True)
    if not cases:
        print("창을 못 만들었다 — 판정 불가([[78]]: 무엇이 창에 없었는지부터 적어라)")
        return 2

    rows = []
    for c in cases:
        print("\n── %s %s msg=%d 대상 %s (앞선 성공 msg=%s)"
              % (c["target"], c["sim"], c["msg"], ",".join(c["ids"]), c["prior_at"]), flush=True)
        body = arms(c)
        tally = collections.OrderedDict()
        for arm, txt in body.items():
            hits, answers = 0, []
            for _k in range(a.n):
                try:
                    ans = gen(a.port, txt + ASK)
                except Exception as e:
                    ans = "ERR %r" % (e,)
                answers.append(" ".join(str(ans).split())[:120])
                hits += 1 if reissued(ans, c["ids"]) else 0
            tally[arm] = hits
            print("   %-8s 재제출 %d/%d   | %s" % (arm, hits, a.n, answers[0][:88]), flush=True)
        rows.append({"target": c["target"], "sim": c["sim"], "tag": c["tag"],
                     "ids": c["ids"], "tally": tally, "n": a.n})

    print("\n" + "=" * 100)
    for t in ("074", "051"):
        rs = [r for r in rows if r["target"] == t]
        if not rs:
            print("%s: 창 0 — 판정 불가" % t)
            continue
        agg = collections.Counter()
        for r in rs:
            for k, v in r["tally"].items():
                agg[k] += v
        tot = sum(r["n"] for r in rs)
        print("%s 합계(재제출/시행): %s  (시행 %d)"
              % (t, {k: "%d/%d" % (agg[k], tot) for k in ("A_live", "B_bare", "C_named", "N_len")},
                 tot))
    print("판정 규칙: 074 는 C_named 가 A_live·N_len 보다 **낮아야** 하고,")
    print("           051 은 C_named 가 A_live 만큼 **높아야** 한다. 둘 다 만족해야 문면이 산다.")
    print("⚠A_live 가 074 에서 재제출을 재현 못 하면 그 창은 불공정하다([[62]] 2b).")

    p = os.path.join(REP, "x548_dup_deny_iso_2026_08_26.json")
    with open(p, "w", encoding="utf-8") as fh:
        json.dump({"rows": rows, "deny_named": DENY_NAMED}, fh, ensure_ascii=False, indent=2)
    print("산출: %s" % p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
