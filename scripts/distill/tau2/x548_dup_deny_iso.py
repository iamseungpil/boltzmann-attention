# -*- coding: utf-8 -*-
r"""x548 - 격리: **이유와 조치를 실은 거절**이 중복 실행을 멈추는가, 정당한 재제출은 살리는가.

## 왜 (2026-08-26 · x547 다음)

x547 이 재생으로 확정했다: 중복 실행을 **전부** 빼도 만점 sim 14/14 는 그대로고(비용 0),
0점 sim 142 중 **8** 이 산다. 정제 술어(*순수 반복만 차단*)는 8 중 1 만 살려 더 나쁘다.
남은 위험은 **051 하나** — gold 가 *거절·상환 뒤의 같은 인자 재제출*을 요구한다. 그런데
코퍼스의 051 sim 은 전부 0점이라 그 비용은 **관측된 적이 없다**(반증된 게 아니다).

사용자 지적: *"차단할 때 차단 이유와 조치할 수 있는 방법을 알려주어서 같은 실수 반복하지
않게 하는거 아니었나?"* — 그렇다([[64]]). 그러면 051 의 탈출구는 술어가 아니라 **문면**이다.

재생은 *실행을 뺐을 때의 점수*(G1/G2)만 답한다. **거절당한 모델의 다음 수**(G3)가 여기다.

## ⛔1차 판은 무효였다 (2026-08-26 · 기록으로 남긴다)

1차는 평문 대화 한 통에 *"호출할 게 없으면 정확히 NONE 이라고 답하라"* 를 붙였다. 결과는
네 팔 전부 NONE 이었고 **A_live 가 라이브 재실행을 재현하지 못했다**([[62]] 2b 불공정).
창은 무죄였다 — 12 메시지에 손님 질문도 앞선 크레딧 성공 4건도 다 들어 있었다. 진범은
**answer_format** 이다([[78]] 2026-08-24 와 같은 자리): 라이브는 도구 호출 API 로 돌았는데
프로브가 *아무것도 안 하는 선택지*를 손수 만들어 줬다. 채점도 틀렸다 — id 등장만 세서
`approve_credit_limit_increase` 안의 카드 id 를 *재제출* 로 셌다.

2차(이 파일)는 셋을 고친다:
  · 도구 표면을 **env 선언에서** 실어 준다(`env.get_tools()[].openai_schema` · 저작 0)
  · 형식 지시를 **없앤다** — 팔 사이 차이만 남는다
  · 채점을 **가른다**: ⒜같은 호출 재발행(`mut_key` 동일) ⒝아무 행동도 안 함(과잉 억제·[[70]])

## 팔 ([[57]])

    A_live   창 그대로, 모델이 다음 턴을 만든다   <- 라이브 재실행을 **재현해야** 한다
    B_bare   창 + 그 시도 + **이름 없는 거절**
    C_named  창 + 그 시도 + **이유·앞선 결과·조치**  <- 수리 후보
    N_len    창 + 그 시도 + 같은 길이 무관 문장

시도(assistant tool_calls)는 **모델 자신이 라이브에서 낸 그 호출을 축자로** 쓴다.

## 판정은 두 표적을 **함께** 읽는다

    074  C_named 의 재발행이 B_bare·N_len 보다 낮아야 한다 (멈춰야 한다)
    051  C_named 가 행동을 **얼리면 안 된다** — 재발행이 A_live 만큼 나와야 한다

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

# ★C_proceed — **사용자 제안 2026-08-26**: *"호출을 지우고, 지운 이유를 달아서 왜 지웠는지
#   알려주고, 그냥 다음 단계를 진행하라고 하면 안되나?"*  발명이 아니다 — 이미 살아 있는
#   읽기 가드(`[DUPLICATE-READ]`)의 마지막 문장이 정확히 그 모양이다:
#     *"If you already have the information you need, proceed to the next step instead of
#       searching again."*
#   쓰기 쪽에만 그 문장이 없었다. 그래서 이 팔은 **그 문면을 쓰기로 옮긴 것**이다.
#   문면은 **셋을 명시로** 담는다(사용자 2026-08-26 축자: *"조용히 지우는게 아니라, 중복은
#   지우고, 지운 사실을 명시적으로 알려주고, 이것에 대해서 더 하지 말고, 그 다음 단계로
#   진행하라고 하는거다"*):
#     ⑴이 호출은 **지워졌다**(실행되지 않았다)  ⑵이 건은 **더 하지 마라**  ⑶**다음 단계로 가라**
#   ⑵가 읽기 가드의 *"Do NOT repeat this exact search"* 에 해당한다 — 초안에 그것만 빠져 있었다.
DENY_PROCEED = (
    "Error: [DUPLICATE-WRITE] This exact call (same tool, same arguments) already succeeded "
    "earlier in this conversation, so this call was REMOVED and not run - running it twice "
    "would apply the same change twice. It ran at message {at} and returned:" + NL + NL +
    "{result}" + NL + NL +
    "That change is already done. Do NOT attempt this change again and do not do anything "
    "further about it. Use the result above and proceed to the next step.")
# ★D_escape — 위에 **단서 한 문장**을 더한 판. 051 처럼 gold 가 재제출을 요구하는 자리를
#   여는가, 아니면 그냥 모델을 헷갈리게 하는가를 가르려고 나란히 둔다([[57]] 한 칸 차이).
DENY_ESCAPE = (DENY_PROCEED + " If something HAS changed since that message that makes this "
               "call correct a second time, state what changed and send the call again; "
               "it will go through.")
DENY_BARE = "Error: This call could not be completed. Please continue."


def chat(port, messages, tools, maxtok=420):
    payload = {"model": MODEL, "temperature": 0.0, "max_tokens": maxtok,
               "messages": messages}
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
    req = urllib.request.Request("http://127.0.0.1:%d/v1/chat/completions" % port,
                                 data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            return json.loads(r.read().decode("utf-8"))["choices"][0]["message"]
    except urllib.error.HTTPError as e:
        # ★400 을 '요청이 나쁘다'로 뭉뚱그리면 그 다음을 못 고친다 — 본문을 그대로 올린다([[25]]).
        raise RuntimeError("HTTP %s: %s" % (e.code, e.read().decode("utf-8", "replace")[:400]))


def load(tag):
    p = os.path.join(SIMS, tag + ".results.json.gz")
    if not os.path.exists(p):
        return []
    d = json.load(gzip.open(p, "rt", encoding="utf-8", errors="replace"))
    return d.get("simulations") or []


def result_of(ms, i, tcid):
    for j in range(i + 1, len(ms)):
        m = ms[j]
        if str(m.get("role")) == "tool" and str(m.get("id") or "") == tcid:
            return str(m.get("content") or "")
    return ""


def safe_start(ms, i, w=None):
    """도구 호출/결과 짝이 잘리지 않는 시작점 — `i-w` 이하의 마지막 user 턴.

    `w=0` 이면 **전량 프리픽스**(라이브가 실제로 본 것). 창이 좁으면 *왜 그 행동을 했는지*가
    빠져 A_live 가 라이브를 재현하지 못한다(2026-08-26 실측: 12 메시지 창에서 074 행동 0/4)."""
    w = W if w is None else w
    if w <= 0:
        return 0
    for k in range(max(0, i - w), -1, -1):
        if str(ms[k].get("role")) == "user":
            return k
    return 0


def to_openai(ms, cap=1200):
    """창을 OpenAI 형식으로. **메시지별 내용은 `cap` 자로 자른다** — 라이브 창 그대로면
    도구 덤프 + 정책 7,541자 + 도구 17 스키마가 문맥을 넘겨 서버가 400 을 낸다(실측).
    x542 도 같은 처리를 한다(1,600자). 자른 사실은 꼬리표로 남긴다([[25]])."""
    def cut(s):
        s = str(s or "")
        return s if len(s) <= cap else (s[:cap] + " …[truncated]")

    out = []
    for m in ms:
        role = str(m.get("role"))
        if role == "assistant":
            msg = {"role": "assistant", "content": cut(m.get("content"))}
            tcs = []
            for tc in (m.get("tool_calls") or []):
                tcs.append({"id": str(tc.get("id") or ""), "type": "function",
                            "function": {"name": str(F.nameof(tc)),
                                         "arguments": json.dumps(F.argsof(tc),
                                                                 ensure_ascii=False)}})
            if tcs:
                msg["tool_calls"] = tcs
            out.append(msg)
        elif role == "tool":
            out.append({"role": "tool", "tool_call_id": str(m.get("id") or ""),
                        "content": cut(m.get("content"))})
        elif role == "user":
            out.append({"role": "user", "content": cut(m.get("content"))})
    return out


def keyset(tcs):
    """이 호출들의 `mut_key` 집합 — 가드가 쓰는 바로 그 동일성."""
    out = set()
    for tc in tcs:
        out.add(F.mut_key(str(F.nameof(tc)), F.argsof(tc)))
    return out


def cases_074(tag="bank_t7358_d074_20260826"):
    out = []
    for s in load(tag):
        ms = s.get("messages") or []
        done = {}
        for i, m in enumerate(ms):
            if str(m.get("role")) != "assistant":
                continue
            rep = []
            for tc in (m.get("tool_calls") or []):
                a = F.argsof(tc)
                inner = a.get("arguments")
                if isinstance(inner, str):
                    try:
                        inner = json.loads(inner)
                    except Exception:
                        inner = {}
                tool = str(a.get("agent_tool_name") or "")
                if "apply_checking_account_credit" not in tool:
                    continue
                k = F.mut_key(str(F.nameof(tc)), a)
                tcid = str(tc.get("id") or "")
                res = result_of(ms, i, tcid)
                if k in done:
                    rep.append(tc)
                elif res and not res.lstrip().startswith("Error:"):
                    done[k] = (i, res)
            if len(rep) >= 2:
                at, res = list(done.values())[0]
                out.append({"target": "074", "task": s.get("task_id"), "tag": tag,
                            "sim": "%s#s%s" % (s.get("task_id"), s.get("seed")),
                            "msg": i, "attempt": rep, "prior_at": at,
                            "prior_result": res[:700], "ms": ms})
                break
    return out


def cases_051(tags):
    out = []
    for tag in tags:
        for s in load(tag):
            if str(s.get("task_id")) != "task_051":
                continue
            ms = s.get("messages") or []
            done, denied = {}, False
            for i, m in enumerate(ms):
                if str(m.get("role")) != "assistant":
                    continue
                for tc in (m.get("tool_calls") or []):
                    a = F.argsof(tc)
                    tool = str(a.get("agent_tool_name") or "")
                    tcid = str(tc.get("id") or "")
                    res = result_of(ms, i, tcid)
                    ok = bool(res) and not res.lstrip().startswith("Error:")
                    if "deny_credit_limit_increase" in tool and ok:
                        denied = True
                    if "submit_credit_limit_increase_request" not in tool:
                        continue
                    k = F.mut_key(str(F.nameof(tc)), a)
                    if k in done and denied:
                        at, r = done[k]
                        out.append({"target": "051", "task": "task_051", "tag": tag,
                                    "sim": "%s#s%s" % (s.get("task_id"), s.get("seed")),
                                    "msg": i, "attempt": [tc], "prior_at": at,
                                    "prior_result": r[:700], "ms": ms})
                        break
                    if k not in done and ok:
                        done[k] = (i, res)
                if out and out[-1]["tag"] == tag and out[-1]["sim"].startswith("task_051"):
                    break
            if len(out) >= 3:
                return out
    return out


def build_arms(c, system, filler_txt, w=None, cap=1200):
    base = to_openai(c["ms"][safe_start(c["ms"], c["msg"], w):c["msg"]], cap)
    head = ([{"role": "system", "content": system}] if system else []) + base
    att = {"role": "assistant", "content": "",
           "tool_calls": [{"id": str(tc.get("id") or ("x%d" % n)), "type": "function",
                           "function": {"name": str(F.nameof(tc)),
                                        "arguments": json.dumps(F.argsof(tc),
                                                                ensure_ascii=False)}}
                          for n, tc in enumerate(c["attempt"])]}
    proceed = DENY_PROCEED.format(at=c["prior_at"], result=c["prior_result"])
    escape = DENY_ESCAPE.format(at=c["prior_at"], result=c["prior_result"])

    def denied(text):
        return head + [att] + [{"role": "tool", "tool_call_id": t["id"], "content": text}
                               for t in att["tool_calls"]]

    return collections.OrderedDict((
        ("A_live", head),
        ("B_bare", denied(DENY_BARE)),
        ("C_proceed", denied(proceed)),
        ("D_escape", denied(escape)),
        ("N_len", denied("Error: " + (filler_txt or "")[:len(proceed)])),
    ))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8141)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--win", type=int, default=0, help="0 = 전량 프리픽스")
    ap.add_argument("--cap", type=int, default=500, help="메시지당 내용 절단")
    ap.add_argument("--tags051", default="bank_n97_gpu1_main_20260806,"
                                         "bank_n97_gpu1_main_20260805,"
                                         "bank_all97_nt1_v2_20260718")
    a = ap.parse_args(argv)

    # ⛔정본 tau2 확인 — 이 머신엔 클론이 둘이고 `go_stack.sh:20 GO_TAU2` 만 우리 것이다.
    #   잘못 잡히면 변종 목록도 서명도 달라 **조용히 다른 판본이 답한다**(오늘 두 번 걸렸다).
    import tau2
    if "scratch/tau2-bench" not in str(getattr(tau2, "__file__", "")).replace("\\", "/"):
        print("⛔잘못된 tau2 를 불렀다: %s" % (getattr(tau2, "__file__", "?"),))
        print("   이렇게 부른다: cd /home/woori/scratch/tau2-bench && "
              "PYTHONPATH=src:$R/scripts/distill/tau2 python $R/.../x548_dup_deny_iso.py")
        return 3
    from tau2.registry import registry
    from tau2.domains.banking_knowledge.environment import get_tasks
    tasks = {t.id: t for t in get_tasks()}

    cases = cases_074() + cases_051([t.strip() for t in a.tags051.split(",") if t.strip()])
    print("=" * 100)
    print("x548 v2 격리 — 표적 %d (074 %d · 051 %d)"
          % (len(cases), sum(1 for c in cases if c["target"] == "074"),
             sum(1 for c in cases if c["target"] == "051")), flush=True)
    if not cases:
        print("창 0 — 판정 불가")
        return 2

    rows = []
    for c in cases:
        task = tasks.get(str(c["task"]))
        env = registry.get_env_constructor("banking_knowledge")(
            retrieval_variant="alltools", task=task)
        tools = [t.openai_schema for t in env.get_tools()]
        system = env.get_policy()
        fil = None
        for m in c["ms"]:
            if str(m.get("role")) == "tool":
                for s in re.split(r"(?<=\.)\s+", str(m.get("content") or "")):
                    s = s.strip()
                    if 200 < len(s) < 1400:
                        fil = s
                        break
            if fil:
                break
        blocked = keyset(c["attempt"])
        print("\n── %s %s msg=%d · 시도 %d 건 · 도구 %d · 정책 %d자 (앞선 성공 msg=%s)"
              % (c["target"], c["sim"], c["msg"], len(c["attempt"]), len(tools),
                 len(system or ""), c["prior_at"]), flush=True)
        tally = {}
        for arm, msgs in build_arms(c, system, fil, a.win, a.cap).items():
            re_n = act_n = 0
            sample = ""
            for _k in range(a.n):
                try:
                    msg = chat(a.port, msgs, tools)
                except Exception as e:
                    sample = sample or ("ERR %r" % (e,))
                    continue
                tcs = msg.get("tool_calls") or []
                names = []
                for t in tcs:
                    fn = (t.get("function") or {})
                    try:
                        args = json.loads(fn.get("arguments") or "{}")
                    except Exception:
                        args = {}
                    names.append(str(fn.get("name")))
                    if F.mut_key(str(fn.get("name")), args) in blocked:
                        re_n += 1
                        break
                if tcs:
                    act_n += 1
                if not sample:
                    sample = (",".join(names) if names
                              else " ".join(str(msg.get("content") or "").split())[:90])
            tally[arm] = {"reissue": re_n, "acted": act_n}
            print("   %-8s 재발행 %d/%d · 행동 %d/%d | %s"
                  % (arm, re_n, a.n, act_n, a.n, sample[:80]), flush=True)
        rows.append({"target": c["target"], "sim": c["sim"], "tag": c["tag"],
                     "msg": c["msg"], "n": a.n, "tally": tally})

    print("\n" + "=" * 100)
    for t in ("074", "051"):
        rs = [r for r in rows if r["target"] == t]
        if not rs:
            print("%s: 창 0 — 판정 불가" % t)
            continue
        tot = sum(r["n"] for r in rs)
        agg = {k: [sum(r["tally"][k]["reissue"] for r in rs),
                   sum(r["tally"][k]["acted"] for r in rs)]
               for k in ("A_live", "B_bare", "C_proceed", "D_escape", "N_len")}
        print("%s (시행 %d) 재발행/행동: %s"
              % (t, tot, {k: "%d/%d · %d" % (v[0], tot, v[1]) for k, v in agg.items()}))
    print("판정: 074 는 C_named 의 **재발행**이 B_bare·N_len 보다 낮아야 하고,")
    print("      051 은 C_named 의 **행동**이 A_live 만큼 유지돼야 한다(얼면 [[70]] 매도).")
    print("⚠A_live(074)가 재발행을 재현 못 하면 그 창은 불공정하다([[62]] 2b).")

    p = os.path.join(REP, "x548_dup_deny_iso_2026_08_26.json")
    with open(p, "w", encoding="utf-8") as fh:
        json.dump({"rows": rows, "deny_proceed": DENY_PROCEED, "deny_escape": DENY_ESCAPE},
                  fh, ensure_ascii=False, indent=2)
    print("산출: %s" % p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
