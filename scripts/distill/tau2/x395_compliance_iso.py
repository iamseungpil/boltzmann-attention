# -*- coding: utf-8 -*-
r"""x395 — **격리 프로브: 이행(knowing–doing)**. 정책 문장이 도달했는데 안 부른 자리를 격리에서 다시 묻는다.

## 왜 ([[62]] ①격리로 결손 측정 · [[18]] 정보-맞춘 격리 의무)
t7326 실측: 빠진 gold 도구 중 **그 도구를 지시하는 정책 문장이 궤적에 축자로 도달한 것이 21건**.
정보는 왔는데 안 불렀다 ⇒ 남는 물음은 *부하·전달* 인가 *이행* 인가. 그건 격리로만 갈린다.

    격리에서 되면  → 결손은 부하·전달 → 레버는 **전달(부하 축소)** 로 충분 · 학습 표적 아님
    격리에서도 안 되면 → **이행 결손 확정** → 학습 arm 을 여는 근거

## 팔
    A_min   손님 요청 한 줄 + **그 도구를 지시하는 정책 절차 문장(축자)** + 원장 요약(읽은 것/뜬 것)
    B_full  실제 궤적의 대화 전문(도구 응답은 절단) — 같은 질문
    C_neg   A_min 인데 절차 문장을 **같은 길이의 무내용 문구**로 교체(부정통제 · [[57]])

질문은 하나다: **"다음에 호출할 도구 하나를 JSON 으로 내라."** 형식 부담을 0 에 가깝게 둔다 —
여기서도 못 부르면 그것은 형식·배선이 아니라 **선택·이행**이다.

## 채점 (결정론 · LLM 0)
    hit_exact   낸 도구 이름 == 그 sim 이 빠뜨린 gold 도구
    hit_anygold 낸 이름이 그 태스크의 gold 도구 집합 안
    said_only   도구를 안 내고 산문·설명만 냈다(= 말로 대체 · knowing–doing 의 직접 관측)

사용: py -3 x395_compliance_iso.py [--n 4] [--cases 14] [--arms A_min,B_full,C_neg]
"""
import argparse
import collections
import io
import json
import os
import re
import sys
import threading
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import t2_forensic as F  # noqa: E402

BK = "/home/woori/scratch/tau2-bench/data/tau2/domains/banking_knowledge"
DOCDIR = os.path.join(BK, "documents")
TAGS = ["bank_t7326_halfA_20260819q", "bank_t7326_halfB_20260819q"]
SUF = ".results.json.gz"
PORTS = [8140, 8141]
MODEL = "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8"
STEP_RE = re.compile(r"(?im)^\s*(?:\d+[).]|[-*])\s*(.{15,})$")

SYS = ("You are the tool-calling module of a Rho-Bank support agent. "
       "Reply with ONE JSON object only: {\"tool\": \"<tool name>\", \"arguments\": {…}}. "
       "No prose, no markdown fence. If you believe no tool call is needed, "
       "reply {\"tool\": null, \"reason\": \"…\"}.")

# ★write 표적에는 "다음 한 수"가 불공정하다(그 앞에 선행 read 가 정당하게 올 수 있다).
#   그래서 **남은 호출 계획**도 함께 묻고, 표적이 계획 안에 있는지를 따로 센다.
SYS_PLAN = ("You are the planning module of a Rho-Bank support agent. "
            "Reply with ONE JSON object only: {\"plan\": [{\"tool\": \"<name>\"}, …]} "
            "listing the remaining tool calls in order (max 8). No prose, no markdown fence.")

FILLER = ("Please continue handling this customer's request carefully and professionally, "
          "keeping the conversation clear and the customer informed at every step. ")


def load_docs():
    out = {}
    for fn in sorted(os.listdir(DOCDIR)):
        if not fn.endswith(".json"):
            continue
        try:
            d = json.load(io.open(os.path.join(DOCDIR, fn), encoding="utf-8", errors="replace"))
        except Exception:
            continue
        out[str(d.get("id") or fn[:-5])] = {"title": str(d.get("title") or ""),
                                            "content": str(d.get("content") or "")}
    return out


def gold_names(sim):
    out = set()
    for ck in ((sim.get("reward_info") or {}).get("action_checks") or []):
        a = ck.get("action") or {}
        ar = a.get("arguments") or {}
        nm = (ar.get("agent_tool_name") or ar.get("user_tool_name")
              or ar.get("discoverable_tool_name") or a.get("name"))
        if nm:
            out.add(str(nm))
    return out


def called_names(sim):
    out = collections.Counter()
    for _m, tc in F.calls(sim):
        a = F.argsof(tc)
        out[str(F.inner_name(a) or F.nameof(tc))] += 1
    return out


def ledger_of(sim):
    """원장 요약 — 무엇을 불렀고 어떤 엔티티가 목록에 떴나(판단 0·문자열 수집만)."""
    calls, ents = [], set()
    for m in (sim.get("messages") or []):
        for tc in (m.get("tool_calls") or []):
            a = F.argsof(tc)
            calls.append(str(F.inner_name(a) or F.nameof(tc)))
        if m.get("role") == "tool":
            for e in re.findall(r"\b((?:chk|sav|dbc|txn|cc|acc)_[A-Za-z0-9_]+)\b",
                                str(m.get("content") or "")):
                ents.add(e)
    seen, uniq = set(), []
    for c in calls:
        if c not in seen:
            seen.add(c)
            uniq.append(c)
    return uniq, sorted(ents)[:40]


def proc_lines(docs, tool):
    """그 도구를 지시하는 정책 절차 줄(축자)."""
    out = []
    for _id, d in docs.items():
        if tool not in d["content"]:
            continue
        for ln in STEP_RE.findall(d["content"]):
            s = " ".join(ln.split())
            if tool in s or re.match(r"(?i)\s*(before|first)\b", s):
                out.append("[%s] %s" % (d["title"][:48], s))
    return out[:8]


TOOLNAME_RE = re.compile(r"\b([a-z][a-z_]{6,}_\d{4})\b|\b(apply_for_credit_card|"
                         r"transfer_to_human_agents|log_verification|verify_identity|"
                         r"get_credit_card_accounts_by_user|get_referrals_by_user|"
                         r"submit_referral|submit_transaction|get_card_last_4_digits)\b")


def tool_universe(docs):
    """**문서가 정의한 도구 이름 전량**. 격리 팔에도 이걸 줘야 정보-맞춤이 된다.

    ⚠2026-08-19 스모크 실측: 이 목록 없이 물으면 모델이 `get_atm_fee_discrepancies` 같은
      **없는 이름을 지어낸다** — 그건 이행 결손이 아니라 우리가 재료를 뺀 것이다([[18]]).
    """
    out = set()
    for d in docs.values():
        for m in TOOLNAME_RE.finditer(d["content"]):
            out.add(m.group(1) or m.group(2))
    return sorted(out)


def user_ask(sim):
    for m in (sim.get("messages") or []):
        if m.get("role") == "user" and (m.get("content") or ""):
            return " ".join(str(m["content"]).split())[:400]
    return ""


def convo(sim, maxmsg=60, cut=400, tail=None):
    """대화를 텍스트로. `tail=N` 이면 **마지막 N 메시지**만(문맥 용량-반응 실험용)."""
    msgs = sim.get("messages") or []
    msgs = msgs[-tail:] if tail else msgs[:maxmsg]
    lines = []
    for m in msgs:
        r = m.get("role")
        if r == "tool":
            lines.append("TOOL RESULT: " + " ".join(str(m.get("content") or "").split())[:cut])
        elif r == "assistant":
            tcs = m.get("tool_calls") or []
            if tcs:
                for tc in tcs:
                    lines.append("ASSISTANT CALL: %s %s"
                                 % (F.nameof(tc), json.dumps(F.argsof(tc), ensure_ascii=False)[:200]))
            elif m.get("content"):
                lines.append("ASSISTANT: " + " ".join(str(m["content"]).split())[:cut])
        elif r == "user" and m.get("content"):
            lines.append("USER: " + " ".join(str(m["content"]).split())[:cut])
    return "\n".join(lines)


def call(port, msgs, temp, maxtok=400):
    body = json.dumps({"model": MODEL, "messages": msgs, "temperature": temp,
                       "max_tokens": maxtok}).encode("utf-8")
    req = urllib.request.Request("http://127.0.0.1:%d/v1/chat/completions" % port, data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.loads(r.read().decode("utf-8"))
    return d["choices"][0]["message"]["content"]


def plan_names(txt):
    """계획 배열에서 도구 이름만 순서대로."""
    t = re.sub(r"^```(?:json)?|```$", "", txt.strip(), flags=re.M).strip()
    i, j = t.find("{"), t.rfind("}")
    try:
        o = json.loads(t[i:j + 1])
    except Exception:
        return re.findall(r'"tool"\s*:\s*"([^"]+)"', t)
    out = []
    for step in (o.get("plan") or []):
        if isinstance(step, dict) and step.get("tool"):
            out.append(str(step["tool"]))
        elif isinstance(step, str):
            out.append(step)
    return out


def parse_tool(txt):
    t = re.sub(r"^```(?:json)?|```$", "", txt.strip(), flags=re.M).strip()
    i, j = t.find("{"), t.rfind("}")
    if i >= 0 and j > i:
        try:
            o = json.loads(t[i:j + 1])
            nm = o.get("tool")
            if isinstance(nm, dict):
                nm = nm.get("name")
            return (str(nm) if nm else None), o
        except Exception:
            pass
    m = re.search(r'"tool"\s*:\s*"([^"]+)"', t)
    return (m.group(1) if m else None), None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--cases", type=int, default=14)
    ap.add_argument("--arms", default="A_min,B_full,C_neg")
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--mode", default="next,plan")
    a = ap.parse_args()

    docs = load_docs()
    TOOLS = tool_universe(docs)
    cases = []
    for tag in TAGS:
        for sim in F.scored(tag, SUF):
            rw = (sim.get("reward_info") or {}).get("reward")
            if (rw or 0) >= 1.0:
                continue
            gn, cn = gold_names(sim), called_names(sim)
            for ck in ((sim.get("reward_info") or {}).get("action_checks") or []):
                if ck.get("action_match"):
                    continue
                aa = ck.get("action") or {}
                ar = aa.get("arguments") or {}
                nm = str(ar.get("agent_tool_name") or ar.get("user_tool_name")
                         or ar.get("discoverable_tool_name") or aa.get("name") or "")
                if not nm or cn.get(nm):
                    continue                      # 아예 안 부른 것만
                pl = proc_lines(docs, nm)
                if not pl:
                    continue                      # 정책 문장이 있는 것만(도달 조건은 아래에서)
                body = " ".join(" ".join(str(m.get("content") or "").split())
                                for m in (sim.get("messages") or []) if m.get("role") == "tool")
                reached = [s for s in pl if s.split("] ", 1)[-1][:55] in body]
                if not reached:
                    continue                      # **도달한 것만** — 이 프로브의 표적
                cases.append({"tag": tag, "task": F.task_id(sim), "trial": sim.get("trial"),
                              "tool": nm, "lines": pl, "reached": len(reached),
                              "gold": sorted(gn), "sim": sim})
    # (task, tool) 중복 제거 — 같은 결손을 두 번 세지 않는다
    seen, uniq = set(), []
    for c in cases:
        k = (c["task"], c["tool"])
        if k in seen:
            continue
        seen.add(k)
        uniq.append(c)
    uniq = uniq[:a.cases]
    print("문서가 정의한 도구 %d개 — 모든 팔에 동일하게 제공(정보-맞춤)" % len(TOOLS))
    print("표적 %d건 (정책 문장 도달 ∧ 호출 0)" % len(uniq))
    for c in uniq:
        print("  %-9s t%-2s %-44s 절차줄 %d(도달 %d)"
              % (c["task"], c["trial"], c["tool"], len(c["lines"]), c["reached"]))

    jobs = []
    for c in uniq:
        sim = c["sim"]
        calls_, ents = ledger_of(sim)
        ask = user_ask(sim)
        led = ("지금까지 호출한 도구: %s\n대화에 등장한 레코드 id: %s"
               % (", ".join(calls_[:25]) or "(없음)", ", ".join(ents[:25]) or "(없음)"))
        proc = "\n".join("- " + s for s in c["lines"])
        neg = FILLER * max(1, len(proc) // len(FILLER) + 1)
        neg = neg[:len(proc)]
        tools = "# 호출 가능한 도구(정책 문서가 정의한 것)\n" + ", ".join(TOOLS) + "\n\n"
        base = tools + ("# 손님 요청\n%s\n\n# 지금까지의 진행(원장)\n%s\n\n" % (ask, led))
        q = ("\n\n# 질문\n지금 시점에서 **다음에 호출할 도구 하나**를 정하라. "
             "JSON 하나로만 답하라: {\"tool\": \"<이름>\", \"arguments\": {…}}")
        prompts = {
            "A_min": base + "# 정책 절차(축자)\n" + proc + q,
            "C_neg": base + "# 안내\n" + neg + q,
            "B_full": tools + "# 대화 전문\n" + convo(sim) + "\n\n# 정책 절차(축자)\n" + proc + q,
        }
        # ★용량-반응 팔: 문맥 길이만 바꾸고 나머지는 동일(간섭 질량의 단조 효과를 잰다)
        for nn in (4, 8, 16, 32, 64):
            prompts["B_tail%d" % nn] = (tools + "# 대화(마지막 %d 메시지)\n" % nn
                                        + convo(sim, tail=nn) + "\n\n# 정책 절차(축자)\n" + proc + q)
        qplan = ("\n\n# 질문\n지금 시점에서 **남은 호출을 순서대로** 최대 8개까지 계획하라. "
                 "JSON 하나로만: {\"plan\": [{\"tool\": \"<이름>\"}, …]}")
        for arm in a.arms.split(","):
            for mode in a.mode.split(","):
                body = prompts[arm] if mode == "next" else prompts[arm].replace(q, qplan)
                sysmsg = SYS if mode == "next" else SYS_PLAN
                for k in range(a.n):
                    jobs.append({"case": c, "arm": arm, "k": k, "mode": mode,
                                 "temp": (0.0 if k == 0 else a.temp),
                                 "msgs": [{"role": "system", "content": sysmsg},
                                          {"role": "user", "content": body}]})

    print("\n작업 %d건 · 모델 %s" % (len(jobs), MODEL))
    lock, out = threading.Lock(), []

    def work(idx):
        while True:
            with lock:
                if not jobs:
                    return
                j = jobs.pop(0)
            try:
                txt = call(PORTS[idx % len(PORTS)], j["msgs"], j["temp"])
            except Exception as e:
                txt = "ERROR " + str(e)[:200]
            nm, obj = parse_tool(txt)
            plan = []
            if j["mode"] == "plan":
                plan = plan_names(txt)
                nm = plan[0] if plan else nm
            c = j["case"]
            row = {"task": c["task"], "trial": c["trial"], "tool": c["tool"], "arm": j["arm"],
                   "mode": j["mode"], "k": j["k"], "pred": nm, "plan": plan,
                   "hit_exact": bool(nm) and nm == c["tool"],
                   "hit_anygold": bool(nm) and nm in c["gold"],
                   "hit_in_plan": c["tool"] in plan,
                   "said_only": (not nm) and not plan,
                   "raw": txt[:400]}
            with lock:
                out.append(row)
                print("  %-9s %-7s k=%d pred=%-44s exact=%s anygold=%s"
                      % (row["task"], row["arm"], row["k"], str(nm)[:44],
                         row["hit_exact"], row["hit_anygold"]))

    ths = [threading.Thread(target=work, args=(i,)) for i in range(a.workers)]
    [t.start() for t in ths]
    [t.join() for t in ths]

    print("\n## 요약 (팔별)")
    print("%-8s %-6s %5s %9s %10s %9s %8s" % ("arm", "mode", "n", "exact", "anygold", "in_plan", "말만"))
    for arm in a.arms.split(","):
        for mode in a.mode.split(","):
            rs = [r for r in out if r["arm"] == arm and r["mode"] == mode]
            if not rs:
                continue
            print("%-8s %-6s %5d %9.2f %10.2f %9.2f %8.2f"
                  % (arm, mode, len(rs), sum(r["hit_exact"] for r in rs) / float(len(rs)),
                     sum(r["hit_anygold"] for r in rs) / float(len(rs)),
                     sum(r["hit_in_plan"] for r in rs) / float(len(rs)),
                     sum(r["said_only"] for r in rs) / float(len(rs))))

    print("\n## 표적별 (exact 적중률)")
    print("%-9s %-44s %s" % ("task", "빠뜨린 도구", " ".join("%-8s" % x for x in a.arms.split(","))))
    for c in uniq:
        cells = []
        for arm in a.arms.split(","):
            rs = [r for r in out if r["arm"] == arm and r["task"] == c["task"] and r["tool"] == c["tool"]]
            hit = sum((r["hit_in_plan"] if r["mode"] == "plan" else r["hit_exact"]) for r in rs)
            cells.append("%-8s" % ("%d/%d" % (hit, len(rs)) if rs else "-"))
        print("%-9s %-44s %s" % (c["task"], c["tool"][:44], " ".join(cells)))

    o = os.path.normpath(os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026",
                                      "x395_compliance_iso.json"))
    io.open(o, "w", encoding="utf-8").write(json.dumps(out, ensure_ascii=False, indent=1))
    print("\n원자료: %s" % o)
    return 0


if __name__ == "__main__":
    sys.exit(main())
