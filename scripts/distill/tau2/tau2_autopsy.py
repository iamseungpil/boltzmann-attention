#!/usr/bin/env python
"""τ² 궤적 전수 autopsy — 왜 통과/실패했나 기제 분류 (2026-06-14).

입력 = t2_run_gated 의 results.json (simulations[] = 궤적, reward_info.action_checks = gold 액션별 매치).
각 태스크별로 추출·분류:
  - reward / termination_reason
  - gold read/write 액션 수 + 매치 수 (write 매치 = 통과 핵심)
  - 에이전트 tool-call 시퀀스 / tool 에러 수
  - 첫 auth-call(find_user*) provenance: 인자값 ∈ 이전 user 발화? (날조 검출)
  - ask vs fetch 행동: 질문(?)턴 수 vs read-tool 호출 수
  - 실패 1차 분류(증거 기반·투명):
      PASS                : reward>0
      no_auth             : find_user 매치 실패(인증 자체 실패) → 이후 불가
      fab_auth            : 첫 auth 인자 날조(이전 user 발화에 없음)
      premature_refuse    : write 0회 + 거부성 발화("cannot","only","unable","sorry") = 유효요청 조기거부
      over_ask            : 질문턴 ≥2 & write 0 & tool 호출 빈약 = 묻기만 함(fetch 안 함)
      agent_collapse      : termination=agent error/max_steps/too_many_errors or tool에러≥3
      wrong_write         : auth OK인데 write 액션 불일치(틀린 행동)
      no_write            : auth OK인데 write 시도 0 (기타)
      other

Usage: tau2_autopsy.py <sim_dir_or_results.json> [--dump TASK_ID] [--full]
"""
import argparse, json, os, sys
from collections import Counter

REFUSE = ("cannot", "can't", "unable", "only allows", "not able", "i'm sorry", "i am sorry",
          "not possible", "won't be able", "policy", "not allowed")
AUTH_PREFIX = ("find_user", "get_user", "find_customer")


def load(path):
    if os.path.isdir(path):
        path = os.path.join(path, "results.json")
    return json.load(open(path, encoding="utf-8"))


def agent_calls(msgs):
    out = []
    for m in msgs:
        if m.get("role") == "assistant":
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function", tc)
                out.append((fn.get("name"), fn.get("arguments")))
    return out


def tool_errors(msgs):
    n = 0
    first = None
    for m in msgs:
        if m.get("role") == "tool":
            err = m.get("error")
            c = str(m.get("content") or "")
            if err or c.lower().startswith("error"):
                n += 1
                if first is None:
                    first = c[:80]
    return n, first


def ask_turns(msgs):
    n = 0
    for m in msgs:
        if m.get("role") == "assistant" and not m.get("tool_calls"):
            c = str(m.get("content") or "")
            if "?" in c:
                n += 1
    return n


def refuse_turns(msgs):
    n = 0
    for m in msgs:
        if m.get("role") == "assistant" and not m.get("tool_calls"):
            c = str(m.get("content") or "").lower()
            if any(r in c for r in REFUSE):
                n += 1
    return n


def prior_user_text(msgs, upto_idx):
    return " \n ".join(str(m.get("content") or "") for m in msgs[:upto_idx] if m.get("role") == "user").lower()


def first_auth_provenance(msgs):
    """첫 auth-call의 인자값이 이전 user 발화에 있나. (name, grounded_bool, args) / None."""
    for i, m in enumerate(msgs):
        if m.get("role") == "assistant":
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function", tc)
                nm = (fn.get("name") or "")
                if any(nm.startswith(p) for p in AUTH_PREFIX):
                    ut = prior_user_text(msgs, i)
                    args = fn.get("arguments")
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except Exception:
                            args = {}
                    vals = [str(v) for v in (args.values() if isinstance(args, dict) else []) if v not in (None, "")]
                    # 식별 가능한 값(길이>=2)만; 전부 user 발화에 있으면 grounded
                    chk = [v for v in vals if len(str(v)) >= 2]
                    grounded = all(str(v).lower() in ut for v in chk) if chk else None
                    return nm, grounded, args
    return None


def classify(sim):
    ri = sim.get("reward_info") or {}
    reward = ri.get("reward", 0.0)
    term = sim.get("termination_reason")
    acts = ri.get("action_checks") or []
    reads = [a for a in acts if a.get("tool_type") == "read"]
    writes = [a for a in acts if a.get("tool_type") == "write"]
    read_hit = [a for a in reads if a.get("action_match")]
    write_hit = [a for a in writes if a.get("action_match")]
    auth_actions = [a for a in acts if any((a["action"].get("name") or "").startswith(p) for p in AUTH_PREFIX)]
    auth_hit = [a for a in auth_actions if a.get("action_match")]
    msgs = sim.get("messages") or []
    calls = agent_calls(msgs)
    n_err, first_err = tool_errors(msgs)
    asks = ask_turns(msgs)
    refuses = refuse_turns(msgs)
    agent_writes = [c for c in calls if not any(c[0].startswith(p) for p in AUTH_PREFIX)
                    and c[0] not in ("get_order_details", "get_product_details", "list_all_product_types",
                                     "get_user_details", "calculate", "think", "transfer_to_human_agents")]
    prov = first_auth_provenance(msgs)

    if reward and reward > 0:
        cls = "PASS"
    elif auth_actions and not auth_hit:
        cls = "no_auth"
    elif prov and prov[1] is False:
        cls = "fab_auth"
    elif term in ("agent_error", "too_many_errors", "max_steps", "max_errors") or n_err >= 3:
        cls = "agent_collapse"
    elif not agent_writes and refuses >= 1:
        cls = "premature_refuse"
    elif not agent_writes and asks >= 2:
        cls = "over_ask"
    elif writes and not write_hit and agent_writes:
        cls = "wrong_write"
    elif writes and not agent_writes:
        cls = "no_write"
    else:
        cls = "other"
    return {
        "task": sim.get("task_id"), "reward": reward, "term": term, "cls": cls,
        "reads": "%d/%d" % (len(read_hit), len(reads)), "writes": "%d/%d" % (len(write_hit), len(writes)),
        "auth": "%d/%d" % (len(auth_hit), len(auth_actions)),
        "calls": len(calls), "agent_writes": len(agent_writes), "tool_err": n_err,
        "asks": asks, "refuses": refuses,
        "prov": (prov[0], prov[1]) if prov else None, "first_err": first_err,
        "n_msgs": len(msgs),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--dump", default=None, help="task_id 궤적 전문 출력")
    ap.add_argument("--full", action="store_true", help="모든 태스크 tool-call 시퀀스 출력")
    a = ap.parse_args()
    r = load(a.path)
    sims = r["simulations"]

    if a.dump is not None:
        s = next((x for x in sims if str(x.get("task_id")) == str(a.dump)), None)
        if not s:
            print("no such task", a.dump); return
        print("=== task %s  reward=%s term=%s ===" % (a.dump, (s.get("reward_info") or {}).get("reward"), s.get("termination_reason")))
        for m in s["messages"]:
            role = m.get("role")
            if role == "assistant" and m.get("tool_calls"):
                for tc in m["tool_calls"]:
                    fn = tc.get("function", tc)
                    print("  [A->CALL] %s(%s)" % (fn.get("name"), str(fn.get("arguments"))[:120]))
            elif role == "assistant":
                print("  [A] %s" % str(m.get("content"))[:200])
            elif role == "user":
                print("  [U] %s" % str(m.get("content"))[:200])
            elif role == "tool":
                print("    [T] %s" % str(m.get("content"))[:120])
        return

    rows = [classify(s) for s in sims]
    rows.sort(key=lambda x: (x["cls"] != "PASS", str(x["cls"])))
    print("=== τ² AUTOPSY: %s  (n=%d) ===" % (a.path, len(sims)))
    print("%-5s %-7s %-16s %-6s %-6s %-6s %4s %3s %3s %3s %s" %
          ("task", "reward", "class", "auth", "reads", "writes", "calls", "aw", "err", "ask", "prov/term"))
    for x in rows:
        print("%-5s %-7.2f %-16s %-6s %-6s %-6s %4d %3d %3d %3d %s | %s" %
              (str(x["task"]), x["reward"], x["cls"], x["auth"], x["reads"], x["writes"],
               x["calls"], x["agent_writes"], x["tool_err"], x["asks"],
               str(x["prov"]), x["term"]))
    print("\n=== 실패 모드 분포 ===")
    for cls, n in Counter(x["cls"] for x in rows).most_common():
        print("  %-16s %d" % (cls, n))
    npass = sum(1 for x in rows if x["cls"] == "PASS")
    print("\npass=%d/%d=%.2f" % (npass, len(rows), npass / len(rows)))
    # provenance 요약
    provs = [x["prov"] for x in rows if x["prov"]]
    gr = sum(1 for p in provs if p[1] is True); fab = sum(1 for p in provs if p[1] is False)
    print("auth provenance: grounded=%d fabricated=%d (n_auth_call=%d)" % (gr, fab, len(provs)))
    if a.full:
        print("\n=== 전 태스크 tool-call 시퀀스 ===")
        for s in sims:
            seq = " > ".join(c[0] for c in agent_calls(s.get("messages") or []))
            print("  task %-4s [%s] %s" % (s.get("task_id"), (s.get("reward_info") or {}).get("reward"), seq[:160]))


if __name__ == "__main__":
    main()
