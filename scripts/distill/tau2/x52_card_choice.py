"""Was the right card already in the agent's hands when it recommended the wrong one?

Five B4 failures die on a single argument: `apply_for_credit_card.card_type`. That is
either a retrieval problem (the right card was never surfaced) or a selection problem
(it was surfaced and something else was chosen). The two have nothing in common as
prescriptions, so the question is settled here rather than argued.

`check_card_application_fit` returns the eligible set with each card's facts, so for
every such failure this reports:

  gold-in-candidates   the card gold wanted appeared in a fit result or a KB hit
  recommended          the card the agent named to the customer
  challenged           the customer pushed back on the recommendation afterwards

If gold was in the candidate set, no index and no retrieval lever reaches this failure.
"""

import argparse
import glob
import gzip
import json
import re

SIM = ("/home/woori/workspace_common/boltzmann-attention-pi/"
       "reports/facet_rft_2026/sim_results")

ARMS = {
    "A":  "bank_ax33n_gpu*_20260803g",
    "B4": "bank_b4_gpu*_20260803h",
}

# The customer disputing the choice, as opposed to asking a new question.
CHALLENGE = re.compile(
    r"\b(best option|top recommendation|better card|vs any other|are you sure|"
    r"definitely the best|is that right|instead of|why (not|that) card|"
    r"other (cards|options)|compare)\b", re.I)


def norm(a):
    if isinstance(a, str):
        try:
            return json.loads(a)
        except Exception:
            return {}
    return a if isinstance(a, dict) else {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="A,B4")
    args = ap.parse_args()

    for arm in args.arms.split(","):
        print(f"\n=== [{arm}] card_type 실패 ===")
        for p in sorted(glob.glob(f"{SIM}/{ARMS[arm]}.results.json.gz")):
            for s in json.load(gzip.open(p, "rt", encoding="utf-8")).get("simulations") or []:
                ri = s.get("reward_info") or {}
                if (ri.get("reward") or 0.0) == 1.0:
                    continue
                gold = None
                for c in ri.get("action_checks") or []:
                    a = c.get("action") or {}
                    if not c.get("action_match") and a.get("name") == "apply_for_credit_card":
                        gold = norm(a.get("arguments")).get("card_type")
                if not gold:
                    continue

                # Everything the tools handed back, and everything the agent said.
                tool_text, agent_text, user_text = [], [], []
                applied = None
                for m in s.get("messages") or []:
                    role, c = m.get("role"), m.get("content")
                    if role == "tool" and isinstance(c, str):
                        tool_text.append(c)
                    elif role == "assistant" and isinstance(c, str):
                        agent_text.append(c)
                    elif role == "user":
                        if isinstance(c, str):
                            user_text.append(c)
                        for tc in m.get("tool_calls") or []:
                            if (tc.get("name") or "") == "apply_for_credit_card":
                                applied = norm(tc.get("arguments")).get("card_type")
                tools, said = "\n".join(tool_text), "\n".join(agent_text)

                in_cand = gold.lower() in tools.lower()
                # Which cards did the agent name? Any "... Card" phrase in its prose.
                named = set(re.findall(r"\b([A-Z][\w+]*(?: [A-Z][\w+]*){0,3} Card)\b", said))
                challenged = sum(1 for u in user_text if CHALLENGE.search(u))

                print(f"  {s['task_id']}/t{s.get('trial')}")
                print(f"    gold            : {gold}")
                print(f"    후보에 있었나     : {'YES — 회수 아님' if in_cand else 'no'}")
                print(f"    손님이 실제 신청  : {applied}")
                print(f"    에이전트가 언급한 카드: {', '.join(sorted(named)) or '-'}")
                print(f"    손님 재도전 횟수  : {challenged}")


if __name__ == "__main__":
    main()
