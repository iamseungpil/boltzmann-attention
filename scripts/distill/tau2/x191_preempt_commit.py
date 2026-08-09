# -*- coding: utf-8 -*-
r"""x191 — **블록을 자기-약속보다 먼저 보내면 14B 도 붙는가** (유료 0·사용자 발의).

## 왜

x190: `no_table B_ops`(지목+근거)가 32B 는 두 태스크 8/8·날조 0 인데 **14B/100 은 1/8** —
궤적에 이미 박힌 자기 약속(`Cobalt Blue`)을 우리 블록이 못 이긴다. 같은 14B 가 x184 의
**대화-없는** 조건에서는 `+commit→gold` 8/8 이었다.

그리고 배관 확인: `t2_gate_patch.py:6178` `rw_fb = ((am.tool_calls or [None])[0], _ufb)` —
`am` 은 **이미 생성된 에이전트 메시지**다. 우리 블록은 **설계상 항상 약속 뒤**에 온다.

사용자 발의: *"자기 약속을 서브 진행되는 동안 멈추게 하면 되지 않나"* — 즉 **선제 주입**.
그 배선을 하기 전에, 그 조건이 실제로 이기는지 먼저 잰다(대리 = 꼬리 약속 턴 제거 + 블록).

## 축

  ctx   full      대화 그대로 (약속 포함)          — x190 재현 기준선
        -commit   꼬리 assistant 턴 제거           — **선제 주입의 대리**
  block none · B_min(지목만) · B_ops(지목+피연산자+선택 행)
        · B_rank(B_ops + 통과 집합 상위 3위)       — x190 의 `Q_second` 실패 보완
  q     Q_main · Q_second · Q_why    표는 **어느 arm 에도 없다**(메인은 표를 안 갖는다)

## 읽는 법

  · `-commit + B_ops` 에서 14B/100 이 붙으면  → **선제 주입 배선이 정당화된다.** 처방 확정.
  · 그래도 안 붙으면                          → 스케일 바닥이 실재 → 14B 단독 배치는 불가.
  · `B_rank` 가 `Q_second` 를 고치면           → 블록에 순위를 싣는 것으로 후속이 닫힌다.

실행: python x191_preempt_commit.py [N]
"""
import collections
import json
import os
import re
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import x149_choice_isolation as X                              # noqa: E402
import x150_choice_ablation as Y                               # noqa: E402
import t2_factdag as FD                                        # noqa: E402
import t2_ledger as LG                                         # noqa: E402
from gate_interpreter import load_domain_a2                     # noqa: E402

URL = os.environ.get("T2_PROBE_URL", "http://localhost:8140/v1/chat/completions")
MODEL = os.environ.get("T2_PROBE_MODEL", "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8")
TAG = os.environ.get("T2_PROBE_TAG", "bank_elig_20260809i")
CASE = {"task_099": {"days": 730, "deposit": 30000},
        "task_100": {"days": 65, "deposit": 31000}}
ACCOUNT_AXIS = "referrer_tenure_days"
LEAD = "Here is a customer-service conversation so far.\n\n"
Q_SECOND = ("Which ONE business checking account is the SECOND best for this customer, "
            "if the customer wants to maximise the bonus THEY receive? "
            "Answer with the account name only, nothing else.")
Q_WHY = ("Why that account? State the referral bonus amount the customer would receive "
         "and where that number comes from. Two sentences at most.")


def call(prompt, choices=None, temp=0.0, mx=12):
    body = {"model": MODEL, "temperature": temp, "max_tokens": mx,
            "messages": [{"role": "user", "content": prompt}]}
    if choices:
        body["guided_choice"] = list(choices)
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return " ".join((json.load(r)["choices"][0]["message"]["content"] or "").split())


def drop_tail_assistant(ms):
    out = list(ms)
    while out and out[-1].get("role") == "assistant":
        out.pop()
    return out


def drop_named_sentences(text, names):
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    return " ".join(s for s in sents if not any(nm in s for nm in names))


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    a2 = load_domain_a2("banking_knowledge")
    spec = next(s for s in a2["ledger_metrics"] if s.get("eligible_text"))
    rows = a2["policy_ontology"]["rows"]
    axes = spec["eligible"]["show_axes"]
    maps = {ax: FD._a3_map(rows, {"axis": ax}) for ax in axes}
    bax = next(a for a in axes if "bonus" in a.lower() and "referrer" in a.lower())
    name = lambda l: l.strip().split(":")[0].strip()          # noqa: E731

    def bval(s):
        v = (maps.get(bax) or {}).get(s)
        try:
            return float(str(v[0]).replace(",", ""))
        except Exception:
            return -1.0

    print("model=%s · n=%d" % (MODEL, n))
    out = []
    for task, case in CASE.items():
        gold = X.GOLD[task]
        lines = LG.eligible_text(case["days"], {}, maps, spec,
                                 {"qualifying_deposit_usd": case["deposit"]}).strip().splitlines()
        body = [l for l in lines if l.startswith("  ") and ":" in l]
        ALL = [name(l) for l in body]
        CHK = [s for s in ALL if s in (maps.get(ACCOUNT_AXIS) or {})]
        rank = sorted(CHK, key=lambda s: (-bval(s), s))
        second = rank[1]
        gold_row = next(l.strip() for l in body if name(l) == gold)
        facts = drop_named_sentences(X.FACTS[task], ALL)

        b_min = "An isolated check was run on the policy constants on record.\nResult: %s." % gold
        b_ops = b_min + "\nOperands it used: %s\nThe row it selected: %s" % (facts, gold_row)
        b_rank = b_ops + ("\nThe next best eligible ones, in order: %s"
                          % "; ".join("%s (%s=%d)" % (s, bax, bval(s)) for s in rank[1:4]))

        ms = Y.msgs_of(TAG, task)
        convs = [("full   ", Y.render(ms)),
                 ("-commit", Y.render(drop_tail_assistant(ms)))]
        blocks = [("none  ", ""), ("B_min ", b_min), ("B_ops ", b_ops), ("B_rank", b_rank)]

        print("\n" + "=" * 100)
        print("%s  gold=%r($%d) · 2위=%r($%d)" % (task, gold, bval(gold), second, bval(second)))
        print("=" * 100)
        print("  %-8s %-7s | %-19s | %-19s | %s"
              % ("ctx", "block", "Q_main(gold)", "Q_second(2위)", "Q_why 정합/날조"))
        for clabel, conv in convs:
            for blabel, blk in blocks:
                pre = LEAD + conv + (("\n\n" + blk) if blk else "")
                c1 = collections.Counter(call(pre + "\n\n" + X.QUESTION, ALL,
                                              0.0 if i == 0 else 0.7) for i in range(n))
                c2 = collections.Counter(call(pre + "\n\n" + Q_SECOND, ALL,
                                              0.0 if i == 0 else 0.7) for i in range(n))
                okn = bad = 0
                whys = []
                for i in range(min(n, 5)):
                    w = call(pre + "\n\n" + X.QUESTION + "\n" + Q_WHY, None,
                             0.0 if i == 0 else 0.7, 120)
                    whys.append(w)
                    nums = {x.replace(",", "") for x in re.findall(r"\d[\d,]*", w)}
                    if str(int(bval(gold))) in nums:
                        okn += 1
                    elif nums:
                        bad += 1
                print("  %-8s %-7s | %d/%-2d %-14s | %d/%-2d %-14s | %d/%d · 틀림 %d"
                      % (clabel, blabel, c1.get(gold, 0), n, c1.most_common(1)[0][0][:14],
                         c2.get(second, 0), n, c2.most_common(1)[0][0][:14],
                         okn, min(n, 5), bad))
                out.append({"task": task, "ctx": clabel.strip(), "block": blabel.strip(),
                            "n": n, "main_gold": c1.get(gold, 0), "main_dist": dict(c1),
                            "second_gold": c2.get(second, 0), "second_dist": dict(c2),
                            "why_ok": okn, "why_bad": bad, "why": whys})

    json.dump(out, open(os.environ.get("T2_X191_OUT", "x191_out.json"), "w"), indent=1)
    print("\n  -commit+B_ops 에서 14B/100 이 붙으면 선제 주입 배선이 정당화된다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
