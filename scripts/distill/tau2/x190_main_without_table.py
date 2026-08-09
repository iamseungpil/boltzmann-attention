# -*- coding: utf-8 -*-
r"""x190 — **메인에서 표를 빼도 되는가**, 그리고 **후속 질문을 감당하는가** (유료 0·사용자 지시).

설계 근거 = `ANCHOR_SLOT_SPEC_2026_08_09.md` · `ISOLATION_GRID_DESIGN_2026_08_09.md`.

## 왜

x187 전 셀 대조에서 `L3`(대화 없음)가 `L0`(대화 포함)를 **2모델×20셀 전부 파레토 지배**했다.
표를 **메인**에 둘 측정 근거는 0 이고, 토큰 회계도 크다(표 2,611~2,895자 × sim당 1~2회 주입 ×
이후 전 턴 재전송). 그러나 *"메인에 표 없음"* arm 은 **아직 잰 적이 없다.**

이 프로브가 재는 것 둘:
  ⒜ **결정** — 메인이 대화(자기 약속 포함)를 그대로 갖고, 표 대신 **서브 블록**만 받으면 붙는가
  ⒝ **후속 감당** — 표 없이 *"왜 그 계좌인가"* · *"두 번째로 좋은 것은"* 에 답할 때 **날조하는가**([[25]])

⒝ 가 이 구조의 진짜 위험이다. 토큰 이득만 보고 넘어가면 안 된다.

## 축

  ctx  with_table_asc / with_table_desc   현행 (대화 + 표 + 사실 + 질문)
       no_table + B_min                   대화 + **지목 한 줄** + 질문
       no_table + B_ops                   대화 + **지목 + 그 행의 상수 + 피연산자**(규격서 §2) + 질문
       no_table + B_nopick                대화 + **근거만·지목 없음**(D1=OFF) + 질문

  q    Q_main   현행 결정 질문                      (guided · gold = 엔진 argmax)
       Q_second "두 번째로 좋은 것"                 (guided · gold = 엔진 2위)
       Q_why    "왜 그 계좌인가 · 보너스 액수와 출처" (자유생성 · 숫자 정합·날조 검사)

## 읽는 법

  · `B_min` 이 `with_table` 을 이기면    → **우리 블록이 옛 자기-약속을 이긴다**(x184 예측). 표 제거 가능.
  · `B_nopick` 이 지면                    → 지목 없이 블록만으론 부족 → D1 결정이 필요해진다.
  · `Q_second` 가 표 없이도 맞으면        → 블록이 후속을 감당한다. 틀리면 **표를 완전히 뺄 수 없다**.
  · `Q_why` 의 숫자가 틀리면              → **날조**. 토큰 이득보다 이 손실이 크다([[25]]).

실행: python x190_main_without_table.py [N]
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
        head = [l for l in lines if not (l.startswith("  ") and ":" in l)]
        body = [l for l in lines if l.startswith("  ") and ":" in l]
        ALL = [name(l) for l in body]
        CHK = [s for s in ALL if s in (maps.get(ACCOUNT_AXIS) or {})]
        rank = sorted(CHK, key=lambda s: (-bval(s), s))
        second = rank[1]
        gold_row = next(l.strip() for l in body if name(l) == gold)
        facts = drop_named_sentences(X.FACTS[task], ALL)      # x186·R5: 피연산자만
        conv = Y.render(Y.msgs_of(TAG, task))

        # 서브 블록 — 엔진이 조립한다(규격서 §2). 세 세기.
        b_min = ("An isolated check was run on the policy constants on record.\n"
                 "Result: %s." % gold)
        b_ops = (b_min + "\nOperands it used: %s\nThe row it selected: %s"
                 % (facts, gold_row))
        b_nopick = ("An isolated check was run on the policy constants on record.\n"
                    "Operands it used: %s\n"
                    "%d products passed the eligibility arithmetic; the check did not "
                    "recommend one." % (facts, len(body)))

        def tbl(rev):
            order = sorted(body, key=name, reverse=rev)
            return "\n".join(head[:1] + order + head[1:]).strip()

        ctxs = [("with_table_asc ", LEAD + conv + "\n\n" + tbl(False) + "\n\n" + facts),
                ("with_table_desc", LEAD + conv + "\n\n" + tbl(True) + "\n\n" + facts),
                ("no_table B_min ", LEAD + conv + "\n\n" + b_min),
                ("no_table B_ops ", LEAD + conv + "\n\n" + b_ops),
                ("no_table B_nopk", LEAD + conv + "\n\n" + b_nopick)]

        print("\n" + "=" * 104)
        print("%s  gold=%r($%d) · 엔진 2위=%r($%d) · 블록 길이 min %d · ops %d · nopick %d"
              % (task, gold, bval(gold), second, bval(second),
                 len(b_min), len(b_ops), len(b_nopick)))
        print("  문맥 길이: with_table %d자 · no_table(B_ops) %d자  (%.0f%% 절감)"
              % (len(ctxs[0][1]), len(ctxs[3][1]),
                 100.0 * (1 - len(ctxs[3][1]) / float(len(ctxs[0][1])))))
        print("=" * 104)
        print("  %-16s | %-20s | %-20s | %s"
              % ("ctx", "Q_main(gold)", "Q_second(2위)", "Q_why 숫자정합/날조"))
        for clabel, pre in ctxs:
            c1 = collections.Counter(call(pre + "\n\n" + X.QUESTION, ALL,
                                          0.0 if i == 0 else 0.7) for i in range(n))
            c2 = collections.Counter(call(pre + "\n\n" + Q_SECOND, ALL,
                                          0.0 if i == 0 else 0.7) for i in range(n))
            whys, okn, bad = [], 0, 0
            for i in range(min(n, 5)):
                w = call(pre + "\n\n" + X.QUESTION + "\n" + Q_WHY, None,
                         0.0 if i == 0 else 0.7, 120)
                whys.append(w)
                nums = {x.replace(",", "") for x in re.findall(r"\d[\d,]*", w)}
                if str(int(bval(gold))) in nums:
                    okn += 1
                elif nums:
                    bad += 1
            print("  %-16s | %d/%-2d %-15s | %d/%-2d %-15s | 맞음 %d/%d · 틀린수 %d"
                  % (clabel, c1.get(gold, 0), n, c1.most_common(1)[0][0][:15],
                     c2.get(second, 0), n, c2.most_common(1)[0][0][:15],
                     okn, min(n, 5), bad))
            out.append({"task": task, "ctx": clabel.strip(), "n": n,
                        "main_gold": c1.get(gold, 0), "main_dist": dict(c1),
                        "second_gold": c2.get(second, 0), "second_dist": dict(c2),
                        "why_num_ok": okn, "why_num_bad": bad, "why": whys})
        print("  [Q_why 샘플·%s] %s" % (ctxs[3][0].strip(), out[-2]["why"][0][:220]))

    json.dump(out, open(os.environ.get("T2_X190_OUT", "x190_out.json"), "w"), indent=1)
    print("\n  B_min 이 with_table 을 이기면 표 제거 가능 · Q_second/Q_why 가 틀리면 날조 위험([[25]]).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
