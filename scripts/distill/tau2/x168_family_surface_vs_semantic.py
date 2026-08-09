# -*- coding: utf-8 -*-
r"""x168 — "가족"은 **표면 문자열**인가 **의미 범주**인가 (유료 0·§6.4 급소 ①).

## 왜

§6.3 은 정박이 $Y$ 를 *가족*으로 좁힌다고 했고, 가족을 **이름의 공통 낱말**로 잡았다. 사후적
정의다. A3 는 계열을 **기계적으로** 준다(`source.doc`: `business_checking` / `checking` /
`business_credit_cards` / `credit_cards`) ⇒ 두 정의가 **다른 답을 예측하는 지목**을 고르면 갈린다.

x161(다른 세션)은 치환 후보를 *Green·Blue 가 아닌 것* 에서 골랐는데 그건 카드라서
표면 가족 최대(Business Platinum $300) = 의미 범주 최대(같은 $300)로 **교락**이다.

## 이미 손안의 증거 (A_orig)

`Hunter Green` 은 `business_checking` 계열이다.
  · 표면("Green") 최대 = **Lime Green $200**
  · 의미(business_checking) 최대 = **World Blue $300** ( = gold = 무정박 답)
관측(x160·x167 자유생성 10/10) = **Lime Green** ⇒ **표면이 이긴다.** 이 파일은 그것을
독립 지목들로 굳힌다.

## arm 과 **사전 등록 예측** (돌리기 전에 적는다·[[08]])

| 지목 | 표면 가설 | 의미 가설 |
|---|---|---|
| Hunter Green (양성통제) | Lime Green      | World Blue |
| Dark Green              | Lime Green      | Bluest |
| Gold Years              | **Business Gold Rewards Card** (계열을 건너뛴다) | Bluest |
| EcoCard                 | (공유 낱말 없음) World Blue | Platinum Rewards Card |

**`Gold Years` 가 결정적이다** — 표면 가설은 당좌에서 **기업 신용카드로 건너뛰는** 예측을
내고, 의미 가설은 당좌 안에 머문다. 건너뛰면 표면 확정이다.
**`EcoCard` 는 반대 방향**: 공유 낱말이 없는데도 카드로 좁혀지면 의미가 살아 있다는 뜻이다.

⚠gold 가족(Blue) 지목은 교락이라 쓰지 않는다(표면 최대 = World Blue = 무정박 답).

실행: py -3 x168_family_surface_vs_semantic.py [N]   (8140 = 32B 필요)
"""
import collections
import json
import os
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
TASK = "task_099"
TRIGGER = 26                     # 절벽을 지는 메시지 index (C348·다른 세션 x161 과 동일)
NAMED = "Hunter Green"
ANCHORS = ["Hunter Green", "Dark Green", "Gold Years", "EcoCard"]


def guided_full(prompt, choices, temp):
    body = json.dumps({"model": MODEL, "temperature": temp, "max_tokens": 12,
                       "guided_choice": list(choices),
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return " ".join((json.load(r)["choices"][0]["message"]["content"] or "").split())


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    a2 = load_domain_a2("banking_knowledge")
    spec = next(s for s in a2["ledger_metrics"] if s.get("eligible_text"))
    rows = a2["policy_ontology"]["rows"]
    axes = spec["eligible"]["show_axes"]
    maps = {ax: FD._a3_map(rows, {"axis": ax}) for ax in axes}
    table = LG.eligible_text(730, {}, maps, spec, {"qualifying_deposit_usd": 30000}).strip()
    base = table + "\n\n" + X.FACTS[TASK] + "\n\n" + X.QUESTION
    CH = [l.strip().split(":")[0].strip() for l in table.splitlines() if l.startswith("  ")]

    # ── 두 가족 정의를 **둘 다 기계적으로** 만든다 (프로브가 분류를 발명하지 않는다) ──
    bonus_ax = next((ax for ax in axes if "bonus" in ax.lower()), axes[0])
    bonus = {s: FD._num(v[0]) if hasattr(FD, "_num") else float(str(v[0]).replace(",", ""))
             for s, v in (maps.get(bonus_ax) or {}).items()}
    cat = {}
    for r in rows:
        s, d = (r or {}).get("subject"), ((r or {}).get("source") or {}).get("doc")
        if s and d and s not in cat:
            cat[s] = "_".join(str(d).split("_")[1:3])      # doc_<계열>_... 의 계열부
    def surname(c):                                        # 표면: 공유되는 낱말
        w = c.split()
        for k in ("Green", "Blue", "Gold", "Silver", "Platinum"):
            if k in w:
                return k
        return w[-1] if w else c

    def best(pool):
        pool = [c for c in pool if bonus.get(c) is not None]
        return max(pool, key=lambda c: bonus[c]) if pool else None

    print("model=%s · 후보 %d" % (MODEL, len(CH)))
    print("\n%-16s %-10s %-22s %-24s %s"
          % ("지목", "보너스", "계열(A3)", "표면가족 최대", "의미범주 최대"))
    pred = {}
    for a in ANCHORS:
        sf = [c for c in CH if surname(c) == surname(a)]
        sm = [c for c in CH if cat.get(c) and cat.get(c) == cat.get(a)]
        pred[a] = (best(sf), best(sm))
        print("%-16s %-10s %-22s %-24s %s"
              % (a, bonus.get(a), cat.get(a), pred[a][0], pred[a][1]))

    MS = Y.msgs_of(os.environ.get("T2_PROBE_TAG", "bank_elig_20260809i"), TASK)

    def render(msgs):
        return ("Here is a customer-service conversation so far.\n\n"
                + Y.render(msgs) + "\n\n" + base)

    def swap(name):
        ms = [dict(m) for m in MS]
        ms[TRIGGER]["content"] = str(ms[TRIGGER].get("content") or "").replace(NAMED, name)
        return ms

    print("\n%-16s %-26s %-16s %s" % ("지목", "관측(자유생성)", "표면예측", "의미예측"))
    for a in ANCHORS:
        got = [guided_full(render(swap(a)), CH, 0.0 if i == 0 else 0.7) for i in range(n)]
        top = collections.Counter(got).most_common(2)
        sf, sm = pred[a]
        mark = ("표면 ✓" if top[0][0] == sf else ("의미 ✓" if top[0][0] == sm else "둘 다 아님"))
        print("%-16s %-26s %-16s %-16s  %s" % (a, top, sf, sm, mark))
    got = [guided_full(render(MS[:TRIGGER] + MS[TRIGGER + 1:]), CH, 0.0 if i == 0 else 0.7)
           for i in range(n)]
    print("%-16s %-26s (정박 없음·baseline)" % ("(#26 제거)", collections.Counter(got).most_common(2)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
