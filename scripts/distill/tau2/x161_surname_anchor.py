# -*- coding: utf-8 -*-
"""x161 — 정박은 *이름* 단위인가 *성(姓)* 단위인가 + §6.2a 가 요구하는 B′(방아쇠 제거).

## 왜 (2026-08-09 · C348 교차 + 사용자 질문 "j 가 아니라 내용 아닌가")

절벽을 지는 메시지는 `[26] assistant` = **우리 사과·전환 문장**이고 거기 지목된 계좌는
**Hunter Green** 이다. 그런데 무너진 뒤 32B 가 고르는 것은 **Lime Green**(x160 자유생성 5/5).
지목된 이름을 베끼지 않고 **같은 성(Green) 가족**으로 미끄러진다면, 정박의 단위는 토큰-정확
이름이 아니라 **가족**이다(C333 `Green ↔ Green Fee-Free` 와 같은 기전).

## arm (전부 같은 궤적·같은 base·바꾸는 것은 [26] 한 줄뿐)

  A_orig    : 원본 그대로 (지목 = Hunter **Green**)
  B_removed : [26] **제거** — C348 의 "0.9948 복귀" 재현 = §6.2a 의 B′ baseline
  C_<X>     : [26] 의 지목 계좌만 **다른 성**으로 치환(gold 가족 Blue 는 피한다 — 교락)
              가족이 따라오면 성-단위 정박 확정, Lime Green 에 남으면 반증.

측정 = `guided_choice` 첫 토큰 분포를 **가족(성)별로 합산**. 1글자 토큰도 접두로 귀속시킨다
(x159 자기정정 #9: 붕괴 행 질량 0.97 이 `'L'` 에 앉는다 — 버리면 argmax 를 오귀속한다).

실행: py -3 x161_surname_anchor.py     (8140=32B 필요)
"""
import json
import math
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

URL = "http://localhost:8140/v1"
TASK = "task_099"
TAG = os.environ.get("T2_PROBE_TAG", "bank_elig_20260809i")
TRIGGER = 26          # 절벽을 지는 메시지 index (C348)
NAMED = "Hunter Green"


def model_of(u):
    with urllib.request.urlopen(u + "/models", timeout=30) as r:
        return json.load(r)["data"][0]["id"]


def dist(model, prompt, choices, top=20):
    body = json.dumps({"model": model, "temperature": 0.0, "max_tokens": 1,
                       "logprobs": True, "top_logprobs": top,
                       "guided_choice": list(choices),
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(URL + "/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        d = json.load(r)
    return {e["token"].strip(): e["logprob"]
            for e in d["choices"][0]["logprobs"]["content"][0]["top_logprobs"]}


def family_mass(dst, fams):
    """토큰 확률을 **가족**에 귀속. 1글자 토큰도 포함하되, 여러 가족의 접두면 배분 불가로 별도 계상."""
    out = {f: 0.0 for f in fams}
    amb = 0.0
    for tok, lp in dst.items():
        p = math.exp(lp)
        hit = [f for f, names in fams.items()
               if any(n.lower().startswith(tok.lower()) for n in names)]
        if len(hit) == 1:
            out[hit[0]] += p
        elif len(hit) > 1:
            amb += p
    return out, amb


def main():
    a2 = load_domain_a2("banking_knowledge")
    spec = next(s for s in a2["ledger_metrics"] if s.get("eligible_text"))
    rows = a2["policy_ontology"]["rows"]
    maps = {ax: FD._a3_map(rows, {"axis": ax}) for ax in spec["eligible"]["show_axes"]}
    table = LG.eligible_text(730, {}, maps, spec, {"qualifying_deposit_usd": 30000}).strip()
    base = table + "\n\n" + X.FACTS[TASK] + "\n\n" + X.QUESTION
    CH = [l.strip().split(":")[0].strip() for l in table.splitlines() if l.startswith("  ")]
    # 가족 = 이름의 **마지막 낱말**(Green/Blue/Card…)이 아니라, 실제로 공유되는 성 토큰으로 잡는다.
    fams = {}
    for c in CH:
        w = c.split()
        key = "Green" if "Green" in w else ("Blue" if "Blue" in w else
              ("Card" if w[-1] == "Card" else w[0]))
        fams.setdefault(key, []).append(c)
    model = model_of(URL)
    MS = Y.msgs_of(TAG, TASK)
    print("model=%s  후보 %d  가족 %s" % (model, len(CH),
          {k: len(v) for k, v in sorted(fams.items(), key=lambda e: -len(e[1]))}))
    print("\n[방아쇠 원문 #%d] %s" % (TRIGGER, " ".join(
        str(MS[TRIGGER].get("content") or "").split())[:190]))

    # gold 가족(Blue)이 아닌 성으로 치환 후보를 고른다 — 교락 회피
    alts = [c for c in CH if "Green" not in c and "Blue" not in c][:2]

    def render(msgs):
        return ("Here is a customer-service conversation so far.\n\n"
                + Y.render(msgs) + "\n\n") + base

    def swap(name):
        ms = [dict(m) for m in MS]
        ms[TRIGGER]["content"] = str(ms[TRIGGER].get("content") or "").replace(NAMED, name)
        return ms

    arms = [("A_orig  (지목=%s)" % NAMED, MS),
            ("B_removed (#%d 제거)" % TRIGGER, MS[:TRIGGER] + MS[TRIGGER + 1:])]
    arms += [("C_%s" % a, swap(a)) for a in alts]

    print("\n%-26s %8s | %s" % ("arm", "p(gold)", "가족별 질량 상위"))
    for label, ms in arms:
        d = dist(model, render(ms), CH)
        fm, amb = family_mass(d, fams)
        gold_p = sum(math.exp(lp) for t, lp in d.items()
                     if len(t) >= 2 and "world blue".startswith(t.lower()))
        top = sorted(fm.items(), key=lambda e: -e[1])[:4]
        print("%-26s %8.4f | %s%s" % (label, gold_p,
              "  ".join("%s=%.3f" % (k, v) for k, v in top if v > 1e-4),
              ("  [모호 %.3f]" % amb if amb > 1e-3 else "")))
    print("\n판정: C arm 의 질량이 치환한 성으로 따라가면 **성-단위 정박** · "
          "Green/L-계에 남으면 반증(이름 무관한 위치 효과)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
