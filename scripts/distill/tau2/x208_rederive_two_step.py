# -*- coding: utf-8 -*-
r"""x208 — 재도출 **2단 전체**(지목 → 불일치 재질의)를 격리로 재현한다 (유료 0 · 엔진 변경 0).

## 왜 x207 로는 부족한가

x207 은 **첫 지목만** 쟀다. 라이브 경로는 두 칸이다:

    rederive_choice → (없으면 끝) → formalize_objective_axis → mismatch_value → reask → rederive_choice

그리고 호출부는 `if _pick and _sp2.get("reask_prompt")` 라, **첫 지목이 `NONE` 이면 D1c 가 아예
안 돈다.** 098 이 라이브 3/3 침묵한 이유가 이것이다 — 첫 칸이 막혀 둘째 칸이 기회를 못 얻는다.

그래서 이 프로브는 **엔진과 같은 순서로 두 칸을 다 돈다**. 재구현이 아니라 같은 함수를 부른다
(`LG.mismatch_value` · A2 `reask_prompt`).

## 팔

  CUR       현행 문구 (`asked=""` · NONE 조항 있음)
  NOCLAUSE  손님-말 블록과 NONE 조항을 뺀 문구

각 팔에서 **1단 답**과 **2단 최종답**을 따로 센다. 이 대비가 이 파일의 전부다 —
*"조항을 빼면 D1c 가 돌 기회를 얻는가, 그리고 그것이 답을 고치는가."*

⚠목적 축은 **손님 말에서 형식화**한다(정적 선언 = [[23]] 위반). 엔진과 같이 `axes` 집합의
  원소인지만 검사한다. 못 구하면 D1c 를 건너뛴다 — 모르는 것을 기준으로 재질의하지 않는다.
⚠엔진은 이름을 돌려주지 않는다 — `mismatch_value` 는 **값** 두 개만 돌려주고 고르는 것은 모델이다.

실행: python x208_rederive_two_step.py [N]
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

import t2_ledger as LG                                          # noqa: E402
import t2_factdag as FD                                         # noqa: E402
from gate_interpreter import load_domain_a2                      # noqa: E402
from x207_rederive_objective import CASE                         # noqa: E402

URL = os.environ.get("T2_PROBE_URL", "http://localhost:8141/v1/chat/completions")
MODEL = os.environ.get("T2_PROBE_MODEL", "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8")
# 손님의 첫 발화 — 목적 축 형식화는 **이 말**에서 나온다(우리가 축을 고르지 않는다).
SAID = {
    "task_098": ("Hey! My roommate just got her first real job and she's going to deposit her "
                 "first paycheck - around $600. I want to refer her to open one of the accounts "
                 "you offer. Which one gives the best combined referral bonus - the total of what "
                 "I get plus what she gets?"),
    "task_099": ("My friend is starting a new business and will deposit about $30,000. I want to "
                 "refer her. Which account gives me the biggest referral bonus?"),
    "task_100": ("My former business partner just incorporated her LLC and will deposit about "
                 "$31,000. I want to refer her so I get the biggest possible referral bonus."),
}


def ask(prompt, choices, temp=0.0, mx=24):
    body = {"model": MODEL, "temperature": temp, "max_tokens": mx,
            "messages": [{"role": "user", "content": prompt}]}
    if choices:
        body["guided_choice"] = list(choices)
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return " ".join((json.load(r)["choices"][0]["message"]["content"] or "").split())


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 8
    a2 = load_domain_a2("banking_knowledge")
    sp = next(x for x in a2["ledger_metrics"] if x.get("eligible_text"))
    cfg = sp["eligible"]
    rows = (a2.get("policy_ontology") or {}).get("rows") or []
    axes_all = ((a2.get("policy_ontology") or {}).get("axes") or {})
    kb = LG.subject_kinds(rows, cfg.get("kind_field") or "kind")
    maps0 = {ax: FD._a3_map(rows, {"axis": ax}) for ax in (cfg.get("show_axes") or [])}
    tpl, reask = sp["rederive_prompt"], sp.get("reask_prompt")
    noclause = tpl.split("The customer says:")[0].rstrip() + (
        "\n\nAnswer with one name copied exactly from the list above, and nothing else.")
    print("reask_prompt 선언: %s" % ("있음" if reask else "**없음 — D1c 불가**"))
    out = {}
    for task, c in CASE.items():
        maps, _d = LG.restrict_to_kind(maps0, kb, c["kind"])
        tbl = (LG.eligible_text(c["days"], c.get("tally") or {}, maps, sp, c["stated"]) or "").strip()
        erows = LG.eligible_text(c["days"], c.get("tally") or {}, maps, sp, c["stated"],
                                 as_rows=True) or []
        names = [s for s, _b in erows]
        facts = "\n".join(["days since the earliest account was opened = %d" % c["days"]]
                          + ["%s = %s" % (k, LG._num(v)) for k, v in sorted(c["stated"].items())])
        print("\n%s  표 %d행: %s   gold=%r" % (task, len(names), ", ".join(names), c["gold"]))

        # 목적 축을 **손님 말에서** 형식화한다 (엔진과 같은 방식·집합 검사만)
        axlist = "\n".join("  %s — %s" % (k, axes_all[k]) for k in sorted(axes_all))
        axraw = ask((sp.get("objective_axis_prompt") or "").format(axes=axlist, text=SAID[task]),
                    sorted(axes_all) + ["NONE"], 0.0)
        oax = axraw if axraw in axes_all else None
        omap = (maps or {}).get(oax) or {}
        print("  목적 축 형식화: %r%s" % (axraw, "" if oax else "  ← 집합 밖 = D1c 건너뜀"))

        for arm, base in (("CUR", tpl), ("NOCLAUSE", noclause)):
            first, final = collections.Counter(), collections.Counter()
            for i in range(n):
                t = 0.0 if i == 0 else 0.7
                p0 = (base.format(table=tbl, facts=facts, asked="") if arm == "CUR"
                      else base.format(table=tbl, facts=facts))
                try:
                    pick = ask(p0, names + ["NONE"], t)
                except Exception as e:
                    pick = "ERR %s" % type(e).__name__
                first[pick] += 1
                # ── D1c: 엔진과 **같은 조건** — 첫 지목이 있어야 돈다
                fin = pick
                if pick in names and oax and reask:
                    mm = LG.mismatch_value(erows, omap, pick)
                    if mm:
                        p2 = p0 + "\n\n" + reask.format(axis=oax, chosen=LG._num(mm[0]),
                                                        best=LG._num(mm[1]))
                        try:
                            fin = ask(p2, names + ["NONE"], t)
                        except Exception as e:
                            fin = "ERR %s" % type(e).__name__
                final[fin] += 1
            h1 = sum(v for k, v in first.items() if str(k).strip() == c["gold"])
            h2 = sum(v for k, v in final.items() if str(k).strip() == c["gold"])
            out["%s/%s" % (arm, task)] = [h1, h2, n]
            print("  %-9s 1단 %d/%d %-28s → 2단 %d/%d %s"
                  % (arm, h1, n, first.most_common(1), h2, n, final.most_common(1)))
    json.dump(out, open(os.environ.get("T2_X208_OUT", "x208_out.json"), "w"), indent=1)
    print("\n※ 조항을 빼는 것의 값어치는 1단 정확도가 아니라 **D1c 가 돌 기회를 얻는가** 다."
          "\n  2단이 098 을 살리고 099·100 을 안 죽이면 그때 문구를 고친다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
