# -*- coding: utf-8 -*-
r"""x183 — **마진 축**: 099 와 100 을 하나의 법칙으로 이을 수 있는가 (유료 0).

## 왜

x182 가 격리에서의 지배 변수를 **후보 오염(카드 혼입)** 으로 바꿔 놓았다. 카드를 후보에서
빼면(표는 불변) 099 는 10정렬 중 **9개가 8/8** 인데 100 은 여전히 갈린다(5/8·8/8·0/8 혼재).
남은 차이의 1순위 후보는 **정답의 마진**이다 — 통과 집합 안에서 gold 와 2위의 보너스 차:

  · 099 : `World Blue` 300 vs `Lime Green` 200        → **Δ = +100**
  · 100 : `Hunter Green` 175 vs `Cobalt Blue`/`Sky Blue` 150 → **Δ = +25**

그리고 대화가 붙으면 100 의 2위(`Cobalt Blue`)는 **에이전트 자신이 직전 턴에 약속한 이름**
이다(끝에서 189자·JSON 인자 템플릿). ⇒ *정박은 마진과 싸운다* 는 읽기가 가능하다.

## 축 (합성 통제 — 프로브 전용)

  Δ     gold 행의 `referrer_bonus_usd` 만 올려 2위와의 차를 10·25·50·100·150 으로 만든다.
        ⚠**정책값 날조가 아니라 명시적 합성 통제**다. A2·엔진·출하 경로엔 들어가지 않는다.
        나머지 행·이름·개수·후보목록·정렬은 전부 고정(이름순만 쓰므로 순서도 불변).
  ctx   facts (완전 격리) · full (실제 대화 = 정박 포함)
  sort  name_asc · name_desc
  후보  chk (카드 제외 — x182 가 그 축을 이미 분리했다)

## 읽는 법

  · 두 태스크가 **같은 Δ 문턱**에서 붙으면        → 공통 기전 = **마진 대비 잡음**, 태스크 차이는 Δ 하나로 환원.
  · 100 만 문턱이 높으면                          → Δ 외 잔여(정박 세기·능력)가 남는다.
  · `full` 문턱이 `facts` 보다 높으면             → **정박이 요구 마진을 올린다**(정량화 가능).
  · Δ 를 끝까지 올려도 `full`/100 이 안 붙으면    → 정박은 증거로 못 이긴다 = 격리기 필수.

실행: python x183_margin_vs_anchor.py [N]
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
TAG = os.environ.get("T2_PROBE_TAG", "bank_elig_20260809i")
CASE = {"task_099": {"days": 730, "deposit": 30000},
        "task_100": {"days": 65, "deposit": 31000}}
ACCOUNT_AXIS = "referrer_tenure_days"
DELTAS = [10, 25, 50, 100, 150]


def guided_full(prompt, choices, temp):
    body = json.dumps({"model": MODEL, "temperature": temp, "max_tokens": 12,
                       "guided_choice": list(choices),
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return " ".join((json.load(r)["choices"][0]["message"]["content"] or "").split())


def set_axis(line, axis, value):
    """우리가 만든 줄에서 **그 축 토큰만** 바꾼다 (구조적 치환·다른 필드 불변)."""
    subj, rest = line.split(":", 1)
    toks = [t.strip() for t in rest.split(",")]
    toks = [("%s=%d" % (axis, value)) if t.startswith(axis + "=") else t for t in toks]
    return "%s: %s" % (subj, ", ".join(toks))


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    a2 = load_domain_a2("banking_knowledge")
    spec = next(s for s in a2["ledger_metrics"] if s.get("eligible_text"))
    rows = a2["policy_ontology"]["rows"]
    axes = spec["eligible"]["show_axes"]
    maps = {ax: FD._a3_map(rows, {"axis": ax}) for ax in axes}
    bax = next(a for a in axes if "bonus" in a.lower() and "referrer" in a.lower())
    name = lambda l: l.strip().split(":")[0].strip()          # noqa: E731

    def bval(nm):
        v = (maps.get(bax) or {}).get(nm)
        try:
            return float(str(v[0]).replace(",", ""))
        except Exception:
            return -1.0

    print("model=%s · n=%d · 조작축=%s · Δ=%s" % (MODEL, n, bax, DELTAS))
    out = []
    for task, case in CASE.items():
        gold = X.GOLD[task]
        lines = LG.eligible_text(case["days"], {}, maps, spec,
                                 {"qualifying_deposit_usd": case["deposit"]}).strip().splitlines()
        head = [l for l in lines if not (l.startswith("  ") and ":" in l)]
        body = [l for l in lines if l.startswith("  ") and ":" in l]
        ALL = [name(l) for l in body]
        CHK = [s for s in ALL if s in (maps.get(ACCOUNT_AXIS) or {})]
        runner = max((s for s in CHK if s != gold), key=bval)
        base_delta = bval(gold) - bval(runner)
        ctxfull = Y.render(Y.msgs_of(TAG, task))

        print("\n" + "=" * 96)
        print("%s  gold=%r(%d) · chk 2위=%r(%d) · 실제 Δ=%+d"
              % (task, gold, bval(gold), runner, bval(runner), base_delta))
        print("=" * 96)
        print("  %-7s | %-19s | %-19s | %-19s | %s"
              % ("Δ", "facts/asc", "facts/desc", "full/asc", "full/desc"))
        for d in DELTAS:
            newv = int(bval(runner) + d)
            mod = [set_axis(l, bax, newv) if name(l) == gold else l for l in body]
            order_a = sorted(mod, key=name)
            order_d = sorted(mod, key=name, reverse=True)
            cells = []
            for ctx in ("facts", "full"):
                for order in (order_a, order_d):
                    tbl = "\n".join(head[:1] + order + head[1:]).strip()
                    prompt = tbl + "\n\n" + X.FACTS[task] + "\n\n" + X.QUESTION
                    if ctx == "full":
                        prompt = ("Here is a customer-service conversation so far.\n\n"
                                  + ctxfull + "\n\n" + prompt)
                    c = collections.Counter()
                    for i in range(n):
                        try:
                            c[guided_full(prompt, CHK, 0.0 if i == 0 else 0.7)] += 1
                        except Exception as e:
                            c["ERR %s" % type(e).__name__] += 1
                    g = c.get(gold, 0)
                    cells.append("%d/%d %-13s" % (g, n, c.most_common(1)[0][0][:13]))
                    out.append({"task": task, "delta": d, "gold_bonus": newv, "ctx": ctx,
                                "sort": "asc" if order is order_a else "desc",
                                "gold_hit": g, "n": n, "dist": dict(c)})
            print("  %+-6d | %s" % (d, " | ".join(cells)))

    json.dump(out, open(os.environ.get("T2_X183_OUT", "x183_out.json"), "w"), indent=1)
    print("\n  같은 Δ 문턱에서 두 태스크가 붙으면 공통 기전 = 마진 대비 잡음.")
    print("  full 문턱이 facts 보다 높으면 정박이 요구 마진을 올린다(정량화).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
