# -*- coding: utf-8 -*-
r"""x188 — **사실을 LLM이 어디까지 정확히 만드는가** (유료 0·사용자 지시·상류 검정).

설계서: `reports/facet_rft_2026/ISOLATION_GRID_DESIGN_2026_08_09.md` §3.

`formalize_case_facts`(t2_ledger.py:232)가 실제로 무엇을 내는지를 **A2 템플릿 그대로** 재현해
잰다. 요청 항목은 `eligible.criteria` 중 `operand=="stated"` 인 축 + `policy_ontology.axes`
설명이다 — 엔진·프로브에 도메인 어휘를 새로 쓰지 않는다.

## 채점은 gold 를 안 본다 (전부 닫힌 술어·[[22]])

  1) **키 준수**   요청한 축 이름만 왔는가 (지어낸 키 = 0 이어야 한다)
  2) **인용 실재** `quote` 가 대화에 축자로 있는가
  3) **값 정합**   `value` 가 그 인용 안의 수와 맞는가

⚠ (2)(3)이 통과해도 *"옳은 문장을 골랐는가"* 는 열린 술어라 여기서 안 잰다. 대신 값을 전부
   찍어 사람이 본다. 이것이 이 프로브의 한계다.

## 문맥 사다리

  full       전 궤적
  last10     마지막 10 메시지
  user_only  손님 발화만  (형식화에 필요한 최소·[[18]] 정보-맞춤의 하한 후보)

실행: python x188_fact_accuracy.py [N]
"""
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

import x150_choice_ablation as Y                               # noqa: E402
from gate_interpreter import load_domain_a2                     # noqa: E402

URL = os.environ.get("T2_PROBE_URL", "http://localhost:8140/v1/chat/completions")
MODEL = os.environ.get("T2_PROBE_MODEL", "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8")
TAG = os.environ.get("T2_PROBE_TAG", "bank_elig_20260809i")
TASKS = ("task_099", "task_100")


def ask(prompt, temp):
    body = json.dumps({"model": MODEL, "temperature": temp, "max_tokens": 400,
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return (json.load(r)["choices"][0]["message"]["content"] or "").strip()


def norm(s):
    return " ".join(str(s or "").split()).lower()


def parse_obj(txt):
    m = re.search(r"\{.*\}", txt or "", re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    a2 = load_domain_a2("banking_knowledge")
    spec = next(s for s in a2["ledger_metrics"] if s.get("case_facts_prompt"))
    tpl = spec["case_facts_prompt"]
    crit = (spec.get("eligible") or {}).get("criteria") or []
    axd = ((a2.get("policy_ontology") or {}).get("axes") or {})
    wanted = [(c["axis"], axd[c["axis"]]) for c in crit
              if c.get("operand") == "stated" and axd.get(c.get("axis"))]
    if not wanted:
        print("요청 항목 0 — A2 에 stated 기준이나 축 설명이 없다. 중단.")
        return 1
    names = [k for k, _ in wanted]
    items = ("\n".join("  %s  — %s" % (k, v) for k, v in wanted)
             + "\n\nFill in this exact skeleton, dropping any item the conversation does not "
               "state:\n{"
             + ", ".join('"%s": {"value": <integer>, "quote": "<exact sentence>"}' % k
                         for k in names)
             + "}\nDo not invent other keys. Do not report anything that is not in this list.")

    print("model=%s · n=%d · 요청 축 %s" % (MODEL, n, names))
    out = []
    for task in TASKS:
        ms = Y.msgs_of(TAG, task)
        ladders = [("full     ", ms),
                   ("last10   ", ms[-10:]),
                   ("user_only", [m for m in ms if m.get("role") == "user"])]
        print("\n" + "=" * 100)
        print("%s" % task)
        print("=" * 100)
        for label, sub in ladders:
            text = Y.render(sub)
            hay = norm(text)
            prompt = tpl.replace("{items}", items).replace("{text}", text)
            key_ok = quote_ok = val_ok = 0
            got, extra = [], set()
            for i in range(n):
                try:
                    obj = parse_obj(ask(prompt, 0.0 if i == 0 else 0.7))
                except Exception as e:
                    obj = None
                    print("   ERR %r" % (e,))
                if obj is None:
                    got.append("PARSE_FAIL")
                    continue
                bad = [k for k in obj if k not in names]
                extra |= set(bad)
                key_ok += 1 if not bad else 0
                rec = {}
                for k in names:
                    v = obj.get(k)
                    if not isinstance(v, dict):
                        continue
                    q, val = v.get("quote"), v.get("value")
                    inctx = norm(q) in hay if q else False
                    nums = re.findall(r"\d[\d,]*", str(q or "").replace(",", ""))
                    match = str(val) in [x.replace(",", "") for x in nums]
                    rec[k] = (val, "Q✓" if inctx else "Q✗", "V✓" if match else "V✗")
                    quote_ok += 1 if inctx else 0
                    val_ok += 1 if match else 0
                got.append(rec)
            print("  %-9s (%5d자) 키준수 %d/%d · 인용실재 %d · 값정합 %d %s"
                  % (label, len(text), key_ok, n, quote_ok, val_ok,
                     ("· 지어낸 키 %s" % sorted(extra)) if extra else ""))
            for r in got[:3]:
                print("      %s" % (r,))
            out.append({"task": task, "ctx": label.strip(), "n": n, "key_ok": key_ok,
                        "quote_ok": quote_ok, "val_ok": val_ok, "extra": sorted(extra),
                        "samples": [str(r) for r in got]})

    json.dump(out, open(os.environ.get("T2_X188_OUT", "x188_out.json"), "w"), indent=1)
    print("\n  (1)(2)(3) 중 하나라도 낮으면 상류가 새는 것 — 선택 격자 결론보다 먼저 고친다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
