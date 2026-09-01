# -*- coding: utf-8 -*-
"""x730 — **날짜는 접지가 아니라 formalize 다** (2026-09-01·사용자 지적).

⛔무엇이 잘못됐었나: 나는 *"gold 날짜가 손님 발화에 축자로 있나"* 를 **정규식으로** 재고
  42%(좁히면 38.7%)가 없다며 레버를 기각했다. 그 측정 자체가 **패턴 매칭**이라 [[59]] 위반이고,
  formalize 물음에 접지 계기를 갖다 댄 것이다. 손님은 *"Nov 7"* 처럼 말한다(082 msg32 실물).

이 프로브가 재는 것: **손님 발화만** 준 서브가 분쟁 건별 `discovery_date` 를 **형식화**해 낼 수 있나.
  · 서브는 gold 를 보지 않는다([[23]]). 대조는 프로브가 사후에 한다.
  · 엔진은 **비교만** 한다 — 날짜를 만들지 않는다([[10]]/[[52]]).
판정: 정확 일치율. 높으면 formalize 배선이 정당하고, 낮으면 재료가 실제로 없는 것이다([[78]]).

사용: x730_date_formalize_probe.py <base_url> <model> [태스크수]
"""
import json
import os
import re
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import t2_forensic as F

TOOL = "file_debit_card_transaction_dispute_6281"
ARG = "discovery_date"
Q = ("You are given only the customer's messages from a bank support chat. "
     "For EACH transaction the customer disputes, output the date the customer says they "
     "NOTICED/DISCOVERED it, as MM/DD/YYYY. Use only what the customer states or implies "
     "(relative expressions count). If the customer does not indicate it, output null.\n"
     'Answer as JSON: {"items":[{"merchant":"...","amount":"...","discovery_date":"MM/DD/YYYY|null"}]}')


def ask(base, model, text, timeout=180):
    body = {"model": model, "temperature": 0.0, "max_tokens": 700,
            "messages": [{"role": "user", "content": text}],
            "response_format": {"type": "json_schema", "json_schema": {"name": "d", "schema": {
                "type": "object", "properties": {"items": {"type": "array", "items": {
                    "type": "object",
                    "properties": {"merchant": {"type": "string"}, "amount": {"type": "string"},
                                   "discovery_date": {"type": ["string", "null"]}},
                    "required": ["discovery_date"]}}}, "required": ["items"]}}}}
    req = urllib.request.Request(base.rstrip("/") + "/chat/completions",
                                 data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.load(r)
    return ((d.get("choices") or [{}])[0] or {}).get("message", {}).get("content") or ""


def main():
    base = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8141/v1"
    model = sys.argv[2] if len(sys.argv) > 2 else "Qwen/Qwen3.8-27B-FP8"
    want = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    seen, rows = set(), []
    for tg, s in F.iter_all_sims():
        ti = s.get("task_id")
        if ti in seen:
            continue
        try:
            d = F.mutation_diff(s, tag=tg)
        except Exception:
            continue
        gold = []
        for r in (d.get("gold") or []):
            n, _, a = str(r.get("key")).partition("|")
            if TOOL not in n:
                continue
            try:
                a = json.loads(a)
            except Exception:
                continue
            if ARG in a:
                gold.append(a)
        if not gold:
            continue
        seen.add(ti)
        rows.append((ti, s, gold))
        if len(rows) >= want:
            break

    hit = tot = 0
    for ti, s, gold in rows:
        utter = "\n".join("CUSTOMER: " + str(m.get("content") or "")
                          for m in (s.get("messages") or []) if m.get("role") == "user")
        try:
            out = ask(base, model, Q + "\n\n=== customer messages ===\n" + utter[:12000])
            got = json.loads(re.search(r"\{.*\}", out, re.S).group(0)).get("items") or []
        except Exception as e:
            print("  %-9s 서브 실패: %r" % (ti, e))
            continue
        golds = [str(g.get(ARG)) for g in gold]
        subs = [str(i.get("discovery_date")) for i in got]
        for g in golds:
            tot += 1
            if g in subs:
                hit += 1
        print("  %-9s gold=%s · sub=%s" % (ti, golds, subs))
    print("정확 일치 %d/%d (%.0f%%)" % (hit, tot, 100.0 * hit / max(tot, 1)))


if __name__ == "__main__":
    main()
