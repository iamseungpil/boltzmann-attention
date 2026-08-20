# -*- coding: utf-8 -*-
r"""x431 — **스펙이 gold 를 고르나**: 빈칸 사유 확정 → 제약 형식화 → 결정론 필터 (사용자 지시 2026-08-20)

## 위치 ([[62]] 절차)
    ① 사실표 저작        x430 (LLM 형식화 + 엔진 인용 검산) — 1차 완료
    ①b 빈칸 사유 확정    여기 · 문서 미기재인가 추출 실패인가
    ② 제약 형식화        여기 · **손님 발화 축자만** 보고 닫힌 술어로 (LLM)
    ③ 결정론 필터        여기 · 엔진이 표에 술어를 적용 (판단 0·산술과 비교뿐)
    ④ 판정              스펙이 gold 를 고르나 · 못 고르면 그 태스크는 **미결정**

## 규율
★②의 입력은 **손님 발화뿐**이다. 문서도 gold 도 안 넣는다 — 제약은 손님이 말한 것이지
  정답에서 역산한 것이 아니어야 한다([[23]]).
★③의 엔진은 **비교만** 한다(==·<=·>=·존재). 어느 속성을 볼지는 ②가 정한다([[10]]).
★gold 는 ④ 채점에만 등장한다.
★스펙이 못 고르면 그것은 **모델의 실패가 아니라 우리 표·스펙의 한계**로 먼저 읽는다([[55]]).

사용: py -3 x431_spec_selects.py [--port 8141] [--fill-blanks]
"""
import argparse
import collections
import io
import json
import os
import re
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import x423_choice_isolation as I  # noqa: E402
import x426_free_gates as G  # noqa: E402
import x427_catalog_minimal as CM  # noqa: E402
import x430_account_facts as FT  # noqa: E402

MODEL = "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8"
TBL = os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026", "x430_account_facts_llm.json")
ATTRS = [x[0] for x in FT.ATTRS]
RE_MONEY = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def ask(port, sysmsg, body, maxtok=700):
    req = urllib.request.Request(
        "http://127.0.0.1:%d/v1/chat/completions" % port,
        data=json.dumps({"model": MODEL, "temperature": 0.0, "max_tokens": maxtok,
                         "messages": [{"role": "system", "content": sysmsg},
                                      {"role": "user", "content": body}]}).encode("utf-8"),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        raw = json.loads(r.read().decode("utf-8"))["choices"][0]["message"]["content"]
    i, j = raw.find("{"), raw.rfind("}")
    try:
        return json.loads(raw[i:j + 1]) if i >= 0 and j > i else {}
    except Exception:
        return {}


def num(v):
    """'$0.00' · 'None' · '1% of withdrawal' → 수. 못 읽으면 None(=비교 불가로 남긴다)."""
    s = str(v).strip().lower()
    if s in ("none", "no", "n/a", "-", ""):
        return 0.0
    m = RE_MONEY.search(s.replace(",", ""))
    return float(m.group(0)) if m else None


def fill_blanks(table, docdir, family, port):
    """빈칸마다 **표적 질의** — 문서가 값을 주나. 안 주면 `absent` 로 확정한다."""
    prefix = "doc_%s_" % family
    byc = collections.defaultdict(list)
    for f in sorted(os.listdir(docdir)):
        if not (f.startswith(prefix) and f.endswith(".json")):
            continue
        cls = re.sub(r"_\d+\.json$", "", f[len(prefix):])
        with io.open(os.path.join(docdir, f), encoding="utf-8") as fh:
            d = json.load(fh)
        byc[cls].append((f.replace(".json", ""), (d.get("title") or "") + ". " + (d.get("content") or "")))
    sysmsg = ("Answer ONLY from the documents. Reply with ONE JSON object: "
              "{\"value\": \"<verbatim>\", \"quote\": \"<verbatim sentence>\"} if the documents state it, "
              "otherwise {\"absent\": true}. Never paraphrase.")
    filled = absent = dropped = 0
    for cls, row in table.items():
        docs = byc.get(cls) or []
        if not docs:
            continue
        joined = " ".join(" ".join(t.split()) for _i, t in docs)
        blob = "\n\n".join("### %s\n%s" % (i, t) for i, t in docs)[:40000]
        for attr in ATTRS:
            if row.get(attr, {}).get("values"):
                continue
            got = ask(port, sysmsg, "# Documents for the %s\n%s\n\n# Question\nWhat is the %s?\n"
                      % (cls, blob, attr.replace("_", " ")), maxtok=300)
            if got.get("absent"):
                row[attr] = {"values": [], "conflict": False, "evidence": [], "absent": True}
                absent += 1
                continue
            val, q = str(got.get("value", "")).strip(), " ".join(str(got.get("quote", "")).split())
            if val and q and q in joined and val.replace(" ", "") in q.replace(" ", ""):
                row[attr] = {"values": [val], "conflict": False,
                             "evidence": [{"value": val, "doc": cls, "quote": q[:220]}]}
                filled += 1
            else:
                row[attr] = {"values": [], "conflict": False, "evidence": [], "unresolved": True}
                dropped += 1
    print("빈칸 처리 — 채움 %d · 문서 미기재 확정 %d · 검산 탈락(미해결) %d" % (filled, absent, dropped))
    return table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8141)
    ap.add_argument("--docdir", default=FT.DOCDIR)
    ap.add_argument("--family", default="checking_accounts")
    ap.add_argument("--fill-blanks", action="store_true")
    a = ap.parse_args()

    with io.open(os.path.abspath(TBL), encoding="utf-8") as f:
        table = json.load(f)
    if a.fill_blanks:
        table = fill_blanks(table, a.docdir, a.family, a.port)
        with io.open(os.path.abspath(TBL).replace(".json", "_filled.json"), "w", encoding="utf-8") as f:
            json.dump(table, f, ensure_ascii=False, indent=1)

    # ② 제약 형식화 — 손님 발화만 본다
    seen, cs = set(), []
    for c in I.cases(60):
        if c["arg"] != "account_class":
            continue
        k = (c["task"], c["trial"])
        if k in seen:
            continue
        seen.add(k)
        cs.append(c)
    sysmsg = ("You turn a customer's stated requirements into a machine-checkable spec. "
              "Reply with ONE JSON object only: {\"constraints\": [{\"attribute\": \"<one of the given "
              "names>\", \"op\": \"==|<=|>=|exists|absent\", \"value\": <number or null>, "
              "\"because\": \"<the customer's own words>\"}]}. Use ONLY requirements the customer "
              "actually stated. If they stated none for an attribute, omit it.")
    print("\n=== ②③④ 스펙 → 필터 → 판정 (사례 %d) ===" % len(cs))
    rows = []
    for c in cs:
        said = G.customer_said(c["sim"], c["msg_i"])
        spec = ask(a.port, sysmsg, "# Customer's own words\n%s\n\n# Attribute names you may use\n%s\n"
                   % (said[:6000], ", ".join(ATTRS)))
        cons = [x for x in (spec.get("constraints") or []) if isinstance(x, dict)]
        surv, why = [], []
        for cls, row in table.items():
            ok = True
            for con in cons:
                at, op = con.get("attribute"), con.get("op")
                if at not in ATTRS:
                    continue
                vals = (row.get(at) or {}).get("values") or []
                if op == "exists":
                    ok = ok and bool(vals)
                    continue
                if op == "absent":
                    ok = ok and not vals
                    continue
                if not vals:
                    continue                      # 미기재는 **거르지 않는다**(과차단 방지·C462)
                v, t = num(vals[0]), con.get("value")
                if v is None or t is None:
                    continue
                t = float(t)
                ok = ok and ((v == t) if op == "==" else (v <= t) if op == "<=" else
                             (v >= t) if op == ">=" else True)
            if ok:
                surv.append(cls)
        gold_key = c["gold"].lower().replace(" ", "_").replace("-", "-")
        hit = [s for s in surv if s.replace("_", " ").replace("(checking)", "").strip()
               == c["gold"].lower().replace("account", "account").strip()
               or s.startswith(c["gold"].lower().split()[0])]
        rows.append({"task": c["task"], "trial": c["trial"], "gold": c["gold"],
                     "n_constraints": len(cons), "n_surv": len(surv), "surv": surv,
                     "gold_in": bool(hit), "unique": len(surv) == 1 and bool(hit),
                     "spec": cons})
        print("  %-9s t%s gold=%-22s 제약 %d개 → 생존 %2d %s%s"
              % (c["task"], c["trial"], c["gold"][:22], len(cons), len(surv),
                 "· gold 포함" if hit else "· ⛔gold 탈락",
                 " ✅유일" if (len(surv) == 1 and hit) else ""))
        for con in cons[:4]:
            print("        %-34s %-6s %-8s ← %s" % (con.get("attribute"), con.get("op"),
                                                    con.get("value"), str(con.get("because"))[:60]))
    n = len(rows)
    if n:
        print("\n  ★스펙이 gold 를 남긴 사례 **%d/%d** · 유일하게 고른 사례 **%d/%d**"
              % (sum(x["gold_in"] for x in rows), n, sum(x["unique"] for x in rows), n))
    p = os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026", "x431_spec_selects.json")
    with io.open(os.path.abspath(p), "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)
    print("\n→ %s" % os.path.abspath(p))
    return 0


if __name__ == "__main__":
    sys.exit(main())
