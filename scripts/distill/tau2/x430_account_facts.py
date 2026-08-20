# -*- coding: utf-8 -*-
r"""x430 — **계좌 클래스 사실표를 정책 축자로 저작** (사용자 지시 2026-08-20 · C462 전범)

## 왜
x428·x429 로 *"상한은 formalize 없이 못 잰다"* 까지 왔다. C462(`get_atm_fee_discrepancies`)가 밟은
길이 유일하게 정직한 길이다 — **정책 축자 추출(에이전트·gold 미접촉) → 그 위에서 판정**.

여기서는 **checking 계좌 클래스 전체 × 속성 전체**를 문서에서 뽑는다. 표적 태스크가 무엇을 필요로
하는지에 따라 속성을 고르지 않는다(고르면 gold-fit 이 된다·[[23]]). 뽑는 규칙은 하나 —
**문서가 값을 명시한 속성이면 전부 싣는다.**

## 규율
★값마다 **문서 id + 축자 문장**을 함께 싣는다. 못 대면 그 칸은 비운다(추정 금지).
★한 속성에 **서로 다른 값**이 여러 문서에서 나오면 충돌로 표시하고 둘 다 남긴다(우리가 고르지 않는다).
★gold 를 보지 않는다. 이 스크립트는 `sim_results` 를 읽지 않는다 — 문서 디렉터리만 읽는다.

사용: py -3 x430_account_facts.py [--docdir ...] [--family checking_accounts]
"""
import argparse
import collections
import io
import json
import os
import re
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

DOCDIR = "/home/woori/scratch/tau2-bench/data/tau2/domains/banking_knowledge/documents"

# ★계좌 태스크는 계열을 넘나든다(2026-08-20 실측): 063 의 gold `Silver Plus Account` 는
#   `business_savings_accounts_silver_plus_saver_*` 다. checking 만 담은 표로는 **후보에 없어서**
#   판정 자체가 불가능했다 — 모델 실패로 읽으면 오진이다([[55]]).
FAMILIES = ["checking_accounts", "savings_accounts",
            "business_checking_accounts", "business_savings_accounts"]

# 속성 = (이름, 그 속성을 부르는 축자 표현들). 값은 문장 안의 첫 통화·퍼센트·정수.
ATTRS = [
    ("monthly_maintenance_fee", ["monthly maintenance fee", "monthly maintenance"]),
    ("waiver_min_daily_balance", ["minimum daily balance", "waiver requirement", "minimum balance to waive"]),
    ("overdraft_fee", ["overdraft fee", "overdraft fees", "overdrafts"]),
    ("overdraft_protection_transfer_fee", ["overdraft protection transfer"]),
    ("oon_atm_fee", ["out-of-network atm", "out of network atm", "foreign atm"]),
    ("free_oon_atm_per_month", ["free out-of-network atm", "free out of network atm"]),
    ("apy", ["apy"]),
    ("daily_atm_withdrawal_limit", ["daily atm withdrawal limit", "atm withdrawal limit"]),
    ("min_age", ["minimum primary holder age"]),
    ("max_age", ["maximum primary holder age"]),
    ("monthly_deposit_limit", ["direct deposits up to", "mobile check deposit daily limit"]),
    # ★G7(2026-08-20) — 표 결손 보강. **손님이 실제로 물은 속성만** 더한다(출처=손님 발화·[[23]]).
    #   055 스펙이 `foreign_transaction_fee`·`currency_holding` 을 요구했는데 표에 없어 gold 가 탈락했다
    #   — 모델이 아니라 우리 표의 결손이었다([[55]]). 063 은 종이명세서를 물었다.
    # ⚠`overdraft_coverage` 는 뺐다(2026-08-20): `overdraft_protection_transfer_fee` 와 같은 개념인데
    #   전자는 문서가 **값을 주지 않아** 늘 비어 있었고, 형식화가 그 빈 칸을 고르면 `exists` 가 오작동해
    #   057 t1 에서 gold(blue·이체료 $12.50 기재)가 죽었다. 한 개념은 **값이 있는 칸 하나**로만 둔다.
    ("foreign_transaction_fee", ["foreign transaction fee", "international transaction fee"]),
    ("oon_atm_reimbursement", ["atm fee reimbursement", "reimburse", "rebate"]),
    ("foreign_currency_holding", ["foreign currency", "multi-currency", "hold currency"]),
    ("paper_statement_fee", ["paper statement"]),
    ("wire_transfer_fee", ["wire transfer fee", "outgoing wire"]),
]
def _attrs_from_a2():
    """★속성 목록은 **A2 구조 선언**(`catalog_attrs`)에서 읽는다(2026-08-20·사용자 지적·[[24]]).

    전에는 이 파일 안에 내가 지은 16개가 박혀 있었다 — 정본 층이 아니라 스크립트에 든 도메인 지식은
    전이도 안 되고 갈라진다([[05]]·[[67]]). A2 가 없으면 아래 내장 목록으로 폴백한다(오프라인 실행용).
    """
    try:
        p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "a2",
                         "banking_knowledge.specific.json")
        with io.open(p, encoding="utf-8") as f:
            d = json.load(f).get("catalog_attrs") or {}
        got = [(k, v.get("aliases") or []) for k, v in d.items() if v.get("aliases")]
        return got or None
    except Exception:
        return None


RE_VAL = re.compile(r"(\$\s?\d[\d,]*(?:\.\d+)?|\d+(?:\.\d+)?\s?%|\bnone\b|\bno\b|\bunlimited\b|\b\d+\b)", re.I)


def sentences(txt):
    return [" ".join(s.split()) for s in re.split(r"(?<=[.!?])\s+|\n", txt) if s.strip()]


def norm_val(v):
    v = v.strip()
    if v.lower() in ("none", "no"):
        return "0"
    return v.replace(" ", "")


def llm_table(files, docdir, family, port, model):
    """★C410 전범: **LLM 이 형식화하고 엔진은 인용 실재만 검산한다**([[10]]).

    정규식 판본은 `No overdraft fees` 처럼 **부정이 속성어 앞에 오는** 형태를 계통적으로 놓쳤다
    (하필 057 의 결정 속성이다). 엔진이 도메인 문장을 더 뜯는 것은 [[59]] 위반 방향이므로,
    형식화는 LLM 에게 넘기고 엔진은 **① 인용이 문서에 축자로 있나 ② 값이 그 인용 안에 있나**만 본다.
    검산에 걸리면 그 칸은 **버린다**(지어낸 값이 표에 남는 것이 최악이다·[[25]]).
    """
    import urllib.request
    prefix = "doc_%s_" % family
    byc = collections.defaultdict(list)
    for f in files:
        cls = re.sub(r"_\d+\.json$", "", f[len(prefix):])
        with io.open(os.path.join(docdir, f), encoding="utf-8") as fh:
            d = json.load(fh)
        byc[cls].append((f.replace(".json", ""), (d.get("title") or "") + ". " + (d.get("content") or "")))
    names = [x[0] for x in ATTRS]
    # ★단위를 함께 싣는다(2026-08-20): `1% of withdrawal amount (max $3.00)` 와 `$2.50` 은 **비교 불가**다.
    #   엔진이 사용량으로 환산하려면 unit·cap 이 있어야 한다 — 그 둘도 문서가 말한 것만 받는다.
    sysmsg = ("You extract documented facts. Reply with ONE JSON object only: "
              "{\"<attribute>\": {\"value\": \"<verbatim value>\", \"unit\": \"USD|percent|count|text\", "
              "\"cap\": <number or null>, \"quote\": \"<verbatim sentence from the documents "
              "containing that value>\"}}. 'cap' is a documented maximum for that fee, else null. "
              "Omit any attribute the documents do not state. Never paraphrase; the quote must "
              "appear character-for-character.")
    out = {}
    for cls in sorted(byc):
        blob = "\n\n".join("### %s\n%s" % (i, t) for i, t in byc[cls])[:40000]
        body = ("# Documents for the %s\n%s\n\n# Attributes to extract\n%s\n"
                % (cls, blob, ", ".join(names)))
        req = urllib.request.Request(
            "http://127.0.0.1:%d/v1/chat/completions" % port,
            data=json.dumps({"model": model, "temperature": 0.0, "max_tokens": 1200,
                             "messages": [{"role": "system", "content": sysmsg},
                                          {"role": "user", "content": body}]}).encode("utf-8"),
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                raw = json.loads(r.read().decode("utf-8"))["choices"][0]["message"]["content"]
        except Exception as e:
            print("  ERROR %s: %r" % (cls, e))
            continue
        i, j = raw.find("{"), raw.rfind("}")
        try:
            got = json.loads(raw[i:j + 1]) if i >= 0 and j > i else {}
        except Exception:
            got = {}
        docs_join = " ".join(" ".join(t.split()) for _i, t in byc[cls])
        row, kept, dropped = {}, 0, 0
        for k, v in (got.items() if isinstance(got, dict) else []):
            if k not in names or not isinstance(v, dict):
                continue
            val, q = str(v.get("value", "")).strip(), " ".join(str(v.get("quote", "")).split())
            if not val or not q or q not in docs_join or val.replace(" ", "") not in q.replace(" ", ""):
                dropped += 1
                continue
            unit = str(v.get("unit") or "").strip()
            cap = v.get("cap")
            try:
                cap = float(cap) if cap is not None else None
            except Exception:
                cap = None
            row[k] = {"values": [val], "conflict": False,
                      "unit": unit if unit in ("USD", "percent", "count", "text") else "",
                      "cap": cap,
                      "evidence": [{"value": val, "doc": cls, "quote": q[:220]}]}
            kept += 1
        for k in names:
            row.setdefault(k, {"values": [], "conflict": False, "evidence": []})
        out[cls] = row
        print("  %-26s 채택 %2d · 검산탈락 %2d" % (cls[:26], kept, dropped))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docdir", default=DOCDIR)
    ap.add_argument("--family", default="checking_accounts")
    ap.add_argument("--out", default=None)
    ap.add_argument("--llm", action="store_true", help="C410 전범: LLM 형식화 + 엔진 인용 검산")
    ap.add_argument("--port", type=int, default=8141)
    ap.add_argument("--model", default="Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8")
    a = ap.parse_args()
    if a.llm:
        fams = FAMILIES if a.family == "all" else [a.family]
        out, nfiles = {}, 0
        for fam in fams:
            pre = "doc_%s_" % fam
            files = sorted(f for f in os.listdir(a.docdir) if f.startswith(pre) and f.endswith(".json"))
            nfiles += len(files)
            print("x430(LLM) · %s · 문서 %d" % (fam, len(files)))
            part = llm_table(files, a.docdir, fam, a.port, a.model)
            for k, v in part.items():
                v["_family"] = fam
                out[k if k not in out else "%s@%s" % (k, fam)] = v
        print("— 인용 검산 통과분만 · 클래스 %d · 문서 %d" % (len(out), nfiles))
        attrs = [x[0] for x in ATTRS]
        print("\n%-26s %s" % ("class", " ".join("%-13s" % x[:13] for x in attrs[:6])))
        for cls in sorted(out):
            print("%-26s %s" % (cls[:26], " ".join("%-13s" % ("|".join(out[cls][k]["values"])[:13] or "-")
                                                   for k in attrs[:6])))
        p = a.out or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..",
                                  "reports", "facet_rft_2026", "x430_account_facts_llm.json")
        with io.open(os.path.abspath(p), "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=1)
        print("\n→ %s" % os.path.abspath(p))
        return 0

    pre = "doc_%s_" % a.family
    files = sorted(f for f in os.listdir(a.docdir) if f.startswith(pre) and f.endswith(".json"))
    table = collections.defaultdict(lambda: collections.defaultdict(list))
    for f in files:
        cls = re.sub(r"_\d+\.json$", "", f[len(pre):])
        with io.open(os.path.join(a.docdir, f), encoding="utf-8") as fh:
            d = json.load(fh)
        txt = (d.get("title") or "") + ". " + (d.get("content") or "")
        for s in sentences(txt):
            low = s.lower()
            for attr, keys in ATTRS:
                if not any(k in low for k in keys):
                    continue
                # ★값은 속성어 **바로 뒤**에서만 읽는다(2026-08-20 수리): 창을 문장 전체로 두면
                #   *"a maximum of 3 overdraft fees can be charged"* 의 3 이 overdraft_fee 로 들어오고
                #   *"waived when your minimum daily balance is $1,350"* 의 1,350 이 월 수수료로 들어온다.
                #   허용 형태 = `속성: 값` · `| 속성 | 값 |` · `속성 is/of 값` 뿐이고 창은 40자다.
                k0 = min((k for k in keys if k in low), key=lambda k: low.find(k))
                pos = low.find(k0) + len(k0)
                seg = s[pos:pos + 40]
                if not re.match(r"\s*(?:[:|=]|\bis\b|\bof\b|\bper\b|\bat\b|\s)+", seg):
                    continue
                m = RE_VAL.search(seg)
                if not m:
                    continue
                table[cls][attr].append({"value": norm_val(m.group(1)),
                                         "doc": f.replace(".json", ""), "quote": s[:220]})
    print("=" * 108)
    print("x430 · %s · 클래스 %d · 문서 %d" % (a.family, len(table), len(files)))
    print("=" * 108)
    attrs = [x[0] for x in ATTRS]
    print("%-26s %s" % ("class", " ".join("%-12s" % x[:12] for x in attrs[:6])))
    out = {}
    for cls in sorted(table):
        row, cells = {}, []
        for attr in attrs:
            vals = table[cls].get(attr) or []
            uniq = sorted({v["value"] for v in vals})
            row[attr] = {"values": uniq, "conflict": len(uniq) > 1,
                         "evidence": vals[:4]}
            cells.append(("|".join(uniq)[:12]) if uniq else "-")
        out[cls] = row
        print("%-26s %s" % (cls[:26], " ".join("%-12s" % c for c in cells[:6])))
    conf = [(c, k) for c, r in out.items() for k, v in r.items() if v["conflict"]]
    print("\n충돌(한 속성에 값 2개 이상) %d건 — 우리가 고르지 않고 둘 다 남긴다" % len(conf))
    for c, k in conf[:12]:
        print("   %-24s %-34s %s" % (c[:24], k, "|".join(out[c][k]["values"])[:40]))
    miss = [(c, k) for c, r in out.items() for k, v in r.items() if not v["values"]]
    print("\n미기재(문서가 값을 안 준 칸) %d" % len(miss))
    p = a.out or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..",
                              "reports", "facet_rft_2026", "x430_account_facts.json")
    with io.open(os.path.abspath(p), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("\n→ %s" % os.path.abspath(p))
    return 0


if __name__ == "__main__":
    sys.exit(main())
