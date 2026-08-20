# -*- coding: utf-8 -*-
r"""x452 — **조건부 값을 담는 사실표 재추출** (2026-08-21·무료·G1 수리)

## 왜 (사용자 지시 *"표 고쳐서 격리 다시 재라"*)
`x451`(G1 격리)에서 사실표는 4 사례 중 **1** 만 고쳤다. 원인은 모델이 아니라 **표의 구멍**이다:
`silver_plus_account.apy` 가 **`absent`** 로 적혀 있는데 문서에는 축자로 *"APY tiers"* ·
*"Silver Plus offers tiered APY with two levels"* · *"What APY can I earn? — tiered APY"* 가 있다.
`x431.fill_blanks` 의 계약이 **단일 스칼라**(`{"value","quote"}` 하나)라 **계층 값이 담기지 않고**
모델이 `absent` 로 답했다. 표가 정답 후보의 값을 비워 두면 **표가 답을 정한다**([[25]]).

전수 상태(수리 전): 채움 346 · **absent 550** · APY 가 빈 클래스 **24/56**.

## 무엇을 바꾸나 — 계약 하나뿐
    이전  {"value": "<축자>", "quote": "<축자 문장>"}          ← 값 하나만
    이후  {"values": [{"value","condition","quote"}, …]}      ← **조건부 여러 값**
검산은 그대로 **닫힌 술어**뿐([[59]]): 인용이 문서에 실재 · 값이 그 인용 안에 실재 ·
조건이 있으면 그것도 그 인용 안에 실재. 뜻은 안 본다. 엔진은 담기만 하고 **고르지 않는다**.

## 재료의 출처
계열 목록은 **A2 선언 `catalog_arg_families.account_class`** 에서 읽는다 — 코드에 도메인 목록을
적지 않는다([[05]]·[[71]]). 문서는 그 계열의 파일명 규약으로만 모은다(`x431.fill_blanks` 와 동일).
결과는 **새 파일**로 쓴다 — 원본을 덮지 않는다(되돌릴 수 있어야 한다).

사용: (리모트·cwd=tau2 · PYTHONPATH=src:…) py x452_conditional_facts.py --port 8140
"""
import argparse
import collections
import io
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import x430_account_facts as FT         # noqa: E402  DOCDIR·ATTRS
import x431_spec_selects as X           # noqa: E402  ask 정본(사본 금지·[[67]])

REP = os.path.abspath(os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026"))
SRC = os.path.join(REP, "x430_account_facts_llm_filled.json")
_N = re.compile(r"[^0-9a-z%$.]+")
_PRE = "doc" + "_"                      # 파일명 규약(계열은 선언에서 온다)

SYS = ("Answer ONLY from the documents. Some values depend on a condition (a balance tier, a "
       "product holding, an age, a period). Reply with ONE JSON object: "
       "{\"values\": [{\"value\": \"<verbatim>\", \"condition\": \"<verbatim, or empty string if it "
       "always applies>\", \"quote\": \"<the verbatim sentence that contains them>\"}]} listing EVERY "
       "distinct value the documents state, or {\"absent\": true} if the documents never state it. "
       "Never paraphrase: the value and the condition must both appear inside their own quote.")


def declared_families():
    """계열 목록은 **A2 선언**에서 읽는다 — 우리가 적지 않는다([[71]] 계약 2항)."""
    with io.open(os.path.join(HERE, "a2", "banking_knowledge.gate.json"), encoding="utf-8") as f:
        d = json.load(f)
    fams = ((d.get("catalog_arg_families") or {}).get("account_class") or [])
    return [str(x) for x in fams]


def norm(t):
    return _N.sub(" ", str(t or "").lower()).strip()


def contained(small, big):
    s, b = norm(small), norm(big)
    return bool(s) and s in b


def docs_by_class(families):
    out = collections.defaultdict(list)
    for f in sorted(os.listdir(FT.DOCDIR)):
        if not f.endswith(".json"):
            continue
        for fam in families:
            pre = _PRE + fam + "_"
            if f.startswith(pre):
                cls = re.sub(r"_\d+\.json$", "", f[len(pre):])
                with io.open(os.path.join(FT.DOCDIR, f), encoding="utf-8") as fh:
                    d = json.load(fh)
                out[cls].append((f[:-5], (d.get("title") or "") + ". " + (d.get("content") or "")))
                break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8140)
    ap.add_argument("--out", default="x430_account_facts_conditional.json")
    ap.add_argument("--attrs", default="", help="비우면 x430.ATTRS 전부")
    a = ap.parse_args()
    fams = declared_families()
    attrs = [x.strip() for x in a.attrs.split(",") if x.strip()] or [n for n, _al in FT.ATTRS]

    tab = json.load(io.open(SRC, encoding="utf-8"))
    byc = docs_by_class(fams)
    print("=" * 96)
    print("x452 · 선언 계열 %d · 클래스 %d · 속성 %d · **빈칸만** 다시 묻는다"
          % (len(fams), len(byc), len(attrs)))
    print("   계열(A2 선언): %s" % ", ".join(fams))
    print("=" * 96)

    added = kept = still = rejected = 0
    for cls in sorted(byc):
        row = tab.get(cls)
        if not isinstance(row, dict):
            row = tab[cls] = {}
        docs = byc[cls]
        joined = " ".join(" ".join(t.split()) for _i, t in docs)
        blob = "\n\n".join("### %s\n%s" % (i, t) for i, t in docs)[:40000]
        hits = []
        for attr in attrs:
            cell = row.get(attr) or {}
            if isinstance(cell, dict) and cell.get("values"):
                kept += 1
                continue
            got = X.ask(a.port, SYS, "# Documents for the %s\n%s\n\n# Question\nWhat is the %s?\n"
                        % (cls, blob, attr.replace("_", " ")), maxtok=600) or {}
            vs = got.get("values")
            if not isinstance(vs, list) or not vs:
                row[attr] = {"values": [], "conflict": False, "evidence": [], "absent": True}
                still += 1
                continue
            keep = []
            for it in vs:
                if not isinstance(it, dict):
                    continue
                v = str(it.get("value") or "")
                c = str(it.get("condition") or "")
                q = str(it.get("quote") or "")
                if not (v and q and contained(q, joined) and contained(v, q)):
                    rejected += 1
                    continue
                if c and not contained(c, q):
                    c = ""                       # 조건이 인용 밖이면 **버린다**(값은 남긴다)
                keep.append({"value": v.strip(), "condition": c.strip(),
                             "quote": " ".join(q.split())[:240], "doc": cls})
            if keep:
                row[attr] = {"values": [k["value"] for k in keep], "conditional": keep,
                             "conflict": len({k["value"] for k in keep}) > 1, "evidence": keep}
                added += 1
                hits.append("%s=%s" % (attr, "/".join(k["value"] for k in keep)[:44]))
            else:
                row[attr] = {"values": [], "conflict": False, "evidence": [], "absent": True}
                still += 1
        if hits:
            print("  %-30s %s" % (cls[:30], " · ".join(hits[:4])))

    p = os.path.join(REP, a.out)
    with io.open(p, "w", encoding="utf-8") as f:
        json.dump(tab, f, ensure_ascii=False, indent=1)
    print("\n새로 채운 칸 %d · 이미 있던 칸 %d · 여전히 없음 %d · 검산 탈락 항목 %d"
          % (added, kept, still, rejected))
    print("→ %s" % p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
