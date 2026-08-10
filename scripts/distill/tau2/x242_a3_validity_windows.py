# -*- coding: utf-8 -*-
r"""x242 — A3 에 **유효 구간**을 선언한다 (빌드 시점 · LLM 이 읽고 엔진은 검산만 · 유료 0).

## 왜 (x235 사다리 · 070/071 · ⛔0 ①②)

`R3_PROMO` **8/8** ↔ `R4_EXPIRED` **0/8**. 활성 프로모션을 주면 닫히고, 만료된 것을 **함께**
주면 무너진다. 만료라고 **말해 주는 것으로는 안 된다**(0/8·C404). 071 라이브가 그 재현이다 —
활성·만료 문서가 둘 다 문맥에 있었고 만료가 미는 이름으로 갔다.

⇒ 엔진이 할 일은 **하나**: *효력 없는 문서를 뺀다.* 그 비교에 필요한 상수(구간)는 **대화와
무관한 정책 상수**이므로 A3 가 빌드 시점에 들고 있는 것이 맞다(C405⒠·A3 자체가 그런 것이다).

## 왜 이것이 떠먹이기가 아닌가 ([[62]]·[[50]] ADB)

규칙이 **코퍼스 전체에 대해 하나**다 — *"이 문서가 스스로 유효 구간을 말하는가"*. 태스크를
보지 않고, 상품도 축도 값도 채우지 않는다. 그 결과가 몇 행인지는 코퍼스가 정한다(실측: 698 중
**5행**). 070/071 의 프로모션이 거기 들어오는 것은 **그 문서들이 구간을 말하기 때문**이지
우리가 골랐기 때문이 아니다.

## 분담

    LLM   문서 → 구간(시작·끝)과 **그 말을 한 문장 축자**            ← 해석([[52]])
    엔진  축자가 문서에 **실재하는지** · 날짜 형식이 맞는지만 검산     ← 이론
    엔진  런타임: 현재 시각이 구간 밖이면 뺀다(날짜 산수뿐)           ← `t2_search.drop_expired`
    모델  남은 것 중 고르기                                          ← 끝까지 모델

⚠엔진은 문서에서 날짜를 **뽑지 않는다** — 정규식으로 뜯으면 그것이 [[59]] 위반이다. 여기서도
  후보를 좁히는 사전-정규식을 두지 않고 **전 문서를 LLM 에 준다**(느리지만 규율이 산다).
⚠못 정하면 **안 적는다**. 모르는 문서는 런타임에 **남는다**(모름 ≠ 만료·[[25]]).
⚠두 층에 **바이트 동일**로 쓴다([[24]] — 한쪽만 고치면 조용히 갈린다).

실행: py -3 x242_a3_validity_windows.py [--limit N] [--apply]
"""
import argparse
import collections
import glob
import io
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from x216_read_and_offset import chat                             # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
LAYERS = ["a2/banking_knowledge.specific.json", "a2/banking_knowledge.gate.json"]
DOCS = "/home/woori/scratch/tau2-bench/data/tau2/domains/banking_knowledge/documents"
ISO = re.compile(r"^\d{4}-\d{2}-\d{2}$")

ASK = ("Below is one internal policy document.\n"
       "Does the document itself state a period during which it is in effect "
       "(a start and an end)?\n"
       'If it does, reply ONLY with JSON: {"from": "YYYY-MM-DD", "to": "YYYY-MM-DD", '
       '"quote": "<the exact sentence fragment that states it, copied verbatim>"}\n'
       'If it does not, reply ONLY with: {}\n'
       "Do not infer a period from anything other than the document's own words.\n\n"
       "DOCUMENT:\n%s")


def norm(s):
    return " ".join(str(s or "").split())


def read_corpus(limit=None):
    out = []
    for p in sorted(glob.glob(os.path.join(DOCS, "doc_*.json"))):
        try:
            o = json.load(open(p, encoding="utf-8"))
        except Exception:
            continue
        if o.get("id"):
            out.append((o["id"], norm(o.get("content"))))
        if limit and len(out) >= limit:
            break
    return out


def formalize(doc_id, body):
    """LLM 이 읽고, 엔진은 **검산만** 한다. 검산에 걸리면 조용히 버린다(모르면 안 뺀다)."""
    try:
        raw = chat(ASK % body[:6000], None, 0.0, 200).get("content", "") or ""
    except Exception as e:
        return None, "호출 실패 %s" % type(e).__name__
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        return None, "JSON 없음"
    try:
        o = json.loads(m.group(0))
    except Exception:
        return None, "JSON 파손"
    if not o:
        return None, None                                   # 구간 없음 = 정상
    f, t, q = o.get("from"), o.get("to"), norm(o.get("quote"))
    if not (ISO.match(str(f or "")) and ISO.match(str(t or ""))):
        return None, "날짜 형식 아님(%r~%r)" % (f, t)
    if not q or q not in body:
        return None, "축자가 문서에 없다"                    # [[22]] 근거-우선: 인용 실재 검증
    if f > t:
        return None, "구간 역전"
    return {"doc": doc_id, "from": f, "to": t, "quote": q[:300]}, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    corpus = read_corpus(a.limit or None)
    print("코퍼스 %d 문서 — 전부 LLM 에 준다(사전 정규식 0·[[59]])" % len(corpus))
    rows, why = [], collections.Counter()
    for i, (doc_id, body) in enumerate(corpus):
        row, err = formalize(doc_id, body)
        if row:
            rows.append(row)
            print("  [%4d] %-52s %s ~ %s | %s" % (i, doc_id, row["from"], row["to"],
                                                  row["quote"][:60]))
        elif err:
            why[err] += 1
        if (i + 1) % 100 == 0:
            print("  … %d/%d · 구간 %d · 기각 %s" % (i + 1, len(corpus), len(rows), dict(why)))
    print("\n구간을 말하는 문서 **%d행** · 검산 기각 %s" % (len(rows), dict(why) or "없음"))
    if not a.apply:
        print("\n(--apply 없이는 쓰지 않는다)")
        return 0
    for rel in LAYERS:
        p = os.path.join(HERE, rel)
        txt = io.open(p, encoding="utf-8").read()
        doc = json.loads(txt)
        if json.dumps(doc, ensure_ascii=False, indent=1) + ("\n" if txt.endswith("\n") else "") \
                != txt:
            print("  중단: %s 재직렬화가 바이트 동일하지 않다" % rel)
            return 1
        doc["policy_ontology"]["doc_windows"] = rows
        doc["policy_ontology"]["_note_doc_windows"] = (
            "★출처 = 정책 문서 축자(빌드 시점 LLM 형식화·x242). 규칙은 코퍼스 전체에 하나 — "
            "'이 문서가 스스로 유효 구간을 말하는가'. 태스크도 상품도 보지 않는다([[23]]·[[62]]). "
            "엔진은 축자 실재와 날짜 형식만 검산하고, 런타임에는 현재 시각과 비교해 **효력 없는 "
            "문서를 뺄 뿐**이다(t2_search.drop_expired). 못 정한 문서는 안 적고, 안 적힌 문서는 "
            "런타임에 남는다(모름 ≠ 만료·[[25]]). 근거: x235 R3 8/8 ↔ R4_EXPIRED 0/8.")
        out = json.dumps(doc, ensure_ascii=False, indent=1) + ("\n" if txt.endswith("\n") else "")
        io.open(p, "w", encoding="utf-8", newline="").write(out)
        print("  기록: %-40s doc_windows %d행" % (rel, len(rows)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
