# -*- coding: utf-8 -*-
r"""x203 — A3 행에 **종류**(`kind`)와 **범위**(`scope`)를 붙인다 (빌드 시점·엔진 0·유료 0).

## 왜 (사용자 지시 2026-08-09 · x201 측정)

> *"카드와 예금이 같은 빈칸 채우기를 하면 안 된다. **타입별로 따로 만들고 채워야** 한다.
>  먼저 **타입 제약에 따라 채우게 A2/A3 를 구성**해야 하지 않나."*

098 의 통과 표에는 개인 체킹 5 · 사업자 카드 6 · 카드 3 이 함께 실리고, 모델은 카드의 단일
최대 수를 집는다(`A_iso` 0/8). 한 줄로 말해 주는 전달 팔도 **0/8**. 종류로 거른 표는 8/8 이고,
LLM 이 종류를 고르는 2단 구성도 8/8 이다(x201).

## 지어내지 않는다

종류는 **행이 이미 인용하고 있는 출처 문서 id** 에서 나온다. 전 주어가 정확히 한 문서군에
속함을 실측으로 확인했다(충돌 0). 규칙은 두 줄이다:

    kind  = `doc_<문서군>_...` 의 문서군                      (business_checking_accounts 등)
    scope = 문서 슬러그가 `(general)` 이면 범위, 아니면 제품

⚠엔진이 문서 id 를 뜯으면 그것이 도메인 패턴매칭이다([[59]]). 그래서 **여기서(빌드 시점)**
  유도해 필드로 적어 두고, 엔진은 그 필드를 읽기만 한다.
⚠한 주어가 여러 문서군에 걸치면 **강제하지 않고 비워 둔다** — 비면 종류 필터에 안 걸려 표에
  그대로 남는다(모름 ≠ 탈락·[[25]]).

실행: py -3 x203_tag_a3_kind.py [--apply]
"""
import argparse
import collections
import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
LAYERS = ["a2/banking_knowledge.specific.json", "a2/banking_knowledge.gate.json"]
GROUPS = ("business_checking_accounts", "business_credit_cards", "checking_accounts",
          "credit_cards", "savings_accounts", "bank_accounts")


def group_of(doc):
    for g in GROUPS:                       # 긴 것 우선 (business_* 가 먼저 걸리게 정렬돼 있다)
        if doc.startswith("doc_" + g + "_"):
            return g
    return None


def derive(rows):
    """주어 → (kind, scope). 갈리면 None (강제하지 않는다)."""
    ks, sc = collections.defaultdict(set), collections.defaultdict(set)
    for r in rows:
        s = str(r.get("subject") or "").strip()
        doc = str((r.get("source") or {}).get("doc") or "")
        g = group_of(doc)
        if not (s and g):
            continue
        ks[s].add(g)
        sc[s].add("general" if "_(general)_" in doc else "product")
    out = {}
    for s in ks:
        k = sorted(ks[s])[0] if len(ks[s]) == 1 else None
        v = sorted(sc[s])[0] if len(sc[s]) == 1 else None
        out[s] = (k, v)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    staged = {}
    for rel in LAYERS:
        p = os.path.join(HERE, rel)
        txt = io.open(p, encoding="utf-8").read()
        doc = json.loads(txt)
        if json.dumps(doc, ensure_ascii=False, indent=1) + ("\n" if txt.endswith("\n") else "") != txt:
            print("  중단: %s 재직렬화가 바이트 동일하지 않다" % rel)
            return 1
        rows = doc["policy_ontology"]["rows"]
        table = derive(rows)
        n = 0
        for r in rows:
            s = str(r.get("subject") or "").strip()
            k, v = table.get(s, (None, None))
            if k:
                r["kind"] = k
                n += 1
            if v:
                r["scope"] = v
        staged[rel] = (doc, n, txt.endswith("\n"))
        amb = sorted(s for s, (k, _v) in table.items() if not k)
        print("  %-38s 태깅 %d/%d행 · 미확정 주어 %s"
              % (rel, n, len(rows), amb or "없음"))
        if rel == LAYERS[0]:
            cnt = collections.Counter(k for k, _v in table.values() if k)
            sco = collections.Counter(v for _k, v in table.values() if v)
            print("     종류: %s" % dict(cnt))
            print("     범위: %s" % dict(sco))

    a_rows = staged[LAYERS[0]][0]["policy_ontology"]["rows"]
    b_rows = staged[LAYERS[1]][0]["policy_ontology"]["rows"]
    same = json.dumps(a_rows, ensure_ascii=False, sort_keys=True) == \
        json.dumps(b_rows, ensure_ascii=False, sort_keys=True)
    print("  두 층 rows 동일: %s" % ("OK" if same else "**불일치**"))
    if not same:
        return 1
    if not a.apply:
        print("\n(미적용 — 쓰려면 --apply)")
        return 0
    for rel, (doc, _n, nl) in staged.items():
        io.open(os.path.join(HERE, rel), "w", encoding="utf-8", newline="").write(
            json.dumps(doc, ensure_ascii=False, indent=1) + ("\n" if nl else ""))
        print("  wrote %s" % rel)
    return 0


if __name__ == "__main__":
    sys.exit(main())
