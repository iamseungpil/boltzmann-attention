# -*- coding: utf-8 -*-
r"""x434 — **문서 미기재인가 추출 실패인가**: 추출을 격리해 본다 (사용자 지시 2026-08-20)

## 왜
x431 의 빈칸 처리는 *"문서 미기재 확정 94~98"* 을 냈는데, 그것은 **모델이 `absent: true` 라고 선언한
것을 우리가 그대로 믿은 값**이다. 그리고 그 빈칸이 지금 선호 랭킹을 막고 있다 — 순위 보류가
38·36·27 이고 전부 미기재 탓이다. 그러니 먼저 물어야 한다: **정말 문서에 없나, 아니면 섞여서 못 찾나.**

## 팔 (같은 물음 · 문맥 크기만 다르다 · [[18]] 정보를 빼지 않는다)
    A_blob    그 클래스 **문서 전부를 한 덩어리로**(현행 방식 재현·40k자)
    B_perdoc  **문서 하나씩** 따로 묻는다(선택 없이 전수) — 어느 하나가 값을 내면 채택
    C_window  그 문서 안에서 **값이 있을 만한 창**만(문서를 3등분해 각각) — 더 좁힌 격리

★문서를 **고르지 않는다**. 속성어로 문서를 추리면 그것이 [[59]] 패턴매칭이다 — 전수로 돈다.
★엔진 검산은 그대로: 인용이 그 문서에 **축자로 실재**하고 값이 인용 안에 있어야 채택.
★**민감도 통제**: 이미 값이 있는 칸을 같은 팔로 다시 물어 **회수율**을 잰다. 회수율이 낮으면
  팔 자체가 둔한 것이지 문서가 없는 게 아니다([[57]] 부정통제 자리).

## 사전 고정 해석
- `B/C 가 A 의 absent 를 상당수 회수` ⇒ **추출 실패였다**(우리 층·섞임이 가린다) → 표를 다시 짓는다
- `B/C 도 0 에 가까움` ∧ `민감도 통제 높음` ⇒ **문서 미기재 확정** → 그 축은 ASK 표적이다
- `민감도 통제도 낮음` ⇒ 팔이 둔하다 ⇒ 이 프로브로는 판정 불가(수치 인용 금지)

사용: py -3 x434_extract_isolation.py [--cells 24] [--port 8141]
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

import x430_account_facts as FT  # noqa: E402
import x431_spec_selects as S  # noqa: E402

ATTRS = [x[0] for x in FT.ATTRS]
SYS = ("Answer ONLY from the text given. Reply with ONE JSON object: "
       "{\"value\": \"<verbatim>\", \"quote\": \"<verbatim sentence>\"} if the text states it, "
       "otherwise {\"absent\": true}. Never paraphrase.")


def load_docs(docdir, family):
    pre = "doc_%s_" % family
    byc = collections.defaultdict(list)
    for f in sorted(os.listdir(docdir)):
        if not (f.startswith(pre) and f.endswith(".json")):
            continue
        cls = re.sub(r"_\d+\.json$", "", f[len(pre):])
        with io.open(os.path.join(docdir, f), encoding="utf-8") as fh:
            d = json.load(fh)
        byc[cls].append((f.replace(".json", ""), (d.get("title") or "") + ". " + (d.get("content") or "")))
    return byc


def try_one(port, text, attr, joined):
    got = S.ask(port, SYS, "# Text\n%s\n\n# Question\nWhat is the %s?\n"
                % (text[:38000], attr.replace("_", " ")), maxtok=300)
    if got.get("absent"):
        return None
    val = str(got.get("value", "")).strip()
    q = " ".join(str(got.get("quote") or "").split())
    if val and q and S.cite_norm(q) in S.cite_norm(joined) and \
            S.cite_norm(val).replace(" ", "") in S.cite_norm(q).replace(" ", ""):
        return {"value": val, "quote": q[:200]}
    return None


def thirds(t):
    n = max(1, len(t) // 3)
    return [t[:n], t[n:2 * n], t[2 * n:]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8141)
    ap.add_argument("--docdir", default=FT.DOCDIR)
    ap.add_argument("--family", default="checking_accounts")
    ap.add_argument("--cells", type=int, default=24)
    ap.add_argument("--control", type=int, default=12)
    a = ap.parse_args()

    tblp = os.path.abspath(S.TBL).replace(".json", "_filled.json")
    with io.open(tblp, encoding="utf-8") as f:
        table = json.load(f)
    byc = load_docs(a.docdir, a.family)

    absent, filled = [], []
    for cls, row in table.items():
        if cls not in byc or not isinstance(row, dict):
            continue
        for at in ATTRS:
            cell = row.get(at) or {}
            if cell.get("values"):
                filled.append((cls, at, cell["values"][0]))
            elif cell.get("absent"):
                absent.append((cls, at))
    print("=" * 100)
    print("x434 · 추출 격리 · %s · absent 칸 %d · 채워진 칸 %d" % (a.family, len(absent), len(filled)))
    print("=" * 100)

    def run_arms(cls, at):
        docs = byc[cls]
        joined = " ".join(t for _i, t in docs)
        res = {}
        res["A_blob"] = try_one(a.port, "\n\n".join("### %s\n%s" % (i, t) for i, t in docs), at, joined)
        got = None
        for _i, t in docs:
            got = try_one(a.port, t, at, joined)
            if got:
                break
        res["B_perdoc"] = got
        got = None
        for _i, t in docs:
            for part in thirds(t):
                got = try_one(a.port, part, at, joined)
                if got:
                    break
            if got:
                break
        res["C_window"] = got
        return res

    tal = collections.Counter()
    rows = []
    for cls, at in absent[:a.cells]:
        r = run_arms(cls, at)
        for k, v in r.items():
            tal[(k, bool(v))] += 1
        rec = {"cls": cls, "attr": at, "kind": "absent",
               "A": r["A_blob"], "B": r["B_perdoc"], "C": r["C_window"]}
        rows.append(rec)
        if r["B_perdoc"] or r["C_window"]:
            print("  ★회수 %-26s %-30s B=%s C=%s" % (cls[:26], at,
                  (r["B_perdoc"] or {}).get("value"), (r["C_window"] or {}).get("value")))
    print("\n## absent 로 확정했던 칸 %d 개 재시도" % min(len(absent), a.cells))
    n = min(len(absent), a.cells)
    for k in ("A_blob", "B_perdoc", "C_window"):
        print("   %-10s 값을 찾음 %2d/%d (%.0f%%)" % (k, tal[(k, True)], n, 100.0 * tal[(k, True)] / n if n else 0))

    print("\n## 민감도 통제 — 이미 값이 있는 칸을 같은 팔로 다시 (n=%d)" % a.control)
    ct = collections.Counter()
    import hashlib
    pick = sorted(filled, key=lambda x: hashlib.md5(("%s|%s" % (x[0], x[1])).encode()).hexdigest())[:a.control]
    for cls, at, val in pick:
        r = run_arms(cls, at)
        for k, v in r.items():
            ct[(k, bool(v))] += 1
        rows.append({"cls": cls, "attr": at, "kind": "control", "truth": val,
                     "A": r["A_blob"], "B": r["B_perdoc"], "C": r["C_window"]})
    for k in ("A_blob", "B_perdoc", "C_window"):
        print("   %-10s 회수 %2d/%d (%.0f%%)" % (k, ct[(k, True)], len(pick),
                                               100.0 * ct[(k, True)] / len(pick) if pick else 0))
    p = os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026", "x434_extract_isolation.json")
    with io.open(os.path.abspath(p), "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)
    print("\n→ %s" % os.path.abspath(p))
    return 0


if __name__ == "__main__":
    sys.exit(main())
