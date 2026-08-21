# -*- coding: utf-8 -*-
r"""x457 — 감사 인용에서 **A3 블록을 생성** (2026-08-21·오프라인·LLM 0·[[72]] 1회 저작)

## 왜 (재추출이 필요 없다)
원래 계획은 *감사 → 축 목록 갱신 → `x452` 로 사실표를 다시 채움* 이었다. 그런데 `x453` 전수
스캔이 이미 **`(클래스, 축, 값, 문서, 오프셋, 길이, 축자 인용)` 2,497건을 검산까지 마쳐** 내놓는다
(위치 못 잡음 0 · 절 범위 검산 실패 0). **감사 산출물이 곧 사실표다** — 다시 뽑는 것은 같은 값을
두 번 사는 것이고, 두 벌이 생기면 조용히 갈라진다([[67]]).

사용자 지시(2026-08-21): *"어차피 런타임에 모든 문서 읽을려면 비용이 많이 든다. 이번 한번에
빠짐 없이 기록하라."* · *"문서 id, 오프셋과 읽을 길이를 지정해주면 되지 않을까?"*

## 무엇을 내나 (셋 · 전부 A3 형식 그대로)
    catalog_attrs   정본 축 → {aliases, 관측 클래스 수, 예시, 형태-타입}
    policy_facts    (subject=클래스, axis, value, condition, sources[]) 행
    sub_docs        **격리 서브에 넘길 읽기 명세** — 클래스 → 축 → [{doc, off, len}]
`sub_docs` 가 bm25 를 대체하는 물건이다: 엔진은 선언된 (문서, 오프셋, 길이)를 **자르기만** 한다.

## 규율 (전부 결과 보기 전에 고정 · 엔진 판단 0)
    · 행은 **위치가 잡히고 절 범위 검산을 통과한** 인용에서만 나온다(`span_ok`)
    · 같은 (클래스, 축)에 **서로 다른 값**이 있으면 **둘 다 남기고 `conflict`** 로 표시한다 —
      우리가 고르지 않는다([[62]]). 조건(`condition`)이 필요한 행은 **정확히 이 행들**이므로
      조건 형식화 패스는 여기로만 좁힌다(닫힌 술어: 서로 다른 값이 2개 이상).
    · 같은 값이 여러 문서에 있으면 **행 하나 + `sources[]` 여럿**(출처를 버리지 않는다)
    · 축 이름은 `x455` 병합 결과가 있으면 그 정본명으로 접는다. 없으면 원시 이름 그대로.
    · 타입은 **형태만** 본다(`$`→USD · `%`→percent · 숫자→count · 그 외 text). 뜻은 안 본다([[59]]).
    ⛔gold·태스크·실패 사례를 보지 않는다 — `sim_results` 를 열지 않는다([[23]]).

## 산출은 **제안**이다
A3 병합은 사람이 1회 검토한 뒤 별도로 하고, 두 층을 바이트 동일로 맞춘다([[24]]).

사용: py x457_a3_from_cites.py [--groups x455_axis_groups.json]
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

import x430_account_facts as FT         # noqa: E402  현행 선언(비교용)

REP = os.path.abspath(os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026"))


def type_form(value):
    """값의 **형태만** 보고 타입을 붙인다 — 뜻은 안 본다([[59]]·`x430.RE_VAL` 동형)."""
    v = str(value or "").strip()
    if not v:
        return "text"
    if v.startswith("$") or re.match(r"^-?\$?[\d,]+(\.\d+)?$", v) and "$" in v:
        return "USD"
    if v.endswith("%") or re.match(r"^-?[\d.]+\s*%$", v):
        return "percent"
    if re.match(r"^-?[\d,]+(\.\d+)?$", v):
        return "count"
    return "text"


def declared_family_groups():
    """A3 선언에서 **계열 묶음**을 읽는다 — `catalog_arg_families` 의 키(예: account_class ·
    card_type)와 그 아래 계열 목록. 코드에 계열 이름을 적지 않는다([[71]] 계약 2항)."""
    p = os.path.join(HERE, "a2", "banking_knowledge.specific.json")
    with io.open(p, encoding="utf-8") as f:
        fam = (json.load(f).get("catalog_arg_families") or {})
    out = {}
    for key, lst in fam.items():
        for f2 in (lst or []):
            out["doc_%s_" % f2] = key
    return out


def family_of(doc_id, prefixes):
    """문서 id → 선언된 계열 묶음 키. 어느 접두사에도 안 걸리면 `(공용)`.

    **형태만** 본다(파일명 접두사 대조·[[59]]). 공용 문서(`bank_accounts_(general)` 등)는
    한 계열의 것이 아니라 여러 계열이 함께 쓰는 정책이므로 별도로 센다 — 섞임으로 치지 않는다.
    """
    d = str(doc_id or "")
    for pre, key in prefixes.items():
        if d.startswith(pre):
            return key
    return "(공용)"


def load_groups(path):
    """`x455` 병합 결과 → {원시 이름: 정본명}. 없으면 빈 사전(원시 이름 그대로 쓴다)."""
    if not path:
        return {}, {}
    p = os.path.join(REP, path)
    if not os.path.exists(p):
        print("⚠병합 결과 없음(%s) — 원시 축 이름을 그대로 쓴다" % path)
        return {}, {}
    with io.open(p, encoding="utf-8") as f:
        g = json.load(f)
    fold, alias = {}, collections.defaultdict(list)
    for grp in (g.get("groups") or []):
        can = grp.get("canonical")
        for m in (grp.get("members") or []):
            fold[m] = can
        alias[can] += (grp.get("aliases") or [])
    return fold, alias


def sub_docs_block(cites, axes, always, fold):
    """한 서브가 쓸 **읽기 명세 선언**을 만든다 — `{always:[id], spans:{class:[{doc,off,len}]}}`.

    사용자 축자(2026-08-21): *"shell 로 해야 한다. bm25 는 approximate 이다. 정확한 100% 문서
    링크를 shell 로 읽어와서 격리해야 한다."* 이 블록이 그 링크다 — 엔진은 선언된 범위를
    **자르기만** 하고 검색하지 않는다.

    `axes` 는 **A3 가 선언**하는 축 목록이고(이 서브가 무엇을 알아야 하는가), 출처는 그 도구의
    `op.reducers` 범주(base·checking·card·relationship·tier)와 `doc_045`/`doc_046` 이 정의한
    범주다 — gold 가 아니다([[23]]). 엔진은 그 목록을 **문자열로 대조**만 한다([[59]]).
    `always` 는 범주 **정의**를 담은 공용 정책 문서다: `_045` 가 *"Relationship bonuses for
    holding multiple Rho-Bank products"* 로 relationship 을 정의하고, `_046` 이 checking max1 을,
    `_012` 가 어느 페어링이 자격인지를 말한다. 그 셋이 없으면 `kind` 분류가 불가능하다.
    """
    want = {str(x) for x in (axes or [])}
    spans = collections.defaultdict(list)
    for raw, lst in cites.items():
        ax = fold.get(raw, raw)
        if raw not in want and ax not in want:
            continue
        for c in lst:
            if c.get("span_ok") is not True or c.get("section_off") is None:
                continue
            spec = {"doc": c.get("doc"), "off": c.get("section_off"), "len": c.get("section_len")}
            if spec not in spans[c.get("class")]:
                spans[c.get("class")].append(spec)
    # ★전달 단위 = **문서**(2026-08-21 실측으로 확정). 절 단위로 자르면 감사가 축으로 잡지
    #   못한 줄이 통째로 사라진다 — `green_account_(checking)_001` 의
    #   *"Boost a linked savings account's APY: Gold +0.75% or Silver +0.25%"* 가 **어떤 span
    #   에도 안 들어간다**(093·094 가 둘 다 필요로 하는 값이다). 문서 중앙 길이가 989자라
    #   문서 단위가 절보다 크게 비싸지도 않다(093 9편 12,510자 · 094 11편 12,387자).
    #   오프셋은 **근거 앵커로 남긴다** — 어느 줄이 그 축을 말했는지 감사할 수 있어야 한다.
    docs = {k: sorted({x["doc"] for x in v}) for k, v in spans.items()}
    return {"_note_": ("전달 단위는 **문서**다 — 엔진은 선언된 id 를 읽어 넘기기만 하고 검색하지 "
                       "않는다. `spans` 는 그 문서 안 어디가 그 축을 말했는지의 **근거 앵커**이고 "
                       "자르는 데 쓰지 않는다(절로 자르면 축으로 안 잡힌 줄이 사라진다·실측). "
                       "축 목록의 출처 = op.reducers 범주 + doc_045/046 의 범주 정의."),
            "axes": sorted(want), "always": list(always),
            "by_class": {k: v for k, v in sorted(docs.items())},
            "spans": {k: v for k, v in sorted(spans.items())}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", default="x453_attr_coverage_all.json")
    ap.add_argument("--groups", default="x455_axis_groups.json")
    ap.add_argument("--out", default="x457_a3_blocks.json")
    ap.add_argument("--sub-axes", default="",
                    help="이 서브가 알아야 할 축(A3 선언·쉼표)")
    ap.add_argument("--sub-always", default="",
                    help="범주 정의를 담은 공용 정책 문서 id(쉼표)")
    a = ap.parse_args()

    with io.open(os.path.join(REP, a.audit), encoding="utf-8") as f:
        aud = json.load(f)
    fold, alias = load_groups(a.groups)
    cites = aud.get("cites") or {}
    print("=" * 96)
    print("x457 · 감사 인용 %d건 · 원시 축 %d · 병합 사전 %d항"
          % (sum(len(v) for v in cites.values()), len(cites), len(fold)))
    print("=" * 96)

    # ── 행 만들기: (subject, axis, value) 로 접고 출처는 전부 남긴다 ──────────
    fam_pre = declared_family_groups()
    axis_fams = collections.defaultdict(set)
    axis_famvals = collections.defaultdict(set)
    rows = collections.OrderedDict()
    dropped_unlocated = 0
    axis_classes = collections.defaultdict(set)
    axis_cites = collections.Counter()
    raw_of = collections.defaultdict(set)
    for raw, lst in cites.items():
        ax = fold.get(raw, raw)
        raw_of[ax].add(raw)
        for c in lst:
            if c.get("span_ok") is not True or c.get("section_off") is None:
                dropped_unlocated += 1
                continue
            key = (c.get("class"), ax, str(c.get("value") or "").strip())
            r = rows.get(key)
            if r is None:
                r = rows[key] = {"subject": c.get("class"), "axis": ax,
                                 "value": str(c.get("value") or "").strip(),
                                 "condition": None, "sources": []}
            src = {"doc": c.get("doc"), "off": c.get("section_off"),
                   "len": c.get("section_len"), "section": c.get("section"),
                   "quote": c.get("quote")}
            if src not in r["sources"]:
                r["sources"].append(src)
            axis_classes[ax].add(c.get("class"))
            axis_cites[ax] += 1
            _fk = family_of(c.get("doc"), fam_pre)
            axis_fams[ax].add(_fk)
            axis_famvals[(ax, _fk)].add(str(c.get("value") or "")[:24])

    # 충돌 표시 = 같은 (subject, axis) 에 서로 다른 값이 2개 이상 (닫힌 술어)
    byax = collections.defaultdict(list)
    for (subj, ax, _v), r in rows.items():
        byax[(subj, ax)].append(r)
    n_conflict = 0
    for k, rs in byax.items():
        if len({r["value"] for r in rs}) > 1:
            n_conflict += len(rs)
            for r in rs:
                r["conflict"] = True

    facts = list(rows.values())

    # ── catalog_attrs ────────────────────────────────────────────────────────
    cat = collections.OrderedDict()
    for ax in sorted(axis_classes, key=lambda x: (-len(axis_classes[x]), x)):
        ex = None
        for r in facts:
            if r["axis"] == ax and r["sources"]:
                ex = {"class": r["subject"], "value": r["value"],
                      "doc": r["sources"][0]["doc"], "quote": r["sources"][0]["quote"]}
                break
        types = collections.Counter(type_form(r["value"]) for r in facts if r["axis"] == ax)
        cat[ax] = {"aliases": sorted(set(alias.get(ax) or [])),
                   "n_classes": len(axis_classes[ax]), "n_cites": axis_cites[ax],
                   "type_form": (types.most_common(1)[0][0] if types else "text"),
                   "type_votes": dict(types), "example": ex,
                   "_merged_from": sorted(raw_of[ax])}

    # ── sub_docs: 클래스 → 축 → 읽기 명세(중복 제거) ─────────────────────────
    sub = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in facts:
        for s in r["sources"]:
            spec = {"doc": s["doc"], "off": s["off"], "len": s["len"]}
            if spec not in sub[r["subject"]][r["axis"]]:
                sub[r["subject"]][r["axis"]].append(spec)
    sub = {c: {ax: v for ax, v in d.items()} for c, d in sub.items()}

    # 전달 부피 — 클래스별로 **모든 축**을 다 넘길 때의 자수(중복 span 제거)
    vol = {}
    for c, d in sub.items():
        seen, n = set(), 0
        for _ax, specs in d.items():
            for s in specs:
                k = (s["doc"], s["off"], s["len"])
                if k not in seen:
                    seen.add(k)
                    n += int(s["len"] or 0)
        vol[c] = {"chars": n, "spans": len(seen)}

    # ── ★계열-섞임 검사 (2026-08-21·사용자 지시 *"검사 붙여라"*) ────────────────
    #   계기: `interest_rate` 는 예금에선 APY 와 같은 값인데(0.11 · 1.25 · 5.0 · 7.5%)
    #   **카드에선 차입 이자율**이다(20.99% · 20.49% — *"Carried balances accrue interest at"*).
    #   이름이 같다고 접으면 카드 APR 이 예금 APY 축의 후보로 섞이고, 그 축이 하필 093/094 의
    #   `expected_apy` 다 — *"못 찾아 0.0"* 보다 나쁜 실패가 된다.
    #   술어는 닫혀 있다: 그 축의 인용이 **선언된 계열 묶음 둘 이상**에서 왔는가(공용 문서 제외).
    #   엔진은 표시만 하고 **가르지 않는다** — 정본 절차의 *"사람 1회 확인"* 이 이 자리다.
    mixed = []
    for ax in sorted(axis_fams):
        keys = {k for k in axis_fams[ax] if k != "(공용)"}
        if len(keys) > 1:
            mixed.append({"axis": ax, "family_groups": sorted(keys),
                          "values_by_group": {k: sorted(axis_famvals[(ax, k)])[:8]
                                              for k in sorted(keys)},
                          "n_cites": axis_cites[ax]})
    for ax, m in cat.items():
        m["family_groups"] = sorted(axis_fams.get(ax) or [])

    have = {n for n, _al in FT.ATTRS}
    payload = {"audit": a.audit, "groups": a.groups,
               "catalog_attrs": cat, "policy_facts": facts, "sub_docs": sub,
               "family_mixed_axes": mixed,
               "report": {"n_facts": len(facts), "n_axes": len(cat),
                          "n_family_mixed": len(mixed),
                          "n_conflict_rows": n_conflict,
                          "n_dropped_unlocated": dropped_unlocated,
                          "n_classes": len(sub),
                          "declared_now": sorted(have),
                          "declared_covered": sorted(n for n in have if n in cat),
                          "delivery_volume": vol}}
    # ── 서브별 읽기 명세 선언 ────────────────────────────────────────────────
    if a.sub_axes:
        _axes = [x.strip() for x in a.sub_axes.split(",") if x.strip()]
        _always = [x.strip() for x in a.sub_always.split(",") if x.strip()]
        blk = sub_docs_block(cites, _axes, _always, fold)
        payload["sub_docs_block"] = blk
        _nd = sum(len(v) for v in blk["by_class"].values())
        _n = sum(len(v) for v in blk["spans"].values())
        print(NLC + "[서브 읽기 명세] 축 %d · 공용 %d편 · 클래스 %d · 문서 %d · 앵커 span %d"
              % (len(blk["axes"]), len(blk["always"]), len(blk["by_class"]), _nd, _n))
        for cls in sorted(blk["by_class"])[:10]:
            print("    %-34s %2d편" % (cls[:34], len(blk["by_class"][cls])))

    p = os.path.join(REP, a.out)
    with io.open(p, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    print("\n[산출물] → %s" % p)

    print("\n" + "=" * 96)
    print("사실 행 %d · 정본 축 %d · 클래스 %d · 충돌 행 %d · 위치 없어 버린 인용 %d"
          % (len(facts), len(cat), len(sub), n_conflict, dropped_unlocated))
    vs = sorted(v["chars"] for v in vol.values())
    print("클래스별 전달 부피(전 축): 중앙 %d자 · 최대 %d자 (span 중앙 %d개)"
          % (vs[len(vs) // 2] if vs else 0, vs[-1] if vs else 0,
             sorted(v["spans"] for v in vol.values())[len(vol) // 2] if vol else 0))
    print("현행 선언 16종 중 이 축 집합이 덮는 것: %d" % len(payload["report"]["declared_covered"]))

    print("\n[상위 축 · 관측 클래스 수]")
    for ax, m in list(cat.items())[:25]:
        print("  %-34s %2d클래스 %3d인용 %-7s ← %d이름"
              % (ax[:34], m["n_classes"], m["n_cites"], m["type_form"], len(m["_merged_from"])))

    print("\n★[계열-섞임 축 %d개 — 접으면 안 되는 후보. 손으로 본다]" % len(mixed))
    for m in mixed[:20]:
        print("  %-30s %s" % (m["axis"][:30], " | ".join(
            "%s=%s" % (k, ",".join(v)[:34]) for k, v in m["values_by_group"].items())))
    print("\n[조건이 필요한 자리 — 같은 (클래스, 축)에 값이 여럿]")
    shown = 0
    for (subj, ax), rs in byax.items():
        if len({r["value"] for r in rs}) > 1 and shown < 12:
            print("  %-26s %-24s → %s" % (subj[:26], ax[:24],
                                          " | ".join(sorted({r["value"][:18] for r in rs}))))
            shown += 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
