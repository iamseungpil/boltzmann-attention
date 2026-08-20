# -*- coding: utf-8 -*-
r"""x451 — **계좌 클래스 선택**: 목록만으로는 왜 안 되나, 무엇을 주면 갈리나 (2026-08-21·격리·무료)

## 왜 (G1 · 사용자 지시 *"8개 태스크 pass 통과부터"* → *"G1 부터 돌려라"*)
t7333 변이 전수 분해에서 055·063·070 의 실패는 전부 `open_bank_account_4821.account_class` 다.
그런데 A2 는 이미 `write_arg_enum` 으로 **계열별 유효 목록을 강제**하고 있었고(`T2_WRITE_ARG_ENUM=1`
은 t7333 PIN 에 켜져 있었다) 그래도 틀렸다 ⇒ **목록은 받았는데 그중 틀린 것을 고른다.**
카드축에는 `check_card_application_fit`(문서 사실 적격표) + `catalog_arg_docs`(문서 배달)가 있는데
**계좌축에는 둘 다 없다**. 이 프로브는 **어느 조각이 필요한지**를 격리로 가른다([[62]] 순서).

## 팔 (전부 유효 목록을 준다 — 그것이 현행 라이브다)
    E_enum    유효 클래스 **목록만**                          = 현행 재현
    F_facts   목록 + **문서 사실표**(속성 + 축자 인용)         카드축 적격표의 대응물(`x430`·gold 미참조)
    D_docs    목록 + **그 계열 문서 본문**                     `spend_category` 배달의 대응물
    N_sham    목록 + **다른 계열** 문서 같은 편수              부정통제([[57]])
  ⚠재료는 전부 **파일명 규약과 선언**에서만 나온다 — 우리가 어느 클래스가 맞는지 말하지 않는다.

## 채점 (닫힌 술어만)
    correct   gold 의 `account_class` 와 일치 (진단 라벨·[[69]]·A2 저작에 쓰지 않는다·[[23]])
    in_enum   낸 값이 그 계열 유효 목록 안에 있나 (날조 검출)

사용: (리모트·cwd=tau2 · PYTHONPATH=src:…) py x451_account_class_iso.py --port 8141
"""
import argparse
import collections
import glob
import gzip
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

import t2_forensic as F                 # noqa: E402
import x431_spec_selects as X           # noqa: E402  ask 정본
import x430_account_facts as FT         # noqa: E402  DOCDIR
import x448_index_vs_all_iso as V       # noqa: E402  form_norm·as_cat(사본 금지·[[67]])

REP = os.path.abspath(os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026"))
BASE = os.path.join(REP, "sim_results")
TOOL = "open_bank_account_4821"
FACTS = os.path.join(REP, "x430_account_facts_llm_filled.json")

SYS = ("You are the module that decides ONE thing for a Rho-Bank support agent: which account class "
       "to open for this customer. Reply with ONE JSON object only: "
       "{\"account_class\": \"<exactly one name from the CANDIDATES list>\", "
       "\"why\": \"<one short sentence>\"}. Pick the class that satisfies what the customer asked for. "
       "Use a name from CANDIDATES verbatim.")


def group_map():
    with io.open(os.path.join(HERE, "a2", "banking_knowledge.gate.json"), encoding="utf-8") as f:
        d = json.load(f)
    for spec in (d.get("write_arg_enum") or []):
        if spec.get("arg") == "account_class" and spec.get("group_map"):
            return spec["group_map"]
    return {}


def classes_of(family):
    """계열 → 클래스 슬러그. **파일명 규약만** 읽는다(뜻 해석 0·live 의 enum 과 같은 원천)."""
    out = []
    for p in sorted(glob.glob(os.path.join(FT.DOCDIR, "doc_%s_*.json" % family))):
        b = os.path.basename(p)[:-5]
        m = re.match(r"^doc_%s_(.+)_(\d+)$" % re.escape(family), b)
        if not m:
            continue
        c = m.group(1)
        if c not in out:
            out.append(c)
    return out


def docs_of(family, klass=None, pad=1200):
    out = []
    pat = "doc_%s_%s_*.json" % (family, klass) if klass else "doc_%s_*.json" % family
    for p in sorted(glob.glob(os.path.join(FT.DOCDIR, pat))):
        try:
            d = json.load(io.open(p, encoding="utf-8"))
        except Exception:
            continue
        out.append((d.get("id") or os.path.basename(p)[:-5], str(d.get("title") or ""),
                    str(d.get("content") or "")[:pad]))
    return out


def facts_block(classes):
    """사실표 → 클래스별 **문서가 명시한 값 + 축자 인용**. 없는 칸은 비운다(추정 0)."""
    try:
        tab = json.load(io.open(FACTS, encoding="utf-8"))
    except Exception:
        return ""
    lines = []
    for c in classes:
        row = tab.get(c)
        if not isinstance(row, dict):
            continue
        bits = []
        for attr, v in sorted(row.items()):
            if not isinstance(v, dict):      # `_note_*` 등 메타 키는 건너뛴다(1차 실측 크래시)
                continue
            vals = (v or {}).get("values") or []
            if not vals:
                continue
            ev = ((v or {}).get("evidence") or [{}])[0]
            bits.append("    %-32s %-10s  “%s”" % (attr, vals[0], str(ev.get("quote") or "")[:90]))
        if bits:
            lines.append("  %s\n%s" % (c, "\n".join(bits)))
    return "\n".join(lines)


def cases():
    """055·063·070 sim 에서 `open_bank_account` 호출 시점 — 손님 발화·account_type·gold 클래스."""
    out, seen = [], set()
    mut = F.mutating_tools()
    for p in sorted(glob.glob(os.path.join(BASE, "bank_t7333_*_20260821c.results.json.gz"))):
        try:
            d = json.load(gzip.open(p, "rt", encoding="utf-8"))
        except Exception:
            continue
        for s in (d.get("simulations") or []):
            tid = str(s.get("task_id") or "")
            if tid not in ("task_055", "task_063", "task_070"):
                continue
            g = [x for x in (F.mutation_diff(s, mut).get("gold") or [])
                 if (x.get("name") or "") == TOOL]
            if not g:
                continue
            msgs = s.get("messages") or []
            said = " \n".join(" ".join(str(m.get("content") or "").split())
                              for m in msgs if m.get("role") == "user" and m.get("content"))
            for gi in g:
                a = gi.get("args") or {}
                k = (tid, str(a.get("account_type")), str(a.get("account_class")))
                if k in seen:
                    continue
                seen.add(k)
                out.append({"task": tid, "account_type": str(a.get("account_type") or ""),
                            "gold": str(a.get("account_class") or ""), "said": said[:5000]})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8141)
    ap.add_argument("--tag", default="g1a")
    ap.add_argument("--arms", default="E_enum,F_facts,D_docs,N_sham")
    ap.add_argument("--maxchars", type=int, default=60000)
    a = ap.parse_args()
    arms = [x.strip() for x in a.arms.split(",") if x.strip()]
    gm = group_map()
    cs = cases()
    print("=" * 100)
    print("x451 · 사례 %d · 계열 사상 %s" % (len(cs), gm))
    print("=" * 100)

    rows = []
    for c in cs:
        fam = gm.get(c["account_type"]) or ""
        klasses = classes_of(fam)
        # ★후보는 **라이브와 같은 표기**로 준다 — `t2_search._disp_name` 정본(슬러그 ↔ 표시명이
        #   갈리면 조용히 빗나간다·FIX-6 전례). 우리가 이름을 짓지 않는다([[67]]).
        import t2_search as _ts
        disp = {_ts._disp_name(k): k for k in klasses}
        enum = "CANDIDATES (%s):\n%s\n" % (c["account_type"],
                                           "\n".join("  - %s" % d for d in disp))
        others = [f for f in set(gm.values()) if f != fam]
        print("\n%s · type=%s · gold=%s · 후보 %d"
              % (c["task"], c["account_type"], c["gold"], len(klasses)))
        for arm in arms:
            if arm == "E_enum":
                mat = ""
            elif arm == "F_facts":
                mat = "DOCUMENTED FACTS (value — verbatim quote):\n" + facts_block(klasses)
            elif arm == "D_docs":
                mat = "DOCUMENTS:\n" + "\n\n".join(
                    "### %s — %s\n%s" % (i, t, b) for i, t, b in docs_of(fam))
            else:
                mat = "DOCUMENTS:\n" + "\n\n".join(
                    "### %s — %s\n%s" % (i, t, b) for i, t, b in docs_of(others[0] if others else fam))
            body = (enum + "\n" + mat[:a.maxchars] + "\n\n# What the customer said\n%s\n" % c["said"])
            ans = X.ask(a.port, SYS, body, maxtok=300) or {}
            v = str(ans.get("account_class") or "").strip()
            slug = re.sub(r"[^a-z0-9]+", "_", v.lower()).strip("_")
            gold_slug = re.sub(r"[^a-z0-9]+", "_", c["gold"].lower()).strip("_")
            ok = bool(slug) and (slug == gold_slug or gold_slug.startswith(slug) or slug.startswith(gold_slug))
            in_enum = any(slug == k or slug.startswith(k) or k.startswith(slug) for k in klasses) if slug else False
            rows.append({"task": c["task"], "type": c["account_type"], "gold": c["gold"], "arm": arm,
                         "answer": v, "correct": ok, "in_enum": in_enum, "chars": len(mat)})
            print("   %-8s -> %-34s 정답=%-5s 목록안=%-5s %6d자"
                  % (arm, v[:34], ok, in_enum, len(mat)))

    p = os.path.join(REP, "x451_%s.json" % a.tag)
    with io.open(p, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)
    print("\n" + "=" * 100)
    print("%-10s %-10s %-10s" % ("팔", "정답", "목록안"))
    for arm in arms:
        rs = [r for r in rows if r["arm"] == arm]
        if rs:
            print("%-10s %-10s %-10s"
                  % (arm, "%d/%d" % (sum(1 for r in rs if r["correct"]), len(rs)),
                     "%d/%d" % (sum(1 for r in rs if r["in_enum"]), len(rs))))
    print("→ %s" % p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
