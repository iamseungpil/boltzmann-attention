# -*- coding: utf-8 -*-
r"""x455 — 감사가 뽑은 원시 축 이름을 **정본 축으로 병합** (2026-08-21·무료·A3 완결 저작)

## 왜 (사용자 지시 2026-08-21 축자)
*"판단을 하는게 왜 나쁜가? 비용 성능 측면에서 한번 하는게 가장 효율적이면 그렇게 하면 된다.
그걸 자꾸 피할려고 하니까 해결이 안되는 거다. 우리 특허에 기술된대로 비용측면에서 그게 제일
싸면 완벽한 a2 a3 한번 만드는게 낫다."* ⇒ [[72]].

`x453` 은 코퍼스가 **값을 명시하는 속성**과 **정책이 요건이라 말한 축**을 전수로 뽑는다. 그런데
이름이 LLM 슬러그라 **같은 축이 여러 이름으로 갈린다** — 3계열 실측에서:

    minimum_balance_requirement 9클래스 · minimum_balance 4 · ongoing_minimum_balance 4
    minimum_balance_to_maintain_the_account 2 · balance 2 · daily_balance 3   ← 전부 같은 축

갈린 채로 A3 에 실으면 ⑴추출 프롬프트의 속성 목록이 수백 줄로 부풀어 정밀도가 떨어지고
⑵같은 축이 여러 칸에 흩어져 판정이 불가능하다. 그래서 **병합이 저작의 본체**다.

## 분담 ([[10]]·[[59]] — 엔진은 뜻을 안 본다)
    LLM    같은 축을 가리키는 이름들을 **묶는다**(정본명 + 문서가 쓰는 축자 표현)
    엔진   배치 · **분할 검산**(모든 입력 이름이 정확히 한 군에) · **축자 실재 검산** · 집계
⛔엔진이 이름 문자열을 비교해 묶지 않는다(`minimum` 이 겹치니 같은 축이다 = [[59]] 위반).

## 채택 규칙 (결과 보기 전에 고정 · x453 의 규칙을 군 단위로 올린 것뿐)
    · 병합은 **관측된 전체 이름**에 대해 한다 — `adopt` 만 묶으면 `..._to_maintain_the_account`(2클래스)
      같은 조각이 큰 축에 붙지 못하고 버려진다.
    · **군의 채택** = 그 군의 **어느 구성원이라도** x453 의 채택 기준을 만족하면 채택
      (정책이 요건이라 말함 ∪ 값이 `--minclasses` 이상 클래스에서 명시됨).
    · 축마다 문서 id + 축자 인용을 남긴다. 못 대면 넣지 않는다.
    ⛔gold·태스크·실패 사례는 보지 않는다. 이 스크립트는 `sim_results` 를 읽지 않는다.

## 산출
`x455_axis_groups.json` — 군 전체(검토용) + `catalog_attrs_candidate`(A3 형식 그대로).
**A3 병합은 사람이 1회 검토한 뒤 별도로** 한다(정본 §4 절차·두 층 동기화 [[24]]).

사용: (리모트·cwd=tau2 · PYTHONPATH=src:…) py x455_axis_consolidate.py --port 8141
"""
import argparse
import collections
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import x430_account_facts as FT         # noqa: E402  DOCDIR
import x431_spec_selects as X           # noqa: E402  ask 정본(사본 금지·[[67]])
import x452_conditional_facts as C      # noqa: E402  선언 읽기·docs_by_class·contained

REP = os.path.abspath(os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026"))
NLC = chr(10)

SYS = (
    "You group attribute names that were extracted from retail-bank product documents. "
    "Different names can denote the SAME underlying attribute of an account (for example, a "
    "balance the customer must keep to hold the account). Group by what the quoted policy "
    "sentence is about, not by how the name is spelled.\n"
    "Reply with ONE JSON object:\n"
    '{"groups": [{"canonical": "<short snake_case name for the attribute>", '
    '"members": ["<an input name, copied exactly>", ...], '
    '"aliases": ["<a phrase copied character-for-character out of one of the quotes shown for '
    'this group, that names the attribute>", ...]}]}\n'
    "Rules: every input name must appear in exactly one group; never invent an input name that "
    "was not shown; a name with no partner forms a group of one; each alias must be copied "
    "verbatim from a quote shown for that group - never paraphrase and never write a name that "
    "is not in the documents."
)

SYS_MERGE = (
    "You are merging attribute names that came from different batches, so the same attribute may "
    "appear more than once under different names. Same rules as before: group by what the quoted "
    "policy sentence is about.\n"
    "Reply with ONE JSON object:\n"
    '{"groups": [{"canonical": "<short snake_case name>", "members": ["<an input name, copied '
    'exactly>", ...], "aliases": ["<phrase copied verbatim from a quote shown for this group>", '
    "...]}]}\n"
    "Every input name must appear in exactly one group; never invent one."
)


def _ex(aud, name):
    """그 이름의 대표 예시 — 값 예시가 없으면 요건 예시로 (x453 의 두 사전)."""
    e = (aud.get("example") or {}).get(name)
    if e:
        return {"class": e.get("class"), "value": e.get("value"), "quote": e.get("quote")}
    r = (aud.get("req_example") or {}).get(name)
    if r:
        return {"class": r.get("class"), "value": r.get("requirement"), "quote": r.get("quote")}
    return None


def _batch_body(aud, names):
    out = []
    for n in names:
        e = _ex(aud, n) or {}
        ncls = len(set((aud.get("observed") or {}).get(n) or [])
                   | set((aud.get("requirements") or {}).get(n) or []))
        out.append("- %s  (%d classes)  value: %s\n  quote: %s"
                   % (n, ncls, str(e.get("value"))[:60], str(e.get("quote"))[:200]))
    return "# Attribute names to group\n" + "\n".join(out) + "\n"


def token_batches(names, size):
    """배치를 **형태만**으로 묶는다 — 같은 토큰을 가진 이름이 한 배치에 오도록(뜻 0·`slug` 동형).

    ⚠1차 실행 실측(2026-08-21): `sorted(names)` 로 자르니 40개 → 평균 **29군**으로 거의 안
    접혔다. 동의어가 알파벳순으로 멀어 **같은 배치에서 만날 기회 자체가 없었기** 때문이다
    (`balance` · `daily_balance` · `minimum_balance` · `ongoing_minimum_balance`).

    배치는 **후보를 같이 보여줄 뿐** 묶는 판단은 LLM 이 한다 — 엔진은 이름의 문자열 토큰만
    보고 누구를 나란히 놓을지 정할 뿐, 무엇이 같은 축인지는 말하지 않는다([[59]]).
    분할은 유지된다: 한 이름은 정확히 한 배치에만 들어간다.
    """
    idx = collections.defaultdict(list)
    for n in names:
        for t in set(str(n).split("_")):
            if t:
                idx[t].append(n)
    # 토큰 군을 만들고(같은 토큰 = 붙어 있음), **작은 군은 한 배치에 채워 넣는다**.
    # ⚠군마다 배치를 하나씩 내면 1,392 이름이 153 배치가 된다(평균 9개) — 호출만 많고 일이 적다.
    used, cells = set(), []
    for _t, grp in sorted(idx.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        pend = [n for n in grp if n not in used]
        while len(pend) >= 2:
            cell, pend = pend[:size], pend[size:]
            cells.append(cell)
            used.update(cell)
    rest = [n for n in names if n not in used]
    cells += [rest[i:i + size] for i in range(0, len(rest), size)]
    out, cur = [], []
    for cell in cells:
        if cur and len(cur) + len(cell) > size:
            out.append(cur)
            cur = []
        cur += cell
    if cur:
        out.append(cur)
    assert sum(len(c) for c in out) == len(names)
    return out


def _parse_groups(got, allowed):
    """LLM 답에서 군을 꺼내되 **입력에 없던 이름은 버린다**(날조 차단·엔진 판단 0)."""
    gs = []
    for g in ((got or {}).get("groups") or []):
        if not isinstance(g, dict):
            continue
        can = str(g.get("canonical") or "").strip()
        mem = [str(m).strip() for m in (g.get("members") or []) if str(m).strip() in allowed]
        als = [" ".join(str(a).split()) for a in (g.get("aliases") or []) if str(a).strip()]
        if can and mem:
            gs.append({"canonical": can, "members": mem, "aliases": als})
    return gs


def _assign_once(groups, names):
    """분할 검산: 한 이름이 여러 군에 들어가면 **첫 군만** 남기고, 어디에도 없으면 홑군으로."""
    seen, dup = set(), []
    for g in groups:
        keep = []
        for m in g["members"]:
            if m in seen:
                dup.append(m)
            else:
                seen.add(m)
                keep.append(m)
        g["members"] = keep
    groups = [g for g in groups if g["members"]]
    missing = [n for n in names if n not in seen]
    for n in missing:
        groups.append({"canonical": n, "members": [n], "aliases": [], "_singleton": True})
    return groups, dup, missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8141)
    ap.add_argument("--audit", default="x453_attr_coverage_all.json")
    ap.add_argument("--out", default="x455_axis_groups.json")
    ap.add_argument("--minclasses", type=int, default=5)
    ap.add_argument("--batch", type=int, default=40)
    a = ap.parse_args()

    with io.open(os.path.join(REP, a.audit), encoding="utf-8") as f:
        aud = json.load(f)
    observed = aud.get("observed") or {}
    reqs = aud.get("requirements") or {}
    names = sorted(set(observed) | set(reqs))
    print("=" * 96)
    print("x455 · 입력 원시 축 %d종 (관측 %d · 요건 %d) · 배치 %d"
          % (len(names), len(observed), len(reqs), a.batch))
    print("=" * 96)

    # ── 1차: 배치별 병합 ──────────────────────────────────────────────────────
    groups, audit_dup, audit_miss = [], [], []
    _batches = token_batches(names, a.batch)
    print("1차 배치 %d개 (토큰 공유로 묶음 — 동의어가 같은 배치에 오게)" % len(_batches))
    for bi, chunk in enumerate(_batches):
        got = X.ask(a.port, SYS, _batch_body(aud, chunk), maxtok=2400) or {}
        gs = _parse_groups(got, set(chunk))
        gs, dup, miss = _assign_once(gs, chunk)
        groups.extend(gs)
        audit_dup.extend(dup)
        audit_miss.extend(miss)
        print("  배치 %3d/%3d  이름 %2d → 군 %2d  (중복배정 %d · 미배정 %d)"
              % (bi + 1, len(_batches), len(chunk), len(gs), len(dup), len(miss)))

    # ── 2차: 배치를 넘는 중복 병합 (정본명끼리 다시 묶는다) ──────────────────
    rep = {}
    for g in groups:
        for m in g["members"]:
            e = _ex(aud, m)
            if e and e.get("quote"):
                rep.setdefault(g["canonical"], e)
                break
        rep.setdefault(g["canonical"], {"value": "", "quote": ""})
    cans = sorted({g["canonical"] for g in groups})
    print("\n1차 군 %d → 2차 병합 입력 %d" % (len(groups), len(cans)))
    merged = []
    _mb = token_batches(cans, a.batch)
    for bi, chunk in enumerate(_mb):
        body = "# Attribute names to group\n" + "\n".join(
            "- %s\n  quote: %s" % (n, str((rep.get(n) or {}).get("quote"))[:200]) for n in chunk)
        got = X.ask(a.port, SYS_MERGE, body + "\n", maxtok=2400) or {}
        gs = _parse_groups(got, set(chunk))
        gs, dup, miss = _assign_once(gs, chunk)
        merged.extend(gs)
        audit_dup.extend(dup)
        audit_miss.extend(miss)
        print("  2차 배치 %3d/%3d  %2d → %2d  (중복 %d · 미배정 %d)"
              % (bi + 1, len(_mb), len(chunk), len(gs), len(dup), len(miss)))

    # 1차 군을 2차 군으로 접는다 (정본명 → 최종 정본명)
    fold = {}
    for g in merged:
        for m in g["members"]:
            fold[m] = g["canonical"]
    final = collections.defaultdict(lambda: {"members": [], "aliases": []})
    for g in groups:
        top = fold.get(g["canonical"], g["canonical"])
        final[top]["members"].extend(g["members"])
        final[top]["aliases"].extend(g.get("aliases") or [])
    for g in merged:
        final[g["canonical"]]["aliases"].extend(g.get("aliases") or [])

    # ── 검산: alias 가 코퍼스에 축자 실재하나 (닫힌 술어·정본 quote_in) ──────
    fams = C.declared_families()
    byc = C.docs_by_class(fams)
    corpus = " ".join(" ".join(t.split()) for cl in byc for _i, t in byc[cl])
    print("\n검산 코퍼스: 계열 %d · 클래스 %d · %d자" % (len(fams), len(byc), len(corpus)))

    out_groups, alias_drop = [], []
    for can, g in sorted(final.items()):
        mem = sorted(set(g["members"]))
        als = []
        for al in sorted(set(g["aliases"])):
            if C.contained(al, corpus):
                als.append(al)
            else:
                alias_drop.append((can, al))
        cls = set()
        req = False
        for m in mem:
            cls |= set(observed.get(m) or []) | set(reqs.get(m) or [])
            req = req or bool(reqs.get(m))
        ex = None
        for m in mem:
            ex = ex or _ex(aud, m)
        adopt = req or len(cls) >= a.minclasses
        out_groups.append({"canonical": can, "members": mem, "aliases": als,
                           "n_classes": len(cls), "classes": sorted(cls),
                           "is_requirement": req, "adopt": bool(adopt and ex), "example": ex})

    kept = [g for g in out_groups if g["adopt"]]
    # A3 `catalog_attrs` 형식 그대로 (type/unit_votes 는 x430 재실행이 채운다 — 여기선 비운다)
    cand = {}
    for g in sorted(kept, key=lambda x: -x["n_classes"]):
        cand[g["canonical"]] = {
            "aliases": g["aliases"],
            "n_documented": g["n_classes"],
            "is_requirement": g["is_requirement"],
            "example": g["example"],
            "_merged_from": g["members"],
        }

    payload = {"audit": a.audit, "minclasses": a.minclasses,
               "n_input_names": len(names), "n_groups": len(out_groups),
               "n_adopted": len(kept), "groups": out_groups,
               "alias_rejected": alias_drop,
               "llm_double_assigned": sorted(set(audit_dup)),
               "llm_unassigned_singletons": sorted(set(audit_miss)),
               "catalog_attrs_candidate": cand}
    p = os.path.join(REP, a.out)
    with io.open(p, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    print("\n[산출물 선기록] → %s" % p)

    print("\n" + "=" * 96)
    print("원시 %d종 → 군 %d → **채택 %d**  (alias 검산 탈락 %d)"
          % (len(names), len(out_groups), len(kept), len(alias_drop)))
    print("\n[채택 축 · 클래스 수 내림차순]")
    for g in sorted(kept, key=lambda x: -x["n_classes"])[:60]:
        print("  %-34s %2d클래스 %s  ← %d개 이름  %s"
              % (g["canonical"][:34], g["n_classes"], "요건" if g["is_requirement"] else "    ",
                 len(g["members"]), ", ".join(g["members"][:4])[:70]))
    print("\n[채택 안 된 군 %d — 요건도 아니고 %d클래스 미만]" % (len(out_groups) - len(kept), a.minclasses))
    print("  " + ", ".join(g["canonical"] for g in out_groups if not g["adopt"])[:600])
    return 0


if __name__ == "__main__":
    sys.exit(main())
