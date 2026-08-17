# -*- coding: utf-8 -*-
r"""x364 — **자격축**(개인/사업자)을 격리로 먼저 잰다. 사용자 지시 2026-08-17 축자:

    *"상류에서 **자격축**은 꼭 클래스 결정에 도움이 된다. 꼭 확인해서 넣어라.
      **하류 클래스 결정에 참조하게 하라**."*

⛔0번 규칙([[62]]): **넣기 전에 잰다.** 이 프로브가 재는 것은 셋이다 — ①자격 한 줄을 만들 수
있는가 ②상류(군 선택)에 결손이 남아 있는가 ③하류(클래스 결정)에서 그 줄이 답을 바꾸는가.

## ⚠먼저: 이 축을 세운 근거가 오늘 **바뀌었다** (2026-08-18 라이브 로그 전수 재계수)

핸드오프 §0 의 근거는 *"요청 밖 군 29회 중 19회(66%)가 개인 손님에게 `business_*`"* 였다.
t7305 두 팔 로그의 `[T2_DOCGROUP]` **125줄 전수**를 다시 세면:

    business_* 를 고른 군 결정            49회
    그중 모델이 **개인 군도 같이 답했는데** 우리 필터가 지운 것   **48회 (98%)**

즉 그 자리는 **모델의 결손이 아니라 우리 배관**(C516 포함관계 필터)이었고 이미 수리됐다([[55]]).
그래서 이 프로브는 **수리 뒤의 잔여 결손**을 다시 잰다 — 결손이 0 이면 자격축은 그 자리에서
**살 것이 없다**(그건 좋은 소식이고, 레버를 안 짓는 것이 정답이다).
⚠블록 ②는 정본 `t2_search.groups_in` 을 **부른다**(사본 금지·[[67]]). `x361` 은 옛 필터를 베껴
  갖고 있어 자기 자신이 개인 군을 지웠다 — 그 수치는 이 프로브가 대체한다.

## 블록 ① 형식화 — 자격 한 줄을 만들 수 있나 (없으면 축 폐기)

재료원 3종(전부 라이브에서 에이전트가 실제로 갖는 것):

    F_DB    손님 DB 레코드(users 행 + 그 손님 계좌 행) — `doc_index` 무관·도메인 저작 0
    F_SAY   손님 자신의 말(대본 축자)
    F_BOTH  둘 다

LLM 이 `INDIVIDUAL`/`BUSINESS` 를 답하고 **근거 한 줄을 원문에서 인용**한다 → 엔진은 `quote_in`
으로 **인용 실재만** 검산한다([[66]]·C45 동형·판정은 LLM·엔진 판단 0).
채점 라벨(분석 전용·[[23]] 레버 무관): gold 축의 군이 `business_` 로 시작하면 BUSINESS.
⚠caveat: 사업자가 **개인** 상품을 물을 수 있다 — 라벨 불일치는 그 자체로 오답이 아니므로 표에
  태스크를 축자로 인쇄한다.

## 블록 ② 상류(군 선택) — 수리 뒤 잔여 결손 + 자격 한 줄의 효과

    A_GRP  라이브 축자(A2 `group_prompt` · 손님 말)                    ← 기준선
    B_GRP  같은 것 + **자격 한 줄을 머리에**                            ← 레버

    HIT       요청 군(gold 축이 요구하는 군)을 골랐나
    EXTRA     요청 밖 군 개수                     ← 낮을수록 좋다
    EXTRA_BIZ 그중 `business_*`                   ← 이 축이 산다고 주장한 바로 그 조각

## 블록 ③ 하류(클래스 결정) — 사용자 지시의 본체

`x357 v2` 가 남긴 **판정 줄(L-V)을 그대로 재사용**한다(같은 재료·같은 태스크·재계산 0):

    A_REF   판정 줄만                       ← 현행 최선(L-V)
    B_ELIG  판정 줄 + **자격 한 줄을 머리에**  ← 레버
    D_NEG   판정 줄 + **자격을 뒤집은 줄**    ← 부정통제(내용이 읽히는가)

요구원 2종(리뷰 ③): `_S` = 대본 전문 · `_L` = 라이브형 인용(A2 `requirement_prompt` 산출).

## 판정 (사전 고정 · 결과보다 **먼저** 인쇄된다)

    ①  라벨 적중 ≥20/24 ∧ 인용 검산 ≥20/24 (어느 재료원이든)  → 자격 한 줄이 **만들어진다**
        어느 재료원도 미달                                    → ⛔축 폐기(형식화 불가)
    ②  A_GRP 의 EXTRA_BIZ = 0                                → 상류 결손 **없음**(수리가 닫았다)
        EXTRA_BIZ > 0 이고 B_GRP 가 그것을 줄이며 HIT 손실 0    → 상류에서 산다
        B_GRP 의 EXTRA 가 A 보다 크다                          → 폐기(과잉)
    ③  표적군 B_ELIG − A_REF ≥ +2 ∧ D_NEG ≤ A_REF            → 선별 통과(합성 후보)
        D_NEG ≥ B_ELIG                                       → ⛔이득이 **내용이 아니다** ⇒ 폐기
        회귀군 B_ELIG < A_REF − 1                             → ⛔해악 ⇒ 승격 금지

실행(리모트·무료 포트 8141):
  T2_PROBE_URL=http://localhost:8141/v1/chat/completions \
  /home/woori/venvs/seka_env/bin/python x364_eligibility_axis_iso.py [part] [nparts]
"""
import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/home/woori/scratch/tau2-bench/src")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from x216_read_and_offset import chat                             # noqa: E402
import t2_probe as P                                              # noqa: E402
import t2_search as TS                                            # noqa: E402
import x341_docbody_verdict as X341                               # noqa: E402
import x351_order_lever_iso as X                                  # noqa: E402
import x357_verdict_carry_multitask as M                          # noqa: E402

REPORTS = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "..", "..", "..", "reports", "facet_rft_2026"))
DB = os.path.normpath(os.path.join(M.TASKS_DIR, "..", "db.json"))

ELIG = ("Below is what this bank has on file for a customer.\n\n{record}\n\n"
        "Is this customer an individual person or a business entity? Answer on the first line "
        "with INDIVIDUAL or BUSINESS. On the second line, quote VERBATIM one line from the text "
        "above that decides your answer. Nothing else.")
LINE = "This customer is a {v} customer — \"{q}\""
PICK = ("{lines}\n\nWhich document groups does the customer ask about? List every group that one "
        "of their requests is about, and no others. Reply with group names only.")


def db_records():
    """손님 → **DB 레코드 축자**(users 행 + 그 손님 계좌 행). 에이전트가 도구로 받는 것과 같은 꼴.

    ⚠gold 를 안 본다([[23]]): 손님을 찾는 열쇠는 **DB 의 이름이 대본에 나오는가**뿐이다 —
      라이브 에이전트가 이름/DOB 로 조회하는 것과 같은 경로다. 못 찾으면 그 태스크는 **뺀다**.
    """
    d = json.load(io.open(DB, encoding="utf-8"))
    users = (d.get("users") or {}).get("data") or {}
    accts = (d.get("accounts") or {}).get("data") or {}
    cards = (d.get("credit_card_accounts") or {}).get("data") or {}
    out = {}
    for uid, row in users.items():
        mine = [a for a in accts.values() if str(a.get("user_id")) == str(uid)]
        mine += [a for a in cards.values() if str(a.get("user_id")) == str(uid)]
        out[uid] = {"name": str(row.get("name") or ""),
                    "text": json.dumps({"customer": row, "accounts_on_file": mine},
                                       ensure_ascii=False, indent=1)}
    return out


def det_ask(body, maxtok=200):
    """온도 0 ×2 — 같으면 확정(n=1), 다르면 비결정으로 표시한다(t2_probe.run 규약 동형)."""
    a = str((chat(body, None, 0.0, maxtok) or {}).get("content") or "")
    b = str((chat(body, None, 0.0, maxtok) or {}).get("content") or "")
    return a, (a.strip() == b.strip())


def elig_of(ans, text):
    """첫 낱말 = 자격 · 둘째 줄 이하 = 근거 인용(원문 실재만 검산). 엔진 판단 0."""
    ls = [x.strip().strip('"').strip() for x in str(ans or "").split("\n") if x.strip()]
    if not ls:
        return None, ""
    head = ls[0].upper().replace("*", "").strip()
    v = ("INDIVIDUAL" if head.startswith("INDIVIDUAL")
         else ("BUSINESS" if head.startswith("BUSINESS") else None))
    q = next((l for l in ls[1:] if TS.quote_in(l, text)), "")
    return v, q


def flip(v):
    return "BUSINESS" if v == "INDIVIDUAL" else "INDIVIDUAL"


def carried(res, key):
    return sum(1 for r in res if r["arms"].get(key, 0) > 0)


def main():
    part = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    nparts = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    a2 = X.a2_load()
    po = a2.get("policy_ontology") or {}
    gtpl, dtpl = str(po.get("group_prompt") or ""), str(po.get("doc_decide_prompt") or "")
    names_g = sorted((po.get("doc_index") or {}).keys())
    if not (gtpl and dtpl and names_g):
        print("A2 group_prompt/doc_decide_prompt/doc_index 없음 — 중단(계기 결함)")
        return 1
    listing = "\n".join("  %s" % g for g in names_g)

    # ── x357 v2 가 남긴 판정 줄(재계산 0). 없으면 블록 ③은 건너뛴다(조용히 넘기지 않는다).
    jobs357 = []
    for fn in sorted(os.listdir(REPORTS)):
        if fn.startswith("x357v2_part") and fn.endswith(".json"):
            jobs357 += (json.load(io.open(os.path.join(REPORTS, fn), encoding="utf-8"))
                        or {}).get("res") or []
    recs = db_records()

    # ── 태스크 목록 (블록 ①②의 단위 = 태스크 · 블록 ③의 단위 = 축)
    tasks = {}
    for fn in sorted(os.listdir(M.TASKS_DIR)):
        if not fn.endswith(".json"):
            continue
        tid = fn[:-5]
        if tid in M.EXCLUDE:
            continue
        req = M.instructions(tid)
        axes = X341.gold_axes(tid)
        want = sorted(set(g for ax, gold in axes.items() for g in [M.group_for(ax, gold)] if g))
        if not (req and want):
            continue
        uid = next((u for u, r in recs.items() if r["name"] and r["name"] in req), None)
        tasks[tid] = {"task": tid, "req": req, "want": want, "uid": uid,
                      "label": "BUSINESS" if any(g.startswith("business_") for g in want)
                               else "INDIVIDUAL"}
    keys = [t for i, t in enumerate(sorted(tasks)) if i % nparts == part]
    with_db = sum(1 for t in keys if tasks[t]["uid"])
    print("x364 · 조각 %d/%d · 태스크 %d개(DB 레코드 매칭 %d) · x357v2 축 %d개 · 군 후보 %d개"
          % (part, nparts, len(keys), with_db, len(jobs357), len(names_g)))
    print("판정(사전 고정): ①라벨 ≥20/24 ∧ 인용 ≥20/24 → 형식화 가능 · 전부 미달 → 축 폐기 · "
          "②A_GRP EXTRA_BIZ=0 → 상류 결손 없음 · ③표적군 B_ELIG−A_REF ≥+2 ∧ D_NEG ≤ A_REF → "
          "선별 통과 · D_NEG ≥ B_ELIG → 이득이 내용이 아님(폐기) · 회귀군 B < A−1 → 해악\n")

    # ══ 블록 ① 형식화 ─────────────────────────────────────────────────────────────
    print("── 블록 ① 자격 형식화 (재료원 3종)")
    form = {}
    for tid in keys:
        t = tasks[tid]
        rec = recs.get(t["uid"] or "", {}).get("text", "")
        srcs = {"F_SAY": t["req"]}
        if rec:
            srcs["F_DB"] = rec
            srcs["F_BOTH"] = rec + "\n\nThe customer says:\n" + t["req"]
        row = {}
        for k in ("F_DB", "F_SAY", "F_BOTH"):
            if k not in srcs:
                continue
            ans, det = det_ask(ELIG.format(record=srcs[k][:6000]), 200)
            v, q = elig_of(ans, srcs[k])
            row[k] = {"v": v, "q": q, "det": det, "ok": int(v == t["label"]), "cited": int(bool(q))}
        form[tid] = {"label": t["label"], "src": srcs, "out": row}
        print("   %-9s gold라벨 %-10s %s" % (tid, t["label"], " · ".join(
            "%s %s%s%s" % (k, row[k]["v"], "" if row[k]["cited"] else "(인용 실패)",
                           "" if row[k]["det"] else "⚠비결정") for k in sorted(row))))

    best = None
    for k in ("F_BOTH", "F_DB", "F_SAY"):
        rows = [form[t]["out"][k] for t in keys if k in form[t]["out"]]
        if not rows:
            continue
        ok, cit = sum(r["ok"] for r in rows), sum(r["cited"] for r in rows)
        print("   %-7s n=%d · 라벨 적중 %d · 인용 검산 통과 %d" % (k, len(rows), ok, cit))
        if best is None or (ok, cit) > best[1]:
            best = (k, (ok, cit))
    if not best:
        print("⛔자격 한 줄을 만들 재료가 없다 — 중단")
        return 1
    SRC = best[0]
    print("   ⇒ 블록 ②③이 쓰는 재료원 = **%s**(라벨 %d·인용 %d)\n" % (SRC, best[1][0], best[1][1]))

    def line_of(tid, negate=False):
        o = (form.get(tid) or {}).get("out", {}).get(SRC) or {}
        if not (o.get("v") and o.get("q")):
            return ""            # 자격 미확정·근거 미검산 → **안 싣는다**(fail-safe·[[25]])
        return LINE.format(v=flip(o["v"]) if negate else o["v"], q=o["q"][:200])

    # ══ 블록 ② 상류(군 선택) ───────────────────────────────────────────────────────
    print("── 블록 ② 상류 군 선택 (정본 `t2_search.groups_in` 파싱)")
    g_res = []
    for tid in keys:
        t = tasks[tid]
        el = line_of(tid)
        sel = {}
        for arm, text in (("A_GRP", t["req"]),
                          ("B_GRP", (el + "\n\n" + t["req"]) if el else "")):
            if not text:
                continue
            raw, det = det_ask(gtpl.format(groups=listing, text=text), 200)
            sel[arm] = {"sel": TS.groups_in(raw, names_g), "det": det, "raw": raw[:200]}
        want = set(t["want"])
        for arm, v in sel.items():
            v["hit"] = int(bool(want & set(v["sel"])))
            v["extra"] = [g for g in v["sel"] if g not in want]
            v["extra_biz"] = [g for g in v["extra"] if g.startswith("business_")]
        g_res.append({"task": tid, "want": t["want"], "label": t["label"], "arms": sel})
        print("   %-9s 요청군 %-34s A %s · B %s"
              % (tid, ",".join(t["want"]),
                 ",".join(sel.get("A_GRP", {}).get("sel") or ["-"]),
                 ",".join(sel.get("B_GRP", {}).get("sel") or ["-"])))

    # ══ 블록 ③ 하류(클래스 결정) ───────────────────────────────────────────────────
    print("\n── 블록 ③ 하류 클래스 결정 (x357v2 판정 줄 재사용)")
    d_res = []
    for j in jobs357:
        tid = j.get("task")
        if tid not in tasks or tid not in keys:
            continue
        el, elneg = line_of(tid), line_of(tid, negate=True)
        if not el:
            continue
        idx = (po.get("doc_index") or {}).get(M.group_for(j["axis"], j["gold"])) or {}
        classes = [c for c in sorted(idx) if c != "_general_"]
        if not classes:
            continue
        cand = str(po.get("decide_candidates_text")).format(
            candidates=", ".join(X.disp(c) for c in classes))
        req_s = X.block([tasks[tid]["req"]])
        arms = []
        for suf, key in (("_S", "script"), ("_L", "live")):
            lines = "\n".join((j.get("lines") or {}).get(key) or ())
            if not lines:
                continue
            for tag, head in (("A_REF", ""), ("B_ELIG", el + "\n\n"), ("D_NEG", elneg + "\n\n")):
                arms.append((tag + suf, dtpl.format(ask=head + req_s + "\n\n" + cand,
                                                    material=lines)))
        if not any(a[0].startswith("A_REF") for a in arms):
            continue
        arms = [("A_REF", a[1]) if a[0] == "A_REF_S" else a for a in arms]   # run() 기준선 요구
        r = P.run("x364-%s-%s" % (tid, j["axis"]),
                  {"tag": "task-def", "task": tid, "cut": 0, "sim": "-", "base": ""},
                  arms, {"GOLD": X.disp(j["gold_class"])},
                  "(판정은 전 축 합산 후·위 문구 그대로)", "", None, 8, 3,
                  det=True, names=[X.disp(c) for c in classes])
        d_res.append({"task": tid, "axis": j["axis"], "gold_class": j["gold_class"],
                      "rate": j.get("rate", 0.0), "elig": el,
                      "arms": dict((k, v["GOLD"][0]) for k, v in (r or {}).items())})

    # ══ 합산 ──────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 96)
    for k in ("F_DB", "F_SAY", "F_BOTH"):
        rows = [form[t]["out"][k] for t in keys if k in form[t]["out"]]
        if rows:
            print("① %-7s n=%-3d 라벨 적중 %-3d 인용 검산 %-3d" %
                  (k, len(rows), sum(r["ok"] for r in rows), sum(r["cited"] for r in rows)))
    for arm in ("A_GRP", "B_GRP"):
        rows = [r["arms"][arm] for r in g_res if arm in r["arms"]]
        if rows:
            print("② %-6s n=%-3d HIT %-3d EXTRA %-3d **EXTRA_BIZ %d**"
                  % (arm, len(rows), sum(x["hit"] for x in rows),
                     sum(len(x["extra"]) for x in rows), sum(len(x["extra_biz"]) for x in rows)))
    for nm, rows in (("표적군(census 0%)", [r for r in d_res if r["rate"] <= 0]),
                     ("회귀군(census >0%)", [r for r in d_res if r["rate"] > 0])):
        if not rows:
            continue
        print("③ %s n=%d · " % (nm, len(rows)) + " · ".join(
            "%s %d" % (k, carried(rows, k)) for k in
            ("A_REF", "B_ELIG_S", "D_NEG_S", "A_REF_L", "B_ELIG_L", "D_NEG_L")))
    out = os.path.join(REPORTS, "x364_part%d.json" % part)
    with io.open(out, "w", encoding="utf-8") as f:
        f.write(json.dumps({"form": dict((t, {"label": form[t]["label"], "out": form[t]["out"]})
                                         for t in form),
                            "groups": g_res, "decide": d_res, "src": SRC},
                           ensure_ascii=False, indent=1, default=str))
    print("저장: %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
