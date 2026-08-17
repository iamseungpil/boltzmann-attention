# -*- coding: utf-8 -*-
r"""x357 — 판정 이월 레버의 **다태스크 확장 + 회귀군**. 선별(n=8)에서 확증(n≈30)으로.

## 왜

`x356b`(055·n=8): `A_REF` 3/8 → **`B_VERDICT` 6/8** · **`D_NEG` 0/8**(통제 유효) ⇒ 선별 통과.
n=8 에선 +3 이 잡음과 안 갈린다(Fisher p≈.31) ⇒ **n 을 키운다**. 마침 같은 축이 있는 태스크가
**35개**다(오늘 전수 계수: card 19·savings 13·checking 9·referral 6·business 6).

★사용자 지적(2026-08-17·축자): *"부정통제 유효라는 말이 다른 pass 하는 태스크에 부정적인 영향을
  안 미친다로 보면 되나?"* — 아니다. 그래서 이 프로브는 **회귀군**을 함께 잰다:

    표적군  census pass **0%** 인 선택축 태스크   ← 열리는가
    회귀군  census pass **>0%** 인 선택축 태스크   ← **떨어지지 않는가**(등대 §1.3 상쇄 계측)

## 셀 (태스크마다 · det n=1 · 라이브 본문 구성)

    A_REF      후보 전원의 문서 전문 (요구 = 대본 전문)         ← 현행
    B_VERDICT  **판정 줄만** (요구 = 대본 전문)                 ← 레버
    A_LIVEREF  문서 전문 (요구 = **① 파이프라인 산출**)         ← 라이브-맞춘 기준선 ★v2
    B_LIVEREQ  판정 줄만 (요구 = **① 파이프라인 산출**)         ← ★v2 진짜 천장(리뷰 ③)
    D_NEG      **무관 요구로 다시 만든** 판정 줄               ← 부정통제(x356 1차 붕괴 수리판)

★v2 수리(리뷰 2026-08-17):
  ① **채점 오염 수리** — `t2_probe._count` 가 단순 포함이라 형제 후보가 gold 를 품으면 오답을
     적중으로 셌다(34축 중 5축). 낱말 경계 + **후보 최장매칭 완전일치**로 바꾸고(`names=`),
     gold 는 **후보 클래스 하나로 확정**한 뒤 채점한다(확정 애매하면 그 축 제외).
  ③ **요구원 정합** — 확증에 쓴 요구가 라이브보다 부유하면 안 된다(등대 §1.4) ⇒ `A_LIVEREF`/
     `B_LIVEREQ` 추가.
  ④ **감사** — 판정 줄 축자와 OK/CONFLICTS/UNCLEAR·근거검산 통과 수를 결과 JSON 에 영속.

요구 텍스트는 **태스크 정의의 손님 대본 축자**(`user_scenario.instructions`)를 쓴다 — gold 가 아니라
손님이 말하는 내용이고, 태스크마다 같은 규칙으로 뽑힌다.
⚠**누설 검사**: 대본에 gold 클래스 이름이 이미 들어 있으면 그 태스크는 **뺀다**(공짜 정답).

## 판정 (사전 고정 · 결과보다 먼저 인쇄)

    표적군 B−A ≥ +5 (n≥20)  ∧ D_NEG ≤ 2 → **확증**: 레버 승격 검토(그 다음이 유료 A/B)
    표적군 B−A ≥ +2         → 선별 유지(더 키운다)
    **회귀군 B < A − 2**    → ⛔**해악**: 통과하던 것을 깬다 ⇒ 승격 금지(조건부 적용만 검토)
    D_NEG ≥ 3               → 계기 무효

실행(리모트·두 포트 분할):
  T2_PROBE_URL=http://localhost:8140/... python x357_verdict_carry_multitask.py 0 2
  T2_PROBE_URL=http://localhost:8141/... python x357_verdict_carry_multitask.py 1 2
  (인자 = 이 조각 번호 / 전체 조각 수 · 결과 JSON 은 조각별로 저장하고 합산은 마지막에)
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
import x355_violation_form_iso as V                               # noqa: E402
import x356_verdict_carry_iso as W                                # noqa: E402

TASKS_DIR = "/home/woori/scratch/tau2-bench/data/tau2/domains/banking_knowledge/tasks"
CENSUS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..",
                      "reports", "facet_rft_2026", "bank_task_taxonomy_20260810.json")
EXCLUDE = {"task_005", "task_102", "task_069"}      # [[68]] 제외·표적금지
PER_DOC = 1800
AX2GROUP = {"checking": "checking_accounts", "savings": "savings_accounts",
            "business_checking": "business_checking_accounts",
            "business_savings": "business_savings_accounts"}


def instructions(tid):
    p = os.path.join(TASKS_DIR, tid + ".json")
    t = json.load(io.open(p, encoding="utf-8"))
    return " ".join(str(((t.get("user_scenario") or {}).get("instructions")) or "").split())


def rates():
    d = json.load(io.open(os.path.normpath(CENSUS), encoding="utf-8"))
    return dict((x["id"], x.get("rate", 0.0)) for x in d)


def gold_class_of(gold, classes):
    """gold 값 → **후보 목록 안의 클래스 하나**로 확정(분석 전용·[[23]] 레버 무관).

    ★리뷰 지적(C515): 채점을 문자열 포함으로 하면 형제가 gold 를 품어 오답을 적중으로 센다
      (`Evergreen` ⊃ `Green` · `Light Blue` ⊃ `Blue`). 그래서 gold 를 **후보 하나로 확정**하고,
      채점은 최장매칭 후보와의 완전일치로 한다. 확정이 애매하면 그 축은 **뺀다**([[25]] 추정 금지)."""
    ns = [(c, X.disp(c)) for c in classes]
    exact = [c for c, d in ns if X341.norm(d) == X341.norm(gold)]
    if len(exact) == 1:
        return exact[0], "exact"
    part = [c for c, d in ns if P._word_in(d, gold)]
    if len(part) == 1:
        return part[0], "unique-substr"
    return None, ("ambiguous:%d" % len(part))


def group_for(ax, gold):
    if ax == "card":
        return "business_credit_cards" if "business" in X341.norm(gold) else "credit_cards"
    return AX2GROUP.get(ax)


def main():
    part = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    nparts = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    a2 = X.a2_load()
    po = a2.get("policy_ontology") or {}
    corpus = {}
    for fn in sorted(os.listdir(X.DOCS)):
        d = json.load(io.open(os.path.join(X.DOCS, fn), encoding="utf-8"))
        corpus[str(d.get("id") or fn)] = str(d.get("content") or "")
    rate = rates()
    tpl = str(po.get("doc_decide_prompt") or "")
    live = lambda ask, mat: tpl.format(ask=ask, material=mat)        # noqa: E731
    neg_sim = X.sim_of(X.AUX, X.NEG_TASK, X.NEG_SEED)
    neg = " ".join(str((neg_sim.get("messages") or [])[X.NEG_MSG].get("content") or "").split())

    # ── 표적 목록 만들기(기계·누설 검사 포함)
    jobs, skipped = [], {"leak": [], "nogroup": [], "nodocs": []}
    for fn in sorted(os.listdir(TASKS_DIR)):
        if not fn.endswith(".json"):
            continue
        tid = fn[:-5]
        if tid in EXCLUDE:
            continue
        axes = X341.gold_axes(tid)
        for ax, gold in axes.items():
            g = group_for(ax, gold)
            if not g:
                skipped["nogroup"].append((tid, ax))
                continue
            idx = (po.get("doc_index") or {}).get(g) or {}
            classes = [c for c in sorted(idx) if c != "_general_"]
            if not classes:
                skipped["nodocs"].append((tid, g))
                continue
            req = instructions(tid)
            if not req:
                continue
            if X341.norm(gold) and X341.norm(gold) in X341.norm(req):
                skipped["leak"].append((tid, ax, gold))     # ★대본이 답을 이미 말한다 → 제외
                continue
            gc, how = gold_class_of(gold, classes)
            if not gc:
                skipped.setdefault("ambiguous", []).append((tid, ax, gold, how))
                continue
            jobs.append({"task": tid, "axis": ax, "gold": gold, "gold_class": gc, "how": how,
                         "group": g, "classes": classes, "req": req, "rate": rate.get(tid, 0.0)})
    jobs = [j for i, j in enumerate(jobs) if i % nparts == part]
    print("x357 · 조각 %d/%d · 표적 %d개(표적군 %d·회귀군 %d) · 누설제외 %d · 그룹없음 %d"
          % (part, nparts, len(jobs), sum(1 for j in jobs if j["rate"] <= 0),
             sum(1 for j in jobs if j["rate"] > 0), len(skipped["leak"]), len(skipped["nogroup"])))
    print("판정(사전 고정): 표적군 B−A ≥+5(n≥20) ∧ D_NEG ≤2 → 확증 · +2 이상 → 선별 유지 · "
          "**회귀군 B < A−2 → 해악(승격 금지)** · D_NEG ≥3 → 계기 무효\n")

    res = []
    for j in jobs:
        idx = (po.get("doc_index") or {}).get(j["group"]) or {}

        def mat_of(cls):
            ids = [d for c in cls for d in (idx.get(c) or ())]
            docs, _m = TS.read_docs(list(dict.fromkeys(ids)), corpus=corpus)
            return TS.as_material(docs, (), per_doc=PER_DOC)

        cand_line = str(po.get("decide_candidates_text")).format(
            candidates=", ".join(X.disp(c) for c in j["classes"]))
        names = [X.disp(c) for c in j["classes"]]      # ★채점: 후보 전체 → 최장매칭(C515)
        marks = {"GOLD": X.disp(j["gold_class"])}
        # ★리뷰 ③(등대 §1.4): 확증에 쓴 요구가 라이브보다 부유하면 안 된다.
        #   R_script = 대본 전문(v1) · **R_live = ① 파이프라인 산출**(A2 프롬프트로 LLM 이 인용을
        #   내고 엔진이 `quote_in` 으로 검산) · R_neg = 무관 요구.
        rq_tpl = str(po.get("requirement_prompt") or "")
        q_live = X.quotes(rq_tpl, j["req"], "q_live/%s" % j["task"]) or []
        reqs = {"script": X.block([j["req"]]),
                "live": X.block(q_live) if q_live else "",
                "neg": X.block([neg])}
        stats = dict((k, {"OK": 0, "CONFLICTS": 0, "UNCLEAR": 0, "cited": 0})
                     for k in ("script", "live", "neg"))
        lines = {"script": [], "live": [], "neg": []}
        for c in j["classes"]:
            m = mat_of([c])
            for key in ("script", "live", "neg"):
                if not reqs[key]:
                    continue
                ans = str((chat(V.VIO.format(req=reqs[key], doc=m[:6000]),
                                None, 0.0, 500) or {}).get("content") or "")
                vv = V.verdict(ans)
                why = W.cite_line(ans, m)
                tag = "CONFLICTS" if vv is True else ("OK" if vv is False else "UNCLEAR")
                stats[key][tag] += 1
                stats[key]["cited"] += 1 if why else 0
                lines[key].append("- %s: %s%s" % (X.disp(c), tag, (" — " + why[:200]) if why else ""))
        arms = [("A_REF", live(reqs["script"] + "\n\n" + cand_line, mat_of(j["classes"]))),
                ("B_VERDICT", live(reqs["script"] + "\n\n" + cand_line, "\n".join(lines["script"])))]
        if reqs["live"]:
            arms.append(("A_LIVEREF", live(reqs["live"] + "\n\n" + cand_line, mat_of(j["classes"]))))
            arms.append(("B_LIVEREQ", live(reqs["live"] + "\n\n" + cand_line,
                                           "\n".join(lines["live"]))))
        arms.append(("D_NEG", live(reqs["neg"] + "\n\n" + cand_line, "\n".join(lines["neg"]))))
        print("── %s/%s · gold=%r(%s) · 후보 %d · census %.2f · 라이브형 인용 %d"
              % (j["task"], j["axis"], X.disp(j["gold_class"]), j["how"], len(j["classes"]),
                 j["rate"], len(q_live)))
        r = P.run("x357-%s-%s" % (j["task"], j["axis"]),
                  {"tag": "task-def", "task": j["task"], "cut": 0, "sim": "-", "base": ""},
                  arms, marks, "(판정은 전 태스크 합산 후·위 문구 그대로)", "", None, 8, 3,
                  det=True, names=names)
        res.append({"task": j["task"], "axis": j["axis"], "gold": j["gold"],
                    "gold_class": j["gold_class"], "how": j["how"], "rate": j["rate"],
                    "n_live_quotes": len(q_live), "stats": stats, "lines": lines,
                    "arms": dict((k, v["GOLD"][0]) for k, v in (r or {}).items())})

    print("\n" + "=" * 96)
    for name, rows in (("표적군(census 0%)", [x for x in res if x["rate"] <= 0]),
                       ("회귀군(census >0%)", [x for x in res if x["rate"] > 0])):
        n = len(rows)
        h = lambda k: sum(1 for x in rows if x["arms"].get(k, 0) > 0)      # noqa: E731
        print("%s n=%d · A_REF %d · **B_VERDICT %d**(Δ%+d) · A_LIVEREF %d · **B_LIVEREQ %d**(Δ%+d)"
              " · D_NEG %d"
              % (name, n, h("A_REF"), h("B_VERDICT"), h("B_VERDICT") - h("A_REF"),
                 h("A_LIVEREF"), h("B_LIVEREQ"), h("B_LIVEREQ") - h("A_LIVEREF"), h("D_NEG")))
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..",
                       "reports", "facet_rft_2026", "x357v2_part%d.json" % part)
    with io.open(os.path.normpath(out), "w", encoding="utf-8") as f:
        f.write(json.dumps({"res": res, "skipped": skipped}, ensure_ascii=False, indent=1,
                           default=str))
    print("저장: %s" % os.path.normpath(out))
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
