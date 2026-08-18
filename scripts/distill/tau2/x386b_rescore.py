# -*- coding: utf-8 -*-
r"""x386b — **x386 재채점**: 정답 라벨을 *판정 줄의 갈림*이 아니라 **표적 유무**로 바꾼다.

## 왜 갈아야 했나 (1차 채점 자기무효)

x386 1차는 *"VIOLATES ≥1 이면 선택 태스크"* 를 정답으로 썼다. **틀렸다** — C535 가 이미 반례를
기록해 뒀는데 내가 그대로 썼다:

  · 무정보(VIOLATES 0)에 **선택 태스크**가 섞인다 — 024 는 *"best return"* 이라 **최대화** 요구이고
    아무 후보도 위반하지 않는다. 그래도 gold 는 `apply_for_credit_card(card_type=…)` 다.
  · 갈림(VIOLATES ≥1)에 **비선택 태스크**가 섞인다 — 040(분쟁)·050(한도 증액)은 손님 요구가
    상품 문서와 부딪혀 VIOLATES 가 뜨지만 **고를 상품이 없다**.

⇒ 판정 줄의 갈림은 **태스크 종류의 대리변수가 아니다**. 모델의 답(라벨+인용)은 정답과 무관하게
수집됐으므로 **채점만** 다시 한다(런 재실행 0).

## 바른 정답 (x377 과 같은 규칙)

그 군의 **후보 표시명이 gold 액션 인자에 실재하는가** = 그 결정이 후보 집합 위에서 이뤄지는가.
⚠gold 는 **판정용 조회**로만 쓴다([[23]]) — 레버 재료로 넘어가지 않는다.

## 판정 (결과 보기 전 재고정)

    표적 있는 컷 CHOOSE ≥80% ∧ 표적 없는 컷 NOT_CHOOSE ≥80% ∧ 인용검산 ≥80% → **게이트 진행**
    한쪽만 충족                                                            → 편향 — 손실 방향 계측 필요
    둘 다 미달                                                              → 라벨 축 폐기 ⇒ A3 절차 선언
"""
import collections
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

ROOT = "/home/woori/scratch/tau2-bench/data/simulations"
LOGS = "/home/woori/scratch/logs"
RAW = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "..", "..", "..", "reports", "facet_rft_2026",
                                    "x386_taskkind_gate.json"))
TAGS = ["bank_t7313_treat_20260818h", "bank_t7312_treat_20260818g",
        "bank_t7310_treat_20260818e", "bank_t7314_treat_20260818j"]


def sims_of(tag):
    p = os.path.join(ROOT, tag, "results.json")
    if not os.path.exists(p):
        return {}
    doc = json.load(io.open(p, encoding="utf-8"))
    return {str(s.get("task_id")): s for s in (doc.get("simulations") or doc.get("results") or [])}


def gold_blob(sim):
    buf = []
    for ck in ((sim.get("reward_info") or {}).get("action_checks") or []):
        a = ck.get("action") or ck
        buf.append(str(a.get("name") or ""))
        for v in (a.get("arguments") or {}).values():
            buf.append(v if isinstance(v, str) else json.dumps(v, ensure_ascii=False))
    return " || ".join(buf).lower()


def cand_names(tag, task, turn):
    """그 컷에서 라이브가 실제로 실은 후보 표시명 — 사이드카 판정 줄 축자."""
    p = os.path.join(LOGS, "fb_%s.jsonl" % tag)
    if not os.path.exists(p):
        return []
    for ln in io.open(p, encoding="utf-8", errors="replace"):
        try:
            r = json.loads(ln)
        except Exception:
            continue
        if (r.get("kind") == "verdict-lines" and str(r.get("simtag", "")).startswith(task)
                and r.get("turn") == turn):
            out = []
            for l in str(r.get("text") or "").splitlines():
                l = l.strip()
                if l.startswith("- ") and ":" in l:
                    out.append(l[2:].split(":", 1)[0].strip())
            return out
    return []


def main():
    rows = json.load(io.open(RAW, encoding="utf-8"))["rows"]
    sims = {}
    for t in TAGS:
        sims[t.split("_")[1] + ("treat" if "treat" in t else "ctl")] = sims_of(t)
    full = {t.split("_")[1]: t for t in TAGS}

    print("=" * 104)
    print("x386b · 재채점(정답 = **표적 유무**) · 컷 %d" % len(rows))
    print("판정(재고정): 표적有 CHOOSE ≥80%% ∧ 표적無 NOT_CHOOSE ≥80%% ∧ 인용검산 ≥80%% → 게이트 진행")
    print("=" * 104)
    agg = collections.Counter()
    out = []
    for r in rows:
        tag = full.get(r["tag"])
        sim = (sims_of(tag) or {}).get(r["task"]) if tag else None
        if sim is None:
            continue
        gb = gold_blob(sim)
        names = cand_names(tag, r["task"], r["turn"])
        hit = [n for n in names if n and n.split(" (")[0].lower() in gb]
        want = "CHOOSE" if hit else "NOT_CHOOSE"
        ok = int(r["kind"] == want)
        agg[(want, "n")] += 1
        agg[(want, "hit")] += ok
        agg[("q", "ok")] += int(r["quote_ok"])
        agg[("q", "n")] += 1
        out.append(dict(r, want=want, correct=ok, target=hit[:2]))
        print("  %-9s %-6s t%-3s VIO=%-2d 표적=%-4s 기대=%-11s 답=%-11s %s | %s"
              % (r["task"], r["tag"], r["turn"], r["vio"], "있음" if hit else "없음",
                 want, r["kind"] or "(실패)", "✓" if ok else "✗",
                 (", ".join(hit[:2]))[:34]))

    print("")
    a, an = agg[("CHOOSE", "hit")], agg[("CHOOSE", "n")]
    b, bn = agg[("NOT_CHOOSE", "hit")], agg[("NOT_CHOOSE", "n")]
    q = agg[("q", "ok")] / max(1, agg[("q", "n")])
    print("## 집계  표적有 CHOOSE %d/%d · 표적無 NOT_CHOOSE %d/%d · 인용검산 %.0f%%"
          % (a, an, b, bn, 100 * q))
    ra = a / an if an else 0
    rb = b / bn if bn else 0
    if an and bn and ra >= 0.8 and rb >= 0.8 and q >= 0.8:
        v = "**게이트 진행** — 라벨이 표적 유무를 가른다"
    elif (an and ra >= 0.8) or (bn and rb >= 0.8):
        v = ("편향 — %s 쪽만 충족(CHOOSE %.0f%% · NOT_CHOOSE %.0f%%). 손실 방향 계측 필요"
             % ("CHOOSE" if ra >= 0.8 else "NOT_CHOOSE", 100 * ra, 100 * rb))
    else:
        v = "라벨 축 폐기 ⇒ **A3 절차 선언**(③)"
    print("판정: %s" % v)
    p = RAW.replace(".json", "_rescored.json")
    io.open(p, "w", encoding="utf-8").write(json.dumps({"rows": out, "verdict": v},
                                                       ensure_ascii=False, indent=1))
    print("원자료: %s" % p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
