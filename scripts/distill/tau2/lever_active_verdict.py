# -*- coding: utf-8 -*-
r"""레버가 **발화한 sim 만으로** 다시 계산한다 (2026-08-21·오프라인·LLM 0)

## 왜 (사용자 지시 *"발화한 sim 만 다시 계산하라"*)
070 포렌식이 드러낸 것: 그 태스크에서 값 주석은 **0회** 발화했고 `value_formula` 는 A2 에서
`check_card_application_fit` **하나에만** 선언돼 있다. 즉 070 에서 `val` 팔과 `ctl` 팔은 **거동상
동일**하고, `val 070 = 1.0` ↔ `ctl 070 = 0/2` 는 **레버 효과가 아니라 런-간 변동**이다.

⇒ 팔은 **레버가 발화한 sim 에서만** 다르다. 발화 안 한 sim 은 두 팔이 바이트 동일이므로 총계에
  섞으면 **희석**된다. 여기서는 sim 단위로 발화 여부를 세고, **발화 sim 에 한정한 짝**을 낸다.

## 발화 판정 (닫힌 술어 — 로그 마커 문자열의 존재)
    값 주석   `documented_return_for_stated_spend` 가 그 sim 블록에 있다
    배달      `T2_ARG_DOC_SUB` 가 그 sim 블록에 있다
⚠로그 마크는 **전달했다는 뜻이지 효과가 있었다는 뜻이 아니다**([[55]]) — 여기서는 *"두 팔이
  갈릴 수 있었나"* 만 판정한다. 그 이상은 주장하지 않는다.
⚠sim 블록 경계는 `[sim=<task>#<seed>]` 접두사로 자른다. 같은 태스크의 다른 trial 이 같은 seed
  라벨을 쓰면 합쳐질 수 있다 — 그래서 **태스크 단위**로도 함께 인쇄한다.
"""
import collections
import glob
import gzip
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

BASE = os.path.abspath(os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026", "sim_results"))
M_VAL = "documented_return_for_stated_spend"
M_DOC = "T2_ARG_DOC_SUB"

ARMS = {
    "ctl":   ["bank_t7333_ctl_hot_20260821c", "bank_t7333_ctl_rest_20260821c"],
    "val":   ["bank_t7334_val_hot_20260821", "bank_t7334_val_rest_20260821"],
    "treat": ["bank_t7333_treat_hot_20260821c", "bank_t7333_treat_rest_20260821c"],
}


def results(tag):
    p = os.path.join(BASE, tag + ".results.json.gz")
    if os.path.exists(p):
        return json.load(gzip.open(p, "rt", encoding="utf-8")).get("simulations") or []
    return []


def logtext(tag):
    p = os.path.join(BASE, tag + ".log.gz")
    if os.path.exists(p):
        return gzip.open(p, "rt", encoding="utf-8", errors="replace").read()
    return ""


def fired_tasks(tag):
    """그 런에서 **어느 태스크가** 마커를 냈나 (sim 블록 → 태스크)."""
    t = logtext(tag)
    out = {"val": collections.Counter(), "doc": collections.Counter()}
    for b in re.split(r"(?=\[sim=task_)", t):
        m = re.match(r"\[sim=(task_\d+)#", b)
        if not m:
            continue
        tid = m.group(1)
        if M_VAL in b:
            out["val"][tid] += 1
        if M_DOC in b:
            out["doc"][tid] += 1
    return out


def main():
    per = {}
    fire = {}
    for arm, tags in ARMS.items():
        sims = []
        for t in tags:
            sims.extend(results(t))
        per[arm] = sims
        f = {"val": collections.Counter(), "doc": collections.Counter()}
        for t in tags:
            g = fired_tasks(t)
            f["val"].update(g["val"])
            f["doc"].update(g["doc"])
        fire[arm] = f
        if not sims:
            print("⚠%s: 결과 없음(아직 영속 전일 수 있다)" % arm)

    print("=" * 96)
    print("레버 발화 기준 재계산 · ctl %d sim · val %d sim · treat %d sim"
          % (len(per["ctl"]), len(per["val"]), len(per["treat"])))
    print("=" * 96)

    tasks = sorted({str(s.get("task_id")) for a in per for s in per[a]})
    print("\n[태스크별] 발화(treat 기준) · 팔별 pass")
    print("%-11s %-8s %-8s %-9s %-9s %-9s" % ("태스크", "값발화", "배달발화", "ctl", "val", "treat"))
    active, inactive = [], []
    for tid in tasks:
        fv = fire["treat"]["val"][tid]
        fd = fire["treat"]["doc"][tid]
        row = []
        for arm in ("ctl", "val", "treat"):
            ss = [s for s in per[arm] if str(s.get("task_id")) == tid]
            p = sum(1 for s in ss if ((s.get("reward_info") or {}).get("reward") or 0) >= 1.0)
            row.append((p, len(ss)))
        print("%-11s %-8s %-8s %-9s %-9s %-9s"
              % (tid, fv or "-", fd or "-",
                 "%d/%d" % row[0], "%d/%d" % row[1], "%d/%d" % row[2]))
        (active if (fv or fd) else inactive).append((tid, row))

    def tot(group, i):
        return (sum(r[i][0] for _t, r in group), sum(r[i][1] for _t, r in group))

    print("\n★[레버가 발화한 태스크만]  (%s)" % ", ".join(t for t, _r in active))
    print("   ctl %d/%d · val %d/%d · treat %d/%d"
          % (tot(active, 0) + tot(active, 1) + tot(active, 2)))
    print("\n[발화 0 인 태스크 — 두 팔이 거동상 동일, 차이는 잡음]  (%s)"
          % ", ".join(t for t, _r in inactive))
    print("   ctl %d/%d · val %d/%d · treat %d/%d"
          % (tot(inactive, 0) + tot(inactive, 1) + tot(inactive, 2)))
    print("\n[전체]")
    allg = active + inactive
    print("   ctl %d/%d · val %d/%d · treat %d/%d"
          % (tot(allg, 0) + tot(allg, 1) + tot(allg, 2)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
