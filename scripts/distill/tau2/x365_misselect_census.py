# -*- coding: utf-8 -*-
r"""x365 — **오선택(mis-selection) 기준 97 태스크 전수 분류** (사용자 지시 2026-08-17 축자:

    *"referral 군과 카드 banking class 선택군 등 오선택 관련된 기준으로 97 태스크를 분류하여
      L1~L6 레버로 안 되는 새로운 레버가 필요한 부분이 있는지 확인하라"*)

## 무엇을 세는가 (전부 기계·LLM 호출 0·비용 0)

태스크마다 **고를 것이 있는가**, 있다면 **무엇 중에서** 고르는가를 gold 액션 꼴로 읽는다
(`x341.gold_axes` 정본 재사용). 축 종류:

    class-축   `open_bank_account`     → 계좌 클래스 하나 (checking·saving·business_*)
    card-축    `apply_for_credit_card` → 카드 클래스 하나 (개인·사업자)
    ref-축     `submit_referral`       → **추천 계좌 타입** 하나 ← 군 매핑이 **없다**

그리고 각 축에 대해 **후보 집합이 어디서 오는가**를 본다:

    군 매핑 O  A3 `doc_index[group]` 키가 후보 집합 = **L-V(판정 이월)가 닿는다**
    군 매핑 X  후보 집합이 A3 에 없다        = **현행 레버가 못 닿는다** ← 새 레버 후보

## 왜 이 두 가지를 가르나

`VERDICT_CARRY_AND_DEANCHOR_DESIGN` §2b ⑩ 이 이미 구멍을 하나 적어 놨다: *"referral 축 5개는
군 매핑이 없어 이 레버가 닿지 않는다"*. 이 census 는 그 구멍이 **몇 개짜리이고 pass 를 얼마나
쥐고 있는지**를 97 전수로 확정한다 — 새 레버가 필요한지는 그 다음에 답할 수 있다.

⚠**gold 를 읽지만 레버가 아니다**([[23]]): 여기서 나온 값은 표적 집계용이고 엔진·A2 에 안 들어간다.
⚠**지도는 후보 좁히기 전용**(권위본 §3-B 경고): 실제 병목은 궤적이 갈라야 한다.

실행: /home/woori/venvs/seka_env/bin/python x365_misselect_census.py
"""
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

import x341_docbody_verdict as X341                               # noqa: E402
import x351_order_lever_iso as X                                  # noqa: E402
import x357_verdict_carry_multitask as M                          # noqa: E402

REPORTS = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "..", "..", "..", "reports", "facet_rft_2026"))
RECENT = os.path.join(REPORTS, "recent_task_rates_20260817.json")


def recent_rates():
    try:
        d = json.load(io.open(RECENT, encoding="utf-8"))
    except Exception:
        return {}
    if isinstance(d, dict):
        d = d.get("tasks") or d.get("rates") or d
    if isinstance(d, dict):
        return dict((k, (v.get("rate") if isinstance(v, dict) else v)) for k, v in d.items())
    return dict((x.get("id"), x.get("rate")) for x in d if isinstance(x, dict))


def main():
    a2 = X.a2_load()
    po = a2.get("policy_ontology") or {}
    idx = po.get("doc_index") or {}
    census = M.rates()
    recent = recent_rates()
    rows = []
    for fn in sorted(os.listdir(M.TASKS_DIR)):
        if not fn.endswith(".json"):
            continue
        tid = fn[:-5]
        req = M.instructions(tid)
        axes = X341.gold_axes(tid)
        r = census.get(tid, 0.0)
        rr = recent.get(tid)
        best = rr if rr is not None else r
        ax_rows = []
        for ax, gold in sorted(axes.items()):
            g = M.group_for(ax, gold)
            classes = [c for c in sorted((idx.get(g) or {})) if c != "_general_"] if g else []
            leak = bool(X341.norm(gold) and X341.norm(gold) in X341.norm(req))
            ax_rows.append({"axis": ax, "gold": gold, "group": g, "n_cand": len(classes),
                            "leak": leak,
                            "kind": ("ref" if ax == "referral" else
                                     ("card" if ax == "card" else "class"))})
        rows.append({"task": tid, "axes": ax_rows, "census": r, "recent": rr, "rate": best,
                     "excluded": tid in ("task_005", "task_102"),
                     "noTarget": tid == "task_069"})

    n = len(rows)
    print("x365 · 태스크 %d개 · A3 군 %d개 · 최근 실측 있는 태스크 %d개\n"
          % (n, len(idx), sum(1 for x in rows if x["recent"] is not None)))

    # ── ⑴ 오선택 축의 종류별 분포
    print("── ⑴ 선택 축 분류 (축 단위)")
    kind = collections.Counter()
    mapped, unmapped = [], []
    for x in rows:
        for a in x["axes"]:
            kind[(a["kind"], bool(a["group"]), a["n_cand"] > 1, a["leak"])] += 1
            (mapped if a["group"] and a["n_cand"] > 1 else unmapped).append((x["task"], a))
    print("   %-6s %-8s %-10s %-8s %s" % ("종류", "군매핑", "후보>1", "누설", "축 수"))
    for k, v in sorted(kind.items(), key=lambda z: (-z[1], z[0])):
        print("   %-6s %-8s %-10s %-8s %d"
              % (k[0], "O" if k[1] else "**X**", "O" if k[2] else "X", "O" if k[3] else "-", v))

    # ── ⑵ 태스크 단위 버킷
    def bucket(x):
        if x["excluded"]:
            return "Z_제외(벤치결함)"
        axs = x["axes"]
        if not axs:
            return "C_선택축 없음"
        if any(a["group"] and a["n_cand"] > 1 and not a["leak"] for a in axs):
            return "A_선택축·군매핑 O (L-V 도달)"
        if any(not a["group"] for a in axs):
            return "B_선택축·군매핑 **X** (레버 미도달)"
        if all(a["leak"] for a in axs if a["group"]):
            return "D_선택축 있으나 **대본이 답을 말함**(선택 부하 0)"
        return "E_선택축·후보 1개(고를 것이 없음)"

    print("\n── ⑵ 태스크 버킷 · pass(최근 우선)")
    buckets = collections.defaultdict(list)
    for x in rows:
        buckets[bucket(x)].append(x)
    for b in sorted(buckets):
        ts = buckets[b]
        zero = [t for t in ts if (t["rate"] or 0) <= 0]
        print("   %-38s n=%-3d · pass 0%% %-3d · 평균 %.2f · 최근실측 %d"
              % (b, len(ts), len(zero), sum(t["rate"] or 0 for t in ts) / (len(ts) or 1),
                 sum(1 for t in ts if t["recent"] is not None)))
        print("      %s" % ", ".join("%s(%.2f%s)" % (t["task"][5:], t["rate"] or 0,
                                                     "" if t["recent"] is None else "★")
                                     for t in ts))

    # ── ⑶ 군 매핑 없는 축 전수(새 레버 후보)
    print("\n── ⑶ **군 매핑 없는 선택 축 전수** (현행 L-V·판정 이월이 못 닿는 자리)")
    for tid, a in sorted(unmapped):
        if a["group"]:
            continue
        rate = next(x["rate"] for x in rows if x["task"] == tid)
        print("   %-9s axis=%-10s gold=%-28r 후보집합=A3 에 없음 · pass %.2f"
              % (tid, a["axis"], a["gold"], rate or 0))

    # ── ⑷ 선택축이 아예 없는 태스크(다른 축이 병목)
    print("\n── ⑷ 선택 축이 **없는** 태스크 (오선택 레버가 살 것이 없는 자리)")
    none_ax = [x for x in rows if not x["axes"] and not x["excluded"]]
    print("   n=%d · pass 0%% %d개" % (len(none_ax), sum(1 for x in none_ax
                                                        if (x["rate"] or 0) <= 0)))
    print("   %s" % ", ".join("%s(%.2f)" % (x["task"][5:], x["rate"] or 0) for x in none_ax))

    out = os.path.join(REPORTS, "x365_misselect_census.json")
    with io.open(out, "w", encoding="utf-8") as f:
        f.write(json.dumps(rows, ensure_ascii=False, indent=1, default=str))
    print("\n저장: %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
