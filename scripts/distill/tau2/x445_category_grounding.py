# -*- coding: utf-8 -*-
r"""x445 — **범주 주장에 근거가 있었나** (⒠안 1부 · 오프라인 전수 · 2026-08-20 밤)

## 왜 (사용자 지적 → C568 정정)
*"트럭 구매가 operations 인가"* 는 **KB 에 명시돼 있다**(`..._business_gold_rewards_card_003` =
*"What Qualifies as Operations Spend?"*). 즉 024 의 실패는 판단 불가가 아니라 **근거 없이 범주를 낸 것**이다.
⒠안 = 범주 주장에 **KB 축자 인용**을 요구하고 엔진은 **인용의 실재만** 확인, 못 대면 **기본 요율로 강등**.

## 1부에서 재는 것 (여기)
배선을 만들기 전에 **얼마나 자주 강등될지**를 기존 궤적으로 추정한다 — 유료 0·LLM 0.

    자리 = `check_card_application_fit(spend_category=X)` 를 부른 sim
    물음 = 그 호출 **이전에** 도착한 도구 출력 어딘가에 **그 범주 키워드가 실재**했나
           (= 모델이 근거를 댈 수 있었을 상태였나)
    ⇒ 없었으면 ⒠ 아래에서 그 호출은 **기본 요율로 강등**된다

★엔진 판단 0 — 키워드 **존재 확인**뿐이다([[59]]ⓐ·C45 동형). 뜻은 해석하지 않는다.
★gold 미등장.

사용: py -3 x445_category_grounding.py
"""
import collections
import glob
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

import t2_forensic as F  # noqa: E402

REP = os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026")
FIT = "check_card_application_fit"


def tags():
    base = os.path.abspath(os.path.join(REP, "sim_results"))
    return sorted({os.path.basename(p).replace(".results.json.gz", "")
                   for p in glob.glob(os.path.join(base, "*.results.json.gz"))})


def scan(sim):
    """첫 `spend_category` 호출 이전의 도구 출력에 그 키워드가 있었나."""
    msgs = sim.get("messages") or []
    seen = []
    for i, m in enumerate(msgs):
        for tc in (m.get("tool_calls") or []):
            if F.nameof(tc) != FIT:
                continue
            cat = str((F.argsof(tc) or {}).get("spend_category") or "").strip().lower()
            if not cat:
                continue
            body = " ".join(seen).lower()
            return {"cat": cat, "grounded": cat in body, "n_prior_tool_msgs": len(seen),
                    "task": F.task_id(sim), "reward": (sim.get("reward_info") or {}).get("reward")}
        if (m.get("role") or "") == "tool":
            seen.append(str(m.get("content") or ""))
    return None


def main():
    rows = []
    for t in tags():
        try:
            sims = F.sims(t, ".results.json.gz")
        except Exception:
            continue
        for s in sims:
            r = scan(s)
            if r:
                r["tag"] = t
                rows.append(r)
    n = len(rows)
    g = sum(1 for r in rows if r["grounded"])
    print("=" * 96)
    print("x445 · `spend_category` 를 낸 sim %d — 그 호출 **이전** 도구 출력에 그 키워드가 실재한 비율" % n)
    print("=" * 96)
    print("  근거 있음 %d/%d (%.2f) · 없음 %d ⇒ ⒠ 아래에서 **강등될 호출 = %.0f%%**"
          % (g, n, (g / float(n) if n else 0), n - g, 100.0 * (n - g) / max(1, n)))
    bycat = collections.Counter((r["cat"], r["grounded"]) for r in rows)
    print(chr(10) + "  범주별:")
    for cat in sorted({c for c, _ in bycat}):
        ok, no = bycat[(cat, True)], bycat[(cat, False)]
        print("     %-20s 근거 %3d · 무근거 %3d (%.2f)" % (cat, ok, no, ok / float(ok + no)))
    print(chr(10) + "  태스크별(상위 8):")
    byt = collections.Counter((r["task"], r["grounded"]) for r in rows)
    tasks = sorted({t for t, _ in byt}, key=lambda t: -(byt[(t, True)] + byt[(t, False)]))
    for t in tasks[:8]:
        ok, no = byt[(t, True)], byt[(t, False)]
        print("     %-10s 근거 %3d · 무근거 %3d" % (t, ok, no))
    p = os.path.abspath(os.path.join(REP, "x445_category_grounding.json"))
    with io.open(p, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)
    print(chr(10) + "→ %s" % p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
