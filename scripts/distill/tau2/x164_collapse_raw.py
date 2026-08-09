# -*- coding: utf-8 -*-
r"""x164 — 붕괴 행을 **날것으로** 읽고, 절벽을 만든 메시지를 본다 (유료 0).

왜: x163 이 j\*=27 을 확정했지만 그 행의 **관측질량이 0.033** 이었다 — 96.7% 가 우리 버킷 밖이다.
x159 각주 ⒜ 가 예고한 대로 붕괴 행은 **1글자 토큰**에 질량이 앉는데 `x157.lp_of` 는 (다른
오귀속을 막으려고) 2자 이상만 센다. 그래서 그 행의 엔트로피도 argmax 도 **신뢰할 수 없다**.
⇒ 여기서는 **아무 필터 없이** top-20 을 그대로 찍고, 버킷 합산 전 원본을 눈으로 본다([[55]]:
계기를 먼저 의심한다·[[08]]: 집계 전에 원본).

그리고 전이가 **단 한 메시지**에 걸려 있으므로(j=26 gold p≈0.995 → j=27 붕괴) *"그 메시지가
무엇인가"* 가 곧 다음 질문이다. 마지막 몇 메시지를 축자로 찍는다 — 해석 없이 원문만.

실행: T2_PROBE_URL=... T2_PROBE_MODEL=... py -3 x164_collapse_raw.py [JLIST]
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import x149_choice_isolation as X                              # noqa: E402
import x150_choice_ablation as Y                               # noqa: E402
import x157_entrainment_lambda as P                            # noqa: E402
import t2_factdag as FD                                        # noqa: E402
import t2_ledger as LG                                         # noqa: E402
from gate_interpreter import load_domain_a2                     # noqa: E402


def main():
    js = [int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else "25,26,27").split(",")]
    a2 = load_domain_a2("banking_knowledge")
    spec = next(s for s in a2["ledger_metrics"] if s.get("eligible_text"))
    rows = a2["policy_ontology"]["rows"]
    maps = {ax: FD._a3_map(rows, {"axis": ax}) for ax in spec["eligible"]["show_axes"]}
    table = LG.eligible_text(730, {}, maps, spec, {"qualifying_deposit_usd": 30000}).strip()
    base = table + "\n\n" + X.FACTS[P.TASK] + "\n\n" + X.QUESTION
    choices = [l.strip().split(":")[0].strip() for l in table.splitlines() if l.startswith("  ")]
    msgs = Y.msgs_of(os.environ.get("T2_PROBE_TAG", "bank_elig_20260809i"), P.TASK)

    print("model=%s · 궤적 %d · 후보 %d" % (P.MODEL, len(msgs), len(choices)))
    print("후보 원본: %s" % ", ".join(choices))

    for j in js:
        pre = ("Here is a customer-service conversation so far.\n\n"
               + Y.render(msgs[:j]) + "\n\n") if j else ""
        d = P.first_token_dist(pre + base, choices)
        tot = sum(math.exp(v) for v in d.values())
        print("\n=== j=%d · top-20 원본 (필터 0) · 합계질량 %.4f ===" % (j, tot))
        for k, v in sorted(d.items(), key=lambda kv: -kv[1])[:12]:
            p = math.exp(v)
            print("   %-16r p=%.4f  logp=%8.3f  %s" % (k, p, v, "#" * int(round(p * 40))))

    # ── 절벽을 만든 메시지 (해석 없이 축자) ──────────────────────────────────
    print("\n=== 마지막 3 메시지 축자 (전이가 걸린 자리) ===")
    for i in range(max(0, len(msgs) - 3), len(msgs)):
        m = msgs[i]
        role = getattr(m, "role", None) or (m.get("role") if isinstance(m, dict) else "?")
        txt = getattr(m, "content", None) or (m.get("content") if isinstance(m, dict) else "")
        txt = " ".join(str(txt or "").split())
        print("\n--- 메시지 #%d  role=%s  len=%d" % (i + 1, role, len(txt)))
        print("    " + txt[:900])
    return 0


if __name__ == "__main__":
    sys.exit(main())
