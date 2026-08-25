# -*- coding: utf-8 -*-
r"""x531 — ②범주 레버(`T2_VERDICT_GATE`)가 **발화하는가** (2026-08-25·무료·정본 함수 직접 호출)

## 왜
`x451` 실측: 같은 표·같은 손님 발화인데 후보를 **둘로 줄이면** 1/4 → **3/4**(P_pair).
⇒ ②범주의 결손은 비교 능력이 아니라 **후보 수**다([[63]] 제거형).

라이브에서 그 자리에 있는 유일한 기계가 `_verdict_gate_fb` 다 — 제출값이 유효 명단 **안**인데
손님 요구와 충돌하면, **LLM 자신의 판정 줄**과 충돌하지 않는 후보 명단을 돌려준다(엔진은 조회만).
그런데 기본 OFF 이고 침묵 조건이 여섯이라(**템플릿 미선언·코퍼스 부재·요구 인용 0·판정 없음·
UNCLEAR·근거 미검산**) **죽은 배선일 수 있다**. 켜기 전에 그것부터 잰다([[62]]·死배선에 런 금지).

## 무엇을 하나
057·063·055 sim 의 실제 문맥으로 `_verdict_gate_fb` 를 **정본 그대로** 부른다(사본 0·[[67]]).
반환이 있으면 발화, None 이면 어느 fail-safe 에서 멈췄는지 stderr 로 남는다.

## 채점 (닫힌 술어)
    fired      반환이 비어 있지 않은가
    len        돌려준 문면 길이
⛔이 프로브는 *정답률*을 재지 않는다 — **발화 여부**만 본다.

사용: (리모트·cwd=tau2) py -3 x531_verdict_gate_fires.py --port 8141
"""
import argparse
import gzip
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

import x516_induction_target_iso as X16       # noqa: E402  LLM 어댑터 재사용([[67]])
import t2_gate_patch as G                     # noqa: E402  정본 게이트
import t2_forensic as F                       # noqa: E402
from gate_interpreter import load_domain_a2   # noqa: E402

REP = os.path.abspath(os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026"))
SIMS = os.path.join(REP, "sim_results")
RUNS = ("bank_t7348_halfB_20260824", "bank_t7348_halfA_20260824")
TASKS = ("task_057", "task_063", "task_055")


class _Agent(object):
    """게이트가 보는 표면만 — LLM 호출은 `_LA` 어댑터가 받는다."""

    def __init__(self, port):
        self.llm = "probe"
        self.llm_args = {}
        self._probe_port = port


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8141)
    ap.add_argument("--out", default=os.path.join(REP, "x531_verdict_gate_fires_2026_08_25.json"))
    a = ap.parse_args()

    a2 = load_domain_a2("banking_knowledge")
    spec = None
    for s in (a2.get("write_arg_enum") or []):
        if s.get("arg") == "account_class":
            spec = s
            break
    if spec is None:
        raise SystemExit("account_class 선언이 없다 — 중단")
    gm = spec.get("group_map") or {}
    di = ((a2 or {}).get("policy_ontology") or {}).get("doc_index") or {}

    # ★LLM 어댑터를 정본 서브 경로에 꽂는다(프로브가 프롬프트를 쓰지 않는다·[[78]]①)
    import tau2.agent.llm_agent as la_mod
    la_mod.generate = X16._LA(a.port).generate

    rows = []
    for tag in RUNS:
        p = os.path.join(SIMS, tag + ".results.json.gz")
        if not os.path.exists(p):
            continue
        d = json.load(gzip.open(p, "rt", encoding="utf-8", errors="replace"))
        for sim in (d.get("simulations") or []):
            if sim.get("task_id") not in TASKS:
                continue
            msgs = sim.get("messages") or []
            # 제출값·군은 **그 궤적이 실제로 낸 것**에서 가져온다(우리가 짓지 않는다)
            val, grp = "", ""
            for t in F.trajectory_actions(sim):
                if "open_bank_account" not in str(t.get("inner") or t.get("outer") or ""):
                    continue
                ar = t.get("args") or {}
                inner = ar.get("arguments")
                if isinstance(inner, str):
                    try:
                        inner = json.loads(inner)
                    except Exception:
                        inner = {}
                inner = inner if isinstance(inner, dict) else {}
                val = str(inner.get("account_class") or "")
                grp = gm.get(str(inner.get("account_type") or "")) or ""
                if val:
                    break
            if not val:
                rows.append({"task": sim.get("task_id"), "seed": sim.get("seed"),
                             "fired": False, "why": "그 궤적에 account_class 제출이 없다"})
                print("  %-10s seed=%-8s 제출값 없음 — 건너뜀" % (sim.get("task_id"), sim.get("seed")))
                continue
            subs = di.get(grp) or {}
            try:
                fb = G._verdict_gate_fb(_Agent(a.port), msgs, a2, grp, val, subs, spec)
            except Exception as e:
                fb = None
                print("  %-10s seed=%-8s ERROR %r" % (sim.get("task_id"), sim.get("seed"), repr(e)[:90]))
            rows.append({"task": sim.get("task_id"), "seed": sim.get("seed"), "val": val,
                         "group": grp, "fired": bool(fb), "len": (len(str(fb)) if fb else 0),
                         "head": (str(fb)[:180] if fb else "")})
            print("  %-10s seed=%-8s val=%-26s group=%-24s fired=%s len=%d"
                  % (sim.get("task_id"), sim.get("seed"), val[:26], grp[:24],
                     bool(fb), len(str(fb)) if fb else 0))

    with io.open(a.out, "w", encoding="utf-8") as f:
        json.dump({"probe": "x531", "date": "2026-08-25",
                   "question": "T2_VERDICT_GATE 가 ②범주 자리에서 발화하는가",
                   "note": "발화 여부만 잰다 — 정답률이 아니다.",
                   "rows": rows}, f, ensure_ascii=False, indent=1)
    n = sum(1 for r in rows if r.get("fired"))
    print("\n[x531] 발화 %d/%d → %s" % (n, len(rows), a.out))


if __name__ == "__main__":
    main()
