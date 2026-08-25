# -*- coding: utf-8 -*-
r"""x529 — **우리 게이트가 gold 호출 자체를 거절하는가** (2026-08-25·무료·LLM 0·결정론)

## 관측 (t7348 · 정본 `t2_forensic.mutation_diff`)
```
085 s373753  BLOCKED **17** — 전부 file_debit_card_transaction_dispute_6281 ·
             transaction_id = btxn_a1b2c3d4e501 = **gold 의 거래 id**. 인자만 바꿔 재시도 → 계속 deny
040 s373753  BLOCKED **12** — file_credit_card_transaction_dispute_4829 · txn_a1b2c3d4e503 = gold 거래
079 s626729  BLOCKED 7    055 BLOCKED 3/2 (T2_DISPATCH_ROLE 이 gold 도구 deposit_check 를 deny)
```
즉 표적 도구도 gold 거래도 맞는데 **우리 검증이 통과를 안 시킨다**. [[55]] 상 최우선 자리다.

## 무엇을 재나 (닫힌 술어 · 판단 0)
게이트 술어를 **정본 함수 그대로** 부른다(`t2_gate_patch._wev_deny_msgs`). 두 입력을 넣는다:
    A_model  그 sim 에서 **실제로 막힌 호출**      → 거절 재현 확인
    B_gold   `reward_info` 의 **gold 호출 축자**   → ★게이트가 정답도 막나
`B_gold` 가 deny 면 그 태스크는 **우리 게이트 때문에 원리상 통과 불가**다(가장 강한 판정).
`B_gold` 가 통과면 게이트는 정당하고 결손은 인자 쪽이다 — 그때는 A 와 gold 의 **필드 diff** 가 답이다.

⚠gold 는 **채점·진단용**으로만 쓴다([[23]]) — 레버·임계를 gold 로 고르지 않는다.
⚠이 프로브는 게이트 하나(`T2_WRITE_EVIDENCE`)만 본다. `operator-scope`(085·079)와
  `T2_DISPATCH_ROLE`(055)은 진입점이 달라 별도다 — 그 둘은 목록만 내고 끝낸다.

사용: py -3 x529_gate_blocks_gold.py
"""
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

import t2_forensic as F          # noqa: E402  정본 변이 집합
import t2_gate_patch as G        # noqa: E402  정본 게이트 술어
from gate_interpreter import load_domain_a2   # noqa: E402  선언은 정본 층에서

REP = os.path.abspath(os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026"))
SIMS = os.path.join(REP, "sim_results")
RUNS = ("bank_t7348_halfA_20260824", "bank_t7348_halfB_20260824")
TASKS = ("task_040", "task_085", "task_079", "task_055", "task_094")


class _TC(object):
    """도구 호출 어댑터 — 게이트가 보는 표면만 갖춘다(name · arguments)."""

    def __init__(self, name, args):
        self.name = name
        self.arguments = args
        self.function = None


def _prefix_upto(msgs, name, args):
    """그 호출이 처음 나타나는 메시지 **직전까지**를 돌려준다(닫힌 술어: 이름 + 인자 일부 문자열)."""
    key = str(name or "")
    probe = json.dumps(args, ensure_ascii=False, sort_keys=True)[:60]
    for i, m in enumerate(msgs):
        blob = json.dumps(m, ensure_ascii=False)
        if key and key in blob and (probe[:24] in blob or not probe):
            return msgs[:i]
    return msgs


def main():
    a2 = load_domain_a2("banking_knowledge")
    specs = (a2 or {}).get("write_evidence_specs") or []
    print("=" * 96)
    print("x529 · write_evidence_specs %d개 · 태스크 %s" % (len(specs), list(TASKS)))
    print("=" * 96)
    mut = F.mutating_tools()
    rows = []
    for tag in RUNS:
        p = os.path.join(SIMS, tag + ".results.json.gz")
        if not os.path.exists(p):
            continue
        d = json.load(gzip.open(p, "rt", encoding="utf-8", errors="replace"))
        for s in (d.get("simulations") or []):
            if s.get("task_id") not in TASKS:
                continue
            m = F.mutation_diff(s, mut)
            msgs = s.get("messages") or []
            blocked = m.get("blocked") or []
            gold = m.get("gold") or []
            got = {"run": tag, "task": s.get("task_id"), "seed": s.get("seed"),
                   "n_blocked": len(blocked), "n_gold": len(gold), "checks": []}
            pairs = []
            for b in blocked[:3]:
                pairs.append(("A_model", b.get("name"), b.get("args") or {}))
            for g in gold[:3]:
                pairs.append(("B_gold", g.get("name"), g.get("args") or {}))
            for arm, name, args in pairs:
                # ★문맥은 **그 호출 시점까지**로 자른다 (2026-08-25 계기 정정): sim 전량을 주면
                #   나중 메시지의 근거가 딸려 들어와 게이트가 전부 통과로 나온다 — 첫 판에서
                #   그렇게 나왔고, 그건 게이트가 안 막았다는 뜻이 아니라 **내가 문맥을 더 준 것**이다.
                _cut = _prefix_upto(msgs, name, args)
                try:
                    fb = G._wev_deny_msgs(_cut, _TC(name, args), specs)
                except Exception as e:
                    got["checks"].append({"arm": arm, "tool": name,
                                          "verdict": "ERROR", "detail": repr(e)[:120]})
                    continue
                got["checks"].append({"arm": arm, "tool": name,
                                      "verdict": "DENY" if fb else "pass",
                                      "detail": (str(fb)[:160] if fb else "")})
            rows.append(got)
            print("\n%s seed=%s · blocked %d · gold %d"
                  % (got["task"], got["seed"], got["n_blocked"], got["n_gold"]))
            for c in got["checks"]:
                print("   %-8s %-42s %s %s"
                      % (c["arm"], str(c["tool"])[:42], c["verdict"], c["detail"][:90]))

    out = os.path.join(REP, "x529_gate_blocks_gold_2026_08_25.json")
    with io.open(out, "w", encoding="utf-8") as f:
        json.dump({"probe": "x529", "date": "2026-08-25",
                   "question": "우리 write-evidence 게이트가 gold 호출 자체를 거절하는가",
                   "gate": "t2_gate_patch._wev_deny_msgs (정본·사본 0)",
                   "limits": ["게이트 하나만 본다 — operator-scope·DISPATCH_ROLE 은 진입점이 달라 별도.",
                              "gold 는 진단용([[23]]) · 레버·임계 선택에 쓰지 않는다.",
                              "메시지 문맥은 sim 전량이다 — 실제 deny 시점의 부분 문맥과 다를 수 있다."],
                   "rows": rows}, f, ensure_ascii=False, indent=1)
    print("\n→ %s" % out)


if __name__ == "__main__":
    main()
