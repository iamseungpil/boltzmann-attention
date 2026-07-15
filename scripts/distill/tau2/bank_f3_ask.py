# -*- coding: utf-8 -*-
"""bank_f3_ask.py — (b) F3 의미경계(enum) ASK-부분사정권 정량 (2026-07-16).

F3-의미경계(C95 26.2%) = enum NL→정규화. 분리:
  - enum 값이 USER 발화에 존재 → 정보 O → NL→enum 정규화(F3-mapping) 또는 ASK-confirm으로 닫힘 = 우리 inner 사정권.
  - enum 값이 USER 부재 → 고객이 미제공 → 진짜 ASK 필요(user-원천) 또는 경계.
sim-레벨: F3 sim의 enum 필드 전부 user-present면 '정규화/ASK-closable', 하나라도 부재면 'user-ASK 필요'.

발명 금지·literal floor(정규화 저평가). bank_frontier_mechanism/perstep 재사용.
사용: py bank_f3_ask.py
"""
import json, glob, re, sys, io, os
from collections import Counter

import bank_perstep_decomp as P
import bank_frontier_mechanism as M
_ABOX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "a2", "banking_knowledge.gate.json")


def user_text(s):
    return " ".join(str(m.get("content")) for m in (s.get("messages") or [])
                    if m.get("role") == "user").lower()


def main():
    abox = json.load(open(_ABOX, encoding="utf-8"))
    cmap = P.load_compute_fields(abox)
    files = sorted(glob.glob("C:/tmp/traj/*_banking.json"))

    sim_cls = Counter()
    field_cls = Counter()
    field_by_name = {}
    n_f3 = 0
    for f in files:
        d = json.load(open(f, encoding="utf-8"))
        for s in d.get("simulations", []):
            ri = s.get("reward_info") or {}
            if ri.get("reward") in (None, 1.0):
                continue
            if tuple(ri.get("reward_basis") or []) != ("DB",):
                continue
            if str(s.get("termination_reason")) == "too_many_errors":
                continue
            r = P.decompose_sim(s, abox, cmap)
            gaf = M.ga_wrong_fields(s, abox, cmap)
            ctx_all = " ".join(str(m.get("content")) for m in (s.get("messages") or [])
                               if m.get("role") in ("user", "tool")).lower()
            if M.sim_tier_grounded(r["layers"], gaf, ctx_all, cmap) != "F3-의미경계(enum NL→정규화)":
                continue
            n_f3 += 1
            utext = user_text(s)
            # 이 F3 sim의 enum/judgment 필드들 user-present 여부
            enum_fields = [(tf, fld, val) for (tf, fld, val) in gaf
                           if M._ENUM_FIELD.search(fld) or M._JUDGMENT.search(fld)]
            all_present = True
            any_field = False
            for (tf, fld, val) in enum_fields:
                pres = M.P_val_present(val, utext)
                if pres is None:
                    continue
                any_field = True
                field_cls["user-present(정규화/ASK-confirm)" if pres else "user-부재(진짜 ASK)"] += 1
                rec = field_by_name.setdefault(fld, [0, 0])
                rec[0 if pres else 1] += 1
                if not pres:
                    all_present = False
            if not any_field:
                sim_cls["bool/판정불가"] += 1
            elif all_present:
                sim_cls["전 enum user-present → 정규화/ASK-confirm closable"] += 1
            else:
                sim_cls["일부 user-부재 → 진짜 user-ASK 필요"] += 1

    print("=== (b) F3 의미경계 sim의 ASK-부분사정권 (DB-basis·F3 sim %d) ===" % n_f3)
    tot = sum(sim_cls.values())
    for k, v in sim_cls.most_common():
        print("  %-46s %5d (%.1f%%)" % (k, v, 100 * v / max(tot, 1)))
    print("\n  enum/judgment 필드-레벨 user-present:")
    ft = sum(field_cls.values())
    for k, v in field_cls.most_common():
        print("  %-40s %6d (%.1f%%)" % (k, v, 100 * v / max(ft, 1)))
    print("\n  필드별 present/absent Top12:")
    for fld, (p, ab) in sorted(field_by_name.items(), key=lambda x: -(x[1][0] + x[1][1]))[:12]:
        print("    %-32s present=%4d absent=%4d" % (fld, p, ab))
    closable = sim_cls["전 enum user-present → 정규화/ASK-confirm closable"]
    print("\n  ★F3 중 정규화/ASK-confirm closable(정보 O) = %.1f%% · 진짜 user-ASK 필요 = %.1f%%"
          % (100 * closable / max(tot, 1), 100 * sim_cls["일부 user-부재 → 진짜 user-ASK 필요"] / max(tot, 1)))
    print("  [[08]] user-present=literal/token floor(정규화 저평가)·NL→enum 매핑 자체=F3 능력(우리 inner router 표적).")


if __name__ == "__main__":
    main()
