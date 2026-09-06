#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""x786 — E-PLAN 다건 신호 **정정판** census (2026-09-06).

⛔x785 의 오류 셋을 고친다:
  ⑴ 전 메시지에 돌렸다 → 원장은 `if role == "user"` 만 먹인다(t2_eplan_patch.py:600)
  ⑵ `_enum_items` 를 표적으로 봤다 → task_004 의 실제 방아쇠는 **`_parse_qty`** 다
     (user 발화 실측: enum=0 · **qty=3**). 내가 지목한 «필드 이름 나열» 문장은
     **에이전트가 쓴 것**이라 원장에 애초에 안 들어간다.
  ⑶ 「97% 에서 발화」는 ⑴의 인공물이다.

★`_parse_qty` 의 결함 (실측)
    "1) Full name … 2) Verification … 3) New email address"  -> qty=3   ← 서수를 수량으로
    "I need to dispute three charges on my card"             -> qty=3   ← 진짜 수량
  둘을 구분하지 못한다. 앞은 **번호 매긴 답장**이고 뒤는 **세 건 요청**이다.

★술어 사슬 (t2_eplan_patch.py)
    _parse_qty(user_text) -> ledger.qty_mentioned      (:275-277)
    required_qty() = max(planned qty, qty_mentioned, 1) (:293)
    qty_sig = required_qty() >= 2 and not qty_item_covered(...)  (:348)
    discovery_L1 = (has_tok or qty_sig or multi_entity_hint) and not listed (:347-351)
    -> L1 deny = "목록부터 뽑아라"

★대안 (선언 구동 · gold 무참조)
  A2 `eplan.entity_key` 의 **실물 id 가 도구 출력에 몇 개** 나왔나. 서수도, 필드 이름도,
  상점 이름도 id 가 아니라 안 걸린다. 의미 판단 0 = [[22]] 닫힌 술어.

⛔읽기만 한다.
"""
import collections
import glob
import json
import os
import re
import sys

sys.path.insert(0, "/home/woori/workspace_common/boltzmann-attention-pi/scripts/distill/tau2")
import t2_eplan_patch as E                                            # noqa: E402

SIM = "/home/woori/scratch/tau2-bench/data/simulations"
A2 = json.load(open("/home/woori/workspace_common/boltzmann-attention-pi/scripts/distill/tau2/"
                    "a2/banking_knowledge.gate.json", encoding="utf-8"))
EK = ((A2.get("eplan") or {}).get("entity_key")) or "transaction_id"
IDJ = re.compile(r'"%s"\s*:\s*"([^"]{4,})"' % re.escape(EK))
IDT = re.compile(r'\b%s\b[:=]?\s*([A-Za-z0-9_\-]{4,})' % re.escape(EK))


def main():
    rw = {}
    for ln in open("/home/woori/scratch/x768/q38_sims.txt", encoding="utf-8"):
        p = ln.split()
        if len(p) >= 4:
            rw[(p[0], p[1])] = 1 if (p[3] not in ("None", "") and float(p[3]) >= 1) else 0
    rows = []
    for tag in sorted({t for t, _ in rw}):
        f = os.path.join(SIM, tag, "results.json")
        if not os.path.exists(f):
            continue
        try:
            r = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        for s in (r.get("simulations") or []):
            k = (tag, s.get("task_id"))
            if k not in rw:
                continue
            ms = s.get("messages") or []
            # ★원장과 **같은 방식**: user 발화만
            uq = ue = 0
            trig = None
            for m in ms:
                if (m.get("role") or "") != "user":
                    continue
                c = str(m.get("content") or "")
                q, e = E._parse_qty(c), E._enum_items(c)
                if q > uq:
                    uq = q
                    trig = c[:110].replace("\n", " ")
                ue = max(ue, e)
            ids = set()
            for m in ms:
                if (m.get("role") or "") != "tool":
                    continue
                c = str(m.get("content") or "")
                ids |= set(IDJ.findall(c)) | set(IDT.findall(c))
            rows.append((tag, s.get("task_id"), rw[k], uq, ue, len(ids), trig))

    n = len(rows)
    qsig = [r for r in rows if r[3] >= 2]
    esig = [r for r in rows if r[4] >= E._ENUM_MIN]
    asig = [r for r in rows if r[5] >= 2]
    print("Q3.8 sim %d · 전체 통과율 %.0f%%\n" % (n, 100 * sum(r[2] for r in rows) / max(n, 1)))
    print("%-34s %6s %8s" % ("신호 (user 발화만)", "sim", "그중통과"))
    print("%-34s %6d %8d" % ("_parse_qty >= 2  (구·실제 방아쇠)", len(qsig), sum(r[2] for r in qsig)))
    print("%-34s %6d %8d" % ("_enum_items >= 3 (내가 오인한 것)", len(esig), sum(r[2] for r in esig)))
    print("%-34s %6d %8d" % ("대안: entity id >= 2", len(asig), sum(r[2] for r in asig)))
    print()
    tab = collections.Counter()
    for _, _, ok, q, e, a, _t in rows:
        tab[(q >= 2, a >= 2)] += 1
    print("=== 교차표: _parse_qty>=2 × 대안(entity id>=2) ===")
    print("   %-16s %8s %8s" % ("", "대안 O", "대안 X"))
    print("   %-16s %8d %8d" % ("qty O", tab[(True, True)], tab[(True, False)]))
    print("   %-16s %8d %8d" % ("qty X", tab[(False, True)], tab[(False, False)]))
    print()
    only_q = [r for r in rows if r[3] >= 2 and r[5] < 2]
    print("★qty 만 잡는 %d sim (대안으로 바꾸면 신호가 사라진다) · 그중 통과 %d"
          % (len(only_q), sum(r[2] for r in only_q)))
    print("   ⚠단 이것은 **신호**이지 반려가 아니다 — 실제 반려는 `not listed` 관문을 더 지나야 한다.")
    print()
    print("   방아쇠 문장 표본(통과 sim 우선):")
    for r in sorted(only_q, key=lambda x: -x[2])[:6]:
        print("     %-11s rw=%d qty=%d | %s" % (r[1], r[2], r[3], (r[6] or "")[:76]))


if __name__ == "__main__":
    main()
