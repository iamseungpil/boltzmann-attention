#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""x36: QP arm **중간 판정** — 완주를 기다리지 않고 지금 있는 것으로 재현성·발화 계수를 낸다.

사용자 지적(2026-08-02): *"지금까지 수행된 것으로 재현성 발화 계수 할 수 있지 않나?"* — 맞다.
  · 발화 계수 = `T2_SG_ISOLATE_TRACE` 자료가 **이미 전량 기록**돼 있다(라이브 진행과 무관).
  · 재현성 = QP arm은 `bank_qpnt2`(4태스크×nt2) + `bank_qp32p1`(32태스크×1)로 **태스크별 최대 3 trial**,
    OFF arm은 Y2-C 3 pass가 있다.

산출:
  A. 태스크별 pass 행렬 (OFF: Y2-C p1/p2/p3 · QP: qpnt2 t0/t1 + qp32 p1) + 항상실패/flip/항상통과
  B. 5런 전패 코어(022·019 등)가 QP에서 몇 번 통과했나 = 승격 근거의 핵심
  C. trace 발화 계수 — operand 수 · 핀 선언 · kind 분포 · **핀 방향**(정책측/행측/양쪽/그외)
  D. 판정 재계산 — 엔진 `_quote_pin_check`를 그대로 import해 라이브 operand에 적용(verdict 분포)

주의: qp32p1은 **진행 중**이라 부분 표본이다. 완주 표본이 아닌 곳에서 sd/비율을 확정하지 말 것
      (핸드오프 §7-1 함정: 부분 표본 sd가 4배 낙관 편향을 냈다). 여기 수치는 **중간 관측**이다.
"""
import argparse
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

ap = argparse.ArgumentParser()
ap.add_argument("--simdir", default=os.path.join(HERE, "..", "..", "..",
                                                 "reports", "facet_rft_2026", "sim_results"))
ap.add_argument("--live", default="/home/woori/scratch/tau2-bench/data/simulations",
                help="진행 중 런의 results.json 루트(없으면 생략)")
ap.add_argument("--trace", default="/home/woori/scratch/tau2-bench/1")
A = ap.parse_args()


def load(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def rewards(d):
    out = {}
    for s in d.get("simulations", []):
        r = (s.get("reward_info") or {}).get("reward")
        out.setdefault(str(s.get("task_id")), []).append(1 if (r or 0) >= 1 else 0)
    return out


# ── A. 태스크별 pass 행렬 ─────────────────────────────────────────────────────
OFF = collections.defaultdict(list)      # task -> [pass1, pass2, pass3]
for tagset, label in ((("bank_y2c2_gpu0_20260731", "bank_y2c2_gpu0b_20260731",
                        "bank_y2c2_gpu1_20260731"), "p1"),
                      (("bank_y2cp2_gpu0_20260801", "bank_y2cp2_gpu1_20260801"), "p2"),
                      (("bank_y2cp3_gpu0_20260801", "bank_y2cp3_gpu1_20260801"), "p3")):
    got = {}
    for t in tagset:
        p = os.path.join(A.simdir, t + ".results.json.gz")
        if os.path.exists(p):
            for k, v in rewards(load(p)).items():
                got.setdefault(k, []).extend(v)
    for k, v in got.items():
        OFF[k].append(max(v) if v else 0)

QP = collections.defaultdict(list)
p = os.path.join(A.simdir, "bank_qpnt2_20260801.results.json.gz")
if os.path.exists(p):
    for k, v in rewards(load(p)).items():
        QP[k].extend(v)                  # nt=2 → 2개
live_n = 0
for g in (0, 1):
    lp = os.path.join(A.live, "bank_qp32p1_gpu%d_20260802" % g, "results.json")
    if os.path.exists(lp):
        d = load(lp)
        live_n += len(d.get("simulations", []))
        for k, v in rewards(d).items():
            QP[k].extend(v)

print("OFF arm(Y2-C) 태스크 %d · QP arm 태스크 %d (라이브 부분표본 %d sim 포함)"
      % (len(OFF), len(QP), live_n))
print("\n" + "=" * 84)
print("A. 태스크별 pass — OFF=Y2-C p1p2p3 · QP=qpnt2 t0t1 + qp32p1")
print("  %-10s %-12s %-14s %s" % ("task", "OFF", "QP", "비고"))
tasks = sorted(set(list(OFF) + list(QP)))
flips_qp, core_off = [], []
for t in tasks:
    o, q = OFF.get(t, []), QP.get(t, [])
    os_ = "".join(str(x) for x in o) or "-"
    qs = "".join(str(x) for x in q) or "-"
    note = []
    if o and sum(o) == 0:
        core_off.append(t)
        if q and sum(q) > 0:
            note.append("★OFF 전패 → QP 통과")
    if q and 0 < sum(q) < len(q):
        flips_qp.append(t)
    if o and q and sum(o) == len(o) and sum(q) == 0:
        note.append("⚠OFF 전승 → QP 전패")
    print("  %-10s %-12s %-14s %s" % (t, os_, qs, " ".join(note)))

print("\n  OFF 전패 태스크 %d개 · 그중 QP에서 1회 이상 통과 = **%d개** (%s)"
      % (len(core_off),
         sum(1 for t in core_off if sum(QP.get(t, [])) > 0),
         ", ".join(t for t in core_off if sum(QP.get(t, [])) > 0) or "없음"))
qp_multi = [t for t in tasks if len(QP.get(t, [])) >= 2]
print("  QP trial≥2 태스크 %d개 — 항상실패 %d / flip %d / 항상통과 %d"
      % (len(qp_multi),
         sum(1 for t in qp_multi if sum(QP[t]) == 0),
         sum(1 for t in qp_multi if 0 < sum(QP[t]) < len(QP[t])),
         sum(1 for t in qp_multi if sum(QP[t]) == len(QP[t]))))

# ── C·D. trace 발화 계수 + 판정 재계산 ────────────────────────────────────────
print("\n" + "=" * 84)
print("C/D. trace 발화 계수 + 엔진 판정 재계산")
if not os.path.exists(A.trace):
    print("  trace 없음: %s" % A.trace)
    sys.exit(0)

import t2_scaffold_get as SG                                    # noqa: E402
from tau2.utils.utils import DATA_DIR                           # noqa: E402

A2 = json.load(open(os.path.join(HERE, "a2", "banking_knowledge.specific.json"), encoding="utf-8"))
TOOL = next(t for t in A2["scaffold_get_tools"] if t["name"] == "get_reward_discrepancies")
ISO = TOOL["variants"]["ratefix"]["isolate"]
QPD = ISO["quote_pin"]
QF, QMIN = ISO.get("quote_field", "exclusion_quote"), ISO.get("quote_min") or 0
POLF, KINDF, ROWF = QPD["policy_field"], QPD["kind_field"], QPD["row_field"]

ALL_DOCS = SG._load_domain_docs("banking_knowledge")
DOCNORM = {}
db = json.load(open(os.path.join(str(DATA_DIR), "tau2", "domains", "banking_knowledge",
                                 "db.json"), encoding="utf-8"))
ROWOF = {str(k): v for k, v in
         (db.get("credit_card_transaction_history") or {}).get("data", {}).items()}


def docnorm_for(card):
    if card not in DOCNORM:
        ds = [x for x in ALL_DOCS if x["title"].startswith(str(card) + ": ")]
        DOCNORM[card] = SG._norm_ground(" ".join(x["content"] for x in ds))
    return DOCNORM[card]


tot = pins = 0
kinds = collections.Counter()
direc = collections.Counter()
verd = collections.Counter()
pin_names = collections.Counter()
rowside = []
for ln in open(A.trace, encoding="utf-8"):
    try:
        rec = json.loads(ln)
    except Exception:
        continue
    card = rec.get("group")
    for tid, v in (rec.get("operands") or {}).items():
        if not isinstance(v, dict):
            continue
        tot += 1
        pin = str(v.get(POLF) or "").strip()
        kind = str(v.get(KINDF) or "").strip()
        if kind:
            kinds[kind] += 1
        if not pin:
            continue
        pins += 1
        pin_names[pin] += 1
        r = ROWOF.get(tid) or {}
        q = str(v.get(QF) or "")
        pn, qn = SG._norm_ground(pin), SG._norm_ground(q)
        mn = SG._norm_ground(str(r.get(ROWF) or ""))
        inq, inm = SG._tok_in(pn, qn), (pn == mn)
        d = ("policy_side" if inq and not inm else "ROW_SIDE" if inm and not inq
             else "both" if inq and inm else "neither")
        direc[d] += 1
        if d == "ROW_SIDE":
            rowside.append((tid, r.get(ROWF), pin))
        vd, _info = SG._quote_pin_check(QPD, v, r, QF, QMIN, docnorm_for(card))
        verd[vd] += 1

print("  operand %d · **핀 선언 %d** · kind %s" % (tot, pins, dict(kinds)))
print("  핀 방향: %s" % dict(direc))
if pins:
    rs = direc.get("ROW_SIDE", 0)
    print("  ⇒ **핀 방향 오류율 = %d/%d (%.1f%%)**  [C289 라이브 관측 1/7의 표본 확대]"
          % (rs, pins, 100.0 * rs / pins))
for tid, m, p in rowside[:6]:
    print("     ROW_SIDE %s  merchant=%r  pin=%r" % (tid[-6:], m, p))
print("  판정(엔진 재계산): %s" % dict(verd))
print("  핀 이름 top: %s" % ", ".join("%s×%d" % (k, v) for k, v in pin_names.most_common(8)))
