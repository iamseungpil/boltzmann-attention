#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""x32: **핀-방향 오류 · 차단-층 귀속** 전수 프로브 (무료·로컬 vLLM·gold 0·user-sim 0).

동기(핸드오프 2026-08-01 §6c-12-5 · 원장 C289):
  라이브에서 서브가 `exclusion_policy_merchant`를 **반대쪽(행 merchant)에서 복사**하는 것이
  처음 관측됐다(Thrive Market 핀). 그런데 표본이 **핀 7개 중 1건**뿐이라 방향 오류율도,
  "표가 고유하게 차단한 사례"도 실증되지 않았다(C286 ③: 차단 축은 구 가드 대비 무승부).

질문 3개 (전부 계측·처방 아님):
  Q1 **핀 방향 오류율** — 핀이 quote가 아니라 행 merchant에서 왔는가? (n을 라이브 7 → 전수로)
  Q2 **차단-층 귀속** — 오적용을 실제로 막는 게 1층(quote 축자)인가 2층(핀∈quote)인가 **표**인가.
     C289에서는 1층이 잡았고 표 고유 기여는 미실증이다. 층별 반사실을 계수한다.
  Q3 **재질의 회수율** — 엔진의 현행 재질의(A2 문구 불변)가 방향 오류를 고치는가.
     고치면 §6c-12-5의 "A2 문구 수정"은 불필요하고, 못 고치면 문구가 표적이다.

방법(충실 재현·합성 0):
  엔진 `_isolate_inject`의 서브-콜은 **자립형 단일 UserMessage**다. 같은 A2 선언
  (`inject_instructions`·`operand_schema`·`group_by`·`max_batch`·문서 title-접두 스코프)으로
  프롬프트를 축자 조립하고, 판정은 엔진 함수 `_quote_pin_check`(ON)와 C197 경로(OFF)를
  **그대로 import**해 돌린다 — 프로브가 판정을 재구현하지 않는다([[03b]]).
  모집단 = banking_knowledge `credit_card_transaction_history` **전 행**(사례-선별 없음).

출력: JSON(행별 원자료+판정) + 요약표. gold 미열람 · DB 미변경 · 유료 호출 0.
"""
import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="http://localhost:8141/v1")
ap.add_argument("--model", default="openai/Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8")
ap.add_argument("--out", default="/home/woori/scratch/x32run/x32_pin_direction.json")
ap.add_argument("--conc", type=int, default=4)
ap.add_argument("--limit", type=int, default=0, help="청크 상한(스모크용·0=전수)")
ap.add_argument("--retry", type=int, default=1, help="1=엔진과 동일한 재질의 1회 수행")
A = ap.parse_args()

import tau2.agent.llm_agent as la                        # noqa: E402
from tau2.data_model.message import UserMessage          # noqa: E402
from tau2.utils.utils import DATA_DIR                    # noqa: E402
import t2_scaffold_get as SG                             # noqa: E402  (판정 함수 재사용)

DOMAIN = "banking_knowledge"
DOM = os.path.join(str(DATA_DIR), "tau2", "domains", DOMAIN)

# ── A2 선언 로드 (엔진과 같은 자리) ───────────────────────────────────────────
A2 = json.load(open(os.path.join(HERE, "a2", "banking_knowledge.specific.json"), encoding="utf-8"))
TOOL = next(t for t in A2["scaffold_get_tools"] if t["name"] == "get_reward_discrepancies")
ISO = TOOL["variants"]["ratefix"]["isolate"]
QP = ISO["quote_pin"]
QUOTE_F = ISO.get("quote_field", "exclusion_quote")
RATE_F = ISO.get("rate_field", "base_rate")
QMIN = ISO.get("quote_min") or 0
ID_F = ISO["id_field"]
GKEYS = ISO["group_by"] if isinstance(ISO["group_by"], list) else [ISO["group_by"]]
DOC_KEY = ISO.get("doc_key", GKEYS[0])
KEEP = set(ISO.get("row_fields") or [])
MB = int(ISO.get("max_batch") or 0)
POLICY_F = str(QP.get("policy_field") or "")
KIND_F = str(QP.get("kind_field") or "")
ROW_F = str(QP.get("row_field") or "")

# ── 행·문서 (env 전량·선별 0) ─────────────────────────────────────────────────
db = json.load(open(os.path.join(DOM, "db.json"), encoding="utf-8"))
ROWS = list((db.get("credit_card_transaction_history") or {}).get("data", {}).values())
ALL_DOCS = SG._load_domain_docs(DOMAIN)

groups = {}
for r in ROWS:
    if isinstance(r, dict):
        groups.setdefault(tuple(str(r.get(k)) for k in GKEYS), []).append(r)

JOBS = []            # (gk, gval, docstr, docnorm, rows_chunk)
for gk, g_all in sorted(groups.items()):
    gval = g_all[0].get(DOC_KEY)
    docs = [x for x in ALL_DOCS if x["title"].startswith(str(gval) + ": ")]
    if not docs:
        continue
    docnorm = SG._norm_ground(" ".join(x["content"] for x in docs))
    docstr = "\n\n".join("### %s\n%s" % (x["title"], x["content"]) for x in docs)
    chunks = [g_all[i:i + MB] for i in range(0, len(g_all), MB)] if MB > 0 else [g_all]
    for c in chunks:
        JOBS.append((gk, gval, docstr, docnorm, c))
if A.limit:
    JOBS = JOBS[:A.limit]

print("모집단: 거래 %d · 그룹 %d · 청크(=서브콜) %d · max_batch %s · conc %d"
      % (len(ROWS), len(groups), len(JOBS), MB or "∞", A.conc))
print("판정: ON=_quote_pin_check(엔진 import) · OFF=C197 경로(엔진 코드 축자 재현)\n")

KW = {"api_base": A.base, "api_key": "dummy", "temperature": 0.0,
      "max_tokens": 4096, "timeout": 1800.0, "num_retries": 1}


def build_prompt(gval, docstr, rows_chunk):
    raw = [{k: v for k, v in r.items() if k in KEEP} for r in rows_chunk]
    ids = [str(r.get(ID_F)) for r in rows_chunk]
    schema = json.dumps({i: ISO.get("operand_schema", {}) for i in ids}, ensure_ascii=False)
    return ISO["inject_instructions"].format(
        group=gval, docs=docstr, schema=schema,
        items=json.dumps(raw, ensure_ascii=False, indent=1)), ids


def gen(prompt, name):
    try:
        um = UserMessage(role="user", content=prompt)
    except TypeError:
        um = UserMessage(content=prompt)
    r = la.generate(model=A.model, tools=None, messages=[um], call_name=name, **KW)
    return getattr(r, "content", None) or ""


def c197_ok(v, r, docnorm):
    """OFF arm = 구 가드(C197). 엔진 t2_scaffold_get.py:753-762 축자 재현(raw substring·토큰경계 아님)."""
    q = str((v or {}).get(QUOTE_F) or "").strip()
    if not q:
        return True                                   # quote 없으면 검사 대상 아님(rate 유지)
    qn = SG._norm_ground(q)
    fv = SG._norm_ground(str((r or {}).get(ISO.get("quote_must_contain_field")) or ""))
    return (len(q) >= int(QMIN)) and (qn in docnorm) and bool(fv) and (fv in qn)


def classify_pin(v, r, docnorm):
    """Q1: 핀이 어느 쪽에서 왔는가 (닫힌 문자열 검사만·판단 0)."""
    pin = str((v or {}).get(POLICY_F) or "").strip()
    q = str((v or {}).get(QUOTE_F) or "").strip()
    merch = str((r or {}).get(ROW_F) or "").strip()
    if not pin:
        return "none"
    pn, qn, mn = SG._norm_ground(pin), SG._norm_ground(q), SG._norm_ground(merch)
    in_q, in_m = SG._tok_in(pn, qn), (pn == mn)
    if in_q and not in_m:
        return "policy_side"          # 정책 쪽에서 복사 = 계약대로
    if in_m and not in_q:
        return "ROW_SIDE"             # ★C289형 방향 오류
    if in_q and in_m:
        return "both"                 # 양쪽에 다 있음(구분 불가·무해)
    return "neither"                  # 어느 쪽도 아님(날조/조각)


LOCK = threading.Lock()
OUT = []
DONE = [0]
T0 = time.time()


def run_job(job):
    gk, gval, docstr, docnorm, rows_chunk = job
    prompt, ids = build_prompt(gval, docstr, rows_chunk)
    rowof = {str(r.get(ID_F)): r for r in rows_chunk}
    try:
        got = SG._merge_json(gen(prompt, "x32_inject"), set(ids))
    except Exception as e:
        with LOCK:
            DONE[0] += 1
            print("  [%3d/%3d] %s ⚠생성 실패 %r" % (DONE[0], len(JOBS), gval, e), flush=True)
        return
    recs = []
    for tid in ids:
        v = got.get(tid) or {}
        r = rowof[tid]
        vd, info = SG._quote_pin_check(QP, v, r, QUOTE_F, QMIN, docnorm)
        recs.append({
            "txn": tid, "card": gval, "category": r.get("category"),
            "merchant": r.get(ROW_F),
            "rate": v.get(RATE_F), "quote": str(v.get(QUOTE_F) or ""),
            "pin": str(v.get(POLICY_F) or ""), "kind": str(v.get(KIND_F) or ""),
            "verdict": vd, "why": (info or {}).get("why"),
            "pin_dir": classify_pin(v, r, docnorm),
            "off_keeps_rate": c197_ok(v, r, docnorm),
            "on_keeps_rate": vd in ("pass", "category"),
        })
    # ── 재질의(엔진과 동일 조건·문구 A2 불변) ────────────────────────────────
    if A.retry:
        RETRY_V = ("lookup_missing", "kind_missing")
        bad = [x for x in recs
               if (x["verdict"] in RETRY_V
                   or (x["verdict"] == "reject" and x["why"] == "pin_not_in_quote"))
               and x["rate"] is not None]
        rp = QP.get("lookup_retry_prompt") or QP.get("retry_prompt")
        if bad and rp:
            fb = "\n".join("- %s: %s" % (x["txn"], SG._qp_note(
                QP.get("lookup_note") if x["verdict"] == "lookup_missing" else QP.get("reject_note"),
                {"pin": x["pin"], "why": x["why"]}, rowof[x["txn"]], QP)) for x in bad)
            extra = "\n\n\u2605FEEDBACK on item(s) %s:\n%s\n%s" % (
                ", ".join(x["txn"] for x in bad), fb, rp)
            try:
                got2 = SG._merge_json(gen(prompt + extra, "x32_retry"),
                                      {x["txn"] for x in bad})
            except Exception:
                got2 = {}
            for x in bad:
                v2 = got2.get(x["txn"])
                if not v2:
                    x["retry"] = {"resp": None}
                    continue
                r2 = rowof[x["txn"]]
                vd2, info2 = SG._quote_pin_check(QP, v2, r2, QUOTE_F, QMIN, docnorm)
                x["retry"] = {"verdict": vd2, "why": (info2 or {}).get("why"),
                              "pin": str(v2.get(POLICY_F) or ""),
                              "kind": str(v2.get(KIND_F) or ""),
                              "quote": str(v2.get(QUOTE_F) or ""),
                              "rate": v2.get(RATE_F),
                              "pin_dir": classify_pin(v2, r2, docnorm),
                              "recovered": vd2 in ("pass", "category")}
    with LOCK:
        OUT.extend(recs)
        DONE[0] += 1
        el = time.time() - T0
        vds = " ".join("%s:%s" % (x["txn"][-4:], x["verdict"]) for x in recs)
        print("  [%3d/%3d %5.0fs] %-24s %s" % (DONE[0], len(JOBS), el, str(gval)[:24], vds),
              flush=True)


with ThreadPoolExecutor(max_workers=A.conc) as ex:
    list(ex.map(run_job, JOBS))

# ── 집계 ──────────────────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(A.out), exist_ok=True)
json.dump({"n_rows": len(ROWS), "n_jobs": len(JOBS), "model": A.model, "records": OUT},
          open(A.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


def hist(key, rows=None):
    h = {}
    for x in (rows if rows is not None else OUT):
        h[x.get(key)] = h.get(x.get(key), 0) + 1
    return dict(sorted(h.items(), key=lambda kv: -kv[1]))


print("\n" + "=" * 78)
print("전체 operand %d (서브콜 %d)" % (len(OUT), len(JOBS)))
downgrade = [x for x in OUT if x["quote"]]
print("\n[모집단] 강등-주장(quote 비어있지 않음) = %d / %d" % (len(downgrade), len(OUT)))
print("\n[Q1 핀 방향]  (강등-주장 중)")
for k, n in hist("pin_dir", downgrade).items():
    print("   %-12s %3d" % (k, n))
rowside = [x for x in downgrade if x["pin_dir"] == "ROW_SIDE"]
print("   ⇒ 방향 오류율 = %d/%d" % (len(rowside), len(downgrade)))

print("\n[Q2 차단-층 귀속]  ON arm verdict")
for k, n in hist("verdict", downgrade).items():
    print("   %-16s %3d" % (k, n))
lay = {"1층 quote_unverbatim": 0, "2층 pin_not_in_quote": 0, "표 reject_member": 0,
       "표 lookup_missing": 0, "kind_missing": 0, "통과(pass/category)": 0}
for x in downgrade:
    if x["verdict"] == "reject" and x["why"] == "quote_unverbatim":
        lay["1층 quote_unverbatim"] += 1
    elif x["verdict"] == "reject":
        lay["2층 pin_not_in_quote"] += 1
    elif x["verdict"] == "reject_member":
        lay["표 reject_member"] += 1
    elif x["verdict"] == "lookup_missing":
        lay["표 lookup_missing"] += 1
    elif x["verdict"] == "kind_missing":
        lay["kind_missing"] += 1
    else:
        lay["통과(pass/category)"] += 1
for k, n in lay.items():
    print("   %-22s %3d" % (k, n))
uniq = [x for x in downgrade if x["verdict"] in ("reject_member",) and x["off_keeps_rate"]]
print("   ★표 고유 차단(구 가드는 통과시켰는데 표가 막음) = %d" % len(uniq))
for x in uniq[:8]:
    print("      %s %s / pin=%r" % (x["txn"][-6:], x["merchant"], x["pin"]))

print("\n[ON vs OFF rate 유지]  (강등-주장 %d 중)" % len(downgrade))
both = sum(1 for x in downgrade if x["on_keeps_rate"] and x["off_keeps_rate"])
on_only = [x for x in downgrade if x["on_keeps_rate"] and not x["off_keeps_rate"]]
off_only = [x for x in downgrade if not x["on_keeps_rate"] and x["off_keeps_rate"]]
neither = sum(1 for x in downgrade if not x["on_keeps_rate"] and not x["off_keeps_rate"])
print("   둘 다 유지 %d · **ON만 유지(회수) %d** · **OFF만 유지(표가 새로 막음) %d** · 둘 다 드롭 %d"
      % (both, len(on_only), len(off_only), neither))
for x in on_only[:8]:
    print("      회수 %s %-28s pin=%r" % (x["txn"][-6:], x["merchant"], x["pin"]))

if A.retry:
    rt = [x for x in OUT if x.get("retry")]
    rec = [x for x in rt if (x["retry"] or {}).get("recovered")]
    print("\n[Q3 재질의]  대상 %d · 회수 %d" % (len(rt), len(rec)))
    for x in rt[:10]:
        r = x["retry"] or {}
        print("   %s %-24s %s/%s → %s/%s" % (x["txn"][-6:], str(x["merchant"])[:24],
                                             x["verdict"], x["pin_dir"],
                                             r.get("verdict"), r.get("pin_dir")))
    rs = [x for x in rt if x["pin_dir"] == "ROW_SIDE"]
    print("   ⇒ 방향 오류 중 재질의 대상 %d · 그중 방향 교정 %d"
          % (len(rs), sum(1 for x in rs if (x["retry"] or {}).get("pin_dir") == "policy_side")))

print("\n→ %s" % A.out)
