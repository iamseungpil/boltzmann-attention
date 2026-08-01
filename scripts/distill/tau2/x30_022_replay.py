#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""x30: task_022 격리-서브 replay (설계서 rev2 §6-3·무료·로컬 vLLM만·user-sim 0).

목적: C275/C279의 인과사슬을 **표 도입 후** 재현해 판정한다.
  ⓐ `txn_ba8b473f295d`(Target - Eco Collection)의 rate가 **생존**하는가(C275 1차 원인 해소)
  ⓑ `select_discrepant`가 **10건**을 내는가(gold 10건과 대조 — 단 *회수는 보장 아님*·C279 ⒢-9)
  ⓒ coverage 라인이 "77 of 77"인가
  ⓓ 드롭 사유별 계수(quote-날조 / 핀-비포함 / 표-조회실패 / 표-비구성원 / kind-결측 / sub-미채움)
  ⓔ ★서브가 ba8b에 declare한 pin·kind 실물(= C279 ⑧ "개체 혼동 vs 범주 과적용" 관측의 계기)

비교를 위해 **ON/OFF 2 arm**을 같은 입력으로 돌린다(OFF = C197 현행 = ba8b 드롭이 재현돼야 함).
`T2_SG_ISOLATE_TRACE`로 서브 산출 원본을 남긴다(2차 리뷰 지적: 관측 불가의 원인이 trace 미설정).

사용(리모트): seka python x30_022_replay.py --base http://localhost:8140/v1
"""
import argparse, json, os, re, sys, types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="http://localhost:8140/v1")
ap.add_argument("--model", default="openai/Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8")
ap.add_argument("--user", default="f9bf8de0be")          # task_022 고객(env 유래)
ap.add_argument("--outdir", default="/home/woori/scratch/x30run")
ap.add_argument("--tag", default="022")
A = ap.parse_args()
os.makedirs(A.outdir, exist_ok=True)

import tau2.agent.llm_agent as la                         # noqa: E402
from tau2.data_model.message import UserMessage           # noqa: E402
from tau2.utils.utils import DATA_DIR                     # noqa: E402
import t2_scaffold_get as SG                              # noqa: E402
import t2_compute as TC                                   # noqa: E402

DOM = os.path.join(str(DATA_DIR), "tau2", "domains", "banking_knowledge")
A2 = json.load(open(os.path.join(HERE, "a2", "banking_knowledge.specific.json"), encoding="utf-8"))
TOOL = next(t for t in A2["scaffold_get_tools"] if t["name"] == "get_reward_discrepancies")
VAR = TOOL["variants"]["ratefix"]
ISO, OP = VAR["isolate"], VAR["op"]

# ── 입력 = env 레코드(거래 + 계좌) · 라이브와 동일하게 원시 필드만 ──────────────
db = json.load(open(os.path.join(DOM, "db.json"), encoding="utf-8"))


def _rows(o, out):
    if isinstance(o, dict):
        if o.get("user_id") == A.user and "transaction_id" in o and "merchant_name" in o:
            out.append(o)
        for v in o.values():
            _rows(v, out)
    elif isinstance(o, list):
        for v in o:
            _rows(v, out)


raw = []
_rows(db, raw)


def num(x):
    m = re.findall(r"-?\d[\d,]*\.?\d*", str(x))
    return float(m[0].replace(",", "")) if m else None


ACC = {}


def _acc(o):
    if isinstance(o, dict):
        if o.get("user_id") == A.user and "card_type" in o and "date_of_account_open" in o:
            ACC[str(o["card_type"])] = o["date_of_account_open"]
        for v in o.values():
            _acc(v)
    elif isinstance(o, list):
        for v in o:
            _acc(v)


_acc(db)
ROWS0 = [{"transaction_id": r["transaction_id"], "transaction_amount": num(r["transaction_amount"]),
          "rewards_earned": num(r["rewards_earned"]), "transaction_date": r["transaction_date"],
          "credit_card_type": r["credit_card_type"], "merchant_name": r["merchant_name"],
          "category": r["category"], "account_open": ACC.get(str(r["credit_card_type"]))}
         for r in raw]
print(f"입력 거래 {len(ROWS0)}행 (user {A.user}) · 계좌 {len(ACC)}종")

GOLD_BY_USER = {"f9bf8de0be": ["txn_ba8b473f295d", "txn_ffeede5eeacd", "txn_4f5e249acc6c", "txn_30cb41175311",
        "txn_508e275e27af", "txn_097c072e7df3", "txn_f84fa27a1b54", "txn_8a8873edaac2",
        "txn_4d2b795c6f98", "txn_1a4918670323"]}   # 대조용(채점 gold·판정엔 미사용·[[23]] 저작 무관)
GOLD = GOLD_BY_USER.get(A.user, [])

orch = types.SimpleNamespace(
    agent=types.SimpleNamespace(llm=A.model, llm_args={
        "api_base": A.base, "api_key": "dummy", "temperature": 0.0,
        "max_tokens": 8192, "timeout": 2400.0, "num_retries": 1}),
    environment=types.SimpleNamespace(domain_name="banking_knowledge"))


def run(flag):
    os.environ["T2_QUOTE_PIN"] = flag
    os.environ["T2_SG_ISOLATE_TRACE"] = os.path.join(A.outdir, f"trace_{A.tag}_qp{flag}.jsonl")
    rows = [dict(r) for r in ROWS0]
    ctx = {"transactions": rows}
    SG._sub_inject(orch, {"name": TOOL["name"]}, ISO, ctx, la, UserMessage)
    ids = TC.apply_op(OP, ctx)
    st = ctx.get("_sg_stats") or {}
    notes = list(getattr(orch, "_t2_qp_notes", []) or [])
    byid = {r["transaction_id"]: r for r in rows}
    return {"ids": ids or [], "stats": st, "notes": notes, "rows": byid}


OUT = {}
for flag in ("1", "0"):
    print(f"\n{'='*70}\n[arm T2_QUOTE_PIN={flag}] 실행")
    R = OUT[flag] = run(flag)
    st = R["stats"]
    ba = R["rows"].get("txn_ba8b473f295d", {})
    print(f"  ⓐ ba8b base_rate = {ba.get('base_rate')!r}  (생존={'YES' if ba.get('base_rate') is not None else 'NO'})")
    print(f"  ⓑ discrepant {len(R['ids'])}건 · gold와 일치 {len(set(R['ids']) & set(GOLD))}/10"
          f" · gold 외 {sorted(set(R['ids']) - set(GOLD))[:4]}")
    print(f"  ⓒ coverage: {st.get('judged')}/{st.get('total')} (skipped {st.get('skipped')})"
          f" · 결핍필드 {st.get('missing_fields')}")
    print(f"  ⓓ ba8b ∈ 결과? {'YES' if 'txn_ba8b473f295d' in (R['ids'] or []) else 'NO'}")
    if R["notes"]:
        print(f"  quote-pin 표면화 {len(R['notes'])}건:")
        for n in R["notes"][:6]:
            print(f"     - {n[:150]}")

# ── ⓔ 서브가 ba8b에 declare한 것 (trace 직독) ──────────────────────────────────
print(f"\n{'='*70}\nⓔ 서브 산출 원본(trace) — ba8b·pin/kind")
for flag in ("1", "0"):
    p = os.path.join(A.outdir, f"trace_{A.tag}_qp{flag}.jsonl")
    if not os.path.exists(p):
        print(f"  [qp={flag}] trace 없음"); continue
    for ln in open(p, encoding="utf-8"):
        try:
            rec = json.loads(ln)
        except Exception:
            continue
        ops = rec.get("operands") or {}
        if "txn_ba8b473f295d" in ops:
            print(f"  [qp={flag}] group={rec.get('group')} → {json.dumps(ops['txn_ba8b473f295d'], ensure_ascii=False)[:400]}")

print("\n" + "=" * 70 + "\n★핀 선언 전수 — 019형 오적용 유무")
for flag in ("1","0"):
    p2 = os.path.join(A.outdir, f"trace_{A.tag}_qp{flag}.jsonl")
    if not os.path.exists(p2): continue
    print(f"  [qp={flag}]")
    for ln in open(p2, encoding="utf-8"):
        try: rec = json.loads(ln)
        except Exception: continue
        for tid, o in (rec.get("operands") or {}).items():
            if isinstance(o, dict) and (o.get("exclusion_policy_merchant") or o.get("exclusion_quote")):
                mer = next((r["merchant_name"] for r in ROWS0 if r["transaction_id"] == tid), "?")
                print(f"     {tid[:18]} merchant={mer:24s} pin={o.get('exclusion_policy_merchant')!r} "
                      f"kind={o.get('exclusion_pin_kind')!r} rate={o.get('base_rate')}")

# ── ⓓ 드롭 사유별 계수 (2차 리뷰: 엔진/sub/표 귀속 분리) ────────────────────────
print(f"\n{'='*70}\nⓓ 드롭 사유별 계수 (arm ON)")
R1 = OUT["1"]
cnt = {}
for tid, r in R1["rows"].items():
    if r.get("base_rate") is None:
        cnt["rate 없음(드롭 or sub-미채움)"] = cnt.get("rate 없음(드롭 or sub-미채움)", 0) + 1
print("  ", cnt or "(없음)")
print(f"  표면화 노트 {len(R1['notes'])}건 (사유 문구가 귀속을 준다)")

json.dump({k: {"ids": v["ids"], "stats": v["stats"], "notes": v["notes"]} for k, v in OUT.items()},
          open(os.path.join(A.outdir, f"x30_{A.tag}_result.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print(f"\n→ {A.outdir}/x30_result.json · trace_qp{{0,1}}.jsonl")
