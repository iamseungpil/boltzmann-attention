#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""공유(general) 문서 주입 프로브 (무료·2026-07-18 NIGHT3·task_028 EcoCard ×100 오독 대응).

배경(포렌식 확정): 028 오탐 5 = EcoCard-Green 셀서 서브가 base_rate=500/100(정답 5) — KB의
"$5.00 points per dollar" 함정 표기 + 해소 문서(general_006 "5 points per dollar"·1pt=$0.01)가
title 접두 필터 밖이라 미주입. 부하 아님(격리 정상).

설계(사용자 지시): A2에 카드별 문서(제목 접두) + 공유 general 문서(파일명 접두=KB 자체 분류)를
명시하고 엔진은 합집합 주입. 큐레이션 0(카테고리 전량)·엔진 리터럴 0.

이 프로브 = 라이브 재현 fidelity([[30]] 프로브≠라이브 함정 회피):
  - 그룹핑 = live A2 그대로 (card×category)·행 전량(대표 추출 없음)
  - 프롬프트 = A2 `inject_instructions`/`operand_schema` 그대로 포맷
  - arm A(card)=카드 문서만(028 실패 재현 기대: EcoCard-Green→500)
  - arm B(shared)=카드+general 25 문서 합집합(수정안: →5 기대)
  - 대상 = task_020/026/027/028 사용자 전 셀 → 수정 + 무회귀(Δspurious≤0) 동시 계측
판정: 행별 sub base_rate vs gold rate(gold_pts/amount). promo 셀은 base×promo도 병기(정독용·[[08]]).

Run: python3 bank_shared_docs_probe.py --base http://localhost:8140/v1 --arms card,shared
"""
import argparse
import json
import os
import re
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.stdout.reconfigure(encoding="utf-8")
from bank_fab_probes import post  # noqa: E402
import bank_rate_f1_gate_probe as P  # noqa: E402
import t2_scaffold_get as SG  # noqa: E402

DOM = P.DOM_DEFAULT
TASKS = ["task_020", "task_026", "task_027", "task_028"]
SHARED_PREFIX_DEFAULT = "doc_credit_cards_credit_cards_(general)"


def _num(s):
    return float(re.sub(r"[^0-9.]", "", str(s)))


def load_iso_spec():
    """A2 gate.json서 isolate 스펙 추출(dict/list 재귀)."""
    spec = json.load(open(os.path.join(HERE, "a2", "banking_knowledge.gate.json"), encoding="utf-8"))
    found = []

    def walk(o):
        if isinstance(o, dict):
            if "isolate" in o and isinstance(o["isolate"], dict) and "inject_instructions" in o["isolate"]:
                found.append(o["isolate"])
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
    walk(spec)
    assert found, "A2 isolate spec 없음"
    return found[0]


def load_docs():
    """도메인 문서 전량(파일명 보존 — 공유 스코프 = 파일명 접두)."""
    dd = os.path.join(DOM, "documents")
    out = []
    for fn in sorted(os.listdir(dd)):
        if fn.endswith(".json"):
            d = json.load(open(os.path.join(dd, fn), encoding="utf-8"))
            out.append({"file": fn, "title": d.get("title") or "", "content": d.get("content") or ""})
    return out


def gold_and_rows():
    """4태스크 사용자의 전 거래(라이브 ctx 재현: 원시 DB 레코드 + account_open) + gold rate.
    gold = 벤치 유도(dispute 아닌 거래=기록 옳음 · dispute 거래=update_rewards의 new_rewards)."""
    tasks = json.load(open(os.path.join(DOM, "tasks.json"), encoding="utf-8"))
    db = json.load(open(os.path.join(DOM, "db.json"), encoding="utf-8"))
    tx = db["credit_card_transaction_history"]["data"]
    accts = {}
    for a in db["credit_card_accounts"]["data"].values():
        accts[(a["user_id"], a["card_type"])] = a["date_of_account_open"]
    fixed, users = {}, {}
    for t in tasks:
        if t.get("id") not in TASKS:
            continue
        for act in (t.get("evaluation_criteria") or {}).get("actions", []) or []:
            args = act.get("arguments") or {}
            inner = args.get("arguments")
            if isinstance(inner, str):
                try:
                    inner = json.loads(inner)
                except Exception:
                    continue
            if act.get("name") == "call_discoverable_agent_tool" and \
                    "update_transaction_rewards" in (args.get("agent_tool_name") or ""):
                fixed[inner["transaction_id"]] = _num(inner["new_rewards_earned"])
            elif act.get("name") == "call_discoverable_user_tool" and isinstance(inner, dict) \
                    and inner.get("user_id"):
                users[t["id"]] = inner["user_id"]
    rows, gold = [], {}
    for uid in sorted(set(users.values())):
        for r in tx.values():
            if r["user_id"] != uid:
                continue
            rr = dict(r)
            rr["account_open"] = accts.get((uid, r["credit_card_type"]))
            rows.append(rr)
            pts = fixed.get(r["transaction_id"], _num(r["rewards_earned"]))
            gold[r["transaction_id"]] = pts / _num(r["transaction_amount"])
    return users, rows, gold


def run_cell(base, model, iso, gval, grows, docs, temp):
    """라이브 _sub_inject 프롬프트 재현."""
    keep = set(iso.get("row_fields") or [])
    raw = [{k: v for k, v in r.items() if k in keep} for r in grows]
    ids = [str(r.get(iso["id_field"])) for r in grows]
    docstr = "\n\n".join("### %s\n%s" % (x["title"], x["content"]) for x in docs)
    schema = json.dumps({i: iso.get("operand_schema", {}) for i in ids}, ensure_ascii=False)
    prompt = iso["inject_instructions"].format(group=gval, docs=docstr, schema=schema,
                                               items=json.dumps(raw, ensure_ascii=False, indent=1))
    r = post(base, {"model": model, "temperature": temp, "max_tokens": 3000, "n": 1,
                    "messages": [{"role": "user", "content": prompt}]}, timeout=600)
    ch = r["choices"][0]
    return SG._merge_json(ch["message"].get("content") or "", set(ids)), len(prompt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8140/v1")
    ap.add_argument("--model", default="Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8")
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--arms", default="card,shared")
    ap.add_argument("--shared_prefix", default=SHARED_PREFIX_DEFAULT)
    ap.add_argument("--only_card", default="", help="카드명 필터(빈=전부)")
    a = ap.parse_args()

    iso = load_iso_spec()
    all_docs = load_docs()
    users, rows, gold = gold_and_rows()
    print("★공유문서 프로브 · tasks=%s · users=%s · rows=%d" % (TASKS, users, len(rows)))
    shared_docs = [d for d in all_docs if d["file"].startswith(a.shared_prefix)]
    print("shared(general) docs=%d (%d chars)\n" % (len(shared_docs), sum(len(d["content"]) for d in shared_docs)))

    gkeys = iso["group_by"] if isinstance(iso["group_by"], list) else [iso["group_by"]]
    doc_key = iso.get("doc_key", gkeys[0])
    groups = defaultdict(list)
    for r in rows:
        groups[tuple(str(r.get(k)) for k in gkeys)].append(r)

    tally = defaultdict(lambda: [0, 0])
    for gk in sorted(groups):
        grows = groups[gk]
        gval = grows[0].get(doc_key)
        if a.only_card and gval != a.only_card:
            continue
        card_docs = [d for d in all_docs if d["title"].startswith(str(gval) + ": ")]
        if not card_docs:
            print("### %s — 카드문서 0 SKIP" % (gk,))
            continue
        for arm in a.arms.split(","):
            docs = card_docs if arm == "card" else card_docs + [d for d in shared_docs if d not in card_docs]
            try:
                out, plen = run_cell(a.base, a.model, iso, gval, grows, docs, a.temp)
            except Exception as e:
                print("### %s [%s] ERR %r" % (gk, arm, str(e)[:80]))
                continue
            ok = bad = 0
            det = []
            for r in grows:
                tid = str(r["transaction_id"])
                g = round(gold[tid], 2)
                v = out.get(tid) or {}
                try:
                    br = float(v.get("base_rate"))
                except Exception:
                    br = None
                pm = v.get("promo_mult") or 1
                try:
                    pm = float(pm)
                except Exception:
                    pm = 1.0
                hit = br is not None and (round(br, 2) == g or round(br * pm, 2) == g)
                ok += hit
                bad += (not hit)
                if not hit:
                    det.append("%s(%s@%s): sub=%s promo=%s gold=%s" % (tid[-6:], r["merchant_name"],
                                                                       r["category"], br, pm, g))
            tally[arm][0] += ok
            tally[arm][1] += len(grows)
            print("### %s [%s] 문서%d·행%d·prompt %dch → 정확 %d/%d %s"
                  % ("×".join(gk), arm, len(docs), len(grows), plen, ok, len(grows),
                     (" | MISS: " + " ; ".join(det)) if det else ""))
    print("\n=== 합계 ===")
    for arm, (ok, n) in sorted(tally.items()):
        print("arm=%-6s %d/%d (%.0f%%)" % (arm, ok, n, 100.0 * ok / n if n else 0))


if __name__ == "__main__":
    main()
