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
TASKS = ["task_018", "task_020", "task_021", "task_022", "task_026", "task_027", "task_028", "task_029"]
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
    fixed, users, disputed = {}, {}, set()
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
                if inner.get("transaction_id"):
                    disputed.add(inner["transaction_id"])
    rows, gold, mustflag = [], {}, set()
    for uid in sorted(set(users.values())):
        for r in tx.values():
            if r["user_id"] != uid:
                continue
            rr = dict(r)
            rr["account_open"] = accts.get((uid, r["credit_card_type"]))
            rows.append(rr)
            tid = r["transaction_id"]
            if tid in fixed:
                gold[tid] = int(fixed[tid])          # 정수 포인트(엔진 비교 재현)
            elif tid in disputed:
                mustflag.add(tid)                    # update gold 없음: "기록≠기대"만 요구(018/021/022/029)
            else:
                gold[tid] = int(_num(r["rewards_earned"]))   # dispute 아님 = 기록이 옳음
    return users, rows, gold, mustflag


def run_cell(base, model, iso, gval, grows, docs, temp, extra=""):
    """라이브 _sub_inject 프롬프트 재현. extra=재질의 피드백(범위위반 시)."""
    keep = set(iso.get("row_fields") or [])
    raw = [{k: v for k, v in r.items() if k in keep} for r in grows]
    ids = [str(r.get(iso["id_field"])) for r in grows]
    docstr = "\n\n".join("### %s\n%s" % (x["title"], x["content"]) for x in docs)
    schema = json.dumps({i: iso.get("operand_schema", {}) for i in ids}, ensure_ascii=False)
    prompt = iso["inject_instructions"].format(group=gval, docs=docstr, schema=schema,
                                               items=json.dumps(raw, ensure_ascii=False, indent=1)) + extra
    r = post(base, {"model": model, "temperature": temp, "max_tokens": 3000, "n": 1,
                    "messages": [{"role": "user", "content": prompt}]}, timeout=600)
    ch = r["choices"][0]
    return SG._merge_json(ch["message"].get("content") or "", set(ids)), len(prompt)


# ★재질의 피드백 초안(포팅 시 A2 `range_retry_prompt`로 이동 — 계약문=A2 소속 [[05]]).
#   도메인 리터럴 0: 범위값은 A2 선언·id는 데이터·카드/상인/문서 문구 인용 없음.
RETRY_FEEDBACK = ("\n\n★FEEDBACK: your base_rate for item(s) {ids} was outside the declared valid "
                  "range [{lo}, {hi}] for this field. Report the rate NUMBER exactly as the policy "
                  "states it per dollar; do NOT convert units or scale by 100. Reply again with the "
                  "same full JSON object.")


def _rate_of(v):
    try:
        return float(v.get("base_rate"))
    except Exception:
        return None


def apply_fix(base, model, iso, gval, grows, docs, temp, out, lo, hi, docnorm):
    """라이브 _sub_inject 수정 경로 시뮬 (2026-07-19 consensus 제거·[[10]]):
    범위위반 행 → 그룹 1회 재질의(피드백) → 위반행만 갱신. **엔진은 값을 생성하지 않는다**
    (서브가 재생성). Patagonia류 무근거 강등은 서브 프롬프트(elevated-rate)로 원천 수정."""
    ids = [str(r.get(iso["id_field"])) for r in grows]
    n_retry = 0
    bad = [i for i in ids if not (_rate_of(out.get(i) or {}) is not None and lo <= _rate_of(out.get(i) or {}) <= hi)]
    if bad:
        extra = RETRY_FEEDBACK.format(ids=", ".join(bad), lo=lo, hi=hi)
        try:
            out2, _ = run_cell(base, model, iso, gval, grows, docs, temp, extra=extra)
        except Exception:
            out2 = {}
        for i in bad:
            v2 = out2.get(i)
            if v2 is not None and _rate_of(v2) is not None and lo <= _rate_of(v2) <= hi:
                out[i] = v2
                n_retry += 1
    return out, n_retry, 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8140/v1")
    ap.add_argument("--model", default="Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8")
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--arms", default="card,shared", help="card|shared|fix (fix=card문서+범위재질의+consensus가드)")
    ap.add_argument("--shared_prefix", default=SHARED_PREFIX_DEFAULT)
    ap.add_argument("--rate_range", default="0,20", help="fix arm: A2 선언 시뮬 범위 lo,hi")
    ap.add_argument("--max_batch", type=int, default=-1,
                    help="fix arm 청크 크기 오버라이드(-1=A2 선언값·0=통짜·2=primacy+recency 최소검정)")
    ap.add_argument("--only_card", default="", help="카드명 필터(빈=전부)")
    a = ap.parse_args()
    lo, hi = (float(x) for x in a.rate_range.split(","))

    iso = load_iso_spec()
    all_docs = load_docs()
    users, rows, gold, mustflag = gold_and_rows()
    print("★공유문서 프로브 · tasks=%s · users=%s · rows=%d" % (TASKS, users, len(rows)))
    shared_docs = [d for d in all_docs if d["file"].startswith(a.shared_prefix)]
    print("shared(general) docs=%d (%d chars)\n" % (len(shared_docs), sum(len(d["content"]) for d in shared_docs)))

    gkeys = iso["group_by"] if isinstance(iso["group_by"], list) else [iso["group_by"]]
    doc_key = iso.get("doc_key", gkeys[0])
    # ★사용자별 그룹핑 — 라이브 재현([[30]]): get_reward_discrepancies는 *한 사용자* 거래만 받는다.
    #   user_id를 그룹키에 포함해야 셀 크기·modal이 라이브와 같다(구 전역병합=23행 EcoGreen=비대표).
    groups = defaultdict(list)
    for r in rows:
        groups[tuple([r.get("user_id")] + [str(r.get(k)) for k in gkeys])].append(r)

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
        docnorm = SG._norm_ground(" ".join(x["content"] for x in card_docs))
        # ★max_batch 재현(라이브 A2 동일·[[30]]): fix arm은 A2 선언대로 청킹(1이면 행당 개별 호출).
        #   --max_batch>=0 이면 CLI 오버라이드(배치 크기 스윕용·2=primacy+recency 최소검정).
        mb = int(iso.get("max_batch") or 0) if a.max_batch < 0 else a.max_batch
        for arm in a.arms.split(","):
            docs = card_docs if arm in ("card", "fix") else \
                card_docs + [d for d in shared_docs if d not in card_docs]
            chunks = [grows[i:i + mb] for i in range(0, len(grows), mb)] \
                if (arm == "fix" and mb > 0) else [grows]
            out = {}
            n_retry = n_cons = 0
            plen = 0
            try:
                for ch in chunks:
                    o, pl = run_cell(a.base, a.model, iso, gval, ch, docs, a.temp)
                    plen = max(plen, pl)
                    if arm == "fix":
                        o, nr, _ = apply_fix(a.base, a.model, iso, gval, ch, docs, a.temp,
                                             o, lo, hi, docnorm)
                        n_retry += nr
                    out.update(o)
            except Exception as e:
                print("### %s [%s] ERR %r" % (gk, arm, str(e)[:80]))
                continue
            ok = 0
            det = []
            for r in grows:
                tid = str(r["transaction_id"])
                amt = _num(r["transaction_amount"])
                rec = _num(r["rewards_earned"])
                v = out.get(tid) or {}
                br = _rate_of(v)
                try:
                    pm = float(v.get("promo_mult") or 1)
                except Exception:
                    pm = 1.0
                hit = False
                if tid in mustflag:
                    # update gold 없는 dispute 행: 엔진 판정(|expected-recorded|>tol=1)이 발화해야 정답.
                    #   promo 날짜판정은 라이브 엔진 몫이라 근사: base와 base×promo 둘 다 기록과 일치하면 미발화=MISS.
                    if br is not None:
                        hit = all(abs(amt * rr - rec) > 1 for rr in {br, br * pm})
                    tag = "MUSTFLAG"
                else:
                    gp = gold[tid]                   # 정수 포인트(엔진과 동일 비교)
                    if br is not None:
                        for rr in ({br, br * pm}):
                            if int(amt * rr) == gp or round(amt * rr) == gp:
                                hit = True
                    tag = "gold_pts=%s" % gp
                ok += hit
                if not hit:
                    det.append("%s(%s@%s): sub=%s promo=%s rec=%s %s" % (tid[-6:], r["merchant_name"],
                                                                         r["category"], br, pm, int(rec), tag))
            tally[arm][0] += ok
            tally[arm][1] += len(grows)
            print("### %s [%s] 문서%d·행%d·prompt %dch·retry%d·cons%d → 정확 %d/%d %s"
                  % ("×".join(gk), arm, len(docs), len(grows), plen, n_retry, n_cons, ok, len(grows),
                     (" | MISS: " + " ; ".join(det)) if det else ""))
    print("\n=== 합계 ===")
    for arm, (ok, n) in sorted(tally.items()):
        print("arm=%-6s %d/%d (%.0f%%)" % (arm, ok, n, 100.0 * ok / n if n else 0))


if __name__ == "__main__":
    main()
