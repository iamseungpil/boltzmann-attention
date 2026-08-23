# -*- coding: utf-8 -*-
"""x498 — ATM 원장 전사 **격리 프로브**: 새 계약에서 서브가 인출 전량을 뽑아내는가 (2026-08-24).

## 왜 이걸 먼저 재나 ([[62]] ①·[[67]] 0단계)

`get_atm_fee_discrepancies` 수리는 오프라인에서 9계좌 gold 를 정확히 맞힌다. 그런데 그건 **내가**
원장을 전사해 준 상태에서다. 라이브에서 전사하는 것은 격리 서브다. 그리고 새 계약이 요구하는 것이
늘었다 — 수수료 라인만이 아니라 **인출 전량**(Bluest 9행·Purple 17행)과 `rebate_amount`·`date`.
서브가 못 뽑으면 성적이 오르는 게 아니라 **기권이 는다**([[25]] 모르면 단언하지 않는다).
⇒ 유료 런 전에 이 한 칸을 먼저 잰다([[09]] 무료 검증 선행).

## 무엇이 격리되나 ([[18]] 정보-맞춘 격리)

**fetch 가 아니라 formalize 만** 잰다. 서브가 레코드를 이미 읽은 상태(= 도구 결과가 문맥에 있는
마감 라운드)를 재현하고, 거기서 배열을 뽑아내는 능력만 본다. 레코드는 **코퍼스에서 건진 env 도구
출력 축자**다(`x498_env_records.json` — 재구성 0). 프롬프트 조립은 `t2_scaffold_get._sub_fetch_formalize`
의 실물 형태를 따른다: `instructions` + `=== REFERENCE ===` + `answer_format`, 그리고 마감 라운드는
도구 없이 `guided_json`(A2 `operand_schema`).

## 팔 ([[57]] 부정통제)

    A_new   워킹트리 A2 계약(인출 단위·date·rebate_amount) + 워킹트리 op
    B_old   런 sha `ee18d797` 계약(수수료 라인 단위) + 그 sha 의 op

B_old 는 **구조적으로** 부재를 못 만든다 — 이 팔이 gold 를 못 맞히는 것은 서브의 실패가 아니라
계약의 실패다. 두 팔의 차이가 곧 수리가 산 것이고, A_new 의 미달분이 남은 위험이다.

## 채점 (gold 는 마지막 한 칸에서만·[[23]])

    rows        서브가 낸 행 수 ↔ 원장의 인출 수(내 축자 전사 = `test_atm_ledger_close.LEDGERS`)
    fields      date·network·fee_amount·withdrawal_amount·rebate_amount 정확도
    judged      엔진이 판정한 행 / 기권한 행(`_sg_stats`)
    net==gold   엔진에 그대로 물렸을 때 계좌 순보정액이 gold 와 같은가  ← 유일한 gold 접촉

프롬프트에는 gold 도, 정답 금액도, 기대 행 수도 들어가지 않는다.

    py -3 x498_atm_extract_iso.py --k 3
    py -3 x498_atm_extract_iso.py --k 3 --arms A_new       # 한 팔만
    T2_PROBE_URL=http://localhost:8141/v1/chat/completions  (기본값·[[30]] 포트 분리)
"""
import argparse
import io
import json
import os
import re
import subprocess
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from t2_compute import apply_op                                    # noqa: E402
import test_atm_ledger_close as L                                  # noqa: E402  (원장 축자 = 정본)

REPORTS = os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026")
RECORDS = os.path.join(REPORTS, "x498_env_records.json")
OUT = os.path.join(REPORTS, "x498_atm_extract_iso.json")
OLD_SHA = "ee18d797"                       # t7346 이 돈 엔진 sha(meta 축자)
A2REL = "scripts/distill/tau2/a2/banking_knowledge.specific.json"
URL = os.environ.get("T2_PROBE_URL", "http://localhost:8141/v1/chat/completions")
MODEL = os.environ.get("T2_PROBE_MODEL", "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8")

# 계좌 id → (원장 라벨, 클래스, gold) — 원장은 test_atm_ledger_close 에서 그대로 가져온다([[67]]).
ACCT = {
    "chk_lj82d4f1a9": "072 Bluest", "chk_538bfb9cba": "072 Light Green",
    "chk_kj93a7b2e1_1": "073 Blue", "chk_kj93a7b2e1_2": "073 Green",
    "chk_kj93a7b2e1_3": "073 Light Green", "chk_ar72c5d8e3_1": "074 Purple",
    "chk_ar72c5d8e3_2": "074 Light Blue", "chk_ar72c5d8e3_3": "074 Dark Green",
    "chk_ar72c5d8e3_4": "074 Evergreen",
}


def ledger_for(label):
    for lab, cls, gold, rows in L.LEDGERS:
        if lab.startswith(label):
            return cls, gold, rows
    raise KeyError(label)


def a2_entry(sha=None):
    """A2 선언을 읽어 온다 — 재료는 **선언에서** 나온다([[71]] 2)."""
    if sha:
        root = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
        raw = subprocess.run(["git", "show", "%s:%s" % (sha, A2REL)], cwd=root,
                             capture_output=True).stdout.decode("utf-8", "replace")
        d = json.loads(raw)
    else:
        d = json.load(io.open(os.path.join(HERE, "a2", "banking_knowledge.specific.json"),
                              encoding="utf-8"))
    return [t for t in d["scaffold_get_tools"]
            if t.get("name") == "get_atm_fee_discrepancies"][0]


def build_messages(entry, acct_id, acct_class, records):
    """`_sub_fetch_formalize` 의 실물 프롬프트 형태 + 이미 fetch 한 마감 라운드 상태."""
    iso = entry["isolate"]
    ref = {k: (acct_id if k == "account_id" else acct_class)
           for k in (iso.get("ref_params") or ["account_id", "account_class"])}
    prompt = "%s\n\n=== REFERENCE ===\n%s\n\n%s" % (
        iso["instructions"], json.dumps(ref, ensure_ascii=False, indent=1), iso["answer_format"])
    call_id = "call_x498"
    getter = (iso.get("getter_tools") or ["call_discoverable_agent_tool"])[0]
    return [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": None, "tool_calls": [{
            "id": call_id, "type": "function",
            "function": {"name": getter, "arguments": json.dumps(
                {"agent_tool_name": "get_bank_account_transactions_9173",
                 "arguments": json.dumps({"account_id": acct_id})})}}]},
        {"role": "tool", "tool_call_id": call_id, "content": records},
    ]


def call(messages, schema, temperature=0.0):
    body = {"model": MODEL, "messages": messages, "temperature": temperature,
            "max_tokens": 4096}
    if schema:
        body["guided_json"] = schema
        body["guided_decoding_backend"] = "xgrammar"
    req = urllib.request.Request(URL, data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.loads(r.read().decode("utf-8"))
    return d["choices"][0]["message"].get("content") or ""


def parse_rows(txt):
    try:
        return json.loads(txt).get("transactions") or []
    except Exception:
        pass
    m = re.search(r"\{.*\}", txt, re.S)
    if not m:
        return []
    try:
        return json.loads(m.group(0)).get("transactions") or []
    except Exception:
        return []


def score(rows, cls, gold, truth, op):
    """서브가 낸 배열을 **그대로** 엔진에 물린다 — 프로브가 손대지 않는다."""
    ctx = {"account_class": cls, "transactions": [dict(r) for r in rows]}
    ids = apply_op(op, ctx)
    st = ctx.get("_sg_stats") or {}
    det = ctx.get("_sg_details") or []
    net = round(sum(d.get("delta") or 0 for d in det), 2)
    # 필드 정확도 — 인출 금액으로 원장 행과 짝짓고(도구 id 는 팔마다 다르다) 필드를 대조한다.
    tkey = {}
    for r in truth:
        tkey.setdefault(round(float(r["withdrawal_amount"]), 2), []).append(r)
    hit = miss = 0
    for r in rows:
        try:
            cand = tkey.get(round(float(r.get("withdrawal_amount")), 2)) or []
        except Exception:
            miss += 1
            continue
        ok = any(abs(float(r.get("fee_amount") or 0) - float(c["fee_amount"])) < 1e-6
                 and str(r.get("network")) == str(c["network"]) for c in cand)
        hit += 1 if ok else 0
        miss += 0 if ok else 1
    return {"n_rows": len(rows), "n_truth": len(truth), "judged": st.get("judged"),
            "skipped": st.get("skipped"), "missing_fields": st.get("missing_fields") or {},
            "net": net, "gold": gold, "net_ok": abs(net - gold) < 1e-6,
            "field_hit": hit, "field_miss": miss, "n_ids": len(ids or [])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--arms", default="A_new,B_old")
    ap.add_argument("--accounts", default="")
    a = ap.parse_args()

    recs = json.load(io.open(RECORDS, encoding="utf-8"))["accounts"]
    arms = {}
    for nm in a.arms.split(","):
        nm = nm.strip()
        if nm == "A_new":
            arms[nm] = a2_entry(None)
        elif nm == "B_old":
            arms[nm] = a2_entry(OLD_SHA)
    want = [x for x in (a.accounts.split(",") if a.accounts else list(recs)) if x in recs]

    report = {"url": URL, "model": MODEL, "k": a.k, "old_sha": OLD_SHA, "runs": []}
    print("endpoint %s · k=%d · 계좌 %d · 팔 %s" % (URL, a.k, len(want), list(arms)))
    for arm, entry in arms.items():
        schema = (entry.get("isolate") or {}).get("operand_schema")
        op = entry["op"]
        print("\n" + "=" * 78)
        print("ARM %s   (schema %s · order_field=%s · rebate=%s)"
              % (arm, "yes" if schema else "no", op.get("order_field"),
                 bool(op.get("rebate"))))
        print("%-20s %-8s %-14s %-16s %s" % ("account", "rows", "judged/skip", "net vs gold",
                                             "field ok"))
        for aid in want:
            label = ACCT[aid]
            cls, gold, truth = ledger_for(label)
            msgs = build_messages(entry, aid, recs[aid]["account_class"], recs[aid]["text"])
            for i in range(a.k):
                try:
                    txt = call(msgs, schema, temperature=0.0 if i == 0 else 0.7)
                    rows = parse_rows(txt)
                    sc = score(rows, cls, gold, truth, op)
                    err = ""
                except Exception as e:
                    rows, sc, err = [], {}, repr(e)[:160]
                report["runs"].append({"arm": arm, "account": aid, "label": label,
                                       "sample": i, "score": sc, "error": err,
                                       "rows": rows})
                if err:
                    print("%-20s ERROR %s" % (label[:20], err))
                else:
                    print("%-20s %-8s %-14s %-16s %s"
                          % (label[:20], "%d/%d" % (sc["n_rows"], sc["n_truth"]),
                             "%s/%s" % (sc["judged"], sc["skipped"]),
                             "%.2f %s %.2f" % (sc["net"], "==" if sc["net_ok"] else "!=", gold),
                             "%d/%d" % (sc["field_hit"], sc["field_hit"] + sc["field_miss"])))
        io.open(OUT, "w", encoding="utf-8").write(
            json.dumps(report, ensure_ascii=False, indent=1) + "\n")

    print("\n" + "=" * 78)
    for arm in arms:
        rs = [r for r in report["runs"] if r["arm"] == arm and not r["error"]]
        if not rs:
            continue
        okn = sum(1 for r in rs if r["score"].get("net_ok"))
        full = sum(1 for r in rs if r["score"].get("n_rows") == r["score"].get("n_truth"))
        skip = sum(r["score"].get("skipped") or 0 for r in rs)
        print("%s: net==gold %d/%d · 행 전량 %d/%d · 기권 행 합 %d"
              % (arm, okn, len(rs), full, len(rs), skip))
    print("→ %s" % os.path.basename(OUT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
