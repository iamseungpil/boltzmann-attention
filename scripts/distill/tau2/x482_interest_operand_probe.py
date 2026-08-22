# -*- coding: utf-8 -*-
r"""x482 — `get_interest_correction` 서브가 principal·actual_apy 를 낼 수 있는가 (2026-08-22·무료)

## 왜 (093 pass 를 막는 **둘째** 갈래)
x480/x481 이 첫 갈래(checking boost 표기)를 확정·수리했다. 남은 것은 이것이다 —
`get_interest_correction` 의 격리 서브가 **`principal=-1; actual_apy=-1`** 을 내고 폐기된다
(t7341 7회·t7343 4회 재현). 폐기 → 폴백 → 메인이 추측한 `actual_apy=3.75` → grounding 드롭 →
도구 `None` → 모델이 금액을 자기 계산 → `T2_WRITE_EVIDENCE` deny 의 사슬이 거기서 시작된다.

정답은 **레코드와 A2 공식**에서 나온다(gold 미참조·[[23]]):
    레코드   `current_holdings: 144000.00`
    거래     `MONTHLY INTEREST CREDIT  amount: 480.0`  (10/31/2025)
    A2 공식  actual_apy = monthly credit x 12 / principal x 100
           ⇒ 480 x 12 / 144000 x 100 = **4.0**

## 팔 (한 번에 한 변수만)
    A_asis   현행 선언 그대로 — ref=account_id 뿐이고 서브가 **getter 로 직접 읽어야** 한다
    B_raw    + 계좌 레코드·거래 내역 **원문**을 REFERENCE 에 실어 준다(x481 R3 와 같은 처방)
    N_neg    A_asis 반복 — 부정통제([[57]])
⇒ A 가 실패하고 B 가 되면 결손은 **전달**이다([[62]] ②: getter 경로가 서브에서 안 산다).
  둘 다 실패하면 산술 자체가 안 되는 것이고, 그때만 그 단계에 결정론을 얹는다([[62]] ③).

## 환경 — 라이브와 **같은 플래그**를 켠다
x481 1차의 자기 결함(문법 플래그 누락)을 반복하지 않는다([[18]] 정보-맞춘 격리):
`T2_SG_ISOLATE`·`T2_SG_GROUND`·`T2_SG_DOCS`·`T2_SG_SCHEMA` 를 라이브와 같이 켠다.

사용: (리모트·cwd=x482run) py x482_interest_operand_probe.py --port 8141 --n 4
"""
import argparse
import collections
import glob
import gzip
import io
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

import t2_scaffold_get as SG                # noqa: E402  피측정 대상(정본)
import x448_index_vs_all_iso as IVA         # noqa: E402  Sandbox 재사용([[67]])

REP = os.path.abspath(os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026"))
BASE = os.path.join(REP, "sim_results")
DECL = "get_interest_correction"
TASK = "task_093"


def declaration():
    """선언은 A3 정본에서 읽는다([[71]] 2항)."""
    p = os.path.join(HERE, "a2", "banking_knowledge.specific.json")
    with io.open(p, encoding="utf-8") as f:
        d = json.load(f)
    for s in (d.get("scaffold_get_tools") or []):
        if s.get("name") == DECL:
            return s
    raise SystemExit("선언 %s 없음" % DECL)


def harvest(pred, limit=1):
    """궤적의 **도구 출력 원문**을 조건으로 집는다(우리가 짓지 않는다)."""
    out = []
    for p in sorted(glob.glob(os.path.join(BASE, "*.results.json.gz")), reverse=True):
        try:
            with gzip.open(p, "rt", encoding="utf-8") as f:
                d = json.load(f)
        except Exception:
            continue
        for s in (d.get("simulations") or []):
            if s.get("task_id") != TASK:
                continue
            for m in (s.get("messages") or []):
                if m.get("role") != "tool":
                    continue
                c = str(m.get("content") or "")
                if pred(c):
                    out.append(c)
                    if len(out) >= limit:
                        return out
    return out


def truth_from_records(rec, txn):
    """정답을 **레코드·거래에서** 계산한다(gold 미참조). 닫힌 술어: 숫자 두 개와 A2 공식."""
    mh = re.search(r"current_holdings:\s*([0-9][0-9,\.]*)", rec or "")
    mc = None
    for blk in re.split(r"\n\s*\d+\.\s", txn or ""):
        if "INTEREST CREDIT" in blk.upper():
            mc = re.search(r"amount:\s*([0-9][0-9,\.]*)", blk)
            if mc:
                break
    if not (mh and mc):
        return None
    principal = float(mh.group(1).replace(",", ""))
    credit = float(mc.group(1).replace(",", ""))
    return {"principal": principal, "credit": credit,
            "actual_apy": round(credit * 12 / principal * 100, 6)}


class _Orch(object):
    """서브가 요구하는 최소 표면 (x456/x481 과 동형·라이브와 같은 env/모델 문자열)."""

    def __init__(self, tool_names, model, base, env, messages=None):
        import types
        want = set(tool_names or [])
        tools = [t for t in (env.get_tools() or []) if getattr(t, "name", None) in want]
        self.agent = types.SimpleNamespace(
            tools=tools, llm="openai/%s" % model,
            llm_args={"api_base": base, "api_key": "dummy", "temperature": 0.0})
        self.environment = env
        self._msgs = messages or []

    def get_messages(self):
        return self._msgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8141)
    ap.add_argument("--model", default="Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--arms", default="A_asis,B_raw,N_neg")
    ap.add_argument("--out", default="x482_interest_operand.json")
    a = ap.parse_args()

    os.environ.setdefault("OPENAI_API_BASE", "http://localhost:%d/v1" % a.port)
    os.environ.setdefault("OPENAI_BASE_URL", "http://localhost:%d/v1" % a.port)
    os.environ.setdefault("OPENAI_API_KEY", "EMPTY")
    # ★라이브와 같은 플래그 (x481 1차 결함 재발 방지·[[18]])
    for k in ("T2_SG_ISOLATE", "T2_SG_GROUND", "T2_SG_DOCS", "T2_SG_SCHEMA"):
        os.environ[k] = "1"
    os.environ.pop("T2_SG_ISOFB", None)

    d = declaration()
    iso = d.get("isolate") or {}

    rec = harvest(lambda c: "current_holdings" in c and "level:" in c)
    txn = harvest(lambda c: "INTEREST CREDIT" in c.upper() and "amount:" in c)
    if not rec or not txn:
        raise SystemExit("궤적에서 레코드/거래 원문을 못 찾았다 (rec=%d txn=%d)" % (len(rec), len(txn)))
    truth = truth_from_records(rec[0], txn[0])
    if not truth:
        raise SystemExit("레코드에서 정답을 계산하지 못했다 — 채점 불가")

    # ref(account_id)도 궤적에서: 레코드 원문의 첫 savings 계좌 id
    mid = re.search(r"account_id:\s*(sav_[a-z0-9_]+)", rec[0])
    if not mid:
        raise SystemExit("레코드에서 account_id 를 못 읽었다")
    acct = mid.group(1)

    sb = IVA.Sandbox()
    print("=" * 96)
    print("x482 · %s · ref=%s · getter=%s" % (DECL, acct, iso.get("getter_tools")))
    print("     레코드에서 계산한 정답: principal=%s · credit=%s · actual_apy=%s"
          % (truth["principal"], truth["credit"], truth["actual_apy"]))
    print("=" * 96)

    def run_env(tcs):
        from tau2.data_model.message import ToolMessage
        out = []
        for t in tcs:
            try:
                txt = str(sb.env.use_tool(t.name, **(t.arguments or {})))
                err = False
            except Exception as e:
                txt, err = "ERROR: %r" % (e,), True
            out.append(ToolMessage(id=t.id, role="tool", requestor="assistant",
                                   content=txt, error=err))
        return out

    want = set(x.strip() for x in a.arms.split(",") if x.strip())
    rows = []
    for arm, use_raw in (("A_asis", False), ("B_raw", True), ("N_neg", False)):
        if arm not in want:
            continue
        iso_a = json.loads(json.dumps(iso))
        ctx = {"account_id": acct}
        if use_raw:
            # x481 R3 와 같은 처방 — 원문을 REFERENCE 에 그대로 싣는다(엔진 파싱 0).
            iso_a["ref_params"] = list(iso_a.get("ref_params") or []) + ["account_records",
                                                                        "transactions_raw"]
            ctx["account_records"] = rec[0]
            ctx["transactions_raw"] = txn[0]
        orch = _Orch(iso_a.get("getter_tools") or [], a.model,
                     "http://localhost:%d/v1" % a.port, sb.env)
        print("\n── %s (원문 전달=%s) ─────────────────────" % (arm, use_raw))
        for i in range(a.n):
            c = dict(ctx)
            try:
                got = SG._sub_fetch_formalize(orch, d, iso_a, c, run_env)
            except Exception as e:
                got = None
                print("  #%d EXC %r" % (i, e))
            pr = (got or {}).get("principal")
            ap_ = (got or {}).get("actual_apy")

            def _f(x):
                try:
                    return float(x)
                except Exception:
                    return None
            pr_ok = _f(pr) is not None and abs(_f(pr) - truth["principal"]) < 1e-6
            ap_ok = _f(ap_) is not None and abs(_f(ap_) - truth["actual_apy"]) < 1e-6
            rows.append({"arm": arm, "i": i, "returned": bool(got),
                         "principal": pr, "actual_apy": ap_,
                         "principal_ok": pr_ok, "actual_apy_ok": ap_ok})
            print("  #%d returned=%-5s principal=%-12s (%s) actual_apy=%-10s (%s)"
                  % (i, bool(got), pr, "OK" if pr_ok else "X", ap_, "OK" if ap_ok else "X"))

    agg = collections.defaultdict(lambda: {"n": 0, "returned": 0, "pr": 0, "ap": 0, "both": 0})
    for x in rows:
        s = agg[x["arm"]]
        s["n"] += 1
        s["returned"] += 1 if x["returned"] else 0
        s["pr"] += 1 if x["principal_ok"] else 0
        s["ap"] += 1 if x["actual_apy_ok"] else 0
        s["both"] += 1 if (x["principal_ok"] and x["actual_apy_ok"]) else 0

    payload = {"declaration": DECL, "task": TASK, "account_id": acct, "truth": truth,
               "rows": rows, "summary": {k: dict(v) for k, v in agg.items()}}
    with io.open(os.path.join(REP, a.out), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)

    print("\n" + "=" * 96)
    print("판정 — 레코드에서 계산한 정답: principal=%s · actual_apy=%s"
          % (truth["principal"], truth["actual_apy"]))
    for arm in ("A_asis", "N_neg", "B_raw"):
        if arm not in agg:
            continue
        s = agg[arm]
        print("  %-8s 둘 다 맞음 %d/%d · principal %d/%d · actual_apy %d/%d · 답반환 %d/%d"
              % (arm, s["both"], s["n"], s["pr"], s["n"], s["ap"], s["n"], s["returned"], s["n"]))
    print("\n[산출물] → %s" % os.path.join(REP, a.out))
    print("해석: A≈N 이고 B 가 높으면 결손은 **전달**(getter 경로가 서브에서 안 산다)·[[62]] ②.")
    print("      B 도 실패하면 산술이 안 되는 것이고, 그 단계에만 결정론([[62]] ③).")


if __name__ == "__main__":
    main()
