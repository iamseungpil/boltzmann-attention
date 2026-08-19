# -*- coding: utf-8 -*-
r"""x394 — **격리 프로브: 계산은 누가 해야 하나**. LLM 이 *형식화*와 *산술* 중 무엇을 못 하는지 가른다.

## 왜 (사용자 지시 2026-08-19)
> *"LLM 이 격리로 금액계산 같은 걸 잘하는지부터 확인하라. 이전에 계산은 무조건 엔진이 한다는 데에는
>  LLM 이 계산을 잘 못한다는 **가정**이 있었다. 엄밀하게 결정하라. … 우리 가정은 LLM 은 유한계로
>  formalize 하고 결정론기가 그걸 verify 하는 것이다."*

가정을 실측으로 바꾼다([[62]] ①격리로 결손 측정). t7326 의 ARGDIFF 53 건 중 093·094(이자 차액)와
073·074(ATM 수수료 환급)가 *"엔진이 계산해 줘야 하나"* 라는 물음을 만든 자리다.

## 무엇을 가르나 (세 지표는 서로 독립이다)
    quote_ok   모델이 인용한 상수가 **제공 문서에 축자로 있는가**(근거 검산 · [[22]] 따름정리)
    arith_ok   모델이 **자기가 낸 성분/입력으로** 계산을 맞게 했는가 (엔진이 재계산해 대조)
    final_ok   최종 금액이 gold 와 같은가 (gold 는 **채점에만** 쓴다 · 입력에 절대 넣지 않는다)

  · arith_ok 높고 final_ok 낮음 ⇒ 결손은 **형식화(어느 성분이 적용되나)** — 엔진이 계산할 이유가 없다.
  · arith_ok 낮음 ⇒ 결손은 **산술** — 그때만 엔진 계산이 정당하다.

## 팔 (정보-맞춘 격리 · [[18]])
    A_min    태스크가 선언한 required_documents 전문 + 그 손님의 계좌·거래 레코드(전량) + 요청 한 줄
    A_focus  같은 레코드 + **지배 문서만**(전달 부하만 줄인 팔 — 계산은 안 해 준다)
    C_neg    같은 레코드 + **무관 문서**(부정통제 · [[57]]). 여기서도 맞으면 프로브가 재료를 안 재는 것이다.

⚠레코드는 db.json 에서 결정론으로 만든다(궤적에서 뽑지 않는다 — 실패 런은 그 read 를 안 했다).
⚠required_documents 는 env 메타데이터다. **계측에만** 쓴다(런타임 레버가 읽으면 [[23]] 위반).

사용: py -3 x394_calc_formalize_iso.py [--n 8] [--tasks 093,094,073,074] [--arms A_min,A_focus,C_neg]
"""
import argparse
import collections
import io
import json
import os
import re
import sys
import threading
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

BK = "/home/woori/scratch/tau2-bench/data/tau2/domains/banking_knowledge"
DOCDIR = os.path.join(BK, "documents")
PORTS = [8140, 8141]
MODEL = "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8"

# 태스크 → 흐름 종류와 지배 문서 제목(축자). gold 액션은 **채점에만** 쓴다.
SPEC = {
    "task_093": {"kind": "interest", "who": "Somchai Prasert",
                 "gov": ["Internal: Submitting Interest Discrepancy Reports",
                         "Internal: Applying Credits to Savings Accounts"]},
    "task_094": {"kind": "interest", "who": "Wei-Ting Lin",
                 "gov": ["Internal: Submitting Interest Discrepancy Reports",
                         "Internal: Applying Credits to Savings Accounts"]},
    "task_073": {"kind": "fee", "who": "Kim Junho",
                 "gov": ["Internal: Applying Credits and Rebates to Checking Accounts",
                         "Internal: Retrieving Bank Account Transaction History"]},
    "task_074": {"kind": "fee", "who": "Ahmad Razali bin Mohd Yusof",
                 "gov": ["Internal: Applying Credits and Rebates to Checking Accounts",
                         "Internal: Retrieving Bank Account Transaction History"]},
}

SCHEMA_INTEREST = """{
  "components": [{"name": "<APY 성분 이름>", "rate": <숫자>, "quote": "<문서 축자 인용 한 문장>"}],
  "expected_apy": <숫자>,
  "actual_apy": <숫자>,
  "actual_interest_credited": <숫자>,
  "balance": <숫자>,
  "amount_difference": <숫자>,
  "account_id": "<계좌 id>"
}"""

SCHEMA_FEE = """{
  "accounts": [{
     "account_id": "<계좌 id>",
     "items": [{"date": "<MM/DD/YYYY>", "description": "<거래 설명>",
                "charged": <실제 청구된 수수료>, "correct": <정책상 올바른 수수료>,
                "quote": "<그 수수료율을 정한 문서 축자 인용>"}],
     "refund": <이 계좌에 환급할 총액>
  }]
}"""

SYS = ("You are a Rho-Bank support agent's analysis module. "
       "Answer ONLY with a single JSON object matching the requested schema. No prose, no markdown fence.")


def load_docs():
    out = {}
    for fn in sorted(os.listdir(DOCDIR)):
        if not fn.endswith(".json"):
            continue
        try:
            d = json.load(io.open(os.path.join(DOCDIR, fn), encoding="utf-8", errors="replace"))
        except Exception:
            continue
        out[str(d.get("id") or fn[:-5])] = {"title": str(d.get("title") or ""),
                                            "content": str(d.get("content") or "")}
    return out


def load_tasks():
    return {str(t.get("id")): t for t in json.load(
        io.open(os.path.join(BK, "tasks.json"), encoding="utf-8", errors="replace"))}


def load_db():
    return json.load(io.open(os.path.join(BK, "db.json"), encoding="utf-8", errors="replace"))


def user_by_name(db, name):
    for uid, u in (db["users"]["data"] or {}).items():
        if str(u.get("name", "")).strip().lower() == name.strip().lower():
            return uid, u
    return None, None


def records_for(db, uid):
    """그 손님의 계좌 전량 + 각 계좌 거래내역 전량 (판단 0 · 필터 0)."""
    accts = {aid: a for aid, a in (db["accounts"]["data"] or {}).items()
             if str(a.get("user_id")) == str(uid)}
    tx = {}
    hist = (db.get("bank_account_transaction_history") or {}).get("data") or {}
    for aid in accts:
        v = hist.get(aid)
        if v is not None:
            tx[aid] = v
    if not tx:                       # 스키마가 평면일 때 대비
        for k, v in hist.items():
            if isinstance(v, dict) and str(v.get("account_id")) in accts:
                tx.setdefault(str(v["account_id"]), []).append(v)
    return accts, tx


def gold_of(task, kind):
    """gold 는 **채점 전용**. 프롬프트에 넣지 않는다."""
    acts = ((task.get("evaluation_criteria") or {}).get("actions") or [])
    out = {"amounts": {}, "expected_apy": None, "actual_apy": None}
    for a in acts:
        ar = a.get("arguments") or {}
        inner = ar.get("arguments")
        if isinstance(inner, str):
            try:
                inner = json.loads(inner)
            except Exception:
                inner = {}
        nm = ar.get("agent_tool_name") or a.get("name")
        # 래퍼면 중첩 arguments, 아니면 평면 — 둘 다 본다(형태가 액션마다 다르다).
        pay = inner if isinstance(inner, dict) and inner else ar
        if nm in ("apply_savings_account_credit_6831", "apply_checking_account_credit_5829"):
            if pay.get("amount") is not None:
                out["amounts"][str(pay.get("account_id"))] = float(pay["amount"])
        if nm == "submit_interest_discrepancy_report_7294" and pay.get("expected_apy") is not None:
            out["expected_apy"] = float(pay.get("expected_apy"))
            out["actual_apy"] = float(pay.get("actual_apy"))
            if pay.get("amount_difference") is not None:
                out["amounts"].setdefault(str(pay.get("account_id")),
                                          float(pay["amount_difference"]))
    return out


def opening(task):
    ins = str((task.get("user_scenario") or {}).get("instructions") or "")
    m = re.search(r'\*\*Opening:\*\*\s*"([^"]+)"', ins)
    if m:
        return m.group(1)
    m = re.search(r'\*\*Your situation:\*\*\s*(.{40,400})', ins, re.S)
    return " ".join(m.group(1).split()) if m else " ".join(ins.split())[:300]


def build_prompt(kind, docs_txt, accts, tx, ask):
    schema = SCHEMA_INTEREST if kind == "interest" else SCHEMA_FEE
    job = ("이 손님의 저축계좌 이자 지급이 정책상 받아야 할 금액과 얼마나 차이 나는지 계산하라. "
           "적용되는 APY 성분을 문서에서 모두 찾아 합산하고, 실제 지급된 이자와 비교하라."
           if kind == "interest" else
           "이 손님의 각 체크(checking) 계좌에서 **잘못 청구된 수수료**를 찾아 계좌별 환급액을 계산하라. "
           "정책 문서의 수수료 조건과 거래내역을 대조하라.")
    return [
        {"role": "system", "content": SYS},
        {"role": "user", "content":
            "# 정책 문서\n" + docs_txt +
            "\n\n# 계좌 레코드\n" + json.dumps(accts, ensure_ascii=False, indent=1) +
            "\n\n# 거래 내역\n" + json.dumps(tx, ensure_ascii=False, indent=1) +
            "\n\n# 손님 요청\n" + ask +
            "\n\n# 할 일\n" + job +
            "\n\n# 출력 형식 (JSON 하나만)\n" + schema +
            "\n\n각 quote 는 위 정책 문서에 **축자로 있는 문장**이어야 한다(검산한다). 숫자는 숫자로만 쓴다."},
    ]


def call(port, msgs, temp, maxtok=2000):
    body = json.dumps({"model": MODEL, "messages": msgs, "temperature": temp,
                       "max_tokens": maxtok}).encode("utf-8")
    req = urllib.request.Request("http://127.0.0.1:%d/v1/chat/completions" % port, data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.loads(r.read().decode("utf-8"))
    return d["choices"][0]["message"]["content"]


def parse_json(txt):
    t = txt.strip()
    t = re.sub(r"^```(?:json)?|```$", "", t, flags=re.M).strip()
    i, j = t.find("{"), t.rfind("}")
    if i < 0 or j <= i:
        return None
    try:
        return json.loads(t[i:j + 1])
    except Exception:
        try:
            return json.loads(re.sub(r",\s*([}\]])", r"\1", t[i:j + 1]))
        except Exception:
            return None


def near(a, b, tol=0.011):
    try:
        return abs(float(a) - float(b)) <= tol
    except Exception:
        return False


def score(kind, obj, docs_txt, gold):
    """결정론 채점 — LLM 판단 0."""
    r = {"parsed": obj is not None, "quote_ok": None, "arith_ok": None, "final_ok": False,
         "apy_ok": None, "detail": ""}
    if not obj:
        return r
    flat = " ".join(docs_txt.split())
    quotes, ok = [], 0
    if kind == "interest":
        comps = obj.get("components") or []
        quotes = [str(c.get("quote") or "") for c in comps]
        try:
            s = sum(float(c.get("rate")) for c in comps)
            r["arith_ok"] = near(s, obj.get("expected_apy"), 0.011)
        except Exception:
            r["arith_ok"] = False
        # 자기정합 ②: 금액차 = 실제이자 × (expected/actual − 1)
        try:
            ai = float(obj.get("actual_interest_credited"))
            ea, aa = float(obj.get("expected_apy")), float(obj.get("actual_apy"))
            pred = ai * (ea / aa - 1.0) if aa else None
            if pred is not None and not near(pred, obj.get("amount_difference"), 0.5):
                r["detail"] += "금액차 자기정합✗(pred %.2f vs %s) " % (pred, obj.get("amount_difference"))
                r["arith_ok"] = False
        except Exception:
            pass
        gamt = list(gold["amounts"].values())
        r["final_ok"] = any(near(obj.get("amount_difference"), g, 0.011) for g in gamt)
        if gold["expected_apy"] is not None:
            r["apy_ok"] = (near(obj.get("expected_apy"), gold["expected_apy"], 0.0011)
                           and near(obj.get("actual_apy"), gold["actual_apy"], 0.0011))
        r["detail"] += "amt=%s exp_apy=%s act_apy=%s" % (obj.get("amount_difference"),
                                                        obj.get("expected_apy"), obj.get("actual_apy"))
    else:
        accs = obj.get("accounts") or []
        arith = True
        got = {}
        for a in accs:
            items = a.get("items") or []
            quotes += [str(x.get("quote") or "") for x in items]
            try:
                s = sum(float(x.get("charged")) - float(x.get("correct")) for x in items)
                if not near(s, a.get("refund"), 0.011):
                    arith = False
            except Exception:
                arith = False
            try:
                got[str(a.get("account_id"))] = float(a.get("refund"))
            except Exception:
                pass
        r["arith_ok"] = arith
        gm = {k: v for k, v in gold["amounts"].items()}
        pos = {k: v for k, v in got.items() if v}
        r["final_ok"] = (set(pos.keys()) == set(gm.keys())
                         and all(near(pos[k], gm[k], 0.011) for k in gm))
        r["detail"] = json.dumps(got, ensure_ascii=False)[:120]
    if quotes:
        for q in quotes:
            qq = " ".join(str(q).split())
            if len(qq) >= 12 and qq[:60] in flat:
                ok += 1
        r["quote_ok"] = ok / float(len(quotes))
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--tasks", default="task_093,task_094,task_073,task_074")
    ap.add_argument("--arms", default="A_min,A_focus,C_neg")
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()

    docs, tasks, db = load_docs(), load_tasks(), load_db()
    by_title = {}
    for did, d in docs.items():
        by_title.setdefault(d["title"], did)
    unrelated = [did for did, d in docs.items()
                 if "Regulation E" in d["title"] or "credit card" in d["title"].lower()][:6]

    jobs = []
    for tid in a.tasks.split(","):
        sp = SPEC[tid]
        t = tasks[tid]
        uid, _u = user_by_name(db, sp["who"])
        accts, tx = records_for(db, uid)
        gold = gold_of(t, sp["kind"])
        ask = opening(t)
        req = [d for d in (t.get("required_documents") or []) if d in docs]
        gov = [by_title[g] for g in sp["gov"] if g in by_title]
        packs = {
            "A_min": req or gov,
            "A_focus": gov,
            "C_neg": unrelated,
        }
        for arm in a.arms.split(","):
            ids = packs[arm]
            dtxt = "\n\n".join("## %s (%s)\n%s" % (docs[i]["title"], i, docs[i]["content"]) for i in ids)
            msgs = build_prompt(sp["kind"], dtxt, accts, tx, ask)
            for k in range(a.n):
                jobs.append({"task": tid, "arm": arm, "k": k, "kind": sp["kind"],
                             "msgs": msgs, "docs": dtxt, "gold": gold,
                             "temp": (0.0 if k == 0 else a.temp)})

    print("작업 %d건 (태스크 %d × 팔 %d × n %d) · 모델 %s"
          % (len(jobs), len(a.tasks.split(",")), len(a.arms.split(",")), a.n, MODEL))
    lock = threading.Lock()
    out = []

    def work(idx):
        while True:
            with lock:
                if not jobs:
                    return
                j = jobs.pop(0)
            port = PORTS[idx % len(PORTS)]
            try:
                txt = call(port, j["msgs"], j["temp"])
            except Exception as e:
                txt = "ERROR " + str(e)[:200]
            obj = parse_json(txt)
            s = score(j["kind"], obj, j["docs"], j["gold"])
            row = {"task": j["task"], "arm": j["arm"], "k": j["k"], "temp": j["temp"],
                   "raw": txt[:1200]}
            row.update(s)
            with lock:
                out.append(row)
                print("  %-9s %-8s k=%d parsed=%s quote=%s arith=%s final=%s %s"
                      % (row["task"], row["arm"], row["k"], row["parsed"],
                         ("%.2f" % row["quote_ok"]) if row["quote_ok"] is not None else "-",
                         row["arith_ok"], row["final_ok"], str(row["detail"])[:70]))

    ths = [threading.Thread(target=work, args=(i,)) for i in range(a.workers)]
    [t.start() for t in ths]
    [t.join() for t in ths]

    print("\n## 요약 (태스크 × 팔)")
    print("%-10s %-8s %5s %7s %7s %7s %6s" % ("task", "arm", "n", "parsed", "quote", "arith", "final"))
    agg = collections.defaultdict(list)
    for r in out:
        agg[(r["task"], r["arm"])].append(r)
    for k in sorted(agg):
        rs = agg[k]
        qs = [r["quote_ok"] for r in rs if r["quote_ok"] is not None]
        print("%-10s %-8s %5d %7.2f %7s %7.2f %6.2f"
              % (k[0], k[1], len(rs),
                 sum(1 for r in rs if r["parsed"]) / float(len(rs)),
                 ("%.2f" % (sum(qs) / len(qs))) if qs else "-",
                 sum(1 for r in rs if r["arith_ok"]) / float(len(rs)),
                 sum(1 for r in rs if r["final_ok"]) / float(len(rs))))

    o = os.path.normpath(os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026",
                                      "x394_calc_formalize_iso.json"))
    io.open(o, "w", encoding="utf-8").write(json.dumps(out, ensure_ascii=False, indent=1))
    print("\n원자료: %s" % o)
    return 0


if __name__ == "__main__":
    sys.exit(main())
