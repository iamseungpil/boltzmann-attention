# -*- coding: utf-8 -*-
r"""x481 — 격리 서브가 **잔액-조건부 base APY 티어**를 고를 수 있는가 (2026-08-22·무료·[[62]] 측정)

## 왜 (093 원인 확정 뒤의 유일한 열린 질문)
x480 이 근인을 확정했다: 093 은 네 런 전부 같은 두 변이가 MISSING 이고, 갈린 것은 **base APY**
하나다 — 산출 `2.775` ↔ 정답 `4.275`, 차이 **1.5**.
정책 문서가 그 1.5 를 설명한다(`doc_savings_accounts_silver_account_003` 축자):

    | Balance tier          | Balance range      | APY  |
    | Below threshold       | Less than $10,000  | 2.5% |
    | At or above threshold | At least $10,000   | 4.0% |

계좌 잔액은 **144,000** 이므로 문서상 base 는 `4.0` 이다. 검산도 맞는다 —
`144000 × (4.275 − 4.0)/100 / 12 = 33.0` = gold amount.
즉 서브는 **두 숫자 중 낮은 쪽을 집었다**. 그런데 선언을 보면 서브가 받는 REFERENCE 는
`savings_account_type` 과 `customer_products` 뿐이고 **잔액이 없다** — 티어를 고를 근거가
원리상 없다. 지시문도 base 가 조건부라는 사실을 말하지 않는다("the account type's base APY").

⇒ 여기서 재는 것은 딱 하나: **잔액을 주면 서브가 티어를 옳게 고르는가.**
   되면 레버는 **전달**뿐이다([[62]] ②). 줘도 안 되면 그 단계에만 결정론을 얹는다([[62]] ③).
   ⛔이 프로브 **전에** 수리를 짓지 않는다 — 사용자 지시 *"프로브부터 재라"*.

## 팔 (한 번에 한 변수만)
    A_asis      현행 선언 그대로 (REFERENCE = 계좌종류 + 손님상품)      ← 기준선
    B_bal       + REFERENCE 에 **잔액** 한 줄                            ← 재료만 추가
    C_bal_hint  + 잔액 + **base 가 잔액-조건부일 수 있다**는 지시 한 문장 ← 재료 + 지시
    N_neg       A_asis 를 같은 횟수 반복                                 ← 부정통제([[57]])
⇒ N_neg 는 *"그냥 여러 번 굴리면 우연히 4.0 이 나오는가"* 를 막는다. A 와 N 이 같으면
  B/C 의 차이는 재료·지시 덕이고, N 만으로도 4.0 이 나오면 이 측정은 무효다.

## 채점 (gold 미참조·[[23]]·[[69]])
정답은 **문서와 잔액**에서 나온다: 잔액 ≥ 임계면 높은 티어. gold(4.275)는 x480 에서 이미 봤고
여기서는 `reward_info` 를 **열지 않는다**. 세는 것 —
    returned    서브가 dict 를 돌려줬나
    base_value  components 의 kind=='base' 값
    tier_ok     base 가 **문서의 높은 티어 값**과 같은가 (임계·값 모두 문서에서 읽는다)
    gate1_kept  관문1 이 드롭하지 않은 component 수(인용 검증 통과분)

## 재료·환경
선언은 A3 정본에서 읽고([[71]] 2항), 전달은 `T2_SG_DOCS=1` = 엔진이 선언된 문서 id 를 잘라
넘긴다(검색 0·[[71]] 3항). 도구·모델·인자는 라이브와 같은 것을 쓴다(`x448.Sandbox` 재사용·
`x456` 과 동형·사본 금지 [[67]]). ref 조합은 **실제 궤적에서 에이전트 자신이 낸 인자**다.

사용: (리모트·cwd=tau2) py x481_apy_tier_probe.py --port 8141 --n 6
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

import t2_scaffold_get as SG            # noqa: E402  피측정 대상(정본)
import x448_index_vs_all_iso as IVA     # noqa: E402  Sandbox 재사용([[67]])

REP = os.path.abspath(os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026"))
BASE = os.path.join(REP, "sim_results")
DECL = "get_correct_savings_apy"
TASK = "task_093"
BAL_KEY = "current_balance"
# ★지시 한 문장 — 도메인 수치 0(임계도 값도 쓰지 않는다). 문서를 읽으라고만 한다([[05]]).
HINT = (" NOTE: an account's base APY may be tiered by balance. If the documents give more than "
        "one base APY for this account type, read the tier table and use the row whose balance "
        "range contains the balance given in REFERENCE.")


def declaration():
    """선언은 A3 정본에서 읽는다 — 코드에 적지 않는다([[71]] 2항)."""
    p = os.path.join(HERE, "a2", "banking_knowledge.specific.json")
    with io.open(p, encoding="utf-8") as f:
        d = json.load(f)
    for s in (d.get("scaffold_get_tools") or []):
        if s.get("name") == DECL:
            return s
    raise SystemExit("선언 %s 없음" % DECL)


def harvest_ref():
    """093 궤적에서 **에이전트 자신이 낸** ref 인자 하나(가장 최근)."""
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
                for tc in (m.get("tool_calls") or []):
                    if tc.get("name") != DECL:
                        continue
                    ar = tc.get("arguments") or {}
                    if isinstance(ar, str):
                        try:
                            ar = json.loads(ar)
                        except Exception:
                            ar = {}
                    ref = {k: ar[k] for k in ("savings_account_type", "customer_products")
                           if ar.get(k)}
                    if len(ref) == 2:
                        return ref, os.path.basename(p)
    return None, None


def harvest_balance():
    """잔액도 **궤적의 도구 출력**에서 읽는다(우리가 짓지 않는다·gold 미참조)."""
    pat = re.compile(r"current_holdings:\s*([0-9][0-9,\.]*)")
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
                c = str(m.get("content") or "")
                if "sav_" not in c or "current_holdings" not in c:
                    continue
                vals = [float(x.replace(",", "")) for x in pat.findall(c)]
                if vals:
                    return max(vals), os.path.basename(p)
    return None, None


def tier_truth(env):
    """문서의 티어 표에서 **높은 티어 값과 임계**를 읽는다 (gold 미참조·엔진 리터럴 0).

    술어는 닫혀 있다: 'At least $X' / 'Less than $X' 두 줄과 그 행의 % 값만 본다.
    """
    best = None
    for t in SG._corpus_texts(_ORCH_HOLDER[0], ["kb"]):
        if "Balance tier" not in t or "Balance range" not in t:
            continue
        hi = re.search(r"At least \$?([0-9][0-9,\.]*)\s*\|\s*([0-9\.]+)\s*%", t)
        lo = re.search(r"Less than \$?([0-9][0-9,\.]*)\s*\|\s*([0-9\.]+)\s*%", t)
        if hi:
            cand = {"threshold": float(hi.group(1).replace(",", "")),
                    "high": float(hi.group(2)),
                    "low": float(lo.group(2)) if lo else None}
            if best is None:
                best = cand
    return best


_ORCH_HOLDER = [None]


class _Orch(object):
    """서브·관문1 이 요구하는 최소 표면 (x456 과 동형·라이브와 같은 env/모델 문자열)."""

    def __init__(self, tool_names, model, base, env):
        import types
        want = set(tool_names or [])
        tools = [t for t in (env.get_tools() or []) if getattr(t, "name", None) in want]
        self.agent = types.SimpleNamespace(
            tools=tools, llm="openai/%s" % model,
            llm_args={"api_base": base, "api_key": "dummy", "temperature": 0.0})
        self.environment = env


def base_of(comps):
    for c in (comps or []):
        if isinstance(c, dict) and str(c.get("kind")) == "base":
            try:
                return float(c.get("value"))
            except Exception:
                return None
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8141)
    ap.add_argument("--model", default="Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8")
    ap.add_argument("--n", type=int, default=6, help="팔마다 반복 횟수")
    ap.add_argument("--arms", default="A_asis,B_bal,C_bal_hint,N_neg")
    ap.add_argument("--out", default="x481_apy_tier_probe.json")
    a = ap.parse_args()

    os.environ.setdefault("OPENAI_API_BASE", "http://localhost:%d/v1" % a.port)
    os.environ.setdefault("OPENAI_BASE_URL", "http://localhost:%d/v1" % a.port)
    os.environ.setdefault("OPENAI_API_KEY", "EMPTY")
    os.environ["T2_SG_ISOLATE"] = "1"
    os.environ["T2_SG_GROUND"] = "1"
    os.environ["T2_SG_DOCS"] = "1"          # 선언된 id 전달([[71]])·라이브와 같은 경로
    os.environ.pop("T2_SG_ISOFB", None)

    d = declaration()
    iso = d.get("isolate") or {}
    ref, src = harvest_ref()
    bal, bsrc = harvest_balance()
    if not ref:
        raise SystemExit("093 궤적에서 ref 를 못 찾았다")
    if bal is None:
        raise SystemExit("093 궤적에서 잔액을 못 찾았다")

    sb = IVA.Sandbox()
    orch = _Orch(iso.get("getter_tools") or [], a.model,
                 "http://localhost:%d/v1" % a.port, sb.env)
    _ORCH_HOLDER[0] = orch
    kb = SG._corpus_texts(orch, ["kb"])
    truth = tier_truth(orch)

    print("=" * 96)
    print("x481 · %s · ref=%s (%s)" % (DECL, json.dumps(ref, ensure_ascii=False)[:90], src))
    print("     잔액 %s (%s) · 관문1 KB %d편" % (bal, bsrc, len(kb)))
    print("     문서에서 읽은 티어: %s" % json.dumps(truth, ensure_ascii=False))
    print("=" * 96)
    if not truth:
        raise SystemExit("티어 표를 문서에서 못 읽었다 — 채점 불가(측정 중단)")
    want_base = truth["high"] if bal >= truth["threshold"] else truth["low"]
    print("  ⇒ 문서상 옳은 base = %s (잔액 %s %s 임계 %s)"
          % (want_base, bal, ">=" if bal >= truth["threshold"] else "<", truth["threshold"]))

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

    want_arms = set(x.strip() for x in a.arms.split(",") if x.strip())
    rows = []
    for arm in ("A_asis", "B_bal", "C_bal_hint", "N_neg"):
        if arm not in want_arms:
            continue
        iso_a = json.loads(json.dumps(iso))          # 팔마다 선언 사본(원본 불변)
        ctx = dict(ref)
        if arm in ("B_bal", "C_bal_hint"):
            iso_a["ref_params"] = list(iso_a.get("ref_params") or []) + [BAL_KEY]
            ctx[BAL_KEY] = bal
        if arm == "C_bal_hint":
            for k in ("instructions",):
                if iso_a.get(k):
                    iso_a[k] = str(iso_a[k]) + HINT
            if isinstance(iso_a.get("docs"), dict) and iso_a["docs"].get("instructions"):
                iso_a["docs"]["instructions"] = str(iso_a["docs"]["instructions"]) + HINT
        print("\n── %s ──────────────────────────────────────────────" % arm)
        for i in range(a.n):
            c = dict(ctx)
            try:
                got = SG._sub_fetch_formalize(orch, d, iso_a, c, run_env)
            except Exception as e:
                got = None
                print("  #%d EXC %r" % (i, e))
            comps = (got or {}).get("components") or []
            bv = base_of(comps)
            kept = None
            if got:
                merged = dict(c)
                merged.update(got)
                try:
                    SG._ground_operands(orch, d, merged)
                    kept = len(merged.get("components") or [])
                except Exception:
                    kept = None
            ok = (bv is not None and abs(bv - want_base) < 1e-9)
            rows.append({"arm": arm, "i": i, "returned": bool(got), "n_components": len(comps),
                         "base_value": bv, "tier_ok": ok, "gate1_kept": kept,
                         "components": comps})
            print("  #%d returned=%-5s comps=%-2d base=%-8s tier_ok=%-5s gate1_kept=%s"
                  % (i, bool(got), len(comps), bv, ok, kept))

    agg = collections.defaultdict(lambda: {"n": 0, "returned": 0, "tier_ok": 0, "comps": 0})
    for x in rows:
        s = agg[x["arm"]]
        s["n"] += 1
        s["returned"] += 1 if x["returned"] else 0
        s["tier_ok"] += 1 if x["tier_ok"] else 0
        s["comps"] += x["n_components"]

    payload = {"declaration": DECL, "task": TASK, "ref": ref, "balance": bal,
               "tier_truth": truth, "want_base": want_base, "rows": rows,
               "summary": {k: dict(v) for k, v in agg.items()}}
    p = os.path.join(REP, a.out)
    with io.open(p, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)

    print("\n" + "=" * 96)
    print("판정 — 문서상 옳은 base = %s" % want_base)
    for arm in ("A_asis", "N_neg", "B_bal", "C_bal_hint"):
        if arm not in agg:
            continue
        s = agg[arm]
        print("  %-11s tier_ok %d/%d · 답반환 %d/%d · component 합 %d"
              % (arm, s["tier_ok"], s["n"], s["returned"], s["n"], s["comps"]))
    print("\n[산출물] → %s" % p)
    print("해석 규칙: A≈N 이고 B(또는 C)가 크게 높으면 결손은 **재료/지시 전달**이다([[62]] ②).")
    print("           B·C 도 A 와 같으면 주는 것으로는 안 되는 것이고, 그 단계에만 결정론([[62]] ③).")


if __name__ == "__main__":
    main()
