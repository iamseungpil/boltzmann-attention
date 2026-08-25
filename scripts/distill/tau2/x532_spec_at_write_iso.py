# -*- coding: utf-8 -*-
r"""x532 — 085 격리: **write 결정점에 도구 명세를 되붙이면 인자 키가 맞는가** (무료·2026-08-25)

## 왜 이 물음인가 (진단이 오늘 바뀌었다)

핸드오프 2026-08-25 §2 는 085 를 *"파라미터 **키 허용목록** 레버가 필요하다"* 로 남겼다.
궤적 직독(`bank_t7348_halfB_20260824`·085 두 sim)이 그 전제를 깼다:

  · `unlock_discoverable_agent_tool` 의 **반환문**이 파라미터 17개와 enum 4종을 전부 담아
    **msg22** 에 도착한다(2,975자·양 sim 바이트 동일).
  · 첫 오답 write 는 **msg68 / msg80**. 거리 **46 · 58**.
  · 그 사이 모델은 `debit_card_id`·`category`·`police_report_file`·`date_first_noticed`·
    `type_of_transaction`·`physical_card_in_possession`… 를 지어내며 13턴을 태운다.

⇒ 재료 부재가 아니라 **거리**다 — 큐 `plan_2026_08_24_pm.common_diagnosis` 축자
   *"재료는 상류에 있고 결정점에 없다"* 와 같은 모양(①금액: 정책 msg3 ↔ 크레딧 msg43~68).
⇒ 그러므로 A2 키 선언은 **틀린 수리**다. 옳은 후보는 *env 가 이미 보낸 그 블록을 결정점에
   되붙이기* 이고, 그것은 도메인 낱말 0 · 텍스트 파싱 0([[59]]) · 선택 0([[62]]③④)이다.

## 배선 지점은 이미 있고 **침묵한다**

라이브 로그 축자(같은 런·085): `[T2_DECIDE_BEFORE_WRITE] 축 미상 — 무발화
tool=file_debit_card_transaction_dispute (A2 가 이 write 의 선택 인자를 선언하지 않았다)`
— **33회**. 그 가지가 이 프로브가 통과하면 붙일 자리다.

## 팔 ([[57]] 부정통제 포함)

    A_asis   실패 직전 창 그대로                  ← **계기 생존 검사**: 라이브 오답 키를 재현해야 한다
    B_spec   창 + **그 도구의 unlock 반환문 축자**  ← 수리 후보
    N_neg    창 + **같은 길이의 무관한 도구 결과**  ← 길이·부하가 아니라 내용임을 가른다

A_asis 가 오답 키를 재현 못 하면 격리가 불공정한 것이고 **판정하지 않는다**([[62]] 2b).

## 채점 — 닫힌 술어만·gold 미접촉([[23]])

시그니처 키 집합(`tools.py:942-958` 축자)에 대해 ①낸 키 전부가 그 안인가(`keys_ok`)
②밖의 키 몇 개인가(`extra`). **어느 값이 옳은지는 채점하지 않는다** — gold 를 안 본다.

사용: (리모트·cwd=scripts/distill/tau2) py -3 x532_spec_at_write_iso.py --port 8140
"""
import argparse
import gzip
import io
import json
import os
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
SIMS = os.path.abspath(os.path.join(HERE, "..", "..", "..",
                                    "reports", "facet_rft_2026", "sim_results"))
OUT = os.path.abspath(os.path.join(HERE, "..", "..", "..",
                                   "reports", "facet_rft_2026",
                                   "x532_spec_at_write_2026_08_25.json"))
MODEL = "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8"
TASK = "task_085"
TOOL = "file_debit_card_transaction_dispute_6281"
RUNS = ("bank_t7348_halfB_20260824",)

# 시그니처 키 — tau2-bench/src/tau2/domains/banking_knowledge/tools.py:942-958 축자.
# 채점 전용이다(엔진에 안 들어간다). gold 아님·env 오류문 아님([[23]]).
SIG_KEYS = ["transaction_id", "account_id", "card_id", "user_id", "dispute_category",
            "transaction_date", "discovery_date", "disputed_amount", "transaction_type",
            "card_in_possession", "pin_compromised", "contacted_merchant",
            "police_report_filed", "written_statement_provided",
            "provisional_credit_eligible", "customer_max_liability_amount", "card_action"]

ASK = ("\n\nYou are about to call the tool `%s`.\n"
       "Reply with ONLY the JSON object of arguments for that call - no prose, no code fence.\n"
       % TOOL)


def gen(port, body, maxtok=700):
    payload = {"model": MODEL, "temperature": 0.0, "max_tokens": maxtok,
               "messages": [{"role": "user", "content": body}]}
    req = urllib.request.Request("http://127.0.0.1:%d/v1/chat/completions" % port,
                                 data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read().decode("utf-8"))["choices"][0]["message"]["content"]


def keys_of(txt):
    """산출에서 최상위 JSON 객체의 키 — 파싱 실패면 None(판정하지 않는다·[[25]])."""
    i, j = txt.find("{"), txt.rfind("}")
    if i < 0 or j <= i:
        return None
    try:
        d = json.loads(txt[i:j + 1])
    except Exception:
        return None
    if not isinstance(d, dict):
        return None
    if len(d) == 1 and isinstance(list(d.values())[0], dict):
        d = list(d.values())[0]          # {"arguments": {...}} 형태
    return sorted(str(k) for k in d)


def render(msgs):
    out = []
    for m in msgs:
        c = str(m.get("content") or "")
        tc = m.get("tool_calls") or []
        if tc and not c:
            c = json.dumps(tc, ensure_ascii=False)[:1200]
        if not c:
            continue
        out.append("[%s] %s" % (m.get("role"), c[:2400]))
    return "\n\n".join(out)


def windows(limit=0):
    """(sim, 창, unlock 반환문, 같은 길이 무관 블록) — 전부 궤적 축자·지어냄 0.

    창 = **실패 write 직전** 의 마지막 W 메시지. unlock 반환문은 구조로만 찾는다:
    `unlock_discoverable_agent_tool` 호출 **바로 뒤의 tool 메시지**(텍스트 파싱 0·[[59]]).
    """
    W = 10
    cases = []
    for tag in RUNS:
        rp = os.path.join(SIMS, tag + ".results.json.gz")
        if not os.path.exists(rp):
            continue
        d = json.load(gzip.open(rp, "rt", encoding="utf-8", errors="replace"))
        for s in (d.get("simulations") or []):
            if s.get("task_id") != TASK:
                continue
            ms = s.get("messages") or []
            spec, filler = None, None
            for i, m in enumerate(ms):
                if m.get("role") != "assistant":
                    continue
                blob = json.dumps(m.get("tool_calls") or [], ensure_ascii=False)
                if "unlock_discoverable_agent_tool" in blob and TOOL in blob:
                    for j in range(i + 1, min(i + 4, len(ms))):
                        if ms[j].get("role") == "tool":
                            spec = str(ms[j].get("content") or "")
                            break
                if spec:
                    break
            if not spec:
                continue
            for m in ms:                       # 부정통제: 같은 길이의 **무관한** 도구 결과
                c = str(m.get("content") or "")
                if (m.get("role") == "tool" and TOOL not in c
                        and abs(len(c) - len(spec)) < len(spec)):
                    filler = c[:len(spec)]
            hits = [i for i, m in enumerate(ms)
                    if m.get("role") == "tool"
                    and str(m.get("content") or "").startswith("Error: Invalid arguments")]
            for i in hits[:3]:                 # sim 당 앞 3개 — 뒤쪽은 같은 창의 반복이다
                cases.append({"sim": "%s#s%s" % (s.get("task_id"), s.get("seed")),
                              "at": i, "win": render(ms[max(0, i - W):i - 1]),
                              "spec": spec, "filler": filler or spec[:200]})
    return cases[:limit] if limit else cases


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8140)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(argv)
    cases = windows(a.limit)
    if not cases:
        print("no windows")
        return 1
    rows, agg = [], {}
    for c in cases:
        arms = {"A_asis": c["win"],
                "B_spec": c["win"] + "\n\n[tool] " + c["spec"],
                "N_neg": c["win"] + "\n\n[tool] " + c["filler"]}
        for name, body in arms.items():
            try:
                txt = gen(a.port, body + ASK)
            except Exception as e:
                txt = "!!%r" % (e,)
            ks = keys_of(txt)
            extra = None if ks is None else [k for k in ks if k not in SIG_KEYS]
            ok = None if ks is None else (not extra)
            rows.append({"sim": c["sim"], "at": c["at"], "arm": name,
                         "keys": ks, "extra": extra, "keys_ok": ok,
                         "raw": txt[:400]})
            d = agg.setdefault(name, {"n": 0, "ok": 0, "unparsed": 0, "extra_total": 0})
            d["n"] += 1
            if ks is None:
                d["unparsed"] += 1
            else:
                d["ok"] += 1 if ok else 0
                d["extra_total"] += len(extra)
            print("%-8s %-18s at=%-4s ok=%s extra=%s" % (name, c["sim"], c["at"], ok, extra),
                  flush=True)
    fair = agg.get("A_asis", {}).get("ok", 0) < agg.get("A_asis", {}).get("n", 1)
    out = {"probe": "x532", "date": "2026-08-25", "task": TASK, "tool": TOOL,
           "n_windows": len(cases), "agg": agg,
           "instrument_survives": fair,
           "instrument_note": ("A_asis 가 오답 키를 재현했다 — 격리가 공정하다"
                               if fair else
                               "A_asis 가 전부 통과했다 = 이 창은 라이브 실패를 재현하지 못한다. "
                               "판정하지 않는다([[62]] 2b)."),
           "rows": rows}
    io.open(OUT, "w", encoding="utf-8").write(json.dumps(out, ensure_ascii=False, indent=1))
    print("\n== agg ==")
    for k, v in agg.items():
        print(" %-8s ok=%d/%d unparsed=%d extra_keys=%d" % (k, v["ok"], v["n"],
                                                            v["unparsed"], v["extra_total"]))
    print(out["instrument_note"])
    print("->", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
