# -*- coding: utf-8 -*-
"""X25 — 기준(criterion) 프로브: wrong-pick의 상류가 **계산**인가 (2026-07-31·GPU만·gold 불필요).

**왜**: C266이 남긴 가설 — 후보를 4~7개로 줄이고 그 안에 gold가 있어도 못 고르고 순서에 흔들린다
(안정 0/16) ⇒ *"고를 기준이 안 서 있다"*. 이 태스크군의 기준은 대개 **레코드에서 파생되는 수치**다
(예: 정책상 기대 적립 vs 실제 `rewards_earned`의 불일치). 그렇다면 결손은 F3(참조매칭)이 아니라
**F2 symbolic(도출·비교)**이고, 처방은 결정론 *선택기*가 아니라 **결정론 계산 → 선택**이다(§1.2).

**gold를 안 쓴다**([[23]]·[[03b]]): 채점은 **산술**로 한다 — 모델이 레코드마다 내놓은 파생값을
우리가 레코드 원문의 숫자로 재계산해 맞는지 본다. 정답 id를 몰라도 판정된다.

측정(사전 선언):
  ① 기준 진술 — 모델이 "무엇을 비교해야 하는지"를 **필드 이름 수준으로** 대는가
  ② 파생값 산술 — 레코드별 기대 적립을 내놓게 하고 **우리가 재계산**해 일치율을 본다
  ③ 불일치 지목 — ②의 자기 값 기준으로 **자기 답이 내부일관**한가(값과 지목이 맞물리는가)
  ★①이 되고 ②가 안 되면 = **계산 결손**(F2) · ①부터 안 되면 = 기준 미형성 · 둘 다 되는데 ③이
   깨지면 = **집행 결손**(알고도 그 값을 못 쓴다 = 결정론 실행으로 닫히는 자리)

용법: py -3 x25_criterion_probe.py --cases txn_cases_v4.jsonl --base http://…/v1 --out r.jsonl
"""
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import x22_txnid_isoprobe as X22   # noqa: E402

TXN = X22.TXN
NUM = re.compile(r"-?\d[\d,]*\.?\d*")
FIELD = re.compile(r"^\s*([a-z_]+)\s*:\s*(.+?)\s*$", re.M)


def records_of(case):
    """후보 id → {필드: 값} (레코드 원문 파싱·도메인 리터럴 0)."""
    out = {}
    for t, blk in zip(case["candidates"], case.get("cand_records") or []):
        d = {}
        for m in FIELD.finditer(str(blk)):
            d[m.group(1)] = m.group(2)
        out[t] = d
    return out


def ask(model, base, prompt):
    import litellm
    try:
        r = litellm.completion(model="openai/" + model, api_base=base, api_key="x",
                               temperature=0.0, messages=[{"role": "user", "content": prompt}])
        return (r.choices[0].message.content or "").strip()
    except Exception as e:
        return "ERROR: %r" % (e,)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    cases = [json.loads(l) for l in open(a.cases, encoding="utf-8")]
    rows = []
    for c in cases:
        recs = records_of(c)
        base_ctx = X22.prompt_for(c, "A_policy").split("\n\n", 1)[-1]

        # ① 기준 진술 — 필드 이름으로 답하게 한다(자유 서술이면 채점이 흐려진다)
        q1 = (base_ctx +
              "\n\nBefore choosing anything: which record FIELDS must be compared to decide, and "
              "what is the rule? Answer in one line as: FIELDS=<comma separated field names> | "
              "RULE=<one sentence>.")
        a1 = ask(a.model, a.base, q1)
        fields = re.findall(r"[a-z_]{4,}", a1.split("RULE")[0].lower())
        known = set().union(*[set(d) for d in recs.values()]) if recs else set()
        named = sorted(set(f for f in fields if f in known))

        # ② 파생값 — 레코드별로 '있어야 할 값'을 내게 하고 우리가 재계산해 채점
        q2 = (base_ctx +
              "\n\nFor EACH transaction listed, state the value that the rule says it SHOULD have, "
              "as JSON: {\"<transaction_id>\": <number>, ...}. Numbers only, no prose.")
        a2t = ask(a.model, a.base, q2)
        got = {}
        for m in re.finditer(r"(txn_[0-9a-f]+)\D{0,12}(-?\d[\d,]*\.?\d*)", a2t):
            got[m.group(1)] = m.group(2).replace(",", "")

        # ③ 내부일관 — 자기 값과 실제 값이 어긋나는 레코드를 자기가 지목하는가
        q3 = (base_ctx +
              "\n\nUsing your own computation, which transaction_id has a MISMATCH between the "
              "expected value and the recorded one? Answer with exactly one id and nothing else.")
        a3 = ask(a.model, a.base, q3)
        pick = next(iter(TXN.findall(a3)), None)

        # 자기값 대비 실제값이 다른 후보 집합(우리가 계산) — gold 미사용
        def numof(t, key_hint=("rewards_earned", "points", "amount")):
            d = recs.get(t) or {}
            for k in d:
                if any(h in k for h in key_hint):
                    n = NUM.search(d[k].replace(",", ""))
                    if n:
                        return float(n.group(0))
            return None
        selfmis = []
        for t, v in got.items():
            actual = numof(t)
            try:
                if actual is not None and abs(float(v) - actual) > 1e-6:
                    selfmis.append(t)
            except Exception:
                pass
        row = {"task": c["task"], "n_cand": len(c["candidates"]),
               "fields_named": named, "n_fields": len(named),
               "n_values": len(got), "self_mismatch": len(selfmis),
               "pick": pick, "pick_in_selfmis": (pick in selfmis) if pick else False,
               "pick_is_gold": (pick == c["gold"])}
        rows.append(row)
        print("  %-10s 필드 %-28s 값 %2d/%2d · 자기불일치 %2d · 지목 일관=%-5s (gold=%s)"
              % (row["task"], ",".join(named)[:28], row["n_values"], row["n_cand"],
                 row["self_mismatch"], row["pick_in_selfmis"], row["pick_is_gold"]))

    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n = len(rows)
    print("\n=== 판정")
    print(" ① 기준을 필드명으로 댄 사례: %d/%d (평균 %.1f개)"
          % (sum(1 for r in rows if r["n_fields"]), n, sum(r["n_fields"] for r in rows) / n))
    print(" ② 후보 전건에 파생값을 낸 사례: %d/%d"
          % (sum(1 for r in rows if r["n_values"] >= r["n_cand"]), n))
    print(" ③ 자기 계산과 지목이 일관: %d/%d" % (sum(1 for r in rows if r["pick_in_selfmis"]), n))
    print("    (참고) 지목이 gold와 일치: %d/%d" % (sum(1 for r in rows if r["pick_is_gold"]), n))
    print("\n읽는 법: ①O·②X = **계산 결손(F2)** / ①X = 기준 미형성 / ①O·②O·③X = **집행 결손**")


if __name__ == "__main__":
    main()
