# -*- coding: utf-8 -*-
r"""x541 - 선언 인자 ↔ 정책 행 **동일성 조인**을 결정점에 놓으면 답이 바뀌는가 (무료·2026-08-25)

## 무엇을 재나

040 의 열린 축 하나 — `eligible_for_provisional_credit`. 라이브(t7354 grpB1 t0)는 gold 분쟁
8건 전부에 `'Yes'` 를 보냈고 gold 는 그중 **2건만** True 다. 그 인자에 대해 A3 는 행을 갖고 있다:

    axis = eligible_for_provisional_credit
    quote = "eligible_for_provisional_credit (boolean) - Agent must determine this based on the
             Provisional Credit Eligibility Guidelines article in this knowledge base.
             Pass true or false."

즉 정책은 *"이 값은 그 문서를 보고 정하라"* 고 말한다([[64]] 무엇을 하면 풀리나). 물음은 하나다 —
**그 줄을 결정점에 놓으면 모델의 답이 달라지는가.**

## 팔 ([[57]] 부정통제)

    A_asis    분쟁 write 직전 창 그대로              ← 라이브 오답(전건 'Yes')을 재현해야 한다
    B_join    창 + **동일성 조인 산출 축자**          ← 수리 후보(엔진이 낼 바로 그 문자열)
    N_axis    창 + **같은 A3 의 축 안 맞는 행**(비슷한 길이) ← 글이 늘어서가 아님을 가른다

N_axis 가 A3 에서 나온다는 점이 중요하다 — 출처·문체·형식이 같고 **다른 것은 축이 맞느냐뿐**이다.

## 채점 - gold 미접촉

이 프로브는 **정답을 판정하지 않는다**. A3 가 준 것은 규칙이 아니라 *어디를 보라*는 포인터이고,
그 문서는 이 궤적에 오지 않았다. 그래서 재는 것은 **답의 분포**다:
`true`/`false`/기타 를 팔별로 세고 원문을 남긴다.

  반증조건: B_join 의 분포가 A_asis 와 같으면 이 조인은 이 자리에서 **아무것도 배달하지 않는다**.
  ⚠분포가 달라져도 *옳아졌다*는 뜻은 아니다 — 그 판정은 gold 를 봐야 하고 여기서는 하지 않는다.

사용: (리모트·cwd=scripts/distill/tau2) py -3 x541_arg_policy_join_iso.py --port 8141 --n 4
"""
import argparse
import collections
import gzip
import io
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
SIMS = os.path.abspath(os.path.join(HERE, "..", "..", "..",
                                    "reports", "facet_rft_2026", "sim_results"))
REP = os.path.abspath(os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026"))
MODEL = "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8"
TASK = "task_040"
TOOL = "file_credit_card_transaction_dispute_4829"
ARG = "eligible_for_provisional_credit"
RUNS = ("bank_t7354_grpB1_20260825",)
NL = chr(10)
ASK = (NL + NL + "You are filing this credit card dispute now." + NL +
       "Reply with ONLY the value you will pass as `" + ARG + "` - "
       "the single word true or false, nothing else.")


class _M(object):
    def __init__(self, c):
        self.content = c


def gen(port, body, maxtok=12):
    import urllib.request
    payload = {"model": MODEL, "temperature": 0.0, "max_tokens": maxtok,
               "messages": [{"role": "user", "content": body}]}
    req = urllib.request.Request("http://127.0.0.1:%d/v1/chat/completions" % port,
                                 data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read().decode("utf-8"))["choices"][0]["message"]["content"]


def foil(a2, join_txt, used_axes):
    """부정통제 — 같은 A3 에서 **축이 안 맞는** 행을 비슷한 길이만큼 모은다.

    ⚠고르지 않는다: 축 이름 사전순으로 앞에서부터 채운다(결정론·점수 0).
    """
    import t2_gate_patch as G
    want = len(join_txt)
    out, n = [], 0
    for r in sorted(G._policy_facts(a2), key=lambda x: (str(x.get("axis")), str(x.get("value")))):
        ax = str(r.get("axis") or "")
        if not ax or ax in used_axes:
            continue
        for s in (r.get("sources") or []):
            q = str(s.get("quote") or "").strip()
            if not q:
                continue
            out.append("- %s: %s" % (ax, q))
            n += len(out[-1]) + 1
            break
        if n >= want:
            break
    return NL.join(out)


def windows():
    W = 12
    import t2_gate_patch as G
    from gate_interpreter import load_domain_a2
    a2 = load_domain_a2("banking_knowledge")
    cases, skipped = [], []
    for tag in RUNS:
        p = os.path.join(SIMS, tag + ".results.json.gz")
        if not os.path.exists(p):
            continue
        d = json.load(gzip.open(p, "rt", encoding="utf-8", errors="replace"))
        for s in (d.get("simulations") or []):
            if s.get("task_id") != TASK:
                continue
            ms = s.get("messages") or []
            sim = "%s#s%s" % (s.get("task_id"), s.get("seed"))
            cut = None
            for i, m in enumerate(ms):
                blob = json.dumps(m.get("tool_calls") or [], ensure_ascii=False)
                if TOOL in blob and "unlock" not in blob and ARG in blob:
                    cut = i
                    break
            if cut is None:
                skipped.append({"sim": sim, "why": "그 인자를 쓴 분쟁 호출이 없다"})
                continue
            declared = list(G._declared_params_by_tool(
                [_M(str(m.get("content") or "")) for m in ms[:cut]]).get(TOOL) or {})
            join = G._policy_rows_for(a2, declared)
            if not join:
                skipped.append({"sim": sim, "why": "조인이 0행 - 선언 인자 %d" % len(declared)})
                continue
            used = {ln[2:].split(": ", 1)[0] for ln in join.splitlines()}
            txt = []
            for m in ms[max(0, cut - W):cut]:
                c = str(m.get("content") or "").strip()
                if c:
                    txt.append("[%s] %s" % (m.get("role"), c[:1500]))
            if not txt:
                skipped.append({"sim": sim, "why": "창이 비었다"})
                continue
            cases.append({"sim": sim, "win": (NL + NL).join(txt), "join": join,
                          "foil": foil(a2, join, used), "declared": len(declared)})
    return cases, skipped


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8141)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--out", default=os.path.join(REP, "x541_arg_policy_join_2026_08_25.json"))
    a = ap.parse_args(argv)
    cases, skipped = windows()
    for sk in skipped:
        print("건너뜀 %-22s %s" % (sk["sim"], sk["why"]))
    if not cases:
        io.open(a.out, "w", encoding="utf-8").write(json.dumps(
            {"probe": "x541", "cases": 0, "skipped": skipped}, ensure_ascii=False, indent=1))
        print("창 0 - 판정하지 않는다")
        return 1
    print("창 %d개" % len(cases))
    for c in cases:
        print("  %-22s 선언 인자 %d · 조인 %d자 · 통제 %d자"
              % (c["sim"], c["declared"], len(c["join"]), len(c["foil"])))
    rows = []
    dist = collections.defaultdict(collections.Counter)
    for c in cases:
        arms = {"A_asis": c["win"],
                "B_join": c["win"] + NL + NL + "[policy]" + NL + c["join"]}
        if c["foil"]:
            arms["N_axis"] = c["win"] + NL + NL + "[policy]" + NL + c["foil"]
        for arm, body in sorted(arms.items()):
            for k in range(a.n):
                try:
                    txt = gen(a.port, body + ASK)
                except Exception as e:
                    txt = "!!%r" % (e,)
                m = re.search(r"\b(true|false)\b", txt, re.I)
                got = m.group(1).lower() if m else "(기타)"
                dist[arm][got] += 1
                rows.append({"sim": c["sim"], "arm": arm, "k": k, "got": got, "raw": txt[:90]})
                print("%-7s %-22s k=%d -> %-7s %s" % (arm, c["sim"], k, got, txt[:40].replace(NL, " ")),
                      flush=True)
    same = dict(dist.get("A_asis") or {}) == dict(dist.get("B_join") or {})
    out = {"probe": "x541", "date": "2026-08-25", "task": TASK, "arg": ARG,
           "scoring": "정답 판정 없음 - 팔별 답 분포만. A3 가 준 것은 규칙이 아니라 포인터이고 "
                      "그 문서는 이 궤적에 오지 않았다. gold 미접촉.",
           "falsifier": "B_join 의 분포가 A_asis 와 같으면 이 조인은 이 자리에서 아무것도 "
                        "배달하지 않는다. 달라져도 '옳아졌다'는 뜻은 아니다.",
           "dist": {k: dict(v) for k, v in dist.items()},
           "join_delivers_nothing": same, "skipped": skipped, "rows": rows}
    io.open(a.out, "w", encoding="utf-8").write(json.dumps(out, ensure_ascii=False, indent=1))
    print(NL + "== 분포 ==")
    for k in sorted(dist):
        print("  %-7s %s" % (k, dict(dist[k])))
    print("B_join == A_asis ?  %s" % ("**같다 = 배달 0**" if same else "다르다"))
    print("->", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
