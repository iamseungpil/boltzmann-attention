# -*- coding: utf-8 -*-
"""추천 선택은 **능력**인가 **부하**인가 — 정보-맞춘 격리 프로브([[18]]·등대 §1.4).

배경: 3런 연속 절차(검증·계좌 read·원장 조회)는 gold와 일치하는데 **마지막 선택만** 틀렸고,
런마다 다른 계좌가 제출됐다(100: Hunter Green / World Blue / Cobalt Blue). 궤적 정독 결과
에이전트는 그 턴에 **추천을 명시하지 않았고**, 손님이 언급된 것 중에서 골랐다.

그래서 물어야 할 것은 *"모델이 고를 줄 아는가"* 다. 규율([[18]]): 격리 프로브는 **에이전트가
그 지점에 실제로 갖고 있던 정보와 맞춰야** 한다. 정보가 더 빈약하면 '부하 없음'이 정보량
차이일 뿐이고, 더 풍부하면 반대로 과대평가된다.

  A_minimal  : 제약표(축별 값)만 + 질문           — 우리 층이 실제로 전달한 것과 같은 사실
  B_fullctx  : 그 sim의 결정점까지의 실제 대화 + 같은 질문

A 는 맞고 B 는 틀리면 **부하**(정보는 있는데 궤적에서 못 쓴다) — 처방은 분해·표면화 자리 조정.
A 도 틀리면 **능력/경계** — 표면화를 더 얹어도 안 움직인다.

유료 0(로컬 vllm). 실행: py -3 x149_choice_isolation.py [TAG] [N]
"""
import gzip
import json
import os
import re
import sys
import urllib.request
import collections

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

URL = os.environ.get("T2_PROBE_URL", "http://localhost:8140/v1/chat/completions")
MODEL = os.environ.get("T2_PROBE_MODEL", "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8")
HERE = os.path.dirname(os.path.abspath(__file__))
SIMS = os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026", "sim_results")

GOLD = {"task_100": "Hunter Green", "task_099": "World Blue"}
# 그 sim 의 손님 사실(궤적에 실재하는 것만 — gold 를 보고 쓴 것이 아니다)
FACTS = {
    "task_100": "The customer's first checking account was opened 65 days ago. "
                "The business being referred was just incorporated and will deposit about $31,000.",
    "task_099": "The customer has been a checking customer for about 2 years. "
                "The business being referred was just formed and will deposit about $30,000. "
                "The customer already owns Navy Blue, Cobalt Blue and Hunter Green business checking.",
}
QUESTION = ("Which ONE business checking account should this customer refer the business to, "
            "if the customer wants to maximise the bonus THEY receive? "
            "Answer with the account name only, nothing else.")


def ask(prompt, temp):
    body = json.dumps({"model": MODEL, "temperature": temp, "max_tokens": 40,
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.load(r)["choices"][0]["message"]["content"].strip()


def a3_table():
    """A3 에서 business checking 관련 축을 표로 — 우리 층이 결정점에 실제로 실어 보내는 것과 같은 사실."""
    import t2_factdag as FD
    po = json.load(open(os.path.join(HERE, "a2", "banking_knowledge.specific.json"),
                        encoding="utf-8"))["policy_ontology"]
    rows = po["rows"]
    axes = ["referrer_bonus_usd", "referrer_tenure_days", "qualifying_deposit_usd",
            "annual_referral_limit", "company_max_age_years"]
    maps = {a: FD._a3_map(rows, {"axis": a}) for a in axes}
    subs = sorted({s for a in axes for s in maps[a]})
    out = ["Policy constants on record (each from a retrieved document):"]
    for s in subs:
        bits = []
        for a in axes:
            if s in maps[a]:
                bits.append("%s=%s" % (a, maps[a][s][0]))
        if bits:
            out.append("  %s: %s" % (s, ", ".join(bits)))
    return "\n".join(out)


FORM = ["International Checking", "Business Checking", "Checking", "Account", "Balance"]


def _head(s):
    t = str(s).strip()
    for f in FORM:
        if t.endswith(" " + f):
            t = t[: -(len(f) + 1)].strip()
    return t


def a3_table_clean():
    """같은 상품의 여러 표기를 **머리 규칙으로 합친** 표.

    왜 필요한가: raw 표는 우리 A3의 표기 분리를 그대로 안고 있다 — `World Blue Balance`(보너스)와
    `World Blue International Checking`(tenure)이 **다른 행**이라, 모델이 *"Balance 쪽은 tenure
    제약이 없네"* 로 읽을 수 있다. 그러면 0/5 가 능력 측정이 아니라 **표 결함 측정**이 된다.
    이 arm 은 그 교락을 뺀다(축 값이 갈리면 그 축은 `conflict` 로 표시하고 고르지 않는다).
    """
    import t2_factdag as FD
    po = json.load(open(os.path.join(HERE, "a2", "banking_knowledge.specific.json"),
                        encoding="utf-8"))["policy_ontology"]
    rows = po["rows"]
    axes = ["referrer_bonus_usd", "referrer_tenure_days", "qualifying_deposit_usd",
            "annual_referral_limit", "company_max_age_years"]
    maps = {a: FD._a3_map(rows, {"axis": a}) for a in axes}
    merged = {}
    for a in axes:
        for s, (v, _q) in maps[a].items():
            h = _head(s)
            cur = merged.setdefault(h, {})
            if a in cur and cur[a] != v:
                cur[a] = "conflict"
            else:
                cur[a] = v
    out = ["Policy constants on record (one row per product; each value from a retrieved document):"]
    for h in sorted(merged):
        bits = ["%s=%s" % (a, merged[h][a]) for a in axes if a in merged[h]]
        if bits:
            out.append("  %s: %s" % (h, ", ".join(bits)))
    return "\n".join(out)


def ctx_of(tag, task, upto_role="assistant"):
    """그 sim 의 **결정점까지** 실제 대화. 결정점 = 손님이 submit_referral 을 부르기 직전."""
    p = os.path.join(SIMS, tag + ".json.gz")
    d = json.load(gzip.open(p, "rt", encoding="utf-8"))
    sim = next(s for s in d["simulations"] if s["task_id"] == task)
    cut = len(sim["messages"])
    for i, m in enumerate(sim["messages"]):
        if any(tc.get("name") == "submit_referral" for tc in (m.get("tool_calls") or [])):
            cut = i
            break
    lines = []
    for m in sim["messages"][:cut]:
        role = m.get("role")
        c = " ".join(str(m.get("content") or "").split())
        tc = ", ".join(str(t.get("name")) for t in (m.get("tool_calls") or []))
        if tc:
            c = (c + " [called: %s]" % tc).strip()
        if c:
            lines.append("%s: %s" % (role.upper(), c[:900]))
    return "\n".join(lines[-40:])


def score(ans, gold):
    return gold.lower() in str(ans or "").lower()


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "bank_remeas_20260808f"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    table = a3_table()
    clean = a3_table_clean()
    print("A3 표 raw %d줄 · clean %d줄" % (len(table.splitlines()), len(clean.splitlines())))
    print(clean)
    res = collections.defaultdict(list)
    for task, gold in GOLD.items():
        A = table + "\n\n" + FACTS[task] + "\n\n" + QUESTION
        Ac = clean + "\n\n" + FACTS[task] + "\n\n" + QUESTION
        B = ("Here is a customer-service conversation so far.\n\n" + ctx_of(tag, task)
             + "\n\n" + table + "\n\n" + FACTS[task] + "\n\n" + QUESTION)
        for label, prompt in (("A_minimal", A), ("A_clean", Ac), ("B_fullctx", B)):
            for i in range(n):
                try:
                    a = ask(prompt, 0.0 if i == 0 else 0.7)
                except Exception as e:
                    a = "ERR %r" % (e,)
                res[(task, label)].append(a)
    print()
    for (task, label), answers in sorted(res.items()):
        hit = sum(1 for a in answers if score(a, GOLD[task]))
        print("%-9s %-10s gold=%-14s %d/%d  %s"
              % (task, label, GOLD[task], hit, len(answers),
                 [re.sub(r"\s+", " ", a)[:26] for a in answers]))
    print("\n판정 규율([[18]]): A 맞고 B 틀리면 **부하** · A 도 틀리면 **능력/경계**")
    return 0


if __name__ == "__main__":
    sys.exit(main())
