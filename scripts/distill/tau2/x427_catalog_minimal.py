# -*- coding: utf-8 -*-
r"""x427 — **카탈로그 선택: 조건과 사실만 남기면 풀리나** (사용자 지시 2026-08-20)

*"엄격하게 필요한 정책과 KB 데이터로만 줄여서도 풀리지 않나"* · *"카탈로그 형태 판단은 조건 넘기고 판단하라"*

## 이 축이 무엇인지 먼저 (x423·게이트 판독으로 확정된 것)
- 손님은 **gold 이름을 말한 적이 없다**(10/10). 오히려 **손님이 말한 이름이 오답인 경우가 5/10**.
  057 t0 은 손님이 *"go ahead … with opening the **Green Fee-Free Account**"* 라고 하는데, 그건
  **에이전트 자신의 잘못된 추천을 손님이 되받은 것**이다 — 되받음을 권위로 쓰면 안 된다([[21]]).
- 그러므로 이 축은 이름-전사(transcription)가 아니라 **제약 → 카탈로그 매칭**이다.
  손님 축자: *"I have some specific things I need … tell me which accounts are right for me"*(055) ·
  *"can you help me pick the right one"*(057).
- 우리 카드 도구 명세는 **최종 선택을 하지 않는다**고 못박고 있다 — *"Choose ONLY among the eligible
  cards; verify … the customer's **soft preferences** before the final pick"*. 즉 랭킹은 모델 몫이다.

## 팔 (같은 물음 · 재료만 다르다)
    N_neg     손님 발화 + **후보 이름 목록만**(사실 0)              ← 부정통제 [[57]]
    M_facts   손님 발화 + **후보별 사실 슬라이스**(축자·최소)        ← 엄격 축소
    E_full    손님 발화 + 배달된 증거 **전량**                       ← x423 E_evidence 재현

## 사전 고정 해석 (결과 보기 전)
- `M_facts ≫ E_full`   ⇒ 잡음이 병목 = **축소가 산다**(부하). 레버는 전달 축.
- `M_facts ≈ N_neg`    ⇒ 사실을 코앞에 줘도 못 고른다 = **능력/경계**. 그 자리만 결정론 후보.
- `M_facts ≈ E_full ≫ N_neg` ⇒ 사실이 결정적이지만 축소는 무의미(둘 다 도달).
- 전 팔이 낮으면 ⇒ **제약이 gold 를 유일하게 정하지 못하는지** 먼저 의심한다(미결정 후보·[[68]] 계보).

★후보 목록·사실 슬라이스는 **배달된 도구-결과에서 형태로만** 뽑는다(뜻 해석 0). gold 는 채점에만([[23]]).

사용: py -3 x427_catalog_minimal.py [--n 3] [--port 8141]
"""
import argparse
import collections
import io
import json
import os
import re
import sys
import threading

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import x395_compliance_iso as X  # noqa: E402
import x397_pvi_channel as P  # noqa: E402
import x423_choice_isolation as I  # noqa: E402
import x426_free_gates as G  # noqa: E402

ARGS = ("account_class", "card_type", "account_type")
FACT_PAD = 900          # 후보당 사실 슬라이스 길이
RE_NAME = re.compile(r"\b(?:[A-Z][A-Za-z\-+]* ){1,3}(?:Account|Card)\b")

SYS = ("You are the argument-filling module of a Rho-Bank support agent. "
       "Reply with ONE JSON object only: {\"value\": \"<exact catalog name>\"}. No prose.")


def catalog_names(docs, gold):
    """배달 본문에 등장한 **같은 꼴의 카탈로그 이름** 전부(형태로만)."""
    tail = "Account" if gold.strip().endswith("Account") else ("Card" if gold.strip().endswith("Card") else None)
    out = set()
    for d in docs:
        for m in RE_NAME.finditer(d):
            nm = " ".join(m.group(0).split())
            if tail is None or nm.endswith(tail):
                out.add(nm)
    out.add(gold)
    return sorted(out)


def fact_slice(name, docs, pad=FACT_PAD):
    for d in docs:
        i = d.find(name)
        if i >= 0:
            return d[max(0, i - 120):i + pad]
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--port", type=int, default=8141)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()

    cs, seen = [], set()
    for c in I.cases(60):
        if c["arg"] not in ARGS:
            continue
        k = (c["task"], c["trial"], c["arg"])
        if k in seen:
            continue
        seen.add(k)
        cs.append(c)
    print("=" * 100)
    print("x427 · 카탈로그 최소재료 격리 · 사례 %d · 팔 3 · n=%d · 포트 %d" % (len(cs), a.n, a.port))
    print("=" * 100)

    jobs = []
    for c in cs:
        docs = G.delivered(c["sim"], c["msg_i"])
        names = catalog_names(docs, c["gold"])
        said = G.customer_said(c["sim"], c["msg_i"])
        head = ("# 손님이 말한 것(축자)\n%s\n\n# 채워야 하는 것\n도구 `%s` 의 인자 `%s`.\n"
                % (said[:6000], c["tool"], c["arg"]))
        cand = "\n# 후보(카탈로그 이름·배달 본문에서 수집)\n%s\n" % ", ".join(names)
        facts = "\n# 후보별 사실(축자)\n" + "\n\n".join(
            "## %s\n%s" % (n, fact_slice(n, docs)) for n in names if fact_slice(n, docs))
        q = "\n# 질문\n어느 것인가. JSON 하나로만: {\"value\": \"<정확한 이름>\"}\n"
        arms = {"N_neg": head + cand + q,
                "M_facts": head + cand + facts + q,
                "E_full": head + I.evidence_block(c["sim"], c["msg_i"]) + q}
        c["_names"] = names
        for an, body in arms.items():
            for k in range(a.n):
                jobs.append({"c": c, "arm": an, "k": k, "body": body,
                             "temp": (0.0 if k == 0 else a.temp)})
    print("사례별 후보 수: %s" % collections.Counter(len(c["_names"]) for c in cs).most_common())
    print("작업 %d건" % len(jobs))

    lock, out = threading.Lock(), []

    def work(_i):
        while True:
            with lock:
                if not jobs:
                    return
                j = jobs.pop(0)
            try:
                d = P.post(a.port, "/v1/chat/completions",
                           {"model": X.MODEL,
                            "messages": [{"role": "system", "content": SYS},
                                         {"role": "user", "content": j["body"]}],
                            "temperature": j["temp"], "max_tokens": 60})
                raw = d["choices"][0]["message"]["content"]
            except Exception as e:
                raw = "ERROR " + str(e)[:160]
            v = I.parse_value(raw)
            c = j["c"]
            rec = {"task": c["task"], "trial": c["trial"], "arg": c["arg"], "arm": j["arm"],
                   "k": j["k"], "gold": c["gold"], "live": c["live"],
                   "got": None if v is None else str(v)[:60],
                   "hit": v is not None and I.norm(v) == I.norm(c["gold"]),
                   "same_as_live": v is not None and I.norm(v) == I.norm(c["live"]),
                   "n_cand": len(c["_names"])}
            with lock:
                out.append(rec)

    ths = [threading.Thread(target=work, args=(i,)) for i in range(a.workers)]
    [t.start() for t in ths]
    [t.join() for t in ths]

    print("\n## 팔별")
    print("%-10s %6s %8s %12s" % ("arm", "n", "HIT", "LIVE-REPEAT"))
    for arm in ("N_neg", "M_facts", "E_full"):
        r = [x for x in out if x["arm"] == arm]
        if not r:
            continue
        n = float(len(r))
        print("%-10s %6d %8.3f %12.3f"
              % (arm, len(r), sum(x["hit"] for x in r) / n, sum(x["same_as_live"] for x in r) / n))

    print("\n## 사례별 (M_facts / E_full)")
    for c in cs:
        m = [x for x in out if x["arm"] == "M_facts" and x["task"] == c["task"]
             and x["trial"] == c["trial"] and x["arg"] == c["arg"]]
        e = [x for x in out if x["arm"] == "E_full" and x["task"] == c["task"]
             and x["trial"] == c["trial"] and x["arg"] == c["arg"]]
        print("   %-9s t%s %-14s 후보 %2d · gold %-24s M %d/%d · E %d/%d · M 답 %s"
              % (c["task"], c["trial"], c["arg"], len(c["_names"]), c["gold"][:24],
                 sum(x["hit"] for x in m), len(m), sum(x["hit"] for x in e), len(e),
                 (m[0]["got"] if m else "-")))

    p = os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026", "x427_catalog_minimal.json")
    with io.open(os.path.abspath(p), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("\n→ %s" % os.path.abspath(p))
    return 0


if __name__ == "__main__":
    sys.exit(main())
