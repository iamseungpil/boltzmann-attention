# -*- coding: utf-8 -*-
"""x169 — 내 대표 사례(Hunter 지목 → Lime)가 **정박**이었나, 아니면 **옳은 추론**이었나.

## 왜 (2026-08-09 저녁·C351 반증 뒤 자기검정)

C351 이 "모델은 지목된 이름을 베끼지 않는다"를 반증했다(Navy 지목 → Navy 7/10). 그러면 내
대표 사례가 왜 안 베꼈는지를 다시 물어야 한다 — 나는 *"가족 최대로 미끄러진다"* 로 읽었지만,
궤적에는 그것 말고 **Hunter 를 피할 옳은 이유**가 들어 있다:

  `[24]` *"you have already referred … **9 Hunter Green** Accounts"* · 표의 Hunter Green
  `annual_referral_limit=10` ⇒ **Hunter 는 올해 9/10 소진**. 남은 자리 1.

즉 Hunter 를 두고 Lime 을 고른 것이 **정박의 미끄러짐이 아니라 한도를 읽은 정상 추론**일 수
있다. 그렇다면 §6.3 의 간판 사례는 정박 현상이 아니라 **모델이 옳게 판단한 사례**이고,
"비복사"라는 관찰 자체가 잘못 귀속된 것이다.

## 설계 (제거만·상수 조작 0)

표는 고정(full eligible). 바꾸는 것은 ⒜ 정박 이름 ⒝ `[24]`(한도 소진 보고)의 유무뿐.

  1 Hunter 정박 · [24] 있음  : 기지 = Lime Green
  2 Hunter 정박 · **[24] 제거**: ★**Hunter 로 바뀌면** = 원래 이유는 한도(정상 추론)
                                 여전히 Lime 이면 = 정박 효과(내 원 서술 지지)
  3 Navy  정박 · [24] 있음   : C351 재현 확인(7/10 복사)
  4 Navy  정박 · [24] 제거   : 통제 — [24] 는 Hunter 전용 정보이므로 변화 없어야 한다

n=10(greedy 1 + 표집 9) — C351 이 7/10 을 보고했으므로 같은 해상도로 맞춘다.

실행: py -3 x169_limit_vs_anchor.py     (8140=32B)
"""
import collections
import json
import os
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import x149_choice_isolation as X                              # noqa: E402
import x150_choice_ablation as Y                               # noqa: E402
import t2_factdag as FD                                        # noqa: E402
import t2_ledger as LG                                         # noqa: E402
from gate_interpreter import load_domain_a2                     # noqa: E402

URL = "http://localhost:8140/v1"
TASK = "task_099"
TAG = os.environ.get("T2_PROBE_TAG", "bank_elig_20260809i")
TRIGGER = 26          # 사과·전환 문장 (C348)
COUNTS = 24           # 한도 소진 보고 ("9 Hunter Green Accounts")
NAMED = "Hunter Green"
N = 10


def model_of(u):
    with urllib.request.urlopen(u + "/models", timeout=30) as r:
        return json.load(r)["data"][0]["id"]


def main():
    a2 = load_domain_a2("banking_knowledge")
    spec = next(s for s in a2["ledger_metrics"] if s.get("eligible_text"))
    rows = a2["policy_ontology"]["rows"]
    maps = {ax: FD._a3_map(rows, {"axis": ax}) for ax in spec["eligible"]["show_axes"]}
    table = LG.eligible_text(730, {}, maps, spec, {"qualifying_deposit_usd": 30000}).strip()
    base = table + "\n\n" + X.FACTS[TASK] + "\n\n" + X.QUESTION
    model = model_of(URL)
    MS = Y.msgs_of(TAG, TASK)
    print("model=%s" % model)
    print("[%d 한도보고] %s" % (COUNTS, " ".join(
        str(MS[COUNTS].get("content") or "").split())[:150]))

    def prompt(anchor, drop_counts):
        ms = [dict(m) for m in MS]
        ms[TRIGGER]["content"] = str(ms[TRIGGER].get("content") or "").replace(NAMED, anchor)
        if drop_counts:                     # ★인덱스 이동을 피하려고 정박 치환을 **먼저** 한다
            ms = ms[:COUNTS] + ms[COUNTS + 1:]
        return ("Here is a customer-service conversation so far.\n\n"
                + Y.render(ms) + "\n\n" + base)

    arms = [("1 Hunter정박 · [24]있음", NAMED, False),
            ("2 Hunter정박 · [24]제거 ★", NAMED, True),
            ("3 Navy정박 · [24]있음", "Navy Blue", False),
            ("4 Navy정박 · [24]제거", "Navy Blue", True)]

    for label, anchor, drop in arms:
        pr = prompt(anchor, drop)
        outs = []
        for i in range(N):
            body = json.dumps({"model": model, "temperature": 0.0 if i == 0 else 0.7,
                               "max_tokens": 24,
                               "messages": [{"role": "user", "content": pr}]}).encode()
            req = urllib.request.Request(URL + "/chat/completions", data=body,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=300) as r:
                outs.append(" ".join(
                    json.load(r)["choices"][0]["message"]["content"].split())[:26])
        tally = collections.Counter(outs).most_common(3)
        copied = sum(1 for o in outs if anchor.lower() in o.lower())
        gold = sum(1 for o in outs if X.score(o, X.GOLD[TASK]))
        print("%-24s 복사 %2d/%d · gold %2d/%d · %s"
              % (label, copied, N, gold, N,
                 " | ".join("%s×%d" % (k, v) for k, v in tally)))
    print("\n판정: arm2 가 Hunter 복사로 바뀌면 → 내 간판 사례(Hunter→Lime)는 **정박이 아니라**"
          " 한도 9/10 을 읽은 **정상 추론**이었다는 뜻 ⇒ §6.3 관찰의 귀속 자체가 잘못이다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
