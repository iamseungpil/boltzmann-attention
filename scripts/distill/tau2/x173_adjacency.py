# -*- coding: utf-8 -*-
r"""x173 — **인접성이 원인인가**: 오답 바로 앞의 이웃을 붙였다 뗐다 한다 (유료 0·교란 통제).

## 왜 (C355 가 남긴 유일한 [?])

C355: 내용·이름·개수·후보목록 전부 고정하고 **행 순서만** 바꾸면 답이 뒤집힌다. 그리고
오답을 내는 배열은 **알파벳순 계열뿐**이다. 알파벳순에서 오답 주변은

    Light Blue(16) · Light Green(17) · **Lime Green(18)** · Navy Blue(19)

로 **첫 글자를 공유하는 행이 바로 앞에 붙어 있고**, 붕괴 토큰이 `'L'`(0.967) 이었다.
이것이 기전인지, 아니면 그저 *"알파벳순을 흔들면 아무거나 고쳐진다"* 인지 **아직 안 갈렸다**.

## 교차 설계 (붙였다 뗐다 + **짝지은 통제**)

    A orig            알파벳순 그대로                         ← 기준: 오답
    B break_L         `Light Blue`+`Light Green` 을 맨 앞으로   ← 인접만 깬다
    C break_ctrl      `Blue`+`Bluest` 를 맨 앞으로             ← **짝지은 통제**: 같은 2행 이동,
                                                             오답과 멀고 첫 글자도 다름
    D insert_L        정답 배열(shuffle_1)에 `Light *` 둘을
                      **오답 바로 앞에** 삽입                  ← 인접만 만든다
    E insert_ctrl     같은 자리에 `Blue`+`Bluest` 삽입          ← **짝지은 통제**

**인접성이 기전이면** B=정답 · C=오답 · D=오답 · E=정답 이어야 한다.
**아무 교란이나 듣는 것이면** B 와 C 가 **같이** 정답이 된다(그러면 인접 서술은 죽는다).
**삽입이 무효면** D 도 정답 그대로다.

⚠이동/삽입은 **행 텍스트를 바꾸지 않는다** — 자리만 옮긴다. `guided_choice` 목록은 C355 통제와
같게 **알파벳순 고정**이다(원인이 목록이 아님은 이미 배제됐다).

실행: py -3 x173_adjacency.py [N]   (8140 = 32B 필요)
"""
import collections
import json
import os
import random
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

URL = os.environ.get("T2_PROBE_URL", "http://localhost:8140/v1/chat/completions")
MODEL = os.environ.get("T2_PROBE_MODEL", "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8")
TASK = "task_099"
GOLD, WRONG = "World Blue", "Lime Green"
NEI = ["Light Blue", "Light Green"]          # 알파벳순에서 오답 바로 앞 두 행(첫 글자 공유)
CTRL = ["Blue", "Bluest"]                    # 짝지은 통제: 2행·오답과 멀고 첫 글자 다름


def guided_full(prompt, choices, temp):
    body = json.dumps({"model": MODEL, "temperature": temp, "max_tokens": 12,
                       "guided_choice": list(choices),
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return " ".join((json.load(r)["choices"][0]["message"]["content"] or "").split())


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    a2 = load_domain_a2("banking_knowledge")
    spec = next(s for s in a2["ledger_metrics"] if s.get("eligible_text"))
    rows = a2["policy_ontology"]["rows"]
    maps = {ax: FD._a3_map(rows, {"axis": ax}) for ax in spec["eligible"]["show_axes"]}
    lines = LG.eligible_text(730, {}, maps, spec,
                             {"qualifying_deposit_usd": 30000}).strip().splitlines()
    head = [l for l in lines if not (l.startswith("  ") and ":" in l)]
    body = [l for l in lines if l.startswith("  ") and ":" in l]
    name = lambda l: l.strip().split(":")[0].strip()          # noqa: E731
    FIXED = [name(l) for l in body]
    MS = Y.msgs_of(os.environ.get("T2_PROBE_TAG", "bank_elig_20260809i"), TASK)
    pre = "Here is a customer-service conversation so far.\n\n" + Y.render(MS) + "\n\n"

    def run(order):
        tbl = "\n".join(head[:1] + order + head[1:]).strip() if head else "\n".join(order)
        base = tbl + "\n\n" + X.FACTS[TASK] + "\n\n" + X.QUESTION
        return collections.Counter(guided_full(pre + base, FIXED, 0.0 if i == 0 else 0.7)
                                   for i in range(n))

    def to_front(lst, names):
        picked = [l for l in lst if name(l) in names]
        return picked + [l for l in lst if name(l) not in names]

    def insert_before(lst, names, target):
        picked = [l for l in lst if name(l) in names]
        rest = [l for l in lst if name(l) not in names]
        out = []
        for l in rest:
            if name(l) == target:
                out.extend(picked)
            out.append(l)
        return out

    sh = list(body)
    random.Random(1).shuffle(sh)             # C355 에서 정답을 낸 배열
    arms = [("A orig       알파벳순", body),
            ("B break_L    Light* 앞으로", to_front(body, NEI)),
            ("C break_ctrl Blue* 앞으로", to_front(body, CTRL)),
            ("D insert_L   셔플+Light* 인접", insert_before(sh, NEI, WRONG)),
            ("E insert_ctrl 셔플+Blue* 인접", insert_before(sh, CTRL, WRONG)),
            ("F shuffle_1  (통제·원본)", sh)]

    print("model=%s · 본문 %d행 · 목록은 알파벳순 고정" % (MODEL, len(body)))
    print("\n%-30s %-24s %-32s %s" % ("arm", "오답 앞 2행", "분포", "gold"))
    for label, order in arms:
        nm = [name(l) for l in order]
        i = nm.index(WRONG)
        c = run(order)
        g = c.get(GOLD, 0)
        print("%-30s %-24s %-32s %d/%d %s"
              % (label, str(nm[max(0, i - 2):i]), c.most_common(2), g, n,
                 "★정답" if g > n // 2 else ""))
    print("\n  인접이 기전이면 B=정답·C=오답·D=오답·E=정답.")
    print("  B 와 C 가 **같이** 정답이면 인접 서술은 죽고 '아무 교란이나 듣는다'가 남는다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
