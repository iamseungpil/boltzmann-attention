# -*- coding: utf-8 -*-
r"""x174 — 한 행을 **모든 자리에** 놓아 본다: 효과의 모양을 가설 없이 본다 (유료 0).

## 왜 (가설 여덟 개가 죽은 뒤)

이 스레드에서 길이·명령·비난·가족·밀어내기·개수·이름·**인접** 여덟 가설이 전부 통제로 죽었다.
아홉 번째를 세우는 대신 **한 변수를 끝까지 쓸어 곡선을 본다**.

확정된 것(C355·x173): 내용·이름·개수·후보목록 고정하고 **행 순서만** 바꾸면 답이 뒤집힌다.
알파벳 기반 배열은 오답 · 셔플 기반은 정답이고, **지역 조작 2행으로는 경계를 못 넘는다**.
유일하게 통한 지역 조작은 **오답 자신을 맨앞/맨뒤로** 보내는 것이었다.

## 설계

나머지를 **알파벳순 그대로 두고** `Lime Green`(오답) 한 행만 1..N 모든 자리에 넣는다.
그리고 같은 쓸기를 `World Blue`(정답)로도 한다 — **짝지은 통제**(C355 에서 정답 이동은
효과가 없었으므로, 곡선이 평평해야 한다).

가설을 세우지 않는다. 나오는 것은 **자리에 대한 함수**이고, 그 모양이 다음 질문을 정한다:
  · 계단(양 끝만 정답) · 단조 · 주기적 · 특정 구간만 — 각각 다른 이야기다.

⚠행 텍스트는 안 바꾼다. `guided_choice` 목록은 알파벳순 **고정**(C355 통제와 동일).

실행: py -3 x174_position_sweep.py [N] [STEP]   (8140 = 32B 필요)
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

URL = os.environ.get("T2_PROBE_URL", "http://localhost:8140/v1/chat/completions")
MODEL = os.environ.get("T2_PROBE_MODEL", "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8")
TASK = "task_099"
GOLD, WRONG = "World Blue", "Lime Green"


def guided_full(prompt, choices, temp):
    body = json.dumps({"model": MODEL, "temperature": temp, "max_tokens": 12,
                       "guided_choice": list(choices),
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return " ".join((json.load(r)["choices"][0]["message"]["content"] or "").split())


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    step = int(sys.argv[2]) if len(sys.argv) > 2 else 2
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

    def place(target, pos):
        item = [l for l in body if name(l) == target]
        rest = [l for l in body if name(l) != target]
        return rest[:pos] + item + rest[pos:]

    N = len(body)
    print("model=%s · 본문 %d행 · 나머지는 알파벳순 고정 · n=%d" % (MODEL, N, n))
    for target, label in ((WRONG, "오답 %s" % WRONG), (GOLD, "정답 %s" % GOLD)):
        home = FIXED.index(target) + 1
        print("\n=== %s 를 자리별로 (알파벳 자리 = %d) ===" % (label, home))
        print("%-6s %-34s %s" % ("자리", "분포", "gold"))
        for pos in list(range(0, N, step)) + [N - 1]:
            c = run(place(target, pos))
            g = c.get(GOLD, 0)
            print("%-6d %-34s %d/%d %s" % (pos + 1, c.most_common(2), g, n,
                                           "★" * (g * 10 // max(n, 1))))
    print("\n  가설을 세우지 않는다 — 곡선의 **모양**이 다음 질문을 정한다.")
    print("  정답 쓸기는 짝지은 통제다(C355: 정답 이동은 효과 없었다 ⇒ 평평해야 한다).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
