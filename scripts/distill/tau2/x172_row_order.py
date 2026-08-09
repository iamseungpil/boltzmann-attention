# -*- coding: utf-8 -*-
r"""x172 — **행 순서만 바꾼다**: 내용을 하나도 안 건드리고 답이 바뀌는가 (유료 0).

## 왜 (x171 이 강제한 실험)

x171 개명 교차에서 B 와 C 는 **본문 집합이 완전히 같고**(17행 multiset 동일) 카드 행 텍스트도
**바이트 동일**한데 결과가 정반대였다. 유일한 차이는 그 행이 표의 **5번째냐 15번째냐** 였다.

    B: EcoCard 15/17 → Lime Green 10/10 (오답)
    C: EcoCard  5/17 → World Blue 10/10 (정답)

⇒ 행 **순서**가 답을 바꾼다는 가설이 강제된다. 이 프로브는 **내용·이름·개수를 전부 고정**하고
**순서만** 바꾼다. 제거도 개명도 없다 — 교란이 원리적으로 0 이다.

## arm (전부 같은 25행·같은 정박)

  orig      알파벳순(현행 `eligible_text` 가 내는 순서)   ← 기준: Lime Green
  reversed  역순
  gold_1st  정답(World Blue)을 맨 앞으로
  gold_last 정답을 맨 뒤로
  wrong_1st 오답(Lime Green)을 맨 앞으로
  wrong_last 오답을 맨 뒤로
  shuffle_k 고정 시드 셔플 ×3 (시드는 인자로 박아 재현 가능)

## 사전 등록 예측

순서가 인과면 **어떤 재배열은 내용 변화 0 으로 정답을 되살린다.** 그러면 지금까지의
카드·가족·개수 서술이 전부 **자리 효과의 그림자**일 수 있다.
순서가 무관하면 전 arm 이 Lime Green 이고, x171 의 B/C 차이는 다른 데서 온 것이다.

## ⚠우리 코드에 걸리는 함의

`test_eligible_filter` 는 *"정렬은 이름순이다 — 우리가 argmax 하지 않는다"* 를 **중립성 근거**로
못박아 뒀다. 순서가 답을 바꾸면 알파벳순은 **중립이 아니라 우리가 모르고 당기던 레버**다.

실행: py -3 x172_row_order.py [N]   (8140 = 32B 필요)
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
    MS = Y.msgs_of(os.environ.get("T2_PROBE_TAG", "bank_elig_20260809i"), TASK)
    pre = "Here is a customer-service conversation so far.\n\n" + Y.render(MS) + "\n\n"

    FIXED = [name(l) for l in body]        # 알파벳순 후보 목록(원본 순서)

    def run(order, fixed_choices=False):
        """⚠표 순서를 바꾸면 `guided_choice` **후보 목록 순서도 같이** 바뀐다 — 효과가 표가
        아니라 목록에서 올 수 있다. `fixed_choices=True` 면 목록을 **알파벳순으로 고정**하고
        표만 재배열한다 ⇒ 두 축이 갈린다(2026-08-09 통제)."""
        tbl = "\n".join(head[:1] + order + head[1:]).strip() if head else "\n".join(order)
        ch = FIXED if fixed_choices else [name(l) for l in order]
        base = tbl + "\n\n" + X.FACTS[TASK] + "\n\n" + X.QUESTION
        return collections.Counter(guided_full(pre + base, ch, 0.0 if i == 0 else 0.7)
                                   for i in range(n))

    def move(lst, nm, to_front):
        item = [l for l in lst if name(l) == nm]
        rest = [l for l in lst if name(l) != nm]
        return item + rest if to_front else rest + item

    arms = [("orig      알파벳순", body),
            ("reversed  역순", body[::-1]),
            ("gold_1st  정답 맨앞", move(body, GOLD, True)),
            ("gold_last 정답 맨뒤", move(body, GOLD, False)),
            ("wrong_1st 오답 맨앞", move(body, WRONG, True)),
            ("wrong_last 오답 맨뒤", move(body, WRONG, False))]
    for seed in (1, 2, 3):
        sh = list(body)
        random.Random(seed).shuffle(sh)
        arms.append(("shuffle_%d  시드셔플" % seed, sh))

    print("model=%s · 본문 %d행 · 순서만 바꾼다(내용·이름·개수 고정)" % (MODEL, len(body)))
    print("\n%-24s %-8s %-8s %-34s %s" % ("arm", "정답위치", "오답위치", "분포", "gold"))
    for label, order in arms:
        nm = [name(l) for l in order]
        c = run(order)
        g = c.get(GOLD, 0)
        print("%-24s %-8s %-8s %-34s %d/%d %s"
              % (label, "%d/%d" % (nm.index(GOLD) + 1, len(nm)),
                 "%d/%d" % (nm.index(WRONG) + 1, len(nm)),
                 c.most_common(2), g, n, "★정답" if g > n // 2 else ""))
    # ── 통제: 후보 목록은 **알파벳순 고정**, 표만 재배열 ────────────────────
    print("\n=== 통제: guided_choice 목록을 알파벳순으로 고정하고 표만 재배열 ===")
    print("%-24s %-34s %s" % ("arm", "분포", "gold"))
    for label, order in arms:
        c = run(order, fixed_choices=True)
        g = c.get(GOLD, 0)
        print("%-24s %-34s %d/%d %s" % (label, c.most_common(2), g, n,
                                        "★정답" if g > n // 2 else ""))
    print("\n  위 표와 같은 패턴이면 원인은 **표 순서** · 패턴이 사라지면 원인은 **후보 목록 순서**다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
