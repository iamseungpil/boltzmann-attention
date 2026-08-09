# -*- coding: utf-8 -*-
r"""x178 — **정렬돼 있음**과 **자리**를 가른다: 학습 prior 가설의 직접 검정 (유료 0·사용자 발의).

## 왜

C356: 정렬 규칙 8개 중 오름차순 셋(`name_asc`·`bonus_asc`·`limit_asc`)만 오답이다.
사용자 가설 — *"모델이 학습 때 오름차순 정렬된 표를 많이 봐서 생긴 prior 아닌가"*.

⚠소박한 형태는 우리 데이터와 **반대**를 예측한다: `bonus_asc` 는 **마지막이 최고 보너스=정답**
이므로 *"오름차순은 뒤가 좋다"* prior 라면 도와야 하는데 실패했고, 모델은 **끝에서 한두 칸
앞**을 골랐다. 그리고 `name_asc` 는 서열 의미가 없고 `cat_name`(안에서는 알파벳)은 통과한다.
⇒ 논하지 말고 **가른다**.

## 가르는 착안

*"정렬돼 있다"* 와 *"각 행이 어느 자리에 있다"* 를 분리한다:

    오름차순 표에서 **정답과 먼 인접 두 행만 맞바꾼다**
    ⇒ 자리는 거의 그대로인데 **"이 목록은 정렬됨"이 거짓**이 된다

  · 효과가 **사라지면** → 모델이 **정렬성 자체**를 감지한다 ⇒ prior 가설 **지지**
  · 효과가 **남으면**   → 정렬성이 아니라 **자리**다 ⇒ prior 가설 **기각**

## arm

  A asc_pure      오름차순 그대로                         ← 기준: 오답
  B swap_2_3      2·3행 맞바꿈(정답에서 가장 멂)
  C swap_12_13    12·13행 맞바꿈(중간)
  D swap_23_24    23·24행 맞바꿈(정답 바로 앞·정답 자리는 불변)
  E rot1          전체를 한 칸 회전(정렬성 깨짐·자리 전부 1칸 이동)
  F desc_pure     내림차순                                ← 통제: 정답

⚠B·C·D 는 **정답(25번째) 자리를 안 건드린다**. 오직 정렬성만 깬다.

실행: py -3 x178_sortedness.py [N] [URL] [MODEL]
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

TASK = "task_099"
GOLD = "World Blue"


def guided_full(url, model, prompt, choices, temp):
    body = json.dumps({"model": model, "temperature": temp, "max_tokens": 12,
                       "guided_choice": list(choices),
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return " ".join((json.load(r)["choices"][0]["message"]["content"] or "").split())


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    url = sys.argv[2] if len(sys.argv) > 2 else \
        os.environ.get("T2_PROBE_URL", "http://localhost:8140/v1/chat/completions")
    model = sys.argv[3] if len(sys.argv) > 3 else \
        os.environ.get("T2_PROBE_MODEL", "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8")

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

    asc = sorted(body, key=name)

    def swap(lst, i, j):
        out = list(lst)
        out[i], out[j] = out[j], out[i]
        return out

    arms = [("A asc_pure    오름차순", asc),
            ("B swap_2_3    2·3 맞바꿈", swap(asc, 1, 2)),
            ("C swap_12_13  12·13 맞바꿈", swap(asc, 11, 12)),
            ("D swap_23_24  23·24 맞바꿈", swap(asc, 22, 23)),
            ("E rot1        한 칸 회전", asc[1:] + asc[:1]),
            ("F desc_pure   내림차순", sorted(body, key=name, reverse=True))]

    print("model=%s · 본문 %d행 · 정렬성만 깬다(정답 자리 불변: B·C·D)" % (model, len(body)))
    print("\n%-26s %-8s %-10s %-32s %s" % ("arm", "정답자리", "정렬됨?", "분포", "gold"))
    for label, order in arms:
        nm = [name(l) for l in order]
        c = collections.Counter(
            guided_full(url, model, pre + "\n".join(head[:1] + order + head[1:]).strip()
                        + "\n\n" + X.FACTS[TASK] + "\n\n" + X.QUESTION,
                        FIXED, 0.0 if i == 0 else 0.7) for i in range(n))
        g = c.get(GOLD, 0)
        print("%-26s %-8s %-10s %-32s %d/%d %s"
              % (label, "%d/%d" % (nm.index(GOLD) + 1, len(nm)),
                 "예" if nm == sorted(nm) else "아니오", c.most_common(2), g, n,
                 "★정답" if g > n // 2 else ""))
    print("\n  B·C·D 에서 효과가 사라지면 → 모델이 **정렬성 자체**를 감지한다(prior 가설 지지).")
    print("  B·C·D 가 그대로 오답이면 → 정렬성이 아니라 **자리**다(prior 가설 기각).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
