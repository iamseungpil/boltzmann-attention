# -*- coding: utf-8 -*-
r"""x179 — **순수 회전**: 상대 순서를 보존한 채 자리만 민다 (유료 0·교란 원리적 0).

## 왜 (x178 이 강제)

x178: 정렬성을 세 위치에서 깨도 **효과 0**(두 모델) ⇒ 학습 prior 가설 **기각**. 대신 **한 칸
회전**(정답 25→24)만으로 32B 가 정답으로 돌아섰다. 그리고 x176 에서 표 뒤에 종결자를 넣어도
안 고쳐진 것이 여기서 설명된다 — 그 조작들은 정답을 여전히 **마지막 행**으로 남겼다.

⇒ 수렴하는 진술: **마지막 행의 후보가 불리하다.** 텍스트 종결자로는 안 고쳐지고 **다른 행을
그 자리에 넣어야** 고쳐진다.

## 왜 회전인가

x174 의 자리 쓸기는 **한 행을 옮기면 다른 행도 밀리는** 교란이 있었다. 회전은 **상대 순서를
완전히 보존**하고 전체를 통째로 민다 ⇒ 바뀌는 것은 *어느 행이 어느 자리에 오는가* 뿐이고
행 사이의 순서 관계는 하나도 안 바뀐다. 교란이 원리적으로 없다.

    rot_k = asc[k:] + asc[:k]     ⇒ 정답 자리 = 25-k (k=0..)

## 사전 등록 예측

  · *"마지막 행이 불리하다"* 가 맞으면 → **k=0 만 오답**, k≥1 은 정답(32B)
  · 불리한 **구간**이 있으면 → 어떤 k 까지 오답이 이어지고, 그 폭이 모델마다 다르다
    (14B 는 k=1 에서도 오답이었으므로 폭이 더 넓다는 예측)
  · 무관하면 → k 에 관계없이 오답 = 회전이 아니라 다른 것

⚠`guided_choice` 목록은 고정. 행 텍스트 불변. 제거·개명·삽입 없음.

실행: py -3 x179_rotation.py [N] [URL] [MODEL] [KMAX]
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
    url = sys.argv[2] if len(sys.argv) > 2 else "http://localhost:8140/v1/chat/completions"
    model = sys.argv[3] if len(sys.argv) > 3 else "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8"
    kmax = int(sys.argv[4]) if len(sys.argv) > 4 else 8

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
    N = len(asc)

    print("model=%s · %d행 · 순수 회전(상대 순서 보존)" % (model, N))
    print("\n%-5s %-10s %-16s %-32s %s" % ("k", "정답자리", "마지막 행", "분포", "gold"))
    for k in range(0, min(kmax, N)):
        order = asc[k:] + asc[:k]
        nm = [name(l) for l in order]
        c = collections.Counter(
            guided_full(url, model,
                        pre + "\n".join(head[:1] + order + head[1:]).strip()
                        + "\n\n" + X.FACTS[TASK] + "\n\n" + X.QUESTION,
                        FIXED, 0.0 if i == 0 else 0.7) for i in range(n))
        g = c.get(GOLD, 0)
        print("%-5d %-10s %-16s %-32s %d/%d %s"
              % (k, "%d/%d" % (nm.index(GOLD) + 1, N), nm[-1][:16], c.most_common(2), g, n,
                 "★정답" if g > n // 2 else ""))
    print("\n  k=0 만 오답이면 '마지막 행이 불리하다' 확정 · 구간이면 그 폭이 모델별 값이다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
