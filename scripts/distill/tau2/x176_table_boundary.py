# -*- coding: utf-8 -*-
r"""x176 — **마지막 행이 흡수되는가**: 표 끝에 경계를 주면 순서 효과가 사라지는가 (유료 0).

## 왜 (x175 가 강제한 배관 의심·[[55]])

x175 에서 실패한 세 정렬(`name_asc`·`bonus_asc`·`limit_asc`)은 **정답이 전부 맨 마지막 행**
(25/25)이었고, 통과한 일곱은 정답이 1·2·2·6·3·17·11 이었다. 게다가 오름차순 두 arm 의 답은
`Lime Green` 도 아니고 **끝에서 한두 칸 앞** 행(`Business Platinum`)이다 — 모델이 **맨 끝을 피한다**.

그런데 우리 `eligible_text` 템플릿은 `{eligible}` 로 **끝난다**(`test_eligible_filter` 가
*"통과 집합 뒤에 산문이 붙지 않는다"* 로 못박은 그것). 그래서 조립하면

    ...마지막 표 행
    <빈 줄>
    FACTS...

가 되고, **마지막 행 뒤에 표의 종결자가 없다**. 그 행이 다음 블록에 흡수되면, 지금까지의
"순서 효과"는 모델 현상이 아니라 **우리 조립의 경계 결함**이다.

## arm (실패한 세 정렬 × 경계 유무)

  bare      현행 그대로(종결자 없음)                  ← 알려진 결과: 오답
  rule      표 뒤에 구분선 한 줄
  label     표 뒤에 *"(end of table)"* 한 줄
  blank2    빈 줄 하나 더

경계를 주는 것만으로 정답이 돌아오면 **원인은 우리 조립**이고, 앞선 C352~C355 의 "순서 효과"는
전부 그 결함의 그림자다. 그대로면 순서 효과는 실재한다.

⚠경계 줄은 **표 내용이 아니다** — 후보를 더하지도 빼지도 않는다. `guided_choice` 목록은 고정.

실행: py -3 x176_table_boundary.py [N]   (8140 = 32B 필요)
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
GOLD = "World Blue"
TAILS = [("bare   현행(종결자 없음)", ""),
         ("rule   구분선", "\n---"),
         ("label  표 끝 표시", "\n(end of table)"),
         ("blank2 빈 줄 하나 더", "\n")]


def guided_full(prompt, choices, temp):
    body = json.dumps({"model": MODEL, "temperature": temp, "max_tokens": 12,
                       "guided_choice": list(choices),
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return " ".join((json.load(r)["choices"][0]["message"]["content"] or "").split())


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    a2 = load_domain_a2("banking_knowledge")
    spec = next(s for s in a2["ledger_metrics"] if s.get("eligible_text"))
    rows = a2["policy_ontology"]["rows"]
    axes = spec["eligible"]["show_axes"]
    maps = {ax: FD._a3_map(rows, {"axis": ax}) for ax in axes}
    lines = LG.eligible_text(730, {}, maps, spec,
                             {"qualifying_deposit_usd": 30000}).strip().splitlines()
    head = [l for l in lines if not (l.startswith("  ") and ":" in l)]
    body = [l for l in lines if l.startswith("  ") and ":" in l]
    name = lambda l: l.strip().split(":")[0].strip()          # noqa: E731
    FIXED = [name(l) for l in body]

    def axis_val(nm, ax):
        v = (maps.get(ax) or {}).get(nm)
        try:
            return float(str(v[0]).replace(",", ""))
        except Exception:
            return -1.0
    bax = next((a for a in axes if "bonus" in a.lower()), axes[0])
    lax = next((a for a in axes if "limit" in a.lower()), axes[-1])
    orders = [("name_asc ", sorted(body, key=name)),
              ("bonus_asc", sorted(body, key=lambda l: axis_val(name(l), bax))),
              ("limit_asc", sorted(body, key=lambda l: axis_val(name(l), lax)))]

    MS = Y.msgs_of(os.environ.get("T2_PROBE_TAG", "bank_elig_20260809i"), TASK)
    pre = "Here is a customer-service conversation so far.\n\n" + Y.render(MS) + "\n\n"

    print("머리말 %d줄 · 본문 %d행 · 표 뒤 종결자 유무만 바꾼다" % (len(head), len(body)))
    print("  ⚠현행 조립에서 마지막 행 다음 문자열: %r"
          % (("\n".join(head[:1] + orders[0][1] + head[1:]).strip()[-60:] + "\n\n"
              + X.FACTS[TASK])[:90]))
    print("\n%-12s %-24s %-32s %s" % ("정렬", "종결자", "분포", "gold"))
    for oname, order in orders:
        for tname, tail in TAILS:
            tbl = "\n".join(head[:1] + order + head[1:]).strip() + tail
            base = tbl + "\n\n" + X.FACTS[TASK] + "\n\n" + X.QUESTION
            c = collections.Counter(guided_full(pre + base, FIXED, 0.0 if i == 0 else 0.7)
                                    for i in range(n))
            g = c.get(GOLD, 0)
            print("%-12s %-24s %-32s %d/%d %s"
                  % (oname, tname, c.most_common(2), g, n, "★정답" if g > n // 2 else ""))
    print("\n  경계만 주고 정답이 돌아오면 → 원인은 **우리 조립**이고 C352~C355 는 그 그림자다.")
    print("  그대로면 → 순서 효과는 실재한다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
