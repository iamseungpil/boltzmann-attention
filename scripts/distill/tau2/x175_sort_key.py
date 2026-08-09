# -*- coding: utf-8 -*-
r"""x175 — **정렬 규칙을 축으로 쓴다**: 어떤 정렬이 어떤 답을 부르는가 (유료 0·사용자 발의).

## 왜 (일반화 가능한 형태로 옮기기)

C355 는 *"순서를 바꾸면 답이 뒤집힌다"* 를 세웠지만 비교한 것은 **이름순 대 무작위**뿐이다.
그래서 *"정렬 규칙 일반"* 에 대해서는 아직 아무 말도 못 한다. 그리고 사용자가 제안한 형태
(*"같은 값이 연속하면 편향"*)는 x173 D 가 이미 한 번 부딪혔다 — 오답 바로 앞에 같은 이웃을
붙여도 정답이었다(지역 연속은 무효·효과는 전역).

⇒ **정렬 키 자체를 축으로** 쓸어야 일반화가 선다. 이 프로브는 같은 25행을 여러 규칙으로
정렬하고 답을 본다. 행 텍스트는 안 바꾼다. `guided_choice` 목록은 알파벳순 **고정**.

## 축

  name_asc / name_desc      이름순 — 알파벳이 특별한가, 아무 결정론적 정렬이나 그런가
  bonus_asc / bonus_desc    **과제 기준(보너스)** 정렬 — 값 정렬이 편향을 만드는가
  limit_asc / limit_desc    과제와 **무관한 값**(연간 한도) 정렬 — 값이면 아무 값이나 그런가
  cat_name                  범주 묶음 → 이름순 (계열끼리 모으기)
  shuffle_1/2/3             기준선(무작위)

## 읽는 법 (가설이 아니라 판정 규칙)

  · 규칙마다 답이 갈리면      → *"정렬은 자유 파라미터"* 가 **강한 형태**로 선다
  · 무작위만 정답이면         → **결정론적 정렬 자체**가 위험하다는 진술
  · `bonus_desc` 가 정답이면  → 실무 처방 즉시 도출(**과제 기준으로 정렬하라**)
  · 전부 같으면              → C355 의 효과는 이 표에 특유하고 일반화 못 한다

⚠단일 태스크·단일 정박·단일 모델이다. 여기서 나오는 것은 **존재 증명과 방향**이지 법칙이 아니다.

실행: py -3 x175_sort_key.py [N]   (8140 = 32B 필요)
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
TASK = os.environ.get("T2_TASK", "task_099")
GOLD = getattr(X, "GOLD", {}).get(TASK, "World Blue")
WRONG = "Lime Green"


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
    # 손님 값은 태스크마다 다르다(x155 CASE 실측) — 표 자체가 달라지므로 함께 매개변수화한다.
    CASE = {"task_099": {"days": 730, "deposit": 30000},
            "task_100": {"days": 65, "deposit": 31000}}[TASK]
    lines = LG.eligible_text(CASE["days"], {}, maps, spec,
                             {"qualifying_deposit_usd": CASE["deposit"]}).strip().splitlines()
    head = [l for l in lines if not (l.startswith("  ") and ":" in l)]
    body = [l for l in lines if l.startswith("  ") and ":" in l]
    name = lambda l: l.strip().split(":")[0].strip()          # noqa: E731
    FIXED = [name(l) for l in body]

    # 정렬 키는 **A3 값**에서 읽는다 (프로브가 상수를 발명하지 않는다)
    def axis_val(nm, ax):
        v = (maps.get(ax) or {}).get(nm)
        try:
            return float(str(v[0]).replace(",", ""))
        except Exception:
            return -1.0
    bax = next((a for a in axes if "bonus" in a.lower()), axes[0])
    lax = next((a for a in axes if "limit" in a.lower()), axes[-1])
    cat = {}
    for r in rows:
        s, d = (r or {}).get("subject"), ((r or {}).get("source") or {}).get("doc")
        if s and d and s not in cat:
            cat[s] = "_".join(str(d).split("_")[1:3])

    MS = Y.msgs_of(os.environ.get("T2_PROBE_TAG", "bank_elig_20260809i"), TASK)
    pre = "Here is a customer-service conversation so far.\n\n" + Y.render(MS) + "\n\n"

    def run(order):
        tbl = "\n".join(head[:1] + order + head[1:]).strip() if head else "\n".join(order)
        base = tbl + "\n\n" + X.FACTS[TASK] + "\n\n" + X.QUESTION
        return collections.Counter(guided_full(pre + base, FIXED, 0.0 if i == 0 else 0.7)
                                   for i in range(n))

    by = lambda k, rev=False: sorted(body, key=k, reverse=rev)   # noqa: E731
    arms = [("name_asc   이름 오름", by(name)),
            ("name_desc  이름 내림", by(name, True)),
            ("bonus_asc  보너스 오름", by(lambda l: axis_val(name(l), bax))),
            ("bonus_desc 보너스 내림", by(lambda l: axis_val(name(l), bax), True)),
            ("limit_asc  한도 오름", by(lambda l: axis_val(name(l), lax))),
            ("limit_desc 한도 내림", by(lambda l: axis_val(name(l), lax), True)),
            ("cat_name   범주→이름", by(lambda l: (cat.get(name(l), "~"), name(l))))]
    for s in (1, 2, 3):
        sh = list(body)
        random.Random(s).shuffle(sh)
        arms.append(("shuffle_%d  무작위" % s, sh))

    print("model=%s · task=%s · gold=%r · 본문 %d행 · n=%d" % (MODEL, TASK, GOLD, len(body), n))
    print("\n%-24s %-8s %-32s %s" % ("정렬 규칙", "정답위치", "분포", "gold"))
    for label, order in arms:
        nm = [name(l) for l in order]
        c = run(order)
        g = c.get(GOLD, 0)
        pos = ("%d/%d" % (nm.index(GOLD) + 1, len(nm))) if GOLD in nm else "표에없음"
        print("%-24s %-8s %-32s %d/%d %s"
              % (label, pos, c.most_common(2), g, n, "★정답" if g > n // 2 else ""))
    print("\n  규칙마다 갈리면 '정렬은 자유 파라미터'가 강한 형태로 선다.")
    print("  무작위만 정답이면 '결정론적 정렬 자체가 위험' · bonus_desc 가 정답이면 처방 즉시 도출.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
