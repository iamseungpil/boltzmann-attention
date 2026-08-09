# -*- coding: utf-8 -*-
r"""x192 — **D1c**: 엔진이 불일치를 잡고 **재질의**하면 서브가 스스로 고치는가 (유료 0).

규격서 `ANCHOR_SLOT_SPEC_2026_08_09.md` §5 D1c. 이 프로브가 그 결정의 유일한 미측정 지점이다.

## 왜

D1b 는 **끈다** — 지목은 서브의 LLM 이 통과 집합에서 고른다([[05]] Q2 보존). 그런데 서브도
틀린다: 격리에서 **14B/100 은 0/8**(`Cobalt Blue`) · **32B/100 내림차순 0/8**.
그리고 메인은 그 오답을 못 잡는다 — **근거를 붙일수록 더 못 잡는다**(x185: 틀린 지목 +
올바른 숫자 근거 → 14B 4/8→0/8).

⇒ 유일하게 남은 안전장치가 **엔진 재계산**이다. 단 답을 덮어쓰면 D1b 를 켜는 것이 되므로,
[[52]] 형태로 **질문 트리거**로만 쓴다: 값만 말하고 **이름은 말하지 않는다**.

## 축 (2턴 대화 — 1차 답을 받고 재질의)

  no_reask     1차 답 그대로 (기준선)
  reask_plain  **무내용 재시도** — "Answer again." ★[[57]] 부정 통제 의무. 이게 고치면
               재질의의 효과는 정보가 아니라 **재시도 자체**다
  reask_value  엔진 재계산 결과를 **값으로만**: "네가 고른 것은 <axis>=150 이다. 통과 집합의
               최댓값은 175 다. 다시 답하라." — **이름 없음**(D1b 보존)
  reask_name   gold 이름을 말한다 — **상한 참조**(= D1b ON 이면 얼마나 되는가)

  sort name_asc·name_desc   task 099·100   model 32B·14B   후보 chk(타입 제한)

## 읽는 법

  · `reask_value` 가 `no_reask` 를 이기고 **`reask_plain` 은 못 이기면** → D1c 성립. 처방 확정.
  · `reask_plain` 도 같이 이기면                → 효과는 정보가 아니라 재시도 → D1c 근거 없음([[57]]).
  · `reask_value` ≪ `reask_name` 이면           → 값만으로는 부족 → D1b 를 켜야 하는 압력.

실행: python x192_reask_on_mismatch.py [N]
"""
import collections
import json
import os
import re
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import x149_choice_isolation as X                              # noqa: E402
import t2_factdag as FD                                        # noqa: E402
import t2_ledger as LG                                         # noqa: E402
from gate_interpreter import load_domain_a2                     # noqa: E402

URL = os.environ.get("T2_PROBE_URL", "http://localhost:8140/v1/chat/completions")
MODEL = os.environ.get("T2_PROBE_MODEL", "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8")
CASE = {"task_099": {"days": 730, "deposit": 30000},
        "task_100": {"days": 65, "deposit": 31000}}
ACCOUNT_AXIS = "referrer_tenure_days"


def chat(msgs, choices, temp):
    body = {"model": MODEL, "temperature": temp, "max_tokens": 12,
            "guided_choice": list(choices), "messages": msgs}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return " ".join((json.load(r)["choices"][0]["message"]["content"] or "").split())


def drop_named_sentences(text, names):
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    return " ".join(s for s in sents if not any(nm in s for nm in names))


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    a2 = load_domain_a2("banking_knowledge")
    spec = next(s for s in a2["ledger_metrics"] if s.get("eligible_text"))
    rows = a2["policy_ontology"]["rows"]
    axes = spec["eligible"]["show_axes"]
    maps = {ax: FD._a3_map(rows, {"axis": ax}) for ax in axes}
    # objective_axis 는 라이브에서 `formalize_objective` 가 낸다. 프로브는 그 자리를 고정한다.
    OBJ = next(a for a in axes if "bonus" in a.lower() and "referrer" in a.lower())
    name = lambda l: l.strip().split(":")[0].strip()          # noqa: E731

    def val(s):
        v = (maps.get(OBJ) or {}).get(s)
        try:
            return float(str(v[0]).replace(",", ""))
        except Exception:
            return -1.0

    print("model=%s · n=%d · objective_axis=%s" % (MODEL, n, OBJ))
    out = []
    for task, case in CASE.items():
        gold = X.GOLD[task]
        lines = LG.eligible_text(case["days"], {}, maps, spec,
                                 {"qualifying_deposit_usd": case["deposit"]}).strip().splitlines()
        head = [l for l in lines if not (l.startswith("  ") and ":" in l)]
        b = [l for l in lines if l.startswith("  ") and ":" in l]
        ALL = [name(l) for l in b]
        CHK = [s for s in ALL if s in (maps.get(ACCOUNT_AXIS) or {})]
        facts = drop_named_sentences(X.FACTS[task], ALL)
        best = max(CHK, key=val)
        assert best == gold, "엔진 argmax 와 gold 불일치"

        print("\n" + "=" * 96)
        print("%s  gold=%r(%s=%d) · 후보 %d(chk)" % (task, gold, OBJ, val(gold), len(CHK)))
        print("=" * 96)
        print("  %-10s | %-13s | %-13s | %-13s | %s"
              % ("sort", "no_reask", "reask_plain*", "reask_value", "reask_name"))
        for slabel, rev in (("name_asc ", False), ("name_desc", True)):
            order = sorted(b, key=name, reverse=rev)
            tbl = "\n".join(head[:1] + order + head[1:]).strip()
            p1 = tbl + "\n\n" + facts + "\n\n" + X.QUESTION
            cells, dists = [], []
            for arm in ("no_reask", "reask_plain", "reask_value", "reask_name"):
                c = collections.Counter()
                for i in range(n):
                    t = 0.0 if i == 0 else 0.7
                    a1 = chat([{"role": "user", "content": p1}], CHK, t)
                    if arm == "no_reask":
                        c[a1] += 1
                        continue
                    if arm == "reask_plain":
                        rq = "Answer again."
                    elif arm == "reask_value":
                        # 엔진 재계산 — **값만** 말한다(이름 없음 = D1b 보존)
                        rq = ("A deterministic recheck of the same table: the option you named "
                              "has %s=%d, and the highest %s among the eligible options is %d. "
                              "Answer again." % (OBJ, val(a1), OBJ, val(best)))
                    else:
                        rq = ("A deterministic recheck of the same table selected %s "
                              "(%s=%d). Answer again." % (best, OBJ, val(best)))
                    a2_ = chat([{"role": "user", "content": p1},
                                {"role": "assistant", "content": a1},
                                {"role": "user", "content": rq}], CHK, t)
                    c[a2_] += 1
                cells.append("%d/%d %-8s" % (c.get(gold, 0), n, c.most_common(1)[0][0][:8]))
                dists.append(dict(c))
                out.append({"task": task, "sort": slabel.strip(), "arm": arm,
                            "gold_hit": c.get(gold, 0), "n": n, "dist": dict(c)})
            print("  %-10s | %s | %s | %s | %s" % (slabel, cells[0], cells[1], cells[2], cells[3]))

    json.dump(out, open(os.environ.get("T2_X192_OUT", "x192_out.json"), "w"), indent=1)
    print("\n  * reask_plain = 무내용 재시도 부정 통제([[57]]). 이게 같이 오르면 D1c 근거 없음.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
