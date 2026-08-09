# -*- coding: utf-8 -*-
r"""x184 — **오염원 최소대응쌍**: 오름차순 실패를 만드는 것은 *한 문장*인가, 그 문장의 *이름*인가.

## 왜

x181 의 절제가 인과 대상을 하나로 좁혔다(32B·task_099·`name_asc`):

    full 0/8  ·  **-commit 8/8**  ·  -accounts 0/8  ·  none 8/8

`-commit`(꼬리 assistant 턴 제거)이 `none`(대화 전체 제거)과 **결과가 같고**,
`-accounts`(보유계좌 read 제거)는 `full` 과 **결과가 같다**. ⇒ 인과 대상 = **꼬리의
자기-약속 턴** 하나다. 그 턴은 모델 자신이 그 sim 에서 생성한 것이고, 답을 이미 말한다
(099 `Hunter Green` · 100 `Cobalt Blue`).

x183 은 그 정박이 **증거로 안 밀린다**를 세웠다 — gold 보너스를 2위보다 +150 까지 올려도
`full/asc` 는 099 0/8(+150 에서만 3/8) · 100 0/8 로 붙지 않는다.

그래서 남은 질문 둘: ⒜ **그 한 문장만으로 재현되는가**(대화 나머지 없이) ⒝ 효과가 **지목된
이름을 따라가는가**(약속 행위 자체가 아니라). ⒝ 는 초안 §6.1 이 열어 둔 가름 프로브다.

## 축 (한 문장 단위 최소대응쌍)

  iso          표 + 사실 + 질문 (주입 없음)
  +commit      꼬리 자기-약속 턴 **한 개만** 앞에 붙인다 (대화 나머지 전부 없음)
  +commit→gold 같은 문장에서 지목 이름만 **gold** 로 치환
  +commit→alt  같은 문장에서 지목 이름만 **보너스가 가장 비슷한 제3 후보**로 치환
               (구조적 선택 — 도메인 어휘를 안 쓴다·정박과 값이 맞다)
  +commit−name 지목 이름을 `that account` 로 치환 (**약속 행위만 남기고 이름 제거**)
  −tail        대화 전체에서 꼬리만 뺀 것 (같은 런 안의 양성 통제)

  sort  name_asc · name_desc      후보  표의 모든 주어(= `full` 과 같은 조건)

## 읽는 법

  · `+commit` 이 `full` 을 재현하면      → **오염은 한 문장으로 충분**. [S]
  · `+commit→alt` 의 오답이 그 이름으로 따라가면 → 효과는 **지목된 이름**(복사)이다.
  · `+commit−name` 도 지면                → 이름이 아니라 **약속 행위/문장 형식**이다.
  · `+commit→gold` 가 8/8 이면            → 정박은 방향 무관하게 **답을 결정**한다(옳든 그르든).

실행: python x184_commit_minimal_pair.py [N]
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
TAG = os.environ.get("T2_PROBE_TAG", "bank_elig_20260809i")
CASE = {"task_099": {"days": 730, "deposit": 30000},
        "task_100": {"days": 65, "deposit": 31000}}
ACCOUNT_AXIS = "referrer_tenure_days"
LEAD = "Here is a customer-service conversation so far.\n\n"


def guided_full(prompt, choices, temp):
    body = json.dumps({"model": MODEL, "temperature": temp, "max_tokens": 12,
                       "guided_choice": list(choices),
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return " ".join((json.load(r)["choices"][0]["message"]["content"] or "").split())


def last_named(text, choices):
    """문맥에서만 정하는 정박 이름 = 가장 늦게 끝나는 후보(같은 자리면 더 긴 쪽). gold 안 봄."""
    best, pos, ln = None, -1, -1
    for c in choices:
        p = text.rfind(c)
        if p < 0:
            continue
        if p + len(c) > pos or (p + len(c) == pos and len(c) > ln):
            best, pos, ln = c, p + len(c), len(c)
    return best


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    a2 = load_domain_a2("banking_knowledge")
    spec = next(s for s in a2["ledger_metrics"] if s.get("eligible_text"))
    rows = a2["policy_ontology"]["rows"]
    axes = spec["eligible"]["show_axes"]
    maps = {ax: FD._a3_map(rows, {"axis": ax}) for ax in axes}
    bax = next(a for a in axes if "bonus" in a.lower() and "referrer" in a.lower())
    name = lambda l: l.strip().split(":")[0].strip()          # noqa: E731

    def bval(nm):
        v = (maps.get(bax) or {}).get(nm)
        try:
            return float(str(v[0]).replace(",", ""))
        except Exception:
            return -1.0

    print("model=%s · n=%d" % (MODEL, n))
    out = []
    for task, case in CASE.items():
        gold = X.GOLD[task]
        lines = LG.eligible_text(case["days"], {}, maps, spec,
                                 {"qualifying_deposit_usd": case["deposit"]}).strip().splitlines()
        head = [l for l in lines if not (l.startswith("  ") and ":" in l)]
        body = [l for l in lines if l.startswith("  ") and ":" in l]
        ALL = [name(l) for l in body]
        CHK = [s for s in ALL if s in (maps.get(ACCOUNT_AXIS) or {})]

        ms = Y.msgs_of(TAG, task)
        tail = [m for m in ms if m.get("role") == "assistant"][-1]
        tail_txt = "ASSISTANT: " + " ".join(str(tail.get("content") or "").split())
        rest_txt = Y.render([m for m in ms if m is not tail])
        anchor = last_named(tail_txt, ALL)
        alt = min((s for s in CHK if s not in (gold, anchor)),
                  key=lambda s: (abs(bval(s) - bval(anchor)), s))

        print("\n" + "=" * 98)
        print("%s  gold=%r(%d) · 정박=%r(%d) · alt=%r(%d)"
              % (task, gold, bval(gold), anchor, bval(anchor), alt, bval(alt)))
        print("  꼬리 문장 %d자 · 대화 나머지 %d자" % (len(tail_txt), len(rest_txt)))
        print("=" * 98)
        arms = [("iso         ", ""),
                ("+commit     ", LEAD + tail_txt + "\n\n"),
                ("+commit→gold", LEAD + tail_txt.replace(anchor, gold) + "\n\n"),
                ("+commit→alt ", LEAD + tail_txt.replace(anchor, alt) + "\n\n"),
                ("+commit−name", LEAD + tail_txt.replace(anchor, "that account") + "\n\n"),
                ("−tail       ", LEAD + rest_txt + "\n\n")]
        print("  %-12s | %-22s | %s" % ("arm", "name_asc", "name_desc"))
        for alabel, pre in arms:
            cells = []
            for rev in (False, True):
                order = sorted(body, key=name, reverse=rev)
                tbl = "\n".join(head[:1] + order + head[1:]).strip()
                prompt = pre + tbl + "\n\n" + X.FACTS[task] + "\n\n" + X.QUESTION
                c = collections.Counter()
                for i in range(n):
                    try:
                        c[guided_full(prompt, ALL, 0.0 if i == 0 else 0.7)] += 1
                    except Exception as e:
                        c["ERR %s" % type(e).__name__] += 1
                cells.append("%d/%d %-17s" % (c.get(gold, 0), n, c.most_common(1)[0][0][:17]))
                out.append({"task": task, "arm": alabel.strip(),
                            "sort": "desc" if rev else "asc",
                            "gold_hit": c.get(gold, 0), "n": n, "dist": dict(c)})
            print("  %-12s | %s | %s" % (alabel, cells[0], cells[1]))

    json.dump(out, open(os.environ.get("T2_X184_OUT", "x184_out.json"), "w"), indent=1)
    print("\n  +commit 이 full 을 재현하면 오염은 한 문장으로 충분(S).")
    print("  →alt 로 오답이 따라가면 이름 복사 · −name 도 지면 약속 행위/형식.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
