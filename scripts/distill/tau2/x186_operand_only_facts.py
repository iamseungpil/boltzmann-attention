# -*- coding: utf-8 -*-
r"""x186 — **사실은 피연산자만이어야 한다**: 격리 프롬프트가 후보 이름을 나르는가 (유료 0·사용자 지적).

## 왜

사용자 지적: *"표를 보고 사실을 만들 때 오답을 이미 만든 것 아닌가."*

라이브 경로는 그렇지 않다 — `formalize_case_facts` 가 내는 것은 **{자격 축: (값, 인용)}** 이고
(`eligible.criteria` = tenure·tally·deposit·company_age), 표를 요약하지 않는다. 그러나
**프로브 고정 문구는 다르다**:

  task_099 FACTS = "…been a checking customer for about 2 years. …will deposit about $30,000.
                    **The customer already owns Navy Blue, Cobalt Blue and Hunter Green
                    business checking.**"
  task_100 FACTS = "…first checking account was opened 65 days ago. …deposit about $31,000."

099 의 셋째 문장은 **어느 자격 기준의 피연산자도 아니면서 후보 셋을 이름으로 부른다** —
그중 `Hunter Green` 이 곧 정박이자 초안 §6.1 이 7B 실패를 귀속한 *보유-언급 포획*이다.
100 에는 그런 문장이 없다. ⇒ **두 태스크의 격리 조건이 애초에 비대칭이었다**([[55]] 우리 배관).

## 축 (문장 단위·구조 규칙 — 도메인 어휘 안 씀)

  facts_full     현행 고정 문구 그대로
  facts_operand  **후보 이름을 포함한 문장을 뺀다** (판별 = 표의 주어 목록에 걸리는가)
  bare           사실 없음 (x182 대조)

  sort  name_asc · name_desc      후보  all · chk(카드 제외)

## 읽는 법

  · 099 가 `facts_operand` 에서 **좋아지면**  → 우리 사실 문구가 정박을 나르고 있었다. 처방 = 피연산자만.
  · 099 가 **나빠지면**                        → 그 문장이 타입 제약을 붙잡아 주고 있었다(교환).
  · 100 은 두 arm 이 같아야 한다               → 양성 통제(그 태스크엔 뺄 문장이 없다).

실행: python x186_operand_only_facts.py [N]
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


def guided_full(prompt, choices, temp):
    body = json.dumps({"model": MODEL, "temperature": temp, "max_tokens": 12,
                       "guided_choice": list(choices),
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return " ".join((json.load(r)["choices"][0]["message"]["content"] or "").split())


def drop_named_sentences(text, names):
    """후보 **이름을 부르는 문장**만 뺀다. 판별은 표의 주어 목록 대조뿐이다."""
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    return " ".join(s for s in sents if not any(nm in s for nm in names))


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    a2 = load_domain_a2("banking_knowledge")
    spec = next(s for s in a2["ledger_metrics"] if s.get("eligible_text"))
    rows = a2["policy_ontology"]["rows"]
    axes = spec["eligible"]["show_axes"]
    maps = {ax: FD._a3_map(rows, {"axis": ax}) for ax in axes}
    name = lambda l: l.strip().split(":")[0].strip()          # noqa: E731

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

        full = X.FACTS[task]
        oper = drop_named_sentences(full, ALL)
        print("\n" + "=" * 100)
        print("%s  gold=%r" % (task, gold))
        print("  facts_full    (%3d자) %s" % (len(full), full))
        print("  facts_operand (%3d자) %s" % (len(oper), oper or "(전부 제거됨)"))
        print("  → 제거된 문장 있음: %s" % (oper != full))
        print("=" * 100)
        print("  %-13s | %-19s | %-19s | %-19s | %s"
              % ("facts", "asc/all", "desc/all", "asc/chk", "desc/chk"))
        for flabel, ftxt in (("facts_full   ", full), ("facts_operand", oper), ("bare         ", "")):
            cells = []
            for choices in (ALL, CHK):
                for rev in (False, True):
                    order = sorted(body, key=name, reverse=rev)
                    tbl = "\n".join(head[:1] + order + head[1:]).strip()
                    mid = ("\n\n" + ftxt) if ftxt else ""
                    prompt = tbl + mid + "\n\n" + X.QUESTION
                    c = collections.Counter()
                    for i in range(n):
                        try:
                            c[guided_full(prompt, choices, 0.0 if i == 0 else 0.7)] += 1
                        except Exception as e:
                            c["ERR %s" % type(e).__name__] += 1
                    cells.append("%d/%d %-14s" % (c.get(gold, 0), n, c.most_common(1)[0][0][:14]))
                    out.append({"task": task, "facts": flabel.strip(),
                                "choices": "all" if choices is ALL else "chk",
                                "sort": "desc" if rev else "asc",
                                "gold_hit": c.get(gold, 0), "n": n, "dist": dict(c)})
            print("  %-13s | %s | %s | %s | %s"
                  % (flabel, cells[0], cells[1], cells[2], cells[3]))

    json.dump(out, open(os.environ.get("T2_X186_OUT", "x186_out.json"), "w"), indent=1)
    print("\n  099 가 operand 에서 좋아지면 우리 사실 문구가 정박을 나르고 있었다.")
    print("  100 은 두 arm 이 같아야 한다(뺄 문장 없음) = 양성 통제.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
