# -*- coding: utf-8 -*-
r"""x171 — **이름을 맞바꾼다**: 효과가 이름을 따라가는가, 자리에 남는가 (유료 0·교란 0).

## 왜 이 설계가 깨끗한가

leave-one-in(x170·C354)이 **필드가 완전히 같은 세 행**을 남겼다:

    EcoCard:             annual_referral_limit=7   → 정답 10/10
    Green Rewards Card:  annual_referral_limit=7   → 정답 10/10
    Silver Rewards Card: annual_referral_limit=7   → **오답 10/10**

정책 내용이 **바이트 동일**한데 하나만 효과를 지탱한다 ⇒ 원인은 행의 내용이 아니라 **이름**이다.
그러면 개명 실험에서 흔히 남는 교란(상수가 달라서 그런 것 아니냐)이 **원리적으로 없다**.
정책 상수를 지어내지 않는다 — **라벨만 맞바꾼다**([[03b]]).

## 교차 설계 (crossover)

    A  Silver Rewards Card  (원본)            → 알려진 결과: 오답
    B  Silver Rewards Card → EcoCard 로 개명   → 효과가 이름을 따라가면 **정답**
    C  EcoCard             (원본)            → 알려진 결과: 정답
    D  EcoCard → Silver Rewards Card 로 개명   → 효과가 이름을 따라가면 **오답**

B 와 D 가 뒤집히면 **이름이 원인**이고, 안 뒤집히면 이름 밖의 무언가(자리·순서 등)다.
A·C 는 재현 통제이고, 개명은 표와 `guided_choice` 목록 **양쪽에** 일관되게 적용한다.

⚠개명 대상은 **정박된 이름도 답도 아니다**(Hunter Green·Lime Green·World Blue 는 안 건드린다) —
그쪽을 건드리면 효과 자체를 지우는 것이라 질문이 사라진다.

실행: py -3 x171_rename_crossover.py [N]   (8140 = 32B 필요)
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
TRIGGER = 26
HOT, COLD = "Silver Rewards Card", "EcoCard"


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
    table = LG.eligible_text(730, {}, maps, spec, {"qualifying_deposit_usd": 30000}).strip()
    lines = table.splitlines()
    cat = {}
    for r in rows:
        s, d = (r or {}).get("subject"), ((r or {}).get("source") or {}).get("doc")
        if s and d and s not in cat:
            cat[s] = "_".join(str(d).split("_")[1:3])

    def name_of(l):
        return l.strip().split(":")[0].strip()

    cards = [name_of(l) for l in lines if l.startswith("  ") and ":" in l
             and "credit" in (cat.get(name_of(l)) or "")]
    MS = Y.msgs_of(os.environ.get("T2_PROBE_TAG", "bank_elig_20260809i"), TASK)
    pre = "Here is a customer-service conversation so far.\n\n" + Y.render(MS) + "\n\n"

    def run(keep_card, rename_to=None):
        """카드는 `keep_card` 한 장만 남기고, 필요하면 그 행의 **이름만** 바꾼다."""
        out = []
        for l in lines:
            if l.startswith("  ") and ":" in l:
                nm = name_of(l)
                if nm in cards and nm != keep_card:
                    continue
                if rename_to and nm == keep_card:
                    l = l.replace(nm, rename_to, 1)
            out.append(l)
        ch = [name_of(l) for l in out if l.startswith("  ") and ":" in l]
        base = "\n".join(out).strip() + "\n\n" + X.FACTS[TASK] + "\n\n" + X.QUESTION
        got = [guided_full(pre + base, ch, 0.0 if i == 0 else 0.7) for i in range(n)]
        row = next(l.strip() for l in out if l.startswith("  ") and
                   name_of(l) == (rename_to or keep_card))
        return collections.Counter(got), row

    arms = [("A  원본 %s" % HOT, HOT, None),
            ("B  %s → %s" % (HOT, COLD), HOT, COLD),
            ("C  원본 %s" % COLD, COLD, None),
            ("D  %s → %s" % (COLD, HOT), COLD, HOT)]
    print("model=%s · 카드 %d장 중 1장만 남긴다" % (MODEL, len(cards)))
    print("\n%-34s %-46s %s" % ("arm", "남은 카드 행(축자)", "분포 (n=%d)" % n))
    for label, keep, ren in arms:
        c, row = run(keep, ren)
        gold = c.get("World Blue", 0)
        print("%-34s %-46s %-30s gold=%d/%d %s"
              % (label, row[:46], c.most_common(2), gold, n,
                 "★정답" if gold > n // 2 else "**오답**"))
    print("\n  B 와 D 가 뒤집히면 **이름이 원인**이다(필드는 바이트 동일하므로 교란 0).")
    print("  안 뒤집히면 이름 밖의 무언가다 — 그때는 자리·순서를 다음 축으로 잡는다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
