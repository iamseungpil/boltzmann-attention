# -*- coding: utf-8 -*-
r"""x182 — **완전 격리에서 정렬 축을 다시 쓴다** + 내림차순 오답의 정체(유료 0·사용자 지시).

## 왜 (x181 이 방향을 뒤집었다)

x181 은 x175 의 `full` 문맥을 재현하면서 **대화 없는 `none`** 을 함께 쟀고, 결과가 반대였다:

| task | ctx | name_asc | name_desc |
|---|---|---|---|
| 099 | full | 0/8 `Lime Green` | **8/8 gold** |
| 099 | none | **8/8 gold** | 0/8 `Business Platinum Rewards Card` |
| 100 | full | 0/8 `Cobalt Blue` | 0/8 `Business Platinum` |
| 100 | none | **8/8 gold** | 0/8 `Business Platinum Rewards Card` |

⇒ ⒜ *"우리 오름차순만 진다"*(C356·C357)는 **대화 접두부가 있을 때만** 참이고,
   격리하면 오름차순이 **유일하게 이기는** 배열이다.
   ⒝ **task_100 도 순서에 반응한다** — C359 의 "복제 실패"는 문맥 조건에서만 참이다.
   ⒞ 격리 오답이 **두 태스크 모두 `Business Platinum Rewards Card`** 다. 그것은
      `referrer_bonus_usd` 의 **전역 최댓값(300)** 이고 **checking 계좌가 아니라 카드**다.
      질문은 *"business checking account"* 를 묻는다 ⇒ **타입 제약 상실** 후보.

## 이 프로브가 가르는 것 (2×2×10)

  ctx      bare     표 + 질문만 (우리 FACTS 문장도 뺀다 — 격리를 끝까지 민다)
           facts    표 + 궤적유래 사실 + 질문 (= x181 `none`)
  choices  all      후보 목록 = 표의 모든 주어 (현행)
           chk      후보 목록에서 **카드를 뺀다** — ⚠표는 **한 글자도 안 바꾼다**(자리 교란 0).
                    카드 판별은 **구조**로 한다: 계좌 축(`referrer_tenure_days`)이 표에 있는
                    행만 남긴다. 도메인 어휘를 안 쓴다([[59]]).
  sort     name/bonus/limit × asc·desc · cat_name · shuffle 1~3   (x175 와 같은 10종)

## 읽는 법

  · `chk` 가 내림차순을 고치면        → 격리 실패 = **타입 제약 상실**(카드 혼입). 처방 = 후보 분리.
  · `chk` 로도 내림차순이 지면        → 방향 효과가 **격리에서도** 실재(카드와 무관).
  · `bare` 와 `facts` 가 갈리면       → 우리 FACTS 문장이 레버다(우리 문구 · [[55]]).
  · 두 태스크가 같이 움직이면         → **공통 기전**. 갈리면 태스크-특유.

실행: python x182_isolated_sort.py [N]
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
import t2_factdag as FD                                        # noqa: E402
import t2_ledger as LG                                         # noqa: E402
from gate_interpreter import load_domain_a2                     # noqa: E402

URL = os.environ.get("T2_PROBE_URL", "http://localhost:8140/v1/chat/completions")
MODEL = os.environ.get("T2_PROBE_MODEL", "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8")
CASE = {"task_099": {"days": 730, "deposit": 30000},
        "task_100": {"days": 65, "deposit": 31000}}
ACCOUNT_AXIS = "referrer_tenure_days"      # 계좌 상품에만 문서화된 축 = 카드/계좌 구조 판별자


def guided_full(prompt, choices, temp):
    body = json.dumps({"model": MODEL, "temperature": temp, "max_tokens": 12,
                       "guided_choice": list(choices),
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return " ".join((json.load(r)["choices"][0]["message"]["content"] or "").split())


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    a2 = load_domain_a2("banking_knowledge")
    spec = next(s for s in a2["ledger_metrics"] if s.get("eligible_text"))
    rows = a2["policy_ontology"]["rows"]
    axes = spec["eligible"]["show_axes"]
    maps = {ax: FD._a3_map(rows, {"axis": ax}) for ax in axes}
    name = lambda l: l.strip().split(":")[0].strip()          # noqa: E731

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

    print("model=%s · n=%d · 계좌 판별축=%s" % (MODEL, n, ACCOUNT_AXIS))
    out = []
    for task, case in CASE.items():
        gold = X.GOLD[task]
        lines = LG.eligible_text(case["days"], {}, maps, spec,
                                 {"qualifying_deposit_usd": case["deposit"]}).strip().splitlines()
        head = [l for l in lines if not (l.startswith("  ") and ":" in l)]
        body = [l for l in lines if l.startswith("  ") and ":" in l]
        ALL = [name(l) for l in body]
        CHK = [s for s in ALL if s in (maps.get(ACCOUNT_AXIS) or {})]

        by = lambda k, rev=False: sorted(body, key=k, reverse=rev)   # noqa: E731
        arms = [("name_asc", by(name)), ("name_desc", by(name, True)),
                ("bonus_asc", by(lambda l: axis_val(name(l), bax))),
                ("bonus_desc", by(lambda l: axis_val(name(l), bax), True)),
                ("limit_asc", by(lambda l: axis_val(name(l), lax))),
                ("limit_desc", by(lambda l: axis_val(name(l), lax), True)),
                ("cat_name", by(lambda l: (cat.get(name(l), "~"), name(l))))]
        for s in (1, 2, 3):
            sh = list(body)
            random.Random(s).shuffle(sh)
            arms.append(("shuffle_%d" % s, sh))

        print("\n" + "=" * 100)
        print("%s  gold=%r  표 %d행 · 후보 all=%d · chk=%d (카드 %d개 제외)"
              % (task, gold, len(body), len(ALL), len(CHK), len(ALL) - len(CHK)))
        print("  제외되는 후보: %s" % ", ".join(s for s in ALL if s not in CHK))
        print("  gold 이 chk 후보에 있는가: %s" % (gold in CHK))
        print("=" * 100)
        hdr = "  %-11s | %-17s | %-17s | %-17s | %s" % (
            "sort", "bare/all", "bare/chk", "facts/all", "facts/chk")
        print(hdr)
        for slabel, order in arms:
            tbl = "\n".join(head[:1] + order + head[1:]).strip() if head else "\n".join(order)
            cells = []
            for ctx in ("bare", "facts"):
                for ch, choices in (("all", ALL), ("chk", CHK)):
                    mid = ("\n\n" + X.FACTS[task]) if ctx == "facts" else ""
                    prompt = tbl + mid + "\n\n" + X.QUESTION
                    c = collections.Counter()
                    for i in range(n):
                        try:
                            c[guided_full(prompt, choices, 0.0 if i == 0 else 0.7)] += 1
                        except Exception as e:
                            c["ERR %s" % type(e).__name__] += 1
                    g = c.get(gold, 0)
                    top = c.most_common(1)[0]
                    cells.append("%d/%d %-11s" % (g, n, top[0][:11]))
                    out.append({"task": task, "sort": slabel, "ctx": ctx, "choices": ch,
                                "gold_hit": g, "n": n, "dist": dict(c)})
            print("  %-11s | %s" % (slabel, " | ".join(cells)))

    json.dump(out, open(os.environ.get("T2_X182_OUT", "x182_out.json"), "w"), indent=1)
    print("\n  chk 가 내림차순을 고치면 격리 실패 = 타입 제약 상실(카드 혼입).")
    print("  chk 로도 지면 방향 효과가 격리에서도 실재. bare↔facts 가 갈리면 우리 FACTS 가 레버.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
