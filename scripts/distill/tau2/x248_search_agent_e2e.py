# -*- coding: utf-8 -*-
r"""x248 — **검색 에이전트를 071 실물로 끝에서 끝까지** (격리 · 유료 0 · 로컬 LLM · 배선 전 관문).

## 왜 (사용자 지시 2026-08-11 *"3번 실행"* = 071 배선 · ⛔0)

배선 전에 **체인 전체가 실제 코퍼스에서 정답을 내는지** 본다. 라이브에 꽂고 나서 재면 무엇이
틀렸는지 못 가른다(오늘만 그 실수를 두 번 했다).

  ① LLM   손님의 말 → **문서군 하나**(닫힌 집합 = A3 `doc_index` 키)          ← 해석
  ② 엔진   색인 → 문서 읽기 → **만료 제거** → 축자 재료                        ← 이론
  ③ LLM   남은 것 중 **고르기**(격리 문맥)                                     ← 끝까지 모델

## 071 이 요구하는 것 (권위본 gold)

  business checking → **Sky Blue**        (활성 고지 013 의 1순위)
  business savings  → **Gold Saver Account** (활성 고지 015 의 1순위)
  ⚠만료 고지가 미는 것이 정확히 오답이다: `Lime Green`(014) · `Gold Plus Saver`(016).
    라이브 071 은 실제로 `Lime Green` 으로 갔다.

## 팔 (요청 2개 × n · 재료 구성만 다르다)

  W_ALL      군 문서 + **효력 있는** 유효창 문서 전부      ← 정본 후보
  W_GENERAL  군 문서 + 효력 있는 `_general_` 문서만        ← 좁힌 판
  W_NONE     군 문서만                                    ← 부정 통제(S4 형·0/8 이어야 한다)
  W_EXPIRED  군 문서 + 유효창 문서 **전부(만료 포함)**     ← R4 재현(만료를 안 빼면 무너지는가)

⚠gold 는 채점에만 쓴다([[23]]). ⚠요구문은 태스크 정의 축자다.

실행(리모트·GPU1): T2_PROBE_URL=http://localhost:8141/v1/chat/completions \
                   python x248_search_agent_e2e.py [N]
"""
import collections
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import t2_search as S                                             # noqa: E402
from x216_read_and_offset import chat                             # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DOM = "/home/woori/scratch/tau2-bench/data/tau2/domains/banking_knowledge"
DOCS = os.path.join(DOM, "documents")
NOW = "2025-11-14"
GOLD = {"business_checking": "Sky Blue", "business_savings": "Gold Saver Account"}


def a2():
    return json.load(open(os.path.join(HERE, "a2", "banking_knowledge.gate.json"),
                          encoding="utf-8"))


def requirements():
    p = os.path.join(DOM, "tasks", "task_071.json")
    o = json.load(open(p, encoding="utf-8"))
    o = o[0] if isinstance(o, list) and o else o
    return " ".join(((o.get("user_scenario") or {}).get("instructions") or "").split())


def formalize_group(ask, groups):
    """① 손님의 말 → 문서군 **하나**. 엔진은 답이 **A3 키 집합의 원소인지**만 본다([[22]])."""
    listing = "\n".join("  %s" % g for g in sorted(groups))
    p = ("A customer service agent needs to look up policy documents.\n"
         "Which ONE of these document groups covers what the customer is asking about?\n"
         "Groups:\n%s\n\nCustomer:\n%s\n\nReply with the group name only." % (listing, ask[:2500]))
    try:
        raw = " ".join(str(chat(p, None, 0.0, 24).get("content", "") or "").split())
    except Exception as e:
        return None, "ERR %s" % type(e).__name__
    hit = sorted((g for g in groups if g and g.lower() in raw.lower()), key=len, reverse=True)
    return (hit[0] if hit else None), raw


def decide(material, ask, n, gold):
    """③ 남은 것 중 고르기 — 격리 문맥(요구 + 재료)뿐이고 대화 잔여물은 없다."""
    q = ("%s\n\n%s\n\nWhich ONE account should the agent recommend? "
         "Answer with the account name only." % (ask, material))
    c = collections.Counter()
    for i in range(n):
        try:
            t = chat(q, None, 0.0 if i == 0 else 0.7, 24).get("content", "") or ""
        except Exception as e:
            t = "ERR %s" % type(e).__name__
        c[" ".join(str(t).split())[:44]] += 1
    hit = sum(v for k, v in c.items()
              if re.fullmatch(r"\**%s( Account)?\**\.?" % re.escape(gold), str(k).strip(), re.I))
    return hit, c


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 8
    A = a2()
    groups = list((A.get("policy_ontology") or {}).get("doc_index") or {})
    req = requirements()
    # ★첫 판의 결함 (C417⒠): 두 축 모두에 **시나리오 전문을 통째로** 줬고, 그 안에는 체킹
    #   이야기가 훨씬 많다. 그래서 savings 요청에도 형식화가 `business_checking_accounts` 를
    #   골랐다(그런데 재료가 겹쳐 답은 맞았다 = **틀린 형식화가 우연히 통과**). 결정점은 **요청
    #   하나**에 하나이므로 지금 무엇을 묻는지가 **앞**에 와야 한다 — 라이브에서도 그 턴의 요청이
    #   문맥의 끝이다. 축 문장을 머리에 두고 다시 잰다(문구 신설 0·순서만 바꾼다).
    asks = {"business_checking":
            "The customer is asking which BUSINESS CHECKING account to open.\n\n" + req,
            "business_savings":
            "The customer is asking which BUSINESS SAVINGS account to open.\n\n" + req}
    print("A3 문서군 %d · 유효창 %d행 · 코퍼스 %s"
          % (len(groups), len((A["policy_ontology"].get("doc_windows") or [])), DOCS))
    n_idx, n_tot, ratio = S.index_coverage(A, DOCS)
    print("시야(색인 커버리지): %d/%d = %.0f%%\n" % (n_idx, n_tot, 100 * ratio))
    for axis, ask in asks.items():
        gold = GOLD[axis]
        g, raw = formalize_group(ask, groups)
        print("=" * 96)
        print("%s · gold=%r · ① 형식화 → %s (raw=%r)" % (axis, gold, g or "집합 밖 = 침묵", raw[:60]))
        if not g:
            print("   군을 못 골랐다 — 엔진은 아무것도 하지 않는다(모르면 안 뺀다).")
            continue
        for arm, kw in (("W_ALL", dict(windowed="all")),
                        ("W_GENERAL", dict(windowed="general")),
                        ("W_NONE", dict(windowed="none"))):
            mat, info = S.material_for(A, g, DOCS, NOW, **kw)
            hit, c = decide(mat, ask, n, gold)
            print("  %-10s 재료 %6d자 · 문서 %d(뺀 것 %d) · gold %d/%d   %s"
                  % (arm, len(mat), info["kept"], len(info["dropped"]), hit, n, c.most_common(2)))
        # R4 재현 — 만료를 **안 빼면** 어떻게 되는가(엔진의 유일한 일을 끄는 팔)
        mat_e, info_e = S.material_for(A, g, DOCS, None)          # now=None ⇒ 아무것도 안 뺀다
        hit_e, c_e = decide(mat_e, ask, n, gold)
        print("  %-10s 재료 %6d자 · 문서 %d(뺀 것 0) · gold %d/%d   %s"
              % ("W_EXPIRED", len(mat_e), info_e["kept"], hit_e, n, c_e.most_common(2)))
    print("\n※ 읽는 법 — `W_ALL`/`W_GENERAL` 이 높고 `W_NONE`·`W_EXPIRED` 가 낮아야"
          "\n  **엔진이 하는 유일한 일(만료 제거)** 이 값을 산 것이다. `W_EXPIRED` 가 높으면"
          "\n  유효창 레버는 불필요하고, `W_NONE` 이 높으면 문서 회수 자체가 불필요하다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
