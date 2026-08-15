# -*- coding: utf-8 -*-
r"""x329 — **후보 집합을 접으면 지는가**: 커밋 직전까지 손님에게 보인 상품 수 × 통과.

## 왜

003 실물(4 sim): 통과한 **유일한** sim 만 자격 충족 카드 **3장을 다 제시**했고, 진 세 sim 은
*"the Platinum Rewards Card **seems to be the best fit**"* 처럼 **한두 장으로 접었다**.
003 은 손님이 직접 신청하는 태스크라(`requestor: "user"`) 우리가 접는 순간 손님이 가진
**연회비-최소 규칙이 작동할 자리가 사라진다**. 반대편에는 069·055·071 이 있다 —
거기서는 **지워야 할 후보를 안 지운다**. 둘 다 *후보 집합을 다루지 못하는* 한 축이다.

n=4 로는 축을 주장할 수 없다([[57]]·C483). 그래서 **전 태스크에서 기계적으로** 센다.

## 어떻게 (저작 0)

상품 어휘는 **A3 `doc_index` 의 주어 키**에서 온다(빌드 시 파일명에서 유도한 것·65개).
우리가 이름을 짓지 않는다는 것이 요점이다([[59]]).

  · **커밋** = 인자에 `account_class`/`card_type`/`referred_account_type` 이 실린 첫 도구 호출
  · **제시 집합** = 그 커밋 **이전** 어시스턴트 본문에 등장한 **서로 다른 주어 수**
  · 손님-가시 본문만 센다(도구 출력·우리 층 주입은 제외 — 손님이 고를 수 있는 것이 기준)

## 읽는 법 (사전 고정)

  통과 sim 의 제시 수 > 실패 sim 의 제시 수      → "접으면 진다" 지지
  차이 없음                                      → 003 은 우연 · 축 주장 철회
  ⚠**상관이다.** 긴 궤적일수록 많이 언급되므로 **커밋 이전**으로 자르고, 커밋이 없는 sim 은 제외한다.
  ⚠태스크마다 후보 수가 다르므로 **태스크별로도** 병기한다(pooled 만 보면 심슨 역설이 난다).

사용: py x329_offered_set_size.py [tag ...]
"""
import collections
import io
import re
import sys

try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
except Exception:
    pass

import t2_forensic as F                                            # noqa: E402
from t2_gate_patch import load_domain_a2                           # noqa: E402

COMMIT_KEYS = ("account_class", "card_type", "referred_account_type")


def vocabulary():
    """A3 주어 키 → 본문에서 찾을 정규식. 슬러그를 표면형으로 되돌리기만 한다."""
    a2 = load_domain_a2("banking_knowledge")
    di = ((a2.get("policy_ontology") or {}).get("doc_index") or {})
    subs = set()
    for g in di.values():
        for s in g:
            if s != "_general_":
                subs.add(s)
    out = {}
    for s in subs:
        t = re.sub(r"\(.*?\)", " ", s).replace("_", " ").strip()
        if len(t) < 4:
            continue
        out[s] = re.compile(r"\b" + r"\s+".join(re.escape(w) for w in t.split()) + r"\b", re.I)
    return out


def commit_index(sim):
    """상품을 확정하는 첫 호출의 메시지 색인. 없으면 None."""
    for i, m in enumerate(sim.get("messages") or []):
        for tc in (m.get("tool_calls") or []):
            a = F.argsof(tc)
            inner = a.get("arguments")
            blob = str(inner) + str(a)
            if any(k in blob for k in COMMIT_KEYS):
                return i
    return None


def offered(sim, cut, vocab):
    """커밋 이전 **어시스턴트 본문**에 등장한 서로 다른 주어 수."""
    txt = " ".join(str(m.get("content") or "")
                   for m in (sim.get("messages") or [])[:cut]
                   if m.get("role") == "assistant")
    return {s for s, rx in vocab.items() if rx.search(txt)}


def main(tags):
    vocab = vocabulary()
    print("어휘(A3 주어) %d개\n" % len(vocab))
    rows = []
    for tag in tags:
        for s in F.scored(tag):
            c = commit_index(s)
            if c is None:
                continue
            names = offered(s, c, vocab)
            rows.append((F.task_id(s), (s.get("reward_info") or {}).get("reward") == 1.0,
                         len(names), sorted(names)[:6]))
    if not rows:
        print("커밋이 있는 sim 없음"); return
    import statistics as st
    for lab in (True, False):
        sel = [r[2] for r in rows if r[1] == lab]
        if sel:
            print("%s n=%2d · 제시 수 중앙값 %.1f · 평균 %.1f · 분포 %s"
                  % ("PASS" if lab else "FAIL", len(sel), st.median(sel),
                     sum(sel) / float(len(sel)), sorted(sel)))
    print("\n태스크별(같은 태스크 안에서만 비교해야 뜻이 있다):")
    bytask = collections.defaultdict(lambda: ([], []))
    for t, ok, n, _ in rows:
        bytask[t][0 if ok else 1].append(n)
    for t in sorted(bytask):
        p, f = bytask[t]
        if not (p and f):
            continue
        print("   %-10s PASS %s ↔ FAIL %s" % (t, sorted(p), sorted(f)))
    print("\n(둘 다 있는 태스크만 위에 나온다 — 한쪽만 있는 태스크는 대조가 안 된다)")
    print("\n건별:")
    for t, ok, n, names in sorted(rows):
        print("   %-10s %-5s %2d  %s" % (t, "PASS" if ok else "FAIL", n, names))


if __name__ == "__main__":
    main(sys.argv[1:] or ["bank_t7295_a_20260815n", "bank_t7295_b_20260815n"])
