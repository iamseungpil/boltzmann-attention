# -*- coding: utf-8 -*-
r"""x570 — **에이전트 산문 속 통화 수치**의 출처 (유료 0 · 코퍼스 계수).

## 왜 (2026-08-27 · 016 의 마지막 칸)

016 은 도구 인자가 아니라 **문장**에서 진다. t7365 `s1567` msg[42] 축자 —
    *"Currently, your friend has made transactions totaling **$350**. To reach the $500 threshold,
      they need to make additional purchases totaling **$150**."*
직전 도구 응답은 `No records found in 'credit_card_transaction_history'` 였다. `$350` 은 **출처가
하나도 없고**, `$150` 은 그 위의 산수이며, 손님은 그 $150 을 그대로 찍었다(msg[43]·[45]).

우리 검사는 이 자리를 안 본다 — `_provenance_deny` 는 **도구 호출 인자만** 본다. 산문 수치의
자리는 `t2_source.unsourced_claims` 인데 그 독스트링이 약점을 자인한다:
*"한 줄 단위 근접성이지 의미 대조가 아니다."*

## 분류 (그 발화 **이전** 문맥과만 맞댄다)

    doc      도구 출력에 그 수가 있다
    user     손님 발화에만 있다
    own      **앞선 자기 발화에만** 있다 ← 날조가 스스로 번지는 통로($150 이 이것이다)
    absent   어디에도 없다              ← `$350` 이 이것이다

⛔규칙을 제안하지 않는다. 통과 sim 에 `absent`/`own` 이 얼마나 되는지가 규칙의 모양을 정한다
([[70]] — 여기서 오차단하면 멀쩡한 발화를 막는다).

사용: PYTHONIOENCODING=utf-8 py -3 x570_prose_figure_census.py
"""
import argparse
import collections
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import t2_forensic as F                                             # noqa: E402
import x567_numeric_arg_census as X567                              # noqa: E402

CUR = re.compile(r"\$\s?(\d[\d,]*(?:\.\d\d?)?)")


def figs(text):
    return [X567.digits(m) for m in CUR.findall(text or "")]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default="bank_t7365_hard0_20260827,bank_t7364_hard0_20260827,"
                                      "bank_t7363_hard0_20260827,bank_t7356_grpA1_20260826,"
                                      "bank_t7356_grpA2_20260826,bank_t7356_grpA3_20260826,"
                                      "bank_t7356_grpA4_20260826,bank_t7356_grpB3_20260826")
    a = ap.parse_args(argv)

    cls = collections.Counter()
    persim = collections.defaultdict(lambda: [0, 0])
    rows = []
    for tag in [t.strip() for t in a.tags.split(",") if t.strip()]:
        try:
            sims = F.scored(tag)
        except Exception:
            continue
        for s in sims:
            ms = s.get("messages") or []
            tid, rw = F.task_id(s), (s.get("reward_info") or {}).get("reward")
            tool_t, user_t, own_t = [], [], []
            hit = False
            for i, m in enumerate(ms):
                role, c = m.get("role"), " ".join(str(m.get("content") or "").split())
                if role == "assistant" and c:
                    for g in figs(c):
                        d = any(X567.in_text(g, t) for t in tool_t)
                        u = any(X567.in_text(g, t) for t in user_t)
                        o = any(X567.in_text(g, t) for t in own_t)
                        k = "doc" if d else ("user" if u else ("own" if o else "absent"))
                        cls[k] += 1
                        if k in ("absent", "own"):
                            rows.append((tid, F.simtag(s).split("#")[-1], i, k, g, rw))
                            if k == "absent":
                                hit = True
                if role == "tool":
                    tool_t.append(c)
                elif role == "user":
                    user_t.append(c)
                elif role == "assistant":
                    own_t.append(c)
            if hit:
                persim[tid][0] += 1
                if rw and rw >= 1.0:
                    persim[tid][1] += 1

    tot = sum(cls.values()) or 1
    print("# x570 — 에이전트 산문의 통화 수치 %d 건" % tot)
    for k in ("doc", "user", "own", "absent"):
        print("   %-7s %4d (%2.0f%%)" % (k, cls[k], 100.0 * cls[k] / tot))
    print()
    print("## `absent` 전건 (출처 0)")
    for r in sorted([x for x in rows if x[3] == "absent"]):
        print("   %-9s %-9s msg[%3d] $%-10s r=%s" % (r[0], r[1], r[2], r[4], r[5]))
    print()
    print("## `own` — 앞선 자기 발화에만 있는 수 (날조가 번지는 통로) 상위")
    c2 = collections.Counter("%s|$%s" % (r[0], r[4]) for r in rows if r[3] == "own")
    for k, n in c2.most_common(10):
        print("   %-28s %d" % (k, n))
    print()
    print("## 부호표 ([[70]] ② · `absent` 기준)")
    for t in sorted(persim):
        print("   %-9s sim %-3d · reward 1.0 %d ⇒ %s"
              % (t, persim[t][0], persim[t][1], "손실 가능" if persim[t][1] else "손실 불가"))
    return 0




# ─────────────────────────────────────────────────────────────────────────────
# ★2차 계수 (2016 을 덮는 축) — **기록에 관한 주장**의 수치는 레코드 덤프에서 와야 한다.
#   1차 계수는 016 을 못 잡았다: `$350` 도 어느 정책 문서엔 실재해서 `doc` 으로 셌기 때문이다.
#   갈라야 할 것은 *"어느 텍스트에 있나"* 가 아니라 **어느 종류의 텍스트에 있나** 다 —
#   손님이 무엇을 했는지는 **레코드**가 말하고, 정책 산문은 요건을 말한다.
#   실측(8 런·기록 관련 문장의 통화 수치 892): dump 39% · **policy 30%** · user 23% · none 8%.
#   016 msg[42] 축자 *"Currently, your friend has made transactions totaling $350."* = **policy**
#   (직전 도구는 `No records found`). 부호표 = 9 태스크 23 sim · reward 1.0 **0**.
#   ⚠`policy` 30% 전부가 결함은 아니다 — *"요건은 $500"* 같은 문장은 정책이 맞는 출처다.
#     요건 주장과 **실적 주장**을 가르는 것은 의미 판단이라 LLM 몫이다([[22]]) — 엔진은
#     그 분류를 받아 **덤프 대조만** 한다. 이 계수의 어휘 필터는 **계수용**이고 규칙이 아니다.
REC_WORD = re.compile(r"\b(transactions?|balance|spent|spending|history|account)\b", re.I)


def record_claim_sources(ms):
    """기록에 관한 문장의 통화 수치 → (분류, 수, 문장). 계수 전용."""
    out, dump, policy, user = [], [], [], []
    for i, m in enumerate(ms):
        role = m.get("role")
        c = " ".join(str(m.get("content") or "").split())
        if role == "assistant" and c:
            for sent in re.split(r"(?<=[.!?])\s+", c):
                if not REC_WORD.search(sent):
                    continue
                for g in figs(sent):
                    d = any(X567.in_text(g, t) for t in dump)
                    p = any(X567.in_text(g, t) for t in policy)
                    u = any(X567.in_text(g, t) for t in user)
                    k = "dump" if d else ("policy" if p else ("user" if u else "none"))
                    out.append((i, k, g, sent[:120]))
        if role == "tool":
            (dump if "Record ID:" in c else policy).append(c)
        elif role == "user":
            user.append(c)
    return out


if __name__ == "__main__":
    sys.exit(main())
