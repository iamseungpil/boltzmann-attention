# -*- coding: utf-8 -*-
r"""x567 — **수치 write 인자**의 값이 어디서 왔는가 (유료 0 · 코퍼스 계수).

## 왜 (2026-08-27 · t7365 가 016 을 인자 하나까지 좁힌 뒤)

t7365 `task_016#s1567` msg[43] 축자 —
    `submit_transaction{user_id: friend_user_5839, credit_card_type: 'Silver Rewards Card',
                        merchant_name: 'Best Buy', category: 'Shopping', **amount: 150**}`
gold 는 같은 네 인자에 **amount '750'** 이고, 그 수를 든 문서는 **아홉 메시지 전에** 왔다
(msg[37] 축자 *"must be approved and spend at least $750 within 60 days"*). 전달 결손이 아니다.

그런데 우리 검사는 그 자리를 **안 본다** — `_provenance_deny` 가
  ⑴ `_hint_hit(k, DEFAULT_ARG_HINTS)` 로 인자 **이름**을 거르는데 `amount` 가 그 목록에 없고
  ⑵ `len(s) < 4` 로 짧은 값을 건너뛴다(`150` 이 그것이다)
⇒ 정책 유래 수치는 **검사 자체를 안 받는다**(핸드오프 2026-08-27 §5 가 남겨 둔 칸).

## 이 프로브가 하는 일 — **세기만** 한다

변이 도구(`mutating_tools`)의 호출마다, **수치 값**을 가진 인자를 그 호출 **이전** 문맥과 맞댄다:

    doc     도구 출력(문서·레코드)에 그 수가 있다        ← 문서 유래
    user    손님 발화에만 있다                          ← 손님이 정한 금액(정당할 수 있다)
    absent  어디에도 없다                               ← 016 의 `150` 이 여기다

⛔규칙을 제안하지 않는다. `user` 가 얼마나 되는지가 규칙의 모양을 정한다 — 손님이 부르는
금액을 막으면 그것이 곧 오차단이다([[70]]).

사용: PYTHONIOENCODING=utf-8 py -3 x567_numeric_arg_census.py
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

import gate_interpreter as GI                                       # noqa: E402
import t2_dominance as DOM                                          # noqa: E402
import t2_forensic as F                                             # noqa: E402
import t2_gate_patch as G                                           # noqa: E402
import x564_arg_producer_census as X564                             # noqa: E402

NUM = re.compile(r"^\$?\d{1,3}(?:,\d{3})*(?:\.\d+)?$|^\$?\d+(?:\.\d+)?$")


def digits(v):
    """비교용 자릿수 문자열 — `$1,750.00` · `1750` · `1750.0` 을 같은 것으로 본다."""
    s = str(v).strip().lstrip("$").replace(",", "")
    if s.endswith(".0"):
        s = s[:-2]
    if s.endswith(".00"):
        s = s[:-3]
    return s


def in_text(num, text):
    """그 수가 텍스트에 **수로서** 있는가 — 자릿수 경계까지 본다(15 가 150 에 걸리지 않게)."""
    return re.search(r"(?<![\d.])" + re.escape(num) + r"(?![\d])", text) is not None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default="bank_t7365_hard0_20260827,bank_t7364_hard0_20260827,"
                                      "bank_t7363_hard0_20260827,bank_t7356_grpA1_20260826,"
                                      "bank_t7356_grpA2_20260826,bank_t7356_grpA3_20260826,"
                                      "bank_t7356_grpA4_20260826,bank_t7356_grpB3_20260826")
    a = ap.parse_args(argv)
    a2 = GI.load_domain_a2("banking_knowledge") or {}
    mut = F.mutating_tools()
    hints = tuple(set(G.DEFAULT_ARG_HINTS) | set(a2.get("identifying_arg_types") or ()))

    cls = collections.Counter()
    by = collections.defaultdict(collections.Counter)
    absent_rows, skipped = [], collections.Counter()
    persim = collections.defaultdict(lambda: [0, 0])
    for tag in [t.strip() for t in a.tags.split(",") if t.strip()]:
        try:
            sims = F.scored(tag)
        except Exception:
            continue
        for s in sims:
            ms = s.get("messages") or []
            tid, rw = F.task_id(s), (s.get("reward_info") or {}).get("reward")
            tool_txt, user_txt = [], []
            hit = False
            for i, m in enumerate(ms):
                role = m.get("role")
                if role == "tool":
                    tool_txt.append((i, " ".join(str(m.get("content") or "").split())))
                elif role == "user":
                    user_txt.append((i, " ".join(str(m.get("content") or "").split())))
                for tc in (m.get("tool_calls") or ()):
                    nm = F.inner_name(F.argsof(tc)) or F.nameof(tc)
                    if not any(F.nameof2(nm) == F.nameof2(w) if hasattr(F, "nameof2") else
                               nm == w or nm.startswith(w) for w in mut):
                        continue
                    for k, v in (DOM._args_dict(X564._TC(tc)) or {}).items():
                        sv = str(v).strip()
                        if not NUM.match(sv):
                            continue
                        num = digits(sv)
                        if len(num) < 2:
                            continue
                        d = any(j < i and in_text(num, t) for j, t in tool_txt)
                        u = any(j < i and in_text(num, t) for j, t in user_txt)
                        c = "doc" if d else ("user" if u else "absent")
                        cls[c] += 1
                        by["%s|%s" % (nm[:26], k)][c] += 1
                        if not G._hint_hit(k, hints) or len(sv) < 4:
                            skipped[c] += 1
                        if c == "absent":
                            hit = True
                            absent_rows.append((tid, F.simtag(s).split("#")[-1], nm[:26], k, sv, rw))
            if hit:
                persim[tid][0] += 1
                if rw and rw >= 1.0:
                    persim[tid][1] += 1

    tot = sum(cls.values()) or 1
    print("# x567 — 변이 도구의 수치 인자 %d 건" % tot)
    for c in ("doc", "user", "absent"):
        print("   %-7s %4d (%2.0f%%)   그중 **현행 검사가 건너뛰는 것** %d"
              % (c, cls[c], 100.0 * cls[c] / tot, skipped[c]))
    print()
    print("## 도구|인자 별 (absent 가 있는 것만)")
    for k in sorted(by, key=lambda x: -by[x]["absent"]):
        if by[k]["absent"]:
            print("   %-40s doc %-4d user %-4d absent %d"
                  % (k, by[k]["doc"], by[k]["user"], by[k]["absent"]))
    print()
    print("## `absent` 전건")
    for r in sorted(absent_rows):
        print("   %-9s %-9s %-26s %-16s %-10s r=%s" % r)
    print()
    print("## 부호표 ([[70]] ②)")
    for t in sorted(persim):
        print("   %-9s sim %-3d · reward 1.0 %d ⇒ %s"
              % (t, persim[t][0], persim[t][1], "손실 가능" if persim[t][1] else "손실 불가"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
