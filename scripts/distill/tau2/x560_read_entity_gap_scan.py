# -*- coding: utf-8 -*-
r"""x560 — `T2_READ_PER_ENTITY` 의 **발화면과 부호표**를 유료 0 으로 잰다.

## 무엇을 재나 ([[70]] 레버 판정 의무 ②태스크별 부호표)

`t2_dominance.read_entity_gap` 이 **어느 sim 에서 비지 않는가**, 그리고 그 sim 들의 `reward` 는
무엇인가. 이 술어가 켜지면 우리 층은 그 자리에서 선행 read 를 **주체를 지목해** 다시 요구한다 —
요구는 턴을 먹으므로([[70]] 선언 주석의 경고 축자: *"read 강제는 턴을 먹는다"*) 순매수인지
순매도인지는 **발화면 × reward** 로만 갈린다.

## 범위 (훅 §74 세 물음)

⑴ 태그 = 최근 두 세대뿐(`--tags` 기본값) — 같은 hard-0 10 태스크. 코퍼스 전량 순회 안 한다.
⑵ 선행 확인 = `tasks__20260824/TASK_016.md:72` 가 그 read 를 our_layer 로 이미 귀속했다.
   엔티티-인지 충족 판정은 `arg_source_reads`(나열 전용) 어디에도 없다 — grep 확인.
⑶ 한 런으로는 부호표가 안 된다 ⇒ 두 세대.

## 채점 ⛔gold 무참조([[23]])

`reward_info.reward` 만 읽는다([[69]] 채점 단위 = reward). 술어는 도구 호출 **인자만** 본다.

사용: PYTHONIOENCODING=utf-8 py -3 x560_read_entity_gap_scan.py
"""
import argparse
import collections
import os
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
from t2_precedence import declarations, SRC_REQUIRE_BEFORE, SRC_REQUIRES_READS   # noqa: E402


class _TC(object):
    """영속 궤적의 dict 호출을 술어가 보는 형상으로. 값은 **정본 판독기**가 낸 그대로."""

    def __init__(self, d):
        self.name = F.nameof(d)
        self.arguments = F.argsof(d)


def _unwrap(tc):
    """라이브 `_exact_tool_name` 과 같은 규칙 — `call_` 래퍼만 안쪽 이름으로 푼다.
    unlock/give 는 **안쪽 도구를 실행하지 않으므로** 자기 이름을 지킨다."""
    nm = str(getattr(tc, "name", "") or "")
    if nm.startswith("call_"):
        return F.inner_name(getattr(tc, "arguments", None) or {}) or nm
    return nm


class _M(object):
    def __init__(self, m):
        self.tool_calls = [_TC(tc) for tc in (m.get("tool_calls") or ())]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default="bank_t7363_hard0_20260827,bank_t7356_grpB3_20260826,"
                                      "bank_t7356_grpA1_20260826,bank_t7356_grpA2_20260826,"
                                      "bank_t7356_grpA3_20260826,bank_t7356_grpA4_20260826")
    ap.add_argument("--domain", default="banking_knowledge")
    a = ap.parse_args(argv)

    a2 = GI.load_domain_a2(a.domain) or {}
    reads = sorted({r for _dep, rs in declarations(a2, (SRC_REQUIRE_BEFORE, SRC_REQUIRES_READS))
                    for r in rs})
    print("# x560 — 선언된 선행 read %d 종" % len(reads))
    for r in reads:
        print("   ·", r)
    print()

    rows, fired = [], collections.defaultdict(list)
    for tag in [t.strip() for t in a.tags.split(",") if t.strip()]:
        try:
            sims = F.scored(tag)
        except Exception as e:
            print("[skip] %s: %r" % (tag, e))
            continue
        for s in sims:
            ms = [_M(m) for m in (s.get("messages") or ())]
            rw = (s.get("reward_info") or {}).get("reward")
            gaps = {}
            for r in reads:
                g = DOM.read_entity_gap(ms, r, _unwrap)
                if g:
                    gaps[r] = g
            rows.append((tag, F.task_id(s), F.simtag(s), rw, gaps))
            if gaps:
                fired[F.task_id(s)].append((rw, gaps))

    print("## 발화면 — 술어가 비지 않는 sim")
    print("%-26s %-9s %-7s %s" % ("tag", "task", "reward", "gap"))
    print("-" * 104)
    for tag, tid, st, rw, gaps in sorted(rows):
        if gaps:
            g1 = "; ".join("%s ← %s" % (k.split("_by_")[-1] or k,
                                        ", ".join("%s=%s" % (p, v) for v, p in sorted(gp.items())))
                           for k, gp in sorted(gaps.items()))
            print("%-26s %-9s %-7s %s" % (tag[:26], tid, rw, g1[:70]))

    n = len(rows)
    nf = sum(1 for r in rows if r[4])
    print()
    print("## 부호표 ([[70]] ②)")
    print("  채점된 sim %d · 발화 %d (%.0f%%)" % (n, nf, 100.0 * nf / n if n else 0))
    for tid in sorted(fired):
        rws = [rw for rw, _g in fired[tid]]
        pos = sum(1 for x in rws if x and x >= 1.0)
        print("  %-9s 발화 %-3d · reward 1.0 인 것 %d  ⇒ %s"
              % (tid, len(rws), pos,
                 "손실 가능(통과 sim에서 발화)" if pos else "손실 불가(전부 reward 0)"))
    print()
    print("⚠이 표는 **발화면**이지 효과가 아니다. 요구가 실제로 read 를 부르게 하는지는 미측정.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
