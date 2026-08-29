# -*- coding: utf-8 -*-
r"""x594 — **레버 원장**. 런마다 레버가 무엇을 사고 무엇을 팔았는지 한 장으로 (모델 0 · 무료 · 계수만).

## 왜 (사용자 지시 2026-08-29 · *"레버 원장 상설화 하라"*)

2026-08-29 에 `_over_rows` 가 **39회 발화하고 한 sim 도 구제하지 못한 채** 074 를 통째로 팔았다.
그 사실은 내가 **사후에 찾아 나서서** 나왔다 — 런이 끝난 아침에 자동으로 나오는 표가 없었다.
등대 §1 제1원리가 *"레버는 하나를 사면 하나를 판다"* 인데, **파는 쪽을 재는 상설 계기가 없었다.**

이 원장이 내는 네 칸이 그날 그 결함을 아침에 붉게 만들었을 것들이다:

    ① 발화        몇 번 떴나 (stderr **와** 도구 반환문 양쪽)
    ② 창조한 턴   그 뒤 같은 도구가 다시 불렸나 (= 우리가 만든 재시도)
    ③ 부호        발화한 sim 의 pass ↔ 발화 안 한 sim 의 pass
    ④ 선택성      막은 값이 **gold 와 같았나** (= 정답을 막았나)

## 두 번 물린 함정을 술어에 박는다

  ⒜ **마커가 한 곳에 살지 않는다.** `[T2_SG_ROW_COUNT]` 는 stderr 에 있고 `[components]` 는
     **도구 반환문**에 있다. 오늘 후자를 stderr 에서 세어 `0` 을 얻고 *"미발화"* 로 읽었다
     (`TASK_094.md` §8 이 이미 박제한 함정·[[55]]). ⇒ 양쪽을 **다** 센다.
  ⒝ **채점이 없는 sim 이 있다.** `max_steps`·`context_window_exceeded` 로 끝나면 `reward_basis`
     가 없고 `db_check` 가 null 이라 변이표가 무의미하다(2026-08-29 40 sim 중 5). ⇒ ③은
     **채점된 sim 만** 분모로 쓴다.

## ④ 선택성에 대하여 ([[23]] 경계)

레버 문면에 든 숫자를 그 sim 의 **gold 인자 값**과 대조한다. 이것은 **진단**이지 선택이 아니다 —
어떤 임계도 값도 gold 로 고르지 않는다([[69]] *"gold 일치율은 진단용 보조"*). 이 칸이 오늘
*"정답 총액 3종을 막고 오답은 0회 막았다"* 를 만들었다.

## 쓰기

    py -3 x594_lever_ledger.py <태그> [태그…] [--base <기준선태그>]
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

import t2_forensic as F                                          # noqa: E402

MARK = re.compile(r"\[(T2_[A-Z0-9_]+)\]")
SIMTAG = re.compile(r"\[sim=(task_[0-9]+#s[0-9]+)\]")
NUM = re.compile(r"(?<![\w.])(\d+\.\d+|\d+)(?![\w.])")


def _levers():
    """레버 이름은 **정본에서 읽는다**([[67]] 사본 금지). 실패하면 로그가 알려 준 것만 쓴다."""
    names = set()
    try:
        import t2_levers as L
        for v in (L.CELLS or {}).values():
            if isinstance(v, (list, tuple)) and len(v) > 3 and isinstance(v[3], (list, tuple)):
                names |= {str(x) for x in v[3]}
        for grp in ("META", "HARNESS", "DECLARATIVE", "ARM_ONLY", "DEFAULT_ON", "NOT_LAUNCHED"):
            names |= {str(x) for x in (getattr(L, grp, None) or [])}
    except Exception as ex:
        print("(정본 레버 목록 미적재: %r — 로그에서 관측된 마커만 쓴다)" % (ex,))
    return names


def _returns(sim):
    """도구 **반환문**(role=tool 본문). 마커가 여기 사는 레버가 있다."""
    out = []
    for m in (sim.get("messages") or []):
        if isinstance(m, dict) and m.get("role") == "tool":
            out.append(str(m.get("content") or ""))
    return out


def _gold_numbers(sim, tag):
    """이 sim 의 gold 인자에 든 수치 집합 (진단 전용·[[23]])."""
    got = set()
    d = F.mutation_diff(sim, F.mutating_tools(), tag=tag) or {}
    for k in ("gold", "matched", "missing", "wrongarg"):
        for e in (d.get(k) or ()):
            a = (e or {}).get("args") if isinstance(e, dict) else None
            for n in NUM.findall(json.dumps(a or {}, ensure_ascii=False)):
                got.add(n.rstrip("0").rstrip(".") if "." in n else n)
    return got


def _norm(n):
    return n.rstrip("0").rstrip(".") if "." in n else n


def collect(tag):
    """한 런의 원장 재료. 모든 계수는 sim 단위로 귀속된다."""
    try:
        sims = F.sims(tag)
    except Exception as ex:
        print("(못 읽음) %s : %r" % (tag, ex))
        return None
    log = ""
    try:
        log = F.log_text(tag) or ""
    except Exception:
        pass

    scored, rew, term = {}, {}, {}
    for s in sims:
        st = F.simtag(s)
        ri = s.get("reward_info") or {}
        scored[st] = bool(ri.get("reward_basis"))
        rew[st] = ri.get("reward")
        term[st] = F.term_reason(s)

    fire = collections.defaultdict(lambda: collections.Counter())      # lever -> sim -> n
    blocked_gold = collections.defaultdict(lambda: collections.Counter())
    blocked_other = collections.defaultdict(lambda: collections.Counter())
    goldnum = {F.simtag(s): _gold_numbers(s, tag) for s in sims}

    # ⒜-1 stderr 채널
    for line in log.splitlines():
        st = SIMTAG.search(line)
        if not st:
            continue
        st = st.group(1)
        for lv in set(MARK.findall(line)):
            fire[lv][st] += 1
            gn = goldnum.get(st) or set()
            for n in NUM.findall(line):
                (blocked_gold if _norm(n) in gn else blocked_other)[lv][st] += 1
    # ⒜-2 도구 반환문 채널 — 오늘 여기서 0 을 미발화로 오독했다
    for s in sims:
        st = F.simtag(s)
        for body in _returns(s):
            for lv in set(MARK.findall(body)):
                fire[lv][st] += 1
            if "[components]" in body:
                fire["(반환문)components"][st] += 1

    # ② 창조한 턴 — sim 별 **같은 도구 최다 반복**.
    #   영속 궤적만 세면 안 된다 — 재생성이 막힌 호출을 지운다([[30]]). 우리 도구 반환 마커
    #   (`[T2_SCAFFOLD_GET] <이름> ->`)를 함께 세어 큰 쪽을 쓴다. 그 마커가 곧 "우리가 답을
    #   돌려준 횟수"이고, 재시도 루프는 거기서 부풀어 오른다.
    repeat = {}
    for s in sims:
        c = collections.Counter()
        for _m, tc in F.calls(s):          # 정본은 (message, tool_call) 쌍을 낸다
            a = F.argsof(tc) or {}
            c[F.inner_name(a) or F.nameof(tc) or "?"] += 1
        repeat[F.simtag(s)] = c.most_common(1)[0] if c else ("-", 0)
    RET = re.compile(r"\[sim=(task_[0-9]+#s[0-9]+)\].*?T2_SCAFFOLD_GET\] ([a-z_0-9]+) ->")
    rc = collections.defaultdict(lambda: collections.Counter())
    for st, nm in RET.findall(log):
        rc[st][nm] += 1
    for st, c in rc.items():
        nm, n = c.most_common(1)[0]
        if n > (repeat.get(st) or ("-", 0))[1]:
            repeat[st] = (nm + "(반환)", n)

    return {"tag": tag, "sims": [F.simtag(s) for s in sims], "scored": scored,
            "rew": rew, "term": term, "fire": fire, "repeat": repeat,
            "bg": blocked_gold, "bo": blocked_other}


def report(runs, base=None):
    known = _levers()
    print("=" * 118)
    print("# 레버 원장 — %s" % ", ".join(r["tag"] for r in runs if r))
    print("=" * 118)
    allsims = [(r, st) for r in runs if r for st in r["sims"]]
    nsc = sum(1 for r, st in allsims if r["scored"][st])
    print("sim %d (채점 %d · 미채점 %d — 미채점은 ③ 분모에서 뺀다)"
          % (len(allsims), nsc, len(allsims) - nsc))
    print()
    print("%-26s %6s %6s %8s %8s %9s %s"
          % ("레버", "발화", "sim", "발화시pass", "미발화pass", "gold값차단", "정본등재"))
    print("-" * 118)
    levs = sorted({lv for r in runs if r for lv in r["fire"]})
    rows = []
    for lv in levs:
        n = sum(sum(r["fire"][lv].values()) for r in runs if r)
        hit = {(r["tag"], st) for r in runs if r for st in r["fire"][lv]}
        a = [(r, st) for r, st in allsims if r["scored"][st] and (r["tag"], st) in hit]
        b = [(r, st) for r, st in allsims if r["scored"][st] and (r["tag"], st) not in hit]
        pa = sum(1 for r, st in a if r["rew"][st] == 1.0)
        pb = sum(1 for r, st in b if r["rew"][st] == 1.0)
        bg = sum(sum(r["bg"][lv].values()) for r in runs if r)
        bo = sum(sum(r["bo"][lv].values()) for r in runs if r)
        rows.append((lv, n, len(hit), pa, len(a), pb, len(b), bg, bo))
    def _susp(r):
        lv, n, ns, pa, na, pb, nb, bg, bo = r
        if not nb:                       # 전 sim 발화 = 대조가 없다 = 이 표로는 판정 불가
            return (0, -n)
        harm = (pb / float(nb)) - (pa / float(na) if na else 0.0)
        return (1 + (2 if bg and not pa else 0) + (1 if harm > 0 else 0), harm)
    for r in sorted(rows, key=_susp, reverse=True):
        lv, n, ns, pa, na, pb, nb, bg, bo = r
        flag = "" if (lv in known or lv.startswith("(")) else "  ⚠미등재"
        warn = "  ⛔정답차단" if (bg and not pa) else ""
        cmp_ = "전sim" if not nb else "%d/%d" % (pb, nb)
        print("%-26s %6d %6d %8s %8s %9s %s%s"
              % (lv, n, ns, "%d/%d" % (pa, na), cmp_, "%d/%d" % (bg, bg + bo), flag, warn))
    print()
    print("② 창조한 턴 — sim 별 같은 도구 최다 반복(우리가 만든 재시도의 대리 지표)")
    for r in runs:
        if not r:
            continue
        for st in r["sims"]:
            nm, c = r["repeat"][st]
            if c >= 5:
                print("   %-26s %-20s %-34s ×%-3d reward=%-5s %s%s"
                      % (r["tag"][5:26], st, nm[:34], c, r["rew"][st], r["term"][st],
                         "" if r["scored"][st] else "  (미채점)"))
    print()
    print("⚠읽기 규율: ③은 **상관**이지 인과가 아니다(레버가 뜨는 sim 은 어려운 sim 이다).")
    print("  인과는 같은 sha 의 A/B 로만 말한다([[70]] 판정 의무 ①). 이 표는 **어디를 볼지** 고르는 데 쓴다.")
    print("  `gold값차단` 이 크고 `발화시pass` 가 0 이면 그 레버부터 열어라 — 2026-08-29 `_over_rows` 가 그 모양이었다.")


def main(argv=None):
    args = list(argv or sys.argv[1:])
    base = None
    if "--base" in args:
        i = args.index("--base")
        base = args[i + 1] if i + 1 < len(args) else None
        del args[i:i + 2]
    if not args:
        print(__doc__.strip().splitlines()[0])
        print("사용: py -3 x594_lever_ledger.py <태그> [태그…] [--base <기준선태그>]")
        return 2
    report([collect(t) for t in args], base=base)
    return 0


if __name__ == "__main__":
    sys.exit(main())
