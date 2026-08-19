# -*- coding: utf-8 -*-
r"""x400 — x399 재검증: **완료 주장도 표적에 묶는다** (사용자 지시 2026-08-19 밤)

## 왜
x399 는 `claim = 아무 assistant 텍스트나 DONE_RE 매치` 였다 — **표적에 안 묶였다**.
앞 세션은 *의도* 쪽만 조여서 PLAN_ONLY 58 -> 10 으로 무너뜨렸는데, **완료 주장 쪽은 안 조였다**.
같은 계기 결함이 남아 있으면 "허위 완료 신념 42%" 도 같은 방식으로 무너질 수 있다. 대칭으로 조인다.

## 묶기 규칙 (결정론·해석 0)
어떤 assistant 문장이 **이 gold 액션에 관한 것**이려면 둘 중 하나:
    (a) 도구 이름이 축자로 그 문장에 있다
    (b) 그 액션의 **목적어 토큰**(chk_/sav_/dbc_/txn_/cc_/acc_ id 또는 금액)이 그 문장에 있다
둘 다 불가능한 액션(=ops 0 이고 이름이 전 본문에 없음)은 **UNTIEABLE** 로 따로 센다 —
조용히 어느 통에도 넣지 않는다.

사용: py -3 x400_belief_strict.py
"""
import collections, io, json, os, re, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import t2_forensic as F
import x396_saying_vs_doing as C
import x399_closure_forms as N

TAGS, SUF = C.TAGS, C.SUF


def sents(text):
    """문장 단위 (묶기 단위를 문단이 아니라 문장으로 좁힌다)."""
    t = " ".join((text or "").split())
    return [s for s in re.split(r"(?<=[.!?])\s+|\n+", t) if s.strip()]


def about(s, nm, ops):
    return (nm in s) or bool(ops and any(o in s for o in ops))


def main():
    rows = []
    for tag in TAGS:
        for sim in F.scored(tag, SUF):
            if ((sim.get("reward_info") or {}).get("reward") or 0) >= 1.0:
                continue
            gold = C.gold_rows(sim)
            texts = C.assistant_texts(sim)
            allsent = [s for t in texts for s in sents(t)]
            body = " ".join(allsent)
            calls = C.called(sim)
            for g in gold:
                if g["match"]:
                    continue
                nm, ops = g["name"], C.operand_tokens(g["args"])
                named = nm in body
                tieable = bool(ops) or named
                # --- 느슨(x399 방식): 표적 무관 전체 본문
                loose_done = bool(C.DONE_RE.search(body))
                loose_int = bool(C.INTENT_RE.search(body))
                # --- 엄격: 그 액션을 가리키는 문장 안에서만
                mine = [s for s in allsent if about(s, nm, ops)]
                tight_done = next((s for s in mine if C.DONE_RE.search(s)), None)
                tight_int = next((s for s in mine if C.INTENT_RE.search(s)), None)
                rows.append({"task": F.task_id(sim), "trial": sim.get("trial"), "name": nm,
                             "type": g["type"], "ncalls": calls.get(nm, 0),
                             "nops": len(ops), "named": named, "tieable": tieable,
                             "nmine": len(mine),
                             "loose_done": loose_done, "loose_int": loose_int,
                             "tight_done": (tight_done or "")[:150],
                             "tight_int": (tight_int or "")[:150],
                             "x399": N.form_of(sim, g, gold)[0]})

    print("=" * 108)
    print("x400 · 완료 주장을 표적에 묶으면 무엇이 남나 (미매치 gold %d건)" % len(rows))
    print("=" * 108)

    print("\n## ⓐ 묶기 가능성 — 애초에 표적에 묶을 수 있는 액션이 몇 개인가")
    print("  목적어 토큰 있음(ops>0)     %3d" % sum(1 for r in rows if r["nops"]))
    print("  도구 이름 본문 축자 등장     %3d" % sum(1 for r in rows if r["named"]))
    print("  ⇒ 묶을 수 있음(둘 중 하나)   %3d" % sum(1 for r in rows if r["tieable"]))
    print("  ⛔UNTIEABLE(둘 다 없음)      %3d   ← 이 건들은 어떤 신념 주장도 불가" %
          sum(1 for r in rows if not r["tieable"]))

    print("\n## ⓑ 느슨 vs 엄격 — 같은 144건을 두 계기로")
    def bucket(r, tight):
        if r["ncalls"]:
            return "MISCALLED"
        if tight:
            if not r["tieable"]:
                return "UNTIEABLE"
            if r["tight_done"]:
                return "CLAIM_TIED"
            if r["tight_int"]:
                return "INTENT_TIED"
            return "SILENT_ON_TARGET"
        if r["loose_done"]:
            return "CLAIM_LOOSE"
        if r["loose_int"]:
            return "INTENT_LOOSE"
        return "SILENT"
    L = collections.Counter(bucket(r, False) for r in rows)
    T = collections.Counter(bucket(r, True) for r in rows)
    print("  %-20s %-8s | %-20s %s" % ("느슨(x399식)", "건수", "엄격(표적묶음)", "건수"))
    for a, b in zip(["CLAIM_LOOSE", "INTENT_LOOSE", "SILENT", "MISCALLED"],
                    ["CLAIM_TIED", "INTENT_TIED", "SILENT_ON_TARGET", "MISCALLED"]):
        print("  %-20s %-8d | %-20s %d" % (a, L[a], b, T[b]))
    print("  %-20s %-8s | %-20s %d" % ("", "", "UNTIEABLE", T["UNTIEABLE"]))

    print("\n## ⓒ x399 형태 × 엄격 판정 (교차)")
    x = collections.defaultdict(collections.Counter)
    for r in rows:
        x[r["x399"]][bucket(r, True)] += 1
    cols = ["CLAIM_TIED", "INTENT_TIED", "SILENT_ON_TARGET", "UNTIEABLE", "MISCALLED"]
    print("  %-20s %s" % ("x399 형태", " ".join("%-17s" % c for c in cols)))
    for k in sorted(x, key=lambda z: -sum(x[z].values())):
        print("  %-20s %s  (%d)" % (k, " ".join("%-17d" % x[k][c] for c in cols), sum(x[k].values())))

    print("\n## ⓓ 표적에 묶인 **완료 주장** 축자 (전량)")
    n = 0
    for r in rows:
        if bucket(r, True) == "CLAIM_TIED":
            n += 1
            print("  %-9s t%-2s %-34s %s" % (r["task"], r["trial"], r["name"][:34], r["tight_done"]))
    print("  ⇒ %d건" % n)

    print("\n## ⓔ 표적에 묶인 **의도 문장** 축자 (전량)")
    n = 0
    for r in rows:
        if bucket(r, True) == "INTENT_TIED":
            n += 1
            print("  %-9s t%-2s %-34s %s" % (r["task"], r["trial"], r["name"][:34], r["tight_int"]))
    print("  ⇒ %d건" % n)

    out = os.path.normpath(os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026",
                                        "x400_belief_strict.json"))
    io.open(out, "w", encoding="utf-8").write(json.dumps(rows, ensure_ascii=False, indent=1))
    print("\n원자료: %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
