# -*- coding: utf-8 -*-
r"""x580 — 같은 sha A/B 두 팔을 **부호표로** 맞대는 아침용 보고서 (모델 0 · 무료 · 계수만).

## 왜

밤샘 A/B(`bank_t7371_treat_*` ↔ `bank_t7372_control_*`)가 끝나면 아침에 물을 것은 하나다:
**어느 태스크가 0→1 이고 어느 태스크가 1→0 인가.** 총점 Δ 로 말하면 안 된다(C594·[[70]] ②).
그 표를 사람이 손으로 만들면 또 사본이 생기므로 여기서 한 번에 낸다.

## 무엇을 세나 (판단 0 · 전부 정본 함수)

  ⑴ 태스크별 reward 짝 → **0→1 / 1→0 / 불변** 세 칸
  ⑵ sim 별 `gap`(`mutation_diff` 의 missing+wrongarg+extra+dup) 변화
  ⑶ 레버 자국 팔별 계수 — `[T2_SG_ROW_COUNT]` · `This audit is INCOMPLETE` · `침묵`
  ⑷ ★도구가 건넨 **총액 분포** — 이 A/B 가 직접 사는 것이 *틀린 총액을 안 내보내는 것*이라
     `computed by this tool, is X` 를 팔별로 센다
  ⑸ ★`operand-size` 의 `sub` ↔ `종류 계수` 짝 — 술어가 옳게 섰는지(모자란 자리에서만)

⛔이 프로브는 아무것도 안 고치고 아무 값도 안 만든다. gold 는 `mutation_diff` 안에서만 쓰인다.

사용: PYTHONPATH=. py -3 x580_ab_report.py [--treat TAG] [--control TAG]
"""
import argparse
import collections
import io
import json
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

OUT = os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026")

MARKS = {
    "ROW_COUNT 발화": r"\[T2_SG_ROW_COUNT\]",
    "short 문면 배달": r"This audit is INCOMPLETE",
    "ACTIONREQ 침묵": r"T2_ACTIONREQ\] 침묵",
    "표적=submit_transaction": r"formalized_target=submit_transaction",
    "FORCE_ACTION": r"\[T2_FORCE_ACTION\]",
    "Traceback": r"Traceback",
}
# 2026-08-28 수리 — **어느 자국이 어디에 있는지** 명시한다.
#   핸드오프 §5-5 가 "계수기가 옆을 본다"로 적어 둔 결함이다. `This audit is INCOMPLETE` 와
#   `computed by this tool, is` 는 **도구 반환문**이라 궤적(results.json)에만 있고 로그엔 없다.
#   실측(t7376): "computed by this tool, is" -> 로그 0회 · 궤적 8회.
#   => 로그에서 세면 (4) 총액 패널이 **구조적으로 빈 표**가 된다(값이 없는 게 아니라 안 보는 것).
#   `operand-size` 는 우리 층 인쇄라 로그가 맞다(같은 런 로그 10회) — 그래서 출처를 나눈다.
MARK_SRC = {"short 문면 배달": "traj"}          # 명시 안 된 것은 로그
RE_TOTAL = re.compile(r"computed by this tool, is ([-\d.]+)")   # <- 궤적에서 센다
RE_SIZE = re.compile(r"operand-size (\S+?)\.\w+: sub=(\d+) rows · source=(\d+) rows(?: · (\w+)=(\d+) rows)?")


def arm_data(tag):
    """한 팔의 (태스크별 reward, sim별 gap, 로그 자국, 총액 분포, operand-size 짝)."""
    try:
        sims = F.sims(tag)
    except Exception as e:
        return None, "결과를 못 읽었다: %r" % (e,)
    if not sims:
        return None, "sim 0"
    mut = F.mutating_tools()
    rew, gaps = collections.defaultdict(list), {}
    for s in sims:
        rew[F.task_id(s)].append((s.get("reward_info") or {}).get("reward"))
        d = F.mutation_diff(s, mut, tag=tag) or {}
        gaps[F.simtag(s)] = sum(len(d.get(k) or ()) for k in ("missing", "wrongarg", "extra", "dup"))
    try:
        log = F.log_text(tag) or ""
    except Exception:
        log = ""
    traj = "\n".join(str(m.get("content") or "")
                     for s in sims for m in (s.get("messages") or []))
    marks = {k: len(re.findall(v, traj if MARK_SRC.get(k) == "traj" else log))
             for k, v in MARKS.items()}
    totals = collections.Counter(RE_TOTAL.findall(traj))
    sizes = [{"tool": m[0], "sub": int(m[1]), "source": int(m[2]),
              "kind": m[3] or None, "kind_rows": int(m[4]) if m[4] else None}
             for m in RE_SIZE.findall(log)]
    return {"n": len(sims), "reward": {k: v for k, v in rew.items()}, "gap": gaps,
            "marks": marks, "totals": dict(totals), "sizes": sizes,
            "term": dict(collections.Counter(F.term_reason(s) for s in sims))}, None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--treat", default="bank_t7371_treat_20260828")
    ap.add_argument("--control", default="bank_t7372_control_20260828")
    a = ap.parse_args(argv)

    T, terr = arm_data(a.treat)
    C, cerr = arm_data(a.control)
    print("# x580 — 같은 sha A/B 대조표")
    print("  처치 %s : %s" % (a.treat, ("sim %d" % T["n"]) if T else terr))
    print("  대조 %s : %s" % (a.control, ("sim %d" % C["n"]) if C else cerr))
    if not (T and C):
        print("")
        print("⛔한 팔이 없다 — **대조표를 만들지 않는다**([[25]]). 있는 팔만 아래에 인쇄한다.")
    print("")

    tasks = sorted(set((T or {}).get("reward", {})) | set((C or {}).get("reward", {})))
    up, down, flat = [], [], []
    print("=" * 92)
    print("★부호표 — 태스크별 (총점 Δ 로 말하지 않는다·C594)")
    print("=" * 92)
    print("  %-10s %-18s %-18s %s" % ("태스크", "대조", "처치", "판정"))
    print("  " + "-" * 88)
    for t in tasks:
        cv = sorted((C or {}).get("reward", {}).get(t) or [], key=lambda x: (x is None, x))
        tv = sorted((T or {}).get("reward", {}).get(t) or [], key=lambda x: (x is None, x))
        cp = sum(1 for r in cv if r == 1.0)
        tp = sum(1 for r in tv if r == 1.0)
        verdict = "불변"
        if tp > cp:
            verdict = "★0→1  (+%d)" % (tp - cp); up.append(t)
        elif tp < cp:
            verdict = "⛔1→0  (-%d)" % (cp - tp); down.append(t)
        else:
            flat.append(t)
        print("  %-10s %-18s %-18s %s" % (t, str(cv), str(tv), verdict))
    if T and C:
        print("")
        print("  산 것 %d %s · 판 것 %d %s · 불변 %d"
              % (len(up), up or "", len(down), down or "", len(flat)))
        print("  총계  대조 %d/%d ↔ 처치 %d/%d"
              % (sum(1 for v in C["reward"].values() for r in v if r == 1.0), C["n"],
                 sum(1 for v in T["reward"].values() for r in v if r == 1.0), T["n"]))
        if down:
            print("  ⛔**판 것이 있다** — 이 A/B 는 순증이 아니다. 그 태스크부터 부검하라([[70]]).")

    print("")
    print("=" * 92)
    print("gap 변화 (같은 sim 끼리)")
    print("=" * 92)
    for st in sorted(set((T or {}).get("gap", {})) | set((C or {}).get("gap", {}))):
        cg = (C or {}).get("gap", {}).get(st)
        tg = (T or {}).get("gap", {}).get(st)
        mark = ""
        if isinstance(cg, int) and isinstance(tg, int) and cg != tg:
            mark = "  ★%+d" % (tg - cg)
        print("  %-22s 대조 %-4s → 처치 %-4s%s" % (st, cg, tg, mark))

    print("")
    print("=" * 92)
    print("레버 자국 · 총액 · 전사 행수")
    print("=" * 92)
    print("  %-24s %-10s %s" % ("자국", "대조", "처치"))
    for k in MARKS:
        print("  %-24s %-10s %s" % (k, (C or {}).get("marks", {}).get(k),
                                    (T or {}).get("marks", {}).get(k)))
    print("")
    print("  ★도구가 건넨 총액 분포 (이 A/B 가 직접 사는 것)")
    print("     대조: %s" % ((C or {}).get("totals") or "(없음)"))
    print("     처치: %s" % ((T or {}).get("totals") or "(없음)"))
    print("")
    print("  ★operand-size — 서브가 넘긴 행 ↔ 선언된 종류의 레코드 수")
    for nm, arm in (("대조", C), ("처치", T)):
        rows = (arm or {}).get("sizes") or []
        short = [r for r in rows if r.get("kind_rows") and r["sub"] < r["kind_rows"]]
        print("     %s: 관측 %d · **모자란 자리 %d** %s"
              % (nm, len(rows), len(short),
                 [("%s %d/%d" % (r["kind"], r["sub"], r["kind_rows"])) for r in short[:6]]))
    print("")
    print("  종료사유  대조 %s · 처치 %s"
          % ((C or {}).get("term"), (T or {}).get("term")))

    print("")
    print("판독:")
    print("  ⑴ **판 것(1→0)이 있으면** 순증이 아니다 — 합성을 풀어 어느 노브인지 갈라야 한다.")
    print("  ⑵ 처치의 `모자란 자리` 에서 총액이 사라지고 short 문면이 배달됐으면 술어는 제 일을 했다.")
    print("  ⑶ 그런데도 pass 가 안 움직였으면, 산 것은 *틀린 수를 안 내보낸 것*이고 pass 는 그 위에 있다.")

    dst = os.path.join(OUT, "x580_ab_report_2026_08_28.json")
    with io.open(dst, "w", encoding="utf-8") as f:
        json.dump({"probe": "x580_ab_report", "date": "2026-08-28",
                   "treat_tag": a.treat, "control_tag": a.control,
                   "treat": T, "control": C, "treat_error": terr, "control_error": cerr,
                   "sign": {"up": up, "down": down, "flat": flat},
                   "limits": ["reward 가 유일한 성적이다([[69]]) — gap·자국은 진단 보조.",
                              "두 팔은 **서버가 다르다**(8140 ↔ 8141·argv 는 포트만 다름).",
                              "로그 자국은 문자열 계수일 뿐 인과가 아니다([[08]]).",
                              "자국 출처 분리: 도구 반환문(총액·short)은 궤적, 우리 층 인쇄는 로그."]},
                  f, ensure_ascii=False, indent=1)
    print("")
    print("-> %s" % os.path.normpath(dst))
    return 0


if __name__ == "__main__":
    sys.exit(main())
