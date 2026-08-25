# -*- coding: utf-8 -*-
"""아침 검토 한 방 — 밤샘런을 **재유도 없이** 판정하는 데 필요한 것만 순서대로 인쇄한다.

사용자 지시 2026-08-25 밤: *"가장 효율적으로 내일 아침에 작업 다시 검토할 수 있는 계획을 세워라."*

    py -3 morning_review.py                 # 기본 = t7356 (대조 t7355·t7354)
    py -3 morning_review.py --tag t7357     # 다른 런

인쇄 순서는 **판정 순서**다([[69]]·[[08]]·[[25]]):
  §1 회수  — 산출물이 tracked 인가. 아니면 그 배치는 아직 판정할 수 없다([[30]]).
  §2 성적  — reward 만. 배치별·태스크별. gold 일치율은 성적이 아니다([[69]]).
  §3 발화  — 어젯밤 배선한 레버가 라이브에서 **말을 했는가**(死배선 조기 발견·[[24]]).
  §4 남은 차이 — 실패한 sim 마다 gold 행 ↔ 궤적 호출을 자연키로 짝지어 **인자 차이만** 인쇄.
                 이것이 어젯밤 085·040 의 축을 가른 바로 그 표다.
  §5 다음 수 — 열려 있는 축과, 그 축을 이미 잰 파일 이름.

⚠이 스크립트는 **아무것도 계산해 만들지 않는다** — 정본 `t2_forensic` 이 내는 값을 옮길 뿐이다.
"""
import argparse
import collections
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import t2_forensic as F   # noqa: E402  (정본 재사용·사본 금지 [[67]])

MARKERS = ["T2_SPEC_AT_WRITE", "T2_RULE_AT_WRITE", "T2_WRITE_ARG_TYPE", "T2_WRITE_ARG_ENUM",
           "T2_WRITE_ARG_FAB", "T2_SPEC_ARG_FACTS", "T2_SG_RECORD_ORDER",
           "T2_ARG_POLICY_AT_WRITE", "T2_SG_PROMPT_V2", "Traceback"]

OPEN_AXES = [
    ("085", "분쟁 커버리지 — gold 분쟁 3건 중 몇 건을 내는가",
     "어젯밤 t7355 스모크는 1건만 냈다(user_stop). t7354 는 4건 냈고 2건이 gold 축자 일치. "
     "nt10 이 이 물음에 답한다. 큐 findings_2026_08_25_night.N1"),
    ("040", "eligible_for_provisional_credit 판단 6행",
     "규칙은 doc_credit_cards_(general)_015 에 축자로 있고 **그 문서가 궤적에 안 온다**. "
     "A3 에는 포인터 행만 있다 → x541 이 그 포인터를 결정점에 놓았을 때를 쟀다."),
    ("040", "partial_refund_amount / resolution_requested 1행", "미측정"),
    ("074", "전사 순서 — T2_SG_RECORD_ORDER 의 첫 라이브",
     "격리는 x536(72샘플)·x539(4계좌 4/4·부정통제 부순다). 스모크 로그의 `재배열적용=` 수를 볼 것."),
]


def batches(tag):
    out = []
    for p in F.all_result_files():
        t = F.tag_of_file(p)
        if tag in t:
            out.append(t)
    return sorted(set(out))


def rewards(tag):
    rows = []
    for b in batches(tag):
        try:
            sims = F.sims(b, ".results.json.gz")
        except Exception:
            try:
                sims = F.sims(b)
            except Exception:
                continue
        for s in sims:
            ri = s.get("reward_info") or {}
            rows.append((b, str(s.get("task_id")), s.get("trial"), ri.get("reward")))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="t7356")
    ap.add_argument("--ref", default="t7355")
    ap.add_argument("--diffs", type=int, default=2, help="태스크당 인쇄할 실패 sim 수")
    a = ap.parse_args(argv)

    bs = batches(a.tag)
    print("=" * 78)
    print("§1 회수 — 산출물이 tracked 인가 (아니면 판정 불가·[[30]])")
    print("=" * 78)
    if not bs:
        print("  ⛔%s 배치가 하나도 없다. 런이 안 돌았거나 회수가 안 됐다." % a.tag)
        print("     확인: /home/woori/scratch/logs/bank_%s_chain.log" % a.tag)
        return 1
    for b in bs:
        sc = F.sidecar_status(b)
        print("  %-40s 사이드카=%s" % (b, sc))
    print("  ⚠사이드카가 present 가 아니면 **우리 층 거절은 판정 불가**다 — 침묵을 증거로 읽지 마라.")

    print()
    print("=" * 78)
    print("§2 성적 — reward 만 ([[69]] gold 일치율은 성적이 아니다)")
    print("=" * 78)
    rows = rewards(a.tag)
    per = collections.defaultdict(list)
    for b, tid, tr, rw in rows:
        per[tid].append(rw)
    tot = sum(1 for _, _, _, rw in rows if (rw or 0) > 0)
    print("  전체 %d sim · reward>0 **%d**" % (len(rows), tot))
    for tid in sorted(per):
        v = per[tid]
        print("    %-10s %d/%d   %s" % (tid, sum(1 for x in v if (x or 0) > 0), len(v),
                                        [x for x in v]))
    ref = rewards(a.ref)
    if ref:
        rp = collections.defaultdict(list)
        for b, tid, tr, rw in ref:
            rp[tid].append(rw)
        print("  대조 %s:" % a.ref)
        for tid in sorted(rp):
            v = rp[tid]
            print("    %-10s %d/%d" % (tid, sum(1 for x in v if (x or 0) > 0), len(v)))
    print("  ★판정선 = **표적의 0→1**. 총점 Δ 금지 · 레버가 여럿이라 개별 귀속 불가(C594).")

    print()
    print("=" * 78)
    print("§3 발화 — 배선한 레버가 라이브에서 말을 했는가 ([[24]] 死배선)")
    print("=" * 78)
    for b in bs:
        try:
            txt = F.log_text(b) if hasattr(F, "log_text") else ""
        except Exception:
            txt = ""
        if not txt:
            p = F.path_for(b, ".log.gz")
            try:
                with F.topen(p) as f:
                    txt = f.read()
            except Exception:
                txt = ""
        if not txt:
            print("  %-40s (로그 없음)" % b)
            continue
        cnt = {m: txt.count(m) for m in MARKERS}
        print("  %-40s %s" % (b, " · ".join("%s=%d" % (k.replace("T2_", ""), v)
                                            for k, v in cnt.items() if v)))
        dead = [m for m in MARKERS if m != "Traceback" and cnt.get(m, 0) == 0]
        if dead:
            print("       ⚠무발화: %s" % ", ".join(x.replace("T2_", "") for x in dead))

    print()
    print("=" * 78)
    print("§4 남은 차이 — gold 행 ↔ 궤적 호출을 자연키로 짝지어 **인자만**")
    print("=" * 78)
    shown = collections.Counter()
    for b in bs:
        try:
            sims = F.sims(b, ".results.json.gz")
        except Exception:
            continue
        for s in sims:
            rw = (s.get("reward_info") or {}).get("reward")
            tid = str(s.get("task_id"))
            if (rw or 0) > 0 or shown[tid] >= a.diffs:
                continue
            shown[tid] += 1
            try:
                d = F.action_diff(s, tag=b)
            except Exception as e:
                print("  %s trial %s — action_diff 실패 %r" % (tid, s.get("trial"), e))
                continue
            print("\n  ── %s trial=%s reward=%s basis=%s (gold 변이 %d · matched %d)"
                  % (tid, s.get("trial"), rw, d["basis"], d["n_gold"], d["n_matched"]))
            print("     ⚠basis 에 ACTION 이 없으면 matched 는 **성적에 0 기여**다(TASK_085 §0).")
            tried = collections.defaultdict(list)
            for t in d["tried"]:
                tried[(t["outer"], t["inner"])].append(t)
            for g in d["rows"]:
                if g["bench_match"]:
                    continue
                ga = g.get("args") or {}
                cands = tried.get((g["outer"], g["inner"]), [])
                key = ga.get("transaction_id") or ga.get("account_id") or ""
                best = None
                for t in cands:
                    ta = t.get("args") or {}
                    if key and ta.get("transaction_id") not in (None, key) and \
                            ta.get("account_id") not in (None, key):
                        pass
                    df = [k for k in set(ga) | set(ta) if ga.get(k) != ta.get(k)]
                    if best is None or len(df) < len(best[1]):
                        best = (t, df)
                if best is None:
                    print("     MISSING %s.%s  (같은 이름 호출 0)" % (g["outer"], g["inner"]))
                    continue
                t, df = best
                print("     MISSING %s.%s  최근접 msg=%s ok=%s 남은 차이 %d"
                      % (g["outer"], g["inner"], t.get("msg_i"), t["ok"], len(df)))
                ta = t.get("args") or {}
                for k in sorted(df)[:10]:
                    print("        %-34s gold=%r got=%r" % (k, ga.get(k), ta.get(k)))

    print()
    print("=" * 78)
    print("§5 다음 수 — 열려 있는 축과 그것을 이미 잰 파일")
    print("=" * 78)
    for tid, axis, note in OPEN_AXES:
        print("  [%s] %s" % (tid, axis))
        print("        %s" % note)
    print("""
  정본 큐   reports/facet_rft_2026/x509_axis_queue_2026_08_24.json
            → findings_2026_08_25_night (어젯밤 판정·반증조건 축자)
  인계문    _cdp_private_local/HANDOFF_2026_08_25_NIGHT2.md
  격리 산출  x536(순서 2×2) · x538/x538b(책임한도) · x539(도메인 일반 순서) ·
            x540(명세 파생 등가성) · x541(선언 조인)
  ⛔재유도 금지 — 위 수치는 인용하고, 새로 만들지 마라([[74]]).
""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
