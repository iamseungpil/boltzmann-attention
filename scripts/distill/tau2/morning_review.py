# -*- coding: utf-8 -*-
"""아침 검토 한 방 — 밤샘런을 **재유도 없이** 판정하는 데 필요한 것만 순서대로 인쇄한다.

사용자 지시 2026-08-25 밤: *"가장 효율적으로 내일 아침에 작업 다시 검토할 수 있는 계획을 세워라."*

    py -3 morning_review.py                      # 기본 = bank_night (대조 bank_x721)
    py -3 morning_review.py --tag bank_x7xx      # 다른 런

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
try:                       # 로컬(Windows cp949) 콘솔에서도 돌게 한다 — 리모트는 UTF-8
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import t2_forensic as F   # noqa: E402  (정본 재사용·사본 금지 [[67]])

MARKERS = ["[T2_WRITE_EVIDENCE]", "[T2_VIEW_COMPACT]", "[T2_CLAIMPROV]",
           "[T2_P2]", "[T2_ARG_DOC_SUB]"]
"""라이브에서 **말을 해야 하는** 계기(0이면 死배선·[[24]]/[[81]]). 문자열은 전부 소스에서 확인:
   `t2_gate_patch.py:1892,1930`(forbid=/skip=policy-branch) · `:8846` · `:14765` ·
   `t2_run_gated.py:748`(P2 사다리) · `t2_scaffold_get.py:2871`."""

ALARMS = ["**TRUNC**", "SALVAGED=", "declaration failed", "[T2_TRUNCGUARD]", "Traceback"]
"""**0이어야 하는** 것. 여기 하나라도 잡히면 그 배치는 성적 이전에 계기 문제다.
   · `**TRUNC**`(`t2_run_gated.py:895`) — 사용자 게이트 2026-09-01: *"TRUNC 는 1개도 발생하면 안된다"*.
   · `SALVAGED=`(`:894`) — 살리기가 돌았다 = 네이티브 파싱이 실패했다([[84]] 표면형↔파서 짝).
   · `declaration failed` — 선언 프로브가 JSON 을 못 냈다(상한 부족의 자국·§U-1).
   · `[T2_TRUNCGUARD]` — 절단 재생성이 돌았다(발화 자체가 절단의 증거)."""

OPEN_AXES = [
    ("TRUNC", "게이트 0 — 어젯밤 각 레인 2건이 남아 있었다",
     "처방은 상한이었다: PROBE 512->2048->8192 · JUDGE 8192->16384(프로필 축자 근거 포함). "
     "실물은 `agent_claimprov mt=2048 -> gen=2048 TRUNC reason=0B content=6970B` = **답이 상한을 넘겼다**"
     "(사고 아님). 아침에 ALARMS 의 `**TRUNC**` 가 0 이 아니면 그 태스크의 프롬프트 길이부터 본다."),
    ("012/055/057/101", "회귀 게이트 — 오늘 t3prime 통과분이 어젯밤 다섯 변경에도 버티는가",
     "하나라도 떨어지면 어젯밤 스택은 되돌린다([[70]] 무엇을 팔았나를 그 자리에서 본다). "
     "★055 는 이미 한 번 떨어졌다(account_class 'Silver Plus'->'Silver'). 후보 원인 셋: "
     "되살아난 claimprov 가드 · 뷰 임계 · nt=1 분산. 가드를 끈 짝런이 유일한 분리 수단이다([[57]])."),
    ("046/048/049", "조건부 금지(§U-1)가 라이브에서 잉여 로깅을 지웠는가",
     "판정 단위는 reward 가 아니라 **변이집합의 EXTRA**다([[69]]). 048 은 어젯밤 "
     "MISSING 0·WRONGARG 1·EXTRA 0 까지 왔고 남은 칸은 `log{cc_e3f4a5b6c7_eco, annual_fee}`(msg43). "
     "그 칸은 [[23]] 출처가 깨끗하지 않아 **일부러 안 닫았다** — 아침에 다시 열지 마라."),
    ("084", "뷰 임계 상향(T2_VIEW_COMPACT_MINTOTAL=344064)이 msg108 근거를 살렸는가",
     "확정된 원인은 우리 압축이다(그 시점 doc_031 category enum·card_action 매핑이 전부 다이제스트·재도착 0). "
     "귀속은 **다이제스트 집합 재구성**으로 가른다 — 총점 Δ 로 귀속하지 마라(런에 변경이 다섯이다)."),
    ("010/062/063/065/067/068/093/094/095", "재측정 9 — 오늘 **다른 팔**에서 통과한 것들",
     "단일 팔 수치가 없으면 pass 율을 인용할 수 없다([[54]]). 이 9개는 성적이 아니라 "
     "**비교가능성**을 위해 다시 잰 것이다. 팔을 섞어 집계하지 마라."),
    ("053/102/069", "분모 제외 후보 — 아침에 재론하지 마라",
     "053 = gold 결함(§T-19 · 도달 불가) · 102 = 순환 트리거 + per-task reward_basis 불일치 · "
     "069 = 표적 금지. 새 근거 없이 다시 조사하는 것은 [[40]] 위반이다."),
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
    ap.add_argument("--tag", default="bank_night")
    ap.add_argument("--ref", default="bank_x721")
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
    # ★배치별로 먼저 가른다 — 배치가 곧 팔인 경우가 많고, 합치면 [[54]] 비교가능성이 깨진다.
    #   (2026-09-02 실측: `--tag bank_x721` 은 viewscale 팔과 ctl 팔을 **한 줄로 합쳐** 찍고 있었다.)
    perb = collections.defaultdict(list)
    for b, tid, tr, rw in rows:
        perb[b].append(rw)
    if len(perb) > 1:
        print("  ⚠배치 %d개 — **팔이 다르면 합치지 마라**([[54]]). 배치별:" % len(perb))
        for b in sorted(perb):
            v = perb[b]
            print("    %-44s %d/%d" % (b, sum(1 for x in v if (x or 0) > 0), len(v)))
    print("  전체 %d sim · reward>0 **%d**  (배치가 같은 팔일 때만 이 줄을 인용하라)"
          % (len(rows), tot))
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
        print("  %-40s %s" % (b, " · ".join("%s=%d" % (k.strip("[]").replace("T2_", ""), v)
                                            for k, v in cnt.items() if v)))
        dead = [m for m in MARKERS if cnt.get(m, 0) == 0]
        if dead:
            print("       ⚠무발화(死배선 후보): %s"
                  % ", ".join(x.strip("[]").replace("T2_", "") for x in dead))
        al = {m: txt.count(m) for m in ALARMS}
        bad = {k: v for k, v in al.items() if v}
        if bad:
            print("       ⛔경보: %s" % " · ".join("%s=%d" % (k, v) for k, v in bad.items()))
        else:
            print("       ✅경보 0 (TRUNC·SALVAGED·declaration failed·TRUNCGUARD·Traceback)")

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
