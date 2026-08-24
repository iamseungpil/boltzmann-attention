# -*- coding: utf-8 -*-
"""x515 - 재생 게이트: 수리 술어를 **기존 궤적에 다시 태워** 무엇이 바뀌는지 센다 (2026-08-24).

사용자 승인 2026-08-24 — *"A 로 가라"* (재생 하네스를 먼저 짓는다 · 런 0회).

왜 재생인가 ([[62]](1) · t7348 의 교훈):
  t7348 은 9 커밋 1,173 삽입을 **라이브 미측정 상태로** 태웠고 그중 A-1 은 40 sim 에서 표적
  호출이 1건이었다. 수리가 **라이브 거동을 바꾸는지**를 먼저 무료로 재고, 안 바뀌면 안 태운다.

★재생이 답할 수 있는 것과 없는 것 (이 경계를 넘지 마라):
  G1 술어가 바뀌나        — 결정론. 모델 0회. **이 파일이 재는 것.**
  G2 바뀐 산출이 모델 앞에 서나 — 사이드카가 권위(055 의 `outcome="clobbered"` 가 인과를 끊은 자리).
  G3 모델이 다르게 행동하나  — ⛔재생으로 **답할 수 없다**. 격리 프로브(무료·8141)나 런이 필요하다.
  ⇒ G1 이 0 이면 그 수리는 런에 태울 값이 없다. G1 이 커도 G3 은 별개다.

표적 (x514 종합의 매수 후보):
  R1  `_exec_side` UNKNOWN→user 폴백 제거 (t2_gate_patch.py:8701-8709)
      `t2_role.executor_of` 는 이미 판정 불가 시 `UNKNOWN(None)` 을 돌려주고 그 계약이
      *"판정할 수 없으면 그 문장을 빼야지, 추측해서 말하면 T1(사실 모순)이 다시 생긴다"* 인데,
      호출부가 `return "assistant" if _n in _agent_names else "user"` 로 **user 에 떨어뜨린다**.
      코드 주석 자신이 *"거동 보존을 위해 종전대로 … 조이는 것은 별도 측정 단계다"* 라고 적어
      둔, 바로 그 별도 측정 단계다.
      실물: `task_016.user_tools = ["submit_transaction","apply_for_credit_card"]` 인데
      `pending_user` 에 `submit_referral` 이 들어가 `formalized_target=submit_referral` 로 고정.

계기: 로그의 `[T2_ACTIONREQ]` 줄이 `pending_user`·`pending_agent`·`formalized_target` 을
      **런타임 값 그대로** 인쇄한다 ⇒ 재유도 없이 직독한다. 태스크 `user_tools` 는 결과 파일의
      `tasks[]` 선언(env)에서 읽는다 — gold 아님([[23]]).

실행: PYTHONIOENCODING=utf-8 python x515_replay_gate.py [tag ...]
"""
import collections
import glob
import gzip
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

import t2_forensic as F          # noqa: E402

SIMS = os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026", "sim_results")
OUT = os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026")

SIMRE = re.compile(r"\[sim=(task_\d+#s\d+)\]")
AREQ = re.compile(r"\[T2_ACTIONREQ\] window=(\w+) pending_user=(\[[^\]]*\]) "
                  r"pending_agent=(\[[^\]]*\]) formalized_target=(\S+)")

# discoverable 래퍼는 `executor_of` 가 `user_tools.get_discoverable_tools` 로도 USER 를 준다.
# 결과 파일의 `tasks[].user_tools` 는 그 하위 목록을 안 싣기도 하므로 **관대 판정**을 따로 센다.
DISC_USER = ("call_discoverable_user_tool",)


def parse_list(s):
    try:
        return [x.strip().strip("'\"") for x in s[1:-1].split(",") if x.strip()]
    except Exception:
        return []


def main(argv):
    pats = argv or ["bank_t7348_halfA_20260824", "bank_t7348_halfB_20260824",
                    "bank_t7346_halfA_20260822", "bank_t7346_halfB_20260822"]
    user_tools = {}          # task -> set(env 선언)
    gold_user = collections.defaultdict(set)   # task -> gold requestor=user 이름
    for p in pats:
        rp = os.path.join(SIMS, p + ".results.json.gz")
        if not os.path.exists(rp):
            rp = os.path.join(SIMS, p + "_results.json.gz")
        if not os.path.exists(rp):
            continue
        d = json.load(gzip.open(rp, "rt", encoding="utf-8", errors="replace"))
        for t in (d.get("tasks") or []):
            tid = t.get("id")
            if not tid:
                continue
            user_tools[tid] = set(t.get("user_tools") or [])
            for a in ((t.get("evaluation_criteria") or {}).get("actions") or []):
                if a.get("requestor") == "user" and a.get("name"):
                    gold_user[tid].add(a["name"])

    # ── 0. 안전 검정: 수리가 **필요한 이름을 떨어뜨리지 않는가**
    #    gold 의 requestor=user 액션이 그 태스크 `user_tools` 안에 다 있으면, 이름을 좁혀도
    #    정답 유도는 살아남는다. 하나라도 밖에 있으면 그 태스크에서는 수리가 해가 된다.
    print("=" * 100)
    print("(0) 안전 검정 — gold 의 requestor=user 액션이 태스크 `user_tools` 안에 있나")
    print("=" * 100)
    unsafe = []
    for tid in sorted(gold_user):
        out = sorted(gold_user[tid] - user_tools.get(tid, set()) - set(DISC_USER))
        mark = "OK" if not out else "⛔밖에 있음: " + ", ".join(out)
        if out:
            unsafe.append(tid)
        print("  %-10s gold-user %-46s %s"
              % (tid, ",".join(sorted(gold_user[tid]))[:46], mark))
    print("")
    print("  ⇒ 수리가 해가 되는 태스크: %s" % (", ".join(unsafe) if unsafe else "**없음**"))

    # ── 1. G1 재생: 각 ACTIONREQ 발화에서 표적이 이름 안에 있나
    rows = collections.defaultdict(lambda: {"turns": 0, "false": 0, "true": 0,
                                            "user_branch": 0, "not_user_branch": 0,
                                            "targets": collections.Counter(),
                                            "false_targets": collections.Counter(),
                                            "sims": set(), "false_sims": set(),
                                            "cand_false": collections.Counter()})
    for p in pats:
        lp = os.path.join(SIMS, p + ".log.gz")
        if not os.path.exists(lp):
            continue
        with gzip.open(lp, "rt", encoding="utf-8", errors="replace") as f:
            for ln in f:
                if "[T2_ACTIONREQ]" not in ln:
                    continue
                m0, m1 = SIMRE.search(ln), AREQ.search(ln)
                if not (m0 and m1):
                    continue
                simtag = m0.group(1)
                tid = simtag.split("#")[0]
                pend_u = parse_list(m1.group(2))
                tgt = m1.group(4)
                allow = user_tools.get(tid, set()) | set(DISC_USER)
                r = rows[tid]
                r["turns"] += 1
                r["sims"].add("%s|%s" % (p, simtag))
                r["targets"][tgt] += 1
                for c in pend_u:
                    if c not in allow:
                        r["cand_false"][c] += 1
                # ★게이트 술어를 그대로 쓴다: 손님-실행 안내 분기는 `if _utgt in _upending:` 이다
                #   (t2_gate_patch.py). 표적이 **에이전트 도구**(pending_agent)이거나 `None` 이면
                #   그 분기는 애초에 안 탄다 — 그것을 '이름밖' 으로 세면 폭발반경이 6배로 뻥튀기된다
                #   (초판이 그 실수를 했다: 1,090/1,268. 실제 분기 발화는 아래가 센다).
                if tgt not in pend_u:
                    r["not_user_branch"] += 1
                    continue
                r["user_branch"] += 1
                if tgt in allow:
                    r["true"] += 1
                else:
                    r["false"] += 1
                    r["false_targets"][tgt] += 1
                    r["false_sims"].add("%s|%s" % (p, simtag))

    print("")
    print("=" * 100)
    print("(1) G1 재생 — `_exec_side` UNKNOWN→user 폴백을 제거하면 무엇이 사라지나")
    print("=" * 100)
    print("%-10s %6s %8s %7s %7s  %-30s %s"
          % ("task", "ACTREQ", "user분기", "이름밖", "이름안", "사라지는 표적", "후보서 빠지는 이름"))
    print("-" * 100)
    tot = collections.Counter()
    for tid in sorted(rows):
        r = rows[tid]
        for k in ("turns", "false", "true", "user_branch"):
            tot[k] += r[k]
        tot["sims"] += len(r["sims"])
        tot["false_sims"] += len(r["false_sims"])
        ft = " · ".join("%s×%d" % kv for kv in r["false_targets"].most_common(2)) or "-"
        cf = " · ".join("%s×%d" % kv for kv in r["cand_false"].most_common(2)) or "-"
        print("%-10s %6d %8d %7d %7d  %-30s %s"
              % (tid, r["turns"], r["user_branch"], r["false"], r["true"], ft[:30], cf[:34]))
    print("-" * 100)
    print("%-10s %6d %8d %7d %7d  (이름밖 sim %d / 전체 %d)"
          % ("합계", tot["turns"], tot["user_branch"], tot["false"], tot["true"],
             tot["false_sims"], tot["sims"]))
    print("")
    print("판독: **user분기** = `_utgt in _upending` 이 참이라 손님-실행 안내가 실제로 걸린 발화.")
    print("      **이름밖** = 그중 그 태스크 env 선언에 없는 도구를 실행하라고 한 것 = 수리하면")
    print("      **사라진다**. **이름안** = 수리해도 유지 = 우리가 파는 것이 아니다([[70]]).")

    # ── 2. G2: 그 발화가 모델 앞에 실제로 섰나 (사이드카가 권위)
    print("")
    print("=" * 100)
    print("(2) G2 배달 검정 — 사라질 발화가 모델 앞에 실제로 섰나 (사이드카)")
    print("=" * 100)
    delivered = collections.Counter()
    for p in pats:
        try:
            sc = F.sidecar(p)
        except Exception:
            sc = {}
        for tid, r in rows.items():
            for key in r["false_sims"]:
                run, simtag = key.split("|", 1)
                if run != p:
                    continue
                got = 0
                for o in (sc.get(simtag) or []):
                    if (o.get("kind") or "") != "reminder-user":
                        continue
                    txt = o.get("text") or ""
                    if "[ACTION]" in txt and (o.get("len") or 0) > 0:
                        got += 1
                delivered[(tid, "sims")] += 1
                if got:
                    delivered[(tid, "delivered_sims")] += 1
                    delivered[(tid, "rows")] += got
    print("%-10s %10s %12s %10s" % ("task", "이름밖 sim", "배달된 sim", "배달 행수"))
    print("-" * 100)
    d_tot = collections.Counter()
    for tid in sorted(rows):
        if not delivered[(tid, "sims")]:
            continue
        print("%-10s %10d %12d %10d" % (tid, delivered[(tid, "sims")],
                                        delivered[(tid, "delivered_sims")],
                                        delivered[(tid, "rows")]))
        d_tot["sims"] += delivered[(tid, "sims")]
        d_tot["dsims"] += delivered[(tid, "delivered_sims")]
        d_tot["rows"] += delivered[(tid, "rows")]
    print("-" * 100)
    print("%-10s %10d %12d %10d" % ("합계", d_tot["sims"], d_tot["dsims"], d_tot["rows"]))
    print("")
    print("⚠사이드카가 없으면 '안 섰다'가 아니라 **'모른다'** 다([[25]]). 배달 0 인 태스크는")
    print("  사이드카 유무부터 확인하라 — 침묵을 증거로 읽지 마라.")

    # ── 3. R2 재생: `basis_max_chars` 스윕 (근거 창을 넓히면 무엇이 통과하나)
    #    ⛔[[23]] 경고: 072 반증자는 *"gold 072_7 이 12000 에서 전부 접지된다"* 로 값을 지목했다.
    #      그 계수는 **진단**으로는 유효하지만, 그것을 근거로 파라미터를 12000 으로 고르면
    #      **gold 로 임계를 고르는 것**이라 실험이 무효가 된다. 그래서 여기서는 값을 고르지 않고
    #      **스윕의 양쪽(무엇이 통과로 바뀌나 ↔ 무엇이 오통과로 바뀌나)만** 인쇄한다.
    #      실제 채택은 정책·env 로부터 도출된 이유가 있을 때만 하라.
    print("")
    print("=" * 100)
    print("(3) R2 스윕 — `basis_max_chars` 를 넓히면 모델이 낸 값의 접지가 어떻게 변하나")
    print("=" * 100)
    try:
        import t2_subcall as SC
    except Exception as e:
        SC = None
        print("  t2_subcall 없음: %r" % (e,))
    if SC is not None:
        class _M(object):
            __slots__ = ("role", "content", "error")

            def __init__(self, d):
                self.role = d.get("role")
                self.content = d.get("content")
                self.error = bool(d.get("error"))

        CAPS = (4000, 8000, 12000, 20000, 0)
        MUT = F.mutating_tools()
        swp = collections.defaultdict(lambda: collections.Counter())
        for p in pats:
            rp = os.path.join(SIMS, p + ".results.json.gz")
            if not os.path.exists(rp):
                continue
            d = json.load(gzip.open(rp, "rt", encoding="utf-8", errors="replace"))
            for s in (d.get("simulations") or []):
                tid = s.get("task_id")
                msgs = [_M(m) for m in (s.get("messages") or [])]
                try:
                    md = F.mutation_diff(s, MUT)
                except Exception:
                    continue
                for grp, items in (("갈림", md.get("wrongarg") or []),
                                   ("맞음", md.get("matched") or [])):
                    for e in items:
                        mi = e.get("msg_i")
                        if mi is None:
                            continue
                        vals = [str(v) for v in (e.get("args") or {}).values()
                                if v not in (None, "", True, False) and len(str(v)) >= 3]
                        if not vals:
                            continue
                        for cap in CAPS:
                            basis = SC.recent_tool_text(msgs[:mi], cap, "recent")
                            ok = sum(1 for v in vals if v in basis)
                            swp[(tid, grp, cap)]["vals"] += len(vals)
                            swp[(tid, grp, cap)]["grounded"] += ok
                            swp[(tid, grp, cap)]["calls"] += 1
                            if ok == len(vals):
                                swp[(tid, grp, cap)]["full"] += 1
        print("%-10s %-5s  %s" % ("task", "판정", "  ".join("cap%-6s" % (c or "∞") for c in CAPS)))
        print("-" * 100)
        for tid in sorted({k[0] for k in swp}):
            for grp in ("맞음", "갈림"):
                cells = []
                for cap in CAPS:
                    c = swp.get((tid, grp, cap))
                    cells.append("%-9s" % (("%d/%d" % (c["full"], c["calls"])) if c else "·"))
                if any(x.strip() != "·" for x in cells):
                    print("%-10s %-5s  %s" % (tid, grp, "  ".join(cells)))
        print("")
        print("판독: 값 = **인자 전부가 근거 창 안에 있는 호출 수 / 그 판정의 호출 수**.")
        print("      '갈림' 행이 cap 을 넓혀 올라가면 그것은 **오통과가 늘어나는 것**이다 —")
        print("      우리가 파는 쪽이다([[70]]). '맞음' 행이 올라가는 것만이 사는 쪽이다.")
        print("⛔[[23]]: 이 표로 cap 값을 고르지 마라. gold 적합으로 임계를 고르면 실험이 무효다.")

    print("")
    print("=" * 100)
    print("G3 은 이 파일이 답하지 않는다")
    print("=" * 100)
    print("  후보집합에서 이름이 빠지면 격리 서브의 `formalized_target` 이 **무엇으로 바뀌는지**는")
    print("  모델을 다시 돌려야 안다. 무료 격리 프로브(8141)로 재고, 그 전에는 '문면이 사라진다'")
    print("  까지만 주장하라([[62]](1)).")

    out = {"probe": "x515_replay_gate", "date": "2026-08-24",
           "repair": "R1 `_exec_side` UNKNOWN→user 폴백 제거 (t2_gate_patch.py:8701-8709)",
           "runs": list(pats),
           "safety": {"gold_user_outside_user_tools": unsafe},
           "user_tools": {k: sorted(v) for k, v in user_tools.items()},
           "gold_user": {k: sorted(v) for k, v in gold_user.items()},
           "g1": {tid: {"turns": r["turns"], "user_branch": r["user_branch"],
                        "false": r["false"], "true": r["true"],
                        "sims": len(r["sims"]), "false_sims": len(r["false_sims"]),
                        "false_targets": dict(r["false_targets"]),
                        "cand_false": dict(r["cand_false"])}
                  for tid, r in rows.items()},
           "g2": {"%s|%s" % k: v for k, v in delivered.items()},
           "r2_basis_sweep": {"%s|%s|%s" % k: dict(v) for k, v in swp.items()} if SC else {},
           "r2_verdict": "★4000→12000 은 072 에서 **아무것도 바꾸지 않는다**(맞음 1/7·갈림 1/11 불변)."
                         " 코퍼스 전체로도 12000 에서 움직이는 것은 093(맞음 3/8→4/8·갈림 0/2→1/2)"
                         " 과 098(맞음 4/6→5/6) 뿐이고, cap∞ 로 열면 사는 쪽과 파는 쪽이 같이 는다"
                         "(073 맞음 5→6 · 갈림 6→8). ⇒ **R2 는 G1 이 사실상 0 이므로 런에 태울"
                         " 값이 없다.** ⚠이 계수는 **모델이 실제로 낸 값**의 접지를 잰 것이고,"
                         " 072 반증자가 잰 것은 **gold 072_7 의 값**이다 — 서로 다른 양이므로 모순이"
                         " 아니다. 다만 gold 적합으로 cap 을 고르는 것은 [[23]] 위반이다.",
           "limits": ["G3(모델이 다르게 행동하나)은 재생으로 답할 수 없다 — 격리 프로브 필요.",
                      "`tasks[].user_tools` 는 env 선언이고 gold 가 아니다([[23]]).",
                      "discoverable 래퍼는 관대 판정으로 USER 에 남겼다(`DISC_USER`).",
                      "사이드카 부재 = 모른다([[25]])."]}
    dst = os.path.join(OUT, "x515_replay_gate_2026_08_24.json")
    with io.open(dst, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("")
    print("-> %s" % os.path.normpath(dst))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
