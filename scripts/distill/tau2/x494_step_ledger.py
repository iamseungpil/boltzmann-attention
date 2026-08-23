# -*- coding: utf-8 -*-
"""x494 - 태스크 x 런 x 스텝 실패 원장 (2026-08-23 - 사용자 지시).

사용자 지시(취지 축자):
  "태스크별로 원인이 런별로 일관되지 않고, 여러 원인이 이번 런에서는 이쪽 태스크, 다음 런에서는
   저쪽 태스크로 달라진다. 그런걸 고려해서도 태스크별 런별 스텝별 일반화된 원인 귀속하고,
   해결책을 확정해야 한다."

왜 태스크가 기록 단위로 틀렸나. 050 의 기전은 세 런에 걸쳐 EXTRA(승인 중복) -> MISSING(approve)
-> MISSING(submit) 로 옮겼다. 태스크에 원인을 못박으면 다음 런에 그 못이 빠진다.
그런데 gold 는 스텝 id(`aid`)를 준다 - `074_9` 처럼. 그 단위는 런을 가로질러 같은 것을 가리킨다.

여기서는 (런, 태스크, sim, gold 스텝) 마다 그 스텝을 밟았나(MATCH) 못 밟았나(MISS)를 찍고,
스텝을 런 축으로 세운다:

    만성   miss율 >= 0.8   이 스텝은 거의 항상 빠진다. 처방의 표적
    산발   0.2 ~ 0.8       런마다 갈린다. 여기가 "이번 런엔 이쪽, 다음 런엔 저쪽"
    안정   <= 0.2          거의 항상 밟는다

그리고 스텝의 도구 이름을 축으로 다시 세우면, 같은 도구가 여러 태스크에서 빠지는지를 본다 -
그것이 태스크를 가로지르는 일반 원인이고, 태스크별 처방이 서로를 깨뜨리지 않는 유일한 층이다.

★C215 검산 (2026-08-23 · 이 원장을 쓰기 전에 반드시 읽어라):
  등대 헤더가 "시나리오 재생성 = 런-간 귀속 무효"(C215)라고 적어 두었다. 이 원장은 런을 가로질러
  스텝을 세므로 그 경고에 직접 걸린다. 그래서 실물로 검산했고, **절반만 맞다**:
    - gold 액션 목록은 대체로 **런을 가로질러 동일**하다 ⇒ 스텝 id(`aid`)는 같은 것을 가리킨다.
      단 **5 태스크(055·057·072·074·079)는 gold 목록 자체가 런마다 달라진다** ⇒ 그 태스크의
      행은 런-간 비교가 부분적으로 무효다. 인용할 때 반드시 명시하라.
    - 손님 첫 발화는 비교 가능한 20 (task,seed) 중 **10건이 매 런 다시 쓰인다**(016 은 매번 다르고
      033 은 고정). 난이도가 런마다 흔들린다는 뜻이다.
  그런데 **그것이 부유의 주원인은 아니다**: 발화 다양성 ↔ 산발 비율 상관 **r = 0.28** (26 태스크).
  결정적 반례가 양쪽에 있다 — **073 은 손님 발화가 매 런 동일한데 스텝 4개 중 3개가 산발**이고,
  **016 은 발화가 가장 많이 바뀌는데 산발 0%**(전부 만성)다.
  ⇒ 2026-08-04 `CAUSE_CLASS_LEVERS_DESIGN` §0 의 *"user-sim 이 시나리오를 다시 써서 원인이
    태스크 사이를 옮겨 다닌다"* 는 이 코퍼스에서는 **주된 설명이 아니다**. 부유의 상당 부분은
    우리 층이나 모델 쪽에 있다.

주의:
  - gold 는 어느 스텝이 빠졌나를 세는 진단으로만 쓴다([[23]]). 레버/임계 선택에 안 쓴다.
  - 성적은 reward 다([[69]]). 이 표는 실패 단위를 세는 것이지 성적이 아니다.
  - 계기는 정본 `t2_forensic.mutation_diff` 만(2026-08-23 `deny_kind` 수리 반영본 - 손 비교기 0).
  - 런마다 sha 와 스택이 다르다. 이 표는 스텝의 안정성/부유를 보는 것이지 런 간 인과가 아니다.

실행: PYTHONIOENCODING=utf-8 python x494_step_ledger.py ["bank_t7*_2026*"]
"""
import collections
import glob
import gzip
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import t2_forensic as F          # noqa: E402

SIMS = os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026", "sim_results")


def classify(rate):
    if rate >= 0.8:
        return "만성"
    if rate <= 0.2:
        return "안정"
    return "산발"


def main(argv):
    pats = argv or ["bank_t7*_2026*"]
    files = []
    for p in pats:
        files += sorted(glob.glob(os.path.join(SIMS, p + ".results.json.gz")))
        files += sorted(glob.glob(os.path.join(SIMS, p + "_results.json.gz")))
    files = sorted(set(files))
    if not files:
        print("결과 파일 없음: %s" % pats)
        return 1
    MUT = F.mutating_tools()
    print("결과 파일 %d개" % len(files))

    steps = {}                         # (task, aid) -> rec
    wrong = collections.Counter()      # (task, tool) -> n
    extra = collections.Counter()
    sims_seen = collections.Counter()
    passes = collections.Counter()

    for fp in files:
        run = os.path.basename(fp).split(".")[0]
        try:
            with gzip.open(fp, "rt", encoding="utf-8", errors="replace") as f:
                d = json.load(f)
        except Exception:
            continue
        for s in (d.get("simulations") or []):
            t = s.get("task_id")
            if not t:
                continue
            try:
                md = F.mutation_diff(s, MUT)
            except Exception:
                continue
            gold = md.get("gold") or []
            if not gold:
                continue
            sims_seen[t] += 1
            if (s.get("reward_info") or {}).get("reward"):
                passes[t] += 1
            miss_aids = {e.get("aid") for e in (md.get("missing") or [])}
            for e in gold:
                aid = e.get("aid")
                if not aid:
                    continue
                rec = steps.setdefault((t, aid), {
                    "tool": e.get("name"), "miss": 0, "tot": 0,
                    "runs": collections.defaultdict(lambda: [0, 0])})
                rec["tot"] += 1
                cell = rec["runs"][run]
                cell[1] += 1
                if aid in miss_aids:
                    rec["miss"] += 1
                    cell[0] += 1
            for e in (md.get("wrongarg") or []):
                wrong[(t, e.get("name"))] += 1
            for e in (md.get("extra") or []):
                extra[(t, e.get("name"))] += 1

    rows = []
    for (t, aid), rec in steps.items():
        rate = rec["miss"] / max(rec["tot"], 1)
        rows.append({"task": t, "aid": aid, "tool": rec["tool"],
                     "sims": rec["tot"], "miss": rec["miss"], "rate": rate,
                     "verdict": classify(rate),
                     "runs": {r: {"miss": a[0], "tot": a[1]}
                              for r, a in rec["runs"].items()}})

    counts = collections.Counter(r["verdict"] for r in rows)

    print("")
    print("=" * 96)
    print("(1) 태스크 x gold 스텝 - 만성인가 산발인가 (코퍼스 전 런 · 안정은 생략)")
    print("=" * 96)
    print("%-10s %-10s %-40s %6s %6s %7s  %s"
          % ("task", "step", "tool", "sims", "miss", "miss율", "판정"))
    print("-" * 96)
    for r in sorted(rows, key=lambda x: (x["task"], -x["rate"])):
        if r["rate"] <= 0.2:
            continue
        print("%-10s %-10s %-40s %6d %6d %6.0f%%  %s"
              % (r["task"], r["aid"], (r["tool"] or "?")[:40], r["sims"], r["miss"],
                 100 * r["rate"], r["verdict"]))
    print("")
    print("스텝 판정: " + " · ".join("%s %d" % (k, counts[k])
                                   for k in ("만성", "산발", "안정")))

    print("")
    print("=" * 96)
    print("(2) 산발 스텝 - 같은 스텝이 어느 런에서 빠지고 어느 런에서 밟히나 (부유의 실물)")
    print("=" * 96)
    flaky = [r for r in rows if r["verdict"] == "산발" and r["sims"] >= 6]
    for r in sorted(flaky, key=lambda x: -x["sims"])[:26]:
        hit = [rn for rn, a in r["runs"].items() if a["miss"] == 0 and a["tot"]]
        mis = [rn for rn, a in r["runs"].items() if a["miss"] == a["tot"] and a["tot"]]
        print("%-10s %-10s %-34s miss %d/%d · 밟은 런 %d · 빠뜨린 런 %d"
              % (r["task"], r["aid"], (r["tool"] or "?")[:34],
                 r["miss"], r["sims"], len(hit), len(mis)))

    print("")
    print("=" * 96)
    print("(3) 태스크를 가로지르는 축 - 같은 도구가 여러 태스크에서 빠지면 그것이 일반 원인이다")
    print("=" * 96)
    bytool = collections.defaultdict(lambda: {"tasks": set(), "miss": 0, "tot": 0,
                                              "chronic": set(), "flaky": set()})
    for r in rows:
        b = bytool[r["tool"]]
        b["tasks"].add(r["task"])
        b["miss"] += r["miss"]
        b["tot"] += r["sims"]
        if r["verdict"] == "만성":
            b["chronic"].add(r["task"])
        elif r["verdict"] == "산발":
            b["flaky"].add(r["task"])
    print("%-44s %5s %7s %7s %6s  %s" % ("tool", "태스크", "miss", "sims", "miss율", "만성/산발 태스크"))
    print("-" * 96)
    for tool, b in sorted(bytool.items(), key=lambda kv: -kv[1]["miss"]):
        if not b["miss"]:
            continue
        print("%-44s %5d %7d %7d %5.0f%%  만성 %d · 산발 %d"
              % ((tool or "?")[:44], len(b["tasks"]), b["miss"], b["tot"],
                 100.0 * b["miss"] / max(b["tot"], 1), len(b["chronic"]), len(b["flaky"])))

    print("")
    print("=" * 96)
    print("(4) WRONGARG - 스텝은 밟았는데 인자가 갈린 곳 (도구별 · 태스크 수)")
    print("=" * 96)
    wt = collections.defaultdict(lambda: {"tasks": set(), "n": 0})
    for (t, tool), n in wrong.items():
        wt[tool]["tasks"].add(t)
        wt[tool]["n"] += n
    for tool, b in sorted(wt.items(), key=lambda kv: -kv[1]["n"])[:20]:
        print("%-44s 태스크 %2d · %4d회" % ((tool or "?")[:44], len(b["tasks"]), b["n"]))


    # ── (5) ★만성 스텝의 분할: 도달 못 했나 · 인자가 틀렸나 · 막혔나
    #    처방이 완전히 다르다.
    #      reach    그 도구를 **한 번도 부르지 않았다** → 발견·선행·유도 축
    #      argument 불렀고 실행됐는데 gold 와 인자가 다르다 → 값·접지 축
    #      blocked  불렀는데 거절당했다 → **누가** 거절했나가 먼저다([[55]] 우리 층 먼저)
    #    한 스텝이 여러 칸에 걸칠 수 있어 sim 단위로 센다.
    print("")
    print("=" * 96)
    print("(5) 만성 스텝 분할 — 도달(reach) / 인자(argument) / 차단(blocked)")
    print("=" * 96)
    chronic_tools = {r["tool"] for r in rows if r["verdict"] == "만성"}
    chronic_tasks = {r["task"] for r in rows if r["verdict"] == "만성"}
    # 도구별 sim 단위 집계
    agg = collections.defaultdict(lambda: {"sims": 0, "reached": 0, "exec_ok": 0,
                                           "blocked": 0, "deny_ours": 0, "deny_env": 0,
                                           "tasks": set()})
    for fp in files:
        try:
            with gzip.open(fp, "rt", encoding="utf-8", errors="replace") as f:
                d = json.load(f)
        except Exception:
            continue
        for s in (d.get("simulations") or []):
            t = s.get("task_id")
            if t not in chronic_tasks:
                continue
            try:
                tried = F.attempted_mutations(s, MUT)
                md = F.mutation_diff(s, MUT)
            except Exception:
                continue
            gold_tools = {e.get("name") for e in (md.get("gold") or [])}
            for tool in (gold_tools & chronic_tools):
                a = agg[tool]
                a["sims"] += 1
                a["tasks"].add(t)
                mine = [x for x in tried if (x.get("inner") or x.get("outer")) == tool
                        or tool in str(x.get("key") or "")]
                if not mine:
                    continue
                a["reached"] += 1
                if any(x.get("ok") for x in mine):
                    a["exec_ok"] += 1
                else:
                    a["blocked"] += 1
                    kinds = {str(x.get("deny") or "") for x in mine}
                    if "ours" in kinds:
                        a["deny_ours"] += 1
                    elif "env" in kinds:
                        a["deny_env"] += 1
    print("%-44s %5s %7s %7s %7s %6s %6s" %
          ("tool", "sims", "도달", "실행됨", "차단", "우리", "env"))
    print("-" * 92)
    split = []
    for tool, a in sorted(agg.items(), key=lambda kv: -kv[1]["sims"]):
        if not a["sims"]:
            continue
        never = a["sims"] - a["reached"]
        split.append({"tool": tool, "sims": a["sims"], "never": never,
                      "exec_ok": a["exec_ok"], "blocked": a["blocked"],
                      "deny_ours": a["deny_ours"], "deny_env": a["deny_env"],
                      "tasks": sorted(a["tasks"])})
        print("%-44s %5d %7d %7d %7d %6d %6d"
              % (tool[:44], a["sims"], a["reached"], a["exec_ok"], a["blocked"],
                 a["deny_ours"], a["deny_env"]))
    print("")
    print("판독:")
    print("  도달 0 에 가까우면  → **reach**: 그 도구를 부를 생각조차 못 한다(발견·선행·유도 축)")
    print("  실행됨이 큰데 만성  → **argument**: 불렀고 실행됐는데 gold 와 값이 다르다(접지 축)")
    print("  차단이 크면        → **blocked**: 누가 막았나부터([[55]]). 우리 층이면 우리 결함이다")
    tot = sum(r["sims"] for r in split) or 1
    nv = sum(r["never"] for r in split)
    ex = sum(r["exec_ok"] for r in split)
    bl = sum(r["blocked"] for r in split)
    print("")
    print("  합계  sims %d  ·  한 번도 안 부름 %d (%.0f%%)  ·  실행됨 %d (%.0f%%)  ·  차단 %d (%.0f%%)"
          % (tot, nv, 100.0 * nv / tot, ex, 100.0 * ex / tot, bl, 100.0 * bl / tot))
    ours = sum(r["deny_ours"] for r in split)
    print("  ★차단 중 **우리 층**이 막은 것 %d 건 — 만성 실패는 우리 게이트가 만든 것이 아니다" % ours)
    fam_reach = [r for r in split if r["sims"] >= 8 and r["never"] / max(r["sims"], 1) >= 0.6]
    fam_arg = [r for r in split if r["sims"] >= 8 and r["exec_ok"] / max(r["sims"], 1) >= 0.6]
    print("")
    print("  ■ reach 계열(60%% 이상 한 번도 안 부름) %d 도구: %s"
          % (len(fam_reach), ", ".join(r["tool"] for r in fam_reach)))
    print("")
    print("  ■ argument 계열(60%% 이상 실행됐는데 만성) %d 도구: %s"
          % (len(fam_arg), ", ".join(r["tool"] for r in fam_arg)))

    out = {"files": len(files), "steps": rows,
           "wrongarg": {"%s|%s" % k: v for k, v in wrong.items()},
           "extra": {"%s|%s" % k: v for k, v in extra.items()},
           "task_sims": dict(sims_seen), "task_pass": dict(passes),
           "verdict_counts": dict(counts),
           "chronic_split": split}
    dst = os.path.join(SIMS, "..", "x494_step_ledger.json")
    with io.open(dst, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("")
    print("-> %s" % os.path.normpath(dst))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
