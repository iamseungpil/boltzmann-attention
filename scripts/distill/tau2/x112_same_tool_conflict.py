# -*- coding: utf-8 -*-
"""같은 턴에 **같은 도구를 두고** "지금 하라"와 "하지 마라"가 함께 도착한 횟수.

설계서 `LEVER_ARBITRATION_PHASE_CONDITION_DESIGN_2026_08_06` §7.2e-정정이 이 수치를 근거로
강등을 철회했지만(“86턴 / 41 sim”), 그 정의를 구현한 **커밋된 스크립트가 없었고** 수치는
완주 전 부분 런(우리 문구 912행) 기준이었다. 정의를 여기 고정해 완주분에서 다시 센다([[08]]).

**정의(닫힘·전부 우리 문구 안에서 판정)**
  · 대상 행 = 사이드카 `kind ∈ {reminder-user, tool-deny}` (모델 재생성분 `reminder-assistant` 제외 — x104 규약)
  · 표적    = 본문 ∩ 도구 레지스트리(`a2/env_surface.json`) — x104 `targets_of`와 동일
  · 지시 극성:
      DO    = 그 도구 이름 주변에 명령형 호출 어구(call/give/run/use/hand over/re-issue …)가 있다
      DONT  = 그 도구 이름 주변에 차단·금지 어구(do NOT/blocked/cannot/must not/prohibit/without …)가 있다
    두 어구가 **같은 문장 안**에 있을 때만 극성을 인정한다(문서 전체를 보면 모든 긴 문구가 양성이 된다).
  · 모순    = 같은 (sim, turn)에서 **한 행은 DO, 다른 행은 DONT**이고 표적이 **겹친다**.
    같은 행 안의 DO+DONT는 모순이 아니다(하나의 문구가 조건부로 말하는 것).

  usage:  x112_same_tool_conflict.py <tag>        e.g. 20260806b
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
from x104_lever_arbitration_census import (TAGRE, OURS, lever_of,        # noqa: E402
                                           targets_of, tool_universe)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

TAG = sys.argv[1] if len(sys.argv) > 1 else "20260806b"
DOMAIN = os.environ.get("X112_DOMAIN", "banking_knowledge")
SIM_REMOTE = ("/home/woori/workspace_common/boltzmann-attention-pi/"
              "reports/facet_rft_2026/sim_results")
SIM = SIM_REMOTE if os.path.isdir(SIM_REMOTE) else os.path.abspath(
    os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026", "sim_results"))

DO = re.compile(r"\b(call|give|run|use|hand (?:it )?over|re-?issue|execute|invoke)\b", re.I)
DONT = re.compile(r"(do NOT|don't|do not|must not|cannot|can't|blocked|prohibit\w*|"
                  r"not allowed|without|before you|never)", re.I)
SENT = re.compile(r"[^.!?\n]+[.!?\n]?")


def polarity(text, tool):
    """이 문구가 그 도구에 대해 DO/DONT/양쪽 중 무엇을 말했는가 — **문장 단위**로 본다."""
    do = dont = False
    pat = re.compile(r"(?<![a-z_0-9])%s(?![a-z_0-9])" % re.escape(tool))
    for s in SENT.findall(text or ""):
        if not pat.search(s):
            continue
        if DONT.search(s):
            dont = True
        elif DO.search(s):
            do = True
    return do, dont


def load_rows():
    files = sorted(glob.glob("/home/woori/scratch/logs/fb_n97_gpu*_%s.jsonl" % TAG))
    files += [p for p in [os.path.join(SIM, "fb_%s.jsonl.gz" % TAG)] if os.path.exists(p)]
    rows = []
    for f in files:
        op = gzip.open if f.endswith(".gz") else io.open
        for line in op(f, "rt", encoding="utf-8"):
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows, files


def main():
    rows, files = load_rows()
    ours = [r for r in rows if r.get("kind") in OURS]
    universe = tool_universe(DOMAIN)
    print("== 계기 규약 ==")
    print("  파일: %s" % ", ".join(os.path.basename(f) for f in files))
    print("  전체 %d행 · 우리 문구 %d행 · sim %d개"
          % (len(rows), len(ours), len({r.get("sim") for r in ours})))
    print("  극성 판정 = 도구 이름이 **있는 문장 안에서만** DO/DONT 어구를 본다\n")

    byturn = collections.defaultdict(list)
    for r in ours:
        byturn[(r.get("sim"), r.get("turn"))].append(r)

    turns_with_text = len([k for k, v in byturn.items() if any((x.get("text") or "").strip() for x in v)])
    # 설계서 §7.2e-정정과 **같은 분모**를 내기 위해 중간 단계도 센다:
    #   both = 한 턴에 DO 행과 DONT 행이 (도구 일치와 무관하게) 함께 온 턴
    both_turns = 0
    conflict_turns, conflict_sims = 0, set()
    tgt_count = collections.Counter()
    pair_count = collections.Counter()
    samples = []
    for (sim, turn), rs in sorted(byturn.items()):
        do_map, dont_map = collections.defaultdict(list), collections.defaultdict(list)
        for r in rs:
            txt = r.get("text") or ""
            for t in targets_of(r, universe):
                d, n = polarity(txt, t)
                if n:
                    dont_map[t].append(r)
                elif d:
                    do_map[t].append(r)
        if do_map and dont_map:
            both_turns += 1
        hit = False
        for t in set(do_map) & set(dont_map):
            # 같은 행이 양쪽에 든 경우는 모순이 아니다 — 서로 다른 행이어야 한다.
            if not any(a is not b for a in do_map[t] for b in dont_map[t]):
                continue
            hit = True
            tgt_count[t] += 1
            pair_count[(lever_of(do_map[t][0]), lever_of(dont_map[t][0]))] += 1
            if len(samples) < 6:
                samples.append((sim, turn, t, do_map[t][0], dont_map[t][0]))
        if hit:
            conflict_turns += 1
            conflict_sims.add(sim)

    print("== 결과 (완주분) ==")
    print("  우리 문구가 나간 턴            %d" % turns_with_text)
    print("  DO 행과 DONT 행이 함께 온 턴      %d" % both_turns)
    print("  ★그중 **같은 도구**를 가리킨 턴    **%d** (%.0f%%) · sim %d개"
          % (conflict_turns, 100.0 * conflict_turns / max(1, both_turns), len(conflict_sims)))
    print("  (우리 문구 턴 대비 %.0f%%)" % (100.0 * conflict_turns / max(1, turns_with_text)))
    print("\n  표적별:")
    for t, n in tgt_count.most_common(10):
        print("    %-42s %3d턴" % (t, n))
    print("\n  레버 쌍(DO ← → DONT) 상위:")
    for (a, b), n in pair_count.most_common(10):
        print("    %-24s ↔ %-24s %3d" % (a, b, n))
    print("\n  축자 표본:")
    for sim, turn, t, a, b in samples:
        print("    [%s turn %s] %s" % (sim, turn, t))
        print("      DO   [%s] %s" % (lever_of(a), re.sub(r"\s+", " ", (a.get("text") or ""))[:150]))
        print("      DONT [%s] %s" % (lever_of(b), re.sub(r"\s+", " ", (b.get("text") or ""))[:150]))


if __name__ == "__main__":
    main()
