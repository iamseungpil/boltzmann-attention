# -*- coding: utf-8 -*-
"""x137 — `split`의 **천장**을 독립 기준으로 잰다 (유료 0 · LLM 호출 0 · 로컬 실행).

왜: x135의 오라클은 **`split`을 두 번 돌려 합친 것**이라, `split` 1패스가 100%인 것은
*"코퍼스에 있는 걸 다 찾았다"* 가 아니라 *"두 번 돌려도 새로 나오는 게 없다"*(자기 일관성)에
가깝다. `split`이 **매번 똑같이 놓치는 것**이 있으면 오라클에도 없어서 영원히 100%로 보인다.
⇒ **`split`과 무관한 방식**으로 후보를 뽑아 대조해야 천장이 보인다.

방법(의도적으로 단순·기계적):
  1. 각 sim이 **실제로 회수한 텍스트**에서 *"…N days"* 형태의 **기간 진술 문장**을 정규식으로 전수 수집
     (KB 전체가 아니라 회수분 — 에이전트가 본 적 없는 것을 세면 비교가 틀린다).
  2. x135 오라클의 **인용문**이 그 문장을 덮는지 본다(부분문자열 양방향).
  3. **안 덮인 문장**을 목록으로 낸다 ⇒ 사람이 per-case로 읽어 *진짜 누락*과 *무관 진술*을 가른다.

⚠이것은 **분석 도구**이지 엔진이 아니다([[59]]는 **엔진**이 도메인 텍스트를 뜯는 것을 금지한다).
여기서 나온 어떤 문자열도 엔진·A2로 들어가지 않는다.
⚠정규식은 **내가 고른 것**이라 그 자체가 한계다 — 다만 `split`과 **독립**이라는 점이 요점이다.
이 프로브가 후보를 0건 내면 *"천장이 없다"* 가 아니라 *"이 패턴으로는 못 봤다"* 이다.

usage: x137_split_ceiling_probe.py [--axis threshold]
"""

import argparse
import gzip
import io
import json
import os
import re
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
SIMS = os.path.join(REPO, "reports", "facet_rft_2026", "sim_results")
DIRS = {"dp": "bank_stack_dp_20260808p", "lim": "bank_stack_lim_20260808n"}

# 기간 진술 후보 — 넓게 잡고 per-case로 거른다(좁게 잡으면 천장을 못 본다)
PAT = re.compile(
    r"[^.\n|]{0,120}?(?:at least|minimum|minimums|no less than|maintained|for)\s"
    r"[^.\n|]{0,60}?\b(\d{1,3})\s*days[^.\n|]{0,60}", re.I)
PAT_TABLE = re.compile(r"\|[^|\n]{0,60}\|\s*(\d{1,3})\s*days\s*\|", re.I)


def _norm(s):
    return " ".join(str(s).split())


def sim_texts(dirname, task, idx):
    d = json.load(gzip.open(os.path.join(SIMS, dirname + ".json.gz"), "rt",
                            encoding="utf-8", errors="replace"))
    sims = [s for s in (d.get("simulations") or []) if s.get("task_id") == task]
    out = []
    for m in sims[idx].get("messages") or []:
        if m.get("role") not in ("tool", "user"):
            continue
        c = m.get("content")
        if isinstance(c, list):
            c = "\n".join(str(x) for x in c)
        out.append(str(c or ""))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--axis", default="threshold")
    a = ap.parse_args()

    res = json.load(gzip.open(os.path.join(
        SIMS, "x135_excerpt_arms_%s.json.gz" % a.axis), "rt", encoding="utf-8"))

    tot_cand = tot_cov = 0
    for rec in res["sims"]:
        short, task, idx = rec["spec"].split(":")
        texts = sim_texts(DIRS[short], task, int(idx))
        hay = "\n".join(texts)

        cands = set()
        for m in PAT.finditer(hay):
            cands.add(_norm(m.group(0)))
        for m in PAT_TABLE.finditer(hay):
            cands.add(_norm(m.group(0)))

        # 오라클이 낸 **인용문** 전량 — 이것이 split이 실제로 근거로 댄 문장이다
        quotes = []
        for arm in ("split",):
            for r in rec["arms"][arm]:
                pass
        # 인용문은 arms에 저장돼 있지 않으므로 오라클 쌍의 키·값으로는 못 덮는다 ⇒
        # split이 낸 쌍의 **수**가 문장에 등장하는지로 덮음을 판정한다(보수적: 덮었다고 보기 쉬움)
        nums = set()
        for p in rec["oracle"]["pairs"]:
            try:
                nums.add(int(p.rsplit("=", 1)[1]))
            except Exception:
                pass

        uncovered = [c for c in sorted(cands)
                     if not any(re.search(r"\b%d\s*days" % n, c, re.I) for n in nums)]
        tot_cand += len(cands)
        tot_cov += len(cands) - len(uncovered)
        print("\n== %s · 회수 텍스트 %d건 · 기간 진술 후보 %d · 오라클 수치로 덮인 %d"
              % (rec["spec"], len(texts), len(cands), len(cands) - len(uncovered)))
        print("   오라클 쌍 %d: %s" % (len(rec["oracle"]["pairs"]),
                                       ", ".join(rec["oracle"]["pairs"][:6])))
        for u in uncovered[:12]:
            print("   ☐ %s" % u[:150])
        if len(uncovered) > 12:
            print("   … %d건 더" % (len(uncovered) - 12))

    print("\n" + "=" * 90)
    print("합계 · 기간 진술 후보 %d · 오라클 수치로 덮인 %d (%.0f%%) · 안 덮인 %d"
          % (tot_cand, tot_cov, 100.0 * tot_cov / tot_cand if tot_cand else 0,
             tot_cand - tot_cov))
    print("⚠'안 덮인'이 곧 누락은 아니다 — **per-case로 읽어야** 진짜 누락과 무관 진술이 갈린다"
          "(추천받는 사람의 지출 기한 등은 관계기간 문턱이 아니다).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
