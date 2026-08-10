# -*- coding: utf-8 -*-
r"""x217 — **무엇이 읽기를 방해하는가**: 여러 거부 사례에서 격리→블록 성장 (유료 0 · 엔진 0).

## 가설 (사용자 2026-08-10)

> *"이전 프롬프트에서 읽기를 방해하는 steer 가 있을 것 같다. 지시나 이런 걸 격리하면 읽기도 될
>  것 같다. 어떤 게 방해하는 원인인지 추적해 보라."*
> *"실제로 안 읽은 경우가 많지 않나? 그 경우 몇 개를 더 해서 확실히. 격리에서 늘려가면서."*

## 이미 나온 것 (x216·C398)

  ISO      깨끗한 문맥 + 요구        **7/8** 호출
  ISO_NO   깨끗한 문맥 + 요구 없음   **0/8** (대신 `KB_search_bm25` 6/8 — **기본 prior 는 검색**)
  라이브   요구 12회 + 이름 변환 12회 → **0회**

⇒ prior 는 실재하되 **깨끗한 문맥에서는 요구가 그것을 뒤집는다**. 궤적에서 뒤집기가 죽는다.

## 설계 — 요구·도구 스키마는 **완전히 고정**하고 문맥만 바꾼다

§1 성분 절제 (무엇이 방해하나)

  T_FULL      실제 거부 궤적 + 요구
  T_noKB      − KB 검색 결과 메시지
  T_noOURS    − **우리가 주입한 문장**
  T_noASSIST  − 이전 어시스턴트 발화(자기-정박)
  T_noUSER    − 손님의 이관 요청·거절 발화
  ISO         깨끗 + 요구                 ← 천장
  ISO_LONG    깨끗 + 무관한 텍스트 패딩(같은 길이) + 요구   ← **길이 통제**
  ISO_NO      깨끗 + 요구 없음            ← prior 기본값(부정 통제)

§2 블록 성장 (얼마나 얹으면 죽나) — **깨끗한 격리에서 시작해** 궤적 블록을 뒤에서부터 얹는다.

⚠**자기적발 두 번**: ⒜ 팔이 무엇을 담았는지 인쇄하지 않고 재면 안 된다(오늘 프로브 결함 3회의
  공통 원인). ⒝ 1차에서 *"안 읽은 sim"* 만 조건으로 걸었더니 KB 0·우리 주입 0 인 짧은 궤적이
  뽑혀 `T_FULL` 이 10/10 이었다 — **방해가 없는 문맥에는 잴 것이 없다**. 이제 혼잡 실재를 조건에
  넣고 **여러 사례**를 돈다.

실행: python x217_read_steer_trace.py [N] [--cases K]
"""
import collections
import glob
import gzip
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from x216_read_and_offset import TOOLS, DEMAND, SAID, chat, called_target   # noqa: E402

PATS = ["/home/woori/scratch/tau2-bench/data/simulations/*/results.json",
        "/home/woori/workspace_common/boltzmann-attention-pi/reports/facet_rft_2026/"
        "sim_results/*.json.gz"]
OURS = ("[T2_", "Policy constants on record", "grouped by the status each record carries",
        "Date arithmetic on the records above", "[COMPUTED FACTS]",
        "was NOT checked against any allowance", "no room left this year")
PAD = ("The bank's lobby hours are unchanged this quarter. Parking validation is offered at the "
       "garage next door. The quarterly newsletter is mailed to registered addresses. ")


def pick_cases():
    """**실제 거부 사례**들 — 계좌를 안 읽었고 **혼잡이 실재**하는 궤적, 혼잡 순."""
    cands = []
    for pat in PATS:
        for p in sorted(glob.glob(pat)):
            try:
                f = gzip.open(p, "rt", encoding="utf-8") if p.endswith(".gz") \
                    else open(p, encoding="utf-8")
                d = json.load(f)
            except Exception:
                continue
            if not isinstance(d, dict):
                continue
            for s in d.get("simulations") or []:
                if not isinstance(s, dict) or s.get("task_id") not in ("task_099", "task_100"):
                    continue
                if (s.get("reward_info") or {}).get("reward") == 1:
                    continue
                msgs = s.get("messages") or []
                blob = "\n".join(str(m.get("content") or "") for m in msgs)
                calls = " ".join(str((tc.get("function") or tc).get("name") or "")
                                 + str((tc.get("function") or tc).get("arguments") or "")
                                 for m in msgs for tc in (m.get("tool_calls") or []))
                if "get_all_user_accounts" in calls:
                    continue                    # 읽은 sim 은 재현 대상이 아니다
                kb = blob.count("Score:")
                ours = sum(blob.count(sig) for sig in OURS)
                if len(msgs) < 8 or (kb == 0 and ours == 0):
                    continue                    # 혼잡이 없으면 잴 것이 없다
                cands.append((kb + ours, kb, ours, len(msgs), len(blob),
                              os.path.basename(os.path.dirname(p)), s.get("trial"), msgs))
    cands.sort(key=lambda x: -x[0])
    print("거부 사례 후보 %d개 (혼잡 순):" % len(cands))
    for c in cands[:8]:
        print("   KB %-3d 우리 %-3d 메시지 %-3d %7d자  %s trial=%s"
              % (c[1], c[2], c[3], c[4], c[5], c[6]))
    return cands


def render(msgs, drop=()):
    parts = []
    for m in msgs:
        role = m.get("role")
        raw = str(m.get("content") or "")
        c = " ".join(raw.split())
        if "kb" in drop and role == "tool" and "Score:" in raw:
            continue
        if "ours" in drop and any(sig in raw for sig in OURS):
            continue
        if "assist" in drop and role == "assistant":
            continue
        if "user" in drop and role == "user" and re.search(
                r"transfer|human agent|i (can'?t|don'?t|won'?t)|not comfortable", c, re.I):
            continue
        for tc in (m.get("tool_calls") or []):
            fn = tc.get("function") or tc
            parts.append("[%s calls %s]" % (role, fn.get("name")))
        if c:
            parts.append("[%s] %s" % (role, c))
    return "\n".join(parts)


def kind_of(b):
    if any(x in b for x in OURS):
        return "우리주입"
    if "Score:" in b:
        return "KB결과"
    if b.startswith("[tool]"):
        return "도구출력"
    if b.startswith("[assistant"):
        return "어시스턴트"
    if b.startswith("[user"):
        return "손님"
    return "기타"


def measure(body, n, with_demand=True):
    c = collections.Counter()
    for i in range(n):
        p = body + (("\n\n" + DEMAND) if with_demand else "") + "\n\nWhat do you do next?"
        try:
            m = chat(p, TOOLS, 0.0 if i == 0 else 0.7)
        except Exception:
            c["ERR"] += 1
            continue
        c["호출O" if called_target(m) else "호출X"] += 1
        for tc in (m.get("tool_calls") or []):
            c["도구:" + str((tc.get("function") or {}).get("name"))] += 1
    return c


def run_case(tag, trial, msgs, n):
    full = render(msgs)
    out = {}
    arms = [("T_FULL", full, True),
            ("T_noKB", render(msgs, ("kb",)), True),
            ("T_noOURS", render(msgs, ("ours",)), True),
            ("T_noASSIST", render(msgs, ("assist",)), True),
            ("T_noUSER", render(msgs, ("user",)), True),
            ("ISO", SAID["task_099"], True),
            ("ISO_LONG", SAID["task_099"] + "\n\n" + PAD * max(1, len(full) // len(PAD)), True),
            ("ISO_NO", SAID["task_099"], False)]
    print("\n" + "=" * 96)
    print("%s trial=%s · 메시지 %d · %d자 · n=%d" % (tag, trial, len(msgs), len(full), n))
    print("  %-11s %8s %6s %6s %6s %6s" % ("팔", "자수", "KB", "우리", "어시", "손님"))
    for name, body, _wd in arms:
        print("  %-11s %8d %6d %6d %6d %6d"
              % (name, len(body), body.count("Score:"), sum(body.count(x) for x in OURS),
                 body.count("[assistant]"), body.count("[user]")))
    print("\n§1 성분 절제")
    for name, body, wd in arms:
        c = measure(body, n, wd)
        out["%s/%s/%s" % (tag, trial, name)] = [c["호출O"], n]
        print("  %-11s 읽기 %2d/%d   %s"
              % (name, c["호출O"], n,
                 [x for x in c.most_common(3) if str(x[0]).startswith("도구")]))

    print("\n§2 깨끗한 격리에서 블록 늘리기")
    blocks = []
    for m in msgs:
        role = m.get("role")
        c0 = " ".join(str(m.get("content") or "").split())
        piece = []
        for tc in (m.get("tool_calls") or []):
            fn = tc.get("function") or tc
            piece.append("[%s calls %s]" % (role, fn.get("name")))
        if c0:
            piece.append("[%s] %s" % (role, c0))
        if piece:
            blocks.append("\n".join(piece))
    base = SAID["task_099"]
    depths = [d for d in sorted(set([0, 1, 2, 3, 4, 6, 8, 12, 16, 24, len(blocks)]))
              if d <= len(blocks)]
    prev = None
    for k in depths:
        body = base if k == 0 else ("\n".join(blocks[-k:]) + "\n\n" + base)
        c = measure(body, n, True)
        out["%s/%s/grow-%02d" % (tag, trial, k)] = [c["호출O"], n]
        top = "(깨끗)" if k == 0 else "%s: %s" % (kind_of(blocks[-k]),
                                                 blocks[-k].split("]")[0][:20] + "]")
        mark = "  ← **꺾임**" if prev is not None and c["호출O"] <= prev - max(2, n // 3) else ""
        prev = c["호출O"]
        print("  +%2d블록 %7d자 읽기 %2d/%d  맨위=%-30s%s" % (k, len(body), c["호출O"], n,
                                                              top[:30], mark))
    return out


def main():
    n, ncase = 8, 3
    argv = sys.argv[1:]
    for i, a in enumerate(argv):
        if a.isdigit():
            n = int(a)
        if a.startswith("--cases"):
            ncase = int(a.split("=", 1)[-1] if "=" in a else argv[i + 1])
    cands = pick_cases()
    if not cands:
        print("⚠혼잡이 실재하는 거부 사례를 못 찾았다.")
        return 1
    allout = {}
    for c in cands[:ncase]:
        allout.update(run_case(c[5], c[6], c[7], n))
    json.dump(allout, open(os.environ.get("T2_X217_OUT", "x217_out.json"), "w"), indent=1)
    print("\n※ ISO 높고 T_FULL 낮은데 ISO_LONG 이 높으면 길이가 아니다.")
    print("  여러 사례에서 **같은 종류의 블록**에서 꺾이면 그것이 기전이다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
