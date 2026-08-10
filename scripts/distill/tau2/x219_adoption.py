# -*- coding: utf-8 -*-
r"""x219 — **전달된 정답이 왜 채택되지 않는가** (격리 A/B · 유료 0 · 엔진 0).

## 왜 (부검 C399)

`bank_alllevers_20260810` 사이드카 실측 — 우리 결정 블록은 **예외 없이 정답을 담아 나갔다**:
`Blue`(098 gold) ×3 · `World Blue`(099) ×3 · `Hunter Green`(100) ×2 ·
`Platinum Rewards Card — the user has too many referral processes going on`(010) ×4.

그런데 098 은 3/4 실패하며 `Light Blue` 를, 099 는 `Navy Blue` 를 제출했다. 즉 **답 자체가
문맥에 있는데 궤적이 이긴다.** 서브를 더 잘 만드는 문제가 아니라 **채택** 문제다.

## 팔 (실제 실패 궤적 + 실제 블록 문구 · 손님의 실제 질문 고정)

  A_ASIS      궤적 그대로 + 블록(원래 자리)             ← 라이브 재현
  B_LAST      같은 것, 블록을 **맨 끝**(질문 직전)으로
  C_NOKB      KB 검색 결과 제거 + 블록                   ← 경쟁 스니펫 제거
  D_ONLY      **블록 + 질문만** (혼잡 0)                 ← 천장
  E_NOBLOCK   궤적 그대로, **블록 없음**                 ← 부정 통제(오답이 나와야 한다)
  F_TRIM      궤적에서 **KB 스니펫 본문만** 잘라내고 제목/ID 는 남김 + 블록

⚠어느 팔에서도 우리가 답을 강요하지 않는다 — 블록은 라이브에서 나간 **그 문구 그대로**다.
⚠팔마다 무엇이 담겼는지(블록 유무·KB 수·자수) 먼저 인쇄하고 잰다.

채점: gold 이름을 **정확히** 말하는가(부분 문자열 금지 — `Blue` 가 `Light Blue` 에 걸린다).

실행: python x219_adoption.py [N]
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

from x216_read_and_offset import chat                            # noqa: E402

PATS = ["/home/woori/scratch/tau2-bench/data/simulations/*/results.json",
        "/home/woori/workspace_common/boltzmann-attention-pi/reports/facet_rft_2026/"
        "sim_results/*.json.gz"]
FBS = "/home/woori/scratch/logs/fb_bank_alllevers_20260810.jsonl"
BLOCK_SIG = "A separate check was run on the policy constants"
GOLD = {"task_098": "Blue", "task_099": "World Blue", "task_100": "Hunter Green"}
ASK = ("Which account type should I use for the referral? Answer with the account name only.")


def live_blocks():
    """라이브에서 **실제로 나간** 결정 블록 문구를 답별로 모은다(우리가 짓지 않는다)."""
    out = {}
    if not os.path.exists(FBS):
        return out
    for ln in open(FBS, encoding="utf-8", errors="replace"):
        try:
            o = json.loads(ln)
        except Exception:
            continue
        t = o.get("text") or ""
        if BLOCK_SIG not in t:
            continue
        m = re.search(r"It answers: ([^.\n]+)", t)
        if m:
            out.setdefault(m.group(1).strip(), t.strip())
    return out


def pick(task, want_wrong):
    """그 태스크의 **실패** sim 중 가장 혼잡한 것."""
    best = None
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
                if not isinstance(s, dict) or s.get("task_id") != task:
                    continue
                if (s.get("reward_info") or {}).get("reward") == 1:
                    continue
                msgs = s.get("messages") or []
                blob = "\n".join(str(m.get("content") or "") for m in msgs)
                kb = blob.count("Score:")
                if kb == 0 or len(msgs) < 8:
                    continue
                if best is None or kb > best[0]:
                    best = (kb, os.path.basename(os.path.dirname(p)), s.get("trial"), msgs)
    return best


def render(msgs, mode="full"):
    parts = []
    for m in msgs:
        role = m.get("role")
        raw = str(m.get("content") or "")
        c = " ".join(raw.split())
        if role == "tool" and "Score:" in raw:
            if mode == "nokb":
                continue
            if mode == "trim":
                # 제목·ID 는 남기고 **본문만** 자른다 — 경쟁 스니펫의 내용만 없앤다
                c = " ".join(re.sub(r"Content:.*?(?=(?:\d+\.\s)|$)", "Content: [trimmed] ",
                                    raw, flags=re.S).split())
        for tc in (m.get("tool_calls") or []):
            fn = tc.get("function") or tc
            parts.append("[%s calls %s]" % (role, fn.get("name")))
        if c:
            parts.append("[%s] %s" % (role, c))
    return "\n".join(parts)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 8
    blocks = live_blocks()
    print("라이브 결정 블록 %d종: %s" % (len(blocks), sorted(blocks)))
    out = {}
    for task in ("task_098", "task_099"):
        gold = GOLD[task]
        blk = blocks.get(gold)
        if not blk:
            print("\n%s — 그 답의 라이브 블록 문구를 못 찾았다. 건너뛴다." % task)
            continue
        got = pick(task, gold)
        if not got:
            print("\n%s — 혼잡한 실패 sim 을 못 찾았다." % task)
            continue
        kb, tag, trial, msgs = got
        full = render(msgs)
        arms = [("A_ASIS", full + "\n\n" + blk),
                ("B_LAST", full + "\n\n" + blk),          # 아래에서 위치만 다르게 조립
                ("C_NOKB", render(msgs, "nokb") + "\n\n" + blk),
                ("D_ONLY", blk),
                ("E_NOBLOCK", full),
                ("F_TRIM", render(msgs, "trim") + "\n\n" + blk)]
        print("\n" + "=" * 96)
        print("%s  %s trial=%s · KB %d · %d자 · gold=%r · n=%d"
              % (task, tag, trial, kb, len(full), gold, n))
        for name, body in arms:
            print("   %-10s 블록 %s · KB %2d · %7d자"
                  % (name, "O" if BLOCK_SIG in body else "X", body.count("Score:"), len(body)))
        for name, body in arms:
            c = collections.Counter()
            for i in range(n):
                # A_ASIS 는 블록을 **중간**(마지막 손님 발화 앞)에 둔다 — 위치 효과를 가른다
                if name == "A_ASIS":
                    cut = body.rfind("\n[user]")
                    body2 = (body[:cut] + "\n" + blk + body[cut:]) if cut > 0 else body
                    body2 = body2.replace("\n\n" + blk, "", 1)
                else:
                    body2 = body
                p = body2 + "\n\n" + ASK
                try:
                    t = chat(p, None, 0.0 if i == 0 else 0.7, 24).get("content", "")
                except Exception as e:
                    t = "ERR %s" % type(e).__name__
                c[" ".join(str(t).split())[:40]] += 1
            hit = sum(v for k, v in c.items()
                      if re.fullmatch(r"\**%s( Account)?\**\.?" % re.escape(gold),
                                      str(k).strip(), re.I))
            out["%s/%s" % (task, name)] = [hit, n]
            print("  %-10s gold %d/%d   %s" % (name, hit, n, c.most_common(2)))
    json.dump(out, open(os.environ.get("T2_X219_OUT", "x219_out.json"), "w"), indent=1)
    print("\n※ E_NOBLOCK 이 낮고 D_ONLY 가 높은 것은 당연하다 — 가르는 것은 **A/B/C/F** 다."
          "\n  B_LAST 가 A_ASIS 보다 높으면 **위치**, C_NOKB·F_TRIM 이 높으면 **경쟁 스니펫**이 범인이다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
