# -*- coding: utf-8 -*-
r"""x393 — **정책 문서가 궤적에 닿았나**: 빠진 gold 액션마다 *그 절차를 적어 둔 문서*가 배달됐는지 센다.

## 왜 (2026-08-19 사용자 지시: *"정책에 없다는 게 말이 안 된다 — 정책을 다시 정밀하게 보라"*)

`x392` 는 빠진 gold 를 NEVER/ARGDIFF 로 갈랐지만 *왜 안 불렀나* 는 두 갈래다:
  ⒜ **절차가 정책에 없다** → A2/A3 저작이 gold 경유가 되어 금지([[23]])
  ⒝ **절차는 정책에 있는데 그 문서가 궤적에 안 왔다** → 결손은 *지식*이 아니라 **회수·전달**
축자 확인 결과 ⒜ 는 대부분 거짓이다 — 예: `apply_checking_account_credit_5829` 문서에
*"Check the transaction history to confirm each fee discrepancy"* 와 6단계 `## Procedure` 가 있고,
`submit_interest_discrepancy_report_7294` 문서에는 *"3) Check transaction history using
get_bank_account_transactions_9173 …"* 가 **도구 이름째** 적혀 있다. ⇒ 남는 물음은 ⒝ 이고, 그건 센다.

## 무엇을 세나 (결정론·LLM 0·판단 0)

1. 정책 문서 698개를 읽어 **도구 이름 → 그 이름을 적은 문서** 색인을 만든다(문자열 포함만).
2. 각 sim 의 KB 회수 응답에서 `ID: doc_…` 를 뽑아 **배달된 문서 집합**을 만든다.
3. 빠진 gold 액션마다 그 지배 문서가 **배달됐는가 / 언제**를 붙인다.

⚠이것은 *도달* 만 잰다. 배달됐는데도 안 따랐다면 그건 다른 결손이다(그 갈래도 표에 남는다).

사용: py -3 x393_policy_reach.py <tag> [<tag> …]
"""
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

import t2_forensic as F  # noqa: E402

SUF = ".results.json.gz"
DOCDIR = "/home/woori/scratch/tau2-bench/data/tau2/domains/banking_knowledge/documents"
TASKS = "/home/woori/scratch/tau2-bench/data/tau2/domains/banking_knowledge/tasks.json"
DOCID_RE = re.compile(r"\bID:\s*(doc_[A-Za-z0-9_()\-]+)")
STEP_RE = re.compile(r"(?im)^\s*(?:\d+[).]|[-*])\s*(.{15,})$")


def load_docs():
    out = {}
    for fn in sorted(os.listdir(DOCDIR)):
        if not fn.endswith(".json"):
            continue
        try:
            d = json.load(io.open(os.path.join(DOCDIR, fn), encoding="utf-8", errors="replace"))
        except Exception:
            continue
        out[str(d.get("id") or fn[:-5])] = (str(d.get("title") or ""), str(d.get("content") or ""))
    return out


def required_docs():
    """태스크가 선언한 `required_documents` — **env 메타데이터**(정답 액션이 아니다).

    ⚠계측용으로만 쓴다. 런타임 레버가 이걸 읽으면 *문서 정답지*를 떠먹이는 것이라 [[23]] 위반이다.
    """
    try:
        d = json.load(io.open(TASKS, encoding="utf-8", errors="replace"))
    except Exception:
        return {}
    return {str(t.get("id")): list(t.get("required_documents") or []) for t in d}


def main():
    tags = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not tags:
        print("usage: x393_policy_reach.py <tag> ...")
        return 2
    docs = load_docs()
    print("정책 문서 %d개 로드" % len(docs))

    j = os.path.normpath(os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026",
                                      "x392_block_join.json"))
    ends = json.load(io.open(j, encoding="utf-8"))["ends"] if os.path.exists(j) else []
    end_by = {(e["task"], str(e["trial"]), e["name"]): e for e in ends}

    # 도구 이름 → 그 이름을 적은 문서
    names = sorted({e["name"] for e in ends})
    owner = {n: [k for k, (_t, c) in docs.items() if n in c] for n in names}

    print("\n## §A gold 도구를 **이름으로 적은** 정책 문서")
    print("%-46s %s" % ("gold 도구", "문서(제목)"))
    for n in names:
        ds = owner.get(n) or []
        print("%-46s %s" % (n[:46], " · ".join(docs[d][0][:44] for d in ds[:2]) or "★없음"))

    rows = []
    for tag in tags:
        for sim in F.scored(tag, SUF):
            task, trial = F.task_id(sim), str(sim.get("trial"))
            rw = (sim.get("reward_info") or {}).get("reward")
            delivered = {}          # doc_id -> 첫 배달 turn
            for i, m in enumerate(sim.get("messages") or []):
                if m.get("role") != "tool":
                    continue
                t = m.get("turn_idx", i)
                try:
                    t = int(t)
                except Exception:
                    t = i
                for d in DOCID_RE.findall(str(m.get("content") or "")):
                    delivered.setdefault(d, t)
            for key, e in end_by.items():
                if key[0] != task or key[1] != trial or e["tag"] != tag:
                    continue
                ds = owner.get(e["name"]) or []
                hit = [d for d in ds if d in delivered]
                rows.append({"task": task, "trial": trial, "reward": rw, "name": e["name"],
                             "code": e["code"], "docs": len(ds), "hit": len(hit),
                             "turn": min([delivered[d] for d in hit]) if hit else None,
                             "ndel": len(delivered)})

    req = required_docs()
    print("\n## §A-2 태스크 `required_documents` 배달률 (env 메타·계측 전용)")
    print("%-9s %-3s %-5s %-6s %s" % ("task", "tr", "rw", "필수", "배달/필수 · 미배달"))
    for tag in tags:
        for sim in F.scored(tag, SUF):
            task, trial = F.task_id(sim), str(sim.get("trial"))
            rw = (sim.get("reward_info") or {}).get("reward")
            dl = set()
            for m in (sim.get("messages") or []):
                if m.get("role") == "tool":
                    dl |= set(DOCID_RE.findall(str(m.get("content") or "")))
            rq = req.get(task) or []
            miss = [d for d in rq if d not in dl]
            print("%-9s %-3s %-5s %-6d %d/%d · %s"
                  % (task, trial, rw, len(rq), len(rq) - len(miss), len(rq),
                     " ".join(x.replace("doc_", "")[:32] for x in miss[:3]) or "(전부 배달)"))

    print("\n## §B-0 **문서 id 가 아니라 절차 문장이 왔나** (bm25 는 조각을 준다)")
    # 지배 문서에서 *그 도구를 지시하는 절차 줄*만 뽑아, 그 줄의 앞머리가 궤적 본문에 있는지 본다.
    sent = collections.Counter()
    detail_sent = []
    for tag in tags:
        for sim in F.scored(tag, SUF):
            task, trial = F.task_id(sim), str(sim.get("trial"))
            body = "\n".join(str(m.get("content") or "") for m in (sim.get("messages") or [])
                             if m.get("role") == "tool")
            body = " ".join(body.split())
            for key, e in end_by.items():
                if key[0] != task or key[1] != trial or e["tag"] != tag:
                    continue
                lines = []
                for d in (owner.get(e["name"]) or []):
                    for ln in STEP_RE.findall(docs[d][1]):
                        s = " ".join(ln.split())
                        if e["name"] in s or re.search(r"(?i)\b(before|must|first|check|review)\b", s):
                            lines.append(s)
                if not lines:
                    sent[(e["code"], "절차줄없음")] += 1
                    continue
                hit = [s for s in lines if s[:55] in body]
                # ⚠도달한 줄이 *그 도구를 지시하는* 줄인지 가른다 — 아니면 "정책이 왔다"가 과대주장이 된다.
                strong = [s for s in hit if e["name"] in s
                          or re.match(r"(?i)\s*(before|first)\b", s)]
                sent[(e["code"], ("문장도달-강" if strong else "문장도달-약") if hit
                      else "문장미도달")] += 1
                hit = strong or hit
                detail_sent.append({"task": task, "trial": trial, "name": e["name"],
                                    "code": e["code"], "reward": e["reward"],
                                    "hit": bool(hit), "n": len(lines),
                                    "ex": (hit[0] if hit else lines[0])[:90]})
    ks2 = ["문장도달-강", "문장도달-약", "문장미도달", "절차줄없음"]
    print("%-12s %s" % ("", " ".join("%-11s" % k for k in ks2)))
    for code in ("MATCH", "NEVER", "ARGDIFF", "OURS", "ENV_REJECT"):
        if sum(sent[(code, k)] for k in ks2):
            print("%-12s %s" % (code, " ".join("%-11d" % sent[(code, k)] for k in ks2)))

    print("\n## §B-1 실패 sim 에서 **절차 문장이 실제로 왔는데도** 빠진 gold")
    for d in sorted(detail_sent, key=lambda x: (x["task"], x["trial"])):
        if d["code"] in ("MATCH",) or (d["reward"] or 0) >= 1.0 or not d["hit"]:
            continue
        print("  %-9s %-3s %-40s %-9s ← %s" % (d["task"], d["trial"], d["name"][:40],
                                               d["code"], d["ex"]))

    print("\n## §B 결말 × 지배 문서 배달 여부")
    agg = collections.Counter()
    for r in rows:
        k = "문서없음" if not r["docs"] else ("배달됨" if r["hit"] else "미배달")
        agg[(r["code"], k)] += 1
    ks = ["배달됨", "미배달", "문서없음"]
    print("%-12s %s" % ("", " ".join("%-8s" % k for k in ks)))
    for code in ("MATCH", "NEVER", "ARGDIFF", "OURS", "ENV_REJECT", "NOTATION"):
        if sum(agg[(code, k)] for k in ks):
            print("%-12s %s" % (code, " ".join("%-8d" % agg[(code, k)] for k in ks)))

    print("\n## §C 미매치 gold 전량 — 지배 문서가 왔나")
    print("%-9s %-3s %-44s %-10s %-8s %s" % ("task", "tr", "gold 도구", "결말", "문서배달", "배달turn"))
    for r in sorted(rows, key=lambda x: (x["task"], x["trial"])):
        if r["code"] == "MATCH" or (r["reward"] or 0) >= 1.0:
            continue
        st = "문서없음" if not r["docs"] else ("O" if r["hit"] else "X")
        print("%-9s %-3s %-44s %-10s %-8s %s"
              % (r["task"], r["trial"], r["name"][:44], r["code"], st,
                 r["turn"] if r["turn"] is not None else "-"))

    out = os.path.normpath(os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026",
                                        "x393_policy_reach.json"))
    io.open(out, "w", encoding="utf-8").write(json.dumps(rows, ensure_ascii=False, indent=1))
    print("\n원자료: %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
