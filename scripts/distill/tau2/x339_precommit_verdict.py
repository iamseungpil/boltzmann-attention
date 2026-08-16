# -*- coding: utf-8 -*-
r"""x339 — **t7303(커밋-이전 전달) 판정**. 순서는 런처 주석에 **사전 고정**된 그대로다.

    ⓐ배선   treat 에서 `[T2_DELIVER_PRECOMMIT] 선-배달` > 0 · ctl **0** · infra 0.
             실패면 성적을 안 읽는다.
    ⓑ**1차 종점 = 첫 지목 이전 도달 sim 비율**. 배달 지점은 로그의 `turn=N` 이고 이 N 은
             엔진 축자로 `len(state.messages)`(메시지 개수) — 그래서 `first_named()` 가 주는
             **메시지 index 와 같은 좌표계**다. 도달 = `deliver_at <= first_named_idx`.
    ⓒ선택   후보 이름 중 무엇을 **처음** 말했나 / 마지막까지 무엇을 말했나.
    ⓓ성적   `reward` 만(C486: `action_match` 는 소수점 표기로 오탐).
    ⓔ부작용 지연(duration) · 군 오선택(배달된 문서군) · over-action(write 호출 수) ·
             **098 불변**.

⚠n=12/팔 · 잡음 바닥 ±4 ⇒ **ⓒⓓ 차이는 인용 금지**([[57]]·런처 주석). 이 런은 ⓑ 를 사러 갔다.
⚠후보 이름 어휘는 **env 에서** 뽑는다(문서군 파일명 + 레코드의 class/type 값) — gold 무참조
  ([[23]]). 엔진이 아니라 **분석**이 쓰는 어휘다.
⚠폐기 태그 `…20260816f`(a2 언바인드) · `…20260816g`(서브의 *결정*을 배달)는 손대지 않는다.

실행(리모트):
    /home/woori/venvs/seka_env/bin/python x339_precommit_verdict.py
"""
import collections
import io
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import t2_forensic as F                                            # noqa: E402

ARMS = [("ctl", "bank_t7303_ctl_20260816h"), ("treat", "bank_t7303_treat_20260816h")]
DOM = "/home/woori/scratch/tau2-bench/data/tau2/domains/banking_knowledge"
DELIVER = r"T2_DELIVER_PRECOMMIT\] 선-배달 turn=(\d+)"
SEARCHAG = r"T2_SEARCH_AGENT\] 축 처리 완료: (\S+)"
WRITEISH = ("apply_for_credit_card", "open_", "submit_", "deposit_", "transfer_",
            "close_", "approve_", "deny_", "update_", "log_verification")


def candidate_names():
    """후보 이름 어휘 = **env 유래**. ⑴문서군 파일명 → 사람이 읽는 이름 ⑵레코드의 class/type 값."""
    names = set()
    docs = os.path.join(DOM, "documents")
    if os.path.isdir(docs):
        for fn in os.listdir(docs):
            stem = re.sub(r"[._]\d+$", "", os.path.splitext(fn)[0])
            stem = re.sub(r"_\d+$", "", stem)
            if len(stem) > 6 and "_" in stem:
                names.add(" ".join(w.capitalize() for w in stem.split("_")))
    try:
        db = json.load(io.open(os.path.join(DOM, "db.json"), encoding="utf-8"))
    except Exception:
        db = {}
    for tbl in db.values():
        rows = tbl.values() if isinstance(tbl, dict) else (tbl if isinstance(tbl, list) else [])
        for r in rows:
            if not isinstance(r, dict):
                continue
            for k in ("account_class", "credit_card_type", "card_type", "account_type"):
                v = r.get(k)
                if isinstance(v, str) and len(v) > 4:
                    names.add(v)
    # 짧은 이름이 긴 이름의 부분열이면(예: 'Bronze Rewards Card' ⊂ 'Business Bronze …')
    # 긴 것부터 찾도록 정렬만 해 둔다(판정은 첫 매치 index 이므로 집합 그대로 쓴다).
    return sorted(names, key=len, reverse=True)


def named_first(sim, names):
    """첫 지목 (index, 이름). `t2_forensic.first_named` 와 같은 규칙 + 어느 이름인지 보고."""
    idx = F.first_named(sim, names)
    if idx is None:
        return None, None
    txt = str((sim.get("messages") or [])[idx].get("content") or "")
    hit = next((n for n in names if re.search(re.escape(n), txt, re.I)), None)
    return idx, hit


def main():
    names = candidate_names()
    print("후보 이름 어휘 %d개 (env 유래·예: %s)\n" % (len(names), ", ".join(names[:3])))
    summary = {}
    for arm, tag in ARMS:
        sims = F.sims(tag)
        deliv = F.turns_of(tag, DELIVER, sims)
        groups = F.by_sim(tag, SEARCHAG, sims)
        print("=" * 78)
        print("[%s] %s · n=%d" % (arm, tag, len(sims)))
        rows = []
        for s in sorted(sims, key=lambda x: (F.task_id(x), str(x.get("seed")))):
            key = F.simtag(s)
            dl = [t for t in (deliv.get(key) or []) if t is not None]
            fi, fn = named_first(s, names)
            arrived = (min(dl) <= fi) if (dl and fi is not None) else (bool(dl) and fi is None)
            rw = (s.get("reward_info") or {}).get("reward")
            calls = [F.nameof(tc) for _m, tc in F.calls(s)]
            wr = sum(1 for c in calls if c and any(w in c for w in WRITEISH))
            rows.append({
                "sim": key, "task": F.task_id(s), "deliver": dl, "first_idx": fi,
                "first_name": fn, "arrived": arrived, "reward": rw,
                "dur": round(s.get("duration") or 0, 1), "term": F.term_reason(s),
                "writes": wr, "groups": [g for _i, g in (groups.get(key) or [])],
                "ncalls": len(calls),
            })
            print("  %-22s deliver=%-8s first=%-5s %-28s arrived=%-5s reward=%-4s "
                  "dur=%-6s writes=%-2d term=%s"
                  % (key, dl or "-", fi if fi is not None else "-", (fn or "-")[:28],
                     arrived, rw, rows[-1]["dur"], wr, rows[-1]["term"]))
        summary[arm] = rows

    print("\n" + "=" * 78)
    print("ⓐ 배선")
    for arm, _t in ARMS:
        rs = summary[arm]
        n_d = sum(1 for r in rs if r["deliver"])
        infra = sum(1 for r in rs if r["term"] not in ("user_stop", "agent_stop", "max_steps"))
        print("   %-5s 선-배달 %d/%d sim · infra %d · 배달턴 %s"
              % (arm, n_d, len(rs), infra, sorted(t for r in rs for t in r["deliver"])))

    print("ⓑ 1차 종점 — 첫 지목 이전 도달 sim 비율")
    for arm, _t in ARMS:
        rs = [r for r in summary[arm] if r["task"] != "task_098"]
        print("   %-5s %d/%d (%s)" % (arm, sum(1 for r in rs if r["arrived"]), len(rs),
                                      " ".join("%s:%s" % (r["task"][-3:], "Y" if r["arrived"]
                                                          else "n") for r in rs)))

    print("ⓒ 첫 지목 이름 (태스크별)")
    for arm, _t in ARMS:
        c = collections.Counter((r["task"][-3:], r["first_name"] or "-") for r in summary[arm])
        print("   %-5s %s" % (arm, dict(c)))

    print("ⓓ 성적 (reward · 태스크별)")
    for arm, _t in ARMS:
        by = collections.defaultdict(list)
        for r in summary[arm]:
            by[r["task"]].append(r["reward"])
        print("   %-5s %s" % (arm, {k: "%d/%d" % (sum(1 for x in v if x == 1.0), len(v))
                                    for k, v in sorted(by.items())}))

    print("ⓔ 부작용")
    for arm, _t in ARMS:
        rs = summary[arm]
        d98 = [r["reward"] for r in rs if r["task"] == "task_098"]
        print("   %-5s 평균 duration %.0fs · 평균 write %.1f · 평균 호출 %.1f · 098 %d/%d"
              % (arm, sum(r["dur"] for r in rs) / max(1, len(rs)),
                 sum(r["writes"] for r in rs) / max(1, len(rs)),
                 sum(r["ncalls"] for r in rs) / max(1, len(rs)),
                 sum(1 for x in d98 if x == 1.0), len(d98)))
        gs = collections.Counter(g for r in rs for g in r["groups"])
        print("         문서군: %s" % dict(gs.most_common(6)))

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "..", "..", "reports", "facet_rft_2026",
                       "x339_precommit_verdict.json")
    with io.open(out, "w", encoding="utf-8") as f:
        f.write(json.dumps(summary, ensure_ascii=False, indent=1))
    print("\n저장: %s" % os.path.normpath(out))


if __name__ == "__main__":
    main()
