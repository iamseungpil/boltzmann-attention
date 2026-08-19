# -*- coding: utf-8 -*-
r"""x402 — 완료 주장이 **거짓인가**: 주장 ↔ 로그 대조 (닫힌 술어) + 라이브 종료 방식

## ⑴ 주장 진위 (엔진이 답을 만들지 않는다 — 로그에 있나 없나만 본다)
주장 문장에서 **id 토큰**(chk_/sav_/dbc_/cc_/txn_/acc_/ac 접두)을 뽑아, 그 id 를 인자로 가진
**성공한 write 호출**이 그 sim 안에 있는지 본다.
    SUPPORTED   있다 -> 참말(다른 gold 를 놓친 것과 별개)
    UNSUPPORTED 없다 -> 거짓 완료 주장
    NO_ID       id 가 없어 검산 불가 (조용히 어느 쪽에도 안 넣는다)

## ⑵ 라이브 종료 방식 — 격리의 {"tool": null} 에 대응하는 것이 라이브에 있나
실패 sim 마다 종료사유 · 마지막 assistant 메시지가 도구호출인지 산문인지 · 남은 미매치 gold 수
"""
import collections, io, json, os, re, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import t2_forensic as F
import x396_saying_vs_doing as C
from x400_belief_strict import sents

ID_RE = re.compile(r"\b(?:chk|sav|dbc|txn|cc|acc)_[A-Za-z0-9_]+\b")
ENVERR = ("Error:", "NOT_VERIFIED", "not been given", "Unknown", "Invalid", "cannot be")
READ_HINT = ("get_", "list_", "search_", "find_", "check_", "read_", "fetch_", "retrieve_")


def succ_write_ids(sim):
    """성공한 **비-read** 호출들이 건드린 id 집합 -> {id: [도구…]}"""
    R = {}
    for m in (sim.get("messages") or []):
        if m.get("role") == "tool" and m.get("id"):
            R[m["id"]] = " ".join(str(m.get("content") or "").split())
    out = collections.defaultdict(list)
    for m, tc in F.calls(sim):
        a = F.argsof(tc)
        nm = str(F.inner_name(a) or F.nameof(tc))
        body = R.get(tc.get("id"), "")
        if not body or any(p in body for p in ENVERR):
            continue
        if any(nm.startswith(h) for h in READ_HINT):
            continue
        for i in ID_RE.findall(json.dumps(a, ensure_ascii=False, default=str)):
            out[i].append(nm)
    return out


def main():
    print("=" * 110)
    print("x402 · ⑴ 완료 주장 진위 (주장 ↔ 로그)")
    print("=" * 110)
    tot = collections.Counter(); rows = []
    endrow = []
    for tag in C.TAGS:
        for sim in F.scored(tag, C.SUF):
            rw = ((sim.get("reward_info") or {}).get("reward") or 0)
            task, tr = F.task_id(sim), sim.get("trial")
            miss = [g for g in C.gold_rows(sim) if not g["match"]]
            if rw < 1.0:
                W = succ_write_ids(sim)
                for t in C.assistant_texts(sim):
                    for s in sents(" ".join(t.split())):
                        if not C.DONE_RE.search(s):
                            continue
                        ids = set(ID_RE.findall(s))
                        if not ids:
                            tot["NO_ID"] += 1; rows.append((task, tr, "NO_ID", s, ""))
                            continue
                        hit = [i for i in ids if i in W]
                        if hit:
                            tot["SUPPORTED"] += 1
                            rows.append((task, tr, "SUPPORTED", s, ",".join(sorted(set(W[hit[0]])))[:40]))
                        else:
                            tot["UNSUPPORTED"] += 1
                            rows.append((task, tr, "UNSUPPORTED", s, "id=" + ",".join(sorted(ids))[:40]))
            # ⑵ 종료 방식 (실패 sim 전량)
            msgs = sim.get("messages") or []
            last_a = next((m for m in reversed(msgs) if m.get("role") == "assistant"), None)
            kind = "?" if last_a is None else ("TOOLCALL" if (last_a.get("tool_calls") or []) else "PROSE")
            if rw < 1.0:
                endrow.append((task, tr, F.term_reason(sim), kind, len(miss),
                               " ".join(str((last_a or {}).get("content") or "").split())[:90]))
    for k in ("UNSUPPORTED", "SUPPORTED", "NO_ID"):
        print("  %-12s %3d" % (k, tot[k]))
    print("\n## 거짓으로 판정된 주장 축자 (UNSUPPORTED 전량)")
    for r in rows:
        if r[2] == "UNSUPPORTED":
            print("  %-9s t%-2s %s\n%s%s" % (r[0], r[1], r[3][:112], " " * 16, r[4]))
    print("\n## 참으로 판정된 주장 (SUPPORTED · 최대 8)")
    n = 0
    for r in rows:
        if r[2] == "SUPPORTED" and n < 8:
            n += 1; print("  %-9s t%-2s %-88s [%s]" % (r[0], r[1], r[3][:88], r[4]))

    print("\n" + "=" * 110)
    print("x402 · ⑵ 라이브 실패 sim 의 종료 방식 (격리의 {\"tool\": null} 대응물이 있나)")
    print("=" * 110)
    print("  종료사유 분포: %s" % dict(collections.Counter(e[2] for e in endrow)))
    print("  마지막 assistant: %s" % dict(collections.Counter(e[3] for e in endrow)))
    print("\n  %-9s %-3s %-22s %-9s %-5s %s" % ("task", "tr", "term", "last", "miss", "마지막 본문"))
    for e in sorted(endrow):
        print("  %-9s %-3s %-22s %-9s %-5s %s" % (e[0], e[1], str(e[2])[:22], e[3], e[4], e[5]))
    return 0

sys.exit(main())
