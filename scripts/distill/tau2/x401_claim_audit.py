# -*- coding: utf-8 -*-
r"""x401 — 완료 주장 **전수 정독**: 묶기 강도 3단계 + 축자 목록 인쇄 ([[08]] 포렌식)

x400 에서 문장 단위 묶기로 CLAIM_TIED=0 이 나왔다. 계기가 너무 빡셀 수 있다(계좌를 색 이름으로
부르면 id 토큰이 문장에 없다). 강도를 나눠 재고, **완료 주장 문장을 전부 인쇄해서 눈으로 읽는다**.
    S1 문장 단위 : 같은 문장에 이름 또는 목적어 토큰
    S2 메시지 단위: 같은 assistant 메시지 안에 이름 또는 목적어 토큰
    S3 sim 단위  : x399 방식(무제한)
"""
import collections, io, os, re, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import t2_forensic as F
import x396_saying_vs_doing as C
from x400_belief_strict import sents

def main():
    print("=" * 112)
    print("x401 · 실패 sim 의 완료 주장 전수 (묶기 강도 3단계)")
    print("=" * 112)
    lvl = collections.Counter(); allclaims = []
    for tag in C.TAGS:
        for sim in F.scored(tag, C.SUF):
            if ((sim.get("reward_info") or {}).get("reward") or 0) >= 1.0:
                continue
            task, tr = F.task_id(sim), sim.get("trial")
            gold = C.gold_rows(sim)
            miss = [g for g in gold if not g["match"]]
            texts = C.assistant_texts(sim)
            calls = C.called(sim)
            for mi, t in enumerate(texts):
                msg = " ".join(t.split())
                for s in sents(msg):
                    if not C.DONE_RE.search(s):
                        continue
                    # 이 주장이 미매치 gold 중 어느 것과 묶이나
                    s1 = [g["name"] for g in miss
                          if g["name"] in s or any(o in s for o in C.operand_tokens(g["args"]))]
                    s2 = [g["name"] for g in miss
                          if g["name"] in msg or any(o in msg for o in C.operand_tokens(g["args"]))]
                    lev = "S1" if s1 else ("S2" if s2 else "S3")
                    lvl[lev] += 1
                    allclaims.append((task, tr, lev, s, sorted(set(s1 or s2))[:2],
                                      sum(calls.values())))
    print("\n## 묶기 강도별 완료 주장 문장 수")
    for k in ("S1", "S2", "S3"):
        print("  %-4s %3d" % (k, lvl[k]))
    print("\n## 축자 전수 (task/trial/강도/총호출수 · 주장문 · 묶인 미매치 gold)")
    for task, tr, lev, s, tie, nc in allclaims:
        print("  %-9s t%-2s %-3s call%-3d %s" % (task, tr, lev, nc, s[:118]))
        if tie:
            print("            %s-> %s" % (" " * 12, ", ".join(tie)))
    print("\n  총 %d문장" % len(allclaims))
    return 0

sys.exit(main())
