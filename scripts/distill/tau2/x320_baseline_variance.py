# -*- coding: utf-8 -*-
r"""x320 — **기준선 분산 측정**: n=8 로 무엇을 가를 수 있고 무엇은 못 가르나.

사건(2026-08-14 야간): 같은 컷·같은 프롬프트의 A_REF 가
    x318 에서 **1/8** · x319 에서 **5/8**
로 나왔다. 본문은 **바이트 동일**(21,288자·sha 7e467761e25e)이므로 계기 차이가 아니라
**표집 분산**이다. 그러면 오늘 n=8 로 내린 판정 중 *중간 크기 차이*에 기댄 것은 전부 흔들린다.

이 프로브는 레버를 재지 않는다. **계기의 눈금**을 잰다:
  · 같은 본문을 **블록 5개 × n=8** 로 돌려 블록별 적중을 본다(우리 프로브 1회 = 1블록)
  · 온도 0 만 8회 따로 — 우리가 i==0 에만 쓰는 그 설정이 실제로 결정론인지
  · 산출 질의의 **문자열 다양성**도 센다(같은 답을 반복하는가, 매번 새로 짓는가)

결과로 얻을 것: *"n=8 에서 k/8 차이는 언제 신호이고 언제 잡음인가"* 의 실측 기준.
이 수치가 나오기 전에는 오늘 프로브들의 **중간 차이 판정을 인용하지 않는다**([[08]]).

채점은 x318 축자(질의 → 하네스 bm25 → doc 017 회수·기계적).

실행(리모트·8141): T2_PROBE_URL=http://localhost:8141/v1/chat/completions \
  /home/woori/venvs/seka_env/bin/python x320_baseline_variance.py [블록수] [블록크기]
"""
import collections
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/home/woori/scratch/tau2-bench/src")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from x216_read_and_offset import chat                             # noqa: E402
import t2_forensic as F                                           # noqa: E402
import x313_bailout_iso as B                                      # noqa: E402
import x318_query_formation_iso as Q                              # noqa: E402


def run_block(pipe, body, k, all_zero=False):
    hits, outs = 0, []
    for i in range(k):
        temp = 0.0 if (all_zero or i == 0) else 0.7
        try:
            r = chat(body, None, temp, 120)
        except Exception as e:
            r = {"content": "ERR %s" % type(e).__name__}
        out = " ".join(str(r.get("content") or "").split())[:120]
        ok = Q.score_query(pipe, out)
        hits += ok
        outs.append(out)
    return hits, outs


def main():
    nb = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 5
    k = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 8
    raw = Q.load_docs()
    pipe = Q.bm25_pipe(raw)
    sim = next(s for s in F.scored(Q.TAG) if F.task_id(s) == Q.TASK)
    body = "\n".join([B.HEAD, "", B.transcript(sim, 54)]) + Q.ASK_Q
    print("x320 · 본문 %d자 · 블록 %d × n=%d (우리 프로브 1회 = 1블록)\n" % (len(body), nb, k))

    blocks, allouts = [], []
    for b in range(nb):
        h, outs = run_block(pipe, body, k)
        blocks.append(h)
        allouts += outs
        print("  블록 %d: %d/%d" % (b + 1, h, k), flush=True)
    print("\n블록별: %s" % blocks)
    print("최소 %d · 최대 %d · 평균 %.1f · 폭 %d" % (min(blocks), max(blocks),
                                                sum(blocks) / float(nb),
                                                max(blocks) - min(blocks)))
    hz, outz = run_block(pipe, body, k, all_zero=True)
    print("\n온도 0 만 %d회: %d/%d · 서로 다른 응답 %d종" % (k, hz, k, len(set(outz))))
    c = collections.Counter(allouts)
    print("표집 %d회 중 서로 다른 질의 %d종 · 최빈 %d회: %s"
          % (len(allouts), len(c), c.most_common(1)[0][1], c.most_common(1)[0][0][:70]))
    print("\n※ 이 폭이 곧 n=8 프로브의 **잡음 바닥**이다. 두 팔의 차이가 이 폭 안이면 "
          "신호로 인용하지 말 것 — 오늘 A_REF 가 1/8 ↔ 5/8 로 흔들린 것이 그 실물이다.")


if __name__ == "__main__":
    main()
