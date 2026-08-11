# -*- coding: utf-8 -*-
r"""x258 — **우리 문장은 한 턴만 산다** · x241 의 라이브 재현을 고쳐 다시 잰다 (유료 0 · 8140).

## 왜 (사용자 지적 2026-08-11 → 코드 확인)

`t2_gate_patch.py:5234` — 매 턴 생성 버퍼는 `work = list(state.messages)` 로 **커밋 히스토리에서
새로 짓는다**. 우리 `fb` 는 **비커밋**이므로 다음 턴에 `work` 를 다시 지으면 **사라진다.**
⇒ 우리 텍스트의 수명은 **한 턴**이고 턴을 가로질러 **쌓이지 않는다.**

그런데 `x241.ctx_with_ours` 는 사이드카 문장을 **모든 이전 턴 자리에** 되돌려 넣는다. 그건
*"우리 문장이 누적됐다면"* 의 문맥이고 **모델이 본 적 없는 것**이다. 그 팔이 낸 `H_LIVE_TRUE`
**1/8** 위에 C408⒝ 의 *"우리 문장 자체가 부하다"* 가 서 있다 — **근거가 부실하다.**

## 무엇을 가르나

  A_FREE     커밋 히스토리만 (우리 문장 0)            ← 상한
  H_TURN     **그 턴의 우리 문장만** 되돌림           ← **진짜 라이브 재현**
  H_ACCUM    모든 이전 턴의 우리 문장 되돌림          ← x241 이 쟀던 것(재현용)

읽는 법:
  `H_TURN ≈ A_FREE` ≫ `H_ACCUM`  ⇒ 부하는 **누적 가정의 산물**이었다. C408⒝ 를 정정한다.
  `H_TURN ≈ H_ACCUM` ≪ `A_FREE`  ⇒ 한 턴 분량만으로도 부하다. C408⒝ 는 살아남는다(이유가 바뀔 뿐).

⚠어느 쪽이든 x249·x250·x256 의 **대조**는 안 흔들린다 — 그 팔들은 같은 문맥 위에서 치환 하나만
  다르다. 흔들리는 것은 *"메인이 우리 문장으로 오염돼 있다"* 는 서사뿐이다.

실행(리모트): T2_PROBE_URL=http://localhost:8140/v1/chat/completions python x258_turnscope_load_probe.py [N]
"""
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from x216_read_and_offset import chat                             # noqa: E402
import x241_uncalled_unlock_probe as U                            # noqa: E402

TAG = "bank_cf_20260811b"
RES = "/home/woori/scratch/tau2-bench/data/simulations/%s/results.json" % TAG
FB = "/home/woori/scratch/logs/fb_%s.jsonl" % TAG
TASK = "task_099"


def ours_by_turn(sim):
    import t2_fbsidecar as S

    class _M(object):
        def __init__(s, r, c):
            s.role, s.content = r, c

    key = S._sim_key([_M(m.get("role"), m.get("content")) for m in sim["messages"]])
    out = collections.defaultdict(list)
    for ln in open(FB, encoding="utf-8", errors="replace"):
        o = json.loads(ln)
        if o.get("sim") == key and (o.get("text") or "").strip():
            out[o.get("turn")].append(" ".join(o["text"].split()))
    return out


def build(sim, cut, mode, ours):
    """mode: free | turn | accum. 궤적은 셋 다 동일 — 우리 문장을 어디에 두느냐만 다르다."""
    out = []
    for i, m in enumerate(sim["messages"][:cut]):
        r, c = m.get("role"), " ".join(str(m.get("content") or "").split())
        tcs = [tc.get("name") or (tc.get("function") or {}).get("name")
               for tc in (m.get("tool_calls") or [])]
        if any(tcs):
            out.append("[%s calls] %s" % (r, ", ".join(x for x in tcs if x)))
        if c:
            out.append("[%s] %s" % (r, c[:700]))
        if mode == "accum":
            for t in ours.get(i, ()):
                out.append("[system] %s" % t[:900])
    if mode == "turn":
        # ★그 턴의 것만. 사이드카 `turn` 은 기록 시점의 `len(messages)` 라 정확히 한 값이 아니라
        #   **cut 주변**에 걸린다 — 그래서 cut-1..cut+1 을 그 턴으로 본다(관대하게 잡아 준다:
        #   관대함이 H_TURN 을 낮추는 쪽이므로 결론에 유리하게 편향되지 않는다).
        near = [t for k in (cut - 1, cut, cut + 1) for t in ours.get(k, ())]
        for t in near:
            out.append("[system] %s" % t[:900])
    return "\n".join(out)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 8
    d = json.load(open(RES, encoding="utf-8"))
    sims = [s for s in d["simulations"]
            if s["task_id"] == TASK and (s["reward_info"] or {}).get("reward") != 1]
    print("실패 궤적 %d개 · n=%d\n" % (len(sims), n))
    grand = collections.Counter()
    for sim in sims[:3]:
        cut = None
        for i, m in enumerate(sim["messages"]):
            for tc in (m.get("tool_calls") or []):
                if (tc.get("name") or "") == "unlock_discoverable_agent_tool":
                    cut = i + 2
        if cut is None:
            continue
        ours = ours_by_turn(sim)
        tools = U.tools_of(sim)
        near = sum(len(ours.get(k, ())) for k in (cut - 1, cut, cut + 1))
        allc = sum(len(v) for k, v in ours.items() if k < cut)
        print("== trial %s · 요구 턴 %d · 그 턴 우리문장 %d개 / 누적 %d개"
              % (sim.get("trial"), cut, near, allc))
        for mode, label in (("free", "A_FREE"), ("turn", "H_TURN"), ("accum", "H_ACCUM")):
            body = build(sim, cut, mode, ours)
            c = collections.Counter()
            for i in range(n):
                try:
                    r = chat(body, tools, 0.0 if i == 0 else 0.7, 200)
                except Exception as e:
                    r = {"content": "ERR %s" % type(e).__name__}
                c[U.scored(r)] += 1
            grand[label] += c["HIT"]
            print("  %-9s 문맥 %6d자 · HIT %d/%d   %s"
                  % (label, len(body), c["HIT"], n, c.most_common(3)))
    print("\n합계: " + " · ".join("%s %d" % (k, v) for k, v in grand.items()))
    print("※ H_TURN 이 A_FREE 에 가까우면 C408⒝ 의 부하 서사는 **누적 가정의 산물**이다.")


if __name__ == "__main__":
    main()
