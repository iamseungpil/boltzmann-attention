# -*- coding: utf-8 -*-
"""x161 — 재도출이 **무엇을 보고** 그 답을 냈나 (유료 0·[[08]] 궤적 정독).

왜 따로 만드나: C345 의 원인 진술(*"라이브 조립본이 프로브가 잰 구성이 아니다"*)은 집계
(`value: model gave 0` 15/17)와 프로브 대조로 세운 것이고, **궤적을 읽지 않았다**. [[08]] 은
집계에서 결론 직행을 금한다 — calc 레버(단위 OK·라이브 31/342)와 길이 법칙(집계가 만든 오답)이
그 전례다. 그래서 이 도구는 sim 마다 **우리 층이 실제로 실어 보낸 글자**를 그대로 꺼낸다.

가르는 것 (x158 이 10/10 을 낸 구성과 대조):
  · 통과 집합에 **몇 행**이 남았나 — 예치로 걸렀다면 줄고, 못 걸렀으면 전 행이 남는다
  · 그 행들에 **대화-피연산자 축**이 실렸나 (걸렀다는 증거)
  · 재도출이 받은 **사실 문장이 몇 줄**인가 — `days` 한 줄이면 프로브의 두꺼운 FACTS 가 아니다
  · 그래서 재도출이 **무엇을 답했나**, 그리고 에이전트가 **무엇을 제출했나**

읽는 곳: 사이드카(`fb_<TAG>.jsonl` = 우리 층 발화의 유일한 기록·비커밋) + 결과(궤적·보상).
실행: py -3 x161_rederive_forensic.py <fb_TAG.jsonl> <results.json[.gz]> [task_099]
"""
import collections
import gzip
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def _load(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt", encoding="utf-8") as fh:
        return json.load(fh)


def _records(path):
    out = []
    with open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out


def _text_of(rec):
    """사이드카 레코드에서 우리가 실제로 실어 보낸 글자를 꺼낸다(키 이름은 판마다 다르다)."""
    for k in ("text", "content", "emitted", "msg", "body"):
        v = rec.get(k)
        if isinstance(v, str) and v.strip():
            return v
    return ""


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    fb, res = sys.argv[1], sys.argv[2]
    want = sys.argv[3] if len(sys.argv) > 3 else None

    recs = _records(fb)
    data = _load(res)
    sims = data.get("simulations") or data.get("results") or []

    # 사이드카는 sim id 를 갖고 있을 수도, 지문만 있을 수도 있다 — 둘 다 받는다.
    by_sim = collections.defaultdict(list)
    for r in recs:
        by_sim[r.get("sim_id") or r.get("sim") or r.get("fingerprint") or "?"].append(r)

    print("사이드카 레코드 %d · sim %d · 결과 sim %d" % (len(recs), len(by_sim), len(sims)))

    for s in sims:
        task = s.get("task_id") or (s.get("task") or {}).get("id")
        if want and task != want:
            continue
        rid = s.get("id") or s.get("simulation_id") or "?"
        rew = (s.get("reward_info") or {}).get("reward", s.get("reward"))
        print("\n" + "=" * 78)
        print("sim=%s  task=%s  reward=%s" % (str(rid)[:12], task, rew))

        # ── 우리 층이 실어 보낸 통과 집합 ──────────────────────────────────
        blocks = [t for t in (_text_of(r) for r in by_sim.get(rid, [])) if t]
        if not blocks:                      # id 로 못 붙으면 전 레코드에서 task 로 좁힌다
            blocks = [_text_of(r) for r in recs
                      if (r.get("task_id") or r.get("task")) == task and _text_of(r)]
        rows = []
        for b in blocks:
            rows += [l.rstrip() for l in b.splitlines() if l.startswith("  ") and ":" in l]
        uniq = list(dict.fromkeys(rows))
        print("  통과 집합 행 수 = %d" % len(uniq))
        for l in uniq[:12]:
            print("    " + l.strip()[:150])

        # ── 에이전트의 마지막 발화 두 개 (무엇을 근거로 말했나) ─────────────
        msgs = s.get("messages") or (s.get("simulation") or {}).get("messages") or []
        said = [m for m in msgs if (m.get("role") == "assistant") and (m.get("content") or "")]
        for m in said[-2:]:
            print("  --- agent: " + " ".join(str(m.get("content")).split())[:400])

    return 0


if __name__ == "__main__":
    sys.exit(main())
