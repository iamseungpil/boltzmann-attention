#!/usr/bin/env python
"""mech_autopsy — 실패 sim 전수 기전-클러스터링 (P2·C175·웨이브 파이프라인의 무료 진단기).

입력: results.json 체크포인트 경로들(shell glob 가능). 각 실패 sim에 대해
  실행경로 분해(C161 규율: EXEC/UNLOCK/DIRECT-SUFF·인자 문자열 매칭 금지) +
  gold-replay DB-diff(부기 reg vs 상태 state 분리) + 궤적 신호(사임문·루프·KB쿼리)
를 결합해 기전 라벨을 붙이고 클러스터 표를 출력한다.

라벨(우선순위 순·복수 가능):
  INFRA          nmsg=0 또는 infrastructure_error
  OVER_ACTION    ONLY-PRED 상태 diff(gold에 없는 write 수행)
  WRONG_VALUE    DIFF 리프(수행했으나 값 불일치)
  MISSING_WRITE  ONLY-GOLD 상태 diff(gold write 미수행)
  NEVER_CALLED   reg-diff 도구를 EXEC/UNLOCK/직접 시도조차 0회
  CALL_PATH      reg-diff 도구를 직접-suffixed로 호출(dispatcher 미경유)
  RESIGN         GB2 공지/사임문 존재
  LOOP           동일 (도구,인자) 3회+ 반복
  SHORT_STALL    nmsg<16 & EXEC 0 (초기 스톨)
사용:
  PYTHONPATH=<tau2>/src python mech_autopsy.py <ckpt1.json> [ckpt2.json ...]
"""
import glob
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, "/home/woori/scratch/tau2-bench/src")
from tau2.registry import registry            # noqa: E402
from tau2.data_model.simulation import Results  # noqa: E402

_SUFF = re.compile(r"_\d{3,4}$")


def _args(tc):
    a = tc.get("arguments") if isinstance(tc, dict) else getattr(tc, "arguments", None)
    if isinstance(a, str):
        try:
            a = json.loads(a)
        except Exception:
            a = {}
    return a if isinstance(a, dict) else {}


def _tcs(m):
    md = m.model_dump() if hasattr(m, "model_dump") else m
    return md, (md.get("tool_calls") or [])


def diff(g, p, path="", out=None):
    if type(g) is not type(p):
        out.append(("TYPE", path, repr(g)[:60], repr(p)[:60])); return
    if isinstance(g, dict):
        for k in sorted(set(g) | set(p), key=str):
            if k not in g:
                out.append(("ONLY-PRED", path + "." + str(k), "", repr(p[k])[:90]))
            elif k not in p:
                out.append(("ONLY-GOLD", path + "." + str(k), repr(g[k])[:90], ""))
            else:
                diff(g[k], p[k], path + "." + str(k), out)
    elif isinstance(g, list):
        if len(g) != len(p):
            out.append(("LEN", path, str(len(g)), str(len(p))))
        for i in range(min(len(g), len(p))):
            diff(g[i], p[i], "%s[%d]" % (path, i), out)
    elif g != p:
        out.append(("DIFF", path, repr(g)[:60], repr(p)[:60]))


def autopsy_sim(sim, task, env_ctor):
    md0 = sim.messages
    ist = getattr(task, "initial_state", None)
    ia = (ist.initialization_data, ist.initialization_actions,
          list(ist.message_history or [])) if ist else (None, None, [])
    ri = sim.reward_info
    rew = getattr(ri, "reward", None) if ri else None
    term = str(getattr(sim, "termination_reason", ""))
    n = len(md0)
    labels = []
    feat = {"reward": rew, "nmsg": n, "term": term.split(".")[-1]}
    if n == 0 or "INFRA" in term.upper():
        return ["INFRA"], feat, []
    # 실행경로 + 신호
    execs, unlocks, directs, calls = [], [], [], []
    resign = kbq = 0
    for m in md0:
        md, tcs = _tcs(m)
        c = str(md.get("content") or "")
        if md.get("role") == "assistant" and "no further actions I can take" in c:
            resign += 1
        for tc in tcs:
            nm = str(tc.get("name") or "")
            a = _args(tc)
            calls.append((nm, json.dumps(a, sort_keys=True)[:120]))
            if nm == "call_discoverable_agent_tool":
                execs.append(_SUFF.sub("", str(a.get("agent_tool_name") or "")))
            elif nm == "unlock_discoverable_agent_tool":
                unlocks.append(_SUFF.sub("", str(a.get("agent_tool_name") or "")))
            elif _SUFF.search(nm):
                directs.append(_SUFF.sub("", nm))
            if nm.startswith("KB_search"):
                kbq += 1
    loops = [k for k, v in Counter(calls).items() if v >= 3]
    feat.update(execs=len(execs), kbq=kbq, resign=resign, loops=len(loops))
    # DB-diff
    gold = env_ctor(retrieval_variant="no_knowledge"); gold.set_state(*ia)
    for a in (task.evaluation_criteria.actions or []):
        try:
            gold.make_tool_call(tool_name=a.name, requestor=a.requestor, **a.arguments)
        except Exception:
            pass
    pred = env_ctor(retrieval_variant="no_knowledge"); pred.set_state(ia[0], ia[1], list(md0))
    out = []
    diff(gold.tools.db.model_dump(), pred.tools.db.model_dump(), "", out)
    reg = [d for d in out if d[1].startswith(".agent_discoverable_tools")]
    st = [d for d in out if not d[1].startswith(".agent_discoverable_tools")]
    feat.update(reg=len(reg), state=len(st))
    if any(d[0] == "ONLY-PRED" for d in st):
        labels.append("OVER_ACTION")
    if any(d[0] == "DIFF" for d in st):
        labels.append("WRONG_VALUE")
    if any(d[0] == "ONLY-GOLD" for d in st):
        labels.append("MISSING_WRITE")
    # reg-diff 도구의 시도 여부
    attempted = set(execs) | set(unlocks) | set(directs)
    reg_tools = []
    for d in reg:
        mm = re.search(r"'tool_name': '([^']+)'", (d[2] or d[3]) or "")
        if mm:
            reg_tools.append(_SUFF.sub("", mm.group(1)))
    nc = [t for t in reg_tools if t not in attempted]
    cp = [t for t in reg_tools if t in set(directs)]
    if nc:
        labels.append("NEVER_CALLED")
    if cp:
        labels.append("CALL_PATH")
    if resign:
        labels.append("RESIGN")
    if loops:
        labels.append("LOOP")
    if n < 16 and not execs:
        labels.append("SHORT_STALL")
    top = ["%s %s" % (d[0], d[1][:70]) for d in (st + reg)[:4]]
    return (labels or ["UNCLASSIFIED"]), feat, top


def main():
    paths = []
    for p in sys.argv[1:]:
        paths += glob.glob(p)
    env_ctor = registry.get_env_constructor("banking_knowledge")
    tasks = {t.id: t for t in registry.get_tasks_loader("banking_knowledge")()}
    clusters = defaultdict(list)
    rows = []
    for cp in paths:
        try:
            res = Results.load(Path(cp))
        except Exception as e:
            print("[skip]", cp, e); continue
        for sim in res.simulations:
            t = tasks.get(sim.task_id)
            if t is None:
                continue
            ri = sim.reward_info
            rew = getattr(ri, "reward", None) if ri else None
            if rew == 1.0:
                clusters["PASS"].append(sim.task_id); continue
            try:
                labels, feat, top = autopsy_sim(sim, t, env_ctor)
            except Exception as e:
                labels, feat, top = ["AUTOPSY_ERR"], {"err": str(e)[:60]}, []
            key = "+".join(labels)
            clusters[key].append(sim.task_id)
            rows.append((sim.task_id, key, feat, top))
    print("=== CLUSTERS (fail mechanisms) ===")
    for k in sorted(clusters, key=lambda x: -len(clusters[x])):
        ids = sorted(set(clusters[k]))
        print("%-42s %2d  %s" % (k, len(ids), ",".join(i.replace("task_", "") for i in ids)))
    print("\n=== PER-TASK DETAIL (fails) ===")
    for tid, key, feat, top in sorted(rows):
        print("%s  [%s]  %s" % (tid, key, feat))
        for tline in top:
            print("      ", tline)


if __name__ == "__main__":
    main()
