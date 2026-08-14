#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""db_match 진단: gold DB vs agent DB 직접 diff (2026-07-24·C149 · 2026-08-15 전수화).

db_match = strict full-DB 해시(`get_dict_hash(db.model_dump())`)이라 **어떤 필드 차이도 False**
(부분점수 없음). 이 도구는 gold(golden_actions 리플레이) DB와 agent(궤적 리플레이) DB의
`model_dump()` 를 재귀 diff 해 *정확히 어떤 DB 필드가 다른지* 출력한다.

⚠banking 은 `agent_discoverable_tools`(호출된 discoverable 도구 CALLED 기록·reads+writes)가
해시에 포함돼 **read-coverage 도 db_match 에 영향**한다 — 못 읽은 read 하나로도 실패한다.

⚠판정 지표 규율(C486·2026-08-15 확장): `action_match` 는 쓰지 마라. 하네스는 래퍼의 중첩
`arguments` 를 **문자열로 비교**하므로(`tasks.py:195` `tool_args == action_args`) 모델이 JSON 을
들여쓰기해 내면 의미가 같아도 False 다. 판정은 `reward`/`db_match`, 원인은 **이 diff** 로 한다.
(banking 97 태스크 중 88 = DB 기준 · 9 만 ACTION 기준이라 점수 영향은 그 9개에 국한.)

Run(remote):
  seka python dbdiff_task.py <SAVE_TAG> [task_id|ALL] [domain]
  seka python dbdiff_task.py <SAVE_TAG> ALL --summary     # 태스크×원인 교차표만
필수: PYTHONPATH=tau2-bench/src · no_knowledge variant 로 KB 임베딩(OpenAI) 우회.
"""
import collections
import io
import re
import sys
from pathlib import Path

from loguru import logger

from tau2.registry import registry
from tau2.data_model.simulation import Results

logger.remove()                      # 환경 리플레이가 도구 응답 전문을 DEBUG 로 쏟는다(신호 아님)

try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
except Exception:
    pass

ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
SUMMARY_ONLY = "--summary" in sys.argv[1:]
MAX_LINES = 40                       # sim 당 diff 출력 상한(교차표는 전수를 센다)

TAG = ARGS[0] if len(ARGS) > 0 else "bank_reg043fix_treat_20260724"
TASK_ID = ARGS[1] if len(ARGS) > 1 else "task_043"
DOMAIN = ARGS[2] if len(ARGS) > 2 else "banking_knowledge"
SIMROOT = "/home/woori/scratch/tau2-bench/data/simulations"

env_ctor = registry.get_env_constructor(DOMAIN)
tasks = {t.id: t for t in registry.get_tasks_loader(DOMAIN)()}
results = Results.load(Path("%s/%s/results.json" % (SIMROOT, TAG)))


def diff(g, p, path="", out=None):
    """재귀 diff 를 줄 리스트로 모은다(옛 판은 즉시 print 했다)."""
    if type(g) != type(p):
        out.append("TYPE %s: %r vs %r" % (path, g, p)); return
    if isinstance(g, dict):
        for k in sorted(set(g) | set(p), key=str):
            if k not in g:
                out.append("ONLY-PRED %s.%s = %r" % (path, k, p[k]))
            elif k not in p:
                out.append("ONLY-GOLD %s.%s = %r" % (path, k, g[k]))
            else:
                diff(g[k], p[k], path + "." + str(k), out)
    elif isinstance(g, list):
        if len(g) != len(p):
            out.append("LEN %s: gold=%d pred=%d" % (path, len(g), len(p)))
        for i in range(min(len(g), len(p))):
            diff(g[i], p[i], "%s[%d]" % (path, i), out)
    elif g != p:
        out.append("DIFF %s: gold=%r pred=%r" % (path, g, p))


def bucket(line):
    """diff 한 줄 → 원인 버킷. **판정은 하지 않는다** — DB 경로의 최상위 두 칸을 이름으로 쓴다.

    `agent_discoverable_tools` 는 read-coverage 축이라 write 축과 반드시 분리해 센다."""
    kind = line.split(" ", 1)[0]
    m = re.match(r"[A-Z-]+ \.([^.\[]+)(?:\[[^\]]*\])?\.?([^.\[ ]*)", line)
    top = m.group(1) if m else "?"
    sub = m.group(2) if m and m.group(2) else ""
    if top == "agent_discoverable_tools":
        return "READ-COVERAGE(%s)" % kind
    return "%s:%s%s" % (kind, top, ("." + sub) if sub else "")


def run_sim(task, sim):
    istate = task.initial_state
    gold = env_ctor(retrieval_variant="no_knowledge")
    gold.set_state(istate.initialization_data, istate.initialization_actions,
                   list(istate.message_history or []))
    for a in (task.evaluation_criteria.actions or []):
        try:
            gold.make_tool_call(tool_name=a.name, requestor=a.requestor, **a.arguments)
        except Exception as e:
            print("  gold ERR %s: %r" % (a.name, e))

    pred = env_ctor(retrieval_variant="no_knowledge")
    # strict=True 면 궤적의 tool_call id 가 어긋날 때 예외로 죽는다(우리 재생성 채널이 만드는
    # 경우가 있다). 하네스도 소급 재채점 경로에선 strict=False 를 쓴다 — 같은 규격으로 맞추되
    # strict 실패 사실은 버려지지 않게 되돌린다.
    # ⚠`strict` 는 이 예외를 우회하지 못한다 — id 짝맞춤은 `get_actions_from_messages` 안의
    # 하드 검사다. 리플레이 불능 자체가 결과이므로 삼키지 말고 그대로 올린다.
    try:
        pred.set_state(istate.initialization_data, istate.initialization_actions,
                       list(sim.messages))
    except ValueError as e:
        return None, ["REPLAY-FAIL .messages = %r" % (str(e)[:120],)], []

    lines = []
    diff(gold.tools.db.model_dump(), pred.tools.db.model_dump(), "", lines)
    ulines = []
    if gold.user_tools:
        diff(gold.user_tools.db.model_dump(), pred.user_tools.db.model_dump(), "", ulines)
    match = (gold.tools.get_db_hash() == pred.tools.get_db_hash())
    return match, lines, ulines


def main():
    ids = sorted(tasks) if TASK_ID.upper() == "ALL" else [TASK_ID]
    cross = collections.defaultdict(collections.Counter)
    simcount = collections.Counter()
    for tid in ids:
        task = tasks.get(tid)
        if task is None:
            continue
        sims = [s for s in results.simulations if s.task_id == tid]
        for n, sim in enumerate(sims):
            simcount[tid] += 1
            match, lines, ulines = run_sim(task, sim)
            for ln in lines + ulines:
                cross[tid][bucket(ln)] += 1
            if SUMMARY_ONLY:
                continue
            print("=" * 92)
            print("%s#%d  db_match=%s  diff줄=%d(agent)+%d(user)"
                  % (tid, n, match, len(lines), len(ulines)))
            for ln in lines[:MAX_LINES]:
                print("  " + ln)
            if len(lines) > MAX_LINES:
                print("  … +%d 줄" % (len(lines) - MAX_LINES))
            for ln in ulines[:MAX_LINES]:
                print("  [user] " + ln)

    print("#" * 92)
    print("# 교차표 — 태스크 × DB-diff 원인 (전 sim 합산)")
    for tid in sorted(cross):
        print("  %s (n=%d)" % (tid, simcount[tid]))
        for b, c in cross[tid].most_common(12):
            print("      %-46s %d" % (b, c))
    clean = [t for t in ids if simcount[t] and not cross[t]]
    if clean:
        print("# diff 0(=DB 일치) 태스크: %s" % ", ".join(clean))


main()
