#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""x772 - 흐름/범위 레버군의 **발화 실측** (리모트 · GPU 0 · CPU 만).

맡은 넷 중 궤적에서 재현 가능한 셋을 엔진 술어 그대로 46 핀 sim 에 적용한다.

  T2_GIVE_REQUIRED      `G._give_required_fb(prefix, shim_orch)` 를 매 접두에서 호출.
  T2_DUP_WRITE          `_succeeded_mut_keys` + `_once_key_of`/`_mut_key_of` + `_is_effective_write`
                        (현행 = mut|once 둘 다 · D6 = once 만) 두 팔을 나란히.
  T2_SCOPE_ALL          로그에서 센다(이 파일 밖) - 여기서는 침묵된 `chosen` 의 write/read 판정만.

⛔ 판정하지 않는다. 세기만 한다.
"""
import io, os, sys, json
from pathlib import Path
from loguru import logger

logger.remove()
try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
except Exception:
    pass

REPO = "/home/woori/workspace_common/boltzmann-attention-pi"
sys.path.insert(0, REPO + "/scripts/distill/tau2")
import t2_gate_patch as G                       # noqa: E402
import gate_interpreter as GI                   # noqa: E402
from tau2.registry import registry              # noqa: E402
from tau2.data_model.simulation import Results  # noqa: E402

SIMROOT = "/home/woori/scratch/tau2-bench/data/simulations"
DOMAIN = "banking_knowledge"
A2 = GI.load_domain_a2(DOMAIN)
env_ctor = registry.get_env_constructor(DOMAIN)


class Shim(object):
    """`_give_required_fb` 가 보는 것은 `orch.environment` 뿐이다."""
    def __init__(self, env):
        self.environment = env
        self._t2_orch = None


ENV = env_ctor()
SHIM = Shim(ENV)
try:
    import t2_axis_levers as _AX
    AG, USER_REG = _AX.registry_from_env(SHIM)
except Exception as e:
    AG, USER_REG = set(), set()
    print("REGFAIL %r" % (e,))
print("USER_REG(%d) = %s" % (len(USER_REG), sorted(USER_REG)))
print("AGENT_DISC(%d)" % len(AG))

PAIRS = [ln.split() for ln in open(sys.argv[1]).read().strip().splitlines() if ln.strip()]
cache = {}

for tag, tid, simid in PAIRS:
    if tag not in cache:
        try:
            cache[tag] = Results.load(Path("%s/%s/results.json" % (SIMROOT, tag)))
        except Exception as e:
            print("LOADFAIL %s %s %r" % (tid, tag, e))
            cache[tag] = None
    res = cache[tag]
    if res is None:
        continue
    sim = next((s for s in res.simulations if s.id == simid), None)
    if sim is None:
        print("NOSIM %s %s" % (tid, tag))
        continue
    msgs = list(sim.messages or [])

    # ── T2_GIVE_REQUIRED ────────────────────────────────────────────────
    gr_turns, gr_tools = [], []
    for i, m in enumerate(msgs):
        if str(getattr(m, "role", "")) != "assistant":
            continue
        try:
            fb = G._give_required_fb(msgs[:i + 1], SHIM)
        except Exception as e:
            fb = None
        if fb:
            gr_turns.append(i)
            # 문면에 실린 도구명을 축자로 뽑는다
            for tok in str(fb).split("`"):
                if tok in USER_REG:
                    gr_tools.append(tok)
                    break

    # 손님이 시도한 user-tool 과 give 여부(재료 자체)
    tried, given = {}, set()
    for m in msgs:
        for tc in (getattr(m, "tool_calls", None) or []):
            nm = str(getattr(tc, "name", "") or "")
            a = G._args_dict(tc) or {}
            if nm == "call_discoverable_user_tool":
                x = str(a.get("discoverable_tool_name") or "")
                tried[x] = tried.get(x, 0) + 1
            if nm == "give_discoverable_user_tool" or G._eff_tool_name(tc) == "give_discoverable_user_tool":
                given.add(str(a.get("discoverable_tool_name") or ""))

    # ── T2_DUP_WRITE (현행 mut|once  vs  D6 once-only) ──────────────────
    dup_cur, dup_d6 = [], []
    for i, m in enumerate(msgs):
        if str(getattr(m, "role", "")) != "assistant":
            continue
        tcs = getattr(m, "tool_calls", None) or []
        if not tcs:
            continue
        try:
            dupmap = G._succeeded_mut_keys(msgs[:i], A2)
        except Exception:
            continue
        for dc in tcs:
            try:
                if not G._is_effective_write(G._eff_tool_name(dc), A2):
                    continue
                ok = G._once_key_of(dc, A2)
                mk = G._mut_key_of(dc)
            except Exception:
                continue
            hit = None
            for cand in (ok, mk):
                if cand and cand in dupmap:
                    hit = cand
                    break
            if hit:
                kind = "once" if str(hit).startswith("once|") else "mut"
                dup_cur.append((i, G._eff_tool_name(dc), kind))
                if kind == "once":
                    dup_d6.append((i, G._eff_tool_name(dc), kind))

    print("SIM %s msgs=%d reward=%s" % (tid, len(msgs),
                                        (sim.reward_info.reward if sim.reward_info else "?")))
    print("  GIVE_REQ fires=%d turns=%s tools=%s | tried=%s given=%s"
          % (len(gr_turns), gr_turns[:6], sorted(set(gr_tools)), tried, sorted(given)))
    print("  DUP_WRITE cur=%d d6=%d detail=%s" % (len(dup_cur), len(dup_d6), dup_cur[:6]))
