# -*- coding: utf-8 -*-
r"""x249 — **우리 `[ORDER]` 가 부를 수 없는 이름을 부르라고 한다** (격리 · 유료 0 · 8140 · 새 엔진 0).

## 왜 (099 실패 2/3 의 자리 · [[55]] ①우리 배관 → ②우리 문구)

`bank_kb_20260811` 의 099 실패 두 시행은 **발견 도구를 unlock 만 하고 호출하지 않았다**(gold
`099_2` 미충족·그 한 칸만 빠져 DB 불일치). 사이드카 축자를 보면 unlock **직후 턴에도** 우리 층이
같은 문장을 그대로 다시 낸다:

    Error: [ORDER] 'submit_referral' cannot be carried out yet ...
      This has to hold first: the prior read(s) ... (do it with: get_all_user_accounts_by_user_id).
      Why it is still outstanding: get_all_user_accounts_by_user_id has not been called ...
      Steps that are possible right now: get_all_user_accounts_by_user_id_3847

**둘 다 에이전트가 부를 수 없는 이름이다.** 이 env 의 발견형 도구는 도구 목록에 서지 않고
`call_discoverable_agent_tool(agent_tool_name=...)` **디스패처로만** 불린다(gold 액션이 그 형태다).
C300 이 프런티어 이름을 접미사형(`..._3847`)으로 고쳤지만 **호출 형식**은 아직 아무 데서도 말하지
않는다 ⇒ 모델이 할 수 있는 최선은 `unlock_...` 이고, 그것을 하고 나면 **같은 요구가 또 온다**.
[[64]](거부는 *무엇을 하면 풀리나*를 담아라) 와 C414(접힘이 자기 원인을 재생산한다)의 같은 병이다.

## ⛔0 (결손을 먼저 쟀는가)

쟀다 — x241/C408: 같은 턴에서 궤적만 주면 **`A_FREE` 8/8**, 우리 문장을 되돌리면 **1/8**.
⇒ 격리에서 되므로 레버는 **전달(부하 축소)뿐**이고, 이 프로브는 *우리 문구를 고치는 것*과
*우리 문구를 지우는 것* 중 어느 쪽이 그 8/8 을 되찾는지만 가른다. **결정론기 순증 0.**

## 팔 (궤적 2개 × n · 계기 = 다음 도구 호출 하나 · 지시문 없음)

  A_LIVE      라이브 재현(궤적 + 사이드카 축자)               ← 재현 팔(기준·낮아야 한다)
  B_CALLFORM  우리 문장에서 **호출 형식만** 이름을 바꾼다     ← [[64]] 처방 팔
  C_DROP      unlock 이후의 우리 `[ORDER]` 문장을 **뺀다**    ← 뺄셈 팔([[63]])
  D_FREE      궤적만(우리 문장 0)                            ← 양성 통제(x241 A_FREE 재현)
  B_ENGINE    **엔진이 실제로 낼 문자열**로 치환(출시본과 동일·[[03b]])

★B 는 **인자를 주지 않는다** — 무엇을 넣을지는 끝까지 모델이 고른다(파라미터는 이미 unlock 출력이
  말했다). 바꾸는 것은 *어느 도구로 부르는가* 하나이고 그것은 env 규약(기계 도출·[[23]] opex 0)이다.
★문구는 **사이드카 축자에만** 손댄다 — 궤적(모델·손님 발화)은 한 글자도 안 고친다.

통과 = 다음 호출이 `call_discoverable_agent_tool` 이고 인자에 그 잠긴 도구 이름이 있는 것.

실행(리모트): python x249_order_callform_probe.py [N]
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
import x238_action_forensic as X                                  # noqa: E402
import x241_uncalled_unlock_probe as U                            # noqa: E402

TAG = "bank_kb_20260811"
TASK = "task_099"
LOCKED = "get_all_user_accounts_by_user_id_3847"
BARE = "get_all_user_accounts_by_user_id"
DISPATCH = "call_discoverable_agent_tool"


def callform(text):
    """우리 문장 안의 **부를 수 없는 이름**을 실제 호출 형식으로 바꾼다(값 0·지시 0).

    근거는 env 규약뿐이다: 발견형 도구는 도구 목록에 없고 디스패처로만 불린다. 인자는 손대지
    않는다 — unlock 출력이 이미 파라미터를 말했고 무엇을 넣을지는 모델이 고른다([[05]] Q2).
    """
    form = '%s(agent_tool_name="%s")' % (DISPATCH, LOCKED)
    out = text
    out = out.replace("(do it with: %s)" % BARE, "(do it with: %s)" % form)
    out = out.replace("%s has not been called in this conversation" % BARE,
                      "%s has not been called in this conversation - it is a discoverable tool, so "
                      "it is not in your tool list; the only way to run it is %s" % (BARE, form))
    # 프런티어 목록도 같은 규약으로 (C300 이 접미사까지만 고쳤다)
    out = out.replace("your choice): %s" % LOCKED, "your choice): %s" % form)
    return out


def engine_form():
    """**엔진이 실제로 낼 문자열**을 엔진에서 가져온다(두 벌 금지·[[03b]]).

    스텁은 이 env 의 구조만 재현한다 — 디스패처 두 개(스키마로 갈린다)와 발견형 레지스트리.
    문구도 치환 규칙도 우리가 다시 쓰지 않는다.
    """
    import t2_gate_patch as G

    class _T(object):
        def __init__(s, name, props):
            s.openai_schema = {"function": {"name": name, "parameters": {
                "type": "object", "properties": {p: {"type": "string"} for p in props}}}}

    class _A(object):
        tools = [_T("unlock_discoverable_agent_tool", ["agent_tool_name"]),
                 _T("call_discoverable_agent_tool", ["agent_tool_name", "arguments"])]

    class _K(object):
        def get_discoverable_tools(s):
            return {LOCKED}

    class _E(object):
        tools = _K()

    return G._call_form_map(_A(), _E(), [BARE, LOCKED])


def build(sim, cut, mode):
    """라이브 재현 문맥 — `mode` 로 **우리 문장만** 갈아 끼우거나 뺀다."""
    import t2_fbsidecar as FB

    class _M(object):
        def __init__(s, r, c):
            s.role, s.content = r, c

    keyed = FB._sim_key([_M(m.get("role"), m.get("content")) for m in sim["messages"]])
    ours = collections.defaultdict(list)
    if mode != "free":
        p = "/home/woori/scratch/logs/fb_%s.jsonl" % TAG
        for ln in open(p, encoding="utf-8", errors="replace"):
            o = json.loads(ln)
            if o.get("sim") == keyed and (o.get("text") or "").strip():
                ours[o.get("turn")].append(" ".join(o["text"].split()))
    unlock_at = max(i for i, m in enumerate(sim["messages"][:cut])
                    for tc in (m.get("tool_calls") or [])
                    if ((tc.get("function") or {}).get("name") or tc.get("name"))
                    == "unlock_discoverable_agent_tool")
    out, kept = [], 0
    for i, m in enumerate(sim["messages"][:cut]):
        r, c = m.get("role"), " ".join(str(m.get("content") or "").split())
        tcs = [(tc.get("function") or {}).get("name") or tc.get("name")
               for tc in (m.get("tool_calls") or [])]
        if tcs:
            out.append("[%s calls] %s" % (r, ", ".join(tcs)))
        if c:
            out.append("[%s] %s" % (r, c[:700]))
        for t in ours.get(i, ()):
            if mode == "drop" and i >= unlock_at and "[ORDER]" in t:
                continue
            if mode == "callform":
                t = callform(t)
            elif mode == "engine":
                for _k, _v in sorted(_ENGINE.items(), key=lambda kv: -len(kv[0])):
                    if _k in t and _v not in t:
                        t = t.replace(_k, _v)
            out.append("[system] %s" % t[:1100])
            kept += 1
    return "\n".join(out), kept, unlock_at


_ENGINE = {}


def main():
    global _ENGINE
    _ENGINE = engine_form()
    print("엔진 문구 축자: %s\n" % _ENGINE.get(BARE, "(없음)"))
    n = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 8
    sims = [s for s in X.load(TAG)
            if s["task_id"] == TASK and (s.get("reward_info") or {}).get("reward") != 1]
    print("실패 궤적 %d개 · n=%d · URL=%s\n"
          % (len(sims), n, os.environ.get("T2_PROBE_URL", "localhost:8140")))
    grand = collections.Counter()
    for sim in sims:
        cut = None
        for i, m in enumerate(sim["messages"]):
            for tc in (m.get("tool_calls") or []):
                if ((tc.get("function") or {}).get("name") or tc.get("name")) \
                        == "unlock_discoverable_agent_tool":
                    cut = i + 2
        if cut is None:
            print("  (unlock 없음 — 건너뜀 trial=%s)" % sim.get("trial"))
            continue
        tools = U.tools_of(sim)
        print("== trial %s · 요구 턴 %d · 도구 %d개" % (sim.get("trial"), cut, len(tools)))
        for mode, label in (("live", "A_LIVE"), ("callform", "B_CALLFORM"),
                            ("engine", "B_ENGINE"), ("drop", "C_DROP"), ("free", "D_FREE")):
            body, kept, ua = build(sim, cut, mode)
            c = collections.Counter()
            for i in range(n):
                try:
                    r = chat(body, tools, 0.0 if i == 0 else 0.7, 200)
                except Exception as e:
                    r = {"content": "ERR %s" % type(e).__name__}
                c[U.scored(r)] += 1
            grand[label] += c["HIT"]
            print("  %-11s 문맥 %6d자 · 우리문장 %2d개 · HIT %d/%d   %s"
                  % (label, len(body), kept, c["HIT"], n, c.most_common(3)))
    print("\n합계: " + " · ".join("%s %d" % (k, v) for k, v in grand.items()))
    print("※ B 가 A 보다 높으면 처방은 **문구가 이름을 대는 것**([[64]]) 이고,"
          "\n  C 만 높으면 처방은 **그 문장을 안 내는 것**([[63]] 뺄셈)이다. D 는 상한이다.")


if __name__ == "__main__":
    main()
