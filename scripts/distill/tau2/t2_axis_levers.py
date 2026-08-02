#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""t2_axis_levers: x38 6축 포렌식이 확정한 실패 기전에 대한 **표면화 레버 모음**.

정본 = `FAILURE_AXES_REDESIGN_2026_08_02` · `RUNAWAY_AXIS_REDESIGN_2026_08_02`.
전부 **표면화**다 — 거부·재작성·값 생성 0([[10]]). 도메인 리터럴 0(이름은 env 레지스트리, 문구는 A2).
각 레버는 **환경변수 플래그로 기본 OFF**이고, 미설정이면 거동 변화 0.

| 플래그 | 축 | 표적(x41 전수 계수) |
|---|---|---|
| `T2_TOOL_CHANNEL`  | 3⒜ 채널 오분류 + 폭주⒝ 방출 불가 | 140회 / 86 sim |
| `T2_TERMINAL_TURN` | 1ⓐ 이관 동의 후 미실행 | (banking 한정 재계수 필요·§주의) |
| `T2_FIT_DIFF`      | 2 후보 미판별 | fit 호출 133회 **전건 ≥2장** / 99 sim |
| `T2_SCALAR_ARRAY`  | 3⒞ 배치화 | 단수명 필드 배열 (banking 6회) |

★계측기 주의(x41에서 실제로 밟음): `item_ids`처럼 **원래 배열인 파라미터**를 배치화로 세면 안 된다 —
  판정은 **단수 이름(복수형 아님) 필드에 배열이 온 경우**로 좁힌다.
"""
import json
import os
import re

_SCAFFOLD_HINT = ("scaffold_get_tools",)          # A2에서 스캐폴드 도구 이름을 읽는다(리터럴 0)


def _norm(s):
    return re.sub(r"[^a-z0-9]+", "", str(s or "").lower())


def _content(msg):
    c = getattr(msg, "content", None)
    return c if isinstance(c, str) else ""


def _set_content(msg, text):
    try:
        msg.content = text
    except Exception:
        pass
    return msg


def registry_from_env(orch):
    """★도구 종류 레지스트리를 **env에서 기계 도출**한다 — A2에 이름을 적지 않는다.
    env는 discoverable 도구를 `DISCOVERABLE_ATTR` 속성으로 표시한다
    (`tau2/domains/*/tools.py` · C208 판정: *"env 레지스트리에서 기계-도출되므로 opex 0"*).
    ⇒ **capex/opex 0 · 새 ABox로 자동 전이**([[05]]). 도출 실패 시 빈 집합 = 레버 무발화."""
    agent_d, user_d = set(), set()
    try:
        from tau2.environment.toolkit import DISCOVERABLE_ATTR
    except Exception:
        try:
            from tau2.environment.tool import DISCOVERABLE_ATTR
        except Exception:
            DISCOVERABLE_ATTR = "__discoverable__"
    env = getattr(orch, "environment", None)
    for holder, dst in ((getattr(env, "tools", None), agent_d),
                        (getattr(env, "user_tools", None), user_d)):
        objs = holder if isinstance(holder, (list, tuple, set)) else [holder]
        for o in objs:
            if o is None:
                continue
            target = o
            for attr in ("_toolkit", "toolkit", "__self__"):
                target = getattr(target, attr, target)
            for nm in dir(target):
                if nm.startswith("_"):
                    continue
                try:
                    meth = getattr(target, nm)
                except Exception:
                    continue
                if getattr(meth, DISCOVERABLE_ATTR, False):
                    dst.add(nm)
    return agent_d, user_d


def registry_from_a2(a2):
    """도구 종류 레지스트리 — A2/env 선언에서 도출한다(우리가 이름을 적지 않는다).
    반환 (scaffold, agent_disc, user_disc). 선언이 없으면 빈 집합 → 레버 무발화."""
    scaffold, agent_d, user_d = set(), set(), set()
    if not isinstance(a2, dict):
        return scaffold, agent_d, user_d
    for t in (a2.get("scaffold_get_tools") or []):
        if isinstance(t, dict) and t.get("name"):
            scaffold.add(str(t["name"]))
    reg = a2.get("tool_registry") or {}
    for k, dst in (("agent_discoverable", agent_d), ("user_discoverable", user_d)):
        for n in (reg.get(k) or []):
            dst.add(str(n))
    return scaffold, agent_d, user_d


def channel_note(name, args, scaffold, agent_d, user_d, unlocked, tpl):
    """축 3⒜ — 요청된 도구의 **종류**가 호출 채널과 어긋나는가(집합 멤버십만·판단 0)."""
    want = str(args.get("agent_tool_name") or args.get("discoverable_tool_name") or "").strip()
    if not want:
        return None
    if name == "unlock_discoverable_agent_tool":
        if want in scaffold:
            return tpl.get("is_scaffold", "").format(tool=want) or None
        if want in user_d and want not in agent_d:
            return tpl.get("is_user_tool", "").format(tool=want) or None
    if name == "give_discoverable_user_tool" and want in agent_d and want not in user_d:
        return tpl.get("is_agent_tool", "").format(tool=want) or None
    if name == "call_discoverable_agent_tool":
        if want in user_d and want not in agent_d:
            return tpl.get("is_user_tool", "").format(tool=want) or None
        if want not in unlocked and want in agent_d:
            return tpl.get("not_unlocked", "").format(tool=want) or None
    return None


def mention_note(said_text, called_names, agent_d, user_d, unlocked, tpl):
    """폭주⒝ — 본문에서 **레지스트리 이름**을 말했는데 그 경로로 부르지 않았나.
    술어 = 유한 열거 멤버십(정규식으로 이름을 *추출*하지 않는다·C279 R5 계보)."""
    out = []
    for nm in sorted(agent_d | user_d):
        if not nm or nm in called_names:
            continue
        if nm not in (said_text or ""):
            continue
        if nm in agent_d and nm not in unlocked:
            m = tpl.get("mentioned_agent", "").format(tool=nm)
        elif nm in user_d:
            m = tpl.get("mentioned_user", "").format(tool=nm)
        else:
            continue
        if m:
            out.append(m)
    return out[:2]


def fit_diff_note(text, tpl, max_fields=8):
    """축 2 — fit 반환이 후보 ≥2면 **값이 갈리는 필드만** 비교표로. 순위·추천 금지([[10]])."""
    cards = re.findall(r"'card': '([^']+)'", text or "")
    if len(cards) < 2:
        return None
    facts = []
    for m in re.finditer(r"'card': '([^']+)', 'facts': \{(.*?)\}", text or "", re.S):
        d = {}
        for km in re.finditer(r"'([a-z_()' ]+)': ([^,}]+)", m.group(2)):
            d[km.group(1).strip("' ")] = km.group(2).strip()
        facts.append((m.group(1), d))
    if len(facts) < 2:
        return (tpl.get("multi_plain", "") or "").format(n=len(cards)) or None
    keys = set()
    for _c, d in facts:
        keys |= set(d)
    diff = [k for k in sorted(keys) if len({d.get(k) for _c, d in facts}) > 1]
    if not diff:
        return (tpl.get("no_diff", "") or "").format(n=len(cards)) or None
    rows = []
    for c, d in facts:
        rows.append("%s: %s" % (c, ", ".join("%s=%s" % (k, d.get(k, "-"))
                                             for k in diff[:max_fields])))
    return (tpl.get("diff", "") or "").format(n=len(cards), fields=", ".join(diff[:max_fields]),
                                              table=" | ".join(rows)) or None


_PLURAL = ("s", "es", "list", "ids", "names", "types", "reasons")


def scalar_array_note(args, tpl):
    """축 3⒞ 배치화 — **단수 이름** 필드에 배열이 왔는가(복수형 파라미터는 정상)."""
    hits = []

    def scan(d, path=""):
        if not isinstance(d, dict):
            return
        for k, v in d.items():
            if isinstance(v, str) and v.strip()[:1] == "{":
                try:
                    scan(json.loads(v), path + str(k) + ".")
                    continue
                except Exception:
                    pass
            if isinstance(v, dict):
                scan(v, path + str(k) + ".")
            elif isinstance(v, list) and len(v) > 1:
                nm = str(k)
                if not any(nm.lower().endswith(p) for p in _PLURAL):
                    hits.append((path + nm, len(v)))
    scan(args)
    if not hits:
        return None
    f, n = hits[0]
    return (tpl.get("scalar_array", "") or "").format(field=f, n=n) or None


def terminal_turn_note(user_text, tokens, called_transfer, tpl):
    """축 1ⓐ — 손님 발화에 **이관 토큰이 축자로** 있는데 아직 호출이 없다."""
    if called_transfer or not user_text:
        return None
    if not any(t and t in user_text for t in (tokens or [])):
        return None
    return tpl.get("already_asked", "") or None
