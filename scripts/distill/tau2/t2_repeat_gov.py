#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""t2_repeat_gov: **반복 거버너** — 반복 채널 5겹의 문구·순서를 한 곳에서 조립.

정본 = `LEVER_CONSOLIDATION_DESIGN_2026_08_02` §2a (2026-08-02 사용자 리뷰 L1 반영판).
★L1 규율: 이 모듈은 **행동 변경이 아니다** — 판정 술어(cache·seen·loop_k·replay-safe·크기 게이트)는
호출자(t2_gate_patch)가 현행 그대로 계산해 넘기고, 거버너는 **문구 조립과 사다리 순서만** 소유한다.
문구는 A2 `repeat_governor` 블록에서 읽되, **기본값 = 현행 리터럴 축자**(A2 부재 시 바이트 동일).

플래그: `T2_REPEAT_GOV=1`일 때만 호출자가 이 경로를 탄다(기본 OFF = 레거시 조립 그대로).
동등성: `test_repeat_gov.py`가 전 단에서 레거시 문자열과 **바이트 동일**을 검정하고,
교체 전 x45가 코퍼스 수준에서 재확인한다(§6 순서).
"""
import json
import os

_HERE = os.path.dirname(os.path.abspath(__file__))

# ── 기본 문구 = 현행 리터럴 **축자** (t2_gate_patch.py 2026-08-02 HEAD 기준) ────
_DEFAULTS = {
    "stub": ("[DUPLICATE-READ] This exact call (same tool, same arguments) was "
             "already executed earlier in this conversation; its full output is "
             "shown above and has not changed. Refer to that output instead of "
             "re-reading."),
    "stub_digested": ("[DUPLICATE-READ] This exact call was already executed earlier; its "
                      "output was COMPACTED from view to save space and has not changed. Do "
                      "NOT re-run it. If a tool needs that data, pass it BY REFERENCE as "
                      "@last:{tool} instead of re-reading."),
    "redirect": (" Do NOT repeat this exact search. If you are looking up a discoverable "
                 "tool, note that a bare function-name query matches no document text — "
                 "search PLAIN WORDS describing the action/step (the everyday words a policy "
                 "document would use), not the tool's function name. If you already have the "
                 "information you need, proceed to the next step instead of searching again."),
    "escalate": (" You have now issued this IDENTICAL call {n} times and the result "
                 "has not changed once — repeating it again cannot produce new "
                 "information. Change what you do: use DIFFERENT search words, or "
                 "act on the information you already have, or ask the customer. Do "
                 "not issue this same call again."),
    "cap": (" [REPEAT-CAP] This identical call has now been issued {n} times and is "
            "no longer being executed. Stop this line of action: state to the "
            "customer what you could not resolve, or take a DIFFERENT action. "
            "This has been recorded as an unresolved blocker."),
}

_CFG = None


def _phrases():
    """A2 L1(base/shared.json) `repeat_governor` 블록 — 부재/키 누락 시 기본값(=현행 축자)."""
    global _CFG
    if _CFG is None:
        cfg = {}
        try:
            p = os.path.join(_HERE, "a2", "base", "shared.json")
            with open(p, encoding="utf-8") as f:
                cfg = (json.load(f) or {}).get("repeat_governor") or {}
        except Exception:
            cfg = {}
        _CFG = {k: cfg.get(k) or v for k, v in _DEFAULTS.items()}
    return _CFG


def ladder(tool_name, n_rep, is_search, digested, cap_k):
    """사다리 한 번의 판정 — **문구 조립·순서만**(판정 술어는 호출자 소유·L1).

    입력(전부 호출자가 현행 술어로 계산):
      tool_name  : 도구 이름(다이제스트 탈출구 문구용)
      n_rep      : 이 (도구,인자) 키의 스텁 누적 횟수(호출자 `_t2_dup_rep` — 현행과 동일 시점에 +1)
      is_search  : 검색류 판정(현행: 이름·인자에 search/bm25/kb_/grep)
      digested   : 뷰-압축 다이제스트 키(현행: k∈cache ∧ cache[k]∈_dgset ∧ NO_DIGEST_REEXEC=1)
      cap_k      : 캡 K(0/None=비활성·현행 T2_REPEAT_CAP·정본은 RUNAWAY §2c)
    반환: (content, error_flag, capped)
    사다리(CONSOLIDATION §2a L1 명세 순서): 본문(다이제스트/일반) → +redirect(검색) → +esc/cap(n 기준).
    """
    ph = _phrases()
    body = (ph["stub_digested"].format(tool=tool_name or "") if digested
            else ph["stub"])
    redir = ph["redirect"] if is_search else ""
    esc = ""
    capped = False
    if cap_k and cap_k > 3 and n_rep >= cap_k:
        esc = ph["cap"].format(n=n_rep)
        capped = True
    elif n_rep >= 3:
        esc = ph["escalate"].format(n=n_rep)
    return body + redir + esc, (n_rep >= 3), capped
