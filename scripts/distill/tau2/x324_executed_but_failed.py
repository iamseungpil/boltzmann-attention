# -*- coding: utf-8 -*-
"""x324 — `MISS-EXECUTED` 규명: **성공 실행인데 gold 액션 체크가 False** 인 자리.

물음: 우리 포렌식의 인자 정규화가 **너무 헐렁해서** 같다고 부르는가(=우리 계기 결함),
아니면 하네스가 **순서/상태**를 함께 보기 때문인가(=진짜 결손)?

방법: 해당 gold `action_check` 의 원문과, 궤적서 같은 도구를 부른 **모든** 호출의 원문 인자를
나란히 찍는다. 판정은 하지 않는다 — 문자열을 그대로 보여준다([[55]] 계기 먼저).

사용: py x324_executed_but_failed.py <tag> [<tag>...]
"""
import io
import json
import sys

try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
except Exception:
    pass

import t2_forensic as F                                            # noqa: E402


def norm(a):
    """비교용 정규화(우리 포렌식이 쓰는 것과 같은 취지: 중첩 문자열 인자를 푼다)."""
    if isinstance(a, str):
        try:
            a = json.loads(a)
        except Exception:
            return a
    if isinstance(a, dict):
        return {k: norm(v) for k, v in sorted(a.items())}
    return a


def run(tags):
    for tag in tags:
        print("#" * 92)
        print("# %s" % tag)
        for s in F.sims(tag):
            key = F.sim_key(s)
            checks = (s.get("reward_info") or {}).get("action_checks") or []
            if not checks:
                continue
            # 궤적의 (대상도구 → 원문 인자 리스트)
            seen = {}
            for i, (_m, tc) in enumerate(F.calls(s)):
                args = F.argsof(tc)
                tgt = F.inner_name(args) or F.nameof(tc)
                inner = args.get("arguments", args)
                seen.setdefault(tgt, []).append((i, inner))
            for ck in checks:
                if ck.get("action_match") is not False:
                    continue
                act = ck.get("action") or {}
                ar = act.get("arguments") or {}
                tgt = (ar.get("agent_tool_name") or ar.get("user_tool_name")
                       or ar.get("tool_name") or act.get("name"))
                got = seen.get(str(tgt))
                if not got:
                    continue                       # NOTCALLED 는 이 프로브의 대상이 아니다
                gold_inner = ar.get("arguments", ar)
                gn = norm(gold_inner)
                hits = [(i, g) for i, g in got if norm(g) == gn]
                if not hits:
                    continue                       # ARGDIFF 는 대상 아님
                print("-" * 92)
                print("%s  도구=%s" % (key, tgt))
                print("  gold  : %s" % json.dumps(gn, ensure_ascii=False, sort_keys=True)[:400])
                print("  일치호출: %s" % [i for i, _g in hits])
                # ★원문 그대로 — 정규화가 지워버릴 수 있는 차이(표기·타입)를 보이기 위해.
                print("  gold원문: %r" % (gold_inner,))
                for i, g in hits:
                    print("  got[%d] : %r" % (i, g))
                print("  체크원문: %s" % json.dumps(
                    {k: v for k, v in ck.items() if k != "action"}, ensure_ascii=False)[:600])


if __name__ == "__main__":
    run(sys.argv[1:] or ["bank_t7295_a_20260815n", "bank_t7295_b_20260815n"])
