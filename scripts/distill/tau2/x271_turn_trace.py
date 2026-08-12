# -*- coding: utf-8 -*-
r"""x271 — 한 sim 을 **턴별로** 정독한다 (사용자 지시 2026-08-12: *"완료된 sim 만 정밀 추적
포렌식하라. 원인 확정하고 대책 마련하라"*).

집계·마커로는 원인이 안 나온다([[08]]). 그래서 한 줄에 하나씩 —

  · 손님/어시스턴트 발화 요지 · 도구 호출(이름 + 핵심 인자)
  · **우리 층이 그 자리에 넣은 문장**(tool 메시지의 error 본문 = 실제로 모델이 읽은 것)
  · 그 값이 공식 명칭 집합의 원소인지 (write 인자에 한해)

⚠**비커밋 채널은 궤적에 없다**(C298·`reminder-user`) — 그건 사이드카(`fb_<tag>.jsonl`)에만
  남으므로 §2 에서 따로 인쇄한다. 이 구분을 놓치면 *"우리가 말한 적 없다"* 로 오독한다(C369).

실행: python x271_turn_trace.py <tag> [task_id] [trial]
"""
import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from x266_decide_ask_axis import a2 as _a2                          # noqa: E402

SIMS = os.environ.get("X271_SIMDIR", "/home/woori/scratch/tau2-bench/data/simulations")
LOGS = os.environ.get("X271_LOGS", "/home/woori/scratch/logs")


def names():
    di = (_a2().get("policy_ontology") or {}).get("doc_index") or {}
    return {g: {" ".join(w.capitalize() for w in str(k).split("_")) for k in s}
            for g, s in di.items()}


def brief(t, n=110):
    return " ".join(str(t or "").split())[:n]


def main():
    tag = sys.argv[1]
    want_task = sys.argv[2] if len(sys.argv) > 2 else None
    want_tr = int(sys.argv[3]) if len(sys.argv) > 3 else None
    NAMES = names()
    d = json.load(io.open(os.path.join(SIMS, tag, "results.json"), encoding="utf-8"))

    for si, s in enumerate(d["simulations"]):
        if want_task and s["task_id"] != want_task:
            continue
        ri = s.get("reward_info")
        if ri is None:
            continue                       # 미완 sim 은 건너뛴다
        if want_tr is not None and si != want_tr:
            continue
        fails = [(a.get("action") or {}).get("action_id")
                 for a in (ri.get("action_checks") or []) if not a.get("action_match")]
        print("=" * 104)
        print("%s  sim#%d  reward=%.1f  실패칸=%s  종료=%s"
              % (s["task_id"], si, ri.get("reward") or 0, fails or "없음",
                 s.get("termination_reason")))
        print("-" * 104)
        for i, m in enumerate(s.get("messages") or []):
            role = m.get("role")
            c = m.get("content")
            tcs = m.get("tool_calls") or []
            if tcs:
                for tc in tcs:
                    args = tc.get("arguments") or {}
                    nm = tc.get("name")
                    inner = args.get("arguments")
                    try:
                        inner = json.loads(inner) if isinstance(inner, str) else (inner or {})
                    except Exception:
                        inner = {}
                    extra = ""
                    if isinstance(inner, dict) and inner.get("account_class") is not None:
                        cls = str(inner.get("account_class"))
                        typ = str(inner.get("account_type"))
                        grp = {"business_checking": "business_checking_accounts",
                               "business_savings": "business_savings_accounts",
                               "checking": "checking_accounts",
                               "savings": "savings_accounts"}.get(typ)
                        mem = cls in NAMES.get(grp, set()) if grp else None
                        extra = "  class=%r type=%s %s" % (
                            cls, typ, "✅集內" if mem else ("❌集外" if mem is False else "·축미상"))
                    tgt = args.get("agent_tool_name") or ""
                    print("[%3d] %-9s CALL %s%s%s"
                          % (i, role, nm, (" → " + str(tgt)) if tgt else "", extra))
            elif role == "tool":
                err = m.get("error")
                mark = "‼OUR-DENY" if (err and str(c or "").lstrip().startswith("Error:")) else "  tool"
                print("[%3d] %-9s %s %s" % (i, role, mark, brief(c, 150)))
            elif c:
                print("[%3d] %-9s %s" % (i, role, brief(c, 140)))

        print("\n§2 비커밋 채널 — 사이드카에만 남는 것 (궤적엔 없다)")
        p = os.path.join(LOGS, "fb_%s.jsonl" % tag)
        n = 0
        if os.path.exists(p):
            for line in io.open(p, encoding="utf-8", errors="replace"):
                try:
                    o = json.loads(line)
                except Exception:
                    continue
                if o.get("kind") in ("reminder-user", "reminder-assistant") and o.get("text"):
                    n += 1
                    print("   turn=%s kind=%s | %s" % (o.get("turn"), o.get("kind"),
                                                       brief(o.get("text"), 150)))
        print("   (%d건)" % n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
