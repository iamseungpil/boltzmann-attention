# -*- coding: utf-8 -*-
r"""x488 — **093: 우리 deny 가 크레딧 write 를 막는가** 격리 A/B/N/C (2026-08-23·[[62]]·사용자 지시)

## 왜 (t7346 093 = 0/2 · 변이 = `missing apply_savings_account_credit_6831`)
스모크(reward 1.0)는 크레딧을 실행했고 본런 두 sim 은 **한 번도 실행하지 않았다**. 로그 축자:
```
[T2_RESOLVE_CAP] 리셋(실행): 새 실행 ['unlock_discoverable_agent_tool']
[T2_OUR_NAMES]   등재 name=apply_savings_account_credit_6831
[T2_DISCOVERY_STEP2] deny name=apply_savings_account_credit_6831 (…**미unlock**…)
[T2_RESOLVE] deny tool=unlock_discoverable_agent_tool arg=agent_tool_name reason=operator-scope
```
**"unlock 안 됐으니 못 쓴다" 와 "unlock 도 못 한다" 를 동시에 말한 교착**이다([[64]] 고칠 길이
구조적으로 막힌 거부). 건수: 본런 s626729 4건 · s373753 4건 ↔ **통과 스모크 0건**.

`operator-scope` 는 `t2_resolve.py:221` 에서 난다 — `find_intent` 형식화기가 뽑은 `want` 와
모델이 고른 `chosen` 이 다르고 둘 다 아직 미실행이면 **unlock 을 deny** 하고 두 후보의 범위를
나란히 인쇄한다. 그런데 이 태스크에서 그 둘은 대안이 아니다 — 정책이 **공동-요구**로 선언한다:
```
doc_bank_accounts_bank_accounts_(general)_043 (Internal: Applying Credits to Savings Accounts)
  "Use the apply_savings_account_credit_6831 tool to add a credit transaction to a savings account."
  "For interest corrections, after applying the credit, you should also submit an interest
   discrepancy report using submit_interest_discrepancy_report_7294 …"
```
⇒ gold 를 보지 않고 정책만으로 도출된다([[23]] 통과).

## 무엇을 가르나 ([[62]] — 레버·선언을 짓기 **전에**)
결정점 = 보정액이 계산된 직후(`get_interest_correction` 뒤)의 **다음 assistant 발화**.
    "우리 문구가 막은 것" → B_scope 에서만 크레딧이 사라진다 ⇒ 처방은 **우리 deny 를 좁히는 것**
    "모델이 애초에 안 한다" → A_free 도 크레딧 0 ⇒ deny 는 무죄이고 원인은 다른 데 있다
    "정책을 주면 된다"     → C_policy 가 A 보다 크게 오르면 처방은 **전달**([[62]]②)

## 팔 (한 변수만 다르다 · 주입 자리는 문맥 맨 끝 user 메시지 = 라이브 비커밋 채널과 동형)
    A_free    결정점까지 궤적 축자 · 주입 0        — 우리 deny 가 **없는** 조건
    B_scope   + 라이브가 실제로 보낸 그 문구       — `t2_resolve.OPERATOR_SCOPE_FB` 를 실제
              후보쌍·실제 도구 설명으로 채워 **엔진 자신이 만들게** 한다(내가 쓰지 않는다)
    N_neg     + 무내용 한 줄                       — [[57]] 부정통제
    C_policy  + doc_043 **전문**(코퍼스 축자)       — 정책 전달만으로 풀리나

## ★재현 게이트는 A 가 아니라 **B** 다 (x485·x487 과 다르다 — 이유가 있다)
라이브가 받은 것은 *"deny 문구가 있는 문맥"* 이므로, 격리가 재현해야 할 것은 **B_scope** 이고
거기서 크레딧이 안 나와야 계기가 산다. A_free 는 *"그 문구가 없었다면"* 의 반사실이라 라이브에
대응물이 없다(C596: 우리 문면은 비커밋이라 궤적에 없다 — 그래서 A 는 자동으로 '덜 지시된' 문맥이고,
여기서는 그것이 **결함이 아니라 처치**다).

## 채점 (닫힌 술어·gold 무접촉)
후보쌍은 **우리 로그**의 `operator-scope: … (A, B)` 줄에서 읽는다(엔진 자신의 산출·리터럴 0).
    credit  다음 호출이 unlock/call 로 **앞 후보**(크레딧)를 겨눔
    report  뒤 후보(보고서)를 겨눔 · other_tool · no_tool
어느 쪽이 옳은지 이 파일은 모른다 — 계수만 낸다([[69]] 성적은 본런 reward 에서만).

## [[71]] 4문
  1) 기능 하나 — 각 재생은 **다음 발화 하나**. 채점은 바깥에서 이름 대조.
  2) 재료는 선언·엔진 산출에서 — 후보쌍=우리 로그 · deny 문구=`t2_resolve` 상수+실제 도구 설명 ·
     정책문=코퍼스 문서. 이 파일에 도구명·문서 id 리터럴 0.
  3) 전달 = 도출된 id → 코퍼스 정확 집기(검색·유사도 0).
  4) 엔진 해석·선택·순위 0 — 문서 전문·엔진 자기 문구를 그대로 싣는다(argmax·"정답은 X" 0).

## 실행
    cd /home/woori/scratch/tau2-bench && \
    PYTHONPATH=src:scripts/distill/tau2 PYTHONIOENCODING=utf-8 \
    /home/woori/venvs/seka_env/bin/python .../x488_093_unlock_deny_iso.py --port 8141
    # 배선만(LLM 0·GPU 불요): ... --wiring-only
"""
import argparse
import copy
import io
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import t2_forensic as F                     # noqa: E402
import t2_resolve as R                      # noqa: E402  deny 문구 정본(내가 쓰지 않는다)
import x465_transfer_doc_iso as X465        # noqa: E402  재생 관용구(사본 0·[[67]])

REP = os.path.abspath(os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026"))
NLC = chr(10)
TASK = "task_093"
WRAP = ("unlock_discoverable_agent_tool", "call_discoverable_agent_tool",
        "give_discoverable_user_tool", "call_discoverable_user_tool")

RE_SCOPE = re.compile(r"\[T2_RESOLVE\] operator-scope: [^(]*\(([^)]*)\)")
NUDGE = "[NOTICE] Please continue with the customer's request."
POLICY_HEAD = ("[KB DELIVERY] Read the following before choosing your next action. This is, in full "
               "and verbatim, the knowledge-base document that defines how this credit is applied.")


def scope_pairs(tag, key):
    """그 sim 에서 **우리 엔진이 인쇄한 후보쌍** — 로그 축자. 이 파일에 도구명 리터럴 0."""
    try:
        txt = F.log_text(tag)
    except Exception:
        txt = ""
    out = []
    for ln in txt.splitlines():
        if key not in ln:
            continue
        m = RE_SCOPE.search(ln)
        if m:
            pair = tuple(x.strip() for x in m.group(1).split(",") if x.strip())
            if len(pair) == 2 and pair not in out:
                out.append(pair)
    return out


def anchor(sim, names):
    """결정점 = 그 sim 에서 **뒤 후보(보고서)를 실제로 실행한** assistant 발화 index.

    닫힌 술어(래퍼 해제 후 이름 일치)만 — 발화의 뜻은 판정하지 않는다([[59]]).
    라이브는 그 자리에서 크레딧 대신 보고서를 골랐다. 우리는 그 한 발화를 다시 만든다.
    """
    for i, m in enumerate(sim.get("messages") or []):
        if str(m.get("role") or "") != "assistant":
            continue
        for tc in (m.get("tool_calls") or []):
            a = F.argsof(tc)
            inner = str(a.get("agent_tool_name") or a.get("user_tool_name") or "")
            if (inner or str(F.nameof(tc))) in names:
                return i
    return None


def classify(calls, credit, report):
    for nm, ag in calls:
        ag = ag or {}
        inner = str(ag.get("agent_tool_name") or ag.get("user_tool_name") or "")
        target = inner or str(nm or "")
        if target == credit:
            return "credit"
        if target == report:
            return "report"
    if calls:
        return "other_tool"
    return "no_tool"


def build(msgs, cut, extra=None):
    ctx = copy.deepcopy(msgs[:cut])
    if extra:
        ctx.append({"role": "user", "content": extra})
    return ctx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8141)
    ap.add_argument("--model", default="Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8")
    ap.add_argument("--tag", default="bank_t7346_halfA_20260822")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--arms", default="A_free,B_scope,N_neg,C_policy")
    ap.add_argument("--maxchars", type=int, default=90000)
    ap.add_argument("--wiring-only", action="store_true")
    ap.add_argument("--out", default="x488_093_unlock_deny_iso.json")
    a = ap.parse_args()

    import x448_index_vs_all_iso as IVA
    sb = IVA.Sandbox()
    tools = list(sb.env.get_tools() or [])
    names = {t.name for t in tools}
    print("=" * 100)
    print("x488 · env 도구 %d종 · 재생 = 실물 스키마 + 메시지 객체 + la.generate (C584)" % len(tools))

    import t2_search as TS
    corpus = TS.corpus_from_env(sb.env) or {}

    srcs = []
    for s in F.sims(a.tag, suffix=".results.json.gz"):
        if str(s.get("task_id")) != TASK:
            continue
        key = F.simtag(s) or ""
        pairs = scope_pairs(a.tag, key)
        pair = next((p for p in pairs if p[0] in names and p[1] in names), None)
        if not pair:
            print("  ⚠건너뜀 %s — 우리 로그에 후보쌍이 없다/레지스트리 불일치: %r" % (key, pairs))
            continue
        credit, report = pair
        i = anchor(s, {report})
        if i is None:
            print("  ⚠건너뜀 %s — 뒤 후보 실행 발화를 못 찾았다([[55]])" % key)
            continue
        srcs.append({"key": key, "sim": s, "credit": credit, "report": report, "cut": i})
        print("  원천 %-22s 후보쌍=(%s, %s) · 결정점=[%d]" % (key, credit, report, i))
    if not srcs:
        raise SystemExit("원천 0 — 태그·로그부터 본다([[55]])")

    # ── deny 문구는 **엔진이 만든다** — 상수 + 실제 도구 설명(내가 쓰지 않는다·[[71]]②) ──
    class _Holder(object):
        def __init__(self, t):
            self.name, self.description = t.name, getattr(t, "description", "") or ""

    class _FakeAgent(object):
        tools = [_Holder(t) for t in tools]
        _t2_orch = None

    for s in srcs:
        _sc = [(n, R._tool_scope(_FakeAgent(), n)) for n in (s["credit"], s["report"])]
        _sc = [(n, d) for n, d in _sc if d]
        if len(_sc) < 2:
            raise SystemExit("도구 설명을 못 읽었다 — 레지스트리부터([[55]]): %r" % (_sc,))
        s["scope_fb"] = R.OPERATOR_SCOPE_FB.format(
            chosen=s["credit"], scopes="; ".join("'%s' = %s" % (n, d) for n, d in _sc))
        # 정책문: 크레딧 도구 이름을 담은 코퍼스 문서 **전부**(선택·순위 0·검색 0)
        ids = sorted(d for d, body in corpus.items() if s["credit"] in str(body or ""))
        blob, missing = sb.cat(list(ids))
        for did in list(missing):
            if corpus.get(did):
                blob += NLC + NLC + "### %s%s%s" % (did, NLC, corpus[did])
                missing.remove(did)
        s["policy_ids"], s["policy"] = ids, blob[:a.maxchars]
        print("  %s · deny 문구 %d자 · 정책 문서 %d편 %d자 %s"
              % (s["key"][-12:], len(s["scope_fb"]), len(ids), len(s["policy"]),
                 ",".join(ids)[:80]))
        if missing:
            print("    ⚠집기 실패(조용히 넘기지 않는다): %r" % (sorted(missing),))
    if a.wiring_only:
        print(NLC + "[배선] wiring-only 종료 — LLM 0·GPU 0")
        return 0

    arms = [x.strip() for x in a.arms.split(",") if x.strip()]
    base = "http://localhost:%d/v1" % a.port
    rows = []
    for s in srcs:
        msgs = s["sim"].get("messages") or []
        extra = {"A_free": None, "B_scope": s["scope_fb"], "N_neg": NUDGE,
                 "C_policy": POLICY_HEAD + NLC + NLC + s["policy"]}
        for arm in arms:
            ctx = build(msgs, s["cut"], extra.get(arm))
            print(NLC + "── %s / %-9s (컷=[%d] 주입=%d자) ────────"
                  % (s["key"][-12:], arm, s["cut"], len(extra.get(arm) or "")))
            for k, t in enumerate([0.0] + [a.temperature] * a.n):
                try:
                    r = X465.replay(ctx, tools, a.model, base, t)
                except Exception as e:
                    print("  #%d EXC %r" % (k, e))
                    rows.append({"src": s["key"], "arm": arm, "k": k, "cat": "EXC"})
                    continue
                cat = classify(r.calls, s["credit"], s["report"])
                rows.append({"src": s["key"], "arm": arm, "k": k, "temp": t, "cat": cat,
                             "calls": [[nm, json.dumps(ag, ensure_ascii=False,
                                                       default=str)[:160]] for nm, ag in r.calls],
                             "text": " ".join(r.text.split())[:240]})
                print("  #%d t=%.1f  %-10s %s" % (k, t, cat,
                                                  ",".join(nm for nm, _ in r.calls) or "-"))

    cats = ["credit", "report", "other_tool", "no_tool", "EXC"]
    print(NLC + "=" * 100)
    print("%-10s %s" % ("팔", " ".join("%-12s" % c for c in cats)))
    for arm in arms:
        rs = [r for r in rows if r["arm"] == arm]
        if not rs:
            continue
        print("%-10s %s (n=%d)" % (arm, " ".join(
            "%-12s" % ("%d/%d" % (sum(1 for r in rs if r["cat"] == c), len(rs))) for c in cats),
            len(rs)))
    p = os.path.join(REP, a.out)
    with io.open(p, "w", encoding="utf-8") as f:
        json.dump({"task": TASK, "tag": a.tag,
                   "sources": [{"key": s["key"], "credit": s["credit"], "report": s["report"],
                                "cut": s["cut"], "scope_fb": s["scope_fb"],
                                "policy_ids": s["policy_ids"],
                                "policy_chars": len(s["policy"])} for s in srcs],
                   "rows": rows}, f, ensure_ascii=False, indent=1)
    print(NLC + "★재현 게이트는 **B_scope** 다 — 거기서 credit 이 낮아야 계기가 산다(A 가 아니다·머리말).")
    print("  A_free 가 credit 을 내고 B_scope 가 못 내면 → **우리 deny 가 그 write 를 막았다**.")
    print("  A_free 도 credit 0 이면 → deny 는 무죄, 원인은 다른 층이다([[55]] 다음 칸으로).")
    print("  C_policy 가 A 보다 크게 오르면 → 처방 축은 **전달**이다([[62]]②).")
    print("  N_neg 이 A 와 같아야 '뭐라도 찔러서'가 아니라 **내용**이라 말할 수 있다([[57]]).")
    print("→ %s" % p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
