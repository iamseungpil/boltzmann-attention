# -*- coding: utf-8 -*-
r"""x489 — **093: 막으면 못 하고, 안 막으면 하는가** 게이팅 롤아웃 (2026-08-23·[[62]]·사용자 지시)

## 왜 이 프로브가 또 필요한가 (x488 이 답하지 못한 것)
x488 은 **문구**를 갈랐고 내 가설을 반증했다 — `OPERATOR_SCOPE_FB` 는 크레딧을 억제하지 않고
오히려 민다(A_free 5/12 → B_scope **11/12**, C_policy 12/12, N_neg 4/12 = A 와 동일).
그런데 x488 의 `credit` 은 *"실행됐다"* 가 아니라 **"모델이 그것을 골랐다"** 일 뿐이다 — 재생은
unlock 상태를 강제하지 않으므로 라이브의 **게이팅**을 재현하지 않는다. 라이브에서 실제로 일어난 것은:
```
[T2_DISCOVERY_STEP2] deny name=apply_savings_account_credit_6831 (…미unlock…)   ← 호출을 막고
[T2_RESOLVE] deny tool=unlock_discoverable_agent_tool reason=operator-scope       ← unlock 도 막는다
```
환경도 같은 말을 한다(실측): 미unlock 호출 → `Error: Tool '…' has not been unlocked. You must
first use 'unlock_discoverable_agent_tool' …`. 즉 **모델이 고르는 것과 실행되는 것은 다른 사건**이고,
093 의 gold 변이는 *"실행 안 됨"* 이다([[69]] 채점 단위).

## 그래서 무엇을 하나 — 문맥 재생이 아니라 **환경을 물린 롤아웃**
프리픽스의 도구 호출을 **실제 환경에 되돌려** 상태(unlock·검증·잔액)를 세운 뒤, 결정점부터
최대 K 턴을 굴린다. 생성 → 실행 → 결과 부착 → 다시 생성. 종료 = 크레딧이 **성공 실행**되거나 K 소진.
⇒ 종점이 [[69]] 와 같은 단위(**실행**)가 된다.

## 팔 (한 변수 = unlock 을 막는가)
    GATE_OFF     막지 않는다 — 모델이 unlock 하면 그대로 실행
    GATE_ON_TEXT 라이브처럼 막는다: 크레딧 unlock 시도를 **실행하지 않고** 그 발화를 버린 뒤
                 `t2_resolve.OPERATOR_SCOPE_FB`(엔진이 조립)를 붙여 **재생성**한다(라이브 동형)
    GATE_ON_MUTE 같은 차단인데 **무내용 한 줄**만 붙인다 — 차단 자체와 문구를 가른다([[57]])
판정: GATE_OFF 에서 실행되고 GATE_ON_* 에서 안 되면 **우리 차단이 그 write 를 막는다**가 확정된다.
      GATE_OFF 에서도 안 되면 차단은 무죄이고 원인은 다른 층이다([[55]] 다음 칸).
      GATE_ON_TEXT ↔ GATE_ON_MUTE 차이는 **문구 몫**(x488 결과와 대조).

## ⚠이 프로브가 재지 못하는 것 / 무효 조건
  ⓐ **프리픽스 충실도**가 전부다. 새 환경에 프리픽스를 되돌릴 때 원래와 다른 결과가 나오면
     그 롤아웃은 다른 세계다 ⇒ 호출마다 원래 결과의 **오류 여부**와 대조해 `fidelity` 로 인쇄하고,
     불일치가 있으면 그 원천을 **무효로 표시**한다(조용히 넘기지 않는다·[[55]]).
  ⓑ 우리 층의 다른 레버들(비커밋 문면 전부)은 롤아웃에 없다 — C596 이 말한 '덜 지시된 문맥'이다.
     그래서 **절대 수준은 라이브와 같지 않다**. 해석 가능한 양은 **팔 사이의 차이**뿐이다.
  ⓒ user-sim 이 없다 — 손님이 새로 말하지 않는다. 손님 발화가 필요한 국면은 이 프로브 밖이다.

## [[71]] 4문
  1) 기능 하나 — 각 롤아웃은 *"크레딧이 실행되는가"* 하나만 답한다.
  2) 재료는 선언·엔진 산출에서 — 후보쌍은 **우리 로그**의 `operator-scope: … (A, B)` 줄에서,
     deny 문구는 `t2_resolve` 상수 + 실제 도구 설명에서. 이 파일에 도구명 리터럴 0.
  3) 전달 0 — 이 프로브는 아무 재료도 배달하지 않는다(문서·정답 0). 환경만 물린다.
  4) 엔진 해석·선택·순위 0 — 무엇을 부를지는 끝까지 모델이 정한다. 우리는 **막거나 말거나** 뿐.

## 실행
    cd /home/woori/scratch/tau2-bench && \
    PYTHONPATH=src:scripts/distill/tau2 PYTHONIOENCODING=utf-8 \
    /home/woori/venvs/seka_env/bin/python .../x489_093_gate_rollout_iso.py --port 8141
    # 배선만(LLM 0·GPU 불요·프리픽스 충실도까지 검사): ... --wiring-only
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
import t2_resolve as R                      # noqa: E402
import x465_transfer_doc_iso as X465        # noqa: E402  재생 관용구(사본 0·[[67]])
import x488_093_unlock_deny_iso as X488     # noqa: E402  후보쌍·설명 조회 재사용(사본 0)

REP = os.path.abspath(os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026"))
NLC = chr(10)
TASK = "task_093"
UNLOCK = "unlock_discoverable_agent_tool"
MUTE = "[NOTICE] Please continue with the customer's request."


def is_err(text):
    """닫힌 술어 — 환경이 낸 오류인가. 환경 축자 접두사만 본다(뜻 해석 0·[[59]])."""
    return str(text or "").strip().lower().startswith("error")


def replay_prefix(env, msgs, cut):
    """프리픽스의 도구 호출을 **실제 환경에 되돌린다**. 반환 (일치, 전체, 불일치 목록)."""
    rec = {m.get("id"): str(m.get("content") or "")
           for m in msgs if str(m.get("role") or "") == "tool" and m.get("id")}
    ok = tot = 0
    bad = []
    for m in msgs[:cut]:
        if str(m.get("role") or "") != "assistant":
            continue
        for tc in (m.get("tool_calls") or []):
            nm, args = F.nameof(tc), F.argsof(tc)
            tot += 1
            try:
                got = str(env.use_tool(nm, **(args or {})))
            except Exception as e:
                got = "Error: %r" % (e,)
            want = rec.get(tc.get("id"), "")
            if is_err(got) == is_err(want):
                ok += 1
            else:
                bad.append((nm, "재생=%s 원래=%s" % ("ERR" if is_err(got) else "OK",
                                                   "ERR" if is_err(want) else "OK")))
    return ok, tot, bad


def tool_msg(tc, content):
    """궤적과 **같은 모양**의 tool 메시지(사본 0 — 필드는 원 호출에서 가져온다)."""
    return {"role": "tool", "id": tc.get("id"), "content": content,
            "requestor": tc.get("requestor") or "assistant"}


def as_dict(m):
    """생성된 AssistantMessage → 궤적 dict (다음 턴 문맥에 다시 넣기 위해)."""
    if isinstance(m, dict):
        return m
    tcs = []
    for tc in (getattr(m, "tool_calls", None) or []):
        tcs.append({"id": getattr(tc, "id", None), "name": F.nameof(tc),
                    "arguments": F.argsof(tc),
                    "requestor": getattr(tc, "requestor", None) or "assistant"})
    return {"role": "assistant", "content": getattr(m, "content", None) or "",
            "tool_calls": tcs or None}


def rollout(env, ctx, tools, model, base, temp, credit, block, feedback, K, log):
    """한 롤아웃 — 생성→실행→부착. 크레딧이 **성공 실행**되면 즉시 True."""
    ctx = copy.deepcopy(ctx)
    regens = 0
    for turn in range(K):
        r = X465.replay(ctx, tools, model, base, temp)
        am = {"role": "assistant", "content": r.text or "",
              "tool_calls": [{"id": "x489_%d_%d" % (turn, i), "name": nm, "arguments": ag,
                              "requestor": "assistant"}
                             for i, (nm, ag) in enumerate(r.calls)] or None}
        # ★차단 — 라이브 동형: 그 발화를 **커밋하지 않고** 문구를 붙여 재생성한다.
        if block and regens < 3:
            hit = next((tc for tc in (am["tool_calls"] or [])
                        if tc["name"] == UNLOCK
                        and str((tc["arguments"] or {}).get("agent_tool_name")) == credit), None)
            if hit is not None:
                regens += 1
                log.append("turn=%d 차단(regen %d): unlock %s" % (turn, regens, credit))
                if feedback:
                    ctx = ctx + [{"role": "user", "content": feedback}]
                continue
        ctx = ctx + [am]
        if not am["tool_calls"]:
            log.append("turn=%d 산문(도구 0)" % turn)
            continue
        for tc in am["tool_calls"]:
            nm, args = tc["name"], tc["arguments"] or {}
            try:
                out = str(env.use_tool(nm, **args))
            except Exception as e:
                out = "Error: %r" % (e,)
            ctx = ctx + [tool_msg(tc, out)]
            inner = str(args.get("agent_tool_name") or args.get("user_tool_name") or "")
            log.append("turn=%d %s(%s) -> %s" % (turn, nm, inner, "ERR" if is_err(out) else "OK"))
            if inner == credit and not is_err(out):
                return True, turn, regens
    return False, None, regens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8141)
    ap.add_argument("--model", default="Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8")
    ap.add_argument("--tag", default="bank_t7346_halfA_20260822")
    ap.add_argument("--n", type=int, default=3, help="팔당 롤아웃 수(원천마다)")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--turns", type=int, default=5, help="롤아웃 최대 턴")
    ap.add_argument("--arms", default="GATE_OFF,GATE_ON_TEXT,GATE_ON_MUTE")
    ap.add_argument("--wiring-only", action="store_true")
    ap.add_argument("--out", default="x489_093_gate_rollout_iso.json")
    a = ap.parse_args()

    import x448_index_vs_all_iso as IVA
    tools = list(IVA.Sandbox().env.get_tools() or [])
    print("=" * 100)
    print("x489 · env 도구 %d종 · 롤아웃(생성→실행→부착) · 최대 %d턴" % (len(tools), a.turns))

    srcs = []
    for s in F.sims(a.tag, suffix=".results.json.gz"):
        if str(s.get("task_id")) != TASK:
            continue
        key = F.simtag(s) or ""
        pairs = X488.scope_pairs(a.tag, key)
        env0 = IVA.Sandbox().env
        pair = next((p for p in pairs
                     if X488._disc_desc(env0, p[0]) and X488._disc_desc(env0, p[1])), None)
        if not pair:
            print("  ⚠건너뜀 %s — 후보쌍 없음: %r" % (key, pairs))
            continue
        credit, report = pair
        cut = X488.anchor(s, {report})
        if cut is None:
            print("  ⚠건너뜀 %s — 결정점 없음([[55]])" % key)
            continue
        ok, tot, bad = replay_prefix(env0, s.get("messages") or [], cut)
        print("  원천 %-22s 결정점=[%d] · 프리픽스 충실도 %d/%d%s"
              % (key, cut, ok, tot, ("  ⚠불일치 %d: %s" % (len(bad), bad[:3])) if bad else ""))
        srcs.append({"key": key, "sim": s, "credit": credit, "report": report, "cut": cut,
                     "fidelity": [ok, tot], "bad": bad[:6], "valid": not bad})
    if not srcs:
        raise SystemExit("원천 0 — 태그·로그부터 본다([[55]])")

    # deny 문구는 엔진이 만든다(x488 과 같은 경로·내가 쓰지 않는다)
    class _H(object):
        def __init__(self, n, d):
            self.name, self.description = n, d
    for s in srcs:
        e = IVA.Sandbox().env
        hold = [_H(t.name, getattr(t, "description", "") or "") for t in tools]
        hold += [_H(n, X488._disc_desc(e, n)) for n in (s["credit"], s["report"])]

        class _FA(object):
            tools = hold
            _t2_orch = None
        sc = [(n, R._tool_scope(_FA(), n)) for n in (s["credit"], s["report"])]
        sc = [(n, d) for n, d in sc if d]
        s["fb"] = R.OPERATOR_SCOPE_FB.format(
            chosen=s["credit"], scopes="; ".join("'%s' = %s" % (n, d) for n, d in sc))
        print("  %s · deny 문구 %d자" % (s["key"][-12:], len(s["fb"])))
    if a.wiring_only:
        print(NLC + "[배선] wiring-only 종료 — LLM 0·GPU 0 (충실도까지 검사했다)")
        return 0

    arms = {"GATE_OFF": (False, None), "GATE_ON_TEXT": (True, "fb"), "GATE_ON_MUTE": (True, MUTE)}
    base = "http://localhost:%d/v1" % a.port
    rows = []
    for s in srcs:
        msgs = s["sim"].get("messages") or []
        ctx0 = copy.deepcopy(msgs[:s["cut"]])
        for arm in [x.strip() for x in a.arms.split(",") if x.strip()]:
            block, fbk = arms[arm]
            fb = s["fb"] if fbk == "fb" else fbk
            print(NLC + "── %s / %-12s (컷=[%d] 차단=%s) ────────"
                  % (s["key"][-12:], arm, s["cut"], block))
            for k in range(a.n):
                env = IVA.Sandbox().env          # ★롤아웃마다 새 환경(상태 오염 금지)
                ok, tot, bad = replay_prefix(env, msgs, s["cut"])
                log = []
                try:
                    hit, turn, regens = rollout(env, ctx0, tools, a.model, base,
                                                0.0 if k == 0 else a.temperature,
                                                s["credit"], block, fb, a.turns, log)
                except Exception as e:
                    print("  #%d EXC %r" % (k, e))
                    rows.append({"src": s["key"], "arm": arm, "k": k, "credit_exec": None,
                                 "err": repr(e)[:200]})
                    continue
                rows.append({"src": s["key"], "arm": arm, "k": k, "credit_exec": bool(hit),
                             "turn": turn, "regens": regens, "prefix_fidelity": [ok, tot],
                             "trace": log[:24]})
                print("  #%d  credit_exec=%-5s turn=%s regen=%d | %s"
                      % (k, hit, turn, regens, " · ".join(log[-3:])[:120]))

    print(NLC + "=" * 100)
    print("%-14s %-16s %-10s" % ("팔", "credit 실행", "평균 regen"))
    for arm in [x.strip() for x in a.arms.split(",") if x.strip()]:
        rs = [r for r in rows if r["arm"] == arm and r.get("credit_exec") is not None]
        if not rs:
            continue
        hit = sum(1 for r in rs if r["credit_exec"])
        rg = sum(r.get("regens") or 0 for r in rs) / float(len(rs))
        print("%-14s %-16s %-10.1f" % (arm, "%d/%d" % (hit, len(rs)), rg))
    p = os.path.join(REP, a.out)
    with io.open(p, "w", encoding="utf-8") as f:
        json.dump({"task": TASK, "tag": a.tag, "turns": a.turns,
                   "sources": [{k: v for k, v in s.items() if k not in ("sim",)} for s in srcs],
                   "rows": rows}, f, ensure_ascii=False, indent=1)
    print(NLC + "판정: GATE_OFF 에서 실행되고 GATE_ON_* 에서 안 되면 **우리 차단이 그 write 를 막는다**.")
    print("      GATE_OFF 도 0 이면 차단은 무죄 — 원인은 다른 층이다([[55]] 다음 칸).")
    print("      TEXT ↔ MUTE 차이가 **문구 몫**이다(x488: 문구는 오히려 크레딧 쪽으로 밀었다).")
    print("      ⚠프리픽스 충실도 불일치가 있는 원천은 결과를 쓰지 말 것(머리말 ⓐ).")
    print("→ %s" % p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
