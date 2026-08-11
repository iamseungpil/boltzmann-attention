# -*- coding: utf-8 -*-
r"""x250 — **필수 인자를 빈 문자열로 쓴다 · 아무도 안 막는다** (격리 · 유료 0 · 8140 · 새 엔진 0).

## 왜 (010 t2 · C415⒣ → C416⒜ 의 남은 질문)

`bank_kb_20260811` 의 010 t2 는 gold `010_0` 한 칸에서 죽었다 —
`log_verification(..., date_of_birth="", ...)`. 이름·주소·이메일·전화는 **방금 읽은 레코드에서
그대로** 옮겼는데 생일만 비웠다. 값은 대화에 있다(turn 9 축자: `date_of_birth: 04/17/1979`).
직전에 손님이 *"생일은 채팅으로 주기 싫다"*(turn 15) 고 말했고, 에이전트는 그것을 **"우리도 그
값이 없다"** 로 옮긴 것으로 보인다.

**우리 층은 한 번도 울지 않았다.** `write_arg_grounding` 은 *"값 없음 = skip"* 이라(:1149 축자)
**빈 값이 구조적으로 통과**하고, `log_verification` 의 `grounded_args` 는 `time_verified` 하나뿐이다.
즉 *근거 없는 값*을 보는 규칙은 있는데 **필수 인자가 비었다**를 보는 규칙이 없다([[64]] 의 짝
문제: 거부가 *말을 못 하는* 것과 거부가 *울지 않는* 것).

## ⛔0 (레버 전에 결손을 격리로 잰다)

이 프로브가 그 측정이다. **아직 아무것도 짓지 않는다.** 묻는 것은 둘뿐:
  ⑴ 이 자리에서 결손이 재현되는가(A 가 빈 값을 다시 쓰는가) — 아니면 표집이었나.
  ⑵ 재현된다면 **이름만 대는 거부**로 닫히는가(B), 아니면 이름 없는 거부로는 안 되는가(C).
B 가 닫으면 처방은 *엔진이 값을 채우는 것*이 아니라 **거부가 인자 이름을 대는 것** 하나다 —
값은 끝까지 모델이 고른다(레코드는 이미 대화에 있다).

## 팔 (n · 계기 = 다음 도구 호출 하나 · 지시문 없음)

  A_LIVE      라이브 재현(궤적 + 사이드카 축자)                    ← 재현 팔
  B_NAME      + 빈 인자를 **이름으로** 짚는 거부([[64]] 처방)      ← 처방 팔
  C_GENERIC   + 이름 없는 거부(`fix the flagged call first`)       ← 부정 통제(C414 의 그 형태)
  D_FREE      궤적만(우리 문장 0)                                  ← 상한

통과 = 다음 호출이 `log_verification` 이고 `date_of_birth` 가 **레코드 값과 정확히 같은** 것.
부분 점수로 `EMPTY`(빈 값 재현)·`OTHER` 를 따로 센다.

실행(리모트): python x250_empty_required_arg_probe.py [N]
"""
import collections
import json
import os
import re
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
TASK = "task_010"
WRITE = "log_verification"
ARG = "date_of_birth"

# 거부 문구 — B 는 **인자 이름만** 댄다(값 0·출처 지목 0). C 는 현행 형제 문구 축자.
DENY_NAME = ("Error: [ARG-EMPTY] the call to %s left the required argument '%s' as an empty "
             "string. An empty string is not a value. Re-issue the call with %s filled in, or do "
             "not file the record at all." % (WRITE, ARG, ARG))
DENY_GENERIC = "Error: resolve the flagged call(s) first; do not call this tool yet."


def gold_value(sim, upto):
    """레코드 축자에서 그 인자의 값을 **읽기만** 한다 — 채점용이지 모델에 주지 않는다."""
    for m in sim["messages"][:upto]:
        if m.get("role") != "tool":
            continue
        c = str(m.get("content") or "")
        mo = re.search(r"%s:\s*([0-9/\-]+)" % ARG, c)
        if mo:
            return mo.group(1).strip()
    return None


def build(sim, cut, mode):
    import t2_fbsidecar as FB

    class _M(object):
        def __init__(s, r, c):
            s.role, s.content = r, c

    keyed = FB._sim_key([_M(m.get("role"), m.get("content")) for m in sim["messages"]])
    ours = collections.defaultdict(list)
    if mode != "free":
        for ln in open("/home/woori/scratch/logs/fb_%s.jsonl" % TAG,
                       encoding="utf-8", errors="replace"):
            o = json.loads(ln)
            if o.get("sim") == keyed and (o.get("text") or "").strip():
                ours[o.get("turn")].append(" ".join(o["text"].split()))
    out = []
    for i, m in enumerate(sim["messages"][:cut]):
        r, c = m.get("role"), " ".join(str(m.get("content") or "").split())
        tcs = [(tc.get("function") or {}).get("name") or tc.get("name")
               for tc in (m.get("tool_calls") or [])]
        if tcs:
            out.append("[%s calls] %s" % (r, ", ".join(tcs)))
        if c:
            out.append("[%s] %s" % (r, c[:700]))
        for t in ours.get(i, ()):
            out.append("[system] %s" % t[:900])
    return "\n".join(out)


def score(msg, gold):
    for tc in (msg.get("tool_calls") or []):
        f = tc.get("function") or {}
        if (f.get("name") or "") != WRITE:
            continue
        try:
            a = json.loads(f.get("arguments") or "{}")
        except Exception:
            a = {}
        v = str(a.get(ARG, "")).strip()
        if gold and v == gold:
            return "HIT"
        return "EMPTY" if not v else "OTHER(%s)" % v[:12]
    names = [(tc.get("function") or {}).get("name") for tc in (msg.get("tool_calls") or [])]
    return names[0] if names else "(발화만)"


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 8
    def _flat(tc):
        """결과 파일의 tool_call 은 평평하다(`name`/`arguments`=dict) — 응답 형식과 다르다."""
        f = tc.get("function") or {}
        nm = f.get("name") or tc.get("name")
        av = f.get("arguments") if f else tc.get("arguments")
        if isinstance(av, str):
            try:
                av = json.loads(av)
            except Exception:
                av = {}
        return nm, (av or {})

    def _empty_write(s):
        for m in s["messages"]:
            for tc in (m.get("tool_calls") or []):
                nm, a = _flat(tc)
                if nm == WRITE and not str(a.get(ARG, "")).strip():
                    return True
        return False

    cands = [s for s in X.load(TAG)
             if s["task_id"] == TASK and (s.get("reward_info") or {}).get("reward") != 1
             and _empty_write(s)]
    if not cands:
        print("결손 궤적 없음 — 이 프로브는 이 태그에 적용되지 않는다.")
        return
    sim = cands[0]
    cut = None
    for i, m in enumerate(sim["messages"]):
        for tc in (m.get("tool_calls") or []):
            if _flat(tc)[0] == WRITE and cut is None:
                cut = i
    gold = gold_value(sim, cut)
    tools = U.tools_of(sim)
    print("trial %s · 요구 턴 %d · 도구 %d개 · 레코드 값 %r · n=%d\n"
          % (sim.get("trial"), cut, len(tools), gold, n))
    # ★거부는 **시도 뒤에** 온다 (2026-08-11 자기교정·첫 판 무효). 첫 판은 거부 문구를 호출
    #   *이전* 문맥 끝에 붙였고 네 팔이 전부 EMPTY 8/8 로 나왔다 — 그건 처방을 잰 것이 아니라
    #   *예고*를 잰 것이다. 라이브의 반사실은 **모델이 빈 인자로 부른 뒤 우리 층이 되돌리는 것**
    #   이다. 그래서 거부 팔은 ⑴모델 자신의 그 호출을 축자로 싣고 ⑵그 결과 자리에 거부를 놓는다.
    #   (handoff §9 *"프로브 하네스의 상한이 신호를 만든다"* 의 같은 종류 사고다.)
    bad = None
    for tc in (sim["messages"][cut].get("tool_calls") or []):
        if _flat(tc)[0] == WRITE:
            bad = _flat(tc)[1]
    attempt = ("\n[assistant calls] %s\n[assistant] arguments: %s"
               % (WRITE, json.dumps(bad, ensure_ascii=False)))

    for mode, label, extra in (("live", "A_LIVE", None), ("live", "B_NAME", DENY_NAME),
                               ("live", "C_GENERIC", DENY_GENERIC), ("free", "D_FREE", None)):
        body = build(sim, cut, mode)
        if extra:
            body += attempt + "\n[tool] " + extra
        c = collections.Counter()
        for i in range(n):
            try:
                r = chat(body, tools, 0.0 if i == 0 else 0.7, 300)
            except Exception as e:
                r = {"content": "ERR %s" % type(e).__name__}
            c[score(r, gold)] += 1
        print("  %-10s 문맥 %6d자 · HIT %d/%d · EMPTY %d   %s"
              % (label, len(body), c["HIT"], n, c["EMPTY"], c.most_common(4)))
    print("\n※ A 의 EMPTY 가 높아야 결손이 재현된 것이다(낮으면 표집이었고 이 프로브는 무효)."
          "\n  B≫C 면 처방은 **거부가 인자 이름을 대는 것** 하나다([[64]]).")


if __name__ == "__main__":
    main()
