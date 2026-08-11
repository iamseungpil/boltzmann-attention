# -*- coding: utf-8 -*-
r"""x256 — **결정값은 맞는데 쓰기 인자에 안 실린다** (격리 · 유료 0 · 8140 · 새 엔진 0).

## 왜 (070_4 · 사용자 제안 "호출 타입별 액션 서브")

`bank_all6b_20260811` 의 070 은 **한 칸**에서 죽었다 — `070_4`.
  gold : `open_bank_account_4821(user_id=…, account_type="business_checking", account_class="Sky Blue")`
  실제 : 같은 도구·같은 타입인데 **`account_class="Light Blue Account"`**
**부르긴 불렀고 성공까지 했다.** 미호출이 아니라 **인자 오선택**이다.

그런데 그 값은 **우리가 이미 맞힌다**: C417/x248 의 격리 검색 에이전트가 checking 축 gold
`Sky Blue` 를 **8/8**, savings 축 `Gold Saver Account` 를 **8/8** 로 낸다. 즉 **결정은 서 있고
전달이 죽는다** — x228·CALL_FORM·ARG_EMPTY 에 이은 **네 번째 같은 자리**다.

## ⛔0 (레버 전에 결손을 격리로 잰다)

이 프로브가 그 측정이다. 묻는 것 둘:
  ⑴ 결손이 재현되는가 — `A_LIVE` 가 다시 틀린 class 를 쓰는가(아니면 표집이었나).
  ⑵ **결정값만 전달하면** 모델이 그 값을 **중첩 JSON 인자**에 정확히 실어 부르는가(`B_VALUE`).
     실으면 레버는 **전달뿐**이고, 타입별 액션 서브는 *표기*만 하면 된다([[62]] ②).
     못 실으면 결손은 전달이 아니라 **호출 형식 구성**이고 처방이 달라진다(`C_FORM`이 가른다).

⚠**값을 주는 것이 이 프로브의 목적이다** — 값을 고르는 능력을 재는 것이 아니다. 그 능력은
  C417 이 격리에서 이미 8/8 로 쟀다. 여기서 재는 것은 *그 값이 호출까지 살아남는가* 하나다.
  라이브에서 값의 출처는 **격리 결정 서브(LLM)** 이지 엔진 표 조회가 아니다([[62]] ④).

## 팔 (n · 계기 = 다음 도구 호출 하나 · 지시문 없음)

  A_LIVE    라이브 재현(궤적 + 사이드카 축자)          ← 재현 팔(틀려야 한다)
  B_VALUE   + 결정값만 축자로 전달(호출 형식 언급 0)    ← 전달 팔
  C_FORM    + 호출 형식만(값 언급 0)                   ← 형식 팔
  D_BOTH    값 + 형식
  E_FREE    궤적만(우리 문장 0)                        ← 상한/부정 통제

통과 = `call_discoverable_agent_tool(open_bank_account_4821)` 이고 중첩 인자의
`account_class` 가 **gold 축자와 정확히 같은** 것. 부분 점수로 `CLASS_WRONG`·`NOCALL` 을 센다.

실행(리모트): python x256_dispatcher_write_probe.py [N]
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
import x241_uncalled_unlock_probe as U                            # noqa: E402

TAG = "bank_all6b_20260811"
TASK = "task_070"
TOOL = "open_bank_account_4821"
DISPATCH = "call_discoverable_agent_tool"
RES = "/home/woori/scratch/tau2-bench/data/simulations/%s/results.json" % TAG
FB = "/home/woori/scratch/logs/fb_%s.jsonl" % TAG


def gold_args(sim):
    for a in (sim["reward_info"].get("action_checks") or []):
        ac = a["action"]
        if ac["name"] == DISPATCH and TOOL in json.dumps(ac["arguments"]):
            inner = ac["arguments"].get("arguments")
            return json.loads(inner) if isinstance(inner, str) else (inner or {})
    return {}


def build(sim, cut, with_ours=True):
    import t2_fbsidecar as S

    class _M(object):
        def __init__(s, r, c):
            s.role, s.content = r, c

    ours = collections.defaultdict(list)
    if with_ours and os.path.exists(FB):
        key = S._sim_key([_M(m.get("role"), m.get("content")) for m in sim["messages"]])
        for ln in open(FB, encoding="utf-8", errors="replace"):
            o = json.loads(ln)
            if o.get("sim") == key and (o.get("text") or "").strip():
                ours[o.get("turn")].append(" ".join(o["text"].split()))
    out = []
    for i, m in enumerate(sim["messages"][:cut]):
        r, c = m.get("role"), " ".join(str(m.get("content") or "").split())
        tcs = [tc.get("name") for tc in (m.get("tool_calls") or [])]
        if tcs:
            out.append("[%s calls] %s" % (r, ", ".join(x for x in tcs if x)))
        if c:
            out.append("[%s] %s" % (r, c[:700]))
        for t in ours.get(i, ()):
            out.append("[system] %s" % t[:900])
    return "\n".join(out)


def score(msg, gold):
    for tc in (msg.get("tool_calls") or []):
        f = tc.get("function") or {}
        if (f.get("name") or "") != DISPATCH:
            return f.get("name") or "?"
        try:
            a = json.loads(f.get("arguments") or "{}")
        except Exception:
            return "BADJSON"
        if TOOL not in str(a.get("agent_tool_name") or ""):
            return "OTHER_TOOL"
        inner = a.get("arguments")
        try:
            inner = json.loads(inner) if isinstance(inner, str) else (inner or {})
        except Exception:
            return "BADNESTED"
        cls = str(inner.get("account_class") or "").strip()
        typ = str(inner.get("account_type") or "").strip()
        if cls == str(gold.get("account_class")) and typ == str(gold.get("account_type")):
            return "HIT"
        if not cls:
            return "NOCLASS"
        return "CLASS_WRONG(%s)" % cls[:16]
    return "(발화만)"


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 8
    d = json.load(open(RES, encoding="utf-8"))
    sim = [s for s in d["simulations"] if s["task_id"] == TASK][0]
    gold = gold_args(sim)
    # 첫 `open_bank_account` 디스패처 호출 직전까지 = 모델이 그 호출을 지어야 하는 자리
    cut = None
    for i, m in enumerate(sim["messages"]):
        for tc in (m.get("tool_calls") or []):
            if (tc.get("name") or "") == DISPATCH and TOOL in json.dumps(
                    tc.get("arguments"), ensure_ascii=False) and cut is None:
                cut = i
    tools = U.tools_of(sim)
    print("요구 턴 %d · 도구 %d개 · gold %s · n=%d\n"
          % (cut, len(tools), json.dumps(gold, ensure_ascii=False), n))

    # 문구는 **값만** / **형식만** 으로 갈라 둔다 — 섞으면 무엇이 샀는지 못 가른다.
    value_txt = ("[DECIDED] The account class the retrieved documents support for this request "
                 "is: %s" % gold.get("account_class"))
    form_txt = ("[CALL-FORM] %s is a discoverable tool - it is not in your tool list. The way to "
                "run it is %s(agent_tool_name=\"%s\", arguments=\"{…}\") where the inner string is "
                "JSON holding that tool's own parameters." % (TOOL, DISPATCH, TOOL))

    live = build(sim, cut, True)
    free = build(sim, cut, False)
    arms = [("A_LIVE", live), ("B_VALUE", live + "\n[system] " + value_txt),
            ("C_FORM", live + "\n[system] " + form_txt),
            ("D_BOTH", live + "\n[system] " + value_txt + "\n[system] " + form_txt),
            ("E_FREE", free)]
    for label, body in arms:
        c = collections.Counter()
        for i in range(n):
            try:
                r = chat(body, tools, 0.0 if i == 0 else 0.7, 300)
            except Exception as e:
                r = {"content": "ERR %s" % type(e).__name__}
            c[score(r, gold)] += 1
        print("  %-9s 문맥 %6d자 · HIT %d/%d   %s"
              % (label, len(body), c["HIT"], n, c.most_common(4)))
    print("\n※ A 가 낮아야 결손이 재현된 것이다. B 가 높으면 레버는 **전달뿐**이고 타입별 액션 "
          "서브는 표기만 하면 된다.\n  C 만 높으면 결손은 값이 아니라 **호출 형식 구성**이다.")


if __name__ == "__main__":
    main()
