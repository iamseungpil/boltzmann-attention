# -*- coding: utf-8 -*-
r"""x259 — 도구 결과를 **어떤 모양으로** 뷰에 두어야 하나 (격리 · 유료 0 · 8140 · 새 엔진 0).

## 왜 (사용자 설계 → 오늘 측정이 남긴 유일한 미결)

사용자 구상 = *"메인 뷰에는 호출과 결론만, 상세는 서브 안에서"*. 배관은 이미 있다 —
`_compact_view` 가 **생성 뷰에서만** 큰 도구 출력을 다이제스트로 갈아 끼우고 커밋 히스토리는
그대로 둔다(replay·게이트 불변). 없는 것은 **다이제스트의 내용**이다: 지금은
`head 300 + tail 150` **기계 절단**이고(주석 축자: *"엔진의 내용 추출/합성 0"*), 사용자 제안은
그 자리에 **서브 LLM 의 결론**을 넣자는 것이다.

**길이는 이유가 아니다** — x258 에서 누적 19.8K자가 24/24 로 가장 높았다. 정당화는 잡음이어야
하고, 잡음의 실측 증거는 070_4 다: gold `Sky Blue` 인데 `Light Blue Account` 로 오정박(0/8).

## ⛔0 — 이 자리는 결손이 확실히 재현된다

x256 에서 같은 cut·같은 궤적으로 `A_LIVE` **0/8**(8/8 전부 `Light Blue Account`)·`E_FREE` **0/8**.
x258 처럼 *부정 통제 만점이라 무효* 가 되는 자리가 아니다(handoff §9).

## 팔 (n · 계기 = 다음 도구 호출 하나 · 지시문 없음)

  A_FULL    도구 출력 전문 (지금 라이브)                        ← 재현 팔(0/8 이어야 한다)
  B_TRUNC   **엔진의 기계 절단**을 그대로 적용(`_compact_view`)  ← 현행 압축
  C_SUB     **서브 LLM 요약**으로 대체(도메인 일반 프롬프트)     ← 사용자 제안
  D_VALUE   결정값 한 줄(x256 `B_VALUE` 재현)                    ← 상한 6/8

⚠**C_SUB 의 프롬프트는 답의 모양을 담지 않는다** — *"이번 요청에 관련된 것만 축자로 남겨라.
  순위·추천·요약 판단 금지"*. 담는 순간 떠먹이기이고 `account_class` 는 **채점 칸 자체**다([[62]]).
⚠B 는 **엔진 함수를 그대로 호출**한다(문구·상수 재작성 0·[[03b]]).

실행: T2_PROBE_URL=http://localhost:8140/v1/chat/completions python x259_digest_shape_probe.py [N]
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
import x256_dispatcher_write_probe as X                           # noqa: E402
import t2_gate_patch as G                                         # noqa: E402

RES = X.RES
TASK = X.TASK
TOOL = X.TOOL

# 도메인 일반 · 답의 모양 0 · 판단 금지
SUB_PROMPT = ("The customer's current request is below, followed by one tool output.\n"
              "Return only the lines of that output which bear on the request, copied "
              "verbatim. Do not rank, recommend, choose, summarise in your own words, or "
              "add anything. If nothing bears on it, return the single word NONE.\n\n"
              "REQUEST:\n%s\n\nTOOL OUTPUT:\n%s")


class _M(object):
    """`_compact_view` 가 기대하는 최소 인터페이스 (`copy.copy` + `.content` 대입)."""
    def __init__(s, role, content, mid):
        s.role, s.content, s.id = role, content, mid


def last_user(sim, cut):
    for m in reversed(sim["messages"][:cut]):
        if m.get("role") == "user" and (m.get("content") or "").strip():
            return " ".join(str(m["content"]).split())[:900]
    return ""


def digests(sim, cut, mode, req):
    """cut 이전 도구 출력 → {index: 대체 텍스트}. mode = trunc | sub."""
    idx = [i for i, m in enumerate(sim["messages"][:cut])
           if m.get("role") == "tool" and len(str(m.get("content") or "")) > 800]
    out = {}
    if mode == "trunc":
        msgs = [_M(m.get("role"), str(m.get("content") or ""), i)
                for i, m in enumerate(sim["messages"][:cut])]
        view, dg = G._compact_view(msgs, keep_recent=0, min_len=800,
                                   min_total=0, msg_cap=0)
        for i, v in enumerate(view):
            if getattr(v, "id", None) in dg:
                out[i] = v.content
    else:
        for i in idx:
            c = str(sim["messages"][i].get("content") or "")
            try:
                r = chat(SUB_PROMPT % (req, c[:6000]), None, 0.0, 700)
                t = (r.get("content") or "").strip()
                if t and t.upper() != "NONE":
                    out[i] = t
            except Exception as e:
                print("   (서브 요약 실패 idx=%d: %r)" % (i, type(e).__name__))
    return out


def build(sim, cut, repl, extra=None):
    out = []
    for i, m in enumerate(sim["messages"][:cut]):
        r = m.get("role")
        c = repl.get(i, " ".join(str(m.get("content") or "").split()))
        tcs = [tc.get("name") for tc in (m.get("tool_calls") or [])]
        if any(tcs):
            out.append("[%s calls] %s" % (r, ", ".join(x for x in tcs if x)))
        if str(c).strip():
            out.append("[%s] %s" % (r, str(c)[:2400]))
    if extra:
        out.append("[system] " + extra)
    return "\n".join(out)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 8
    d = json.load(open(RES, encoding="utf-8"))
    sim = [s for s in d["simulations"] if s["task_id"] == TASK][0]
    gold = X.gold_args(sim)
    cut = None
    for i, m in enumerate(sim["messages"]):
        for tc in (m.get("tool_calls") or []):
            if (tc.get("name") or "") == X.DISPATCH and TOOL in json.dumps(
                    tc.get("arguments"), ensure_ascii=False) and cut is None:
                cut = i
    tools = U.tools_of(sim)
    req = last_user(sim, cut)
    print("요구 턴 %d · gold %s · n=%d\n손님 요청 축자: %s\n"
          % (cut, json.dumps(gold, ensure_ascii=False), n, req[:200]))

    tr = digests(sim, cut, "trunc", req)
    sb = digests(sim, cut, "sub", req)
    print("다이제스트 대상 %d개 · 기계절단 %d · 서브요약 %d\n" % (len(tr), len(tr), len(sb)))
    val = "[DECIDED] The account class the retrieved documents support for this request is: %s" \
          % gold.get("account_class")

    arms = [("A_FULL", {}, None), ("B_TRUNC", tr, None), ("C_SUB", sb, None),
            ("D_VALUE", {}, val)]
    for label, repl, extra in arms:
        body = build(sim, cut, repl, extra)
        c = collections.Counter()
        for i in range(n):
            try:
                r = chat(body, tools, 0.0 if i == 0 else 0.7, 300)
            except Exception as e:
                r = {"content": "ERR %s" % type(e).__name__}
            c[X.score(r, gold)] += 1
        print("  %-9s 문맥 %6d자 · HIT %d/%d   %s"
              % (label, len(body), c["HIT"], n, c.most_common(3)))
    print("\n※ n=8 의 표집 변동은 ±2 다(x258 에서 같은 문맥이 5/8↔3/8). 두 칸 차이로 결론 금지.")
    print("  C_SUB 가 A_FULL 을 넘으면 다이제스트를 서브로 바꾸는 것이 값을 산다.")
    print("  B_TRUNC 가 A_FULL 보다 낮으면 **현행 압축이 해롭다**(지금 라이브에서 켜져 있다).")


if __name__ == "__main__":
    main()
