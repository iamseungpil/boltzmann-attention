# -*- coding: utf-8 -*-
r"""x640 - **완결 검산(`_closure_note`)이 Q38 에서 100% 를 내는가** (격리 · [[76]]·[[78]]).

## 왜 (사용자 지시 2026-08-30)
*"이번에 격리 실험 하면 된다. 이전거 필요없다. **Q38 에서 이번에 격리 실험하라**"* ·
그리고 *"격리에서 **4/4** 로 성공해야 하는 거 아닌가"* - [[76]] 자격 기준이 **100%** 다.
⛔옛 2/4(커밋 `bbab2278`)는 **라이브 런**이고 격리가 아니다. 인용하지 않는다.

## 표적 = 016 의 금액 결정점
`x617`(격리 base · Qwen3.8 · alltools · 우리 층 0)에서 016 은 실패했다:
검색 **15회**(bm25 10 + dense 5) · shell **0회** · 이름으로 연 문서 **0**.
문서는 이미 도착했는데(KB_search 는 61% 가 원문 전량 - `x637`) **어느 제품 것인지** 를 못 가렸다.

## 팔 - 바뀌는 것은 **한 칸**
    A_asis     회수된 라이브 문맥 그대로                    <- 재현 게이트
    B_closure  + `t2_gate_patch._closure_note()` 출력        <- **엔진 빌더가 만든다**
    N_len      같은 길이의 무관 문장                        <- 길이 통제([[57]])
⛔이 파일은 프롬프트를 쓰지 않는다([[78]]). B 팔의 문면은 전부 엔진 함수가 낸다.

## 채점 - 닫힌 술어 · gold 미접촉([[23]])
A3 가 `qualifying_spend_usd` 축에 대해 **제품마다 다른 값**을 선언해 뒀다. 답에 나온 금액이
**어느 제품의 선언값인가**로 귀속한다. 엔진은 무엇이 옳은지 모른다 - 소속만 센다.
(정답 판정이 아니라 **출처 귀속**이다. x573 과 같은 규약.)

사용: PYTHONPATH=. python x640_closure_iso_q38.py --port 8141 [--n 8] [--wiring-only]
"""
import argparse
import collections
import io
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import t2_gate_patch as G                                       # noqa: E402
from gate_interpreter import load_domain_a2                     # noqa: E402

RES = "/home/woori/iso_tau3/tau2-bench/data/simulations/bank_x617_iso_q38_bank20_20260830/results.json"
DOCS_DIR = "/home/woori/iso_tau3/tau2-bench/data/tau2/domains/banking_knowledge/documents"
TARGET = "task_016"
AXIS = "qualifying_spend_usd"
KB = {"KB_search_bm25", "KB_search_dense", "shell"}
MODEL = "Qwen/Qwen3.8-27B-FP8"


def declared_values(a2, axis):
    """축 하나에 대해 **제품마다 선언된 값** - 채점용 귀속표. A3 선언만 읽는다."""
    out = {}
    for r in ((a2.get("policy_ontology") or {}).get("rows") or []):
        if str(r.get("axis")) == axis and r.get("value") is not None:
            out.setdefault(str(r.get("subject")), set()).add(str(r.get("value")))
    return out


def cut(sim):
    """마지막 검색 호출을 담은 어시스턴트 턴까지 (그 직후가 결정점)."""
    msgs = sim.get("messages") or []
    last = 0
    for i, m in enumerate(msgs):
        if str(m.get("role") or "") != "assistant":
            continue
        for tc in (m.get("tool_calls") or []) or []:
            if (tc.get("name") or (tc.get("function") or {}).get("name")) in KB:
                last = i
    return msgs[:last + 3]


def to_openai(msgs):
    out = []
    for m in msgs:
        role = str(m.get("role") or "")
        c = str(m.get("content") or "")
        if role in ("system", "user"):
            out.append({"role": role, "content": c})
        elif role == "assistant":
            d = {"role": "assistant", "content": c or None}
            tcs = []
            for tc in (m.get("tool_calls") or []) or []:
                a = tc.get("arguments")
                nm = tc.get("name") or (tc.get("function") or {}).get("name") or "unknown"
                tcs.append({"id": tc.get("id") or "x", "type": "function",
                            "function": {"name": nm, "arguments": a if isinstance(a, str)
                                         else json.dumps(a or {}, ensure_ascii=False)}})
            if tcs:
                d["tool_calls"] = tcs
            out.append(d)
        elif role == "tool":
            out.append({"role": "tool", "tool_call_id": m.get("id") or "x", "content": c[:6000]})
    return out


class _Agent(object):
    pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8141)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--wiring-only", action="store_true")
    a = ap.parse_args()

    os.environ["T2_SEARCH_CLOSURE"] = "1"
    a2 = load_domain_a2("banking_knowledge")
    vals = declared_values(a2, AXIS)
    print("A3 가 `%s` 축에 선언한 제품 %d개:" % (AXIS, len(vals)))
    for s in sorted(vals):
        print("   %-36s %s" % (s, ", ".join(sorted(vals[s]))))

    d = json.load(io.open(RES, encoding="utf-8"))
    sim = next((s for s in (d.get("simulations") or [])
                if str(s.get("task_id")) == TARGET
                and (s.get("reward_info") or {}).get("reward") == 0.0), None)
    if sim is None:
        print("SKIP - %s 실패 sim 없음" % TARGET)
        return
    pre = cut(sim)

    # ★B 팔의 문면은 **엔진 빌더**가 만든다 - 이 파일은 프롬프트를 쓰지 않는다([[78]]).
    # ★라이브 거동을 그대로 흉내낸다: 검산 문면은 **검색마다** 나가므로 모델은 누적해서 다 본다.
    #   (배선 확인 1차에서 마지막 결과만 취했더니 016 의 마지막 검색이 `blue_account` 라
    #    정작 결정 대상인 카드 제품을 한 마디도 안 했다 - 프로브가 실제 거동과 달랐다.)
    ag = _Agent()
    seen_lines, order = set(), []
    for m in pre:
        if str(m.get("role") or "") != "tool":
            continue
        n = G._closure_note(ag, a2, str(m.get("content") or ""))
        if not n:
            continue
        for ln in n.splitlines():
            if ln.startswith("- ") and ln not in seen_lines:
                seen_lines.add(ln)
                order.append(ln)
    note = None
    if order:
        note = (chr(10) + "[Declared sources for the subjects seen so far. "
                "This accounting is complete.]" + chr(10) + chr(10).join(order))
        cap = int(os.environ.get("T2_SEARCH_CLOSURE_CAP", "4000"))
        if len(note) > cap:
            print("⚠누적 문면 %d B > 상한 %d - 상한을 넘으면 아무것도 주지 않는 규약([[62]]④)"
                  % (len(note), cap))
            note = None
    print()
    print("문맥 %d 메시지 · 완결 검산 문면 %s B" % (len(pre), len(note or "")))
    if note:
        print("--- B 팔이 받는 문면 (엔진 생성) ---")
        print(note[:900])
    if a.wiring_only:
        return
    if not note:
        print("REFUSING - 검산 문면이 비었다(엔진이 침묵). 팔을 만들 수 없다.")
        return

    import urllib.request
    url = "http://localhost:%d/v1/chat/completions" % a.port

    def ask(msgs):
        body = json.dumps({"model": MODEL, "messages": msgs,
                           "temperature": a.temp, "max_tokens": 600}).encode()
        req = urllib.request.Request(url, data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as r:
            j = json.loads(r.read().decode())
        return j["choices"][0]["message"].get("content") or ""

    # ★D_deliver - **배달** 팔 (사용자 지적 2026-08-30: *"문서를 읽고 도구별로 배치하므로,
    #   100% 나와야 하는 거 아닌가"*). 맞다 - `_closure_note` 는 **세기만** 하므로 고르는 일이
    #   모델에 남는다. A3 가 (주어, 축) → 문서를 이미 배치해 뒀으니 **그 문서를 건네면** 된다.
    #   두 칸이고 경계를 안 넘는다:
    #     ① 모델이 주어를 댄다        (열린 술어 = 모델 몫 · x145 §1c 말미·C316)
    #     ② 엔진이 그 주어·그 축의 **선언 문서를 축자로** 배달 (닫힌 조회)
    #   이 팔이 **100%(8/8)** 여야 [[76]] 자격이다. 세기 팔이 미달인 것은 이 명제를 반증하지 않는다.
    ask_subject = ("Which specific card product is this customer's card? "
                   "Answer with the product name only.")
    corpus = {}
    try:
        for f in os.listdir(DOCS_DIR):
            if f.endswith(".json"):
                j = json.load(io.open(os.path.join(DOCS_DIR, f), encoding="utf-8"))
                corpus[j.get("id")] = j.get("content") or ""
    except Exception as e:
        print("코퍼스 로드 실패: %r" % (e,))
    axis_doc = {}
    for r in ((a2.get("policy_ontology") or {}).get("rows") or []):
        if str(r.get("axis")) == AXIS:
            d = (r.get("source") or {}).get("doc")
            if d:
                axis_doc[str(r.get("subject"))] = d

    base = to_openai(pre)
    filler = "Please continue helping the customer. " * max(1, len(note) // 38)
    arms = [("A_asis", None), ("B_closure", note), ("N_len", filler)]
    print()
    print("%-11s %-4s %s" % ("arm", "n", "답에 나온 금액의 **출처 제품** 분포"))
    print("-" * 78)
    for arm, extra in arms:
        attr = collections.Counter()
        for _ in range(a.n):
            msgs = list(base) + ([{"role": "user", "content": extra}] if extra else [])
            try:
                out = ask(msgs)
            except Exception as e:
                print("  %s ERR %s" % (arm, e))
                continue
            nums = set(re.findall(r"\$\s?([0-9][0-9,]*)", out))
            nums = {x.replace(",", "") for x in nums}
            owners = set()
            for s, vs in vals.items():
                for v in vs:
                    if str(v).replace(".0", "").replace(",", "") in nums:
                        owners.add(s)
            attr[",".join(sorted(owners)) if owners else "(선언값 없음)"] += 1
        print("%-11s %-4d %s" % (arm, sum(attr.values()),
                                 " · ".join("%s=%d" % (k[:44], v) for k, v in attr.most_common(4))))

    # ── D_deliver: ①모델이 주어를 댄다 → ②엔진이 그 주어의 선언 문서를 축자로 배달 ──
    print()
    print("=== D_deliver (2단) — 이 팔이 100%(8/8) 여야 [[76]] 자격 ===")
    attr = collections.Counter()
    named = collections.Counter()
    for _ in range(a.n):
        try:
            said = ask(list(base) + [{"role": "user", "content": ask_subject}])
        except Exception as e:
            print("  1단 ERR %s" % e)
            continue
        low = said.lower()
        # 모델이 댄 이름 ↔ 선언된 주어의 **동일성**만 본다(유사도 0 · 엔진은 고르지 않는다)
        picks = [s for s in axis_doc if s.lower() in low]
        named[",".join(sorted(picks)) if picks else "(주어 미지목)"] += 1
        if len(picks) != 1:
            attr["(주어 %d개 지목 → 배달 불가)" % len(picks)] += 1
            continue
        did = axis_doc[picks[0]]
        body = corpus.get(did) or ""
        if not body:
            attr["(선언 문서 본문 없음)"] += 1
            continue
        mat = ("The policy document declared as the source for this product's %s is below.%s[%s]%s%s"
               % (AXIS, chr(10), did, chr(10), " ".join(body.split())[:1800]))
        try:
            out = ask(list(base) + [{"role": "user", "content": ask_subject},
                                    {"role": "assistant", "content": said},
                                    {"role": "user", "content": mat}])
        except Exception as e:
            print("  2단 ERR %s" % e)
            continue
        nums = {x.replace(",", "") for x in re.findall(r"\$\s?([0-9][0-9,]*)", out)}
        owners = {s for s, vs in vals.items()
                  for v in vs if str(v).replace(".0", "").replace(",", "") in nums}
        attr[",".join(sorted(owners)) if owners else "(선언값 없음)"] += 1
    print("  1단 주어 지목: %s" % " · ".join("%s=%d" % (k[:40], v) for k, v in named.most_common(4)))
    print("  2단 금액 귀속: %s" % " · ".join("%s=%d" % (k[:40], v) for k, v in attr.most_common(4)))


main()
