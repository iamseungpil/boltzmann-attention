# -*- coding: utf-8 -*-
r"""x181 — **099 와 100 을 가르는 것은 무엇인가**: 자기-정박(직전 약속) 대 표 순서 (유료 0).

## 왜 (C359 가 바꿔 놓은 질문)

C355~C358 은 task_099 에서 *"행 순서 방향이 답을 100%↔0% 로 가른다"* 를 세웠고, C359 는
task_100 에서 그것이 **전혀 재현되지 않는다**(10정렬×2모델 전부 gold 0/8, 답은 언제나
`Cobalt Blue`)를 세웠다. 그래서 물어야 할 것은 *"왜 오름차순이 나쁜가"* 가 아니라
**"두 태스크를 가르는 변수가 무엇인가"** 다.

## [[55]] 배관 먼저 — 무료 포렌식이 먼저 찾아낸 것

이 프로브들이 표 앞에 붙이는 **접두 대화의 마지막 메시지**가 두 태스크 모두
*에이전트 자신이 이미 답을 말해 버린 턴* 이다(구조 동일·`msgs_of` 의 cut 이
`submit_referral` 직전이라 그렇다):

  · 099 끝 : "...proceed with ... the **Hunter Green** Account, which will maximize
             your referral bonus"                      — 산문 · 끝에서 428자
  · 100 끝 : "...we recommend referring ... a **Cobalt Blue** business checking
             account, as it offers a higher referral bonus of $150" +
             ```json {"account_type": "Cobalt Blue"} ``` — **JSON 인자 템플릿** · 끝에서 189자

그리고 **관측된 오답이 정확히 그 이름들이다**: 099 는 14B 오답이 `Hunter Green`,
중립 라벨판(x180)에선 두 모델 다 `O15`(= `Hunter Green`) · 100 은 전 배열 `Cobalt Blue`.
⇒ 후보 공통 기전 = **자기-정박**(직전 턴의 자기 약속을 되풀이). 순서는 그 정박을
**깨느냐 못 깨느냐**를 좌우하는 조절 변수일 뿐일 수 있다.

## 축 (구조로 자른다 — 도메인 텍스트를 뜯지 않는다·[[59]])

  ctx  full        접두 대화 그대로 (x175 재현 기준선)
       -commit     **꼬리의 assistant 메시지 제거** = 자기 약속만 뺀다 (보유계좌 read 는 남는다)
       -accounts   `call_discoverable_agent_tool` 의 tool 응답 제거 = 보유계좌 노출만 뺀다
       none        대화 없음 (표 + 사실 + 질문)
  sort name_asc / name_desc   (C357 의 방향 축)

## 판정 규칙 (가설이 아니라 읽는 법)

  · `-commit` 에서 **두 태스크 다** gold 로 붙으면      → 공통 기전 = **자기-정박**. 순서는 조절 변수.
  · `-commit` 에서 100 이 **순서 민감**해지면          → 099/100 차이 = **정박 세기**(같은 기전).
  · `none` 에서 099 `name_asc` 가 여전히 틀리면        → 순서는 정박과 **독립**인 별개 원인.
  · `-accounts` 만 고치면                              → 기전은 약속이 아니라 **보유 계좌 포획**.

부수 계측: 매 셀마다 답이 **정박 이름**(후보 중 접두 대화에 *가장 늦게* 등장한 것)과
같은 비율을 함께 센다 — 정박 이름은 gold 를 보지 않고 문맥에서만 정한다.

⚠ 접두 대화가 없는 `none` 은 두 태스크가 비대칭이다 — `X.FACTS["task_099"]` 는 보유 계좌를
   문장으로 적고 있고 100 은 안 적는다(궤적 유래 고정 문구). 출력에 그대로 표시한다.

실행: python x181_anchor_vs_order.py [N] [MODEL_URL]
"""
import collections
import json
import os
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import x149_choice_isolation as X                              # noqa: E402
import x150_choice_ablation as Y                               # noqa: E402
import t2_factdag as FD                                        # noqa: E402
import t2_ledger as LG                                         # noqa: E402
from gate_interpreter import load_domain_a2                     # noqa: E402

URL = os.environ.get("T2_PROBE_URL", "http://localhost:8140/v1/chat/completions")
MODEL = os.environ.get("T2_PROBE_MODEL", "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8")
TAG = os.environ.get("T2_PROBE_TAG", "bank_elig_20260809i")
# 손님 값은 궤적 유래(x155 CASE 실측) — gold 에서 온 것이 아니다.
CASE = {"task_099": {"days": 730, "deposit": 30000},
        "task_100": {"days": 65, "deposit": 31000}}
ACCT_TOOL = "call_discoverable_agent_tool"


def guided_full(prompt, choices, temp):
    body = json.dumps({"model": MODEL, "temperature": temp, "max_tokens": 12,
                       "guided_choice": list(choices),
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return " ".join((json.load(r)["choices"][0]["message"]["content"] or "").split())


def drop_tail_assistant(ms):
    """꼬리의 assistant 턴(=자기 약속)만 뗀다. 역할로만 판정하고 내용은 안 본다."""
    out = list(ms)
    while out and out[-1].get("role") == "assistant":
        out.pop()
    return out


def drop_accounts_read(ms):
    """`call_discoverable_agent_tool` 의 **응답 메시지**만 뗀다 (호출한 assistant 턴은 남긴다)."""
    out, skip_next_tool = [], False
    for m in ms:
        if skip_next_tool and m.get("role") == "tool":
            skip_next_tool = False
            continue
        out.append(m)
        skip_next_tool = any((tc or {}).get("name") == ACCT_TOOL
                             for tc in (m.get("tool_calls") or []))
    return out


def anchor_of(text, choices):
    """문맥에서만 정하는 정박 이름 = 후보 중 접두 대화에 **가장 늦게** 등장한 것 (gold 안 봄).

    ⚠계기 결함 수리(2026-08-09·첫 판): 후보에 `Green`·`Blue` 처럼 **다른 후보의 부분
      문자열**인 이름이 있어서 `rfind` 가 `Hunter Green` 안의 `Green` 을 잡았고, 정박 열이
      전부 0/8 로 무의미해졌다. 같은 자리에서는 **가장 긴 후보**가 이기게 한다.
    """
    best, pos, ln = None, -1, -1
    for c in choices:
        p = text.rfind(c)
        if p < 0:
            continue
        end = p + len(c)
        if end > pos or (end == pos and len(c) > ln):
            best, pos, ln = c, end, len(c)
    return best


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    a2 = load_domain_a2("banking_knowledge")
    spec = next(s for s in a2["ledger_metrics"] if s.get("eligible_text"))
    rows = a2["policy_ontology"]["rows"]
    axes = spec["eligible"]["show_axes"]
    maps = {ax: FD._a3_map(rows, {"axis": ax}) for ax in axes}
    name = lambda l: l.strip().split(":")[0].strip()          # noqa: E731

    print("model=%s · tag=%s · n=%d" % (MODEL, TAG, n))
    grand = []
    for task, case in CASE.items():
        gold = X.GOLD[task]
        lines = LG.eligible_text(case["days"], {}, maps, spec,
                                 {"qualifying_deposit_usd": case["deposit"]}).strip().splitlines()
        head = [l for l in lines if not (l.startswith("  ") and ":" in l)]
        body = [l for l in lines if l.startswith("  ") and ":" in l]
        FIXED = [name(l) for l in body]                        # 후보 목록 = 알파벳 고정

        ms = Y.msgs_of(TAG, task)
        ctxs = [("full     ", Y.render(ms)),
                ("-commit  ", Y.render(drop_tail_assistant(ms))),
                ("-accounts", Y.render(drop_accounts_read(ms))),
                ("none     ", "")]
        sorts = [("name_asc ", sorted(body, key=name)),
                 ("name_desc", sorted(body, key=name, reverse=True))]

        print("\n" + "=" * 96)
        print("%s  gold=%r  통과 %d행  손님=(%d일·$%d)"
              % (task, gold, len(body), case["days"], case["deposit"]))
        full_txt = Y.render(ms)
        anc = anchor_of(full_txt, FIXED)
        print("  정박(문맥 최후 등장 후보) = %r   · FACTS 에 보유계좌 문장 = %s"
              % (anc, "있음" if "already owns" in X.FACTS[task] else "없음"))
        print("=" * 96)
        print("  %-10s %-10s %-6s %-7s %s" % ("ctx", "sort", "gold", "정박", "분포"))
        for clabel, ctext in ctxs:
            for slabel, order in sorts:
                tbl = "\n".join(head[:1] + order + head[1:]).strip() if head else "\n".join(order)
                base = tbl + "\n\n" + X.FACTS[task] + "\n\n" + X.QUESTION
                pre = ("Here is a customer-service conversation so far.\n\n" + ctext + "\n\n") \
                    if ctext else ""
                a = anchor_of(ctext, FIXED) if ctext else None
                c = collections.Counter()
                for i in range(n):
                    try:
                        c[guided_full(pre + base, FIXED, 0.0 if i == 0 else 0.7)] += 1
                    except Exception as e:
                        c["ERR %s" % type(e).__name__] += 1
                g = c.get(gold, 0)
                ah = c.get(a, 0) if a else 0
                print("  %-10s %-10s %d/%-4d %-7s %s"
                      % (clabel, slabel, g, n, ("%d/%d" % (ah, n)) if a else "-",
                         c.most_common(3)))
                grand.append((task, clabel.strip(), slabel.strip(), g, n, a, ah))

    print("\n" + "-" * 96)
    print("판정: `-commit` 이 두 태스크를 다 고치면 공통 기전 = 자기-정박(순서는 조절 변수).")
    print("      `none` 에서 099 name_asc 가 여전히 틀리면 순서는 독립 원인.")
    print("      `-accounts` 만 고치면 기전은 약속이 아니라 보유계좌 포획.")
    json.dump([{"task": t, "ctx": c, "sort": s, "gold_hit": g, "n": nn,
                "anchor": a, "anchor_hit": ah} for t, c, s, g, nn, a, ah in grand],
              open(os.environ.get("T2_X181_OUT", "x181_out.json"), "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
