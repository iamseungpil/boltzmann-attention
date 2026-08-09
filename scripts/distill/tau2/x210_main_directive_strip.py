# -*- coding: utf-8 -*-
r"""x210 — 메인 채널에서 **지시를 빼면** 어떻게 되나 (격리 A/B · 유료 0 · 엔진 0).

## 규칙 (사용자 지시 2026-08-10)

> *"꼭 지시가 필요하면 격리 서브 에이전트 환경에서 꼭 필요한 지시만 최소로 하도록 하라."*

⇒ **메인 채널은 값만.** 지금 메인에 나가는 문장 다섯이 지시를 섞고 있고, 그중 하나는
**C388 에서 내가 넣은 것**이다(`window_history_text` 의 *"retrieve it and say which applies"*).
규칙에 맞는 형태는 **산수만 주고 꼬리말을 아예 달지 않는 것**인데, x200 에는 그 팔이 없었다 —
`OLD`(부정 꼬리말)와 `CUR`(지시 꼬리말)만 비교했다. 그래서 **꼬리말이 애초에 필요한지**를
모른 채 하나를 다른 하나로 바꾼 셈이다.

## 팔 (task_010 · 손님의 실제 되묻는 발화)

  OLD    부정 꼬리말   *"…it does not say why any record carries the status it carries"*
  CUR    지시 꼬리말   *"…retrieve it and say which applies"*   ← 지금 라이브(규칙 위반)
  BARE   **꼬리말 없음 — 산수만**                                ← 규칙에 맞는 형태
  BAREDOC / CURDOC / OLDDOC  각각 + 상태 정의 문서 축자
  D_null 문장 없이 원장만 (부정 통제)

## 읽는 법

  · BARE 가 CUR 이상이면 → 꼬리말은 **불필요**하고 규칙대로 빼면 된다(가장 싼 결론).
  · BARE 가 OLD 수준이면 → 꼬리말이 짐을 지고 있다. 그때는 **서브로 옮길 수 있는지**를 본다.
  · *DOC 팔이 전부 8/8 이면 → 진짜 레버는 문서 도달이고 꼬리말 논쟁은 부차적이다.

실행: python x210_main_directive_strip.py [N]
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

import t2_ledger as LG                                          # noqa: E402
from gate_interpreter import load_domain_a2                      # noqa: E402
from x200_disclaimer_ab import ROWS, LEDGER, ASK, DOC, CAUSE, ESCAPE   # noqa: E402

URL = os.environ.get("T2_PROBE_URL", "http://localhost:8141/v1/chat/completions")
MODEL = os.environ.get("T2_PROBE_MODEL", "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8")

OLD_TAIL = (" This says how many records already fell inside the window when each of these was "
            "made - it does not say why any record carries the status it carries.")


def ask(prompt, temp):
    body = {"model": MODEL, "temperature": temp, "max_tokens": 220,
            "messages": [{"role": "user", "content": prompt}]}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return " ".join((json.load(r)["choices"][0]["message"]["content"] or "").split())


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 8
    show = "--show" in sys.argv
    a2 = load_domain_a2("banking_knowledge")
    sp = a2["ledger_metrics"][0]
    st = LG.status_breakdown(ROWS, sp)
    wh = LG.window_history(ROWS, sp)                 # 현행(지시 꼬리말)
    head = wh.split(" This is arithmetic on the dates")[0]        # 산수까지만
    variants = {"OLD": st + "\n" + head + OLD_TAIL, "CUR": (st + wh), "BARE": st + "\n" + head}
    print("[BARE] (규칙에 맞는 형태)\n%s\n" % variants["BARE"].strip())
    out = {}
    for arm in ("OLD", "CUR", "BARE", "OLDDOC", "CURDOC", "BAREDOC", "D_null"):
        base = "" if arm == "D_null" else variants[arm.replace("DOC", "")]
        with_doc = arm.endswith("DOC")
        c, texts = collections.Counter(), []
        for i in range(n):
            p = LEDGER + (("\n\n" + base.strip()) if base else "")
            if with_doc:
                p += "\n\n" + DOC
            p += ("\n\nThe customer asks:\n%s\n\nAnswer the customer in two or three sentences."
                  % ASK)
            try:
                t = ask(p, 0.0 if i == 0 else 0.7)
            except Exception as e:
                t = "ERR %s" % type(e).__name__
            texts.append(t)
            lo = t.lower()
            c["이유O" if any(k in lo for k in CAUSE) else "이유X"] += 1
            c["이관O" if any(k in lo for k in ESCAPE) else "이관X"] += 1
        out[arm] = [c["이유O"], n]
        print("  %-8s 이유 %d/%d · 이관 %d/%d" % (arm, c["이유O"], n, c["이관O"], n))
        if show:
            print("      | " + texts[0][:260])
    json.dump(out, open(os.environ.get("T2_X210_OUT", "x210_out.json"), "w"), indent=1)
    print("\n※ BARE ≥ CUR 이면 꼬리말은 불필요하다 — 규칙대로 뺀다."
          "\n  BARE ≈ OLD 면 꼬리말이 짐을 지고 있으니 **서브로 옮길 수 있는지**를 다음에 잰다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
