# -*- coding: utf-8 -*-
r"""x274 — 절차에 **근거를 요구하면 날조가 사라지는가** (유료 0 · 엔진 0 · 사용자 설계 2026-08-12).

## 왜 (라이브 실측 · C439⒠③ · x271 턴별 정독)

`bank_lever_070_20260812c` trial1: 모델이 **정책에 없는 절차를 발명했다** — *"Secure Form Link"*
로 EIN/TIN 을 받겠다며 16턴을 태우고(*"I apologize for the oversight"* 8회) 계좌 개설에 도달하지
못했다. 계열 ③(write 미시도)의 절반이 이 모양이다.

## 왜 이 형태인가 (사용자: *"절차도 근거 확인하면 되지 않나"*)

이미 우리가 쓰는 계약이다 — [[22]] 따름정리(근거-우선 formalize) · C417 규약(*LLM 형식화 ·
엔진은 **인용 실재**만 검산*) · A2 `procedures[*]._source`/`nodes[*]._quote` 가 그 자리다.
엔진은 *"이 절차가 옳은가"* 를 판정하지 않는다 — **회수된 텍스트에 그 인용이 실재하는가**만 본다
(닫힌 술어·[[22]]). 어떤 절차를 밟을지는 여전히 모델 몫이다.

⚠**DAG 와 상보적**: 요건 그래프는 *있어야 할 것을 안 했다*(누락)를 보고, 근거 확인은
*없는 것을 했다*(**날조**)를 본다. `Secure Form Link` 는 그래프에 **없는 노드**라 DAG 는 조용하다.

## 팔 (n=8 · 계기는 **날조 절차가 답에 있는가**)

  A_LIVE        라이브 문맥 그대로 (발명 직전 턴까지)        ← 결손 재현 기대
  B_CITE        같은 문맥 + **근거 요구 한 줄**
  C_CITE_ISO    손님 요청만(격리) + 근거 요구
  D_ISO_NOCITE  격리 · 요구 없음 (부정 통제·[[57]])

계기: `INVENT`(secure form/link/portal/upload/EIN 제출 요구) · `CITED`(인용을 실제로 달았고 그
인용이 **회수 텍스트에 실재**) · `PLAIN`(둘 다 아님).
⚠인용 실재는 **엔진이 할 검산을 프로브가 그대로 흉내낸다** — substring 판정뿐이다.

실행: T2_PROBE_URL=http://localhost:8141/v1/chat/completions python3 x274_procedure_provenance.py [N]
"""
import collections
import io
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from x216_read_and_offset import chat                               # noqa: E402
import x256_dispatcher_write_probe as X256                          # noqa: E402

SIMS = os.environ.get("X274_SIMDIR", "/home/woori/scratch/tau2-bench/data/simulations")
TAG = os.environ.get("X274_TAG", "bank_lever_070_20260812c")
TASK = "task_070"
TRIAL = int(os.environ.get("X274_TRIAL", "1"))

# 근거 요구 — 도메인 낱말 0. 이기면 그대로 A2 슬롯이 된다([[03b]]).
CITE_FB = (
    "[PROCEDURE-SOURCE] before you describe any procedure or ask the customer for anything, "
    "quote the sentence from the retrieved documents that requires it, verbatim, in "
    "quotation marks. If no retrieved document states it, that step does not exist here - "
    "do not describe it and do not ask for what it would need."
)

INVENT = re.compile(r"secure form|form link|\bportal\b|upload (?:your |the )?(?:document|ein|tin)"
                    r"|\bEIN\b|\bTIN\b|https?://|\[.*link.*\]", re.I)
QUOTED = re.compile(r"[\"“]([^\"”]{25,})[\"”]")


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 8
    d = json.load(io.open(os.path.join(SIMS, TAG, "results.json"), encoding="utf-8"))
    sims = [s for s in d["simulations"] if s["task_id"] == TASK]
    sim = sims[TRIAL]
    msgs = sim["messages"]
    # 발명 직전 = 손님이 "무엇이 필요하냐"고 물은 턴. 그 자리까지를 문맥으로 쓴다.
    cut = None
    for i, m in enumerate(msgs):
        if m.get("role") == "user" and re.search(r"what info do you need|before we start",
                                                 str(m.get("content") or ""), re.I):
            cut = i
            break
    if cut is None:
        cut = max(i for i, m in enumerate(msgs) if m.get("role") == "user") // 2
    ask = str(msgs[cut].get("content") or "")
    tools = X256.U.tools_of(sim)
    live = X256.build(sim, cut, True)
    # 회수 텍스트 = 그 시점까지의 tool 출력 (인용 실재 검산용 — 엔진이 할 일과 같은 것)
    hay = "\n".join(str(m.get("content") or "") for m in msgs[:cut] if m.get("role") == "tool")
    print("문맥 %d자 · 회수 텍스트 %d자 · cut=%d\n요청: %s\n"
          % (len(live), len(hay), cut, " ".join(ask.split())[:130]))

    def score(r):
        t = str((r or {}).get("content") or "")
        inv = bool(INVENT.search(t))
        qs = QUOTED.findall(t)
        real = [q for q in qs if q[:40] in hay]
        if inv:
            return "INVENT"
        if real:
            return "CITED(%d)" % len(real)
        if qs:
            return "QUOTE_UNREAL(%d)" % len(qs)
        return "PLAIN"

    arms = (("A_LIVE", live),
            ("B_CITE", live + "\n[system] " + CITE_FB),
            ("C_CITE_ISO", ask + "\n\n[system] " + CITE_FB),
            ("D_ISO_NOCITE", ask))
    for label, body in arms:
        c = collections.Counter()
        for i in range(n):
            try:
                r = chat(body, tools, 0.0 if i == 0 else 0.7, 400)
            except Exception as e:
                r = {"content": "ERR %s" % type(e).__name__}
            c[score(r)] += 1
        print("  %-13s INVENT %d/%d   %s" % (label, c["INVENT"], n, c.most_common(3)))
    print("\n※ A 높고 B 낮음 ⇒ 근거 요구가 날조를 닫는다(그 문자열이 출시본)."
          "\n  A·B 둘 다 높음 ⇒ 요구로는 안 닫힌다 — 다른 축(DAG·경쟁)으로."
          "\n  D 가 낮으면 결손이 **문맥에서** 오는 것이고, 높으면 요청 자체가 유인이다."
          "\n  `QUOTE_UNREAL` 이 나오면 인용까지 지어낸 것 — 엔진 검산이 그것을 잡는다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
