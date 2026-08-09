# -*- coding: utf-8 -*-
r"""x193 — 목적 축 형식화가 `NONE` 을 낸 것은 **모델이 아니라 우리 창** 때문인가 (유료 0).

## 왜

라이브 런 r(C374)에서 두 sim 다 `[T2_OBJ_AXIS] raw='NONE'` 이 나왔고, 그 결과 결정 블록의
순위(`runners`)가 **빈 채로** 나가고 **D1c 는 아예 실행되지 않았다**. 즉 x191 의 `B_rank` 와
x192 의 재질의는 라이브에서 한 번도 안 돌았다.

구판 `formalize_objective_axis` 는 대화를 전부 이어 붙인 뒤 **꼬리 6000자**만 봤다. 목적을
말하는 문장은 손님의 **첫 발화**에 있고 결정점은 턴 24~26 이므로, 그 창에는 검색된 KB 문서만
들어온다(오프라인 계산: tool+user 총 099 28,523자·100 15,044자·첫 발화 창 밖 2/2).

이 프로브는 그 진단을 **모델에게 직접 물어** 확인한다 — 같은 대화·같은 프롬프트·같은 축 목록에
**창만 바꾼다**. 유료 런으로 검증할 일이 아니다([[09]]).

## 축

  window : tail6000  구판(전체 연결 후 꼬리 6000자)   ← 라이브에서 NONE 을 낸 그 창
           excerpt   신판(`LG._excerpt` 항목별 3000자 절단 + 총예산 90000)
  scope  : upto      결정점까지의 메시지만 (라이브가 실제로 본 것)
           full      대화 전체 (상한)
  model  : 8140 32B · 8141 14B      task : 실제 런 r 의 099·100 궤적

판정은 라이브와 **같은 검사**를 쓴다 — 자유 생성 후 A2 축 집합 원소 여부(최장 우선).

## 읽는 법

  · tail6000 이 NONE 이고 excerpt 가 축을 내면  → 원인은 창(우리 배관). 수정이 곧 처방.
  · 둘 다 NONE 이면                              → 창이 아니다. 프롬프트·축 설명을 봐야 한다.
  · 둘 다 축을 내면                              → 라이브의 NONE 은 제3의 원인(재현 실패).

실행: python x193_objective_axis_window.py [N]
"""
import gzip
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
from gate_interpreter import load_domain_a2                     # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "..", "..", ".."))
RUN = os.path.join(REPO, "reports", "facet_rft_2026", "sim_results",
                   "bank_anchorslot_20260809r.json.gz")
MODELS = [("32B", "http://localhost:8140/v1/chat/completions",
           "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8"),
          ("14B", "http://localhost:8141/v1/chat/completions",
           "Qwen/Qwen2.5-14B-Instruct")]
# 라이브가 블록을 낸 턴 (사이드카 `turn` 기록). 그 시점까지가 형식화가 실제로 본 문맥이다.
DECISION_TURN = {"task_099": 26, "task_100": 24}


def chat(url, model, prompt, temp):
    body = {"model": model, "temperature": temp, "max_tokens": 24,
            "messages": [{"role": "user", "content": prompt}]}
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return " ".join((json.load(r)["choices"][0]["message"]["content"] or "").split())


def decide(raw, names):
    """라이브 `formalize_objective_axis` 와 **같은** 판정 — 축 집합 원소만 받는다([[22]])."""
    hit = sorted((a for a in names if a and a.lower() in raw.lower()), key=len, reverse=True)
    return hit[0] if hit else None


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    a2 = load_domain_a2("banking_knowledge")
    spec = next(s for s in a2["ledger_metrics"] if s.get("objective_axis_prompt"))
    tpl = spec["objective_axis_prompt"]
    axes = (a2.get("policy_ontology") or {}).get("axes") or {}
    names = list(axes)
    listing = "\n".join("  %s — %s" % (k, axes[k]) for k in names)

    sims = json.load(gzip.open(RUN, "rt", encoding="utf-8"))["simulations"]
    print("axes=%d · n=%d · run=%s" % (len(names), n, os.path.basename(RUN)))

    rows = []
    for s in sims:
        task = s["task_id"]
        msgs = s["messages"]
        cut = DECISION_TURN.get(task, len(msgs))
        for scope, sel in (("upto", msgs[:cut]), ("full", msgs)):
            tx = [str(m.get("content") or "") for m in sel
                  if m.get("role") in ("tool", "user")]
            wins = {"tail6000": " ".join("\n".join(tx).split())[-6000:],
                    "excerpt": "\n---\n".join(LG._excerpt(tx))}
            for win, hay in wins.items():
                prompt = tpl.format(axes=listing, text=hay)
                for name, url, model in MODELS:
                    got = []
                    for _ in range(n):
                        try:
                            raw = chat(url, model, prompt, 0.0)
                        except Exception as e:
                            raw = "ERR %r" % (e,)
                        got.append(decide(raw, names) or "NONE")
                    hits = sum(1 for g in got if g != "NONE")
                    top = max(set(got), key=got.count)
                    rows.append((task, scope, win, name, len(hay), hits, n, top))
                    print("  %-9s %-5s %-9s %-4s chars=%-6d axis %d/%d  top=%s"
                          % (task, scope, win, name, len(hay), hits, n, top))

    print("\n요약 — 창별 축 획득률")
    for win in ("tail6000", "excerpt"):
        sub = [r for r in rows if r[2] == win]
        print("  %-9s %d/%d" % (win, sum(r[5] for r in sub), sum(r[6] for r in sub)))
    print("\n※ 판정: tail6000 이 지고 excerpt 가 이기면 라이브 NONE 의 원인은 우리 창이다.")


if __name__ == "__main__":
    main()
