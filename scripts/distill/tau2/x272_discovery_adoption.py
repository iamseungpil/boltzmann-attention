# -*- coding: utf-8 -*-
r"""x272 — `[DISCOVERY-REQUIRED]` 는 닿는데 왜 안 듣는가 (유료 0 · 엔진 0 · [[18]] 정보-맞춘 격리).

## 왜 (C441 후속 · 라이브 실측)

`bank_lever_070_20260812d`: 사이드카에 `DISCOVERY-REQUIRED` **5건**(=실제 전달) · 그런데
`unlock_discoverable_agent_tool` **0회** · `call_discoverable_agent_tool` **0회**. 두 시행 다
발견 체인에 **한 번도 진입하지 않았다**. `arrived` 통과 · `acted` 실패 = C425·[[64]] 계보.

## 가르는 것

  A_LIVE      실제 궤적 문맥(그 턴까지) + 출시본 문구       ← 결손 재현 기대
  B_ISO       **손님 요청 + 문구만**(격리)                   ← 여기서 되면 병은 부하/경쟁
  C_NOTEXT    격리인데 문구 **없음** (부정 통제·[[57]])       ← 이게 높으면 문구 공로 아님
  D_LIVE_ONLYTEXT  실제 문맥에서 **우리 다른 문장 전부 제거** + 문구  ← 경쟁 여부 분리

계기: 다음 응답이 ⓐ`KB_search_bm25` ⓑ`unlock_discoverable_agent_tool` ⓒ`transfer` ⓓ산문.
**체인 진입 = ⓐ 또는 ⓑ** 를 부르는 것(문구가 요구하는 첫 걸음).

⚠문구는 **사이드카에서 그대로 가져온다**(출시 문구 = 측정 문구·[[03b]]). 프로브가 새로 쓰지 않는다.

실행(리모트·GPU1): T2_PROBE_URL=http://localhost:8141/v1/chat/completions python3 x272_discovery_adoption.py [N]
"""
import collections
import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from x216_read_and_offset import chat                               # noqa: E402
import x256_dispatcher_write_probe as X256                          # noqa: E402

SIMS = os.environ.get("X272_SIMDIR", "/home/woori/scratch/tau2-bench/data/simulations")
LOGS = os.environ.get("X272_LOGS", "/home/woori/scratch/logs")
TAG = os.environ.get("X272_TAG", "bank_lever_070_20260812d")
TASK = "task_070"
MARK = "[DISCOVERY-REQUIRED]"


def shipped_text():
    """출시 문구를 사이드카에서 축자로 가져온다 — 프로브가 문구를 만들지 않는다([[03b]])."""
    p = os.path.join(LOGS, "fb_%s.jsonl" % TAG)
    for line in io.open(p, encoding="utf-8", errors="replace"):
        try:
            o = json.loads(line)
        except Exception:
            continue
        t = o.get("text") or ""
        if MARK in t:
            return " ".join(t.split())
    return None


def classify(msg):
    tcs = (msg or {}).get("tool_calls") or []
    names = [str((tc.get("function") or {}).get("name") or "") for tc in tcs]
    if any("unlock_discoverable" in n for n in names):
        return "UNLOCK"
    if any("KB_search" in n for n in names):
        return "KB_SEARCH"
    if any("transfer" in n for n in names):
        return "TRANSFER"
    if names:
        return "OTHER(%s)" % names[0][:22]
    return "PROSE"


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 8
    txt = shipped_text()
    if not txt:
        print("사이드카에서 출시 문구를 못 찾았다 — 중단(문구를 새로 만들지 않는다).")
        return 2
    print("출시 문구(%d자): %s\n" % (len(txt), txt[:150]))

    d = json.load(io.open(os.path.join(SIMS, TAG, "results.json"), encoding="utf-8"))
    sim = [s for s in d["simulations"]
           if s["task_id"] == TASK and (s.get("reward_info") or None) is not None][0]
    msgs = sim["messages"]
    # 발견 요구가 나간 자리 ≈ 마지막 손님 발화 직후. 그 시점까지를 라이브 문맥으로 쓴다.
    cut = max(i for i, m in enumerate(msgs) if m.get("role") == "user")
    users = [str(m.get("content") or "") for m in msgs[:cut + 1] if m.get("role") == "user"]
    tools = X256.U.tools_of(sim)

    live = X256.build(sim, cut, True)
    live_clean = X256.build(sim, cut, False)          # 우리 문장 제거판
    ask = " --- ".join(users[-2:])

    arms = (("A_LIVE", live + "\n[system] " + txt),
            ("B_ISO", ask + "\n\n[system] " + txt),
            ("C_NOTEXT", ask),
            ("D_LIVE_ONLYTEXT", live_clean + "\n[system] " + txt))
    print("문맥 크기: live %d · clean %d · iso %d\n" % (len(live), len(live_clean), len(ask)))
    for label, body in arms:
        c = collections.Counter()
        for i in range(n):
            try:
                r = chat(body, tools, 0.0 if i == 0 else 0.7, 300)
            except Exception as e:
                r = {"content": "ERR %s" % type(e).__name__}
            c[classify(r)] += 1
        entered = c["UNLOCK"] + c["KB_SEARCH"]
        print("  %-16s 체인진입 %d/%d   %s" % (label, entered, n, c.most_common(4)))
    print("\n※ B 높고 A 낮음 ⇒ 병은 **부하/경쟁**(문맥이 문구를 이긴다) — 처방은 경쟁 제거."
          "\n  A·B 둘 다 낮음 ⇒ 병은 **문구**(무엇을 하라는지 실행 가능하게 못 댐)."
          "\n  C 가 높으면 문구의 공로가 아니다(프로브 무효·[[57]]).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
