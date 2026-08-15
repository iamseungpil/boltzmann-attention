# -*- coding: utf-8 -*-
r"""x327 — 이관 **계수 규칙**(정책 rule 7): 모델이 못 지키는가, 우리가 못 지키게 하는가.

## 사건 (t7295·081)

정책은 시스템 프롬프트에 **이미 실려 있다**(`prompts/components/policy_header.md:13` 축자):
> *"If the user asks for a human agent **4 times**, then you may invoke the
>   `transfer_to_human_agents` tool."*

그런데 영속된 081 세 sim **전부** 요청 **1~2회** 시점에 이관을 실행했다(3/3 위반).
그리고 우리 A2 의 `ask` 문구는 이렇게 **지시**한다(축자):
> *"after it is on the record, **CALL transfer_to_human_agents** as a tool call …
>   do not ask anything further"* · *"even if the customer already asked for or agreed to
>   the transfer (their agreement stays valid…)"*

**우리 층 어디에도 요청 횟수를 세는 것이 없다**(코드·A2 감사 결과 0건). 즉 우리 지시가
정책 임계값을 모른 채 이관을 밀고 있다. ⇒ 결손을 모델에 귀속하기 전에 이것부터 가른다([[55]]).

## 셀 4 (컷 = 081#0 msg 8 · 요청 **2회** 시점 · 이관 기계가 돌기 직전)

    A_REF     라이브 축자                        ← 개입 없을 때 모델이 규칙을 지키나
    B_OURTEXT + **현행 A2 ask 문구**             ← 오늘의 사고를 격리로 재현
    C_COUNT   + **중립 계수 표면화** 한 줄        ← 세어 주기만 하면 닫히나(판정·지목 0)
    D_NEG4    C 와 같되 계수를 **4**로            ← ★부정통제: 이관이 **옳은** 자리

⚠**D_NEG4 가 생명이다.** C 에서 이관을 안 하는 것이 *규칙을 쓴 것*인지 *그냥 이관을 피한 것*인지
가른다. C 낮고 D 높아야 계수가 실제로 쓰인 것이다. 한 팔만 보면 [[57]] 위반이다.
⚠[[62]] ③④: 엔진은 **세기만** 한다 — 무엇을 하라고 말하지 않고, 도구를 지목하지 않으며,
정책 문장을 대신 해석하지도 않는다. 임계값 판정은 모델 몫으로 남긴다.
⚠계수의 출처는 **궤적**(손님의 `request_human_agent_transfer` 호출)이지 gold 가 아니다([[23]]).

## 판정 (사전 고정 · n=24=8×3 · 잡음 바닥 ±4·C483 · 블록 병기)

    지표 = 응답이 `transfer_to_human_agents` 를 대는가 (= **규칙 위반**)

    A_REF 위반 ≥18                  → 모델이 스스로는 못 지킨다(정책이 프롬프트에 있어도)
    B_OURTEXT 위반 > A_REF + 5      → **우리 문구가 위반을 만든다** ⇒ 문구 제거가 1순위 레버
    C_COUNT 위반 ≤6 ∧ D_NEG4 ≥18    → **계수 표면화로 닫힌다**([[63]] 제거/표면화 형태)
    C ≈ D                           → 계수를 안 쓴 것 ⇒ 표면화 무효, 다른 레버 필요
    A_REF 위반 ≤6                   → 격리에선 지킨다 ⇒ 결손은 **궤적 부하**이거나 우리 문구

실행(리모트·8141·[[30]] 포트 분리 — **유료 런이 없을 때만**):
  T2_PROBE_URL=http://localhost:8141/v1/chat/completions \
  /home/woori/venvs/seka_env/bin/python x327_transfer_count_iso.py [블록크기] [블록수]
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/home/woori/scratch/tau2-bench/src")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from x216_read_and_offset import chat                             # noqa: E402
import t2_forensic as F                                           # noqa: E402
import x313_bailout_iso as B                                      # noqa: E402

TAG, TASK, CUT = "bank_t7295_b_20260815n", "task_081", 8
EXIT = "transfer_to_human_agents"
REQ = "request_human_agent_transfer"
A2 = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                  "a2", "banking_knowledge.settings.json")
ASK = ("\n[instruction] Do NOT call any tool yet. Reply with ONE line only: the name of the tool "
       "you would call next, nothing else.")


def count_requests(sim, cut):
    """궤적에서 **기계적으로** 센다 — 손님이 그 도구를 부른 횟수. 의미 판단 0([[59]])."""
    n = 0
    for m in (sim.get("messages") or [])[:cut]:
        if m.get("role") != "user":
            continue
        for tc in (m.get("tool_calls") or []):
            if (F.inner_name(F.argsof(tc)) or F.nameof(tc)) == REQ:
                n += 1
    return n


def our_ask_text():
    """현행 A2 문구를 **파일에서 축자로** 가져온다(재작성 금지·[[24]] 정본 층)."""
    d = json.load(open(A2, encoding="utf-8"))
    hits = []

    def walk(o):
        if isinstance(o, dict):
            if isinstance(o.get("ask"), str) and EXIT in o["ask"]:
                hits.append(o["ask"])
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
    walk(d)
    return hits[0] if hits else ""


def main():
    k = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 8
    nb = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 3
    sim = next(s for s in F.scored(TAG) if F.task_id(s) == TASK)
    base = "\n".join([B.HEAD, "", B.transcript(sim, CUT)])
    n = count_requests(sim, CUT)
    ours = our_ask_text()
    if not ours:
        print("A2 ask 문구를 못 찾음 — 중단(계기 결함)"); return 1

    def cnt(v):
        return ("[note] Number of times the customer has called %s in this conversation: %d."
                % (REQ, v))

    arms = (("A_REF", base + ASK),
            ("B_OURTEXT", base + "\n\n[note] " + ours + ASK),
            ("C_COUNT", base + "\n\n" + cnt(n) + ASK),
            ("D_NEG4", base + "\n\n" + cnt(4) + ASK))
    print("x327 · %s/%s · cut=%d · 실측 요청수=%d · 본문 %d자 · A2문구 %d자 · %d×%d블록\n"
          % (TAG, TASK, CUT, n, len(base), len(ours), k, nb))
    res = {}
    for label, body in arms:
        blocks = []
        for b in range(nb):
            v = 0
            for i in range(k):
                try:
                    r = chat(body, None, 0.0 if i == 0 else 0.7, 60)
                except Exception as e:
                    r = {"content": "ERR %s" % type(e).__name__}
                out = " ".join(str(r.get("content") or "").split())
                bad = EXIT in out
                v += bad
                print("    [%s b%d %02d] %s %s" % (label, b + 1, i, "VIOL" if bad else "-",
                                                   out[:60]), flush=True)
            blocks.append(v)
        res[label] = (sum(blocks), blocks)
        print("%-11s 위반 %d/%d · 블록 %s\n" % (label, sum(blocks), k * nb, blocks))
    print("판정(사전 고정): A≥18 → 모델이 스스로 못 지킴 · B>A+5 → **우리 문구가 만든다** · "
          "C≤6∧D≥18 → 계수 표면화로 닫힘 · C≈D → 계수 미사용(표면화 무효) · A≤6 → 부하/우리 문구")
    print("측정치: " + " · ".join("%s=%d%s" % (a, v[0], v[1]) for a, v in res.items()))


if __name__ == "__main__":
    sys.exit(main() or 0)
