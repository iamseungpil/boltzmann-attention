# -*- coding: utf-8 -*-
r"""x572 — **회수된 라이브 프롬프트 위에서** 자기-정박을 잰다 (유료 0).

## 왜 이 프로브가 앞의 것들과 다른가

문맥을 `render(영속 메시지)` 로 짓지 않는다. `T2_PROMPT_DUMP` 가 회수한 **모델이 실제로 본
그 바이트**를 쓴다. 영속 궤적에는 우리 층 비커밋 주입이 없어서(t7366 실측 **5,216자** 차이)
그 위에서 지은 격리는 다른 대화였다 — 오늘 두 번의 iso↔live 불일치가 그것이었다([[78]] 확장).

## 실측 (t7366 스모크 `task_016#s626729`)

진단 단언이 **프롬프트에 실제로 도착해 있다**(turn 21 부터 16개 프롬프트):
    `A separate check was run … It answers: Silver Rewards Card | 2025-10-12 …`
그런데 에이전트는 그 뒤 msg[20]·[22]·[24] 에서 **전부 Bronze** 라고 말한다.
프롬프트 안 단어 수: turn 21 Silver 12 : Bronze 12 → turn 29 12:15~17 → turn 52 21:22.
**자기 발화가 쌓이며 우리 단언 한 줄을 수적으로 덮는다.**

⇒ 결손은 **전달이 아니다**([[62]] ②). 물어야 할 것은: 그 모순을 **가리키기만** 하면 갈리는가.

## 팔 — 새 사실 0 · 지시 0

    A_asis   회수된 프롬프트 그대로            ← 재현 게이트(Bronze 가 나와야 한다)
    B_point  + *"당신 답이 방금 받은 검사와 다른 기록을 댄다"*   ← **이름도 값도 새로 안 준다**
    N_len    길이만 맞춘 무관 문장([[57]])

⛔ASK 를 붙이지 않는다. 라이브는 지시 없이 이어 쓴다 — 오늘 `x569` 에서 물음에 처치를 섞어
  결손을 스스로 지웠다.

## 채점 — 닫힌 술어

답이 원장의 어느 이름을 대는가만 센다(Silver ↔ Bronze). 어느 것이 정답인지 엔진은 모른다.

사용: PYTHONPATH=. py -3 x572_anchor_on_live_prompt_iso.py --port 8140 [--wiring-only]
"""
import argparse
import collections
import gzip
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import x559_016_row_pick_iso as X559                                # noqa: E402

NL = chr(10)
SC = os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026", "sim_results",
                  "fb_bank_t7366_smoke_20260827.jsonl.gz")
MARK = "separate check was run"


def prompts(path=SC, simtag="task_016#s626729", minlen=4000):
    out = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                r = json.loads(ln)
            except Exception:
                continue
            if (r.get("kind") == "prompt" and r.get("simtag") == simtag
                    and len(r.get("text") or "") > minlen):
                out.append(r)
    out.sort(key=lambda r: (r.get("turn") or 0, len(r.get("text") or "")))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8140)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--wiring-only", action="store_true")
    a = ap.parse_args(argv)

    pr = [r for r in prompts() if MARK in str(r.get("text") or "")]
    if not pr:
        print("단언을 실은 프롬프트가 없다 — 판정하지 않는다([[25]])", file=sys.stderr)
        return 2
    base = str(pr[0].get("text"))
    ans = re.search(r"It answers:\s*([^|\n]+)", base)
    ans = (ans.group(1).strip() if ans else "")
    print("# x572 — 회수된 라이브 프롬프트 turn=%s · %d자" % (pr[0].get("turn"), len(base)))
    print("   프롬프트가 실은 검사 답: %s" % ans)
    print("   프롬프트 안 이름 수: Silver %d · Bronze %d"
          % (len(re.findall("Silver Rewards Card", base)),
             len(re.findall("Bronze Rewards Card", base))))
    say = (NL + NL + "Error: [SOURCE] the record you are naming is not the one the check you were "
           "given names. Either name the record that check names, or say plainly why that check "
           "is wrong. Do not carry forward a record you named earlier without re-checking it.")
    if a.wiring_only:
        print("--- B_point 문면 ---")
        print("   " + " ".join(say.split()))
        print("   ⚠문면에 이름 누출: %s"
              % ([n for n in ("Silver", "Bronze") if n in say] or "없음"))
        return 0

    adds = {"A_asis": "", "B_point": say,
            "N_len": NL + NL + ("[note] " + "the information gathered so far in this "
                                "conversation remains current. " * 5)[:len(say)]}
    print()
    print("%-8s %-5s %-58s %s" % ("팔", "temp", "답 앞부분", "지목"))
    print("-" * 104)
    tally = collections.defaultdict(collections.Counter)
    for nm in ("A_asis", "B_point", "N_len"):
        body = base + adds[nm]
        for tp, cnt in ((0.0, 1), (a.temp, a.n)):
            for _ in range(cnt):
                try:
                    rep = " ".join(str(X559.gen(a.port, body, 200, tp)).split())
                except Exception as e:
                    print("%-8s %-5s 호출 실패: %r" % (nm, tp, e))
                    continue
                s_n = len(re.findall("Silver", rep))
                b_n = len(re.findall("Bronze", rep))
                pick = ("Silver" if s_n > b_n else ("Bronze" if b_n > s_n else "무지목/동수"))
                tally[nm][pick] += 1
                print("%-8s %-5s %-58s %s (S%d/B%d)" % (nm, tp, rep[:58], pick, s_n, b_n))
    print()
    print("## 지목 분포")
    for nm in ("A_asis", "B_point", "N_len"):
        print("   %-8s %s" % (nm, dict(tally[nm])))
    print()
    print("⚠A_asis 가 이미 Silver 면 결손이 아니다([[62]] 2b). N_len 이 같으면 길이다([[57]]).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
