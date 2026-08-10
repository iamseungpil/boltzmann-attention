# -*- coding: utf-8 -*-
r"""x212 — 010 의 **부하**를 정보-맞춘 격리 replay 로 잰다 (유료 0 · 엔진 변경 0).

## 왜 이 형태인가 (§1.4 · [[18]])

> $\text{load}(s) = p_{iso}(s) - p_{traj}(s) > 0$ — 격리하면 푸는데 궤적서 못 푼다.
> ★측정 규율: $p_{iso}$ 는 **에이전트가 그 지점에 실제로 갖고 있던 정보와 맞춰야** 한다.

C394 가 세운 것: 정의 문장이 궤적에 **축자로 들어온 sim 43개 중 통과 4**(9.3%), 안 들어온
37개는 **0**. 같은 정의를 짧고 깨끗하게 주면 x210 에서 **8/8**. 그 사이가 부하다.

그러나 x210 의 8/8 은 **원장 4행 + 정의만** 준 팔이라 정보가 실제 궤적보다 **빈약하다** —
규율대로면 그건 부하가 아니라 정보량 차이일 수 있다. 그래서 여기서는 **실제 실패 궤적의
문맥을 그대로** 쓰고, 우리 층이 얹을 **한 줄만** 더한다.

## 팔 (실패 sim 중 **정의가 문맥에 있던 것**만 고른다)

  RAW      그 지점의 실제 문맥 그대로 + 손님의 실제 되묻기          ← $p_{traj}$
  COMPACT  같은 문맥 + **우리가 얹을 한 줄**(정의 인용을 값으로)     ← 부하 축소 레버
  STRIP    같은 문맥에서 **정의를 지운 것** (부정 통제)              ← 0 이어야 한다

⚠**COMPACT 줄에는 지시가 없다** — 값(인용)과 출처만 적는다(규칙 E: 메인은 값만).
⚠정의 인용은 **궤적에 실재하는 문자열**을 그대로 쓴다(엔진이 지어내지 않는 것과 같은 규율).

실행: python x212_load_replay.py [N]
"""
import collections
import glob
import gzip
import json
import os
import re
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from x200_disclaimer_ab import CAUSE, ESCAPE                      # noqa: E402

URL = os.environ.get("T2_PROBE_URL", "http://localhost:8141/v1/chat/completions")
MODEL = os.environ.get("T2_PROBE_MODEL", "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8")
SIMS = ["/home/woori/scratch/tau2-bench/data/simulations/*/results.json",
        "/home/woori/workspace_common/boltzmann-attention-pi/reports/facet_rft_2026/"
        "sim_results/*.json.gz"]
DEFN = "REJECTED - the user has too many referral processes going on"
SIG = "too many referral processes"
BUDGET = int(os.environ.get("T2_X212_BUDGET", "60000"))   # 문자 예산(문맥 길이 통제)


def load_cases(limit=16):
    """**실패했고 정의가 문맥에 있던** 010 sim 을 고른다."""
    out, seen = [], set()
    for pat in SIMS:
        for p in sorted(glob.glob(pat)):
            try:
                f = gzip.open(p, "rt", encoding="utf-8") if p.endswith(".gz") else open(p, encoding="utf-8")
                d = json.load(f)
            except Exception:
                continue
            if not isinstance(d, dict):
                continue
            for s in d.get("simulations") or []:
                if not isinstance(s, dict) or s.get("task_id") != "task_010":
                    continue
                if (s.get("reward_info") or {}).get("reward") == 1:
                    continue
                msgs = s.get("messages") or []
                blob = "\n".join(str(m.get("content") or "") for m in msgs)
                if SIG not in blob:
                    continue
                key = (len(msgs), blob[:200])
                if key in seen:
                    continue
                seen.add(key)
                out.append((os.path.basename(p), s.get("trial"), msgs))
                if len(out) >= limit:
                    return out
    return out


def render(msgs, upto=None):
    """그 지점까지의 문맥을 **역할 표시와 함께** 문자열로. 예산 초과 시 앞을 자른다."""
    parts = []
    for m in msgs[:upto]:
        c = " ".join(str(m.get("content") or "").split())
        tc = m.get("tool_calls") or []
        if tc:
            for t in tc:
                fn = t.get("function") or t
                a = fn.get("arguments")
                a = a if isinstance(a, str) else json.dumps(a, ensure_ascii=False)
                parts.append("[%s calls %s(%s)]" % (m.get("role"), fn.get("name"), a[:200]))
        if c:
            parts.append("[%s] %s" % (m.get("role"), c))
    txt = "\n".join(parts)
    return txt[-BUDGET:] if len(txt) > BUDGET else txt


def last_probe_point(msgs):
    """손님이 *'그건 왜를 답하지 않는다'* 며 **처음 되묻는** 지점.

    ⚠**자기적발 (1차 실행)**: 구판은 조건에 맞는 **마지막** user 턴을 골랐는데, 그건 이미
      *"네, 상담원으로 연결해 주세요"* 라고 **이관을 요청한 뒤**였다. 그 지점에서는 이관이
      옳은 행동이라 세 팔이 전부 6/6 이관으로 같아졌다 — 부하를 잰 것이 아니라 **재는 자리를
      틀린 것**이다. 결정이 갈리는 자리는 **이관 요청 전 첫 되묻기**다.
    """
    for i, m in enumerate(msgs):
        if m.get("role") != "user":
            continue
        c = str(m.get("content") or "").lower()
        if "transfer" in c or "human agent" in c or "###" in c:
            continue                      # 이관을 요청한 턴은 결정 지점이 아니다
        if ("why" in c or "reason" in c or "specific" in c) and i > 2:
            return i
    return max((i for i, m in enumerate(msgs) if m.get("role") == "user"), default=len(msgs) - 1)


def ask(prompt, temp):
    body = {"model": MODEL, "temperature": temp, "max_tokens": 220,
            "messages": [{"role": "user", "content": prompt}]}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return " ".join((json.load(r)["choices"][0]["message"]["content"] or "").split())


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 6
    # `--slice i/k` — 사례 목록을 k 조각으로 나눠 i 번째만 돈다(두 GPU 에 나눠 싣기 위함).
    part, parts = 0, 1
    for a in sys.argv[1:]:
        if a.startswith("--slice"):
            v = a.split("=", 1)[-1] if "=" in a else sys.argv[sys.argv.index(a) + 1]
            part, parts = (int(x) for x in v.split("/"))
    cases = load_cases()
    if parts > 1:
        cases = [c for i, c in enumerate(cases) if i % parts == part]
    print("사례 %d개 (slice %d/%d · n=%d)" % (len(cases), part, parts, n))
    if not cases:
        print("⚠조건에 맞는 sim 이 없다 (실패 + 정의 문맥 도달)")
        return 1
    out = {}
    agg = collections.Counter()
    for tag, trial, msgs in cases:
        i = last_probe_point(msgs)
        ctx = render(msgs, i)
        askmsg = " ".join(str(msgs[i].get("content") or "").split())
        present = SIG in ctx
        print("\n" + "=" * 92)
        print("%s trial=%s · 문맥 %d자 · 정의 문맥내 실재: %s" % (tag, trial, len(ctx), present))
        print("손님: %s" % askmsg[:180])
        if not present:
            print("  ⚠예산 절단으로 정의가 잘렸다 — 이 사례는 건너뛴다(통제 오염).")
            continue
        # 우리 층이 얹을 **한 줄** — 값과 출처만, 지시 없음
        line = ("\nFrom a document already retrieved (Understanding Credit Card Referral "
                "Statuses): \"%s\"." % DEFN)
        strip = re.sub(re.escape(DEFN), "[removed]", ctx)
        strip = re.sub(r"[^\n]*too many referral processes[^\n]*", "[removed]", strip)
        arms = [("RAW", ctx), ("COMPACT", ctx + line), ("STRIP", strip)]
        for arm, body in arms:
            c = collections.Counter()
            first = None
            for k in range(n):
                p = (body + "\n\nThe customer now asks:\n" + askmsg
                     + "\n\nAnswer the customer in two or three sentences.")
                try:
                    t = ask(p, 0.0 if k == 0 else 0.7)
                except Exception as e:
                    t = "ERR %s" % type(e).__name__
                first = first or t
                lo = t.lower()
                c["이유O" if any(x in lo for x in CAUSE) else "이유X"] += 1
                c["이관O" if any(x in lo for x in ESCAPE) else "이관X"] += 1
            out["%s/%s/%s" % (tag, trial, arm)] = [c["이유O"], n]
            agg[arm + "/hit"] += c["이유O"]
            agg[arm + "/n"] += n
            agg[arm + "/esc"] += c["이관O"]
            print("  %-8s 이유 %d/%d · 이관 %d/%d" % (arm, c["이유O"], n, c["이관O"], n))
            print("      | " + (first or "")[:220])
    print("\n" + "=" * 92)
    print("합계 (판정 지표 = **이유 진술**. 이관은 관측일 뿐 — gold 는 손님이 제출하는 것이고")
    print("      그 조건은 *구체적 사유*이지 이관 여부가 아니다.)")
    for arm in ("RAW", "COMPACT", "STRIP"):
        if agg[arm + "/n"]:
            print("  %-8s 이유 %3d/%-3d (%.0f%%) · 이관 %d"
                  % (arm, agg[arm + "/hit"], agg[arm + "/n"],
                     100.0 * agg[arm + "/hit"] / agg[arm + "/n"], agg[arm + "/esc"]))
    json.dump(dict(out, _agg=dict(agg)),
              open(os.environ.get("T2_X212_OUT", "x212_out.json"), "w"), indent=1)
    print("\n※ RAW 가 낮고 COMPACT 가 높으면 → **부하 확정**이고 레버는 그 한 줄이다."
          "\n  RAW ≈ COMPACT 면 → 한 줄로는 부하가 안 줄고 다른 형태를 찾아야 한다."
          "\n  STRIP 이 RAW 와 같으면 → 정의는 애초에 안 읽히고 있었다(도달≠사용).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
