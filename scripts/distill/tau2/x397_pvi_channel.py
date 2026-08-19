# -*- coding: utf-8 -*-
r"""x397 — **PVI(점별 V-정보) + 채널 여유**: 형식 효과를 *정보량*으로 환산한다.

## 왜 (사용자 지시 2026-08-19: "PVI 먼저 돌려라")
Blackwell 정보서열은 **베이즈-합리 관측자에게는 같은 정보의 재포장이 성능을 바꿀 수 없다**고 말한다.
그런데 우리 실측은 재포장(대화체↔원장체)에서 행동이 갈렸다. 두 가지 중 하나다:
  ⒜ 정보량이 실제로 다르다(우리가 팔을 잘못 짰다·교락) ⒝ 관측자가 **계산 제약**을 받는다.
`V-information`(Xu 2020 / Ethayarajh 2110.08420)은 ⒝를 정량화하는 틀이고, 그 점별 형태 **PVI** 는
파인튜닝 없이 **조건부 로그가능도 차**로 근사할 수 있다:

    PVI(x → y) ≈ log2 p(y | x) − log2 p(y | ∅)

여기서 y = *그 자리에서 옳은 호출*, x = 팔이 준 재료, ∅ = 재료 없는 기준선.
⇒ 팔별 평균 PVI = **그 형식이 이 모델에게 전달한 유용 정보량(bit)**. 형식만 바꿔 PVI 가 변하면
   그것이 ⒝ 의 직접 증거다(정보는 같은데 *쓸 수 있는* 정보가 다르다).

## 같이 재는 것 — 채널 여유 (E5 로짓 프로브)
결정 시점 **첫 토큰**의 분포에서 `{`(JSON 호출 개시) 대 알파벳(산문 개시) 질량을 가른다.
행동 지표(EMIT/말만)와 달리 **연속량**이라 팔 간 비교가 표본에 덜 휘둘린다.

⚠이 스크립트는 **생성하지 않는다**(teacher-forced 스코어링 + 1토큰 분포). 채점에 gold 를 쓰지만
  프롬프트에는 안 넣는다 — 재료는 `x395` 의 팔 구성을 **그대로 import** 해서 쓴다([[67]] 사본 금지).
⚠포트 8141(GPU1)만 쓴다 — 8140 은 재베이스라인 런이 점유 중이다.

사용: py -3 x397_pvi_channel.py [--port 8141] [--arms A_min,B_full,...]
"""
import argparse
import collections
import io
import json
import math
import os
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import t2_forensic as F  # noqa: E402
import x395_compliance_iso as X  # noqa: E402

MODEL = X.MODEL
# Qwen2.5-Instruct = ChatML. 템플릿을 축자로 적는다(렌더 불일치는 계기 결함이 되므로 G0 에서 확인 대상).
TPL = ("<|im_start|>system\n%s<|im_end|>\n<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n")


def post(port, path, body, timeout=300):
    req = urllib.request.Request("http://127.0.0.1:%d%s" % (port, path),
                                 data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def score_suffix(port, prefix, suffix):
    """log p(suffix | prefix) — teacher forced. 자연로그 합과 토큰 수를 돌려준다."""
    d = post(port, "/v1/completions",
             {"model": MODEL, "prompt": prefix + suffix, "max_tokens": 0,
              "echo": True, "logprobs": 1})
    lp = d["choices"][0]["logprobs"]
    offs, toks = lp["text_offset"], lp["token_logprobs"]
    n0 = len(prefix)
    tot, cnt = 0.0, 0
    for off, v in zip(offs, toks):
        if off >= n0 and v is not None:
            tot += v
            cnt += 1
    return tot, cnt


def first_token_dist(port, prefix, k=20):
    d = post(port, "/v1/completions",
             {"model": MODEL, "prompt": prefix, "max_tokens": 1, "logprobs": k})
    tl = (d["choices"][0]["logprobs"].get("top_logprobs") or [{}])[0] or {}
    return {str(t): float(v) for t, v in tl.items()}


def channel_mass(dist):
    """첫 토큰 질량을 **JSON 개시** 대 **산문 개시** 로 가른다(해석 0·문자만 본다)."""
    j = t = o = 0.0
    for tok, lp in dist.items():
        p = math.exp(lp)
        s = tok.strip()
        if s.startswith("{") or s.startswith('{"') or s == '{"':
            j += p
        elif s[:1].isalpha() or s.startswith('"'):
            t += p
        else:
            o += p
    return j, t, o


def build_cases(docs):
    """x395 와 **동일한** 표적 12개."""
    cases = []
    for tag in X.TAGS:
        for sim in F.scored(tag, X.SUF):
            if ((sim.get("reward_info") or {}).get("reward") or 0) >= 1.0:
                continue
            cn = X.called_names(sim)
            for ck in ((sim.get("reward_info") or {}).get("action_checks") or []):
                if ck.get("action_match"):
                    continue
                aa = ck.get("action") or {}
                ar = aa.get("arguments") or {}
                nm = str(ar.get("agent_tool_name") or ar.get("user_tool_name")
                         or ar.get("discoverable_tool_name") or aa.get("name") or "")
                if not nm or cn.get(nm):
                    continue
                pl = X.proc_lines(docs, nm)
                if not pl:
                    continue
                body = " ".join(" ".join(str(m.get("content") or "").split())
                                for m in (sim.get("messages") or []) if m.get("role") == "tool")
                if not [s for s in pl if s.split("] ", 1)[-1][:55] in body]:
                    continue
                cases.append({"task": F.task_id(sim), "trial": sim.get("trial"),
                              "tool": nm, "accept": sorted(X.accept_names(ck)),
                              "lines": pl, "sim": sim})
    seen, uniq = set(), []
    for c in cases:
        k = (c["task"], c["tool"])
        if k in seen:
            continue
        seen.add(k)
        uniq.append(c)
    return uniq[:12]


def arm_prompts(c, TOOLS):
    """x395 의 팔 문면을 **그대로** 재구성 + PVI 기준선 두 개."""
    sim = c["sim"]
    calls_, ents = X.ledger_of(sim)
    ask = X.user_ask(sim)
    led = ("지금까지 호출한 도구: %s\n대화에 등장한 레코드 id: %s"
           % (", ".join(calls_[:25]) or "(없음)", ", ".join(ents[:25]) or "(없음)"))
    proc = "\n".join("- " + s for s in c["lines"])
    neg = (X.FILLER * max(1, len(proc) // len(X.FILLER) + 1))[:len(proc)]
    tools = "# 호출 가능한 도구(정책 문서가 정의한 것)\n" + ", ".join(TOOLS) + "\n\n"
    q = ("\n\n# 질문\n지금 시점에서 **다음에 호출할 도구 하나**를 정하라. "
         "JSON 하나로만 답하라: {\"tool\": \"<이름>\", \"arguments\": {…}}")
    base = tools + ("# 손님 요청\n%s\n\n# 지금까지의 진행(원장)\n%s\n\n" % (ask, led))
    out = {
        "NULL_bare": q.lstrip("\n"),
        "NULL_tools": tools + q.lstrip("\n"),
        "A_min": base + "# 정책 절차(축자)\n" + proc + q,
        "C_neg": base + "# 안내\n" + neg + q,
        "B_full": tools + "# 대화 전문\n" + X.convo(sim) + "\n\n# 정책 절차(축자)\n" + proc + q,
    }
    for nn in (4, 8, 16, 32):
        out["B_tail%d" % nn] = (tools + "# 대화(마지막 %d 메시지)\n" % nn + X.convo(sim, tail=nn)
                                + "\n\n# 정책 절차(축자)\n" + proc + q)
    # ★교락 제거 팔 (2026-08-19 딥리서치 실측): tail 창은 **종결부 비율**이 창 길이와 ρ=−1.000 으로
    #   완전 공선이다(tail4 0.891 → head60 0.141) — 그리고 마지막 user 메시지가 **23/23 종결 센티널**이다.
    #   ⇒ 지금까지의 "문맥이 길수록 좋다"는 길이가 아니라 **종결부 오염**일 수 있다.
    #     길이를 맞추고 종결부만 뺀 팔로 그 둘을 가른다.
    msgs = (sim.get("messages") or [])[:X.close_index(sim)] or (sim.get("messages") or [])[:1]

    def _render(sub, label):
        return (tools + "# 대화(%s)\n" % label + X.convo({"messages": sub})
                + "\n\n# 정책 절차(축자)\n" + proc + q)

    out["B_pre4"] = _render(msgs[:4], "처음 4 메시지")           # tail4 와 길이 동일·종결부 0
    mid = len(msgs) // 2
    out["B_mid32"] = _render(msgs[max(0, mid - 16):mid + 16] or msgs[:32], "중간 32 메시지")
    last_call = 0
    for i, m in enumerate(msgs):
        if m.get("tool_calls"):
            last_call = i
    out["B_prefix"] = _render(msgs[:last_call + 2] if last_call else msgs[:8], "마지막 호출 시점까지")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8141)
    ap.add_argument("--arms", default=("NULL_bare,NULL_tools,C_neg,B_tail4,B_tail8,B_tail16,B_tail32,"
                                       "B_full,A_min,B_pre4,B_mid32,B_prefix"))
    ap.add_argument("--base", default="NULL_tools")
    a = ap.parse_args()

    docs = X.load_docs()
    TOOLS = X.tool_universe(docs)
    cases = build_cases(docs)
    arms = a.arms.split(",")
    print("표적 %d · 팔 %d · 포트 %d · 모델 %s" % (len(cases), len(arms), a.port, MODEL))
    print("PVI 기준선 = %s · y = 그 자리 옳은 호출의 JSON 앞머리\n" % a.base)

    rows = []
    for c in cases:
        P = arm_prompts(c, TOOLS)
        # 래퍼가 정답인 표적(040·055)은 **래퍼 이름도 정답**이다 — 변형 중 최대 로그가능도를 쓴다.
        y_variants = ['{"tool": "%s", "arguments": {' % n for n in (c.get("accept") or [c["tool"]])]
        # ★기권 문자열도 같이 잰다 — 우리 SYS 가 {"tool": null} 기권을 **명시적으로 허가**했고
        #   `said_only` 가 그것을 셌다. "산문 이탈"인지 "기권"인지는 이 값이 가른다.
        y_ab = '{"tool": null'
        per = {}
        for arm in arms:
            pre = TPL % (X.SYS, P[arm])
            try:
                cand = [score_suffix(a.port, pre, yv) for yv in y_variants]
                lp, n = max(cand, key=lambda z: z[0])
                lpa, _na = score_suffix(a.port, pre, y_ab)
                dist = first_token_dist(a.port, pre)
                j, t, o = channel_mass(dist)
            except Exception as e:
                print("  ERROR %s %s: %r" % (c["task"], arm, e))
                continue
            per[arm] = {"logp": lp, "ntok": n, "logp_abstain": lpa,
                        "json_mass": j, "text_mass": t, "other": o}
        if a.base not in per:
            continue
        b = per[a.base]["logp"]
        for arm in per:
            per[arm]["pvi_bits"] = (per[arm]["logp"] - b) / math.log(2)
        rows.append({"task": c["task"], "tool": c["tool"], "arms": per})
        print("  %-9s %-40s " % (c["task"], c["tool"][:40])
              + " ".join("%s=%+.1f" % (k.replace("B_tail", "t").replace("NULL_", "0"),
                                       per[k]["pvi_bits"]) for k in arms if k in per))

    print("\n## 팔별 요약 (표적 %d개 평균)" % len(rows))
    print("%-11s %10s %10s %10s %10s %12s" % ("arm", "PVI(bits)", "logp(y)", "P({)", "P(prose)", "정답−기권"))
    for arm in arms:
        v = [r["arms"][arm] for r in rows if arm in r["arms"]]
        if not v:
            continue
        print("%-11s %10.2f %10.1f %10.3f %10.3f %12.2f"
              % (arm, sum(x["pvi_bits"] for x in v) / len(v),
                 sum(x["logp"] for x in v) / len(v),
                 sum(x["json_mass"] for x in v) / len(v),
                 sum(x["text_mass"] for x in v) / len(v),
                 sum((x["logp"] - x["logp_abstain"]) / math.log(2) for x in v) / len(v)))

    print("\n## 표적-대응 대비 (PVI 차·bits)")
    for A, B in (("A_min", "B_full"), ("A_min", "B_tail4"), ("A_min", "C_neg"),
                 ("B_full", "B_tail4"), ("C_neg", "NULL_tools"),
                 ("B_pre4", "B_tail4"), ("B_mid32", "B_tail32"), ("B_prefix", "B_full")):
        d = [r["arms"][A]["pvi_bits"] - r["arms"][B]["pvi_bits"]
             for r in rows if A in r["arms"] and B in r["arms"]]
        if not d:
            continue
        m = sum(d) / len(d)
        sd = math.sqrt(sum((x - m) ** 2 for x in d) / max(len(d) - 1, 1))
        se = sd / math.sqrt(len(d)) if d else 0
        win = sum(1 for x in d if x > 0)
        print("  %-8s − %-10s = %+6.2f bits (SD %.2f · SE %.2f · t=%.2f · 승 %d/%d)"
              % (A, B, m, sd, se, (m / se if se else float("nan")), win, len(d)))

    out = os.path.normpath(os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026",
                                        "x397_pvi_channel.json"))
    io.open(out, "w", encoding="utf-8").write(json.dumps(rows, ensure_ascii=False, indent=1))
    print("\n원자료: %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
