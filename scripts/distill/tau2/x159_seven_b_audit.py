# -*- coding: utf-8 -*-
"""x159 — x157 스케일 사다리의 **계기 감사** + 절벽 위치-내용 통제 + 되꽂기 framing 귀속.

## 왜 (2026-08-09 리뷰에서 나온 3건 · [[55]] 순서: 배관→문구→계기→모델)

  ⑴ 7B j=0 의 p≈0.025 는 균등(1/25=0.04)보다 **낮다** — "우연 수준"이 아니라 체계적 쏠림.
     얕은 휴리스틱(능력 결론)인지 첫-토큰 prior 교락(계기 결함)인지 argmax·자유생성 교차로 가른다.
  ⑵ 7B j=27 의 −27.631 = log(1e-12) = **센서 바닥값**(top_logprobs=20 < 후보 25 ⇒ 설계상 검열).
     각 측정점의 검열량(잔여 질량·top-20 밖 후보 수)을 표기한다.
  ⑶ lp_of 의 접두 합산이 **첫 낱말을 공유하거나 접두 토큰을 공유하는 후보**를 합쳐 셀 수 있다.
     후보 쌍의 공통 접두를 전수 점검한다.

## 덧붙인 두 통제 (같은 자극·무료)

  ⑷ 절벽 위치-내용 통제(32B·8140): P1 반증의 "위치 가중" 해석은 교락 상태다 — x156 이
     내용 귀속(assistant:numbers)을 시사했고 그 발화는 궤적 끝에 몰린다. 마지막 5 메시지를
     [단독 / 앞자리 / 제자리]로 옮겨 절벽이 내용을 따라가는지 위치를 따라가는지 가른다.
  ⑸ 되꽂기 framing 귀속(32B·8140): §7.3 부정통제 ⒞ 계열 — 되꽂기의 유효 성분이
     *언급*(mention)인지 *선택 선언*(authoritative selection)인지. x151/x154 가 이미
     "정보 추가 = 0"을 보였으므로, 남은 축은 같은 이름을 [bare mention / 선언] 두 형태로 얹는 것.

실행: py -3 x159_seven_b_audit.py [KMAX=6]
  서버 고정: 8140=32B(GPTQ-Int8) · 8141=7B — 모델명은 /v1/models 로 자가 확인(하드코딩 금지).
"""
import json
import math
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

URL_32B = "http://localhost:8140/v1"
URL_7B = "http://localhost:8141/v1"
TASK = "task_099"
GOLD_HEAD = "World"
ANCHOR = "Navy"
TAG = os.environ.get("T2_PROBE_TAG", "bank_elig_20260809i")
FLOOR = math.log(1e-12)


def model_of(base_url):
    with urllib.request.urlopen(base_url + "/models", timeout=30) as r:
        return json.load(r)["data"][0]["id"]


def chat(base_url, model, prompt, temp=0.0, max_tokens=40, **extra):
    body = {"model": model, "temperature": temp, "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}]}
    body.update(extra)
    req = urllib.request.Request(base_url + "/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)["choices"][0]


def first_token_dist(base_url, model, prompt, choices, top=20):
    c = chat(base_url, model, prompt, max_tokens=1, logprobs=True,
             top_logprobs=top, guided_choice=list(choices))
    lp = c["logprobs"]["content"][0]["top_logprobs"]
    return {e["token"].strip(): e["logprob"] for e in lp}


def head_probs(dist, heads):
    """각 head 의 확률(접두 토큰 합산·x157 lp_of 와 같은 규칙) + 검열 진단.

    반환: ({head: p}, covered_mass) — covered = top-20 토큰의 확률합(1−covered = 검열 질량).
    """
    ps = {}
    for h in heads:
        p = 0.0
        for k, v in dist.items():
            kk = k.lower()
            if len(kk) >= 2 and h.lower().startswith(kk):
                p += math.exp(v)
        ps[h] = p
    covered = sum(math.exp(v) for v in dist.values())
    return ps, covered


def render_msgs(ms):
    return "Here is a customer-service conversation so far.\n\n" + Y.render(ms) + "\n\n"


def main():
    kmax = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    a2 = load_domain_a2("banking_knowledge")
    spec = next(s for s in a2["ledger_metrics"] if s.get("eligible_text"))
    rows = a2["policy_ontology"]["rows"]
    maps = {ax: FD._a3_map(rows, {"axis": ax}) for ax in spec["eligible"]["show_axes"]}
    table = LG.eligible_text(730, {}, maps, spec, {"qualifying_deposit_usd": 30000}).strip()
    base = table + "\n\n" + X.FACTS[TASK] + "\n\n" + X.QUESTION
    CHOICES = [l.strip().split(":")[0].strip() for l in table.splitlines() if l.startswith("  ")]
    heads = []
    for c in CHOICES:
        h = c.split()[0]
        if h not in heads:
            heads.append(h)
    MSGS = Y.msgs_of(TAG, TASK)
    uni = math.log(1.0 / len(CHOICES))
    m32 = model_of(URL_32B)
    m7 = model_of(URL_7B)
    print("32B=%s  7B=%s" % (m32, m7))
    print("후보 %d · 고유 첫낱말 %d · 균등 lnP=%.3f · MSGS=%d" %
          (len(CHOICES), len(heads), uni, len(MSGS)))

    # ── ⑶ 후보 충돌 전수 점검 (오프라인) ────────────────────────────────────
    print("\n[⑶ 충돌] 첫낱말 공유 후보:")
    for h in heads:
        share = [c for c in CHOICES if c.split()[0] == h]
        if len(share) > 1:
            print("   %-10s ← %s" % (h, share))
    print("[⑶ 충돌] 접두(≥2자) 공유 head 쌍:")
    pairs = [(a, b) for i, a in enumerate(heads) for b in heads[i + 1:]
             if a.lower()[:2] == b.lower()[:2]]
    print("   %s" % (pairs if pairs else "없음"))

    def with_traj(msgs, tail=""):
        pre = render_msgs(msgs) if msgs else ""
        return pre + base + (("\n\n" + tail) if tail else "")

    def probe(base_url, model, msgs, tail="", label=""):
        d = first_token_dist(base_url, model, with_traj(msgs, tail), CHOICES)
        ps, covered = head_probs(d, heads)
        rank = sorted(ps, key=ps.get, reverse=True)
        g = ps.get(GOLD_HEAD, 0.0)
        print("  %-26s lnP(gold)=%8.3f  gold순위=%2d/%d  argmax=%-10s(p=%.3f)  "
              "top20질량=%.3f%s"
              % (label, math.log(max(g, 1e-12)), rank.index(GOLD_HEAD) + 1, len(heads),
                 rank[0], ps[rank[0]], covered,
                 "  ⚠검열" if covered < 0.98 else ""))
        return ps, covered

    # ── ⑴⑵ 7B j-사다리: argmax·gold 순위·검열 표기 ──────────────────────────
    print("\n[⑴⑵ 7B 사다리 · 8141] (lnP=−27.631 은 바닥값)")
    grid = sorted({0} | {round(len(MSGS) * i / kmax) for i in range(1, kmax + 1)})
    for j in grid:
        probe(URL_7B, m7, MSGS[:j], label="j=%d" % j)

    # ── ⑴ 7B 자유생성 교차 검증 (x149 A_minimal 동형·프록시 검증) ───────────
    print("\n[⑴ 7B 자유생성 j=0 ×5] (guided 없음 · x149 와 같은 채점)")
    for i in range(5):
        a = chat(URL_7B, m7, base, temp=0.0 if i == 0 else 0.7)["message"]["content"]
        a = " ".join(str(a).split())
        print("  %s  %s" % ("HIT " if X.score(a, X.GOLD[TASK]) else "miss", a[:80]))

    # ── ⑷ 절벽 위치-내용 통제 (32B · 8140) ──────────────────────────────────
    last5 = MSGS[-5:]
    first22 = MSGS[:-5]
    print("\n[⑷ 32B 절벽 통제 · 8140] 절벽이 내용을 따라가면 위치 무관하게 낮아야 한다")
    probe(URL_32B, m32, [], label="j=0 (기준)")
    probe(URL_32B, m32, first22, label="앞22 단독 (재측정)")
    probe(URL_32B, m32, last5, label="뒤5 단독")
    probe(URL_32B, m32, last5 + first22, label="뒤5 를 앞으로")
    probe(URL_32B, m32, MSGS, label="전체 27 (재측정)")

    # ── ⑸ 되꽂기 framing 귀속 (32B · j=27) ─────────────────────────────────
    hand = ("A separate analysis, working only from the policy constants on record and this "
            "customer's stated situation, selects: %s Blue." % GOLD_HEAD)
    mention = "For reference, one of the accounts on record is %s Blue." % GOLD_HEAD
    print("\n[⑸ 32B 되꽂기 framing · j=27] mention 이 못 올리고 선언만 올리면 유효 성분=선언")
    probe(URL_32B, m32, MSGS, label="(없음)")
    probe(URL_32B, m32, MSGS, tail=mention, label="bare mention")
    probe(URL_32B, m32, MSGS, tail=hand, label="선언(되꽂기)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
