# -*- coding: utf-8 -*-
"""x163 — 절벽이 **어떤 모양인가**: j\* 이분탐색 + 엔트로피 궤적 + 온도 스윕 (유료 0·한 덤프).

## 가르려는 것 (세 읽기·초안 §6.0/§6.1·C346)

  ⑴ **차등 누적** — log-odds 는 λ-가중 차등을 따른다(P5)  ⇒ 엔트로피 **단조 감소**
  ⑵ **종단 λ 가중** — 마지막 자리가 구조적으로 무겁다      ⇒ 앞은 평탄, **끝에서만** 감소
  ⑶ **검색 상전이** — 혼합이 임계에서 붕괴(스핀글라스)      ⇒ 임계까지 **유지** 후 **급락**
                                                              + 임계 근방 **요동 증가**
                                                              + **j\* 가 온도(β)를 따라 이동**

⑴⑵는 온도로 j\* 가 체계적으로 움직일 이유가 없다 — **온도 스윕이 ⑶의 강한 검정**이다.

## 왜 추가 호출이 0 인가

x157 이 이미 `guided_choice` 로 첫 토큰 top-20 logprob 을 뜬다. 온도 스케일은 같은 로짓의
재정규화(`q ∝ p^(1/T)`)이므로 **덤프 하나로 전 온도**를 얻는다. 새 추론 없음.

## 계기 각주 (x159 가 남긴 것·정직하게 표시한다)

 · top-20 < 후보 수 ⇒ **꼬리 검열**. 관측 질량(1−잔여)을 매 행에 함께 낸다 — 잔여가 크면
   그 행의 엔트로피는 **하한**이다.
 · 후보 여럿이 첫 낱말을 공유한다(`Business` 계열) ⇒ 엔트로피는 **첫-낱말 버킷** 위의 값이다.
   버킷 수를 함께 출력해 이 사실을 숨기지 않는다.
 · 1글자 토큰 오귀속은 x157 `lp_of` 가 이미 막는다(2자 이상·접두 일치).

실행: T2_PROBE_URL=... T2_PROBE_MODEL=... py -3 x163_cliff_shape.py [JLIST]
      예) x163_cliff_shape.py 0,9,18,22,23,24,25,26,27
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import x149_choice_isolation as X                              # noqa: E402
import x150_choice_ablation as Y                               # noqa: E402
import x157_entrainment_lambda as P                            # noqa: E402
import t2_factdag as FD                                        # noqa: E402
import t2_ledger as LG                                         # noqa: E402
from gate_interpreter import load_domain_a2                     # noqa: E402

TEMPS = (0.5, 0.7, 1.0, 1.4, 2.0)       # T<1 = 날카롭게(β↑) · T>1 = 뭉갬(β↓)


def buckets(dist, choices):
    """첫-낱말 버킷 위의 확률(관측분) — 후보 이름의 첫 낱말로 모은다."""
    heads = sorted({c.split()[0] for c in choices if c.split()}, key=len, reverse=True)
    out = {}
    for h in heads:
        p = math.exp(P.lp_of(dist, h))
        if p > 1e-12:
            out[h] = p
    return out


def entropy(ps):
    tot = sum(ps) or 1.0
    return -sum((p / tot) * math.log(p / tot) for p in ps if p > 0)


def rescale(bk, T):
    """같은 로짓의 온도 재정규화 — q ∝ p^(1/T). 추가 추론 없음."""
    q = {k: v ** (1.0 / T) for k, v in bk.items()}
    s = sum(q.values()) or 1.0
    return {k: v / s for k, v in q.items()}


def main():
    js = [int(x) for x in (sys.argv[1] if len(sys.argv) > 1
                           else "0,9,18,22,23,24,25,26,27").split(",")]
    a2 = load_domain_a2("banking_knowledge")
    spec = next(s for s in a2["ledger_metrics"] if s.get("eligible_text"))
    rows = a2["policy_ontology"]["rows"]
    maps = {ax: FD._a3_map(rows, {"axis": ax}) for ax in spec["eligible"]["show_axes"]}
    table = LG.eligible_text(730, {}, maps, spec, {"qualifying_deposit_usd": 30000}).strip()
    base = table + "\n\n" + X.FACTS[P.TASK] + "\n\n" + X.QUESTION
    choices = [l.strip().split(":")[0].strip() for l in table.splitlines() if l.startswith("  ")]
    msgs = Y.msgs_of(os.environ.get("T2_PROBE_TAG", "bank_elig_20260809i"), P.TASK)

    print("model=%s" % P.MODEL)
    print("궤적 %d 메시지 · 후보 %d · 첫-낱말 버킷 %d · gold=%s · 앵커=%s"
          % (len(msgs), len(choices), len({c.split()[0] for c in choices}),
             P.GOLD_HEAD, P.ANCHOR))
    print("\n%-5s %9s %9s %8s %7s %6s  %s"
          % ("j", "logP(g)", "logP(a)", "H(nats)", "관측질량", "버킷", "argmax"))

    rowdata = []
    for j in js:
        pre = ("Here is a customer-service conversation so far.\n\n"
               + Y.render(msgs[:j]) + "\n\n") if j else ""
        d = P.first_token_dist(pre + base, choices)
        bk = buckets(d, choices)
        obs = sum(bk.values())
        g, a = P.lp_of(d, P.GOLD_HEAD), P.lp_of(d, P.ANCHOR)
        top = max(bk, key=bk.get) if bk else "?"
        print("%-5d %9.3f %9.3f %8.3f %7.3f %6d  %s%s"
              % (j, g, a, entropy(list(bk.values())), obs, len(bk), top,
                 "" if top == P.GOLD_HEAD else "   ← 붕괴"))
        rowdata.append((j, bk))

    # ── j* = argmax 가 gold 를 떠나는 최초 j (온도별) ────────────────────────
    print("\n=== 온도 스윕 (같은 덤프 재정규화·추가 호출 0) ===")
    print("%-6s %6s  %s" % ("T", "j*", "j별 argmax(gold=✓)"))
    for T in TEMPS:
        marks, jstar = [], None
        for j, bk in rowdata:
            if not bk:
                marks.append("?")
                continue
            q = rescale(bk, T)
            top = max(q, key=q.get)
            ok = (top == P.GOLD_HEAD)
            marks.append("✓" if ok else "x")
            if not ok and jstar is None:
                jstar = j
        print("%-6.2f %6s  %s" % (T, jstar if jstar is not None else "없음", " ".join(marks)))
    print("\n  ⑶ 상전이면 j* 가 T 를 따라 **단조 이동**한다(T↓ 일찍·T↑ 늦게).")
    print("  ⑴⑵면 j* 는 T 에 **무반응**이어야 한다 — 그것이 이 표의 판정이다.")

    print("\n=== 엔트로피 궤적 (T=1.0) ===")
    for j, bk in rowdata:
        h = entropy(list(bk.values()))
        print("  j=%-3d H=%6.3f  %s" % (j, h, "#" * int(round(h * 20))))
    print("  ⑴ 단조 감소 / ⑵ 끝에서만 감소 / ⑶ 유지 후 급락+요동 — 경로가 판정이다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
