#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""load_graded_probe.py — Phase-L1: load 차원을 통제 증감한 *격리* probe (gpt-4.1 0·로컬).

목적: operand 고정·한 load 차원만 증가 → failure-onset → 차원별 L*(N) scale-response.
     관측(load_obs)의 *상관*을 통제생성으로 *인과* 확정. 멀티스케일(7B/14B/32B)서 차원 은퇴 순서 검증.
★[[05]]: 합성은 ABox(db 변형)만·도메인분기 0. operand 고정(정답 trivial·부하만 변동).

차원 (구현):
  L_interf — N개 유사 변형(같은 product·다른 option) 중 지정 option 매칭. N 증감 = 간섭만.
  (확장 예정: L_state·L_branch·L_len)

사용: python load_graded_probe.py --dim interf --levels 1,2,4,8,16 --k 30 --agent_base ... --agent_model ...
"""
import argparse, json, urllib.request, re

RETAIL = "/home/woori/scratch/tau2-bench/data/tau2/domains/retail"
db = json.load(open(RETAIL + "/db.json"))
prods = db["products"]


def ask(prompt, model, base, mx=30):
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}],
                       "temperature": 0.0, "max_tokens": mx}).encode()
    req = urllib.request.Request(base.rstrip("/") + "/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=120).read())["choices"][0]["message"]["content"]


def oid(i):
    return f"#W{1000000 + i}"


def interf_instances(level, k):
    """L_interf: 같은 product의 N=level 변형을 N개 주문으로·지정 option의 주문 찾기.
    operand 고정(정확 매칭 trivial)·부하=N개 유사항목 스캔/간섭. deterministic(seed 무관)."""
    insts = []
    # 변형 >= level 인 product만 (유사 간섭 보장)
    cand = [(pid, p) for pid, p in prods.items() if len(p.get("variants") or {}) >= level]
    cand.sort(key=lambda x: x[0])
    ci = 0
    while len(insts) < k and cand:
        pid, p = cand[ci % len(cand)]
        vs = list((p.get("variants") or {}).items())
        vs.sort(key=lambda x: x[0])
        n = level
        # 회전으로 다른 변형부분집합·다른 target 선택 (k개 다양화)
        start = (ci // len(cand)) % max(1, len(vs) - n + 1)
        chosen = vs[start:start + n] if len(vs) >= start + n else vs[:n]
        if len(chosen) < n:
            ci += 1
            continue
        tgt_idx = ci % n
        tgt_item, tgt_v = chosen[tgt_idx]
        # 주문 배치 (target 위치도 회전)
        lines = []
        ans = None
        for j, (iid, v) in enumerate(chosen):
            o = oid(len(insts) * 100 + j)
            lines.append(f"  {o}: {p['name']} options={v['options']}")
            if iid == tgt_item:
                ans = o
        prompt = (f"Customer's orders (each contains one {p['name']}):\n" + "\n".join(lines) +
                  f"\n\nCustomer wants the order whose {p['name']} has EXACTLY these options: {tgt_v['options']}.\n"
                  f"Output ONLY that order_id.")
        insts.append((prompt, ans))
        ci += 1
    return insts


GEN = {"interf": interf_instances}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", default="interf", choices=list(GEN))
    ap.add_argument("--levels", default="1,2,4,8,16")
    ap.add_argument("--k", type=int, default=30)
    ap.add_argument("--agent_base", default="http://localhost:8360/v1")
    ap.add_argument("--agent_model", default="Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8")
    a = ap.parse_args()
    levels = [int(x) for x in a.levels.split(",")]
    print(f"=== load-graded probe · dim={a.dim} · model={a.agent_model} · k={a.k} · gpt-4.1 0 ===")
    print(f"  (operand 고정·부하만 증감 → accuracy vs load-level = onset)\n")
    print(f"  {'level':>6} {'acc':>6} {'n':>4}   wrong-examples")
    curve = []
    for lv in levels:
        insts = GEN[a.dim](lv, a.k)
        if not insts:
            print(f"  {lv:>6} {'SKIP':>6} {0:>4}   (생성 0건·이 차원/level 미지원=결과 아님·[[08]])")
            continue
        ok = 0
        wrong = []
        for prompt, ans in insts:
            try:
                out = ask(prompt, a.agent_model, a.agent_base)
            except Exception as e:
                wrong.append(f"ERR:{type(e).__name__}")
                continue
            hit = ans is not None and ans.lstrip("#") in out.replace("#", "")
            ok += hit
            if not hit and len(wrong) < 2:
                wrong.append(f"want {ans} got '{out.strip()[:30]}'")
        acc = ok / max(len(insts), 1)
        curve.append((lv, acc, len(insts)))
        print(f"  {lv:>6} {acc:>6.2f} {len(insts):>4}   {wrong}")
    print(f"\n  curve(level→acc): {[(l, round(a,2)) for l,a,_ in curve]}")
    # onset = acc가 0.5 밑으로 떨어지는 첫 level
    onset = next((l for l, ac, _ in curve if ac < 0.5), None)
    print(f"  L*(onset<0.5) = {onset}  (작을수록 그 scale의 부하내성 낮음)")


if __name__ == "__main__":
    main()
