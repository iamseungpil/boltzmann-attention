# -*- coding: utf-8 -*-
r"""x424 — **formalize(LLM) → calc(엔진)** 이 수치 인자를 사나

## 왜 (사용자 지시 2026-08-20 · 등대 §1.2 F2b)
`PERTASK` §2: 093·094·074·072 의 실패는 **문서에 없는 수(계산으로만 나오는 값)** 다 — 094 는 gold 값이
궤적에 **20/20 미도착**. 등대 §1.4 는 이 칸을 *"형식화(LLM)→결정론 실행(argmin/filter)"* [P] 로 적어 뒀지만
**이 벤치에서 짝 비교로 잰 적이 없다**. 여기서 잰다.

## 팔 (셋 다 **같은 재료** · 다른 것은 누가 산술을 하느냐뿐)
    A_direct     재료 + "그 값 하나를 내라"                        ← 모델이 끝까지 계산
    B_formalize  재료 + "**계산하지 말고** 식과 피연산자를 내라"    ← 산술은 엔진이 한다
    C_neg        재료에서 **수치 표를 뺀 것** + A 와 같은 질문      ← 부정통제 [[57]]

★엔진은 **산술만** 한다 — `+ - * / ( )` 와 숫자뿐인 AST 평가기다. 이름·규칙·값 선택을 엔진이 하면
  그것은 gold 프로그램 재작성이다([[62]]). 그래서 피연산자도 **모델이 준 것만** 쓴다.
★재료는 그 sim 이 실제로 받은 도구-결과 축자다([[18]] 정보-맞춘 격리). gold 는 채점에만 쓴다([[23]]).

## 사전 고정 해석 (결과 보기 전)
- `B_formalize > A_direct` (짝 비교)        ⇒ 산술 이관이 산다 ⇒ F2b 는 결정론 실행이 답이다(등대 [P]→[M]).
- `B ≈ A` 이고 둘 다 `> C_neg`              ⇒ 병목은 산술이 아니라 **어느 수를 넣을지**(선택) — x423 축과 같은 물음.
- `B ≈ A ≈ C_neg`                           ⇒ 재료가 부족하다 ⇒ 전달·수집 축(073 형)으로 돌린다.
- B 의 **식은 맞는데 피연산자가 틀린** 비율이 높으면 ⇒ 잔여는 계산이 아니라 ⋈ 참조매칭이다.

사용: py -3 x424_formalize_calc.py [--n 3] [--port 8141] [--max-cases 24]
"""
import argparse
import ast
import collections
import io
import json
import os
import sys
import threading

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import t2_forensic as F  # noqa: E402
import x395_compliance_iso as X  # noqa: E402
import x397_pvi_channel as P  # noqa: E402
import x423_choice_isolation as I  # noqa: E402

MAT = 5200            # 재료 최대 길이
TAIL_TOOL = 14        # 재료로 쓸 도구-결과 개수(뒤에서부터)

SYS_A = ("You are the argument-filling module of a Rho-Bank support agent. "
         "Reply with ONE JSON object only: {\"value\": <number>}. No prose.")
SYS_B = ("You are the FORMALIZER of a Rho-Bank support agent. You do NOT compute. "
         "Reply with ONE JSON object only: "
         "{\"expr\": \"<arithmetic expression using only numbers, + - * / ( )>\"}. "
         "Put the actual numbers you read from the materials into the expression. No prose.")

SAFE = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Add, ast.Sub, ast.Mult, ast.Div,
        ast.USub, ast.UAdd, ast.Constant, ast.Load)


def calc(expr):
    """엔진의 몫 — **산술만**. 이름·호출·첨자는 전부 거절한다(판단 0)."""
    try:
        tree = ast.parse(str(expr).strip(), mode="eval")
    except Exception:
        return None
    for node in ast.walk(tree):
        if not isinstance(node, SAFE):
            return None
        if isinstance(node, ast.Constant) and not isinstance(node.value, (int, float)):
            return None
    try:
        return float(eval(compile(tree, "<calc>", "eval"), {"__builtins__": {}}, {}))
    except Exception:
        return None


def numeric_cases(maxn):
    """WRONGARG 의 **수치 필드**마다 사례 하나."""
    out, per = [], collections.Counter()
    for tag in I.RUNS:
        for sim in F.sims(tag, I.SUF):
            if ((sim.get("reward_info") or {}).get("reward") or 0) >= 1.0:
                continue
            d = F.mutation_diff(sim)
            gold_by = {}
            for g in d["gold"]:
                gold_by.setdefault(g["name"], []).append(g)
            for w in d["wrongarg"]:
                gs = gold_by.get(w["name"]) or []
                if not gs:
                    continue
                g = gs[0]
                for k in sorted(set(g["args"]) | set(w["args"])):
                    gv, wv = g["args"].get(k), w["args"].get(k)
                    if gv is None or str(gv) == str(wv) or I.is_choice(gv):
                        continue
                    key = (F.task_id(sim), w["name"], k)
                    if per[key] >= 2:
                        continue
                    per[key] += 1
                    out.append({"task": F.task_id(sim), "trial": sim.get("trial"), "tool": w["name"],
                                "arg": k, "gold": float(gv), "live": wv, "msg_i": w["msg_i"],
                                "sim": sim})
                    if len(out) >= maxn:
                        return out
    return out


def materials(c, with_numbers=True):
    """그 결정점 **이전에 도착한 도구-결과**를 축자로. C_neg 는 숫자를 지운다."""
    parts = []
    for i, m in enumerate(c["sim"].get("messages") or []):
        if i >= c["msg_i"] or m.get("role") != "tool":
            continue
        parts.append(" ".join(str(m.get("content") or "").split()))
    parts = parts[-TAIL_TOOL:]
    txt = "\n\n".join(parts)[:MAT]
    if not with_numbers:
        txt = "".join(("#" if ch.isdigit() else ch) for ch in txt)
    return txt


def parse_obj(raw, key):
    t = str(raw or "")
    i, j = t.find("{"), t.rfind("}")
    if i >= 0 and j > i:
        try:
            o = json.loads(t[i:j + 1])
            if isinstance(o, dict) and key in o:
                return o[key]
        except Exception:
            pass
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--port", type=int, default=8141)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--max-cases", type=int, default=24)
    a = ap.parse_args()

    cs = numeric_cases(a.max_cases)
    print("=" * 100)
    print("x424 · formalize→calc · 사례 %d (고유 task/arg %d) · 팔 3 · n=%d · 포트 %d"
          % (len(cs), len(set((c["task"], c["arg"]) for c in cs)), a.n, a.port))
    print("=" * 100)

    jobs = []
    for c in cs:
        ask = X.user_ask(c["sim"])
        head = ("# 손님 요청\n%s\n\n# 채워야 하는 것\n도구 `%s` 의 인자 `%s`.\n" % (ask, c["tool"], c["arg"]))
        matn = "\n# 재료(축자·도구 응답)\n%s\n" % materials(c, True)
        matx = "\n# 재료(축자·도구 응답)\n%s\n" % materials(c, False)
        qa = "\n# 질문\n그 인자에 넣을 수 하나를 내라. JSON: {\"value\": <수>}\n"
        qb = ("\n# 질문\n계산하지 마라. 그 수를 만드는 **산술식**을 내라 — 숫자와 + - * / ( ) 만.\n"
              "JSON: {\"expr\": \"<식>\"}\n")
        evid = I.evidence_block(c["sim"], c["msg_i"])
        for arm, sysmsg, body in (("A_direct", SYS_A, head + matn + qa),
                                  ("B_formalize", SYS_B, head + matn + qb),
                                  ("C_neg", SYS_A, head + matx + qa),
                                  ("E_direct_full", SYS_A, head + evid + qa),
                                  ("F_formalize_full", SYS_B, head + evid + qb)):
            for k in range(a.n):
                jobs.append({"c": c, "arm": arm, "k": k, "sys": sysmsg, "body": body,
                             "temp": (0.0 if k == 0 else a.temp)})
    print("작업 %d건" % len(jobs))

    lock, out = threading.Lock(), []

    def work(_i):
        while True:
            with lock:
                if not jobs:
                    return
                j = jobs.pop(0)
            try:
                d = P.post(a.port, "/v1/chat/completions",
                           {"model": X.MODEL,
                            "messages": [{"role": "system", "content": j["sys"]},
                                         {"role": "user", "content": j["body"]}],
                            "temperature": j["temp"], "max_tokens": 200})
                raw = d["choices"][0]["message"]["content"]
            except Exception as e:
                raw = "ERROR " + str(e)[:160]
            c = j["c"]
            if j["arm"] == "B_formalize":
                expr = parse_obj(raw, "expr")
                val = calc(expr)
                got, ok = val, (val is not None and abs(val - c["gold"]) < 1e-6)
                extra = {"expr": None if expr is None else str(expr)[:120]}
            else:
                v = parse_obj(raw, "value")
                try:
                    got = None if v is None else float(v)
                except Exception:
                    got = None
                ok = got is not None and abs(got - c["gold"]) < 1e-6
                extra = {"expr": None}
            rec = {"task": c["task"], "trial": c["trial"], "tool": c["tool"], "arg": c["arg"],
                   "arm": j["arm"], "k": j["k"], "gold": c["gold"], "live": c["live"],
                   "got": got, "hit": bool(ok), "parsed": got is not None, "raw": raw[:160]}
            rec.update(extra)
            with lock:
                out.append(rec)
                if len(out) % 30 == 0:
                    print("  ... %d/%d" % (len(out), len(out) + len(jobs)))

    ths = [threading.Thread(target=work, args=(i,)) for i in range(a.workers)]
    [t.start() for t in ths]
    [t.join() for t in ths]

    print("\n## 팔별 (gold 수치 일치 / 파싱)")
    print("%-12s %6s %8s %8s" % ("arm", "n", "HIT", "PARSED"))
    for arm in ("C_neg", "A_direct", "B_formalize", "E_direct_full", "F_formalize_full"):
        r = [x for x in out if x["arm"] == arm]
        if not r:
            continue
        n = float(len(r))
        print("%-12s %6d %8.3f %8.3f"
              % (arm, len(r), sum(x["hit"] for x in r) / n, sum(x["parsed"] for x in r) / n))

    print("\n## 짝 비교 (같은 사례·같은 k 에서 B 대 A)")
    key = lambda x: (x["task"], x["trial"], x["arg"], x["k"])
    for an, bn in (("A_direct", "B_formalize"), ("E_direct_full", "F_formalize_full")):
        A = {key(x): x for x in out if x["arm"] == an}
        B = {key(x): x for x in out if x["arm"] == bn}
        both = sorted(set(A) & set(B))
        if not both:
            continue
        up = sum(1 for k in both if B[k]["hit"] and not A[k]["hit"])
        dn = sum(1 for k in both if A[k]["hit"] and not B[k]["hit"])
        print("   %-16s vs %-16s 짝 %3d · 형식화만 성공 %d · 직답만 성공 %d · 둘다 %d"
              % (an, bn, len(both), up, dn, sum(1 for k in both if A[k]["hit"] and B[k]["hit"])))

    print("\n## 태스크별 (B_formalize)")
    bt = collections.defaultdict(list)
    for x in out:
        if x["arm"] == "F_formalize_full":
            bt[(x["task"], x["arg"])].append(x)
    for k in sorted(bt):
        v = bt[k]
        print("   %-9s %-22s %d/%d   식 예: %s"
              % (k[0], k[1][:22], sum(x["hit"] for x in v), len(v),
                 (v[0].get("expr") or "-")[:60]))

    p = os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026", "x424_formalize_calc.json")
    with io.open(os.path.abspath(p), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("\n→ %s" % os.path.abspath(p))
    return 0


if __name__ == "__main__":
    sys.exit(main())
