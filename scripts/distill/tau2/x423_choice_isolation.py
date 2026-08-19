# -*- coding: utf-8 -*-
r"""x423 — **선택 오류는 능력인가 부하인가**: 스키마·정책만 세운 격리

## 왜 (사용자 지시 2026-08-20 · [[62]] ①격리로 결손 측정 · [[18]] 정보-맞춘 격리)
`PERSTEP_CAUSE_2026_08_20` §1b 실측: 우리가 쓴 틀린 값 **155/177 이 앞선 도구-결과에 축자로 있던 값**이고
gold 값도 **124/177 은 그 호출 이전에 도착**해 있었다. 즉 발명이 아니라 **오선택**이다.
남는 물음 하나: 그 오선택이 **능력**인가, 아니면 긴 궤적이 지우는 **부하**인가.
그건 격리로만 갈린다 — 정보를 빼지 않고 **부하만** 덜어 낸다.

## 팔 (넷 다 같은 물음 · 재료만 다르다)
    C_neg     손님 요청 + 도구·인자 이름만                      ← 부정통제 [[57]]
    A_schema  C_neg + **그 도구의 스키마 축자**(궤적에 도착한 `Tool unlocked:`/문서 블록)
    B_policy  A_schema + **그 도구를 언급한 정책 문서 축자**(궤적에 이미 도착한 것만)
    D_live    C_neg + **실제 궤적 꼬리**(라이브 부하 재현)

★재료는 전부 **그 sim 이 실제로 받은 도구-결과에서 축자로 잘라 온다**. 새 정보를 넣지 않는다
  ([[18]]). gold 는 **채점에만** 쓴다 — 프롬프트에 안 들어간다([[23]]).
★엔진 파싱 0 — 문서에서 값을 뽑지 않고 **슬라이스를 통째로** 준다([[59]]).

## 사전 고정 해석 (결과 보기 전에 못 박는다)
- `B_policy ≈ C_neg` (둘 다 낮음)      ⇒ 스키마·정책을 코앞에 줘도 못 고른다 = **능력/경계**.
                                          그 자리에만 결정론 후보([[62]] ③).
- `B_policy ≫ D_live`                  ⇒ **부하**다. 레버는 전달(재제시)뿐 — 새 판단을 엔진이 하면 안 된다.
- `A_schema ≈ B_policy ≫ C_neg`        ⇒ **스키마 하나로 닫힌다**(정책 불필요).
- `B_policy ≫ A_schema`                ⇒ 정책 문서가 결정한다 ⇒ 전달 표적이 문서다.
- 전 팔이 낮고 gold 가 문서에서 유일하게 결정되지도 않으면 ⇒ **결정 불가 사례**로 표시하고
  그 태스크는 표적에서 뺀다([[68]] 계보).

사용: py -3 x423_choice_isolation.py [--n 3] [--port 8141] [--max-cases 40]
"""
import argparse
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

RUNS = ["bank_t7326_halfA_20260819q", "bank_t7326_halfB_20260819q",
        "bank_t7328_halfA_20260819r", "bank_t7328_halfB_20260819r2"]
SUF = ".results.json.gz"
PAD = 2600            # 문서 슬라이스 최대 길이(축자)
TAIL = 10             # D_live 꼬리 메시지 수

SYS = ("You are the argument-filling module of a Rho-Bank support agent. "
       "Reply with ONE JSON object only: {\"value\": <value>}. No prose, no markdown fence.")


def is_choice(v):
    """숫자가 아닌 값 = 선택 축(문자열 멤버·id·enum). 숫자는 x424(계산)로 보낸다."""
    s = str(v).strip()
    if not s:
        return False
    try:
        float(s)
        return False
    except Exception:
        return True


def env_args(tool):
    p = os.path.join(HERE, "a2", "env_surface.json")
    with io.open(p, encoding="utf-8") as f:
        d = json.load(f)
    return (d["banking_knowledge"]["tools"].get(tool) or {}).get("args") or []


def env_desc(tool):
    p = os.path.join(HERE, "a2", "env_surface.json")
    with io.open(p, encoding="utf-8") as f:
        d = json.load(f)
    return (d["banking_knowledge"]["tools"].get(tool) or {}).get("desc") or ""


def schema_slice(sim, tool, upto):
    """그 도구의 **인자 이름이 전부 들어 있는** 도구-결과를 축자로.

    발견-도구는 `Tool unlocked:` 응답이 파라미터 블록을 싣지만 **네이티브 도구는 안 싣는다**(스키마가
    도구 목록에 상주하기 때문). 그 경우 `env_surface.json` 의 **환경 선언 축자**(desc + args)로 대신한다 —
    출처가 환경이므로 [[23]] 안이고, 라이브 에이전트도 같은 것을 갖고 있었다([[18]] 정보-맞춘).
    """
    args = set(env_args(tool))
    best = ""
    if args:
        for i, m in enumerate(sim.get("messages") or []):
            if i >= upto or m.get("role") != "tool":
                continue
            b = " ".join(str(m.get("content") or "").split())
            if tool in b and all(a in b for a in args):
                best = b
    if not best:
        d = env_desc(tool)
        if d:
            best = "Tool: %s\nArguments: %s\n%s" % (tool, ", ".join(sorted(args)), d)
    return best[:PAD]


def policy_slices(sim, tool, upto, skip="", arg=""):
    """그 도구 이름이 나오는 **다른** 도구-결과들(문서)을 축자로 이어 붙인다.

    도구 이름을 문서가 안 부르는 경우가 있다(네이티브 도구). 그때는 **인자 이름**(`account_class` 등)이
    나오는 문서로 대신한다 — 그 인자의 후보 목록을 정하는 것이 바로 그 문서이기 때문이다.
    둘 다 없으면 빈 문자열(그 사례는 정책 팔이 스키마 팔과 같아진다 — 표에 그대로 드러난다).
    """
    def collect(needle):
        out, seen = [], 0
        for i, m in enumerate(sim.get("messages") or []):
            if i >= upto or m.get("role") != "tool":
                continue
            b = " ".join(str(m.get("content") or "").split())
            if needle not in b or (skip and b[:200] == skip[:200]):
                continue
            out.append(b[:PAD])
            seen += len(b)
            if seen > PAD * 2:
                break
        return out
    got = collect(tool)
    if not got and arg:
        got = collect(arg)
    return "\n\n".join(got)


def cases(maxn):
    """WRONGARG 의 **비수치 필드**마다 사례 하나. gold 는 채점용으로만 들고 다닌다."""
    out = []
    per = collections.Counter()
    for tag in RUNS:
        for sim in F.sims(tag, SUF):
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
                    if str(gv) == str(wv) or gv is None or not is_choice(gv):
                        continue
                    key = (F.task_id(sim), w["name"], k)
                    if per[key] >= 2:          # 태스크·인자당 최대 2 sim
                        continue
                    per[key] += 1
                    sch = schema_slice(sim, w["name"], w["msg_i"])
                    out.append({"task": F.task_id(sim), "trial": sim.get("trial"), "tag": tag,
                                "tool": w["name"], "arg": k, "gold": str(gv), "live": str(wv),
                                "msg_i": w["msg_i"], "sim": sim, "schema": sch,
                                "policy": policy_slices(sim, w["name"], w["msg_i"], sch, k)})
                    if len(out) >= maxn:
                        return out
    return out


def build_arms(c):
    """팔 4종의 프롬프트 본문. **x425 가 이 함수를 그대로 쓴다** — 프롬프트가 갈리면 두 측정이 못 붙는다([[67]])."""
    sim = c["sim"]
    head = ("# 손님 요청\n%s\n\n# 지금 채워야 하는 것\n도구 `%s` 의 인자 `%s` 하나의 값.\n"
            % (X.user_ask(sim), c["tool"], c["arg"]))
    q = ("\n# 질문\n그 인자에 넣을 값 하나만 내라. JSON 하나로만: {\"value\": <값>}\n")
    sch = ("\n# 도구 스키마(축자)\n%s\n" % c["schema"]) if c["schema"] else ""
    pol = ("\n# 정책 문서(축자)\n%s\n" % c["policy"]) if c["policy"] else ""
    win = (sim.get("messages") or [])[:c["msg_i"]][-TAIL:]
    live = "\n# 대화(꼬리 %d메시지)\n%s\n" % (len(win),
                                          X.convo({"messages": win}, respect_close=False))
    return {"C_neg": head + q,
            "A_schema": head + sch + q,
            "B_policy": head + sch + pol + q,
            "D_live": head + live + q,
            "E_evidence": head + sch + evidence_block(sim, c["msg_i"]) + q}


EVID_BUDGET = 26000


def evidence_block(sim, upto, budget=EVID_BUDGET):
    """★진짜 정보-맞춘 격리([[18]]): 그 결정점까지 **에이전트가 받은 도구-결과 전부**를 축자로 주고,
    어시스턴트 자기 산문과 대화 잡음만 뺀다. 정보는 그대로 두고 **부하만** 던다.

    1차 실행에서 `B_policy` 팔은 40 사례 중 **30 에서 정답 문자열이 팔 안에 아예 없었다** — 그 팔로 잰
    낮은 적중은 능력이 아니라 우리 프로브의 정보 빈약이다. 이 팔이 그 결함을 고친다.
    """
    parts, tot = [], 0
    for i, m in enumerate(sim.get("messages") or []):
        if i >= upto or m.get("role") != "tool":
            continue
        b = " ".join(str(m.get("content") or "").split())
        if not b:
            continue
        parts.append(b)
        tot += len(b)
    while tot > budget and len(parts) > 1:
        tot -= len(parts.pop(0))            # 오래된 것부터 버린다(최근 증거 우선)
    return "\n\n".join(["\n# 회수한 증거 전부(축자·도구 응답 %d건)" % len(parts)] + parts) + "\n"


def norm(v):
    return " ".join(str(v).split()).strip().strip('"').lower()


def parse_value(raw):
    t = str(raw or "").strip()
    if "```" in t:
        t = t.split("```")[1] if len(t.split("```")) > 1 else t
        t = t.split("\n", 1)[-1] if t[:4].lower() in ("json",) else t
    i, j = t.find("{"), t.rfind("}")
    if i >= 0 and j > i:
        try:
            o = json.loads(t[i:j + 1])
            if isinstance(o, dict) and "value" in o:
                return o["value"]
        except Exception:
            pass
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--port", type=int, default=8141)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--max-cases", type=int, default=40)
    a = ap.parse_args()

    cs = cases(a.max_cases)
    print("=" * 100)
    print("x423 · 선택 격리 · 사례 %d (고유 task/arg %d) · 팔 4 · n=%d · 포트 %d"
          % (len(cs), len(set((c["task"], c["arg"]) for c in cs)), a.n, a.port))
    print("스키마 슬라이스 있음 %d / 정책 슬라이스 있음 %d"
          % (sum(1 for c in cs if c["schema"]), sum(1 for c in cs if c["policy"])))
    print("=" * 100)

    jobs = []
    for c in cs:
        for an, body in build_arms(c).items():
            adm = norm(c["gold"]) in " ".join(body.split()).lower()
            for k in range(a.n):
                jobs.append({"c": c, "arm": an, "k": k, "temp": (0.0 if k == 0 else a.temp),
                             "body": body, "admissible": adm})
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
                            "messages": [{"role": "system", "content": SYS},
                                         {"role": "user", "content": j["body"]}],
                            "temperature": j["temp"], "max_tokens": 120})
                raw = d["choices"][0]["message"]["content"]
            except Exception as e:
                raw = "ERROR " + str(e)[:160]
            v = parse_value(raw)
            c = j["c"]
            rec = {"task": c["task"], "trial": c["trial"], "tool": c["tool"], "arg": c["arg"],
                   "arm": j["arm"], "k": j["k"], "gold": c["gold"], "live": c["live"],
                   "got": None if v is None else str(v)[:80],
                   "hit": v is not None and norm(v) == norm(c["gold"]),
                   "same_as_live": v is not None and norm(v) == norm(c["live"]),
                   "parsed": v is not None, "admissible": j.get("admissible")}
            with lock:
                out.append(rec)
                if len(out) % 40 == 0:
                    print("  ... %d/%d" % (len(out), len(out) + len(jobs)))

    ths = [threading.Thread(target=work, args=(i,)) for i in range(a.workers)]
    [t.start() for t in ths]
    [t.join() for t in ths]

    print("\n## 팔별 (gold 적중 / 라이브 오답 재현 / 파싱)")
    print("%-10s %6s %10s %12s %8s" % ("arm", "n", "HIT", "LIVE-REPEAT", "PARSED"))
    for arm in ("C_neg", "A_schema", "B_policy", "D_live", "E_evidence"):
        r = [x for x in out if x["arm"] == arm]
        if not r:
            continue
        n = float(len(r))
        print("%-10s %6d %10.3f %12.3f %8.3f"
              % (arm, len(r), sum(x["hit"] for x in r) / n,
                 sum(x["same_as_live"] for x in r) / n, sum(x["parsed"] for x in r) / n))

    print("\n## 인자별 B_policy 적중")
    byarg = collections.defaultdict(list)
    for x in out:
        if x["arm"] == "B_policy":
            byarg[(x["task"], x["arg"])].append(x["hit"])
    for k in sorted(byarg):
        v = byarg[k]
        print("   %-9s %-30s %d/%d" % (k[0], k[1][:30], sum(v), len(v)))

    p = os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026", "x423_choice_isolation.json")
    with io.open(os.path.abspath(p), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("\n→ %s" % os.path.abspath(p))
    return 0


if __name__ == "__main__":
    sys.exit(main())
