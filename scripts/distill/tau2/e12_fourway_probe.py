#!/usr/bin/env python3
"""E12 — tau2 결정점 4지선다 프로브: 전체 히스토리 vs 정보-맞춘 짧은 문맥.

물음(사용자 2026-07-09): *"부하 문제면 짧은 턴으로 풀게 하고 히스토리는 외부 loop이 관리하는 게 맞지 않나?"*
그리고 4지선다: 1 GET(도구 호출) · 2 FIND(문맥서 찾기) · 3 INFER(문맥서 유도) · 4 ASK(사용자에게 묻기).

**②−① 이 load의 크기다.** ①에서 규칙이 안 먹는 건 이미 안다(policy.md:18 · C30).

정답 라벨 = gold 값의 *거처* 로 자동 도출 (오늘 주소 21건에 쓴 방법):
  gold 값이 도구 출력/사용자 발화에 문자열로 있음        → FIND
  없지만 producer 도구(get_*)로 얻을 수 있음(DB에 존재)  → GET
  문맥의 값들로 유도해야 함(최저가·합계 등)              → INFER   (본 프로브선 미분류·GET/FIND 우선)
  어디에도 없음(사용자만 안다)                           → ASK

arm:
  full   : 원 궤적의 접두 전체 (통제군 · 충실도 검정됨 50/50)
  short  : 정보-맞춤 압축 문맥 — {사용자 요구 요약, 문맥에 실재하는 값들과 그 출처, 도구 목록}
  short+rule : short + 규칙 문장 1줄

★C10 경고: 짧게 자르면 채널 폐쇄로 성능이 떨어질 수 있고, 그걸 '능력 없음'으로 오독하면 안 된다.
그래서 short는 **정보를 버리지 않는다** — 원 문맥의 *값과 출처*를 전부 싣되 대화 서사만 제거한다.

Run: python3 e12_fourway_probe.py --arm full,short,short_rule --n 40
"""

import argparse
import gzip
import json
import re
import sys
import urllib.request

SIM = "/home/woori/workspace_common/boltzmann-attention-pi/reports/facet_rft_2026/sim_results/"
DBP = "/home/woori/scratch/tau2-bench/data/tau2/domains/retail/db.json"
POLICY = "/home/woori/scratch/tau2-bench/data/tau2/domains/retail/policy.md"

RULE = ("Rule: never invent an argument value. Every value you write must come from the user's words "
        "or from a tool's output. If you don't have it, get it or ask.")

CHOICES = """You must choose exactly ONE of these for the argument in question, and answer with JSON only:
{"choice": "GET",   "tool": "<tool name to call to obtain the value>"}
{"choice": "FIND",  "value": "<the exact value, copied from the context above>"}
{"choice": "INFER", "criterion": "<how to derive it from values above, e.g. min price>"}
{"choice": "ASK",   "question": "<what to ask the user>"}
Answer with the JSON object only."""


def _norm(t):
    return re.sub(r"\s+", " ", str(t).lower().replace("#", "")).strip()


def load_db():
    db = json.load(open(DBP, encoding="utf-8"))
    item2prod = {}
    for pid, p in db["products"].items():
        for v in (p.get("variants") or {}):
            item2prod[str(v)] = str(pid)
    return db, item2prod


DB, ITEM2PROD = load_db()
ORDERS, USERS = DB["orders"], DB["users"]


def gold_label(sim, tc, key, gold_value, upto):
    """gold 값의 거처 → 정답 갈래.

    ★upto = 결정 시점 인덱스. 그 *이전* 문맥만 본다.
    (버그 2026-07-09: sim 전체를 보면 결정 이후 조회로 들어온 값까지 FIND로 오라벨됨.)
    """
    if not gold_value:
        return "ASK"          # gold에 그 값이 없다 = 사용자만 안다
    gv = _norm(gold_value)
    prefix = sim["messages"][:upto]
    user_txt = " ".join(_norm(m.get("content")) for m in prefix
                        if m.get("role") == "user" and isinstance(m.get("content"), str))
    tool_txt = " ".join(_norm(m.get("content")) for m in prefix
                        if m.get("role") == "tool" and isinstance(m.get("content"), str))
    if gv in user_txt or gv in tool_txt:
        return "FIND"
    if key in ("new_item_ids", "item_ids") and str(gold_value) in ITEM2PROD:
        return "GET"
    if key == "payment_method_id":
        for u in USERS.values():
            if str(gold_value) in (u.get("payment_methods") or {}):
                return "GET"
    if key == "address1":
        for u in USERS.values():
            if (u.get("address") or {}).get("address1") == gold_value:
                return "GET"
        for o in ORDERS.values():
            if (o.get("address") or {}).get("address1") == gold_value:
                return "GET"
        return "ASK"          # 새 주소는 사용자만 안다
    return "ASK"


def values_with_source(sim, upto):
    """정보-맞춤 압축: 접두 문맥에 실재하는 값 + 출처. 서사만 버리고 값은 전부 싣는다."""
    out = []
    for m in sim["messages"][:upto]:
        r = m.get("role")
        c = m.get("content")
        if not isinstance(c, str) or not c.strip():
            continue
        if r == "user":
            out.append(("user said", c.strip()[:400]))
        elif r == "tool":
            out.append(("tool output", c.strip()[:900]))
    return out


def build_full(sim, upto, key, tcname):
    msgs = [{"role": "system", "content": open(POLICY, encoding="utf-8").read()}]
    for m in sim["messages"][:upto]:
        r = m.get("role")
        if r == "user":
            msgs.append({"role": "user", "content": m.get("content") or ""})
        elif r == "assistant":
            msgs.append({"role": "assistant", "content": m.get("content") or ""})
        elif r == "tool":
            msgs.append({"role": "user", "content": f"[tool output] {m.get('content')}"})
    msgs.append({"role": "user", "content":
                 f"You are about to call `{tcname}`. You need the argument `{key}`.\n{CHOICES}"})
    return msgs


def build_short(sim, upto, key, tcname, tools, rule):
    lines = [f"- {src}: {txt}" for src, txt in values_with_source(sim, upto)]
    ctx = "\n".join(lines)
    sysm = "You are a tool-using assistant." + (("\n" + RULE) if rule else "")
    body = (f"Everything known so far:\n{ctx}\n\n"
            f"Available tools: {', '.join(tools)}\n\n"
            f"You are about to call `{tcname}`. You need the argument `{key}`.\n{CHOICES}")
    return [{"role": "system", "content": sysm}, {"role": "user", "content": body}]


def chat(base, model, msgs, timeout=150):
    body = json.dumps({"model": model, "messages": msgs, "temperature": 0.0,
                       "max_tokens": 160}).encode()
    req = urllib.request.Request(base.rstrip("/") + "/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())["choices"][0]["message"].get("content") or ""


def parse_choice(txt):
    m = re.search(r"\{.*?\}", txt or "", re.S)
    if not m:
        return None, {}
    try:
        d = json.loads(m.group(0))
        return d.get("choice"), d
    except Exception:
        return None, {}


def main():
    sys.path.insert(0, "/home/woori/scratch")
    from e11a_isolated_probe import find_violations, WRITE

    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8140/v1")
    ap.add_argument("--model", default="Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8")
    ap.add_argument("--arm", default="full,short,short_rule")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--tag", default="fl32b_floor_retail_t4")
    a = ap.parse_args()

    sims = json.load(gzip.open(SIM + a.tag + ".results.json.gz"))["simulations"]
    viol = find_violations(sims, a.n)
    tools = sorted(WRITE | {"get_order_details", "get_product_details", "get_user_details",
                            "get_item_details", "find_user_id_by_email", "find_user_id_by_name_zip",
                            "list_all_product_types", "calculate"})

    from collections import Counter
    labels, rows = [], []
    for sim, idx, tc, key, val, want in viol:
        gv = None
        for ac in (sim.get("reward_info") or {}).get("action_checks") or []:
            act = ac.get("action") or {}
            if act.get("name") == tc.get("name"):
                v = (act.get("arguments") or {}).get(key)
                gv = v[0] if isinstance(v, list) and v else v
                break
        lab = gold_label(sim, tc, key, gv, idx)
        if lab:
            labels.append((sim, idx, tc, key, lab))
    print(f"라벨링된 결정점 {len(labels)} / 위반 {len(viol)}")
    print("정답 분포:", dict(Counter(l[4] for l in labels)), flush=True)

    arms = a.arm.split(",")
    stats = {arm: Counter() for arm in arms}
    for sim, idx, tc, key, lab in labels:
        row = {"task": str(sim.get("task_id")), "trial": sim.get("trial"), "arg": key, "gold": lab}
        for arm in arms:
            if arm == "full":
                msgs = build_full(sim, idx, key, tc.get("name"))
            else:
                msgs = build_short(sim, idx, key, tc.get("name"), tools, rule=(arm == "short_rule"))
            try:
                ch, _ = parse_choice(chat(a.base, a.model, msgs))
            except Exception:
                ch = "ERR"
            row[arm] = ch or "unparsed"
            stats[arm][("✓" if ch == lab else "✗") + f" {ch}"] += 1
        rows.append(row)
        print("  " + json.dumps(row, ensure_ascii=False), flush=True)

    print(f"\n{'arm':<12}{'정확도':>9}{'GET':>7}{'FIND':>7}{'INFER':>7}{'ASK':>7}{'unparsed':>10}")
    for arm in arms:
        c = stats[arm]
        n = sum(c.values()) or 1
        acc = sum(v for k, v in c.items() if k.startswith("✓")) / n
        pick = Counter(k.split(" ", 1)[1] for k in c.elements())
        print(f"{arm:<12}{acc:>9.2f}{pick['GET']:>7}{pick['FIND']:>7}{pick['INFER']:>7}"
              f"{pick['ASK']:>7}{pick['unparsed']:>10}")
    print("\n★ short − full = load 의 크기.  short_rule − short = 짧은 문맥서 규칙의 값.")


if __name__ == "__main__":
    main()
