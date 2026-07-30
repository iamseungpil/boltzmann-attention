# -*- coding: utf-8 -*-
"""X2 E-RESTATE-M — 재진술-충실도 오프라인 측정기 (논문② Table 1·EXPERIMENT_PLAN §1-X2).

오프라인 분석 도구(런타임 scaffold 아님 — P2-b 무관·[[08]] 포렌식 계열).
정의: assistant 산문에 등장하는 통화-금액이, 그 시점까지의 원장
      (도구 출력 ∪ 유저 발화)에 부재하면 = 미접지 재진술(암산/날조 후보).
함정 대응(handoff §8): results.json의 messages 전문 사용(덤프-절단 아님)·
      대조는 정규화 변형 집합(콤마·float 정형·후행 0)으로.

사용: py -3 x2_restatement_fidelity.py <glob...>  (기본: 인자 경로들)
출력: 런별/합계 표 + 예시 JSONL(스팟 정독용·reports/.../x2_examples.jsonl)
"""
import gzip, json, re, sys, os, glob, collections

MONEY = re.compile(r"[$]\s?(\d[\d,]*(?:\.\d{1,2})?)")
BARE_DEC = re.compile(r"(?<![\d.])(\d{1,3}(?:,\d{3})*\.\d{2})(?![\d%])")  # 1,234.56 형


def variants(num):
    s = num.replace(",", "").lstrip("$").strip()
    v = {s}
    try:
        f = float(s)
        v.add(repr(f))
        v.add("%.2f" % f)
        if f == int(f):
            v.add(str(int(f)))
    except ValueError:
        return v
    if "." in s:
        t = s.rstrip("0").rstrip(".")
        if t:
            v.add(t)
    return v


def amounts(text):
    """span-dedup(2026-07-30 gold 감사 결함 수정): MONEY·BARE_DEC 중복 매칭 제거 —
    같은 위치의 금액은 1회만. + 동일 턴 내 같은 금액의 반복 등장도 1회로(주장 단위)."""
    spans, seen_vals, out = set(), set(), []
    for rx in (MONEY, BARE_DEC):
        for m in rx.finditer(text or ""):
            key = (m.start(1) // 1, m.group(1))
            span_overlap = any(not (m.end(1) <= s or m.start(1) >= e) for s, e in spans)
            if span_overlap or m.group(1) in seen_vals:
                continue
            spans.add((m.start(1), m.end(1)))
            seen_vals.add(m.group(1))
            out.append(m.group(1))
    return out


def norm_ctx(parts):
    return " ".join(parts).replace(",", "")


def analyze_sim(sim):
    msgs = sim.get("messages") or []
    ledger_parts = []   # 도구 출력 + 유저 발화 원문 (콤마 정규화는 대조 시)
    rows = []
    for i, m in enumerate(msgs):
        role = m.get("role")
        content = m.get("content")
        content = content if isinstance(content, str) else ""
        if role in ("tool", "user"):
            ledger_parts.append(content)
            continue
        if role != "assistant" or not content:
            continue
        ams = amounts(content)
        if not ams:
            continue
        ctx = norm_ctx(ledger_parts)
        for a in ams:
            grounded = any(v in ctx for v in variants(a))
            rows.append({
                "turn_idx": i, "amount": a, "grounded": grounded,
                "snippet": content[max(0, content.find(a) - 60):content.find(a) + 40],
            })
    return rows


def main(paths):
    per_run = {}
    examples = []
    for p in paths:
        try:
            with gzip.open(p, "rt", encoding="utf-8") as f:
                d = json.load(f)
        except Exception as e:
            print("SKIP %s: %r" % (p, e)); continue
        sims = d.get("simulations") or []
        agent_model = ((d.get("info") or {}).get("agent_info") or {}).get("llm") or \
                      (d.get("info") or {}).get("agent_model") or "?"
        tot = collections.Counter()
        for s in sims:
            rows = analyze_sim(s)
            rw = (s.get("reward_info") or {}).get("reward")
            passed = (rw == 1 or rw == 1.0)
            for r in rows:
                tot["amounts"] += 1
                key = "ungrounded" if not r["grounded"] else "grounded"
                tot[key] += 1
                tot[("%s_%s" % (key, "pass" if passed else "fail"))] += 1
                if not r["grounded"] and len(examples) < 400:
                    examples.append({"run": os.path.basename(p), "task": s.get("task_id"),
                                     "reward": rw, **r})
            tot["sims"] += 1
            if any(not r["grounded"] for r in rows):
                tot["sims_with_ungrounded"] += 1
        per_run[os.path.basename(p)] = (agent_model, tot)
    # 표 출력
    print("%-52s %-28s %6s %6s %6s %8s" % ("run", "agent", "sims", "amts", "unGr", "unGr%"))
    agg = collections.Counter()
    for run, (model, t) in sorted(per_run.items()):
        n, a, u = t["sims"], t["amounts"], t["ungrounded"]
        agg.update(t)
        print("%-52s %-28s %6d %6d %6d %7.1f%%" % (run[:52], str(model)[-28:], n, a, u,
                                                   (100.0 * u / a) if a else 0.0))
    a, u = agg["amounts"], agg["ungrounded"]
    print("TOTAL sims=%d amounts=%d ungrounded=%d (%.1f%%) | sims_with_ungrounded=%d" %
          (agg["sims"], a, u, (100.0 * u / a) if a else 0, agg["sims_with_ungrounded"]))
    print("pass-sim unGr=%d fail-sim unGr=%d" % (agg["ungrounded_pass"], agg["ungrounded_fail"]))
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "..", "..", "reports", "facet_rft_2026", "x2_examples.jsonl")
    with open(os.path.normpath(out), "w", encoding="utf-8") as f:
        for e in examples:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    print("examples ->", os.path.normpath(out), len(examples))


if __name__ == "__main__":
    args = sys.argv[1:]
    paths = []
    for a in args:
        paths.extend(glob.glob(a))
    main(sorted(set(paths)))
