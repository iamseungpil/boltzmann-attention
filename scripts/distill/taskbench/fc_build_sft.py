#!/usr/bin/env python
"""P2: TaskBench+SOPBench native-FC 궤적 통합 -> 단일 SFT JSONL.

설계 = NATIVE_FC_CONVERTER_DESIGN_2026_06_14.md v2 §5·§4(D1,D5,D7).
- ★per-trajectory 랜덤 전역 alias(func_NNNN, 벤치 단서 0): 같은 도구가 궤적마다 다른 alias →
  모델은 tools= 컨텍스트서 심볼을 *복사*해야 호출 가능 = R1(grounding) 강제. lexical 암기 차단.
  --no_alias = ablation(실명 유지).
- loss-mask = assistant-only(출력에 supervise="assistant" 명시; 토큰 마스킹은 P3 학습 하네스).
- QC(D7): 스키마 유효·assistant call ∈ tools·tool 페어링(tool_call_id 매칭)·구성통계·합성결과 비율(리스크#2 지표).

Usage: fc_build_sft.py --inputs a.jsonl b.jsonl ... --out sft.jsonl [--no_alias] [--seed 42] [--max_per_source N]
"""
import argparse, json, random
from collections import Counter


def alias_traj(ex, rng, do_alias):
    names = [t["function"]["name"] for t in ex["tools"]]
    if do_alias:
        al = ["func_%04d" % i for i in range(1, len(names) + 1)]
        rng.shuffle(al)
        amap = dict(zip(names, al))
        for t in ex["tools"]:
            t["function"]["name"] = amap[t["function"]["name"]]
        for m in ex["messages"]:
            if m.get("role") == "assistant":
                for tc in m.get("tool_calls", []):
                    tc["function"]["name"] = amap.get(tc["function"]["name"], tc["function"]["name"])
    return ex


def qc(ex):
    errs = []
    toolnames = {t["function"]["name"] for t in ex["tools"]}
    open_ids = set()
    if not ex["messages"] or ex["messages"][0]["role"] != "system":
        errs.append("no-system")
    n_sup = 0
    for m in ex["messages"]:
        r = m["role"]
        if r == "assistant":
            if m.get("tool_calls"):
                n_sup += 1
                for tc in m["tool_calls"]:
                    if tc["function"]["name"] not in toolnames:
                        errs.append("call-not-in-tools:%s" % tc["function"]["name"])
                    open_ids.add(tc["id"])
                    # arguments must be valid json string
                    try:
                        json.loads(tc["function"]["arguments"])
                    except Exception:
                        errs.append("bad-args-json")
            elif (m.get("content") or "").strip():
                n_sup += 1
        elif r == "tool":
            if m.get("tool_call_id") and m["tool_call_id"] not in open_ids:
                # SOPBench ids may not perfectly pair if upstream dropped; flag soft
                errs.append("orphan-tool-result")
    if n_sup == 0:
        errs.append("no-supervised-turn")
    return errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no_alias", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_per_source", type=int, default=0)
    ap.add_argument("--max_per_bench", type=int, default=0,
                    help="벤치별 궤적 상한(균형=리스크#2 합성결과 오염 통제). 0=무제한")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    allex = []
    for path in a.inputs:
        recs = [json.loads(l) for l in open(path, encoding="utf-8")]
        if a.max_per_source:
            recs = recs[:a.max_per_source]
        allex.extend(recs)
    rng.shuffle(allex)
    if a.max_per_bench:
        kept, perb = [], Counter()
        for ex in allex:
            b = ex["_meta"]["bench"]
            if perb[b] < a.max_per_bench:
                perb[b] += 1
                kept.append(ex)
        allex = kept

    bench = Counter()
    sT = sF = 0
    n_calls = []
    n_turns = []
    synth_results = real_results = 0
    qc_fail = Counter()
    clean = []
    for ex in allex:
        ex = alias_traj(ex, rng, not a.no_alias)
        errs = qc(ex)
        if errs:
            for e in errs:
                qc_fail[e.split(":")[0]] += 1
            # hard errors drop; soft (orphan-tool-result) keep
            if any(not e.startswith("orphan-tool-result") for e in errs):
                continue
        b = ex["_meta"]["bench"]
        bench[b] += 1
        ss = ex["_meta"].get("should_succeed")
        if b == "sopbench":
            sT += 1 if ss else 0
            sF += 0 if ss else 1
        calls = sum(len(m.get("tool_calls", [])) for m in ex["messages"] if m["role"] == "assistant")
        n_calls.append(calls)
        n_turns.append(sum(1 for m in ex["messages"] if m["role"] == "assistant"))
        for m in ex["messages"]:
            if m["role"] == "tool":
                c = str(m.get("content", ""))
                if c.startswith('{"status": "ok"'):
                    synth_results += 1
                else:
                    real_results += 1
        ex["supervise"] = "assistant"
        clean.append(ex)

    with open(a.out, "w", encoding="utf-8") as f:
        for ex in clean:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    tot = len(clean)
    print("=== SFT BUILD ===")
    print("total trajectories: %d  (alias=%s)" % (tot, not a.no_alias))
    print("bench mix:", dict(bench))
    print("SOPBench should_T:should_F = %d:%d" % (sT, sF))
    print("avg calls/traj %.2f  avg assistant-turns/traj %.2f" % (sum(n_calls) / max(tot, 1), sum(n_turns) / max(tot, 1)))
    tr = synth_results + real_results
    print("tool-results: real %d / synthetic %d (synthetic frac %.1f%% = result-pollution indicator)"
          % (real_results, synth_results, 100 * synth_results / max(tr, 1)))
    print("QC flags:", dict(qc_fail))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
