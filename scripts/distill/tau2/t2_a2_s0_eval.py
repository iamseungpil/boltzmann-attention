#!/usr/bin/env python
"""S0 held-out 평가: 합성 holdout (NL,spec) 쌍에 대해 모델 컴파일 -> GT 대비 채점.

학습-포맷 프롬프트(빌드 산출물의 messages 그대로) -> chat completions (temp 0,
json_object) -> 채점: exact-EM(canonical JSON) · gate_recall · applies_F1
(t2_a2_size_census.score 재사용 — ref=GT spec).

Usage: t2_a2_s0_eval.py --holdout /home/woori/scratch/a2_s0/sft_a2_s0_holdout.jsonl \
  --endpoint http://localhost:8000/v1 --model base:Qwen/Qwen2.5-7B-Instruct \
  --model s0:a2s0 --out /home/woori/scratch/a2_s0/holdout_eval
"""
import argparse, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from t2_a2_size_census import call, parse, score  # noqa: E402


def canon(o):
    return json.dumps(o, sort_keys=True, separators=(",", ":"))


def struct_canon(spec):
    """structure-EM 키: 게이트 kind + applies_to + satisfiers만 (prose 필드 제외).

    EM 0/15 부검(2026-06-12): canonical-EM은 predicate/ask 문구 변주에 깨짐 —
    구조 동등성만 비교하는 지표가 G-A2-1의 실질 기준.
    """
    items = []
    for k, v in (spec or {}).items():
        if k.startswith("_"):
            continue
        kind = k.split("_", 1)[1] if "_" in k else k  # G1_AUTH_FIRST -> AUTH_FIRST
        items.append((kind, canon(sorted(v.get("applies_to", []) or [])),
                      canon(v.get("satisfiers") or {})))
    return canon(sorted(items))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", required=True)
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--model", action="append", required=True, help="name:served_model")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    recs = [json.loads(l) for l in open(a.holdout, encoding="utf-8") if l.strip()]
    print(f"[s0-eval] holdout n={len(recs)}")
    for m in a.model:
        name, served = m.split(":", 1)
        em = sem = grs = afs = parsed = 0
        details = []
        for r in recs:
            sysm = r["messages"][0]["content"]
            usr = r["messages"][1]["content"]
            gt = json.loads(r["messages"][2]["content"])
            try:
                txt = call(a.endpoint, served, sysm, usr)
            except Exception as e:
                details.append({"id": r["meta"]["id"], "error": str(e)[:120]})
                continue
            gen = parse(txt)
            if gen is not None:
                parsed += 1
            gr, af, ng = score(gt, gen)
            ok_em = gen is not None and canon(gen) == canon(gt)
            ok_sem = gen is not None and struct_canon(gen) == struct_canon(gt)
            em += ok_em
            sem += ok_sem
            grs += gr
            afs += af
            details.append({"id": r["meta"]["id"], "style": r["meta"]["style"],
                            "em": ok_em, "structEM": ok_sem, "gate_recall": round(gr, 3),
                            "applies_F1": round(af, 3), "n_gates": ng})
        n = len(recs)
        print(f"{name}: EM={em}/{n} structEM={sem}/{n} gate_recall={grs / n:.3f} "
              f"applies_F1={afs / n:.3f} parsed={parsed}/{n}")
        json.dump(details, open(f"{a.out}.{name}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
