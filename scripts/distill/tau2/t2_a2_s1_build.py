#!/usr/bin/env python
"""S1 학습셋 빌드 (A2_FRONTEND §3 S1 — 실-도메인 verified distill 스모크).

합성 S0 데이터(과적합 위험)에 **실 도메인 (정책NL, 교사-컴파일 spec) 쌍**을 섞어
held-out 실 도메인 전이를 교정. 실 레코드 = (실 policy.md 전문, A1 카탈로그, Fable-5
컴파일 spec). 합성 dataset 포맷으로 정규화해 기존 t2_a2_s0_build_sft가 처리.

수용(설계 정신): 실 spec은 replay 필터 통과분만 — telecom 검증기 부재라 스모크에선
frontier 교사 신뢰(P-A2-0 airline replay-clean 근거); P5(대형모델 교사)·SOPBench
도메인 합류 시 replay 필터 정식 적용. airline은 학습 제외 = held-out 평가축
(S0-v2 airline census 0.528과 직접 비교).

Usage: t2_a2_s1_build.py --synth specs/a2_s0_dataset_v6.jsonl \
  --real retail:specs/retail_policy.md:tau2_adapter/retail_tool_catalog.json:specs/retail_gate_spec_fable5.json \
  --real telecom:specs/s1_inputs/telecom_policy.md:specs/s1_inputs/telecom_tool_catalog.json:specs/s1_inputs/telecom_gate_spec_fable5.json \
  --oversample 5 --out specs/a2_s1_dataset.jsonl
"""
import argparse, json


def norm_catalog(cat):
    """실 카탈로그({enum,tools} 또는 tools-직접)를 합성 포맷(tool->{type,required})로."""
    tools = cat.get("tools", cat)
    return {k: {"type": v.get("type", "GENERIC"), "required": v.get("required", [])}
            for k, v in tools.items()}


def strip_meta(spec):
    return {k: v for k, v in spec.items() if not k.startswith("_")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth", required=True, help="합성 S0 dataset jsonl")
    ap.add_argument("--real", action="append", default=[],
                    help="name:policy.md:catalog.json:spec.json (학습 편입 실 도메인)")
    ap.add_argument("--oversample", type=int, default=1,
                    help="실 레코드 복제 횟수 (소수 실 도메인 신호 증폭)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.synth, encoding="utf-8") if l.strip()]
    n_synth = len(rows)
    real_recs = []
    for r in a.real:
        name, pol, cat, spec = r.split(":", 3)
        rec = {
            "id": f"real_{name}", "style": "real-domain", "qc": "teacher-fable5",
            "domain_hint": name,
            "catalog": norm_catalog(json.load(open(cat, encoding="utf-8"))),
            "spec": strip_meta(json.load(open(spec, encoding="utf-8"))),
            "policy_nl": open(pol, encoding="utf-8").read(),
        }
        real_recs.append(rec)
    for rec in real_recs:
        rows.extend([dict(rec) for _ in range(a.oversample)])
    with open(a.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[s1-build] synth={n_synth} real={len(real_recs)}x{a.oversample} "
          f"-> total={len(rows)} -> {a.out}")
    for rec in real_recs:
        print(f"  real: {rec['domain_hint']} gates={len(rec['spec'])} "
              f"tools={len(rec['catalog'])} nl_chars={len(rec['policy_nl'])}")


if __name__ == "__main__":
    main()
