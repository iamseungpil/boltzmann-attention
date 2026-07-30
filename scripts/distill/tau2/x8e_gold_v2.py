#!/usr/bin/env python3
"""X8-(e) gold 규약 v2 생성 — 8라벨 → 9라벨 개정 (2026-07-30).

`X8_GOLD_LABEL_PROTOCOL §3b` 개정 반영 (사용자 지적 "transfer 가 ask 아닌가"):
  `ASK`     → `REQ_INFO`   (우리 프레임의 ASK = 에이전트→유저·C48 위계와 이름 충돌 해소)
  `REQUEST` → `REQ_ACT` + `ESCALATE` 분리 (이관은 전용 결정론 경로를 타므로 별도 라벨)

★`REQ_ACT`/`ESCALATE` 분리는 **기계 변환이 아니라 건별 판정**이다(아래 ESC 집합은 48건 전문을
다시 읽고 정한 것). 두 라벨을 **함께** 받는 경우가 있다(u002·u005: 행동 요구와 이관 요구를 둘 다 함).

용법: py -3 x8e_gold_v2.py   (→ x8_gold_labels_v2.jsonl)
"""
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_SIM = os.path.abspath(os.path.join(_HERE, "..", "..", "..",
                                    "reports", "facet_rft_2026", "sim_results"))

# 이관 요구가 있는 건 (전문 재정독·11건)
ESCALATE = {"u002", "u003", "u005", "u006", "u021", "u022", "u023", "u030",
            "u039", "u046", "u047"}
# 이관 **외에** 별도 행동 요구도 함께 있는 건 (둘 다 부여·2건)
ALSO_REQ_ACT = {"u002",   # "submit the disputes from your side, OR escalate to your team"
                "u005"}   # "Can we verify another way?(전화 인증 시도) ... 안 되면 이관"


def convert(acts, sid):
    out = []
    for a in acts:
        if a == "ASK":
            out.append("REQ_INFO")
        elif a == "REQUEST":
            if sid in ESCALATE:
                out.append("ESCALATE")
                if sid in ALSO_REQ_ACT:
                    out.append("REQ_ACT")
            else:
                out.append("REQ_ACT")
        else:
            out.append(a)
    return sorted(set(out))


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    src = os.path.join(_SIM, "x8_gold_labels.jsonl")
    dst = os.path.join(_SIM, "x8_gold_labels_v2.jsonl")
    rows = [json.loads(l) for l in open(src, encoding="utf-8")]
    ids = {r["sample_id"] for r in rows}
    assert ESCALATE <= ids, f"미존재 id: {ESCALATE - ids}"
    assert ALSO_REQ_ACT <= ESCALATE, "ALSO_REQ_ACT 는 ESCALATE 부분집합이어야"

    from collections import Counter
    before, after = Counter(), Counter()
    n_multi_b = n_multi_a = 0
    with open(dst, "w", encoding="utf-8") as f:
        for r in rows:
            before.update(r["acts"])
            n_multi_b += len(r["acts"]) > 1
            v2 = convert(r["acts"], r["sample_id"])
            after.update(v2)
            n_multi_a += len(v2) > 1
            f.write(json.dumps({"sample_id": r["sample_id"], "acts": v2,
                                "slots": r["slots"], "note": r["note"],
                                "acts_v1": r["acts"]}, ensure_ascii=False) + "\n")
    print(f"v1 라벨 분포: {dict(before.most_common())}")
    print(f"v2 라벨 분포: {dict(after.most_common())}")
    print()
    print(f"다중-act: v1 {n_multi_b}/{len(rows)} → v2 {n_multi_a}/{len(rows)}")
    print(f"ESCALATE {after['ESCALATE']}건 · REQ_ACT {after['REQ_ACT']}건 "
          f"(v1 REQUEST {before['REQUEST']} → 합 {after['ESCALATE'] + after['REQ_ACT']}·"
          f"둘 다 받은 {len(ALSO_REQ_ACT)}건 때문에 증가)")
    print(f"REQ_INFO {after['REQ_INFO']}건 (v1 ASK {before['ASK']})")
    print(f"\n[saved] {dst}")


if __name__ == "__main__":
    main()
