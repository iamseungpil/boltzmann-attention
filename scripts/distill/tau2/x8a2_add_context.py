#!/usr/bin/env python3
"""X8-(a2) 표본에 직전 에이전트 턴을 덧붙임 — arm Actx([[18]] B_fullctx 근사)용.

★왜 별도 스크립트인가: `x8a_sample_utterances.py`를 다시 돌려 필드를 추가하면 표본 자체가
바뀔 위험이 있고(층화 셀 재추첨), 그러면 이미 사람이 붙인 gold(`x8_gold_labels.jsonl`)와
정렬이 깨진다. 그래서 **선택은 건드리지 않고** 기존 표본 파일에 필드만 추가한다.
검증: sample_id 순서·집합이 불변임을 확인 후 저장.

용법: py -3 x8a2_add_context.py
"""
import glob
import gzip
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_SIM = os.path.abspath(os.path.join(_HERE, "..", "..", "..",
                                    "reports", "facet_rft_2026", "sim_results"))
_SAMPLE = os.path.join(_SIM, "x8_sample_utterances.jsonl")


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    rows = [json.loads(l) for l in open(_SAMPLE, encoding="utf-8")]
    before = [r["sample_id"] for r in rows]
    need = {}
    for i, r in enumerate(rows):
        need.setdefault(r["file"], []).append(i)

    filled = 0
    for fn, idxs in need.items():
        path = os.path.join(_SIM, fn)
        if not os.path.exists(path):
            print(f"[warn] {fn} 없음")
            continue
        data = json.load(gzip.open(path, "rt", encoding="utf-8"))
        sims = {s.get("id"): s for s in data.get("simulations") or []}
        for i in idxs:
            r = rows[i]
            sim = sims.get(r["sim_id"])
            if not sim:
                continue
            msgs = sim.get("messages") or []
            ti = r["turn_idx"]
            prev = ""
            for j in range(ti - 1, -1, -1):
                m = msgs[j]
                if m.get("role") == "assistant":
                    prev = str(m.get("content") or "").strip()
                    # 도구-호출만 있고 본문이 없는 턴은 건너뛰고 더 앞을 본다
                    if prev:
                        break
            r["prev_agent"] = prev
            if prev:
                filled += 1

    after = [r["sample_id"] for r in rows]
    assert before == after, "표본 순서가 바뀌었다 — gold 정렬 깨짐"
    with open(_SAMPLE, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"prev_agent 채움 {filled}/{len(rows)} · sample_id 불변 확인 ✓")
    empt = [r["sample_id"] for r in rows if not r.get("prev_agent")]
    if empt:
        print(f"직전 에이전트 본문 없음(첫 턴 등) {len(empt)}: {empt}")
    print(f"[saved] {_SAMPLE}")


if __name__ == "__main__":
    main()
