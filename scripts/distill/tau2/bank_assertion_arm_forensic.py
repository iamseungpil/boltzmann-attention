# -*- coding: utf-8 -*-
"""assertion-provenance arm 사후 포렌식 (설계문 §4 계측·[[08]] 집계금지).

pass/count에서 결론 직행 금지 — 궤적서 아래를 직접 센다:
  (1) 종료사유 분포 (crash/infra 배제)
  (2) ★regen 後 producer를 *실제로* 불렀나 (= 레버가 샀나) vs 같은 주장 재발화 (= thrash)
  (3) 도구 시퀀스 전수
  (4) over-block 후보: 발화했는데 그 대화가 producer와 무관했나 (원문 정독용 출력)

사용: py -3 bank_assertion_arm_forensic.py <results.json[.gz]> [--tag NAME]
"""
import gzip, json, io, sys, os

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


def _open(p):
    return gzip.open(p, "rt", encoding="utf-8") if p.endswith(".gz") else open(p, encoding="utf-8")


def _names(m):
    return [tc.get("name") or (tc.get("function") or {}).get("name")
            for tc in (m.get("tool_calls") or [])]


def main(path, tag):
    with _open(path) as f:
        data = json.load(f)
    sims = data.get("simulations") or []
    print("=" * 74)
    print(f"# {tag}  ({os.path.basename(path)})  sims={len(sims)}")
    term = {}
    for sim in sims:
        term[sim.get("termination_reason")] = term.get(sim.get("termination_reason"), 0) + 1
    print(f"# (1) 종료사유 분포: {term}   ← crash/infra 있으면 결론 보류")
    rewards = [(s.get("reward_info") or {}).get("reward") for s in sims]
    print(f"# pass(reward): {rewards}  (★점추정 — 단독 결론 금지)")

    for si, sim in enumerate(sims):
        msgs = sim.get("messages") or []
        seq, prod_idx = [], []
        for i, m in enumerate(msgs):
            if m.get("role") != "assistant":
                continue
            for n in _names(m):
                if n:
                    seq.append(n)
                    if n == "get_reward_discrepancies":
                        prod_idx.append(i)
        print(f"\n--- sim{si}  reward={(sim.get('reward_info') or {}).get('reward')} "
              f"종료={sim.get('termination_reason')}")
        print(f"  (3) 도구 시퀀스({len(seq)}): {seq}")
        print(f"  ★producer(get_reward_discrepancies) 호출: {len(prod_idx)} @msg{prod_idx}")
        print(f"  (2) 판정: {'★샀다(regen→producer 호출)' if prod_idx else '☒못샀다(호출 0 — thrash/무효 후보)'}")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    tag = "arm"
    if "--tag" in sys.argv:
        tag = sys.argv[sys.argv.index("--tag") + 1]
    main(args[0], tag)
