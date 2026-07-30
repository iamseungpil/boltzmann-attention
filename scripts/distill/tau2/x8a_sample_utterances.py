#!/usr/bin/env python3
"""X8-(a) triage 재료 추출 — day6~9c 유저 발화 층화 표본 (2026-07-30).

X8(`EXPERIMENT_PLAN_PATENT_PAPERS_2026_07_30.md` §1)의 1단계. 대상 formalize 기능 2종:
  ① user_act  분류      — 이 발화가 무엇을 하는가
  ② slot 추출-quote     — 이 발화가 공급하는 값을 축자로 뽑을 수 있는가

★gold 프로토콜(계획서 리뷰 수정 4-2): 라벨 출처 = day6~9c 유저 발화 **층화 표본 수작업 gold**.
이 스크립트는 **표본만** 만든다 — gold는 사람이 읽어서 별도 파일에 쓴다(모델로 gold를 만들면
[[03b]] cheating). 표본 크기는 전수 정독이 가능한 규모로 잡는다.

★층화 축(재현성 위해 명시):
  - day (6/7/8/9c)        — 스택 세대
  - 태스크 성패 (pass/fail) — 실패-원천 편중 방지
  - 발화 위치 (first/mid/last) — 첫 요청 vs 정보 공급 vs 종결이 기능적으로 다름
  - 슬롯 유무 (볼드 마커 `**…**` 존재) — slot 추출 대상 확보

★영속화([[30]] 구멍 교정): X4·X5 프로브가 stdout만 내서 원시 출력이 소실됐다(2026-07-30 확인).
이 계열 스크립트는 **항상 파일로 쓴다**.

용법: py -3 x8a_sample_utterances.py [--n 48] [--out sample.jsonl]
"""
import argparse
import glob
import gzip
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
_SIM = os.path.abspath(os.path.join(_HERE, "..", "..", "..",
                                    "reports", "facet_rft_2026", "sim_results"))
_DAYS = ["day6front", "day7front", "day8front", "day9cfront"]
_BOLD = re.compile(r"\*\*(.+?)\*\*")


def load_pool():
    """day6~9c results 에서 유저 발화 전수 추출."""
    pool = []
    for day in _DAYS:
        for path in sorted(glob.glob(os.path.join(_SIM, f"bank_{day}*.results.json.gz"))):
            try:
                data = json.load(gzip.open(path, "rt", encoding="utf-8"))
            except Exception as e:
                print(f"[warn] {os.path.basename(path)}: {e}", file=sys.stderr)
                continue
            for sim in data.get("simulations") or []:
                msgs = sim.get("messages") or []
                rw = (sim.get("reward_info") or {}).get("reward")
                users = [(i, m) for i, m in enumerate(msgs) if m.get("role") == "user"]
                for k, (idx, m) in enumerate(users):
                    text = str(m.get("content") or "").strip()
                    if not text:
                        continue
                    pos = "first" if k == 0 else ("last" if k == len(users) - 1 else "mid")
                    pool.append({
                        "day": day.replace("front", ""),
                        "file": os.path.basename(path),
                        "sim_id": sim.get("id"),
                        "task_id": (sim.get("info") or {}).get("task_id") or sim.get("task_id"),
                        "reward": rw,
                        "passed": (None if rw is None else bool(rw and rw >= 1.0)),
                        "turn_idx": idx,
                        "user_turn_k": k,
                        "n_user_turns": len(users),
                        "pos": pos,
                        "has_slot": bool(_BOLD.search(text)),
                        "bold_spans": _BOLD.findall(text),
                        "text": text,
                    })
    return pool


def stratify(pool, n, seed=0):
    """층화 표본: (day, passed, pos, has_slot) 셀에서 균등하게 라운드로빈."""
    cells = defaultdict(list)
    for r in pool:
        cells[(r["day"], r["passed"], r["pos"], r["has_slot"])].append(r)
    rng = random.Random(seed)
    for v in cells.values():
        rng.shuffle(v)
    keys = sorted(cells, key=lambda k: (-len(cells[k]), str(k)))
    out, i = [], 0
    while len(out) < n and any(cells[k] for k in keys):
        k = keys[i % len(keys)]
        if cells[k]:
            out.append(cells[k].pop())
        i += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(_SIM, "x8_sample_utterances.jsonl"))
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    pool = load_pool()
    print(f"모집단: 유저 발화 {len(pool)}건 "
          f"(sim {len({(r['file'], r['sim_id']) for r in pool})}개 · day {sorted({r['day'] for r in pool})})")
    for ax in ("day", "pos", "has_slot", "passed"):
        print(f"  {ax:9s} {dict(Counter(r[ax] for r in pool))}")

    samp = stratify(pool, args.n, args.seed)
    print(f"\n층화 표본 {len(samp)}건 (seed={args.seed}) — 셀 축=(day,passed,pos,has_slot)")
    for ax in ("day", "pos", "has_slot", "passed"):
        print(f"  {ax:9s} {dict(Counter(r[ax] for r in samp))}")
    print(f"  슬롯 스팬 총 {sum(len(r['bold_spans']) for r in samp)}개")

    for i, r in enumerate(samp):
        r["sample_id"] = f"u{i:03d}"
    with open(args.out, "w", encoding="utf-8") as f:
        for r in samp:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n[saved] {args.out}")
    print("⇒ 다음: 이 파일을 **전수 정독**해 gold 라벨을 별도 파일에 작성(모델 생성 금지·[[03b]]).")


if __name__ == "__main__":
    main()
