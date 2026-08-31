# -*- coding: utf-8 -*-
"""X600 — 기존 retail 전수 측정치 집계 (2026-08-29·사용자 지시 *"히스토리 뒤져라"*).

왜: Qwen3.8-27B 기준으로 갈아타기 전에 **우리가 이미 가진 retail 수치**가 무엇인지 원자료에서
    직접 센다. 문서 인용은 소급 정정을 놓친다(오늘 특허 감사에서 철회된 수치가 세 문서에
    살아 있었다). 채점 기준은 `reward`([[69]]).

무엇을 내나 (파일마다):
    agent 모델 · user-sim 모델 · num_trials · retrieval_config · git_commit
    태스크 수 · sim 수 · **pass^1**(전 sim 평균 성공) · pass^k(모든 k 시행 통과 태스크 비율)

⚠pass^1 과 pass^k 는 다른 것이다 — 리더보드 헤드라인은 pass^1 이고, 그것은 *다수 시행 평균*이다.
"""
import glob
import gzip
import io
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SIMS = os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026", "sim_results")


def load(p):
    with gzip.open(p, "rt", encoding="utf-8") as f:
        return json.load(f)


def model_of(info, key):
    v = (info or {}).get(key) or {}
    if isinstance(v, dict):
        for k in ("llm", "model", "llm_agent", "llm_user", "name"):
            if v.get(k):
                return str(v[k])
        return json.dumps(v, ensure_ascii=False)[:60]
    return str(v)[:60]


def tally(path):
    d = load(path)
    info = d.get("info") or {}
    sims = d.get("simulations") or []
    by_task = {}
    for s in sims:
        ri = s.get("reward_info") or {}
        r = ri.get("reward")
        if r is None:
            continue
        by_task.setdefault(s.get("task_id"), []).append(float(r))
    if not by_task:
        return None
    allr = [r for v in by_task.values() for r in v]
    nt = int(info.get("num_trials") or max(len(v) for v in by_task.values()))
    row = {
        "file": os.path.basename(path),
        "agent": model_of(info, "agent_info"),
        "user": model_of(info, "user_info"),
        "nt": nt,
        "retrieval": info.get("retrieval_config"),
        "sha": (info.get("git_commit") or "")[:8],
        "tasks": len(by_task),
        "sims": len(allr),
        "pass1": round(sum(1 for r in allr if r >= 0.999) / float(len(allr)), 4),
        "mean_reward": round(sum(allr) / float(len(allr)), 4),
    }
    for k in (2, 3, 4):
        elig = [v for v in by_task.values() if len(v) >= k]
        if elig:
            row["pass%d" % k] = round(
                sum(1 for v in elig if all(r >= 0.999 for r in v[:k])) / float(len(elig)), 4)
    return row


def main():
    paths = sorted(glob.glob(os.path.join(SIMS, "*retail*.results.json.gz")))
    rows = []
    for p in paths:
        try:
            r = tally(p)
        except Exception as e:
            print("  [skip] %s: %r" % (os.path.basename(p), e))
            continue
        if r:
            rows.append(r)
    rows.sort(key=lambda r: -r["pass1"])
    print("retail 전수 측정 %d건 (원자료 집계·reward 기준)" % len(rows))
    print("%-44s %5s %5s %6s %7s %7s %7s %7s  %s"
          % ("file", "task", "sims", "nt", "pass^1", "pass^2", "pass^3", "pass^4", "retrieval"))
    for r in rows:
        print("%-44s %5d %5d %6d %7.3f %7s %7s %7s  %s"
              % (r["file"][:44], r["tasks"], r["sims"], r["nt"], r["pass1"],
                 r.get("pass2", "-"), r.get("pass3", "-"), r.get("pass4", "-"),
                 r["retrieval"]))
    print()
    for r in rows:
        print("%-44s agent=%s" % (r["file"][:44], r["agent"]))
        print("%-44s user =%s  sha=%s" % ("", r["user"], r["sha"]))


if __name__ == "__main__":
    main()
