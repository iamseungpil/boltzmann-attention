# -*- coding: utf-8 -*-
r"""x647 - **도는 런을 중간에 읽고, 멈춘 자리에서 이어 붙인다** (유료 0 · 런에 영향 0).

## 왜 (사용자 지시 2026-08-30)
*"중간에 멈추고 resume 하거나, 중간에 포렌식할 수 있게 하라.
  **너무 오랜 시간 끝날 때까지 기다리지 않게** 하라."*

`tau2 run` 은 `results.json` 을 **sim 하나 끝날 때마다** 갱신한다(실측: 2분 만에 1건 기록).
그래서 런이 도는 중에도 전부 읽을 수 있다 - 이 도구는 **읽기만** 하고 런을 건드리지 않는다.

## 모드

    --status     진행·pass·ETA·태스크별 한 줄            (기본)
    --forensic   실패 sim 의 per-step 요약 (도구 순서·변이·종료사유)
    --remaining  아직 안 돈 태스크 id 를 공백 구분으로 출력 -> resume 용
    --resume-cmd 이어서 돌릴 `tau2 run` 명령을 그대로 찍는다 (복사해 실행)

사용: PYTHONPATH=. python x647_run_watch.py --tag <SAVE_TO> [--status|--forensic|--remaining]
"""
import argparse
import collections
import io
import json
import os
import statistics
import sys

SIMDIR = "/home/woori/iso_tau3/tau2-bench/data/simulations"
KB = {"KB_search_bm25", "KB_search_dense", "KB_search", "shell"}
READ_HINT = ("get_", "find_", "list_", "KB_search", "shell", "calculate", "check_")


def load(tag):
    p = os.path.join(SIMDIR, tag, "results.json")
    if not os.path.exists(p):
        return None, p
    for _ in range(3):                     # 쓰는 중이면 한 번 더 시도
        try:
            return json.load(io.open(p, encoding="utf-8")), p
        except Exception:
            pass
    return None, p


def all_task_ids():
    sys.path.insert(0, "/home/woori/iso_tau3/tau2-bench/src")
    import contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        from tau2.registry import registry
        tasks = registry.get_tasks_loader("banking_knowledge")()
    return [str(getattr(t, "id", None) or (t.get("id") if isinstance(t, dict) else None))
            for t in tasks]


def status(d, tag, planned):
    sims = d.get("simulations") or []
    done = [s for s in sims if (s.get("reward_info") or {}).get("reward") is not None]
    ps = [s for s in done if (s.get("reward_info") or {}).get("reward") == 1.0]
    durs = [s.get("duration") or 0 for s in done]
    print("=== %s ===" % tag)
    print("  완료 %d / 계획 %s · pass %d · **reward 평균 %.4f**"
          % (len(done), planned or "?", len(ps),
             sum(((s.get("reward_info") or {}).get("reward") or 0) for s in done) / max(len(done), 1)))
    if durs:
        med = statistics.median(durs)
        print("  소요: 중앙 %.1f분 · 평균 %.1f분 · 누적 %.1f시간"
              % (med / 60, statistics.mean(durs) / 60, sum(durs) / 3600))
        if planned:
            left = int(planned) - len(done)
            print("  남은 %d개 · **ETA 약 %.1f시간** (중앙값 기준)" % (left, left * med / 3600))
    print("  종료사유: %s" % dict(collections.Counter(str(s.get("termination_reason")) for s in done)))
    print()
    print("  %-10s %-7s %-8s %-7s %s" % ("task", "reward", "분", "메시지", "종료"))
    for s in sorted(done, key=lambda x: str(x.get("task_id"))):
        print("  %-10s %-7s %-8.1f %-7d %s"
              % (s.get("task_id"), (s.get("reward_info") or {}).get("reward"),
                 (s.get("duration") or 0) / 60, len(s.get("messages") or []),
                 s.get("termination_reason")))


def forensic(d, tag):
    sims = [s for s in (d.get("simulations") or [])
            if (s.get("reward_info") or {}).get("reward") == 0.0]
    print("=== %s · 실패 %d건 per-step 요약 ===" % (tag, len(sims)))
    for s in sorted(sims, key=lambda x: str(x.get("task_id"))):
        msgs = s.get("messages") or []
        seq, writes, errs = [], [], 0
        for i, m in enumerate(msgs):
            for tc in (m.get("tool_calls") or []) or []:
                n = tc.get("name") or (tc.get("function") or {}).get("name") or "?"
                a = tc.get("arguments")
                if isinstance(a, str):
                    try:
                        a = json.loads(a)
                    except Exception:
                        a = {}
                eff = str((a or {}).get("agent_tool_name") or n)
                seq.append("S" if eff == "shell" else ("K" if "KB_search" in eff else eff))
                if not any(eff.startswith(x) or x in eff for x in READ_HINT):
                    writes.append(eff)
                    for mj in msgs[i + 1:]:
                        if mj.get("role") == "tool" and mj.get("id") == tc.get("id"):
                            b = str(mj.get("content") or "")
                            if mj.get("error") or b.lstrip().startswith("Error"):
                                errs += 1
                            break
        srch = sum(1 for x in seq if x in ("S", "K"))
        print("  %-10s 메시지 %3d · 검색 %2d · 변이 %2d · env거부 %d · 종료 %s"
              % (s.get("task_id"), len(msgs), srch, len(writes), errs, s.get("termination_reason")))
        print("      순서: %s" % " ".join(seq[:22]))
        if writes:
            print("      변이: %s" % ", ".join(sorted(set(writes))[:5]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--planned", type=int, default=None)
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--forensic", action="store_true")
    ap.add_argument("--remaining", action="store_true")
    ap.add_argument("--resume-cmd", action="store_true")
    ap.add_argument("--port", type=int, default=8143)
    a = ap.parse_args()

    d, p = load(a.tag)
    if d is None:
        print("결과 없음: %s" % p)
        return
    done_ids = {str(s.get("task_id")) for s in (d.get("simulations") or [])
                if (s.get("reward_info") or {}).get("reward") is not None}

    if a.remaining or a.resume_cmd:
        rest = [t for t in all_task_ids() if t not in done_ids]
        if a.remaining:
            print(" ".join(rest))
            return
        print("# 완료 %d · 남은 %d — 아래를 그대로 실행하면 이어서 돈다" % (len(done_ids), len(rest)))
        print("cd /home/woori/iso_tau3/tau2-bench && /home/woori/iso_tau3/venv/bin/tau2 run \\")
        print("  --domain banking_knowledge --retrieval-config alltools \\")
        print("  --agent-llm 'openai/Qwen/Qwen3.8-27B-FP8' \\")
        print("  --agent-llm-args '{\"api_base\":\"http://localhost:%d/v1\",\"api_key\":\"dummy\","
              "\"temperature\":0.0,\"timeout\":900.0,\"num_retries\":2}' \\" % a.port)
        print("  --user-llm 'openrouter/openai/gpt-5.2' \\")
        print("  --user-llm-args '{\"temperature\":0.0,\"reasoning_effort\":\"low\","
              "\"timeout\":900.0,\"num_retries\":2}' \\")
        print("  --num-trials 1 --max-concurrency 1 --max-steps 200 --seed 300 --timeout 7200 \\")
        print("  --task-ids %s \\" % " ".join(rest))
        print("  --save-to %s_resume" % a.tag)
        print()
        print("# ⚠`--save-to` 를 바꿔야 기존 결과를 덮지 않는다. 집계는 두 파일을 합쳐서 낸다.")
        return

    if a.forensic:
        forensic(d, a.tag)
    else:
        status(d, a.tag, a.planned)


main()
