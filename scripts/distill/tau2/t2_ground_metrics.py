#!/usr/bin/env python
"""§7 조건부 grounding 지표 — ground_OK를 3원인(spec-fail / C0=resolve-미emit / P2b=producer-미호출)으로 분해.

bare ground_OK는 오진(A2_GROUNDING_WIRING_DESIGN §7·위험2). 이 스크립트가:
  분모율  P(resolve emitted)            = resolve_selection emit한 task / 전체 task   ← C0
          P(producer called | emitted)  = producer 호출된 emission / 전체 emission     ← P2b
  ★조건부  P(ground_OK | emitted ∧ called) = ground_OK / (producer_present emission)   ← spec 순수품질
  routed 분해(ok/fetch/ask_refine)·도메인별.

입력 = T2_GROUND_LOG(JSONL·매 emission) + results 디렉토리(전체 task 수). 후자 없으면 emit율은 call-level만.

Run: py -3 t2_ground_metrics.py --log <ground.jsonl> [--results data/simulations/<save_to>]
"""
import argparse
import json
import os
from collections import Counter


def _load_log(path):
    evs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    evs.append(json.loads(line))
                except Exception:
                    pass
    return evs


def _n_tasks(results_dir):
    """results.json서 전체 (task,trial) 시뮬 수 — emit율 분모."""
    if not results_dir:
        return None
    rj = os.path.join(results_dir, "results.json")
    if not os.path.exists(rj):
        return None
    try:
        with open(rj, encoding="utf-8") as f:
            d = json.load(f)
        sims = d.get("simulations", d if isinstance(d, list) else [])
        return len(sims) if isinstance(sims, list) else None
    except Exception:
        return None


def _rate(num, den):
    return f"{num}/{den} = {num/den:.3f}" if den else f"{num}/0 = n/a"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--results", default=None, help="data/simulations/<save_to> (전체 task 분모)")
    a = ap.parse_args()

    evs = _load_log(a.log)
    if not evs:
        print("[metrics] 이벤트 0 — resolve_selection emission이 전혀 없음 (=C0 지배 신호: 모델이 op-naming 안 함).")
        return

    by_domain = Counter(e.get("domain") for e in evs)
    n_emit = len(evs)                                            # call-level emissions
    n_present = sum(1 for e in evs if e.get("producer_present"))  # 후보 컨테이너 trace에 존재
    n_called = sum(1 for e in evs if e.get("producer_called"))    # producer 도구 실제 호출
    n_ground = sum(1 for e in evs if e.get("ground_OK"))
    routed = Counter(e.get("routed") for e in evs)
    tasks_emitted = len({e.get("task_id") for e in evs if e.get("task_id") is not None})
    n_tasks = _n_tasks(a.results)

    print(f"=== §7 조건부 grounding 지표 ({a.log}) ===")
    print(f"도메인 분포: {dict(by_domain)}")
    print(f"resolve emission(call-level): {n_emit}  ·  emit한 task: {tasks_emitted}")
    print()
    print("── 분모율 (왜 낮은가의 3원인) ──")
    if n_tasks:
        print(f"  C0  P(resolve emitted, task-level)      = {_rate(tasks_emitted, n_tasks)}   "
              f"← 낮으면 모델이 resolve_selection 안 emit(학습/프롬프트·wiring 무관)")
    else:
        print(f"  C0  (task-level emit율: --results 미지정 — call-level emission={n_emit}만)")
    print(f"  P2b P(producer present | emitted)       = {_rate(n_present, n_emit)}   "
          f"← 낮으면 상류 gather 미호출(→ §5a fetch 라우팅)")
    print(f"      P(producer CALLED tool | emitted)   = {_rate(n_called, n_emit)}   (직접 신호)")
    print()
    print("── ★조건부 (spec 순수 품질·이게 wiring 검증 핵심) ──")
    print(f"  P(ground_OK | emitted ∧ present)        = {_rate(n_ground, n_present)}")
    print(f"  (참고) bare P(ground_OK | emitted)       = {_rate(n_ground, n_emit)}   ← 교란된 수치·헤드라인 금지")
    print()
    print(f"── routed 분해 ──  {dict(routed)}")
    print("   ok=resolve 성공 · fetch=후보부재→producer 호출 유도(P2b) · ask_refine=후보있으나 비유일/anchor불명")
    print()
    # 진단 한 줄
    if n_tasks and tasks_emitted / max(n_tasks, 1) < 0.2:
        print("[진단] C0 지배 — 병목은 wiring 아니라 resolve_selection emit율(모델 op-naming). 조건부는 깨끗해도 별 트랙.")
    elif n_present and n_ground / n_present < 0.5:
        print("[진단] spec-fail 지배 — 조건부 낮음. grounding-spec/포맷 결함 (진짜 NO-GO 후보).")
    elif n_emit and n_present / n_emit < 0.5:
        print("[진단] P2b 지배 — producer 미호출. §5a fetch-우선 라우팅이 다음 턴 fetch 유도하는지 확인.")
    else:
        print("[진단] 조건부 ground_OK 양호 — decidable 부분을 engine+A2가 결정론 처리 (GO 방향).")


if __name__ == "__main__":
    main()
