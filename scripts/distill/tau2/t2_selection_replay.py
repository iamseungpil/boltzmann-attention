#!/usr/bin/env python
"""선택-formalize raw-context replay eval (A안·학습 0·user-sim 0·크기 비교용).

실 τ² e2e sim에서 **각 resolve_selection 직전 컨텍스트**(task 발화 + raw fetch 출력=실제 카탈로그)를
뽑아, *대상 모델*에 같은 컨텍스트로 resolve_selection을 강제 emit시킨다. 그 emit을 새 grounding-spec
(_ground·관계대수 투영) + autopsy(_classify)로 채점 → **선택-formalize 품질을 크기별로 깨끗이**.

왜 기존 tau2_op_eval이 아니라 이것: op-eval은 *깨끗한 속성 리스트를 떠먹여* wrong-key 잔여(검색키≠선택키)를
못 봄. 여기선 **raw 카탈로그**(실 e2e와 동일)를 줘서 모델이 스스로 속성을 골라야 함 = 잔여 충실 재현.
user-sim 변동 0(단발 replay)·동일 컨텍스트·크기만 변화 = 신뢰 scale 곡선.

지표: emit율 P(resolve emitted) + ground_OK율 + autopsy 분류(UNIQUE_OK/KEY_MISMATCH/UNDER_DET/VALUE/NO_CATALOG).
정직: 이건 **선택-formalize 성분 정확도**(카탈로그 주어진·단발)지 τ² task 성공률 아님(혼동 금지).

Run: PY t2_selection_replay.py --sim data/simulations/<dir> --spec a2/<domain>.grounding.json \
       --model <served> --base http://localhost:PORT/v1 [--out x.json]
"""
import argparse
import json
import os
import sys
from collections import Counter

_HERE = os.path.dirname(os.path.abspath(__file__))
_MA = os.path.join(_HERE, "..", "ma")
sys.path.insert(0, _HERE)
sys.path.insert(0, _MA)
from t2_resolve_patch import _ground               # noqa: E402  spec 구동 투영(관계대수)
from t2_ground_autopsy import _classify, _outs_before  # noqa: E402  5분류 + prefix 파싱
from tau2_op_resolver import resolve_op_tau2        # noqa: E402
from synth_native_eval import chat_toolcall         # noqa: E402  tool_call 강제 호출
from synth_to_nativefc import RESOLVE_SELECTION_SCHEMA, SYSTEM  # noqa: E402


def _ctx_text(prefix):
    """resolve 직전 prefix → (user 발화 모음, raw tool 출력 모음) 텍스트. raw 카탈로그 보존."""
    users, tools = [], []
    for m in prefix:
        r = m.get("role")
        if r == "user" and isinstance(m.get("content"), str):
            users.append(m["content"])
        elif r == "tool" and not m.get("error") and isinstance(m.get("content"), str):
            tools.append(m["content"])
    convo = "\n".join(users[-12:])                 # 최근 user 발화(task 점진공개)
    data = "\n".join(tools[-8:])                    # 최근 fetch 출력(raw 카탈로그)
    return convo, data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", required=True, help="컨텍스트 소스 e2e sim 디렉토리")
    ap.add_argument("--spec", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--base", default="http://localhost:8351/v1")
    ap.add_argument("--drop", action="store_true", help="navigation-key drop arm(autopsy --drop과 동일)")
    ap.add_argument("--clarify", action="store_true",
                    help="#6 instruction-clarity: 검색키≠선택키를 도메인-일반으로 강하게 명시(학습 0)")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    # #6 arm: 도메인-일반 명시(retail/airline 특정 안 함·'후보 간 차이나는 속성' vs '검색/카테고리어')
    CLARIFY = ("\n\nIMPORTANT — name ONLY attributes that DIFFER between the candidate items in the "
               "fetched data (the per-item option fields, e.g. color/size/cabin/price). Do NOT use the "
               "search/query parameters or category labels you used to FIND the candidates (such as the "
               "product type/category, or the origin/destination/date you searched) — those already "
               "located the candidate set and are NOT selection attributes. Pick the ONE item by its "
               "differentiating attributes." if a.clarify else "")
    with open(a.spec, encoding="utf-8") as f:
        spec = json.load(f)
    cs = spec["candidate_source"]
    with open(os.path.join(a.sim, "results.json"), encoding="utf-8") as f:
        sims = json.load(f).get("simulations", [])

    n_cases = n_emit = n_ground = 0
    cls = Counter()
    rows = []
    for s in sims:
        msgs = s.get("messages", [])
        for i, m in enumerate(msgs):
            if m.get("role") != "assistant":
                continue
            for tc in (m.get("tool_calls") or []):
                if tc.get("name") != "resolve_selection":
                    continue
                n_cases += 1
                prefix = msgs[:i]
                convo, data = _ctx_text(prefix)
                user = (f"Conversation so far:\n{convo}\n\n"
                        f"Data you have already fetched (the actual catalog/records):\n{data}\n\n"
                        f"Now call resolve_selection to identify the exact item the user wants, "
                        f"naming it by its selectable attributes (from the fetched data).{CLARIFY}")
                emit, _content = chat_toolcall(a.base, a.model, [RESOLVE_SELECTION_SCHEMA],
                                               [{"role": "system", "content": SYSTEM},
                                                {"role": "user", "content": user}])
                args = (emit or {}).get("args") if emit else None
                if not emit or args is None:
                    cls["NO_EMIT"] += 1                      # C0: 모델이 resolve_selection 안 부름
                    rows.append({"emit": False})
                    continue
                n_emit += 1
                outs = _outs_before(msgs, i)                 # 동일 prefix의 raw 출력으로 채점
                wk = set((args.get("among") or {}).keys())
                if isinstance(args.get("attr"), str):
                    wk.add(args["attr"])
                cat, anchor, present = _ground(outs, spec, want_keys=wk)
                rid = resolve_op_tau2(args, cat, anchor_id=anchor) if cat else None
                if rid:
                    n_ground += 1
                c, _info = _classify(args, cat, anchor, present, cs, drop=a.drop)
                cls[c] += 1
                rows.append({"emit": True, "args": {k: args.get(k) for k in ("op", "attr", "among", "set")},
                             "rid": rid, "class": c})

    print(f"model={a.model} sim={os.path.basename(a.sim)} drop={a.drop} | resolve-points={n_cases}")
    print(f"  P(resolve emitted)        = {n_emit}/{n_cases} = {n_emit/max(n_cases,1):.3f}")
    print(f"  P(ground_OK | emitted)    = {n_ground}/{max(n_emit,1)} = {n_ground/max(n_emit,1):.3f}  ← 선택-formalize 품질")
    print(f"  autopsy class: {dict(cls.most_common())}")
    if a.out:
        json.dump({"model": a.model, "sim": a.sim, "drop": a.drop,
                   "n_cases": n_cases, "n_emit": n_emit, "n_ground": n_ground,
                   "cls": dict(cls), "rows": rows},
                  open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print("wrote", a.out)


if __name__ == "__main__":
    main()
