#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""fexec_iso_probe.py — FORMALIZE-EXEC V0 격리 재현 (NEXT_LEVER_GEN §2.6·무료).

A′/COMP 궤적의 C클래스 결정점(t20·t37·t79) 재생 — **형식화 정확도 EM 선측정**:
  결정점 = confirm-write 도구의 변형-선택 인자(P-B fire 채널 동형: hints 매칭·
  값 문맥-실재·후보 record ≥2). 그 시점까지의 전사+후보 record로 형식화 프롬프트를
  구성(라이브 build_formalize_prompt 그대로) → 32B 서브콜 → per-결정점 gold-기준과
  EM 대조(op/field/constraints 분해 채점).

★V0 게이트: EM 불통과 = 스택 미편입(C23 재발 방지). 서버(기본 http://localhost:8140/v1)
  불가 시 `--dry` — 프롬프트+기대JSON 케이스 파일 덤프만(실측은 서버 가용 시).

usage:
  py -3 fexec_iso_probe.py --results <gz> --tasks 20,37,79 --dry --out fexec_v0_cases.jsonl
  py -3 fexec_iso_probe.py --results <gz> --tasks 20 --base http://localhost:8140/v1 \
        --model Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8
"""
import argparse, gzip, json, os, sys, types

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import t2_formalize_exec as FX  # noqa: E402
from t2_gate_patch import (  # noqa: E402
    _domain_a2, _confirm_write_tools, _grounded_candidates, _parse_tool_outputs,
    _iter_scalars, _key_tokens, _sig, _min_enclosing_record)

# per-결정점 gold-기준 (설계 §2.2 표 — op/field/제약 field 집합. 값은 궤적-의존이라
# constraints는 field-집합 EM으로 채점·value는 참고 출력).
GOLD = {
    "20": {"op": "argmax", "field": "price", "cons_fields": {"size", "available"},
           "note": "주문과 같은 size(9) 중 최고가 신발 — CALC-EXT 정적 spec 불가 칸"},
    "37": {"op": "filter", "field": None, "cons_fields": {"price"},
           "note": "예산 이하 조합 — 실행기는 필터 재료까지·조합 선택은 에이전트"},
    "79": {"op": "filter", "field": None, "cons_fields": {"color"},
           "note": "다른 레코드와 attr-match(같은 색) — cross-record 제약"},
}


def load_sims(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt", encoding="utf-8") as f:
        return json.load(f)["simulations"]


def _obj_msgs(msgs):
    """gz dict 메시지 → attr 객체(라이브 헬퍼 getattr-호환)."""
    out = []
    for m in msgs:
        tcs = []
        for tc in (m.get("tool_calls") or []):
            ar = tc.get("arguments")
            if isinstance(ar, str):
                try:
                    ar = json.loads(ar)
                except Exception:
                    ar = {}
            tcs.append(types.SimpleNamespace(id=tc.get("id"), name=tc.get("name"),
                                             arguments=ar, requestor="assistant"))
        out.append(types.SimpleNamespace(role=m.get("role"), content=m.get("content"),
                                         error=bool(m.get("error")), id=m.get("id"),
                                         tool_calls=tcs or None))
    return out


def _ctx_text(msgs):
    parts = [str(m.content) for m in msgs
             if m.role in ("user", "tool") and m.content is not None]
    return " ".join(parts).lower()


def _wide_candidates(arg_key, fab_value, msgs, limit=24):
    """probe-전용 광폭 후보: key-매칭 ∪ sig-매칭 (엔진 `key_cands or sig_cands` 게이트 해제).
    ★근거(2026-07-11 실측): product.variants가 dict-key id라 첫 조우가 enclosing key
    'variants'로 sig_cands에 선점(shadowing)→ 엔진 후보-표면에서 variant id 전량 탈락
    = t20형 결정점에 live P-B 좌석이 구조적으로 안 열림(보고 항목). 엔진은 불변(회귀 회피)."""
    toks = _key_tokens(arg_key)
    want = _sig(fab_value)
    seen, key_c, sig_c = set(), [], []
    for out in _parse_tool_outputs(msgs, lenient=True):
        if not isinstance(out, (dict, list)):
            continue
        for kk, vv in _iter_scalars(out):
            s = str(vv).strip()
            if len(s) < 4 or s in seen:
                continue
            kl = str(kk or "").lower()
            if any(t in kl or kl in t for t in toks):
                seen.add(s)
                key_c.append(s)
            elif want != "other" and _sig(s) == want:
                seen.add(s)
                sig_c.append(s)
    return (key_c + sig_c)[:limit]


def _records_for(cands, msgs, limit=16):
    """후보 → (cand, 최소 enclosing record dict) — FX._candidate_record_dicts 동형(광폭판)."""
    outs = _parse_tool_outputs(msgs, lenient=True)
    recs = []
    for c in cands[:limit]:
        rec = None
        for out in outs:
            if isinstance(out, (dict, list)):
                r = _min_enclosing_record(out, str(c).strip())
                if r is not None:
                    rec = r
                    break
        recs.append((c, rec))
    return recs


def _rec_name(rec):
    """record의 표시명(있으면) — 다품목 write에서 어느 결정인지 식별용 메타.
    variant record(name 부재)는 options 요약으로 폴백."""
    if isinstance(rec, dict):
        for k in ("name", "product_name", "title"):
            if isinstance(rec.get(k), str):
                return rec[k]
        if isinstance(rec.get("options"), dict):
            return json.dumps(rec["options"], ensure_ascii=False, default=str)[:80]
    return None


def decision_points(sim, write_tools, hints=("item",)):
    """궤적 → [(msgs_upto, arg_key, cur_value, records, fires_live, rec_name)].
    EM-케이스 = 광폭 후보(_wide_candidates) 멤버십 — 형식화 정확도 측정이 목적.
    다품목 write(t20=4치환)는 값마다 결정점 1건(어느 값이 gold-기준[신발] 결정인지는
    rec_name 메타로 식별). fires_live = live P-B fire 조건 동형(엔진 후보-표면·limit=8)."""
    msgs = _obj_msgs(sim.get("messages") or [])
    pts = []
    seen_vals = set()
    for i, m in enumerate(msgs):
        if m.role != "assistant" or not m.tool_calls:
            continue
        upto = msgs[:i]
        ctx = _ctx_text(upto)
        for tc in m.tool_calls:
            if tc.name not in write_tools:
                continue
            for k, v in (tc.arguments or {}).items():
                if not any(h in k.lower() for h in hints):
                    continue
                vals = v if isinstance(v, list) else [v]
                for val in vals:
                    s = str(val).strip()
                    if len(s) < 4 or s.lower() not in ctx or (k, s) in seen_vals:
                        continue
                    wide = _wide_candidates(k, s, upto)
                    if len(wide) < 2 or not any(s.lower() == str(c).lower() for c in wide):
                        continue
                    live = _grounded_candidates(k, s, upto, lenient=True)
                    fires_live = len(live) >= 2 and any(s.lower() == str(c).lower() for c in live)
                    records = _records_for(wide, upto)
                    if sum(1 for _, r in records if isinstance(r, dict)) >= 2:
                        seen_vals.add((k, s))
                        cur_rec = next((r for c, r in records if str(c) == s), None)
                        pts.append((upto, k, s, records, fires_live, _rec_name(cur_rec)))
    return pts


def em_score(spec, gold):
    """분해 EM: op / field / constraints field-집합."""
    if spec is None:
        return {"op": False, "field": False, "cons": False, "em": False}
    op_ok = spec["op"] == gold["op"]
    f_ok = (spec.get("field") or None) == gold["field"] or \
        (gold["field"] is None and not spec.get("field"))
    got_cf = {str(c["field"]).strip().lower() for c in spec.get("constraints") or []}
    cons_ok = got_cf == {f.lower() for f in gold["cons_fields"]}
    return {"op": op_ok, "field": f_ok, "cons": cons_ok, "em": op_ok and f_ok and cons_ok}


def call_server(base, model, prompt, timeout=180):
    import urllib.request
    body = json.dumps({"model": model, "temperature": 0.0, "max_tokens": 400,
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(base.rstrip("/") + "/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    out = json.load(urllib.request.urlopen(req, timeout=timeout))
    return out["choices"][0]["message"]["content"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--tasks", default="20,37,79",
                    help="task id 목록 또는 'all'(★v1.1 §2.6: 전-sim 수확 — 희소 타입 "
                         "n≥30 보완 + 통과-sim over-fire 계측 동시 획득)")
    ap.add_argument("--domain", default="retail")
    ap.add_argument("--dry", action="store_true", help="서버 없이 케이스 덤프만")
    ap.add_argument("--out", default=None, help="--dry 케이스 jsonl 출력 경로")
    ap.add_argument("--base", default="http://localhost:8140/v1")
    ap.add_argument("--model", default="Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8")
    a = ap.parse_args()
    sims = load_sims(a.results)
    a2 = _domain_a2(a.domain)
    wt = _confirm_write_tools(a2)
    all_tasks = a.tasks.strip().lower() == "all"
    tasks = None if all_tasks else [t.strip() for t in a.tasks.split(",") if t.strip()]
    cases = []
    for s in sims:
        tid = str(s.get("task_id"))
        if not all_tasks and tid not in tasks:
            continue
        rew = (s.get("reward_info") or {}).get("reward")
        for upto, k, cur, records, fires_live, rname in decision_points(s, wt):
            prompt = FX.build_formalize_prompt(upto, k, cur, records)
            cases.append({"task": tid, "trial": s.get("trial"), "reward": rew,
                          "arg": k, "cur": cur,
                          "n_records": len(records), "fires_live": fires_live,
                          "record_name": rname, "gold": GOLD.get(tid), "prompt": prompt})
    print("== FORMALIZE-EXEC V0: %d decision point(s) (tasks %s) =="
          % (len(cases), "ALL(456·v1.1 §2.6)" if all_tasks else tasks))
    if not all_tasks:
        for c in cases:
            print("  t%s tr%s arg=%s cur=%s (%s) records=%d fires_live=%s gold_op=%s"
                  % (c["task"], c["trial"], c["arg"], c["cur"], c["record_name"],
                     c["n_records"], c["fires_live"], (c["gold"] or {}).get("op")))
    else:
        npass = sum(1 for c in cases if (c["reward"] or 0) >= 1)
        print("  passing-sim 결정점(over-fire 계측분)=%d · failing-sim 결정점=%d"
              % (npass, len(cases) - npass))
    nl = sum(1 for c in cases if c["fires_live"])
    print("  live P-B 좌석 커버리지: %d/%d (미커버=variants dict-key shadowing — 보고 항목)"
          % (nl, len(cases)))
    if a.dry:
        if a.out:
            with open(a.out, "w", encoding="utf-8") as f:
                for c in cases:
                    f.write(json.dumps(c, ensure_ascii=False, default=list) + "\n")
            print("[dry] %d case(s) → %s (32B 서버 가용 시 --dry 없이 재실행=EM 실측)" % (len(cases), a.out))
        return
    # 실측 (서버 필요)
    agg = {"n": 0, "op": 0, "field": 0, "cons": 0, "em": 0}
    for c in cases:
        try:
            ans = call_server(a.base, a.model, c["prompt"])
        except Exception as e:
            print("  t%s tr%s: SERVER_ERR %s" % (c["task"], c["trial"], type(e).__name__))
            continue
        spec = FX.parse_formalize(ans)
        sc = em_score(spec, c["gold"]) if c["gold"] else {}
        agg["n"] += 1
        for kk in ("op", "field", "cons", "em"):
            agg[kk] += bool(sc.get(kk))
        print("  t%s tr%s arg=%s: spec=%s | EM %s"
              % (c["task"], c["trial"], c["arg"],
                 json.dumps(spec, ensure_ascii=False, default=str) if spec else "UNSURE", sc))
    if agg["n"]:
        print("\nEM 집계: n=%d op=%.2f field=%.2f cons=%.2f full-EM=%.2f"
              % (agg["n"], agg["op"] / agg["n"], agg["field"] / agg["n"],
                 agg["cons"] / agg["n"], agg["em"] / agg["n"]))
        print("V0 게이트: full-EM 낮으면 스택 미편입(C23 재발 방지·§2.6)")


if __name__ == "__main__":
    main()
