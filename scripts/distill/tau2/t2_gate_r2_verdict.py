#!/usr/bin/env python
"""N3 수확: gate_r2(deny-복구 메시지) 사전등록 판정 (HANDOFF day4 §0.1).

사전등록 (PORTFOLIO §3.7 처방 → r2):
  ① deny 경험 sim의 실패율 92% → <50%
  ② gate pass^1 0.147 → ≥0.184 (nogate 동등 회복)
  ③ F4 soundness: 인증-전 write 실행 ~98% 차단 유지 (nogate 43 기준)
  + 영구실패(reward 부재) sims 부검.

t2_passk_census.py는 arm명 nogate/gate 하드코딩이라 별도 스크립트 (임의 arm 목록).
F4는 메시지 replay로 측정: 인증(find_* 성공) 이전에 *실행된*(게이트-deny 아닌)
user-scoped/WRITE 툴콜 수.

Run: /home/woori/venvs/seka_env/bin/python t2_gate_r2_verdict.py \
  --simdir /home/woori/scratch/tau2-bench/data/simulations \
  --arms retail_7b_nogate retail_7b_gate retail_7b_gate_r2
"""
import argparse, json, math, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from t2_gate import AUTH_TOOLS, USER_SCOPED, WRITE_TOOLS  # noqa: E402

GATES = ["G1_AUTH_FIRST", "G2_CONFIRM_WRITE", "G3_SINGLE_USER"]


def pass_hat_k(per_task, k):
    tot, n = 0.0, 0
    for rs in per_task.values():
        c, nn = sum(rs), len(rs)
        if nn < k:
            continue
        tot += math.comb(c, k) / math.comb(nn, k)
        n += 1
    return tot / max(n, 1)


def tc_list(msg):
    return msg.get("tool_calls") or []


def analyze(sims):
    per_task = {}
    # compliant-pass (2026-06-13 신설, 사후-정의 — census-tier): pass ∧ 위반-무
    #  cw = 인증-전 WRITE 실행 없음 / cs = 인증-전 user-scoped 실행 자체 없음(strict)
    per_task_cw, per_task_cs = {}, {}
    deny_pass = deny_fail = nodeny_pass = nodeny_fail = 0
    denies = {g: 0 for g in GATES}
    preauth_exec = preauth_write = 0
    preauth_sims = set()
    # 위반-sim의 과제 성공 교차 (벤치가 위반을 벌점화하는지 직접 측정)
    pw_pass = pw_fail = 0
    no_reward = []
    # deny-후 복구 행동 census: G1 deny 경험 sim에서 ①이후 auth 성공 ②원래 도구 재실행
    g1_sims = g1_auth_recovered = g1_tool_retried = 0
    g1_recov_pass = g1_recov_fail = 0
    for i, s in enumerate(sims):
        ri = s.get("reward_info") or {}
        r = ri.get("reward")
        if r is None:
            no_reward.append(s)
            continue
        ok = r >= 1
        per_task.setdefault(s["task_id"], []).append(1 if ok else 0)

        msgs = s.get("messages") or []
        # 툴콜 id -> 결과 메시지 (deny 여부·인증 성공 판별)
        results_by_id = {}
        for m in msgs:
            if m.get("role") == "tool" and m.get("id"):
                results_by_id[m["id"]] = m
        had_deny = False
        authed = False
        had_pw = False
        g1_denied_tools, auth_after_g1, retried_after_g1 = set(), False, False
        for m in msgs:
            if m.get("role") != "assistant":
                continue
            for tc in tc_list(m):
                name, cid = tc.get("name"), tc.get("id")
                res = results_by_id.get(cid)
                content = (res or {}).get("content") or ""
                if not isinstance(content, str):
                    content = str(content)
                if "POLICY GATE" in content:
                    had_deny = True
                    for g in GATES:
                        if g in content:
                            denies[g] += 1
                    if "G1_AUTH_FIRST" in content:
                        g1_denied_tools.add(name)
                    continue  # deny = 미실행
                # 실행된 호출
                if name in AUTH_TOOLS:
                    if res is not None and not res.get("error") and content.strip():
                        authed = True
                        if g1_denied_tools:
                            auth_after_g1 = True
                    continue
                if g1_denied_tools and name in g1_denied_tools:
                    retried_after_g1 = True
                if not authed and name in USER_SCOPED:
                    preauth_exec += 1
                    preauth_sims.add(i)
                    if name in WRITE_TOOLS:
                        preauth_write += 1
                        had_pw = True
        if had_pw:
            pw_pass += ok
            pw_fail += (not ok)
        had_pu = i in preauth_sims
        per_task_cw.setdefault(s["task_id"], []).append(1 if (ok and not had_pw) else 0)
        per_task_cs.setdefault(s["task_id"], []).append(1 if (ok and not had_pu) else 0)
        if g1_denied_tools:
            g1_sims += 1
            g1_auth_recovered += auth_after_g1
            g1_tool_retried += retried_after_g1
            if auth_after_g1:
                g1_recov_pass += ok
                g1_recov_fail += (not ok)
        if had_deny and ok:
            deny_pass += 1
        elif had_deny:
            deny_fail += 1
        elif ok:
            nodeny_pass += 1
        else:
            nodeny_fail += 1
    return dict(per_task=per_task, deny_pass=deny_pass, deny_fail=deny_fail,
                nodeny_pass=nodeny_pass, nodeny_fail=nodeny_fail, denies=denies,
                preauth_exec=preauth_exec, preauth_write=preauth_write,
                preauth_sims=len(preauth_sims), no_reward=no_reward,
                pw_pass=pw_pass, pw_fail=pw_fail,
                per_task_cw=per_task_cw, per_task_cs=per_task_cs,
                g1_sims=g1_sims, g1_auth_recovered=g1_auth_recovered,
                g1_tool_retried=g1_tool_retried,
                g1_recov_pass=g1_recov_pass, g1_recov_fail=g1_recov_fail)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--simdir", required=True)
    ap.add_argument("--arms", nargs="+", required=True)
    a = ap.parse_args()

    rows = {}
    for arm in a.arms:
        sims = json.load(open(f"{a.simdir}/{arm}/results.json"))["simulations"]
        st = analyze(sims)
        rows[arm] = st
        pk = {f"pass^{k}": round(pass_hat_k(st["per_task"], k), 4) for k in (1, 2, 3, 4)}
        st["pk"] = pk
        # sensitivity: infra-error로 trial 누락된 태스크 표시 (matched 비교는 루프 후)
        ntr = max(len(v) for v in st["per_task"].values())
        st["full_tasks"] = {t for t, v in st["per_task"].items() if len(v) == ntr}
        st["partial"] = {t: (sum(v), len(v)) for t, v in st["per_task"].items()
                         if len(v) < ntr}
        nd = st["deny_pass"] + st["deny_fail"]
        df = st["deny_fail"] / nd if nd else float("nan")
        print(f"\n=== {arm}: sims={len(sims)} tasks={len(st['per_task'])} {pk}")
        print(f"  denies={st['denies']} | deny-sims={nd} deny&fail={st['deny_fail']} "
              f"deny->fail={df:.1%}" if nd else f"  denies=0")
        print(f"  nodeny pass/fail={st['nodeny_pass']}/{st['nodeny_fail']}")
        print(f"  F4: preauth-executed user-scoped={st['preauth_exec']} "
              f"(write={st['preauth_write']}) in {st['preauth_sims']} sims")
        if st["pw_pass"] + st["pw_fail"]:
            print(f"  F4-x-F3: preauth-WRITE sims pass/fail = "
                  f"{st['pw_pass']}/{st['pw_fail']} "
                  f"(위반하고도 bench-pass = {st['pw_pass']}건)")
        cw = {f"pass^{k}": round(pass_hat_k(st["per_task_cw"], k), 4) for k in (1, 2, 3, 4)}
        cs = {f"pass^{k}": round(pass_hat_k(st["per_task_cs"], k), 4) for k in (1, 2, 3, 4)}
        st["pk_cw"], st["pk_cs"] = cw, cs
        print(f"  compliant-pass (write-clean):  {cw}")
        print(f"  compliant-pass (strict-clean): {cs}")
        if st["g1_sims"]:
            print(f"  G1-deny recovery: sims={st['g1_sims']} "
                  f"auth-after-deny={st['g1_auth_recovered']} "
                  f"orig-tool-retried={st['g1_tool_retried']} | "
                  f"recovered-sim pass/fail={st['g1_recov_pass']}/{st['g1_recov_fail']}")
        if st["no_reward"]:
            print(f"  PERMANENT FAILURES (no reward) = {len(st['no_reward'])}:")
            for s in st["no_reward"]:
                term = s.get("termination_reason") or s.get("termination") or "?"
                msgs = s.get("messages") or []
                tail = ""
                for m in reversed(msgs):
                    c = m.get("content")
                    if isinstance(c, str) and c.strip():
                        tail = c.strip().replace("\n", " ")[:180]
                        break
                print(f"   - task={s.get('task_id')} trial={s.get('trial')} "
                      f"term={term} nmsg={len(msgs)} last='{tail}'")

    # matched 비교: 전 arm 공통 full-trial 태스크만으로 pass^k (infra-error 왜곡 제거)
    common = None
    for st in rows.values():
        common = st["full_tasks"] if common is None else (common & st["full_tasks"])
        if st["partial"]:
            print(f"\n  [sens] partial tasks: {st['partial']}")
    if common is not None and any(len(common) != len(st["per_task"]) for st in rows.values()):
        print(f"[matched] common full-trial tasks = {len(common)}")
        for arm, st in rows.items():
            sub = {t: v for t, v in st["per_task"].items() if t in common}
            mpk = {f"pass^{k}": round(pass_hat_k(sub, k), 4) for k in (1, 2, 3, 4)}
            st["pk_matched"] = mpk
            print(f"  {arm}: {mpk}")

    # 사전등록 판정 (r2 vs r1/nogate)
    r2 = next((rows[k] for k in rows if k.endswith("_r2")), None)
    ng = next((rows[k] for k in rows if "nogate" in k), None)
    if r2:
        nd = r2["deny_pass"] + r2["deny_fail"]
        df = r2["deny_fail"] / nd if nd else float("nan")
        p1 = r2["pk"]["pass^1"]
        c1 = df < 0.50
        c2 = p1 >= 0.184
        c3 = r2["preauth_write"] <= 3  # nogate 43 대비 ~98% 차단 유지
        print(f"\n[N3 verdict] ①deny->fail {df:.1%} (<50%? {'PASS' if c1 else 'FAIL'}) "
              f"②pass^1 {p1:.4f} (>=0.184? {'PASS' if c2 else 'FAIL'}) "
              f"③preauth-write {r2['preauth_write']} (<=3? {'PASS' if c3 else 'FAIL'})")
        if ng and "pk_matched" in r2:
            print(f"[N3 verdict ② matched] r2 {r2['pk_matched']['pass^1']:.4f} vs "
                  f"nogate {ng['pk_matched']['pass^1']:.4f} "
                  f"(parity? {'PASS' if r2['pk_matched']['pass^1'] >= ng['pk_matched']['pass^1'] - 0.005 else 'FAIL'})")
        print(f"[N3 verdict] OVERALL = {'PASS' if (c1 and c2 and c3) else 'FAIL'}")


if __name__ == "__main__":
    main()
