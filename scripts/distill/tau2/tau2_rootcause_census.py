#!/usr/bin/env python
"""τ² 전수 궤적 root-cause census (2026-06-15) — P7 vs P2b 분별.

autopsy(증상 분류)보다 깊게: 각 궤적의 *첫 실패 이벤트*와 *그 후 행동*을 추출해
"agent_collapse 루프의 trigger"를 정밀 분별:
  - fab_then_loop  : 날조 인자→error→동일호출 반복      = P2b(root) + P7(증폭)
  - fab_then_switch: 날조 인자→error→다른시도(복구 시도) = P2b(root)·P7 부분작동
  - gate_then_loop : 정책/confirm gate→동일호출 반복     = P7(root·날조 무관)
  - gate_then_switch: gate→다른시도                      = 정상 복구 방향
  - legit_then_loop: 정당호출 error→동일반복             = P7(root)
  - auth_fab       : 첫 auth 인자 날조→진행불가          = P1/P8(root)
  - no_error_fail  : 에러 없이 실패(over-ask/refuse/미도달)
  - pass

각 호출 인자의 provenance = 이전 user 발화 ∪ 이전 tool 출력에 등장? (없으면 날조).
첫 에러 후 동일 (name,args) 반복수 = 루프길이. 다른 호출 등장 = switch.

Usage: tau2_rootcause_census.py <sim_dir_or_results.json> [--show]
"""
import argparse, json, os, re
from collections import Counter

ERR_RE = re.compile(r"\b(error|not found|not exist|invalid|cannot|unable|"
                    r"policy gate|requires|denied|no .* found|too many)\b", re.I)


def load(path):
    if os.path.isdir(path):
        path = os.path.join(path, "results.json")
    return json.load(open(path, encoding="utf-8"))


def parse_args(a):
    if isinstance(a, str):
        try:
            return json.loads(a)
        except Exception:
            return {}
    return a or {}


def call_seq(msgs):
    """[(idx, name, args_dict, tool_output_str_or_None)] in order."""
    out = []
    for i, m in enumerate(msgs):
        if m.get("role") == "assistant":
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function", tc)
                nm = fn.get("name") or ""
                ar = parse_args(fn.get("arguments"))
                # 다음 tool 메시지 = 이 호출의 출력
                nxt = msgs[i + 1] if i + 1 < len(msgs) else {}
                outp = str(nxt.get("content")) if nxt.get("role") == "tool" else None
                out.append((i, nm, ar, outp))
    return out


def is_error(outp):
    return outp is not None and bool(ERR_RE.search(outp))


def context_before(msgs, idx):
    """idx 이전의 user 발화 + tool 출력 텍스트(소문자) 합본 = provenance 풀."""
    parts = []
    for m in msgs[:idx]:
        r = m.get("role")
        if r in ("user", "tool"):
            parts.append(str(m.get("content") or ""))
    return " \n ".join(parts).lower()


def arg_fabricated(args, ctx):
    """식별가능 인자값(len>=4) 중 ctx(이전 user∪tool)에 *없는* 게 있으면 날조."""
    vals = []
    for v in (args.values() if isinstance(args, dict) else []):
        for vv in (v if isinstance(v, list) else [v]):
            s = str(vv)
            if len(s) >= 4 and not s.replace(".", "").isdigit() or (len(s) >= 5 and s.isdigit()):
                vals.append(s)
    if not vals:
        return None  # 판정불가(짧은 인자만)
    fab = [v for v in vals if v.lower() not in ctx]
    return (len(fab) > 0, fab[:3])


def classify(sim):
    ri = sim.get("reward_info") or {}
    reward = ri.get("reward", 0.0)
    term = sim.get("termination_reason")
    msgs = sim.get("messages") or []
    seq = call_seq(msgs)
    if reward and reward > 0:
        return {"task": sim.get("task_id"), "root": "pass", "term": term,
                "n_call": len(seq), "detail": ""}

    # 첫 에러 호출 찾기
    first_err = None
    for k, (idx, nm, ar, outp) in enumerate(seq):
        if is_error(outp):
            first_err = (k, idx, nm, ar, outp)
            break

    if first_err is None:
        # 에러 없이 실패 = over-ask/refuse/미도달
        return {"task": sim.get("task_id"), "root": "no_error_fail", "term": term,
                "n_call": len(seq), "detail": "no tool error; %d calls" % len(seq)}

    k, idx, nm, ar, outp = first_err
    ctx = context_before(msgs, idx)
    fab = arg_fabricated(ar, ctx)
    is_auth = nm.startswith(("find_user", "get_user_id", "find_customer", "authenticate"))
    is_gate = bool(re.search(r"policy gate|requires|confirm|denied|not allowed", outp, re.I))

    # 첫 에러 후 행동: 동일 (name,args) 반복수 vs 다른 호출
    after = seq[k + 1:]
    same = sum(1 for (_, n2, a2, _) in after if n2 == nm and a2 == ar)
    diff = [n2 for (_, n2, a2, _) in after if not (n2 == nm and a2 == ar)]
    loop = same >= 2  # 첫 에러 외 2회+ 동일 = 루프

    # provenance-getter 시도했나(복구 방향): get_user_details 등 producer 호출
    tried_producer = any(n2 in ("get_user_details", "get_order_details", "list_all_product_types",
                                "get_product_details") and n2 != nm for n2 in diff)

    if is_auth and fab and fab[0]:
        root = "auth_fab"
    elif fab and fab[0]:
        root = "fab_then_loop" if loop else "fab_then_switch"
    elif is_gate:
        root = "gate_then_loop" if loop else "gate_then_switch"
    else:
        root = "legit_then_loop" if loop else "legit_then_switch"

    return {"task": sim.get("task_id"), "root": root, "term": term, "n_call": len(seq),
            "first_err_call": nm, "fab": (fab[1] if fab else None), "same_repeat": same,
            "diff_after": len(diff), "tried_producer": tried_producer,
            "detail": "%s(%s) err=%s | repeat=%d diff=%d producer=%s"
                      % (nm, (fab[1][0] if fab and fab[1] else ""), outp[:40].replace("\n", " "),
                         same, len(diff), tried_producer)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--show", action="store_true")
    a = ap.parse_args()
    sims = load(a.path)["simulations"]
    rows = [classify(s) for s in sims]
    rows.sort(key=lambda x: x["root"])

    print("=== τ² ROOT-CAUSE CENSUS: %s (n=%d) ===" % (a.path, len(rows)))
    for x in rows:
        print("  task %-4s %-18s | %s" % (str(x["task"]), x["root"], x.get("detail", "")))
    print("\n=== root 분포 ===")
    for r, n in Counter(x["root"] for x in rows).most_common():
        print("  %-18s %d" % (r, n))

    # ★핵심 분별 집계
    fab_root = sum(1 for x in rows if x["root"] in ("fab_then_loop", "fab_then_switch", "auth_fab"))
    gate_root = sum(1 for x in rows if x["root"] in ("gate_then_loop", "gate_then_switch"))
    legit_root = sum(1 for x in rows if x["root"] in ("legit_then_loop", "legit_then_switch"))
    loops = sum(1 for x in rows if "loop" in x["root"])
    switches = sum(1 for x in rows if "switch" in x["root"])
    producer = sum(1 for x in rows if x.get("tried_producer"))
    print("\n=== ★P2b vs P7 분별 ===")
    print("  날조-trigger(P2b root: fab+auth_fab) : %d" % fab_root)
    print("  gate-trigger                          : %d" % gate_root)
    print("  legit-error-trigger                   : %d" % legit_root)
    print("  --- 에러 후 행동 ---")
    print("  동일호출 루프(P7 미작동)              : %d" % loops)
    print("  다른시도(P7 부분작동/복구방향)        : %d" % switches)
    print("  producer-getter 시도(2-hop gather)    : %d / %d 실패궤적" %
          (producer, sum(1 for x in rows if x["root"] != "pass")))
    print("\n해석: 날조-trigger ≫ gate/legit → 근본=P2b(fetchable 날조)·P7은 증폭(루프).")
    print("     gate/legit-trigger 루프 多 → P7이 독립 근본.")


if __name__ == "__main__":
    main()
