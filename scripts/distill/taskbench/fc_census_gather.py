#!/usr/bin/env python
"""M1 census (v7 게이트): 변환 데이터가 도구출력→arg threading을 값-복제로 보존하나 (2026-06-15).

리뷰(사용자): v7 진짜 게이트 = withholding이 아니라 *변환기가 도구출력→arg를 학습 궤적에 보존하는가*.
값을 3부류로 분리(§2 표 정정):
  (b) tool-produced-as-arg : 인자값이 *이전 tool 출력*에 있음(user 발화엔 없음) = order_id/session_token 유형 = 타깃 스킬
  (c) user-only           : 인자값이 *user 발화*에 있음 = ask-gather/withholding 대상
  (none)                  : 둘 다 아님 = upfront-or-fabricated
추가:
  (a) decision-only getter : getter 호출이나 그 출력이 이후 어떤 arg로도 안 쓰임 = precondition/결정용(arg 아님)
도메인별 집계 + (b) 예시 = "threading 보존" 게이트 판정.

Usage: fc_census_gather.py --in sop_all.jsonl [--getter_map getter_map.json] [--examples N]
"""
import argparse, json, re
from collections import Counter, defaultdict

RES = re.compile(r"[A-Za-z0-9_@.#-]{4,}")


def load_getter_names(path):
    if not path:
        return set()
    gm = json.load(open(path, encoding="utf-8"))
    names = set()
    for dom in gm.values():
        for lst in dom.values():
            names.update(lst)
    return names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--getter_map", default="")
    ap.add_argument("--examples", type=int, default=6)
    a = ap.parse_args()
    getters = load_getter_names(a.getter_map)

    per = defaultdict(lambda: {"traj": 0, "b_traj": 0, "c_traj": 0,
                               "b_vals": 0, "c_vals": 0, "none_vals": 0,
                               "getter_calls": 0, "getter_used_as_arg": 0})
    b_keys = Counter()
    b_examples = []

    for line in open(a.inp, encoding="utf-8"):
        ex = json.loads(line)
        dom = ex["_meta"].get("domain")
        msgs = ex["messages"]
        d = per[dom]; d["traj"] += 1
        user_text = " \n ".join(str(m.get("content") or "") for m in msgs if m.get("role") == "user")
        # 누적 tool 출력 + getter 출력값(이후 arg 사용 추적)
        tool_seen = ""
        getter_out_vals = []   # (value, producing_tool)
        traj_b = traj_c = False
        # 먼저 전체 arg 값 집합(getter 출력이 arg로 쓰이는지 사후 판정용)
        all_arg_blobs = []
        for m in msgs:
            if m.get("role") == "assistant":
                for tc in m.get("tool_calls") or []:
                    all_arg_blobs.append(str(tc["function"]["arguments"]))
        all_args_text = " ".join(all_arg_blobs)

        for i, m in enumerate(msgs):
            r = m.get("role")
            if r == "tool":
                c = str(m.get("content") or "")
                tool_seen += " \n " + c
            elif r == "assistant":
                for tc in m.get("tool_calls") or []:
                    nm = tc["function"]["name"]
                    if nm in getters:
                        d["getter_calls"] += 1
                    try:
                        args = json.loads(tc["function"]["arguments"])
                    except Exception:
                        continue
                    if not isinstance(args, dict):
                        continue
                    for k, v in args.items():
                        if not isinstance(v, (str, int, float)):
                            continue
                        sv = str(v)
                        if len(sv) < 4:
                            continue
                        in_tool = sv in tool_seen          # 이전 tool 출력서 옴
                        in_user = sv in user_text
                        if in_tool and not in_user:
                            d["b_vals"] += 1; b_keys["%s/%s" % (dom, k)] += 1; traj_b = True
                            if len(b_examples) < a.examples:
                                b_examples.append((dom, nm, k, sv[:30]))
                        elif in_user:
                            d["c_vals"] += 1; traj_c = True
                        else:
                            d["none_vals"] += 1
        if traj_b:
            d["b_traj"] += 1
        if traj_c:
            d["c_traj"] += 1
        # getter 출력이 arg로 쓰였나 (a vs b 구분 보조): getter 호출 후 출력 토큰이 all_args에 등장?
        # tool_seen에서 토큰 추출 후 arg에 있나 — 근사
        for tok in set(RES.findall(tool_seen)):
            if tok in all_args_text and len(tok) >= 4:
                d["getter_used_as_arg"] += 1
                break

    print("=== M1 CENSUS: %s ===" % a.inp)
    print("%-14s %6s %8s %8s | %8s %8s %8s | %8s" %
          ("domain", "traj", "b_traj", "c_traj", "b_vals", "c_vals", "none", "gettercall"))
    tot = defaultdict(int)
    for dom, d in sorted(per.items()):
        print("%-14s %6d %8d %8d | %8d %8d %8d | %8d" %
              (dom, d["traj"], d["b_traj"], d["c_traj"], d["b_vals"], d["c_vals"], d["none_vals"], d["getter_calls"]))
        for k in d:
            tot[k] += d[k]
    print("-" * 80)
    print("%-14s %6d %8d %8d | %8d %8d %8d | %8d" %
          ("TOTAL", tot["traj"], tot["b_traj"], tot["c_traj"], tot["b_vals"], tot["c_vals"], tot["none_vals"], tot["getter_calls"]))
    print("\n★게이트: b_traj/traj = %d/%d = %.1f%% (도구출력→arg threading 보존된 궤적)" %
          (tot["b_traj"], tot["traj"], 100 * tot["b_traj"] / max(tot["traj"], 1)))
    print("c_traj/traj = %d/%d = %.1f%% (user-only arg = withholding 대상)" %
          (tot["c_traj"], tot["traj"], 100 * tot["c_traj"] / max(tot["traj"], 1)))
    print("\n-- (b) tool-produced-as-arg 상위 키 --")
    for k, c in b_keys.most_common(20):
        print("   %-40s %d" % (k, c))
    print("\n-- (b) 예시 (domain, 호출도구, 인자키, 값) --")
    for e in b_examples:
        print("  ", e)


if __name__ == "__main__":
    main()
