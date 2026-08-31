# -*- coding: utf-8 -*-
"""x615 — t7391_reg12 task 60(retail) 격리 프로브.

물음 3개. 전부 **닫힌 술어**로만 답한다. 모델 호출 0 · 프롬프트 저작 0 · gold 는 채점에만
(어느 write 가 정답인지 보고 팔을 고르지 않는다 — 팔은 정책 축자와 게이트 **선언 술어**에서 나왔다).

  ① A_LIVE  : 라이브와 같은 입력(A2 retail gates + 궤적 축자 msg[1])으로 `GateInterpreter.check`
              를 부르면 msg 8 의 write 가 통과하는가 — 라이브 재현.
  ② N_NEG   : 매치 토큰 2개만 지운 반사실 · 무관 발화 부정통제([[57]]).
  ③ 팔 A/B/C/D : confirm 술어 후보 4종을 **두 런의 실행 write 전수**에 적용하고
              **gold write 오차단 수**를 함께 보고한다([[70]] ± 공개).
                A = 현행            CONFIRM_RE.search(last_user)
                B = TASK_12 P1 의 값싼 조작화  A ∧ (확인 발화 직전 assistant 가 텍스트-전용)
                C = A ∧ (확인 발화 index > 이 write 인자를 낸 마지막 tool 메시지 index)
                D = C 의 정제 — '생산자'를 **read 도구 결과**로만 한정(직전 write 결과 제외)
"""
import gzip, json, sys
sys.path.insert(0, "C:/workspace/ba-frft/scripts/distill/tau2")
sys.stdout.reconfigure(encoding="utf-8")
import gate_interpreter as GI
import t2_gate_patch as GP
import t2_forensic as F

BASE = "C:/workspace/ba-frft/reports/facet_rft_2026/sim_results/"
MUT = F.mutating_tools("retail")
GATES = GI.load_domain_a2("retail")["gates"]
WRITE = ("modify_pending_order_items",
         {"order_id": "#W5061109", "item_ids": ["3694871183"],
          "new_item_ids": ["8555936349"], "payment_method_id": "paypal_3742148"})


def _sim60(fn):
    d = json.load(gzip.open(BASE + fn, "rt", encoding="utf-8"))
    return [x for x in d["simulations"] if x["task_id"] == "60"][0]


def _mk():
    g = GI.GateInterpreter(GATES)          # T2_GATE_KINDS 미설정 = 전체 kind
    g.state.auth_user = "chen_johnson_4204"  # msg2/3 에서 실제로 확립된 값
    return g


def part1():
    s = _sim60("t7391_reg12.results.json.gz")
    u1 = s["messages"][1]["content"]
    print("① A_LIVE  check(modify_pending_order_items, last_user=msg[1]) ->",
          _mk().check(WRITE[0], WRITE[1], last_user_msg=u1, transfer_msg_sent=None))
    print("   CONFIRM_RE 매치:", GI.CONFIRM_RE.findall(u1))

    class M:
        def __init__(s_, r, c):
            s_.role, s_.content = r, c
    mm = [M(m.get("role"), m.get("content")) for m in s["messages"][:8]]
    print("   _regen_last_user 가 고른 발화 == msg[1] ?",
          GP._regen_last_user(mm) == u1)

    u2 = u1.replace("make sure the", "ensure the").replace(
        "confirm that explicitly", "state that plainly")
    print("② N_NEG1 (토큰 2개만 치환) 매치:", GI.CONFIRM_RE.findall(u2), "->",
          _mk().check(WRITE[0], WRITE[1], last_user_msg=u2, transfer_msg_sent=None)[:2])
    print("   N_NEG2 (무관 발화) ->",
          _mk().check(WRITE[0], WRITE[1],
                      last_user_msg="My earbuds are white.", transfer_msg_sent=None)[:2])
    # 손님이 요구한 뺄셈이 유일해인가 — env 사실만(gold 무참조)
    var = json.loads(s["messages"][7]["content"])["variants"]
    paid = 256.67
    cand = [k for k, v in var.items()
            if v["options"].get("color") == "blue" and v["available"] and v["price"] <= paid]
    print("   blue ∧ available ∧ price<=%.2f -> %d 후보 %s" % (paid, len(cand), sorted(cand)))
    print("   그중 water resistance == 'not resistant' ->",
          [k for k in cand if var[k]["options"].get("water resistance") == "not resistant"])


def arms(fn, tag):
    d = json.load(gzip.open(BASE + fn, "rt", encoding="utf-8"))
    rs = []
    for s in sorted(d["simulations"], key=lambda x: (int(x["task_id"]), x["trial"])):
        ms = s["messages"]
        owner = {}
        for i, m in enumerate(ms):
            if m.get("role") != "assistant":
                continue
            for off, tc in enumerate(m.get("tool_calls") or []):
                if i + 1 + off < len(ms) and ms[i + 1 + off].get("role") == "tool":
                    owner[i + 1 + off] = tc.get("name")
        for i, m in enumerate(ms):
            if m.get("role") != "assistant":
                continue
            for tc in (m.get("tool_calls") or []):
                if tc.get("name") not in MUT:
                    continue
                nxt = ms[i + 1] if i + 1 < len(ms) else {}
                if not (nxt.get("role") == "tool"
                        and not str(nxt.get("content", "")).startswith("Error")):
                    continue
                j = max([k for k in range(i)
                         if ms[k].get("role") == "user" and ms[k].get("content")] or [-1])
                lu = (ms[j].get("content") if j >= 0 else "") or ""
                mt = GI.CONFIRM_RE.search(lu)
                A = bool(mt)
                prevtxt = (j - 1 >= 0 and ms[j - 1].get("role") == "assistant"
                           and bool(ms[j - 1].get("content"))
                           and not ms[j - 1].get("tool_calls"))
                vals = []
                for v in (tc.get("arguments") or {}).values():
                    vals += [str(x) for x in (v if isinstance(v, list) else [v])]
                p_all = p_read = -1
                for k in range(i):
                    if ms[k].get("role") != "tool":
                        continue
                    c = ms[k].get("content") or ""
                    if any(v and v.lstrip("#") in c for v in vals):
                        p_all = k
                        if owner.get(k) not in MUT:
                            p_read = k
                gold = any(a["action"]["name"] == tc.get("name")
                           and a["action"]["arguments"] == (tc.get("arguments") or {})
                           for a in (s["reward_info"].get("action_checks") or []))
                rs.append(dict(t=s["task_id"], msg=i, tool=tc["name"], j=j,
                               tok=(mt.group(0) if mt else None),
                               A=A, B=A and prevtxt, C=A and j > p_all, D=A and j > p_read,
                               gold=gold))
    print("===== %s · 실행된 write %d" % (tag, len(rs)))
    print("%5s %4s %-32s %5s %-8s %-5s %-5s %-5s %-5s %s"
          % ("task", "msg", "tool", "lastU", "token", "A", "B", "C", "D", "goldARG"))
    for r in rs:
        print("%5s %4d %-32s %5d %-8s %-5s %-5s %-5s %-5s %s"
              % (r["t"], r["msg"], r["tool"], r["j"], r["tok"],
                 r["A"], r["B"], r["C"], r["D"], r["gold"]))
    for a in "ABCD":
        bg = [(r["t"], r["msg"]) for r in rs if r["gold"] and not r[a]]
        print("  %s: 통과 %d/%d · **gold write 오차단 %d** %s · 비-gold 통과 %d"
              % (a, sum(1 for r in rs if r[a]), len(rs), len(bg), bg,
                 sum(1 for r in rs if (not r["gold"]) and r[a])))
    print()


if __name__ == "__main__":
    part1()
    print()
    arms("t7391_reg12.results.json.gz", "TREAT t7391_reg12 (fc0055dc)")
    arms("hist_gpt52_reg12_PASS.results.json.gz", "CTRL hist_gpt52_reg12_PASS (5ebebbe8)")
