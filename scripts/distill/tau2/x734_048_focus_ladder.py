# -*- coding: utf-8 -*-
"""x734 — **결정점 문서 격리가 해결법인가** (048 빼기 실패·사다리 측정).

물음(사용자 2026-09-01): *"결정점에서 관련 문서들 격리가 해결법인가?"* ·
*"격리만으로 단계 빼기가 되면 그걸로 먼저 하고, 안되면 실행 엔진을 쓰는건 어떤가?"*
답하려면 **어디서 무너지는지**를 재야 한다([[62]]). 사다리:

  F_BINARY  지배 문서 + 그 카드의 적격성 결과 → **"새 사유를 기록할까? yes/no"**
            (가장 좁은 물음·생성 없음. 여기서도 틀리면 전달 문제가 아니다 ⇒ [[63]])
  D_FOCUS   같은 문서 + 같은 상태 + 손님 발화 → **카드별 쓰기 집합** (= 결정점 문서 격리)
  E_NODOC   D_FOCUS 에서 **정책 문서만 제거** · **부정통제**([[57]])

x733(A_EACH 장당1콜 · B_ALL 4장1콜 · C_STRIP 도구출력제거)과 같은 채점이라 붙여 읽는다.
F↔D 차이 = **물음의 형태**, D↔E 차이 = **문서의 유무**.

⚠문서 id·상태 도구를 코드에 적지 않는다 — **A2 `procedures` 선언을 읽는다**([[71]]/[[05]]).
   선언에 없으면 이 프로브는 돌지 않는다(그것이 곧 선언 결손의 신호다).
규율: gold 는 채점자로만 쓴다([[23]]) · 엔진은 비교만 한다([[10]]/[[52]]).

사용: x734_048_focus_ladder.py <base_url> <model> <tag> [반복] [절차id]
"""
import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import t2_forensic as F
import x430_account_facts as FT  # noqa: E402  (정본 DOCDIR·[[67]])
from x733_048_percard_probe import (DB, TASK, TRUNC, USER_ID, ask, cards_from_db,
                                    gold_per_card, materials)

HERE = os.path.dirname(os.path.abspath(__file__))
PROC_ID = "credit_card_closure_retention"


def declaration(proc_id):
    """A2 선언을 읽는다. **우리가 고르지 않는다.**"""
    with io.open(os.path.join(HERE, "a2", "banking_knowledge.gate.json"), encoding="utf-8") as f:
        d = json.load(f)
    for p in d.get("procedures") or []:
        if p.get("id") == proc_id:
            return p
    raise SystemExit("A2 에 절차 선언이 없다: %s" % proc_id)


def declared_docs(proc):
    """`_source` 항목은 '<doc id> (제목)' 꼴 — 선언된 id 만 취한다."""
    out = []
    for s in proc.get("_source") or []:
        did = str(s).split()[0].strip()
        p = os.path.join(FT.DOCDIR, "%s.json" % did)
        if not os.path.exists(p):
            print("  ⚠선언된 문서를 못 찾음: %s" % did)
            continue
        dd = json.load(io.open(p, encoding="utf-8"))
        out.append("### %s\n%s" % (dd.get("title"), dd.get("content") or ""))
    return out


def declared_state_tools(proc, upto="log_reason"):
    """`log_reason` 노드 **앞의** 도구 노드 = 적격성 검사. 선언 순서를 그대로 쓴다."""
    tools = []
    for n in proc.get("nodes") or []:
        if n.get("id") == upto:
            break
        if n.get("tool"):
            tools.append(n["tool"])
    return tools


def state_results(sim, account_id, tools):
    """그 카드에 대한 적격성 도구 **결과 축자**. 호출·인자·결과 잇기는 전부 **정본 헬퍼**를
    쓴다([[67]] 사본 금지). 선별 기준은 선언된 도구 이름 + 인자로 넘긴 account_id 이지
    본문 패턴매칭이 아니다([[59]])."""
    res = {m["id"]: str(m.get("content") or "")
           for m in (sim.get("messages") or [])
           if m.get("role") == "tool" and m.get("id")}
    out = []
    for _m, tc in F.calls(sim):
        a = F.argsof(tc)
        nm = str(F.inner_name(a) or F.nameof(tc))
        if nm not in tools:
            continue
        vals = [v for v in (F.flat_args(a) or {}).values() if isinstance(v, str)]
        if account_id not in vals:
            continue
        body = res.get(tc.get("id"))
        if body:
            out.append("TOOL RESULT (%s):\n%s" % (nm, body[:900]))
    return out


BIN_Q = ("You are a bank agent following the bank's internal protocol, quoted verbatim below, "
         "followed by the results of the eligibility checks you just ran on ONE credit card "
         "account.\n\n=== POLICY (verbatim) ===\n%s\n=== END POLICY ===\n\n"
         "=== ACCOUNT ===\n%s\n\n=== ELIGIBILITY CHECK RESULTS ===\n%s\n=== END ===\n\n"
         "The customer has asked to close this account.\n"
         "Question: according to the policy above, should you now log a NEW closure reason for "
         "this account using %s?")

ACT_Q = ("You are reviewing a bank support case. %s\n\n"
         "=== ACCOUNT ===\n%s\n\n=== ELIGIBILITY CHECK RESULTS FOR THIS ACCOUNT ===\n%s\n\n"
         "=== WHAT THE CUSTOMER SAID (whole conversation) ===\n%s\n=== END ===\n\n"
         "Decide which write actions should be performed on THIS account. Choose only from:\n%s\n"
         'If none should be performed, answer ["none"].')


def main():
    base = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8141/v1"
    model = sys.argv[2] if len(sys.argv) > 2 else "Qwen/Qwen3.8-27B-FP8"
    tag = sys.argv[3] if len(sys.argv) > 3 else "bank_x731_qB_t3prime_20260901_1621"
    reps = int(sys.argv[4]) if len(sys.argv) > 4 else 3
    pid = sys.argv[5] if len(sys.argv) > 5 else PROC_ID

    sim = next((s for s in F.sims(tag) if s.get("task_id") == TASK), None)
    if sim is None:
        print("sim 없음")
        return 2

    proc = declaration(pid)
    docs = declared_docs(proc)
    stools = declared_state_tools(proc)
    log_tool = next((n.get("tool") for n in (proc.get("nodes") or [])
                     if n.get("id") == "log_reason"), None)
    if not docs or not stools or not log_tool:
        print("선언 결손: docs=%d state_tools=%s log_tool=%s" % (len(docs), stools, log_tool))
        return 3

    rows = cards_from_db(DB, USER_ID)
    ids = [r["account_id"] for r in rows]
    gold = gold_per_card(sim, ids)
    vocab = sorted({t.get("name") for t in F.attempted_mutations(sim) if t.get("name")})
    vocab_txt = "\n".join("- " + v for v in vocab)
    says = materials(sim, with_tools=False)
    govern = "\n\n".join(docs)

    acct = {r["account_id"]: "id=%s | %s | balance %s"
            % (r["account_id"], r.get("card_type"), r.get("current_balance")) for r in rows}
    st = {i: "\n".join(state_results(sim, i, stools)) or "(no eligibility checks were run)"
          for i in ids}
    gold_log = {i: (log_tool in gold[i]) for i in ids}

    print("선언: 문서 %d개 %d자 · 적격성 도구 %s · 기록도구 %s"
          % (len(docs), len(govern), stools, log_tool))
    for i in ids:
        print("  %-22s 상태재료 %4d자 · gold 기록? %-5s · gold 집합 %s"
              % (i, len(st[i]), gold_log[i], sorted(gold[i]) or ["none"]))

    enum = vocab + ["none"]
    sch_bin = {"type": "object", "required": ["log_new_reason", "policy_basis"], "properties": {
        "log_new_reason": {"type": "boolean"},
        "policy_basis": {"type": "string"}}}
    sch_act = {"type": "object", "required": ["actions"], "properties": {
        "actions": {"type": "array", "items": {"type": "string", "enum": enum}}}}

    def norm(a):
        s = set(a or [])
        s.discard("none")
        return s

    res = {}
    for arm in ("F_BINARY", "D_FOCUS", "E_NODOC"):
        hits = {i: 0 for i in ids}
        preds = {i: [] for i in ids}
        for rep in range(reps):
            for i in ids:
                if arm == "F_BINARY":
                    r = ask(base, model, BIN_Q % (govern, acct[i], st[i], log_tool),
                            sch_bin, max_tokens=3072)
                    if r is None:
                        preds[i].append("무응답")
                        continue
                    got = bool(r.get("log_new_reason"))
                    preds[i].append("log" if got else "skip")
                    hits[i] += 1 if got == gold_log[i] else 0
                else:
                    head = ("The bank's governing policy documents are quoted verbatim below."
                            "\n\n=== POLICY ===\n%s\n=== END POLICY ===" % govern) \
                        if arm == "D_FOCUS" else "(No policy documents are provided.)"
                    r = ask(base, model, ACT_Q % (head, acct[i], st[i], says, vocab_txt),
                            sch_act, max_tokens=4096)
                    if r is None:
                        preds[i].append("무응답")
                        continue
                    p = norm(r.get("actions"))
                    preds[i].append(sorted(p))
                    hits[i] += 1 if p == gold[i] else 0
            print("  %s rep%d 완료" % (arm, rep), flush=True)
        res[arm] = (hits, preds)

    print("\n=== 사다리 결과 (카드별 정확일치 / %d회) ===" % reps)
    for arm in ("F_BINARY", "D_FOCUS", "E_NODOC"):
        hits, preds = res[arm]
        print("%-9s 합계 %2d/%d" % (arm, sum(hits.values()), reps * len(ids)))
        for i in ids:
            print("   %-22s %d/%d  %s" % (i, hits[i], reps, preds[i]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
