# -*- coding: utf-8 -*-
r"""x575 — 표적 서브가 **재료를 받으면** `submit_transaction` 을 지목하는가 (유료 0).

## 왜 여기인가 (2026-08-28 · 핸드오프 §6 ① 의 정정판)

`T2_CARD_DOCS` 가 016 을 **절반** 샀다(t7367 2/4 · t7368 1/2 · 대조 0/6). 남은 두 시행은
손님이 `submit_transaction` 을 안 찍어서 죽는다. 라이브가 그것을 소유권 문구로 밀지 못하는
이유는 이미 계기로 고정돼 있다 — **중재 표적이 42/42 전부 `submit_referral`**
(`bank_t7367_hard0_20260827.log.gz` · `[T2_ARBITRATE] push dominated target=submit_referral`).

## 안 재본 축은 **문장이 아니라 재료**다 ([[74]] 검색 결과)

핸드오프는 이것을 *"물음의 범위"* 라고 적었는데 그 안에 두 가설이 섞여 있고 **하나는 이미
음성**이다:

    x516  후보집합에서 `submit_referral` 제외        → gold **0/39** (38/39 가 에이전트 디스패처)
    x517  물음의 **동사 프레임** 교체(agent CALL →
          must be EXECUTED … or by the customer)     → gold **0/39/0/39/0/39** (세 팔 전부)

세 팔·두 프로브가 공유한 것은 **창**이다 — `x516_induction_target_iso.py:163`
`win = [msgs[j] for j in uidx[max(0, k - 5):k + 1]]` = **손님 발화뿐**. 라이브도 같다:
t7367 `[T2_SUBWIN]` 76 호출이 전부 `user_msgs=1..8`(도구 출력·우리 층 주입 0자).
그리고 016 gold `submit_transaction` 은 **원장 상태(IN_PROGRESS)+정책 수치($750)** 에서만
도출된다 ⇒ 그 창 밖이다. `t2_resolve.py:840` 의 계기 주석이 이미 그렇게 적어 뒀다.

⇒ 이 프로브가 가르는 것: **재료를 넣으면 갈리는가**, 그리고 그것이 **문장과 어떻게 섞이는가**.

## 팔 — 2×2 + 부정통제 ([[57]])

    A_asis          정본 물음 · 재료 없음      ← 라이브 재현 게이트(42/42 `submit_referral`)
    B_mat           정본 물음 · 재료 있음      ← 재료만으로 갈리나 (물음은 여전히 *"ONLY … user asked"*)
    C_neutral       x517 중립 물음 · 재료 없음  ← x517 재현 게이트(gold 0)
    D_neutral_mat   x517 중립 물음 · 재료 있음  ← **전체 처치**
    N_neg           x517 중립 물음 · **같은 길이·정보 0** 재료 ← 길이가 아니라 재료임을 가른다

N_neg 이 D 만큼 움직이면 산 것은 재료가 아니라 **문맥이 길어진 것**이다.

## 재료 — 저작 0 · 선언이 이미 **배달하는 두 문장** ([[71]]·[[23]])

라이브가 그 자리에 실제로 내보내는 것을 정본 생산자로 그대로 만든다. 프로브가 문면을 쓰지
않는다([[78]] ①):

    diagnose_choice(...)   → a2 `ledger_metrics[].diagnosed_text.format(answer=…)`
    requirement_choice(...) → a2 `ledger_metrics[].requirement_text.format(answer=…)`

문서는 A3 `policy_ontology.doc_index[군][주어]` 가 지목한 것만 읽는다(검색·유사도·선별 0).
gold 는 **채점에만** 쓰고 프롬프트 어디에도 안 들어간다.

## 계기 — 사본 0 ([[67]])

프롬프트·파싱·집합소속을 베끼지 않는다. 정본 `t2_resolve.formalize_intent_tool` 을 그대로
부르고 **선언 오버라이드 두 칸**(`ask`·`material`)만 바꾼다. 창·후보집합·어댑터는
`x516_induction_target_iso` 를, 원장 행은 `x554`, 문서 읽기는 `t2_search` 를 import 한다.
`--wiring-only` 는 팔별 프롬프트를 찍고 **A_asis 가 개정 전 문면과 바이트 동일**임을 검정한다.

## ⛔[[62]] 4문 (이 편집·이 프로브)

  ①격리로 재봤나 — 이 자리는 x516(39창)·x517(39창×3팔)이 쟀고 **둘 다 재료 고정**이었다.
    재료 축은 미측정이고 그것이 이 프로브다.
  ②격리에서 성공하나 — **미지**. 그것을 재려고 연다.
  ③사라지는 모델 판단 — **0**. `material=None` 이면 프롬프트가 바이트 동일이고 라이브 경로는
    이 인자를 주지 않는다. 고르는 것은 여전히 서브다.
  ④엔진이 argmax·"정답은 X" 를 내나 — **아니오**. 엔진은 답이 `action_tools` 의 원소인지만
    본다(종전과 동일). 재료도 엔진이 요약하지 않고 선언된 템플릿을 그대로 나른다.

## ⚠공정성 ([[62]] 2b)

재료는 라이브에서 **원장 도구가 돌아온 뒤에야** 우리 손에 있다. 그래서 창마다 `armed`
(그 창의 마지막 손님 발화가 원장 도착 뒤인가)를 함께 재고, 집계를 **전체**와 **armed 만**
둘로 낸다. armed 밖에서 오른 gold 는 라이브가 살 수 없는 점수다.

사용: (리모트) cd $REPO/scripts/distill/tau2 &&
      PYTHONPATH=. py -3 x575_target_scope_iso.py --port 8140 [--wiring-only]
"""
import argparse
import collections
import gzip
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import gate_interpreter as GI                                       # noqa: E402
import t2_gate_patch as G                                           # noqa: E402
import t2_ledger as LG                                              # noqa: E402
import t2_resolve as RZ                                             # noqa: E402
import t2_search as SRCH                                            # noqa: E402
import x516_induction_target_iso as X16                             # noqa: E402
import x554_diag_mispick_iso as X554                                # noqa: E402

NL = chr(10)
OUT = os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026")
SIMS = os.path.join(OUT, "sim_results")
DOCS = "/home/woori/scratch/tau2-bench/data/tau2/domains/banking_knowledge/documents"

# 큐 세대만 본다 — `T2_CARD_DOCS` 가 켜진 두 런 ([[74]] §74-b 세대-뭉개기 금지)
RUNS = ("bank_t7367_hard0_20260827", "bank_t7368_hard0_20260827")

# x517 이 선언한 중립 물음 — **그 프로브의 산출물에서 인용**한다(여기서 새로 쓰지 않는다).
X517_JSON = os.path.join(OUT, "x517_question_frame_iso_2026_08_24.json")

ARMS = ("A_asis", "B_mat", "C_neutral", "D_neutral_mat", "N_neg")


def neutral_ask():
    """x517 의 `C_neutral` 문장을 그 프로브의 결과 파일에서 읽는다(사본 0)."""
    d = json.load(io.open(X517_JSON, encoding="utf-8"))
    return str((d.get("asks") or {}).get("C_neutral") or "") or None


def gold_user_actions(task):
    """gold 의 `requestor=user` 액션 이름 — **채점에만** 쓴다([[23]])."""
    for tag in RUNS:
        p = os.path.join(SIMS, tag + ".results.json.gz")
        if not os.path.exists(p):
            continue
        d = json.load(gzip.open(p, "rt", encoding="utf-8", errors="replace"))
        for t in (d.get("tasks") or []):
            if t.get("id") == task:
                out = sorted({x.get("name") for x in
                              ((t.get("evaluation_criteria") or {}).get("actions") or [])
                              if x.get("requestor") == "user"})
                if out:
                    return out
    return []


def sim_layout(tag, simtag, keys):
    """(원장 도구 출력의 메시지 인덱스, 손님 발화의 메시지 인덱스 목록).

    공정성 계측용이다([[62]] 2b): 원장이 도착한 **뒤의** 창에서만 우리 층이 재료를 손에 쥔다.
    ⚠근사하지 않는다 — x516 의 `turn_k` 는 손님 발화 목록의 첨자이므로 `uidx[turn_k]` 로
      **절대 위치**를 얻어 맞댄다(핸드오프 §5: *재려는 것 대신 그 옆의 것을 보지 마라*).
    위치 찾기뿐이고 파싱은 안 한다 — 행은 `x554.rows_from_traj` 가 만든다.
    """
    import t2_forensic as F
    files = {F.tag_of_file(q): q for q in F.all_result_files()}
    if tag not in files:
        return -1, []
    sim = next((x for x in F.sims(files[tag]) if F.simtag(x) == simtag), None)
    if sim is None:
        return -1, []
    lidx, uidx = -1, []
    for i, m in enumerate(sim.get("messages") or []):
        if m.get("role") == "user":
            uidx.append(i)
        elif (lidx < 0 and m.get("role") == "tool"
              and all(("%s:" % k) in str(m.get("content") or "") for k in keys)):
            lidx = i
    return lidx, uidx


def null_like(s):
    """같은 길이 · 사실 0 ([[57]]). 문장 형태는 두고 **기록도 이름도 수도 안 담는다**."""
    f = "the check returned nothing that names a record or states a figure and "
    return (f * (len(s) // len(f) + 1))[:len(s)]


class _Cap(X16._LA):
    """x516 어댑터에 **프롬프트 포획**만 얹는다(생성 경로는 그대로)."""

    def __init__(self, port, maxtok=64):
        X16._LA.__init__(self, port, maxtok)
        self.last = None

    def generate(self, model=None, tools=None, messages=None, call_name=None, **kw):
        self.last = str(getattr(messages[0], "content", "") or "")
        return X16._LA.generate(self, model=model, tools=tools, messages=messages,
                                call_name=call_name, **kw)


def expected_asis(action_tools, users, ask=None):
    """개정 **전** 문면을 그대로 재구성한다 — 바이트 동일 검정용(래칫)."""
    return ("The user is talking to a customer-service agent. Based ONLY on what the user asked, "
            + (ask or RZ.ASK_AGENT_CALL) +
            "Reply with the exact tool name, or 'none' if none applies." + NL +
            "Tools: " + ", ".join(sorted(action_tools)) + NL +
            "User said:" + NL + "- " + (NL + "- ").join(u[:300] for u in users) +
            NL + 'Reply JSON only: {"tool": "<name or none>"}')


def materials(port, tag, simtag, spec, a3r, idx, keys, docdir):
    """그 sim 에 대해 **라이브가 배달하는 두 문장** — 정본 생산자로 만든다(저작 0)."""
    rows = X554.rows_from_traj(tag, simtag, keys)
    if not rows:
        return None, "원장 행 없음"
    blk = LG.onto_context(rows, spec, a3r)
    ag, la = X16._Agent(), _Cap(port, 160)
    dg = LG.diagnose_choice(ag, la, X16._UM, spec, blk, rows)
    if not dg:
        return None, "진단 침묵"
    diag = str(spec.get("diagnosed_text") or "").format(answer=dg[1]).strip()

    pick = None
    for fam, subs in (idx or {}).items():
        for s in G._subject_keys(subs):
            if G._slug_disp(s).strip().lower() == str(dg[0]).strip().lower():
                pick = (fam, s)
                break
        if pick:
            break
    if not pick:
        return None, "색인 밖 이름: %r" % (dg[0],)
    ids = list((idx.get(pick[0]) or {}).get(pick[1]) or ())
    docs, miss = SRCH.read_docs(ids, doc_dir=docdir)
    if not docs:
        return None, "문서 0 (없는 id %d)" % len(miss)
    body = NL.join("ID: " + k + NL + docs[k] for k in sorted(docs))
    rq = LG.requirement_choice(ag, la, X16._UM, spec, body, dg[0], sorted(docs))
    if not rq:
        return None, "요건 침묵(인용 없음)"
    req = str(spec.get("requirement_text") or "").format(answer=rq).strip()
    return {"subject": dg[0], "docs": len(docs), "missing": len(miss),
            "real": [diag, req], "null": [null_like(diag), null_like(req)]}, None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8140)
    ap.add_argument("--task", default="task_016")
    ap.add_argument("--docs", default=DOCS)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--wiring-only", action="store_true")
    a = ap.parse_args(argv)

    ask_neutral = neutral_ask()
    if not ask_neutral:
        print("x517 의 중립 물음을 못 읽었다 — 돌리지 않는다([[25]])", file=sys.stderr)
        return 2

    a2 = GI.load_domain_a2("banking_knowledge") or {}
    spec = next((s for s in (a2.get("ledger_metrics") or []) if s.get("diagnose_prompt")), None)
    if not spec:
        print("선언에 원장 스펙이 없다", file=sys.stderr)
        return 2
    keys = list(spec.get("row_keys") or ())
    a3r = ((a2.get("policy_ontology") or {}).get("rows")) or ()
    idx = (a2.get("policy_ontology") or {}).get("doc_index") or {}

    X16.RUNS = RUNS                     # 창 만드는 기계는 x516 것 그대로 (사본 0)
    X16.TASK = a.task
    cases = X16.windows()
    if a.limit:
        cases = cases[:a.limit]
    if not cases:
        print("재료 없음 — %s 의 ACTIONREQ 줄을 못 찾았다. 돌리지 않는다([[25]])." % a.task)
        return 1

    gold = gold_user_actions(a.task)
    print("# x575 — 표적 서브 재료 축 · %s · 창 %d개 · gold(requestor=user)=%s"
          % (a.task, len(cases), gold))

    # ── 재료: sim 하나당 한 번 (라이브도 memo 로 한 번이다)
    mats, why = {}, {}
    lidx, uidx = {}, {}
    for st in sorted({(c["run"], c["simtag"]) for c in cases}):
        tag, simtag = st
        lidx[st], uidx[st] = sim_layout(tag, simtag, keys)
        m, err = materials(a.port, tag, simtag, spec, a3r, idx, keys, a.docs)
        mats[st] = m
        why[st] = err
        print("  %-30s %-20s 원장 msg[%s] · 손님 발화 %d · %s"
              % (tag[-22:], simtag, lidx[st], len(uidx[st]),
                 ("주어=%s · 문서 %d · 재료 %d자"
                  % (m["subject"], m["docs"], sum(len(x) for x in m["real"]))) if m else ("침묵: %s" % err)))
    if not any(mats.values()):
        print("어느 sim 에서도 재료가 안 나왔다 — 판정하지 않는다([[25]])", file=sys.stderr)
        return 2

    la = _Cap(a.port, 64)
    ag = X16._Agent()

    def ask_of(arm):
        return None if arm in ("A_asis", "B_mat") else ask_neutral

    def mat_of(arm, st):
        m = mats.get(st)
        if not m or arm in ("A_asis", "C_neutral"):
            return None
        return m["null"] if arm == "N_neg" else m["real"]

    if a.wiring_only:
        c = next((x for x in cases if mats.get((x["run"], x["simtag"]))), cases[0])
        st = (c["run"], c["simtag"])
        for arm in ARMS:
            RZ.formalize_intent_tool(ag, la, X16._UM, c["msgs"], list(c["cands"]),
                                     ask=ask_of(arm), material=mat_of(arm, st))
            print("--- %s (%d자) ---" % (arm, len(la.last or "")))
            print(la.last)
            print("")
        users = [str(getattr(m, "content", "") or "") for m in c["msgs"]
                 if getattr(m, "role", None) == "user"][-6:]
        RZ.formalize_intent_tool(ag, la, X16._UM, c["msgs"], list(c["cands"]))
        same = (la.last == expected_asis(list(c["cands"]), users))
        print("래칫 — A_asis 가 개정 전 문면과 바이트 동일: %s" % ("OK" if same else "★불일치"))
        return 0 if same else 3

    print("")
    print("%-22s %-4s %-5s %s" % ("sim", "k", "armed", " · ".join("%-22s" % x for x in ARMS)))
    print("-" * 132)
    tally = collections.defaultdict(collections.Counter)
    rows_out = []
    for i, c in enumerate(cases):
        st = (c["run"], c["simtag"])
        # armed = 그 창의 **마지막 손님 발화**가 원장 도착 뒤인가 = 재료를 손에 쥔 자리인가.
        _u, _l = uidx.get(st) or [], lidx.get(st, -1)
        armed = (bool(mats.get(st)) and _l >= 0 and 0 <= c["turn_k"] < len(_u)
                 and _u[c["turn_k"]] > _l)
        row = {"run": c["run"], "simtag": c["simtag"], "turn_k": c["turn_k"],
               "live_target": c["live_target"], "armed": armed}
        for arm in ARMS:
            got = RZ.formalize_intent_tool(ag, la, X16._UM, c["msgs"], list(c["cands"]),
                                           ask=ask_of(arm), material=mat_of(arm, st))
            row[arm] = got
            tally[("all", arm)][str(got)] += 1
            if gold and got in gold:
                tally[("all", arm)]["__GOLD__"] += 1
            if armed:
                tally[("armed", arm)][str(got)] += 1
                if gold and got in gold:
                    tally[("armed", arm)]["__GOLD__"] += 1
        rows_out.append(row)
        print("%-22s %-4d %-5s %s"
              % (c["simtag"], c["turn_k"], "Y" if armed else "-",
                 " · ".join("%-22s" % str(row[x]) for x in ARMS)))

    n_all = len(cases)
    n_arm = sum(1 for r in rows_out if r["armed"])
    print("")
    print("=" * 96)
    print("결과 — 팔별 산출 (창 %d · armed %d · 서브콜 %d회)" % (n_all, n_arm, la.calls))
    print("=" * 96)
    for scope, n in (("all", n_all), ("armed", n_arm)):
        print("  [%s]" % scope)
        for arm in ARMS:
            cnt = tally[(scope, arm)]
            g = cnt.get("__GOLD__", 0)
            dist = " · ".join("%s×%d" % kv for kv in cnt.most_common() if kv[0] != "__GOLD__")
            print("    %-14s gold %d/%d   %s" % (arm, g, n, dist))
    print("")
    print("판독:")
    print("  A_asis 에서 라이브 표적(`submit_referral`)이 재현돼야 이 격리가 그 자리다.")
    print("  D 의 gold 가 C 보다 크고 **N_neg 은 안 그러면** 산 것은 재료다([[57]]).")
    print("  B 가 A 만큼이면 재료만으로는 안 되고 **범위 문장과 함께**여야 한다는 뜻이다.")
    print("  D 도 gold 0 이면 결손은 재료가 아니라 **그 위**에 있고, 표적 서브는 이 태스크의")
    print("  경로가 아니다 — 016 의 남은 두 시행은 다른 자리에서 사야 한다.")

    out = {"probe": "x575_target_scope_iso", "date": "2026-08-28", "task": a.task,
           "runs": list(RUNS), "gold_user": gold, "asks": {"A_asis": RZ.ASK_AGENT_CALL,
                                                           "C_neutral": ask_neutral},
           "n_windows": n_all, "n_armed": n_arm, "subcalls": la.calls,
           "materials": {("%s|%s" % k): ({"subject": v["subject"], "docs": v["docs"],
                                          "missing": v["missing"], "real": v["real"]}
                                         if v else {"silent": why.get(k)})
                         for k, v in mats.items()},
           "ledger_msg_index": {("%s|%s" % k): v for k, v in lidx.items()},
           "tally": {"%s.%s" % k: dict(v) for k, v in tally.items()},
           "rows": rows_out,
           "limits": ["창·후보집합은 `x516_induction_target_iso` 가 만든 것 그대로(사본 0).",
                      "temperature 0 · 창마다 1회 — n 은 **창 수**이지 재시행 수가 아니다.",
                      "gold 는 채점에만 썼다. 프롬프트·후보집합 어디에도 안 들어간다([[23]]).",
                      "`armed` 는 손님 발화 수로 근사한 하한이다 — 창의 절대 위치를 x516 이 "
                      "내지 않아서다. armed 밖의 gold 는 라이브가 살 수 없는 점수로 읽어라.",
                      "재료는 선언된 두 템플릿(`diagnosed_text`·`requirement_text`)을 정본 "
                      "생산자로 채운 것뿐이고 프로브가 쓴 도메인 문장은 0이다."]}
    dst = os.path.join(OUT, "x575_target_scope_iso_2026_08_28.json")
    with io.open(dst, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("")
    print("-> %s" % os.path.normpath(dst))
    return 0


if __name__ == "__main__":
    sys.exit(main())
