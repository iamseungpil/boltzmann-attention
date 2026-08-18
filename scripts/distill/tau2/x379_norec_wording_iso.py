# -*- coding: utf-8 -*-
r"""x379 — **조회 실패 문면이 재조회 루프를 여는가**(격리 ② · 핸드오프 §5⑵ · [[64]] 정면 표적).

## 무엇이 관측됐나 (t7313 treat `task_040` 라이브 · 사이드카 축자)

    [6x] Error: [SIGNATURE] give_discoverable_user_tool takes only `discoverable_tool_name` …
    [5x] Error: resolve the flagged call(s) first; do not call this tool yet.      ← 이름 없음
    [4x] Error: [WRITE-GROUNDING] the value 'your_credit_card_account_id' …
    · `[T2_MATERIAL_GATE] stop=resolve_cap(정체 3회)` 가 **turn 56→104 짝수 턴마다 59회**
    · 같은 sim 이 turn 104·2,780초 (`GO_MAX_STEPS=150` 이 없었으면 t7307 처럼 런이 죽는다)

## 이 프로브가 잰다 (문면 하나만 갈아 끼운다 · 문맥은 라이브 축자)

A2 `no_record_template`(v1)은 *"…then call this tool again"* 으로 닫혀 **종료 분기가 없다**
(x33·x34 가 D1 으로 기록). 같은 A2 에 **이미 저작된 `no_record_template_v2`** 가 있는데
`T2_NOREC_BRANCH` 로 잠겨 있고 **`go_stack.sh` 는 그 플래그를 안 켠다** — 라이브에서 한 번도
쓰인 적이 없다([[60]] 위반 후보). 즉 여기서 살 것은 새 결정론이 아니라 **이미 있는 문면**이다.

    A_REF    v1 축자(라이브 현행)                                  ← 기준선
    B_V2     v2 축자(같은 인자 재조회 금지 + 종결 분기 + 순서)      ← 이미 저작된 문면
    C_NAME   v2 + **그 자리의 다음 행동 이름**(핸드오프 원안)        ← 한 수 더 준다
    D_NEG    v1 + **무관한 도구 이름**                              ← 계기(이름을 읽나)

## 채점 (결정론 · gold 무참조 · [[23]])

  ⑴ `repeat` — 직전에 실패한 호출을 **인자까지 그대로** 다시 냈는가(루프의 정의)
  ⑵ `newarg` — 같은 도구를 **다른 식별자**로 냈는가(v2 가 사려는 회복 경로)
  ⑶ `progress` — 다른 도구로 넘어갔는가(이름·안쪽 이름 접미사 일치)

## 판정 (사전 고정 · 결과보다 먼저 인쇄)

    B 의 repeat 가 A 보다 **과반 컷에서 낮다**       → 문면이 산다 ⇒ 처방 = `T2_NOREC_BRANCH` ON
    B ≈ A 이고 C 만 열린다                          → 문면만으로는 부족·**행동 이름**이 본체
    A·B·C 전부 repeat                               → 문면 축이 아니다 ⇒ [[62]]③ 그 단계 결정론 검토
    D_NEG 가 A 와 다르게 움직인다                    → 모델이 이름을 읽는다(채널 생존)
    네 팔이 전부 동일                                → **계기 무효**(결과 인용 금지)

⚠엔진 수정 0 · 오프라인 · 8141 전용([[30]] 포트 분리).
실행: T2_PROBE_URL=http://localhost:8141/v1/chat/completions \
      /home/woori/venvs/seka_env/bin/python x379_norec_wording_iso.py [태그 ...]
"""
import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/home/woori/scratch/tau2-bench/src")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import t2_forensic as F                                   # noqa: E402
import x370_infomatched_give_iso as G                     # noqa: E402
import x351_order_lever_iso as X                          # noqa: E402

ANCHOR = "the account record has not been fetched yet"
ASK = ("You are the bank's support agent. Continue this conversation and make the tool call "
       "you would make now.")
# D_NEG 가 댈 **무관한 도구** — 레지스트리 실재 확인 후에만 쓴다.
# ⚠1차 실행 결함: 후보가 하나(`get_exchange_rate`)뿐이라 레지스트리에 없자 `sorted(reg)[0]` =
#   `call_discoverable_agent_tool` 로 떨어졌다. 그것은 **그 자리에서 그럴듯한 다음 수**라
#   부정통제가 안 된다(무관해야 '이름을 읽는가'를 잰다). 후보를 두고, 없으면 조회-계열이
#   아닌 것 중 첫 번째를 쓴다 — 선택 규칙은 고정이고 엔진 판단 0.
NEG_CANDIDATES = ("transfer_to_human_agents", "send_verification_code", "get_exchange_rate")
DEFAULT_TAGS = ["bank_t7310_ctl_20260818e", "bank_t7310_treat_20260818e",
                "bank_t7312_ctl_20260818g", "bank_t7312_treat_20260818g"]


def templates():
    """A2 축자 — v1·v2 를 **읽기만** 한다(문면 저작 0·[[23]] gold 무참조)."""
    a2 = X.a2_load()
    spec = None
    if spec is None:                     # ★구조를 가정하지 않는다 — **모양으로** 찾는다.
        #   1차 실행이 `scaffold_get_tools` 를 dict 로 가정해 죽었다(실제는 **list**).
        def walk(o):
            if isinstance(o, dict):
                if o.get("no_record_template"):
                    return o
                for x in o.values():
                    r = walk(x)
                    if r:
                        return r
            elif isinstance(o, list):
                for x in o:
                    r = walk(x)
                    if r:
                        return r
            return None
        spec = walk(a2)
    if not spec:
        return "", ""
    return (str(spec.get("no_record_template") or ""),
            str(spec.get("no_record_template_v2") or ""))


def cuts_from(tag):
    """라이브에서 **그 문면이 실제로 나간 자리**만 컷으로 삼는다(합성 0).

    ⚠sim 당 **첫 발화 하나만** 쓴다: 한 sim 안의 재발화는 독립 표본이 아니고(같은 문맥이
      길어질 뿐) 72 컷 × 4 팔 × 2 회로 계기가 커진다. 첫 자리가 *루프가 시작되는 자리*다.
    """
    out, seen = [], set()
    for r in F.sidecar_rows(tag):
        txt = str(r.get("text") or "")
        st = str(r.get("simtag") or "")
        if ANCHOR not in txt or st in seen:
            continue
        seen.add(st)
        out.append({"tag": tag, "simtag": st, "turn": r.get("turn")})
    return out


def last_failed_call(sim, upto_idx):
    """직전에 **실패한 호출**(이름+인자 축자) — `repeat` 판정의 기준점."""
    last = None
    for i, m in enumerate((sim.get("messages") or [])[:upto_idx]):
        for tc in (m.get("tool_calls") or ()):
            last = (F.nameof(tc), json.dumps(F.argsof(tc), sort_keys=True, ensure_ascii=False))
    return last


def upto_index(sim, turn):
    msgs = sim.get("messages") or []
    for i, m in enumerate(msgs):
        ti = m.get("turn_idx")
        if ti is not None and int(ti) >= int(turn):
            return i
    return len(msgs)


def emitted(msg):
    for tc in ((msg or {}).get("tool_calls") or ()):
        f = tc.get("function") or tc
        return str(f.get("name") or ""), str(f.get("arguments") or "")
    return "", ""


def det2(prompt, tools, first=520, retry=960):
    """`G.det` + **절단 재시도** — 1차 실행에서 호출의 ~20%가 `HTTP 400` 으로 죽었다.

    ⚠원인은 모델이 아니라 예산이다: `tool_choice="required"` 로 강제된 도구-호출 JSON 이
      `max_tokens` 에서 잘리면 서버 파서가 *Invalid JSON: EOF while parsing* 로 **400** 을 낸다
      (x370 v5b 가 같은 부류를 겪었다). 빈 팔은 비교를 **편향**시키므로 조용히 두지 않는다.
    ⚠재시도는 **예산만** 바꾼다 — 프롬프트·온도·도구 목록 불변(측정 대상 불변).
    """
    msg, det = G.det(prompt, tools, first)
    if isinstance(msg, dict) and msg.get("_err"):
        msg, det = G.det(prompt, tools, retry)
    return msg, det


def next_action_name(sim, upto_idx):
    """C_NAME 이 댈 **다음 행동 이름** — 그 대화에서 이미 **해금·노출된** 이름 중 미호출인 것.

    ⚠gold 를 안 본다([[23]]): 출처는 대화에 이미 등장한 도구 이름뿐이고, 고르는 규칙은
      *아직 호출되지 않은 것 중 가장 최근에 노출된 것* — 엔진 판단 0(순서 하나뿐).
    """
    msgs = (sim.get("messages") or [])[:upto_idx]
    called = set()
    for m in msgs:
        for tc in (m.get("tool_calls") or ()):
            called.add(F.nameof(tc))
            called.add(str(F.inner_name(F.argsof(tc)) or ""))
    seen = []
    for m in msgs:
        txt = str(m.get("content") or "")
        for tok in txt.replace("'", " ").replace('"', " ").replace("`", " ").split():
            t = tok.strip(".,()[]{}:;")
            if t.startswith("get_") and t not in called and t not in seen:
                seen.append(t)
    return seen[-1] if seen else ""


def main():
    tags = [a for a in sys.argv[1:] if not a.startswith("--")] or DEFAULT_TAGS
    v1, v2 = templates()
    if not (v1 and v2):
        print("⛔A2 에서 v1/v2 문면을 못 읽었다 — 중단(계기 결함)")
        return 1
    tools = G.agent_tool_specs()
    reg = set(t["function"]["name"] for t in tools)
    neg = next((n for n in NEG_CANDIDATES if n in reg),
               next((n for n in sorted(reg)
                     if not n.startswith(("get_", "call_", "unlock_"))), sorted(reg)[0]))

    print("=" * 104)
    print("x379 · 조회 실패 문면 격리 · 태그 %s · 도구 %d개 · D_NEG 도구=%s"
          % (",".join(tags), len(tools), neg))
    print("판정(사전 고정): B repeat < A 과반 → 문면이 산다(처방=T2_NOREC_BRANCH ON) · "
          "B 와 A 같고 C 만 열림 → 행동 이름이 본체 · 전부 repeat → 문면 축 아님 · "
          "네 팔 동일 → 계기 무효")
    print("=" * 104)

    rows = []
    for tag in tags:
        sims = {F.simtag(s): s for s in F.scored(tag, ".results.json.gz")}
        for c in cuts_from(tag):
            sim = sims.get(c["simtag"])
            if sim is None or c["turn"] is None:
                continue
            ui = upto_index(sim, c["turn"])
            base = G.convo(sim, ui)
            lf = last_failed_call(sim, ui)
            if not base or not lf:
                continue
            nm = next_action_name(sim, ui)
            arms = [("A_REF", v1), ("B_V2", v2),
                    ("C_NAME", v2 + ((" The next step is to call %s." % nm) if nm else "")),
                    ("D_NEG", v1.replace("get_user_information_by_name/by_email/by_id", neg))]
            got = {}
            for an, body in arms:
                prompt = base + "\n\ntool: " + " ".join(body.split()) + "\n\n" + ASK
                msg, det = det2(prompt, tools)
                enm, ear = emitted(msg)
                same_tool = int(bool(enm) and enm == lf[0])
                try:
                    same_arg = int(json.dumps(json.loads(ear or "{}"), sort_keys=True,
                                              ensure_ascii=False) == lf[1])
                except Exception:
                    same_arg = 0
                got[an] = {"tool": enm, "args": ear, "det": det,
                           "repeat": int(bool(same_tool and same_arg)),
                           "newarg": int(bool(same_tool and not same_arg)),
                           "progress": int(bool(enm) and not same_tool)}
            rows.append({"task": c["simtag"].split("#")[0], "tag": tag.split("_")[1],
                         "arm": ("treat" if "treat" in tag else "ctl"), "turn": c["turn"],
                         "prev": lf[0], "namehint": nm, "got": got})
            print("  %-9s %-6s %-5s turn=%-3s prev=%-28s | %s%s"
                  % (rows[-1]["task"], rows[-1]["tag"], rows[-1]["arm"], c["turn"], lf[0],
                     " · ".join("%s:%s" % (a, ("repeat" if got[a]["repeat"] else
                                               "newarg" if got[a]["newarg"] else
                                               "prog(%s)" % (got[a]["tool"] or "없음")))
                                for a in ("A_REF", "B_V2", "C_NAME", "D_NEG")),
                     "" if all(got[a]["det"] for a in got) else "  ⚠비결정"))

    if not rows:
        print("")
        print("⛔컷 0 — 계기 결함(결과 없음)")
        return 1
    n = len(rows)
    agg = {a: {k: sum(r["got"][a][k] for r in rows) for k in ("repeat", "newarg", "progress")}
           for a in ("A_REF", "B_V2", "C_NAME", "D_NEG")}
    print("")
    print("## 집계  n=%d" % n)
    for a in ("A_REF", "B_V2", "C_NAME", "D_NEG"):
        print("  %-7s repeat %2d · newarg %2d · progress %2d" %
              (a, agg[a]["repeat"], agg[a]["newarg"], agg[a]["progress"]))
    same_all = sum(1 for r in rows
                   if len({r["got"][a]["tool"] + r["got"][a]["args"]
                           for a in ("A_REF", "B_V2", "C_NAME", "D_NEG")}) == 1)
    if same_all == n:
        v = "⛔**계기 무효** — 네 팔이 전부 같은 호출(결과 인용 금지)"
    elif agg["B_V2"]["repeat"] * 2 < agg["A_REF"]["repeat"]:
        v = "**문면이 산다** ⇒ 처방 = `T2_NOREC_BRANCH` ON (이미 저작된 v2·새 결정론 0)"
    elif agg["C_NAME"]["repeat"] < agg["B_V2"]["repeat"]:
        v = "문면만으론 부족 — **다음 행동 이름**이 본체([[64]])"
    elif agg["A_REF"]["repeat"] == agg["B_V2"]["repeat"] == agg["C_NAME"]["repeat"] == n:
        v = "전부 repeat — 문면 축이 아니다 ⇒ [[62]]③ 그 단계 결정론 검토"
    else:
        v = "혼합 — 컷별 표를 읽고 판단(집계 직행 금지·[[08]])"
    print("판정: %s" % v)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..",
                       "reports", "facet_rft_2026", "x379_norec_wording.json")
    io.open(os.path.normpath(out), "w", encoding="utf-8").write(
        json.dumps({"rows": rows, "n": n, "agg": agg, "verdict": v},
                   ensure_ascii=False, indent=1))
    print("원자료: %s" % os.path.normpath(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
