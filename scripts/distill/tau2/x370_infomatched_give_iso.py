# -*- coding: utf-8 -*-
r"""x370 — **정보-맞춘 격리 재설계**(x338 무효 판정 후속·[[18]] 의무·⛔0①).

## 왜 (2026-08-17 워크플로 판정)

`x338` 의 *"격리 24/24"* 는 **답이 본문 마지막 줄에 있던 정보-과잉 프로브**였다 ⇒ 무효.
같은 사이트의 다른 격리(`x362`)는 네 팔 전부 GIVE 0/2 로 **반대 답**을 냈다.
⇒ C 버킷 결손이 **격리에서도 나는가**는 지금 **측정이 없다**. 그게 갈려야만 [[62]]②③ 이 정해진다:
   격리에서 되면 레버는 **전달뿐** · 격리에서도 실패하면 **그 단계에만** 결정론.

## 설계 요건 3개 (전부 x338 의 실패에서 도출)

1. **답 문장 제거 팔**을 반드시 넣는다. 같은 국면을 유지한 채 **env 가 답을 흘리는 문장만** 뺀다.
   (종전 부정통제는 국면과 답을 **동시에** 바꿔 교란을 못 갈랐다.)
2. **MARKS 를 접미사까지** 검사한다(`submit_cash_back_dispute_0589` — 접두만 맞으면 오적중).
3. 출력 절단을 늘려 **인자를 원자료에 보존**한다(x338 은 `out[:60]` 로 인자를 잘랐다).

## 셀

    A_LIVE     결정 시점의 라이브 대화 축자 (개입 0)            ← 기준선
    B_NOLEAK   같은 대화에서 **답 이름이 든 도구-출력 문장만** 제거 ← 진짜 격리
    D_NEG      **다른 태스크**의 대화(같은 길이대)              ← 부정통제
  지표: `give_discoverable_user_tool` 를 **정확한 도구 이름으로** 방출했는가(접미사 일치).

## 부정통제 쌍 (태스크 선택도 사전 고정)

    task_034  선택이 없는 태스크 · census .904 · 전수 3/3 통과   ← 여기서 실패하면 계기 결함
    task_014  1스텝 · enum 19개가 스키마로 문맥에 있음 · 61 sim 전수 gold reason **0**
              ← 여기서 격리도 실패하면 **능력**(전달 아님)

## 판정 (사전 고정 · 결과보다 먼저 인쇄)

    A_LIVE 높고 B_NOLEAK 급락        → 그 성적은 **env 가 흘린 답**이었다(x338 재현) ⇒ 격리 무효
    A≈B 둘 다 높음                    → 격리에서 된다 ⇒ 레버는 **전달뿐**([[62]]②)
    A≈B 둘 다 낮음                    → 격리에서도 실패 ⇒ **그 단계에만** 결정론 검토([[62]]③)
    D_NEG 가 A 와 비슷                → 계기 무효(대화를 안 읽는다)

실행: T2_PROBE_URL=http://localhost:8141/v1/chat/completions \
      /home/woori/venvs/seka_env/bin/python x370_infomatched_give_iso.py
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

from x216_read_and_offset import chat                             # noqa: E402
import t2_forensic as F                                           # noqa: E402
import x364_eligibility_axis_iso as E                             # noqa: E402

SIMROOT = "/home/woori/scratch/tau2-bench/data/simulations"
GIVE = "give_discoverable_user_tool"
ASK = ("You are the bank's support agent. Continue this conversation. If the customer must "
       "run something themselves, hand them the tool. Make the tool call you would make now.")


def all_sims():
    """리모트 시뮬레이션 전수(영속 여부 무관 — 그 자체가 §5-11 의 취약점이므로 로그에 남긴다)."""
    for d in sorted(os.listdir(SIMROOT)):
        p = os.path.join(SIMROOT, d, "results.json")
        if not os.path.exists(p):
            continue
        try:
            data = json.load(io.open(p, encoding="utf-8"))
        except Exception:
            continue
        for s in (data.get("simulations") or data.get("results") or ()):
            yield d, s


def convo(sim, upto):
    """대화 축자 — 손님·도구 출력·어시스턴트 본문을 **자르지 않고** 잇는다(정보-맞춤)."""
    out = []
    for i, m in enumerate(sim.get("messages") or ()):
        if i >= upto:
            break
        role = m.get("role")
        txt = " ".join(str(m.get("content") or "").split())
        if txt:
            out.append("%s: %s" % (role, txt))
        for tc in (m.get("tool_calls") or ()):
            out.append("assistant calls: %s(%s)"
                       % (F.nameof(tc), json.dumps(F.argsof(tc), ensure_ascii=False)))
    return "\n".join(out)


def strip_leak(text, name):
    """**답 이름이 든 줄만** 뺀다 — 국면은 그대로 두고 누설만 제거(요건 1·정규식 0)."""
    return "\n".join(l for l in text.split("\n") if name not in l)


def emitted_give(msg, name):
    """**방출된 tool_call** 로 채점한다 — 라이브와 같은 채널(v3 수리).

    v1/v2 는 본문 문자열에서 이름을 찾았는데, 도구 스키마를 주면 모델은 본문이 아니라
    `tool_calls` 로 답한다 ⇒ 문자열 채점은 그 자체로 전량 탈락이었다(양성통제 1/8 의 절반 몫).
    ⚠접미사까지 정확히 본다(요건 2): 인자 안의 대상 이름이 **정확히 그것**이어야 한다.
    """
    blob = ""
    for tc in ((msg or {}).get("tool_calls") or ()):
        f = tc.get("function") or tc
        if str(f.get("name") or "") == GIVE:
            blob += str(f.get("arguments") or "")
    return bool(blob) and (name in blob)


def _call(body, tools, maxtok):
    """서버 오류를 **본문까지** 인쇄하고 그 팔만 비운다 — 한 행 때문에 전체를 잃지 않는다.

    ★v5b 실측: 3행째에서 `HTTP 400` 으로 **전체 실행이 죽었다**(도구 목록 자체는 정상 확인).
      계기가 조용히 죽는 것보다 나쁜 것은 **시끄럽게 죽어 앞선 행까지 버리는 것**이다.
    """
    try:
        return chat(body, tools, 0.0, maxtok, None, "required") or {}
    except Exception as e:
        detail = ""
        try:
            detail = e.read().decode()[:300]
        except Exception:
            pass
        print("     ⚠호출 실패(이 팔만 비움): %r %s" % (e, detail), file=sys.stderr, flush=True)
        return {"_err": "%r" % (e,)}


def det(body, tools=None, maxtok=220):
    # ★v4: `tool_choice="required"` — *무언가를 부르라*고만 한다. **무엇을** 부를지는 여전히
    #   모델 몫이고 그것이 측정 대상이다.
    a = _call(body, tools, maxtok)
    b = _call(body, tools, maxtok)
    same = (json.dumps(a.get("tool_calls"), sort_keys=True, default=str)
            == json.dumps(b.get("tool_calls"), sort_keys=True, default=str)
            and str(a.get("content") or "").strip() == str(b.get("content") or "").strip())
    return a, same


def registries():
    """env 런타임 레지스트리 — **손님-측** discoverable 만이 give 의 정당한 대상이다."""
    from tau2.domains.banking_knowledge import tools as T
    attr = getattr(T, "DISCOVERABLE_ATTR", "__discoverable__")
    cls = getattr(T, "KnowledgeUserTools", None)
    return set(n for n in dir(cls or ()) if not n.startswith("_")
               and callable(getattr(cls, n, None))
               and getattr(getattr(cls, n), attr, False))


def agent_tool_specs():
    """라이브 에이전트가 **실제로 갖는 도구 목록**을 env 에서 만들어 준다(정보-맞춤의 핵심).

    ★v2 자기무효(2026-08-18): v1·v2 는 대화를 **평문으로만** 주고 *"도구 호출을 적어라"* 라고
      물었다. 라이브 에이전트는 `give_discoverable_user_tool` 의 **스키마를 갖고 있다** — 그것을
      빼면 정보-맞춤이 아니라 **다른 방향의 정보-결핍**이다. 양성통제 `P_HINT` 가 **1/8** 로
      떨어져 그 사실을 인쇄했다(가드가 잡았다). ⇒ env 의 공개 메서드에서 스키마를 만들어 싣는다.
    ⚠도구 이름·설명·인자는 **env 에서 읽는다**(하드코딩 0·도메인 어휘 0·[[05]]).
    ⚠discoverable 은 잠겨 있으므로 **뺀다** — 라이브도 unlock 전에는 안 보인다.
    """
    import inspect
    from tau2.domains.banking_knowledge import tools as T
    attr = getattr(T, "DISCOVERABLE_ATTR", "__discoverable__")
    cls = getattr(T, "KnowledgeTools", None)
    specs = []
    for n in sorted(dir(cls or ())):
        if n.startswith("_"):
            continue
        m = getattr(cls, n, None)
        if not callable(m) or getattr(m, attr, False):
            continue
        try:
            sig = inspect.signature(m)
        except Exception:
            continue
        props, req = {}, []
        for pn, p in sig.parameters.items():
            if pn == "self":
                continue
            props[pn] = {"type": "string"}
            if p.default is inspect.Parameter.empty:
                req.append(pn)
        doc = " ".join(str(m.__doc__ or "").split())[:120]   # ★v5b: 요청 크기 축소
        specs.append({"type": "function", "function": {
            "name": n, "description": doc,
            "parameters": {"type": "object", "properties": props, "required": req}}})
    return specs


def main():
    # ── give 가 **실재 손님-측 도구**를 향한 컷만 고른다.
    #   ★1차 실행 자기무효(2026-08-18): 이 필터가 없어 `apply_for_credit_card`·
    #     `send_verification_code` 같은 **레지스트리 밖 이름**을 향한 실패한 give 까지 컷으로 잡았고,
    #     태스크도 알파벳 앞 8개(001~008 = A 버킷)가 뽑혀 **표적 가족을 하나도 안 쟀다**.
    #     그 실행은 A=B=D=0(전 팔 무신호)이라 판별력이 없었다 — 결과 인용 금지.
    udisc = registries()
    TOOLS = agent_tool_specs()
    cuts = []
    for run, s in all_sims():
        tid = str(s.get("task_id") or (s.get("task") or {}).get("id") or "")
        msgs = s.get("messages") or []
        for i, m in enumerate(msgs):
            for tc in (m.get("tool_calls") or ()):
                if F.nameof(tc) == GIVE:
                    nm = str(F.argsof(tc).get("discoverable_tool_name") or "")
                    if nm in udisc:            # ★레지스트리 실재만
                        cuts.append({"run": run, "task": tid, "cut": i, "name": nm,
                                     "sim": s, "seed": str(s.get("seed"))})
                    break
    # gold 가 give 를 요구하는 태스크를 **우선**한다(분석 전용 — 레버는 이 목록을 안 쓴다).
    #   ★v2 결함: `jobs[:8]` 이 번호순 앞 8개(001~014)를 집어 표적 가족(019~022)을 하나도 못 쟀다.
    want = set()
    for fn in sorted(os.listdir(E.M.TASKS_DIR)):
        if not fn.endswith(".json"):
            continue
        t = json.load(io.open(os.path.join(E.M.TASKS_DIR, fn), encoding="utf-8"))
        if any(str(a.get("name") or "") == GIVE
               for a in ((t.get("evaluation_criteria") or {}).get("actions") or ())):
            want.add(fn[:-5])
    seen, jobs = set(), []
    for c in sorted(cuts, key=lambda x: (0 if x["task"] in want else 1, x["task"], x["run"])):
        if c["task"] in seen:
            continue
        seen.add(c["task"])
        jobs.append(c)
    print("   gold 가 give 를 요구하는 태스크 %d개 · 그중 라이브 give 컷이 있는 것 %d개"
          % (len(want), sum(1 for j in jobs if j["task"] in want)))
    jobs = jobs[:8]
    print("x370 v4 · 손님-측 레지스트리 %d종 · env 도구 스키마 %d종(정보-맞춤) · give 컷 %d"
          % (len(udisc), len(TOOLS), len(cuts)))
    print("판정(사전 고정): A 높고 B 급락 → env 가 답을 흘렸다(격리 무효) · A≈B 높음 → 전달뿐 · "
          "A≈B 낮음 → 그 단계에만 결정론 검토 · D_NEG≈A → 계기 무효\n")
    if not jobs:
        print("⚠give 컷이 하나도 없다 — 이 프로브는 성립하지 않는다(그 자체가 결과다)")
        return 1

    neg = convo(jobs[-1]["sim"], jobs[-1]["cut"]) if len(jobs) > 1 else ""
    res = []
    for j in jobs:
        base = convo(j["sim"], j["cut"])
        # 팔 = (본문, 도구목록). 양성통제가 **둘**이다:
        #   `P_ONLY`  답을 알려 주고 **도구 목록도 give 하나로** 제한 → 하네스·채점이 살아 있는가
        #   `P_HINT`  답을 알려 주되 도구는 **28종 전부** → 고를 수 있는가
        #   ★v5 수리(2026-08-18): v3·v4 는 이 자리의 문자열 치환이 **조용히 안 먹어** 도구 목록을
        #     아예 안 실었다(그래서 전 팔이 산문·전 팔 0). 내 진단 *"납작한 대화 형태 탓"* 도
        #     그래서 틀렸다 — 원인은 **패치가 적용 안 된 것**이었다. 치환 대신 앵커 편집으로 고치고,
        #     확인은 *substring 존재*가 아니라 **바뀐 코드를 직접 읽어서** 한다.
        hint = base + "\n\nsystem: The customer must run %s themselves." % j["name"]
        only = [t for t in TOOLS if (t.get("function") or {}).get("name") == GIVE]
        arms = {"P_ONLY": (hint, only), "P_HINT": (hint, TOOLS), "A_LIVE": (base, TOOLS),
                "B_NOLEAK": (strip_leak(base, j["name"]), TOOLS),
                "D_NEG": ((neg if j is not jobs[-1] else convo(jobs[0]["sim"], jobs[0]["cut"])),
                          TOOLS)}
        row = {"task": j["task"], "run": j["run"], "cut": j["cut"], "name": j["name"],
               "n_lines": len(base.split("\n")),
               "n_stripped": len(base.split("\n")) - len(arms["B_NOLEAK"][0].split("\n")),
               "arms": {}}
        for k in ("P_ONLY", "P_HINT", "A_LIVE", "B_NOLEAK", "D_NEG"):
            _body, _tools = arms[k]
            ans, d = det(_body[-9000:] + "\n\n" + ASK, _tools)
            _names = ",".join(str((tc.get("function") or tc).get("name") or "")
                              for tc in (ans.get("tool_calls") or ())) or "-"
            row["arms"][k] = {
                "hit": int(emitted_give(ans, j["name"])), "det": d, "called": _names,
                "err": ans.get("_err"),
                "out": ("CALLS=" + _names + " | "                      # ★이름을 **먼저**
                        + json.dumps(ans.get("tool_calls"), default=str)[:300]
                        + " | TEXT " + " ".join(str(ans.get("content") or "").split())[:200])}
        res.append(row)
        print("── %s/%s cut=%d 이름=%s · 누설 %d줄 · Ponly %d · Phint %d · A %d · B %d · D %d"
              % (j["task"], j["run"][:24], j["cut"], j["name"], row["n_stripped"],
                 row["arms"]["P_ONLY"]["hit"], row["arms"]["P_HINT"]["hit"],
                 row["arms"]["A_LIVE"]["hit"], row["arms"]["B_NOLEAK"]["hit"],
                 row["arms"]["D_NEG"]["hit"]))
        for k in ("P_ONLY", "P_HINT", "A_LIVE", "B_NOLEAK"):
            print("     %-9s 부른 도구=%s" % (k, row["arms"][k]["called"]))

    n = len(res)
    po = sum(r["arms"]["P_ONLY"]["hit"] for r in res)
    pz = sum(r["arms"]["P_HINT"]["hit"] for r in res)
    a = sum(r["arms"]["A_LIVE"]["hit"] for r in res)
    b = sum(r["arms"]["B_NOLEAK"]["hit"] for r in res)
    d = sum(r["arms"]["D_NEG"]["hit"] for r in res)
    print("\n" + "=" * 96)
    print("n=%d · **P_ONLY %d**(도구1개) · P_HINT %d(도구28개) · A_LIVE %d · **B_NOLEAK %d** · "
          "D_NEG %d · 누설 줄이 있던 컷 %d"
          % (n, po, pz, a, b, d, sum(1 for r in res if r["n_stripped"])))
    if po < max(1, int(0.5 * n)):
        print("   → ⛔**계기 무효**: 도구를 하나만 줘도 못 부른다 ⇒ 하네스·채점을 고치기 전엔 "
              "어떤 0 도 결손으로 읽지 마라")
    elif pz < max(1, int(0.5 * n)):
        print("   → **결손의 이름 = 고르기**: 도구 1개면 부르는데 28개 중에서는 못 고른다 "
              "(답을 알려 줘도) ⇒ 전달이 아니라 **선택** 축이다(C519 와 같은 방향)")
    elif False:
        print("   → ⛔**계기 무효**: 답을 알려 줘도 못 낸다 ⇒ 질문 문면·채점을 고치기 전엔 "
              "다른 팔의 0 을 결손으로 읽지 마라(1차 실행이 정확히 이 함정에 빠졌다)")
    elif d >= a and a:
        print("   → ⛔계기 무효(부정통제가 같이 맞는다)")
    elif a and not b:
        print("   → **x338 재현**: 그 성적은 env 가 흘린 답이었다 ⇒ 격리 24/24 무효 확정")
    elif b >= max(1, int(0.75 * n)):
        print("   → 격리에서 된다 ⇒ 레버는 **전달뿐**([[62]]②)")
    else:
        print("   → 격리에서도 실패 ⇒ **그 단계에만** 결정론 검토([[62]]③)")
    out = os.path.join(E.REPORTS, "x370_infomatched_give.json")
    with io.open(out, "w", encoding="utf-8") as f:
        f.write(json.dumps(res, ensure_ascii=False, indent=1, default=str))
    print("저장: %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
