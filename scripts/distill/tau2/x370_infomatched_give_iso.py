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
ASK = ("You are the bank's support agent. Continue this conversation. If the customer must run "
       "something themselves, hand them the tool. Reply with the tool call you make now, as "
       "NAME(arguments), and nothing else.")


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


def emitted_give(ans, name):
    """접미사까지 정확한 이름으로 give 를 냈는가(요건 2)."""
    s = " ".join(str(ans or "").split())
    return (GIVE in s) and (name in s)


def det(body, maxtok=220):
    a = str((chat(body, None, 0.0, maxtok) or {}).get("content") or "")
    b = str((chat(body, None, 0.0, maxtok) or {}).get("content") or "")
    return a, (a.strip() == b.strip())


def registries():
    """env 런타임 레지스트리 — **손님-측** discoverable 만이 give 의 정당한 대상이다."""
    from tau2.domains.banking_knowledge import tools as T
    attr = getattr(T, "DISCOVERABLE_ATTR", "__discoverable__")
    cls = getattr(T, "KnowledgeUserTools", None)
    return set(n for n in dir(cls or ()) if not n.startswith("_")
               and callable(getattr(cls, n, None))
               and getattr(getattr(cls, n), attr, False))


def main():
    # ── give 가 **실재 손님-측 도구**를 향한 컷만 고른다.
    #   ★1차 실행 자기무효(2026-08-18): 이 필터가 없어 `apply_for_credit_card`·
    #     `send_verification_code` 같은 **레지스트리 밖 이름**을 향한 실패한 give 까지 컷으로 잡았고,
    #     태스크도 알파벳 앞 8개(001~008 = A 버킷)가 뽑혀 **표적 가족을 하나도 안 쟀다**.
    #     그 실행은 A=B=D=0(전 팔 무신호)이라 판별력이 없었다 — 결과 인용 금지.
    udisc = registries()
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
    seen, jobs = set(), []
    for c in sorted(cuts, key=lambda x: (x["task"], x["run"])):
        if c["task"] in seen:
            continue
        seen.add(c["task"])
        jobs.append(c)
    jobs = jobs[:8]
    print("x370 v2 · 손님-측 레지스트리 %d종 · 그 이름을 향한 give 컷 %d · 태스크 %d개"
          % (len(udisc), len(cuts), len(jobs)))
    print("판정(사전 고정): A 높고 B 급락 → env 가 답을 흘렸다(격리 무효) · A≈B 높음 → 전달뿐 · "
          "A≈B 낮음 → 그 단계에만 결정론 검토 · D_NEG≈A → 계기 무효\n")
    if not jobs:
        print("⚠give 컷이 하나도 없다 — 이 프로브는 성립하지 않는다(그 자체가 결과다)")
        return 1

    neg = convo(jobs[-1]["sim"], jobs[-1]["cut"]) if len(jobs) > 1 else ""
    res = []
    for j in jobs:
        base = convo(j["sim"], j["cut"])
        # ★P_HINT = **양성통제**(1차 실행에서 빠져 있던 것). 답을 대놓고 알려 줬는데도 못 내면
        #   그것은 모델이 아니라 **계기**(질문 문면·채점)가 고장난 것이다.
        #   *"지표를 만들면 그 지표가 아는 정답을 먼저 맞히는지 한 번 돌려라"*(오늘 교훈 반복).
        arms = {"P_HINT": base + "\n\nsystem: The customer must run %s themselves." % j["name"],
                "A_LIVE": base, "B_NOLEAK": strip_leak(base, j["name"]),
                "D_NEG": neg if j is not jobs[-1] else convo(jobs[0]["sim"], jobs[0]["cut"])}
        row = {"task": j["task"], "run": j["run"], "cut": j["cut"], "name": j["name"],
               "n_lines": len(base.split("\n")),
               "n_stripped": len(base.split("\n")) - len(arms["B_NOLEAK"].split("\n")),
               "arms": {}}
        for k in ("P_HINT", "A_LIVE", "B_NOLEAK", "D_NEG"):
            ans, d = det(arms[k][-9000:] + "\n\n" + ASK)
            row["arms"][k] = {"hit": int(emitted_give(ans, j["name"])), "det": d,
                              "out": " ".join(ans.split())[:200]}
        res.append(row)
        print("── %s/%s cut=%d 이름=%s · 누설 %d줄 제거 · P %d · A %d · B %d · D %d"
              % (j["task"], j["run"][:26], j["cut"], j["name"], row["n_stripped"],
                 row["arms"]["P_HINT"]["hit"], row["arms"]["A_LIVE"]["hit"],
                 row["arms"]["B_NOLEAK"]["hit"], row["arms"]["D_NEG"]["hit"]))
        for k in ("A_LIVE", "B_NOLEAK"):
            print("     %-9s %s" % (k, row["arms"][k]["out"][:120]))

    n = len(res)
    pz = sum(r["arms"]["P_HINT"]["hit"] for r in res)
    a = sum(r["arms"]["A_LIVE"]["hit"] for r in res)
    b = sum(r["arms"]["B_NOLEAK"]["hit"] for r in res)
    d = sum(r["arms"]["D_NEG"]["hit"] for r in res)
    print("\n" + "=" * 96)
    print("n=%d · **P_HINT %d**(양성통제) · A_LIVE %d · **B_NOLEAK %d** · D_NEG %d · "
          "누설 줄이 있던 컷 %d" % (n, pz, a, b, d, sum(1 for r in res if r["n_stripped"])))
    if pz < max(1, int(0.5 * n)):
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
