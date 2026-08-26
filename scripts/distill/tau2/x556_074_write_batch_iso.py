# -*- coding: utf-8 -*-
r"""x556 — 074 의 남은 한 칸: **한 턴에 네 건을 쓰면 계좌 값이 옮겨붙나**

## 관측 (x555 · t7362 C_scope task_074#s626729 · 2026-08-26)
우리 도구 `get_atm_fee_discrepancies` 의 net 은 **4/4 정확**하다(그 sim 의 msg[48]·[52]·[56]·[60]
축자에서 difference 를 부호대로 더한 값 = 27.00 / 14.50 / 4.75 / 3.70). 모델은 msg[73] 에서
**한 턴에 네 건**을 병렬로 냈고 `_2`·`_3`·`_4` 는 정확한데 `_1` 만 **14.50**(= `_2` 의 값)이었다.
산수 오류가 아니다 — 음수를 낀 세 계좌가 전부 부호합으로 맞았다(`x542` 절댓값-합 기전 **미재현**).
남은 형상은 **계좌 간 값 전이**이고, 그 자리가 네 건이 한 턴에 몰린 결정점이다([[65]]).

## 무엇을 재나 — 바뀌는 것은 **요청한 계좌 목록 하나**뿐
문맥은 라이브 전 접두를 **그대로** 쓴다(⛔짧은 창은 결손을 지운다 — 오늘 x551 이 그 자리에서
부풀었다). 팔 사이의 차이는 *"몇 건을 한 턴에 요구하는가 · 어느 순서로"* 하나다.

    A_all4    네 계좌를 한 번에 (_1.._4)      ← 라이브. **여기서 `_1`=14.5 가 재현돼야 판정한다**
    B_one1    `_1` 하나만
    N_one2    `_2` 하나만                     ← 부정 통제: 한-건 팔이 그냥 첫 수를 베끼는 게 아님
    N_rev     네 계좌를 **역순**으로 (_4.._1) ← 위치 효과인가 개수 효과인가

## 채점 — 닫힌 술어 · gold 미접촉([[23]])
표적 금액은 gold 가 아니라 **우리 도구가 그 sim 에서 낸 difference 줄의 부호합**이다.
그 줄은 우리가 만든 문자열이므로 파싱은 [[59]] 허용역이다(도메인 텍스트 해석 아님).
답에서 `account_id`/`amount` 쌍을 긁어 계좌별로 맞댄다.

## [[62]] 4문
  ① 결손 측정 = x555 per-step(궤적 축자·mutation_diff 3 MATCHED / 1 WRONGARG / 1 MISSING).
  ② 재료는 **이미 닿아 있다**(우리 도구 4/4 정확) ⇒ 이 자리의 레버는 계산이 아니라 **부하**뿐이다.
  ③ 사라지는 모델 판단 0 — 금액을 고르는 것은 끝까지 모델이다. 엔진이 합을 내지 않는다.
  ④ 순위·최댓값·*"정답은 X"* 0. 표적은 **채점에만** 쓰고 프롬프트에는 안 들어간다.

⚠라이브 에이전트는 뷰 압축본을 봤고 이 프로브는 **원 궤적 접두**를 쓴다. 둘이 다르면 A_all4 가
  재현하지 못한다 — 그러면 판정하지 않는다([[62]] 2b). 재현 여부를 먼저 인쇄한다([[78]]).

사용: (리모트·cwd=scripts/distill/tau2) py -3 x556_074_write_batch_iso.py --port 8140
      --wiring-only 로 모델 없이 문맥 크기·표적·요청부만 확인(무료).
"""

import argparse
import json
import os
import re
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import t2_forensic as F                                              # noqa: E402

MODEL = "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8"
NL = chr(10)
# 우리 도구의 반환 문면(`difference $X`)과 계좌 id — 둘 다 **우리가 만든 문자열**이다.
RX_DIFF = re.compile(r"difference \$(-?[0-9]+(?:\.[0-9]+)?)")
RX_ACC = re.compile(r"(chk_[a-z0-9]+_\d)")
RX_PAIR = re.compile(r"account_id\"?\s*[:=]\s*\"?(chk_[a-z0-9]+_\d)\"?"
                     r".{0,120}?amount\"?\s*[:=]\s*\"?(-?[0-9]+(?:\.[0-9]+)?)", re.S)


def gen(port, body, maxtok=420, temp=0.0):
    payload = {"model": MODEL, "temperature": temp, "max_tokens": maxtok,
               "messages": [{"role": "user", "content": body}]}
    req = urllib.request.Request("http://127.0.0.1:%d/v1/chat/completions" % port,
                                 data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read().decode("utf-8"))["choices"][0]["message"]["content"]


def write_index(ms, tool):
    """그 도구를 **실제로 실행한** 첫 메시지 인덱스(= 결정점). 없으면 None."""
    for i, m in enumerate(ms):
        for tc in (m.get("tool_calls") or ()):
            nm = F.inner_name(F.argsof(tc)) or F.nameof(tc)
            if tool in str(nm) and "unlock" not in str(F.nameof(tc)):
                return i
    return None


def targets(ms, upto):
    """계좌 → **우리 도구가 낸 difference 의 부호합**. 자기 출력 파싱([[59]] 허용역).

    ⚠계좌 id 는 **출력 본문에 없다** — 그 도구의 반환 문면은 거래 id 만 담는다. 초판이 본문에서
      계좌를 찾다가 표적을 하나도 못 세웠다. id 는 **바로 앞 호출의 인자**에 있다(라이브가 계좌마다
      한 번씩 부른다). 짝짓기는 위치 술어이지 내용 판단이 아니다([[25]] 모르면 안 쓴다).
    """
    out = {}
    for i, m in enumerate(ms[:upto]):
        if m.get("role") != "tool":
            continue
        c = " ".join(str(m.get("content") or "").split())
        if "difference $" not in c:
            continue
        acc = None
        for j in range(i - 1, -1, -1):
            tcs = ms[j].get("tool_calls") or ()
            for tc in tcs:
                ar = F.argsof(tc)
                inner = ar.get("arguments")
                if isinstance(inner, str):
                    try:
                        inner = json.loads(inner)
                    except Exception:
                        inner = {}
                cand = (inner or {}).get("account_id") if isinstance(inner, dict) else None
                cand = cand or ar.get("account_id")
                if cand:
                    acc = cand
                    break
            if acc or tcs:
                break
        vals = [float(v) for v in RX_DIFF.findall(c)]
        if acc and vals:
            out[acc] = round(sum(vals), 2)
    return out


def render(ms, upto, cap=0):
    """전 접두를 역할 표시와 함께 텍스트로. 자르지 않는다(⛔짧은 창 금지)."""
    parts = []
    for m in ms[:upto]:
        role = m.get("role")
        c = " ".join(str(m.get("content") or "").split())
        tcs = []
        for tc in (m.get("tool_calls") or ()):
            tcs.append("%s(%s)" % (F.nameof(tc), json.dumps(F.argsof(tc), ensure_ascii=False)))
        if tcs:
            c = (c + " " if c else "") + "TOOL_CALLS: " + " ".join(tcs)
        if not c:
            continue
        parts.append("[%s] %s" % (role, c[:cap] if cap else c))
    return NL.join(parts)


def ask(accs, tool):
    """요청부 — **계좌 목록 하나만** 팔마다 다르다."""
    lst = ", ".join(accs)
    return (NL + NL + "Issue the fee_refund credit now for: " + lst + "."
            + NL + "Reply with ONLY the tool call arguments, one JSON object per account, "
            "each with exactly the keys account_id, amount, credit_type. Nothing else.")


def score(reply, want):
    """답 → {계좌: 금액}. 닫힌 술어(정규식 회수·해석 0)."""
    got = {}
    for a, v in RX_PAIR.findall(str(reply or "")):
        got.setdefault(a, v)
    return {a: (got.get(a), want.get(a),
                got.get(a) is not None and abs(float(got[a]) - want[a]) < 1e-6)
            for a in want}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8140)
    ap.add_argument("--tag", default="bank_t7362_C_scope_20260826")
    ap.add_argument("--sim", default="task_074#s626729")
    ap.add_argument("--tool", default="apply_checking_account_credit")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--wiring-only", action="store_true")
    a = ap.parse_args(argv)

    sims = [s for s in F.sims(a.tag) if F.simtag(s) == a.sim]
    if not sims:
        print("그 sim 이 없다 (%s / %s)" % (a.tag, a.sim), file=sys.stderr)
        return 2
    ms = sims[0].get("messages") or []
    w = write_index(ms, a.tool)
    if w is None:
        print("그 도구의 실행 자리를 못 찾았다 — 판정하지 않는다([[25]])", file=sys.stderr)
        return 2
    want = targets(ms, w)
    ctx = render(ms, w)
    accs = sorted(want)
    print("# x556 — 074 한-턴 다건 쓰기 격리")
    print("  결정점 msg[%d] · 전 접두 %d msg · %d자" % (w, w, len(ctx)))
    print("  표적(우리 도구 difference 부호합·gold 미접촉): %s"
          % json.dumps(want, ensure_ascii=False))
    live = {}
    for tc in (ms[w].get("tool_calls") or ()):
        ar = F.argsof(tc)
        inner = ar.get("arguments")
        if isinstance(inner, str):
            try:
                inner = json.loads(inner)
            except Exception:
                inner = {}
        if isinstance(inner, dict) and inner.get("account_id"):
            live[inner["account_id"]] = inner.get("amount")
    print("  라이브가 그 턴에 낸 것: %s" % json.dumps(live, ensure_ascii=False))
    if not (want and accs):
        print("  ⛔표적을 못 세웠다 — 판정하지 않는다.")
        return 3

    plan = [("A_all4", accs), ("B_one1", accs[:1]), ("N_one2", accs[1:2]),
            ("N_rev", list(reversed(accs)))]
    print()
    print("## 팔 (요청한 계좌 목록만 다르다)")
    for nm, aa in plan:
        print("  %-8s %s" % (nm, ", ".join(aa)))
    if a.wiring_only:
        print()
        print("(--wiring-only · 모델 호출 0)")
        print("--- 요청부 예시(A_all4) ---")
        print(ask(accs, a.tool).strip())
        print("--- 문맥 꼬리 400자 ---")
        print(ctx[-400:])
        return 0

    print()
    print("%-8s %-5s %-42s %s" % ("팔", "temp", "계좌별 (낸 값 ↔ 표적)", "맞춘 수"))
    print("-" * 120)
    tally = {}
    for nm, aa in plan:
        body = ctx + ask(aa, a.tool)
        for tp, n in ((0.0, 1), (a.temp, a.n)):
            for _ in range(n):
                try:
                    rep = " ".join(str(gen(a.port, body, 420, tp)).split())
                except Exception as e:
                    print("%-8s %-5s 호출 실패: %r" % (nm, tp, e))
                    continue
                sc = score(rep, {k: want[k] for k in aa})
                cells = " ".join("%s:%s%s" % (k.split("_")[-1], v[0],
                                              "=" if v[2] else "≠%s" % v[1])
                                 for k, v in sorted(sc.items()))
                ok = sum(1 for v in sc.values() if v[2])
                tally.setdefault((nm, tp), [0, 0])
                tally[(nm, tp)][0] += ok
                tally[(nm, tp)][1] += len(sc)
                print("%-8s %-5s %-42s %d/%d" % (nm, tp, cells[:42], ok, len(sc)))
    print()
    print("## 판정")
    a0 = tally.get(("A_all4", 0.0), [0, 0])
    print("  A_all4(temp 0) %d/%d — 라이브 오답(`_1`=14.5) 재현 %s"
          % (a0[0], a0[1], "**됨**" if a0[0] < a0[1] else "⛔안 됨 ⇒ 판정하지 마라([[62]] 2b)"))
    for nm, _ in plan:
        row = " · ".join("temp %.1f: %d/%d" % (t, v[0], v[1])
                         for (n2, t), v in sorted(tally.items()) if n2 == nm)
        print("  %-8s %s" % (nm, row))
    print()
    print("⚠N_one2 가 B_one1 과 같이 맞으면 한-건 팔의 이득은 **계좌 특정이 아니라 건수**다.")
    print("⚠N_rev 가 `_1` 을 고치면 그것은 **위치 효과**이지 건수 효과가 아니다([[57]]).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
