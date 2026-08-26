# -*- coding: utf-8 -*-
r"""x558 — 079 의 남은 칸: **자기가 고른 옵션의 값을 문서에서 못 집는다**

## 관측 (x555·x557 · t7362 A_ctl task_079#s626729 · 2026-08-26)
손님이 msg[108] 에서 *"For my **Evergreen Account** card, I need it **as fast as possible**"* 라고
하자 모델은 msg[109]·[115] 에서 `delivery_option=RUSH` 로 재주문했는데 `delivery_fee` 는 **0** 을
썼다(gold 35). 그 값은 **문맥에 있었다** — 우리가 msg[84]·[94] 로 배달한 티어 문서 축자:

    PREMIUM TIER:
      - Free expedited shipping on all replacements (delivery_fee: $0 for both STANDARD and EXPEDITED)
      - Rush shipping available (delivery_fee: $35)

**인접한 두 줄**이고 앞 줄이 $0, 뒷 줄이 $35 다. 모델은 RUSH 를 고르고 앞 줄의 값을 썼다.
계좌 간 전이가 아니라 **옵션 ↔ 값 짝짓기 실패**다(내 첫 '방송' 가설은 이 자리에서 기각됐다).

## 종점 — **자기 정합**(gold 미접촉·[[23]])
gold 를 열지 않는다. 모델이 **스스로 고른** `delivery_option` 에 대해, **우리가 배달한 문서가
그 옵션에 대해 명시한 값**을 `delivery_fee` 에 썼는가만 본다. 문서가 문맥에 있으므로 이 판정은
환경 텍스트만으로 닫힌다. 옵션→값 표는 배달된 그 블록에서 **축자로** 뽑는다.

## 팔 — 바뀌는 것은 **표면화 한 줄**뿐
    A_asis     라이브 문맥 그대로                    ← 재현 게이트(RUSH 인데 fee 0 이 나와야 한다)
    B_optline  같은 블록에서 **옵션별 fee 줄만** 축자로 한 번 더 얹는다(고르기 0·값 계산 0)
    N_len      길이만 맞춘 선택 무관 문장([[57]])

## [[62]] 4문
  ① 결손 측정 = x555/x557 per-step(궤적 축자·`mutation_diff` WRONGARG 2 · MISSING 1).
  ② 격리에서 모델이 성공하는가 — **이 프로브가 그것을 잰다**. 재료는 이미 닿아 있으므로
     레버 후보는 **전달 형태(표면화)** 뿐이고 계산·선택은 하지 않는다.
  ③ 사라지는 모델 판단 0 — 어느 옵션을 고를지도, 어떤 값을 쓸지도 끝까지 모델이다.
  ④ 엔진 출력에 최댓값·*"정답은 X"* 0. 표적은 **채점에만** 쓰고 프롬프트엔 안 들어간다.

사용: (리모트·cwd=scripts/distill/tau2) py -3 x558_option_fee_pairing_iso.py --port 8140
      --wiring-only 로 모델 없이 문맥·표·요청부만 확인(무료).
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
# 배달된 문서의 fee 줄 — **환경이 쓴 문자열**이다(우리가 고른 값이 아니다).
RX_FEE = re.compile(r"delivery_fee:\s*\$(\d+(?:\.\d+)?)([^)]*)")
RX_OPT = re.compile(r"\b(STANDARD|EXPEDITED|RUSH)\b", re.I)
RX_OUT_OPT = re.compile(r"delivery_option\"?\s*[:=]\s*\"?([A-Z]+)", re.I)
RX_OUT_FEE = re.compile(r"delivery_fee\"?\s*[:=]\s*\"?(-?\d+(?:\.\d+)?)", re.I)


def gen(port, body, maxtok=300, temp=0.0):
    payload = {"model": MODEL, "temperature": temp, "max_tokens": maxtok,
               "messages": [{"role": "user", "content": body}]}
    req = urllib.request.Request("http://127.0.0.1:%d/v1/chat/completions" % port,
                                 data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read().decode("utf-8"))["choices"][0]["message"]["content"]


def tier_block(ms, upto, tier):
    """배달된 문서에서 그 티어 절을 **축자로** 잘라 온다. 없으면 ''."""
    for m in reversed(ms[:upto]):
        if m.get("role") != "tool":
            continue
        c = " ".join(str(m.get("content") or "").split())
        i = c.find(tier + " TIER:")
        if i < 0:
            continue
        j = min([k for k in (c.find(t + " TIER:", i + 5)
                             for t in ("ENTRY", "MID", "PREMIUM", "ELITE")) if k > i] or [len(c)])
        return c[i:j].strip()
    return ""


def option_fees(block):
    """티어 절 → {옵션: 값}. **불릿 단위**로 읽는다(판단 0·계산 0).

    ⚠초판은 `delivery_fee: $X` **뒤쪽**에서만 옵션 낱말을 찾고 대문자만 봤다. 그래서
      `- Rush shipping available (delivery_fee: $35)` 를 **통째로 놓쳤다**(옵션이 앞에 있고
      소문자 `Rush` 다) — 정작 이 프로브가 재려는 그 줄이다. 배선 검증이 잡았다.
      ⇒ 불릿 하나를 통으로 보고 대소문자를 무시한다.
    """
    out = {}
    for b in re.split(r"\s+-\s+", block or ""):
        mm = RX_FEE.search(b)
        if not mm:
            continue
        for o in set(x.upper() for x in RX_OPT.findall(b)):
            out.setdefault(o, mm.group(1))
    return out


def render(ms, upto):
    """라이브 **생성 뷰**로 만든 뒤 텍스트화 — 원 접두 그대로면 30만자라 문맥에 안 들어간다.

    ⚠전 접두를 쓰라는 규율은 *창을 짧게 잘라 결손을 지우지 마라*는 뜻이지, 라이브가 본 적 없는
      원문을 넣으라는 뜻이 아니다. 라이브는 `T2_VIEW_COMPACT` 로 압축된 뷰를 봤고, 그 압축은
      정본 `t2_gate_patch._compact_view` 다 — `x467.compact_view_dicts` 로 그대로 부른다([[67]]).
    """
    try:
        import x467_policy_boolean_doc_iso as X467
        ms = X467.compact_view_dicts(list(ms[:upto]))[0]
        upto = len(ms)
    except Exception as e:
        print("[warn] 뷰 압축 실패(원문 사용): %r" % (e,), file=sys.stderr)
    parts = []
    for m in ms[:upto]:
        c = " ".join(str(m.get("content") or "").split())
        tcs = ["%s(%s)" % (F.nameof(tc), json.dumps(F.argsof(tc), ensure_ascii=False))
               for tc in (m.get("tool_calls") or ())]
        if tcs:
            c = (c + " " if c else "") + "TOOL_CALLS: " + " ".join(tcs)
        if c:
            parts.append("[%s] %s" % (m.get("role"), c))
    return NL.join(parts)


ASK = (NL + NL + "Place the replacement card order the customer just asked for, for that one "
       "account. Reply with ONLY the tool call arguments as a single JSON object with the keys "
       "account_id, card_design, delivery_option, delivery_fee, design_fee, reason. Nothing else.")


def arms(block, fees):
    """팔 = **표면화 한 줄**. 값은 배달된 절에서 축자로 오고 엔진은 고르지 않는다."""
    line = "; ".join("%s = $%s" % (o, fees[o]) for o in sorted(fees))
    b = (NL + NL + "[fee lines from the tier section already in front of you, one per shipping "
         "option, verbatim] " + line + ".")
    n = (NL + NL + "[note] the tier section already in front of you was retrieved earlier in this "
         "conversation and has not changed since; treat it as current and complete.")
    return [("A_asis", ""), ("B_optline", b), ("N_len", n)]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8140)
    ap.add_argument("--tag", default="bank_t7362_A_ctl_20260826")
    ap.add_argument("--task", default="task_079")
    ap.add_argument("--tier", default="PREMIUM")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--wiring-only", action="store_true")
    a = ap.parse_args(argv)

    sims = [s for s in F.sims(a.tag) if F.task_id(s) == a.task]
    if not sims:
        print("그 sim 이 없다", file=sys.stderr)
        return 2
    ms = sims[0].get("messages") or []
    # 결정점 = 손님이 빠른 배송을 요구한 **뒤** 첫 재주문 자리
    w = None
    for i, m in enumerate(ms):
        for tc in (m.get("tool_calls") or ()):
            nm = F.inner_name(F.argsof(tc)) or F.nameof(tc)
            if "order_debit_card" in str(nm) and i > 106:
                w = i
                break
        if w:
            break
    if w is None:
        print("재주문 자리를 못 찾았다 — 판정하지 않는다([[25]])", file=sys.stderr)
        return 2

    blk = tier_block(ms, w, a.tier)
    fees = option_fees(blk)
    ctx = render(ms, w)
    live = {}
    for tc in (ms[w].get("tool_calls") or ()):
        ar = F.argsof(tc)
        inner = ar.get("arguments")
        if isinstance(inner, str):
            try:
                inner = json.loads(inner)
            except Exception:
                inner = {}
        if isinstance(inner, dict) and inner.get("delivery_option"):
            live = inner
    print("# x558 — 079 옵션↔값 짝짓기 격리")
    print("  결정점 msg[%d] · 전 접두 %d msg · %d자" % (w, w, len(ctx)))
    print("  배달된 %s TIER 절 %d자" % (a.tier, len(blk)))
    print("  그 절이 옵션별로 말하는 값: %s" % json.dumps(fees, ensure_ascii=False))
    print("  라이브가 낸 것: option=%s fee=%s design=%s"
          % (live.get("delivery_option"), live.get("delivery_fee"), live.get("card_design")))
    if not fees:
        print("  ⛔옵션별 값을 못 세웠다 — 판정하지 않는다.")
        return 3

    plan = arms(blk, fees)
    print()
    print("## 팔 (표면화 한 줄만 다르다)")
    for nm, add in plan:
        print("  %-10s +%d자" % (nm, len(add)))
    if a.wiring_only:
        print()
        print("--- 배달된 절 축자 ---")
        print(blk[:700])
        for nm, add in plan:
            if add:
                print("--- %s 추가분 ---%s" % (nm, add))
        print("--- 요청부 ---" + ASK)
        return 0

    print()
    print("%-10s %-5s %-10s %-8s %-10s %s" % ("팔", "temp", "option", "fee", "정합?", "design"))
    print("-" * 100)
    tally = {}
    for nm, add in plan:
        body = ctx + add + ASK
        for tp, n in ((0.0, 1), (a.temp, a.n)):
            for _ in range(n):
                try:
                    rep = " ".join(str(gen(a.port, body, 300, tp)).split())
                except Exception as e:
                    print("%-10s %-5s 호출 실패: %r" % (nm, tp, e))
                    continue
                mo = RX_OUT_OPT.search(rep)
                mf = RX_OUT_FEE.search(rep)
                opt = (mo.group(1).upper() if mo else None)
                fee = (mf.group(1) if mf else None)
                want = fees.get(opt)
                ok = (want is not None and fee is not None
                      and abs(float(fee) - float(want)) < 1e-6)
                tally.setdefault((nm, tp), [0, 0])
                tally[(nm, tp)][0] += 1 if ok else 0
                tally[(nm, tp)][1] += 1
                dm = re.search(r"card_design\"?\s*[:=]\s*\"?([A-Z]+)", rep, re.I)
                print("%-10s %-5s %-10s %-8s %-10s %s"
                      % (nm, tp, opt or "-", fee or "-",
                         "O" if ok else ("X(문서 %s)" % want if want else "?"),
                         dm.group(1) if dm else "-"))
    print()
    print("## 판정")
    a0 = tally.get(("A_asis", 0.0), [0, 0])
    print("  A_asis(temp 0) 정합 %d/%d — 라이브 불일치 재현 %s"
          % (a0[0], a0[1], "**됨**" if a0[0] == 0 else "⛔안 됨 ⇒ 판정하지 마라([[62]] 2b)"))
    for nm, _ in plan:
        print("  %-10s %s" % (nm, " · ".join("temp %.1f: %d/%d" % (t, v[0], v[1])
                                             for (n2, t), v in sorted(tally.items()) if n2 == nm)))
    print()
    print("⚠N_len 이 B_optline 과 같으면 그 이득은 **길이**다([[57]]).")
    print("⚠종점은 **자기 정합**이다 — 옵션 선택이 옳은지는 이 프로브가 사지 않는다([[69]]).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
