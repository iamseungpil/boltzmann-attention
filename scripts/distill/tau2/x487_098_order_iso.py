# -*- coding: utf-8 -*-
r"""x487 — **098: 자기-정박이 진짜 원인인가 · 순서를 바꿔서 잰다** (2026-08-23·[[62]]·사용자 지시)

사용자 지시 축자: *"098은 격리에서 순서에 따른 자기 정박이 실제 원인인지부터 확인하라.
순서를 바꾸어서 실험하라."*

## 무엇을 아는가 (x486 전수 + 정박-이전 구간 차분)
  · 098 은 81 sim 중 55 pass / 26 fail — **원래 불안정**하다(이웃 계좌명을 집는다·[[53]]).
    Light Blue 11 · Dark Green 4 · EcoCard 4 · Green Fee-Free 2 · **Bluest 2** · …
  · 다만 최근 1단계 계보(t7326·t7328·t7335·t7336)는 **7/7** 이었고 t7346 이 0/2 로 깼다.
  · ★**정박 이전 구간**(로그줄 154~163 · 양쪽 비슷)만 세면 차이가 **한 덩어리**다:
        t7336 에만 발화 : T2_SEARCH_AGENT 2 · T2_DOCGROUP 1 · T2_GROUPORDER 1 ·
                          T2_DOCDECIDE 1 · T2_SEARCH_ON_PROCEED 1 · T2_DECISION_CARRY 1
        t7346 에만 발화 : T2_MATERIAL_GATE 1 · T2_PROV 1
        나머지 ±1~2.5 (ARBITRATE·PHASE_PRECEDE 류 다발 태그·n=2 라 잡음 수준)
    ⇒ 통과 런은 **모델이 계좌명을 말하기 전에** 검색서브→축분해→답→배달→운반이 다 돌았고,
      실패 런은 그 사슬이 정박 시점까지 **한 번도 안 돌았다**.
  · 궤적 축자: t7346 [24] assistant 가 'Bluest' 를 처음 말하고, 우리 답('Blue Account')은
    [26] 이후에야 온다. t7336 은 [26] assistant 가 'Blue' 를 말하기 전에 이미 왔다.

## 그래서 무엇이 아직 안 증명됐나
*"늦게 와서 못 고쳤다"*(자기-정박)는 **아직 가설**이다. 늦음이 원인이려면 **같은 내용**이
  · 정박 **전**에 오면 답을 바꾸고
  · 정박 **후**에 오면 못 바꿔야
한다. 그 둘을 한 변수(위치)만 바꿔서 잰다.

## 팔 (내용은 네 팔에서 **완전히 동일**·위치만 다르다)
    A_pre        [24] 직전까지 · 주입 0        → 재현 확인(모델이 'Bluest' 를 스스로 내나)
    B_pre_deliv  [24] 직전까지 + 우리 답 주입  → 정박 **전** 전달이 답을 바꾸나
    C_post       [25] 까지(모델이 이미 'Bluest' 를 말했고 손님이 받았다) · 주입 0
    D_post_deliv [25] 까지 + **같은** 우리 답 주입 → 정박 **후** 전달이 답을 바꾸나
    N_neg        [24] 직전까지 + 무내용 한 줄  → [[57]] 부정통제
  ⇒ **B ↔ D 가 이 실험의 전부다.** 내용이 같으므로 둘의 차이는 **순서/정박**뿐이다.

## ⚠정직하게 남길 것 (이 프로브가 재지 **않는** 것)
  ⓐ 배달 내용은 우리 엔진 자신의 산출(`[T2_DOCDECIDE] → …`)을 **로그에서 읽어** 쓴다. 라이브
     문면은 비커밋이라 궤적에 없다(C596) — 축자 복원이 불가능하다. 그래서 **절대 수준**
     (B 가 몇 개 맞히나)은 레버 주장이 될 수 없다. 해석 가능한 양은 **B − D 차이**뿐이다.
  ⓑ 그 답이 gold 와 같은 문자열이라는 사실은 이 프로브를 **떠먹이기로 만든다** — 의도적이다.
     떠먹여도 정박 뒤엔 안 바뀐다면 그것이 곧 정박의 증거다([[62]]④ 는 *레버* 를 금지하는
     규칙이지, 인과를 가르는 **계측**까지 금지하지 않는다. 이 파일은 레버가 아니다).
  ⓒ n 은 원천 2 sim × 팔당 6 = 12/팔. 방향은 볼 수 있어도 [S] 는 못 된다.

## [[71]] 4문
  1) 기능 하나 — 각 재생은 **다음 발화 하나**만 만든다. 채점은 바깥에서 문자열 대조.
  2) 재료는 선언·엔진 산출에서 — 계좌명 후보는 **우리 로그의 `T2_DOCDECIDE`** 에서 읽는다.
     이 파일에 계좌명 리터럴 0([[63]] 떠먹이기 금지의 형식은 지킨다).
  3) 전달 = 그 답을 그대로. 검색·유사도 0.
  4) 엔진 해석·순위 0 — 우리 답을 **있는 그대로** 싣고, 무엇을 제출할지는 모델이 정한다.
"""
import argparse
import collections
import copy
import io
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import t2_forensic as F                     # noqa: E402
import x465_transfer_doc_iso as X465        # noqa: E402  재생 관용구(사본 0·[[67]])

REP = os.path.abspath(os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026"))
NLC = chr(10)
TASK = "task_098"
WRITE = "submit_referral"
ARG = "account_type"

RE_DEC = re.compile(r"\[T2_DOCDECIDE\] → '([^']*)'")
DELIV = ("[KB] The knowledge-base sweep of the account documents returned this answer for the "
         "customer's account: %s")
NUDGE = "[NOTICE] Please continue with the customer's request."


def our_answer(tag, key):
    """그 sim 에서 **우리 검색 서브가 낸 답** — 로그 축자. 없으면 None(조용히 넘기지 않는다)."""
    try:
        txt = F.log_text(tag)
    except Exception:
        txt = ""
    got = []
    for ln in txt.splitlines():
        if key not in ln:
            continue
        m = RE_DEC.search(ln)
        if m and m.group(1):
            got.append(m.group(1))
    return got


def anchor_index(sim, needle):
    """모델이 그 값을 **처음 말한** assistant 발화 index — 닫힌 술어(부분문자열)."""
    for i, m in enumerate(sim.get("messages") or []):
        if str(m.get("role") or "") != "assistant":
            continue
        if needle in str(m.get("content") or ""):
            return i
    return None


def submitted_type(calls):
    """다음 발화가 낸 `submit_referral.account_type` (래퍼 해제 포함). 없으면 None."""
    for nm, ag in calls:
        ag = ag or {}
        inner = str(ag.get("agent_tool_name") or ag.get("user_tool_name") or "")
        if (inner or str(nm or "")) == WRITE:
            args = ag.get("arguments") if isinstance(ag.get("arguments"), dict) else ag
            v = (args or {}).get(ARG)
            # ★첫 호출이 인자 없이 나가는 sim 이 있다(census: `sub=None,Bluest Account`).
            #   첫 항목을 그대로 돌려주면 그 원천이 통째로 버려진다 — **값이 있는 첫 호출**을 본다.
            if v:
                return v
    return None


def build(msgs, cut, extra=None):
    ctx = copy.deepcopy(msgs[:cut])
    if extra:
        ctx.append({"role": "user", "content": extra})
    return ctx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8141)
    ap.add_argument("--model", default="Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8")
    ap.add_argument("--tag", default="bank_t7346_halfB_20260822")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--arms", default="A_pre,B_pre_deliv,C_post,D_post_deliv,N_neg")
    ap.add_argument("--wiring-only", action="store_true")
    ap.add_argument("--out", default="x487_098_order_iso.json")
    a = ap.parse_args()

    import x448_index_vs_all_iso as IVA
    sb = IVA.Sandbox()
    tools = list(sb.env.get_tools() or [])
    print("=" * 100)
    print("x487 · env 도구 %d종 · 재생 = 실물 스키마 + 메시지 객체 + la.generate (C584)" % len(tools))

    srcs = []
    for s in F.sims(a.tag, suffix=".results.json.gz"):
        if str(s.get("task_id")) != TASK:
            continue
        key = F.simtag(s) or ""
        msgs = s.get("messages") or []
        live = submitted_type(
            [(t["inner"] or t["outer"], t["args"]) for t in F.trajectory_actions(s)])
        ans = our_answer(a.tag, key)
        if not (live and ans):
            print("  ⚠건너뜀 %s — 라이브 제출값=%r · 우리 답=%r" % (key, live, ans))
            continue
        i_anchor = anchor_index(s, str(live).split()[0])   # 'Bluest Account' → 'Bluest'
        if i_anchor is None:
            print("  ⚠건너뜀 %s — 정박 발화를 못 찾았다([[55]])" % key)
            continue
        srcs.append({"key": key, "sim": s, "live": live, "answers": ans,
                     "i_anchor": i_anchor})
        print("  원천 %-22s 라이브 제출='%s' · 정박 발화=[%d] · 우리 답=%s"
              % (key, live, i_anchor, ans))
    if not srcs:
        raise SystemExit("원천 0 — 태그·로그부터 본다([[55]])")

    # 배달 텍스트: 우리 답 **전부**(선택·순위 0). 네 팔에서 문자열이 완전히 같다.
    for s in srcs:
        s["deliver"] = DELIV % ", ".join(s["answers"])
        print("  배달문(%s·%d자): %s" % (s["key"][-12:], len(s["deliver"]), s["deliver"][:120]))
    if a.wiring_only:
        print(NLC + "[배선] wiring-only 종료 — LLM 0·GPU 0")
        return 0

    arms = [x.strip() for x in a.arms.split(",") if x.strip()]
    base = "http://localhost:%d/v1" % a.port
    rows = []
    for s in srcs:
        msgs = s["sim"].get("messages") or []
        i = s["i_anchor"]
        plan = {"A_pre": (i, None), "B_pre_deliv": (i, s["deliver"]),
                "C_post": (i + 2, None), "D_post_deliv": (i + 2, s["deliver"]),
                "N_neg": (i, NUDGE)}
        for arm in arms:
            cut, extra = plan[arm]
            if cut > len(msgs):
                print("  ⚠%s/%s 컷 %d > 메시지 %d — 건너뜀" % (s["key"], arm, cut, len(msgs)))
                continue
            ctx = build(msgs, cut, extra)
            print(NLC + "── %s / %-12s (컷=[%d] 주입=%d자) ────────"
                  % (s["key"][-12:], arm, cut, len(extra or "")))
            for k, t in enumerate([0.0] + [a.temperature] * a.n):
                try:
                    r = X465.replay(ctx, tools, a.model, base, t)
                except Exception as e:
                    print("  #%d EXC %r" % (k, e))
                    rows.append({"src": s["key"], "arm": arm, "k": k, "cat": "EXC"})
                    continue
                sub = submitted_type(r.calls)
                txt = " ".join(r.text.split())
                # 닫힌 채점: 제출 인자가 있으면 그 값, 없으면 발화에 등장한 계좌명 후보.
                # ★채점 수리(2026-08-23 1차 관측): 라이브 [24] 는 `Bluest`(단독)라고 말했는데
                #   전체 문자열(`Bluest Account`)만 찾으면 **정박 발화를 못 본다** — A_pre 6/6 이
                #   빈칸으로 찍혔다. 후보는 전체 값과 **머리 토큰**을 함께 본다(`anchor_index` 와
                #   같은 규칙). 후보 목록은 우리 답 + 라이브 값에서만 오고 리터럴은 0이다.
                cands = set()
                for x in s["answers"] + [s["live"]]:
                    if x:
                        cands.add(x)
                        cands.add(str(x).split()[0])
                hit = sorted((c for c in cands if c in txt), key=len, reverse=True)
                seen, keep = set(), []
                for c in hit:                 # 'Bluest Account' 가 잡히면 'Bluest' 는 안 센다
                    if not any(c in k for k in keep):
                        keep.append(c)
                        seen.add(c)
                cat = ("sub:%s" % sub) if sub else ("say:%s" % ",".join(keep) if keep else "none")
                rows.append({"src": s["key"], "arm": arm, "k": k, "temp": t, "cat": cat,
                             "calls": [nm for nm, _ in r.calls], "text": txt[:240]})
                print("  #%d t=%.1f  %-30s %s" % (k, t, cat[:30],
                                                  ",".join(nm for nm, _ in r.calls) or "-"))

    cats = sorted({r["cat"] for r in rows})
    print(NLC + "=" * 100)
    print("%-14s %s" % ("팔", " ".join("%-26s" % c[:26] for c in cats)))
    for arm in arms:
        rs = [r for r in rows if r["arm"] == arm]
        if not rs:
            continue
        print("%-14s %s (n=%d)" % (arm, " ".join(
            "%-26s" % ("%d/%d" % (sum(1 for r in rs if r["cat"] == c), len(rs)))
            for c in cats), len(rs)))
    p = os.path.join(REP, a.out)
    with io.open(p, "w", encoding="utf-8") as f:
        json.dump({"task": TASK, "tag": a.tag,
                   "sources": [{"key": s["key"], "live": s["live"], "answers": s["answers"],
                                "i_anchor": s["i_anchor"], "deliver": s["deliver"]}
                               for s in srcs], "rows": rows}, f, ensure_ascii=False, indent=1)
    print(NLC + "판정: A_pre 가 라이브 값을 재현해야 격리가 산다(아니면 [[55]]·결과 폐기).")
    print("      ★**B_pre_deliv ↔ D_post_deliv** — 내용이 같으므로 차이는 **순서**뿐이다.")
    print("        B 만 답을 바꾸면 자기-정박이 원인이고 처방 축은 **도착 시점**이다.")
    print("        둘 다 바꾸면 정박이 아니라 **미전달**이 원인(시점 무관).")
    print("        둘 다 못 바꾸면 전달로 안 닫히는 것이고 정박 가설은 기각이다.")
    print("→ %s" % p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
