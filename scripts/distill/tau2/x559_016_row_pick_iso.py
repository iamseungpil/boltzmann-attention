# -*- coding: utf-8 -*-
r"""x559 — 016 의 진짜 결정점: **어느 추천 행이 아직 지급되지 않았나**

## 관측 (t7363 `task_016#s626729` · 2026-08-27)
    msg[17] tool       원장 도착 — `Silver Rewards Card · referral_status: IN_PROGRESS · 11/13`
    msg[20] assistant  *"One of the referrals, specifically the **Bronze** Rewards Card, was
                        **completed** on 11/11"*            ← 여기서 행을 잘못 골랐다
    msg[22]·[24]       Bronze 를 두 번 더 복창(자기-정박)
    msg[25] tool       우리 검색이 **Silver** 요건 배달 — *"spend at least $750 within 60 days"*
    msg[26] assistant  그 문서를 읽고 **또 Bronze** 로 답한다
손님은 *"보너스를 못 받았다"* 고 했다. **COMPLETE 행은 이미 지급된 행**이고(상태 정의도 우리가
배달한다) 에이전트는 그것을 골랐다. ⇒ 016 의 결손은 전달도 발화도 아니라 **행 오선택**이다.

## 무엇을 재나 — 바뀌는 것은 **표면화 한 줄**뿐
문맥은 라이브 전 접두(msg[20] 직전)를 그대로 쓴다.

    A_asis   그대로                                  ← 재현 게이트(**COMPLETE 행**이 나와야 한다)
    B_rows   원장 행을 `이름 (날짜): 상태` **한 줄씩** 재인쇄 — 우리가 이미 받은 그 행들이다
    N_len    길이만 맞춘 선택 무관 문장([[57]])

## 채점 — 닫힌 술어 · gold 미접촉([[23]])
gold 를 열지 않는다. 답이 지목한 이름이 원장에서 **COMPLETE 로만 존재하는 행**이면 오답이다
(*이미 지급된 것을 미지급이라 말한 것*). 상태 값은 도구 출력에서, 그 뜻은 우리가 배달한 정의에서
온다. 엔진은 **어느 행이 답인지 고르지 않는다** — 오답의 필요조건 하나만 검사한다.

## [[62]] 4문
  ① 결손 = t7363·t7356 궤적 축자(위). ② 재료는 **이미 닿아 있다**(원장·상태 정의 둘 다) ⇒
  레버 후보는 **표면화**뿐이고 계산·선택 0. ③ 사라지는 모델 판단 0 — 고르는 것은 끝까지 모델이다.
  ④ 순위·최댓값·*"정답은 X"* 0. 표적은 채점에만 쓰고 프롬프트에 안 들어간다.

사용: (리모트·cwd=scripts/distill/tau2) py -3 x559_016_row_pick_iso.py --port 8140
      --wiring-only 로 모델 없이 문맥·원장·요청부만 확인(무료).
"""

import argparse
import collections
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
import x554_diag_mispick_iso as X554                                 # noqa: E402

MODEL = "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8"
NL = chr(10)
# ⚠**행 단위로 묻는다.** 이름만 물으면 채점이 공허해진다 — 이 원장에서는 네 이름이 **전부**
#   COMPLETE 와 非COMPLETE 를 동시에 이고 있어서(배선 검증이 잡았다) *"COMPLETE 인 이름"* 이
#   하나도 없다. 그것이 바로 016 의 병이고 `T2_DIAG_UNAMBIGUOUS` 가 침묵하는 이유다.
#   라이브도 실은 행으로 답했다 — *"the Bronze Rewards Card, was completed on **11/11/2025**"*.
ASK = (NL + NL + "The customer says the referral bonus they are owed has not arrived. "
       "Reply with ONLY that referral's account type and its date, exactly as written in the "
       "records above, in the form `<account type> | <date>`. Nothing else.")


def gen(port, body, maxtok=64, temp=0.0):
    payload = {"model": MODEL, "temperature": temp, "max_tokens": maxtok,
               "messages": [{"role": "user", "content": body}]}
    req = urllib.request.Request("http://127.0.0.1:%d/v1/chat/completions" % port,
                                 data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read().decode("utf-8"))["choices"][0]["message"]["content"]


def render(ms, upto):
    """전 접두를 텍스트로. 크면 라이브 생성 뷰로 압축한다(정본 `_compact_view` 경유)."""
    raw = sum(len(str(m.get("content") or "")) for m in ms[:upto])
    if raw > 120000:
        try:
            import x467_policy_boolean_doc_iso as X467
            ms = X467.compact_view_dicts(list(ms[:upto]))[0]
            upto = len(ms)
        except Exception as e:
            print("[warn] 뷰 압축 실패(원문): %r" % (e,), file=sys.stderr)
    out = []
    for m in ms[:upto]:
        c = " ".join(str(m.get("content") or "").split())
        tcs = ["%s(%s)" % (F.nameof(tc), json.dumps(F.argsof(tc), ensure_ascii=False))
               for tc in (m.get("tool_calls") or ())]
        if tcs:
            c = (c + " " if c else "") + "TOOL_CALLS: " + " ".join(tcs)
        if c:
            out.append("[%s] %s" % (m.get("role"), c))
    return NL.join(out)


def row_of(reply, rows, spec):
    """답 → 그 답이 지목한 **행**과 그 행의 상태. 닫힌 술어(이름·날짜 축자 대조·해석 0).

    이름만으로는 행이 안 정해지므로 **이름과 날짜가 둘 다 맞는 행**만 채택한다. 날짜를 안 대면
    `(이름, None)` 으로 남기고 **판정하지 않는다**([[25]]).
    """
    g, dt, st = spec.get("group_field"), spec.get("date_field"), spec.get("status_field")
    r_l = " ".join(str(reply or "").split()).lower()
    best = None
    for r in rows or ():
        nm, d = str(r.get(g) or ""), str(r.get(dt) or "")
        if nm and d and nm.lower() in r_l and d in r_l:
            best = r
            break
    if best is not None:
        return "%s | %s" % (best.get(g), best.get(dt)), str(best.get(st) or "")
    nm = next((k for k in {str(r.get(g)) for r in rows or ()} if k and k.lower() in r_l), None)
    return (nm, None) if nm else (None, None)


def by_status(rows, spec):
    """이름 → 그 이름이 이고 있는 상태 집합 (정본 `t2_ledger.status_multiplicity` 재사용)."""
    import t2_ledger as LG
    return LG.status_multiplicity(rows, spec)


def row_lines(rows, spec):
    """`이름 (날짜): 상태` **행 순서 그대로**. 정렬·선택 0 — 우리가 받은 행의 재인쇄다."""
    g, dt, st = spec.get("group_field"), spec.get("date_field"), spec.get("status_field")
    return "; ".join("%s (%s): %s" % (r.get(g), r.get(dt), r.get(st)) for r in rows)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8140)
    ap.add_argument("--tags", default="bank_t7363_hard0_20260827,bank_t7356_grpB3_20260826")
    ap.add_argument("--sim", default="task_016#s626729")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--wiring-only", action="store_true")
    a = ap.parse_args(argv)

    import gate_interpreter as GI
    spec = next((s for s in ((GI.load_domain_a2("banking_knowledge") or {})
                             .get("ledger_metrics") or []) if s.get("diagnose_prompt")), None)
    if not spec:
        print("선언에 원장 스펙이 없다", file=sys.stderr)
        return 2
    keys = list(spec.get("row_keys") or ())

    ctxs = []
    for tag in [t.strip() for t in a.tags.split(",") if t.strip()]:
        sims = [s for s in F.sims(tag) if F.simtag(s) == a.sim]
        if not sims:
            print("[skip] %s 에 %s 없음" % (tag, a.sim))
            continue
        ms = sims[0].get("messages") or []
        # 결정점 = 원장이 도착한 뒤 **첫 assistant 발화**
        led = next((i for i, m in enumerate(ms) if m.get("role") == "tool"
                    and all(("%s:" % k) in str(m.get("content") or "") for k in keys)), None)
        if led is None:
            print("[skip] %s 원장 메시지를 못 찾음" % tag)
            continue
        # ⚠**본문이 있는** 첫 assistant 발화를 찾는다. 도구 호출만 든 턴을 잡으면 라이브가
        #   그 자리에서 한 말이 빈 문자열이라 재현 게이트를 세울 수 없다(배선 검증이 잡았다).
        w = next((i for i in range(led + 1, len(ms))
                  if ms[i].get("role") == "assistant"
                  and len(" ".join(str(ms[i].get("content") or "").split())) > 40), None)
        rows = X554.rows_from_traj(tag, a.sim, keys)
        if not (w and rows):
            print("[skip] %s 결정점/행 없음" % tag)
            continue
        ctxs.append((tag, ms, led, w, rows))

    if not ctxs:
        print("잴 문맥이 없다 — 판정하지 않는다([[25]])", file=sys.stderr)
        return 2

    print("# x559 — 016 행 오선택 격리")
    for tag, ms, led, w, rows in ctxs:
        m = by_status(rows, spec)
        done = sorted(k for k, v in m.items() if v == {"COMPLETE"})
        rowstat = collections.Counter(str(r.get(spec.get("status_field")) or "") for r in rows)
        print("  %-34s 원장 msg[%d] · 결정점 msg[%d] · 행 %d" % (tag[:34], led, w, len(rows)))
        print("      상태: %s" % json.dumps({k: sorted(v) for k, v in sorted(m.items())},
                                            ensure_ascii=False))
        print("      이름 단위로 COMPLETE 전용: %s  ⇒ **이름으로는 채점 불가**" % (done or "없음"))
        print("      **행** 단위 상태 분포: %s  ⇒ 행으로 물어야 갈린다" % dict(rowstat))
        live = " ".join(str(ms[w].get("content") or "").split())
        print("      라이브가 그 자리에서 한 말: %s" % live[:150])

    plan = [("A_asis", ""), ("B_rows", None), ("N_len", None)]
    if a.wiring_only:
        tag, ms, led, w, rows = ctxs[0]
        print()
        print("--- B_rows 추가분 ---")
        print("[the records above, one line per referral, verbatim] " + row_lines(rows, spec)[:600])
        print("--- 요청부 ---" + ASK)
        print("--- 문맥 %d자 ---" % len(render(ms, w)))
        return 0

    print()
    print("%-34s %-9s %-5s %-26s %s" % ("tag", "팔", "temp", "지목", "판정"))
    print("-" * 110)
    tally = collections.defaultdict(lambda: [0, 0])
    for tag, ms, led, w, rows in ctxs:
        base = render(ms, w)
        stat = by_status(rows, spec)
        paid = {k for k, v in stat.items() if v == {"COMPLETE"}}
        adds = {"A_asis": "",
                "B_rows": (NL + NL + "[the records above, one line per referral, verbatim] "
                           + row_lines(rows, spec) + "."),
                "N_len": (NL + NL + "[note] the records above were retrieved earlier in this "
                          "conversation and have not changed since; treat them as current and "
                          "complete for this customer.")}
        for nm, _ in plan:
            body = base + adds[nm] + ASK
            for tp, n in ((0.0, 1), (a.temp, a.n)):
                for _ in range(n):
                    try:
                        rep = " ".join(str(gen(a.port, body, 64, tp)).split())
                    except Exception as e:
                        print("%-34s %-9s %-5s 호출 실패: %r" % (tag[:34], nm, tp, e))
                        continue
                    pick, st_of = row_of(rep, rows, spec)
                    tally[(tag, nm)][1] += 1
                    ok = st_of is not None and st_of != "COMPLETE"
                    if ok:
                        tally[(tag, nm)][0] += 1
                    print("%-34s %-9s %-5s %-26s %s"
                          % (tag[:34], nm, tp, pick or "원장 밖",
                             ("%s" % st_of) if st_of else "판정 불가"))
    print()
    print("## 판정 (지목이 **미지급 행**인 비율)")
    for tag, ms, led, w, rows in ctxs:
        row = " · ".join("%s %d/%d" % (nm, tally[(tag, nm)][0], tally[(tag, nm)][1])
                         for nm, _ in plan)
        print("  %-34s %s" % (tag[:34], row))
    print()
    print("⚠A_asis 가 **이미 지급된 행**을 안 고르면 재현 실패다 — 판정하지 마라([[62]] 2b).")
    print("⚠N_len 이 B_rows 와 같으면 그 이득은 **길이**다([[57]]).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
