# -*- coding: utf-8 -*-
r"""x341 — **t7304(결정-자리 배달 객체 교체) 판정**. 순서 = 설계서 §4 사전 고정 그대로.

    ⓐ 배선(양팔 동일 계기·**부착** 기준): `[T2_DECISION_CARRY] … 부착 (N자)` sim 별 분포.
       통과 = treat 055·024 각 sim ≥10k 부착 ≥1회 ∧ ctl ≥10k 부착 0 ∧ 대용량 CLOBBER 0 ∧ infra 0.
       보조: `[T2_DOC_DELIVERY] skipped` 수(055 4/4 skip 이면 ⓐ 실패), `[T2_CP2_CLOBBER]`.
    ⓑ 1차 종점 = **축별 최종 제출 클래스의 gold 일치** — 양팔 궤적에서 **같은 코드**로 계산.
       gold 는 태스크 정의(evaluation_criteria)에서 읽는다(분석 전용·[[23]] 레버 무관).
    ⓒ 기전 직독 재료: 부착 턴·최종 클래스 첫 발화 msg 를 나란히 인쇄(직독은 사람이).
    ⓓ 성적 = `reward` 만(C486).
    ⓔ 부작용: CWE·건너뜀·098 불변·duration/write/호출 중앙값·문서군 분포.

⚠양성 통제: 발사 전 t7303 태그에 물려 기대값(ctl 대용량 부착 0 · 055 축 일치 ctl=checking 1/4·
  savings 0/4)을 재현해야 한다 — `--selftest` 가 그것이다(t7303 형 계기-결함 재발 방지).
⚠분석 스크립트다: 여기서의 이름 정규화는 [[59]](엔진 금지)와 무관하다.

실행:  /home/woori/venvs/seka_env/bin/python x341_docbody_verdict.py [ctl_tag treat_tag]
       /home/woori/venvs/seka_env/bin/python x341_docbody_verdict.py --selftest
"""
import collections
import io
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import t2_forensic as F                                            # noqa: E402

DOM = "/home/woori/scratch/tau2-bench/data/tau2/domains/banking_knowledge"
ATTACH = r"T2_DECISION_CARRY\] 이 턴 재생성 버퍼에 부착 \((\d+)자\)"
HOLD = r"T2_DECIDE_BEFORE_WRITE\] write 1턴 유예"
GROUP = r"T2_SEARCH_AGENT\] group=(\S+)"
CLOBBER = r"T2_CP2_CLOBBER\] \S+ 가 미소비 배달물 (\d+)자"
SKIPPED = r"T2_DOC_DELIVERY\] skipped"
CWE = "ContextWindowExceededError"
BIG = 10000


def norm(x):
    """기계 정규화(분석 전용): 소문자·영숫자만. 'purple_account' ≈ 'Purple Account'."""
    return re.sub(r"[^a-z0-9]", "", str(x or "").lower())


def gold_axes(task_id):
    """태스크 정의에서 **축→gold 클래스** 를 뽑는다(분석 전용).

    형태: open_bank_account 류 = (account_type 축, account_class 값) ·
          apply_for_credit_card = ('card', card_type) · submit_referral = ('referral', account_type).
    """
    p = os.path.join(DOM, "tasks", task_id + ".json")
    try:
        t = json.load(io.open(p, encoding="utf-8"))
    except Exception:
        return {}
    out = {}
    for a in ((t.get("evaluation_criteria") or {}).get("actions") or []):
        args = a.get("arguments") or {}
        inner = args.get("arguments")
        if isinstance(inner, str):
            try:
                inner = json.loads(inner)
            except Exception:
                inner = {}
        if isinstance(inner, dict) and inner.get("account_class"):
            out[str(inner.get("account_type") or "?")] = inner["account_class"]
        if a.get("name") == "apply_for_credit_card" and args.get("card_type"):
            out["card"] = args["card_type"]
        if a.get("name") == "submit_referral" and args.get("account_type"):
            out["referral"] = args["account_type"]
    return out


def final_axes(sim):
    """sim 의 **축별 마지막 제출값**(양팔 같은 규칙). {축: 값}. write 만 본다(발화는 ⓒ 직독)."""
    out = {}
    for _m, tc in F.calls(sim):
        n = F.nameof(tc) or ""
        a = F.argsof(tc) or {}
        inner = a.get("arguments")
        if isinstance(inner, str):
            try:
                inner = json.loads(inner)
            except Exception:
                inner = {}
        inner = inner if isinstance(inner, dict) else {}
        tool = str(a.get("agent_tool_name") or a.get("discoverable_tool_name") or n)
        if "open_" in tool and (inner.get("account_class") or inner.get("account_subtype")):
            ax = str(inner.get("account_type") or "?").lower()
            ax = "checking" if "check" in ax else ("savings" if "sav" in ax else ax)
            out[ax] = inner.get("account_class") or inner.get("account_subtype")
        if n == "apply_for_credit_card" and a.get("card_type"):
            out["card"] = a["card_type"]
        if n == "submit_referral" and a.get("account_type"):
            out["referral"] = a["account_type"]
    return out


def axis_match(task_id, sim, golds):
    """{축: (제출값, gold, 일치?)} — gold 에 있는 축만."""
    fin = final_axes(sim)
    out = {}
    for ax, g in golds.items():
        v = fin.get(ax)
        out[ax] = (v, g, v is not None and norm(v) == norm(g))
    return out


def attach_then(sim, sizes):
    """★ⓒ 기전 분류(리뷰 ②): 대용량 부착이 있었다면, 그 **직후 어시스턴트 생성**이
    `tool_call`-only 였는지 텍스트였는지. 부착 턴을 로그에서 정확히 못 짚으므로 보수적으로
    **모든 어시스턴트 메시지의 유형 분포**를 요약한다(텍스트가 하나도 없으면 tool-only 우세).
    ⚠이것은 분류이지 판정이 아니다 — 최종 귀속은 사람이 궤적을 직독해서 한다([[08]])."""
    if not any(x >= BIG for x in sizes):
        return "-"
    txt = sum(1 for m in (sim.get("messages") or [])
              if m.get("role") == "assistant" and str(m.get("content") or "").strip())
    tc = sum(1 for m in (sim.get("messages") or [])
             if m.get("role") == "assistant" and (m.get("tool_calls") or []))
    return "txt%d/tc%d" % (txt, tc)


def arm_report(name, tag):
    sims = F.sims(tag)
    log = F.log_text(tag)
    att = F.by_sim(tag, ATTACH, sims)
    hold = F.by_sim(tag, HOLD, sims)
    grp = F.by_sim(tag, GROUP, sims)
    clb = F.by_sim(tag, CLOBBER, sims)
    skp = F.by_sim(tag, SKIPPED + ".*", sims)
    rows = []
    print("=" * 96)
    print("[%s] %s · n=%d · CWE(로그 전체) %d" % (name, tag, len(sims), log.count(CWE)))
    for s in sorted(sims, key=lambda x: (F.task_id(x), str(x.get("seed")))):
        key = F.simtag(s)
        sizes = sorted((int(v) for _i, v in (att.get(key) or [])), reverse=True)
        golds = gold_axes(F.task_id(s))
        am = axis_match(F.task_id(s), s, golds)
        nbig = sum(1 for x in sizes if x >= BIG)
        clob_big = sum(1 for _i, v in (clb.get(key) or []) if int(v) >= BIG)
        rows.append({
            "sim": key, "task": F.task_id(s), "attach_sizes": sizes[:6], "n_big": nbig,
            "clobber_big": clob_big, "skipped": len(skp.get(key) or []),
            "axes": {ax: {"final": v, "gold": g, "match": mt} for ax, (v, g, mt) in am.items()},
            "n_match": sum(1 for v in am.values() if v[2]), "n_axes": len(am),
            "hold": len(hold.get(key) or []),
            "groups": [g for _i, g in (grp.get(key) or [])],
            "attach_then": attach_then(s, sizes),
            "reward": (s.get("reward_info") or {}).get("reward"),
            "dur": round(s.get("duration") or 0, 1), "term": F.term_reason(s),
            "ncalls": len(list(F.calls(s))),
        })
        r = rows[-1]
        print("  %-22s big부착 %d hold %d skip %d clob %d · 후속=%s · 축 %d/%d %s · R=%s dur=%s"
              % (key, nbig, r["hold"], r["skipped"], clob_big, r["attach_then"],
                 r["n_match"], r["n_axes"],
                 {ax: (v[0] or "-")[:22] + ("=G" if v[2] else "≠" + (v[1] or "")[:14])
                  for ax, v in am.items()}, r["reward"], r["dur"]))
    return rows


def main():
    if "--selftest" in sys.argv:
        tags = [("ctl", "bank_t7303_ctl_20260816h"), ("treat", "bank_t7303_treat_20260816h")]
        print("★양성 통제(t7303): 기대 = ctl big부착 0 · 055 축일치 ctl checking 1/4 · savings 0/4")
    else:
        a = sys.argv[1:] or ["bank_t7304_ctl_20260816j", "bank_t7304_treat_20260816j"]
        tags = [("ctl", a[0]), ("treat", a[1])]
    R = {}
    for n, t in tags:
        R.setdefault(n, [])
        for one in t.split(","):          # 055 는 nt=8 로 따로 도므로 태그가 둘일 수 있다
            R[n] += arm_report(n, one)

    print("\n" + "=" * 96)
    print("ⓐ 배선 (부착 기준·양팔 동일 계기)")
    for n in ("ctl", "treat"):
        rs = R[n]
        tgt = [r for r in rs if r["task"] in ("task_055", "task_024")]
        infra = sum(1 for r in rs if r["term"] not in ("user_stop", "agent_stop", "max_steps"))
        print("   %-5s big부착 sim %d/%d(표적 %d/%d) · **hold %d** · clobber %d · skip %d · infra %d"
              % (n, sum(1 for r in rs if r["n_big"]), len(rs),
                 sum(1 for r in tgt if r["n_big"]), len(tgt),
                 sum(r["hold"] for r in rs),
                 sum(r["clobber_big"] for r in rs), sum(r["skipped"] for r in rs), infra))
    print("ⓑ 1차 종점 — 축별 최종 제출 gold 일치 (055 합산 0~8 · 024 0~4)")
    for n in ("ctl", "treat"):
        for tid in ("task_055", "task_024", "task_098"):
            rs = [r for r in R[n] if r["task"] == tid]
            print("   %-5s %s  %d/%d" % (n, tid, sum(r["n_match"] for r in rs),
                                         sum(r["n_axes"] for r in rs)))
    print("ⓓ reward")
    for n in ("ctl", "treat"):
        by = collections.defaultdict(list)
        for r in R[n]:
            by[r["task"]].append(r["reward"])
        print("   %-5s %s" % (n, {k: "%d/%d" % (sum(1 for x in v if x == 1.0), len(v))
                                  for k, v in sorted(by.items())}))
    print("ⓒ 기전 분류 (대용량 부착 sim 의 어시스턴트 텍스트/툴콜 수)")
    for n in ("ctl", "treat"):
        print("   %-5s %s" % (n, [r["attach_then"] for r in R[n] if r["attach_then"] != "-"] or "없음"))
    print("ⓔ 오군 부착 (요구축 밖 문서군 발화 수)")
    WANT = {"checking_accounts", "savings_accounts", "business_credit_cards"}
    for n in ("ctl", "treat"):
        bad = collections.Counter(g for r in R[n] for g in r["groups"] if g not in WANT)
        print("   %-5s %s" % (n, dict(bad.most_common(6)) or "없음"))
    print("ⓔ 부작용 (중앙값)")
    for n in ("ctl", "treat"):
        by = collections.defaultdict(list)
        for r in R[n]:
            by[r["task"]].append(r)
        med = lambda xs: sorted(xs)[len(xs) // 2]                          # noqa: E731
        print("   %-5s %s" % (n, {k: "dur %d · calls %d" % (med([x["dur"] for x in v]),
                                                            med([x["ncalls"] for x in v]))
                                  for k, v in sorted(by.items())}))

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "..", "..", "reports", "facet_rft_2026",
                       "x341_docbody_verdict.json")
    with io.open(out, "w", encoding="utf-8") as f:
        f.write(json.dumps(R, ensure_ascii=False, indent=1, default=str))
    print("\n저장: %s" % os.path.normpath(out))


if __name__ == "__main__":
    main()
