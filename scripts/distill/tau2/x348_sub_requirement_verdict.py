# -*- coding: utf-8 -*-
r"""x348 — **t7305(요구 주입 A/B) 판정**. 순서 = 런처 `run_t7305_20260817a.sh` §판정 사전 고정 그대로.

    ⓐ 배선  `[T2_SUB_REQUIREMENT] 인용 N개 중 원문 검증 통과 M개` — treat >0 · **ctl 0** · infra 0.
             보조 = `[T2_SUB_RECORDS]`(정규식 이설 배선·0 이어도 종전과 같음) · 무발화/호출실패 수.
    ⓑ 1차    **055 축별 최종 제출 gold 일치 합**(0~16) — `x341` 의 코드를 **그대로 import** 해서
             양팔 같은 규칙으로 잰다([[67]] 사본 금지). **GO = ctl 대비 +5 이상.**
    ⓒ 기전  `[T2_DOCDECIDE] → '…'` 값이 gold 클래스로 바뀌었는가. 값은 **축자로 인쇄**하고
             (직독은 사람이·[[08]]) 기계 판정은 **정규화 완전일치만** 쓴다.
             ⚠부분문자열 금지 — `Blue` ⊂ `Navy Blue`·`Silver` ⊂ `Silver Plus` 로 x347 형
             충돌(`$32,500` ⊃ `2,500`)이 그대로 재발한다. 포함 여부는 `~포함` 으로 **따로** 표시.
    ⓓ 성적  `reward` 만(C486 — `action_match` 는 소수점 표기에서 거짓 False 를 낸다).
    ⓔ 부작용 지연(요구 서브 1회 추가) · **098 불변** · CWE · 종료사유.

★계기 자기검정(`--selftest`) — 오늘만 계기 결함 9건이었다(핸드오프 §4).
  ⑴ 파서 양성통제: 알려진 축자 줄에서 (3, 2) 를 뽑는가 · 숫자 없는 줄은 빈 목록인가.
  ⑵ 실물 로그 양성통제: **t7304j**(플래그가 존재하지 않던 런)에 물려
     요구 발화 **0** ∧ DOCDECIDE 발화 **>0** ∧ sim **>0**. DOCDECIDE 가 0 이면 로그 조인이
     죽은 것이므로(2026-08-15 `simtag` 사고 동형) **계기 FAIL** 로 종료한다.

⚠정규식 0 — 숫자는 `split`+`isdigit` 로 읽는다(0번 규칙 강화판·핸드오프 §0). 마커 자리 잡기는
  정본 `t2_forensic.by_sim` 이 하고, 이 파일은 **우리가 찍은 줄**만 다룬다(도메인 산문 파싱 0).

실행:  /home/woori/venvs/seka_env/bin/python x348_sub_requirement_verdict.py [ctl_tags treat_tags]
       (태그는 콤마 구분 — 055 팔과 aux 팔이 다른 태그다)
       /home/woori/venvs/seka_env/bin/python x348_sub_requirement_verdict.py --selftest
"""
import collections
import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import t2_forensic as F                                            # noqa: E402
import x341_docbody_verdict as X                                   # noqa: E402  (gold/축 규칙 재사용)

REQ = r"\[T2_SUB_REQUIREMENT\] 인용 .*"
REQ_SKIP = r"\[T2_SUB_REQUIREMENT\] (?:건너뜀|호출 실패).*"
RECS = r"\[T2_SUB_RECORDS\] .*"
RECS_SKIP = r"\[T2_SUB_RECORDS\] 호출 실패.*"
DECIDE = r"\[T2_DOCDECIDE\] → .*"
GROUP = r"\[T2_SEARCH_AGENT\] group=(\S+)"
CWE = "ContextWindowExceededError"

DEF_CTL = "bank_t7305_ctl_20260817a,bank_t7305_ctlaux_20260817a"
DEF_TREAT = "bank_t7305_treat_20260817a,bank_t7305_treataux_20260817a"


def nums(line, after="인용 "):
    """줄에서 정수만 순서대로(정규식 0). `인용 3개 중 … 통과 2개` → [3, 2].

    ⚠`after` 뒤부터 읽는다 — 마커 자체에 숫자가 있다(`T2_SUB_REQUIREMENT` 의 **2**).
      2026-08-17 자기검정이 이 오독([2,3,2])을 잡았다(핸드오프 §4 계기결함 부류)."""
    s = str(line or "")
    i = s.find(after) if after else -1
    s = s[i + len(after):] if i >= 0 else ("" if after else s)
    out = []
    tok = ""
    for ch in s:
        if ch.isdigit():
            tok += ch
        else:
            if tok:
                out.append(int(tok))
            tok = ""
    if tok:
        out.append(int(tok))
    return out


def decided(line):
    """`[T2_DOCDECIDE] → 'Silver Plus'` → `Silver Plus`(문자열 연산만·정규식 0)."""
    s = str(line or "")
    i = s.find("→")
    s = s[i + 1:].strip() if i >= 0 else s
    for q in ("'", '"'):
        a, b = s.find(q), s.rfind(q)
        if a >= 0 and b > a:
            return s[a + 1:b]
    return s.strip()


def pair_axis(dec_hits, grp_hits):
    """DOCDECIDE 매치를 **바로 앞선 group 줄**에 붙인다(줄번호 순서만 씀 — 추정 아님).

    group 줄이 앞에 없으면 축을 `?` 로 남긴다([[25]] 모르면 안 뺀다)."""
    grps = sorted(grp_hits or [])
    out = []
    for ln, line in sorted(dec_hits or []):
        ax = "?"
        for gl, g in grps:
            if gl < ln:
                ax = g
            else:
                break
        out.append((ln, ax, decided(line)))
    return out


def log_tag(tag):
    """로그를 어느 태그에서 읽나. **aux 팔은 자기 로그 파일이 없다**(2026-08-17 실측) —
    런처가 `t2_launch <본팔> && t2_launch <aux>` 를 한 setsid 셸에서 돌려 stdout 이 본팔
    로그로 간다. 그대로 두면 aux 발화가 **0 으로 보인다**(죽은 계기·[[55]] 부류)."""
    if F.log_text(tag) or "aux" not in tag:
        return tag
    return tag.replace("aux_", "_")


def arm(name, tags):
    """한 팔 = 태그 여럿(055 + aux). sim 별 행 목록."""
    rows = []
    print("=" * 100)
    for tag0 in tags:
        sims = F.sims(tag0)
        tag = log_tag(tag0)                       # 결과는 tag0, 로그는 tag(aux 는 부모)
        log = F.log_text(tag)
        req = F.by_sim(tag, REQ, sims)
        req_x = F.by_sim(tag, REQ_SKIP, sims)
        rec = F.by_sim(tag, RECS, sims)
        rec_x = F.by_sim(tag, RECS_SKIP, sims)
        dec = F.by_sim(tag, DECIDE, sims)
        grp = F.by_sim(tag, GROUP, sims)
        print("[%s] %s · n=%d · CWE %d · 로그 %s %d줄"
              % (name, tag0, len(sims), log.count(CWE),
                 ("(부모 %s)" % tag) if tag != tag0 else "",
                 len(log.split("\n")) if log else 0))
        for s in sorted(sims, key=lambda x: (F.task_id(x), str(x.get("seed")))):
            key = F.simtag(s)
            tid = F.task_id(s)
            golds = X.gold_axes(tid)
            am = X.axis_match(tid, s, golds)
            qs = [nums(l) for _i, l in (req.get(key) or [])]
            axes_dec = pair_axis(dec.get(key), grp.get(key))
            # ⓒ gold 일치 = **정규화 완전일치만**. 포함은 따로 센다(부분문자열 충돌 금지).
            gnorm = {X.norm(v): ax for ax, v in golds.items()}
            eq = [(ax, v) for _ln, ax, v in axes_dec if X.norm(v) in gnorm]
            has = [(ax, v) for _ln, ax, v in axes_dec
                   if X.norm(v) not in gnorm and any(g and g in X.norm(v) for g in gnorm)]
            rows.append({
                "sim": key, "task": tid, "tag": tag0, "log_tag": tag,
                "req_calls": len(qs), "req_quotes": sum(q[0] for q in qs if q),
                "req_pass": sum(q[1] for q in qs if len(q) > 1),
                "req_silent": len(req_x.get(key) or []),
                "rec_calls": len(rec.get(key) or []), "rec_fail": len(rec_x.get(key) or []),
                "decide": [(ax, v) for _ln, ax, v in axes_dec],
                "decide_eq": eq, "decide_has": has,
                "axes": {ax: {"final": v, "gold": g, "match": m} for ax, (v, g, m) in am.items()},
                "n_match": sum(1 for v in am.values() if v[2]), "n_axes": len(am),
                "reward": (s.get("reward_info") or {}).get("reward"),
                "dur": round(s.get("duration") or 0, 1), "term": F.term_reason(s),
                "ncalls": len(list(F.calls(s))),
            })
            r = rows[-1]
            print("  %-22s 요구 %d회(인용 %d/검증 %d·무발화 %d) · recs %d · 결정 %s · 축 %d/%d %s"
                  " · R=%s dur=%s %s"
                  % (key, r["req_calls"], r["req_quotes"], r["req_pass"], r["req_silent"],
                     r["rec_calls"],
                     [(ax, (v or "")[:26]) for ax, v in r["decide"]] or "-",
                     r["n_match"], r["n_axes"],
                     {ax: (v[0] or "-")[:20] + ("=G" if v[2] else "≠" + (v[1] or "")[:14])
                      for ax, v in am.items()},
                     r["reward"], r["dur"], r["term"]))
    return rows


def verdict(R):
    print("\n" + "=" * 100)
    print("ⓐ 배선 — treat 발화 >0 ∧ ctl 0 ∧ infra 0")
    for n in ("ctl", "treat"):
        rs = R[n]
        infra = sum(1 for r in rs if r["term"] not in ("user_stop", "agent_stop", "max_steps"))
        print("   %-5s 요구발화 sim %d/%d · 호출 %d회 · 인용 %d → 검증통과 %d · 무발화 %d"
              " · SUB_RECORDS 호출 %d(실패 %d) · infra %d"
              % (n, sum(1 for r in rs if r["req_calls"]), len(rs),
                 sum(r["req_calls"] for r in rs), sum(r["req_quotes"] for r in rs),
                 sum(r["req_pass"] for r in rs), sum(r["req_silent"] for r in rs),
                 sum(r["rec_calls"] for r in rs), sum(r["rec_fail"] for r in rs), infra))

    print("ⓑ 1차 종점 — 축별 최종 제출 gold 일치 (055 0~16 합산·GO=+5)")
    tot = {}
    for n in ("ctl", "treat"):
        for tid in ("task_055", "task_024", "task_098"):
            rs = [r for r in R[n] if r["task"] == tid]
            m, a = sum(r["n_match"] for r in rs), sum(r["n_axes"] for r in rs)
            tot[(n, tid)] = m
            print("   %-5s %s  %d/%d" % (n, tid, m, a))
    d = tot.get(("treat", "task_055"), 0) - tot.get(("ctl", "task_055"), 0)
    print("   ★055 Δ = %+d  →  %s" % (d, "GO" if d >= 5 else ("미결" if d > 0 else "NO-GO")))

    print("ⓒ 기전 — DOCDECIDE 가 gold 클래스로 바뀌었나 (정규화 완전일치만)")
    for n in ("ctl", "treat"):
        rs = [r for r in R[n] if r["task"] in ("task_055", "task_024")]
        print("   %-5s gold-일치 결정 %d개 · gold-포함(≠일치·충돌주의) %d개 · 결정 총 %d개"
              % (n, sum(len(r["decide_eq"]) for r in rs), sum(len(r["decide_has"]) for r in rs),
                 sum(len(r["decide"]) for r in rs)))
        vals = collections.Counter(v for r in rs for _ax, v in r["decide"])
        print("         값 분포: %s" % dict(vals.most_common(8)))

    print("ⓓ reward")
    for n in ("ctl", "treat"):
        by = collections.defaultdict(list)
        for r in R[n]:
            by[r["task"]].append(r["reward"])
        print("   %-5s %s" % (n, {k: "%d/%d" % (sum(1 for x in v if x == 1.0), len(v))
                                  for k, v in sorted(by.items())}))

    print("ⓔ 부작용 (중앙값) · 098 불변 확인 · 종료사유")
    med = lambda xs: sorted(xs)[len(xs) // 2] if xs else 0                  # noqa: E731
    for n in ("ctl", "treat"):
        by = collections.defaultdict(list)
        for r in R[n]:
            by[r["task"]].append(r)
        print("   %-5s %s" % (n, {k: "dur %d · calls %d" % (med([x["dur"] for x in v]),
                                                            med([x["ncalls"] for x in v]))
                                  for k, v in sorted(by.items())}))
    for n in ("ctl", "treat"):
        t = collections.Counter(r["term"] for r in R[n])
        print("   %-5s 종료 %s" % (n, dict(t)))


def selftest():
    ok = True
    print("★⑴ 파서 양성통제")
    a = nums("[T2_SUB_REQUIREMENT] 인용 3개 중 원문 검증 통과 2개")
    b = nums("[T2_SUB_REQUIREMENT] 건너뜀(무발화): KeyError")
    c = decided("2026-08-17 [sim=task_055#s1] [T2_DOCDECIDE] → 'Silver Plus Savings'")
    print("   nums=%r  nums(무발화)=%r  decided=%r" % (a, b, c))
    if a != [3, 2] or b != [] or c != "Silver Plus Savings":
        print("   ✗ FAIL — 파서가 아는 정답을 못 맞힌다"); ok = False
    print("★⑵ 실물 로그 양성통제 — t7304j(플래그 부재 런): 요구 0 ∧ DOCDECIDE >0 ∧ sim >0")
    R = {"ctl": arm("ctl", ["bank_t7304_ctl_20260816j"]),
         "treat": arm("treat", ["bank_t7304_treat_20260816j"])}
    nsim = len(R["ctl"]) + len(R["treat"])
    nreq = sum(r["req_calls"] for n in R for r in R[n])
    ndec = sum(len(r["decide"]) for n in R for r in R[n])
    print("   sim %d · 요구발화 %d · DOCDECIDE %d" % (nsim, nreq, ndec))
    if nsim == 0 or ndec == 0:
        print("   ✗ FAIL — 로그 조인이 죽었다(simtag 미스 동형). 이 계기로 t7305 판정 금지"); ok = False
    if nreq != 0:
        print("   ✗ FAIL — 플래그 없던 런에서 요구 발화가 잡힌다(오탐)"); ok = False
    print("\n%s" % ("계기 OK — t7305 판정에 써도 된다" if ok else "계기 FAIL — 고치기 전에 쓰지 마라"))
    return 0 if ok else 1


def main():
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    a = [x for x in sys.argv[1:] if not x.startswith("-")]
    ctl = (a[0] if len(a) > 0 else DEF_CTL).split(",")
    treat = (a[1] if len(a) > 1 else DEF_TREAT).split(",")
    R = {"ctl": arm("ctl", ctl), "treat": arm("treat", treat)}
    verdict(R)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "..", "..", "reports", "facet_rft_2026",
                       "x348_sub_requirement_verdict.json")
    with io.open(out, "w", encoding="utf-8") as f:
        f.write(json.dumps(R, ensure_ascii=False, indent=1, default=str))
    print("\n저장: %s" % os.path.normpath(out))


if __name__ == "__main__":
    main()
