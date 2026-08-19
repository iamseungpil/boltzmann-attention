# -*- coding: utf-8 -*-
r"""x399 — **완료 신념은 한 형태인가**: 실패를 *신념의 형태*로 재분류한다.

## 왜 (사용자 지시 2026-08-19)
"조기 종결/허위 완료"로 묶었지만 그 안에 서로 다른 것들이 섞여 있을 수 있다 — 아무것도 안 하고
했다고 하는 것과, 4계좌 중 1계좌만 하고 다 했다고 하는 것과, 손님에게 도구를 넘기고 끝났다고 하는 것은
**다른 결손이고 처방도 다르다**. 형태가 몇 개인지 궤적으로 센다.

## 두 자료를 같은 축으로 가른다
  ⑴ **라이브 궤적**(t7326 40 sim) — 미매치 gold 마다 *신념의 형태*를 궤적 증거로 판정
  ⑵ **격리 기권**(`x395`/`x398` 의 `{"tool": null, "reason": …}`) — 이유 문면이 **무엇을 주장하는지**

## 형태 코드 (사전 고정 우선순위·첫 매치·새 이름 만들지 않는다 [[48]])
    DENY_MISREAD        직전 도구 응답이 거부·에러인데 완료를 주장 (우리 층 deny 포함)
    PARTIAL_GENERALIZED 같은 도구의 gold 가 여럿인데 일부만 실행하고 전부 했다고 주장
    DELEGATED           손님에게 도구를 넘기고(give) 그것을 완료로 침
    READ_AS_WRITE       조회만 하고 처리됐다고 침 (write 호출 0)
    PROPOSED_AS_DONE    호출도 give 도 0 인데 완료를 주장
    PLAN_ONLY           "하겠다"만 하고 종료 (완료 주장 없음)
    NO_CLAIM            완료도 의도도 말한 적 없음 (신념 아님 — 계획 부재)

⚠판정은 **호출 이력·도구 응답·엔티티 수**로 한다. 문면은 형태를 *확인*하는 데만 쓰고, 문면만으로
  형태를 정하지 않는다. 기권 이유는 별도 축(무엇을 주장하나)으로 세고 축자 목록을 인쇄한다.

사용: py -3 x399_closure_forms.py
"""
import collections
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

import t2_forensic as F  # noqa: E402
import x395_compliance_iso as X  # noqa: E402
import x396_saying_vs_doing as C  # noqa: E402

TAGS = X.TAGS
SUF = X.SUF
GIVE = "give_discoverable_user_tool"
ENVERR = ("Error:", "NOT_VERIFIED", "not been given", "Unknown", "Invalid", "cannot be")

# 기권 이유가 **무엇을 주장하는가** — 축자 목록(해석 0·인쇄된다)
REASON_PAT = [
    ("ALREADY_DONE", ("already been", "already applied", "have been applied", "has been applied",
                      "already submitted", "already filed", "already processed", "already completed")),
    ("CONFIRMED_BY_USER", ("confirmed by the user", "user has confirmed", "customer confirmed")),
    ("NO_FURTHER", ("no further action", "no additional action", "nothing further",
                    "no further tool", "no more action", "not required at this point")),
    ("USER_WILL_DO", ("user will", "customer will", "the user needs to", "user must",
                      "instructed the user", "user can now")),
    ("TRANSFERRED", ("transferred", "human agent", "escalat")),
    ("INSUFFICIENT", ("not enough information", "need more information", "insufficient",
                      "unclear", "cannot determine")),
]


def resp_of(sim):
    """tool_call id -> 응답 본문."""
    out = {}
    for m in (sim.get("messages") or []):
        if m.get("role") == "tool" and m.get("id"):
            out[m["id"]] = " ".join(str(m.get("content") or "").split())
    return out


def form_of(sim, g, gold_rows):
    """한 미매치 gold 액션의 **신념 형태**(사전 고정 우선순위·첫 매치)."""
    nm = g["name"]
    R = resp_of(sim)
    calls = []          # (turn, wrapper, inner, 응답)
    for m in (sim.get("messages") or []):
        for tc in (m.get("tool_calls") or []):
            a = F.argsof(tc)
            calls.append((m.get("turn_idx"), F.nameof(tc), str(F.inner_name(a) or F.nameof(tc)),
                          R.get(tc.get("id"), "")))
    mine = [c for c in calls if c[2] == nm]
    ok = [c for c in mine if c[3] and not any(p in c[3] for p in ENVERR)]
    err = [c for c in mine if c[3] and any(p in c[3] for p in ENVERR)]
    gave = [c for c in calls if c[1] == GIVE and c[2] == nm]

    texts = [str(m.get("content") or "") for m in (sim.get("messages") or [])
             if m.get("role") == "assistant" and m.get("content")]
    claim = next((t for t in texts if C.DONE_RE.search(" ".join(t.split()))), None)
    intent = next((t for t in texts if C.INTENT_RE.search(" ".join(t.split()))), None)

    # 같은 도구를 요구하는 gold 가 몇 개이고 몇 개가 매치됐나(부분 일반화 판정)
    same = [x for x in gold_rows if x["name"] == nm]
    matched = [x for x in same if x["match"]]

    if not claim and not intent:
        return "NO_CLAIM", ""
    if claim and err and not ok:
        return "DENY_MISREAD", err[0][3][:70]
    if claim and len(same) >= 2 and matched:
        return "PARTIAL_GENERALIZED", "gold %d개 중 매치 %d" % (len(same), len(matched))
    if claim and gave and not ok:
        return "DELEGATED", "give %d회·call 성공 0" % len(gave)
    if claim and not mine:
        anyw = [c for c in calls if c[3] and not any(p in c[3] for p in ENVERR)]
        if g["type"] == "write" and anyw:
            return "READ_AS_WRITE", "다른 호출 %d회·이 도구 0회" % len(anyw)
        return "PROPOSED_AS_DONE", ""
    if claim:
        return "PROPOSED_AS_DONE", "호출 %d회(성공 %d)" % (len(mine), len(ok))
    return "PLAN_ONLY", " ".join((intent or "").split())[:60]


def reason_form(txt):
    low = " ".join((txt or "").split()).lower()
    for name, keys in REASON_PAT:
        if any(k in low for k in keys):
            return name
    return "OTHER"


def main():
    print("=" * 104)
    print("x399 · 완료 신념의 형태 — 라이브 궤적 + 격리 기권")
    print("=" * 104)

    rows = []
    for tag in TAGS:
        for sim in F.scored(tag, SUF):
            if ((sim.get("reward_info") or {}).get("reward") or 0) >= 1.0:
                continue
            gold = C.gold_rows(sim)
            for g in gold:
                if g["match"]:
                    continue
                code, ev = form_of(sim, g, gold)
                rows.append({"task": F.task_id(sim), "trial": sim.get("trial"),
                             "name": g["name"], "type": g["type"], "form": code, "ev": ev[:70]})

    print("\n## ⑴ 라이브 — 미매치 gold %d건의 신념 형태" % len(rows))
    cc = collections.Counter(r["form"] for r in rows)
    for k, v in cc.most_common():
        print("  %-20s %3d  (%.0f%%)" % (k, v, 100.0 * v / len(rows)))
    print("\n  ※ NO_CLAIM 은 신념이 아니라 **계획 부재**다 — 조기 종결 모수에서 빼야 한다.")
    belief = [r for r in rows if r["form"] != "NO_CLAIM"]
    print("  ⇒ 신념이 있는 것 %d건 · 형태 %d종" % (len(belief), len({r["form"] for r in belief})))

    print("\n## 형태 × 태스크 (신념 있는 것만)")
    bt = collections.defaultdict(collections.Counter)
    for r in belief:
        bt[r["form"]][r["task"]] += 1
    for k in sorted(bt, key=lambda x: -sum(bt[x].values())):
        print("  %-20s %s" % (k, " ".join("%s×%d" % (t.replace("task_", ""), n)
                                          for t, n in bt[k].most_common(8))))

    print("\n## 실물 (형태마다 최대 2건·축자)")
    seen = collections.Counter()
    for r in belief:
        if seen[r["form"]] >= 2:
            continue
        seen[r["form"]] += 1
        print("  %-20s %-9s t%-2s %-38s %s"
              % (r["form"], r["task"], r["trial"], r["name"][:38], r["ev"]))

    # ⑵ 격리 기권 이유
    print("\n## ⑵ 격리 기권의 **주장 내용** (축자 키워드 목록으로만 분류)")
    for name, keys in REASON_PAT:
        print("   %-18s %s" % (name, ", ".join(keys[:4])))
    tot = collections.Counter()
    src = 0
    for fn in ("x395_compliance_iso.json", "x398_closure_confound.json"):
        p = os.path.normpath(os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026", fn))
        if not os.path.exists(p):
            print("   (%s 없음 — 건너뜀)" % fn)
            continue
        d = json.load(io.open(p, encoding="utf-8"))
        for r in d:
            raw = r.get("raw") or ""
            if not re.search(r'"tool"\s*:\s*null', raw):
                continue
            src += 1
            m = re.search(r'"reason"\s*:\s*"([^"]*)"', raw)
            tot[reason_form(m.group(1) if m else raw)] += 1
    print("\n   기권 %d건" % src)
    for k, v in tot.most_common():
        print("   %-18s %3d  (%.0f%%)" % (k, v, 100.0 * v / max(src, 1)))

    out = os.path.normpath(os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026",
                                        "x399_closure_forms.json"))
    io.open(out, "w", encoding="utf-8").write(json.dumps(
        {"live": rows, "abstain": dict(tot)}, ensure_ascii=False, indent=1))
    print("\n원자료: %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
