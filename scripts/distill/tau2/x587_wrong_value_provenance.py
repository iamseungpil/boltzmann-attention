# -*- coding: utf-8 -*-
r"""x587 - 실패한 변이의 **값이 어디서 왔는지** per-step 으로 귀속한다 (모델 0 · 무료).

## 왜 (2026-08-28 밤 · 사용자 지시 *"통계로 뭉개지 말고 per step 귀속하라"*)

reward 실패의 단위는 변이 집합이다([[69]]). 그런데 *"WRONGARG 몇 건"* 은 원인이 아니다.
원인은 **그 틀린 값이 어느 스텝에서 어디로부터 들어왔는가**다. 이 프로브는 그것만 한다.

## 어떻게 (닫힌 술어 · 추측 0)

  · 호출↔반환은 **tool_call id** 로 맞춘다(`msgs[i].id == tool_call.id`). 한 메시지에 병렬 호출이
    있으면 앞뒤 위치로 추정하는 방식은 **틀린다** - 첫 판에서 실제로 틀렸다(s626729 msg[54] 3중 호출).
  · 값 일치는 `$` 표기 또는 소수 정확 일치만 인정한다. 맨 `9` 는 KB 본문 점수와도 맞아
    오탐한다 - 첫 판에서 실제로 오탐했다(msg[63] KB 결과를 출처로 지목).
  · 판정은 네 갈래:
        OUR_TOOL_TOTAL  우리 비교기가 그 계좌에 대해 낸 총액과 같다 -> **우리가 틀린 값을 건넴**([[25]])
        USER_SAID       role=user 발화에 그 값이 있다 -> 손님 주장을 도구보다 우선([[21]]·[[25]])
        OTHER_TOOL      다른 role=tool 본문에 있다   -> 문맥 내 오배정
        NOT_IN_CONTEXT  어디에도 없다                 -> 자체 계산 또는 날조
  · 우리 비교기가 **그 계좌에 대해 무엇을 말했는지**를 항상 나란히 찍는다([[55]] 순서: 우리 배관 먼저).

⛔집계를 결론으로 쓰지 않는다. 모든 행이 `sim # msg` 로 짚힌다.
"""
import json
import re
import sys

sys.path.insert(0, "/home/woori/workspace_common/boltzmann-attention-pi/scripts/distill/tau2")
import t2_forensic as F

RE_TOTAL = re.compile(r"computed by this tool, is ([-\d.]+)")
RE_ACCT = re.compile(r"(chk_[a-z0-9_]+)")


def money_pat(v):
    try:
        f = float(v)
    except Exception:
        return None
    forms = {("%g" % f), ("%.1f" % f), ("%.2f" % f)}
    if f == int(f):
        forms.add(str(int(f)))
    alt = "|".join(re.escape(x) for x in sorted(forms))
    # $ 가 붙었거나, 소수점 두 자리 이상 정확 표기일 때만 인정한다(KB 점수 오탐 차단)
    return re.compile(r"\$(" + alt + r")(?![\d])|(?<![\d.])(" + re.escape("%.2f" % f) + r")(?![\d])")


def comparator_totals(msgs):
    """tool_call id 로 맞춘 {반환 msg_i: (계좌, 총액)} - 추측 없음."""
    call = {}
    for m in msgs:
        for tc in (m.get("tool_calls") or []):
            a = tc.get("arguments")
            a = a if isinstance(a, str) else json.dumps(a or {}, ensure_ascii=False)
            mm = RE_ACCT.search(a)
            if tc.get("id"):
                call[tc["id"]] = (str(tc.get("name") or ""), mm.group(1) if mm else None)
    out = {}
    for i, m in enumerate(msgs):
        if m.get("role") != "tool" or not m.get("id"):
            continue
        t = RE_TOTAL.findall(str(m.get("content") or ""))
        if not t:
            continue
        nm, acct = call.get(m["id"], ("", None))
        out[i] = (acct, t[-1])
    return out


def attribute(sim):
    msgs = sim.get("messages") or []
    tot = comparator_totals(msgs)
    d = F.mutation_diff(sim, F.mutating_tools(), tag=None) or {}
    rows = []
    for kind in ("wrongarg", "extra"):
        for e in (d.get(kind) or ()):
            if not isinstance(e, dict):
                continue
            args = e.get("args") or {}
            inner = args.get("arguments")
            if isinstance(inner, str):
                try:
                    args = json.loads(inner)
                except Exception:
                    pass
            amt = args.get("amount")
            i = e.get("msg_i")
            if amt is None or i is None:
                continue
            mm = RE_ACCT.search(json.dumps(args, ensure_ascii=False))
            acct = mm.group(1) if mm else None
            ours = None
            for j in sorted(k for k in tot if k < i):
                if tot[j][0] == acct:
                    ours = (j, tot[j][1])
            pat = money_pat(amt)
            src, at = "NOT_IN_CONTEXT", None
            if ours is not None:
                try:
                    if abs(float(ours[1]) - float(amt)) < 1e-9:
                        src, at = "OUR_TOOL_TOTAL", ours[0]
                except Exception:
                    pass
            if src == "NOT_IN_CONTEXT" and pat:
                for j in range(i - 1, -1, -1):
                    if msgs[j].get("role") != "user":
                        continue
                    if pat.search(str(msgs[j].get("content") or "")):
                        src, at = "USER_SAID", j
                        break
            if src == "NOT_IN_CONTEXT" and pat:
                for j in range(i - 1, -1, -1):
                    if msgs[j].get("role") != "tool":
                        continue
                    if pat.search(str(msgs[j].get("content") or "")):
                        src, at = "OTHER_TOOL", j
                        break
            rows.append({"kind": kind, "msg_i": i, "acct": acct, "submitted": amt,
                         "our_total": (ours[1] if ours else None),
                         "our_total_at": (ours[0] if ours else None),
                         "source": src, "source_at": at})
    miss = [e for e in (d.get("missing") or ()) if isinstance(e, dict)]
    return rows, len(miss)


def main(argv=None):
    for tag in ((argv or sys.argv[1:]) or ["bank_t7378_074_20260828"]):
        try:
            sims = F.sims(tag)
        except Exception as e:
            print("(못 읽음) %s : %r" % (tag, e)); continue
        print("#" * 112)
        print("# %s" % tag)
        print("#" * 112)
        for s in sims:
            r = (s.get("reward_info") or {}).get("reward")
            if r == 1.0:
                continue
            rows, nmiss = attribute(s)
            key = F.simtag(s)
            if not rows:
                print("%-24s reward=%-5s  금액 변이 없음 (missing %d) - 다른 실패형" % (key, r, nmiss))
                continue
            print("%-24s reward=%-5s  (missing %d)" % (key, r, nmiss))
            for x in rows:
                print("   msg[%3s] %-20s 제출 %-7s | 우리비교기 %-7s @msg[%s] | 출처 %-15s @msg[%s]"
                      % (x["msg_i"], x["acct"], x["submitted"], x["our_total"],
                         x["our_total_at"], x["source"], x["source_at"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
