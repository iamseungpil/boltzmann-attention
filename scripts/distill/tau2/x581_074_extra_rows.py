# -*- coding: utf-8 -*-
r"""x581 — 074 의 **초과 행**이 무엇인지 원장과 맞대어 센다 (모델 0 · 무료 · 계수만).

## 왜 (2026-08-28 저녁)

072 를 닫고 나서 남은 같은-도구 결함이 074 인데, **방향이 반대**다:

    072 Light Green  전사 sub=9 /kind=10  — **부족** → 도구총액 5.00   (gold 3.50)
    074 chk_..._2    전사 sub=17/kind=16  — **초과** → 도구총액 18.50  (gold 14.50)

`x542` 가 같은 계좌를 `+2.50+2.50+2.50+4.00+4.00+1.50−2.50 = **14.50**`(7행)로 적어 뒀는데
t7372 에서는 **8행·18.50** 이다 ⇒ **행 하나·정확히 +4.00** 이 더 붙었다.

그 한 행이 무엇인가가 이 프로브의 유일한 물음이다. `x525` 주석이 세 후보를 적어 뒀다:

    ⒜ 날조 id      (`btxn_ar_lb_08f` 를 냈는데 원장엔 `btxn_ar_lb_08f_err` 뿐)
    ⒝ 이중 덮음    (인출 id + 같은 날 수수료 id 를 **둘 다** 냈다)
    ⒞ 같은 날 수수료 2건을 각각 한 행으로

## 무엇을 세나 (판단 0 · 닫힌 술어뿐)

비교기 반환의 `btxn_...` 각각에 대해 **원장에 실재하는가 · 어떤 type 인가 · 중복인가**.
원장은 그 계좌의 `get_bank_account_transactions_9173` 출력(`Record ID:` 덤프)이다.
⛔gold 는 안 본다 — 14.50 은 주석의 참고값이고 검정에 안 들어간다([[23]]).

사용: (리모트) PYTHONPATH=. py -3 x581_074_extra_rows.py [--tags a,b] [--task task_074]
"""
import argparse
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

import t2_forensic as F                                             # noqa: E402

OUT = os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026")
ANCHOR = "whose net charge does NOT match"
RE_ROW = re.compile(r"(btxn_\w+) \(charged \$([\d.]+), documented fee \$([\d.]+), "
                    r"difference \$(-?[\d.]+)\)")
RE_TOT = re.compile(r"computed by this tool, is ([-\d.]+)")
RE_REC = re.compile(r"Record ID:\s*(\S+)")


def ledger_of(msgs, upto):
    """`upto` 이전의 **가장 가까운** 레코드 덤프 → {id: type}. 위치 찾기뿐(해석 0)."""
    for j in range(upto - 1, -1, -1):
        c = str((msgs[j] or {}).get("content") or "")
        if "Record ID:" not in c:
            continue
        out = {}
        for blk in re.split(r"\n\s*\n", c):
            m = RE_REC.search(blk)
            if not m:
                continue
            t = re.search(r"(?im)^\s*type:\s*(\w+)", blk)
            out[m.group(1)] = (t.group(1) if t else "?")
        if out:
            return out, j
    return {}, -1


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="task_074")
    ap.add_argument("--tags", default="bank_t7372_control_20260828")
    a = ap.parse_args(argv)

    rows_out = []
    for tag in [t.strip() for t in a.tags.split(",") if t.strip()]:
        try:
            sims = F.sims(tag)
        except Exception as e:
            print("[skip] %s — %r" % (tag, e))
            continue
        for s in sims:
            if F.task_id(s) != a.task:
                continue
            msgs = s.get("messages") or []
            print("=" * 96)
            print("== %s · %s · reward=%s"
                  % (tag[5:30], F.simtag(s), (s.get("reward_info") or {}).get("reward")))
            for i, m in enumerate(msgs):
                c = str(m.get("content") or "")
                if ANCHOR not in c:
                    continue
                rows = RE_ROW.findall(c)
                if not rows:
                    continue
                tot = RE_TOT.search(c)
                led, li = ledger_of(msgs, i)
                signed = round(sum(float(r[3]) for r in rows), 2)
                cnt = collections.Counter(r[0] for r in rows)
                kinds = collections.Counter(led.values())
                print("  msg[%3d] 판정행 %d · 부호합 %.2f · 도구총액 %s · 원장 msg[%d] %s"
                      % (i, len(rows), signed, (tot.group(1) if tot else "없음"), li, dict(kinds)))
                rec = {"tag": tag, "sim": F.simtag(s), "msg": i, "n": len(rows),
                       "signed": signed, "tool_total": (tot.group(1) if tot else None),
                       "ledger_kinds": dict(kinds), "rows": []}
                for rid, ch, doc, dif in rows:
                    typ = led.get(rid)
                    flag = []
                    if typ is None:
                        flag.append("★원장에 없다(날조)")
                    elif typ != "atm_withdrawal":
                        flag.append("type=%s (인출이 아니다)" % typ)
                    if cnt[rid] > 1:
                        flag.append("★중복 ×%d" % cnt[rid])
                    print("      %-26s charged %-7s doc %-7s diff %-7s %s"
                          % (rid, ch, doc, dif, " · ".join(flag)))
                    rec["rows"].append({"id": rid, "charged": ch, "documented": doc,
                                        "difference": dif, "ledger_type": typ,
                                        "dup": cnt[rid], "flags": flag})
                # ★차액별 묶음 — `+4.00` 이 몇 개인지가 이 프로브의 표적이다
                byd = collections.Counter(r[3] for r in rows)
                print("      차액 분포: %s" % dict(byd))
                rec["by_diff"] = dict(byd)
                rows_out.append(rec)

    if not rows_out:
        print("비교기 반환을 못 찾았다 — 판정하지 않는다([[25]])", file=sys.stderr)
        return 2

    print("")
    print("=" * 96)
    print("요약 — 계좌별 판정행 수 · 도구총액 · 날조/중복")
    print("=" * 96)
    for r in rows_out:
        fab = sum(1 for x in r["rows"] if x["ledger_type"] is None)
        dup = sum(1 for x in r["rows"] if x["dup"] > 1)
        nonw = sum(1 for x in r["rows"] if x["ledger_type"] not in (None, "atm_withdrawal"))
        print("  %-20s msg[%3d] 행 %-3d 총액 %-7s · 날조 %d · 중복 %d · 비인출 %d"
              % (r["sim"], r["msg"], r["n"], r["tool_total"], fab, dup, nonw))
    print("")
    print("판독:")
    print("  ⒜ 날조 > 0 이면 원인은 **원장에 없는 id 를 서브가 지어낸 것**이다.")
    print("  ⒝ 중복 > 0 이면 **같은 행을 두 번 넘긴 것**이다.")
    print("  ⒞ 둘 다 0 인데 행이 기대보다 많으면, 원인은 id 가 아니라 **인출↔수수료 짝짓기**다")
    print("     (같은 인출을 수수료 줄 수만큼 쪼갰다) — 그때는 `비인출` 칸이 신호다.")

    dst = os.path.join(OUT, "x581_074_extra_rows_2026_08_28.json")
    with io.open(dst, "w", encoding="utf-8") as f:
        json.dump({"probe": "x581_074_extra_rows", "date": "2026-08-28",
                   "task": a.task, "tags": a.tags, "rows": rows_out,
                   "limits": ["원장은 **그 반환 직전의 가장 가까운** 레코드 덤프로 잡았다 — "
                              "계좌를 id 로 맞댄 것이 아니라 위치로 잡은 근사다.",
                              "gold 는 안 봤다. 14.50 은 주석의 참고값이고 검정에 안 들어간다.",
                              "행 수·차액은 **비교기 반환문**을 읽은 것이지 서브 산출 원문이 아니다 "
                              "(그 원문은 사이드카에 있다)."]},
                  f, ensure_ascii=False, indent=1)
    print("")
    print("-> %s" % os.path.normpath(dst))
    return 0


if __name__ == "__main__":
    sys.exit(main())
