# -*- coding: utf-8 -*-
"""x162 — **우리 층이 파는 우물을 센다** (유료 0·§6.0/x159⑸ 의 직접 계량).

왜: §6.0 은 표가 안 듣는 이유를 *"전 후보 동시 언급 = 차등 0"* 으로 설명하고, x159⑸ 는 λ 가 큰
항의 성분을 **종단의 단일-후보 언급**으로 특정했다(무주장 bare mention p=0.999 = 선언 p=1.000).
그렇다면 **우리 자신이 내보내는 문장**도 같은 자에 걸린다 — 우리가 한 후보만 이름 대면 그것이
바로 큰 λ 항이고, 그 후보가 오답이면 우리는 오답 우물을 파는 장치다(등대 §1 제1원리: 부작용
없는 레버는 없다·게이트 자신도 역효과).

이 도구가 재는 것 (사이드카 = 우리 층 발화의 유일한 기록):
  ⒜ 레코드를 **단일-후보 / 다-후보 / 무-후보** 로 분해한다 — 단일-후보가 고-λ 채널이다
  ⒝ 그 단일-후보 언급이 **어느 이름에 쏠리는가** (오답에 쏠리면 우리가 앵커를 판다)
  ⒞ **채널별**로 나눈다 (어느 레버가 그 항을 만드는가 = 끌 대상이 아니라 **고칠 대상**)
  ⒟ **턴 분포** — 종단성이 λ 의 성분이므로 늦은 턴의 단일-후보 언급이 가장 무겁다
  ⒠ 다-후보 레코드 안에서 이름별 **등장 횟수가 균등한가** (안 균등하면 표도 차등을 만든다)

후보 이름은 **A3 정책 온톨로지에서** 가져온다 — 이 스크립트가 도메인 어휘를 짓지 않는다([[59]]).
실행: py -3 x162_our_mention_load.py <fb_TAG.jsonl>
"""
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import t2_factdag as FD                                        # noqa: E402
from gate_interpreter import load_domain_a2                     # noqa: E402

DOMAIN = "banking_knowledge"


def subjects():
    a2 = load_domain_a2(DOMAIN) or {}
    rows = (a2.get("policy_ontology") or {}).get("rows") or ()
    spec = next((s for s in (a2.get("ledger_metrics") or []) if s.get("eligible_text")), {})
    axes = list((spec.get("eligible") or {}).get("show_axes") or ())
    names = set()
    for ax in axes:
        names |= set(FD._a3_map(rows, {"axis": ax}) or {})
    # 축 선언이 비면 원장 전 행에서 주어를 모은다(계기가 조용히 0을 내지 않게)
    if not names:
        for r in rows:
            s = (r or {}).get("subject")
            if s:
                names.add(s)
    return sorted(names, key=len, reverse=True)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    names = subjects()
    if not names:
        print("A3 주어 0 — 계기 실패이지 결과 아님")
        return 1
    recs = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8") if l.strip()]
    print("사이드카 레코드 %d · A3 주어 %d" % (len(recs), len(names)))

    single = collections.Counter()          # 단일-후보 언급이 어느 이름에 쏠리나
    single_ch = collections.Counter()       # 그 항을 만드는 채널
    kinds = collections.Counter()
    per_ch = collections.defaultdict(collections.Counter)
    turns_single, turns_multi = [], []
    uneven = []

    for r in recs:
        txt = r.get("text") or ""
        hit = collections.Counter()
        for n in names:                     # 긴 이름부터 세고 지운다(부분 문자열 이중계수 방지)
            c = txt.count(n)
            if c:
                hit[n] = c
                txt = txt.replace(n, " ")
        ch = r.get("channel") or "?"
        if not hit:
            kinds["무-후보"] += 1
            per_ch[ch]["무-후보"] += 1
            continue
        if len(hit) == 1:
            kinds["단일-후보"] += 1
            per_ch[ch]["단일-후보"] += 1
            single[next(iter(hit))] += 1
            single_ch[ch] += 1
            turns_single.append(r.get("turn"))
        else:
            kinds["다-후보"] += 1
            per_ch[ch]["다-후보"] += 1
            turns_multi.append(r.get("turn"))
            lo, hi = min(hit.values()), max(hit.values())
            if hi > lo:                     # 표 안에서도 차등이 생기는가
                uneven.append((r.get("channel"), r.get("turn"), hit.most_common(3)))

    print("\n=== ⒜ 레코드 분해 (고-λ 채널 = 단일-후보) ===")
    for k, v in kinds.most_common():
        print("  %-10s %3d  (%.0f%%)" % (k, v, 100.0 * v / len(recs)))

    print("\n=== ⒝ 단일-후보 언급이 쏠리는 이름 ===")
    for n, c in single.most_common(10):
        print("  %-42s %3d" % (n, c))

    print("\n=== ⒞ 그 항을 만드는 채널 ===")
    for c, v in single_ch.most_common():
        print("  %-24s %3d" % (c, v))

    print("\n=== ⒟ 턴 분포 (종단성 = λ 성분) ===")
    for label, ts in (("단일-후보", turns_single), ("다-후보", turns_multi)):
        ts = [t for t in ts if isinstance(t, int)]
        if ts:
            ts_sorted = sorted(ts)
            print("  %-10s n=%3d  min=%2d  중앙=%2d  max=%2d" %
                  (label, len(ts), ts_sorted[0], ts_sorted[len(ts_sorted) // 2], ts_sorted[-1]))

    print("\n=== ⒠ 다-후보 레코드 안의 차등 (표가 정말 균등한가) ===")
    print("  불균등 레코드 %d / %d" % (len(uneven), len(turns_multi)))
    for ch, tn, top in uneven[:6]:
        print("    %-18s turn=%-3s %s" % (ch, tn, top))

    print("\n=== 채널별 전체 ===")
    for ch, cc in sorted(per_ch.items(), key=lambda kv: -sum(kv[1].values())):
        print("  %-24s %s" % (ch, dict(cc)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
