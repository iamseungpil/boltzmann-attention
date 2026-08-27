# -*- coding: utf-8 -*-
"""일회성 — `a2/split/banking_knowledge.core.json` 의 `scaffold_get_tools` 를 정본에 맞춘다.

## 왜 (2026-08-28)

`test_atm_fee_op` ⑷ 가 **3사본 바이트 동일**을 요구하는데 그 검정이 언제부턴가 붉었고, 그
래칫은 **어느 드라이버의 배터리에도 없어서** 아무도 못 봤다. 실측:

    gate ↔ specific   10/10 동일   ← 라이브 두 층은 맞아 있다
    specific ↔ split  **6/10 어긋남**
      check_card_closure_eligibility · check_cli_eligibility · get_atm_fee_discrepancies
      get_correct_savings_apy · get_interest_correction · get_reward_discrepancies

가장 최근 원인은 `2109a64e`(2026-08-26 `{delta_total}` 복귀)가 이 사본을 빼먹은 것이지만
어긋남은 그보다 넓다. **런타임은 이 파일을 안 읽는다** — 참조처는 `_ins_*`/`_upd_*` 일회성
프로그램들뿐이다(`grep -rln "a2/split"`). 그래서 死배선 위험은 없고, 위험은 다른 데 있다:
**다음 사람이 틀린 사본을 고친다.** 그리고 붉은 래칫은 배터리 전체를 못 믿게 만든다.

## 무엇을 하나

`specific`(= `gate`)의 `scaffold_get_tools` 항목을 이름으로 맞춰 **그대로 복사**한다.
저작 0 · 판단 0 · 다른 키는 건드리지 않는다. 어느 키가 왜 달랐는지 인쇄한다.
"""
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

CANON = os.path.join(HERE, "a2", "banking_knowledge.specific.json")
PEER = os.path.join(HERE, "a2", "banking_knowledge.gate.json")
MIRROR = os.path.join(HERE, "a2", "split", "banking_knowledge.core.json")


def find_holder(o):
    """`scaffold_get_tools` 를 든 **dict 자체**를 돌려준다(교체하려면 소유자가 필요하다)."""
    if isinstance(o, dict):
        if "scaffold_get_tools" in o:
            return o
        for v in o.values():
            r = find_holder(v)
            if r is not None:
                return r
    if isinstance(o, list):
        for v in o:
            r = find_holder(v)
            if r is not None:
                return r
    return None


def main():
    canon = json.load(io.open(CANON, encoding="utf-8"))
    peer = json.load(io.open(PEER, encoding="utf-8"))
    mirror = json.load(io.open(MIRROR, encoding="utf-8"))
    hc, hp, hm = find_holder(canon), find_holder(peer), find_holder(mirror)
    if not (hc and hp and hm):
        print("⛔ scaffold_get_tools 를 못 찾았다 — 중단.")
        return 2

    # ⛔정본 둘이 어긋나 있으면 **여기서 고르지 않는다** — 어느 쪽이 옳은지는 사람이 정한다.
    cmap = {t.get("name"): t for t in hc["scaffold_get_tools"]}
    pmap = {t.get("name"): t for t in hp["scaffold_get_tools"]}
    bad = [n for n in set(cmap) | set(pmap) if cmap.get(n) != pmap.get(n)]
    if bad:
        print("⛔ 라이브 두 층(gate·specific)이 %d 개에서 어긋난다 — 거울을 맞추기 전에 그것부터"
              " 정해야 한다: %s" % (len(bad), sorted(bad)[:4]))
        return 2
    print("정본 두 층 동일 확인 (도구 %d)" % len(cmap))

    changed = []
    for i, t in enumerate(list(hm["scaffold_get_tools"])):
        n = t.get("name")
        c = cmap.get(n)
        if c is None:
            print("  %-38s 정본에 없다 — 그대로 둔다" % n)
            continue
        if t == c:
            continue
        keys = sorted(set(t) | set(c))
        why = [k for k in keys if t.get(k) != c.get(k)]
        hm["scaffold_get_tools"][i] = json.loads(json.dumps(c, ensure_ascii=False))
        changed.append(n)
        print("  %-38s 맞춤 · 달랐던 키 %d: %s" % (n, len(why), ", ".join(why[:5])))

    extra = [t.get("name") for t in hm["scaffold_get_tools"] if t.get("name") not in cmap]
    if extra:
        print("  ⚠거울에만 있는 도구(그대로 둔다): %s" % extra)

    if not changed:
        print("이미 동일 — 쓰지 않는다")
        return 0
    with io.open(MIRROR, "w", encoding="utf-8", newline="\n") as f:
        json.dump(mirror, f, ensure_ascii=False, indent=1)
        f.write("\n")
    print("맞춘 도구 %d: %s" % (len(changed), ", ".join(changed)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
