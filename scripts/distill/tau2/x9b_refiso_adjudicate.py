#!/usr/bin/env python3
"""U3' 페르소나 프로브 판정 — 기권/값 축을 분리해 재집계 (2026-07-30 야간).

x9의 요약은 응답 **전문 동등**으로 일치를 셌다. 그래서 양쪽 다 기권(UNSURE)인데 뒤 산문만 다른
쌍이 "불일치"로 들어간다(7B 판 §6 각주가 지적한 그 아티팩트가 32B에서 45건을 만들었다).
행동이 바뀌었는지 판정하려면 축을 갈라야 한다:

  ① **기권 결정**(UNSURE 여부) — 이게 실제 행동 분기다(엔진은 UNSURE면 개입하지 않는다)
  ② **값 선택** — 양쪽이 답을 낸 쌍에서만 의미가 있다

따라서 판정은 세 칸으로 나온다: 둘 다 기권(행동 동일) / 기권 결정이 갈림(**행동 변화**) /
둘 다 답했고 값이 갈림(**행동 변화**). 산문 차이는 어느 칸에도 들어가지 않는다.

용법: py -3 x9b_refiso_adjudicate.py <rows.jsonl>
"""
import argparse
import gzip
import json
import os
import sys
from collections import Counter


def is_abstain(s):
    """엔진 관점의 기권 = 응답이 UNSURE로 시작(또는 UNSURE만 있음)."""
    t = str(s or "").strip().upper()
    return t.startswith("UNSURE") or t == ""


def val(s):
    """값 응답의 정규화 — 첫 토큰(따옴표·구두점 제거). UNSURE면 None."""
    if is_abstain(s):
        return None
    t = str(s).strip().strip("'\"`").split()
    return t[0].strip("'\"`.,;:") if t else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rows")
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    # 영속화 규율([[30]]·야간 함정 3): 리모트 repo에는 `.gz`만 남긴다 ⇒ 판정 도구가 gz를 직접 읽어야
    # 평문 사본을 만들 이유가 없어진다.
    _open = (lambda p: gzip.open(p, "rt", encoding="utf-8")) if args.rows.endswith(".gz") \
        else (lambda p: open(p, encoding="utf-8"))
    rows = [json.loads(l) for l in _open(args.rows) if l.strip()]
    ok = [r for r in rows if "error" not in r]
    print(f"행 {len(rows)} · 유효 {len(ok)} · 에러 {len(rows) - len(ok)}")
    if not ok:
        sys.exit("유효 행 0")
    eps = {r.get("endpoint") for r in ok}
    mods = {r.get("model") for r in ok}
    print(f"endpoint={eps} model={mods}")

    cls = Counter()
    changed, val_diff = [], []
    for r in ok:
        ab, at = is_abstain(r["base"]), is_abstain(r["treat"])
        if ab and at:
            cls["둘 다 기권(행동 동일)"] += 1
        elif ab != at:
            cls["기권 결정이 갈림(★행동 변화)"] += 1
            changed.append(r)
        else:
            vb, vt = val(r["base"]), val(r["treat"])
            if vb == vt:
                cls["둘 다 같은 값(행동 동일)"] += 1
            else:
                cls["값이 갈림(★행동 변화)"] += 1
                val_diff.append(r)
    n = len(ok)
    print("\n" + "=" * 72)
    print("판정 (기권/값 축 분리 · 산문 차이 배제)")
    print("=" * 72)
    for k, v in sorted(cls.items(), key=lambda x: -x[1]):
        print(f"  {k:32s} {v:4d}  ({100.0 * v / n:.1f}%)")
    same = cls["둘 다 기권(행동 동일)"] + cls["둘 다 같은 값(행동 동일)"]
    print(f"\n  ⇒ 행동 동일 {same}/{n} = {100.0 * same / n:.1f}% · "
          f"행동 변화 {n - same}/{n} = {100.0 * (n - same) / n:.1f}%")

    print(f"\n기권율: base {sum(is_abstain(r['base']) for r in ok)}/{n} · "
          f"treat {sum(is_abstain(r['treat']) for r in ok)}/{n}")

    # ★해상도의 정직한 분모: 같은 케이스의 시드 3개는 독립 표본이 아니다(temp 0에서는 특히).
    #   값 축의 유효 표본 = **양쪽이 답한 서로 다른 케이스 수**이고, 양쪽 기권 케이스는
    #   불변의 증거가 아니라 **구별력 없음**(no-information)이다.
    # ⚠자기정정: 초판이 `sim`으로 묶어 "케이스 4개"를 냈다. 한 궤적(sim)에 REF_ISO 적용 케이스가
    #   여러 개 있으므로 케이스 축은 `i`(케이스 인덱스)다. 궤적 축은 그보다 더 거친 상관 단위라
    #   **둘 다** 보고한다(궤적 4개에서 나온 27케이스는 27개의 독립 표본이 아니다).
    def group(keyf):
        g = {}
        for r in ok:
            g.setdefault(keyf(r), []).append(r)
        return g

    for label, keyf in (("케이스(i)", lambda r: r.get("i")),
                        ("궤적(sim)", lambda r: r.get("sim"))):
        g = group(keyf)
        both_ans = [c for c, rs in g.items()
                    if any(not is_abstain(r["base"]) and not is_abstain(r["treat"]) for r in rs)]
        all_abst = [c for c, rs in g.items()
                    if all(is_abstain(r["base"]) and is_abstain(r["treat"]) for r in rs)]
        print(f"{label} {len(g)}개 중 값 축 유효(양쪽 답함 시드 ≥1) = **{len(both_ans)}** · "
              f"전 시드 양쪽 기권 = {len(all_abst)}")
    print("  ⇒ 값 축 결론의 분모는 위 '값 축 유효' 수다(쌍 %d개가 아니다)." % n)

    if changed:
        print(f"\n--- 기권 결정이 갈린 {len(changed)}건 (어느 arm이 답했나) ---")
        d = Counter("treat만 답함" if is_abstain(r["base"]) else "base만 답함" for r in changed)
        for k, v in d.items():
            print(f"  {k}: {v}")
        for r in changed[:12]:
            who = "treat" if is_abstain(r["base"]) else "base"
            print(f"  sim={str(r['sim'])[:14]} seed={r['seed']} {who} 답={val(r[who])} "
                  f"agent_live={r.get('agent_chose')}")
    if val_diff:
        print(f"\n--- 값이 갈린 {len(val_diff)}건 ---")
        for r in val_diff[:12]:
            print(f"  sim={str(r['sim'])[:14]} seed={r['seed']} base={val(r['base'])} "
                  f"treat={val(r['treat'])} agent_live={r.get('agent_chose')}")

    print("\n⚠판정 규율([[08]]): 위 '행동 변화' 칸이 0이 아니면 U3'는 행동-불변이 아니다. "
          "정확도 판정은 gold 없이는 불가 — 별건(§C 게이트)으로 gold 대조가 필요하다.")


if __name__ == "__main__":
    main()
