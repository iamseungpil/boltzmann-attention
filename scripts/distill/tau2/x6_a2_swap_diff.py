# -*- coding: utf-8 -*-
"""X6-(a) 유한성 증거 — 도메인 스왑 간 선언-필드/기능군 diff (EXPERIMENT_PLAN §1-X6 후단).

측정: 기존 A2 3도메인(banking_knowledge·retail·airline)에서
  ① **엔진이 해석하는 키(기능군)** 집합의 도메인 간 diff — 불변이면 공집합이 기대(유한성)
  ② 각 키의 **내용 크기**(도메인 공급분) — 적응 비용의 프록시
분모 규율(§0c 명확화 1): base 필드/enum만 유한성 분모, `enum_from(domain.*)` 내용은 분모 밖.
주석 키(`_note_*`·`_meta`)는 문서용이므로 제외.
"""
import json, io, glob, os, collections, sys


def load(p):
    return json.load(io.open(p, encoding="utf-8"))


def spec_keys(d):
    """엔진이 해석하는 스펙 키(기능군) — 주석·메타 제외."""
    return {k for k in d if not k.startswith("_")}


def size_of(v):
    """도메인 공급분의 크기(항목 수)."""
    if isinstance(v, dict):
        return len(v)
    if isinstance(v, list):
        return len(v)
    return 1


def main():
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "a2")
    doms = {}
    for p in sorted(glob.glob(os.path.join(base, "*.gate.json"))):
        name = os.path.basename(p).split(".")[0]
        doms[name] = load(p)
    print("=== A2 gate specs: %d domains ===" % len(doms))
    keysets = {k: spec_keys(v) for k, v in doms.items()}
    for name, ks in keysets.items():
        tot = sum(size_of(doms[name][k]) for k in ks)
        print("  %-20s spec-keys=%2d  content-items=%d" % (name, len(ks), tot))

    allk = set().union(*keysets.values())
    common = set.intersection(*keysets.values())
    print("\n=== 기능군(스펙 키) 지형 ===")
    print("  union=%d  intersection=%d" % (len(allk), len(common)))
    print("  공통 키:", ", ".join(sorted(common)))
    print("\n=== 도메인별 고유 키 (유한성 위협 = 엔진 해석기 추가 필요분) ===")
    for name, ks in keysets.items():
        only = ks - set().union(*[v for n, v in keysets.items() if n != name])
        print("  %-20s only=%2d : %s" % (name, len(only), ", ".join(sorted(only)) or "-"))

    # 쌍별 diff
    print("\n=== 쌍별 diff (스왑 시 추가/제거되는 기능군 수) ===")
    names = sorted(keysets)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            add = keysets[b] - keysets[a]
            rem = keysets[a] - keysets[b]
            print("  %s -> %s: +%d / -%d  (add: %s)" %
                  (a, b, len(add), len(rem), ", ".join(sorted(add))[:110] or "-"))

    # 내용 크기 = 적응 비용 프록시
    print("\n=== 공통 기능군의 도메인별 내용 크기 (적응 비용 프록시) ===")
    rows = []
    for k in sorted(common):
        rows.append((k, [size_of(doms[n][k]) for n in names]))
    print("  %-28s %s" % ("key", "  ".join("%-12s" % n[:12] for n in names)))
    for k, sizes in rows:
        print("  %-28s %s" % (k, "  ".join("%-12d" % s for s in sizes)))


if __name__ == "__main__":
    main()
