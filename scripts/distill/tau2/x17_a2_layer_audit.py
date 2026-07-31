# -*- coding: utf-8 -*-
"""X17 — A2 계층 분리 감사 + **capex/opex 회계** (2026-07-31·무료).

사용자 지시: "A2가 도메인 일반과 도메인 특화로 확실히 분리돼 있나? capex/opex 계산할 때 확실히
분리할 수 있게 **별도 파일**로 만들어라."

★두 가지를 구분해야 회계가 정직해진다:
  ① **층(layer)** — CORE / EXT / DISCARD. 이미 `a2/split/<domain>.<layer>.json`으로 갈려 있다.
     그러나 이건 **레버 계열** 분류이지 비용 분류가 아니다(CORE에도 도메인 내용이 들어간다).
  ② **출처(provenance)** — **도메인마다 새로 써야 하는가**. 이게 opex의 정의다.
     판정은 기계로 한다: **3도메인에서 값이 바이트-동일한 키 = BASE(1회 저작)**,
     그 외 = **DOMAIN(도메인마다 저작)**.

산출:
  · 등가 검증: merge(core, ext, discard) == 단일 파일 (분리가 **내용 손실 없음**을 증명)
  · 비용표: 도메인별 BASE/DOMAIN 키 수·직렬화 바이트
  · `--emit`: BASE 키를 `a2/base/shared.json`으로, 나머지를 `a2/<domain>.domain.json`으로 기록

용법: py -3 x17_a2_layer_audit.py [--emit]
"""
import argparse
import io
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_A2 = os.path.join(_HERE, "a2")
DOMAINS = ["banking_knowledge", "retail", "airline"]
LAYERS = ["core", "ext", "discard"]


def load(path):
    with io.open(path, encoding="utf-8") as f:
        return json.load(f)


def real_keys(d):
    return {k: v for k, v in d.items() if not k.startswith("_")}


def canon(v):
    return json.dumps(v, sort_keys=True, ensure_ascii=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emit", action="store_true", help="BASE/DOMAIN 파일 기록")
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    mono, layers = {}, {}
    for dom in DOMAINS:
        p = os.path.join(_A2, dom + ".gate.json")
        if not os.path.exists(p):
            print("  ⚠%s 단일 파일 없음" % dom)
            continue
        mono[dom] = real_keys(load(p))
        layers[dom] = {}
        for lay in LAYERS:
            q = os.path.join(_A2, "split", "%s.%s.json" % (dom, lay))
            layers[dom][lay] = real_keys(load(q)) if os.path.exists(q) else {}

    print("=" * 78)
    print("① 등가 검증 — merge(core, ext, discard) == 단일 파일")
    print("=" * 78)
    all_ok = True
    for dom in mono:
        merged = {}
        dup = []
        for lay in LAYERS:
            for k, v in layers[dom][lay].items():
                if k in merged:
                    dup.append(k)
                merged[k] = v
        missing = sorted(set(mono[dom]) - set(merged))
        extra = sorted(set(merged) - set(mono[dom]))
        diff = sorted(k for k in set(merged) & set(mono[dom]) if canon(merged[k]) != canon(mono[dom][k]))
        ok = not (missing or extra or diff or dup)
        all_ok &= ok
        print("  %-20s 키 %2d = core %2d + ext %2d + discard %2d  → %s"
              % (dom, len(mono[dom]), len(layers[dom]["core"]), len(layers[dom]["ext"]),
                 len(layers[dom]["discard"]), "✅등가" if ok else "❌불일치"))
        for label, items in (("누락", missing), ("잉여", extra), ("값 다름", diff), ("중복", dup)):
            if items:
                print("      %s: %s" % (label, items[:8]))
    print("  ⇒ 분리가 내용을 잃지 않는가: %s" % ("**예**" if all_ok else "**아니오 — 먼저 고칠 것**"))

    print("\n" + "=" * 78)
    print("② ★비용 회계 — BASE(3도메인 값 동일=1회 저작) vs DOMAIN(도메인마다 저작)")
    print("=" * 78)
    counts = {}
    for dom in mono:
        for k in mono[dom]:
            counts.setdefault(k, {})[dom] = canon(mono[dom][k])
    base_keys, dom_keys = [], []
    for k, per in sorted(counts.items()):
        vals = set(per.values())
        if len(per) >= 2 and len(vals) == 1:
            base_keys.append(k)
        else:
            dom_keys.append(k)
    print("  **BASE**(≥2도메인에 있고 값 동일) %d키: %s" % (len(base_keys), base_keys or "(없음)"))
    print("  DOMAIN(도메인 저작) %d키" % len(dom_keys))
    print()
    print("  %-20s %6s %8s %8s %10s" % ("도메인", "총키", "BASE키", "DOM키", "DOM바이트"))
    for dom in mono:
        b = [k for k in mono[dom] if k in base_keys]
        dk = [k for k in mono[dom] if k not in base_keys]
        nb = sum(len(canon(mono[dom][k])) for k in dk)
        print("  %-20s %6d %8d %8d %10d" % (dom, len(mono[dom]), len(b), len(dk), nb))
    print("\n  ⚠**층(CORE/EXT)과 비용(BASE/DOMAIN)은 다른 축이다** — CORE에도 도메인 내용이 들어간다.")
    print("   새 도메인 1개의 opex = 위 **DOM키/DOM바이트**이지 CORE 유무가 아니다.")

    # 층 × 비용 교차표
    print("\n  층 × 비용 교차(banking):")
    dom = "banking_knowledge"
    if dom in layers:
        for lay in LAYERS:
            ks = list(layers[dom][lay])
            b = [k for k in ks if k in base_keys]
            print("    %-8s 총 %2d · BASE %2d · DOMAIN %2d  %s"
                  % (lay, len(ks), len(b), len(ks) - len(b), ("BASE=" + ",".join(b)) if b else ""))

    if args.emit:
        outb = os.path.join(_A2, "base")
        os.makedirs(outb, exist_ok=True)
        shared = {}
        for k in base_keys:
            for dom in mono:
                if k in mono[dom]:
                    shared[k] = mono[dom][k]
                    break
        shared["_note"] = ("X17 생성물: 3도메인에서 **값이 바이트-동일**한 키만 모았다 = 1회 저작 = "
                           "새 도메인 추가 시 opex 0. 손으로 고치지 말 것(재생성).")
        io.open(os.path.join(outb, "shared.json"), "w", encoding="utf-8", newline="\n").write(
            json.dumps(shared, ensure_ascii=False, indent=1))
        print("\n[emit] a2/base/shared.json (%d키)" % len(base_keys))
        for dom in mono:
            only = {k: v for k, v in mono[dom].items() if k not in base_keys}
            only["_note"] = ("X17 생성물: 이 도메인이 **저작해야 하는** 내용(opex 계상 대상). "
                             "`a2/base/shared.json`과 병합하면 단일 파일과 등가.")
            io.open(os.path.join(_A2, "%s.domain.json" % dom), "w",
                    encoding="utf-8", newline="\n").write(json.dumps(only, ensure_ascii=False, indent=1))
            print("[emit] a2/%s.domain.json (%d키)" % (dom, len(only) - 1))

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
