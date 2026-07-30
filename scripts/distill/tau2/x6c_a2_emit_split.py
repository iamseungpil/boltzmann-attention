# -*- coding: utf-8 -*-
"""X6-(c) A2 파일 2층 실분할 — `<domain>.core.json` + `<domain>.ext.json` (+ `.discard.json`).

지시(2026-07-30 사용자): "A2를 도메인 특화로 2개로 쪼개야 하면 쪼개라."
분류 기준은 `x6b_a2_split.py`의 CORE/DISCARD 집합을 그대로 재사용(단일 정본).

**엔진 로더는 건드리지 않는다**(설계→리뷰→구현 규율): 본 스크립트는 분할 파일을 *생성*만 하고,
기존 `<domain>.gate.json`은 그대로 남긴다. 병합 로더(core+ext) 배선은 별건 리뷰 대상이며,
분할 파일이 원본과 무손실인지(키·내용 동일) 자체 검증까지 수행한다.

산출: a2/split/<domain>.{core,ext,discard}.json + 무손실 검증 리포트
"""
import json, io, os, glob, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from x6b_a2_split import CORE, DISCARD, cls   # 분류 정본 재사용


def main():
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "a2")
    out = os.path.join(base, "split")
    os.makedirs(out, exist_ok=True)
    ok = True
    for p in sorted(glob.glob(os.path.join(base, "*.gate.json"))):
        dom = os.path.basename(p).split(".")[0]
        d = json.load(io.open(p, encoding="utf-8"))
        layers = {"core": {}, "ext": {}, "discard": {}}
        notes = {k: v for k, v in d.items() if k.startswith("_")}
        for k, v in d.items():
            if k.startswith("_"):
                continue
            layers[cls(k).lower()][k] = v
        for name, obj in layers.items():
            payload = dict(obj)
            payload["_meta"] = {
                "domain": dom, "layer": name,
                "source": os.path.basename(p),
                "classifier": "x6b_a2_split.CORE/DISCARD (2026-07-30)",
                "note": {
                    "core": "도메인-불변 기대 기능군 — 유한성 주장의 분모(X6-c로 검정)",
                    "ext": "도메인-특화 스펙 = 정직한 per-domain opex (특허 유한성 주장에서 제외)",
                    "discard": "P2-b 폐기·anti-drift 금지분 (스위치-오버 시 기본-OFF)",
                }[name],
            }
            fp = os.path.join(out, "%s.%s.json" % (dom, name))
            io.open(fp, "w", encoding="utf-8").write(
                json.dumps(payload, ensure_ascii=False, indent=1))
        # 무손실 검증: 3층 합집합 == 원본 비주석 키, 내용 동일
        union = {}
        for name in layers:
            union.update(layers[name])
        orig = {k: v for k, v in d.items() if not k.startswith("_")}
        same_keys = set(union) == set(orig)
        same_vals = all(json.dumps(union[k], sort_keys=True) == json.dumps(orig[k], sort_keys=True)
                        for k in orig)
        ok = ok and same_keys and same_vals
        print("%-20s core=%2d ext=%2d discard=%2d | notes=%d | lossless keys=%s vals=%s"
              % (dom, len(layers["core"]), len(layers["ext"]), len(layers["discard"]),
                 len(notes), same_keys, same_vals))
    print("\nALL LOSSLESS:", ok, "->", out)
    print("엔진 로더 미배선(설계→리뷰→구현 규율) — 병합 로더는 별건 리뷰")


if __name__ == "__main__":
    main()
