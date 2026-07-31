# -*- coding: utf-8 -*-
"""X18 — A2 3층 분리·감사 (2026-07-31·무료).

설계 정본 = `A2_THREE_LAYER_SPLIT_DESIGN_2026_07_31.md`. 사용자 정의(축자):
  L1 shared   = 벤치마크 공통·수정 0으로 그대로 씀        → `a2/base/shared.json`
  L2 settings = 도메인별로 값은 바꾸나 **구조는 동일**      → `a2/<domain>.settings.json`
  L3 specific = 그 도메인에만 있는 도구·규칙(엔진과 callback) → `a2/<domain>.specific.json`

판정은 **키의 도메인 분포**로 한다(값이 아니라):
  · 모든 도메인이 같은 값 → L1   · 2+ 도메인에 존재 → L2   · 한 도메인에만 존재 → L3
★"env에서 자동 도출되면 L2"라는 더 강한 기준은 **반증됐다**(설계 §1-1): `action_tools`의 자연스러운
  생성 규칙 `mutating ∩ exposed`는 10개를 내는데 저장값은 7개다. 규칙을 7개에 맞추는 것은 정답에
  생성기를 맞추는 것이라 금지([[03b]]). 그래서 자동화는 **주장하지 않는다**.

용법:
  py -3 x18_a2_three_layer.py            # 분류·비용표만 출력(파일 안 건드림)
  py -3 x18_a2_three_layer.py --emit     # settings/specific 기록 + 등가 검증
  py -3 x18_a2_three_layer.py --verify   # 병합 == <domain>.gate.json 인지만 검사(테스트용·종료코드)
"""
import argparse
import io
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_A2 = os.path.join(_HERE, "a2")
DOMAINS = ["banking_knowledge", "retail", "airline"]


def load(path):
    with io.open(path, encoding="utf-8") as f:
        return json.load(f)


def dump(path, obj):
    with io.open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(obj, ensure_ascii=False, indent=1))


def canon(v):
    return json.dumps(v, sort_keys=True, ensure_ascii=False)


def real(d):
    return {k: v for k, v in d.items() if not k.startswith("_")}


def classify(mono):
    """키 → 층. mono = {domain: 단일파일 dict}."""
    present, layer = {}, {}
    for dom, d in mono.items():
        for k in real(d):
            present.setdefault(k, {})[dom] = canon(d[k])
    for k, by_dom in present.items():
        if len(by_dom) == len(mono) and len(set(by_dom.values())) == 1:
            layer[k] = "L1"                      # 전 도메인 동일 값 = 그대로 복사
        elif len(by_dom) >= 2:
            layer[k] = "L2"                      # 구조 공유·값만 다름
        else:
            layer[k] = "L3"                      # 그 도메인 전용
    return layer, present


def split_domain(d, layer):
    """단일 도메인 dict를 (L1키, settings, specific)으로. `_note_<key>`는 제 키를 따라간다."""
    out = {"L1": {}, "L2": {}, "L3": {}}
    for k, v in d.items():
        if k.startswith("_"):
            continue
        out[layer.get(k, "L3")][k] = v
    for k, v in d.items():                        # 주석 키 배치
        if not k.startswith("_"):
            continue
        if k == "_meta":
            out["L2"][k] = v
            continue
        owner = k[len("_note_"):] if k.startswith("_note_") else None
        out[layer.get(owner, "L3") if owner in layer else "L3"][k] = v
    return out


def _engine_capex():
    """엔진 폐포의 크기 = 한 번만 내는 비용. 스코프는 x6h의 import 폐포와 **같은 것**을 쓴다
    (손 목록은 스코프 누락을 낳는다·x6h 리뷰 B1)."""
    import ast as _ast
    try:
        from x6h_engine_literal_audit import discover_engine_files
        files = discover_engine_files()
    except Exception:
        files = []
    loc = code = fns = 0
    for f in files:
        try:
            src = io.open(os.path.join(_HERE, f), encoding="utf-8", errors="replace").read()
        except Exception:
            continue
        lines = src.splitlines()
        loc += len(lines)
        code += sum(1 for l in lines if l.strip() and not l.strip().startswith("#"))
        try:
            fns += sum(1 for n in _ast.walk(_ast.parse(src))
                       if isinstance(n, (_ast.FunctionDef, _ast.AsyncFunctionDef)))
        except Exception:
            pass
    return {"files": len(files), "loc": loc, "code": code, "fns": fns}


def _a2_read_sites(layer):
    """엔진 소스가 A2 키를 **이름으로 직독**하는 지점 수(층별). L3 수치가 곧 callback 리팩터 범위."""
    import re as _re
    try:
        from x6h_engine_literal_audit import discover_engine_files
        files = discover_engine_files()
    except Exception:
        files = []
    src = ""
    for f in files:
        try:
            src += io.open(os.path.join(_HERE, f), encoding="utf-8", errors="replace").read()
        except Exception:
            pass
    out = {"L1": 0, "L2": 0, "L3": 0}
    for k, L in layer.items():
        out[L] = out.get(L, 0) + len(_re.findall(r'["\']%s["\']' % _re.escape(k), src))
    return out


def merged(base, settings, specific):
    m = dict(real(base))
    m.update(settings)
    m.update(specific)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emit", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--sync-mono", action="store_true",
                    help="분리 파일(정본) → <domain>.gate.json(레거시 생성물) 재생성. "
                         "키 순서는 기존 단일파일을 따른다(잡음 diff 방지).")
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    mono = {}
    for dom in DOMAINS:
        p = os.path.join(_A2, dom + ".gate.json")
        if os.path.exists(p):
            mono[dom] = load(p)
    base_path = os.path.join(_A2, "base", "shared.json")
    base = load(base_path) if os.path.exists(base_path) else {}
    # base에 이미 올라간 키도 L1으로 취급(단일파일엔 없다)
    layer, present = classify(mono)
    for k in real(base):
        layer[k] = "L1"

    if args.sync_mono:
        # 정본은 분리 파일이다(설계 §3). 레거시 read-site 105곳이 아직 단일파일을 읽으므로
        # 여기서 **생성물로** 다시 쓴다. 새 키는 뒤에 붙이고 기존 키 순서는 보존한다.
        for dom in DOMAINS:
            st_p = os.path.join(_A2, dom + ".settings.json")
            sp_p = os.path.join(_A2, dom + ".specific.json")
            if not (os.path.exists(st_p) and os.path.exists(sp_p)):
                continue
            parts = {}
            for p in (st_p, sp_p):
                parts.update({k: v for k, v in load(p).items() if k != "_note_layer"})
            old = mono.get(dom, {})
            out = {k: parts[k] for k in old if k in parts}
            out.update({k: v for k, v in parts.items() if k not in out})
            dump(os.path.join(_A2, dom + ".gate.json"), out)
            print("  %-18s gate.json 재생성 %d키 (분리 파일 = 정본)" % (dom, len(out)))
        return

    if args.verify:
        bad = 0
        for dom in DOMAINS:
            s = os.path.join(_A2, dom + ".settings.json")
            sp = os.path.join(_A2, dom + ".specific.json")
            if not (os.path.exists(s) and os.path.exists(sp)):
                print("  ⚠%s 분리 파일 없음 — --emit 먼저" % dom)
                bad += 1
                continue
            m = merged(base, real(load(s)), real(load(sp)))
            want = real(mono.get(dom, {}))
            ok = canon(m) == canon({**real(base), **want})
            print("  %-18s 병합 == 단일파일+base : %s" % (dom, "✅" if ok else "❌"))
            bad += (not ok)
        sys.exit(1 if bad else 0)

    print("=" * 78)
    print("① 층 분류 — L1 공통 / L2 구조공유·값만 / L3 그 도메인 전용")
    print("=" * 78)
    rows = []
    for dom in DOMAINS:
        d = real(mono.get(dom, {}))
        cnt = {"L1": 0, "L2": 0, "L3": 0}
        byt = {"L1": 0, "L2": 0, "L3": 0}
        for k, v in d.items():
            L = layer.get(k, "L3")
            cnt[L] += 1
            byt[L] += len(canon(v))
        rows.append((dom, cnt, byt))
        print("  %-18s L1 %2d키 · L2 %2d키 %6dB · L3 %2d키 %7dB"
              % (dom, cnt["L1"] + len(real(base)), cnt["L2"], byt["L2"], cnt["L3"], byt["L3"]))
    print("\n  L1(base):", sorted(real(base)))
    print("  L2      :", sorted(k for k, L in layer.items() if L == "L2"))

    print("\n" + "=" * 78)
    print("② ★새 도메인 1개 추가 비용 (이 분리가 답하려는 질문)")
    print("=" * 78)
    print("  L1 = 0 (그대로 복사) · L2 = 같은 구조에 값 채우기 · L3 = 저작 + 엔진 접속")
    for dom, cnt, byt in rows:
        print("    %-18s 채울 L2 %d키 · 새로 쓴 L3 %d키(%dB)"
              % (dom, cnt["L2"], cnt["L3"], byt["L3"]))
    zero = [d for d, c, _ in rows if c["L3"] == 0]
    if zero:
        print("  ⇒ **%s 는 L3가 0** = L2만으로 도메인이 선다(최소 비용 실측 사례)." % ", ".join(zero))

    # ── §③ capex vs opex 회계 (논문·특허용·사용자 지시 2026-07-31) ──────────────
    print("\n" + "=" * 78)
    print("③ ★capex(한 번 짓는다) vs opex(도메인마다 낸다)")
    print("=" * 78)
    cap = _engine_capex()
    print("  **capex** — 엔진 import 폐포(라이브 드라이버 `t2_run_gated.py` 기준·x6h와 동일 스코프)")
    print("     파일 %d · 전체 %,d줄 · 코드 %,d줄 · 함수 %d개"
          .replace(",d", "d") % (cap["files"], cap["loc"], cap["code"], cap["fns"]))
    print("     이 비용은 **도메인 수와 무관**하다 — 새 도메인이 늘어도 다시 짓지 않는다.")
    reads = _a2_read_sites(layer)
    print("\n  엔진이 A2를 읽는 지점 %d곳 (층별):" % sum(reads.values()))
    for L in ("L1", "L2", "L3"):
        print("     %-4s %3d곳%s" % (L, reads.get(L, 0),
              "   ← 이만큼이 도메인-특화 분기 = callback 계약 리팩터 범위(설계 §4)"
              if L == "L3" else ""))

    print("\n  **opex** — 도메인마다 내는 것")
    print("     %-18s %8s %10s %8s %10s %10s"
          % ("도메인", "L2키", "L2바이트", "L3키", "L3바이트", "L3줄"))
    for dom, cnt, byt in rows:
        d = real(mono.get(dom, {}))
        l3_lines = sum(json.dumps(v, ensure_ascii=False, indent=1).count("\n") + 1
                       for k, v in d.items() if layer.get(k) == "L3")
        print("     %-18s %8d %10d %8d %10d %10d"
              % (dom, cnt["L2"], byt["L2"], cnt["L3"], byt["L3"], l3_lines))

    print("\n  ★**새 도메인 1개 추가 비용** (한 줄):")
    print("     엔진 0줄 · L1 0(그대로 복사) · **L2 %d키 채우기** · L3는 넣고 싶은 만큼"
          % max(c["L2"] for _, c, _ in rows))
    zero = [(d, c) for d, c, _ in rows if c["L3"] == 0]
    if zero:
        print("     실측 하한 = **%s**: L2 %d키만으로 도메인이 섰다(L3 0키·엔진 수정 0)."
              % (zero[0][0], zero[0][1]["L2"]))
    print("     ⚠단 지금은 L3를 넣으려면 **엔진 분기도 함께** 늘어난다(위 L3 %d곳). callback 계약"
          % reads.get("L3", 0))
    print("       전까지는 'opex만 증가'가 아니라 **capex도 증가**한다 — 정직한 회계는 이쪽이다.")

    if args.emit:
        print("\n" + "=" * 78)
        print("③ 기록 + 등가 검증")
        print("=" * 78)
        for dom in DOMAINS:
            if dom not in mono:
                continue
            parts = split_domain(mono[dom], layer)
            sp_path = os.path.join(_A2, dom + ".specific.json")
            st_path = os.path.join(_A2, dom + ".settings.json")
            st = dict(parts["L2"])
            st.setdefault("_note_layer", "L2 settings — 구조는 3층 공통, 값만 이 도메인. "
                                   "정본=A2_THREE_LAYER_SPLIT_DESIGN_2026_07_31.md")
            sp = dict(parts["L3"])
            sp.setdefault("_note_layer", "L3 specific — 이 도메인에만 있는 도구·규칙. 엔진 접속은 "
                                   "현재 하드와이어(44사이트)이고 callback 계약은 설계 §4(미구현).")
            dump(st_path, st)
            dump(sp_path, sp)
            if parts["L1"]:
                print("  ⚠%s: 단일파일에 L1 키가 남아 있다 → base로 올려야 한다: %s"
                      % (dom, sorted(parts["L1"])))
            m = merged(base, real(st), real(sp))
            ok = canon(m) == canon({**real(base), **real(mono[dom])})
            print("  %-18s settings %2d키 · specific %2d키 → 병합 등가 %s"
                  % (dom, len(real(st)), len(real(sp)), "✅" if ok else "❌ 불일치"))


if __name__ == "__main__":
    main()
