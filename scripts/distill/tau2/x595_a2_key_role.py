# -*- coding: utf-8 -*-
"""X595 — A2 최상위 키별 **엔진 사용법** 증거표 (2026-08-29·무료·읽기 전용).

왜 이것인가 (사용자 지시 2026-08-29):
    A1 = 도메인 특성이 필요없는 **일반화된 규칙**
    A2 = 도메인 특성을 따르나 **엔진에서는 공동으로 사용**하고 도메인별로 **특성값만** 바꾸는 항목
    A3 = 도메인별 특화된 정보로 **공통된 규칙이 없는** 내용
  그리고 *"airline, retail, banking 을 모두 같은 기준으로 재배치하라."*

⛔**옛 판정 기준(`x18`)은 키의 도메인 분포였다** — "2+ 도메인에 있으면 L2". 지금은 banking 만
  개발돼 있어서(airline L3 0키) 분포로 나누면 *"뱅킹이 원래 크다"* 와 *"뱅킹만 투자했다"* 가
  교락된다. 새 기준은 **엔진이 그 값을 어떻게 쓰느냐**이고, 그것을 판정하려면 세는 것이 아니라
  **엔진 소스의 그 줄**을 봐야 한다. 이 프로브는 그 줄을 모은다 — 판정하지 않는다([[62]]).

단위는 **최상위 키**다. 근거: `gate_interpreter.load_domain_a2` 가 L1→L2→L3 를 최상위 키로
병합하고, 엔진의 접촉면이 `a2.get("<키>")` 직독이다(설계 §4·읽기 지점 266곳). 중첩 필드명
(`op`·`a`·`b` …)은 한 최상위 키의 **값 내부 구조**이지 배치 단위가 아니다.

용법:
  py -3 x595_a2_key_role.py                 # 요약표
  py -3 x595_a2_key_role.py --key gates     # 한 키의 증거 전문
  py -3 x595_a2_key_role.py --emit out.json # 증거표 JSON
"""
import argparse
import io
import json
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_A2 = os.path.join(_HERE, "a2")
DOMAINS = ["banking_knowledge", "retail", "airline"]


def _read(path):
    if not os.path.exists(path):
        return None
    with io.open(path, encoding="utf-8") as f:
        return json.load(f)


def notes(dom):
    """`_note_<키>` 주석 = 그 키가 무엇에서 왔는지에 대한 **이미 있는 기록**([[23]] 의무).

    ★왜 증거인가: 2026-07-31 출처 감사가 키마다 *"구조 선언(= 이 도메인에 무엇이 있는지를 적은
    것) · 출처 = env 레지스트리 + KB 문서"* 인지, 아니면 **정책 축자**를 인용해 쓴 규칙인지를
    적어 뒀다. 앞쪽은 새 도메인에서 **레지스트리를 읽어 값을 세팅**하면 되고, 뒤쪽은 **문장을
    써야** 한다 — 사용자가 2026-08-29 에 지정한 바로 그 판정선이다.
    """
    out = {}
    for fn in (dom + ".settings.json", dom + ".specific.json"):
        d = _read(os.path.join(_A2, fn)) or {}
        for k, v in d.items():
            if k.startswith("_note_") and k != "_note_layer":
                out[k[len("_note_"):]] = str(v)
    return out


def domain_layers(dom):
    """도메인 하나의 (L2, L3, mono) 를 파일 그대로 돌려준다. 병합하지 않는다."""
    return (
        _read(os.path.join(_A2, dom + ".settings.json")) or {},
        _read(os.path.join(_A2, dom + ".specific.json")) or {},
        _read(os.path.join(_A2, dom + ".gate.json")) or {},
    )


def engine_files():
    """엔진 폐포 = 라이브 드라이버의 import 폐포. 손 목록은 스코프 누락을 낳는다(x6h 리뷰 B1)."""
    try:
        from x6h_engine_literal_audit import discover_engine_files
        return discover_engine_files()
    except Exception as e:
        sys.stderr.write("[x595] discover_engine_files 실패: %r\n" % (e,))
        return []


def read_sites(keys):
    """키 -> [(파일, 줄번호, 그 줄 원문)]. 엔진이 **이름으로 직독**하는 지점.

    ⚠세는 것이 목적이 아니다. 판정자가 그 줄을 읽어야 하므로 원문을 함께 낸다.
    """
    pats = {k: re.compile("[\"']" + re.escape(k) + "[\"']") for k in keys}
    sites = {k: [] for k in keys}
    for f in engine_files():
        p = os.path.join(_HERE, f)
        try:
            src = io.open(p, encoding="utf-8", errors="replace").read().splitlines()
        except Exception:
            continue
        for i, line in enumerate(src, 1):
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            for k, pat in pats.items():
                if pat.search(line):
                    sites[k].append((os.path.basename(f), i, s[:200]))
    return sites


def nested_names(v, out=None, depth=0):
    """값 안의 **중첩 필드명** → 등장 횟수.

    ★왜 횟수인가(2026-08-29 자기교정): 처음엔 이름의 **집합**으로 비율을 냈는데,
    값이 *도메인 낱말을 키로 삼는 맵*이면(`{"open_account": {...}, "file_dispute": {...}}`)
    그 낱말들이 전부 '필드'로 세어져 비율이 눌린다 — `scaffold_get_tools` 가 387필드·0.333 로
    나온 이유다. **스키마 슬롯은 형제 항목마다 되풀이되고 내용 키는 한 번 나온다.** 그래서
    반복 이름만 따로 본다.
    """
    if out is None:
        out = {}
    if depth > 8:
        return out
    if isinstance(v, dict):
        for k, vv in v.items():
            if isinstance(k, str) and not k.startswith("_"):
                out[k] = out.get(k, 0) + 1
            nested_names(vv, out, depth + 1)
    elif isinstance(v, list):
        for vv in v[:400]:
            nested_names(vv, out, depth + 1)
    return out


_IDENT = re.compile(r"^[a-z][a-z0-9_]*$")


def leaves(v, out=None, depth=0):
    """값의 **잎 문자열**과 수/불리언을 모은다.

    ★왜 잎인가 (2026-08-29): 엔진에는 도메인 낱말이 0이라(모든 읽기가 `a2.get(K)`) *엔진 코드*로는
    A2 와 A3 가 안 갈린다. 사용자 정의가 가르는 지점은 **새 도메인이 이 칸을 채울 때 무엇을 하느냐**다
    — A2 는 *"특성값만 바꾸는"* 것이고 A3 는 *"공통된 규칙이 없는 내용"*을 쓰는 것이다.
    그 차이는 잎에 있다: 잎이 env 표면의 **식별자**(도구·인자·필드 이름)면 채우기이고,
    잎이 **산문**이면 저작이다.
    """
    if out is None:
        out = {"ident": 0, "prose": 0, "short": 0, "num": 0, "prose_bytes": 0}
    if depth > 10:
        return out
    if isinstance(v, dict):
        for k, vv in v.items():
            if not (isinstance(k, str) and k.startswith("_")):
                leaves(vv, out, depth + 1)
    elif isinstance(v, list):
        for vv in v:
            leaves(vv, out, depth + 1)
    elif isinstance(v, bool) or isinstance(v, (int, float)):
        out["num"] += 1
    elif isinstance(v, str):
        s = v.strip()
        if len(s) >= 60 or s.count(" ") >= 7:
            out["prose"] += 1
            out["prose_bytes"] += len(s.encode("utf-8"))
        elif _IDENT.match(s):
            out["ident"] += 1
        else:
            out["short"] += 1
    return out


def file_literals(files):
    """파일별 따옴표 문자열 리터럴 집합. 엔진이 **이름으로 아는** 것의 전부."""
    pat = re.compile("[\"']([A-Za-z_][A-Za-z0-9_]{2,})[\"']")
    out = {}
    for f in files:
        p = os.path.join(_HERE, f)
        try:
            src = io.open(p, encoding="utf-8", errors="replace").read()
        except Exception:
            src = ""
        out[os.path.basename(f)] = set(pat.findall(src))
    return out


def shape(v):
    """값의 형상 — 판정의 두 번째 근거. 도메인 낱말이 값에 있느냐를 사람이 보게 만든다."""
    if isinstance(v, dict):
        return {"t": "dict", "n": len(v), "sample": list(v.keys())[:6]}
    if isinstance(v, list):
        head = v[0] if v else None
        return {"t": "list", "n": len(v),
                "elem": type(head).__name__,
                "sample": (list(head.keys())[:6] if isinstance(head, dict)
                           else json.dumps(v[:3], ensure_ascii=False)[:200])}
    if isinstance(v, bool):
        return {"t": "bool", "v": v}
    if isinstance(v, (int, float)):
        return {"t": "num", "v": v}
    return {"t": "str", "n": len(str(v)), "sample": str(v)[:200]}


def nbytes(v):
    return len(json.dumps(v, ensure_ascii=False).encode("utf-8"))


def build():
    per_dom = {d: domain_layers(d) for d in DOMAINS}
    base = _read(os.path.join(_A2, "base", "shared.json")) or {}

    rows = {}
    for k in base:
        if k.startswith("_"):
            continue
        rows.setdefault(k, {"key": k, "in": {}, "file_layer": {}})
        rows[k]["in"]["_shared"] = shape(base[k])
        rows[k]["in"]["_shared"]["bytes"] = nbytes(base[k])   # L1 도 회계에 들어가야 한다
        rows[k]["file_layer"]["_shared"] = "L1"

    for dom, (l2, l3, mono) in per_dom.items():
        for lay, d in (("L2", l2), ("L3", l3)):
            for k, v in d.items():
                if k.startswith("_"):
                    continue
                r = rows.setdefault(k, {"key": k, "in": {}, "file_layer": {}})
                r["in"][dom] = shape(v)
                r["in"][dom]["bytes"] = nbytes(v)
                r["file_layer"][dom] = lay

    sites = read_sites(list(rows))
    lits = file_literals(engine_files())
    allnotes = dict((d, notes(d)) for d in DOMAINS)

    raw = {}
    for dom, (l2, l3, mono) in per_dom.items():
        for d in (l2, l3):
            for k, v in d.items():
                if not k.startswith("_"):
                    raw.setdefault(k, {})[dom] = v
    for k, v in base.items():
        if not k.startswith("_"):
            raw.setdefault(k, {})["_shared"] = v

    for k, r in rows.items():
        r["sites"] = sites.get(k, [])
        r["n_sites"] = len(r["sites"])
        r["site_files"] = sorted(set(f for f, _, _ in r["sites"]))
        r["domains"] = sorted(d for d in r["in"] if d != "_shared")
        r["n_domains"] = len(r["domains"])
        r["bytes_total"] = sum(v.get("bytes", 0) for v in r["in"].values())

        # ★스키마 해석도 — 엔진이 그 값의 **중첩 필드명**을 아는 비율.
        # 공지: 이름 대조는 충돌한다. 그래서 **그 키를 읽는 파일들**로만 대조 범위를 좁히고
        # 3자 미만 이름은 버린다(`op`·`a`·`b` 가 온 세상과 충돌한다).
        cnt = {}
        for v in (raw.get(k) or {}).values():
            nested_names(v, cnt)
        cnt = dict((n, c) for n, c in cnt.items() if len(n) >= 3)
        known = set()
        for f in r["site_files"]:
            known |= (lits.get(f) or set())

        names = set(cnt)
        hit = sorted(names & known)
        r["n_fields"] = len(names)
        r["n_fields_known"] = len(hit)
        r["fields_unknown"] = sorted(names - known, key=lambda n: -cnt[n])[:12]
        r["schema_ratio"] = (round(len(hit) / float(len(names)), 3) if names else None)

        # 반복 이름 = 스키마 슬롯 후보(형제마다 되풀이). 한 번뿐인 이름은 내용 키다.
        rep = set(n for n, c in cnt.items() if c >= 3)
        rhit = sorted(rep & known)
        r["n_slots"] = len(rep)
        r["n_slots_known"] = len(rhit)
        r["slots_unknown"] = sorted(rep - known, key=lambda n: -cnt[n])[:12]
        r["slot_ratio"] = (round(len(rhit) / float(len(rep)), 3) if rep else None)
        r["n_content_only"] = len(names) - len(rep)

        lv = {"ident": 0, "prose": 0, "short": 0, "num": 0, "prose_bytes": 0}
        for v in (raw.get(k) or {}).values():
            leaves(v, lv)
        r["leaves"] = lv
        tot = lv["ident"] + lv["prose"] + lv["short"] + lv["num"]
        r["n_leaves"] = tot
        r["prose_share"] = (round(lv["prose"] / float(tot), 3) if tot else None)
        r["prose_bytes"] = lv["prose_bytes"]

        nt = ""
        for d in DOMAINS:
            nt += (allnotes.get(d, {}).get(k) or "")
        r["note_struct"] = ("구조 선언" in nt)
        r["note_len"] = len(nt)

        # ★사전-판정(사용자 기준 2026-08-29): *"airline/retail 로 갈 때 새 항목을 저작하지 않고
        #   값 세팅만 할 수 있냐"*. 기계가 답할 수 있는 만큼만 답하고 나머지는 REVIEW 로 남긴다
        #   — 여기서 전부 정하는 척하면 그게 [[62]] 위반이다.
        if r["n_sites"] == 0:
            r["pre"] = "DEAD"                       # 엔진이 안 읽는다
        elif r["file_layer"].get("_shared") == "L1" or (
                r["n_domains"] == len(DOMAINS)
                and len(set(json.dumps(v, sort_keys=True, ensure_ascii=False)
                            for v in (raw.get(k) or {}).values())) == 1):
            r["pre"] = "A1"                         # 값이 도메인과 무관 = 엔진 기본값으로
        elif r["n_domains"] >= 2:
            r["pre"] = "A2"                         # ★실증: 다른 도메인이 이미 채웠다
        elif r["prose_share"] in (0, 0.0, None) or r["note_struct"]:
            r["pre"] = "A2"                         # 잎이 전부 식별자 / 주석이 '구조 선언'
        elif r["prose_share"] >= 0.5:
            r["pre"] = "A3"                         # 잎의 과반이 산문 = 저작
        else:
            r["pre"] = "REVIEW"
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--key")
    ap.add_argument("--emit")
    a = ap.parse_args()
    rows = build()

    if a.emit:
        with io.open(a.emit, "w", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(rows, ensure_ascii=False, indent=1))
        print("[x595] wrote %s (%d keys)" % (a.emit, len(rows)))
        return

    if a.key:
        r = rows.get(a.key)
        if not r:
            print("no such key: %s" % a.key)
            return
        print(json.dumps({kk: vv for kk, vv in r.items() if kk != "sites"},
                         ensure_ascii=False, indent=1))
        print("--- 엔진 읽기 지점 %d ---" % r["n_sites"])
        for f, i, s in r["sites"]:
            print("  %s:%d  %s" % (f, i, s))
        return

    print("키 %d개 · 엔진 폐포 %d파일" % (len(rows), len(engine_files())))
    print("%-30s %-7s %3s %4s %8s %6s %6s %7s %5s  %s"
          % ("key", "pre", "dom", "site", "bytes", "ident", "prose", "prose%", "note", "layer"))
    rank = {"A1": 0, "A2": 1, "REVIEW": 2, "A3": 3, "DEAD": 4}
    order = sorted(rows.values(), key=lambda r: (rank[r["pre"]], -r["bytes_total"]))
    for r in order:
        fl = ",".join("%s=%s" % (d[:4], l) for d, l in sorted(r["file_layer"].items()))
        lv = r["leaves"]
        print("%-30s %-7s %3d %4d %8d %6d %6d %7s %5s  %s"
              % (r["key"][:30], r["pre"], r["n_domains"], r["n_sites"], r["bytes_total"],
                 lv["ident"], lv["prose"],
                 ("-" if r["prose_share"] is None else "%.2f" % r["prose_share"]),
                 ("STRC" if r["note_struct"] else ""), fl))
    agg = {}
    for r in rows.values():
        a2 = agg.setdefault(r["pre"], [0, 0, 0])
        a2[0] += 1
        a2[1] += r["bytes_total"]
        a2[2] += r["prose_bytes"]
    print()
    print("%-7s %4s %10s %12s" % ("pre", "keys", "bytes", "prose_bytes"))
    for c in ("A1", "A2", "REVIEW", "A3", "DEAD"):
        v = agg.get(c) or [0, 0, 0]
        print("%-7s %4d %10d %12d" % (c, v[0], v[1], v[2]))
    print()
    print("읽기 0곳 = %d키 (死선언 후보 — 이름이 아니라 소비를 봐야 한다)"
          % sum(1 for r in rows.values() if r["n_sites"] == 0))
    print("ratio=0 & 읽힘 = %d키 (엔진이 스키마를 모른다 = 통과만 시킨다)"
          % sum(1 for r in rows.values() if r["n_sites"] and r["schema_ratio"] == 0))


if __name__ == "__main__":
    main()
