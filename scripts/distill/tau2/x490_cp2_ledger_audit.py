# -*- coding: utf-8 -*-
r"""x490 — **CP2 배달 원장 검산** (2026-08-23·무료·오프라인·LLM 0 · R4 의 짝).

## 무엇을 내는가 — 팔-대칭 종점 하나
사이드카(`fb_<tag>.jsonl[.gz]`)의 `agent=cp2` 행을 읽어 배달물마다 생애를 닫는다:
```
assign  →  attached | clobbered | ctx_skip  |  (미종결 = 잔존)
```
그리고 **양 팔에서 같은 규칙으로** sim 당 ⓐ대입 ⓑ도달(attached) ⓒ자수 합 ⓓ도달률을 낸다.

⛔**이 스크립트는 `T2_CP2_QUEUE` 를 읽지 않는다.** C502 가 무효가 된 이유가 바로 그것이었다 —
   1차 종점이 플래그 켠 팔에만 존재할 수 있는 로그 줄이라 *"0/8 → 8/8"* 이 처치 배정의 재인쇄가
   됐다(원장 축자). 팔 이름은 태그로만 들어오고, 계산은 두 팔에서 문자 그대로 동일하다.

## 검산식 (이게 서야 나머지 수치를 믿는다)
    대입 = attached + clobbered + ctx_skip + 잔존          … 닫힌 분할
    ⚠`open` 이 예외로 죽은 건은 **분모에서도 사라진다**(행동 검정 [9b] 가 박아둔 한계) —
      그래서 로그의 `[T2_CP2_TRACK] open 실패` 줄을 **함께 세어** 보고한다. 그 수가 0 이 아니면
      검산식이 성립해도 분모가 이미 줄어든 상태다([[25]] 계기가 조용히 틀리면 안 된다).

## `via=asub` 를 따로 센다
`_am_sub` 가 그 회차를 가져가면 `_gen` 이 안 불리고 `work` 는 비커밋 감사 서브콜로만 간다.
그 회차도 `attached` 지만 **행동을 커밋하는 생성기에는 못 갔다** — 지금 남은 결함의 이름이라
합계와 따로 낸다(판정은 안 한다·[[62]]).

용법:
    py -3 x490_cp2_ledger_audit.py --tags bank_x_ctl,bank_x_treat
    py -3 x490_cp2_ledger_audit.py --tags ... --json x490_out.json
"""
import argparse
import collections
import glob
import gzip
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

REP = os.path.abspath(os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026"))
SIMS = os.path.join(REP, "sim_results")
OUTCOMES = ("attached", "clobbered", "ctx_skip")
RE_OPENFAIL = re.compile(r"\[T2_CP2_TRACK\] open 실패")


def _open_any(p):
    return (gzip.open(p, "rt", encoding="utf-8", errors="replace") if p.endswith(".gz")
            else io.open(p, encoding="utf-8", errors="replace"))


def sidecar_rows(tag):
    """그 태그의 사이드카 `agent=cp2` 행 전부. 보관/리모트 두 이름 형태를 다 받는다."""
    pats = [os.path.join(SIMS, "fb_%s.jsonl.gz" % tag), os.path.join(SIMS, "fb_%s.jsonl" % tag),
            os.path.join(SIMS, "%s.fb.jsonl.gz" % tag),
            "/home/woori/scratch/logs/fb_%s.jsonl" % tag]
    rows, seen = [], []
    for p in pats:
        for f in glob.glob(p):
            seen.append(os.path.basename(f))
            for ln in _open_any(f):
                try:
                    o = json.loads(ln)
                except Exception:
                    continue
                if str(o.get("agent")) == "cp2":
                    rows.append(o)
    return rows, seen


def open_failures(tag):
    """로그에서 `open 실패` 줄 수 — 분모에서 사라진 배달물의 개수."""
    n = 0
    for p in glob.glob(os.path.join(SIMS, "%s*.log.gz" % tag)) + \
            glob.glob("/home/woori/scratch/logs/%s*.log" % tag):
        try:
            for ln in _open_any(p):
                if RE_OPENFAIL.search(ln):
                    n += 1
        except Exception:
            continue
    return n


def audit(tag):
    rows, files = sidecar_rows(tag)
    life = collections.OrderedDict()
    for o in rows:
        cid = o.get("cp2_id")
        if not cid:
            continue
        rec = life.setdefault(cid, {"sim": str(cid).split("#")[0], "tag": o.get("cp2_tag"),
                                    "n": o.get("cp2_n") or 0, "disp": o.get("cp2_disp"),
                                    "outcome": None, "via": None, "turn": o.get("turn")})
        if o.get("ev") == "close":
            rec["outcome"] = o.get("outcome")
            rec["via"] = o.get("cp2_via")
    n_assign = len(life)
    by_out = collections.Counter(v["outcome"] or "OPEN" for v in life.values())
    chars = collections.Counter()
    for v in life.values():
        chars[v["outcome"] or "OPEN"] += int(v["n"] or 0)
    via_asub = sum(1 for v in life.values() if v["via"] == "asub")
    sims = len({v["sim"] for v in life.values()})
    closed = sum(by_out[o] for o in OUTCOMES)
    return {"tag": tag, "files": files, "assign": n_assign, "sims": sims,
            "by_outcome": dict(by_out), "chars": dict(chars), "via_asub": via_asub,
            "open_fail": open_failures(tag),
            "balanced": (closed + by_out["OPEN"]) == n_assign,
            "by_tag": dict(collections.Counter(v["tag"] for v in life.values())),
            "life": life}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", required=True, help="쉼표 구분. 두 팔을 나란히 주면 대조표가 나온다")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    outs = []
    for t in [x.strip() for x in a.tags.split(",") if x.strip()]:
        r = audit(t)
        outs.append(r)
        print("=" * 96)
        print("%s · 사이드카 %s" % (t, r["files"] or "★없음(회수 안 됨 — 러너 배선부터)"))
        if not r["assign"]:
            print("  cp2 행 0 — T2_FB_SIDECAR 가 켜졌는지, 회수가 됐는지 먼저 본다([[55]])")
            continue
        print("  sim %d · 대입 %d · 검산식 %s"
              % (r["sims"], r["assign"], "성립" if r["balanced"] else "★깨짐"))
        for k in OUTCOMES + ("OPEN",):
            n = r["by_outcome"].get(k, 0)
            print("    %-10s %4d건  %8d자" % (k, n, r["chars"].get(k, 0)))
        print("    %-10s %4d건  ← attached 중 커밋 생성기에 못 간 것(감사 서브콜만)"
              % ("via=asub", r["via_asub"]))
        if r["open_fail"]:
            print("    ⚠open 실패 %d건 — 그만큼 **분모에서도 사라졌다**" % r["open_fail"])
        att = r["by_outcome"].get("attached", 0)
        print("  도달률(attached/대입) = %.1f%%  ·  sim 당 대입 %.1f"
              % (100.0 * att / r["assign"], float(r["assign"]) / max(1, r["sims"])))
        print("  배달 자리별 대입: %s" % r["by_tag"])

    if len(outs) >= 2 and all(o["assign"] for o in outs[:2]):
        A, B = outs[0], outs[1]
        print("=" * 96)
        print("팔 대조 (같은 규칙·같은 분모 정의)")
        print("  %-24s %10s %10s" % ("지표", A["tag"][-18:], B["tag"][-18:]))
        for k in ("assign",):
            print("  %-24s %10d %10d" % (k, A[k], B[k]))
        for k in OUTCOMES + ("OPEN",):
            print("  %-24s %10d %10d" % (k, A["by_outcome"].get(k, 0), B["by_outcome"].get(k, 0)))
        print("  %-24s %10d %10d" % ("via=asub", A["via_asub"], B["via_asub"]))
        ra = 100.0 * A["by_outcome"].get("attached", 0) / A["assign"]
        rb = 100.0 * B["by_outcome"].get("attached", 0) / B["assign"]
        print("  %-24s %9.1f%% %9.1f%%   Δ=%+.1f%%p" % ("도달률", ra, rb, rb - ra))
        print("  ⚠이 표는 **배달 도달**만 말한다. 성적은 reward 로만 판정한다([[69]]).")

    if a.json:
        p = a.json if os.path.isabs(a.json) else os.path.join(REP, a.json)
        with io.open(p, "w", encoding="utf-8") as f:
            json.dump([{k: v for k, v in o.items() if k != "life"} for o in outs], f,
                      ensure_ascii=False, indent=1)
        print("\n[JSON] %s" % p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
