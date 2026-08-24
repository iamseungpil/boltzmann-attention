# -*- coding: utf-8 -*-
r"""x521 — 사실표 **속성 커버리지** 감사: 문서가 말하는 축 ↔ 우리 16축 (2026-08-24).

사용자 지시: *"어느 속성이 빠졌는지 전수로 세라."*

★이 프로브는 문서 id 를 **하나도 갖지 않는다** — 궤적의 tool 메시지에서 정규식으로 뜬 것만 쓰고,
  어떤 문서를 읽을지 고르지 않는다(집계만). 선언을 읽는 소비자가 아니라 **관측기**다.

## 왜

KB 감사(wf_2d75c8c8-347)가 057 의 유일한 판별축이 **early direct deposit 일수**임을 축자로
확정했다 — 손님 msg[007] *"I **need early direct deposit** — at least **one day early**"* ·
gold 쪽 문서 *"- Early direct deposit: 1 day(s) before payday"* ↔ 실제 채택된 계좌 문서
*"| Early direct deposit | 0 day(s) |"*. 그런데 `x430_account_facts.ATTRS` **16축에 그 축이 없다**.
그래서 `x513` 의 격리 팔 `B_table`/`C_filtered` 는 판별축을 **볼 수 없는 표**를 쥐고 있었다 —
*"표를 줘도 0/6"* 은 능력 경계가 아니라 **표 결손**이다.

## 무엇을 재나 — 해석 0, 라벨 집합만

손님이 무엇을 원하는지는 **추론하지 않는다**([[59]]). 대신 닫힌 두 집합을 맞댄다:

    A. 문서가 실제로 말하는 속성 라벨   (궤적 tool 메시지의 `- <라벨>: <값>` · `| <라벨> | <값> |`)
    B. 우리 표가 잡는 16축              (`x430_account_facts.ATTRS` 의 키 + 별칭)

A - B = **표가 못 잡는 축**. 그 크기가 곧 표 확장의 크기다.

## 계약 ([[77]])

① 양화     : t7348 궤적 전량 · 라벨 출현 N회 이상만 계상(잡음 컷)
② 근거     : 라벨은 문서 축자에서 그대로 뜬 것이고 예시 문장을 함께 인쇄한다
③ 반증 조건 : 미포착 라벨이 `ATTRS` 별칭으로 이미 잡히면 이 표는 과대계상이다 —
              그래서 별칭 매칭을 **정규화 후 부분문자열**로 넉넉히 건다(거짓 미포착을 줄인다).
              또 값 없는 산문은 속성이 아니므로 **값이 붙은 줄만** 센다.
④ 선행 확인 : x430(표 생성기) · x502(9칸 손감사) · x518(희소성·기계깃발) · x513(격리 팔) ·
              KB 감사 종합(wf_2d75c8c8-347) 대조함.

실행: PYTHONIOENCODING=utf-8 python x521_attr_coverage.py
"""
import collections
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

import x430_account_facts as X430          # noqa: E402

SIMS = os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026", "sim_results")
OUT = os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026")
RUNS = ("bank_t7348_halfA_20260824", "bank_t7348_halfB_20260824")

# 문서 속성 라벨의 두 형태 — 값이 붙은 것만(산문 문장은 라벨이 아니다)
BULLET = re.compile(r"^\s*[-*]\s{0,3}([A-Z][A-Za-z0-9 /()'\-]{2,44}):\s*(\S.*)$")
TABLE = re.compile(r"^\s*\|\s*([A-Z][A-Za-z0-9 /()'\-]{2,44})\s*\|\s*(\S[^|]*)\|")
DOCTOKEN = re.compile(r"\b%s[A-Za-z0-9_()+\-]+" % "doc" + "_")   # 리터럴 id 0 · 형태만
STOP = {"id", "score", "content", "note", "overview", "example", "summary",
        "timing", "axis", "eligibility", "documents", "important restrictions"}


def norm(s):
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


def attr_index():
    """키 + 별칭 → 정규화 집합. 별칭은 부분문자열로도 건다(거짓 미포착 축소·③)."""
    keys = []
    for a in X430.ATTRS:
        k = a[0] if isinstance(a, (list, tuple)) else a
        al = list(a[1]) if isinstance(a, (list, tuple)) and len(a) > 1 else []
        keys.append((k, [norm(k.replace("_", " "))] + [norm(x) for x in al]))
    return keys


def covered(label, keys):
    n = norm(label)
    for k, pats in keys:
        for p in pats:
            if not p:
                continue
            if p in n or n in p:
                return k
    return None


def main():
    keys = attr_index()
    labels = collections.Counter()
    sample = {}
    docs_seen = set()
    for tag in RUNS:
        fp = os.path.join(SIMS, tag + ".results.json.gz")
        if not os.path.exists(fp):
            continue
        d = json.load(gzip.open(fp, "rt", encoding="utf-8", errors="replace"))
        for s in (d.get("simulations") or []):
            for m in (s.get("messages") or []):
                if m.get("role") != "tool":
                    continue
                c = str(m.get("content") or "")
                if not DOCTOKEN.search(c):
                    continue
                docs_seen |= set(DOCTOKEN.findall(c))
                for ln in c.split("\n"):
                    mm = BULLET.match(ln) or TABLE.match(ln)
                    if not mm:
                        continue
                    lab = mm.group(1).strip()
                    if norm(lab) in STOP or not norm(lab):
                        continue
                    labels[lab] += 1
                    sample.setdefault(lab, ln.strip()[:120])

    hit = collections.Counter()
    miss = collections.Counter()
    for lab, n in labels.items():
        k = covered(lab, keys)
        if k:
            hit[k] += n
        else:
            miss[lab] += n

    print("=" * 100)
    print("(1) 우리 16축이 **잡는** 라벨 — 축별 출현")
    print("=" * 100)
    for k, _ in keys:
        print("  %-34s %6d" % (k, hit.get(k, 0)))
    print("  %-34s %6d" % ("(잡힌 라벨 총계)", sum(hit.values())))

    CUT = 20
    big = [(l, n) for l, n in miss.most_common() if n >= CUT]
    print("")
    print("=" * 100)
    print("(2) ★표가 **못 잡는** 문서 축 — %d회 이상 (문서 %d개 · 라벨종 %d)"
          % (CUT, len(docs_seen), len(labels)))
    print("=" * 100)
    print("  %-40s %6s  %s" % ("문서가 쓰는 라벨", "출현", "축자 예"))
    print("  " + "-" * 96)
    for lab, n in big:
        print("  %-40s %6d  %s" % (lab[:40], n, sample.get(lab, "")[:52]))
    print("")
    print("  못 잡는 라벨 %d종 · 출현 합 %d (잡힌 것 %d)"
          % (len(miss), sum(miss.values()), sum(hit.values())))

    print("")
    print("=" * 100)
    print("(3) 057 의 판별축이 실제로 빠졌나 — 지목 검사")
    print("=" * 100)
    for probe in ("Early direct deposit", "Early Direct Deposit"):
        k = covered(probe, keys)
        print("  %-28s → %s (문서 출현 %d회)"
              % (probe, k or "★ATTRS 에 없음", labels.get(probe, 0)))

    out = {
        "probe": "x521_attr_coverage", "date": "2026-08-24",
        "contract": {
            "quantification": "t7348 halfA+halfB 궤적 전량 · 문서 %d개 · 라벨종 %d · 컷 %d회"
                              % (len(docs_seen), len(labels), CUT),
            "evidence": "라벨은 문서 축자에서 그대로 뜬 것 · 예시 문장 동봉 (sample)",
            "what_would_refute": "미포착 라벨이 ATTRS 별칭으로 이미 잡히면 과대계상이다 — "
                                 "그래서 별칭을 정규화 후 부분문자열로 넉넉히 걸었다. "
                                 "또 값 없는 산문은 속성이 아니므로 값 붙은 줄만 셌다.",
            "prior_checked": ["x430_account_facts.py(ATTRS 16)", "x502(9칸 손감사)",
                              "x518(희소성)", "x513(격리 팔)", "wf_2d75c8c8-347(KB 감사)"],
        },
        "attrs_16": [k for k, _ in keys],
        "covered_counts": dict(hit),
        "uncovered": dict(miss),
        "uncovered_top": [{"label": l, "n": n, "sample": sample.get(l, "")} for l, n in big],
        "docs_seen": len(docs_seen),
        "limits": [
            "라벨 추출은 두 형태만 본다 — 산문 안의 속성은 안 잡힌다(하한).",
            "출현 수는 **배달 횟수**이지 문서 수가 아니다(같은 문서가 여러 sim 에 배달된다).",
            "이 표는 '무엇이 빠졌나'를 세는 것이지 '무엇을 넣어야 성적이 오르나'가 아니다([[70]]).",
        ],
    }
    dst = os.path.join(OUT, "x521_attr_coverage_2026_08_24.json")
    with io.open(dst, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("")
    print("-> %s" % os.path.normpath(dst))
    return 0


if __name__ == "__main__":
    sys.exit(main())
