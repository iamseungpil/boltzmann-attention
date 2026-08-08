# -*- coding: utf-8 -*-
"""x142 — **축 후보를 코퍼스가 제안하게 한다** (유료 0 · LLM 호출 0 · 로컬 실행).

왜 이런 형태인가(★[[23]] 준수): 축을 **내가 타이핑하면** 그 출처가 어디인지 아무도 못 댄다.
특히 이 세션은 `tasks.json`의 **정답 노트를 읽은 뒤**라(task_100의 *"$175"*, task_101의 *"$400"*),
거기서 축을 고르면 **gold 경유 저작 = 실험 무효**다([[23]]). ⇒ 축 후보는 **정책 문서가 스스로
붙여 놓은 라벨**에서만 뽑는다. 이 도구는 그 라벨을 세는 일만 한다 — 무엇을 채택할지는 판단이고,
그 판단은 **정책 축자 인용을 근거로** 설계서에 남긴다(설계서 §9-2가 요구하는 절차의 형태).

방법(기계적):
  1. 본문에 `referral`이 있는 문서만 본다(추천 정책 문서 = 결정점이 쓰는 것).
  2. **라벨 붙은 수치**만 긁는다 — 문서가 이미 이름을 붙여 둔 사실이다:
       `- **Annual maximum**: 15 referral bonuses per calendar year`  → 라벨 `annual maximum`
       `| Referrer tenure | 30 days |`                                → 라벨 `referrer tenure`
  3. 라벨을 **소문자·공백 정리**만 해서 센다(동의어 병합 금지 — 병합은 판단이다).
  4. 라벨별 **빈도 · 상품 수 · 단위(달러/일/년/건) · 축자 표본**을 인쇄한다.

⚠**분석 도구이지 엔진이 아니다**([[59]]는 엔진을 규율한다). 여기서 나온 문자열은 런타임에 안 간다.
⚠라벨 패턴은 내가 골랐다 — 라벨을 안 붙인 산문 사실은 이 인구조사에 안 잡힌다(하한이다).

★**pass 둘이어야 한다**(2026-08-08 자기정정) — 라벨 pass는 *"Confirm your company is within 4 years
of formation."* 같은 **라벨 없는 산문을 구조적으로 못 본다**. 그것 하나만 믿고 *"코퍼스에 없다"* 고
단정했다가 실물에서 뒤집혔다(`company_max_age_years`). ⇒ `--prose`가 두 번째 제안자다:
**문서의 자기 구조**(자격·요건·제한 섹션 제목) 밑의 수치 문장을 뽑는다. 제목도 문서가 쓴 것이지
내가 고른 어휘가 아니다. **한 스캔의 침묵을 '없음'으로 읽지 않는다.**

usage: x142_axis_candidate_census.py --docs <dir> [--min 3] [--samples 2]
       x142_axis_candidate_census.py --docs <dir> --prose      # 라벨 없는 산문 pass
"""

import argparse
import collections
import glob
import io
import json
import os
import re
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

EMPH = re.compile(r"[*`]")
NUM = re.compile(r"\$?\s*(\d[\d,]*(?:\.\d+)?)\s*(%|percent|days?|years?|months?|hours?)?", re.I)
# 라벨 붙은 수치 두 형태 — 문서가 **스스로** 이름을 붙인 자리만 본다
LINE = re.compile(r"^[-*\d.)\s]*([A-Za-z][A-Za-z /&'’\-]{2,40}?)\s*:\s*(.+)$")
ROW = re.compile(r"^\|\s*([A-Za-z][^|]{2,40}?)\s*\|\s*([^|]+?)\s*\|")


def _norm(s):
    return " ".join(EMPH.sub("", str(s or "")).split())


def unit_of(rest):
    m = NUM.search(rest)
    if not m:
        return None, None
    val, u = m.group(1), (m.group(2) or "").lower()
    if u.startswith("day"):
        unit = "일"
    elif u.startswith("year"):
        unit = "년"
    elif u.startswith("month"):
        unit = "개월"
    elif u in ("%", "percent"):
        unit = "%"
    elif "$" in rest[:max(0, m.start() + 1)] or "$" in rest[:m.start() + 2]:
        unit = "달러"
    else:
        unit = "건/수"
    return val, unit


def family(doc_id):
    return re.sub(r"_\d{3}$", "", str(doc_id or ""))


# ── 산문 pass — 제안자는 **문서의 자기 섹션 제목**이다 ─────────────────────────────
HEAD = re.compile(r"^#{1,6}\s+(.+)$")
HEAD_KEEP = re.compile(r"(eligib|qualif|require|restrict|limit|criteria|threshold|terms)", re.I)
HAS_NUM = re.compile(r"\d")
LABELLED = re.compile(r"^[-*\d.)\s]*[A-Za-z][A-Za-z /&'’\-]{2,40}?\s*:\s*\S|^\|")


def prose_pass(docs_dir, samples):
    """자격·요건 섹션 밑의 **라벨 없는 수치 문장**을 섹션 제목별로 모은다."""
    buckets = collections.defaultdict(list)
    for p in sorted(glob.glob(os.path.join(docs_dir, "*.json"))):
        d = json.load(io.open(p, encoding="utf-8"))
        text = str(d.get("content") or "")
        if "referral" not in text.lower() and "refer " not in text.lower():
            continue
        did = d.get("id") or os.path.basename(p)
        head = ""
        for raw in text.split("\n"):
            line = _norm(raw)
            m = HEAD.match(line)
            if m:
                head = _norm(m.group(1))
                continue
            if not head or not HEAD_KEEP.search(head):
                continue
            body = re.sub(r"^[-*\d.)\s]+", "", line)
            if not body or not HAS_NUM.search(body) or LABELLED.match(line):
                continue
            buckets[head.lower()].append((did, body))

    print("산문 pass — 자격·요건 섹션의 **라벨 없는 수치 문장**")
    print("=" * 96)
    for head, items in sorted(buckets.items(), key=lambda kv: -len(kv[1])):
        fams = {family(d) for d, _ in items}
        print("\n[%s] 문장 %d · 상품계열 %d" % (head[:60], len(items), len(fams)))
        seen = set()
        for did, body in items:
            key = re.sub(r"\d+", "#", body)
            if key in seen:
                continue
            seen.add(key)
            print("   [%s] %s" % (did[-26:], body[:130]))
            if len(seen) >= samples:
                break
    print("\n" + "=" * 96)
    print("⚠제목도 **문서가 쓴 것**이지 내가 고른 어휘가 아니다. 그래도 이 pass 역시 하한이다 —")
    print("  제목 없는 자리의 진술은 여전히 안 보인다. 두 pass의 침묵도 '없음'의 증명은 아니다.")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", required=True)
    ap.add_argument("--min", type=int, default=3, help="이 횟수 미만 라벨은 접는다")
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--prose", action="store_true", help="라벨 없는 산문 pass(제안자 = 섹션 제목)")
    a = ap.parse_args()

    if a.prose:
        return prose_pass(a.docs, max(a.samples, 4))

    labels = collections.defaultdict(lambda: {"n": 0, "fams": set(), "units": collections.Counter(),
                                              "ex": []})
    ndocs = 0
    for p in sorted(glob.glob(os.path.join(a.docs, "*.json"))):
        d = json.load(io.open(p, encoding="utf-8"))
        text = str(d.get("content") or "")
        if "referral" not in text.lower() and "refer " not in text.lower():
            continue
        ndocs += 1
        did = d.get("id") or os.path.basename(p)
        for raw_line in text.split("\n"):
            line = _norm(raw_line)
            if not line:
                continue
            m = ROW.match(line) or LINE.match(line)
            if not m:
                continue
            label, rest = _norm(m.group(1)).lower(), _norm(m.group(2))
            val, unit = unit_of(rest)
            if val is None or not label:
                continue
            e = labels[label]
            e["n"] += 1
            e["fams"].add(family(did))
            e["units"][unit] += 1
            if len(e["ex"]) < a.samples:
                e["ex"].append((did, line[:120]))

    print("추천 정책 문서 %d개 · 라벨 붙은 수치 라벨 %d종" % (ndocs, len(labels)))
    print("=" * 96)
    rows = sorted(labels.items(), key=lambda kv: (-kv[1]["n"], kv[0]))
    shown = 0
    for label, e in rows:
        if e["n"] < a.min:
            continue
        shown += 1
        units = " ".join("%s×%d" % (u, n) for u, n in e["units"].most_common(3))
        print("\n%-34s 등장 %3d · 상품계열 %3d · 단위 %s" % (label[:34], e["n"], len(e["fams"]), units))
        for did, ex in e["ex"]:
            print("      [%s] %s" % (did[-26:], ex))
    print("\n" + "=" * 96)
    print("인쇄한 라벨 %d종(≥%d회) / 전체 %d종" % (shown, a.min, len(labels)))
    print("⚠병합은 하지 않았다 — 동의어를 붙이는 것은 판단이고, 판단은 인용을 근거로 설계서에 남긴다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
