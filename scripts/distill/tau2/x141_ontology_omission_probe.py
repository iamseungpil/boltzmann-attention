# -*- coding: utf-8 -*-
"""x141 — A3 온톨로지 v0의 **누락을 독립 기준으로 잰다** (유료 0 · LLM 호출 0 · 로컬 실행).

왜: v0는 `x140`이 **LLM에게 문서마다 물어** 만든 것이다. 그 산출물을 같은 LLM·같은 프롬프트로
검사하면 **자기 채점**이다 — C315가 실물로 보여준 함정(`split` 2패스 합집합을 오라클로 삼아
`split`이 100%로 보였다). ⇒ **x140과 무관한 방식**으로 후보를 뽑아 대조해야 누락이 보인다.

방법(의도적으로 단순·기계적):
  1. 문서 **698개 전수**를 문장으로 쪼갠다(표 행 포함).
  2. 축의 **산문 정의**만 보고 짠 정규식으로 후보 문장을 넓게 긁는다(v0의 답을 보고 짜지 않는다).
  3. 후보를 v0와 **(문서, 값) 단위로** 대조한다 — **숫자 단위로 대조하면 안 된다**(C315:
     `Navy Blue=60` 때문에 `Hunter Green=60`의 누락이 '덮인 것'으로 세어졌다).
     ★그리고 **상품 계열 단위로 한 번 더** 대조한다: 같은 상품의 *다른 문서*가 이미 그 값을 냈다면
     온톨로지는 그 사실을 아는 것이고, 문서 단위 미덮음은 중복 문장을 안 담았다는 뜻일 뿐이다.
     계열 키는 문서 **파일명**에서 기계적으로 얻는다(`…_beige_012` → `…_beige`) — 본문을 안 뜯는다.
     ⇒ **계열에서도 안 덮인 것**만이 *"온톨로지가 모르는 값"* 후보다.
  4. 안 덮인 후보를 **문서·축자 인용과 함께** 낸다 ⇒ 사람이 per-case로 읽어 *진짜 누락*과
     *축 밖 진술*을 가른다([[08]] — 집계에서 결론 직행 금지).
  5. 역방향도 낸다: v0 행 중 **이 정규식이 못 본 것** = 정규식의 사각지대 크기.

⚠**한계(먼저 적는다)**
  · 정규식은 **내가 고른 것**이다. 후보 0이면 *"누락 없음"* 이 아니라 *"이 그물로는 못 봤다"* 다.
  · **(문서, 값) 대조는 문서 안의 귀속을 못 잰다.** 한 문서가 같은 값을 두 주어에 대해 말하면
    v0가 하나만 담아도 '덮임'으로 세어진다 ⇒ 그 경우를 **따로 플래그**해서 사람에게 넘긴다.
  · 이것은 **분석 도구이지 엔진이 아니다**([[59]]는 엔진을 규율한다). 여기서 나온 어떤 문자열도
    엔진·A2·A3로 들어가지 않는다.

usage: x141_ontology_omission_probe.py --docs <dir> --ontology <a3_policy_ontology_v0.json[.gz]>
         [--dump 40]
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

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ── 후보 그물 — **축의 산문 정의에서만** 파생한다 (v0 산출물을 보고 짜지 않는다) ──────────
#   referrer_tenure_days: "추천인이 이미 체킹 계좌를 보유해 온 최소 일수(관계기간)"
#   annual_referral_limit: "그 상품에 대해 연간 허용되는 추천 보너스 최대 건수"
NET = {
    "referrer_tenure_days": {
        "num": re.compile(r"\b(\d{1,4})\s*(?:calendar\s+)?days?\b", re.I),
        # 기간-문턱을 가리키는 표지 중 **하나라도** 있으면 후보 (넓게)
        "mark": re.compile(r"\b(at least|minimum|minimums|no less than|no fewer than|"
                           r"maintained|maintain|relationship duration|tenure|held|holder|"
                           r"account (?:age|history)|existing|prior to|before)\b", re.I),
    },
    "annual_referral_limit": {
        "num": re.compile(r"\b(\d{1,4})\b"),
        "mark": re.compile(r"\b(per (?:calendar )?year|annual|annually|per year|"
                           r"in any 12|calendar year)\b", re.I),
        # 연간 무언가가 다 상한은 아니다 — 추천/보너스/상한 어휘를 함께 요구(넓게)
        "mark2": re.compile(r"\b(referral|referrals|refer|bonus|bonuses|limit|limits|"
                            r"maximum|max|cap|capped|up to|no more than)\b", re.I),
    },
}


def _norm(s):
    return " ".join(str(s).split())


def sentences(text):
    """마크다운 산문 + 표 행 + 번호 목록을 문장 단위로 쪼갠다."""
    t = str(text or "").replace("\r", "")
    out = []
    for line in t.split("\n"):
        line = line.strip()
        if not line:
            continue
        if line.startswith("|"):            # 표는 행이 곧 진술이다
            out.append(_norm(line))
            continue
        for p in re.split(r"(?<=[.!?])\s+", line):
            p = _norm(p)
            if p:
                out.append(p)
    return out


def candidates(sent):
    """이 문장이 어느 축의 후보인가 → [(axis, value), …]. 축마다 여러 수치일 수 있다."""
    got = []
    for ax, net in NET.items():
        if not net["mark"].search(sent):
            continue
        if "mark2" in net and not net["mark2"].search(sent):
            continue
        for m in net["num"].finditer(sent):
            try:
                v = int(m.group(1))
            except Exception:
                continue
            if v <= 0 or v > 3650:
                continue
            got.append((ax, v))
    return got


FAMILY = re.compile(r"_\d{3}$")


def family(doc_id):
    """상품 계열 키 — 파일명의 말미 일련번호만 떼어낸다(본문 해석 아님)."""
    return FAMILY.sub("", str(doc_id or ""))


def load_ontology(path):
    op = gzip.open if path.endswith(".gz") else io.open
    with op(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", required=True)
    ap.add_argument("--ontology", required=True)
    ap.add_argument("--dump", type=int, default=40, help="안 덮인 후보 인쇄 상한")
    a = ap.parse_args()

    onto = load_ontology(a.ontology)
    rows = onto.get("rows") or []
    # v0 색인 — **문서 단위**로 (값 → 행). 전역 숫자 집합은 만들지 않는다(C315).
    by_doc = collections.defaultdict(list)
    by_fam = collections.defaultdict(list)
    for r in rows:
        did = str((r.get("source") or {}).get("doc") or "")
        by_doc[did].append(r)
        by_fam[family(did)].append(r)
    print("v0: 행 %d · 문서 %d개서 나옴 · 축 %s"
          % (len(rows), len(by_doc), ", ".join(sorted(onto.get("axes") or {}))))

    docs = sorted(glob.glob(os.path.join(a.docs, "*.json")))
    print("문서 전수 %d개 스캔 (LLM 0)" % len(docs))

    stat = collections.Counter()
    uncovered = []          # (doc, title, axis, value, sentence) — 문서 단위 미덮음
    unknown = []            # (doc, title, axis, value, sentence) — **계열에서도** 미덮음
    ambiguous = []          # (doc, axis, value, n_cands, n_rows) — 같은 값 다수 후보
    cand_quotes = collections.defaultdict(list)   # doc → 후보 문장들(역방향 대조용)

    for p in docs:
        d = json.load(io.open(p, encoding="utf-8"))
        did = d.get("id") or os.path.basename(p)
        title = _norm(d.get("title") or "")
        drows = by_doc.get(did) or []

        per_docval = collections.Counter()        # (axis, value) → 후보 문장 수
        seen = {}
        for s in sentences(d.get("content") or ""):
            cs = candidates(s)
            if cs:
                cand_quotes[did].append(s)
            for ax, v in cs:
                stat["후보(문장×값)"] += 1
                per_docval[(ax, v)] += 1
                seen.setdefault((ax, v), s)

        for (ax, v), n_c in sorted(per_docval.items()):
            hit = [r for r in drows if r.get("axis") == ax and r.get("value") == v]
            if hit:
                stat["덮임"] += 1
                if n_c > len(hit):
                    # 같은 문서에서 같은 값을 말하는 문장이 v0 행보다 많다 ⇒ 숫자 뒤에 숨을 수 있다
                    ambiguous.append((did, ax, v, n_c, len(hit)))
            else:
                stat["안 덮임"] += 1
                uncovered.append((did, title, ax, v, seen[(ax, v)]))
                fam_hit = [r for r in (by_fam.get(family(did)) or [])
                           if r.get("axis") == ax and r.get("value") == v]
                if not fam_hit:
                    stat["계열에서도 안 덮임"] += 1
                    unknown.append((did, title, ax, v, seen[(ax, v)]))

    # 역방향 — v0 행의 인용을 이 그물이 후보로 잡았나 (사각지대 크기)
    blind = []
    for r in rows:
        did = str((r.get("source") or {}).get("doc") or "")
        q = _norm((r.get("source") or {}).get("quote") or "")
        cq = cand_quotes.get(did) or []
        if not any((q in c) or (c in q) for c in cq):
            blind.append(r)

    print("\n" + "=" * 96)
    print("후보 (문서,축,값) 조합 %d · **v0가 덮은 %d** · 문서 단위 안 덮은 %d · "
          "★**상품 계열에서도 안 덮은 %d**"
          % (stat["덮임"] + stat["안 덮임"], stat["덮임"], stat["안 덮임"],
             stat["계열에서도 안 덮임"]))
    print("정규식 사각: v0 행 %d개 중 **이 그물이 후보로도 못 잡은 행 %d개**" % (len(rows), len(blind)))
    print("⚠'안 덮은'이 곧 누락은 아니다 — **per-case로 읽어야** 진짜 누락과 축 밖 진술이 갈린다.")

    by_ax = collections.Counter(u[2] for u in uncovered)
    for ax in sorted(NET):
        print("   %-24s 안 덮인 후보 %d" % (ax, by_ax.get(ax, 0)))

    print("\n── ★상품 계열에서도 안 덮인 후보 = **온톨로지가 모르는 값** 후보 ─────────────")
    for did, title, ax, v, s in unknown:
        print("★ [%s] %s = %d" % (did[-34:], ax, v))
        print("   제목: %s" % title[:88])
        print("   축자: %s" % s[:180])
    if not unknown:
        print("   (없음 — 이 그물로는)")

    print("\n── 문서 단위로 안 덮인 후보 (계열에 있는 것 포함 · 축자) ──────────────────")
    for did, title, ax, v, s in uncovered[:a.dump]:
        print("☐ [%s] %s = %d" % (did[-34:], ax, v))
        print("   제목: %s" % title[:88])
        print("   축자: %s" % s[:180])
    if len(uncovered) > a.dump:
        print("… %d건 더 (--dump 로 늘린다)" % (len(uncovered) - a.dump))

    if ambiguous:
        print("\n── ⚠같은 (문서,값)에 후보 문장이 v0 행보다 많다 = **귀속이 숫자 뒤에 숨을 수 있다** ──")
        for did, ax, v, n_c, n_r in ambiguous[:20]:
            print("   [%s] %s=%d · 후보 문장 %d vs v0 행 %d" % (did[-34:], ax, v, n_c, n_r))
        if len(ambiguous) > 20:
            print("   … %d건 더" % (len(ambiguous) - 20))

    if blind:
        print("\n── 정규식이 못 본 v0 행 (그물의 한계 실물) ────────────────────────────────")
        for r in blind[:15]:
            print("   %s / %s = %s · [%s] %s"
                  % (r.get("subject"), r.get("axis"), r.get("value"),
                     str((r.get("source") or {}).get("doc"))[-24:],
                     _norm((r.get("source") or {}).get("quote"))[:80]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
