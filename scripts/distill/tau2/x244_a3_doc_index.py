# -*- coding: utf-8 -*-
r"""x244 — A3 에 **문서 색인**을 선언한다: `(문서군, 주어) → 문서 id` (빌드 시점 · 유료 0 · LLM 0).

## 왜 (사용자 지시 2026-08-11 *"070 1번부터"* · C405⒟)

070 의 결손은 검색이다 — 프로모션 문서 4개를 **전부 미회수**했고 질의에 고유명이 0이었다.
처방은 정해져 있다: **shell + A3 로 문서를 결정한다**(BM25·임베딩 금지). 그런데 결정론이
따라갈 **링크가 없었다** — A3 의 41개 링크는 전부 추천-축 문서고 사업자 체킹 79문서는 밖이다
([[50]] ADB: 링크 커버리지가 곧 시야).

## 왜 이 모양인가 (x243 이 결정론을 한 층 줄였다)

축으로 문서를 골라 줄 필요가 **없다**:

    S1 축별 문장 + 활성 프로모션   8/8
    S2 제품 문서 **전문** + 프로모션 8/8      ← 축 선별 없이도 닫힌다
    S3 문서 **앞 400자** + 프로모션  8/8      ← 예산도 작다
    S4 문서만(프로모션 없음)        0/8      ← 유효창이 본체로 남는다

⇒ 색인은 `(주어 → 문서 id)` 로 **끝난다**. 축 어휘도, 축 형식화 슬롯도 짓지 않는다(⛔0 ③).

## 어디서 오는가 ([[23]])

**파일명뿐**이다 — `doc_<문서군>_<주어>_NNN.json`. env 에서 기계로 나오므로 저작 비용 0 이고,
x203 이 종류(`kind`)를 유도할 때 쓴 **바로 그 규칙**이다. 값도 축도 적지 않는다: **id 만**.

⚠엔진이 런타임에 파일명을 뜯으면 [[59]] 위반이다. 그래서 **여기(빌드 시점)**에서 유도해 적고,
  엔진은 적힌 것을 읽기만 한다(`t2_search.docs_for`).
⚠`(general)` 은 주어가 아니라 **범위**다 — 주어 없는 문서군 공통 문서로 따로 적는다.
⚠두 층에 바이트 동일로 쓴다([[24]]).

실행: py -3 x244_a3_doc_index.py [--apply]
"""
import argparse
import collections
import glob
import io
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
LAYERS = ["a2/banking_knowledge.specific.json", "a2/banking_knowledge.gate.json"]
DOCS = "/home/woori/scratch/tau2-bench/data/tau2/domains/banking_knowledge/documents"
# 문서군 — x203 의 목록에서 출발하되 **코퍼스가 실제로 가진 것**으로 채운다.
#
# ★첫 판이 184 문서를 놓쳤고, 하필 그 안에 **071 의 문서군(`business_savings_accounts` 82건)**
#   이 있었다. x203 의 목록은 *종류 태깅*이 필요했던 군만 담고 있었을 뿐 코퍼스 전체가 아니다 —
#   그 목록을 색인에 그대로 쓴 것이 [[50]] ADB 가 경고한 **시야 결손**을 그대로 만든 것이다.
#   ⇒ 목록을 손으로 늘리는 대신 **코퍼스에서 유도**한다: 파일명은 `doc_<군>_<주어>_NNN` 이고
#     같은 군은 여러 주어를 가지므로, *두 개 이상의 서로 다른 주어를 거느린 최장 접두사* 가 군이다.
#   ⚠빌드 시점 규칙이다. 엔진은 이 유도를 하지 않고 적힌 색인을 읽기만 한다([[59]]).
from x203_tag_a3_kind import GROUPS as _KIND_GROUPS                # noqa: E402

_NUM = re.compile(r"^(.*)_(\d+)$")


def corpus_groups():
    """군 어휘를 코퍼스에서 유도한다 — 주어를 둘 이상 거느린 접두사 중 **가장 긴 것**."""
    stems = []
    for p in sorted(glob.glob(os.path.join(DOCS, "doc_*.json"))):
        m = _NUM.match(os.path.basename(p)[:-5])
        if m:
            stems.append(m.group(1)[4:])                 # `doc_` 를 뗀 `<군>_<주어>`
    subs = collections.defaultdict(set)
    for s in stems:
        parts = s.split("_")
        for k in range(1, len(parts)):                   # 가능한 모든 분할점
            subs["_".join(parts[:k])].add("_".join(parts[k:]))
    # ★첫 판은 **가장 긴** 접두사를 골랐고, 그래서 `business_savings_accounts` 가
    #   `…_gold`·`…_silver` 로 쪼개졌다(주어가 `plus_saver` 가 되어 제품 이름이 사라졌다).
    #   군은 *많이 묶는 것*이지 *긴 것*이 아니다 ⇒ **덮는 문서 수** 우선의 탐욕 피복으로 고른다.
    files = collections.defaultdict(set)                 # 접두사 → 그 접두사로 시작하는 파일 stem
    for s in stems:
        parts = s.split("_")
        for k in range(1, len(parts)):
            files["_".join(parts[:k])].add(s)
    cand = sorted((g for g, v in subs.items() if len(v) >= 2),
                  key=lambda g: (-len(files[g]), len(g)))
    taken, out = set(), []
    for g in cand:
        mine = {s for s in files[g] if s not in taken}
        if len(mine) < 2 or len({s[len(g) + 1:] for s in mine}) < 2:
            continue
        taken |= mine
        out.append(g)
    return sorted(set(out) | set(_KIND_GROUPS), key=len, reverse=True)


def _common_prefix(stems):
    """그 군에 **실제로 들어온** 파일들의 최장 공통 접두사 (주어가 최소 한 토막은 남게)."""
    parts = [s.split("_") for s in stems]
    k = 0
    while all(len(p) > k + 1 and p[k] == parts[0][k] for p in parts):
        k += 1
    return "_".join(parts[0][:k]) if k else None


GROUPS = corpus_groups()


def index():
    """`{문서군: {주어: [문서 id]}}` · 주어 없는 것은 `_general_` 아래."""
    out = collections.defaultdict(lambda: collections.defaultdict(list))
    other = []
    for p in sorted(glob.glob(os.path.join(DOCS, "doc_*.json"))):
        b = os.path.basename(p)[:-5]
        g = next((g for g in GROUPS if b.startswith("doc_" + g + "_")), None)
        if not g:
            other.append(b)
            continue
        m = re.match(r"doc_%s_(.+)_(\d+)$" % re.escape(g), b)
        if not m:
            other.append(b)
            continue
        subj = m.group(1)
        key = "_general_" if "(general)" in subj or subj == g else subj
        out[g][key].append(b)
    # ★배정이 끝난 뒤 이름을 고쳐 단다 (2026-08-11·사용자 지적 *"군 이름이 뭔가"*).
    #   군 후보는 **가장 많이 덮는** 접두사로 뽑히므로 이름이 실제보다 짧게 나온다 — 사업자
    #   세이빙 82문서가 `business` 라는 이름을 달았다. 이름이 내부 키로만 쓰이면 무해하지만
    #   이것은 **닫힌 집합으로 모델에게 보여 줄 이름**이기도 하다: `business` 를 주고 *"사업자
    #   세이빙 계좌"* 를 고르라 하면 못 고른다(C376 주어 불일치와 같은 자리).
    #   ⇒ 그 군에 **실제로 들어온 파일들의 최장 공통 접두사**로 다시 부른다. 재배정은 없다.
    renamed = {}
    for g, subs in out.items():
        stems = [b[4:] for v in subs.values() for b in v]
        pre = _common_prefix(stems) if stems else None
        if not pre or pre == g:
            renamed[g] = subs
            continue
        fixed = collections.defaultdict(list)
        for v in subs.values():
            for b in v:
                s = b[4:][len(pre) + 1:]
                fixed["_general_" if "(general)" in s or not s else s].append(b)
        renamed[pre] = fixed
    return renamed, other


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    idx, other = index()
    total = sum(len(v) for g in idx.values() for v in g.values())
    print("문서군 %d · 색인된 문서 %d · 규칙 밖 %d" % (len(idx), total, len(other)))
    for g in sorted(idx):
        subs = idx[g]
        n = sum(len(v) for v in subs.values())
        print("  %-28s 주어 %2d · 문서 %3d   %s"
              % (g, len([k for k in subs if k != "_general_"]), n,
                 ", ".join(sorted(k for k in subs if k != "_general_")[:6])))
    if other:
        print("  ⚠규칙 밖(색인 안 함·시야 밖으로 계수한다): %d 건 예: %s" % (len(other), other[:3]))
    if not a.apply:
        print("\n(--apply 없이는 쓰지 않는다)")
        return 0
    rows = {g: {s: sorted(v) for s, v in subs.items()} for g, subs in idx.items()}
    for rel in LAYERS:
        p = os.path.join(HERE, rel)
        txt = io.open(p, encoding="utf-8").read()
        doc = json.loads(txt)
        if json.dumps(doc, ensure_ascii=False, indent=1) + ("\n" if txt.endswith("\n") else "") \
                != txt:
            print("  중단: %s 재직렬화가 바이트 동일하지 않다" % rel)
            return 1
        doc["policy_ontology"]["doc_index"] = rows
        doc["policy_ontology"]["_note_doc_index"] = (
            "★출처 = env 파일명뿐(`doc_<군>_<주어>_NNN.json`·빌드 시점 유도·x244·저작 0). "
            "x203 이 종류를 유도할 때 쓴 같은 규칙이다. **id 만** 적고 값·축은 적지 않는다 — "
            "x243 실측이 축 선별 없이 닫힘을 보였다(S2/S3 8/8 · 축별 문장 S1 8/8 과 동률 · "
            "프로모션 빼면 S4 0/8). 엔진은 런타임에 파일명을 뜯지 않고 여기 적힌 것을 읽기만 "
            "한다([[59]]). 시야 = 이 색인의 크기다([[50]] ADB).")
        out = json.dumps(doc, ensure_ascii=False, indent=1) + ("\n" if txt.endswith("\n") else "")
        io.open(p, "w", encoding="utf-8", newline="").write(out)
        print("  기록: %-40s 문서군 %d · 문서 %d" % (rel, len(rows), total))
    return 0


if __name__ == "__main__":
    sys.exit(main())
