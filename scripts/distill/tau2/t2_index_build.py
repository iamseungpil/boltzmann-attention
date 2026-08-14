# -*- coding: utf-8 -*-
"""**다중 색인 빌더** — 문서·도구를 여러 축으로 선언한다 (도메인-일반·기계 도출·저작 0).

사용자 지시(2026-08-14 야간): *"도구 설명 표면화하라. 그리고 문서도 다중으로 의미를 설명하라.
한 가지 형태로 하지 말고 다중으로 접근할 수 있게 하라."* · *"general 이 상품에 관한 게 아니면
의미적으로 어떤 걸 나타내는지 확실히 하라."*

## 왜 (측정 근거)

  x316  이름만 46개 → **0/8** · 정답 이름 1개 → 4/8 · 정답+방해자 4개 → 0/8
  x318  같은 컷에서 bm25 질의 **1/8** ↔ 오답 도구를 시도해 **설명을 읽은 뒤 8/8**
        (msg58 축자: *"apply_statement_credit_8472 는 **신용카드 계좌용**이지 체킹용이 아니다"*)
  ⇒ 열쇠는 "더 보여주기"가 아니라 **경계를 말해 주기**다([[63]] — 모델은 스스로 배제를 못 한다).
    도구 **설명**은 무엇이 아닌지를 말하고, 이름은 말하지 않는다.

## 축 (전부 기계 도출 · 판정 0 · gold 무접촉)

  filename  군·주어            — 기존 `doc_index` 와 같은 출처(x244)
  tools     그 문서가 **이름을 대는** 레지스트리 도구들 (정규식·레지스트리 교집합)
  kind      **도구를 대면 `action`, 아니면 `reference`** ← 어휘가 아니라 **구조**로 가른다
            (`## Procedure` 같은 영어 표제어에 기대지 않는다 — 도메인·언어 불변)
  title     파일의 `title` 필드 그대로
  tool_index  도구 → {설명(첫 줄), 그 도구를 대는 문서들}

`_general_` 의 의미는 **선언하지 않고 잰다**: 이 빌더가 `_general_` 문서 중 `action` 비율을
인쇄한다. 그 수가 곧 *"이 버킷이 제품이 아니라 무엇에 관한 것인가"* 의 답이다.

⚠엔진은 여기서도 **고르지 않는다** — 축을 만들 뿐이고, 어느 축으로 접근할지는 LLM 몫([[62]] ④).
⚠도메인 리터럴 0: 입력은 `documents/` 디렉터리와 **도구 이름 집합**뿐이다([[05]]·[[16]] 일반화 3축).
"""
import collections
import glob
import io
import json
import os
import re

TOOLNAME = re.compile(r"\b[a-z][a-z_]{5,}_\d{4}\b")
FNAME = re.compile(r"^doc_(?P<group>.+?)_(?P<subject>[^_]+(?:_[^_]+)*?)_(?P<n>\d+)$")


def load_docs(doc_dir):
    out = []
    for p in sorted(glob.glob(os.path.join(doc_dir, "*.json"))):
        try:
            out.append(json.load(io.open(p, encoding="utf-8")))
        except Exception:
            continue
    return out


def tool_names_from_module(mod):
    """도구 이름 → 설명(docstring 첫 문장). 소스에서 기계 추출·저작 0."""
    import inspect
    out = {}
    for _n, cls in inspect.getmembers(mod, inspect.isclass):
        for fn, f in inspect.getmembers(cls, inspect.isfunction):
            if fn.startswith("_"):
                continue
            d = " ".join((f.__doc__ or "").split())
            if d:
                out[fn] = d
    return out


def facets(docs, tool_desc):
    """문서마다 다중 축. `kind` 는 **도구를 대는가**로 가른다(구조 기준·어휘 무관)."""
    names = set(tool_desc or {})
    out = {}
    for d in docs:
        did = d.get("id") or ""
        body = "%s\n%s" % (d.get("title") or "", d.get("content") or "")
        tools = sorted(set(TOOLNAME.findall(body)) & names)
        m = FNAME.match(did)
        out[did] = {
            "title": d.get("title") or "",
            "group": (m.group("group") if m else ""),
            "subject": (m.group("subject") if m else ""),
            "tools": tools,
            "kind": "action" if tools else "reference",
        }
    return out


def tool_index(fac, tool_desc):
    """도구 → {설명, 그 도구를 대는 문서들}. **설명이 축의 본체**다(x318 8/8 의 출처)."""
    by = collections.defaultdict(list)
    for did, f in fac.items():
        for t in f["tools"]:
            by[t].append(did)
    return {t: {"description": tool_desc.get(t, ""), "docs": sorted(by.get(t, []))}
            for t in sorted(tool_desc or {})}


def build(doc_dir, tool_desc):
    docs = load_docs(doc_dir)
    fac = facets(docs, tool_desc)
    return {"doc_facets": fac, "tool_index": tool_index(fac, tool_desc)}


def summarize(idx):
    fac = idx["doc_facets"]
    ti = idx["tool_index"]
    kinds = collections.Counter(f["kind"] for f in fac.values())
    gen = [f for f in fac.values() if f["subject"] == "(general)" or "general" in f["subject"]]
    gen_action = sum(1 for f in gen if f["kind"] == "action")
    print("문서 %d · action %d · reference %d" % (len(fac), kinds["action"], kinds["reference"]))
    print("도구 %d종 · 설명 있는 것 %d · 문서에 등장하는 것 %d"
          % (len(ti), sum(1 for v in ti.values() if v["description"]),
             sum(1 for v in ti.values() if v["docs"])))
    if gen:
        print("`_general_` 계열 문서 %d · 그중 action %d (%.0f%%)  ← 이 버킷의 의미"
              % (len(gen), gen_action, 100.0 * gen_action / len(gen)))
    orphan = [t for t, v in ti.items() if not v["docs"]]
    print("문서가 한 건도 안 대는 도구 %d종%s"
          % (len(orphan), (" 예: " + ", ".join(orphan[:4])) if orphan else ""))
    multi = [t for t, v in ti.items() if len(v["docs"]) > 1]
    print("문서 2건 이상이 대는 도구 %d종" % len(multi))


def main(argv):
    doc_dir = argv[0] if argv else "."
    import importlib
    modname = argv[1] if len(argv) > 1 else "tau2.domains.banking_knowledge.tools"
    mod = importlib.import_module(modname)
    idx = build(doc_dir, tool_names_from_module(mod))
    summarize(idx)
    out = argv[2] if len(argv) > 2 else None
    if out:
        io.open(out, "w", encoding="utf-8").write(
            json.dumps(idx, ensure_ascii=False, indent=1))
        print("기록:", out)
    return 0


if __name__ == "__main__":
    import sys
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    except Exception:
        pass
    sys.exit(main(sys.argv[1:]))
