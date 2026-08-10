# -*- coding: utf-8 -*-
r"""x237 — **A3 선언만 주면 스스로 검색하는가** (격리 · shell 단독 · 유료 0 · 엔진 0).

## 왜 (사용자 지시 2026-08-10)

> *"격리해서 A3 선언만 읽고도 검색 못 하는지 확인하자."*

검색 에이전트(결정론 grep)를 짓기 전에 **결손을 먼저 잰다**(⛔0 ①). A3 는 이미 153/153 행이
`source.doc` + 축자 인용을 갖고 있고(41 문서·41 주어), 파일명 규약(`doc_<group>_<name>_NNN.json`)
자체가 인덱스다. 그걸 **보여 주면** 모델이 그 파일을 집어 오는가?

## 팔 (도구는 `shell` 하나 — 정본 retrieval·BM25/embedding 금지)

  A_BARE      손님 요구 + shell                          ← 부정 통제(오늘 라이브와 같은 조건)
  B_A3        + **A3 선언 축자**(주어 → source.doc + 인용) ← 선언만으로 되나
  C_A3_NAMING + 파일명 규약 한 줄                          ← 규약을 명시하면 되나

## 무엇을 세는가 (이번엔 **실제 회수까지**)

  ⑴ 명령 원문(표본 인쇄)  ⑵ 고유명 포함  ⑶ **그 명령이 코퍼스에서 실제로 무엇을 반환하는가**
     — 명령을 **실행하지 않고** 우리가 해석해 매칭한다(모델이 낸 셸을 돌리지 않는다):
       `grep -r "X"` → 내용에 X 를 담은 문서 수 · `find -name "*Y*"` → 파일명 매칭 수 · `ls` → 0

⚠판정은 *"사업자 체킹 문서를 실제로 가져오는가"* 다. 의도가 아니라 **결과**를 센다.

실행: python x237_a3_search.py [N]
"""
import collections
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from x216_read_and_offset import chat                            # noqa: E402

DOM = "/home/woori/scratch/tau2-bench/data/tau2/domains/banking_knowledge"
A3P = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "a2", "banking_knowledge.specific.json")
TOOLS = [{"type": "function", "function": {
    "name": "shell",
    "description": ("Run standard Unix commands in the knowledge base directory: "
                    "ls, cat, head, grep -r, find."),
    "parameters": {"type": "object", "properties": {"command": {"type": "string"}},
                   "required": ["command"]}}}]
NAMING = ("Knowledge base files are named doc_<group>_<name>_NNN.json, "
          "for example doc_business_checking_accounts_sky_blue_007.json.")
PROPER = re.compile(r"(sky[_ ]blue|lime[_ ]green|hunter[_ ]green|cobalt[_ ]blue|navy[_ ]blue|"
                    r"true[_ ]blue|beige|world[_ ]blue|business[_ ]checking)", re.I)


def a3_block(limit=18):
    """A3 선언 축자 — 주어 · 축 · 값 · **출처 문서와 인용**."""
    d = json.load(open(A3P, encoding="utf-8"))
    rows = (d.get("policy_ontology") or {}).get("rows") or []
    seen, out = set(), ["Policy ontology on record (subject, axis, value, source document):"]
    for r in rows:
        s = r.get("source") or {}
        key = (r.get("subject"), s.get("doc"))
        if key in seen:
            continue
        seen.add(key)
        out.append('  %-22s %-24s %-10s  [%s] "%s"'
                   % (str(r.get("subject"))[:22], str(r.get("axis"))[:24],
                      str(r.get("value"))[:10], s.get("doc"), str(s.get("quote"))[:70]))
        if len(out) > limit:
            break
    return "\n".join(out)


def corpus():
    docs = {}
    for p in glob.glob(os.path.join(DOM, "documents", "doc_*.json")):
        try:
            d = json.load(open(p, encoding="utf-8"))
        except Exception:
            continue
        docs[os.path.basename(p)] = str(d.get("content") or "") + " " + str(d.get("title") or "")
    return docs


DOCS = None


def evaluate(cmd):
    """모델이 낸 셸 명령을 **실행하지 않고** 해석해 매칭 결과를 센다."""
    global DOCS
    if DOCS is None:
        DOCS = corpus()
    hits = []
    m = re.search(r"grep[^\"']*[\"']([^\"']+)[\"']", cmd)
    if m:
        pat = m.group(1)
        try:
            rx = re.compile(pat, re.I)
        except Exception:
            rx = re.compile(re.escape(pat), re.I)
        hits = [n for n, c in DOCS.items() if rx.search(c) or rx.search(n)]
    else:
        m = re.search(r"find[^\n]*-name\s+[\"']?\*?([A-Za-z0-9_\-]+)\*?[\"']?", cmd)
        if m:
            pat = m.group(1).lower()
            hits = [n for n in DOCS if pat in n.lower()]
        elif re.match(r"\s*(ls|cat|head|tail)\b", cmd):
            if "business_checking" in cmd:
                hits = [n for n in DOCS if "business_checking" in n]
    biz = [h for h in hits if "business_checking_accounts" in h]
    return len(hits), len(biz)


def task_req():
    o = json.load(open(os.path.join(DOM, "tasks", "task_070.json"), encoding="utf-8"))
    o = o[0] if isinstance(o, list) and o else o
    return " ".join(((o.get("user_scenario") or {}).get("instructions") or "").split())[:1800]


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 8
    req, a3 = task_req(), a3_block()
    print(a3[:600], "\n")
    arms = [("A_BARE", req),
            ("B_A3", req + "\n\n" + a3),
            ("C_A3_NAMING", req + "\n\n" + a3 + "\n" + NAMING)]
    ASK = "Search the knowledge base for what you need. Make one shell tool call."
    for name, body in arms:
        proper, gotbiz, shown, tot_hits = 0, 0, [], 0
        for i in range(n):
            try:
                r = chat(body + "\n\n" + ASK, TOOLS, 0.0 if i == 0 else 0.7, 200)
            except Exception:
                continue
            tcs = r.get("tool_calls") or []
            if not tcs:
                continue
            f = tcs[0].get("function") or tcs[0]
            args = f.get("arguments") or "{}"
            try:
                cmd = json.loads(args).get("command", "") if isinstance(args, str) else \
                    (args or {}).get("command", "")
            except Exception:
                cmd = str(args)
            h, b = evaluate(str(cmd))
            tot_hits += h
            if b:
                gotbiz += 1
            if PROPER.search(str(cmd)):
                proper += 1
            if len(shown) < 2:
                shown.append(" ".join(str(cmd).split())[:120])
        print("  %-12s 고유명 %d/%d · **사업자문서 회수 %d/%d** · 평균 매칭 %.1f"
              % (name, proper, n, gotbiz, n, tot_hits / max(1, n)))
        for s in shown:
            print("        $ %s" % s)
    print("\n※ 판정 — B/C 에서 **사업자문서 회수**가 A 보다 뚜렷이 높으면 결손은 *전달*이고,"
          "\n  A3 선언을 보여 주는 것만으로 닫힌다(결정론 grep 을 지을 필요가 없다)."
          "\n  셋 다 낮으면 검색 에이전트(결정론 읽기)가 정당해진다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
