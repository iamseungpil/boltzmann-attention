# -*- coding: utf-8 -*-
r"""x236 — **070 은 검색을 왜 못했나**: 도구 선택·질의·재시도 (격리 · 유료 0 · 엔진 0).

## 왜 (오늘 궤적 실측)

070 은 세 도구를 **다 갖고 있었다**(`alltools` = `KB_search_bm25` · `KB_search_dense` · `shell`).
그런데 —

  · `KB_search_dense` **3회** (2번째와 3번째 질의가 **완전히 동일**)
  · `KB_search_bm25` **0회** · `shell`(grep) **0회**
  · 질의 축자: *"business checking account with at least $15/month in ATM fee rebates, zero
    overdraft fees, minimum balance under $10,000, and at least 1% APY"* — **고유명 0**
  · 결과: 회수 30건 중 사업자 체킹 문서 **0건**, 프로모션 문서 **0건**

⇒ 도구가 없어서가 아니라 **안 바꿔서** 졌다. 그것이 능력인지 부하인지를 여기서 가른다(⛔0 ①).

## 팔 (도구는 **실제 스키마 그대로** 준다 · 호출을 세는 것이지 정답을 채점하지 않는다)

  ISO_BARE    손님 요구 + 3도구                      ← 격리에서도 dense 만 쓰나(=능력)
  ISO_CAT     + **카탈로그**(제품 이름 ↔ 문서 접두사) ← 이름을 알면 고유명으로 검색하나
  ISO_RETRY   요구 + 3도구 + *"1차 검색이 빈 결과였다"* ← 실패 후 **바꾸나**(가장 싼 처방)
  TRAJ        실패 궤적 문맥 + 같은 도구               ← 궤적에서 죽나(C398 형 재현)

세는 것: ⑴ 고른 도구 ⑵ 질의에 **고유명**이 들어가는가 ⑶ (RETRY) 도구·질의가 **바뀌는가**.

⚠카탈로그는 **문서 id 에서 빌드 시점 유도**다(x203 과 같은 방법·새 저작 0·gold 무참조).
⚠이 프로브는 **정답을 채점하지 않는다** — 행동(무엇을 부르는가)만 센다.

실행: python x236_search_strategy.py [N]
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
# ★정본 retrieval (사용자 결정 2026-08-10): **shell(grep) + A3 로 문서 결정**.
#   BM25·embedding 은 **금지**다 — 070 실측이 그 이유다(dense 3회·같은 질의 반복·사업자 문서 0건).
#   환경에 `prompts/grep_only.md` 설정이 이미 있어 하네스 개조 없이 성립한다.
#   ⚠[[54]] 비교 규격: `alltools`(3도구) 수치와 **섞지 않는다**. 별도 라벨로 보고한다.
TOOLS = [
    {"type": "function", "function": {
        "name": "shell",
        "description": ("Run standard Unix commands in the knowledge base directory: "
                        "ls, cat, head, grep -r, find."),
        "parameters": {"type": "object",
                       "properties": {"command": {"type": "string"}},
                       "required": ["command"]}}},
]
EMPTY = ("Your previous search returned no business checking account documents - "
         "the results were about personal checking and savings accounts.")


def catalog():
    """제품 이름 ↔ 문서 접두사 (문서 id 에서 유도 · 새 저작 0)."""
    per = collections.defaultdict(set)
    for p in glob.glob(os.path.join(DOM, "documents", "doc_*.json")):
        m = re.match(r"doc_([a-z_]+?)_([a-z_]+?)_(\d+)$", os.path.basename(p)[:-5])
        if not m:
            continue
        group, prod = m.group(1), m.group(2)
        if "business" in group or "checking" in group or "savings" in group:
            per[group].add(prod)
    lines = ["Document catalogue on record (file names in the knowledge base):"]
    for g in sorted(per):
        names = sorted(n for n in per[g] if n != "general")
        if names:
            lines.append("  %s: %s" % (g, ", ".join(n.replace("_", " ").title()
                                                    for n in names[:12])))
            lines.append("      files look like doc_%s_<name>_NNN.json" % g)
    return "\n".join(lines)


def task_req():
    o = json.load(open(os.path.join(DOM, "tasks", "task_070.json"), encoding="utf-8"))
    o = o[0] if isinstance(o, list) and o else o
    return " ".join(((o.get("user_scenario") or {}).get("instructions") or "").split())[:1800]


def traj_ctx():
    """오늘 실패 궤적에서 **첫 검색 직전까지**의 문맥."""
    p = "/home/woori/scratch/tau2-bench/data/simulations/bank_m3_20260810s/results.json"
    try:
        d = json.load(open(p, encoding="utf-8"))
    except Exception:
        return None
    for s in d.get("simulations") or []:
        if s.get("task_id") != "task_070":
            continue
        out = []
        for m in s.get("messages") or []:
            c = " ".join(str(m.get("content") or "").split())
            if c:
                out.append("[%s] %s" % (m.get("role"), c[:600]))
            if len(out) > 12:
                break
        return "\n".join(out)
    return None


PROPER = re.compile(r"\b(Sky Blue|Lime Green|Hunter Green|Cobalt Blue|Navy Blue|True Blue|"
                    r"Beige|World Blue|Gold Saver|Silver Plus|Gold Plus)\b", re.I)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 8
    req, cat, tj = task_req(), catalog(), traj_ctx()
    print(cat, "\n")
    arms = [("ISO_BARE", req),
            ("ISO_CAT", req + "\n\n" + cat),
            ("ISO_RETRY", req + "\n\n" + EMPTY),
            ("TRAJ", (tj or req))]
    ASK = "Search the knowledge base for what you need. Make one tool call."
    for name, body in arms:
        if name == "TRAJ" and tj is None:
            print("  TRAJ 건너뜀(궤적 없음)")
            continue
        tools_used, proper, shells = collections.Counter(), 0, 0
        for i in range(n):
            try:
                r = chat(body + "\n\n" + ASK, TOOLS, 0.0 if i == 0 else 0.7, 200)
            except Exception as e:
                tools_used["ERR %s" % type(e).__name__] += 1
                continue
            tcs = r.get("tool_calls") or []
            if not tcs:
                tools_used["(호출 없음)"] += 1
                continue
            f = (tcs[0].get("function") or tcs[0])
            nm = f.get("name")
            args = str(f.get("arguments") or "")
            tools_used[nm] += 1
            if PROPER.search(args):
                proper += 1
            if nm == "shell" and ("grep" in args or "find" in args or "ls" in args):
                shells += 1
        print("  %-10s 도구 %s · 고유명 %d/%d · shell탐색 %d"
              % (name, dict(tools_used), proper, n, shells))
    print("\n※ 읽는 법 — ISO_BARE 에서 bm25/shell 이 나오면 결손은 **부하**(궤적에서만 죽는다)."
          "\n  격리에서도 dense 만이면 **능력**이고, 그때 ISO_CAT 이 정당해진다."
          "\n  ISO_RETRY 만으로 바뀌면 필요한 것은 *'빈 결과였다'는 사실의 전달* 하나다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
