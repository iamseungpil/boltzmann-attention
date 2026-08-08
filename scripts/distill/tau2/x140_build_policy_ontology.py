# -*- coding: utf-8 -*-
"""x140 — **A3 정책 온톨로지 생성기**: 문서 전수를 한 번 읽고 관계로 굳힌다 (유료 0·로컬 vllm).

정본 = `A3_POLICY_ONTOLOGY_DESIGN_2026_08_08.md`.
§0z 정의: *"scaffold·결정론이 비용과 완결성을 위해 미리 formalize한 정책 규칙을,
각 도구·결정기에서 찾기 쉽게 형식화한 관계"*.

이 도구가 지키는 것:
  · **문서 전수**(698개) — 완결성의 근거는 *모집단을 안다*는 것이다(§0a). 표본을 뽑지 않는다.
  · **문서 하나씩**(`split`) — C313 실측: 덩어리로 물으면 12~44%, 항목별이면 92~100%.
  · **인용 실재 검증** — 모델이 낸 인용문이 **그 문서에** 실제로 있어야 채택한다([[22]] 근거-우선).
  · **축은 닫힌 집합** — 선언 밖 축은 버린다(§2a op 동결과 같은 규율).
  · **증분 저장** — 중단돼도 이어서 한다(오늘 캐시를 끝에서만 저장해 25분을 날릴 뻔했다).
  · **충돌은 인쇄** — 같은 (주어, 축)에 다른 값이면 **양쪽 인용과 함께** 남긴다([[25]]).

⚠이 파일은 **빌드 도구**이지 런타임 엔진이 아니다. 프롬프트에 도메인 어휘가 있는 것은 여기까지
허용된다([[59]]는 엔진을 규율한다) — 산출물만 A3로 가고, 이 스크립트는 런타임에 안 돈다.
⚠v0의 축은 **둘**(관계기간 문턱·연간 상한)이다. 늘리는 절차는 설계서 §9-2.

usage: x140_build_policy_ontology.py --docs <dir> --out ontology.json \
         [--base http://localhost:8140/v1] [--limit 20] [--save-every 20]
"""

import argparse
import collections
import glob
import io
import json
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# 축 정의 — **닫힌 집합**. 각 축은 (이름, 산문 정의, 맞댈 사실 이름, 비교).
AXES = {
    "referrer_tenure_days": {
        "desc": "the minimum number of days the REFERRER must already have held a checking "
                "account (relationship duration / tenure) to be eligible to refer",
        "against": "tenure_days", "compare": "ge"},
    "annual_referral_limit": {
        "desc": "the maximum number of referral bonuses allowed per year for that product",
        "against": "type_usage", "compare": "le"},
}

PROMPT = """You are reading ONE internal banking policy document.

Extract ONLY facts of these kinds:
{axes}

Rules:
- Report a fact ONLY if this document states it explicitly. Do not infer.
- "subject" is the product/account name exactly as this document writes it.
- "quote" must be copied VERBATIM from the document (a contiguous span, >= 12 chars).
- If the document states nothing of these kinds, return an empty list.

Return JSON only, no prose:
[{{"subject": "...", "axis": "...", "value": <integer>, "quote": "..."}}]

DOCUMENT:
{text}
"""


class Agent(object):
    def __init__(self, model, base):
        self.llm = model if model.startswith("openai/") else "openai/" + model
        self.llm_args = {"temperature": 0.0, "api_base": base, "api_key": "dummy"}


def ask(agent, la, UM, text):
    axes = "\n".join("- %s: %s" % (k, v["desc"]) for k, v in AXES.items())
    prompt = PROMPT.format(axes=axes, text=text[:90000])
    try:
        um = UM(role="user", content=prompt)
    except TypeError:
        um = UM(content=prompt)
    kw = {k: v for k, v in dict(agent.llm_args).items() if "tool" not in k}
    sub = la.generate(model=agent.llm, tools=None, messages=[um], call_name="x140", **kw)
    return getattr(sub, "content", None) or ""


def parse(raw, doc_text, doc_id):
    """모델 응답 → 채택 행. **인용이 그 문서에 실재할 때만** 채택한다."""
    import re
    m = re.search(r"\[.*\]", str(raw or ""), re.S)
    if not m:
        return [], 0
    try:
        rows = json.loads(m.group(0))
    except Exception:
        return [], 0
    hay = " ".join(str(doc_text).split())
    out, rejected = [], 0
    for r in rows if isinstance(rows, list) else []:
        if not isinstance(r, dict):
            rejected += 1
            continue
        ax, subj = str(r.get("axis") or ""), str(r.get("subject") or "").strip()
        q = " ".join(str(r.get("quote") or "").split())
        try:
            val = int(str(r.get("value")).strip())
        except Exception:
            rejected += 1
            continue
        if ax not in AXES or not subj or val <= 0 or len(q) < 12 or q not in hay:
            rejected += 1                     # 축 밖 · 인용 부재 = 버린다
            continue
        out.append({"applies_to": {"axis_default": True},
                    "subject": subj, "axis": ax, "value": val,
                    "against": AXES[ax]["against"], "compare": AXES[ax]["compare"],
                    "when": [],
                    "source": {"doc": doc_id, "quote": q}})
    return out, rejected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default="http://localhost:8140/v1")
    ap.add_argument("--model", default="Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8")
    ap.add_argument("--limit", type=int, default=0, help="스모크용 문서 수 제한")
    ap.add_argument("--save-every", type=int, default=20)
    a = ap.parse_args()

    import tau2.agent.llm_agent as la
    from tau2.data_model.message import UserMessage as UM
    agent = Agent(a.model, a.base)

    docs = sorted(glob.glob(os.path.join(a.docs, "*.json")))
    if a.limit:
        docs = docs[:a.limit]

    done = {}
    if os.path.exists(a.out):                 # ★재개 — 이미 읽은 문서는 건너뛴다
        prev = json.load(io.open(a.out, encoding="utf-8"))
        done = {d: True for d in prev.get("docs_done", [])}
        rows = list(prev.get("rows", []))
        stats = collections.Counter(prev.get("stats", {}))
        print("재개: 문서 %d개 완료 · 행 %d" % (len(done), len(rows)))
    else:
        rows, stats = [], collections.Counter()

    def save():
        io.open(a.out, "w", encoding="utf-8").write(json.dumps(
            {"axes": {k: v["desc"] for k, v in AXES.items()},
             "docs_total": len(docs), "docs_done": sorted(done),
             "rows": rows, "stats": dict(stats)}, ensure_ascii=False, indent=1))

    for i, p in enumerate(docs, 1):
        d = json.load(io.open(p, encoding="utf-8"))
        did = d.get("id") or os.path.basename(p)
        if did in done:
            continue
        try:
            raw = ask(agent, la, UM, str(d.get("content") or ""))
        except Exception as e:
            stats["호출 실패"] += 1
            print("  ⚠%s 호출 실패 %r" % (did[-28:], e), file=sys.stderr)
            continue
        got, rej = parse(raw, d.get("content") or "", did)
        rows.extend(got)
        done[did] = True
        stats["문서"] += 1
        stats["채택"] += len(got)
        stats["거절"] += rej
        if got:
            print("  [%3d/%d] %-30s +%d행" % (i, len(docs), did[-30:], len(got)), flush=True)
        if len(done) % a.save_every == 0:
            save()

    save()

    # ── 병합·충돌 ────────────────────────────────────────────────────────────
    by_key = collections.defaultdict(list)
    for r in rows:
        by_key[(r["subject"], r["axis"])].append(r)
    conflicts = {k: v for k, v in by_key.items() if len({x["value"] for x in v}) > 1}

    print("\n" + "=" * 92)
    print("문서 %d/%d · 채택 행 %d · 거절 %d · 호출 실패 %d"
          % (stats["문서"], len(docs), stats["채택"], stats["거절"], stats["호출 실패"]))
    print("서로 다른 (주어, 축) = **%d개** · 그중 **값 충돌 %d개**" % (len(by_key), len(conflicts)))
    for ax in AXES:
        n = len({k for k in by_key if k[1] == ax})
        print("   %-24s 주어 %d개" % (ax, n))
    if conflicts:
        print("\n★값 충돌 — **자동으로 고르지 않는다**(사람이 판정·[[25]]):")
        for (subj, ax), v in sorted(conflicts.items())[:10]:
            print("   %s / %s" % (subj, ax))
            for x in v:
                print("      %3d  [%s] %s" % (x["value"], x["source"]["doc"][-24:],
                                              x["source"]["quote"][:80]))
    print("\n저장: %s" % a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
