# -*- coding: utf-8 -*-
"""x145 — 온톨로지 행에 **`applies_to` 결속을 찍고 도구별 인덱스**를 낸다 (유료 0 · LLM 0).

정본 = `A3_POLICY_ONTOLOGY_DESIGN_2026_08_08.md` §1c.
§1c가 요구한 것: 런타임의 질문은 *"World Blue의 문턱은?"* 이 아니라
**"지금 `submit_referral`을 하려는데 걸리는 정책 규칙이 전부 무엇인가?"** 다. 그래서 행이
**소비자를 함께 진다**. 그러면 조회가 검색이 아니라 **인덱스 조회**가 된다.

결속의 출처([[23]]): 축마다 `x140.AXES[axis]["applies_to"]["basis"]`에 **env 시그니처 축자 또는
정책 축자**를 적어 두었다. 결속은 **저작 판단**이라 근거를 못 대면 넣지 않는다.

⚠**§1c의 형태에서 하나 벗어난다**: 설계는 `{"tool": …, "operand_arg": …}` 단수인데 여기서는
`consumers: [ … ]` **목록**이다. 한 축이 두 결정점에서 소비되는 경우가 실재하기 때문이다
(`holder_min_age_years` = 계좌 개설 자격이면서 추천 자격). 단수로 두면 **둘 중 하나를 버리는
거짓 선택**이 된다.
⚠**빈 `consumers`는 결함이 아니라 사실이다** — 보너스 금액·예치 하한처럼 *권고를 만들 때 쓰는
피연산자*는 도구 경계에서 판정되지 않는다. 억지로 붙이면 **없는 게이트를 지어내는 것**이다.
⚠주어 해소(인자 값 ↔ `subject`)는 **여전히 모델 몫**이다(§1c 말미·C316). 인덱스는 후보를 좁힐 뿐.
⚠**분석·빌드 도구이지 런타임 엔진이 아니다**([[59]]).

usage: x145_bind_applies_to.py --ontology <in.json[.gz]> [--out <out.json>] [--by-tool]
"""

import argparse
import collections
import gzip
import io
import json
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from x140_build_policy_ontology import AXES          # noqa: E402  — 결속표의 유일한 집


def load(path):
    op = gzip.open if str(path).endswith(".gz") else io.open
    with op(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def stamp(onto):
    """행마다 그 축의 결속을 찍는다. 축 밖 행은 남기되 결속 없음으로 표시한다."""
    unknown = collections.Counter()
    for r in onto.get("rows") or []:
        ax = AXES.get(r.get("axis"))
        if not ax:
            unknown[r.get("axis")] += 1
            r["applies_to"] = {"consumers": [], "basis": "축 선언에 없음"}
            continue
        r["applies_to"] = ax["applies_to"]
    return unknown


def by_tool(onto):
    idx = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in onto.get("rows") or []:
        for c in ((r.get("applies_to") or {}).get("consumers") or []):
            idx[c["tool"]][r["axis"]].append((r["subject"], r["value"], c["operand_arg"]))
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ontology", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--by-tool", action="store_true")
    a = ap.parse_args()

    onto = load(a.ontology)
    unknown = stamp(onto)
    rows = onto.get("rows") or []
    bound = sum(1 for r in rows if (r["applies_to"].get("consumers")))
    print("행 %d · 결속된 행 %d · 결속 없는 행 %d" % (len(rows), bound, len(rows) - bound))
    for ax in AXES:
        cs = [c["tool"] for c in AXES[ax]["applies_to"]["consumers"]]
        n = sum(1 for r in rows if r.get("axis") == ax)
        print("   %-34s %-42s 행 %d" % (ax, ", ".join(cs) or "(도구 경계 아님)", n))
    if unknown:
        print("⚠축 선언 밖의 행: %s" % dict(unknown))

    idx = by_tool(onto)
    if a.by_tool:
        for tool in sorted(idx):
            print("\n" + "=" * 88)
            print("by_tool[%s] — 이 도구에 걸리는 정책 규칙 전부" % tool)
            for ax in sorted(idx[tool]):
                items = sorted(idx[tool][ax])
                print("  %-34s 주어 %d (인자=%s)" % (ax, len(items), items[0][2]))
                for subj, val, _ in items[:6]:
                    print("      %-34s %s" % (subj[:34], val))
                if len(items) > 6:
                    print("      … %d개 더" % (len(items) - 6))

    if a.out:
        onto["applies_to_bound"] = True
        io.open(a.out, "w", encoding="utf-8").write(json.dumps(onto, ensure_ascii=False, indent=1))
        print("\n저장: %s" % a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
