# -*- coding: utf-8 -*-
r"""x461 — 생성된 읽기 명세를 **A3 두 층에 쓴다** (2026-08-21·오프라인·LLM 0)

## 무엇을
`x457` 이 만든 `sub_docs_block` 을 `scaffold_get_tools[<name>].isolate.docs` 로 넣는다.
[[24]]: **정본(`specific.json`)과 `gate.json` 을 함께** 고치고 바이트 동일로 맞춘다 — 한쪽만
고치면 死코드거나 등가게이트 FAIL 이다(2026-08-03 실증).

## 무엇이 들어가나
    axes      이 서브가 알아야 할 축 — 출처는 그 도구의 `op.reducers` 범주와
              `doc_045`/`doc_046` 이 정의한 범주다([[23]] gold 미참조)
    always    범주 **정의**를 담은 공용 정책 문서 3편
    by_class  클래스 → 그 축을 말한 문서 id (전달 단위 = **문서**)
`spans`(문서 안 오프셋)는 A3 에 싣지 않는다 — 감사 산출물(`x453_attr_coverage_all.json` 의
`cites`)에 있고, 그쪽이 근거 앵커의 정본이다. A3 는 **무엇을 읽을지**만 말한다([[71]]).

## 왜 절이 아니라 문서인가 (실측)
절로 자르면 감사가 축으로 잡지 못한 줄이 사라진다 — `green_account_(checking)_001` 의
*"Boost a linked savings account's APY: Gold +0.75% or Silver +0.25%"* 가 **어떤 span 에도
안 들어가고**, 093·094 가 둘 다 그 값을 필요로 한다. 문서 중앙 989자라 비용도 비슷하다.

사용: py x461_write_a3_subdocs.py --tool get_correct_savings_apy [--dry]
"""
import argparse
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

REP = os.path.abspath(os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026"))
LAYERS = ("banking_knowledge.specific.json", "banking_knowledge.gate.json")

NOTE = (
    "★출처(2026-08-21·[[23]]): 축 목록 = 이 도구의 `op.reducers` 범주(base·checking·card·"
    "relationship·tier) + 공용 정책이 정의한 범주. `doc_..._045` 축자 \"Relationship bonuses "
    "for holding multiple Rho-Bank products\" 가 relationship 을, 같은 문서가 카드 max1 을, "
    "`doc_..._046` 이 checking max1 을, `doc_..._012` 가 어느 페어링이 자격인지를 말한다 — "
    "그 셋이 없으면 kind 분류가 불가능하므로 always 다. || by_class = 코퍼스 전수 감사"
    "(`x453`·698편)에서 **그 축의 값을 축자 인용으로 검산한 문서**만. gold 미참조. || "
    "전달 단위는 **문서**다: 절로 자르면 감사가 축으로 안 잡은 줄이 사라진다(실측 — "
    "green checking 의 'Gold +0.75% or Silver +0.25%' 가 어떤 span 에도 없다). 문서 안 "
    "오프셋 앵커는 `x453_attr_coverage_all.json` 의 `cites` 에 있다. || 엔진은 선언된 id 를 "
    "읽어 넘기기만 하고 **검색하지 않는다**([[71]] · bm25 는 baseline)."
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tool", default="get_correct_savings_apy")
    ap.add_argument("--blocks", default="x457_a3_blocks.json")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()

    with io.open(os.path.join(REP, a.blocks), encoding="utf-8") as f:
        blk = (json.load(f) or {}).get("sub_docs_block")
    if not blk:
        raise SystemExit("sub_docs_block 이 없다 — x457 을 --sub-axes 와 함께 돌려라")
    docs = {"_note_": NOTE, "axes": blk["axes"], "always": blk["always"],
            "by_class": blk["by_class"]}
    n_docs = sum(len(v) for v in docs["by_class"].values())
    print("넣을 것: 축 %d · 공용 %d편 · 클래스 %d · 문서 %d"
          % (len(docs["axes"]), len(docs["always"]), len(docs["by_class"]), n_docs))

    wrote = []
    for layer in LAYERS:
        p = os.path.join(HERE, "a2", layer)
        with io.open(p, encoding="utf-8") as f:
            d = json.load(f)
        tools = d.get("scaffold_get_tools") or []
        hit = [t for t in tools if t.get("name") == a.tool]
        if not hit:
            raise SystemExit("%s 에 %s 선언이 없다" % (layer, a.tool))
        iso = hit[0].get("isolate")
        if not isinstance(iso, dict):
            raise SystemExit("%s 의 %s 에 isolate 가 없다" % (layer, a.tool))
        before = json.dumps(iso.get("docs"), ensure_ascii=False, sort_keys=True)
        iso["docs"] = docs
        after = json.dumps(iso.get("docs"), ensure_ascii=False, sort_keys=True)
        print("  %-34s docs: %s → %d자" % (layer, "없음" if before == "null" else "%d자" % len(before),
                                           len(after)))
        if not a.dry:
            with io.open(p, "w", encoding="utf-8") as f:
                json.dump(d, f, ensure_ascii=False, indent=1)
            wrote.append(p)

    # ── 두 층 동일성 검산 (바이트가 아니라 **그 서브트리**가 같은지·[[24]]) ──
    got = []
    for layer in LAYERS:
        with io.open(os.path.join(HERE, "a2", layer), encoding="utf-8") as f:
            d = json.load(f)
        t = next(x for x in (d.get("scaffold_get_tools") or []) if x.get("name") == a.tool)
        got.append(json.dumps((t.get("isolate") or {}).get("docs"),
                              ensure_ascii=False, sort_keys=True))
    print("두 층 `docs` 동일: %s" % ("예" if got[0] == got[1] else "★아니오"))
    if got[0] != got[1]:
        raise SystemExit("두 층이 갈렸다 — 중단([[24]])")
    print("쓴 파일: %s" % (", ".join(os.path.basename(x) for x in wrote) if wrote else "(dry)"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
