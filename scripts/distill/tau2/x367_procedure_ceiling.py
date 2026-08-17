# -*- coding: utf-8 -*-
r"""x367 — **절차 DAG 축의 상한**: 정책 문서가 순서를 주는 태스크는 몇 개인가.

## 왜 (사용자 제안 2026-08-18 축자)

*"이건 DAG 로 e-plan 으로 A2 A3 로 기술하고 엔진이 프로세스를 따라가게 하면 안 되나? LLM 이
프로세스를 선택하게 하고, 구체적 프로세스는 결정론으로 하는 거다."*

그 기계는 **이미 있다**(A2 `procedures` 6개 · `t2_procedure.py` · `t2_gate_patch.py:6253~6332` ·
`T2_PROCEDURE=1` · 라이브 발화 t7295 30회·97태스크 런 26회). 문제는 **채울 수 있는 칸이 몇 개냐**다.

⛔[[23]]: DAG 노드의 출처는 **정책 문서 축자**여야 한다. gold 를 보고 순서를 쓰면 그건 gold
프로그램 재작성이고 실험이 무효다. 그러므로 이 축의 상한 = **정책이 순서를 주는 태스크 수**다.

낱말 grep(큐 10종)으로는 698 문서 중 15개만 걸렸는데, 그 결과는 **하한이 확실하다** — A2 의
`cash_back_dispute` 절차가 출처로 대는 `doc_credit_cards_credit_cards_(general)_003` 이 그 grep 에
안 걸린다. 순서를 다른 말로 주는 문서가 더 있다는 뜻이다 ⇒ **문서를 읽어야 한다**([[59]]: 읽기는
LLM 몫·엔진은 인용 실재만 검산).

## 셀 (문서마다·det n=1)

    Q_ORDER   "이 문서는 상담원이 따라야 할 **단계·순서**를 지시하는가? YES/NO +
               그렇게 만드는 문장을 원문에서 축자 인용" → 엔진은 `quote_in` 만
    D_NEG     같은 질문을 **다른 문서 본문**에 걸어 만든 대조(무관 문서가 YES 나오면 계기 무효)

## 판정 (사전 고정)

    YES 문서를 required_documents 로 갖는 **C 버킷 태스크 수** = 이 축의 상한
    상한 ≥ 20  → DAG 축을 S2 스모크에 태울 값이 있다
    상한 < 10  → 이 축은 소수 태스크용이고 43 가족은 **정책이 안 준다** ⇒ 다른 축으로
    D_NEG YES 비율 ≥ 1/3 → 계기 무효(판정기가 아무 문서에나 YES)

실행: T2_PROBE_URL=http://localhost:8141/v1/chat/completions \
      /home/woori/venvs/seka_env/bin/python x367_procedure_ceiling.py [part] [nparts]
"""
import collections
import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/home/woori/scratch/tau2-bench/src")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import t2_search as TS                                            # noqa: E402
import x351_order_lever_iso as X                                  # noqa: E402
import x357_verdict_carry_multitask as M                          # noqa: E402
import x364_eligibility_axis_iso as E                             # noqa: E402

ASK = ("Below is one internal policy document from a bank's knowledge base.\n\n{doc}\n\n"
       "Does this document tell the support agent a REQUIRED ORDER of steps or tool calls that "
       "must be followed? Answer on the first line with YES or NO. If YES, quote VERBATIM on the "
       "second line the sentence that imposes the order. Nothing else.")


def main():
    part = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    nparts = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    docs = {}
    for fn in sorted(os.listdir(X.DOCS)):
        d = json.load(io.open(os.path.join(X.DOCS, fn), encoding="utf-8"))
        docs[str(d.get("id") or fn)] = str(d.get("content") or "")
    keys = [k for i, k in enumerate(sorted(docs)) if i % nparts == part]
    cen = json.load(io.open(os.path.join(E.REPORTS, "x365_misselect_census.json"),
                            encoding="utf-8"))
    cbucket = set(r["task"] for r in cen if not r["axes"] and not r["excluded"])
    need = collections.defaultdict(list)
    for fn in sorted(os.listdir(M.TASKS_DIR)):
        if not fn.endswith(".json"):
            continue
        t = json.load(io.open(os.path.join(M.TASKS_DIR, fn), encoding="utf-8"))
        for rd in (t.get("required_documents") or ()):
            rid = rd if isinstance(rd, str) else str((rd or {}).get("id") or rd)
            need[rid].append(fn[:-5])
    print("x367 · 조각 %d/%d · 문서 %d개 · C버킷 %d개" % (part, nparts, len(keys), len(cbucket)))
    print("판정(사전 고정): YES 문서를 required_documents 로 갖는 C버킷 태스크 수 = 축의 상한 · "
          "≥20 → S2 에 태운다 · <10 → 43 가족엔 정책이 없다 ⇒ 다른 축 · D_NEG YES ≥1/3 → 계기 무효\n")

    res, neg_yes = [], 0
    others = sorted(docs)
    for n, k in enumerate(keys):
        body = docs[k]
        ans, det = E.det_ask(ASK.format(doc=body[:6000]), 200)
        head = " ".join(str(ans or "").split()).upper()
        yes = head.startswith("YES")
        q = ""
        for l in [x.strip().strip('"').strip() for x in str(ans or "").split("\n") if x.strip()][1:]:
            if TS.quote_in(l, body):
                q = l
                break
        # 부정통제: 같은 질문을 **다른 문서**에 (10건마다 1회·비용 통제)
        if n % 10 == 0:
            alt = others[(others.index(k) + len(others) // 3) % len(others)]
            a2, _d = E.det_ask(ASK.format(doc=docs[alt][:6000]), 200)
            neg_yes += 1 if " ".join(str(a2 or "").split()).upper().startswith("YES") else 0
        res.append({"doc": k, "yes": int(yes), "cited": int(bool(q)), "q": q[:200], "det": det,
                    "tasks": need.get(k, [])})
        if yes:
            print("   YES %-58s 태스크 %s" % (k[:58], ",".join(x[5:] for x in need.get(k, []))
                                             or "-"))

    ys = [r for r in res if r["yes"] and r["cited"]]
    tset = set(t for r in ys for t in r["tasks"])
    print("\n" + "=" * 96)
    print("문서 %d · YES %d · YES∧인용검산 %d · 그 문서를 요구하는 태스크 %d · **C버킷 교집합 %d**"
          % (len(res), sum(r["yes"] for r in res), len(ys), len(tset), len(tset & cbucket)))
    print("C버킷 교집합: %s" % ", ".join(sorted(x[5:] for x in (tset & cbucket))))
    print("부정통제(무관 문서) YES %d/%d" % (neg_yes, (len(keys) + 9) // 10))
    out = os.path.join(E.REPORTS, "x367_part%d.json" % part)
    with io.open(out, "w", encoding="utf-8") as f:
        f.write(json.dumps(res, ensure_ascii=False, indent=1, default=str))
    print("저장: %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
