# -*- coding: utf-8 -*-
r"""x328 — **"레코드는 맞고 이름만 틀린"** 실패의 전수 census.

## 왜

t7295 DB diff 를 읽다 보면 같은 모양이 계속 나온다 — gold 와 우리 레코드가 **한 필드만** 다르고
나머지(고객·소득·상태·날짜·잔액)는 **완전히 같다**:

    071  gold `Sky Blue` / `Gold Saver Account`      ↔ 우리 `Lime Green` / `Bronze Saver Account`
    003  gold `Silver Rewards Card`                  ↔ 우리 `Platinum` / `Gold Rewards Card`
    099  gold `World Blue Account`                   ↔ 우리 `Navy Blue Account`
    069  gold `Blue` / `Silver Plus` / `Silver Rewards` ↔ 우리 `Bluest` / `Gold` / `Business Platinum`

⇒ 도달·형식·집행은 다 됐고 **고른 상품 이름 하나**로 0 이 된다. 이 축이 얼마나 큰지 세지 않고는
우선순위를 못 정한다([[08]]: 집계에서 결론 직행 금지 — 그래서 **sim 목록까지** 찍는다).

## 무엇을 세나 (기계적·도메인 판단 0)

gold 에만 있는 레코드 G 와 우리에게만 있는 레코드 P 를 같은 테이블에서 짝짓되,
**공통 키 중 정확히 한 칸만 다르고 나머지가 전부 같은** 짝만 센다(자동생성 id 는 제외).
그 한 칸의 **필드 이름**으로 분류한다 — 무엇이 옳은 상품인지는 판정하지 않는다([[59]]).

    WRONG-PRODUCT   다른 칸이 상품/등급 이름 계열      (`*_class`·`card_type`·`*_type`)
    WRONG-ENUM      그 밖의 값 한 칸                   (예: `closure_reason` 표기)

⚠짝이 안 지어지는 미이행(ONLY-GOLD 단독)은 **여기서 세지 않는다** — 그것은 COVERAGE 축이다.

사용(리모트): PYTHONPATH=tau2-bench/src seka python x328_wrong_product_census.py [tag ...]
"""
import collections
import io
import sys

from loguru import logger

from tau2.registry import registry
from tau2.data_model.simulation import Results

try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
except Exception:
    pass
logger.remove()

SIMROOT = "/home/woori/scratch/tau2-bench/data/simulations"
DOMAIN = "banking_knowledge"
# 자동생성 식별자 — 값이 다른 것이 당연하므로 비교에서 뺀다(도메인 어휘 아님·키 이름 규약).
IDLIKE = ("_id", "id")
PRODUCT = ("_class", "card_type", "account_type", "referred_account_type", "_tier", "_level")


def is_id(k):
    return k.endswith(IDLIKE[0]) or k == IDLIKE[1] or k.endswith("_ids")


def is_product(k):
    return any(p in k for p in PRODUCT)


def rows_of(db):
    """테이블명 → {레코드키: dict}."""
    out = {}
    for t, v in (db or {}).items():
        if isinstance(v, dict) and isinstance(v.get("data"), dict):
            out[t] = v["data"]
    return out


def pair(gold_rows, pred_rows):
    """공통 키 중 **정확히 한 칸**만 다른 (G,P) 짝 → [(테이블, 필드, gold값, pred값)]."""
    hits = []
    for t, g in gold_rows.items():
        p = pred_rows.get(t) or {}
        gonly = {k: v for k, v in g.items() if k not in p}
        ponly = {k: v for k, v in p.items() if k not in g}
        for _gk, gv in gonly.items():
            if not isinstance(gv, dict):
                continue
            for _pk, pv in ponly.items():
                if not isinstance(pv, dict):
                    continue
                keys = [k for k in set(gv) | set(pv) if not is_id(k)]
                diff = [k for k in keys if gv.get(k) != pv.get(k)]
                if len(diff) == 1:
                    hits.append((t, diff[0], gv.get(diff[0]), pv.get(diff[0])))
    return hits


def main(tags):
    env_ctor = registry.get_env_constructor(DOMAIN)
    tasks = {t.id: t for t in registry.get_tasks_loader(DOMAIN)()}
    kind = collections.Counter()
    detail = []
    nfail = 0
    for tag in tags:
        results = Results.load(__import__("pathlib").Path("%s/%s/results.json" % (SIMROOT, tag)))
        for sim in results.simulations:
            ri = sim.reward_info
            if ri is None or ri.reward == 1.0:
                continue
            nfail += 1
            task = tasks.get(sim.task_id)
            if task is None:
                continue
            ist = task.initial_state
            def build(msgs, acts=None):
                e = env_ctor(retrieval_variant="no_knowledge")
                e.set_state(ist.initialization_data if ist else None,
                            ist.initialization_actions if ist else None, list(msgs))
                for a in (acts or []):
                    try:
                        e.make_tool_call(tool_name=a.name, requestor=a.requestor, **a.arguments)
                    except Exception:
                        pass
                return e
            try:
                gold = build(list((ist.message_history if ist else None) or []),
                             task.evaluation_criteria.actions or [])
                pred = build(list(sim.messages))
            except Exception:
                continue
            hits = pair(rows_of(gold.tools.db.model_dump()), rows_of(pred.tools.db.model_dump()))
            for t, f, gv, pv in hits:
                k = "WRONG-PRODUCT" if is_product(f) else "WRONG-ENUM"
                kind[k] += 1
                detail.append((sim.task_id, k, t, f, gv, pv))
    print("실패 sim %d · 한 칸만 다른 짝 %d" % (nfail, sum(kind.values())))
    for k, v in kind.most_common():
        print("   %-14s %d" % (k, v))
    print("\n건별(태스크·종류·테이블·필드·gold ↔ ours):")
    for d in sorted(detail):
        print("   %-10s %-14s %-26s %-22s %r ↔ %r" % d)
    sims = len({d[0] for d in detail})
    print("\n관련 태스크 %d개 · 이 축이 닿는 실패 %d건" % (sims, len(detail)))


if __name__ == "__main__":
    main(sys.argv[1:] or ["bank_t7295_a_20260815n", "bank_t7295_b_20260815n"])
