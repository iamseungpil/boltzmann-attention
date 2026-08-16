# -*- coding: utf-8 -*-
r"""bank_pmap — 97 태스크를 **P1~P6 레버 축으로 기계 분류**(2026-08-16·사용자 지시).

## 핵심 술어: **그 값은 어디서 와야 하는가**

느슨한 1차 규칙(P3 87/97·P6 85/97)은 표적을 못 좁혔다. 훨씬 날카로운 닫힌 술어가 있다 —

    gold 액션의 인자 값이 **손님 발화에 축자로 있는가?**
      · 있다        → 받아 적으면 된다(축 없음)
      · 없다 + id 꼴 → **조회해서 전사**해야 한다      ⇒ **P3**(출처·형식)
      · 없다 + 수    → **계산**해야 한다               ⇒ **P6**
      · 없다 + 이름  → **문서에서** 와야 한다          ⇒ **P1**(전달)

여기에 형태 축 둘을 더한다:

    같은 gold 도구를 3회 이상 반복  ⇒ **P5**(열거 완결)
    `fail_wrote == 0` 이고 0%       ⇒ **P4**(방출·학습 축)
    유효창(프로모션) 낱말이 있다     ⇒ **P2**(만료 제거)

## 이것이 무엇이고 무엇이 아닌가

**분석 도구다. 레버가 아니다.** gold 를 읽지만 그것은 *표적 우선순위를 정하기 위한 집계*이고,
여기서 나온 어떤 값도 엔진·A2 에 들어가지 않는다([[23]] — gold 참조 금지는 **레버**에 대한 규율).
⚠그리고 이것은 여전히 **사전 지도**다. 실제 병목은 궤적이 있어야 갈린다 — 오늘 실물로 둘 다
겪었다: 069(자격인 줄 알았으나 **벤치 결함**) · 024(지도는 P3+P6, 궤적은 **자기-정박**).
⇒ 표적 확정 전에 **궤적·문서 대조 필수**([[62]]).
"""
import collections
import io
import json
import os
import re
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
CENSUS = os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026",
                      "bank_task_taxonomy_20260810.json")
TASKS = "/home/woori/scratch/tau2-bench/data/tau2/domains/banking_knowledge/tasks.json"
ID = re.compile(r"^[a-z0-9]{8,}$|^txn_|^acct|_\d{4}$")
NUM = re.compile(r"^\$?[\d,]+(\.\d+)?$|points$|%$")
WINDOW = re.compile(r"promo|promotion|expire|valid until|active from", re.I)


def norm(x):
    return re.sub(r"[\s,$]", "", str(x)).lower()


def values_of(action):
    """gold 인자에서 **잎 값**만 뽑는다(중첩 `arguments` 문자열도 푼다)."""
    out = []

    def walk(v):
        if isinstance(v, dict):
            for x in v.values():
                walk(x)
        elif isinstance(v, list):
            for x in v:
                walk(x)
        elif isinstance(v, str):
            s = v.strip()
            if s.startswith("{") and s.endswith("}"):
                try:
                    walk(json.loads(s))
                    return
                except Exception:
                    pass
            out.append(s)
        elif isinstance(v, (int, float)):
            out.append(str(v))
    walk(action.get("arguments") or {})
    return [v for v in out if v and len(v) > 1]


def classify(t, task, scen):
    ps = set()
    acts = t.get("acts") or []
    rep = collections.Counter(acts)
    if rep and max(rep.values()) >= 3:
        ps.add("P5")
    if t.get("n", 0) and t.get("fail_wrote", 0) == 0 and t.get("rate", 0) == 0:
        ps.add("P4")
    if WINDOW.search(scen):
        ps.add("P2")

    sn = norm(scen)
    gold = ((task or {}).get("evaluation_criteria") or {}).get("actions") or []
    for a in gold:
        for v in values_of(a):
            if norm(v) in sn:          # 손님이 말해 준 값 = 축 없음
                continue
            if ID.search(v.lower()):
                ps.add("P3")
            elif NUM.search(v):
                ps.add("P6")
            elif " " in v or v[:1].isupper():
                ps.add("P1")
    return sorted(ps) or ["미분류"]


def main():
    d = json.load(io.open(CENSUS, encoding="utf-8"))
    T, S = {}, {}
    if os.path.exists(TASKS):
        raw = json.load(io.open(TASKS, encoding="utf-8"))
        for t in (raw if isinstance(raw, list) else raw.get("tasks", [])):
            T[str(t.get("id"))] = t
            S[str(t.get("id"))] = json.dumps(t.get("user_scenario") or {}, ensure_ascii=False)

    rows, cnt, multi = [], collections.Counter(), collections.Counter()
    for t in d:
        ps = classify(t, T.get(t["id"]), S.get(t["id"], ""))
        rows.append((t["id"], t["rate"], t["n"], t.get("fail_wrote", 0), ps))
        for x in ps:
            cnt[x] += 1
        multi[len(ps)] += 1

    print("97 태스크 P-지도 (사전 지도·귀속 아님 · 태스크 정의 %d/97)\n" % len(T))
    print("축별 태스크 수:", dict(cnt.most_common()))
    print("축 개수 분포:", dict(sorted(multi.items())))

    print("\n== 0% 태스크(59)의 축 조합 ==")
    for c, k in collections.Counter("+".join(ps) for i, r, n, w, ps in rows
                                    if r == 0).most_common(10):
        print("  %-24s %d" % (c, k))

    print("\n== 표적 2·3 ==")
    for tid in ("task_019", "task_020", "task_022", "task_027", "task_028", "task_029", "task_024"):
        for i, r, n, w, ps in rows:
            if i == tid:
                print("  %s rate=%2.0f%% → %s" % (i, 100 * r, "+".join(ps)))

    print("\n== 축별 0% 명부 ==")
    for ax in ("P1", "P2", "P3", "P4", "P5", "P6"):
        ids = [i.replace("task_", "") for i, r, n, w, ps in rows if r == 0 and ax in ps]
        print("  %-3s %2d: %s" % (ax, len(ids), " ".join(ids)))

    print("\n== 단일 축 태스크(가장 깨끗한 표적) ==")
    for i, r, n, w, ps in sorted(rows, key=lambda x: x[1]):
        if len(ps) == 1 and ps != ["미분류"]:
            print("  %s rate=%2.0f%% n=%d → %s" % (i, 100 * r, n, ps[0]))


if __name__ == "__main__":
    main()
