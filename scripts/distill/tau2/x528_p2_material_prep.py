# -*- coding: utf-8 -*-
r"""x528 — P2(저작·배선)를 아침에 바로 시작할 수 있게 **재료만 깔아 둔다** (2026-08-25·무료·CPU)

## 왜 (사용자 물음: *"단계별로 계획했던 실험을 모두 하는건가?"* → 아니다)
밤 배치는 **측정**뿐이다. 큐 P2 의 두 칸이 비어 있고 그 둘은 **저작**이라 무인 실행이 위험하다
([[23]]: 노드마다 정책 축자 출처를 못 대면 넣으면 안 된다). 그래서 저작은 아침으로 남기고,
**저작에 필요한 축자를 지금 뽑아 둔다** — 판단 0 · 선택 0 · LLM 0.

## 무엇을 뽑나
  ⑴ **016 자격 조건 축자** — A2 `procedures` 저작의 `_note_` 출처가 될 문장들.
     `### Qualification criteria` 절을 **그 sim 이 실제로 받은 도구 출력**에서 그대로 뜬다.
     출처 표기는 `런 · seed · 메시지 index` 로 한다 — 문서 id 를 코드가 만들지 않는다([[71]]).
  ⑵ **057·063 배달 census** — `T2_` 마커가 그 궤적에서 실제로 어떻게 찍혔나.
     큐 P1 ②범주 레인이 요구한 관측이고, 표 가설이 죽으면 남는 갈래가 이것이다.

## 규율
  · gold 미접촉([[23]]) · 새 분류 0 · 판정 문장 0 — **원자료만 모은다**
  · 부재는 **검색 범위와 함께** 적는다(§77-b)

사용: py -3 x528_p2_material_prep.py
"""
import gzip
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
REP = os.path.normpath(os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026"))
SIMS = os.path.join(REP, "sim_results")
RUNS = ("bank_t7348_halfA_20260824", "bank_t7348_halfB_20260824",
        "bank_t7346_halfA_20260822", "bank_t7346_halfB_20260822")
NEEDLE = "Qualification criteria"


def sims_of(task):
    out = []
    for tag in RUNS:
        p = os.path.join(SIMS, tag + ".results.json.gz")
        if not os.path.exists(p):
            continue
        d = json.load(gzip.open(p, "rt", encoding="utf-8", errors="replace"))
        for s in (d.get("simulations") or []):
            if s.get("task_id") == task:
                out.append((tag, s))
    return out


def qualification_quotes():
    """016 자격 조건 축자 — 절 단위로 그대로. 출처 = 런·seed·메시지 index."""
    hits, searched = [], []
    for tag, s in sims_of("task_016"):
        searched.append("%s#%s" % (tag, s.get("seed")))
        for i, m in enumerate(s.get("messages") or []):
            if m.get("role") != "tool":
                continue
            c = str(m.get("content") or "")
            j = c.find(NEEDLE)
            if j < 0:
                continue
            hits.append({"run": tag, "seed": s.get("seed"), "msg_index": i,
                         "verbatim": c[max(0, j - 200):j + 1200]})
    return hits, searched


def delivery_census(task):
    """마커 census — 그 태스크 줄만. 판정 없이 센다."""
    pat = re.compile(r"\[(T2_[A-Z0-9_]+)\]")
    per_run, searched = {}, []
    for tag in RUNS:
        lp = os.path.join(SIMS, tag + ".log.gz")
        if not os.path.exists(lp):
            continue
        searched.append(lp)
        counts, samples = {}, []
        with gzip.open(lp, "rt", encoding="utf-8", errors="replace") as f:
            for ln in f:
                if ("task_%s#" % task) not in ln:
                    continue
                for mk in pat.findall(ln):
                    counts[mk] = counts.get(mk, 0) + 1
                if "clobber" in ln.lower() or "DECIDE" in ln:
                    if len(samples) < 12:
                        samples.append(ln.strip()[:300])
        if counts:
            per_run[tag] = {"markers": dict(sorted(counts.items(), key=lambda x: -x[1])),
                            "decide_or_clobber_lines": samples}
    return per_run, searched


def main():
    q, q_searched = qualification_quotes()
    d057, d_searched = delivery_census("057")
    d063, _ = delivery_census("063")
    out = {
        "probe": "x528", "date": "2026-08-25",
        "purpose": "P2 저작을 아침에 바로 시작하기 위한 재료 — 판정 0 · 저작 0",
        "016_qualification_quotes": {
            "n": len(q), "searched_sims": q_searched, "hits": q,
            "note": "A2 `procedures` 노드의 `_note_` 출처로 그대로 쓸 수 있는 축자([[23]] 통과)."},
        "057_delivery_census": {"per_run": d057, "searched_logs": d_searched},
        "063_delivery_census": {"per_run": d063},
    }
    p = os.path.join(REP, "x528_p2_material_2026_08_25.json")
    with io.open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("[x528] 016 자격조건 축자 %d건 (검색 sim %d)" % (len(q), len(q_searched)))
    for tag, v in d057.items():
        print("[x528] 057 %-34s 마커 상위 %s · DECIDE/clobber 줄 %d"
              % (tag, list(v["markers"].items())[:6], len(v["decide_or_clobber_lines"])))
    for tag, v in d063.items():
        print("[x528] 063 %-34s 마커 상위 %s" % (tag, list(v["markers"].items())[:6]))
    print("[x528] → %s" % p)


if __name__ == "__main__":
    main()
