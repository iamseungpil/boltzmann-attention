#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""★read-path vs write-path 분리 계측 (무료·오프라인·전수·[[08]])

**왜 재나** (`NABAOS_PREEMPTION_AUDIT_2026_07_18 §2.1`): NabaOS(`2603.10060`)의 threat model은
**read-path서 끝난다** — 전부 *"사용자에게 거짓을 말했는가"*로 종결한다(`724-751`·Stage 6 =
*"Trust-Annotated Output"*). ⇒ **지어낸 인자로 WRITE가 실행됐는데 그 값을 사용자에게 보고하지 않으면**,
그들 영수증은 **충실히** 기록되고(`input_hash`), **아무도 거짓을 듣지 않으며**, 피해는 **세상에** 남는다
= **그들에게 없는 표면**. 감사 §2.1은 *"우리 `record` 46%가 정확히 거기 산다"*고 **주장**했는데
**계측된 적이 없다**. 그 주장이 참인지 거짓인지 이 스크립트가 정한다.

**판정 기준 = 엔진 정본 재사용** ([[03b]] — 새 기준을 지어내면 그 자체로 실험 무효):
  · **날조** = `t2_gate_patch._first_fab_call` **그대로**. ctx = `_ctx_from_messages` = **user+tool 텍스트만**
    (assistant 제외) ⇒ 날조 = *사용자도 도구도 말한 적 없는 인자 값*.
  · **보고** = 그 값이 **assistant content(산문)**에 등장하는가 — 매칭도 엔진 `_ctx_has`(# 정규화 포함) 재사용.
    ★ctx가 assistant를 **제외**하므로 두 축은 **독립**이다(날조 판정이 보고 판정을 오염시키지 않는다).
  · **WRITE** = tau2 env의 `@is_tool(ToolType.WRITE)`에서 추출한 이름 집합(`--write-tools`로 주입·
    스크립트에 도구명 하드코딩 0). 미주입이면 write/read 축 없이 보고 축만 낸다.

**2×2** = (WRITE|READ) × (보고|미보고). **관심 칸 = WRITE ∧ 미보고** = NabaOS 사각.

Run:
  python3 bank_readpath_writepath.py --write-tools "a,b,c" bank_kon_… bank_koff_…
"""
import argparse
import gzip
import json
import os
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
SIMDIR = os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026", "sim_results")

from t2_gate_patch import _first_fab_call, _ctx_has, DEFAULT_ARG_HINTS  # noqa: E402


class _TC:
    """엔진 `_first_fab_call`이 기대하는 최소 인터페이스(name/arguments) — 영속 dict → 객체 shim.
    (`bank_fab_probes.prov_regen_replay`와 동일 패턴.)"""

    def __init__(self, d):
        self.name = d.get("name")
        self.arguments = d.get("arguments")


class _AM:
    def __init__(self, tcs):
        self.tool_calls = tcs


def load(tag):
    p = os.path.join(SIMDIR, f"{tag}.results.json.gz")
    with gzip.open(p, "rt", encoding="utf-8") as f:
        return json.load(f)["simulations"]


def _text(c):
    return c if isinstance(c, str) else ("" if c is None else str(c))


def all_fabs(am, ctx):
    """이 assistant 메시지의 **모든** 날조 인자 — 엔진 함수를 exclude 누적으로 반복 호출
    (엔진이 PROV-RESCUE-PERARG서 쓰는 바로 그 방식). 새 술어를 쓰지 않는다."""
    out, excl = [], set()
    while True:
        hit = _first_fab_call(am, ctx, DEFAULT_ARG_HINTS, exclude=frozenset(excl))
        if hit is None:
            return out
        tc, k, s = hit
        out.append((tc.name, k, s))
        excl.add((id(tc), k, s))


def analyze(tag, write_tools):
    sims = load(tag)
    cells = Counter()
    rows = []
    for s in sims:
        msgs = s.get("messages") or []
        # assistant 산문 전수(궤적 전체 — 호출 前/後 무관하게 "사용자에게 말했나")
        prose = " ".join(_text(m.get("content")) for m in msgs
                         if m.get("role") == "assistant").lower()
        ctx_parts = []
        for m in msgs:
            role, content = m.get("role"), m.get("content")
            tcs = m.get("tool_calls") or []
            if role == "assistant" and tcs:
                ctx = " ".join(ctx_parts).lower()
                for tname, k, val in all_fabs(_AM([_TC(t) for t in tcs]), ctx):
                    reported = _ctx_has(val, prose)
                    kind = ("WRITE" if tname in write_tools else
                            "READ" if write_tools else "?")
                    cells[(kind, "보고" if reported else "미보고")] += 1
                    rows.append(dict(sim=s.get("id"), tool=tname, arg=k, val=val,
                                     kind=kind, reported=reported))
            if role in ("user", "tool") and content is not None:
                ctx_parts.append(_text(content))
    return cells, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tags", nargs="+")
    ap.add_argument("--write-tools", default="")
    ap.add_argument("--dump", type=int, default=6, help="궤적 정독용 표본 건수([[08]])")
    a = ap.parse_args()
    wt = {x.strip() for x in a.write_tools.split(",") if x.strip()}
    print(f"WRITE 도구({len(wt)}): {sorted(wt) or '(미주입 — 보고 축만)'}\n")

    tot = Counter()
    for tag in a.tags:
        try:
            cells, rows = analyze(tag, wt)
        except FileNotFoundError:
            print(f"[{tag}] 없음 — skip")
            continue
        n = sum(cells.values())
        print(f"=== {tag} — 날조 인자 {n}건")
        if not n:
            print("    (없음)\n")
            continue
        for kind in ("WRITE", "READ", "?"):
            r, u = cells[(kind, "보고")], cells[(kind, "미보고")]
            if r or u:
                print(f"    {kind:5s}  보고={r:3d}  미보고={u:3d}   (미보고율 {100*u/(r+u):.0f}%)")
        tot.update(cells)
        for row in rows[:a.dump]:
            print(f"      · {row['kind']:5s} {'보고' if row['reported'] else '미보고'} "
                  f"{row['tool']}({row['arg']}={row['val'][:42]!r}) sim={row['sim']}")
        print()

    n = sum(tot.values())
    if not n:
        return
    print("=" * 66)
    print(f"★합계 — 날조 인자 {n}건")
    for kind in ("WRITE", "READ", "?"):
        r, u = tot[(kind, "보고")], tot[(kind, "미보고")]
        if r or u:
            print(f"  {kind:5s}  보고={r:3d}  미보고={u:3d}   (미보고율 {100*u/(r+u):.0f}%)")
    w_un = tot[("WRITE", "미보고")]
    print(f"\n★NabaOS 사각(WRITE ∧ 미보고) = {w_un}/{n} = {100*w_un/n:.0f}%")
    print("  → 0%에 가까우면 §2.1(write-path)은 **우리 데이터가 지지하지 않는다**(감사 재작성 필요).")


if __name__ == "__main__":
    main()
