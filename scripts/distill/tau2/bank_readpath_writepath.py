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

from t2_gate_patch import _first_fab_call, _ctx_has, _args_dict, DEFAULT_ARG_HINTS  # noqa: E402


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


def _leaves(v):
    """중첩 dict/list의 문자열 잎 — `--all-args`용(엔진 `_flatten`은 hint 매칭 인자만 받으므로 별도)."""
    if isinstance(v, dict):
        for x in v.values():
            yield from _leaves(x)
    elif isinstance(v, (list, tuple)):
        for x in v:
            yield from _leaves(x)
    elif v is not None:
        yield str(v)


def all_args_fabs(tcs, ctx, write_tools):
    """★`--all-args` (2026-07-18·사용자 ❷): 엔진 `_first_fab_call`은 **hint 매칭 인자만** 본다
    (`_hint_hit('record'|'amount'|'date') = False`) → **WRITE 도구의 금액/날짜/참조 인자 날조를 구조적으로
    못 본다**. 그 사각을 열기 위해 **WRITE 도구에 한해 모든 인자 잎**을 같은 술어(`_ctx_has`·len≥4)로 본다.
    ⚠️**오탐이 는다**: 식별값과 달리 **계산된 값**(합계 등)은 ctx에 축자로 없는 게 **정당**하다.
    ⇒ 집계로 결론 금지·**per-case 정독 필수**([[08]])."""
    out = []
    for tc in tcs:
        if tc.name not in write_tools:
            continue
        for k, v in (_args_dict(tc) or {}).items():
            for val in _leaves(v):
                s = str(val).strip()
                if len(s) >= 4 and not _ctx_has(s, ctx):
                    out.append((tc.name, k, s))
    return out


def analyze(tag, write_tools, all_args=False):
    sims = load(tag)
    cells = Counter()
    rows = []
    for s in sims:
        msgs = s.get("messages") or []
        # assistant 산문 전수(궤적 전체 — 호출 前/後 무관하게 "값을 사용자에게 말했나")
        prose = " ".join(_text(m.get("content")) for m in msgs
                         if m.get("role") == "assistant").lower()
        ctx_parts = []
        for i, m in enumerate(msgs):
            role, content = m.get("role"), m.get("content")
            tcs = m.get("tool_calls") or []
            if role == "assistant" and tcs:
                ctx = " ".join(ctx_parts).lower()
                shim = [_TC(t) for t in tcs]
                hits = (all_args_fabs(shim, ctx, write_tools) if all_args
                        else all_fabs(_AM(shim), ctx))
                # 행위-보고 프록시(3분법·사용자 ❷): 이 호출 **後** assistant 산문이 있나
                acted = any(_text(mm.get("content")).strip()
                            for mm in msgs[i + 1:] if mm.get("role") == "assistant")
                for tname, k, val in hits:
                    kind = ("WRITE" if tname in write_tools else
                            "READ" if write_tools else "?")
                    if _ctx_has(val, prose):
                        bucket = "a.값-보고"
                    elif acted:
                        bucket = "b.행위만-보고"
                    else:
                        bucket = "c.미보고"
                    cells[(kind, bucket)] += 1
                    rows.append(dict(sim=s.get("id"), tool=tname, arg=k, val=val,
                                     kind=kind, bucket=bucket))
            if role in ("user", "tool") and content is not None:
                ctx_parts.append(_text(content))
    return cells, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tags", nargs="+")
    ap.add_argument("--write-tools", default="")
    ap.add_argument("--all-args", action="store_true",
                    help="WRITE 도구의 **모든** 인자(hint 밖 amount/date/reference 포함) — 사각 열기")
    ap.add_argument("--dump", type=int, default=6, help="궤적 정독용 표본 건수([[08]])")
    a = ap.parse_args()
    wt = {x.strip() for x in a.write_tools.split(",") if x.strip()}
    BUCKETS = ("a.값-보고", "b.행위만-보고", "c.미보고")
    print(f"WRITE 도구({len(wt)}): {sorted(wt) or '(미주입 — 보고 축만)'}")
    print(f"모드: {'--all-args (WRITE 전 인자·오탐↑·per-case 정독 필수)' if a.all_args else '엔진 hint 인자만'}")
    print("★반증조건 **사전등록**(결과 보기 前): WRITE ∧ (b+c) == 0 이면 **인자-축 절단선 폐기·정직 보고**.\n")

    tot = Counter()
    allrows = []
    for tag in a.tags:
        try:
            cells, rows = analyze(tag, wt, a.all_args)
        except FileNotFoundError:
            print(f"[{tag}] 없음 — skip")
            continue
        n = sum(cells.values())
        print(f"=== {tag} — 날조 후보 {n}건")
        if not n:
            print("    (없음)\n")
            continue
        for kind in ("WRITE", "READ", "?"):
            row = [cells[(kind, b)] for b in BUCKETS]
            if any(row):
                print(f"    {kind:5s}  " + "  ".join(f"{b}={c:3d}" for b, c in zip(BUCKETS, row)))
        tot.update(cells)
        allrows += rows
        for row in rows[:a.dump]:
            print(f"      · {row['kind']:5s} {row['bucket']:12s} "
                  f"{row['tool']}({row['arg']}={row['val'][:40]!r}) sim={row['sim']}")
        print()

    n = sum(tot.values())
    if not n:
        print("총 0건 — 사전등록 반증조건 발동: 인자-축 절단선 폐기.")
        return
    print("=" * 70)
    print(f"★합계 — 날조 후보 {n}건")
    for kind in ("WRITE", "READ", "?"):
        row = [tot[(kind, b)] for b in BUCKETS]
        if any(row):
            print(f"  {kind:5s}  " + "  ".join(f"{b}={c:3d}" for b, c in zip(BUCKETS, row)))
    wbc = tot[("WRITE", "b.행위만-보고")] + tot[("WRITE", "c.미보고")]
    print(f"\n★NabaOS 사각(WRITE ∧ b+c) = {wbc}")
    print("  " + ("→ **0 = 사전등록 반증조건 발동**: 인자-축 절단선 폐기·정직 보고."
                  if wbc == 0 else
                  f"→ 0 아님 ⇒ **per-case 정독 필수**([[08]]·계산된 값=정당한 오탐 배제). 아래 전건:"))
    if wbc:
        for r in allrows:
            if r["kind"] == "WRITE" and r["bucket"] != "a.값-보고":
                print(f"    · {r['bucket']:12s} {r['tool']}({r['arg']}={r['val'][:60]!r}) sim={r['sim']}")


if __name__ == "__main__":
    main()
