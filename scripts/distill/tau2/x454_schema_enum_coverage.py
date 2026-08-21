# -*- coding: utf-8 -*-
r"""x454 — **도구 스키마가 이미 말해 주는 것**의 커버리지 계량 (2026-08-21·무료·LLM 0·G3 1단계)

## 왜 (사용자 지시 *"047 부터 가라"* → *"1번부터 진행하라"*)
047 의 실패 넷 중 셋은 레버가 없어서가 아니라 **선언이 그 도구에 안 붙어서** 났다. 로그 전수
(047 블록 2,437)에서 `[T2_WRITE_ARG_ENUM]` **0회** · call-form **0회** · unlock 힌트 문구 **0회**.
그런데 필요한 정보는 **도구가 자기 계약에 축자로** 적어 두었다:

    log_credit_card_closure_reason_4521(… closure_reason: str)
        "Must be one of: 'annual_fee', 'not_using_card', 'found_better_card', …"
    apply_statement_credit_8472(… amount: float …)
        "(number) The credit amount in dollars (e.g., 25.00 …)"

⇒ 열거값·타입을 **손으로 A2 에 적을 이유가 없다**. [[23]] 판정 순서의 최상단(*"env 기계도출 =
opex 0"*)이고, 손으로 적으면 그때부터 gold 경유 위험과 유지 비용이 생긴다.

## 이 스크립트가 하는 일 (읽기만·LLM 0·엔진 판단 0)
환경이 들고 있는 **도구 시그니처와 docstring** 에서
  ⑴ 인자별 **선언 타입**(`str`/`float`/`int`/…)
  ⑵ docstring 이 *"Must be one of: …"* 꼴로 **열거**를 주는 인자
를 세고, **몇 개 도구·몇 개 인자**를 덮는지 인쇄한다. 덮는 폭이 크면 이것 자체가 도메인-일반 레버다.

⚠이건 **도메인 텍스트 패턴매칭이 아니다**([[59]]): 읽는 대상은 정책 문서가 아니라 **도구의 기계
계약**(파이썬 시그니처 + 자기 docstring)이다. 값의 뜻을 해석하지 않고 **목록을 그대로 옮긴다**.
⚠A2 에 손으로 적지 않는다 — 이 스크립트는 **계량만** 한다. 선언 여부는 격리 결과를 보고 정한다([[62]]).

사용: (리모트·cwd=tau2 · PYTHONPATH=src:…) py x454_schema_enum_coverage.py
"""
import argparse
import collections
import inspect
import io
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

REP = os.path.abspath(os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026"))

# ★도구 **계약** 문면의 고정 관용구다(도메인 어휘가 아니다). tau2 가 모든 discoverable 도구에
#   같은 꼴로 쓴다 — 값의 뜻은 안 본다.
_ONEOF = re.compile(r"must be one of:\s*(\[[^\]]*\]|[^\n]+)", re.I)
_QUOTED = re.compile(r"'([^']+)'")


def enum_of(doc, arg):
    """docstring 의 그 인자 줄에서 열거 목록만 그대로 옮긴다(해석 0)."""
    for line in str(doc or "").split("\n"):
        s = line.strip()
        if not s.lower().startswith(arg.lower() + " ("):
            continue
        m = _ONEOF.search(s)
        if not m:
            return []
        return _QUOTED.findall(m.group(1))
    return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="x454_schema_coverage.json")
    a = ap.parse_args()
    from tau2.domains.banking_knowledge.environment import get_environment
    tk = get_environment(retrieval_variant="alltools").tools

    names = [n for n in dir(tk) if not n.startswith("_") and callable(getattr(tk, n, None))]
    rows, tally = [], collections.Counter()
    for n in sorted(names):
        f = getattr(tk, n)
        try:
            sig = inspect.signature(f)
        except (TypeError, ValueError):
            continue
        doc = inspect.getdoc(f) or ""
        params = []
        for pname, p in sig.parameters.items():
            if pname == "self":
                continue
            ann = p.annotation
            tname = getattr(ann, "__name__", None) or (str(ann) if ann is not inspect._empty else "")
            vals = enum_of(doc, pname)
            params.append({"arg": pname, "type": tname, "enum": vals})
            tally["args"] += 1
            if tname:
                tally["typed"] += 1
            if vals:
                tally["enum_args"] += 1
        if not params:
            continue
        rows.append({"tool": n, "params": params})
        tally["tools"] += 1
        if any(p["enum"] for p in params):
            tally["tools_with_enum"] += 1
        if any(p["type"] in ("float", "int") for p in params):
            tally["tools_with_number"] += 1

    print("=" * 96)
    print("x454 · 도구 %d · 인자 %d · **타입 선언된 인자 %d** · **열거를 주는 인자 %d**"
          % (tally["tools"], tally["args"], tally["typed"], tally["enum_args"]))
    print("   열거를 가진 도구 %d · 수치 인자를 가진 도구 %d"
          % (tally["tools_with_enum"], tally["tools_with_number"]))
    print("=" * 96)
    print("\n[열거를 주는 인자 전부] — 손으로 적을 필요가 없는 목록")
    for r in rows:
        for p in r["params"]:
            if p["enum"]:
                print("  %-44s %-22s %d개  %s"
                      % (r["tool"][:44], p["arg"][:22], len(p["enum"]),
                         ", ".join(p["enum"])[:60]))
    print("\n[수치 타입 인자] — 문자열로 보내면 env 가 타입 에러로 죽는 자리")
    for r in rows:
        for p in r["params"]:
            if p["type"] in ("float", "int"):
                print("  %-44s %-22s %s" % (r["tool"][:44], p["arg"][:22], p["type"]))

    p = os.path.join(REP, a.out)
    with io.open(p, "w", encoding="utf-8") as f:
        json.dump({"tally": dict(tally), "tools": rows}, f, ensure_ascii=False, indent=1)
    print("\n→ %s" % p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
