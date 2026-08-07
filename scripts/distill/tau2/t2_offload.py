# -*- coding: utf-8 -*-
"""C5 이관 — **엔진은 연산자와 서식 템플릿을 갖고, 도메인은 필드 목록만 준다.**

계약 정의 = `GENERAL_CONTRACTS_DESIGN_2026_08_06.md` §2-C5.

두 가지만 한다:
  operators()   엔진이 가진 연산자 이름 (도메인 0)
  run(spec,ctx) 그 연산자를 구조화된 값 위에서 실행

★**operand는 엔진이 만들지 않는다**(사용자 지시): 어느 계좌·어느 필드인가는 해석이고 LLM 몫이다.
★**엔진은 도메인 텍스트를 읽지 않는다**([[59]]): 도구 출력 → 구조화는 모델이 하고, 엔진은 그
   구조 위의 산수만 한다. 이 규칙은 hook으로 강제된다(`scaffold_guard`).

지금은 기존 두 구현의 **단일 진입점**이다 — 복사하지 않는다:
  `t2_compute.apply_op`  일반 op 라이브러리(count_where·days_between·diff·lookup_table·…)
  `t2_ledger`            원장 산수(창 잔여·그룹 누적) — 전사는 모델, 산수는 엔진
"""

import t2_compute as _c
import t2_ledger as _lg

__all__ = ["operators", "run", "ledger_facts", "OPS"]

OPS = ("const", "ref", "min", "max", "argmin", "argmax", "sum", "count_where",
       "diff", "clamp", "lookup_table", "days_between", "if_then", "bool_expr", "filter")


def operators():
    return tuple(OPS)


def run(spec, ctx):
    """A2가 선언한 op 스펙을 ctx(구조화된 값) 위에서 실행. 실패·미확정 = None(=미개입).

    미확정을 0이나 추정으로 채우지 않는 것이 계약이다 — 채우면 우리 층이 **없는 사실을 만든다**
    ([[25]]가 기록한 우리 출력 결함 4건이 전부 그 형태였다).
    """
    try:
        return _c.apply_op(spec, ctx)
    except Exception:
        return None


def ledger_facts(rows, spec, now=None):
    """구조화된 원장 행 위의 산수 → 표면화 문구. 행은 **모델이 전사한 것**이어야 한다."""
    return _lg.facts_text(rows, spec, now=now)
