#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""x373 — 요구-인용이 왜 전량 기각되는가 (C532⒢ · S3 게이팅 항목).

## 왜

t7310(S2 스모크)에서 `T2_VERDICT_CARRY` 는 트리거 자리 **9/9 = 100%** 발화했다. 그런데 트리거
자체가 안 선 sim 이 있다 — `task_098` 은 인용 **1개 중 통과 0**, `task_024` 는 두 번째 호출에서
**3개 중 통과 0** 이었다. 098 은 군 선택이 결과를 가르는 태스크라(C502) **표적에서 새는 자리**다.

로그에 개수만 있어 *옳은 거부*(모델이 축자를 안 지킴)인지 *과한 검산*(우리 `quote_in` 이 참인
인용을 떨어뜨림)인지 가를 수 없었다. 이 프로브가 그것을 가른다.

## 무엇을 정본에서 가져오는가 (사본 0 · [[67]])

  · 프롬프트 = A2 `policy_ontology.requirement_prompt` **축자**(도메인 어휘는 A2 몫)
  · 호출·파싱 = `t2_search.sub_requirements` **그대로**(JSON 경계 `find`/`rfind`·정규식 0)
  · 검산 = `t2_search.quote_in` **그대로**(강조 제거 + 공백 합침을 양쪽 대칭)
  ⇒ **shim 은 전송뿐**이다(오프라인이라 orchestrator 의 agent 가 없다).

## 계기 규율

  ⚠**양성통제 필수**: 같은 런에서 통과한 태스크(055 = 3/3·6/6)를 같이 돌린다. 그 팔도 0 이면
    재현 자체가 실패한 것이고 아무 결론도 못 낸다([[08]]·C524 *"양성통제 없는 0 은 결손이 아니다"*).
  ⚠**입력 근사 자인**: 라이브는 그 호출 시점까지의 user 메시지를 넘긴다. 로그 줄에 `turn=` 이
    없어 시점을 모르므로 여기서는 **전체 user 메시지**를 쓴다 — 상위집합이라 *검산은 더 쉬워진다*.
    ⇒ 여기서도 기각되면 그 기각은 **시점 탓이 아니다**(단방향 논증).
  ⚠온도 0 ×2(det) — 답이 갈리면 결정론이 깨진 것으로 인쇄하고 판단을 미룬다.

사용: PYTHONPATH=<tau2>/src T2_PROBE_URL=http://localhost:8141/... python x373_reqquote_reject.py
"""
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from x216_read_and_offset import chat            # noqa: E402  (프로브 전송 정본)
import t2_forensic as F                          # noqa: E402
import t2_search as TS                           # noqa: E402

TAG = "bank_t7310_treat_20260818e"
TASKS = [("task_098", "표적(기각)"), ("task_024", "표적(2차 호출 기각)"),
         ("task_055", "양성통제(통과)")]
A2 = json.load(io.open(os.path.join(HERE, "a2", "banking_knowledge.specific.json"),
                       encoding="utf-8"))
SPEC = A2["policy_ontology"]


class _UM(object):
    def __init__(self, role=None, content=None):
        self.role, self.content = role or "user", content


class _Agent(object):
    llm = None
    llm_args = {}


class _LA(object):
    """전송 shim — 정본 `sub_generate` 가 부르는 `generate` 계약만 흉내낸다."""
    @staticmethod
    def generate(model=None, tools=None, messages=None, call_name=None, **kw):
        body = getattr(messages[0], "content", "")
        r = chat(body, None, 0.0, 900)

        class _R(object):
            pass
        o = _R()
        o.content = str((r or {}).get("content") or "")
        return o


def user_text(sim):
    """엔진과 **같은 조립**: user 역할 메시지를 개행 둘로 잇는다(`t2_gate_patch:_utxt`)."""
    return "\n\n".join(str(m.get("content") or "")
                       for m in (sim.get("messages") or [])
                       if str(m.get("role")) == "user" and m.get("content"))


def main():
    print("=" * 78)
    print("x373 — 요구-인용 기각 원인 (옳은 거부 ↔ 과한 검산)")
    print("=" * 78)
    print("\n판정(사전 고정·결과보다 먼저):")
    print("  · 양성통제(055)가 0 이면 → **재현 실패**, 아무 결론도 내지 않는다")
    print("  · 055 통과 ∧ 098/024 인용이 원문에 **없다** → 옳은 거부(모델이 축자를 안 지킴)")
    print("  · 055 통과 ∧ 098/024 인용이 원문에 **있는데** quote_in 이 떨어뜨림 → 과한 검산(우리 층)\n")

    sims = {F.task_id(s): s for s in F.scored(TAG, ".results.json.gz")}
    for task, why in TASKS:
        s = sims.get(task)
        print("-" * 78)
        print("[%s] %s" % (task, why))
        if s is None:
            print("  ⚠sim 없음 — 건너뜀"); continue
        ut = user_text(s)
        print("  user 텍스트 %d자 · user 메시지 %d개"
              % (len(ut), sum(1 for m in (s.get("messages") or [])
                              if str(m.get("role")) == "user")))
        outs = []
        for rnd in (1, 2):
            qs = TS.sub_requirements(_Agent(), _LA(), _UM, SPEC, ut)   # ★정본 그대로
            outs.append(qs)
            ok = [q for q in qs if TS.quote_in(q, ut)]                 # ★정본 그대로
            print("  라운드%d: 인용 %d개 · 검산 통과 %d개" % (rnd, len(qs), len(ok)))
            for q in qs:
                passed = TS.quote_in(q, ut)
                raw_in = q in ut
                print("     %s %r%s" % ("✓" if passed else "✗", q[:88],
                                        "" if passed else
                                        ("   (원문에 그대로 있음! ⇒ 검산 문제)" if raw_in
                                         else "   (원문에 없음 ⇒ 축자 아님)")))
        if outs[0] != outs[1]:
            print("  ⚠온도 0 인데 두 답이 다르다 — 결정론 아님. 판단 보류.")

    print("\n" + "=" * 78)
    print("⚠입력 근사: 라이브는 호출 시점까지의 user 메시지만 준다. 여기서는 **전체**를 줬으므로")
    print("  검산이 더 쉬운 조건이다 — 그런데도 기각되면 그 기각은 시점 탓이 아니다.")
    print("=" * 78)


if __name__ == "__main__":
    main()
