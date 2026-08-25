# -*- coding: utf-8 -*-
"""identifying_arg_types 래칫 (2026-08-25) — **선언한 힌트가 실제로 걸리는가**.

왜 이 검정이 필요한가 (2026-08-25 실물): A2 는 `time_verified` 를 힌트로 선언해 두었고 그
_note_ 는 *"'time_verified'는 해당 인자 key에만 **substring-매칭**"* 이라고 적었다. 그런데
`_hint_hit` 은 2026-07-16 에 **토큰 접두**로 바뀌었다(`"id" in "provided"` 오탐 수리). 그 뒤로
`_hint_hit("time_verified", ("time_verified",))` 는 **False** 다 — 키를 토큰으로 쪼개면
['time','verified'] 이고 어느 토큰도 'time_verified' 로 시작하지 않는다. 즉 그 선언은 **한 달 넘게
죽어 있었고** 아무 계기도 그것을 말하지 않았다([[24]] 死배선·[[25]] 침묵을 증거로 읽지 마라).

이 검정이 지키는 것:
  ① 선언이 존재하고 두 정본 층(settings·gate)이 **같은 값**이다([[24]])
  ② 선언한 토큰마다 `_hint_hit` 이 **실제로 참**이 되는 키가 있다 — 죽은 선언 금지
     (아는 죽음은 KNOWN_DEAD 에 이유·날짜와 함께 적고, 새 죽음은 실패시킨다)
  ③ 엔진이 그 선언을 여전히 병합한다(`_hints` enrich 자리 생존)
"""
import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# 토큰 → 그 힌트가 걸려야 하는 **실재 인자 이름**(env 도구 스키마에서 온 이름).
WITNESS = {
    "digit": "card_last_4_digits",
}
# 죽어 있음을 **알고 있는** 선언. 되살리려면 폭발 반경을 재고 이 표에서 지운다.
KNOWN_DEAD = {
    "time_verified": ("2026-08-25 확인 — `_hint_hit` 이 2026-07-16 에 substring 에서 토큰 접두로 "
                      "바뀌면서 죽었다. 살리려면 토큰을 'time' 으로 바꿔야 하고, t7354 전 배치 "
                      "실측 폭발 반경은 0건이나 원 근거(LOGV_TIME_FAB 59 sim)는 그 로스터 밖이라 "
                      "측정 없이 켜지 않는다."),
}


def main():
    import json
    from gate_interpreter import load_domain_a2
    import t2_gate_patch as G

    a2 = load_domain_a2("banking_knowledge")
    decl = list(a2.get("identifying_arg_types") or ())
    assert decl, "identifying_arg_types 선언이 사라졌다 — 날조 가드의 시야가 좁아진다"

    gate = json.load(io.open(os.path.join(HERE, "a2", "banking_knowledge.gate.json"),
                             encoding="utf-8"))

    def find(o):
        if isinstance(o, dict):
            if "identifying_arg_types" in o:
                return o["identifying_arg_types"]
            for v in o.values():
                r = find(v)
                if r is not None:
                    return r
        elif isinstance(o, list):
            for v in o:
                r = find(v)
                if r is not None:
                    return r
        return None

    g = find(gate)
    assert sorted(g or []) == sorted(decl), (
        "정본 층과 gate.json 이 갈렸다([[24]]): %r vs %r" % (decl, g))

    hints = tuple(set(G.DEFAULT_ARG_HINTS) | set(decl))
    dead = []
    for tok in decl:
        if tok in KNOWN_DEAD:
            continue
        w = WITNESS.get(tok)
        assert w, ("선언 토큰 %r 에 증인 인자 이름이 없다 — WITNESS 에 env 스키마의 "
                   "실재 인자 이름을 적어라(죽은 선언 방지)" % tok)
        if not G._hint_hit(w, hints):
            dead.append((tok, w))
    assert not dead, ("선언했는데 걸리지 않는 힌트가 있다(死선언): %r" % dead)

    src = io.open(os.path.join(HERE, "t2_gate_patch.py"), encoding="utf-8").read()
    assert 'a2.get("identifying_arg_types")' in src, "엔진이 이 선언을 더 이상 병합하지 않는다"
    print("OK identifying_arg_types: 선언 %d · 두 층 일치 · 살아 있는 토큰 %d · 기지 死선언 %d"
          % (len(decl), len(decl) - len(set(KNOWN_DEAD) & set(decl)), len(set(KNOWN_DEAD) & set(decl))))


if __name__ == "__main__":
    main()
