# -*- coding: utf-8 -*-
r"""x381 — **리셋이 왜 0회인가**: 래치 sim 의 실제 messages 로 `_resolve_cap_ok` 를 재생한다
(원장 C537ⓓ · 무료 · CPU 만 · LLM 0 · GPU 0 · 엔진 수정 0).

## 무엇이 모순인가

`x380`(56 sim): 래치 뒤 **53/53 이 새 도구를 성공 실행**했다. 그런데 리셋 조건이 바로 그것인데
`[T2_RESOLVE_CAP] 리셋` 마커는 6런 전부 **0회**다(마커는 `a627a18b`·09:19 커밋으로 세 런 전부에
들어 있다). ⇒ **리셋 경로가 죽었거나 마커가 죽었거나** 둘 중 하나다. 추론으로 닫지 않는다([[55]]).

## 어떻게 가르나 (사본 금지 — **라이브와 같은 함수**를 부른다·[[03b]]·[[67]])

영속 gz 의 messages 를 얇은 shim 으로 감싸 `t2_gate_patch._resolve_cap_ok(self, msgs, a2)` 에
그대로 먹인다. 상태는 라이브가 만드는 그대로 세팅한다:

    self._t2_resolve_deny = 3                       (캡에 걸린 상태)
    self._t2_resolve_done = _executed_tool_names(첫 stop 시점까지의 messages, a2)   (발화 시점 스냅샷)

그 다음 **접두사를 한 줄씩 늘리며** 호출해, 리셋이 도는 첫 지점을 찾는다.

    리셋이 돈다   → 함수는 산다 ⇒ 라이브에서 안 돈 이유는 **호출부**다
                   (`_mgate` 사슬에서 앞 분기가 먼저 걸려 이 함수에 안 닿는다 — `:6875` 참조)
    리셋이 안 돈다 → **술어가 죽었다** ⇒ `_executed_tool_names`/`_exact_tool_name` 가 그 실행을
                   *새 이름*으로 보지 않는다(래퍼 이름만 늘거나 실패 표지로 걸러진다)

두 답의 처방이 다르므로 이것을 가르기 전에는 캡을 손대지 않는다([[62]]).

사용(리모트): /home/woori/venvs/seka_env/bin/python x381_cap_replay.py [태그 ...]
"""
import gzip
import io
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, "/home/woori/scratch/tau2-bench/src")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
# ★계기 결함 자기검출(2026-08-18 1차 실행): 엔진의 리셋 마커는 **stderr** 로 나가는데 그 문면이
#   한글이다. ssh 로 붙은 stderr 가 ASCII 면 그 `print` 가 UnicodeEncodeError 를 던지고,
#   그것을 감싼 `except Exception: pass` 가 **리셋 대입까지 같이 삼킨다** ⇒ 1차 실행의
#   *"15/15 리셋 안 됨"* 은 엔진 결론이 아니라 **내 파이프 결론**일 수 있다. 먼저 막고 잰다([[55]]).
try:
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import t2_forensic as F                                      # noqa: E402

DEFAULT_TAGS = ["bank_t7310_ctl_20260818e", "bank_t7310_treat_20260818e",
                "bank_t7312_treat_20260818g"]
MAXSIM = 6                                                   # 태그당 상한(재생은 CPU 만 쓴다)


class TC(object):
    """tool_call shim — 엔진이 읽는 속성만 갖는다(`name`·`arguments`·`id`)."""

    def __init__(self, d):
        f = d.get("function") or d
        self.name = f.get("name") or d.get("name")
        self.arguments = f.get("arguments", d.get("arguments"))
        self.id = d.get("id")


class M(object):
    """message shim — `role`·`content`·`tool_calls`·`id`·`tool_call_id`·`error`."""

    def __init__(self, d):
        self.role = d.get("role")
        self.content = d.get("content")
        self.tool_calls = [TC(t) for t in (d.get("tool_calls") or [])]
        self.id = d.get("id")
        self.tool_call_id = d.get("tool_call_id")
        self.error = d.get("error", False)


class Agent(object):
    """`self` shim — 엔진은 `getattr/setattr` 만 쓴다."""
    pass


def a2_load():
    """세 층 병합 — 라이브와 같은 경로(`load_domain_a2`)가 있으면 그것을 쓴다([[24]])."""
    try:
        import t2_a2 as _a2
        for fn in ("load_domain_a2", "load"):
            if hasattr(_a2, fn):
                return getattr(_a2, fn)("banking_knowledge")
    except Exception:
        pass
    out = {}
    for name in ("banking_knowledge.settings.json", "banking_knowledge.specific.json"):
        p = os.path.join(HERE, "a2", name)
        if os.path.exists(p):
            out.update(json.load(io.open(p, encoding="utf-8")))
    return out


def main():
    tags = [a for a in sys.argv[1:] if not a.startswith("--")] or DEFAULT_TAGS
    import t2_gate_patch as GP                                # 라이브와 같은 모듈
    a2 = a2_load()
    print("=" * 100)
    print("x381 · resolve_cap 재생 · 태그 %s · A2 키 %d · failure_markers %d"
          % (",".join(t.split("_")[1] for t in tags), len(a2),
             len(a2.get("failure_markers") or ())))
    print("판정(사전 고정): 재생에서 리셋이 돈다 → 함수는 살아 있고 **호출부**가 원인 · "
          "안 돈다 → **술어가 죽었다**(새 이름으로 안 보인다)")
    print("=" * 100)

    live_ok = fn_ok = 0
    for tag in tags:
        stops = F.turns_of(tag, r"stop=resolve_cap")
        n = 0
        for s in F.scored(tag, ".results.json.gz"):
            key = F.simtag(s)
            st = [t for t in (stops.get(key) or []) if t is not None]
            if not st or n >= MAXSIM:
                continue
            n += 1
            msgs = [M(d) for d in (s.get("messages") or [])]
            first = min(min(st), len(msgs))
            self = Agent()
            snap = GP._executed_tool_names(msgs[:first], a2)      # ★라이브 함수
            self._t2_resolve_deny = 3
            self._t2_resolve_done = set(snap)
            hit, hit_names = None, []
            for i in range(first, len(msgs) + 1):
                self._t2_resolve_deny = 3                        # 매 검사 전 캡 상태로 되돌린다
                # ★부정통제(1차 실행 교훈): 엔진 내부의 `except: pass` 가 무엇을 삼키는지
                #   보이도록 **같은 계산을 밖에서도** 해 둔다. 둘이 갈리면 삼킨 것이다.
                outside = bool(GP._executed_tool_names(msgs[:i], a2) - snap)
                ok = GP._resolve_cap_ok(self, msgs[:i], a2)
                if outside and not ok:
                    print("       ⛔밖에선 새 이름이 보이는데 함수는 리셋 안 함 (i=%d) — "
                          "엔진 내부 예외 삼킴 의심" % i)
                    break
                if ok:
                    hit = i
                    hit_names = sorted(GP._executed_tool_names(msgs[:i], a2) - snap)[:5]
                    break
            grew = sorted(GP._executed_tool_names(msgs, a2) - snap)
            print("  %-9s %-6s %-5s first=%-4s 스냅샷 %2d개 · 끝까지 새 이름 %2d개 · 리셋 %s"
                  % (F.task_id(s), tag.split("_")[1],
                     ("treat" if "treat" in tag else "ctl"), first, len(snap), len(grew),
                     ("i=%d %s" % (hit, ",".join(x[:20] for x in hit_names)))
                     if hit is not None else "**안 됨**"))
            if hit is not None:
                fn_ok += 1
            else:
                live_ok += 1
            if hit is None and grew:
                print("       ⚠새 이름은 있는데 리셋이 안 됐다 → 술어/스냅샷 축을 본다: %s"
                      % ",".join(x[:24] for x in grew[:5]))

    print("")
    print("## 판정")
    if fn_ok and not live_ok:
        v = ("**함수는 산다** — 재생에서 전부 리셋됐다 ⇒ 라이브 0회의 원인은 **호출부**다"
             "(`_mgate` 사슬에서 앞 분기가 먼저 걸려 이 함수에 안 닿는 경로를 세야 한다)")
    elif live_ok and not fn_ok:
        v = ("**술어가 죽었다** — 실제 messages 로도 리셋이 안 된다 ⇒ `_executed_tool_names`/"
             "`_exact_tool_name` 가 그 실행을 *새 이름*으로 보지 않는다. 수리 대상은 캡이 아니라 "
             "이 술어다([[57]] 인자 변화 기준)")
    else:
        v = "혼합 %d 됨 / %d 안 됨 — sim 별 표를 읽는다([[08]] 집계 직행 금지)" % (fn_ok, live_ok)
    print(v)
    return 0


if __name__ == "__main__":
    sys.exit(main())
