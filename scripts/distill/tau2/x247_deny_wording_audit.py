# -*- coding: utf-8 -*-
r"""x247 — **거부 문구 전수 감사**: 무엇이 틀렸나 · 무엇을 하면 풀리나 (유료 0 · 모델 0).

## 왜 (사용자 지시 2026-08-11 · 메모리 `64-deny-must-name-the-fix` · 원장 C413·C414)

`_FB_GENERIC`(*"resolve the flagged call(s) first"*) 하나가 30 sim 중 6건을 라이브락에 빠뜨렸다.
그 문구는 **해소할 대상을 말하지 않는다**. 같은 병이 다른 문구에도 있는지 **세어 본다**.

## 기준 (둘 다 있어야 통과)

  ⒜ **무엇이 틀렸나** — 어느 도구·어느 인자·어느 선행 단계인지 **지목**한다.
      신호: 슬롯(`{...}`) · 도구 이름 · 대문자 태그 뒤의 구체 명사구.
  ⒝ **무엇을 하면 풀리나** — 다음 한 수를 **명령형**으로 준다.
      신호: `call ` · `search ` · `ask ` · `tell ` · `re-attempt` · `first,` · `then ` 등.

⚠이 감사는 **문자열 검사**다([[M]]). "통과"가 좋은 문구라는 뜻은 아니고, **탈락은 확실한 결함**
  이라는 뜻이다(둘 중 하나가 아예 없다). 판정은 사람이 읽고 확정한다.
⚠A2 문구는 **정본 두 층**에서 읽는다([[24]]) — 한 층만 보면 갈린 것을 못 본다.

실행: py -3 x247_deny_wording_audit.py [--all]
"""
import argparse
import collections
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
LAYERS = ["a2/banking_knowledge.gate.json", "a2/banking_knowledge.specific.json"]
ENGINE = ["t2_gate_patch.py", "t2_resolve.py", "t2_ledger.py"]

ACT = re.compile(r"\b(call|search|ask|tell|give|unlock|re-?attempt|retry|read|copy|run|use|"
                 r"restate|first|then|instead|do not conclude)\b", re.I)
NAMED = re.compile(r"\{[a-z_]+\}|`[^`]+`|'[A-Za-z_][A-Za-z0-9_]{3,}'|\b[a-z_]+_[a-z_]+\b")
DENYISH = re.compile(r"^\s*(Error:|Note:|\[[A-Z][A-Z0-9 _-]+\])")
# ★대상은 **모델이 실제로 받는 문구**뿐이다. 첫 판은 우리 stderr 마크(`[T2_*]`)까지 잡아 249개를
#   탈락시켰는데 그건 로그이지 거부가 아니다 — 모델에 안 보이므로 [[64]] 규칙의 대상이 아니다.
#   (계기 문자열을 규칙으로 재는 것은 x240 의 토큰 상한 사고와 같은 종류의 오측정이다.)
OURS = re.compile(r"^\s*\[T2_[A-Z0-9_]+\]")


def model_facing(t):
    return DENYISH.match(t) and not OURS.match(t) and "%" not in t[:60]


def walk(o, path=""):
    if isinstance(o, dict):
        for k, v in o.items():
            for x in walk(v, path + "/" + str(k)):
                yield x
    elif isinstance(o, list):
        for i, v in enumerate(o):
            for x in walk(v, path + "[%d]" % i):
                yield x
    elif isinstance(o, str):
        yield path, o


def judge(t):
    """(무엇이 틀렸나, 무엇을 하면 풀리나) — 둘 다 있어야 통과."""
    return bool(NAMED.search(t)), bool(ACT.search(t))


def a2_strings():
    out = []
    for rel in LAYERS:
        p = os.path.join(HERE, rel)
        try:
            d = json.load(open(p, encoding="utf-8"))
        except Exception:
            continue
        for path, t in walk(d):
            if path.endswith("_note") or "/_note" in path:
                continue
            s = " ".join(t.split())
            if len(s) > 24 and model_facing(s):
                out.append((rel + path, s))
    return out


def engine_strings():
    out = []
    for rel in ENGINE:
        p = os.path.join(HERE, rel)
        try:
            src = open(p, encoding="utf-8").read()
        except Exception:
            continue
        for m in re.finditer(r'"((?:Error:|Note:|\[[A-Z][A-Z0-9 _-]+\])[^"]{20,})"', src):
            # ★암묵 연결을 이어 붙인다. 첫 판은 리터럴 **한 조각**만 보고 판정해서
            #   `"Error: [WRITE-GROUNDING] value '{val}' for {arg}"` 처럼 **다음 줄에 이어지는**
            #   문구를 *다음 한 수 없음*으로 잘못 탈락시켰다(x240 토큰 상한과 같은 종류의 오측정).
            j, parts = m.end(), [m.group(1)]
            while True:
                nxt = re.match(r'(?:\s|\\\n)*"([^"]*)"', src[j:])
                if not nxt:
                    break
                parts.append(nxt.group(1))
                j += nxt.end()
            s = " ".join("".join(parts).split())
            if not model_facing(s):
                continue
            line = src[:m.start()].count("\n") + 1
            out.append(("%s:%d" % (rel, line), s))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="통과한 것까지 인쇄")
    a = ap.parse_args()
    rows = a2_strings() + engine_strings()
    # 같은 문구가 여러 자리에 있으면 한 번만 판정한다(층 동기화는 [[24]] 검정이 따로 본다)
    seen, uniq = set(), []
    for where, t in rows:
        if t in seen:
            continue
        seen.add(t)
        uniq.append((where, t))
    bad = []
    for where, t in uniq:
        named, act = judge(t)
        if not (named and act):
            bad.append((where, t, named, act))
    print("거부성 문구 %d개(중복 제거) · **탈락 %d개**\n" % (len(uniq), len(bad)))
    for where, t, named, act in sorted(bad, key=lambda r: len(r[1])):
        miss = " · ".join(x for x, ok in (("무엇이 틀렸나 없음", named),
                                          ("다음 한 수 없음", act)) if not ok)
        print("  ⛔ %-46s %s" % (where[-46:], miss))
        print("     %s" % t[:200])
    if a.all:
        print("\n— 통과(문자열 기준) %d개 —" % (len(uniq) - len(bad)))
        for where, t in uniq:
            if all(judge(t)):
                print("  ok %-44s %s" % (where[-44:], t[:110]))
    print("\n※ 탈락은 **확실한 결함**이다(둘 중 하나가 아예 없다). 통과는 문자열 기준일 뿐이니"
          "\n  라이브에서 3회 이상 반복되는 문구가 있으면 그것부터 사람이 읽어라([[64]] 감시 지표).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
