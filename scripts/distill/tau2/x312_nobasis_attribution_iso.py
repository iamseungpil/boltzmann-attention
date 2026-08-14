# -*- coding: utf-8 -*-
r"""x312 — x311 E_NOBASIS(2/8)의 **귀속**: 일반 계약이 날조를 산 것인가, 아니면 계기가 샌 것인가.

배경(2026-08-14 야간 §4·§5):
  x311 이 A2 write_initiation 을 일반 계약으로 바꿔 출시했다(B_GEN 8/8 = A_DOM 8/8 ·
  D_GEN75F 8/8 로 근거 폭이 인자임도 확정). 그러나 날조 부정통제 **E_NOBASIS 가 2/8** —
  사전 고정 문턱(≥2 → 무효)에 걸렸다. 그런데 E 는 **한 자리에서 두 가지가 동시에 달랐다**:
    (i) 계약: 도메인 필드 → 일반(`{tool, arguments}`)
    (ii) 근거: 있음 → 없음
  게다가 세 번째 누수가 있다 — 075 컷은 정의상 **손님이 도구명을 대며 실행을 요구한** 자리다
  (`cut75`). 근거를 지워도 `asks` 에 그 이름이 남아 있으므로, 모델이 이름을 되뇐 것을
  "근거 없이 지어냈다"로 셀 수 있다. x311 의 `hit()` 도 GEN_JSON 이 스스로 제공하는
  `"arguments"` 라는 낱말을 정답 표지로 쓰고 있어(형식 충족=적중) 판정이 느슨하다.

이 프로브는 **사이트를 075 nobasis 로 고정**하고 한 번에 하나씩만 바꾼다:

셀 5 (n=8·컷은 x311 축자):
  A_DOM_NB    도메인 계약 · 근거 없음                    ← (i) 만 B 와 다름
  B_GEN_NB    일반 계약   · 근거 없음                    ← x311 E_NOBASIS 재현
  C_GEN_NB_NONAME  일반 계약 · 근거 없음 · **손님 발화에서 도구명 제거**
                                                        ← 이름 출처를 끊는다(누수 (iii))
  D_GEN_BASIS 일반 계약   · 근거 있음(전체 원장)          ← 양성 통제(D_GEN75F 재현·사이트 생존)
  E_DOM_BASIS 도메인 계약 · 근거 있음(전체 원장)          ← (i) 의 근거-있음 짝

판정(사전 고정):
  A_DOM_NB ≈ B_GEN_NB (차 ≤1)      → 계약은 날조의 인자가 **아니다** ⇒ 일반 계약 유지
  B_GEN_NB − A_DOM_NB ≥3           → 일반화가 날조를 산다 ⇒ **도메인 필드 복원**
  C_GEN_NB_NONAME ≤1 ∧ B_GEN_NB ≥2 → 2/8 의 정체는 날조가 아니라 **손님이 댄 이름의 반복**
                                      ⇒ x311 E 는 통제 위반이 아님(계기 누수) ⇒ 일반 계약 유지
  D_GEN_BASIS ≤5                   → 사이트가 죽었다 = 이 판 전체 무효(재설계)

계기(x311 보다 엄격·둘 다 인쇄해 비교 가능하게 남긴다):
  loose  x311 `hit()` 축자                 — 도구명 ∧ ("arguments"|"account_id"|"user_id")
  strict 도구명 ∧ **비어 있지 않은 인자 값이 실제로 있음**(형식 낱말만으론 적중 아님)
판정은 strict 로 한다. loose 는 x311 과의 대조를 위해 병기한다.

실행(리모트·8141): T2_PROBE_URL=http://localhost:8141/v1/chat/completions \
  /home/woori/venvs/seka_env/bin/python x312_nobasis_attribution_iso.py [N]
"""
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

from x216_read_and_offset import chat                             # noqa: E402
import x302_kindmap_ownership_probe as K                          # noqa: E402
import x308_write_isolate_iso as S                                # noqa: E402
import x311_generic_contract_iso as G                             # noqa: E402


def strict_hit(r, tool):
    """도구명 ∧ **비어 있지 않은 인자 값**. GEN_JSON 이 제공하는 낱말만으론 적중이 아니다.

    JSON 이 깨져 파싱이 안 되면 정규식으로 `"key": <값>` 한 쌍이라도 실재하는지만 본다
    (값의 내용은 판단하지 않는다 — 이 프로브는 '무엇을 지어냈나'가 아니라 '지어냈나'만 센다)."""
    t = str(r.get("content") or "")
    if tool not in t:
        return False
    obj = None
    try:
        m = re.search(r"\{.*\}", t, re.S)
        obj = json.loads(m.group(0)) if m else None
    except Exception:
        obj = None
    if isinstance(obj, dict) and isinstance(obj.get("calls"), list):
        # 파싱이 된 경우에만 그 결과가 **권위**다(기권 `{"calls": []}` 을 정직하게 0으로 센다).
        for c in obj["calls"]:
            if isinstance(c, dict) and c.get("tool") == tool:
                a = c.get("arguments")
                if isinstance(a, dict) and any(
                        v not in (None, "", [], {}) for v in a.values()):
                    return True
        return False
    # 산문/깨진 JSON — 이름 뒤에 실제 인자 값 한 쌍이라도 있으면 착수로 센다(안전측: 놓치지 않음).
    body = t[t.index(tool):]
    return bool(re.search(r'"[A-Za-z_]+"\s*:\s*("[^"]+"|\d)', body))


def strip_name(txt, tool):
    """손님 발화에서 **도구명만** 지운다(다른 낱말은 건드리지 않는다·자리표시자로 대체)."""
    return txt.replace(tool, "the tool they mentioned")


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 8
    s75 = G.sim_of(G.TAG75, G.TASK75)
    c75 = G.cut75(s75)
    if c75 is None:
        print("075 컷 없음 — 중단")
        return
    note75 = K.NOTE_T % G.TOOL75
    print("x312 · 075 cut=%d · n=%d · URL=%s\n"
          % (c75, n, os.environ.get("T2_PROBE_URL", "8140⚠")))

    asks = G.asks(s75, c75)
    basis = G.lines(s75, c75, whole=True)

    def body(fmt, with_basis, noname=False):
        a = [strip_name(x, G.TOOL75) for x in asks] if noname else list(asks)
        note = (K.NOTE_T % "the tool they mentioned") if noname else note75
        return "\n".join([S.SUB_HEAD, ""] + a + [""]
                         + (basis if with_basis else []) + ["", note, fmt])

    arms = (
        ("A_DOM_NB", body(S.SUB_JSON, False)),
        ("B_GEN_NB", body(G.GEN_JSON, False)),
        ("C_GEN_NB_NONAME", body(G.GEN_JSON, False, noname=True)),
        ("D_GEN_BASIS", body(G.GEN_JSON, True)),
        ("E_DOM_BASIS", body(S.SUB_JSON, True)),
    )
    res = {}
    for label, b in arms:
        ks = kl = 0
        cnt = collections.Counter()
        for i in range(n):
            try:
                r = chat(b, None, 0.0 if i == 0 else 0.7, 1500)
            except Exception as e:
                r = {"content": "ERR %s" % type(e).__name__}
            sh = strict_hit(r, G.TOOL75)
            lh = G.hit(r, G.TOOL75)
            ks += sh
            kl += lh
            cnt["strict" if sh else ("loose-only" if lh else "none")] += 1
            print("    [%s %02d] %s" % (label, i, "HIT" if sh else ("~" if lh else "-")),
                  flush=True)
        res[label] = ks
        print("%-16s strict %d/%d · loose %d/%d · %s (본문 %d자)\n"
              % (label, ks, n, kl, n, dict(cnt), len(b)))
    print("※ 판정(사전 고정): |A_DOM_NB−B_GEN_NB|≤1 → 계약은 인자 아님(일반 계약 유지) · "
          "B−A≥3 → 도메인 필드 복원 · C_NONAME≤1∧B≥2 → 2/8 은 손님이 댄 이름의 반복(계기 누수) · "
          "D_GEN_BASIS≤5 → 사이트 사망=판 무효.")
    print("   측정치: " + " · ".join("%s=%d" % (k, v) for k, v in res.items()))


if __name__ == "__main__":
    main()
