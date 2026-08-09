r"""x167 — `'L'` 의 정체를 닫고, **순위 전체가 가족 편향인지** 본다 (유료 0·§8.3⒝).

## 왜

붕괴 행의 1위가 1글자 토큰 `'L'` 이라 Light Blue / Light Green / Lime Green 중 무엇인지
미해결이었다(§6.1 각주·§8.3⒝). 그리고 이것은 지금 **기전 판정에 직접 걸린다** — §6.3 은
정박이 $Y$ 를 **가족**으로 좁힌다고 하는데, 그렇다면 argmax 뿐 아니라 **순위 전체가 그 가족에
쏠려야** 한다. 어제 뜬 j=27 원본은 `'L' .967 · Sky .0126 · World .0118 · Hunter .0086` 로
2·3위가 **Blue 가족**이라 가족 지배가 안 보인다 — 단 1위의 정체가 미정이라 판정이 걸려 있다.

⚠**샘플링 온도로는 웅덩이 구조를 못 잰다**(2026-08-09 교정): `q ∝ p^(1/T)` 는 **단조**라
순서를 보존한다 ⇒ 어떤 T 에서도 argmax 가 안 바뀐다. 내가 x163 에서 *"T 무반응"* 을 결과로
보고한 것은 측정이 아니라 **항등식**이었다. β(어텐션 내부 척도)는 별개이고 white-box 가 필요하다.
이 프로브는 **β 없이** 볼 수 있는 것만 본다.

## 방법

`guided_choice` 로 출력을 후보에 강제하되 **토큰을 넉넉히** 준다 ⇒ 완성 문자열이 곧 후보
이름이다(1글자 토큰 모호성이 사라진다). 그 위에서 **가족 라벨은 A3 주어의 마지막 낱말**로
기계적으로 만든다(예: `... Green` / `... Blue` / `... Card`) — 프로브가 분류를 발명하지 않는다.

실행: T2_PROBE_URL=... T2_PROBE_MODEL=... py -3 x167_resolve_L.py [N]
"""
import collections
import json
import os
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import x149_choice_isolation as X                              # noqa: E402
import x150_choice_ablation as Y                               # noqa: E402
import x157_entrainment_lambda as P                            # noqa: E402
import t2_factdag as FD                                        # noqa: E402
import t2_ledger as LG                                         # noqa: E402
from gate_interpreter import load_domain_a2                     # noqa: E402


def guided_full(prompt, choices, temp):
    """후보에 강제하되 이름이 **끝까지** 나오게 한다 — 1글자 토큰 모호성 제거."""
    body = json.dumps({"model": P.MODEL, "temperature": temp, "max_tokens": 12,
                       "guided_choice": list(choices),
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(P.URL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        d = json.load(r)
    return " ".join((d["choices"][0]["message"]["content"] or "").split())


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    a2 = load_domain_a2("banking_knowledge")
    spec = next(s for s in a2["ledger_metrics"] if s.get("eligible_text"))
    rows = a2["policy_ontology"]["rows"]
    maps = {ax: FD._a3_map(rows, {"axis": ax}) for ax in spec["eligible"]["show_axes"]}
    table = LG.eligible_text(730, {}, maps, spec, {"qualifying_deposit_usd": 30000}).strip()
    base = table + "\n\n" + X.FACTS[P.TASK] + "\n\n" + X.QUESTION
    choices = [l.strip().split(":")[0].strip() for l in table.splitlines() if l.startswith("  ")]
    msgs = Y.msgs_of(os.environ.get("T2_PROBE_TAG", "bank_elig_20260809i"), P.TASK)

    # 가족 = 이름의 **마지막 낱말**(기계적·프로브가 분류를 발명하지 않는다)
    fam = {c: (c.split()[-1] if c.split() else "?") for c in choices}
    print("model=%s · 후보 %d · 가족 %s" % (P.MODEL, len(choices),
                                          dict(collections.Counter(fam.values()))))

    ctx = "Here is a customer-service conversation so far.\n\n"
    arms = collections.OrderedDict()
    arms["j=27 전체(정박 Hunter Green)"] = ctx + Y.render(msgs) + "\n\n" + base
    arms["j=26 (정박 직전)"] = ctx + Y.render(msgs[:len(msgs) - 1]) + "\n\n" + base
    arms["j=0 (깨끗)"] = base

    for label, prompt in arms.items():
        got = [guided_full(prompt, choices, 0.0 if i == 0 else 0.7) for i in range(n)]
        cnt = collections.Counter(got)
        famcnt = collections.Counter(fam.get(g, "?") for g in got)
        print("\n=== %s ===" % label)
        print("  이름: %s" % cnt.most_common(4))
        print("  가족: %s" % famcnt.most_common(4))
    print("\n  §6.3 가족-웅덩이 읽기가 맞으면 정박 arm 의 **가족 분포가 한 가족에 쏠려야** 한다.")
    print("  흩어지면 정박은 argmax 만 옮기고 순위 전체를 가족으로 끌지는 않는 것이다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
