# -*- coding: utf-8 -*-
"""X19b — [[23]] **결정적 검사**: 리터럴이 *gold에만* 있나 (원격 전용·도메인 데이터 필요).

x19가 만든 검토 큐는 "env·정책·KB 어디에도 없다"까지만 말한다. 그것만으로는 **gold 유래**와
**우리 scaffold가 지은 이름**을 못 가른다. 이 스크립트가 가른다:

    literal ∈ gold(task set)  ∧  ∉ 비-gold(정책·KB·DB)   →  ★[[23]] 위반 확정
    literal ∉ gold ∧ ∉ 비-gold                            →  우리 발명(scaffold 어휘·정당)
    literal ∈ 비-gold                                      →  정당한 저작

⚠**비대칭 금지**(2026-07-31 실측 교훈): env는 집합 소속으로, gold는 부분문자열로 재면
`request_human`(= env 도구 `request_human_agent_transfer`의 접두사) 같은 **접두사 패턴**이
"gold 전용"으로 **거짓 고발**된다. env 쪽도 접두사 매칭을 허용해야 한다(x19 `_env_prefix`).

용법(원격): python x19b_goldcheck_remote.py <domain> <literal> [literal ...]
"""
import glob
import os
import sys

ROOT = "/home/woori/workspace_common/boltzmann-attention/external/tau2-bench/data/tau2/domains"


def env_names(domain):
    """★4차 교정(2026-07-31): 코퍼스는 **문서만** 본다. env 도구 레지스트리는 배포 시점에 가진
    정당한 출처인데 문서에 안 적힌 도구(`apply_for_credit_card`·`submit_referral` = user-side
    실행 도구)가 있어서, 그걸 'gold 전용'으로 **거짓 고발**했다. env 소속을 먼저 본다."""
    try:
        from tau2.registry import registry
        env = registry.get_env_constructor(domain)()
    except Exception:
        return set()
    names = set()
    for kit_name in ("tools", "user_tools"):
        kit = getattr(env, kit_name, None)
        if kit is None:
            continue
        for a in dir(kit):
            if a.startswith("_"):
                continue
            try:
                if kit.has_tool(a):
                    names.add(a)
            except Exception:
                pass
    return names


def corpora(domain):
    gold, other = [], []
    for p in glob.glob(os.path.join(ROOT, domain, "**", "*"), recursive=True):
        if not os.path.isfile(p):
            continue
        try:
            txt = open(p, encoding="utf-8", errors="replace").read()
        except Exception:
            continue
        ("task" in p.lower() and gold or other).append(txt)
    return "\n".join(gold), "\n".join(other)


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(2)
    domain, lits = sys.argv[1], sys.argv[2:]
    gold, other = corpora(domain)
    envn = env_names(domain)
    print("gold %d자 · 비-gold %d자 · env 도구 %d개" % (len(gold), len(other), len(envn)))
    print("%-36s %-6s %-7s %-5s %s" % ("literal", "gold", "비-gold", "env", "판정"))
    bad = 0
    for t in lits:
        g, n = t in gold, t in other
        e = (t in envn) or any(x.startswith(t) or t.startswith(x) for x in envn if len(t) >= 5)
        if e:
            v = "env 레지스트리 실재(정당)"
        elif g and not n:
            v = "★★gold 전용 = [[23]] 위반"
            bad += 1
        elif n:
            v = "비-gold에 있음(정당)"
        else:
            v = "어디에도 없음 = 우리 발명(scaffold 어휘)"
        print("%-36s %-6s %-7s %-5s %s" % (t, g, n, e, v))
    print("\n위반 %d건" % bad)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
