# -*- coding: utf-8 -*-
"""x389 — **구현·측정은 됐는데 선언이 없어 꺼져 있는 레버** 전수 조사.

사용자 지시(2026-08-18 축자): *"이미 구현되고 실험되어서 효과 확인된 것중에 이렇게 선언되지
않아서 꺼져 있는 레버들이 있는지 전수 조사하라"*

발단 = `T2_PROV_OURS`: 2026-08-06 에 구현되고 정본 설계서(`CONFLICT_ARBITRATION_THEORY §3-T3`)까지
있는데 **어느 런처에도 없어** 한 번도 켜진 적이 없었다. t7320 에서 그 대가가 실측됐다 —
읽기 루틴이 지목한 이름을 같은 층의 출처 가드가 `operator-fab` 으로 막았다.

## 방법 (결정론·LLM 0)
  ⑴ `t2_levers.audit_unset()` = 코드가 읽는데 **셀에도 런처에도 없는** 플래그
  ⑵ 각 플래그를 **원장**(RESEARCH_MASTER §3)과 **설계서 폴더** 전문에서 찾는다
  ⑶ 원장에 등급 표식([S]/[M])이 같은 항목 안에 있으면 *측정 흔적 있음*으로 분류
  ⚠판정이 아니라 **후보 목록**이다 — 등급의 방향(양성/음성/철회)은 사람이 읽어야 한다.
     원장에 이름이 있다고 효과가 양성이라는 뜻이 아니다([[60]]: 끄지 마라 ≠ 전부 켜라).
"""
import io
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import glob

import t2_levers as L

HERE = os.path.dirname(os.path.abspath(__file__))
REPORTS = os.path.abspath(os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026"))
MASTER = os.path.join(REPORTS, "RESEARCH_MASTER.md")


def ledger_hits(flag, text):
    """그 플래그를 언급한 원장 항목들 — (C번호, 등급표식, 발췌)."""
    out = []
    for m in re.finditer(re.escape(flag), text):
        a = text.rfind("| **C", 0, m.start())
        if a < 0:
            a = max(0, m.start() - 400)
        b = text.find("\n", m.end())
        seg = text[a:b if b > 0 else m.end() + 300]
        cid = re.match(r"\| \*\*(★*C\d+)\*\*", seg)
        grades = sorted(set(re.findall(r"\[[SMD?]\]", seg)))
        out.append((cid.group(1) if cid else "?", ",".join(grades) or "-",
                    " ".join(seg[:160].split())))
    return out


def main():
    lch = sorted(glob.glob(os.path.join(HERE, "run_t7*.sh"))) + \
        [os.path.join(HERE, "go_stack.sh"), os.path.join(HERE, "run_one.sh")]
    rows = L.audit_unset(launchers=lch)
    master = io.open(MASTER, encoding="utf-8").read() if os.path.exists(MASTER) else ""
    docs = {}
    for p in glob.glob(os.path.join(REPORTS, "*.md")):
        try:
            docs[os.path.basename(p)] = io.open(p, encoding="utf-8", errors="replace").read()
        except Exception:
            pass

    print("셀에도 런처에도 없는 플래그: %d" % len(rows))
    print("원장=RESEARCH_MASTER 언급 · 설계서=reports/*.md 언급\n")
    graded, named, silent = [], [], []
    for flag, src in rows:
        hits = ledger_hits(flag, master)
        dsn = sorted(n for n, t in docs.items() if flag in t and n != os.path.basename(MASTER))
        if any(h[1] not in ("-",) for h in hits):
            graded.append((flag, src, hits, dsn))
        elif hits or dsn:
            named.append((flag, src, hits, dsn))
        else:
            silent.append((flag, src))

    print("=" * 78)
    print("① 원장에 **등급 표식과 함께** 이름이 있는 것 — 측정 흔적 있음 (%d)" % len(graded))
    print("=" * 78)
    for flag, src, hits, dsn in graded:
        print("\n★ %-26s (%s)" % (flag, src))
        for cid, g, seg in hits[:3]:
            print("   %-8s %-10s %s" % (cid, g, seg[:130]))
        if dsn:
            print("   설계서: %s" % ", ".join(dsn[:3]))

    print("\n" + "=" * 78)
    print("② 이름은 있으나 등급 표식 없음 (%d)" % len(named))
    print("=" * 78)
    for flag, src, hits, dsn in named:
        where = ("원장 %d건" % len(hits)) if hits else ""
        where += (" · 설계서 %s" % ", ".join(dsn[:2])) if dsn else ""
        print("  %-28s %s" % (flag, where))

    print("\n" + "=" * 78)
    print("③ 원장에도 설계서에도 이름이 없음 = 근거 미상 (%d)" % len(silent))
    print("=" * 78)
    print("  " + ", ".join(f for f, _ in silent))
    return 0


if __name__ == "__main__":
    sys.exit(main())
