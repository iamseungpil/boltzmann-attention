# -*- coding: utf-8 -*-
r"""x265 — **어느 레버가 격리 근거를 갖고 있나** (감사 · 유료 0 · 모델 0 · GPU 0).

## 왜 (사용자 지시 2026-08-11: *"엔진이 해야 된다던 부분을 엄밀히 격리한 적 없지 않나"*)

오늘 하루가 그 필요를 증명했다 — 070_4 에 대해 내가 내놓은 진단 **넷이 전부** 측정에 지워졌고,
마지막에는 **내가 손으로 쓴 프롬프트의 0/8** 을 모델 무능으로 읽어 하마터면 **채점 칸을 가져가는
엔진 필터**를 지을 뻔했다(C432⒝). 그런데 **어느 부품이 격리 근거를 가졌는지 목록이 없다.**

⛔0([[62]])은 **2026-08-09 에야** 섰다. 그 이전 부품 대부분은 근거 없이 들어갔을 수 있고,
그중 **기본 ON** 인 것이 있으면 지금도 라이브에서 돌고 있다.

## 무엇을 세나

  ① 엔진이 읽는 `T2_*` 플래그 전수 (`os.environ.get`)
  ② 그중 `go_stack.sh` 가 **기본으로 켜는 것**
  ③ 각각에 대해 **격리 근거**가 있는가:
       [S] 프로브 파일이 그 플래그를 이름으로 다루고 **원장 행이 셀 수(n/n)를 인용**
       [M] 원장에 이름은 나오나 프로브 파일 없음
       [?] 둘 다 없음
  ④ **기본 ON ∧ [?]** = 지금 라이브에서 도는데 근거가 없는 것 ← 이 감사의 표적

⚠판정은 **문자열 기준**이라 등급 [M] 이다(x247·x255 와 같다). 수는 자리를 가리킬 뿐이고,
  고칠지는 사람이 축자를 읽고 정한다. **"발화 0 = 위반 0" 이 아니다.**

실행: python x265_lever_evidence_audit.py
"""
import glob
import io
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
# .../ba-frft/scripts/distill/tau2 에서 세 번 올라가면 ba-frft
LEDGER = os.path.normpath(os.path.join(HERE, "..", "..", "..",
                                       "reports", "facet_rft_2026", "RESEARCH_MASTER.md"))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def read(p):
    try:
        return io.open(p, encoding="utf-8", errors="replace").read()
    except Exception:
        return ""


def main():
    engine = "".join(read(os.path.join(HERE, f)) for f in
                     ("t2_gate_patch.py", "t2_eplan_patch.py", "t2_search.py",
                      "t2_dominance.py", "t2_source.py", "t2_precedence.py"))
    flags = sorted(set(re.findall(r'os\.environ\.get\(\s*"(T2_[A-Z0-9_]+)"', engine)))

    stack = read(os.path.join(HERE, "go_stack.sh"))
    on = set()
    for m in re.finditer(r"^\s*export\s+(T2_[A-Z0-9_]+)=([^\s#]+)", stack, re.M):
        if m.group(2).strip('"\'') == "1":
            on.add(m.group(1))

    ledger = read(LEDGER)
    rows = [l for l in ledger.split("\n") if l.startswith("| **C")]
    # ★양성 대조 (C428): 계기의 실패가 **음성 관측과 구별되지 않는다**. 첫 판이 원장 경로를
    #   틀려 빈 문자열을 읽었고, 그것이 *"175개 전부 근거 없음"* 으로 나왔다 — 그럴듯한 결론이라
    #   하마터면 그대로 보고할 뻔했다. 알려진 값이 잡히는지 먼저 확인하고, 안 잡히면 **중단**한다.
    known = ("T2_CALL_FORM", "T2_ARG_EMPTY")
    miss = [k for k in known if not any(k in r for r in rows)]
    if not rows or miss:
        print("중단 — 감사기가 원장을 못 읽는다: %s · 행 %d · 못 찾은 표지 %s"
              % (LEDGER, len(rows), miss), file=sys.stderr)
        return 2
    print("원장 %d행 · 양성 대조 통과(%s)\n" % (len(rows), ", ".join(known)))
    probes = {os.path.basename(p): read(p) for p in glob.glob(os.path.join(HERE, "x*.py"))}
    tests = {os.path.basename(p): read(p) for p in glob.glob(os.path.join(HERE, "test_*.py"))}

    cell = re.compile(r"\b\d+/\d+\b")
    out = []
    for f in flags:
        cited = [r for r in rows if f in r]
        has_cells = any(cell.search(r) for r in cited)
        # ⚠자기 자신과 감사기류는 근거가 아니다(이름만 나온다) — 첫 판이 `x265` 를 근거로 셌다.
        pfiles = sorted(k for k, v in probes.items()
                        if f in v and not k.startswith(("x265", "x257", "x247", "x255")))
        tfiles = sorted(k for k, v in tests.items() if f in v)
        if pfiles and has_cells:
            grade = "[S]"
        elif cited:
            grade = "[M]"
        else:
            grade = "[?]"
        out.append((f, f in on, grade, pfiles, tfiles, len(cited)))

    n_on = sum(1 for r in out if r[1])
    print("엔진이 읽는 T2_* 플래그 **%d** · `go_stack.sh` 기본 ON **%d**\n" % (len(flags), n_on))
    for g in ("[S]", "[M]", "[?]"):
        sel = [r for r in out if r[2] == g]
        print("%s %d개 (기본 ON %d)" % (g, len(sel), sum(1 for r in sel if r[1])))
    print("\n" + "=" * 78)
    print("★표적 — **기본 ON 인데 격리 근거가 없다** ([?] · 지금 라이브에서 돈다)")
    print("=" * 78)
    bad = [r for r in out if r[1] and r[2] == "[?]"]
    for f, _o, _g, _p, tf, _c in sorted(bad):
        print("  %-34s 회귀 %s" % (f, ",".join(tf) or "**없음**"))
    print("  (%d개)" % len(bad))

    print("\n" + "-" * 78)
    print("기본 ON · 원장 언급은 있으나 프로브 없음 ([M])")
    print("-" * 78)
    for f, _o, g, _p, _t, c in sorted(r for r in out if r[1] and r[2] == "[M]"):
        print("  %-34s 원장 %d행" % (f, c))

    print("\n" + "-" * 78)
    print("근거 있는 것 ([S] · 기본 ON 여부 표시)")
    print("-" * 78)
    for f, o, g, p, _t, _c in sorted(r for r in out if r[2] == "[S]"):
        print("  %-34s %s  %s" % (f, "ON " if o else "off", ",".join(p[:2])))

    print("\n※ 문자열 기준 = [M]. 이 감사가 보는 것: 플래그 이름이 프로브/원장에 나오는가와 "
          "원장 행에 셀 수가 있는가.\n  **보지 않는 것**: 그 프로브가 *이 플래그를* 정말 잰 것인지, "
          "셀 수가 그 주장에 대한 것인지, 부정 통제가 있었는지.\n  ⇒ [S] 는 *후보*이지 확정이 "
          "아니다. [?] 가 **기본 ON** 인 것부터 사람이 읽는다.")


if __name__ == "__main__":
    main()
