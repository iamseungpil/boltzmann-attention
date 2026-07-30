# -*- coding: utf-8 -*-
"""X13 SU-MT — 대본(scripted user) 추출 + V1 selftest (MT_PROBE_DESIGN rev2 §4·2026-07-30 야간).

대본원 = `bank_day{6,8,9c}front{A,B}_*.results.json.gz` = **우리 스택 궤적**(설계서 §1-b 반증:
`*_A/_B.log.gz`와 `*front{A,B}.results.json.gz`는 같은 런의 두 채널이다).

한 sim에서 뽑는 것:
  · 대본 = user 메시지 content를 **원 순서대로**
  · 문맥 길이 프록시 = tool 출력 총 문자수(축 2의 "장문 지점" 선별용)
  · task_id · sim id · 유저 턴 수

⚠이 모듈은 **추출만** 한다. env 실행·모델 호출은 `x13_su_mt_probe.py`가 맡는다(분리 = V1을
모델 없이 돌리기 위해서다).

V1(발사 전 검증): 추출한 대본의 **순서·개수가 원 궤적과 일치**하는지 독립 경로로 재확인.
용법: py -3 x13_su_mt_cases.py            # V1 selftest + 가용량 표
      py -3 x13_su_mt_cases.py --json out.json --top 12
"""
import argparse
import glob
import gzip
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_SIM = os.path.abspath(os.path.join(_HERE, "..", "..", "..",
                                    "reports", "facet_rft_2026", "sim_results"))
# 우리 스택 + 궤적이 둘 다 있는 런만(설계서 §4). frontier 비교군·smoke는 제외.
GLOBS = ["bank_day6front[AB]_*.results.json.gz",
         "bank_day8front[AB]_*.results.json.gz",
         "bank_day9cfront[AB]_*.results.json.gz"]


def _msgs(sim):
    return sim.get("messages") or []


def _content(m):
    c = m.get("content")
    return c if isinstance(c, str) else ""


def extract(sim, source):
    """한 sim → 케이스 dict. 대본은 user 메시지 content의 원 순서."""
    ms = _msgs(sim)
    script = [_content(m) for m in ms if m.get("role") == "user"]
    tool_chars = sum(len(_content(m)) for m in ms if m.get("role") == "tool")
    return {
        "source": source,
        "sim": str(sim.get("id") or "")[:12],
        "task_id": sim.get("task_id"),
        "n_user": len(script),
        "n_msg": len(ms),
        "tool_chars": tool_chars,          # 장문 조건 프록시(축 2 선별 기준)
        "script": script,
    }


def load_cases():
    cases = []
    for g in GLOBS:
        for path in sorted(glob.glob(os.path.join(_SIM, g))):
            try:
                d = json.load(gzip.open(path, "rt", encoding="utf-8"))
            except Exception as e:
                print("  ⚠읽기 실패 %s: %r" % (os.path.basename(path), e))
                continue
            for sim in d.get("simulations") or []:
                c = extract(sim, os.path.basename(path))
                if c["n_user"] >= 2:        # 대본이 1턴이면 다중턴이 아니다
                    cases.append(c)
    return cases


def v1_selftest(cases):
    """V1 — 추출 순서·개수가 원 궤적과 일치하는지 **독립 경로**로 재확인.

    독립성: `extract`는 role 필터 + 순차 append를 쓴다. 검증은 원본을 다시 열어
    **인덱스 기반**으로 user 위치를 뽑아 대조한다(같은 코드 경로를 재사용하지 않는다).
    """
    by_src = {}
    for c in cases:
        by_src.setdefault(c["source"], []).append(c)
    checked = mismatch = 0
    for src, cs in by_src.items():
        d = json.load(gzip.open(os.path.join(_SIM, src), "rt", encoding="utf-8"))
        sims = {str(s.get("id") or "")[:12]: s for s in (d.get("simulations") or [])}
        for c in cs:
            s = sims.get(c["sim"])
            if s is None:
                mismatch += 1
                print("  ✗ sim 소실 %s/%s" % (src, c["sim"]))
                continue
            ms = _msgs(s)
            idx = [i for i, m in enumerate(ms) if m.get("role") == "user"]
            ref = [_content(ms[i]) for i in idx]
            checked += 1
            if ref != c["script"]:
                mismatch += 1
                print("  ✗ 대본 불일치 %s/%s (원 %d vs 추출 %d)"
                      % (src, c["sim"], len(ref), len(c["script"])))
            if sorted(idx) != idx:
                mismatch += 1
                print("  ✗ 순서 이상 %s/%s" % (src, c["sim"]))
    return checked, mismatch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    ap.add_argument("--top", type=int, default=0, help="tool_chars 상위 N개만 저장(축 2 장문 선별)")
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    cases = load_cases()
    if not cases:
        sys.exit("케이스 0 — GLOBS/경로 확인")

    srcs = sorted({c["source"] for c in cases})
    tasks = sorted({c["task_id"] for c in cases})
    print("케이스 %d · 원본 파일 %d · 서로 다른 task_id %d" % (len(cases), len(srcs), len(tasks)))
    uts = sorted(c["n_user"] for c in cases)
    print("유저턴: 합 %d · min %d · median %d · max %d"
          % (sum(uts), uts[0], uts[len(uts) // 2], uts[-1]))
    tc = sorted(c["tool_chars"] for c in cases)
    print("도구출력 문자수(장문 프록시): median %d · max %d" % (tc[len(tc) // 2], tc[-1]))
    print("\n원본별:")
    for s in srcs:
        cs = [c for c in cases if c["source"] == s]
        print("  %-46s sims %2d · 유저턴 %3d" % (s[:46], len(cs), sum(c["n_user"] for c in cs)))

    print("\n=== V1 selftest (대본 순서·개수 대조·독립 경로) ===")
    checked, mismatch = v1_selftest(cases)
    print("  대조 %d건 · 불일치 %d건 → %s" % (checked, mismatch, "PASS" if mismatch == 0 else "FAIL"))

    # ★표본 상관 정직 보고(설계서 §4): 케이스 수와 **궤적/태스크 수**를 둘 다 낸다.
    print("\n⚠표본 상관: 케이스 %d개는 서로 다른 task %d개에서 나온다(중복 태스크는 런 간 재시행)."
          % (len(cases), len(tasks)))

    if args.json:
        out = sorted(cases, key=lambda c: -c["tool_chars"])
        if args.top:
            out = out[:args.top]
        json.dump(out, open(args.json, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print("\n[saved] %s (%d 케이스%s)"
              % (args.json, len(out), " · tool_chars 상위" if args.top else ""))
    if mismatch:
        sys.exit(1)


if __name__ == "__main__":
    main()
