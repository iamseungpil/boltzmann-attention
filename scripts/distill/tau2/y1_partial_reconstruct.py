#!/usr/bin/env python3
"""Y1 부분 수확 — 진행 로그의 Status 라인에서 per-sim 결과를 역산 (2026-07-30).

배경: tau2 러너는 `results.json`을 **런 종료 시에만** 쓴다. Y1 본런이 장기 sim 2건에 슬롯을
물려 정체됐고(task_022 77분·task_025 55분) 완주 ETA가 불확실해, 사용자 지시로 부분 수확한다.
문제는 로그에 per-sim 기록이 없다는 것 — 러닝 평균만 찍힌다.

★역산 원리: Status 라인이 (N완료, avg, running 집합)을 준다.
  ① N이 N→N+1 될 때 **running에서 빠진 태스크** = 방금 완료된 sim
  ② `round(avg × N)`의 증분 = 그 sim의 pass(0/1)
     — reward가 0/1이므로 avg×N은 pass 개수. avg가 2dp 반올림이라 오차 ≤ 0.005×N ≤ 0.16(N≤32)
       이므로 round()가 정수 pass 수를 **정확히** 복원한다.
  ③ 한 블록에서 2건이 동시에 빠지면 개별 귀속 불가 → `ambig` 표시([[08]] 무음 추정 금지).

Y1의 목적은 점수가 아니라 **동일 스택 nt=2 편차 폭(판정 임계)**이므로, 핵심 산출은
**두 trial이 모두 끝난 태스크 쌍**의 flip 여부다. 쌍이 아닌 단일 trial은 편차 표본이 아니다.

입력 = Status 블록 덤프(한 줄 1블록). 생성 =
  tr '\\n' '\\r' < y1main.log | grep -oE 'Status: ...' | sed 's/\\r/ /g'
용법: py -3 y1_partial_reconstruct.py <status_dump.txt> [--json out.json]
"""
import argparse
import json
import re
import sys

_STATUS = re.compile(r"Status:\s*(\d+)/(\d+) complete\. Avg reward:\s*([0-9.]+)\s*\(N=(\d+)\)")
_RUN = re.compile(r"(task_\d+)\.(\d+)\((\d+)s\)")


def parse(path):
    seq = []
    for line in open(path, encoding="utf-8", errors="replace"):
        m = _STATUS.search(line)
        if not m:
            continue
        n, tot, avg, nn = int(m.group(1)), int(m.group(2)), float(m.group(3)), int(m.group(4))
        if n != nn:                                   # N과 완료수 불일치 = 파싱 이상
            continue
        # running 목록은 Status 뒤 ~200자 안. 그 뒤 로그 노이즈의 task_ 언급을 배제하기 위해
        # "N running:" 이후 구간만 본다.
        tail = line[m.end():]
        rm = re.search(r"(\d+) running:", tail)
        run = {}
        if rm:
            seg = tail[rm.end():rm.end() + 260]
            for tid, tr, sec in _RUN.findall(seg):
                run[f"{tid}.{tr}"] = int(sec)
        seq.append({"n": n, "total": tot, "avg": avg, "running": run})
    return seq


def reconstruct(seq):
    done, prev = [], None
    for cur in seq:
        if prev is not None and cur["n"] > prev["n"]:
            left = [t for t in prev["running"] if t not in cur["running"]]
            gained = round(cur["avg"] * cur["n"]) - round(prev["avg"] * prev["n"])
            step = cur["n"] - prev["n"]
            for t in left:
                done.append({"sim": t, "dur_s": prev["running"][t],
                             "passed": (bool(gained) if (len(left) == 1 and step == 1) else None),
                             "ambig": not (len(left) == 1 and step == 1),
                             "gained_in_block": gained, "left_in_block": len(left)})
        prev = cur
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("--json")
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    seq = parse(args.dump)
    if not seq:
        sys.exit("Status 블록 0 — 덤프 형식 확인")
    last = seq[-1]
    print(f"Status 블록 {len(seq)}개 · 최종 {last['n']}/{last['total']} "
          f"avg={last['avg']} (pass≈{round(last['avg'] * last['n'])})")
    print(f"현재 실행 중: {last['running']}")
    print()

    done = reconstruct(seq)
    amb = [d for d in done if d["ambig"]]
    print(f"=== 역산된 완료 sim {len(done)}개 (최종 N={last['n']}) ===")
    if len(done) != last["n"]:
        print(f"⚠️역산 {len(done)} ≠ 보고 N {last['n']} — 첫 블록 이전 완료분은 running 관측이 "
              f"없어 귀속 불가(정상). 아래는 관측 구간분만.")
    print(f"{'sim':16s} {'초':>7s}  결과")
    for d in done:
        r = "?귀속모호" if d["passed"] is None else ("PASS" if d["passed"] else "fail")
        print(f"{d['sim']:16s} {d['dur_s']:7d}  {r}"
              + (f"  (동시종료 {d['left_in_block']}·블록증분 {d['gained_in_block']})"
                 if d["ambig"] else ""))
    if amb:
        print(f"\n⚠️귀속 모호 {len(amb)}건 — 한 Status 블록에서 2건 이상 동시 종료. "
              f"개별 pass를 추정하지 않는다([[08]]).")

    by = {}
    for d in done:
        tid, tr = d["sim"].rsplit(".", 1)
        by.setdefault(tid, {})[tr] = d
    pairs = {k: v for k, v in by.items() if len(v) >= 2}
    print()
    print("=" * 70)
    print(f"★nt=2 완결 쌍 {len(pairs)}개 = 편차 폭의 **유효 표본** (Y1의 목적)")
    print("=" * 70)
    flips = agree = unk = 0
    for k, v in sorted(pairs.items()):
        ps = [v[t]["passed"] for t in sorted(v)]
        ds = [v[t]["dur_s"] for t in sorted(v)]
        if None in ps:
            verdict, unk = "?(모호 포함)", unk + 1
        elif len(set(ps)) > 1:
            verdict, flips = "★FLIP", flips + 1
        else:
            verdict, agree = ("일치(둘 다 PASS)" if ps[0] else "일치(둘 다 fail)"), agree + 1
        show = ["PASS" if p else ("fail" if p is False else "?") for p in ps]
        print(f"  {k}: {show}  {verdict}   dur={ds}")
    print()
    print(f"판정: FLIP {flips} · 일치 {agree} · 모호 {unk}  (쌍 {len(pairs)})")
    if flips + agree:
        print(f"⇒ **동일 스택 nt=2 flip률 = {flips}/{flips + agree} "
              f"= {100.0 * flips / (flips + agree):.0f}%** (모호 제외)")
    print("⚠️[[08]]: 이 수는 *판정 임계*를 세우는 용도다. 점수(avg)로 스택 성능을 읽지 말 것.")
    single = sorted(set(by) - set(pairs))
    print(f"\n단일 trial만 완료(편차 표본 아님) {len(single)}: {single}")

    if args.json:
        json.dump({"final": last, "done": done, "pairs": {k: {t: v[t] for t in v}
                                                          for k, v in pairs.items()},
                   "flips": flips, "agree": agree, "ambiguous": unk},
                  open(args.json, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"\n[saved] {args.json}")


if __name__ == "__main__":
    main()
