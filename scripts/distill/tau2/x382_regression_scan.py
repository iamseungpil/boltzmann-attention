# -*- coding: utf-8 -*-
r"""x382 — **회귀 태스크 전수 탐지**(사용자 지시 2026-08-18 · 무료 · CPU · LLM 0 · GPU 0).

## 왜

`task_098` 이 **같은 시드(626729)에서 6런 연속 pass → 4런 연속 fail** 로 떨어졌다(08-17 04:10 →
08-18 11:29 사이). 양팔 모두 죽었으니 S2 레버가 아니라 **공유 스택 변경**이다. 그리고 나는
`sim_results/*.gz` 만 보고 *"098 은 통과한 적 없다"* 고 잘못 말했다 — **조회 범위를 결과로
착각**했다. 원본은 리모트 `data/simulations/*` 883 런이다.

⇒ 같은 일이 **다른 태스크에도** 일어났는지 전수로 본다. 대상 = staged census 의 **20 태스크**
(스모크 4 + 1단계 16).

## 방법 (결정론 · 판단 0)

  · 런 = `results.json` 의 **mtime** 으로 시각을 잡는다(태그 문자열 파싱 금지 — 이름 규칙이 여러 벌).
  · **PRE** = `--since`(기본 08-14 00:00) ~ **컷**(기본 08-18 09:19 = 회귀 커밋 `a627a18b`)
    **POST** = 컷 이후
  · 태스크마다 ⑴전체 pass 율 ⑵**시드 맞춘** pass 율(같은 시드가 양쪽에 있을 때만)을 낸다.
  · 판정은 **시드 맞춘 쪽으로만** 한다 — 시드가 다르면 난이도가 다르다(098 이 그 함정이었다).

## 판정 (사전 고정)

    시드-맞춤 PRE 전부 pass ∧ POST 전부 fail   → **회귀**(REGRESS)
    시드-맞춤 PRE 전부 fail ∧ POST 전부 pass   → 개선(IMPROVE)
    양쪽 섞임                                  → 불변/노이즈(MIXED)
    한쪽에 자료 없음                            → 판정 불가(NODATA) — 없는 것을 있다고 하지 않는다

사용(리모트): /home/woori/venvs/seka_env/bin/python x382_regression_scan.py [--since MMDD] [--cut MMDDHHMM]
"""
import io
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = "/home/woori/scratch/tau2-bench/data/simulations"
ROSTER = os.path.normpath(os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026",
                                       "S3_STAGED_ROSTERS_2026_08_18.json"))


def arg(name, default):
    for a in sys.argv[1:]:
        if a.startswith(name + "="):
            return a.split("=", 1)[1]
    return default


def stamp(mmddhhmm, year=2026):
    return time.mktime(time.strptime("%d%s" % (year, mmddhhmm), "%Y%m%d%H%M"))


def roster():
    """20 태스크 = 스모크 4 + 1단계 16 (명부 정본에서 읽는다 · 하드코딩 금지)."""
    try:
        d = json.load(io.open(ROSTER, encoding="utf-8"))
    except Exception as e:
        print("⚠명부 정본을 못 읽었다(%r) — 중단" % (e,))
        return []
    # ⚠명부는 번호를 **맨숫자**(`"003"`)로 적는다 — 궤적의 `task_id` 는 `task_003` 이다.
    #   `stage1` 이 스모크 4를 이미 포함한 **20** 이다(핸드오프 §3: 되돌려 포함).
    out = []
    for k in ("smoke", "stage1"):
        v = d.get(k) or d.get(k + "_tasks") or []
        if isinstance(v, dict):
            v = v.get("tasks") or []
        out += [(t if str(t).startswith("task_") else "task_%s" % t) for t in v]
    if not out:                      # 구조가 다르면 모양으로 (task_NNN 문자열 전수)
        def walk(o):
            if isinstance(o, str) and o.startswith("task_"):
                out.append(o)
            elif isinstance(o, dict):
                for x in o.values():
                    walk(x)
            elif isinstance(o, list):
                for x in o:
                    walk(x)
        walk(d)
    seen, uniq = set(), []
    for t in out:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    return uniq


def main():
    since = stamp(arg("--since", "08140000"))
    cut = stamp(arg("--cut", "08180919"))
    tasks = roster()
    if not tasks:
        return 1
    print("=" * 100)
    print("x382 · 회귀 전수 탐지 · 대상 %d 태스크 · PRE %s~컷 · 컷 %s"
          % (len(tasks), time.strftime("%m-%d %H:%M", time.localtime(since)),
             time.strftime("%m-%d %H:%M", time.localtime(cut))))
    print("판정(사전 고정): **시드 맞춘** PRE 전부 pass ∧ POST 전부 fail → 회귀 · 반대면 개선 · "
          "섞이면 MIXED · 한쪽 없으면 NODATA")
    print("=" * 100)

    want = set(tasks)
    pre, post = {}, {}          # task -> {seed: [reward…]}
    runs_pre, runs_post = set(), set()
    for d in sorted(os.listdir(ROOT)):
        p = os.path.join(ROOT, d, "results.json")
        if not os.path.exists(p):
            continue
        mt = os.path.getmtime(p)
        if mt < since:
            continue
        try:
            doc = json.load(io.open(p, encoding="utf-8"))
        except Exception:
            continue
        bucket = pre if mt < cut else post
        (runs_pre if mt < cut else runs_post).add(d)
        for s in (doc.get("simulations") or doc.get("results") or []):
            t = str(s.get("task_id") or "")
            if t not in want:
                continue
            rw = (s.get("reward_info") or {}).get("reward")
            bucket.setdefault(t, {}).setdefault(str(s.get("seed")), []).append(
                1 if (rw or 0) >= 1.0 else 0)

    print("런 — PRE %d · POST %d" % (len(runs_pre), len(runs_post)))
    print("")
    hdr = "%-10s %-16s %-16s %-22s %s"
    print(hdr % ("task", "PRE 전체", "POST 전체", "시드-맞춤(PRE→POST)", "판정"))
    print("-" * 100)
    verdicts = {}
    for t in tasks:
        a, b = pre.get(t) or {}, post.get(t) or {}
        pa = [x for v in a.values() for x in v]
        pb = [x for v in b.values() for x in v]
        common = sorted(set(a) & set(b))
        ma = [x for s in common for x in a[s]]
        mb = [x for s in common for x in b[s]]
        if not common or not ma or not mb:
            v = "NODATA"
        elif all(ma) and not any(mb):
            v = "★REGRESS"
        elif not any(ma) and all(mb):
            v = "IMPROVE"
        elif sum(ma) / len(ma) > sum(mb) / len(mb):
            v = "약회귀"
        elif sum(ma) / len(ma) < sum(mb) / len(mb):
            v = "약개선"
        else:
            v = "MIXED"
        verdicts[t] = v
        print(hdr % (t,
                     ("%d/%d" % (sum(pa), len(pa))) if pa else "-",
                     ("%d/%d" % (sum(pb), len(pb))) if pb else "-",
                     ("시드%d · %d/%d → %d/%d" % (len(common), sum(ma), len(ma), sum(mb), len(mb)))
                     if common and ma and mb else "(공통 시드 없음)",
                     v))

    print("")
    print("## 요약")
    for v in ("★REGRESS", "약회귀", "MIXED", "약개선", "IMPROVE", "NODATA"):
        ts = [t for t in tasks if verdicts[t] == v]
        if ts:
            print("  %-9s %2d : %s" % (v, len(ts), " ".join(ts)))
    out = os.path.normpath(os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026",
                                        "x382_regression_scan.json"))
    io.open(out, "w", encoding="utf-8").write(json.dumps(
        {"since": since, "cut": cut, "verdicts": verdicts,
         "pre_runs": sorted(runs_pre), "post_runs": sorted(runs_post)},
        ensure_ascii=False, indent=1))
    print("원자료: %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
