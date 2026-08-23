# -*- coding: utf-8 -*-
"""x496 - `권한 월권` 표적 재계수 (2026-08-23).

왜 다시 세나. `t2_levers.GAPS` 의 유일한 빈 칸이 `권한 월권` 이고 그 표적은 **12/14 (2026-08-04)**
로 적혀 있다. 그 뒤 스택이 크게 바뀌었다. [[31]] 규칙 4 = **표적 실재 선행**(`LEVER_CAUSE_MAP` §4
규율 6): 레버를 짓기 전에 **현 코퍼스에서 표적을 다시 센다**. 표적이 사라졌으면 짓지 않는다.

무엇을 세나 (`ACTION_HANDOFF_LEVERS_DESIGN_2026_08_04` §3 L1″ 술어 a 를 그대로 옮긴 것):

    술어 a : 카드 결정 직전 assistant 발화의 후보 이름 집합  ⊇  필터가 반환한 `eligible`

    표적   = 술어 a 가 깨졌고(=축약했고) **그 축약이 gold 카드를 지운** sim
             (= 결정 시점에 정답이 손님 앞에 없었다)

술어는 전부 닫혀 있다([[22]]):
  - `eligible` 은 env 도구 `check_card_application_fit` 이 **구조화해서** 돌려준다
    (`{'eligible': [{'card': ..., 'facts': {...}}, ...]}`) — 우리가 해석하지 않는다.
  - 축약 여부 = **집합 포함 검사**. 엔진은 고르지 않는다([[62]]).
  - 이 파일에 카드 이름 리터럴 0 — 이름은 전부 도구 출력에서 온다([[59]]·[[05]]).

⚠gold 는 **표적을 세는 진단**으로만 쓴다([[23]]). 레버는 gold 를 보지 않는다 —
  레버가 하는 일은 *"eligible 을 축약 없이 넘겼는가"* 뿐이고 그 판정에 gold 가 안 들어간다.

⚠이 프로브가 못 재는 것: 축약이 **손님의 선택을 바꿨는지**. 그것은 인과이고 A/B 몫이다.
  여기서 나오는 것은 **표적의 크기**뿐이다.

실행: PYTHONIOENCODING=utf-8 python x496_authority_target_recount.py ["bank_t7*_2026*"]
"""
import ast
import collections
import glob
import gzip
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

SIMS = os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026", "sim_results")
FIT = "check_card_application_fit"


def _parse_result(body):
    """도구 결과 본문 -> eligible 카드 이름 리스트. 파싱 실패면 []."""
    txt = str(body or "").strip()
    obj = None
    for loader in (json.loads, ast.literal_eval):
        try:
            obj = loader(txt)
            break
        except Exception:
            continue
    if not isinstance(obj, dict):
        return []
    out = []
    for e in (obj.get("eligible") or []):
        if isinstance(e, dict) and e.get("card"):
            out.append(str(e["card"]))
        elif isinstance(e, str):
            out.append(e)
    return out


def _gold_cards(sim):
    """gold 가 요구한 카드 인자 값(진단 전용)."""
    out = []
    for ck in ((sim.get("reward_info") or {}).get("action_checks") or []):
        a = (ck.get("action") or {}).get("arguments") or {}
        inner = a.get("arguments")
        if isinstance(inner, str):
            try:
                inner = json.loads(inner)
            except Exception:
                inner = None
        for src in (a, inner or {}):
            if not isinstance(src, dict):
                continue
            for k, v in src.items():
                if "card" in str(k).lower() and isinstance(v, str) and v:
                    out.append(v)
    return out


def main(argv):
    pats = argv or ["bank_t7*_2026*"]
    files = []
    for p in pats:
        files += sorted(glob.glob(os.path.join(SIMS, p + ".results.json.gz")))
        files += sorted(glob.glob(os.path.join(SIMS, p + "_results.json.gz")))
    files = sorted(set(files))
    print("결과 파일 %d개" % len(files))

    rows = []
    for fp in files:
        run = os.path.basename(fp).split(".")[0]
        try:
            with gzip.open(fp, "rt", encoding="utf-8", errors="replace") as f:
                d = json.load(f)
        except Exception:
            continue
        for s in (d.get("simulations") or []):
            ms = s.get("messages") or []
            # 1) fit 호출과 그 결과에서 eligible 집합
            elig, fit_at = [], None
            for i, m in enumerate(ms):
                for tc in (m.get("tool_calls") or []):
                    if tc.get("name") != FIT:
                        continue
                    res = next((x for x in ms[i:] if x.get("role") == "tool"
                                and x.get("id") == tc.get("id")), None)
                    e = _parse_result((res or {}).get("content"))
                    if e:
                        elig, fit_at = e, i          # 마지막 fit 결과를 쓴다
            if not elig:
                continue
            # 2) 카드 결정 지점 = fit 이후 첫 카드-신청 호출(주체 불문). 없으면 궤적 끝.
            anchor = len(ms)
            for i in range(fit_at + 1, len(ms)):
                for tc in (ms[i].get("tool_calls") or []):
                    nm = str(tc.get("name") or "")
                    args = json.dumps(tc.get("arguments") or "", ensure_ascii=False)
                    if "apply_for_credit_card" in nm or "apply_for_credit_card" in args:
                        anchor = min(anchor, i)
            # 3) 그 직전 assistant 발화(본문 있는 것)
            last_txt = ""
            for i in range(anchor - 1, fit_at - 1, -1):
                if ms[i].get("role") == "assistant" and str(ms[i].get("content") or "").strip():
                    last_txt = str(ms[i]["content"])
                    break
            mentioned = [c for c in elig if c.lower() in last_txt.lower()]
            missing = [c for c in elig if c not in mentioned]
            golds = _gold_cards(s)
            gold_shown = any(g.lower() in last_txt.lower() for g in golds) if golds else None
            rows.append({
                "run": run, "task": s.get("task_id"), "seed": s.get("seed"),
                "reward": (s.get("reward_info") or {}).get("reward"),
                "n_eligible": len(elig), "n_mentioned": len(mentioned),
                "narrowed": bool(missing), "n_missing": len(missing),
                "gold_cards": golds, "gold_shown": gold_shown,
                "no_guidance": not last_txt.strip(),
            })

    n = len(rows)
    narrowed = [r for r in rows if r["narrowed"]]
    target = [r for r in narrowed if r["gold_shown"] is False]
    nog = [r for r in rows if r["no_guidance"]]
    print("")
    print("=" * 88)
    print("권한 월권 표적 재계수 (현 코퍼스)")
    print("=" * 88)
    print("  fit 이 eligible 을 돌려준 sim            %4d" % n)
    print("  그중 **축약**했다(eligible 일부 미언급)   %4d  (%.0f%%)"
          % (len(narrowed), 100.0 * len(narrowed) / max(n, 1)))
    print("  ★표적 = 축약했고 **gold 카드가 안 보였다** %4d  (%.0f%%)"
          % (len(target), 100.0 * len(target) / max(n, 1)))
    print("  (참고) 결정 직전 assistant 본문이 아예 없음 %4d" % len(nog))
    if target:
        tp = sum(1 for r in target if r["reward"])
        print("  표적 sim 의 pass 율                      %d/%d" % (tp, len(target)))
    rest = [r for r in rows if r not in target]
    if rest:
        rp = sum(1 for r in rest if r["reward"])
        print("  나머지 sim 의 pass 율                    %d/%d" % (rp, len(rest)))

    print("")
    print("태스크별 (표적/축약/전체 · eligible 평균)")
    byt = collections.defaultdict(lambda: [0, 0, 0, 0])
    for r in rows:
        b = byt[r["task"]]
        b[2] += 1
        b[3] += r["n_eligible"]
        if r["narrowed"]:
            b[1] += 1
        if r in target:
            b[0] += 1
    print("%-10s %8s %8s %8s %10s" % ("task", "표적", "축약", "전체", "eligible"))
    print("-" * 50)
    for t in sorted(byt, key=lambda x: -byt[x][0]):
        b = byt[t]
        print("%-10s %8d %8d %8d %10.1f" % (t, b[0], b[1], b[2], b[3] / max(b[2], 1)))

    dst = os.path.join(SIMS, "..", "x496_authority_target_recount.json")
    with io.open(dst, "w", encoding="utf-8") as f:
        json.dump({"n": n, "narrowed": len(narrowed), "target": len(target),
                   "no_guidance": len(nog), "rows": rows}, f,
                  ensure_ascii=False, indent=1)
    print("")
    print("-> %s" % os.path.normpath(dst))
    print("")
    print("판정 규칙: 표적이 현 코퍼스에서 사라졌으면 L1″ 을 짓지 않는다([[31]] 규칙 4).")
    print("           표적이 남아 있으면 그 크기가 곧 L1″ 의 상한이다.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
