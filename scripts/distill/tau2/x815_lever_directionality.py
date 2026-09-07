#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""x815 — 레버 **방향성** d(ρ) · **유효 방향성** d_eff(ρ) 산출기 (설계 = `x814 rev2`).

## 왜 (x814 rev2 §2)
레버가 «어떤 도구를 부르라 / 부르지 마라» 를 밀면, **그 반대가 정답인 태스크**가 생긴다.
그 집합을 **런 전에** gold 로부터 세는 것이 이 프로브다.

    ⊕(ρ) = { t : gold(t) 가 ρ 가 미는 행동을 요구 }
    ⊖(ρ) = { t : push(ρ) ∉ tools(gold(t))  ∧  block(ρ) ∈ tools(gold(t)) }
    d(ρ)     = |⊖| / |T|                        ← gold 만 (런 0)
    d_eff(ρ) = |⊖ ∩ {ρ 가 실제 발화한 t}| / |T|   ← gold + 기존 로그 (런 0)

**d_eff 가 실제 위험량이다** — 술어가 ⊖ 태스크에서 발화하지 않으면 피해는 0이다.

## 규율
* **gold 출처 = `tasks.json` 의 `evaluation_criteria.actions`** — 런과 무관한 참조 해답.
  이것을 읽는 것은 **측정**이지 A2 저작이 아니다([[23]] 는 저작 금지이지 분석 금지가 아니다).
* **엔진 리터럴 0** — `push/block` 도구명은 전부 **선언 파일**에서 온다([[58]]/[[05]]).
  이 스크립트에 도메인 도구명을 쓰지 않는다.
* **LLM 판정 0** — 전부 집합 연산(P2b 프록시 회피).
* 산출은 **JSON 으로 영속**한다([[77]]② 검산 경로 · rev1 이 이걸 안 해서 반려됐다).

## 사용
    py -3 x815_lever_directionality.py \
        --tasks <tasks.json> \
        --decl  ../../../reports/facet_rft_2026/analysis_decl/lever_action_map_banking.json \
        [--logs <디렉터리…>]   # d_eff 용. 없으면 d 만 산출
        [--out  <out.json>]

⚠ **S3(미분류 레버 배터리 판정) 전에는 이 산출물의 d 값을 인용하지 마라** — 좌석이 바뀌면
   ⊖ 도 바뀐다(x814 rev2 §6).
"""
import argparse
import collections
import glob
import gzip
import io
import json
import os
import re
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


# ─── gold ────────────────────────────────────────────────────────────────────
def gold_tools(tasks_path):
    """태스크 → gold 가 요구하는 **도구명 집합**.

    래퍼(call_/give_/unlock_discoverable_*)는 바깥 이름과 **중첩 인자의 대상 이름**을
    둘 다 넣는다 — 선언이 어느 쪽으로 쓰든 걸리게. 판단 0·문자열 수집만.
    """
    with io.open(tasks_path, encoding="utf-8") as f:
        d = json.load(f)
    ts = d if isinstance(d, list) else (d.get("tasks") or [])
    out, basis = {}, {}
    for t in ts:
        tid = str(t.get("id"))
        ec = t.get("evaluation_criteria") or {}
        s = set()
        for a in (ec.get("actions") or []):
            nm = a.get("name")
            if nm:
                s.add(str(nm))
            ar = a.get("arguments") or {}
            for k in ("agent_tool_name", "discoverable_tool_name", "user_tool_name"):
                if ar.get(k):
                    s.add(str(ar[k]))
        out[tid] = s
        rb = ec.get("reward_basis")
        basis[tid] = ",".join(rb) if isinstance(rb, (list, tuple)) else str(rb or "")
    return out, basis


# ─── 발화 (d_eff) ────────────────────────────────────────────────────────────
def fired_tasks(log_dirs, markers):
    """레버 마커별 **발화한 태스크 집합**.

    로그 줄의 sim 태그(`[sim=task_NNN#s…]`)로 태스크를 귀속한다. 태그가 없는 줄은
    **버린다** — 귀속 못 하는 발화를 세면 d_eff 가 부풀기 때문이다([[25]]).
    """
    simre = re.compile(r"\[sim=(task_\d+)#")
    hit = collections.defaultdict(set)
    seen_files = 0
    for d in log_dirs:
        for p in sorted(glob.glob(os.path.join(d, "*.log")) + glob.glob(os.path.join(d, "*.log.gz"))):
            seen_files += 1
            op = gzip.open if p.endswith(".gz") else io.open
            try:
                with op(p, "rt", encoding="utf-8", errors="ignore") as f:
                    for ln in f:
                        m = simre.search(ln)
                        if not m:
                            continue
                        tid = m.group(1)
                        for lv, pats in markers.items():
                            if any(x in ln for x in pats):
                                hit[lv].add(tid)
            except Exception:
                continue
    return hit, seen_files


def marker_patterns(decl):
    """선언의 `marker`(그 **처방**의 축자)를 쓴다.

    ⛔플래그명에서 태그를 **유도하지 않는다** — 2026-09-07 실측: 선언에 `T2_FOLLOWUP_REQUIRED`
      를 쓰면 실제 태그 `[T2_FOLLOWUP]` 과 안 맞아 발화가 **0 으로 잘못** 잡힌다(x810 이 잰
      «태그↔플래그 이름 불일치 35건」과 같은 병). 또 계열 태그 `[T2_RESOLVE]` 는 다른 처방까지
      삼켜 d_eff 를 부풀린다. ⇒ **처방 단위 축자만** 선언에서 받는다.
      `marker` 미선언 레버는 **d_eff 를 내지 않는다**(모른다고 적는다·[[25]]).
    """
    out = {}
    for lv in decl.get("levers") or []:
        mk = [str(x) for x in (lv.get("marker") or []) if x]
        if mk:
            out[lv["lever"]] = mk
    return out


# ─── 본체 ────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--decl", required=True)
    ap.add_argument("--logs", nargs="*", default=[])
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    G, BASIS = gold_tools(a.tasks)
    with io.open(a.decl, encoding="utf-8") as f:
        decl = json.load(f)
    T = sorted(G)
    print("gold 태스크 %d · 선언 레버 %d" % (len(T), len(decl.get("levers") or [])))
    print("선언: %s" % os.path.basename(a.decl))
    print()

    fired, nfiles = ({}, 0)
    if a.logs:
        fired, nfiles = fired_tasks(a.logs, marker_patterns(decl))
        print("로그 파일 %d 개에서 발화 귀속" % nfiles)
        print()

    rows, res = [], {}
    for lv in decl.get("levers") or []:
        name = lv["lever"]
        push = set(lv.get("push") or [])
        block = set(lv.get("block") or [])
        # ⊕ = gold 가 push 를 요구  |  ⊖ = push 없고 block 을 요구
        plus = {t for t in T if push & G[t]} if push else set()
        minus = {t for t in T if (not (push & G[t])) and (block & G[t])} if block else \
                ({t for t in T if not (push & G[t])} if push else set())
        # push-only 레버: gold 가 그 도구를 **요구하지 않는데** 밀면 EXTRA 위험 ⇒ ⊖ = 여집합
        f = fired.get(name, set())
        eff = minus & f if f else set()
        d = len(minus) / len(T)
        d_eff = (len(eff) / len(T)) if f else None
        rows.append((name, lv.get("prescription_class", "?"), len(plus), len(minus), d,
                     (len(f) if f else None), (len(eff) if f else None), d_eff))
        res[name] = {
            "prescription_class": lv.get("prescription_class"),
            "audit_seat": lv.get("audit_seat"),
            "push": sorted(push), "block": sorted(block),
            "plus": sorted(plus), "minus": sorted(minus),
            "d": round(d, 4),
            "fired_tasks": sorted(f) if f else None,
            "minus_and_fired": sorted(eff) if f else None,
            "d_eff": (round(d_eff, 4) if d_eff is not None else None),
            "source": lv.get("source"),
        }

    print("%-34s %-5s %5s %5s %7s %7s %7s %8s" %
          ("레버", "R", "⊕", "⊖", "d", "발화T", "⊖∩발화", "d_eff"))
    print("-" * 88)
    for n, rc, p, m, d, nf, ne, de in rows:
        print("%-34s %-5s %5d %5d %7.3f %7s %7s %8s" %
              (n, rc, p, m, d,
               ("-" if nf is None else nf), ("-" if ne is None else ne),
               ("-" if de is None else "%.3f" % de)))
    print()
    print("⚠ R4(표면화)는 구성적 d=0 이라 판정 면제(x814 rev2 §2c).")
    print("⚠ S3(미분류 레버 배터리 판정) 전에는 이 d 값을 인용하지 마라(x814 rev2 §6).")

    if a.out:
        with io.open(a.out, "w", encoding="utf-8") as f:
            json.dump({"n_tasks": len(T), "tasks": T, "reward_basis": BASIS,
                       "decl": os.path.basename(a.decl), "n_log_files": nfiles,
                       "levers": res}, f, ensure_ascii=False, indent=1)
        print("\n영속: %s" % a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
