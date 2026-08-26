# -*- coding: utf-8 -*-
r"""x555 — t7362 A/B 의 **이미 끝난 실패 sim** per-step 포렌식 (런 중 · 읽기 전용)

## 왜 지금 (사용자 지시 2026-08-26 밤 · *"이미 실패한 케이스 포렌식하여 원인 확정하라"*)
세 팔이 아직 도는 중이지만 러너는 sim 이 끝날 때마다 `results.json` 을 갱신한다. 끝난 실패
sim 은 **지금 판정 가능한 재료**이고, 그것을 지금 파면 `AB DONE` 을 기다리는 시간이 공짜가 된다.

## 규율 ([[73]] 실험 루프 · [[55]] 진단 순서 · [[08]] 집계 금지)
- 채점 단위는 **`reward`** 다([[69]]). `action_match` 는 진단 보조일 뿐 성적이 아니다.
- 실패 단위는 **변이 집합**(MISSING · WRONGARG · EXTRA · DUP)이고 정본 비교기는
  `t2_forensic.mutation_diff` / `action_diff` 다 — 손 비교기를 만들지 않는다([[67]]).
- ⛔**사이드카를 보기 전에 귀속하지 마라**([[30]]·핸드오프 §5.2). 우리 층 거절은 재생성 채널로
  나가고 `_ap_regen` 이 원 메시지를 **교체**하므로 영속 궤적의 BLOCKED 칸이 비어 있어도
  *"안 막았다"* 가 아니다. `sidecar != 'present'` 면 그 칸은 **모른다**로 남는다([[25]]).
- 귀속 순서 = **우리 배관 → 우리 문면 → 계기 → 그제서야 모델**([[55]]).

## 무엇을 인쇄하나 (판단 0 · 재료만)
sim 마다 ①채점 축과 reward ②변이/액션 대조표 ③사이드카 상태와 **우리가 반려한 행**(채널별·turn)
④그 sim 에서 발화한 우리 레버 마커 ⑤마지막 assistant 발화 축자. 원인 확정은 이 재료 **위에서**
사람이 한다 — 이 스크립트는 결론 문장을 만들지 않는다([[08]] 집계→결론 직행 금지).

사용: (리모트·cwd=scripts/distill/tau2) py -3 x555_t7362_perstep.py
      --tags bank_t7362_A_ctl_20260826,bank_t7362_B_say_20260826,bank_t7362_C_scope_20260826
"""

import argparse
import collections
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import t2_forensic as F                                            # noqa: E402

DEFAULT = ("bank_t7362_A_ctl_20260826,bank_t7362_B_say_20260826,"
           "bank_t7362_C_scope_20260826")
RX_MARK = re.compile(r"\[(T2_[A-Z0-9_]+|OPERATOR-DIRECT|SIGNATURE|OFFICIAL-NAME)\]")


def head(s, n=110):
    return " ".join(str(s or "").split())[:n]


def label_rows(rows):
    """변이/액션 행 → 사람이 읽을 한 줄들 (정본 `label` 재사용·[[67]])."""
    out = []
    for r in rows or ():
        nm = r.get("name") or r.get("key") or "?"
        out.append(head(r.get("label") or nm, 96))
    return out


def sim_markers(tag, simtag):
    """그 sim 에서 발화한 **우리 레버 마커**를 세어 준다(무엇이 말했는가 · 판단 0)."""
    c = collections.Counter()
    try:
        txt = F.log_text(tag)
    except Exception:
        return c
    for ln in (txt or "").splitlines():
        if ("[sim=%s]" % simtag) not in ln:
            continue
        m = RX_MARK.search(ln)
        if m:
            c[m.group(1)] += 1
    return c


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default=DEFAULT)
    ap.add_argument("--only-failed", action="store_true", default=True)
    ap.add_argument("--all", dest="only_failed", action="store_false")
    a = ap.parse_args(argv)

    tags = [t.strip() for t in a.tags.split(",") if t.strip()]
    for tag in tags:
        print("=" * 100)
        print("# %s" % tag)
        print(F.sidecar_note(tag))
        try:
            sims = F.sims(tag)
        except Exception as e:
            print("  결과 없음(%r) — 아직 한 sim 도 안 끝났거나 경로가 다르다" % (e,))
            continue
        scored = [s for s in sims if s.get("reward_info") is not None]
        print("  기록된 sim %d · 채점된 sim %d" % (len(sims), len(scored)))
        den = F.sidecar_denies(tag)
        for s in scored:
            rw = (s.get("reward_info") or {}).get("reward")
            st = F.simtag(s)
            if a.only_failed and rw == 1.0:
                print("\n-- %-20s reward %.1f  (통과 — 대조로만 인쇄)" % (st, rw))
                continue
            print("\n" + "-" * 96)
            print("-- %-20s reward %s · 축 %s · 종료 %s · msg %d · %ss"
                  % (st, rw, F.reward_basis(s) or "?", F.term_reason(s),
                     len(s.get("messages") or []), int(s.get("duration") or 0)))

            basis = F.reward_basis(s)
            if "ACTION" in (basis or []):
                d = F.action_diff(s, tag=tag)
            else:
                d = F.mutation_diff(s, tag=tag)
            for k in ("missing", "wrongarg", "extra", "dup", "matched", "blocked"):
                rows = d.get(k) or []
                if rows:
                    print("   %-9s %d" % (k.upper(), len(rows)))
                    for ln in label_rows(rows)[:6]:
                        print("      · %s" % ln)
            print("   [sidecar] %s · join=%s · 재생성으로 지워진 반려 %d"
                  % (d.get("sidecar"), d.get("regen_join"),
                     len(d.get("regen_blocked") or ())))
            for r in (d.get("regen_blocked") or ())[:6]:
                print("      ⊘ turn=%s ch=%s %s"
                      % (r.get("turn"), r.get("channel"), head(r.get("text"), 88)))

            rows = den["simtag"].get(st) or den["fp"].get(st) or []
            if rows:
                ch = collections.Counter(r.get("channel") or "?" for r in rows)
                print("   [우리 층 반려] %d 행 · 채널 %s"
                      % (len(rows), dict(ch.most_common(6))))
                seen = set()
                for r in rows:
                    t = head(r.get("text"), 100)
                    if t in seen:
                        continue
                    seen.add(t)
                    print("      ✗ turn=%s %s" % (r.get("turn"), t))
                    if len(seen) >= 6:
                        break

            mk = sim_markers(tag, st)
            if mk:
                print("   [레버 발화] %s" % dict(mk.most_common(10)))

            msgs = s.get("messages") or []
            last = [m for m in msgs if m.get("role") == "assistant" and m.get("content")]
            if last:
                print("   [마지막 assistant 축자] %s" % head(last[-1].get("content"), 220))
    print()
    print("⚠이 인쇄물은 **재료**다. 원인 확정은 이 위에서 하고, 우리-층 귀속은 반증까지 통과한 뒤에만"
          " 한다([[73]]·[[77]]).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
