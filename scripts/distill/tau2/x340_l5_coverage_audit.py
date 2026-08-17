# -*- coding: utf-8 -*-
r"""⚠**핵심 주장 기각 (2026-08-18·원장 C521ⓢ)** — 이 감사가 낸 *"`T2_COVERAGE_FOLLOWUP` 死배선"* 은 틀렸다.

이유: 인쇄 마커가 `[T2_COVERAGE_FU]` 인데 `[T2_COVERAGE_FOLLOWUP]` 으로 찾았다(**이름 불일치**).
실측: 라이브 발화 **141회**(sim_results 402 로그) — 다만 표적이 `get_reward_discrepancies`·
`get_atm_fee_discrepancies` **2 도구뿐**이라 C 버킷 대부분에서 **표적 밖**일 뿐 고장이 아니다.
⇒ 교훈: **자기-마커 부재를 사망으로 읽지 말 것**(`T2_GUIDED`·`T2_SURFACE_BUS` 도 `[T2_LEVER]`
버스로 3,335·1,690회 발화한다). 마커 이름은 **코드에서 읽어** 확인하라.

아래는 원문 그대로 남긴다.
"""
r"""x340 — **L5(coverage-followup) 배선 감사**. 핸드오프 §4 #5 의 전제를 실측으로 검정한다.

## 왜 (전제 재검)

2026-08-16 핸드오프는 L5 를 *"죽은 배선 — 019 의 op 가 `[coverage]` 라인을 안 찍는다
(`_sg_stats` 미기록) ⇒ 그 태스크군을 겨냥한 레버가 발화 0"* 으로 적었다. 근거는 **런 로그
grep 0건**이었는데, `[coverage]` 는 **stderr 로그가 아니라 도구 결과 텍스트**에 붙는다
(`t2_scaffold_get` 이 `_txt` 에 append) — 즉 로그에 없는 것이 당연하다. 계기 오독일 수 있다.

그래서 이 감사는 **메시지**를 본다:
  ⑴ `get_reward_discrepancies` 등 판정도구가 몇 번 불렸나
  ⑵ 그 결과에 `[coverage] a of b rows … (c could not be verified)` 가 붙었나
  ⑶ **c > 0 인 적이 있나** — 이것이 `_coverage_pending` 의 전제이자 L5 의 발화 조건
  ⑷ 로그에 `[T2_COVERAGE_FU] fired` 가 있나

⇒ ⑵가 0 이면 **배선 결함**(핸드오프가 맞다) · ⑵>0 ∧ ⑶=0 이면 **전제 미성립**(레버가 옳게
침묵한 것이고 고칠 배선이 없다) · ⑶>0 ∧ ⑷=0 이면 **그때가 진짜 죽은 배선**이다.

판정 규칙은 엔진 것을 그대로 쓴다(`_COVERAGE_RE` 동형·같은 해소 규칙: 같은 도구의 skipped==0
결과가 뒤에 오면 해소). 도메인 판단 0.

실행(리모트):
    /home/woori/venvs/seka_env/bin/python x340_l5_coverage_audit.py [tag …]
"""
import collections
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import t2_forensic as F                                            # noqa: E402

DEFAULT_TAGS = ["bank_t7301_probe2_20260816d", "bank_t7302_probe2b_20260816e"]
COV = re.compile(r"\[coverage\] (\d+) of (\d+) rows were checked \((\d+) could not be verified\)")


def coverage_of(sim):
    """도구 결과 메시지에서 `[coverage]` 를 뽑고 `_coverage_pending` 과 **같은 규칙**으로 해소."""
    id2name, pend, seen = {}, None, []
    for m in (sim.get("messages") or []):
        for tc in (m.get("tool_calls") or []):
            id2name[tc.get("id")] = F.nameof(tc)
        if m.get("role") != "tool":
            continue
        c = m.get("content")
        if not isinstance(c, str):
            continue
        mt = COV.search(c)
        if not mt:
            continue
        nm = id2name.get(m.get("id"))
        seen.append((nm, int(mt.group(1)), int(mt.group(2)), int(mt.group(3))))
        if int(mt.group(3)) > 0:
            pend = (nm, mt.group(0))
        elif pend and pend[0] == nm:
            pend = None
    return seen, pend


def main():
    tags = sys.argv[1:] or DEFAULT_TAGS
    tot = collections.Counter()
    for tag in tags:
        sims = F.sims(tag)
        fired = F.by_sim(tag, r"(T2_COVERAGE_FU\] fired.*)", sims)
        print("=" * 78)
        print("[%s] n=%d" % (tag, len(sims)))
        for s in sorted(sims, key=lambda x: (F.task_id(x), str(x.get("seed")))):
            key = F.simtag(s)
            seen, pend = coverage_of(s)
            names = [c for c in (F.nameof(tc) for _m, tc in F.calls(s)) if c]
            judge = [n for n in names if "discrepanc" in n or "get_" in n]
            skipped_any = sum(1 for _n, _j, _t, sk in seen if sk > 0)
            tot["sims"] += 1
            tot["cov"] += len(seen)
            tot["skipped>0"] += skipped_any
            tot["pending"] += 1 if pend else 0
            tot["fired"] += len(fired.get(key) or [])
            print("  %-22s 판정도구호출 %-2d · coverage %-2d %s · skipped>0 %d · pending %s · FU %d"
                  % (key, len(judge), len(seen),
                     ("[" + " ".join("%d/%d(-%d)" % (j, t, sk) for _n, j, t, sk in seen[:4]) + "]")
                     if seen else "", skipped_any, "Y" if pend else "n",
                     len(fired.get(key) or [])))
    print("=" * 78)
    print("합계: %s" % dict(tot))
    print("\n판정")
    if tot["cov"] == 0:
        print("  ⇒ **배선 결함** — 판정도구 결과에 [coverage] 가 한 번도 안 붙었다(핸드오프 #5 유지).")
    elif tot["skipped>0"] == 0:
        print("  ⇒ **전제 미성립** — [coverage] 는 붙었고 미판정 행이 0 이라 L5 의 발화 조건이")
        print("     성립한 적이 없다. 고칠 배선이 없다 ⇒ S2 는 표적을 바꾸거나 폐기.")
    elif tot["fired"] == 0:
        print("  ⇒ **진짜 죽은 배선** — 미판정 행이 있었는데 [T2_COVERAGE_FU] 가 0 회. 수리 대상.")
    else:
        print("  ⇒ 발화 > 0 — 레버는 살아 있다(효과는 별도 측정).")


if __name__ == "__main__":
    main()
