#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""★게이트-유발 메시지 귀속 (무료·오프라인·[[08]]) — "이 날조는 모델이 낸 건가, 게이트가 만든 건가?"

문제: 라이브 로그(`[T2_PROV] regen fired …`)는 5-sim 동시실행이라 **뒤섞여** sim 귀속이 안 된다.
발견: 영속 궤적의 `raw_data.usage.prompt_tokens`가 그 답을 갖고 있다.
  regen은 **작업본**에 (거부된 호출 + 피드백)을 덧붙여 재생성한다 → 채택된 메시지의 prompt_tokens는
  **공식 대화보다 큰** 프롬프트에서 나온다. 그 다음 공식 메시지는 그 덧붙임이 사라진 프롬프트를 쓴다
  → **prompt_tokens가 감소**한다. 대화가 길어지는데 프롬프트가 줄어드는 건 **regen 말고는 불가능**.

판정: assistant[i]의 pt가 assistant[i+1]의 pt보다 크면 → assistant[i] = **게이트 개입(regen) 산물**.
      (보수적: 그 사이 공식 메시지 토큰만큼 더 커야 하므로 실제 개입은 더 많을 수 있다 = **하한**.)

용도: §20 perseveration 카운트(kon 3 vs koff 10-14)를 **게이트-유발 vs 자발**로 재분류.
      C107이 `prov_reloc`으로 보인 인과(PROV regen → 도구명 날조 24/24)를 **라이브 궤적에서 확인**.

Run: python3 bank_regen_attribution.py bank_kon_20260717_key bank_koff_20260717_key
"""
import gzip
import json
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
SIMDIR = os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026", "sim_results")


def load(tag):
    p = os.path.join(SIMDIR, f"{tag}.results.json.gz")
    with gzip.open(p, "rt", encoding="utf-8") as f:
        return json.load(f)["simulations"]


def pt_of(m):
    return ((m.get("raw_data") or {}).get("usage") or {}).get("prompt_tokens")


def name_of(m):
    tcs = m.get("tool_calls") or []
    return tcs[0].get("name") if tcs else None


def analyze(tag, toolset=None):
    sims = load(tag)
    rows, cnt = [], Counter()
    for si, s in enumerate(sims):
        msgs = s["messages"]
        asst = [(j, m) for j, m in enumerate(msgs) if m.get("role") == "assistant" and pt_of(m)]
        for a in range(len(asst) - 1):
            j, m = asst[a]
            j2, m2 = asst[a + 1]
            pt, pt2 = pt_of(m), pt_of(m2)
            regen = pt > pt2  # ★프롬프트 축소 = 버려진 덧붙임이 있었다 = 게이트 개입
            nm = name_of(m)
            if nm and toolset is not None and nm not in toolset:
                cnt[("날조", "게이트-유발" if regen else "자발")] += 1
                rows.append((tag, si, j, nm, pt, pt2, regen))
            elif regen:
                cnt[("비-날조", "게이트-유발")] += 1
    return rows, cnt


def main():
    tags = sys.argv[1:] or ["bank_kon_20260717_key", "bank_koff_20260717_key"]
    # 제공 도구 집합 = 궤적서 실제 성공한 호출 이름 ∪ tool-역할 응답이 정상이던 이름.
    # 간단·보수적으로: 차단 메시지("is not one of your available tools")를 받은 호출 = 비존재.
    for tag in tags:
        sims = load(tag)
        print("=" * 72)
        print(f"### {tag}")
        blocked = set()
        for s in sims:
            ms = s["messages"]
            for j, m in enumerate(ms):
                if m.get("role") == "tool" and isinstance(m.get("content"), str) \
                        and "not one of your available tools" in m["content"]:
                    for k in range(j - 1, -1, -1):
                        if ms[k].get("role") == "assistant" and name_of(ms[k]):
                            blocked.add(name_of(ms[k]))
                            break
        print(f"  비존재(차단당한) 도구명: {sorted(blocked)}")
        tot = Counter()
        for si, s in enumerate(sims):
            msgs = s["messages"]
            asst = [(j, m) for j, m in enumerate(msgs) if m.get("role") == "assistant" and pt_of(m)]
            for a in range(len(asst)):
                j, m = asst[a]
                nm = name_of(m)
                if nm not in blocked:
                    continue
                pt = pt_of(m)
                pt2 = pt_of(asst[a + 1][1]) if a + 1 < len(asst) else None
                regen = (pt2 is not None and pt > pt2)
                tot["게이트-유발" if regen else "자발/미상"] += 1
                print(f"  sim{si} [{j:2d}] {nm:42s} pt={pt} → next={pt2} "
                      f"{'★게이트-유발(프롬프트 축소)' if regen else '자발/미상'}")
        print(f"  ⇒ 도구명 날조 {sum(tot.values())}건: " +
              " · ".join(f"{k} {v}" for k, v in tot.most_common()))


if __name__ == "__main__":
    main()
