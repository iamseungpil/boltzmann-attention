# -*- coding: utf-8 -*-
r"""t2_gap — **격리↔라이브 차이 분해기**(2026-08-16·사용자 지시).

## 왜

격리에서 열린 레버가 라이브에서 두 번 연속 **null** 이었다:

    촉구(T2_ACT_DEMAND)   격리 2/24 → 16/24   ↔ 라이브 8/20 ↔ 9/20 · over-action 2→8   (C492)
    전달 복구(t7296)      배선 5 → 31         ↔ 라이브 1/16 ↔ 1/16 · 지연 1.8×          (C488)

그때마다 *"타이밍 때문일 것"* · *"경합 때문일 것"* 이라고 **말했지 재지 않았다**. 격리와 라이브
사이는 지금 **빈 구간**이고, 이 모듈이 그 사이를 사다리로 채운다 — 라이브에만 있는 요인을
**한 번에 하나씩** 되돌려 넣고, 떨어지는 칸에 귀속한다.

## 사다리 (각 단은 바로 아래 단 + 요인 하나)

    I0_CORE    손님 발화 축자만 + 재료            기준선(격리)
    I1_CTX     + **실제 궤적 문맥**               ← 부하([[18]])
    I2_EARLY   + 재료를 **결정점보다 앞**에 배치  ← 타이밍(라이브에선 미리 와야 한다)
    I3_RIVAL   + **같은 자리에 경쟁 문구**        ← 경합(우리 레버끼리 자리를 다툰다)
    I4_TOOL    + **도구 호출로 요구**(텍스트 아님) ← 끝맺음(C489 knowing–doing)
    (L)        라이브                             나머지 = 비용·손님 변동

각 칸의 낙차 = 그 요인의 몫. **차 ≥5 만 인용**(잡음 바닥 ±4·C483).

## 규율

⚠경쟁 문구·재료는 **궤적과 A2 에서 축자로** 가져온다(내가 지어내면 그 팔은 내 문장을 재는 것이다).
  못 찾으면 그 단은 **건너뛴다**(침묵) — 대체물을 지어 넣지 않는다([[25]]).
⚠도구 바인딩은 **환경이 든 스키마 전체**다. 하나만 묶으면 그것이 지목이 된다(x322: 24/24 → 0/24).
⚠[[62]] ③: 이 모듈은 **재기만** 한다. 어떤 답도 고르지 않고 어떤 도구도 지목하지 않는다.

## 쓰는 법

    /home/woori/venvs/seka_env/bin/python t2_gap.py <tag> <task> <cut> <group> [k] [nb]
"""
import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/home/woori/scratch/tau2-bench/src")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import t2_forensic as F                                           # noqa: E402
import t2_probe as P                                              # noqa: E402
import t2_search as S                                             # noqa: E402
import x313_bailout_iso as X                                      # noqa: E402

DOCS = "/home/woori/scratch/tau2-bench/data/tau2/domains/banking_knowledge/documents"
A2P = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "a2", "banking_knowledge.specific.json")
# 우리 층이 낸 문구를 궤적에서 알아보는 표지(축자 — 이 문자열들은 우리가 인쇄한 것이다)
OURS = ("NOT_VERIFIED", "GROUNDING WARNING", "not discovered from any prior search",
        "TRANSFER NOTICE", "could not be verified")


def user_only(sim, cut, limit=6):
    """손님 발화만(격리 기준선) — 궤적 축자, 요약 0."""
    out = []
    for m in (sim.get("messages") or [])[:cut]:
        if m.get("role") == "user":
            c = " ".join(str(m.get("content") or "").split())
            if c:
                out.append("[user] " + c[:900])
    return "\n".join(out[-limit:])


def rival_text(sim, cut):
    """궤적에서 **우리 층이 낸 문구**를 축자로 하나 집는다. 없으면 빈 문자열(그 단 건너뜀)."""
    for m in reversed((sim.get("messages") or [])[:cut]):
        c = " ".join(str(m.get("content") or "").split())
        if m.get("role") == "tool" and any(w in c for w in OURS):
            return c[:600]
    return ""


def schemas():
    """환경이 든 도구 스키마 **전체** — 기계 추출·저작 0·지목 0."""
    try:
        from tau2.registry import registry
        env = registry.get_env_constructor("banking_knowledge")(retrieval_variant="no_knowledge")
    except Exception as e:
        print("도구 스키마 실패(그 단 건너뜀): %r" % (e,))
        return []
    out = []
    for t in env.get_tools():
        try:
            out.append(t.openai_schema)
        except Exception:
            continue
    return out


def ladder(tag, task, cut, group, marks, ask, k=8, nb=3, early=6):
    sim = next(s for s in F.scored(tag) if F.task_id(s) == task)
    st = P.site(tag, task, cut)
    a2 = json.load(io.open(A2P, encoding="utf-8"))
    material, info = S.material_for(a2, group, doc_dir=DOCS, windowed="general", per_doc=400)
    core = user_only(sim, cut)
    rival = rival_text(sim, cut)
    early_base = "\n".join([X.HEAD, "", material, "", X.transcript(sim, cut)])

    print("t2_gap · %s/%s · cut=%d · group=%s · 재료 %d자(유지 %d) · 손님축자 %d자 · 경쟁문구 %s\n"
          % (tag, task, cut, group, len(material), info["kept"], len(core),
             ("%d자" % len(rival)) if rival else "없음(I3 건너뜀)"))

    # I0 는 사이트 본문을 쓰지 않으므로 site.base 를 손님 축자로 갈아 끼운 별도 site 를 만든다
    core_site = dict(st)
    core_site["base"] = "\n".join([X.HEAD, "", core])
    print("── I0_CORE / I1_CTX / I2_EARLY (재료 위치·문맥 축) ──")
    r0 = P.run("gap:I0", core_site, [("A_REF", ""), ("I0_CORE", material)], marks,
               "I0 대비 I1 낙차 = 부하 · I1 대비 I2 낙차 = 타이밍(부호 반대면 미리 주는 것이 낫다)",
               ask, None, k, nb)
    early_site = dict(st)
    early_site["base"] = early_base
    r1 = P.run("gap:I1", st, [("A_REF", ""), ("I1_CTX", material)], marks,
               "(위와 같은 사다리)", ask, None, k, nb)
    r2 = P.run("gap:I2", early_site, [("A_REF", ""), ("I2_EARLY", "")], marks,
               "(위와 같은 사다리)", ask, None, k, nb)

    r3 = None
    if rival:
        print("── I3_RIVAL (경합 축) ──")
        r3 = P.run("gap:I3", st, [("A_REF", ""), ("I3_RIVAL", material + "\n\n[tool] " + rival)],
                   marks, "I2/I1 대비 낙차 = 경합(같은 자리를 다투는 우리 문구)", ask, None, k, nb)

    # ── I5_EPHEMERAL (2026-08-16·C498 로 알게 된 **다섯째 요인**) ──
    # 라이브의 배달은 `state.messages` 가 아니라 **그 턴의 재생성 버퍼**에만 붙는다(비커밋·C298).
    # 즉 재료는 한 턴만 살아 있고, 예산 3회를 대화 초반에 쓰면 **결정점에는 없다**(t7298 055 실측:
    # 배달 `대화텍스트 1` 부터 3회 소진 · 궤적 재료 표지 0건 · 선택 0/4 ↔ 격리 24/24).
    # 이 단은 그 상태를 격리로 재현한다 — 재료를 **앞 구간에만** 두고 결정점에서는 뺀다.
    print("── I5_EPHEMERAL (비영속 축·재료를 앞에만 두고 결정점에서 뺀다) ──")
    head_only = "\n".join([X.HEAD, "", material, "", X.transcript(sim, max(1, cut - early))])
    eph_site = dict(st)
    eph_site["base"] = head_only + "\n" + X.transcript(sim, cut)[-1200:]
    r5 = P.run("gap:I5", eph_site, [("A_REF", ""), ("I5_EPHEMERAL", "")], marks,
               "I2_EARLY(재료가 결정 프롬프트에 있음) 대비 낙차 = **비영속**(있었는데 사라짐)",
               ask, None, k, nb)

    r4 = None
    tools = schemas()
    if tools:
        print("── I4_TOOL (끝맺음 축·도구 %d개 바인딩·지목 0) ──" % len(tools))
        r4 = P.run("gap:I4", st, [("A_REF", ""), ("I4_TOOL", material)], marks,
                   "텍스트로는 대는데 도구로는 안 부르면 그 낙차가 **끝맺음**이다(C489)",
                   "\n[instruction] Carry out the next step of this request now.",
                   None, k, nb, tools=tools)

    print("\n═══ 사다리 요약 (지표별·차 ≥5 만 인용) ═══")
    for m in marks:
        row = []
        for lab, r in (("I0_CORE", r0), ("I1_CTX", r1), ("I2_EARLY", r2),
                       ("I5_EPHEMERAL", r5), ("I3_RIVAL", r3), ("I4_TOOL", r4)):
            if r and lab in r:
                row.append((lab, r[lab][m][0]))
        print("  %-8s %s" % (m, " → ".join("%s %d" % (a, b) for a, b in row)))
        for i in range(1, len(row)):
            d = row[i - 1][1] - row[i][1]
            if P.cite(row[i - 1][1], row[i][1]):
                print("      ▼ %s → %s 낙차 %d = **%s 에 귀속**"
                      % (row[i - 1][0], row[i][0], d, row[i][0].split("_")[-1]))
    return {"I0": r0, "I1": r1, "I2": r2, "I3": r3, "I4": r4}


def main():
    if len(sys.argv) < 5:
        print(__doc__.strip().split("## 쓰는 법")[-1].strip())
        return 1
    tag, task, cut, group = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
    k = int(sys.argv[5]) if len(sys.argv) > 5 else 8
    nb = int(sys.argv[6]) if len(sys.argv) > 6 else 3
    marks = {"SKY": "SKY BLUE", "LIME": "LIME GREEN", "PURPLE": "PURPLE", "BLUEST": "BLUEST"}
    ask = ("\n[instruction] Do NOT call any tool. Reply with ONE line only: the full official name "
           "of the ONE account class you would open for this customer, nothing else.")
    ladder(tag, task, cut, group, marks, ask, k, nb)


if __name__ == "__main__":
    sys.exit(main() or 0)
