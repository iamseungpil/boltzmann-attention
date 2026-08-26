# -*- coding: utf-8 -*-
r"""x550 - 074·085·040 이 실패한 **우리 층** 세 자리를 격리로 잰다. 런 0회·모델 0회.

## 왜 (2026-08-26 · 사용자 지시 *"셋을 격리로 재고 한번에 수리하라"*)

t7360 스모크에서 셋 다 0.0 이었고, 사이드카 포렌식이 셋 다 **우리 층 관여**를 보여줬다
(반려 074 **29행** · 085 **40행** · 040 **19행** — `regen_join='simtag'` 정확 조인).
⛔영속 궤적만 보면 이 셋이 **안 보인다** — `_ap_regen` 이 원 메시지를 교체하기 때문이다([[30]]).
실제로 나는 이 세션에서 074 를 *"모델이 도구를 안 돌렸다"* 로 한 번 오귀속했고, 사이드카가
그것을 뒤집었다([[55]] 우리 배관 먼저).

수리 **전에** 각 자리의 폭발 반경을 잰다([[62]]① · [[70]] 무엇을 사고 파나).

    §1 074  `[OPERATOR-PROVENANCE]` 가 **우리 A2 스캐폴드 도구**를 "발명한 이름"이라 막는가
    §2 085  `[OPERATOR-SCOPE]` 25행이 전부 **read** 도구인가 · 그 반려가 무엇을 사는가
    §3 040  `eligible_for_provisional_credit` 를 정할 **재료가 궤적에 오는가**

## §1 이 왜 결함인가 — 술어가 아니라 **처방**이 틀렸다([[64]])

`get_atm_fee_discrepancies` 는 A2 `scaffold_get_tools` 10개 중 하나이고 env 표면에 **없다**
(discoverable 도 아니다). 모델이 그것을 디스패처로 감싸 부른 것은 모델의 실수가 맞다.
그런데 우리 문면은:

    "[OPERATOR-PROVENANCE] tool name 'get_atm_fee_discrepancies' was not discovered from any
     prior search/listing result — do NOT invent tool names. Search/list the available tools
     first (getter KB_search_bm25), then use one of the discovered names."

⇒ *"네가 지어냈다 · 검색해라"*. 모델은 시킨 대로 `KB_search_bm25` 를 **3회**(msg 40·42·44) 돌리고
turn 48 에 unlock 을 시도해 env 오류까지 맞았다. env **자신의** 문면은 옳게 말한다 —
*"If it is a tool you already have, **call it directly**."* ⇒ 우리 문면이 env 문면보다 나쁘다.
[[64]]: 거부는 *무엇이 틀렸나* + *무엇을 하면 풀리나* 둘 다 담아야 하는데 후자가 **틀린 방향**이다.
[[25]]: 우리 도구 출력은 100% 정답 의무 — 우리가 준 도구를 "발명"이라 부른 것은 허위다.

## 반증 조건 ([[77]]③ — 주장과 동시에)

  §1 `chosen` 이 A2 `scaffold_get_tools` 에도 레지스트리에도 없으면 → 우리 문면이 옳다(결함 아님).
  §2 `[OPERATOR-SCOPE]` 의 `chosen` 이 write 이거나, 반려 뒤 그 도구가 **끝내 실행되지 않았다면**
     → 그 반려는 재료를 막은 것이 아니라 오선택을 막은 것일 수 있다(결함 판정 보류).
  §3 궤적에 자격 문구가 **오면** → 040 의 결손은 전달이 아니라 **선택**이다(다른 수리).

## 표본 ([[74]]-b · 훅 §74 ⑴⑶)

`--runs N`(기본 12) 최근 런의 **사이드카**를 태그별로 센다. 사이드카가 유일한 권위다
(`sidecar_status != 'present'` 인 태그는 **셈에서 빼고 그 사실을 인쇄한다** — 빈 값을 0 으로
읽지 않는다·`regen_blocked` 독스트링 경고).

실행: PYTHONIOENCODING=utf-8 py -3 x550_three_gate_iso.py [--runs 12]
"""
import collections
import gzip
import io
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import t2_forensic as F                                        # noqa: E402
import t2_gate_patch as G                                      # noqa: E402

DOMAIN = "banking_knowledge"
OUT = os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026",
                   "x550_three_gate_iso_2026_08_26.json")

RX_PROV = re.compile(r"\[OPERATOR-PROVENANCE\] tool name '([^']+)'")
RX_SCOPE = re.compile(r"\[OPERATOR-SCOPE\] you called '([^']+)'")


def load_sims(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt", encoding="utf-8", errors="replace") as fh:
        d = json.load(fh)
    if isinstance(d, dict):
        d = d.get("simulations") or d.get("results") or []
    return d if isinstance(d, list) else []


def recent(n):
    fs = [p for p in F.all_result_files() if p.endswith(".results.json.gz")]
    fs.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return fs[:n]


def surfaces():
    """도구 이름의 **출처 분류** — 전부 선언에서 읽는다(도메인 리터럴 0)."""
    a2 = G._domain_a2(DOMAIN)
    scaffold = {str(t.get("name")) for t in (a2.get("scaffold_get_tools") or [])
                if isinstance(t, dict) and t.get("name")}
    surf = json.load(io.open(os.path.join(HERE, "a2", "env_surface.json"),
                             encoding="utf-8"))[DOMAIN]["tools"]
    disc = {k for k, v in surf.items() if v.get("discoverable")}
    return a2, scaffold, set(surf), disc


def classify(name, scaffold, envall, disc):
    base = re.sub(r"_\d+$", "", str(name or ""))
    if name in scaffold or base in scaffold:
        return "A2-스캐폴드(직접 호출)"
    if name in disc or base in disc:
        return "env-discoverable"
    if name in envall or base in envall:
        return "env-직접"
    return "미상"


def denies(files):
    """태그별 사이드카 반려행. 사이드카 없는 태그는 **셈에서 제외**하고 따로 인쇄."""
    rows, missing = collections.OrderedDict(), []
    for p in files:
        tag = F.tag_of_file(p)
        if F.sidecar_status(tag) != "present":
            missing.append(tag)
            continue
        ix = F.sidecar_denies(tag)
        for s in load_sims(p):
            st, join, rr = F.regen_blocked(s, tag=tag, idx=ix)
            if st != "present":
                continue
            body = []
            for r in rr:
                body.append(r if isinstance(r, str)
                            else str((r.get("body") or r.get("text") or "")))
            rows.setdefault(tag, []).append((str(s.get("task_id")), s, body, join))
    return rows, missing


def s1_provenance(rows, scaffold, envall, disc):
    print("=" * 100)
    print("§1  074 — `[OPERATOR-PROVENANCE]` 가 무엇을 '발명'이라 부르나")
    print("=" * 100)
    kinds, per_tag, examples = collections.Counter(), collections.Counter(), []
    for tag, sims in rows.items():
        for task, s, body, join in sims:
            for b in body:
                m = RX_PROV.search(b)
                if not m:
                    continue
                nm = m.group(1)
                k = classify(nm, scaffold, envall, disc)
                kinds[(k, nm)] += 1
                per_tag[tag] += 1
                if k.startswith("A2-스캐폴드") and len(examples) < 6:
                    examples.append((tag, task, nm))
    tot = sum(kinds.values())
    print("  반려 %d건 — `chosen` 의 출처 분류:" % tot)
    agg = collections.Counter()
    for (k, nm), c in kinds.items():
        agg[k] += c
    for k, c in agg.most_common():
        mark = "  ← ★우리가 준 도구다(허위 문면)" if k.startswith("A2-스캐폴드") else ""
        print("     %-24s %4d%s" % (k, c, mark))
    print("\n  ★A2 스캐폴드 도구를 막은 이름 (도구별):")
    for (k, nm), c in sorted(kinds.items(), key=lambda x: -x[1]):
        if k.startswith("A2-스캐폴드"):
            print("     %-40s %3d회" % (nm, c))
    print("\n  실물 (태그·태스크):")
    for t, task, nm in examples:
        print("     %-40s %-10s %s" % (t[:40], task, nm))
    return {"total": tot, "by_kind": dict(agg),
            "scaffold_blocked": {nm: c for (k, nm), c in kinds.items()
                                 if k.startswith("A2-스캐폴드")}}


def s2_scope(rows):
    print("\n" + "=" * 100)
    print("§2  085 — `[OPERATOR-SCOPE]` 는 무엇을 막나 (read 인가 · 끝내 실행됐나)")
    print("=" * 100)
    read_c, write_c = collections.Counter(), collections.Counter()
    later_ok, later_no, per_task = 0, 0, collections.Counter()
    for tag, sims in rows.items():
        for task, s, body, join in sims:
            execed = {G._eff_tool_name(_TC(str(F.nameof(tc)), F.argsof(tc)))
                      for m in (s.get("messages") or [])
                      for tc in (m.get("tool_calls") or [])}
            for b in body:
                m = RX_SCOPE.search(b)
                if not m:
                    continue
                nm = m.group(1)
                base = re.sub(r"_\d+$", "", nm)
                per_task[task] += 1
                if G._READ_PREFIX_RE.match(base):
                    read_c[base] += 1
                else:
                    write_c[base] += 1
                if base in execed:
                    later_ok += 1
                else:
                    later_no += 1
    tot = sum(read_c.values()) + sum(write_c.values())
    print("  반려 %d건 — read %d (%.0f%%) · write %d"
          % (tot, sum(read_c.values()),
             100.0 * sum(read_c.values()) / max(1, tot), sum(write_c.values())))
    print("  반려 뒤 그 도구가 **끝내 실행됨** %d ↔ **끝내 미실행** %d" % (later_ok, later_no))
    print("     (실행됨 = 반려가 재료를 막은 게 아니라 **턴만 태웠다**는 뜻)")
    print("\n  가장 많이 막힌 read 도구:")
    for n, c in read_c.most_common(8):
        print("     %-42s %3d" % (n, c))
    if write_c:
        print("\n  write 쪽:")
        for n, c in write_c.most_common(5):
            print("     %-42s %3d" % (n, c))
    print("\n  태스크별 (상위 8):")
    for t, c in per_task.most_common(8):
        print("     %-12s %3d" % (t, c))
    return {"total": tot, "read": sum(read_c.values()), "write": sum(write_c.values()),
            "later_executed": later_ok, "never_executed": later_no,
            "top_read": read_c.most_common(8), "per_task": per_task.most_common(10)}


class _TC(object):
    def __init__(self, name, args):
        self.name, self.arguments = name, args


def s3_provisional(files):
    print("\n" + "=" * 100)
    print("§3  040 — `eligible_for_provisional_credit` 의 **재료가 궤적에 오는가**")
    print("=" * 100)
    pat = re.compile(r"provisional[ _]credit", re.I)
    rule = re.compile(r"provisional[^.]{0,160}(eligib|not eligible|does not qualify|qualif)",
                      re.I)
    hits, ruled, per_role = 0, [], collections.Counter()
    tasks = collections.Counter()
    for p in files:
        tag = F.tag_of_file(p)
        for s in load_sims(p):
            task = str(s.get("task_id"))
            for i, m in enumerate(s.get("messages") or []):
                c = str(m.get("content") or "")
                if not c or not pat.search(c):
                    continue
                hits += 1
                per_role[str(m.get("role"))] += 1
                tasks[task] += 1
                mm = rule.search(c)
                if mm and len(ruled) < 8:
                    ruled.append((tag, task, str(m.get("role")), i,
                                  " ".join(c[max(0, mm.start() - 90):mm.end() + 120].split())))
    print("  'provisional credit' 를 담은 메시지 %d건 · 역할별 %s" % (hits, dict(per_role)))
    print("  태스크별 상위: %s" % tasks.most_common(6))
    print("\n  **자격 규칙처럼 보이는 문장** (도구/문서 출력에서만 의미가 있다):")
    if not ruled:
        print("     ★없다 — 궤적 어디에도 자격을 가르는 문장이 안 온다")
        print("       ⇒ 040 의 이 칸은 **전달 결손**이고, 모델이 고를 재료가 애초에 없다.")
    for tag, task, role, i, txt in ruled:
        print("     [%s %s %s#%d] %s" % (tag[:28], task, role, i, txt[:210]))
    return {"messages_with_phrase": hits, "by_role": dict(per_role),
            "rule_like": [list(r) for r in ruled], "tasks": tasks.most_common(10)}


def main():
    n = 12
    if "--runs" in sys.argv:
        n = int(sys.argv[sys.argv.index("--runs") + 1])
    a2, scaffold, envall, disc = surfaces()
    files = recent(n)
    print("[표본] 최근 런 %d 개 · A2 스캐폴드 도구 %d 종" % (len(files), len(scaffold)))
    rows, missing = denies(files)
    print("[사이드카] present %d 태그 · **없음 %d 태그**%s"
          % (len(rows), len(missing), (" → " + ", ".join(t[:26] for t in missing[:6]))
             if missing else ""))
    print("           (없는 태그는 셈에서 뺐다 — 빈 값을 0 으로 읽지 않는다)")
    r1 = s1_provenance(rows, scaffold, envall, disc)
    r2 = s2_scope(rows)
    r3 = s3_provisional(files)
    print("\n" + "=" * 100)
    print("판정")
    print("=" * 100)
    print("  §1 우리 스캐폴드 도구를 '발명'이라 막은 건수 = **%d**"
          % sum(r1["scaffold_blocked"].values()))
    print("  §2 OPERATOR-SCOPE %d건 중 read %d · 반려 뒤 끝내 실행됨 %d"
          % (r2["total"], r2["read"], r2["later_executed"]))
    print("  §3 자격 규칙 문장이 궤적에 온 횟수 = **%d**" % len(r3["rule_like"]))
    with io.open(OUT, "w", encoding="utf-8") as fh:
        fh.write(json.dumps({"provenance": r1, "scope": r2, "provisional": r3,
                             "runs": [F.tag_of_file(p) for p in files],
                             "sidecar_missing": missing}, ensure_ascii=False, indent=1))
    print("\n  → %s" % os.path.abspath(OUT))


if __name__ == "__main__":
    main()
