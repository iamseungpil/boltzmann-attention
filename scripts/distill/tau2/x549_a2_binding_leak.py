# -*- coding: utf-8 -*-
r"""x549 - `_a2_of(self)` 가 `unified()` 안에서 **항상 None** 이다. 런 0회·모델 0회.

## 왜 (2026-08-26 · t7359 스모크가 잡은 결함의 원인 규명)

스모크 첫 판이 이것을 인쇄했다:

    [sim=task_040#s626729] [T2_DUP_WRITE] deny tool=unlock_discoverable_agent_tool (앞선 성공 msg=4)

핸드오프의 가설은 *"술어에서 절차적 래퍼(unlock 계열)를 빼라"* 였다. **그 가설은 틀렸다** —
술어는 a2 만 주면 이미 unlock 을 뺀다(§1). 진짜 원인은 **재료가 안 오는 것**이다:

    `unified()` 는 `LLMAgent._generate_next_message = unified` (t2_gate_patch.py:13993) 로
    **에이전트**에 설치된다. 그런데 `_a2_of(obj)` 는 `obj.environment` 를 찾는다(:3228) —
    그 속성은 **orchestrator** 에만 있다. `init_inject`(:7376) 가 에이전트에 심는 것은
    `_t2_a2` 와 `_t2_orch` 지 `environment` 가 아니다.
    ⇒ `unified` 안의 `_a2_of(self)` **다섯 자리 전부 None** 을 받는다.

같은 함수가 정본 바인딩을 이미 갖고 있다(:7654 `a2 = getattr(self, "_t2_a2", None)`).
즉 자격이 아니라 **위치/전달**이다([[76]]⒜ — 격리는 멀쩡한데 라이브가 다른 것을 받는다).

## ★선행 확인 ([[74]] 규칙 ① · [[77]] ④ — grep 한 경로를 적는다)

`grep -rn "_a2_of|_is_effective_write|WRITE_PROV|unlock.*write" reports/facet_rft_2026/*.md`
두 정본이 이 실패 모드를 **이미** 적어 뒀다. 새 발견이 아니라 **예고된 회귀의 실현**이다:

  ⒜ `ENGINE_LITERAL_REMEDIATION_DESIGN_2026_07_30.md` §8-B (축자):
     *"①**순서 의존** — `_is_effective_write` 가 해당 도메인의 `_domain_a2()` 보다 먼저 불리면
       집합이 비어 `give_…`/`unlock_…` 이 write 로 판정되고, 이는 `:4531` 이 **회귀 조건으로
       못박은** `_is_effective_write("give_…")=False` 가 정확히 무너지는 시나리오다."*
     그 문서의 개정 처방은 *"호출부 **6곳** 수정 … **6곳 모두 a2 가 근처에 있는 오케스트레이터
     래퍼 안**"* 이었다. ⇒ **그 전제가 `unified` 에서 거짓이다.** `unified` 의 `self` 는
     오케스트레이터가 아니라 **에이전트**다. 전역은 없앴는데 **전달을 안 했다**.

  ⒝ `LEVER_ROSTER_CANONICAL_2026_08_19.md:248` (축자):
     *"`T2_WRITE_PROV` | 마크 12,181 : 실발화 **3** (524:1) | ★먼저 계기 수리 … 그 다음
       창 12,038 ↔ regen 141 의 **격차 원인**"* — 그 격차의 원인이 **미해결로 남아 있다**.
     §3 이 그 물음의 답 후보를 잰다(상시 켜진 레버가 이 결함 위에 앉아 있다).

  ⒞ 래칫이 왜 못 잡았나: `test_c241_u1_predicate.py` 는 **순수 함수** `_is_effective_write(n, a2)`
     만 검정한다. *"호출부가 a2 를 정말 넘기는가"* 를 검정하는 칸이 **없다** ⇒ 격리 100% 인 채
     라이브가 깨져 있었다([[76]]·[[78]]).

## 무엇을 재나 ([[70]] — 무엇을 사고 무엇을 파나)

    §1 술어      a2 유무로 판정이 뒤집히는 **도구 이름** (env 표면 전수·런 0회)
    §2 DUP_WRITE 라이브 술어가 **측정 표본 밖**으로 몇 번 발화하나
                 (x546/x547 이 잰 것 = `mutating_tools` 기준 · 라이브 = `_is_effective_write`)
    §3 상시 레버 `T2_WRITE_PROV=1`·`T2_CLAIM_PROV=1`(go_stack:110·162) 둘 다
                 `_any_effective_write(msgs, _a2_of(self))` 로 갈린다 ⇒ 판정이 몇 sim 뒤집혀 있나.
                 **이것이 이 결함의 실제 폭발 반경**이다.

## 표본 ([[74]]-b 세대 미뭉갬 · 훅 §74 ⑴⑶ 답)

⑴ **최근 런 태그만** 본다(`--runs N`·기본 12·파일 mtime 순). 전 코퍼스를 더하지 않는다.
⑵ 위 §선행 확인 참조 — ⒜⒝ 에 **이미 있고**, 이 프로브는 그 미해결 물음에 수치를 댄다.
⑶ §1 은 **런 0회**로 답한다(순수 함수 × env 표면). §2·§3 만 궤적이 필요하고, 궤적은 술어를
   먹일 **입력 표본**일 뿐이라 최근 런으로 족하다. 결과는 **태그별로** 인쇄한다.

반증 조건([[77]]③ — 주장과 **동시에**):
  - `unified` 안에서 `_a2_of(self)` 가 non-None 이면(=에이전트에 `.environment` 가 있으면) 거짓.
  - 병합 A2 에 `eplan.unlock_tool` 이 없으면 원인이 다른 것(술어 자체 결함).
  - §3 뒤집힘이 0 이면 *"상시 레버가 이 결함 위에 앉아 있다"* 는 거짓 — 결함은 DUP_WRITE 국소.

실행: PYTHONIOENCODING=utf-8 py -3 x549_a2_binding_leak.py [--runs 12]
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
from x544_dup_credit_regrade import tool_result_ok             # noqa: E402

DOMAIN = "banking_knowledge"
OUT = os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026",
                   "x549_a2_binding_leak_2026_08_26.json")


class _TC(object):
    """도구 호출 스텁 — `_eff_tool_name`/`_mut_key_of` 가 보는 속성만 갖는다."""

    def __init__(self, name, args, tcid=""):
        self.name = name
        self.arguments = args
        self.id = tcid


def load(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt", encoding="utf-8", errors="replace") as fh:
        d = json.load(fh)
    if isinstance(d, dict):
        d = d.get("simulations") or d.get("results") or []
    return d if isinstance(d, list) else []


def recent_files(n):
    """최근 런 n 개의 결과 파일 (mtime 순) — 세대를 뭉개지 않기 위해 표본을 좁힌다."""
    fs = [p for p in F.all_result_files() if p.endswith(".results.json.gz")]
    fs.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return fs[:n]


# ─────────────────────────── §0 배선 ───────────────────────────
def s0_binding():
    print("=" * 98)
    print("§0  배선 — `_a2_of(self)` 는 `unified()` 안에서 무엇을 받나")
    print("=" * 98)
    lines = io.open(os.path.join(HERE, "t2_gate_patch.py"), encoding="utf-8").read().split("\n")

    install = [i + 1 for i, l in enumerate(lines)
               if re.search(r"LLMAgent\._generate_next_message\s*=\s*unified", l)]
    inject = [i + 1 for i, l in enumerate(lines) if re.search(r"ag\._t2_a2\s*=", l)]
    sets_env = [i + 1 for i, l in enumerate(lines)
                if re.search(r"(self|ag|agent)\.environment\s*=[^=]", l)]
    print("  `LLMAgent._generate_next_message = unified`      줄 %s" % (install or "없음"))
    print("  `ag._t2_a2 = a2` (init_inject 가 에이전트에 심는 것)  줄 %s" % (inject or "없음"))
    print("  에이전트에 `.environment` 를 심는 곳                줄 %s" % (sets_env or "**없음**"))

    us = next(i for i, l in enumerate(lines)
              if l.startswith("    def unified(self, message, state):"))
    ue = next((i for i in range(us + 1, len(lines))
               if re.match(r"^    (def |[A-Za-z])", lines[i])), len(lines))
    hits = [i + 1 for i in range(us, ue) if "_a2_of(self)" in lines[i]]
    canon = [i + 1 for i in range(us, ue)
             if re.search(r'a2\s*=\s*getattr\(self,\s*"_t2_a2"', lines[i])]
    print("  unified 본문 = 줄 %d..%d" % (us + 1, ue))
    print("    `_a2_of(self)` 호출 %d 자리: %s" % (len(hits), hits))
    print("    같은 함수의 **정본 바인딩** `getattr(self,'_t2_a2')`: 줄 %s" % canon)

    class _AgentLike(object):
        pass

    ag = _AgentLike()
    ag._t2_a2 = G._domain_a2(DOMAIN)          # init_inject 가 하는 그대로
    ag._t2_orch = None
    got = G._a2_of(ag)
    print("\n  실측: init_inject 가 심은 그대로의 에이전트에 `_a2_of` 를 걸면 → %s" % (
        "**None** ⇒ 재료가 안 온다" if got is None else "dict(%d keys) ⇒ 반증됨" % len(got)))
    return {"install": install, "inject": inject, "sets_env": sets_env,
            "a2_of_sites": hits, "canonical_binding": canon,
            "a2_of_agent_is_none": got is None}


# ─────────────────────────── §1 술어 ───────────────────────────
def s1_predicate():
    print("\n" + "=" * 98)
    print("§1  술어 — a2 유무로 판정이 뒤집히는 도구 이름 (env 표면 전수 · 런 0회)")
    print("=" * 98)
    a2 = G._domain_a2(DOMAIN)
    ep = (a2 or {}).get("eplan") or {}
    print("  반증점검: 병합 A2 의 `eplan.unlock_tool` = %r" % ep.get("unlock_tool"))
    surf = json.load(io.open(os.path.join(HERE, "a2", "env_surface.json"),
                             encoding="utf-8"))[DOMAIN]["tools"]
    print("  A2 가 선언한 절차적 집합 = %s" % sorted(G._a2_procedural(a2)))
    flip = []
    for nm in sorted(surf):
        w_none, w_a2 = G._is_effective_write(nm, None), G._is_effective_write(nm, a2)
        if w_none != w_a2:
            flip.append({"tool": nm, "none": w_none, "a2": w_a2,
                         "env_mutates": bool(surf[nm].get("mutates"))})
    print("\n  env 표면 도구 %d 종 중 판정이 뒤집히는 것 **%d 종**:" % (len(surf), len(flip)))
    for r in flip:
        print("     %-38s  a2=None → write=%-5s | a2 → write=%-5s | env.mutates=%s"
              % (r["tool"], r["none"], r["a2"], r["env_mutates"]))
    print("\n  ⇒ 라이브(a2=None)는 이 %d 종을 **write 로 오인**한다. 스모크가 잡은 것이 그 중 하나다."
          % len(flip))
    return flip


# ─────────────────────────── 코퍼스 공통 ───────────────────────────
def sim_calls(sim):
    """(msg_idx, _TC) 목록 — assistant 호출만."""
    out = []
    for i, m in enumerate(sim.get("messages") or []):
        if str(m.get("role")) != "assistant":
            continue
        for tc in (m.get("tool_calls") or []):
            out.append((i, _TC(str(F.nameof(tc)), F.argsof(tc),
                               str((tc.get("id") if isinstance(tc, dict) else "") or ""))))
    return out


# ─────────────────────────── §2 DUP_WRITE ───────────────────────────
def s2_dup_scope(files, a2):
    print("\n" + "=" * 98)
    print("§2  T2_DUP_WRITE — 라이브 술어가 **측정 표본 밖**으로 몇 번 발화하나")
    print("=" * 98)
    per_tag = collections.OrderedDict()
    tools_none, tools_a2 = collections.Counter(), collections.Counter()
    for p in sorted(files):
        tag = F.tag_of_file(p)
        try:
            sims = load(p)
        except Exception:
            continue
        for s in sims:
            row = per_tag.setdefault(tag, {"sims": 0, "fire_none": 0, "fire_a2": 0})
            row["sims"] += 1
            msgs = s.get("messages") or []
            calls = sim_calls(s)
            for use_a2, bag, key in ((None, tools_none, "fire_none"), (a2, tools_a2, "fire_a2")):
                seen, fired = set(), 0
                for i, tc in calls:
                    if not G._is_effective_write(G._eff_tool_name(tc), use_a2):
                        continue
                    if not tool_result_ok(msgs, i, tc.id):
                        continue
                    k = G._mut_key_of(tc)
                    if not k:
                        continue
                    if k in seen:
                        fired += 1
                        bag[G._eff_tool_name(tc)] += 1
                    else:
                        seen.add(k)
                if fired:
                    row[key] += 1
    tot_s = sum(r["sims"] for r in per_tag.values())
    tot_n = sum(r["fire_none"] for r in per_tag.values())
    tot_a = sum(r["fire_a2"] for r in per_tag.values())
    print("  표본 sim %d · 중복 발화 sim: a2=None **%d** ↔ a2 있음 **%d**" % (tot_s, tot_n, tot_a))
    extra = {t: n for t, n in tools_none.items() if tools_a2.get(t, 0) == 0}
    print("\n  a2=None 일 때 발화 대상 도구 (상위 12):")
    for t, n in tools_none.most_common(12):
        print("     %-42s %4d%s" % (t, n, "   ← a2 면 안 잡힘" if t in extra else ""))
    print("\n  ⇒ **a2 미전달 탓에만** 발화하는 호출 %d 건 / 도구 %d 종: %s"
          % (sum(extra.values()), len(extra), sorted(extra)))
    print("\n  런 태그별 (세대 미뭉갬 · 발화 sim 수가 다른 태그만):")
    diff = [(t, r) for t, r in per_tag.items() if r["fire_none"] != r["fire_a2"]]
    for t, r in diff:
        print("     %-44s sim %3d | none %2d ↔ a2 %2d"
              % (t[:44], r["sims"], r["fire_none"], r["fire_a2"]))
    if not diff:
        print("     (없음)")
    return {"sims": tot_s, "fire_none": tot_n, "fire_a2": tot_a,
            "extra_calls": sum(extra.values()), "extra_tools": sorted(extra),
            "per_tag": {t: r for t, r in per_tag.items()}}


# ─────────────────────────── §3 상시 레버 ───────────────────────────
def s3_always_on(files, a2):
    print("\n" + "=" * 98)
    print("§3  ★폭발 반경 — 상시 켜진 `T2_WRITE_PROV`·`T2_CLAIM_PROV` 가 갈리는 자리")
    print("=" * 98)
    print("  둘 다 `_any_effective_write(state.messages, _a2_of(self))` 로 갈린다")
    print("  (:13240 `if …: break` · :4476 · :4489 — go_stack.sh:110·162 에서 `=1`).")
    per_tag = collections.OrderedDict()
    flip_sims, delta = [], []
    for p in sorted(files):
        tag = F.tag_of_file(p)
        try:
            sims = load(p)
        except Exception:
            continue
        for s in sims:
            row = per_tag.setdefault(tag, {"sims": 0, "true_none": 0, "true_a2": 0, "flip": 0})
            row["sims"] += 1
            fn = fa = None
            for i, tc in sim_calls(s):
                eff = G._eff_tool_name(tc)
                if fn is None and G._is_effective_write(eff, None):
                    fn = i
                if fa is None and G._is_effective_write(eff, a2):
                    fa = i
                if fn is not None and fa is not None:
                    break
            row["true_none"] += 1 if fn is not None else 0
            row["true_a2"] += 1 if fa is not None else 0
            rw = (s.get("reward_info") or {}).get("reward")
            if (fn is not None) != (fa is not None):
                row["flip"] += 1
                flip_sims.append((tag, str(F.sim_key(s)), rw))
            elif fn is not None and fa is not None and fn != fa:
                delta.append((tag, str(F.sim_key(s)), fn, fa, fa - fn,
                              len(s.get("messages") or [])))
    tot_s = sum(r["sims"] for r in per_tag.values())
    tn = sum(r["true_none"] for r in per_tag.values())
    ta = sum(r["true_a2"] for r in per_tag.values())
    print("\n  sim %d 중 `_any_effective_write` 참: a2=None **%d (%.1f%%)** ↔ a2 **%d (%.1f%%)**"
          % (tot_s, tn, 100.0 * tn / max(1, tot_s), ta, 100.0 * ta / max(1, tot_s)))
    print("  ★전-sim 판정이 **뒤집히는** sim: **%d**" % len(flip_sims))
    for t, k, rw in flip_sims[:12]:
        print("     %-40s %-26s reward=%s" % (t[:40], k[:26], rw))
    if len(flip_sims) > 12:
        print("     … 외 %d" % (len(flip_sims) - 12))
    print("\n  뒤집히진 않지만 **참이 되는 시점**이 늦어지는 sim: %d" % len(delta))
    if delta:
        d = sorted(x[4] for x in delta)
        print("     메시지 지연 중앙값 %d · 최대 %d (창이 그 사이에 열리면 판정이 다르다)"
              % (d[len(d) // 2], d[-1]))
        for t, k, fn, fa, dl, n in sorted(delta, key=lambda x: -x[4])[:8]:
            print("     %-34s %-22s none@%-4d a2@%-4d Δ%-4d (msgs=%d)"
                  % (t[:34], k[:22], fn, fa, dl, n))
    print("\n  런 태그별:")
    for t, r in per_tag.items():
        print("     %-44s sim %3d | 참 none %3d ↔ a2 %3d | 뒤집힘 %2d"
              % (t[:44], r["sims"], r["true_none"], r["true_a2"], r["flip"]))
    return {"sims": tot_s, "true_none": tn, "true_a2": ta,
            "flip": len(flip_sims), "delayed": len(delta),
            "flip_examples": flip_sims[:40]}


def main():
    n = 12
    if "--runs" in sys.argv:
        n = int(sys.argv[sys.argv.index("--runs") + 1])
    a2 = G._domain_a2(DOMAIN)
    r0 = s0_binding()
    r1 = s1_predicate()
    files = recent_files(n)
    print("\n[표본] 최근 런 %d 개 (mtime 순 · 세대 미뭉갬 — 아래는 전부 태그별)" % len(files))
    for p in files:
        print("     %s" % F.tag_of_file(p))
    r2 = s2_dup_scope(files, a2)
    r3 = s3_always_on(files, a2)
    print("\n" + "=" * 98)
    print("판정 ([[77]] 네 칸)")
    print("=" * 98)
    print("  ①주장  : `_a2_of(self)` 가 `unified` 안 **%d 자리 전부**에서 None 이다"
          % len(r0["a2_of_sites"]))
    print("           (`self`=LLMAgent 에 `.environment` 가 없다 · 설치 줄 %s)" % r0["install"])
    print("  ②근거  : §0 실측 None · §1 뒤집히는 도구 %d 종 · 라이브 로그 `deny tool=unlock_…`"
          % len(r1))
    print("  ③반증  : 독스트링 §반증 조건 셋 — 셋 다 이 실행에서 통과했다면 주장 생존")
    print("  ④선행  : ENGINE_LITERAL_REMEDIATION §8-B(예고) · LEVER_ROSTER:248(미해결 격차)")
    print("           · test_c241_u1_predicate.py(순수 함수만 검정 — 배선 칸 없음)")
    print("\n  폭발 반경: DUP_WRITE 표본 밖 발화 **%d 건** · 상시 레버 판정 뒤집힘 **%d sim** / %d"
          % (r2["extra_calls"], r3["flip"], r3["sims"]))
    out = {"binding": r0, "flips": r1, "dup_write": r2, "always_on": r3,
           "sample_runs": [F.tag_of_file(p) for p in files]}
    with io.open(OUT, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(out, ensure_ascii=False, indent=1))
    print("\n  → %s" % os.path.abspath(OUT))


if __name__ == "__main__":
    main()
