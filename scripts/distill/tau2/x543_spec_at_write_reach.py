# -*- coding: utf-8 -*-
r"""x543 - 재생(G1): `T2_SPEC_AT_WRITE`/`T2_RULE_AT_WRITE` 의 **도달 사슬**을 저장 궤적에 다시 태운다.

## 관측 (2026-08-26 · 재유도 아님 · 로그 직독)

t7356 15 배치에서 도달 표지 `[T2_SPEC_DIST]` 가 **14 개에서 0 · grpB1 에서만 1**.
이 표지는 `if _wc is not None:` 안 `if not _darg:` 가지 **맨 위에서 무조건** 인쇄된다
(`t2_gate_patch.py:10989`) ⇒ 세 레버(`SPEC_AT_WRITE`·`RULE_AT_WRITE`·`ARG_POLICY_AT_WRITE`)는
조건을 **볼 기회조차** 없었다.

핸드오프 §8-2 의 가설(*`T2_SPEC_ARG_FACTS` 가 `en_fb` 를 선점해 굶긴다*)은 **코드상 성립하지
않는다**: 저쪽이 채우는 것은 `en_fb`, 이쪽이 채우는 것은 `dw_fb` 이고, 이 가지의 가드 목록
(`t2_gate_patch.py:10925-10932`)에 `en_fb` 는 **들어 있지 않다**.

## 도달 사슬 6칸 · 이 파일이 가르는 칸

    (1) T2_DECIDE_BEFORE_WRITE=1     런 env 에 `=1` (확인됨)
    (2) not do_gate                  <- 재생 불가(런타임 상태)
    (3) 다른 16 개 `*_fb` 가 전부 None <- 재생 불가(런타임 상태)
    (4) _t2_dwrite_deny == 0         **코드로 제거됨** - 대입 4곳이 전부 이 가지 *안*
                                     (:11007 · :11036 · :11067 · :11101) => 가지가 안 돌면 소모도 없다
    (5) _wc is not None              ★이 파일
    (6) not _darg                    ★이 파일

=> **(5)(6) 을 통과하는 sim 수 >> 실제 `SPEC_DIST` 인쇄 수** 이면 원인은 (2)(3) 이다.
   반대로 (5) 나 (6) 에서 대부분이 죽으면 그것이 원인이다([[77]] (3) 반증 조건).

## 경계 ([[62]] · x515 §G)

  G1 술어가 무엇을 돌려주나   - 결정론 · 모델 0회 · **이 파일이 재는 것**
  G3 모델이 다르게 행동하나   - 재생으로 답할 수 없다. 격리 프로브나 런이 필요하다.

## 술어는 전부 **정본에서 import** 한다([[67]] 사본 금지 · [[59]] 패턴매칭 0)

    `_confirm_write_tools` · `_write_choice_arg` · `_eff_tool_name` · `_exact_tool_name` · `_domain_a2`
    궤적 읽기는 `t2_forensic`(`sims`·`nameof`·`argsof`)뿐. gold 미접촉([[23]]).

실행: PYTHONIOENCODING=utf-8 py -3 x543_spec_at_write_reach.py --tags t7356 t7358
"""
import argparse
import collections
import gzip
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import t2_forensic as F                                        # noqa: E402
from t2_gate_patch import (_domain_a2, _confirm_write_tools,    # noqa: E402
                           _write_choice_arg, _eff_tool_name, _exact_tool_name,
                           _env_spec_for, _declared_rules_for)

REPORTS = os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026")
OUT = os.path.join(REPORTS, "x543_spec_at_write_reach_2026_08_26.json")


class _TC(object):
    """저장 궤적의 tool_call(dict) -> 정본 술어가 기대하는 **속성 접근** 모양.

    변환은 `t2_forensic` 의 두 정규화(`nameof`·`argsof`)뿐이다 - 이름 규칙·의미 판단 0."""
    __slots__ = ("name", "arguments")

    def __init__(self, tc):
        self.name = str(F.nameof(tc) or "")
        self.arguments = F.argsof(tc)


class _M(object):
    """`_env_spec_for` 가 보는 것만 있는 메시지 껍데기 — role · content · tool_calls."""
    __slots__ = ("role", "content", "tool_calls")

    def __init__(self, d):
        self.role = d.get("role")
        self.content = d.get("content")
        self.tool_calls = [_TC(t) for t in (d.get("tool_calls") or [])]


def predict(a2, w, prefix):
    """수리 후 **이 자리에서 무엇이 나가나**. 런타임 `state.messages` 는 이 write 직전까지의
    이력이므로 prefix 로 근사한다. 반환 = (레버 이름, 거리, 명세 길이)."""
    _spec, _si, _sd = _env_spec_for(w, prefix)
    _rules = _declared_rules_for(w, a2)
    _sfire = bool(_spec) and _sd >= 8
    # elif 사슬이면 명세가 먼저 이긴다 — **둘 다 있는데 규칙이 죽는 자리**를 세는 것이 요점.
    lever = "SPEC(선점)+RULE(죽음)" if (_sfire and _rules) else (
        "SPEC" if _sfire else ("RULE" if _rules else "(무발화)"))
    return lever, _sd, "명세 %d · 규칙 %d" % (len(_spec) if _spec else 0,
                                             len(_rules) if _rules else 0)


def wrset_of(a2, docs_at_write):
    """`t2_gate_patch.py:10955-10962` 의 그 집합. `T2_DOCS_AT_WRITE` 는 런에서 **0** 이었다."""
    s = set(_confirm_write_tools(a2)) | set(
        ((a2 or {}).get("eplan") or {}).get("write_tools") or [])
    if docs_at_write:
        s |= {c.get("tool") for c in ((a2 or {}).get("choice_grounding") or []) if c.get("tool")}
        rv = (a2 or {}).get("recommendation_verify") or {}
        if rv.get("action_tool"):
            s.add(rv["action_tool"])
    return {x for x in s if x}


def batches(tag):
    out = []
    for p in F.all_result_files():
        t = F.tag_of_file(p)
        if tag in t:
            out.append(t)
    return sorted(set(out))


def observed_spec_dist(batch):
    """로그에 실제로 인쇄된 `[T2_SPEC_DIST]` 줄 수. 없으면 None(=회수 안 됨·[[30]])."""
    p = os.path.join(REPORTS, "sim_results", batch + ".log.gz")
    if not os.path.exists(p):
        return None
    n = 0
    with gzip.open(p, "rt", encoding="utf-8", errors="replace") as fh:
        for ln in fh:
            if "T2_SPEC_DIST" in ln:
                n += 1
    return n


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", default=["t7356"])
    ap.add_argument("--domain", default="banking_knowledge")
    a = ap.parse_args(argv)

    a2 = _domain_a2(a.domain)
    if not a2:
        print("A2 를 못 읽었다 (domain=%s) - 판정 불가([[25]])" % a.domain)
        return 2
    base = wrset_of(a2, False)
    withdocs = wrset_of(a2, True)
    print("=" * 96)
    print("x543 재생 - 도달 사슬 (5)(6) · domain=%s" % a.domain)
    print("=" * 96)
    print("_wrset (DOCS_AT_WRITE=0 · 런과 같음) = %d 개: %s" % (len(base), sorted(base)))
    print("_wrset (DOCS_AT_WRITE=1 이면)        = %d 개 (+%d)"
          % (len(withdocs), len(withdocs - base)))
    if withdocs - base:
        print("   추가분: %s" % sorted(withdocs - base))

    report = {"domain": a.domain, "wrset_base": sorted(base),
              "wrset_docs_extra": sorted(withdocs - base), "tags": {}}

    for tag in a.tags:
        bs = batches(tag)
        if not bs:
            print("\n%s: 회수된 배치 0 - 판정 불가([[30]])" % tag)
            report["tags"][tag] = {"batches": 0}
            continue
        tot_calls = tot_member = tot_named = tot_unnamed = 0
        sims_any = sims_reach = sims_member = 0
        per_tool = collections.Counter()
        per_arg = collections.Counter()
        pred_ct = collections.Counter()
        pred_rows = []
        obs_total = 0
        obs_known = 0
        rows = []
        for b in bs:
            try:
                ss = F.sims(b, ".results.json.gz")
            except Exception as e:
                print("  [%s] 결과 못 읽음: %r" % (b, e))
                continue
            o = observed_spec_dist(b)
            if o is not None:
                obs_total += o
                obs_known += 1
            for s in ss:
                sims_any += 1
                reach = member = 0
                msgs = s.get("messages") or []
                for _mi, m in enumerate(msgs):
                    if str(m.get("role")) != "assistant":
                        continue
                    for tc in (m.get("tool_calls") or []):
                        tot_calls += 1
                        w = _TC(tc)
                        eff, exact = _eff_tool_name(w), _exact_tool_name(w)
                        if not (eff in base or exact in base or w.name in base):
                            continue
                        tot_member += 1
                        member += 1
                        if member == 1:
                            # sim 당 상한이 1이므로 **첫 자리**가 실제로 나가는 자리다.
                            _lv, _dist, _ln = predict(a2, w, [_M(x) for x in msgs[:_mi]])
                            pred_ct[_lv] += 1
                            pred_rows.append({"batch": b, "task": s.get("task_id"),
                                              "tool": eff, "lever": _lv,
                                              "dist": _dist, "len": _ln})
                        darg, _dax = _write_choice_arg(a2, w)
                        per_tool[(eff, bool(darg))] += 1
                        if darg:
                            tot_named += 1
                            per_arg[str(darg)] += 1
                        else:
                            tot_unnamed += 1
                            reach += 1
                if member:
                    sims_member += 1
                if reach:
                    sims_reach += 1
                    rows.append({"batch": b, "task": s.get("task_id"),
                                 "unnamed_calls": reach})
        print("\n" + "-" * 96)
        print("[%s] 배치 %d · sim %d" % (tag, len(bs), sims_any))
        print("  도구 호출 전량                      %6d" % tot_calls)
        print("  (5) _wrset 안 (=_wc is not None)    %6d" % tot_member)
        print("      (6) _darg 있음 -> **가지 밖**    %6d" % tot_named)
        print("      (6) _darg 없음 -> 가지 안(도달)  %6d" % tot_unnamed)
        print("  => 도달했어야 할 sim 수             %6d / %d" % (sims_reach, sims_any))
        print("  => **수리 후** 자리가 생기는 sim 수 %6d / %d  (=_wrset 안 호출이 1건 이상)"
              % (sims_member, sims_any))
        print("  => 로그에 실제 인쇄된 SPEC_DIST     %6d  (로그 회수 배치 %d/%d)"
              % (obs_total, obs_known, len(bs)))
        if per_tool:
            print("  도구별 (이름 · _darg 있음?) -> 호출수:")
            for (t, named), n in per_tool.most_common(12):
                print("     %-46s darg=%-5s %4d" % (t[:46], named, n))
        if per_arg:
            print("  _darg 로 지목된 인자 이름: %s" % dict(per_arg.most_common(8)))
        if pred_rows:
            print("  ── 수리 후 **첫 자리**에서 나갈 것 (sim 당 상한 1):")
            for r in pred_rows:
                print("     %-9s %-40s %-16s dist=%-4s len=%s"
                      % (r["task"], r["tool"][:40], r["lever"], r["dist"], r["len"]))
            print("     합계: %s" % dict(pred_ct))
        report["tags"][tag] = {
            "batches": len(bs), "sims": sims_any, "calls": tot_calls,
            "in_wrset": tot_member, "darg_named": tot_named, "darg_none": tot_unnamed,
            "sims_should_reach": sims_reach, "observed_spec_dist": obs_total,
            "log_batches_seen": obs_known,
            "per_tool": {"%s|darg=%s" % (k[0], k[1]): v for k, v in per_tool.items()},
            "per_arg": dict(per_arg), "rows": rows[:40],
        }

    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    print("\n산출: %s" % os.path.abspath(OUT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
