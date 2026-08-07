# -*- coding: utf-8 -*-
"""x132 — 파생-사실 DAG **단계 0 게이트**: 노드 재서술이 현행과 **같은 값**을 내는가 (유료 0).

정본 = `FACT_DAG_DESIGN_2026_08_08.md` rev4 §7·§7a.

게이트의 정확한 범위(§7a):
  · 대조 대상 = **값**(노드 출력)이지 렌더 문자열이 아니다. 현행 렌더는 한 템플릿이 네 값을 한
    문자열로 조립하므로(`t2_ledger.py:363`) 노드별 `text`로 가면 구조적으로 달라진다 —
    그것을 게이트로 두면 단계 0이 통과 불가가 되고, *"단계 0을 건너뛰지 않는다"* 가 무력해진다.
  · `formalize` 노드는 LLM을 부르므로 값 동일이 성립할 수 없다 ⇒ **같은 모델 응답을 양쪽에**
    먹이고(캐시) **엔진 측 처리**만 본다.

★그래서 이 프로브는 발췌 규칙 차이를 **일부러 섞지 않는다**. 캐시 키는 프롬프트 이름이므로 현행
경로와 노드 경로가 **같은 raw**를 받는다. §1a(단일 발췌 규칙)가 만드는 거동 변화는 단계 1의
게이트⒞에서 따로 잰다 — 한 단계에 두 변경을 넣으면 귀속이 사라진다(§7 ⚠와 같은 규율).

usage (리모트·vllm 필요):
  x132_factdag_stage0.py --dirs bank_stack_lim_20260808n,bank_stack_dp_20260808p \
      [--cache x132_raw_cache.json] [--base http://localhost:8140/v1] [--model …]
캐시가 채워져 있으면 `--base` 없이도 돈다(무료·재현 가능).
"""

import argparse
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
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, HERE)

import t2_ledger as LG              # noqa: E402
import t2_factdag as F              # noqa: E402
from gate_interpreter import load_domain_a2   # noqa: E402


def _load(dirname):
    cands = [os.path.join(REPO, "reports", "facet_rft_2026", "sim_results", dirname + ".json.gz")]
    cands += glob.glob(os.path.join(os.path.expanduser("~"), "scratch", "tau2-bench",
                                    "data", "simulations", dirname, "results.json"))
    for p in cands:
        if os.path.exists(p):
            op = gzip.open if p.endswith(".gz") else open
            with op(p, "rt", encoding="utf-8", errors="replace") as fh:
                return json.load(fh), p
    raise SystemExit("결과를 못 찾았다: %s" % dirname)


def _fam(n):
    return re.sub(r"_\d+$", "", str(n or ""))


def _tool_outputs(sim):
    """(가족 이름 → 가장 최근 반환 본문). 디스패처면 안쪽 이름으로 푼다."""
    by_id, calls, order = {}, {}, []
    for m in sim.get("messages") or []:
        for tc in (m.get("tool_calls") or []):
            args = tc.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {}
            inner = (args or {}).get("agent_tool_name") or ""
            nm = str(tc.get("name") or "")
            calls[tc.get("id")] = inner if (nm.startswith("call_") and inner) else nm
            order.append(tc.get("id"))
        if m.get("role") == "tool":
            by_id[m.get("id")] = m
    out = {}
    for cid in order:                       # 뒤에 온 것이 앞을 덮는다 = "가장 최근"
        msg, name = by_id.get(cid), calls.get(cid)
        if msg is None or not name:
            continue
        c = msg.get("content")
        if isinstance(c, list):
            c = "\n".join(str(x) for x in c)
        out[_fam(name)] = str(c or "")
    return out


def _texts(sim):
    out = []
    for m in sim.get("messages") or []:
        if m.get("role") not in ("tool", "user"):
            continue
        c = m.get("content")
        if isinstance(c, list):
            c = "\n".join(str(x) for x in c)
        out.append(str(c or ""))
    return out


def _owning_spec(a2, node):
    """프롬프트 이름 → 그 이름을 가진 선언. **이름 공간이 아직 선언별**이라 규칙을 명시한다.

    ⓐ 노드의 `tool:` 입력이 가리키는 선언에 그 이름이 있으면 그것
    ⓑ 없으면 그 이름을 가진 선언이 **하나뿐일 때만** 그것
    ⓒ 둘 이상이면 **바이트 동일할 때만** 채택(아니면 모호 = 에러)
    (`now_prompt`가 두 선언에 복제돼 있어 ⓒ가 실제로 필요하다 — 확인 결과 바이트 동일.)
    """
    key = node.get("prompt")
    specs = list((a2 or {}).get("ledger_metrics") or [])
    tool = next((i[5:] for i in (node.get("inputs") or ()) if str(i).startswith("tool:")), None)
    if tool:
        for s in specs:
            if s.get("trigger_tool") == tool and s.get(key):
                return s
    have = [s for s in specs if s.get(key)]
    if not have:
        raise SystemExit("프롬프트 %r 를 가진 선언이 없다" % key)
    if len({s[key] for s in have}) > 1:
        raise SystemExit("프롬프트 %r 가 선언마다 다르다 — 노드가 어느 것을 쓸지 모호하다" % key)
    return have[0]


class Cache(object):
    """(sim, 프롬프트) → 모델 raw 응답. **양쪽 경로가 같은 응답을 본다**(§7a)."""

    def __init__(self, path, agent=None, la=None, UserMessage=None):
        self.path, self.agent, self.la, self.UM = path, agent, la, UserMessage
        self.d = {}
        if path and os.path.exists(path):
            self.d = json.load(io.open(path, encoding="utf-8"))
        self.misses = 0

    def get(self, sim_key, prompt_name, template, text):
        k = "%s::%s" % (sim_key, prompt_name)
        if k in self.d:
            return self.d[k]
        if self.agent is None:
            self.misses += 1
            return None
        prompt = template.format(text=text) if "{text}" in template else template
        if "{keys}" in template:            # 행 전사 프롬프트는 키 목록도 받는다
            return None                     # (아래 rows 경로에서 keys를 채워 부른다)
        raw = self._call(prompt)
        self.d[k] = raw
        return raw

    def get_rows(self, sim_key, prompt_name, template, text, keys):
        k = "%s::%s" % (sim_key, prompt_name)
        if k in self.d:
            return self.d[k]
        if self.agent is None:
            self.misses += 1
            return None
        raw = self._call(template.format(keys=", ".join(keys), text=text))
        self.d[k] = raw
        return raw

    def _call(self, prompt):
        try:
            um = self.UM(role="user", content=prompt)
        except TypeError:
            um = self.UM(content=prompt)
        kw = {k: v for k, v in dict(getattr(self.agent, "llm_args", None) or {}).items()
              if "tool" not in k}
        sub = self.la.generate(model=self.agent.llm, tools=None, messages=[um],
                               call_name="x132_stage0", **kw)
        return getattr(sub, "content", None) or ""

    def save(self):
        if self.path:
            io.open(self.path, "w", encoding="utf-8").write(
                json.dumps(self.d, ensure_ascii=False, indent=1))


class _Agent(object):
    def __init__(self, model, base):
        self.llm = model if model.startswith("openai/") else "openai/" + model
        self.llm_args = {"temperature": 0.0, "api_base": base, "api_key": "dummy"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", required=True)
    ap.add_argument("--tasks", default="")
    ap.add_argument("--domain", default="banking_knowledge")
    ap.add_argument("--cache", default=os.path.join(HERE, "x132_raw_cache.json"))
    ap.add_argument("--base", default="")
    ap.add_argument("--model", default="Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8")
    a = ap.parse_args()

    agent = la = UM = None
    if a.base:
        import tau2.agent.llm_agent as la           # noqa: F811
        from tau2.data_model.message import UserMessage as UM   # noqa: F811
        agent = _Agent(a.model, a.base)
    cache = Cache(a.cache, agent, la, UM)

    a2 = load_domain_a2(a.domain)
    nodes = F.load(a2)
    print("선언 %d노드 · ledger_metrics %d선언 · 캐시 %s"
          % (len(nodes), len(a2.get("ledger_metrics") or []), a.cache))

    want = set(t.strip() for t in a.tasks.split(",") if t.strip())
    checks = fails = skipped = 0

    for dirname in [d.strip() for d in a.dirs.split(",") if d.strip()]:
        data, src = _load(dirname)
        print("\n" + "=" * 90 + "\n== %s (%s)" % (dirname, os.path.basename(src)))
        for si, sim in enumerate(data.get("simulations") or []):
            tid = sim.get("task_id")
            if want and tid not in want:
                continue
            sim_key = "%s::%s::%d" % (dirname, tid, si)
            texts, tools = _texts(sim), _tool_outputs(sim)
            hay = " ".join("\n".join(texts).split())

            # ── 모델 응답을 먼저 캐시에 채운다(양쪽이 같은 것을 본다) ──────────────
            raws = {}
            ok = True
            for n in nodes:
                if n["op"] != "formalize":
                    continue
                spec = _owning_spec(a2, n)
                tpl = spec[n["prompt"]]
                src_in = (n.get("inputs") or [""])[0]
                if src_in.startswith("tool:"):
                    body = tools.get(src_in[5:])
                    if body is None:
                        raws[n["out"]] = None       # 그 도구를 안 불렀다 = 정당한 미계산
                        continue
                    sel, _ = F.excerpt([body])
                    raws[n["out"]] = cache.get_rows(sim_key, n["prompt"], tpl,
                                                    "\n---\n".join(sel),
                                                    list(n["params"]["row_keys"]))
                else:
                    sel, _ = F.excerpt(texts)
                    raws[n["out"]] = cache.get(sim_key, n["prompt"], tpl, "\n---\n".join(sel))
                if raws[n["out"]] is None and (src_in == "corpus" or tools.get(src_in[5:])):
                    ok = False
            if not ok:
                skipped += 1
                print("  %-9s %-10s 캐시 미스 — `--base` 로 채워라" % (tid, "SKIP"))
                continue

            vals, trace = F.evaluate(nodes, F.Inputs(corpus=texts, tools=tools),
                                     ask=lambda n, _t: raws.get(n["out"]))

            # ── 현행 경로: 같은 raw 위에서 현행 함수로 값 만들기 ────────────────────
            cur = {}
            for spec in (a2.get("ledger_metrics") or []):
                trig = spec.get("trigger_tool")
                body = tools.get(trig)
                if body is None:
                    continue
                node = next((n for n in nodes if ("tool:" + trig) in (n.get("inputs") or ())), None)
                rows = LG.parse_rows(raws.get(node["out"]) or "", list(spec["row_keys"]))
                now = LG.parse_scalar(raws.get("today") or "", spec["date_formats"])
                rem, inwin, tally = LG.window_and_tally(rows, spec, now)
                _first, days = LG.earliest_age(rows, spec, now)
                cur[trig] = {"rows": rows, "now": now, "rem": rem, "inwin": inwin,
                             "tally": tally, "days": days}
                if spec.get("limit_prompt"):
                    cur[trig]["limits"] = LG.parse_pairs(raws.get("doc_limits") or "",
                                                         "limit", hay)[0]
                if spec.get("threshold_prompt"):
                    cur[trig]["mins"] = LG.parse_pairs(raws.get("doc_minimums") or "",
                                                       "min_days", hay)[0]

            # ── 대조 ────────────────────────────────────────────────────────────────
            bad = []

            def eq(what, x, y):
                checks_local.append(what)
                if x != y:
                    bad.append("%s: 노드=%r 현행=%r" % (what, x, y))

            checks_local = []
            for trig, c in cur.items():
                node = next(n for n in nodes if ("tool:" + trig) in (n.get("inputs") or ()))
                eq("%s rows" % trig, vals.get(node["out"]) or [], c["rows"])
                if c["tally"] and any(n["op"] == "tally" and n["inputs"][0] == node["out"]
                                      for n in nodes):
                    tn = next(n for n in nodes
                              if n["op"] == "tally" and n["inputs"][0] == node["out"])
                    eq("%s tally" % trig, vals.get(tn["out"]) or {}, c["tally"])
                wn = next((n for n in nodes if n["op"] == "window_remaining"
                           and n["inputs"][0] == node["out"]), None)
                if wn is not None and c["rem"] is not None:
                    eq("%s window" % trig, vals.get(wn["out"]),
                       {"remaining": c["rem"], "used": c["inwin"]})
                dn = next((n for n in nodes if n["op"] == "days_since_earliest"
                           and n["inputs"][0] == node["out"]), None)
                if dn is not None and c["days"] is not None:
                    got = vals.get(dn["out"]) or {}
                    eq("%s days" % trig, got.get("days"), c["days"])
                if "limits" in c:
                    eq("%s limits" % trig, vals.get("doc_limits") or {}, c["limits"])
                if "mins" in c:
                    eq("%s minimums" % trig, vals.get("doc_minimums") or {}, c["mins"])
                eq("%s today" % trig, vals.get("today"), c["now"])

            checks += len(checks_local)
            fails += len(bad)
            print("  %-9s %-4s 대조 %d건%s"
                  % (tid, "FAIL" if bad else "OK", len(checks_local),
                     ("\n      " + "\n      ".join(bad)) if bad else ""))
            for line in F.format_trace(trace).splitlines():
                print("      " + line)

    cache.save()
    print("\n%s 대조 %d건 · 불일치 %d · 스킵 %d%s"
          % ("[단계 0 게이트 PASS]" if (fails == 0 and checks) else "[FAIL]",
             checks, fails, skipped,
             "" if checks else " — 대조 0건이면 통과가 아니다"))
    return 1 if (fails or not checks) else 0


if __name__ == "__main__":
    sys.exit(main())
