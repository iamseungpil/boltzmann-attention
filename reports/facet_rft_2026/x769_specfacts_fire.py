# -*- coding: utf-8 -*-
r"""x769 - 맡은 레버군 `T2_SPEC_AT_WRITE` + `T2_SPEC_ARG_FACTS` 의 **발화 실측**.

[[78]] 격리->배선: 프롬프트를 쓰지 않고 **엔진 함수 자체**(`t2_gate_patch`)를 부른다.
[[69]]: 채점축은 results 의 reward_info.reward 를 그대로 읽는다(재계산 0).
GPU 0 - 회수된 캠페인 results.json.gz 만 읽는다.
"""
import os, sys, io, gzip, json, glob, collections

SIMS = r"C:\workspace\ba-frft\reports\facet_rft_2026\sim_results"
ENG  = r"C:\workspace\ba-frft\scripts\distill\tau2"
sys.path.insert(0, ENG)
import t2_gate_patch as G

LO, HI = "2026-09-03T14", "2026-09-05T03"


class TC(object):
    def __init__(self, d):
        self.name = d.get("name") or ""
        self.arguments = d.get("arguments")
        self.id = d.get("id")


class M(object):
    def __init__(self, d):
        self.role = d.get("role") or ""
        self.content = d.get("content")
        self.tool_calls = [TC(t) for t in (d.get("tool_calls") or [])]


def load_campaign():
    best = {}
    for p in glob.glob(os.path.join(SIMS, "*.results.json.gz")):
        try:
            d = json.load(gzip.open(p, "rt", encoding="utf-8"))
        except Exception:
            continue
        for s in (d.get("simulations") or []):
            ts = str(s.get("start_time") or s.get("timestamp") or "")
            if not (LO <= ts[:13] <= HI):
                continue
            t = s.get("task_id")
            if t not in best or ts > best[t][0]:
                best[t] = (ts, os.path.basename(p), s)
    return best


def specfacts_hits(msgs):
    """T2_SPEC_ARG_FACTS 의 두 술어를 **엔진 코드 그대로** 재적용. sim 당 캡도 재현."""
    seen = set()
    type_hits, enum_hits = [], []
    for i, m in enumerate(msgs):
        if m.role != "assistant" or not m.tool_calls:
            continue
        dpt = G._declared_params_by_tool(msgs[:i])     # 이 턴까지의 문맥
        for c in m.tool_calls:
            tn = str(G._exact_tool_name(c) or "")
            d2 = dpt.get(tn) or {}
            if not d2:
                continue
            av = dict(G._prov_scan_args(c, selectors=None))
            bad = [k for k, v in av.items()
                   if (d2.get(k) or ("", []))[0] == "boolean" and not isinstance(v, bool)]
            if bad and (tn, "\0bool") not in seen:
                seen.add((tn, "\0bool"))
                type_hits.append((i, tn, sorted(bad),
                                  [av.get(k) for k in sorted(bad)]))
                break                                   # 엔진: en_fb 설정 후 break
            fired = False
            for ek, ev in sorted(av.items()):
                en3 = (d2.get(ek) or ("", []))[1]
                es = str(ev).strip()
                if not en3 or not es or es in en3:
                    continue
                if (tn, ek, es) in seen:
                    continue
                seen.add((tn, ek, es))
                enum_hits.append((i, tn, ek, es, en3))
                fired = True
                break
            if fired:
                break
    return type_hits, enum_hits


def specatwrite_hits(msgs, wrset):
    """T2_SPEC_AT_WRITE 의 술어(`_env_spec_for` + dist>=MIN + 도구당 1회).
    ⚠라이브에서는 이 자리가 `T2_DECIDE_BEFORE_WRITE` 안에 있어 **안 돈다** - 반사실치다."""
    seen, hits = set(), []
    MIN = 8
    for i, m in enumerate(msgs):
        if m.role != "assistant" or not m.tool_calls:
            continue
        wc = next((c for c in m.tool_calls
                   if G._eff_tool_name(c) in wrset or (c.name or "") in wrset), None)
        if wc is None:
            continue
        spec, si, sd = G._env_spec_for(wc, msgs[:i])
        if not spec or sd < MIN:
            hits.append((i, G._eff_tool_name(wc), si, sd, 0, "MISS"))
            continue
        k = str(G._exact_tool_name(wc) or "")
        if not k or k in seen:
            hits.append((i, G._eff_tool_name(wc), si, sd, len(spec), "CAPPED"))
            continue
        seen.add(k)
        hits.append((i, G._eff_tool_name(wc), si, sd, len(spec), "FIRE"))
    return hits


if __name__ == "__main__":
    best = load_campaign()
    rows = []
    for t, (ts, f, s) in sorted(best.items()):
        r = ((s.get("reward_info") or {}).get("reward"))
        rows.append((t, ts, f, r, s))
    npass = sum(1 for r in rows if (r[3] or 0) >= 1.0)
    print("[x769] 캠페인 재구성: 태스크 %d · pass %d · fail %d"
          % (len(rows), npass, len(rows) - npass))
    out = {}
    for t, ts, f, r, s in rows:
        if (r or 0) >= 1.0:
            continue
        msgs = [M(x) for x in (s.get("messages") or [])]
        th, eh = specfacts_hits(msgs)
        out[t] = dict(tag=f, sim=s.get("id"), reward=r, nmsg=len(msgs),
                      type_hits=th, enum_hits=eh)
    json.dump(out, io.open(os.path.join(os.path.dirname(__file__),
                                        "x769_out.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    nt = sum(len(v["type_hits"]) for v in out.values())
    ne = sum(len(v["enum_hits"]) for v in out.values())
    print("[x769] SPEC_ARG_FACTS 실패 %d sim: type-deny %d · enum-deny %d"
          % (len(out), nt, ne))
    print("[x769] type 발화 태스크:",
          sorted(k for k, v in out.items() if v["type_hits"]))
    print("[x769] enum 발화 태스크:",
          sorted(k for k, v in out.items() if v["enum_hits"]))
