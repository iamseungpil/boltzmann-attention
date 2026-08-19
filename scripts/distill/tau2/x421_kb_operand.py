# -*- coding: utf-8 -*-
r"""x421 - **적합한 정책·KB 를 주면 operand 를 맞추는가** (사용자 지시 2026-08-19)

## x420 R_doc 의 결함
x420 은 문서를 **gold 값 문자열**로 찾았다(`doc_slice(docs, needles)`). 093 의 `33`, 094 의 `140`
같은 값으로 검색하면 무관한 문서가 걸린다 — 그래서 `R_doc ≈ R_neg` 였을 수 있다.
**값이 든 문서**가 아니라 **값을 유도하는 규칙이 든 문서**를 줘야 한다.

## 이번 팔 (전부 재생 격리 위에)
    A_asis    재생 그대로
    B_kb      + **주제 검색으로 뽑은 KB 문서 top-k 전문**
              질의 = 손님 요청 + 도구 이름 (gold 미참조) · 점수 = tf-idf 코사인(결정론)
    C_policy  + 정책 본문(sim["policy"]) 전문 + B_kb
    D_neg     + **무작위 문서 k개**(부정통제 [[57]])

## 사전 고정 해석
- `B_kb`/`C_policy` 에서 EXACT 가 오르면 => **적합한 문서를 주면 값을 맞춘다** ⇒ 병목은 회수/전달.
- 안 오르면 => 문서를 제대로 줘도 못 맞춘다 ⇒ **유도·계산 결손**. 그 단계에만 결정론 후보([[62]]).
- `D_neg` 가 `B_kb` 만큼 오르면 문서 내용이 아니라 **길이/형식**이 작용한 것이다.

⚠검색은 gold 를 안 본다. 질의는 손님 첫 발화 + 도구 이름 + 도구 설명뿐이다.
"""
import argparse
import io
import json
import math
import os
import re
import sys
import threading
import collections

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import t2_forensic as F  # noqa: E402
import x395_compliance_iso as X  # noqa: E402
import x396_saying_vs_doing as C  # noqa: E402
import x397_pvi_channel as P  # noqa: E402

MSG_CAP = 3000
TOTAL_CAP = 60000
DOC_CAP = 4500

SYS_ARG = ("You fill in the arguments for ONE named bank tool call. "
           "Reply with ONE JSON object only: {\"arguments\": {…}}. "
           "Use ONLY the parameter names given in the tool schema. No prose, no markdown fence.")

TOK = re.compile(r"[a-z0-9_]+")


def env_tools():
    d = json.load(io.open(os.path.join(HERE, "a2", "env_surface.json"), encoding="utf-8"))
    return d["banking_knowledge"]["tools"]


def flat(a):
    a = F.norm_args(a)
    if isinstance(a, dict) and isinstance(a.get("arguments"), dict):
        a = a["arguments"]
    if isinstance(a, dict) and isinstance(a.get("arguments"), str):
        try:
            a = json.loads(a["arguments"])
        except Exception:
            pass
    return a if isinstance(a, dict) else {}


class Index(object):
    """tf-idf 코사인 — 결정론·엔진이 답을 만들지 않는다(문서를 고를 뿐)."""

    def __init__(self, docs):
        self.ids = list(docs.keys())
        self.docs = docs
        self.tf, df = [], collections.Counter()
        for i in self.ids:
            t = TOK.findall(((docs[i].get("title") or "") + " " + (docs[i].get("content") or "")).lower())
            c = collections.Counter(t)
            self.tf.append(c)
            for w in c:
                df[w] += 1
        n = float(len(self.ids)) or 1.0
        self.idf = {w: math.log(1.0 + n / (1.0 + v)) for w, v in df.items()}
        self.norm = []
        for c in self.tf:
            s = math.sqrt(sum((v * self.idf.get(w, 0.0)) ** 2 for w, v in c.items())) or 1.0
            self.norm.append(s)

    def top(self, query, k=3):
        q = collections.Counter(TOK.findall(query.lower()))
        out = []
        for j, c in enumerate(self.tf):
            s = sum(v * self.idf.get(w, 0.0) * c.get(w, 0) * self.idf.get(w, 0.0) for w, v in q.items())
            if s > 0:
                out.append((s / self.norm[j], self.ids[j]))
        out.sort(reverse=True)
        return [i for _s, i in out[:k]]

    def render(self, ids):
        parts = []
        for i in ids:
            d = self.docs[i]
            body = " ".join((d.get("content") or "").split())[:DOC_CAP]
            parts.append("[%s]\n%s" % ((d.get("title") or i)[:80], body))
        return "\n\n".join(parts)


def render_msgs(msgs):
    out, cut = [], 0
    for m in msgs:
        c = " ".join(str(m.get("content") or "").split())
        if len(c) > MSG_CAP:
            c = c[:MSG_CAP] + " …[TRUNCATED]"
            cut += 1
        r = m.get("role")
        if r == "assistant":
            if c:
                out.append("ASSISTANT: " + c)
            for tc in (m.get("tool_calls") or []):
                out.append("ASSISTANT_TOOL_CALL: %s %s"
                           % (F.nameof(tc),
                              json.dumps(F.argsof(tc), ensure_ascii=False, default=str)[:1200]))
        elif r == "user":
            out.append("CUSTOMER: " + c)
        elif r == "tool":
            out.append("TOOL_RESULT: " + c)
    t = "\n".join(out)
    trimmed = False
    if len(t) > TOTAL_CAP:
        t = "…[EARLIER OMITTED]…\n" + t[-TOTAL_CAP:]
        trimmed = True
    return t, cut, trimmed


def parse_args_obj(raw):
    t = re.sub(r"^```(?:json)?|```$", "", (raw or "").strip()).strip()
    m = re.search(r"\{.*\}", t, re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except Exception:
        return None
    if not isinstance(d, dict):
        return None
    a = d.get("arguments", d)
    if isinstance(a, str):
        try:
            a = json.loads(a)
        except Exception:
            return None
    return a if isinstance(a, dict) else None


def norm(v):
    s = str(v).strip()
    try:
        return "%g" % float(s)
    except Exception:
        return s.lower()


def score(pred, ref):
    if pred is None or not ref:
        return 0.0, False
    same = [k for k in ref if k in pred and norm(pred[k]) == norm(ref[k])]
    return len(same) / float(len(ref)), len(same) == len(ref)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--port", type=int, default=8141)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--smoke", type=int, default=0)
    a = ap.parse_args()

    TOOLS = env_tools()
    MUT = {k for k, v in TOOLS.items() if v.get("mutates")}
    docs = X.load_docs()
    idx = Index(docs)
    print("KB 문서 %d · tf-idf 색인 완료" % len(docs), flush=True)

    cases = []
    for tag in C.TAGS:
        for sim in F.scored(tag, C.SUF):
            if ((sim.get("reward_info") or {}).get("reward") or 0) >= 1.0:
                continue
            msgs = sim.get("messages") or []
            live = {}
            for i, m in enumerate(msgs):
                for tc in (m.get("tool_calls") or []):
                    ar = F.argsof(tc)
                    nm = str(F.inner_name(ar) or F.nameof(tc))
                    fa = flat(ar)
                    if nm in MUT and fa and nm not in live:
                        live[nm] = i
            for g in C.gold_rows(sim):
                nm = g["name"]
                if nm not in MUT or nm not in TOOLS:
                    continue
                ga = flat(g["args"])
                if not ga or set(ga.keys()) <= {"agent_tool_name", "discoverable_tool_name"}:
                    continue
                cases.append({"task": F.task_id(sim), "trial": sim.get("trial"), "tool": nm,
                              "gold": ga, "cut": live.get(nm, X.close_index(sim)), "sim": sim})
    if a.smoke:
        seen, pick = set(), []
        for c in cases:
            k = (c["task"], c["tool"])
            if k in seen:
                continue
            seen.add(k)
            pick.append(c)
            if len(pick) >= a.smoke:
                break
        cases = pick
    print("표적 %d건" % len(cases), flush=True)

    rnd_ids = [idx.ids[(i * 137) % len(idx.ids)] for i in range(a.k)]
    neg_txt = idx.render(rnd_ids)

    jobs = []
    for c in cases:
        sim = c["sim"]
        spec = TOOLS[c["tool"]]
        body, ncut, trimmed = render_msgs((sim.get("messages") or [])[:c["cut"]])
        schema = ("# 도구 스키마(환경 선언)\n%s\n  파라미터: %s\n  설명: %s\n"
                  % (c["tool"], ", ".join(spec.get("args") or []),
                     " ".join(str(spec.get("desc") or "").split())[:400]))
        # ★질의는 gold 를 안 본다 — 손님 요청 + 도구 이름 + 도구 설명
        q = "%s %s %s" % (X.user_ask(sim), c["tool"].replace("_", " "),
                          " ".join(str(spec.get("desc") or "").split())[:200])
        kb_ids = idx.top(q, a.k)
        kb_txt = idx.render(kb_ids)
        c["_kb"] = [(docs[i].get("title") or i)[:60] for i in kb_ids]
        c["_ncut"], c["_trim"] = ncut, trimmed
        head = "# 지금까지의 대화(원문 재생)\n" + body + "\n\n" + schema
        ask = ("\n\n# 질문\n바로 지금 `%s` 를 호출한다. **인자를 정확한 값으로** 채워라. "
               "스키마의 파라미터 이름만 쓴다. JSON 하나로만: {\"arguments\": {…}}" % c["tool"])
        pol = " ".join(str(sim.get("policy") or "").split())[:8000]
        arms = {
            "A_asis": head + ask,
            "B_kb": head + "\n# 관련 지식베이스 문서(주제 검색 top-%d)\n" % a.k + kb_txt + "\n" + ask,
            "C_policy": head + "\n# 정책 전문\n" + pol + "\n\n# 관련 지식베이스 문서\n" + kb_txt + "\n" + ask,
            "D_neg": head + "\n# 관련 지식베이스 문서\n" + neg_txt + "\n" + ask,
        }
        for an, bd in arms.items():
            for kk in range(a.n):
                jobs.append({"c": c, "arm": an, "k": kk, "temp": (0.0 if kk == 0 else a.temp),
                             "msgs": [{"role": "system", "content": SYS_ARG},
                                      {"role": "user", "content": bd}]})
    print("작업 %d건 · 절단 %d · 앞부분 생략 %d"
          % (len(jobs), sum(c["_ncut"] for c in cases), sum(1 for c in cases if c["_trim"])),
          flush=True)
    print("검색 예시:", flush=True)
    for c in cases[:4]:
        print("   %-9s %-40s -> %s" % (c["task"], c["tool"][:40], " | ".join(c["_kb"])), flush=True)

    lock, out = threading.Lock(), []

    def work(_i):
        while True:
            with lock:
                if not jobs:
                    return
                j = jobs.pop(0)
            try:
                d = P.post(a.port, "/v1/chat/completions",
                           {"model": X.MODEL, "messages": j["msgs"],
                            "temperature": j["temp"], "max_tokens": 400})
                raw = d["choices"][0]["message"]["content"]
            except Exception as e:
                raw = "ERROR " + str(e)[:160]
            pr = parse_args_obj(raw)
            gp, ge = score(pr, j["c"]["gold"])
            with lock:
                out.append({"task": j["c"]["task"], "trial": j["c"]["trial"], "tool": j["c"]["tool"],
                            "arm": j["arm"], "k": j["k"], "parsed": pr is not None,
                            "gold_part": gp, "gold_exact": ge,
                            "kb": j["c"]["_kb"], "raw": raw[:220]})
                if len(out) % 60 == 0:
                    print("  ... %d/%d" % (len(out), len(out) + len(jobs)), flush=True)

    ths = [threading.Thread(target=work, args=(i,)) for i in range(a.workers)]
    [t.start() for t in ths]
    [t.join() for t in ths]

    print("\n## 팔별")
    print("%-10s %6s %11s %11s %8s" % ("arm", "n", "GOLD_EXACT", "GOLD_PART", "PARSED"))
    for arm in ("A_asis", "D_neg", "B_kb", "C_policy"):
        r = [x for x in out if x["arm"] == arm]
        if not r:
            continue
        n = float(len(r))
        print("%-10s %6d %11.3f %11.3f %8.2f"
              % (arm, len(r), sum(x["gold_exact"] for x in r) / n,
                 sum(x["gold_part"] for x in r) / n, sum(x["parsed"] for x in r) / n))

    print("\n## 쌍대 부호검정 (표적별 GOLD_PART)")
    K = lambda x: (x["task"], x["trial"], x["tool"])
    tg = sorted(set(K(x) for x in out))
    for A, B in (("B_kb", "A_asis"), ("C_policy", "A_asis"), ("B_kb", "D_neg")):
        up = dn = eq = 0
        for t in tg:
            xa = [x["gold_part"] for x in out if K(x) == t and x["arm"] == A]
            xb = [x["gold_part"] for x in out if K(x) == t and x["arm"] == B]
            if not xa or not xb:
                continue
            va, vb = sum(xa) / len(xa), sum(xb) / len(xb)
            up += va > vb + 1e-9
            dn += va < vb - 1e-9
            eq += abs(va - vb) <= 1e-9
        print("  %-10s vs %-10s 상승 %2d / 하락 %2d / 동률 %2d" % (A, B, up, dn, eq))

    print("\n## 표적별 (A_asis / D_neg / B_kb / C_policy) GOLD_PART")
    for t in tg:
        cells = []
        for arm in ("A_asis", "D_neg", "B_kb", "C_policy"):
            r = [x["gold_part"] for x in out if K(x) == t and x["arm"] == arm]
            cells.append("%.2f" % (sum(r) / len(r)) if r else " -  ")
        kb = next((x["kb"] for x in out if K(x) == t), [])
        print("  %-9s t%-2s %-40s %s   KB: %s"
              % (t[0], t[1], t[2][:40], "  ".join(cells), " | ".join(kb)[:70]))

    o = os.path.normpath(os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026",
                                      "x421_kb_operand%s.json" % ("_smoke" if a.smoke else "")))
    io.open(o, "w", encoding="utf-8").write(json.dumps(out, ensure_ascii=False, indent=1))
    print("\n원자료: %s" % o)
    return 0


if __name__ == "__main__":
    sys.exit(main())
