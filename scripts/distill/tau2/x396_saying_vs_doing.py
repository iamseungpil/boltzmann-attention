# -*- coding: utf-8 -*-
r"""x396 — **말했나 / 안 말했나 / 잘못 불렀나**: 세 결손을 섞지 않고 궤적으로 가른다.

## 왜 (사용자 교정 2026-08-19)
> *"knowing–doing 은 **알지만 도구호출로 안 하고 text 로 한다**는 것 아닌가? 문제를 섞지 말고 명확하게 하라."*

맞다. 앞서 나는 세 가지를 한 이름으로 묶어 말했다. 여기서 **관측 가능한 정의**로 가른다 —
판정은 전부 궤적 축자이고, 새 이름을 만들지 않는다([[48]] 4축: 관측실패/원인 분리).

## 정의 (미매치 gold 액션 하나마다 **정확히 하나**로 배정 · 사전 고정 우선순위)

    ① SAID_NOT_DONE   그 도구/행위를 **어시스턴트 본문이 말했는데** 호출은 0회   ← knowing–doing 그 자체
       · 하위 CLAIMED_DONE : 본문이 **이미 했다고 주장**까지 함(거짓 완료)
       · 하위 SAID_INTENT  : 하겠다/해야 한다고만 말함
    ② NEVER_MENTIONED  본문에도 없고 호출도 0회                                  ← 계획·국면 결손(말한 적조차 없다)
    ③ MISCALLED        호출은 했는데 인자가 어긋남                                ← 선택·형식화 결손

⚠①은 **채널 대체**(말 ↔ 호출)이고 ②는 **계획 부재**, ③은 **operand 오선택**이다. 서로 다른 결손이고
  레버도 다르다. 이 표가 나오기 전에는 "knowing–doing 이 지배적" 같은 말을 하지 않는다([[08]]).

## 말했다의 판정 (결정론 · 두 신호만 · 해석 0)
    ⓐ 도구 이름 축자 등장 (본문 content 안)
    ⓑ 완료/의도 문형 + **그 행위의 목적어 토큰**(계좌·카드·금액 id) 동시 등장
       — 문형은 도메인 일반 영어 표현 목록으로 고정하고 목록을 인쇄한다(포렌식 전용·런타임 레버 아님).

사용: py -3 x396_saying_vs_doing.py [<tag> …]
"""
import collections
import io
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import t2_forensic as F  # noqa: E402

SUF = ".results.json.gz"
TAGS = ["bank_t7326_halfA_20260819q", "bank_t7326_halfB_20260819q"]

# 완료 주장 문형(도메인 일반·축자 목록 · 포렌식 전용)
DONE_PAT = [r"has been (?:successfully )?(?:applied|submitted|filed|processed|completed|opened|closed|ordered|credited|deposited|updated|transferred)",
            r"have been (?:successfully )?(?:applied|submitted|filed|processed|completed|credited|updated)",
            r"I have (?:applied|submitted|filed|processed|completed|opened|closed|ordered|credited|updated|transferred)",
            r"(?:successfully )?(?:applied|submitted|filed|processed) (?:the |a )",
            r"is now (?:open|closed|frozen|active|submitted|applied)"]
INTENT_PAT = [r"I (?:will|'ll|am going to|need to|should|must) (?:now )?(?:apply|submit|file|process|open|close|order|credit|deposit|update|check|review|retrieve|look up|pull up|verify)",
              r"[Ll]et me (?:apply|submit|file|process|open|close|order|credit|check|review|retrieve|look up|pull)",
              r"[Nn]ext,? I(?:'ll| will)",
              r"[Ww]e (?:will|need to|should) (?:apply|submit|file|open|close|order|credit|check|review)"]
DONE_RE = re.compile("|".join(DONE_PAT))
INTENT_RE = re.compile("|".join(INTENT_PAT))


def gold_rows(sim):
    out, seen = [], set()
    for ck in ((sim.get("reward_info") or {}).get("action_checks") or []):
        a = ck.get("action") or {}
        ar = a.get("arguments") or {}
        nm = (ar.get("agent_tool_name") or ar.get("user_tool_name")
              or ar.get("discoverable_tool_name") or a.get("name") or "?")
        inner = ar.get("arguments", None)
        aid = str(a.get("action_id") or "")
        k = aid or (str(nm) + json.dumps(ar, ensure_ascii=False, sort_keys=True))
        if k in seen:
            continue
        seen.add(k)
        out.append({"name": str(nm), "type": ck.get("tool_type"), "match": bool(ck.get("action_match")),
                    "args": ar if inner is None else inner, "aid": aid})
    return out


def called(sim):
    out = collections.Counter()
    for _m, tc in F.calls(sim):
        out[str(F.inner_name(F.argsof(tc)) or F.nameof(tc))] += 1
    return out


def assistant_texts(sim):
    return [str(m.get("content") or "") for m in (sim.get("messages") or [])
            if m.get("role") == "assistant" and (m.get("content") or "")]


def operand_tokens(args):
    """그 gold 액션의 목적어 토큰(계좌·카드·거래 id 와 금액)."""
    out = set()
    a = F.norm_args(args)
    if isinstance(a, dict) and isinstance(a.get("arguments"), dict):
        a = a["arguments"]
    if not isinstance(a, dict):
        return out
    for k, v in a.items():
        s = str(v)
        if re.match(r"^(?:chk|sav|dbc|txn|cc|acc)_[A-Za-z0-9_]+$", s):
            out.add(s)
        elif k in ("amount", "amount_difference") and s:
            try:
                out.add("%g" % float(s))
            except Exception:
                pass
    return out


def main():
    tags = [a for a in sys.argv[1:] if not a.startswith("--")] or TAGS
    rows = []
    for tag in tags:
        for sim in F.scored(tag, SUF):
            rw = (sim.get("reward_info") or {}).get("reward")
            task, trial = F.task_id(sim), sim.get("trial")
            cn = called(sim)
            texts = assistant_texts(sim)
            body = " ".join(" ".join(t.split()) for t in texts)
            for g in gold_rows(sim):
                if g["match"]:
                    continue
                nm = g["name"]
                if cn.get(nm):
                    code, ev = "MISCALLED", "호출 %d회" % cn[nm]
                else:
                    named = nm in body
                    ops = operand_tokens(g["args"])
                    hit_done, hit_int, ev = None, None, ""
                    for t in texts:
                        t1 = " ".join(t.split())
                        if not ops or any(o in t1 for o in ops) or named:
                            if hit_done is None and DONE_RE.search(t1):
                                m = DONE_RE.search(t1)
                                hit_done = t1[max(0, m.start() - 40):m.start() + 90]
                            if hit_int is None and INTENT_RE.search(t1):
                                m = INTENT_RE.search(t1)
                                hit_int = t1[max(0, m.start() - 20):m.start() + 90]
                    if hit_done:
                        code, ev = "CLAIMED_DONE", hit_done
                    elif named or hit_int:
                        code, ev = "SAID_INTENT", (hit_int or ("이름 축자 언급: " + nm))
                    else:
                        code, ev = "NEVER_MENTIONED", ""
                rows.append({"tag": tag, "task": task, "trial": trial, "reward": rw,
                             "name": nm, "type": g["type"], "code": code, "ev": ev[:120]})

    print("=" * 112)
    print("x396 · 말했나/안 말했나/잘못 불렀나 · 태그 %s" % ", ".join(tags))
    print("완료 문형 %d · 의도 문형 %d (축자 목록·포렌식 전용)" % (len(DONE_PAT), len(INTENT_PAT)))
    print("=" * 112)

    c = collections.Counter(r["code"] for r in rows)
    ct = collections.Counter((r["code"], r["type"]) for r in rows)
    types = sorted({r["type"] for r in rows}, key=str)
    print("\n## 미매치 gold %d건의 배정 (실패 sim 포함 전량)" % len(rows))
    print("%-16s %s  합계" % ("", " ".join("%-8s" % str(t)[:8] for t in types)))
    for code in ("CLAIMED_DONE", "SAID_INTENT", "NEVER_MENTIONED", "MISCALLED"):
        if c[code]:
            print("%-16s %s  %d" % (code, " ".join("%-8d" % ct[(code, t)] for t in types), c[code]))
    kd = c["CLAIMED_DONE"] + c["SAID_INTENT"]
    print("\n  ⇒ ①knowing–doing(말했는데 호출 0) = %d  ②계획 부재(언급조차 없음) = %d  ③오선택 = %d"
          % (kd, c["NEVER_MENTIONED"], c["MISCALLED"]))

    print("\n## ① 실물 — 말했는데 안 한 자리 (축자)")
    for r in rows:
        if r["code"] in ("CLAIMED_DONE", "SAID_INTENT") and (r["reward"] or 0) < 1.0:
            print("  %-9s t%-2s %-11s %-42s %s"
                  % (r["task"], r["trial"], r["code"], r["name"][:42], r["ev"]))

    print("\n## 태스크×시행 배정")
    print("%-9s %-3s %-5s %s" % ("task", "tr", "rw", "배정"))
    for k in sorted({(r["task"], str(r["trial"])) for r in rows}):
        rs = [r for r in rows if r["task"] == k[0] and str(r["trial"]) == k[1]]
        d = collections.Counter(r["code"] for r in rs)
        print("%-9s %-3s %-5s %s" % (k[0], k[1], rs[0]["reward"],
                                     " ".join("%s=%d" % (a, b) for a, b in sorted(d.items()))))

    out = os.path.normpath(os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026",
                                        "x396_saying_vs_doing.json"))
    io.open(out, "w", encoding="utf-8").write(json.dumps(rows, ensure_ascii=False, indent=1))
    print("\n원자료: %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
