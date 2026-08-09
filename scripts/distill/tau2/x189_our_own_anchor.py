# -*- coding: utf-8 -*-
r"""x189 — **우리 자신의 경고 문구가 정박을 설치하는가** (유료 0·[[25]] 직결).

## 왜

출처 추적에서 나온 것: 에이전트가 실제로 읽는 피드백 채널(`fb_*.jsonl.gz`)에 **우리 층이
후보 이름을 부르는 문장을 반복해서 싣고 있다**. `bank_elig_20260809i` 실측:

    [unmatched_text · turn 22·26 · kind=reminder-user]
    "The counts above include Cobalt Blue Account 3; Hunter Green Account 9;
     Navy Blue Account 2, which was NOT checked against any allowance: ..."

`Hunter Green` 은 task_099 의 **정박이자 오답**이다. 그 문장의 취지는 정반대다 —
C327 이 세운 대로 *"판정 못 한 그룹을 조용히 빼면 침묵이 통과로 읽힌다"* 는 경고다.
그러나 등대 §1.3 제1원리대로 **부작용 없는 레버는 없다**: 경고가 이름을 세 개 띄우고,
x184 는 *지목된 이름이 답을 결정한다* 를 이미 세웠다.

⇒ 우리 도구 출력이 유일한 근거원을 오염시키는지 재는 것은 [[25]] 조항 그 자체다.

## 축

  iso            표 + 피연산자 사실 + 질문 (주입 없음)
  +unmatched     위 앞에 **우리 실제 문구 그대로** (fb 채널에서 읽어 온다·리터럴 저작 0)
  +unmatched_nn  같은 문장에서 **이름만 지운 판**(숫자·경고 유지) — 이름이 레버인지 문장이 레버인지

  sort name_asc·name_desc   choices all·chk   task 099·100

⚠ task_100 은 추천 기록이 0건이라 그 문장이 원래 안 나간다 — 여기에 **같은 문장을 이식**하면
   *"남의 문장도 정박을 설치하는가"* 의 통제가 된다(양성이면 태스크 무관 = 형식 효과).

## 읽는 법

  · `+unmatched` 가 iso 를 무너뜨리면    → **우리 경고가 정박을 설치한다**. C327 처방에 반대편 계측 필요.
  · `+unmatched_nn` 이 안 무너뜨리면     → 레버는 **이름 나열**이지 경고 자체가 아니다 → 처방: 이름 없이 세는 법.
  · 둘 다 무너뜨리면                     → 문장 자체(주의 전환)가 레버 → C327 재설계.
  · 100 도 같이 무너지면                 → 태스크 무관 = 형식 효과(더 강한 진술).

실행: python x189_our_own_anchor.py [N]
"""
import collections
import gzip
import json
import os
import re
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import x149_choice_isolation as X                              # noqa: E402
import t2_factdag as FD                                        # noqa: E402
import t2_ledger as LG                                         # noqa: E402
from gate_interpreter import load_domain_a2                     # noqa: E402

URL = os.environ.get("T2_PROBE_URL", "http://localhost:8140/v1/chat/completions")
MODEL = os.environ.get("T2_PROBE_MODEL", "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8")
HERE = os.path.dirname(os.path.abspath(__file__))
SIMS = os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026", "sim_results")
FB = os.environ.get("T2_FB_TAG", "fb_bank_elig_20260809i")
CASE = {"task_099": {"days": 730, "deposit": 30000},
        "task_100": {"days": 65, "deposit": 31000}}
ACCOUNT_AXIS = "referrer_tenure_days"


def guided_full(prompt, choices, temp):
    body = json.dumps({"model": MODEL, "temperature": temp, "max_tokens": 12,
                       "guided_choice": list(choices),
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return " ".join((json.load(r)["choices"][0]["message"]["content"] or "").split())


def our_unmatched_line(spec):
    """우리 층이 실제로 내보낸 문장을 **피드백 채널에서 읽어 온다** (프로브가 저작하지 않는다)."""
    mark = str(spec.get("unmatched_text") or "").split("{")[0].strip()
    if not mark:
        return None
    p = os.path.join(SIMS, FB + ".jsonl.gz")
    best = None
    for line in gzip.open(p, "rt", encoding="utf-8"):
        if not line.strip():
            continue
        t = str(json.loads(line).get("text") or "")
        i = t.find(mark)
        if i < 0:
            continue
        seg = t[i:].split("\n")[0].strip()
        if best is None or len(seg) > len(best):
            best = seg
    return best


def drop_named_sentences(text, names):
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    return " ".join(s for s in sents if not any(nm in s for nm in names))


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    a2 = load_domain_a2("banking_knowledge")
    spec = next(s for s in a2["ledger_metrics"] if s.get("eligible_text"))
    uspec = next((s for s in a2["ledger_metrics"] if s.get("unmatched_text")), spec)
    rows = a2["policy_ontology"]["rows"]
    axes = spec["eligible"]["show_axes"]
    maps = {ax: FD._a3_map(rows, {"axis": ax}) for ax in axes}
    name = lambda l: l.strip().split(":")[0].strip()          # noqa: E731

    line = our_unmatched_line(uspec)
    if not line:
        print("fb 채널에서 우리 unmatched 문장을 못 찾았다 — 중단(%s)" % FB)
        return 1
    print("model=%s · n=%d" % (MODEL, n))
    print("우리 실제 문구 (%d자):\n  %s\n" % (len(line), line))

    out = []
    for task, case in CASE.items():
        gold = X.GOLD[task]
        lines = LG.eligible_text(case["days"], {}, maps, spec,
                                 {"qualifying_deposit_usd": case["deposit"]}).strip().splitlines()
        head = [l for l in lines if not (l.startswith("  ") and ":" in l)]
        body = [l for l in lines if l.startswith("  ") and ":" in l]
        ALL = [name(l) for l in body]
        CHK = [s for s in ALL if s in (maps.get(ACCOUNT_AXIS) or {})]
        facts = drop_named_sentences(X.FACTS[task], ALL)      # x186 처방 반영: 피연산자만
        # 이름만 지운 판 — 나열 구간을 통째로 빼고 경고·숫자는 남긴다
        nn = re.sub(r"(?:%s)[^;,.]*" % "|".join(re.escape(s) for s in sorted(ALL, key=len, reverse=True)),
                    "a group", line)

        print("\n" + "=" * 100)
        print("%s  gold=%r · 사실(피연산자만)=%s" % (task, gold, facts))
        print("  이름 지운 판: %s" % nn[:150])
        print("=" * 100)
        print("  %-16s | %-17s | %-17s | %-17s | %s"
              % ("arm", "asc/all", "desc/all", "asc/chk", "desc/chk"))
        for alabel, pre in (("iso            ", ""),
                            ("+unmatched     ", line + "\n\n"),
                            ("+unmatched_nn  ", nn + "\n\n")):
            cells = []
            for choices in (ALL, CHK):
                for rev in (False, True):
                    order = sorted(body, key=name, reverse=rev)
                    tbl = "\n".join(head[:1] + order + head[1:]).strip()
                    prompt = pre + tbl + "\n\n" + facts + "\n\n" + X.QUESTION
                    c = collections.Counter()
                    for i in range(n):
                        try:
                            c[guided_full(prompt, choices, 0.0 if i == 0 else 0.7)] += 1
                        except Exception as e:
                            c["ERR %s" % type(e).__name__] += 1
                    cells.append("%d/%d %-12s" % (c.get(gold, 0), n, c.most_common(1)[0][0][:12]))
                    out.append({"task": task, "arm": alabel.strip(),
                                "choices": "all" if choices is ALL else "chk",
                                "sort": "desc" if rev else "asc",
                                "gold_hit": c.get(gold, 0), "n": n, "dist": dict(c)})
            print("  %-16s | %s | %s | %s | %s"
                  % (alabel, cells[0], cells[1], cells[2], cells[3]))

    json.dump(out, open(os.environ.get("T2_X189_OUT", "x189_out.json"), "w"), indent=1)
    print("\n  무너지면 [[25]] 조항 — 우리 출력이 유일한 근거원을 오염시킨다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
