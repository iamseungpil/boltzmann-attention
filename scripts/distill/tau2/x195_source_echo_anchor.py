# -*- coding: utf-8 -*-
r"""x195 — `[SOURCE]` 가 **에이전트 자신의 문장을 축자로 되읽어 주는 것**이 정박을 설치하는가 (유료 0).

## 왜

런 s 의 099 두 trial 은 **결정 블록을 바이트 동일하게** 받았다(턴 28·`It answers: World Blue`
+ 근거 + 순위 3개). 그런데 하나는 `World Blue` 를 내고 하나는 `Navy Blue` 를 냈다. 사이드카
전수 대조에서 **두 trial 이 다른 것은 사실상 하나**다:

    FAIL  turn 20 [SOURCE] you stated **4** thing(s) ... : "you have referred 9 Hunter Green
          Accounts"; "you have referred 3 Cobalt Blue Accounts"; "you have referred 2 Navy
          Blue Accounts"; "you have already earned 14 referrals"
    PASS  turn 26 [SOURCE] you stated **1** thing(s) ... : "you have already referred 9
          Hunter Green Accounts this year"

`unmatched_text` 는 양쪽 2회로 **동일**했고, 그 문구는 C366 에서 이미 **12셀 전부 무해**로
음성 판정됐다(*"정박은 지목이 만들지 언급이 만들지 않는다"*). 그러나 C366 이 시험한 것은
`unmatched_text` **하나뿐**이고, `[SOURCE]` 는 형태가 다르다 — 제3자 서술이 아니라
**에이전트 자신의 주장을 따옴표에 넣어 돌려준다**. C363/C366 의 경계선(지목 vs 언급)에서
자기 주장의 축자 인용이 어느 쪽인지는 **미측정**이다.

## 축

  iso              표 + 피연산자 사실 + 질문 (주입 없음·기준선)
  +unmatched       C366 의 음성 통제를 이 하네스에서 **재현**한다(재현 안 되면 나머지도 못 읽는다)
  +source_1        PASS trial 이 받은 실제 `[SOURCE]` 문장 (주장 1건)
  +source_4        FAIL trial 이 받은 실제 `[SOURCE]` 문장 (주장 4건·보유 계좌 3종을 이름으로)
  +source_4_nn     같은 문장에서 **후보 이름만 중립 라벨로** 치환 — 인용 *형식*이 레버인지
                   그 안의 *이름*이 레버인지 가른다

  sort name_asc·name_desc · choices all·chk · task 099·100

문구는 전부 **피드백 채널에서 읽어 온다**(프로브가 저작하지 않는다·x189 규율).
task_100 에는 그 문장이 원래 안 나가므로 이식이 곧 *"남의 문장도 설치하는가"* 통제가 된다.

## 읽는 법

  · `+source_4` 만 무너지고 `+source_1` 은 안 무너지면 → **인용 건수(=이름 수)가 레버**.
    처방: `[SOURCE]` 는 건수를 말하되 **후보 이름을 다시 인쇄하지 않는다**.
  · `+source_4_nn` 도 같이 무너지면 → 레버는 이름이 아니라 **자기 문장 되읽기라는 형식**.
    처방: 인용 대신 지시(무엇을 검색하라)만 남긴다.
  · 넷 다 안 무너지면 → 런 s 의 1/2 차이는 이 축이 아니다. `[SOURCE]` 는 면책되고,
    남은 후보는 SEARCH-EXHAUST 동반·발명 도구 3회다(둘 다 별도 프로브).
  · `+unmatched` 가 여기서 무너지면 → 하네스가 C366 과 다른 것을 재고 있다. **먼저 그걸 고친다.**

실행: python x195_source_echo_anchor.py [N]
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
import x189_our_own_anchor as X189                             # noqa: E402
import t2_factdag as FD                                        # noqa: E402
import t2_ledger as LG                                         # noqa: E402
from gate_interpreter import load_domain_a2                     # noqa: E402

URL = os.environ.get("T2_PROBE_URL", "http://localhost:8140/v1/chat/completions")
MODEL = os.environ.get("T2_PROBE_MODEL", "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8")
HERE = os.path.dirname(os.path.abspath(__file__))
SIMS = os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026", "sim_results")
FB_SRC = os.environ.get("T2_FB_SRC", "fb_bank_anchorslot_20260809s")
CASE = {"task_099": {"days": 730, "deposit": 30000},
        "task_100": {"days": 65, "deposit": 31000}}
ACCOUNT_AXIS = "referrer_tenure_days"


def source_lines():
    """실제 `[SOURCE]` 문장을 fb 채널에서 읽어 **주장 건수별로** 돌려준다(저작 0)."""
    p = os.path.join(SIMS, FB_SRC + ".jsonl.gz")
    found = {}
    for line in gzip.open(p, "rt", encoding="utf-8"):
        if not line.strip():
            continue
        for seg in str(json.loads(line).get("text") or "").split("\n"):
            if "[SOURCE]" not in seg:
                continue
            m = re.search(r"you stated (\d+) thing", seg)
            if not m:
                continue
            k = int(m.group(1))
            if k not in found or len(seg) > len(found[k]):
                found[k] = seg.strip()
    return found


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    a2 = load_domain_a2("banking_knowledge")
    spec = next(s for s in a2["ledger_metrics"] if s.get("eligible_text"))
    uspec = next((s for s in a2["ledger_metrics"] if s.get("unmatched_text")), spec)
    rows = a2["policy_ontology"]["rows"]
    axes = spec["eligible"]["show_axes"]
    maps = {ax: FD._a3_map(rows, {"axis": ax}) for ax in axes}
    name = lambda l: l.strip().split(":")[0].strip()          # noqa: E731

    srcs = source_lines()
    if not srcs:
        print("fb 채널에서 [SOURCE] 문장을 못 찾았다 — 중단(%s)" % FB_SRC)
        return 1
    one = srcs.get(min(srcs)) or ""
    many = srcs.get(max(srcs)) or ""
    if min(srcs) == max(srcs):
        print("⚠주장 건수가 한 종류뿐이다(%s) — 1건/4건 대조가 성립하지 않는다." % sorted(srcs))
    unm = X189.our_unmatched_line(uspec) or ""
    print("model=%s · n=%d · 건수별 [SOURCE] %s" % (MODEL, n, sorted(srcs)))
    print("  1건: %s" % one[:160])
    print("  다건: %s" % many[:160])

    out = []
    for task, case in CASE.items():
        gold = X.GOLD[task]
        lines = LG.eligible_text(case["days"], {}, maps, spec,
                                 {"qualifying_deposit_usd": case["deposit"]}).strip().splitlines()
        head = [l for l in lines if not (l.startswith("  ") and ":" in l)]
        body = [l for l in lines if l.startswith("  ") and ":" in l]
        ALL = [name(l) for l in body]
        CHK = [s for s in ALL if s in (maps.get(ACCOUNT_AXIS) or {})]
        facts = X189.drop_named_sentences(X.FACTS[task], ALL)
        # 이름만 중립 라벨로 — 인용 형식은 그대로 둔다
        nn = re.sub("|".join(re.escape(s) for s in sorted(ALL, key=len, reverse=True)),
                    "that account", many) if ALL else many

        print("\n" + "=" * 100)
        print("%s  gold=%r" % (task, gold))
        print("=" * 100)
        print("  %-16s | %-17s | %-17s | %-17s | %s"
              % ("arm", "asc/all", "desc/all", "asc/chk", "desc/chk"))
        for alabel, pre in (("iso            ", ""),
                            ("+unmatched     ", (unm + "\n\n") if unm else ""),
                            ("+source_1      ", one + "\n\n"),
                            ("+source_4      ", many + "\n\n"),
                            ("+source_4_nn   ", nn + "\n\n")):
            cells = []
            for choices in (ALL, CHK):
                for rev in (False, True):
                    order = sorted(body, key=name, reverse=rev)
                    tbl = "\n".join(head[:1] + order + head[1:]).strip()
                    prompt = pre + tbl + "\n\n" + facts + "\n\n" + X.QUESTION
                    c = collections.Counter()
                    for i in range(n):
                        try:
                            c[X189.guided_full(prompt, choices, 0.0 if i == 0 else 0.7)] += 1
                        except Exception as e:
                            c["ERR %s" % type(e).__name__] += 1
                    cells.append("%d/%d %-12s" % (c.get(gold, 0), n, c.most_common(1)[0][0][:12]))
                    out.append({"task": task, "arm": alabel.strip(),
                                "choices": "all" if choices is ALL else "chk",
                                "sort": "desc" if rev else "asc",
                                "gold_hit": c.get(gold, 0), "n": n, "dist": dict(c)})
            print("  %-16s | %s | %s | %s | %s"
                  % (alabel, cells[0], cells[1], cells[2], cells[3]))

    json.dump(out, open(os.environ.get("T2_X195_OUT", "x195_out.json"), "w"), indent=1)
    print("\n※ `+unmatched` 가 여기서 무너지면 C366 재현 실패다 — 그것부터 고친다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
