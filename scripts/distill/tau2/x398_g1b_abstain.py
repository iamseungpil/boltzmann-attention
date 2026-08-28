# -*- coding: utf-8 -*-
import io, json, collections, re, sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
S = "/home/woori/workspace_common/boltzmann-attention-pi/reports/facet_rft_2026/x395_compliance_iso_g1b_cls.json"
raw = json.load(io.open(S, encoding="utf-8"))
ab = [r for r in raw if r["cls"] == "BLANK_ABSTAIN"]
print("BLANK_ABSTAIN 총 %d 건" % len(ab))
PAT = [("DONE_ALREADY(이미 완료했다)", r"(?i)already|completed|완료|no further action|submitted successfully"),
       ("HANDED_OFF(인간 이관됨)", r"(?i)human agent|transferred|이관|인공"),
       ("NOTHING_NEEDED(불필요)", r"(?i)not (required|needed|necessary)|필요하지 않|no tool")]
cc = collections.Counter()
for r in ab:
    t = r["raw"]
    tags = [n for n, p in PAT if re.search(p, t)]
    cc[" + ".join(tags) if tags else "OTHER"] += 1
for k, v in cc.most_common():
    print("  %-60s %d" % (k, v))
print()
print("ABSTAIN 안에 표적 도구 이름이 축자로 등장: %d / %d"
      % (sum(1 for r in ab if r["tool"] in r["raw"]), len(ab)))
print("ABSTAIN 안에 물음표(되묻기): %d / %d" % (sum(1 for r in ab if "?" in r["raw"]), len(ab)))
print("ABSTAIN reason 길이 중앙: %d자" % sorted(len(r["raw"]) for r in ab)[len(ab) // 2])
print()
print("## 팔별 EMIT 안에서 '무엇을 냈나' (표적 아닌 것 상위)")
for arm in ["A_min", "B_full", "B_tail32", "B_tail16", "B_tail8", "B_tail4"]:
    rs = [r for r in raw if r["arm"] == arm and r["cls"] == "EMIT" and not r["cls_exact"]]
    c = collections.Counter(r["cnm"] for r in rs)
    print("%-9s 오답 EMIT %2d :: %s" % (arm, len(rs), ", ".join("%s=%d" % (k, v) for k, v in c.most_common(5))))
print()
print("## 전체 216행 확정 합계")
c = collections.Counter(r["cls"] for r in raw)
print(dict(c), "합", sum(c.values()))
