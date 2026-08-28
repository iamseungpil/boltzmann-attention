# -*- coding: utf-8 -*-
"""x397 G1-b — x395 raw 전량 4분류 재집계 (기존 로그만·추가 런 0)."""
import io, json, os, re, sys, collections
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import x395_compliance_iso as X

SRC = "/home/woori/workspace_common/boltzmann-attention-pi/reports/facet_rft_2026/x395_compliance_iso.json"
DOCS = X.load_docs(); UNIV = set(X.tool_universe(DOCS))

RULES = r"""
## 분류 규칙 (결정론·LLM 0 · 이 규칙으로만 셌다)
전처리: raw 문자열 그대로. ```fence 는 정규식 ^```(json)?|```$ 로 제거. JSON 후보 = 첫 { .. 마지막 }.
우선순위: EMIT > BLANK_ABSTAIN > MALFORMED > NAME_ONLY > BLANK_OTHER (위에서 걸리면 아래는 안 본다)

EMIT           strict json.loads 성공 AND o["tool"] 가 비지 않은 문자열(dict 면 .name)
               - EMIT_INUNIV   그 이름이 문서-정의 도구 목록 안
               - EMIT_INVENTED 그 이름이 목록 밖(날조)
BLANK_ABSTAIN  strict json.loads 성공 AND tool 키가 null 이거나 없음 = "도구 불필요" 명시 선언
               - ABSTAIN_NAMED   reason 산문 안에 목록 도구 이름이 축자로 있음
               - ABSTAIN_UNNAMED 없음
MALFORMED      텍스트에 { 와 "tool" 이 있는데 strict 파싱 실패
               - MALF_RECOVER  정규식 "tool"\s*:\s*"([^"]+)" 로 이름 복구됨(구 parse_tool 이 적중 처리하던 자리)
               - MALF_DEAD     복구 불가
NAME_ONLY      위 어디도 아닌데 목록 도구 이름이 텍스트에 축자로 등장(= JSON 아닌 산문에서 이름을 댐)
BLANK_OTHER    도구 이름이 아예 없음
               - ERROR(호출 실패) / EMPTY / ASK_BACK(물음표 포함) / PROSE
※ 절단 경고: x395 는 raw 를 [:400] 으로 잘라 저장한다 -> 길이 400 인 행은 MALFORMED 가 절단 아티팩트일 수 있다.
"""

FENCE = re.compile(r"^```(?:json)?|```$", re.M)
REGEX_TOOL = re.compile(r'"tool"\s*:\s*"([^"]+)"')


def classify(txt):
    t = FENCE.sub("", (txt or "").strip()).strip()
    if t.startswith("ERROR "):
        return "BLANK_OTHER", "ERROR", None
    if not t:
        return "BLANK_OTHER", "EMPTY", None
    i, j = t.find("{"), t.rfind("}")
    obj = None
    if i >= 0 and j > i:
        try:
            obj = json.loads(t[i:j + 1])
        except Exception:
            obj = None
    if isinstance(obj, dict):
        nm = obj.get("tool")
        if isinstance(nm, dict):
            nm = nm.get("name")
        if isinstance(nm, str) and nm.strip():
            nm = nm.strip()
            return "EMIT", ("EMIT_INUNIV" if nm in UNIV else "EMIT_INVENTED"), nm
        named = any(u in t for u in UNIV)
        return "BLANK_ABSTAIN", ("ABSTAIN_NAMED" if named else "ABSTAIN_UNNAMED"), None
    if "{" in t and '"tool"' in t:
        m = REGEX_TOOL.search(t)
        return "MALFORMED", ("MALF_RECOVER" if m else "MALF_DEAD"), (m.group(1) if m else None)
    hits = [u for u in UNIV if u in t]
    if hits:
        hits.sort(key=len, reverse=True)
        return "NAME_ONLY", "NAME_ONLY", hits[0]
    if "?" in t or "？" in t:
        return "BLANK_OTHER", "ASK_BACK", None
    return "BLANK_OTHER", "PROSE", None


def main():
    raw = json.load(io.open(SRC, encoding="utf-8"))
    print(RULES)
    print("원자료 %s . 행 %d" % (SRC, len(raw)))
    print("팔: %s" % dict(collections.Counter(r["arm"] for r in raw)))
    print("모드: %s" % dict(collections.Counter(r["mode"] for r in raw)))
    print("문서-정의 도구 목록 크기: %d" % len(UNIV))
    print("raw 길이>=400 (절단 의심) 행: %d" % sum(1 for r in raw if len(r["raw"]) >= 400))
    for r in raw:
        c, s, nm = classify(r["raw"])
        r["cls"], r["sub"], r["cnm"] = c, s, nm
        r["cls_exact"] = bool(nm) and nm == r["tool"]
    ARMS = ["A_min", "B_full", "B_tail32", "B_tail16", "B_tail8", "B_tail4"]
    CLS = ["EMIT", "NAME_ONLY", "MALFORMED", "BLANK_ABSTAIN", "BLANK_OTHER"]
    print("\n## 교차표  팔 x 분류  (mode=next . 각 칸 = 건수(비율))")
    print("%-9s %s   %s" % ("arm", " ".join("%-14s" % c for c in CLS), "n"))
    for arm in ARMS:
        rs = [r for r in raw if r["arm"] == arm]
        if not rs:
            continue
        cc = collections.Counter(r["cls"] for r in rs)
        print("%-9s %s   %d" % (arm, " ".join("%-14s" % ("%d (%.0f%%)" % (cc[c], 100.0 * cc[c] / len(rs))) for c in CLS), len(rs)))
    print("\n## 세부 하위분류")
    for arm in ARMS:
        rs = [r for r in raw if r["arm"] == arm]
        if not rs:
            continue
        cc = collections.Counter(r["sub"] for r in rs)
        print("%-9s %s" % (arm, ", ".join("%s=%d" % (k, v) for k, v in cc.most_common())))
    print("\n## 구 채점 대비 (hit_exact / said_only 는 원본 필드)")
    print("%-9s %10s %10s %12s %12s %8s" % ("arm", "hit_exact", "said_only", "EMIT&정확", "cls_exact", "차이"))
    for arm in ARMS:
        rs = [r for r in raw if r["arm"] == arm]
        if not rs:
            continue
        he = sum(r["hit_exact"] for r in rs)
        so = sum(r["said_only"] for r in rs)
        ce = sum(r["cls_exact"] for r in rs)
        ee = sum(1 for r in rs if r["cls"] == "EMIT" and r["cnm"] == r["tool"])
        print("%-9s %10s %10s %12s %12s %8s" % (arm, "%d/%d" % (he, len(rs)), "%d/%d" % (so, len(rs)),
                                                "%d/%d" % (ee, len(rs)), "%d/%d" % (ce, len(rs)), "%+d" % (ce - he)))
    print("\n## said_only 의 정체 (구 필드 said_only=True 인 행만 새 분류로 분해)")
    for arm in ARMS:
        rs = [r for r in raw if r["arm"] == arm and r["said_only"]]
        if not rs:
            continue
        print("%-9s n=%-3d %s" % (arm, len(rs), ", ".join("%s=%d" % (k, v) for k, v in collections.Counter(r["sub"] for r in rs).most_common())))
    print("\n## B_tail4 전량 분해 (표적별 . * = 표적 도구 적중)")
    tgt = collections.defaultdict(list)
    for r in raw:
        if r["arm"] == "B_tail4":
            tgt[(r["task"], r["tool"])].append(r)
    for k, rs in sorted(tgt.items()):
        print("  %-9s %-40s %s" % (k[0], k[1][:40],
              " | ".join("k%d %s:%s%s" % (r["k"], r["sub"], (r["cnm"] or "-")[:32], "*" if r["cls_exact"] else "")
                         for r in sorted(rs, key=lambda x: x["k"]))))
    print("\n## A_min 대조 (같은 표적 . BLANK_ABSTAIN 이 있나)")
    tgt2 = collections.defaultdict(list)
    for r in raw:
        if r["arm"] == "A_min":
            tgt2[(r["task"], r["tool"])].append(r)
    for k, rs in sorted(tgt2.items()):
        print("  %-9s %-40s %s" % (k[0], k[1][:40],
              " | ".join("k%d %s:%s%s" % (r["k"], r["sub"], (r["cnm"] or "-")[:32], "*" if r["cls_exact"] else "")
                         for r in sorted(rs, key=lambda x: x["k"]))))
    print("\n## BLANK_ABSTAIN reason 축자 표본 (팔별 최대 2건)")
    for arm in ARMS:
        rs = [r for r in raw if r["arm"] == arm and r["cls"] == "BLANK_ABSTAIN"]
        for r in rs[:2]:
            print("  [%s] %s / %s :: %s" % (arm, r["task"], r["tool"][:34], r["raw"][:240].replace("\n", " ")))
    print("\n## NAME_ONLY / MALFORMED 축자 전량")
    n = 0
    for r in raw:
        if r["cls"] in ("NAME_ONLY", "MALFORMED"):
            n += 1
            print("  [%s] %s / %s / %s :: %s" % (r["arm"], r["task"], r["tool"][:30], r["sub"], r["raw"][:300].replace("\n", " ")))
    if n == 0:
        print("  (0 건)")
    print("\n## ABSTAIN 이 나온 표적 x 팔 (용량-반응)")
    tools = sorted(set((r["task"], r["tool"]) for r in raw))
    print("%-9s %-40s %s" % ("task", "빠뜨린 도구", " ".join("%-9s" % a for a in ARMS)))
    for t in tools:
        cells = []
        for arm in ARMS:
            rs = [r for r in raw if r["arm"] == arm and r["task"] == t[0] and r["tool"] == t[1]]
            ab = sum(1 for r in rs if r["cls"] == "BLANK_ABSTAIN")
            ex = sum(1 for r in rs if r["cls_exact"])
            cells.append("%-9s" % ("%d*/%dab" % (ex, ab)))
        print("%-9s %-40s %s" % (t[0], t[1][:40], " ".join(cells)))
    out = SRC.replace(".json", "_g1b_cls.json")
    io.open(out, "w", encoding="utf-8").write(json.dumps(raw, ensure_ascii=False, indent=1))
    print("\n분류 부착 원자료: %s" % out)


main()
