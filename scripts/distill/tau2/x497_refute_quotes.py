# -*- coding: utf-8 -*-
"""x497 — 원인-스텝 포렌식(Trace 8종)의 **인용 검산**. 반증의 첫 팔이자 기계적인 팔.

배경: 2026-08-23 의 워크플로 `wf_44a4979c-6c4` 는 Trace 8/8 을 냈지만 Refute 는 1/8 만 돌고
세션과 함께 끝났다(원본은 `CAUSE_STEP_FORENSIC_RAW_2026_08_23.json` 에 회수돼 있다).
[[31]] 규칙 6 = **반증 전 승격 금지** — 그래서 남은 7종을 잇는다.

이 프로브가 하는 것은 반증의 **기계로 되는 부분** 하나다: 각 `evidence_quote`·`anchor` 안의
축자 조각이 그 sim 의 실제 메시지 덤프(또는 로그·레포 파일)에 **정말 있는가**. 이것이 통과해야
그 다음(귀속 다툼·대안 설명)이 의미를 갖는다. 완료된 Refute 1종이 손으로 했던 바로 그 검산이고,
[[66]] 의 *"인용-근거(가리키기 + substring 검산)"* 와 같은 형태다.

판단은 하지 않는다 — 있으면 FOUND, 없으면 MISSING 으로 세고 어디를 봤는지 적을 뿐이다.
(엔진이 도메인 텍스트를 해석하지 않는다·[[59]])

    py -3 x497_refute_quotes.py               # 전체
    py -3 x497_refute_quotes.py 3             # trace 인덱스만
"""
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

import t2_forensic as F                                            # noqa: E402

RAW = os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026",
                   "CAUSE_STEP_FORENSIC_RAW_2026_08_23.json")
OUT = os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026",
                   "x497_refute_quotes.json")

MIN_FRAG = 14                       # 이보다 짧은 조각은 일반적이라 근거가 못 된다
ELIDE = re.compile(r"…|\.\.\.|\[\.\.\.\]")
WS = re.compile(r"\s+")


# 인용자가 **덧붙인** 표기와 활자 변형 — 조각과 건초더미 **양쪽에 똑같이** 적용한다.
#   ⚠한쪽에만 걸면 검산이 느슨해지는 게 아니라 **틀린다**. 여기서 지우는 것은 의미가 아니라
#     마크다운 강조(`**`)와 활자 따옴표·대시 같은 표기 변형뿐이다(2026-08-24: 미확인 175 중
#     다수가 인용문 안의 `**$2,000**` 처럼 분석자가 굵게 칠한 자리였다).
_EMPH = re.compile(r"\*+")
_TYPO = {"‘": "'", "’": "'", "“": '"', "”": '"',
         "–": "-", "—": "-", "‑": "-", " ": " ",
         "‑": "-", "＂": '"'}


def norm(s):
    """공백을 접고 표기 변형을 통일한다(대소문자·낱말은 그대로 — 축자 검산이므로)."""
    t = str(s)
    for a, b in _TYPO.items():
        t = t.replace(a, b)
    t = _EMPH.sub("", t)
    return WS.sub(" ", t).strip()


def unescape(s):
    return s.replace('\\"', '"').replace("\\'", "'").replace("\\n", " ").replace("\\\\", "\\")


def fragments(quote):
    """인용 안의 축자 조각을 뽑는다: "..." · “...” · `...` · '...'(로그 값)."""
    q = unescape(quote)
    out = []
    for pat in (r'"([^"]{6,})"', r'“([^”]{6,})”', r'`([^`]{6,})`', r"val='([^']{4,})'"):
        for m in re.finditer(pat, q):
            out.append(m.group(1))
    frags = []
    for piece in out:
        for part in ELIDE.split(piece):
            part = norm(part).strip(" .·:;,→-")
            if len(part) >= MIN_FRAG:
                frags.append(part)
    # 중복 제거(순서 유지)
    seen, uniq = set(), []
    for f in frags:
        if f not in seen:
            seen.add(f)
            uniq.append(f)
    return uniq


# ── sim 덤프 ────────────────────────────────────────────────────────────────
_DUMP = {}


def sim_dump(ref):
    """'tag|task_072#s626729' → 그 sim 의 전체 텍스트(메시지·도구호출·도구결과).

    ⚠파일 명명이 두 가지다(`.results.json.gz` / `_results.json.gz`) — 2026-08-19 사고와 같은
      함정이므로 둘 다 시도한다(`t2_forensic.all_result_files` 주석 참조).
    """
    if ref in _DUMP:
        return _DUMP[ref]
    tag, _, key = ref.partition("|")
    allsims, err0 = None, ""
    for suf in ("_results.json.gz", ".results.json.gz"):
        try:
            allsims = F.sims(tag, suf)
            if allsims:
                break
        except Exception as e:
            err0 = str(e)
    if not allsims:
        _DUMP[ref] = ("", "LOAD_FAIL: %s" % (err0 or "no sims"))
        return _DUMP[ref]
    hit = None
    for s in allsims:
        if F.simtag(s) == key or F.sim_key(s) == key:
            hit = s
            break
    if hit is None:
        _DUMP[ref] = ("", "SIM_NOT_FOUND (tag has %d sims)" % len(allsims))
        return _DUMP[ref]
    parts = []
    for m in (hit.get("messages") or []):
        parts.append(json.dumps(m, ensure_ascii=False))
    parts.append(json.dumps(hit.get("reward_info") or {}, ensure_ascii=False))
    _DUMP[ref] = (norm(unescape(" ".join(parts))), "")
    return _DUMP[ref]


_LOG = {}


def log_dump(tag):
    if tag not in _LOG:
        try:
            _LOG[tag] = norm(F.log_text(tag) or "")
        except Exception:
            _LOG[tag] = ""
    return _LOG[tag]


_REPO = {}
_PATHS = None


def _find(name):
    """파일 이름 → repo 상대 경로(첫 매치). 앵커가 경로 없이 이름만 대는 일이 많다."""
    global _PATHS
    if _PATHS is None:
        _PATHS = {}
        root = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
        for dp, _dn, fn in os.walk(root):
            if ".git" in dp:
                continue
            for f in fn:
                _PATHS.setdefault(f, os.path.relpath(os.path.join(dp, f), root).replace("\\", "/"))
    # 앵커가 이름을 줄여 대는 일이 흔하다(`specific.json` = `banking_knowledge.specific.json`).
    # 어느 쪽을 뜻했는지는 **고르지 않는다** — 후보 전부를 검색 대상에 넣는다(존재 검사이므로).
    cand = [p for f, p in _PATHS.items() if f == name or f.endswith("." + name) or f.endswith(name)]
    return sorted(set(cand), key=len)


def repo_text(name, sha=None):
    """앵커가 파일 이름을 대면 그 파일도 검색 대상에 넣는다.

    ⚠**런 sha 를 먼저 본다.** 워킹트리는 이미 고쳐져 있을 수 있어서, 워킹트리에서 못 찾는 것을
      *"날조"* 로 읽으면 반대로 틀린다(2026-08-24 실물: `"Light Blue Account": null` 은 런 sha 에는
      있고 워킹트리에는 없다 — 그 사이에 수리가 들어갔기 때문이다).
    """
    key = (name, sha)
    if key in _REPO:
        return _REPO[key]
    root = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
    chunks = []
    for rel in _find(name):
        if sha:
            import subprocess
            try:
                chunks.append(subprocess.run(["git", "show", "%s:%s" % (sha, rel)], cwd=root,
                                             capture_output=True).stdout.decode("utf-8", "replace"))
            except Exception:
                pass
        else:
            try:
                chunks.append(io.open(os.path.join(root, rel), encoding="utf-8").read())
            except Exception:
                pass
    _REPO[key] = norm(" ".join(chunks))
    return _REPO[key]


_SHA = {}


def sha_of(tag):
    """런 태그 → 그 런이 돈 엔진 sha(meta 파일 축자). 없으면 None."""
    base = tag.split("_")[1] if "_" in tag else tag
    if base in _SHA:
        return _SHA[base]
    p = os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026", "sim_results",
                     "bank_%s.meta.json" % base)
    sha = None
    try:
        sha = json.load(io.open(p, encoding="utf-8")).get("sha")
    except Exception:
        for q in (os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026",
                               "sim_results", tag + ".meta.json"),):
            try:
                sha = json.load(io.open(q, encoding="utf-8")).get("sha")
            except Exception:
                pass
    _SHA[base] = sha
    return sha


FILE_RE = re.compile(r"([A-Za-z0-9_]+\.(?:py|json|md|sh))")


PATH_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)+$")
FMT_RE = re.compile(r"%[sdfr]|%\.\d")


TAG_RE = re.compile(r"^(bank_[A-Za-z0-9_]+|bx_[A-Za-z0-9_]+|[A-Za-z0-9_]+\|task_\d+#s\d+)$")
KO_RE = re.compile(r"[가-힣]")


def classify(frag):
    """조각이 **축자 인용**인지 **가리키기**인지 가른다 — 둘을 같은 잣대로 재면 반증이 틀린다.

    path   = `a.b.c` 꼴 선언 경로(=포인터). 문자열로 파일에 있을 리가 없다.
    format = `%d` 가 든 코드 템플릿(런타임 문자열과 소스가 다르다).
    quote  = 나머지 — 이것만 substring 으로 판정한다.
    """
    if PATH_RE.match(frag) or TAG_RE.match(frag):
        return "path"
    if FMT_RE.search(frag):
        return "format"
    if KO_RE.search(frag):
        return "prose"          # 인용 구분자 안에 섞여 든 **분석자 자신의 한국어 서술**
    return "quote"


def check(frags, haystacks):
    """축자 조각만 판정한다. path/format 은 따로 세어 '미확인' 에 섞지 않는다."""
    found, missing, skipped = [], [], []
    for f in frags:
        kind = classify(f)
        if kind != "quote":
            skipped.append({"frag": f, "kind": kind})
            continue
        if any(f in h for h in haystacks if h):
            found.append(f)
        else:
            missing.append(f)
    return found, missing, skipped


def main():
    raw = json.load(io.open(RAW, encoding="utf-8"))
    only = int(sys.argv[1]) if len(sys.argv) > 1 else None
    report = {"source": os.path.basename(RAW), "run": raw.get("run"), "traces": []}
    tot_f = tot_m = tot_s = 0

    for i, tr in enumerate(raw["traces"]):
        if only is not None and i != only:
            continue
        print("=" * 78)
        print("TRACE %d  %s" % (i, tr["tool"]))
        trep = {"index": i, "tool": tr["tool"], "modes": [], "claims": [], "sim_errors": []}
        for fm in tr["failure_modes"]:
            ref = fm.get("sim") or ""
            dump, err = sim_dump(ref)
            tag = ref.partition("|")[0]
            hay = [dump, log_dump(tag)]
            frags = fragments(fm.get("evidence_quote") or "")
            found, missing, sk = check(frags, hay)
            tot_f += len(found)
            tot_m += len(missing)
            tot_s += len(sk)
            row = {"sim": ref, "aid": fm.get("aid"), "mode": fm.get("mode"),
                   "attribution": fm.get("attribution"), "pointers": sk,
                   "n_frag": len(frags), "found": len(found), "missing": missing}
            if err:
                row["sim_error"] = err
                trep["sim_errors"].append({"sim": ref, "error": err})
            trep["modes"].append(row)
            mark = "ok  " if not missing and not err else "MISS"
            print("  %s %-52s %-9s %2d/%2d %s"
                  % (mark, ref.split("|")[-1] + " " + str(fm.get("aid")), fm.get("mode"),
                     len(found), len(frags), err))
            for m in missing:
                print("        ✗ %s" % (m[:110]))
        # 이 trace 가 인용한 런들의 엔진 sha — 앵커의 코드/선언은 **그때 것**과 대조해야 한다
        shas = sorted({s for s in (sha_of(r.partition("|")[0]) for r in tr["sims_read"]) if s})
        # ★claim 의 근거는 이 trace 가 읽은 **어느 sim 에나** 있을 수 있다(앵커가 sim 을 다시
        #   대지 않는 일이 흔하다). 그래서 haystack 은 trace 의 증거 기반 전체다 — 좁게 잡으면
        #   있는 것을 '없다'고 판정하게 된다(2026-08-24 실물: `[coverage] 1 of 13 …` 은 실재했다).
        trace_hay = []
        for r in tr["sims_read"]:
            d, _e = sim_dump(r)
            trace_hay.append(d)
        for tg in sorted({r.partition("|")[0] for r in tr["sims_read"]}):
            trace_hay.append(log_dump(tg))
        for c in tr["our_layer_claims"]:
            anchor = c.get("anchor") or ""
            frags = fragments(anchor)
            files = set(FILE_RE.findall(anchor))
            hay, at_head = [], []
            for f in files:
                for sha in shas:
                    hay.append(repo_text(f, sha))
                h = repo_text(f)
                at_head.append(h)
                hay.append(h)
            for ref in re.findall(r"(bank_[A-Za-z0-9_]+)\|(task_\d+#s\d+)", anchor):
                d, _e = sim_dump("%s|%s" % ref)
                hay.append(d)
                hay.append(log_dump(ref[0]))
            for tg in set(re.findall(r"(bank_[A-Za-z0-9_]+)", anchor)):
                hay.append(log_dump(tg))
            hay.extend(trace_hay)
            found, missing, sk = check(frags, hay)
            # 런 sha 에는 있는데 **지금 워킹트리에는 없는** 조각 = 그 사이에 바뀐 자리
            changed = [f for f in found if not any(f in h for h in at_head)] if files else []
            tot_f += len(found)
            tot_m += len(missing)
            tot_s += len(sk)
            trep["claims"].append({"claim": (c.get("claim") or "")[:200],
                                   "files": sorted(files), "shas": shas, "n_frag": len(frags),
                                   "found": len(found), "missing": missing, "pointers": sk,
                                   "gone_at_head": changed})
            print("  %s CLAIM %s" % ("ok  " if not missing else "MISS",
                                     norm(c.get("claim"))[:90]))
            for m in missing:
                print("        ✗ %s" % (m[:110]))
        report["traces"].append(trep)

    report["totals"] = {"found": tot_f, "missing": tot_m, "pointers": tot_s}
    io.open(OUT, "w", encoding="utf-8").write(
        json.dumps(report, ensure_ascii=False, indent=1) + "\n")
    print("\n조각 %d 중 확인 %d · 미확인 %d  →  %s"
          % (tot_f + tot_m, tot_f, tot_m, os.path.basename(OUT)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
