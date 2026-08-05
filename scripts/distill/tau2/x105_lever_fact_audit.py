# -*- coding: utf-8 -*-
"""Is what our levers assert true of the authority that decides it?

Two defects ended the 2026-08-06 arc, and neither was about scheduling levers against each
other — each was a lever stating something false and no instrument asking whether it was:

  task_022   we told the model `submit_cash_back_dispute_0589` was "missing its numeric suffix"
             and that "that exact name does not exist". The name is complete and the tool is
             real; it is user-side. Ninety minutes and seventy-six turns went into that loop.
  task_035   `verify_gather_prefix` was declared, mirrored into three files, quoted in a note —
             and read by exactly one line that wrapped the string in `set()`, making a set of
             characters. Half a session attributed a lost pass to the lever it silently disabled.

`x99` audits where a lever's *content* came from and `x95` audits whether two of our sentences
give opposite orders. Neither asks the flatter question: **does this assertion agree with the
registry, the schema, and the corpus?** That is this file. Four passes, all mechanical:

  A  이름   every tool-shaped name we speak or declare, against the env registry — and which
            side it is on, because "does not exist" and "wrong channel" are different sentences
  B  인자   every (tool → argument) declaration, against the environment's own schema
  C  死설정  every declared key, against the engine that is supposed to read it
  D  인용   every `_quote` / verbatim, against the policy and knowledge-base corpus
            (needs the corpus: run where T2_KB_DOCS_DIR resolves, otherwise reported unverified)

What it does **not** check: whether a rule is a good idea, whether a claim about the *conversation*
is true (that is runtime, not static), or anything outside the axes above. Silence here is not
proof of correctness — it is proof that these four questions were asked.

  usage: x105_lever_fact_audit.py [domain]        default banking_knowledge
"""

import collections
import glob
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
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

DOMAIN = sys.argv[1] if len(sys.argv) > 1 else "banking_knowledge"
A2DIR = os.path.join(HERE, "a2")
ENGINE = sorted(glob.glob(os.path.join(HERE, "t2_*.py")) + glob.glob(os.path.join(HERE, "gate_interpreter.py")))

# 이름처럼 생긴 토큰: discoverable 접미사형 또는 레지스트리에 실재하는 이름. 순수 패턴은 쓰지 않는다
# ([[22]]·C279: 패턴 규칙은 조용한 오탐) — 패턴은 **후보 수집**에만 쓰고 판정은 레지스트리가 한다.
CAND = re.compile(r"(?<![A-Za-z0-9_])([a-z][a-z0-9]*(?:_[a-z0-9]+){1,6})(?![A-Za-z0-9_])")

# ★후보 좁힘(1차 실행에서 배운 것): 이 도메인의 discoverable 도구는 **4자리** 접미사다(`_0589`).
#   문서 id(`doc_003`)·태스크 id(`task_019`)·카드 id(`bronze_001`)는 3자리라 그것만으로도 갈리지만,
#   접두어가 명백한 것은 이름으로 세지 않는다 — 감사자가 오탐에 파묻히면 축이 죽는다([[55]]).
NOT_TOOL = re.compile(r"^(doc|task|probe|txn|cc|acc|usr)_")
TOOLISH = re.compile(r"_\d{4}$")
SELFTEST = re.compile(r'^\s*if __name__ == ["\']__main__["\']', re.M)


def in_selftest(src, idx):
    """이 위치가 파일의 self-test 블록(`if __name__ == "__main__"`) 안인가."""
    m = SELFTEST.search(src)
    return bool(m) and idx > m.start()


def load_env(domain):
    p = os.path.join(A2DIR, "env_surface.json")
    d = (json.load(io.open(p, encoding="utf-8")) or {}).get(domain) or {}
    tools = d.get("tools") or {}
    agent = {n for n, v in tools.items() if (v or {}).get("side") != "user_tools"}
    user = set(d.get("discoverable_user_tools") or [])
    for n, v in tools.items():
        if (v or {}).get("side") == "user_tools":
            user.add(n)
    return tools, agent, user


def a2_files(domain):
    out = []
    for pat in ("%s.settings.json", "%s.specific.json", "%s.gate.json"):
        p = os.path.join(A2DIR, pat % domain)
        if os.path.exists(p):
            out.append(p)
    p = os.path.join(A2DIR, "base", "shared.json")
    if os.path.exists(p):
        out.append(p)
    return out


def walk(node, path=""):
    """(경로, 키, 값) 전수 — 死설정 판정과 문자열 수집에 함께 쓴다."""
    if isinstance(node, dict):
        for k, v in node.items():
            yield path, k, v
            for r in walk(v, path + "/" + str(k)):
                yield r
    elif isinstance(node, list):
        for i, v in enumerate(node):
            for r in walk(v, path + "[%d]" % i):
                yield r


def strings_of(node):
    if isinstance(node, str):
        yield node
    elif isinstance(node, dict):
        for v in node.values():
            for s in strings_of(v):
                yield s
    elif isinstance(node, list):
        for v in node:
            for s in strings_of(v):
                yield s


def main():
    tools, AGENT, USER = load_env(DOMAIN)
    REG = AGENT | USER
    files = a2_files(DOMAIN)
    docs = {f: json.load(io.open(f, encoding="utf-8")) for f in files}
    engine_src = {f: io.open(f, encoding="utf-8", errors="replace").read() for f in ENGINE}
    all_engine = "\n".join(engine_src.values())

    print("== 감사 대상 ==")
    print("  도메인 %s · A2 파일 %d · 엔진 파일 %d" % (DOMAIN, len(files), len(ENGINE)))
    print("  레지스트리: agent %d · user %d (env_surface)" % (len(AGENT), len(USER)))
    print("  ⚠ 이 감사는 **네 축만** 본다 — 규칙의 타당성·런타임 주장·축 밖 오류는 보지 않는다.\n")

    # ── A. 우리가 말하는 이름이 실재하는가 ─────────────────────────────────
    print("== A. 이름 — 우리가 말하거나 선언한 도구명이 레지스트리에 있는가 ==")
    said = collections.defaultdict(set)          # name -> {출처}

    def keep(n):
        return (n in REG or TOOLISH.search(n)) and not NOT_TOOL.match(n)

    for f, doc in docs.items():
        base = os.path.basename(f)
        for s in strings_of(doc):
            for m in CAND.findall(s):
                if keep(m):
                    said[m].add(base)
    for f, src in engine_src.items():
        base = os.path.basename(f)
        for mo in re.finditer(r"[\"']([a-z][a-z0-9]*(?:_[a-z0-9]+){1,6})[\"']", src):
            m = mo.group(1)
            if keep(m):
                said[m].add(base + ("(self-test)" if in_selftest(src, mo.start()) else ""))
    ghost = {n: v for n, v in said.items() if n not in REG}
    print("  말한 이름 %d개 중 레지스트리 밖 = %d개" % (len(said), len(ghost)))
    for n, where in sorted(ghost.items()):
        live = [w for w in where if "self-test" not in w]
        mark = "⚠" if live else "·"      # self-test 전용이면 모델에게 가지 않는다
        print("    %s %-46s ← %s" % (mark, n, ",".join(sorted(where))))
    if not ghost:
        print("    (없음)")
    print("  · user-측으로 분류되는 이름 %d개: %s" % (len(USER), ", ".join(sorted(USER))))
    print("    ⇒ 이 이름들에 대해 '존재하지 않는다'고 말하는 문구가 있으면 그것이 022 클래스다.")

    # ── B. (도구 → 인자) 선언이 환경 스키마와 맞는가 ────────────────────────
    print("\n== B. 인자 — 선언한 (도구 → 인자)가 env 스키마에 있는가 ==")
    bad, checked = [], 0
    for f, doc in docs.items():
        for path, k, v in walk(doc):
            if not isinstance(v, dict):
                continue
            if k not in ("tools", "name_args"):
                continue
            for tool, arg in v.items():
                if not isinstance(arg, str) or tool not in tools:
                    continue
                checked += 1
                if arg not in (tools[tool].get("args") or []):
                    bad.append((os.path.basename(f), path + "/" + k, tool, arg,
                                tools[tool].get("args")))
    print("  대조 %d건 · 불일치 %d건" % (checked, len(bad)))
    for b in bad:
        print("    ⚠ %s %s: %s(%s) — env args=%s" % b)
    if not bad:
        print("    (없음)")

    # ── C. 선언한 키를 읽는 소비자가 있는가 (035 클래스) ────────────────────
    print("\n== C. 死설정 — 선언된 키를 엔진이 읽는가 ==")
    SKIP = re.compile(r"^_note|^_meta$|^_")

    def consumed(k):
        return (('"%s"' % k) in all_engine or ("'%s'" % k) in all_engine
                or ("get(%r)" % k) in all_engine)

    # ★스키마 필드 vs 데이터 맵 — 형제로 가른다(1차 실행의 오탐 78건이 전부 데이터였다).
    #   엔진이 **이름으로 읽는 형제가 하나라도 있는 dict** = 필드 구조체다. 거기서 아무도 안 읽는
    #   키는 죽은 필드다(`verify_gather_prefix`가 정확히 그 모양이었다: 형제 satisfiers·kind·
    #   predicate는 읽히고 자기만 안 읽혔다). 형제가 **아무도** 안 읽히면 그 dict는 순회되는
    #   데이터 맵이고(가맹점→카테고리 등), 그 키들은 죽은 게 아니다.
    dead, structs = [], 0
    for f, doc in docs.items():
        base = os.path.basename(f)
        for path, k, v in walk(doc):
            if not isinstance(v, dict):
                continue
            fields = [x for x in v if isinstance(x, str) and not SKIP.match(x)]
            if not fields:
                continue
            live = [x for x in fields if consumed(x)]
            if not live:
                continue                       # 데이터 맵 — 이 축의 대상이 아니다
            structs += 1
            for x in fields:
                if not consumed(x) and x not in REG:
                    dead.append((x, base, (path + "/" + str(k)).lstrip("/"), sorted(live)[:3]))
    seen = set()
    uniq = [d for d in dead if not (d[0] in seen or seen.add(d[0]))]

    # ★그중 **모델에게 갈 문구**만 따로 세운다 — 이 축의 진짜 표적이다. 죽은 데이터 필드는 낭비지만,
    #   **죽은 문구**는 "우리가 그 말을 한다"고 믿게 만든다(x95의 정적 목록도 그만큼 부풀어 있다).
    #   `verify_gather_prefix`와 같은 계열이고, 1차 실행이 여기서 `feedback_resolved`를 잡았다.
    def is_msg(v):
        if not isinstance(v, str) or len(v) < 40:
            return False
        return v.lstrip().startswith("Error:") or re.match(r"\s*\[[A-Z][A-Z0-9_\- ]+\]", v) \
            or "you " in v.lower()

    dead_msgs = []
    for f, doc in docs.items():
        base = os.path.basename(f)
        for path, k, v in walk(doc):
            if isinstance(k, str) and is_msg(v) and not consumed(k) and not SKIP.match(k):
                dead_msgs.append((k, base, path.lstrip("/"), str(v)[:90].replace("\n", " ")))
    seen2 = set()
    dm = [d for d in dead_msgs if not ((d[0], d[2]) in seen2 or seen2.add((d[0], d[2])))]

    print("  필드 구조체 %d개 검사 · 아무도 이름으로 읽지 않는 필드 %d종" % (structs, len(uniq)))
    print("  ★그중 **모델에게 갈 문구인데 도달 경로가 없는 것** = %d건:" % len(dm))
    for k, base, path, head in dm:
        print("    ⚠ %s/%s (%s)" % (path or "root", k, base))
        print("        %r" % head)
    if not dm:
        print("    (없음)")
    print("  ⚠ 이 축의 한계(오탐): 키 이름이 아니라 **순회**로 소비되는 구조(절차 노드·전송 사유 표)는")
    print("     죽지 않았는데도 위 %d종에 섞인다. 판정은 소비처 grep으로 확정할 것 —" % len(uniq))
    print("     1차 실행에서 `pay_credit_card_from_checking`이 그 오탐이었다(`_fam()`이 접미사 정규화).")

    # ── D. 인용이 코퍼스에 실재하는가 ───────────────────────────────────────
    print("\n== D. 인용 — 선언된 축자(`_quote`/verbatim)가 코퍼스에 있는가 ==")
    corpus_dir = os.environ.get("T2_KB_DOCS_DIR")
    corpus = ""
    if corpus_dir and os.path.isdir(corpus_dir):
        for p in glob.glob(os.path.join(corpus_dir, "**", "*"), recursive=True):
            if os.path.isfile(p):
                try:
                    corpus += io.open(p, encoding="utf-8", errors="replace").read() + "\n"
                except Exception:
                    pass
    quotes = []
    for f, doc in docs.items():
        for path, k, v in walk(doc):
            if isinstance(v, str) and k in ("_quote", "verbatim") and v.strip():
                quotes.append((os.path.basename(f), path + "/" + k, v.strip()))
    if not corpus:
        print("  코퍼스 미해결(T2_KB_DOCS_DIR) — 인용 %d건 **미검증**." % len(quotes))
        print("  ⇒ 리모트에서 이 스크립트를 다시 돌려야 이 축이 닫힌다(무료).")
    else:
        norm = re.sub(r"\s+", " ", corpus)
        miss = [(f, p, q) for f, p, q in quotes if re.sub(r"\s+", " ", q) not in norm]
        print("  인용 %d건 · 코퍼스에서 못 찾음 %d건" % (len(quotes), len(miss)))
        for f, p, q in miss:
            print("    ⚠ %s %s: %r" % (f, p, q[:110]))
        if not miss:
            print("    (전건 실재)")


if __name__ == "__main__":
    main()
