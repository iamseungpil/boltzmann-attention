# -*- coding: utf-8 -*-
"""One task, everything needed to name its cause and then name the lever.

The batch forensics say where failures stand; they cannot say *why* a particular turn went
the way it did, and a table row like "부재 판정 실패" is a label, not a cause. This prints,
for one task, the six things an honest attribution needs:

  §0 gold        what the task actually asks for — actions with their arguments, the
                 information the agent had to communicate, and the scenario the customer
                 was given (the customer's claim is an outside claim, [[21]]).
  §1 채점        per trial, which gold action first missed, and on what basis it was scored.
  §2 궤적        every turn: the model's prose, its calls with arguments, what came back,
                 and **who said it** — env or us (source-based, never by wording: x106 got
                 that wrong in both directions when it guessed from the text).
  §3 우리 층     the sidecar for this simulation — the reminders that never reach the
                 trajectory. Without it "our layer said nothing" is not an available
                 conclusion ([[55]]).
  §4 KB 대조     for every search the run made, whether the corpus actually holds the
                 words. This is the difference between "the model failed to find it" and
                 "it is not there" — 012 turns on exactly that distinction.
  §5 레버        which levers fired in this simulation, so the write-up can say whether an
                 existing lever missed, spoke and was ignored, or does not exist yet.

  usage:  x109_task_dossier.py task_012[,task_014] [--tag 20260806] [--wide] [--no-trace]
          (repo 디렉터리에서 실행할 것 — A2·엔진 소스가 없으면 출처 판정이 조용히 빈다)
"""

import collections
import glob
import gzip
import hashlib
import io
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from x106_n97live_batch_forensic import our_templates, eff          # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
TAU2 = os.environ.get("GO_TAU2", "/home/woori/scratch/tau2-bench")
DOMAIN = os.path.join(TAU2, "data", "tau2", "domains", "banking_knowledge")
SIMDIRS = [os.path.join(TAU2, "data", "simulations"),
           os.path.join(REPO, "reports", "facet_rft_2026", "sim_results")]
SIDECARS = ["/home/woori/scratch/logs/fb_*.jsonl",
            os.path.join(REPO, "reports", "facet_rft_2026", "sim_results", "fb_*.jsonl.gz")]
TAGRE = re.compile(r"\[([A-Z][A-Z0-9_\- ]{2,60})\]")
SIDECAR_FILES = sorted(os.path.basename(p) for pat in SIDECARS for p in glob.glob(pat))
STOP = set("the a an and or of to for my me i in on is are was were with please can you it "
           "that this what how do does did my have has need want would like get".split())

ARGS = [a for a in sys.argv[1:] if not a.startswith("-")]
TAG = "20260806"
if "--tag" in sys.argv:
    TAG = sys.argv[sys.argv.index("--tag") + 1]
WIDE = "--wide" in sys.argv
NOTRACE = "--no-trace" in sys.argv
CALLS = "--calls" in sys.argv
CUT = 1200 if WIDE else 420
TASKS = [t if t.startswith("task_") else "task_" + t for t in (ARGS[0].split(",") if ARGS else [])]


def jopen(path):
    op = gzip.open if path.endswith(".gz") else io.open
    with op(path, "rt", encoding="utf-8", errors="replace") as fh:
        return json.load(fh)


def load_sims():
    out = []
    seen = set()
    for base in SIMDIRS:
        pats = [os.path.join(base, "bank_n97_gpu*" + TAG + "*", "results.json"),
                os.path.join(base, "bank_n97_gpu*" + TAG + "*.results.json.gz")]
        for pat in pats:
            for p in sorted(glob.glob(pat)):
                src = os.path.basename(p).replace(".results.json.gz", "")
                if src == "results.json":
                    src = os.path.basename(os.path.dirname(p))
                if src in seen:
                    continue
                seen.add(src)
                try:
                    d = jopen(p)
                except Exception as e:
                    print("  (읽기 실패 %s: %s)" % (p, e))
                    continue
                for s in d.get("simulations") or []:
                    s["_src"] = src
                    out.append(s)
    return out


def load_tasks():
    by = {}
    for p in glob.glob(os.path.join(DOMAIN, "tasks*.json")) + \
            glob.glob(os.path.join(DOMAIN, "tasks", "*.json")):
        try:
            d = jopen(p)
        except Exception:
            continue
        for t in (d.get("tasks") if isinstance(d, dict) else d) or []:
            if isinstance(t, dict) and t.get("id"):
                by.setdefault(t["id"], (os.path.basename(p), t))
    return by


def load_sidecar():
    """사이드카를 **파일별로** 싣는다 — 키가 sim 고유가 아니기 때문이다.

    `t2_fbsidecar._sim_key`는 첫 유저 발화의 sha1 앞 12자다. 같은 태스크를 여러 런에서 돌리면
    **다른 런의 행이 같은 키로 섞인다**(2026-08-06 실측: 전반부 012 sim에 스모크 런의 104행이 붙었다).
    그래서 (파일, 키)로 싣고, 조회할 때 **그 sim이 속한 arm의 파일만** 본다.
    """
    by = collections.defaultdict(list)
    for pat in SIDECARS:
        for p in sorted(glob.glob(pat)):
            op = gzip.open if p.endswith(".gz") else io.open
            try:
                with op(p, "rt", encoding="utf-8", errors="replace") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            r = json.loads(line)
                        except Exception:
                            continue
                        if r.get("sim"):
                            r["_file"] = os.path.basename(p)
                            by[(os.path.basename(p), r["sim"])].append(r)
            except Exception:
                continue
    return by


def sidecar_files_for(src):
    """arm 태그(`bank_n97_gpu0_main_20260806b`)에 대응하는 사이드카 파일명 조각.

    드라이버가 `fb_n97_gpu<G>_<DATE>.jsonl`로 쓴다(run_n97_nt2.sh). 대응 파일이 하나도 없으면
    그 arm은 **사이드카를 켜지 않은 것**이고, 그 sim의 '0건'은 침묵의 증거가 아니다.
    """
    m = re.search(r"gpu(\d)", src or "")
    d = re.search(r"(\d{8}[a-z]?)$", src or "")
    if not (m and d):
        return []
    # ⚠태그는 **동등 비교**여야 한다. 부분 문자열로 보면 전반부(`…20260806`)가 잔여
    #   런의 파일(`…20260806b`)을 자기 것으로 집어삼킨다 — 1차 수정이 정확히 그렇게 틀렸다.
    out = []
    for f in SIDECAR_FILES:
        ft = re.search(r"_(\d{8}[a-z]?)\.jsonl", f)
        if ft and ft.group(1) == d.group(1) and ("gpu" + m.group(1)) in f:
            out.append(f)
    return out


def rows_for(s, side):
    """이 sim의 사이드카 행 — arm에 맞는 파일에서만."""
    out = []
    for f in sidecar_files_for(s.get("_src")):
        out += side.get((f, sim_key(s))) or []
    return out


def sim_key(s):
    """사이드카 조인 키 — **results.json의 id로는 조인되지 않는다**(2026-08-06 실측: 교집합 0).

    `t2_fbsidecar._sim_key`는 tau2 내부 id를 못 보므로 **첫 유저 발화의 sha1 앞 12자**를 지문으로 쓴다.
    이 함수는 그 규칙의 재현이다. 이걸 맞추기 전까지 이 도구의 "사이드카 0건" 출력은 전부 무의미했다.
    """
    for m in s.get("messages") or []:
        if m.get("role") == "user":
            c = m.get("content")
            if isinstance(c, str) and c.strip():
                return hashlib.sha1(c.strip().encode("utf-8")).hexdigest()[:12]
    return "nouser"


def corpus():
    docs = {}
    for p in glob.glob(os.path.join(DOMAIN, "documents", "*")):
        if os.path.isdir(p):
            continue
        try:
            docs[os.path.basename(p)] = io.open(p, encoding="utf-8", errors="replace").read()
        except Exception:
            pass
    return docs


def short(x, n=None):
    n = n or CUT
    s = x if isinstance(x, str) else json.dumps(x, ensure_ascii=False, sort_keys=True)
    s = re.sub(r"[ \t]+", " ", s.replace("\n", " ⏎ ")).strip()
    return s if len(s) <= n else s[:n] + " …(+%d자)" % (len(s) - n)


def is_ours(text, ours):
    """출처 판정 — 문면 추측 금지([[55]]), 그리고 **부분 문자열 포함은 신호가 아니다**.

    1차판은 우리 문구의 40자 접두가 메시지 *앞부분 어디에든* 있으면 우리 것으로 봤다.
    그 규칙이 012 t1에서 순수한 KB 검색 결과(env)를 '우리'로 찍었다 — A2 문구 조각이
    문서 본문과 겹친 것이다. 우리 층은 **메시지를 앞에서 시작한다**(deny는 `Error:`,
    표면화는 대괄호 태그). 그러므로 판정은 **접두 일치 또는 선두 태그**로 좁힌다.
    """
    body = text.lstrip()
    if body.startswith("Error:"):
        body = body[len("Error:"):].lstrip()
    if TAGRE.match(body):                      # 선두 [T2_…]/[POLICY GATE …]/[SEARCH-EXHAUST] 등
        return True
    return any(body.startswith(pre[:40]) for pre in ours)


def gold_actions(task):
    ec = task.get("evaluation_criteria") or {}
    return ec.get("actions") or [], ec.get("communicate_info") or [], ec.get("nl_assertions") or []


def print_task_def(tid, tasks):
    print("=" * 100)
    print("== %s ==" % tid)
    if tid not in tasks:
        print("  (태스크 정의를 찾지 못함 — DOMAIN=%s)" % DOMAIN)
        return None
    fname, t = tasks[tid]
    print("  정의 파일: %s" % fname)
    ui = t.get("user_scenario") or {}
    ins = ui.get("instructions") or {}
    print("\n-- §0a 손님이 받은 시나리오(=외부 주장의 출처) --")
    # banking_knowledge는 instructions가 산문 한 덩어리다(다른 도메인은 슬롯 딕셔너리).
    if isinstance(ins, str):
        print("  %s" % short(ins, 4000))
    else:
        for k in ("domain", "reason_for_call", "task_instructions", "known_info", "unknown_info"):
            if ins.get(k):
                print("  %-18s %s" % (k, short(ins[k], 1600)))
    if ui.get("persona"):
        print("  %-18s %s" % ("persona", short(ui["persona"], 400)))
    if t.get("description"):
        print("  %-18s %s" % ("description", short(t["description"], 600)))
    # 손님이 들고 있는 도구와 태스크가 요구하는 문서 — 채널·회수 판정의 전제다.
    if t.get("user_tools"):
        print("  %-18s %s" % ("user_tools", short(t["user_tools"], 400)))
    if t.get("required_documents"):
        print("  %-18s %s" % ("required_documents", short(t["required_documents"], 800)))
    acts, comm, nls = gold_actions(t)
    print("\n-- §0b gold 액션 (%d) --" % len(acts))
    for i, a in enumerate(acts):
        print("  %2d. %-38s %s" % (i + 1, a.get("name"), short(a.get("arguments"), 700)))
        if a.get("requestor") and a["requestor"] != "assistant":
            print("      requestor=%s" % a["requestor"])
    if comm:
        print("-- §0c 전달해야 하는 정보 --")
        for c in comm:
            print("   · %s" % short(c, 300))
    if nls:
        print("-- §0d nl 어서션 --")
        for c in nls:
            print("   · %s" % short(c, 300))
    return t


def print_scoring(s, task=None):
    ri = s.get("reward_info") or {}
    print("\n-- §1 채점 (trial %s · %s · %s) --" % (s.get("trial"), s["_src"], s.get("id")))
    # ★채점 기준을 먼저 찍는다(2026-08-06 018 실측): reward_basis=DB인 태스크는 gold 액션이
    #   전부 ✗여도 **통과**한다. 기준 없이 액션 표만 읽으면 통과한 sim을 실패로 귀속하게 된다.
    ec = ((task or {}).get("evaluation_criteria") or {}) if task else {}
    print("  reward=%s  종료=%s  메시지=%d  소요=%.0fs  **reward_basis=%s**"
          % (ri.get("reward"), s.get("termination_reason"), len(s.get("messages") or []),
             s.get("duration") or 0, ec.get("reward_basis") or "?"))
    db = ri.get("db_check")
    if db is not None:
        print("  db_check=%s %s" % (getattr(db, "get", lambda k: None)("db_match") if isinstance(db, dict) else db,
                                    short((db or {}).get("db_reward") if isinstance(db, dict) else "", 200)))
    for i, a in enumerate(ri.get("action_checks") or []):
        print("  %s gold#%d %-36s %s" % ("✓" if a.get("action_match") else "✗", i + 1,
                                         (a.get("action") or {}).get("name"),
                                         "" if a.get("action_match") else short((a.get("action") or {}).get("arguments"), 240)))
    for key in ("nl_assertions", "communicate_checks"):
        for c in (ri.get(key) or []):
            if isinstance(c, dict) and c.get("met") is False:
                print("  ✗ %s: %s" % (key, short(c.get("nl_assertion") or c.get("note") or c, 240)))


def print_trace(s, ours):
    print("\n-- §2 궤적 --")
    turn = 0
    for m in s.get("messages") or []:
        role = m.get("role")
        if role == "assistant":
            turn += 1
            txt = m.get("content")
            if txt:
                print("  [%02d A] %s" % (turn, short(txt)))
            for tc in (m.get("tool_calls") or []):
                print("  [%02d A→call] %s %s" % (turn, eff(tc), short(tc.get("arguments"), CUT)))
        elif role == "user":
            txt = m.get("content")
            if txt:
                print("  [%02d U] %s" % (turn, short(txt)))
            for tc in (m.get("tool_calls") or []):
                print("  [%02d U→call] %s %s" % (turn, eff(tc), short(tc.get("arguments"), CUT)))
        elif role == "tool":
            txt = str(m.get("content") or "")
            src = "우리" if is_ours(txt, ours) else "env"
            mo = TAGRE.search(txt[:200])
            tag = ("[%s]" % mo.group(1)) if mo else ""
            head = "%s%s" % (src, tag)
            print("  [%02d T:%s] %s" % (turn, head, short(txt, CUT if src == "env" else max(CUT, 900))))


def print_calls(s):
    """실행된 호출만 순서대로 — **DB 채점 태스크의 유일한 대조 수단**.

    `db_check`는 `{db_match, db_reward}`만 준다(어느 행이 왜 틀렸는지 없음). 그래서 같은 태스크의
    통과 trial과 실패 trial의 **호출 원장을 나란히 놓는 것**이 원인을 가르는 방법이다. 손님 호출과
    에이전트 호출을 구분해 찍는다(gold의 requestor와 맞춰 읽어야 하므로).
    """
    print("\n-- §2b 호출 원장 (requestor | 도구 | 인자) --")
    n = 0
    for m in s.get("messages") or []:
        role = m.get("role")
        for tc in (m.get("tool_calls") or []):
            if role not in ("assistant", "user"):
                continue
            n += 1
            print("  %2d %-9s %-34s %s" % (n, "손님" if role == "user" else "에이전트",
                                           eff(tc), short(tc.get("arguments"), 220)))
    if not n:
        print("  (호출 없음)")


def print_sidecar(s, side):
    rows = rows_for(s, side)
    print("\n-- §3 우리 층(사이드카) — %d건 --" % len(rows))
    if not rows:
        print("  (이 sim의 사이드카 없음 — '우리 층이 말하지 않았다'는 결론 불가)")
        return
    kinds = collections.Counter(r.get("kind") for r in rows)
    print("  종류: %s" % dict(kinds))
    for r in rows:
        t = str(r.get("text") or "")
        if not t.strip():
            continue
        mo = TAGRE.search(t[:200])
        print("  turn %-3s %-18s %s%s" % (r.get("turn"), r.get("kind"),
                                          ("[%s] " % mo.group(1)) if mo else "", short(t, 700)))


def print_kb(s, docs):
    """검색 질의가 코퍼스에 실재하는가 — '못 찾은 것'과 '없는 것'을 가른다."""
    qs = []
    for m in s.get("messages") or []:
        for tc in (m.get("tool_calls") or []):
            a = tc.get("arguments")
            if isinstance(a, str):
                try:
                    a = json.loads(a)
                except Exception:
                    a = {}
            if isinstance(a, dict):
                for k in ("query", "search_query", "q", "keywords"):
                    if a.get(k):
                        qs.append((eff(tc), str(a[k])))
    if not qs:
        return
    print("\n-- §4 검색 질의 × 코퍼스 실재 (%d질의 · 문서 %d) --" % (len(qs), len(docs)))
    for name, q in qs:
        toks = [w for w in re.findall(r"[a-zA-Z_][a-zA-Z_0-9]{2,}", q.lower()) if w not in STOP]
        hits = {}
        for fn, txt in docs.items():
            low = txt.lower()
            n = sum(1 for w in set(toks) if w in low)
            if n:
                hits[fn] = n
        best = sorted(hits.items(), key=lambda kv: -kv[1])[:3]
        allw = [w for w in set(toks) if not any(w in t.lower() for t in docs.values())]
        print("  %-18s %-60s 최다일치 %s" % (name, short(q, 60),
                                           ", ".join("%s(%d/%d)" % (f, n, len(set(toks))) for f, n in best) or "없음"))
        if allw:
            print("      ⚠코퍼스에 **한 문서에도 없는 단어**: %s" % ", ".join(sorted(allw)[:12]))


DOCID_RE = re.compile(r"\bID:\s*(doc_[A-Za-z0-9_().\-]+)")
SCORE_RE = re.compile(r"^\s*Score:\s*([0-9]*\.?[0-9]+)\s*$", re.M)


def print_retrieval(s):
    """회수가 자라는가 — 소진·무득점 두 술어를 궤적에서 그대로 재현한다.

    두 레버가 이 자리에 걸려 있고 둘 다 012를 표적으로 등재돼 있다:
      `T2_KB_NOHIT_SURFACE`  = 반환 문서가 **전부 0점**인 검색이 K회 연속(t2_gate_patch `_kb_zero_hit`)
      `T2_SEARCH_EXHAUST`    = **새 문서 id가 0**인 검색이 K회 연속(dry streak) ∧ 그 턴이 사임(도구 없는 산문)
    술어를 여기서 다시 계산하면, 사이드카가 없는 런에서도 "발화할 수 있었는가"를 무료로 가른다.
    (발화했는가 ≠ 발화할 수 있었는가 — 후자가 거짓이면 그 레버는 이 실패를 애초에 못 잡는다.)
    """
    seen, dry, zero = set(), 0, 0
    rows = []
    turn = 0
    for m in s.get("messages") or []:
        if m.get("role") == "assistant":
            turn += 1
            resign = not (m.get("tool_calls") or []) and str(m.get("content") or "").strip()
            if resign:
                rows.append((turn, "(사임 턴: 도구 없는 산문)", "", "", dry, zero))
            continue
        if m.get("role") != "tool":
            continue
        txt = str(m.get("content") or "")
        ids = DOCID_RE.findall(txt)
        scores = [float(x) for x in SCORE_RE.findall(txt)]
        if not ids and not scores:
            continue
        new = [i for i in ids if i not in seen]
        seen.update(ids)
        dry = 0 if new else dry + 1
        if scores:
            zero = zero + 1 if all(v == 0.0 for v in scores) else 0
        rows.append((turn, "검색 반환", "%d문서(신규 %d)" % (len(ids), len(new)),
                     ("점수 %.1f~%.1f" % (min(scores), max(scores))) if scores else "점수행 없음",
                     dry, zero))
    if not rows:
        return
    print("\n-- §4b 회수 성장 · 술어 재현 (dry=신규0 연속 · zero=전0점 연속) --")
    for turn, what, a, b, d, z in rows:
        print("  턴%-3d %-24s %-18s %-16s dry=%d zero=%d" % (turn, what, a, b, d, z))
    print("  ⇒ 최대 dry=%d · 최대 zero=%d (문턱은 각각 T2_SEARCH_EXHAUST_TH=2 · T2_KB_NOHIT_K=2)"
          % (max(r[4] for r in rows), max(r[5] for r in rows)))


def main():
    ours = our_templates(HERE)
    if not ours:
        print("⚠A2·엔진 소스에서 우리 문구를 하나도 못 읽었다 — 출처 판정 불가(repo에서 실행할 것)")
    sims = load_sims()
    tasks = load_tasks()
    docs = corpus()
    side = load_sidecar()
    if not TASKS:
        bytask = collections.defaultdict(list)
        for s in sims:
            bytask[s["task_id"]].append(s)
        print("태스크 인자가 없다. 사용 가능한 태스크 %d종 (sim %d)" % (len(bytask), len(sims)))
        fails = [(t, [x for x in v if (x.get("reward_info") or {}).get("reward") != 1.0])
                 for t, v in sorted(bytask.items())]
        print("실패 trial이 있는 태스크: %s"
              % ", ".join("%s(%d/%d)" % (t, len(f), len(bytask[t])) for t, f in fails if f))
        return
    for tid in TASKS:
        t = print_task_def(tid, tasks)
        mine = sorted([s for s in sims if s.get("task_id") == tid], key=lambda x: x.get("trial") or 0)
        if not mine:
            print("  (이 태그에 sim 없음: tag=%s)" % TAG)
            continue
        for s in mine:
            print_scoring(s, t)
            if CALLS:
                print_calls(s)
            print_sidecar(s, side)
            print_retrieval(s)
            print_kb(s, docs)
            if not NOTRACE:
                print_trace(s, ours)


if __name__ == "__main__":
    main()
