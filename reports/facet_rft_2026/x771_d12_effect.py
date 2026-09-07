# -*- coding: utf-8 -*-
"""x771 — D12 효과 프로브 (`user_action_feedback` 오부착 · 표적 014)

★관문 4칸을 이 파일 하나로 실측한다.
  ⑴ 결함이 지금 코드에 있나        — 원문 + `git log` 로 **수리 상태**를 판정한다
  ⑵ 그 결함이 실제 실패에 닿는가   — 회수 궤적/사이드카에서 그 발화·그 칸을 짚는다
  ⑶ ★수리 전/후 판정이 갈리는가   — **같은 재료**에 두 술어를 먹여 «죽는 칸» 이 달라지는지 센다
  ⑷ [[70]] 파는 것                 — 회수분 **전수**로 정당한 부착이 함께 죽는지 센다

규격([[78]]): 프롬프트 저작 0 · 사본 0.
  두 팔 모두 **엔진 자신의 소스 바이트를 실행**한다 — 재타이핑하면 드리프트가 생긴다([[67]]).
    PRE  = `git show 8f056511^:scripts/distill/tau2/t2_gate_patch.py` 의 `rw_fb` 대입 한 줄
    POST = 현재 `t2_gate_patch.py` 의 `_rw_c` 블록 (커밋 8f056511 이 넣은 것)
  이관 집합은 `t2_gate_patch._transfer_tools(a2)` 를 **직접 호출**해 얻는다(A2 도출·엔진 리터럴 0).
  재료는 가짜 입력 0 — `sim_results/` 회수분(trace/log)에서 뽑은 **실제 부착 사건**뿐.

[[57]] 부정통제 2종:
  NC-1 되돌리기 — PRE 팔을 같은 재료에 먹이면 014 의 이관 호출이 다시 죽는가
  NC-2 A2 제거  — `_transfer_tools({})` = ∅ 이면 POST 팔이 PRE 팔과 **판정 동일**인가
                  (= 갈림의 출처가 A2 선언이지 엔진 하드코딩이 아니라는 증거 · [[05]])
"""
import gzip
import io
import json
import os
import re
import subprocess
import sys
import textwrap
from collections import Counter, defaultdict

REPO = r"C:\workspace\ba-frft"
ENG = os.path.join(REPO, "scripts", "distill", "tau2")
SIMS = os.path.join(REPO, "reports", "facet_rft_2026", "sim_results")
FIXCOMMIT = "8f056511"          # D12 수리 커밋 (아래에서 존재를 검증한다)
sys.path.insert(0, ENG)


def say(*a):
    print(" ".join(str(x) for x in a))


# ═══════════════════════════════════════════════════════════════════════════
# ⑴ 결함이 지금 코드에 있나
# ═══════════════════════════════════════════════════════════════════════════
def _git(*args):
    return subprocess.run(["git", "-C", REPO] + list(args), capture_output=True,
                          text=True, encoding="utf-8", errors="replace").stdout


RE_RWFB = re.compile(r"^[ \t]*rw_fb = .*_ufb.*$", re.M)


def gate1():
    now = io.open(os.path.join(ENG, "t2_gate_patch.py"), encoding="utf-8").read()
    pre = _git("show", "%s^:scripts/distill/tau2/t2_gate_patch.py" % FIXCOMMIT)
    assert pre, "PRE 소스를 못 가져왔다 — 커밋 %s 가 없다" % FIXCOMMIT
    subj = _git("log", "-1", "--format=%h %ad %s", "--date=short", FIXCOMMIT).strip()

    say("=" * 84)
    say("[GATE 1] 결함이 지금 코드에 있나")
    say("   수리 커밋 : %s" % subj)
    m_now = RE_RWFB.search(now)
    m_pre = RE_RWFB.search(pre)
    nline_now = now[:m_now.start()].count("\n") + 1
    nline_pre = pre[:m_pre.start()].count("\n") + 1
    say("   PRE  t2_gate_patch.py:%d  %s" % (nline_pre, m_pre.group(0).strip()))
    say("   NOW  t2_gate_patch.py:%d  %s" % (nline_now, m_now.group(0).strip()))
    buggy_now = "(am.tool_calls or [None])[0], _ufb" in m_now.group(0)
    say("   ⇒ 구판 무조건 부착이 **현재 코드에 남아 있나** = %s" % buggy_now)
    # ⓐ 문면
    def _clause(src):
        i = src.index('"Error: [ACTION] ')
        j = src.index('.replace("{tool}", _utgt)', i)
        body = src[i:j]
        body = re.sub(r"#[^\n]*", "", body)          # 주석 제거
        return re.sub(r"\s+", " ", "".join(re.findall(r'"([^"]*)"', body)))
    say("   ⓐ 문면 PRE : ...%s..."
        % _clause(pre)[_clause(pre).find("so do not search"):][:110])
    say("   ⓐ 문면 NOW : ...%s..."
        % _clause(now)[_clause(now).find("so do not search"):][:110])
    say("   ⇒ 판정: %s" % ("**결함 생존**" if buggy_now
                          else "**이미 수리됨** (커밋 %s · 회수분은 전부 그 이전이라 PRE 거동)" % FIXCOMMIT))
    return now, pre


# ═══════════════════════════════════════════════════════════════════════════
# ⑵ 재료 회수 — sim_results 전수에서 실제 `resolve_write` 부착 사건만 뽑는다
# ═══════════════════════════════════════════════════════════════════════════
RE_AUD = re.compile(r"\[T2_STACK\] audit route=(.*?) chose=(\[.*?\]) differs=")
RE_RES = re.compile(r"\[T2_RESOLVE\] user-action instruct target=(\S+)")
RE_MAT = re.compile(r"\[T2_MATERIAL_GATE\].*?turn=(\d+) calls=(\S*)")
RE_SIM = re.compile(r"^\[sim=([^\]]+)\]\s*(.*)$")
RE_RW = re.compile(r"\('resolve_write', '([^']+)'\)")


def _iter_lines():
    for fn in sorted(os.listdir(SIMS)):
        p = os.path.join(SIMS, fn)
        if fn.startswith("trace_") and fn.endswith(".jsonl.gz"):
            run = fn[len("trace_"):-len(".jsonl.gz")]
            try:
                for l in gzip.open(p, "rt", encoding="utf-8", errors="replace"):
                    l = l.strip()
                    if not l:
                        continue
                    try:
                        r = json.loads(l)
                    except Exception:
                        continue
                    yield (run, r.get("sim"), str(r.get("line") or ""), "trace")
            except Exception:
                continue
        elif fn.endswith(".log.gz"):
            run = fn[:-len(".log.gz")]
            try:
                for l in gzip.open(p, "rt", encoding="utf-8", errors="replace"):
                    m = RE_SIM.match(l.strip())
                    if m:
                        yield (run, m.group(1), m.group(2), "log")
            except Exception:
                continue


def recover_events():
    """실제 부착 사건 = audit 줄의 `chose` 에 ('resolve_write', Y) 가 있는 것.

    Y = `_eff_tool_name(am.tool_calls[0])` 이다 — `rw_fb[0]` 이 초안의 첫 호출이고
    `_fbtag` 가 그 **호출 객체**에 `resolve_write` 태그를 걸었기 때문(t2_gate_patch:12837).
    ⚠**채널 분리가 이 프로브의 급소**다. `_fbtag` 는 변수 이름으로 태그를 붙이므로
      `rw_fb` 대입 **여섯 곳 전부**가 audit 에 `('resolve_write', …)` 로 찍힌다
      (:10303 resolve_write 반환 · :10314 resolve-verify · :11135/:11469/:11497 등).
      D12 수리는 그중 **`_ufb`(ACTION) 채널 한 곳**만 건드린다. 그래서 ACTION 채널의
      자기 계기 — `[T2_RESOLVE] user-action instruct target=X` (그 대입 직후 print) —
      가 **그 턴에 찍힌 사건만** D12 사정권이다. 소비하면 즉시 비운다(턴 넘어 전이 금지).
    직전 `[T2_MATERIAL_GATE] ... calls=` = 그 턴 초안의 전체 호출 목록(찍힌 때만).
    """
    st = defaultdict(lambda: {"utgt": None, "calls": None, "turn": None})
    evs, seen = [], Counter()
    for (run, sim, text, src) in _iter_lines():
        s = st[(run, sim, src)]
        m = RE_RES.search(text)
        if m:
            s["utgt"] = m.group(1)
            continue
        m = RE_MAT.search(text)
        if m:
            s["turn"] = int(m.group(1))
            c = m.group(2)
            s["calls"] = [] if c in ("-", "") else [x for x in c.split(",") if x]
            continue
        m = RE_AUD.search(text)
        if not m:
            continue
        rw = RE_RW.search(m.group(2))
        if not rw:
            continue
        utgt = s["utgt"]
        s["utgt"] = None                       # 소비 즉시 비운다 — 턴 넘어 전이 금지
        sig = (run, sim, s["turn"], utgt, rw.group(1))
        seen[(sig, src)] += 1
        evs.append(dict(run=run, sim=sim, turn=s["turn"], utgt=utgt,
                        landed=rw.group(1), calls=list(s["calls"] or []), src=src,
                        occ=seen[(sig, src)], action_channel=(utgt is not None)))
    # 같은 런이 trace 와 log 에 둘 다 있으면 한 벌만 남긴다(중복 계수 방지)
    by = defaultdict(list)
    for e in evs:
        by[(e["run"], e["sim"], e["turn"], e["utgt"], e["landed"], e["occ"])].append(e)
    out = []
    for _k, g in by.items():
        tr = [x for x in g if x["src"] == "trace"]
        out.append((tr or g)[0])
    return out


# ═══════════════════════════════════════════════════════════════════════════
# ⑶ 두 팔 — 엔진 소스 바이트를 그대로 실행한다
# ═══════════════════════════════════════════════════════════════════════════
class _Call(object):
    """회수된 **이름**을 나르는 껍데기. 판단 0 — `_eff_tool_name` 이 읽는 필드만 갖는다."""
    def __init__(self, name):
        self.name = name
        self.arguments = {}
        self.id = None


class _AM(object):
    def __init__(self, names):
        self.tool_calls = [_Call(n) for n in names]


def _arm_pre(pre_src):
    m = RE_RWFB.search(pre_src)
    return textwrap.dedent(m.group(0)), "t2_gate_patch(PRE):%d" % (pre_src[:m.start()].count("\n") + 1)


def _arm_post(now_src):
    m = RE_RWFB.search(now_src)
    start = now_src.rfind("\n_rw_c = None", 0, m.start())
    if start < 0:
        start = now_src.rfind("_rw_c = None", 0, m.start())
        start = now_src.rfind("\n", 0, start)
    blk = now_src[start + 1:m.end()]
    return textwrap.dedent(blk), "t2_gate_patch(NOW):%d-%d" % (
        now_src[:start].count("\n") + 2, now_src[:m.end()].count("\n") + 1)


def make_evaluator(code, G, a2, xfer_override=None):
    """엔진 소스 조각을 실행해 `rw_fb` 를 얻는다. 이름 바인딩만 우리가 채운다."""
    compiled = compile(code, "<engine-slice>", "exec")

    def ev(names, utgt):
        ns = {
            "am": _AM(names), "_ufb": "ACTION-TEXT", "_utgt": utgt, "a2": a2,
            "_transfer_tools": (lambda _a2: xfer_override) if xfer_override is not None
                               else G._transfer_tools,
            "_eff_tool_name": G._eff_tool_name,
            "_sys": type("S", (), {"stderr": open(os.devnull, "w")})(),
            "print": lambda *a, **k: None,
        }
        exec(compiled, ns)
        rw = ns.get("rw_fb")
        if not rw or rw[0] is None:
            return None
        return G._eff_tool_name(rw[0])
    return ev


def main():
    now_src, pre_src = gate1()

    import gate_interpreter as gi
    import t2_gate_patch as G
    a2 = gi.load_domain_a2("banking_knowledge")
    xfer = set(G._transfer_tools(a2) or ())
    say("")
    say("[엔진 술어 직접 호출] t2_gate_patch._transfer_tools(load_domain_a2('banking_knowledge'))")
    say("   -> %r   (a2['transfer_tools'] 명시=%r ⇒ notice 게이트 applies_to 에서 도출)"
        % (sorted(xfer), a2.get("transfer_tools")))
    say("   a2['user_action_feedback'] = %r  ⇒ 라이브 문면은 엔진 기본값이다"
        % a2.get("user_action_feedback"))

    pre_code, pre_at = _arm_pre(pre_src)
    post_code, post_at = _arm_post(now_src)
    say("")
    say("[두 팔 · 엔진 소스 바이트] PRE=%s  POST=%s (%d줄)"
        % (pre_at, post_at, post_code.count("\n") + 1))

    ev_pre = make_evaluator(pre_code, G, a2)
    ev_post = make_evaluator(post_code, G, a2)

    say("")
    say("=" * 84)
    say("[GATE 2] 회수분에서 실제 부착 사건 뽑기 (가짜 입력 0)")
    allevs = recover_events()
    other_ch = [e for e in allevs if not e["action_channel"]]
    evs = [e for e in allevs if e["action_channel"]]
    say("   rw_fb 부착 사건 전체 = %d" % len(allevs))
    say("   ├ ACTION 채널(`_ufb` · D12 사정권)                 = %d" % len(evs))
    say("   └ 그 밖 rw_fb 채널(resolve-verify 등 · 수리 무관)   = %d  ← 분모에서 뺀다"
        % len(other_ch))
    say("   사건 %d · 런 %d · sim %d · 태스크 %d"
        % (len(evs), len({e["run"] for e in evs}), len({(e["run"], e["sim"]) for e in evs}),
           len({str(e["sim"]).split("#")[0] for e in evs})))
    say("   태스크별: %s" % Counter(str(e["sim"]).split("#")[0] for e in evs).most_common(14))
    say("   부착된 호출(landed): %s" % Counter(e["landed"] for e in evs).most_common(12))
    say("   문면 표적(_utgt)   : %s" % Counter(e["utgt"] for e in evs).most_common(10))

    say("")
    say("   ── 014 실물 (관문 ⑵) ──")
    n14 = 0
    for e in evs:
        if str(e["sim"]).startswith("task_014"):
            n14 += 1
            say("   run=%s sim=%s turn=%s  _utgt=%s → 부착=%s  (MATERIAL_GATE calls=%s)"
                % (e["run"], e["sim"], e["turn"], e["utgt"], e["landed"], e["calls"]))
    fbp = os.path.join(SIMS, "fb_bank_re151med1_20260904_0255.jsonl.gz")
    if os.path.exists(fbp):
        for i, l in enumerate(gzip.open(fbp, "rt", encoding="utf-8")):
            r = json.loads(l)
            if r.get("kind") == "tool-deny" and "do not transfer for this" in (r.get("text") or "") \
                    and str(r.get("simtag") or "").startswith("task_014"):
                say("   fb 사이드카 row=%d turn=%s kind=%s simtag=%s"
                    % (i, r.get("turn"), r.get("kind"), r.get("simtag")))
                say("   축자: %r" % ((r.get("text") or "")[:140]))

    say("")
    say("=" * 84)
    say("[GATE 3 ★] 같은 재료 · 수리 전 ↔ 수리 후 — 죽는 칸이 갈리는가")
    rows = []
    for e in evs:
        names = e["calls"] if e["calls"] else [e["landed"]]
        if e["landed"] in names:
            names = [e["landed"]] + [n for n in names if n != e["landed"]]
        else:
            names = [e["landed"]] + names
        kp, kq = ev_pre(names, e["utgt"]), ev_post(names, e["utgt"])
        rows.append(dict(e, names=names, killed_pre=kp, killed_post=kq, flip=(kp != kq)))
    n = len(rows)
    fl = [r for r in rows if r["flip"]]
    say("   전수 %d 사건 · **판정 갈림 %d (%.1f%%)**" % (n, len(fl), 100.0 * len(fl) / max(1, n)))
    say("")
    say("   갈리는 사건 전수:")
    for r in fl:
        say("     %-38s %-22s turn=%-4s _utgt=%-24s 죽는칸 전='%s' → 후=%s"
            % (r["run"][:38], r["sim"], r["turn"], r["utgt"], r["killed_pre"], r["killed_post"]))
    if not fl:
        say("     (없음)")

    say("")
    say("=" * 84)
    say("[GATE 4 · [[70]]] 파는 것 — 회수분 전수 부호표")
    buy = [r for r in rows if r["killed_pre"] in xfer]
    legit = [r for r in rows if r["killed_pre"] == r["utgt"]]
    other = [r for r in rows if r["killed_pre"] not in xfer and r["killed_pre"] != r["utgt"]]
    sold = [r for r in rows if r["flip"] and r["killed_pre"] not in xfer]
    kept = [r for r in rows if not r["flip"]]
    say("   ① 사는 것 = 부착이 **이관 도구**를 죽인 사건 (014 결함군)      = %d" % len(buy))
    for r in buy:
        say("        %s %s turn=%s _utgt=%s" % (r["run"][:36], r["sim"], r["turn"], r["utgt"]))
    say("   ② 정당한 부착 = 죽은 칸이 곧 문면 표적                        = %d" % len(legit))
    say("   ③ 그 밖 = 표적도 이관도구도 아닌 칸이 죽었다                  = %d" % len(other))
    say("        분포: %s" % Counter(r["killed_pre"] for r in other).most_common(12))
    say("   ★파는 것(수리 후 안 죽게 된 **비이관** 칸)                    = %d" % len(sold))
    for r in sold:
        say("        %s %s turn=%s 전='%s'" % (r["run"][:36], r["sim"], r["turn"], r["killed_pre"]))
    say("   판정 불변(전달 그대로)                                        = %d/%d" % (len(kept), n))

    # ★파는 것 둘째 항 — 부착이 사라진 턴은 그 문면이 **그 턴에** 전달되지 않는다
    #   (`rw_fb=None` 이면 UserMessage 리마인더 조건 `not am.tool_calls` 도 못 넘는다:13075).
    #   그래서 sim 단위로 **그 표적의 ACTION 문면이 통째로 사라지는가**를 센다.
    say("")
    say("   ── 파는 것 ② 문면 전달 손실 (sim 단위) ──")
    per = defaultdict(list)
    for r in rows:
        per[(r["run"], r["sim"], r["utgt"])].append(r)
    lost, partial = [], []
    for k, g in per.items():
        if not any(x["flip"] for x in g):
            continue
        (lost if all(x["killed_post"] is None for x in g) else partial).append((k, len(g)))
    say("     그 표적의 ACTION 부착이 그 sim 에서 **전부** 사라짐 = %d sim-표적" % len(lost))
    say("     일부만/다른 칸으로 이동(여전히 전달)                = %d sim-표적" % len(partial))
    # 부착이 사라져도 같은 문면이 그 sim 에서 **리마인더 채널**로 이미 갔는가 (사이드카 실측)
    checked = hit_rem = nofile = 0
    total_loss = []
    for (run, sim, utgt), _g in lost:
        p = os.path.join(SIMS, "fb_%s.jsonl.gz" % run)
        if not os.path.exists(p):
            nofile += 1
            continue
        checked += 1
        needle = "[ACTION] '%s'" % utgt
        # ⚠[[55]] 계기 주의: 구판 사이드카는 `simtag` 이 없고 `sim` 이 **해시**다 —
        #   sim 정합을 못 하니 그때는 **런 단위 존재**만 말하고 «손실» 이라고 하지 않는다.
        sim_hit = run_hit = has_tag = False
        try:
            for l in gzip.open(p, "rt", encoding="utf-8", errors="replace"):
                r = json.loads(l)
                if r.get("simtag"):
                    has_tag = True
                if r.get("kind") not in ("reminder-user", "reminder-assistant"):
                    continue
                if needle not in (r.get("text") or ""):
                    continue
                run_hit = True
                if str(r.get("simtag") or r.get("sim") or "") == str(sim):
                    sim_hit = True
                    break
        except Exception:
            pass
        if sim_hit or (run_hit and not has_tag):
            hit_rem += 1
        else:
            total_loss.append((run, sim, utgt, "sim정합=%s·런존재=%s·simtag유=%s"
                              % (sim_hit, run_hit, has_tag)))
    say("     └ 그중 사이드카가 있는 %d 건 중 **같은 문면이 그 sim 의 리마인더 채널로 이미 전달** = %d"
        % (checked, hit_rem))
    say("       (사이드카 없음 %d · 리마인더로도 안 갔으면 그 sim 은 그 문면을 통째로 잃는다)"
        % nofile)
    say("     ⇒ [[70]] 실비용 = 전달이 통째로 사라지는 sim-표적 %d 건" % (checked - hit_rem))
    for row in total_loss:
        say("        전달손실 %s %s _utgt=%s  [%s]" % (row[0][:36], row[1], row[2], row[3]))
    say("")
    say("   ── 사는 것: 태스크별 (이관 호출이 살아나는 사건) ──")
    say("     %s" % Counter(str(r["sim"]).split("#")[0] for r in buy).most_common(20))
    FAIL46 = set("""010 027 029 038 039 041 048 060 061 084 097 079 007 026 037 040 046 054 055
056 063 064 066 067 069 071 077 078 082 085 086 101 087 091 088 014 015 051 068 092 059""".split())
    hit = sorted({str(r["sim"]).split("#")[0] for r in buy
                  if str(r["sim"]).split("_")[-1].split("#")[0] in FAIL46})
    say("     이번 캠페인 실패 46 과 겹치는 태스크: %s" % hit)

    say("")
    say("=" * 84)
    say("[[57]] 부정통제")
    n14r = [r for r in rows if str(r["sim"]).startswith("task_014")]
    ok1 = bool(n14r) and all(r["killed_pre"] in xfer and r["killed_post"] != r["killed_pre"]
                             for r in n14r)
    say("   NC-1 되돌리기: PRE 팔에서 014 사건 %d 건이 전부 이관 호출을 죽이는가 = %s" % (len(n14r), ok1))
    for r in n14r:
        say("        turn=%s  PRE 죽는칸=%s  ·  POST 죽는칸=%s  (초안 calls=%s)"
            % (r["turn"], r["killed_pre"], r["killed_post"], r["names"]))
    ev_post0 = make_evaluator(post_code, G, a2, xfer_override=set())
    same = sum(1 for r in rows if ev_post0(r["names"], r["utgt"]) == r["killed_pre"])
    say("   NC-2 A2 제거: _transfer_tools(...)=∅ 로 POST 팔을 돌리면 PRE 와 판정 동일 = %d/%d"
        % (same, n))
    say("        ⇒ 전건 동일이면 갈림의 출처는 **A2 선언**이지 엔진 리터럴이 아니다([[05]])")
    ev_pre_x = make_evaluator(pre_code, G, a2, xfer_override=set())
    say("   NC-3 무의미 대조: PRE 팔은 이관집합을 아예 참조하지 않는다 = %s"
        % all(ev_pre_x(r["names"], r["utgt"]) == r["killed_pre"] for r in rows))

    say("")
    say("=" * 84)
    verdict = "PROBE-PASS" if len(fl) > 0 else "PROBE-FAIL"
    say("[판정] ⑶ 갈림 %d/%d ⇒ **%s**" % (len(fl), n, verdict))

    outp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_ep_work",
                        "x771_d12_effect.json")
    try:
        os.makedirs(os.path.dirname(outp), exist_ok=True)
        json.dump(dict(n=n, flips=len(fl), xfer=sorted(xfer), buy=len(buy), legit=len(legit),
                       other=len(other), sold=len(sold), rows=rows),
                  io.open(outp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        say("[산출] %s" % outp)
    except Exception as e:
        say("[산출 실패] %r" % (e,))


if __name__ == "__main__":
    main()
