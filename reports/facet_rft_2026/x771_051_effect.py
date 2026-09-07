# -*- coding: utf-8 -*-
r"""x771 - **효과 프로브 · 후보 051 (D6)**: `[DUPLICATE-WRITE]` 억제를 A2 선언(`write_once_keys`)이
있는 write 로만 한정하면, 수리 전/후로 **판정이 갈리는가**. 그리고 무엇을 파는가([[70]]).

## 이 프로브가 답해야 하는 네 칸 (사용자 관문 · 2026-09-05)

  ⑴ 결함이 **지금 코드에 있다** — 파일:줄 + 축자           -> PART 1 (소스에서 직접 읽어 대조)
  ⑵ 그 결함이 **실제 실패에 닿는다** — 회수 궤적/사이드카   -> PART 2 (라이브 deny 4발 + 대화 축자)
  ⑶ ★**수리 전/후 판정이 갈린다** — 같은 재료·두 술어      -> PART 3 (arm A_pre / B_post)
  ⑷ **파는 것**을 같은 프로브에서 센다 — 회수분 전수        -> PART 4~5 (deny 221건 전수 분류)

## 규격 ([[78]] 격리 -> 배선 · 프롬프트 저작 0 · 모델 0 · GPU 0)

  * 팔은 **선언 오버라이드 한 칸**뿐이다: `once_only = False | True`.
  * 술어는 **엔진 것을 직접 부른다** — `t2_gate_patch` 를 import 해서
    `_succeeded_mut_keys` · `_once_key_of` · `_mut_key_of` · `_is_effective_write` ·
    `_eff_tool_name` 을 그대로 쓴다. 사본 없음([[67]]).
  * A_pre 의 등록은 **엔진 함수 자체**(`_succeeded_mut_keys`)를 호출하고, 내가 쓴 복제본이
    그것과 **키 집합까지 동일**한지 매 케이스에서 assert 한다 (드리프트 방지).
  * 재료는 **회수 궤적의 실제 호출**이다. 파생 변형을 쓴 자리는 `[파생]` 으로 표시하고
    무엇을 바꿨는지 적는다.

## ⚠재료의 한 가지 제약 (숨기지 않는다)

deny 된 호출은 **궤적에서 제거**되므로 회수 results 에 그 호출의 인자가 없다(설계서 §1f C2 축자:
*"deny 된 호출은 궤적에서 제거되어 `attempted_mutations` 에 없다"*). 그래서 「turn 61 에 모델이 낸
두 번째 submit」의 재료로 **msg23 의 호출을 축자 그대로** 쓴다. 이것은 날조가 아니라 **검사 대상
술어 자신이 단언한 사실**이다 — 라이브 deny 가 걸렸다는 것이 곧
`_mut_key_of(2번째) == _mut_key_of(msg23)` 이고, `_mut_key_of` 는 이름+인자 전체의 동치다.
(문면도 이를 축자로 말한다: *"This exact call (same tool, same arguments) already succeeded"*.)

사용: PYTHONIOENCODING=utf-8 py -3 x771_051_effect.py
"""
import collections
import glob
import gzip
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ENG = os.path.abspath(os.path.join(HERE, "..", "..", "scripts", "distill", "tau2"))
SIMS = os.path.join(HERE, "sim_results")
sys.path.insert(0, ENG)

import t2_gate_patch as G          # noqa: E402  ★엔진 술어 원본
from gate_interpreter import load_domain_a2  # noqa: E402

GATE_SRC = os.path.join(ENG, "t2_gate_patch.py")
RUN051 = "bank_k8143med1_20260904_0135"
RUN072 = "bank_t7383_vsmoke_20260828"
OUT = {}


# ────────────────────────────────────────────────────────────────────────────
# 재료 어댑터 — 회수 JSON 의 dict 를 엔진 술어가 읽는 속성 객체로 (변환 0 · 값 그대로)
# ────────────────────────────────────────────────────────────────────────────
class TC(object):
    def __init__(self, d):
        self.id = d.get("id")
        self.name = d.get("name")
        self.arguments = d.get("arguments")
        self.requestor = d.get("requestor", "assistant")

    def __repr__(self):
        return "TC(%s,%s)" % (self.name, json.dumps(self.arguments, ensure_ascii=False)[:70])


class MSG(object):
    def __init__(self, d):
        self.role = d.get("role")
        self.id = d.get("id")
        self.content = d.get("content")
        self.error = d.get("error", False)
        self.tool_calls = [TC(t) for t in (d.get("tool_calls") or [])]


def load_sim(run, task_id):
    p = os.path.join(SIMS, run + ".results.json.gz")
    d = json.load(gzip.open(p, "rt", encoding="utf-8", errors="replace"))
    for s in (d.get("simulations") or []):
        if str(s.get("task_id")) == task_id:
            return s
    raise SystemExit("no %s in %s" % (task_id, run))


# ────────────────────────────────────────────────────────────────────────────
# ★두 팔 — 차이는 `once_only` 한 칸뿐이다.
#   A_pre  (수리 전 = 지금 코드) : 등록 (mut, once) · 조회 (once, mut)
#   B_post (수리 후)             : 등록 (once,)     · 조회 (once,)
# 엔진 축자 대응:
#   등록 t2_gate_patch.py:6121  `for k in (_mut_key_of(tc), _once_key_of(tc, a2w)):`
#   조회 t2_gate_patch.py:12278 `for _cand in (_once_key_of(_dc, _a2_of(self)), _mut_key_of(_dc)):`
# ────────────────────────────────────────────────────────────────────────────
def register(msgs, a2w, once_only):
    """`_succeeded_mut_keys` 의 팔-매개변수 판. 로직은 엔진 :6104-6127 축자 복제."""
    out = {}
    ms = list(msgs or [])
    for i, m in enumerate(ms):
        for tc in (getattr(m, "tool_calls", None) or []):
            if not G._is_effective_write(G._eff_tool_name(tc), a2w):
                continue
            tid = getattr(tc, "id", None)
            for j in range(i + 1, len(ms)):
                mj = ms[j]
                if str(getattr(mj, "role", "")) != "tool" or getattr(mj, "id", None) != tid:
                    continue
                body = str(getattr(mj, "content", "") or "")
                if not getattr(mj, "error", False) and not body.lstrip().startswith("Error:"):
                    keys = ((G._once_key_of(tc, a2w),) if once_only
                            else (G._mut_key_of(tc), G._once_key_of(tc, a2w)))
                    for k in keys:
                        if k and k not in out:
                            out[k] = (i, body)
                break
    return out


def verdict(msgs, cand, a2w, once_only):
    """엔진 :12276-12322 의 조회부 축자 복제. 반환 (판정, 키, 앞선 msg, 문면종류)."""
    dupmap = register(msgs, a2w, once_only)
    if once_only == (WIRED == "B_post"):
        # ★드리프트 방지 — **지금 배선된 팔**의 등록은 엔진 함수 자체와 키 집합까지 같아야 한다.
        #   2026-09-05 수리 전에는 그 팔이 A_pre 였고 수리 후에는 B_post 다. 어느 쪽이 배선돼
        #   있든 «내 복제본 == 라이브 엔진» 을 매 케이스 검산한다(팔이 바뀌어도 증거가 안 샌다).
        eng = G._succeeded_mut_keys(msgs, a2w)
        assert set(eng) == set(dupmap), (
            "%s 복제본이 엔진 _succeeded_mut_keys 와 다르다" % WIRED)
    if not G._is_effective_write(G._eff_tool_name(cand), a2w):
        return ("PASS(not-write)", None, None, None)
    cands = ((G._once_key_of(cand, a2w),) if once_only
             else (G._once_key_of(cand, a2w), G._mut_key_of(cand)))
    for c in cands:
        if c and c in dupmap:
            at, _res = dupmap[c]
            tpl = "ONCE_FB" if str(c).startswith("once|") else "MUT_FB"
            return ("DENY", c, at, tpl)
    return ("PASS", None, None, None)


def show(tag, msgs, cand, a2w, note=""):
    a = verdict(msgs, cand, a2w, once_only=False)
    b = verdict(msgs, cand, a2w, once_only=True)
    flip = "★갈림" if a[0] != b[0] else "  불변"
    print("  %-34s A_pre=%-6s(%s)  B_post=%-6s(%s)  %s %s"
          % (tag, a[0], a[3] or "-", b[0], b[3] or "-", flip, note))
    OUT.setdefault("cases", []).append(
        {"case": tag, "pre": a[0], "pre_tpl": a[3], "pre_at": a[2],
         "post": b[0], "post_tpl": b[3], "flip": a[0] != b[0], "note": note})
    return a, b


def hr(t):
    print("\n" + "=" * 96 + "\n" + t + "\n" + "=" * 96)


# ══════════════════════════════════════════════════════════════════════════
hr("PART 1 — ⑴ 코드가 지금 **어느 팔**로 배선돼 있나 (소스에서 직접 읽는다)")
# ★2026-09-05 — 종전 PART 1 은 «결함 축자 4줄이 있나» 만 물었다. 수리를 실은 뒤에는 그 물음이
#   자동으로 MISS 가 되어 프로브가 자기 수리를 «실패» 로 읽는다. 물음을 **배선된 팔의 식별**로
#   바꾼다: 결함 축자 2줄(A_pre) ↔ D6 축자 2줄(B_post) 중 어느 쪽이 파일에 있는가.
#   줄번호는 참고값이고 판정은 **축자 검색**이다([[77]] 검색 경로 명시).
src = open(GATE_SRC, encoding="utf-8").read().splitlines()


def _hits(txt):
    return [i + 1 for i, s in enumerate(src) if s.strip() == txt]


PRE = [(6121, "for k in (_mut_key_of(tc), _once_key_of(tc, a2w)):"),
       (12314, "for _cand in (_once_key_of(_dc, _a2_of(self)), _mut_key_of(_dc)):")]
POST = [(0, "for k in (_once_key_of(tc, a2w),):"),
        (0, "for _cand in (_once_key_of(_dc, _a2_of(self)),):")]
COMMON = [(12306, "_dupmap = _succeeded_mut_keys(state.messages, _a2_of(self))"),
          (12321, '_tpl = (_DUP_WRITE_ONCE_FB if str(_dk).startswith("once|")')]
for tag, group in (("A_pre(결함)", PRE), ("B_post(D6)", POST), ("공통", COMMON)):
    for ln, txt in group:
        h = _hits(txt)
        print("  [%s] %-12s %-56s -> 줄 %s"
              % ("O" if h else "-", tag, txt[:56], h or "없음"))
pre_n = sum(1 for _, t in PRE if _hits(t))
post_n = sum(1 for _, t in POST if _hits(t))
com_n = sum(1 for _, t in COMMON if _hits(t))
WIRED = ("A_pre" if (pre_n == 2 and post_n == 0) else
         "B_post" if (post_n == 2 and pre_n == 0) else "MIXED")
alive = (WIRED == "A_pre")      # ⑴ «결함이 살아 있나» = «A_pre 로 배선돼 있나»
print("  => 배선된 팔 = %s   (A_pre 축자 %d/2 · B_post 축자 %d/2 · 공통 %d/2)"
      % (WIRED, pre_n, post_n, com_n))
assert WIRED != "MIXED", "두 자리가 서로 다른 팔이다 — 반쪽 수리"
assert com_n == 2, "공통 축자가 사라졌다 — 블록 자체가 바뀌었다"
OUT["wired_arm"] = WIRED
a2b = load_domain_a2("banking_knowledge")
woks = a2b.get("write_once_keys") or []
print("  A2 banking_knowledge write_once_keys = %d 건: %s"
      % (len(woks), [w.get("applies_when", {}).get("prefix") or w.get("applies_to") for w in woks]))
OUT["alive"] = alive

# ══════════════════════════════════════════════════════════════════════════
hr("PART 2 — ⑵ 그 결함이 실제 실패에 닿나 (회수 궤적 + 사이드카)")
sim = load_sim(RUN051, "task_051")
raw = sim["messages"]
msgs = [MSG(m) for m in raw]
print("  run=%s  task_051  reward=%s  msgs=%d"
      % (RUN051, (sim.get("reward_info") or {}).get("reward"), len(msgs)))

# 라이브 deny 회수 (사이드카)
den = []
for ln in gzip.open(os.path.join(SIMS, "fb_%s.jsonl.gz" % RUN051), "rt",
                    encoding="utf-8", errors="replace"):
    if "DUPLICATE-WRITE" not in ln:
        continue
    d = json.loads(ln)
    if d.get("kind") == "tool-deny" and "task_051" in str(d.get("simtag")):
        den.append(d)
print("  사이드카 tool-deny(DUPLICATE-WRITE) = %d 발 · turn=%s"
      % (len(den), [d["turn"] for d in den]))
print("  문면 축자(앞 150자): %s" % str(den[0]["text"])[:150].replace("\n", " "))
OUT["live_denies_051"] = [d["turn"] for d in den]

# 대화 축자 — deny 가 손님에게 닿은 자리
print("  m65 축자: %s" % str(raw[65].get("content") or "")[:160].replace("\n", " "))
print("  m66 축자: %s" % str(raw[66].get("content") or "")[:160].replace("\n", " "))
print("  m70 축자: %s" % str(raw[70].get("content") or "")[:160].replace("\n", " "))

# ★deny 와 PIN_READ 의 순서 — 어느 쪽이 상류인가 (설계서 §1d-6 이 남긴 칸)
order = []
for ln in gzip.open(os.path.join(SIMS, RUN051 + ".log.gz"), "rt",
                    encoding="utf-8", errors="replace"):
    if "task_051" not in ln:
        continue
    if "[T2_DUP_WRITE] deny" in ln:
        order.append("DUP_DENY")
    elif "[T2_PIN_READ] pinned call_discoverable" in ln:
        order.append("PIN_call")
    elif "tool_choice={" in ln and "unified_regen" in ln:
        order.append("REGEN_forced")
i0 = order.index("DUP_DENY")
print("  로그 순서(첫 deny 부터 3칸): %s" % order[i0:i0 + 3])
OUT["order_after_first_deny"] = order[i0:i0 + 3]

# 재료: msg23 의 실제 호출
c23 = msgs[23].tool_calls[0]
print("  재료 = msg23 축자 호출: %s" % json.dumps(raw[23]["tool_calls"][0]["arguments"],
                                                  ensure_ascii=False))
print("  mut_key  = %s" % G._mut_key_of(c23))
print("  once_key = %r   <- 선언 없음 = 정책이 반복을 금지한 적 없다" % G._once_key_of(c23, a2b))

# ══════════════════════════════════════════════════════════════════════════
hr("PART 3 — ⑶ ★수리 전/후 판정이 갈리는가 (같은 재료 · 팔 한 칸)")
print("[T1] 051 라이브 재현 — 창=msgs[0:61](결제 성공 m60 포함) · 후보=msg23 축자 재제출")
a, b = show("T1 051 resubmit @turn61", msgs[:61], c23, a2b,
            "gold 051_7 = 051_2 바이트 동일")
t1_flip = (a[0] == "DENY" and b[0] == "PASS")

print("\n[T1b] 창 사다리 — 첫 제출 직전에는 두 팔 모두 통과해야 한다(술어가 '무조건 거부'가 아님)")
show("T1b 창=msgs[0:23] (앞선 성공 없음)", msgs[:23], c23, a2b, "부정통제")

print("\n[T2] 같은 sim 의 **다른 미선언 write** — 파는 것의 실물 모양")
c59 = msgs[59].tool_calls[0]          # pay_credit_card_from_checking (실제 실행 · m60 성공)
show("T2 pay_credit_card 재호출 @61", msgs[:61], c59, a2b, "미선언 -> 보호 상실")
show("T2b 같은 호출 @창 msgs[0:59]", msgs[:59], c59, a2b, "부정통제")

print("\n[T3] ★선언된 write 는 보호가 유지돼야 한다 — 회수 궤적 재료(task_072 · %s)" % RUN072)
sim72 = load_sim(RUN072, "task_072")
m72 = [MSG(m) for m in sim72["messages"]]
found = []          # (호출 index, 결과 index, tc)
for i, m in enumerate(m72):
    for tc in m.tool_calls:
        if "apply_checking_account_credit" not in str(G._exact_tool_name(tc)):
            continue
        for j in range(i + 1, len(m72)):
            if m72[j].role == "tool" and m72[j].id == tc.id:
                if not m72[j].error and not str(m72[j].content or "").lstrip().startswith("Error:"):
                    found.append((i, j, tc))
                break
acc_i, acc_j, acc_c = found[0]
print("  회수 재료 %d 건(성공한 apply_checking_account_credit):" % len(found))
for (i, j, tc) in found:
    print("      msg%-3d -> 결과 msg%-3d  %s" % (i, j, json.dumps(tc.arguments,
                                                                 ensure_ascii=False)))
W72 = m72[:acc_j + 1]                       # ★결과 메시지까지 포함해야 등록된다
print("  once_key = %s" % G._once_key_of(acc_c, a2b))
show("T3 동일 호출 재제출", W72, acc_c, a2b, "선언됨 -> 둘 다 막혀야")

# [파생] 같은 계좌·다른 금액 = t7378 task_074 가 실제로 뚫었던 모양
d = dict(acc_c.arguments)
inner = json.loads(d["arguments"]) if isinstance(d.get("arguments"), str) else {}
amt_key = next((k for k in inner if "amount" in k.lower()), None)
if amt_key:
    inner2 = dict(inner)
    inner2[amt_key] = float(inner[amt_key]) + 15.5
    d2 = dict(d)
    d2["arguments"] = json.dumps(inner2)
    cvar = TC({"id": "x771-var", "name": acc_c.name, "arguments": d2})
    show("T4 [파생] 같은 계좌·다른 금액", W72, cvar, a2b,
         "t7378 074 형 · once 만 잡는다")
# 다른 계좌 = 오차단이 없어야 — ★파생이 아니라 **같은 sim 의 실제 두 번째 계좌 호출**
if len(found) > 1:
    show("T5 다른 계좌(회수 실물 msg%d)" % found[1][0], W72, found[1][2], a2b,
         "둘 다 통과해야")

# ══════════════════════════════════════════════════════════════════════════
hr("PART 4 — ⑷ [[70]] 무엇을 파는가 · 회수분 **전수** (fb 사이드카 전체)")
ONCE_MARK = "This tool was already run successfully for this same"
MUT_MARK = "This exact call (same tool, same arguments)"
cls = collections.Counter()
per_run = collections.defaultdict(collections.Counter)
tgt_mut = collections.Counter()
tgt_once = collections.Counter()
sim_kind = {}          # (run, simtag) -> set of kinds
route_tgt = {}         # (run, simtag, turn) -> target
for p in sorted(glob.glob(os.path.join(SIMS, "fb_*.jsonl.gz"))):
    run = os.path.basename(p)[3:-len(".jsonl.gz")]
    try:
        rows = []
        for ln in gzip.open(p, "rt", encoding="utf-8", errors="replace"):
            if "dup_write" not in ln and "DUPLICATE-WRITE" not in ln:
                continue
            rows.append(json.loads(ln))
    except Exception as e:
        print("  ERR %s %r" % (p, e))
        continue
    for d in rows:
        if d.get("kind") == "route" and d.get("agent") == "dup_write":
            route_tgt[(run, str(d.get("simtag")), d.get("turn"))] = d.get("target")
    for d in rows:
        if d.get("kind") != "tool-deny":
            continue
        t = str(d.get("text") or "")
        if "DUPLICATE-WRITE" not in t:
            continue
        k = "once" if ONCE_MARK in t else ("mut" if MUT_MARK in t else "?")
        cls[k] += 1
        per_run[run][k] += 1
        st = str(d.get("simtag"))
        sim_kind.setdefault((run, st), set()).add(k)
        # target: 같은 sim 의 route 중 turn 이 가장 가까운(>=) 것
        cands = [(tt, v) for (r2, s2, tt), v in route_tgt.items()
                 if r2 == run and s2 == st and tt >= d.get("turn", 0)]
        tg = min(cands)[1] if cands else "?"
        (tgt_once if k == "once" else tgt_mut)[tg] += 1
print("  회수 사이드카 DUPLICATE-WRITE tool-deny 총 %d 발" % sum(cls.values()))
print("    once(선언·수리 후에도 유지) = %d      mut(수리가 없애는 것) = %d"
      % (cls["once"], cls["mut"]))
print("  ↳ 수리는 회수분 deny 의 %.1f%% 를 없앤다" % (100.0 * cls["mut"] / max(1, sum(cls.values()))))
print("  없어지는 deny 의 표적(도구별):")
for k, v in tgt_mut.most_common():
    print("      %-42s %3d" % (k, v))
print("  유지되는 deny 의 표적:")
for k, v in tgt_once.most_common():
    print("      %-42s %3d" % (k, v))
OUT["census"] = {"once": cls["once"], "mut": cls["mut"],
                 "mut_targets": dict(tgt_mut), "once_targets": dict(tgt_once)}

# ── 4b. ★도메인 분리 — 이번 발사는 banking 97 이다([[79]] 프레임)
#    회수 코퍼스에서 banking sim 은 task_id 가 `task_NNN` · retail 은 순수 숫자다(실측:
#    bank_t7360_smoke ids=['task_050',...] vs bank_t7391_retail ids=['0','1','2']).
bank_cls = collections.Counter()
bank_tgt = collections.Counter()
for (run, st), ks in sim_kind.items():
    pass
for p in sorted(glob.glob(os.path.join(SIMS, "fb_*.jsonl.gz"))):
    run = os.path.basename(p)[3:-len(".jsonl.gz")]
    for ln in gzip.open(p, "rt", encoding="utf-8", errors="replace"):
        if "DUPLICATE-WRITE" not in ln:
            continue
        d = json.loads(ln)
        if d.get("kind") != "tool-deny":
            continue
        st = str(d.get("simtag"))
        if not st.startswith("task_"):
            continue                     # retail
        t = str(d.get("text") or "")
        k = "once" if ONCE_MARK in t else "mut"
        bank_cls[k] += 1
        cands = [(tt, v) for (r2, s2, tt), v in route_tgt.items()
                 if r2 == run and s2 == st and tt >= d.get("turn", 0)]
        bank_tgt[(k, min(cands)[1] if cands else "?")] += 1
print("\n  ★banking 만 (이번 발사 프레임 [[79]]): 총 %d 발 · once %d · mut %d"
      % (sum(bank_cls.values()), bank_cls["once"], bank_cls["mut"]))
for (k, tg), v in sorted(bank_tgt.items()):
    print("      %-5s %-42s %3d" % (k, tg, v))
OUT["census_banking"] = {"once": bank_cls["once"], "mut": bank_cls["mut"],
                         "targets": {"%s|%s" % k: v for k, v in bank_tgt.items()}}

print("\n  ★P6c 가 지목한 «미선언 유일성 write 3종» 이 회수분에서 실제로 막힌 적이 있나:")
for nm in ("deposit_check", "order_replacement_credit_card",
           "request_temporary_debit_card_limit_increase"):
    print("      %-46s mut-deny %d 건" % (nm, tgt_mut.get(nm, 0)))
OUT["p6c_three"] = {nm: tgt_mut.get(nm, 0) for nm in
                    ("deposit_check", "order_replacement_credit_card",
                     "request_temporary_debit_card_limit_increase")}

# ══════════════════════════════════════════════════════════════════════════
hr("PART 5 — ⑷ 파는 것의 값: mut-deny 가 걸린 sim 의 **보상**은 얼마였나")
rew = {}
for (run, st) in sim_kind:
    p = os.path.join(SIMS, run + ".results.json.gz")
    if run in rew or not os.path.exists(p):
        continue
    try:
        dd = json.load(gzip.open(p, "rt", encoding="utf-8", errors="replace"))
        rew[run] = {str(s.get("task_id")): (s.get("reward_info") or {}).get("reward")
                    for s in (dd.get("simulations") or [])}
    except Exception:
        rew[run] = {}
tab = collections.Counter()
rows = []
for (run, st), ks in sorted(sim_kind.items()):
    tid = st.split("#")[0]
    r = (rew.get(run) or {}).get(tid)
    if r is None and not tid.startswith("task_"):
        r = (rew.get(run) or {}).get("task_" + tid)
    kind = "mut" if "mut" in ks else "once"
    tab[(kind, "reward=%s" % r)] += 1
    rows.append((run, st, kind, r))
print("  mut-deny 가 걸린 sim 의 보상 분포 (sim 단위):")
for (kind, r), v in sorted(tab.items()):
    print("      %-5s %-14s %3d sim" % (kind, r, v))
print("  ⚠reward=None = 그 run 의 results 에서 task_id 로 못 찾은 것(주로 retail 인덱스 표기)")
print("\n  mut-deny 가 걸렸는데 **통과한(reward=1.0)** sim = 수리가 위험을 지는 자리:")
risky = [r for r in rows if r[2] == "mut" and r[3] == 1.0]
bank_risky = [r for r in risky if r[1].startswith("task_")]
for r in risky:
    print("      %-34s %-18s %s" % (r[0], r[1], "  <-BANKING" if r[1].startswith("task_") else ""))
print("      계 %d sim (그중 banking %d · retail %d)"
      % (len(risky), len(bank_risky), len(risky) - len(bank_risky)))
OUT["risky_pass_sims"] = [(r[0], r[1]) for r in risky]
OUT["risky_pass_banking"] = [(r[0], r[1]) for r in bank_risky]
print("\n  ★banking 위험 sim 이 실제로 무엇을 막고 통과했나(도구):")
for (run, st) in [(r[0], r[1]) for r in bank_risky]:
    tg = sorted({v for (r2, s2, tt), v in route_tgt.items() if r2 == run and s2 == st})
    print("      %-34s %-18s -> %s" % (run, st, tg))
    OUT.setdefault("risky_banking_tools", {})["%s|%s" % (run, st)] = tg
OUT["reward_table"] = {"%s|%s" % k: v for k, v in tab.items()}

# ══════════════════════════════════════════════════════════════════════════
hr("PART 6 — [[70]] 부호표: 수리안 3 지선을 같은 계기로 나란히")
print("  회수분 DUPLICATE-WRITE deny 총 %d (banking %d)" % (sum(cls.values()), sum(bank_cls.values())))
print("  ┌ 선택지 ─────────────────────┬ 사는 것(deny 소멸) ┬ 파는 것(보호 유지) ─┐")
print("  │ ①현행 (T2_DUP_WRITE=1)      │        0           │  221 (bank 54)      │")
print("  │ ②D6 = 선언된 write 만       │  213 (bank 52)     │    8 (bank  2)      │")
print("  │ ③레버 OFF (=go_stack 정본)  │  221 (bank 54)     │    0                │")
print("  └─────────────────────────────┴────────────────────┴─────────────────────┘")
OUT["signtable"] = {"current": {"buy": 0, "sell": sum(cls.values())},
                    "D6_declared_only": {"buy": cls["mut"], "sell": cls["once"]},
                    "lever_off": {"buy": sum(cls.values()), "sell": 0}}

# ── [[81]] 배선 실측 — 정본과 런처가 같은 값인가 (추정 말고 파일에서 읽는다)
_LAUNCH = ("go_stack.sh", "run_ours_task.sh", "run_night_ab.sh", "run_t7363_night.sh",
           "run_t7364.sh", "run_t7365.sh")
print("\n  [[81]] `T2_DUP_WRITE` 실측 값:")
_vals = {}
for _fn in _LAUNCH:
    _p = os.path.join(ENG, _fn)
    _v = (re.findall(r"T2_DUP_WRITE=(\d)",
                     open(_p, encoding="utf-8", errors="replace").read())
          if os.path.exists(_p) else [])
    _vals[_fn] = _v
    print("      %-22s %s" % (_fn, _v or "(없음)"))
_agree = len({v for vv in _vals.values() for v in vv}) == 1
print("      => 정본 ↔ 런처 일치: %s" % _agree)
OUT["wiring_T2_DUP_WRITE"] = {"values": _vals, "agree": _agree}

# ══════════════════════════════════════════════════════════════════════════
hr("PART 6b — ★배선 후 검산: 라이브 엔진이 정말 B_post 인가 (2026-09-05 수리)")
# 프로브의 두 팔은 **복제본**이다. 수리를 실은 뒤 물어야 할 것은 «복제본이 갈리나» 가 아니라
# «라이브 엔진이 B_post 팔과 같나» 다. 같은 재료로 엔진 함수를 직접 불러 키 집합을 대조한다.
_bind = []
for _tag, _w in (("051 창=msgs[0:61]", msgs[:61]), ("072 창(선언 write 포함)", W72)):
    _eng = set(G._succeeded_mut_keys(_w, a2b))
    _isb = (_eng == set(register(_w, a2b, True)))
    _isa = (_eng == set(register(_w, a2b, False)))
    _bind.append((_tag, _isb, _isa))
    print("  %-26s 엔진원장 == B_post:%-5s  == A_pre:%-5s  (키 %d개)"
          % (_tag, _isb, _isa, len(_eng)))
live_is_post = all(b[1] for b in _bind) and not all(b[2] for b in _bind)
print("  => 라이브 엔진 = %s" % ("B_post ★수리가 배선됐다" if live_is_post else "B_post 아님"))
_lv = verdict(msgs[:61], c23, a2b, once_only=(WIRED == "B_post"))
_ov = verdict(msgs[:61], c23, a2b, once_only=(WIRED != "B_post"))
print("  051 재제출 @turn61 — 배선된 팔(%s)=%s · 반대 팔=%s"
      % (WIRED, _lv[0], _ov[0]))
live_flip = (_lv[0] == "PASS" and _ov[0] == "DENY") if WIRED == "B_post" else t1_flip
OUT["postfix_check"] = {"live_is_post": live_is_post, "wired_verdict": _lv[0],
                        "other_arm_verdict": _ov[0], "live_flip": live_flip}

hr("판정")
print("  ⑴ 배선된 팔           : %s  (결함 생존 = %s)"
      % (WIRED, "YES" if OUT["alive"] else "NO — 수리가 실렸다"))
print("  ⑵ 실패에 닿음         : 라이브 deny %d 발 (turn %s)"
      % (len(OUT["live_denies_051"]), OUT["live_denies_051"]))
print("  ⑶ ★전/후 판정 갈림    : %s" % ("YES (DENY -> PASS)" if t1_flip else "NO"))
print("  ⑷ 파는 것             : mut-deny %d 발 소멸 · once-deny %d 발 유지 · 위험 sim %d(banking %d)"
      % (cls["mut"], cls["once"], len(risky), len(bank_risky)))
print("     banking 만          : mut %d 소멸 · once %d 유지" % (bank_cls["mut"], bank_cls["once"]))
# ★판정은 **배선된 팔에 따라 다른 것을 요구한다**.
#   A_pre(수리 전) : 결함이 살아 있고 전/후가 갈리면 PROBE-PASS  = «실어라»
#   B_post(수리 후): 엔진이 정말 B_post 이고 배선된 팔에서 051 이 PASS 면 WIRED-PASS = «실렸다»
if WIRED == "A_pre":
    _v = "PROBE-PASS" if (alive and t1_flip) else "PROBE-FAIL"
else:
    print("  ⑸ ★배선 검산          : 엔진==B_post %s · 배선팔 051 판정 %s · [[81]] 일치 %s"
          % (live_is_post, OUT["postfix_check"]["wired_verdict"], _agree))
    _v = ("WIRED-PASS" if (live_is_post and live_flip and _agree) else "WIRED-FAIL")
print("\n  VERDICT = %s" % _v)
OUT["verdict"] = _v

jp = os.path.join(HERE, "x771_051_effect.json")
json.dump(OUT, open(jp, "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
print("  -> %s" % jp)
