# -*- coding: utf-8 -*-
r"""x276 — 추천-수렴 격리 프로브 (③ⓒ · 유료 0 · 엔진 0 · 설계 2026-08-12).

정본 설계서: reports/facet_rft_2026/PROBE_X276_X277_DESIGN_2026_08_12.md §1.

## 왜 (설계서 §0.1 · [[08]] 포렌식 실측)

`bank_dbw_on_20260812` task_071 은 write-미도달 시행에서 **후보를 열어 둔 채 검색만 반복**했다:
  · trial0 (16 호출): dense 검색 7회 — 요구가 올 때마다 추천을 갈아치움
    (Platinum Reserve → Sky Blue → True Blue → Gold Plus Saver) → 끝내 transfer 검색 ×3 · write 0.
  · trial1 (23 호출): 4개 상품 사양 검색 순환 · agent-측 write 0.
태스크 지시는 정확히 반대 — *"Get ONE recommendation ... not give you options to choose from"*.
요구(닫힌 술어들)로 후보를 **제거해 하나로 수렴**해야 한다. ⇒ **[[63]] 빼기 실패의 신규 후보**
(같은 기전·기준만 다름: "요구사항 충족").

## 질문 ([[62]] ① — 레버 전에 결손을 격리로 잰다)

요구사항 전부 + 이미 회수된 KB 사실이 손에 있을 때, 모델은 **하나로 수렴**할 수 있는가 —
못 하면 능력 결손, 하면 전달/문면 문제. 이 프로브는 측정이지 레버가 아니다.

## v2 사유 (설계서 §5b 사전등록 · 원장 C449 — 1회차 실측이 프로브 결함 둘을 선고)

  결함 A: N_NEG 과유도 OVERCLAIM 6/8 — B_ONE 주명령이 탈출 갈래를 지배 → 탈출 갈래를 동급
          명령으로 승격 + 오답 정의 명시(N_NEG 은 같은 상수를 쓰므로 자동 반영).
  결함 B: trial1 절단 오염 — cut=20 문맥에서 A_LIVE TRANSFER 7/8 → 절단 규칙 v2(아래) +
          절단 후 transfer 토큰 잔존 시 그 trial 자기 무효(SKIP).

## 절단 v2 (사전등록 · 설계서 §5b 결함 B · 정보-맞춤 [[18]])

  trial0: cut=35 고정 — user[34] 가 마지막 요구 발화
          ("I can't do True Blue if it charges any overdraft fee ... keep Gold Plus Saver") 직후.
  trial1: 기계 규칙 v2 — cut = min(첫 공식 상품명-포함 assistant 발화 인덱스,
          transfer 토큰('transfer' 낱말 대소문자 무시 또는 '###TRANSFER###')이 어느 메시지에든
          처음 등장하는 인덱스). 둘 다 못 찾으면 절단점 미발견 — 그 trial SKIP(자기 무효 선고).
  공통 게이트(두 trial 다·발화 전): 절단 문맥에 KB 검색 tool 출력 ≥1 필수 — 없으면 SKIP.
          절단 후 문맥에 transfer 토큰 잔존 시 "절단 오염 — SKIP"(자기 무효 선고).

## 팔 (trial 별 n=8 · i==0 temp0 나머지 0.7 · mx=450)

  A_LIVE  절단 문맥 그대로(+사이드카) → 다음 행동      ← 라이브 재현(재검색 루프?) · tools 有
  B_ONE   + "state the ONE ..." 직문                    ← 수렴 **능력** · tools=None
  C_SUB   + "eliminate ... state what remains"          ← **지시된 빼기**([[63]] 선행 대조) · tools=None
  D_FREE  절단 문맥 − **우리 사이드카 문장만** 제거     ← 루프가 우리 문구에서 오는가([[55]] 귀속·
          (단일 델타·x256 E_FREE 동형)                     리뷰 B1) · tools 有
  D_CEIL  user 발화+tool 출력만(구 D_MIN)               ← 격리 **상한** 기록용·귀속용 아님(리뷰 B1) · tools 有
  N_NEG   + 어떤 상품도 못 채우는 요구 변조             ← 부정통제([[57]]) — 답이 **달라져야** 한다

⛔[[03b]]/[[23]]: gold(상품명)는 프롬프트에 절대 넣지 않는다 — 채점에서만,
  그것도 `reward_info.action_checks` 기계 추출로만 쓴다. 산출물은 팔별 빈도표다.

실행: T2_PROBE_URL=http://localhost:8141/v1/chat/completions python3 x276_converge_iso.py [N]
"""
import collections
import io
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from x216_read_and_offset import chat                              # noqa: E402
import x241_uncalled_unlock_probe as U                             # noqa: E402

SIMS = os.environ.get("X276_SIMDIR", "/home/woori/scratch/tau2-bench/data/simulations")
TAG = os.environ.get("X276_TAG", "bank_dbw_on_20260812")
TASK = "task_071"
FB = "/home/woori/scratch/logs/fb_%s.jsonl" % TAG
HERE = os.path.dirname(os.path.abspath(__file__))
DISPATCH = "call_discoverable_agent_tool"
WRITE_SUB = "open_bank_account"          # gold 추출용 부분 문자열(태스크 도구 이름 판정만)

# 팔 문면 (도메인 상품명 0 · 설계서 §1 표의 델타 축자)
# [[23]] provenance: 이전 판의 "checking 1 + savings 1" 두-축 열거는 절단 문맥 발화에서
# 축자 출처를 확인하지 못해 제거 — gold 의 {business_checking, business_savings} 짝 구조와
# 구별 불가(출처 못 대면 넣지 마라). 축 구조는 채점에서만 쓴다(A3 doc_index 기계 분류 +
# 설계서 §1 리뷰 B2 "그룹별 정확히 하나" 사전등록 술어 — gold 는 채점 전용이 합법).
# [[23]] 출처(2026-08-12 리모트 축자 확인): 두-축(checking 1 + savings 1) 구조는 gold 가 아니라
#   **손님 발화 자체**다 — trial1 user[6]: "recommend **one** business checking and **one**
#   business savings that fit them (n...)" · trial0 user[1]: "open a new business checking
#   account ... and I also want to open a business savings acc...". 채점 술어(축별 정확히 1)와
#   문면이 같은 출처에서 나오므로 되돌린다(수정 1번의 단축은 출처 미확인 시점의 보수 조치였다).
# v2 (설계서 §5b 결함 A · C449 N_NEG OVERCLAIM 6/8): 탈출 갈래를 동급 명령으로 승격 +
# 오답 정의 명시. N_NEG 팔은 같은 상수를 쓰므로 자동 반영(문구 자체가 부정통제 대상 — 분리 금지).
B_ONE_FB = (
    "[system] Based only on what is already in this conversation, state the ONE business "
    "checking account and the ONE business savings account that satisfy every requirement "
    "the customer stated - name exactly one of each. If no single account satisfies every "
    "stated requirement, or the retrieved documents are not sufficient to decide, say "
    "exactly that instead — naming a best-effort pick that fails a stated requirement "
    "counts as a wrong answer."
)
C_SUB_FB = (
    "[system] List the candidate accounts that appear in the retrieved documents, then "
    "eliminate every candidate that fails at least one of the customer's stated "
    "requirements, and state only what remains."
)
# [[57]] 부정통제: 어떤 상품도 못 채우는 요구(모바일 수표 입금 한도 $10M/일) — 정답이 달라져야 한다.
N_NEG_FB = (
    "[user] One more requirement: the mobile check deposit limit must be at least "
    "$10,000,000 per day."
)

TRANSFER = re.compile(r"\btransfer\b", re.I)
# ★v3 결함 C (설계서 §5b v3 개정): v2 의 `\btransfer\b` 는 **정당한 상품 사양**
#   ("wire transfer"·"same-day ACH transfer")을 오염으로 판정해 trial0 전 문맥을
#   SKIP 시켰다. 이관-신호만 남긴다 — 전부 env·user-sim 프로토콜의 닫힌 문자열이고
#   도메인 산문 해석 0(자금 이체는 이 집합에 없다).
TRANSFER_TOK = re.compile(r"###TRANSFER###|transfer_to_human|human agent|transfer me", re.I)
# [[57]] 계기: 적합-판정에 정박된 구문만 NOFIT — 일상 부정어("no monthly fee")가 단독
# 매치되면 무효 게이트(N NOFIT ≤5/8)가 발화 불능이 된다(감사 지적 2). v2 B_ONE_FB 가 직접
# 가르치는 탈출 문형 "No single (business checking) account satisfies ..." 도 잡도록 개재
# 수식어(single·business/checking/savings)만 허용 — 뒤쪽 meet/satisf/fit/qualif 정박은
# 유지해 과매치를 막는다(지적 2 취지 보존).
NOFIT = re.compile(r"\b(?:no|none of the)\s+(?:single\s+)?(?:(?:business|checking|savings)\s+){0,2}"
                   r"(?:account|candidate|option|product)s?\b.{0,80}"
                   r"\b(?:meet|satisf|fit|qualif)"
                   r"|does\s*n[o']t\s+meet|not\s+sufficient|insufficient|missing\s+fact", re.I)
# 설계서 §1 리뷰 B2 배제-문맥 어휘(±80자 창): eliminate/not/fail/exclude/rule out/
# won't work/doesn't meet 계열.
EXCL = re.compile(r"eliminat|\bnot\b|fail|exclud|rule[sd]? out|won'?t work"
                  r"|doesn'?t meet|does not meet", re.I)


def official_names():
    """A3 doc_index 주어 → 공식 명칭 집합. 기계 전개뿐(엔진 선별 0·[[59]]). x270 34~42행 사본."""
    p = os.path.join(HERE, "a2", "banking_knowledge.specific.json")
    a = json.load(io.open(p, encoding="utf-8"))
    di = (a.get("policy_ontology") or {}).get("doc_index") or {}
    out = {}
    for g, subs in di.items():
        out[g] = {" ".join(w.capitalize() for w in str(k).split("_")) for k in subs}
    return out


def names_by_axis():
    """공식명을 checking/savings 축으로 기계 분류 — 군 이름의 낱말 소속만 본다(선별 0).

    `_general_` 파생 키만 제외한다(상품명이 아니라 색인 자리표시자 — 기계 규칙: strip 후
    'general' 또는 길이<3). 그 외 이름은 전부 포함한다(선별하지 않는다).
    """
    axis = {"checking": set(), "savings": set()}
    for g, names in official_names().items():
        gl = str(g).lower()
        key = "checking" if "checking" in gl else ("savings" if "savings" in gl else None)
        if key is None:
            continue
        for nm in names:
            s = nm.strip()
            if len(s) < 3 or s.lower() == "general":
                continue
            axis[key].add(s)
    return axis


AXIS = names_by_axis()


def name_spans(text):
    """공식명 매치를 **leftmost-longest** 비겹침 탐욕으로 찾는다 — 정규식/집합 소속만.

    구 긴-이름-우선 소진 규칙은 접두 수식형에서 오인식('sky blue account' 에서 개인
    'Blue Account'@4 가 business 'Sky Blue'@0 을 밀어냄 — 감사 지적 6). (start, -len)
    정렬 후 앞선 것부터 선택하면 'Sky Blue'@0 이 이기고 겹치는 후자는 탈락한다.
    반환: (정규화 소문자 텍스트, [(start, end, 이름, 축), ...]).
    """
    low = " ".join(str(text or "").split()).lower()
    cand = []
    for ax, s in AXIS.items():
        for nm in s:
            pat = r"(?<![a-z0-9])%s(?![a-z0-9])" % re.escape(nm.lower())
            for m in re.finditer(pat, low):
                cand.append((m.start(), m.start() - m.end(), nm, ax))
    used = [False] * len(low)
    hits = []
    for st, neg_len, nm, ax in sorted(cand):
        en = st - neg_len
        if any(used[st:en]):
            continue
        for k in range(st, en):
            used[k] = True
        hits.append((st, en, nm, ax))
    return low, hits


def name_hits(text):
    """text 에 등장하는 공식명을 축별 **고유 집합**으로 — name_spans 의 집계 뷰."""
    found = {"checking": set(), "savings": set()}
    for _st, _en, nm, ax in name_spans(text)[1]:
        found[ax].add(nm)
    return found


def conv_names(text):
    """설계서 §1 리뷰 B2 사전등록 술어 — **포함 판정 금지**(감사 지적 4).

    각 등장 위치 ±80자 창에 EXCL(배제 어휘) 매치가 있으면 배제-문맥 등장으로 태깅.
    반환: 축별 **비-배제 문맥 등장** 이름 집합(배제-문맥 전용 이름은 빠진다).
    """
    low, hits = name_spans(text)
    live = {"checking": set(), "savings": set()}
    for st, en, nm, ax in hits:
        if not EXCL.search(low[max(0, st - 80):en + 80]):
            live[ax].add(nm)
    return live


def gold_classes(sim):
    """gold = action_checks 기계 추출(하드코딩 0 · x256.gold_args 60~66행 방식).

    open_bank_account* 디스패치의 중첩 인자에서 {account_type: account_class} 를 모은다.
    """
    out = {}
    # reward_info 부재/None(중도 종료·크래시 시행)이어도 죽지 않고 gold={} 로 진행
    # (score 는 gc/gs 빈 문자열이면 CONV+OFF 처리 — 감사 지적 7).
    for a in ((sim.get("reward_info") or {}).get("action_checks") or []):
        ac = a["action"]
        if ac["name"] == DISPATCH and WRITE_SUB in json.dumps(ac["arguments"]):
            inner = ac["arguments"].get("arguments")
            try:
                inner = json.loads(inner) if isinstance(inner, str) else (inner or {})
            except Exception:
                inner = {}
            t = str(inner.get("account_type") or "").strip()
            c = str(inner.get("account_class") or "").strip()
            if t and c:
                out[t] = c
    return out


def build(sim, cut, with_ours=True):
    """문맥 조립 — x256_dispatcher_write_probe.build 71~93행 사본(FB 경로만 태그별 인자화)."""
    import t2_fbsidecar as S

    class _M(object):
        def __init__(s, r, c):
            s.role, s.content = r, c

    ours = collections.defaultdict(list)
    if with_ours and os.path.exists(FB):
        key = S._sim_key([_M(m.get("role"), m.get("content")) for m in sim["messages"]])
        for ln in open(FB, encoding="utf-8", errors="replace"):
            o = json.loads(ln)
            if o.get("sim") == key and (o.get("text") or "").strip():
                ours[o.get("turn")].append(" ".join(o["text"].split()))
    out = []
    for i, m in enumerate(sim["messages"][:cut]):
        r, c = m.get("role"), " ".join(str(m.get("content") or "").split())
        tcs = [tc.get("name") for tc in (m.get("tool_calls") or [])]
        if tcs:
            out.append("[%s calls] %s" % (r, ", ".join(x for x in tcs if x)))
        if c:
            out.append("[%s] %s" % (r, c[:700]))
        for t in ours.get(i, ()):
            out.append("[system] %s" % t[:900])
    return "\n".join(out)


def build_min(sim, cut):
    """D_CEIL — user 발화 + tool 출력만 [:cut]. assistant·사이드카 전부 제거(다중 델타).

    격리 **상한** 기록 전용·[[55]] 귀속용 아님(리뷰 B1) — 귀속은 단일-델타 D_FREE 가 한다.
    """
    out = []
    for m in sim["messages"][:cut]:
        r = m.get("role")
        if r not in ("user", "tool"):
            continue
        c = " ".join(str(m.get("content") or "").split())
        if c:
            out.append("[%s] %s" % (r, c[:700]))
    return "\n".join(out)


def kb_out_count(msgs):
    """절단 문맥 안 KB 검색 tool 출력 수 — tool_call id 짝(x273 live_kb_hits 48~57행 동형)."""
    byid = {}
    for m in msgs:
        for tc in (m.get("tool_calls") or []):
            byid[tc.get("id")] = tc.get("name")
    n = 0
    for m in msgs:
        if m.get("role") == "tool" and str(byid.get(m.get("id")) or "").startswith("KB_search"):
            n += 1
    return n


def find_cut(sim, trial):
    """절단 규칙 v2(사전등록 · 설계서 §5b 결함 B · 모듈 docstring §절단 v2).

    trial0 = 35 고정. trial1 = min(첫 공식명-포함 assistant 인덱스, transfer 토큰이
    어느 메시지에든 — content·tool_calls 직렬화 포함 — 처음 등장하는 인덱스).
    둘 다 없으면 None(§5b 규칙 무매치 — 호출측이 SKIP·자기 무효 선고 방침).
    """
    if trial == 0:
        return 35            # user[34] = 마지막 요구 발화 직후(설계서 §1 "마지막 요구 발화 직후")
    # ★v3 결함 D (설계서 §5b v3 개정): 추천 후보는 **첫 KB 검색 tool 출력 이후**만.
    #   v2 는 손님이 말한 자기 기존 계좌를 모델이 되뇐 msg[3] 을 추천으로 잡아 cut=3 →
    #   문맥에 검색 결과 0 → SKIP 이었다. 회수 전 언급은 추천이 아니다(기계 규칙 —
    #   임의 상수 하한으로 되돌리는 것이 아니다).
    kb_idx = None
    for i, m in enumerate(sim["messages"]):
        if m.get("role") == "tool" and "KB_search" in json.dumps(m, ensure_ascii=False):
            kb_idx = i
            break
        c = str(m.get("content") or "")
        if m.get("role") == "tool" and re.search(r"\bID:\s*doc_", c):
            kb_idx = i       # 검색 결과 형식(문서 id 열거)도 회수로 인정
            break
    rec_idx = None
    for i, m in enumerate(sim["messages"]):
        if m.get("role") != "assistant" or (kb_idx is None or i <= kb_idx):
            continue
        f = name_hits(str(m.get("content") or ""))
        if f["checking"] or f["savings"]:
            rec_idx = i      # 첫 추천 발화(회수 이후) — 문맥은 그 직전까지
            break
    tr_idx = None
    for i, m in enumerate(sim["messages"]):
        if TRANSFER_TOK.search(json.dumps(m, ensure_ascii=False)):
            tr_idx = i       # transfer 토큰 첫 등장(어느 role 이든 · tool_calls 인자 포함)
            break
    cands = [x for x in (rec_idx, tr_idx) if x is not None]
    return min(cands) if cands else None


def score(r, gold):
    """기계 채점 — 설계서 §1 리뷰 B2 술어. argmax·'정답은 X' 출력 없음(산출물은 빈도표).

    CONV = 그룹별 정확히 하나의 상품이 **비-배제 문맥**에 등장 ∧ 나머지는 부재이거나
    배제-문맥 전용(감사 지적 4 — 단순 포함 판정은 C_SUB 의 나열-후-배제 정답을 MULTI 로
    뒤집는다). MULTI = 어느 축이든 비-배제 등장 2개 이상.
    """
    msg = r or {}
    for tc in (msg.get("tool_calls") or []):
        nm = str((tc.get("function") or {}).get("name") or "")
        if "KB_search" in nm:
            return "SEARCH"
        if re.search(r"unlock|call_discoverable|verify|log_verification", nm, re.I):
            return "ACT"
        return "TOOL(%s)" % nm[:20]
    t = str(msg.get("content") or "")
    live = conv_names(t)
    nc, ns = len(live["checking"]), len(live["savings"])
    if nc == 1 and ns == 1:
        c_nm, s_nm = next(iter(live["checking"])), next(iter(live["savings"]))
        gc = str(gold.get("business_checking") or "").lower()
        gs = str(gold.get("business_savings") or "").lower()
        ok = gc and gs and c_nm.lower() == gc and s_nm.lower() == gs
        return "CONV+GOLD" if ok else "CONV+OFF"
    if nc >= 2 or ns >= 2:
        return "MULTI"
    if TRANSFER.search(t):
        return "TRANSFER"
    return "PLAIN"


def score_neg(r, gold):
    """N_NEG 전용([[57]]): 공식명 짝을 여전히 단정하면 OVERCLAIM, 부적합 선언이면 NOFIT."""
    base = score(r, gold)
    if base.startswith("CONV"):
        return "OVERCLAIM"
    if base == "MULTI":
        return base          # 짝을 여전히 나열 = 미분화 — NOFIT 자격 없음(감사 지적 2)
    t = str((r or {}).get("content") or "")
    if NOFIT.search(t):
        return "NOFIT"
    return base


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 8
    d = json.load(io.open(os.path.join(SIMS, TAG, "results.json"), encoding="utf-8"))
    # trial 필드로 정렬 선택 — 리스트 순서 가정 금지(감사 지적 3: cut=35 는 trial0 전용이라
    # 순서가 어긋나면 측정 무효. bank_givemiss_replay.py:35 관례 동형).
    sims = sorted([s for s in d["simulations"] if s["task_id"] == TASK],
                  key=lambda s: s.get("trial") or 0)
    print("x276 — 추천-수렴 격리 (%s · %s · 시행 %d개 · n=%d)\n" % (TAG, TASK, len(sims), n))

    for trial in (0, 1):
        # trial 필드 일치로 선택 — 정렬 후 위치 인덱싱은 trial0 부재(크래시)·부분 런에서
        # trial0 전용 cut=35 를 엉뚱한 궤적에 적용한다(감사 지적 3 의 정렬 교정은 절반).
        sim = next((s for s in sims if (s.get("trial") or 0) == trial), None)
        if sim is None:
            print("trial%d: 시행 없음 — 건너뜀" % trial)
            continue
        gold = gold_classes(sim)
        cut = find_cut(sim, trial)
        if cut is None:
            print("trial%d: 절단점 미발견(§5b 규칙 무매치) — SKIP\n" % trial)
            continue
        tools = U.tools_of(sim)
        live = build(sim, cut, True)
        free = build(sim, cut, False)   # D_FREE — 우리 사이드카 문장만 제거(단일 델타·리뷰 B1)
        mini = build_min(sim, cut)      # D_CEIL — 상한 기록용
        # 정보-충분성 캐비앳: cut 이전 tool 출력에 등장한 공식명 — B_ONE 이 "이미 있는 것만으로"
        # 답해야 하므로, 여기 gold 명칭이 없으면 B 실패는 능력이 아니라 회수 부족이다.
        hay = "\n".join(str(m.get("content") or "") for m in sim["messages"][:cut]
                        if m.get("role") == "tool")
        seen = name_hits(hay)
        print("== trial%d  cut=%d · 문맥 %d자 · 도구 %d개 · gold=%s"
              % (trial, cut, len(live), len(tools), json.dumps(gold, ensure_ascii=False)))
        print("   cut 이전 tool 출력의 공식명 — checking: %s / savings: %s"
              % (sorted(seen["checking"]) or "없음", sorted(seen["savings"]) or "없음"))
        # v2 발화-전 게이트(설계서 §5b · trial0 의 cut=35 고정도 같은 검증을 통과해야 발화):
        # (1) 정보-맞춤 [[18]] — 절단 문맥에 KB 검색 tool 출력 ≥1 필수.
        kb_n = kb_out_count(sim["messages"][:cut])
        if kb_n < 1:
            print("   KB 검색 tool 출력 0 (정보-맞춤 [[18]] 미충족) — SKIP\n")
            continue
        # (2) 절단 후 문맥(모델이 실제로 보는 세 본문)에 transfer 토큰 잔존 = 자기 무효.
        dirty = [lb for lb, b in (("live", live), ("free", free), ("mini", mini))
                 if TRANSFER_TOK.search(b)]
        if dirty:
            print("   절단 오염 — SKIP (transfer 토큰 잔존: %s)\n" % ",".join(dirty))
            continue

        arms = (("A_LIVE", live, tools, score),
                ("B_ONE", live + "\n" + B_ONE_FB, None, score),
                ("C_SUB", live + "\n" + C_SUB_FB, None, score),
                ("D_FREE", free, tools, score),
                ("D_CEIL", mini, tools, score),
                ("N_NEG", live + "\n" + N_NEG_FB + "\n" + B_ONE_FB, None, score_neg))
        for label, body, tl, fn in arms:
            c = collections.Counter()
            for i in range(n):
                try:
                    r = chat(body, tl, 0.0 if i == 0 else 0.7, 450)
                except Exception as e:
                    # ERR 은 별도 버킷 — PLAIN 위장 금지([[55]] 계기 불신·감사 지적 8).
                    c["ERR(%s)" % type(e).__name__] += 1
                    continue
                c[fn(r, gold)] += 1
            # 전 버킷 인쇄 — 상위 6개 절단은 합산 항(SEARCH+MULTI+TRANSFER ≥5/8)의 저빈도
            # 버킷을 지워 빈도표 오독을 만든다([[08]]·n=8 이라 출력 부담 없음).
            print("  %-7s %s" % (label, c.most_common()))
        print("")

    print("※ 판정표(설계서 §1 리뷰 S1 사전등록·n=8 기준 — 문턱 축자):"
          "\n  A 재현         = A CONV ≤1/8 ∧ (SEARCH+MULTI+TRANSFER) ≥5/8"
          " ⇒ 결손 재현 — 재현돼야 아래 행 판정 가능."
          "\n  B 수렴         = B CONV ≥7/8 ⇒ 능력 있음+지시-위치 민감 — 레버=재제시 시점"
          "(수렴 결정점에 기존 지시 재제시·STEP2 동형). 결정론 금지([[62]]②)."
          "\n  B 실패·C 수렴  = B CONV ≤2/8 ∧ C CONV ≥7/8 ⇒ 빼기 지시 필요 — [[63]] 갱신"
          "(새 기준 축)·레버 문면=C 축자([[03b]])."
          "\n  B·C 다 실패    = 둘 다 CONV ≤2/8 ⇒ 격리서도 수렴 불능=능력 — 그 단계만"
          " 최소 결정론 후보([[62]]③·[[63]] 떠먹이기 금지 심사 선행)."
          "\n  중간           = 어느 팔이든 3~6/8 ⇒ 미판정 — n=16 증량 후 재판정(사후 해석 금지)."
          "\n  D_FREE 만 회복 = D_FREE CONV − A CONV ≥ +4/8 ⇒ 우리 문구가 루프 유발 —"
          " 문구 수술(ⓐ 동형·[[55]]). D_CEIL 은 상한 기록만(귀속 아님·리뷰 B1)."
          "\n  N 무효         = N NOFIT ≤5/8 ⇒ 프로브 과유도=무효 — 문구 재설계 후 재측정."
          "\n  CONV+OFF 는 수렴은 했으나 gold 와 다른 짝 — 능력/전달 판정에는 수렴 여부,"
          "\n  gold 일치는 부가 태그로만 읽는다. tool 출력에 gold 명칭이 없던 trial 의 B 실패는"
          " 회수 부족 캐비앳을 적용한다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
