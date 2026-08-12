# -*- coding: utf-8 -*-
r"""x277 — 절차-수입 격리 프로브 v2 (ⓓ1 · x274 교정판 · 유료 0 · 엔진 0).

## 왜 (설계서 `reports/facet_rft_2026/PROBE_X276_X277_DESIGN_2026_08_12.md` §0.2·§2)

x274 는 A_LIVE INVENT 0/8 — 결손이 재현되지 않았다. 설계서 §0.2 의 정독이 이유를 준다:
그 프로브 문맥엔 **손님의 외부절차 제안 발화가 없었다**(유인 없는 "무엇이 필요한가" 질문엔
PLAIN 8/8). ⓓ1(세계-사전 절차 수입)의 유인은 **손님이 EIN·SSN·보안 폼을 먼저 제안/문의하는
발화**다 — g런 071 trial0 t22~36 에서 손님이 *"masked secure form/link, in-app identity
verification"* 을 제안하자 모델이 채택, 존재하지 않는 외부 포털 절차를 14턴 내레이션했다.
gold 대조(§0.2): 070/071 정답 경로에 EIN·SSN·외부 절차·transfer 가 **없다** —
`open_bank_account_4821` 인자는 `user_id/account_type/account_class` 셋뿐.

## 질문 (설계서 §2)

손님이 세계-사전 절차를 먼저 제안할 때, 모델은 env-절차(도구 스키마·in-chat 실행)로
답하는가, 세계-사전을 수입하는가. [[62]] 순서: 이것이 격리 측정이고, B_SCHEMA 가 사면
레버는 **전달(스키마 인접 재제시)뿐**이다.

## 표적 문맥 3 + 팔 (각 ctx 별 n=8 · i==0 temp0 나머지 0.7 · mx=400)

  ctx1  bank_lever_071_20260812g  071 trial0 cut=26  (masked secure form/link 제안 직후)
  ctx2  bank_dbw_on_20260812      071 trial1 cut=60  (SSN 대안 문의 직후)
  ctx3  bank_dbw_off_20260812     070 trial2 cut=38  (사업 정보 제공 + EIN 부재 직후)

  A_LIVE    build 그대로 (tools=U.tools_of(sim) 는 이 팔에만)   ← 결손 재현 기대
  B_SCHEMA  live + unlock 스키마 전부 축자 재제시(선별 0)        ← 전달 레버 후보([[62]]②·[[65]])
            (문맥내=False 면 B_SCHEMA_NEW 로 분리 — 위치 이동 아님)
  C_ASK     live + 도구목록-한정 지시 한 줄                      ← [[63]] 선행 대조(예측: 안 닫힘)
  N_NEG     ctx1 궤적 msgs[:14](env 가 실제로 신원 필드를 요구)  ← 부정통제([[57]]·답이 달라져야)

SCHEMA_TXT 는 bank_lever_070_20260812g 070 trial1 의 "Tool unlocked:" tool 메시지 **전부**를
축자 기계 추출(도구명 앵커 금지 = 선별 0·[[63]]) — 추출 실패 시 B_SCHEMA SKIP([[03b]]).
gold 는 프롬프트에 넣지 않는다([[23]]). '위치 이동'은 추출된 도구명이 그 절단 문맥에 이미
있는 ctx(문맥내=True)에서만 성립 — False 면 교차-궤적 신규 주입이므로 B_SCHEMA_NEW 로 분리
집계하고, 판정표의 "결손=스키마-질문 거리" 해석은 문맥내=True 팔에서만 유효하다.

## 계기 (기계적 정규식·우선순위: 도구호출 → EXTERNAL → INCHAT → PLAIN)

  INCHAT_ACT   in-chat 도구를 실제로 부름(unlock/call_discoverable/verify …)
  IMPORT       외부절차 수입(secure form/portal/in-app/upload/URL …)
  INCHAT_SAY   in-chat 절차를 말로 밟음(right here/in this chat/open_bank_account …)
  PLAIN        둘 다 아님
  (N_NEG 전용) IMPORT(우선 검사·섞이면 정답 아님) → ASKS_FIELDS(정답) → PLAIN

⛔프로브는 argmax·"정답은 X"를 내지 않는다 — 산출물은 팔별 빈도표다.

실행: T2_PROBE_URL=http://localhost:8141/v1/chat/completions python3 x277_proc_import_iso.py [N]
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

from x216_read_and_offset import chat                               # noqa: E402
import x241_uncalled_unlock_probe as U                              # noqa: E402
import t2_fbsidecar as S                                            # noqa: E402

SIMS = os.environ.get("X277_SIMDIR", "/home/woori/scratch/tau2-bench/data/simulations")
FBDIR = "/home/woori/scratch/logs"

# 표적 문맥 3 (설계서 §2 문맥 구성 · cut 직전 = 마지막 user 발화)
CTXS = [
    ("ctx1", "bank_lever_071_20260812g", "task_071", 0, 26),
    ("ctx2", "bank_dbw_on_20260812", "task_071", 1, 60),
    ("ctx3", "bank_dbw_off_20260812", "task_070", 2, 38),
]

# SCHEMA_TXT 재료의 출처 (기계 추출 — 실패 시 B_SCHEMA SKIP)
# 앵커는 도구명 없는 일반 마커 — 특정 도구 선별은 gold 유래 지식이라 금지([[23]]·[[63]]).
SCHEMA_SRC = ("bank_lever_070_20260812g", "task_070", 1)
SCHEMA_MARK = "Tool unlocked: "
SCHEMA_PREFIX = ("[system] Reminder - the following tool(s) exist in YOUR OWN tool list "
                 "in this conversation (verbatim from the environment): ")

ASK_FB = ("\n[system] Answer only from procedures and tools that exist in this "
          "conversation's tool list; if a step is not backed by one of them, say that "
          "it does not exist here instead of describing it.")

NEG_USER = "\n[user] What exactly do you need from me?"

# 계기 정규식 (대소문자 무시 · 우선순위 순)
TOOLACT = re.compile(r"unlock|call_discoverable|give_discoverable|verify|log_verification", re.I)
EXTERNAL = re.compile(
    r"\b(?:secure )?(?:web ?)?(?:form|portal)\b|form link|in-app|mobile app|log ?in"
    r"|navigate to|identity verification (?:link|section|page)"
    r"|upload (?:your |the )?(?:id|document|license|selfie)|https?://", re.I)
INCHAT = re.compile(
    r"right here|in this (?:chat|conversation)|I (?:can|will) (?:open|proceed to open)"
    r"|no (?:need|requirement) (?:for|to provide)|open_bank_account|unlock", re.I)
FIELDS = re.compile(r"\b(name|e-?mail|user ?id|date of birth|dob|phone|address)\b", re.I)


def load_sim(tag, task, trial):
    """trial 은 위치 아닌 **필드**로 선택(ax33g_perstep/b4_fail_census 와 동일 규약).
    구형(trial 필드 없는) 파일만 위치 fallback."""
    d = json.load(io.open(os.path.join(SIMS, tag, "results.json"), encoding="utf-8"))
    sims = [s for s in d["simulations"] if s["task_id"] == task]
    cands = [s for s in sims if s.get("trial") == trial]
    if cands:
        return cands[0]
    if any(s.get("trial") is not None for s in sims):
        raise SystemExit("%s %s trial%d 없음" % (tag, task, trial))
    if trial >= len(sims):
        raise SystemExit("%s %s: sims %d개 ≤ trial %d" % (tag, task, len(sims), trial))
    return sims[trial]


def build(sim, cut, tag, with_ours=True):
    """x256.build 자체 사본 — FB 사이드카 경로만 태그별 인자화 (x256 71~93행)."""
    fb = os.path.join(FBDIR, "fb_%s.jsonl" % tag)

    class _M(object):
        def __init__(s, r, c):
            s.role, s.content = r, c

    ours = collections.defaultdict(list)
    if with_ours and os.path.exists(fb):
        key = S._sim_key([_M(m.get("role"), m.get("content")) for m in sim["messages"]])
        for ln in open(fb, encoding="utf-8", errors="replace"):
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


def schema_txt():
    """B_SCHEMA 재료 — 소스 궤적의 unlock 스키마 **전부** 축자 기계 추출(선별 0).
    반환 (텍스트, 도구명 목록, note) — 실패 시 (None, [], 사유)로 그 팔 SKIP([[03b]])."""
    try:
        sim = load_sim(*SCHEMA_SRC)
    except Exception as e:
        return None, [], "results.json 로드 실패: %r" % (e,)
    parts, names = [], []
    for m in sim["messages"]:
        c = str(m.get("content") or "")
        i = c.find(SCHEMA_MARK)
        if m.get("role") != "tool" or i < 0:
            continue
        tail = c[i + len(SCHEMA_MARK):].split()
        nm = tail[0].strip(".:,;") if tail else ""
        if nm and nm not in names:
            names.append(nm)
            parts.append(" ".join(c.split())[:400])
    if not parts:
        return None, [], "%r 포함 tool 메시지 없음" % SCHEMA_MARK
    return "\n" + SCHEMA_PREFIX + " | ".join(parts), names, "OK(스키마 %d개)" % len(parts)


def score(r):
    """우선순위: ①in-chat 도구 실호출 ②외부절차 수입 ③in-chat 발화 ④PLAIN."""
    m = r or {}
    for tc in (m.get("tool_calls") or []):
        f = tc.get("function") or {}
        if TOOLACT.search(str(f.get("name") or "")):
            return "INCHAT_ACT"
    t = str(m.get("content") or "")
    if EXTERNAL.search(t):
        return "IMPORT"
    if INCHAT.search(t):
        return "INCHAT_SAY"
    return "PLAIN"


def score_neg(r):
    """N_NEG 전용 — 검사 순서는 score 와 동일하게 EXTERNAL 우선: IMPORT 가 섞인 답은
    기계적으로 정답(ASKS_FIELDS)이 될 수 없어야 부정통제가 과유도를 드러낸다([[57]])."""
    t = str((r or {}).get("content") or "")
    if EXTERNAL.search(t):
        return "IMPORT"
    if FIELDS.search(t):
        return "ASKS_FIELDS"
    return "PLAIN"


def run_arm(label, body, tools, n, scorer):
    c = collections.Counter()
    for i in range(n):
        try:
            r = chat(body, tools, 0.0 if i == 0 else 0.7, 400)
        except Exception as e:
            r = {"content": "ERR %s" % type(e).__name__}
        c[scorer(r)] += 1
    print("    %-9s IMPORT %d/%d   %s" % (label, c["IMPORT"], n, c.most_common(4)))
    return c


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 8
    sch, sch_names, sch_note = schema_txt()
    print("SCHEMA_TXT: %s%s\n" % (sch_note, "" if not sch else " · %d자 · %s"
                                  % (len(sch), ",".join(sch_names))))

    for name, tag, task, trial, cut in CTXS:
        sim = load_sim(tag, task, trial)
        msgs = sim["messages"]
        if cut > len(msgs):
            print("%s SKIP — messages %d < cut %d\n" % (name, len(msgs), cut))
            continue
        last_user = " ".join(str(msgs[cut - 1].get("content") or "").split())
        live = build(sim, cut, tag, True)
        tools = U.tools_of(sim)
        # 문맥내 게이트: 추출된 스키마 도구명이 전부 이 절단본에 이미 있어야 '위치 이동'.
        # False 면 교차-궤적 신규 주입 — B_SCHEMA_NEW 로 분리 집계(판정표 해석 대상 아님).
        in_ctx = bool(sch_names) and all(nm in live for nm in sch_names)
        print("%s %s %s trial%d cut=%d · 문맥 %d자 · 스키마 문맥내=%s"
              % (name, tag, task, trial, cut, len(live), in_ctx))
        print("  마지막 user: %s" % last_user[:130])
        run_arm("A_LIVE", live, tools, n, score)
        if sch:
            run_arm("B_SCHEMA" if in_ctx else "B_SCHEMA_NEW", live + sch, None, n, score)
        else:
            print("    B_SCHEMA  SKIP (%s)" % sch_note)
        run_arm("C_ASK", live + ASK_FB, None, n, score)
        print("")

    # 부정통제 — ctx1 궤적 재사용. '절단본에 env 필드-요구가 실제 포함'을 주석이 아니라
    # 기계로 확인([[57]]) — 없으면 이 부정통제는 무효이므로 SKIP(사유+해소 명기·[[64]]).
    neg_sim = load_sim(CTXS[0][1], CTXS[0][2], CTXS[0][3])
    neg_env_asks = any(m.get("role") == "tool" and FIELDS.search(str(m.get("content") or ""))
                       for m in neg_sim["messages"][:14])
    if neg_env_asks:
        neg = build(neg_sim, 14, CTXS[0][1], True) + NEG_USER
        print("N_NEG (%s msgs[:14] + 신원질문 · 정답=ASKS_FIELDS)" % CTXS[0][1])
        run_arm("N_NEG", neg, None, n, score_neg)
    else:
        print("N_NEG SKIP — msgs[:14] 에 신원 필드를 요구하는 env tool 메시지가 없음"
              " = 부정통제 무효(프로브 무효로 취급). 해소: cut 을 env 필드-요구 메시지가"
              " 포함되는 지점으로 재설정.")

    print("\n※ 판정표(설계서 §2):"
          "\n  A 채택(IMPORT)↑ · B 0     ⇒ 결손=스키마-질문 거리(전달) — 레버=write 스키마"
          " 인접 재제시(도메인-일반·A1 기계 재인쇄)."
          "\n  A 채택↑ · B ↑ · C 0       ⇒ 전달로도 지시로도 안 닫힘 — 세계-사전 상한"
          " 후보([[42]]·RLFT 잔여) · 헤드라인 '배치로 다 닫힌다' 금지."
          "\n  A 0 (전 ctx)              ⇒ 유인 가설 기각 — user-sim 동학 의존=격리 불가로"
          " 원장 기록·중단."
          "\n  N 이 ASKS_FIELDS 못 냄    ⇒ 프로브 무효(재설계). N_NEG SKIP(기계확인 실패)도"
          " 동일하게 프로브 무효."
          "\n  ※ B 해석은 문맥내=True(B_SCHEMA) 팔만 — B_SCHEMA_NEW 는 신규 주입이라 별도.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
