# -*- coding: utf-8 -*-
r"""x278 — ctx1 상한 진단 프로브 (③ⓑ 잔여 · C448⒞ 후속 · 유료 0 · 엔진 0).

정본: `reports/facet_rft_2026/PROBE_X276_X277_DESIGN_2026_08_12.md` §5c (사전등록 축자 —
단 D_PLACE 행은 실행 전 개정 2026-08-12 반영, 아래 팔 표와 설계서 §5c 개정 줄 참조).

## 왜 (원장 C448⒞)

x277 실측: 같은 유인·같은 스키마 재제시(B_SCHEMA)인데 **ctx2 는 IMPORT 8/8→1/8 로 닫혔고
ctx1 은 8/8 그대로**였다. 차이는 어디서 오는가 — 전달의 잔여 축인가, 진짜 상한([[42]] RLFT
후보)인가.

## 차이 후보 (§5c · 관측된 것)

  ⓐ 문맥 길이 (ctx1 13K vs ctx2 42K)                              → 직접 팔 없음(해석용)
  ⓑ 발견형 디스패치의 **in-chat 실행 증거** 유무 — ctx1 은 직접-도구(verify 체인)만
     실행했고 unlock→call 성공 사례가 없다; ctx2 는 있다          → C_EVID 가 검사
  ⓒ 주입 위치 (현행 = 문맥 끝)                                    → D_PLACE 가 검사

## 표적 (frozen) · 팔 6 (각 n=8 · i==0 temp0 나머지 0.7 · mx=400)

  bank_lever_071_20260812g  task_071  trial(필드)=0  cut=26
  (SIMS 경로·FB 사이드카 경로는 x277 과 동일 규약)

  A_LIVE    build 그대로 (tools=U.tools_of(sim) 는 이 팔에만)      ← x277 재현 기준선
  B_SCHEMA  live + unlock 스키마 전부 축자 재제시 — x277 과 동일 팔(앵커): 일반 앵커
            "Tool unlocked: " 전수 축자 추출(도구명 선별 0)·SCHEMA_PREFIX 동일.
            문맥내=False 면 B·C·D·E 전부 _NEW 접미로 분리(위치 이동 아님)
  C_EVID    B + 설계 §5c 축자 "같은 태그 딴 시행": 표적 TAG(071g) task_071 의 TRIAL 제외
            trial 을 오름차순 기계 스캔 — "Tool unlocked:" 포함 tool 메시지와 그 **다음**
            tool 메시지(unlock 재등장·오류 마커 불검출 = 실행 성공 결과) 첫 쌍을 축자
            각 300자 이식. 채택 쌍은 실행 로그에 축자 인쇄(판정 전 검증 가능)·추출 실패
            시 SKIP print                                           ← 가설 ⓑ
  D_PLACE   스키마 문구를 문맥 끝이 아니라 **마지막 user 발화 직전** 줄에 삽입 — 설계서
            §5c D_PLACE 행 개정(2026-08-12·실행 전) 반영: 원문 "손님 SSN 발화 직후"는
            ctx1 에 SSN 발화가 없고(ctx2 앵커 문면) '직후'=문맥 끝=B 와 동일 배치라
            축자 적용 불능. 절단본에 user 발화가 없으면 팔 정의 불성립으로 SKIP ← 가설 ⓒ
  E_COMBO   B + C_ASK(도구목록-한정 지시·x277 ASK_FB 축자) 합성([[19]])
  N_NEG     x277 과 동일 문맥: 같은 궤적 msgs[:14] + 신원질문 · 검증 강화판 — tool
            메시지에서 **서로 다른** 신원 필드 ≥2종 기계 확인(\bname\b 단독은 JSON
            "name": 키 오탐 가능)·매치 필드/발췌 인쇄·실패 시 SKIP — 부정통제([[57]])

채점은 x277 의 score/score_neg 정규식 축자 사본(우선순위: 도구호출 → EXTERNAL → INCHAT
→ PLAIN · N_NEG 은 EXTERNAL 우선). chat() 예외는 채점하지 않고 ERR 로 별도 집계하며
ERR>0 팔은 판정 보류 — 죽은 계기가 IMPORT 0/8=닫힘으로 읽히는 거짓 양성 차단([[55]]).
gold 는 프롬프트에 넣지 않는다([[23]]).
⛔프로브는 argmax·"정답은 X"를 내지 않는다 — 산출물은 팔별 빈도표다.

[[62]] 4문(§5c): ①결손은 C448⒞ 로 측정됨(ctx1 IMPORT 8/8) ②이 프로브가 격리-성공 여부를
가른다 ③어느 팔도 정답·절차를 대신 쓰지 않는다(증거·스키마 전부 env 축자) ④빈도표만.
[[05]] 3질문(§5c): 도메인-특화 순증 0(축자 이식·위치 변주)·동결 0·행동 수행 0.

실행: T2_PROBE_URL=http://localhost:8141/v1/chat/completions python3 x278_ctx1_diag.py [N]
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

SIMS = os.environ.get("X278_SIMDIR", "/home/woori/scratch/tau2-bench/data/simulations")
FBDIR = "/home/woori/scratch/logs"

# 표적 단일 문맥 (설계서 §5c · frozen g071t0 · cut 직전 = 마지막 user 발화)
TAG, TASK, TRIAL, CUT = "bank_lever_071_20260812g", "task_071", 0, 26

# SCHEMA_TXT 재료의 출처 (기계 추출 — 실패 시 해당 팔 SKIP).
# B_SCHEMA 는 x277 과 동일 팔(앵커)이므로 출처도 x277 SCHEMA_SRC 와 동일하게 유지.
# EVID_TXT 는 별도 상수 없음 — 설계 §5c 축자 "같은 태그 딴 시행" = 표적 TAG 의 TRIAL 제외
# trial 기계 스캔(evid_txt 참조). 070g 이식은 사전등록 이탈이라 제거.
# 앵커는 도구명 없는 일반 마커 — 특정 도구 선별은 gold 유래 지식이라 금지([[23]]·[[63]]).
SCHEMA_SRC = ("bank_lever_070_20260812g", "task_070", 1)
SCHEMA_MARK = "Tool unlocked: "
SCHEMA_PREFIX = ("[system] Reminder - the following tool(s) exist in YOUR OWN tool list "
                 "in this conversation (verbatim from the environment): ")
EVID_PREFIX = ("[system] For reference, earlier in a similar conversation these tool "
               "results occurred (verbatim): ")

ASK_FB = ("\n[system] Answer only from procedures and tools that exist in this "
          "conversation's tool list; if a step is not backed by one of them, say that "
          "it does not exist here instead of describing it.")

NEG_USER = "\n[user] What exactly do you need from me?"

# 계기 정규식 (x277 축자 사본 · 대소문자 무시 · 우선순위 순)
TOOLACT = re.compile(r"unlock|call_discoverable|give_discoverable|verify|log_verification", re.I)
EXTERNAL = re.compile(
    r"\b(?:secure )?(?:web ?)?(?:form|portal)\b|form link|in-app|mobile app|log ?in"
    r"|navigate to|identity verification (?:link|section|page)"
    r"|upload (?:your |the )?(?:id|document|license|selfie)|https?://", re.I)
INCHAT = re.compile(
    r"right here|in this (?:chat|conversation)|I (?:can|will) (?:open|proceed to open)"
    r"|no (?:need|requirement) (?:for|to provide)|open_bank_account|unlock", re.I)
FIELDS = re.compile(r"\b(name|e-?mail|user ?id|date of birth|dob|phone|address)\b", re.I)
# C_EVID '실행 성공' 기계 검증(닫힌 술어·선별 아님): 다음 tool 메시지가 오류 결과면 그 쌍은
# 성공 증거가 아니다 — 우리 층이 '성공'을 단정-날조하지 않도록([[25]]·[[48]] 날조 주체: 우리 층).
ERRMARK = re.compile(r"\berror\b|\bfailed\b", re.I)


def load_sim(tag, task, trial):
    """trial 은 위치 아닌 **필드**로 선택(ax33g_perstep/b4_fail_census 와 동일 규약).
    구형(trial 필드 없는) 파일만 위치 fallback. (x277 축자 사본)"""
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


def build(sim, cut, tag, with_ours=True, inject_at=None, inject_text=None):
    """x277.build 사본 + inject 인자(D_PLACE 전용: 메시지 inject_at 직전에 한 줄 삽입).
    inject 미지정 시 x277 과 바이트 동일한 문맥을 낸다."""
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
        if inject_at is not None and i == inject_at and inject_text:
            out.append(inject_text)
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
    반환 (텍스트, 도구명 목록, note) — 실패 시 (None, [], 사유)로 그 팔 SKIP([[03b]]).
    (x277 사본 — 추출 로직 동일. except 만 SystemExit 포획으로 확대: load_sim 의
    SystemExit 는 Exception 의 하위가 아니라 'SKIP 보장'이 크래시로 깨졌었음.)"""
    try:
        sim = load_sim(*SCHEMA_SRC)
    except (Exception, SystemExit) as e:
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


def evid_txt():
    """C_EVID 재료 — 설계 §5c 축자 "같은 태그 딴 시행": 표적 TAG·TASK 의 TRIAL 제외
    trial 을 오름차순 기계 스캔해, "Tool unlocked:" 포함 tool 메시지와 그 **다음** tool
    메시지를 축자 각 300자 추출(도구명 선별 0·[[63]]). 닫힌 기계 규칙 둘(선별 아님):
      ①다음 tool 메시지가 또 unlock 이면 실행 결과가 아니므로 다음 쌍으로.
      ②다음 tool 메시지가 오류 마커(ERRMARK)에 걸리면 성공 결과가 아니므로 다음 쌍으로
        — '성공' 라벨을 우리 층이 검증 없이 단정하지 않는다([[25]]).
    반환 (텍스트, note, pair) — 채택 pair 는 호출부가 축자 인쇄(판정 전 검증 가능).
    실패 시 (None, 사유+해소, None) → C_EVID SKIP print([[03b]]·[[64]])."""
    try:
        d = json.load(io.open(os.path.join(SIMS, TAG, "results.json"), encoding="utf-8"))
        allsims = [s for s in d["simulations"] if s["task_id"] == TASK]
    except (Exception, SystemExit) as e:
        return None, "results.json 로드 실패: %r" % (e,), None
    if any(s.get("trial") is not None for s in allsims):
        cands = sorted((s for s in allsims
                        if s.get("trial") is not None and s.get("trial") != TRIAL),
                       key=lambda s: s["trial"])
    else:  # 구형(trial 필드 없음)은 위치 규약 — load_sim 과 동일
        cands = [s for k, s in enumerate(allsims) if k != TRIAL]
    for s in cands:
        msgs = s.get("messages") or []
        for i, m in enumerate(msgs):
            c = str(m.get("content") or "")
            if m.get("role") != "tool" or SCHEMA_MARK not in c:
                continue
            nxt = None
            for j in range(i + 1, len(msgs)):
                if msgs[j].get("role") == "tool":
                    nxt = str(msgs[j].get("content") or "")
                    break
            if nxt is None or SCHEMA_MARK in nxt or ERRMARK.search(nxt):
                continue
            pair = (" ".join(c.split())[:300], " ".join(nxt.split())[:300])
            return ("\n" + EVID_PREFIX + pair[0] + " | " + pair[1],
                    "OK(unlock+실행결과 2건 · %s %s trial=%s)"
                    % (TAG, TASK, s.get("trial")), pair)
    return None, ("같은 태그(%s %s) 딴 trial %d개에 unlock→성공-실행결과 tool 쌍 없음 —"
                  " 해소: 다른 출처(예: 070g)를 쓰려면 실행 전에 설계서 §5c 에 출처 변경"
                  " 개정 줄(사유 포함)을 먼저 기입([[64]]·사전등록)"
                  % (TAG, TASK, len(cands))), None


def score(r):
    """우선순위: ①in-chat 도구 실호출 ②외부절차 수입 ③in-chat 발화 ④PLAIN. (x277 축자 사본)"""
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
    기계적으로 정답(ASKS_FIELDS)이 될 수 없어야 부정통제가 과유도를 드러낸다([[57]]).
    (x277 축자 사본)"""
    t = str((r or {}).get("content") or "")
    if EXTERNAL.search(t):
        return "IMPORT"
    if FIELDS.search(t):
        return "ASKS_FIELDS"
    return "PLAIN"


def run_arm(label, body, tools, n, scorer):
    """chat() 예외는 채점하지 않고 ERR 로 별도 집계 — 예전처럼 content="ERR ..." 로 치환해
    채점기에 넣으면 PLAIN 으로 흘러, 서버 다운이 IMPORT 0/8=닫힘(레버 발견)으로 읽히는
    거짓 양성이 된다([[55]] 죽은 계기). ERR>0 팔은 요약 줄에 '판정 보류' 표시."""
    c = collections.Counter()
    errs = collections.Counter()
    for i in range(n):
        try:
            r = chat(body, tools, 0.0 if i == 0 else 0.7, 400)
        except Exception as e:
            errs[type(e).__name__] += 1
            continue
        c[scorer(r)] += 1
    flag = ""
    if errs:
        c["ERR"] = sum(errs.values())
        flag = "   ⚠ERR %s ⇒ 이 팔 판정 보류([[55]])" % dict(errs)
    print("    %-12s IMPORT %d/%d   %s%s" % (label, c["IMPORT"], n, c.most_common(4), flag))
    return c


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 8
    sch, sch_names, sch_note = schema_txt()
    ev, ev_note, ev_pair = evid_txt()
    print("SCHEMA_TXT: %s%s" % (sch_note, "" if not sch else " · %d자 · %s"
                                % (len(sch), ",".join(sch_names))))
    print("EVID_TXT:   %s%s" % (ev_note, "" if not ev else " · %d자" % len(ev)))
    if ev_pair:  # 채택 쌍 축자 인쇄 — '성공' 라벨을 로그에서 판정 전 검증 가능하게
        print("  [C_EVID pair·unlock ] %s" % ev_pair[0])
        print("  [C_EVID pair·실행결과] %s" % ev_pair[1])
    print("")

    sim = load_sim(TAG, TASK, TRIAL)
    msgs = sim["messages"]
    if CUT > len(msgs):
        print("SKIP — messages %d < cut %d" % (len(msgs), CUT))
        return 1
    last_user = " ".join(str(msgs[CUT - 1].get("content") or "").split())
    # D_PLACE 삽입점: 절단 문맥 안 마지막 user 메시지의 인덱스(그 줄 직전에 삽입 — 설계서
    # §5c D_PLACE 행 개정 2026-08-12 참조). user 발화가 없으면 폴백하지 않고 팔 SKIP
    # (폴백=팔 정의의 조용한 무효화 — 무효는 스스로 선고한다·[[64]]).
    user_idxs = [i for i in range(CUT) if msgs[i].get("role") == "user"]
    place_idx = user_idxs[-1] if user_idxs else None
    live = build(sim, CUT, TAG, True)
    tools = U.tools_of(sim)
    # 문맥내 게이트(x277 동일): 추출된 스키마 도구명이 전부 이 절단본에 이미 있어야
    # '위치 이동'. False 면 교차-궤적 신규 주입 — 그 위에 쌓는 C/D/E 도 증거/위치/합성
    # 해석이 성립하지 않으므로 B·C·D·E 전부 _NEW 접미로 분리 집계.
    in_ctx = bool(sch_names) and all(nm in live for nm in sch_names)
    print("ctx1 %s %s trial%d cut=%d · 문맥 %d자 · 스키마 문맥내=%s · D_PLACE 삽입점=msg[%s]"
          % (TAG, TASK, TRIAL, CUT, len(live), in_ctx, place_idx))
    print("  마지막 user: %s" % last_user[:130])

    run_arm("A_LIVE", live, tools, n, score)
    if sch:
        sfx = "" if in_ctx else "_NEW"
        run_arm("B_SCHEMA" + sfx, live + sch, None, n, score)
        if ev:
            run_arm("C_EVID" + sfx, live + sch + ev, None, n, score)
        else:
            print("    C_EVID       SKIP (%s)" % ev_note)
        if place_idx is not None:
            place = build(sim, CUT, TAG, True, inject_at=place_idx,
                          inject_text=sch.lstrip("\n"))
            run_arm("D_PLACE" + sfx, place, None, n, score)
        else:
            print("    D_PLACE      SKIP — 절단본에 user 발화 없음('마지막 user 발화 직전'"
                  " 팔 정의 불성립) · 해소: cut 을 user 발화 포함 지점으로 재설정([[64]])")
        run_arm("E_COMBO" + sfx, live + sch + ASK_FB, None, n, score)
    else:
        print("    B_SCHEMA/C_EVID/D_PLACE/E_COMBO SKIP (%s)" % sch_note)

    # 부정통제 — x277 과 동일 문맥: 같은 궤적 msgs[:14] + 신원질문. '절단본에 env 필드-
    # 요구가 실제 포함'을 주석이 아니라 기계로 확인([[57]]) — 없으면 SKIP(사유+해소·[[64]]).
    # 검증 강화(채점 score_neg 는 불변): \bname\b 단독은 tool 메시지 JSON 의 "name": 키에도
    # 매치되므로(따옴표=비단어문자) 서로 다른 필드 **2종 이상** + 매치 발췌 인쇄를 요구.
    neg_hits = {}
    for m in msgs[:14]:
        if m.get("role") != "tool":
            continue
        c = str(m.get("content") or "")
        for mt in FIELDS.finditer(c):
            k = re.sub(r"[^a-z]", "", mt.group(0).lower())  # e-mail/email 등 표기 정규화
            neg_hits.setdefault(k, " ".join(c.split())[:160])
    if len(neg_hits) >= 2:
        neg = build(sim, 14, TAG, True) + NEG_USER
        print("\nN_NEG (%s msgs[:14] + 신원질문 · 정답=ASKS_FIELDS)" % TAG)
        print("  기계확인: 서로 다른 신원 필드 %d종 — 매치 발췌(진짜 신원 요구인지 로그에서"
              " 확인 가능):" % len(neg_hits))
        for k in sorted(neg_hits):
            print("    %-12s %s" % (k, neg_hits[k]))
        run_arm("N_NEG", neg, None, n, score_neg)
    else:
        print("\nN_NEG SKIP — msgs[:14] tool 메시지에서 서로 다른 신원 필드 2종 이상을 찾지"
              " 못함(매치=%s · 1종 단독은 JSON \"name\": 키 오탐 가능성 배제 불가)"
              " = 부정통제 무효(프로브 무효로 취급). 해소: cut 을 env 필드-요구 메시지가"
              " 포함되는 지점으로 재설정." % (sorted(neg_hits) or "없음"))

    print("\n※ 판정표(설계서 §5c · 사전등록 · n=8):"
          "\n  어느 팔이든 IMPORT ≤1/8   ⇒ 그 축이 닫는다(전달 잔여)."
          "\n  전 팔 ≥4/8                ⇒ ctx1 은 배치로 안 닫히는 상한 후보 — handoff §3.4"
          " 경고대로 헤드라인에서 \"배치로 다 닫힌다\" 금지의 실물 2호로 기록."
          "\n  중간(2~3/8)               ⇒ n=16 증량 후 재판정(사후 해석 금지)."
          "\n  N ASKS_FIELDS ≤5/8        ⇒ 프로브 무효(재설계). N_NEG SKIP(기계확인 실패)도"
          " 동일하게 프로브 무효."
          "\n  ⚠C_EVID 는 정보 순증이 있는 팔(증거 이식) — 닫혀도 \"전달\"이 아니라 **재료"
          " 보강**으로 분류하고, 레버화하려면 그 재료가 라이브에서 기계-조달 가능한지"
          "([[59]]: env 축자 재인쇄만)를 별도 확인한다."
          "\n  ⚠ERR>0 팔은 판정 보류 — 죽은 계기를 레버 발견으로 읽지 않는다([[55]])."
          "\n  ※ B·C·D·E 해석은 문맥내=True 에서만 — _NEW 접미 팔은 교차-궤적 신규 주입"
          " 위라 증거(C)/위치(D)/합성(E) 해석이 성립하지 않음: 별도 집계·판정표 대상 아님.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
