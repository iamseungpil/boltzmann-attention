# -*- coding: utf-8 -*-
r"""x281 — 완료-대조표 격리 프로브 (유료 0 · 엔진 0 · x280 후속).

## 왜 (x280 실측)

x280 은 "손님이 요청한 개설 중 아직 안 끝난 것"을 (message_index, quote) 로 열거시켰다.
j런 071 t1 에서:

  cut=31 (개설 **0건** 시점)  → REMAIN_CITED(2) **8/8** — 요구 2건을 유효 인용으로 정확히 열거
  cut=41 (checking 만 열린 시점 · 직전 발화가 축자로 "이제 business savings 를 열겠다")
                              → REMAIN_NONE **8/8** — **틀렸다**(savings 미완인데 "없다")

즉 상태가 **아무것도 안 열림**일 땐 요청 집합을 그대로 읽어내는데, **일부만 열림**이 되는
순간 답이 무너진다. 가설: 결손은 열거가 아니라 **요청 집합에서 이미 완료된 것을 빼는
대조**다 — 모델은 스스로 배제(빼기)를 못한다([[63]] 계열).

## 이 프로브가 재는 것 ([[62]]②)

**대조를 전달하면 회복하는가.** 완료분(이미 열린 계좌)을 문맥에서 **기계로 추출해 표로
표면화**하고, 요청 집합은 모델이 그대로 읽게 둔다. 회복하면 레버 = 전달(부하 축소)이고,
회복 안 하면 능력 축([[13]] 학습·스케일)이다.

## 문맥 (동결 · 절단은 전부 open_points() 기계 규칙)

  ctx_P41  (bank_lever4_a_20260812j · task_071 · trial 1) cut = **두 번째 성공 open 인덱스**
           (등록 좌표 41) — savings 미완 · **정답 = 남아 있다**
  ctx_D43  같은 sim · cut = 두 번째 성공 open **이후 첫 실패 open 시도 인덱스**(등록 43)
           — 둘 다 완료 · **정답 = 없다**(과유도 검사)
  ctx_070  (같은 태그 · task_070 · trial 0) cut = **마지막 성공 open 인덱스**(등록 70)
           — 요청 1건 완료 · **정답 = 없다**(과유도 검사)

기계 계산값이 등록 좌표와 다르면 그 ctx 는 "등록 좌표 불일치 — 판정 보류"를 인쇄한다
(x279 규약). 규칙이 절단을 못 만들면 SKIP(사유 + 해소 · [[64]]).

## 팔 2 (각 ctx · n=8 · i==0 temp0 나머지 0.7 · mx=400 · tools=U.tools_of)

  A_LIVE   x280 의 REMAIN_FB 축자 그대로                       ← 재현 기준선
  B_TABLE  같은 문맥 + **기계 추출 대조표** 한 블록을 REMAIN_FB **앞**에 삽입
           ← 팔 간 단일 델타 = 표 블록뿐

## 대조표 (기계 추출 · [[59]] 준수 — 도메인 해석 0)

절단 문맥에서 **성공한 open 디스패치의 결과 메시지**(open_points 성공분의 바로 다음 tool
메시지)만 골라 그 content 를 **공백 정규화해 200자까지** 축자로 뽑아 한 줄로 잇는다.
행을 open 디스패치에 결속하는 이유: 'opened successfully' substring 만으로는 개설이 아닌
도구 결과(케이스·티켓·분쟁 등)도 행이 되어, 헤더의 사실 진술이 거짓이 된다([[25]] —
우리 도구 출력은 100% 참이어야 한다).

  "[system] Accounts already opened in this conversation (verbatim from the tool results
   above): - <축자 발췌> ; - <축자 발췌>"

행 접두에 메시지 인덱스를 넣지 않는다 — 계약(REMAIN_FB)의 message_index 는 **손님(user)
메시지** 좌표인데 tool 메시지 번호를 노출하면 모델이 그것을 인용해 k=0(FABRICATED)이 되고,
이는 좌표 혼선인데도 능력 결손으로 오귀속된다.

엔진이 하는 일은 substring 검사·공백 정규화·자르기뿐이다 — 계좌 종류를 파싱하지도,
어느 요청과 짝인지 판정하지도 않는다(대조 자체는 여전히 모델 몫). 추출이 0건이면 그 ctx 의
B_TABLE 은 **SKIP** 인쇄 — 표를 지어내지 않는다([[03b]]).

## 채점 (x280 make_scorer 사본 그대로 · 의미 판정 0)

  ①JSON 파스·키 ②message_index 범위 ③그 메시지 role=="user" ④quote 가 **그 메시지**에
  축자 실재(공백 정규화) ⑤len(정규화 quote) ≥ 15 → 유효 항목 수 k.
  버킷: REMAIN_NONE / REMAIN_CITED(k) / REMAIN_FABRICATED / PLAIN (+ERR=판정 보류·[[55]])

## 판정 문턱 (사전등록 · footer 로도 인쇄)

  재현:    ctx_P41 A_LIVE  REMAIN_NONE  ≥6/8 (=0.75·n 증량 시 비율 고정) ⇒ x280 결손 재현
  닫힘:    ctx_P41 B_TABLE **REMAIN_CITED(1)** ≥6/8(=0.75) ⇒ **대조표가 닫는다** — 레버 =
           전달(요청↔완료 대조 표면화 · 엔진은 문자열 추출만). **CITED(k≥2)** 는
           **미-차감(no-subtraction) 버킷**으로 별도 집계하며 닫힘에 산입하지 않는다
           (x280 cut=31 의 정답 상태가 CITED(2) — 빼기를 못 해도 같은 칸에 들어가면
           표가 대조를 못 닫아도 '닫는다'로 읽힌다).
  과유도:  ctx_D43 또는 ctx_070 에서 (B_TABLE CITED − A_LIVE CITED) ≥2/8(=0.25) ⇒ **무효**
           (표가 없는 잔여를 만들어냄). A_LIVE CITED 자체가 ≥2/8 이면 그 ctx 는
           부정통제 불량으로 **판정 보류**([[57]] — 기저 발생률을 표 탓으로 돌리지 않는다)
  안 닫힘: ctx_P41 B_TABLE REMAIN_NONE  ≥6/8(=0.75) ⇒ 전달로 안 닫힘 = 능력 축([[13]])
  중간 2~3/8 ⇒ n=16 증량(문턱은 비율 사전등록이라 12/16 로 환산) · ERR>0 팔은 판정 보류

[[62]] 4문: ①결손 = x280 실측(cut=31 CITED(2) 8/8 ↔ cut=41 NONE 8/8) ②격리에서 표로
회복하면 레버는 전달뿐 ③엔진이 대신 하는 판단 0 — 무엇이 남았는지·무엇과 짝인지는 전부
모델이 답한다(엔진=substring 추출) ④argmax·"정답은 X" 0 — 산출물은 팔별 빈도표다.
[[05]] 3질문: 도메인-특화 0(추출 술어는 도구 결과 문자열 하나·계좌 어휘 0) · 모델/scaffold
동결 · 행동 수행 0(읽기만).
[[03b]]/[[23]]: gold 미주입 — 정답 계좌명도, task gold 도, reward 도 프롬프트에 넣지 않는다
(대조표는 **모델 자신의 궤적**에 실재하는 도구 결과 축자다).

덤프: X281_DUMP=<경로.jsonl> 이면 (ctx, arm, i, content, bucket) 을 JSONL 로 남긴다
(채점 무영향 · FABRICATED 의 실제 원인이 좌표 치환인지 날조인지를 궤적으로 확정 · [[08]]).

실행: T2_PROBE_URL=http://localhost:8141/v1/chat/completions python3 x281_completion_table.py [N]
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

SIMS = os.environ.get("X281_SIMDIR", "/home/woori/scratch/tau2-bench/data/simulations")
TAG = os.environ.get("X281_TAG", "bank_lever4_a_20260812j")
DUMP = os.environ.get("X281_DUMP", "")      # 응답 원문 JSONL 덤프(채점 무영향 · [[08]])

# 절단 앵커·추출 마커 — 실험-코드 상수(절단·표 추출에만 사용). 프롬프트에는 이 상수 자체가
# 아니라 궤적에 실재하는 축자 문자열만 들어간다.
WRITE_MARK = "open_bank_account"
OK_MARK = "opened successfully"

# 문맥 등록 (동결) — (이름, 태그, task, trial, 절단 규칙 키, 등록 좌표, 주석).
# 등록 좌표는 **대조용**이다: 기계 계산값과 다르면 판정 보류를 인쇄하고(x279 규약)
# 프롬프트에는 어떤 좌표도 주입하지 않는다.
CTX_SPEC = (
    ("ctx_P41", TAG, "task_071", 1, "ok2", 41,
     "두 번째 성공 open 직전(checking 만 열림) — savings 미완 · 정답 = 남아 있다"),
    ("ctx_D43", TAG, "task_071", 1, "fail_after_ok2", 43,
     "두 번째 성공 open 이후 첫 실패 시도 직전 — 둘 다 완료 · 정답 = 없다(과유도 검사)"),
    ("ctx_070", TAG, "task_070", 0, "ok_last", 70,
     "마지막 성공 open 직전 — 요청 1건 완료 · 정답 = 없다(과유도 검사)"),
)

# x280 REMAIN_FB 축자 사본 — 팔 간 단일 델타를 표 블록 하나로 묶기 위해 한 글자도 바꾸지
# 않는다([[03b]] · 재현 기준선).
REMAIN_FB = (
    "\n[system] Task: from the transcript above, output exactly one line - the JSON "
    "{\"remaining\": [{\"account_type\": \"<type>\", \"message_index\": <index of the customer "
    "message asking for it>, \"quote\": \"<verbatim sentence from that message>\"}]}. "
    "Include only account openings the customer asked for that have NOT already been "
    "completed successfully in this transcript. If every account the customer asked for "
    "has already been opened, reply with exactly the single word: none"
)
TABLE_HEAD = ("\n[system] Accounts already opened in this conversation (verbatim from the "
              "tool results above): ")
REM_START = re.compile(r"\{\s*[\"']remaining[\"']")
NONE_WORD = re.compile(r"^\s*[`\"']?none[`\"'.]?\s*$", re.I)


def _norm(s):
    """x280 사본 — 채점용(소문자화 포함)."""
    return " ".join(str(s or "").split()).lower()


def _ws(s):
    """대조표 전용 공백 정규화 — **소문자화 없음**(축자 발췌이므로 원문 대소문자 보존)."""
    return " ".join(str(s or "").split())


def _tc_args(tc):
    """x280 사본."""
    a = tc.get("arguments")
    return a if isinstance(a, str) else json.dumps(a or {}, ensure_ascii=False)


def load_sim(tag, task, trial):
    """x280.load_sim 사본 + (tag, task, trial) 파라미터화 — 세 ctx 가 두 시행(071 t1 ·
    070 t0)에 걸쳐 있어 모듈 전역 고정으로는 못 돈다. trial 은 위치가 아니라 **필드**."""
    d = json.load(io.open(os.path.join(SIMS, tag, "results.json"), encoding="utf-8"))
    sims = [s for s in d["simulations"] if s["task_id"] == task]
    cands = [s for s in sims if s.get("trial") == trial]
    if cands:
        return cands[0]
    raise SystemExit("trial=%s 시행 없음 (%s %s)" % (trial, tag, task))


def open_points(msgs):
    """x280 사본 — (인덱스, 성공여부) · open 디스패치 전수. 기계 판정뿐(도구명·결과 문자열)."""
    out = []
    for i, m in enumerate(msgs):
        if m.get("role") != "assistant":
            continue
        for tc in (m.get("tool_calls") or []):
            if WRITE_MARK in _tc_args(tc) and not str(tc.get("name") or "").startswith("unlock"):
                nxt = str(msgs[i + 1].get("content") or "") if i + 1 < len(msgs) else ""
                out.append((i, OK_MARK in nxt))
                break
    return out


def pick_cut(kind, pts):
    """절단 규칙(전부 open_points 기계 계산) → (cut, 실패사유). 실패 시 사유는 해소까지
    적는다([[64]])."""
    ok = [i for i, good in pts if good]
    bad = [i for i, good in pts if not good]
    if kind == "ok2":
        if len(ok) < 2:
            return None, ("성공 open 이 %d건 — 두 번째 성공 open 이 없다 · 해소: 부분-완료"
                          " 국면이 실재하는 시행으로 문맥 등록을 개정(사유 기입) 후 재실행"
                          % len(ok))
        return ok[1], None
    if kind == "fail_after_ok2":
        if len(ok) < 2:
            return None, ("성공 open 이 %d건 — 기준점(두 번째 성공)이 없다 · 해소: 위와 동일"
                          % len(ok))
        later = [i for i in bad if i > ok[1]]
        if not later:
            return None, ("두 번째 성공 open(msg %d) 이후 실패 open 시도가 없다 · 해소:"
                          " 중복-시도가 실재하는 시행으로 문맥 등록을 개정 후 재실행" % ok[1])
        return later[0], None
    if kind == "ok_last":
        if not ok:
            return None, ("성공 open 이 0건 · 해소: 개설이 성공한 시행으로 문맥 등록을"
                          " 개정 후 재실행")
        return ok[-1], None
    return None, "미등록 절단 규칙 %r · 해소: CTX_SPEC 의 규칙 키를 확인" % (kind,)


def build(sim, cut):
    """x280 사본 — [i] 인덱스 접두(계약의 message_index 좌표를 모델이 봐야 한다)."""
    out = []
    for i, m in enumerate(sim["messages"][:cut]):
        r = m.get("role")
        c = " ".join(str(m.get("content") or "").split())
        tcs = [tc.get("name") for tc in (m.get("tool_calls") or [])]
        if tcs:
            out.append("[%d] [%s calls] %s" % (i, r, ", ".join(x for x in tcs if x)))
        if c:
            out.append("[%d] [%s] %s" % (i, r, c[:700]))
    return "\n".join(out)


def opened_rows(msgs, cut, ok_res):
    """대조표 원자재 — **성공한 open 디스패치의 결과 메시지**에서 축자 발췌.

    술어는 넷뿐이다: 인덱스 ∈ ok_res(성공 open 디스패치의 다음 메시지) · role=="tool" ·
    OK_MARK substring · 공백 정규화 후 200자 절단. 계좌 종류를 파싱하지 않고, 어느 요청과
    짝인지도 판정하지 않는다([[59]] — 대조는 여전히 모델 몫). open 결속이 없으면 개설이
    아닌 도구 결과가 행이 되어 헤더 문장이 거짓이 된다([[25]]).
    행 접두에 메시지 인덱스를 넣지 않는다(계약의 message_index 좌표와 충돌 · 오귀속 방지).
    반환 (rows, 스킵진단): rows 가 비면 진단에 '어떤 role 이 OK_MARK 를 담고 있었는지'를
    넣어 해소를 지목한다([[64]])."""
    rows = []
    seen_roles = collections.Counter()
    for i, m in enumerate(msgs[:cut]):
        c = _ws(m.get("content") or "")
        if OK_MARK not in c:
            continue
        seen_roles[str(m.get("role"))] += 1
        if m.get("role") != "tool" or i not in ok_res:
            continue
        rows.append("- %s" % c[:200])
    if rows:
        return rows, None
    why = ("절단 문맥에 role=='tool' + %r 인 메시지가 0건" % OK_MARK)
    if seen_roles:
        why += " (OK_MARK 를 담은 role 분포: %s — 추출 규칙의 role 조건을 궤적 표기에" \
               " 맞춰 개정 후 재실행)" % dict(seen_roles)
    else:
        why += " · 해소: 이 절단 이전에 성공 개설이 실재하는 좌표로 문맥 등록을 개정 후 재실행"
    return [], why


def make_scorer(msgs, cut):
    """x280 사본 그대로 — 검증 ①~⑤ · 유효 항목 수 k 만 센다(요청의 타당성은 판정 0)."""
    def scorer(r):
        text = str((r or {}).get("content") or "")
        tail = text.strip().splitlines()[-1] if text.strip() else ""
        if NONE_WORD.match(text) or NONE_WORD.match(tail):
            return "REMAIN_NONE"
        obj = None
        for m in REM_START.finditer(text):
            try:
                obj, _ = json.JSONDecoder().raw_decode(text[m.start():])
                break
            except ValueError:
                continue
        if obj is None:
            return "REMAIN_NONE" if (re.search(r"\bnone\b", text, re.I)
                                     and len(text) < 80) else "PLAIN"
        items = obj.get("remaining") if isinstance(obj, dict) else None
        if not items:
            return "REMAIN_NONE"
        k = 0
        for it in items:
            if not isinstance(it, dict):
                continue
            idx, q = it.get("message_index"), it.get("quote")
            if (not isinstance(idx, int) or isinstance(idx, bool)
                    or not isinstance(q, str) or not (0 <= idx < cut)):
                continue
            if msgs[idx].get("role") != "user":
                continue
            if _norm(q) not in _norm(msgs[idx].get("content") or ""):
                continue
            if len(_norm(q)) < 15:
                continue
            k += 1
        return "REMAIN_CITED(%d)" % k if k else "REMAIN_FABRICATED"
    return scorer


def _dump(rec):
    """X281_DUMP 가 있으면 한 줄 JSONL 추가 — 채점·프롬프트 무영향(사후 포렌식용·[[08]])."""
    if not DUMP:
        return
    try:
        with io.open(DUMP, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass


def run_arm(label, body, tools, n, scorer, ctx=""):
    """x280 사본 — chat() 예외는 채점하지 않고 ERR 로 별도 집계(죽은 계기가 '닫힘'으로
    읽히는 거짓 양성 차단·[[55]])."""
    c = collections.Counter()
    errs = collections.Counter()
    for i in range(n):
        try:
            r = chat(body, tools, 0.0 if i == 0 else 0.7, 400)
        except Exception as e:
            errs[type(e).__name__] += 1
            _dump({"ctx": ctx, "arm": label, "i": i, "content": None,
                   "bucket": "ERR:%s" % type(e).__name__})
            continue
        b = scorer(r)
        _dump({"ctx": ctx, "arm": label, "i": i,
               "content": str((r or {}).get("content") or ""), "bucket": b})
        c[b] += 1
    flag = "   ⚠ERR %s ⇒ 판정 보류([[55]])" % dict(errs) if errs else ""
    print("    %-9s %s%s" % (label, dict(c), flag))
    return c


def cited_k(c, k):
    """CITED(k) 정확 일치 개수 — 닫힘 문턱은 **CITED(1)** 만 센다(빼기 성공)."""
    return c["REMAIN_CITED(%d)" % k]


def summarize(label, c, n, hold_cut=False):
    """사전등록 문턱이 쓰는 축(NONE·CITED(1)·미차감 CITED(k≥2))을 한 줄로 — argmax·결론
    인쇄 아님(빈도표). k별 내역을 병기해 '빼기를 못 한 CITED(2)'가 닫힘으로 읽히는 것을
    막는다."""
    cited = sum(v for k, v in c.items() if k.startswith("REMAIN_CITED"))
    nosub = cited - cited_k(c, 1)
    det = " · ".join("CITED(%s)×%d" % (k[13:-1], v)
                     for k, v in sorted(c.items()) if k.startswith("REMAIN_CITED"))
    judged = sum(c.values())
    hold = "" if judged >= n else "   ⚠판정 보류(미채점 %d = ERR)" % (n - judged)
    if hold_cut:
        hold += "   ⚠등록 좌표 불일치 — 이 행 판정 보류 · 해소: 문맥 등록·궤적 대조 후 재실행"
    print("      └ %-9s NONE %d/%d · CITED(1) %d/%d · 미차감 CITED(k≥2) %d/%d ·"
          " CITED계 %d · FABRICATED %d · PLAIN %d%s%s"
          % (label, c["REMAIN_NONE"], n, cited_k(c, 1), n, nosub, n, cited,
             c["REMAIN_FABRICATED"], c["PLAIN"],
             ("   [%s]" % det if det else ""), hold))


def run_ctx(name, tag, task, trial, kind, expect_cut, note, n):
    try:
        sim = load_sim(tag, task, trial)
    except (Exception, SystemExit) as e:
        print("%s SKIP — 로드 실패: %r · 해소: 태그/task/trial 확인 후 재실행([[64]])\n"
              % (name, e))
        return
    msgs = sim["messages"]
    pts = open_points(msgs)
    cut, why = pick_cut(kind, pts)
    print("%s %s %s trial=%s (reward=%s)  규칙=%s · %s"
          % (name, tag, task, trial, (sim.get("reward_info") or {}).get("reward"),
             kind, note))
    print("  open 디스패치: %s" % (pts,))
    if cut is None:
        print("  SKIP — %s\n" % why)
        return
    if cut == 0:
        print("  SKIP — cut=0(절단 문맥 0 메시지 = 측정 대상 부재) · 해소: 문맥이 실재하는"
              " 시행으로 등록 개정 후 재실행([[64]])\n")
        return
    hold_cut = (cut != expect_cut)
    if hold_cut:
        print("  ⚠ 등록 좌표 불일치(등록 cut=%d, 기계 실측 %d) — **이 ctx 판정 보류**:"
              " 문맥 등록·궤적 대조 후 재실행([[64]]·x279 규약)" % (expect_cut, cut))
    live = build(sim, cut)
    tools = U.tools_of(sim)
    scorer = make_scorer(msgs, cut)
    print("  cut=%d · 문맥 %d자" % (cut, len(live)))
    print("  직전 msg[%d] %s: %s"
          % (cut - 1, msgs[cut - 1].get("role"),
             _ws(msgs[cut - 1].get("content") or "")[:130]))
    ca = run_arm("A_LIVE", live + REMAIN_FB, tools, n, scorer, name)
    summarize("A_LIVE", ca, n, hold_cut)
    ok_res = {i + 1 for i, good in pts if good}
    rows, skip_why = opened_rows(msgs, cut, ok_res)
    if not rows:
        print("    B_TABLE  SKIP — %s ([[03b]] 표를 지어내지 않는다)\n" % skip_why)
        return
    table = TABLE_HEAD + " ; ".join(rows)
    print("  대조표(%d행·%d자): %s" % (len(rows), len(table), _ws(table)[:220]))
    cb = run_arm("B_TABLE", live + table + REMAIN_FB, tools, n, scorer, name)
    summarize("B_TABLE", cb, n, hold_cut)
    a_cit = sum(v for k, v in ca.items() if k.startswith("REMAIN_CITED"))
    b_cit = sum(v for k, v in cb.items() if k.startswith("REMAIN_CITED"))
    lo = int(0.25 * n + 0.5)
    print("      └ 과유도 Δ(B−A) CITED = %d/%d (문턱 ≥%d/%d = 0.25 사전등록)%s"
          % (b_cit - a_cit, n, lo, n,
             "   ⚠A_LIVE CITED %d/%d ≥%d — 부정통제 불량으로 판정 보류([[57]])"
             % (a_cit, n, lo) if a_cit >= lo else ""))
    print("")


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 8
    print("x281 — 완료-대조표 격리 프로브 (x280 후속 · 팔별 n=%d · 유료 0)" % n)
    print("가설: 결손은 열거가 아니라 **요청 집합에서 완료분을 빼는 대조**([[63]] 계열).")
    print("")
    for name, tag, task, trial, kind, expect_cut, note in CTX_SPEC:
        try:
            run_ctx(name, tag, task, trial, kind, expect_cut, note, n)
        except Exception as e:
            print("%s SKIP — 실행 예외 %r · 해소: 해당 시행의 messages/tool_calls 표기를"
                  " 확인 후 재실행([[64]])\n" % (name, e))
    hi, lo = int(0.75 * n + 0.5), int(0.25 * n + 0.5)
    print("※ 판정 문턱(사전등록 · 비율 고정 · n=%d):" % n
          + "\n  재현:    ctx_P41 A_LIVE  REMAIN_NONE  ≥%d/%d (=0.75) ⇒ x280 결손 재현."
            % (hi, n)
          + "\n  닫힘:    ctx_P41 B_TABLE **REMAIN_CITED(1)** ≥%d/%d (=0.75) ⇒ **대조표가"
            " 닫는다** — 레버 = 전달(요청↔완료 대조 표면화 · 엔진은 문자열 추출만)."
            % (hi, n)
          + "\n           CITED(k≥2)=**미차감 버킷** — 닫힘에 산입하지 않는다(빼기 실패)."
          + "\n  과유도:  ctx_D43 또는 ctx_070 에서 (B_TABLE CITED − A_LIVE CITED) ≥%d/%d"
            " (=0.25) ⇒ **무효**(표가 없는 잔여를 만들어냄)." % (lo, n)
          + "\n           A_LIVE CITED 자체가 ≥%d/%d 이면 그 ctx 는 부정통제 불량으로"
            " 판정 보류([[57]])." % (lo, n)
          + "\n  안 닫힘: ctx_P41 B_TABLE REMAIN_NONE  ≥%d/%d (=0.75) ⇒ 전달로 안 닫힘 ="
            " 능력 축([[13]] 학습·스케일)." % (hi, n)
          + "\n  중간(0.25~0.5 구간) ⇒ n=16 증량(비율 문턱 그대로 환산 · 사후 해석 금지) ·"
            " ERR>0 팔은 판정 보류([[55]])."
          + "\n  SKIP·등록 좌표 불일치 ctx 는 해당 행 판정 불가(사유·해소는 위 줄에 인쇄).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
