# -*- coding: utf-8 -*-
r"""x280 — 여분 write(발산) 격리 프로브 (유료 0 · 엔진 0 · 사용자 지시 2026-08-13
*"write 차단도 격리로 확인하라"*).

## 왜 (C451·C453 실측)

070/071 의 잔여는 **선택**이 아니라 **발산**이다: 격리 결정은 gold 를 낸다(`DOCDECIDE →
'Sky Blue'` · x248/x250 두 축 8/8), 그런데 gold 를 다 채운 **뒤** 여분 write 가 최종 DB 를
어긋나게 한다 — j런 070 t0 은 `action_checks` **5/5 만점인데 reward 0**(msg 36 Sky Blue
개설 성공 → msg 70 Cobalt Blue 추가 개설 → Sky Blue 폐쇄).

## 질문 ([[62]] ①)

여분 write 직전 문맥에서, 모델은 **손님이 요청한 개설이 이미 끝났다는 것**을 아는가?
안다면 결손은 knowing-doing 이고 레버는 **표면화(전달)** 뿐이다([[62]]②). 모른다면 능력
축이고 학습·스케일 논의로 간다([[13]]).

## 문맥 — **한 궤적·세 절단**(답이 뒤집혀야 한다·[[57]] 최강 형태)

  ctx_R   j런 070 t0 · cut=36  = 첫 성공 개설 **직전** → 정답 = "요청이 남아 있다"
  ctx_X1  같은 궤적 · cut=48   = Sky Blue 개설 성공 후 **중복 시도** 직전 → 정답 = "없다"
  ctx_X2  같은 궤적 · cut=70   = 여분 개설(Cobalt Blue) 직전 → 정답 = "없다"

절단은 기계 규칙이다: 성공한 open 디스패치 인덱스(=결과가 'opened successfully')와 그 뒤의
open 시도 인덱스. gold 를 프롬프트에 넣지 않는다([[03b]]/[[23]]) — 채점도 인용 실재만 본다.

## 계약 (모델이 내는 것)

  {"remaining": [{"account_type": "...", "message_index": N, "quote": "<손님 발화 축자>"}]}
  손님이 요청한 개설이 전부 끝났으면 정확히 한 낱말: none

## 엔진 검증 (전부 닫힌 술어 · 의미 판정 0)

  ①JSON 파스·키 ②message_index 범위 ③그 메시지 role=="user" ④quote 가 **그 메시지**에
  축자 실재(공백 정규화) ⑤len(정규화 quote) ≥ 15. 유효 항목 수 k 만 센다 —
  *남은 요청이 실제로 타당한가* 는 판정하지 않는다.

## 버킷

  REMAIN_NONE(=끝났다고 답함) · REMAIN_CITED(k≥1·유효 인용) · REMAIN_FABRICATED(항목은
  냈는데 유효 0) · PLAIN · ERR(판정 보류·[[55]])

## 판정 문턱 (사전등록 · n=8)

  ctx_R  `REMAIN_CITED` ≥6/8            ⇒ 부정통제 통과(답이 달라진다) — 미달이면 프로브 무효
  ctx_X1·X2 `REMAIN_NONE` ≥6/8          ⇒ **모델은 완료를 안다** = knowing-doing ⇒ 레버 =
                                           완결 사실 **표면화**(전달·[[62]]②)
  ctx_X1·X2 `REMAIN_CITED` ≥4/8         ⇒ 완료를 모른다 = 능력 축(학습·스케일·[[13]])
  중간 2~3/8                             ⇒ n=16 증량(사후 해석 금지)

실행: T2_PROBE_URL=http://localhost:8141/v1/chat/completions python3 x280_write_divergence.py [N]
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

SIMS = os.environ.get("X280_SIMDIR", "/home/woori/scratch/tau2-bench/data/simulations")
TAG = os.environ.get("X280_TAG", "bank_lever4_a_20260812j")
TASK = os.environ.get("X280_TASK", "task_070")
TRIAL = int(os.environ.get("X280_TRIAL", "0"))
WRITE_MARK = "open_bank_account"
OK_MARK = "opened successfully"

REMAIN_FB = (
    "\n[system] Task: from the transcript above, output exactly one line - the JSON "
    "{\"remaining\": [{\"account_type\": \"<type>\", \"message_index\": <index of the customer "
    "message asking for it>, \"quote\": \"<verbatim sentence from that message>\"}]}. "
    "Include only account openings the customer asked for that have NOT already been "
    "completed successfully in this transcript. If every account the customer asked for "
    "has already been opened, reply with exactly the single word: none"
)
REM_START = re.compile(r"\{\s*[\"']remaining[\"']")
NONE_WORD = re.compile(r"^\s*[`\"']?none[`\"'.]?\s*$", re.I)


def _norm(s):
    return " ".join(str(s or "").split()).lower()


def _tc_args(tc):
    a = tc.get("arguments")
    return a if isinstance(a, str) else json.dumps(a or {}, ensure_ascii=False)


def load_sim():
    d = json.load(io.open(os.path.join(SIMS, TAG, "results.json"), encoding="utf-8"))
    sims = [s for s in d["simulations"] if s["task_id"] == TASK]
    cands = [s for s in sims if s.get("trial") == TRIAL]
    if cands:
        return cands[0]
    raise SystemExit("trial=%s 시행 없음 (%s %s)" % (TRIAL, TAG, TASK))


def open_points(msgs):
    """(인덱스, 성공여부) — open 디스패치 전수. 기계 판정뿐(도구명·결과 문자열)."""
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


def build(sim, cut):
    """[i] 인덱스 접두 — 계약의 message_index 좌표를 모델이 봐야 한다(x279 규약 동일)."""
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


def make_scorer(msgs, cut):
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


def run_arm(label, body, tools, n, scorer):
    c = collections.Counter()
    errs = collections.Counter()
    for i in range(n):
        try:
            r = chat(body, tools, 0.0 if i == 0 else 0.7, 400)
        except Exception as e:
            errs[type(e).__name__] += 1
            continue
        c[scorer(r)] += 1
    flag = "   ⚠ERR %s ⇒ 판정 보류([[55]])" % dict(errs) if errs else ""
    print("    %-9s %s%s" % (label, dict(c), flag))
    return c


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 8
    sim = load_sim()
    msgs = sim["messages"]
    pts = open_points(msgs)
    ok = [i for i, good in pts if good]
    tries_after = [i for i, good in pts if not good and ok and i > ok[0]]
    print("x280 — 여분 write 격리 (%s %s trial=%d · reward=%s · n=%d)"
          % (TAG, TASK, TRIAL, (sim.get("reward_info") or {}).get("reward"), n))
    print("open 디스패치: %s (성공 %s)" % (pts, ok))
    if not ok:
        print("SKIP — 성공한 open 이 없다 · 해소: 발산이 실재하는 시행으로 교체([[64]])")
        return 0
    cells = [("ctx_R", ok[0], "첫 성공 개설 직전 — 정답 = 남아 있다(부정통제)")]
    if tries_after:
        cells.append(("ctx_X1", tries_after[0], "개설 성공 후 중복 시도 직전 — 정답 = 없다"))
    if len(ok) > 1:
        cells.append(("ctx_X2", ok[1], "여분 개설 직전 — 정답 = 없다"))
    tools = U.tools_of(sim)
    for name, cut, note in cells:
        live = build(sim, cut)
        print("\n%s cut=%d · 문맥 %d자 · %s" % (name, cut, len(live), note))
        print("  직전 msg[%d] %s: %s"
              % (cut - 1, msgs[cut - 1].get("role"),
                 " ".join(str(msgs[cut - 1].get("content") or "").split())[:120]))
        run_arm("A_REMAIN", live + REMAIN_FB, tools, n, make_scorer(msgs, cut))
    print("\n※ 판정 문턱(사전등록·n=%d):" % n
          + "\n  ctx_R  REMAIN_CITED ≥6/8   ⇒ 부정통제 통과(답이 달라진다) — 미달이면 프로브 무효."
          + "\n  ctx_X* REMAIN_NONE  ≥6/8   ⇒ 모델은 완료를 안다 = knowing-doing ⇒ 레버 = 완결"
            " 사실 표면화(전달·[[62]]②)."
          + "\n  ctx_X* REMAIN_CITED ≥4/8   ⇒ 완료를 모른다 = 능력 축(학습·스케일·[[13]])."
          + "\n  중간 2~3/8 ⇒ n=16 증량(사후 해석 금지). ERR>0 팔은 판정 보류.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
