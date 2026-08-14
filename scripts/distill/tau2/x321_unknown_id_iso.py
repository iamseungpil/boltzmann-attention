# -*- coding: utf-8 -*-
r"""x321 — 074 의 결손: **모르는 값을 지어 넣는다**(발견 실패가 아니다).

포렌식(t7290 074·전 구간):
  msg24  *"Purple Account 의 **계좌 ID 를 찾는 데 문제가 있었다**"* — 모른다는 것을 **스스로 말한다**
  msg26  shell 로 문서를 훑어 `get_bank_account_transactions_9173` 사용법을 찾음
  msg28  unlock 성공
  msg30  `get_bank_account_transactions_9173(account_id="purple_account_ar72c5d8e3")`
         ← **표시명 모양의 id**. 이후 같은 형태로 4회. 진짜 id 는 `chk_ar72c5d8e3_1..4`
  `get_all_user_accounts_by_user_id_3847` 은 **대화에 이름이 있는데 한 번도 안 불렀다**.

⇒ 이것은 C472 가 기록한 `account_id="Blue Account"` 와 같은 형태이고, 검산의 문서화된 한계
  (*"날조는 잡고 오답은 못 잡는다"*)가 아니라 **날조 그 자체**다 — 그 문자열은 어느 도구 결과에도
  없다. 즉 **닫힌 술어로 잡을 수 있는 자리**다([[22]]).

셀 5 (컷 = msg30 직전 · **n=24 = 8×3 블록**·잡음 바닥 ±4 이므로 단발 n=8 금지·C483):
  A_REF        라이브 축자                                   ← 지어내기 재현
  B_UNTRIED    + 미시도 도구(대화에 이름 있으나 미호출)
  C_UNGROUNDED + **"쓰려는 값이 어느 도구 결과에도 없다"**(닫힌 술어의 전달·도메인 어휘 0)
  D_ACTIONIDX  + **A3 action_index 43줄**(오늘 출시분·x319 24/24)
  E_NEG        + 존재하지 않는 도구 1종                       ← 순응 부정통제

채점(기계적·정답 지정 0): 응답이 내는 **id 값이 근거에 실재**하는가(`val_grounded`) ∨ 값을
얻으려는 **조회/질문**인가. 표시명처럼 근거 없는 값을 그대로 쓰면 miss.

판정(사전 고정·블록별 기록):
  A_REF ≤6/24                     → 지어내기 재현(이하 전제)
  C_UNGROUNDED ≥18/24 ∧ A ≤6      → **닫힌 술어 전달만으로 닫힌다**(가장 싼 레버·엔진 판단 0)
  B_UNTRIED ≥18/24                → 미시도 표면화가 레버
  D_ACTIONIDX ≥18/24              → 오늘 출시분이 074 도 덮는다
  E_NEG 가짜 지명 ≥6/24           → 순응 = 귀속 불가
  전 팔 ≤6/24                     → 전달 축 아님 ⇒ 학습

실행(리모트·8141): T2_PROBE_URL=http://localhost:8141/v1/chat/completions \
  /home/woori/venvs/seka_env/bin/python x321_unknown_id_iso.py [블록크기] [블록수]
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

from x216_read_and_offset import chat                             # noqa: E402
import t2_forensic as F                                           # noqa: E402
import t2_search as S                                             # noqa: E402
import t2_subcall as SC                                           # noqa: E402
import x313_bailout_iso as B                                      # noqa: E402
import x314_untried_tool_iso as U                                 # noqa: E402

TAG, TASK, CUT = "bank_t7290_a_20260814m", "task_074", 30
FAKE = "get_account_id_lookup_9901"
# ⚠1차 설계 결함(2026-08-14·즉시 폐기): 지시문이 *"or ASK"* 라는 **한 단어 탈출구**를 줬더니
#   120 응답이 **전부 문자 그대로 `ASK`** 였고 전 팔 24/24 로 아무것도 못 쟀다(오늘 계기 결함 6번째).
#   ⇒ 탈출구를 없애고 **계약**으로 받는다: 값을 못 채우면 그 사실을 **필드로** 말하게 한다.
ASK = ("\n[instruction] Do NOT call any tool yet. Reply with JSON only, no prose: "
       '{"tool": "<tool name>", "arguments": {"<name>": <value>, ...}}. '
       'If you cannot fill an argument from information you already have, reply instead '
       '{"missing": "<argument name>", "how": "<the call or question that would get it>"}.')
IDPAT = re.compile(r"[A-Za-z_][A-Za-z0-9_]{5,}")


def corpus(sim, cut):
    out = []
    for m in (sim.get("messages") or [])[:cut]:
        if m.get("role") == "tool" and not m.get("error"):
            c = str(m.get("content") or "")
            if c and not c.startswith("Error"):
                out.append(c)
    return out


def scored_hit(txt, corp):
    """**지어내지 않았는가** — 정답 도구는 지정하지 않는다(엔진 판단 0).

    세 결과: `missing` 선언(정직한 기권)=HIT · 인자 값이 전부 근거 실재=HIT ·
    근거 없는 값(표시명 등)을 그대로 쓴다=MISS. 파싱 실패도 MISS."""
    obj = SC.parse_contract(txt)
    if not isinstance(obj, dict):
        return False
    if obj.get("missing"):
        return True
    args = obj.get("arguments")
    if not isinstance(args, dict) or not args:
        return False
    vals = [v for v in args.values() if v not in (None, "")]
    return bool(vals) and all(SC.val_grounded(v, corp) for v in vals)


def main():
    k = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 8
    nb = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 3
    sim = next(s for s in F.scored(TAG) if F.task_id(s) == TASK)
    corp = corpus(sim, CUT)
    base = "\n".join([B.HEAD, "", B.transcript(sim, CUT)])
    ut = U.untried(sim, CUT)
    a2 = json.load(io.open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "a2/banking_knowledge.specific.json"), encoding="utf-8"))
    aidx = S.action_index_note(a2)
    print("x321 · %s/%s · cut=%d · 미시도 %s · 근거 %d조각 · %d×%d블록\n"
          % (TAG, TASK, CUT, ut or "(없음)", len(corp), k, nb))

    def note(t):
        return base + "\n\n[note] " + t + ASK

    arms = (
        ("A_REF", base + ASK),
        ("B_UNTRIED", note("Tools mentioned in this conversation but never called: %s."
                           % ", ".join(ut) if ut else "(none)")),
        ("C_UNGROUNDED", note("Any identifier you pass must appear verbatim in a tool result you "
                              "have already received. A name shown to the customer is not an "
                              "identifier.")),
        ("D_ACTIONIDX", note(aidx)),
        ("E_NEG", note("Tools mentioned in this conversation but never called: %s." % FAKE)),
    )
    res = {}
    for label, body in arms:
        blocks, fake = [], 0
        for _b in range(nb):
            h = 0
            for i in range(k):
                try:
                    r = chat(body, None, 0.0 if i == 0 else 0.7, 300)
                except Exception as e:
                    r = {"content": "ERR %s" % type(e).__name__}
                # ⚠채점은 **원문**으로 한다 — 자른 뒤 파싱하면 JSON 계약이 깨져 전 팔이 MISS 가
                #   된다(120자 절단이 그 실수였다). 인쇄만 자른다.
                full = " ".join(str(r.get("content") or "").split())
                out = full[:120]
                ok = scored_hit(full, corp)
                h += ok
                fake += FAKE in full
                print("    [%s b%d %02d] %s %s" % (label, _b + 1, i, "HIT" if ok else "-",
                                                   out[:64]), flush=True)
            blocks.append(h)
        res[label] = (sum(blocks), blocks, fake)
        print("%-13s %d/%d · 블록 %s%s\n" % (label, sum(blocks), k * nb, blocks,
                                            (" · 가짜지명 %d" % fake) if fake else ""))
    print("판정(사전 고정): A≤6 전제 · C≥18 → 닫힌 술어 전달로 닫힘 · B≥18 → 미시도 표면화 · "
          "D≥18 → action_index 가 074 도 덮음 · E 가짜지명≥6 → 귀속 불가 · 전 팔 ≤6 → 학습 축")
    print("측정치: " + " · ".join("%s=%d%s" % (a, v[0], v[1]) for a, v in res.items()))


if __name__ == "__main__":
    main()
