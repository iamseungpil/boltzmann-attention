# -*- coding: utf-8 -*-
r"""x308 — 처방 시험: write **착수**를 격리 서브로 옮기면 나가는가 (x307 이 licence 를 준 유일 레버).

x307 확정: 073 credit 축은 **knowing-doing** 이다 — 같은 최소 문맥에서
  A_REF/B_ONE  target **0/8** (전부 `get_user_information_by_name` 어트랙터)
  C_ASK        knows **7/8** (도구명 + 금액을 텍스트로 정확히 지목)
[[62]] ②: 격리에서 아는데 궤적서 못 하면 레버는 **전달·배치뿐**이다. 계산·선택 대행은 무효.
선례: FIX-10(fetch_formalize 격리 서브)이 fee 산출에서 같은 형태로 통했고(문면-도달 갭 3연속의
처방), `_sub_fetch_formalize` 는 **메인 턴 소모 0**·엔진 리터럴 0으로 이미 배관돼 있다.

셀 4 (n=8·전 팔 같은 사실·같은 도구·변수는 **배치와 요구 형식**뿐):
  A_MAIN   x307 A_REF 축자(메인 문맥 + 소유권 note)                 ← 기준선(0/8 재현)
  B_SUB    **서브 프레이밍**: 이 결정만 담은 짧은 지시 + 같은 fee 결과 축자 + 같은 도구 목록
           ("이 요청을 성취하는 호출을 지금 만들어라")               ← 처방 팔
  C_SUBQ   B_SUB 와 동일하되 **답을 JSON 으로 요구**(형식 고정·도구 호출 대신 인자 산출)
           ← 착수가 막힌 것인지 *도구 호출 형식*이 막힌 것인지 가름
  D_NOBASIS  B_SUB 에서 **fee 결과 줄을 제거**(근거 없음)             ← 날조 부정통제([[25]])
           반드시 호출/값 산출이 **없어야** 한다. 나오면 B_SUB 결과는 무효.

판정(사전 고정):
  B_SUB ≥6 ∧ A_MAIN ≤2 ∧ D_NOBASIS ≤1 → **배치 레버 출시 후보**(write-착수 서브·FIX-10 동형)
  B_SUB ≤2 ∧ C_SUBQ ≥6 → 막힌 것은 **도구-호출 채널**이지 결정이 아님 → 결정론 실행 배관 검토
  B_SUB ≤2 ∧ C_SUBQ ≤2 → 서브도 안 열림 → 이 축은 배치로도 안 사짐(경계 후보·[[45]])
  D_NOBASIS ≥2 → 근거 없이 지어낸다 = 프로브 무효·서브 안전판 선결(FIX-11 계열)

실행(리모트·8141): T2_PROBE_URL=http://localhost:8141/v1/chat/completions \
  /home/woori/venvs/seka_env/bin/python x308_write_isolate_iso.py [N]
"""
import collections
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
import x238_action_forensic as X                                  # noqa: E402
import x241_uncalled_unlock_probe as U                            # noqa: E402
import x302_kindmap_ownership_probe as K                          # noqa: E402
import x306_phase_fold_probe as F                                 # noqa: E402

TARGET = K.TARGET
# 서브 지시문 = 기존 격리 서브의 계약 형태(결정 하나·근거는 아래 줄·판단은 LLM 몫).
SUB_HEAD = ("You are handling ONE decision in isolation. Below are the customer's request and the "
            "verbatim results of the fee-audit tool for this customer's checking accounts. "
            "Using ONLY those results, carry out the correction the policy requires now.")
SUB_JSON = ("Reply with JSON only, no prose: "
            '{"calls": [{"tool": "<tool name>", "account_id": "<id>", "amount": <number>, '
            '"credit_type": "<type>"}]}. If the results give no basis, reply {"calls": []}.')


def fee_lines(sim, cut):
    """fee 도구 산출 줄만 축자로 (x306 early_ctx 에서 그 줄만 추린다 — 더하는 값 0)."""
    return [ln for ln in F.early_ctx(sim, cut).split("\n")
            if "does NOT match" in ln or "net correction" in ln]


def ask_lines(sim, cut):
    """손님 요구 축자(첫 줄 + 컷 직전 마지막 user)."""
    rows = F.early_ctx(sim, cut).split("\n")
    return [r for r in rows if r.startswith("[user] ")]


def json_hit(r):
    """C_SUBQ·D_NOBASIS 채점: 도구명 + 금액이 담긴 호출을 산출했나(모델 자기 출력만 본다)."""
    t = str(r.get("content") or "")
    m = re.search(r'"calls"\s*:\s*\[(.*?)\]', t, re.S)
    body = m.group(1) if m else t
    return bool(TARGET in body and re.search(r'"amount"\s*:\s*\d', body))


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 8
    sim = next(s for s in X.load(K.TAG) if s["task_id"] == K.TASK
               and s.get("reward_info") is not None)
    cut = F.cut_of(sim)
    note = K.NOTE_T % ", ".join(K.kind_matches())
    tools = U.tools_of(sim)
    a_main = F.early_ctx(sim, cut) + "\n[system] " + note
    fees, asks = fee_lines(sim, cut), ask_lines(sim, cut)
    sub = "\n".join([SUB_HEAD, ""] + asks + [""] + fees + ["", note])
    sub_nb = "\n".join([SUB_HEAD, ""] + asks + ["", note])          # 근거 제거
    print("x308 cut=%d · A_MAIN %d자 · B_SUB %d자(fee줄 %d) · D_NOBASIS %d자 · n=%d · URL=%s\n" % (
        cut, len(a_main), len(sub), len(fees), len(sub_nb), n,
        os.environ.get("T2_PROBE_URL", "8140(기본⚠)")))
    arms = (("A_MAIN", a_main, tools, False),
            ("B_SUB", sub, tools, False),
            ("C_SUBQ", sub + "\n" + SUB_JSON, None, True),
            ("D_NOBASIS", sub_nb + "\n" + SUB_JSON, None, True))
    for label, body, tl, as_json in arms:
        hit = 0
        cnt = collections.Counter()
        for i in range(n):
            try:
                r = chat(body, tl, 0.0 if i == 0 else 0.7, 1500)
            except Exception as e:
                r = {"content": "ERR %s" % type(e).__name__}
            if as_json:
                ok = json_hit(r)
                cnt["call-emitted" if ok else "none/other"] += 1
            else:
                k = F.classify(r)
                ok = k == "target"
                cnt[k] += 1
            hit += ok
            print("    [%s %02d] %s" % (label, i, "HIT" if ok else "-"), flush=True)
        print("%-10s %d/%d · %s\n" % (label, hit, n, dict(cnt)))
    print("※ 판정(사전 고정): B_SUB ≥6 ∧ A_MAIN ≤2 ∧ D_NOBASIS ≤1 → 배치 레버 출시 후보 · "
          "B_SUB ≤2 ∧ C_SUBQ ≥6 → 도구-호출 채널이 병목 · 둘 다 ≤2 → 배치로도 안 사짐(경계) · "
          "D_NOBASIS ≥2 → 무효(날조 안전판 선결).")


if __name__ == "__main__":
    main()
