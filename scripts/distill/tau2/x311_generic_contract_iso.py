# -*- coding: utf-8 -*-
r"""x311 — write-착수 계약의 **일반화**: 도메인 필드를 빼도 되나, 그리고 다른 착수에도 옮겨지나.

배경(2026-08-14):
  · FIX-13 의 A2 `write_initiation.answer_format` 은 **073형 필드**를 박고 있다
    (`account_id`/`amount`/`credit_type`) — [[03b]](측정본=출시본) 때문에 x308 축자를 유지했지만
    minimize-A2 관점에선 도메인 순증이고, 실물에서 대가가 나왔다:
  · t7289 075 는 **계좌 개설** 착수가 필요한데(개설 필드는 `user_id`/`account_type`/`account_class`)
    그 구간에서 서브는 **제안 0건** — 계약이 안 맞아 도울 수 없었다. 075 는 msg31 에서 손님이
    도구명까지 대며 실행을 요구했는데 msg32 에서 **안내문만** 냈다(knowing-doing 동형·gold 2칸 공백).

질문 둘:
  ⓐ 계약에서 도메인 필드를 빼고 `{tool, arguments{...}}` 로 해도 073 에서 **그대로 되는가**
  ⓑ 그 일반 계약이 **075 착수**(다른 도구·다른 필드)에도 옮겨지는가 — 옮겨지면 FIX-13 은
     073 전용이 아니라 **착수 결손 일반**의 레버가 된다

셀 5 (n=8·073 컷은 x308 축자·075 컷은 손님이 도구명을 댄 그 자리 직후):
  A_DOM      073 · 현 출시 계약(도메인 필드)            ← 기준선(x308 C_SUBQ 8/8 재현 확인)
  B_GEN      073 · **일반 계약**({tool, arguments})      ← ⓐ
  C_GEN75    075 · 일반 계약 · 근거=**직전 도구결과**    ← ⓑ (현 배선과 같은 근거 폭)
  D_GEN75F   075 · 일반 계약 · 근거=**대화 전체 도구결과** ← 근거 폭이 인자인가
             (075 의 `user_id` 는 msg17 에서 나왔고 그 뒤 손님 발화가 여러 번 있다 —
              현 배선(`recent_tool_text`)은 그걸 못 본다)
  E_NOBASIS  075 · 일반 계약 · **근거 제거**             ← 날조 부정통제([[25]])

판정(사전 고정):
  B_GEN ≥6 ∧ A_DOM ≥6      → 도메인 필드 제거해도 유지 ⇒ **A2 일반화 출시**
  C_GEN75 ≥6               → 착수 일반 레버 확정(075 도 덮인다)
  C_GEN75 ≤2 ∧ D_GEN75F ≥6 → **근거 폭이 인자** ⇒ 배선의 corpus 를 전체 원장으로 이설
  E_NOBASIS ≥2             → 무효(날조) · B_GEN ≤2 → 도메인 필드가 실제로 일하고 있다(유지)

실행(리모트·8141): T2_PROBE_URL=http://localhost:8141/v1/chat/completions \
  /home/woori/venvs/seka_env/bin/python x311_generic_contract_iso.py [N]
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
import x302_kindmap_ownership_probe as K                          # noqa: E402
import x306_phase_fold_probe as F                                 # noqa: E402
import x308_write_isolate_iso as S                                # noqa: E402

TAG75 = "bank_t7289_b_20260814k"
TASK75 = "task_075"
TOOL75 = "open_bank_account_4821"
GEN_JSON = ('Reply with JSON only, no prose: {"calls": [{"tool": "<tool name>", '
            '"arguments": {"<name>": <value>, ...}}]}. '
            'If the results give no basis, reply {"calls": []}.')


def sim_of(tag, task):
    return next(s for s in X.load(tag) if s["task_id"] == task
                and s.get("reward_info") is not None)


def cut75(sim):
    """컷 = 손님이 **도구명을 대며 실행을 요구한** 발화 직후(그 자리에서 라이브가 안내문을 냈다)."""
    for i, m in enumerate(sim["messages"]):
        if m.get("role") == "user" and TOOL75 in str(m.get("content") or ""):
            return i + 1
    return None


def lines(sim, cut, whole=False):
    """근거 줄 — whole=False 는 직전 손님 발화 이후(현 배선), True 는 대화 전체 성공 도구결과."""
    msgs = sim["messages"][:cut]
    if not whole:
        last_user = max([i for i, m in enumerate(msgs) if m.get("role") == "user"] or [-1])
        msgs = msgs[last_user + 1:]
    out = []
    for m in msgs:
        if m.get("role") != "tool" or m.get("error"):
            continue
        c = " ".join(str(m.get("content") or "").split())
        if c and not c.startswith("Error"):
            out.append(c[:1500])
    return out


def asks(sim, cut):
    return ["[user] " + " ".join(str(m.get("content")).split())[:500]
            for m in sim["messages"][:cut] if m.get("role") == "user"][-3:]


def hit(r, tool):
    t = str(r.get("content") or "")
    m = re.search(r'"calls"\s*:\s*\[(.*?)\]\s*\}', t, re.S)
    body = m.group(1) if m else t
    return bool(tool in body and re.search(r'"arguments"|"account_id"|"user_id"', body))


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 8
    s73 = sim_of(K.TAG, K.TASK)
    c73 = F.cut_of(s73)
    note73 = K.NOTE_T % ", ".join(K.kind_matches())
    s75 = sim_of(TAG75, TASK75)
    c75 = cut75(s75)
    if c75 is None:
        print("075 컷 없음")
        return
    note75 = K.NOTE_T % TOOL75
    print("x311 · 073 cut=%d · 075 cut=%d · n=%d · URL=%s\n" % (
        c73, c75, n, os.environ.get("T2_PROBE_URL", "8140⚠")))

    def body(sim, cut, note, fmt, whole=False, nobasis=False):
        b = lines(sim, cut, whole)
        return "\n".join([S.SUB_HEAD, ""] + asks(sim, cut) + [""]
                         + ([] if nobasis else b) + ["", note, fmt])

    arms = (
        ("A_DOM", body(s73, c73, note73, S.SUB_JSON), K.TARGET),
        ("B_GEN", body(s73, c73, note73, GEN_JSON), K.TARGET),
        ("C_GEN75", body(s75, c75, note75, GEN_JSON), TOOL75),
        ("D_GEN75F", body(s75, c75, note75, GEN_JSON, whole=True), TOOL75),
        ("E_NOBASIS", body(s75, c75, note75, GEN_JSON, nobasis=True), TOOL75),
    )
    for label, b, tool in arms:
        k = 0
        cnt = collections.Counter()
        for i in range(n):
            try:
                r = chat(b, None, 0.0 if i == 0 else 0.7, 1500)
            except Exception as e:
                r = {"content": "ERR %s" % type(e).__name__}
            ok = hit(r, tool)
            k += ok
            cnt["call" if ok else "none/other"] += 1
            print("    [%s %02d] %s" % (label, i, "HIT" if ok else "-"), flush=True)
        print("%-10s %d/%d · %s (본문 %d자)\n" % (label, k, n, dict(cnt), len(b)))
    print("※ 판정(사전 고정): B_GEN≥6∧A_DOM≥6 → A2 일반화 출시 · C_GEN75≥6 → 착수 일반 레버 · "
          "C≤2∧D≥6 → 근거 폭이 인자(전체 원장으로 이설) · E_NOBASIS≥2 → 무효.")


if __name__ == "__main__":
    main()
