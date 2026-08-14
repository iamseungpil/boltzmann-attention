# -*- coding: utf-8 -*-
r"""x307 — 073 write 착수 부담의 종류 가름: 다계좌 짝인가 · 착수 자체인가 · 모르는 것인가.

선행(전부 같은 사이트·같은 note 문면):
  x302b  라이브 전체 문맥 + 이름 3종/1종 note        **0/8**
  x306   최소 문맥(이력 0) + 같은 note               **0/8**  ← 국면 아님(사전등록 판정)
         · 실패 형태는 세 사이트 공통 어트랙터 `get_user_information_by_name` 8/8
  x300b/x298b  075·087 에선 같은 계열 note 가 **8/8·7/8** 로 연다
⇒ 073 만 다르다. 남은 후보는 **인자 부담**(계좌별 금액 짝)과 **착수 자체**(knowing-doing)다.

⚠**금액을 주지 않는다**([[62]] ③·[[23]]): 계좌별 net correction 은 **도구가 이미 축자로 말한다**
(`... requires ONE fee_refund credit for the net correction ... of THIS account = $1.50`).
이 프로브는 그 줄을 **더하지 않고 빼서** 부담만 줄인다 — 계산 대행 0·gold 무접촉.

셀 3 (n=8·컷/문면은 x306 축자 재사용·변수는 부담뿐):
  A_REF   x306 P_EARLY 축자(3계좌 전부 + note)        ← 기준선(0/8 재현 확인)
  B_ONE   **한 계좌만** 남긴 최소 문맥 + 같은 note     ← 다계좌 짝 부담 제거
  C_ASK   A_REF + *"도구를 부르지 말고 텍스트로: 다음 행동과 인자는 무엇인가"*  ← knowing 측정

판정(사전 고정):
  B_ONE ≥6 ∧ A_REF ≤2 → 부담 = **다계좌 짝/coverage** → 처방 = 계좌별 분해 전달(배치·기존 계열)
  B_ONE ≤2 ∧ C_ASK(도구명+금액 언급) ≥6 → **knowing-doing** = write 착수가 경계 → 문면 아닌 배치
  B_ONE ≤2 ∧ C_ASK ≤2 → 다음 행동을 모른다 = 재료/지식 축
  이름-특이성 부정통제는 x304 D_NEG(무관 이름 추종 0/8)로 이미 확보 — 재발사 없음([[57]]).

실행(리모트·8141): T2_PROBE_URL=http://localhost:8141/v1/chat/completions \
  /home/woori/venvs/seka_env/bin/python x307_write_initiation_iso.py [N]
"""
import collections
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
ASK = ("\n[user] Before you do anything: do NOT call any tools now. In text only, tell me "
       "exactly what the next action should be and with which arguments.")


def one_account(sim, cut):
    """A_REF 에서 **fee 결과 줄을 한 계좌분만** 남긴다 — 더하는 게 없고 빼기만 한다."""
    rows, kept = [], 0
    for ln in F.early_ctx(sim, cut).split("\n"):
        if "does NOT match" in ln or "net correction" in ln:
            kept += 1
            if kept > 1:
                continue
        rows.append(ln)
    return "\n".join(rows)


def knows(r):
    """텍스트가 다음 행동을 실제로 지목했나 — 도구명 ∧ 금액 표기(모델 자기 출력 채점)."""
    t = str(r.get("content") or "")
    return bool(TARGET in t and re.search(r"\$?\d+\.\d{2}", t))


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 8
    sim = next(s for s in X.load(K.TAG) if s["task_id"] == K.TASK
               and s.get("reward_info") is not None)
    cut = F.cut_of(sim)
    note = K.NOTE_T % ", ".join(K.kind_matches())
    tools = U.tools_of(sim)
    a_ref = F.early_ctx(sim, cut)
    b_one = one_account(sim, cut)
    print("x307 cut=%d · A_REF %d자 · B_ONE %d자 · n=%d · URL=%s\n" % (
        cut, len(a_ref), len(b_one), n, os.environ.get("T2_PROBE_URL", "8140(기본⚠)")))
    arms = (("A_REF", a_ref + "\n[system] " + note, False),
            ("B_ONE", b_one + "\n[system] " + note, False),
            ("C_ASK", a_ref + "\n[system] " + note + ASK, True))
    for label, body, ask in arms:
        hit = 0
        cnt = collections.Counter()
        for i in range(n):
            try:
                r = chat(body, tools, 0.0 if i == 0 else 0.7, 1500)
            except Exception as e:
                r = {"content": "ERR %s" % type(e).__name__}
            if ask:
                ok = knows(r)
                hit += ok
                cnt["knows" if ok else "no-name/amount"] += 1
                if i < 2:
                    print("    [%s %02d] %s" % (label, i,
                                                " ".join(str(r.get("content") or "").split())[:220]))
            else:
                k = F.classify(r)
                hit += k == "target"
                cnt[k] += 1
                print("    [%s %02d] %s" % (label, i, k), flush=True)
        print("%-7s %s %d/%d · %s\n" % (label, "knows" if ask else "target", hit, n, dict(cnt)))
    print("※ 판정(사전 고정): B_ONE ≥6 ∧ A_REF ≤2 → 다계좌 짝 부담 · B_ONE ≤2 ∧ C_ASK ≥6 → "
          "knowing-doing(착수 경계) · 둘 다 ≤2 → 재료/지식 축.")


if __name__ == "__main__":
    main()
