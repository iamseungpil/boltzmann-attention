# -*- coding: utf-8 -*-
r"""x314 — 이탈이 **새로 가능한 길**을 주면 열리는가: 미시도 도구 표면화.

선행(C476·x313·사이트 2 동일): A_REF 0/8(이탈 재현) · **B_KNOW 8/8**(남은 일을 다 안다) ·
**C_LIST 0/8**(그 목록을 돌려줘도 이관) · D_NONOTE 0/8(우리 문구는 인자 아님) · E_NOBASIS 0/8.
⇒ knowing-doing 인데 **전달이 안 열었다**. 다만 C_LIST 가 준 것은 *목표의 재진술*이지
**새로 가능한 길**이 아니었다 — 두 컷 모두 직전에 반복 실패가 쌓여 있다(074 는 같은 잘못된
`account_id` 로 4회). 이관은 "길이 없다"는 판단의 합리적 귀결일 수 있다.

여기서 재는 것: 술어 = **대화에 이름이 등장했으나 한 번도 호출되지 않은 도구**.
  · 닫힌 술어다([[22]]) — 변이 불변·엔진이 이미 가진 두 집합의 차집합(이름 집합 ∖ 호출 집합)
  · **gold 무접촉**([[23]]) — 출처는 KB 문서/도구 출력이지 채점표가 아니다
  · 엔진은 **고르지 않는다** — 차집합을 그대로 나열할 뿐이고 무엇을 부를지는 LLM 이 정한다([[62]] ④)
074 에서 이 술어가 집는 것은 `get_all_user_accounts_by_user_id_3847` — 실제로 **한 번도 안 불렀고**,
그래서 진짜 계좌 id 를 가진 적이 없다(표시명 모양 id 를 4회 사용·C475 ⒡ 포렌식).

셀 4 (n=8·컷·본문은 x313 축자·변수는 동봉물뿐):
  A_REF       x313 축자                                  ← 이탈 재현 기준선
  B_UNTRIED   + "다음 도구는 아직 한 번도 호출되지 않았다: <차집합>"
  C_FAILNOTE  + "네 호출 X 는 인자 때문에 결과를 못 냈다"(궤적 사실 전사·새 정보 0)
  D_NEG       + 같은 형식으로 **레지스트리에 없는 이름** 한 개   ← 이름 순응 부정통제([[57]]·x309 5/8)

판정(사전 고정):
  B_UNTRIED ≥6 ∧ A_REF ≤2 ∧ D_NEG ≤2 → **미시도 표면화가 레버**(전달로 열리되 '새 길'일 때만)
  B_UNTRIED ≥6 ∧ D_NEG ≥3            → 이름만 대면 따른다 = 귀속 불가·레버 주장 금지
  B_UNTRIED ≤2                        → 새 길을 줘도 안 열린다 ⇒ 이탈은 전달 축이 아니다(학습/스케일 축)
  C_FAILNOTE ≥6                       → 실패 사실 전사만으로 열림(더 싼 레버)

실행(리모트·8141): T2_PROBE_URL=http://localhost:8141/v1/chat/completions \
  /home/woori/venvs/seka_env/bin/python x314_untried_tool_iso.py [N]
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
import t2_forensic as F                                           # noqa: E402
import x313_bailout_iso as B                                      # noqa: E402

TOOLNAME = re.compile(r"\b[a-z][a-z_]{5,}_\d{4}\b")
FAKE = "get_account_overview_6610"        # 레지스트리에 없는 이름(부정통제)

# 사이트 = 이탈이 실측된 자리 중 **미시도 집합이 비지 않은** 것(048 은 0개라 제외·판정 대상 아님).
# 표면화 **규모**가 다른 셋을 일부러 섞는다 — 1개(074)와 19개(087·072)가 같은 방향이면
# "긴 목록은 소음"이라는 대안 설명이 함께 죽는다.
SITES = (("bank_t7290_a_20260814m", "task_074"),      # 미시도 1개(그 하나가 결정적)
         ("bank_t7286_a_20260814h", "task_087"),      # 미시도 19개
         ("bank_t7290_a_20260814m", "task_072"))      # 미시도 19개


def untried(sim, cut):
    """대화에 이름이 등장했으나 **한 번도 호출되지 않은** 도구 — 두 집합의 차집합뿐(판단 0)."""
    seen, called = set(), set()
    for m in (sim.get("messages") or [])[:cut]:
        seen |= set(TOOLNAME.findall(str(m.get("content") or "")))
        for tc in (m.get("tool_calls") or []):
            called.add(F.inner_name(F.argsof(tc)) or F.nameof(tc))
            seen |= set(TOOLNAME.findall(str(F.argsof(tc))))
    return sorted(seen - called)


def failed_calls(sim, cut):
    """궤적 사실 전사: 결과가 오류였던 호출의 (도구, 인자) — 새 정보 0."""
    out, byid = [], {}
    for m in (sim.get("messages") or [])[:cut]:
        for tc in (m.get("tool_calls") or []):
            byid[tc.get("id")] = F.label(F.nameof(tc), F.argsof(tc))
        if m.get("role") == "tool" and (m.get("error")
                                        or str(m.get("content") or "").startswith("Error")):
            nm = byid.get(m.get("id"))
            if nm:
                out.append(nm)
    return sorted(set(out))


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 8
    for tag, task in SITES:
        try:
            sim = next(s for s in F.scored(tag) if F.task_id(s) == task)
        except StopIteration:
            print("%s / %s 없음" % (tag, task))
            continue
        cut = B.transfer_cut(sim)
        if cut is None:
            print("%s / %s 컷 없음" % (tag, task))
            continue
        base = "\n".join([B.HEAD, "", B.transcript(sim, cut)])
        ut = untried(sim, cut)
        fc = failed_calls(sim, cut)
        print("\n%s\n# %s / %s · cut=%d · 미시도 %s · 오류호출 %s · n=%d\n%s"
              % ("=" * 88, tag, task, cut, ut or "(없음)", fc or "(없음)", n, "=" * 88))
        if not ut:
            print("미시도 집합이 비어 B_UNTRIED 를 못 만든다 — 이 사이트 건너뜀(판정 제외)")
            continue

        def note(lines):
            return base + "\n\n[note] " + " ".join(lines)

        arms = [
            ("A_REF", base),
            ("B_UNTRIED", note(["These tools have been mentioned in this conversation but have "
                                "NEVER been called yet: %s." % ", ".join(ut)])),
            ("D_NEG", note(["These tools have been mentioned in this conversation but have NEVER "
                            "been called yet: %s." % FAKE])),
        ]
        if fc:
            arms.insert(2, ("C_FAILNOTE",
                            note(["Your earlier call(s) to %s returned no usable result."
                                  % ", ".join(fc)])))
        res = {}
        for label, body in arms:
            k = 0
            cnt = collections.Counter()
            for i in range(n):
                try:
                    r = chat(body, None, 0.0 if i == 0 else 0.7, 1200)
                except Exception as e:
                    r = {"content": "ERR %s" % type(e).__name__}
                ok = B.acted(r)
                # 부정통제는 **그 가짜 이름을 실제로 부르는가**까지 본다(순응의 강한 형태)
                if label == "D_NEG" and FAKE in str(r.get("content") or ""):
                    cnt["fake-named"] += 1
                k += ok
                cnt["act" if ok else ("bail" if B.bailed(r) else "other")] += 1
                print("    [%s %02d] %s" % (label, i, "HIT" if ok else "-"), flush=True)
            res[label] = k
            print("%-11s %d/%d · %s\n" % (label, k, n, dict(cnt)))
        print("판정(사전 고정): B≥6∧A≤2∧D_NEG≤2 → 미시도 표면화가 레버 · B≥6∧D_NEG≥3 → 이름 순응"
              "(귀속 불가) · B≤2 → 전달 축 아님 · C_FAILNOTE≥6 → 더 싼 레버")
        print("측정치: " + " · ".join("%s=%d" % (k, v) for k, v in res.items()))


if __name__ == "__main__":
    main()
