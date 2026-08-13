# -*- coding: utf-8 -*-
r"""x287 — not_discoverable deny 에 레지스트리 목록 동봉이 날조-접힘을 여는가 (073 t1 실물).

사전등록: `PROBE_X283_X285_DESIGN_2026_08_13.md` §x287.

문맥 = t7273 073 t1, 날조 이름(`apply_atm_fee_corrections_8472`) 호출 직전 prefix.
  A_LIVE  현행 `feedback_not_discoverable` 축자("there is none" 로 끝)
  B_LIST  A + `feedback_registry_listing` 축자({names}=레지스트리 기계 나열) — 출시본과 동일
          (t2_resolve 가 아니라 A2 키에서 읽는다 — 두 벌 금지·[[03b]])
  D_NEG   무내용 통제
HIT = 다음 호출(들)이 unlock/call 로 `apply_checking_account_credit` 가족을 표적.
문턱(사전 고정): B ≥6/8 ∧ A ≤2/8. 미달 → 키 제거(침묵=종전 거동)·L 백로그 이관.

실행(리모트·8141): T2_PROBE_URL=http://localhost:8141/v1/chat/completions \
  python x287_registry_listing_probe.py [N]
"""
import collections
import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from x216_read_and_offset import chat                             # noqa: E402
import x238_action_forensic as X                                  # noqa: E402
import x241_uncalled_unlock_probe as U                            # noqa: E402
import x283_discovery_reach_probe as P                            # noqa: E402

TAG = "bank_t7273_20260813s"
TASK = "task_073"
FAB = "apply_atm_fee_corrections_8472"
TARGET_FAM = "apply_checking_account_credit"
NEG = "Note: please continue helping the customer with their request."


def a2_texts():
    here = os.path.dirname(os.path.abspath(__file__))
    a = json.load(io.open(os.path.join(here, "a2", "banking_knowledge.specific.json"),
                          encoding="utf-8"))
    nd = None
    lst = None

    def walk(d):
        nonlocal nd, lst
        if isinstance(d, dict):
            for k, v in d.items():
                if k == "feedback_not_discoverable" and isinstance(v, str):
                    nd = v
                elif k == "feedback_registry_listing" and isinstance(v, str):
                    lst = v
                else:
                    walk(v)
        elif isinstance(d, list):
            for x in d:
                walk(x)
    walk(a)
    return nd, lst


def hit(resp):
    for tc in (resp.get("tool_calls") or []):
        nm = (tc.get("function") or {}).get("name") or tc.get("name") or ""
        raw = (tc.get("function") or {}).get("arguments") or tc.get("arguments") or ""
        s = raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=False)
        if TARGET_FAM in s or TARGET_FAM in nm:
            return "HIT"
    c = str(resp.get("content") or "")
    return "HIT" if TARGET_FAM in c else "MISS"


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 8
    nd, lst = a2_texts()
    if not (nd and lst):
        print("A2 문면 미발견 — 중단")
        return
    sims = [s for s in X.load(TAG)
            if s["task_id"] == TASK and s.get("reward_info") is not None]
    if not sims:
        print("궤적 없음")
        return
    sim = sims[0]
    # cut = 날조 발화가 커밋된 어시스턴트 턴(그 직전까지 재생)
    cut = None
    for i, m in enumerate(sim["messages"]):
        if m.get("role") == "assistant" and FAB in str(m.get("content") or ""):
            cut = i
            break
    if cut is None:
        print("컷 지점 없음")
        return
    # 레지스트리 = env 부재(오프라인)라 x283 상수가 아니라 **런 로그의 실측 목록**을 쓸 수 없어
    # 이름 나열은 unlock 결과들에서 아는 실명 + gold 도구로 재현 불가 — 대신 t7273 런에서
    # 폴백이 실제 나열했던 레지스트리 원소는 로그에 있다. 여기서는 A2 출시 규약과 동일하게
    # {names} 를 대화에 등장한 레지스트리 실명 집합으로 치운다(전부 _NNNN 접미사형).
    names = sorted({w.strip(".,()'\"") for m in sim["messages"]
                    for w in str(m.get("content") or "").split()
                    if w.strip(".,()'\"").rsplit("_", 1)[-1].isdigit()
                    and len(w.strip(".,()'\"")) > 8})
    names = [x for x in names if x != FAB]
    if TARGET_FAM + "_5829" not in names:
        names.append(TARGET_FAM + "_5829")
    listing = lst.replace("{names}", ", ".join(sorted(names)))
    a_txt = nd.replace("{name}", FAB)
    tools = U.tools_of(sim)
    P.TAG = TAG
    ours = P.our_lines(sim)
    print("073 t%s cut=%d · n=%d · names=%d개 · URL=%s\n" % (
        sim.get("trial"), cut, n, len(names), os.environ.get("T2_PROBE_URL", "localhost:8140")))
    for label, deny in (("A_LIVE", a_txt), ("B_LIST", a_txt + listing), ("D_NEG", a_txt + " " + NEG)):
        out = []
        for i, m in enumerate(sim["messages"][:cut]):
            r, c = m.get("role"), " ".join(str(m.get("content") or "").split())
            tcs = [(tc.get("function") or {}).get("name") or tc.get("name")
                   for tc in (m.get("tool_calls") or [])]
            if tcs:
                out.append("[%s calls] %s" % (r, ", ".join(tcs)))
            if c:
                out.append("[%s] %s" % (r, c[:600]))
            for t in ours.get(i, ()):
                out.append("[system] %s" % t[:1000])
        out.append("[assistant calls] call_discoverable_agent_tool(agent_tool_name=%s)" % FAB)
        out.append("[tool] %s" % deny)
        body = "\n".join(out)
        cnt = collections.Counter()
        for i in range(n):
            try:
                r = chat(body, tools, 0.0 if i == 0 else 0.7, 300)
            except Exception as e:
                r = {"content": "ERR %s" % type(e).__name__}
            cnt[hit(r)] += 1
        print("%-7s 문맥 %6d자 · HIT %d/%d" % (label, len(body), cnt["HIT"], n))
    print("\n※ 문턱(사전 고정): B≥6/8 ∧ A≤2/8 → 목록-동봉 확정. 미달 → A2 키 제거(종전 거동).")


if __name__ == "__main__":
    main()
