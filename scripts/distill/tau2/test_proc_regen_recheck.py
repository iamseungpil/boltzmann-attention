# -*- coding: utf-8 -*-
"""재생성이 낸 호출도 절차 게이트를 받는가 (A-1 · 2026-08-23).

`tasks__20260822/TASK_050.md` §7-① 이 확정한 결손: `_ap_regen` 이 낸 tool_call 은
`gate`·`T2_UNLOCK_NAME`·`T2_UNLOCK_PROV` 만 다시 받고 `T2_PROCEDURE` 는 **평가조차
받지 않은 채** 커밋됐다. t7346 `task_050` trial 0 이 그렇게 승인 호출을 먼저 커밋해
요청-제출 write 를 빠뜨렸고 DB 해시가 갈렸다(reward 0.0). 같은 sha·같은 A2 의 trial 1 은
동일 호출이 원본 am 에 있었기에 deny 를 받고 선행을 먼저 밟아 1.0 을 받았다.

이 검정이 잡는 것은 두 가지다.
  ① **술어는 이미 준비돼 있었다** — 실제 t7346 궤적을 그대로 먹여, 문제의 승인 호출에
     `t2_procedure.decide` 가 `deny missing=submit_request,disputes,pending_replacement`
     를 낸다는 것을 보인다. 즉 결손은 술어가 아니라 배선이었다.
  ② **배선이 그 자리에 있다** — `_ap_regen` 함수 본문(AST 로 그 함수만 떼어)에서
     절차 재평가·cap 공유·플래그 가드가 실재하는지 본다. `proc_fb` 死배선(2026-08-05)이
     정확히 이 자리에서 났고, 그때 배운 것은 *로그 마크 ≠ 전달*([[55]])이다.

⚠이 검정은 **라이브 전달을 증명하지 못한다** — 그것은 런 로그의 `[T2_PROCEDURE] regen-*`
  라인이 할 일이다. 여기서 증명하는 것은 술어와 배선의 실재까지다.
"""

import ast
import gzip
import io
import json
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import t2_gate_patch as G          # noqa: E402
import t2_procedure as P           # noqa: E402

fail = []


def check(name, ok, detail=""):
    print("  %-58s %s%s" % (name, "PASS" if ok else "FAIL", (" — " + detail) if detail else ""))
    if not ok:
        fail.append(name)


def objs(messages):
    """궤적 dict → 엔진이 보는 모양(속성 접근·tool_calls 는 name/arguments/id)."""
    out = []
    for m in messages:
        tcs = [types.SimpleNamespace(name=tc.get("name"), arguments=tc.get("arguments"),
                                     id=tc.get("id")) for tc in (m.get("tool_calls") or [])]
        out.append(types.SimpleNamespace(role=m.get("role"), tool_calls=tcs or None,
                                         content=m.get("content"), id=m.get("id"),
                                         tool_call_id=m.get("tool_call_id"),
                                         error=m.get("error", False)))
    return out


# ─────────────────────────────────────────────────────────────────────────────
print("① 술어 — 실제 t7346 task_050 궤적에서 그 승인 호출이 거부되는가")

RES = os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026", "sim_results",
                   "bank_t7346_halfB_20260822.results.json.gz")
A2 = json.load(io.open(os.path.join(HERE, "a2", "banking_knowledge.specific.json"),
                       encoding="utf-8"))
PROCS = A2.get("procedures")
check("A2 가 procedures 를 선언한다", bool(PROCS))

if os.path.exists(RES):
    with gzip.open(RES, "rt", encoding="utf-8") as f:
        d = json.load(f)

    def decide_at(sim, want_reward):
        """그 sim 의 '승인' 유효호출 시점마다 (msg_idx, verdict, missing)."""
        rows = []
        ms = sim["messages"]
        for i, m in enumerate(ms):
            for tc in (m.get("tool_calls") or []):
                c = types.SimpleNamespace(name=tc.get("name"), arguments=tc.get("arguments"),
                                          id=tc.get("id"))
                eff = G._exact_tool_name(c)
                # 표적 = 이 절차의 결정 노드 도구(선언에서 읽는다 · 리터럴 0)
                if eff not in DECISION_TOOLS:
                    continue
                hist = objs(ms[:i])
                ar = G._args_dict(c)
                also = {str(ar.get(k)) for k in
                        ("agent_tool_name", "user_tool_name", "discoverable_tool_name")
                        if ar.get(k)}
                dc = P.decide(PROCS, eff, ar, G._executed_tool_counts(hist), also_names=also,
                              unlocked=G._unlocked_names(hist, A2),
                              pattern=(A2.get("discoverable_name_check") or {}).get("pattern"))
                rows.append((i, dc.get("verdict"), tuple(dc.get("missing") or []),
                             (dc.get("notes") or [""])[0]))
        return rows

    # 결정 노드 도구를 **선언에서** 뽑는다 — 이 파일에 도구명 리터럴을 적지 않기 위해서다([[59]]).
    DECISION_TOOLS = set()
    for _pr in (PROCS or []):
        for _nd in (_pr.get("nodes") or []):
            if not (_nd.get("requires") or []):
                continue
            DECISION_TOOLS |= set(P._tools_of(_nd))
    check("선언에서 결정 노드 도구를 읽었다", bool(DECISION_TOOLS), str(sorted(DECISION_TOOLS))[:90])

    sims = [s for s in d["simulations"] if s.get("task_id") == "task_050"]
    check("t7346 halfB 에 task_050 sim 2건", len(sims) == 2, str(len(sims)))
    by_rw = {(s.get("reward_info") or {}).get("reward"): s for s in sims}

    fail_rows = decide_at(by_rw.get(0.0), 0.0) if 0.0 in by_rw else []
    pass_rows = decide_at(by_rw.get(1.0), 1.0) if 1.0 in by_rw else []

    denies = [r for r in fail_rows if r[1] == "deny"]
    check("reward 0.0 궤적: 결정 호출이 거부된다", bool(denies),
          str([(r[0], r[2]) for r in fail_rows]))
    if denies:
        miss = denies[0][2]
        check("거부 사유가 요청-제출 선행을 포함", len(miss) >= 1 and "submit" in " ".join(miss),
              ",".join(miss))
        note = denies[0][3]
        check("문면이 [PROCEDURE] 표면화 형식", note.lstrip().startswith("[PROCEDURE]"),
              note[:60])
    check("reward 1.0 궤적: 같은 도구가 통과한다(선행을 밟았으므로)",
          bool(pass_rows) and all(r[1] == "pass" for r in pass_rows),
          str([(r[0], r[1]) for r in pass_rows]))
else:
    print("  · 런 결과 gz 없음 — 술어 검정 skip (%s)" % RES)

# ─────────────────────────────────────────────────────────────────────────────
print("\n② 배선 — `_ap_regen` 본문에 그 재평가가 실재하는가")

SRC = io.open(os.path.join(HERE, "t2_gate_patch.py"), encoding="utf-8").read()
_tree = ast.parse(SRC)
_fn = next((n for n in ast.walk(_tree)
            if isinstance(n, ast.FunctionDef) and n.name == "_ap_regen"), None)
check("`_ap_regen` 함수를 AST 에서 찾았다", _fn is not None)

body = ast.get_source_segment(SRC, _fn) or "" if _fn is not None else ""
check("본문이 `t2_procedure` 를 import 한다", "import t2_procedure" in body)
check("본문이 `.decide(` 를 호출한다", ".decide(" in body)
check("`T2_PROCEDURE` 가 켜져 있을 때만 판다(거동보존 가드)",
      'os.environ.get("T2_PROCEDURE") == "1"' in body)
check("`T2_PROC_REGEN` 플래그로 부정통제가 가능하다",
      'os.environ.get("T2_PROC_REGEN"' in body)
check("cap 을 메인 경로와 공유한다(`_t2_proc_deny`)", "_t2_proc_deny" in body)
check("cap 상한 이름도 공유한다(`T2_PROCEDURE_CAP`)", "T2_PROCEDURE_CAP" in body)
check("거부 문면 접두 규칙이 메인 경로와 같다",
      'startswith("Error:")' in body and '"Error: "' in body)
check("실패 시 원본을 유지하는 경로가 있다", "keeping original" in body)

# ★도메인 리터럴 0([[05]]/[[59]]) — 새 블록이 도구명·필드값을 적으면 A2 가 아니라 엔진이
#   도메인을 아는 것이 된다. 디스패처 인자 키 3종은 메인 절차 게이트가 이미 쓰는 프레임워크
#   어휘라 같은 예외로 둔다(그 셋 말고 밑줄+숫자 접미사 이름이 있으면 잡는다).
import re  # noqa: E402
_newblk = body[body.find("A-1 절차 재평가"):] if "A-1 절차 재평가" in body else ""
_code_only = "\n".join(l for l in _newblk.split("\n") if not l.lstrip().startswith("#"))
_suffixed = sorted(set(re.findall(r"\"[a-z_]+_[0-9]{3,}\"", _code_only)))
check("새 블록에 접미사-도구명 리터럴 0", not _suffixed, str(_suffixed))

print("\nRESULT: %s" % ("ALL PASS" if not fail else "FAIL %s" % fail))
sys.exit(1 if fail else 0)
