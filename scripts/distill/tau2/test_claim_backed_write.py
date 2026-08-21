# -*- coding: utf-8 -*-
"""050형 재현 검정 (2026-08-21·T7335 halfB): 성공한 approve가 unbacked로 오판되지 않는다.

무엇을 막는 검정인가 (정본 `T7335_NT1_FORENSIC_HALFB_2026_08_21.md` task_050):
  approve/submit이 실행 원장에 실재하는데 CLAIMPROV가 unbacked=2를 내고, 그 거짓 피드백
  (*"the conversation ledger shows NO such event ... Either actually do it now"*)을 모델이
  문자대로 따라 **같은 승인을 재호출 = DUP 변이**를 우리가 제조했다([[25]]·[[64]]).
  코드상 그 판정에 도달하는 유일한 경로 = tool-지목이 원장 이름 집합 밖일 때
  kind-색인·센티널을 보지 않고 즉시 미입증하던 분기 → 수리 = `kind_fallback_on_miss`
  (과거형 claims 한정) + A2 event_map **완결 저작**(레지스트리 기계 도출·gold 미참조).

라이브 재료를 그대로 잰다([[03b]]): A2 = `gate_interpreter.load_domain_a2` 합성 claim_prov ·
레지스트리 = `tau2_domain_toolnames.json`. 오프라인 전용. 실행: py -3 test_claim_backed_write.py
"""
import io
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import gate_interpreter as GI                                  # noqa: E402
import t2_gate_patch as G                                      # noqa: E402

FAILED = []


def chk(cond, label):
    print(("  OK   " if cond else "  FAIL ") + label)
    if not cond:
        FAILED.append(label)


class _TC:
    def __init__(self, name, arguments=None):
        self.name, self.arguments = name, (arguments or {})


class _Msg:
    def __init__(self, tool_calls=None, role="assistant", content=""):
        self.tool_calls, self.role, self.content = (tool_calls or []), role, content


def _evs_of(msgs):
    """호출부(:agent_claimprov)와 동일 규약: 원명 + effective명(디스패처 unwrap·접미 strip)."""
    evs = set()
    for m in msgs:
        for tc in (getattr(m, "tool_calls", None) or []):
            evs.add(str(getattr(tc, "name", "") or ""))
            evs.add(G._eff_tool_name(tc))
    return evs


def main():
    a2 = GI.load_domain_a2("banking_knowledge")
    emap = (a2.get("claim_prov") or {}).get("event_map") or {}
    chk(bool(emap), "라이브 A2 합성에 claim_prov.event_map 존재")

    # ── 050 원장 재현: verify 체인 + read 체인 + submit/approve (디스패처 경유) ──
    msgs_050 = [
        _Msg(tool_calls=[_TC("get_user_information_by_name", {"customer_name": "X"})]),
        _Msg(tool_calls=[_TC("verify_identity", {}), _TC("log_verification", {})]),
        _Msg(tool_calls=[_TC("call_discoverable_agent_tool",
                             {"agent_tool_name": "check_cli_eligibility_8412"})]),
        _Msg(tool_calls=[_TC("call_discoverable_agent_tool",
                             {"agent_tool_name": "submit_credit_limit_increase_request_7392"})]),
        _Msg(tool_calls=[_TC("call_discoverable_agent_tool",
                             {"agent_tool_name": "approve_credit_limit_increase_5847"})]),
    ]
    evs = _evs_of(msgs_050)

    # [①] 050 재현 — tool-지목이 원장 밖(철자 혼성)이어도 실행된 write는 unbacked가 아니다.
    #     (라이브 로그 실측: 1차 평가 unbacked=2 → 2차 평가 같은 원장·같은 주장에 0 — 판정 뒤집힘.
    #      지목이 정확하면 구제되던 것이 지목이 빗나가면 즉시 미입증이 되던 분기가 원인.)
    cl_050 = [
        {"kind": "record_update", "what": "approved the credit limit increase",
         "tool": "approve_credit_limit_increase_request"},          # 혼성 지목(원장 밖)
        {"kind": "record_update", "what": "submitted the CLI request",
         "tool": "submit_credit_limit_increase"},                    # 축약 지목(원장 밖)
    ]
    unb = G._claim_unbacked(cl_050, emap, evs, msgs_050, a2, kind_fallback_on_miss=True)
    chk(unb == [], "①050 재현: 성공한 approve/submit는 지목이 빗나가도 unbacked가 아니다")

    # [②] 완결 저작이 실효인가 — 센티널 없이도 계열-접두(approve_/submit_)로 착지한다.
    emap_nosent = dict(emap)
    emap_nosent["record_update"] = [p for p in emap["record_update"]
                                    if p != "__effective_write__"]
    unb = G._claim_unbacked(cl_050, emap_nosent, evs, msgs_050, a2, kind_fallback_on_miss=True)
    chk(unb == [], "②센티널 제거 후에도 approve_/submit_ 접두로 입증 (event_map 완결)")

    # [③] 무지목 주장(kind 색인 경로)도 입증된다.
    cl_notool = [{"kind": "record_update", "what": "approved the credit limit increase"}]
    chk(G._claim_unbacked(cl_notool, emap_nosent, evs, msgs_050, a2) == [],
        "③무지목·kind=record_update: approve_ 접두로 입증")

    # [④] 음성통제([[57]]): write가 실행된 적 없으면 같은 주장은 그대로 잡힌다 — 날조 탐지 생존.
    msgs_read = [_Msg(tool_calls=[_TC("get_credit_limit_increase_history_4829", {})])]
    unb = G._claim_unbacked(cl_050, emap, _evs_of(msgs_read), msgs_read, a2,
                            kind_fallback_on_miss=True)
    chk(len(unb) == 2, "④음성통제: write 0인 원장에서는 두 주장 다 미입증 (레버 생존)")

    # [⑤] pending(미래형) 보존: 강등은 기본 OFF — 미이행 약속은 무관 write가 있어도 잡힌다.
    pend = [{"kind": "record_update", "what": "will file the dispute",
             "tool": "file_credit_card_transaction_dispute_4829"}]
    chk(len(G._claim_unbacked(pend, emap, evs, msgs_050, a2)) == 1,
        "⑤pending 보존: 지목 미스 약속은 기본 경로에서 그대로 미입증 (038형 탈출 방어)")

    # [⑥] 완결 저작 감사: 레지스트리 전수의 mutating 도구가 event_map 어느 kind에든 걸린다.
    #     기계 규칙(A2 `_note_event_map_completion_2026_08_21`와 동일·gold 미참조):
    #     읽기 접두(엔진 `_READ_PREFIX_RE`+query_)·디스패처(call_/unlock_/list_)·verify 쌍·
    #     env 플레이스홀더(example_) 제외 — 나머지 전부가 대상.
    reg = json.load(io.open(os.path.join(HERE, "tau2_domain_toolnames.json"),
                            encoding="utf-8"))["banking_knowledge"]
    pats = []
    for k, spec in emap.items():
        for p in (spec if isinstance(spec, list) else [spec]):
            if p != "__effective_write__":
                pats.append(p)
    skip_re = re.compile(r"^(get|search|list|lookup|find|retrieve|read|view|check|query"
                         r"|call|unlock|example)_")
    uncovered = []
    for name in reg:
        base = re.sub(r"_\d+$", "", name)
        if skip_re.match(base) or base in ("verify_identity", "log_verification", "shell"):
            continue
        if not any(base.startswith(p) or name.startswith(p) for p in pats):
            uncovered.append(name)
    chk(uncovered == [], "⑥레지스트리 완결: 미커버 mutating 도구 0 (%s)" % (uncovered[:4],))

    # [⑦] transfer-류 완결이 창 판정에도 실린다 (_is_transfer_call = event_map.transfer 재사용).
    am_init = _Msg(tool_calls=[_TC("initial_transfer_to_human_agent_0218", {})])
    am_emg = _Msg(tool_calls=[_TC("emergency_credit_bureau_incident_transfer_1114", {})])
    chk(G._is_transfer_call(am_init, emap) and G._is_transfer_call(am_emg, emap),
        "⑦이관-류 호출(초기/비상 변형)에도 최후 감사 창이 열린다")

    print("\n%s  (%d 실패)" % ("PASS" if not FAILED else "FAIL", len(FAILED)))
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
