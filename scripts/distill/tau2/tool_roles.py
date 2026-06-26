#!/usr/bin/env python
"""tool_roles — 도구 *속성*서 게이트 role 도출 (A2_GENERALIZATION_DESIGN S1·도메인-일반·엔진).
명명규약 + openai_schema 인자 + .returns(pydantic) 필드로 도출 → A2 tool-name 하드리스트 제거.
도출(클린): write·user_scoped·auth·handoff·owner_path(owned-entity=detail-getter 반환에 owner_field).
잔여(이름만으론 불가·도구행동 의존): precond_status (modify_pending_order_address도 'pending' 포함하나 미검사
  → 이름파싱=false-block). = 최소 A2 또는 behavior-probe로(여기 미포함).
검정: roles(retail/airline) == 현 gate.json applies_to/satisfiers (test_tool_roles).
"""
import re
READ_PREFIX = ("get_", "find_", "list_", "search_", "calculate")
WRITE_PREFIX = ("modify_", "cancel_", "exchange_", "return_", "book_", "update_", "send_",
                "change_", "open_", "close_", "submit_", "apply_", "deposit_", "activate_",
                "freeze_", "unfreeze_", "order_", "reset_", "pay_", "approve_", "deny_",
                "give_", "file_", "set_", "log_", "clear_", "request_", "unlock_")
OWNER_FIELD = "user_id"


def _args(t):
    sch = getattr(t, "openai_schema", {}) or {}
    return set((sch.get("function", {}).get("parameters", {}).get("properties") or {}).keys())


import typing


def _mf(tp):
    mf = getattr(tp, "model_fields", None)
    return set(mf.keys()) if mf else set()


def _return_fields(t):
    """.returns = wrapper 모델(단일 'returns' 필드, 실제타입=그 annotation). unwrap(Optional/List 포함)."""
    r = getattr(t, "returns", None)
    mf = getattr(r, "model_fields", None)
    if not mf:
        return set()
    if "returns" in mf:                          # wrapper → 실제 반환타입 추출
        ann = mf["returns"].annotation
        for cand in [ann] + list(typing.get_args(ann)):
            f = _mf(cand)
            if f:
                return f
        return set()
    return set(mf.keys())


def roles(tools, owner_field=OWNER_FIELD):
    names = list(tools)
    read = lambda n: n.startswith(READ_PREFIX)
    write = {n for n in names if n.startswith(WRITE_PREFIX) and not read(n)}
    handoff = {n for n in names if "transfer_to_human" in n}
    # auth/identity-producer: 식별자(user_id) 생산 read 도구 (retail find_user_id_*; airline=∅)
    auth = {n for n in names if read(n) and owner_field in n}
    # owned-entity: get_<e>_details.returns 에 owner_field 有 (order=owned·product=catalog 자동구분)
    owned_ids = set()
    for n in names:
        m = re.match(r"get_(\w+)_details$", n)
        if m and owner_field in _return_fields(tools[n]):
            owned_ids.add(m.group(1) + "_id")
    user_scoped = {n for n in names if (owner_field in _args(tools[n])) or (_args(tools[n]) & owned_ids)}
    owner_path = {}
    for n in user_scoped:
        eid = next((x for x in _args(tools[n]) if x in owned_ids), None)
        if eid:
            owner_path[n] = [eid, "get_" + eid[:-3] + "_details", owner_field]
    return dict(write=write, user_scoped=user_scoped, auth=auth, handoff=handoff,
                owned_ids=owned_ids, owner_path=owner_path)
