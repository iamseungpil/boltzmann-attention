# -*- coding: utf-8 -*-
"""v6 도메인-일반성 오프라인 전이 unit — 엔진 무수정·A2 스왑만 ([[11]] 키스톤 1단).

증명 대상: v6 신규 3종(READALL·L10·COV)이 banking/airline A2를 소비해
  (i) 작동(READALL 술어·COV 재료 빌드) (ii) 스펙 부재 시 우아한 침묵(무해 강등)
을 엔진 수정 0으로 달성하는가. 도메인 record는 실제 스키마 형태의 합성값.
"""
import sys, os, json, types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("T2_GATE_KINDS", "auth,confirm")

from t2_gate_patch import readall_unread, membership_violation, _cov_parse_ids, _cov_formalize_M
import t2_eplan_patch as ep


class Msg:
    def __init__(self, role, content, error=False, id=None, tool_calls=None):
        self.role, self.content, self.error = role, content, error
        self.id, self.tool_calls = id, tool_calls


def load_a2(domain):
    with open(os.path.join(HERE, "a2", "%s.gate.json" % domain), encoding="utf-8") as f:
        return json.load(f)


# ── 1. banking: READALL이 banking A2 eplan으로 작동 (엔진 무수정) ──
def test_banking_readall():
    a2 = load_a2("banking_knowledge")
    spec = a2["eplan"]
    assert spec["entity_key"] == "account_id"
    # 합성 banking 궤적: enumerator가 계좌 2개 나열·detail은 1개만 읽음
    enum_out = json.dumps({"accounts": [
        {"account_id": "acc_9001", "type": "credit"},
        {"account_id": "acc_9002", "type": "credit"}]})
    det_out = json.dumps({"account_id": "acc_9001", "transactions": [
        {"transaction_id": "tx_1", "amount": 42.0}]})
    tc_enum = types.SimpleNamespace(name=spec["list_enumerator"],
                                    arguments={"user_id": "u1"}, id="c1")
    tc_det = types.SimpleNamespace(name=spec["detail_reader"],
                                   arguments={"account_id": "acc_9001"}, id="c2")
    msgs = [Msg("user", "check both of my credit cards for issues"),
            Msg("assistant", None, tool_calls=[tc_enum]),
            Msg("tool", enum_out, id="c1"),
            Msg("assistant", None, tool_calls=[tc_det]),
            Msg("tool", det_out, id="c2")]
    led = ep.build_ledger_from_messages(msgs, spec, set())
    unread = readall_unread(led.listed, led.examined)
    assert unread == ["acc_9002"], (sorted(led.listed), sorted(led.examined), unread)
    print("PASS banking_readall (A2-스왑·엔진 무수정: unread=acc_9002)")


# ── 2. banking: L10은 items_id_path 부재 → 우아한 침묵 ──
def test_banking_l10_graceful():
    a2 = load_a2("banking_knowledge")
    spec = a2["eplan"]
    assert "items_id_path" not in spec
    d = {"account_id": "acc_9001", "item_ids": ["tx_1"]}
    msgs = [Msg("tool", json.dumps({"account_id": "acc_9001", "transactions": []}))]
    assert membership_violation(d, spec, msgs) is None, "spec 부재 = 침묵이어야"
    print("PASS banking_l10_graceful (items_id_path 미선언 → no-op)")


# ── 3. banking: COV 재료(record 요약·id 파싱)가 banking record로 작동 ──
def test_banking_cov_materials():
    a2 = load_a2("banking_knowledge")
    spec = a2["eplan"]
    r1 = json.dumps({"account_id": "acc_9001", "status": "active"})
    r2 = json.dumps({"account_id": "acc_9002", "status": "active"})
    msgs = [Msg("user", "dispute the bad charges on both cards"),
            Msg("tool", r1), Msg("tool", r2)]
    # 격리 서브콜 스텁: 두 계좌 다 커버라고 응답
    la = types.SimpleNamespace(generate=lambda **kw: types.SimpleNamespace(
        content='{"ids": ["acc_9001", "acc_9002"]}'))
    agent = types.SimpleNamespace(llm="m", llm_args={})
    class UM:
        def __init__(self, role=None, content=None):
            self.role, self.content = role or "user", content
    M = _cov_formalize_M(agent, la, UM, msgs, spec, a2)
    assert M == ["acc_9001", "acc_9002"], M
    # 발명 id 차단도 도메인 무관
    assert _cov_parse_ids('{"ids": ["acc_9001", "acc_FAKE"]}', {"acc_9001", "acc_9002"}) == ["acc_9001"]
    print("PASS banking_cov_materials (M 산출 재료·grounded 교집합)")


# ── 4. airline: eplan A2 저작 후 동일 엔진으로 작동 ──
def test_airline_readall():
    a2 = load_a2("airline")
    spec = a2.get("eplan")
    assert spec, "airline eplan 미저작 — 본 테스트 전에 A2 추가 필요"
    enum_out = json.dumps({"user_id": "u1", "reservations": [
        {"reservation_id": "RSV001"}, {"reservation_id": "RSV002"},
        {"reservation_id": "RSV003"}]})
    det_out = json.dumps({"reservation_id": "RSV002", "flights": [
        {"flight_number": "HAT001", "date": "2024-05-01"}]})
    tc_enum = types.SimpleNamespace(name=spec["list_enumerator"],
                                    arguments={"user_id": "u1"}, id="c1")
    tc_det = types.SimpleNamespace(name=spec["detail_reader"],
                                   arguments={"reservation_id": "RSV002"}, id="c2")
    msgs = [Msg("user", "cancel my basic economy reservations"),
            Msg("assistant", None, tool_calls=[tc_enum]),
            Msg("tool", enum_out, id="c1"),
            Msg("assistant", None, tool_calls=[tc_det]),
            Msg("tool", det_out, id="c2")]
    led = ep.build_ledger_from_messages(msgs, spec, set())
    unread = readall_unread(led.listed, led.examined)
    assert unread == ["RSV001", "RSV003"], (sorted(led.listed), sorted(led.examined), unread)
    print("PASS airline_readall (A2-스왑·엔진 무수정: unread=RSV001,RSV003)")


# ── 5. airline: eplan 자체가 없던 상태(=구 airline A2)의 우아한 침묵 ──
def test_no_eplan_graceful():
    assert readall_unread(set(), set()) == []
    assert membership_violation({"x": 1}, {}, []) is None
    print("PASS no_eplan_graceful (spec 없음 = 전 가드 침묵)")


if __name__ == "__main__":
    test_banking_readall()
    test_banking_l10_graceful()
    test_banking_cov_materials()
    test_airline_readall()
    test_no_eplan_graceful()
    print("ALL PASS (5/5) - v6 엔진 무수정, A2 스왑 소비 실증(오프라인)")
