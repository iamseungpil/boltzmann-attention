# -*- coding: utf-8 -*-
"""비커밋 피드백 사이드카 기록 (MT_PROBE_DESIGN §1-d · 2026-07-30 야간).

★왜 필요한가 (실측 근거):
게이트/해소 피드백 중 일부는 **작업버퍼(`work`)에만 붙고 `state.messages`에는 커밋되지 않는다**
(`t2_gate_patch.py:4558` "채널 절대규칙(1849·replay-clean)"). 그 위생 자체는 옳다 — C208에서
mutating 도구 출력에 덧붙였다가 tau2 evaluation replay를 깨뜨린 사고의 반작용이다.

그런데 부작용이 하나 있다: **그 피드백은 궤적에 흔적이 없다.** 실측(같은 런 대조):

    discovery-required   stderr  day6A 22 · day8A 25 · day9cA 29   →  궤적 0 · 0 · 0
    [DUPLICATE-READ]     stderr  0                                 →  궤적 19 · 11

⇒ 비커밋 채널로 나간 것은 **메시지-수준 포렌식이 원천적으로 불가능**하다(집계만 남는다).
[[08]]("결론 전 전수 궤적 포렌식")이 작동할 수 없는 상태이므로, **궤적을 건드리지 않고**
별도 파일에 발화 사실만 남긴다.

★설계 제약 3가지:
  1. **궤적 불변** — 이 모듈은 messages를 절대 만지지 않는다(replay 위생 유지).
  2. **기본 no-op** — `T2_FB_SIDECAR` 미설정이면 아무것도 하지 않는다(행동 변화 0).
  3. **닫힌 술어만 기록** — 채널·이유 코드·길이·해시. 산문 내용 판정은 하지 않는다([[22]]).
     본문 저장은 `T2_FB_SIDECAR_TEXT=1`일 때만(기본 = 해시+길이만).

sim 식별: tau2 sim id는 이 지점에서 안 보인다. 대신 **첫 유저 발화의 해시**를 지문으로 쓴다
(한 sim 안에서 불변·sim 간 구분·도메인 일반). 턴 = 커밋된 메시지 수.

용법(측정 시): `export T2_FB_SIDECAR=/path/fb.jsonl` 후 평소대로 런.
"""
import hashlib
import json
import os
import sys
import threading

_LOCK = threading.Lock()
_WARNED = False


def _sim_key(messages):
    """첫 유저 발화 해시 = sim 지문(도메인-일반·엔진 내부 id 불요)."""
    for m in messages or []:
        if getattr(m, "role", None) == "user" or (isinstance(m, dict) and m.get("role") == "user"):
            c = getattr(m, "content", None)
            if c is None and isinstance(m, dict):
                c = m.get("content")
            if isinstance(c, str) and c.strip():
                return hashlib.sha1(c.strip().encode("utf-8")).hexdigest()[:12]
    return "nouser"


def record(kind, text, messages=None, **meta):
    """비커밋 피드백 1건 기록. 실패해도 **절대 예외를 올리지 않는다**(런을 깨지 않기 위해).

    kind  = 채널/레버 이름(예: 'action-required', 'have_value', 'discovery-required')
    text  = 모델에게 간 문구(기본은 해시·길이만 저장)
    meta  = 닫힌 부가 정보(target 도구명·reason 코드 등)
    """
    path = os.environ.get("T2_FB_SIDECAR")
    if not path:
        return
    global _WARNED
    try:
        s = "" if text is None else str(text)
        row = {
            "kind": str(kind),
            "sim": _sim_key(messages),
            "turn": len(messages or []),
            "len": len(s),
            "sha": hashlib.sha1(s.encode("utf-8")).hexdigest()[:12],
        }
        for k, v in meta.items():
            if isinstance(v, (str, int, float, bool)) or v is None:
                row[k] = v
        if os.environ.get("T2_FB_SIDECAR_TEXT") == "1":
            row["text"] = s[:4000]
        line = json.dumps(row, ensure_ascii=False)
        with _LOCK:
            with open(path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
    except Exception as e:                       # 기록 실패가 런을 깨면 안 된다
        if not _WARNED:
            _WARNED = True
            print("[T2_FB_SIDECAR] 기록 실패(이후 무음): %r" % (e,), file=sys.stderr, flush=True)


def record_many(items, messages=None, **meta):
    """`fb` 리스트처럼 여러 건을 한 번에. items = [(kind, text), ...] 또는 메시지 객체 리스트."""
    for it in items or []:
        if isinstance(it, (tuple, list)) and len(it) == 2:
            record(it[0], it[1], messages, **meta)
        else:
            role = getattr(it, "role", None) or (it.get("role") if isinstance(it, dict) else None)
            content = getattr(it, "content", None)
            if content is None and isinstance(it, dict):
                content = it.get("content")
            err = bool(getattr(it, "error", False))
            record("tool-deny" if role == "tool" and err else ("reminder-%s" % role),
                   content, messages, **meta)


if __name__ == "__main__":
    # ── selftest: no-op 기본 · 설정 시 기록 · 예외 무해 ─────────────────────────
    import tempfile

    class M:
        def __init__(self, role, content, error=False):
            self.role, self.content, self.error = role, content, error

    msgs = [M("system", "sys"), M("user", "I want to dispute a charge."), M("assistant", "ok")]

    os.environ.pop("T2_FB_SIDECAR", None)
    record("action-required", "some feedback", msgs)          # no-op 이어야 함
    print("1) 미설정 no-op: OK")

    p = os.path.join(tempfile.gettempdir(), "fb_sidecar_selftest.jsonl")
    if os.path.exists(p):
        os.remove(p)
    os.environ["T2_FB_SIDECAR"] = p
    record("action-required", "do the discovery chain", msgs, target="call_discoverable_agent_tool",
           reason="discovery-required")
    record_many([M("tool", "Error: resolve the flagged call first", error=True),
                 M("user", "[REMINDER] you have the value already")], msgs, arm="test")
    rows = [json.loads(l) for l in open(p, encoding="utf-8")]
    assert len(rows) == 3, rows
    assert rows[0]["kind"] == "action-required" and rows[0]["reason"] == "discovery-required"
    assert rows[0]["sim"] == rows[1]["sim"] == rows[2]["sim"], "sim 지문 불변이어야 함"
    assert rows[0]["turn"] == 3
    assert rows[1]["kind"] == "tool-deny" and rows[2]["kind"] == "reminder-user"
    assert "text" not in rows[0], "기본은 본문 미저장"
    print("2) 기록 3건 · sim 지문 불변 · 본문 미저장: OK")

    os.environ["T2_FB_SIDECAR_TEXT"] = "1"
    record("x", "with text", msgs)
    rows = [json.loads(l) for l in open(p, encoding="utf-8")]
    assert rows[-1].get("text") == "with text"
    print("3) TEXT=1 시 본문 저장: OK")

    os.environ["T2_FB_SIDECAR"] = "/nonexistent_dir_xyz/fb.jsonl"
    record("y", "should not raise", msgs)                      # 예외 없이 무음 경고
    print("4) 기록 실패 무해: OK")

    os.environ.pop("T2_FB_SIDECAR_TEXT", None)
    os.environ.pop("T2_FB_SIDECAR", None)
    os.remove(p)
    print("ALL PASS (4/4)")
