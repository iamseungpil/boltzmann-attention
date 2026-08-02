#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""t2_surface_bus: **표면화 버스** v0 — 모든 노트/문구 부착의 단일 출구.

정본 = `LEVER_CONSOLIDATION_DESIGN_2026_08_02` §2b (2026-08-02 리뷰 L2 반영판).
레버는 `register()`로 문구를 등록만 하고, 부착은 `flush()`가 **불변식 4종을 한 번** 집행하며 수행한다:

  ①replay 불변 — env가 mutating으로 보는 도구 출력에는 부착 금지(§2at 일반화·사고 3회의 종결).
    판정자는 호출자가 주입(기존 `_dedup_cache_safe`와 동일 술어·이 모듈은 도구/도메인을 모른다).
  ②정직 불변 — 문구가 전제하는 상태를 선언형으로 받는다(`requires`). 검사 실패 = 부착 거부 + beat.
    v0 검사 = 호출자가 넘긴 상태 dict와의 **닫힌 대조**만(의미 판단 0).
  ③전역 예산 — (sim, channel, text-head) 당 상한(기본 2·`T2_AXIS_NOTE_CAP` 승계).
  ④순서 — 채널 우선순위 고정: deny_reason > correction > guidance > mark.

★L2: fail-open은 **guidance·mark만**. `deny_reason`은 버스 실패 시 **최소 고정 문구로 fail-closed**
  (사유 없는 거부 = D5 동형을 버스가 자가 생산하지 않기 위해).

플래그: 채널별 이관 — v0은 axis notes만(`T2_SURFACE_BUS=1`). OFF = 기존 직접 부착 그대로.
beat: 부착/거부 모두 `[T2_LEVER] T2_SURFACE_BUS <채널>`로 증명(x44 편입).
"""
import os
import sys

_ORDER = {"deny_reason": 0, "correction": 1, "guidance": 2, "mark": 3}
_FAILSAFE_DENY = ("Error: [POLICY GATE] this call was denied; reason unavailable — "
                  "do not retry the same call")


class SurfaceBus(object):
    """orchestrator(시뮬)당 1개. 무상태 레버들이 등록하고, 결과 조립 시점에 flush."""

    def __init__(self, note_cap=None):
        try:
            self._cap = int(note_cap if note_cap is not None
                            else os.environ.get("T2_AXIS_NOTE_CAP", "2") or 2)
        except Exception:
            self._cap = 2
        self._fired = {}          # (channel, head) -> count  (시뮬 수명)
        self._pending = []        # [(channel, text, requires)]

    # ── 레버 쪽 API ──────────────────────────────────────────────────────────
    def register(self, channel, text, requires=None):
        """문구 등록만 — 부착하지 않는다. channel ∈ _ORDER(밖이면 mark로 강등·정직하게 beat)."""
        if not text:
            return
        ch = channel if channel in _ORDER else "mark"
        self._pending.append((ch, str(text), dict(requires or {})))

    # ── 버스 집행 ────────────────────────────────────────────────────────────
    def flush(self, attach_ok, state=None):
        """불변식 집행 후 부착할 문구 목록을 순서대로 반환(부착 자체는 호출자 — 메시지 소유권).

        attach_ok: bool — ①replay 불변(호출자가 기존 술어로 계산: 이 대상에 부착해도 되는가)
        state: dict — ②정직 불변 대조용(문구의 requires 키가 전부 state에서 참인가)
        반환: [text, ...] (④순서 적용·③예산 통과분만)
        """
        out, pend = [], self._pending
        self._pending = []
        try:
            pend.sort(key=lambda x: _ORDER.get(x[0], 9))
            for ch, text, req in pend:
                if not attach_ok:
                    # ★L2: deny-사유만 fail-closed — 나머지는 무부착 통과(fail-open)
                    if ch == "deny_reason":
                        out.append(_FAILSAFE_DENY)
                        self._beat(ch, "failclosed")
                    else:
                        self._beat(ch, "skipped_replay")
                    continue
                if state is not None and any(not state.get(k2) for k2 in req):
                    self._beat(ch, "honesty_reject")     # ②전제 불성립 = 부착 거부
                    continue
                key = (ch, text[:60])
                self._fired[key] = self._fired.get(key, 0) + 1
                if self._fired[key] > self._cap:          # ③예산
                    self._beat(ch, "budget")
                    continue
                out.append(text)
                self._beat(ch, "attached")
        except Exception as e:                            # 버스 자체 결함 = fail-open(+deny는 위에서)
            print("[T2_SURFACE_BUS] flush 실패(무부착 통과): %r" % (e,),
                  file=sys.stderr, flush=True)
        return out

    def _beat(self, ch, what):
        try:
            from t2_lever_beat import beat
            beat("T2_SURFACE_BUS", "%s:%s" % (ch, what))
        except Exception:
            pass


def get_bus(orch):
    """orchestrator에 버스 1개(시뮬 수명 — 예산이 sim 단위)."""
    b = getattr(orch, "_t2_surface_bus", None)
    if b is None:
        b = orch._t2_surface_bus = SurfaceBus()
    return b
