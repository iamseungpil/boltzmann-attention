# -*- coding: utf-8 -*-
r"""x553 — A-3 선행 측정: 재무장 술어 ⑵에서 **assistant 를 빼면 무엇이 죽나**

## 무엇을 재나 (사용자 지시 2026-08-26 · 정본 `TASK_055.md` §7 A-3)
`_rearm_subjects` 술어 ⑵ 는 배달 이후 창에서 계열 표시명을 찾을 때 지금 **user 와 assistant 를
둘 다** 본다(`t2_gate_patch.py:3953`). A-3 = assistant 를 뺀다 — 에이전트가 제 발화에서 계열명을
한 번 부른 것만으로 축이 다시 열리는 **자기확증**을 없애자는 처방이다.

사는 것은 *"발화가 몇 건 줄어드나"* 가 아니다. 문서가 **±필수**로 못박았다([[70]]) — 죽는 발화
가운데 **reward 1.0 sim 의 발화**가 있으면 A-3 는 순손실 후보이고, 전부 reward 0 sim 이면 공짜다.

## 어떻게 재나 — 술어를 다시 짜지 않는다([[67]] 사본 금지)
정본 `_rearm_subjects` 를 **같은 궤적에 두 번** 부른다. 두 팔의 차이는 인자 하나뿐이다.

    as_is   궤적 그대로                              ← 라이브가 한 것
    A3      assistant 메시지의 content 만 None 으로   ← 술어 안의 역할 필터가 content 없는
            (역할·인덱스는 보존)                       메시지를 건너뛰므로 `roles=("user",)` 과
                                                      **거동 동치**다. 인덱스를 보존하는 이유는
                                                      창이 `messages[served_at:]` 인덱스 슬라이스라서.

`served`(기배달 계열)와 `served_at`(창 시작)은 **로그가 축자로 준다**:
  · `served`   = `[T2_SEARCH_REARM] … (기배달 X,Y)`
  · `served_at`= 그 sim·그 군의 **직전 배달** 줄 `[T2_SEARCH_AGENT] group=… turn=N` 의 N.
    `_record_served` 가 적는 값과 그 print 의 값이 같은 표현식(`len(messages or [])`)이고 둘
    사이에 `messages` 를 늘리는 자리가 없다(`t2_gate_patch.py:4195`↔`:4306`/`:4341`).

## ⛔재현 게이트 ([[62]] 2b · [[78]] · 오늘 x551·x552 가 두 번 고장난 자리)
에이전트가 본 `messages` 와 **영속 궤적**이 같은 리스트라는 보장은 없다(`_ap_regen` 이 원
메시지를 교체한다·[[30]]). 그래서 as_is 재생이 로그의 관측 신규 계열을 **그대로 재현하지 못하면
그 발화는 판정하지 않는다** — `[?]` 로 남기고 분모에서 뺀다. 계기가 못 재는 것을 0 으로 읽지
않는다([[25]] 모르면 안 뺀다).

## 부수 측정 — *"손님이 아예 부른 적이 있나"*
같은 정본 술어를 `served_at=0` 로 한 번 더 부른다(user-only). 창 밖(배달 이전)에서라도 손님이
그 계열을 부른 적이 있으면 A-3 의 손실은 **창 경계 때문**이고, 아예 없으면 그 발화는 처음부터
에이전트 자기발화뿐이었다는 뜻이다. 판정용이 아니라 원인 구분용이다([[77]] ③).

## [[62]] 4문
  ① 결손 측정 — 이 스크립트가 재는 것이 그 측정이다(레버를 짓지 않는다·플래그 0·거동 0).
  ② 격리 ↔ 라이브 — 재현 게이트가 둘의 차이를 **먼저 인쇄**한다.
  ③ 사라지는 모델 판단 0 — 읽기 전용. 궤적도 gold 도 건드리지 않는다([[23]] `reward_info.reward`
     한 칸만 ± 표에 쓴다·선택 기준으로 쓰지 않는다).
  ④ 순위·최댓값·지목 문장 0.

무료: 영속 궤적과 로그만 읽는다. 모델 호출 0 · GPU 0.

    usage:  x553_rearm_role_split.py [--tags GLOB] [--recent N] [--json OUT]
"""

import argparse
import fnmatch
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import t2_forensic as F                                                  # noqa: E402
import t2_gate_patch as G                                                # noqa: E402

RX_DELIVER = re.compile(
    r"\[sim=(?P<sim>[^\]]+)\] \[T2_SEARCH_AGENT\] group=(?P<g>\S+) · 문서 \d+.*? turn=(?P<turn>\d+)")
RX_REARM_NEW = re.compile(
    r"\[sim=(?P<sim>[^\]]+)\] \[T2_SEARCH_REARM\] group=(?P<g>\S+) 신규 대상 (?P<new>\S+)"
    r" \(기배달 (?P<served>[^)]*)\)")
RX_REARM_DEL = re.compile(
    r"\[sim=(?P<sim>[^\]]+)\] \[T2_SEARCH_REARM\] group=(?P<g>\S+) 델타 배달 \d+자"
    r" \(문서 \d+·뺀 것 \d+\) turn=(?P<turn>\d+)")


class _M(object):
    """정본 술어가 보는 두 속성만 갖는 메시지 껍데기 (`role`·`content`)."""
    __slots__ = ("role", "content")

    def __init__(self, role, content):
        self.role = role
        self.content = content


class _Agent(object):
    """`_rearm_subjects` 가 읽는 두 속성만 갖는 에이전트 껍데기."""

    def __init__(self, served, served_at):
        self._t2_search_served = served
        self._t2_search_served_at = served_at


def _csv(s):
    s = (s or "").strip()
    return [] if s in ("", "-") else [x for x in (p.strip() for p in s.split(",")) if x]


def firings_in_log(text):
    """로그 전문 → 재무장 **발화** 목록(발생 순서대로).

    `served_at` 은 그 (sim, 군)의 **직전 배달** turn 이다. 재무장 자신의 배달 줄은 신규-대상 줄
    **뒤에** 찍히므로 순차 처리만으로 옛 값이 잡힌다. 직전 배달이 로그에 없으면 `None` 으로
    남긴다 — 순서로 추정하지 않는다([[25]]).
    """
    last_deliver, out, pending = {}, [], {}
    for ln in (text or "").splitlines():
        if "[T2_SEARCH_REARM]" in ln:
            m = RX_REARM_NEW.search(ln)
            if m:
                ev = {"sim": m.group("sim"), "group": m.group("g"),
                      "new": _csv(m.group("new")), "served": _csv(m.group("served")),
                      "served_at": last_deliver.get((m.group("sim"), m.group("g"))),
                      "fire_turn": None}
                out.append(ev)
                pending[(ev["sim"], ev["group"])] = ev
                continue
            m = RX_REARM_DEL.search(ln)
            if m:
                ev = pending.pop((m.group("sim"), m.group("g")), None)
                if ev is not None:
                    ev["fire_turn"] = int(m.group("turn"))
                continue
        m = RX_DELIVER.search(ln)
        if m:
            last_deliver[(m.group("sim"), m.group("g"))] = int(m.group("turn"))
    return out


def gold_write_after(sim, fire_turn):
    """★반증 술어([[77]] ③) — 그 sim 의 **gold write** 가 발화 시점 **이후**에도 있었나.

    reward 는 sim 단위라 *"통과 sim 에서 죽는 발화"* 는 순손실 **후보**일 뿐이다. 발화보다
    **앞에서** 이미 모든 gold write 가 끝났다면 그 발화는 통과의 원인일 수 없다 — (+) 귀속이
    거짓이다. 닫힌 술어(도구 이름 일치 + 인덱스 비교)이고 gold 는 **진단용으로만** 읽는다([[23]]).
    반환 True = 반증 못 함(순손실 후보로 남는다) · False = 반증됨.
    """
    want = {(g.get("action") or {}).get("name")
            for g in (F.gold_actions(sim) or []) if g.get("tool_type") == "write"}
    want.discard(None)
    if not want:
        return None
    for i, m in enumerate((sim.get("messages") or [])):
        if i < int(fire_turn):
            continue
        for tc in (m.get("tool_calls") or ()):
            nm = (tc.get("function") or {}).get("name") or tc.get("name")
            if nm in want:
                return True
    return False


def replay(po, msgs, group, served, served_at):
    """정본 술어 1회 호출 → 신규 계열 집합(없으면 빈 집합)."""
    ag = _Agent({group: set(served or ())}, {group: int(served_at or 0)})
    g, new = G._rearm_subjects(ag, po, [group], {group}, msgs)
    return set(new or ()) if g else set()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default="*", help="태그 glob (기본 = 발화가 있는 전량)")
    ap.add_argument("--recent", type=int, default=0, help="최근 N 개 태그만")
    ap.add_argument("--domain", default="banking_knowledge")
    ap.add_argument("--json", default="")
    a = ap.parse_args()

    import gate_interpreter as GI
    po = (GI.load_domain_a2(a.domain) or {}).get("policy_ontology") or {}
    if not (po.get("doc_index")):
        print("A2 `policy_ontology.doc_index` 없음 — 잴 수 없다", file=sys.stderr)
        return 2

    # ★파일명이 두 가지다(`.results.json.gz` / `_results.json.gz`) — `path_for(tag, suffix)` 는
    #   suffix 를 고정으로 붙여 **점 명명을 통째로 놓친다**. 그래서 태그가 아니라 `all_result_files`
    #   가 준 **경로 그대로**를 넘긴다(`path_for` 는 실재 경로를 그대로 받는다). 첫 판이 이 자리에서
    #   85/85 를 `NO_TRAJ` 로 뱉었다 — 재현 게이트가 아니었으면 *"발화가 하나도 없다"* 로 읽혔다.
    files = {}
    for p in F.all_result_files():
        files.setdefault(F.tag_of_file(p), p)
    tags = [t for t in files if fnmatch.fnmatch(t, a.tags)]
    tags.sort(key=lambda t: os.path.getmtime(files[t]))
    if a.recent:
        tags = tags[-a.recent:]

    rows, scanned = [], []
    for tag in tags:
        try:
            text = F.log_text(tag)
        except Exception:
            continue
        if "[T2_SEARCH_REARM]" not in (text or ""):
            continue
        evs = firings_in_log(text)
        if not evs:
            continue
        scanned.append((tag, len(evs)))
        try:
            by_sim = {F.simtag(s): s for s in F.sims(files[tag])}
        except Exception as e:
            print("[warn] %s: 궤적 로드 실패 %r" % (tag, e), file=sys.stderr)
            by_sim = {}
        for ev in evs:
            sim = by_sim.get(ev["sim"])
            r = dict(ev, tag=tag)
            if sim is None:
                r.update(status="NO_TRAJ", verdict="?", reward=None)
                rows.append(r)
                continue
            r["reward"] = (sim.get("reward_info") or {}).get("reward")
            r["nmsg"] = len(sim.get("messages") or [])
            if ev["served_at"] is None:
                r.update(status="NO_SERVED_AT", verdict="?")
                rows.append(r)
                continue
            # ★창의 **오른쪽 끝**도 라이브와 맞춘다. 술어는 `messages[served_at:]` 를 보는데
            #   라이브의 그 리스트는 발화 시점에서 끝나 있었고 영속 궤적은 **그 뒤까지** 담는다.
            #   자르지 않으면 발화 이후의 손님 발화가 A-3 팔을 부풀려 *"A-3 도 살린다"* 는
            #   거짓 KEEP 을 만든다(초판에서 REPRO_SUPERSET 6건이 그 증상이었다).
            if ev["fire_turn"] is None:
                r.update(status="NO_FIRE_TURN", verdict="?")
                rows.append(r)
                continue
            raw = (sim.get("messages") or [])[:int(ev["fire_turn"])]
            allm = [_M(m.get("role"), m.get("content")) for m in raw]
            useronly = [_M(m.role, m.content if m.role == "user" else None) for m in allm]
            obs = set(ev["new"])
            asis = replay(po, allm, ev["group"], ev["served"], ev["served_at"])
            if not (obs and obs <= asis):
                r.update(status="NO_REPRO", verdict="?", asis=sorted(asis))
                rows.append(r)
                continue
            a3 = replay(po, useronly, ev["group"], ev["served"], ev["served_at"])
            anyw = replay(po, useronly, ev["group"], ev["served"], 0)
            r.update(status="REPRO" if obs == asis else "REPRO_SUPERSET",
                     asis=sorted(asis), a3=sorted(a3), user_anywhere=sorted(anyw),
                     verdict="KEEP" if a3 else "KILL",
                     verdict3="KEEP" if anyw else "KILL",
                     gold_after=gold_write_after(sim, ev["fire_turn"]))
            rows.append(r)

    print("# x553 — A-3(재무장 술어에서 assistant 제거) 선행 측정")
    print("스캔한 태그 %d · 발화 %d (영속 로그에 `[T2_SEARCH_REARM]` 이 있는 전량)"
          % (len(scanned), len(rows)))
    print()
    print("%-34s %-22s %-24s %-7s %-5s %-7s %s"
          % ("tag", "sim", "group", "reward", "판정", "재현", "관측 신규 → A-3 잔존"))
    print("-" * 150)
    for r in rows:
        print("%-34s %-22s %-24s %-7s %-5s %-7s %s → %s%s"
              % (r["tag"][:34], r["sim"][:22], r["group"][:24],
                 ("-" if r.get("reward") is None else "%.1f" % r["reward"]),
                 r["verdict"], r["status"][:7],
                 ",".join(r["new"]) or "-",
                 ",".join(r.get("a3") or []) or "∅",
                 "" if r.get("verdict") != "KILL" or r.get("user_anywhere")
                 else "  (손님 발화에 아예 없음)"))

    judged = [r for r in rows if r["verdict"] in ("KEEP", "KILL")]
    print()
    print("## 판정 가능")
    print("  %d / %d  (판정 불가: %s)"
          % (len(judged), len(rows),
             " · ".join("%s %d" % (k, sum(1 for r in rows if r["status"] == k))
                        for k in ("NO_TRAJ", "NO_SERVED_AT", "NO_FIRE_TURN", "NO_REPRO")
                        if any(r["status"] == k for r in rows)) or "없음"))
    print()
    print("## 두 팔 — ⛔끄기/켜기가 아니라 **절충**이다([[70]])")
    print("  A-3  = user-only · 창은 그대로(배달 이후)  ← 문서가 적은 처방")
    print("  A-3′ = user-only · 창을 **전 접두**로       ← 같은 화자 축, 창 경계만 되돌린 판")
    arms = (("A-3 ", "verdict"), ("A-3′", "verdict3"))
    print()
    print("  %-6s %-10s %-10s %s" % ("팔", "죽는 발화", "reward 1.0", "순손실 후보 태스크"))
    for lab, key in arms:
        k = [r for r in judged if r[key] == "KILL"]
        k1 = [r for r in k if r.get("reward") == 1.0]
        print("  %-6s %2d/%-7d %-10d %s"
              % (lab, len(k), len(judged), len(k1),
                 ", ".join(sorted({r["sim"].split("#")[0] for r in k1})) or "없음"))
    print()
    print("## 태스크별 부호표 ([[70]] 판정 의무 ②)")
    print("  %-10s %6s %8s %8s %8s" % ("task", "발화", "A-3 kill", "A-3′ kill", "r=1 발화"))
    tasks = sorted({r["sim"].split("#")[0] for r in judged})
    for t in tasks:
        rs = [r for r in judged if r["sim"].startswith(t + "#")]
        print("  %-10s %6d %8d %8d %8d"
              % (t, len(rs), sum(1 for r in rs if r["verdict"] == "KILL"),
                 sum(1 for r in rs if r["verdict3"] == "KILL"),
                 sum(1 for r in rs if r.get("reward") == 1.0)))
    print()
    print("## 순손실 후보 축자 (reward 1.0 sim 에서 죽는 발화)")
    for lab, key in arms:
        for r in judged:
            if r[key] == "KILL" and r.get("reward") == 1.0:
                print("  %-4s %-34s %-20s %-22s %s"
                      % (lab, r["tag"][:34], r["sim"][:20], r["group"][:22], ",".join(r["new"])))
    print()
    print("## 반증([[77]] ③) — 그 (+) 발화보다 **뒤에** gold write 가 있었나")
    for lab, key in arms:
        k1 = [r for r in judged if r[key] == "KILL" and r.get("reward") == 1.0]
        alive = [r for r in k1 if r.get("gold_after")]
        print("  %-4s (+) 발화 %d 중 **반증되지 않은** 것 %d  (%s)"
              % (lab, len(k1), len(alive),
                 ", ".join("%s@%s" % (r["sim"].split("#")[0], r["tag"].split("_")[1])
                           for r in alive) or "없음"))
    print("  반증된 발화 = 그 sim 의 gold write 가 **전부 발화 이전**에 끝났다 ⇒ 통과의 원인일 수 없다.")
    print()
    print("⚠reward 는 **sim 단위**다 — 통과 sim 에서 죽는 발화는 순손실 *후보*이지 확정 손실이")
    print("  아니다. 반증 조건([[77]] ③): 그 sim 의 per-step 에서 델타 배달 본문이 통과 행동")
    print("  **이전에 읽히지 않았다면** 그 (+) 귀속은 거짓이다.")

    if a.json:
        import json as _j
        with open(a.json, "w", encoding="utf-8") as f:
            _j.dump(rows, f, ensure_ascii=False, indent=1)
        print("\n(json → %s)" % a.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
