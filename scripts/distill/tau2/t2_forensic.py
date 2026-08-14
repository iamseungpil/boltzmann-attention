# -*- coding: utf-8 -*-
"""궤적 포렌식 **정본 라이브러리** — 로딩·래퍼 해제·호출 추출.

사용자 지시(2026-08-14 야간·`t2_subcall` 때와 같은 지적): 새 포렌식을 쓸 때마다 사본을 짜지 마라.
실측 중복(11개 `bank_*forensic|audit` 스크립트): `sim_results` 경로 **5사본** · 로더 **6사본** ·
`nameof/argsof` 류 다수. 사본은 조용히 갈라진다 — C470 의 계기 4수리(래퍼 대상-도구 색인·
중첩 `9.50↔9.5` 정규화·시도vs실행·seed 매칭)는 **한 사본에만** 들어갔고 다른 사본들은 여전히
옛 방식으로 세고 있었다. 그 갈라짐이 "073 성공 실행을 NOTCALLED 로 오분류"를 낳았다.

여기 있는 것은 **읽기·해제·집계 보조뿐**이다 — 판정(무엇이 실패인가)은 각 포렌식이 자기 물음에
맞게 한다. 이 파일은 도메인 판단을 하지 않는다(래퍼 이름 상수는 tau2 하네스 프로토콜이지
banking 도메인 어휘가 아니다).

새 포렌식은 여기서 import 한다:
    import t2_forensic as F
    for tag in tags:
        for s in F.sims(tag):
            for m, tc in F.calls(s):
                print(F.label(F.nameof(tc), F.argsof(tc)))
"""
import collections
import gzip
import io
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026", "sim_results")
FBDIR = "/home/woori/scratch/logs"          # 사이드카(리모트 전용·없으면 조용히 빈 값)

# tau2 하네스의 발견-도구 래퍼 프로토콜(도메인 어휘 아님)
UNLOCK = "unlock_discoverable_agent_tool"
GIVE = "give_discoverable_user_tool"
CALLA = "call_discoverable_agent_tool"
CALLU = "call_discoverable_user_tool"
WRAPPERS = (UNLOCK, GIVE, CALLA, CALLU)


LIVE = "/home/woori/scratch/tau2-bench/data/simulations"     # 리모트 갓-끝난 런(영속 前)
TRANSFER = ("transfer_to_human_agents", "request_human_agent_transfer")


def path_for(tag, suffix="_results.json.gz"):
    """태그 → 경로. 리모트 라이브 결과가 있으면 그쪽을 먼저 본다(영속 前 런 감사용).
    절대경로/파일명을 그대로 줘도 받는다(옛 사본들의 호출 형태 호환)."""
    if os.path.isabs(tag) or os.path.exists(tag):
        return tag
    live = os.path.join(LIVE, tag, "results.json")
    if os.path.exists(live):
        return live
    p = os.path.join(BASE, tag + suffix)
    return p if os.path.exists(p) else os.path.join(BASE, tag)


def load(tag, suffix="_results.json.gz"):
    """결과 JSON 을 통째로 반환(gz/평문 자동)."""
    p = path_for(tag, suffix)
    op = gzip.open(p, "rt", encoding="utf-8") if p.endswith(".gz") else io.open(p, encoding="utf-8")
    with op as f:
        return json.load(f)


def sims(tag, suffix="_results.json.gz"):
    """sim 리스트. 결과 파일의 두 형태(`{"simulations": [...]}` / 리스트)를 함께 받는다."""
    d = load(tag, suffix)
    if isinstance(d, dict):
        d = d.get("simulations") or d.get("results") or []
    return d if isinstance(d, list) else []


def scored(tag, suffix="_results.json.gz"):
    """채점된 sim 만(reward_info 실재) — 진행 중/크래시 sim 을 성적에 섞지 않기 위해."""
    return [s for s in sims(tag, suffix) if s.get("reward_info") is not None]


def sidecar(tag):
    """우리 채널 사이드카(`fb_<tag>.jsonl`)를 simtag 별로. 없으면 빈 dict(로컬)."""
    out = collections.defaultdict(list)
    p = os.path.join(FBDIR, "fb_%s.jsonl" % tag)
    if not os.path.exists(p):
        return out
    for ln in io.open(p, encoding="utf-8", errors="replace"):
        try:
            o = json.loads(ln)
        except Exception:
            continue
        out[o.get("simtag") or "?"].append(o)
    return out


def nameof(tc):
    return (tc.get("function") or {}).get("name") or tc.get("name") or ""


def argsof(tc):
    """인자 dict. 문자열이면 푼다(못 풀면 `_raw` 로 보존 — 조용히 버리지 않는다)."""
    a = (tc.get("function") or {}).get("arguments", tc.get("arguments"))
    if isinstance(a, str):
        try:
            a = json.loads(a)
        except Exception:
            a = {"_raw": a}
    return a if isinstance(a, dict) else {}


def inner_name(args):
    """래퍼가 감싼 **대상 도구** 이름."""
    return (args.get("agent_tool_name") or args.get("user_tool_name")
            or args.get("tool_name") or "")


def label(name, args):
    """unlock/give/call 은 대상 도구까지 붙여야 의미가 있다(래퍼 이름만으론 무정보·C470 계기수리)."""
    t = inner_name(args)
    if not t:
        return name
    pre = {UNLOCK: "unlock", GIVE: "give", CALLA: "call", CALLU: "callu"}.get(name)
    return "%s:%s" % (pre, t) if pre else name


def calls(sim):
    """(message, tool_call) 쌍을 순서대로. 어시스턴트 턴의 호출만."""
    for m in sim.get("messages") or []:
        for tc in (m.get("tool_calls") or []):
            yield m, tc


def call_labels(sim):
    """궤적의 호출 라벨 순열(대상-도구 해제 포함)."""
    return [label(nameof(tc), argsof(tc)) for _m, tc in calls(sim)]


def targets(sim):
    """궤적에 **실제로 등장한** 대상 도구 이름 집합(래퍼 안쪽까지 본다)."""
    out = set()
    for _m, tc in calls(sim):
        n = nameof(tc)
        a = argsof(tc)
        out.add(inner_name(a) or n)
    return out


def assistant_text(sim, last=True):
    """손님-가시 본문(어시스턴트 content). last=True 면 마지막 하나."""
    txts = [str(m.get("content") or "") for m in (sim.get("messages") or [])
            if m.get("role") == "assistant" and m.get("content")]
    if not txts:
        return ""
    return txts[-1] if last else "\n".join(txts)


def gold_actions(sim):
    """채점 기준의 gold 액션 리스트(형태 차이를 흡수·없으면 빈 리스트)."""
    ri = sim.get("reward_info") or {}
    for k in ("action_checks", "actions"):
        v = ri.get(k)
        if isinstance(v, list):
            return v
    return []


def write_tools(tag_or_doc):
    """gold 채점표가 **write** 로 표시한 도구 이름(래퍼는 대상 도구까지).

    ⚠사본 갈라짐 실물: `bank_miss_turn_audit` 은 `agent_tool_name/user_tool_name` 만 보고
    `bank_trigger_window_audit` 은 `tool_name` 도 봤다 — 같은 이름의 함수가 서로 다른 집합을
    반환하고 있었다. 정본은 **셋 다** 본다(넓은 쪽이 안전측: 놓친 write 를 만들지 않는다)."""
    doc = tag_or_doc if isinstance(tag_or_doc, dict) else {"simulations": sims(tag_or_doc)}
    out = set()
    for s in (doc.get("simulations") or []):
        for ck in ((s.get("reward_info") or {}).get("action_checks") or []):
            if ck.get("tool_type") != "write":
                continue
            a = ck.get("action") or {}
            ar = a.get("arguments") or {}
            out.add(str(a.get("name")))
            for k in ("agent_tool_name", "user_tool_name", "tool_name"):
                if ar.get(k):
                    out.add(str(ar[k]))
    return {n for n in out if n and n != "None"}


def term_reason(sim):
    return (sim.get("termination_reason") or sim.get("term") or "?")


def task_id(sim):
    return sim.get("task_id") or "?"


def sim_key(sim):
    """태스크+시행 식별(시행 인덱스 이름이 런마다 달라 세 후보를 순서대로 본다)."""
    for k in ("trial", "trial_id", "seed"):
        if sim.get(k) is not None:
            return "%s#%s" % (task_id(sim), sim.get(k))
    return task_id(sim)
