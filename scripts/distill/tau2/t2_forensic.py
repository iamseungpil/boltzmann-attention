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


def transfer_msg_index(sim):
    """어시스턴트가 이관을 처음 부른 **메시지 색인**(호출 순번이 아니다).

    ⚠두 색인 공간을 섞지 말 것 — 2026-08-14 야간 실물: 호출 순번(예: 23)을 메시지 색인으로 써서
    궤적을 앞에서 잘라 버렸고, 그 결과 "손님이 이관을 요구했다"가 전부 '아니오'로 집계됐다."""
    for i, m in enumerate(sim.get("messages") or []):
        for tc in (m.get("tool_calls") or []):
            nm = label(nameof(tc), argsof(tc)).split(":")[-1]
            if nm in TRANSFER or "transfer_to_human" in nm or "human_agent_transfer" in nm:
                return i
    return None


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
    """태스크+시행 식별(시행 인덱스 이름이 런마다 달라 세 후보를 순서대로 본다).

    ⚠**로그 조인에 쓰지 마라** — 로그의 `[sim=...]` 태그는 `s<seed>` 이고 이것은 `trial` 우선이다.
      조인은 `simtag()`/`by_sim()` 을 쓴다(C491⒠·2026-08-16 같은 함정에 두 번 걸린 뒤 정본화).
    """
    for k in ("trial", "trial_id", "seed"):
        if sim.get(k) is not None:
            return "%s#%s" % (task_id(sim), sim.get(k))
    return task_id(sim)


def simtag(sim):
    """**로그 조인 키** — 로그가 찍는 `[sim=task_055#s626729]` 와 바이트 동일하게 만든다.

    실물 사고: `sim_key`(trial 우선)로 로그를 찾으면 **전부 미스**한다(빈 결과를 *"발화 없음"* 으로
    오독하게 된다). 2026-08-15 C491⒠ 에서 한 번, 2026-08-16 배달↔선택 정합에서 또 한 번 걸렸다.
    """
    s = sim.get("seed")
    return "%s#s%s" % (task_id(sim), s) if s is not None else sim_key(sim)


def log_text(tag):
    """런 로그 전문(gz). 없으면 빈 문자열(침묵) — 예외로 분석을 죽이지 않는다."""
    import gzip as _gz
    p = path_for(tag, ".log.gz")
    if not p or not os.path.exists(p):
        return ""
    with _gz.open(p, "rt", encoding="utf-8", errors="replace") as f:
        return f.read()


def first_named(sim, names):
    """**첫 지목** — 어시스턴트가 후보 이름을 처음 입에 올린 메시지 index (없으면 None).

    왜 필요한가(2026-08-16·055·024 공통 기전): 지목이 박히면 **그 뒤에 온 재료는 안 먹는다**.
      · 055 — 첫 지목 msg 7~15 · 그 뒤 배달 3회 · 이름 안 바뀜 · open 은 msg 31~46
      · 024 — 첫 지목 msg 4 · msg 7 에 gold 문서가 **검색 1위**로 들어왔는데도 안 바꿈
    그래서 P1 의 1차 종점은 배달 **횟수**가 아니라 *"첫 지목 이전에 도달했는가"* 다.
    `names` = 후보 이름 목록(도메인 어휘는 **호출부**가 준다 — 엔진은 판단하지 않는다·[[59]]).
    """
    import re as _re
    rx = _re.compile("|".join(_re.escape(n) for n in names), _re.I) if names else None
    if rx is None:
        return None
    for i, m in enumerate(sim.get("messages") or []):
        if m.get("role") != "assistant":
            continue
        if rx.search(str(m.get("content") or "")):
            return i
    return None


def turns_of(tag, pattern, sims_=None):
    """로그 매치의 **턴 번호**(`turn=N`)를 sim 별로. 턴을 안 찍는 줄은 None 으로 남긴다.

    ⚠턴을 안 찍는 로그 줄이 있다(예: 검색 재료 배달). 그때는 **모른다고 남기지**, 순서로
      추정하지 않는다([[25]] *모르면 안 뺀다* 의 계기판).
    """
    import re as _re
    out = {}
    for k, hits in by_sim(tag, pattern, sims_).items():
        vals = []
        for _i, s in hits:
            m = _re.search(r"turn=(\d+)", s if isinstance(s, str) else "")
            vals.append(int(m.group(1)) if m else None)
        out[k] = vals
    return out


def by_sim(tag, pattern, sims_=None):
    """로그를 **sim 별로** 훑어 `pattern` 매치를 모은다 → {simtag: [(줄번호, 매치), …]}.

    프로브마다 로그를 다시 grep 하던 것을 여기로 모은다([[67]]). `sims_` 를 주면 그 sim 들의
    `simtag` 만 남긴다(태스크 필터). 매치는 `re.search` 의 group(1) 이 있으면 그것, 없으면 줄 전체.
    """
    import re as _re
    rx = _re.compile(pattern)
    want = {simtag(s) for s in sims_} if sims_ else None
    out = {}
    for i, line in enumerate(log_text(tag).split("\n")):
        m0 = _re.search(r"\[sim=(task_\d+#\w+)\]", line)
        if not m0:
            continue
        key = m0.group(1)
        if want is not None and key not in want:
            continue
        m = rx.search(line)
        if m:
            out.setdefault(key, []).append((i, m.group(1) if m.groups() else line.strip()))
    return out
