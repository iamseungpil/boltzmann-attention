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


def all_result_files():
    """★영속된 결과 파일 **전량**. 명명이 두 가지다 — `.results.json.gz` 와 `_results.json.gz`(밑줄).

    2026-08-19 사고: 임시 스크립트들이 `glob("*.results.json.gz")` 만 써서 **t7273~t7299 전 구간을
    통째로 놓쳤다**(파일 250 ↔ 실제 419). 그 위에서 낸 이력 수치가 전부 틀렸다
    (task_073 `1/18` → 실제 `8/58` · task_050 `4/64` → `5/79`).
    ⇒ 이력·코퍼스 조사는 **반드시 이 함수**를 쓴다. glob 을 직접 쓰지 마라([[67]]).
    """
    import glob as _g
    out = set(_g.glob(os.path.join(BASE, "*results.json.gz")))
    out |= set(_g.glob(os.path.join(BASE, "*results.json")))
    return sorted(out, key=lambda p: os.path.getmtime(p))


def tag_of_file(path):
    """결과 파일 경로 → 런 태그(두 명명 모두)."""
    b = os.path.basename(path)
    for suf in (".results.json.gz", "_results.json.gz", ".results.json", "_results.json"):
        if b.endswith(suf):
            return b[: -len(suf)]
    return b


def iter_all_sims(want_tasks=None):
    """전 코퍼스 순회 — (tag, sim) 를 yield. `want_tasks` 가 있으면 그 태스크만."""
    for p in all_result_files():
        try:
            raw = (gzip.open(p, "rb").read().decode("utf-8", "replace")
                   if p.endswith(".gz") else io.open(p, encoding="utf-8").read())
        except Exception:
            continue
        if want_tasks and not any(t in raw for t in want_tasks):
            continue
        try:
            d = json.loads(raw)
        except Exception:
            continue
        if isinstance(d, dict):
            d = d.get("simulations") or d.get("results") or []
        if not isinstance(d, list):
            continue
        tg = tag_of_file(p)
        for s in d:
            if want_tasks and str(s.get("task_id")) not in want_tasks:
                continue
            yield tg, s


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


def _as_dict(tc):
    """호출 하나를 dict 로 본다 — **영속 JSON 의 dict** 든 **라이브 `ToolCall` 객체**든.

    2026-08-21: 궤적 재생 프로브(`x459`)가 `la.generate` 의 응답을 그대로 넘겼더니
    `'ToolCall' object has no attribute 'get'` 로 죽었다. 로더는 여태 영속 dict 만 봤는데,
    같은 물음(*"무엇을 불렀나"*)이 라이브 객체에도 필요하다. 사본을 만들지 않고 여기서 흡수한다.
    """
    if isinstance(tc, dict):
        return tc
    out = {}
    for k in ("name", "arguments", "id", "requestor", "function"):
        v = getattr(tc, k, None)
        if v is not None:
            out[k] = v
    f = out.get("function")
    if f is not None and not isinstance(f, dict):
        out["function"] = {"name": getattr(f, "name", None),
                           "arguments": getattr(f, "arguments", None)}
    return out


def nameof(tc):
    tc = _as_dict(tc)
    return (tc.get("function") or {}).get("name") or tc.get("name") or ""


def argsof(tc):
    """인자 dict. 문자열이면 푼다(못 풀면 `_raw` 로 보존 — 조용히 버리지 않는다)."""
    tc = _as_dict(tc)
    a = (tc.get("function") or {}).get("arguments", tc.get("arguments"))
    if isinstance(a, str):
        try:
            a = json.loads(a)
        except Exception:
            a = {"_raw": a}
    return a if isinstance(a, dict) else {}


def inner_name(args):
    """래퍼가 감싼 **대상 도구** 이름.

    ⚠2026-08-20 계기 수리: `discoverable_tool_name` 이 빠져 있었다. `give_/call_discoverable_user_tool`
      은 그 키로만 대상을 싣기 때문에 **손님-측 실행의 대상이 통째로 안 보였고**, 그 탓에 017 t1 은
      `reward=1.0` 인데도 우리 대조가 gold 3건을 MISSING 으로, 같은 실행 5건을 EXTRA 로 셌다."""
    return (args.get("agent_tool_name") or args.get("user_tool_name")
            or args.get("discoverable_tool_name") or args.get("tool_name") or "")


def label(name, args):
    """unlock/give/call 은 대상 도구까지 붙여야 의미가 있다(래퍼 이름만으론 무정보·C470 계기수리)."""
    t = inner_name(args)
    if not t:
        return name
    pre = {UNLOCK: "unlock", GIVE: "give", CALLA: "call", CALLU: "callu"}.get(name)
    return "%s:%s" % (pre, t) if pre else name


def norm_args(a):
    """인자를 **의미 단위로** 정규화한다 — 중첩 JSON 문자열을 풀고 스칼라를 문자열화.

    ★왜 (2026-08-17·워크플로 실측): 벤치 채점기 `tasks.py:195` 가 `tool_args == action_args` 로
      **중첩 JSON을 문자열째** 비교한다. `call_discoverable_*` 의 `arguments` 는 JSON 문자열이라
      **공백·키 순서만 달라도** 같은 실행이 `action_match=false` 로 찍힌다. 6런 gold_nested
      **1,222건 중 121건(SEMANTIC_ONLY)** 이 그 형태였다.
    ⚠**reward 에는 영향이 없다**(`reward_basis=['DB']` 태스크는 DB 해시로 채점되고, 손님 호출은
      dict 로 로깅된다) — 오염되는 것은 **우리 포렌식·센서스의 주장**이다. 그래서 이 함수는
      *벤치를 고치는 것이 아니라 우리 계기를 고친다*. pass 는 여전히 `reward` 로만 읽는다(C486).
    ⚠판단 0: 값의 뜻을 해석하지 않고 **표기만** 맞춘다(정규식 0·[[59]]).
    """
    if isinstance(a, str):
        try:
            a = json.loads(a)
        except Exception:
            return " ".join(a.split())
    if isinstance(a, dict):
        return dict((str(k), norm_args(v)) for k, v in a.items())
    if isinstance(a, (list, tuple)):
        return [norm_args(x) for x in a]
    if isinstance(a, bool) or a is None:
        return a
    if isinstance(a, (int, float)):
        # 채점기가 "750" 과 750 을 가르는 자리가 있어 **문자열로 접는다**(양쪽 같은 변환).
        return ("%g" % a) if isinstance(a, float) else str(a)
    return " ".join(str(a).split())


def args_equal(x, y):
    """두 인자 묶음이 **같은 실행**인가. `norm_args` 를 양쪽에 같이 걸고 비교한다."""
    return norm_args(x) == norm_args(y)


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
    """런 로그 전문. 없으면 빈 문자열(침묵) — 예외로 분석을 죽이지 않는다.

    ⚠**`path_for` 를 쓰지 않는다**(2026-08-16 자기 결함): 그 함수는 *라이브 시뮬 디렉터리*를 먼저
      보는데 거기 있는 것은 `results.json`(평문)이라, `.log.gz` 를 달라고 하면 **평문 JSON 을
      gzip 으로 열려다 죽는다**(`BadGzipFile: b'{\\n'`). 로그는 두 자리만 본다 —
      영속본(`sim_results/<tag>.log.gz`) → 리모트 원본(`/home/woori/scratch/logs/<tag>.log`).
    """
    import gzip as _gz
    gzp = os.path.join(BASE, tag + ".log.gz")
    if os.path.exists(gzp):
        with _gz.open(gzp, "rt", encoding="utf-8", errors="replace") as f:
            return f.read()
    raw = os.path.join("/home/woori/scratch/logs", tag + ".log")
    if os.path.exists(raw):
        with io.open(raw, encoding="utf-8", errors="replace") as f:
            return f.read()
    return ""


def trace(tag):
    """런의 **구조화 계기**(`trace_<tag>.jsonl`) — `{sim, turn, mark, line}` 행 리스트.

    ★왜 이것이 정본인가 (2026-08-18 감사): stderr 로그 줄은 **8%만** `turn=` 을 찍는다
      (t7310 treat: 2,541 줄 중 215). 그런데 **같은 사건이 이 파일에는 turn 과 함께** 들어 있다
      (2,558 줄 중 turn 없는 줄 **17** = 모듈 기동 줄뿐 · `T2_RESOLVE` 40/40 · `T2_VERDICT` 12/12 ·
      `T2_ACTIONREQ` 52/52 · `T2_MATERIAL_GATE` 198/198).
      즉 **sim·turn 단위 분석은 처음부터 가능했고**, 못 한 것은 `turns_of` 가 로그 *텍스트*에서
      `turn=` 을 긁었기 때문이다 — 그 탓에 24 sim 전부 `None` 이 나와 손해 측정을 접었다(C530⒟).
    영속본(`sim_results/<tag>.trace.jsonl.gz`) → 리모트 원본 순으로 본다. 없으면 빈 리스트(침묵).
    """
    import gzip as _gz
    out = []
    gzp = os.path.join(BASE, tag + ".trace.jsonl.gz")
    raw = os.path.join(FBDIR, "trace_" + tag + ".jsonl")
    if os.path.exists(gzp):
        op = _gz.open(gzp, "rt", encoding="utf-8", errors="replace")
    elif os.path.exists(raw):
        op = io.open(raw, encoding="utf-8", errors="replace")
    else:
        return out
    with op as f:
        for ln in f:
            try:
                out.append(json.loads(ln))
            except Exception:
                continue
    return out


def sidecar_rows(tag):
    """우리 층이 **보낸 문장**(`fb_<tag>.jsonl`) — `{sim, turn, channel, text, …}` 행 리스트.

    trace 가 *어느 기구가 말했는가*를 남기면 이쪽은 *무엇을 말했는가*를 남긴다(둘은 다르다).
    실측(t7310 treat): 266 줄 · sim 12 · 채널 17종 · **turn 없는 줄 0**.
    """
    import gzip as _gz
    out = []
    gzp = os.path.join(BASE, tag + ".fb.jsonl.gz")
    raw = os.path.join(FBDIR, "fb_" + tag + ".jsonl")
    if os.path.exists(gzp):
        op = _gz.open(gzp, "rt", encoding="utf-8", errors="replace")
    elif os.path.exists(raw):
        op = io.open(raw, encoding="utf-8", errors="replace")
    else:
        return out
    with op as f:
        for ln in f:
            try:
                out.append(json.loads(ln))
            except Exception:
                continue
    return out


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
    rows = trace(tag)
    if rows:
        # ★정본 경로(2026-08-18): 구조화 계기에서 **turn 필드**를 읽는다. 로그 텍스트는 8%만
        #   `turn=` 을 찍지만 이 파일은 관심 마커 전부 turn 을 갖고 있다(`trace` 독스트링).
        rx = _re.compile(pattern)
        out = {}
        for d in rows:
            k = d.get("sim")
            if not k:
                continue                      # 모듈 기동 줄(sim 없음) — 사건이 아니다
            if sims_ is not None and k not in sims_:
                continue
            ln = str(d.get("line") or "")
            if not rx.search(ln):
                continue
            t = d.get("turn")
            out.setdefault(k, []).append(int(t) if isinstance(t, int) or
                                         (isinstance(t, str) and t.isdigit()) else None)
        return out
    # 폴백: trace 가 없는 옛 런 — 종전대로 로그 텍스트에서 긁고, 없으면 **모른다고 남긴다**
    out = {}
    for k, hits in by_sim(tag, pattern, sims_).items():
        vals = []
        for _i, s2 in hits:
            m = _re.search(r"turn=(\d+)", s2 if isinstance(s2, str) else "")
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


# ─────────────────────────────────────────────────────────────────────────────
# 채점 단위 = **변이 집합** ([[69]] · C545)
# ─────────────────────────────────────────────────────────────────────────────
# DB 채점 태스크의 점수를 만드는 것은 최종 DB 상태뿐이고, 그것을 만드는 것은 **성공한 변이
# 호출의 집합**이다. read 는 아무리 놓쳐도 해시에 안 남는다. `x416_db_diff.py` 가 이 판정을
# 스크립트 안에 갖고 있어 import 가 안 됐다 — 사본이 갈라지기 전에 정본으로 올린다([[67]]).
#
# 변이 여부는 `a2/env_surface.json` 의 `mutates` 플래그(환경 선언·축자)로만 본다([[59]]).
# 실패한 호출을 성공으로 세지 않기 위해 도구-결과 본문의 거절 표지를 본다. 그 표지가
# **우리 것인지 환경 것인지**를 함께 돌려준다 — 우리 게이트가 막은 변이는 모델 결손이 아니다([[55]]).

# 거절 판정은 **누가 그 문장을 썼는가**로 가른다(축자 확인 2026-08-20):
#   · 환경 = 도구-결과가 `Error:` 로 **시작**한다(tau2 규약). 본문 중간의 "Error:" 는 KB 문서 인용이다
#     (실측 16건 전부 검색 결과). "Invalid"·"cannot be" 는 성공 응답 본문에도 흔하다(217·79건) —
#     substring 으로 세면 성공한 write 를 막힌 것으로 오분류한다.
#   · 우리 = `[READ-FIRST]`(t2_scaffold_get requires_reads) · `NOT_VERIFIED`(신원 게이트). 둘 다 env
#     소스에 없음을 확인했다(`tau2-bench/src` grep 0). "has not been given to you by the agent" 는
#     반대로 **환경 것**이다(`domains/banking_knowledge/tools.py`) — 우리 것으로 세면 안 된다.
OURS_DENY = ("[READ-FIRST]", "NOT_VERIFIED")
# ★A-7⑵ (2026-08-23·079): env 는 `Error:` 로만 거절하지 않는다. 실패 서술로 시작하는
#   본문(`Failed to …`)을 성공으로 세면 그 실행이 MATCHED 가 되고, 뒤따르는 재시도가
#   DUP 위양성으로 찍힌다 — 079 의 DUP 주장이 그렇게 태어났다. 프레임워크가 쓰는 실패
#   서두만 본다(도메인 어휘 0·이 파일은 오프라인 포렌식 라이브러리다).
ENV_FAIL_PREFIX = ("Error:", "Failed to ")

# 래퍼 4종의 역할 분리: 부여(grant)는 DB 를 안 바꾸고, 실행(call)만 바꾼다.
GRANTS = (UNLOCK, GIVE)
EXECS = (CALLA, CALLU)


def mutating_tools(domain="banking_knowledge"):
    """`mutates=True` 로 **환경이 선언한** 도구 이름 집합."""
    p = os.path.join(HERE, "a2", "env_surface.json")
    with io.open(p, encoding="utf-8") as f:
        d = json.load(f)
    return {k for k, v in (d[domain]["tools"]).items() if v.get("mutates")}


def flat_args(a):
    """래퍼 중첩(`arguments` 안의 `arguments`·JSON 문자열)을 풀어 **대상 도구의 인자**만 남긴다."""
    a = norm_args(a)
    if isinstance(a, dict) and isinstance(a.get("arguments"), dict):
        a = a["arguments"]
    if isinstance(a, dict) and isinstance(a.get("arguments"), str):
        try:
            a = json.loads(a["arguments"])
        except Exception:
            pass
    return a if isinstance(a, dict) else {}


def mut_key(name, args):
    """변이 하나의 동일성 = 이름 + 인자(문자열 접기)."""
    return name + "|" + json.dumps({k: str(v) for k, v in sorted(args.items())}, ensure_ascii=False)


def deny_kind(body):
    """도구-결과 본문이 거절인가 · 누가 거절했나 → ('', None) | ('ours'|'env', 표지)."""
    b = body.lstrip()
    for p in OURS_DENY:
        if p in b:
            return "ours", p
    for _p in ENV_FAIL_PREFIX:
        if b.startswith(_p):
            return "env", b[:60]
    return "", None


def gold_mutations(sim, mut=None):
    """gold 채점표가 요구하는 **변이** 행만(read gold 는 점수와 무관하므로 뺀다)."""
    mut = mutating_tools() if mut is None else mut
    out, seen = [], set()
    for ck in ((sim.get("reward_info") or {}).get("action_checks") or []):
        a = ck.get("action") or {}
        ar = a.get("arguments") or {}
        outer = str(a.get("name") or "")
        if outer in GRANTS:
            continue                      # 부여는 DB 를 안 바꾼다 — 변이로 세면 gold 가 부풀고
        nm = str(inner_name(ar) or outer or "?")   #   같은 실행이 MISSING+EXTRA 로 두 번 찍힌다
        if nm not in mut:
            continue
        inner = ar.get("arguments", None)
        args = flat_args(ar if inner is None else inner)
        if not args:
            continue
        k = mut_key(nm, args)
        if k in seen:
            continue
        seen.add(k)
        out.append({"name": nm, "args": args, "match": bool(ck.get("action_match")),
                    "aid": str(a.get("action_id") or ""), "key": k})
    return out


def attempted_mutations(sim, mut=None):
    """궤적이 **시도한** 변이 호출 전부 — 거절된 것까지 포함해 돌려준다.

    `ok=True` 만이 DB 를 바꾼 것이고, `ok=False` 는 *시도했으나 막힘*이다. 그 구분이 없으면
    "안 했다(MISSING)" 와 "막혔다" 가 한 칸에 뭉쳐 원인이 사라진다.
    """
    mut = mutating_tools() if mut is None else mut
    res = {m["id"]: " ".join(str(m.get("content") or "").split())
           for m in (sim.get("messages") or []) if m.get("role") == "tool" and m.get("id")}
    msgs = sim.get("messages") or []
    idx = {id(m): i for i, m in enumerate(msgs)}
    out = []
    for m, tc in calls(sim):
        a = argsof(tc)
        nm = str(inner_name(a) or nameof(tc))
        if nm not in mut:
            continue
        if str(nameof(tc)) in GRANTS:
            continue                      # unlock·give 는 부여일 뿐 DB 를 안 바꾼다
        args = flat_args(a)
        if not args:
            continue
        body = res.get(tc.get("id"), "")
        kind, marker = deny_kind(body)
        out.append({"name": nm, "args": args, "key": mut_key(nm, args),
                    "msg_i": idx.get(id(m)), "ok": not kind, "deny": kind,
                    "marker": marker, "result": body[:300]})
    return out


def mutation_diff(sim, mut=None):
    """gold 변이 집합 ↔ 성공한 변이 집합 → MISSING · WRONGARG · EXTRA · MATCHED · BLOCKED.

    · MISSING  gold 변이인데 **성공한 같은 호출이 없다**
    · WRONGARG 같은 도구를 성공시켰는데 인자가 gold 와 다르다
    · EXTRA    gold 에 없는 도구를 성공시켰다 (050 의 승인 중복이 이 칸이다)
    · BLOCKED  시도했으나 거절당한 변이 (누가 거절했는지 `deny` 에 남는다)
    """
    mut = mutating_tools() if mut is None else mut
    gold = gold_mutations(sim, mut)
    tried = attempted_mutations(sim, mut)
    done = [t for t in tried if t["ok"]]
    blocked = [t for t in tried if not t["ok"]]
    gkeys = {g["key"] for g in gold}
    dkeys = {d["key"] for d in done}
    gnames = {g["name"] for g in gold}
    missing = [g for g in gold if g["key"] not in dkeys]
    wrong = [d for d in done if d["key"] not in gkeys and d["name"] in gnames]
    extra = [d for d in done if d["key"] not in gkeys and d["name"] not in gnames]
    matched = [d for d in done if d["key"] in gkeys]
    # ★중복(DUP) — 집합으로 세면 사라지는 실패다. 050 은 `approve_credit_limit_increase` 를
    #   **두 번** 성공시켜 DB 가 어긋났는데, 두 번째 호출의 key 는 gold 안에 있으므로 EXTRA 도
    #   WRONGARG 도 아니다. 배수를 세지 않으면 그 sim 은 "변이 집합 일치"로 보인다(실측 3 sim).
    gcnt = collections.Counter(g["key"] for g in gold)
    dup = []
    seen = collections.Counter()
    for d in done:
        seen[d["key"]] += 1
        if seen[d["key"]] > gcnt.get(d["key"], 0):
            dup.append(d)
    dup = [d for d in dup if d["key"] in gkeys]
    return {"gold": gold, "done": done, "blocked": blocked, "missing": missing,
            "wrongarg": wrong, "extra": extra, "matched": matched, "dup": dup,
            "clean": not (missing or wrong or extra or dup)}


# ─────────────────────────────────────────────────────────────────────────────
# 채점 단위 = **액션 집합** (`reward_basis=["ACTION"]` 태스크 · A16 / OL-49)
# ─────────────────────────────────────────────────────────────────────────────
# ★왜 별개의 함수인가 (2026-08-22·t7336 마스터 §6.1 A16):
#   `mutation_diff` 는 DB-채점 태스크의 물음(*"DB 해시를 만든 변이 집합이 gold 와 같은가"*)에
#   맞춰져 있어 필터가 둘 걸려 있다 — ⑴`unlock`/`give`(= `GRANTS`)는 DB 를 안 바꾸므로 제외
#   ⑵대상 이름이 `mutates=True` 집합 밖이면 제외. 그런데 **ACTION-채점** 태스크(t7336 20 태스크
#   중 033 하나가 `reward_basis=["ACTION"]`·마스터 §1.1)에서는 gold 액션 자체가 `unlock` 과
#   read 래퍼로 이루어져 있어, 두 필터가 **전 항목을 지운다**(OL-49: 033 양 trial ·
#   072#0·074#0·079#1 에서 전 칸 빈칸). 그래서 ACTION-basis 태스크마다 손으로 표를 만들고
#   있었고, 그것이 [[67]] 이 금지한 사본 제조의 입구였다.
#   ⇒ 여기서는 **필터를 걸지 않는다**: `action_checks` 행을 그대로 세고, 궤적에서 같은 실행을
#     찾는다. 판정(무엇이 실패인가)은 여전히 각 포렌식 몫이고 이 함수는 대조표만 만든다.
# ⚠[[69]]: 성적은 언제나 `reward` 다. `action_match` 는 **진단 보조**이고 성적이 아니다
#   (마스터 §1.1 반례 2건: 017#0 은 불일치 2건인데 reward 1.0 · 050#1 도 불일치인데 1.0).
#   그래서 이 표는 `reward` 를 대체하지 않고 **함께** 실린다(`reward` 키를 같이 돌려준다).
# ⚠판단 0 — 산수·집합·문자열뿐이다([[59]]). 어느 행이 중요한지 고르지 않는다.


def reward_basis(sim):
    """이 sim 의 채점 축 — `["DB"]` / `["ACTION"]` / 없으면 `[]`(채점표 부재)."""
    v = (sim.get("reward_info") or {}).get("reward_basis")
    if isinstance(v, str):
        return [v]
    return list(v) if isinstance(v, (list, tuple)) else []


def action_key(outer, inner, args):
    """액션 하나의 동일성 = 바깥 도구 + 대상 도구 + 인자(문자열 접기·`norm_args` 공유)."""
    return "%s|%s|%s" % (outer, inner or "",
                         json.dumps({k: norm_args(v) for k, v in sorted((args or {}).items())},
                                    ensure_ascii=False, sort_keys=True))


def gold_actions_flat(sim):
    """`action_checks` 를 **필터 없이** 편다 — read·grant·exec 전부.

    각 행: outer(래퍼 이름) · inner(대상 도구) · args(래퍼 해제 후 대상 인자) · aid ·
    `bench_match`(= 벤치 채점기가 찍은 `action_match`·**우리 판정이 아니다**) · key.
    """
    out = []
    for ck in ((sim.get("reward_info") or {}).get("action_checks") or []):
        a = ck.get("action") or {}
        ar = a.get("arguments") or {}
        outer = str(a.get("name") or "")
        inner = str(inner_name(ar) or "")
        inner_args = ar.get("arguments", None)
        args = flat_args(ar if inner_args is None else inner_args)
        out.append({"outer": outer, "inner": inner, "args": args,
                    "aid": str(a.get("action_id") or ""),
                    "tool_type": ck.get("tool_type"),
                    "bench_match": bool(ck.get("action_match")),
                    "key": action_key(outer, inner, args)})
    return out


def trajectory_actions(sim):
    """궤적의 호출을 gold 와 **같은 모양**으로 편다(필터 0) + 결과의 거절 여부.

    `ok=True` = 그 호출의 tool 결과가 거절이 아니다. 거절이면 `deny` 에 누가 막았는지 남는다
    (`deny_kind` 재사용·사본 0). 결과 메시지를 못 찾으면 `ok=True`(관측 부재를 실패로 세지 않는다).
    """
    res = {m.get("id"): " ".join(str(m.get("content") or "").split())
           for m in (sim.get("messages") or [])
           if m.get("role") == "tool" and m.get("id")}
    msgs = sim.get("messages") or []
    idx = {id(m): i for i, m in enumerate(msgs)}
    out = []
    for m, tc in calls(sim):
        a = argsof(tc)
        outer = str(nameof(tc))
        inner = str(inner_name(a) or "")
        inner_args = a.get("arguments", None)
        args = flat_args(a if inner_args is None else inner_args)
        body = res.get(tc.get("id"), "")
        kind, marker = deny_kind(body) if body else ("", None)
        out.append({"outer": outer, "inner": inner, "args": args,
                    "key": action_key(outer, inner, args),
                    "msg_i": idx.get(id(m)), "ok": not kind, "deny": kind,
                    "marker": marker, "requestor": tc.get("requestor")})
    return out


def action_diff(sim):
    """gold 액션 집합 ↔ 궤적 → MATCH · MISSING(+원인 칸) (A16 / OL-49).

    ★MATCH/MISSING 의 **권위는 벤치의 `action_match`** 다 — 우리가 다시 판정하지 않는다.
      이유(실물): 004 의 gold 액션은 `transfer_to_human_agents(summary, reason)` 이고 궤적은
      **같은 도구를 같은 인자 키로** 불렀지만 `summary` 는 자유 산문이라 인자 축자 비교로는
      영영 불일치다. 그런데 벤치는 `action_match=True` 로 찍었다. 우리가 여기서 "어느 인자까지
      같아야 같은 실행인가"를 정하면 그것은 **채점기를 다시 쓰는 일**이고, 그 판정은 우리 것이지
      성적이 아니다([[69]]·[[25]] 확인 안 한 것을 단언하지 않는다).
    ⇒ 칸 구성:
      · matched / missing   `action_match` 그대로 (bench 권위)
      · 각 행의 **원인 칸**(우리 관측·판단 0): `called_exact`(래퍼·대상·인자 전부 일치한 성공 호출)
        · `called_name`(래퍼·대상만 일치·인자는 다름) · `blocked`(시도했으나 거절·누가 막았는지)
      · `strict_missing`  인자까지 축자 일치하는 성공 호출이 없는 행 — **진단 보조**다.
        `missing` 과 갈리는 자리가 곧 *"표기 차이인가 진짜 미수행인가"* 의 물음이고,
        `norm_args` 독스트링이 잰 121건(SEMANTIC_ONLY)이 그 자리다.
    ⚠`reward` 를 같이 돌려준다 — 성적은 이 표가 아니라 그 수다([[69]]).
    ⚠필터 0: `mutates` 도 `GRANTS` 도 걸지 않는다. 그 둘이 ACTION-basis 태스크에서 전 항목을
      지웠다는 것이 OL-49 다(033 양 trial · 072#0 · 074#0 · 079#1).
    """
    gold = gold_actions_flat(sim)
    tried = trajectory_actions(sim)
    ok_exact = collections.Counter(t["key"] for t in tried if t["ok"])
    ok_name = collections.Counter((t["outer"], t["inner"]) for t in tried if t["ok"])
    blocked_by_key, blocked_by_name = {}, {}
    for t in tried:
        if not t["ok"]:
            blocked_by_key.setdefault(t["key"], t)
            blocked_by_name.setdefault((t["outer"], t["inner"]), t)
    used = collections.Counter()
    rows = []
    for g in gold:
        used[g["key"]] += 1
        r = dict(g)
        r["called_exact"] = used[g["key"]] <= ok_exact.get(g["key"], 0)
        r["called_name"] = ok_name.get((g["outer"], g["inner"]), 0) > 0
        b = blocked_by_key.get(g["key"]) or blocked_by_name.get((g["outer"], g["inner"]))
        r["blocked"] = None if b is None else {"deny": b["deny"], "marker": b["marker"],
                                               "msg_i": b["msg_i"]}
        rows.append(r)
    matched = [r for r in rows if r["bench_match"]]
    missing = [r for r in rows if not r["bench_match"]]
    return {"basis": reward_basis(sim), "reward": (sim.get("reward_info") or {}).get("reward"),
            "rows": rows, "gold": gold, "tried": tried,
            "matched": matched, "missing": missing,
            "blocked": [r for r in missing if r["blocked"]],
            "strict_missing": [r for r in rows if not r["called_exact"]],
            "n_gold": len(gold), "n_matched": len(matched),
            "clean": not missing}
