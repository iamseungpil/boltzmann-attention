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


def is_gzip(p):
    """그 파일이 정말 gzip 인가 — **이름이 아니라 매직 바이트**로 답한다(닫힌 술어).

    `.gz` 라는 이름은 *주장*이고 파일이 권위다([[25]]). 실물(2026-08-24 확인): 영속본
    `bank_cwe_batch_a2_20260719.log.gz` 와 `bank_cwe_batch_b_20260719.log.gz` 는 이름만 `.gz`
    이고 내용은 평문이다. 이름을 믿은 자리가 `BadGzipFile: b'{\n'` 로 **터졌고**, 그 예외
    하나가 `test_forensic_sidecar_authority` 를 통째로 죽이고 있었다 — 계기가 입력 하나에
    터지면 그 계기로 잰 것이 전부 끊긴다. `log_text` 독스트링이 같은 예외를 2026-08-16 에
    이미 적어 놓고 **그 자리만** 피해 갔다.
    """
    try:
        with io.open(p, "rb") as f:
            return f.read(2) == b"\x1f\x8b"
    except Exception:
        return False


def topen(p, errors="replace"):
    """텍스트 읽기용 파일 객체 — gzip 여부는 `is_gzip` 이 정한다(이름은 안 본다)."""
    if is_gzip(p):
        return gzip.open(p, "rt", encoding="utf-8", errors=errors)
    return io.open(p, encoding="utf-8", errors=errors)


def iter_all_sims(want_tasks=None):
    """전 코퍼스 순회 — (tag, sim) 를 yield. `want_tasks` 가 있으면 그 태스크만."""
    for p in all_result_files():
        try:
            with topen(p) as _f:
                raw = _f.read()
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
    with topen(p, errors="strict") as f:
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
    """우리 채널 사이드카를 **조인키(simtag) 별로**. 없으면 빈 dict.

    ⚠2026-08-23 수리(R2): 이 함수는 **리모트 평문 한 자리**(`FBDIR/fb_<tag>.jsonl`)만 봤다.
      로컬 영속본은 `sim_results/` 에 **110 개**(`fb_<tag>.jsonl.gz` 96 + `<tag>.fb.jsonl.gz` 14)
      있는데 그 전부가 안 읽혔고, 같은 파일을 `sidecar_rows` 는 (한 자리만) 읽고 있었다 —
      한 정본 안에서 리더가 둘로 갈린 자리다([[67]]). 경로 결의는 이제 `sidecar_paths` 하나다.
    """
    out = collections.defaultdict(list)
    for o in sidecar_rows(tag):
        out[o.get("simtag") or o.get("sim") or "?"].append(o)
    return out


# ── 사이드카 = **우리 층 거절의 권위** ────────────────────────────────────────
# ★왜 권위인가 (2026-08-23·refute_1⑸·refute_4⑵·refute_6⑶ 실측):
#   우리 거절은 재생성 채널로 나가고 `_ap_regen` 이 **원 어시스턴트 메시지를 교체**한다. 그래서
#   막힌 호출은 영속 `sim["messages"]` 에도, 그래서 `mutation_diff` 의 BLOCKED 칸에도 **없다**.
#   실측: `[OFFICIAL-NAME]`·`[SIGNATURE]` 는 영속 결과파일 462 개 전량에서 **0 건**인데 같은 런의
#   사이드카(110 파일)에는 각각 **226 행(24 런)·730 행(59 런)** 이 sim·turn·본문 전문과
#   함께 있다(2026-08-23 이 저장소에서 재측정한 수·refute_1 의 188 은 96 파일 부분집합 기준).
#   그 공백을 읽고 세 건의
#   반증이 *"우리 층 표지가 없으니 env 가 했다"* 로 갔다 — **침묵을 증거로 읽은 것**이다([[25]]).
#   ⇒ 우리 층 거절의 유무는 사이드카에 묻고, 사이드카가 **없으면 '모른다'로 남긴다**.
SIDECAR_NAMES = ("fb_%s.jsonl.gz", "%s.fb.jsonl.gz", "fb_%s.jsonl", "%s.fb.jsonl")


def sidecar_paths(tag):
    """사이드카 파일 경로 **전부**(존재하는 것만·정본 결의). 로컬 영속본 → 리모트 원본 순.

    명명이 두 가지다(`fb_<tag>.jsonl.gz` 96 · `<tag>.fb.jsonl.gz` 14) — `all_result_files` 가
    결과파일 두 명명에서 겪은 것과 **같은 사고**이므로 여기서 한 번에 닫는다([[67]]).
    """
    out = []
    for pat in SIDECAR_NAMES:
        p = os.path.join(BASE, pat % tag)
        if os.path.exists(p):
            out.append(p)
    p = os.path.join(FBDIR, "fb_%s.jsonl" % tag)
    if os.path.exists(p):
        out.append(p)
    return out


def sidecar_status(tag):
    """`'present'` | `'absent'` — **침묵을 증거로 읽지 않기 위한 계기판**.

    `absent` 는 *"우리 층이 아무것도 안 막았다"* 가 아니라 *"막았는지 모른다"* 다. t7305 이후
    어떤 stage-1 러너도 사이드카를 회수하지 않아 t7336·t7346 은 이 값이 `absent` 다.
    """
    return "present" if sidecar_paths(tag) else "absent"


def sidecar_note(tag):
    """포렌식이 **그대로 인쇄할 한 줄** — 없을 때 무엇을 하면 풀리는지까지 적는다([[64]])."""
    ps = sidecar_paths(tag)
    if ps:
        return "[sidecar] present: %s" % os.path.basename(ps[0])
    return ("[sidecar] ABSENT for %s — 우리 층 거절 유무는 **판정 불가**다(침묵≠부재·[[25]]). "
            "러너의 회수 블록(`fb_$TAG.jsonl` → `sim_results/fb_$TAG.jsonl.gz`)이 돌아야 "
            "이 칸이 채워진다." % tag)


def fb_sim_fingerprint(sim):
    """사이드카의 `sim` 필드(=첫 user 발화 sha1[:12])를 영속 sim 에서 **같은 규칙으로** 만든다.

    `t2_fbsidecar._sim_key` 와 바이트 동일한 규칙이다(사본 아님·조인용 역함수).
    ⚠거칠다: user-sim temp 0.0 에서는 같은 태스크의 nt 시행이 **한 키로 병합**된다. 그래서
      조인은 `simtag` 를 먼저 쓰고, 이 지문은 `simtag` 가 없는 옛 런에서만 폴백으로 쓴다.
    """
    import hashlib as _h
    for m in (sim.get("messages") or []):
        if m.get("role") != "user":
            continue
        c = m.get("content")
        if isinstance(c, str) and c.strip():
            return _h.sha1(c.strip().encode("utf-8")).hexdigest()[:12]
    return "nouser"


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
        with topen(gzp) as f:                    # 이름이 `.gz` 라도 매직 바이트가 정한다
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
    영속본 → 리모트 원본 순으로 본다. 없으면 빈 리스트(침묵).
    ⚠2026-08-23 수리(R2·사이드카와 **같은 결손**): 이 함수는 `<tag>.trace.jsonl.gz` 한 명명만
      봤고 로컬에는 `trace_<tag>.jsonl.gz` 가 **57 개** 더 있다(그 명명 22 ↔ 이 명명 57).
      그래서 `turns_of` 가 그 57 런에서 조용히 로그-텍스트 폴백으로 떨어져 turn 을 8% 만 얻었다.
    """
    import gzip as _gz
    out = []
    for p in trace_paths(tag):
        with topen(p) as f:
            for ln in f:
                try:
                    out.append(json.loads(ln))
                except Exception:
                    continue
        break
    return out


TRACE_NAMES = ("trace_%s.jsonl.gz", "%s.trace.jsonl.gz", "trace_%s.jsonl", "%s.trace.jsonl")


def trace_paths(tag):
    """구조화 계기 파일 경로 전부(존재하는 것만). 로컬 영속본 → 리모트 원본 순."""
    out = []
    for pat in TRACE_NAMES:
        p = os.path.join(BASE, pat % tag)
        if os.path.exists(p):
            out.append(p)
    p = os.path.join(FBDIR, "trace_%s.jsonl" % tag)
    if os.path.exists(p):
        out.append(p)
    return out


def sidecar_rows(tag):
    """우리 층이 **보낸 문장**(`fb_<tag>.jsonl`) — `{sim, turn, channel, text, …}` 행 리스트.

    trace 가 *어느 기구가 말했는가*를 남기면 이쪽은 *무엇을 말했는가*를 남긴다(둘은 다르다).
    실측(t7310 treat): 266 줄 · sim 12 · 채널 17종 · **turn 없는 줄 0**.
    ⚠2026-08-23 수리(R2): 경로 결의를 `sidecar_paths` 로 옮겼다 — 이 함수는 `<tag>.fb.jsonl.gz`
      한 명명만 봐서 `fb_<tag>.jsonl.gz` **96 개**를 통째로 놓치고 있었다.
    """
    import gzip as _gz
    out = []
    for p in sidecar_paths(tag):
        with topen(p) as f:
            for ln in f:
                try:
                    out.append(json.loads(ln))
                except Exception:
                    continue
        break                              # 한 명명만 채택(중복 명명은 같은 런의 사본이다)
    return out


def sidecar_denies(tag):
    """우리 층이 **도구 호출을 반려한** 행만 → `{'simtag': {...}, 'fp': {...}}` 두 색인.

    닫힌 술어다: `kind == 'tool-deny'` 는 `t2_fbsidecar.record_many` 가 *role=tool 이고
    error=True 인 우리 메시지*에만 붙이는 라벨이다(우리 파일 소유·해석 0).
    코퍼스 실측(사이드카 110 파일): tool-deny **8,145 행** · 그중 `simtag` 보유 6,860.
    """
    by_st, by_fp = collections.defaultdict(list), collections.defaultdict(list)
    for r in sidecar_rows(tag):
        if r.get("kind") != "tool-deny":
            continue
        if r.get("simtag"):
            by_st[r["simtag"]].append(r)
        if r.get("sim"):
            by_fp[r["sim"]].append(r)
    return {"simtag": by_st, "fp": by_fp}


def regen_blocked(sim, tag=None, idx=None):
    """★재생성으로 **영속 궤적에서 지워진** 우리 층 반려 → `(상태, 조인방식, 행들)`.

    상태 = `'present'`(사이드카 있음·행 수가 곧 답) · `'absent'`(사이드카 없음) ·
           `'unknown'`(호출부가 tag 를 안 줬다). **`absent`/`unknown` 에서 빈 리스트를
           '안 막혔다'로 읽지 마라** — 그 오독이 057 5 sim 을 *'모델이 시도 안 함'* 으로
           찍게 만든 자리다(refute_6⑶).
    조인 = `'simtag'`(정확) · `'fp'`(첫-유저-발화 지문·**nt 시행이 병합된다·거칢**) · None.
    """
    if not tag:
        return "unknown", None, []
    if sidecar_status(tag) == "absent":
        return "absent", None, []
    ix = sidecar_denies(tag) if idx is None else idx
    rows = ix["simtag"].get(simtag(sim))
    if rows:
        return "present", "simtag", rows
    rows = ix["fp"].get(fb_sim_fingerprint(sim))
    if rows:
        return "present", "fp", rows
    return "present", None, []


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

# ─────────────────────────────────────────────────────────────────────────────
# 우리 표지 원장 — **누가 그 표지를 소유하는가**를 파일로 판정한다(열거·판단 0)
# ─────────────────────────────────────────────────────────────────────────────
# ★결손 (2026-08-23·refute_1⑷ 실측·전 코퍼스 13,534 sim 재현):
#   `deny_kind` 는 위 두 줄짜리 `OURS_DENY` 만 우리 것으로 셌다. 영속 tool 본문에서 표지로
#   시작하는 거절 **3,650 건**의 실제 분포는 —
#       ours 로 제대로:  READ-FIRST 224
#       env 로 오귀속:   BYREF 398 · ARGS-FORMAT 108 · PRE-ACTION-KB 24 · RESULT-SIGN 1 = 531
#       거절로 안 세짐:  DUPLICATE-READ 2,340 · DUPLICATE-COMPUTE 214 = 2,554
#         (같은 자리에 `POLICY_QA` 341 · `GROUNDING WARNING` 508 이 섞여 있으나 그 둘은 거절이
#          **아니다** — 아래 «거절 문면 원장» ⓑ 참조. 표지 소유만으로 세면 성공 호출이 막힌
#          것으로 찍힌다.)
#   마지막 부류는 `('', None)` 을 돌려줘 `attempted_mutations` 의 `ok = not kind` 를 **True** 로
#   만든다 = *막힌 변이를 실행된 것으로 센다*(변이 도구 위 27 건 실측·`mutation_diff` 의
#   done/dup 칸 직접 오염). 즉 이것은 레버가 아니라 **자[尺]의 눈금 결손**이다.
#
# ★판정 방식 = 닫힌 술어 둘, 열거 0 ([[59]]·[[66]]):
#   ⑴ **소유** — 표지 문자열이 *우리 파일*(이 디렉터리의 `t2_*.py` · `a2/**.json`)에 있는가.
#   ⑵ **발화 증언** — 사이드카(우리 층이 보낸 문장의 원장)에 그 표지로 시작하는 행이 있는가.
#   둘 다 집합 소속 검사다. 표지 이름은 **우리 채널 식별자**이지 도메인 어휘가 아니며,
#   목록은 손으로 적지 않고 **파일에서 유도**한다(적으면 그 순간 갈라진다·[[67]]).
#   실측 정합: 영속 본문에 실제로 나타나는 표지는 위 **8 종뿐**이고 8/8 이 ⑴로 잡힌다
#   (= env 가 저작한 표지-머리 본문은 이 코퍼스에 **0 건**이다).
OUR_SOURCE_GLOBS = ("t2_*.py", os.path.join("a2", "*.json"), os.path.join("a2", "*", "*.json"))
# 본문 머리의 표지: 앞에 붙을 수 있는 것은 프레임워크 접두 둘뿐(`Error:` = tau2 규약 · `Note:` =
# 우리 알림 채널). 표지 자체는 대문자로 시작하는 대문자/숫자/`_`/`-` 토큰이다.
_MARKER_RX = None
_OUR_MARKERS = None
_ATTESTED = None


def _marker_rx():
    global _MARKER_RX
    if _MARKER_RX is None:
        import re as _re
        _MARKER_RX = _re.compile(r"^(?:Error:\s*|Note:\s*)?\[([A-Z][A-Z0-9_\-]{2,})\]")
    return _MARKER_RX


def marker_of(body):
    """본문 **머리**의 표지 토큰(대괄호 없이) 또는 None. 본문 중간의 대괄호는 보지 않는다.

    머리만 보는 이유: KB 문서 인용 안의 대괄호를 거절로 접으면 **성공한 write 가 막힌 것으로**
    찍힌다(`OURS_DENY` 주석의 substring 사고와 같은 방향).
    """
    m = _marker_rx().match((body or "").lstrip())
    return m.group(1) if m else None


def our_markers():
    """⑴**소유** 원장 — 우리 파일에 리터럴로 있는 표지 집합(1회 계산·캐시)."""
    global _OUR_MARKERS
    if _OUR_MARKERS is None:
        import glob as _g
        import re as _re
        rx = _re.compile(r"\[([A-Z][A-Z0-9_\-]{2,})\][ .,:]")
        got = set()
        for pat in OUR_SOURCE_GLOBS:
            for p in _g.glob(os.path.join(HERE, pat)):
                try:
                    with io.open(p, encoding="utf-8", errors="replace") as f:
                        got.update(rx.findall(f.read()))
                except Exception:
                    continue
        _OUR_MARKERS = frozenset(got)
    return _OUR_MARKERS


def attested_markers():
    """⑵**발화 증언** 원장 — 사이드카 전량에서 *머리 표지*로 실제 나간 집합(1회 계산·캐시).

    소유 원장이 못 잡는 자리를 여기서 받는다: 재생성 채널로만 나가는 표지는 우리 파일에
    있어도 형태가 다를 수 있고, 반대로 사이드카에는 **간 문장 그대로** 남는다.
    """
    global _ATTESTED
    if _ATTESTED is None:
        import glob as _g
        import gzip as _gz
        got = set()
        seen = set()
        for pat in ("fb_*.jsonl.gz", "*.fb.jsonl.gz", "fb_*.jsonl", "*.fb.jsonl"):
            for p in _g.glob(os.path.join(BASE, pat)):
                if p in seen:
                    continue
                seen.add(p)
                try:
                    with topen(p) as f:
                        for ln in f:
                            try:
                                o = json.loads(ln)
                            except Exception:
                                continue
                            mk = marker_of(o.get("text") or "")
                            if mk:
                                got.add(mk)
                except Exception:
                    continue
        _ATTESTED = frozenset(got)
    return _ATTESTED


_NOTICE_LEDGER = None
# 우리가 **통지**를 실어 나르는 A2 키(닫힌 집합 — 우리 파일의 키 이름이지 도메인 텍스트가 아니다)
NOTICE_KEY_SUFFIXES = ("_note", "_notice", "notice_text", "_template", "_hint")
NOTICE_MIN = 40


def our_notice_ledger(refresh=False):
    """⑶**통지 문면** 원장 {축자조각: 출처파일} — *거절이 아닌* 우리 층 문장(1회 계산·캐시).

    ⑴⑵ 와 `ours_deny_prefixes` 는 셋 다 **거절**을 겨눈다. 그런데 우리 층은 거절이 아닌 문장도
    궤적에 싣는다 — 성공 출력 앞의 주석과 A2 가 선언한 통지가 그것이다. 그 둘을 못 가르면
    *우리가 낸 문장*을 세는 소비자(`t2_gap.rival_text`)가 손 목록을 따로 들게 된다([[67]]).

      ⓐ **공백을 품은 머리 표지** — `[GROUNDING WARNING]`(`t2_scaffold_get.py`). `our_markers()`
         의 정규식은 표지 안의 공백을 안 받는다. 그 정규식을 넓히면 `deny_kind` ⓷ 의 소유
         판정이 같이 움직여 **R1 이 방금 교정한 자[尺]의 눈금이 바뀐다** ⇒ 넓히지 않고
         **별도 원장**으로 받는다. 엔진 파일에서만 유도한다(A2 산문의 대괄호를 안 집으려고).
      ⓑ **A2 선언 통지** — `unverified_note`·`notice_text` 처럼 `Error:` 를 안 쓰는 문면.
         키 이름이 닫힌 접미 집합에 들면 그 값의 앞 `NOTICE_MIN` 자를 조각으로 쓴다. 통지는
         **덧붙여** 나가므로(본문 중간) 접두가 아니라 **포함**으로 본다.

    손 목록 0 — 둘 다 파일에서 유도하고 출처를 함께 돌려준다(`ours_deny_prefixes` 와 같은 규율).
    ⚠이 독스트링에 `%` 보간을 쓰지 마라 — 그러면 첫 문장이 `Expr(BinOp)` 가 되어 **독스트링이
      아니게 되고**(`__doc__ is None`), 아래 독스트링 제외가 이 파일 자신을 못 걸러 표지 출처가
      인쇄 자리 대신 이 설명으로 찍힌다. 초판이 정확히 그렇게 틀렸다.
    """
    global _NOTICE_LEDGER
    if _NOTICE_LEDGER is None or refresh:
        import glob as _g
        import re as _re
        got = {}
        import ast as _ast
        rx = _re.compile(r"\[([A-Z][A-Z0-9_\- ]{2,})\]")
        # ⚠**문자열 리터럴에서만** 뽑는다. 파일 전문을 훑으면 *주석에 적힌 표지*까지 들어와
        #   출처가 인쇄 자리가 아닌 언급 자리로 찍힌다(초판이 `[GROUNDING WARNING]` 을
        #   `t2_scaffold_get.py` 가 아니라 이 파일의 주석으로 귀속했다). 출처를 못 대면
        #   그것은 다시 손 목록이다 — `ours_deny_prefixes` 와 같은 규율.
        for p in sorted(_g.glob(os.path.join(HERE, "t2_*.py"))):
            try:
                with io.open(p, encoding="utf-8", errors="replace") as f:
                    tree = _ast.parse(f.read())
            except Exception:
                continue
            # 독스트링도 `ast.Constant` 다 — 문장으로 서 있는 문자열(`Expr`)은 인쇄가 아니라
            # 설명이므로 뺀다. 이 파일의 `deny_kind` 독스트링이 `[GROUNDING WARNING]` 을
            # 언급하는 탓에 출처가 다시 언급 자리로 찍혔다.
            _docs = {id(n.value) for n in _ast.walk(tree)
                     if isinstance(n, _ast.Expr) and isinstance(n.value, _ast.Constant)}
            for node in _ast.walk(tree):
                if not (isinstance(node, _ast.Constant) and isinstance(node.value, str)):
                    continue
                if id(node) in _docs:
                    continue
                for tok in rx.findall(node.value):
                    if " " not in tok:
                        continue                  # 공백 없는 표지는 ⑴ 이 이미 갖고 있다
                    got.setdefault("[%s]" % tok, os.path.basename(p))

        def walk(o, src, key=None):
            if isinstance(o, dict):
                for k, v in o.items():
                    walk(v, src, k)
            elif isinstance(o, list):
                for v in o:
                    walk(v, src, key)
            elif isinstance(o, str) and key:
                # `_` 로 시작하는 키는 A2 의 **출처 주석**이다(`_note_` 규약·[[23]]) — 나가는
                # 문장이 아니므로 원장에 넣으면 메모리 링크까지 "우리가 낸 문장"이 된다.
                if str(key).startswith("_"):
                    return
                if not str(key).endswith(NOTICE_KEY_SUFFIXES):
                    return
                s = " ".join(o.split())
                if len(s) < NOTICE_MIN or s.startswith("Error:"):
                    return                        # 거절은 `ours_deny_prefixes` 몫이다
                got.setdefault(s[:NOTICE_MIN], src)

        for p in _g.glob(os.path.join(HERE, A2_SOURCE_GLOB), recursive=True):
            try:
                with io.open(p, encoding="utf-8") as f:
                    walk(json.load(f), os.path.basename(p))
            except Exception:
                continue
        _NOTICE_LEDGER = got
    return _NOTICE_LEDGER


# ─────────────────────────────────────────────────────────────────────────────
# 거절 **문면** 원장 — 우리 파일이 저작한 거절 본문의 축자 접두(파일에서 유도·열거 0)
# ─────────────────────────────────────────────────────────────────────────────
# ★소유 원장(위 ⑴⑵)만으로 못 가르는 자리가 둘 남는다 — 전 코퍼스 462 파일 실측(2026-08-23):
#   ⓐ **동적 표지**: `Error: [POLICY GATE {gid}]` 는 표지 토큰이 코드에 리터럴로 없다. 영속
#      본문 **764 건**(G4_TRANSFER_MSG 274·G1_AUTH_FIRST 223·RETRY_LOOP 161·G7 28·RETRY_ESCALATE
#      28·G2 25·G5 23·G3 2)이 소유 원장에서 빠져 `env` 로 떨어진다 = **우리 게이트가 막은 것을
#      환경 탓으로** 돌린다. 이것이 이 결함의 최대 덩어리다.
#   ⓑ **우리 표지를 단 성공 본문**: 우리 표지가 붙었다고 다 거절인 것이 아니다.
#      `[POLICY_QA] {answer}` 는 a2 의 **`return_template`**(= 기능-서브가 **답한** 것·341 건),
#      `[GROUNDING WARNING] …` 은 성공 결과 **앞에 덧붙인 주석**(508 건·뒤에 원 출력이 그대로
#      붙는다)이다. 표지 소유만 보고 ours-거절로 접으면 **성공한 호출이 `ok=False`** 가 된다 —
#      고치려던 것과 반대 방향의 같은 오분류다.
# ⇒ 표지 **이름**이 아니라 **문면 자체**를 원장으로 삼는다. 두 출처에서 기계로 뽑는다:
#   · 엔진(`t2_*.py`) — AST 로 *도구-메시지 본문 자리*의 **선두 리터럴**만: `content=` 인자 ·
#     `content=` 로 넘어가는 이름에의 대입 · `return`. 대입이 **자기 자신을 참조**하면
#     (`_txt = "[X] …%s" % (…, _txt)` = 기존 본문에 덧붙이는 주석) 제외한다 — 이 한 줄이
#     `[GROUNDING WARNING]` 을 손 목록 없이 걸러낸다.
#   · A2 선언 — 값이 `Error: [` 로 시작하는 문자열(= tau2 오류 채널로 나가는 우리 문면).
#     성공 문면은 이 접두를 쓰지 않으므로 **키 목록 없이** 갈린다(`return_template` 이 증인).
# 각 문면은 첫 **자리표시자**(`%s`·`{x}`) 앞까지 잘라 축자 접두로 쓴다. 접두 비교는 닫힌
# 술어이고, 게이트가 늘면 파일에서 자동으로 따라온다 — 손으로 적은 목록은 그 순간 갈린다([[67]]).
ENGINE_SOURCE_GLOB = "t2_*.py"
A2_SOURCE_GLOB = os.path.join("a2", "**", "*.json")
# 접두 하한: 이보다 짧으면 env 본문과 충돌한다. 실물 하한 사례 = a2 의 `Error: '{name}' is
# missing …` → 접두 `Error: '`(8자). 그 8자로 재면 env 의 `Error: 'x' …` 형 본문까지 삼킨다.
DENY_PREFIX_MIN = 10
_DENY_PREFIXES = None
_DENY_RX = None
_PH_RX = None
_HEAD_RX = None


def _head_rx():
    """접두가 **표지 머리**로 시작하는가. `marker_of` 와 달리 닫는 대괄호를 요구하지 않는다 —
    동적 표지(`Error: [POLICY GATE {gid}]`)는 잘린 자리가 대괄호 **안**이기 때문이다."""
    global _HEAD_RX
    if _HEAD_RX is None:
        import re as _re
        _HEAD_RX = _re.compile(r"^(?:Error:\s*|Note:\s*)?\[[A-Z]")
    return _HEAD_RX


def _ph_rx():
    """포맷 자리표시자(`%s`·`%-8.2f`·`{name}`)."""
    global _PH_RX
    if _PH_RX is None:
        import re as _re
        _PH_RX = _re.compile(r"%[-+ #0-9.*]*[sdrifgeoxXc%]|\{[^{}]*\}")
    return _PH_RX


def literal_prefix(s):
    """포맷 문면 → 첫 자리표시자 앞까지의 **축자 접두**(공백 정규화)."""
    s = " ".join((s or "").split())
    m = _ph_rx().search(s)
    return s[:m.start()] if m else s


def _lead_consts(node):
    """식이 만들어내는 문자열의 **선두 리터럴**들(분기는 전부 모은다)."""
    import ast as _ast
    out = []
    if isinstance(node, _ast.Constant) and isinstance(node.value, str):
        out.append(node.value)
    elif isinstance(node, _ast.JoinedStr):          # f"..." → 첫 조각만
        for v in node.values:
            if isinstance(v, _ast.Constant) and isinstance(v.value, str):
                out.append(v.value)
            break
    elif isinstance(node, _ast.BinOp):              # a + b · fmt % args
        out += _lead_consts(node.left)
    elif isinstance(node, _ast.IfExp):              # x if c else y → 양쪽
        out += _lead_consts(node.body) + _lead_consts(node.orelse)
    elif isinstance(node, _ast.BoolOp):             # a or b → 양쪽
        for v in node.values:
            out += _lead_consts(v)
    elif isinstance(node, _ast.Tuple) and node.elts:   # return (body, flag)
        out += _lead_consts(node.elts[0])
    elif (isinstance(node, _ast.Call) and isinstance(node.func, _ast.Attribute)
          and node.func.attr == "format"):
        out += _lead_consts(node.func.value)
    return out


def _engine_deny_prefixes():
    """엔진 소스 AST → 도구-메시지 본문 자리의 선두 리터럴 접두 {접두: 파일}."""
    import ast as _ast
    import glob as _g
    got = {}
    for p in sorted(_g.glob(os.path.join(HERE, ENGINE_SOURCE_GLOB))):
        if os.path.abspath(p) == os.path.abspath(__file__):
            continue                       # 자기 자신(포렌식 라이브러리)은 발화자가 아니다
        try:
            with io.open(p, encoding="utf-8") as f:
                txt = f.read()
        except Exception:
            continue
        if "content=" not in txt:          # 도구-메시지 본문을 만들지 않는 모듈은 건너뛴다(속도)
            continue
        try:
            tree = _ast.parse(txt, p)
        except Exception:
            continue                       # 파싱 불가 모듈은 조용히 뺀다(원장이 작아질 뿐)
        base = os.path.basename(p)
        mod = {"names": set(), "lits": [], "assign": []}
        scopes = [mod]
        stack = [(tree, mod)]
        while stack:                       # 재귀 없이 1회 패스 — 스코프는 가장 가까운 함수
            node, sc = stack.pop()
            if isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef, _ast.Lambda)):
                sc = {"names": set(), "lits": [], "assign": []}
                scopes.append(sc)
            if isinstance(node, _ast.Call):
                for kw in (node.keywords or []):
                    if kw.arg == "content":
                        sc["lits"] += _lead_consts(kw.value)
                        if isinstance(kw.value, _ast.Name):
                            sc["names"].add(kw.value.id)
            elif isinstance(node, _ast.Assign):
                tgt = {t.id for t in node.targets if isinstance(t, _ast.Name)}
                if tgt:
                    sc["assign"].append((tgt, node.value))
            elif isinstance(node, _ast.Return) and node.value is not None:
                sc["lits"] += _lead_consts(node.value)
            for ch in _ast.iter_child_nodes(node):
                stack.append((ch, sc))
        for sc in scopes:
            for tgt, val in sc["assign"]:
                hit = tgt & (sc["names"] | {"content"})
                if not hit:
                    continue
                used = {n.id for n in _ast.walk(val) if isinstance(n, _ast.Name)}
                if used & hit:
                    continue               # 자기 참조 = 기존 본문에 덧붙이는 주석(거절 아님)
                sc["lits"] += _lead_consts(val)
            for v in sc["lits"]:
                got.setdefault(v, base)
    return got


def _a2_deny_prefixes():
    """A2 선언 → `Error: [` 로 시작하는 문면 {원문: 파일}."""
    import glob as _g
    got = {}

    def walk(o, src):
        if isinstance(o, dict):
            for v in o.values():
                walk(v, src)
        elif isinstance(o, list):
            for v in o:
                walk(v, src)
        elif isinstance(o, str) and o.lstrip().startswith("Error: ["):
            got.setdefault(o.lstrip(), src)

    for p in _g.glob(os.path.join(HERE, A2_SOURCE_GLOB), recursive=True):
        try:
            with io.open(p, encoding="utf-8") as f:
                walk(json.load(f), os.path.basename(p))
        except Exception:
            continue
    return got


def ours_deny_prefixes(refresh=False):
    """우리 층이 저작한 **거절 본문 접두** 원장 {접두: 출처파일}(1회 계산·캐시).

    감사용으로 그대로 출력해도 되게 출처를 함께 돌려준다 — 어떤 접두가 어디서 왔는지 못 대면
    그것은 다시 손 목록이다([[23]] 와 같은 규율: 출처를 못 대는 항목은 넣지 않는다).
    """
    global _DENY_PREFIXES, _DENY_RX
    if _DENY_PREFIXES is None or refresh:
        import re as _re
        raw = {}
        raw.update(_engine_deny_prefixes())
        raw.update(_a2_deny_prefixes())
        out = {}
        for lit, src in raw.items():
            q = literal_prefix(lit)
            # ★표지가 닫히면 거기서 끊는다. 코퍼스는 **여러 엔진 판본**에 걸쳐 있고 문구는 판본마다
            #   손질된다 — 문장 전체를 접두로 쓰면 옛 런의 같은 거절이 안 잡힌다(자[尺]가 판본에
            #   따라 눈금이 달라지는 셈). 채널을 식별하는 것은 표지이지 뒤따르는 산문이 아니다.
            #   닫는 대괄호가 없는 것은 **동적 표지**(`Error: [POLICY GATE {gid}]`)뿐이고 그때만
            #   리터럴 머리를 그대로 쓴다.
            j = q.find("]")
            if j >= 0:
                q = q[:j + 1]
            if len(q) >= DENY_PREFIX_MIN and _head_rx().match(q):
                out.setdefault(q, src)
        # 접두끼리 포함 관계면 **짧은 쪽**만 남긴다(같은 판정·비교 횟수만 늘 뿐).
        keys = sorted(out, key=len)
        keep = {}
        for k in keys:
            if not any(k.startswith(s) for s in keep):
                keep[k] = out[k]
        _DENY_PREFIXES = keep
        _DENY_RX = (_re.compile("|".join(_re.escape(k) for k in sorted(keep, key=len)))
                    if keep else None)
    return _DENY_PREFIXES


def _deny_rx():
    ours_deny_prefixes()
    return _DENY_RX


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
    """도구-결과 본문이 거절인가 · 누가 거절했나 → ('', None) | ('ours'|'env'|'unknown', 표지).

    순서(전부 닫힌 술어 = 집합 소속·문자열 접두 비교뿐. 의미 판단 0·[[59]]):
      ⓵ 옛 두 문면(`OURS_DENY`) 은 종전대로 substring 으로 — 거동 보존.
      ⓶ **우리가 저작한 거절 문면**(`ours_deny_prefixes`·파일에서 유도)으로 시작하면 `ours`.
         동적 표지(`Error: [POLICY GATE …]`)와 `Error:` 를 안 쓰는 거절 스텁
         (`[DUPLICATE-READ]`·`[DUPLICATE-COMPUTE]`)이 여기서 잡힌다.
      ⓷ **머리 표지**가 있을 때 — 오류 채널(`ENV_FAIL_PREFIX`)로 나왔는가로 가른다.
         · 오류 채널 O + 표지를 우리가 소유·발화 → `ours`
         · 오류 채널 O + 원장 어디에도 없음   → `unknown` = *모른다*. **`env` 로 찍지 않는다**:
           확인 안 한 것을 단언하면 그 칸이 다시 오귀속의 근거가 된다([[25]]).
         · 오류 채널 X → **거절이 아니다**. 우리 표지를 달고 나가는 본문에는 성공 출력이 섞여
           있다 — `[POLICY_QA]`(a2 `return_template`·341 건)·`[GROUNDING WARNING]`(성공 결과
           앞의 주석·508 건). 표지 소유만 보고 접으면 성공 호출이 `ok=False` 가 된다.
      ⓸ 표지 없이 실패 접두면 `env`.
    ⚠`ok = not kind` 로 쓰는 쪽(`attempted_mutations`)에서 `unknown` 은 **막힌 것으로** 센다 —
      표지-머리 + 실패 접두는 구조상 거절이고, 누가 했는지만 미상이다.
    ⚠`T2_FORENSIC_DENY_LEGACY=1` 이면 ⓶⓷ 를 통째로 끄고 수리 전 판정(⓵+⓸)을 그대로 돌려준다 —
      옛 수치를 재현해 **차이가 어디서 났는지**를 같은 코드로 보이기 위한 것이다.
    """
    b = (body or "").lstrip()
    for p in OURS_DENY:
        if p in b:
            return "ours", p
    if os.environ.get("T2_FORENSIC_DENY_LEGACY") != "1":
        rx = _deny_rx()
        m = rx.match(" ".join(b.split())) if rx is not None else None
        if m:
            return "ours", m.group(0)[:60]
        mk = marker_of(b)
        if mk:
            for _p in ENV_FAIL_PREFIX:
                if b.startswith(_p):
                    owned = mk in our_markers() or mk in attested_markers()
                    return ("ours" if owned else "unknown"), "[%s]" % mk
            return "", None
    for _p in ENV_FAIL_PREFIX:
        if b.startswith(_p):
            return "env", b[:60]
    return "", None


def ours_text(body):
    """본문을 **우리 층이 저작했는가** → bool. 거절만이 아니라 주석·통지까지 포함한다.

    `deny_kind` 는 설계대로 **거절**만 가른다. 그런데 격리↔라이브 사다리(`t2_gap.rival_text`)
    처럼 *우리가 낸 문장이면 종류를 안 가리고* 집어야 하는 소비자가 있고, 정본에 그 술어가
    없어서 그쪽이 다섯 문자열짜리 **사본**을 들고 있었다([[67]] 위반). 그 사본을 여기로 올린다.

      ⓵ `deny_kind(body)[0] == 'ours'`                     — 저작 거절 전량
      ⓶ 머리 표지 ∈ `our_markers() ∪ attested_markers()`   — 공백 없는 표지
      ⓷ `our_notice_ledger()` 조각 **포함**                — 공백 표지 + A2 통지

    ★t7346(2026-08-22·40 sim·tool 메시지 993)로 사본과 대조한 실측:
        사본만 잡던 것 47 (`could not be verified` 33 · `GROUNDING WARNING` 14) ⇒ ⓷ 이 받는다
        정본만 잡던 것 45                                                      ⇒ 사본이 놓치던 것
        둘 다 잡던 것 51 (`NOT_VERIFIED`)
      즉 사본은 **양쪽으로** 갈라져 있었다 — `deny_kind` 로 그냥 바꿨으면 47 을 잃었다.
    ⚠전부 닫힌 술어다: 집합 소속과 문자열 포함뿐이고 의미 판단은 0 이다([[59]]).
    """
    b = (body or "").lstrip()
    if deny_kind(b)[0] == "ours":
        return True
    mk = marker_of(b)
    if mk and (mk in our_markers() or mk in attested_markers()):
        return True
    flat = " ".join(b.split())
    return any(k in flat for k in our_notice_ledger())


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


def mutation_diff(sim, mut=None, tag=None):
    """gold 변이 집합 ↔ 성공한 변이 집합 → MISSING · WRONGARG · EXTRA · MATCHED · BLOCKED.

    · MISSING  gold 변이인데 **성공한 같은 호출이 없다**
    · WRONGARG 같은 도구를 성공시켰는데 인자가 gold 와 다르다
    · EXTRA    gold 에 없는 도구를 성공시켰다 (050 의 승인 중복이 이 칸이다)
    · BLOCKED  시도했으나 거절당한 변이 (누가 거절했는지 `deny` 에 남는다)

    ★★BLOCKED 는 **영속 궤적이 보여주는 만큼만** 센다 — 그런데 우리 층 거절의 상당수는
      재생성 채널로 나가고 `_ap_regen` 이 원 어시스턴트 메시지를 **교체**하므로 그 호출은
      `messages` 에 아예 없다. ⇒ **BLOCKED 가 비었다고 '안 막혔다'가 아니다.**
      그 판정의 권위는 사이드카이고, `tag` 를 주면 아래 세 칸이 함께 온다:
        `sidecar`      'present' | 'absent' | 'unknown'(tag 미제공)
        `regen_blocked` 재생성으로 지워진 우리 층 반려 행(사이드카 원문·turn·channel 포함)
        `regen_join`   'simtag'(정확) | 'fp'(지문·nt 시행 병합·거칢) | None
      `sidecar != 'present'` 인데 두 칸이 비어 있으면 그것은 **모른다**는 뜻이다([[25]]).
      `iter_all_sims()` 는 `(tag, sim)` 를 주므로 호출부는 그 tag 를 그대로 넘기면 된다.
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
    sc, join, rb = regen_blocked(sim, tag)
    return {"gold": gold, "done": done, "blocked": blocked, "missing": missing,
            "wrongarg": wrong, "extra": extra, "matched": matched, "dup": dup,
            "sidecar": sc, "regen_join": join, "regen_blocked": rb,
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


def action_diff(sim, tag=None):
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
    sc, join, rb = regen_blocked(sim, tag)
    return {"basis": reward_basis(sim), "reward": (sim.get("reward_info") or {}).get("reward"),
            "rows": rows, "gold": gold, "tried": tried,
            "matched": matched, "missing": missing,
            "blocked": [r for r in missing if r["blocked"]],
            # ★`blocked` 는 영속 궤적이 보여주는 만큼만이다 — 재생성으로 지워진 반려는
            #   `regen_blocked` 에만 남고, `sidecar != 'present'` 면 그 칸은 *모른다*다.
            "sidecar": sc, "regen_join": join, "regen_blocked": rb,
            "strict_missing": [r for r in rows if not r["called_exact"]],
            "n_gold": len(gold), "n_matched": len(matched),
            "clean": not missing}
