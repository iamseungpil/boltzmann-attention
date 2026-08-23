# -*- coding: utf-8 -*-
"""R2 — **우리 거절은 영속 궤적에 안 남는다. 자[尺]가 그것을 알아야 한다.** (2026-08-23)

## 무엇을 재는가
우리 층 거절은 재생성 채널로 나가고 `_ap_regen` 이 **원 어시스턴트 메시지를 교체**한다. 그래서
막힌 호출은 영속 `sim["messages"]` 에도, 따라서 `mutation_diff` 의 BLOCKED 칸에도 **없다**.
분석자는 그 공백을 *"우리 층 표지가 없으니 env 가 했다"* 로 읽었고, 그 아티팩트 위에 반증
셋이 세워졌다(refute_1⑷⑸ · refute_4⑵ · refute_6⑶). 이것은 레버가 아니라 **자의 눈금**이다.

## 세 결손 (전부 계기이지 모델이 아니다)
  ⒜ `sidecar()` 가 리모트 평문 한 자리만 봐서 **로컬 영속본 110 개**를 통째로 못 읽었다.
     같은 파일을 `sidecar_rows()` 는 (다른 한 명명만) 읽고 있었다 — 정본 안에서 리더가
     둘로 갈린 자리([[67]]). `trace()` 도 같은 병이었다(`trace_<tag>.jsonl.gz` 57 개 실명).
  ⒝ `deny_kind` 가 우리 표지 2 종만 셌다. 영속 본문의 표지-머리 거절 **3,650 건** 중
     ours 224 · **env 오귀속 531** · **거절로 안 세짐 2,895**. 마지막 부류는
     `attempted_mutations` 의 `ok = not kind` 를 True 로 만들어 *막힌 변이를 실행으로* 센다.
  ⒞ 사이드카가 **없을 때** 그 사실이 표에 안 나타나 침묵이 증거로 읽혔다([[25]]).

## 검정 구성 (전부 실물 코퍼스 — 픽스처만 통과하는 술어를 만들지 않기 위해)
  ① 양성대조   옛 술어를 이 파일 안에 동결해 두고 **결손을 재현**한다
  ② 수리 후    같은 입력에서 눈금이 맞는가
  ③ 부정통제   env 거절은 계속 env · 성공 본문은 계속 성공 · **모르면 'unknown'** 으로 남는가
  ④ 래칫      디스크의 사이드카/계기 파일 **전부**가 경로 결의에 잡히는가
"""
import collections
import glob
import gzip
import io
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import t2_forensic as F          # noqa: E402

fail = []


def check(name, ok, detail=""):
    print("  %-62s %s%s" % (name, "PASS" if ok else "FAIL", (" — " + detail) if detail else ""))
    if not ok:
        fail.append(name)


# ── 옛 술어 동결본 — 결손을 **재현**하기 위해서만 존재한다(엔진 경로 아님) ──────────────
def _old_deny_kind(body):
    b = body.lstrip()
    for p in ("[READ-FIRST]", "NOT_VERIFIED"):
        if p in b:
            return "ours", p
    for _p in ("Error:", "Failed to "):
        if b.startswith(_p):
            return "env", b[:60]
    return "", None


def _old_sidecar_rows(tag):
    """옛 경로 결의: `<tag>.fb.jsonl.gz` 한 명명 + 리모트 평문뿐."""
    gzp = os.path.join(F.BASE, tag + ".fb.jsonl.gz")
    if os.path.exists(gzp):
        return sum(1 for _ in gzip.open(gzp, "rt", encoding="utf-8", errors="replace"))
    raw = os.path.join(F.FBDIR, "fb_" + tag + ".jsonl")
    if os.path.exists(raw):
        return sum(1 for _ in io.open(raw, encoding="utf-8", errors="replace"))
    return 0


# 실물 정박점 — 전부 refute_1/4/6 이 축자로 지목한 자리다
T_SIDE = "bank_t7305_treat_20260817a"          # 사이드카 있음(`fb_<tag>.jsonl.gz` 명명)
T_MUT = "bank_y2cp2_gpu0_20260801"             # 변이 호출 위 [DUPLICATE-READ] 27 건
S_MUT = "task_008#s626729"
# ★사이드카가 **어디에도 없는** 태그. 실재 런을 여기 박으면 앵커가 썩는다 — 초판은
#   `bank_t7346_halfA_20260822` 를 *"미회수 런"* 으로 박았는데, 그 런의 사이드카는 안 쓰인 게
#   아니라 **안 회수된 것**이었고 원본이 리모트에 그대로 있었다(2026-08-24 에 366 개를 회수했다).
#   재는 것은 *"어떤 런이 미회수인가"* 가 아니라 *"없을 때 없다고 말하는가"* 이므로, 존재하지
#   않는 이름이 그 술어의 옳은 정박점이다.
T_NONE = "bank_no_such_run_00000000"

# 실물 정박점이 없는 체크아웃(얕은 클론 등)에서는 **건너뛰되 침묵하지 않는다** — 이 검정은
# 유료 런의 게이트 배터리에 들어가므로, 코퍼스 부재로 런을 막지는 않고 사실만 인쇄한다.
skipped = []


def have(*names):
    ok = all(F.sidecar_paths(n) or os.path.exists(F.path_for(n, "_results.json.gz"))
             or os.path.exists(F.path_for(n, ".results.json.gz")) for n in names)
    if not ok:
        skipped.append(", ".join(names))
    return ok


print("① 양성대조 — 옛 술어가 무엇을 놓쳤나 (결손 재현)")
check("옛 술어: `[DUPLICATE-READ]` 를 **거절로 세지 않았다**",
      _old_deny_kind("[DUPLICATE-READ] This exact call was already executed") == ("", None))
check("옛 술어: `Error: [BYREF]` 를 **env 로 오귀속**했다",
      _old_deny_kind("Error: [BYREF] refer to the record you read")[0] == "env")
# ⚠이 음성대조는 **리모트 원본이 없을 때**의 결손을 재현한다. 옛 결의는 두 자리를 봤다 —
#   `<tag>.fb.jsonl.gz`(점 명명)와 리모트 평문 — 그리고 `fb_<tag>.jsonl.gz`(밑줄 명명)를
#   놓쳤다. 이 기계에는 리모트 평문이 살아 있어서 옛 결의도 행을 읽는다. 그러니 그 자리를
#   **비워 놓고** 재현한다. 안 그러면 이 대조가 기계마다 참·거짓이 갈린다(2026-08-24 실물:
#   366 개 계기 파일을 회수하고 나서 이 줄이 무엇을 재는지가 드러났다).
_fbdir_saved = F.FBDIR
try:
    F.FBDIR = os.path.join(F.BASE, "__no_such_raw_dir__")
    _old_n = _old_sidecar_rows(T_SIDE)
finally:
    F.FBDIR = _fbdir_saved
check("옛 경로 결의: 리모트 원본이 없으면 `fb_<tag>.jsonl.gz` 를 **0 행**으로 읽었다",
      _old_n == 0, "옛 %d 행" % _old_n)
check("정본 결의는 같은 조건에서 읽는다(밑줄 명명)", len(F.sidecar_rows(T_SIDE)) > 0,
      "%d 행" % len(F.sidecar_rows(T_SIDE)))

print("\n② 수리 후 — 사이드카가 권위다")
mut = F.mutating_tools()
if have(T_SIDE):
    rows = F.sidecar_rows(T_SIDE)
    check("같은 태그를 이제 읽는다", len(rows) > 0, "%d 행" % len(rows))
    check("`sidecar()` 와 `sidecar_rows()` 가 **같은 파일**을 본다(리더 갈림 0)",
          sum(len(v) for v in F.sidecar(T_SIDE).values()) == len(rows))
    check("상태 계기: present", F.sidecar_status(T_SIDE) == "present")

    # 영속 궤적에는 없고 사이드카에는 있다 — 이 비대칭이 이 항목의 전부다
    # ⚠`path_for` 는 **라이브 심 디렉터리를 먼저** 본다 — 거기 있는 것은 평문 `results.json`
    #   인데 이름은 `.gz` 를 달라고 했으니, 이름을 믿고 gzip 으로 열면 `BadGzipFile` 로 터진다.
    #   이 줄이 그 함정에 걸려 이 검정을 통째로 죽이고 있었다(2026-08-24). 이름이 아니라
    #   매직 바이트로 정하는 정본 리더를 쓴다([[67]]).
    with F.topen(F.path_for(T_SIDE, "_results.json.gz")) as _f:
        raw = _f.read()
    n_persist = raw.count("OFFICIAL-NAME")
    n_side = sum(1 for r in rows if "OFFICIAL-NAME" in (r.get("text") or ""))
    check("영속 궤적에는 `[OFFICIAL-NAME]` 이 **0 건**", n_persist == 0, "%d" % n_persist)
    check("같은 런 사이드카에는 있다", n_side > 0, "%d 행" % n_side)

    sims_side = F.sims(T_SIDE, "_results.json.gz")
    n_regen = n_persisted_blocked = 0
    for s in sims_side:
        d = F.mutation_diff(s, mut, T_SIDE)
        n_regen += len(d["regen_blocked"])
        n_persisted_blocked += len(d["blocked"])
    check("`mutation_diff` 가 재생성으로 지워진 반려를 싣는다", n_regen > n_persisted_blocked,
          "regen %d ↔ 영속 blocked %d" % (n_regen, n_persisted_blocked))
    d0 = F.mutation_diff(sims_side[0], mut, T_SIDE)
    check("조인은 정확한 simtag 로 붙는다", d0["regen_join"] == "simtag", str(d0["regen_join"]))

# 변이 도구 위의 우리 거절 — 옛 눈금은 이것을 **실행**으로 셌다
if have(T_MUT):
    got = None
    for s in F.sims(T_MUT, ".results.json.gz"):
        if F.simtag(s) == S_MUT:
            got = s
            break
    check("실물 sim 을 찾았다 (%s|%s)" % (T_MUT, S_MUT), got is not None)
    if got is not None:
        tried = F.attempted_mutations(got, mut)
        old_ok = sum(1 for t in tried if not _old_deny_kind(t["result"])[0])
        new_ours = sum(1 for t in tried if t["deny"] == "ours")
        d = F.mutation_diff(got, mut, T_MUT)
        check("옛 눈금: 막힌 변이를 **실행**으로 셌다", old_ok > len(d["done"]),
              "옛 done %d → 지금 done %d" % (old_ok, len(d["done"])))
        check("지금: 그 호출들이 우리 층 BLOCKED 로 간다", new_ours > 0, "%d 건" % new_ours)
        check("BLOCKED 의 주체가 표기된다([[55]])",
              {t["deny"] for t in d["blocked"]} <= {"ours", "env", "unknown"},
              str(collections.Counter(t["deny"] for t in d["blocked"])))

print("\n③ 부정통제 — 과잉귀속·과잉계수 0")
check("env `Error:` 는 그대로 env", F.deny_kind("Error: nope")[0] == "env")
check("env `Failed to …` 도 그대로 env",
      F.deny_kind("Failed to log verification: Record may already exist.")[0] == "env")
check("env 소유 문면은 우리 것으로 안 센다",
      F.deny_kind("Error: The tool has not been given to you by the agent.")[0] == "env")
check("성공 본문은 거절이 아니다", F.deny_kind('{"ok": true}')[0] == "")
check("본문 **중간**의 대괄호는 표지가 아니다",
      F.deny_kind('{"doc": "see [SECTION] below"}')[0] == "")
check("원장에 없는 표지 + 실패 접두 → **unknown**(env 로 단언하지 않는다)",
      F.deny_kind("Error: [ZZQ_NOT_OURS] x") == ("unknown", "[ZZQ_NOT_OURS]"))
check("원장에 없는 표지 + 실패 접두 없음 → 거절 아님(과잉계수 0)",
      F.deny_kind("[ZZQ_NOT_OURS] x") == ("", None))

# ★침묵을 증거로 읽지 않는가 — 이 항목이 태어난 오독 그 자체
check("사이드카 미회수 런은 `absent` 로 드러난다", F.sidecar_status(T_NONE) == "absent")
sc, join, rb = F.regen_blocked({"task_id": "task_003", "seed": 1, "messages": []}, T_NONE)
check("그 런의 `regen_blocked` 는 `absent`(= 0 이 아니라 **모름**)",
      (sc, join, rb) == ("absent", None, []), str((sc, join)))
check("`sidecar_note` 가 무엇을 하면 풀리는지까지 말한다([[64]])",
      "ABSENT" in F.sidecar_note(T_NONE) and "sim_results" in F.sidecar_note(T_NONE))
if have(T_NONE):
    sims_none = F.sims(T_NONE, ".results.json.gz")
    dn = F.mutation_diff(sims_none[0], mut, T_NONE)
    check("표에도 그 사실이 실린다", dn["sidecar"] == "absent" and dn["regen_blocked"] == [])
    check("tag 를 안 주면 `unknown`(옛 호출부 호환 · 침묵 아님)",
          F.mutation_diff(sims_none[0], mut)["sidecar"] == "unknown")
    check("`action_diff` 도 같은 세 칸을 싣는다",
          set(F.action_diff(sims_none[0], T_NONE)) >= {"sidecar", "regen_join", "regen_blocked"})

# 원장 자체의 부정통제 — 소유 원장이 **env 문면까지** 삼키면 안 된다
om = F.our_markers()
check("소유 원장이 비어 있지 않다", len(om) > 0, "%d 종" % len(om))
JUNK = ("SECTION", "WARNING", "ERROR", "NOTE", "INFO", "SUCCESS", "RESULT")
check("원장에 일반어가 섞이지 않았다(과잉귀속 위험)",
      not [j for j in JUNK if j in om], str([j for j in JUNK if j in om]))

print("\n④ 래칫 — 디스크의 계기 파일 전부가 경로 결의에 잡히는가")
side_files = sorted(set(glob.glob(os.path.join(F.BASE, "fb_*.jsonl*")))
                    | set(glob.glob(os.path.join(F.BASE, "*.fb.jsonl*"))))
miss = []
for fp in side_files:
    b = os.path.basename(fp)
    tg = b[3:].split(".jsonl")[0] if b.startswith("fb_") else b.split(".fb.jsonl")[0]
    if fp not in F.sidecar_paths(tg):
        miss.append(b)
check("사이드카 %d 파일 전부가 `sidecar_paths` 에 잡힌다" % len(side_files),
      not miss, ("누락 %d: %s" % (len(miss), miss[:3])) if miss else "누락 0")
tr_files = sorted(set(glob.glob(os.path.join(F.BASE, "trace_*.jsonl*")))
                  | set(glob.glob(os.path.join(F.BASE, "*.trace.jsonl*"))))
miss = []
for fp in tr_files:
    b = os.path.basename(fp)
    tg = b[6:].split(".jsonl")[0] if b.startswith("trace_") else b.split(".trace.jsonl")[0]
    if fp not in F.trace_paths(tg):
        miss.append(b)
check("계기 %d 파일 전부가 `trace_paths` 에 잡힌다" % len(tr_files),
      not miss, ("누락 %d: %s" % (len(miss), miss[:3])) if miss else "누락 0")

print("\n⑤ 실물 분포 — 표지-머리 본문이 **env 로 오귀속되지 않는가**")
# 전 코퍼스(13,534 sim)는 75 초라 러너의 90 초 예산을 먹는다. 표지가 나오는 구간만 본다.
# ⚠검정 명제는 *"표지가 붙었으면 전부 거절"* 이 **아니다** — 우리 표지를 단 성공 본문이 있다
#   (`[POLICY_QA]` 답변 · `[GROUNDING WARNING]` 주석). 불변식은 하나다:
#   **env 는 우리 표지-머리 본문을 저작하지 않는다** ⇒ 그 자리에 `env` 가 찍히면 오귀속이다.
RX = re.compile(r"^(?:Error:\s*)?\[([A-Z][A-Z0-9_\-]{2,})\]")
cnt, mis, ours = collections.Counter(), collections.Counter(), 0
for fp in sorted(glob.glob(os.path.join(F.BASE, "bank_y2cp2_*results.json.gz")))[:2]:
    for s in F.sims(fp):
        for m in (s.get("messages") or []):
            if m.get("role") != "tool":
                continue
            b = " ".join(str(m.get("content") or "").split()).lstrip()
            mm = RX.match(b)
            if not mm:
                continue
            cnt[mm.group(1)] += 1
            k = F.deny_kind(b)[0]
            if k == "ours":
                ours += 1
            elif k == "env":
                mis[mm.group(1)] += 1
if have(T_MUT):
    check("표지-머리 본문이 실재한다", sum(cnt.values()) > 0,
          "%d 건 / %d 종" % (sum(cnt.values()), len(cnt)))
    check("그중 우리 층 거절이 실제로 귀속된다(수리가 문다)", ours > 0, "%d 건" % ours)
check("**env 로 찍히는 표지-머리 본문 0**(오귀속 0)", not mis,
      str(mis.most_common(5)) if mis else "오귀속 0")

print("\n⑥ 이음매 — 러너가 **쓰는 이름** ↔ 라이브러리가 **찾는 이름**")
# 이 둘이 갈리면 회수는 도는데 포렌식은 계속 침묵을 본다 — 두 파일에 걸친 유일한 계약이다.
check("사이드카 1순위 명명 = 러너 산출물", F.SIDECAR_NAMES[0] == "fb_%s.jsonl.gz")
check("계기 1순위 명명 = 러너 산출물", F.TRACE_NAMES[0] == "trace_%s.jsonl.gz")

RUNNERS = ("run_t7346_overnight_stage1_20260822.sh", "run_t7347_cp2ledger_smoke_20260823.sh")
for rn in RUNNERS:
    rp = os.path.join(HERE, rn)
    if not os.path.exists(rp):
        check("러너 %s 실재" % rn, False)
        continue
    txt = io.open(rp, encoding="utf-8", errors="replace").read()
    code = "\n".join(l for l in txt.split("\n") if not l.strip().startswith("#"))
    check("%s: fb·trace 두 계기를 함께 회수한다" % rn, "for _S in fb trace" in code)
    check("%s: `sim_results/` 로 gz 로 내린다" % rn,
          ".jsonl.gz" in code and "sim_results" in code)
    check("%s: 없을 때 **없다고 인쇄**한다(침묵 금지·[[64]])" % rn, "미회수" in code)
    # git 추적 — 회수만 하고 add 를 안 하면 리모트에만 남고 `sim_results/` 에는 안 온다.
    add_blk = "\n".join(l for l in code.split("\n")
                        if "sim_results" in l and (".jsonl.gz" in l or "git add" in l))
    check("%s: git 추적에 사이드카가 실린다" % rn, "fb_" in add_blk)
    check("%s: git 추적에 계기(trace)가 실린다" % rn, "trace_" in add_blk)
# ★스모크와 본런이 갈리면 또 한쪽만 회수한다(2026-08-06 사고의 뿌리) — stage-1 은 두 자리 다.
st = io.open(os.path.join(HERE, RUNNERS[0]), encoding="utf-8", errors="replace").read()
ncall = len([l for l in st.split("\n")
             if "harvest_instr " in l and not l.strip().startswith("#")])
check("stage-1 러너: 스모크와 본런 **양쪽**에서 회수한다", ncall >= 2, "호출 %d 자리" % ncall)

# 왕복 — 러너가 만든 이름을 라이브러리가 그대로 집는가(임시 BASE 로 격리)
import tempfile                                                       # noqa: E402
_old_base = F.BASE
try:
    td = tempfile.mkdtemp(prefix="t2fx_")
    F.BASE = td
    for nm in ("fb_zz_tag.jsonl.gz", "trace_zz_tag.jsonl.gz"):
        with gzip.open(os.path.join(td, nm), "wt", encoding="utf-8") as f:
            f.write('{"kind": "tool-deny", "simtag": "task_001#s1", "turn": 3, '
                    '"text": "Error: [SIGNATURE] re-issue with the declared argument only"}\n')
    check("왕복: 회수 이름을 `sidecar_paths` 가 집는다", len(F.sidecar_paths("zz_tag")) == 1)
    check("왕복: 회수 이름을 `trace_paths` 가 집는다", len(F.trace_paths("zz_tag")) == 1)
    check("왕복: 상태가 present 로 바뀐다", F.sidecar_status("zz_tag") == "present")
    sc, jn, rb = F.regen_blocked({"task_id": "task_001", "seed": 1, "messages": []}, "zz_tag")
    check("왕복: 그 반려가 sim 에 조인된다", (sc, jn, len(rb)) == ("present", "simtag", 1),
          str((sc, jn, len(rb))))
finally:
    F.BASE = _old_base

if skipped:
    print("\n⚠실물 정박점 부재로 건너뛴 절: %s — 코퍼스가 없는 체크아웃이다(런은 막지 않는다)"
          % "; ".join(sorted(set(skipped))))
print("\nRESULT: %s" % ("ALL PASS" if not fail else "FAIL %s" % fail))
sys.exit(1 if fail else 0)
