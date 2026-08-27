# -*- coding: utf-8 -*-
r"""`freeze.py` 다중 hold 래칫 — **실물 FREEZE.json 사본**으로 검정 (git 0 · 모델 0 · 무료).

## 왜 (2026-08-28)

GPU 가 둘이 되면서 런 두 개가 동시에 돈다. 동결이 한 칸이면 두 가지가 조용히 깨진다:

    ⑴ 뒤에 뜬 런은 `--on` 이 거부돼 **동결 없이** 돈다
    ⑵ 먼저 끝난 런의 `--off` 가 **남의 동결까지** 풀어 버린다

어느 쪽이든 한 런이 [S] 를 잃는데 로그에는 정상으로 보인다. 이 검정이 고정하는 것:

  ① 구판 파일(hold 목록 없음·active)에 `--on` 하면 **한 칸짜리 목록으로 이관**되고 둘이 산다
  ② 각 hold 는 **자기 `path_hashes`** 로 판정된다(남의 변경에 물들지 않는다)
  ③ 태그 없는 `--off` 는 **가장 오래된 것 하나만** 풀고 나머지는 살아 있다
  ④ 마지막 hold 를 풀어야 `active` 가 내려간다
  ⑤ 같은 태그를 두 번 걸지 않는다

⚠ 이 검정은 `freeze.main()` 을 **직접 부르지 않는다** — 실제 repo 상태를 건드리게 되므로
   순수 로직만 재현할 수 없다. 대신 사본 파일 위에서 모듈의 `PATH` 만 바꿔 끼우고, git 을
   부르는 세 함수(`sha`·`now`·`path_hashes`·`dirty`)를 **결정론 스텁**으로 갈아 끼운다.
   갈아 끼운 것이 무엇인지 아래에 그대로 적어 둔다([[25]] 계기가 무엇을 쟀는지 말하게 한다).

사용: PYTHONPATH=. py -3 test_freeze_multihold.py
"""
import importlib.util
import io
import json
import os
import shutil
import sys
import tempfile

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
REP = os.path.abspath(os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026"))
FAIL = []


def chk(cond, what, extra=""):
    print("  %-4s %s%s" % ("ok" if cond else "FAIL", what, ("  %s" % extra) if extra else ""))
    if not cond:
        FAIL.append(what)


def load_module(tmp):
    spec = importlib.util.spec_from_file_location("freeze_t", os.path.join(REP, "freeze.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    m.PATH = tmp
    return m


def run(m, argv, hashes, head):
    """모듈의 git 창구를 스텁으로 갈고 main() 을 부른다. 반환 (exit, stdout+stderr)."""
    m.sha = lambda: head
    m.now = lambda: "2026-08-28T00:00:00+09:00"
    m.dirty = lambda: []
    m.path_hashes = lambda paths: dict(hashes)
    old_argv, old_out, old_err = sys.argv, sys.stdout, sys.stderr
    buf = io.StringIO()
    sys.argv = ["freeze.py"] + argv
    sys.stdout = sys.stderr = buf
    try:
        rc = m.main()
    finally:
        sys.argv, sys.stdout, sys.stderr = old_argv, old_out, old_err
    return rc, buf.getvalue()


def state(tmp):
    return json.load(io.open(tmp, encoding="utf-8"))


src = os.path.join(REP, "FREEZE.json")
if not os.path.exists(src):
    print("실물 FREEZE.json 이 없다 — 판정하지 않는다([[25]])")
    sys.exit(1)
live = json.load(io.open(src, encoding="utf-8"))
print("# 실물 FREEZE.json: active=%s tag=%s holds=%s"
      % (live.get("active"), live.get("tag"), "있음" if live.get("holds") else "없음(구판)"))

fd, tmp = tempfile.mkstemp(suffix=".json")
os.close(fd)
shutil.copyfile(src, tmp)
try:
    m = load_module(tmp)
    # 구판 한 칸짜리 active 상태를 만들어 둔다(실물이 이미 그 모양이면 그대로 쓴다)
    base = {"active": True, "sha": "aaaaaaa", "tag": "bank_tAAA", "reason": "first",
            "at": "2026-08-28T00:00:00+09:00", "paths": ["scripts/distill/tau2/a2/"],
            "path_hashes": {"scripts/distill/tau2/a2/": "h_old"}}
    io.open(tmp, "w", encoding="utf-8").write(json.dumps(base, ensure_ascii=False))

    print("")
    print("① 구판 active 위에 두 번째 hold 를 건다")
    rc, out = run(m, ["--on", "--tag", "bank_tBBB", "--reason", "second"],
                  {"scripts/distill/tau2/a2/": "h_new"}, "bbbbbbb")
    s = state(tmp)
    chk(rc == 0, "두 번째 --on 이 통과한다(종전엔 거부됐다)", out.strip()[:70])
    chk(len(s.get("holds") or []) == 2, "hold 가 둘이다", str([h.get("tag") for h in (s.get("holds") or [])]))
    chk(s.get("active") is True, "active 는 그대로 True")
    chk((s["holds"][0].get("path_hashes") or {}).get("scripts/distill/tau2/a2/") == "h_old"
        and (s["holds"][1].get("path_hashes") or {}).get("scripts/distill/tau2/a2/") == "h_new",
        "② 각 hold 가 **자기 기준**을 들고 있다")

    print("")
    print("③ 같은 태그를 또 걸지 않는다")
    rc2, _ = run(m, ["--on", "--tag", "bank_tBBB", "--reason", "dup"],
                 {"scripts/distill/tau2/a2/": "h_new"}, "bbbbbbb")
    chk(rc2 == 1, "중복 태그 --on 은 거부(exit 1)")
    chk(len(state(tmp).get("holds") or []) == 2, "거부됐으므로 hold 수 불변")

    print("")
    print("④ 태그 없는 --off 는 **가장 오래된 것 하나만** 푼다")
    rc3, out3 = run(m, ["--off"], {"scripts/distill/tau2/a2/": "h_new"}, "bbbbbbb")
    s = state(tmp)
    chk("bank_tAAA" in out3, "푼 것은 첫 번째 hold 다", out3.strip().splitlines()[0][:80])
    chk(rc3 == 2, "그 hold 는 기준이 h_old 였으므로 **뚫림으로 판정**된다(exit 2)")
    chk(s.get("active") is True, "남은 hold 가 있으므로 active 는 **유지**된다")
    chk([h.get("tag") for h in (s.get("holds") or [])] == ["bank_tBBB"], "남은 hold = 두 번째")
    chk(s.get("tag") == "bank_tBBB", "최상위 거울이 남은 hold 로 옮겨간다(구판 독자용)")

    print("")
    print("⑤ 마지막 hold 를 풀면 active 가 내려간다")
    rc4, out4 = run(m, ["--off", "--tag", "bank_tBBB"],
                    {"scripts/distill/tau2/a2/": "h_new"}, "bbbbbbb")
    s = state(tmp)
    chk(rc4 == 0, "기준이 그대로이므로 **유효**로 판정(exit 0)", out4.strip().splitlines()[-1][:70])
    chk(s.get("active") is False, "active=False")
    chk(len(s.get("released") or []) == 2, "푼 hold 둘이 기록에 남는다")

    print("")
    print("⑥ 없는 태그를 풀려고 하면 거부")
    io.open(tmp, "w", encoding="utf-8").write(json.dumps(base, ensure_ascii=False))
    rc5, _ = run(m, ["--off", "--tag", "bank_tZZZ"], {"scripts/distill/tau2/a2/": "h_old"}, "aaaaaaa")
    chk(rc5 == 1, "없는 태그 --off 는 exit 1")
    chk(state(tmp).get("active") is True, "그리고 아무것도 풀지 않는다")
finally:
    os.unlink(tmp)

print("")
print("RESULT: %s" % ("PASS" if not FAIL else "FAIL (%d) %s" % (len(FAIL), FAIL[:3])))
sys.exit(0 if not FAIL else 1)
