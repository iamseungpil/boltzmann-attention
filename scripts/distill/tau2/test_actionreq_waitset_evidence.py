# -*- coding: utf-8 -*-
"""`[T2_ACTIONREQ]` 대기집합 = **정적**, `[T2_PENDING_DISC]` = **죽은 배선** — 로그 전수 재현 검정.

수리 항목 **R8-pending-disc-dead**(2026-08-23)가 `T2_PENDING_DISCOVERED` 블록을 **제거**한
근거를, 로컬 `reports/facet_rft_2026/sim_results/*.log.gz` 전수 스캔으로 **다시 계산**한다.
`test_pending_discovered.py` 가 *소스 계약*을 지킨다면 이 파일은 *현상*을 지킨다.

## 재현하는 결함 (양성대조)
  ⑴ **레버 발화 0** — `[T2_PENDING_DISC]` 0줄(error no-op 조차 0)인데 같은 자리의 숙주
     `[T2_ACTIONREQ]` 는 만 줄 단위로 찍힌다 ⇒ "켜져 있다고 믿게 만드는" 죽은 배선.
  ⑵ **대기집합이 정적** — `pending_user` 토큰에 env 의 discoverable 손님 도구가 **0회**.
     그래서 런타임에 건네지는 도구는 넛지의 표적이 **될 수 없다**.
  ⑶ **종료 술어 부재** — 손님이 실행하는 도구는 `[T2_ACTIONREQ]` **전 줄**에 남아 있다.
     제거 경로 `_uacts - _effall` 의 `_effall` 은 `state.messages` 만 보는데 손님 실행은
     거기 없기 때문 ⇒ discoverable 을 더했다면 영원히 pending 이고 넛지가 안 끝난다([[57]]).

## 부정통제 (숫자가 탐지자 고장에서 나온 게 아님)
  ⓐ **제거 술어는 살아 있다** — 에이전트-측 래퍼 `call_discoverable_user_tool` 은 전 줄이
     아니라 **일부** 줄에서만 pending 이다(0 < n < 전체). 즉 ⑶의 100% 는 술어가 죽어서 나온
     값이 아니라 *손님 실행이 관측되지 않기* 때문이다.
  ⓑ **표적기는 살아 있다** — `formalized_target=give_discoverable_user_tool` 이 0 이 아니다.
     병목 행동(에이전트-측 인계)은 **정적 집합만으로 이미 지목된다** ⇒ 제거해도 커버리지 손실 0.

로그가 없으면 SKIP(exit 0). `T2_TEST_SKIP_LOGS=1` 로도 건너뛴다.
실행: `PYTHONIOENCODING=utf-8 py -3 test_actionreq_waitset_evidence.py`
"""
import collections
import glob
import gzip
import io
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

LOGDIR = os.path.normpath(os.path.join(HERE, "..", "..", "..",
                                       "reports", "facet_rft_2026", "sim_results"))
ENV_SURFACE = os.path.join(HERE, "a2", "env_surface.json")

# 로그 문면은 우리 층이 찍은 **고정 서식**이다(도메인 어휘 0). 이름은 전부 env/A2 에서 읽는다.
RX = re.compile(r"\[T2_ACTIONREQ\] window=open pending_user=\[([^\]]*)\] "
                r"pending_agent=\[([^\]]*)\] formalized_target=(\S+)")

_bad = 0


def chk(ok, why):
    global _bad
    print("  %s %s" % ("O" if ok else "X", why))
    if not ok:
        _bad += 1
    return ok


def discoverable_from_env():
    """discoverable 손님 도구 = **env 레지스트리에서** 읽는다(엔진 리터럴 0·[[05]])."""
    try:
        with io.open(ENV_SURFACE, encoding="utf-8") as fh:
            surf = json.load(fh)
    except Exception:
        return set(), {}
    out, per = set(), {}
    for dom, blob in (surf or {}).items():
        names = set((blob or {}).get("discoverable_user_tools") or [])
        per[dom] = names
        out |= names
    return out, per


def scan():
    files = sorted(glob.glob(os.path.join(LOGDIR, "*.log.gz")))
    tok = collections.Counter()
    tgt = collections.Counter()
    sets = collections.Counter()
    n_req = n_disc = n_disc_err = 0
    for f in files:
        try:
            with gzip.open(f, "rt", encoding="utf-8", errors="replace") as fh:
                for ln in fh:
                    if "[T2_PENDING_DISC]" in ln:
                        n_disc += 1
                        if "error (no-op)" in ln:
                            n_disc_err += 1
                    if "[T2_ACTIONREQ]" not in ln:
                        continue
                    m = RX.search(ln)
                    if not m:
                        continue
                    n_req += 1
                    names = tuple(sorted(
                        x.strip().strip("'").strip('"')
                        for x in m.group(1).split(",") if x.strip()))
                    for x in names:
                        tok[x] += 1
                    sets[names] += 1
                    tgt[m.group(3)] += 1
        except Exception as e:
            print("  ! 읽기 실패(건너뜀): %s — %r" % (os.path.basename(f), e))
    return files, n_req, n_disc, n_disc_err, tok, sets, tgt


def main():
    if os.environ.get("T2_TEST_SKIP_LOGS") == "1":
        print("SKIP — T2_TEST_SKIP_LOGS=1")
        return 0
    if not os.path.isdir(LOGDIR):
        print("SKIP — 로그 디렉터리 없음: %s" % LOGDIR)
        return 0

    t0 = time.time()
    files, n_req, n_disc, n_disc_err, tok, sets, tgt = scan()
    print("스캔: %d 파일 · %.1fs · [T2_ACTIONREQ] %d줄" % (len(files), time.time() - t0, n_req))
    if not files or not n_req:
        print("SKIP — 스캔 대상 로그 0(파일 %d · ACTIONREQ %d)" % (len(files), n_req))
        return 0

    disc, per_dom = discoverable_from_env()
    print("env discoverable 손님 도구 = %s" % (sorted(disc) or "(없음)"))
    print("pending_user 토큰 = %s" % (tok.most_common(),))
    print("distinct pending_user 집합 = %d종" % len(sets))
    for k, v in sets.most_common():
        print("    %6d  %s" % (v, list(k)))
    print("formalized_target 상위 = %s" % (tgt.most_common(8),))
    print()

    print("[1] 양성대조 ⑴ — 레버는 숙주가 살아 있는 자리에서 **한 번도 발화하지 않았다**")
    chk(n_req > 0, "숙주 `[T2_ACTIONREQ]` 가 살아 있다 (%d줄)" % n_req)
    chk(n_disc == 0, "`[T2_PENDING_DISC]` 0줄 (%d)" % n_disc)
    chk(n_disc_err == 0, "`[T2_PENDING_DISC] error (no-op)` 조차 0줄 (%d)" % n_disc_err)

    print("[2] 양성대조 ⑵ — 대기집합이 정적이라 discoverable 은 표적이 될 수 없다")
    chk(bool(disc), "env 레지스트리가 discoverable 을 실제로 선언한다 (%d개)" % len(disc))
    hit = sorted(disc & set(tok))
    chk(not hit, "`pending_user` 에 discoverable 0회 (있으면: %s)" % (hit or "-"))
    hit_t = sorted(disc & set(tgt))
    chk(not hit_t, "`formalized_target` 에 discoverable 0회 (있으면: %s)" % (hit_t or "-"))
    chk(len(sets) <= 2, "distinct pending_user 집합 <=2종 (=사실상 고정 목록) (%d)" % len(sets))

    print("[3] 양성대조 ⑶ — 손님이 실행하는 도구는 대기집합에서 **빠지지 않는다**")
    always = sorted(n for n, c in tok.items() if c == n_req)
    sometimes = sorted(n for n, c in tok.items() if 0 < c < n_req)
    print("    항상 pending = %s" % always)
    print("    가끔 빠짐    = %s" % sometimes)
    chk(len(always) >= 1,
        "전 줄(%d/%d)에 남아 있는 손님-실행 도구가 >=1개 ⇒ 종료 술어 없음" % (n_req, n_req))

    print("[4] 부정통제 ⓐ — 제거 술어 `_uacts - _effall` 자체는 살아 있다")
    chk(len(sometimes) >= 1,
        "일부 줄에서만 pending 인 토큰이 >=1개 (%s) ⇒ [3]의 100%%는 술어 고장이 아니다"
        % (sometimes or "-"))

    print("[5] 부정통제 ⓑ — 병목 표적은 **정적 집합만으로 이미 지목된다**(제거 손실 0)")
    give = [k for k in tgt if k.startswith("give_")]
    chk(bool(give), "`formalized_target` 에 give 계열이 실재 (%s)"
        % ([(k, tgt[k]) for k in give] or "-"))
    chk(sum(tgt[k] for k in give) > 0,
        "give 계열 지목 횟수 > 0 (%d)" % sum(tgt[k] for k in give))

    print("\n%s" % ("test_actionreq_waitset_evidence PASS" if not _bad
                    else "test_actionreq_waitset_evidence FAIL %d건" % _bad))
    return 1 if _bad else 0


if __name__ == "__main__":
    sys.exit(main())
