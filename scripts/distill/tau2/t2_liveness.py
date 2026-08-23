# -*- coding: utf-8 -*-
"""레버 **생존 감사** 정본 — 켜졌나가 아니라 **전달했나**를 묻는다.

사용자 지시(2026-08-14 야간): *"지금까지 계속 이미 있던 원칙들을 안 쓰고 원점에서 다시 하고 있다.
원칙을 라이브러리화하고 검색도 하나로 고정하고, 다른 레버도 라이브러리로 고정해서 다시 하지
않게 하라."*

**왜 이 모듈이 필요한가.** `t2_levers` 는 레버가 *무엇인지* 안다. 런처는 레버를 *켠다*. 그런데
둘 다 **그 레버가 실제로 무언가를 전달했는지는 모른다**. 오늘 그 구멍에 하루를 태웠다:

  · `T2_SEARCH_AGENT` 는 t7290 072 에서 **10회 발화하고 10회 전부 침묵**했다
    (`now 미확정 · 원값 None`). 온톨로지 문서 색인은 695/698 을 덮고 있었고 필요한 문서
    (`bank_accounts_(general)_017`)도 색인 안에 있었다 — **레버는 있었고 배선이 죽어 있었다**.
  · 그걸 모른 채 "072 는 retrieval 문제인가 학습 문제인가"를 격리 프로브로 몇 시간 팠다.
    측정한 결손의 상당 부분이 **우리 죽은 레버가 만든 것**이었다.
  · 같은 부류의 선례가 이미 메모리에 있다([[55]]): `proc_fb` 死배선이 deny 11회를 인쇄로 만들었다.

⇒ 규칙: **결손을 모델에게 귀속하기 전에 이 감사를 돌린다.** [[55]](우리 배관 먼저)의 실행 형태다.

판정은 문면이 아니라 **우리 자신의 로그 프로토콜**로 한다(도메인 판단 0·[[59]]):
    DELIVERED  그 태그의 줄이 있고 침묵-표지가 없다 (또는 **설계된 침묵**=`BENIGN`)
    SILENCED   침묵·건너뜀·무발화·미확정·실패 표지가 붙어 있다  ← **여기가 위험 구간**
    ABSENT     그 태그의 줄이 아예 없다(켜졌는데 도달조차 못 함 or 안 켬)

사용:
    py t2_liveness.py <log> [<log>...]          # 런 로그 감사
    from t2_liveness import audit               # 라이브러리로
"""
import collections
import gzip
import glob
import io
import json
import os
import re
import sys

# 우리 로그가 침묵을 알릴 때 쓰는 말들(엔진이 스스로 찍는 문면·도메인 어휘 0)
SILENT_MARKS = ("침묵", "건너뜀", "무발화", "미발화", "미확정", "실패", "폐기", "못 찾",
                "no-op", "skip")
FBDIR = "/home/woori/scratch/logs"   # 사이드카 위치(리모트 전용·없으면 도달 축 생략)
TAG = re.compile(r"\[(T2_[A-Z0-9_]+)\]")

# ★설계된 침묵 (2026-08-14·이 감사의 첫 오탐 교정).
#   침묵에는 두 종류가 있다: **못 해서** 조용한 것과 **할 필요가 없어서** 조용한 것.
#   초판은 둘을 못 갈라 `T2_ACTION_HISTORY` 를 DEAD 로 찍었는데, 그 침묵은 U1
#   (`_dispatch_since_last_user`·완료된 write 되감기 방지)이 **제 일을 한** 표시였다.
#   ⇒ 사유 문면으로 선언한다. 여기 없는 침묵은 전부 **확인 대상**으로 남긴다(안전측:
#   모르면 위험으로 센다). 새 항목을 넣을 때는 **그 침묵이 옳다는 근거**를 함께 적어라.
BENIGN = {
    # U1 출시분(2026-08-13 야간·write 중복 36건 중 15건이 우리 문구 직후였다):
    #   디스패치가 이미 성공했으면 재-발견을 요구하지 않는 것이 정상 동작이다.
    "T2_ACTION_HISTORY": ("디스패치 성공",),
}


def _benign(tag, body):
    return any(w in body for w in BENIGN.get(tag, ()))
SIM = re.compile(r"\[sim=(task_\d+)")


def audit(paths):
    """로그들 → {태그: {"delivered": n, "silenced": n, "reasons": Counter, "sims": set}}"""
    out = collections.defaultdict(
        lambda: {"delivered": 0, "silenced": 0,
                 "by_design": 0, "reasons": collections.Counter(), "sims": set()})
    for p in paths:
        for ln in io.open(p, encoding="utf-8", errors="replace"):
            m = TAG.search(ln)
            if not m:
                continue
            tag = m.group(1)
            rec = out[tag]
            s = SIM.search(ln)
            if s:
                rec["sims"].add(s.group(1))
            body = ln[m.end():]
            hit = next((w for w in SILENT_MARKS if w in body), None)
            if hit and _benign(tag, body):
                rec["by_design"] = rec.get("by_design", 0) + 1
                rec["delivered"] += 1        # 제 일을 한 침묵은 생존으로 센다
            elif hit:
                rec["silenced"] += 1
                # 사유는 그 줄에서 **엔진이 적은 문구**를 그대로 짧게 남긴다(해석 0)
                rec["reasons"][" ".join(body.split())[:70]] += 1
            else:
                rec["delivered"] += 1
    return out


# ─────────────────────────────────────────────────────────────────────────────
# ★중재 축 (2026-08-23 · 사용자 물음 *"이걸 측정할 방법이 없나"*).
#
# `audit()` 는 **한 레버가 전달했나**를 본다. 그런데 이 스택에서 가장 자주 발화하는 기제는
# 레버가 아니라 **중재자**다 — t7346 40 sim 에서 `[T2_MATERIAL_GATE] stop=` 만 **439회**
# (sim당 11회)다. 그 439회 중 몇 개가 결정점 앞이었고 몇 개가 옳았는지 **한 번도 안 쟀다**.
# 등대 §1.3: *"합성은 무한후퇴가 아니라 **측정된 상쇄**여야 한다"* — 중재자도 계측 대상이다.
#
# 여기서 재는 것(전부 우리 자신의 로그 프로토콜·도메인 판단 0):
#   ⑴ **정지율** — sim 당·로그 100줄당 정지 수. **pass ↔ fail 로 갈라 본다**([[57]] 부정통제의
#      관찰 형태: pass sim 에서도 같은 비율이면 그 정지는 원인이 아니다).
#   ⑵ **채널이 열린 적 있나** — 그 sim 에서 결정 재료가 한 번이라도 배달됐나
#      (`T2_DECISION_CARRY` · `T2_SEARCH_AGENT 축 처리 완료`). 정지 N회·배달 0회면 그 관문은
#      그 sim 에서 **채널을 영구히 닫은 것**이다(C494/C495 가 055 에서 본 그 모양).
#   ⑶ **억제가 지연인가 손실인가** — `[T2_STACK] window suppressed tag=X` 뒤에 같은 sim 에서
#      X 가 다시 발화했나. 다시 났으면 지연, 아니면 손실.
#   ⑷ **덮어쓴 바이트** — `[T2_CP2_CLOBBER]` 가 버린 배달물 크기 합.
#
# ⚠이 함수가 **답하지 못하는 것**: 정지가 **옳았는지**. 관찰 자료는 상관까지만 준다.
#   인과는 라이브 A/B(중재 파라미터 조정) 또는 환경-롤아웃 하네스만 준다 — 결정점 **재생**으로는
#   못 한다(C596: 비커밋 채널은 재생으로 재현되지 않는다. x487 이 그것으로 무효가 됐다).
SIMFULL = re.compile(r"\[sim=(task_\d+#s\d+)\]")
MG = re.compile(r"\[T2_MATERIAL_GATE\] stop=([a-z_]+)(\([^)]*\))?\s+turn=(\d+)")
WS = re.compile(r"\[T2_STACK\] window suppressed tag=(\S+)")
CL = re.compile(r"\[T2_CP2_CLOBBER\][^0-9]*?(\d+)자를 버리고")
DELIVERED_MARKS = ("[T2_DECISION_CARRY]", "[T2_SEARCH_AGENT] 축 처리 완료")


def arbitration(log_paths, results=None):
    """로그(+선택적으로 {sim_tag: reward}) → sim 별 중재 원장.

    sim_tag = `task_016#s626729` (results 의 `task_id` + `seed` 로 조립되는 그 키).
    """
    per = collections.defaultdict(
        lambda: {"stops": collections.Counter(), "stop_turns": [],
                 "winners": collections.Counter(), "suppressed": [],
                 "clobbered_bytes": 0, "deliveries": 0, "lines": 0,
                 "fires": collections.defaultdict(list), "reward": None})
    for p in log_paths:
        opener = gzip.open if str(p).endswith(".gz") else io.open
        with opener(p, "rt", encoding="utf-8", errors="replace") as f:
            for ln in f:
                sm = SIMFULL.search(ln)
                if not sm:
                    continue
                rec = per[sm.group(1)]
                rec["lines"] += 1
                m = MG.search(ln)
                if m:
                    rec["stops"][m.group(1)] += 1
                    rec["stop_turns"].append((m.group(1), int(m.group(3))))
                    if m.group(1) == "other_lever" and m.group(2):
                        for w in m.group(2).strip("()").split(","):
                            rec["winners"][w.strip()] += 1
                w = WS.search(ln)
                if w:
                    rec["suppressed"].append([w.group(1), None, rec["lines"]])
                c = CL.search(ln)
                if c:
                    rec["clobbered_bytes"] += int(c.group(1))
                if any(k in ln for k in DELIVERED_MARKS):
                    rec["deliveries"] += 1
                t = TAG.search(ln)
                if t:
                    rec["fires"][t.group(1)].append(rec["lines"])
    # ⑶ 억제가 지연인가 손실인가. 태그 이름 규약이 달라(`tag=claimprov` ↔ `[T2_CLAIMPROV]`)
    #    대문자로 맞춘다. 맞는 태그가 아예 없으면 판정을 `None` 으로 남긴다(추측 금지).
    for rec in per.values():
        rows = []
        for name, _, at in rec["suppressed"]:
            want = "T2_" + name.upper()
            fires = rec["fires"].get(want)
            rows.append((name, (any(n > at for n in fires) if fires else None)))
        rec["suppressed"] = rows
        rec.pop("fires", None)
    if results:
        for k, v in results.items():
            if k in per:
                per[k]["reward"] = v
    return dict(per)


def rewards_from_results(paths):
    """results.json(.gz) → {`task_id#s<seed>`: reward}. 조인 키 정본."""
    out = {}
    for p in paths:
        opener = gzip.open if str(p).endswith(".gz") else io.open
        with opener(p, "rt", encoding="utf-8", errors="replace") as f:
            d = json.load(f)
        for s in (d.get("simulations") or []):
            key = "%s#s%s" % (s.get("task_id"), s.get("seed"))
            out[key] = (s.get("reward_info") or {}).get("reward")
    return out


def report_arbitration(per):
    """pass ↔ fail 로 갈라 정지율을 인쇄한다([[57]] 부정통제의 관찰 형태)."""
    known = [(k, v) for k, v in per.items() if v.get("reward") is not None]
    groups = {"PASS": [v for _, v in known if v["reward"]],
              "FAIL": [v for _, v in known if not v["reward"]]}
    print("%-5s %4s %7s %11s %12s %8s %12s" %
          ("군", "sim", "정지", "정지/100줄", "resolve_cap", "배달", "배달0 sim"))
    print("-" * 68)
    for g in ("PASS", "FAIL"):
        rows = groups.get(g) or []
        if not rows:
            continue
        stops = sum(sum(r["stops"].values()) for r in rows)
        lines = sum(r["lines"] for r in rows) or 1
        cap = sum(r["stops"].get("resolve_cap", 0) for r in rows)
        dl = sum(r["deliveries"] for r in rows)
        zero = sum(1 for r in rows if r["deliveries"] == 0)
        print("%-5s %4d %7d %11.2f %12d %8d %12d" %
              (g, len(rows), stops, 100.0 * stops / lines, cap, dl, zero))
    sup = [s for v in per.values() for s in v["suppressed"]]
    if sup:
        again = sum(1 for _, r in sup if r is True)
        lost = sum(1 for _, r in sup if r is False)
        unk = sum(1 for _, r in sup if r is None)
        print("")
        print("억제 %d건 — 같은 sim 에서 재발화 %d(=지연) · 재발화 없음 %d(=손실) · 판정불가 %d"
              % (len(sup), again, lost, unk))
    print("CP2 가 버린 배달물 합계 %d자" % sum(v["clobbered_bytes"] for v in per.values()))
    print("")
    print("※ 이 표는 **상관**까지만 준다. 정지가 옳았는지는 라이브 A/B(중재 파라미터) 또는")
    print("  환경-롤아웃만 답한다 — 결정점 재생은 못 한다(C596).")
    return groups


def report(res, min_silent_ratio=0.5):
    rows = sorted(res.items(), key=lambda kv: -(kv[1]["silenced"]))
    print("%-28s %8s %8s %6s  %s" % ("lever", "delivered", "silenced", "sims", "판정"))
    print("-" * 104)
    dead = []
    for tag, r in rows:
        tot = r["delivered"] + r["silenced"]
        ratio = (r["silenced"] / tot) if tot else 0.0
        verdict = ("⚠DEAD" if r["delivered"] == 0 and r["silenced"] else
                   ("⚠주로 침묵" if ratio >= min_silent_ratio else "ok"))
        if verdict != "ok":
            dead.append((tag, r, ratio))
        print("%-28s %8d %8d %6d  %s" % (tag, r["delivered"], r["silenced"],
                                         len(r["sims"]), verdict))
    if dead:
        print("\n# 침묵 사유 (전달 0 이거나 침묵이 과반인 레버)")
        for tag, r, ratio in dead:
            print("  [%s] 침묵 %d/%d" % (tag, r["silenced"], r["delivered"] + r["silenced"]))
            for reason, n in r["reasons"].most_common(3):
                print("      %3d× %s" % (n, reason))
    print("\n※ DELIVERED 는 '전달했다'이지 '옳았다'가 아니다. 이 표는 **배선 생존**만 본다 — "
          "효과 판정은 격리 프로브와 라이브 대조가 한다([[57]]).")
    return dead


def delivery(tags):
    """**도달 축** — 사이드카(`fb_<tag>.jsonl`)의 `arrived` 로 *모델 입력에 들어갔는가*를 본다.

    ★왜 따로 필요한가 (2026-08-14 하루에 **두 번** 오판했다):
      ⑴ 아침: 도구 반환문을 **stderr 로그**에서 grep 해 "미발화"로 읽었다 — 실제 7회 발화(C475).
      ⑵ 저녁: ACTION_INDEX 43줄이 **궤적(results.json)에 없다**고 "한 글자도 안 갔다"로 읽었다 —
         사이드카는 `decision_carry arrived 11/11` 이었다. 우리 채널 일부는 **재생성 버퍼**로
         주입되고 메시지로 영속되지 않는다(C443 설계). **궤적 부재는 도달 부재가 아니다.**
    ⇒ 세 축을 **각자 사는 자리에서** 읽는다: 로그=발화 · 사이드카=도달 · 궤적=손님-가시.

    반환: {채널: {"fired": n, "arrived": n}} · 사이드카가 없으면 빈 dict(로컬)."""
    out = collections.defaultdict(lambda: {"fired": 0, "arrived": 0})
    for tag in tags:
        p = os.path.join(FBDIR, "fb_%s.jsonl" % tag)
        if not os.path.exists(p):
            continue
        for ln in io.open(p, encoding="utf-8", errors="replace"):
            try:
                o = json.loads(ln)
            except Exception:
                continue
            k = str(o.get("agent") or o.get("mark") or "?")
            out[k]["fired"] += 1
            if o.get("arrived") is True:
                out[k]["arrived"] += 1
    return dict(out)


def report_delivery(dl):
    if not dl:
        print("\n(사이드카 없음 — 도달 축 생략. 리모트에서 돌리면 채워진다)")
        return []
    print("\n%-24s %7s %8s  %s" % ("채널", "발화", "arrived", "판정"))
    print("-" * 72)
    bad = []
    for k, v in sorted(dl.items(), key=lambda kv: -kv[1]["fired"]):
        if k == "?":
            continue
        r = v["arrived"] / float(v["fired"]) if v["fired"] else 0.0
        verdict = "⚠도달 0" if v["arrived"] == 0 else ("⚠일부만" if r < 0.8 else "ok")
        if verdict != "ok":
            bad.append(k)
        print("%-24s %7d %8d  %s" % (k, v["fired"], v["arrived"], verdict))
    print("※ 발화 ≠ 도달. 로그에 찍혔다고 모델이 봤다는 뜻이 아니다 — 이 표가 그 사각이다.")
    return bad


def main(argv):
    paths = []
    for a in argv:
        paths += sorted(glob.glob(a)) if any(c in a for c in "*?[") else [a]
    paths = [p for p in paths if os.path.exists(p)]
    if not paths:
        print(__doc__)
        return 1
    print("감사 대상 %d 파일\n" % len(paths))
    dead = report(audit(paths))
    tags = [os.path.basename(p).replace(".log", "") for p in paths]
    baddel = report_delivery(delivery(tags))
    print("\n결과: 레버 %d종 · 침묵 위험 %d종 · 도달 위험 %d종"
          % (len(audit(paths)), len(dead), len(baddel)))
    return 0


if __name__ == "__main__":
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    except Exception:
        pass
    sys.exit(main(sys.argv[1:]))
