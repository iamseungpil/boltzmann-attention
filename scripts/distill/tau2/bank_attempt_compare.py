# -*- coding: utf-8 -*-
"""bank_attempt_compare.py — 동일 태스크 floor vs full-stack의 write-시도 양상 대조 (2026-07-17).

사용자: "never-attempt 양상을 floor랑 비교. floor도 never-attempt 비슷하지 않나?"
매칭 태스크(양쪽 다 존재)마다: 미충족 gold-write를 시도(도구 호출)/무시도로 분해 + reward.
scaffold가 시도를 늘렸나(개선) 줄였나(역효과) 동일-태스크 비교.
"""
import json, gzip, re, sys, io, os, argparse
from collections import Counter
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
fam = lambda n: re.sub(r"_\d+$", "", str(n or ""))
_READ = re.compile(r"^(get|search|list|lookup|find|retrieve|read|view|check)_", re.I)
_PROC = re.compile(r"(^log_|_verification$|^kb_|^shell$|discoverable|transfer_to_human|give_|unlock_)", re.I)
isw = lambda n: bool(fam(n)) and not _READ.match(fam(n)) and not _PROC.search(fam(n))
def nd(x):
    if isinstance(x, str):
        try: x = json.loads(x)
        except Exception: return {}
    return x if isinstance(x, dict) else {}
def load(path):
    op = gzip.open if path.endswith(".gz") else open
    d = json.load(op(path, "rt", encoding="utf-8"))
    return {s.get("task_id"): s for s in d.get("simulations", [])}
def attempt_split(s):
    """미충족 gold-write: (attempted, never_attempted, reward, n_write_calls)."""
    ri = s.get("reward_info") or {}
    called = set(); ncalls = 0
    for m in (s.get("messages") or []):
        for tc in (m.get("tool_calls") or []):
            nm = tc.get("name")
            if nm == "call_discoverable_agent_tool":
                tfam = fam(nd(tc.get("arguments")).get("agent_tool_name", "")); called.add(tfam)
                if isw(tfam): ncalls += 1
            elif nm: called.add(fam(nm))
    att = nev = 0
    for ac in (ri.get("action_checks") or []):
        a = ac.get("action") or {}; outer = nd(a.get("arguments"))
        atn = outer.get("agent_tool_name", "") or a.get("name", "")
        if not isw(atn): continue
        met = ac.get("action_reward"); met = met if met is not None else (1.0 if ac.get("action_match") else 0.0)
        if float(met) >= 1.0: continue
        if fam(atn) in called: att += 1
        else: nev += 1
    return att, nev, ri.get("reward"), ncalls
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--floor", required=True); ap.add_argument("--fullstack", required=True)
    a = ap.parse_args()
    fl = load(a.floor); fs = load(a.fullstack)
    common = sorted(set(fl) & set(fs))
    print("매칭 태스크: %d (floor∩fullstack)" % len(common))
    print("%-10s | floor: att/nev rew wcalls | fullstack: att/nev rew wcalls | Δ" % "task")
    F = Counter(); S = Counter(); fpass = spass = 0
    for t in common:
        fa, fn, fr, fw = attempt_split(fl[t]); sa, sn, sr, sw = attempt_split(fs[t])
        F["att"] += fa; F["nev"] += fn; S["att"] += sa; S["nev"] += sn
        F["wcalls"] += fw; S["wcalls"] += sw
        fpass += int(fr == 1.0); spass += int(sr == 1.0)
        d = "시도↑" if sw > fw else ("시도↓" if sw < fw else "=")
        print("%-10s |  %d/%d  r=%s  wc=%d  |  %d/%d  r=%s  wc=%d  | %s" %
              (t, fa, fn, fr, fw, sa, sn, sr, sw, d))
    print("\n=== 합계 (매칭 %d) ===" % len(common))
    print("floor    : 미충족 attempted %d · never %d (%.0f%% never) · write-calls %d · pass %d" %
          (F["att"], F["nev"], 100*F["nev"]/max(F["att"]+F["nev"], 1), F["wcalls"], fpass))
    print("fullstack: 미충족 attempted %d · never %d (%.0f%% never) · write-calls %d · pass %d" %
          (S["att"], S["nev"], 100*S["nev"]/max(S["att"]+S["nev"], 1), S["wcalls"], spass))
    print("판정: write-calls floor≈fullstack면 scaffold가 시도-행동 안 바꿈(never-attempt=floor속성). fullstack<floor면 역효과.")
if __name__ == "__main__":
    main()
