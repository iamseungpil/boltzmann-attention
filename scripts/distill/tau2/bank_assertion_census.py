# -*- coding: utf-8 -*-
"""3 sim 전수: 도구호출 시퀀스 + '리워드 결론'을 *누가* 냈는가 (에이전트 vs user-sim). [[08]]"""
import gzip, json, re, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

path = r"C:\workspace\ba-frft\reports\facet_rft_2026\sim_results\bank_t019g_20260716.results.json.gz"
with gzip.open(path, "rt", encoding="utf-8") as f:
    data = json.load(f)

# '이상 없다/맞다'류 결론 표지 (탐색용 grep일 뿐·판정은 원문 정독으로)
OK_PAT = re.compile(r"(no discrepanc|seem correct|are correct|calculated correctly|align|as expected|"
                    r"accurate|match(es)? the (expected|advertised)|everything (looks|seems))", re.I)

for si, sim in enumerate(data["simulations"]):
    msgs = sim.get("messages") or []
    print("=" * 70)
    print(f"SIM {si}  reward={(sim.get('reward_info') or {}).get('reward')}  "
          f"종료={(sim.get('termination_reason'))}  msgs={len(msgs)}")
    calls = []
    for i, m in enumerate(msgs):
        for tc in (m.get("tool_calls") or []):
            nm = tc.get("name") or (tc.get("function") or {}).get("name")
            calls.append(nm)
    print(f"  도구 시퀀스({len(calls)}): {calls}")
    print(f"  ★get_reward_discrepancies 호출: {calls.count('get_reward_discrepancies')}")
    print("  --- '리워드 OK' 표지 발화 (누가) ---")
    for i, m in enumerate(msgs):
        c = m.get("content")
        if isinstance(c, str) and OK_PAT.search(c):
            who = m.get("role")
            snip = OK_PAT.search(c)
            s = max(0, snip.start() - 90)
            print(f"    [{i}] {who:9s}: ...{c[s:snip.end()+90]}...".replace("\n", " "))
    print()
