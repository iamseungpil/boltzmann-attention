# -*- coding: utf-8 -*-
"""에이전트가 KB 파일계통에 대고 `submit_referral` 을 직접 grep 한 적이 있나 —
있으면 그 다음 tool 응답이 «코퍼스 전수»에 대한 부재 증거가 된다."""
import gzip, glob, json
cmds = {}
for p in sorted(glob.glob("reports/facet_rft_2026/sim_results/*.results.json.gz")):
    try: raw = gzip.open(p,'rt',encoding='utf-8',errors='replace').read()
    except Exception: continue
    if "submit_referral" not in raw: continue
    try: d = json.loads(raw)
    except Exception: continue
    for s in d.get("simulations",[]) or []:
        msgs = s.get("messages",[]) or []
        for i, m in enumerate(msgs):
            for tc in (m.get("tool_calls") or []):
                blob = json.dumps(tc, ensure_ascii=False)
                if "submit_referral" not in blob: continue
                nm = (tc.get("name") or (tc.get("function") or {}).get("name"))
                if nm not in ("shell", "KB_search", "KB_search_bm25", "KB_search_dense"): continue
                # 다음 tool 응답
                resp = ""
                for j in range(i+1, min(i+4, len(msgs))):
                    if msgs[j].get("role") == "tool":
                        resp = str(msgs[j].get("content") or ""); break
                key = (nm, blob[:300])
                cmds.setdefault(key, [0, resp])
                cmds[key][0] += 1
for (nm, blob), (n, resp) in sorted(cmds.items(), key=lambda kv:-kv[1][0]):
    print("### (%d) %s  %s" % (n, nm, blob[:260]))
    print("    -> %s" % resp[:400].replace("\n", " | "))
    print()
