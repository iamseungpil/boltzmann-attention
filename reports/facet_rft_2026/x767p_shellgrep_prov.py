# -*- coding: utf-8 -*-
"""env 파일계통 전수 grep 의 «출력 없음» 실물 — 출처(파일·태스크·sim·메시지 index) 포함."""
import gzip, glob, json, os
for p in sorted(glob.glob("reports/facet_rft_2026/sim_results/*.results.json.gz")):
    try: raw = gzip.open(p,'rt',encoding='utf-8',errors='replace').read()
    except Exception: continue
    if "submit_referral" not in raw: continue
    try: d = json.loads(raw)
    except Exception: continue
    for si, s in enumerate(d.get("simulations",[]) or []):
        msgs = s.get("messages",[]) or []
        for i, m in enumerate(msgs):
            for tc in (m.get("tool_calls") or []):
                blob = json.dumps(tc, ensure_ascii=False)
                nm = (tc.get("name") or (tc.get("function") or {}).get("name"))
                if nm != "shell" or "submit_referral" not in blob: continue
                cmd = (tc.get("arguments") or {}).get("command") or ""
                resp = ""; ri = None
                for j in range(i+1, min(i+5, len(msgs))):
                    if msgs[j].get("role") == "tool":
                        resp = str(msgs[j].get("content") or ""); ri = j; break
                if resp.strip() not in ("(no output)", ""):
                    continue
                print("FILE   %s" % os.path.basename(p))
                print("SIM    idx=%d id=%s task=%s" % (si, str(s.get("id"))[:8], s.get("task_id")))
                print("MSG    call idx=%d -> tool idx=%s" % (i, ri))
                print("CMD    %s" % cmd)
                print("RESP   %r" % resp)
                print()
