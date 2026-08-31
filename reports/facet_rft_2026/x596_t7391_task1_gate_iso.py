# -*- coding: utf-8 -*-
"""격리 프로브 — t7391 retail task 1 turn18: 어느 게이트가 막는가.
재료는 전부 궤적/선언에서 읽는다(프롬프트 저작 0). [[78]]"""
import sys, io, json, gzip, os
sys.path.insert(0, r"C:\workspace\ba-frft\scripts\distill\tau2")
os.chdir(r"C:\workspace\ba-frft\scripts\distill\tau2")
from gate_interpreter import GateInterpreter

RES = r"C:\workspace\ba-frft\reports\facet_rft_2026\sim_results\t7391_reg12.results.json.gz"
REF = r"C:\workspace\ba-frft\reports\facet_rft_2026\sim_results\hist_gpt52_reg12_PASS.results.json.gz"
d = json.load(gzip.open(RES, "rt", encoding="utf-8"))
sim = [x for x in d["simulations"] if x["task_id"] == "1"][0]
ms = sim["messages"]
u1, u17 = ms[1]["content"], ms[17]["content"]
gold = sim["reward_info"]["action_checks"][-1]["action"]
tool, args = gold["name"], gold["arguments"]

# 후보 레코드 = 참조 런의 DISAMBIGUATION NOTE 축자(같은 유저·같은 DB)
ref = json.load(gzip.open(REF, "rt", encoding="utf-8"))
note = [x for x in ref["simulations"] if x["task_id"] == "1"][0]["messages"][19]["content"]
recs = {}
for ln in note.splitlines():
    if ln.startswith("- #W"):
        cid, body = ln[2:].split(": ", 1)
        recs[cid] = json.loads(body)
recs["#W2378156"]["status"] = "delivered"      # t7391 에서는 아직 write 전
ORDERS = list(recs.keys())

def resolve_field(path, a):
    if path[-1] == "orders":
        return ORDERS
    cid = a.get("order_id")
    return (recs.get(cid) or {}).get(path[-1]) if cid else None

def resolve_owner(path, a):
    return "yusuf_rossi_9620" if a.get("order_id") in recs else None

def fetch_record(producer, id_arg, cid):
    return recs.get(cid)

a2 = json.load(io.open("a2/retail.gate.json", encoding="utf-8"))
for label, kinds in [("실행된 설정(T2_GATE_KINDS 미설정)", None),
                     ("정본 설정(SCAFFOLD_AUDIT_RULE0 §92)",
                      "auth,confirm,ownership,notice,preconditions,constraints")]:
    gl = a2["gates"] if kinds is None else [g for g in a2["gates"] if g["kind"] in set(kinds.split(","))]
    gi = GateInterpreter(gl, resolvers={"resolve_field": resolve_field,
                                        "resolve_owner": resolve_owner,
                                        "fetch_record": fetch_record})
    gi.state.auth_user = "yusuf_rossi_9620"
    print("=" * 78)
    print(label, "· 게이트 %d개" % len(gl))
    for turn, lu in [("turn16(확인 전)", u1), ("turn18(확인 후)", u17), ("turn20(가상 재시도)", u17)]:
        ok, gid, why = gi.check(tool, args, last_user_msg=lu, transfer_msg_sent=None)
        print("  %-20s allowed=%-5s gate=%s" % (turn, ok, gid))
        if why:
            print("     사유 %d자 | %s" % (len(why), why[:150].replace("\n", " ")))
