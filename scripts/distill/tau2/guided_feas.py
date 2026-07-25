import json, requests
BASE="http://localhost:8141/v1/chat/completions"; M="Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8"
# Minimal realistic setup: clean schema (dispatcher only) + KB text that names an out-of-schema tool
TOOLS=[
 {"type":"function","function":{"name":"unlock_discoverable_agent_tool","description":"Unlock a discoverable tool by name.",
  "parameters":{"type":"object","properties":{"agent_tool_name":{"type":"string"}},"required":["agent_tool_name"]}}},
 {"type":"function","function":{"name":"call_discoverable_agent_tool","description":"Call a previously unlocked discoverable tool.",
  "parameters":{"type":"object","properties":{"agent_tool_name":{"type":"string"},"arguments":{"type":"string"}},"required":["agent_tool_name","arguments"]}}},
 {"type":"function","function":{"name":"get_current_time","description":"Get current time.","parameters":{"type":"object","properties":{}}}},
]
ALLOWED=[t["function"]["name"] for t in TOOLS]
KB=("Knowledge base excerpt:\n"
    "To list every account a customer holds, use get_all_user_accounts_by_user_id_3847 with the user_id.\n"
    "Note: discoverable tools must first be unlocked, then invoked through the dispatcher.\n")
USER=KB+"\nPlease retrieve all accounts for user 224959b99e now."
CHAT="Hi, just saying hello. Please reply in one short sentence, no tool calls."

def post(body,label):
    try:
        r=requests.post(BASE,json=body,timeout=120)
        if r.status_code!=200:
            return ("HTTP%d"%r.status_code, r.text[:400])
        m=r.json()["choices"][0]["message"]
        tcs=m.get("tool_calls") or []
        names=[tc["function"]["name"] for tc in tcs]
        inner=[]
        for tc in tcs:
            try:
                a=json.loads(tc["function"]["arguments"]);
                if "agent_tool_name" in a: inner.append(a["agent_tool_name"])
            except Exception: pass
        return ("OK", {"content":(m.get("content") or "")[:120],"tool_names":names,"inner":inner})
    except Exception as e:
        return ("EXC", str(e)[:300])

print("=== allowed schema names:",ALLOWED)

# T1 baseline: auto, no constraint
b1={"model":M,"messages":[{"role":"user","content":USER}],"tools":TOOLS,"tool_choice":"auto","temperature":0.7,"max_tokens":300}
for i in range(4):
    print("T1_auto[%d]"%i, post(b1,"t1"))

# T2 tool_choice=required (forces a call every turn -> test conversation breakage)
b2=dict(b1); b2["tool_choice"]="required"
print("T2_required_toolturn", post(b2,"t2"))
b2c=dict(b2); b2c["messages"]=[{"role":"user","content":CHAT}]
print("T2_required_chatturn(should break)", post(b2c,"t2c"))

# T3 auto + structured_outputs grammar (vLLM 0.11 style) -- does server accept?
GRAMMAR = r'''
root ::= text | call
text ::= [^<]+
call ::= "<tool_call>" ws "{" ws "\"name\"" ws ":" ws name ws "," ws "\"arguments\"" ws ":" ws obj ws "}" ws "</tool_call>"
name ::= "\"unlock_discoverable_agent_tool\"" | "\"call_discoverable_agent_tool\"" | "\"get_current_time\""
obj ::= "{" [^}]* "}"
ws ::= [ \n\t]*
'''
for key in ("structured_outputs","guided_grammar"):
    b3=dict(b1)
    if key=="structured_outputs": b3["structured_outputs"]={"grammar":GRAMMAR}
    else: b3["extra_body"]={"guided_grammar":GRAMMAR}
    print("T3_%s"%key, post(b3,"t3")[0:1], str(post(b3,"t3"))[:300])

# T4 top-level guided_grammar (older vLLM API surface)
b4=dict(b1); b4["guided_grammar"]=GRAMMAR
print("T4_toplevel_guided_grammar", str(post(b4,"t4"))[:400])

# T5 guided_regex on name field via response_format? test json_schema for tool args (not names)
b5=dict(b1); b5["guided_regex"]=r".*"
print("T5_guided_regex_accepted?", str(post(b5,"t5"))[:200])
