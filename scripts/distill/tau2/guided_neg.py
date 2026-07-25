import json, requests
BASE="http://localhost:8141/v1/chat/completions"; M="Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8"
TOOLS=[
 {"type":"function","function":{"name":"unlock_discoverable_agent_tool","description":"Unlock a discoverable tool by name.",
  "parameters":{"type":"object","properties":{"agent_tool_name":{"type":"string"}},"required":["agent_tool_name"]}}},
 {"type":"function","function":{"name":"call_discoverable_agent_tool","description":"Call an unlocked discoverable tool.",
  "parameters":{"type":"object","properties":{"agent_tool_name":{"type":"string"},"arguments":{"type":"string"}},"required":["agent_tool_name","arguments"]}}},
]
USER="Please retrieve all accounts for user 224959b99e now. Use the discoverable tool get_all_user_accounts_by_user_id_3847."

def post(body):
    try:
        r=requests.post(BASE,json=body,timeout=120)
        if r.status_code!=200: return {"http":r.status_code,"err":r.text[:300]}
        m=r.json()["choices"][0]["message"]
        return {"content":(m.get("content") or "")[:200],
                "tool_names":[tc["function"]["name"] for tc in (m.get("tool_calls") or [])]}
    except Exception as e: return {"exc":str(e)[:200]}

BASEBODY={"model":M,"messages":[{"role":"user","content":USER}],"tools":TOOLS,"tool_choice":"auto",
          "temperature":0.0,"max_tokens":200}

# ===== NEGATIVE CONTROL: grammar that permits ONLY the literal word BLOCKED =====
ONLY_BLOCKED = 'root ::= "BLOCKED"\n'
print("### If grammar is ENFORCED -> content=='BLOCKED', no tool_calls. If IGNORED -> normal tool calls.\n")
for key in ("structured_outputs_grammar","toplevel_guided_grammar","extra_body_guided_grammar","response_format_grammar"):
    b=dict(BASEBODY)
    if key=="structured_outputs_grammar": b["structured_outputs"]={"grammar":ONLY_BLOCKED}
    elif key=="toplevel_guided_grammar":  b["guided_grammar"]=ONLY_BLOCKED
    elif key=="extra_body_guided_grammar":b["extra_body"]={"guided_grammar":ONLY_BLOCKED}
    else: b["response_format"]={"type":"structural_tag"}  # probe acceptance only
    print("%-30s %s"%(key, post(b)))

# ===== same negative control WITHOUT tools (does grammar work at all on this server?) =====
b=dict(BASEBODY); b.pop("tools"); b.pop("tool_choice"); b["guided_grammar"]=ONLY_BLOCKED
print("\n%-30s %s"%("no_tools+toplevel_grammar", post(b)))
b2=dict(BASEBODY); b2.pop("tools"); b2.pop("tool_choice"); b2["structured_outputs"]={"grammar":ONLY_BLOCKED}
print("%-30s %s"%("no_tools+structured_outputs", post(b2)))

# ===== guided_choice (simplest enforceable) without tools =====
b3=dict(BASEBODY); b3.pop("tools"); b3.pop("tool_choice"); b3["guided_choice"]=["ALPHA","BETA"]
print("%-30s %s"%("no_tools+guided_choice", post(b3)))
b4=dict(BASEBODY); b4["guided_choice"]=["ALPHA","BETA"]
print("%-30s %s"%("tools+guided_choice", post(b4)))

# ===== tool_choice = named function (hard constraint that IS supported) =====
b5=dict(BASEBODY); b5["tool_choice"]={"type":"function","function":{"name":"unlock_discoverable_agent_tool"}}
print("\n%-30s %s"%("tool_choice=named", post(b5)))
