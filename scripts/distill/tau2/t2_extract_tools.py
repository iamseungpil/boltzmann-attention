#!/usr/bin/env python
"""tau2 A1 adapter: extract tool catalog from a tau2 domain (BENCH_PORTFOLIO §3.5 ①).

Outputs (per domain):
  <out>/<domain>_tools_openai.json  - list of OpenAI function schemas (agent `tools` param)
  <out>/<domain>_tool_catalog.json  - {name: {type: READ|WRITE|GENERIC, required: [...], params: [...]}}
                                      + "enum" (all names) — gate/guided 입력.

Run on remote (CPU): cd /home/woori/scratch/tau2-bench && PYTHONPATH=src \
  /home/woori/venvs/seka_env/bin/python $REPO/scripts/distill/tau2/t2_extract_tools.py \
  --domain retail --out /home/woori/scratch/tau2_adapter
"""
import argparse, importlib, json, os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="retail")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    mod = importlib.import_module(f"tau2.domains.{a.domain}.environment")
    env = mod.get_environment()
    tools = env.tools.get_tools()

    openai_schemas, catalog = [], {}
    for name, tool in tools.items():
        sch = tool.openai_schema
        openai_schemas.append(sch)
        fn = sch.get("function", sch)
        params = fn.get("parameters", {})
        catalog[name] = {
            "type": str(env.tools.tool_type(name)).split(".")[-1],
            "required": params.get("required", []),
            "params": sorted(params.get("properties", {}).keys()),
        }

    os.makedirs(a.out, exist_ok=True)
    json.dump(openai_schemas, open(f"{a.out}/{a.domain}_tools_openai.json", "w"), indent=1)
    json.dump({"enum": sorted(catalog), "tools": catalog},
              open(f"{a.out}/{a.domain}_tool_catalog.json", "w"), indent=1)
    by_type = {}
    for name, c in catalog.items():
        by_type.setdefault(c["type"], []).append(name)
    print(f"[t2_extract] {a.domain}: {len(catalog)} tools -> {a.out}")
    for t, names in sorted(by_type.items()):
        print(f"  {t}: {len(names)} {sorted(names)}")


if __name__ == "__main__":
    main()
