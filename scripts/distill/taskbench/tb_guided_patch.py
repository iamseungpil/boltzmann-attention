#!/usr/bin/env python
"""grounded-copy v1: idempotently patch TaskBench inference.py to inject vllm 0.11
structured_outputs (per-request guided decoding) when env TB_GUIDED=1 +
TB_GUIDED_SCHEMA=<schema.json>. 미설정 시 payload 불변(no-op) — A/B 동일 파이프.

Usage: python tb_guided_patch.py /home/woori/scratch/JARVIS_tb/taskbench/inference.py
"""
import sys

MARK = "_TB_GUIDED"
HELPER = '''
# --- grounded-copy v1 (TB_GUIDED) ---
import os as _tb_os, json as _tb_json
_TB_GUIDED = {}
if _tb_os.environ.get("TB_GUIDED") and _tb_os.environ.get("TB_GUIDED_SCHEMA"):
    _TB_GUIDED = {"structured_outputs":
                  {"json": _tb_json.load(open(_tb_os.environ["TB_GUIDED_SCHEMA"]))}}
# --- end grounded-copy v1 ---
'''

ANCHOR = '"stop": None\n    })'
REPLACE = '"stop": None,\n        **_TB_GUIDED\n    })'


def main():
    path = sys.argv[1]
    src = open(path).read()
    if MARK in src:
        print("[patch] already applied")
        return
    if ANCHOR not in src:
        sys.exit("[patch] FAIL: payload anchor not found")
    src = src.replace(ANCHOR, REPLACE, 1)
    lines = src.split("\n")
    ins = max(i for i, l in enumerate(lines) if l.startswith(("import ", "from "))) + 1
    lines.insert(ins, HELPER)
    open(path, "w").write("\n".join(lines))
    print(f"[patch] applied to {path}")


if __name__ == "__main__":
    main()
