#!/usr/bin/env python3
"""Patch JARVIS taskbench/inference.py in place (idempotent, .bak backup):
  1. `results = []` init after get_event_loop — upstream crash when log_first_detail
     is False (`results +=` on undefined name). COWORKER_REQUEST_TB_SCALE §4-3.
  2. `"chat_template_kwargs": {"enable_thinking": false}` in the request payload —
     pins Qwen3 hybrid-thinking models to non-thinking (v2 공통 통제). Qwen2.5 chat
     templates ignore the kwarg, so it is safe to apply unconditionally; this is the
     same method Track A applied (inference.py payload patch), keeping 두 트랙 동일.
Usage: tb_patch_inference.py <taskbench_dir>
"""
import sys
from pathlib import Path

p = Path(sys.argv[1]) / "inference.py"
src = p.read_text()
orig = src

if "results = []  # tbpatch" not in src:
    needle = "    loop = asyncio.get_event_loop()\n"
    assert needle in src, "results-init anchor not found"
    src = src.replace(
        needle,
        needle + "    results = []  # tbpatch: fix uninit crash when log_first_detail=False\n",
        1)

if "chat_template_kwargs" not in src:
    needle = '"stream": False,'
    assert needle in src, "payload anchor not found"
    src = src.replace(
        needle,
        '"chat_template_kwargs": {"enable_thinking": False},  # tbpatch: Qwen3 non-thinking pin\n'
        '        ' + needle,
        1)

if src != orig:
    bak = p.with_suffix(".py.bak")
    bak.exists() or bak.write_text(orig)
    p.write_text(src)
    print(f"[tbpatch] patched {p}")
else:
    print(f"[tbpatch] already patched {p}")
