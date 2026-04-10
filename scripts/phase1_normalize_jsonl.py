#!/usr/bin/env python3
"""Normalize concatenated JSON objects into proper JSONL.

Useful for dataset dumps that contain `}{` concatenations instead of one JSON
object per line.
"""

import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description='Normalize concatenated JSON to JSONL.')
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=None)
    parser.add_argument('--force', action='store_true')
    return parser.parse_args()


def iter_objects(text):
    decoder = json.JSONDecoder()
    idx = 0
    n = len(text)
    while idx < n:
        while idx < n and text[idx].isspace():
            idx += 1
        if idx >= n:
            break
        obj, end = decoder.raw_decode(text, idx)
        yield obj
        idx = end


def main():
    args = parse_args()
    text = args.input.read_text()
    if '\n' in text and not args.force:
        first_line = text.splitlines()[0].strip()
        try:
            json.loads(first_line)
            print(f'[phase1_normalize_jsonl] looks already line-delimited: {args.input}')
            return
        except Exception:
            pass

    objects = list(iter_objects(text))
    out_path = args.output or args.input
    with out_path.open('w') as handle:
        for obj in objects:
            handle.write(json.dumps(obj, ensure_ascii=False) + '\n')
    print(f'[phase1_normalize_jsonl] wrote {len(objects)} rows to {out_path}')


if __name__ == '__main__':
    main()
