#!/usr/bin/env python3
"""Audit whether the SEKA / ontology robustness stack is runnable.

This is a lightweight readiness check for local machines or E8. It does not
load models or run benchmarks. It only verifies that the required repository
surface, datasets, Python modules, and optional AdaSEKA hooks are present.
"""

import argparse
import importlib.util
import json
from pathlib import Path


REQUIRED_MODULES = [
    'torch',
    'transformers',
    'datasets',
    'spacy',
    'nltk',
    'dataclasses_json',
]

OPTIONAL_MODULES = [
    'huggingface_hub',
]

ADASEKA_PATTERNS = [
    'adaseka',
    'router',
    'mixture',
    'expert',
]


def parse_args():
    parser = argparse.ArgumentParser(
        description='Audit SEKA / ontology experiment readiness.'
    )
    parser.add_argument(
        '--repo-root',
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument('--json-out', type=Path, default=None)
    parser.add_argument(
        '--strict',
        action='store_true',
        help='Exit non-zero when required items are missing.',
    )
    parser.add_argument(
        '--require-adaseka',
        dest='require_adaseka',
        action='store_true',
        help='Fail if no AdaSEKA-like code surface is found.',
    )
    return parser.parse_args()


def has_module(name):
    return importlib.util.find_spec(name) is not None


def gather_dir_state(path):
    return {
        'exists': path.exists(),
        'is_dir': path.is_dir(),
    }


def main():
    args = parse_args()
    repo_root = args.repo_root
    scripts_dir = repo_root / 'scripts'
    seka_root = repo_root / 'external' / 'SEKA'
    benchmarks_dir = seka_root / 'benchmarks'
    src_dir = seka_root / 'src'
    data_dir = seka_root / 'data'
    counterfact_path = data_dir / 'pasta_bench' / 'counterfact.jsonl'
    biosbias_path = data_dir / 'pasta_bench' / 'biosbias.jsonl'
    hf_cache = Path.home() / '.cache' / 'huggingface'

    required_modules = {name: has_module(name) for name in REQUIRED_MODULES}
    optional_modules = {name: has_module(name) for name in OPTIONAL_MODULES}

    adaseka_hits = []
    search_roots = [scripts_dir]
    if seka_root.exists():
        search_roots.extend([benchmarks_dir, src_dir])
    for root in search_roots:
        if not root.exists():
            continue
        for path in root.rglob('*'):
            if not path.is_file():
                continue
            if path.resolve() == Path(__file__).resolve():
                continue
            if path.suffix.lower() not in {'.py', '.md', '.txt', '.json'}:
                continue
            try:
                text = path.read_text(errors='ignore')
            except OSError:
                continue
            lowered = text.lower()
            for pattern in ADASEKA_PATTERNS:
                if pattern in lowered:
                    adaseka_hits.append(str(path.relative_to(repo_root)))
                    break

    result = {
        'repo_root': str(repo_root),
        'paths': {
            'repo_root': gather_dir_state(repo_root),
            'scripts': gather_dir_state(scripts_dir),
            'external_seka': gather_dir_state(seka_root),
            'external_seka_benchmarks': gather_dir_state(benchmarks_dir),
            'external_seka_src': gather_dir_state(src_dir),
            'external_seka_data': gather_dir_state(data_dir),
            'counterfact_jsonl': {
                'exists': counterfact_path.exists(),
                'path': str(counterfact_path),
            },
            'biosbias_jsonl': {
                'exists': biosbias_path.exists(),
                'path': str(biosbias_path),
            },
            'huggingface_cache': gather_dir_state(hf_cache),
        },
        'modules': {
            'required': required_modules,
            'optional': optional_modules,
        },
        'adaseka': {
            'found_any_surface': bool(adaseka_hits),
            'hits': adaseka_hits[:20],
        },
    }

    missing_required = [
        f'module:{name}' for name, ok in required_modules.items() if not ok
    ]
    if not seka_root.exists():
        missing_required.append('path:external/SEKA')
    if seka_root.exists() and not benchmarks_dir.exists():
        missing_required.append('path:external/SEKA/benchmarks')
    if seka_root.exists() and not src_dir.exists():
        missing_required.append('path:external/SEKA/src')
    if seka_root.exists() and not data_dir.exists():
        missing_required.append('path:external/SEKA/data')
    if not counterfact_path.exists():
        missing_required.append('data:counterfact')
    if args.require_adaseka and not adaseka_hits:
        missing_required.append('surface:AdaSEKA')

    result['missing_required'] = missing_required
    result['ready_for_seka_smoke'] = len(missing_required) == 0

    text = json.dumps(result, indent=2)
    print(text)
    if args.json_out:
        args.json_out.write_text(text + '\n')

    if args.strict and missing_required:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
