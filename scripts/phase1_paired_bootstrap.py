#!/usr/bin/env python3
"""Compute paired win/loss counts and bootstrap confidence intervals.

Input files can be JSONL or CSV. Each row must contain a shared example id and
metric columns such as `efficacy` or `paraphrase`.
"""

import argparse
import csv
import json
import random
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description='Paired bootstrap comparison for per-example evaluation exports.'
    )
    parser.add_argument('--a', type=Path, required=True, help='Baseline/export A')
    parser.add_argument('--b', type=Path, required=True, help='Baseline/export B')
    parser.add_argument('--label-a', default='a')
    parser.add_argument('--label-b', default='b')
    parser.add_argument('--id-field', default='example_id')
    parser.add_argument(
        '--metrics',
        default='efficacy,paraphrase',
        help='Comma-separated metric fields.',
    )
    parser.add_argument('--bootstrap-iters', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--json-out', type=Path, default=None)
    return parser.parse_args()


def load_rows(path):
    suffix = path.suffix.lower()
    if suffix == '.jsonl':
        with path.open() as handle:
            return [json.loads(line) for line in handle if line.strip()]
    if suffix == '.csv':
        with path.open(newline='') as handle:
            return list(csv.DictReader(handle))
    raise ValueError(f'Unsupported file type: {path}')


def to_float(value):
    if value is None or value == '':
        return None
    return float(value)


def percentile(sorted_values, q):
    if not sorted_values:
        return None
    idx = min(max(int(q * (len(sorted_values) - 1)), 0), len(sorted_values) - 1)
    return sorted_values[idx]


def main():
    args = parse_args()
    metrics = [m.strip() for m in args.metrics.split(',') if m.strip()]

    rows_a = {str(row[args.id_field]): row for row in load_rows(args.a)}
    rows_b = {str(row[args.id_field]): row for row in load_rows(args.b)}
    shared_ids = sorted(set(rows_a) & set(rows_b))
    if not shared_ids:
        raise SystemExit('No shared example ids found')

    per_metric = {}
    rng = random.Random(args.seed)

    for metric in metrics:
        paired = []
        wins_a = 0
        wins_b = 0
        ties = 0
        for example_id in shared_ids:
            va = to_float(rows_a[example_id].get(metric))
            vb = to_float(rows_b[example_id].get(metric))
            if va is None or vb is None:
                continue
            paired.append((example_id, va, vb))
            if va > vb:
                wins_a += 1
            elif vb > va:
                wins_b += 1
            else:
                ties += 1

        if not paired:
            per_metric[metric] = {
                'n': 0,
                'error': 'no shared numeric rows',
            }
            continue

        observed_delta = sum(vb - va for _, va, vb in paired) / len(paired)
        samples = []
        for _ in range(args.bootstrap_iters):
            sample = [paired[rng.randrange(len(paired))] for _ in range(len(paired))]
            samples.append(sum(vb - va for _, va, vb in sample) / len(sample))
        samples.sort()

        per_metric[metric] = {
            'n': len(paired),
            'mean_a': sum(va for _, va, _ in paired) / len(paired),
            'mean_b': sum(vb for _, _, vb in paired) / len(paired),
            'delta_b_minus_a': observed_delta,
            'wins': {
                args.label_a: wins_a,
                args.label_b: wins_b,
                'tie': ties,
            },
            'bootstrap_ci95': {
                'low': percentile(samples, 0.025),
                'high': percentile(samples, 0.975),
            },
        }

    result = {
        'label_a': args.label_a,
        'label_b': args.label_b,
        'id_field': args.id_field,
        'metrics': per_metric,
        'shared_ids': len(shared_ids),
    }
    text = json.dumps(result, indent=2)
    print(text)
    if args.json_out:
        args.json_out.write_text(text + '\n')


if __name__ == '__main__':
    main()
