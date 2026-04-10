#!/usr/bin/env python3
"""Export CounterFact benchmark outputs into per-example JSONL/CSV rows.

This is a thin adapter over SEKA's saved `efficacy.json` and `paraphrase.json`
artifacts. The upstream evaluator already stores per-example samples, but the
paper-facing paired bootstrap script expects a flat table with shared ids.
"""

import argparse
import csv
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description='Flatten CounterFact benchmark JSON into per-example rows.'
    )
    parser.add_argument('--efficacy-json', type=Path, required=True)
    parser.add_argument('--paraphrase-json', type=Path, required=True)
    parser.add_argument('--out-jsonl', type=Path, required=True)
    parser.add_argument('--out-csv', type=Path, default=None)
    parser.add_argument(
        '--allow-missing',
        action='store_true',
        help='Allow mismatched efficacy/paraphrase id sets instead of failing.',
    )
    return parser.parse_args()


def load_json(path):
    with path.open() as handle:
        return json.load(handle)


def efficacy_label(target_score, comparator_score):
    return float(target_score > comparator_score)


def efficacy_magnitude(target_score, comparator_score):
    return float(target_score - comparator_score)


def build_rows(efficacy_obj, paraphrase_obj, allow_missing=False):
    efficacy_samples = {
        str(sample['id']): sample for sample in efficacy_obj.get('samples', [])
    }
    paraphrase_samples = {
        str(sample['id']): sample for sample in paraphrase_obj.get('samples', [])
    }

    efficacy_ids = set(efficacy_samples)
    paraphrase_ids = set(paraphrase_samples)
    if not allow_missing and efficacy_ids != paraphrase_ids:
        missing_in_paraphrase = sorted(efficacy_ids - paraphrase_ids)
        missing_in_efficacy = sorted(paraphrase_ids - efficacy_ids)
        raise ValueError(
            'Mismatched sample ids between efficacy/paraphrase. '
            f'missing_in_paraphrase={missing_in_paraphrase[:10]} '
            f'missing_in_efficacy={missing_in_efficacy[:10]}'
        )

    shared_ids = sorted(
        efficacy_ids & paraphrase_ids,
        key=lambda x: (int(x) if x.isdigit() else x),
    )
    rows = []
    for example_id in shared_ids:
        efficacy_sample = efficacy_samples[example_id]
        paraphrase_sample = paraphrase_samples[example_id]
        target_score = float(efficacy_sample['target_score'])
        comparator_score = float(efficacy_sample['comparator_score'])
        prompts = paraphrase_sample.get('prompts', [])
        rows.append(
            {
                'example_id': example_id,
                'efficacy': efficacy_label(target_score, comparator_score),
                'efficacy_magnitude': efficacy_magnitude(
                    target_score, comparator_score
                ),
                'efficacy_target_score': target_score,
                'efficacy_comparator_score': comparator_score,
                'paraphrase': float(paraphrase_sample['efficacy_score']),
                'paraphrase_magnitude': float(
                    paraphrase_sample['efficacy_magnitude']
                ),
                'paraphrase_prompt_count': len(prompts),
            }
        )
    return rows


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as handle:
        for row in rows:
            handle.write(json.dumps(row) + '\n')


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        'example_id',
        'efficacy',
        'efficacy_magnitude',
        'efficacy_target_score',
        'efficacy_comparator_score',
        'paraphrase',
        'paraphrase_magnitude',
        'paraphrase_prompt_count',
    ]
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    args = parse_args()
    efficacy_obj = load_json(args.efficacy_json)
    paraphrase_obj = load_json(args.paraphrase_json)
    rows = build_rows(
        efficacy_obj,
        paraphrase_obj,
        allow_missing=args.allow_missing,
    )

    write_jsonl(args.out_jsonl, rows)
    if args.out_csv:
        write_csv(args.out_csv, rows)

    summary = {
        'rows': len(rows),
        'out_jsonl': str(args.out_jsonl),
        'out_csv': str(args.out_csv) if args.out_csv else None,
    }
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
