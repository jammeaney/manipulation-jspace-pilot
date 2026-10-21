"""Report saved per-token replay errors and plot them, without model loading."""
import argparse
import csv
import hashlib
from html import escape
import json
import math
from pathlib import Path


THRESHOLD = 0.5  # Existing diagnostic threshold; not a numerical pass/fail criterion.


def diagnostic_rows(transcript, validation):
    full = transcript['prompt_ids'] + transcript['generated_ids']
    ids_hash = hashlib.sha256(json.dumps(full, separators=(',', ':')).encode()).hexdigest()
    if (full != transcript['full_ids'] or ids_hash != transcript['ids_sha256']
            or validation['replayed_ids_sha256'] != ids_hash or not validation['token_identity']):
        raise ValueError('Transcript/replay token identity mismatch')
    threshold = validation['diagnostics'].get('flag_threshold_nats', THRESHOLD)
    if threshold != THRESHOLD:
        raise ValueError('This report requires the existing 0.5-nat diagnostic threshold')
    saved = validation['per_generated_token']
    if len(saved) != len(transcript['generated_ids']):
        raise ValueError('Incomplete per-generated-token diagnostics')
    rows = []
    for j, item in enumerate(saved):
        position = len(transcript['prompt_ids']) + j
        token = transcript['tokens'][position]
        generation, replay = item['generation_logprob'], item['replay_logprob']
        error = abs(generation - replay)
        if (item['generated_index'] != j or item['token_id'] != full[position]
                or item['prediction_position'] != position - 1 or token['id'] != full[position]
                or generation != transcript['raw_sampled_logprobs'][j]
                or not all(math.isfinite(v) for v in (generation, replay, error))
                or not math.isclose(error, item['absolute_error_nats'], abs_tol=1e-8)):
            raise ValueError('Misaligned or inconsistent per-token diagnostic')
        rows.append(dict(generated_index=j, token_id=item['token_id'],
            token_text=token['decoded_piece'], token_string=token['token_string'],
            generation_logprob=generation, replay_logprob=replay,
            absolute_error_nats=error, flagged=error > THRESHOLD))
    if not rows:
        raise ValueError('No generated-token diagnostics')
    return rows


def plot_errors(rows, path):
    left, right, top, bottom = 80, 920, 55, 345
    xmax = max(1, max(r['generated_index'] for r in rows))
    ymax = max(THRESHOLD * 1.2, max(r['absolute_error_nats'] for r in rows) * 1.1)
    def point(row):
        return (left + row['generated_index']/xmax*(right-left),
                bottom - row['absolute_error_nats']/ymax*(bottom-top))
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="960" height="420" viewBox="0 0 960 420">',
             '<rect width="100%" height="100%" fill="white"/>',
             '<g font-family="sans-serif" font-size="13">',
             '<text x="80" y="25">Absolute replay error (nats); red points exceed 0.5 nats</text>']
    for i in range(6):
        y = bottom-i/5*(bottom-top)
        x = left+i/5*(right-left)
        parts += [f'<path d="M {left} {y} H {right}" stroke="#ddd"/>',
                  f'<text x="20" y="{y+4}">{ymax*i/5:.3g}</text>',
                  f'<text x="{x}" y="370" text-anchor="middle">{xmax*i/5:.0f}</text>']
    y = bottom-THRESHOLD/ymax*(bottom-top)
    parts += [f'<path d="M {left} {y} H {right}" stroke="#c22" stroke-dasharray="6 4"/>',
              f'<text x="{left+10}" y="{y-5}">0.5-nat threshold</text>']
    points = ' '.join(f'{x:.3f},{y:.3f}' for x, y in map(point, rows))
    parts.append(f'<polyline points="{points}" fill="none" stroke="#1769aa" stroke-width="1.5"/>')
    for row in rows:
        x, y = point(row)
        color = '#c22' if row['flagged'] else '#1769aa'
        title = escape(f"Index {row['generated_index']}: {row['token_text']!r}; error {row['absolute_error_nats']:.6g} nats")
        parts.append(f'<circle cx="{x}" cy="{y}" r="3" fill="{color}"><title>{title}</title></circle>')
    parts += ['<text x="500" y="405" text-anchor="middle">Generated-token index (zero-based)</text>',
              '<text x="15" y="200" transform="rotate(-90 15 200)" text-anchor="middle">Absolute error (nats)</text>', '</g></svg>']
    with path.open('x') as stream:
        stream.write('\n'.join(parts)+'\n')


def report(sample_dir, output_dir, indices=None):
    transcript = json.loads((sample_dir / 'transcript.json').read_text())
    validation = json.loads((sample_dir / 'replay_validation.json').read_text())
    rows = diagnostic_rows(transcript, validation)
    if indices is not None and any(i < 0 or i >= len(rows) for i in indices):
        raise ValueError('Requested generated-token index out of range')
    selected = rows if indices is None else [rows[i] for i in indices]
    output_dir.mkdir(parents=True, exist_ok=False)
    with (output_dir / 'replay_tokens.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(selected)
    # Plot all positions even when the tabular report selects individual indices.
    plot_errors(rows, output_dir / 'replay_errors.svg')
    print(json.dumps({'reported_tokens': len(selected), 'plotted_tokens': len(rows),
        'flagged_tokens': sum(r['flagged'] for r in rows), 'output_dir': str(output_dir)}))
    if indices is not None:
        for row in selected:
            print(json.dumps(row, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sample-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True, help='Fresh directory; never overwritten')
    parser.add_argument('--indices', type=int, nargs='+', help='Zero-based generated indices; default: all')
    args = parser.parse_args()
    report(args.sample_dir, args.output_dir, args.indices)


if __name__ == '__main__':
    main()
