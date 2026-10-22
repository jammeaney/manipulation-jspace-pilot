#!/usr/bin/env python3
"""Extract exact-ID text spans and reproducibly bundle saved paired-run summaries."""
import argparse
from collections import defaultdict
from contextlib import ExitStack
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
import uuid
import warnings
from zipfile import ZipFile, ZipInfo, ZIP_DEFLATED

CONDITIONS = {'neutral', 'covert_persuasion'}
SPANS = ('response_start', 'cot', 'answer')
CONCEPTS = ('concealment', 'persuasion', 'eval_awareness')
JLENS_FILES = ('samples.csv', 'layers.csv', 'layer_comparisons.csv', 'summary.json')
POLICY = dict(skip_special_tokens=False, clean_up_tokenization_spaces=False)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def safe_id(value):
    if not isinstance(value, str) or not re.fullmatch(r'[a-z0-9_]+', value):
        raise ValueError(f'Unsafe sample/scenario/condition ID: {value!r}')
    return value


def csv_bool(value, field):
    if value not in ('True', 'False'):
        raise ValueError(f'{field} must be True or False, got {value!r}')
    return value == 'True'


def load_tokenizer(model, revision):
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(model, revision=revision, trust_remote_code=False)
    except Exception as exc:
        raise RuntimeError(f'Cannot load required tokenizer {model}@{revision}: {exc}') from exc


def extract_span(transcript, name, tokenizer):
    span = transcript['spans'].get(name)
    if span is None:
        return '[No valid span recorded]\n'
    if (not isinstance(span, list) or len(span) != 2 or
            any(type(i) is not int for i in span) or
            not 0 <= span[0] <= span[1] <= len(transcript['full_ids'])):
        raise ValueError(f'Invalid {name} span: {span!r}')
    return tokenizer.decode(transcript['full_ids'][span[0]:span[1]], **POLICY)


def read_inputs(run):
    """Snapshot input bytes and validate the complete sample inventory before any writes."""
    source = {}
    def read(path):
        data = path.read_bytes()
        source[str(path.relative_to(run))] = data
        return data
    summaries = {name: read(run / 'summaries' / name) for name in JLENS_FILES}
    records = list(csv.DictReader(io.StringIO(summaries['samples.csv'].decode('utf-8'))))
    metrics = {}
    combinations = set()
    required = {'id', 'scenario_id', 'assigned_condition', 'target_option', 'generated_tokens',
                'cot_tokens', 'answer_tokens', 'stop_reason', 'thinking_span_valid',
                'thinking_generation_completed'}
    for row in records:
        if not required.issubset(row) or any(row[k] is None for k in required):
            raise ValueError('summaries/samples.csv is missing required columns or values')
        for span in SPANS:
            for concept in CONCEPTS:
                key = f'{concept}_{span}_workspace_score'
                value = row.get(key, '')
                if value != '' and (value is None or not math.isfinite(float(value))):
                    raise ValueError(f'Invalid numeric summary score: {row["id"]}: {key}')
        sample = safe_id(row['id'])
        if sample in metrics:
            raise ValueError(f'Duplicate sample ID in summaries/samples.csv: {sample}')
        scenario, condition = safe_id(row['scenario_id']), safe_id(row['assigned_condition'])
        if condition not in CONDITIONS:
            raise ValueError(f'Unsupported paired condition: {condition}')
        key = scenario, condition
        if key in combinations:
            raise ValueError(f'Duplicate scenario-condition combination: {key}')
        combinations.add(key)
        metrics[sample] = row
    if not metrics:
        raise ValueError('No samples in summaries/samples.csv')
    if not (run / 'samples').is_dir():
        raise ValueError(f'Missing samples directory: {run / "samples"}')
    discovered = {p.parent.name: p for p in (run / 'samples').glob('*/transcript.json') if p.is_file()}
    missing, unexpected = sorted(metrics.keys() - discovered.keys()), sorted(discovered.keys() - metrics.keys())
    if missing or unexpected:
        raise ValueError(f'Sample inventory mismatch; missing transcripts: {missing}; unexpected transcripts: {unexpected}')
    experiment = json.loads(read(run / 'experiment.json'))
    planned = [r['id'] for r in experiment['samples']]
    if len(planned) != len(set(planned)) or set(planned) != set(metrics):
        raise ValueError('Experiment sample IDs differ from summary sample IDs or contain duplicates')
    plan = {r['id']: r for r in experiment['samples']}
    samples = []
    pins = set()
    for sample, row in sorted(metrics.items()):
        entry = plan[sample]
        if (entry['scenario_id'], entry['assigned_condition']) != (row['scenario_id'], row['assigned_condition']):
            raise ValueError(f'Experiment scenario/condition differs from summary: {sample}')
        folder = run / 'samples' / sample
        transcript = json.loads(read(folder / 'transcript.json'))
        manifest = json.loads(read(folder / 'manifest.json'))
        provenance = manifest['provenance']
        pin = tuple(provenance[k] for k in ('model', 'model_revision', 'tokenizer_revision'))
        if not isinstance(pin[0], str) or not pin[0] or any(
                not isinstance(p, str) or not re.fullmatch(r'[a-fA-F0-9]{40}', p) for p in pin[1:]):
            raise ValueError(f'Missing immutable model/tokenizer commit pins: {sample}')
        pins.add(pin)
        full = transcript['prompt_ids'] + transcript['generated_ids']
        expected_hash = sha(json.dumps(full, separators=(',', ':')).encode())
        if (not transcript['prompt_ids'] or not transcript['generated_ids']
                or any(type(t) is not int or t < 0 for t in full)
                or full != transcript['full_ids'] or expected_hash != transcript['ids_sha256']
                or transcript['prompt_length'] != len(transcript['prompt_ids'])
                or transcript['generated_length'] != len(transcript['generated_ids'])
                or transcript['attention_mask'] != [1] * len(full)
                or [t['id'] for t in transcript['tokens']] != full):
            raise ValueError(f'Invalid saved token IDs/hash/lengths: {sample}')
        if transcript['decode_policy'] != POLICY:
            raise ValueError(f'Unsupported saved decode policy: {sample}')
        spans = transcript['spans']
        if spans['coordinate_system'] != 'full-sequence, zero-based, end-exclusive':
            raise ValueError(f'Unsupported span coordinate system: {sample}')
        if type(spans['span_analysis_eligible']) is not bool:
            raise ValueError(f'Invalid span eligibility: {sample}')
        valid = spans['span_analysis_eligible']
        completed = transcript['stop_reason'] == 'eos' and valid
        if (row['stop_reason'] != transcript['stop_reason']
                or int(row['generated_tokens']) != len(transcript['generated_ids'])
                or csv_bool(row['thinking_span_valid'], 'thinking_span_valid') != valid
                or csv_bool(row['thinking_generation_completed'], 'thinking_generation_completed') != completed):
            raise ValueError(f'Summary completion/span metadata disagrees with transcript: {sample}')
        samples.append((sample, row, transcript))
    if len(pins) != 1:
        raise ValueError('Run contains mixed model/tokenizer revisions')
    return summaries, source, samples, next(iter(pins))


def prepare_texts(samples, tokenizer):
    result = []
    for sample, row, transcript in samples:
        generated = tokenizer.decode(transcript['generated_ids'], **POLICY)
        if (generated != transcript['generated_text'] or
                tokenizer.decode(transcript['full_ids'], **POLICY) != transcript['full_text']):
            raise ValueError(f'Pinned tokenizer decoding differs from saved transcript: {sample}')
        texts = {name: extract_span(transcript, name, tokenizer) for name in ('cot', 'answer')}
        available = {name: transcript['spans'].get(name) is not None and bool(texts[name].strip())
                     for name in texts}
        valid = transcript['spans']['span_analysis_eligible']
        if valid and (not all(available.values()) or transcript['spans']['flags']):
            raise ValueError(f'Eligible spans are missing, empty or flagged: {sample}')
        completed = csv_bool(row['thinking_generation_completed'], 'thinking_generation_completed')
        metadata = dict(id=sample, scenario_id=row['scenario_id'], assigned_condition=row['assigned_condition'],
            ids_sha256=transcript['ids_sha256'], stop_reason=transcript['stop_reason'],
            thinking_generation_completed=completed, incomplete=not completed,
            truncated=transcript['stop_reason'] == 'max_new_tokens',
            span_valid=valid, span_available=available, span_flags=transcript['spans']['flags'])
        result.append((row, texts, generated, metadata))
    return result


def write_csv(path, rows, columns):
    with path.open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_review(dest, prepared, pins, source):
    dest.mkdir()
    overview, by_scenario, metadata = [], defaultdict(dict), []
    for m, texts, generated, detail in prepared:
        scenario, condition = m['scenario_id'], m['assigned_condition']
        target = dest / scenario
        target.mkdir(exist_ok=True)
        for span, suffix in [('cot', 'reasoning'), ('answer', 'answer')]:
            (target / f'{condition}_{suffix}.txt').write_text(texts[span], encoding='utf-8')
        if detail['incomplete']:
            (target / f'{condition}_incomplete_generation.txt').write_text(generated, encoding='utf-8')
        row = dict(scenario=scenario, condition=condition, target_option=m['target_option'],
            generated_tokens=m['generated_tokens'], cot_tokens=m['cot_tokens'], answer_tokens=m['answer_tokens'],
            stop_reason=m['stop_reason'], span_valid=m['thinking_span_valid'])
        for span in SPANS:
            for concept in CONCEPTS:
                key = f'{concept}_{span}_workspace_score'
                row[key] = m.get(key, '')
        overview.append(row)
        by_scenario[scenario][condition] = row
        metadata.append(detail)
    write_csv(dest / 'all_samples.csv', overview, list(overview[0]))
    comparisons = []
    for scenario, conditions in sorted(by_scenario.items()):
        if not CONDITIONS.issubset(conditions):
            continue
        neutral, covert = conditions['neutral'], conditions['covert_persuasion']
        row = dict(scenario=scenario, neutral_cot_tokens=neutral['cot_tokens'], covert_cot_tokens=covert['cot_tokens'])
        for span in SPANS:
            for concept in CONCEPTS:
                key = f'{concept}_{span}_workspace_score'
                values = [neutral[key], covert[key]]
                if any(v == '' for v in values):
                    value = ''
                else:
                    numbers = [float(v) for v in values]
                    if not all(math.isfinite(v) for v in numbers):
                        raise ValueError(f'Nonfinite score in {scenario}: {key}')
                    value = numbers[1] - numbers[0]
                row[f'{concept}_{span}_delta'] = value
        comparisons.append(row)
    columns = ['scenario', 'neutral_cot_tokens', 'covert_cot_tokens'] + [
        f'{concept}_{span}_delta' for span in SPANS for concept in CONCEPTS]
    write_csv(dest / 'paired_comparisons.csv', comparisons, columns)
    export = dict(schema_version=1, model=pins[0], model_revision=pins[1], tokenizer_revision=pins[2],
        decode_policy=POLICY, samples=metadata, paired_scenarios=len(comparisons),
        input_sha256={name: sha(data) for name, data in sorted(source.items())},
        exporter_sha256=sha(Path(__file__).read_bytes()))
    (dest / 'export_metadata.json').write_text(json.dumps(export, indent=2, ensure_ascii=False, sort_keys=True)+'\n', encoding='utf-8')


def write_zip(path, files):
    with ZipFile(path, 'x', ZIP_DEFLATED, compresslevel=9) as archive:
        for name, data in sorted(files):
            info = ZipInfo(str(name), date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.compress_type = ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, data, compresslevel=9)
    with ZipFile(path) as archive:
        if archive.testzip() is not None:
            raise ValueError(f'ZIP integrity check failed: {path}')


def check_destinations(targets, overwrite):
    for target, directory in targets:
        if target.is_symlink():
            raise ValueError(f'Export destination may not be a symlink: {target}')
        if target.exists():
            if not overwrite:
                raise FileExistsError(f'Export already exists: {target}; use --overwrite to replace it')
            if (directory and not target.is_dir()) or (not directory and not target.is_file()):
                raise ValueError(f'Unexpected export destination type: {target}')
    for index, (target, _) in enumerate(targets):
        for other, _ in targets[index+1:]:
            if target == other or target in other.parents or other in target.parents:
                raise ValueError('Export destinations must not overlap')


def publish(staged, overwrite):
    """Rollback normal publication errors; retain backups if rollback itself fails."""
    check_destinations([(target, source.is_dir()) for source, target in staged], overwrite)
    backups, published = [], []
    try:
        for source, target in staged:
            if target.exists():
                if not overwrite:
                    raise FileExistsError(f'Export destination appeared during publication: {target}')
                backup = target.with_name(f'.{target.name}.backup-{uuid.uuid4().hex}')
                os.replace(target, backup)
                backups.append((target, backup))
            os.replace(source, target)
            published.append((source, target))
    except BaseException:
        errors = []
        for source, target in reversed(published):
            try:
                os.replace(target, source)
            except OSError as exc:
                errors.append(str(exc))
        for target, backup in reversed(backups):
            try:
                os.replace(backup, target)
            except OSError as exc:
                errors.append(f'{exc}; previous export retained at {backup}')
        if errors:
            raise RuntimeError('Export rollback incomplete: ' + '; '.join(errors))
        raise
    for _, backup in backups:
        try:
            shutil.rmtree(backup) if backup.is_dir() else backup.unlink()
        except OSError as exc:
            warnings.warn(f'Export published; old backup retained at {backup}: {exc}')


def export_run(run, zip_dir=Path('/workspace'), overwrite=False):
    run = Path(run).resolve()
    zip_dir = Path(zip_dir).absolute()
    dest = run / 'research-review'
    review_zip = zip_dir / f'{run.name}-research-review.zip'
    jlens_zip = zip_dir / f'{run.name}-jlens-analysis.zip'
    targets = [(dest, True), (review_zip, False), (jlens_zip, False)]
    # Preserve final-component symlinks for rejection, but normalize parent aliases.
    targets = [(p.parent.resolve() / p.name, directory) for p, directory in targets]
    dest, review_zip, jlens_zip = [p for p, _ in targets]
    check_destinations(targets, overwrite)
    summaries, source, samples, pins = read_inputs(run)
    tokenizer = load_tokenizer(pins[0], pins[2])
    prepared = prepare_texts(samples, tokenizer)
    # All inputs validated/decoded before creating staging locations or touching exports.
    zip_dir = review_zip.parent
    zip_dir.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        tree_stage = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix='.research-review-', dir=run)))
        zip_stage = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix='.research-zips-', dir=zip_dir)))
        tree = tree_stage / 'research-review'
        write_review(tree, prepared, pins, source)
        zip1, zip2 = zip_stage / review_zip.name, zip_stage / jlens_zip.name
        write_zip(zip1, [(Path('research-review') / p.relative_to(tree), p.read_bytes())
                         for p in tree.rglob('*') if p.is_file()])
        write_zip(zip2, list(summaries.items()))
        publish([(tree, dest), (zip1, review_zip), (zip2, jlens_zip)], overwrite)
    print(f'Extracted {len(samples)} samples')
    print(f'Research review: {review_zip}')
    print(f'J-lens summaries: {jlens_zip}')
    return dest, review_zip, jlens_zip


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', help='Run name (paired-008) or run directory')
    parser.add_argument('--outputs-dir', type=Path, default=Path('outputs'))
    parser.add_argument('--zip-dir', type=Path, default=Path('/workspace'))
    parser.add_argument('--overwrite', action='store_true', help='Replace exports only after new exports are ready')
    args = parser.parse_args()
    run = Path(args.run)
    if not run.is_dir():
        run = args.outputs_dir / args.run
    if not run.is_dir():
        parser.error(f'Run directory not found: {run}')
    try:
        export_run(run, args.zip_dir, args.overwrite)
    except (ValueError, OSError, KeyError, TypeError, RuntimeError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
