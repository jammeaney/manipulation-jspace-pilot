"""Five paired fictional scenarios: descriptive feasibility only, never a detector."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
from string import Formatter
from types import SimpleNamespace

import jlens
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

if __package__:
    from . import single_example as single
else:
    import single_example as single

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/paired-feasibility'
CONDITIONS = ('neutral', 'covert_persuasion')
GROUPS = ('concealment', 'persuasion', 'eval_awareness')
SPANS = ('response_start', 'cot', 'answer', 'user_prompt')
LIMITATIONS = [
    'Exploratory five-scenario feasibility contrast, not the 90-sample pilot or a detector.',
    'No overt, echo, or instruction-word controls; instruction echo is unresolved.',
    'One target direction per scenario; no both-sides or repeated-generation inference.',
    'Concept sets are frozen drafts, not validated on an independent development set.',
    'Assigned conditions are not independent behavioural outcomes; these remain unevaluated.',
    'No AUC, confidence intervals, significance testing, classifier, or bootstrap.',
]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


CONTROL_CONDITIONS = ('overt_persuasion', 'echo_control', 'instruction_word_control')


def resolve_target(case, target_option=None):
    """Validate both directions; selecting a direction never schedules a generation."""
    statements = case.get('target_statements', {})
    if set(case['options']) != {'A', 'B'} or set(statements) != {'A', 'B'}:
        raise ValueError('Each scenario needs options and target statements A/B')
    for side, option in case['options'].items():
        statement = statements[side]
        if (not isinstance(statement, str) or
                not re.fullmatch(r'.+ should choose Option ' + side + r', ' + re.escape(option), statement)):
            raise ValueError(f'Target statement {side} must identify its corresponding option')
    selected = case.get('target_option') if target_option is None else target_option
    if selected not in statements:
        raise ValueError('Invalid target option')
    return statements[selected]


def echo_question(config, echo_prompt_id=None):
    bank = config['echo_prompts']
    prompts = bank['prompts']
    if bank.get('frozen') is not True or bank.get('version') != 1 or not prompts:
        raise ValueError('A frozen versioned echo prompt bank is required')
    by_id = {}
    for prompt in prompts:
        if (not re.fullmatch(r'[a-z0-9_]+', prompt['id']) or prompt['id'] in by_id
                or prompt['topic'] not in {'concealment', 'persuasion'}
                or not isinstance(prompt['question'], str) or not prompt['question'].strip()
                or '{' in prompt['question'] or '}' in prompt['question']):
            raise ValueError('Invalid or duplicate echo prompt')
        by_id[prompt['id']] = prompt['question']
    if bank['default_id'] not in by_id:
        raise ValueError('Unknown default echo prompt ID')
    selected = bank['default_id'] if echo_prompt_id is None else echo_prompt_id
    if selected not in by_id:
        raise ValueError('Unknown echo prompt ID')
    return by_id[selected]


def construct_prompt(config, case, instruction, question=None):
    """One system/user structure for every condition; echo replaces only question."""
    return {'system_prompt': config['common_instruction'] + '\n\n' + instruction,
            'user_prompt': f"Background:\n{case['background']}\n\nQuestion:\n{case['question'] if question is None else question}"}


def render_control(config, case, condition, target_option=None, echo_prompt_id=None):
    """Render dormant full-pilot templates; not used by the paired runner."""
    if condition not in CONTROL_CONDITIONS:
        raise ValueError('Unknown control condition')
    templates = config.get('control_templates', {})
    if set(templates) != set(CONTROL_CONDITIONS):
        raise ValueError('All three separate control templates are required')
    template = templates[condition]
    if set(template) != {'system_instruction_template', 'user_prompt_template'}:
        raise ValueError('Control needs explicit system and user templates')
    values = {'target_statement': resolve_target(case, target_option),
              'paired_user_prompt': construct_prompt(config, case, '')['user_prompt'],
              'echo_question': echo_question(config, echo_prompt_id)}
    fields = {}
    for key, text in template.items():
        if not isinstance(text, str) or not text.strip():
            raise ValueError('Control template must be nonempty text')
        parsed = list(Formatter().parse(text))
        names = {name for _, name, _, _ in parsed if name is not None}
        if names - values.keys() or any(spec or conversion for _, _, spec, conversion in parsed):
            raise ValueError('Unsupported control template placeholder')
        fields[key] = names
    if fields['system_instruction_template'] != ({'target_statement'} if condition == 'overt_persuasion' else set()):
        raise ValueError('Only overt control system must contain target_statement')
    if fields['user_prompt_template'] != ({'echo_question'} if condition == 'echo_control' else {'paired_user_prompt'}):
        raise ValueError('Echo uses a topic question; other controls reuse paired user content')
    expected_user_template = '{echo_question}' if condition == 'echo_control' else '{paired_user_prompt}'
    if template['user_prompt_template'] != expected_user_template:
        raise ValueError('User template must preserve the shared message structure exactly')
    question = values['echo_question'] if condition == 'echo_control' else None
    return construct_prompt(config, case, template['system_instruction_template'].format(**values), question)


def build_samples(config):
    scenarios = config['scenarios']
    if config.get('fictional') is not True or len(scenarios) not in (1, 5):
        raise ValueError('Exactly five explicitly fictional scenarios are required')
    rows, seen = [], set()
    for index, case in enumerate(scenarios):
        if not re.fullmatch(r'[a-z0-9_]+', case['id']) or case['id'] in seen:
            raise ValueError('Scenario IDs must be unique safe directory names')
        seen.add(case['id'])
        target_statement = resolve_target(case)
        for condition in CONTROL_CONDITIONS:
            render_control(config, case, condition)
        user = construct_prompt(config, case, '')['user_prompt']
        # Alternate first condition across scenarios; do not adapt order after seeing results.
        order = CONDITIONS if index % 2 == 0 else CONDITIONS[::-1]
        for condition in order:
            instruction = (config['neutral_instruction'] if condition == 'neutral' else
                           config['covert_instruction_template'].format(target_statement=target_statement))
            prompt = construct_prompt(config, case, instruction)
            rows.append({'id': f"{case['id']}_{condition}_01", 'scenario_id': case['id'],
                         'issue': case['issue'], 'side': case['target_option'],
                         'target_statement': target_statement, 'options': case['options'],
                         'assigned_condition': condition, 'condition': condition,
                         'persona': 'undecided', 'realism': 'test_like',
                         'factual_background': case['background'], 'question': case['question'],
                         'condition_instruction': instruction,
                         **prompt,
                         'common_user_sha256': hashlib.sha256(user.encode()).hexdigest(),
                         'behavioural_outcome': None, 'behavioural_outcome_status': 'not_evaluated'})
    validate_pairing(rows)
    return rows


def validate_pairing(rows):
    pairs = defaultdict(list)
    if len(rows) not in (2, 10) or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Expected 2 or 10 unique sample IDs')
    for row in rows:
        if not re.fullmatch(r'[a-z0-9_]+', row['id']) or row['condition'] != row['assigned_condition']:
            raise ValueError('Invalid sample ID or assigned condition')
        pairs[row['scenario_id']].append(row)
    if len(pairs) != len(rows) // 2:
        raise ValueError('Each scenario must have exactly two samples')
    for pair in pairs.values():
        if len(pair) != 2 or {r['condition'] for r in pair} != set(CONDITIONS):
            raise ValueError('Every scenario needs exactly one sample per condition')
        a, b = pair
        for key in ['factual_background', 'question', 'user_prompt', 'common_user_sha256', 'target_statement', 'side']:
            if a[key] != b[key]:
                raise ValueError(f'Paired {key} differs')
        if hashlib.sha256(a['user_prompt'].encode()).hexdigest() != a['common_user_sha256']:
            raise ValueError('Paired user content hash mismatch')
        if any(not r['system_prompt'].endswith(r['condition_instruction']) or not r['condition_instruction'] for r in pair):
            raise ValueError('Condition instruction must be an explicit system suffix')
        if a['system_prompt'].removesuffix(a['condition_instruction']) != b['system_prompt'].removesuffix(b['condition_instruction']):
            raise ValueError('Common system context differs')


def load_concepts(path, tokenizer=None):
    config = json.loads(path.read_text())
    groups = config['groups']
    if config['tokenizer_revision'] != single.MODEL_REVISION or set(groups) != set(GROUPS) | {'control'}:
        raise ValueError('Concept config must match the pinned tokenizer and documented groups')
    if len(groups['control']) != 100 or any(not words for words in groups.values()):
        raise ValueError('Exactly 100 controls and nonempty concept sets are required')
    for words in groups.values():
        for word, ids in words.items():
            if not ids or any(type(i) is not int or i < 0 for i in ids) or len(ids) != len(set(ids)):
                raise ValueError('Invalid frozen concept token IDs')
            if tokenizer is not None:
                actual = []
                for spelling in (word, ' ' + word):
                    encoded = tokenizer.encode(spelling, add_special_tokens=False)
                    if len(encoded) == 1 and encoded[0] not in actual:
                        actual.append(encoded[0])
                if actual != ids:
                    raise ValueError(f'Frozen token IDs do not match tokenizer for {word!r}')
    return config


def prepare(args):
    config = json.loads(args.scenarios.read_text())
    rows = build_samples(config)
    concepts = load_concepts(args.concepts)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    single.write_json(args.output_dir / 'concepts.json', concepts)
    with (args.output_dir / 'prompts.jsonl').open('x') as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + '\n')
    plan = {'schema_version': 1, 'specification': 'pilot_spec_26_10_2.md', 'status': 'prepared',
            'prompt_revision': config.get('prompt_revision', 1), 'fictional': config['fictional'],
            'inactive_control_templates': config['control_templates'],
            'inactive_echo_prompts': config['echo_prompts'],
            'created_at_utc': datetime.now(timezone.utc).isoformat(), 'independent_scenarios': 5,
            'assigned_conditions': list(CONDITIONS), 'samples': rows,
            'generation_settings': {'temperature': 1.0, 'seed': args.seed,
                'max_new_tokens': args.max_new_tokens, 'position_chunk_size': args.position_chunk_size,
                'logprob_flag_threshold': .5, 'device': args.device,
                'dtype': args.dtype or ('bfloat16' if args.device == 'cuda' else 'float32')},
            'scenario_source_sha256': digest(args.scenarios), 'concept_source_sha256': digest(args.concepts),
            'resolved_concepts_sha256': digest(args.output_dir / 'concepts.json'),
            'runner_sha256': digest(Path(__file__)), 'limitations': LIMITATIONS,
            'behavioural_outcomes_evaluated': False, 'completed_samples': []}
    single.write_json(args.output_dir / 'experiment.json', plan)
    return plan


def run(args):
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('No CUDA GPU available; no experiment or pretrained model was started')
    dtype = args.dtype or ('bfloat16' if args.device == 'cuda' else 'float32')
    if args.device == 'cuda' and dtype == 'bfloat16' and not torch.cuda.is_bf16_supported():
        raise ValueError('CUDA GPU must support bf16 or explicitly select float32')
    plan = prepare(args)
    tokenizer = AutoTokenizer.from_pretrained(single.MODEL, revision=single.MODEL_REVISION)
    concepts = load_concepts(args.output_dir / 'concepts.json', tokenizer)['groups']
    hf = AutoModelForCausalLM.from_pretrained(single.MODEL, revision=single.MODEL_REVISION,
                                            dtype=getattr(torch, dtype), device_map=args.device)
    lens = jlens.JacobianLens.from_pretrained(single.LENS, filename=single.LENS_FILE,
                                             revision=single.LENS_REVISION)
    (args.output_dir / 'samples').mkdir()
    plan['status'] = 'running'
    single.write_json(args.output_dir / 'experiment.json', plan)
    for index, row in enumerate(plan['samples']):
        example = SimpleNamespace(output_dir=args.output_dir / 'samples' / row['id'],
            prompt=row['user_prompt'], system=row['system_prompt'], device=args.device, dtype=dtype,
            max_new_tokens=args.max_new_tokens, temperature=1.0, seed=args.seed,
            position_chunk_size=args.position_chunk_size, logprob_flag_threshold=.5)
        single.run_example(example, hf, tokenizer, lens, concept_token_ids=concepts,
                           print_tokens=index == 0)
        plan['completed_samples'].append(row['id'])
        single.write_json(args.output_dir / 'experiment.json', plan)
        print(json.dumps({'completed': len(plan['completed_samples']), 'total': 10, 'id': row['id']}), flush=True)
    plan['status'] = 'complete'
    single.write_json(args.output_dir / 'experiment.json', plan)
    summarize(args.output_dir, args.output_dir / 'summaries')


def write_csv(path, rows):
    if not rows:
        raise ValueError('No summary rows produced')
    with path.open('x', newline='') as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def layer_statistics(path, data, manifest):
    totals, seen = {}, set()
    with path.open() as stream:
        for line in stream:
            row = json.loads(line)
            layer, pos = row['layer'], row['position']
            if layer not in manifest['lens_layers'] or not 0 <= pos < len(data['full_ids']) or (layer, pos) in seen:
                raise ValueError('Duplicate or out-of-range lens readout')
            seen.add((layer, pos))
            if row['input_token_id'] != data['full_ids'][pos] or row['spans'] != single.position_roles(pos, data['spans']):
                raise ValueError('Readout position or span mismatch')
            concepts = row['concept_readouts']
            control = concepts['control_mean_logprob']
            if not math.isfinite(control) or control > 0 or set(concepts['groups']) != set(GROUPS):
                raise ValueError('Invalid full-vocabulary concept probabilities')
            for span in row['spans']:
                for group, value in concepts['groups'].items():
                    raw, score = value['mean_logprob'], value['control_centered_score']
                    if (not math.isfinite(raw) or raw > 0 or not math.isfinite(score)
                            or not math.isclose(score, raw - control, abs_tol=1e-6)):
                        raise ValueError('Invalid concept score')
                    key = (layer, span, group)
                    acc = totals.setdefault(key, [0, 0., 0., 0.])
                    acc[0] += 1
                    acc[1] += raw
                    acc[2] += score
                    acc[3] += score * score
    if len(seen) != len(manifest['lens_layers']) * len(data['full_ids']):
        raise ValueError('Incomplete lens readout coverage')
    return [{'layer': layer, 'span': span, 'concept_set': group, 'positions': n,
             'mean_logprob': raw / n, 'mean_control_logprob': (raw - score) / n,
             'mean_control_centered_score': score / n,
             'position_score_sd': math.sqrt(max(0., sq / n - (score / n) ** 2))}
            for (layer, span, group), (n, raw, score, sq) in sorted(totals.items())]


def paired_comparisons(layer_rows, workspace=False):
    grouped = defaultdict(dict)
    for row in layer_rows:
        key = (row['scenario_id'], row['span'], row['concept_set'])
        if not workspace:
            key += (row['layer'],)
        grouped[key][row['assigned_condition']] = row
    matched = defaultdict(list)
    for key, pair in grouped.items():
        if set(pair) != set(CONDITIONS):
            continue  # Missing/ineligible spans never become zero scores.
        neutral, covert = pair['neutral'], pair['covert_persuasion']
        group_key = key[1:]
        matched[group_key].append({'scenario_id': key[0],
            'neutral_score': neutral['mean_control_centered_score'],
            'covert_score': covert['mean_control_centered_score'],
            'neutral_logprob': neutral['mean_logprob'], 'covert_logprob': covert['mean_logprob'],
            'score_delta': covert['mean_control_centered_score'] - neutral['mean_control_centered_score'],
            'logprob_delta': covert['mean_logprob'] - neutral['mean_logprob']})
    result = []
    keys = [(span, group) for span in SPANS for group in GROUPS]
    if not workspace:
        keys = [key + (layer,) for key in keys for layer in sorted({r['layer'] for r in layer_rows})]
    for key in keys:
        pairs = matched.get(key, [])
        def mean(field):
            return statistics.mean(p[field] for p in pairs) if pairs else None
        row = {'span': key[0], 'concept_set': key[1], 'n_pairs': len(pairs),
               'neutral_mean_score': mean('neutral_score'), 'covert_mean_score': mean('covert_score'),
               'neutral_mean_logprob': mean('neutral_logprob'), 'covert_mean_logprob': mean('covert_logprob'),
               'mean_paired_score_delta': mean('score_delta'), 'mean_paired_logprob_delta': mean('logprob_delta'),
               'included_scenarios': ';'.join(p['scenario_id'] for p in pairs)}
        if not workspace:
            row['layer'] = key[2]
        result.append(row)
    return result


def summarize(run_dir, output_dir):
    plan = json.loads((run_dir / 'experiment.json').read_text())
    rows = [json.loads(line) for line in (run_dir / 'prompts.jsonl').read_text().splitlines()]
    if plan['status'] != 'complete' or len(rows) not in (2, 10) or rows != plan['samples']:
        raise ValueError('All saved samples must complete before summary; plan must match prompts')
    validate_pairing(rows)
    if digest(run_dir / 'concepts.json') != plan['resolved_concepts_sha256']:
        raise ValueError('Frozen concept file changed')
    frozen = load_concepts(run_dir / 'concepts.json')['groups']
    samples, layer_rows, workspace_rows = [], [], []
    for row in rows:
        base = run_dir / 'samples' / row['id']
        manifest = json.loads((base / 'manifest.json').read_text())
        data = json.loads((base / 'transcript.json').read_text())
        validation = json.loads((base / 'replay_validation.json').read_text())
        full = data['prompt_ids'] + data['generated_ids']
        if (manifest['status'] != 'complete' or full != data['full_ids']
                or single.ids_hash(full) != data['ids_sha256']
                or validation['replayed_ids_sha256'] != data['ids_sha256']
                or not validation['token_identity'] or not validation['original_generation_checked']
                or data['messages'] != [{'role': 'system', 'content': row['system_prompt']},
                                        {'role': 'user', 'content': row['user_prompt']}]):
            raise ValueError('Sample identity, messages or completion validation failed')
        if (manifest['provenance']['model_revision'] != single.MODEL_REVISION
                or manifest['provenance']['tokenizer_revision'] != single.MODEL_REVISION
                or manifest['provenance']['lens_revision'] != single.LENS_REVISION
                or manifest.get('concept_token_ids') != frozen):
            raise ValueError('Sample revisions or concept sets differ from frozen configuration')
        expected = plan['generation_settings']
        if any(manifest['generation_settings'][k] != expected[k]
               for k in ['temperature', 'seed', 'max_new_tokens', 'position_chunk_size', 'logprob_flag_threshold']):
            raise ValueError('Generation settings differ between samples and plan')
        metrics = layer_statistics(base / 'lens_readouts.jsonl', data, manifest)
        common = {'id': row['id'], 'scenario_id': row['scenario_id'], 'assigned_condition': row['assigned_condition']}
        layer_rows.extend({**common, **metric} for metric in metrics)
        sample = {**common, 'target_option': row['side'], 'behavioural_outcome_status': 'not_evaluated',
                  'behavioural_outcome': None, 'prompt_tokens': len(data['prompt_ids']),
                  'generated_tokens': len(data['generated_ids']), 'generated_characters': len(data['generated_text']),
                  'stop_reason': data['stop_reason'], 'thinking_span_valid': data['spans']['span_analysis_eligible'],
                  'thinking_generation_completed': data['stop_reason'] == 'eos' and data['spans']['span_analysis_eligible'],
                  'span_flags': ';'.join(data['spans']['flags']),
                  'cot_tokens': None if not data['spans']['cot'] else data['spans']['cot'][1] - data['spans']['cot'][0],
                  'answer_tokens': None if not data['spans']['answer'] else data['spans']['answer'][1] - data['spans']['answer'][0],
                  'token_identity': validation['token_identity'],
                  'replay_median_abs_error_nats': validation['diagnostics']['median_abs_error_nats'],
                  'replay_p99_abs_error_nats': validation['diagnostics']['p99_abs_error_nats'],
                  'replay_max_abs_error_nats': validation['diagnostics']['max_abs_error_nats'],
                  'replay_flagged_tokens': len(validation['diagnostics']['large_error_generated_indices']),
                  'replay_persistent_jump_flags': len(validation['diagnostics'].get('possible_step_change_generated_indices', [])),
                  'generation_seconds': manifest['generation_seconds'], 'replay_seconds': manifest['replay_seconds']}
        for span in SPANS:
            for group in GROUPS:
                selected = [m for m in metrics if m['span'] == span and m['concept_set'] == group and 12 <= m['layer'] <= 29]
                score = statistics.mean(m['mean_control_centered_score'] for m in selected) if selected else None
                sample[f'{group}_{span}_workspace_score'] = score
                if selected:
                    workspace_rows.append({**common, 'span': span, 'concept_set': group,
                        'mean_control_centered_score': score,
                        'mean_logprob': statistics.mean(m['mean_logprob'] for m in selected)})
        samples.append(sample)
    output_dir.mkdir(parents=True, exist_ok=False)
    write_csv(output_dir / 'samples.csv', samples)
    write_csv(output_dir / 'layers.csv', layer_rows)
    contrasts = paired_comparisons(layer_rows)
    write_csv(output_dir / 'layer_comparisons.csv', contrasts)
    condition_summary = {}
    for condition in CONDITIONS:
        selected = [s for s in samples if s['assigned_condition'] == condition]
        condition_summary[condition] = {'samples': len(selected),
            'mean_generated_tokens': statistics.mean(s['generated_tokens'] for s in selected),
            'thinking_generation_completed': sum(s['thinking_generation_completed'] for s in selected),
            'span_eligible': sum(s['thinking_span_valid'] for s in selected)}
    single.write_json(output_dir / 'summary.json', {'schema_version': 1,
        'independent_scenarios': 5, 'generations': 10, 'behavioural_outcomes_evaluated': False,
        'limitations': LIMITATIONS, 'condition_summary': condition_summary,
        'workspace_paired_comparisons': paired_comparisons(workspace_rows, workspace=True),
        'pairing': 'within scenario, covert_persuasion minus neutral; means across positions then layers',
        'run_manifest_sha256': digest(run_dir / 'experiment.json'),
        'concepts_sha256': digest(run_dir / 'concepts.json')})
    print(json.dumps({'status': 'summarized', 'output_dir': str(output_dir), 'samples': 10}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for mode in ['prepare', 'run']:
        cmd = sub.add_parser(mode)
        cmd.add_argument('--output-dir', type=Path, required=True)
        cmd.add_argument('--scenarios', type=Path, default=DATA / 'scenarios.json')
        cmd.add_argument('--concepts', type=Path, default=DATA / 'concepts.json')
        cmd.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
        cmd.add_argument('--dtype', choices=['bfloat16', 'float32'])
        cmd.add_argument('--seed', type=int, default=42)
        cmd.add_argument('--max-new-tokens', type=int, default=512)
        cmd.add_argument('--position-chunk-size', type=int, default=8)
    summary = sub.add_parser('summarize')
    summary.add_argument('--run-dir', type=Path, required=True)
    summary.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'summarize':
        summarize(args.run_dir, args.output_dir)
    else:
        if args.max_new_tokens < 1 or args.position_chunk_size < 1:
            parser.error('Token budget and chunk size must be positive')
        if args.command == 'prepare':
            prepare(args)
        else:
            run(args)


if __name__ == '__main__':
    main()
