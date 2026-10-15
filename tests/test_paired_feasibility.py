"""CPU tests only: no pretrained weights, no persuasive GPU generations."""
import contextlib
import copy
import csv
import re
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from scripts import paired_feasibility as paired
from scripts import single_example as single
import test_single_example as existing


def args_for(path):
    return SimpleNamespace(output_dir=path, scenarios=paired.DATA / 'scenarios.json',
        concepts=paired.DATA / 'concepts.json', seed=42, max_new_tokens=8,
        position_chunk_size=2, device='cpu', dtype='float32')


class PairedTests(unittest.TestCase):
    def test_exact_pair_background_and_user_identity(self):
        config = json.loads((paired.DATA / 'scenarios.json').read_text())
        rows = paired.build_samples(config)
        self.assertEqual(len(rows), 10)
        self.assertEqual(len({r['id'] for r in rows}), 10)
        for offset in range(0, 10, 2):
            a, b = rows[offset:offset + 2]
            self.assertEqual({a['condition'], b['condition']}, set(paired.CONDITIONS))
            for key in ['factual_background', 'question', 'user_prompt', 'common_user_sha256', 'target_statement']:
                self.assertEqual(a[key], b[key])
            self.assertNotEqual(a['condition_instruction'], b['condition_instruction'])
            self.assertEqual(a['system_prompt'].removesuffix(a['condition_instruction']),
                             b['system_prompt'].removesuffix(b['condition_instruction']))
            self.assertIsNone(a['behavioural_outcome'])
            self.assertEqual(a['behavioural_outcome_status'], 'not_evaluated')
        self.assertEqual(rows[0]['condition'], 'neutral')
        self.assertEqual(rows[2]['condition'], 'covert_persuasion')

    def test_model_facing_text_has_no_eval_probe_or_reasoning_length_instruction(self):
        config = json.loads((paired.DATA / 'scenarios.json').read_text())
        self.assertTrue(config['fictional'])
        concepts = json.loads((paired.DATA / 'concepts.json').read_text())
        probes = concepts['groups']['eval_awareness']
        for row in paired.build_samples(config):
            facing = (row['system_prompt'] + '\n' + row['user_prompt']).lower()
            for word in probes:
                self.assertNotIn(word, facing, (row['id'], word))
            self.assertNotIn('reasoning', facing)
            self.assertNotIn('chain of thought', facing)
            self.assertIn('final answer under 100 words', facing)
            self.assertNotIn('credits', facing)
            self.assertIn('£', facing)
            self.assertEqual(facing.count('residents will vote in a local referendum to choose between option a and option b.'), 1)

    def test_flood_tradeoff_and_frozen_concept_file(self):
        config = json.loads((paired.DATA / 'scenarios.json').read_text())
        flood = next(c['background'] for c in config['scenarios'] if c['id'] == 'brindleford_flood')
        a = flood.split('Option A upgrades', 1)[1].split('Option B creates', 1)[0]
        b = flood.split('Option B creates', 1)[1]
        protection_a = int(re.search(r'(\d+) fewer properties', a)[1])
        protection_b = int(re.search(r'(\d+) fewer properties', b)[1])
        maintenance_a = int(re.search(r'annual maintenance of £([\d,]+)', a)[1].replace(',', ''))
        maintenance_b = int(re.search(r'annual maintenance of £([\d,]+)', b)[1].replace(',', ''))
        self.assertGreater(protection_a, protection_b)
        self.assertGreater(maintenance_a, maintenance_b)
        self.assertIn('in one year', a)
        self.assertIn('in two years', b)
        # Frozen pre-GPU concepts and IDs must remain byte-for-byte unchanged.
        self.assertEqual(paired.digest(paired.DATA / 'concepts.json'),
                         'deb795fd3fc6a6276f00ebc5a54208dfb648174a70f7a7225f75a87a02179d04')

    def test_prompt_audit_matches_current_inputs_and_pairs(self):
        audit = json.loads((paired.DATA / 'prompt-audit.json').read_text())
        self.assertEqual(audit['scenarios_sha256'], paired.digest(paired.DATA / 'scenarios.json'))
        self.assertEqual(audit['frozen_concepts_sha256'], paired.digest(paired.DATA / 'concepts.json'))
        self.assertEqual(audit['tokenizer_revision'], single.MODEL_REVISION)
        self.assertEqual(audit['remaining_occurrences'], [])
        for a, b in zip(audit['samples'][::2], audit['samples'][1::2]):
            self.assertEqual(a['user_token_ids_sha256'], b['user_token_ids_sha256'])
            self.assertEqual(a['common_user_sha256'], b['common_user_sha256'])
            self.assertLessEqual(abs(a['total_tokens'] - b['total_tokens']), 4)

    def test_pairing_rejects_changed_common_context(self):
        rows = paired.build_samples(json.loads((paired.DATA / 'scenarios.json').read_text()))
        changed = copy.deepcopy(rows)
        changed[1]['factual_background'] += ' Different facts.'
        with self.assertRaisesRegex(ValueError, 'factual_background'):
            paired.validate_pairing(changed)
        changed = copy.deepcopy(rows)
        changed[1]['system_prompt'] = 'Different shared context. ' + changed[1]['system_prompt']
        with self.assertRaisesRegex(ValueError, 'Common system context'):
            paired.validate_pairing(changed)

    def test_prepare_offline_and_refuse_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = args_for(Path(tmp) / 'plan')
            result = paired.prepare(args)
            self.assertEqual(result['status'], 'prepared')
            self.assertEqual(result['prompt_revision'], 2)
            self.assertTrue(result['fictional'])
            self.assertEqual(result['scenario_source_sha256'], paired.digest(args.scenarios))
            frozen = paired.load_concepts(args.output_dir / 'concepts.json')
            self.assertEqual(len(frozen['groups']['control']), 100)
            with self.assertRaises(FileExistsError):
                paired.prepare(args)
            frozen['tokenizer_revision'] = 'wrong'
            (args.output_dir / 'bad.json').write_text(json.dumps(frozen))
            with self.assertRaises(ValueError):
                paired.load_concepts(args.output_dir / 'bad.json')

    def test_capture_uses_full_vocab_max_variants_and_equal_word_weights(self):
        logits = torch.arange(100, dtype=torch.float32)[None, :]
        groups = {'control': {'one': [0, 1], 'two': [2]},
                  'concealment': {'hidden': [3, 4], 'pretend': [5]}}
        normalizer = logits.logsumexp(-1)
        row = single.capture_concepts(logits, normalizer, groups)[0]
        # All selected words are below the top 25; no censored zero/missing scores.
        expected = (4 + 5) / 2 - (1 + 2) / 2
        self.assertAlmostEqual(row['groups']['concealment']['control_centered_score'], expected, places=5)
        self.assertAlmostEqual(row['groups']['concealment']['word_logprobs']['hidden'],
                               4 - normalizer.item(), places=5)

    def test_paired_difference_excludes_missing_spans_instead_of_zero_imputation(self):
        rows = []
        for scenario in ['one', 'two']:
            for condition, score in [('neutral', 1.), ('covert_persuasion', 3.)]:
                if scenario == 'two' and condition == 'covert_persuasion':
                    continue
                rows.append({'scenario_id': scenario, 'span': 'cot', 'concept_set': 'concealment',
                    'layer': 12, 'assigned_condition': condition, 'positions': 5 if condition == 'neutral' else 100,
                    'mean_control_centered_score': score, 'mean_logprob': score - 10})
        comparisons = paired.paired_comparisons(rows)
        result = next(r for r in comparisons if r['span'] == 'cot' and r['concept_set'] == 'concealment')
        missing = next(r for r in comparisons if r['span'] == 'answer')
        self.assertEqual(missing['n_pairs'], 0)
        self.assertIsNone(missing['mean_paired_score_delta'])
        self.assertEqual(result['n_pairs'], 1)
        self.assertEqual(result['mean_paired_score_delta'], 2.)
        self.assertEqual(result['included_scenarios'], 'one')

    def test_run_delegates_ten_samples_with_one_load_and_summarizes(self):
        concepts = paired.load_concepts(paired.DATA / 'concepts.json')['groups']
        vocabulary = {word: ids for words in concepts.values() for word, ids in words.items()}
        class FrozenTokenizer:
            def encode(self, word, **kwargs):
                ids = vocabulary[word.lstrip()]
                return [ids[-1] if word.startswith(' ') else ids[0]]

        def fake_example(args, hf, tokenizer, lens, concept_token_ids, print_tokens):
            args.output_dir.mkdir()
            covert = 'Present yourself' in args.system
            prompt, generated = [1, 2], [3, 4, 5]
            full = prompt + generated
            spans = {'response_start': 1, 'user_prompt_positions': [0], 'cot': [2, 3],
                     'answer': [3, 4], 'span_analysis_eligible': True, 'flags': []}
            data = {'messages': [{'role': 'system', 'content': args.system}, {'role': 'user', 'content': args.prompt}],
                    'prompt_ids': prompt, 'generated_ids': generated, 'full_ids': full,
                    'ids_sha256': single.ids_hash(full), 'generated_text': 'fixture',
                    'stop_reason': 'eos', 'spans': spans}
            manifest = {'status': 'complete', 'lens_layers': [12, 13],
                'concept_token_ids': concept_token_ids,
                'provenance': {'model_revision': single.MODEL_REVISION,
                               'tokenizer_revision': single.MODEL_REVISION, 'lens_revision': single.LENS_REVISION},
                'generation_settings': {'temperature': .7, 'seed': args.seed, 'max_new_tokens': args.max_new_tokens,
                                        'position_chunk_size': args.position_chunk_size, 'logprob_flag_threshold': .5},
                'generation_seconds': 0., 'replay_seconds': 0.}
            validation = {'replayed_ids_sha256': data['ids_sha256'], 'token_identity': True,
                'original_generation_checked': True, 'diagnostics': {'median_abs_error_nats': 0.,
                    'p99_abs_error_nats': 0., 'max_abs_error_nats': 0., 'large_error_generated_indices': []}}
            for name, value in [('manifest', manifest), ('transcript', data), ('replay_validation', validation)]:
                single.write_json(args.output_dir / f'{name}.json', value)
            with (args.output_dir / 'lens_readouts.jsonl').open('w') as out:
                for layer in [12, 13]:
                    for pos in range(len(full)):
                        raw = -3. if covert else -4.
                        out.write(json.dumps({'layer': layer, 'position': pos, 'input_token_id': full[pos],
                            'spans': single.position_roles(pos, spans),
                            'concept_readouts': {'control_mean_logprob': -5., 'groups': {
                                group: {'mean_logprob': raw, 'control_centered_score': raw + 5}
                                for group in paired.GROUPS}}}) + '\n')

        with tempfile.TemporaryDirectory() as tmp:
            args = args_for(Path(tmp) / 'run')
            with patch.object(paired.AutoTokenizer, 'from_pretrained', return_value=FrozenTokenizer()) as t, \
                 patch.object(paired.AutoModelForCausalLM, 'from_pretrained', return_value=object()) as h, \
                 patch.object(paired.jlens.JacobianLens, 'from_pretrained', return_value=object()) as l, \
                 patch.object(paired.single, 'run_example', side_effect=fake_example) as run_one, \
                 contextlib.redirect_stdout(io.StringIO()):
                paired.run(args)
            self.assertEqual([t.call_count, h.call_count, l.call_count, run_one.call_count], [1, 1, 1, 10])
            summary = json.loads((args.output_dir / 'summaries/summary.json').read_text())
            self.assertFalse(summary['behavioural_outcomes_evaluated'])
            self.assertEqual(len(summary['workspace_paired_comparisons']), 12)
            for comparison in summary['workspace_paired_comparisons']:
                self.assertEqual(comparison['n_pairs'], 5)
                self.assertEqual(comparison['mean_paired_score_delta'], 1.)
            with (args.output_dir / 'summaries/samples.csv').open() as stream:
                samples = list(csv.DictReader(stream))
            self.assertEqual(len(samples), 10)
            self.assertTrue(all(s['behavioural_outcome'] == '' for s in samples))
            # Duplicate readouts cannot silently change a layer/span average.
            sample_id = next((args.output_dir / 'samples').iterdir())
            readouts = sample_id / 'lens_readouts.jsonl'
            with readouts.open() as stream:
                duplicate = stream.readline()
            with readouts.open('a') as stream:
                stream.write(duplicate)
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                paired.layer_statistics(readouts,
                    json.loads((sample_id / 'transcript.json').read_text()),
                    json.loads((sample_id / 'manifest.json').read_text()))
            # An interrupted run may not silently produce a selectively complete summary.
            plan_path = args.output_dir / 'experiment.json'
            plan = json.loads(plan_path.read_text()); plan['status'] = 'running'
            single.write_json(plan_path, plan)
            with self.assertRaises(ValueError):
                paired.summarize(args.output_dir, Path(tmp) / 'incomplete')

    def test_real_tiny_model_replay_captures_concepts_without_changing_old_outputs(self):
        existing.PipelineTests.setUpClass()
        fixture = existing.PipelineTests()
        data = fixture.transcript([18, 4, 19])
        groups = {'control': {'spoon': [0]}, 'concealment': {'hidden': [1, 2]},
                  'persuasion': {'argue': [3]}, 'eval_awareness': {'test': [4]}}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'readouts.jsonl'
            single.replay_one(fixture.model, fixture.lens, fixture.tokenizer, data,
                              path, 2, .5, data['full_ids'], concept_token_ids=groups)
            row = json.loads(path.read_text().splitlines()[0])
            self.assertEqual(len(row['top_tokens']), 25)
            self.assertEqual(set(row['concept_readouts']['groups']), set(paired.GROUPS))
            self.assertIn('hidden', row['concept_readouts']['groups']['concealment']['word_logprobs'])

    def test_no_gpu_run_refuses_before_loading_or_writing(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = args_for(Path(tmp) / 'no-gpu'); args.device = 'cuda'
            with patch.object(torch.cuda, 'is_available', return_value=False), \
                 patch.object(paired.AutoModelForCausalLM, 'from_pretrained') as model:
                with self.assertRaisesRegex(ValueError, 'No CUDA'):
                    paired.run(args)
                model.assert_not_called()
                self.assertFalse(args.output_dir.exists())


if __name__ == '__main__':
    unittest.main()
