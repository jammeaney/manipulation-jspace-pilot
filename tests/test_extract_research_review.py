"""CPU-only saved-artifact fixtures. Tokenizer loading is mocked; no downloads."""
import contextlib
import csv
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
from zipfile import ZipFile

from scripts import extract_research_review as export


class ByteTokenizer:
    pieces = {1: b'<think>', 2: b'\n', 3: b'\xe2', 4: b'\x82', 5: b'\xac',
              6: b'</think>', 7: b'OK', 8: b'<|im_end|>', 9: b'P'}

    def decode(self, ids, **kwargs):
        if kwargs != export.POLICY:
            raise AssertionError('Special tokens and whitespace must be preserved')
        return b''.join(self.pieces[i] for i in ids).decode('utf-8', errors='replace')


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False), encoding='utf-8')


def save_metrics(run, rows):
    export.write_csv(run / 'summaries/samples.csv', rows, list(rows[0]))


def rewrite_metrics(run, change):
    path = run / 'summaries/samples.csv'
    with path.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    change(rows)
    path.unlink()
    save_metrics(run, rows)


def make_run(root, stop='eos', valid=True):
    run = root / 'outputs/paired-008'
    (run / 'summaries').mkdir(parents=True)
    tokenizer = ByteTokenizer()
    samples, metrics = [], []
    for condition in ('neutral', 'covert_persuasion'):
        sample = f'town_{condition}_01'
        samples.append(dict(id=sample, scenario_id='town', assigned_condition=condition))
        folder = run / 'samples' / sample
        folder.mkdir(parents=True)
        prompt = [9, 1, 2]
        generated = [3, 4, 5, 6, 7] if valid else [3, 4, 5, 7]
        if stop == 'eos':
            generated += [8]
        full = prompt + generated
        transcript = dict(prompt_ids=prompt, generated_ids=generated, full_ids=full,
            ids_sha256=export.sha(json.dumps(full, separators=(',', ':')).encode()),
            prompt_length=len(prompt), generated_length=len(generated), attention_mask=[1]*len(full),
            tokens=[dict(id=i, position=j, decoded_piece=tokenizer.decode([i], **export.POLICY))
                    for j, i in enumerate(full)], stop_reason=stop, decode_policy=export.POLICY,
            generated_text=tokenizer.decode(generated, **export.POLICY),
            full_text=tokenizer.decode(full, **export.POLICY),
            spans=dict(coordinate_system='full-sequence, zero-based, end-exclusive',
                       cot=[2, 6] if valid else None, answer=[7, 8] if valid else None,
                       span_analysis_eligible=valid, flags=[] if valid else ['missing_closing_delimiter']))
        write_json(folder / 'transcript.json', transcript)
        write_json(folder / 'manifest.json', dict(provenance=dict(model='Qwen/Qwen3.5-4B',
            model_revision='a'*40, tokenizer_revision='a'*40)))
        row = dict(id=sample, scenario_id='town', assigned_condition=condition, target_option='A',
            generated_tokens=str(len(generated)), cot_tokens='4' if valid else '',
            answer_tokens='1' if valid else '', stop_reason=stop, thinking_span_valid=str(valid),
            thinking_generation_completed=str(stop == 'eos' and valid))
        for span in export.SPANS:
            for concept in export.CONCEPTS:
                row[f'{concept}_{span}_workspace_score'] = '1' if condition == 'neutral' else '2'
        metrics.append(row)
    save_metrics(run, metrics)
    for name in export.JLENS_FILES[1:]:
        (run / 'summaries' / name).write_text('{}\n' if name.endswith('.json') else 'fixture\n')
    write_json(run / 'experiment.json', dict(samples=samples))
    return run


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.run = make_run(self.root)
        self.zip_dir = self.root / 'exports'
        self.real_loader = export.load_tokenizer
        self.tokenizer_patch = patch.object(export, 'load_tokenizer', return_value=ByteTokenizer())
        self.loader = self.tokenizer_patch.start()
        self.addCleanup(self.tokenizer_patch.stop)

    def perform(self, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return export.export_run(self.run, self.zip_dir, **kwargs)

    def test_unicode_span_decoding_pins_special_tokens_and_reproducible_archives(self):
        before = {p: p.read_bytes() for p in self.run.rglob('*') if p.is_file()}
        dest, zip1, zip2 = self.perform()
        self.loader.assert_called_once_with('Qwen/Qwen3.5-4B', 'a'*40)
        self.assertEqual((dest/'town/neutral_reasoning.txt').read_text(), '\n€')
        self.assertEqual((dest/'town/neutral_answer.txt').read_text(), 'OK')
        # Individually decoded byte fragments really are lossy; the exporter doesn't join them.
        transcript = json.loads((self.run/'samples/town_neutral_01/transcript.json').read_text())
        self.assertNotEqual(''.join(t['decoded_piece'] for t in transcript['tokens'][2:6]), '\n€')
        self.assertFalse((dest/'town/neutral_incomplete_generation.txt').exists())
        with ZipFile(zip1) as archive:
            self.assertIn('research-review/export_metadata.json', archive.namelist())
            self.assertIsNone(archive.testzip())
        with ZipFile(zip2) as archive:
            self.assertEqual(set(archive.namelist()), set(export.JLENS_FILES))
        originals = zip1.read_bytes(), zip2.read_bytes()
        self.perform(overwrite=True)
        self.assertEqual((zip1.read_bytes(), zip2.read_bytes()), originals)
        self.assertTrue(all(p.read_bytes() == data for p, data in before.items()))

    def test_missing_and_unexpected_transcripts_listed_together(self):
        folder = self.run/'samples/town_neutral_01'
        shutil.copytree(folder, self.run/'samples/unexpected_01')
        (folder/'transcript.json').unlink()
        with self.assertRaisesRegex(ValueError, 'missing transcripts:.*town_neutral_01.*unexpected transcripts:.*unexpected_01'):
            self.perform()
        self.loader.assert_not_called()
        self.assertFalse((self.run/'research-review').exists())

    def test_duplicate_sample_ids_and_scenario_condition_combinations(self):
        for kind in ('id', 'combination'):
            with self.subTest(kind=kind):
                data = (self.run/'summaries/samples.csv').read_bytes()
                if kind == 'id':
                    rewrite_metrics(self.run, lambda rows: rows.append(rows[0]))
                    message = 'Duplicate sample ID'
                else:
                    rewrite_metrics(self.run, lambda rows: rows[1].update(assigned_condition='neutral'))
                    message = 'Duplicate scenario-condition'
                with self.assertRaisesRegex(ValueError, message):
                    self.perform()
                (self.run/'summaries/samples.csv').write_bytes(data)
        self.assertFalse(self.zip_dir.exists())

    def test_existing_directory_or_either_zip_rejected_without_overwrite(self):
        self.zip_dir.mkdir()
        targets = [self.run/'research-review', self.zip_dir/'paired-008-research-review.zip',
                   self.zip_dir/'paired-008-jlens-analysis.zip']
        for index, target in enumerate(targets):
            with self.subTest(target=target):
                if index == 0:
                    target.mkdir(); (target/'stale.txt').write_text('keep')
                else:
                    target.write_bytes(b'keep')
                with self.assertRaises(FileExistsError):
                    self.perform()
                if index == 0:
                    self.assertEqual((target/'stale.txt').read_text(), 'keep')
                    shutil.rmtree(target)
                else:
                    self.assertEqual(target.read_bytes(), b'keep'); target.unlink()

    def test_truncation_with_valid_spans_preserves_full_generation(self):
        shutil.rmtree(self.run); self.run = make_run(self.root, stop='max_new_tokens', valid=True)
        dest, _, _ = self.perform()
        detail = json.loads((dest/'export_metadata.json').read_text())['samples'][0]
        self.assertTrue(detail['truncated'])
        self.assertTrue(detail['span_valid'])
        self.assertFalse(detail['thinking_generation_completed'])
        self.assertEqual((dest/'town/neutral_reasoning.txt').read_text(), '\n€')
        self.assertEqual((dest/'town/neutral_incomplete_generation.txt').read_text(), '€</think>OK')

    def test_invalid_spans_distinguished_from_truncation_and_eos_preserved(self):
        shutil.rmtree(self.run); self.run = make_run(self.root, stop='eos', valid=False)
        dest, _, _ = self.perform()
        detail = json.loads((dest/'export_metadata.json').read_text())['samples'][0]
        self.assertFalse(detail['truncated'])
        self.assertFalse(detail['span_valid'])
        self.assertEqual(detail['span_flags'], ['missing_closing_delimiter'])
        self.assertEqual((dest/'town/neutral_incomplete_generation.txt').read_text(), '€OK<|im_end|>')

    def test_usable_coordinates_preserved_even_when_global_span_eligibility_is_false(self):
        for path in (self.run / 'samples').glob('*/transcript.json'):
            data = json.loads(path.read_text())
            data['spans'].update(span_analysis_eligible=False, flags=['extra_closing_delimiter'])
            write_json(path, data)
        rewrite_metrics(self.run, lambda rows: [r.update(thinking_span_valid='False',
            thinking_generation_completed='False') for r in rows])
        dest, _, _ = self.perform()
        self.assertEqual((dest / 'town/neutral_reasoning.txt').read_text(), '\n€')
        self.assertEqual((dest / 'town/neutral_answer.txt').read_text(), 'OK')
        detail = json.loads((dest/'export_metadata.json').read_text())['samples'][0]
        self.assertFalse(detail['span_valid'])
        self.assertTrue(all(detail['span_available'].values()))
        self.assertTrue((dest / 'town/neutral_incomplete_generation.txt').exists())

    def test_bad_span_or_revision_fails_before_output_creation(self):
        path = self.run / 'samples/town_neutral_01/transcript.json'
        original = path.read_bytes()
        data = json.loads(original)
        data['spans']['cot'] = [-1, 100]
        write_json(path, data)
        with self.assertRaisesRegex(ValueError, 'Invalid cot span'):
            self.perform()
        self.assertFalse((self.run / 'research-review').exists())
        self.assertFalse(self.zip_dir.exists())
        path.write_bytes(original)
        path = path.with_name('manifest.json')
        data = json.loads(path.read_text())
        data['provenance']['tokenizer_revision'] = 'main'
        write_json(path, data)
        with self.assertRaisesRegex(ValueError, 'immutable model/tokenizer'):
            self.perform()
        self.assertFalse(self.zip_dir.exists())

    def test_overwrite_replaces_stale_tree_instead_of_reusing_it(self):
        dest, _, _ = self.perform()
        (dest/'stale.txt').write_text('remove')
        (dest/'town/neutral_incomplete_generation.txt').write_text('stale')
        self.perform(overwrite=True)
        self.assertFalse((dest/'stale.txt').exists())
        self.assertFalse((dest/'town/neutral_incomplete_generation.txt').exists())

    def assert_previous_intact(self, before):
        self.assertEqual({p: p.read_bytes() for p in before}, before)
        self.assertFalse(list(self.run.glob('.research-review-*')))
        self.assertFalse(list(self.zip_dir.glob('.research-zips-*')))

    def saved_exports(self):
        dest, zip1, zip2 = self.perform()
        return {p: p.read_bytes() for p in [zip1, zip2, *dest.rglob('*')] if p.is_file()}

    def test_overwrite_validation_or_tokenizer_failure_preserves_previous(self):
        before = self.saved_exports()
        self.loader.side_effect = RuntimeError('Cannot load required tokenizer')
        with self.assertRaisesRegex(RuntimeError, 'Cannot load required tokenizer'):
            self.perform(overwrite=True)
        self.assert_previous_intact(before)
        self.loader.side_effect = None
        rewrite_metrics(self.run, lambda rows: rows[0].update(thinking_generation_completed='False'))
        with self.assertRaisesRegex(ValueError, 'completion/span metadata'):
            self.perform(overwrite=True)
        self.assert_previous_intact(before)

    def test_zip_failure_removes_partial_staging_and_preserves_previous_exports(self):
        before = self.saved_exports()
        original = export.write_zip
        def fail_second(path, files):
            if path.name.endswith('jlens-analysis.zip'):
                path.write_bytes(b'partial ZIP')
                raise OSError('injected ZIP failure')
            original(path, files)
        with patch.object(export, 'write_zip', side_effect=fail_second):
            with self.assertRaisesRegex(OSError, 'injected ZIP failure'):
                self.perform(overwrite=True)
        self.assert_previous_intact(before)

    def test_publication_failure_rolls_back_all_previous_exports(self):
        before = self.saved_exports()
        original = export.os.replace
        def fail_publication(source, target):
            if Path(source).parent.name.startswith('.research-zips-') and Path(source).name.endswith('jlens-analysis.zip'):
                raise OSError('injected publication failure')
            original(source, target)
        with patch.object(export.os, 'replace', side_effect=fail_publication):
            with self.assertRaisesRegex(OSError, 'injected publication failure'):
                self.perform(overwrite=True)
        self.assert_previous_intact(before)
        self.assertFalse(list(self.run.glob('.*.backup-*')))
        self.assertFalse(list(self.zip_dir.glob('.*.backup-*')))

    def test_cli_run_name_and_zip_names_preserved(self):
        with patch('sys.argv', ['extract_research_review.py', 'paired-008', '--outputs-dir',
                str(self.root/'outputs'), '--zip-dir', str(self.zip_dir)]), \
             contextlib.redirect_stdout(io.StringIO()):
            export.main()
        self.assertTrue((self.zip_dir/'paired-008-research-review.zip').exists())
        self.assertTrue((self.zip_dir/'paired-008-jlens-analysis.zip').exists())

    def test_tokenizer_loading_error_names_required_pin(self):
        # Exercise the real helper, without any network request.
        with patch('transformers.AutoTokenizer.from_pretrained', side_effect=OSError('offline')):
            with self.assertRaisesRegex(RuntimeError, 'Qwen/Qwen3.5-4B@'+'a'*40):
                self.real_loader('Qwen/Qwen3.5-4B', 'a'*40)


if __name__ == '__main__':
    unittest.main()
