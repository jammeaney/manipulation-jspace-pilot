"""Saved-artifact reporting tests; no model or tokenizer required."""
import contextlib
import copy
import csv
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import replay_report as report


def fixture():
    ids = [10, 20, 21, 22]
    digest = hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest()
    transcript = dict(prompt_ids=[10], generated_ids=[20, 21, 22], full_ids=ids,
        ids_sha256=digest, raw_sampled_logprobs=[-2., -3., -4.],
        tokens=[dict(id=i, token_string=f'token{i}', decoded_piece=f'<{i}>') for i in ids])
    validation = dict(token_identity=True, replayed_ids_sha256=digest,
        diagnostics=dict(flag_threshold_nats=.5), per_generated_token=[
            dict(generated_index=j, token_id=i, prediction_position=j,
                 generation_logprob=-2.-j, replay_logprob=-2.-j+error,
                 absolute_error_nats=error)
            for j, (i, error) in enumerate(zip([20, 21, 22], [0., .5, .75]))])
    return transcript, validation


class ReplayReportTests(unittest.TestCase):
    def test_token_text_logprobs_and_strict_threshold(self):
        data, validation = fixture()
        rows = report.diagnostic_rows(data, validation)
        self.assertEqual(rows[2]['generated_index'], 2)
        self.assertEqual(rows[2]['token_text'], '<22>')
        self.assertEqual(rows[2]['generation_logprob'], -4.)
        self.assertEqual(rows[2]['replay_logprob'], -3.25)
        self.assertEqual(rows[2]['absolute_error_nats'], .75)
        self.assertEqual([r['flagged'] for r in rows], [False, False, True])

    def test_fail_loudly_on_incomplete_or_misaligned_diagnostics(self):
        data, validation = fixture()
        for kind in ('incomplete', 'index', 'token', 'probability', 'hash'):
            changed = copy.deepcopy(validation)
            if kind == 'incomplete':
                changed['per_generated_token'].pop()
            elif kind == 'index':
                changed['per_generated_token'][1]['generated_index'] = 0
            elif kind == 'token':
                changed['per_generated_token'][0]['token_id'] = 100
            elif kind == 'probability':
                changed['per_generated_token'][0]['generation_logprob'] = -20.
            else:
                changed['replayed_ids_sha256'] = 'wrong'
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                report.diagnostic_rows(data, changed)

    def test_cli_selection_csv_and_full_sequence_svg_without_inference(self):
        data, validation = fixture()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'transcript.json').write_text(json.dumps(data))
            (root / 'replay_validation.json').write_text(json.dumps(validation))
            output = root / 'report'
            with patch('sys.argv', ['replay_report.py', '--sample-dir', str(root),
                                   '--output-dir', str(output), '--indices', '1', '2']), \
                 contextlib.redirect_stdout(io.StringIO()):
                report.main()
            with (output / 'replay_tokens.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([r['generated_index'] for r in rows], ['1', '2'])
            self.assertEqual(rows[1]['token_text'], '<22>')
            svg = (output / 'replay_errors.svg').read_text()
            self.assertIn('0.5-nat threshold', svg)
            self.assertIn('fill="#c22"', svg)
            self.assertIn('Index 0:', svg)
            self.assertIn('&lt;22&gt;', svg)
            with self.assertRaises(ValueError):
                report.report(root, root / 'bad', [3])
            self.assertFalse((root / 'bad').exists())
            with self.assertRaises(FileExistsError):
                report.report(root, output)


if __name__ == '__main__':
    unittest.main()
