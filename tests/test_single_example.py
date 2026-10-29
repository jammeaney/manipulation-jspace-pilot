"""Single-example CPU tests use random tiny weights, not pretrained GPU results."""
import contextlib
import copy
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import jlens
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
import torch
from transformers import PreTrainedTokenizerFast, Qwen3_5TextConfig, Qwen3_5ForCausalLM

from scripts import single_example as single


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        vocab = {token: i for i, token in enumerate([
            '[UNK]', '<|im_start|>', '<|im_end|>', '<think>', '</think>',
            'system', 'user', 'assistant', 'You', 'are', 'a', 'helpful',
            '.', 'What', 'is', '2', '+', '?', 'reasoning', 'answer'])}
        vocab.update({f'token{i}': i for i in range(20, 64)})
        backend = Tokenizer(WordLevel(vocab, unk_token='[UNK]'))
        backend.pre_tokenizer = Whitespace()
        cls.tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=backend, unk_token='[UNK]', eos_token='<|im_end|>',
            additional_special_tokens=['<|im_start|>', '<|im_end|>'])
        cls.tokenizer.add_tokens(['<think>', '</think>'])
        cls.tokenizer.chat_template = (
            "{% for m in messages %}{{ '<|im_start|>' + m.role + '\\n' + m.content + '<|im_end|>\\n' }}{% endfor %}"
            "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n<think>\\n' }}{% endif %}")
        # Actual newlines in the Jinja literals, not literal backslash-n output.
        cls.tokenizer.chat_template = cls.tokenizer.chat_template.replace('\\n', '\n')
        torch.manual_seed(7)
        config = Qwen3_5TextConfig(
            vocab_size=64, hidden_size=64, intermediate_size=128,
            num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
            head_dim=32, linear_num_key_heads=2, linear_num_value_heads=2,
            linear_key_head_dim=16, linear_value_head_dim=16,
            layer_types=['linear_attention', 'full_attention'], eos_token_id=2,
            rope_parameters={'rope_type': 'default', 'rope_theta': 10000.,
                             'partial_rotary_factor': .25, 'mrope_section': [2, 1, 1]})
        cls.hf = Qwen3_5ForCausalLM(config).eval()
        cls.model = jlens.from_hf(cls.hf, cls.tokenizer, force_bos=False)
        cls.lens = jlens.JacobianLens(
            {i: torch.eye(64) + .01 * torch.randn(64, 64) for i in range(2)},
            n_prompts=1, d_model=64)
        cls.messages, cls.prompt, cls.user_positions = single.prepare_prompt(
            cls.tokenizer, 'You are a helpful assistant.', 'What is 2 + 2?')

    def transcript(self, generated=None):
        if generated is None:
            generated, lps, _ = single.generate_one(self.hf, self.prompt, 5, .4, 42, [])
        else:
            lps = [-1.] * len(generated)
        full = self.prompt + generated
        return {'prompt_ids': self.prompt, 'generated_ids': generated, 'full_ids': full,
                'prompt_length': len(self.prompt), 'generated_length': len(generated),
                'attention_mask': [1] * len(full), 'ids_sha256': single.ids_hash(full),
                'raw_sampled_logprobs': lps,
                'tokens': single.token_records(self.tokenizer, self.prompt, generated, lps),
                'generated_text': single.decode(self.tokenizer, generated),
                'display_text': single.decode(self.tokenizer, generated, True),
                'full_text': single.decode(self.tokenizer, full),
                'spans': single.find_spans(self.prompt, generated, self.tokenizer,
                                            self.user_positions, [2])}

    def test_cached_generation_raw_probabilities_and_replay(self):
        data = self.transcript()
        generated = data['generated_ids']
        # Independent full forward: check sampled raw probability and next-token alignment.
        with torch.inference_mode():
            logits = self.hf(input_ids=torch.tensor([data['full_ids']]), use_cache=False).logits[0]
            raw = logits[len(self.prompt) - 1].float()
        expected = raw.log_softmax(-1)[generated[0]].item()
        scaled = (raw / .4).log_softmax(-1)[generated[0]].item()
        self.assertAlmostEqual(data['raw_sampled_logprobs'][0], expected, places=5)
        self.assertGreater(abs(expected - scaled), 1e-4)
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / 'lens.jsonl'
            result = single.replay_one(self.model, self.lens, self.tokenizer, data,
                                       path, 2, .5, data['full_ids'])
            rows = [json.loads(line) for line in path.read_text().splitlines()]
        self.assertTrue(result['token_identity'])
        self.assertLess(result['diagnostics']['max_abs_error_nats'], 1e-4)
        self.assertEqual(result['per_generated_token'][0]['prediction_position'], len(self.prompt) - 1)
        self.assertEqual(len(rows), len(data['full_ids']) * 2)
        self.assertEqual(len(rows[0]['top_tokens']), 25)
        for layer in [0, 1]:
            self.assertTrue(any(row['layer'] == layer and 'response_start' in row['spans'] for row in rows))
        self.assertTrue(all(t['logprob'] <= 0 for row in rows for t in row['top_tokens']))

    def test_non_argmax_sample_records_its_own_raw_probability(self):
        with torch.inference_mode():
            raw = self.hf(input_ids=torch.tensor([self.prompt]),
                          use_cache=False, logits_to_keep=1).logits[0, -1].float()
        # multinomial returns an index within the sorted top-k candidates.
        selected = raw.topk(20).indices[1].item()
        with patch.object(single.torch, 'multinomial', return_value=torch.tensor([1])):
            ids, lps, _ = single.generate_one(self.hf, self.prompt, 1, .7, 42, [])
        self.assertEqual(ids, [selected])
        self.assertAlmostEqual(lps[0], raw.log_softmax(-1)[selected].item(), places=5)
        self.assertNotAlmostEqual(lps[0], raw.log_softmax(-1).max().item(), places=5)

    def test_reproducible_sampling_and_explicit_eos(self):
        a = single.generate_one(self.hf, self.prompt, 3, .7, 10, [])
        b = single.generate_one(self.hf, self.prompt, 3, .7, 10, [])
        self.assertEqual(a, b)
        c = single.generate_one(self.hf, self.prompt, 3, .7, 10, [a[0][0]])
        self.assertEqual(c[0], [a[0][0]])
        self.assertEqual(c[2], 'eos')

    def test_valid_spans_and_invalid_thinking_cases(self):
        valid = self.transcript([18, 4, 19, 2])['spans']
        self.assertTrue(valid['span_analysis_eligible'])
        self.assertEqual(valid['cot'], [len(self.prompt), len(self.prompt) + 1])
        self.assertEqual(valid['answer'], [len(self.prompt) + 2, len(self.prompt) + 3])
        for generated, flag in [([18], 'missing_closing_delimiter'),
                                ([4, 19], 'empty_reasoning'),
                                ([18, 4, 2], 'empty_final_answer'),
                                ([18, 4, 4], 'extra_closing_delimiter'),
                                ([3, 18, 4], 'missing_or_extra_opening_delimiter')]:
            spans = self.transcript(generated)['spans']
            self.assertIn(flag, spans['flags'])
            self.assertFalse(spans['span_analysis_eligible'])
            self.assertIsNone(spans['cot'])
            self.assertIsNone(spans['answer'])
        self.assertTrue(self.user_positions)
        self.assertTrue(all(i < len(self.prompt) for i in self.user_positions))

    def test_tampering_is_rejected(self):
        data = self.transcript([18, 4, 19])
        for field in ['full_ids', 'generated_ids', 'tokens', 'generated_text', 'attention_mask']:
            changed = copy.deepcopy(data)
            if field == 'tokens':
                changed[field][0]['token_string'] = 'changed'
            elif field == 'generated_text':
                changed[field] += 'changed'
            else:
                changed[field][0] += 1
            with self.assertRaises(ValueError, msg=field):
                single.validate_transcript(changed, self.tokenizer, data['full_ids'])
        changed = self.transcript([18, 4, 19])
        with self.assertRaisesRegex(ValueError, 'original generation'):
            single.validate_transcript(changed, self.tokenizer, changed['full_ids'] + [1])

    def test_diagnostics_do_not_fail_on_numerical_disagreement(self):
        result = single.numerical_diagnostics([-1.] * 4, [-1., -1., -2., -2.], .5)
        self.assertEqual(result['large_error_generated_indices'], [2, 3])
        self.assertEqual(result['max_abs_error_nats'], 1.)
        self.assertEqual(result['median_abs_error_nats'], .5)
        self.assertEqual(result['possible_step_change_generated_indices'], [2])

    def test_main_writes_reloadable_outputs_and_refuses_overwrite(self):
        with TemporaryDirectory() as tmp:
            target = Path(tmp) / 'run'
            argv = ['single_example', '--device', 'cpu', '--max-new-tokens', '3',
                    '--position-chunk-size', '2', '--output-dir', str(target)]
            with patch('sys.argv', argv), \
                 patch.object(single.AutoTokenizer, 'from_pretrained', return_value=self.tokenizer), \
                 patch.object(single.AutoModelForCausalLM, 'from_pretrained', return_value=self.hf), \
                 patch.object(single.jlens.JacobianLens, 'from_pretrained', return_value=self.lens), \
                 contextlib.redirect_stdout(io.StringIO()):
                single.main()
            manifest = json.loads((target / 'manifest.json').read_text())
            self.assertEqual(manifest['status'], 'complete')
            self.assertIsNone(manifest['peak_gpu_allocated_bytes'])
            self.assertEqual(manifest['provenance']['model_revision'], single.MODEL_REVISION)
            transcript = json.loads((target / 'transcript.json').read_text())
            single.validate_transcript(transcript, self.tokenizer)
            self.assertTrue(json.loads((target / 'replay_validation.json').read_text())['token_identity'])
            with patch('sys.argv', argv), self.assertRaises(FileExistsError):
                single.main()


if __name__ == '__main__':
    unittest.main()
