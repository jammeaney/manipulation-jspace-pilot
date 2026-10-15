"""CPU regression tests; these do not validate pretrained lens quality."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import jlens
import torch
from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

spec = importlib.util.spec_from_file_location(
    "sanity", Path(__file__).resolve().parents[1] / "scripts/sanity_check.py")
sanity = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sanity)


class Tokenizer:
    bos_token_id = None

    def __call__(self, text, **kwargs):
        # Fixed, nontrivial sequence for the tiny model; no checkpoint downloads.
        return SimpleNamespace(input_ids=torch.tensor([[1, 3, 5, 7, 9]]))

    def encode(self, word, **kwargs):
        return {"Italy": [2], " Italy": [3], "Rome": [4], " Rome": [4]}.get(word, [5, 6])

    def decode(self, ids):
        return str(ids)


class ReadoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(10)
        config = Qwen3_5TextConfig(
            vocab_size=32, hidden_size=64, intermediate_size=128,
            num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
            head_dim=32, linear_num_key_heads=2, linear_num_value_heads=2,
            linear_key_head_dim=16, linear_value_head_dim=16,
            layer_types=["linear_attention", "full_attention"],
            rope_parameters={"rope_type": "default", "rope_theta": 10000.,
                             "partial_rotary_factor": .25, "mrope_section": [2, 1, 1]})
        cls.model = jlens.from_hf(Qwen3_5ForCausalLM(config), Tokenizer(), force_bos=False)
        cls.lens = jlens.JacobianLens(
            {i: torch.eye(64) + .01 * torch.randn(64, 64) for i in range(2)},
            n_prompts=1, d_model=64)
        cls.args = SimpleNamespace(atol=1e-6, rtol=1e-6, top_k=5)

    def test_exact_ids_match_apply_on_real_hybrid_model(self):
        result = sanity.run_step0("exact-id", self.model, self.lens, Tokenizer(), self.args)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["input_ids"], [1, 3, 5, 7, 9])
        self.assertEqual(set(result["max_abs_error_by_layer"]), {"0", "1"})

    def test_walkthrough_uses_upstream_position_and_separate_completion(self):
        with patch.object(sanity, "exact_readout", wraps=sanity.exact_readout) as readout:
            result = sanity.run_step0("walkthrough", self.model, self.lens, Tokenizer(), self.args)
        self.assertEqual(result["prompt"], sanity.BOOT_PROMPT)
        self.assertEqual(result["position"], -2)
        self.assertEqual([call.kwargs["position"] for call in readout.call_args_list], [-2, -1])
        self.assertEqual(set(result["layers"]), {"0", "1"})
        self.assertIn("model_at_minus1_completion", result)

    def test_comparison_detects_perturbed_reference(self):
        original = self.lens.apply
        def changed(*args, **kwargs):
            logits, final, ids = original(*args, **kwargs)
            logits[0] += 1
            return logits, final, ids
        with patch.object(self.lens, "apply", side_effect=changed):
            with self.assertRaises(AssertionError):
                sanity.run_step0("exact-id", self.model, self.lens, Tokenizer(), self.args)

    def test_comparison_detects_changed_token_ids(self):
        original = self.lens.apply
        def changed(*args, **kwargs):
            logits, final, ids = original(*args, **kwargs)
            return logits, final, ids + 1
        with patch.object(self.lens, "apply", side_effect=changed):
            with self.assertRaisesRegex(ValueError, "token identity"):
                sanity.run_step0("exact-id", self.model, self.lens, Tokenizer(), self.args)

    def test_one_based_ranks_and_variants_without_filtering(self):
        logits = torch.tensor([100., 90., 3., 80., 70., 0., -1.])
        result = sanity.describe_logits(logits, Tokenizer(), ["Italy", "Rome", "absent"], 3)
        self.assertEqual(result["top_ids"], [0, 1, 3])
        self.assertEqual(result["concepts"]["Italy"]["best_rank"], 3)
        self.assertEqual(result["concepts"]["Rome"]["best_rank"], 4)
        self.assertIsNone(result["concepts"]["absent"]["best_rank"])

    def test_target_boundary_excludes_merged_space_target(self):
        def tokenizer(text, **kwargs):
            self.assertEqual(text, "Fact: the Atlantic")
            # Final token includes the trailing prompt space and target word.
            return {"input_ids": [1, 2, 3], "offset_mapping": [(0, 5), (5, 9), (9, 18)]}
        ids = sanity.target_prefix(tokenizer, "Fact: the ", "Atlantic")
        self.assertEqual(ids.tolist(), [[1, 2]])

    def test_multihop_aggregates_layers_and_fails_low_hit_rate(self):
        import contextlib
        import io
        import tempfile
        import json
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "items.json"
            path.write_text(json.dumps({"items": [
                {"name": f"item{i}", "prompt": "p", "target": "t", "intermediates": ["Italy"]}
                for i in range(10)]}))
            args = SimpleNamespace(multihop_data=path, top_k=2)
            for rank, expected in [(1, "PASS"), (20, "FAIL")]:
                logits = torch.arange(32, dtype=torch.float)
                logits[3] = 100 if rank == 1 else -10
                logits[2] = logits[3]
                with patch.object(sanity, "target_prefix", return_value=torch.tensor([[1, 2]])), \
                     patch.object(sanity, "exact_readout", return_value=({0: logits, 1: logits}, logits)), \
                     contextlib.redirect_stdout(io.StringIO()):
                    result = sanity.run_step0("multihop", self.model, self.lens, Tokenizer(), args)
                self.assertTrue(result["status"].startswith(expected))
                self.assertEqual(result["items"], 10)
                self.assertEqual(result["top10_hits"], 10 if rank == 1 else 0)


if __name__ == "__main__":
    unittest.main()
