"""Check the pinned artifacts; optionally run one pretrained-model lens forward.

Default mode downloads the tokenizer, config and lens, but no model weights.
Use --check walkthrough|multihop|exact-id|all for Section 3 validation.
The legacy --run-forward smoke test is preserved. No generation or replay.
"""

import argparse
from importlib.metadata import version
import json
import time
from pathlib import Path

import jlens
import torch
from jlens.hooks import ActivationRecorder
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen3.5-4B"
MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
LENS = "neuronpedia/jacobian-lens"
LENS_REVISION = "16a01f309fcec900fdcec3f4cd5b64f3d00e4d5a"
LENS_FILE = "qwen3.5-4b/jlens/Salesforce-wikitext/Qwen3.5-4B_jacobian_lens_n1000.pt"


BOOT_PROMPT = "Fact: The currency used in the country shaped like a boot is"


def token_variants(tokenizer, word):
    """Only single-token spellings; rank each separately without vocab filtering."""
    variants = {}
    for spelling in (word, " " + word):
        ids = tokenizer.encode(spelling, add_special_tokens=False)
        if len(ids) == 1:
            variants[spelling] = ids[0]
    return variants


def describe_logits(logits, tokenizer, words, top_k):
    logits = logits.float().cpu().reshape(-1)
    if not torch.isfinite(logits).all():
        raise ValueError("Non-finite logits")
    ids = logits.topk(min(top_k, logits.numel())).indices.tolist()
    ranks = {}
    for word in words:
        variants = token_variants(tokenizer, word)
        ranks[word] = {
            "variants": {s: {"id": i, "rank": 1 + int((logits > logits[i]).sum())}
                         for s, i in variants.items()},
            "best_rank": min((1 + int((logits > logits[i]).sum())
                              for i in variants.values()), default=None),
        }
    return {"top_ids": ids, "top_tokens": [tokenizer.decode([i]) for i in ids],
            "concepts": ranks}


@torch.inference_mode()
def exact_readout(model, lens, ids, layers, position=-1):
    """One position only: never allocate layer × sequence × vocabulary outputs."""
    ids = ids.to(model.input_device)
    final = model.n_layers - 1
    with ActivationRecorder(model.layers, at=sorted(set(layers) | {final})) as recorder:
        model.forward(ids)
    outputs = {}
    for layer in layers:
        residual = recorder.activations[layer][0, position].unsqueeze(0).float()
        outputs[layer] = model.unembed(lens.transport(residual, layer)).float().cpu()
    final_logits = model.unembed(
        recorder.activations[final][0, position].unsqueeze(0).float()
    ).float().cpu()
    return outputs, final_logits


def target_prefix(tokenizer, prompt, target):
    """Locate the target with offsets, including tokens that merge its leading space."""
    encoded = tokenizer(prompt + target, add_special_tokens=False,
                        return_offsets_mapping=True)
    boundary = next((i for i, (_, end) in enumerate(encoded["offset_mapping"])
                     if end > len(prompt)), None)
    if boundary is None or boundary == 0:
        raise ValueError("Cannot locate a nonempty prefix immediately before target")
    return torch.tensor([encoded["input_ids"][:boundary]], dtype=torch.long)


def run_step0(check, model, lens, tokenizer, args):
    layers = lens.source_layers
    if check == "walkthrough":
        ids = model.encode(BOOT_PROMPT)
        # Upstream explicitly uses -2, not the prompt's last token.
        outputs, final = exact_readout(model, lens, ids, layers, position=-2)
        reports = {str(l): describe_logits(v, tokenizer, ["Italy", "Rome", "euro"], args.top_k)
                   for l, v in outputs.items()}
        _, completion = exact_readout(model, lens, ids, [], position=-1)
        middle = {str(l): {w: reports[str(l)]["concepts"][w]["best_rank"]
                          for w in ("Italy", "Rome")} for l in layers if 12 <= l <= 29}
        return {"prompt": BOOT_PROMPT, "position": -2,
                "readout_token": tokenizer.decode([ids[0, -2].item()]),
                "upstream_example_layers": [8, 16, 24, 30], "layers": reports,
                "middle_layer_concept_ranks": middle,
                "model_at_minus2": describe_logits(final, tokenizer, ["euro"], args.top_k),
                "model_at_minus1_completion": describe_logits(completion, tokenizer, ["euro"], args.top_k),
                "status": "readouts computed; inspect concept ranks (no quality pass inferred)"}
    if check == "exact-id":
        reference, final_reference, ids = lens.apply(
            model, BOOT_PROMPT, layers=layers, positions=[-2])
        saved = model.encode(BOOT_PROMPT).detach().cpu().clone()
        if not torch.equal(ids.cpu(), saved):
            raise ValueError("Exact-ID comparison failed token identity")
        actual, final_actual = exact_readout(model, lens, saved, layers, position=-2)
        differences = {}
        for layer in layers:
            differences[str(layer)] = (actual[layer] - reference[layer]).abs().max().item()
            torch.testing.assert_close(actual[layer], reference[layer],
                                       atol=args.atol, rtol=args.rtol)
        torch.testing.assert_close(final_actual, final_reference,
                                   atol=args.atol, rtol=args.rtol)
        return {"status": "PASS", "position": -2, "token_identity": True,
                "input_ids": saved[0].tolist(), "atol": args.atol, "rtol": args.rtol,
                "max_abs_error_by_layer": differences,
                "model_max_abs_error": (final_actual - final_reference).abs().max().item()}
    data = json.loads(args.multihop_data.read_text())["items"]
    if len(data) < 10:
        raise ValueError("Step 0 requires at least ten multihop items")
    results = []
    hits = count = 0
    for item in data[:10]:
        ids = target_prefix(tokenizer, item["prompt"], item["target"])
        outputs, _ = exact_readout(model, lens, ids, layers)
        layer_reports = {str(l): describe_logits(v, tokenizer, item["intermediates"], args.top_k)
                         for l, v in outputs.items()}
        concepts = {}
        for word in item["intermediates"]:
            ranks = {l: report["concepts"][word]["best_rank"]
                     for l, report in layer_reports.items()}
            valid = {l: rank for l, rank in ranks.items() if rank is not None}
            if not valid:
                raise ValueError(f"{item['name']}: no single-token spelling for {word!r}")
            best = min(valid.values())
            concepts[word] = {"min_rank": best,
                              "best_layers": [int(l) for l, rank in valid.items() if rank == best],
                              "rank_by_layer": ranks}
            count += 1
            hits += best <= 10
        result = {"name": item["name"], "prefix_ids": ids[0].tolist(),
                  "readout_token": tokenizer.decode([ids[0, -1].item()]),
                  "intermediates": concepts, "layers": layer_reports}
        results.append(result)
        print(json.dumps({"check": "multihop-item", **result}), flush=True)
    return {"items": len(results), "intermediates": count, "top10_hits": hits,
            "pass_at_10": hits / count, "majority_in_top10": hits > count / 2,
            "status": "PASS" if hits > count / 2 else "FAIL: most intermediates not in top 10"}


def measured_step0(check, model, lens, tokenizer, args):
    gpu = model.input_device.type == "cuda"
    if gpu:
        torch.cuda.synchronize(model.input_device)
        torch.cuda.reset_peak_memory_stats(model.input_device)
    start = time.perf_counter()
    result = run_step0(check, model, lens, tokenizer, args)
    if gpu:
        torch.cuda.synchronize(model.input_device)
    print(json.dumps({"check": check, **result,
                      "runtime_seconds": time.perf_counter() - start,
                      "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(model.input_device) if gpu else None,
                      "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(model.input_device) if gpu else None,
                      "gpu_name": torch.cuda.get_device_name(model.input_device) if gpu else None}), flush=True)
    return not str(result["status"]).startswith("FAIL")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-forward", action="store_true")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--check", choices=["walkthrough", "multihop", "exact-id", "all"],
                        help="Run a Step 0 check (implies model loading); default remains artifact-only")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--multihop-data", type=Path,
                        default=Path(__file__).resolve().parents[1] / "data/evaluations/lens-eval-multihop.json")
    args = parser.parse_args()
    if args.top_k < 1 or args.atol < 0 or args.rtol < 0:
        parser.error("top-k must be positive and tolerances nonnegative")
    config = AutoConfig.from_pretrained(MODEL, revision=MODEL_REVISION)
    text_config = config.get_text_config()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REVISION)
    lens = jlens.JacobianLens.from_pretrained(
        LENS, filename=LENS_FILE, revision=LENS_REVISION
    )
    if lens.d_model != text_config.hidden_size:
        raise ValueError("Model and lens residual widths differ")
    if not lens.source_layers or any(
        not 0 <= layer < text_config.num_hidden_layers
        or matrix.shape != (lens.d_model, lens.d_model)
        or not torch.isfinite(matrix).all()
        for layer, matrix in lens.jacobians.items()
    ):
        raise ValueError("Invalid lens layers, shapes or matrix values")

    prompt_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": "What is 2 + 2?"}],
        tokenize=True, add_generation_prompt=True, enable_thinking=True,
        return_tensors="pt",
    )
    # Transformers versions may return a BatchEncoding instead of a tensor.
    if not isinstance(prompt_ids, torch.Tensor):
        prompt_ids = prompt_ids["input_ids"]
    rendered = tokenizer.decode(
        prompt_ids[0], skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
    if tokenizer.encode(rendered, add_special_tokens=False) != prompt_ids[0].tolist():
        raise ValueError("This prompt did not round-trip exactly")
    thinking_ids = {
        token: tokenizer.encode(token, add_special_tokens=False)
        for token in ("<think>", "</think>")
    }
    if any(len(ids) != 1 for ids in thinking_ids.values()):
        raise ValueError("Thinking delimiters are not single tokens")
    print(json.dumps({
        "model_revision": MODEL_REVISION,
        "lens_revision": LENS_REVISION,
        "lens_layers": lens.source_layers,
        "d_model": lens.d_model,
        "n_prompts": lens.n_prompts,
        "thinking_ids": thinking_ids,
        "prompt_ids": prompt_ids[0].tolist(),
        "rendered_prompt": rendered,
        "cuda_available": torch.cuda.is_available(),
        "package_versions": {name: version(name) for name in
                             ["torch", "torchvision", "transformers", "accelerate", "jlens"]},
        "status": "artifact and dimensional checks passed; forward not yet run",
    }, indent=2))
    if not args.run_forward and not args.check:
        return
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("No CUDA GPU available; no model forward was run")
    if args.device == "cuda" and not torch.cuda.is_bf16_supported():
        parser.error("This GPU does not support the requested bfloat16 forward")

    hf = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=MODEL_REVISION,
        dtype=torch.bfloat16 if args.device == "cuda" else torch.float32,
        device_map=args.device,
    )
    model = jlens.from_hf(hf, tokenizer, force_bos=False)
    if args.check:
        checks = ["walkthrough", "exact-id", "multihop"] if args.check == "all" else [args.check]
        passed = True
        for check in checks:
            passed = measured_step0(check, model, lens, tokenizer, args) and passed
        if not passed:
            raise SystemExit(1)
        return
    # Feed the saved IDs directly, without re-rendering or tokenizing text.
    saved_ids = prompt_ids.clone()
    input_ids = saved_ids.to(model.input_device)
    if not torch.equal(input_ids.cpu(), saved_ids):
        raise ValueError("Input IDs changed")
    layer = lens.source_layers[len(lens.source_layers) // 2]
    with torch.inference_mode():
        with ActivationRecorder(model.layers, at=[layer]) as recorder:
            model.forward(input_ids)
        residual = recorder.activations[layer][0, -1:].float()
        logits = model.unembed(lens.transport(residual, layer)).float()
        if not torch.isfinite(logits).all():
            raise ValueError("Non-finite readout")
        top_ids = logits[0].topk(10).indices.tolist()
    print(json.dumps({
        "status": "exact-ID forward and lens readout passed",
        "device": args.device, "layer": layer, "top_token_ids": top_ids,
        "top_tokens": [tokenizer.decode([i]) for i in top_ids],
    }, indent=2))


if __name__ == "__main__":
    main()
