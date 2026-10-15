"""Check the pinned artifacts; optionally run one pretrained-model lens forward.

Default mode downloads the tokenizer, config and lens, but no model weights.
The optional forward is a smoke test, not a lens-quality benchmark or replay test.
"""

import argparse
import json

import jlens
import torch
from jlens.hooks import ActivationRecorder
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen3.5-4B"
MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
LENS = "neuronpedia/jacobian-lens"
LENS_REVISION = "16a01f309fcec900fdcec3f4cd5b64f3d00e4d5a"
LENS_FILE = "qwen3.5-4b/jlens/Salesforce-wikitext/Qwen3.5-4B_jacobian_lens_n1000.pt"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-forward", action="store_true")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    args = parser.parse_args()
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
        "status": "artifact and dimensional checks passed; forward not yet run",
    }, indent=2))
    if not args.run_forward:
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
