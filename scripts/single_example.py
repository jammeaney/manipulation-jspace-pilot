"""Capture and replay one thinking response; no dataset, judges or analysis."""
import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import subprocess
import time

import jlens
from jlens.hooks import ActivationRecorder
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

if __package__:
    from .sanity_check import MODEL, MODEL_REVISION, LENS, LENS_REVISION, LENS_FILE
else:
    from sanity_check import MODEL, MODEL_REVISION, LENS, LENS_REVISION, LENS_FILE


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def ids_hash(ids):
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def decode(tokenizer, ids, skip_special_tokens=False):
    return tokenizer.decode(ids, skip_special_tokens=skip_special_tokens,
                            clean_up_tokenization_spaces=False)


def prepare_prompt(tokenizer, system, user):
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                       enable_thinking=True)
    if hasattr(ids, "input_ids"):
        ids = ids.input_ids
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                            enable_thinking=True)
    encoded = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
    if encoded["input_ids"] != ids:
        raise ValueError("Rendered prompt offsets do not match actual generation IDs")
    header = "<|im_start|>user\n"
    start = rendered.index(header) + len(header)
    if rendered[start:start + len(user)] != user:
        raise ValueError("Cannot locate the exact user content in the pinned chat template")
    user_positions = [i for i, (a, b) in enumerate(encoded["offset_mapping"])
                      if b > start and a < start + len(user)]
    if not user_positions:
        raise ValueError("No user-content tokens found")
    return messages, ids, user_positions


@torch.inference_mode()
def generate_one(hf, prompt_ids, max_new_tokens, temperature, seed, eos_ids):
    """Raw HF logits -> temperature sampling; retain only the sampled scalar lp.

    Batch size one, no padding, top_p=1, top_k=0, no penalties/processors.
    KV caching is explicit. No full-vocabulary logits history is retained.
    """
    hf.eval()
    device = hf.get_input_embeddings().weight.device
    generator = torch.Generator(device=device).manual_seed(seed)
    next_input = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    cache = None
    generated, logprobs = [], []
    for step in range(max_new_tokens):
        attention_mask = torch.ones((1, len(prompt_ids) + step), dtype=torch.long, device=device)
        output = hf(input_ids=next_input, attention_mask=attention_mask,
                    past_key_values=cache, use_cache=True, logits_to_keep=1)
        cache = output.past_key_values
        if cache is None:
            raise ValueError("Model did not return a generation cache")
        raw = output.logits[0, -1].float()
        if not torch.isfinite(raw).all():
            raise ValueError("Non-finite generation logits")
        sampled = torch.multinomial(torch.softmax(raw / temperature, dim=-1), 1,
                                    generator=generator)
        token = sampled.item()
        logprobs.append((raw[token] - raw.logsumexp(-1)).item())
        generated.append(token)
        if token in eos_ids:
            return generated, logprobs, "eos"
        next_input = sampled.reshape(1, 1)
    return generated, logprobs, "max_new_tokens"


def find_spans(prompt, generated, tokenizer, user_positions, eos_ids):
    full = prompt + generated
    opening = tokenizer.encode("<think>", add_special_tokens=False)
    closing = tokenizer.encode("</think>", add_special_tokens=False)
    if len(opening) != 1 or len(closing) != 1:
        raise ValueError("Thinking delimiters must be single tokens")
    open_id, close_id = opening[0], closing[0]
    opens = [i for i, token in enumerate(full) if token == open_id]
    closes = [len(prompt) + i for i, token in enumerate(generated) if token == close_id]
    prompt_opens = [i for i in opens if i < len(prompt)]
    flags = []
    if len(opens) != 1 or not prompt_opens:
        flags.append("missing_or_extra_opening_delimiter")
    if not closes:
        flags.append("missing_closing_delimiter")
    if full.count(close_id) > 1 or close_id in prompt:
        flags.append("extra_closing_delimiter")
    cot = answer = None
    if prompt_opens and closes:
        cot = [prompt_opens[-1] + 1, closes[0]]
        end = len(full) - (1 if generated and generated[-1] in eos_ids else 0)
        answer = [closes[0] + 1, end]
        if not decode(tokenizer, full[cot[0]:cot[1]]).strip():
            flags.append("empty_reasoning")
        if not decode(tokenizer, full[answer[0]:answer[1]]).strip():
            flags.append("empty_final_answer")
    valid = not flags
    return {"coordinate_system": "full-sequence, zero-based, end-exclusive",
            "response_start": len(prompt) - 1, "user_prompt_positions": user_positions,
            "thinking_open_positions": opens, "generated_thinking_close_positions": closes,
            "cot": cot if valid else None, "answer": answer if valid else None,
            "span_analysis_eligible": valid, "flags": flags}


def token_records(tokenizer, prompt, generated, logprobs):
    full = prompt + generated
    return [{"position": i, "id": token,
             "token_string": tokenizer.convert_ids_to_tokens(token),
             "decoded_piece": decode(tokenizer, [token]),
             "source": "prompt" if i < len(prompt) else "generated",
             "raw_sampled_logprob": None if i < len(prompt) else logprobs[i - len(prompt)]}
            for i, token in enumerate(full)]


def validate_transcript(data, tokenizer, original_ids=None):
    full = data["prompt_ids"] + data["generated_ids"]
    if (not data["prompt_ids"] or not data["generated_ids"]
            or data["prompt_length"] != len(data["prompt_ids"])
            or data["generated_length"] != len(data["generated_ids"])
            or data["attention_mask"] != [1] * len(full)):
        raise ValueError("Invalid unpadded lengths or attention mask")
    if full != data["full_ids"] or ids_hash(full) != data["ids_sha256"]:
        raise ValueError("Saved token sequence/hash mismatch")
    if original_ids is not None and full != original_ids:
        raise ValueError("Saved token IDs differ from original generation")
    if len(data["generated_ids"]) != len(data["raw_sampled_logprobs"]):
        raise ValueError("Sampled-token log-probability count mismatch")
    if data["tokens"] != token_records(tokenizer, data["prompt_ids"], data["generated_ids"],
                                        data["raw_sampled_logprobs"]):
        raise ValueError("Token strings or per-token records differ from saved IDs")
    if decode(tokenizer, data["generated_ids"]) != data["generated_text"]:
        raise ValueError("Generated transcript text mismatch")
    if decode(tokenizer, full) != data["full_text"]:
        raise ValueError("Full transcript text mismatch")
    if decode(tokenizer, data["generated_ids"], True) != data["display_text"]:
        raise ValueError("Display transcript text mismatch")
    return full


def position_roles(position, spans):
    roles = []
    if position == spans["response_start"]:
        roles.append("response_start")
    if position in spans["user_prompt_positions"]:
        roles.append("user_prompt")
    for name in ("cot", "answer"):
        span = spans[name]
        if span and span[0] <= position < span[1]:
            roles.append(name)
    return roles


def numerical_diagnostics(generation, replay, threshold):
    errors = (torch.tensor(generation, dtype=torch.float64) -
              torch.tensor(replay, dtype=torch.float64)).abs()
    if not len(errors) or not torch.isfinite(errors).all():
        raise ValueError("Empty or non-finite replay diagnostic")
    # A possible persistent jump is diagnostic only, never an identity failure.
    jumps = [i for i in range(1, len(errors))
             if errors[i] - errors[i - 1] > threshold
             and errors[i:].median() - errors[:i].median() > threshold]
    return {"median_abs_error_nats": errors.quantile(.5).item(),
            "p99_abs_error_nats": errors.quantile(.99).item(),
            "max_abs_error_nats": errors.max().item(), "flag_threshold_nats": threshold,
            "large_error_generated_indices": (errors > threshold).nonzero().flatten().tolist(),
            "possible_step_change_generated_indices": jumps}


@torch.inference_mode()
def replay_one(model, lens, tokenizer, data, readouts_path, chunk_size, threshold,
               original_ids=None):
    full = validate_transcript(data, tokenizer, original_ids)
    ids = torch.tensor([full], dtype=torch.long, device=model.input_device)
    if ids.cpu().tolist()[0] != full:
        raise ValueError("Replay tensor IDs differ from saved IDs")
    if lens.d_model != model.d_model or any(
        not 0 <= layer < model.n_layers for layer in lens.source_layers
    ):
        raise ValueError("Lens/model dimension or layer mismatch")
    final = model.n_layers - 1
    with ActivationRecorder(model.layers, at=sorted(set(lens.source_layers) | {final})) as recorder:
        model.forward(ids)  # Exactly one cache-free full-sequence replay, no tokenization.
    if ids.cpu().tolist()[0] != full:
        raise ValueError("Model mutated replay token IDs")
    count, prompt_len = len(data["generated_ids"]), len(data["prompt_ids"])
    replay_logprobs = []
    for start in range(0, count, chunk_size):
        stop = min(start + chunk_size, count)
        residual = recorder.activations[final][0, prompt_len + start - 1:prompt_len + stop - 1]
        logits = model.unembed(residual.float()).float()
        targets = ids[0, prompt_len + start:prompt_len + stop]
        lp = logits.gather(1, targets[:, None]).squeeze(1) - logits.logsumexp(-1)
        replay_logprobs.extend(lp.cpu().tolist())
    records = 0
    with readouts_path.open("x") as stream:
        for layer in lens.source_layers:
            for start in range(0, len(full), chunk_size):
                stop = min(start + chunk_size, len(full))
                residual = recorder.activations[layer][0, start:stop].float()
                logits = model.unembed(lens.transport(residual, layer)).float()
                if not torch.isfinite(logits).all():
                    raise ValueError("Non-finite lens readout")
                normalizer = logits.logsumexp(-1)
                values, indices = logits.topk(min(25, logits.shape[-1]), dim=-1)
                values = (values - normalizer[:, None]).cpu().tolist()
                indices = indices.cpu().tolist()
                for offset, (token_ids, logprobs) in enumerate(zip(indices, values)):
                    position = start + offset
                    stream.write(json.dumps({
                        "layer": layer, "workspace_band": 12 <= layer <= 29,
                        "position": position, "input_token_id": full[position],
                        "spans": position_roles(position, data["spans"]),
                        "top_tokens": [{"id": token, "token_string": tokenizer.convert_ids_to_tokens(token),
                                        "logprob": lp} for token, lp in zip(token_ids, logprobs)]
                    }, ensure_ascii=False, allow_nan=False) + "\n")
                    records += 1
    generation = data["raw_sampled_logprobs"]
    return {"token_identity": True, "original_generation_checked": original_ids is not None,
            "generated_text_identity": True, "full_text_identity": True,
            "replayed_ids_sha256": ids_hash(full), "replayed_token_count": len(full),
            "lens_readout_records": records,
            "diagnostics": numerical_diagnostics(generation, replay_logprobs, threshold),
            "per_generated_token": [{"generated_index": j, "token_id": data["generated_ids"][j],
                "prediction_position": prompt_len + j - 1, "generation_logprob": a,
                "replay_logprob": b, "absolute_error_nats": abs(a - b)}
                for j, (a, b) in enumerate(zip(generation, replay_logprobs))]}


def provenance(device, dtype):
    root = Path(__file__).resolve().parents[1]
    commit = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                            capture_output=True, text=True)
    dirty = subprocess.run(["git", "-C", str(root), "status", "--porcelain"],
                           capture_output=True, text=True)
    return {"model": MODEL, "model_revision": MODEL_REVISION,
            "tokenizer_revision": MODEL_REVISION, "lens": LENS,
            "lens_revision": LENS_REVISION, "lens_file": LENS_FILE,
            "jlens_revision": "581d398613e5602a5af361e1c34d3a92ea82ba8e",
            "packages": {p: version(p) for p in ["torch", "torchvision", "transformers", "jlens", "accelerate"]},
            "python": platform.python_version(), "platform": platform.platform(),
            "device": device, "dtype": dtype, "cuda_build": torch.version.cuda,
            "gpu": torch.cuda.get_device_name() if device == "cuda" else None,
            "git_commit": commit.stdout.strip() if commit.returncode == 0 else None,
            "git_dirty": bool(dirty.stdout) if dirty.returncode == 0 else None,
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory; never overwritten")
    parser.add_argument("--prompt", default="What is 2 + 2? Keep your reasoning brief and answer in one sentence.")
    parser.add_argument("--system", default="You are a helpful assistant.")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--dtype", choices=["bfloat16", "float32"], help="Default: bf16 CUDA, float32 CPU")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=.7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--position-chunk-size", type=int, default=8)
    parser.add_argument("--logprob-flag-threshold", type=float, default=.5)
    args = parser.parse_args()
    if args.max_new_tokens < 1 or args.position_chunk_size < 1 or not 0 < args.temperature < float("inf"):
        parser.error("Token counts/chunk size and finite temperature must be positive")
    if not 0 < args.logprob_flag_threshold < float("inf"):
        parser.error("Diagnostic threshold must be finite and positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("No CUDA GPU available; no pretrained model was loaded")
    dtype = args.dtype or ("bfloat16" if args.device == "cuda" else "float32")
    if args.device == "cuda" and dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
        parser.error("GPU lacks bf16 support; explicitly choose --dtype float32 if memory permits")
    user, system = args.prompt.strip(), args.system.strip()
    if not user or not system:
        parser.error("Prompt and system message must be nonempty")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    metadata = provenance(args.device, dtype)
    settings = {"max_new_tokens": args.max_new_tokens, "temperature": args.temperature,
                "seed": args.seed, "do_sample": True, "top_p": 1.0, "top_k": 0,
                "batch_size": 1, "enable_thinking": True, "use_cache": True,
                "logprob_basis": "raw model logits before temperature; natural logarithm",
                "position_chunk_size": args.position_chunk_size,
                "logprob_flag_threshold": args.logprob_flag_threshold,
                "penalties_and_other_logits_processors": "none"}
    manifest = {"schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "status": "initializing", "provenance": metadata, "generation_settings": settings}
    manifest_path = args.output_dir / "manifest.json"
    write_json(manifest_path, manifest)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REVISION)
    messages, prompt, user_positions = prepare_prompt(tokenizer, system, user)
    hf = AutoModelForCausalLM.from_pretrained(MODEL, revision=MODEL_REVISION,
                                            dtype=getattr(torch, dtype), device_map=args.device)
    eos_ids = sorted({tokenizer.eos_token_id, hf.config.get_text_config().eos_token_id} - {None})
    if not eos_ids:
        raise ValueError("No explicit stopping token IDs found")
    if len(prompt) + args.max_new_tokens > hf.config.get_text_config().max_position_embeddings:
        raise ValueError("Requested sequence exceeds model context limit")
    settings["eos_token_ids"] = eos_ids
    settings["pad_token_id"] = None
    metadata["attention_implementation"] = hf.config.get_text_config()._attn_implementation
    metadata["deterministic_algorithms_enabled"] = torch.are_deterministic_algorithms_enabled()
    metadata["cuda_matmul_allow_tf32"] = torch.backends.cuda.matmul.allow_tf32
    metadata["cudnn_allow_tf32"] = torch.backends.cudnn.allow_tf32
    sync(args.device)
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    generated, logprobs, stop_reason = generate_one(hf, prompt, args.max_new_tokens,
                                                  args.temperature, args.seed, eos_ids)
    sync(args.device)
    manifest["generation_seconds"] = time.perf_counter() - start
    original_ids = prompt + generated
    data = {"schema_version": 1, "messages": messages, "prompt_ids": prompt,
            "generated_ids": generated, "full_ids": original_ids, "ids_sha256": ids_hash(original_ids),
            "prompt_length": len(prompt), "generated_length": len(generated),
            "attention_mask": [1] * len(original_ids), "stop_reason": stop_reason,
            "raw_sampled_logprobs": logprobs, "tokens": token_records(tokenizer, prompt, generated, logprobs),
            "decode_policy": {"skip_special_tokens": False, "clean_up_tokenization_spaces": False},
            "generated_text": decode(tokenizer, generated), "display_text": decode(tokenizer, generated, True),
            "full_text": decode(tokenizer, original_ids),
            "spans": find_spans(prompt, generated, tokenizer, user_positions, eos_ids)}
    transcript_path = args.output_dir / "transcript.json"
    write_json(transcript_path, data)
    manifest["status"] = "captured; replay pending"
    write_json(manifest_path, manifest)
    # Reload the persisted IDs, compare to the actual generation, and replay them directly.
    saved = json.loads(transcript_path.read_text())
    validate_transcript(saved, tokenizer, original_ids)
    for token in saved["tokens"]:
        print(json.dumps({"boundary_token": token}, ensure_ascii=False), flush=True)
    lens = jlens.JacobianLens.from_pretrained(LENS, filename=LENS_FILE, revision=LENS_REVISION)
    model = jlens.from_hf(hf, tokenizer, force_bos=False)
    sync(args.device)
    start = time.perf_counter()
    validation = replay_one(model, lens, tokenizer, saved, args.output_dir / "lens_readouts.jsonl",
                            args.position_chunk_size, args.logprob_flag_threshold, original_ids)
    sync(args.device)
    manifest["replay_seconds"] = time.perf_counter() - start
    write_json(args.output_dir / "replay_validation.json", validation)
    manifest.update(status="complete", lens_layers=lens.source_layers,
                    workspace_band=[l for l in lens.source_layers if 12 <= l <= 29],
                    span_analysis_eligible=saved["spans"]["span_analysis_eligible"],
                    peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated() if args.device == "cuda" else None,
                    peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved() if args.device == "cuda" else None)
    write_json(manifest_path, manifest)
    print(json.dumps({"status": "complete", "output_dir": str(args.output_dir),
                      "stop_reason": stop_reason, "spans": saved["spans"],
                      "token_identity": validation["token_identity"],
                      "diagnostics": validation["diagnostics"]}), flush=True)


if __name__ == "__main__":
    main()
