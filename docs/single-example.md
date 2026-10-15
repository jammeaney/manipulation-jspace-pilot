# One thinking generation and exact-ID replay

This is a single benign example from `pilot_spec_26_10_2.md`, not the pilot
pipeline. It does not generate a dataset, apply persuasion prompts, calculate
word-list scores/classification metrics, run judges, or perform bootstrap analysis.
Step 0 passed on RunPod as reported by the user (see `feasibility.md`). The
uploaded single-example run now also documents successful pretrained GPU
capture/replay: see the [diagnostic report](results/single-example-001.md).

## Run on RunPod

Use the clean pinned environment in [gpu-validation.md](gpu-validation.md),
including matched Torch 2.14.1 / torchvision 0.29.1 wheels. No new dependencies
are required. From a checkout containing this PR:

```bash
source /workspace/jspace-gpu-clean/bin/activate
export HF_HOME=/workspace/.cache/huggingface
export OMP_NUM_THREADS=2
python scripts/single_example.py \
  --output-dir outputs/single-example-001 \
  --device cuda --max-new-tokens 256 --temperature 0.7 --seed 42
```

The output directory must not exist: prior results are never overwritten. Change
its name for each run. `--prompt` and `--system` accept one message each; the
benign default asks `2 + 2` with brief reasoning. `--position-chunk-size` defaults
to 8; reduce it if readout buffers exhaust GPU memory. `--dtype float32` enables
a numerical diagnostic rerun if sufficient memory is available. `--device cpu`
is supported, defaulting to float32, but the real 4B model has substantial memory
and runtime costs. CPU regression tests do not load these pretrained weights.

The model, tokenizer, lens and jlens revisions are the existing pinned values.
Generation uses explicit cached Transformers forward calls, batch size one,
no padding, temperature sampling, top-p 1 / top-k 0, no penalties or other logit
processors. It uses a device-local seeded Torch generator. Each sampled token's
probability comes from **raw, unscaled model logits**, not the sampling distribution
or the argmax token. A raw vector is discarded after recording its sampled scalar
probability. Stopping is explicit on either the tokenizer EOS (248046) or model
EOS (248044), or the new-token limit. EOS is retained in the saved IDs.

## Outputs

- `manifest.json`: schema version, model/tokenizer/lens/jlens revisions, exact
  settings, package/Python versions, Git commit/dirty state and script checksum,
  device, dtype, CUDA build, attention implementation, numerical backend flags,
  timings and peak allocated/reserved GPU bytes. Status remains `initializing`
  or `captured; replay pending` if execution fails before completion.
- `transcript.json`: messages, prompt/generated/full IDs, sequence checksum,
  unpadded lengths and all-ones attention mask, stop reason, raw sampled-token
  log-probabilities, canonical tokenizer strings and per-ID decoded pieces,
  complete generated/full text, display text, and span boundaries/eligibility.
- `replay_validation.json`: exact sequence/text equality, per-token generation
  versus replay log-probabilities, absolute differences, median/p99/maximum
  difference, large-error indices and possible persistent jumps.
- `lens_readouts.jsonl`: one record for every **fitted layer × full-sequence
  token position**, including response start, user content, reasoning and answer.
  Each record includes the layer, position, input token ID, span memberships,
  whether the layer is in workspace band 12–29, and top-25 vocabulary IDs,
  canonical token strings and normalized natural-log probabilities.

Full-vocabulary readouts exist only for one small position chunk at a time.
Normalization uses the entire vocabulary's logsumexp before selecting top 25;
these are not probabilities renormalized over the top tokens. All layer/position
records stream to disk. Other vocabulary probabilities and word-list scores are
not stored; no frozen word lists/control list were supplied for this task. All
saved IDs remain available for a later readout with explicitly selected words.

The transcript is written and read back before replay. The loaded IDs are compared
against the actual generation sequence, the full-ID checksum and token records.
A tensor built from those saved IDs is checked before and after the single
cache-free, hooked model forward. No decoding/re-tokenizing or chat re-rendering
is used to construct replay input. User offsets are derived and checked against
the actual generation prompt **before generation**, solely to identify its span.

Replay probability for generated token `j` uses the actual final-model logits at
`prompt_length + j - 1`. This is separate from the lens's transported readouts.
Identity, text, count and finite-value failures are hard errors. Numerical
log-probability differences are **diagnostics**, not identity failures: errors
above `--logprob-flag-threshold` (default 0.5 nats) and possible persistent jumps
are flagged, without claiming that this threshold has been calibrated.

## Thinking boundaries

All coordinates are zero-based full-sequence token indices with end-exclusive
span ends. `response_start` is the last prompt token, immediately before the first
generated token. The thinking opener is located in the prompt, and the first
closing delimiter is located in the generated IDs. The actual pinned Qwen
header ends `<think>\n`, so the opener is **not literally the last prompt token**;
response start is the trailing newline token. The reasoning range between the
opener and closer includes that header whitespace, as specified by the token
boundary definition. The final answer starts after the closing delimiter and
excludes a terminal EOS token. Thinking delimiters themselves are excluded.

Missing closing delimiters (including token-limit truncation), empty reasoning,
empty final answers, and extra delimiters are recorded as flags. Reasoning/answer spans are then null
and ineligible for span-specific analysis; replay and general per-token readouts
still run. A `complete` manifest means execution completed, **not** that valid
reasoning and answer spans were obtained. Check `span_analysis_eligible` and
`stop_reason`; rerun with a larger token limit if appropriate. Do not silently
substitute a final answer for an incomplete reasoning response.

The script prints every token with its ID, canonical string, decoded piece and
position for boundary inspection. Complete-string decoding defines text equality:
`skip_special_tokens=False`, `clean_up_tokenization_spaces=False`. A separate
display view removes special tokens. Per-ID decoded pieces may contain replacement
characters for split byte sequences; joining those pieces is not an identity
check. IDs are the source of truth.

## Validation evidence and pending GPU check

All 14 CPU regression tests pass, including the seven existing Step 0 tests.
The new tests use a tiny, randomly initialized hybrid Qwen3.5 and synthetic
lens. They cover cached sampling, independent raw-versus-temperature probability
verification, full replay/log-probability alignment, per-layer/position outputs,
span edge cases, corruption detection, seed repeatability, explicit stopping,
end-to-end artifact reload and refusal to overwrite results. Run them with:

```bash
OMP_NUM_THREADS=2 python -m unittest discover -s tests -v
```

The actual cached pinned Qwen tokenizer also passed a CPU prompt/offset/span
check with a synthetic continuation, and the CUDA CLI guard refused execution
before loading weights or creating an output directory.

**Codex did not execute this on GPU.** The uploaded RunPod artifacts have now
been checked and summarized in [the result report](results/single-example-001.md).
For another run, use a fresh directory and inspect the four artifacts: identity must pass, valid thinking
and answer spans should be present, and numerical differences must be reviewed.
Share the manifest and replay diagnostics to record measured GPU results. A seed
and revision pins aid reproduction; they do not guarantee bitwise identical
sampling across devices, drivers, kernels or package versions. Generation/replay
timings exclude weight loading; GPU peaks include resident weights and readout
work, while reserved memory includes allocator caches. These outputs are ignored
by Git under `outputs/`; no generated model responses are included in the PR.
