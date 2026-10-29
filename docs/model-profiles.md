# Model-specific original Jacobian Lens profiles

The existing Qwen3.5-4B experiment remains the default. No prompt data, original
frozen concept artifact, sampler settings, probability calculation or replay
precision has changed. This extension adds a Qwen3.6-27B profile and an explicitly
experimental GPT-OSS-20B single-example path. No J++/`jpp` or oracle artifact is used.

## Profiles and current evidence

| Model | Model/tokenizer commit | Original lens commit | Width / layers / fitted layers | Paired-run policy |
| --- | --- | --- | --- | --- |
| `Qwen/Qwen3.5-4B` | `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` | `16a01f309fcec900fdcec3f4cd5b64f3d00e4d5a` | 2560 / 32 / 0–30 | Existing validated path; unchanged |
| `Qwen/Qwen3.6-27B` | `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9` | `e166deb5ad540932100ef803c8c9bcc2a5453694` | 5120 / 64 / 0–62 | Requires a complete, profile-bound smoke report |
| `openai/gpt-oss-20b` | `6cee5e81ee83917806bbde320786a8fb61efebee` | `e166deb5ad540932100ef803c8c9bcc2a5453694` | 2880 / 24 / 0–22 | Gated pending fitting-precision and pretrained-result review |

All lenses come from `neuronpedia/jacobian-lens`:

```text
qwen3.5-4b/jlens/Salesforce-wikitext/Qwen3.5-4B_jacobian_lens_n1000.pt
qwen3.6-27b/jlens/Salesforce-wikitext/Qwen3.6-27B_jacobian_lens_n1000.pt
gpt-oss-20b/jlens/Salesforce-wikitext/gpt-oss-20b_jacobian_lens.pt
```

`scripts/model_profiles.py` defines the reviewed combinations.
`--model` selects one; optional `--model-revision`, `--lens`, `--lens-revision`
assert that same combination. Unknown models, moving aliases, unreviewed overrides
and alternate lens repositories are rejected. Supporting a different revision
requires a reviewed profile update and revalidation, not bypassing compatibility.

New models require the original `jlens` source commit
`581d398613e5602a5af361e1c34d3a92ea82ba8e` to be recorded by the installation's
`direct_url.json`. Version `0.1.0` alone cannot distinguish implementations.
Install the existing requirements if this source check fails; do not replace
packages with a moving branch or a J++ implementation.

User-provided RunPod evidence (not GPU tests executed by Codex):

- Both pinned tokenizers loaded. Qwen27B's thinking prompt ends `<think>\n`;
  its opening/closing delimiters independently resolved to single-token IDs.
- GPT-OSS's pinned template recognizes reasoning effort and renders Harmony
  `analysis`/`final` channels. Actual native markers are `<|start|>`, `<|message|>`,
  `<|end|>`, `<|return|>`; old-looking `<|im_start|>`/`<|fim_suffix|>` strings were
  ordinary multitoken text and are not accepted as protocol controls.
- Both original lens artifacts loaded with matching dimensions/layers. Qwen27B
  records 1000 fitting prompts; GPT-OSS records **457**, despite a requested 1000.
- Qwen27B's pretrained `lens.apply`/hook comparison had zero error at every fitted
  layer on an A100-SXM4-80GB. Peak allocation was approximately 51.1 GiB. Its
  128-token generation produced reasoning but hit the cap before closing thinking.
  That result does **not** qualify as a complete smoke report for paired runs.
- GPT-OSS fitting config reports BF16 dtype, a B200, a backfilled environment and
  no recorded jlens commit/model commit. Dtype alone does not prove whether MXFP4
  weights were dequantized. Exact fitting-model revisions are not established by
  the lens files/credits alone for either new model.

Runtime compatibility tests do not prove historical fitting provenance. These
limitations remain visible even when a local smoke passes.

## Workflow on RunPod

Activate the existing clean experiment environment and work from this PR revision.
No dependency upgrade is required by this PR. Keep cache and outputs on persistent
storage; do not mix model runs in one directory. The default 4B commands continue
working without any new flags.

```bash
cd /root/manipulation-jspace-pilot
# Activate your existing .venv here.
export HF_HOME=/workspace/.cache/huggingface
export OMP_NUM_THREADS=2
mkdir -p outputs/model-concepts
```

First freeze **separate** concept IDs with the selected pinned tokenizer. This
loads tokenizer files only; it does not load model weights or use pilot outcomes:

```bash
python scripts/model_profiles.py --model Qwen/Qwen3.6-27B \
  --output outputs/model-concepts/qwen36-27b.json
```

The source defaults to the unchanged `data/paired-feasibility/concepts.json`.
Every original probe/control word is retokenized using plain and leading-space
forms, keeping single-token variants in the same order as the original rule.
The new artifact records model/tokenizer pins, word-set fingerprint, source hash
and exact IDs. It refuses missing single-token words/unknown tokens rather than
silently replacing words, dropping controls, changing group sizes or copying 4B
IDs. If it stops, the unavailable words are listed and require a separate design
review. Do not tune the list on paired outputs. Actual new-model IDs are not
fabricated in this PR; generate/freeze them on the tokenizer-accessible RunPod.

Then run one **complete** example, with original-method checks and saved replay:

```bash
python scripts/single_example.py --model Qwen/Qwen3.6-27B \
  --output-dir outputs/qwen36-27b-validation-001 \
  --device cuda --dtype bfloat16 --temperature 1.0 --seed 42 \
  --max-new-tokens 1024 --position-chunk-size 8 --smoke-checks \
  --prompt 'What is 17 multiplied by 19?'
```

This increases only this explicitly requested validation run's token cap. It does
not change the single-example default (256) or paired default (512). No reasoning
brevity instruction is added to this prompt. All outputs require fresh directories.

`--smoke-checks` compares exact-ID hooks against the original `lens.apply` at
position -2 of the upstream boot/country/currency prompt (all fitted layers,
`atol=rtol=1e-3`), saves per-layer Italy/Rome/euro ranks, and requires an Italy/Rome
intermediate in the top 10 at some fitted layer. It does not fit a new lens.
`smoke_checks.json` is evidence, not a classifier or a full multihop evaluation.
The normal pipeline then generates one response with raw sampled-token LPs and
replays the exact saved sequence. It retains all original bounded readouts.

`model_validation.json` has status `complete_smoke_pass` only when the method
comparison, intermediate check, exact token identity and **EOS + valid nonempty
thinking/answer spans** all pass. A token-limited run stays incomplete, even if
some spans are usable. Inspect saved replay numerical diagnostics too: 0.5 nats
remains a diagnostic flag, not a new automatic exclusion threshold.

The paired-run gate checks profile, dtype/loading mode, package versions (including
optional attention/convolution/kernel packages and the CUDA build), source
hashes, the report's artifact hashes and the underlying saved completion/identity
checks. Keep the report beside its source artifacts. A stale or changed report,
truncated response, altered artifact or different code/runtime is rejected. Repeat
the smoke after relevant code/environment changes; do not edit a report to force
success. This is reproducibility validation, not a signed/adversary-proof attestation.

Only after reviewing a successful report, preview the unchanged five pairs offline:

```bash
python scripts/paired_feasibility.py prepare --model Qwen/Qwen3.6-27B \
  --concepts outputs/model-concepts/qwen36-27b.json \
  --output-dir outputs/paired-qwen36-preview
```

Then the corresponding paired command is:

```bash
python scripts/paired_feasibility.py run --model Qwen/Qwen3.6-27B \
  --concepts outputs/model-concepts/qwen36-27b.json \
  --validation-report outputs/qwen36-27b-validation-001/model_validation.json \
  --output-dir outputs/paired-qwen36-001 \
  --device cuda --dtype bfloat16 --temperature 1.0 --seed 42 \
  --max-new-tokens 512 --position-chunk-size 8
```

The 512-token budget here is the preserved paired default, not a guarantee that
27B reasoning will finish. Choose and document a larger budget explicitly before
the paired run if desired; a successful 1024-token smoke cannot prove completion
at 512. No automatic retries or outcome-based selection are introduced. Model and
lens load once for the paired run. Existing one-scenario/two-sample inputs also
remain supported; the dataset, conditions and summary weighting are unchanged.

Recompute summaries without models or GPUs into another new directory:

```bash
python scripts/paired_feasibility.py summarize \
  --run-dir outputs/paired-qwen36-001 --output-dir outputs/paired-qwen36-review
```

## GPT-OSS remains experimental

The Harmony parser discovers/validates single-token native controls from the pinned
tokenizer/template, including the channel marker, rather than hardcoding IDs or
accepting a round-tripping multitoken string. It requires one assistant analysis
message followed by one assistant final message. Tool/other channels, repeated or
reordered channels, malformed headers, empty spans and unexpected stopping tokens
are flagged. Valid final content cut off at a token limit can retain span eligibility
while generation completion stays false. Native header/control tokens are excluded
from the reasoning/answer spans; response start remains the last prompt token.
A refusal with no analysis is preserved and marked ineligible, not given invented
reasoning. Unsupported tokenizer/template grammar fails **before weights load**.

For a **candidate BF16-dequantized** single-example smoke on the A100:

```bash
python scripts/single_example.py --model openai/gpt-oss-20b \
  --experimental-smoke --gpt-oss-loading bf16-dequantized --reasoning-effort medium \
  --output-dir outputs/gptoss20-validation-bf16-001 \
  --device cuda --dtype bfloat16 --temperature 1.0 --seed 42 \
  --max-new-tokens 1024 --position-chunk-size 8 --smoke-checks \
  --prompt 'What is 17 multiplied by 19?'
```

Both experimental modes use an explicit eager, sink-aware attention path; this
choice is recorded rather than silently selecting a different attention backend.
`bf16-dequantized` uses the actual Transformers `Mxfp4Config(dequantize=True)`;
it does not recover an unknown pre-quantization checkpoint. It is **not asserted
to match the lens fitting precision**. Its roughly 42-GB BF16 weight payload plus
runtime should be assessed with an initial single example, not ten generations.
No such pretrained test was run for this PR. Alternatively `native-mxfp4` explicitly
requests native quantized loading; unavailable kernels/fallback changes fail rather
than silently becoming BF16. Neither mode installs kernels or changes dependencies
on the user's behalf. Both record requested and effective loading metadata.

**GPT-OSS paired `run` is blocked regardless of a smoke report.** Once the fitting
loader/provenance and real pretrained results are reviewed, a separate profile
change can enable it. There is no override that labels an unreviewed precision
compatible. Tokenizer-only concept freezing/offline preparation may be inspected,
but are not permission to launch paired generation or evidence of compatibility.

The shared sampler is unchanged: temperature from CLI, top-k 20, top-p .95 inside
top-k, generated-only presence penalty 1.5 before temperature, raw sampled LPs before
processors. This is the experiment's shared sampling rule, not a claim of native
GPT-recommended settings. Its penalty also applies to previously generated control
tokens; protocol/length/compliance differences remain model-specific confounds.

## Metadata, scoring and validation scope

New-model manifests/transcripts retain exact original input messages, rendered
native prompt, template hash/kwargs, model/lens IDs and revisions, original jlens
commit, actual loader class/quantization config, fitted prompt count, raw IDs/LPs,
boundaries, diagnostics and code/runtime hashes. GPT may move the supplied system
instruction into its native developer message and insert a date; the original
rendered prompt is captured **before generation** and never recreated for replay.
No code assumes the control tokens are all listed in `all_special_tokens`.

The workspace rule is the same 38–92% of depth with floor at both ends, intersected
with fitted layers: **4B 12–29; 27B 24–58; GPT20B 9–22**. Every fitted layer/position
still has bounded readouts, with corrected workspace flags. Summaries use the saved
model-specific band and reject mixed profiles, precision or lens artifacts. Group
LP/max-over-form/control subtraction and aggregation formulas are unchanged.
Missing scores stay null; eligibility denominators remain explicit. Compare
conditions **within each model**; do not pool raw scores across tokenizers as if
calibrated. Assigned labels remain separate from unevaluated behavioural outcomes.

CPU checks exercise tiny random models and mocked loaders—never pretrained weights.
The 68-test PR suite passes, including 20 new tests covering profiles, concept
freezing, source/revision/J++ rejection, gates, Harmony parsing, native template
snapshotting, tiny GPT/Qwen exact-ID replay and 64-layer summary integration. The
existing non-argmax sampler test's stale mock was corrected to return a **top-k
candidate index**, without changing the sampler itself.

```bash
OMP_NUM_THREADS=2 PYTHONPATH=tests python -W error::ResourceWarning -m unittest \
  test_model_profiles test_paired_feasibility test_single_example test_sanity_check \
  test_replay_report test_extract_research_review -v
```

No GPU experiments, weight downloads, concept-list changes, dependency upgrades,
new scenarios, judges, classifiers or bootstrap analyses were performed for this PR.
