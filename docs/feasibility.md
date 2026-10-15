# J-space pilot: feasibility assessment

Assessed 15 October 2026 against `pilot_spec_15_10.md`. **Conditional go for a
one-example GPU validation; not yet a go for the pilot.** The exact artifacts
exist and their dimensions agree. Exact-ID lens readout is feasible through
existing hooks. Stock Inspect HF output cannot satisfy the specified replay
contract. No GPU is exposed here (`torch.cuda.is_available() == False`, zero
devices); no pretrained-model forward or generation has been executed.

## Verified facts

### Model and pretrained lens

| Artifact | Evidence and result |
| --- | --- |
| `Qwen/Qwen3.5-4B` | HF API returns public, ungated model at commit `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`. Config and tokenizer downloaded; both weight shards return successful Hub file metadata requests. Full model weights were **not** downloaded. |
| Specified lens | The exact path below exists on `qwen-n1000`, resolved to commit `16a01f309fcec900fdcec3f4cd5b64f3d00e4d5a`. Downloaded all 406,332,644 bytes and loaded with `JacobianLens.from_pretrained`. SHA-256 matches the Hub ETag: `1f9a8f8fd593f0ffec1a9640993257ca4560f8ae3e5602315643d5cc6818534e`. |
| Compatibility | Model text config: 32 layers, width 2,560, vocabulary 248,320. Lens: width 2,560, 1,000 fitting prompts, 31 finite matrices of shape `[2560,2560]`, layers **0–30**. This is the exact pairing named in the upstream walkthrough. Shape and documented pairing are verified; useful pretrained readouts and compatibility with the precise model weight revision remain untested. |

Exact lens path (unchanged from the specification):

```text
qwen3.5-4b/jlens/Salesforce-wikitext/Qwen3.5-4B_jacobian_lens_n1000.pt
```

Sources: [model API](https://huggingface.co/api/models/Qwen/Qwen3.5-4B),
[pinned model config](https://huggingface.co/Qwen/Qwen3.5-4B/blob/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a/config.json),
[lens artifact](https://huggingface.co/neuronpedia/jacobian-lens/blob/16a01f309fcec900fdcec3f4cd5b64f3d00e4d5a/qwen3.5-4b/jlens/Salesforce-wikitext/Qwen3.5-4B_jacobian_lens_n1000.pt),
[walkthrough](https://github.com/anthropics/jacobian-lens/blob/581d398613e5602a5af361e1c34d3a92ea82ba8e/walkthrough.ipynb).
The lens serialization contains matrices, layers, width and prompt count, not
the model revision used during fitting; matching shapes alone cannot prove that provenance.

### Actual lens API and exact IDs

At upstream commit `581d398613e5602a5af361e1c34d3a92ea82ba8e`:

```python
lens = jlens.JacobianLens.from_pretrained(repo, filename=path, revision=revision)
model = jlens.from_hf(hf_model, tokenizer, force_bos=False)
layer_logits, model_logits, input_ids = lens.apply(
    model, prompt, layers=[16], positions=[-1], max_seq_len=512
)
```

`apply` takes a **string**, internally tokenizes it and truncates at 512 tokens
by default. Its output is logits, not log-probabilities: each layer maps to
`[selected_positions, vocabulary]`; use `log_softmax(..., dim=-1)` for scores.
The notebook uses the boot/currency prompt at **position -2**, not -1. Its
multihop evaluation file and documented min-over-layers rank metric exist;
neither their expected ranks nor the Italy/Rome readouts were tested here.

There is **no direct token-ID argument to `apply`**. Existing APIs permit a small
caller-side implementation without modifying upstream: capture block outputs
with `jlens.hooks.ActivationRecorder(model.layers, at=layers)`, call
`model.forward(input_ids)`, select `recorder.activations[layer][0, positions]`,
cast residuals to float32, then call
`model.unembed(lens.transport(residuals, layer))`. This bypasses tokenization
entirely. Use inference mode. The checked-in sanity script demonstrates this
route, with a pretrained-model forward explicitly opt-in.

Sources: [lens.py](https://github.com/anthropics/jacobian-lens/blob/581d398613e5602a5af361e1c34d3a92ea82ba8e/jlens/lens.py),
[HF adapter](https://github.com/anthropics/jacobian-lens/blob/581d398613e5602a5af361e1c34d3a92ea82ba8e/jlens/hf.py),
[hooks](https://github.com/anthropics/jacobian-lens/blob/581d398613e5602a5af361e1c34d3a92ea82ba8e/jlens/hooks.py),
[evaluation conventions](https://github.com/anthropics/jacobian-lens/blob/581d398613e5602a5af361e1c34d3a92ea82ba8e/data/evaluations/README.md).

### Thinking preservation and Inspect HF

Tested with the **actual pinned tokenizer**: `enable_thinking=True` renders a
generation header ending `assistant\n<think>\n`. `<think>` is ID **248068** and
`</think>` is **248069**. Both are single tokens but **neither is marked special**:
both survive decoding with `skip_special_tokens=True`. The opening delimiter
belongs to the prompt, not necessarily the generated continuation. Historical
assistant reasoning before the latest user turn is removed by this template;
passing `preserve_thinking=True` did not retain it in the tested two-turn example.
Do not assume newer Qwen-family documentation describes this pinned template.
The text config's EOS is **248044** (`<|endoftext|>`), whereas the tokenizer's
EOS is **248046** (`<|im_end|>`). Verify effective generation stopping explicitly;
these two IDs must not be treated as interchangeable during capture/replay.

Inspect AI **0.3.280** HF source is byte-identical to the inspected GitHub commit
`3f9c2e3fb4cc4799577971f61d29a5c8cd381ed4`:

- `enable_thinking` is a model argument forwarded to the chat template.
- The no-tool generation path returns ordinary assistant text, without parsing
  reasoning into a separate `ContentReasoning` field. Thinking text and the
  generated closing delimiter can remain, while the opening delimiter is in
  the prompt. Other tokens classified as special, including turn endings, are
  removed by its decoder.
- Prompt IDs and generated IDs exist inside `_generate_batch` but are discarded
  from `GenerateOutput` and the returned `ModelOutput`. Public sample metadata
  does **not** automatically receive them.
- `GenerateConfig(logprobs=True)` requests raw `output_logits` and applies
  log-softmax. However, `extract_logprobs` reports the **top-1** token and its
  probability, without gathering the probability of the actual sampled ID.
  With temperature 0.7 sampling, these can differ. Increasing `top_logprobs`
  does not guarantee that the sampled token is included.

Sources: [Inspect HF provider](https://github.com/UKGovernmentBEIS/inspect_ai/blob/3f9c2e3fb4cc4799577971f61d29a5c8cd381ed4/src/inspect_ai/model/_providers/hf.py),
[pinned chat template](https://huggingface.co/Qwen/Qwen3.5-4B/blob/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a/chat_template.jinja).

## Things tested successfully

- Installed `requirements.txt` into `/workspace/.venvs/jspace-pilot` using Python
  3.12 and CPU PyTorch 2.14.1; `pip check` passes. Imports of Torch,
  Transformers 5.5.4, jlens 0.1.0 and Inspect 0.3.280 pass.
- Ran `scripts/sanity_check.py` in its default artifact-only mode: real lens
  download/load, finite-value and shape checks, real tokenizer prompt-ID
  round-trip and thinking delimiter checks pass. **No model weights or forward.**
  Its explicit CUDA-forward request also correctly exited with status 2 on this
  machine before downloading model weights; that is a guard check, not a GPU pass.
- A separate CPU probe used a tiny **randomly initialized**, two-layer hybrid
  Qwen3.5 with an **identity**, synthetic lens. Exact IDs traversed real jlens
  hooks without a tokenizer; final unembedding matched ordinary model logits.
  This validates the adapter route, not the pretrained lens's quality.
- Called Inspect's actual `extract_logprobs` with a synthetic sampled token:
  sampled token log-probability was -1.31326, but reported token was the argmax
  with -0.31326. This reproduces the replay diagnostic problem without generation.
- Built the real text configuration on the meta device and tied weights:
  4,205,751,296 parameters, no weights loaded. `from_config` needs the extracted
  text config; passing the composite config directly raised `AttributeError`.
  Inspected `from_pretrained` has the text-config extraction branch.

## GPU memory estimate (not measured)

Batch size 1, unquantized bf16 text model, 2,048 total tokens:

| Allocation | Estimate |
| --- | --- |
| Text model parameters | **7.83 GiB** from meta-device parameter count; full multimodal loading is larger. The two published shards total 9.32 GB on disk. |
| Lens matrices | **0.757 GiB** float32, held on CPU by the loader; approximately 25 MiB per matrix transferred for a layer readout. File is approximately 388 MiB. |
| Recorded residuals, all 32 blocks | Approximately **0.313 GiB** in bf16, excluding forward temporaries. |
| Full-attention KV cache | Approximately **64 MiB** at 2,048 tokens: 8 attention layers × 2 × 4 KV heads × 256 head width × tokens × 2 bytes. Hybrid linear layers also retain recurrent/convolution state (roughly 50 MiB). |
| One full-vocabulary readout | Approximately **1.89 GiB** float32, before extra log-softmax/casting buffers. |
| Default `apply` CPU outputs | **60.6 GiB** for 31 lens layers plus model logits, all positions, float32. This exceeds this machine's approximately 33 GiB host RAM. |

16 GB is plausible for short, carefully bounded inference but **not a verified
budget** for the proposed all-token workflow. Use 24 GB for the first GPU smoke
test if available. Reading a few positions is cheap; all-position vocabulary
outputs and Inspect's retained generation logits can dominate memory. An
all-token implementation will need bounded readout buffers and immediate
reduction to scores/top tokens, including host memory planning. This is a
resource constraint, not an implementation included in this task. CPU float32
model parameters alone would take approximately 15.7 GiB.

## Unverified assumptions and specification issues

1. **Replay is feasible in principle, not with the stock Inspect records.** A
   small capture adapter is needed to save actual prompt/continuation IDs,
   masks, unpadded lengths, stop/EOS handling, model/tokenizer revisions and
   each sampled token's raw log-probability. Compare replay logits at
   `prompt_length + generated_index - 1` against that generated ID. Compare raw
   model probabilities on both sides, not temperature/top-p processed scores.
   No such adapter or pretrained replay comparison was implemented here.
2. **Text equality needs a defined view.** Decode continuation IDs with the
   same special-token/cleanup policy to compare with Inspect's generated text;
   retain a separate lossless full-sequence decoding. Full prompt+continuation
   decoding with specials cannot equal Inspect's answer-only, cleaned text.
3. **Exactly one delimiter pair is not guaranteed.** The opening token is in
   the header; truncation can omit the closing token, and token-like strings in
   user text or repeated output can create additional occurrences. Thinking ON
   does not guarantee a nonempty reasoning span. Bound checks to the relevant
   assistant turn and record incomplete cases explicitly.
4. Model/lens dimensions and source compatibility are established, but the
   pretrained forward, boot example, ten multihop ranks, generated reasoning,
   bf16/cache numerical differences and GPU peak memory remain **unrun**.
   A 0.5-nat threshold has not been calibrated. Float32 costs twice the model
   parameter memory and may exceed the original 16 GB budget.
5. `apply`'s default truncation violates lossless long replay. Lens source
   layers exclude the final layer. Define zero-based workspace-band rounding
   explicitly and intersect it with fitted layers. The upstream notebook's
   final cell **fits a new lens**; executing every cell is unnecessary setup.
6. The pinned model is a hybrid linear/full-attention architecture. Transformers
   >=5.5 is a jlens dependency. Its source supplies a Torch fallback, so
   `flash-linear-attention` and `causal-conv1d` are optional accelerators, not
   required for the CPU interface checks. GPU performance is untested.
7. The primary **concealment** score is missing from section 7's named analysis
   list. Section 6 also needs to specify how to handle words with only one valid
   single-token form and when to maximize over space/no-space forms. These are
   specification ambiguities, not changes made to the experiment. APE paths,
   Docent APIs, paper-band claims and future 27B hardware were not independently
   validated; they are outside this first setup's execution scope.

## Blockers and next smallest step

The remaining execution blocker is **GPU access**, plus the known Inspect
capture limitation before an exact replay test. No missing HF credential was
found: the required artifacts are public and downloads worked unauthenticated.

Use the existing isolated checkout; do not create a worktree. Current setup:

```bash
cd /workspace/manipulation-jspace-pilot
source /workspace/.venvs/jspace-pilot/bin/activate
export HF_HOME=/workspace/.cache/huggingface
export OMP_NUM_THREADS=2
python scripts/sanity_check.py
# On a GPU machine; NOT executed here:
python scripts/sanity_check.py --run-forward --device cuda
```

The script's optional forward loads the pinned text model and runs a one-position
exact-ID lens readout; it does not generate text or validate replay. The next
small task is to run that GPU check and the notebook's boot readout, then capture
**one** thinking generation with IDs/raw sampled-token probabilities and compare
one replay before proceeding to the ten multihop checks. Do not start the dataset,
judges or analysis pipeline yet.

Minimal dependencies are in `requirements.txt`; jlens is pinned to the inspected
commit. `huggingface-hub` and NumPy arrive through jlens/Transformers; Accelerate
supports model placement. No judge, Docent, notebook or analysis packages were
added. Install the hardware-appropriate verified PyTorch wheel first; the tested
environment uses the official CPU index. GPU-wheel installation remains untested.

Network access initially returned proxy `403 Forbidden`, then became available.
Hugging Face domain additions were saved in the environment draft:
`huggingface.co`, `*.huggingface.co`, `*.hf.co`, `*.xethub.hf.co`.
Reusable CPU installation and activation instructions are also saved in the draft.
Review/save those settings and publish the environment to retain configuration;
no publication or fresh-task restoration was verified here.
