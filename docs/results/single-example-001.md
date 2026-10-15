# RunPod single-example-001: diagnostic summary

The uploaded `single-example-001.zip` documents a successful **single-example**
GPU generation/replay run. This report independently checks its saved artifacts
on CPU; it does not claim a new GPU execution or validate the full pilot.

## Run and integrity

- **NVIDIA A40**, bf16, CUDA build 13.0, SDPA; Python 3.12.3.
- Torch 2.14.1, torchvision 0.29.1, Transformers 5.5.4, Accelerate 1.15.0, jlens 0.1.0.
- Pinned model/tokenizer, lens and jlens revisions match the configured versions.
- Temperature 0.7, seed 42, maximum 256 new tokens, position chunks of 8.
- **39 prompt + 249 generated = 288 tokens**; stopped on EOS 248046, not the limit.
- Final answer: **“2 plus 2 equals 4.”**
- Reasoning and answer spans are eligible with no delimiter flags. Response start
  is position 38 (the header newline); reasoning is `[38,277)`, answer `[278,287)`.
  The reasoning span includes one prompt newline: 239 span positions, of which
  238 are generated. User content occupies positions 14–31.
- Saved prompt/generated/full IDs, SHA-256, text views, canonical token strings,
  span boundaries and per-token probability alignment pass independent consistency checks.
  The run records that replay IDs matched the original in-memory generation;
  that historical comparison cannot be re-executed from files alone.
- All **8,928 distinct layer/position records** are present: layers 0–30 × positions
  0–287. Each has 25 unique IDs, finite sorted log-probabilities, correct span labels
  and top-25 probability mass ≤1. Full-vocabulary normalization cannot be independently
  recomputed from top-25 outputs alone; the matching script implements it.

## Replay log-probability differences

Absolute differences are in **nats**, comparing the sampled token’s raw model
log-probability during cached generation with its full-sequence replay probability.
All 249 token comparisons and prediction-position offsets were checked.

| Statistic | Absolute difference |
| --- | ---: |
| Median | 3.81469727e-06 |
| Mean | 0.00455674014 |
| 95th percentile | 0.0367435455 |
| 99th percentile | 0.0778353882 |
| Maximum | 0.117702484 |

- 97/249 differences are exactly zero; 26 exceed 0.01 nats and 7 exceed 0.05.
- **None exceeds 0.5 nats**; the saved diagnostic reports no persistent-jump flags.
- Generated reasoning tokens: median `3.8147e-6`, p99 `0.0796991`, max `0.117702`.
- Nine answer-span tokens: median `0`, p99 `0.000197525`, max `0.000200272`.

Largest five differences (zero-based generated/full-sequence indices):

| Generated index | Full position | Decoded token | Difference (nats) |
| ---: | ---: | --- | ---: |
| 180 | 219 | `"\n\n"` | 0.117702484 |
| 175 | 214 | `" simplicity"` | 0.101499557 |
| 200 | 239 | `" sum"` | 0.0859680176 |
| 167 | 206 | `" the"` | 0.0690250397 |
| 130 | 169 | `" One"` | 0.0641384125 |

These are small relative to the configured flag threshold and are compatible with
bf16/cache-versus-full-pass differences. This single run does **not** establish
their cause or calibrate the threshold; a float32 replay of the same saved IDs
would be needed to isolate precision effects. Exact-ID equality is a separate
hard check and passed according to the saved run.

## Per-layer response-start readouts

Unfiltered top three tokens at **position 38**, decoded for readability. Strings
are JSON-escaped, preserving leading spaces and whitespace; these are lens
distributions, not the model’s actual next-token distribution at every layer.

| Layer | Workspace band | Top three tokens | Top-1 probability |
| ---: | :---: | --- | ---: |
| 0 |  | `````[" ``", " --", " <"]````` | 0.6488 |
| 1 |  | `````[" `", " <", " ``"]````` | 0.4097 |
| 2 |  | `````[" `", "ly", " <"]````` | 0.7323 |
| 3 |  | `````[" `", " <", "ly"]````` | 0.5125 |
| 4 |  | `````["ly", " `", " *"]````` | 0.4207 |
| 5 |  | `````[";", ",", "ly"]````` | 0.2572 |
| 6 |  | `````[",", ";", "."]````` | 0.7043 |
| 7 |  | `````[",", ";", "."]````` | 0.5020 |
| 8 |  | `````[";", ",", "·"]````` | 0.4151 |
| 9 |  | `````["<\|endoftext\|>", ".", "  \n"]````` | 0.5612 |
| 10 |  | `````["<\|endoftext\|>", ".", "<\|im_end\|>"]````` | 0.8518 |
| 11 |  | `````["\n\n", "<\|im_end\|>", "<\|endoftext\|>"]````` | 0.9742 |
| 12 | yes | `````["...", "<\|endoftext\|>", "\n\n"]````` | 0.4322 |
| 13 | yes | `````["<\|endoftext\|>", " thinking", " thought"]````` | 0.0317 |
| 14 | yes | `````["!”.", "!!!!", "!)."]````` | 0.0335 |
| 15 | yes | `````["<\|im_end\|>", ".).", "...)."]````` | 0.0552 |
| 16 | yes | `````["**?", ")?", "?**"]````` | 0.0330 |
| 17 | yes | `````["\t\n\n\n", "问", "回答"]````` | 0.0266 |
| 18 | yes | `````["）**", "回答", "思考"]````` | 0.0281 |
| 19 | yes | `````["！", "这个问题", "回答"]````` | 0.0527 |
| 20 | yes | `````["!).", "!!!!!", "**!"]````` | 0.0350 |
| 21 | yes | `````["!**", "!]", "!\");"]````` | 0.0446 |
| 22 | yes | `````["用户", " user", "user"]````` | 0.6168 |
| 23 | yes | `````["用户", " user", "user"]````` | 0.3545 |
| 24 | yes | `````[" user", "用户", " User"]````` | 0.2734 |
| 25 | yes | `````["用户", " user", "用户的"]````` | 0.4057 |
| 26 | yes | `````["Thinking", " Thinking", "用户"]````` | 0.3209 |
| 27 | yes | `````["Thinking", " Thinking", " thinking"]````` | 0.6823 |
| 28 | yes | `````[" user", "Thinking", "用户"]````` | 0.4217 |
| 29 | yes | `````["Thinking", " Thinking", " thinking"]````` | 0.7817 |
| 30 |  | `````["Thinking", " Thinking", "The"]````` | 0.9756 |

Early layers are dominated by punctuation/control tokens. The middle band is
mixed: layer 13 includes “thinking”/“thought”; layers 17–19 include Chinese terms
for question/answer/thinking. Layers 22–25 emphasize “user”/`用户`, and layers
26–27 and 29–30 place **“Thinking” first**, matching the first generated token.
Layer 28 returns to “user”. This is structured, position-dependent content, but
not evidence for persuasion, concealment or manipulation separation.

As a second, explicitly illustrative position, token **280** is `" plus"` in the
final answer. The top lens token is `" equals"` at layers 20, 24 and 29; at layer
30 it is a space. Thus late readouts can contain arithmetic-related continuation
content even though response-start readouts are often punctuation-heavy. This
is not an aggregate concept-recovery score. The JSON summary includes this anchor
and, for every layer, span-wise top-1 token frequencies and mean top-25 mass.

## Resources and provenance

- Generation: **13.96 s**; replay plus lens extraction: **8.42 s** (weight loading excluded).
- Peak allocated GPU memory: **7.950 GiB**; reserved: **8.031 GiB**.
- These measurements apply to this short, chunked bf16 run, not the full pilot.
- The manifest records a **dirty worktree** at Git commit
  `dad9f66d998a204230c6da1738066a73d3462f75`. However, its script SHA-256
  `c89cd9f1ed5ffd5b64cd6b1d323cf68ecf7375cd96f2cc0db6bbed734a620ebf`
  exactly matches the checked-in `scripts/single_example.py`. Other local changes
  are unspecified; a complete transitive dependency freeze and driver version
  were not included. Reproduce future runs from a clean checkout and retain those.

[Machine-readable summary](single-example-001-summary.json) contains source-file
checksums, exact manifest/settings, independently calculated statistics and
all 31 layer summaries. Raw uploaded outputs remain in ignored
`outputs/single-example-001/` locally and are not added to Git. No raw reasoning
is copied into this report. The archive SHA-256 is:

`8dcf1ca236260f7e3262d859c04e642d3af852d2acf50d91d2add7b9d86345ff`
