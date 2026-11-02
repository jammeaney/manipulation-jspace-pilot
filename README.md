# Investigating Covert Persuasion Using the Jacobian Lens

## Overview

This project investigates whether the Jacobian Lens (J-lens) can provide useful insights into how covert persuasion is represented within large language models.

I'm particularly interested in how evidence from internal representations might differ from what we can observe in a model's chain of thought (CoT) and generated output. The current pilot is a feasibility study focused on establishing whether J-lens produces interpretable signals under different prompting conditions.

## Pilot study

The pilot uses Qwen3.5-4B and five fictional local referendum scenarios. Each scenario is presented under two conditions:

- **Neutral:** The model is instructed to provide a balanced response.
- **Covert persuasion:** The model is instructed to steer its response towards a specified option without explicitly disclosing that objective.

Generated token sequences are replayed through J-lens to examine three draft concept sets: persuasion, concealment and evaluation awareness.

The analysis compares concept scores between conditions, across model layers and different token spans. The aim is to establish whether these measurements are technically feasible and whether they reveal patterns worth investigating further.

## Current status

The generation and J-lens replay pipeline is operational, with successful token-identity validation.

Initial results suggest differences in concept scores between prompting conditions, although incomplete generations have limited comparisons across reasoning and final-answer spans.

An important unresolved question is whether these differences reflect persuasive intent or simply the wording of the system instructions.

## Models, workspace band, word lists and replay

`--model-config` selects the model, its pinned revision and its lens together:

| Config | Model @ commit | Lens (`neuronpedia/jacobian-lens`, branch `qwen-n1000` @ `16a01f30…`) | Layers / width | Workspace band |
| --- | --- | --- | --- | --- |
| `qwen3.5-4b` (default) | `Qwen/Qwen3.5-4B` @ `851bf6e8…` | `qwen3.5-4b/jlens/Salesforce-wikitext/Qwen3.5-4B_jacobian_lens_n1000.pt` | 32 / 2560 | 12–29 |
| `qwen3.6-27b` | `Qwen/Qwen3.6-27B` @ `6a9e13bd…` | `qwen3.6-27b/jlens/Salesforce-wikitext/Qwen3.6-27B_jacobian_lens_n1000.pt` (plain J-lens) | 64 / 5120 | 24–58 |

Both lenses fit layers 0 to L−2. Both tokenizers give `<think>` = 248068, `</think>` = 248069 and stop IDs
248044 (`<|endoftext|>`) and 248046 (`<|im_end|>`). The workspace band is 38%–92% of depth, rounded down at
both ends, intersected with the lens source layers. Each manifest records it.

`data/paired-feasibility/concepts.json` is the frozen word list. It now includes a **placebo** group
(theory, season, method, history, garden, journey, quality, pattern). Token IDs are resolved with each
model's own tokenizer at run time: whichever of `word` / `" " + word` is a single token, taking the max
if both are. Words with neither are dropped and logged. The resolved IDs and dropped words are saved in
the run's `concepts.json`, `experiment.json` and each sample manifest. No word is dropped for either model.
Note: "quality" occurs once in the shared Eldermere background, which is identical in both conditions.

Every summary table (`samples.csv`, `layer_comparisons.csv`, `summary.json`) includes placebo. Raw
log-probability changes, control-word log-probability changes and control-adjusted scores are shown
side by side for every group (adjusted Δ = raw Δ − control Δ).

`paired_feasibility.py replay` re-scores a finished run's saved token IDs with the current word lists,
without generating anything. It reuses the saved transcripts and the saved concept IDs (and stops if the
tokenizer disagrees with them). It also re-runs the transcript and token-identity checks and the replay
log-probability diagnostics, and compares against the source run's replay. Output goes to a new directory.

### Commands used (2026-11-02, A100-80GB)

```bash
source .venv/bin/activate
export HF_HOME=/workspace/.cache/huggingface OMP_NUM_THREADS=2

# Artifact report (no weights): layers, lens layers/width, band, thinking and stop IDs
python scripts/sanity_check.py --model-config qwen3.5-4b
python scripts/sanity_check.py --model-config qwen3.6-27b

# 27B sanity checks: boot→Italy readout and exact-ID readout
python scripts/sanity_check.py --model-config qwen3.6-27b --check walkthrough
python scripts/sanity_check.py --model-config qwen3.6-27b --check exact-id

# 27B single-example smoke (its report is required by the paired-run gate)
python scripts/single_example.py --model-config qwen3.6-27b \
  --output-dir outputs/qwen36-27b-validation-004 \
  --device cuda --dtype bfloat16 --temperature 1.0 --seed 42 \
  --max-new-tokens 1024 --position-chunk-size 8 --smoke-checks \
  --prompt 'What is 17 multiplied by 19?'

# 27B five-scenario paired run, with paired-008's scenarios (prompt revision 5) and settings
python scripts/paired_feasibility.py run --model-config qwen3.6-27b \
  --scenarios data/paired-feasibility/scenarios_no_word_limit.json \
  --validation-report outputs/qwen36-27b-validation-004/model_validation.json \
  --output-dir outputs/paired-qwen36-placebo-001 \
  --device cuda --dtype bfloat16 --temperature 1.0 --seed 42 \
  --max-new-tokens 8000 --position-chunk-size 8

# Replay-only placebo scoring of an existing 4B run (no generation)
python scripts/paired_feasibility.py replay --model-config qwen3.5-4b \
  --source-run outputs/paired-008 --output-dir outputs/paired-008-placebo-replay
```

The paired-008 replay needs `outputs/paired-008` with its `samples/*/transcript.json` files; it was
not on the GPU machine when this was written. The same command was checked on `outputs/paired-006`.

## Next steps

The immediate priority is to obtain complete paired generations and investigate the resulting J-lens measurements. Further work will explore appropriate controls and how internal signals relate to evidence in CoT and final responses.

This is an exploratory investigation of model representations, not an attempt to develop a covert-persuasion classifier or detector.
