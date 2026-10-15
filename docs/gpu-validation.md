# Step 0 validation and a clean RunPod environment

This implements Section 3 of `pilot_spec_26_10_2.md`. At review time **none of
its three checks was implemented**: the old script checked artifact dimensions
and optionally read one layer on a chat-formatted `2 + 2` prompt. That smoke test
is preserved. Its punctuation-heavy layer-15 output is not a failed reproduction
of the boot example: the prompt, position and layer coverage differ.

| Section 3 check | New command | Evidence produced |
| --- | --- | --- |
| Boot → Italy/Rome → euro walkthrough | `--check walkthrough` | Exact upstream raw prompt, position **-2**, unfiltered top tokens and one-based concept ranks at all fitted layers (0–30), explicit middle-layer ranks (12–29), upstream example layer indices (8,16,24,30). |
| Ten multihop items | `--check multihop` | First ten pinned upstream items; each intermediate's rank at every layer, minimum rank and best layers; top-10 hit rate and whether a strict majority hit. |
| Exact IDs versus `lens.apply` | `--check exact-id` | Same short boot prompt, same IDs, same -2 position; compares every fitted layer's full logit vector and final-model logits, reports maximum absolute differences and assertion tolerances. |

## Run on the GPU

From the repository root, in the clean environment described below:

```bash
export HF_HOME=/workspace/.cache/huggingface
export OMP_NUM_THREADS=2
python scripts/sanity_check.py --check walkthrough --device cuda --top-k 10
python scripts/sanity_check.py --check exact-id --device cuda
python scripts/sanity_check.py --check multihop --device cuda
# Or all three, loading model weights once:
mkdir -p outputs
python scripts/sanity_check.py --check all --device cuda > outputs/step0.log
```

No flag still runs only artifact checks. `--run-forward --device cuda` still
runs the original one-layer chat smoke test. `--device cpu` is supported for
all modes but loads the pretrained model in float32; allow substantial RAM and
runtime. `--help` lists the options. Output starts with the existing artifact
summary, followed by JSON records (including each multihop item); the overall
log is not a single JSON document. Redirecting stdout does not hide error output.

Each Step 0 check reports synchronized runtime (excluding model loading) and
peak allocated/reserved GPU bytes, including resident model weights. Reserved
memory can include allocator caches left by earlier checks. CPU runs report null
GPU metrics. Readouts are restricted to one position; full-sequence/full-vocabulary
arrays are never produced. The walkthrough performs a second forward to show
ordinary next-token logits at **-1** separately: its -2 model logits predict the
next prompt token, so do not mistake those for the completion distribution.

Inspect raw, unfiltered vocabulary results; punctuation is not removed to make
concepts look better. For Italy/Rome/euro and each multihop intermediate we report
single-token `word` and `" " + word` variants, taking the best rank and retaining
variant details. A missing single-token intermediate is an error. Ranks are
one-based, with tied values receiving `1 + count(logits > target_logit)`.

For multihop, tokenize `prompt + target` with offsets and retain only the prefix
before the first token overlapping the target. This handles a trailing space
merging into the target token. The target itself is **not** fed to the model.
The readout is the last prefix token, immediately before the target, as required
by upstream evaluation conventions. `--multihop-data PATH` allows another file
with the same schema, but the default is the vendored, pinned upstream file.

Multihop exits nonzero if no strict majority of intermediate concepts rank in
the top ten. Exact-ID comparison raises on any token/logit mismatch or assertion
failure; defaults are `--atol 0.001 --rtol 0.001`. These numerical tolerances are
not lens-quality criteria. The walkthrough reports its raw evidence and explicitly
**does not infer a quality pass** from a successful forward. Review middle-layer
Italy/Rome ranks and the separate euro completion rank before claiming success.
No fitting cell, generation, replay, pilot dataset, judging or analysis is run.

## Dependencies and what is actually known

| Component | Version / provenance |
| --- | --- |
| Python | 3.12 used for local validation; clean setup below assumes Python 3.12 is installed. |
| Torch | 2.14.1; previous project pin. CPU build tested here. Exact RunPod CUDA build/driver was not supplied. |
| torchvision | **0.29.1**; user reports this fixed `torchvision::nms` and enabled the RunPod forward. Added explicitly rather than inheriting an image installation. |
| Transformers | 5.5.4, pinned in the project. |
| Accelerate | 1.15.0, now pinned to the locally installed version. |
| jlens | `581d398613e5602a5af361e1c34d3a92ea82ba8e` (0.1.0). |
| Inspect AI | 0.3.280, existing pin; not used by these Step 0 checks. |
| Model / lens | Existing exact pinned revisions in `scripts/sanity_check.py`, unchanged. |

The working RunPod forward and torchvision fix are **user-reported results**.
They do not establish that these new Step 0 checks passed. The current Codex
machine has no GPU or RunPod connection details, so no new pretrained GPU
forward or GPU memory measurement was executed here. The original feasibility
report describes its earlier CPU-only investigation, not the current RunPod state.

## Reproduce without inherited image packages

Use a fresh virtual environment **without** `--system-site-packages`. The only
image prerequisites are Python 3.12 with venv support, Git, and a working NVIDIA
driver/device supplied by the host. Package installs must use that environment's
Python. Do not upgrade or uninstall packages in the RunPod image's base Python.

```bash
cd /workspace/manipulation-jspace-pilot
# Choose a new directory; do not reuse an environment with unknown packages.
env -u PYTHONPATH -u PYTHONHOME python3.12 -m venv /workspace/jspace-gpu-clean
source /workspace/jspace-gpu-clean/bin/activate
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
python -m pip install --upgrade pip
nvidia-smi
# This official wheel pair exists in the CUDA 13.0 index. It requires a host
# driver compatible with CUDA 13.0; check that before installing this pair.
python -m pip install --index-url https://download.pytorch.org/whl/cu130 \
  'torch==2.14.1+cu130' 'torchvision==0.29.1+cu130'
python -m pip install -r requirements.txt
python -m pip check
python - <<'PY'
import sys, torch, torchvision, transformers, accelerate, jlens
from torchvision.ops import nms
assert sys.prefix != sys.base_prefix, 'Not running in a virtual environment'
assert torch.cuda.is_available(), 'CUDA unavailable'
assert torch.cuda.is_bf16_supported(), 'GPU does not support bf16'
# Exercise a compiled torchvision operator, rather than just importing it.
print('NMS:', nms(torch.tensor([[0., 0., 1., 1.]], device='cuda'),
                  torch.tensor([1.], device='cuda'), .5).tolist())
print('Python:', sys.version, 'venv:', sys.prefix)
print('Torch:', torch.__version__, 'CUDA:', torch.version.cuda)
print('torchvision:', torchvision.__version__)
print('Transformers:', transformers.__version__, 'Accelerate:', accelerate.__version__)
print('GPU:', torch.cuda.get_device_name())
PY
python scripts/sanity_check.py --check walkthrough --device cuda
# Capture the resolved transitive versions and host driver for this actual run.
python -m pip freeze > /workspace/jspace-gpu-freeze.txt
nvidia-smi > /workspace/jspace-gpu-driver.txt
```

The CUDA 13.0 wheel pair's availability was checked against the official Torch
index; its installation/operation on RunPod was **not** tested here. If the host
cannot support CUDA 13.0, select an officially supported Torch/torchvision wheel
pair for that driver and record the difference; do not mix CPU and CUDA wheels.
The CUDA driver is a host prerequisite, not a Python dependency. Optional
`flash-linear-attention` and `causal-conv1d` are not needed for these checks;
the pinned Transformers implementation has a Torch fallback.

For a CPU-only clean environment, use the same steps but install Torch and
torchvision from `https://download.pytorch.org/whl/cpu`, omit the GPU/NMS probe,
and use `--device cpu` for a deliberate pretrained forward. A fresh isolated CPU install passed `pip check`, imports, a compiled NMS
operation, artifact checks and seven regression tests here. The CUDA guard also
correctly refused a pretrained walkthrough request on this machine. This does
not validate CUDA or the pretrained walkthrough. The requirement file pins
direct dependencies, not every transitive dependency. Preserve the generated
freeze and driver files with GPU results for a reproducible run.

Local regression command (no checkpoint download or pretrained model forward):

```bash
OMP_NUM_THREADS=2 python -m unittest discover -s tests -v
```

The tests use a tiny randomly initialized hybrid Qwen3.5 and synthetic lens to
check exact-ID equivalence, detect deliberate logit corruption, verify rank
calculation, target-boundary handling and multihop pass/fail aggregation.
