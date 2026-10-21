# Saved replay diagnostics and summary counts

Existing artifacts can be analyzed **without regenerating responses, replaying the
model, loading a tokenizer, or recomputing J-lens readouts**. `transcript.json`
already saves token strings/decoded pieces and generation log-probabilities;
`replay_validation.json` saves replay log-probabilities and errors for each
generated index. The report joins these records by index, verifying IDs/hashes,
prediction positions, probabilities and coverage before writing output.

For an existing sample, on the machine holding `outputs/paired-004`:

```bash
python scripts/replay_report.py \
  --sample-dir outputs/paired-004/samples/SAMPLE_ID \
  --output-dir outputs/paired-004/replay-reports/SAMPLE_ID \
  --indices 10 25 100
```

Replace `SAMPLE_ID` and the example indices with actual values. Indices are
zero-based **generated-token** positions, not full-sequence positions. Omit
`--indices` to report all generated tokens. Requested indices must exist.

The new directory contains:

- `replay_tokens.csv`: index, token ID, decoded token text, tokenizer token string,
  generation log-probability, replay log-probability, absolute error in nats and
  a flag for errors **strictly greater than 0.5 nats**. Selected indices are also
  printed to the console. CSV quoting preserves whitespace/newlines in tokens.
- `replay_errors.svg`: error at **all** generated positions, the unchanged
  0.5-nat threshold line and red points at flagged positions. Hover tooltips
  identify the token/index/error. Plotting uses only the Python standard library.

Original artifacts are read-only. A new output directory is required to prevent
overwrites. Keep reports private: token-level text is generated-response content.
No real pretrained artifacts were available for this validation; reporting was
executed against synthetic saved-artifact fixtures.

## Sample counts and eligibility

The experiment manifest's scenario count, progress total, final summary scenario
and generation counts, and summary console sample count now derive from actual
scenario/sample records. Existing single-scenario/two-sample runs therefore report
one scenario and two generations; five-scenario runs report five and ten. Old
manifests with a stale `independent_scenarios: 5` can be re-summarized because the
new summary derives this count from sample records rather than copying that field.
The supported run sizes and paired design are unchanged.

Summary safeguards remain: the saved plan must match `prompts.jsonl`, every
planned sample must have complete, identity-validated artifacts, and concept,
model and sampling settings must agree. An interrupted run or missing sample
fails loudly instead of silently producing a selectively complete summary. Extra
sample directories absent from the plan are not included. No missing outputs are
imputed as zero.

For each span, only existing eligible score records contribute. Reasoning and
answer metrics are excluded when `span_analysis_eligible` is false; response-start
and user-prompt metrics remain available. Missing/empty span scores stay null.
A replay error flag is a numerical diagnostic, not an automatic sample exclusion.
The threshold, replay precision, generation settings and J-lens calculations have
not changed.

Denominators are explicit:

- `layers.csv` retains position counts for each sample/layer/span/concept mean.
- `samples.csv` adds `*_workspace_layers`, the layer count used in each workspace
  mean (zero if that score is absent).
- Condition summaries retain the actual sample count and add
  `mean_generated_tokens_n_samples` and `workspace_eligible_samples` by span/concept.
  Length and completion summaries describe all existing validated samples in that
  condition; span-score eligibility is tracked separately.
- Paired layer/workspace comparisons retain `n_pairs` and included scenario IDs,
  and add `n_samples`, `n_neutral` and `n_covert` for the **same matched subset** used
  for every reported mean/difference. A pair requires both eligible conditions;
  zero eligible pairs gives null means and zero counts. A one-sided eligible
  score is counted in condition eligibility but not in a paired contrast.

Recompute summaries from existing full saved artifacts into a fresh directory:

```bash
python scripts/paired_feasibility.py summarize \
  --run-dir outputs/paired-004 --output-dir outputs/paired-004-summary-review
```

This reads the saved lens JSONL and requires no GPU/model call. It does not repair
incomplete runs or missing per-token diagnostics; those are reported as errors.

CPU regression tests cover selected-index reporting, threshold boundary behavior,
CSV/SVG output, corrupt/misaligned artifacts, single-scenario and five-scenario
progress/summary counts, unmatched/ineligible spans and aggregate denominators.
The focused suites passed **20 CPU tests** (17 paired-feasibility and 3
replay-report tests). No GPU inference was run.
