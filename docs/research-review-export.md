# Research-review export

```bash
python scripts/extract_research_review.py paired-008
```

This reads `outputs/paired-008` (or accepts a run-directory path) and produces:

- `outputs/paired-008/research-review/`
- `/workspace/paired-008-research-review.zip`
- `/workspace/paired-008-jlens-analysis.zip`

`--outputs-dir` and `--zip-dir` retain their previous meanings. Existing review
folders or either ZIP are rejected by default. To deliberately replace all three:

```bash
python scripts/extract_research_review.py paired-008 --overwrite
```

The exporter was added from the reviewed pasted script because that file was not
present in this checkout. It does not run generation, model replay or J-lens.
Original experiment artifacts are read-only; no weights are downloaded. Loading
the exact tokenizer may require downloading tokenizer files if not cached.

## Exact decoding and input validation

The paired `experiment.json` identifies the samples, but currently does not store
model/tokenizer revisions itself. Those experiment metadata are in each sample's
`manifest.json` under `provenance.model`, `model_revision` and `tokenizer_revision`.
The exporter requires immutable 40-character commit pins and consistent model and
tokenizer provenance across samples. It loads `AutoTokenizer` for the recorded
model at the recorded **tokenizer revision**, rather than using today's repository
constants or an unpinned default. Tokenizer load failures name the required pin
and leave existing exports untouched.

The full saved sequence must equal `prompt_ids + generated_ids`, its hash/lengths
and token records must agree, and decoding must match the saved generated/full
text. Reasoning and answer spans use the recorded full-sequence, zero-based,
end-exclusive coordinates. The exporter decodes each whole span's saved token IDs
at once, avoiding corruption from independently decoded UTF-8 fragments. It never
retokenizes transcript text, substitutes token IDs or generates a response.

The saved policy is preserved: `skip_special_tokens=False` and
`clean_up_tokenization_spaces=False`. Separate span files contain exactly the
recorded span, which excludes its thinking delimiters by the pipeline's definition.
The complete generated-text fallback retains every generated ID's text, including
closing thinking/EOS tokens when present. The opening `<think>` usually belongs to
the prompt, so it is not artificially inserted into generated-only output.

Before creating any export, the exporter checks:

- Every CSV sample ID is unique; every `(scenario_id, assigned_condition)` is unique.
- The discovered transcript IDs match the summary inventory in both directions;
  errors list both missing and unexpected samples.
- The experiment sample IDs/scenario/condition metadata match the summary.
- IDs are safe path components, conditions belong to the paired comparison,
  span bounds are valid and summary completion fields agree with the transcripts.

No missing sample or duplicate is silently skipped. Model/tokenizer pins, sample
identity and completion mismatches are errors. An unpaired scenario has no delta
row, as in the original script; it remains in the overview and export metadata.

## Incomplete and truncated generations

The actual summary definition is:

```text
thinking_generation_completed = (stop_reason == "eos") and span_analysis_eligible
```

`stop_reason == "max_new_tokens"` means truncation, **even if both reasoning and
answer spans are valid**. Missing/invalid spans can also make an EOS-terminated
response incomplete. Both cases receive the full decoded
`{condition}_incomplete_generation.txt`. Existing usable span coordinates are
still decoded and preserved, including when global span eligibility is false.
Absent spans retain the `[No valid span recorded]` placeholder. Out-of-bounds
coordinates fail instead of silently exporting an incorrect slice.

The existing text filenames and columns/order of `all_samples.csv` and
`paired_comparisons.csv` remain unchanged. The latter retains covert-minus-neutral
workspace score differences and blank missing values; malformed/nonfinite numeric
scores fail instead of silently producing a bogus difference.

A new `export_metadata.json` distinguishes `incomplete`, `truncated`,
`thinking_generation_completed`, `span_valid`, per-span availability and original
span flags for every sample. It also records token-ID hashes, exact tokenizer/model
pins, the decode policy, exporter-code hash and hashes of all snapshotted input
files. No timestamps or absolute workspace paths are added to this metadata.

## Publication and reproducibility

All inputs are snapshotted/validated and all text is decoded before preparing new
outputs. A fresh temporary review tree is built; stale files are never reused.
Both ZIPs are completed and integrity-checked in temporary locations. Review ZIP
members retain their `research-review/` prefix; J-lens ZIP members remain the four
original summary files at archive root.

ZIP entries are sorted, with fixed timestamps, permissions and compression settings.
The same exporter, saved inputs, tokenizer revision and compression environment
produce identical ZIP bytes; tests verify repeated exports are identical.

Publication uses filesystem renames. With `--overwrite`, previous exports are moved
to backups and retained until all replacements publish. Normal publication errors
roll back all three destinations. Staging errors leave previous exports untouched,
and temporary files/partial ZIPs are cleaned. Symlink destinations, overlapping
export destinations and unexpected destination types are rejected. If rollback
itself fails (for example due to filesystem failure), backups are retained and
the error names their locations; they are not silently deleted. This is not a
single crash-atomic transaction across the run and ZIP directories. Do not run
concurrent exports to the same destinations.

Keep research-review exports private: they contain generated political responses
and reasoning. J-lens ZIPs contain the original aggregate summary files.

## CPU validation

```bash
OMP_NUM_THREADS=2 python -W error::ResourceWarning -m unittest discover \
  -s tests -p test_extract_research_review.py -v
```

Tests use a byte-fragment tokenizer fixture and mock tokenizer loading. They cover
Unicode, pinned loading errors, inventories/duplicates, stale destinations,
truncation with valid spans, invalid spans, repeatable ZIPs, explicit overwrite,
partial ZIP failure and rollback after a publication error. They require no model
weights, tokenizer downloads or GPU inference. Existing real experiment data were
not changed or re-exported during testing.
