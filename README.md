---
license: other
license_name: upstream-terms-preserved
license_link: LICENSE.md
task_categories:
- tabular-classification
configs:
- config_name: default
  data_files:
  - split: train
    path: extract/public/train.csv
  - split: validation
    path: extract/public/eval.csv
  - split: test
    path: extract/private/holdout.csv
---

# Will a U.S. community water system exceed the lead action level in its next Lead and Copper Rule monitoring period? Forward prediction from prior 90th-percentile results

At the first day of a community water system's next Lead and Copper Rule tap-sampling period, will that period's reported 90th-percentile lead concentration exceed the 15 µg/L action level?

Derived from U.S. Environmental Protection Agency (EPA) — Enforcement and Compliance History Online (ECHO), SDWA dataset compiled from the Safe Drinking Water Information System (SDWIS); underlying LCR results are reported to EPA by state/tribal primacy agencies. This is not its official product.

Prediction time: First day, the SAMPLING_START_DATE, of the target PB90 monitoring period. Every prior lead or copper measurement, violation date and milestone date used is strictly before that date; current-vintage system attributes are time-invariant covariates.

Evaluation scope: `retrospective_research`. Public validation is development data. The downloadable labeled holdout is public; it is not cryptographically sealed. No final holdout benchmark score is claimed.

## Measured baseline

The existing runner was used unchanged on public validation. See `evaluation.json` for rows, controls, per-year results and limitations.

```json
{
  "full_model": {
    "auc": 0.7923036225587178,
    "brier": 0.021737287388064033
  },
  "majority_prior": {
    "auc": 0.5,
    "brier": 0.02345361286425233
  },
  "calendar_identity": {
    "auc": 0.6308999306866446,
    "brier": 0.028156884227565192
  },
  "persistence": {
    "auc": 0.7150794976540095,
    "brier": 0.023949405944861946
  }
}
```

## Limitations

- Retrospective single-vintage problem: the download is one quarterly SDWIS snapshot (2026Q3 per the source), so system attributes (population served, owner type, primary source, primacy agency) are current values applied to historical periods. They are treated as time-invariant covariates, and this is not evidence that those values were available at each historical decision instant.
- LCR summary reporting to SDWIS is uneven across primacy agencies; some states report only exceedances or only large/medium systems, so the population is whoever reported, not every U.S. community water system. Coverage by state must be reported and the label base rate interpreted per jurisdiction, not nationally.
- The prior period's 90th percentile is the system's own laboratory result, but its entry into SDWIS may post-date the target period's start. The feature is causally clean (all contributing measurements end before the decision instant) but the public record's operational availability at that date is not asserted.
- Monitoring schedule heterogeneity means the outcome horizon ranges from 6 months to 3 years, so rows are not directly comparable in difficulty; horizon length and period length are supplied as features and the horizon distribution must be reported.
- Rows for the same system are serially correlated and share a stable unobserved plumbing stock, so persistence is a strong baseline and effective sample size is well below row count; the chronological split with embargo limits but does not eliminate this.
- Rule drift inside the window: the December 2021 LCR revisions changed sampling site selection and added a lead trigger level, and the 2024 LCR Improvements lower the action level to 0.010 mg/L from 2027. Labels are therefore restricted to periods starting on or before 2023-12-31 under the 15 µg/L threshold, and results should not be extrapolated to the current rule.
- SDWIS carries no site-level or service-line material data, so the strongest physical predictors (lead service line count, pipe age, water chemistry such as pH and alkalinity) are unavailable; only aggregate 90th-percentile values and system descriptors are usable.
- ECHO's own terms disclaim warranty and discourage robotic querying; redistribution of derived tables is covered by the federal public-domain basis, but any state-supplied third-party material embedded upstream retains whatever rights its originator holds, and no worldwide dedication is claimed.
- Current-vintage administrative snapshot: system size, source, ownership and primacy codes are current values applied to historical periods; their historical availability is not proven.
- LCR summary reporting is uneven across primacy agencies, so the population is the reporting community systems, not every United States community water system.
- No full-row availability ledger exists; training label timing is unverified and this is retrospective research rather than a proven historical forecasting task.
- Rows for the same system are serially correlated and share an unobserved plumbing stock, so effective sample size is below row count.
- Lead service line counts, pipe age, pH and alkalinity are unavailable in SDWIS, so the strongest physical predictors are missing.
- Labels stop at target period start 2023-12-31 to remain under the 15 microgram per litre action level before the 2024 Lead and Copper Rule Improvements transition.
- Reference code lookups are scoped by VALUE_TYPE: LCR violations use the documented RULE_CODE 350 anchor plus VALUE_TYPE-scoped violation codes, and LSLR milestones use EVENT_MILESTONE_CODE. The construction asserts nonzero milestone records and at least one included row with a prior milestone rather than shipping a constant feature; state coding practice may still vary.
- Missing copper, prior-prior lead and target measurements remain missing; unknown outcomes are excluded, never converted to negative labels.
- Rows are system-period events uniquely identified by row_id; PWSID repeats across chronological splits by design, so system-level serial correlation remains and the qualification cluster is PWSID.
- No uncertainty interval; observed differences do not establish universal headroom

## Loading and reproduction

Verify all downloaded files with `verify_dataset.py`. CSV splits live under `extract/`; `meta.json` declares the target and identifiers. `MANIFEST.json` binds data, source snapshot, code and measured evidence.

To reconstruct without network source drift, extract `source-snapshot.tar.gz` into a new cache directory, install `reproduction/requirements.txt` in an isolated Python environment, then run `python reproduction/dataset_run.py --output NEW_EMPTY_DIRECTORY --source-cache CACHE_DIRECTORY`. Compare the qualification artifact digest with this manifest. This performs full validation and baseline evaluation and may need worker resources.

See `LICENSE.md` for source terms and preserved attribution.
