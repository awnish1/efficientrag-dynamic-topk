# Query-Adaptive EfficientRAG

A minimal retrieval extension built on the recovered official source and your notebook evidence. Your supplied notebooks and backups are untouched. **The independent modules and simulated control flow are tested; real checkpoint inference remains unverified here.** Read `AUDIT.md` for the latest recorded experiment and missing resources.

## Run in Colab

1. Open `Run_in_Colab.ipynb` in a fresh T4 GPU runtime.
2. Upload the accompanying `query_adaptive_efficientrag.zip` when the notebook asks, and mount your Drive.
3. Install the small inference requirements, restart the session, and run artifact discovery. It selects checkpoint-8000 from `labeler_20260918_031818`, Filter checkpoint-500, and the saved `reconstructed_hotpot100` files. Missing or ambiguous artifacts cause a clear stop.
4. Generate fresh embeddings from the exact 991-passage corpus. Run the five-question smoke comparison and real fixed-K parity check.
5. Inspect the historical trace check. Then run the 100-question comparison and save the complete run to a new Drive folder.

No Labeler or Filter training is needed. Do not run the old notebook top-to-bottom or install upstream's full requirements (which also install training and model-serving systems). No LLM endpoint or API key is required for this retrieval experiment.

## Local independent checks

From this package directory:

```sh
python3 -m unittest discover -s tests -v
```

The shipped `results/synthetic_smoke/` contains an already executed synthetic smoke run. `synthetic_smoke.py` refuses to overwrite it; move it to a new backup directory if you want to rerun the script. The Colab runner also refuses existing output directories.

## Five configurations

| Experiment | Mode | Initial K | Per-hop changes |
|---|---|---|---|
| A | fixed | 5 | none |
| B | fixed | 10 | none |
| C | query | 3, 5, or 8 from question | none |
| D | dynamic | 5 | controller |
| E | full | 3, 5, or 8 from question | controller |

`policy.json` configures all thresholds, K values, and controller limits. Default K bounds are 3–10 and the maximum number of rounds is four. A common upper ceiling is 40 passages/question, but A can retrieve at most 20 and C depends on its initial K. This is a quality/cost comparison, not an equal-consumed-budget comparison. Duplicate returns count toward retrieval cost; recall uses unique IDs. No extra global-budget stop is added.

Runner invocation (replace placeholders with verified paths):

```sh
python3 run_experiments.py \
  --labeler LABELER_DIRECTORY --filter FILTER_DIRECTORY \
  --corpus CORPUS_JSONL --evaluation EVALUATION_JSONL \
  --embeddings EMBEDDING_DIRECTORY --policy-config policy.json \
  --limit 5 --max-rounds 4 --verify-baseline --output NEW_SMOKE_DIRECTORY
```

Add `--historical-reference SAVED_NEW_LABELER_RESULT_JSONL` to compare B's exact traces to your recorded run. For the full run, change `--limit` to 100 and use a new output directory. Model initialization and index construction are excluded from per-question latency; warmup is shared. `average_search_seconds` includes query embedding and index search. `average_pipeline_seconds` includes policy, retrieval, Labeler and Filter; `wall_seconds` additionally includes loop setup. One fixed-order run is exploratory: repeat runs with varied order before reporting precise latency differences.

## Policy rationale

The analyzer adds one point each for ≥16 words, ≥25 words, two capitalized spans after the first word, a relational marker after the first word, and a comparison marker. Score 0 chooses K=3, scores 1–2 choose 5, and scores ≥3 choose 8. Capitalization is a crude entity proxy, not named-entity recognition. Thresholds are interpretable engineering starting points, not fitted/calibrated complexity estimates.

After a hop, the controller uses only observed passage IDs and CONTINUE labels. If at least 75% of returned passages are repeats and none of the newly returned passages is useful, reduce K by 2. Otherwise, fewer than two useful passages or useful yield below 25% increases K by 2. Otherwise keep K. Clamp to [3,10]. Redundancy has priority over sparse evidence to avoid repeated expansion on an unchanged result set. Empty results propose expansion but the original no-CONTINUE stop prevents another search; this is logged as `applied=false`. The same applies to a proposal at the last hop.

The rule may miss evidence by reducing K, or increase cost without improving recall. These are empirical questions. Tune on a separate development set, lock parameters, then evaluate; do not claim improvements from the provided synthetic checks.

## Results and provenance

- `results/historical_recorded.csv` and `.json`: notebook-reported old/new results only.
- `results/comparison_status.csv`: A–E are marked blocked/unmeasured here; blanks are not zeros.
- `results/synthetic_smoke/`: synthetic traces and evaluator checks, explicitly not HotpotQA measurements.
- `results/local_preflight/`: exact missing local resources and package inventory.
- Each real run writes per-question JSONL, per-mode metrics JSON, `comparison.csv`, artifact/source hashes, versions, question IDs, policy parameters, K distributions, and stop/decision logs.

`within_question_k_changes` distinguishes true adaptation from merely different initial K across questions. If it is zero for D/E, the runner reports that lack of variation. “Retained” means Labeler tag != TERMINATE, matching the historical evaluator. There is no measured Filter-retained-document metric, answer EM/F1, or token usage.

## Apply to a separately recovered source checkout

`extension.patch` contains only retrieval integration plus the new policy/evaluation modules. It leaves your recovered Labeler/Filter implementations intact. Back up the target `src/` first, run `git apply --check /path/to/extension.patch`, then `git apply /path/to/extension.patch`. A mismatch should be inspected, not forced. The runnable bundled checkout already contains these changes and the two notebook-recorded model-path fixes. `CHANGES.md` lists all modified files; `audit/changes_manifest.json` has hashes.

The earlier whole-word reconstruction patch may not have survived into the final historical run. Resolve this with the saved Drive source and trace comparison before claiming historical baseline reproduction. The archived upstream script is used for independent fixed-K equivalence, which alone cannot establish equivalence with an unknown historical source version.
