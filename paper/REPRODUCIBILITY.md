# Reproducibility and handoff

## Scope of the results

This is a local validation study on 100 ordered HotpotQA questions, a 991-passage corpus and 200 gold-document occurrences. It does not reproduce the published EfficientRAG full-benchmark results. The original subset selection process and its independence from earlier training/development are not established. Fixed K=5 and K=15, budget-matched schedules and threshold sweeps remain future controls.

The four evaluated retrieval configurations are fixed K=10, query-only initialization held across hops, evidence-only updates initialized at K=10, and the full extension. Answer EM/F1 and actual Ollama usage are available only for fixed K=10 and the full extension. Missing ablation answer metrics are null in results.json and blank in results.csv, not zero.

## Verified versus reported artifacts

- Local fixed/full retrieval inputs: `/Users/awnishkumar/Desktop/extension files/efficientrag_answer_inputs_20261111T200130/`. Both JSONL hashes match their exported Colab manifests. Question, reference answer and oracle ID order match across methods.
- Final local answer results: `../local_answer_evaluation/full_100_resumed/`. This is the completed output directory. `full_100/` is an interrupted partial run and must not be treated as the final benchmark.
- Retrieval timing repeats: user-supplied Colab output, saved in Drive under `EfficientRAG_New_Run/dynamic_topk_runs/timing_repeats_20261111T192736184415Z`. Three repeats per method; mean/sample SD were reported at two decimal places. These logs have not been independently downloaded and recomputed here.
- Ablation results: user-supplied validated Colab output, saved under `EfficientRAG_New_Run/dynamic_topk_runs/ablations_20261111T194902407525Z`. Their metrics are transcribed in results.csv/json; raw ablation JSONL files remain in Drive.

## Configuration

Retriever: facebook/contriever-msmarco with FAISS. Labeler: microsoft/deberta-v3-small, checkpoint-8000. Filter: checkpoint-500. At most four retrieval rounds. Depth levels: 5, 10, 15. No extension-specific retraining. Adaptive configurations have a maximum possible 60 returned entries, versus 40 for fixed K=10; this is not a matched maximum-passage-budget comparison.

The deterministic policy and thresholds are in `../dynamic_topk_v2/src/efficient_rag/dynamic_topk.py` and `../dynamic_topk_v2/policy.json`. Novelty uses opaque passage IDs previously labeled CONTINUE, not oracle IDs. The controller changes actual retriever K and does not override the restored stopping logic.

The new final selector is `nonterminate_first_occurrence_id_dedup_max10_v1`: non-TERMINATE filtering, round-order traversal, exact-ID deduplication preserving first occurrence, maximum 10 documents. It is explicitly not asserted to reproduce the missing historical selector.

## Retrieval reproduction in the existing Colab environment

Restore the working repository, checkpoint files, corpus, evaluation data, model cache and index first. Use the existing `COMMON`, `RUNS`, `PACKAGE` and `ISOLATED` variables from the notebook. Run each method into a new output directory:

```python
subprocess.run(COMMON + ['--topk_policy', 'fixed', '--topk', '10',
    '--limit', '100', '--suffix', 'fixed-new',
    '--output-dir', str(RUNS/'fixed-new')], check=True)
subprocess.run(COMMON + ['--topk_policy', 'dynamic', '--topk', '10',
    '--limit', '100', '--suffix', 'dynamic-new',
    '--output-dir', str(RUNS/'dynamic-new')], check=True)
```

For ablations, use `../validation_next/colab_ablations.py`. It creates isolated source copies and permits only the intentional policy-module hash change. Never edit checkpoint files. The Colab compatibility patch is inference-only and should be retained consistently for all compared runs. Verify checkpoint keys, eval mode, fixed-mode trace parity and matching input/source hashes before comparing.

## Local answer evaluation

Open the installed Ollama app before running the script. The completed experiment used Ollama 0.32.14, llama3:8b Q4_0, temperature 0 and num_ctx=8192. Exact digest and code/input hashes are in the final provenance.json. The recovered request payload and prompt are retained. Each answer is appended immediately. Resume requires exact provenance agreement.

From the original workspace, the command for a NEW run is:

```bash
python3 outputs/local_answer_evaluation/run_local.py \
 '/Users/awnishkumar/Desktop/extension files/efficientrag_answer_inputs_20261111T200130' \
 --output outputs/local_answer_evaluation/new_evaluation --limit 100
```

Do not use the completed output directory for a new configuration. Existing historical baseline and checkpoint files remain unchanged. An agent-launched server initially failed Metal initialization; the native Ollama app succeeded. The full run was resumed after a tool interruption and an Ollama disconnect. The exact resume notes are stored with final answers.

## Metrics and timing interpretation

Strict EM uses normalized equality, not the recovered evaluator's substring-match field. Token F1 uses normalized answer overlap. Gold recall is document-ID recall, not sentence-level supporting-fact F1 or joint HotpotQA score. Final context IDs were checked against all 200 local answer records with no duplicates or cap violations.

Retrieval timing excludes model loading and answer generation, includes retrieval-pipeline processing and incremental output writing, and synchronizes CUDA at the measurement boundaries. The table uses only the three additional repeats, not the initial timing pair. Ablation timings are single-run observations and not used for timing significance claims.

Ollama token counts refer to successful requests only. They omit retrieval-model compute and failed requests. Prompt counts can reflect server context handling; the configured context window is 8192. Generation wall times were affected by load/caching and interruptions, and must not be reported as controlled end-to-end speedups.

## Paper and slides

`overleaf/main.tex` uses IEEEtran conference mode. Upload the contents of `overleaf/` to an Overleaf project and choose main.tex as the main document. The local PDF is compiled using Tectonic; Overleaf's compiler version can change line breaks slightly, so check page count after import. Do not change IEEE margins/font sizes to force a page count.

The presentation has exactly 10 slides: 1–3 cover existing work and the problem, 4–10 cover the extension, evaluation and limitations. Speaker notes include narration and sources. `video_script.md` provides a separate script. It is a narration draft, not a recorded video.

## Submission still requiring the author

Review authorship, student ID, wording, references, results and institution-specific requirements. Confirm whether the five-page limit includes references, and whether a specific IEEE conference variant is required. Record the presentation video at the professor's requested duration. Upload the approved manuscript to the correct Turnitin assignment and save its receipt/similarity report. No Turnitin submission or similarity check has been performed by this package. Follow the course's authorship and AI-assistance disclosure rules if applicable.
