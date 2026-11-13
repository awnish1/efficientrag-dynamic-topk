# Changes and preservation

The user's `/Users/awnishkumar/Desktop/EfficientRAGF/` files were read only. All deliverables are in a separate package. No original checkpoint, dataset, result, or Drive file was changed. `src/` starts from official commit `47c38cf88a35a18aa3f1864836acd5cffb314992`.

Modified upstream files:

- `src/efficientrag_retrieve.py`: selectable fixed/query/dynamic/full modes; initial K and per-hop policy calls; actual K passed to search; decision/count/timing logs; empty-result handling; maximum-round validation; small sample limit; exclusive-create output to protect existing results. Existing Labeler, Tagger, Filter, and continuation logic remain unchanged.
- `src/conf/config.py`: restored notebook-recorded `microsoft/deberta-v3-small` tokenizer/base model path.
- `src/retrievers/embeddings/embedder.py`: restored notebook-recorded `facebook/contriever-msmarco` model mapping.

Added implementation:

- `src/adaptive_k/complexity.py`: deterministic question features and initial K.
- `src/adaptive_k/controller.py`: validated count-only feedback and bounded next-hop policy.
- `src/adaptive_k/policy.py`: per-question initialization, observed-ID novelty bookkeeping, mode/parameter validation, decision reasons.
- `src/adaptive_k/evaluation.py`: raw/Labeler-retained recall, all-gold coverage, cost/timing and K variation metrics.
- `src/adaptive_k/__init__.py`: package marker.
- `run_experiments.py`: A–E shared-model runner, resource/data checks, fresh in-memory FAISS index, hashes/versions, real baseline parity and optional historical trace comparison, new JSON/CSV outputs.
- `policy.json`: editable starting hyperparameters.
- `tests/test_adaptive.py`: lightweight policy, loop-equivalence, stopping, missing/empty-result, novelty, oracle-invariance, and metric checks.
- `synthetic_smoke.py`: executes the real loop with simulated adapters and explicitly labeled synthetic outputs.
- `Run_in_Colab.ipynb` and `requirements-colab.txt`: artifact recovery, inference-only setup, smoke/full evaluation and backup steps.

Added documentation/evidence: `AUDIT.md`, `README.md`, `ALGORITHM.md`, this file, audit inventories and source diff/hashes, unchanged original retrieval script under `backups/`, historical aggregate results, blocked experiment status, local preflight and synthetic validation outputs.

`extension.patch` covers retrieval integration and the new adaptive modules only; `audit/full_source.diff` includes the two restoration settings as well. The full `src/` tree also contains unchanged upstream files for standalone use. It does not contain recovered trained weights or datasets. Historical training modifications are documented and preserved as notebook cell text, not re-executed or asserted to form one consistent final training source.
