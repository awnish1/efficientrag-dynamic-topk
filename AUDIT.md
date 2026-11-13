# Recovery audit — 11 November 2026

## Evidence and current state

Inspected the 252-cell `efficient.ipynb` (including stored outputs), the eight-cell `ds.ipynb`, and the HTML export. The HTML repeats the notebook's final metrics. **No `efficient1.pdf`, source checkout, model checkpoint, corpus JSONL, embeddings, or raw experiment JSONL was present in the supplied EfficientRAGF folder.** PDF claims could therefore not be independently checked. No existing project files were edited. The screenshots describe the extension, not additional completed experiments.

The official source was recovered from https://github.com/NIL-zhuang/EfficientRAG-official at commit `47c38cf88a35a18aa3f1864836acd5cffb314992`. This is a pinned recovery base, not proof of the exact revision used in Colab. Full cell text/output and a cell inventory are in `audit/`. Cell numbers below are **zero-based notebook positions**, not execution counts (which reset across sessions).

## Latest successful recorded run

Cells 230–232 rebuilt the original 100-question set using saved result IDs and recovered HotpotQA context. The first reconstruction matched only 84/935 saved title/text pairs. Cell 231 fixed the text/title reconstruction and reported **935/935 exact saved-document matches**, all **200/200 gold IDs**, **991 passages**, and **100 questions**. This validates only the 935 previously saved passages against historical text, not all 991 against an original corpus snapshot.

Cells 236–237 report regenerated `(991, 768)` Contriever embeddings. Cell 238 reports all three checkpoint paths existed then. Cells 244 and 246 successfully completed the old/new Labeler comparisons with `--labels 2 --topk 10 --test`, the same Filter checkpoint-500, and the four-round loop. Cell 247 records:

| Recorded run | Raw recall | Labeler-retained recall | Raw all-gold | Retained all-gold | Mean rounds |
|---|---:|---:|---:|---:|---:|
| Older Labeler checkpoint-9000 | 199/200 | 178/200 | 99/100 | 79/100 | 3.47 |
| Newer Labeler checkpoint-8000 | 197/200 | 186/200 | 97/100 | 88/100 | 3.92 |

These are **historical notebook outputs, not newly reproduced results**. End-to-end retrieval-stage timing printed 157.19 s and 179.07 s respectively; these are not isolated FAISS/search timings. The newer checkpoint is the latest successfully evaluated checkpoint, **not proven the best checkpoint by a selection criterion**. Training output reaches approximately epoch 0.79; do not call this completed full-data training.

Exact recorded paths:

- New Labeler: `/content/drive/MyDrive/EfficientRAG/EfficientRAG_New_Run/labeler_fixed_full_data/labeler_20260918_031818/checkpoint-8000`
- Old Labeler: `/content/drive/MyDrive/EfficientRAG_New_Run/labeler_20k_best_checkpoint_9000`
- Filter: `/content/drive/MyDrive/EfficientRAG_New_Run/filter_checkpoint_500`
- Recovered corpus/evaluation: `/content/drive/MyDrive/EfficientRAG_New_Run/reconstructed_hotpot100/{corpus.jsonl,valid.jsonl}`
- Latest result: `/content/drive/MyDrive/EfficientRAG_New_Run/hotpotQA-new-full-labeler8000-filter500-rebuilt.jsonl` (cell 248: 3,125,339 bytes).

Cell 249 subsequently shows a fresh `/content` session. In the initial inspected snapshot, cell 250's Drive search returned no output. During this audit the notebook changed externally: cell 250 was replaced by an unexecuted `drive.mount` cell (no output). The source hashes for the other 251 cells are unchanged. Neither snapshot proves backups were deleted or that Drive is currently mounted. There is no later successful retrieval result. `ds.ipynb` only records training process inspection/interruption and a shell-command SyntaxError.

## Source changes documented in the notebook

| File | Recorded changes and cells | Recovery treatment |
|---|---|---|
| `src/conf/config.py` | DeBERTa-v3-large local path → `microsoft/deberta-v3-small` (17, 129, 191, 220, 241) | Restored small-model setting |
| `src/retrievers/embeddings/embedder.py` | Contriever local path → `facebook/contriever-msmarco` (140, 243) | Restored model setting |
| `src/efficient_rag/model/model.py` | StableDropout → nn.Dropout for newer Transformers (26, 129) | Not reapplied; final run used Transformers 4.46.3, where upstream StableDropout was used successfully |
| `src/efficient_rag/labeler_training.py` | Trainer API/eval batch fixes (21); swapped metric fixes (34); safe gold-label token loss and weighting (36); joint-F1 selection (75); later fresh-source gold mask and CONTINUE-F2 metrics (164–165, 191); all-TERMINATE safe loss (193); 4.46 Trainer args/test data limit (200); FP16 (203); resumable checkpoints every 1000 steps (205); fixed Drive output path (209, 218) | Historical evidence retained; no training performed and no claim all patches coexist in the final source |
| `src/efficient_rag/filter_training.py` | Trainer API/eval batch changes (21), best-checkpoint saving (108), processing_class (200) | Historical evidence retained; inference uses existing Filter weights |
| `src/efficientrag_retrieve.py` | MAX_ITER 4→2 (59), restored 4 (69); whole-word subtoken reconstruction in Labeler and Filter (194), backed up (195) | Four-round safeguard preserved; whole-word provenance unresolved, see below |

The notebook repeatedly reclones repositories and restores selected files. Applying all historical cells in sequence would not reconstruct the final session reliably. Training scripts bundled from upstream are provenance/reference files, **not a recovered, validated training workflow**. Do not restart training from them.

**Whole-word uncertainty:** cell 194 successfully installed a roughly 50-line helper, and cell 195 saved it to `patched_source/efficientrag_retrieve.py`. Cell 216 conditionally clones a fresh official repository; cell 218 restores only the training source. The later cell 240 traceback points to `efficientrag_retrieve.py:285`, matching the upstream script's final line, suggesting the helper was absent in the final run. This is an inference, not proof. The delivered inference base uses upstream token reconstruction. The historical patch source is preserved in `audit/whole_word_patch_cell_194.txt`. Before claiming historical reproduction, compare the archived output traces and obtain/check the Drive source. `extension.patch` can be applied to a recovered source version after `git apply --check`, retaining that version's Labeler/Filter implementation.

## Exact retrieval behavior and extension points

In the pinned upstream source:

- `src/efficientrag_retrieve.py:50`: CLI `--topk`; line 182: function `top_k`; line 202: four-round loop; line 204: `retriever.search(query, top_k=top_k)[0]`.
- `src/retrievers/passage_retriever.py:62–72`: `Retriever.search`; line 65 passes K to the index. Scores returned as `top_scores` are discarded here.
- `src/retrievers/vector_index/faiss_index.py:40–52`: FAISS inner-product search; line 48 receives actual K and returns scores/IDs. No calibrated probability/confidence is available.
- `label_info` exposes token-derived information strings and CONTINUE/TERMINATE predictions; the classifier logits exist internally but are reduced to argmax labels.
- The Labeler decides which passages contribute to the next query; the Filter selects query tokens. The explicit loop stop is **no CONTINUE passages** or **MAX_ITER**. An empty Filter-generated query does not itself stop this implementation. FINISH also does not trigger a separate stop while CONTINUE passages remain. The extension preserves these details, including the final Filter call at the round ceiling.

Thus the screenshot's “Filter governs stopping” is a conceptual description, not the exact implementation. “Retained recall” in cell 247 includes documents whose Labeler tag is not TERMINATE; it does **not** measure Filter token retention. Reports use `labeler_retained_recall` to avoid overstating what was measured.

Only the retrieval loop needs adaptation; no Labeler, Tagger, Filter, retriever model, or training change is needed. The initial policy deliberately ignores scores because the wrapper does not expose them and raw inner products are uncalibrated. It uses CONTINUE counts and observed document-ID duplication only. K reaches the existing FAISS call directly.

## Backups recorded, not currently accessible

- Day-one tar archive `/content/drive/MyDrive/EfficientRAG_day1_backup.tar.gz` (cells 32–33).
- Small checkpoint-750 Labeler / checkpoint-500 Filter and early results (43, 58, 62).
- Older Labeler checkpoint-9000 and best-model copies (83, 86, 93).
- Later Filter checkpoint-8000 and associated experiments (115, 121, 156); **not the Filter used in the final comparison**.
- Nested `EfficientRAG/EfficientRAG_New_Run/patched_source/` contains recorded copies of config, Labeler training, retrieval, and later Filter training source (195, 200, 205, 209, 220).
- Training-data archive `EfficientRAG_official_training_data.zip` (189, 218).
- New Labeler checkpoints 7000 and 8000 were reported complete (222).
- Reconstructed corpus/evaluation and latest new-Labeler result (232, 248). The final old-control result was created locally; a separate final backup operation for it is not shown.
- Final embeddings/index were created in ephemeral Colab storage; the supplied evidence does not establish a final Drive backup of them.

## Compatibility and exact blockers

Earlier failures include PEFT/Accelerate import mismatch (23), NaN token loss (35), missing evaluation fields (150), incomplete notebook code (151), undefined label_info (185), removed StableDropout under Transformers 5.x (198), missing FAISS (212/214), missing IDs in the official archive (228), and the restored large-model path failing (240). Later cells resolve the last retrieval blockers. Broad force-reinstalls also caused NumPy/fsspec/requests conflicts; they are not repeated in the new notebook.

The final successful inference environment records Transformers **4.46.3**, Tokenizers **0.20.3**, FAISS **1.15.1**, and a Tesla T4. A complete final `pip freeze` and model-revision manifest are absent. Earlier Python/PyTorch versions are recorded, but should not be treated as a complete locked final environment. The new guide keeps Colab's CUDA-compatible PyTorch, pins the observed tokenizer/Transformer pair, checks imports, and logs the actual environment.

Local preflight (`results/local_preflight/preflight.json`) reports every required `/content/...` resource absent. This machine is macOS arm64; its Python lacks torch, transformers, faiss-cpu, spaCy, and sentencepiece. No Colab/Drive session is connected. Full GPU inference and research A–E results remain blocked. No retraining or large model download was attempted locally.

## Validation completed here

Dependency-free tests exercise the actual retrieval-loop AST with controlled adapters, including equivalence with the archived original loop, true search-call K changes, stop behavior, maximum rounds, empty results, no-score operation, bounds, per-question state reset, merged document-ID evaluation, and oracle invariance. A synthetic A–E smoke run saves JSONL/JSON/CSV under `results/synthetic_smoke/`; **none of its recall or timing values is a research result**.

The Colab runner additionally supports real-model fixed-K equivalence and comparison against the saved newer-Labeler traces. Those checks have not run here. No answer generator was run; answer EM/F1 and token consumption are unmeasured. Hyperparameters are uncalibrated starting hypotheses and must not be tuned on this 100-question evaluation set.
