# MasterThesisDataScienceOnly
A scripting version of the flow and not end to end

## Index Builder Scripts

Two CLI helpers are available under `scripts/` to build retriever indexes from a JSONL corpus with `title` and `text` fields.

### BM25 Index

Build a BM25 index into a target folder:

```bash
python scripts/build_bm25_index.py \
	--documents-path llm_bayesian_reasoning/data/index_data/documents.jsonl \
	--index-path llm_bayesian_reasoning/data/index_data/bm25_index
```

Useful options:

- `--batch-size`: number of documents processed per batch while indexing
- `--limit`: cap indexing to the first `N` documents for smoke tests
- `--overwrite`: rebuild an existing non-empty index directory
- `--log-level`: one of `DEBUG`, `INFO`, `WARNING`, `ERROR`

### E5 Index

Build an E5 dense index into a target folder:

```bash
python scripts/build_e5_index.py \
	--documents-path llm_bayesian_reasoning/data/index_data/documents.jsonl \
	--index-path llm_bayesian_reasoning/data/index_data/e5_index \
	--model-name intfloat/e5-base-v2 \
	--device cuda
```

Useful options:

- `--model-name`: SentenceTransformer model identifier
- `--device`: device selection such as `auto`, `cpu`, `cuda`, or `cuda:0`
- `--batch-size`: number of documents processed per batch while indexing
- `--limit`: cap indexing to the first `N` documents for smoke tests
- `--overwrite`: rebuild an existing non-empty index directory
- `--log-level`: one of `DEBUG`, `INFO`, `WARNING`, `ERROR`

### Notes

- Both scripts fail by default if the target index directory already exists and is not empty.
- Pass `--overwrite` to remove the existing directory and rebuild the index.
- BM25 artifacts are written in the existing repo format with `bm25.pkl`, metadata, and JSONL files.
- E5 artifacts are written in the existing repo format with `embeddings.npy`, metadata, and JSONL files.
