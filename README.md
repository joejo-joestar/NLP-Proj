# NLP Track B

This repository contains the Track B hallucination-analysis pipeline.

- Person 1 converts RAGTruth-style JSONL data, creates train/validation/test splits, formats samples for a forward pass, and saves model output artifacts.
- Person 2 consumes Person 1 artifacts and computes token-level metrics such as cosine drift, Mahalanobis distance, PCA deviation, logit-lens divergence, and composite scores.

## Repository Structure

```text
.
|-- .gitignore
|-- MarkingRubric.md
|-- README.md
|-- Midsem_Assignment_NLP_CS_F429.pdf
|-- NLP Lit Review.pdf
|-- NLP Research Project (Track B) — Work Distribution Plan.pdf
|-- Person 2 cosine drift module for Track B.pdf
|-- data/
|   |-- sample/
|   |   `-- raw_ragtruth_like.jsonl
|   `-- ragtruth/
|-- dataset/
|   |-- response.jsonl
|   `-- source_info.jsonl
|-- outputs/
|   |-- person1/
|   |-- person1_hf_20/
|   |-- person1_hf_smoke/
|   |-- person2/
|   `-- person2_smoke/
|-- scripts/
|   |-- convert_ragtruth_to_person1.py
|   |-- debug_person1_sample.py
|   |-- debug_person2_metrics.py
|   |-- fit_person2_stats.py
|   |-- run_person1_pipeline.py
|   |-- run_person1_workflow.py
|   `-- run_person2_metrics.py
|-- src/
|   `-- nlp_track_b/
|       |-- __init__.py
|       |-- person1/
|       |   |-- __init__.py
|       |   |-- config.py
|       |   |-- conversion.py
|       |   |-- data.py
|       |   |-- formatting.py
|       |   |-- io_utils.py
|       |   |-- model.py
|       |   |-- pipeline.py
|       |   `-- schemas.py
|       `-- person2/
|           |-- __init__.py
|           |-- artifacts.py
|           `-- metrics.py
`-- tests/
    |-- test_person1_conversion.py
    |-- test_person1_pipeline.py
    `-- test_person2_metrics.py
```

## Directory Guide

- `.gitignore`: excludes local data, generated artifacts, Python caches, and macOS metadata from Git.
- `MarkingRubric.md`: Track B marking notes and rubric summary already present in the GitHub repository.
- `README.md`: project overview, commands, structure, and GitHub push guidance.
- `data/sample/`: small synthetic JSONL fixture used by the Person 1 pipeline tests. This should be committed.
- `data/ragtruth/`: optional local location for converted RAGTruth JSONL files. This is generated data and is ignored by Git.
- `dataset/`: raw RAGTruth input files. This folder is intentionally ignored because it contains the full local dataset.
- `outputs/`: generated pipeline artifacts, split files, Hugging Face outputs, Person 2 metrics, and fitted stats. This folder is intentionally ignored because it is large and reproducible.
- `scripts/`: command-line entry points for conversion, pipeline execution, debugging, and metric generation.
- `src/nlp_track_b/person1/`: Person 1 package for data conversion, splitting, formatting, model forward passes, and output writing.
- `src/nlp_track_b/person2/`: Person 2 package for loading Person 1 artifacts and computing downstream token-level metrics.
- `tests/`: unit and smoke tests for conversion, Person 1 artifact creation, and Person 2 metrics.
- `*.pdf`: project reference and assignment documents. These are small enough for Git; commit them only if they belong in the public project repository.
- `__pycache__/` and `.DS_Store`: local generated files. These should not be committed.

## File Guide

### Person 1 Package

- `src/nlp_track_b/person1/config.py`: dataclass configuration for split ratios, model settings, and pipeline paths.
- `src/nlp_track_b/person1/conversion.py`: converts raw `response.jsonl` and `source_info.jsonl` records into the normalized Person 1 JSONL schema.
- `src/nlp_track_b/person1/data.py`: loads JSONL samples, validates required fields, normalizes text/context, creates grouped train/validation/test splits, and saves split manifests.
- `src/nlp_track_b/person1/formatting.py`: builds the model prompt, full input text, answer token list, and hallucination span alignment.
- `src/nlp_track_b/person1/io_utils.py`: saves per-sample model artifacts and run summaries to `outputs/`.
- `src/nlp_track_b/person1/model.py`: runs either the deterministic mock backend or a Hugging Face causal language model backend and returns hidden states, logits, token ids, and answer-token ranges.
- `src/nlp_track_b/person1/pipeline.py`: orchestrates the Person 1 flow from dataset loading through output artifact creation.
- `src/nlp_track_b/person1/schemas.py`: shared dataclasses for raw samples, formatted samples, token alignment, and model outputs.
- `src/nlp_track_b/person1/__init__.py`: package marker for Person 1 imports.

### Person 2 Package

- `src/nlp_track_b/person2/artifacts.py`: loads and validates Person 1 artifacts from JSON or PyTorch files, discovers artifact paths, and saves Person 2 metric artifacts.
- `src/nlp_track_b/person2/metrics.py`: computes cosine drift, Mahalanobis distance, PCA deviation, logit-lens divergence, normalizer stats, and composite scores over answer-token hidden states.
- `src/nlp_track_b/person2/__init__.py`: package marker for Person 2 imports.

### Scripts

- `scripts/convert_ragtruth_to_person1.py`: CLI wrapper for converting raw RAGTruth files into the Person 1 JSONL schema.
- `scripts/run_person1_pipeline.py`: runs the Person 1 pipeline on an already-converted dataset.
- `scripts/run_person1_workflow.py`: runs conversion and the Person 1 pipeline in one command.
- `scripts/debug_person1_sample.py`: prints one formatted Person 1 sample and tensor-shape details for debugging.
- `scripts/fit_person2_stats.py`: fits train-only Mahalanobis, PCA, and metric normalizer statistics from Person 1 artifacts.
- `scripts/run_person2_metrics.py`: computes Person 2 token-level metrics for one artifact or a directory of artifacts.
- `scripts/debug_person2_metrics.py`: inspects Person 2 metric shapes for a single Person 1 artifact.

### Tests

- `tests/test_person1_conversion.py`: verifies raw RAGTruth conversion into Person 1 JSONL rows.
- `tests/test_person1_pipeline.py`: verifies Person 1 split creation and output artifact fields using the small sample fixture.
- `tests/test_person2_metrics.py`: verifies Person 2 metric shapes, edge cases, and train-fitted statistic usage.

### Data and Output Files

- `data/sample/raw_ragtruth_like.jsonl`: small test fixture that should be committed.
- `dataset/response.jsonl`: raw response records for local conversion. It has 17,790 rows in this checkout and is ignored by Git.
- `dataset/source_info.jsonl`: raw source metadata records for local conversion. It has 2,965 rows in this checkout and is ignored by Git.
- `outputs/person1/`: generated Person 1 full-run artifacts.
- `outputs/person1_hf_20/`: generated Hugging Face artifacts for a 20-sample run.
- `outputs/person1_hf_smoke/`: generated Hugging Face smoke-test artifacts.
- `outputs/person2/`: generated fitted stats and Person 2 metric artifacts.
- `outputs/person2_smoke/`: generated Person 2 smoke-test artifacts.

## Common Commands

Run the full Person 1 workflow with the deterministic mock backend:

```bash
python3 scripts/run_person1_workflow.py --output-dir outputs/person1 --provider mock
```

Run the full Person 1 workflow with the Hugging Face backend:

```bash
python3 scripts/run_person1_workflow.py --output-dir outputs/person1 --provider hf --model-name distilgpt2
```

Convert only:

```bash
python3 scripts/convert_ragtruth_to_person1.py \
  --response-jsonl dataset/response.jsonl \
  --source-info-jsonl dataset/source_info.jsonl \
  --output-jsonl data/ragtruth/raw.jsonl
```

Run the Person 1 pipeline on a converted dataset:

```bash
python3 scripts/run_person1_pipeline.py \
  --dataset data/ragtruth/raw.jsonl \
  --output-dir outputs/person1 \
  --provider mock
```

Fit Person 2 train-only stats:

```bash
python3 scripts/fit_person2_stats.py \
  outputs/person1/model_outputs/train \
  --output outputs/person2/person2_stats.pt
```

Run Person 2 metrics:

```bash
python3 scripts/run_person2_metrics.py \
  outputs/person1/model_outputs/test \
  --stats outputs/person2/person2_stats.pt \
  --output-dir outputs/person2/metrics
```

Run the tests:

```bash
python3 -m unittest discover -s tests -v
```

## GitHub Push Guidance

The GitHub remote for this checkout is:

```text
https://github.com/Chirudeva-Reddy/NLP-Proj.git
```

Use this first-push flow when setting up from a fresh local copy:

```bash
git init
git remote add origin https://github.com/Chirudeva-Reddy/NLP-Proj.git
git fetch origin main
git checkout -b main origin/main
git add .gitignore README.md scripts src tests data/sample \
  "Midsem_Assignment_NLP_CS_F429.pdf" \
  "NLP Lit Review.pdf" \
  "NLP Research Project (Track B) — Work Distribution Plan.pdf" \
  "Person 2 cosine drift module for Track B.pdf"
git commit -m "Document project structure"
git branch -M main
git remote add origin <github-repo-url>
git push -u origin main
```

Push these:

- `.gitignore`
- `README.md`
- `src/`
- `scripts/`
- `tests/`
- `data/sample/`
- Project PDFs, if they are meant to be part of the repository

Do not push these through normal Git:

- `dataset/`: raw local dataset, ignored by request.
- `outputs/`: generated artifacts; this folder is about 17 GB in this checkout.
- `data/ragtruth/`: converted generated dataset files.
- `__pycache__/`, `.DS_Store`, and other local cache/metadata files.

GitHub's regular Git file limits are documented at <https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github>: files over 50 MiB trigger a warning, and files over 100 MiB are blocked. This checkout currently has these GitHub-limit risks:

- `outputs/person1/converted_raw.jsonl`: about 104 MB, blocked by the normal GitHub file limit.
- `outputs/person1/splits/train.jsonl`: about 73 MB, below the hard block but above the warning threshold.
- `outputs/person1_hf_20/model_outputs/**/*.json`: 20 files around 542-544 MB each, blocked by the normal GitHub file limit.
- `outputs/person1_hf_smoke/model_outputs/test/ragtruth_0.json`: about 543 MB, blocked by the normal GitHub file limit.

If large generated artifacts must be shared, use Git LFS, GitHub Releases, or external storage instead of committing them to normal Git history.
