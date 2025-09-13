# Text-Layer Feature-Based Span Extractor for Invoices

A deterministic, end-to-end pipeline for extracting structured information utilizing the benefits of PDFs with native-text layer. 
To create a robust + maintainable system able to extract and process PDF invoices using only the native text layer. 
If I didn't repeat it enough (actually its all ive said til this point); NO BRITTLE FRAGILE CRYBABY RULES NO DUMB OCR 
This system must slay. She's quirky. shes not like othet girls. and id say it again!
No OCR, no vision models, no brittle rules. 
Yes robust feature-based machine learning operating on pdfplumber-extracted text with full provenance tracking.


## Quick Start

```bash
# Install dependencies
pip install -e .

# Run complete pipeline (uses contract_v2 by default)
python scripts/run_pipeline.py pipeline --seed-folder /path/to/seed_pdfs

# Or run individual steps
python scripts/run_pipeline.py ingest --seed-folder /path/to/seed_pdfs
python scripts/run_pipeline.py tokenize
python scripts/run_pipeline.py candidates  
python scripts/run_pipeline.py decode  # Uses v2 by default
python scripts/run_pipeline.py emit    # Uses v2 by default
python scripts/run_pipeline.py report

# Use legacy v1 contract explicitly
python scripts/run_pipeline.py decode --contract-version v1
python scripts/run_pipeline.py emit --contract-version v1
```

## Running Tests

```bash
# Install with dev dependencies
pip install -e .[dev]

# Run all tests
pytest -q

# Run specific test categories
pytest tests/test_idempotency.py -v
pytest tests/test_token_determinism.py -v
pytest tests/test_candidate_bounds.py -v
pytest tests/test_contract_integrity.py -v
pytest tests/test_review_queue.py -v
```

## Data Layout

```
data/
├── ingest/
│   ├── raw/              # Content-addressed PDFs ({sha256}.pdf)
│   └── index.parquet     # Crosswalk: doc_id, sha256, sources, filename, imported_at
├── tokens/               # Per-document token stores ({sha256}.parquet)
├── candidates/           # Per-document candidate spans + features_v1 ({sha256}.parquet)  
├── predictions/          # Contract JSON outputs ({sha256}.json)
├── review/
│   └── queue.parquet     # Human review queue for ABSTAIN/low-confidence decisions
└── logs/                 # Timing and version logs (optional)
```

### Data Schemas

**Index**: Document crosswalk with `doc_id` (external identifier), `sha256` (content hash), `sources` (list of origins), `filename`, `imported_at`

**Tokens**: Word-level text extraction with geometry (`bbox_pdf_units`, `bbox_norm`), typography (`font_name`, `font_size`, `bold`, `italic`), layout (`line_id`, `block_id`, `reading_order`), and stable `token_id` for reproducibility

**Candidates**: Proposed field spans (≤200 per document) with `features_v1` including text analysis, geometric properties, style characteristics, and contextual information

**Predictions**: Contract JSON with expanded field vocabulary (27 fields in v2 including `invoice_number`, `invoice_date`, `due_date`, `total_amount`, `vendor_name`, `customer_account`, etc.; 10 fields in legacy v1) containing `value`, `confidence`, `status` ∈ {PREDICTED, ABSTAIN, MISSING}, and full `provenance` (page, bbox, token_span)

## Determinism & Idempotency

**Determinism**: Same input PDFs always produce identical `token_id`s, candidate sets (given fixed caps), and JSON skeleton. Random components use fixed seeds derived from document SHA256.

**Idempotency**: Re-running any pipeline stage skips already-processed work. Ingestion deduplicates by SHA256. Tokenization, candidate generation, and emission check for existing outputs before processing.

**Version Stamping**: All outputs carry version stamps (`contract_version=v2` by default, `feature_version=v1`, `decoder_version=v1`, `model_version=unscored-baseline`, `calibration_version=none`) for reproducibility and change tracking. Legacy `contract_version=v1` available via `--contract-version v1` flag.

## Architecture & Extensibility

The pipeline is designed for seamless integration with external systems:

**Tomorrow's Connectors**: The `doc_id` field supports multiple source prefixes (`fs:`, `dv:`, `sp:`, `ls:`) while `sha256` remains the stable content key. New connectors (Dynamics/SharePoint/Label Studio) can be added without changing downstream stages.

**Model Evolution**: The `unscored-baseline` decoder will be replaced with trained LightGBM/XGBoost models using the same `features_v1`. Calibration and confidence thresholds can be updated via `calibration_version` without breaking contract compatibility.

**Review Integration**: Human corrections flow back through the review queue (`data/review/queue.parquet`) with full bbox provenance, enabling continuous model improvement and Label Studio integration.

## Pipeline Stages

1. **Ingest**: Content-addressed storage with SHA256 deduplication
2. **Tokenize**: Native text layer extraction with stable IDs  
3. **Candidates**: Balanced span proposals across buckets (date-like, amount-like, ID-like, keyword-proximal, random negatives)
4. **Decode**: Hungarian assignment with per-field NONE bias for confident abstention
5. **Emit**: Contract JSON generation with normalization (ISO dates, decimal amounts, clean IDs) 
6. **Report**: Per-field statistics and processing metrics

Each stage logs timing and version information to `data/logs/` for performance monitoring and audit trails.
