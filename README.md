# Text-Layer Feature-Based Span Extractor for Invoices

> NOTE! this repo is said pipeline in question - but you will never see it as a grown up! I am keeping it a secret (for #internal reasons (among others) (this grew up to pay to save my cats life when he got FIP and was in ICU and had cerebellar herniation same day, and I materialized both life saving treatment that is not accessible and procured funds for them to actually treat him whilst I drove 5 hr to obtain said cure) (aka ya I committed to 2 hefty invoices in one day) (my cat is cured now tho! 90 days of GS treatment later!) (carry on))
> ergo: public code = frozen at a pre-training state

A deterministic, end-to-end pipeline for extracting structured information utilizing the benefits of PDFs with native-text layer. 
To create a robust + maintainable system able to extract and process PDF invoices using only the native text layer. 
If I didn't repeat it enough (actually its all ive said til this point); NO BRITTLE FRAGILE CRYBABY RULES NO DUMB OCR 
This system must slay. She's quirky. shes not like othet girls. and id say it again!
No OCR, no vision models, no brittle rules. 
Yes robust feature-based machine learning operating on pdfplumber-extracted text with full provenance tracking.


## Quick Start

```bash
# Install package
pip install -e .

# Extract single PDF → JSON
make run FILE=path/to/invoice.pdf

# Or use CLI directly
invoicex pipeline --seed-folder path/to/pdfs/
# Alternative entry point:
run-pipeline pipeline --seed-folder path/to/pdfs/

# Run individual stages
invoicex ingest --seed-folder path/to/pdfs/
invoicex tokenize
invoicex candidates  
invoicex decode
invoicex emit
invoicex report

# Check pipeline status
invoicex status

# View outputs
ls artifacts/predictions/  # JSON outputs here
```

### Adding New Vendor Support

The system is designed for seamless vendor integration:

1. **No vendor-specific rules** - The feature-based ML approach learns patterns automatically
2. **Add training data** - Use [Doccano](https://doccano.github.io/doccano/) integration to annotate new vendor formats  
3. **Retrain models** - `invoicex train` updates XGBoost models with new patterns
4. **Deterministic outputs** - Same PDF always produces identical JSON across runs

### Debug Tips

```bash
# Test determinism
make determinism-check

# Run with verbose output
invoicex pipeline --seed-folder pdfs/ --verbose

# Check individual document processing
invoicex status
invoicex report --save

# Doccano annotation workflow
invoicex doccano-pull     # Pull from Doccano API
invoicex doccano-import --in export.json  # Import local export
invoicex doccano-align --all  # Align with candidates
invoicex train            # Train models on aligned data

# Inspect intermediate outputs
ls data/tokens/      # Text extraction
ls data/candidates/  # Proposed field spans  
ls data/predictions/ # Final JSON outputs
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

**Predictions**: Contract JSON with all header fields (`invoice_number`, `invoice_date`, `due_date`, `total_amount_due`, etc.) containing `value`, `confidence`, `status` ∈ {PREDICTED, ABSTAIN, MISSING}, and full `provenance` (page, bbox, token_span)

## Determinism & Idempotency

**Determinism**: Same input PDFs always produce identical `token_id`s, candidate sets (given fixed caps), and JSON skeleton. Random components use fixed seeds derived from document SHA256.

**Idempotency**: Re-running any pipeline stage skips already-processed work. Ingestion deduplicates by SHA256. Tokenization, candidate generation, and emission check for existing outputs before processing.

**Version Stamping**: All outputs carry version stamps (`contract_version=v1`, `feature_version=v1`, `decoder_version=v1`, `model_version=unscored-baseline`, `calibration_version=none`) for reproducibility and change tracking.

## Architecture & Extensibility

The pipeline is designed for seamless integration with external systems:

**Tomorrow's Connectors**: The `doc_id` field supports multiple source prefixes (`fs:`, `dv:`, `sp:`, `dc:`) while `sha256` remains the stable content key. New connectors (Dynamics/SharePoint/Doccano) can be added without changing downstream stages.

**Model Evolution**: The `unscored-baseline` decoder will be replaced with trained XGBoost models using the same `features_v1`. Calibration and confidence thresholds can be updated via `calibration_version` without breaking contract compatibility.

**Review Integration**: Human corrections flow back through the review queue (`data/review/queue.parquet`) with full bbox provenance, enabling continuous model improvement and Doccano integration.

## Doccano Integration

Complete annotation workflow for training custom models using [Doccano](https://doccano.github.io/doccano/):

```bash
# 1. Install and start Doccano
pip install doccano
doccano init && doccano createuser && doccano webserver

# 2. Generate tasks with normalization guards  
cd tools/doccano
python tasks_gen.py --seed-folder ../../seed_pdfs --output ./output

# 3. Import tasks.json to Doccano project and annotate
# 4. Export annotations and import
invoicex doccano-import --in path/to/export.json

# 5. Align labels with candidates using IoU
invoicex doccano-align --all --iou 0.3

# 6. Train XGBoost models on aligned data
invoicex train
```

**Normalization Guards**: Each task includes `normalize_version` and `text_checksum` to prevent drift between annotation and pipeline versions. Alignment validates text consistency before processing.

**Field Mapping**: Doccano labels map directly to contract fields (e.g., `InvoiceNumber` → `invoice_number`, `TotalAmount` → `total_amount`). Line items are spatially grouped when unambiguous.

## Pipeline Stages

1. **Ingest**: Content-addressed storage with SHA256 deduplication
2. **Tokenize**: Native text layer extraction with stable IDs  
3. **Candidates**: Balanced span proposals across buckets (date-like, amount-like, ID-like, keyword-proximal, random negatives)
4. **Decode**: Hungarian assignment with per-field NONE bias for confident abstention
5. **Emit**: Contract JSON generation with normalization (ISO dates, decimal amounts, clean IDs) 
6. **Report**: Per-field statistics and processing metrics

Each stage logs timing and version information to `data/logs/` for performance monitoring and audit trails.
