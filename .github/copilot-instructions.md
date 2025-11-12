# GitHub Copilot Instructions

This document provides guidance for GitHub Copilot when working with the text-feature-span-extract repository.

## Repository Overview

This is a deterministic, end-to-end pipeline for extracting structured information from PDF invoices using native text layers (no OCR, no vision models, no brittle rules). The system uses feature-based machine learning operating on pdfplumber-extracted text with full provenance tracking.

**Core Philosophy:**
- Deterministic: Same PDF always produces identical output
- Feature-based ML: XGBoost models learn patterns automatically
- No vendor-specific rules: System generalizes across invoice formats
- Full provenance: Every extracted value includes page, bbox, and token span

## Technology Stack

- **Language:** Python 3.10+
- **Key Libraries:**
  - `pdfplumber`: Native text extraction from PDFs
  - `pandas` & `pyarrow`: Data processing with Parquet storage
  - `xgboost`: Machine learning for field classification
  - `typer`: CLI interface
  - `scipy`: Hungarian algorithm for field assignment
- **Testing:** pytest with custom determinism and idempotency tests
- **Code Quality:** black, isort, mypy, ruff

## Development Setup

### Installation

```bash
# Install package in editable mode
pip install -e .

# Install with development dependencies
pip install -e .[dev]
```

**Note:** There's a known dependency conflict with `doccano>=1.8.0` and `pandas>=2.0.0`. For core development, you can install dependencies manually excluding doccano if needed.

### Project Structure

```
src/invoices/          # Main package
  ├── ingest.py        # PDF ingestion and content-addressing (SHA256)
  ├── tokenize.py      # Text extraction with stable IDs
  ├── candidates.py    # Span proposal generation with features
  ├── decode.py        # Hungarian assignment + NONE bias
  ├── emit.py          # Contract JSON generation
  ├── report.py        # Statistics and metrics
  └── cli.py           # CLI entry points

tests/                 # Test suite
  ├── test_token_determinism.py
  ├── test_idempotency.py
  ├── test_candidate_bounds.py
  ├── test_contract_integrity.py
  └── test_review_queue.py

data/                  # Pipeline data (not in git)
  ├── ingest/          # Content-addressed PDFs
  ├── tokens/          # Token stores (Parquet)
  ├── candidates/      # Candidate spans + features
  ├── predictions/     # Contract JSON outputs
  └── review/          # Human review queue
```

## Build, Test, and Lint Commands

### Using Make (Preferred)

```bash
make help              # Show all available targets
make install           # Install package
make install-dev       # Install with dev dependencies
make test              # Run all tests
make test-golden       # Run golden tests only
make lint              # Run linting (ruff, mypy)
make format            # Format code (black, isort)
make clean             # Clean generated data
make pipeline          # Run full pipeline on seed_pdfs/
make determinism-check # Test determinism
```

### Direct Commands

```bash
# Testing
pytest -v                                    # All tests
pytest tests/test_token_determinism.py -v   # Specific test

# Linting
ruff check src/ tests/ scripts/              # Check with ruff
mypy src/                                     # Type checking

# Formatting
black src/ tests/ scripts/                   # Format code
isort src/ tests/ scripts/                   # Sort imports
ruff check --fix src/ tests/ scripts/        # Auto-fix issues
```

### Makefile Notes

The Makefile has hardcoded paths for Python in some targets (e.g., `/Library/Frameworks/Python.framework/Versions/3.11/bin/python3`). When making changes, be aware that these paths may need adjustment or use of `python3` directly.

## Code Style and Conventions

### Python Style

- **Line Length:** 88 characters (black default)
- **Type Hints:** Required for all function signatures (enforced by mypy)
- **Import Sorting:** Use isort with black profile
- **Naming:**
  - Functions/variables: `snake_case`
  - Classes: `PascalCase`
  - Constants: `UPPER_SNAKE_CASE`

### Key Patterns

1. **Content Addressing:** All PDFs stored by SHA256 hash
2. **Stable IDs:** `token_id`, `candidate_id` derived deterministically from content
3. **Version Stamping:** All outputs include version metadata for reproducibility
4. **Provenance Tracking:** Every extracted value includes source location (page, bbox, token_span)
5. **Idempotency:** Re-running stages skips already-processed work

### Contract Schema (data/predictions/*.json)

Every JSON output follows a strict contract with fields containing:
- `value`: Extracted/normalized text
- `confidence`: Model confidence score
- `status`: One of {PREDICTED, ABSTAIN, MISSING}
- `provenance`: {page, bbox_pdf_units, bbox_norm, token_span}

Fields include: `invoice_number`, `invoice_date`, `due_date`, `total_amount_due`, `vendor_name`, `vendor_address`, etc.

## Common Workflows

### Adding New Features

When adding new features to the candidate generation:
1. Update `candidates.py` to compute new feature columns
2. Increment `feature_version` in version stamps
3. Update tests in `tests/test_candidate_bounds.py`
4. Ensure determinism is maintained

### Working with Pipeline Stages

Each stage is independent and can be run separately:

```bash
invoicex ingest --seed-folder path/to/pdfs/   # Stage 1: Ingest
invoicex tokenize                              # Stage 2: Extract tokens
invoicex candidates                            # Stage 3: Generate candidates
invoicex decode                                # Stage 4: Assign fields
invoicex emit                                  # Stage 5: Generate JSON
invoicex report                                # Stage 6: Statistics
```

### Testing Determinism

Determinism is critical. Always test after changes:

```bash
make determinism-check  # Automated check
```

This runs the pipeline twice and diffs outputs to ensure identical results.

## Important Notes for Copilot

1. **Never Break Determinism:** Changes must maintain reproducible outputs
2. **Preserve Idempotency:** Don't break stage skipping logic
3. **Maintain Version Stamps:** Update versions when changing schemas/features
4. **Keep Provenance:** Every extraction must track source location
5. **No OCR/Vision:** This is a text-layer-only system by design
6. **Test Coverage:** All changes should maintain or improve test coverage
7. **Type Safety:** Maintain strict type hints (mypy enforced)
8. **Minimal Dependencies:** Only add dependencies if absolutely necessary

## Testing Guidelines

- **Run tests early and often** during development
- Tests may require PDF files in `seed_pdfs/` (currently redacted in repo)
- Golden tests validate against reference outputs in `tests/golden/`
- Idempotency tests ensure re-running stages produces same results
- Determinism tests ensure reproducibility across runs

## CLI Entry Points

Two equivalent entry points exist:
- `invoicex` - Primary entry point
- `run-pipeline` - Alternative entry point

Both support the same commands: `pipeline`, `ingest`, `tokenize`, `candidates`, `decode`, `emit`, `report`, `status`, `train`, `doccano-pull`, `doccano-import`, `doccano-align`.

## Data Persistence

- All intermediate data stored in `data/` directory (not in git)
- Content-addressed storage prevents duplication
- Parquet format for efficient storage and querying
- JSON for final contract outputs
- Artifacts copied to `artifacts/` for convenience

## Integration Points

- **Doccano:** For human annotation and model training (requires separate setup)
- **Review Queue:** Human corrections stored in `data/review/queue.parquet`
- **External Connectors:** System supports multiple source prefixes (fs:, dv:, sp:, dc:) for future integrations

## When Making Changes

1. **Understand the pipeline flow:** ingest → tokenize → candidates → decode → emit → report
2. **Check existing tests** for patterns and expectations
3. **Run linting** before committing: `make lint`
4. **Run tests** to ensure no breakage: `make test`
5. **Verify determinism** if touching core logic: `make determinism-check`
6. **Update documentation** if changing public interfaces
7. **Maintain backward compatibility** with existing data formats

## Avoiding Common Pitfalls

- Don't add randomness without fixing seeds from document SHA256
- Don't change token/candidate ID generation (breaks reproducibility)
- Don't modify contract schema without version bump
- Don't add file I/O in hot paths (use memory-efficient pandas operations)
- Don't assume PDF availability in tests (check for files first)
- Don't remove existing tests without very good reason

## Questions to Ask Before Changes

1. Will this change affect determinism?
2. Does this require a version stamp update?
3. Are existing data formats still compatible?
4. Do tests cover the new behavior?
5. Is the change truly necessary or is there a simpler approach?
