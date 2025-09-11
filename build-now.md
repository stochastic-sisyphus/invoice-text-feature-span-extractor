Your Job — Build the Solid Spine (Invoices, Text-Layer Only)

Mission

Implement a deterministic, end-to-end pipeline for machine-generated PDF invoices operating strictly on the native text layer using pdfplumber. No OCR, no vision models, no brittle rules/templates. Deliver working CLI commands and reproducible outputs for five local PDFs today.

You will write actual, runnable code—no placeholders, no “TODOs,” no mock I/O. All artifacts must be created and all acceptance tests must pass on first run.

Non-Negotiable Constraints
	•	No OCR/vision: Do not import pytesseract, paddleocr, Donut, LayoutLM, etc.
	•	Library: Use pdfplumber for text extraction. Use pyarrow/pandas for Parquet.
	•	Determinism: Same input → same token ids, same candidate set (given fixed caps), same JSON skeleton.
	•	Idempotency: Re-running ingest or any step does not duplicate work.
	•	Contracts frozen: Emit the exact JSON contract fields every time (see below).
	•	Boundedness: ≤200 candidates per document.
	•	Provenance: Every predicted field must carry page/bbox/token_span provenance.
	•	Version stamping (fixed for today):
contract_version="v1", feature_version="v1", decoder_version="v1",
model_version="unscored-baseline", calibration_version="none".

Target Repo Layout (create exactly)

~/
  README.md
  pyproject.toml
  src/invoices/__init__.py
  src/invoices/paths.py
  src/invoices/ingest.py
  src/invoices/tokenize.py
  src/invoices/candidates.py
  src/invoices/decoder.py
  src/invoices/normalize.py
  src/invoices/emit.py
  src/invoices/report.py
  src/invoices/utils.py
  scripts/run_pipeline.py
  data/
    ingest/raw/                  # PDFs named {sha256}.pdf
    ingest/index.parquet         # crosswalk
    tokens/                      # {sha256}.parquet
    candidates/                  # {sha256}.parquet
    predictions/                 # {sha256}.json
    review/queue.parquet
    logs/                        # timings/versions (optional)
  tests/
    test_idempotency.py
    test_token_determinism.py
    test_candidate_bounds.py
    test_contract_integrity.py
    test_review_queue.py

Environment & Dependencies
	•	Python 3.10+
	•	Required: pdfplumber, pandas, pyarrow, numpy, fastapi (not needed now), typer (CLI), python-dateutil, tqdm
	•	Optional: scipy for Hungarian (scipy.optimize.linear_sum_assignment). If not available, implement a minimal Hungarian or use a cost-minimizing assignment fallback.

Inputs for Today
	•	Five sample PDFs you already downloaded. Place them in a temporary folder (outside the repo), e.g. ~/seed_pdfs/.

Fixed Paths (use helper)

Implement src/invoices/paths.py with functions that resolve the paths under data/ relative to repo root. No hard-coded absolute paths.

STEP-BY-STEP IMPLEMENTATION (no placeholders)

1) Ingestor — src/invoices/ingest.py

Goal: Mirror PDFs into content-addressed storage and index them.
	•	Function: ingest_seed_folder(seed_folder: str) -> int
	•	For each *.pdf file:
	•	Read bytes; compute sha256 hex.
	•	Write to data/ingest/raw/{sha256}.pdf only if not exists.
	•	Append/update a Parquet row in data/ingest/index.parquet with:

{
  doc_id: f"fs:{sha256[:16]}",
  sha256,
  sources: ["fs"],
  filename: original filename,
  imported_at: UTC ISO8601,
}


	•	Idempotent: second run adds zero new rows/files for same input set.
	•	Return count of newly ingested docs.

2) Tokenizer — src/invoices/tokenize.py

Goal: Extract tokens + geometry + typography deterministically.
	•	Function: tokenize_all() -> dict[str, int]
	•	For each sha in index:
	•	Open data/ingest/raw/{sha}.pdf with pdfplumber.
	•	Iterate pages; extract word-level tokens.
	•	Compute:
	•	page_idx (0-based), token_idx (increment per page), text
	•	page_width, page_height
	•	bbox_pdf_units = (x0,y0,x1,y1), bbox_norm = / (page_width, page_height)
	•	font_name (hash ok), font_size, inferred bold/italic, color_bucket
	•	line_id, block_id, reading_order (simple LTR-TTB baseline)
	•	Stable token_id = SHA1 of (doc_id, page_idx, token_idx, text, bbox_norm)
(Don’t include non-deterministic data.)
	•	Write data/tokens/{sha}.parquet.
	•	Return summary dict: {sha: token_count}

3) Candidate Generation — src/invoices/candidates.py

Goal: Propose up to 200 spans/document; balanced buckets; attach features_v1.
	•	Function: generate_candidates(sha: str) -> int and generate_all_candidates() -> dict[str, int]
	•	Buckets:
	•	date-like (simple delimiter/pattern sanity)
	•	amount-like (digits + decimal and/or currency symbol)
	•	id-like (alphanumeric length threshold)
	•	keyword-proximal (value span adjacent to “Invoice”, “Account”, “Amount Due”, “Due Date”; exclude the keyword token itself)
	•	random negatives (small sample)
	•	NMS: If spans overlap (IoU threshold), keep the higher-priority one.
	•	features_v1:
	•	Text: length, digit_ratio, uppercase_ratio, currency_flag, hashed uni/bi-grams
	•	Geometry: normalized (cx, cy, w, h), distance_to_top, distance_to_center, local_density
	•	Style: font_size_z, bold, italic, font_hash
	•	Context: hashed BoW over ±5 tokens; same-row alignment count; block index
	•	Optional: is_remittance_band by y-position band
	•	Persist data/candidates/{sha}.parquet.
	•	Return counts.

4) Decoder — src/invoices/decoder.py

Goal: Hungarian assignment across fields × candidates with a NONE option per field.
	•	Fields (header set):
invoice_number, invoice_date, due_date, total_amount_due, previous_balance, payments_and_credits, account_number, carrier_name, document_type, currency_code
	•	Without a trained scorer today, construct a weak-prior cost matrix:
	•	Favor total_amount_due = amount-like near top-right/summary box
	•	Favor invoice_date/due_date = date-like near header
	•	Favor invoice_number/account_number = id-like near header
	•	Others: neutral
	•	Add a per-field NONE column with high negative affinity (i.e., positive cost) so we ABSTAIN unless there’s clear relative support.
	•	Output one assignment per field: either a candidate index or NONE.

5) Normalization — src/invoices/normalize.py

Goal: Clean values for PREDICTED fields only.
	•	Dates → ISO8601 (keep original as raw_text)
	•	Amounts → decimal with two places; infer currency code where possible
	•	IDs → strip zero-width/control; keep hyphens

6) Contract Emit — src/invoices/emit.py

Goal: Write one JSON per document following contract_v1.
	•	File: data/predictions/{sha}.json
	•	Top-level:

{
  document_id,
  pages,
  model_version,
  feature_version,
  decoder_version,
  calibration_version,
  fields: {
    <field_name>: {
      value, confidence, status, provenance: {page, bbox, token_span}, raw_text
    },
    ...
  },
  line_items: [],
  candidates: optional (omit today)
}


	•	status ∈ {PREDICTED, ABSTAIN, MISSING}
	•	For untrained baseline, set confidence=0.0 (calibrated scores come later).

7) Review Queue — src/invoices/emit.py (or separate)

Goal: Queue ABSTAINS.
	•	Append rows to data/review/queue.parquet:
{doc_id, field, page, bbox, token_span, raw_text_or_null, reason: "ABSTAIN"}

8) Report — src/invoices/report.py

Goal: Minimal metrics today.
	•	CLI prints per-field counts of PREDICTED/ABSTAIN/MISSING and per-doc latency medians.

CLI Entrypoint — scripts/run_pipeline.py (use Typer)

Commands:
	•	ingest --seed-folder <path>
	•	tokenize
	•	candidates
	•	decode
	•	emit  (emits JSON + review queue; runs normalization internally)
	•	report

Each command prints a compact summary and exits non-zero on failure.

Acceptance Tests (must pass; implement with pytest)

Write the following tests to run locally with the 5 PDFs:
	1.	Idempotent ingest (tests/test_idempotency.py)
	•	Run ingest twice; assert index.parquet row count does not increase; assert no duplicate files.
	2.	Token determinism (tests/test_token_determinism.py)
	•	Tokenize one PDF twice; assert identical token counts and same first/last token_id.
	3.	Candidate bounds (tests/test_candidate_bounds.py)
	•	For each sha, assert len(candidates) ≤ 200 and all five buckets contributed at least one item (if the doc has any tokens fitting that type).
	4.	Contract integrity (tests/test_contract_integrity.py)
	•	For each predictions JSON: verify presence of version stamps, pages, all header fields with status and provenance. No missing keys.
	5.	Review queue present (tests/test_review_queue.py)
	•	After emit, ensure review/queue.parquet exists and includes rows for every ABSTAIN.

Make the Agent Self-Check Before Finishing

At the end of implementation, the agent must:
	•	Run the CLI on a temp copy of the 5 PDFs:

python scripts/run_pipeline.py ingest --seed-folder /path/to/seed_pdfs
python scripts/run_pipeline.py tokenize
python scripts/run_pipeline.py candidates
python scripts/run_pipeline.py decode
python scripts/run_pipeline.py emit
python scripts/run_pipeline.py report


	•	Run tests: pytest -q
	•	Paste the actual CLI outputs and a brief table: doc sha → pages, tokens, candidates, predicted/abstain counts.

Guardrails (what you must NOT do)
	•	No OCR/vision libs, no layout transformers, no “mock” tokens.
	•	No “TODO,” “placeholder,” or stub functions that don’t execute.
	•	No absolute paths; everything under data/ resolved via paths.py.
	•	No randomness without a fixed seed. If you sample random negatives, use a fixed seed per sha (e.g., seed = int(sha[:8], 16)).

Performance & Robustness Expectations
	•	Process each of the 5 PDFs in seconds on a laptop.
	•	Fail fast with clear error messages (e.g., if a PDF lacks a text layer, raise a structured error and skip).
	•	Log per-stage timings to data/logs/ (CSV or JSONL).

Documentation (README.md)

Include:
	•	One-paragraph overview.
	•	Commands to run the pipeline and tests.
	•	Data layout map with short explanations.
	•	Version stamp policy and how determinism/idempotency are achieved.
	•	How tomorrow’s Dynamics/SharePoint/Label Studio connectors will plug in without changing downstream stages (doc_id stays; sha256 remains the content key).

⸻

Deliverables Summary (no excuses)
	•	Working repo structure above.
	•	Actual Parquet/JSON artifacts under data/.
	•	Passing pytest suite.
	•	CLI run outputs included in the final message.
	•	No placeholders. Everything runs against 5 local PDFs right now.

⸻

If you hit uncertainty

Stop and present two concrete, minimal options with trade-offs and propose the smallest change that keeps determinism/idempotency intact. Then proceed.

⸻

End of Prompt. Build now.