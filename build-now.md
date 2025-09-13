First things first, Royal Quest — Build the Solid Spine (Invoices, Text-Layer Only)

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
=======
DONE!

----

____

NEXT MISSION:

Label Readiness + LS Integration (Native-Text Invoices)

Role & Scope

You will add a label ingestion + alignment path for a production invoice span extractor that operates strictly on the native PDF text layer. Apply surgical, in-place edits only. Preserve repo structure, contracts, determinism, and idempotency. No forks, no duplicate “v2” files, no placeholders, no new dependencies.

Non-Negotiable Invariants
	•	Extractor: pdfplumber only. No OCR/vision, no templates/regex.
	•	Determinism & idempotency: same input → same tokens, same candidates (≤200), same JSON skeleton; safe re-runs.
	•	Provenance: every prediction carries {page, bbox, token_span}.
	•	Contracts (unchanged today):

contract_version="v1"
feature_version="v1"
decoder_version="v1"
model_version="unscored-baseline"
calibration_version="none"


	•	Normalization guard: single normalize_text() in src/invoices/normalize.py + NORMALIZE_VERSION constant + text_len checksum helper. All LS imports must validate against these.
	•	Text space: page markers are literal [[PAGE {i}]] inside the document string; all character indices live in this one normalized string.
	•	Empty-safe behavior: if there are zero annotations, commands still exit 0 and print deterministic zero-summaries.

Field Vocabulary (use these names)
	•	Singleton fields (one value per document):
invoice_number, invoice_date, due_date, issue_date, total_amount, subtotal, tax_amount, discount, currency, remittance_address, bill_to_address, ship_to_address, vendor_name, vendor_address, customer_name, customer_account, purchase_order, invoice_reference, payment_terms, bank_account, routing_number, swift_code, notes, tax_id, contact_name, contact_email, contact_phone
	•	Line-item fields (row-structured):
line_items: [ { description, quantity, unit_price, line_total, provenance, raw_text } ]
	•	LS label → contract key mapping (canonical):
	•	InvoiceNumber→invoice_number; InvoiceDate→invoice_date; DueDate→due_date; IssueDate→issue_date
	•	TotalAmount→total_amount; Subtotal→subtotal; TaxAmount→tax_amount; Discount→discount; Currency→currency
	•	RemittanceAddress→remittance_address; BillToAddress→bill_to_address; ShipToAddress→ship_to_address
	•	VendorName→vendor_name; VendorAddress→vendor_address; CustomerName→customer_name; CustomerAccount→customer_account
	•	PurchaseOrder→purchase_order; InvoiceReference→invoice_reference; PaymentTerms→payment_terms
	•	BankAccount→bank_account; RoutingNumber→routing_number; SWIFTCode→swift_code
	•	Notes→notes; TaxID→tax_id; ContactName→contact_name; ContactEmail→contact_email; ContactPhone→contact_phone
	•	LineItemDescription→line_items[n].description; LineItemQuantity→line_items[n].quantity; LineItemUnitPrice→line_items[n].unit_price; LineItemTotal→line_items[n].line_total

⸻

Acceptance Criteria (must all pass)
	1.	New CLI (Typer in scripts/run_pipeline.py):
	•	labels import --in data/labels/annotations.json
	•	labels align --all | --sha <sha>
	•	Empty input: exit 0 with deterministic “0 docs, 0 spans, 0 aligned” summary.
	•	Non-empty input: write
	•	data/labels/raw/{sha}.jsonl (rows: {sha, label, char_start, char_end, text, normalize_version}),
	•	data/labels/aligned/{sha}.parquet (candidates + y∈{0,1}),
	•	data/labels/index.parquet with aggregates {sha, n_gold, n_matched, n_unmatched_gold, normalize_version, label_set_hash}.
	•	Hard-fail with a clear message if NORMALIZE_VERSION or text_len checksum mismatches.
	2.	Alignment logic:
	•	Use character-range IoU on the normalized document string. Best match per gold span becomes y=1; other candidates for that label in the doc are y=0.
	•	Count UNMATCHED_GOLD when no candidate overlaps a gold span; do not drop these from metrics.
	3.	Candidate spans compatibility:
	•	Every candidate row exposes char_start, char_end in the normalized text space (in addition to page/bbox/typography). Stable ordering; cap ≤200.
	4.	Line-item assembly rule (deterministic):
	•	If grouping of line-item labels into rows is ambiguous, emit line_items: [] and queue those spans for review (no guessing).
	5.	Determinism:
	•	Repeating labels import + labels align on the same inputs produces byte-identical artifacts and identical index aggregates.
	6.	Synthetic test (real I/O, no network):
	•	tests/test_label_alignment.py: build a tiny normalized doc string with [[PAGE 1]]…, write 1–2 synthetic candidates and 1 gold span to the proper locations, run the same IoU matcher, and assert:
	•	best IoU candidate is y=1; others y=0;
	•	n_gold, n_matched, n_unmatched_gold counts are correct.
	7.	Vendor the LS task generator (prevent drift):
	•	Add tools/labelstudio/tasks_gen.py that imports normalize_text and NORMALIZE_VERSION from src/invoices/normalize.py, computes PDF SHA256, copies each PDF to labeled_pdfs/{sha}.pdf, and writes tasks.json with normalize_version + text_len checksum in each task’s data.
	•	Commit tools/labelstudio/labelsettings.xml (the exact interface used).
	•	Update README.md with a short LS workflow + normalization-guard note.

⸻

Plan & Impact (name exact files/functions)
	•	src/invoices/normalize.py — export normalize_text(), NORMALIZE_VERSION, and text_len() checksum helper.
	•	src/invoices/candidates.py — ensure per-candidate char_start,char_end exist in normalized space; add a small char_iou(a_start,a_end,b_start,b_end) helper.
	•	src/invoices/report.py — implement LS importer (JSON → data/labels/raw/*.jsonl) and data/labels/index.parquet writer (with label_set_hash of the active LS labels).
	•	scripts/run_pipeline.py — add labels import and labels align commands (empty-safe, deterministic summaries; normalization guard on non-empty).
	•	tests/test_label_alignment.py — minimal synthetic doc/candidates/gold verifying IoU, y labels, and unmatched-gold accounting.
	•	tools/labelstudio/tasks_gen.py — vendored task generator that imports the pipeline normalizer; writes tasks.json + labeled_pdfs/.
	•	tools/labelstudio/labelsettings.xml — committed LS schema.
	•	README.md — 6-line section describing LS workflow and guards.

⸻

Implementation Rules
	•	No new dependencies; use existing pandas/pyarrow/scipy stack.
	•	Use src/invoices/paths.py for all paths under data/ (no absolute paths).
	•	Stable seeds derived from int(sha[:8],16) if sampling is needed.
	•	Zero randomness in alignment and indexing; sort deterministically.
	•	Clear, single-purpose functions; no dead code; no TODOs.

⸻

Output Required from the Agent
	1.	Impact plan (5 lines max, listing exact files/functions you will touch).
	2.	Continuity check (bullets confirming invariants are preserved; call out and justify any unavoidable variance).
	3.	Changes Applied — return exactly:
	•	One atomic commit with message:

labels: add LS import+align, vendor task_gen, enforce normalize guard

Why:
- Ready the pipeline for incoming annotations without altering spine or deps
- Prevent normalization drift via version+checksum guard
- Lock mapping for singleton vs line-item fields per LS interface

What changed:
- run_pipeline.py — `labels import|align` (empty-safe, deterministic)
- report.py — LS importer → raw jsonl, index parquet
- candidates.py — normalized char spans + IoU helper
- normalize.py — normalize_text(), NORMALIZE_VERSION, text_len()
- tools/labelstudio/tasks_gen.py — repo-pinned task generator
- tools/labelstudio/labelsettings.xml — committed interface
- tests/test_label_alignment.py — synthetic IoU alignment test
- README.md — LS workflow + normalization guard

QA:
- All existing tests + new alignment test green
- Empty-data runs produce identical zero summaries
- Determinism verified on repeat align


	•	Verification transcript: show the key command outputs for:

python scripts/run_pipeline.py labels import --in data/labels/annotations.json
python scripts/run_pipeline.py labels align --all
pytest -q

Include the printed zero-summary on empty input and the deterministic rerun proof.

⸻

Notes to Enforce During Edits
	•	Do not alter existing CLI names for the spine (ingest, tokenize, candidates, decode, emit, report).
	•	Do not change contract_version defaults or field shapes in v1.
	•	Do not “guess” line-item grouping—emit line_items: [] when ambiguous and surface in review.
	•	Fail fast (non-zero) on normalization mismatches with a message including sha, observed_version, expected_version, and both text_len values.

⸻

Begin by in depth understanding of the repo currently. run tree command, and run the pipeline, to see the nuance and context. then give me the 5-line impact plan with exact paths. Then proceed in order.
>>>>>>> 8ee27ef (setup for labels)

⸻

/Browserbase/stagehand_usage /Browserbase/multi_session_guidance /Browserbase/browserbase_system
Label Integration + Contract_v2 (Native-Text Invoices)

ROLE & SCOPE

Implement now a label ingestion + alignment path and add a contract_v2 emit/decoder surface that supports the expanded Label Studio field set. Apply surgical, in-place edits only. Preserve v1 outputs byte-for-byte (default remains v1 unless --contract-version v2 is passed). No forks, no “fixed/v2/v3 copy files,” no placeholders, no new dependencies.

PERMISSIONS (ALLOWED)
	•	Edit only files listed in PLAN & IMPACT.
	•	Run repo commands (shell/Python) to build, verify, and show real outputs. Do not simulate.

PROHIBITIONS (HARD)
	•	Do NOT create/modify/require tools/labelstudio/tasks_gen.py (task gen is external on the annotator VM).
	•	Do NOT add synthetic tests or any scaffolding.
	•	Do NOT alter existing spine CLI names (ingest, tokenize, candidates, decode, emit, report).
	•	Do NOT add dependencies.
	•	Do NOT change v1 contract shapes, file formats, or defaults.

NON-NEGOTIABLE INVARIANTS
	•	Extractor: pdfplumber only. No OCR/vision/templates/regex.
	•	Determinism & idempotency: same input → same tokens, same candidates (≤200), same JSON skeleton; safe re-runs.
	•	Provenance: every prediction carries {page, bbox, token_span}.
	•	Contracts (unchanged today for v1):

contract_version="v1"
feature_version="v1"
decoder_version="v1"
model_version="unscored-baseline"
calibration_version="none"


	•	Document-text normalization guard: ensure src/invoices/normalize.py exports:
	•	normalize_text() with NORMALIZE_VERSION="doctext_nfc_newline_v1" (Unicode NFC + CR/LF→\n + strip)
	•	text_len() helper
All LS imports must validate normalize_version + text_len; importer hard-fails on mismatch (unless --allow-unnormalized, which only waives presence, not length drift).
	•	Text space: page markers are literal [[PAGE {i}]]; all char indices live in this one normalized string.
	•	Empty-safe: zero annotations still exit 0 and print deterministic zero-summaries.
	•	v1 outputs remain byte-for-byte identical when --contract-version v1 (default).

FIELD VOCABULARY (LOCKED FOR v2)

Singletons (0..1 per doc):
invoice_number, invoice_date, due_date, issue_date, total_amount, subtotal, tax_amount, discount, currency, remittance_address, bill_to_address, ship_to_address, vendor_name, vendor_address, customer_name, customer_account, purchase_order, invoice_reference, payment_terms, bank_account, routing_number, swift_code, notes, tax_id, contact_name, contact_email, contact_phone

Line items (row-structured):
line_items: [ { description, quantity, unit_price, line_total, provenance, raw_text } ]

Label Studio → contract_v2 mapping (canonical):
InvoiceNumber→invoice_number; InvoiceDate→invoice_date; DueDate→due_date; IssueDate→issue_date;   TotalAmount→total_amount; Subtotal→subtotal; TaxAmount→tax_amount; Discount→discount; Currency→currency;   RemittanceAddress→remittance_address; BillToAddress→bill_to_address; ShipToAddress→ship_to_address;   VendorName→vendor_name; VendorAddress→vendor_address; CustomerName→customer_name; CustomerAccount→customer_account;   PurchaseOrder→purchase_order; InvoiceReference→invoice_reference; PaymentTerms→payment_terms;   BankAccount→bank_account; RoutingNumber→routing_number; SWIFTCode→swift_code;   Notes→notes; TaxID→tax_id; ContactName→contact_name; ContactEmail→contact_email; ContactPhone→contact_phone;   LineItemDescription→line_items[n].description; LineItemQuantity→line_items[n].quantity;   LineItemUnitPrice→line_items[n].unit_price; LineItemTotal→line_items[n].line_total.

ACCEPTANCE CRITERIA
	1.	New CLI (Typer in scripts/run_pipeline.py)
	•	labels import --in data/labels/annotations.json [--allow-unnormalized]
	•	labels align --all | --sha <sha> [--iou 0.5]
	•	Empty input: print 0 docs, 0 spans, 0 aligned and exit 0.
	•	Non-empty: write
	•	data/labels/raw/{sha}.jsonl rows: {sha,label,char_start,char_end,text,normalize_version,text_len}
	•	data/labels/aligned/{sha}.parquet (schema below)
	•	data/labels/index.parquet (schema below)
	•	Hard-fail on version/length mismatch (show: sha, observed/expected version, observed/expected text_len).
	•	Unknown LS labels are accepted, included in labels_present, and counted; never crash.
	2.	Alignment logic (deterministic, half-open)
	•	Operate in the normalized doc string. Character IoU: IoU([a,b),[c,d)) = |∩|/|∪|.
	•	Best IoU ≥ --iou for same label ⇒ mark y=1; all other candidates of that label in that doc ⇒ y=0.
	•	Tie-breakers: higher IoU → smaller absolute center distance → smaller span length → lexicographic (char_start,char_end).
	•	If no candidate meets threshold: increment UNMATCHED_GOLD (do not drop).
	3.	Aligned schema (concrete)
	•	Columns: sha:str, label:str, cand_id:str, y:int8, char_start:int32, char_end:int32, token_span:list<int32>, page:int32, bbox:list<float32>, features_v1:struct<…>, normalize_version:str
	•	Sort rows by (sha,label,y DESC,char_start,char_end,cand_id) before write.
	4.	Index schema (concrete)
	•	Columns: sha:str, n_gold:int32, n_matched:int32, n_unmatched_gold:int32, labels_present:list<str>, label_set_hash:str, normalize_version:str
	•	label_set_hash = sha256 of sorted unique label names encountered.
	5.	Candidate spans
	•	Derive char_start,char_end deterministically from token offsets in the normalized doc string (preferred).
	•	Candidate cap remains ≤200; ordering stable.
	6.	Line-item rule
	•	If row grouping is ambiguous, emit line_items: [] and append a one-line diagnostic to data/labels/align_summary.jsonl (no guessing).
	7.	Contract_v2 output (emit/decoder)
	•	Add --contract-version {v1|v2} across decode and emit. Default stays v1.
	•	v2 must:
	•	expose the full field vocabulary above (singletons + line_items),
	•	keep v1 text-contract invariants (provenance, determinism),
	•	preserve v1 JSON shapes when --contract-version v1.
	•	No change to existing v1 byte output unless the flag selects v2.
	8.	Determinism proof (no synthetic tests)
	•	Re-running labels import + labels align on identical inputs produces content-identical aligned datasets.
	•	Prove by printing a sha256 over canonical CSV (sha,label,cand_id,y,char_start,char_end); do not rely on Parquet byte identity.

EXTERNAL TASK GENERATOR (ANNOTATOR VM)
	•	Do not add or require tools/labelstudio/tasks_gen.py.
	•	Importer must accept external tasks/exports. Expect per task: data.text (normalized), data.sha256, data.doc_id ("fs:{sha[:16]}").
	•	If normalize_version/text_len are absent, compute text_len locally and enforce repo normalizer; on mismatch, fail with a clear message.

PLAN & IMPACT (EDIT THESE ONLY)
	•	src/invoices/normalize.py — ensure normalize_text(), NORMALIZE_VERSION, text_len().
	•	src/invoices/tokenize.py — expose normalized doc builder with [[PAGE i]] + per-token char offsets (for char ranges).
	•	src/invoices/candidates.py — add char_iou(a_start,a_end,b_start,b_end); optional persisted char ranges.
	•	src/invoices/report.py — LS importer (JSON→data/labels/raw/*.jsonl), index writer (label_set_hash), and aligner.
	•	scripts/run_pipeline.py — add labels import, labels align; extend decode/emit with --contract-version {v1|v2}.
	•	README.md — short “Label Studio workflow” + normalization-guard note.
No other files.

IMPLEMENTATION RULES
	•	No new deps; use existing pandas/pyarrow/scipy.
	•	All paths via src/invoices/paths.py (no absolute paths).
	•	Stable seeds via int(sha[:8],16) only if absolutely needed (avoid).
	•	Zero randomness in align/index; deterministic sort; clear CLI help strings.
	•	No TODOs, no dead code, no scaffolding.

OUTPUT REQUIRED (TEXT ONLY)
	1.	Impact plan — ≤5 lines (exact files/functions you will touch).
	2.	Continuity check — bullets proving invariants intact; call out any unavoidable variance (none expected).
	3.	Changes Applied — single atomic commit:

labels+v2: add LS import+align, enforce normalize guard, and gated contract_v2 (default v1)

Why: enable supervised path + expanded field set without changing v1 spine or deps; prevent normalization drift; deterministic IoU; v1 preserved.
What changed: list touched files from PLAN & IMPACT only.
QA: empty-data zero-summaries; repeat import+align content-hash identical; emit --contract-version v1 byte-identical to prior v1.
	4.	Verification transcript — real outputs only

	•	python scripts/run_pipeline.py labels import --in data/labels/annotations.json
	•	python scripts/run_pipeline.py labels align --all --iou 0.5
	•	Print the canonical content-hash (sha256 over (sha,label,cand_id,y,char_start,char_end)).
	•	Show emit on one doc twice: once with --contract-version v1 (prove identical to pre-change v1), once with --contract-version v2 (show expanded fields present).

Begin by reading the repo, then print the 5-line impact plan with exact paths. Implement in order. Return real outputs + content-hash proof.