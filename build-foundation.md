You can make the “first part” truly solid today with five PDFs—no Dynamics/SharePoint required. Here’s exactly what to implement now (deterministic, end-to-end), what to stub for later, and how to operate it in plain language.

⸻

What to make SOLID right now (with 5 local PDFs)

These six thin layers are your spine. Build them so they run end-to-end, deterministically:
	1.	Ingestor → content-addressed storage + /ingest/index.parquet
	2.	Tokenizer (pdfplumber) → /tokens/{sha256}.parquet with stable token_ids
	3.	Contract Emitter → one predictions/{sha256}.json per doc using your frozen JSON contract
	4.	Candidate Gen v1 → ≤200 candidates/doc, balanced buckets, with features_v1
	5.	Assignment Decoder → Hungarian + per-field NONE bias (so you can ABSTAIN)
	6.	Normalizer + Review Queue → value cleanup + /review/queue.parquet

Everything else (true model scoring, calibration, connectors, UI, Dataverse) is safe to stub.

⸻

Minimal file layout (create once)

/ingest/raw/                 # PDFs stored as {sha256}.pdf
/ingest/index.parquet        # crosswalk: doc_id, sha256, sources, filename, imported_at
/tokens/                     # one {sha256}.parquet per doc (token store)
/candidates/                 # one {sha256}.parquet per doc (candidate spans + features_v1)
/predictions/                # one {sha256}.json per doc (contract_v1)
/review/queue.parquet        # rows for ABSTAIN/low-confidence
/logs/                       # timings + version stamps (optional)


⸻

Deterministic versions (freeze names today)
	•	contract_version = "v1"
	•	feature_version  = "v1"
	•	decoder_version  = "v1"
	•	model_version    = "unscored-baseline" (until you train)
	•	calibration_version = "none"

Stamp these into every JSON you emit.

⸻

EXACT tasks to code today

1) Ingestor (solid)

Purpose: mirror your 5 PDFs into content-addressed storage and index them.

Behavior:
	•	Read PDFs from a “seed” folder you point to.
	•	Compute sha256 of each file’s raw bytes.
	•	Write file to /ingest/raw/{sha256}.pdf (skip if it already exists).
	•	Append/update a row in /ingest/index.parquet with:
	•	doc_id = fs:{sha256[:16]} (you’ll swap to dv:/sp:/ls: later)
	•	sha256, sources = ["fs"], filename, imported_at (UTC ISO)

Acceptance for today:
	•	All 5 PDFs appear in /ingest/raw/ and /ingest/index.parquet.
	•	Re-running the ingestor makes no duplicate rows or files.

Natural language run:

“Ingest from ~/seed_pdfs. Use sha256 filenames under /ingest/raw/. Record crosswalk rows and skip anything already seen.”

⸻

2) Tokenizer (solid)

Purpose: extract native text tokens (no OCR) with geometry + typography.

Behavior for each /ingest/raw/{sha256}.pdf:
	•	Open with pdfplumber.
	•	For each page, extract tokens (words) with:
	•	text, page_idx, token_idx (increment per page)
	•	bbox_pdf_units = (x0,y0,x1,y1), page_width, page_height
	•	bbox_norm = bbox_pdf_units / (page_width,page_height) in [0,1]
	•	font_name (hash acceptable), font_size, inferred bold/italic flags, color_bucket
	•	line_id, block_id, reading_order (simple left-to-right, top-to-bottom is fine today)
	•	Create stable token_id:
	•	token_id = hash(document_id, page_idx, token_idx, text, bbox_norm)
(Use a stable hash; if you re-run, the same token gets the same id.)
	•	Write one Parquet file: /tokens/{sha256}.parquet

Acceptance for today:
	•	Two consecutive runs on the same file give identical token counts and the same first/last token_id.
	•	pages count in your summary matches the actual PDF.

Natural language run:

“Tokenize the 5 ingested PDFs. For each, write /tokens/{sha256}.parquet. Report page and token counts.”

⸻

3) Contract Emitter (solid)

Purpose: always produce the same JSON skeleton, even if the model abstains.

For each doc, emit /predictions/{sha256}.json with:
	•	document_id, model_version, feature_version, decoder_version, calibration_version, pages
	•	fields object containing all header fields:
invoice_number, invoice_date, due_date, total_amount_due, previous_balance, payments_and_credits, account_number, carrier_name, document_type, currency_code
Each field has:
	•	value (string or number when predicted; null if ABSTAIN/MISSING)
	•	confidence (0–1; for the untrained baseline, use 0.0 unless obviously dominant)
	•	status ∈ {PREDICTED, ABSTAIN, MISSING}
	•	provenance: {page, bbox, token_span} (token ids or index range)
	•	raw_text (verbatim string from PDF for the selected candidate)
	•	line_items (present as empty array)
	•	candidates (optional top-k per field; skip today if you like)

Acceptance:
	•	JSON validates and includes all required fields for every doc, even if status=ABSTAIN.

Natural language run:

“Emit contract_v1 JSON per document, even if everything is ABSTAIN, with version stamps and pages.”

⸻

4) Candidate Generation v1 (solid)

Purpose: propose plausible spans; not rules—just proposals.

Per document (cap at ≤200):
	•	Buckets:
	•	date-like (regex or simple parser sanity: contains month/day/year separators)
	•	amount-like (digits+decimal and/or currency symbol)
	•	ID-like (alphanumeric length threshold; few punctuation)
	•	keyword-proximal (take the value span nearest/right/under keywords such as “Invoice”, “Account”, “Amount Due”, “Due Date”; never use the keyword token as the candidate)
	•	random negatives (small sample to help later training)
	•	NMS overlap filter: drop near-duplicate spans (IoU high → keep one).
	•	features_v1 per candidate:
	•	Text: length; digit%; uppercase%; currency flag; hashed uni/bi-grams
	•	Geometry: normalized center/width/height; distance to top; distance to center; local token density
	•	Style: font_size z-score per page; bold/italic; font hash
	•	Context: bag-of-words hash of ±5 neighboring tokens; row alignment count; block index
	•	Optional: is_remittance_band for bottom strip

Write: /candidates/{sha256}.parquet (one row per candidate, with features_v1)

Acceptance:
	•	Each doc has ≤200 candidates; each bucket contributes; NMS is active.

Natural language run:

“Generate up to 200 candidates per doc from the token store, balance the buckets, remove overlaps, and save with features_v1.”

⸻

5) Assignment Decoder (solid, untrained)

Purpose: pick ≤1 candidate per field or choose NONE (→ ABSTAIN).

Do this now (without a trained scorer):
	•	Build a Hungarian assignment across fields × candidates with a NONE column per field.
	•	Score matrix today = weak priors (just to run the pipeline):
	•	total_amount_due: amount-like near top-right or summary box gets slight preference
	•	invoice_date/due_date: date-like near header gets slight preference
	•	invoice_number/account_number: ID-like near header gets slight preference
	•	all others: neutral
	•	Add a per-field NONE_bias (a threshold knob). Start high so you ABSTAIN unless a candidate looks obviously better than peers.
	•	Output one assignment per field:
	•	If NONE chosen → status=ABSTAIN, confidence=0.0, value=null
	•	Else → attach candidate’s provenance and raw text; confidence=0.0 for now (true probabilities come later)

Acceptance:
	•	Decoder emits exactly one row per field (PREDICTED or ABSTAIN). No candidate is reused.

Natural language run:

“Decode with a cautious NONE bias. If nothing clearly stands out, ABSTAIN. Write the result back into the JSON contract.”

⸻

6) Normalizer + Review Queue (solid)

Purpose: clean selected values, and stage work for humans.

Normalization (only for PREDICTED fields):
	•	Dates: parse as printed → output ISO 8601 in value, keep original in raw_text
	•	Amounts: parse to decimal with two places; infer currency when symbol present and store currency code
	•	IDs: strip zero-width/control chars; preserve hyphens

Review queue (today = all ABSTAINS):
	•	Write /review/queue.parquet with rows:
{doc_id, field, page, bbox, token_span, raw_text (if any), reason: "ABSTAIN"}

Acceptance:
	•	Any predicted values are normalized consistently; ABSTAINS appear in the queue.

Natural language run:

“Normalize PREDICTED values; queue every ABSTAIN for review with page and bbox so a human can click it later.”

⸻

Quick smoke tests (do these after each step)
	•	Idempotency: run the ingestor twice; index size should not increase.
	•	Determinism: re-run tokenizer; token counts and a sample of token_ids must match.
	•	Boundedness: candidate count ≤200 every time.
	•	Contract integrity: every predictions JSON has the full field list, version stamps, pages, and either PREDICTED or ABSTAIN for each field.
	•	Latency log (optional): write per-doc timings so you can baseline.

⸻

What to explicitly STUB until tomorrow
	•	Real connectors: Dataverse/SharePoint/Label Studio (just keep the interface boundaries: how to supply doc_id, where sources[] would be appended).
	•	True scorer + calibration: LightGBM/XGBoost training; Platt/isotonic curves; real confidences.
	•	Low-margin logic: ε-band needs calibrated probabilities.
	•	Review UI: start with the queue file; UI comes later.
	•	Dataverse tables: document schemas now; implement writes after the baseline is clean.
	•	Line items: keep line_items: [] for now.

⸻

“Run this pipeline” in plain language (no code)
	1.	Ingest
“Take the PDFs in ~/seed_pdfs. For each file, compute sha256, copy it to /ingest/raw/{sha256}.pdf, and write a row to /ingest/index.parquet. If a file with the same sha256 exists, skip it.”
	2.	Tokenize
“For each entry in /ingest/index.parquet, open /ingest/raw/{sha256}.pdf with pdfplumber. Extract tokens and typography per page. Create stable token_ids. Save to /tokens/{sha256}.parquet.”
	3.	Generate candidates
“From /tokens/{sha256}.parquet, propose up to 200 candidate spans balanced across date-like, amount-like, ID-like, keyword-proximal, and a few random negatives. Apply overlap suppression. Save to /candidates/{sha256}.parquet.”
	4.	Decode
“Build a cost matrix from weak priors and run the Hungarian assignment with a high NONE bias. For each field, either pick a candidate or choose NONE (→ ABSTAIN).”
	5.	Normalize & Emit
“Normalize any selected values (dates→ISO, amounts→decimal+currency, IDs cleaned). Write /predictions/{sha256}.json that includes versions, pages, and full provenance.”
	6.	Queue for review
“Write /review/queue.parquet with every ABSTAIN, including page and bbox so a reviewer can later jump to the exact region.”
	7.	Report
“Print counts per field: PREDICTED vs ABSTAIN vs MISSING and timing per doc.”

⸻

Today’s acceptance checklist (all must be green)
	•	/ingest/index.parquet lists the 5 PDFs; re-run adds nothing.
	•	/tokens/*.parquet exist; token counts and sample token_ids are identical across two runs.
	•	/candidates/*.parquet exist; each doc ≤200 candidates; buckets represented.
	•	/predictions/*.json exist; every file has all header fields with status and provenance; versions present.
	•	/review/queue.parquet exists and lists all ABSTAINS with page+bbox.
	•	A short report prints per-field counts and average latency.

⸻

Tomorrow: slide in the sources without breaking anything
	•	Replace fs:{sha256[:16]} with real dv:/sp:/ls: doc_ids while keeping sha256 as the stable content key.
	•	Append additional source rows for the same sha256 in /ingest/index.parquet (sources[] becomes multi-valued).
	•	Everything downstream (tokens → candidates → decode → predictions) keeps working untouched because it keys by sha256 and reads document_id from the index.

⸻

Why this is “solid”
	•	Deterministic: same input → same token IDs, same candidates, same JSON skeleton.
	•	Idempotent: you can safely re-run any step.
	•	Composable: tomorrow’s connectors and next week’s model drop in without changing the contract or token store.
	•	Auditable: every prediction carries provenance, versions, and can be traced to exact tokens.
---