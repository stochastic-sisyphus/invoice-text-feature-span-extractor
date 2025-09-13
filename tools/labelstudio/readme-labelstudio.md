these are the files used on the data label side. the annotator is using these exact files to label data to train my pipeline on. shes using label studio. 
Integrate Label Studio character-span annotations into the existing invoice text-layer span extractor pipeline without breaking continuity. Add in-place commands to: import LS JSON → align spans to existing candidates → materialize a training dataset. Preserve determinism, idempotency, file layout, contracts, and tests.
Add a Label Studio label ingestion + alignment path that:
	1.	Imports LS exportType=JSON into data/labels/raw/{sha}.jsonl (1 row per labeled span).
	2.	Aligns each labeled character span (normalized text space, doc-level with your [[PAGE i]] markers) to existing data/candidates/{sha}.parquet via character-range IoU, producing data/labels/aligned/{sha}.parquet with y ∈ {0,1} per candidate.
	3.	Builds a lightweight dataset index data/labels/index.parquet with per-doc counts (n_gold, n_matched, n_unmatched_gold) and stamps normalization + label-set hash.
    4.  Enforces shared normalization between task text and pipeline text via NORMALIZE_VERSION guard; hard-fail on mismatch.
	1.	Survey (no edits yet)
Name exact files and functions you will touch (5 lines max):
	•	src/invoices/report.py: add LS importer and dataset index writer.
	•	src/invoices/candidates.py: expose/read candidate char ranges if not already explicit; add tiny alignment helper.
	•	src/invoices/normalize.py: export/centralize normalize_text() and NORMALIZE_VERSION.
	•	scripts/run_pipeline.py: add labels import, labels align, train, eval Typer commands; wire to above.
	•	tests/test_label_alignment.py: minimal synthetic test (keep fixtures tiny).
	2.	Continuity check
Re-state invariants and confirm no file renames, no new deps, no contract drift. If a deviation is unavoidable, propose the smallest local change and how it’s contained.
	3.	Implement (surgical, in-place)
	•	Importer: read LS JSON; write data/labels/raw/{sha}.jsonl with {sha, label, char_start, char_end, text, normalize_version}.
	•	Alignment: IoU over [start,end) character ranges in the normalized doc text; best-match per gold; mark positives; all other candidates of that label in the doc become negatives; count UNMATCHED_GOLD but do not drop.
	•	Dataset index: data/labels/index.parquet with {sha, n_gold, n_matched, n_unmatched_gold, normalize_version, label_set_hash}.
	•	Determinism: sort rows with a stable key; avoid time-varying fields; seed any randomness by int(sha[:8], 16) if needed.
	4.	Tests
Run full suite. Add tests/test_label_alignment.py only; keep it deterministic and minimal.
	5.	Self-audit
a traceability matrix: edited lines → rationale (original invariant or new task) → covering test(s).
	6.	Verification
working real command outputs (lines that matter): counts written, index summary, determinism proof (hash/row count unchanged on re-run).