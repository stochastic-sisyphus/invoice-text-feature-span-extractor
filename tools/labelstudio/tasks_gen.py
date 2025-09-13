#!/usr/bin/env python3
import sys, json, hashlib, pathlib, shutil, unicodedata
import pdfplumber

# Keep this normalization aligned with my pipeline
def normalize_text(s: str) -> str:
    if not s: return ""
    s = unicodedata.normalize("NFC", s)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    return s.strip()

def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1<<20), b""):
            h.update(chunk)
    return h.hexdigest()

def extract_text(path: pathlib.Path) -> str:
    parts = []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            txt = page.extract_text() or ""
            parts.append(f"[[PAGE {i}]]\n{txt}")
    return normalize_text("\n\n".join(parts))

def main(pdf_folder: str, out_path="tasks.json", labeled_dir="labeled_pdfs"):
    folder = pathlib.Path(pdf_folder)
    out = pathlib.Path(out_path)
    labeled = pathlib.Path(labeled_dir)
    labeled.mkdir(parents=True, exist_ok=True)

    tasks = []
    pdfs = sorted(folder.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in: {folder}")
        sys.exit(2)

    for pdf in pdfs:
        sha = sha256_file(pdf)
        text = extract_text(pdf)
        if not text:
            print(f"[skip] no native text: {pdf.name}")
            continue
        shutil.copy2(pdf, labeled / f"{sha}.pdf")
        tasks.append({
            "data": {
                "text": text,
                "sha256": sha,
                "doc_id": f"fs:{sha[:16]}",
                "filename": pdf.name,
                "source_path": str(pdf.resolve())
            }
        })

    out.write_text(json.dumps(tasks, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(tasks)} tasks -> {out.resolve()}")
    print(f"Copied PDFs -> {labeled.resolve()}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tasks_gen.py /path/to/pdfs [tasks.json] [labeled_dir]")
        sys.exit(1)
    main(
        sys.argv[1],
        sys.argv[2] if len(sys.argv) > 2 else "tasks.json",
        sys.argv[3] if len(sys.argv) > 3 else "labeled_pdfs"
    )