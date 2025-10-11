# Label Studio Integration

Complete workflow for invoice field annotation and model training.

## Setup

1. **Generate tasks with normalization guards**:
```bash
cd tools/labelstudio
python tasks_gen.py --seed-folder ../../seed_pdfs --output ./output
```

2. **Import tasks to Label Studio**:
- Upload `output/tasks.json` to Label Studio project
- Use `labelsettings.xml` as the labeling interface
- Annotate invoice fields (InvoiceNumber, TotalAmount, etc.)

3. **Export and align**:
```bash
# Export from Label Studio as JSON
invoicex labels-import --in path/to/export.json

# Align with pipeline candidates using IoU matching
invoicex labels-align --all --iou 0.3

# Train XGBoost models
invoicex train
```

## Normalization Guard

The system enforces text normalization consistency:
- Each task includes `normalize_version` and `text_checksum`
- Import validates against pipeline's current normalizer
- Prevents drift when annotation/pipeline versions diverge

## Field Mapping

Label Studio → Contract fields:
- `InvoiceNumber` → `invoice_number`
- `InvoiceDate` → `invoice_date` 
- `TotalAmount` → `total_amount`
- `VendorName` → `vendor_name`
- `CustomerAccount` → `customer_account`

## Line Items

Line item fields are grouped by spatial proximity:
- `LineItemDescription`, `LineItemQuantity`, `LineItemUnitPrice`, `LineItemTotal`
- Ambiguous groupings → `line_items: []` + review queue
